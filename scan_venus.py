from __future__ import annotations

import base64
import csv
import io
import json
import re
import sys
import tarfile
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

TARGET_DATE = "2026-07-24"
TARGET_END = "20260724"
ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "results"
UNIVERSE_URLS = [
    "https://82.push2.eastmoney.com/api/qt/clist/get",
    "https://push2.eastmoney.com/api/qt/clist/get",
]
HISTORY_HOSTS = [
    "https://push2his.eastmoney.com/api/qt/stock/kline/get",
    "https://71.push2his.eastmoney.com/api/qt/stock/kline/get",
    "https://53.push2his.eastmoney.com/api/qt/stock/kline/get",
]
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Venus8Scan/1.0"


def get_json(base: str, params: dict[str, Any], timeout: int = 25) -> dict[str, Any]:
    url = base + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Referer": "https://quote.eastmoney.com/"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    return json.loads(raw.decode("utf-8", errors="replace"))


def prepare_runtime() -> None:
    raw = base64.b64decode((ROOT / "venus_runtime.b64").read_text(encoding="ascii"))
    vendor = ROOT / "vendor"
    vendor.mkdir(exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tf:
        tf.extractall(vendor)
    sys.path.insert(0, str(vendor))


def load_universe() -> list[dict[str, str]]:
    params = {
        "pn": 1, "pz": 10000, "po": 1, "np": 1, "fltt": 2, "invt": 2,
        "fid": "f3",
        "fs": "m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23,m:0+t:81+s:2048",
        "fields": "f12,f13,f14",
    }
    last = None
    for base in UNIVERSE_URLS:
        try:
            payload = get_json(base, params, timeout=40)
            diff = (((payload or {}).get("data") or {}).get("diff") or [])
            rows = []
            for item in diff:
                code = str(item.get("f12") or "").zfill(6)
                market = str(item.get("f13") if item.get("f13") is not None else "")
                name = str(item.get("f14") or "")
                if code.isdigit() and market in {"0", "1"}:
                    rows.append({"code": code, "market": market, "name": name})
            if len(rows) > 4000:
                print(f"universe={len(rows)} from {base}", flush=True)
                return rows
        except Exception as exc:
            last = exc
            print(f"universe source failed {base}: {exc}", flush=True)
    raise RuntimeError(f"unable to load full A-share universe: {last}")


def fetch_bars(stock: dict[str, str]) -> tuple[dict[str, str], pd.DataFrame | None, str | None]:
    params = {
        "secid": f"{stock['market']}.{stock['code']}",
        "ut": "fa5fd1943c7b386f172d6893dbfba10b",
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "klt": 101,
        "fqt": 1,
        "beg": 0,
        "end": TARGET_END,
        "lmt": 320,
    }
    err = None
    for attempt in range(4):
        base = HISTORY_HOSTS[(hash(stock["code"]) + attempt) % len(HISTORY_HOSTS)]
        try:
            payload = get_json(base, params, timeout=25)
            data = (payload or {}).get("data") or {}
            klines = data.get("klines") or []
            rows = []
            for line in klines:
                p = str(line).split(",")
                if len(p) < 7:
                    continue
                try:
                    rows.append({
                        "date": p[0], "open": float(p[1]), "close": float(p[2]),
                        "high": float(p[3]), "low": float(p[4]), "volume": float(p[5]),
                        "turnover": float(p[6]),
                    })
                except ValueError:
                    continue
            if len(rows) < 80 or rows[-1]["date"] != TARGET_DATE:
                return stock, None, "insufficient_or_suspended"
            return stock, pd.DataFrame(rows[-300:]), None
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            time.sleep(0.35 * (attempt + 1))
    return stock, None, err


def clean_name(name: str) -> bool:
    return not bool(re.search(r"(?:\*?ST|退$|^N|^C)", name or "", re.I))


def compact(item: dict[str, Any], rank: int) -> dict[str, Any]:
    dims = item.get("dimensions") or {}
    raw = item.get("raw_metrics") or {}
    def s(key: str): return (dims.get(key) or {}).get("score")
    return {
        "rank": rank, "code": item.get("symbol"), "name": item.get("name"),
        "score": item.get("total_score"), "trend": s("trend_structure"),
        "pullback": s("pullback_quality"), "momentum": s("momentum_repair"),
        "volume": s("volume_activity"), "price_confirmation": s("price_confirmation"),
        "boll": s("boll_prediction"), "close": raw.get("close"),
        "daily_change_pct": raw.get("daily_change_pct"),
        "drawdown_60d_high_pct": raw.get("drawdown_from_60d_high_pct"),
        "ret_20d_pct": raw.get("ret_20d_pct"), "ret_60d_pct": raw.get("ret_60d_pct"),
        "action_stance": item.get("action_stance"),
        "supports": (item.get("key_evidence") or {}).get("supports") or [],
        "risk_points": item.get("risk_points") or [],
    }


def main() -> None:
    prepare_runtime()
    from src.analysis.golden_pit import score_golden_pit

    universe = load_universe()
    candidates: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    scored = 0
    completed = 0
    with ThreadPoolExecutor(max_workers=48) as pool:
        futures = [pool.submit(fetch_bars, stock) for stock in universe]
        for fut in as_completed(futures):
            stock, frame, error = fut.result()
            completed += 1
            if frame is not None:
                try:
                    scored += 1
                    payload = score_golden_pit(stock["code"], frame, name=stock["name"], period="daily")
                    payload["market_id"] = stock["market"]
                    if payload.get("grade") == "A" or payload.get("is_near_a_grade"):
                        candidates.append(payload)
                except Exception as exc:
                    failures.append({"code": stock["code"], "error": f"score:{type(exc).__name__}: {exc}"})
            elif error and error != "insufficient_or_suspended":
                failures.append({"code": stock["code"], "error": error})
            if completed % 250 == 0:
                print(f"progress={completed}/{len(universe)} scored={scored} candidates={len(candidates)} failures={len(failures)}", flush=True)

    candidates.sort(key=lambda x: (x.get("grade") == "A", int(x.get("total_score") or 0)), reverse=True)
    grade_a = [x for x in candidates if x.get("grade") == "A"]
    clean_a = [x for x in grade_a if clean_name(str(x.get("name") or ""))]
    near_a = [x for x in candidates if x.get("grade") != "A"]
    rows = [compact(x, i) for i, x in enumerate(clean_a, 1)]
    output = {
        "schema": "venus8_full_market_technical_a_scan.v3",
        "target_date": TARGET_DATE,
        "method": "exact Venus8 src.analysis.golden_pit.score_golden_pit",
        "technical_a_rule": "total_score >= 85 and no Venus8 hard exclusion",
        "data_source": {
            "provider": "Eastmoney multi-host daily K-line API",
            "adjustment": "fqt=1 forward-adjusted (qfq)",
            "bar_limit": 320,
        },
        "universe": {"listed_a_shares": len(universe), "scored_with_80_bars_and_target_close": scored, "request_or_score_failures": len(failures)},
        "counts": {"technical_a_all": len(grade_a), "technical_a_clean": len(clean_a), "near_a": len(near_a)},
        "technical_a": grade_a, "technical_a_clean": clean_a, "near_a": near_a[:120], "failures_sample": failures[:100],
    }
    RESULT_DIR.mkdir(exist_ok=True)
    (RESULT_DIR / "venus8_technical_a_20260724.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fields = list(rows[0].keys()) if rows else ["rank", "code", "name", "score"]
    with (RESULT_DIR / "venus8_technical_a_20260724.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    lines = [f"# Venus8 全市场技术A扫描（{TARGET_DATE}）", "", f"- 全市场：{len(universe)}只", f"- 实际评分：{scored}只", f"- 技术A：{len(grade_a)}只", f"- 清理ST/退市/新股前缀后：{len(clean_a)}只", f"- 接近A：{len(near_a)}只", "", "|排名|代码|名称|总分|趋势|回调|动能|量能|价格确认|BOLL|收盘|当日涨跌%|", "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"|{r['rank']}|{r['code']}|{r['name']}|{r['score']}|{r['trend']}|{r['pullback']}|{r['momentum']}|{r['volume']}|{r['price_confirmation']}|{r['boll']}|{r['close']}|{r['daily_change_pct']}|")
    (RESULT_DIR / "venus8_technical_a_20260724.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("VENUS_RESULT_BEGIN")
    print(json.dumps({"counts": output["counts"], "universe": output["universe"], "top": rows[:40]}, ensure_ascii=False))
    print("VENUS_RESULT_END")


if __name__ == "__main__":
    main()
