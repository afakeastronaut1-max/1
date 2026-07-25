from __future__ import annotations

import base64
import csv
import io
import json
import math
import random
import re
import sys
import tarfile
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

TARGET_DATE = "2026-07-24"
ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "results"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/140 Safari/537.36",
    "Accept": "application/json,text/plain,*/*",
    "Referer": "https://quote.eastmoney.com/",
}


def prepare_runtime() -> None:
    raw = base64.b64decode((ROOT / "venus_runtime.b64").read_text(encoding="ascii"))
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tf:
        tf.extractall(ROOT / "vendor")
    sys.path.insert(0, str(ROOT / "vendor"))


def get_json(url: str, timeout: int = 20, retries: int = 3) -> dict[str, Any]:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.load(resp)
        except Exception as exc:
            last = exc
            time.sleep(0.35 * (attempt + 1) + random.random() * 0.2)
    raise RuntimeError(f"request failed: {type(last).__name__}: {last}")


def universe() -> list[dict[str, str]]:
    fs = "m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23,m:0+t:81+s:2048"
    params = {
        "pn": "1", "pz": "10000", "po": "1", "np": "1", "fltt": "2", "invt": "2",
        "fid": "f3", "fs": fs, "fields": "f12,f13,f14",
    }
    hosts = ["82.push2.eastmoney.com", "push2.eastmoney.com", "28.push2.eastmoney.com"]
    for host in hosts:
        try:
            payload = get_json("https://" + host + "/api/qt/clist/get?" + urllib.parse.urlencode(params), timeout=30)
            diff = (((payload or {}).get("data") or {}).get("diff") or [])
            rows = []
            for x in diff:
                code = str(x.get("f12") or "").zfill(6)
                name = str(x.get("f14") or "").strip()
                market = str(x.get("f13") if x.get("f13") is not None else "")
                if re.fullmatch(r"\d{6}", code) and market in {"0", "1"}:
                    rows.append({"code": code, "name": name, "market": market})
            if len(rows) >= 4500:
                return rows
        except Exception as exc:
            print(f"universe host {host} failed: {exc}", flush=True)
    raise RuntimeError("unable to obtain full A-share universe")


def eastmoney_bars(row: dict[str, str]) -> pd.DataFrame:
    params = {
        "secid": f"{row['market']}.{row['code']}", "klt": "101", "fqt": "1",
        "beg": "20240101", "end": TARGET_DATE.replace("-", ""), "lmt": "500",
        "fields1": "f1,f2,f3,f4,f5,f6", "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "ut": "fa5fd1943c7b386f172d6893dbfba10b",
    }
    hosts = ["push2his.eastmoney.com", "72.push2his.eastmoney.com", "7.push2his.eastmoney.com"]
    err: Exception | None = None
    for host in hosts:
        try:
            payload = get_json("https://" + host + "/api/qt/stock/kline/get?" + urllib.parse.urlencode(params))
            klines = (((payload or {}).get("data") or {}).get("klines") or [])
            records = []
            for line in klines:
                p = str(line).split(",")
                if len(p) < 7:
                    continue
                records.append({
                    "date": p[0], "open": float(p[1]), "close": float(p[2]),
                    "high": float(p[3]), "low": float(p[4]), "volume": float(p[5]),
                    "turnover": float(p[6]) if p[6] not in {"", "-"} else np.nan,
                })
            df = pd.DataFrame(records)
            if len(df) >= 80 and str(df.iloc[-1]["date"]) == TARGET_DATE:
                return df.tail(300).reset_index(drop=True)
        except Exception as exc:
            err = exc
    raise RuntimeError(f"eastmoney unavailable: {err}")


def tencent_bars(row: dict[str, str]) -> pd.DataFrame:
    prefix = "sh" if row["market"] == "1" else "sz"
    symbol = prefix + row["code"]
    param = f"{symbol},day,2024-01-01,{TARGET_DATE},500,qfq"
    url = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?" + urllib.parse.urlencode({"param": param})
    payload = get_json(url, timeout=25)
    node = (((payload or {}).get("data") or {}).get(symbol) or {})
    lines = node.get("qfqday") or node.get("day") or []
    records = []
    for p in lines:
        if len(p) < 6:
            continue
        records.append({
            "date": str(p[0]), "open": float(p[1]), "close": float(p[2]),
            "high": float(p[3]), "low": float(p[4]), "volume": float(p[5]), "turnover": np.nan,
        })
    df = pd.DataFrame(records)
    if len(df) < 80 or str(df.iloc[-1]["date"]) != TARGET_DATE:
        raise RuntimeError("tencent history incomplete")
    return df.tail(300).reset_index(drop=True)


def fetch_one(row: dict[str, str]) -> tuple[pd.DataFrame, str]:
    try:
        return eastmoney_bars(row), "eastmoney_qfq"
    except Exception as first:
        try:
            return tencent_bars(row), "tencent_qfq"
        except Exception as second:
            raise RuntimeError(f"eastmoney={first}; tencent={second}")


def safe_number(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: safe_number(v) for k, v in value.items()}
    if isinstance(value, list):
        return [safe_number(v) for v in value]
    return value


def main() -> None:
    prepare_runtime()
    from src.analysis.golden_pit import score_golden_pit

    stocks = universe()
    print(f"universe={len(stocks)} target={TARGET_DATE}", flush=True)
    results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    source_counts: dict[str, int] = {}
    completed = 0

    def worker(row: dict[str, str]) -> dict[str, Any]:
        bars, source = fetch_one(row)
        score = score_golden_pit(symbol=row["code"], name=row["name"], bars=bars, period="daily")
        score["market"] = row["market"]
        score["data_source"] = source
        score["data_bar_count"] = len(bars)
        score["as_of"] = str(bars.iloc[-1]["date"])
        return safe_number(score)

    with ThreadPoolExecutor(max_workers=24) as pool:
        futures = {pool.submit(worker, row): row for row in stocks}
        for future in as_completed(futures):
            row = futures[future]
            completed += 1
            try:
                item = future.result()
                source = str(item.get("data_source") or "unknown")
                source_counts[source] = source_counts.get(source, 0) + 1
                if item.get("grade") == "A" or item.get("is_near_a_grade"):
                    results.append(item)
            except Exception as exc:
                failures.append({"code": row["code"], "name": row["name"], "error": str(exc)[:800]})
            if completed % 250 == 0:
                print(f"progress={completed}/{len(stocks)} candidates={len(results)} failures={len(failures)} sources={source_counts}", flush=True)

    results.sort(key=lambda x: (x.get("grade") == "A", int(x.get("total_score") or 0)), reverse=True)
    grade_a = [x for x in results if x.get("grade") == "A"]
    clean_a = [x for x in grade_a if not re.search(r"(?:\*?ST|退)", str(x.get("name") or ""), re.I)]
    near_a = [x for x in results if x.get("grade") != "A"]

    output = {
        "schema": "venus8_full_market_technical_a_multisource.v1",
        "target_date": TARGET_DATE,
        "method": "exact Venus8 src.analysis.golden_pit.score_golden_pit",
        "technical_a_gate": "total_score >= 85 and no Venus8 hard exclusion",
        "data_policy": "Eastmoney forward-adjusted daily bars; automatic Tencent forward-adjusted fallback",
        "universe_count": len(stocks),
        "success_count": sum(source_counts.values()),
        "failure_count": len(failures),
        "source_counts": source_counts,
        "technical_a_count": len(grade_a),
        "technical_a_clean_count": len(clean_a),
        "near_a_count": len(near_a),
        "technical_a": grade_a,
        "technical_a_clean": clean_a,
        "near_a": near_a[:200],
        "failures_sample": failures[:100],
    }
    RESULT_DIR.mkdir(exist_ok=True)
    (RESULT_DIR / "venus8_technical_a_20260724.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    fields = ["rank", "code", "name", "score", "trend", "pullback", "momentum", "volume", "price_confirmation", "boll", "close", "daily_change_pct", "source", "stance"]
    rows = []
    for rank, item in enumerate(clean_a, 1):
        d = item.get("dimensions") or {}
        raw = item.get("raw_metrics") or {}
        rows.append({
            "rank": rank, "code": item.get("symbol"), "name": item.get("name"), "score": item.get("total_score"),
            "trend": (d.get("trend_structure") or {}).get("score"),
            "pullback": (d.get("pullback_quality") or {}).get("score"),
            "momentum": (d.get("momentum_repair") or {}).get("score"),
            "volume": (d.get("volume_activity") or {}).get("score"),
            "price_confirmation": (d.get("price_confirmation") or {}).get("score"),
            "boll": (d.get("boll_prediction") or {}).get("score"),
            "close": raw.get("close"), "daily_change_pct": raw.get("daily_change_pct"),
            "source": item.get("data_source"), "stance": item.get("action_stance"),
        })
    with (RESULT_DIR / "venus8_technical_a_20260724.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)

    md = [
        f"# Venus8 全市场技术A扫描（{TARGET_DATE}）", "",
        f"- 全市场代码：{len(stocks)}", f"- 成功评分：{sum(source_counts.values())}", f"- 失败：{len(failures)}",
        f"- 技术A：{len(grade_a)}", f"- 剔除ST/退市后：{len(clean_a)}", f"- 数据源：{source_counts}", "",
        "|排名|代码|名称|总分|趋势|回调|动能|量能|确认|BOLL|收盘|涨跌%|来源|",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        md.append(f"|{r['rank']}|{r['code']}|{r['name']}|{r['score']}|{r['trend']}|{r['pullback']}|{r['momentum']}|{r['volume']}|{r['price_confirmation']}|{r['boll']}|{r['close']}|{r['daily_change_pct']}|{r['source']}|")
    (RESULT_DIR / "venus8_technical_a_20260724.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"success": sum(source_counts.values()), "failures": len(failures), "technical_a": len(grade_a), "clean_a": len(clean_a), "sources": source_counts}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
