from __future__ import annotations

import base64
import csv
import io
import json
import re
import struct
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

TARGET_DATE_INT = 20260724
TARGET_DATE = "2026-07-24"
ROOT = Path(__file__).resolve().parent
WORK = ROOT / "work"
RESULT_DIR = ROOT / "results"
TDX_URL = "https://data.tdx.com.cn/vipdoc/hsjday.zip"
RECORD = struct.Struct("<IIIIIfII")


def download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 Venus8Scan/1.0"})
    with urllib.request.urlopen(req, timeout=180) as resp, path.open("wb") as out:
        while True:
            chunk = resp.read(4 * 1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    print(f"downloaded {path.stat().st_size / 1024 / 1024:.1f} MB", flush=True)


def prepare_runtime() -> None:
    raw = base64.b64decode((ROOT / "venus_runtime.b64").read_text(encoding="ascii"))
    vendor = ROOT / "vendor"
    vendor.mkdir(exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tf:
        tf.extractall(vendor)
    sys.path.insert(0, str(vendor))


def valid_code(market: str, code: str) -> bool:
    if market == "sh":
        return code.startswith(("600", "601", "603", "605", "688", "689"))
    if market == "sz":
        return code.startswith(("000", "001", "002", "003", "300", "301"))
    if market == "bj":
        return code.isdigit()
    return False


def member_symbol(name: str) -> tuple[str, str] | None:
    base = Path(name.replace("\\", "/")).name.lower()
    m = re.fullmatch(r"(sh|sz|bj)(\d{6})\.day", base)
    if not m:
        return None
    market, code = m.groups()
    return (market, code) if valid_code(market, code) else None


def parse_day(raw: bytes) -> pd.DataFrame | None:
    n = len(raw) // RECORD.size
    if n < 80:
        return None
    rows = []
    for values in RECORD.iter_unpack(raw[: n * RECORD.size]):
        date, op, hi, lo, cl, amount, volume, _reserved = values
        if date > TARGET_DATE_INT:
            break
        if date < 19900101 or cl <= 0 or hi <= 0 or lo <= 0:
            continue
        rows.append((date, op / 100.0, hi / 100.0, lo / 100.0, cl / 100.0, float(volume), float(amount)))
    if len(rows) < 80 or rows[-1][0] != TARGET_DATE_INT:
        return None
    rows = rows[-300:]
    df = pd.DataFrame(rows, columns=["date_int", "open", "high", "low", "close", "volume", "turnover"])
    df["date"] = pd.to_datetime(df.pop("date_int").astype(str), format="%Y%m%d").dt.strftime("%Y-%m-%d")
    med = float(pd.to_numeric(df["turnover"], errors="coerce").tail(20).median())
    if not np.isfinite(med) or med <= 0 or med > 1e14:
        df["turnover"] = np.nan
    return df[["date", "open", "high", "low", "close", "volume", "turnover"]]


def get_name_map() -> dict[str, str]:
    urls = [
        "https://82.push2.eastmoney.com/api/qt/clist/get?pn=1&pz=10000&po=1&np=1&fltt=2&invt=2&fid=f3&fs=m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23,m:0+t:81+s:2048&fields=f12,f14",
        "https://push2.eastmoney.com/api/qt/clist/get?pn=1&pz=10000&po=1&np=1&fltt=2&invt=2&fid=f3&fs=m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23,m:0+t:81+s:2048&fields=f12,f14",
    ]
    for url in urls:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                payload = json.load(resp)
            diff = (((payload or {}).get("data") or {}).get("diff") or [])
            result = {str(x.get("f12") or "").zfill(6): str(x.get("f14") or "") for x in diff if x.get("f12")}
            if len(result) > 3000:
                return result
        except Exception as exc:
            print(f"name endpoint failed: {exc}", flush=True)
    return {}


def clean_name(name: str) -> bool:
    return not bool(re.search(r"(?:\*?ST|退$|^N|^C)", name or "", re.I))


def compact(item: dict[str, Any], rank: int) -> dict[str, Any]:
    dims = item.get("dimensions") or {}
    raw = item.get("raw_metrics") or {}
    score = lambda key: (dims.get(key) or {}).get("score")
    return {
        "rank": rank,
        "code": item.get("symbol"),
        "name": item.get("name"),
        "score": item.get("total_score"),
        "trend": score("trend_structure"),
        "pullback": score("pullback_quality"),
        "momentum": score("momentum_repair"),
        "volume": score("volume_activity"),
        "price_confirmation": score("price_confirmation"),
        "boll": score("boll_prediction"),
        "close": raw.get("close"),
        "daily_change_pct": raw.get("daily_change_pct"),
        "drawdown_60d_high_pct": raw.get("drawdown_from_60d_high_pct"),
        "ret_20d_pct": raw.get("ret_20d_pct"),
        "ret_60d_pct": raw.get("ret_60d_pct"),
        "action_stance": item.get("action_stance"),
        "supports": (item.get("key_evidence") or {}).get("supports") or [],
        "risk_points": item.get("risk_points") or [],
    }


def main() -> None:
    prepare_runtime()
    from src.analysis.golden_pit import score_golden_pit

    WORK.mkdir(exist_ok=True)
    archive = WORK / "hsjday.zip"
    download(TDX_URL, archive)
    names = get_name_map()
    candidates: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    eligible = 0
    members_seen = 0

    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            parsed = member_symbol(info.filename)
            if parsed is None:
                continue
            members_seen += 1
            market, code = parsed
            try:
                frame = parse_day(zf.read(info))
                if frame is None:
                    continue
                eligible += 1
                payload = score_golden_pit(symbol=code, name=names.get(code, ""), bars=frame, period="daily")
                payload["market"] = market.upper()
                payload["bar_count_input"] = len(frame)
                if payload.get("grade") == "A" or payload.get("is_near_a_grade"):
                    candidates.append(payload)
            except Exception as exc:
                failures.append({"symbol": f"{market}{code}", "error": f"{type(exc).__name__}: {exc}"})
            if members_seen % 500 == 0:
                print(f"progress members={members_seen} eligible={eligible} candidates={len(candidates)}", flush=True)

    candidates.sort(key=lambda x: (x.get("grade") == "A", int(x.get("total_score") or 0)), reverse=True)
    grade_a = [x for x in candidates if x.get("grade") == "A"]
    clean_a = [x for x in grade_a if clean_name(str(x.get("name") or ""))]
    near_a = [x for x in candidates if x.get("grade") != "A"]
    rows = [compact(item, i) for i, item in enumerate(clean_a, 1)]

    output = {
        "schema": "venus8_full_market_technical_a_scan.v2",
        "target_date": TARGET_DATE,
        "method": "exact Venus8 src.analysis.golden_pit.score_golden_pit",
        "technical_a_rule": "total_score >= 85 and no Venus8 hard exclusion",
        "data_source": {
            "provider": "TDX official Shanghai/Shenzhen/Beijing complete daily package",
            "url": TDX_URL,
            "price_basis": "raw unadjusted TDX daily OHLCV; exact Venus8 indicator and grading logic",
            "liquidity_basis": "TDX turnover when sane, otherwise Venus8 volume fallback",
        },
        "universe": {
            "a_share_day_members": members_seen,
            "scored_with_80_bars_and_2026_07_24_close": eligible,
            "score_failures": len(failures),
        },
        "counts": {
            "technical_a_all": len(grade_a),
            "technical_a_clean": len(clean_a),
            "near_a": len(near_a),
        },
        "technical_a": grade_a,
        "technical_a_clean": clean_a,
        "near_a": near_a[:100],
        "failures_sample": failures[:50],
    }
    RESULT_DIR.mkdir(exist_ok=True)
    (RESULT_DIR / "venus8_technical_a_20260724.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    csv_path = RESULT_DIR / "venus8_technical_a_20260724.csv"
    fields = list(rows[0].keys()) if rows else ["rank", "code", "name", "score"]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        f"# Venus8 全市场技术A扫描（{TARGET_DATE}）",
        "",
        f"- 实际进入原生评分：{eligible}只",
        f"- 技术A：{len(grade_a)}只",
        f"- 清理ST/退市/新股前缀后：{len(clean_a)}只",
        f"- 接近A：{len(near_a)}只",
        "",
        "|排名|代码|名称|总分|趋势|回调|动能|量能|价格确认|BOLL|收盘|当日涨跌%|",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(f"|{row['rank']}|{row['code']}|{row['name']}|{row['score']}|{row['trend']}|{row['pullback']}|{row['momentum']}|{row['volume']}|{row['price_confirmation']}|{row['boll']}|{row['close']}|{row['daily_change_pct']}|")
    (RESULT_DIR / "venus8_technical_a_20260724.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("VENUS_RESULT_BEGIN")
    print(json.dumps({"counts": output["counts"], "universe": output["universe"], "top": rows[:30]}, ensure_ascii=False))
    print("VENUS_RESULT_END")


if __name__ == "__main__":
    main()
