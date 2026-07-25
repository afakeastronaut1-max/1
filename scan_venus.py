from __future__ import annotations

import base64
import csv
import io
import json
import os
import re
import struct
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

TARGET_DATE = "2026-07-24"
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "work"
QLIB_DIR = DATA_DIR / "qlib_data"
RESULT_DIR = ROOT / "results"


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 Venus8Scan/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp, path.open("wb") as out:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    if path.stat().st_size < 1024:
        raise RuntimeError(f"download too small: {url}")


def latest_release() -> dict[str, Any]:
    url = "https://api.github.com/repos/chenditc/investment_data/releases/latest"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 Venus8Scan/1.0", "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)


def prepare_runtime() -> None:
    archive_b64 = ROOT / "venus_runtime.b64"
    raw = base64.b64decode(archive_b64.read_text(encoding="ascii"))
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tf:
        tf.extractall(ROOT / "vendor")
    sys.path.insert(0, str(ROOT / "vendor"))


def prepare_qlib() -> tuple[str, str]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    release = latest_release()
    tag = str(release.get("tag_name") or "")
    if not tag:
        raise RuntimeError("latest release has no tag")
    assets = {a.get("name"): a.get("browser_download_url") for a in release.get("assets", [])}
    asset_url = assets.get("qlib_bin.tar.gz") or "https://github.com/chenditc/investment_data/releases/latest/download/qlib_bin.tar.gz"
    archive = DATA_DIR / "qlib_bin.tar.gz"
    print(f"Latest investment_data release: {tag}", flush=True)
    download(str(asset_url), archive)
    if QLIB_DIR.exists():
        import shutil
        shutil.rmtree(QLIB_DIR)
    QLIB_DIR.mkdir(parents=True)
    run(["tar", "-xzf", str(archive), "-C", str(QLIB_DIR), "--strip-components=1"])
    return tag, str(asset_url)


def locate(*parts: str) -> Path:
    direct = QLIB_DIR.joinpath(*parts)
    if direct.exists():
        return direct
    name = parts[-1]
    matches = list(QLIB_DIR.rglob(name))
    if not matches:
        raise FileNotFoundError("/".join(parts))
    if len(parts) >= 2:
        for p in matches:
            if p.parent.name == parts[-2]:
                return p
    return matches[0]


def read_calendar() -> list[str]:
    path = locate("calendars", "day.txt")
    dates = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()[:10]
        if value:
            dates.append(value)
    return dates


def read_instruments() -> list[tuple[str, str, str]]:
    path = locate("instruments", "all.txt")
    result = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = re.split(r"\s+", line.strip())
        if len(parts) >= 3:
            result.append((parts[0].upper(), parts[1][:10], parts[2][:10]))
    return result


def valid_a_share(symbol: str) -> bool:
    s = symbol.upper()
    if s.startswith("SH"):
        code = s[2:]
        return code.startswith(("600", "601", "603", "605", "688", "689"))
    if s.startswith("SZ"):
        code = s[2:]
        return code.startswith(("000", "001", "002", "003", "300", "301"))
    if s.startswith("BJ"):
        return s[2:].isdigit()
    return False


def feature_path(symbol: str, field: str) -> Path | None:
    sym = symbol.lower()
    candidates = [
        QLIB_DIR / "features" / sym / f"{field}.day.bin",
        QLIB_DIR / "features" / sym / f"{field}.bin",
    ]
    for p in candidates:
        if p.exists():
            return p
    matches = list(QLIB_DIR.rglob(f"{field}.day.bin"))
    for p in matches:
        if p.parent.name.lower() == sym:
            return p
    return None


def read_feature(symbol: str, field: str, calendar_len: int) -> np.ndarray | None:
    path = feature_path(symbol, field)
    if path is None:
        return None
    arr = np.fromfile(path, dtype="<f4")
    if len(arr) < 2:
        return None
    start_idx = int(round(float(arr[0])))
    values = arr[1:].astype(float)
    out = np.full(calendar_len, np.nan, dtype=float)
    if start_idx < 0 or start_idx >= calendar_len:
        return None
    end = min(calendar_len, start_idx + len(values))
    out[start_idx:end] = values[: end - start_idx]
    return out


def get_name_map() -> dict[str, str]:
    endpoints = [
        "https://82.push2.eastmoney.com/api/qt/clist/get?pn=1&pz=10000&po=1&np=1&fltt=2&invt=2&fid=f3&fs=m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23,m:0+t:81+s:2048&fields=f12,f14",
        "https://push2.eastmoney.com/api/qt/clist/get?pn=1&pz=10000&po=1&np=1&fltt=2&invt=2&fid=f3&fs=m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23,m:0+t:81+s:2048&fields=f12,f14",
    ]
    for url in endpoints:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                payload = json.load(resp)
            diff = (((payload or {}).get("data") or {}).get("diff") or [])
            result = {str(x.get("f12") or "").zfill(6): str(x.get("f14") or "") for x in diff if x.get("f12")}
            if len(result) > 3000:
                return result
        except Exception as exc:
            print(f"name map endpoint failed: {exc}", flush=True)
    return {}


def build_frame(symbol: str, calendar: list[str], target_idx: int) -> pd.DataFrame | None:
    fields = ["open", "high", "low", "close", "volume", "amount"]
    arrays: dict[str, np.ndarray] = {}
    for field in fields:
        arr = read_feature(symbol, field, len(calendar))
        if arr is not None:
            arrays[field] = arr
    required = {"open", "high", "low", "close", "volume"}
    if not required.issubset(arrays):
        return None
    start_idx = max(0, target_idx - 259)
    data = {"date": calendar[start_idx : target_idx + 1]}
    for field in required:
        data[field] = arrays[field][start_idx : target_idx + 1]
    if "amount" in arrays:
        data["turnover"] = arrays["amount"][start_idx : target_idx + 1]
    else:
        data["turnover"] = np.full(target_idx + 1 - start_idx, np.nan)
    df = pd.DataFrame(data)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["open", "high", "low", "close"])
    if df.empty or str(df.iloc[-1]["date"]) != TARGET_DATE:
        return None
    df.loc[df["volume"] <= 0, "volume"] = np.nan
    return df.reset_index(drop=True)


def main() -> None:
    prepare_runtime()
    from src.analysis.golden_pit import score_golden_pit

    tag, asset_url = prepare_qlib()
    calendar = read_calendar()
    if TARGET_DATE not in calendar:
        raise RuntimeError(f"target date {TARGET_DATE} missing; calendar last={calendar[-1] if calendar else None}; release={tag}")
    target_idx = calendar.index(TARGET_DATE)
    instruments = read_instruments()
    names = get_name_map()

    results = []
    failed = []
    scanned = 0
    for idx, (symbol, start, end) in enumerate(instruments, 1):
        if not valid_a_share(symbol) or start > TARGET_DATE or end < TARGET_DATE:
            continue
        code = symbol[2:]
        try:
            frame = build_frame(symbol, calendar, target_idx)
            if frame is None or len(frame) < 80:
                continue
            scanned += 1
            name = names.get(code) or ""
            payload = score_golden_pit(symbol=code, name=name, bars=frame, period="daily")
            payload["exchange_symbol"] = symbol
            payload["data_bar_count"] = len(frame)
            payload["source_release"] = tag
            if payload.get("grade") == "A" or payload.get("is_near_a_grade"):
                results.append(payload)
        except Exception as exc:
            failed.append({"symbol": symbol, "error": f"{type(exc).__name__}: {exc}"})
        if idx % 500 == 0:
            print(f"progress instruments={idx}/{len(instruments)} scanned={scanned} candidates={len(results)}", flush=True)

    def sort_key(item: dict[str, Any]):
        return (item.get("grade") == "A", int(item.get("total_score") or 0), -len(item.get("exclusions") or []))

    results.sort(key=sort_key, reverse=True)
    grade_a = [x for x in results if x.get("grade") == "A"]
    near_a = [x for x in results if x.get("grade") != "A"]
    clean_a = [x for x in grade_a if not re.search(r"(?:\*?ST|退|N|C)", str(x.get("name") or ""), re.I)]

    output = {
        "schema": "venus8_full_market_technical_a_scan.v1",
        "target_date": TARGET_DATE,
        "method": "exact Venus8 src.analysis.golden_pit.score_golden_pit",
        "technical_a_threshold": "total_score >= 85 and no Venus8 hard exclusion",
        "data_source": {
            "provider": "chenditc/investment_data qlib daily release",
            "release_tag": tag,
            "asset_url": asset_url,
            "price_basis": "qlib split-adjusted/normalized continuous OHLCV",
        },
        "universe": {
            "instrument_rows": len(instruments),
            "a_share_scanned_with_80_bars_and_target_close": scanned,
            "score_failures": len(failed),
        },
        "counts": {
            "technical_a_all": len(grade_a),
            "technical_a_ex_st_delisting_new_prefix": len(clean_a),
            "near_a_b_78_plus_no_exclusion": len(near_a),
        },
        "technical_a": grade_a,
        "technical_a_clean": clean_a,
        "near_a": near_a[:100],
        "failures_sample": failed[:50],
    }
    RESULT_DIR.mkdir(exist_ok=True)
    json_path = RESULT_DIR / "venus8_technical_a_20260724.json"
    json_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    rows = []
    for rank, item in enumerate(clean_a, 1):
        dims = item.get("dimensions") or {}
        rows.append({
            "rank": rank,
            "code": item.get("symbol"),
            "name": item.get("name"),
            "score": item.get("total_score"),
            "trend": (dims.get("trend_structure") or {}).get("score"),
            "pullback": (dims.get("pullback_quality") or {}).get("score"),
            "momentum": (dims.get("momentum_repair") or {}).get("score"),
            "volume": (dims.get("volume_activity") or {}).get("score"),
            "price_confirmation": (dims.get("price_confirmation") or {}).get("score"),
            "boll": (dims.get("boll_prediction") or {}).get("score"),
            "close": (item.get("raw_metrics") or {}).get("close"),
            "daily_change_pct": (item.get("raw_metrics") or {}).get("daily_change_pct"),
            "drawdown_60d_high_pct": (item.get("raw_metrics") or {}).get("drawdown_from_60d_high_pct"),
            "action_stance": item.get("action_stance"),
        })
    csv_path = RESULT_DIR / "venus8_technical_a_20260724.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["rank", "code", "name", "score"])
        writer.writeheader()
        writer.writerows(rows)

    summary_lines = [
        f"# Venus8 全市场技术A扫描（{TARGET_DATE}）",
        "",
        f"- 数据版本：{tag}",
        f"- 实际进入评分：{scanned} 只",
        f"- 技术A：{len(grade_a)} 只",
        f"- 剔除ST/退市/新股前缀后：{len(clean_a)} 只",
        f"- 接近A（B且>=78、无排除）：{len(near_a)} 只",
        "",
        "|排名|代码|名称|总分|趋势|回调|动能|量能|价格确认|BOLL|收盘|当日涨跌%|",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        summary_lines.append(
            f"|{row['rank']}|{row['code']}|{row['name']}|{row['score']}|{row['trend']}|{row['pullback']}|{row['momentum']}|{row['volume']}|{row['price_confirmation']}|{row['boll']}|{row['close']}|{row['daily_change_pct']}|"
        )
    (RESULT_DIR / "venus8_technical_a_20260724.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(json.dumps(output["counts"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
