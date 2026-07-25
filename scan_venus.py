from __future__ import annotations

import base64
import csv
import io
import json
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

TARGET_DATE = "2026-07-24"
ROOT = Path(__file__).resolve().parent
WORK = ROOT / "work"
RESULT_DIR = ROOT / "results"
ARCHIVE_URLS = [
    "https://github.com/chenditc/investment_data/releases/latest/download/qlib_bin.tar.gz",
    "https://github.com/chenditc/investment_data/releases/download/2026-07-25/qlib_bin.tar.gz",
]


def prepare_runtime() -> None:
    raw = base64.b64decode((ROOT / "venus_runtime.b64").read_text(encoding="ascii"))
    vendor = ROOT / "vendor"
    vendor.mkdir(exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tf:
        tf.extractall(vendor)
    sys.path.insert(0, str(vendor))


def download_archive() -> tuple[Path, str]:
    WORK.mkdir(exist_ok=True)
    archive = WORK / "qlib_bin.tar.gz"
    last = None
    for url in ARCHIVE_URLS:
        archive.unlink(missing_ok=True)
        cmd = [
            "curl", "-L", "--fail", "--retry", "4", "--retry-all-errors",
            "--connect-timeout", "30", "--max-time", "1200", "-o", str(archive), url,
        ]
        try:
            subprocess.run(cmd, check=True)
            if archive.stat().st_size < 10_000_000 or not tarfile.is_tarfile(archive):
                raise RuntimeError(f"invalid or too-small qlib archive: {archive.stat().st_size}")
            print(f"archive={archive.stat().st_size / 1024 / 1024:.1f}MB url={url}", flush=True)
            return archive, url
        except Exception as exc:
            last = exc
            print(f"archive source failed {url}: {exc}", flush=True)
    raise RuntimeError(f"all qlib archive sources failed: {last}")


def extract_archive(archive: Path) -> Path:
    target = WORK / "qlib_data"
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    with tarfile.open(archive, "r:gz") as tf:
        tf.extractall(target)
    calendars = list(target.rglob("calendars/day.txt"))
    if not calendars:
        raise RuntimeError("qlib calendar not found after extraction")
    data_root = calendars[0].parent.parent
    required = [data_root / "instruments" / "all.txt", data_root / "features"]
    if not all(p.exists() for p in required):
        raise RuntimeError(f"invalid qlib root: {data_root}")
    print(f"qlib_root={data_root}", flush=True)
    return data_root


def read_calendar(root: Path) -> list[str]:
    values = []
    for line in (root / "calendars" / "day.txt").read_text(encoding="utf-8").splitlines():
        date = line.strip()[:10]
        if date:
            values.append(date)
    if TARGET_DATE not in values:
        raise RuntimeError(f"target date missing; calendar_last={values[-1] if values else None}")
    print(f"calendar_last={values[-1]} target_index={values.index(TARGET_DATE)}", flush=True)
    return values


def read_instruments(root: Path) -> list[tuple[str, str, str]]:
    rows = []
    for line in (root / "instruments" / "all.txt").read_text(encoding="utf-8").splitlines():
        p = re.split(r"\s+", line.strip())
        if len(p) >= 3:
            rows.append((p[0].upper(), p[1][:10], p[2][:10]))
    return rows


def valid_symbol(symbol: str) -> bool:
    s = symbol.upper()
    if s.startswith("SH"):
        return s[2:].startswith(("600", "601", "603", "605", "688", "689"))
    if s.startswith("SZ"):
        return s[2:].startswith(("000", "001", "002", "003", "300", "301"))
    if s.startswith("BJ"):
        return s[2:].isdigit()
    return False


def read_feature(path: Path, calendar_len: int) -> np.ndarray | None:
    if not path.exists():
        return None
    raw = np.fromfile(path, dtype="<f4")
    if raw.size < 2:
        return None
    start = int(round(float(raw[0])))
    if start < 0 or start >= calendar_len:
        return None
    out = np.full(calendar_len, np.nan, dtype=float)
    values = raw[1:].astype(float)
    end = min(calendar_len, start + values.size)
    out[start:end] = values[: end - start]
    return out


def build_frame(root: Path, symbol: str, calendar: list[str], target_idx: int) -> pd.DataFrame | None:
    folder = root / "features" / symbol.lower()
    if not folder.exists():
        return None
    arrays: dict[str, np.ndarray] = {}
    for field in ("open", "high", "low", "close", "volume", "amount"):
        value = read_feature(folder / f"{field}.day.bin", len(calendar))
        if value is not None:
            arrays[field] = value
    if not {"open", "high", "low", "close", "volume"}.issubset(arrays):
        return None
    start = max(0, target_idx - 299)
    frame = pd.DataFrame({
        "date": calendar[start : target_idx + 1],
        "open": arrays["open"][start : target_idx + 1],
        "high": arrays["high"][start : target_idx + 1],
        "low": arrays["low"][start : target_idx + 1],
        "close": arrays["close"][start : target_idx + 1],
        "volume": arrays["volume"][start : target_idx + 1],
        "turnover": arrays.get("amount", np.full(len(calendar), np.nan))[start : target_idx + 1],
    })
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["open", "high", "low", "close"])
    if len(frame) < 80 or frame.iloc[-1]["date"] != TARGET_DATE:
        return None
    return frame.reset_index(drop=True)


def clean_name(name: str) -> bool:
    return not bool(re.search(r"(?:\*?ST|退$|^N|^C)", name or "", re.I))


def compact(item: dict[str, Any], rank: int) -> dict[str, Any]:
    dims = item.get("dimensions") or {}
    raw = item.get("raw_metrics") or {}
    def s(key: str): return (dims.get(key) or {}).get("score")
    return {
        "rank": rank,
        "code": item.get("symbol"),
        "name": item.get("name"),
        "score": item.get("total_score"),
        "trend": s("trend_structure"),
        "pullback": s("pullback_quality"),
        "momentum": s("momentum_repair"),
        "volume": s("volume_activity"),
        "price_confirmation": s("price_confirmation"),
        "boll": s("boll_prediction"),
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

    archive, archive_url = download_archive()
    data_root = extract_archive(archive)
    calendar = read_calendar(data_root)
    target_idx = calendar.index(TARGET_DATE)
    instruments = read_instruments(data_root)

    candidates: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    scored = 0
    selected_universe = 0
    for index, (symbol, start_date, end_date) in enumerate(instruments, 1):
        if not valid_symbol(symbol) or start_date > TARGET_DATE or end_date < TARGET_DATE:
            continue
        selected_universe += 1
        try:
            frame = build_frame(data_root, symbol, calendar, target_idx)
            if frame is None:
                continue
            scored += 1
            code = symbol[2:]
            payload = score_golden_pit(code, frame, name="", period="daily")
            payload["exchange_symbol"] = symbol
            if payload.get("grade") == "A" or payload.get("is_near_a_grade"):
                candidates.append(payload)
        except Exception as exc:
            failures.append({"symbol": symbol, "error": f"{type(exc).__name__}: {exc}"})
        if selected_universe % 500 == 0:
            print(f"progress={selected_universe} scored={scored} candidates={len(candidates)} failures={len(failures)}", flush=True)

    candidates.sort(key=lambda x: (x.get("grade") == "A", int(x.get("total_score") or 0)), reverse=True)
    grade_a = [x for x in candidates if x.get("grade") == "A"]
    clean_a = [x for x in grade_a if clean_name(str(x.get("name") or ""))]
    near_a = [x for x in candidates if x.get("grade") != "A"]
    rows = [compact(item, rank) for rank, item in enumerate(clean_a, 1)]

    output = {
        "schema": "venus8_full_market_technical_a_scan.v5",
        "target_date": TARGET_DATE,
        "method": "exact Venus8 src.analysis.golden_pit.score_golden_pit",
        "technical_a_rule": "total_score >= 85 and no Venus8 hard exclusion",
        "data_source": {
            "provider": "chenditc/investment_data Qlib daily full-market release",
            "archive_url": archive_url,
            "calendar_last": calendar[-1],
            "price_basis": "Qlib continuous adjusted OHLCV",
        },
        "universe": {
            "instrument_rows": len(instruments),
            "active_a_share_symbols": selected_universe,
            "scored_with_80_bars_and_target_close": scored,
            "score_failures": len(failures),
        },
        "counts": {
            "technical_a_all": len(grade_a),
            "technical_a_clean": len(clean_a),
            "near_a": len(near_a),
        },
        "technical_a": grade_a,
        "technical_a_clean": clean_a,
        "near_a": near_a[:150],
        "failures_sample": failures[:100],
    }
    RESULT_DIR.mkdir(exist_ok=True)
    (RESULT_DIR / "venus8_technical_a_20260724.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fields = list(rows[0].keys()) if rows else ["rank", "code", "name", "score"]
    with (RESULT_DIR / "venus8_technical_a_20260724.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        f"# Venus8 全市场技术A扫描（{TARGET_DATE}）", "",
        f"- 活跃A股：{selected_universe}只", f"- 实际评分：{scored}只",
        f"- 技术A：{len(grade_a)}只", f"- 接近A：{len(near_a)}只", "",
        "|排名|代码|总分|趋势|回调|动能|量能|价格确认|BOLL|收盘|当日涨跌%|",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(f"|{row['rank']}|{row['code']}|{row['score']}|{row['trend']}|{row['pullback']}|{row['momentum']}|{row['volume']}|{row['price_confirmation']}|{row['boll']}|{row['close']}|{row['daily_change_pct']}|")
    (RESULT_DIR / "venus8_technical_a_20260724.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("VENUS_RESULT_BEGIN")
    print(json.dumps({"counts": output["counts"], "universe": output["universe"], "top": rows[:50]}, ensure_ascii=False))
    print("VENUS_RESULT_END")


if __name__ == "__main__":
    main()
