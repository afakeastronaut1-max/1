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
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

TARGET_DATE = "2026-07-24"
TARGET_DATE_COMPACT = "20260724"
ROOT = Path(__file__).resolve().parent
WORK = ROOT / "work"
RESULT_DIR = ROOT / "results"
ARCHIVE_URLS = [
    "https://github.com/chenditc/investment_data/releases/latest/download/qlib_bin.tar.gz",
    "https://github.com/chenditc/investment_data/releases/download/2026-07-25/qlib_bin.tar.gz",
]
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Venus8Scan/1.0"


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
        cmd = ["curl", "-L", "--fail", "--retry", "4", "--retry-all-errors", "--connect-timeout", "30", "--max-time", "1200", "-o", str(archive), url]
        try:
            subprocess.run(cmd, check=True)
            if archive.stat().st_size < 10_000_000 or not tarfile.is_tarfile(archive):
                raise RuntimeError(f"invalid qlib archive: {archive.stat().st_size}")
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
        raise RuntimeError("qlib calendar not found")
    root = calendars[0].parent.parent
    if not (root / "instruments" / "all.txt").exists() or not (root / "features").exists():
        raise RuntimeError(f"invalid qlib root: {root}")
    return root


def read_calendar(root: Path) -> list[str]:
    dates = [x.strip()[:10] for x in (root / "calendars" / "day.txt").read_text(encoding="utf-8").splitlines() if x.strip()]
    if not dates:
        raise RuntimeError("empty qlib calendar")
    print(f"calendar_last={dates[-1]}", flush=True)
    if dates[-1] > TARGET_DATE:
        dates = [d for d in dates if d <= TARGET_DATE]
    if dates[-1] != "2026-07-23":
        raise RuntimeError(f"expected historical base through 2026-07-23, got {dates[-1]}")
    return dates


def read_instruments(root: Path) -> list[tuple[str, str, str]]:
    out = []
    for line in (root / "instruments" / "all.txt").read_text(encoding="utf-8").splitlines():
        p = re.split(r"\s+", line.strip())
        if len(p) >= 3:
            out.append((p[0].upper(), p[1][:10], p[2][:10]))
    return out


def valid_symbol(symbol: str) -> bool:
    s = symbol.upper()
    if s.startswith("SH"):
        return s[2:].startswith(("600", "601", "603", "605", "688", "689"))
    if s.startswith("SZ"):
        return s[2:].startswith(("000", "001", "002", "003", "300", "301"))
    if s.startswith("BJ"):
        return s[2:].isdigit()
    return False


def quote_id(symbol: str) -> str:
    s = symbol.lower()
    if s.startswith("bj"):
        return "bj" + s[2:]
    return s


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
    vals = raw[1:].astype(float)
    end = min(calendar_len, start + vals.size)
    out[start:end] = vals[: end - start]
    return out


def build_base_frame(root: Path, symbol: str, calendar: list[str]) -> pd.DataFrame | None:
    folder = root / "features" / symbol.lower()
    if not folder.exists():
        return None
    arrays: dict[str, np.ndarray] = {}
    for field in ("open", "high", "low", "close", "volume", "amount"):
        arr = read_feature(folder / f"{field}.day.bin", len(calendar))
        if arr is not None:
            arrays[field] = arr
    if not {"open", "high", "low", "close", "volume"}.issubset(arrays):
        return None
    start = max(0, len(calendar) - 299)
    frame = pd.DataFrame({
        "date": calendar[start:],
        "open": arrays["open"][start:],
        "high": arrays["high"][start:],
        "low": arrays["low"][start:],
        "close": arrays["close"][start:],
        "volume": arrays["volume"][start:],
        "turnover": arrays.get("amount", np.full(len(calendar), np.nan))[start:],
    })
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["open", "high", "low", "close"])
    if len(frame) < 79 or frame.iloc[-1]["date"] != "2026-07-23":
        return None
    return frame.reset_index(drop=True)


def chunks(items: list[str], size: int) -> list[list[str]]:
    return [items[i:i + size] for i in range(0, len(items), size)]


def http_text(url: str, encoding: str, referer: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Referer": referer, "Accept": "*/*"})
    with urllib.request.urlopen(req, timeout=35) as resp:
        return resp.read().decode(encoding, errors="replace")


def fetch_sina_batch(ids: list[str]) -> dict[str, dict[str, Any]]:
    url = "https://hq.sinajs.cn/list=" + ",".join(ids)
    text = http_text(url, "gb18030", "https://finance.sina.com.cn/")
    out: dict[str, dict[str, Any]] = {}
    for qid, body in re.findall(r'var hq_str_([a-z0-9]+)="([^"]*)";', text, flags=re.I):
        p = body.split(",")
        if len(p) < 32 or p[30] != TARGET_DATE:
            continue
        try:
            op, prev, close, high, low = map(float, (p[1], p[2], p[3], p[4], p[5]))
            volume, amount = float(p[8]), float(p[9])
        except ValueError:
            continue
        if min(op, prev, close, high, low) <= 0:
            continue
        out[qid.lower()] = {"name": p[0], "open": op, "prev_close": prev, "close": close, "high": high, "low": low, "volume": volume, "turnover": amount, "source": "sina"}
    return out


def fetch_tencent_batch(ids: list[str]) -> dict[str, dict[str, Any]]:
    url = "https://qt.gtimg.cn/q=" + ",".join(ids)
    text = http_text(url, "gb18030", "https://gu.qq.com/")
    out: dict[str, dict[str, Any]] = {}
    for qid, body in re.findall(r'v_([a-z0-9]+)="([^"]*)";', text, flags=re.I):
        p = body.split("~")
        if len(p) < 35 or not str(p[30]).startswith(TARGET_DATE_COMPACT):
            continue
        try:
            close, prev, op = float(p[3]), float(p[4]), float(p[5])
            volume = float(p[6]) * 100.0
            high, low = float(p[33]), float(p[34])
            amount = float(p[37]) * 10000.0 if len(p) > 37 and p[37] else np.nan
        except (ValueError, IndexError):
            continue
        if min(op, prev, close, high, low) <= 0:
            continue
        out[qid.lower()] = {"name": p[1], "open": op, "prev_close": prev, "close": close, "high": high, "low": low, "volume": volume, "turnover": amount, "source": "tencent"}
    return out


def fetch_snapshot_batch(ids: list[str]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    errors = []
    try:
        first = fetch_sina_batch(ids)
    except Exception as exc:
        first = {}
        errors.append(f"sina:{type(exc).__name__}:{exc}")
    missing = [x for x in ids if x.lower() not in first]
    if missing:
        try:
            second = fetch_tencent_batch(missing)
            first.update(second)
        except Exception as exc:
            errors.append(f"tencent:{type(exc).__name__}:{exc}")
    return first, errors


def fetch_all_snapshots(symbols: list[str]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    ids = [quote_id(s) for s in symbols]
    batches = chunks(ids, 80)
    snapshots: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=12) as pool:
        futures = [pool.submit(fetch_snapshot_batch, b) for b in batches]
        for idx, fut in enumerate(as_completed(futures), 1):
            data, errs = fut.result()
            snapshots.update(data)
            errors.extend(errs)
            if idx % 10 == 0:
                print(f"snapshot_batches={idx}/{len(batches)} quotes={len(snapshots)} errors={len(errors)}", flush=True)
    print(f"snapshot_coverage={len(snapshots)}/{len(symbols)}", flush=True)
    if len(snapshots) < max(3500, int(len(symbols) * 0.70)):
        raise RuntimeError(f"7/24 snapshot coverage too low: {len(snapshots)}/{len(symbols)}; errors={errors[:10]}")
    return snapshots, errors


def append_snapshot(base: pd.DataFrame, snap: dict[str, Any]) -> pd.DataFrame | None:
    raw_prev = float(snap["prev_close"])
    qlib_prev = float(base.iloc[-1]["close"])
    if raw_prev <= 0 or qlib_prev <= 0:
        return None
    scale = qlib_prev / raw_prev
    row = {
        "date": TARGET_DATE,
        "open": float(snap["open"]) * scale,
        "high": float(snap["high"]) * scale,
        "low": float(snap["low"]) * scale,
        "close": float(snap["close"]) * scale,
        "volume": float(snap["volume"]),
        "turnover": float(snap["turnover"]) if pd.notna(snap.get("turnover")) else np.nan,
    }
    return pd.concat([base, pd.DataFrame([row])], ignore_index=True)


def clean_name(name: str) -> bool:
    return not bool(re.search(r"(?:\*?ST|退$|^N|^C)", name or "", re.I))


def compact(item: dict[str, Any], rank: int) -> dict[str, Any]:
    dims = item.get("dimensions") or {}
    raw = item.get("raw_metrics") or {}
    def s(key: str): return (dims.get(key) or {}).get("score")
    return {"rank": rank, "code": item.get("symbol"), "name": item.get("name"), "score": item.get("total_score"), "trend": s("trend_structure"), "pullback": s("pullback_quality"), "momentum": s("momentum_repair"), "volume": s("volume_activity"), "price_confirmation": s("price_confirmation"), "boll": s("boll_prediction"), "close": raw.get("close"), "daily_change_pct": raw.get("daily_change_pct"), "drawdown_60d_high_pct": raw.get("drawdown_from_60d_high_pct"), "ret_20d_pct": raw.get("ret_20d_pct"), "ret_60d_pct": raw.get("ret_60d_pct"), "action_stance": item.get("action_stance"), "supports": (item.get("key_evidence") or {}).get("supports") or [], "risk_points": item.get("risk_points") or []}


def main() -> None:
    prepare_runtime()
    from src.analysis.golden_pit import score_golden_pit

    archive, archive_url = download_archive()
    root = extract_archive(archive)
    calendar = read_calendar(root)
    instruments = read_instruments(root)
    active = [(s, a, b) for s, a, b in instruments if valid_symbol(s) and a <= "2026-07-23" and b >= "2026-07-23"]
    symbols = [s for s, _, _ in active]
    snapshots, snapshot_errors = fetch_all_snapshots(symbols)

    candidates: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    scored = 0
    snapshot_matched = 0
    source_counts: dict[str, int] = {}
    for index, (symbol, _start, _end) in enumerate(active, 1):
        snap = snapshots.get(quote_id(symbol))
        if snap is None:
            continue
        snapshot_matched += 1
        source_counts[snap["source"]] = source_counts.get(snap["source"], 0) + 1
        try:
            base = build_base_frame(root, symbol, calendar)
            if base is None:
                continue
            frame = append_snapshot(base, snap)
            if frame is None or len(frame) < 80:
                continue
            scored += 1
            payload = score_golden_pit(symbol[2:], frame, name=str(snap.get("name") or ""), period="daily")
            payload["exchange_symbol"] = symbol
            payload["snapshot_source"] = snap["source"]
            if payload.get("grade") == "A" or payload.get("is_near_a_grade"):
                candidates.append(payload)
        except Exception as exc:
            failures.append({"symbol": symbol, "error": f"{type(exc).__name__}: {exc}"})
        if index % 500 == 0:
            print(f"progress={index}/{len(active)} matched={snapshot_matched} scored={scored} candidates={len(candidates)} failures={len(failures)}", flush=True)

    candidates.sort(key=lambda x: (x.get("grade") == "A", int(x.get("total_score") or 0)), reverse=True)
    grade_a = [x for x in candidates if x.get("grade") == "A"]
    clean_a = [x for x in grade_a if clean_name(str(x.get("name") or ""))]
    near_a = [x for x in candidates if x.get("grade") != "A"]
    rows = [compact(x, i) for i, x in enumerate(clean_a, 1)]
    output = {
        "schema": "venus8_full_market_technical_a_scan.v6",
        "target_date": TARGET_DATE,
        "method": "exact Venus8 src.analysis.golden_pit.score_golden_pit",
        "technical_a_rule": "total_score >= 85 and no Venus8 hard exclusion",
        "data_source": {"historical": "chenditc/investment_data Qlib through 2026-07-23", "historical_archive": archive_url, "target_day": "Sina primary + Tencent fallback 2026-07-24 close snapshot", "price_alignment": "target-day raw OHLC multiplied by Qlib_prev_close/raw_prev_close"},
        "universe": {"instrument_rows": len(instruments), "active_a_share_symbols": len(active), "target_day_quotes": len(snapshots), "snapshot_matched": snapshot_matched, "scored_with_80_bars": scored, "score_failures": len(failures), "snapshot_sources": source_counts},
        "counts": {"technical_a_all": len(grade_a), "technical_a_clean": len(clean_a), "near_a": len(near_a)},
        "technical_a": grade_a, "technical_a_clean": clean_a, "near_a": near_a[:150], "failures_sample": failures[:100], "snapshot_error_sample": snapshot_errors[:30],
    }
    RESULT_DIR.mkdir(exist_ok=True)
    (RESULT_DIR / "venus8_technical_a_20260724.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fields = list(rows[0].keys()) if rows else ["rank", "code", "name", "score"]
    with (RESULT_DIR / "venus8_technical_a_20260724.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    lines = [f"# Venus8 全市场技术A扫描（{TARGET_DATE}）", "", f"- 活跃A股：{len(active)}只", f"- 7月24日有效报价：{len(snapshots)}只", f"- 实际评分：{scored}只", f"- 技术A：{len(grade_a)}只", f"- 清理ST/退市/新股前缀后：{len(clean_a)}只", f"- 接近A：{len(near_a)}只", "", "|排名|代码|名称|总分|趋势|回调|动能|量能|价格确认|BOLL|收盘|当日涨跌%|", "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"|{r['rank']}|{r['code']}|{r['name']}|{r['score']}|{r['trend']}|{r['pullback']}|{r['momentum']}|{r['volume']}|{r['price_confirmation']}|{r['boll']}|{r['close']}|{r['daily_change_pct']}|")
    (RESULT_DIR / "venus8_technical_a_20260724.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("VENUS_RESULT_BEGIN")
    print(json.dumps({"counts": output["counts"], "universe": output["universe"], "top": rows[:50]}, ensure_ascii=False))
    print("VENUS_RESULT_END")


if __name__ == "__main__":
    main()
