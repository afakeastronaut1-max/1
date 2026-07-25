from __future__ import annotations

import csv
import json
import math
import random
import re
import struct
import time
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import scan_venus as qlibscan

TARGET_DATE = "2026-07-24"
TARGET_INT = 20260724
PREV_INT = 20260723
ROOT = Path(__file__).resolve().parent
WORK = ROOT / "work_hybrid"
RESULT_DIR = ROOT / "results"
TDX_URL = "https://data.tdx.com.cn/vipdoc/hsjday.zip"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/140 Safari/537.36",
    "Accept": "application/json,text/plain,*/*",
    "Referer": "https://gu.qq.com/",
}


def download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=180) as resp, path.open("wb") as out:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    if path.stat().st_size < 1_000_000:
        raise RuntimeError(f"download too small: {path.stat().st_size}")


def tdx_records() -> dict[str, dict[int, dict[str, float]]]:
    path = WORK / "hsjday.zip"
    print(f"downloading TDX full daily package: {TDX_URL}", flush=True)
    download(TDX_URL, path)
    print(f"TDX package bytes={path.stat().st_size}", flush=True)
    result: dict[str, dict[int, dict[str, float]]] = {}
    fmt = struct.Struct("<IIIIIfII")
    with zipfile.ZipFile(path) as zf:
        members = [x for x in zf.infolist() if x.filename.lower().endswith(".day")]
        print(f"TDX day files={len(members)}", flush=True)
        for info in members:
            base = info.filename.replace("\\", "/").split("/")[-1].lower()
            match = re.fullmatch(r"(sh|sz|bj)(\d{6})\.day", base)
            if not match:
                continue
            symbol = (match.group(1) + match.group(2)).upper()
            raw = zf.read(info)
            rows: dict[int, dict[str, float]] = {}
            start = max(0, len(raw) - 32 * 6)
            start -= start % 32
            for offset in range(start, len(raw) - 31, 32):
                date, op, hi, lo, cl, amount, volume, _ = fmt.unpack_from(raw, offset)
                if date in {PREV_INT, TARGET_INT} and min(op, hi, lo, cl) > 0:
                    rows[date] = {
                        "open": op / 100.0, "high": hi / 100.0, "low": lo / 100.0,
                        "close": cl / 100.0, "volume": float(volume), "turnover": float(amount),
                    }
            if TARGET_INT in rows:
                result[symbol] = rows
    print(f"TDX symbols with {TARGET_DATE}={len(result)}", flush=True)
    return result


def tencent_qfq(symbol: str) -> tuple[pd.DataFrame, str]:
    lower = symbol.lower()
    param = f"{lower},day,2024-01-01,{TARGET_DATE},500,qfq"
    url = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?" + urllib.parse.urlencode({"param": param})
    last: Exception | None = None
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=25) as resp:
                payload = json.load(resp)
            node = (((payload or {}).get("data") or {}).get(lower) or {})
            lines = node.get("qfqday") or node.get("day") or []
            quote = ((node.get("qt") or {}).get(lower) or [])
            name = str(quote[1]) if len(quote) > 1 else ""
            records = []
            for p in lines:
                if len(p) >= 6:
                    records.append({
                        "date": str(p[0]), "open": float(p[1]), "close": float(p[2]),
                        "high": float(p[3]), "low": float(p[4]), "volume": float(p[5]),
                        "turnover": np.nan,
                    })
            df = pd.DataFrame(records)
            if len(df) >= 80 and str(df.iloc[-1]["date"]) == TARGET_DATE:
                return df.tail(300).reset_index(drop=True), name
            raise RuntimeError(f"incomplete Tencent bars={len(df)} last={df.iloc[-1]['date'] if len(df) else None}")
        except Exception as exc:
            last = exc
            time.sleep(0.4 * (attempt + 1) + random.random() * 0.2)
    raise RuntimeError(f"Tencent failed: {type(last).__name__}: {last}")


def append_friday(frame: pd.DataFrame, pair: dict[int, dict[str, float]]) -> pd.DataFrame | None:
    if PREV_INT not in pair or TARGET_INT not in pair or frame.empty:
        return None
    prev_tdx, fri = pair[PREV_INT], pair[TARGET_INT]
    prev_close = float(frame.iloc[-1]["close"])
    if prev_tdx["close"] <= 0 or prev_close <= 0:
        return None
    scale = prev_close / prev_tdx["close"]
    prev_volume = float(frame.iloc[-1].get("volume") or 0.0)
    volume = prev_volume * fri["volume"] / prev_tdx["volume"] if prev_tdx["volume"] > 0 and prev_volume > 0 else fri["volume"]
    prev_turn = float(frame.iloc[-1].get("turnover") or 0.0)
    turnover = prev_turn * fri["turnover"] / prev_tdx["turnover"] if prev_tdx["turnover"] > 0 and prev_turn > 0 else fri["turnover"]
    row = {
        "date": TARGET_DATE,
        "open": fri["open"] * scale, "high": fri["high"] * scale,
        "low": fri["low"] * scale, "close": fri["close"] * scale,
        "volume": volume, "turnover": turnover,
    }
    return pd.concat([frame, pd.DataFrame([row])], ignore_index=True).tail(300).reset_index(drop=True)


def safe(value: Any) -> Any:
    if isinstance(value, np.integer): return int(value)
    if isinstance(value, np.floating):
        value = float(value); return value if math.isfinite(value) else None
    if isinstance(value, float): return value if math.isfinite(value) else None
    if isinstance(value, dict): return {k: safe(v) for k, v in value.items()}
    if isinstance(value, list): return [safe(v) for v in value]
    return value


def main() -> None:
    WORK.mkdir(exist_ok=True)
    qlibscan.prepare_runtime()
    from src.analysis.golden_pit import score_golden_pit

    release_tag, qlib_url = qlibscan.prepare_qlib()
    calendar = qlibscan.read_calendar()
    prev_date = calendar[-1]
    if prev_date != "2026-07-23":
        raise RuntimeError(f"unexpected Qlib last date={prev_date}")
    prev_idx = len(calendar) - 1
    instruments = qlibscan.read_instruments()
    tdx = tdx_records()
    qlibscan.TARGET_DATE = prev_date

    prelim: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    scanned = 0
    for i, (symbol, start, end) in enumerate(instruments, 1):
        if not qlibscan.valid_a_share(symbol) or start > prev_date or symbol not in tdx:
            continue
        try:
            frame = qlibscan.build_frame(symbol, calendar, prev_idx)
            if frame is None or len(frame) < 80:
                continue
            extended = append_friday(frame, tdx[symbol])
            if extended is None:
                continue
            scanned += 1
            item = safe(score_golden_pit(symbol=symbol[2:], name="", bars=extended, period="daily"))
            item["exchange_symbol"] = symbol
            item["preliminary_source"] = "Qlib adjusted through 2026-07-23 + TDX 2026-07-24 relative append"
            raw_score = int(item.get("total_score") or 0)
            exclusions = item.get("exclusions") or []
            pair = tdx[symbol]
            raw_ret = pair[TARGET_INT]["close"] / pair[PREV_INT]["close"] - 1 if PREV_INT in pair else 0.0
            suspected_action = abs(raw_ret) > 0.115
            if item.get("grade") == "A" or (raw_score >= 68 and len(exclusions) <= 1) or suspected_action:
                item["raw_tdx_friday_return"] = raw_ret
                item["suspected_friday_corporate_action_or_large_move"] = suspected_action
                prelim.append(item)
        except Exception as exc:
            failures.append({"symbol": symbol, "stage": "preliminary", "error": str(exc)[:500]})
        if i % 500 == 0:
            print(f"preliminary progress={i}/{len(instruments)} scanned={scanned} verify_pool={len(prelim)}", flush=True)

    print(f"preliminary complete scanned={scanned} verification_pool={len(prelim)}", flush=True)
    verified: list[dict[str, Any]] = []
    verification_failures: list[dict[str, str]] = []

    def verify(item: dict[str, Any]) -> dict[str, Any]:
        bars, name = tencent_qfq(str(item["exchange_symbol"]))
        exact = safe(score_golden_pit(symbol=str(item.get("symbol") or ""), name=name, bars=bars, period="daily"))
        exact["exchange_symbol"] = item["exchange_symbol"]
        exact["data_source"] = "Tencent qfq daily exact verification"
        exact["preliminary_score"] = item.get("total_score")
        exact["data_bar_count"] = len(bars)
        exact["as_of"] = TARGET_DATE
        return exact

    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = {pool.submit(verify, item): item for item in prelim}
        for n, future in enumerate(as_completed(futures), 1):
            item = futures[future]
            try:
                verified.append(future.result())
            except Exception as exc:
                verification_failures.append({"symbol": str(item.get("exchange_symbol")), "stage": "tencent_verify", "error": str(exc)[:500]})
            if n % 50 == 0:
                print(f"verification progress={n}/{len(prelim)} failures={len(verification_failures)}", flush=True)

    verified.sort(key=lambda x: (x.get("grade") == "A", int(x.get("total_score") or 0)), reverse=True)
    grade_a = [x for x in verified if x.get("grade") == "A"]
    clean_a = [x for x in grade_a if not re.search(r"(?:\*?ST|退)", str(x.get("name") or ""), re.I)]
    near_a = [x for x in verified if x.get("grade") != "A" and x.get("is_near_a_grade")]

    output = {
        "schema": "venus8_full_market_technical_a_hybrid.v1",
        "target_date": TARGET_DATE,
        "method": "exact Venus8 score_golden_pit; all final A rows independently verified on Tencent qfq bars",
        "data_chain": {
            "history": f"chenditc/investment_data Qlib release {release_tag} through 2026-07-23",
            "friday_full_market": "TDX official hsjday.zip 2026-07-24",
            "candidate_verification": "Tencent qfq daily through 2026-07-24",
            "qlib_asset": qlib_url,
        },
        "universe_instruments": len(instruments), "preliminary_scanned": scanned,
        "verification_pool": len(prelim), "verified_count": len(verified),
        "preliminary_failures": len(failures), "verification_failures": len(verification_failures),
        "technical_a_count": len(grade_a), "technical_a_clean_count": len(clean_a), "near_a_count": len(near_a),
        "technical_a": grade_a, "technical_a_clean": clean_a, "near_a": near_a[:200],
        "failure_samples": (failures + verification_failures)[:100],
    }
    RESULT_DIR.mkdir(exist_ok=True)
    (RESULT_DIR / "venus8_technical_a_20260724.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    fields = ["rank", "code", "name", "score", "trend", "pullback", "momentum", "volume", "price_confirmation", "boll", "close", "daily_change_pct", "stance"]
    rows = []
    for rank, item in enumerate(clean_a, 1):
        d, raw = item.get("dimensions") or {}, item.get("raw_metrics") or {}
        rows.append({
            "rank": rank, "code": item.get("symbol"), "name": item.get("name"), "score": item.get("total_score"),
            "trend": (d.get("trend_structure") or {}).get("score"), "pullback": (d.get("pullback_quality") or {}).get("score"),
            "momentum": (d.get("momentum_repair") or {}).get("score"), "volume": (d.get("volume_activity") or {}).get("score"),
            "price_confirmation": (d.get("price_confirmation") or {}).get("score"), "boll": (d.get("boll_prediction") or {}).get("score"),
            "close": raw.get("close"), "daily_change_pct": raw.get("daily_change_pct"), "stance": item.get("action_stance"),
        })
    with (RESULT_DIR / "venus8_technical_a_20260724.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    md = [
        f"# Venus8 全市场技术A扫描（{TARGET_DATE}）", "", f"- 初筛覆盖：{scanned}", f"- 腾讯前复权复核：{len(verified)}",
        f"- 技术A：{len(grade_a)}", f"- 非ST/退市技术A：{len(clean_a)}", f"- 复核失败：{len(verification_failures)}", "",
        "|排名|代码|名称|总分|趋势|回调|动能|量能|确认|BOLL|收盘|涨跌%|",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        md.append(f"|{r['rank']}|{r['code']}|{r['name']}|{r['score']}|{r['trend']}|{r['pullback']}|{r['momentum']}|{r['volume']}|{r['price_confirmation']}|{r['boll']}|{r['close']}|{r['daily_change_pct']}|")
    (RESULT_DIR / "venus8_technical_a_20260724.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"preliminary_scanned": scanned, "verification_pool": len(prelim), "verified": len(verified), "technical_a": len(grade_a), "clean_a": len(clean_a), "verify_failures": len(verification_failures)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
