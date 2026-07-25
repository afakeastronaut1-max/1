from __future__ import annotations

import csv
import json
import math
import random
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import scan_venus as qlibscan
from scan_hybrid import safe, tencent_qfq

TARGET_DATE = "2026-07-24"
PREV_DATE = "2026-07-23"
ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "results"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/140 Safari/537.36",
    "Referer": "https://gu.qq.com/",
}


def fetch_quote_chunk(symbols: list[str]) -> dict[str, dict[str, Any]]:
    query = ",".join(s.lower() for s in symbols)
    hosts = ["qt.gtimg.cn", "web.sqt.gtimg.cn"]
    last: Exception | None = None
    for attempt in range(4):
        host = hosts[attempt % len(hosts)]
        try:
            req = urllib.request.Request(f"https://{host}/q={query}", headers=HEADERS)
            with urllib.request.urlopen(req, timeout=25) as resp:
                text = resp.read().decode("gb18030", errors="ignore")
            output: dict[str, dict[str, Any]] = {}
            for line in text.splitlines():
                m = re.search(r'v_([a-z]{2}\d{6})="(.*)";', line, re.I)
                if not m:
                    continue
                symbol = m.group(1).upper()
                p = m.group(2).split("~")
                if len(p) < 38:
                    continue
                try:
                    stamp = str(p[30])
                    current, prev, op, hi, lo = map(float, (p[3], p[4], p[5], p[33], p[34]))
                    volume_hands = float(p[36])
                    amount_wan = float(p[37])
                except Exception:
                    continue
                if not stamp.startswith("20260724") or min(current, prev, op, hi, lo) <= 0:
                    continue
                output[symbol] = {
                    "name": str(p[1]), "date": TARGET_DATE, "open": op, "high": hi, "low": lo,
                    "close": current, "prev_close": prev, "volume": volume_hands * 100.0,
                    "turnover": amount_wan * 10000.0,
                }
            if output:
                return output
            raise RuntimeError("empty quote response")
        except Exception as exc:
            last = exc
            time.sleep(0.35 * (attempt + 1) + random.random() * 0.15)
    raise RuntimeError(f"quote chunk failed: {type(last).__name__}: {last}")


def full_snapshot(symbols: list[str]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    chunks = [symbols[i:i + 55] for i in range(0, len(symbols), 55)]
    output: dict[str, dict[str, Any]] = {}
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(fetch_quote_chunk, chunk): chunk for chunk in chunks}
        for n, future in enumerate(as_completed(futures), 1):
            chunk = futures[future]
            try:
                output.update(future.result())
            except Exception as exc:
                failures.extend(chunk)
                print(f"quote chunk failed size={len(chunk)} error={exc}", flush=True)
            if n % 20 == 0:
                print(f"snapshot chunks={n}/{len(chunks)} symbols={len(output)} failures={len(failures)}", flush=True)
    return output, failures


def append_snapshot(frame: pd.DataFrame, snap: dict[str, Any]) -> pd.DataFrame | None:
    prev_q = float(frame.iloc[-1]["close"])
    prev_raw = float(snap["prev_close"])
    if prev_q <= 0 or prev_raw <= 0:
        return None
    scale = prev_q / prev_raw
    row = {
        "date": TARGET_DATE,
        "open": float(snap["open"]) * scale, "high": float(snap["high"]) * scale,
        "low": float(snap["low"]) * scale, "close": float(snap["close"]) * scale,
        "volume": float(snap["volume"]), "turnover": float(snap["turnover"]),
    }
    return pd.concat([frame, pd.DataFrame([row])], ignore_index=True).tail(300).reset_index(drop=True)


def main() -> None:
    qlibscan.prepare_runtime()
    from src.analysis.golden_pit import score_golden_pit

    release_tag, qlib_url = qlibscan.prepare_qlib()
    calendar = qlibscan.read_calendar()
    if calendar[-1] != PREV_DATE:
        raise RuntimeError(f"Qlib last date is {calendar[-1]}, expected {PREV_DATE}")
    instruments = qlibscan.read_instruments()
    active = [(s, start, end) for s, start, end in instruments if qlibscan.valid_a_share(s) and start <= PREV_DATE]
    symbols = [x[0] for x in active]
    snapshot, quote_failures = full_snapshot(symbols)
    print(f"snapshot complete active={len(active)} quotes={len(snapshot)} failures={len(quote_failures)}", flush=True)

    qlibscan.TARGET_DATE = PREV_DATE
    prev_idx = len(calendar) - 1
    prelim: list[dict[str, Any]] = []
    scan_failures: list[dict[str, str]] = []
    scanned = 0
    for i, (symbol, _, _) in enumerate(active, 1):
        snap = snapshot.get(symbol)
        if not snap:
            continue
        try:
            frame = qlibscan.build_frame(symbol, calendar, prev_idx)
            if frame is None or len(frame) < 80:
                continue
            bars = append_snapshot(frame, snap)
            if bars is None:
                continue
            scanned += 1
            item = safe(score_golden_pit(symbol=symbol[2:], name=str(snap.get("name") or ""), bars=bars, period="daily"))
            item["exchange_symbol"] = symbol
            item["preliminary_source"] = "Qlib qfq through 2026-07-23 + Tencent 2026-07-24 batch close snapshot"
            score = int(item.get("total_score") or 0)
            exclusions = item.get("exclusions") or []
            raw_ret = float(snap["close"]) / float(snap["prev_close"]) - 1.0
            if item.get("grade") == "A" or (score >= 66 and len(exclusions) <= 1) or abs(raw_ret) > 0.115:
                item["snapshot_raw_return"] = raw_ret
                prelim.append(item)
        except Exception as exc:
            scan_failures.append({"symbol": symbol, "error": str(exc)[:500]})
        if i % 500 == 0:
            print(f"score progress={i}/{len(active)} scanned={scanned} verify_pool={len(prelim)}", flush=True)

    print(f"preliminary scanned={scanned} verify_pool={len(prelim)}", flush=True)
    verified: list[dict[str, Any]] = []
    verify_failures: list[dict[str, str]] = []

    def verify(item: dict[str, Any]) -> dict[str, Any]:
        bars, name = tencent_qfq(str(item["exchange_symbol"]))
        exact = safe(score_golden_pit(symbol=str(item.get("symbol") or ""), name=name or str(item.get("name") or ""), bars=bars, period="daily"))
        exact["exchange_symbol"] = item["exchange_symbol"]
        exact["data_source"] = "Tencent qfq daily exact verification"
        exact["preliminary_score"] = item.get("total_score")
        exact["as_of"] = TARGET_DATE
        exact["data_bar_count"] = len(bars)
        return exact

    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = {pool.submit(verify, x): x for x in prelim}
        for n, future in enumerate(as_completed(futures), 1):
            item = futures[future]
            try:
                verified.append(future.result())
            except Exception as exc:
                verify_failures.append({"symbol": str(item.get("exchange_symbol")), "error": str(exc)[:500]})
            if n % 50 == 0:
                print(f"verify progress={n}/{len(prelim)} failures={len(verify_failures)}", flush=True)

    verified.sort(key=lambda x: (x.get("grade") == "A", int(x.get("total_score") or 0)), reverse=True)
    grade_a = [x for x in verified if x.get("grade") == "A"]
    clean_a = [x for x in grade_a if not re.search(r"(?:\*?ST|退)", str(x.get("name") or ""), re.I)]
    near_a = [x for x in verified if x.get("grade") != "A" and x.get("is_near_a_grade")]

    output = {
        "schema": "venus8_full_market_technical_a_snapshot.v1", "target_date": TARGET_DATE,
        "method": "exact Venus8 score_golden_pit; final candidates re-fetched and verified on Tencent qfq daily history",
        "data_chain": {
            "history": f"Qlib release {release_tag} through {PREV_DATE}",
            "friday_snapshot": "Tencent batch close quotes 2026-07-24",
            "candidate_verification": "Tencent qfq daily through 2026-07-24", "qlib_asset": qlib_url,
        },
        "active_universe": len(active), "snapshot_coverage": len(snapshot), "preliminary_scanned": scanned,
        "verification_pool": len(prelim), "verified_count": len(verified),
        "quote_failure_count": len(quote_failures), "scan_failure_count": len(scan_failures), "verification_failure_count": len(verify_failures),
        "technical_a_count": len(grade_a), "technical_a_clean_count": len(clean_a), "near_a_count": len(near_a),
        "technical_a": grade_a, "technical_a_clean": clean_a, "near_a": near_a[:200],
        "failure_samples": (scan_failures + verify_failures)[:100],
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
        f"# Venus8 全市场技术A扫描（{TARGET_DATE}）", "", f"- 活跃A股：{len(active)}", f"- 周五快照覆盖：{len(snapshot)}",
        f"- 实际初筛：{scanned}", f"- 前复权复核：{len(verified)}", f"- 技术A：{len(grade_a)}", f"- 非ST/退市技术A：{len(clean_a)}", "",
        "|排名|代码|名称|总分|趋势|回调|动能|量能|确认|BOLL|收盘|涨跌%|",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        md.append(f"|{r['rank']}|{r['code']}|{r['name']}|{r['score']}|{r['trend']}|{r['pullback']}|{r['momentum']}|{r['volume']}|{r['price_confirmation']}|{r['boll']}|{r['close']}|{r['daily_change_pct']}|")
    (RESULT_DIR / "venus8_technical_a_20260724.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"active": len(active), "snapshot": len(snapshot), "scanned": scanned, "verify_pool": len(prelim), "verified": len(verified), "technical_a": len(grade_a), "clean_a": len(clean_a), "verify_failures": len(verify_failures)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
