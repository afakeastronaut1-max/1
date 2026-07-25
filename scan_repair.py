from __future__ import annotations

import json
import random
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

import scan_snapshot as scan

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/140 Safari/537.36",
    "Accept": "application/json,text/plain,*/*",
    "Referer": "https://quote.eastmoney.com/",
}


def repaired_snapshot(symbols: list[str]):
    output, failed = scan.full_snapshot(symbols)
    if not failed:
        return output, failed
    print(f"repairing quote gaps initial={len(failed)}", flush=True)
    remaining: list[str] = []
    for i in range(0, len(failed), 5):
        chunk = failed[i:i + 5]
        try:
            output.update(scan.fetch_quote_chunk(chunk))
        except Exception:
            remaining.extend(chunk)
        time.sleep(0.08)
    if remaining:
        print(f"individual quote repair remaining={len(remaining)}", flush=True)
    final_remaining: list[str] = []
    for n, symbol in enumerate(remaining, 1):
        try:
            output.update(scan.fetch_quote_chunk([symbol]))
        except Exception:
            final_remaining.append(symbol)
        if n % 50 == 0:
            print(f"individual quote repair={n}/{len(remaining)} unresolved={len(final_remaining)}", flush=True)
        time.sleep(0.04)
    print(f"quote repair complete recovered={len(failed)-len(final_remaining)} unresolved={len(final_remaining)}", flush=True)
    return output, final_remaining


def eastmoney_qfq(exchange_symbol: str):
    code = exchange_symbol[2:]
    market = "1" if exchange_symbol.startswith("SH") else "0"
    params = {
        "secid": f"{market}.{code}", "klt": "101", "fqt": "1",
        "beg": "20240101", "end": "20260724", "lmt": "500",
        "fields1": "f1,f2,f3,f4,f5,f6", "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "ut": "fa5fd1943c7b386f172d6893dbfba10b",
    }
    hosts = ["push2his.eastmoney.com", "72.push2his.eastmoney.com", "7.push2his.eastmoney.com", "33.push2his.eastmoney.com"]
    last = None
    for attempt in range(8):
        host = hosts[attempt % len(hosts)]
        try:
            url = "https://" + host + "/api/qt/stock/kline/get?" + urllib.parse.urlencode(params)
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=25) as resp:
                payload = json.load(resp)
            node = (payload or {}).get("data") or {}
            lines = node.get("klines") or []
            name = str(node.get("name") or "")
            rows = []
            for line in lines:
                p = str(line).split(",")
                if len(p) >= 7:
                    rows.append({
                        "date": p[0], "open": float(p[1]), "close": float(p[2]),
                        "high": float(p[3]), "low": float(p[4]), "volume": float(p[5]),
                        "turnover": float(p[6]) if p[6] not in {"", "-"} else np.nan,
                    })
            df = pd.DataFrame(rows)
            if len(df) >= 80 and str(df.iloc[-1]["date"]) == "2026-07-24":
                return df.tail(300).reset_index(drop=True), name
            raise RuntimeError(f"incomplete eastmoney bars={len(df)}")
        except Exception as exc:
            last = exc
            time.sleep(0.45 * (attempt + 1) + random.random() * 0.15)
    raise RuntimeError(f"Eastmoney qfq failed: {last}")


_original_qfq = scan.tencent_qfq


def repaired_qfq(exchange_symbol: str):
    if exchange_symbol.startswith("BJ"):
        try:
            return eastmoney_qfq(exchange_symbol)
        except Exception as first:
            try:
                return _original_qfq(exchange_symbol)
            except Exception as second:
                raise RuntimeError(f"BJ verification failed: eastmoney={first}; tencent={second}")
    try:
        return _original_qfq(exchange_symbol)
    except Exception as first:
        try:
            return eastmoney_qfq(exchange_symbol)
        except Exception as second:
            raise RuntimeError(f"multi-source qfq failed: tencent={first}; eastmoney={second}")


scan.full_snapshot = repaired_snapshot
scan.tencent_qfq = repaired_qfq

if __name__ == "__main__":
    scan.main()
    result_path = Path("results/venus8_technical_a_20260724.json")
    if result_path.exists():
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        payload["schema"] = "venus8_full_market_technical_a_repaired.v1"
        payload["coverage_repair"] = "Failed Tencent batches retried in groups of 5 and individually; qfq verification falls back across Tencent/Eastmoney, with BJ preferring Eastmoney."
        result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
