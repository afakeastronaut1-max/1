from __future__ import annotations

import importlib
import json
import time
from pathlib import Path

import scan_repair as repair_helpers
import scan_snapshot as snapshot_module

scan = importlib.reload(snapshot_module)
_original_snapshot = scan.full_snapshot
_original_qfq = scan.tencent_qfq


def repaired_snapshot(symbols: list[str]):
    output, failed = _original_snapshot(symbols)
    print(f"coverage repair initial gaps={len(failed)}", flush=True)
    remaining: list[str] = []
    for i in range(0, len(failed), 5):
        chunk = failed[i:i + 5]
        try:
            output.update(scan.fetch_quote_chunk(chunk))
        except Exception:
            remaining.extend(chunk)
        time.sleep(0.08)
    final_remaining: list[str] = []
    for n, symbol in enumerate(remaining, 1):
        try:
            output.update(scan.fetch_quote_chunk([symbol]))
        except Exception:
            final_remaining.append(symbol)
        if n % 50 == 0:
            print(f"coverage individual={n}/{len(remaining)} unresolved={len(final_remaining)}", flush=True)
        time.sleep(0.04)
    print(f"coverage repair recovered={len(failed)-len(final_remaining)} unresolved={len(final_remaining)}", flush=True)
    return output, final_remaining


def repaired_qfq(exchange_symbol: str):
    if exchange_symbol.startswith("BJ"):
        try:
            return repair_helpers.eastmoney_qfq(exchange_symbol)
        except Exception as first:
            try:
                return _original_qfq(exchange_symbol)
            except Exception as second:
                raise RuntimeError(f"BJ verification failed: eastmoney={first}; tencent={second}")
    try:
        return _original_qfq(exchange_symbol)
    except Exception as first:
        try:
            return repair_helpers.eastmoney_qfq(exchange_symbol)
        except Exception as second:
            raise RuntimeError(f"qfq verification failed: tencent={first}; eastmoney={second}")


scan.full_snapshot = repaired_snapshot
scan.tencent_qfq = repaired_qfq

if __name__ == "__main__":
    scan.main()
    path = Path("results/venus8_technical_a_20260724.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema"] = "venus8_full_market_technical_a_repaired.v2"
    payload["coverage_repair"] = "Initial failed Tencent batches retried in groups of 5 and individually; qfq verification falls back between Tencent and Eastmoney, with BJ preferring Eastmoney."
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
