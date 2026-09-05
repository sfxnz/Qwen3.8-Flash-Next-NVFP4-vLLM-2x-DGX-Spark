#!/usr/bin/env python3
"""Compare c5 MTP k=2 SUMMARY against last kept pin c4."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
C4 = json.loads((ROOT / "evidence/opt-c4-native-ctx-262144/bench.txt.summary.json").read_text())
C5 = json.loads((ROOT / "evidence/opt-c5-mtp-k2/bench.txt.summary.json").read_text())


def idx(rows, phase, conc):
    for row in rows:
        if row["phase"] == phase and row["concurrency"] == conc:
            return row
    raise SystemExit(f"missing {phase} c={conc}")


print("phase conc metric c4 c5 delta")
for phase in ("prose", "structured"):
    for conc in (1, 2, 8):
        a = idx(C4, phase, conc)
        b = idx(C5, phase, conc)
        for key in (
            "median_decode_tok_s",
            "median_ttft_s",
            "n",
            "acceptance_len",
            "draft_acceptance_rate",
            "median_completion_tokens",
        ):
            da = a[key]
            db = b[key]
            delta = db - da
            print(f"{phase} {conc} {key} {da} {db} {delta}")
