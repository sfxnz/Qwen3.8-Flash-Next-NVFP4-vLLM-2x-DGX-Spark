#!/usr/bin/env bash
# B1: branch default at 6aefbfc = pin + OVERLAYS_PIN (ple_layer.py, modelopt.py, ple_ops.py stride fix),
# UTIL 0.76, --oom-score-adj 1000, memguard. A first boot at UTIL 0.80 was aborted before any traffic (run-aborted-util080.log).
EXTRA_ARGS='--per-request-spec-decode-metrics summary' ./run.sh > evidence/b1-pin-stride/run.log 2>&1 &
# poll /health, then:
evidence/b1-pin-stride/steps.sh b1-pin-stride b1 gate diverse sampled qaa t2 t3 receipts
python3 ruler_steps.py --anchor-ms 0 0 --json evidence/b1-pin-stride/ruler-noanchor.json   # anchor [59,61] voided the gate ruler; relative boot-min rule kept
