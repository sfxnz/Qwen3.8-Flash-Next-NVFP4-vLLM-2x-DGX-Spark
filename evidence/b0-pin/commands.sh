#!/usr/bin/env bash
# B0: pin WITHOUT the F01 stride overlay (diagnostic), session 1
DIAGNOSTIC=1 OVERLAYS="docker/ple_layer.py docker/modelopt.py" \
  EXTRA_ARGS='--per-request-spec-decode-metrics summary --profiler-config {"profiler":"torch","torch_profiler_dir":"/tmp/qwen38-traces"}' \
  ./run.sh > evidence/b0-pin/run.log 2>&1 &
# poll /health, then (steps.sh is the operator runner copied here):
evidence/b0-pin/steps.sh b0-pin b0 gate ruler qaa profile receipts
