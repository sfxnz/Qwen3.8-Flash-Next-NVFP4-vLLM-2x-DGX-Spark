#!/usr/bin/env bash
# Post-boot probes for c3. Run only after /health is 200.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
OUT="$ROOT/evidence/opt-c3-mtp-fp8-block-scales-overlay"
mkdir -p "$OUT"

curl -sS --max-time 5 -o "$OUT/health.txt" -w '%{http_code}\n' http://127.0.0.1:8000/health | tee "$OUT/health.code"
curl -sS --max-time 5 http://127.0.0.1:8000/v1/models > "$OUT/models.json"
docker inspect qwen38-flash-next-nvfp4 --format '{{.Config.Image}} {{.Created}} {{.State.StartedAt}}' > "$OUT/inspect-image.txt"
docker inspect qwen38-flash-next-nvfp4 --format '{{json .Config.Env}}' > "$OUT/inspect-env.json"
docker inspect qwen38-flash-next-nvfp4 --format '{{json .HostConfig.Binds}}' > "$OUT/inspect-mounts.json"
docker inspect qwen38-flash-next-nvfp4 --format '{{json .Args}}' > "$OUT/inspect-args.json"
docker logs qwen38-flash-next-nvfp4 2>&1 | grep -E 'Unknown vLLM|quantization=|FP8 MoE|KV cache size|Graph capturing finished|Qwen4ExpPLE|w2_weight_scale|v0.28.1|AttributeError|mtp.layers.48' | head -80 > "$OUT/engine-needles.txt" || true
python3 - <<'PY'
import json
from pathlib import Path
out = Path("evidence/opt-c3-mtp-fp8-block-scales-overlay")
env_path = out / "inspect-env.json"
env = json.loads(env_path.read_text())
redacted = []
for item in env:
    if item.startswith(("HF_TOKEN=", "HUGGING_FACE_HUB_TOKEN=")):
        redacted.append(item.split("=", 1)[0] + "=REDACTED")
    else:
        redacted.append(item)
env_path.write_text(json.dumps(redacted))
env = redacted
mounts = json.loads((out / "inspect-mounts.json").read_text())
needles = (out / "engine-needles.txt").read_text()
has_mtp_mount = any("modelopt.py" in m for m in mounts)
mixed = "quantization=modelopt_mixed" in needles
attr = "AttributeError" in needles and "w2_weight_scale_inv" in needles
fp8_moe = "FP8 MoE block scales" in needles
(out / "overlay-check.txt").write_text(
    f"mtp_overlay_mounted={has_mtp_mount}\n"
    f"quantization_modelopt_mixed={mixed}\n"
    f"mtp_attr_error={attr}\n"
    f"fp8_moe_block_scales={fp8_moe}\n"
    f"has_VLLM_PLE_FP8_CHECKPOINT={any(x.startswith('VLLM_PLE_FP8_CHECKPOINT=') for x in env)}\n"
)
print(f"mount={has_mtp_mount} mixed={mixed} attr={attr} fp8_moe={fp8_moe}")
PY
free -h > "$OUT/free-before.txt"
ssh -o BatchMode=yes -o ConnectTimeout=8 spark2 'free -h' > "$OUT/free-before-spark2.txt"

set +e
python3 smoke_thinking.py > "$OUT/thinking-off.json" 2>"$OUT/thinking-off.err"
echo $? > "$OUT/thinking-off.exit"
python3 smoke_tools.py > "$OUT/tools.json" 2>"$OUT/tools.err"
echo $? > "$OUT/tools.exit"
python3 smoke_vision.py > "$OUT/vision.json" 2>"$OUT/vision.err"
echo $? > "$OUT/vision.exit"
python3 smoke_count.py > "$OUT/count.json" 2>"$OUT/count.err"
echo $? > "$OUT/count.exit"
python3 bench_decode.py --runs 3 --concurrency 1 2 8 --max-tokens 200 > "$OUT/bench.txt" 2>"$OUT/bench.err"
echo $? > "$OUT/bench.exit"
set -e
python3 - <<'PY'
import re
from pathlib import Path
text = Path("evidence/opt-c3-mtp-fp8-block-scales-overlay/bench.txt").read_text()
m = re.search(r"SUMMARY\s+(\[.*\])", text, re.S)
if not m:
    raise SystemExit("no SUMMARY in bench.txt")
Path("evidence/opt-c3-mtp-fp8-block-scales-overlay/bench.txt.summary.json").write_text(m.group(1) + "\n")
print(m.group(1)[:400])
PY
curl -sS --max-time 5 -o "$OUT/health-after.txt" -w '%{http_code}\n' http://127.0.0.1:8000/health > "$OUT/health-after.code"
free -h > "$OUT/free-after-bench.txt"
ssh -o BatchMode=yes -o ConnectTimeout=8 spark2 'free -h' > "$OUT/free-after-bench-spark2.txt"
echo probes-done
