#!/usr/bin/env bash
# c3 revert. Recapture was not a new pin. Capture restore-health on the live c2 serve. Do not reboot.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
OUT="$ROOT/evidence/opt-c3-mtp-fp8-block-scales-overlay"
mkdir -p "$OUT"
C2=60bc4af0b1d0839041b34cbf3cda6a34ed76c132

date -u +%Y-%m-%dT%H:%M:%SZ >"$OUT/started-at.txt"
printf '%s\n' "$C2" >"$OUT/last-kept-pin.txt"

{
  echo "last_kept_pin=$C2"
  echo "revert_head=$(git rev-parse HEAD)"
  echo "branch=$(git rev-parse --abbrev-ref HEAD)"
  echo "pin_paths=recipe.yaml run.sh README.md AGENTS.md docker tests kit bench_decode.py smoke_count.py smoke_thinking.py smoke_tools.py smoke_vision.py stop.sh"
  if git diff --exit-code "$C2" -- recipe.yaml run.sh README.md AGENTS.md docker tests kit bench_decode.py smoke_count.py smoke_thinking.py smoke_tools.py smoke_vision.py stop.sh; then
    echo "pin_diff=empty"
  else
    echo "pin_diff=NONEMPTY"
  fi
} >"$OUT/pin-diff-vs-c2.txt"

curl -sS -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8000/health | tee "$OUT/restore-health.code" >"$OUT/restore-health.txt"
curl -sS http://127.0.0.1:8000/v1/models >"$OUT/restore-models.json"
docker inspect qwen38-flash-next-nvfp4 --format '{{json .Config.Cmd}}' >"$OUT/restore-inspect-args.json"
docker inspect qwen38-flash-next-nvfp4 --format '{{json .Config.Env}}' >"$OUT/restore-inspect-env.json"
python3 - "$OUT/restore-inspect-env.json" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
env = json.loads(p.read_text())
out = []
for item in env:
    key = item.split("=", 1)[0]
    if key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        out.append(f"{key}=REDACTED")
    else:
        out.append(item)
p.write_text(json.dumps(out, separators=(",", ":")))
PY
docker inspect qwen38-flash-next-nvfp4 --format '{{.Config.Image}} {{.State.Status}} Started={{.State.StartedAt}}' >"$OUT/restore-inspect-image.txt"
docker inspect qwen38-flash-next-nvfp4 --format '{{json .Mounts}}' >"$OUT/restore-inspect-mounts.json"
docker inspect qwen38-flash-next-nvfp4 --format '{{.State.StartedAt}}' >"$OUT/restore-started-at.txt"
free -h >"$OUT/free-after-restore.txt"
ssh -o BatchMode=yes -o ConnectTimeout=5 spark2 'free -h' >"$OUT/free-after-restore-spark2.txt"
python3 - "$OUT" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
mounts = json.loads((out / "restore-inspect-mounts.json").read_text())
env = json.loads((out / "restore-inspect-env.json").read_text())
src = " ".join(m.get("Source", "") for m in mounts)
dst = " ".join(m.get("Destination", "") for m in mounts)
(out / "overlay-check.txt").write_text(
    "mtp_overlay_mounted=%s\nple_overlay_mounted=%s\nhas_VLLM_PLE_FP8_CHECKPOINT=%s\n"
    % (
        "modelopt.py" in src and "modelopt.py" in dst,
        "ple_layer.py" in src,
        any(x.startswith("VLLM_PLE_FP8_CHECKPOINT=") for x in env),
    )
)
PY

python3 -m unittest discover -s tests -q >"$OUT/unittest.txt" 2>&1
python3 kit/render.py --check >"$OUT/render-check.txt" 2>&1
VALIDATE_ONLY=1 ./run.sh >"$OUT/validate-only.txt" 2>&1

test "$(cat "$OUT/restore-health.code")" = 200
grep -q 'pin_diff=empty' "$OUT/pin-diff-vs-c2.txt"
grep -q 'mtp_overlay_mounted=True' "$OUT/overlay-check.txt"
