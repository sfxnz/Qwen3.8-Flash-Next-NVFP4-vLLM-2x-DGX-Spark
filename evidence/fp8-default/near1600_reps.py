#!/usr/bin/env python3
"""Repeat one longid prompt N times at c=1 (greedy, ignore_eos, 4096) and report distinct outputs + first diffs."""
import hashlib, json, sys
from pathlib import Path
ROOT = Path("/home/sfxnz/projects/ai-lab/recipes/Qwen3.8-Flash-Next-NVFP4-vLLM-2x-DGX-Spark-plan")
sys.path.insert(0, str(ROOT / "quality")); sys.path.insert(0, str(ROOT / "evidence/k3"))
from qlib import OFF, Client
import soak
pid, n, out = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3])
prompt = dict(soak.LONGID)[pid]
c = Client("http://127.0.0.1:8000", timeout=1800)
rows = []
for i in range(n):
    r = c.chat(prompt, max_tokens=4096, temperature=0, ignore_eos=True, return_token_ids=True, **OFF)
    ids = r["token_ids"]; rows.append(ids)
    with out.open("a") as f: f.write(json.dumps({"id": pid, "rep": i, "prompt_tokens": r["usage"]["prompt_tokens"], "token_ids": ids}) + "\n")
    print(i, hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:12], flush=True)
def fd(a, b): return next((k for k, (x, y) in enumerate(zip(a, b)) if x != y), None)
print("distinct", len({json.dumps(x) for x in rows}), "first_diffs_vs_rep0", [fd(rows[0], x) for x in rows])
