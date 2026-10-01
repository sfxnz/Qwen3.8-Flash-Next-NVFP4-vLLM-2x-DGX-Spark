# Session 7: K3, the GDN MTP decode with lazy state commit (2026-09-30)

**Result: K3 is kept as the default** (`GDN_LAZY=1` in `recipe.yaml`, rendered). It is bit-exact against the stock v0.30 CUDA kernel. At c=8 the ruler's decode step is 7–8% faster; at c=1 it is flat. The confirmation boot `final/` is the rendered default: it passed the gate pack and is serving. Rollback: `GDN_LAZY=0`.

Where things are:
- Commands: `commands.sh`.
- Runners: `harness.sh`, `boot.sh`, `steps.sh`, `soak.py`, `table.py`.
- Raw data: `~/projects/data/qwen38-evals/runs/session7/<arm>/`.

## Step 1: standalone harness (spark1, serve down)

As built, the self-test failed on 1 output element. Two GPU-only bugs were fixed in `docker/v030/gdn_lazy.py` (`K3.md` §3a):
1. **Reduction layout.** `tl.reshape(delta, [BV,1])` produced a layout that turned the `q·h` output dot into a per-thread sum instead of the stock `shfl.bfly` butterfly. That caused 1-ulp output misses in about 1 in 10⁴ outputs and 2154 register spills. The fix is `tl.expand_dims(delta, 1)`: now bitwise, with 0 spills.
2. **Epilogue race at 8 warps.** A warp could store `y` over a raw `o` that another warp had not read yet, giving misses of up to 1000 ulp. Fixed with a `debug_barrier`; the same guard is on materialize's t=0 store.

The chosen tunable is BV=64 with 8 warps (BV=32 with 4 warps was +0.07 ms/step at c1).

| Gate | Result |
|---|---|
| selftest | bitwise |
| exact c1 / c8, 256 steps | A0 / A0 (output and committed state n_diff 0) |
| split-tree cross-check | A0 |
| sim block 64 / 1600 | bitwise (117 / 17 crossings, 99 / 16 checkpoints) |
| stress, seeds 1–3: exact c1/2/5/8 × 1024, sim64 × 600 | all pass |
| bench, 36 layers, CUDA graph ABAB×4 | c1 38.77 → 32.15 µs/layer (**−0.24 ms/step**); c8 299.0 → 134.8 µs/layer (**−5.91 ms/step**) |
| writes/layer/request (L2 write sectors; GB10 has no `dram__` counters) | stock 6.33 MB → lazy 1.70 MB (c1) / 1.68 MB (c8); gate ≤ 1.7 (c1 is borderline) |
| CPU interpreter tests | 17 OK |

## Step 2: serve A/B

Both ranks logged `K3 gdn_lazy self-test passed` and `dispatch: ACTIVE`.

| Gate | ref | k3 | final |
|---|---|---|---|
| T0 (30 bursts) | 0 gated | 0 gated | 0 gated |
| smokes | 4/4 | 4/4 | 4/4 |
| T1-G greedy identity (10 × 5) | – | identical 10/10 | identical 10/10 |
| 4096-token greedy crossing the 1600/3200 blocks (3 prompts) | – | identical, before and after the soak | identical |
| T1 / T1-long NLL | – | Δ 0, top-1 1.000 (prefill path; decode is covered by the rows above) | – |
| T3 4k/16k/32k, vision | – | 18/18, 16/16 | 18/18, 16/16 |
| preemptions | 0 | 0 | 0 |

Ruler (anchored 53.8–55.8, ms/step):

| Arm | prose c1 | prose c2 | prose c8 | struct c1 | struct c2 | struct c8 |
|---|---|---|---|---|---|---|
| ref | 52.88 | 61.27 | 80.52 | 54.73 | 58.57 | 75.02 |
| k3 | 52.88 | 60.15 | **74.73 (−7.2%)** | 54.32 | 57.43 | **69.17 (−7.8%)** |
| final | 52.52 | 60.11 | **74.38 (−7.6%)** | 53.96 | 57.69 | **69.24 (−7.7%)** |

`bench_diverse`, ref → final (ms/step · aggregate tok/s):

| Cell | ref | final |
|---|---|---|
| prose c8 | 102.83 · 161.2 | 96.68 · 168.3 (−6.0% ms/step) |
| struct c8 | 106.24 · 189.4 | 101.60 · 195.3 (−4.4% ms/step) |

c1 cells are flat. Frozen bench_decode on final: prose c1 44.8, struct c1 73.8, struct c2 aggregate 138.7 tok/s.

**20-minute soak:**
- **Load:** 8 × 4096-token streams (greedy and T=0.7) crossing the GDN blocks, a short-request worker, and a 32k prefill every 4 minutes.
- **Result:** 106 requests with **0 errors** and 0 preemptions; the long-greedy identity held before and after.
- **Memory:** spark1 stayed at ≥ 13.1 GiB available and ≤ 7.1 GiB swap; memguard never tripped.

## Step 3: wiring

`GDN_LAZY: 1` in `recipe.yaml`. On the v0.30 digest, `run.sh`:
- swaps `gdn_attn.py` for `gdn_lazy_attn.py`, which carries the S1.5 fresh-prefill hunk;
- adds `gdn_lazy_linear_attn.py`;
- sets `VLLM_QWEN38_GDN_LAZY=1` on both ranks.

The pin rollback ignores it, with a note. `OVERLAYS=none` and custom lists without the GDN overlay mount nothing. Tests: `tests/test_gdn_lazy.py::RecipeWiring`. 295 unit tests pass, and render and `VALIDATE_ONLY` pass.

## Open

- K3 runs only on pure spec-decode steps. Mixed steps, and the ~0.5% of eager rows near 1600-token blocks, use the stock path.
- It stays off, failing closed, with a BF16 SSM state or `mamba_cache_mode='all'`.
- `K3.md` §7: stock v0.30 reads column 0 for a 0-draft row after a spec step with a > 1. K3 reproduces this exactly.
- The README's measured decode rows keep the session-4 conditions (c1/c2 move by ≤ 1%).
