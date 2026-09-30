# Session 8: FP8 dense (per_block) on top of K3, then promoted to the default (2026-09-30 / 10-01)

**Outcome:**
- **Owner decision (2026-10-01):** FP8 dense becomes the default for its decode gain. It is the base for `FP8_DENSE: per_block` in `recipe.yaml`; the rollback is `FP8_DENSE=none`.
- **Gates:** FP8 + K3 passed every quality and serving gate, including a new 240-prompt, 6-language paired judge. The T1-B top-1 gate is retired for precision levers on this model, because it is saturated (session 5).
- **Known cost:** long c=1 greedy generations fork run to run more often than on BF16 (see Determinism).
- **Process:** the operator first stopped the flip on that finding and left BF16 serving (`final/`). The flip was applied afterwards from `flip-draft-not-applied.patch` by the orchestrator, per the owner decision.

Raw data and telemetry are in `~/projects/data/qwen38-evals/runs/session8/`.

## Speed (default BF16+K3 → FP8+K3)

| Cell | BF16 | FP8 | Δ |
|---|---|---|---|
| frozen prose c1 tok/s | 44.9 | **56.9** | +27% |
| frozen structured c1 tok/s | 73.2 | **84.3** | +15% |
| frozen structured c2 aggregate | 138.7 | **156.2** | +13% |
| frozen prose c2 aggregate | – | 76.9 | – |
| ruler structured c1 ms/step (anchor-free) | 53.73 | 47.79 | −11.1% |
| ruler structured c8 ms/step | 68.94 | 63.15 | −8.4% |
| diverse prose / struct c1 ms/step | – | – | −12.3% / −12.8% |
| diverse prose / struct c8 ms/step | – | – | −7.9% / −6.7% (prose aggregate +7.3%) |

The ruler's prose c8 cell (one prompt × 8) is +4.4% on FP8, because the text changes and acceptance drops from 2.68 to 2.2–2.3. The 32-prompt diverse cells show no acceptance change. KV pool: 1,704,644 tokens vs 1,606,871.

## Engagement

Both ranks logged:
- the K3 `self-test passed` and `dispatch: ACTIVE` lines;
- `qwen38-fp8-dense: mode=per_block`;
- the Marlin `fp8_block_w8a8` override.

Rank 0 also printed `Selected MarlinFP8ScaledMMLinearKernel`. Rank 1 printed no `Selected` line, but did show Marlin's FP8 warning.

## Gates (FP8 + K3)

| Gate | Result |
|---|---|
| smokes / T0 (30 bursts) / preemptions | 4/4 / 0 gated / 0 |
| T1-G self (10 × 5, 256 tokens) | 1.0 distinct, early divergence 0 |
| T1-D vs BF16 (40 × 1024) | median first divergence 17.0 (gate ≥ 14.25), KL 0.0041, 9/40 identical: PASS |
| paired T2 vs BF16 | GSM8K-250 −1.2 pp (3 lost / 0 gained, p 0.25), IFEval-120 +1.7 pp, tools 60/60, JSON 30/30: PASS |
| T3 needles to 64k / V | 24/24 / 16/16 |
| T4 full GSM8K, thinking on (session 6, same FP8 path; K3 is exact) | 97.65% vs 97.80%, p 0.75 (equal to the BF16 A/A) |
| 15-min soak (8 × 4096-token streams, 32k prefills) | 88 requests, 0 errors, 0 preemptions; spark1 ≥ 12.2 GiB available |

## Multilingual paired judge (`quality/ml_judge.py`, 240 prompts)

Method: the FP8 serve judged its own outputs against the BF16 outputs, blind, in both A/B orders. Only verdicts that agreed in both orders count.

| Lang | FP8 W / T / L | inconsistent | p (FP8 worse) | lang-ID fails BF16 / FP8 |
|---|---|---|---|---|
| en | 4 / 17 / 5 | 11 | 0.50 | 0 / 0 |
| de | 5 / 13 / 6 | 14 | 0.50 | 0 / 0 |
| es | 9 / 5 / 7 | 16 | 0.77 | 0 / 0 |
| fr | 7 / 11 / 4 | 17 | 0.89 | 0 / 0 |
| ja | 7 / 7 / 7 | 16 | 0.60 | 0 / 0 |
| zh | 7 / 8 / 7 | 15 | 0.60 | 0 / 0 |
| **all** | **39 / 61 / 36** | 89 | 0.68 | 0 / 0 |

- **Result: PASS.** No language is significantly worse, and the repeat rate (0.031) and length ratio (1.008) are unchanged.
- **Judge bias:** the judge favours slot B. Running both orders cancels this, but leaves ~40% of pairs inconsistent (mostly long answers cut at 512 tokens).

## Determinism (the cost)

`near1600`: a 1578-token prompt, greedy, `ignore_eos`, 4096 generated tokens, c=1 on an idle serve (`near1600-determinism.txt`, `near1600_reps.py`).

| Build | runs | distinct | modal share | first fork (generated token) |
|---|---|---|---|---|
| BF16 + K3 | 13 | 2 | 12/13 | 501 |
| FP8 + K3 | 9 | 5 | 3/9 | 2432, then 3293 ×5 |
| FP8, K3 off | 5 | 3 | 3/5 | 3293 ×2 |

- **The cause is FP8, not K3:** both FP8 builds give the same set of outputs.
- **Short outputs are exact** (T1-G, identity prompts), and a rerun after the soak matches the pre-soak output.
- **Not the cause:** Marlin's atomic-add reduction, which is off by default (`VLLM_MARLIN_USE_ATOMIC_ADD` unset).
- **Open:** the source (call-to-call variation somewhere in the FP8 path, visible only at near-ties).
- **Scope:** the BF16 default is not perfectly pinned on long runs either (1 fork in 13).
