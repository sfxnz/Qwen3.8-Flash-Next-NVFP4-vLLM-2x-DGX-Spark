# S1.5 c=1 determinism: summary (session 2, 2026-09-29)

**Resolved.** The c=1 non-reproducibility had three independent causes, each fixed by a small header-stamped v0.30 overlay. With all three on U1, every probe is bit-exact run to run: prompt_logprobs at 16–4096 tokens, greedy at 1–16k context (64 and 256 new tokens), T1 A/A and T1-G. Cost: about +1.4 ms/step (+2.3%) at c1, neutral at c8.

## Culprits and fixes

| # | Cause | Symptom | Fix |
|---|---|---|---|
| 1 | FlashInfer CUTLASS NVFP4 MoE fused finalize. vLLM never passes `use_fused_finalize`, so FlashInfer 0.6.18 defaults it to True and the top-10 expert sum runs as BF16 atomics ("not deterministic run-to-run"). NVFP4 activation re-quantisation and routing over 48 layers amplify the ulp noise to nats. | Every position differs from the first scored token. T1 A/A about 93% top-1, KL 0.036. Greedy gives 5/5 distinct outputs. Same on the pin and on v0.30. | `docker/v030/flashinfer_cutlass_moe.py`: `use_fused_finalize=not _MOE_DETERMINISTIC` (`VLLM_QWEN38_MOE_DETERMINISTIC`, default 1; 0 restores stock) |
| 2 | GDN metadata (`gdn_attn.py:251`) splits with the default `treat_short_extends_as_decodes=True`, so a fresh 1-token prompt runs the GDN decode kernel on a leftover state slot. | A 1-token prompt's first token varies by up to 3.2 nats. | `docker/v030/gdn_attn.py`: `treat_short_extends_as_decodes=m.is_prefilling is None` |
| 3 | QSA `persistent_topk` (F23): run-dependent output order, and a run-dependent set when logits tie. | Differences start at position 2052. A tie residual remained at 4096 tokens from position 3931. | `docker/v030/qsa_indexer.py`: full rows are sorted after top-k. On prefill only, full rows are re-selected as everything above the k-th value plus the lowest-index ties. |

Generators: `docker/v030/apply_{moe_finalize,gdn_fresh_prefill,qsa_topk_order}_overlay.py`. Tests: `tests/test_{moe_finalize,gdn_fresh_prefill,qsa_topk_order}_overlay.py`.

## Harness (`repro.py`, N=5, fresh cache_salt, idle serve)

prompt_logprobs. Each cell shows fingerprints / max |Δ| / first differing position.

| Length | U1 | Fix 1 only | Final |
|---|---|---|---|
| 16 | 5 / 2.75 / 1 | exact | exact |
| 200 | 5 / 3.50 / 1 | exact | exact |
| 2048 | 5 / 5.06 / 1 | exact | exact |
| 3072 | 5 / 6.22 / 1 | from 2052 | exact |
| 4096 | 5 / 5.89 / 1 | from 2052 | exact |

Greedy. Each cell shows distinct outputs / first divergence.

| Prompt length | U1 | Final |
|---|---|---|
| 1 | 5 / 0 | 1 |
| 200 | 5 / 6–29 | 1 |
| 8192 | 3 / 7–27 | 1 |
| 16384 | 5 / 30–57 | 1 |

History test (A three times, then filler→A): U1 gives 7 fingerprints, the fixed build 1.

## A/A floors

| Gate | U1 | Final |
|---|---|---|
| T1 A/A top-1 / KL | 93.00% / 0.036 | 100% / 0 |
| T1 target-logprob bit-exact | 0.23% | 100% |
| T1-G distinct / early_div | 5.0 / 0.52 | 1.0 / 0.0 |

**Consequence for gating:** the same-build floor is now exact. Kernel-reorder levers are expected to move logits, and the model amplifies ulp noise into nats. Class-A gates therefore use NLL delta and paired task accuracy (T2), not a top-1 threshold.

## Speed (`ruler_steps --anchor-ms 0 0`, ms/step)

| Cell | U1 | Fix 1 only | All fixes, metrics flag off | All fixes, flag on |
|---|---|---|---|---|
| structured c1 | 60.69 | 62.23 | 61.34 | 62.07 |
| structured c8 | 80.48 | 79.99 | 79.29 | 80.16 |
| prose c1 | 58.93 | 60.53 | 59.48 | 59.52 |
| prose c8 | 91.98 | 86.13 | 84.72 | 85.74 |

- Prose cells are not comparable across the fix: output is now deterministic, so acceptance is pinned (2.37 at c1, 2.56 at c8).
- `--per-request-spec-decode-metrics summary` costs 0.7–0.9 ms/step at c1 and about 1 ms at c8. Drop it from served defaults.

## Not run

The generic bisect arms were not needed. 8192-token prompt_logprobs was not run (about 8 GB transient). Decode-path QSA ties remain possible in theory on very long decodes; none showed at 16k × 256 tokens. There were no memguard trips.
