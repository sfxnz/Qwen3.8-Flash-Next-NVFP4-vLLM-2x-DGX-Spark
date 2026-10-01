# MTP draft-head overlay (L1a, L1b, L1b′; v0.30.0)

`docker/v030/mtp.py` is stock v0.30.0 `vllm/models/qwen4_exp/nvidia/mtp.py` (sha256 `928c4500…eb88`,
image `sha256:4864d466…4dfb56`) plus the hunks in `docker/v030/apply_mtp_overlay.py`. Do not hand-edit it.
Regenerate it with `python3 docker/v030/apply_mtp_overlay.py [--src <stock mtp.py>]`. `--check` exits 1
when the file is stale. Mount it on **both** ranks and only on the v0.30 image
(`OVERLAYS_V030="docker/v030/mtp.py …"`). The generator refuses the pinned nightly's file.

## Knobs

| Feature | How to turn it on (both ranks) | Default |
|---|---|---|
| **L1a** local-argmax drafting | `SPEC_CONFIG` `"use_local_argmax_reduction": true` | overlay makes it bootable; the flag stays off |
| **L1b** reduced-vocab head | `EXTRA_ENV="VLLM_QWEN38_DRAFT_VOCAB=/cache/huggingface/qwen38-draft-vocab/draft_vocab_131072.json"` + L1a | off |
| **L1b′** draft-only FP8 head | `EXTRA_ENV="VLLM_QWEN38_DRAFT_HEAD_FP8=1"` (`auto`, `marlin`, `w8a8`) | off |
| L1b + L1b′ | both envs | the V′ rows are stored FP8 only |

- **Vocab file.** Copy `tools/vocab/draft_vocab_<V>.json` into `$HF_CACHE/qwen38-draft-vocab/` on **both** Sparks.
  The container sees `$HF_CACHE` as `/cache/huggingface`. L1b needs L1a and greedy drafting. It refuses
  `draft_sample_method=probabilistic`.
- **FP8 values.** `1`/`auto` tries Marlin W8A16 first, then CUTLASS W8A8. `marlin` and `w8a8` each force
  one kernel. `0`, `off` and empty all mean off. Any other value logs a warning and stays off.
- **Unknown env.** vLLM logs `Unknown vLLM environment variable detected` for both names. That is a
  warning only, unless the serve sets `--fail-on-environ-validation`.

## What L1b′ does

- **Quantisation at load.** The drafter quantises its own copy of the lm_head rows:
  - E4M3 with one scale per row.
  - The scale is rounded to BF16 and then kept in fp32. That way Marlin (BF16 scales) and CUTLASS
    (fp32 scales) dequantise identically.
  - With L1b on, it quantises this rank's V′/2 rows. Otherwise it quantises this rank's whole shard of
    [124160, 2560].
- **The target head is untouched.** The target keeps its BF16 lm_head. `load_eagle_model` still drops
  the drafter's BF16 copy when it rebinds `lm_head`. Only the FP8 copy is extra.
- **Kernels** (all v0.30 in-image ops):
  - `marlin` (W8A16): `prepare_fp8_layer_for_marlin` + `apply_fp8_marlin_linear`. Weights are FP8 and
    activations stay BF16.
  - `w8a8`: `scaled_fp8_quant(use_per_token_if_dynamic=True)` + `cutlass_scaled_mm` with the per-row
    weight scale.
- **Load-time self-check.** Each kernel runs 4 random rows against the BF16 rows. A kernel that fails
  to build or has max relative error > 0.1 is skipped. If both kernels fail, L1b′ turns off (see
  Fallbacks).
- **Use.** `get_top_tokens` (L1a): FP8 logits → local argmax → [M, 2] all-gather. With the full-vocab
  FP8 head, `compute_logits` also returns all-gathered FP8 logits. This covers probabilistic drafting
  (the plan's D5 case for L1b′) and greedy drafting without L1a.
- **Fallbacks.** These are agreed across TP ranks through a MIN all-reduce on the CPU group, so one rank
  missing an env cannot hang the load:
  - FP8 fails with L1b on → L1b stays on in BF16.
  - FP8 fails without L1b → the stock head, and the drafter's BF16 shard is not retained.
  - A logits scale, a soft-cap, a non-default `head_dtype`, tied embeddings, PP>1, or a padded vocab shard
    also refuses (full-vocab case).
- **Look in `run.log`** for `Draft head (L1b|L1b'|L1b+L1b'): N rows … FP8 marlin|w8a8 …` or a
  `… disabled (reason)` warning.

## Memory, per rank

| Head | Rows/rank | Bytes |
|---|---|---|
| stock (shared with target) | 124160 BF16 | 0 extra |
| L1b V′=131072 | 65536 BF16 | +320 MiB |
| L1b′ full vocab | 124160 FP8 + fp32 scale | +303 MiB (≈0.30 GiB, plan ledger) |
| L1b + L1b′ V′=131072 | 65536 FP8 | +160 MiB |

The buffers are built inside `load_model`, before memory profiling. Load also has a transient peak: an
fp32 chunk of 8192 rows plus Marlin's repack copy.

## Draft vocabulary (tools/vocab, v2)

`tools/draft_subvocab_coverage.py bigcorpus` tokenises 176M tokens from public data under
`~/projects/data/qwen38-vocab-corpus`: chat, wiki, code, math and tool data in en/zh/ja/de/es.
`analyze2` ranks ids from those counts only. It then scores the ranking on all 54,772 greedy completion
tokens from the served model, so the whole set is held out. Full table: `tools/vocab/coverage_report_v2.txt`.

| V′ | ranking | min-domain coverage (worst domain) | prose |
|---|---|---|---|
| 65536 | corpus_max | 0.891 (zh) | 0.919 |
| 98304 | corpus_max | 0.955 (zh) | 0.957 |
| **131072** | **corpus_sum** | **0.987 (prose)** | 0.987 |
| 163840 | blend | 0.9945 (de) | 0.999 |

BPE merge order (lowest ids first) is poor: 0.65 at 131072, because CJK tokens sit at high ids. No V′ ≤
131072 reaches the ≥ 0.99 per-domain gate. The misses are topical long-tail words: 蓝光, `crc`, `OrderedDict`.

## Expected gains (model, not measured)

These are modelled, not measured. Base: the plan's prose c=1 step of 56.9 ms at acceptance 2.455
(43.1 tok/s), with k=3. The head is read at the S1.1 draft-head rate of 230 GB/s, and savings are
realised at 0.85. Acceptance scales as `1 + Σ(α·c)^i`. The FP8 rate is an **assumption** until the S1.1
`fp8_sweep` Marlin row lands.

| Config | MiB/rank | Saved ms/step | prose tok/s | worst-domain tok/s |
|---|---|---|---|---|
| stock | 606 | 0 | 43.1 | 43.1 |
| L1b′ full vocab | 303 | 3.5 | 46.0 (+6.6%) | 46.0 (+6.6%) |
| L1b 131072 BF16 | 320 | 3.3 | 45.2 (+4.8%) | 45.2 (+4.8%) |
| **L1b 131072 + L1b′** | **160** | **5.2** | **46.8 (+8.6%)** | **46.8 (+8.6%)** |
| L1b 163840 + L1b′ | 200 | 4.7 | 47.0 (+8.9%) | 46.8 (+8.4%) |
| L1b 65536 BF16 (plan's old pick) | 160 | 5.2 | 43.7 (+1.2%) | 42.4 (−1.7%) |

**Recommendation.**
- Use **V′ = 131072** with L1b′ (`VLLM_QWEN38_DRAFT_VOCAB=…/draft_vocab_131072.json`,
  `VLLM_QWEN38_DRAFT_HEAD_FP8=1`).
- If the FP8 kernel does not reach ≥ 200 GB/s at M ≤ 4 on sm_121, use L1b 131072 in BF16.
- For CJK- or German-heavy traffic, 163840 costs +40 MiB/rank and raises the worst domain to 0.994.

Gate it as in plan L1:
- ≥ 10 sentinels.
- Structured acceptance exactly 4.0.
- code/CJK/tool acceptance ≥ −5%.
- T1-G 40×5, L1 off vs on.
- For L1b′, check argmax equality against the BF16 head on captured states (K1: ≥ 99.9%).

## Measured (session 4, `evidence/l1-l6-k5-l2/`)

Booted on 2x GB10 (v0.30.0 + determinism overlays). Marlin FP8 built on sm_121 for N = 124160, 81920
and 65536 (`Draft head (L1b'|L1b+L1b'): … FP8 marlin` on rank 0), and the drafter captured CUDA graphs.
Greedy c=1 output was bit-identical to U2 on every arm (T1-G 10 prompts, bench_diverse 64 prompts).

| Arm (all + L6a config) | prose c1 ms/step | struct c1 | struct c8 | diverse CJK acceptance | verdict |
|---|---|---|---|---|---|
| U2 | 60.40 | 61.98 | 81.98 | 2.384 | base |
| L1a | 60.43 | 61.04 | 79.09 | – | kept |
| L1a + L1b′ (full vocab FP8) | 53.09 | 54.78 | 75.62 | 2.382 | **kept (default)** |
| L1a + L1b′ + L1b 131072 | 50.99 | 52.43 | 72.70 | 2.185 (−8.3%) | not kept |
| L1a + L1b′ + L1b 163840 | 51.21 | 52.88 | 73.11 | 2.259 (−5.2%) | not kept (CJK gate −5%) |

`run.sh`: `DRAFT_LOCAL_ARGMAX=1`, `DRAFT_HEAD_FP8=1` default; `DRAFT_VOCAB=draft_vocab_163840.json`
is the opt-in for non-CJK traffic (+2-5% tok/s on prose/code/JSON/tools, −2.4% on CJK).
