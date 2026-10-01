# S1.5 determinism: static analysis and A/A data forensics (2026-09-29)

Read-only work. No GPUs, no containers. Sources:
- `$V030` = `.../7e119315.../scratchpad/src/v030/vllm`
- `$PIN` = `.../src/pinned/vllm`
- FlashInfer 0.6.18.post1 source (the same `_build_meta` commit `8bc3b578` as the v0.30 image), copied from
  `/tmp/claude-1000/-home-sfxnz-projects-ai-lab-recipes-DeepSeek-V4-1-Flash-EXL3-vLLM-2x-DGX-Spark/00916df6-.../scratchpad/research/src/v030/flashinfer` (`$FI` below).
  The pin ships FlashInfer 0.6.18 (`69ff11fc`). That source is not on disk.

Data: `~/projects/data/qwen38-evals/runs/session1/{b0-pin,b1-pin-stride,u1-v030}/{t1-a,t1-b,t1g}`.
Analysis scripts: `/tmp/claude-1000/-home-sfxnz-projects-ai-lab-recipes-Qwen3-8-Flash-Next-NVFP4-vLLM-2x-DGX-Spark/96f4721c-9edc-49e9-8616-f80a1491fe8a/scratchpad/aa.py`, `aa2.py`.

## TL;DR

1. **The likeliest source is FlashInfer CUTLASS NVFP4 MoE "fused finalize".** GEMM2 sums the 10 routed expert outputs per token with
   **BF16 `red.global.add` atomics** in its epilogue. That makes the summation order depend on CTA scheduling on every run.
   `use_fused_finalize` defaults to True, and vLLM never passes it (v0.30 and pin `flashinfer_cutlass_moe.py:367`). It runs in all 48 layers,
   for every token, at c=1.
2. **The magnitude comes from the model, not from the source.** In the A/A data, every pair of runs is the same distance apart:
   within a boot, across boots, pin vs pin+stride, and pin vs v0.30 (mean |Δlogp| 0.148-0.151, top-1 92.6-93.0%).
   Any perturbation, including a 1-ulp atomic reorder, saturates to the same noise floor. FP4 activation re-quantization and
   top-10-of-512 routing across 48 layers act as a chaotic amplifier. So the ~7% is what *any* bitwise nondeterminism looks like on
   this model. It is not necessarily a corruption bug.
3. **The data rules these out for T1:** stale-state leakage (no decay with position, no position excess once margin is controlled),
   QSA top-k/F23 (T1 prompts are ≤1,000 tokens, so QSA is in its dense, identity select-all regime), NCCL (a 2-rank sum is commutative),
   prefix cache (prompt_logprobs requests skip prefix reads), chunked prefill (one chunk) and spec decode (prompt logits come from the
   target prefill only).
4. **The decisive cheap test is a one-line overlay: `use_fused_finalize=False`.** Then check bitwise equality of `tgt_lp` between two T1 captures.
   The baseline is **0.2% of positions bit-exact**. If the finalize is the only source, the prediction is 100% exact.

## 1. A/A data analysis

Setup: `collect_t1.py` sends teacher-forced chat requests (`prompt_logprobs=5`, `max_tokens=1`, `continue_final_message`, thinking off),
24 items × ~970 tokens, on an idle serve (`others_running=0` for every item). Prompts are 676-990 tokens: **one prefill chunk**
(`MAX_NUM_BATCHED_TOKENS=8192`), **eager prefill** (`cudagraph_mode FULL_DECODE_ONLY`), and **no prefix-cache reads**
(`$V030/sampling_params.py:543-547` sets `skip_reading_prefix_cache = prompt_logprobs is not None`).
That is why `cache_salt` changes nothing.

### 1a. Where divergence starts: at the first scored token, in every item, in every config

For all 72 item×config pairs, the first inexact `tgt_lp` is at `score_from` (position 17-20, the first scored token), and |Δ|>0.01 appears
by the next token. The first top-1 flip is usually within 1-10 tokens. Only **0.2-0.3% of all positions have a bit-identical target
logprob** (0.0% at positions <256). This holds even for tokens with p>0.95.
**Every position of every forward pass differs, including the very first scored token.** This is per-forward noise, not accumulated drift.

### 1b. Length: no growth, no threshold, no chunk boundary

| pos bucket | n | top-1 flip (B1 / U1) | exact |
|---|---|---|---|
| [0,32) | 266 | 13.5% / 12.8% | 0.0% |
| [32,64) | 640 | 11.6 / 10.6 | 0.0 |
| [64,128) | 1310 | 9.2 / 7.9 | 0.0 |
| [128,256) | 2560 | 6.9 / 8.0 | 0.0 |
| [256,512) | 5313 | 7.0 / 6.9 | 0.5 |
| [512,768) | 5020 | 6.6 / 6.2 | 0.1-0.2 |
| [768,1100) | 3383 | 7.0 / 6.1 | 0.1 |

The early excess is **entropy, not state**. Bucketed by the top-1/top-2 margin, the flip rate early (pos<64) is equal to or *lower* than
late (pos≥256) in every bucket:

| min margin (nats) | flip all | early <64 | late ≥256 |
|---|---|---|---|
| <0.05 | 55.5% | 46-60% | 56% |
| 0.25-0.5 | 29-30% | 17-22% | 32-33% |
| 0.5-1 | 11.5% | 6-9% | 13-14% |
| 1-2 | 1.1-1.9% | 1.2-1.4% | 1.4-1.8% |
| >2 | 0.0% | 0.0% | 0.0% |

The flip probability is a pure function of margin. It implies logit-difference noise with σ ≈ 0.4-0.5 nats. For a low-probability target
(p<0.2) the std of Δtgt_lp is 0.5-0.96 nats. For p>0.95 it is 0.01-0.08.
Nothing changes at 2,048 tokens (QSA sparsity) or 8,192 (chunking): the corpus never reaches either.

### 1c. Domain: explained by entropy

Per item, English prose / long / code / math flip 0.7-3%, JSON 3-9%, zh 8-10%, and es/de/ja/multi 10-15%. These track the per-domain
mean NLL (prose 0.22, code 0.22, math 0.29, JSON 0.68, zh 1.16, de 1.23, es 1.42, multi 1.56, ja 1.62), so they track the share of
low-margin positions. After margin control (1b), nothing domain-specific remains. Spanish looks worst only because Spanish prose is
high-entropy for this model.

### 1d. Structure of the noise

- **Zero-mean, no quality bias.** Mean NLL A vs B: B1 0.8501 vs 0.8530 (Δ +0.0028 ± 0.0028), U1 0.8510 vs 0.8562 (Δ +0.0051 ± 0.0028).
  Neither run is systematically worse.
- **Independent per token.** The within-item autocorrelation of Δtgt_lp is 0.005-0.035 at lag 1 and about 0 at lags 2/5/20/100.
  A stale initial state (GDN/conv/PLE) would give a correlated error that decays along the sequence. None is present.
- **Saturated.** All 15 pairs among the 6 runs (b0a, b0b, b1a, b1b, u1a, u1b) show mean |Δ| 0.148-0.151 and top-1 92.6-93.0%.
  Same-boot A/A equals cross-boot equals cross-engine. The FLA→FlashInfer GDN prefill swap and the pin→v0.30 swap are deterministic
  numeric changes, yet they add *nothing* beyond A/A. The Δ(b1) vs Δ(u1) correlation is 0.037.
  Interpretation: the system forgets the size of the initial perturbation. Any nondeterministic ulp becomes the same ~0.036 KL.
  That is about twice the typical NVFP4-vs-BF16 KL, the signature of "re-drawing all the discrete rounding and routing decisions" on each run.

### 1e. Greedy repeats (T1-G, 10 prompts × 5 repeats)

The fork positions are **prompt properties shared across configs**. p00 forks at 44 in B0, B1 and U1. p01 forks at 10 (all three),
p07 at 6, p09 at 28, p03 at 4. p04 sometimes forks at **token 0**, which is argmax of the prefill logits, so prefill alone is enough
to diverge greedy output (spec decode is not needed). This matches 1b: repeated noise flips the same low-margin positions.

## 2. Ranked candidate root causes

### #1 (high): FlashInfer CUTLASS NVFP4 MoE fused-finalize BF16 atomics

- **Call site, no override:** `$V030/model_executor/layers/fused_moe/experts/flashinfer_cutlass_moe.py:367-392` calls
  `flashinfer_cutlass_fused_moe(...)` without `use_fused_finalize`. The pin has the identical call at `:367`. Neither tree contains the
  string `fused_finalize` (grep of both).
- **Default is True:** `$FI/fused_moe/core.py:1225` (`use_fused_finalize: bool = True`). `$FI/fused_moe/api.py:233-241` documents it:
  *"Whether supported backends reduce routed outputs in the GEMM2 epilogue (**atomic accumulation**) instead of running a separate reduction kernel."*
- **Eligible on sm_121:** `$FI/data/csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/include/moe_kernels.h:936-944`
  (`mayHaveFinalizeFused`: TMA-WS && sm≥90 && use_fused_finalize) and `.../moe_gemm/moe_gemm_template_dispatch.h:646-664`.
  The latter duplicates every TMA-WS config as `EpilogueFusionType::FINALIZE` for the GEMM2 tactic list, and SM120/121 is in the
  TMA-WS set (`:614-620`, `moe_gemm_template_dispatch_tma_ws.h:212-228`). The autotuner chooses the tactic, and
  `cutlass_fused_moe_kernels.cuh:4547-4560` enables the finalize fusion when that tactic wins.
- **Mechanism:** `$FI/.../cutlass_extensions/epilogue/collective/epilogue_moe_finalize.hpp:406-433,461` sets
  `CopyAtomR2G = SM90_RED_ADD_NOFTZ_BF16x2_V4` for a bf16 output. Each (token, expert) tile adds its scaled result into the
  **BF16** output row with a global atomic. For k=10 routed experts (plus the shared expert, if it is fused as an extra expert), the
  order of the 10 adds changes from run to run. The result differs by about 1 bf16 ulp (2⁻⁸ relative) per element, per MoE layer,
  per token, in 48 layers. That is independent per-token noise injected everywhere, which matches 1a and 1d exactly.
- **Why it is large here (see #2):** the next layer re-quantizes to NVFP4 (e2m1, 16-element blocks with an e4m3 block scale), and the router
  takes top-10 of 512. Ulp noise flips rounding and routing decisions, and the flips compound.
- **Gap in the evidence:** I could not see which GEMM2 tactic the autotuner picked on GB10 for M≈1000 (or M=4). The generated SM120 launcher
  `.inl` is not in the source copy. The experiments below settle it.

**Experiments (cheapest first):**
- **E1 (offline, 1 GPU, serve stopped, ~5 min).** Extend `tools/micro/moe_nvfp4.py`, which already calls `cutlass_fused_moe` on
  sm_121 with the real shapes E=512, k=10, H=2560, I/TP=320. Run the same input twice under `autotune` at M∈{4, 1000} and
  `torch.equal` the outputs. Then run it again with `use_fused_finalize=False`.
  Prediction: fused → not bit-equal at M=1000 (and probably at M=4); unfused → bit-equal on every call.
  First verify that the pin's 0.6.18 has the flag: `python -c "import inspect,flashinfer.fused_moe as m;print(inspect.signature(m.cutlass_fused_moe))"` in both images.
- **E2 (live, one-line overlay).** Add `use_fused_finalize=False,` to the call at `flashinfer_cutlass_moe.py:367` (v0.30: bind-mount a patched copy,
  as `docker/OVERLAYS.md` does). Run `collect_t1.py --limit 2` twice and compute **the exact-equality fraction of `tgt_lp`**, not top-1.
  Pass: ≥99.9% exact and top-1 100%. Then run T1-G: expect 1 distinct output per prompt with `SPEC=mtp`. If not, move on to #4/#5.
  Also record prefill TTFT and c=1/c=8 ms/step; the separate finalize kernel costs a few µs per layer.
- **E2b (no code, alternative).** `MOE_BACKEND=cutlass` selects `NvFp4MoeBackend.VLLM_CUTLASS` (`oracle/nvfp4.py:159`). `CutlassExpertsFp4` supports family 120
  (`experts/cutlass_moe.py:705-710`) and claims batch invariance (`:740-741`). The drafter stays on triton because SPEC_CONFIG
  pins `"moe_backend":"triton"` (`run.sh:54`). run.sh does not refuse `cutlass`; its guards cover only marlin, flashinfer_cutlass and b12x.
  This is unmeasured on sm_121 here, so it is a determinism probe only, not a perf candidate.
- **E3 (stack-wide fallback).** `VLLM_BATCH_INVARIANT=1` rejects FlashInfer CUTLASS (`modular_kernel.py:589`), which forces VLLM_CUTLASS.
  It also swaps aten mm/log_softmax/rms_norm for batch-invariant Triton kernels (`determinism/batch_invariant.py:1055-1185`).
  **Caution:** it also forces `NCCL_P2P_NET_DISABLE=1`, `NCCL_MAX_NCHANNELS=1`, `NCCL_NTHREADS=1` and `NCCL_ALGO=allreduce:tree`
  (`:1141-1158`) on a 2-node IB job. Use it only if E2 leaves residual non-exactness.

### #2 (high, explains magnitude; not a separate bug): the model and quantization stack is chaotic under ulp perturbations

Evidence is §1d: every pairwise distance is saturated, deterministic kernel changes equal A/A, Δ is zero-mean and i.i.d., and the flip
rate depends only on margin. The amplifiers are NVFP4 activation re-quantization in 48 MoE layers, discrete top-10/512 routing
(`num_experts_per_tok 10`, `num_experts 512` in the snapshot `config.json`), the HC 4-stream mixing, and (>2k tokens) the QSA top-512-block selection.
Consequences:
- After #1 is fixed, **any** remaining nondeterministic ulp (see #3-#5) brings back the full ~7%. The gate must be **bitwise
  exactness**, not a top-1 threshold.
- Even with a deterministic stack, a numerically benign kernel swap (for example a GDN or GEMM backend change) will show ~93% top-1 against the
  previous build. **T1-A top-1 ≥98.5% cannot work as a gate for kernel levers on this model.** Keep T1 for bitwise
  determinism (same build), and use NLL-delta and task accuracy for cross-build quality (T2 already does: GSM8K/IFEval paired, 0 lost).
- **E4:** after E2, inject a controlled 1-ulp perturbation. For example, flip `use_fused_finalize` for one layer only, or run with
  `--gdn-prefill-backend triton` (`arg_utils.py:1775-1780`) vs flashinfer. If T1 top-1 against the deterministic baseline again comes out at
  ~93%, chaos is confirmed as the amplifier and the gate redesign is required.

### #3 (low-medium): other tactic- or scheduling-dependent reductions still in the path

None was found in the Python/Triton code reviewed. These are the remaining opaque kernels to bisect if E2 is not fully exact:
- FlashInfer GDN prefill (v0.30 default, `qwen_gdn_linear_attn.py:99-165`). The pin uses the Triton FLA path and has the same noise, so it
  is not the main source. **E5a:** `--gdn-prefill-backend triton` on v0.30 plus E2, then an exactness check.
- The FLA Triton ops contain no atomics (grep of `model_executor/layers/fla`, `mamba`). The only `tl.atomic_add` in the router is an int
  expert counter (`fused_moe/router/base_router.py:93`), which does not affect numerics.
- cuBLAS/cuBLASLt with several streams sharing a workspace (PLE side stream when offload is on; shared-expert aux stream). **E5b:**
  `EXTRA_ENV="CUBLAS_WORKSPACE_CONFIG=:4096:8"` plus E2, then an exactness check.

### #4 (ruled out for T1, real above ~2k tokens): QSA `persistent_topk` order (F23)

`indexer_budget 2048 / compress_ratio 4` gives 512 blocks, and `visible_blocks=(pos+1)//4`. For T1 prompts (≤990 tokens, ≤247 blocks)
`persistent_topk` takes its identity select-all path (findings F23/F05; `nvidia/ops/qsa_indexer.py:473-501`). The expand kernel masks by
`visible` (`:239-266`). Prefill uses `num_splits=1` (`ops/qsa.py:442-474`: bp = rows×1 kv-head/rank >512 → 1 split). Empty
split-K partials in decode store `lse=-inf` and output 0 (`ops/qsa.py:178-197`), so no `torch.empty` garbage is consumed.
**E6:** after E2, run the A/A exactness check on a ~4k-token prompt. Non-exact positions should begin at token ≥2051 and vanish with an exact
`torch.topk` overlay that masks unwritten columns (F23 remedy (b)).

### #5 (ruled out by code and data): stale state reuse between requests

- GDN: `initial_state[~prefill_has_initial_state, ...] = 0` (`$V030/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:1523-1527`).
- PLE short conv: `read_state = state_ok & has_init` masks the state taps (`nvidia/ops/ple.py:392-413`).
- PLE n-gram context: `context.fill_(eos)` before gather (`nvidia/model_state.py:73`); the prefix is EOS-padded when `num_computed_tokens=0`.
- Data: no decay with position and no early excess after margin control (§1b). Noise is independent per token (§1d).
- **E7 (after E2):** send X, then Y, then X again, and compare X with X sent back-to-back (X,X). A difference that depends on the predecessor would expose a slot leak.

### #6 (ruled out): NCCL all-reduce

TP=2: every element is reduced exactly once as `a+b`, which is commutative in IEEE, then broadcast. This holds for ring, tree, LL, LL128 and Simple.
The PLE FP8 path reduces int8 bytes with one owner per row (`vocab_parallel_embedding.py:538-550`; the pinned-host path in
`nvidia/ngram_embedding.py:486-489`), so it is exact.

### #7 (ruled out for T1; may add to greedy decode drift): MTP spec decode

T1 logits come only from the target prefill, and T1-G sometimes forks at token 0. GDN verify/rollback (F08) can add decode-time drift.
**E8:** `SPEC=none` T1-G after E2 separates it.

### #8 (ruled out): prompt_logprobs, chunked prefill, prefix cache

There is one chunk per request (<8192), and prompt_logprobs requests skip prefix-cache reads (`sampling_params.py:543-547`). The log-softmax
is fp32 on vocab-parallel logits that are all-gathered, not reduced.

## 3. Suggested run order (one boot, U1 config)

1. E1 micro (serve down, ~5 min) confirms the mechanism without a boot.
2. Boot U1 + E2 overlay, then T1 --limit 2 ×2 → exact-fraction. T1-G ×5.
3. If <99.9% exact: add E5a (`--gdn-prefill-backend triton`) and E5b (`CUBLAS_WORKSPACE_CONFIG`), one knob at a time. Then try E3.
4. Once exact: E7 (stale state), E6 (4k prompt, F23), E8 (SPEC=none greedy), E4 (chaos confirmation). Measure the E2 perf cost with the ruler.
5. Rewrite the T1 gate: the same-build gate is `tgt_lp` bit-exact ≥99.9%. The cross-build gate uses NLL delta, T2 and paired task accuracy. Drop top-1 ≥98.5%
   for kernel levers.
