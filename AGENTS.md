# AGENTS.md — Qwen3.8-Flash-Next-NVFP4 · 2× DGX Spark

Serve `nvidia/Qwen3.8-Flash-Next-NVFP4` at TP=2. Default image `vllm/vllm-openai:v0.30.0-aarch64@sha256:4864d46625cbc3307623e29ac742030655e27249feba7b97ec925ce4cc4dfb56` (`0.30.0`) with the v0.30 overlay set. Rollback: `IMAGE=vllm/vllm-openai:nightly-aarch64@sha256:df871f170ee7070fbdce162bde08fb616e311570c948a620be0d4b33fe02f87b ./run.sh` (the pin, `0.28.1rc1.dev437+ge962733e0`, mounts the pin overlay set). Snapshot `fab0aec`. Checkpoint credit: [NVIDIA ModelOpt](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4) (sychen52). Stock `v0.27.1` does not register `Qwen4ExpForConditionalGeneration`.

Humans read [README.md](README.md).

## Working rules

- `recipe.yaml` is the source of truth for pins and generated blocks. Edit it, then `python3 kit/render.py`. Do not hand-edit `# BEGIN generated` or `<!-- BEGIN generated` blocks.
- Change one knob at a time against `python3 bench_decode.py`. Revert if it does not beat noise or it regresses another cell. Record the revert in `evidence/`.
- Read unified memory with `free -h`. Never `nvidia-smi` VRAM.
- Exclusive GPUs. Do not start this while another `--gpus all` serve is up.
- Pin `NCCL_IB_HCA`. GB10 exposes four HCAs and two are DOWN. Unpinned NCCL picks a dead one and fails with `unhandled system error`. Defaults in `run.sh` are `enp1s0f1np1` / `rocep1s0f1`.
- Default thinking is off. `chat_template_kwargs`: `enable_thinking=false`. The card template seeds an empty `<think></think>` when thinking is off.
- Overlay sets are per base image (`OVERLAYS=auto` picks by digest). Never hand-edit an overlay; regenerate it with its generator.
  - **v0.30 (default), `OVERLAYS_V030`.** v0.30 loads the MTP FP8_BLOCK_SCALES experts, selects FP8 PLE and fixes F01 natively, so none of the pin overlays apply (`run.sh` refuses them there). The set is the three S1.5 determinism overlays, one API rider and the draft-head overlay:
    - `docker/v030/flashinfer_cutlass_moe.py` (`apply_moe_finalize_overlay.py`): `use_fused_finalize=False`. FlashInfer's fused finalize sums the top-10 experts with BF16 atomics, which made every c=1 greedy run differ (T1 A/A 93%). `VLLM_QWEN38_MOE_DETERMINISTIC=0` restores stock for an A/B.
    - `docker/v030/gdn_attn.py` (`apply_gdn_fresh_prefill_overlay.py`): a fresh 1-token prompt no longer runs the GDN decode kernel on a leftover state slot.
    - `docker/v030/qsa_indexer.py` (`apply_qsa_topk_order_overlay.py`): QSA `persistent_topk` output is sorted, and prefill ties resolve to the lowest index (F23).
    - `docker/v030/serving.py` (`apply_api_overlays.py`): vLLM #56067, the streamed tool-call parser's O(n²) API-server CPU (G05).
    - `docker/v030/mtp.py` (`apply_mtp_overlay.py`): the draft-head levers (session 4, `evidence/l1-l6-k5-l2/`). `run.sh` applies `DRAFT_LOCAL_ARGMAX=1` (L1a, `use_local_argmax_reduction`), `DRAFT_HEAD_FP8=1` (L1b′ FP8 draft head) and `DRAFT_VOCAB` (L1b, default none: −5% CJK acceptance) only with this overlay on the v0.30 digest. `DRAFT_MOE_CONFIG=1` mounts `docker/v030/moe_configs` (L6a). Greedy output stays bit-identical to U2 at c=1; c=1 step −7 ms.
    With these, c=1 greedy is bit-exact run to run (T1 A/A 100%, T1-G 1.0 distinct) for about +1.4 ms/step at c1 (`evidence/s15-determinism/SUMMARY.md`). Do not drop one to win back speed without a new T1-G row. Long generations are not fully pinned: a 1578-token prompt run to 4096 greedy tokens at c=1 on an idle serve gave the same output 12 of 13 times, and one run diverged at generated token 501 (`evidence/fp8-default/near1600-determinism.txt`).
  - **Pin (rollback), `OVERLAYS_PIN`.** `docker/ple_layer.py` (FP8 PLE), `docker/modelopt.py` (MTP FP8_BLOCK_SCALES; day-0 and stock nightly raise `AttributeError: mtp.layers.48.mlp.experts has no parameter 'w2_weight_scale_inv'`) and `docker/ple_ops.py` (vLLM #55375 short-conv stride, F01). Generators: `docker/apply_ple_overlay.py`, `docker/apply_mtp_fp8_overlay.py`, `docker/apply_ple_stride_overlay.py`. The pin keeps greedy c=1 non-determinism (no determinism overlays exist for it). Do not require `VLLM_PLE_FP8_CHECKPOINT`.
- On the v0.30 digest `run.sh` always passes `VLLM_PLE_CPU_OFFLOAD=0` (the v0.30 default of 1 pins 32 GiB of host memory per node, F33) and `VLLM_USE_BREAKABLE_CUDAGRAPH=0` to both ranks. They are not `EXTRA_ENV` entries; do not move them there.
- Serve flags owned by `run.sh` from `recipe.yaml`: `--long-prefill-token-threshold 4800` (P2-5: a short request during a 128k prefill waits ~5 s instead of ~61 s; needs `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB`; 3200 with 4832 batched tokens was measured and not kept, `evidence/extras/`), `--async-scheduling` (already v0.30's resolved default for MTP + mp; explicit), `--mm-processor-kwargs` (images 65536..4194304 px, video `cap_pixels_per_frame`), `--limit-mm-per-prompt` (image 8, video 1), `--mm-processor-cache-gb 1`, and `--chat-template` (`chat_template_alias.jinja`, head only, read-only mount: `reasoning_effort` high/max → xhigh, minimal → low). Regenerate the template with `tools/make_alias_template.py`. Do not pass `--per-request-spec-decode-metrics` on a served default; it costs 0.7-1 ms/step.
- Overlays are data: every overlay starts with an R11 header (`base_image_digest`, `upstream_file`, `upstream_file_sha256`, `upstream_PR`). `run.sh` mounts the `OVERLAYS` list (`auto` picks `OVERLAYS_PIN` / `OVERLAYS_V030` by IMAGE digest) at each `upstream_file`, and refuses a digest mismatch and the pin overlays on any other image. See `docker/OVERLAYS.md`.
- `--moe-backend` stays `auto`. On the pin the V2 runner ignores `SPEC_CONFIG` `moe_backend`, the drafter inherits `--moe-backend`, and its 64x64-refined FP8 blocks run only on Triton (`run.sh` refuses anything else there). v0.30 honours the SPEC_CONFIG `moe_backend: triton`.
- Native `max_position_embeddings` and tokenizer `model_max_length` are 262144. This recipe serves that window. 1048576 is a lab ceiling, not a trained window. Do not treat 1M as a quality pin. vLLM refuses a longer window unless `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`. That env is not YaRN.

`ORCHESTRATE=auto` (default): if SSH to `WORKER_HOST` fails, `run.sh` exits 1. Do not start a TP=2 head rank alone.

## Host memory safety (incident 2026-09-29)

spark1 went into a global host OOM with swap thrash during an in-serve torch-profiler window at `UTIL=0.80` (`evidence/incident-2026-09-29-oom/`).

- Keep `UTIL` at 0.76, `OOM_SCORE_ADJ=1000` and `MEMGUARD=1` (per-node watchdog that `docker kill`s the serve when MemAvailable and SwapFree both stay below 2 GiB for 6 s).
- Keep `docker run --ulimit core=1` in `run.sh`. Without it, a worker abort makes host apport buffer an ~11 GiB core in RAM on both nodes (session 3 T3 64k trip). A limit of 0 does not stop apport.
- No torch-profiler windows on the served default. Profile only on a diagnostic boot with `UTIL` ≤ 0.70, a bounded window and `free -h` watched.
- Before a heavy step (T1/T3, long-context or mixed benches, a boot), run `free -h` on both nodes. Skip the step if spark1 MemAvailable < 6 GiB or swap used > 10 GiB on either node.
- Prompts above 32k tokens go one at a time, with a memory check between them. If memguard trips, record it and stop that test.

## Measurement notes

- Rulers: the frozen `bench_decode.py` is the regression guard; decisions use `ruler_steps.py` ms/step. The structured c1 sentinel anchor is re-set per base: pass `--anchor-ms LO HI` at ±1 ms around the base's measured structured c1 The session-4 default (L1a + L1b′ + L6a) used **`--anchor-ms 53.8 55.8`** (`evidence/l1-l6-k5-l2/SUMMARY.md`). With K3 (session 7 on), structured c1 measures 53.7–54.1, and 53.8–55.8 voided the session-8 gate ruler (6/9 sentinel excursions), so the current default uses **`--anchor-ms 52.8 54.8`** (`evidence/fp8-default/` (`near1600-determinism.txt`)); [60.7, 62.7] (U2) and [59, 61] void every current boot.
- With the determinism overlays greedy output is pinned, so prose acceptance is pinned too (about 2.37 at c1). Prose ms/step and tok/s are not comparable with pre-determinism boots; compare structured cells or re-baseline.
- T1 gates on the determinism base: the same-build floor is exact. Kernel-reorder levers move logits and the model amplifies ulp noise into nats, so class-A gates use NLL delta and paired task accuracy (T2), not a top-1 threshold.
- On builds without the determinism overlays (the pin, stock v0.30) greedy c=1 repeats diverge and T2 tools lands at 57-58/60 on different items per run. That is non-determinism, not a regression.

- `FP8_DENSE=per_block` is the opt-in `lab-fp8-dense` profile (P4-1): about 13% faster at c=1, but it fails the T1-B top-1 gate while passing T1-D, T2, T3 and V (`evidence/k-sweep-precision/SUMMARY.md`). T4 (full GSM8K, thinking on) passes: 97.65% against 97.80%, 6 lost / 4 gained, McNemar p 0.75 (`evidence/extras/SUMMARY.md`). Never make it the default without owner sign-off. Session 8 (`evidence/fp8-default/` (`near1600-determinism.txt`)) re-ran the pack on K3 with the owner's sign-off. Every quality gate passed, including a 6-language paired judge, but the flip was stopped: long c=1 greedy output is not reproducible run to run. A 4096-token generation from a 1578-token prompt gave 5 distinct outputs in 9 runs, against 2 in 13 on BF16. FP8 without K3 behaves the same way, so the cause is FP8 dense and not K3.

## Refuse-guards (`run.sh`)

All run before `VALIDATE_ONLY=1` exits. Overrides in brackets; no bracket means no override.

- non-decimal or zero-padded integers, non-JSON `SPEC_CONFIG` / `COMPILATION_CONFIG`, an `IMAGE` without a digest, `EXTRA_ARGS` re-setting a flag `run.sh` owns (compilation, speculative, MoE backend, batched tokens, seqs, window, eager)
- an overlay that is missing, has no R11 header, names another `base_image_digest`, or is a pin overlay (by content sha) on a non-pin image
- on the pin, no `modelopt.py` or `ops/ple.py` overlay [`DIAGNOSTIC=1`]
- `--max-model-len` above 1048576, `MAX_NUM_SEQS` above 8 without `THROUGHPUT_PROFILE=1` (or above 16 with it), compile `mode` other than 0 on the pin, and a changed `MAX_NUM_BATCHED_TOKENS` / `LONG_PREFILL_TOKEN_THRESHOLD` / `indexer_kv_dtype` with `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=none` [`FORCE_UNSAFE_CTX=1`]
- `--max-model-len` above 262144 when `VLLM_ALLOW_LONG_MAX_MODEL_LEN` is not `1`
- `MOE_BACKEND` other than `auto` on the pin; `b12x` / `flashinfer_b12x` anywhere [`FORCE_UNSAFE_MOE=1`]
- `SPEC=none`, `use_local_argmax_reduction` without a `get_top_tokens` overlay or with probabilistic drafts, `enable_adaptive_verification`, `--enable-batch-sharded-sampling` without a `compute_logits_local` overlay, `VLLM_BATCH_INVARIANT=1` in `EXTRA_ENV` [`DIAGNOSTIC=1`; such boots are never pinned]
- `rejection_sample_method=synthetic` unless `BENCH_ONLY=1`, which binds the API to 127.0.0.1
- `EXTRA_ARGS` setting `--long-prefill-token-threshold`, `--async-scheduling`, `--mm-processor-kwargs`, `--limit-mm-per-prompt`, `--mm-processor-cache-gb` or `--chat-template` (use the `recipe.yaml` variables); `MM_MIN_PIXELS` above `MM_MAX_PIXELS`; a `CHAT_TEMPLATE` file that does not exist

Default occupancy is eight sequences. `THROUGHPUT_PROFILE=1` is the opt-in `throughput` profile (X10, R8): `--max-num-seqs 16`, occupancy row in `evidence/extras/SUMMARY.md` (0 preemptions at 16 diverse streams, T0 with N=16 bursts clean, spark1 ≥ 15 GiB available under load). It is not the default. Do not go above 16, or make 16 the default, without a new occupancy row. Env diagnostics go through `EXTRA_ENV` (both ranks), never a `run.sh` edit.

## Verify

```bash
python3 -m unittest discover -s tests -q
python3 kit/render.py --check
VALIDATE_ONLY=1 ./run.sh
python3 bench_decode.py
```

`VALIDATE_ONLY=1` must also pass for the rollback: `IMAGE=vllm/vllm-openai:nightly-aarch64@sha256:df871f170ee7070fbdce162bde08fb616e311570c948a620be0d4b33fe02f87b VALIDATE_ONLY=1 ./run.sh`.

After `./run.sh` is up, `GET /health` must be 200 and `GET /v1/models` must list `nvidia/Qwen3.8-Flash-Next-NVFP4`. Thinking-off smoke must not start `content` with chain-of-thought. `smoke_tools.py` must emit `get_weather`. `smoke_vision.py` must accept OpenAI `image_url` and must not return "is not a multimodal model". Greedy count 1 to 200 stays consecutive with thinking off. `python3 bench_decode.py` is the frozen 2x decode ruler.

## Never touch

- Live HF tokens
- Floating `:nightly-aarch64` without the digest
- Hand-edited generated README / `run.sh` blocks
- Advertising a 1M needle that was not run
