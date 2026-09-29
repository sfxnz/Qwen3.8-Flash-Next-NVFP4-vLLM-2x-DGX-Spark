# AGENTS.md — Qwen3.8-Flash-Next-NVFP4 · 2× DGX Spark

Serve `nvidia/Qwen3.8-Flash-Next-NVFP4` at TP=2. Official image `vllm/vllm-openai:nightly-aarch64@sha256:df871f170ee7070fbdce162bde08fb616e311570c948a620be0d4b33fe02f87b` (`0.28.1rc1.dev437+ge962733e0`). Snapshot `fab0aec`. Checkpoint credit: [NVIDIA ModelOpt](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4) (sychen52). Stock `v0.27.1` does not register `Qwen4ExpForConditionalGeneration`.

Humans read [README.md](README.md).

## Working rules

- `recipe.yaml` is the source of truth for pins and generated blocks. Edit it, then `python3 kit/render.py`. Do not hand-edit `# BEGIN generated` or `<!-- BEGIN generated` blocks.
- Change one knob at a time against `python3 bench_decode.py`. Revert if it does not beat noise or it regresses another cell. Record the revert in `evidence/`.
- Read unified memory with `free -h`. Never `nvidia-smi` VRAM.
- Exclusive GPUs. Do not start this while another `--gpus all` serve is up.
- Pin `NCCL_IB_HCA`. GB10 exposes four HCAs and two are DOWN. Unpinned NCCL picks a dead one and fails with `unhandled system error`. Defaults in `run.sh` are `enp1s0f1np1` / `rocep1s0f1`.
- Default thinking is off. `chat_template_kwargs`: `enable_thinking=false`. The card template seeds an empty `<think></think>` when thinking is off.
- On the pin keep `docker/ple_layer.py` (qwen4_exp MIXED_PRECISION FP8 PLE), `docker/modelopt.py` (MTP FP8_BLOCK_SCALES) and `docker/ple_ops.py` (vLLM #55375 PLE short-conv state-index stride, F01). Nightly already selects FP8 PLE via `Qwen4ExpPLEFp8EmbeddingMethod`. Do not require `VLLM_PLE_FP8_CHECKPOINT`. Day-0 `qwen38-flash-next@3b0e188` and stock nightly both raise `AttributeError: mtp.layers.48.mlp.experts has no parameter 'w2_weight_scale_inv'`. Regenerate with `python3 docker/apply_ple_overlay.py`, `python3 docker/apply_mtp_fp8_overlay.py` and `python3 docker/apply_ple_stride_overlay.py`. Never hand-edit an overlay.
- Overlays are data: every overlay starts with an R11 header (`base_image_digest`, `upstream_file`, `upstream_file_sha256`, `upstream_PR`). `run.sh` mounts the `OVERLAYS` list (`auto` picks `OVERLAYS_PIN` / `OVERLAYS_V030` by IMAGE digest) at each `upstream_file`, and refuses a digest mismatch and the pin overlays on any other image. See `docker/OVERLAYS.md`.
- On the pin, `--moe-backend` stays `auto`. The V2 runner ignores `SPEC_CONFIG` `moe_backend`, the drafter inherits `--moe-backend`, and its 64x64-refined FP8 blocks run only on Triton.
- Native `max_position_embeddings` and tokenizer `model_max_length` are 262144. This recipe serves that window. 1048576 is a lab ceiling, not a trained window. Do not treat 1M as a quality pin. vLLM refuses a longer window unless `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`. That env is not YaRN.

`ORCHESTRATE=auto` (default): if SSH to `WORKER_HOST` fails, `run.sh` exits 1. Do not start a TP=2 head rank alone.

## Refuse-guards (`run.sh`)

All run before `VALIDATE_ONLY=1` exits. Overrides in brackets; no bracket means no override.

- non-decimal or zero-padded integers, non-JSON `SPEC_CONFIG` / `COMPILATION_CONFIG`, an `IMAGE` without a digest, `EXTRA_ARGS` re-setting a flag `run.sh` owns (compilation, speculative, MoE backend, batched tokens, seqs, window, eager)
- an overlay that is missing, has no R11 header, names another `base_image_digest`, or is a pin overlay (by content sha) on a non-pin image
- on the pin, no `modelopt.py` or `ops/ple.py` overlay [`DIAGNOSTIC=1`]
- `--max-model-len` above 1048576, `MAX_NUM_SEQS` above 8, compile `mode` other than 0 on the pin, and a changed `MAX_NUM_BATCHED_TOKENS` / `--long-prefill-token-threshold` / `indexer_kv_dtype` with `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=none` [`FORCE_UNSAFE_CTX=1`]
- `--max-model-len` above 262144 when `VLLM_ALLOW_LONG_MAX_MODEL_LEN` is not `1`
- `MOE_BACKEND` other than `auto` on the pin; `b12x` / `flashinfer_b12x` anywhere [`FORCE_UNSAFE_MOE=1`]
- `SPEC=none`, `use_local_argmax_reduction` without a `get_top_tokens` overlay or with probabilistic drafts, `enable_adaptive_verification`, `--enable-batch-sharded-sampling` without a `compute_logits_local` overlay, `VLLM_BATCH_INVARIANT=1` in `EXTRA_ENV` [`DIAGNOSTIC=1`; such boots are never pinned]
- `rejection_sample_method=synthetic` unless `BENCH_ONLY=1`, which binds the API to 127.0.0.1

Default occupancy is eight sequences. Do not raise `MAX_NUM_SEQS` above 8 without a new occupancy row. Env diagnostics go through `EXTRA_ENV` (both ranks), never a `run.sh` edit.

## Verify

```bash
python3 -m unittest discover -s tests -q
python3 kit/render.py --check
VALIDATE_ONLY=1 ./run.sh
python3 bench_decode.py
```

After `./run.sh` is up, `GET /health` must be 200 and `GET /v1/models` must list `nvidia/Qwen3.8-Flash-Next-NVFP4`. Thinking-off smoke must not start `content` with chain-of-thought. `smoke_tools.py` must emit `get_weather`. `smoke_vision.py` must accept OpenAI `image_url` and must not return "is not a multimodal model". Greedy count 1 to 200 stays consecutive with thinking off. `python3 bench_decode.py` is the frozen 2x decode ruler.

## Never touch

- Live HF tokens
- Floating `:nightly-aarch64` without the digest
- Hand-edited generated README / `run.sh` blocks
- Advertising a 1M needle that was not run
