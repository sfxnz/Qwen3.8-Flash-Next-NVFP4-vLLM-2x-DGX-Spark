# AGENTS.md — Qwen3.8-Flash-Next-NVFP4 · 2× DGX Spark

Serve `RadixArk/Qwen3.8-Flash-Next-NVFP4` at TP=2. Official day-0 image `vllm/vllm-openai:qwen38-flash-next@sha256:3b0e188ffceb3d07e09c3cb5215433a0020eacf02d7f882ed3a8bfd15454477e`. Snapshot `7b719225`. Stock `v0.27.1` does not register `Qwen4ExpForConditionalGeneration`.

Humans read [README.md](README.md).

## Working rules

- `recipe.yaml` is the source of truth for pins and generated blocks. Edit it, then `python3 kit/render.py`. Do not hand-edit `# BEGIN generated` or `<!-- BEGIN generated` blocks.
- Change one knob at a time against `python3 bench_decode.py` once that table exists. Revert if it does not beat noise or it regresses another cell. Record the revert in `evidence/`.
- Read unified memory with `free -h`. Never `nvidia-smi` VRAM.
- Exclusive GPUs. Do not start this while another `--gpus all` serve is up.
- Pin `NCCL_IB_HCA`. GB10 exposes four HCAs and two are DOWN. Unpinned NCCL picks a dead one and fails with `unhandled system error`. Defaults in `run.sh` are `enp1s0f1np1` / `rocep1s0f1`.
- Default thinking is off. `chat_template_kwargs`: `enable_thinking=false`. The card template seeds an empty `<think></think>` when thinking is off.
- Keep `docker/ple_layer.py`. The day-0 image rejects mixed ModelOpt NVFP4 plus FP8 PLE unless `VLLM_PLE_FP8_CHECKPOINT=1` selects `Qwen3_8FlashNextPLEFp8EmbeddingMethod`. Regenerate with `python3 docker/apply_ple_overlay.py`.
- MTP experts stay BF16. Do not set global `--moe-backend marlin`. This image then builds the unquantized MTP MoE with marlin and exits. Default is `auto`.
- Native `max_position_embeddings` and tokenizer `model_max_length` are 262144. 1048576 is a lab ceiling, not a trained window. Do not treat 1M as a quality pin. The day-0 image refuses that window unless `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`. That env is not YaRN.

`ORCHESTRATE=auto` (default): if SSH to `WORKER_HOST` fails, `run.sh` exits 1. Do not start a TP=2 head rank alone.

## Refuse-guards (`run.sh`)

Exits unless `FORCE_UNSAFE_CTX=1` or `FORCE_UNSAFE_MOE=1`:

- `--max-model-len` above 1048576
- `MAX_NUM_SEQS` above 2
- `MOE_BACKEND=marlin` (unquantized MTP MoE rejects it)
- `MOE_BACKEND=flashinfer_cutlass` or `b12x`
- `VLLM_PLE_FP8_CHECKPOINT` not `1`
- `--max-model-len` above 262144 when `VLLM_ALLOW_LONG_MAX_MODEL_LEN` is not `1`

Do not raise `MAX_NUM_SEQS` on this first occupancy pin.

## Verify

```bash
python3 -m unittest discover -s tests -q
python3 kit/render.py --check
VALIDATE_ONLY=1 ./run.sh
```

After `./run.sh` is up, `GET /health` must be 200 and `GET /v1/models` must list `RadixArk/Qwen3.8-Flash-Next-NVFP4`.

## Never touch

- Live HF tokens
- Floating `:qwen38-flash-next` without the digest
- Hand-edited generated README / `run.sh` blocks
- Advertising a 1M needle that was not run
