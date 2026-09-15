# AGENTS.md — Qwen3.8-Flash-Next-NVFP4 · 2× DGX Spark

Serve `nvidia/Qwen3.8-Flash-Next-NVFP4` at TP=2. Official image `vllm/vllm-openai:nightly-aarch64@sha256:df871f170ee7070fbdce162bde08fb616e311570c948a620be0d4b33fe02f87b` (`0.28.1rc1.dev437+ge962733e0`). Snapshot `fab0aec`. Checkpoint credit: [NVIDIA ModelOpt](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4) (sychen52). Stock `v0.27.1` does not register `Qwen4ExpForConditionalGeneration`.

Humans read [README.md](README.md).

## Working rules

- `recipe.yaml` is the source of truth for pins and generated blocks. Edit it, then `python3 kit/render.py`. Do not hand-edit `# BEGIN generated` or `<!-- BEGIN generated` blocks.
- Change one knob at a time against `python3 bench_decode.py`. The decode score is prose only. Do not keep or revert from a structured or code cell. Revert if prose does not beat noise or it regresses another published bar. Record the revert in `evidence/`.
- Read unified memory with `free -h`. Never `nvidia-smi` VRAM.
- Exclusive GPUs. Do not start this while another `--gpus all` serve is up.
- Pin `NCCL_IB_HCA`. GB10 exposes four HCAs and two are DOWN. Unpinned NCCL picks a dead one and fails with `unhandled system error`. Defaults in `run.sh` are `enp1s0f1np1` / `rocep1s0f1`.
- Default thinking is off. `chat_template_kwargs`: `enable_thinking=false`. The card template seeds an empty `<think></think>` when thinking is off.
- Keep `docker/ple_layer.py` (qwen4_exp MIXED_PRECISION FP8 PLE) and `docker/modelopt.py` (MTP FP8_BLOCK_SCALES). Nightly already selects FP8 PLE via `Qwen4ExpPLEFp8EmbeddingMethod`. Do not require `VLLM_PLE_FP8_CHECKPOINT`. Day-0 `qwen38-flash-next@3b0e188` and stock nightly both raise `AttributeError: mtp.layers.48.mlp.experts has no parameter 'w2_weight_scale_inv'`. Regenerate with `python3 docker/apply_ple_overlay.py` and `python3 docker/apply_mtp_fp8_overlay.py`.
- Do not set global `--moe-backend marlin`. Default is `auto`. MTP experts are FP8_BLOCK_SCALES and use triton after a 64x64 refine.
- Native `max_position_embeddings` and tokenizer `model_max_length` are 262144. This recipe serves that window as a config cap. Public GB10 prefills hang near 74–78k ([vLLM #54629](https://github.com/vllm-project/vllm/issues/54629)). Do not treat 262144 as a measured prefill until L.A.I.L `pp` 8k–64k is green. 1048576 is a lab ceiling, not a trained window. Do not enable YaRN or advertise a 1M needle. `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` is not YaRN.
- Keep `--moe-backend auto`. Do not set global `marlin` or `b12x` without a new quality and Xid row on this SHA. Do not apply L.A.I.L’s generic `qwen38_nvfp4` overlay (`qwen3_coder` + `--kv-cache-dtype fp8`).
- Keep `--no-enable-flashinfer-autotune` (official Flash-Next samples). Keep `--mamba-cache-mode align` (this nightly already selects align when prefix caching is on). Keep `--quantization modelopt` (engine remaps MIXED_PRECISION to `modelopt_mixed`).
- Leave vision on. Do not pass `--language-model-only`.

`ORCHESTRATE=auto` (default): if SSH to `WORKER_HOST` fails, `run.sh` exits 1. Do not start a TP=2 head rank alone.

## Refuse-guards (`run.sh`)

Exits unless `FORCE_UNSAFE_CTX=1` or `FORCE_UNSAFE_MOE=1`:

- `--max-model-len` above 1048576
- `MAX_NUM_SEQS` above 8
- `MOE_BACKEND=marlin`
- `MOE_BACKEND=flashinfer_cutlass` or `b12x`
- `--max-model-len` above 262144 when `VLLM_ALLOW_LONG_MAX_MODEL_LEN` is not `1`
- missing PLE overlay or MTP overlay

Default occupancy is eight sequences. That pin held eight short streams with no drop. Do not raise `MAX_NUM_SEQS` above 8 without a new occupancy row.

## Verify

```bash
python3 -m unittest discover -s tests -q
python3 kit/render.py --check
VALIDATE_ONLY=1 ./run.sh
python3 bench_decode.py
```

After `./run.sh` is up, `GET /health` must be 200 and `GET /v1/models` must list `nvidia/Qwen3.8-Flash-Next-NVFP4`. Thinking-off smoke must not start `content` with chain-of-thought. `smoke_tools.py` must emit `get_weather`. `smoke_vision.py` must accept OpenAI `image_url` and must not return "is not a multimodal model". Greedy count 1 to 200 stays consecutive with thinking off. `python3 bench_decode.py` is the frozen 2x decode ruler. Its default phase is prose. Structured is optional and is not a decode score.

## Never touch

- Live HF tokens
- Floating `:nightly-aarch64` without the digest
- Hand-edited generated README / `run.sh` blocks
- Advertising a 1M needle that was not run
