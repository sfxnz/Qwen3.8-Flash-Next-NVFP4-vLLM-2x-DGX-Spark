# How this Qwen3.8-Flash-Next NVFP4 TP=2 serve occupies GB10

## Overview

The live recipe serves `RadixArk/Qwen3.8-Flash-Next-NVFP4` snapshot `7b719225` across spark1 and spark2 at tensor-parallel 2. The official day-0 image is `vllm/vllm-openai:qwen38-flash-next@sha256:3b0e188ffceb3d07e09c3cb5215433a0020eacf02d7f882ed3a8bfd15454477e`. Stock `v0.27.1` does not register `Qwen4ExpForConditionalGeneration`. Mixed ModelOpt NVFP4 plus FP8 PLE still needs `docker/ple_layer.py` with `VLLM_PLE_FP8_CHECKPOINT=1`.

This occupancy run starts from the boot pin already up at `http://127.0.0.1:8000`. `GET /health` is 200. `GET /v1/models` lists the served name with `max_model_len` 1048576. Inspected args are `--max-num-seqs 2`, `--kv-cache-dtype auto`, `--gpu-memory-utilization 0.80`, MTP-3 with draft MoE `triton`, global `--moe-backend auto`.

## Key concepts

- **Head / worker.** Rank 0 on spark1 owns `0.0.0.0:8000`. Rank 1 on spark2 is `--headless`. Both containers are named `qwen38-flash-next-nvfp4`.
- **GB10 UMA.** 121.7 GiB unified memory per Spark. Read leftover with `free -h`. Do not use `nvidia-smi` VRAM.
- **Mixed quant.** Routed experts are NVFP4 W4A4. The PLE n-gram table is FP8. Attention, MTP, vision, and the rest stay BF16. MTP experts reject a global `--moe-backend marlin`.
- **KV pool.** Boot log sized 2,063,077 tokens, 30.08 GiB. Maximum concurrency for a 1,048,576-token request is 1.97x. Short agentic requests are admission-limited by `--max-num-seqs`, not by that 1M ceiling.
- **CUDA graphs.** Decode capture was 2 graphs, matching seqs=2. Raising seqs recaptures more graphs. It does not grow the KV pool.
- **Thinking off.** `--default-chat-template-kwargs '{"enable_thinking": false}'`. The card template seeds an empty `<think></think>` when thinking is off. The `qwen3` reasoning parser should keep that out of `content`.
- **Tools / vision.** Parser `qwen3_xml`, auto tool choice, native ViT. Multi-modal warmup completed in 19s on this boot.

## How it works

`run.sh` on the head copies itself and the PLE overlay to spark2, starts rank 1, waits 25s for NCCL, then starts rank 0. NCCL is pinned to `enp1s0f1np1` / `rocep1s0f1` because two of four GB10 HCAs are DOWN.

The occupancy metric is concurrent short streams the scheduler will admit without dropping one. The frozen ruler is `python3 bench_decode.py` (prose + structured, c=1 and c=2, 3-run median, thinking off, 200 completion tokens). A wave that loses a stream is a fail, not a faster median.

Correctness is a gate, not a tok/s cell. Thinking-off must not start `content` with chain-of-thought. Tools must emit `get_weather`. Vision must accept OpenAI `image_url` and must not return "is not a multimodal model". Greedy count 1 to 200 must stay consecutive with thinking off.

Occupancy knobs to try after the baseline is green are 2 (live), then 4, then 8. Spark Arena TP=2 on this SHA used seqs=8 at native 262144. This recipe's 1M window is a lab ceiling. The KV pool still holds many short requests. UMA leftover on the live boot is already tight (spark1 about 10 GiB available with 5.5 GiB swap, spark2 about 14 GiB available with 2.6 GiB swap). Raise seqs with `FORCE_UNSAFE_CTX=1` until a keep or revert is logged. Encode the first failing setting as the `run.sh` refuse-guard.

## Where things live

- `run.sh` / `stop.sh` / `recipe.yaml` / `kit/render.py`. Pins and generated blocks.
- `docker/ple_layer.py`. Mixed-quant PLE resolver.
- `bench_decode.py`. Frozen 2x decode ruler.
- `smoke_thinking.py`, `smoke_tools.py`, `smoke_vision.py`, `smoke_count.py`. Correctness probes.
- `evidence/boot/`. First successful TP=2 boot.
- `evidence/baseline/`. seqs=2 probes and the first frozen SUMMARY.
- `evidence/decision.tsv` / `evidence/trail.tsv`. Occupancy keep/revert memory.

## Gotchas

- Native `max_position_embeddings` is 262144. 1048576 needs `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` and is not YaRN. Community 1M YaRN on GB10 hangs on long prefills (vLLM #54629).
- Do not treat 1M as a quality pin. Occupancy here is short-stream admission.
- Global `--moe-backend marlin` exits on unquantized MTP MoE. This boot selected FLASHINFER_CUTLASS for both NVFP4 experts and unquantized MTP. Explicit `marlin` still fails.
- Occupancy pin is `MAX_NUM_SEQS=8`. `run.sh` refuses a value above 8 unless `FORCE_UNSAFE_CTX=1`. seqs=2 and seqs=4 also passed. seqs above 8 is unmeasured.
- Exclusive GPUs. Do not start a second `--gpus all` serve. Replacing this recipe's own container is the occupancy loop.
