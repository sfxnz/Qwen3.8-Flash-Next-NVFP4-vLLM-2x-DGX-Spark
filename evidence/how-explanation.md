# How this Qwen3.8-Flash-Next NVFP4 TP=2 serve occupies GB10

## Overview

The live recipe serves `nvidia/Qwen3.8-Flash-Next-NVFP4` snapshot `fab0aec` across spark1 and spark2 at tensor-parallel 2. The pin is `vllm/vllm-openai:nightly-aarch64@sha256:df871f170ee7070fbdce162bde08fb616e311570c948a620be0d4b33fe02f87b`. Stock `v0.27.1` does not register `Qwen4ExpForConditionalGeneration`. MIXED_PRECISION PLE is stock on this nightly. MTP FP8_BLOCK_SCALES needs `docker/modelopt.py`.

This occupancy run starts from the boot pin already up at `http://127.0.0.1:8000`. `GET /health` is 200. `GET /v1/models` lists the served name with `max_model_len` 1048576. Inspected args are `--max-num-seqs 8`, `--kv-cache-dtype auto`, `--gpu-memory-utilization 0.80`, MTP-3 with draft MoE `triton`, global `--moe-backend auto`. The live container started `2026-09-05T11:48:45Z`.

## Key concepts

- **Head / worker.** Rank 0 on spark1 owns `0.0.0.0:8000`. Rank 1 on spark2 is `--headless`. Both containers are named `qwen38-flash-next-nvfp4`.
- **GB10 UMA.** 121.7 GiB unified memory per Spark. Read leftover with `free -h`. Do not use `nvidia-smi` VRAM.
- **Mixed quant.** Routed experts are NVFP4 W4A4. The PLE n-gram table is FP8. MTP experts are FP8_BLOCK_SCALES group 128. Attention and vision stay BF16.
- **KV pool.** This boot sized 2,025,904 tokens, 28.85 GiB. Maximum concurrency for a 1,048,576-token request is 1.93x. Short agentic requests are admission-limited by `--max-num-seqs`, not by that 1M ceiling.
- **CUDA graphs.** This boot captured 5 FULL graphs plus 4 decode graphs, 0.48 GiB, matching seqs=8. Raising seqs recaptures more graphs. It does not grow the KV pool.
- **Thinking off.** `--default-chat-template-kwargs '{"enable_thinking": false}'`. The card template seeds an empty `<think></think>` when thinking is off. The `qwen3` reasoning parser should keep that out of `content`.
- **Tools / vision.** Parser `qwen3_xml`, auto tool choice, native ViT. Multi-modal warmup completed in 29.7s on this boot.

## How it works

`run.sh` on the head copies itself and the PLE overlay to spark2, starts rank 1, waits 25s for NCCL, then starts rank 0. NCCL is pinned to `enp1s0f1np1` / `rocep1s0f1` because two of four GB10 HCAs are DOWN.

The occupancy metric is concurrent short streams the scheduler will admit without dropping one. The frozen ruler is `python3 bench_decode.py` (prose + structured, c=1 and c=2, 3-run median, thinking off, 200 completion tokens). Extra `--concurrency 8` is the occupancy probe. A wave that loses a stream is a fail, not a faster median. Engine age moves prose a few tok/s, so a restart must recapture before it can count as a win.

Correctness is a gate, not a tok/s cell. Thinking-off must not start `content` with chain-of-thought. Tools must emit `get_weather`. Vision must accept OpenAI `image_url` and must not return "is not a multimodal model". Greedy count 1 to 200 must stay consecutive with thinking off.

Occupancy is already pinned at 8. seqs=2 and seqs=4 also passed. `run.sh` refuses a value above 8 unless `FORCE_UNSAFE_CTX=1`. Spark Arena TP=2 on this SHA used seqs=8 at native 262144. This recipe's 1M window is a lab ceiling. The KV pool still holds many short requests. UMA leftover on this boot is tight (spark1 about 10 GiB available with 5.5 GiB swap, spark2 about 15 GiB available with 3.4 GiB swap). Encode the first failing setting as the `run.sh` refuse-guard.

## Where things live

- `run.sh` / `stop.sh` / `recipe.yaml` / `kit/render.py`. Pins and generated blocks.
- `docker/ple_layer.py`. Mixed-quant PLE resolver.
- `bench_decode.py`. Frozen 2x decode ruler.
- `smoke_thinking.py`, `smoke_tools.py`, `smoke_vision.py`, `smoke_count.py`. Correctness probes.
- `evidence/boot/`. First successful TP=2 boot.
- `evidence/baseline/`. seqs=2 probes and the first frozen SUMMARY.
- `evidence/iter-h2/`. Published occupancy pin at seqs=8.
- `evidence/opt-baseline/`. Current-boot recapture of the frozen ruler. No recipe change.
- `evidence/decision.tsv` / `evidence/trail.tsv`. Occupancy keep/revert memory.

## Gotchas

- Native `max_position_embeddings` is 262144. 1048576 needs `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` and is not YaRN. Community 1M YaRN on GB10 hangs on long prefills (vLLM #54629).
- Do not treat 1M as a quality pin. Occupancy here is short-stream admission.
- Global `--moe-backend marlin` exits on unquantized MTP MoE. This boot selected FLASHINFER_CUTLASS for NVFP4 experts (`nvfp4.py`). Explicit `marlin` still fails.
- Occupancy pin is `MAX_NUM_SEQS=8`. `run.sh` refuses a value above 8 unless `FORCE_UNSAFE_CTX=1`. seqs=2 and seqs=4 also passed. seqs above 8 is unmeasured.
- Official NVIDIA MIXED_PRECISION (`nvidia/Qwen3.8-Flash-Next-NVFP4@fab0aec`) needs `docker/modelopt.py`. Day-0 and stock nightly both miss `w2_weight_scale_inv` on `mtp.layers.48`. The overlay remaps that prefix onto `mtp.layers.0` and dispatches `Fp8MoEMethod` block 128.
- Exclusive GPUs. Do not start a second `--gpus all` serve. Replacing this recipe's own container is the occupancy loop.
