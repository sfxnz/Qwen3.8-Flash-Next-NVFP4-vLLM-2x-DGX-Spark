# Qwen3.8-Flash-Next-NVFP4 · vLLM · 2× DGX Spark

Serve [nvidia/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4) across two NVIDIA DGX Spark (GB10) nodes at tensor-parallel 2.

Routed experts are ModelOpt NVFP4 W4A4. The PLE n-gram table is per-tensor FP8. MTP experts are FP8_BLOCK_SCALES with group size 128. Architecture class is `Qwen4ExpForConditionalGeneration`. Native context is 262,144. This recipe boots `--max-model-len` 1048576 with in-band MTP-3. 1M is a lab ceiling, not a trained window. Community 1M YaRN on GB10 hangs on long prefills ([vLLM #54629](https://github.com/vllm-project/vllm/issues/54629)).

Stock `vllm/vllm-openai:v0.27.1` does not register `qwen4_exp`. The pin is official `vllm/vllm-openai:nightly-aarch64@sha256:df871f170ee7070fbdce162bde08fb616e311570c948a620be0d4b33fe02f87b` (`0.28.1rc1.dev437+ge962733e0`). Nightly already selects MIXED_PRECISION FP8 PLE. Day-0 `qwen38-flash-next@3b0e188` and this nightly both fail MTP load until `docker/modelopt.py` remaps `mtp.layers.48` onto `mtp.layers.0` and dispatches `FP8_BLOCK_SCALES` to `Fp8MoEMethod`. `run.sh` bind-mounts that overlay and `docker/ple_layer.py`.

Pinned snapshot: `fab0aecb760cec45227f6656abcaafa11abca87a`. Checkpoint credit is [NVIDIA ModelOpt](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4) (sychen52).

## Measured on 2× DGX Spark (L.A.I.L lab)

Decode is streamed greedy, thinking off, 200 completion tokens, 3-run median. The table is `python3 bench_decode.py` at c=1 and c=2 on the seqs=8 pin. Do not copy community tok/s into this table.

<!-- BEGIN generated measured from recipe.yaml — edit recipe.yaml and run kit/render.py -->
| Phase | Concurrency | Decode tok/s (median per stream) | Aggregate tok/s | TTFT p50 |
|---|---|---:|---:|---:|
| prose | 1 | 39.6 | 39.6 | 0.15 s |
| prose | 2 | 35.4 | 62.4 | 0.17 s |
| structured | 1 | 66.8 | 66.7 | 0.15 s |
| structured | 2 | 61.4 | 37.6 | 0.16 s |
<!-- END generated measured -->

Native `max_position_embeddings` is 262144. `run.sh` refuses `--max-model-len` above 1048576 unless `FORCE_UNSAFE_CTX=1`. Occupancy is eight sequences. That pin held eight short streams with no drop. seqs=4 and seqs=2 also passed. seqs above 8 is unmeasured. Spark Arena TP=2 on this SHA used MTP-3, kv auto, context 262144, seqs 8.

## Requirements

- Two DGX Sparks on the QSFP RoCE link (stock `10.100.8.1` / `10.100.8.2`)
- Docker + NVIDIA Container Toolkit on both nodes
- About 130 GiB free disk per node for the weights
- SSH from the head node to the worker (`spark2` in this lab)
- Exclusive GPUs. Do not start this recipe while another `--gpus all` serve is up.

```bash
hf auth login
# or: export HF_TOKEN=hf_...
```

## Image

On both nodes:

```bash
docker pull vllm/vllm-openai:nightly-aarch64@sha256:df871f170ee7070fbdce162bde08fb616e311570c948a620be0d4b33fe02f87b
```

`run.sh` pulls a missing image. Regenerate overlays with `python3 docker/apply_ple_overlay.py` and `python3 docker/apply_mtp_fp8_overlay.py`. Do not use stock `vllm/vllm-openai:v0.27.1`.

## Quick start

On the head Spark (`spark1`):

```bash
chmod +x run.sh stop.sh
./run.sh
```

The head script copies itself, `docker/ple_layer.py`, and `docker/modelopt.py` to `spark2`, starts the worker, waits 25s, then starts rank 0. First boot is weight load plus warmup. `run.sh` forwards `SNAPSHOT_SHA`, `HF_CACHE`, `MODEL`, `PLE_OVERLAY`, and `MTP_OVERLAY` to the worker. Download uses `--revision "$SNAPSHOT_SHA"`. If `ORCHESTRATE=auto`, `ROLE=head`, `NNODES>1`, and SSH to `WORKER_HOST` fails, the script exits 1. It does not start a TP=2 head rank alone.

If SSH is not set up, start the worker yourself, then the head:

```bash
# spark2
ROLE=worker ./run.sh

# spark1
ROLE=head ./run.sh
```

Text smoke (thinking off):

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "nvidia/Qwen3.8-Flash-Next-NVFP4",
    "messages": [{"role": "user", "content": "Say hello in one sentence."}],
    "max_tokens": 64,
    "temperature": 0,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

Correctness probes against the live API:

```bash
python3 smoke_thinking.py
python3 smoke_tools.py
python3 smoke_vision.py
python3 smoke_count.py
python3 bench_decode.py
```

`smoke_thinking.py` must not start `content` with chain-of-thought. `smoke_tools.py` must emit `get_weather`. `smoke_vision.py` posts an OpenAI `image_url` and must not return HTTP 400 `is not a multimodal model`. `smoke_count.py` must keep 1 to 200 consecutive with thinking off.

Stop both ranks from the head:

```bash
./stop.sh
```

`ORCHESTRATE=auto` (default) also stops the worker over SSH. If that probe fails, `stop.sh` prints to stderr and exits 1. `ORCHESTRATE=0` stops only the local container. The worker hostname is `WORKER_HOST`, else `$PWD/.run-state/worker_host`, else `spark2`.

## Defaults

<!-- BEGIN generated defaults from recipe.yaml — edit recipe.yaml and run kit/render.py -->
| Setting | Value |
|---|---|
| Image | `vllm/vllm-openai:nightly-aarch64@sha256:df871f170ee7070fbdce162bde08fb616e311570c948a620be0d4b33fe02f87b` |
| Model | `nvidia/Qwen3.8-Flash-Next-NVFP4` |
| `--tensor-parallel-size` / `--nnodes` | 2 / 2 |
| `--max-model-len` | 1048576 (native 262144; 1M is a lab ceiling, not a trained window) |
| `--max-num-seqs` | 8 |
| `--max-num-batched-tokens` | 8192 |
| `--kv-cache-dtype` | `auto` |
| `--moe-backend` | `auto` (NVFP4 walks flashinfer then marlin; MTP FP8_BLOCK_SCALES uses triton. `run.sh` refuses `marlin` and `flashinfer_cutlass`) |
| Checkpoint | `fab0aecb760cec45227f6656abcaafa11abca87a` |
| Speculative | MTP-3 (`SPEC=mtp`, draft MoE `triton`; NVFP4 experts stay `auto`) |
| PLE overlay | `docker/ple_layer.py` (qwen4_exp MIXED_PRECISION FP8) plus `docker/modelopt.py` MTP FP8_BLOCK_SCALES |
| Oversize window | `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` (native 262144; this is not YaRN) |
| Tokenizers / tools / reasoning | auto / `qwen3_xml` / `qwen3` |
| Default thinking | `enable_thinking=false` |
| API | `http://<head>:8000/v1` |
| Container | `qwen38-flash-next-nvfp4` |
| Master port | 29523 |
<!-- END generated defaults -->

MTP draft experts are FP8_BLOCK_SCALES. A global `--moe-backend marlin` is still refused. This recipe defaults to `--moe-backend auto`. Nightly refined MTP block scales from 128x128 to 64x64 to fit TP-sharded intermediate size 320, then used Triton. `run.sh` refuses `marlin` and `flashinfer_cutlass`.

`--max-model-len` 1048576 is the refuse ceiling, not a needle result. vLLM derives 262144 from `max_position_embeddings` and exits unless `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`. That env is not YaRN. `run.sh` refuses a window above 1048576, `MAX_NUM_SEQS` above 8, `MOE_BACKEND=marlin` or `flashinfer_cutlass`, a missing PLE or MTP overlay, and a 1M window without `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`. Nightly already selects MIXED_PRECISION FP8 PLE, so `VLLM_PLE_FP8_CHECKPOINT` is not required and is not passed into the container. `FORCE_UNSAFE_CTX=1` / `FORCE_UNSAFE_MOE=1` override the other guards.

There is no extra Jinja file. The checkpoint `chat_template.jinja` honors `enable_thinking`. Tool calls use the card XML `<tool_call><function=...>` shape (`qwen3_xml`). Reasoning uses `qwen3`.

## Environment

```bash
export HEAD_IP=10.100.8.1
export WORKER_HOST=spark2
export IFACE=enp1s0f1np1
export HCA=rocep1s0f1
export PORT=8000
export MAX_MODEL_LEN=1048576
export MAX_NUM_SEQS=8
```

Pin `NCCL_IB_HCA`. GB10 exposes four HCAs and two of them are DOWN. Unpinned NCCL picks a dead one and fails with `unhandled system error`. Some Spark cookbooks use `enp1s0f0np0` / `rocep1s0f0`. This lab uses `enp1s0f1np1` / `rocep1s0f1`.

## Logs

```bash
docker logs -f qwen38-flash-next-nvfp4
ssh spark2 docker logs -f qwen38-flash-next-nvfp4
```

## Credits

- Checkpoint: [NVIDIA ModelOpt](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4) (sychen52), MIXED_PRECISION NVFP4 plus FP8 PLE plus MTP FP8_BLOCK_SCALES at `fab0aec`.
- Image: [vLLM](https://hub.docker.com/r/vllm/vllm-openai) `nightly-aarch64@sha256:df871f170ee7070fbdce162bde08fb616e311570c948a620be0d4b33fe02f87b`.
- MTP load: this lab. `docker/modelopt.py` remaps `mtp.layers.48` to `mtp.layers.0` and routes `FP8_BLOCK_SCALES` to `Fp8MoEMethod`. Stock vLLM has no RoutedExperts arm for that algo.

## License

Recipe scripts are MIT. `docker/ple_layer.py` and `docker/modelopt.py` are Apache-2.0 vLLM source with overlays. Model weights follow the source model license on Hugging Face.
