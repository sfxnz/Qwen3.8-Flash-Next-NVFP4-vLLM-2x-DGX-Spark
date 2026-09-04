# Qwen3.8-Flash-Next-NVFP4 · vLLM · 2× DGX Spark

Serve [RadixArk/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4) across two NVIDIA DGX Spark (GB10) nodes at tensor-parallel 2.

Routed experts are ModelOpt NVFP4 W4A4. The 51B PLE n-gram table is FP8. Attention, MTP, vision, and the rest stay BF16. Architecture class is `Qwen4ExpForConditionalGeneration`. Native context is 262,144. This recipe boots `--max-model-len` 1048576 with in-band MTP-3. 1M is a lab ceiling, not a trained window. Community 1M YaRN on GB10 hangs on long prefills ([vLLM #54629](https://github.com/vllm-project/vllm/issues/54629)).

Stock `vllm/vllm-openai:v0.27.1` does not register `qwen4_exp`. Use the official day-0 image `vllm/vllm-openai:qwen38-flash-next@sha256:3b0e188ffceb3d07e09c3cb5215433a0020eacf02d7f882ed3a8bfd15454477e`. That image still needs the mixed-quant PLE resolver. `*.ple.*` is excluded from ModelOpt NVFP4 while the tables are FP8. `run.sh` bind-mounts `docker/ple_layer.py` and sets `VLLM_PLE_FP8_CHECKPOINT=1`.

Pinned snapshot: `7b719225242aacd3dbd3f9407468c2ee9a9d2594`.

## Measured on 2× DGX Spark (L.A.I.L lab)

Decode is streamed greedy, thinking off, 200 completion tokens, 3-run median. The table is `python3 bench_decode.py` at c=1 and c=2 on the seqs=8 pin. Do not copy community tok/s into this table.

<!-- BEGIN generated measured from recipe.yaml — edit recipe.yaml and run kit/render.py -->
| Phase | Concurrency | Decode tok/s (median per stream) | Aggregate tok/s | TTFT p50 |
|---|---|---:|---:|---:|
| prose | 1 | 39.5 | 39.4 | 0.22 s |
| prose | 2 | 34.9 | 64.4 | 0.23 s |
| structured | 1 | 66.6 | 66.6 | 0.22 s |
| structured | 2 | 65.2 | 130.4 | 0.23 s |
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
docker pull vllm/vllm-openai:qwen38-flash-next@sha256:3b0e188ffceb3d07e09c3cb5215433a0020eacf02d7f882ed3a8bfd15454477e
```

`run.sh` pulls a missing image. Regenerate the PLE overlay from that digest with `python3 docker/apply_ple_overlay.py`. Do not use stock `vllm/vllm-openai:v0.27.1`.

## Quick start

On the head Spark (`spark1`):

```bash
chmod +x run.sh stop.sh
./run.sh
```

The head script copies itself and `docker/ple_layer.py` to `spark2`, starts the worker, waits 25s, then starts rank 0. First boot is weight load plus warmup. `run.sh` forwards `SNAPSHOT_SHA`, `HF_CACHE`, `MODEL`, and `PLE_OVERLAY` to the worker. Download uses `--revision "$SNAPSHOT_SHA"`. If `ORCHESTRATE=auto`, `ROLE=head`, `NNODES>1`, and SSH to `WORKER_HOST` fails, the script exits 1. It does not start a TP=2 head rank alone.

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
    "model": "RadixArk/Qwen3.8-Flash-Next-NVFP4",
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
| Image | `vllm/vllm-openai:qwen38-flash-next@sha256:3b0e188ffceb3d07e09c3cb5215433a0020eacf02d7f882ed3a8bfd15454477e` |
| Model | `RadixArk/Qwen3.8-Flash-Next-NVFP4` |
| `--tensor-parallel-size` / `--nnodes` | 2 / 2 |
| `--max-model-len` | 1048576 (native 262144; 1M is a lab ceiling, not a trained window) |
| `--max-num-seqs` | 8 |
| `--max-num-batched-tokens` | 8192 |
| `--kv-cache-dtype` | `auto` |
| `--moe-backend` | `auto` (NVFP4 walks flashinfer then marlin; MTP BF16 needs auto/triton. `run.sh` refuses `marlin` and `flashinfer_cutlass`) |
| Checkpoint | `7b719225242aacd3dbd3f9407468c2ee9a9d2594` |
| Speculative | MTP-3 (`SPEC=mtp`, draft MoE `triton`; NVFP4 experts stay `auto`) |
| PLE overlay | `docker/ple_layer.py` with `VLLM_PLE_FP8_CHECKPOINT=1` |
| Oversize window | `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` (native 262144; this is not YaRN) |
| Tokenizers / tools / reasoning | auto / `qwen3_xml` / `qwen3` |
| Default thinking | `enable_thinking=false` |
| API | `http://<head>:8000/v1` |
| Container | `qwen38-flash-next-nvfp4` |
| Master port | 29523 |
<!-- END generated defaults -->

MTP draft experts stay BF16. A global `--moe-backend marlin` exits with `moe_backend='marlin' is not supported for unquantized MoE`. This recipe defaults to `--moe-backend auto`. This boot selected `FLASHINFER_CUTLASS` for both NVFP4 experts and unquantized MTP. Explicit `marlin` still fails on MTP. `run.sh` refuses `marlin` and `flashinfer_cutlass`.

`--max-model-len` 1048576 is the refuse ceiling, not a needle result. The day-0 image derives 262144 from `max_position_embeddings` and exits unless `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`. That env is not YaRN. `run.sh` refuses a window above 1048576, `MAX_NUM_SEQS` above 8, `MOE_BACKEND=marlin` or `flashinfer_cutlass`, a missing `VLLM_PLE_FP8_CHECKPOINT=1` overlay, and a 1M window without `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`. `FORCE_UNSAFE_CTX=1` / `FORCE_UNSAFE_MOE=1` override the other guards.

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

## License

Recipe scripts are MIT. `docker/ple_layer.py` is Apache-2.0 vLLM source with a three-line overlay. Model weights follow the source model license on Hugging Face.
