# Qwen3.8-Flash-Next-NVFP4 · vLLM · 2× DGX Spark

Serve [nvidia/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4) across two NVIDIA DGX Spark (GB10) nodes at tensor-parallel 2.

Routed experts are ModelOpt NVFP4 W4A4. The PLE n-gram table is per-tensor FP8. MTP experts are FP8_BLOCK_SCALES with group size 128. Architecture class is `Qwen4ExpForConditionalGeneration`. Native context is 262,144. This recipe boots `--max-model-len` 262144 with in-band MTP-3. 1M is a lab ceiling, not a trained window. Community 1M YaRN on GB10 hangs on long prefills ([vLLM #54629](https://github.com/vllm-project/vllm/issues/54629)).

Stock `vllm/vllm-openai:v0.27.1` does not register `qwen4_exp`. The default image is official `vllm/vllm-openai:v0.30.0-aarch64@sha256:4864d46625cbc3307623e29ac742030655e27249feba7b97ec925ce4cc4dfb56` (`0.30.0`). v0.30 loads the MTP FP8_BLOCK_SCALES experts, selects FP8 PLE and fixes F01 natively. Three small overlays make c=1 greedy output bit-exact run to run (deterministic FlashInfer CUTLASS MoE finalize, a GDN fresh-prefill split fix and a stable QSA top-k order; `evidence/s15-determinism/SUMMARY.md`), and `serving.py` backports vLLM [#56067](https://github.com/vllm-project/vllm/pull/56067), which removes the O(n²) API-server CPU of streamed tool-call arguments. On v0.30 `run.sh` also passes `VLLM_PLE_CPU_OFFLOAD=0` and `VLLM_USE_BREAKABLE_CUDAGRAPH=0` to both ranks.

The rollback is the previous pin, `IMAGE=vllm/vllm-openai:nightly-aarch64@sha256:df871f170ee7070fbdce162bde08fb616e311570c948a620be0d4b33fe02f87b ./run.sh` (`0.28.1rc1.dev437+ge962733e0`), which mounts the pin overlay set. That nightly already selects MIXED_PRECISION FP8 PLE. Day-0 `qwen38-flash-next@3b0e188` and this nightly both fail MTP load until `docker/modelopt.py` remaps `mtp.layers.48` onto `mtp.layers.0` and dispatches `FP8_BLOCK_SCALES` to `Fp8MoEMethod`. The pinned fused PLE short-conv kernels also ignore the stride of their state indices, so requests prefilled together in one prefill-only step can decode garbage with zero MTP acceptance (F01). `docker/ple_ops.py` carries only the vLLM [#55375](https://github.com/vllm-project/vllm/pull/55375) stride hunks. `run.sh` bind-mounts the overlays named by `OVERLAYS` (see [docker/OVERLAYS.md](docker/OVERLAYS.md)).

Pinned snapshot: `fab0aecb760cec45227f6656abcaafa11abca87a`. Checkpoint credit is [NVIDIA ModelOpt](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4) (sychen52).

## Measured on 2× DGX Spark (L.A.I.L lab)

The table is `python3 bench_decode.py` at c=1 and c=2 on the seqs=8 default (v0.30.0 + overlays). Do not copy community tok/s into this table. Rollback-pin numbers live in `evidence/b1-pin-stride/`. Full gate pack, rulers and long-context numbers: `evidence/u2-v030-default/SUMMARY.md`.

<!-- BEGIN generated measured from recipe.yaml — edit recipe.yaml and run kit/render.py -->
Conditions: streamed greedy, thinking off, max_tokens 200 (prose ≈93 at EOS), 3-run median; max-num-seqs=8, kv auto, context 262144, MTP-3; v0.30.0 + determinism overlays (greedy output and prose acceptance are pinned run to run), recipe defaults at session 3.

| Phase | Concurrency | Decode tok/s (median per stream) | Aggregate tok/s | TTFT p50 |
|---|---|---:|---:|---:|
| prose | 1 | 38.7 | 38.7 | 0.16 s |
| prose (note 1) | 2 | 31.9 | 64.5 | 0.17 s |
| structured | 1 | 64.7 | 64.7 | 0.16 s |
| structured | 2 | 62.9 | 125.8 | 0.17 s |

1. Acceptance is 2.15 at c=2 against 2.37 at c=1, identical in all three session-3 frozen runs: a batch of two computes a different (but reproducible) greedy text than c=1. Compare prose cells only against the same base.
<!-- END generated measured -->

Native `max_position_embeddings` is 262144. This recipe serves that window. `run.sh` refuses `--max-model-len` above 1048576 unless `FORCE_UNSAFE_CTX=1`. Occupancy is eight sequences. seqs above 8 is unmeasured. Spark Arena TP=2 on this SHA used MTP-3, kv auto, context 262144, seqs 8.

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
docker pull vllm/vllm-openai:v0.30.0-aarch64@sha256:4864d46625cbc3307623e29ac742030655e27249feba7b97ec925ce4cc4dfb56
# rollback only:
docker pull vllm/vllm-openai:nightly-aarch64@sha256:df871f170ee7070fbdce162bde08fb616e311570c948a620be0d4b33fe02f87b
```

`run.sh` pulls a missing image. Each overlay names its generator in its R11 header; the v0.30 set is regenerated with `python3 docker/v030/apply_moe_finalize_overlay.py`, `apply_gdn_fresh_prefill_overlay.py`, `apply_qsa_topk_order_overlay.py` and `apply_api_overlays.py`, the pin set with `python3 docker/apply_ple_overlay.py`, `python3 docker/apply_mtp_fp8_overlay.py` and `python3 docker/apply_ple_stride_overlay.py` (see [docker/OVERLAYS.md](docker/OVERLAYS.md)). Do not use stock `vllm/vllm-openai:v0.27.1`.

## Quick start

On the head Spark (`spark1`):

```bash
chmod +x run.sh stop.sh
./run.sh
```

The head script copies itself and every overlay in `OVERLAYS` to `spark2`, starts the worker, waits 25s, then starts rank 0. While the head loads it polls the worker container and exits within 60 s if the worker dies. First boot is weight load plus warmup. `run.sh` forwards every serve setting, including `SNAPSHOT_SHA`, `HF_CACHE`, `MODEL`, `EXTRA_ARGS`, `EXTRA_ENV` and the overlay list, to the worker, shell-quoted. `ensure_weights` checks every shard in `model.safetensors.index.json` plus `config.json` and the tokenizer; containers then run with `HF_HUB_OFFLINE=1` and no HF token. Download uses `--revision "$SNAPSHOT_SHA"`. If `ORCHESTRATE=auto`, `ROLE=head`, `NNODES>1`, and SSH to `WORKER_HOST` fails, the script exits 1. It does not start a TP=2 head rank alone.

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

`smoke_thinking.py` must not start `content` with chain-of-thought. `smoke_tools.py` must emit `get_weather`. `smoke_vision.py` posts an OpenAI `image_url` and must not return HTTP 400 `is not a multimodal model`. The served processor caps images at 4194304 pixels (2048x2048, about 4k vision tokens; `MM_MAX_PIXELS`) instead of the checkpoint's 16777216, and caps video frames per frame. Pre-resize large images on the client anyway: the API server decodes, resizes and normalizes media on a single CPU thread shared by all requests, so a large photo stalls every concurrent image request behind it (G11). At most 8 images and 1 video per prompt. `smoke_count.py` must keep 1 to 200 consecutive with thinking off.

Stop both ranks from the head:

```bash
./stop.sh
```

`ORCHESTRATE=auto` (default) also stops the worker over SSH. If that probe fails, `stop.sh` prints to stderr and exits 1. `ORCHESTRATE=0` stops only the local container. The worker hostname is `WORKER_HOST`, else `.run-state/worker_host` next to `stop.sh`, else `spark2`.

## Defaults

<!-- BEGIN generated defaults from recipe.yaml — edit recipe.yaml and run kit/render.py -->
| Setting | Value |
|---|---|
| Image | `vllm/vllm-openai:v0.30.0-aarch64@sha256:4864d46625cbc3307623e29ac742030655e27249feba7b97ec925ce4cc4dfb56` |
| Model | `nvidia/Qwen3.8-Flash-Next-NVFP4` |
| `--tensor-parallel-size` / `--nnodes` | 2 / 2 |
| `--max-model-len` | 262144 (native window; 1048576 is a lab ceiling, not a trained window) |
| `--max-num-seqs` | 8 |
| `--max-num-batched-tokens` | 8192 |
| `--long-prefill-token-threshold` | 4800 (a short request during a 128k prefill waits ~5 s instead of ~61 s) |
| `--kv-cache-dtype` | `auto` |
| `--moe-backend` | `auto` (NVFP4 target resolves to FlashInfer CUTLASS on GB10; the MTP drafter inherits `auto`, which picks Triton for its 64x64-refined FP8 blocks. On the pin `run.sh` refuses anything but `auto`, and `b12x`/`flashinfer_b12x` everywhere) |
| Checkpoint | `fab0aecb760cec45227f6656abcaafa11abca87a` |
| Speculative | MTP-3 (`SPEC=mtp`; the drafter's SPEC_CONFIG `moe_backend` is `triton`: v0.30 honours it, the pin's V2 runner ignores it) |
| Overlays | `OVERLAYS=auto` → v0.30: `docker/v030/flashinfer_cutlass_moe.py docker/v030/gdn_attn.py docker/v030/qsa_indexer.py docker/v030/serving.py`; pin rollback: `docker/ple_layer.py docker/modelopt.py docker/ple_ops.py` (R11 headers, see `docker/OVERLAYS.md`) |
| v0.30 envs | `VLLM_PLE_CPU_OFFLOAD=0 VLLM_USE_BREAKABLE_CUDAGRAPH=0` on both ranks when IMAGE is the v0.30 digest (F33: offload pins 32 GiB of host memory per node) |
| Scheduling | `--async-scheduling` (`ASYNC_SCHEDULING=1`); `--per-request-spec-decode-metrics` is not passed (costs ~1 ms/step) |
| Vision | `--mm-processor-kwargs` images 65536..4194304 px, video `cap_pixels_per_frame`; `--limit-mm-per-prompt` image 8 / video 1; `--mm-processor-cache-gb 1` |
| Chat template | `chat_template_alias.jinja` (checkpoint template + `reasoning_effort` aliases high/max → xhigh, minimal → low; head only) |
| Indexer logits cap | `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=64` (vLLM #56457 GB10 long-prefill growth) |
| Oversize window | `VLLM_ALLOW_LONG_MAX_MODEL_LEN=0` (required only above 262144; this is not YaRN) |
| Tokenizers / tools / reasoning | auto / `qwen3_xml` / `qwen3` |
| Default thinking | `enable_thinking=false` |
| API | `http://<head>:8000/v1` |
| Container | `qwen38-flash-next-nvfp4` |
| Master port | 29523 |
<!-- END generated defaults -->

MTP draft experts are FP8_BLOCK_SCALES. The pin refines their 128x128 blocks to 64x64 to fit the TP-sharded intermediate size 320, and only Triton accepts that block. The pin's V2 runner ignores `SPEC_CONFIG` `moe_backend`, so the drafter inherits `--moe-backend`. `auto` is therefore the only value that serves both the NVFP4 target (FlashInfer CUTLASS on GB10) and the drafter, and `run.sh` refuses any other value on the pin digest. `b12x` and `flashinfer_b12x` are refused on every image (vLLM #57946, FlashInfer #5446).

`--max-model-len` 262144 is the native window. 1048576 is the refuse ceiling, not a needle result. vLLM derives 262144 from `max_position_embeddings` and exits on a longer window unless `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`. That env is not YaRN. Nightly already selects MIXED_PRECISION FP8 PLE, so `VLLM_PLE_FP8_CHECKPOINT` is not required and is not passed into the container.

`run.sh` refuses, before `VALIDATE_ONLY` exits:

- non-decimal or zero-padded integers (`MAX_NUM_SEQS=010` would be octal 8 to bash and 10 to vLLM), non-JSON `SPEC_CONFIG` / `COMPILATION_CONFIG`, an `IMAGE` without a digest
- an overlay that is missing, has no R11 header, was generated for another digest, or is a pin-only overlay (by content sha) on a non-pin image; on the pin, a missing `modelopt.py` or `ops/ple.py` overlay (`DIAGNOSTIC=1` overrides)
- a window above 1048576, `MAX_NUM_SEQS` above 8, compile `mode` other than 0 on the pin (Inductor copies the PLE table; vLLM #55272), and a changed `MAX_NUM_BATCHED_TOKENS`, any `LONG_PREFILL_TOKEN_THRESHOLD` other than `none`, or `indexer_kv_dtype` with `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=none` (`FORCE_UNSAFE_CTX=1` overrides)
- a window above 262144 without `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`
- `EXTRA_ARGS` that re-sets a flag `run.sh` builds and guards (`--compilation-config`/`-cc`/`-O`, `--speculative-config`, `--moe-backend`, `--max-num-batched-tokens`, `--max-num-seqs`, `--max-model-len`, `--enforce-eager`); use the matching variable
- `MOE_BACKEND` other than `auto` on the pin, and `b12x` / `flashinfer_b12x` anywhere (`FORCE_UNSAFE_MOE=1` overrides)
- `SPEC=none`, `use_local_argmax_reduction` without a `get_top_tokens` overlay or with probabilistic drafts, `enable_adaptive_verification`, `--enable-batch-sharded-sampling` without a `compute_logits_local` overlay, and `VLLM_BATCH_INVARIANT=1` (all fail at boot on this model; `DIAGNOSTIC=1` overrides)
- `rejection_sample_method=synthetic` unless `BENCH_ONLY=1`, which also binds the API to 127.0.0.1
- `EXTRA_ARGS` that sets `--long-prefill-token-threshold`, `--async-scheduling`, `--mm-processor-kwargs`, `--limit-mm-per-prompt`, `--mm-processor-cache-gb` or `--chat-template` (use `LONG_PREFILL_TOKEN_THRESHOLD`, `ASYNC_SCHEDULING`, `MM_*` or `CHAT_TEMPLATE`), `MM_MIN_PIXELS` above `MM_MAX_PIXELS`, and a missing `CHAT_TEMPLATE` file

`EXTRA_ENV='NCCL_DEBUG=INFO VLLM_COMPUTE_NANS_IN_LOGITS=1'` passes extra `-e KEY=VALUE` pairs to both ranks (no spaces inside a value).

`chat_template_alias.jinja` is the checkpoint `chat_template.jinja` plus `reasoning_effort` aliases (high/max → xhigh, minimal → low), generated by `tools/make_alias_template.py` and mounted read-only on the head (`CHAT_TEMPLATE=none` serves the checkpoint template). It honors `enable_thinking`. Tool calls use the card XML `<tool_call><function=...>` shape (`qwen3_xml`). Reasoning uses `qwen3`.

## Environment

```bash
export HEAD_IP=10.100.8.1
export WORKER_HOST=spark2
export IFACE=enp1s0f1np1
export HCA=rocep1s0f1
export PORT=8000
export MAX_MODEL_LEN=262144
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
- Image: [vLLM](https://hub.docker.com/r/vllm/vllm-openai) `v0.30.0-aarch64@sha256:4864d46625cbc3307623e29ac742030655e27249feba7b97ec925ce4cc4dfb56`; rollback `nightly-aarch64@sha256:df871f170ee7070fbdce162bde08fb616e311570c948a620be0d4b33fe02f87b`.
- MTP load on the pin: this lab. `docker/modelopt.py` remaps `mtp.layers.48` to `mtp.layers.0` and routes `FP8_BLOCK_SCALES` to `Fp8MoEMethod`. Stock vLLM has no RoutedExperts arm for that algo.

## License

Recipe scripts are MIT. The overlays under `docker/` and `docker/v030/` are Apache-2.0 vLLM source with patches. Model weights follow the source model license on Hugging Face.
