# FP8 dense allow-list overlay (P4-1, v0.30.0)

`docker/v030/modelopt.py` is stock v0.30.0 `vllm/model_executor/layers/quantization/modelopt.py`
(sha256 `41e00493…1e22`, image `sha256:4864d466…4dfb56`) plus three hunks, rendered by
`docker/v030/apply_fp8_dense_overlay.py`. Do not hand-edit it. The R11 header is at the top of the file.

It is **off by default**. With `VLLM_QWEN38_FP8_DENSE` unset, empty, `0` or `off`, the helper returns
`None` and both call sites return stock `UnquantizedLinearMethod()`. Logic is then identical to stock.

## What it does

`ModelOptMixedPrecisionConfig.get_quant_method` returns `UnquantizedLinearMethod` for BF16 layers in two
places: layers in `exclude_modules` (GDN, QSA, MTP attention) and layers missing from `quantized_layers`
(PLE `kv_proj`). With the env set, a `LinearBase` whose vLLM prefix matches the allow-list instead gets
vLLM's stock online method from `online/fp8.py`:

| `VLLM_QWEN38_FP8_DENSE` | Method | Weights | Activations | Kernel on sm_121 (auto) |
|---|---|---|---|---|
| `per_block` | `Fp8PerBlockOnlineLinearMethod` (:260) | E4M3, fp32 128×128 block scales | dynamic 1×128 group | first supported of DeepGEMM/CUTLASS-block/B12x/**Marlin**/Triton. Check `Selected … for Fp8PerBlockOnlineLinearMethod` in run.log |
| `ptpc` | `Fp8PtpcOnlineLinearMethod` (:340) | E4M3, fp32 per-row scale | dynamic per-token | `CutlassFP8ScaledMMLinearKernel` (sm120 W8A8). FlashInfer FP8 needs per-tensor scales, so it is skipped. Marlin is refused (W8A16 would silently drop A8) |

The weights are quantised at load, layer by layer (`uses_meta_device`). No derived checkpoint is needed.

### The W8A16 arm (plan arm i)

`VLLM_QWEN38_FP8_DENSE=per_block`, plus Marlin scoped to block FP8 only:

```
--kernel-config '{"linear_backend_per_quant": {"fp8_block_w8a8": "marlin"}}'
```

`--linear-backend marlin` also exists in v0.30 (`config/kernel.py:300`, `engine/arg_utils.py:1680`). It is
global, though, and any other layer type without a Marlin kernel falls back with a warning. Prefer the
per-quant key: `_get_linear_backend(quantization="fp8_block_w8a8")` in `kernels/linear/__init__.py:238`.
`MarlinFP8ScaledMMLinearKernel` accepts `kFp8Static128BlockSym` (`scaled_mm/marlin.py:57`). The recipe's
MoE refuse-guard is about `MOE_BACKEND=marlin`. It does not cover linear kernels.

## Allow-list

`VLLM_QWEN38_FP8_DENSE_ALLOW` holds comma-separated Python regexes, applied with `re.search` against the vLLM
prefix. Empty means the default below (tensors stay split, C7):

| Regex | Runtime prefix (example) | Layers | Per-rank (N, K) |
|---|---|---|---|
| `(?:^\|\.)linear_attn\.in_proj_qkvz$` | `language_model.model.layers.0.linear_attn.in_proj_qkvz` | 36 | (8192, 2560) |
| `(?:^\|\.)linear_attn\.out_proj$` | `…layers.0.linear_attn.out_proj` | 36 | (2560, 3072) |
| `(?:^\|\.)self_attn\.qkv_proj$` | `…layers.3.self_attn.qkv_proj`, `mtp.layers.48.self_attn.qkv_proj` | 12 + 1 MTP | (6656, 2560) |
| `(?:^\|\.)self_attn\.o_proj$` | `…layers.3.self_attn.o_proj`, `mtp.layers.48.self_attn.o_proj` | 12 + 1 MTP | (2560, 3072) |
| `(?:^\|\.)ple\.kv_proj$` | `language_model.model.layers.1.ple.kv_proj` (`disable_tp`, replicated) | 1 | (12800, 2560) |

These stay BF16 because nothing in the list matches them: `in_proj_ba` (N=48), `indexer.index_qk_proj`,
shared expert, router `mlp.gate`, hyper-connection, `lm_head` (`ParallelLMHead` is not a `LinearBase`),
MTP `fc_embedding`/`fc_hidden`, and the ViT. `tests/test_fp8_overlay.py` checks all of this against the
real checkpoint index.

## Memory (from safetensors headers, TP=2, per rank)

`python3 docker/v030/apply_fp8_dense_overlay.py --savings`:

```
module           layers   bf16 MB   blk MB  ptpc MB
in_proj_qkvz         36    1509.9    755.2    756.2
kv_proj               1      65.5     32.8     32.8
mtp.o_proj            1      15.7      7.9      7.9
mtp.qkv_proj          1      34.1     17.0     17.1
o_proj               12     188.7     94.4     94.5
out_proj             36     566.2    283.2    283.5
qkv_proj             12     408.9    204.5    204.8
total                99    2789.2   1394.9   1396.7
saving per rank (per_block): 1.394 GB = 1.299 GiB
saving per rank (ptpc): 1.393 GB = 1.297 GiB
```

The plan's §7 ledger says 1.42 GiB. The header-exact figure is **1.30 GiB per rank**. The matching
`GPU_MEMORY_UTILIZATION` offset is 1.299 GiB / 121.7 GiB MemTotal = **−0.011**, not −0.012. Without that
offset, the KV pool grows by about 1.30/1.42 × 107k ≈ 98k tokens. Bytes streamed per verify drop by the
same 1.39 GB (the MTP rows are read k times per step).

## Why an overlay and not stock `--quantization-config`

v0.30 can already compose online quantisation onto a checkpoint config
(`--quantization-config '{"targets": {"re:…": "fp8_per_block"}}'`: `weight_utils.py:255`,
`base_config.py:285`). That path is not used here, for two reasons:

- The MTP draft `ModelConfig` is built with `quantization=` only (`config/speculative.py:1288`), so the
  drafter's quant config never receives `online_quantization_config`. The MTP attention would stay BF16.
- `targets` needs one shorthand per pattern, and a mixed-shard fused layer raises an error.

The overlay acts inside `ModelOptMixedPrecisionConfig`, so it reaches the target and the drafter the
same way. Do not set both mechanisms at once: `resolve_quant_method` raises when an online target hits
a layer that is already quantised.

## Wiring (other owners)

- Mount `docker/v030/modelopt.py` at
  `/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/modelopt.py` on **both**
  ranks, and only on the v0.30 image. Its header digest must equal `IMAGE` (C1 check).
- Pass `VLLM_QWEN38_FP8_DENSE` (and optionally `_ALLOW`) to both ranks through `EXTRA_ENV`. vLLM logs
  "Unknown vLLM environment variable detected" for these names. That is a warning, unless the serve uses
  `--fail-on-environ-validation`.
- Regenerate with
  `python3 docker/v030/apply_fp8_dense_overlay.py [--src <stock modelopt.py>]`. Without `--src` the script
  runs `docker create` + `cp` against the pinned image. `--check` exits 1 when the file is stale.

## Gates (plan P4-1)

This is a class B (precision) change: T0–T3 + V. T1-B is checked per domain **and** T1-D. Every GEMM must
reach ≥150 GB/s in nsys. If a gate fails, step down the fallback ladder: per-channel (`ptpc`) →
per-block-128 → GDN q/k rows BF16 (narrow `_ALLOW`) → off. Confirm in run.log that
`qwen38-fp8-dense: mode=… allow=(…)` and the `Selected <Kernel> for Fp8…OnlineLinearMethod` lines are
present, and that the arm uses the kernel it intends to use.

## Not verified yet

Nothing here has booted. The host has no torch, and GPU runs are out of scope for this change. The tests
exercise the shipped helper code with stub vLLM types.

Things to confirm on the first boot:
- Online layerwise loading for GDN `in_proj_qkvz`. It is a 4-shard `MergedColumnParallelLinear` fed from
  `in_proj_qkv` and `in_proj_z`.
- Online loading inside the MTP drafter's load pass.
- `CUTLASS_BLOCK_FP8_SUPPORTED` on sm_121.

## Measured (session 5, `evidence/k-sweep-precision/`)

**Boot:** first-boot items confirmed on both ranks.
- Log lines: `qwen38-fp8-dense: mode=per_block`, `Applied linear backend override for 'fp8_block_w8a8': 'marlin'`, `Selected MarlinFP8ScaledMMLinearKernel for Fp8PerBlockOnlineLinearMethod`.
- Online loading works for the 4-shard `in_proj_qkvz` and for the MTP drafter.
- `ptpc` selects `CutlassFP8ScaledMMLinearKernel`, so CUTLASS W8A8 runs on sm_121.

**Speed:** per_block, compared with the lossless default.

| Metric | Default | per_block |
|---|---|---|
| structured c1 ms/step | 54.62 | 47.50 (−13.0%) |
| structured c8 ms/step | 75.26 | 69.74 (−7.3%) |
| frozen prose c1 tok/s | 44.0 | 57.1 |
| frozen structured c1 tok/s | 72.2 | 84.2 |
| weights per rank | 63.94 GiB | 62.66 GiB (−1.28) |

**Quality gates:**
- **T1-B: fails** (top-1 0.943, KL 0.032; de, es, ja, multi and tools are above 2× the mean).
- **Passes:** T1-D (median divergence 17.0 against a floor of 14.25, KL 0.004), paired T2 (GSM8K −1.2 pp at p 0.25, IFEval +1.7 pp, tools 60/60, JSON 30/30), T3 to 64k and V.
- **The T1-B gate is saturated on this model:** FP8 on a single 65 MB layer already scores 94.6% top-1.

**How it ships:** as the opt-in `lab-fp8-dense` profile. Set `FP8_DENSE=per_block ./run.sh`; `run.sh` then mounts this overlay and passes the env and the Marlin `--kernel-config`. It is not the default. Promotion needs owner sign-off plus T4.
