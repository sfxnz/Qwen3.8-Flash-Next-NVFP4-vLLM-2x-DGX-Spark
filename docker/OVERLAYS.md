# Overlays

An overlay is a whole upstream vLLM `.py` file, patched, that `run.sh` bind-mounts read-only over the
same file inside the image on both ranks. Every overlay is generated. Never hand-edit one.

## R11 header

Every overlay starts with this block. `run.sh` reads it; nothing else is configured per file.

```
# R11-OVERLAY
# base_image_digest: sha256:<digest of the image the stock file came from>
# upstream_file: vllm/<path inside site-packages>.py
# upstream_file_sha256: <sha256 of that stock file>
# upstream_PR: <upstream PR the hunks mirror, or none (reason)>
# generator: docker/<generator>.py (do not hand-edit)
```

`run.sh`, for each file in `OVERLAYS`:

- refuses a missing file or a file without `base_image_digest` / `upstream_file`;
- refuses `base_image_digest` other than the digest in `IMAGE`;
- refuses the three pin overlays on any other digest, matched by the sha256 of the body below the
  header (`LEGACY_PIN_OVERLAY_SHA256`), so a hand-edited digest line does not get them through;
- refuses two overlays with the same `upstream_file`;
- mounts the file at `/usr/local/lib/python3.12/dist-packages/<upstream_file>` on the head, and
  copies it to `/tmp/qwen38-overlays/` on the worker and mounts it there too.

On the pin, `run.sh` also requires overlays for `vllm/model_executor/layers/quantization/modelopt.py`
and `vllm/models/qwen4_exp/nvidia/ops/ple.py` unless `DIAGNOSTIC=1`.

## Which overlays mount

`OVERLAYS=auto` (default) picks by the digest in `IMAGE`:

| IMAGE digest | List |
|---|---|
| pinned nightly `sha256:df871f17…` | `OVERLAYS_PIN` |
| v0.30.0 `sha256:4864d466…` (default) | `OVERLAYS_V030` |
| anything else | none |

`OVERLAYS="a.py b.py"` names the files directly (relative to the recipe root, or absolute).
`OVERLAYS=none` mounts nothing. The defaults live in `recipe.yaml` `serve.env`.

## v0.30 overlays (default)

| File | Target | Generator | Why |
|---|---|---|---|
| `v030/flashinfer_cutlass_moe.py` | `vllm/model_executor/layers/fused_moe/experts/flashinfer_cutlass_moe.py` | `v030/apply_moe_finalize_overlay.py` | `use_fused_finalize=not _MOE_DETERMINISTIC`: FlashInfer's fused finalize sums the top-10 experts with BF16 atomics (run-to-run noise that the model amplifies to nats). `VLLM_QWEN38_MOE_DETERMINISTIC=0` restores stock. |
| `v030/gdn_attn.py` | `vllm/v1/attention/backends/gdn_attn.py` | `v030/apply_gdn_fresh_prefill_overlay.py` | `treat_short_extends_as_decodes=m.is_prefilling is None`: a fresh 1-token prompt no longer runs the GDN decode kernel on a leftover state slot. |
| `v030/qsa_indexer.py` | `vllm/models/qwen4_exp/nvidia/ops/qsa_indexer.py` | `v030/apply_qsa_topk_order_overlay.py` | Sorted `persistent_topk` rows; prefill ties resolve to the lowest index (F23). |
| `v030/serving.py` | `vllm/entrypoints/openai/chat_completion/serving.py` | `v030/apply_api_overlays.py` | vLLM #56067: streamed tool-call arguments stop costing O(n²) API-server CPU (G05). |

The first three are the S1.5 determinism fixes (`evidence/s15-determinism/SUMMARY.md`): with them c=1 greedy is bit-exact run to run.
Other generated files in `docker/v030/` (`hyperconnection.py`, `modelopt.py`, `mtp.py`, `nccl_twin_cuda_communicator.py`) are lever overlays that are off by default; they mount only when named in `OVERLAYS`.

## Pin overlays (rollback)

| File | Target | Generator | Why |
|---|---|---|---|
| `ple_layer.py` | `vllm/models/qwen4_exp/nvidia/ple_layer.py` | `apply_ple_overlay.py` | Byte-identical to stock on this digest (stock already selects FP8 PLE). Kept for older images. |
| `modelopt.py` | `vllm/model_executor/layers/quantization/modelopt.py` | `apply_mtp_fp8_overlay.py` | Remaps `mtp.layers.48` to checkpoint `mtp.layers.0` and routes MTP `FP8_BLOCK_SCALES` to `Fp8MoEMethod`. v0.30 does this natively. |
| `ple_ops.py` | `vllm/models/qwen4_exp/nvidia/ops/ple.py` | `apply_ple_stride_overlay.py` | Only the vLLM #55375 hunks: `state_idx_stride` on `_ple_conv_kernel` and `_ple_conv_writeback_kernel`, `sid = tl.load(state_idx_ptr + r * state_idx_stride)`, and the wrapper passing `state_indices.stride(0)` (F01). Not the v0.30 file: its required `outer_residual` breaks the pinned `ple_layer.py` call site. |

Regenerate from the image (`docker create` + `docker cp`, no GPU) or from an extracted tree:

```bash
python3 docker/apply_ple_overlay.py
python3 docker/apply_mtp_fp8_overlay.py
python3 docker/apply_ple_stride_overlay.py
# or: python3 docker/apply_ple_stride_overlay.py --src <tree>/vllm/models/qwen4_exp/nvidia/ops/ple.py
```

`apply_ple_stride_overlay.py` refuses a stock file whose sha256 is not the pinned one, because its
hunks are written against that file.

## Adding an overlay

1. Write a generator that reads the stock file (from `--image` or `--src`), applies only the hunks you
   need, and writes the header above.
2. Add the output file to `OVERLAYS_PIN` or `OVERLAYS_V030` in `recipe.yaml`, then `python3 kit/render.py`.
3. Run `VALIDATE_ONLY=1 ./run.sh`. It prints each overlay and its mount target.
4. Add a test in `tests/`.

A file that defines `get_top_tokens` or `compute_logits_local` also unlocks the
`use_local_argmax_reduction` and `--enable-batch-sharded-sampling` guards.
