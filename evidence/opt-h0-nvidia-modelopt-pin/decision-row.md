# h0-nvidia-modelopt-pin

Verdict: reverted.

The official NVIDIA ModelOpt MIXED_PRECISION checkpoint at fab0aec does not boot on the current day-0 image `vllm/vllm-openai:qwen38-flash-next@sha256:3b0e188ffceb3d07e09c3cb5215433a0020eacf02d7f882ed3a8bfd15454477e` (v0.1.dev20073+g8e685d198).

Engine auto-detected `quantization=modelopt_mixed`. Weight load reached the MTP module and raised:

`AttributeError: Layer mtp.layers.48.mlp.experts has no parameter 'w2_weight_scale_inv' for checkpoint weight 'mtp.layers.48.mlp.experts.0.down_proj.weight_scale_inv'`

`EXTRA_ARGS='--quantization modelopt'` was retried once. The flag is present on argv. The engine remapped it to `modelopt_mixed` and failed on the same MTP scale.

Recipe files were restored to `RadixArk/Qwen3.8-Flash-Next-NVFP4` snapshot `7b719225`. Live serve restored. `GET /health` is 200. `/v1/models` lists the RadixArk id.

Do not stack occupancy or decode knobs on this pin. A later candidate needs an image that loads ModelOpt MIXED_PRECISION MTP FP8_BLOCK_SCALES, or a checkpoint whose MTP experts match the day-0 loader.
