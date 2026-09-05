# c3 MTP FP8_BLOCK_SCALES overlay

Metric. Health 200 on `nvidia/Qwen3.8-Flash-Next-NVFP4@fab0aec` with MTP-3 still on. Engine must log `quantization=modelopt_mixed` and must not raise `AttributeError` on `mtp.layers.48.mlp.experts` `w2_weight_scale_inv`. Quality smokes stay green. Frozen `python3 bench_decode.py --runs 3 --concurrency 1 2 8 --max-tokens 200` is the ruler versus `evidence/opt-baseline`.

Direction. Official MIXED_PRECISION pin that boots is better than community RadixArk, even at within-noise speed, if reliability and quality hold.

Stop. This overlay is one hypothesis. Keep it if health is 200, quality passes, and none of the five bars regress beyond noise versus opt-baseline. Revert the overlay and restore the last health-200 pin if boot fails.

Mechanism. Checkpoint MTP experts live at `mtp.layers.0`. Runtime MTP is `mtp.layers.{num_hidden_layers}` (48 on this card). Stock `ModelOptMixedPrecisionConfig` never adds that prefix as a `quantized_layers` candidate, so MTP experts look unquantized. `LINEAR_ALGOS` has no `FP8_BLOCK_SCALES`. RoutedExperts then miss `w2_weight_scale_inv`. The overlay remaps `mtp.layers.48` onto `mtp.layers.0` and dispatches `Fp8MoEMethod` with block size 128.

c1 already pulled this overlay after day-0 `qwen38-flash-next@3b0e188` and stock `nightly-aarch64@df871f17` both died on that AttributeError. c2 kept the same overlay and dropped `VLLM_PLE_FP8_CHECKPOINT`. This run recaptures the live pin. It does not restack seqs, KV, or marlin. c4 native 262144 stays queued.
