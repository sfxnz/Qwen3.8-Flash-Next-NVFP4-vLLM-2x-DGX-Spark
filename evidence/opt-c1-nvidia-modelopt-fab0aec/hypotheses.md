# Opt c1. NVIDIA ModelOpt MIXED_PRECISION pin fab0aec

Metric. Frozen `python3 bench_decode.py --runs 3 --concurrency 1 2 --max-tokens 200` prose tok/s at c=1 and c=2, thinking off. Official vendor pin may keep at within-noise of live opt-baseline 37.29/33.32.

Bars, in order when they conflict:

1. Reliability. Health 200. No OOM. No dropped streams. Refuse-guards still hold.
2. Quality. Thinking off, tools, vision, lossless greedy count 1 to 200, no CoT leak.
3. Concurrency. seqs=8 occupancy pin.
4. Decode prose. Frozen `bench_decode.py` on 2x. Do not invent tok/s.
5. Prefill. TTFT p50 from the same harness.

A keep requires 1 and 2 passing. Official vendor pin may keep at within-noise speed if 1 and 2 hold. None of the five bars may regress beyond noise.

Stop. Boot `nvidia/Qwen3.8-Flash-Next-NVFP4@fab0aec` on day-0 `qwen38-flash-next@3b0e188`. If load dies on `mtp.layers.48` `w2_weight_scale_inv`, digest-pin `nightly-aarch64@df871f17` and retry. If that still dies, overlay MTP FP8_BLOCK_SCALES dispatch. Revert the bundle only when every official image/overlay path that this pair can pull still cannot boot, or when boot succeeds but a keep rule fails.

H0. Official NVIDIA MIXED_PRECISION checkpoint. Routed experts NVFP4. PLE per-tensor FP8. MTP experts FP8_BLOCK_SCALES group 128. Day-0 already failed this pin. This apply still boots day-0 first, then the newer official image.
