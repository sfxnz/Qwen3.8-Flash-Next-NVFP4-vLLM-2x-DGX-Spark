# Opt h0. NVIDIA ModelOpt MIXED_PRECISION pin

Metric. Frozen `python3 bench_decode.py --runs 3 --concurrency 1 2 --max-tokens 200` prose tok/s at c=1 and c=2, thinking off. Direction that counts as better is higher median tok/s with no dropped stream. This iteration is an official vendor pin swap, so a keep may stay within noise of published prose 39.5/34.9 and live opt-baseline 36.4/33.6.

Bars, in order when they conflict:

1. Reliability. No OOM, no dropped streams, refuse-guards still hold, health 200.
2. Quality. Thinking off, tools, vision, lossless greedy count 1 to 200, no CoT leak.
3. Concurrency. seqs=8 occupancy pin. Extra `--concurrency 8` is the occupancy probe.
4. Decode prose. Frozen `bench_decode.py` on 2x. Do not mix harnesses. Do not invent tok/s.
5. Prefill. TTFT p50 from the same harness. Do not buy decode by secretly hurting TTFT.

A keep requires 1 and 2 still passing. Official vendor pin may keep at within-noise speed if 1 and 2 hold. None of the five bars may regress beyond noise.

Frozen ruler. `python3 bench_decode.py --runs 3 --concurrency 1 2 --max-tokens 200`. Extra `--concurrency 8` is occupancy, not a README cell unless it stays lossless.

Stop. One change. Pin `nvidia/Qwen3.8-Flash-Next-NVFP4` at `fab0aec`. Keep image, seqs=8, kv=auto, MTP-3, HCA, and PLE overlay. If auto-detect misses MIXED_PRECISION, retry once with `EXTRA_ARGS='--quantization modelopt'`. If boot fails, revert before stacking knobs.

H0. Official NVIDIA ModelOpt MIXED_PRECISION checkpoint. Routed experts stay NVFP4 W4A4. MTP experts become FP8_BLOCK_SCALES. PLE is per-tensor FP8. Mechanism. The day-0 image plus PLE overlay already loads mixed NVFP4 plus FP8 PLE on the community RadixArk host of the same architecture. The official card is the producer checkpoint (`quant_algo=MIXED_PRECISION`, ModelOpt 0.46). GB10 is not on the NVIDIA hardware list (B200/B300). The current image has no `ModelOptMixedPrecisionConfig` PLE selector, so the overlay stays.
