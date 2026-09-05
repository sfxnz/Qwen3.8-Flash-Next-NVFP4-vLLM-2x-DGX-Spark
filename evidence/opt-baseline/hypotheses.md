# Opt hillclimb, baseline only

Metric. Decode prose tok/s at c=1 and c=2 from frozen `python3 bench_decode.py --runs 3 --concurrency 1 2 --max-tokens 200`, thinking off. Direction that counts as better is higher median tok/s with no dropped stream.

Bars, in order when they conflict:

1. Reliability. No OOM, no dropped streams, refuse-guards still hold, health 200.
2. Quality. Thinking off, tools, vision, lossless greedy count 1 to 200, no CoT leak.
3. Concurrency. Raise max_num_seqs only while every stream still completes.
4. Decode prose. Frozen `bench_decode.py` on 2x. Do not mix harnesses. Do not invent tok/s.
5. Prefill. TTFT p50 from the same harness. Do not buy decode by secretly hurting TTFT.

A keep requires 1 and 2 still passing, at least one of 3/4/5 improved beyond noise, and none of the five regressed beyond noise.

Frozen ruler. `python3 bench_decode.py --runs 3 --concurrency 1 2 --max-tokens 200`. Extra `--concurrency 8` is the occupancy probe, not a README cell unless it stays lossless.

Stop. This file is the h0 capture. Do not change the recipe until a later iteration names a mechanism.

H0. Live seqs=8 pin on the restored RadixArk 7b719225 boot that started 2026-09-05T11:48:45Z. No engine change. Proves the ruler and the correctness gate on the current serve. Prior 19h receipts (prose 36.36/33.60) are a different container.
