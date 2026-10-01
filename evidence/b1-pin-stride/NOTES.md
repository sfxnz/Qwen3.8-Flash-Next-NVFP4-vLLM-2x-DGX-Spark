# B1: pin + 3 overlays (F01 stride fix), session 1, 2026-09-29

Config: branch default at 6aefbfc: pin `df871f17`, `OVERLAYS=auto` -> ple_layer.py, modelopt.py, ple_ops.py; **UTIL 0.76**,
`--oom-score-adj 1000`, memguard on both ranks (`run.log` lines 9 and 17: `memguard pid=`). `EXTRA_ARGS='--per-request-spec-decode-metrics summary'`.
A first B1 boot at UTIL 0.80 (old run.sh) was stopped before any traffic after the OOM incident (`run-aborted-util080.log`).
Boot 13:21:45Z to /health 200 at 13:34:22Z (12.6 min). KV cache **1,640,879 tokens** (6.26x of 262,144), 27.68 GiB per rank.
Needles as B0 (FLASHINFER_CUTLASS NVFP4, drafter TRITON FP8 after 64x64 refine, Triton/FLA GDN prefill, GDN decode cuda). No Traceback or ERROR.

## Memory (free -h)

| When | spark1 avail / swap used | spark2 avail / swap used |
|---|---|---|
| pre-boot | 116 GiB / 3.8 GiB | 117 GiB / 1.8 GiB |
| after boot | 15 GiB / 8.0 GiB | 18 GiB / 4.3 GiB |
| after T2 + T3 (end) | 13 GiB / 7.9 GiB | 16 GiB / 4.3 GiB |

spark1 kept swapping in during decode (vmstat `si` 100-300 KB/s, kswapd0 active); see anomalies in the session summary.

## Results

| Step | Result |
|---|---|
| smokes (thinking, tools, vision, count) | all rc=0 |
| T0 `bench_probe.py` (30 bursts, 130 streams) | **0 gated bursts: 0 zero-acceptance, 0 runaway** (B0 had 7/7/7). first-32 match 0.223 (report-only; c=1 references are themselves non-reproducible) |
| gate `ruler_steps.py` | **BOOT VOID** at the start sentinel: structured c1 61.1-62 ms vs anchor [59, 61], with 72-75 ms excursions |
| `ruler_steps.py --anchor-ms 0 0` (relative boot-min rule only) | not void; 4/15 sentinel excursions; boot_min 60.72 ms. Cells: prose c1 60.43 ms (37.8 tok/s, acc 2.2-2.43); prose c2 65.39 ms (agg 70.5, acc 2.28-2.43); prose c8 94.25 ms (agg 174.3); structured c1 60.83 ms (65.4 tok/s, acc 4.0); structured c2 61.95 ms (**agg 128.4**, acc 4.0 in 3/3); structured c8 81.33 ms (**agg 391.2**, acc 4.0) |
| frozen `bench_decode.py` | prose c1 37.7 tok/s (acc 2.21); prose c2 per-stream 35.2 / agg 64.4 (acc 2.34); structured c1 65.0; structured c2 63.8 / **agg 127.5** (acc 4.0) |
| `bench_diverse` (32 prompts) | prose c1 60.47 ms (acc 2.26); prose c8 110.5 ms (acc 2.23, ~161 tok/s agg); structured c1 60.83 ms (acc 3.12); structured c8 114.4 ms (acc 3.12) |
| `bench_sampled` (A: T=1 top_p .95 top_k 20; 8+8 prompts) | A c1 60.4 ms prose / 60.8 structured, acc 2.71; A c8 120.1 / 112.5 ms, acc 2.66; T1 c1 acc 3.02; T1 c8 69.1 ms, acc 2.91 |
| T1 A/A (`--limit 2` x2, 18,492 tokens) | top-1 **92.73%**, KL 0.0369: **unchanged from B0 (92.79%)**; prose 98.7%, code 98.1%, es 86.7%, de 89.6%, ja 86.9% |
| T1-G self (10 x 5, c=1) | distinct_mean **5.0 / 5**, early_div 0.54, 10/10 prompts diverge (B0 4.8) |
| T2 (`--workers 4`) | tools **58/60** (t08.stream answered without a call; t27.stream added `formal:false`); json 30/30; rep4 64/64, mean repeat-4gram 0.00054; effort 5/5; IFEval-120 105 (87.5%); GSM8K-250 **240 (96.0%)** |
| T3 needles 4k/16k/32k | **18/18**; 32k prefill 10.3-10.5 s for both needles (no prefix reuse, F20) |

Conclusion: the stride overlay fixes F01 (T0 flags and c=2/c=8 slow waves are gone; structured c2 goes from 35.7 to 127.5 agg),
but it does **not** change the c=1 A/A noise (92.7% top-1, 5/5 distinct greedy outputs). That noise is a separate defect.
