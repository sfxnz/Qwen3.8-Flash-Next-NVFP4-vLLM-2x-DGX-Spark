# Occupancy hypotheses

Metric. Concurrent short streams admitted without a dropped request, a thinking-off leak, a tool/vision fail, a broken 1-to-200 count, a TTFT spike, or a UMA leftover collapse. Direction that counts as better is more reliable streams, not a peak tok/s that drops a third request.

Frozen ruler. `python3 bench_decode.py --runs 3 --concurrency 1 2 --max-tokens 200`. Extra `--concurrency` values are occupancy probes, not README cells unless they stay lossless.

Stop. Start at `max_num_seqs_start=2`. Raise one knob at a time (2, 4, 8). Revert the knob when quality, TTFT, or UMA leftover regresses. Encode the failing setting as the `run.sh` refuse-guard. Stop after 8 if 4 and 8 both hold, or earlier on the first revert.

H0. Baseline capture at live seqs=2. No engine change. Proves the ruler and the correctness gate on the boot pin.

H1. seqs=4 still admits two short streams at the frozen c=1/c=2 cells without TTFT or UMA leftover regression, and a c=4 wave does not drop a stream. Mechanism. The KV pool is 2.06M tokens. Short requests are admission-capped by `--max-num-seqs`, not by the 1.97x 1M concurrency figure. CUDA graphs recapture 4 decode shapes.

H2. seqs=8 matches Spark Arena on this SHA at 262144. Same gate. Mechanism. Arena used MTP-3, kv auto, seqs=8 at the native window. This recipe's 1M ceiling still leaves a 2M-token pool, so eight short streams should fit unless graph capture or UMA leftover (already ~10 GiB on spark1 with swap in use) regresses.
