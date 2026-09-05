# c4 native context 262144

## Mechanism

vLLM derives `max_model_len` from `max_position_embeddings=262144`. NVIDIA's own sample is `--max-model-len 262144`. The card says 262K natively, extensible to 1M. This recipe currently serves 1048576 with `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`. That env is not YaRN. Community 1M YaRN on GB10 hangs on long prefills (vLLM #54629).

The 1M ceiling inflates advertised `max_seq_len` and the KV page tables. Spark Arena TP=2 on this SHA used seqs=8 at native 262144. Short-stream occupancy is admission-limited by `--max-num-seqs`, not by the 1M ceiling. Dropping to native should:

1. Boot without `VLLM_ALLOW_LONG_MAX_MODEL_LEN`.
2. Keep seqs=8 eight short streams.
3. Hold or improve TTFT p50 versus last kept pin c2 (0.15s / 0.17s).
4. Stop advertising 1M as the served window.

Do not treat leftover UMA as a keep by itself. Record it.

## Ruler

Frozen `python3 bench_decode.py --runs 3 --concurrency 1 2 8 --max-tokens 200`. Same file hash as `evidence/opt-c2-nightly-aarch64-e962733/harness.sha256`. Last kept pin is c2, not the stale RadixArk opt-baseline.

## Stop

One attempt. Keep if health 200, quality smokes pass, seqs=8 does not drop a stream, and none of reliability, quality, concurrency, decode prose, or TTFT regresses past noise versus c2. Official-vendor native window may keep at within-noise speed.
