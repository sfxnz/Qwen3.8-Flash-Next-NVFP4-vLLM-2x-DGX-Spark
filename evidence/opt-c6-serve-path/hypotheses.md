# c6 official Flash-Next serve path

Quant stays `nvidia/Qwen3.8-Flash-Next-NVFP4` @ `fab0aec`. This change is the serve argv and the MTP overlay, not a second pack.

Metric. Decode score is prose only from `python3 bench_decode.py` (default `--phase prose`). Structured and code cells are diagnostics. No Spark run on this revision.

## What changed

- `--no-enable-flashinfer-autotune` (NVIDIA MTP Usage and official vLLM Flash-Next recipe)
- `--quantization modelopt` (NVIDIA Usage; engine remaps MIXED_PRECISION to `modelopt_mixed`)
- `--mamba-cache-mode align` (this nightly already selects align when prefix caching is on; evidence/opt-c5-mtp-k2/run.log)
- `docker/modelopt.py` backports vLLM #55513 (`FP8_PB_WO` alias, `has_blocked_weights`, shared `fp8_block_config`) and keeps the lab `mtp.layers.48` → `mtp.layers.0` lookup
- README decode table is prose only. 262144 is a config cap until L.A.I.L `pp` 8k–64k is green

## What did not change

- Image digest `df871f17`. Newer nightlies exist and are unbooted
- No `--enable-expert-parallel` (TEP2 + 128×128 needs a Spark NCCL row)
- No global `marlin` or `b12x`
- No L.A.I.L `qwen38_nvfp4` overlay (`qwen3_coder` + `--kv-cache-dtype fp8`)
- No YaRN / `ALLOW_LONG` / 1M
- MTP-3 stays. c5 k=2 was reverted
- Vision stays on

## Keep rule

Do not replace published prose 43.2 / 36.2 until an exclusive TP=2 Spark run repeats `python3 bench_decode.py` on this argv. Nsight targets are GDN 100 vs 99 KiB, then NVFP4 CUTLASS vs cute_dsl, then MTP 64×64 Triton.
