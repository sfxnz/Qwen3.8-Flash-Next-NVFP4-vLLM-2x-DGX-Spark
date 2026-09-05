# c2 nightly-aarch64 e962733

Metric. Official CUDA 13 arm64 vLLM image that loads NVIDIA MIXED_PRECISION at health 200, with thinking-off, tools, vision, and lossless count still green. Frozen ruler is `python3 bench_decode.py --runs 3 --concurrency 1 2 --max-tokens 200`. Occupancy probe is extra `--concurrency 8`. Direction that counts as better is a keep of this official pin if reliability and quality hold and no bar regresses beyond noise.

Stop. One measured boot of `nightly-aarch64@sha256:df871f170ee7070fbdce162bde08fb616e311570c948a620be0d4b33fe02f87b` without `VLLM_PLE_FP8_CHECKPOINT` in the container. Dedicated `qwen38-flash-next` has not been republished. cu129-nightly at the same commit is the wrong CUDA for GB10.

H. Nightly `e962733` already selects MIXED_PRECISION FP8 PLE via stock `Qwen4ExpPLEFp8EmbeddingMethod`. The live engine logs `Unknown vLLM environment variable detected: VLLM_PLE_FP8_CHECKPOINT`. Dropping the refuse-guard and not passing that env still boots because PLE load does not read it. Decode should match c1 (same digest) within noise. MTP still needs `docker/modelopt.py`.
