# S1.1 microbench, session 1 (2026-09-29)

Raw data: `~/projects/data/qwen38-s11/session1/` (both images, pin + v030). Go/no-go: `gonogo.md` / `gonogo.json`.
Wall time ~50 min (12:55-13:45Z) plus a 4 min pin MoE rerun.

- spark2 had v0.30 (id 91d9b077) via docker load with no RepoDigest; `docker pull vllm/vllm-openai@sha256:4864d466...` fixed it (layers present). Same id on both nodes.
- **Fix (small, obvious):** `tools/micro/moe_nvfp4.py` passed the activation global scales as shape (1,); FlashInfer 0.6.18 `fused_moe_120` requires 0-d or (num_experts_on_rank,), so every MoE cell SKIPped on the first pin pass (`moe.prefix.log` in the raw dir). Fix: `.reshape(())` in `quant_scales` (`moe_nvfp4-fix.diff`). v030 ran with the fix; pin MoE was re-run (`run-moe-rerun.log`), 0 SKIP.
- `gemm-algo` writes only the cuBLASLt log (no JSON) by design; the HC kernel choice row comes from it.
- vmstat on spark1 only (`vmstat-spark1.txt`), 10 s interval.
- No step MISSING.
