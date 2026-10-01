# U1: vLLM v0.30.0 by digest, session 1, 2026-09-29

Config: `IMAGE=vllm/vllm-openai:v0.30.0-aarch64@sha256:4864d466...` (id 91d9b077 on both nodes, `image-inspect.txt`), `OVERLAYS=auto`,
which mounts none (`OVERLAYS_V030` is empty). `EXTRA_ENV="VLLM_PLE_CPU_OFFLOAD=0 VLLM_USE_BREAKABLE_CUDAGRAPH=0"`; both exist in v0.30 `envs.py`.
`EXTRA_ARGS='--per-request-spec-decode-metrics summary'`. run.sh defaults at 6aefbfc: UTIL 0.76, oom-score-adj 1000, memguard on both ranks (`run.log`).
**No run.sh change was needed**: `VALIDATE_ONLY=1` passed and no guard blocked v0.30.
Boot 14:27:44Z to /health 200 at 14:40:15Z (12.5 min). **Left running at the end of the session.**

## Boot needles (rank 0)

- `Initialized PLE embedding ... quantization_method=Qwen4ExpPLEFp8EmbeddingMethod ... pinned=False`: offload off, so PLE is not pinned to host
- `Using FlashInfer GDN prefill kernel (requested=auto, head_k_dim=128)` (the pin logs `Triton/FLA`); GDN decode kernel `cuda`
- target `FLASHINFER_CUTLASS` NvFp4 MoE; drafter `FP8 MoE block scales refined ... [64, 64]` + `Using TRITON Fp8 MoE backend`,
  plus `Using default MoE config ... E=512,N=320,device_name=NVIDIA_GB10,dtype=fp8_w8a8` (no tuned Triton config: plan L6a)
- `GPU KV cache size: 1,625,292 tokens`, 6.20x of 262,144; 27.35 GiB per rank. B1 has 1,640,879 tokens and 27.68 GiB at the same UTIL.
- 0 Traceback/ERROR/NCCL WARN lines on either rank (full logs in `~/projects/data/qwen38-evals/runs/session1/u1-v030/`)

## Memory (free -h)

| When | spark1 avail / swap used | spark2 avail / swap used |
|---|---|---|
| after boot | 17 GiB / 8.5 GiB | 18 GiB / 4.2 GiB |
| after bench + T2 + T3 + prefix | 14 GiB / 8.1 GiB | 17 GiB / 4.2 GiB |

spark1 is at least as good as B1 (15 / 13 GiB).

## Results

| Step | Result |
|---|---|
| smokes | all rc=0 |
| T0 `bench_probe.py` | **0 gated bursts / 30** (0 zero-acceptance, 0 runaway); first-32 match 0.354 (B1 0.223) |
| gate ruler | VOID on the absolute anchor [59, 61] (4/9 excursions, boot_min 60.45) |
| `ruler_steps.py --anchor-ms 0 0` | not void, **0/9 sentinel excursions**, boot_min 60.30. prose c1 59.91 ms (41.4 tok/s, acc 2.14-2.56); c2 63.66 (agg 69.2); c8 92.08 (agg 177.7); structured c1 60.97 (65.3); c2 62.58 (agg 127.1, acc 4.0); c8 81.03 (agg 392.7, acc 4.0) |
| frozen `bench_decode.py` | prose c1 **40.2** tok/s (acc 2.49); prose c2 34.4 / agg 68.0 (acc 2.27); structured c1 65.2; structured c2 63.2 / agg 126.3 (acc 4.0) |
| `bench_diverse` | prose c1 59.75 ms (acc 2.24); prose c8 **107.3** (B1 110.5); structured c1 59.69 (acc 3.14); structured c8 111.5 (B1 114.4) |
| `bench_sampled` | A c1 60.3 / 60.1 ms, acc 2.64; A c8 116.9 / 109.2 ms (B1 120.1 / 112.5), acc 2.69; T1 c1 acc 2.95; T1 c8 69.5 ms |
| T1 A/A (limit 2 x2) | top-1 **93.00%**, KL 0.0363: the same noise as B0/B1 |
| T1-G self | distinct_mean 5.0 / 5, early_div 0.52 |
| T1-A U1 vs B1 (`--aa` B1 floor) | top-1 92.58%, KL 0.0361, NLL delta +0.0009, flips 362/368 (p 0.85): **at the A/A floor**. Verdict FAIL only on the per-domain rule (image 30 tok, tools 209 tok, ja); with a 93% floor this is not a signal. Plan wants T1-A vs a **spec-off pin golden**, which does not exist yet (S1.5) |
| T1-G U1 vs B1 | PASS (distinct 5.0 <= 5.0; early_div 0.72 <= 0.74) |
| T2 (`--workers 4`) | tools 57/60 (t11.stream asked a clarifying question; t47 both modes answered without a call); json 30/30; rep4 64/64 (0.00069); effort 5/5; IFEval 106/120 (88.3%); GSM8K-250 **242 (96.8%)** |
| T2 paired vs B1 | GSM8K +0.8 pp (0 lost / 2 gained); IFEval +0.8 pp (1/2); tools -1.7 pp (3/2, p 1.0). FAIL only on the tools 60/60 rule, which **B1 also fails (58/60)**; the failures are different items each run |
| T3 4k/16k/32k | **18/18**; the second needle at 32k takes **0.86 s** (prefix hit) against 9.8-10.1 s cold |
| `probe_prefix.py` (16k) | **prefix reuse works with MTP + mamba on v0.30**: repeat TTFT 4.67 -> 0.95 / 0.96 s (hits 12,800 of 15,994); 3-turn chat 5.12 -> 0.58 / 0.59 s (hits 14,400). On the pin the second prefill is a full miss (B1 T3) |
