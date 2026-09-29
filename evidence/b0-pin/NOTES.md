# B0: pin without the F01 stride overlay (diagnostic), session 1, 2026-09-29

Config: pin image `df871f17`, `DIAGNOSTIC=1 OVERLAYS="docker/ple_layer.py docker/modelopt.py"`,
`EXTRA_ARGS='--per-request-spec-decode-metrics summary --profiler-config {"profiler":"torch","torch_profiler_dir":"/tmp/qwen38-traces"}'`.
**Booted with the old run.sh defaults: UTIL 0.80, no memguard, no oom-score-adj** (before commit 6aefbfc).
Boot 12:20:13Z to /health 200 at 12:32:38Z (12.4 min). KV cache 1,972,456 tokens (7.52x of 262,144); 32.68 GiB KV per rank.
Needles: FLASHINFER_CUTLASS NvFp4 MoE; drafter `TRITON Fp8 MoE` after the 128->64 refine; `Triton/FLA GDN prefill`; GDN decode `cuda`.
Times in this file are UTC (the incident README uses BST = UTC+1).

## Results

| Step | Result |
|---|---|
| (a) smokes: thinking, tools, vision, count | all rc=0 (`gate/smoke-*.out`) |
| (b) T0 deterministic `ruler_steps.py --historical` | **PASS, exact**: flagged = expected F01 set {prose c2 r2, prose c8 r2, structured c2 r1, structured c2 r3, structured c8 r2} |
| (c) `bench_probe.py --self-test` | **rc=3, self-test FAIL on the threshold only**: 7/30 bursts gated (7 zero-acceptance + 7 runaway streams, incl. the >8192-token long pair "Register Register ..."), first-32 match 0.208; non-first barrier flagged 6/23 = 26% < the 30% rule. The probe sees F01; the 30% prior is too tight for this boot |
| (d) frozen `bench_decode.py` | prose c1 36.9 tok/s (acc 2.32), prose c2 agg 26.9 (acc 1.40, F01), structured c1 63.9 (acc 4.0), structured c2 agg 35.7 (acc 2.00, F01) |
| (e) `ruler_steps.py` warm-up + sentinels | **BOOT VOID**: sentinel structured c1 60.3-62.1 ms/step vs anchor [59, 61]; 15/18 excursions. Prose c1 median 59.2 ms/step (acc 2.02-2.35); prose c2 59.3 ms; prose c8 92.3 ms (one wave 69.2 ms with 6 zero-acceptance runaways). Structured cells not reached. SUMMARY refused (3 T0-flagged waves + void) |
| (f) T1 A/A (collect_t1 `--limit 2` x2, 18,492 tokens) | top-1 **92.79%**, KL 0.0366, flip rate 3.8%; de 87.7%, code 97.8% (`t1-aa.json`) |
| (f) T1-G self (10 prompts x 5 repeats, c=1 greedy) | distinct_mean **4.8 / 5**, early_div 0.46, 10/10 prompts diverge (`t1g-self.json`) |
| (g) torch profiler, c=1 prose, 128 tok | 55 steps in 3.45 s; rank0 GPU busy 98.7%. BF16 WMMA GEMM 1.50 s (44%), gemv 0.65 s (19%), NVFP4 grouped GEMM 0.57 s (16.5%), NCCL AR 0.27 s (4.9 ms/step), AG 0.04 s, GDN decode 0.04 s (`profile-kernel-summary.txt`; class regexes are rough) |

The whole boot ran about 2-3% slow (structured c1 60.3-62 ms against a historic 59.5). Suspects: the
`--profiler-config` hook or the per-request spec metrics. B1 carries only the metrics flag, so the two can be told apart.

## Incident (host OOM on spark1)

- Command at 12:50:08Z (13:50 BST): `OUT_DIR=~/projects/data/qwen38-traces/session1-b0-pin tools/micro/profile_window.sh`
  with defaults `CONCURRENCY=1 MAX_TOKENS=128 FLUSH_S=15 STOP_TIMEOUT_S=600`, warm-up then `/start_profile`,
  one streamed 128-token prose request (3.41 s), then `/stop_profile`. `stop_profile` did not return in 600 s (curl rc=28).
- Before the window, spark1 had 9.9 GiB available with 8.0 GiB swap used (`free-pre-gate.txt`). At 13:01Z it had 0.66 GiB available and 15 GiB swap used (`free-end.txt`).
  The serve was stopped at 13:05Z, and spark1 recovered to 115 GiB available (`free-poststop.txt`).
- Trace files: `~/projects/data/qwen38-traces/session1-b0-pin/rank0/` (rank0 147 MB gz + async_llm 2.4 MB) and
  `.../rank1/` (20 MB gz, also on spark2 at the same path). Parsed pickles `rank0.pkl`, `rank1.pkl` sit next to them.
- No more profiler windows this session (coordinator rule 2).

Raw run data: `~/projects/data/qwen38-evals/runs/session1/b0-pin/` (T1/T1-G captures, full docker logs).
