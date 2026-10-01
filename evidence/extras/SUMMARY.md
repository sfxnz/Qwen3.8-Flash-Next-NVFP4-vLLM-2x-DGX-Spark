# Session 6: X10 throughput profile, P2-5 step 2, T4 for lab-fp8-dense (2026-09-30)

**Result:** no default change. The lossless default (v0.30 + determinism overlays + L1a/L1b′/L6a, MTP k=3, `--long-prefill-token-threshold 4800`, `--max-num-seqs 8`) was re-booted as `final/`, passed the confirmation gate pack and is serving.
- **X10** passes every gate and ships as the opt-in **`throughput` profile**: `THROUGHPUT_PROFILE=1` → `--max-num-seqs 16`.
- **P2-5 step 2** (threshold 3200, 4832 batched tokens) is **not kept**. Decoder ITL p99 during a 128k prefill only falls from 3.00 to 2.59 s, against a target of ≤ 0.6 s.
- **T4 for `lab-fp8-dense` passes:** 97.65% vs 97.80% (−0.15 pp; 6 lost / 4 gained; McNemar p 0.75). This equals the default-vs-default A/A (−0.15 pp, p 0.73), so it is inside sampling noise. Promotion still needs owner sign-off, because the session-5 T1-B finding stands.

Commands are in `commands.sh`; the runners are `boot.sh`, `steps.sh` and `table.py`. Raw data and telemetry: `~/projects/data/qwen38-evals/runs/session6/<arm>/`.

## Arms (one boot each)

| Arm | Config on top of the default | KV tokens | spark1 avail after boot |
|---|---|---|---|
| ref | live default | 1,602,620 | 16 GiB |
| x10-seqs16 | `THROUGHPUT_PROFILE=1` (graphs captured to 128) | 1,560,111 | 16 GiB |
| p25-lpt3200 | `LONG_PREFILL_TOKEN_THRESHOLD=3200 MAX_NUM_BATCHED_TOKENS=4832` | 1,687,640 | 16 GiB |
| t4-fp8-dense | `FP8_DENSE=per_block` | 1,701,810 | 15 GiB |
| **final** | rendered default, **serving** | 1,588,450 | 16 GiB |

Under any load in this session, spark1 had at least 13.1 GiB available. Swap stayed ≤ 8.8 GiB (spark1) and ≤ 4.7 GiB (spark2). Memguard never tripped, and no profiler was run.

## Ruler (anchored 53.8–55.8, ms/step)

| Arm | prose c1 | prose c2 | prose c8 | struct c1 | struct c2 | struct c8 |
|---|---|---|---|---|---|---|
| ref | 53.06 | 61.58 | 80.57 | 54.77 | 58.97 | 74.59 |
| x10-seqs16 | 53.08 | 61.22 | 80.90 | 54.48 | 59.06 | 75.57 (+1.31%) |
| p25-lpt3200 | 53.39 | 61.29 | 80.83 | 55.07 | 58.65 | 74.77 |
| **final** | 53.11 | 61.10 | 80.56 | 54.66 | 58.24 | 75.22 |

Frozen `bench_decode.py` on final: prose c1 43.8, structured c1 73.1, structured c2 aggregate 135.5 tok/s.

## 1. X10 throughput profile: kept as opt-in

`bench_diverse.py --concurrency 1 8 16`. Each cell shows ms/step · aggregate tok/s · TTFT p50.

| Cell | ref (seqs 8) | x10 (seqs 16) |
|---|---|---|
| prose c8 | 102.41 · 159.5 · 0.39 s | 103.49 · 159.0 · 0.39 s |
| prose c16 | 104.18 · 162.3 · 5.68 s (8 queued) | 140.05 · **235.0** · 0.48 s |
| struct c8 | 105.84 · 189.8 · 0.39 s | 107.17 · 191.4 · 0.38 s |
| struct c16 | 111.00 · 197.1 · 3.25 s (8 queued) | 151.95 · **254.8** · 0.49 s |

x10 at c=16 against the default at c=8:

| | default, c=8 | x10, c=16 | Δ |
|---|---|---|---|
| prose aggregate | 159.5 tok/s | 235.0 tok/s | **+47%** |
| structured aggregate | 189.8 tok/s | 254.8 tok/s | **+34%** |

The plan projected +46–50%. Per-stream step time rises from ~104 to 140–152 ms.

**Gates:**
- **T0** `--sizes 2 3 5 8 16` (36 bursts, 226 streams): 0 gated.
- **Preemptions:** `num_preemptions` Δ 0.
- **Memory:** spark1 min available under c=16 was 15.0 GiB.
- **Ruler:** worst cell +1.31%, within 1.5%.
- **Smokes:** 4/4.

**Recipe change:** `THROUGHPUT_PROFILE` (default 0). `run.sh` refuses 9..16 without it, and anything above 16 without `FORCE_UNSAFE_CTX=1`.

## 2. P2-5 step 2: not kept

`bench_mixed.py`; ITL p99 was added (RULER_SET v2).

| Metric | 4800 / 8192 | 3200 / 4832 | Rule |
|---|---|---|---|
| decoder ITL p99 during 128k prefill | 3.003 s | 2.590 s (−14%) | ≤ 0.6 s: **FAIL** |
| decoder ITL p99 during 32k prefill | 1.911 s | 1.294 s | – |
| probe TTFT during long prefill p50 | 4.23 s | 2.90 s | no worse: PASS |
| long-prompt TTFT 32k / 128k | 12.41 / 65.21 s | 12.60 / 64.88 s | ≤ +10%: PASS |
| ruler, every cell | – | worst +0.62% | ≤ 1%: PASS |

At 128k depth, step time grows ~0.45 ms per chunk row, so reaching ≤ 0.6 s would need ~1300-row chunks. The real lever is prefill-kernel cost (X3/X4), not the threshold. 3200/4832 stays available as an opt-in pair of env values.

## 3. T4 for lab-fp8-dense: PASS

`quality/t4_gsm8k_full.py --workers 8` (thinking on, T=0.6, top_p 0.95, top_k 20, seed = 20260929 + item index, max_tokens 16384). Each run took 53–57 min.

| Arm | correct / 1319 | acc | vs ref lost / gained | McNemar p |
|---|---|---|---|---|
| ref (default) | 1290 | 97.80% | – | – |
| **fp8-dense** | 1288 | 97.65% | 6 / 4 | 0.754 |
| final (default A/A) | 1288 | 97.65% | 5 / 3 | 0.727 |

`lab-fp8-dense` now has T0–T4 + V. It still fails T1-B top-1 (94.3%), a gate that is saturated on this model, so promotion is an owner decision (R8).

## Open

- Structured aggregate at 16 clients (+34%) is below the plan's projection.
- Decoder stalls during 128k prefills need prefill-kernel work (X3/X4).
- The lab-fp8-dense promotion decision.
