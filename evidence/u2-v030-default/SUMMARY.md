# U2: v0.30.0 + determinism as the recipe default, plus P2-5 (session 3, 2026-09-29)

**Result:** the new default passes the full gate pack and is serving (boot `c-final`, GATE=PASS).

The default is now vLLM v0.30.0 by digest, with the three S1.5 determinism overlays and the #56067 `serving.py` rider. `run.sh` also adds, on every boot:
- `VLLM_PLE_CPU_OFFLOAD=0` and `VLLM_USE_BREAKABLE_CUDAGRAPH=0` on both ranks;
- `--async-scheduling` and `--mm-processor-cache-gb 1`;
- vision caps (`--mm-processor-kwargs`, `--limit-mm-per-prompt`);
- the alias chat template;
- `--long-prefill-token-threshold 4800` (P2-5, kept);
- `--ulimit core=1`.

Two defects surfaced and were fixed:
1. The S1.5 QSA tie-repair overlay could scatter out of bounds on long prefills. It raised a device assert at 64k.
2. A worker abort let host apport buffer an ~11 GiB core in RAM, which tripped memguard.

Per-step commands are in `commands.sh`. Raw data is in `~/projects/data/qwen38-evals/runs/session3/`.

| Dir | Config | Boot | KV cache |
|---|---|---|---|
| `pre/` | the live U1 + 3 determinism overlays (no serving.py, no caps): toolstream and vision baselines | – | – |
| `a-default/` | new default, first QSA overlay | 11.5 min | 1,650,798 tok (6.30x) |
| `a2-default/` | new default, fixed QSA overlay, `--ulimit core=1` | 11.5 min | 1,626,709 tok (6.21x) |
| `b-lpt4800/` | a2 + `--long-prefill-token-threshold 4800` | 11.3 min | 1,595,535 tok (6.09x) |
| `c-final/` | rendered final default, **left serving** | 11.5 min | 1,655,049 tok (6.31x) |

Boot needles (every boot): `Using FlashInfer GDN prefill kernel`, PLE `pinned=False`, drafter `Using TRITON Fp8 MoE backend`. There were no errors apart from the a-default crash.

## Decode (frozen `bench_decode.py`, 3-run medians)

| Cell | U1 (session 1) | a-default | a2-default | **c-final** |
|---|---|---|---|---|
| prose c1 tok/s (acc) | 40.2 (2.49) | 38.8 (2.37) | 38.8 (2.37) | **38.7 (2.37)** |
| prose c2 per-stream / agg (acc) | 34.4 / 68.0 (2.27) | 32.0 / 63.3 (2.15) | 32.3 / 63.5 (2.15) | **31.9 / 64.5 (2.15)** |
| structured c1 tok/s | 65.2 | 63.4 | 64.5 | **64.7** |
| structured c2 per-stream / agg | 63.2 / 126.3 | 62.5 / 125.0 | 62.9 / 125.8 | **62.9 / 125.8** |
| TTFT p50 prose / struct c1 | 0.16 / 0.16 s | 0.16 / 0.16 | 0.16 / 0.16 | 0.16 / 0.16 |

- Prose tok/s is not comparable across the determinism fix. Greedy output is pinned, so acceptance is fixed at 2.37 (c1) and 2.15 (c2); U1's 2.49 was a lucky draw.
- Structured c1 is 0.5–1.8 tok/s below U1. That is the known +1.4 ms/step cost of the determinism fixes (s15).

## Ruler (`ruler_steps.py`, ms/step; agg tok/s in brackets)

| Cell | U1 (anchor-free) | s15 final | a2 anchor-free | a2 anchored | b-lpt4800 | c-final |
|---|---|---|---|---|---|---|
| prose c1 | 59.91 | 59.48 | 60.37 | 59.97 | 60.45 | 60.63 |
| prose c2 | 63.66 (69.2) | – | 66.10 (63.7) | 65.28 (64.5) | 66.51 (63.1) | 66.26 (63.6) |
| prose c8 | 92.08 (177.7) | 84.72 | 85.92 (225.2) | 86.65 (223.2) | 87.25 (221.8) | 86.75 (223.1) |
| structured c1 | 60.97 | 61.34 | 62.10 | 62.28 | 61.51 | 62.56 |
| structured c2 | 62.58 (127.1) | – | 63.52 (125.3) | 63.56 (125.2) | 63.22 (125.9) | 63.55 (125.2) |
| structured c8 | 81.03 (392.7) | 79.29 | 79.84 (398.6) | 80.07 (397.4) | 80.22 (396.7) | 80.28 (396.4) |

- Prose c8 aggregate rose from 177.7 to about 223, because acceptance is pinned at 2.56.
- **Sentinel anchor on this base: `--anchor-ms 60.7 62.7`.** It is re-centred on pooled sentinel values, because boot-to-boot drift is about 1 ms. With it, a2 had 0/9 excursions, b 0/9 and c-final 1/12.

## Companion rulers (a-default)

| Ruler | U1 | a-default |
|---|---|---|
| diverse prose c1 / c8 ms (acc) | 59.75 / 107.3 (2.24 / 2.26) | 60.74 / 108.23 (2.26 / 2.27) |
| diverse structured c1 / c8 ms (acc) | 59.69 / 111.5 (3.14 / 3.12) | 61.21 / 111.80 (3.15 / 3.13) |
| sampled A c1 / c8 prose ms (acc) | 60.3 / 116.9 (2.64 / 2.69) | 60.96 / 115.17 (2.66 / 2.67) |

## Correctness and quality

| Gate | U1 | U2 |
|---|---|---|
| smokes | 4/4 | **4/4** |
| T0 `bench_probe` (30 bursts) | 0 gated | **0 gated** (3 boots) |
| T1-G self (10 × 5) | 5.0 distinct / 0.52 | **1.0 / 0.0** |
| T2 tools / json / rep4 / effort | 57/60, 30/30, 0.00069, 5/5 | **59/60**, 30/30, 0.000694, 5/5 |
| T2 IFEval-120 / GSM8K-250 | 88.3% / 96.8% | **90.0% / 95.6%** |
| reasoning_effort high / max / minimal | high → 400 | **all 200** |
| T3 needles | 18/18 (to 32k) | **30/30 at 4k–128k** (a2) |
| V vision probes | not run | **16/16 + c=2 pair**, incl. the C29 straddle |
| prefix reuse at 16k | repeat 4.67 → 0.95 s | repeat 5.08 → 1.09 s; chat 5.11 → 0.67 s |

- **T2 paired vs U1:** GSM8K −1.2 pp (3 lost / 0 gained, p 0.25; within the ≤2 pp rule), IFEval +1.7 pp, tools +3.3 pp. The one tools miss is `t11.nonstream`, the same ambiguous item U1 missed.
- **T3 cold TTFT:** 4k 1.5–2.3 s, 16k 5.1–6.1 s, 32k 11.0–11.4 s, 64k 21.1–25.9 s, 128k 53.2–62.2 s.

## Vision caps (P2-3)

| Case | before: mm tokens / TTFT / decode stall | **U2** |
|---|---|---|
| sq1024 | 1,026 / 0.86 s / 0.51 s | 1,026 / 0.80 s / 0.51 s |
| photo 4032×3024 | 11,846 / 9.14 s / 5.97 s | **4,017 / 3.09 s / 1.21 s** |
| negative 4608×3456 | 15,554 / 11.60 s / 7.84 s | **4,017 / 2.63 s / 1.20 s** |
| video 30 × 720p | 12,102 / 7.18 s / 3.92 s | 10,932 / 6.79 s / 3.72 s |

## serving.py (#56067; 16k-token streamed tool call)

| | stock | **with overlay** |
|---|---|---|
| API-server CPU | ≈325 CPU-s | **≈65 CPU-s (−80%)** |
| tool gap p50, last quarter | 16.5 s | **70 ms** |
| concurrent text gap p99, last quarter | 18.4 s | **108 ms** |

## P2-5 (`bench_mixed`; a2 baseline vs threshold 4800)

| | a2 | **b-lpt4800** |
|---|---|---|
| probe TTFT during 32k / 128k prefill | 9.39 / **61.30 s** | ~4.5 / **5.20 s** |
| decoder ITL during 128k p50 / max | 2.76 / 7.40 s | **2.10 / 3.00 s** |
| 32k / 128k prefill TTFT | 12.49 / 63.30 s | 13.16 (+5.4%) / 64.68 s (+2.2%) |
| arm A short TTFT p50 / p99 | 0.371 / 0.442 s | 0.368 / 0.573 s |

- The ruler is non-inferior: geomean +0.26%, every cell within 1% of the pooled default runs.
- **Kept** as `LONG_PREFILL_TOKEN_THRESHOLD: 4800`.
- Open: the ITL p99 ≤ 0.6 s target during a 128k prefill is not met; 3200/4832 is the next step.

## Incident: T3 64k on a-default

1. At 19:40 both ranks raised `ScatterGatherKernel.cu:203 index out of bounds` in `_deterministic_topk_ties`. Its threshold came from persistent_topk's picks, including a `-1` pad, so `out.scatter_` could write past `block_topk + 1` columns.
2. The workers aborted. apport read the ~11.4 GiB core into RAM on each node, and spark1 fell to 1.0 GiB available.
3. At 19:42 memguard killed the serve as designed. There was no thrash; sshd, LAIL and Hermes stayed up.

**Fixes:**
- The threshold now comes from an exact `torch.topk`, with a slot clamp. Regression test: `test_tie_repair_ignores_bad_topk_picks`.
- `run.sh` passes `--ulimit core=1`; the kernel logs "RLIMIT_CORE is set to 1, aborting core".
- Re-verified on a2: T3 30/30 to 128k, T1-G 1.0, V 16/16, T0 0 gated.

## Memory (`free -h`, available / swap used)

| When | spark1 | spark2 |
|---|---|---|
| a2 after boot | 16 / 8.2 GiB | 19 / 4.1 |
| a2 after T3 128k | 14 / 8.0 | 17 / 4.1 |
| c-final end (serving) | 14 / 8.0 | 16 / 4.0 |

## Not run / open

- Sentinel TOST and the class table (no profiling on the served default).
- T1-A against a spec-off golden, and T4.
- The 32k `bench_structured` cell and the xgrammar microbench.
- T3 at 250k; P2-5 at 3200.
- An `--async-scheduling` A/B.
- A T2 re-run after the QSA fix.
- A boot of the pin rollback with the new generic flags (validated only).
