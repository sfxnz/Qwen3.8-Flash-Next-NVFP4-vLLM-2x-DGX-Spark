# Step 0b: L6a drafter Triton MoE tune (spark2, one GB10, serve down), 2026-09-29

- `step0-l6a.sh` ran `tune_draft_moe.sh tune` one batch size at a time in draft-priority order
  (1, 2, 4, 8, 16, 32, ...), time-boxed to 45 min. The M=1 search alone (960 configs; the tail
  configs with 4-5 stages took 30-60 s each) took **50.5 min**; the time box was stretched by ~6 min
  (the `timeout` wrapper was removed so the finished search was not lost) and no other size was run.
- `merge_l6a.py`: the tuned M=1 entry replaces the seed's; every other key stays the B200 seed.
  Result: `install/…json` (installed as `docker/v030/moe_configs/…json`; `check` OK).
  M=1: seed {M16,N32,K64,G1,w2,s5} → tuned {M16,N32,K64,G64,w4,s3}.
- `tune_draft_moe.sh bench` ABAB on spark1 (`bench/bench.log`), kernel us, stock default → installed:

| M | stock default | installed | Δ |
|---|---|---|---|
| 1 | 82.07 / 82.31 | 80.62 / 80.34 | −1.8% |
| 2 | 246.2 / 247.9 | 242.4 | −1.6% |
| 4 | 466.7 / 468.6 | 458.2 | −1.9% |
| 8 | 922.9 / 927.0 | 897.1 | −2.9% |
| 32 | 2804.7 / 2807.7 | 2736.1 | −2.5% |

Per step at c=1 (k=3: one M=4 + two M=1 drafter MoE calls) that is ≈ 12 µs; at c=8 (M=32 + 2×M=8)
≈ 120 µs. The plan's L6a gate (draft-graph class time −≥0.1 ms) is not reachable at c=1 from this
microbench; it rides in arm A (config-only) and is judged with it.
