# Boot test of the rendered FP8_DENSE=per_block opt-in (2026-09-30)

`FP8_DENSE=per_block ./run.sh` booted on both Sparks from the rendered recipe (commit 81990e1). It needed no hand-set overlays or EXTRA_*.
- **Log lines on both ranks** (`needles.txt`): `qwen38-fp8-dense: mode=per_block`, the Marlin `fp8_block_w8a8` override and `MarlinFP8ScaledMMLinearKernel`.
- **Smokes:** 4/4.

Frozen `bench_decode.py` (`bench.txt`, `bench-table.txt`), reproducing session 5's p41 arm (57.1 / 84.2):

| Cell | tok/s (acceptance) |
|---|---|
| prose c1 | 56.0 (2.69) |
| prose c2 | 40.1 per stream / 75.2 agg |
| structured c1 | 82.5 (4.0) |
| structured c2 | 76.2 per stream / 152.4 agg |

The lossless default was then restored with `./run.sh`: smokes 4/4, health 200, no FP8_DENSE env (`restore-*.txt`).
