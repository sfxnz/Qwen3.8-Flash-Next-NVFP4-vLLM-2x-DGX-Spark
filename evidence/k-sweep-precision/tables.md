### Ruler (ms/step · acceptance · per-stream tok/s; pooled over every anchor-free ruler run of the boot)

| arm | prose c1 | prose c2 | prose c8 | struct c1 | struct c2 | struct c8 |
|---|---|---|---|---|---|---|
| ref | 52.96 · 2.37 · 44.2 | 61.27 · 2.15 · 34.7 | 81.15 · 2.68 · 32.5 | 54.62 · 4.00 · 72.9 | 58.99 · 4.00 · 67.5 | 75.26 · 4.00 · 52.9 |
| ref2 | 53.29 · 2.37 · 43.9 | 61.59 · 2.15 · 34.6 | 80.41 · 2.68 · 32.8 | 54.40 · 4.00 · 73.2 | 59.25 · 4.00 · 67.2 | 75.39 · 4.00 · 52.8 |
| k2 | 47.39 · 2.13 · 45.0 | 53.71 · 1.96 · 36.5 | 76.41 · 2.14 · 27.7 | 48.65 · 3.00 · 61.1 | 54.70 · 3.00 · 54.3 | 71.07 · 3.00 · 41.8 |
| k4 | 58.73 · 2.58 · 43.0 | 67.15 · 2.20 · 32.1 | 95.43 · 2.73 · 28.4 | 58.87 · 5.00 · 84.5 | 63.47 · 5.00 · 78.4 | 85.56 · 5.00 · 58.1 |
| p40-ssm-bf16 | 52.35 · 2.19 · 41.8 | 59.75 · 2.33 · 38.4 | 75.58 · 2.35 · 30.6 | 53.79 · 4.00 · 74.0 | 57.33 · 4.00 · 69.4 | 70.10 · 4.00 · 56.8 |
| p41-fp8-blk-marlin | 46.59 · 2.69 · 56.6 | 54.25 · 2.26 · 41.4 | 83.06 · 2.33 · 27.7 | 47.50 · 4.00 · 83.8 | 52.40 · 4.00 · 75.9 | 69.74 · 4.00 · 57.1 |
| p41b-fp8-ptpc | 47.44 · 2.40 · 50.6 | 55.70 · 2.14 · 38.1 | 83.87 · 2.18 · 26.9 | 48.24 · 4.00 · 82.5 | 50.84 · 4.00 · 78.3 | 68.68 · 4.00 · 57.9 |
| p41c-fp8-blk-noqkvz | 49.86 · 2.16 · 42.8 | 57.58 · 2.34 · 40.6 | 86.88 · 2.33 · 26.6 | 51.34 · 4.00 · 77.5 | 56.33 · 4.00 · 70.7 | 73.31 · 4.00 · 54.3 |
| final | 53.30 · 2.37 · 43.9 | 61.52 · 2.15 · 34.6 | 80.41 · 2.68 · 32.8 | 54.84 · 4.00 · 72.6 | 59.12 · 4.00 · 67.3 | 74.60 · 4.00 · 53.4 |

### Frozen bench_decode (tok/s; acceptance)

| arm | prose c1 | prose c2 ps/agg | struct c1 | struct c2 ps/agg | acc prose c1/c2 | TTFT p50 c1 |
|---|---|---|---|---|---|---|
| ref | 44.0 | 34.1 / 67.9 | 72.2 | 67.7 / 135.4 | 2.37 / 2.15 | – |
| ref2 | – | – / – | – | – / – | – / – | – |
| k2 | 44.9 | 36.1 / 70.6 | 61.0 | 53.8 / 106.4 | 2.13 / 1.96 | – |
| k4 | 42.9 | 31.9 / 63.5 | 83.6 | 77.6 / 155.1 | 2.58 / 2.20 | – |
| p40-ssm-bf16 | 41.9 | 38.5 / 74.2 | 74.2 | 69.1 / 138.2 | 2.19 / 2.33 | – |
| p41-fp8-blk-marlin | 57.1 | 41.1 / 75.2 | 84.2 | 76.3 / 152.5 | 2.69 / 2.26 | – |
| p41b-fp8-ptpc | 51.7 | 38.8 / 75.2 | 81.9 | 78.0 / 155.8 | 2.40 / 2.14 | – |
| p41c-fp8-blk-noqkvz | 43.2 | 40.8 / 73.0 | 78.3 | 70.8 / 141.6 | 2.16 / 2.34 | – |
| final | 44.2 | 34.2 / 67.9 | 72.5 | 67.9 / 135.7 | 2.37 / 2.15 | – |

### Diverse and sampled (median ms/step · tokens/step · per-stream tok/s)

| arm | diverse/prose/1 | diverse/prose/8 | diverse/structured/1 | diverse/structured/8 | sampled/A/prose/1 | sampled/A/prose/8 | sampled/A/structured/1 | sampled/A/structured/8 | sampled/T1/prose/1 | sampled/T1/prose/8 | sampled/T1/structured/1 | sampled/T1/structured/8 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ref | 53.21 · 2.25 · 42.3 | 103.31 · 2.25 · 21.7 | 53.61 · 3.59 · 66.2 | 107.69 · 3.70 · 34.0 | 53.45 · 2.10 · 38.8 | 111.06 · 2.05 · 18.3 | 54.06 · 3.57 · 65.8 | 107.74 · 3.62 · 33.6 | 53.41 · 1.70 · 31.8 | 66.17 · 1.94 · 29.3 | 54.66 · 3.98 · 72.8 | 66.97 · 3.98 · 59.4 |
| ref2 | 52.73 · 2.25 · 42.7 | 102.89 · 2.26 · 22.0 | 53.03 · 3.59 · 66.9 | 106.05 · 3.62 · 34.5 | – | – | – | – | – | – | – | – |
| k2 | 47.70 · 2.06 · 43.1 | 91.78 · 2.02 · 22.1 | 47.64 · 2.83 · 59.1 | 94.20 · 2.87 · 30.1 | 47.86 · 2.04 · 42.7 | 92.26 · 1.95 · 21.2 | 47.93 · 2.79 · 58.1 | 93.79 · 2.73 · 29.1 | 48.12 · 1.83 · 38.0 | 58.55 · 2.00 · 34.2 | 48.86 · 2.97 · 60.8 | 55.50 · 2.97 · 53.5 |
| k4 | 58.33 · 2.38 · 40.8 | 115.99 · 2.35 · 20.3 | 58.25 · 4.18 · 71.1 | 120.58 · 4.32 · 34.6 | 58.70 · 2.24 · 38.5 | 118.86 · 2.35 · 18.6 | 59.25 · 4.25 · 71.5 | 122.36 · 4.05 · 33.2 | 59.18 · 1.92 · 32.4 | 75.57 · 2.16 · 28.6 | 63.27 · 4.97 · 78.6 | 74.41 · 4.97 · 66.9 |
| p40-ssm-bf16 | 52.61 · 2.26 · 42.8 | 97.84 · 2.26 · 23.1 | 52.70 · 3.57 · 67.5 | 101.81 · 3.64 · 35.8 | – | – | – | – | – | – | – | – |
| p41-fp8-blk-marlin | 46.22 · 2.28 · 49.2 | 96.14 · 2.24 · 23.2 | 46.49 · 3.54 · 76.8 | 99.28 · 3.45 · 35.8 | – | – | – | – | – | – | – | – |
| p41b-fp8-ptpc | – | – | – | – | – | – | – | – | – | – | – | – |
| p41c-fp8-blk-noqkvz | – | – | – | – | – | – | – | – | – | – | – | – |
| final | 53.38 · 2.25 · 41.8 | 104.84 · 2.23 · 21.1 | 53.57 · 3.59 · 66.3 | 107.06 · 3.67 · 34.1 | 53.51 · 2.10 · 38.9 | 110.11 · 2.13 · 18.8 | 53.81 · 3.57 · 66.5 | 109.04 · 3.70 · 34.0 | 54.34 · 1.70 · 31.2 | 69.66 · 1.82 · 26.1 | 55.66 · 3.98 · 71.5 | 69.67 · 3.98 · 57.1 |

### Keep rule vs base (geomean of per-stream tok/s ratios; ≥ +3% and no cell below base − spread)

| arm | cells | geomean | worst cell | cells below spread | verdict |
|---|---|---|---|---|---|
| ref2 | 10 | +0.42% | ruler/prose/1 -0.6% | ruler/prose/1 | no |
| k2 | 18 | -4.89% | ruler/structured/8 -21.0% | diverse/structured/1, diverse/structured/8, ruler/prose/8, ruler/structured/1, ruler/structured/2, ruler/structured/8, sampled/A/structured/1, sampled/A/structured/8, sampled/T1/structured/1, sampled/T1/structured/8 | no |
| k4 | 18 | +2.27% | ruler/prose/8 -12.7% | diverse/prose/1, diverse/prose/8, ruler/prose/1, ruler/prose/2, ruler/prose/8, sampled/T1/prose/8 | no |
| p40-ssm-bf16 | 10 | +2.41% | ruler/prose/8 -5.9% | ruler/prose/1 | no |
| p41-fp8-blk-marlin | 10 | +10.68% | ruler/prose/8 -14.6% | ruler/prose/8 | no |
| p41b-fp8-ptpc | 6 | +6.96% | ruler/prose/8 -17.2% | ruler/prose/8 | no |
| p41c-fp8-blk-noqkvz | 6 | +1.02% | ruler/prose/8 -17.9% | ruler/prose/1, ruler/prose/8 | no |
| final | 18 | -0.99% | sampled/T1/prose/8 -10.9% | diverse/prose/8, ruler/prose/1, sampled/T1/prose/1, sampled/T1/prose/8, sampled/T1/structured/1, sampled/T1/structured/8 | no |

<details><summary>ref2 per cell</summary>

| cell | base tok/s ± spread | arm tok/s | ratio |
|---|---|---|---|
| diverse/prose/1 | 42.3 ± 1.4 | 42.7 | +0.8% |
| diverse/prose/8 | 21.7 ± 0.6 | 22.0 | +1.4% |
| diverse/structured/1 | 66.2 ± 5.4 | 66.9 | +1.0% |
| diverse/structured/8 | 34.0 ± 2.9 | 34.5 | +1.6% |
| ruler/prose/1 | 44.2 ± 0.1 | 43.9 | -0.6% **below** |
| ruler/prose/2 | 34.7 ± 0.4 | 34.6 | -0.5% |
| ruler/prose/8 | 32.5 ± 3.6 | 32.8 | +0.9% |
| ruler/structured/1 | 72.9 ± 0.5 | 73.2 | +0.4% |
| ruler/structured/2 | 67.5 ± 0.7 | 67.2 | -0.4% |
| ruler/structured/8 | 52.9 ± 0.4 | 52.8 | -0.2% |

</details>

<details><summary>k2 per cell</summary>

| cell | base tok/s ± spread | arm tok/s | ratio |
|---|---|---|---|
| diverse/prose/1 | 42.3 ± 1.4 | 43.1 | +1.7% |
| diverse/prose/8 | 21.7 ± 0.6 | 22.1 | +1.5% |
| diverse/structured/1 | 66.2 ± 5.4 | 59.1 | -10.7% **below** |
| diverse/structured/8 | 34.0 ± 2.9 | 30.1 | -11.4% **below** |
| ruler/prose/1 | 44.2 ± 0.1 | 45.0 | +1.8% |
| ruler/prose/2 | 34.7 ± 0.4 | 36.5 | +5.0% |
| ruler/prose/8 | 32.5 ± 3.6 | 27.7 | -14.7% **below** |
| ruler/structured/1 | 72.9 ± 0.5 | 61.1 | -16.2% **below** |
| ruler/structured/2 | 67.5 ± 0.7 | 54.3 | -19.5% **below** |
| ruler/structured/8 | 52.9 ± 0.4 | 41.8 | -21.0% **below** |
| sampled/A/prose/1 | 38.8 ± 1.8 | 42.7 | +10.0% |
| sampled/A/prose/8 | 18.3 ± 1.6 | 21.2 | +15.4% |
| sampled/A/structured/1 | 65.8 ± 3.2 | 58.1 | -11.6% **below** |
| sampled/A/structured/8 | 33.6 ± 1.7 | 29.1 | -13.2% **below** |
| sampled/T1/prose/1 | 31.8 ± 0.0 | 38.0 | +19.8% |
| sampled/T1/prose/8 | 29.3 ± 0.0 | 34.2 | +16.5% |
| sampled/T1/structured/1 | 72.8 ± 0.0 | 60.8 | -16.5% **below** |
| sampled/T1/structured/8 | 59.4 ± 0.0 | 53.5 | -9.9% **below** |

</details>

<details><summary>k4 per cell</summary>

| cell | base tok/s ± spread | arm tok/s | ratio |
|---|---|---|---|
| diverse/prose/1 | 42.3 ± 1.4 | 40.8 | -3.6% **below** |
| diverse/prose/8 | 21.7 ± 0.6 | 20.3 | -6.7% **below** |
| diverse/structured/1 | 66.2 ± 5.4 | 71.1 | +7.3% |
| diverse/structured/8 | 34.0 ± 2.9 | 34.6 | +1.9% |
| ruler/prose/1 | 44.2 ± 0.1 | 43.0 | -2.7% **below** |
| ruler/prose/2 | 34.7 ± 0.4 | 32.1 | -7.7% **below** |
| ruler/prose/8 | 32.5 ± 3.6 | 28.4 | -12.7% **below** |
| ruler/structured/1 | 72.9 ± 0.5 | 84.5 | +16.0% |
| ruler/structured/2 | 67.5 ± 0.7 | 78.4 | +16.2% |
| ruler/structured/8 | 52.9 ± 0.4 | 58.1 | +9.9% |
| sampled/A/prose/1 | 38.8 ± 1.8 | 38.5 | -0.9% |
| sampled/A/prose/8 | 18.3 ± 1.6 | 18.6 | +1.7% |
| sampled/A/structured/1 | 65.8 ± 3.2 | 71.5 | +8.7% |
| sampled/A/structured/8 | 33.6 ± 1.7 | 33.2 | -1.0% |
| sampled/T1/prose/1 | 31.8 ± 0.0 | 32.4 | +1.9% |
| sampled/T1/prose/8 | 29.3 ± 0.0 | 28.6 | -2.4% **below** |
| sampled/T1/structured/1 | 72.8 ± 0.0 | 78.6 | +8.0% |
| sampled/T1/structured/8 | 59.4 ± 0.0 | 66.9 | +12.5% |

</details>

<details><summary>p40-ssm-bf16 per cell</summary>

| cell | base tok/s ± spread | arm tok/s | ratio |
|---|---|---|---|
| diverse/prose/1 | 42.3 ± 1.4 | 42.8 | +1.1% |
| diverse/prose/8 | 21.7 ± 0.6 | 23.1 | +6.1% |
| diverse/structured/1 | 66.2 ± 5.4 | 67.5 | +1.9% |
| diverse/structured/8 | 34.0 ± 2.9 | 35.8 | +5.2% |
| ruler/prose/1 | 44.2 ± 0.1 | 41.8 | -5.5% **below** |
| ruler/prose/2 | 34.7 ± 0.4 | 38.4 | +10.6% |
| ruler/prose/8 | 32.5 ± 3.6 | 30.6 | -5.9% |
| ruler/structured/1 | 72.9 ± 0.5 | 74.0 | +1.5% |
| ruler/structured/2 | 67.5 ± 0.7 | 69.4 | +2.9% |
| ruler/structured/8 | 52.9 ± 0.4 | 56.8 | +7.4% |

</details>

<details><summary>p41-fp8-blk-marlin per cell</summary>

| cell | base tok/s ± spread | arm tok/s | ratio |
|---|---|---|---|
| diverse/prose/1 | 42.3 ± 1.4 | 49.2 | +16.2% |
| diverse/prose/8 | 21.7 ± 0.6 | 23.2 | +6.9% |
| diverse/structured/1 | 66.2 ± 5.4 | 76.8 | +15.9% |
| diverse/structured/8 | 34.0 ± 2.9 | 35.8 | +5.1% |
| ruler/prose/1 | 44.2 ± 0.1 | 56.6 | +28.1% |
| ruler/prose/2 | 34.7 ± 0.4 | 41.4 | +19.3% |
| ruler/prose/8 | 32.5 ± 3.6 | 27.7 | -14.6% **below** |
| ruler/structured/1 | 72.9 ± 0.5 | 83.8 | +15.0% |
| ruler/structured/2 | 67.5 ± 0.7 | 75.9 | +12.6% |
| ruler/structured/8 | 52.9 ± 0.4 | 57.1 | +7.9% |

</details>

<details><summary>p41b-fp8-ptpc per cell</summary>

| cell | base tok/s ± spread | arm tok/s | ratio |
|---|---|---|---|
| ruler/prose/1 | 44.2 ± 0.1 | 50.6 | +14.4% |
| ruler/prose/2 | 34.7 ± 0.4 | 38.1 | +9.8% |
| ruler/prose/8 | 32.5 ± 3.6 | 26.9 | -17.2% **below** |
| ruler/structured/1 | 72.9 ± 0.5 | 82.5 | +13.2% |
| ruler/structured/2 | 67.5 ± 0.7 | 78.3 | +16.0% |
| ruler/structured/8 | 52.9 ± 0.4 | 57.9 | +9.6% |

</details>

<details><summary>p41c-fp8-blk-noqkvz per cell</summary>

| cell | base tok/s ± spread | arm tok/s | ratio |
|---|---|---|---|
| ruler/prose/1 | 44.2 ± 0.1 | 42.8 | -3.2% **below** |
| ruler/prose/2 | 34.7 ± 0.4 | 40.6 | +17.0% |
| ruler/prose/8 | 32.5 ± 3.6 | 26.6 | -17.9% **below** |
| ruler/structured/1 | 72.9 ± 0.5 | 77.5 | +6.4% |
| ruler/structured/2 | 67.5 ± 0.7 | 70.7 | +4.7% |
| ruler/structured/8 | 52.9 ± 0.4 | 54.3 | +2.7% |

</details>

<details><summary>final per cell</summary>

| cell | base tok/s ± spread | arm tok/s | ratio |
|---|---|---|---|
| diverse/prose/1 | 42.3 ± 1.4 | 41.8 | -1.2% |
| diverse/prose/8 | 21.7 ± 0.6 | 21.1 | -2.8% **below** |
| diverse/structured/1 | 66.2 ± 5.4 | 66.3 | +0.0% |
| diverse/structured/8 | 34.0 ± 2.9 | 34.1 | +0.2% |
| ruler/prose/1 | 44.2 ± 0.1 | 43.9 | -0.6% **below** |
| ruler/prose/2 | 34.7 ± 0.4 | 34.6 | -0.4% |
| ruler/prose/8 | 32.5 ± 3.6 | 32.8 | +0.9% |
| ruler/structured/1 | 72.9 ± 0.5 | 72.6 | -0.4% |
| ruler/structured/2 | 67.5 ± 0.7 | 67.3 | -0.2% |
| ruler/structured/8 | 52.9 ± 0.4 | 53.4 | +0.9% |
| sampled/A/prose/1 | 38.8 ± 1.8 | 38.9 | +0.2% |
| sampled/A/prose/8 | 18.3 ± 1.6 | 18.8 | +2.2% |
| sampled/A/structured/1 | 65.8 ± 3.2 | 66.5 | +1.1% |
| sampled/A/structured/8 | 33.6 ± 1.7 | 34.0 | +1.3% |
| sampled/T1/prose/1 | 31.8 ± 0.0 | 31.2 | -1.7% **below** |
| sampled/T1/prose/8 | 29.3 ± 0.0 | 26.1 | -10.9% **below** |
| sampled/T1/structured/1 | 72.8 ± 0.0 | 71.5 | -1.8% **below** |
| sampled/T1/structured/8 | 59.4 ± 0.0 | 57.1 | -3.9% **below** |

</details>
