# Quality gates (plan 0-8 and section 1.3)

Host-side scripts: Python stdlib plus numpy. They only talk to a running serve
over its OpenAI API. None of them starts, stops or touches a container. Each
one writes to `--out DIR`, including a `manifest.json` that records the
server's `/v1/models`, the git rev and dirty flag, argv, and the sha256 of every
quality script and input file. Keep large runs under `~/projects/data/qwen38-evals/runs/`.
Copy only the small JSON verdicts into `evidence/`.

The harness is adapted from the GLM-5.3 recipe's `quality/` (newest worktree).
The GLM `[gMASK]<sop>` raw prompt is replaced by the served Qwen chat template.
Every request goes through `/v1/chat/completions` with thinking off, except T4
and the effort smoke.

| Gate | Capture | Score | Pass rule |
|---|---|---|---|
| T1-A / T1-B | `collect_t1.py --out D` | `score.py t1 REF D [--class B] [--aa AA.json]` | top-1 ≥ 99.5% / 98.5%, mean KL ≤ 0.005 / 0.02, no domain above 2× the mean (C25 floors applied) |
| T1-G | `collect_t1g.py --out D` (40 × 5, c=1) | `score.py t1g LKG D --aa AA.json` | distinct-output mean and first-32 divergence ≤ A/A + 2σ |
| T1-D | `collect_t1d.py --out D` (40 × 1024, top-20) | `score.py t1d GOLD D --floor FLOOR.json` | median first divergence ≥ 0.5 × floor; matched-prefix KL ≤ class threshold |
| T2 | `t2.py --out D` | `score.py t2 REF D [--aa AA.json]` | GSM8K-250 and IFEval-120: McNemar p ≥ 0.05 and drop ≤ 2 pp; tools 60/60 exact arguments; JSON 30/30; repeat-4gram no worse; effort 5/5 |
| T3 | `t3_needles.py --out D [--with-250k]` | self-scoring | 24/24 at ≤128k; at most 2 misses overall |
| V | `vision_probes/run.py --out D` | self-scoring | every synthetic probe exact; c=2 pair right |
| T4 | `t4_gsm8k_full.py --out D` (resumable) | `score.py t4 BASE D` | McNemar not significant and drop ≤ 1.5 pp |

`score.py` exits 0 on PASS or on a report-only result (no floor yet), 1 on FAIL
or ESCALATE, and 2 on INVALID (the item sets or token ids differ).
`--limit N` on every capture gives a smoke run.

## Floors (C25, S1.5)

An A/A floor is the same `score.py` subcommand run on two captures of one
config. Save it with `--json`, then pass it back:

```bash
python3 quality/collect_t1.py --out R/t1-a && python3 quality/collect_t1.py --out R/t1-b
python3 quality/score.py t1 R/t1-a R/t1-b --json R/t1-aa.json          # the floor
python3 quality/score.py t1 R/t1-a R/cand --class B --aa R/t1-aa.json  # the gate
python3 quality/collect_t1d.py --out R/t1d-c8 --workers 8               # golden-vs-golden c=8
python3 quality/score.py t1d R/t1d-gold R/t1d-c8 --json R/t1d-floor.json
python3 quality/score.py threshold --kind agree --stated 0.995 --floor 0.998
```

- For lower-is-better metrics: threshold = max(stated, 2 × floor).
- For agreement metrics: threshold = min(stated, 1 − 2 × (1 − floor)).
- If a class-B top-1 threshold would fall below 97%, the verdict is ESCALATE, not a relaxed pass.
- For T1-G, σ is the A/A standard error over prompts.
- In T1, a domain may exceed 2× the mean only while it stays below 10% of the class threshold. Without that allowance, noise-level KL would trip the per-domain rule.

## Measured on the pinned serve (2026-09-29, df871f17, MTP k=3)

- **The serve is not repeatable at c=1, even idle.** Identical teacher-forced
  requests were sent one after another (4, then 3, then 3 more with a fresh `cache_salt`) on an idle serve (`num_requests_running`
  0 before and after each). Every pair disagreed on 11–14% of top-1 tokens for
  Spanish prose, and target logprobs differed by up to 4.3 nats. This did not
  change with a fresh `cache_salt` each time, so prefix caching is not the
  cause. An idle T1 A/A with 2 items per domain (18,492 tokens) gave:
  - top-1 agreement 93.7% and mean KL 0.033 overall;
  - prose 98.4%, code 98.2%, es 87.1%, de 88.7%, zh 90.9%.

  Greedy generation also diverged run to run at tokens 25–68 of 128. With
  floors this wide, C25 turns T1-A into a top-1 threshold of about 87%. That
  noise has to be understood and removed (S1.5) before T1 can gate anything.
- **Co-prefill corruption (F01) shows up in T2.**
  - Two rep4 prompts sent together: one stream came back as "Register Register ..."
    (repeat-4gram 0.996; 0.0 when sent alone).
  - A `json_schema` request at c=2 returned HTTP 500 (`grammar rejected tokens`,
    backend_xgrammar).

  So T2 and T4 default to `--workers 1`. Use more workers only on a pin with F01
  fixed.
- On the same boot:
  - V passed 16/16, plus the c=2 pair. The straddle probe placed a 4,096-token
    image at tokens 6,728–10,824, which crosses 8,192.
  - T3 passed 12/12 at 4k and 16k.
  - Tools passed 60/60 after three ambiguous prompts were swapped out (see
    `data/t2_ids.json`).
  - JSON-schema passed 30/30 at c=2 in one run; the effort smoke passed 5/5.

## Data

- `data/t1_corpus.jsonl`: 386 chat rows, about 310k scored tokens in pieces of at most 1,000 tokens. It is frozen, and `build_t1_corpus.py` records how it was cut. Domains:
  - prose, code, math, zh/es/de/ja/other languages;
  - json (synthetic);
  - tools (transcripts);
  - long (8 × 16k-token documents; `collect_t1.py --long-whole` sends each one whole, which needs about 8 GB of transient memory at MAX_NUM_BATCHED_TOKENS 8192);
  - image (16 prompts).

  The `ja` domain is thin: 2 pieces from one book.
- `data/prompts40.json`: the 40 prompts that T1-G and T1-D share.
- `data/t2_ids.json`: pinned GSM8K-250 and IFEval-120 ids (seed 20260929, with dataset sha256), plus the 30 tool items.
- `data/tools.json`: 14 tool schemas and 50 prompts (from the GLM harness).
- Datasets that are not committed live under `~/projects/data/qwen38-evals/datasets/` (`QWEN38_EVALS_DIR` overrides). They are downloaded once:

  ```bash
  D=~/projects/data/qwen38-evals/datasets; mkdir -p $D/gsm8k $D/ifeval
  curl -sSfL -o $D/gsm8k/test.jsonl https://raw.githubusercontent.com/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/test.jsonl
  curl -sSfL -o $D/ifeval/input_data.jsonl https://raw.githubusercontent.com/google-research/google-research/26d8ccd/instruction_following_eval/data/input_data.jsonl
  ```

  `t2.py` and `t4_gsm8k_full.py` check both sha256 values against `data/t2_ids.json`.
- `vision_probes/keys.json` pins the pixel sha256 of every probe image and every answer key. After a deliberate probe change, rerun `vision_probes/run.py --write-keys`.

Sources and licenses are in `data/SOURCES.md`. Tests: `python3 -m unittest tests.test_quality`.
