# Quality data: sources and licenses

These files are frozen. A T1 capture is comparable only over the same bytes:
`collect_t1.py` records the corpus sha256 in its manifest, and `score.py`
refuses items whose prompt token ids differ.

## t1_corpus.jsonl (T1 teacher-forced corpus)

386 chat rows, cut with the served tokenizer (fab0aec, via `/tokenize`) into
assistant pieces of at most 1,000 tokens. `build_t1_corpus.py` records the cut.

| Domain | Rows | Tokens | Source | License |
|---|---:|---:|---|---|
| prose | 79 | 70,391 | Project Gutenberg, 12 English books (from the GLM-5.3 frozen corpus) | Public domain (US) |
| zh / es / de / ja / multi | 10 / 6 / 6 / 2 / 18 | 38,341 | Project Gutenberg: zh 24264, 23962, 23950; es 2000; de 22367, 2229; ja 1982; fr, it, pt, fi, nl | Public domain (US) |
| code | 55 | 39,851 | CPython 3.12.3 `Lib/`, Go 1.22.0 `src/`, SQLite 3.45.0 `src/`, lodash 4.17.21, Rust 1.77.0 `library/` | PSF-2.0, BSD-3-Clause, public domain, MIT, MIT OR Apache-2.0 |
| math | 34 | 26,242 | GSM8K train worked solutions; MATH train (EleutherAI/hendrycks_math) | MIT, MIT |
| long | 128 (8 docs x 16) | 122,119 | Gutenberg 1342, 2701, 1661, 84, 98, 345, 205, 145, from 10% of the body | Public domain (US) |
| json | 16 | 12,391 | synthetic, seeded (`build_t1_corpus.py`) | written for this repo |
| tools | 16 | - | transcripts built from `tools.json` | written for this repo |
| image | 16 | - | `vision_probes/probes.py` images with fixed descriptions | written for this repo |

The texts from the GLM-5.3 corpus are that repo's `quality/data/corpus.jsonl`
(the GLM chat-shaped rows are left out), re-cut to Qwen tokens. Each row keeps
its exact `source` and `license`. The training data almost certainly contains
these texts. Use only relative comparisons between two serve configs.

## tools.json

Copied from the GLM-5.3 harness (written for that repo by the same owner):
14 tool schemas and 50 prompts, each with its expected call. T2 uses 30 items
whose expected arguments are complete (`t2_ids.json`).

## t2_ids.json

Item ids only, drawn with seed 20260929. The datasets live outside the repo
(see README) and are sha256-pinned here:

| Task | Source | License |
|---|---|---|
| GSM8K | openai/grade-school-math `test.jsonl` @3101c7d | MIT |
| IFEval | google-research `instruction_following_eval/data/input_data.jsonl` @26d8ccd (510 of 541 prompts are supported by `ifeval.py`) | Apache-2.0 |

`quality/ifeval.py` is the GLM-5.3 harness's stdlib port of the IFEval checks
(google-research, Apache-2.0, Copyright The Google Research Authors).

## License notices for vendored code excerpts

**CPython** (`Lib/heapq.py`, `bisect.py`, `textwrap.py`, `fractions.py`,
`statistics.py`, `functools.py`, `json/decoder.py`, `shlex.py` at v3.12.3):
Copyright (c) 2001-2024 Python Software Foundation; All Rights Reserved.
Used under the PSF License Agreement for Python 3.12
(https://docs.python.org/3.12/license.html).

**Go** (`src/sort/sort.go`, `src/container/heap/heap.go`,
`src/strings/builder.go`, `src/bufio/scan.go` at go1.22.0):

    Copyright (c) 2009 The Go Authors. All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

       * Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
       * Redistributions in binary form must reproduce the above
    copyright notice, this list of conditions and the following disclaimer
    in the documentation and/or other materials provided with the
    distribution.
       * Neither the name of Google Inc. nor the names of its
    contributors may be used to endorse or promote products derived from
    this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
    A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

**SQLite** (`src/func.c`, `src/util.c` at version-3.45.0): public domain.

**lodash** (`lodash.js` at 4.17.21): Copyright OpenJS Foundation and other
contributors <https://openjsf.org/>, MIT License.

**Rust** (`library/core/src/iter/adapters/zip.rs`,
`library/alloc/src/collections/binary_heap/mod.rs`,
`library/core/src/str/pattern.rs` at 1.77.0): Copyright The Rust Project
Developers, dual-licensed MIT OR Apache-2.0; used under MIT.

**GSM8K**: Copyright (c) 2021 OpenAI, MIT License.
**MATH**: Copyright (c) 2021 Dan Hendrycks, MIT License.

MIT License text (applies to lodash, Rust under MIT, GSM8K, MATH):

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
