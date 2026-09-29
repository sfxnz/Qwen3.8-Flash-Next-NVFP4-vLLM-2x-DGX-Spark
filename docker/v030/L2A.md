# L2a: NCCL graph mixing off, with an eager twin communicator (v0.30.0)

Plan item L2a (F13). Booked at **0.89 ms/step at c=1** and 1.02 ms at c=8 from S1.1 on v0.30 (GO,
`evidence/s11-micro/gonogo.md`). This is a booking, not a measurement. Nothing here has run on a GPU yet.

## What it does

`NCCL_GRAPH_MIXING_SUPPORT=0` makes each in-graph all-reduce 14-19 µs cheaper (S1.1, 5-160 KB).
The env var alone is unsafe. With mixing off, NCCL does not support an eager collective launched while a
graph that uses the same communicator is still running. The serve does that on every step: the embedding
all-reduce and the logits/draft all-gathers run outside the FULL decode graph. The sibling's
`flags.md` has the same finding in its `NCCL_GRAPH_MIXING_SUPPORT=0` row.

With `VLLM_QWEN38_NCCL_TWIN=1`, the overlay does four things:

1. It writes `NCCL_GRAPH_MIXING_SUPPORT=0` when `cuda_communicator` is imported. That happens before the
   process's first NCCL communicator exists.
2. It gives the TP group a second `PyNcclCommunicator`, the twin. `pynccl_comm` becomes a router:
   - a call whose stream is capturing goes to the stock (graph) communicator;
   - every other call goes to the twin.

   So the graph communicator only runs captured collectives, and the twin only runs eager ones.
3. Every other group with a PyNccl communicator gets a guard. It passes eager calls through, and it
   raises if a call is captured.
4. It writes per-rank audit lines (see below).

Numerics do not change: the same algorithm, protocol and channels run on the same buffers.

| File | Role |
|---|---|
| `nccl_twin.py` | Router, `attach()`, `install()`, audit lines. Imports only stdlib at the top level. |
| `apply_nccl_twin_overlay.py` | Generator. It refuses any stock sha other than `102d95c6…`. It adds one hook line in `CudaCommunicator.__init__` and embeds `nccl_twin.py` verbatim at the end of the module. |
| `nccl_twin_cuda_communicator.py` | Generated overlay for `vllm/distributed/device_communicators/cuda_communicator.py`. Do not hand-edit it. |
| `nccl_twin_audit.py` | Engagement audit over the `docker logs` of each rank. |
| `tests/test_nccl_twin.py` | Host tests use fakes. `TestInImage` runs in the CPU-only v0.30 container with the overlay mounted. |

Regenerate the overlay with `python3 docker/v030/apply_nccl_twin_overlay.py` (from the image) or with
`--src <stock file>`. Check it is current with `--check`.

## Fail closed

The overlay fails closed at two points.

**At import** (`install`, stock path, nothing patched). The overlay disarms unless all of these hold:
- the lever is `1`;
- the NCCL that vLLM loads is **2.30.7**. The image ships `nvidia-nccl-cu13 2.30.7`, and `NCCLLibrary`
  reports `2.30.7`/`23007`. This is the same release whose source the sibling read.
- `VLLM_USE_NCCL_SYMM_MEM` is off;
- `CudaCommunicator` and `PyNcclCommunicator` match the routed API. Any new public PyNccl method disarms it.

Whenever the overlay does not arm (lever off, or disarmed), it resets any `NCCL_GRAPH_MIXING_SUPPORT`
other than `1` to `1`, NCCL's default. NCCL parses the value with `strtoll`, so `00` or `0x0` also mean 0, and a
disarmed worker can inherit `0` from an armed parent process.

**At communicator init** (`attach`, after arming). Mixing is already off in the process by then, so the
overlay never falls back to a lone stock communicator. Each rank sends `(armed, NCCL version)` to the
others over the group's gloo group. The overlay raises `REFUSED` and stops the boot in any of these cases:
- only some ranks armed;
- the ranks report different or unvalidated NCCL versions;
- a group has more than one rank but no working PyNccl;
- the twin fails its self-test. The test is an eager all-reduce of `rank+1`, which must equal `n(n+1)/2`.

The gloo exchange runs even when the overlay is not armed. It is the only change to the stock path: one
CPU all-gather per communicator at init. It exists so that an unarmed rank can never leave its peer
waiting inside the twin's creation.

## Engagement audit (the gate before any timing)

The sibling saw 0 of 434 NCCL lines. The cause is in NCCL 2.30.7: both NCCL lines below are **INFO**,
not WARN (`src/misc/param.cc:99`, `src/plugin/profiler.cc:339-343,372-376`). The sibling booted with
`NCCL_DEBUG=WARN`, so they could never print. Boot the audit run with
`NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=ENV`.

Each rank logs these lines:
- `qwen38: nccl twin engaged on tp:0 rank=R/2 nccl=2.30.7 NCCL_GRAPH_MIXING_SUPPORT=0 ...`, once.
- `qwen38: nccl twin guard on <group> ...`, once for each other PyNccl group.
- `qwen38: nccl twin graph <k> captured=+N captured_nccl=+M on tp:0 rank=R eager=... total_captured=...`,
  once for every CUDA graph the process captures. This comes from a wrapped
  `torch.cuda.graphs.CUDAGraph.capture_end`, which is wrapped only when the overlay is armed.
- `qwen38: nccl twin eager on tp:0 rank=R eager=E ...`, when the twin's eager call count reaches
  1, 10, 100 and so on.

NCCL's own lines:
- `NCCL_GRAPH_MIXING_SUPPORT set by environment to 0`, once per process. It proves the env reached NCCL
  before its first communicator.
- `... graphUsageMode is set to 0 but the user is capturing graphs ...`, once per captured NCCL API call
  on a mode-0 communicator.

`nccl_twin_audit.py` runs these checks on each rank:
- exactly one `engaged` line;
- no `DISARMED` or `REFUSED` line;
- one graph line per captured graph (`--graphs N`), with more than 0 captured collectives;
- at least one eager line;
- NCCL's env line is present;
- the number of NCCL capture lines equals the captured NCCL calls the router counted;
- the per-graph counts are identical on both ranks.

Fewer NCCL lines than counted calls means the graph communicator is not in mode 0. More NCCL lines means
something else captured a collective on a mode-0 communicator.

## GPU validation (not run; exact commands)

Run it on a session that owns both GPUs. Nothing else may be serving.

```bash
cd ~/projects/ai-lab/recipes/Qwen3.8-Flash-Next-NVFP4-vLLM-2x-DGX-Spark-plan
python3 -m unittest discover -s tests -q
python3 docker/v030/apply_nccl_twin_overlay.py --check
V030=vllm/vllm-openai:v0.30.0-aarch64@sha256:4864d46625cbc3307623e29ac742030655e27249feba7b97ec925ce4cc4dfb56

# 1. Engagement boot (B arm), with NCCL INFO lines for the audit.
IMAGE=$V030 OVERLAYS="docker/v030/nccl_twin_cuda_communicator.py <the other OVERLAYS_V030 files of the A arm>" \
  EXTRA_ENV="VLLM_QWEN38_NCCL_TWIN=1 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=ENV" VALIDATE_ONLY=1 ./run.sh
IMAGE=$V030 OVERLAYS="..." EXTRA_ENV="VLLM_QWEN38_NCCL_TWIN=1 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=ENV" ./run.sh
python3 bench_decode.py            # traffic, so eager calls reach the twin
docker logs qwen38-flash-next-nvfp4 > /tmp/l2a-head.log 2>&1
ssh spark2 docker logs qwen38-flash-next-nvfp4 > /tmp/l2a-worker.log 2>&1
python3 docker/v030/nccl_twin_audit.py head=/tmp/l2a-head.log worker=/tmp/l2a-worker.log
#   First boot: compare "graphs=" with vLLM's capture count, which is
#   len(cudagraph_capture_sizes) x graph kinds (target + drafter) from the engine config line.
#   Also check the per-graph histogram against the model's collectives per forward.
#   Then pin both with --graphs N [--expect-captured T].
free -h                            # twin NCCL buffers: plan budget -0.1..-0.5 GiB, spark1 invariant

# 2. Timing: only after the audit passes. Serve ABAB against the same image and overlays without the lever
#    (and with NCCL_DEBUG=WARN on both arms), bench_decode.py, 2+2 boots; in-serve NCCL kernel us as the
#    primary evidence (plan L2 experiment). Pass: 1 h soak with mixed traffic, a 128k prefill and
#    co-prefill bursts, 0 hangs, 0 NCCL WARN, T1-G, and exact count/tool smokes.
```

The plan has no device-level corruption check here yet. The sibling had one: `tools/nccl_twin_check.py
--device-check`, which ran pipelined steps with an injected fault at step 37. It is not ported. Port it
before the soak if the audit leaves any doubt.

## Risks carried from the sibling

- The graph communicator runs no eager work after its init warm-up: vLLM's eager warm-up runs go to the
  twin. So NCCL makes its runtime connections for all-gather and larger all-reduce protocols on the graph
  communicator during capture. The sibling's two-rank twin test captured and replayed this way on 2.30.7;
  a capture-time NCCL error on the first boot points here.
- NCCL documents `graphUsageMode 0` as "no graphs". The lever depends on 2.30.7 using the mode only to
  select `mixing` (`strongstream.cc`). That is why the overlay only arms on 2.30.7.
- Sibling result: not promoted. It measured −1.3 to −1.8 ms/step, but acceptance moved and tok/s did not
  clear noise. Its projection was −2.4 to −7.1 ms. Book 0.89 ms and treat the ABAB as the verdict.
