# spark1 went dark, 2026-09-29 13:50–14:05 BST

## What happened
- Session 1 step B0 (pin image, `UTIL=0.80`) was serving. spark1 idled at about 3.3 GiB free with 8 GiB of swap already used (`vmstat-spark1-b0.txt`, 13:47).
- At 13:50 the operator opened a torch-profiler window on the live serve. The profiler's host-side trace buffers grew the vLLM worker (`VLLM::Worker_TP`, about 14 GiB of RSS).
- By 13:54 swap was full (16 GiB). The host thrashed for about 11 minutes: `sy` 98–100%, run queue 49–96, and page-cache reads of 0.5–1.9 GB/s. sshd (`oom_score_adj -1000`) was never killed, but it could not page in to answer, so ssh looked dead.
- The kernel OOM killer fired 8 times between 13:54 and 14:05 (`kernel-oom.txt`). It killed the smallest high-`oom_score_adj` processes first: LAIL `next-server` (adj 200, 6 times, restarted each time by `Restart=always`; `NRestarts=7`) and `hermes`. The real consumer, the vLLM worker (adj 0), kept running until the operator stopped the serve at 14:05.
- There was no reboot (`uptime` 9 days). No OOM kills happened on spark2; its NVRM `NV_ERR_NO_MEMORY` lines at 13:30 are normal during model-load memory profiling.

## Root causes
1. The host headroom was too thin at `UTIL=0.80` on spark1, which also runs LAIL, Hermes and agent sessions (about 13 GiB).
2. An in-serve torch-profiler window adds several GiB of host RSS to the worker.
3. The kernel OOM policy picked the wrong victim, and it acts only after minutes of thrash.

## Fixes (run.sh / recipe.yaml)
- `UTIL` 0.80 → 0.76 (plan P2-2). This gives about 4.9 GiB per rank of extra host headroom.
- `docker run --oom-score-adj 1000` (`OOM_SCORE_ADJ`): if the kernel must kill something, it kills the serve and not sshd, LAIL or Hermes.
- `MEMGUARD=1`: a per-node watchdog started with each rank. It runs `docker kill` on the serve when MemAvailable < 2 GiB **and** SwapFree < 2 GiB for 3 samples in a row (6 s), before thrash sets in. It logs to the journal (`-t qwen38-memguard`) and to `.run-state/memguard.log`, and exits when the container is gone. Tests: `tests/test_recipe_ops.py::test_memguard_*`; a live trip was verified on a CPU-only test container.
- Operator rule: no torch-profiler windows on the served default. Profile only on a diagnostic boot with `UTIL ≤ 0.70` and a bounded window (few steps, no stacks), with `free -h` watched.

## Second trip, session 3 (2026-09-29 19:40–19:43 BST): memguard worked, root cause was a core dump

- The v0.30 + determinism default was serving (UTIL 0.76, memguard on). T3 64k (`quality/t3_needles.py --lengths 65536`) ran after a memcheck with spark1 at 12.2 GiB available and 6.9 GiB swap used.
- 19:40:04: a CUDA device-side assert (`ScatterGatherKernel.cu:203 ... index out of bounds`) fired on both ranks during the 64k prefill. The cause was the S1.5 QSA tie-repair overlay (`docker/v030/qsa_indexer.py`): it took its threshold from persistent_topk's picks, so a pad or a non-top pick let more than `block_topk` entries past the threshold, and `scatter_` wrote past its `block_topk + 1` columns. Fixed in the generator (exact `torch.topk` threshold plus a slot clamp), with a regression test (`tests/test_qsa_topk_order_overlay.py::test_tie_repair_ignores_bad_topk_picks`; the old overlay raises `index 513 is out of bounds for dimension 1 with size 513`).
- The worker aborted (SIGABRT). The host `core_pattern` pipes to apport, and apport treats the container's core limit as unlimited. It read the whole ~11 GiB core into RAM on each node (`/var/log/apport.log`: 19:40:04 called, 19:42:08 report written, 19:42:54 11,441,852,416-byte core written). spark1 went from 12 GiB available to MemAvailable 1.0 GB and SwapFree 72 MB.
- 19:42:52: memguard ran `docker kill` on the serve (`a-default/memguard.log`, journal `qwen38-memguard`). sshd, LAIL and Hermes kept running, there was no thrash window, and the host recovered at once (116 GiB available).
- Fix: `run.sh` passes `docker run --ulimit core=1`. The kernel skips a piped core dump when RLIMIT_CORE is exactly 1 (`coredump: ... RLIMIT_CORE is set to 1, aborting core`, verified with an `os.abort()` container). A limit of 0 would not help, because apport reads the core from the pipe regardless of the limit.
- Left behind outside the recipe tree (not removed, not authorized): `/var/crash/_usr_bin_python3.12.0.crash` (~1.5 GB) and the ~11 GiB core under `/var/lib/apport/coredump/`, on both nodes.
