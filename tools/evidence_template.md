# Evidence template: one Spark session

Copy the layout below for every iteration directory `evidence/<iter>/`.
`tools/session_gate.sh evidence/<iter>` produces most of it. Hand-written
files are marked (hand).

## Layout

| Path | Producer | Content |
|---|---|---|
| `hypotheses.md` (hand) | before boot | Knob, class (R12 knob→gate map), primary metric, guard metrics, MDE and n, keep/revert rule |
| `commands.sh` (hand) | before boot | Exact argv: `run.sh` env, then `tools/session_gate.sh` |
| `harness.sha256` | gate | sha256 of `bench_decode.py`, `ruler_steps.py`, `bench_probe.py`, smokes, tools. `bench_decode.py` must match the frozen sha |
| `git-head.txt` | gate | Commit the recipe ran from |
| `health.code`, `models.json` | gate | `/health` 200 and `/v1/models` lists the served name |
| `free-{before,after}.txt`, `free-{before,after}-spark2.txt` | gate | `free -h` on both nodes. Never `nvidia-smi` VRAM |
| `metrics-{before,after}.txt` | gate | Raw `/metrics` snapshot |
| `receipts/<node>-boot.txt` | telemetry receipt | Driver, kernel, BIOS, fwupdmgr (EC, SoC, PD, CX-7), governors, image digest, mounts, redacted env, cache dirs with tree sha256 |
| `receipts/<node>-docker.log` | telemetry receipt | Full `docker logs` of each rank at session start |
| `telemetry/<node>.csv` | telemetry | 1 Hz long-format CSV `ts_unix,ts_iso,node,key,value`: GPU clocks/power/temp/event reasons, cpufreq, per-thread CPU and placement of vLLM threads, PSI, vmstat deltas, meminfo, thermal zones, hwmon, RoCE counter deltas, head `/metrics`, journal lines |
| `smoke-*.{out,err,exit}` | gate | thinking-off, tools, vision, count smokes |
| `probe.{out,json}` | `bench_probe.py` | T0 gate. Per stream: finish_reason, completion, chunks, acceptance, TTFT, text sha, head/tail, flags |
| `ruler.{out,json}` | `ruler_steps.py` | Warm-up, sentinels, cells. Per stream: completion, chunks, decode_s, ms/step, TTFT. Per wave: acceptance and Δnum_drafts cross-check. SUMMARY or `SUMMARY REFUSED` |
| `bench-frozen.{out,err,exit}` | `bench_decode.py` | Frozen ruler, once per session (R2) |
| `docker-tail-{head,spark2}.log` | gate | Last 400 log lines per rank after the bench |
| `engine-needles.txt` | gate | grep of engine log for version, quantisation, KV size, graph capture, errors |
| `gate.txt`, `gate.log` | gate | Step exit codes and `GATE=PASS|FAIL` |
| `verdict.txt` (hand) | after | Keep/revert with the numbers below |
| `decision.tsv` row (hand) | after | One row in `evidence/decision.tsv` |

## Verdict must quote

- `RULER_SET`: the sha256 lines of the rulers used.
- T0: `probe.out` TOTALS line. Gate flags must be 0. First-32 match rate is report-only until S1.5.
- Sentinel line from `ruler.out`: runs, excursions, rate, boot minimum, clean median ms/step, void.
- Per cell from `ruler.json` `cell_stats`: median and min ms/step, every per-wave ms/step, per-wave acceptance, median and min aggregate.
- Any wave with `foreign_traffic` is excluded and listed.
- Frozen `bench_decode.py` SUMMARY for continuity with older rows.
- Telemetry: any `clocks_event_reasons.*` Active, any `vmstat.pswpin.delta > 0`, PSI memory > 0, or thermal peaks during an excursion window.
- `free -h` on both nodes before and after.

## Refusals

- No SUMMARY while any T0 flag fires (`ruler_steps.py` exits 2).
- A boot is void when a sentinel block fails three times (≥35 s between tries).
- Sentinel excursion rate above 1 in 20 is reported and voids the boot for keep decisions.
- Never publish a number from a session whose `gate.txt` says `GATE=FAIL`.
