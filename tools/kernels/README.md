# tools/kernels

## L6a: drafter Triton MoE config (`tune_draft_moe.sh`)

The MTP drafter's experts are FP8 block-scaled and run on Triton `fused_moe`. On v0.30.0 both ranks
log at boot (evidence/u1-v030):

```
WARNING [fused_moe.py:1167] Using default MoE config. Performance might be sub-optimal! Config file not found at
.../fused_moe/configs/E=512,N=320,device_name=NVIDIA_GB10,dtype=fp8_w8a8,block_shape=[64,64].json
```

| Field | Value | Source |
|---|---|---|
| E | 512 | `num_experts` |
| N | 320 | `moe_intermediate_size` 640 / TP 2 (w2 last dim) |
| dtype | `fp8_w8a8` | MTP experts `FP8_BLOCK_SCALES` |
| block | `[64,64]` | checkpoint group 128, refined to 64x64 (320 % 128 != 0) |
| device | `NVIDIA_GB10` | `torch.cuda.get_device_name()` = `NVIDIA GB10`, `[\s/]+` -> `_` |

The name comes from `get_config_file_name()` in `fused_moe.py`. `get_moe_configs()` tries
`$VLLM_TUNED_CONFIG_FOLDER/<name>` first, then the bundled `configs/<name>`. It then picks the entry whose
batch key is closest to M. The image has no GB10 file for this shape, so every draft step uses
`get_default_config()`.

### Files

- `docker/v030/moe_configs/<name>`: the config. Right now it is the **seed**, a byte copy of v0.30.0's
  `E=512,N=320,device_name=NVIDIA_B200,...,block_shape=[64,64].json`. That is the only bundled
  file for this exact shape. It keeps all 14 keys (1 to 8192), not only 1 to 64. Lookup picks the
  nearest key, so a 1..64 file would give 8192-token prefill chunks the M=64 tile. The seed is **not
  measured on GB10**. `tune` replaces every entry. The check's shared-memory estimate is at most
  73.7 KB per entry, under GB10's 101376 B opt-in limit.
- `tests/test_moe_config.py`: the host tests run the checker. In the image, CPU only, the tests also
  run v0.30's real `get_config_file_name`, `get_moe_configs` and `try_get_optimal_moe_config`, with
  the device name mocked to `NVIDIA GB10`. They also run the tuner wrapper's plan mode.

### Subcommands

| Command | GPU | What |
|---|---|---|
| `seed` | no | Re-extract the seed from the image. |
| `check [DIR]` | no | Validate the config. Host python3, no torch. This is the load-time self-test. |
| `tune` | 1 | Run the stock `benchmark_moe.py --tune` from the image, in-process. `OUT` defaults to `evidence/l6a-tune/<stamp>`. |
| `bench` | 1 | Run `benchmark_moe.py` with the stock default config, then with the `CFG` config, ABAB, per batch size. |

The image has no `ray`, so the wrapper uses a stub that calls `BenchmarkWorker` directly on one GPU.
The search space is stock with two changes. `BLOCK_SIZE_K` is fixed at 64, because the kernel clamps
K to `min(block_shape)`. `num_warps=2` twins are added. That gives 960 configs per batch size. The
tuner's model params come from a synthetic `Qwen3MoeForCausalLM` config (`synth`), because
`Qwen4ExpForConditionalGeneration` is not in `get_model_params`. Before tuning, `tune` runs a plan
step on the real GPU and refuses if the target name is not the one above.

### Mount (default OFF)

Set `VLLM_TUNED_CONFIG_FOLDER`. It exists in v0.30 `envs.py`, and the compile-cache hash ignores it.
No overlay is needed. The folder holds only this file. The other folder users are Mamba SSU configs
and LoRA, which look up different names and fall through to stock. The NVFP4 target MoE does not
use Triton configs. The flag is proposed as `DRAFT_MOE_CONFIG=0|1`, v0.30 digest only (see the report
or `run.sh` once wired):

1. Run `tools/kernels/tune_draft_moe.sh check`. On failure, warn and leave the env unset. That
   falls closed to the stock default config.
2. Mount `docker/v030/moe_configs` read-only at `/opt/qwen38-moe-configs` on both ranks. The worker
   copy goes to `/tmp/qwen38-moe-configs/`, like the overlays.
3. Pass `-e VLLM_TUNED_CONFIG_FOLDER=/opt/qwen38-moe-configs`.
4. Engagement: each rank logs `Using configuration from /opt/qwen38-moe-configs/E=512,N=320,...` and
   no `Using default MoE config` line.

### GPU runbook (one Spark, no serve up)

```bash
tools/kernels/tune_draft_moe.sh tune                    # ~960 cfgs x 19 batch sizes
CFG=evidence/l6a-tune/<stamp>/install tools/kernels/tune_draft_moe.sh bench
cp 'evidence/l6a-tune/<stamp>/install/E=512,N=320,device_name=NVIDIA_GB10,dtype=fp8_w8a8,block_shape=[64,64].json' docker/v030/moe_configs/
tools/kernels/tune_draft_moe.sh check && python3 -m unittest discover -s tests -q
```

Gate (plan L6a): the draft-graph class time in serve must drop by at least 0.1 ms. Then run
`bench_decode.py` 1+1. If it does not beat noise, revert with the flag off and record it in `evidence/`.
