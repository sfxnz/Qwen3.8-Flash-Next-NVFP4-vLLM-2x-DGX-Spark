#!/usr/bin/env python3
"""FP8 dense GEMM arms at the P4-1 shapes, one GPU, serve down (plan 0-9, S1.1, D8).

Arms per (N, K, M), all through the image's own vLLM ops (skipped with the
reason when an op is missing or rejects sm_121):
  bf16             F.linear reference
  w8a8_cutlass     per-token dynamic FP8 quant + CUTLASS scaled_mm, per-channel
                   weight scale (Fp8PtpcOnlineLinearMethod's kernel pair)
  w8a8_gemm_only   the scaled_mm alone (activations pre-quantised)
  w8a16_marlin_ch  Marlin FP8 weight-only, per-channel scale
  w8a16_marlin_b128  Marlin FP8 weight-only, 128x128 block scale
                   (Fp8PerBlockOnlineLinearMethod + --linear-backend marlin)
GB/s counts FP8 weight bytes (N*K) for the FP8 arms and N*K*2 for bf16.

  python3 -S fp8_sweep.py run --json OUT [--ms 1,4,8,32]
  python3 fp8_sweep.py plan
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

# P4-1 allow-list, tensors split (C7): GDN in_proj_qkvz / out_proj, QSA qkv / o,
# PLE kv_proj, MTP attention qkv (same per-rank shape as the target QSA qkv).
SHAPES = {
    "gdn_in_proj_qkvz": (8192, 2560),
    "gdn_out_proj|qsa_o": (2560, 3072),
    "qsa_qkv": (6656, 2560),
    "qsa_qkv+index": (7296, 2560),
    "ple_kv_proj": (12800, 2560),
}
MS = (1, 2, 4, 8, 16, 32)
ARMS = ("bf16", "w8a8_cutlass", "w8a8_gemm_only", "w8a16_marlin_ch", "w8a16_marlin_b128")


def _marlin_layer(w_fp8, scale, block):
    """Minimal nn.Module carrying what prepare_fp8_layer_for_marlin reads."""
    import torch

    layer = torch.nn.Module()
    n, k = w_fp8.shape
    layer.output_size_per_partition = n
    layer.input_size_per_partition = k
    layer.orig_dtype = torch.bfloat16
    layer.weight = torch.nn.Parameter(w_fp8, requires_grad=False)
    if block:
        layer.weight_block_size = [128, 128]
        layer.weight_scale_inv = torch.nn.Parameter(scale, requires_grad=False)
    else:
        layer.weight_scale = torch.nn.Parameter(scale, requires_grad=False)
    return layer


def build_arms(n: int, k: int, m: int) -> dict:
    """{arm: (fn, weight_bytes) | error string}."""
    import torch
    import torch.nn.functional as F

    out = {}
    w = torch.randn(n, k, dtype=torch.bfloat16, device="cuda") * 0.02
    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    out["bf16"] = (lambda: F.linear(x, w), n * k * 2)
    fp8 = torch.float8_e4m3fn
    amax = w.abs().amax(dim=1, keepdim=True).float().clamp(min=1e-6)
    s_ch = (amax / 448.0).to(torch.float32)  # [N, 1]
    w8 = (w.float() / s_ch).clamp(-448, 448).to(fp8)  # [N, K]
    try:
        from vllm import _custom_ops as ops

        xq, xs = ops.scaled_fp8_quant(x, use_per_token_if_dynamic=True)
        b = w8.t()  # [K, N] column-major, what cutlass_scaled_mm expects

        def w8a8():
            q, s = ops.scaled_fp8_quant(x, use_per_token_if_dynamic=True)
            return ops.cutlass_scaled_mm(q, b, s, s_ch, torch.bfloat16)

        w8a8()
        out["w8a8_cutlass"] = (w8a8, n * k)
        out["w8a8_gemm_only"] = (lambda: ops.cutlass_scaled_mm(xq, b, xs, s_ch, torch.bfloat16), n * k)
    except Exception as e:  # noqa: BLE001
        out["w8a8_cutlass"] = out["w8a8_gemm_only"] = f"{type(e).__name__}: {e}"[:300]
    for arm, block in (("w8a16_marlin_ch", False), ("w8a16_marlin_b128", True)):
        try:
            from vllm.model_executor.layers.quantization.utils import marlin_utils_fp8 as mu

            if block:
                nb, kb = -(-n // 128), -(-k // 128)
                scale = torch.rand(nb, kb, device="cuda", dtype=torch.float32) * 1e-3 + 1e-4
            else:
                scale = s_ch.view(n).clone()
            layer = _marlin_layer(w8.clone(), scale, block)
            mu.prepare_fp8_layer_for_marlin(layer, size_k_first=False)
            ws = layer.weight_scale_inv if block else layer.weight_scale

            def marlin(layer=layer, ws=ws):
                return mu.apply_fp8_marlin_linear(x, layer.weight, ws, layer.workspace, n, k, None)

            marlin()
            out[arm] = (marlin, n * k)
        except Exception as e:  # noqa: BLE001
            out[arm] = f"{type(e).__name__}: {e}"[:300]
    return out


def run(args) -> dict:
    import torch

    torch.cuda.set_device(0)
    timer = common.Timer(calls=args.calls, replays=args.replays)
    ms = [int(x) for x in args.ms.split(",")] if args.ms else list(MS)
    rows = []
    for name, (n, k) in SHAPES.items():
        for m in ms:
            arms = build_arms(n, k, m)
            for arm in ARMS:
                a = arms[arm]
                r = {"name": name, "n": n, "k": k, "m": m, "arm": arm}
                if isinstance(a, str):
                    r["skipped"] = a
                    print(f"{name:<20} M={m:<2} {arm:<18} SKIP {a[:100]}", flush=True)
                else:
                    fn, nbytes = a
                    r["weight_bytes"] = nbytes
                    for mode in ("hot", "flush"):
                        c = timer.cost(fn, mode)
                        c["gbps"] = common.gbps(nbytes, c["median_us"])
                        r[mode] = c
                    if m in (1, 4, 32) and not args.no_kernels:
                        r["kernels"] = common.profile_kernels(fn, iters=5, pre=timer.flush)
                    print(f"{name:<20} M={m:<2} {arm:<18} flush {r['flush']['median_us']:8.2f} us "
                          f"{r['flush']['gbps']} GB/s | hot {r['hot']['gbps']} GB/s", flush=True)
                rows.append(r)
            del arms
            common.cleanup()
    return {"bench": "fp8", "meta": common.meta(), "rows": rows}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--json")
    p.add_argument("--ms")
    p.add_argument("--calls", type=int, default=20)
    p.add_argument("--replays", type=int, default=40)
    p.add_argument("--no-kernels", action="store_true")
    sub.add_parser("plan")
    args = ap.parse_args(argv)
    if args.cmd == "plan":
        for name, (n, k) in SHAPES.items():
            print(f"fp8 {name} N={n} K={k} M={','.join(map(str, MS))} arms={','.join(ARMS)}")
        return 0
    common.write_json(args.json, run(args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
