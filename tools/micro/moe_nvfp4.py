#!/usr/bin/env python3
"""NVFP4 routed-MoE harness, one GPU, serve down (plan 0-9 C23, S1.1, F10, D2).

FlashInfer cutlass_fused_moe (fused_moe_120, what auto picks on sm_121) with
random NVFP4 weights, called the way vLLM's FlashInferExperts calls it:
  tp2   E=512 local experts, N=320 intermediate per rank, K=2560, top-10
  ep    E=256 local of 512 global (ep_size=2, ep_rank=0), N=640, top-10
Inputs:
  fp4_in   vLLM's separate NVFP4 input quant (flashinfer.fp4_quantize) + MoE: the serve chain
  bf16_in  BF16 activations straight into FlashInfer (L6b), when the API accepts it
Routing: uniform (distinct top-10 per token) and shared (every token picks the
same 10 experts, the c=1 identical-prefix floor). Floor = distinct local
experts x bytes per expert at MICRO_FLOOR_GBPS; bytes per expert at N=320 are
1,382,400 (fc1 819,200 + sf 102,400, fc2 409,600 + sf 51,200). The per-kernel
split (profiled after a flush) gives FC1/FC2 against their own floors.
Best effort: any API mismatch is recorded as "skipped" and the run exits 0.

  python3 -S moe_nvfp4.py run --json OUT
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

K = common.HIDDEN
TOPK = 10
GLOBAL_E = 512
CONFIGS = {
    "tp2": {"e_local": 512, "n": 320, "ep_size": 1},
    "ep": {"e_local": 256, "n": 640, "ep_size": 2},
}
MS = (1, 4, 8, 16, 32)
ROUTINGS = ("uniform", "shared")


def expert_bytes(n: int, k: int = K) -> dict:
    """NVFP4 bytes per expert: packed e2m1 (0.5 B) plus one FP8 scale per 16."""
    fc1 = 2 * n * k // 2 + 2 * n * k // 16
    fc2 = k * n // 2 + k * n // 16
    return {"fc1": fc1, "fc2": fc2, "total": fc1 + fc2}


def distinct_local(ids: list[list[int]], e_local: int, ep_rank: int = 0) -> int:
    lo = ep_rank * e_local
    return len({e for row in ids for e in row if lo <= e < lo + e_local})


def route(m: int, routing: str, seed: int) -> list[list[int]]:
    import random

    rng = random.Random(seed)
    if routing == "shared":
        row = rng.sample(range(GLOBAL_E), TOPK)
        return [list(row) for _ in range(m)]
    return [rng.sample(range(GLOBAL_E), TOPK) for _ in range(m)]


def build(cfg: dict, m: int, routing: str, in_mode: str):
    import torch
    import flashinfer
    from flashinfer.fused_moe import cutlass_fused_moe

    dev = "cuda"
    e, n = cfg["e_local"], cfg["n"]
    fp8 = torch.float8_e4m3fn

    def sf(*shape):
        return (torch.rand(*shape, device=dev) * 1.5 + 0.5).to(fp8)

    w1 = torch.randint(0, 256, (e, 2 * n, K // 2), dtype=torch.uint8, device=dev)
    w2 = torch.randint(0, 256, (e, K, n // 2), dtype=torch.uint8, device=dev)
    w1_sf, w2_sf = sf(e, 2 * n, K // 16), sf(e, K, n // 16)
    a1_gs = torch.ones(1, dtype=torch.float32, device=dev)
    a2_gs = torch.ones(1, dtype=torch.float32, device=dev)
    g1 = torch.full((e,), 1e-3, dtype=torch.float32, device=dev)
    g2 = torch.full((e,), 1e-3, dtype=torch.float32, device=dev)
    # fused_moe_120 wants the activation global scales 0-d (or per local expert), not shape (1,)
    quant_scales = [a1_gs.reshape(()), w1_sf.view(torch.int32), g1, a2_gs.reshape(()), w2_sf.view(torch.int32), g2]
    ids_list = route(m, routing, seed=1000 + m)
    ids = torch.tensor(ids_list, dtype=torch.int32, device=dev)
    weights = torch.full((m, TOPK), 1.0 / TOPK, dtype=torch.float32, device=dev)
    x = torch.randn(m, K, dtype=torch.bfloat16, device=dev)
    out = torch.empty(m, K, dtype=torch.bfloat16, device=dev)
    common_kw = dict(token_selected_experts=ids, token_final_scales=weights,
                     fc1_expert_weights=w1.view(torch.long), fc2_expert_weights=w2.view(torch.long),
                     output_dtype=torch.bfloat16, quant_scales=quant_scales, output=out,
                     ep_size=cfg["ep_size"], ep_rank=0, tune_max_num_tokens=64)

    if in_mode == "fp4_in":
        def fn():
            xq, xsf = flashinfer.fp4_quantize(x, a1_gs, sf_vec_size=16, is_sf_swizzled_layout=True)
            return cutlass_fused_moe(input=xq, input_sf=xsf, **common_kw)
    else:
        def fn():
            return cutlass_fused_moe(input=x, input_sf=None, **common_kw)

    try:  # boot-time autotune as in vLLM; untuned tactics are recorded as such
        from flashinfer.autotuner import autotune

        with autotune(True):
            fn()
        tuned = True
    except Exception:  # noqa: BLE001
        fn()
        tuned = False
    torch.cuda.synchronize()  # random weights: timing only, no numerics claim
    return fn, ids_list, tuned


def classify(kernels: list[dict]) -> dict:
    """Sum kernel us; the first two GEMM-like kernels are FC1 then FC2."""
    gemms = [k for k in kernels if k.get("us") is not None and "gemm" in k["name"].lower()]
    total = sum(k["us"] for k in kernels if k.get("us") is not None)
    out = {"kernels_us": round(total, 2), "n_kernels": len(kernels)}
    if len(gemms) >= 2:
        out["fc1_us"], out["fc2_us"] = gemms[0]["us"], gemms[1]["us"]
    return out


def run(args) -> dict:
    import torch

    torch.cuda.set_device(0)
    timer = common.Timer(calls=args.calls, replays=args.replays)
    rows = []
    for cname, cfg in CONFIGS.items():
        eb = expert_bytes(cfg["n"])
        for routing in ROUTINGS:
            for m in MS:
                for in_mode in ("fp4_in", "bf16_in"):
                    r = {"config": cname, **cfg, "k": K, "topk": TOPK, "m": m, "routing": routing,
                         "input": in_mode, "expert_bytes": eb}
                    try:
                        fn, ids, tuned = build(cfg, m, routing, in_mode)
                    except Exception as e:  # noqa: BLE001
                        r["skipped"] = f"{type(e).__name__}: {e}"[:300]
                        rows.append(r)
                        print(f"moe {cname} {routing} M={m} {in_mode} SKIP {r['skipped'][:120]}", flush=True)
                        common.cleanup()
                        continue
                    d = distinct_local(ids, cfg["e_local"])
                    r.update({"autotuned": tuned, "distinct_local_experts": d,
                              "floor_us": round(common.floor_us(d * eb["total"]), 2),
                              "fc1_floor_us": round(common.floor_us(d * eb["fc1"]), 2),
                              "fc2_floor_us": round(common.floor_us(d * eb["fc2"]), 2)})
                    try:
                        r["flush"] = timer.cost(fn, "flush")
                        r["hot"] = timer.cost(fn, "hot")
                        r["x_floor"] = round(r["flush"]["median_us"] / r["floor_us"], 3) if d else None
                        if not args.no_kernels:
                            r["kernels"] = common.profile_kernels(fn, iters=5, pre=timer.flush)
                            r.update(classify(r["kernels"]))
                            if "fc2_us" in r and r["fc2_floor_us"]:
                                r["fc2_x_floor"] = round(r["fc2_us"] / r["fc2_floor_us"], 3)
                        print(f"moe {cname} {routing:<7} M={m:<2} {in_mode} experts {d:<3} flush "
                              f"{r['flush']['median_us']:8.2f} us floor {r['floor_us']:7.2f} -> {r['x_floor']}x "
                              f"fc2 {r.get('fc2_x_floor')}x", flush=True)
                    except Exception as e:  # noqa: BLE001 - e.g. graph capture refused
                        r["skipped"] = f"timing {type(e).__name__}: {e}"[:300]
                        print(f"moe {cname} {routing} M={m} {in_mode} SKIP {r['skipped'][:120]}", flush=True)
                    rows.append(r)
                    del fn
                    common.cleanup()
    return {"bench": "moe_nvfp4", "meta": common.meta(), "rows": rows}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--json")
    p.add_argument("--calls", type=int, default=12)
    p.add_argument("--replays", type=int, default=30)
    p.add_argument("--no-kernels", action="store_true")
    sub.add_parser("plan")
    args = ap.parse_args(argv)
    if args.cmd == "plan":
        for c, cfg in CONFIGS.items():
            print(f"moe {c} {cfg} K={K} top{TOPK} M={','.join(map(str, MS))} "
                  f"bytes/expert {expert_bytes(cfg['n'])['total']}")
        return 0
    try:
        res = run(args)
    except Exception as e:  # noqa: BLE001 - best effort by design (C23)
        res = {"bench": "moe_nvfp4", "meta": common.meta(), "rows": [], "skipped": f"{type(e).__name__}: {e}"[:300]}
    common.write_json(args.json, res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
