#!/usr/bin/env python3
"""K5 fused HC vs stock HC chain, one GPU, serve down (plan section 4 K5, L4b).

Stock: the image's own Triton glue (vllm.models.qwen4_exp.nvidia.ops.hc) plus
F.linear for the projections - the 5 kernels a serve GatedResidual runs.
Fused: docker/v030/hc_fused.py hcf_forward - H1 combine_stats, H2
down_inject_silu, H3 up_gatemix. Random bf16 weights at the real shapes:
norm [10240], down+inject [336, 10240] (use_combine) or [320, 10240]
(final mixer), up [10240, 320].

Per M in {1,2,4,8,16,32}:
  1. correctness: the load-time self-test (hcf_self_test: M in
     HCF_SELF_TEST_MS x 3 input distributions, residual bitwise, class A vs
     fp64, CUDA-graph replay at M=4 and 32)
     plus this M: residual bitwise, block_input / injection pooled error vs
     fp64 <= 2x stock. A failing M is reported and not timed.
  2. timing with tools/micro/common.Timer (CUDA-graph replay, slot marks):
     hot (one module, L2-resident), stream (8 distinct modules cycled: the
     serve case, DRAM-bound), cold (one module after a 64 MB flush).
  3. per-kernel split from one profiled pass after a flush.
Gate (plan L4): fused stream <= 0.6x stock stream at M=4 and M=32.

  python3 -S tools/kernels/hc_bench.py plan
  python3 tools/kernels/hc_bench.py run --json OUT.json [--variant final] \
      [--ms 1,2,4,8,16,32] [--cfg split_k=8,block_n=16,block_k=128,block_j=16]
Run inside vllm/vllm-openai:v0.30.0-aarch64 with --gpus all, serve DOWN.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "micro"))
import common  # noqa: E402

HC = common.HC_COUNT
H = common.HIDDEN
R = common.HC_LORA
D = HC * H  # 10240
EPS = 1e-6
MS = (1, 2, 4, 8, 16, 32)
STREAM_MODULES = 8
GATE_MS = (4, 32)
GATE_RATIO = 0.6


def rows(variant: str) -> int:
    return R + HC + (-(R + HC)) % 16 if variant == "main" else R  # 336 | 320


def module_bytes(variant: str) -> int:
    # Stock bytes (336 rows incl. pad), as the S1.1 baseline counts them; the
    # fused kernel skips the 12 pad rows, so its x_floor is slightly pessimistic.
    return (rows(variant) * D + D * R) * common.BF16


def load_fused():
    path = ROOT / "docker" / "v030" / "hc_fused.py"
    spec = importlib.util.spec_from_file_location("hc_fused", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_cfg(text: str | None) -> dict:
    out = {}
    for kv in filter(None, (text or "").split(",")):
        k, v = kv.split("=")
        out[k.strip()] = int(v)
    return out


def build(variant: str, m: int, n_modules: int, hcf, cfg: dict):
    import torch
    import torch.nn.functional as F
    from vllm.models.qwen4_exp.nvidia.ops import hc

    dev = "cuda"
    bf = torch.bfloat16
    use_combine = variant == "main"
    nrows = rows(variant)
    g = torch.Generator(device="cpu").manual_seed(m)

    def rn(*s):
        return torch.randn(*s, generator=g).to(dev)

    residual = rn(m, D).to(bf)
    block_out = rn(m, H).to(bf)
    inj = rn(m, HC).to(bf)
    mods = []
    for _ in range(n_modules):
        s = types.SimpleNamespace()
        s.config = types.SimpleNamespace(params_dtype=bf, rms_norm_eps=EPS)
        s.hc_count, s.hidden_size, s.lora_rank, s.use_combine = HC, H, R, use_combine
        s.pad_size = nrows - R - HC if use_combine else 0
        s.hc_norm = types.SimpleNamespace(weight=(rn(D) * 0.3).to(bf))
        wd = types.SimpleNamespace(weight=(rn(nrows, D) / D**0.5 * 4).to(bf))
        if use_combine:
            s.input_mix_weight_down_block_inject = wd
        else:
            s.input_mix_weight_down = wd
        s.input_mix_weight_up = types.SimpleNamespace(weight=(rn(D, R) / R**0.5 * 3).to(bf))
        mods.append(s)
    split = [R, HC, nrows - R - HC] if use_combine else None

    def stock(s, hs, bo, ij):
        new, xn = hc.hc_combine_norm(hs, bo, ij, s.hc_norm.weight, EPS, HC)
        if use_combine:
            lora, inj_out, _ = F.linear(xn, s.input_mix_weight_down_block_inject.weight).split(split, dim=-1)
        else:
            lora, inj_out = F.linear(xn, s.input_mix_weight_down.weight), None
        lora = hc.hc_silu(lora, HC)
        gate = F.linear(lora, s.input_mix_weight_up.weight)
        return new, hc.hc_gate_mix(xn, gate, HC), inj_out

    def fused(s, hs, bo, ij):
        return hcf.hcf_forward(
            hs, bo, ij, s.hc_norm.weight,
            (s.input_mix_weight_down_block_inject if use_combine else s.input_mix_weight_down).weight,
            s.input_mix_weight_up.weight, EPS, HC, R, use_combine, cfg,
        )

    return mods, (residual, block_out, inj), stock, fused


def check(hcf, mods, inputs, stock, fused, self_test: bool) -> dict:
    import torch

    s = mods[0]
    out = {}
    if self_test:
        ok, detail = hcf.hcf_self_test(s, True, True, "cuda", stock_fn=lambda a, b, c: stock(s, a, b, c))
        out["self_test"] = {"ok": ok, "detail": detail}
        if not ok:
            return {**out, "ok": False}
    hs, bo, ij = inputs
    w = hcf._hcf_weights(s)
    st, fu = stock(s, hs, bo, ij), fused(s, hs, bo, ij)
    torch.cuda.synchronize()
    r_bi, r_inj = hcf._hcf_ref64(s, w, hs, bo, ij)
    out["residual_bitwise"] = bool(torch.equal(st[0], fu[0]))
    out["block_input_err"] = {"stock": hcf._hcf_rel(st[1], r_bi), "fused": hcf._hcf_rel(fu[1], r_bi)}
    out["block_input_gross"] = hcf._hcf_elem_bad(st[1], fu[1], r_bi)
    ok = out["residual_bitwise"] and out["block_input_gross"] == 0
    ok = ok and out["block_input_err"]["fused"] <= 2 * out["block_input_err"]["stock"] + 2**-12
    if s.use_combine:
        out["injection_err"] = {"stock": hcf._hcf_rel(st[2], r_inj), "fused": hcf._hcf_rel(fu[2], r_inj)}
        out["injection_gross"] = hcf._hcf_elem_bad(st[2], fu[2], r_inj)
        ok = ok and out["injection_gross"] == 0
    out["ok"] = bool(ok)
    return out


def run(args) -> dict:
    import torch

    torch.cuda.set_device(0)
    hcf = load_fused()
    cfg = {**hcf.HCF_CFG, **parse_cfg(args.cfg)}
    timer = common.Timer(calls=args.calls, replays=args.replays)
    nbytes = module_bytes(args.variant)
    floor = common.floor_us(nbytes)
    ms = tuple(int(x) for x in args.ms.split(","))
    rows_out = []
    for i, m in enumerate(ms):
        try:
            mods, inputs, stock, fused = build(args.variant, m, STREAM_MODULES, hcf, cfg)
        except Exception as e:  # noqa: BLE001
            rows_out.append({"m": m, "skipped": f"{type(e).__name__}: {e}"[:300]})
            print(f"HC-K5 M={m} SKIP {rows_out[-1]['skipped']}", flush=True)
            continue
        r = {"m": m, "module_bytes": nbytes, "floor_us": round(floor, 2)}
        r["check"] = check(hcf, mods, inputs, stock, fused, self_test=(i == 0))
        if not r["check"]["ok"]:
            rows_out.append(r)
            print(f"HC-K5 M={m} CHECK FAILED {r['check']}", flush=True)
            continue
        hs, bo, ij = inputs
        for name, fn in (("stock", stock), ("fused", fused)):
            fns = [lambda s=s, fn=fn: fn(s, hs, bo, ij) for s in mods]
            t = {"hot": timer.cost(fns[0], "hot"), "stream": timer.cost(fns[0], "hot", fns=fns),
                 "cold": timer.cost(fns[0], "flush")}
            for mode in t:
                t[mode]["gbps"] = common.gbps(nbytes, t[mode]["median_us"])
                t[mode]["x_floor"] = round(t[mode]["median_us"] / floor, 3)
            if not args.no_kernels:
                t["kernels"] = common.profile_kernels(fns[0], iters=5, pre=timer.flush)
            r[name] = t
        for mode in ("hot", "stream", "cold"):
            r[f"ratio_{mode}"] = round(r["fused"][mode]["median_us"] / r["stock"][mode]["median_us"], 3)
        rows_out.append(r)
        print(f"HC-K5 M={m:<2} stream stock {r['stock']['stream']['median_us']:7.2f} fused "
              f"{r['fused']['stream']['median_us']:7.2f} us ({r['ratio_stream']}x, fused "
              f"{r['fused']['stream']['x_floor']}x floor) | hot {r['ratio_hot']}x | cold {r['ratio_cold']}x", flush=True)
        del mods
        common.cleanup()
    gate = {}
    for row in rows_out:
        if row.get("m") in GATE_MS and "ratio_stream" in row:
            gate[str(row["m"])] = row["ratio_stream"] <= GATE_RATIO
    return {"bench": "hc_k5", "variant": args.variant, "cfg": cfg, "meta": common.meta(),
            "module_bytes": nbytes, "gate_ratio": GATE_RATIO, "gate_stream": gate, "rows": rows_out}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--json")
    p.add_argument("--variant", choices=("main", "final"), default="main")
    p.add_argument("--ms", default=",".join(map(str, MS)))
    p.add_argument("--cfg", help="override HCF_CFG, e.g. split_k=4,block_j=32")
    p.add_argument("--calls", type=int, default=24)
    p.add_argument("--replays", type=int, default=40)
    p.add_argument("--no-kernels", action="store_true")
    sub.add_parser("plan")
    args = ap.parse_args(argv)
    if args.cmd == "plan":
        for v in ("main", "final"):
            print(f"hc_k5 {v}: M={','.join(map(str, MS))} down [{rows(v)},{D}] up [{D},{R}] "
                  f"bytes {module_bytes(v)} floor {common.floor_us(module_bytes(v)):.1f} us; "
                  f"modes hot,stream({STREAM_MODULES}),cold; gate stream <= {GATE_RATIO}x stock at M={GATE_MS}")
        return 0
    common.write_json(args.json, run(args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
