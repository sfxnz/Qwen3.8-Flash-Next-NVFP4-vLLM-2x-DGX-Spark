#!/usr/bin/env python3
"""K3 harness: GDN MTP decode with lazy state commit vs the stock v0.30 kernel.

Standalone, single GPU (or CPU under TRITON_INTERPRET=1 for the sim/ref parts).
Loads the kernels from docker/v030/gdn_lazy.py (the file the overlay embeds).

Subcommands (GPU, inside vllm/vllm-openai:v0.30.0-aarch64, ONE GPU, no serve up):
  selftest   the load-time self-test (bitwise vs torch.ops._C, graph replay)
  exact      stock vs K3 over N steps, c in {1,8}, k=3, randomized acceptance,
             real layer shapes (H=8, HV=24 per rank): outputs bitwise every step,
             committed state bitwise after a materialize, ulp histogram on miss
  sim        V2 'align' slot scheme end to end (pre-copy at block crossings,
             1600-token checkpoint copies, mixed steps with non-spec rows, slot
             reuse after finish): stock kernel arm vs K3 arm; outputs every step,
             every checkpoint block, every committed state bitwise
  bench      CUDA-graph timing, stock vs K3 lazy vs K3 eager, c=1 and c=8, 36
             layers' worth of distinct state slots (cold-ish L2), ABAB x reps
  ncu        one stock and one lazy launch for ncu (dram bytes per layer per req)

CPU:  python3 tools/kernels/k3_harness.py sim --device cpu --steps 12 (slow, small)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KERNELS = ROOT / "docker" / "v030" / "gdn_lazy.py"


def load_gdl():
    name = "qwen38_gdn_lazy"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, KERNELS)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------- torch reference
def ref_step(torch, mixed, a, b, A_log, dt_bias, si, cu, acc, state, gate, nw, out,
             H, scale, eps, sigmoid):
    """fp32 torch model of the stock fused_gdn_decode_post_conv_mtp semantics
    (no FTZ / MUFU emulation: tolerance comparisons only)."""
    D = 128
    HV = state.shape[1]
    G = HV // H
    n, W = si.shape
    kh = torch.arange(HV, device=state.device) // G
    for r in range(n):
        bos, eos = int(cu[r]), int(cu[r + 1])
        T = eos - bos
        if T <= 0:
            continue
        a_ = int(acc[r])
        src = int(si[r, a_ - 1]) if 1 <= a_ <= W else 0
        if src <= 0 or T > 8:
            out[bos:eos] = 0
            continue
        h = state[src].clone()
        for t in range(T):
            tok = bos + t
            row = mixed[tok].float()
            q = row[: H * D].view(H, D)[kh]
            k = row[H * D: 2 * H * D].view(H, D)[kh]
            v = row[2 * H * D:].view(HV, D)
            q = q * (torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6) * scale)
            k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
            x = a[tok].float() + dt_bias.float()
            sp = torch.where(x > 20, x, torch.log1p(torch.exp(x)))
            decay = torch.exp(-torch.exp(A_log.float()) * sp)
            beta = torch.sigmoid(b[tok].float())
            h = h * decay[:, None, None]
            hk = (h * k[:, None, :]).sum(-1)
            delta = (v - hk) * beta[:, None]
            h = h + k[:, None, :] * delta[:, :, None]
            o = (h * q[:, None, :]).sum(-1).to(torch.bfloat16).float()
            dst = int(si[r, t])
            if dst > 0:
                state[dst] = h
            rstd = torch.rsqrt((o * o).mean(-1, keepdim=True) + eps)
            g = gate[tok].float()
            ga = torch.sigmoid(g) if sigmoid else g * torch.sigmoid(g)
            out[tok] = (o * rstd * nw.float() * ga).to(torch.bfloat16)


def stock_cuda(**kw):
    from vllm import _custom_ops as ops

    ops.fused_gdn_decode_post_conv_mtp(**kw)


# ---------------------------------------------------------------- problem data
def make_layer(torch, device, H, G, gen):
    HV = H * G
    return {
        "H": H, "HV": HV,
        "A_log": (torch.randn(HV, generator=gen) * 0.8 + 0.5).to(device),
        "dt_bias": torch.randn(HV, generator=gen).to(device),
        "norm_w": (1 + 0.1 * torch.randn(128, generator=gen)).to(device),
    }


def make_tokens(torch, device, L, lens, gen):
    H, HV = L["H"], L["HV"]
    ntok = max(1, sum(lens))
    mixed = (torch.randn(ntok, 2 * H * 128 + HV * 128, generator=gen) * 0.6).to(torch.bfloat16)
    ba = (torch.randn(ntok, 2 * HV, generator=gen) * 2).to(torch.bfloat16)
    gate = (torch.randn(ntok, HV, 128, generator=gen) * 1.5).to(torch.bfloat16)
    mixed, ba, gate = mixed.to(device), ba.to(device), gate.to(device)
    b, a = ba[:, :HV], ba[:, HV:]  # strided views, like split_ba
    return mixed, a, b, gate


def bits(torch, x):
    return x.contiguous().view(torch.int16 if x.dtype == torch.bfloat16 else torch.int32)


def ulp_hist(torch, x, y):
    """max / histogram of |ulp| differences of two fp32 or bf16 tensors."""
    xi = bits(torch, x).to(torch.int64)
    yi = bits(torch, y).to(torch.int64)
    d = (xi - yi).abs()
    return {"max_ulp": int(d.max()) if d.numel() else 0,
            "n_diff": int((d > 0).sum()), "n": int(d.numel())}


# --------------------------------------------------------- V2 'align' simulator
class Arm:
    """One implementation arm: its own state tensor and header table."""

    def __init__(self, torch, gdl, state0, L, decode, lazy):
        self.t = torch
        self.gdl = gdl
        self.state = state0.clone()
        self.hdr = torch.zeros(state0.shape[0], dtype=torch.int32, device=state0.device)
        self.L = L
        self.decode = decode  # "stock_cuda" | "k3_eager" | "k3"
        self.lazy = lazy


class AlignSim:
    """Mirrors vLLM v0.30 V2 MambaHybridModelState in 'align' mode for the GDN
    temporal (SSM) state only (conv state untouched by K3):
      preprocess_mamba_align_fused_kernel + precopy (block crossing),
      spec window = block_table[s : s + W], s = (seq_len - 1) // BS,
      postprocess_mamba_fused_kernel (checkpoint copy, num_accepted reset).
    Mixed steps run the non-spec rows as T = 1 decodes reading/writing column 0
    (the chunk-prefill reader) and the spec rows through the eager kernel (the
    FLA reader, which reads column a-1 and writes every column)."""

    def __init__(self, torch, gdl, device, L, W, BS, nreq, nslots, seed,
                 arms=("stock", "k3"), break_fixup=False, break_boundary=False):
        # break_*: negative controls (tests prove the sim catches these bugs)
        self.break_fixup = break_fixup
        self.break_boundary = break_boundary
        self.t = torch
        self.gdl = gdl
        self.dev = device
        self.L = L
        self.W = W
        self.BS = BS
        self.rng = random.Random(seed)
        self.gen = torch.Generator().manual_seed(seed)
        self.free = list(range(nslots - 1, 0, -1))  # 0 = NULL block
        self.reqs = []
        self.next_id = 0
        self.nslots = nslots
        st0 = (torch.randn(nslots, L["HV"], 128, 128, generator=self.gen) * 0.05).to(device)
        self.arms = {}
        for name in arms:
            if name == "stock":
                dec = "stock_cuda" if torch.device(device).type == "cuda" else "k3_eager"
                self.arms[name] = Arm(torch, gdl, st0, L, dec, lazy=False)
            elif name == "k3":
                self.arms[name] = Arm(torch, gdl, st0, L, "k3", lazy=True)
        self.nreq_target = nreq
        self.log = {"steps": 0, "crossings": 0, "checkpoints": 0, "mixed": 0,
                    "reused": 0, "lazy_rows": 0, "spec_rows": 0, "finished": 0,
                    "materialized_rows": 0}
        self._ever = set()
        self.max_life = 40

    # -- request lifecycle
    def _alloc(self, n):
        out = []
        for _ in range(n):
            out.append(self.free.pop())
        return out

    def add_request(self):
        P = self.rng.randint(1, 3 * self.BS)
        life = self.rng.randint(8, self.max_life)
        nblocks = (P + (life + 1) * self.W) // self.BS + self.W + 2
        bt = self._alloc(nblocks)
        if any(b in self._ever for b in bt):
            self.log["reused"] += 1
        self._ever.update(bt)
        req = {"id": self.next_id, "bt": bt, "nc": 0, "acc": 1, "sidx": -1,
               "P": P, "life": life, "prefilling": True, "ckpt": set()}
        self.next_id += 1
        self.reqs.append(req)

    def finish(self, req):
        self.log["state_blocks_checked"] = (self.log.get("state_blocks_checked", 0)
                                            + self._check_req(req))
        self.reqs.remove(req)
        self.free.extend(reversed(req["bt"]))  # LIFO reuse: stale rings and headers
        self.log["finished"] += 1

    # -- helpers
    def _copy(self, src, dst):
        for arm in self.arms.values():
            arm.state[dst] = arm.state[src]

    def preprocess(self, req, T):
        """preprocess_mamba_align_fused_kernel + precopy for one request."""
        src_col = req["sidx"]
        src_off = max(req["acc"] - 1, 0)
        new = (req["nc"] + T + self.BS - 1) // self.BS - 1
        req["sidx"] = new
        if src_col >= 0 and src_col != new:
            self._copy(req["bt"][src_col + src_off], req["bt"][new])
            req["acc"] = 1
            self.log["crossings"] += 1

    def window(self, req, T):
        s = (req["nc"] + T - 1) // self.BS
        cols = req["bt"][s: s + self.W]
        return cols + [0] * (self.W - len(cols))

    def postprocess(self, req, acc):
        req["acc"] = max(acc, 1)
        new_nc = req["nc"] + req["acc"]
        running = new_nc - req["acc"] + 1
        aligned = (new_nc // self.BS) * self.BS
        if aligned >= running:
            bias = aligned - running
            dest = aligned // self.BS - 1
            src = req["sidx"]
            if src == dest:
                req["acc"] = 1
            if not (src == dest and bias == 0):
                self._copy(req["bt"][src + bias], req["bt"][dest])
            req["ckpt"].add(req["bt"][dest])
            self.log["checkpoints"] += 1
        req["nc"] = new_nc

    # -- one engine step
    def step(self):
        t = self.t
        while len(self.reqs) < self.nreq_target:
            self.add_request()
        mixed_step = self.rng.random() < 0.25
        rows = []  # (req, T, kind) kind: spec | nonspec | prefill
        for req in self.reqs:
            if req["prefilling"]:
                rows.append((req, req["P"], "prefill"))
            elif mixed_step and self.rng.random() < 0.3:
                rows.append((req, 1, "nonspec"))  # 0 drafts: prefill-class reader
            else:
                T = self.W if self.rng.random() < 0.7 else self.rng.randint(1, self.W)
                rows.append((req, T, "spec"))
        has_prefill = any(k != "spec" for _, _, k in rows)
        if has_prefill:
            self.log["mixed"] += 1
        for req, T, kind in rows:
            if kind == "prefill":
                req["sidx"] = -1
            self.preprocess(req, T)
        # prefill rows: every arm writes the same fresh state into column 0
        for req, T, kind in rows:
            if kind == "prefill":
                col0 = self.window(req, T)[0]
                init = (t.randn(self.L["HV"], 128, 128, generator=self.gen) * 0.05).to(self.dev)
                for arm in self.arms.values():
                    arm.state[col0] = init
        spec = [(r, T) for r, T, k in rows if k == "spec"]
        nons = [(r, T) for r, T, k in rows if k == "nonspec"]
        pref = [(r, T) for r, T, k in rows if k == "prefill"]
        lens = [T for _, T in spec] + [1 for _ in nons]
        mixed, a, b, gate = make_tokens(t, self.dev, self.L, lens, self.gen)
        si = t.tensor([self.window(r, T) for r, T in spec] + [self.window(r, 1) for r, _ in nons],
                      dtype=t.int32, device=self.dev).view(-1, self.W)
        cu = t.tensor([0] + list(_cumsum(lens)), dtype=t.int32, device=self.dev)
        acc = t.tensor([r["acc"] for r, _ in spec] + [1 for _ in nons], dtype=t.int32,
                       device=self.dev)
        seq = t.tensor([r["nc"] + T for r, T in spec] + [r["nc"] + 1 for r, _ in nons],
                       dtype=t.int32, device=self.dev)
        outs = {}
        for name, arm in self.arms.items():
            outs[name] = self._forward(arm, spec, nons, pref, mixed, a, b, gate, si, cu, acc,
                                       seq, has_prefill)
        base = next(iter(outs.values()))
        for name, o in outs.items():
            if not t.equal(bits(t, o), bits(t, base)):
                raise AssertionError(f"step {self.log['steps']}: outputs differ ({name}) "
                                     f"{ulp_hist(t, o, base)}")
        # sampling -> acceptance, then postprocess
        for req, T in spec:
            self.postprocess(req, self.rng.randint(1, T))
        for req, _ in nons:
            self.postprocess(req, 1)
        for req, T in pref:
            req["prefilling"] = False
            req["nc"] = T
            req["acc"] = 1
        for req in list(self.reqs):
            req["life"] -= 1
            if req["life"] <= 0 and not req["prefilling"]:
                self.finish(req)
        self.log["steps"] += 1

    def _forward(self, arm, spec, nons, pref, mixed, a, b, gate, si, cu, acc, seq, has_prefill):
        t, gdl, L = self.t, self.gdl, self.L
        out = t.zeros(mixed.shape[0], L["HV"], 128, dtype=t.bfloat16, device=self.dev)
        if arm.lazy and (has_prefill or nons) and not self.break_fixup:
            # gdl_fixup: materialize spec + non-spec running rows, drop prefill headers
            idx = [self.window(r, T) for r, T in spec] + [self.window(r, 1) for r, _ in nons] \
                + [self.window(r, T) for r, T in pref]
            flags = [1] * (len(spec) + len(nons)) + [0] * len(pref)
            if idx:
                col0 = t.tensor([c[0] for c in idx], dtype=t.long, device=self.dev)
                self.log["materialized_rows"] += int((arm.hdr[col0] > 0).sum())
                gdl.gdl_materialize_launch(
                    t.tensor(idx, dtype=t.int32, device=self.dev),
                    t.tensor(flags, dtype=t.int32, device=self.dev),
                    L["A_log"], L["dt_bias"], arm.state, arm.hdr)
        n = si.shape[0]
        if n == 0:
            return out
        if has_prefill or nons:
            # stock readers, which know nothing about rings: FLA for spec rows
            # (col a-1, writes every column), chunk kernel for non-spec rows
            # (col 0 -> col 0). GPU: the stock CUDA kernel (same layout semantics);
            # CPU: the eager kernel with a throwaway all-zero header table.
            if torch_is_cuda(self.dev):
                self._run(arm, "stock_cuda", mixed, a, b, gate, si, cu, acc, seq, out,
                          lazy_ok=False)
            else:
                blind = t.zeros_like(arm.hdr)
                gdl.gdl_decode_launch(mixed, a, b, L["A_log"], L["dt_bias"], si, cu, acc, seq,
                                      arm.state, blind, gate, L["norm_w"], out, L["H"],
                                      128 ** -0.5, 1e-6, True, block_size=self.BS,
                                      mode=gdl.GDL_MODE_EAGER)
        else:
            self._run(arm, arm.decode, mixed, a, b, gate, si, cu, acc, seq, out,
                      lazy_ok=arm.lazy)
            if arm.lazy:
                pend = arm.hdr[si[:, 0].long()] > 0
                self.log["lazy_rows"] += int(pend.sum())
                self.log["spec_rows"] += int(si.shape[0])
        return out

    def _run(self, arm, decode, mixed, a, b, gate, si, cu, acc, seq, out, lazy_ok):
        t, gdl, L = self.t, self.gdl, self.L
        if decode == "stock_cuda":
            stock_cuda(mixed_qkv=mixed, a=a, b=b, A_log=L["A_log"], dt_bias=L["dt_bias"],
                       state_indices=si, cu_seqlens=cu, num_accepted_tokens=acc,
                       state=arm.state, output_gate=gate, norm_weight=L["norm_w"], out=out,
                       scale=128 ** -0.5, norm_eps=1e-6, output_gate_activation="sigmoid")
            return
        mode = gdl.GDL_MODE_AUTO if (decode == "k3" and lazy_ok) else gdl.GDL_MODE_EAGER
        bs = 0 if (self.break_boundary and arm.lazy) else self.BS
        gdl.gdl_decode_launch(mixed, a, b, L["A_log"], L["dt_bias"], si, cu, acc, seq,
                              arm.state, arm.hdr, gate, L["norm_w"], out, L["H"],
                              128 ** -0.5, 1e-6, True, block_size=bs, mode=mode)

    def _check_req(self, r):
        """Committed state (after a materialize of the K3 arm) and every
        checkpoint block this request wrote, bitwise across arms."""
        t = self.t
        if r["prefilling"] or r["sidx"] < 0:
            return 0
        cols = r["bt"][r["sidx"]: r["sidx"] + self.W]
        cols = cols + [0] * (self.W - len(cols))
        arms = list(self.arms.values())
        for arm in arms:
            if arm.lazy:
                self.gdl.gdl_materialize_launch(
                    t.tensor([cols], dtype=t.int32, device=self.dev),
                    t.ones(1, dtype=t.int32, device=self.dev),
                    self.L["A_log"], self.L["dt_bias"], arm.state, arm.hdr)
        blocks = set(r["ckpt"]) | {cols[r["acc"] - 1]}
        for blk in sorted(blocks):
            ref = bits(t, arms[0].state[blk])
            for arm in arms[1:]:
                if not t.equal(bits(t, arm.state[blk]), ref):
                    raise AssertionError(f"req {r['id']} state block {blk} differs: "
                                         f"{ulp_hist(t, arm.state[blk], arms[0].state[blk])}")
        return len(blocks)

    def check_states(self):
        n = 0
        for r in list(self.reqs):
            n += self._check_req(r)
        return n


def torch_is_cuda(dev) -> bool:
    return str(dev).startswith("cuda")


def _cumsum(xs):
    s = 0
    for x in xs:
        s += x
        yield s


def run_sim(torch, gdl, device, steps, W=4, BS=16, nreq=3, H=1, G=2, seed=1, nslots=None,
            arms=("stock", "k3"), **brk):
    gen = torch.Generator().manual_seed(seed)
    L = make_layer(torch, device, H, G, gen)
    nslots = nslots or (nreq + 2) * ((3 * BS + 41 * W) // BS + W + 2) + 1
    sim = AlignSim(torch, gdl, device, L, W, BS, nreq, nslots, seed, arms=arms, **brk)
    for _ in range(steps):
        sim.step()
    sim.log["state_blocks_checked"] = sim.log.get("state_blocks_checked", 0) + sim.check_states()
    return sim.log


# ------------------------------------------------------------------- GPU paths
def cmd_selftest(args, torch, gdl):
    ok, detail = gdl.gdl_self_test(torch.device("cuda"), G=3, sigmoid=True)
    print(json.dumps({"ok": ok, "detail": detail}))
    return 0 if ok else 1


def cmd_exact(args, torch, gdl):
    dev = torch.device("cuda")
    gen = torch.Generator().manual_seed(args.seed)
    rng = random.Random(args.seed)
    L = make_layer(torch, dev, 8, 3, gen)  # real per-rank shapes: H=8, HV=24
    W = args.k + 1
    report = {}
    for c in args.conc:
        nslots = 1 + c * W + 4
        st0 = (torch.randn(nslots, L["HV"], 128, 128, generator=gen) * 0.05).to(dev)
        s_s, s_l = st0.clone(), st0.clone()
        hdr = torch.zeros(nslots, dtype=torch.int32, device=dev)
        si = torch.arange(1, 1 + c * W, dtype=torch.int32, device=dev).view(c, W)
        acc = torch.ones(c, dtype=torch.int32, device=dev)
        worst = {"max_ulp": 0, "n_diff": 0}
        for step in range(args.steps):
            lens = [W if rng.random() < 0.8 else rng.randint(1, W) for _ in range(c)]
            mixed, a, b, gate = make_tokens(torch, dev, L, lens, gen)
            cu = torch.tensor([0] + list(_cumsum(lens)), dtype=torch.int32, device=dev)
            seq = torch.full((c,), 1000, dtype=torch.int32, device=dev)
            o_s = torch.zeros(mixed.shape[0], L["HV"], 128, dtype=torch.bfloat16, device=dev)
            o_l = torch.zeros_like(o_s)
            stock_cuda(mixed_qkv=mixed, a=a, b=b, A_log=L["A_log"], dt_bias=L["dt_bias"],
                       state_indices=si, cu_seqlens=cu, num_accepted_tokens=acc, state=s_s,
                       output_gate=gate, norm_weight=L["norm_w"], out=o_s, scale=128 ** -0.5,
                       norm_eps=1e-6, output_gate_activation="sigmoid")
            gdl.gdl_decode_launch(mixed, a, b, L["A_log"], L["dt_bias"], si, cu, acc, seq, s_l,
                                  hdr, gate, L["norm_w"], o_l, 8, 128 ** -0.5, 1e-6, True)
            h = ulp_hist(torch, o_s, o_l)
            if h["n_diff"]:
                worst = {"max_ulp": max(worst["max_ulp"], h["max_ulp"]),
                         "n_diff": worst["n_diff"] + h["n_diff"]}
            acc = torch.tensor([rng.randint(1, T) for T in lens], dtype=torch.int32, device=dev)
        gdl.gdl_materialize_launch(si, torch.ones(c, dtype=torch.int32, device=dev),
                                   L["A_log"], L["dt_bias"], s_l, hdr)
        sh = {"max_ulp": 0, "n_diff": 0}
        for r in range(c):
            col = int(si[r, int(acc[r]) - 1])
            h = ulp_hist(torch, s_s[col], s_l[col])
            sh = {"max_ulp": max(sh["max_ulp"], h["max_ulp"]), "n_diff": sh["n_diff"] + h["n_diff"]}
        report[f"c{c}"] = {"steps": args.steps, "out": worst, "committed_state": sh,
                           "A0": worst["n_diff"] == 0 and sh["n_diff"] == 0}
    print(json.dumps(report, indent=1))
    return 0 if all(v["A0"] for v in report.values()) else 1


def cmd_sim(args, torch, gdl):
    dev = torch.device(args.device)
    log = run_sim(torch, gdl, dev, args.steps, W=args.k + 1, BS=args.block, nreq=args.nreq,
                  H=args.heads, G=3 if dev.type == "cuda" else 2, seed=args.seed)
    print(json.dumps(log, indent=1))
    return 0


def _bench_case(torch, gdl, L, c, W, layers, mode, reps, dev, gen):
    """CUDA graph of `layers` launches (distinct state slots per layer), c reqs."""
    nslots = 1 + layers * c * W
    state = (torch.randn(nslots, L["HV"], 128, 128, generator=gen) * 0.05).to(dev)
    hdr = torch.zeros(nslots, dtype=torch.int32, device=dev)
    lens = [W] * c
    mixed, a, b, gate = make_tokens(torch, dev, L, lens, gen)
    cu = torch.tensor([0] + list(_cumsum(lens)), dtype=torch.int32, device=dev)
    acc = torch.full((c,), 2, dtype=torch.int32, device=dev)
    seq = torch.full((c,), 1000, dtype=torch.int32, device=dev)
    sis = [torch.arange(1 + l * c * W, 1 + (l + 1) * c * W, dtype=torch.int32,
                        device=dev).view(c, W) for l in range(layers)]
    out = torch.zeros(mixed.shape[0], L["HV"], 128, dtype=torch.bfloat16, device=dev)

    def body():
        for si in sis:
            if mode == "stock":
                stock_cuda(mixed_qkv=mixed, a=a, b=b, A_log=L["A_log"], dt_bias=L["dt_bias"],
                           state_indices=si, cu_seqlens=cu, num_accepted_tokens=acc,
                           state=state, output_gate=gate, norm_weight=L["norm_w"], out=out,
                           scale=128 ** -0.5, norm_eps=1e-6, output_gate_activation="sigmoid")
            else:
                gdl.gdl_decode_launch(mixed, a, b, L["A_log"], L["dt_bias"], si, cu, acc, seq,
                                      state, hdr, gate, L["norm_w"], out, L["H"], 128 ** -0.5,
                                      1e-6, True, block_size=0,
                                      mode=gdl.GDL_MODE_EAGER if mode == "eager" else
                                      gdl.GDL_MODE_AUTO)

    body()  # warm / compile; lazy mode: now every row has a pending ring of W
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        body()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        body()
    flush = torch.empty(64 << 20, dtype=torch.uint8, device=dev)
    times = []
    for _ in range(reps):
        flush.zero_()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        g.replay()
        e1.record()
        torch.cuda.synchronize()
        times.append(e0.elapsed_time(e1) * 1000 / layers)
    times.sort()
    return times[len(times) // 2], times[0]


def cmd_bench(args, torch, gdl):
    dev = torch.device("cuda")
    gen = torch.Generator().manual_seed(args.seed)
    L = make_layer(torch, dev, 8, 3, gen)
    W = args.k + 1
    res = {}
    for c in args.conc:
        for rnd in range(args.abab):
            for mode in ("stock", "lazy", "eager") if rnd % 2 == 0 else ("eager", "lazy", "stock"):
                med, best = _bench_case(torch, gdl, L, c, W, args.layers, mode, args.reps, dev, gen)
                res.setdefault(f"c{c}", {}).setdefault(mode, []).append(round(med, 2))
    for c, d in res.items():
        d["us_per_layer_median"] = {m: sorted(v)[len(v) // 2] for m, v in d.items()
                                    if isinstance(v, list)}
        med = d["us_per_layer_median"]
        d["delta_ms_per_step_36_layers"] = round((med["lazy"] - med["stock"]) * 36 / 1000, 3)
    print(json.dumps(res, indent=1))
    return 0


def cmd_ncu(args, torch, gdl):
    """Run under: ncu --metrics dram__bytes_read.sum,dram__bytes_write.sum
    -k regex:'gdn_decode_post_conv_mtp|_gdl_decode_kernel' python3 k3_harness.py ncu"""
    dev = torch.device("cuda")
    gen = torch.Generator().manual_seed(args.seed)
    L = make_layer(torch, dev, 8, 3, gen)
    W = args.k + 1
    for c in args.conc:
        for mode in ("stock", "lazy"):
            _bench_case(torch, gdl, L, c, W, 1, mode, 1, dev, gen)
    print("expect per layer per request: stock ~1.6 MB read + ~6.3 MB write; "
          "lazy ~1.6 MB read + ~1.6 MB write (+ <= 66 KB ring)")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["selftest", "exact", "sim", "bench", "ncu"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--conc", type=int, nargs="+", default=[1, 8])
    ap.add_argument("--block", type=int, default=1600, help="mamba block size (sim)")
    ap.add_argument("--nreq", type=int, default=8)
    ap.add_argument("--heads", type=int, default=8, help="key heads per rank (sim)")
    ap.add_argument("--layers", type=int, default=36)
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--abab", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    import torch

    if args.cmd != "sim" or args.device != "cpu":
        if not torch.cuda.is_available():
            print("needs a GPU (only 'sim --device cpu' runs without one)", file=sys.stderr)
            return 2
    else:
        os.environ.setdefault("TRITON_INTERPRET", "1")
    gdl = load_gdl()
    return {"selftest": cmd_selftest, "exact": cmd_exact, "sim": cmd_sim,
            "bench": cmd_bench, "ncu": cmd_ncu}[args.cmd](args, torch, gdl)


if __name__ == "__main__":
    raise SystemExit(main())
