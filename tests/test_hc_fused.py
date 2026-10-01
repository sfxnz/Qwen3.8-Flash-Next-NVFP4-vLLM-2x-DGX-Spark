#!/usr/bin/env python3
"""K5 fused hyper-connection: overlay integrity (host) + kernel numerics (torch).

Host (stdlib only): docker/v030/hyperconnection.py is exactly stock v0.30.0 +
two dispatch hunks + a verbatim copy of docker/v030/hc_fused.py, carries a
valid R11 header, compiles, and is default-off.

Torch (skips without torch/triton/vllm; run inside the v0.30 image, no GPU):
the overlay module is imported from the file, the stock GatedResidual methods
run on a stub module with random weights, and the embedded K5 kernels run
under TRITON_INTERPRET=1 on CPU. Checked: new residual bitwise equal to the
stock kernel; lora / injection / block_input class A (<= 2x stock's relative
error vs an fp64 reference); arrival counters self-reset; the load-time
self-test passes and fails closed on a broken kernel; the hooks dispatch.

  docker run --rm --entrypoint python3 -e TRITON_INTERPRET=1 -e CUDA_VISIBLE_DEVICES= \
    -v $PWD:/w -w /w vllm/vllm-openai:v0.30.0-aarch64 -m unittest tests.test_hc_fused -v
Set HC_FULL=0 to skip the full-shape (336,10240)/(10240,320) cases.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import py_compile
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "docker" / "v030" / "apply_hc_overlay.py"
KERNELS = ROOT / "docker" / "v030" / "hc_fused.py"
OVERLAY = ROOT / "docker" / "v030" / "hyperconnection.py"
V030_DIGEST = "sha256:4864d46625cbc3307623e29ac742030655e27249feba7b97ec925ce4cc4dfb56"
STOCK_SHA = "2a15d6b22fbe1def4d2bcd85568269d1157c5e1c298f0c6af43c1fb9797489ba"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class OverlayHost(unittest.TestCase):
    def setUp(self):
        self.gen = _load(GEN, "apply_hc_overlay")
        self.text = OVERLAY.read_text()

    def test_header(self):
        head = self.text.split("\n", 8)
        self.assertEqual(head[0], "# R11-OVERLAY")
        self.assertIn(f"# base_image_digest: {V030_DIGEST}", self.text[:800])
        self.assertIn("# upstream_file: vllm/models/qwen4_exp/nvidia/hyperconnection.py", self.text[:800])
        self.assertIn(f"# upstream_file_sha256: {STOCK_SHA}", self.text[:800])

    def test_roundtrip_is_stock_plus_hunks_plus_kernels(self):
        stock, kernels = self.gen.unoverlay(self.text)
        self.assertEqual(hashlib.sha256(stock.encode()).hexdigest(), STOCK_SHA)
        self.assertEqual(kernels, KERNELS.read_text(), "overlay embeds a stale hc_fused.py; rerun the generator")
        self.assertEqual(self.gen.overlay(stock, kernels), self.text)

    def test_refuses_other_stock(self):
        stock, kernels = self.gen.unoverlay(self.text)
        with self.assertRaises(SystemExit):
            self.gen.overlay(stock + "\n", kernels)

    def test_compiles_and_default_off(self):
        py_compile.compile(str(OVERLAY), doraise=True)
        k = KERNELS.read_text()
        self.assertIn('os.environ.get(HCF_ENV, "0") == "1"', k)
        self.assertIn('HCF_ENV = "VLLM_QWEN38_HC_FUSED"', k)
        self.assertNotIn("from __future__", k)
        self.assertEqual(self.text.count("hcf_dispatch(self, hidden_states"), 2)


# ------------------------------------------------------------------ torch side
def _deps():
    try:
        import torch  # noqa: F401
        import triton  # noqa: F401
    except Exception:  # noqa: BLE001
        return None
    import torch

    if not torch.cuda.is_available():
        os.environ.setdefault("TRITON_INTERPRET", "1")
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # keeps vllm's triton enabled
    try:
        mod = _load(OVERLAY, "vllm.models.qwen4_exp.nvidia.hyperconnection_k5")
    except Exception as e:  # noqa: BLE001
        print(f"overlay import failed: {type(e).__name__}: {e}", file=sys.stderr)
        return None
    return torch, mod


_DEPS = None


def deps():
    global _DEPS
    if _DEPS is None:
        _DEPS = _deps() or False
    return _DEPS


class _Lin:
    def __init__(self, torch, w):
        import torch.nn.functional as F

        self.weight = w
        self._f = F.linear

    def __call__(self, x):
        return self._f(x, self.weight)


def make_stub(torch, mod, hc, h, r, use_combine, shared_norm=False, seed=0, dev="cpu"):
    g = torch.Generator().manual_seed(seed)
    d = hc * h
    bf = torch.bfloat16
    s = types.SimpleNamespace()
    s.config = types.SimpleNamespace(params_dtype=bf, rms_norm_eps=1e-6)
    s.hc_count, s.hidden_size, s.lora_rank, s.use_combine = hc, h, r, use_combine
    s.pad_size = (-(r + hc)) % 16 if use_combine else 0
    s.hc_norm = types.SimpleNamespace(
        weight=(torch.randn(h if shared_norm else d, generator=g) * 0.3).to(bf).to(dev)
    )
    rows = r + hc + s.pad_size if use_combine else r
    down = (torch.randn(rows, d, generator=g) / d**0.5 * 4).to(bf).to(dev)
    if use_combine:
        s.input_mix_weight_down_block_inject = _Lin(torch, down)
    else:
        s.input_mix_weight_down = _Lin(torch, down)
    s.input_mix_weight_up = _Lin(torch, (torch.randn(d, r, generator=g) / r**0.5 * 3).to(bf).to(dev))
    GR = mod.GatedResidual
    s.mix = types.MethodType(GR.mix, s)
    s.combine_and_mix = types.MethodType(GR.combine_and_mix, s)
    return s


class KernelNumerics(unittest.TestCase):
    def setUp(self):
        d = deps()
        if not d:
            self.skipTest("torch/triton/vllm v0.30 not importable (run inside the image)")
        self.torch, self.mod = d
        self.dev = "cuda" if self.torch.cuda.is_available() else "cpu"
        self.mod._HCF_STATE.update(enabled=False, failed=False, busy=False)
        self.errs = []

    def _stock(self, s, hs, bo, inj):
        self.mod._HCF_STATE["busy"] = True
        try:
            return s.combine_and_mix(hs, bo, inj) if bo is not None else s.mix(hs)
        finally:
            self.mod._HCF_STATE["busy"] = False

    def _check(self, s, m, with_block, with_inj, dist=0, seed=1):
        torch, mod = self.torch, self.mod
        gen = torch.Generator().manual_seed(seed)
        hs, bo, inj = mod._hcf_inputs(dist, m, s.hc_count, s.hidden_size, self.dev, gen)
        bo = bo if with_block else None
        inj = inj if with_block and with_inj else None
        w = mod._hcf_weights(s)
        self.assertIsNotNone(w)
        st = self._stock(s, hs, bo, inj)
        fu = mod._hcf_run(s, w, hs, bo, inj)
        tag = f"hc={s.hc_count} h={s.hidden_size} r={s.lora_rank} comb={s.use_combine} M={m} blk={with_block} inj={with_inj} dist={dist}"
        self.assertTrue(torch.equal(st[0], fu[0]), f"{tag}: residual not bitwise equal")
        if not with_block:
            self.assertIs(fu[0], hs)
        r_bi, r_inj = mod._hcf_ref64(s, w, hs, bo, inj)
        es, ef = mod._hcf_rel(st[1], r_bi), mod._hcf_rel(fu[1], r_bi)
        self.assertLessEqual(ef, 2 * es + 2**-12, f"{tag}: block_input {ef:.3e} vs stock {es:.3e}")
        self.assertEqual(mod._hcf_elem_bad(st[1], fu[1], r_bi), 0, f"{tag}: block_input gross error")
        if s.use_combine:
            self.assertEqual(tuple(fu[2].shape), (m, s.hc_count))
            self.assertEqual(mod._hcf_elem_bad(st[2], fu[2], r_inj), 0, f"{tag}: injection gross error")
            self.errs.append((mod._hcf_rel(st[2], r_inj), mod._hcf_rel(fu[2], r_inj)))
        else:
            self.assertIsNone(fu[2])
            self.assertIsNone(st[2])
        cnt = mod._HCF_STATE["counters"].get(str(hs.device))
        self.assertTrue(cnt is not None and int(cnt.abs().sum()) == 0, "arrival counters not reset")
        return st, fu, (hs, bo, inj, w)

    def test_reduced_all_variants(self):
        # h=256, r=32: same code paths as (4, 2560, 320) - split-K 8 x BLOCK_K 128,
        # 36 useful rows of a 48-row padded merged weight, K tiles per stream.
        for use_combine in (True, False):
            for shared in (False, True):
                s = make_stub(self.torch, self.mod, 4, 256, 32, use_combine, shared, dev=self.dev)
                for m in (1, 3, 8, 17, 32):
                    for with_block, with_inj in ((True, True), (True, False), (False, False)):
                        self._check(s, m, with_block, with_inj, dist=m % 3)
        # injection has only M*4 values per case: judge its class A error pooled
        es = sum(a * a for a, _ in self.errs) ** 0.5
        ef = sum(b * b for _, b in self.errs) ** 0.5
        print(f"\n  injection pooled rel err fused/stock = {ef / es:.2f} over {len(self.errs)} cases", file=sys.stderr)
        self.assertLessEqual(ef, 2 * es)

    def test_distributions(self):
        s = make_stub(self.torch, self.mod, 4, 256, 32, True, dev=self.dev)
        for dist in range(3):
            for m in (2, 5):
                self._check(s, m, True, True, dist=dist, seed=dist + 10)

    def test_deterministic(self):
        s = make_stub(self.torch, self.mod, 4, 256, 32, True, dev=self.dev)
        _, fu, (hs, bo, inj, w) = self._check(s, 6, True, True)
        again = self.mod._hcf_run(s, w, hs, bo, inj)
        for a, b in zip(fu, again):
            self.assertTrue(self.torch.equal(a, b))

    def test_full_shape(self):
        if os.environ.get("HC_FULL", "1") == "0":
            self.skipTest("HC_FULL=0")
        # Real shapes: merged down+inject (336, 10240), up (10240, 320).
        s = make_stub(self.torch, self.mod, 4, 2560, 320, True, dev=self.dev)
        self.assertEqual(tuple(s.input_mix_weight_down_block_inject.weight.shape), (336, 10240))
        self.assertEqual(tuple(s.input_mix_weight_up.weight.shape), (10240, 320))
        for m in (1, 4, 32):
            self._check(s, m, True, True, dist=m % 3)
        self._check(s, 2, False, False)  # layer-0 mix()
        f = make_stub(self.torch, self.mod, 4, 2560, 320, False, dev=self.dev)  # final mixer (320, 10240)
        self._check(f, 1, True, True, seed=3)

    def test_self_test_passes_and_fails_closed(self):
        mod = self.mod
        s = make_stub(self.torch, mod, 4, 256, 32, True, dev=self.dev)
        ok, detail = mod.hcf_self_test(s, True, True, self.dev, ms=(1, 4, 8))
        self.assertTrue(ok, detail)
        print(f"\n  self-test: {detail}", file=sys.stderr)
        ok, _ = mod.hcf_self_test(s, False, False, self.dev, ms=(1, 4))
        self.assertTrue(ok)
        real = mod.hcf_up_gatemix

        def broken(*a, **k):
            out = real(*a, **k)
            out[:, ::7] = out[:, ::7] * 1.05  # 5% error on 1/7 of columns
            return out

        mod.hcf_up_gatemix = broken
        try:
            ok, detail = mod.hcf_self_test(s, True, True, self.dev, ms=(1, 4))
        finally:
            mod.hcf_up_gatemix = real
        self.assertFalse(ok)
        self.assertIn("block_input", detail)
        self.assertFalse(mod._HCF_STATE["busy"])

    def test_dispatch_gates(self):
        torch, mod = self.torch, self.mod
        s = make_stub(torch, mod, 4, 256, 32, True, dev=self.dev)
        hs = torch.zeros(4, 1024, dtype=torch.bfloat16, device=self.dev)
        bo = torch.zeros(4, 256, dtype=torch.bfloat16, device=self.dev)
        self.assertIsNone(mod.hcf_dispatch(s, hs, bo, None))  # env off
        mod._HCF_STATE["enabled"] = True
        try:
            big = torch.zeros(33, 1024, dtype=torch.bfloat16, device=self.dev)
            self.assertIsNone(mod.hcf_dispatch(s, big, None, None))  # M > 32
            if self.dev == "cpu":
                self.assertIsNone(mod.hcf_dispatch(s, hs, bo, None))  # not CUDA
            mod._HCF_STATE["failed"] = True
            self.assertIsNone(mod.hcf_dispatch(s, hs, bo, None))
            mod._HCF_STATE["failed"] = False
            q = make_stub(torch, mod, 4, 256, 32, True, dev=self.dev)
            q.input_mix_weight_up.weight = q.input_mix_weight_up.weight.float()  # quantized/other layout
            self.assertIsNone(mod._hcf_weights(q))
        finally:
            mod._HCF_STATE.update(enabled=False, failed=False)

    def test_dispatch_fail_closed_paths(self):
        # On CPU the is_cuda gate alone makes every real dispatch None; a
        # CUDA-looking stand-in reaches the self-test / capture / fused branches.
        torch, mod = self.torch, self.mod
        s = make_stub(torch, mod, 4, 256, 32, True, dev=self.dev)
        hs = types.SimpleNamespace(
            shape=(4, 1024), is_cuda=True, dtype=torch.bfloat16, device="cpu",
            dim=lambda: 2, stride=lambda i: (1024, 1)[i],
        )
        calls, verdict, capturing = [], [True], [False]
        real = (mod.hcf_self_test, mod._hcf_run, mod._hcf_counters, torch.cuda.is_current_stream_capturing)
        mod.hcf_self_test = lambda *a, **k: (calls.append("test"), (verdict[0], "stub"))[1]
        mod._hcf_run = lambda *a: "fused"
        mod._hcf_counters = lambda dev: None
        torch.cuda.is_current_stream_capturing = lambda: capturing[0]
        st = mod._HCF_STATE
        saved_tested = set(st["tested"])
        st["tested"].clear()
        st.update(enabled=True, failed=False)
        try:
            capturing[0] = True  # untested variant met during capture -> stock, no test
            self.assertIsNone(mod.hcf_dispatch(s, hs, None, None))
            self.assertEqual(calls, [])
            capturing[0] = False
            verdict[0] = False  # failing self-test -> stock, K5 off process-wide
            self.assertIsNone(mod.hcf_dispatch(s, hs, None, None))
            self.assertTrue(st["failed"])
            verdict[0] = True
            self.assertIsNone(mod.hcf_dispatch(s, hs, None, None))
            self.assertEqual(calls, ["test"])
            st["failed"] = False  # passing self-test -> fused; tested once per variant
            self.assertEqual(mod.hcf_dispatch(s, hs, None, None), "fused")
            self.assertEqual(mod.hcf_dispatch(s, hs, None, None), "fused")
            self.assertEqual(calls, ["test", "test"])
            capturing[0] = True  # tested variant inside capture -> fused
            self.assertEqual(mod.hcf_dispatch(s, hs, None, None), "fused")
            big = types.SimpleNamespace(**{**vars(hs), "shape": (33, 1024)})
            self.assertIsNone(mod.hcf_dispatch(s, big, None, None))  # M > 32
            st["enabled"] = False
            self.assertIsNone(mod.hcf_dispatch(s, hs, None, None))  # env off
        finally:
            mod.hcf_self_test, mod._hcf_run, mod._hcf_counters, torch.cuda.is_current_stream_capturing = real
            st["tested"].clear()
            st["tested"].update(saved_tested)
            st.update(enabled=False, failed=False)

    def test_self_test_covers_dispatch_range(self):
        # Every BLOCK_M the dispatch can launch (16, 32) and Triton's integer
        # specialisations of M (==1, %16==0, other) must be self-tested.
        mod = self.mod
        ms = mod.HCF_SELF_TEST_MS
        self.assertEqual({mod._hcf_block_m(m) for m in ms}, {mod._hcf_block_m(m) for m in range(1, mod.HCF_MAX_M + 1)})
        self.assertIn(1, ms)
        self.assertTrue(any(m % 16 == 0 for m in ms) and any(m % 16 and m > 16 for m in ms))
        self.assertEqual(mod.hcf_self_test.__defaults__[-1], ms)

    def test_hooks_call_dispatch(self):
        torch, mod = self.torch, self.mod
        s = make_stub(torch, mod, 4, 256, 32, True, dev=self.dev)
        seen = []
        real = mod.hcf_dispatch

        def fake(module, hs, bo, inj):
            seen.append(bo is not None)
            return ("fused",)

        mod.hcf_dispatch = fake
        try:
            hs = torch.zeros(2, 1024, dtype=torch.bfloat16, device=self.dev)
            bo = torch.zeros(2, 256, dtype=torch.bfloat16, device=self.dev)
            self.assertEqual(mod.GatedResidual.mix(s, hs), ("fused",))
            self.assertEqual(mod.GatedResidual.combine_and_mix(s, hs, bo, None), ("fused",))
        finally:
            mod.hcf_dispatch = real
        self.assertEqual(seen, [False, True])


if __name__ == "__main__":
    unittest.main()
