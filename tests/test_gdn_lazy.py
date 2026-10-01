#!/usr/bin/env python3
"""K3 lazy GDN state commit: overlay integrity (host) + kernel semantics (CPU).

Host (stdlib only): docker/v030/gdn_lazy_linear_attn.py is exactly stock v0.30.0
qwen_gdn_linear_attn.py + three hooks + a verbatim copy of docker/v030/gdn_lazy.py;
docker/v030/gdn_lazy_attn.py is stock gdn_attn.py + the S1.5 hunk + the K3
metadata hunks; both carry valid R11 headers, compile, and are default-off.

Torch (skips without torch/triton; run inside the v0.30 image, no GPU): the
kernels run under TRITON_INTERPRET=1 with reduced shapes (1-2 key heads, G=2-3)
but the real K = V = 128, W = k+1 = 4, BV = 64 tiling and ring geometry.
Checked: eager rows match an fp32 torch model of the stock kernel; lazy rows are
bitwise equal to eager rows over many steps (outputs every step, committed
state after a materialize) through the load-time self-test code; materialize
reproduces the stock layout; the V2 'align' slot scheme end to end (block
crossings, checkpoint copies, mixed steps with non-spec rows, slot reuse) is
bitwise equal to the stock layout arm, and the same sim FAILS when the fixup
or the boundary rule is removed (negative controls); the dispatch fails closed.

  docker run --rm --network none --memory 4g --entrypoint python3 \
    -e TRITON_INTERPRET=1 -e CUDA_VISIBLE_DEVICES= -v $PWD:/w -w /w \
    vllm/vllm-openai:v0.30.0-aarch64 -m unittest tests.test_gdn_lazy -v
Set GDL_SIM_STEPS (default 24) to lengthen the slot-scheme sim.
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
V030 = ROOT / "docker" / "v030"
GEN = V030 / "apply_gdn_lazy_overlay.py"
KERNELS = V030 / "gdn_lazy.py"
LIN = V030 / "gdn_lazy_linear_attn.py"
ATT = V030 / "gdn_lazy_attn.py"
S15 = V030 / "gdn_attn.py"
HARNESS = ROOT / "tools" / "kernels" / "k3_harness.py"
V030_DIGEST = "sha256:4864d46625cbc3307623e29ac742030655e27249feba7b97ec925ce4cc4dfb56"
LIN_SHA = "b80f8e6f3fff442fb880aebb3209c52f814f66490009f4ac4003de603576d10a"
ATT_SHA = "3831019404a92200a38b43571c040cffeb6638bbdc71e4cb5e680dbc6c93d054"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class OverlayHost(unittest.TestCase):
    def setUp(self):
        self.gen = _load(GEN, "apply_gdn_lazy_overlay")
        self.lin = LIN.read_text()
        self.att = ATT.read_text()

    def test_headers(self):
        for text, up, s in ((self.lin, self.gen.LIN_FILE, LIN_SHA), (self.att, self.gen.ATT_FILE, ATT_SHA)):
            self.assertTrue(text.startswith("# R11-OVERLAY\n"))
            head = text[:1200]
            self.assertIn(f"# base_image_digest: {V030_DIGEST}", head)
            self.assertIn(f"# upstream_file: {up}", head)
            self.assertIn(f"# upstream_file_sha256: {s}", head)
            self.assertIn("# generator: docker/v030/apply_gdn_lazy_overlay.py (do not hand-edit)", head)

    def test_linear_roundtrip_is_stock_plus_hooks_plus_kernels(self):
        stock, kernels = self.gen.unoverlay_linear(self.lin)
        self.assertEqual(sha(stock), LIN_SHA)
        self.assertEqual(kernels, KERNELS.read_text(), "overlay embeds a stale gdn_lazy.py; rerun the generator")
        self.assertEqual(self.gen.overlay_linear(stock, kernels), self.lin)
        self.assertEqual(self.lin.count("gdl_layer_init(self)"), 1)
        self.assertEqual(self.lin.count("if gdl_decode(self, mixed_qkv, a, b, output_gate, core_attn_out, attn_metadata):"), 1)
        self.assertEqual(self.lin.count("gdl_fixup(self, attn_metadata)"), 1)

    def test_attn_roundtrip_and_contains_s15(self):
        stock = self.gen.unoverlay_attn(self.att)
        self.assertEqual(sha(stock), ATT_SHA)
        self.assertEqual(self.gen.overlay_attn(stock), self.att)
        s15 = _load(V030 / "apply_gdn_fresh_prefill_overlay.py", "apply_gdn_fresh_prefill_overlay")
        self.assertEqual(s15.unoverlay(S15.read_text()), stock, "S1.5 overlay built from another stock file")
        for _old, new in s15.HUNKS:
            self.assertEqual(self.att.count(new), 1, "K3 attn overlay must carry the S1.5 hunk")

    def test_refuses_other_stock(self):
        stock, kernels = self.gen.unoverlay_linear(self.lin)
        with self.assertRaises(SystemExit):
            self.gen.overlay_linear(stock + "\n", kernels)
        with self.assertRaises(SystemExit):
            self.gen.overlay_attn(self.gen.unoverlay_attn(self.att) + "\n")

    def test_compiles_and_default_off(self):
        py_compile.compile(str(LIN), doraise=True)
        py_compile.compile(str(ATT), doraise=True)
        k = KERNELS.read_text()
        self.assertIn('GDL_ENV = "VLLM_QWEN38_GDN_LAZY"', k)
        self.assertIn('"env": os.environ.get(GDL_ENV, "0").strip().lower()', k)
        self.assertIn('in ("1", "force")', k)
        self.assertNotIn("from __future__", k)
        self.assertIn('_os.environ.get("VLLM_QWEN38_GDN_LAZY", "0").strip().lower() in ("1", "force")', self.att)

    def test_ring_geometry(self):
        src = KERNELS.read_text()
        ns: dict = {}
        exec("\n".join(l.split("  #")[0] for l in src.splitlines()
                       if l.startswith(("GDL_DIM =", "GDL_BV =", "GDL_TMAX =", "GDL_HEAD_BF16 =",
                                        "GDL_RING_BF16 =", "GDL_RING_OFF ="))), ns)
        # ring = TMAX x (k|v) + a + b, ends exactly at the head's region end,
        # and lies entirely in the last V chunk (rows 96..127), so only chunk 3
        # of a column-1 store can overlap it (handled by the barrier / deferral)
        self.assertEqual(ns["GDL_RING_BF16"], 8 * 256 + 16)
        self.assertEqual(ns["GDL_RING_OFF"] + ns["GDL_RING_BF16"], ns["GDL_HEAD_BF16"])
        first_row = ns["GDL_RING_OFF"] // (2 * ns["GDL_DIM"])
        self.assertGreaterEqual(first_row, 128 - ns["GDL_BV"])


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
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    h = _load(HARNESS, "k3_harness")
    return torch, h.load_gdl(), h


_DEPS = None


def deps():
    global _DEPS
    if _DEPS is None:
        _DEPS = _deps() or False
    return _DEPS


def _dev(torch):
    return "cuda" if torch.cuda.is_available() else "cpu"


class Kernels(unittest.TestCase):
    def setUp(self):
        d = deps()
        if not d:
            self.skipTest("torch/triton unavailable")
        self.torch, self.gdl, self.h = d
        self.dev = _dev(self.torch)

    def _eager_fn(self, hdr_zero_size):
        """Stock-layout stand-in with the stock op's keyword interface: the K3
        kernel in eager mode with a private, always-zero header table."""
        torch, gdl = self.torch, self.gdl

        def fn(mixed_qkv, a, b, A_log, dt_bias, state_indices, cu_seqlens, num_accepted_tokens,
               state, output_gate, norm_weight, out, scale, norm_eps, output_gate_activation):
            hdr = torch.zeros(state.shape[0], dtype=torch.int32, device=state.device)
            H = (mixed_qkv.shape[1] - state.shape[1] * 128) // 256
            seq = torch.zeros(state_indices.shape[0], dtype=torch.int32, device=state.device)
            gdl.gdl_decode_launch(mixed_qkv, a, b, A_log, dt_bias, state_indices, cu_seqlens,
                                  num_accepted_tokens, seq, state, hdr, output_gate, norm_weight,
                                  out, H, scale, norm_eps, output_gate_activation == "sigmoid",
                                  mode=gdl.GDL_MODE_EAGER)
        return fn

    def test_eager_matches_stock_model(self):
        """Eager rows vs an fp32 torch model of the stock kernel (class A on CPU;
        the GPU harness 'exact' proves A0 against the real kernel)."""
        torch, gdl, h = self.torch, self.gdl, self.h
        gen = torch.Generator().manual_seed(1)
        L = h.make_layer(torch, self.dev, 2, 2, gen)
        W, n = 4, 3
        st = torch.randn(1 + n * W + 1, L["HV"], 128, 128, generator=gen).mul(0.05).to(self.dev)
        si = torch.arange(1, 1 + n * W, dtype=torch.int32, device=self.dev).view(n, W)
        acc = torch.tensor([1, 2, 1], dtype=torch.int32, device=self.dev)
        # row 1 reads column 1 (accepted 2): give it a distinct state
        lens = [4, 3, 1]
        mixed, a, b, gate = h.make_tokens(torch, self.dev, L, lens, gen)
        cu = torch.tensor([0, 4, 7, 8], dtype=torch.int32, device=self.dev)
        s_r, s_e = st.clone(), st.clone()
        o_r = torch.zeros(8, L["HV"], 128, dtype=torch.bfloat16, device=self.dev)
        o_e = torch.zeros_like(o_r)
        h.ref_step(torch, mixed, a, b, L["A_log"], L["dt_bias"], si, cu, acc, s_r, gate, L["norm_w"],
                   o_r, 2, 128 ** -0.5, 1e-6, True)
        self._eager_fn(0)(mixed_qkv=mixed, a=a, b=b, A_log=L["A_log"], dt_bias=L["dt_bias"],
                          state_indices=si, cu_seqlens=cu, num_accepted_tokens=acc, state=s_e,
                          output_gate=gate, norm_weight=L["norm_w"], out=o_e, scale=128 ** -0.5,
                          norm_eps=1e-6, output_gate_activation="sigmoid")
        u = h.ulp_hist(torch, o_r, o_e)
        self.assertLessEqual(u["max_ulp"], 1, u)
        self.assertLess(u["n_diff"], u["n"] // 50, u)
        written = sorted({int(si[r, t]) for r, T in enumerate(lens) for t in range(T)})
        for s_ in written:
            err = float((s_r[s_] - s_e[s_]).abs().max())
            self.assertLess(err, 1e-5, f"slot {s_}")
        for s_ in set(range(st.shape[0])) - set(written):
            self.assertTrue(torch.equal(s_e[s_], st[s_]), f"slot {s_} must be untouched")

    def test_lazy_bitwise_equals_eager_via_self_test(self):
        """The load-time self-test (lazy vs stock-layout arm, 6 steps, random
        acceptance, short and padded rows, one forced-eager step)."""
        ok, detail = self.gdl.gdl_self_test(self.dev, G=3, sigmoid=True,
                                            stock_fn=self._eager_fn(0), steps=6)
        self.assertTrue(ok, detail)
        ok, detail = self.gdl.gdl_self_test(self.dev, G=2, sigmoid=False,
                                            dt_dtype=self.torch.bfloat16,
                                            nw_dtype=self.torch.bfloat16,
                                            stock_fn=self._eager_fn(0), steps=3)
        self.assertTrue(ok, detail)

    def test_warp_reduce_variant_self_consistent(self):
        """RED=0 (tl.reduce, the GPU default) path: lazy == eager bitwise too."""
        old = os.environ.get(self.gdl.GDL_RED_ENV)
        os.environ[self.gdl.GDL_RED_ENV] = "0"
        try:
            ok, detail = self.gdl.gdl_self_test(self.dev, G=3, stock_fn=self._eager_fn(0), steps=3)
        finally:
            if old is None:
                os.environ.pop(self.gdl.GDL_RED_ENV, None)
            else:
                os.environ[self.gdl.GDL_RED_ENV] = old
        self.assertTrue(ok, detail)

    def test_self_test_fails_closed(self):
        base = self._eager_fn(0)

        def broken(**kw):
            base(**kw)
            kw["out"].view(self.torch.int16)[0, 0, 0] ^= 1  # one ulp in one element

        ok, detail = self.gdl.gdl_self_test(self.dev, G=2, stock_fn=broken, steps=2)
        self.assertFalse(ok)
        self.assertIn("differ", detail)

    def test_lazy_row_io_and_materialize(self):
        """A lazy row writes only column 0 (state) and the ring tail of column 1;
        materialize then reproduces the stock layout bitwise."""
        torch, gdl, h = self.torch, self.gdl, self.h
        gen = torch.Generator().manual_seed(3)
        L = h.make_layer(torch, self.dev, 1, 2, gen)
        W = 4
        st = torch.randn(1 + W + 1, L["HV"], 128, 128, generator=gen).mul(0.05).to(self.dev)
        si = torch.arange(1, 1 + W, dtype=torch.int32, device=self.dev).view(1, W)
        cu = torch.tensor([0, W], dtype=torch.int32, device=self.dev)
        acc = torch.ones(1, dtype=torch.int32, device=self.dev)
        seq = torch.full((1,), 50, dtype=torch.int32, device=self.dev)
        mixed, a, b, gate = h.make_tokens(torch, self.dev, L, [W], gen)
        ref, lz = st.clone(), st.clone()
        hz = torch.zeros(st.shape[0], dtype=torch.int32, device=self.dev)
        hl = torch.zeros_like(hz)
        o1 = torch.zeros(W, L["HV"], 128, dtype=torch.bfloat16, device=self.dev)
        o2 = torch.zeros_like(o1)
        args = (mixed, a, b, L["A_log"], L["dt_bias"], si, cu, acc, seq)
        gdl.gdl_decode_launch(*args, ref, hz, gate, L["norm_w"], o1, 1, 128 ** -0.5, 1e-6, True,
                              mode=gdl.GDL_MODE_EAGER)
        gdl.gdl_decode_launch(*args, lz, hl, gate, L["norm_w"], o2, 1, 128 ** -0.5, 1e-6, True,
                              mode=gdl.GDL_MODE_AUTO)
        self.assertTrue(torch.equal(o1.view(torch.int16), o2.view(torch.int16)))
        self.assertEqual(int(hl[1]), W)  # pending ring of W records on column 0's block
        self.assertEqual(int(hz.abs().sum()), 0)
        self.assertTrue(torch.equal(lz[1], st[1]), "committed base S_c (accepted=1: col 0 before step)")
        for s_ in (3, 4, 0, 5):
            self.assertTrue(torch.equal(lz[s_], st[s_]), f"lazy must not write slot {s_}")
        ring_rows = lz[2].view(L["HV"], 128, 128)[:, 120:, :]
        body_rows = lz[2].view(L["HV"], 128, 128)[:, :119, :]
        self.assertTrue(torch.equal(body_rows, st[2].view(L["HV"], 128, 128)[:, :119, :]))
        self.assertFalse(torch.equal(ring_rows, st[2].view(L["HV"], 128, 128)[:, 120:, :]))
        gdl.gdl_materialize_launch(si, torch.ones(1, dtype=torch.int32, device=self.dev),
                                   L["A_log"], L["dt_bias"], lz, hl)
        self.assertEqual(int(hl.abs().sum()), 0)
        for t in range(W):
            s_ = int(si[0, t])
            self.assertTrue(torch.equal(lz[s_].view(torch.int32), ref[s_].view(torch.int32)),
                            f"materialized column {t}")

    def test_prefill_flag_only_drops_headers(self):
        torch, gdl = self.torch, self.gdl
        st = torch.randn(6, 2, 128, 128).to(self.dev)
        before = st.clone()
        hdr = torch.tensor([0, 3, 0, 0, 0, 0], dtype=torch.int32, device=self.dev)
        idx = torch.tensor([[1, 2, 3, 4]], dtype=torch.int32, device=self.dev)
        gdl.gdl_materialize_launch(idx, torch.zeros(1, dtype=torch.int32, device=self.dev),
                                   torch.zeros(2, device=self.dev), torch.zeros(2, device=self.dev),
                                   st, hdr)
        self.assertTrue(torch.equal(st, before))
        self.assertEqual(int(hdr.abs().sum()), 0)

    def test_align_slot_scheme_sim(self):
        steps = int(os.environ.get("GDL_SIM_STEPS", "24"))
        log = self.h.run_sim(self.torch, self.gdl, self.dev, steps, W=4, BS=16, nreq=3, H=1, G=2,
                             seed=1)
        print("sim:", log, file=sys.stderr)
        for key in ("crossings", "checkpoints", "mixed", "reused", "lazy_rows",
                    "materialized_rows", "finished"):
            self.assertGreater(log[key], 0, f"sim never exercised {key}: {log}")
        self.assertGreater(log["state_blocks_checked"], 5)

    def test_sim_negative_controls(self):
        for brk in ("break_fixup", "break_boundary"):
            with self.subTest(brk=brk):
                with self.assertRaises(AssertionError):
                    self.h.run_sim(self.torch, self.gdl, self.dev, 40, W=4, BS=16, nreq=3, H=1,
                                   G=2, seed=1, **{brk: True})


class OverlayImport(unittest.TestCase):
    """The overlays import inside the v0.30 vllm package (CPU, no GPU)."""

    def setUp(self):
        if not deps():
            self.skipTest("torch/triton unavailable")
        try:
            import vllm  # noqa: F401
        except Exception as e:  # noqa: BLE001
            self.skipTest(f"vllm unavailable: {e}")

    def test_import_both(self):
        att = _load(ATT, "vllm.v1.attention.backends.gdn_attn_k3")
        f = att.GDNAttentionMetadata.__dataclass_fields__
        for name in ("lazy_hdr", "lazy_seq_lens", "lazy_block_size", "lazy_ns_state_indices",
                     "lazy_ns_prefilling"):
            self.assertIn(name, f)
        lin = _load(LIN, "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn_k3")
        for name in ("gdl_decode", "gdl_fixup", "gdl_layer_init", "QwenGatedDeltaNetAttention"):
            self.assertTrue(hasattr(lin, name), name)
        self.assertFalse(lin.gdl_enabled() and os.environ.get("VLLM_QWEN38_GDN_LAZY") is None)


class Dispatch(unittest.TestCase):
    """gdl_decode / gdl_fixup on a stub layer + metadata (no vllm import)."""

    def setUp(self):
        d = deps()
        if not d:
            self.skipTest("torch/triton unavailable")
        self.torch, self.gdl, self.h = d
        self.dev = _dev(self.torch)
        self.saved = dict(self.gdl._GDL_STATE)
        self.saved["tested"] = set(self.saved["tested"])

    def tearDown(self):
        self.gdl._GDL_STATE.clear()
        self.gdl._GDL_STATE.update(self.saved)

    def _stub(self, W=4, n=2):
        torch = self.torch
        gen = torch.Generator().manual_seed(5)
        L = self.h.make_layer(torch, self.dev, 1, 2, gen)
        layer = types.SimpleNamespace(
            prefix="model.layers.0.linear_attn", num_v_heads=4, num_k_heads=2, tp_size=2,
            head_k_dim=128, head_v_dim=128, layer_norm_epsilon=1e-6,
            A_log=L["A_log"], dt_bias=L["dt_bias"], gdn_decode_kernel="cuda",
            norm=types.SimpleNamespace(activation="sigmoid", weight=L["norm_w"]),
            kv_cache=[None, torch.randn(1 + n * W + 1, 2, 128, 128, generator=gen).mul(0.05).to(self.dev)],
        )
        hdr = torch.zeros(layer.kv_cache[1].shape[0], dtype=torch.int32, device=self.dev)
        si = torch.arange(1, 1 + n * W, dtype=torch.int32, device=self.dev).view(n, W)
        md = types.SimpleNamespace(
            num_spec_decodes=n, spec_state_indices_tensor=si,
            spec_query_start_loc=torch.tensor([0, W, 2 * W], dtype=torch.int32, device=self.dev),
            num_accepted_tokens=torch.ones(n, dtype=torch.int32, device=self.dev),
            lazy_hdr={layer.prefix: hdr}, lazy_block_size=0,
            lazy_seq_lens=torch.full((n,), 40, dtype=torch.int32, device=self.dev),
            lazy_ns_state_indices=None, lazy_ns_prefilling=None)
        mixed, a, b, gate = self.h.make_tokens(torch, self.dev, L, [W] * n, gen)
        out = torch.zeros(n * W, 2, 128, dtype=torch.bfloat16, device=self.dev)
        return layer, md, mixed, a, b, gate, out

    def test_off_by_default_and_fail_closed(self):
        gdl = self.gdl
        layer, md, mixed, a, b, gate, out = self._stub()
        st = gdl._GDL_STATE
        st.update(env="0", failed=False, used=False, tested=set())
        self.assertFalse(gdl.gdl_decode(layer, mixed, a, b, gate, out, md))
        st.update(env="1", failed=True)
        self.assertFalse(gdl.gdl_decode(layer, mixed, a, b, gate, out, md))
        st.update(failed=False, tested=set())  # enabled but variant never self-tested
        self.assertFalse(gdl.gdl_decode(layer, mixed, a, b, gate, out, md))
        md.lazy_hdr = None
        st["tested"].add(gdl._gdl_layer_key(layer))
        self.assertFalse(gdl.gdl_decode(layer, mixed, a, b, gate, out, md))
        self.assertFalse(st["used"])

    def test_decode_then_fixup(self):
        torch, gdl = self.torch, self.gdl
        layer, md, mixed, a, b, gate, out = self._stub()
        st = gdl._GDL_STATE
        st.update(env="1", failed=False, used=False, tested={gdl._gdl_layer_key(layer)})
        s0 = layer.kv_cache[1].clone()
        self.assertTrue(gdl.gdl_decode(layer, mixed, a, b, gate, out, md))
        self.assertTrue(st["used"])
        hdr = md.lazy_hdr[layer.prefix]
        self.assertEqual(int(hdr[1]), 4)
        self.assertEqual(int(hdr[5]), 4)
        # next step is mixed: row 0 stays spec, row 1 became a non-spec row
        md.num_spec_decodes = 1
        md.spec_state_indices_tensor = md.spec_state_indices_tensor[:1]
        md.lazy_ns_state_indices = torch.tensor([[5, 6, 7, 8]], dtype=torch.int32, device=self.dev)
        md.lazy_ns_prefilling = torch.tensor([False], device=self.dev)
        gdl.gdl_fixup(layer, md)
        self.assertEqual(int(hdr.abs().sum()), 0)
        self.assertFalse(torch.equal(layer.kv_cache[1][2], s0[2]))  # candidate 1 materialized
        # a stock-layout decode now (eager arm) reads column a-1 like the stock kernel
        st["failed"] = True
        md.num_spec_decodes = 2
        md.spec_state_indices_tensor = torch.arange(1, 9, dtype=torch.int32, device=self.dev).view(2, 4)
        self.assertTrue(gdl.gdl_decode(layer, mixed, a, b, gate, out, md))  # used -> eager rows
        self.assertEqual(int(hdr.abs().sum()), 0)


class RecipeWiring(unittest.TestCase):
    """GDN_LAZY (recipe.yaml -> run.sh): 1 swaps gdn_attn.py for gdn_lazy_attn.py, adds
    gdn_lazy_linear_attn.py and VLLM_QWEN38_GDN_LAZY=1 on the v0.30 digest; 0 is the stock list."""

    V030_IMAGE = "vllm/vllm-openai:v0.30.0-aarch64@" + V030_DIGEST
    PIN_IMAGE = ("vllm/vllm-openai:nightly-aarch64@"
                 "sha256:df871f170ee7070fbdce162bde08fb616e311570c948a620be0d4b33fe02f87b")

    def run_sh(self, **extra):
        import subprocess
        env = {k: v for k, v in os.environ.items()
               if k not in ("OVERLAYS", "IMAGE", "EXTRA_ENV", "EXTRA_ARGS", "DIAGNOSTIC", "GDN_LAZY", "ROLE")}
        env.update(extra, VALIDATE_ONLY="1")
        return subprocess.run([str(ROOT / "run.sh")], capture_output=True, text=True, cwd=str(ROOT), env=env)

    def overlays(self, out):
        return [l.split()[2] for l in out.splitlines() if l.startswith("==> overlay /")]

    def test_default_is_k3(self):
        self.assertIn('GDN_LAZY="${GDN_LAZY:-1}"', (ROOT / "run.sh").read_text())
        self.assertIn("GDN_LAZY: 1", (ROOT / "recipe.yaml").read_text())
        p = self.run_sh(IMAGE=self.V030_IMAGE)
        self.assertEqual(p.returncode, 0, p.stderr)
        ov = self.overlays(p.stdout)
        self.assertIn(str(ATT), ov)
        self.assertIn(str(LIN), ov)
        self.assertNotIn(str(S15), ov)
        self.assertIn("VLLM_QWEN38_GDN_LAZY=1", p.stdout)

    def test_off_is_stock_list(self):
        p = self.run_sh(IMAGE=self.V030_IMAGE, GDN_LAZY="0")
        self.assertEqual(p.returncode, 0, p.stderr)
        ov = self.overlays(p.stdout)
        self.assertIn(str(S15), ov)
        self.assertNotIn(str(ATT), ov)
        self.assertNotIn(str(LIN), ov)
        self.assertNotIn("VLLM_QWEN38_GDN_LAZY", p.stdout)

    def test_guards(self):
        self.assertNotEqual(self.run_sh(GDN_LAZY="2").returncode, 0)
        # the pin rollback ignores it (overlays are generated against v0.30)
        p = self.run_sh(IMAGE=self.PIN_IMAGE)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("GDN_LAZY=1 ignored", p.stderr)
        self.assertNotIn("VLLM_QWEN38_GDN_LAZY", p.stdout)
        # no overlays / a custom list without the GDN overlay: nothing mounted, env not set
        for extra in ({"OVERLAYS": "none"}, {"OVERLAYS": "docker/v030/serving.py"}):
            p = self.run_sh(IMAGE=self.V030_IMAGE, **extra)
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertNotIn("VLLM_QWEN38_GDN_LAZY", p.stdout)
        # the worker gets the head's resolved list and must not rewrite it
        p = self.run_sh(IMAGE=self.V030_IMAGE, ROLE="worker",
                        OVERLAYS="docker/v030/gdn_lazy_attn.py docker/v030/gdn_lazy_linear_attn.py")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(self.overlays(p.stdout), [str(ATT), str(LIN)])
        self.assertIn("VLLM_QWEN38_GDN_LAZY=1", p.stdout)
        self.assertRegex((ROOT / "run.sh").read_text(), r"FORWARD_VARS=\([^)]*\bGDN_LAZY\b")


if __name__ == "__main__":
    unittest.main()
