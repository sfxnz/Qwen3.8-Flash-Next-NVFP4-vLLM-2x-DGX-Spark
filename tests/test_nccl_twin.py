#!/usr/bin/env python3
"""L2a eager-twin NCCL overlay (docker/v030/nccl_twin*.py).

Host: stdlib only. Proves the overlay is exactly stock + our two hunks with nccl_twin.py embedded
verbatim, and drives the router / attach / install / audit logic against fakes, including an
end-to-end run whose log lines go through the audit.
In the v0.30.0 image (CPU only, no --gpus) with the overlay mounted over the stock file, the
TestInImage cases import the real overlay and check it against the real vLLM/torch classes and the
image's NCCL (2.30.7); on the host they skip. Set V030_SRC=<v0.30 vllm package dir> to also check the
generator against the extracted stock file and the PyNccl API via ast.
"""
from __future__ import annotations

import ast
import contextlib
import importlib.util
import io
import os
import py_compile
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V030 = ROOT / "docker" / "v030"
OVERLAY = V030 / "nccl_twin_cuda_communicator.py"
V030_DIGEST = "sha256:4864d46625cbc3307623e29ac742030655e27249feba7b97ec925ce4cc4dfb56"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


gen = _load(V030 / "apply_nccl_twin_overlay.py", "apply_nccl_twin_overlay")
aud = _load(V030 / "nccl_twin_audit.py", "nccl_twin_audit")


def fresh_twin():
    """A new module instance, so process-wide STATE/ROUTERS start empty in every test."""
    return _load(V030 / "nccl_twin.py", "nccl_twin_under_test")


class Log:
    def __init__(self):
        self.lines = []

    def info(self, msg):
        self.lines.append(msg)

    warning = info


class FakePyNccl:
    """Same public API and stream positions as v0.30 PyNcclCommunicator; records calls."""

    made = []

    def __init__(self, group=None, device=None, rank=0, world_size=2, version="2.30.7"):
        self.group, self.device, self.rank, self.world_size = group, device, rank, world_size
        self.disabled = False
        self.calls = []
        self.nccl = type("L", (), {"ncclGetVersion": staticmethod(lambda: version)})()
        FakePyNccl.made.append(self)

    def _rec(self, name, *a, **k):
        self.calls.append(name)
        return name

    def all_reduce(self, in_tensor, out_tensor=None, op=None, stream=None): return self._rec("all_reduce")
    def all_gather(self, output_tensor, input_tensor, stream=None): return self._rec("all_gather")
    def all_gatherv(self, output_tensor, input_tensor, sizes, stream=None): return self._rec("all_gatherv")
    def reduce_scatter(self, output_tensor, input_tensor, op=None, stream=None): return self._rec("reduce_scatter")
    def reduce_scatterv(self, output_tensor, input_tensor, sizes, op=None, stream=None): return self._rec("reduce_scatterv")
    def reduce(self, output_tensor, input_tensor, op=None, root=0, stream=None): return self._rec("reduce")
    def scatter(self, output_tensor, input_tensor, sizes, root=0, stream=None): return self._rec("scatter")
    def send(self, tensor, dst, stream=None): return self._rec("send")
    def recv(self, tensor, src, stream=None): return self._rec("recv")
    def broadcast(self, tensor, src, stream=None): return self._rec("broadcast")
    def batch_isend_irecv(self, p2p_ops, stream=None): return self._rec("batch_isend_irecv")
    def group_start(self): return self._rec("group_start")
    def group_end(self): return self._rec("group_end")
    def destroy(self): return self._rec("destroy")
    def suspend(self): return self._rec("suspend")
    def resume(self): return self._rec("resume")
    def register_comm_window(self, tensor): return self._rec("register_comm_window")
    def register_comm_window_raw(self, ptr, size): return self._rec("register_comm_window_raw")
    def deregister_comm_window(self, window): return self._rec("deregister_comm_window")

    @classmethod
    def from_unique_id_bytes(cls, unique_id_bytes, rank, world_size, device, library_path=None):
        raise NotImplementedError


class FakeCudaCommunicator:
    def __init__(self, cpu_group=None, device=None, device_group=None, unique_name="", global_ranks=None,
                 global_world_size=None, tcp_store_group=None, use_all2all=False):
        # Mirrors the overlay's lines that check_anchors looks for:
        # self.pynccl_comm = PyNcclCommunicator(
        #     group=self.cpu_group if tcp_store_group is None else tcp_store_group,
        # self.pynccl_comm = _nccl_twin.attach(self, tcp_store_group, PyNcclCommunicator)
        self.cpu_group, self.device, self.unique_name = cpu_group, device, unique_name
        self.world_size = 2
        self.pynccl_comm = FakePyNccl(cpu_group, device)


class NoHook:
    def __init__(self, cpu_group=None, device=None, unique_name="", tcp_store_group=None):
        self.pynccl_comm = None


class FakeGraph:
    def capture_end(self):
        return "ended"


def capturing_flag():
    state = {"on": False}
    return state, (lambda stream=None: state["on"])


def agree(*flags, versions=None):
    """gather() stand-in: every rank reports (flag, version)."""
    versions = versions or ["2.30.7"] * len(flags)
    return lambda cc, tcp, item: [(f, v) for f, v in zip(flags, versions)]


def armed_twin(**kw):
    t = fresh_twin()
    log = Log()
    env = {t.ENV: "1"}
    state = t.install(log, env=env, version=lambda: "2.30.7", symm_enabled=lambda: False,
                      comm_cls=FakeCudaCommunicator, pynccl_cls=FakePyNccl,
                      graph_cls=kw.get("graph_cls") or type("G", (FakeGraph,), {}))
    assert state == "armed", log.lines
    return t, log, env


class TestGenerator(unittest.TestCase):
    def setUp(self):
        self.text = OVERLAY.read_text()

    def test_r11_header(self):
        head = dict(line[2:].split(": ", 1) for line in self.text.splitlines()[1:6] if ": " in line)
        self.assertTrue(self.text.startswith("# R11-OVERLAY\n"))
        self.assertEqual(head["base_image_digest"], V030_DIGEST)
        self.assertEqual(head["upstream_file"], "vllm/distributed/device_communicators/cuda_communicator.py")
        self.assertEqual(head["upstream_file_sha256"], gen.UPSTREAM_FILE_SHA256)
        self.assertIn("upstream_PR", head)
        self.assertEqual(head["generator"], "docker/v030/apply_nccl_twin_overlay.py (do not hand-edit)")
        self.assertTrue(gen.IN_IMAGE.endswith("/dist-packages/" + head["upstream_file"]))

    def test_diff_is_only_our_hunks_and_module_is_verbatim(self):
        src = (V030 / "nccl_twin.py").read_text()
        stock = gen.unoverlay(self.text, src)
        self.assertEqual(gen.sha256(stock), gen.UPSTREAM_FILE_SHA256)
        self.assertEqual(gen.overlay(stock, src), self.text)
        with self.assertRaises(SystemExit):
            gen.overlay(stock + "\n", src)
        with self.assertRaises(ValueError):  # a stale overlay after editing nccl_twin.py
            gen.unoverlay(self.text, src + "# edit\n")

    def test_compiles_and_hook_placement(self):
        with tempfile.TemporaryDirectory() as d:
            py_compile.compile(str(OVERLAY), cfile=os.path.join(d, "o.pyc"), doraise=True)
        tree = ast.parse(self.text)
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "CudaCommunicator")
        init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
        body = ast.unparse(init)
        # The hook sits inside `if self.world_size > 1:` right after the stock comm is built.
        guard = next(s for s in init.body if isinstance(s, ast.If) and "self.world_size > 1" in ast.unparse(s.test))
        self.assertEqual(ast.unparse(guard.body[0]).split("(")[0], "self.pynccl_comm = PyNcclCommunicator")
        self.assertIn(fresh_twin().HOOK, ast.unparse(guard.body[-1]))
        self.assertEqual(body.count("_nccl_twin."), 1)
        # install() runs at import, after the class exists.
        last = ast.unparse(tree.body[-1])
        self.assertEqual(last, "_nccl_twin.install(logger, comm_cls=CudaCommunicator)")

    def test_generator_check_against_extracted_stock(self):
        root = os.environ.get("V030_SRC")
        if not root:
            self.skipTest("set V030_SRC=<v0.30 vllm package dir>")
        src = Path(root) / "distributed/device_communicators/cuda_communicator.py"
        out = subprocess.run([sys.executable, str(V030 / "apply_nccl_twin_overlay.py"), "--src", str(src), "--check"],
                             capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)

    def test_v030_pynccl_api_via_ast(self):
        root = os.environ.get("V030_SRC")
        if not root:
            self.skipTest("set V030_SRC=<v0.30 vllm package dir>")
        t = fresh_twin()
        tree = ast.parse((Path(root) / "distributed/device_communicators/pynccl.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PyNcclCommunicator")
        fns = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
        public = {n for n in fns if not n.startswith("_")}
        self.assertEqual(public, set(t.ROUTED) | set(t.FAN_OUT) | set(t.PASS) | set(t.REFUSED) | set(t.IGNORED))
        for name in t.ROUTED:
            self.assertIn("stream", [a.arg for a in fns[name].args.args], name)


class TestRouter(unittest.TestCase):
    def make(self, twin=True):
        t = fresh_twin()
        state, cap = capturing_flag()
        graph, eager = FakePyNccl(rank=1), (FakePyNccl(rank=1) if twin else None)
        log = Log()
        r = t.GraphEagerRouter(graph, eager, cap, "tp:0", t.stream_positions(FakePyNccl), log)
        return t, r, graph, eager, state, log

    def test_routes_by_capture(self):
        t, r, graph, eager, state, log = self.make()
        r.all_reduce("x")
        r.all_gather("o", "i", stream="s")
        state["on"] = True
        r.all_reduce("x")
        r.all_gatherv("o", "i", [1, 2])
        state["on"] = False
        r.reduce_scatterv("o", "i", sizes=[3, 3, 3])
        self.assertEqual(graph.calls, ["all_reduce", "all_gatherv"])
        self.assertEqual(eager.calls, ["all_reduce", "all_gather", "reduce_scatterv"])
        self.assertEqual((r._n_graph, r._n_graph_nccl, r._n_eager, r._n_eager_nccl), (2, 3, 3, 5))

    def test_stream_argument_is_what_decides(self):
        t = fresh_twin()
        seen = []
        graph, eager = FakePyNccl(), FakePyNccl()
        r = t.GraphEagerRouter(graph, eager, lambda s: seen.append(s) or s == "cap", "tp:0",
                               t.stream_positions(FakePyNccl), Log())
        r.send("t", 1, "cap")          # positional stream
        r.recv("t", 0, stream="eag")   # keyword stream
        r.broadcast("t", 0)            # none: vLLM's current stream
        self.assertEqual(seen, ["cap", "eag", None])
        self.assertEqual((graph.calls, eager.calls), (["send"], ["recv", "broadcast"]))

    def test_fan_out_pass_refuse_readonly_attrs(self):
        t, r, graph, eager, state, log = self.make()
        r.destroy(); r.suspend(); r.resume()
        r.group_start(); r.group_end()
        self.assertEqual(graph.calls, ["destroy", "suspend", "resume", "group_start", "group_end"])
        self.assertEqual(eager.calls, ["destroy", "suspend", "resume"])
        for name in t.REFUSED:
            with self.assertRaisesRegex(RuntimeError, "REFUSED"):
                getattr(r, name)
        with self.assertRaises(AttributeError):
            r.disabled = True
        self.assertEqual((r.rank, r.world_size, r.disabled), (1, 2, False))

    def test_guard_raises_on_capture(self):
        t, r, graph, eager, state, log = self.make(twin=False)
        r.all_reduce("x")
        self.assertEqual(graph.calls, ["all_reduce"])
        state["on"] = True
        with self.assertRaisesRegex(RuntimeError, "REFUSED: all_reduce on group tp:0 was captured"):
            r.all_reduce("x")

    def test_nccl_call_weights(self):
        t = fresh_twin()
        self.assertEqual(t.nccl_calls("all_reduce", ("x",), {}, 0), 1)
        self.assertEqual(t.nccl_calls("all_gatherv", ("o", "i", [1, 2]), {}, 0), 2)
        self.assertEqual(t.nccl_calls("batch_isend_irecv", ([1, 2, 3],), {}, 0), 3)
        self.assertEqual(t.nccl_calls("scatter", ("o", "i", [2, 0, 3]), {"root": 0}, 0), 1)
        self.assertEqual(t.nccl_calls("scatter", ("o", "i", [2, 0, 3]), {}, 2), 1)
        self.assertEqual(t.nccl_calls("scatter", ("o", "i", [2, 0, 3]), {}, 1), 0)

    def test_eager_milestones(self):
        t, r, graph, eager, state, log = self.make()
        for _ in range(120):
            r.all_reduce("x")
        eager_lines = [line for line in log.lines if line.startswith(t.LOG_EAGER)]
        self.assertEqual(len(eager_lines), 3)  # 1, 10, 100
        self.assertIn("eager=100 ", eager_lines[-1])


class TestInstallAttach(unittest.TestCase):
    def test_off_and_reset(self):
        t = fresh_twin()
        log, env = Log(), {"NCCL_GRAPH_MIXING_SUPPORT": "0"}
        self.assertEqual(t.install(log, env=env), "off")
        self.assertEqual(env["NCCL_GRAPH_MIXING_SUPPORT"], "1")
        self.assertIn("REFUSED", log.lines[0])
        self.assertFalse(t.STATE["armed"])

    def test_off_resets_any_value_nccl_may_parse_as_zero(self):
        for value in ("00", " 0", "0x0", "-0", "2"):
            t = fresh_twin()
            env = {"NCCL_GRAPH_MIXING_SUPPORT": value}
            self.assertEqual(t.install(Log(), env=env), "off")
            self.assertEqual(env["NCCL_GRAPH_MIXING_SUPPORT"], "1", repr(value))
        env = {"NCCL_GRAPH_MIXING_SUPPORT": "1"}
        fresh_twin().install(Log(), env=env)
        self.assertEqual(env, {"NCCL_GRAPH_MIXING_SUPPORT": "1"})

    def test_disarm_resets_inherited_zero(self):
        # An armed parent (engine core) writes 0 into the environment its workers inherit; a worker
        # that then disarms must not keep mixing off without a twin.
        t = fresh_twin()
        log, env = Log(), {t.ENV: "1", "NCCL_GRAPH_MIXING_SUPPORT": "0"}
        self.assertEqual(t.install(log, env=env, version=lambda: "2.31.0", symm_enabled=lambda: False,
                                   comm_cls=FakeCudaCommunicator, pynccl_cls=FakePyNccl, graph_cls=FakeGraph),
                         "disarmed")
        self.assertEqual(env["NCCL_GRAPH_MIXING_SUPPORT"], "1")
        self.assertFalse(t.STATE["armed"])
        self.assertTrue(any(line.startswith(t.LOG_DISARMED[0]) for line in log.lines))

    def test_disarms_without_touching_env(self):
        for kw, why in (({"version": lambda: "2.31.0"}, "2.31.0 is not validated"),
                        ({"symm_enabled": lambda: True}, "symmetric memory"),
                        ({"comm_cls": NoHook}, "anchor missing")):
            t = fresh_twin()
            log, env = Log(), {t.ENV: "1"}
            args = dict(version=lambda: "2.30.7", symm_enabled=lambda: False, comm_cls=FakeCudaCommunicator,
                        pynccl_cls=FakePyNccl, graph_cls=FakeGraph)
            args.update(kw)
            self.assertEqual(t.install(log, env=env, **args), "disarmed", why)
            self.assertNotIn("NCCL_GRAPH_MIXING_SUPPORT", env)
            self.assertIn(why, log.lines[-1])
            self.assertTrue(log.lines[-1].startswith(t.LOG_DISARMED[0]))
            self.assertFalse(hasattr(FakeGraph.capture_end, "_qwen38_nccl_twin"))

    def test_unrouted_api_disarms(self):
        class Bigger(FakePyNccl):
            def all_to_all(self, a, stream=None):
                pass

        t = fresh_twin()
        log = Log()
        self.assertEqual(t.install(log, env={t.ENV: "1"}, version=lambda: "2.30.7", symm_enabled=lambda: False,
                                   comm_cls=FakeCudaCommunicator, pynccl_cls=Bigger, graph_cls=FakeGraph), "disarmed")
        self.assertIn("unrouted ['all_to_all']", log.lines[-1])

    def test_not_armed_anywhere_returns_stock(self):
        t = fresh_twin()
        cc = FakeCudaCommunicator(unique_name="tp:0")
        stock = cc.pynccl_comm
        self.assertIs(t.attach(cc, None, FakePyNccl, gather=agree(False, False)), stock)

    def test_armed_tp_gets_twin_and_graph_lines(self):
        class G(FakeGraph):
            pass

        t, log, env = armed_twin(graph_cls=G)
        self.assertEqual(env["NCCL_GRAPH_MIXING_SUPPORT"], "0")
        state, cap = capturing_flag()
        tested = []
        cc = FakeCudaCommunicator(cpu_group="cpu", device="cuda:0", unique_name="tp:0")
        stock = cc.pynccl_comm
        r = t.attach(cc, None, FakePyNccl, gather=agree(True, True), capturing=cap, self_test=tested.append)
        self.assertIs(r._graph, stock)
        self.assertIsNot(r._eager, stock)
        self.assertEqual((r._eager.group, r._eager.device), ("cpu", "cuda:0"))
        self.assertEqual(tested, [r._eager])
        self.assertTrue(any(line.startswith(t.LOG_ENGAGED + " on tp:0 rank=0/2") for line in log.lines))
        state["on"] = True
        r.all_reduce("x"); r.all_reduce("y")
        self.assertEqual(G().capture_end(), "ended")
        G().capture_end()
        graph_lines = [line for line in log.lines if line.startswith(t.LOG_GRAPH)]
        self.assertIn(" 1 captured=+2 captured_nccl=+2 on tp:0", graph_lines[0])
        self.assertIn(" 2 captured=+0 ", graph_lines[1])

    def test_armed_other_group_gets_guard(self):
        t, log, env = armed_twin()
        cc = FakeCudaCommunicator(unique_name="ep:0")
        n = len(FakePyNccl.made)
        r = t.attach(cc, None, FakePyNccl, gather=agree(True, True), capturing=lambda s=None: False)
        self.assertIsNone(r._eager)
        self.assertEqual(len(FakePyNccl.made), n)  # no twin built
        self.assertTrue(log.lines[-1].startswith(t.LOG_GUARD))

    def test_refusals_once_armed(self):
        t, log, env = armed_twin()
        cases = [
            (agree(True, False), None, "armed on some ranks only"),
            (agree(True, True, versions=["2.30.7", "2.31.0"]), None, "2.31.0 is not validated"),
        ]
        for gather, mutate, why in cases:
            cc = FakeCudaCommunicator(unique_name="tp:0")
            with self.assertRaisesRegex(RuntimeError, "REFUSED.*" + why):
                t.attach(cc, None, FakePyNccl, gather=gather, capturing=lambda s=None: False, self_test=lambda x: None)
        cc = FakeCudaCommunicator(unique_name="tp:0")
        cc.pynccl_comm.disabled = True
        with self.assertRaisesRegex(RuntimeError, "REFUSED.*no working PyNccl"):
            t.attach(cc, None, FakePyNccl, gather=agree(True, True), capturing=lambda s=None: False)

        def bad(twin):
            raise RuntimeError("all_reduce gave [0.0]")

        with self.assertRaisesRegex(RuntimeError, "REFUSED: group tp:0: all_reduce gave"):
            t.attach(FakeCudaCommunicator(unique_name="tp:0"), None, FakePyNccl, gather=agree(True, True),
                     capturing=lambda s=None: False, self_test=bad)

    def test_agreement_runs_when_unarmed_too(self):
        t = fresh_twin()  # never installed: STATE armed False
        seen = []
        cc = FakeCudaCommunicator(unique_name="tp:0")
        t.attach(cc, None, FakePyNccl, gather=lambda cc, tcp, item: seen.append(item) or [item, item])
        self.assertEqual(seen, [(False, "2.30.7")])


def serve_logs(t, graph_cls, per_graph=(3, 3, 1), eager_calls=12, nccl_lines=None, env_line=True, rank=0,
               warm=True):
    """Drive a router like a boot + a few steps and return the rank's log text."""
    log = Log()
    t.install(log, env={t.ENV: "1"}, version=lambda: "2.30.7", symm_enabled=lambda: False,
              comm_cls=FakeCudaCommunicator, pynccl_cls=FakePyNccl, graph_cls=graph_cls)
    state, cap = capturing_flag()
    cc = FakeCudaCommunicator(unique_name="tp:0")
    r = t.attach(cc, None, FakePyNccl, gather=agree(True, True), capturing=cap, self_test=lambda x: None)
    lines = []
    if env_line:
        lines.append(f"spark:1:1 [{rank}] NCCL INFO {t.NCCL_ENV_LINE}.")
    captured = 0
    for n in per_graph:
        if warm:
            r.all_reduce("warm")  # eager dummy run between captures
        state["on"] = True
        for _ in range(n):
            r.all_reduce("x")
        captured += n
        state["on"] = False
        graph_cls().capture_end()
    for _ in range(eager_calls):
        r.all_gather("o", "i")
    for _ in range(captured if nccl_lines is None else nccl_lines):
        lines.append(f"spark:1:1 [{rank}] NCCL INFO Comm config {t.NCCL_CAPTURE_LINE} on the stream. Violating ...")
    return "\n".join(f"(Worker_TP{rank} pid=9) INFO {m}" for m in log.lines) + "\n" + "\n".join(lines)


class TestAudit(unittest.TestCase):
    def logs(self, per_rank=None):
        out = {}
        for rank in (0, 1):
            class G(FakeGraph):
                pass
            out[f"r{rank}"] = serve_logs(fresh_twin(), G, rank=rank, **(per_rank or {}).get(rank, {}))
        return out

    def test_markers_match_module(self):
        m = aud.markers()
        t = fresh_twin()
        self.assertEqual(m["LOG_ENGAGED"], t.LOG_ENGAGED)
        self.assertEqual(m["NCCL_CAPTURE_LINE"], t.NCCL_CAPTURE_LINE)

    def test_clean_boot_passes(self):
        problems, parsed = aud.audit(self.logs(), graphs=3, expect_captured=7)
        self.assertEqual(problems, [])
        self.assertEqual([g[1] for g in parsed["r0"]["graphs"]], [3, 3, 1])
        self.assertEqual((parsed["r0"]["total"], parsed["r0"]["nccl_capture"]), (7, 7))
        self.assertEqual(parsed["r0"]["eager"], 10)  # last milestone line (1, 10)
        self.assertIn("collectives per graph: 1 x1, 3 x2", aud.summary("r0", parsed["r0"]))

    def test_sibling_zero_of_n(self):
        problems, _ = aud.audit(self.logs({0: {"nccl_lines": 0, "env_line": False}}))
        self.assertTrue(any("r0: nccl-env" in p for p in problems))
        self.assertTrue(any("r0: nccl-lines: 0 NCCL capture lines vs 7" in p for p in problems))
        self.assertFalse(any(p.startswith("r1") for p in problems))

    def test_extra_nccl_lines_and_expectations(self):
        problems, _ = aud.audit(self.logs({1: {"nccl_lines": 9}}), graphs=4, expect_captured=8)
        self.assertTrue(any("r1: nccl-lines: 9" in p for p in problems))
        self.assertTrue(any("r0: graphs: 3 graph lines, want 4" in p for p in problems))
        self.assertTrue(any("want 8" in p for p in problems))

    def test_rank_mismatch_disarm_and_no_eager(self):
        logs = self.logs({1: {"per_graph": (3, 2, 1), "eager_calls": 0, "warm": False}})
        logs["r0"] += "\n(Worker_TP0) WARNING qwen38: nccl twin DISARMED: boom"
        problems, _ = aud.audit(logs)
        self.assertTrue(any(p.startswith("ranks: per-graph captured counts differ") for p in problems))
        self.assertTrue(any("r0: " in p and "DISARMED: boom" in p for p in problems))
        self.assertTrue(any("r1: eager" in p for p in problems))

    def test_cli_exit_codes(self):
        with tempfile.TemporaryDirectory() as d:
            paths = []
            for rank, text in self.logs().items():
                p = Path(d) / f"{rank}.log"
                p.write_text(text)
                paths.append(f"{rank}={p}")
            sink = io.StringIO()
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                self.assertEqual(aud.main(["--graphs", "3", *paths]), 0)
                self.assertEqual(aud.main(["--graphs", "4", *paths]), 1)
                self.assertEqual(aud.main(["--mode", "warn", "--graphs", "4", *paths]), 0)
            self.assertIn("==> nccl twin audit ok on r0, r1", sink.getvalue())


def _in_image():
    try:
        import torch  # noqa: F401
        from vllm.distributed.device_communicators import cuda_communicator as cc
    except Exception:  # noqa: BLE001
        return None
    return cc if hasattr(cc, "_nccl_twin") else None


class TestInImage(unittest.TestCase):
    """Run inside the v0.30.0 image with the overlay mounted over the stock file (CPU only)."""

    def setUp(self):
        self.cc = _in_image()
        if self.cc is None:
            self.skipTest("needs the v0.30.0 image with nccl_twin_cuda_communicator.py mounted")

    def run_child(self, env_extra):
        code = (
            "import os\n"
            "from vllm.distributed.device_communicators import cuda_communicator as cc\n"
            "import torch\n"
            "t = cc._nccl_twin\n"
            "print('STATE', t.STATE['armed'], os.environ.get('NCCL_GRAPH_MIXING_SUPPORT'),"
            " getattr(torch.cuda.graphs.CUDAGraph.capture_end, '_qwen38_nccl_twin', False))\n"
        )
        env = {k: v for k, v in os.environ.items() if not k.startswith(("VLLM_QWEN38", "NCCL_GRAPH"))}
        env.update(env_extra)
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=300)
        state = [line for line in out.stdout.splitlines() if line.startswith("STATE")]
        self.assertTrue(state, out.stdout + out.stderr)
        return state[0], out.stdout + out.stderr

    def test_real_classes_and_nccl(self):
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
        from vllm.distributed.device_communicators.pynccl_wrapper import NCCLLibrary

        t = self.cc._nccl_twin
        t.check_anchors(self.cc.CudaCommunicator, PyNcclCommunicator)
        self.assertEqual(NCCLLibrary().ncclGetVersion(), "2.30.7")
        self.assertIn(NCCLLibrary().ncclGetVersion(), t.VALIDATED_NCCL)
        pos = t.stream_positions(PyNcclCommunicator)
        self.assertEqual((pos["all_reduce"], pos["all_gatherv"], pos["batch_isend_irecv"]), (3, 3, 1))

    def test_off_by_default(self):
        state, out = self.run_child({})
        self.assertEqual(state, "STATE False None False", out)

    def test_off_resets_bare_env(self):
        state, out = self.run_child({"NCCL_GRAPH_MIXING_SUPPORT": "0"})
        self.assertEqual(state, "STATE False 1 False", out)
        self.assertIn("qwen38: nccl twin REFUSED: NCCL_GRAPH_MIXING_SUPPORT=0 without", out)

    def test_arms_on_real_image(self):
        state, out = self.run_child({"VLLM_QWEN38_NCCL_TWIN": "1"})
        self.assertEqual(state, "STATE True 0 True", out)
        self.assertIn("qwen38: nccl twin armed", out)

    def test_default_path_exchange_over_real_gloo(self):
        """The one stock-path change: attach()'s (armed, version) all-gather over a real gloo group."""
        code = (
            "import os, sys, torch.distributed as dist, torch.multiprocessing as mp\n"
            "from vllm.distributed.device_communicators import cuda_communicator as cc\n"
            "t = cc._nccl_twin\n"
            "class Comm:\n"
            "    disabled = False\n"
            "    nccl = type('L', (), {'ncclGetVersion': staticmethod(lambda: '2.30.7')})()\n"
            "class CC:\n"
            "    unique_name, world_size = 'tp:0', 2\n"
            "def run(rank, port):\n"
            "    dist.init_process_group('gloo', init_method=f'tcp://127.0.0.1:{port}', rank=rank, world_size=2)\n"
            "    c = CC(); c.cpu_group = dist.new_group(backend='gloo'); c.pynccl_comm = Comm()\n"
            "    assert t._gather_default(c, None, (rank == 1, '2.30.%d' % rank)) == [(False, '2.30.0'), (True, '2.30.1')]\n"
            "    assert t.attach(c, None, object) is c.pynccl_comm\n"
            "    print('GLOO_OK', rank, flush=True)\n"
            "    dist.destroy_process_group()\n"
            "if __name__ == '__main__':\n"
            "    mp.spawn(run, args=(29517,), nprocs=2)\n"
        )
        env = {k: v for k, v in os.environ.items() if not k.startswith(("VLLM_QWEN38", "NCCL_GRAPH"))}
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "gloo_twin.py"
            f.write_text(code)
            out = subprocess.run([sys.executable, str(f)], capture_output=True, text=True, env=env, timeout=300)
        text = out.stdout + out.stderr
        self.assertEqual(out.returncode, 0, text)
        self.assertEqual(sorted(line for line in out.stdout.splitlines() if line.startswith("GLOO_OK")),
                         ["GLOO_OK 0", "GLOO_OK 1"], text)

    def test_symm_mem_disarms(self):
        state, out = self.run_child({"VLLM_QWEN38_NCCL_TWIN": "1", "VLLM_USE_NCCL_SYMM_MEM": "1"})
        self.assertEqual(state, "STATE False None False", out)
        self.assertIn("qwen38: nccl twin DISARMED", out)


if __name__ == "__main__":
    unittest.main()
