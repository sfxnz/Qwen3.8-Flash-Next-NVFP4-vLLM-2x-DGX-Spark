"""L2a: NCCL graph mixing off, made safe by an eager-only twin communicator (v0.30.0).

VLLM_QWEN38_NCCL_TWIN=1 (default 0). This file is embedded verbatim in the generated overlay
docker/v030/nccl_twin_cuda_communicator.py (vllm/distributed/device_communicators/cuda_communicator.py),
which calls install() at import and attach() from CudaCommunicator.__init__. Top-level imports are
stdlib only, so the host tests import it without torch. Ported from the DeepSeek-V4.1 sibling's
docker/patch/nccl_eager_twin.py.

Why. With NCCL's default NCCL_GRAPH_MIXING_SUPPORT=1 (graphUsageMode 2) every collective captured
into a CUDA graph carries serialEvent wait/record nodes on NCCL's strong streams, and an eager launch
on a communicator that owns graph plans can queue behind the graph's host nodes (nccl 2.30.7
src/misc/strongstream.cc `mixing = graphUsageMode == 2`, src/enqueue.cc ncclLaunchPrepare).
S1.1 on v0.30.0 (evidence/s11-micro/gonogo.md): in-graph AR 5-160 KB is 14.4-19.3 us cheaper per call
with mixing off; booked c=1 0.89 ms/step, c=8 1.02 ms/step (GO).

Why not the env var alone. With mixing off NCCL does not support a non-captured collective launched
while a graph that uses the same communicator is outstanding (NCCL env docs). The serve launches
eager collectives (embedding all-reduce, logits / draft all-gathers outside the FULL decode graph)
while the target graph is still running (sibling flags.md NCCL_GRAPH_MIXING_SUPPORT=0 row).

What this does (only when armed).
- install() writes NCCL_GRAPH_MIXING_SUPPORT=0 into this process's environment. NCCL reads it once
  per process, at the first ncclCommInitRank (init.cc:2066, param cached), so it must run before any
  NCCL communicator exists: the overlay calls it when cuda_communicator is imported, i.e. before the
  first CudaCommunicator. If a communicator already existed, the write is ignored and mixing stays on
  (safe, no gain); the engagement audit catches that (nccl_twin_audit.py, check nccl-env).
- For the TP group ('tp:*') attach() builds a second PyNcclCommunicator (the eager twin) on the same
  CPU group right after the stock one, and pynccl_comm becomes a GraphEagerRouter: a call whose
  stream is capturing goes to the stock (graph) communicator, every other call goes to the twin. The
  graph communicator only ever runs captured collectives; the twin never runs one.
- Any other group with a PyNccl communicator gets a router without a twin: eager calls pass
  through, a captured call raises, so an unexpected captured use fails the boot at capture.
- Engagement audit lines, per rank: one per captured CUDA graph (torch CUDAGraph.capture_end,
  wrapped only when armed) with the collectives it captured, and one at 1, 10, 100, ... eager calls
  routed to the twin. nccl_twin_audit.py checks them against NCCL's own per-call INFO line.

Fail closed.
- install() disarms (mixing stays on, stock path, nothing patched) unless the lever is on, the NCCL
  that vLLM's PyNccl loads is in VALIDATED_NCCL, NCCL symmetric memory is off and the classes match
  what the router covers (check_anchors). Unless armed, any NCCL_GRAPH_MIXING_SUPPORT other than 1
  (NCCL strtoll-parses "00", " 0", "0x0" as 0 too; a disarmed worker can inherit 0 from an armed
  parent) is reset to 1, NCCL's default (graphUsageMode 2).
- Once armed, mixing is already off in the process, so attach() never falls back to a lone stock
  communicator: ranks that disagree on arming, a comm whose NCCL version differs, a group with more
  than one rank and no working PyNccl, or a failed twin self-test raise REFUSED and stop the boot.
- Both ranks exchange (armed, NCCL version) over the group's CPU group in every attach() call, armed
  or not, so a rank that did not arm cannot leave its peer waiting inside the twin's creation. That
  CPU all-gather at communicator init is the only effect this file has on the stock path.

Numerics: unchanged. The same NCCL algorithm, protocol and channel run on the same buffers; only
which communicator object issues each call changes.
"""

from __future__ import annotations

import inspect
import os

ENV = "VLLM_QWEN38_NCCL_TWIN"
MIXING_ENV = "NCCL_GRAPH_MIXING_SUPPORT"

LOG_ENGAGED = "qwen38: nccl twin engaged"
LOG_GUARD = "qwen38: nccl twin guard"
LOG_GRAPH = "qwen38: nccl twin graph"
LOG_EAGER = "qwen38: nccl twin eager"
LOG_DISARMED = ("qwen38: nccl twin DISARMED", "qwen38: nccl twin REFUSED")
# NCCL 2.30.7's own lines, INFO level (NCCL_DEBUG=INFO, subsystem ENV): the param read, once per
# process (src/misc/param.cc:99), and one per captured collective/p2p API call on a comm whose
# graphUsageMode is 0 (src/plugin/profiler.cc:339-343, 372-376). With NCCL_DEBUG=WARN neither
# prints, which is why the sibling counted 0 of 434.
NCCL_ENV_LINE = "NCCL_GRAPH_MIXING_SUPPORT set by environment to 0"
NCCL_CAPTURE_LINE = "graphUsageMode is set to 0 but the user is capturing graphs"

# NCCL releases whose source was read for this lever (graphUsageMode only selects `mixing`).
# The v0.30.0 image ships nvidia-nccl-cu13 2.30.7; vLLM's NCCLLibrary reports 2.30.7 (23007).
VALIDATED_NCCL = ("2.30.7",)

# PyNcclCommunicator methods that enqueue NCCL work on a stream: routed per call.
ROUTED = (
    "all_reduce",
    "all_gather",
    "all_gatherv",
    "reduce_scatter",
    "reduce_scatterv",
    "reduce",
    "scatter",
    "send",
    "recv",
    "broadcast",
    "batch_isend_irecv",
)
# Lifecycle calls that must reach both communicators.
FAN_OUT = ("destroy", "suspend", "resume")
# ncclGroupStart/End take no communicator: forwarding them to either comm is the same.
PASS = ("group_start", "group_end")
# Window registration binds a buffer to one communicator (NCCL symmetric memory, refused at install).
REFUSED = ("register_comm_window", "register_comm_window_raw", "deregister_comm_window")
# Public PyNcclCommunicator names that need no routing (classmethod constructor).
IGNORED = ("from_unique_id_bytes",)
# The overlay's call in CudaCommunicator.__init__, and the stock lines it must follow.
HOOK = "_nccl_twin.attach(self, tcp_store_group, PyNcclCommunicator)"
INIT_ANCHORS = (
    "self.pynccl_comm = PyNcclCommunicator(",
    "group=self.cpu_group if tcp_store_group is None else tcp_store_group",
    HOOK,
)

# Process-wide state: armed by install(); routers registered by attach().
STATE = {"armed": False, "log": None, "graphs": 0}
ROUTERS: list = []


def lever_on(env) -> bool:
    return (env.get(ENV, "0") or "0").strip() == "1"


def check_nccl_version(version: str) -> None:
    if version not in VALIDATED_NCCL:
        raise RuntimeError(
            f"NCCL {version} is not validated for {MIXING_ENV}=0 (validated: {', '.join(VALIDATED_NCCL)}); "
            "re-read its graphUsageMode handling first"
        )


def group_kind(unique_name: str) -> str:
    """'tp' for vLLM's TP group ('tp:0'), else the name's prefix."""
    return (unique_name or "").split(":", 1)[0]


def stream_positions(pynccl_cls) -> dict:
    """Positional index (self excluded) of each routed method's 'stream' parameter."""
    out = {}
    for name in ROUTED:
        params = list(inspect.signature(getattr(pynccl_cls, name)).parameters)
        if "stream" not in params:
            raise TypeError(f"PyNcclCommunicator.{name} has no stream parameter")
        out[name] = params.index("stream") - 1
    return out


def check_anchors(comm_cls, pynccl_cls) -> None:
    """Raise if CudaCommunicator/PyNcclCommunicator differ from what the router covers."""
    src = inspect.getsource(comm_cls.__init__)
    for anchor in INIT_ANCHORS:
        if anchor not in src:
            raise RuntimeError(f"CudaCommunicator.__init__ anchor missing: {anchor!r}")
    params = inspect.signature(comm_cls.__init__).parameters
    for name in ("cpu_group", "device", "unique_name", "tcp_store_group"):
        if name not in params:
            raise RuntimeError(f"CudaCommunicator.__init__ has no {name!r} parameter")
    public = {
        n
        for klass in pynccl_cls.__mro__[:-1]
        for n, v in vars(klass).items()
        if not n.startswith("_") and (callable(v) or isinstance(v, (classmethod, staticmethod)))
    }
    known = set(ROUTED) | set(FAN_OUT) | set(PASS) | set(REFUSED) | set(IGNORED)
    missing = sorted((set(ROUTED) | set(FAN_OUT) | set(PASS)) - public)
    unknown = sorted(public - known)
    if missing or unknown:
        raise RuntimeError(f"PyNcclCommunicator API changed: missing {missing}, unrouted {unknown}")
    stream_positions(pynccl_cls)


def _arg(args, kwargs, pos: int, name: str):
    if name in kwargs:
        return kwargs[name]
    return args[pos] if len(args) > pos else None


def nccl_calls(method: str, args, kwargs, rank: int) -> int:
    """NCCL API calls (ncclAllReduce, ncclBroadcast, ncclSend, ...) one PyNccl call issues.

    NCCL logs NCCL_CAPTURE_LINE once per captured API call, so the audit compares against this.
    """
    if method in ("all_gatherv", "reduce_scatterv"):
        return len(_arg(args, kwargs, 2, "sizes") or ())
    if method == "batch_isend_irecv":
        return len(_arg(args, kwargs, 0, "p2p_ops") or ())
    if method == "scatter":
        sizes = list(_arg(args, kwargs, 2, "sizes") or ())
        root = _arg(args, kwargs, 3, "root")
        root = 0 if root is None else root
        if rank == root:
            return sum(1 for dst, n in enumerate(sizes) if n and dst != root)
        return 1 if rank < len(sizes) and sizes[rank] > 0 else 0
    return 1


class GraphEagerRouter:
    """Stands in for CudaCommunicator.pynccl_comm.

    graph: the stock communicator, used only for calls whose stream is capturing.
    eager: the twin, used for every other call; None means this group has no twin and a
    captured call raises (guard).
    capturing(stream) -> bool decides per call, on the stream PyNccl will launch on.
    Attribute reads (disabled, world_size, rank, device, nccl, ...) come from graph.
    """

    def __init__(self, graph, eager, capturing, name: str, positions: dict, logger):
        d = self.__dict__
        d["_graph"], d["_eager"], d["_capturing"] = graph, eager, capturing
        d["_name"], d["_positions"], d["_logger"] = name, positions, logger
        d["_rank"] = int(getattr(graph, "rank", 0) or 0)
        d["_n_graph"] = d["_n_graph_nccl"] = d["_n_eager"] = d["_n_eager_nccl"] = 0
        d["_mark"] = d["_mark_nccl"] = 0
        d["_next_milestone"] = 1

    def _pick(self, method: str, args, kwargs):
        stream = kwargs.get("stream")
        pos = self._positions[method]
        if stream is None and len(args) > pos:
            stream = args[pos]
        d = self.__dict__
        calls = nccl_calls(method, args, kwargs, self._rank)
        if self._capturing(stream):
            if self._eager is None:
                raise RuntimeError(
                    f"{LOG_DISARMED[1]}: {method} on group {self._name} was captured into a CUDA graph, but "
                    f"{MIXING_ENV}=0 is only safe for a communicator with an eager twin (tp). Set {ENV}=0."
                )
            d["_n_graph"] += 1
            d["_n_graph_nccl"] += calls
            return self._graph
        d["_n_eager"] += 1
        d["_n_eager_nccl"] += calls
        if self._eager is None:
            return self._graph
        if self._n_eager >= self._next_milestone:
            d["_next_milestone"] *= 10
            self._logger.info(f"{LOG_EAGER} {self._counts()}")
        return self._eager

    def _counts(self) -> str:
        return (
            f"on {self._name} rank={self._rank} eager={self._n_eager} eager_nccl={self._n_eager_nccl} "
            f"total_captured={self._n_graph} total_captured_nccl={self._n_graph_nccl} graphs={STATE['graphs']}"
        )

    def _graph_end(self, index: int) -> None:
        """Called after every CUDA graph capture in this process: log what this group captured."""
        d = self.__dict__
        delta, delta_nccl = self._n_graph - self._mark, self._n_graph_nccl - self._mark_nccl
        d["_mark"], d["_mark_nccl"] = self._n_graph, self._n_graph_nccl
        self._logger.info(f"{LOG_GRAPH} {index} captured=+{delta} captured_nccl=+{delta_nccl} {self._counts()}")

    def __getattr__(self, name):
        if name in REFUSED:
            raise RuntimeError(f"{LOG_DISARMED[1]}: {name} is not routed by {ENV}")
        return getattr(self._graph, name)

    def __setattr__(self, name, value):
        raise AttributeError(f"GraphEagerRouter is read-only ({name})")


def _routed(method: str):
    def call(self, *args, **kwargs):
        return getattr(self._pick(method, args, kwargs), method)(*args, **kwargs)

    call.__name__ = method
    return call


def _fan_out(method: str):
    def call(self, *args, **kwargs):
        out = getattr(self._graph, method)(*args, **kwargs)
        if self._eager is not None:
            getattr(self._eager, method)(*args, **kwargs)
        return out

    call.__name__ = method
    return call


def _passed(method: str):
    def call(self, *args, **kwargs):
        return getattr(self._graph, method)(*args, **kwargs)

    call.__name__ = method
    return call


for _m in ROUTED:
    setattr(GraphEagerRouter, _m, _routed(_m))
for _m in FAN_OUT:
    setattr(GraphEagerRouter, _m, _fan_out(_m))
for _m in PASS:
    setattr(GraphEagerRouter, _m, _passed(_m))
del _m


def stream_is_capturing(stream=None) -> bool:
    """Whether the stream PyNccl will launch on (explicit, else vLLM's current) is capturing.

    Hot path for every eager TP collective: when that stream is already torch's current stream
    (always, for CudaCommunicator's calls) ask directly instead of paying torch.cuda.stream()'s
    two Stream constructions and two set_stream calls.
    """
    import torch
    from vllm.utils.torch_utils import current_stream

    s = current_stream() if stream is None else stream
    if torch._C._cuda_getCurrentStream(s.device_index)[0] == s.stream_id:
        return bool(torch.cuda.is_current_stream_capturing())
    with torch.cuda.stream(s):
        return bool(torch.cuda.is_current_stream_capturing())


def twin_self_test(twin) -> None:
    """Eager all-reduce of (rank + 1) on the twin must give n(n+1)/2 on every rank."""
    import torch

    n = int(twin.world_size)
    x = torch.full((4,), float(twin.rank + 1), dtype=torch.float32, device=twin.device)
    out = twin.all_reduce(x)
    torch.cuda.synchronize(twin.device)
    want = n * (n + 1) / 2
    if out is None or not bool(torch.all(out == want)):
        raise RuntimeError(f"twin self-test: all_reduce gave {None if out is None else out.tolist()}, want {want}")


def _gather_default(cc, tcp_store_group, item):
    """(armed, nccl version) from every rank of the group, over its CPU (gloo) group."""
    if tcp_store_group is not None:
        return list(tcp_store_group.all_gather_obj(item))
    import torch.distributed as dist

    out = [None] * dist.get_world_size(cc.cpu_group)
    dist.all_gather_object(out, item, group=cc.cpu_group)
    return out


def attach(cc, tcp_store_group, pynccl_cls, *, gather=None, capturing=None, self_test=None):
    """Return what CudaCommunicator.pynccl_comm should be (called right after the stock comm).

    Not armed on every rank: the stock communicator, unchanged. Armed: a router (tp) or guard.
    """
    comm = cc.pynccl_comm
    name = getattr(cc, "unique_name", "") or "?"
    armed = bool(STATE["armed"])
    working = comm is not None and not getattr(comm, "disabled", True)
    version = comm.nccl.ncclGetVersion() if working else None
    peers = (gather or _gather_default)(cc, tcp_store_group, (armed, version))
    if not any(p[0] for p in peers):
        return comm
    logger = STATE["log"]
    if not all(p[0] for p in peers):
        raise RuntimeError(
            f"{LOG_DISARMED[1]}: group {name}: {ENV} armed on some ranks only ({peers}); "
            f"{MIXING_ENV}=0 is already set on the armed ones. Arm all ranks or none."
        )
    if not working:
        raise RuntimeError(
            f"{LOG_DISARMED[1]}: group {name} has {cc.world_size} ranks but no working PyNccl communicator, "
            f"so {MIXING_ENV}=0 would reach torch.distributed without a twin. Set {ENV}=0."
        )
    try:
        for p in peers:
            check_nccl_version(p[1])
    except RuntimeError as exc:
        raise RuntimeError(f"{LOG_DISARMED[1]}: group {name}: {exc}. Set {ENV}=0.") from exc
    capturing = capturing or stream_is_capturing
    positions = stream_positions(pynccl_cls)
    rank = f"rank={comm.rank}/{comm.world_size}"
    if group_kind(name) != "tp":
        router = GraphEagerRouter(comm, None, capturing, name, positions, logger)
        logger.info(f"{LOG_GUARD} on {name} {rank}: eager-only, a captured call raises")
        return router
    twin = pynccl_cls(group=cc.cpu_group if tcp_store_group is None else tcp_store_group, device=cc.device)
    if getattr(twin, "disabled", True):
        raise RuntimeError(f"{LOG_DISARMED[1]}: eager twin for {name} came up disabled")
    try:
        if capturing(None):
            raise RuntimeError("the current stream is capturing at communicator init")
        (self_test or twin_self_test)(twin)
    except Exception as exc:  # noqa: BLE001 - mixing is off in this process: stop the boot
        raise RuntimeError(f"{LOG_DISARMED[1]}: group {name}: {exc}. Set {ENV}=0.") from exc
    router = GraphEagerRouter(comm, twin, capturing, name, positions, logger)
    ROUTERS.append(router)
    logger.info(
        f"{LOG_ENGAGED} on {name} {rank} nccl={version} {MIXING_ENV}={os.environ.get(MIXING_ENV)}: "
        "captured collectives -> stock comm, eager -> twin; twin self-test ok"
    )
    return router


def hook_capture_end(graph_cls) -> None:
    """Wrap graph_cls.capture_end so every finished capture reports its collectives per group."""
    orig = graph_cls.capture_end
    if getattr(vars(graph_cls).get("capture_end"), "_qwen38_nccl_twin", False):
        return

    def capture_end(self, *args, **kwargs):
        out = orig(self, *args, **kwargs)
        STATE["graphs"] += 1
        for router in ROUTERS:
            router._graph_end(STATE["graphs"])
        return out

    capture_end._qwen38_nccl_twin = True
    capture_end.__wrapped__ = orig
    graph_cls.capture_end = capture_end


def install(logger, *, env=None, version=None, symm_enabled=None, comm_cls=None, pynccl_cls=None,
            graph_cls=None) -> str:
    """'off' | 'armed' | 'disarmed'. Writes NCCL_GRAPH_MIXING_SUPPORT=0 only when armed.

    version() -> NCCL version string, symm_enabled() -> bool, and the classes default to vLLM's and
    torch's (the overlay passes CudaCommunicator, which is being defined in the same module).
    """
    env = os.environ if env is None else env
    STATE["log"] = logger

    def keep_mixing(why: str) -> None:
        # Not armed: mixing must stay on. NCCL strtoll-parses the value, so anything but "1" may mean 0.
        value = env.get(MIXING_ENV)
        if value is not None and value != "1":
            env[MIXING_ENV] = "1"
            logger.warning(f"{LOG_DISARMED[1]}: {MIXING_ENV}={value} {why} is unsafe here; reset to 1")

    if not lever_on(env):
        keep_mixing(f"without {ENV}=1")
        return "off"
    try:
        if version is None:
            from vllm.distributed.device_communicators.pynccl_wrapper import NCCLLibrary

            version = NCCLLibrary().ncclGetVersion
        check_nccl_version(version())
        if symm_enabled is None:
            from vllm.distributed.device_communicators.pynccl_allocator import is_symmetric_memory_enabled

            symm_enabled = is_symmetric_memory_enabled
        if symm_enabled():
            raise RuntimeError("NCCL symmetric memory (VLLM_USE_NCCL_SYMM_MEM=1) binds windows to the stock comm")
        if pynccl_cls is None:
            from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator as pynccl_cls
        if graph_cls is None:
            import torch

            graph_cls = torch.cuda.graphs.CUDAGraph
        check_anchors(comm_cls, pynccl_cls)
        hook_capture_end(graph_cls)
    except Exception as exc:  # noqa: BLE001 - disarm, never half-apply
        logger.warning(f"{LOG_DISARMED[0]}: {exc!r}; NCCL graph mixing stays on")
        keep_mixing("with the twin disarmed")
        return "disarmed"
    env[MIXING_ENV] = "0"
    STATE["armed"] = True
    logger.info(f"qwen38: nccl twin armed: {MIXING_ENV}=0 written before the first communicator")
    return "armed"
