#!/usr/bin/env python3
"""Patch v0.30.0 distributed/device_communicators/cuda_communicator.py: L2a eager-twin NCCL router.

Two hunks, nothing else changes:
  1. CudaCommunicator.__init__, right after the stock PyNcclCommunicator is built (and after the
     symmetric-memory registration, which install() refuses anyway):
         self.pynccl_comm = _nccl_twin.attach(self, tcp_store_group, PyNcclCommunicator)
     attach() returns the stock communicator unchanged unless every rank armed.
  2. End of the module: docker/v030/nccl_twin.py embedded verbatim as a string, executed into the
     module `vllm_qwen38_nccl_twin`, then install(logger, comm_cls=CudaCommunicator). install() arms
     only with VLLM_QWEN38_NCCL_TWIN=1 and NCCL 2.30.7, writing NCCL_GRAPH_MIXING_SUPPORT=0 before
     the first communicator of the process exists.

Output: docker/v030/nccl_twin_cuda_communicator.py with an R11 header. Mount it read-only at
IN_IMAGE on BOTH ranks, v0.30.0 only (the generator refuses any other stock sha256).
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE_IMAGE = (
    "vllm/vllm-openai:v0.30.0-aarch64@"
    "sha256:4864d46625cbc3307623e29ac742030655e27249feba7b97ec925ce4cc4dfb56"
)
BASE_IMAGE_DIGEST = "sha256:4864d46625cbc3307623e29ac742030655e27249feba7b97ec925ce4cc4dfb56"
UPSTREAM_FILE = "vllm/distributed/device_communicators/cuda_communicator.py"
IN_IMAGE = "/usr/local/lib/python3.12/dist-packages/" + UPSTREAM_FILE
# sha256 of the stock file in the v0.30.0 image (checked with docker run, no GPU).
UPSTREAM_FILE_SHA256 = "102d95c67ec798cf352b817ba92e8abc9da37f30d1bbdf3aab45d3927b627b93"
UPSTREAM_PR = (
    "none (local; port of the DeepSeek-V4.1 sibling's docker/patch/nccl_eager_twin.py, "
    "DSV41_NCCL_EAGER_TWIN)"
)
MODULE = HERE / "nccl_twin.py"
OUT = HERE / "nccl_twin_cuda_communicator.py"

HEADER = f"""\
# R11-OVERLAY
# base_image_digest: {BASE_IMAGE_DIGEST}
# upstream_file: {UPSTREAM_FILE}
# upstream_file_sha256: {UPSTREAM_FILE_SHA256}
# upstream_PR: {UPSTREAM_PR}
# generator: docker/v030/apply_nccl_twin_overlay.py (do not hand-edit)
# features: L2a NCCL_GRAPH_MIXING_SUPPORT=0 + eager-only twin TP communicator,
#   opt-in via VLLM_QWEN38_NCCL_TWIN=1 (default off; source docker/v030/nccl_twin.py)
"""

INIT_OLD = """\
            if is_symmetric_memory_enabled():
                register_nccl_symmetric_ops(self.pynccl_comm)
"""
INIT_NEW = """\
            if is_symmetric_memory_enabled():
                register_nccl_symmetric_ops(self.pynccl_comm)
            # L2a (recipe overlay): eager-twin router; the stock comm unless every rank armed
            # VLLM_QWEN38_NCCL_TWIN=1 (see _nccl_twin below).
            self.pynccl_comm = _nccl_twin.attach(self, tcp_store_group, PyNcclCommunicator)
"""

TAIL_BEGIN = "\n\n# ---- L2a (recipe overlay): docker/v030/nccl_twin.py, embedded verbatim ----\n"
TAIL_FMT = '''\
def _load_nccl_twin():
    import sys
    import types

    name = "vllm_qwen38_nccl_twin"
    mod = sys.modules.get(name)
    if mod is None:
        mod = types.ModuleType(name)
        exec(compile(_NCCL_TWIN_SRC, "<docker/v030/nccl_twin.py>", "exec"), mod.__dict__)
        sys.modules[name] = mod
    return mod


_NCCL_TWIN_SRC = r\'\'\'{src}\'\'\'
_nccl_twin = _load_nccl_twin()
_nccl_twin.install(logger, comm_cls=CudaCommunicator)
'''


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def tail(src: str) -> str:
    if "'''" in src or src.endswith("\\"):
        raise SystemExit("apply_nccl_twin_overlay: nccl_twin.py cannot be embedded in r'''...'''")
    return TAIL_BEGIN + TAIL_FMT.format(src=src)


def overlay(stock: str, src: str) -> str:
    got = sha256(stock)
    if got != UPSTREAM_FILE_SHA256:
        raise SystemExit(
            f"apply_nccl_twin_overlay: stock {UPSTREAM_FILE} sha256 {got} != {UPSTREAM_FILE_SHA256}; "
            "rebase the L2a hunks"
        )
    if stock.count(INIT_OLD) != 1:
        raise SystemExit(f"apply_nccl_twin_overlay: anchor not unique/found: {INIT_OLD[:60]!r}")
    return HEADER + stock.replace(INIT_OLD, INIT_NEW, 1) + tail(src)


def unoverlay(text: str, src: str) -> str:
    """Invert overlay(); tests use it to prove the diff is only these hunks."""
    if not text.startswith(HEADER):
        raise ValueError("R11 header missing or stale")
    text = text[len(HEADER):]
    t = tail(src)
    if not text.endswith(t):
        raise ValueError("embedded nccl_twin.py missing or stale")
    text = text[: -len(t)]
    if text.count(INIT_NEW) != 1:
        raise ValueError("__init__ hook missing")
    return text.replace(INIT_NEW, INIT_OLD, 1)


def extract(image: str, dest: Path) -> None:
    cid = subprocess.check_output(["docker", "create", image], text=True).strip()
    try:
        subprocess.check_call(["docker", "cp", f"{cid}:{IN_IMAGE}", str(dest)])
    finally:
        subprocess.check_call(["docker", "rm", cid], stdout=subprocess.DEVNULL)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--image", default=BASE_IMAGE)
    ap.add_argument("--src", type=Path, help="stock cuda_communicator.py already extracted (skips docker)")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--check", action="store_true", help="exit 1 if --out differs from a fresh render")
    args = ap.parse_args()
    if args.src is not None:
        stock = args.src.read_text()
    else:
        tmp = args.out.with_suffix(".stock.py")
        extract(args.image, tmp)
        stock = tmp.read_text()
        tmp.unlink(missing_ok=True)
    text = overlay(stock, MODULE.read_text())
    if args.check:
        if not args.out.exists() or args.out.read_text() != text:
            print(f"apply_nccl_twin_overlay: {args.out} is stale", file=sys.stderr)
            return 1
        print(f"apply_nccl_twin_overlay: {args.out} up to date")
        return 0
    args.out.write_text(text)
    print(f"apply_nccl_twin_overlay: wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
