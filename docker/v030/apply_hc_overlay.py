#!/usr/bin/env python3
"""Patch v0.30.0 qwen4_exp/nvidia/hyperconnection.py: K5 fused hyper-connection (L4b).

Hook: GatedResidual.mix and GatedResidual.combine_and_mix (the modules own the
weights; ops/hc.py only has weight-free glue, so it stays stock). Each method
first asks hcf_dispatch() for a fused result and falls through to the stock
body when it returns None. The kernels, dispatch and load-time self-test come
from docker/v030/hc_fused.py, embedded verbatim at the end of the overlay, so
the overlay stays one bind-mounted file (OVERLAYS.md mounts only upstream
paths).

Default OFF: fused only with VLLM_QWEN38_HC_FUSED=1 on BOTH ranks, CUDA,
M <= 32, bf16 weights of the expected shapes, and after the per-variant
self-test passed (any failure disables K5 for the process and logs a
warning). Otherwise the stock 5-kernel path runs unchanged.

Output: docker/v030/hyperconnection.py with an R11 header. Mount it read-only
at IN_IMAGE inside the v0.30.0 container on both ranks. Never on the pinned
nightly (the generator refuses any other stock sha).
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
BASE_IMAGE_DIGEST = (
    "sha256:4864d46625cbc3307623e29ac742030655e27249feba7b97ec925ce4cc4dfb56"
)
UPSTREAM_FILE = "vllm/models/qwen4_exp/nvidia/hyperconnection.py"
IN_IMAGE = "/usr/local/lib/python3.12/dist-packages/" + UPSTREAM_FILE
UPSTREAM_FILE_SHA256 = (
    "2a15d6b22fbe1def4d2bcd85568269d1157c5e1c298f0c6af43c1fb9797489ba"
)
UPSTREAM_PR = "none (local K5; fusion target of vLLM issue #54688)"
ENV = "VLLM_QWEN38_HC_FUSED"
KERNELS = HERE / "hc_fused.py"
EMBED_MARK = "# ---- K5 embedded from docker/v030/hc_fused.py (do not hand-edit) ----\n"


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def header(kernels: str) -> str:
    return f"""\
# R11-OVERLAY
# base_image_digest: {BASE_IMAGE_DIGEST}
# upstream_file: {UPSTREAM_FILE}
# upstream_file_sha256: {UPSTREAM_FILE_SHA256}
# upstream_PR: {UPSTREAM_PR}
# generator: docker/v030/apply_hc_overlay.py (do not hand-edit)
# embedded: docker/v030/hc_fused.py sha256 {sha256(kernels)}
# features: K5 fused HC (H1 combine_stats, H2 down_inject_silu, H3 up_gatemix)
#   for M <= 32, opt-in via {ENV}=1, default off, self-tested at load
"""


MIX_OLD = """\
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        xn = grouped_gemma_rmsnorm(
"""
MIX_NEW = """\
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        _k5 = hcf_dispatch(self, hidden_states, None, None)
        if _k5 is not None:
            return _k5
        xn = grouped_gemma_rmsnorm(
"""
CM_OLD = """\
        to every stream with unit weight.
        \"\"\"
        hidden_states, xn = hc_combine_norm(
"""
CM_NEW = """\
        to every stream with unit weight.
        \"\"\"
        if prev_block_output is not None:
            _k5 = hcf_dispatch(self, hidden_states, prev_block_output, prev_injection)
            if _k5 is not None:
                return _k5
        hidden_states, xn = hc_combine_norm(
"""
HUNKS = ((MIX_OLD, MIX_NEW), (CM_OLD, CM_NEW))


def overlay(stock: str, kernels: str) -> str:
    got = sha256(stock)
    if got != UPSTREAM_FILE_SHA256:
        raise SystemExit(
            f"apply_hc_overlay: stock {UPSTREAM_FILE} sha256 {got} "
            f"!= {UPSTREAM_FILE_SHA256}; rebase the K5 hunks"
        )
    if "from __future__" in kernels:
        raise SystemExit("apply_hc_overlay: hc_fused.py must not use __future__ imports")
    text = stock
    for old, new in HUNKS:
        if text.count(old) != 1:
            raise SystemExit(f"apply_hc_overlay: anchor not unique/found: {old[:60]!r}")
        text = text.replace(old, new, 1)
    return header(kernels) + text + "\n\n" + EMBED_MARK + kernels


def unoverlay(text: str) -> tuple[str, str]:
    """Invert overlay() -> (stock, kernels); tests use it to prove the diff."""
    if not text.startswith("# R11-OVERLAY\n"):
        raise ValueError("R11 header missing")
    body, sep, kernels = text.partition("\n\n" + EMBED_MARK)
    if not sep:
        raise ValueError("embedded K5 block missing")
    if not body.startswith(header(kernels)):
        raise ValueError("R11 header stale")
    body = body[len(header(kernels)) :]
    for old, new in reversed(HUNKS):
        if body.count(new) != 1:
            raise ValueError(f"overlay hunk missing: {new[:60]!r}")
        body = body.replace(new, old, 1)
    return body, kernels


def extract(image: str, dest: Path) -> None:
    cid = subprocess.check_output(["docker", "create", image], text=True).strip()
    try:
        subprocess.check_call(["docker", "cp", f"{cid}:{IN_IMAGE}", str(dest)])
    finally:
        subprocess.check_call(["docker", "rm", cid], stdout=subprocess.DEVNULL)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--image", default=BASE_IMAGE)
    ap.add_argument("--src", type=Path, help="stock hyperconnection.py already extracted (skips docker)")
    ap.add_argument("--out", type=Path, default=HERE / "hyperconnection.py")
    ap.add_argument("--check", action="store_true", help="exit 1 if --out differs from a fresh render")
    args = ap.parse_args()
    if args.src is not None:
        stock = args.src.read_text()
    else:
        tmp = args.out.with_suffix(".stock.py")
        extract(args.image, tmp)
        stock = tmp.read_text()
        tmp.unlink(missing_ok=True)
    text = overlay(stock, KERNELS.read_text())
    if args.check:
        if not args.out.exists() or args.out.read_text() != text:
            print(f"apply_hc_overlay: {args.out} is stale", file=sys.stderr)
            return 1
        print(f"apply_hc_overlay: {args.out} up to date")
        return 0
    args.out.write_text(text)
    print(f"apply_hc_overlay: wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
