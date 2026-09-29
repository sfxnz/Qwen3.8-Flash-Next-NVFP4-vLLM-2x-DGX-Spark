#!/usr/bin/env python3
"""Pass use_fused_finalize=False to FlashInfer's CUTLASS fused MoE on v0.30.0 (S1.5 determinism).

Stock v0.30.0 calls flashinfer.fused_moe.cutlass_fused_moe without use_fused_finalize, so
FlashInfer 0.6.18 defaults it to True: the top-k expert reduction is fused into the GEMM2
epilogue with non-associative atomics, and FlashInfer documents the result as "not
deterministic run-to-run". The NVFP4 target MoE (FLASHINFER_CUTLASS) then gives a different
answer for the same request on an idle serve, and NVFP4 activation quantisation amplifies it.
This adds one keyword so FlashInfer takes its non-fused, deterministic finalize kernel.
VLLM_QWEN38_MOE_DETERMINISTIC=0 restores the stock fused (atomic) finalize for A/B; default 1.

Output: docker/v030/flashinfer_cutlass_moe.py with an R11 header (both ranks mount it).
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

BASE_IMAGE = (
    "vllm/vllm-openai:v0.30.0-aarch64@"
    "sha256:4864d46625cbc3307623e29ac742030655e27249feba7b97ec925ce4cc4dfb56"
)
BASE_IMAGE_DIGEST = "sha256:4864d46625cbc3307623e29ac742030655e27249feba7b97ec925ce4cc4dfb56"
UPSTREAM_FILE = "vllm/model_executor/layers/fused_moe/experts/flashinfer_cutlass_moe.py"
IN_IMAGE = "/usr/local/lib/python3.12/dist-packages/" + UPSTREAM_FILE
UPSTREAM_FILE_SHA256 = "52887cec5fe628e0a4e7793deb546b40f6c29f7362b3b436c10ca36934ab9826"
UPSTREAM_PR = "none (local S1.5 fix; FlashInfer cutlass_fused_moe use_fused_finalize doc)"

HEADER = f"""\
# R11-OVERLAY
# base_image_digest: {BASE_IMAGE_DIGEST}
# upstream_file: {UPSTREAM_FILE}
# upstream_file_sha256: {UPSTREAM_FILE_SHA256}
# upstream_PR: {UPSTREAM_PR}
# generator: docker/v030/apply_moe_finalize_overlay.py (do not hand-edit)
"""

OLD = """\
            use_w4_group_scaling=use_w4_group_scaling,
        )
"""
NEW = """\
            use_w4_group_scaling=use_w4_group_scaling,
            # S1.5: the fused finalize epilogue reduces top-k with atomics (non-deterministic).
            use_fused_finalize=not _MOE_DETERMINISTIC,
        )
"""
OLD_IMPORT = """\
import torch
"""
NEW_IMPORT = """\
import os

import torch
"""
OLD_LOGGER = """\
logger = init_logger(__name__)
"""
NEW_LOGGER = """\
logger = init_logger(__name__)

# S1.5: 1 (default) = FlashInfer non-fused deterministic finalize; 0 = stock fused atomic finalize.
_MOE_DETERMINISTIC = os.environ.get("VLLM_QWEN38_MOE_DETERMINISTIC", "1") != "0"
"""
HUNKS = ((OLD_IMPORT, NEW_IMPORT), (OLD_LOGGER, NEW_LOGGER), (OLD, NEW))


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def overlay(stock: str) -> str:
    got = sha256(stock)
    if got != UPSTREAM_FILE_SHA256:
        raise SystemExit(f"apply_moe_finalize_overlay: stock {UPSTREAM_FILE} sha256 {got} != {UPSTREAM_FILE_SHA256}")
    text = stock
    for old, new in HUNKS:
        if text.count(old) != 1:
            raise SystemExit("apply_moe_finalize_overlay: anchor not unique/found")
        text = text.replace(old, new, 1)
    return HEADER + text


def unoverlay(text: str) -> str:
    """Invert overlay(); tests use it to prove the diff is only the one keyword."""
    if not text.startswith(HEADER):
        raise ValueError("R11 header missing or stale")
    text = text[len(HEADER):]
    for old, new in reversed(HUNKS):
        if text.count(new) != 1:
            raise ValueError("overlay hunk missing")
        text = text.replace(new, old, 1)
    return text


def extract(image: str, dest: Path) -> None:
    cid = subprocess.check_output(["docker", "create", image], text=True).strip()
    try:
        subprocess.check_call(["docker", "cp", f"{cid}:{IN_IMAGE}", str(dest)])
    finally:
        subprocess.check_call(["docker", "rm", cid], stdout=subprocess.DEVNULL)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--image", default=BASE_IMAGE)
    ap.add_argument("--src", type=Path, help="stock file already extracted from the image (skips docker)")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "flashinfer_cutlass_moe.py")
    ap.add_argument("--check", action="store_true", help="exit 1 if --out differs from a fresh render")
    args = ap.parse_args()
    if args.src is not None:
        stock = args.src.read_text()
    else:
        tmp = args.out.with_suffix(".stock.py")
        extract(args.image, tmp)
        stock = tmp.read_text()
        tmp.unlink(missing_ok=True)
    text = overlay(stock)
    if args.check:
        if not args.out.exists() or args.out.read_text() != text:
            print(f"apply_moe_finalize_overlay: {args.out} is stale", file=sys.stderr)
            return 1
        print(f"apply_moe_finalize_overlay: {args.out} up to date")
        return 0
    args.out.write_text(text)
    print(f"apply_moe_finalize_overlay: wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
