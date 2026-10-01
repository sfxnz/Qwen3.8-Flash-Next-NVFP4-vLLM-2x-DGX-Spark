#!/usr/bin/env python3
"""Extract stock qwen4_exp ops/ple.py from the pinned image and apply only the vLLM #55375 stride hunks.

The pinned fused PLE short-conv kernels load state indices as `state_idx_ptr + r` and ignore the
stride of a strided block-table column view (mamba 'align' mode, prefix caching on). Requests
prefilled together in a prefill-only step then write their conv state into the wrong slots (F01).
Upstream #55375 passes `state_indices.stride(0)` into both kernels. This copies the pinned file and
adds only that argument. It does not copy the v0.30 file: v0.30 also adds a required
`outer_residual` argument that the pinned ple_layer.py call sites do not pass.
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path

IMAGE = (
    "vllm/vllm-openai:nightly-aarch64@"
    "sha256:df871f170ee7070fbdce162bde08fb616e311570c948a620be0d4b33fe02f87b"
)
IN_IMAGE = "/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia/ops/ple.py"
UPSTREAM_PR = "https://github.com/vllm-project/vllm/pull/55375"
# sha256 of ops/ple.py in the pinned image. The hunks below are written against this file.
STOCK_SHA256 = "cec929bed9c525ce1ecf362966af0da6e382645f5002d878a2f545f40b38e08e"

# (old, new, expected count)
HUNKS = [
    # int state_idx_stride on _ple_conv_kernel and _ple_conv_writeback_kernel (before state_bs, as upstream)
    (
        "    state_bs,\n    state_ws,\n    state_cs,\n    C: tl.constexpr,\n",
        "    state_idx_stride,\n    state_bs,\n    state_ws,\n    state_cs,\n    C: tl.constexpr,\n",
        2,
    ),
    # sid = tl.load(state_idx_ptr + r*state_idx_stride)
    (
        "    sid = tl.load(state_idx_ptr + r).to(tl.int64)\n",
        "    sid = tl.load(state_idx_ptr + r * state_idx_stride).to(tl.int64)\n",
        2,
    ),
    # the ple_conv wrapper computes the stride once
    (
        '    num_warps = 4 if mode == "prefill" else 8\n'
        "    launch_pdl = current_platform.is_arch_support_pdl()\n",
        '    num_warps = 4 if mode == "prefill" else 8\n'
        "    launch_pdl = current_platform.is_arch_support_pdl()\n"
        "    # Pure-prefill indices can be a strided block-table column view.\n"
        "    state_idx_stride = state_indices.stride(0)\n",
        1,
    ),
    # and passes it to _ple_conv_kernel
    (
        "        state_bs,\n        state_ws,\n        state_cs,\n        C=C,\n",
        "        state_idx_stride,\n        state_bs,\n        state_ws,\n        state_cs,\n        C=C,\n",
        1,
    ),
    # and to _ple_conv_writeback_kernel
    (
        "            state_bs,\n            state_ws,\n            state_cs,\n            C=C,\n",
        "            state_idx_stride,\n            state_bs,\n            state_ws,\n            state_cs,\n            C=C,\n",
        1,
    ),
]


def header(image: str, target: str, upstream_sha256: str, upstream_pr: str) -> str:
    """R11 overlay header. run.sh parses it and mounts the file at its upstream path; see docker/OVERLAYS.md."""
    return (
        "# R11-OVERLAY\n"
        f"# base_image_digest: {image.rsplit('@', 1)[-1]}\n"
        f"# upstream_file: {target.split('/dist-packages/', 1)[1]}\n"
        f"# upstream_file_sha256: {upstream_sha256}\n"
        f"# upstream_PR: {upstream_pr}\n"
        f"# generator: docker/{Path(__file__).name} (do not hand-edit)\n"
    )


def extract(image: str, dest: Path) -> None:
    cid = subprocess.check_output(["docker", "create", image], text=True).strip()
    try:
        subprocess.check_call(["docker", "cp", f"{cid}:{IN_IMAGE}", str(dest)])
    finally:
        subprocess.check_call(["docker", "rm", cid], stdout=subprocess.DEVNULL)


def overlay(text: str) -> str:
    for old, new, count in HUNKS:
        found = text.count(old)
        if found != count:
            raise SystemExit(f"apply_ple_stride_overlay: expected {count} of {old!r}, found {found}")
        text = text.replace(old, new)
    return text


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", default=IMAGE)
    ap.add_argument(
        "--src",
        type=Path,
        help="read the stock file from this path (e.g. an extracted vllm tree) instead of `docker create`",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "ple_ops.py",
    )
    args = ap.parse_args()
    if "@sha256:" not in args.image:
        raise SystemExit("apply_ple_stride_overlay: --image must be digest-pinned")
    if args.src is not None:
        stock = args.src.read_bytes()
    else:
        tmp = args.out.with_suffix(".stock.py")
        extract(args.image, tmp)
        stock = tmp.read_bytes()
        tmp.unlink(missing_ok=True)
    sha = hashlib.sha256(stock).hexdigest()
    if sha != STOCK_SHA256:
        raise SystemExit(
            f"apply_ple_stride_overlay: stock ops/ple.py sha256 {sha} != {STOCK_SHA256}. "
            "The hunks were written against the pinned nightly; rebase them for this image."
        )
    body = overlay(stock.decode())
    args.out.write_text(header(args.image, IN_IMAGE, sha, UPSTREAM_PR) + body)
    print(f"apply_ple_stride_overlay: wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
