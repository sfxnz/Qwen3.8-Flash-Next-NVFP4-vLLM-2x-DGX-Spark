#!/usr/bin/env python3
"""Sort QSA top-k block indices so sparse attention sums in a fixed order on v0.30.0 (S1.5).

torch.ops._C.persistent_topk returns the right *set* of compressed blocks but in a
run-dependent *order* (measured in the v0.30 image: same set 20/20, same order 1/20). The
expanded token list feeds the sparse attention tile loop in that order, so any row with
more visible blocks than block_topk (context > 2048 tokens at budget 2048, ratio 4) sums in
a different order each run: prompt logprobs stop being bit-exact from position 2052 on.
This sorts those rows ascending right after the top-k. Rows with visible <= block_topk take
the identity path (0..visible-1, then -1 padding) and are left untouched.
With tied logits persistent_topk also picks a run-dependent *set*; the prefill path
re-selects full rows deterministically (ties to the lowest index). Decode keeps only the sort.

Output: docker/v030/qsa_indexer.py with an R11 header (both ranks mount it).
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
UPSTREAM_FILE = "vllm/models/qwen4_exp/nvidia/ops/qsa_indexer.py"
IN_IMAGE = "/usr/local/lib/python3.12/dist-packages/" + UPSTREAM_FILE
UPSTREAM_FILE_SHA256 = "2c34883a6396bac0238f6308196854b653eadab8bbebbe7dd2d558e30d065cbb"
UPSTREAM_PR = "none (local S1.5 fix; F23 persistent_topk order)"

HEADER = f"""\
# R11-OVERLAY
# base_image_digest: {BASE_IMAGE_DIGEST}
# upstream_file: {UPSTREAM_FILE}
# upstream_file_sha256: {UPSTREAM_FILE_SHA256}
# upstream_PR: {UPSTREAM_PR}
# generator: docker/v030/apply_qsa_topk_order_overlay.py (do not hand-edit)
"""

OLD = """\
    topk_op(
        logits,
        visible_blocks,
        block_indices,
        topk_workspace,
        block_topk,
        logits.shape[1],
    )
"""
NEW = OLD + """\
    # S1.5: persistent_topk's output order is run-dependent; sort full rows so the
    # sparse attention accumulates in a fixed order (identity rows are already sorted).
    if logits.shape[1] > block_topk:
        full = (visible_blocks > block_topk).unsqueeze(1)
        block_indices.copy_(
            torch.where(full, block_indices.sort(dim=1).values, block_indices)
        )
"""

# Prefill only: with tied logits at the k-th boundary persistent_topk also picks a
# run-dependent *set* (measured: same set 1/20 on quantised logits). Rebuild full rows as
# "every logit > threshold, then the lowest-index ties", ascending. O(rows x width); the
# prefill width is bounded by max_seq_len / compress_ratio, decode (full block-table
# width) is left alone.
OLD_DECODE_DEF = """\
def qsa_select_paged_decode(
"""
NEW_DECODE_DEF = """\
def _deterministic_topk_ties(
    logits: torch.Tensor,
    visible_blocks: torch.Tensor,
    block_indices: torch.Tensor,
    block_topk: int,
) -> None:
    \"\"\"S1.5: exact top-k set with ties broken by lowest index, ascending (full rows).\"\"\"
    rows, width = logits.shape
    if width <= block_topk or rows == 0:
        return
    cols = torch.arange(width, device=logits.device, dtype=torch.int32)
    valid = cols.unsqueeze(0) < visible_blocks.unsqueeze(1)
    thr = logits.gather(1, block_indices.long().clamp(0, width - 1)).amin(1, keepdim=True)
    gt = (logits > thr) & valid
    eq = (logits == thr) & valid
    need = block_topk - gt.sum(1, keepdim=True, dtype=torch.int32)
    keep = gt | (eq & (eq.cumsum(1, dtype=torch.int32) <= need))
    slot = torch.where(keep, keep.cumsum(1, dtype=torch.int32) - 1, block_topk)
    out = torch.full((rows, block_topk + 1), -1, dtype=block_indices.dtype, device=logits.device)
    out.scatter_(1, slot.long(), cols.to(block_indices.dtype).expand(rows, -1))
    full = (visible_blocks > block_topk).unsqueeze(1)
    block_indices.copy_(torch.where(full, out[:, :block_topk], block_indices))


def qsa_select_paged_decode(
"""
OLD_PREFILL = """\
        _topk(
            logits,
            visible_blocks[query_slice],
            token_topk,
            compress_ratio,
            block_indices[query_slice],
            topk_workspace,
        )
"""
NEW_PREFILL = OLD_PREFILL + """\
        _deterministic_topk_ties(
            logits,
            visible_blocks[query_slice],
            block_indices[query_slice],
            token_topk // compress_ratio,
        )
"""
HUNKS = ((OLD, NEW), (OLD_DECODE_DEF, NEW_DECODE_DEF), (OLD_PREFILL, NEW_PREFILL))


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def overlay(stock: str) -> str:
    got = sha256(stock)
    if got != UPSTREAM_FILE_SHA256:
        raise SystemExit(f"apply_qsa_topk_order_overlay: stock {UPSTREAM_FILE} sha256 {got} != {UPSTREAM_FILE_SHA256}")
    text = stock
    for old, new in HUNKS:
        if text.count(old) != 1:
            raise SystemExit("apply_qsa_topk_order_overlay: anchor not unique/found")
        text = text.replace(old, new, 1)
    return HEADER + text


def unoverlay(text: str) -> str:
    """Invert overlay(); tests use it to prove the diff is only the one hunk."""
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
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "qsa_indexer.py")
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
            print(f"apply_qsa_topk_order_overlay: {args.out} is stale", file=sys.stderr)
            return 1
        print(f"apply_qsa_topk_order_overlay: {args.out} up to date")
        return 0
    args.out.write_text(text)
    print(f"apply_qsa_topk_order_overlay: wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
