#!/usr/bin/env python3
"""Classify a fresh 1-token prompt as a GDN prefill on v0.30.0 (S1.5 determinism).

On a step with no spec decodes, GDNAttentionMetadataBuilder calls
split_decodes_and_prefills(m, decode_threshold=1) with the default
treat_short_extends_as_decodes=True, so a request whose whole prompt is one token
(num_computed_tokens 0, query_len 1) runs the GDN *decode* path. That path uses the
request's state slot as the initial recurrent/conv state; only the prefill path zeroes
rows without an initial state. The first logits then depend on whatever the slot held
(measured: first-token logprob varies by up to 3 nats run to run, lengths 2+ exact).
The mamba builder already passes treat_short_extends_as_decodes=False; this does the same
for GDN whenever is_prefilling is populated (dummy/capture runs keep stock behaviour).

Output: docker/v030/gdn_attn.py with an R11 header (both ranks mount it).
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
UPSTREAM_FILE = "vllm/v1/attention/backends/gdn_attn.py"
IN_IMAGE = "/usr/local/lib/python3.12/dist-packages/" + UPSTREAM_FILE
UPSTREAM_FILE_SHA256 = "3831019404a92200a38b43571c040cffeb6638bbdc71e4cb5e680dbc6c93d054"
UPSTREAM_PR = "none (local S1.5 fix; mirrors mamba_attn.py treat_short_extends_as_decodes=False)"

HEADER = f"""\
# R11-OVERLAY
# base_image_digest: {BASE_IMAGE_DIGEST}
# upstream_file: {UPSTREAM_FILE}
# upstream_file_sha256: {UPSTREAM_FILE_SHA256}
# upstream_PR: {UPSTREAM_PR}
# generator: docker/v030/apply_gdn_fresh_prefill_overlay.py (do not hand-edit)
"""

OLD = """\
                split_decodes_and_prefills(m, decode_threshold=1)
"""
NEW = """\
                # S1.5: a fresh 1-token prompt is a prefill (zeroed initial state),
                # not a decode that reads a stale state slot.
                split_decodes_and_prefills(
                    m,
                    decode_threshold=1,
                    treat_short_extends_as_decodes=m.is_prefilling is None,
                )
"""
HUNKS = ((OLD, NEW),)


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def overlay(stock: str) -> str:
    got = sha256(stock)
    if got != UPSTREAM_FILE_SHA256:
        raise SystemExit(f"apply_gdn_fresh_prefill_overlay: stock {UPSTREAM_FILE} sha256 {got} != {UPSTREAM_FILE_SHA256}")
    text = stock
    for old, new in HUNKS:
        if text.count(old) != 1:
            raise SystemExit("apply_gdn_fresh_prefill_overlay: anchor not unique/found")
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
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "gdn_attn.py")
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
            print(f"apply_gdn_fresh_prefill_overlay: {args.out} is stale", file=sys.stderr)
            return 1
        print(f"apply_gdn_fresh_prefill_overlay: {args.out} up to date")
        return 0
    args.out.write_text(text)
    print(f"apply_gdn_fresh_prefill_overlay: wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
