#!/usr/bin/env python3
"""Backport vLLM PR #56067 (defer reasoning-usage recount) onto v0.30.0 serving.py.

API-server-only overlay. Thinking-off streams otherwise re-decode and re-parse
the whole generated history on every delta (count_reasoning_tokens via the
parser-engine adapter), which is O(n^2) CPU per stream (finding G05).

Output: docker/v030/serving.py with an R11 header. Mount it at
IN_IMAGE inside the v0.30.0 container on the head rank (the API server only
runs there; the worker rank never imports it).
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
BASE_IMAGE_DIGEST = (
    "sha256:4864d46625cbc3307623e29ac742030655e27249feba7b97ec925ce4cc4dfb56"
)
UPSTREAM_FILE = "vllm/entrypoints/openai/chat_completion/serving.py"
IN_IMAGE = "/usr/local/lib/python3.12/dist-packages/" + UPSTREAM_FILE
# sha256 of the stock file in the v0.30.0 image. The pinned nightly
# df871f17 (0.28.1rc1.dev437) ships a byte-identical file.
UPSTREAM_FILE_SHA256 = (
    "ea1f76074a9587c8054f54d30a6ba748b6a5fc4d90dd82c801504505c1da1f92"
)
UPSTREAM_PR = "vllm-project/vllm#56067 (merged 8b47e8b22caf0022b00df3b421cf2d41419a04a7)"

HEADER = f"""\
# R11-OVERLAY
# base_image_digest: {BASE_IMAGE_DIGEST}
# upstream_file: {UPSTREAM_FILE}
# upstream_file_sha256: {UPSTREAM_FILE_SHA256}
# upstream_PR: {UPSTREAM_PR}
# generator: docker/v030/apply_api_overlays.py (do not hand-edit)
"""

# Hunk 1: count per delta only when continuous usage is requested.
OLD_DELTA = """\
                    if parser is not None:
                        generated_token_ids[i].extend(output.token_ids)
                        previous_reasoning_tokens[i] = parser.count_reasoning_tokens(
                            tuple(generated_token_ids[i])
                        )
"""
NEW_DELTA = """\
                    if parser is not None and self._include_reasoning_tokens_details:
                        generated_token_ids[i].extend(output.token_ids)
                        if include_continuous_usage:
                            previous_reasoning_tokens[i] = (
                                parser.count_reasoning_tokens(
                                    tuple(generated_token_ids[i])
                                )
                            )
"""
# Hunk 2: one recount per nonempty choice before final usage.
OLD_FINAL = """\
                    data = chunk.model_dump_json(exclude_unset=True)
                    yield f"data: {data}\\n\\n"

            # once the final token is handled, if stream_options.include_usage
            # is sent, send the usage
            if include_usage:
"""
NEW_FINAL = """\
                    data = chunk.model_dump_json(exclude_unset=True)
                    yield f"data: {data}\\n\\n"

            if self._include_reasoning_tokens_details and not include_continuous_usage:
                for i, parser in enumerate(parsers):
                    if parser is not None and generated_token_ids[i]:
                        previous_reasoning_tokens[i] = parser.count_reasoning_tokens(
                            tuple(generated_token_ids[i])
                        )

            # once the final token is handled, if stream_options.include_usage
            # is sent, send the usage
            if include_usage:
"""
HUNKS = ((OLD_DELTA, NEW_DELTA), (OLD_FINAL, NEW_FINAL))


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def overlay(stock: str) -> str:
    got = sha256(stock)
    if got != UPSTREAM_FILE_SHA256:
        raise SystemExit(
            f"apply_api_overlays: stock {UPSTREAM_FILE} sha256 {got} "
            f"!= {UPSTREAM_FILE_SHA256}; rebase the #56067 hunks"
        )
    text = stock
    for old, new in HUNKS:
        if text.count(old) != 1:
            raise SystemExit("apply_api_overlays: #56067 anchor not unique/found")
        text = text.replace(old, new, 1)
    return HEADER + text


def unoverlay(text: str) -> str:
    """Invert overlay(); used by tests to prove the diff is only #56067."""
    if not text.startswith(HEADER):
        raise ValueError("R11 header missing or stale")
    text = text[len(HEADER) :]
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
    ap.add_argument(
        "--src",
        type=Path,
        help="stock serving.py already extracted from the image (skips docker)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "serving.py",
    )
    ap.add_argument(
        "--check",
        action="store_true",
        help="exit 1 if --out differs from a fresh render",
    )
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
            print(f"apply_api_overlays: {args.out} is stale", file=sys.stderr)
            return 1
        print(f"apply_api_overlays: {args.out} up to date")
        return 0
    args.out.write_text(text)
    print(f"apply_api_overlays: wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
