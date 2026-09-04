#!/usr/bin/env python3
"""Extract stock ple_layer.py from the pinned image and apply the mixed-quant FP8 PLE gate."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

IMAGE = (
    "vllm/vllm-openai:qwen38-flash-next@"
    "sha256:3b0e188ffceb3d07e09c3cb5215433a0020eacf02d7f882ed3a8bfd15454477e"
)
IN_IMAGE = "/usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py"
OLD = '''def _get_ple_embedding_quant_method(
    quant_config: QuantizationConfig | None,
    prefix: str,
) -> QuantizeMethodBase | None:
    """Select global-scale FP8 only for quantized PLE checkpoint shards."""

    if not isinstance(quant_config, Fp8Config):
        return None
'''
NEW = '''def _get_ple_embedding_quant_method(
    quant_config: QuantizationConfig | None,
    prefix: str,
) -> QuantizeMethodBase | None:
    """Select global-scale FP8 only for quantized PLE checkpoint shards."""
    import os

    if os.environ.get("VLLM_PLE_FP8_CHECKPOINT") == "1":
        return Qwen3_8FlashNextPLEFp8EmbeddingMethod()
    if not isinstance(quant_config, Fp8Config):
        return None
'''


def extract(image: str, dest: Path) -> None:
    cid = subprocess.check_output(["docker", "create", image], text=True).strip()
    try:
        subprocess.check_call(["docker", "cp", f"{cid}:{IN_IMAGE}", str(dest)])
    finally:
        subprocess.check_call(["docker", "rm", cid], stdout=subprocess.DEVNULL)


def overlay(text: str) -> str:
    if "VLLM_PLE_FP8_CHECKPOINT" in text:
        return text
    if OLD not in text:
        raise SystemExit("apply_ple_overlay: stock resolver block not found")
    return text.replace(OLD, NEW, 1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", default=IMAGE)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "ple_layer.py",
    )
    args = ap.parse_args()
    stock = args.out.with_suffix(".stock.py")
    extract(args.image, stock)
    args.out.write_text(overlay(stock.read_text()))
    stock.unlink(missing_ok=True)
    print(f"apply_ple_overlay: wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
