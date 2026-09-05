#!/usr/bin/env python3
"""Extract stock ple_layer.py from the pinned image and apply the mixed-quant FP8 PLE gate."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

IMAGE = (
    "vllm/vllm-openai:nightly-aarch64@"
    "sha256:df871f170ee7070fbdce162bde08fb616e311570c948a620be0d4b33fe02f87b"
)
IN_IMAGE = "/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia/ple_layer.py"
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
    if "Qwen4ExpPLEFp8EmbeddingMethod" in text and "ModelOptMixedPrecisionConfig" in text:
        return text
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
