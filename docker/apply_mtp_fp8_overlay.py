#!/usr/bin/env python3
"""Extract stock modelopt.py and dispatch MIXED_PRECISION MTP FP8_BLOCK_SCALES."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

IMAGE = (
    "vllm/vllm-openai:nightly-aarch64@"
    "sha256:df871f170ee7070fbdce162bde08fb616e311570c948a620be0d4b33fe02f87b"
)
IN_IMAGE = (
    "/usr/local/lib/python3.12/dist-packages/vllm/"
    "model_executor/layers/quantization/modelopt.py"
)
OLD_PREFIX = '''        elif prefix.startswith("model.language_model."):
            candidates.append(
                "language_model.model." + prefix[len("model.language_model.") :]
            )

        return tuple(dict.fromkeys(candidates))
'''
NEW_PREFIX = '''        elif prefix.startswith("model.language_model."):
            candidates.append(
                "language_model.model." + prefix[len("model.language_model.") :]
            )

        # Checkpoint MTP experts are mtp.layers.0. Runtime MTP is
        # mtp.layers.{num_hidden_layers} (48 on this card).
        marker = "mtp.layers."
        idx = prefix.find(marker)
        if idx >= 0:
            rest = prefix[idx + len(marker) :]
            dot = rest.find(".")
            layer_id = rest if dot < 0 else rest[:dot]
            if layer_id.isdigit() and layer_id != "0":
                tail = "" if dot < 0 else rest[dot:]
                candidates.append(prefix[:idx] + marker + "0" + tail)

        return tuple(dict.fromkeys(candidates))
'''
OLD_MOE = '''            if quant_algo == "MXFP8":
                return ModelOptMxFp8FusedMoE(
                    quant_config=self.mxfp8_config,
                    moe_config=layer.moe_config,
                )
            return None
'''
NEW_MOE = '''            if quant_algo == "MXFP8":
                return ModelOptMxFp8FusedMoE(
                    quant_config=self.mxfp8_config,
                    moe_config=layer.moe_config,
                )
            if quant_algo == "FP8_BLOCK_SCALES":
                from vllm.model_executor.layers.quantization.fp8 import (
                    Fp8Config,
                    Fp8MoEMethod,
                )

                group = 128
                for candidate in self._quantized_layer_prefix_candidates(prefix):
                    info = self.quantized_layers.get(candidate)
                    if info and info.get("group_size"):
                        group = int(info["group_size"])
                        break
                block_cfg = Fp8Config(
                    is_checkpoint_fp8_serialized=True,
                    activation_scheme="dynamic",
                    weight_block_size=[group, group],
                )
                return Fp8MoEMethod(block_cfg, layer)
            return None
'''


def extract(image: str, dest: Path) -> None:
    cid = subprocess.check_output(["docker", "create", image], text=True).strip()
    try:
        subprocess.check_call(["docker", "cp", f"{cid}:{IN_IMAGE}", str(dest)])
    finally:
        subprocess.check_call(["docker", "rm", cid], stdout=subprocess.DEVNULL)


def overlay(text: str) -> str:
    if "FP8_BLOCK_SCALES" in text and 'marker = "mtp.layers."' in text:
        return text
    if OLD_PREFIX not in text:
        raise SystemExit("apply_mtp_fp8_overlay: prefix-candidate block not found")
    if OLD_MOE not in text:
        raise SystemExit("apply_mtp_fp8_overlay: RoutedExperts MXFP8 block not found")
    text = text.replace(OLD_PREFIX, NEW_PREFIX, 1)
    return text.replace(OLD_MOE, NEW_MOE, 1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", default=IMAGE)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "modelopt.py",
    )
    args = ap.parse_args()
    stock = args.out.with_suffix(".stock.py")
    extract(args.image, stock)
    args.out.write_text(overlay(stock.read_text()))
    stock.unlink(missing_ok=True)
    print(f"apply_mtp_fp8_overlay: wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
