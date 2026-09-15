#!/usr/bin/env python3
"""Extract stock modelopt.py and apply MIXED_PRECISION MTP block-FP8 dispatch."""
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
OLD_IMPORT = '''from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod
'''
NEW_IMPORT = '''from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.fp8 import Fp8Config, Fp8MoEMethod
from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod
'''
OLD_ALGOS = '''logger = init_logger(__name__)

# Single source of truth for the ModelOpt linear algos.
'''
NEW_ALGOS = '''logger = init_logger(__name__)

# ``FP8_PB_WO`` is ModelOpt's canonical 2D block-FP8 name. Early composed
# Qwen3.8-Flash-Next checkpoints used ``FP8_BLOCK_SCALES`` for the same tensor
# layout, so retain it as a checkpoint-compatibility alias.
_BLOCK_FP8_MOE_ALGOS = ("FP8_PB_WO", "FP8_BLOCK_SCALES")

# Single source of truth for the ModelOpt linear algos.
'''
OLD_INIT = '''        self.w4a16_nvfp4_config = w4a16_nvfp4_config
        self.mxfp8_config = mxfp8_config

    def has_blocked_weights(self) -> bool:
        # Same gate as ModelOptFp8Config.has_blocked_weights, resolved per
        # layer: "+quant_fp8" must be on as soon as any layer is block-scaled.
        return any(
            info.get("quant_algo", "").upper() == "FP8_PB_WO"
            for info in self.quantized_layers.values()
        )
'''
NEW_INIT = '''        self.w4a16_nvfp4_config = w4a16_nvfp4_config
        self.mxfp8_config = mxfp8_config

        block_sizes = {
            int(layer_info.get("group_size", 128))
            for layer_info in quantized_layers.values()
            if layer_info.get("quant_algo", "").upper() in _BLOCK_FP8_MOE_ALGOS
        }
        if len(block_sizes) > 1:
            raise ValueError(
                "MIXED_PRECISION currently requires all block-FP8 MoE layers "
                f"to use one group_size, got {sorted(block_sizes)}."
            )
        block_size = next(iter(block_sizes), 128)
        self.fp8_block_config = Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            weight_block_size=[block_size, block_size],
        )

    def has_blocked_weights(self) -> bool:
        # Same gate as ModelOptFp8Config.has_blocked_weights, resolved per
        # layer: "+quant_fp8" must be on as soon as any layer is block-scaled.
        return any(
            info.get("quant_algo", "").upper() in _BLOCK_FP8_MOE_ALGOS
            for info in self.quantized_layers.values()
        )
'''
OLD_MOE = '''        if isinstance(layer, RoutedExperts):
            if quant_algo == "FP8":
'''
NEW_MOE = '''        if isinstance(layer, RoutedExperts):
            if quant_algo in _BLOCK_FP8_MOE_ALGOS:
                return Fp8MoEMethod(self.fp8_block_config, layer)
            if quant_algo == "FP8":
'''
OLD_INLINE_BLOCK = '''            if quant_algo == "MXFP8":
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
NEW_INLINE_BLOCK = '''            if quant_algo == "MXFP8":
                return ModelOptMxFp8FusedMoE(
                    quant_config=self.mxfp8_config,
                    moe_config=layer.moe_config,
                )
            return None
'''


def extract(image: str, dest: Path) -> None:
    cid = subprocess.check_output(["docker", "create", image], text=True).strip()
    try:
        subprocess.check_call(["docker", "cp", f"{cid}:{IN_IMAGE}", str(dest)])
    finally:
        subprocess.check_call(["docker", "rm", cid], stdout=subprocess.DEVNULL)


def overlay(text: str) -> str:
    if OLD_PREFIX in text:
        text = text.replace(OLD_PREFIX, NEW_PREFIX, 1)
    elif 'marker = "mtp.layers."' not in text:
        raise SystemExit("apply_mtp_fp8_overlay: prefix-candidate block not found")
    if "_BLOCK_FP8_MOE_ALGOS" in text and "fp8_block_config" in text:
        if OLD_INLINE_BLOCK in text:
            text = text.replace(OLD_INLINE_BLOCK, NEW_INLINE_BLOCK, 1)
        return text
    if OLD_IMPORT in text:
        text = text.replace(OLD_IMPORT, NEW_IMPORT, 1)
    elif "from vllm.model_executor.layers.quantization.fp8 import Fp8Config" not in text:
        raise SystemExit("apply_mtp_fp8_overlay: QuantizationConfig import block not found")
    if OLD_ALGOS in text:
        text = text.replace(OLD_ALGOS, NEW_ALGOS, 1)
    elif "_BLOCK_FP8_MOE_ALGOS" not in text:
        raise SystemExit("apply_mtp_fp8_overlay: logger block not found")
    if OLD_INIT in text:
        text = text.replace(OLD_INIT, NEW_INIT, 1)
    elif "fp8_block_config" not in text:
        raise SystemExit("apply_mtp_fp8_overlay: MIXED_PRECISION __init__ block not found")
    if OLD_INLINE_BLOCK in text:
        text = text.replace(OLD_INLINE_BLOCK, NEW_INLINE_BLOCK, 1)
    if OLD_MOE in text:
        text = text.replace(OLD_MOE, NEW_MOE, 1)
    elif "if quant_algo in _BLOCK_FP8_MOE_ALGOS:" not in text:
        raise SystemExit("apply_mtp_fp8_overlay: RoutedExperts dispatch block not found")
    return text


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
