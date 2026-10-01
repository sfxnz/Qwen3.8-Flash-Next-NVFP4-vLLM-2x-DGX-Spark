#!/usr/bin/env python3
"""Extract stock modelopt.py and dispatch MIXED_PRECISION MTP FP8_BLOCK_SCALES."""
from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path

IMAGE = (
    "vllm/vllm-openai:nightly-aarch64@"
    "sha256:df871f170ee7070fbdce162bde08fb616e311570c948a620be0d4b33fe02f87b"
)
UPSTREAM_PR = "none (lab overlay; v0.30 routes MTP FP8_BLOCK_SCALES natively)"
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
        "--src",
        type=Path,
        help="read the stock file from this path (e.g. an extracted vllm tree) instead of `docker create`",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "modelopt.py",
    )
    args = ap.parse_args()
    if "@sha256:" not in args.image:
        raise SystemExit("apply_mtp_fp8_overlay: --image must be digest-pinned")
    if args.src is not None:
        stock = args.src.read_bytes()
    else:
        tmp = args.out.with_suffix(".stock.py")
        extract(args.image, tmp)
        stock = tmp.read_bytes()
        tmp.unlink(missing_ok=True)
    sha = hashlib.sha256(stock).hexdigest()
    args.out.write_text(header(args.image, IN_IMAGE, sha, UPSTREAM_PR) + overlay(stock.decode()))
    print(f"apply_mtp_fp8_overlay: wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
