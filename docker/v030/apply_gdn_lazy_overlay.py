#!/usr/bin/env python3
"""Patch v0.30.0 GDN files for K3: MTP decode with lazy state commit (L3, F08).

Two overlays (both ranks mount both; see docker/v030/K3.md):

1. docker/v030/gdn_lazy_linear_attn.py over
   vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py
   - QwenGatedDeltaNetAttention.__init__: gdl_layer_init(self) runs the load-time
     self-test once per variant, before the FULL cudagraph capture.
   - _forward_core_decode_spec_post_conv_fused_norm: gdl_decode() replaces
     ops.fused_gdn_decode_post_conv_mtp for pure spec batches (lazy rows, or
     eager rows near a 1600-token align boundary).
   - _forward_core (every non-pure-spec batch): gdl_fixup() materializes any
     pending ring into the stock layout before the stock FLA / chunk readers.
   - docker/v030/gdn_lazy.py embedded verbatim at the end.
2. docker/v030/gdn_lazy_attn.py over vllm/v1/attention/backends/gdn_attn.py
   - the S1.5 fresh-prefill hunk (imported from apply_gdn_fresh_prefill_overlay,
     so this file REPLACES docker/v030/gdn_attn.py in OVERLAYS when K3 is on;
     run.sh refuses two overlays of one upstream file);
   - GDNAttentionMetadata.lazy_* fields; the builder allocates one int32 header
     per state block per layer and fills seq_lens / block size / non-spec rows.

Default OFF: every hunk is inert unless VLLM_QWEN38_GDN_LAZY=1 (or =force, which
skips the self-test; harness use only) on BOTH ranks. With the env unset the
overlays behave as stock + S1.5.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE_IMAGE = (
    "vllm/vllm-openai:v0.30.0-aarch64@"
    "sha256:4864d46625cbc3307623e29ac742030655e27249feba7b97ec925ce4cc4dfb56"
)
BASE_IMAGE_DIGEST = (
    "sha256:4864d46625cbc3307623e29ac742030655e27249feba7b97ec925ce4cc4dfb56"
)
SITE = "/usr/local/lib/python3.12/dist-packages/"
ENV = "VLLM_QWEN38_GDN_LAZY"
KERNELS = HERE / "gdn_lazy.py"
EMBED_MARK = "# ---- K3 embedded from docker/v030/gdn_lazy.py (do not hand-edit) ----\n"
UPSTREAM_PR = "none (local K3; deferred-commit pattern of in-tree KDA RecoverSSM)"

LIN_FILE = "vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py"
LIN_SHA = "b80f8e6f3fff442fb880aebb3209c52f814f66490009f4ac4003de603576d10a"
LIN_OUT = HERE / "gdn_lazy_linear_attn.py"
ATT_FILE = "vllm/v1/attention/backends/gdn_attn.py"
ATT_SHA = "3831019404a92200a38b43571c040cffeb6638bbdc71e4cb5e680dbc6c93d054"
ATT_OUT = HERE / "gdn_lazy_attn.py"


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _fresh_prefill_hunks():
    path = HERE / "apply_gdn_fresh_prefill_overlay.py"
    spec = importlib.util.spec_from_file_location("apply_gdn_fresh_prefill_overlay", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.UPSTREAM_FILE_SHA256 == ATT_SHA
    return tuple(mod.HUNKS)


# ------------------------------------------------------------ linear-attn hunks
LIN_INIT_OLD = """\
        logger.info_once("GDN decode kernel: %s", self.gdn_decode_kernel)
"""
LIN_INIT_NEW = """\
        logger.info_once("GDN decode kernel: %s", self.gdn_decode_kernel)
        gdl_layer_init(self)  # K3: self-test before any cudagraph capture
"""
LIN_DEC_OLD = """\
        num_requests = attn_metadata.num_spec_decodes
        ops.fused_gdn_decode_post_conv_mtp(
"""
LIN_DEC_NEW = """\
        num_requests = attn_metadata.num_spec_decodes
        if gdl_decode(self, mixed_qkv, a, b, output_gate, core_attn_out, attn_metadata):
            return  # K3 lazy commit (docker/v030/gdn_lazy.py)
        ops.fused_gdn_decode_post_conv_mtp(
"""
LIN_FIX_OLD = """\
        assert isinstance(attn_metadata, GDNAttentionMetadata)

        if (
            self.enable_packed_recurrent_decode
"""
LIN_FIX_NEW = """\
        assert isinstance(attn_metadata, GDNAttentionMetadata)
        gdl_fixup(self, attn_metadata)  # K3: pending rings -> stock layout

        if (
            self.enable_packed_recurrent_decode
"""
LIN_HUNKS = ((LIN_INIT_OLD, LIN_INIT_NEW), (LIN_DEC_OLD, LIN_DEC_NEW),
             (LIN_FIX_OLD, LIN_FIX_NEW))

# --------------------------------------------------------------- gdn_attn hunks
ATT_FIELDS_OLD = """\
    token_chunk_offset_ptr: torch.Tensor | None = None


class GDNAttentionMetadataBuilder"""
ATT_FIELDS_NEW = """\
    token_chunk_offset_ptr: torch.Tensor | None = None

    # K3 lazy GDN commit (docker/v030/gdn_lazy.py). Unset when
    # VLLM_QWEN38_GDN_LAZY is off.
    lazy_hdr: dict | None = None  # layer name -> int32 [num_blocks]
    lazy_seq_lens: torch.Tensor | None = None  # [batch] num_computed + query_len
    lazy_block_size: int = -1  # align block size; 0 = no align copies
    lazy_ns_state_indices: torch.Tensor | None = None  # [non-spec rows, 1+k]
    lazy_ns_prefilling: torch.Tensor | None = None  # [non-spec rows] bool


class GDNAttentionMetadataBuilder"""
ATT_INIT_OLD = """\
        self.num_accepted_tokens: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs,),
            dtype=torch.int32,
            device=device,
        )

    def _build_chunk_metadata(
"""
ATT_INIT_NEW = """\
        self.num_accepted_tokens: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs,),
            dtype=torch.int32,
            device=device,
        )

        # K3: one int32 header per state block per layer (pending ring length
        # of the committed state in that block; 0 = stock layout).
        import os as _os

        self.lazy_hdr: dict | None = None
        self.lazy_block_size = -1
        _mode = vllm_config.cache_config.mamba_cache_mode
        _nblk = vllm_config.cache_config.num_gpu_blocks
        if (
            _os.environ.get("VLLM_QWEN38_GDN_LAZY", "0").strip().lower() in ("1", "force")
            and self.num_spec > 0
            and _mode in ("none", "align")
            and _nblk
        ):
            self.lazy_hdr = {
                name: torch.zeros(int(_nblk), dtype=torch.int32, device=device)
                for name in layer_names
            }
            self.lazy_block_size = kv_cache_spec.block_size if _mode == "align" else 0

    def _build_chunk_metadata(
"""
ATT_BUILD_OLD = """\
            token_chunk_offset_ptr=token_chunk_offset_ptr,
        )
        return attn_metadata

    def build_for_cudagraph_capture(
"""
ATT_BUILD_NEW = """\
            token_chunk_offset_ptr=token_chunk_offset_ptr,
        )
        if self.lazy_hdr is not None:
            # K3: m.seq_lens is the runner's persistent buffer (graph-safe).
            attn_metadata.lazy_hdr = self.lazy_hdr
            attn_metadata.lazy_block_size = self.lazy_block_size
            attn_metadata.lazy_seq_lens = m.seq_lens
            if (num_prefills > 0 or num_decodes > 0) and m.is_prefilling is not None:
                _n = m.num_reqs
                if spec_sequence_masks_cpu is None:
                    _ns = torch.ones(_n, dtype=torch.bool)
                else:
                    _ns = ~spec_sequence_masks_cpu[:_n]
                attn_metadata.lazy_ns_state_indices = block_table_tensor[:_n][
                    _ns, : self.num_spec + 1
                ]
                attn_metadata.lazy_ns_prefilling = async_tensor_h2d(
                    m.is_prefilling[:_n][_ns], device=query_start_loc.device
                )
        return attn_metadata

    def build_for_cudagraph_capture(
"""


def att_hunks():
    return _fresh_prefill_hunks() + (
        (ATT_FIELDS_OLD, ATT_FIELDS_NEW),
        (ATT_INIT_OLD, ATT_INIT_NEW),
        (ATT_BUILD_OLD, ATT_BUILD_NEW),
    )


def lin_header(kernels: str) -> str:
    return f"""\
# R11-OVERLAY
# base_image_digest: {BASE_IMAGE_DIGEST}
# upstream_file: {LIN_FILE}
# upstream_file_sha256: {LIN_SHA}
# upstream_PR: {UPSTREAM_PR}
# generator: docker/v030/apply_gdn_lazy_overlay.py (do not hand-edit)
# embedded: docker/v030/gdn_lazy.py sha256 {sha256(kernels)}
# features: K3 lazy GDN state commit for MTP verify, opt-in via {ENV}=1,
#   default off, self-tested (bitwise vs stock) at layer init; pair with
#   docker/v030/gdn_lazy_attn.py
"""


def att_header() -> str:
    return f"""\
# R11-OVERLAY
# base_image_digest: {BASE_IMAGE_DIGEST}
# upstream_file: {ATT_FILE}
# upstream_file_sha256: {ATT_SHA}
# upstream_PR: {UPSTREAM_PR}
# generator: docker/v030/apply_gdn_lazy_overlay.py (do not hand-edit)
# features: S1.5 fresh-prefill hunk (same as docker/v030/gdn_attn.py) + K3
#   metadata (inert unless {ENV}=1); replaces docker/v030/gdn_attn.py in OVERLAYS
"""


def _apply(stock: str, want_sha: str, hunks, name: str) -> str:
    got = sha256(stock)
    if got != want_sha:
        raise SystemExit(f"apply_gdn_lazy_overlay: stock {name} sha256 {got} != {want_sha}; "
                         "rebase the K3 hunks")
    text = stock
    for old, new in hunks:
        if text.count(old) != 1:
            raise SystemExit(f"apply_gdn_lazy_overlay: anchor not unique/found in {name}: "
                             f"{old[:60]!r}")
        text = text.replace(old, new, 1)
    return text


def overlay_linear(stock: str, kernels: str) -> str:
    if "from __future__" in kernels:
        raise SystemExit("apply_gdn_lazy_overlay: gdn_lazy.py must not use __future__ imports")
    body = _apply(stock, LIN_SHA, LIN_HUNKS, LIN_FILE)
    return lin_header(kernels) + body + "\n\n" + EMBED_MARK + kernels


def overlay_attn(stock: str) -> str:
    return att_header() + _apply(stock, ATT_SHA, att_hunks(), ATT_FILE)


def _revert(body: str, hunks) -> str:
    for old, new in reversed(hunks):
        if body.count(new) != 1:
            raise ValueError(f"overlay hunk missing: {new[:60]!r}")
        body = body.replace(new, old, 1)
    return body


def unoverlay_linear(text: str) -> tuple[str, str]:
    """Invert overlay_linear() -> (stock, kernels); tests prove the diff."""
    if not text.startswith("# R11-OVERLAY\n"):
        raise ValueError("R11 header missing")
    body, sep, kernels = text.partition("\n\n" + EMBED_MARK)
    if not sep:
        raise ValueError("embedded K3 block missing")
    head = lin_header(kernels)
    if not body.startswith(head):
        raise ValueError("R11 header stale")
    return _revert(body[len(head):], LIN_HUNKS), kernels


def unoverlay_attn(text: str) -> str:
    head = att_header()
    if not text.startswith(head):
        raise ValueError("R11 header missing or stale")
    return _revert(text[len(head):], att_hunks())


def extract(image: str, files: list[str], dest: Path) -> None:
    cid = subprocess.check_output(["docker", "create", image], text=True).strip()
    try:
        for f in files:
            out = dest / f
            out.parent.mkdir(parents=True, exist_ok=True)
            subprocess.check_call(["docker", "cp", f"{cid}:{SITE}{f}", str(out)])
    finally:
        subprocess.check_call(["docker", "rm", cid], stdout=subprocess.DEVNULL)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--image", default=BASE_IMAGE)
    ap.add_argument("--src", type=Path,
                    help="extracted site-packages root holding vllm/... (skips docker)")
    ap.add_argument("--check", action="store_true", help="exit 1 if an output is stale")
    args = ap.parse_args()
    if args.src is not None:
        root = args.src
    else:
        root = HERE / ".gdn_lazy_stock"
        extract(args.image, [LIN_FILE, ATT_FILE], root)
    lin = overlay_linear((root / LIN_FILE).read_text(), KERNELS.read_text())
    att = overlay_attn((root / ATT_FILE).read_text())
    if args.src is None:
        import shutil

        shutil.rmtree(root, ignore_errors=True)
    rc = 0
    for out, text in ((LIN_OUT, lin), (ATT_OUT, att)):
        if args.check:
            if not out.exists() or out.read_text() != text:
                print(f"apply_gdn_lazy_overlay: {out} is stale", file=sys.stderr)
                rc = 1
            else:
                print(f"apply_gdn_lazy_overlay: {out} up to date")
        else:
            out.write_text(text)
            print(f"apply_gdn_lazy_overlay: wrote {out}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
