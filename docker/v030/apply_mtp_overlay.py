#!/usr/bin/env python3
"""Patch v0.30.0 qwen4_exp/nvidia/mtp.py: local-argmax drafting (L1a) + reduced-vocab draft head (L1b).

L1a (F02, G02): Qwen4ExpMTP inherits LocalArgmaxMixin, so
  SPEC_CONFIG '"use_local_argmax_reduction": true' boots instead of raising
  "does not implement get_top_tokens()". Each rank takes a local argmax over
  its 124,160-row lm_head shard and all-gathers only [M, 2] (value, id).
  Same wiring as upstream Qwen3_5MTP (model_executor/models/qwen3_5_mtp.py).

L1b (F02, plan L1b): opt-in reduced-vocab draft head, default OFF.
  Set VLLM_QWEN38_DRAFT_VOCAB=/path/draft_vocab_<V>.json on BOTH ranks (the
  file must exist at that path in both containers) together with
  use_local_argmax_reduction:true. At load, Qwen4ExpMTP.load_weights copies
  the V' selected rows of the checkpoint lm_head.weight into a new buffer
  `draft_vocab_weight` (a new attribute, so load_eagle_model's lm_head
  rebinding at eagle/utils.py:125-143 cannot drop it), split 50/50 across TP
  ranks in sorted-id order, plus `draft_vocab_ids` (the d2t map to full ids).
  Both are allocated inside load_model, i.e. before memory profiling.
  get_top_tokens then runs F.linear over [V'/tp, 2560] + argmax per rank and
  all-gathers [M, 2]. Ties go to the lowest id, as with the full-vocab argmax.
  Any problem (flag off, probabilistic drafts, PP>1, bad json, row missing,
  unexpected logits scale/cap/dtype) logs a warning and falls back to stock
  on every TP rank (agreed via a MIN all-reduce over the TP CPU group).
  Target verify keeps the full head, so outputs stay lossless under greedy
  verification; only acceptance can change.

Output: docker/v030/mtp.py with an R11 header. Mount it read-only at
IN_IMAGE inside the v0.30.0 container on BOTH ranks. Never mount it on the
pinned nightly (different upstream sha; the generator refuses).
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
UPSTREAM_FILE = "vllm/models/qwen4_exp/nvidia/mtp.py"
IN_IMAGE = "/usr/local/lib/python3.12/dist-packages/" + UPSTREAM_FILE
# sha256 of the stock file in the v0.30.0 image (the pinned nightly df871f17
# ships a different file, 081315f0..., so this overlay cannot apply there).
UPSTREAM_FILE_SHA256 = (
    "928c450001ee42cae9f6b0d5e9dc477b63019af2e90baa77505659ea3334eb88"
)
UPSTREAM_PR = (
    "none (local; L1a mirrors upstream Qwen3_5MTP LocalArgmaxMixin wiring, "
    "L1b is FR-Spec arXiv:2502.14856 style)"
)
ENV = "VLLM_QWEN38_DRAFT_VOCAB"

HEADER = f"""\
# R11-OVERLAY
# base_image_digest: {BASE_IMAGE_DIGEST}
# upstream_file: {UPSTREAM_FILE}
# upstream_file_sha256: {UPSTREAM_FILE_SHA256}
# upstream_PR: {UPSTREAM_PR}
# generator: docker/v030/apply_mtp_overlay.py (do not hand-edit)
# features: L1a get_top_tokens via LocalArgmaxMixin (always on);
#   L1b reduced-vocab draft head, opt-in via {ENV}=<json>, default off
"""

# ---------------------------------------------------------------- L1a hunks
L1A_IMPORT_OLD = "from vllm.model_executor.models.interfaces import SupportsPP\n"
L1A_IMPORT_NEW = (
    "from vllm.model_executor.models.interfaces import LocalArgmaxMixin, SupportsPP\n"
)
L1A_CLASS_OLD = "class Qwen4ExpMTP(nn.Module, SupportsPP, Qwen4ExpMixtureOfExperts):\n"
L1A_CLASS_NEW = """\
class Qwen4ExpMTP(
    LocalArgmaxMixin, nn.Module, SupportsPP, Qwen4ExpMixtureOfExperts
):
"""
L1A_HUNKS = ((L1A_IMPORT_OLD, L1A_IMPORT_NEW), (L1A_CLASS_OLD, L1A_CLASS_NEW))

# ---------------------------------------------------------------- L1b hunks
L1B_STDLIB_OLD = "from collections.abc import Iterable\n"
L1B_STDLIB_NEW = "import json\nimport os\nfrom collections.abc import Iterable\n"

L1B_TORCH_OLD = "import torch\nfrom torch import nn\n"
L1B_TORCH_NEW = "import torch\nimport torch.nn.functional as F\nfrom torch import nn\n"

L1B_DIST_OLD = "from vllm.distributed import get_pp_group\n"
L1B_DIST_NEW = """\
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_gather,
)
from vllm.logger import init_logger
"""

L1B_HELPERS_OLD = """\


def _remap_ignored_layers(
"""
L1B_HELPERS_NEW = f"""\

logger = init_logger(__name__)

# L1b reduced-vocab draft head (recipe overlay). Default off.
_DRAFT_VOCAB_ENV = "{ENV}"
_DRAFT_VOCAB_MIN_IDS = 1024


def _load_draft_vocab(path: str, vocab_size: int) -> list[int]:
    \"\"\"Read and validate a draft_vocab_<V>.json id list (sorted, unique).\"\"\"
    with open(path) as f:
        doc = json.load(f)
    ids = doc.get("ids") if isinstance(doc, dict) else doc
    if isinstance(doc, dict) and doc.get("vocab_size", vocab_size) != vocab_size:
        raise ValueError(
            f"draft vocab built for vocab_size={{doc.get('vocab_size')}}, "
            f"model has {{vocab_size}}"
        )
    if not isinstance(ids, list) or len(ids) < _DRAFT_VOCAB_MIN_IDS:
        raise ValueError(f"draft vocab needs >= {{_DRAFT_VOCAB_MIN_IDS}} ids")
    prev = -1
    for i in ids:
        if type(i) is not int or not prev < i < vocab_size:
            raise ValueError("draft vocab ids must be sorted unique ints < vocab")
        prev = i
    return ids


def _draft_vocab_rank_slice(ids: list[int], tp_size: int, tp_rank: int) -> list[int]:
    \"\"\"Balanced contiguous split of the sorted ids; rank 0 gets the lowest.

    Keeping rank order == id order makes the cross-rank argmax (first max
    wins) pick the lowest id on ties, like a full-vocab argmax.
    \"\"\"
    per = -(-len(ids) // tp_size)
    return ids[tp_rank * per : (tp_rank + 1) * per]


def _remap_ignored_layers(
"""

L1B_INIT_OLD = """\
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
"""
L1B_INIT_NEW = """\
        self.logits_processor = LogitsProcessor(config.vocab_size)
        # L1b: filled by load_weights only when the reduced head is enabled.
        self.register_buffer("draft_vocab_weight", None, persistent=False)
        self.register_buffer("draft_vocab_ids", None, persistent=False)
        self.make_empty_intermediate_tensors = (
"""

L1B_LOAD_OLD = """\
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        def remap_weight_names():
            for name, weight in weights:
                remapped_name = _remap_mtp_weight_name(name)
                if remapped_name is not None:
                    yield remapped_name, weight
"""
L1B_LOAD_NEW = """\
    def get_top_tokens(self, hidden_states: torch.Tensor) -> torch.Tensor:
        weight = self.draft_vocab_weight
        if weight is None:
            return super().get_top_tokens(hidden_states)
        # L1b: this rank's V'/tp rows, local argmax, then [M, 2] all-gather.
        logits = F.linear(hidden_states, weight)
        local_idx = logits.argmax(dim=-1, keepdim=True)
        local_val = logits.gather(-1, local_idx).squeeze(-1)
        top = self.draft_vocab_ids[local_idx.squeeze(-1)]
        tp_size = get_tensor_model_parallel_world_size()
        if tp_size == 1:
            return top
        # float32 carries ids < 2**24 exactly (vocab is 248,320).
        pair = torch.stack([local_val.float(), top.float()], dim=-1)
        gathered = tensor_model_parallel_all_gather(pair, dim=-1)
        gathered = gathered.view(hidden_states.shape[0], tp_size, 2)
        best = gathered[:, :, 0].argmax(dim=-1, keepdim=True)
        return gathered[:, :, 1].gather(-1, best).squeeze(-1).to(torch.int64)

    def _draft_vocab_plan(self) -> list[int]:
        \"\"\"Ids this rank keeps for the L1b head; raises if L1b cannot run.\"\"\"
        path = os.environ.get(_DRAFT_VOCAB_ENV, "").strip()
        spec = self.vllm_config.speculative_config
        if spec is None or not spec.use_local_argmax_reduction:
            raise ValueError("needs speculative_config use_local_argmax_reduction")
        if spec.draft_sample_method == "probabilistic":
            raise ValueError("incompatible with draft_sample_method=probabilistic")
        if get_pp_group().world_size != 1:
            raise ValueError("pipeline parallel is not supported")
        if self.config.tie_word_embeddings:
            raise ValueError("tied embeddings are not supported")
        lp = self.logits_processor
        if lp.scale != 1.0 or lp.soft_cap is not None or lp.logits_as_input:
            raise ValueError("logits scale/soft_cap is not supported")
        ids = _load_draft_vocab(path, self.config.vocab_size)
        return _draft_vocab_rank_slice(
            ids,
            get_tensor_model_parallel_world_size(),
            get_tensor_model_parallel_rank(),
        )

    def _finish_draft_vocab(
        self,
        local_ids: list[int] | None,
        rows: torch.Tensor | None,
        err: str,
        requested: bool,
    ) -> None:
        \"\"\"Install the L1b head on every TP rank, or on none (fail closed).

        Every TP rank joins the agreement even with the env unset, so a rank
        that lacks it makes the others fall back instead of hanging load.
        \"\"\"
        head_dtype = self.logits_processor.head_dtype
        ok = local_ids is not None and rows is not None
        if ok and head_dtype not in (None, rows.dtype):
            ok, err = False, f"head_dtype {head_dtype} != lm_head {rows.dtype}"
        if get_tensor_model_parallel_world_size() > 1:
            flag = torch.tensor([1 if ok else 0], dtype=torch.int32)
            torch.distributed.all_reduce(
                flag,
                op=torch.distributed.ReduceOp.MIN,
                group=get_tp_group().cpu_group,
            )
            if ok and int(flag.item()) == 0:
                ok, err = False, "another TP rank could not build it"
        if not ok:
            if not requested:
                return
            logger.warning(
                "%s set but reduced draft head disabled (%s); using the full "
                "lm_head for drafting.",
                _DRAFT_VOCAB_ENV,
                err or "lm_head.weight not seen while loading",
            )
            return
        assert local_ids is not None and rows is not None
        # Read the device only here: on a PP>1 non-last rank lm_head is a
        # PPMissingLayer without .weight (the plan already failed closed).
        device = self.lm_head.weight.device
        self.draft_vocab_weight = rows.to(device).contiguous()
        self.draft_vocab_ids = torch.tensor(local_ids, dtype=torch.int64, device=device)
        logger.info(
            "Reduced draft head (L1b): %d rows on this rank (ids %d..%d), "
            "%.1f MiB; the full lm_head is kept for verify.",
            len(local_ids),
            local_ids[0],
            local_ids[-1],
            rows.numel() * rows.element_size() / 2**20,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        draft_vocab = bool(os.environ.get(_DRAFT_VOCAB_ENV, "").strip())
        local_ids: list[int] | None = None
        captured: dict[str, torch.Tensor] = {}
        err = ""
        if draft_vocab:
            try:
                local_ids = self._draft_vocab_plan()
            except Exception as e:  # fail closed to the stock head
                err = f"{type(e).__name__}: {e}"

        def remap_weight_names():
            for name, weight in weights:
                remapped_name = _remap_mtp_weight_name(name)
                if remapped_name is not None:
                    if local_ids is not None and remapped_name == "lm_head.weight":
                        index = torch.tensor(local_ids, device=weight.device)
                        captured["rows"] = weight.index_select(0, index)
                    yield remapped_name, weight
"""

L1B_RETURN_OLD = """\
        return loader.load_weights(remap_weight_names(), mapper=mapper)


__all__ = ["Qwen4ExpMTP", "Qwen4ExpMultiTokenPredictor"]
"""
L1B_RETURN_NEW = """\
        loaded = loader.load_weights(remap_weight_names(), mapper=mapper)
        if draft_vocab or get_tensor_model_parallel_world_size() > 1:
            self._finish_draft_vocab(
                local_ids, captured.get("rows"), err, draft_vocab
            )
        return loaded


__all__ = ["Qwen4ExpMTP", "Qwen4ExpMultiTokenPredictor"]
"""

L1B_HUNKS = (
    (L1B_STDLIB_OLD, L1B_STDLIB_NEW),
    (L1B_TORCH_OLD, L1B_TORCH_NEW),
    (L1B_DIST_OLD, L1B_DIST_NEW),
    (L1B_HELPERS_OLD, L1B_HELPERS_NEW),
    (L1B_INIT_OLD, L1B_INIT_NEW),
    (L1B_LOAD_OLD, L1B_LOAD_NEW),
    (L1B_RETURN_OLD, L1B_RETURN_NEW),
)
HUNKS = L1A_HUNKS + L1B_HUNKS


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def overlay(stock: str) -> str:
    got = sha256(stock)
    if got != UPSTREAM_FILE_SHA256:
        raise SystemExit(
            f"apply_mtp_overlay: stock {UPSTREAM_FILE} sha256 {got} "
            f"!= {UPSTREAM_FILE_SHA256}; rebase the L1a/L1b hunks"
        )
    text = stock
    for old, new in HUNKS:
        if text.count(old) != 1:
            raise SystemExit(f"apply_mtp_overlay: anchor not unique/found: {old[:60]!r}")
        text = text.replace(old, new, 1)
    return HEADER + text


def unoverlay(text: str) -> str:
    """Invert overlay(); tests use it to prove the diff is only these hunks."""
    if not text.startswith(HEADER):
        raise ValueError("R11 header missing or stale")
    text = text[len(HEADER) :]
    for old, new in reversed(HUNKS):
        if text.count(new) != 1:
            raise ValueError(f"overlay hunk missing: {new[:60]!r}")
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
        help="stock mtp.py already extracted from the image (skips docker)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "mtp.py",
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
            print(f"apply_mtp_overlay: {args.out} is stale", file=sys.stderr)
            return 1
        print(f"apply_mtp_overlay: {args.out} up to date")
        return 0
    args.out.write_text(text)
    print(f"apply_mtp_overlay: wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
