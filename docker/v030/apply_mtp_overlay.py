#!/usr/bin/env python3
"""Patch v0.30.0 qwen4_exp/nvidia/mtp.py: local-argmax drafting (L1a) + reduced-vocab (L1b) and FP8 (L1b') draft heads.

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

L1b' (plan L1b', K1 FP8 variant): opt-in draft-only FP8 head, default OFF.
  Set VLLM_QWEN38_DRAFT_HEAD_FP8=1 (or auto | marlin | w8a8) on BOTH ranks.
  At load the draft quantizes its OWN copy of the lm_head rows to E4M3 with a
  per-row scale (rounded to the model dtype, then kept fp32): this rank's
  V'/tp rows when L1b is on, else this rank's full 124,160-row shard. The
  target keeps its BF16 lm_head (the draft's BF16 copy is still dropped by
  load_eagle_model's rebinding, so the net cost is the FP8 copy only,
  ~0.30 GiB/rank full-vocab). Kernel: Marlin W8A16 (apply_fp8_marlin_linear,
  FP8 weights, BF16 activations) first; if it fails to build or its load-time
  self-check vs the BF16 rows fails, CUTLASS W8A8 (scaled_fp8_quant per-token
  dynamic + cutlass_scaled_mm, per-row weight scale). If neither works it
  falls back (TP-agreed) to L1b BF16 rows or the stock head.
  Uses: get_top_tokens (local argmax, full or reduced rows) and, full-vocab
  only, compute_logits (all-gathered FP8 logits for probabilistic drafting or
  greedy without use_local_argmax_reduction). With L1b + L1b' together the
  reduced rows are stored FP8 only.

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
    "L1b is FR-Spec arXiv:2502.14856 style, L1b' reuses v0.30 Marlin FP8 / "
    "CUTLASS scaled_mm ops)"
)
ENV = "VLLM_QWEN38_DRAFT_VOCAB"
FP8_ENV = "VLLM_QWEN38_DRAFT_HEAD_FP8"

HEADER = f"""\
# R11-OVERLAY
# base_image_digest: {BASE_IMAGE_DIGEST}
# upstream_file: {UPSTREAM_FILE}
# upstream_file_sha256: {UPSTREAM_FILE_SHA256}
# upstream_PR: {UPSTREAM_PR}
# generator: docker/v030/apply_mtp_overlay.py (do not hand-edit)
# features: L1a get_top_tokens via LocalArgmaxMixin (always on);
#   L1b reduced-vocab draft head, opt-in via {ENV}=<json>, default off;
#   L1b' draft-only FP8 lm_head copy, opt-in via {FP8_ENV}=1, default off
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

# L1b reduced-vocab and L1b' FP8 draft heads (recipe overlay). Default off.
_DRAFT_VOCAB_ENV = "{ENV}"
_DRAFT_VOCAB_MIN_IDS = 1024
_DRAFT_HEAD_FP8_ENV = "{FP8_ENV}"
_FP8_MAX = 448.0  # torch.finfo(torch.float8_e4m3fn).max


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


def _draft_head_fp8_kernels(value: str) -> tuple[str, ...]:
    \"\"\"FP8 draft-head kernels to try, in order, for a {FP8_ENV} value.\"\"\"
    v = value.strip().lower()
    if v in ("", "0", "off", "false", "no"):
        return ()
    if v in ("1", "on", "true", "yes", "auto"):
        return ("marlin", "w8a8")
    if v in ("marlin", "w8a16"):
        return ("marlin",)
    if v in ("w8a8", "cutlass"):
        return ("w8a8",)
    raise ValueError(f"{{_DRAFT_HEAD_FP8_ENV}}={{value!r}}: use 1|auto, marlin or w8a8")


def _quantize_rows_fp8(
    w: torch.Tensor, chunk: int = 8192
) -> tuple[torch.Tensor, torch.Tensor]:
    \"\"\"Per-row symmetric E4M3: w[i] ~= q[i] * scale[i] (scale fp32, [N]).

    The scale is rounded to w.dtype first, so Marlin (which keeps scales in
    the activation dtype) and CUTLASS (fp32 scales) dequantize identically.
    Chunked to bound the fp32 transient.
    \"\"\"
    q = torch.empty(w.shape, dtype=torch.float8_e4m3fn, device=w.device)
    scale = torch.empty(w.shape[0], dtype=torch.float32, device=w.device)
    for s in range(0, w.shape[0], chunk):
        blk = w[s : s + chunk].float()
        sc = (blk.abs().amax(dim=1) / _FP8_MAX).to(w.dtype).float()
        sc = torch.where(sc > 0, sc, torch.ones_like(sc))  # all-zero rows
        q[s : s + chunk] = (
            (blk / sc[:, None]).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
        )
        scale[s : s + chunk] = sc
    return q, scale


class _Fp8DraftHead:
    \"\"\"Draft-only FP8 copy of lm_head rows [N, K]; logits(h) -> [M, N].

    kernel "marlin": W8A16, FP8 weights dequantized in the Marlin GEMM,
    activations stay BF16. kernel "w8a8": per-token dynamic FP8 activations
    + CUTLASS scaled_mm with the per-row weight scale.
    \"\"\"

    def __init__(self, rows: torch.Tensor, kernel: str) -> None:
        q, scale = _quantize_rows_fp8(rows)
        self.kernel = kernel
        self.n, self.k = rows.shape
        self.dtype = rows.dtype
        self.nbytes = q.numel() * q.element_size() + scale.numel() * 4
        if kernel == "marlin":
            from vllm.model_executor.layers.quantization.utils import (
                marlin_utils_fp8 as mu,
            )

            layer = nn.Module()
            layer.output_size_per_partition = self.n
            layer.input_size_per_partition = self.k
            layer.orig_dtype = rows.dtype
            layer.weight = nn.Parameter(q, requires_grad=False)
            layer.weight_scale = nn.Parameter(scale, requires_grad=False)
            mu.prepare_fp8_layer_for_marlin(layer, size_k_first=False)
            self.weight = layer.weight.data
            self.scale = layer.weight_scale.data
            self.workspace = layer.workspace
            self._marlin = mu.apply_fp8_marlin_linear
        elif kernel == "w8a8":
            from vllm import _custom_ops as ops

            self._ops = ops
            self.weight = q.t()  # [K, N] column-major, as cutlass_scaled_mm wants
            self.scale = scale.view(-1, 1)
        else:
            raise ValueError(f"unknown FP8 draft-head kernel {{kernel!r}}")

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        if self.kernel == "marlin":
            return self._marlin(
                h, self.weight, self.scale, self.workspace, self.n, self.k, None
            )
        hq, hs = self._ops.scaled_fp8_quant(h, use_per_token_if_dynamic=True)
        return self._ops.cutlass_scaled_mm(hq, self.weight, hs, self.scale, self.dtype)

    def check(self, rows: torch.Tensor) -> float:
        \"\"\"Raise unless logits() matches the BF16 rows to FP8 accuracy.\"\"\"
        h = torch.randn(4, self.k, dtype=rows.dtype, device=rows.device)
        ref = F.linear(h, rows).float()
        got = self.logits(h).float()
        if got.shape != ref.shape or not bool(torch.isfinite(got).all()):
            raise RuntimeError(f"bad output {{tuple(got.shape)}}")
        err = float((got - ref).abs().max() / ref.abs().max().clamp(min=1e-6))
        if err > 0.1:
            raise RuntimeError(f"max rel. error {{err:.3f}} > 0.1 vs BF16")
        return err


def _build_fp8_draft_head(
    rows: torch.Tensor, kernels: tuple[str, ...]
) -> tuple[_Fp8DraftHead | None, str]:
    \"\"\"First kernel that builds and passes its self-check, else (None, why).\"\"\"
    errs = []
    for kernel in kernels:
        try:
            head = _Fp8DraftHead(rows, kernel)
            head.check(rows)
            return head, ""
        except Exception as e:  # noqa: BLE001 - try the next kernel
            errs.append(f"{{kernel}}: {{type(e).__name__}}: {{e}}")
    return None, "; ".join(errs)


def _remap_ignored_layers(
"""

L1B_INIT_OLD = """\
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
"""
L1B_INIT_NEW = """\
        self.logits_processor = LogitsProcessor(config.vocab_size)
        # L1b/L1b': filled by load_weights only when a draft head is enabled.
        self.register_buffer("draft_vocab_weight", None, persistent=False)
        self.register_buffer("draft_vocab_ids", None, persistent=False)
        self.draft_head_fp8: _Fp8DraftHead | None = None
        self.draft_head_full = False  # FP8 rows are this rank's whole shard
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
        ids = self.draft_vocab_ids
        if ids is None:
            return super().get_top_tokens(hidden_states)
        # L1b/L1b': this rank's rows (V'/tp or the whole shard, BF16 or FP8),
        # local argmax, then an [M, 2] all-gather.
        logits = self._draft_head_logits(hidden_states)
        local_idx = logits.argmax(dim=-1, keepdim=True)
        local_val = logits.gather(-1, local_idx).squeeze(-1)
        top = ids[local_idx.squeeze(-1)]
        tp_size = get_tensor_model_parallel_world_size()
        if tp_size == 1:
            return top
        # float32 carries ids < 2**24 exactly (vocab is 248,320).
        pair = torch.stack([local_val.float(), top.float()], dim=-1)
        gathered = tensor_model_parallel_all_gather(pair, dim=-1)
        gathered = gathered.view(hidden_states.shape[0], tp_size, 2)
        best = gathered[:, :, 0].argmax(dim=-1, keepdim=True)
        return gathered[:, :, 1].gather(-1, best).squeeze(-1).to(torch.int64)

    def _draft_head_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.draft_head_fp8 is not None:
            return self.draft_head_fp8.logits(hidden_states)
        return F.linear(hidden_states, self.draft_vocab_weight)

    def _draft_vocab_plan(self) -> list[int]:
        \"\"\"Ids this rank keeps for the L1b head; raises if L1b cannot run.\"\"\"
        path = os.environ.get(_DRAFT_VOCAB_ENV, "").strip()
        spec = self.vllm_config.speculative_config
        if spec is None or not spec.use_local_argmax_reduction:
            raise ValueError("needs speculative_config use_local_argmax_reduction")
        if spec.draft_sample_method == "probabilistic":
            raise ValueError("incompatible with draft_sample_method=probabilistic")
        self._check_draft_head_supported()
        ids = _load_draft_vocab(path, self.config.vocab_size)
        return _draft_vocab_rank_slice(
            ids,
            get_tensor_model_parallel_world_size(),
            get_tensor_model_parallel_rank(),
        )

    def _check_draft_head_supported(self) -> None:
        if get_pp_group().world_size != 1:
            raise ValueError("pipeline parallel is not supported")
        if self.config.tie_word_embeddings:
            raise ValueError("tied embeddings are not supported")
        lp = self.logits_processor
        if lp.scale != 1.0 or lp.soft_cap is not None or lp.logits_as_input:
            raise ValueError("logits scale/soft_cap is not supported")

    def _full_head_rows(self) -> tuple[torch.Tensor, range]:
        \"\"\"This rank's whole lm_head shard (L1b' without L1b); raises if unusable.\"\"\"
        self._check_draft_head_supported()
        si = self.lm_head.shard_indices
        if si.num_org_vocab_padding or si.num_added_elements_padded:
            raise ValueError("padded or added-vocab lm_head shard is not supported")
        w = self.lm_head.weight.data
        if w.dtype not in (torch.bfloat16, torch.float16):
            raise ValueError(f"lm_head dtype {w.dtype} is not BF16/FP16")
        if w.shape[0] != si.num_org_elements:
            raise ValueError(f"lm_head shard has {w.shape[0]} rows")
        return w, range(si.org_vocab_start_index, si.org_vocab_end_index)

    @staticmethod
    def _tp_agree(ok: bool, err: str) -> tuple[bool, str]:
        \"\"\"MIN over TP ranks on the CPU group; every rank must call it.\"\"\"
        if get_tensor_model_parallel_world_size() > 1:
            flag = torch.tensor([1 if ok else 0], dtype=torch.int32)
            torch.distributed.all_reduce(
                flag,
                op=torch.distributed.ReduceOp.MIN,
                group=get_tp_group().cpu_group,
            )
            if ok and int(flag.item()) == 0:
                return False, "another TP rank could not build it"
        return ok, err

    def _finish_draft_head(
        self,
        local_ids: list[int] | None,
        rows: torch.Tensor | None,
        err: str,
        requested: bool,
        fp8_kernels: tuple[str, ...],
        fp8_err: str,
    ) -> None:
        \"\"\"Install L1b and/or L1b' on every TP rank, or on none (fail closed).

        Every TP rank joins both agreements even with the envs unset, so a
        rank that lacks them makes the others fall back instead of hanging.
        \"\"\"
        # L1b: the reduced BF16 rows captured while loading.
        head_dtype = self.logits_processor.head_dtype
        ok = local_ids is not None and rows is not None
        if ok and head_dtype not in (None, rows.dtype):
            ok, err = False, f"head_dtype {head_dtype} != lm_head {rows.dtype}"
        ok, err = self._tp_agree(ok, err)
        if not ok:
            if requested:
                logger.warning(
                    "%s set but reduced draft head disabled (%s); using the "
                    "full lm_head for drafting.",
                    _DRAFT_VOCAB_ENV,
                    err or "lm_head.weight not seen while loading",
                )
            local_ids = rows = None
        else:
            # Read the device only here: on a PP>1 non-last rank lm_head is a
            # PPMissingLayer without .weight (the plan already failed closed).
            rows = rows.to(self.lm_head.weight.device)

        # L1b': FP8 copy of the reduced rows, else of this rank's full shard.
        fp8_requested = bool(fp8_kernels or fp8_err)
        head, full = None, False
        if fp8_kernels:
            try:
                if rows is None:
                    rows, local_ids = self._full_head_rows()
                    full = True
                if head_dtype not in (None, rows.dtype):
                    raise ValueError(f"head_dtype {head_dtype} != lm_head {rows.dtype}")
                head, fp8_err = _build_fp8_draft_head(rows, fp8_kernels)
            except Exception as e:  # fail closed to BF16
                fp8_err = f"{type(e).__name__}: {e}"
        fp8_ok, fp8_err = self._tp_agree(head is not None, fp8_err)
        if not fp8_ok:
            head = None
            if fp8_requested:
                logger.warning(
                    "%s set but FP8 draft head disabled (%s); drafting with "
                    "BF16 rows.",
                    _DRAFT_HEAD_FP8_ENV,
                    fp8_err or "not requested on every rank",
                )
            if full:  # never keep the draft's own BF16 shard alive
                local_ids = rows = None
        if local_ids is None or rows is None:
            return

        device = rows.device
        if isinstance(local_ids, range):
            ids = torch.arange(
                local_ids.start, local_ids.stop, dtype=torch.int64, device=device
            )
        else:
            ids = torch.tensor(local_ids, dtype=torch.int64, device=device)
        self.draft_vocab_ids = ids
        if head is not None:
            self.draft_head_fp8, self.draft_head_full = head, full
            what = f"FP8 {head.kernel}"
            mib = head.nbytes / 2**20
        else:
            self.draft_vocab_weight = rows.contiguous()
            what = f"{rows.dtype}"
            mib = rows.numel() * rows.element_size() / 2**20
        logger.info(
            "Draft head (%s): %d rows on this rank (ids %d..%d), %s, %.1f MiB; "
            "the target keeps its BF16 lm_head for verify.",
            "L1b'" if full else ("L1b+L1b'" if head is not None else "L1b"),
            len(local_ids),
            local_ids[0],
            local_ids[-1],
            what,
            mib,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        draft_vocab = bool(os.environ.get(_DRAFT_VOCAB_ENV, "").strip())
        fp8_kernels: tuple[str, ...] = ()
        fp8_err = ""
        try:
            fp8_kernels = _draft_head_fp8_kernels(
                os.environ.get(_DRAFT_HEAD_FP8_ENV, "")
            )
        except ValueError as e:
            fp8_err = str(e)
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
        requested = draft_vocab or fp8_kernels or fp8_err
        if requested or get_tensor_model_parallel_world_size() > 1:
            self._finish_draft_head(
                local_ids,
                captured.get("rows"),
                err,
                draft_vocab,
                fp8_kernels,
                fp8_err,
            )
        return loaded


__all__ = ["Qwen4ExpMTP", "Qwen4ExpMultiTokenPredictor"]
"""

L1B_LOGITS_OLD = """\
    def compute_logits(
        self, hidden_states: torch.Tensor, spec_step_idx: int = 0
    ) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)
"""
L1B_LOGITS_NEW = """\
    def compute_logits(
        self, hidden_states: torch.Tensor, spec_step_idx: int = 0
    ) -> torch.Tensor | None:
        if self.draft_head_fp8 is not None and self.draft_head_full:
            # L1b': full-vocab FP8 draft logits, gathered like the stock head
            # (probabilistic drafting, or greedy without local argmax).
            logits = self.draft_head_fp8.logits(hidden_states)
            if get_tensor_model_parallel_world_size() > 1:
                logits = self.logits_processor._gather_logits(logits)
            if logits is not None:
                logits = logits[..., : self.logits_processor.org_vocab_size]
            return logits
        return self.logits_processor(self.lm_head, hidden_states)
"""

L1B_HUNKS = (
    (L1B_STDLIB_OLD, L1B_STDLIB_NEW),
    (L1B_TORCH_OLD, L1B_TORCH_NEW),
    (L1B_DIST_OLD, L1B_DIST_NEW),
    (L1B_HELPERS_OLD, L1B_HELPERS_NEW),
    (L1B_INIT_OLD, L1B_INIT_NEW),
    (L1B_LOGITS_OLD, L1B_LOGITS_NEW),
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
            f"!= {UPSTREAM_FILE_SHA256}; rebase the L1a/L1b/L1b' hunks"
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
