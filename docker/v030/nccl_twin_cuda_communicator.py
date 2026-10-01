# R11-OVERLAY
# base_image_digest: sha256:4864d46625cbc3307623e29ac742030655e27249feba7b97ec925ce4cc4dfb56
# upstream_file: vllm/distributed/device_communicators/cuda_communicator.py
# upstream_file_sha256: 102d95c67ec798cf352b817ba92e8abc9da37f30d1bbdf3aab45d3927b627b93
# upstream_PR: none (local; port of the DeepSeek-V4.1 sibling's docker/patch/nccl_eager_twin.py, DSV41_NCCL_EAGER_TWIN)
# generator: docker/v030/apply_nccl_twin_overlay.py (do not hand-edit)
# features: L2a NCCL_GRAPH_MIXING_SUPPORT=0 + eager-only twin TP communicator,
#   opt-in via VLLM_QWEN38_NCCL_TWIN=1 (default off; source docker/v030/nccl_twin.py)
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch
from torch.distributed import ProcessGroup

import vllm.envs as envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.distributed.device_communicators.all_reduce_utils import (
    NCCL_SYMM_MEM_ALL_REDUCE_CONFIG,
    should_nccl_symm_mem_ag_rs,
    should_nccl_symm_mem_allreduce,
)
from vllm.distributed.device_communicators.pynccl import register_nccl_symmetric_ops
from vllm.distributed.device_communicators.pynccl_allocator import (
    is_symmetric_memory_enabled,
)
from vllm.logger import init_logger
from vllm.platforms import current_platform

from ..utils import StatelessProcessGroup
from .aiter_custom_all_reduce import AiterCustomAllreduce
from .base_device_communicator import DeviceCommunicatorBase

logger = init_logger(__name__)


class CudaCommunicator(DeviceCommunicatorBase):
    def __init__(
        self,
        cpu_group: ProcessGroup,
        device: torch.device | None = None,
        device_group: ProcessGroup | None = None,
        unique_name: str = "",
        global_ranks: list[int] | None = None,
        global_world_size: int | None = None,
        tcp_store_group: StatelessProcessGroup | None = None,
        use_all2all: bool = False,
    ):
        super().__init__(
            cpu_group,
            device,
            device_group,
            unique_name,
            global_ranks,
            global_world_size,
            use_all2all=use_all2all,
        )
        # Match the group name exactly so ETP does not enable TP-only backends.
        if unique_name.split(":")[0] != "tp":
            # custom allreduce or torch symm mem can be used only by tp
            use_custom_allreduce = False
            use_torch_symm_mem = False
            use_flashinfer_allreduce = False
            use_flashinfer_pcie_ipc_allreduce = False
            use_aiter_allreduce = False
        else:
            from vllm.distributed.parallel_state import _ENABLE_CUSTOM_ALL_REDUCE

            use_custom_allreduce = _ENABLE_CUSTOM_ALL_REDUCE
            use_torch_symm_mem = envs.VLLM_ALLREDUCE_USE_SYMM_MEM
            # FlashInfer all-reduce does not provide a fixed reduction order.
            use_flashinfer_allreduce = (
                envs.VLLM_ALLREDUCE_USE_FLASHINFER and not envs.VLLM_BATCH_INVARIANT
            )
            use_flashinfer_pcie_ipc_allreduce = (
                envs.VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC
                and not envs.VLLM_BATCH_INVARIANT
            )
            use_aiter_allreduce = use_custom_allreduce and bool(
                rocm_aiter_ops.is_custom_all_reduce_enabled()
            )

        self.use_custom_allreduce = use_custom_allreduce
        self.use_torch_symm_mem = use_torch_symm_mem
        self.use_flashinfer_allreduce = use_flashinfer_allreduce
        self.use_flashinfer_pcie_ipc_allreduce = use_flashinfer_pcie_ipc_allreduce
        self.use_aiter_allreduce = use_aiter_allreduce

        # lazy import to avoid documentation build error
        from vllm.distributed.device_communicators.custom_all_reduce import (
            CustomAllreduce,
        )
        from vllm.distributed.device_communicators.flashinfer_all_reduce import (
            FlashInferAllReduce,
        )
        from vllm.distributed.device_communicators.flashinfer_pcie_ipc_all_reduce import (  # noqa: E501
            FlashInferPcieIpcAllReduce,
        )
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
        from vllm.distributed.device_communicators.quick_all_reduce import (
            QuickAllReduce,
        )
        from vllm.distributed.device_communicators.symm_mem import SymmMemCommunicator

        self.pynccl_comm: PyNcclCommunicator | None = None
        if self.world_size > 1:
            self.pynccl_comm = PyNcclCommunicator(
                group=self.cpu_group if tcp_store_group is None else tcp_store_group,
                device=self.device,
            )
            if is_symmetric_memory_enabled():
                register_nccl_symmetric_ops(self.pynccl_comm)
            # L2a (recipe overlay): eager-twin router; the stock comm unless every rank armed
            # VLLM_QWEN38_NCCL_TWIN=1 (see _nccl_twin below).
            self.pynccl_comm = _nccl_twin.attach(self, tcp_store_group, PyNcclCommunicator)

        self.ca_comm: CustomAllreduce | None = None
        self.qr_comm: QuickAllReduce | None = None
        self.symm_mem_comm: SymmMemCommunicator | None = None
        self.fi_ar_comm: FlashInferAllReduce | None = None
        self.fi_pcie_ipc_ar_comm: FlashInferPcieIpcAllReduce | None = None
        self.aiter_ar_comm: AiterCustomAllreduce | None = None
        self.use_aiter_ag_rs: bool = False

        if use_torch_symm_mem and current_platform.is_cuda():
            self.symm_mem_comm = SymmMemCommunicator(
                group=self.cpu_group,
                device=self.device,
            )

        if self.use_flashinfer_allreduce and self.world_size > 1:
            self.fi_ar_comm = FlashInferAllReduce(
                group=self.cpu_group,
                device=self.device,
            )

        if (
            self.use_flashinfer_pcie_ipc_allreduce
            and self.world_size > 1
            and self.device_group is not None
        ):
            self.fi_pcie_ipc_ar_comm = FlashInferPcieIpcAllReduce(
                group=self.device_group,
                tune_group=self.cpu_group,
                device=self.device,
            )

        if self.use_aiter_allreduce and self.world_size > 1:
            self.aiter_ar_comm = AiterCustomAllreduce(
                group=self.cpu_group,
                device=self.device,
            )

        if use_custom_allreduce and self.aiter_ar_comm is None and self.world_size > 1:
            # Initialize a custom fast all-reduce implementation.
            self.ca_comm = CustomAllreduce(
                group=self.cpu_group,
                device=self.device,
                symm_mem_enabled=(
                    self.symm_mem_comm is not None and not self.symm_mem_comm.disabled
                ),
            )

        # AITER custom all-gather/reduce-scatter DP-attention dispatch/combine
        if (
            "dp" in unique_name
            and self.world_size in (2, 4, 8)
            and current_platform.is_rocm()
            and rocm_aiter_ops.is_custom_all_reduce_enabled()
        ):
            self.aiter_ar_comm = AiterCustomAllreduce(
                group=self.cpu_group,
                device=self.device,
            )
            if self.aiter_ar_comm.disabled:
                self.aiter_ar_comm = None
            else:
                self.use_aiter_ag_rs = True

        if use_custom_allreduce and self.world_size > 1 and current_platform.is_rocm():
            # Initialize a custom quick all-reduce implementation for AMD.
            # Quick reduce is designed as a complement to custom allreduce
            # (vLLM's or AITER's), so it is initialized for either backend.
            # Based on quickreduce (https://github.com/mk1-project/quickreduce).
            # On ROCm, 'use_custom_allreduce==True' means it must currently be
            # an MI300 series.
            self.qr_comm = QuickAllReduce(group=self.cpu_group, device=self.device)

        if self.world_size > 1:
            self._log_all_reduce_backend_selection()

        if self.use_all2all:
            if self.all2all_backend in ("naive", "allgather_reducescatter"):
                from .all2all import AgRsAll2AllManager

                self.all2all_manager = AgRsAll2AllManager(
                    self.cpu_group, tcp_store_group
                )
            elif self.all2all_backend == "deepep_high_throughput":
                from .all2all import DeepEPHTAll2AllManager

                self.all2all_manager = DeepEPHTAll2AllManager(
                    self.cpu_group, tcp_store_group
                )
            elif self.all2all_backend == "deepep_low_latency":
                from .all2all import DeepEPLLAll2AllManager

                self.all2all_manager = DeepEPLLAll2AllManager(
                    self.cpu_group, tcp_store_group
                )
            elif self.all2all_backend in (
                "mori_high_throughput",
                "mori_low_latency",
            ):
                from .all2all import MoriAll2AllManager

                self.all2all_manager = MoriAll2AllManager(
                    self.cpu_group, self.all2all_backend
                )
            elif self.all2all_backend == "deepep_v2":
                from .all2all import DeepEPV2All2AllManager

                self.all2all_manager = DeepEPV2All2AllManager(
                    self.cpu_group,
                    tcp_store_group,
                    device_group=self.device_group,
                )
            elif self.all2all_backend == "nixl_ep":
                from .all2all import NixlEPAll2AllManager

                self.all2all_manager = NixlEPAll2AllManager(
                    self.cpu_group, tcp_store_group
                )
            elif (
                self.all2all_backend == "flashinfer_all2allv"
                or self.all2all_backend == "flashinfer_nvlink_two_sided"
            ):
                if self.all2all_backend == "flashinfer_all2allv":
                    logger.warning_once(
                        "'flashinfer_all2allv' is deprecated and has been renamed to"
                        "'flashinfer_nvlink_two_sided'. It will be removed in a future"
                        "release."
                    )
                from .all2all import FlashInferNVLinkTwoSidedManager

                self.all2all_manager = FlashInferNVLinkTwoSidedManager(
                    self.cpu_group, tcp_store_group
                )
            elif self.all2all_backend == "flashinfer_nvlink_one_sided":
                from .all2all import FlashInferNVLinkOneSidedManager

                self.all2all_manager = FlashInferNVLinkOneSidedManager(self.cpu_group)
            else:
                raise ValueError(f"Unknown all2all backend: {self.all2all_backend}")

            logger.info_once(
                "Using %s all2all manager.",
                self.all2all_manager.__class__.__name__,
                scope="global",
            )

    def _log_all_reduce_backend_selection(self) -> None:
        """Log the all-reduce backends that are active for this group.

        The dispatch chain in ``all_reduce`` tries backends in this order and
        falls through to the next one if the current backend rejects the
        input (size/dtype gates) or is disabled. The list of "enabled"
        backends below is the subset of potential backends that may be
        chosen at dispatch time for this group; the actual per-call choice
        depends on the input tensor.
        """
        all_potential_ar_backends = [
            "FLASHINFER_PCIE_IPC",
            "FLASHINFER",
            "NCCL_SYMM_MEM",
            "QUICK_REDUCE",
            "AITER_CUSTOM",
            "CUSTOM",
            "SYMM_MEM",
            "PYNCCL",
        ]
        enabled_ar_backends: list[str] = []
        if (
            self.fi_pcie_ipc_ar_comm is not None
            and not self.fi_pcie_ipc_ar_comm.disabled
        ):
            enabled_ar_backends.append("FLASHINFER_PCIE_IPC")
        if self.fi_ar_comm is not None and not self.fi_ar_comm.disabled:
            enabled_ar_backends.append("FLASHINFER")
        # Mirror the static preconditions of `should_nccl_symm_mem_allreduce`:
        # VLLM_BATCH_INVARIANT off, NCCL symm mem enabled, world_size meets
        # min_world_size, and world_size either has a tuned entry in
        # `custom_ar_preferred_ranges` or is greater than
        # `always_use_above_world_size`. World sizes that fail the latter (e.g.
        # 5/6/7 with the default config) never dispatch NCCL symm mem
        # regardless of input. The per-tensor-size check inside the function
        # stays as a runtime decision.
        nccl_symm_ws_ok = self.world_size >= NCCL_SYMM_MEM_ALL_REDUCE_CONFIG[
            "min_world_size"
        ] and (
            self.world_size
            in NCCL_SYMM_MEM_ALL_REDUCE_CONFIG["custom_ar_preferred_ranges"]
            or self.world_size
            > NCCL_SYMM_MEM_ALL_REDUCE_CONFIG["always_use_above_world_size"]
        )
        if (
            self.pynccl_comm is not None
            and not self.pynccl_comm.disabled
            and is_symmetric_memory_enabled()
            and not envs.VLLM_BATCH_INVARIANT
            and nccl_symm_ws_ok
        ):
            enabled_ar_backends.append("NCCL_SYMM_MEM")
        if self.qr_comm is not None and not self.qr_comm.disabled:
            enabled_ar_backends.append("QUICK_REDUCE")
        if (
            self.use_aiter_allreduce
            and self.aiter_ar_comm is not None
            and not self.aiter_ar_comm.disabled
        ):
            enabled_ar_backends.append("AITER_CUSTOM")
        if self.ca_comm is not None and not self.ca_comm.disabled:
            enabled_ar_backends.append("CUSTOM")
        if self.symm_mem_comm is not None and not self.symm_mem_comm.disabled:
            enabled_ar_backends.append("SYMM_MEM")
        if self.pynccl_comm is not None and not self.pynccl_comm.disabled:
            enabled_ar_backends.append("PYNCCL")

        logger.info_once(
            "Using %s all-reduce backends (in dispatch order) for group "
            "'%s' out of potential backends: %s.",
            "[" + ", ".join(f"'{b}'" for b in enabled_ar_backends) + "]",
            self.unique_name or "<unnamed>",
            "[" + ", ".join(f"'{b}'" for b in all_potential_ar_backends) + "]",
            scope="global",
        )

    def all_reduce(self, input_):
        fi_ar_comm = self.fi_ar_comm
        use_fi_ar = (
            fi_ar_comm is not None
            and not fi_ar_comm.disabled
            and fi_ar_comm.should_use_fi_ar(input_)
        )

        # since currently we perform copy input -> symm_input -> out-of-place AR
        # return symm_output, we don't need to check if input is symmetric
        if (
            self.pynccl_comm is not None
            and not use_fi_ar
            and should_nccl_symm_mem_allreduce(self.pynccl_comm.world_size, input_)
        ):
            out = torch.ops.vllm.all_reduce_symmetric_with_copy(input_)
            if out is not None:
                return out
        qr_comm = self.qr_comm
        if (
            qr_comm is not None
            and not qr_comm.disabled
            and qr_comm.should_quick_allreduce(input_)
        ):
            out = qr_comm.quick_all_reduce(input_)
            assert out is not None
            return out
        fi_pcie_ipc_ar_comm = self.fi_pcie_ipc_ar_comm
        if fi_pcie_ipc_ar_comm is not None and fi_pcie_ipc_ar_comm.should_use(input_):
            return fi_pcie_ipc_ar_comm.all_reduce(input_)
        if use_fi_ar:
            assert fi_ar_comm is not None
            out = fi_ar_comm.all_reduce(input_)
            assert out is not None
            return out
        aiter_ar_comm = self.aiter_ar_comm
        if (
            self.use_aiter_allreduce
            and aiter_ar_comm is not None
            and not aiter_ar_comm.disabled
            and aiter_ar_comm.should_custom_ar(input_)
        ):
            out = aiter_ar_comm.custom_all_reduce(input_)
            assert out is not None
            return out
        ca_comm = self.ca_comm
        if (
            ca_comm is not None
            and not ca_comm.disabled
            and ca_comm.should_custom_ar(input_)
        ):
            out = ca_comm.custom_all_reduce(input_)
            assert out is not None
            return out
        symm_mem_comm = self.symm_mem_comm
        if symm_mem_comm is not None and symm_mem_comm.should_use_symm_mem(input_):
            out = symm_mem_comm.all_reduce(input_)
            assert out is not None
            return out
        pynccl_comm = self.pynccl_comm
        if pynccl_comm is None or pynccl_comm.disabled:
            out = input_.clone()
            torch.distributed.all_reduce(out, group=self.device_group)
            return out
        assert pynccl_comm is not None
        out = pynccl_comm.all_reduce(input_)
        if out is None:
            # fall back to the default all-reduce using PyTorch.
            # this usually happens during testing.
            # when we run the model, allreduce only happens for the TP
            # group, where we always have either custom allreduce or pynccl.
            out = input_.clone()
            torch.distributed.all_reduce(out, group=self.device_group)
        return out

    def custom_all_gather(self, input_: torch.Tensor) -> torch.Tensor | None:
        ca_comm = self.ca_comm
        if ca_comm is None:
            return None
        return ca_comm.custom_all_gather(input_.contiguous())

    def custom_reduce_scatter(self, input_: torch.Tensor) -> torch.Tensor | None:
        ca_comm = self.ca_comm
        if ca_comm is None:
            return None
        return ca_comm.custom_reduce_scatter(input_.contiguous())

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        # Route uniform dim-0 all-gathers through NVLS symmetric memory when
        # enabled (mirrors reduce_scatter); otherwise fall back to the
        # PyNccl/base-class all-gather. Sequence parallelism's
        # gather-before-GEMM uses dim=0 with tp-aligned (uniform) shards.
        if dim < 0:
            dim += input_.dim()
        if dim == 0 and should_nccl_symm_mem_ag_rs():
            return self._all_gather_symm_mem(input_.contiguous())

        pynccl_comm = self.pynccl_comm
        if pynccl_comm is None or pynccl_comm.disabled:
            return super().all_gather(input_, dim)

        # On ROCm, the base-class all_gather (all_gather_into_tensor) is faster
        # than the manual pynccl + torch.empty + movedim + reshape path below,
        # which adds a per-call output allocation and (for dim != 0) an extra
        # copy on every step. This is on the hot path for TP forward passes, so
        # keep ROCm on the base-class collective to avoid a decode regression.
        if current_platform.is_rocm():
            return super().all_gather(input_, dim)

        input_size = input_.size()
        output_size = (input_size[0] * self.world_size,) + input_size[1:]
        output_tensor = torch.empty(
            output_size, dtype=input_.dtype, device=input_.device
        )
        pynccl_comm.all_gather(output_tensor, input_.contiguous())
        output_tensor = output_tensor.reshape((self.world_size,) + input_size)
        output_tensor = output_tensor.movedim(0, dim)
        return output_tensor.reshape(
            input_size[:dim]
            + (self.world_size * input_size[dim],)
            + input_size[dim + 1 :]
        )

    def reduce_scatter(self, input_: torch.Tensor, dim: int = -1):
        world_size = self.world_size
        pynccl_comm = self.pynccl_comm
        assert pynccl_comm is not None
        if dim < 0:
            # Convert negative dim to positive.
            dim += input_.dim()

        # Note: This will produce an incorrect answer if we don't make
        # the input_tensor contiguous. Possible bug in reduce_scatter_tensor?
        input_tensor = input_.movedim(0, dim).contiguous()

        assert input_tensor.shape[0] % world_size == 0
        chunk_size = input_tensor.shape[0] // world_size
        output_shape = (chunk_size,) + input_tensor.shape[1:]

        if should_nccl_symm_mem_ag_rs():
            output = self._reduce_scatter_symm_mem(input_tensor)
        else:
            output = torch.empty(
                output_shape, dtype=input_tensor.dtype, device=input_tensor.device
            )
            pynccl_comm.reduce_scatter(output, input_tensor)

        # Reshape before returning
        return output.movedim(0, dim).contiguous()

    def reduce_scatterv(
        self, input_: torch.Tensor, dim: int = -1, sizes: list[int] | None = None
    ):
        world_size = self.world_size
        pynccl_comm = self.pynccl_comm
        assert pynccl_comm is not None
        if dim < 0:
            # Convert negative dim to positive.
            dim += input_.dim()

        # 'sizes' is not needed if all inputs in the same group have the same
        # shape
        if sizes is not None and all(s == sizes[0] for s in sizes):
            sizes = None

        # Note: This will produce an incorrect answer if we don't make
        # the input_tensor contiguous. Possible bug in reduce_scatter_tensor?
        input_tensor = input_.movedim(0, dim).contiguous()

        if sizes is not None:
            assert len(sizes) == world_size, f"{len(sizes)} == {world_size}"
            assert input_tensor.shape[0] == sum(sizes)
            chunk_size = sizes[self.rank_in_group]
        else:
            assert input_tensor.shape[0] % world_size == 0
            chunk_size = input_tensor.shape[0] // world_size
        output_shape = (chunk_size,) + input_tensor.shape[1:]

        if self._can_use_aiter_ag_rs(sizes):
            aiter_comm = self.aiter_ar_comm
            assert aiter_comm is not None
            if aiter_comm.should_custom_rs(input_tensor, dim=0):
                output = torch.empty(
                    output_shape, dtype=input_tensor.dtype, device=input_tensor.device
                )
                aiter_comm.custom_reduce_scatter(input_tensor, output, dim=0)
                return output.movedim(0, dim).contiguous()

        # Symmetric memory is only used when all ranks have uniform sizes.
        # ncclCommWindowRegister is collective: asymmetric pool allocations
        # from variable per-rank sizes cause deadlocks.
        use_symm_mem = sizes is None and should_nccl_symm_mem_ag_rs()
        if use_symm_mem:
            output = self._reduce_scatter_symm_mem(input_tensor)
        else:
            output = torch.empty(
                output_shape, dtype=input_tensor.dtype, device=input_tensor.device
            )
            use_deterministic_rs = envs.VLLM_BATCH_INVARIANT and world_size > 2
            if use_deterministic_rs:
                # Reduce to a fixed root (0) for determinism
                reduced = torch.empty_like(input_tensor)
                sizes = sizes if sizes else [chunk_size] * world_size
                pynccl_comm.reduce(reduced, input_tensor, root=0)
                pynccl_comm.scatter(output, reduced, sizes, root=0)
            elif sizes is not None and sizes.count(sizes[0]) != len(sizes):
                pynccl_comm.reduce_scatterv(output, input_tensor, sizes=sizes)
            else:
                pynccl_comm.reduce_scatter(output, input_tensor)

        # Reshape before returning
        return output.movedim(0, dim).contiguous()

    def _get_symm_scratch(
        self,
        role: str,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Persistent, pre-registered NCCL symmetric-memory scratch buffer.

        Allocating a fresh symm tensor per collective pays the
        ``nccl_symm_mem_context`` snapshot + window-registration scan on every
        call (~0.5 ms/RS+AG pair, dwarfing the NVLS transfer itself). Instead we
        allocate once per ``(role, shape, dtype)``, register once, and reuse.

        Safe for serial (eager) sequence parallelism: each collective's result
        is consumed on the same stream before the next same-role collective
        reuses the buffer. Distinct roles (e.g. ``rs_in`` vs ``ag_out``, both
        full-size) get distinct buffers so a reduce-scatter input copy never
        clobbers a still-live all-gather output.
        """
        from vllm.distributed.device_communicators.pynccl_allocator import (
            nccl_symm_mem_context,
        )

        pynccl_comm = self.pynccl_comm
        assert pynccl_comm is not None
        cache = self.__dict__.setdefault("_symm_scratch_bufs", {})
        key = (role, tuple(shape), dtype)
        buf = cache.get(key)
        if buf is None:
            with nccl_symm_mem_context(pynccl_comm):
                buf = torch.empty(shape, dtype=dtype, device=device)
            cache[key] = buf
        return buf

    def _reduce_scatter_symm_mem(
        self,
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """ReduceScatter using NCCL symmetric memory (NVLS).

        Only called for uniform-size reduce_scatter (variable sizes are
        guarded out by the caller to avoid asymmetric ncclCommWindowRegister).
        Uses persistent pre-registered scratch (see _get_symm_scratch).
        """
        from vllm.distributed.device_communicators.pynccl_allocator import (
            is_symmetric_memory_tensor,
        )

        pynccl_comm = self.pynccl_comm
        assert pynccl_comm is not None

        chunk = input_tensor.shape[0] // self.world_size
        output_shape = (chunk,) + tuple(input_tensor.shape[1:])

        symm_output = self._get_symm_scratch(
            "rs_out", output_shape, input_tensor.dtype, input_tensor.device
        )
        # NVLS reduce-scatter (LDMC) requires the input in symmetric memory.
        if is_symmetric_memory_tensor(input_tensor):
            symm_input = input_tensor
        else:
            symm_input = self._get_symm_scratch(
                "rs_in",
                tuple(input_tensor.shape),
                input_tensor.dtype,
                input_tensor.device,
            )
            symm_input.copy_(input_tensor)

        pynccl_comm.reduce_scatter(symm_output, symm_input)
        return symm_output

    def send(self, tensor: torch.Tensor, dst: int | None = None) -> None:
        """Sends a tensor to the destination rank in a blocking way"""
        """NOTE: `dst` is the local rank of the destination rank."""
        if dst is None:
            dst = (self.rank_in_group + 1) % self.world_size

        pynccl_comm = self.pynccl_comm
        if pynccl_comm is not None and not pynccl_comm.disabled:
            pynccl_comm.send(tensor, dst)
        else:
            torch.distributed.send(tensor, self.ranks[dst], self.device_group)

    def recv(
        self, size: torch.Size, dtype: torch.dtype, src: int | None = None
    ) -> torch.Tensor:
        """Receives a tensor from the source rank."""
        """NOTE: `src` is the local rank of the source rank."""
        if src is None:
            src = (self.rank_in_group - 1) % self.world_size

        tensor = torch.empty(size, dtype=dtype, device=self.device)
        pynccl_comm = self.pynccl_comm
        if pynccl_comm is not None and not pynccl_comm.disabled:
            pynccl_comm.recv(tensor, src)
        else:
            torch.distributed.recv(tensor, self.ranks[src], self.device_group)
        return tensor

    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        """Broadcast a tensor from source rank to all ranks."""
        if self.world_size == 1:
            return tensor

        pynccl_comm = self.pynccl_comm
        if pynccl_comm is not None and not pynccl_comm.disabled:
            pynccl_comm.broadcast(tensor, src)
            return tensor
        else:
            raise ValueError("No PyNCCL communicator found")

    def destroy(self):
        if self.pynccl_comm is not None:
            self.pynccl_comm.destroy()
            self.pynccl_comm = None
        if self.ca_comm is not None:
            self.ca_comm = None
        if self.aiter_ar_comm is not None:
            self.aiter_ar_comm.close()
            self.aiter_ar_comm = None
        if self.fi_ar_comm is not None:
            self.fi_ar_comm.destroy()
            self.fi_ar_comm = None
        if self.fi_pcie_ipc_ar_comm is not None:
            self.fi_pcie_ipc_ar_comm.destroy()
            self.fi_pcie_ipc_ar_comm = None
        if self.all2all_manager is not None:
            self.all2all_manager.destroy()
            self.all2all_manager = None  # type: ignore[assignment]

    def _can_use_aiter_ag_rs(self, sizes: list[int] | None) -> bool:
        """Whether the AITER custom AG/RS fast path may run for this collective.

        Requires:
        - uniform batches
        - FULL CUDAgraphs
        """
        if (
            not self.use_aiter_ag_rs
            or self.aiter_ar_comm is None
            or self.aiter_ar_comm.disabled
        ):
            return False
        if sizes is not None:
            return False

        from vllm.config.compilation import CUDAGraphMode
        from vllm.forward_context import get_forward_context

        try:
            ctx = get_forward_context()
        except AssertionError:
            return False
        if ctx.cudagraph_runtime_mode != CUDAGraphMode.FULL:
            return False
        bd = ctx.batch_descriptor
        return bd is not None and bd.uniform

    def suspend(self) -> None:
        if self.pynccl_comm is not None:
            self.pynccl_comm.suspend()

    def resume(self) -> None:
        if self.pynccl_comm is not None:
            self.pynccl_comm.resume()

    def checkpoint_prepare(self) -> None:
        # Only FlashInfer all-reduce and FlashInfer all2all are supported for now.
        from .flashinfer_all_reduce import checkpoint_prepare_fi_ar_workspaces

        checkpoint_prepare_fi_ar_workspaces(self.cpu_group)
        if self.all2all_manager is not None:
            self.all2all_manager.checkpoint_prepare()

    def checkpoint_restore(self) -> None:
        # Only FlashInfer all-reduce and FlashInfer all2all are supported for now.
        from .flashinfer_all_reduce import checkpoint_restore_fi_ar_workspaces

        checkpoint_restore_fi_ar_workspaces(self.cpu_group)
        if self.all2all_manager is not None:
            self.all2all_manager.checkpoint_restore()

    def all_gatherv(
        self,
        input_: torch.Tensor | list[torch.Tensor],
        dim: int = 0,
        sizes: list[int] | None = None,
    ):
        if dim != 0:
            raise NotImplementedError("only dim 0 all-gatherv is supported")
        world_size = self.world_size

        # 'sizes' is not needed if all inputs in the same group have the same
        # shape
        if sizes is not None and all(s == sizes[0] for s in sizes):
            sizes = None

        if self._can_use_aiter_ag_rs(sizes):
            aiter_comm = self.aiter_ar_comm
            assert aiter_comm is not None
            if isinstance(input_, torch.Tensor):
                if aiter_comm.should_custom_ag(input_):
                    out = aiter_comm.custom_all_gather(input_, dim=0)
                    if out is not None:
                        return out
            elif all(aiter_comm.should_custom_ag(inp) for inp in input_):
                outs = [aiter_comm.custom_all_gather(inp, dim=0) for inp in input_]
                if all(o is not None for o in outs):
                    return outs

        pynccl_comm = self.pynccl_comm
        assert pynccl_comm is not None and not pynccl_comm.disabled

        # Symmetric memory is only used when all ranks have uniform sizes.
        # ncclCommWindowRegister is collective: asymmetric pool allocations
        # from variable per-rank sizes cause deadlocks.
        if sizes is None and should_nccl_symm_mem_ag_rs():
            if isinstance(input_, torch.Tensor):
                return self._all_gather_symm_mem(input_)
            return self._all_gather_batched_symm_mem(input_)

        def _all_gather_single(input_: torch.Tensor, sizes: list[int] | None = None):
            input_size = input_.size()
            if sizes is not None:
                assert len(sizes) == world_size
                assert input_.shape[dim] == sizes[self.rank_in_group], (
                    f"{input_.shape[dim]} != {sizes[self.rank_in_group]}"
                )
                output_size = (sum(sizes),) + input_size[1:]
            else:
                output_size = (input_size[0] * world_size,) + input_size[1:]
            # Allocate output tensor.
            output_tensor = torch.empty(
                output_size, dtype=input_.dtype, device=input_.device
            )
            if sizes is not None:
                pynccl_comm.all_gatherv(output_tensor, input_, sizes=sizes)
            else:
                pynccl_comm.all_gather(output_tensor, input_)
            return output_tensor

        if isinstance(input_, torch.Tensor):
            return _all_gather_single(input_, sizes)

        output_list = []
        pynccl_comm.group_start()
        for inp in input_:
            output_list.append(_all_gather_single(inp, sizes=sizes))
        pynccl_comm.group_end()

        return output_list

    def _all_gather_symm_mem(self, input_: torch.Tensor) -> torch.Tensor:
        """AllGather a single tensor using NCCL symmetric memory (NVLS).

        Only the output needs to be in symmetric memory; NCCL does not
        require the AG input to be symmetrically allocated.
        """
        pynccl_comm = self.pynccl_comm
        assert pynccl_comm is not None

        out_size = (input_.size(0) * self.world_size,) + tuple(input_.size()[1:])
        # Persistent pre-registered scratch avoids the per-call symm-mem context
        # snapshot/registration overhead (see _get_symm_scratch).
        symm_output = self._get_symm_scratch(
            "ag_out", out_size, input_.dtype, input_.device
        )
        pynccl_comm.all_gather(symm_output, input_)
        return symm_output

    def _all_gather_batched_symm_mem(
        self, inputs: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        """AllGather a list of tensors using NCCL symmetric memory (NVLS).

        Uses group_start/group_end to batch the collectives.
        Only the output needs to be in symmetric memory (see
        _all_gather_symm_mem).
        """
        from vllm.distributed.device_communicators.pynccl_allocator import (
            nccl_symm_mem_context,
        )

        pynccl_comm = self.pynccl_comm
        assert pynccl_comm is not None
        world_size = self.world_size

        symm_outputs = []
        with nccl_symm_mem_context(pynccl_comm):
            for inp in inputs:
                out_size = (inp.size(0) * world_size,) + inp.size()[1:]
                symm_outputs.append(
                    torch.empty(out_size, dtype=inp.dtype, device=inp.device)
                )

        pynccl_comm.group_start()
        for symm_out, inp in zip(symm_outputs, inputs):
            pynccl_comm.all_gather(symm_out, inp)
        pynccl_comm.group_end()

        return symm_outputs

    def dispatch_router_logits(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]
    ):
        """
        Dispatch the hidden states and router logits to the appropriate device.
        This is a no-op in the base class.
        """

        assert self.all2all_manager is not None
        return self.all2all_manager.dispatch_router_logits(
            hidden_states,
            router_logits,
            is_sequence_parallel,
            extra_tensors,
        )

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]
    ):
        """
        Dispatch the hidden states and topk weights/ids to the appropriate device.
        This is a no-op in the base class.
        """
        assert self.all2all_manager is not None
        return self.all2all_manager.dispatch(
            hidden_states,
            topk_weights,
            topk_ids,
            is_sequence_parallel,
            extra_tensors=extra_tensors,
        )

    def combine(
        self, hidden_states: torch.Tensor, is_sequence_parallel: bool = False
    ) -> torch.Tensor:
        """
        Combine the hidden states and router logits from the appropriate device.
        This is a no-op in the base class.
        """
        assert self.all2all_manager is not None
        return self.all2all_manager.combine(
            hidden_states,
            is_sequence_parallel,
        )

    def batch_isend_irecv(self, p2p_ops: list):
        pynccl_comm = self.pynccl_comm
        if pynccl_comm is not None and not pynccl_comm.disabled:
            pynccl_comm.batch_isend_irecv(p2p_ops)
        else:
            raise ValueError("No PyNCCL communicator found")


# ---- L2a (recipe overlay): docker/v030/nccl_twin.py, embedded verbatim ----
def _load_nccl_twin():
    import sys
    import types

    name = "vllm_qwen38_nccl_twin"
    mod = sys.modules.get(name)
    if mod is None:
        mod = types.ModuleType(name)
        exec(compile(_NCCL_TWIN_SRC, "<docker/v030/nccl_twin.py>", "exec"), mod.__dict__)
        sys.modules[name] = mod
    return mod


_NCCL_TWIN_SRC = r'''"""L2a: NCCL graph mixing off, made safe by an eager-only twin communicator (v0.30.0).

VLLM_QWEN38_NCCL_TWIN=1 (default 0). This file is embedded verbatim in the generated overlay
docker/v030/nccl_twin_cuda_communicator.py (vllm/distributed/device_communicators/cuda_communicator.py),
which calls install() at import and attach() from CudaCommunicator.__init__. Top-level imports are
stdlib only, so the host tests import it without torch. Ported from the DeepSeek-V4.1 sibling's
docker/patch/nccl_eager_twin.py.

Why. With NCCL's default NCCL_GRAPH_MIXING_SUPPORT=1 (graphUsageMode 2) every collective captured
into a CUDA graph carries serialEvent wait/record nodes on NCCL's strong streams, and an eager launch
on a communicator that owns graph plans can queue behind the graph's host nodes (nccl 2.30.7
src/misc/strongstream.cc `mixing = graphUsageMode == 2`, src/enqueue.cc ncclLaunchPrepare).
S1.1 on v0.30.0 (evidence/s11-micro/gonogo.md): in-graph AR 5-160 KB is 14.4-19.3 us cheaper per call
with mixing off; booked c=1 0.89 ms/step, c=8 1.02 ms/step (GO).

Why not the env var alone. With mixing off NCCL does not support a non-captured collective launched
while a graph that uses the same communicator is outstanding (NCCL env docs). The serve launches
eager collectives (embedding all-reduce, logits / draft all-gathers outside the FULL decode graph)
while the target graph is still running (sibling flags.md NCCL_GRAPH_MIXING_SUPPORT=0 row).

What this does (only when armed).
- install() writes NCCL_GRAPH_MIXING_SUPPORT=0 into this process's environment. NCCL reads it once
  per process, at the first ncclCommInitRank (init.cc:2066, param cached), so it must run before any
  NCCL communicator exists: the overlay calls it when cuda_communicator is imported, i.e. before the
  first CudaCommunicator. If a communicator already existed, the write is ignored and mixing stays on
  (safe, no gain); the engagement audit catches that (nccl_twin_audit.py, check nccl-env).
- For the TP group ('tp:*') attach() builds a second PyNcclCommunicator (the eager twin) on the same
  CPU group right after the stock one, and pynccl_comm becomes a GraphEagerRouter: a call whose
  stream is capturing goes to the stock (graph) communicator, every other call goes to the twin. The
  graph communicator only ever runs captured collectives; the twin never runs one.
- Any other group with a PyNccl communicator gets a router without a twin: eager calls pass
  through, a captured call raises, so an unexpected captured use fails the boot at capture.
- Engagement audit lines, per rank: one per captured CUDA graph (torch CUDAGraph.capture_end,
  wrapped only when armed) with the collectives it captured, and one at 1, 10, 100, ... eager calls
  routed to the twin. nccl_twin_audit.py checks them against NCCL's own per-call INFO line.

Fail closed.
- install() disarms (mixing stays on, stock path, nothing patched) unless the lever is on, the NCCL
  that vLLM's PyNccl loads is in VALIDATED_NCCL, NCCL symmetric memory is off and the classes match
  what the router covers (check_anchors). Unless armed, any NCCL_GRAPH_MIXING_SUPPORT other than 1
  (NCCL strtoll-parses "00", " 0", "0x0" as 0 too; a disarmed worker can inherit 0 from an armed
  parent) is reset to 1, NCCL's default (graphUsageMode 2).
- Once armed, mixing is already off in the process, so attach() never falls back to a lone stock
  communicator: ranks that disagree on arming, a comm whose NCCL version differs, a group with more
  than one rank and no working PyNccl, or a failed twin self-test raise REFUSED and stop the boot.
- Both ranks exchange (armed, NCCL version) over the group's CPU group in every attach() call, armed
  or not, so a rank that did not arm cannot leave its peer waiting inside the twin's creation. That
  CPU all-gather at communicator init is the only effect this file has on the stock path.

Numerics: unchanged. The same NCCL algorithm, protocol and channel run on the same buffers; only
which communicator object issues each call changes.
"""

from __future__ import annotations

import inspect
import os

ENV = "VLLM_QWEN38_NCCL_TWIN"
MIXING_ENV = "NCCL_GRAPH_MIXING_SUPPORT"

LOG_ENGAGED = "qwen38: nccl twin engaged"
LOG_GUARD = "qwen38: nccl twin guard"
LOG_GRAPH = "qwen38: nccl twin graph"
LOG_EAGER = "qwen38: nccl twin eager"
LOG_DISARMED = ("qwen38: nccl twin DISARMED", "qwen38: nccl twin REFUSED")
# NCCL 2.30.7's own lines, INFO level (NCCL_DEBUG=INFO, subsystem ENV): the param read, once per
# process (src/misc/param.cc:99), and one per captured collective/p2p API call on a comm whose
# graphUsageMode is 0 (src/plugin/profiler.cc:339-343, 372-376). With NCCL_DEBUG=WARN neither
# prints, which is why the sibling counted 0 of 434.
NCCL_ENV_LINE = "NCCL_GRAPH_MIXING_SUPPORT set by environment to 0"
NCCL_CAPTURE_LINE = "graphUsageMode is set to 0 but the user is capturing graphs"

# NCCL releases whose source was read for this lever (graphUsageMode only selects `mixing`).
# The v0.30.0 image ships nvidia-nccl-cu13 2.30.7; vLLM's NCCLLibrary reports 2.30.7 (23007).
VALIDATED_NCCL = ("2.30.7",)

# PyNcclCommunicator methods that enqueue NCCL work on a stream: routed per call.
ROUTED = (
    "all_reduce",
    "all_gather",
    "all_gatherv",
    "reduce_scatter",
    "reduce_scatterv",
    "reduce",
    "scatter",
    "send",
    "recv",
    "broadcast",
    "batch_isend_irecv",
)
# Lifecycle calls that must reach both communicators.
FAN_OUT = ("destroy", "suspend", "resume")
# ncclGroupStart/End take no communicator: forwarding them to either comm is the same.
PASS = ("group_start", "group_end")
# Window registration binds a buffer to one communicator (NCCL symmetric memory, refused at install).
REFUSED = ("register_comm_window", "register_comm_window_raw", "deregister_comm_window")
# Public PyNcclCommunicator names that need no routing (classmethod constructor).
IGNORED = ("from_unique_id_bytes",)
# The overlay's call in CudaCommunicator.__init__, and the stock lines it must follow.
HOOK = "_nccl_twin.attach(self, tcp_store_group, PyNcclCommunicator)"
INIT_ANCHORS = (
    "self.pynccl_comm = PyNcclCommunicator(",
    "group=self.cpu_group if tcp_store_group is None else tcp_store_group",
    HOOK,
)

# Process-wide state: armed by install(); routers registered by attach().
STATE = {"armed": False, "log": None, "graphs": 0}
ROUTERS: list = []


def lever_on(env) -> bool:
    return (env.get(ENV, "0") or "0").strip() == "1"


def check_nccl_version(version: str) -> None:
    if version not in VALIDATED_NCCL:
        raise RuntimeError(
            f"NCCL {version} is not validated for {MIXING_ENV}=0 (validated: {', '.join(VALIDATED_NCCL)}); "
            "re-read its graphUsageMode handling first"
        )


def group_kind(unique_name: str) -> str:
    """'tp' for vLLM's TP group ('tp:0'), else the name's prefix."""
    return (unique_name or "").split(":", 1)[0]


def stream_positions(pynccl_cls) -> dict:
    """Positional index (self excluded) of each routed method's 'stream' parameter."""
    out = {}
    for name in ROUTED:
        params = list(inspect.signature(getattr(pynccl_cls, name)).parameters)
        if "stream" not in params:
            raise TypeError(f"PyNcclCommunicator.{name} has no stream parameter")
        out[name] = params.index("stream") - 1
    return out


def check_anchors(comm_cls, pynccl_cls) -> None:
    """Raise if CudaCommunicator/PyNcclCommunicator differ from what the router covers."""
    src = inspect.getsource(comm_cls.__init__)
    for anchor in INIT_ANCHORS:
        if anchor not in src:
            raise RuntimeError(f"CudaCommunicator.__init__ anchor missing: {anchor!r}")
    params = inspect.signature(comm_cls.__init__).parameters
    for name in ("cpu_group", "device", "unique_name", "tcp_store_group"):
        if name not in params:
            raise RuntimeError(f"CudaCommunicator.__init__ has no {name!r} parameter")
    public = {
        n
        for klass in pynccl_cls.__mro__[:-1]
        for n, v in vars(klass).items()
        if not n.startswith("_") and (callable(v) or isinstance(v, (classmethod, staticmethod)))
    }
    known = set(ROUTED) | set(FAN_OUT) | set(PASS) | set(REFUSED) | set(IGNORED)
    missing = sorted((set(ROUTED) | set(FAN_OUT) | set(PASS)) - public)
    unknown = sorted(public - known)
    if missing or unknown:
        raise RuntimeError(f"PyNcclCommunicator API changed: missing {missing}, unrouted {unknown}")
    stream_positions(pynccl_cls)


def _arg(args, kwargs, pos: int, name: str):
    if name in kwargs:
        return kwargs[name]
    return args[pos] if len(args) > pos else None


def nccl_calls(method: str, args, kwargs, rank: int) -> int:
    """NCCL API calls (ncclAllReduce, ncclBroadcast, ncclSend, ...) one PyNccl call issues.

    NCCL logs NCCL_CAPTURE_LINE once per captured API call, so the audit compares against this.
    """
    if method in ("all_gatherv", "reduce_scatterv"):
        return len(_arg(args, kwargs, 2, "sizes") or ())
    if method == "batch_isend_irecv":
        return len(_arg(args, kwargs, 0, "p2p_ops") or ())
    if method == "scatter":
        sizes = list(_arg(args, kwargs, 2, "sizes") or ())
        root = _arg(args, kwargs, 3, "root")
        root = 0 if root is None else root
        if rank == root:
            return sum(1 for dst, n in enumerate(sizes) if n and dst != root)
        return 1 if rank < len(sizes) and sizes[rank] > 0 else 0
    return 1


class GraphEagerRouter:
    """Stands in for CudaCommunicator.pynccl_comm.

    graph: the stock communicator, used only for calls whose stream is capturing.
    eager: the twin, used for every other call; None means this group has no twin and a
    captured call raises (guard).
    capturing(stream) -> bool decides per call, on the stream PyNccl will launch on.
    Attribute reads (disabled, world_size, rank, device, nccl, ...) come from graph.
    """

    def __init__(self, graph, eager, capturing, name: str, positions: dict, logger):
        d = self.__dict__
        d["_graph"], d["_eager"], d["_capturing"] = graph, eager, capturing
        d["_name"], d["_positions"], d["_logger"] = name, positions, logger
        d["_rank"] = int(getattr(graph, "rank", 0) or 0)
        d["_n_graph"] = d["_n_graph_nccl"] = d["_n_eager"] = d["_n_eager_nccl"] = 0
        d["_mark"] = d["_mark_nccl"] = 0
        d["_next_milestone"] = 1

    def _pick(self, method: str, args, kwargs):
        stream = kwargs.get("stream")
        pos = self._positions[method]
        if stream is None and len(args) > pos:
            stream = args[pos]
        d = self.__dict__
        calls = nccl_calls(method, args, kwargs, self._rank)
        if self._capturing(stream):
            if self._eager is None:
                raise RuntimeError(
                    f"{LOG_DISARMED[1]}: {method} on group {self._name} was captured into a CUDA graph, but "
                    f"{MIXING_ENV}=0 is only safe for a communicator with an eager twin (tp). Set {ENV}=0."
                )
            d["_n_graph"] += 1
            d["_n_graph_nccl"] += calls
            return self._graph
        d["_n_eager"] += 1
        d["_n_eager_nccl"] += calls
        if self._eager is None:
            return self._graph
        if self._n_eager >= self._next_milestone:
            d["_next_milestone"] *= 10
            self._logger.info(f"{LOG_EAGER} {self._counts()}")
        return self._eager

    def _counts(self) -> str:
        return (
            f"on {self._name} rank={self._rank} eager={self._n_eager} eager_nccl={self._n_eager_nccl} "
            f"total_captured={self._n_graph} total_captured_nccl={self._n_graph_nccl} graphs={STATE['graphs']}"
        )

    def _graph_end(self, index: int) -> None:
        """Called after every CUDA graph capture in this process: log what this group captured."""
        d = self.__dict__
        delta, delta_nccl = self._n_graph - self._mark, self._n_graph_nccl - self._mark_nccl
        d["_mark"], d["_mark_nccl"] = self._n_graph, self._n_graph_nccl
        self._logger.info(f"{LOG_GRAPH} {index} captured=+{delta} captured_nccl=+{delta_nccl} {self._counts()}")

    def __getattr__(self, name):
        if name in REFUSED:
            raise RuntimeError(f"{LOG_DISARMED[1]}: {name} is not routed by {ENV}")
        return getattr(self._graph, name)

    def __setattr__(self, name, value):
        raise AttributeError(f"GraphEagerRouter is read-only ({name})")


def _routed(method: str):
    def call(self, *args, **kwargs):
        return getattr(self._pick(method, args, kwargs), method)(*args, **kwargs)

    call.__name__ = method
    return call


def _fan_out(method: str):
    def call(self, *args, **kwargs):
        out = getattr(self._graph, method)(*args, **kwargs)
        if self._eager is not None:
            getattr(self._eager, method)(*args, **kwargs)
        return out

    call.__name__ = method
    return call


def _passed(method: str):
    def call(self, *args, **kwargs):
        return getattr(self._graph, method)(*args, **kwargs)

    call.__name__ = method
    return call


for _m in ROUTED:
    setattr(GraphEagerRouter, _m, _routed(_m))
for _m in FAN_OUT:
    setattr(GraphEagerRouter, _m, _fan_out(_m))
for _m in PASS:
    setattr(GraphEagerRouter, _m, _passed(_m))
del _m


def stream_is_capturing(stream=None) -> bool:
    """Whether the stream PyNccl will launch on (explicit, else vLLM's current) is capturing.

    Hot path for every eager TP collective: when that stream is already torch's current stream
    (always, for CudaCommunicator's calls) ask directly instead of paying torch.cuda.stream()'s
    two Stream constructions and two set_stream calls.
    """
    import torch
    from vllm.utils.torch_utils import current_stream

    s = current_stream() if stream is None else stream
    if torch._C._cuda_getCurrentStream(s.device_index)[0] == s.stream_id:
        return bool(torch.cuda.is_current_stream_capturing())
    with torch.cuda.stream(s):
        return bool(torch.cuda.is_current_stream_capturing())


def twin_self_test(twin) -> None:
    """Eager all-reduce of (rank + 1) on the twin must give n(n+1)/2 on every rank."""
    import torch

    n = int(twin.world_size)
    x = torch.full((4,), float(twin.rank + 1), dtype=torch.float32, device=twin.device)
    out = twin.all_reduce(x)
    torch.cuda.synchronize(twin.device)
    want = n * (n + 1) / 2
    if out is None or not bool(torch.all(out == want)):
        raise RuntimeError(f"twin self-test: all_reduce gave {None if out is None else out.tolist()}, want {want}")


def _gather_default(cc, tcp_store_group, item):
    """(armed, nccl version) from every rank of the group, over its CPU (gloo) group."""
    if tcp_store_group is not None:
        return list(tcp_store_group.all_gather_obj(item))
    import torch.distributed as dist

    out = [None] * dist.get_world_size(cc.cpu_group)
    dist.all_gather_object(out, item, group=cc.cpu_group)
    return out


def attach(cc, tcp_store_group, pynccl_cls, *, gather=None, capturing=None, self_test=None):
    """Return what CudaCommunicator.pynccl_comm should be (called right after the stock comm).

    Not armed on every rank: the stock communicator, unchanged. Armed: a router (tp) or guard.
    """
    comm = cc.pynccl_comm
    name = getattr(cc, "unique_name", "") or "?"
    armed = bool(STATE["armed"])
    working = comm is not None and not getattr(comm, "disabled", True)
    version = comm.nccl.ncclGetVersion() if working else None
    peers = (gather or _gather_default)(cc, tcp_store_group, (armed, version))
    if not any(p[0] for p in peers):
        return comm
    logger = STATE["log"]
    if not all(p[0] for p in peers):
        raise RuntimeError(
            f"{LOG_DISARMED[1]}: group {name}: {ENV} armed on some ranks only ({peers}); "
            f"{MIXING_ENV}=0 is already set on the armed ones. Arm all ranks or none."
        )
    if not working:
        raise RuntimeError(
            f"{LOG_DISARMED[1]}: group {name} has {cc.world_size} ranks but no working PyNccl communicator, "
            f"so {MIXING_ENV}=0 would reach torch.distributed without a twin. Set {ENV}=0."
        )
    try:
        for p in peers:
            check_nccl_version(p[1])
    except RuntimeError as exc:
        raise RuntimeError(f"{LOG_DISARMED[1]}: group {name}: {exc}. Set {ENV}=0.") from exc
    capturing = capturing or stream_is_capturing
    positions = stream_positions(pynccl_cls)
    rank = f"rank={comm.rank}/{comm.world_size}"
    if group_kind(name) != "tp":
        router = GraphEagerRouter(comm, None, capturing, name, positions, logger)
        logger.info(f"{LOG_GUARD} on {name} {rank}: eager-only, a captured call raises")
        return router
    twin = pynccl_cls(group=cc.cpu_group if tcp_store_group is None else tcp_store_group, device=cc.device)
    if getattr(twin, "disabled", True):
        raise RuntimeError(f"{LOG_DISARMED[1]}: eager twin for {name} came up disabled")
    try:
        if capturing(None):
            raise RuntimeError("the current stream is capturing at communicator init")
        (self_test or twin_self_test)(twin)
    except Exception as exc:  # noqa: BLE001 - mixing is off in this process: stop the boot
        raise RuntimeError(f"{LOG_DISARMED[1]}: group {name}: {exc}. Set {ENV}=0.") from exc
    router = GraphEagerRouter(comm, twin, capturing, name, positions, logger)
    ROUTERS.append(router)
    logger.info(
        f"{LOG_ENGAGED} on {name} {rank} nccl={version} {MIXING_ENV}={os.environ.get(MIXING_ENV)}: "
        "captured collectives -> stock comm, eager -> twin; twin self-test ok"
    )
    return router


def hook_capture_end(graph_cls) -> None:
    """Wrap graph_cls.capture_end so every finished capture reports its collectives per group."""
    orig = graph_cls.capture_end
    if getattr(vars(graph_cls).get("capture_end"), "_qwen38_nccl_twin", False):
        return

    def capture_end(self, *args, **kwargs):
        out = orig(self, *args, **kwargs)
        STATE["graphs"] += 1
        for router in ROUTERS:
            router._graph_end(STATE["graphs"])
        return out

    capture_end._qwen38_nccl_twin = True
    capture_end.__wrapped__ = orig
    graph_cls.capture_end = capture_end


def install(logger, *, env=None, version=None, symm_enabled=None, comm_cls=None, pynccl_cls=None,
            graph_cls=None) -> str:
    """'off' | 'armed' | 'disarmed'. Writes NCCL_GRAPH_MIXING_SUPPORT=0 only when armed.

    version() -> NCCL version string, symm_enabled() -> bool, and the classes default to vLLM's and
    torch's (the overlay passes CudaCommunicator, which is being defined in the same module).
    """
    env = os.environ if env is None else env
    STATE["log"] = logger

    def keep_mixing(why: str) -> None:
        # Not armed: mixing must stay on. NCCL strtoll-parses the value, so anything but "1" may mean 0.
        value = env.get(MIXING_ENV)
        if value is not None and value != "1":
            env[MIXING_ENV] = "1"
            logger.warning(f"{LOG_DISARMED[1]}: {MIXING_ENV}={value} {why} is unsafe here; reset to 1")

    if not lever_on(env):
        keep_mixing(f"without {ENV}=1")
        return "off"
    try:
        if version is None:
            from vllm.distributed.device_communicators.pynccl_wrapper import NCCLLibrary

            version = NCCLLibrary().ncclGetVersion
        check_nccl_version(version())
        if symm_enabled is None:
            from vllm.distributed.device_communicators.pynccl_allocator import is_symmetric_memory_enabled

            symm_enabled = is_symmetric_memory_enabled
        if symm_enabled():
            raise RuntimeError("NCCL symmetric memory (VLLM_USE_NCCL_SYMM_MEM=1) binds windows to the stock comm")
        if pynccl_cls is None:
            from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator as pynccl_cls
        if graph_cls is None:
            import torch

            graph_cls = torch.cuda.graphs.CUDAGraph
        check_anchors(comm_cls, pynccl_cls)
        hook_capture_end(graph_cls)
    except Exception as exc:  # noqa: BLE001 - disarm, never half-apply
        logger.warning(f"{LOG_DISARMED[0]}: {exc!r}; NCCL graph mixing stays on")
        keep_mixing("with the twin disarmed")
        return "disarmed"
    env[MIXING_ENV] = "0"
    STATE["armed"] = True
    logger.info(f"qwen38: nccl twin armed: {MIXING_ENV}=0 written before the first communicator")
    return "armed"
'''
_nccl_twin = _load_nccl_twin()
_nccl_twin.install(logger, comm_cls=CudaCommunicator)
