# Copyright (c) 2026 BAAI. All rights reserved.

"""
ILUVATAR backend implementation.

This backend provides operator implementations for Iluvatar GPUs.
Iluvatar uses a CUDA-compatible architecture.
"""

from __future__ import annotations

import logging
from typing import Optional, Union

import torch

from vllm_fl.dispatch.backends.base import Backend

logger = logging.getLogger(__name__)


def patch_triton_language_for_iluvatar() -> None:
    """Add make_tensor_descriptor stub to triton.language for triton < 3.3.

    triton JIT's DependencyFinder walks all @triton.jit function bodies at
    cache_key computation time (first compile, not import). Even dead-code
    branches guarded by ``USE_TD: tl.constexpr = False`` cause AttributeError
    when ``tl.make_tensor_descriptor`` does not exist, because DependencyFinder
    resolves ``tl.*`` attributes to compute a stable kernel hash.

    The _Stub below is a plain Python callable + hashable object that satisfies
    attribute lookup. It raises at call-time so accidental USE_TD=True usage is
    caught immediately.

    TODO: Remove once minimum supported triton version is >= 3.3.
    """
    try:
        import triton.language as tl

        if hasattr(tl, "make_tensor_descriptor"):
            return  # triton >= 3.3, nothing to do

        class _TensorDescriptorStub:
            """Stub for tl.make_tensor_descriptor — callable and hashable."""

            def __call__(self, *args, **kwargs):
                raise RuntimeError(
                    "tl.make_tensor_descriptor is not available on triton < 3.3. "
                    "Ensure VLLM_TRITON_ATTN_USE_TD=0 (the default on iluvatar)."
                )

            def __hash__(self):
                return hash("_tl_make_tensor_descriptor_stub")

            def __repr__(self):
                return "<tl.make_tensor_descriptor stub for triton<3.3>"

        tl.make_tensor_descriptor = _TensorDescriptorStub()
        logger.info(
            "Patched triton.language: added make_tensor_descriptor stub "
            "(triton < 3.3 detected; USE_TD=False assumed on iluvatar)."
        )
    except Exception as e:
        logger.warning("Failed to patch triton.language for iluvatar: %s", e)


def patch_triton_chained_or_for_iluvatar() -> None:
    """Rewrite chained boolean 'or' chains in vllm triton kernels.

    Iluvatar's triton version (3.2.x) raises UnsupportedLanguageConstruct for
    chained boolean operators like ``A or B or C`` inside @triton.jit functions.
    Rewrites the affected source file in-place (idempotent).

    Must be called from the main process BEFORE Worker subprocesses start,
    so the patched file is on disk when Workers import the module.

    Applies only when triton < 3.3 — later versions support chained boolean
    operators natively.

    TODO: Remove once minimum supported Iluvatar triton version is >= 3.3.
    """
    import re
    import importlib.util
    import pathlib
    import sys

    # Only needed for triton < 3.3.
    try:
        import triton as _triton
        _tv = tuple(int(x) for x in _triton.__version__.split(".")[:2])
        if _tv >= (3, 3):
            return
    except Exception as e:
        logger.warning(
            "patch_triton_chained_or_for_iluvatar: cannot determine triton version, "
            "applying patch defensively: %s", e
        )

    _MARKER = "# _iluvatar_chained_or_patched"

    spec = importlib.util.find_spec("vllm.v1.attention.ops.triton_attention_helpers")
    if spec is None or spec.origin is None:
        logger.warning(
            "patch_triton_chained_or_for_iluvatar: "
            "vllm.v1.attention.ops.triton_attention_helpers not found, skipping."
        )
        return

    fpath = pathlib.Path(spec.origin)
    try:
        src = fpath.read_text()
    except Exception as e:
        logger.warning("patch_triton_chained_or_for_iluvatar: cannot read %s: %s", fpath, e)
        return

    if _MARKER in src:
        return  # already patched

    # Replace: A or B or C  →  (A or B) or C
    _OPERAND = r'(?:not\s+)?(?:\([^)]*\)|\w+)'
    pattern = re.compile(
        r'(' + _OPERAND + r')\s+or\s+(' + _OPERAND + r')\s+or\s+(' + _OPERAND + r')'
    )

    def _rewrite(m: re.Match) -> str:
        return f"({m.group(1)} or {m.group(2)}) or {m.group(3)}"

    new_src, count = re.subn(pattern, _rewrite, src)
    if count == 0:
        return  # nothing to patch

    new_src += f"\n{_MARKER}\n"
    try:
        fpath.write_text(new_src)
    except Exception as e:
        logger.warning(
            "patch_triton_chained_or_for_iluvatar: cannot write %s: %s", fpath, e
        )
        return

    # Clear pycache so Python and triton both see the patched source.
    pycache = fpath.parent / "__pycache__"
    if pycache.exists():
        import shutil
        try:
            shutil.rmtree(pycache)
        except Exception:
            pass  # non-fatal

    # Evict from sys.modules so this process reimports the patched source.
    sys.modules.pop("vllm.v1.attention.ops.triton_attention_helpers", None)

    logger.info(
        "patch_triton_chained_or_for_iluvatar: rewrote %d chained-or expression(s) in %s",
        count, fpath,
    )


def patch_triton_perf_model_for_iluvatar() -> None:
    """
    Patch triton.ops.matmul_perf_model.get_clock_rate_in_khz for iluvatar.

    On iluvatar, neither `ixsmi` nor `libnvidia-ml.so` is available, so the
    default implementation crashes. We replace it with a fixed value that
    matches typical iluvatar BI-V150 SM clock (1500 MHz = 1500000 kHz).

    The function is decorated with @functools.lru_cache so assigning a new
    callable to the module attribute is sufficient — callers import the name
    directly, so we also patch the reference in the testing helper.

    Applies only when triton < 3.3 — later versions handle missing nvsmi/nvml
    natively or remove triton.ops.matmul_perf_model entirely.

    TODO: Remove once minimum supported Iluvatar triton version is >= 3.3.
    """
    try:
        import triton as _triton
        _tv = tuple(int(x) for x in _triton.__version__.split(".")[:2])
        if _tv >= (3, 3):
            return

        import triton.ops.matmul_perf_model as _mpm

        if getattr(_mpm, '_iluvatar_clock_patched', False):
            return

        import functools

        @functools.lru_cache()
        def _iluvatar_get_clock_rate_in_khz():
            # BI-V150 SM clock ~1500 MHz; used only for perf-model heuristics.
            return 1500 * 1e3

        _mpm.get_clock_rate_in_khz = _iluvatar_get_clock_rate_in_khz
        _mpm._iluvatar_clock_patched = True
        logger.info(
            "Patched triton.ops.matmul_perf_model.get_clock_rate_in_khz "
            "for iluvatar (fixed 1500 MHz, no nvsmi/nvml available)."
        )
    except Exception as e:
        logger.warning(
            "Failed to patch triton perf model for iluvatar: %s", e
        )

patch_triton_chained_or_for_iluvatar()
patch_triton_language_for_iluvatar()
patch_triton_perf_model_for_iluvatar()


def patch_sampler_compile_for_iluvatar() -> None:
    # Disable torch.compile on vllm sampler ops for Iluvatar.
    # flagtree triton only supports Iluvatar backend, not cuda target.
    # TODO: Remove once flagtree triton supports cuda inductor target.
    try:
        import importlib
        _tts = importlib.import_module('vllm.v1.sample.ops.topk_topp_sampler')
        if not getattr(_tts, '_iluvatar_compile_patched', False):
            _orig = _tts.compiled_random_sample
            _tts.compiled_random_sample = getattr(_orig, '__wrapped__', _orig)
            _tts._iluvatar_compile_patched = True
            logger.info('patch_sampler_compile_for_iluvatar: unwrapped topk_topp_sampler.compiled_random_sample')
    except Exception as e:
        logger.warning('patch_sampler_compile_for_iluvatar (topk_topp): %s', e)
    try:
        import importlib
        _lp = importlib.import_module('vllm.v1.sample.ops.logprobs')
        if not getattr(_lp, '_iluvatar_compile_patched', False):
            _orig = _lp.batched_count_greater_than
            _lp.batched_count_greater_than = getattr(_orig, '__wrapped__', _orig)
            _lp._iluvatar_compile_patched = True
            logger.info('patch_sampler_compile_for_iluvatar: unwrapped logprobs.batched_count_greater_than')
    except Exception as e:
        logger.warning('patch_sampler_compile_for_iluvatar (logprobs): %s', e)


def patch_topp_launch_for_iluvatar() -> None:
    """Tune standalone Top-p launch geometry for BI-V150.

    The upstream kernel limits its grid to the device compute-unit count. On a
    16-CU BI-V150 this makes a batch of 64 execute four rows serially in each
    program. The kernel has no cross-row communication, so launching up to 64
    programs exposes row-level parallelism without changing the algorithm.

    This patch deliberately covers only the competition's standalone Top-p
    path when a batch is larger than the device's 16-CU grid. Top-k, combined
    Top-k/Top-p, batches up to 16, and unexpected layouts or dtypes retain the
    upstream implementation.
    """
    try:
        import importlib

        _tts = importlib.import_module("vllm.v1.sample.ops.topk_topp_sampler")
        _ttt = importlib.import_module("vllm.v1.sample.ops.topk_topp_triton")
        if getattr(_ttt, "_iluvatar_launch_patched", False):
            return

        original = _ttt.apply_top_k_top_p_triton
        buffers: dict[tuple[torch.device, torch.dtype, int], torch.Tensor] = {}

        def apply_top_k_top_p_triton_iluvatar(
            logits: torch.Tensor,
            k: torch.Tensor | None,
            p: torch.Tensor | None,
            mask_value: float = float("-inf"),
        ) -> torch.Tensor:
            if (
                k is not None
                or p is None
                or logits.ndim != 2
                or logits.dtype != torch.float32
                or logits.shape[0] <= 16
            ):
                return original(logits, k, p, mask_value)

            batch_size, vocab_size = logits.shape
            if logits.stride(1) != 1:
                logits = logits.contiguous()
            if p.ndim != 1 or p.shape[0] != batch_size:
                return original(logits, k, p, mask_value)

            # BI-V150 has 16 CUs. Up to four programs per CU avoids serial row
            # loops at the competition's batch sizes. Keep every other launch
            # parameter at its upstream value so per-row reduction order and
            # Top-p support are unchanged.
            num_programs = min(batch_size, 64)
            buffer_key = (logits.device, logits.dtype, vocab_size)
            buffer = buffers.get(buffer_key)
            if buffer is None or buffer.shape[0] < num_programs:
                buffer_rows = min(1 << (num_programs - 1).bit_length(), 64)
                buffer = logits.new_empty((buffer_rows, vocab_size))
                buffers[buffer_key] = buffer
            if buffer.shape[0] > num_programs:
                buffer = buffer[:num_programs]

            tables = _ttt._TRITON_TABLE_CACHE.get(logits.device)
            if tables is None:
                normal_table = logits.new_tensor(_ttt._NORMAL_CDF_TO_SIGMA_TABLE)
                percentile_table = logits.new_tensor(_ttt._PERCENTILE_TO_STD_TABLE)
                tables = (normal_table, percentile_table)
                _ttt._TRITON_TABLE_CACHE[logits.device] = tables
            normal_table, percentile_table = tables

            _ttt._topk_topp_kernel[(num_programs,)](
                logits,
                logits.stride(0),
                buffer,
                percentile_table,
                normal_table,
                logits,  # Dummy K pointer; TOPK_ENABLED is false.
                p.to(torch.float32),
                BATCH_SIZE=batch_size,
                MASK_VALUE=mask_value,
                VOCAB_SIZE=vocab_size,
                BLOCK_SIZE=8192,
                BLOCK_SIZE_TRUNC=4096,
                TOPK_ENABLED=False,
                TOPP_ENABLED=True,
                num_warps=4,
            )
            return logits

        _ttt.apply_top_k_top_p_triton = apply_top_k_top_p_triton_iluvatar
        _tts.apply_top_k_top_p_triton = apply_top_k_top_p_triton_iluvatar
        _ttt._iluvatar_launch_patched = True
        logger.info(
            "Tuned Iluvatar standalone Top-p grid: programs=min(batch, 64); "
            "upstream block=8192, trunc=4096, warps=4 preserved"
        )
    except Exception as e:
        logger.warning("patch_topp_launch_for_iluvatar: %s", e)


def patch_topp_logits_dump_for_iluvatar() -> None:
    """Optionally dump the exact logits entering vLLM's Top-p filter.

    This is a diagnostic-only hook. It is inactive unless
    ``VLLM_FL_ILUVATAR_TOPP_LOGITS_DUMP`` names an output ``.pt`` file. When
    enabled, the first matching call is copied to CPU before the filter can
    update logits in-place. ``VLLM_FL_ILUVATAR_TOPP_LOGITS_DUMP_BATCH`` can be
    set to require a particular batch size (for example, 64).

    Keeping this at the sampler-op boundary captures post-processor logits,
    i.e. the actual tensor the Top-p kernel sees, without changing model
    execution or the sampling algorithm.
    """
    import os

    output = os.getenv("VLLM_FL_ILUVATAR_TOPP_LOGITS_DUMP")
    if not output:
        return

    try:
        import importlib
        from pathlib import Path
        from threading import Lock

        _tts = importlib.import_module("vllm.v1.sample.ops.topk_topp_sampler")
        if getattr(_tts, "_iluvatar_topp_logits_dump_patched", False):
            return

        batch_value = os.getenv("VLLM_FL_ILUVATAR_TOPP_LOGITS_DUMP_BATCH")
        required_batch = int(batch_value) if batch_value else None
        output_path = Path(output)
        original = _tts.apply_top_k_top_p
        lock = Lock()
        dumped = False

        def _to_cpu(value):
            if value is None:
                return None
            return value.detach().to(device="cpu", copy=True)

        def apply_top_k_top_p_with_dump(logits, k, p):
            nonlocal dumped
            should_dump = (
                not dumped
                and p is not None
                and logits.ndim == 2
                and (required_batch is None or logits.shape[0] == required_batch)
            )
            if should_dump:
                with lock:
                    if not dumped:
                        # Mark before synchronizing/copying so an exception cannot
                        # repeatedly perturb subsequent decoding steps.
                        dumped = True
                        try:
                            output_path.parent.mkdir(parents=True, exist_ok=True)
                            temporary_path = output_path.with_suffix(
                                output_path.suffix + ".tmp"
                            )
                            torch.save(
                                {
                                    "logits": _to_cpu(logits),
                                    "k": _to_cpu(k),
                                    "p": _to_cpu(p),
                                    "shape": tuple(logits.shape),
                                    "dtype": str(logits.dtype),
                                    "device": str(logits.device),
                                },
                                temporary_path,
                            )
                            os.replace(temporary_path, output_path)
                            logger.info(
                                "Dumped Iluvatar Top-p input logits with shape %s to %s",
                                tuple(logits.shape),
                                output_path,
                            )
                        except Exception:
                            # Diagnostics must never break or change sampling.
                            logger.exception(
                                "Failed to dump Iluvatar Top-p logits to %s",
                                output_path,
                            )
            return original(logits, k, p)

        _tts.apply_top_k_top_p = apply_top_k_top_p_with_dump
        _tts._iluvatar_topp_logits_dump_patched = True
        logger.info(
            "Enabled one-shot Iluvatar Top-p logits dump to %s (batch=%s)",
            output_path,
            required_batch if required_batch is not None else "any",
        )
    except Exception as e:
        logger.warning("patch_topp_logits_dump_for_iluvatar: %s", e)


patch_sampler_compile_for_iluvatar()
patch_topp_launch_for_iluvatar()
patch_topp_logits_dump_for_iluvatar()


def patch_torch_inductor_for_iluvatar() -> None:
    """Patch torch._inductor GPUTarget to use the correct triton backend name.

    torch._inductor always passes device_type='cuda' to GPUTarget, but
    flagtree triton only registers an 'iluvatar' (3.6.x) or 'corex' (3.2.x)
    backend -- never 'cuda'.

    Fix: subclass GPUTarget to intercept 'cuda' and remap it to whatever
    backend name flagtree triton actually registered.

    Compatibility: safe to call when not using flagtree -- if triton
    has a 'cuda' backend or no backends at all, the patch is skipped.

    Hardware gate: Iluvatar only.
    TODO: Remove once torch._inductor or flagtree natively handles this.
    """
    try:
        import triton.backends as _tb
        _registered = list(getattr(_tb, 'backends', {}).keys())
        # If 'cuda' is already registered as a backend name, no patch needed
        if 'cuda' in _registered or not _registered:
            logger.debug('patch_torch_inductor_for_iluvatar: triton has cuda backend or no backends, skipping')
            return
        # Probe the actual target string expected by supports_target().
        # Different flagtree triton versions use different conventions:
        #   - 3.2.x: backend name 'iluvatar', supports_target checks == 'cuda'
        #   - 3.6.x: backend name 'iluvatar', supports_target checks == 'corex'
        # We must discover the correct target string at runtime.
        _target = None
        try:
            from triton.backends.compiler import GPUTarget as _GPUTarget
            # Include 'cuda' in probes — some backends accept 'cuda' as target
            for _probe in ('cuda', 'corex', 'iluvatar') + tuple(_registered):
                try:
                    _t = object.__new__(_GPUTarget)
                    _GPUTarget.__init__(_t, _probe, 90, False)
                    for _bname in _registered:
                        if _tb.backends[_bname].compiler.supports_target(_t):
                            _target = _probe
                            break
                except Exception:
                    pass
                if _target:
                    break
        except Exception:
            pass
        if not _target:
            _target = 'corex'  # safe default for all known flagtree versions

        # If the discovered target is already 'cuda', no remapping needed —
        # inductor naturally passes 'cuda' to GPUTarget.
        if _target == 'cuda':
            logger.info(
                "patch_torch_inductor_for_iluvatar: backend '%s' already "
                "supports target='cuda', no GPUTarget remap needed.",
                _registered,
            )
            return
    except Exception:
        logger.debug('patch_torch_inductor_for_iluvatar: cannot inspect triton backends, skipping')
        return

    try:
        import torch._inductor.runtime.triton_heuristics as _th
        from triton.backends.compiler import GPUTarget as _OrigGPUTarget

        if getattr(_th, '_iluvatar_gputarget_patched', False):
            return

        _remap_target = _target

        class _IluvatarGPUTarget(_OrigGPUTarget):
            """GPUTarget wrapper that remaps 'cuda' to the flagtree backend."""
            def __new__(cls, backend, *args, **kwargs):
                if backend == 'cuda':
                    backend = _remap_target
                return super().__new__(cls)

            def __init__(self, backend, *args, **kwargs):
                if backend == 'cuda':
                    backend = _remap_target
                super().__init__(backend, *args, **kwargs)

        _th.GPUTarget = _IluvatarGPUTarget
        _th._iluvatar_gputarget_patched = True
        logger.info(
            "Patched torch._inductor GPUTarget: 'cuda' -> '%s' (iluvatar)",
            _remap_target,
        )
    except Exception as e:
        logger.warning(
            "Failed to patch torch._inductor for iluvatar triton backend: %s", e
        )

patch_torch_inductor_for_iluvatar()


class IluvatarBackend(Backend):
    """
    Iluvatar backend for operator implementations.

    This backend uses Iluvatar libraries to provide high-performance
    operator implementations for Iluvatar GPUs.
    """

    _available: Optional[bool] = None

    @property
    def name(self) -> str:
        return "iluvatar"

    @property
    def vendor(self) -> Optional[str]:
        return "iluvatar"

    def is_available(self) -> bool:
        """
        Check if Iluvatar hardware and libraries are available.

        This method uses the platform's vendor information to determine
        if the device is an Iluvatar GPU.
        """
        if IluvatarBackend._available is None:
            try:
                from vllm.platforms import current_platform
                # Iluvatar GPUs should be detected via vendor_name
                if hasattr(current_platform, 'vendor_name') and current_platform.vendor_name == "iluvatar":
                    IluvatarBackend._available = True
                else:
                    # Fallback: check if CUDA is available with iluvatar device
                    if torch.cuda.is_available():
                        # Try to detect Iluvatar GPU
                        # Iluvatar GPUs typically expose CUDA-compatible interface
                        # We can check device name if available
                        device_name = torch.cuda.get_device_name(0)
                        if "iluvatar" in device_name.lower():
                            IluvatarBackend._available = True
                        else:
                            IluvatarBackend._available = False

                    else:
                        IluvatarBackend._available = False
            except Exception:
                IluvatarBackend._available = False
        return IluvatarBackend._available

    # ==================== Operator Implementations ====================

    def silu_and_mul(self, obj, x: torch.Tensor) -> torch.Tensor:
        """
        SiLU activation followed by element-wise multiplication.

        Args:
            obj: The calling obj (for interface consistency)
            x: Input tensor of shape [..., 2*d]

        Returns:
            Output tensor of shape [..., d]
        """
        from .impl.activation import silu_and_mul_iluvatar

        return silu_and_mul_iluvatar(obj, x)

    def rms_norm(
        self,
        obj,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        RMS normalization.

        Args:
            obj: The calling obj (e.g., RMSNorm layer)
            x: Input tensor
            residual: Optional residual tensor

        Returns:
            Normalized tensor, or tuple of (normalized, residual) if residual is provided
        """
        from .impl.normalization import rms_norm_iluvatar

        return rms_norm_iluvatar(obj, x, residual)

    def rotary_embedding(
        self,
        obj,
        query: torch.Tensor,
        key: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: torch.Tensor,
        rotary_interleaved: bool = False,
        inplace: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary position embedding.

        Args:
            obj: The calling obj (for interface consistency)
            query: Query tensor
            key: Key tensor
            cos: Cosine cache
            sin: Sine cache
            position_ids: Position indices
            rotary_interleaved: Whether to use interleaved rotary
            inplace: Whether to modify tensors in-place

        Returns:
            Tuple of (embedded_query, embedded_key)
        """
        from .impl.rotary import rotary_embedding_iluvatar

        return rotary_embedding_iluvatar(
            obj,
            query,
            key,
            cos,
            sin,
            position_ids,
            rotary_interleaved=rotary_interleaved,
            inplace=inplace,
        )

    def attention_backend(self, use_mla: bool = False, use_sparse: bool = False) -> str:
        """
        Get the attention backend class path for Iluvatar.

        Args:
            use_mla: Whether to use Multi-head Latent Attention (MLA)
            use_sparse: Whether to use Deepseek Sparse Attention (DSA)

        Returns:
            Fully qualified class path string
        """
        from vllm.v1.attention.backends.registry import AttentionBackendEnum

        if use_mla:
            if use_sparse:
                return AttentionBackendEnum.FLASHMLA_SPARSE.get_path()
            return AttentionBackendEnum.FLASHMLA.get_path()

        # flash_attn is not available on iluvatar. Use TRITON_ATTN (the vllm
        # default). The tl.make_tensor_descriptor stub and perf model patches
        # are already applied at module level (above), so triton JIT's
        # DependencyFinder can hash the kernel without AttributeError on
        # triton < 3.3.  The kernel itself uses USE_TD=False at runtime, so
        # the stub is never called.
        return AttentionBackendEnum.TRITON_ATTN.get_path()
