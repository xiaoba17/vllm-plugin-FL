# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

# vLLM's Triton topk_topp kernel fails to compile on MetaX
# (PassManager::run failed in ttgir stage). Patch apply_top_k_top_p
# to always use the PyTorch fallback.
# TODO: remove once FlagGems provides a MetaX-compatible topk_topp kernel.

import os

import torch
import vllm.v1.sample.ops.topk_topp_sampler as topk_topp_sampler
from vllm.logger import init_logger

logger = init_logger(__name__)


def _apply_top_k_top_p_no_triton(
    logits: torch.Tensor, k: torch.Tensor | None, p: torch.Tensor | None
) -> torch.Tensor:
    if p is None and k is None:
        return logits
    if (
        os.environ.get("VLLM_FL_METAX_RADIX_SORT", "0") == "1"
        and k is None
        and p is not None
        and logits.is_cuda
        and logits.dtype == torch.float32
        and logits.ndim == 2
        and logits.shape[1] == 130560
        and logits.is_contiguous()
    ):
        import flag_gems

        # Equal-value index order matches FlagGems' stable radix sorter, not
        # necessarily PyTorch's native sort. Respect alternate registrations.
        registrar = getattr(flag_gems, "current_work_registrar", None)
        if registrar is not None and "sort" in registrar.get_all_ops():
            from ..impl.radix_sort import radix_top_p

            logger.info_once("MetaX tiled radix Top-p enabled; original sampling parameters preserved")
            return radix_top_p(logits, p)
    return topk_topp_sampler.apply_top_k_top_p_pytorch(logits, k, p)


# Replace the dispatch function with one that skips Triton
topk_topp_sampler.apply_top_k_top_p = _apply_top_k_top_p_no_triton
