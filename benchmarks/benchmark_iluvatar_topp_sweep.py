#!/usr/bin/env python3

"""Sweep launch parameters for vLLM's existing Top-p Triton kernel.

This benchmark reuses the installed vLLM kernel verbatim and changes only its
launch-time block sizes, warp count, and grid size. It is an experiment tool;
it does not patch the serving path or replace the official benchmark.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Callable

import torch

from benchmark_iluvatar_topp import (
    benchmark_filter,
    load_real_logits,
    parse_int_list,
    pytorch_reference,
    validate_support,
)


TensorFilter = Callable[
    [torch.Tensor, torch.Tensor | None, torch.Tensor | None], torch.Tensor
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logits-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--vocab-size", type=int, default=130560)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--block-sizes", type=parse_int_list, default=[2048, 4096, 8192])
    parser.add_argument(
        "--block-size-truncs", type=parse_int_list, default=[1024, 2048, 4096]
    )
    parser.add_argument("--num-warps", type=parse_int_list, default=[4])
    parser.add_argument("--program-counts", type=parse_int_list, default=[16])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.vocab_size <= 0:
        parser.error("batch size and vocab size must be positive")
    if not 0.0 < args.top_p <= 1.0:
        parser.error("--top-p must be in (0, 1]")
    if args.warmup < 0 or args.repeats <= 0:
        parser.error("warmup must be non-negative and repeats must be positive")
    return args


def make_configured_filter(
    block_size: int,
    block_size_trunc: int,
    num_warps: int,
    program_count: int,
) -> TensorFilter:
    import vllm.v1.sample.ops.topk_topp_triton as topp

    buffers: dict[tuple[torch.device, torch.dtype, int, int], torch.Tensor] = {}

    def configured_filter(
        logits: torch.Tensor,
        k: torch.Tensor | None,
        p: torch.Tensor | None,
    ) -> torch.Tensor:
        assert logits.ndim == 2 and logits.dtype == torch.float32
        assert k is None, "the initial sweep covers standalone Top-p only"
        assert p is not None and p.ndim == 1 and p.shape[0] == logits.shape[0]
        if logits.stride(1) != 1:
            logits = logits.contiguous()

        batch_size, vocab_size = logits.shape
        num_programs = min(program_count, batch_size)
        buffer_key = (logits.device, logits.dtype, vocab_size, num_programs)
        buffer = buffers.get(buffer_key)
        if buffer is None:
            buffer = logits.new_empty((num_programs, vocab_size))
            buffers[buffer_key] = buffer

        tables = topp._TRITON_TABLE_CACHE.get(logits.device)
        if tables is None:
            normal_table = logits.new_tensor(topp._NORMAL_CDF_TO_SIGMA_TABLE)
            percentile_table = logits.new_tensor(topp._PERCENTILE_TO_STD_TABLE)
            tables = (normal_table, percentile_table)
            topp._TRITON_TABLE_CACHE[logits.device] = tables
        normal_table, percentile_table = tables

        topp._topk_topp_kernel[(num_programs,)](
            logits,
            logits.stride(0),
            buffer,
            percentile_table,
            normal_table,
            logits,  # Dummy K pointer; TOPK_ENABLED is false.
            p.to(torch.float32),
            BATCH_SIZE=batch_size,
            MASK_VALUE=float("-inf"),
            VOCAB_SIZE=vocab_size,
            BLOCK_SIZE=block_size,
            BLOCK_SIZE_TRUNC=block_size_trunc,
            TOPK_ENABLED=False,
            TOPP_ENABLED=True,
            num_warps=num_warps,
        )
        return logits

    return configured_filter


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("this sweep requires a CUDA-compatible accelerator")
    device = torch.device("cuda")
    logits = load_real_logits(
        args.logits_file,
        args.batch_size,
        args.vocab_size,
        device,
    )
    p = torch.full(
        (args.batch_size,), args.top_p, dtype=torch.float32, device=device
    )
    reference = pytorch_reference(logits.clone(), p)
    torch.cuda.synchronize()

    configs = [
        config
        for config in itertools.product(
            args.block_sizes,
            args.block_size_truncs,
            args.num_warps,
            args.program_counts,
        )
        if config[1] <= config[0]
    ]
    report: dict[str, object] = {
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "logits_file": str(args.logits_file),
        "shape": list(logits.shape),
        "top_p": args.top_p,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "results": [],
    }

    for block_size, block_size_trunc, num_warps, program_count in configs:
        config = {
            "block_size": block_size,
            "block_size_trunc": block_size_trunc,
            "num_warps": num_warps,
            "program_count": program_count,
        }
        try:
            filter_fn = make_configured_filter(
                block_size,
                block_size_trunc,
                num_warps,
                program_count,
            )
            candidate = filter_fn(logits.clone(), None, p)
            torch.cuda.synchronize()
            correctness = validate_support(candidate, reference, logits)
            timing = benchmark_filter(
                filter_fn,
                logits,
                p,
                args.warmup,
                args.repeats,
            )
            result = {"config": config, "correctness": correctness, "timing": timing}
            print(
                f"block={block_size:>5} trunc={block_size_trunc:>4} "
                f"warps={num_warps} programs={program_count:>2} "
                f"median={timing['median_ms']:.4f} ms "
                f"mass_min={correctness['candidate_mass_min']:.6f}",
                flush=True,
            )
        except Exception as error:
            result = {
                "config": config,
                "error": f"{type(error).__name__}: {error}",
            }
            print(f"config={config} error={result['error']}", flush=True)
        report["results"].append(result)

    successful = [item for item in report["results"] if "timing" in item]
    if successful:
        successful.sort(key=lambda item: item["timing"]["median_ms"])
        report["best_config"] = successful[0]["config"]
        report["best_median_ms"] = successful[0]["timing"]["median_ms"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
