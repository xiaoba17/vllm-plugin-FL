#!/usr/bin/env python3

"""Benchmark and validate the Top-p filtering path used on Iluvatar GPUs.

The benchmark intentionally measures only Top-p filtering. It does not change
the model's sampling parameters and does not replace the official 4K/16K
end-to-end benchmark. Run it inside the isolated development environment on a
BI-V150 before and after a kernel change, then use the official benchmark for
the final performance claim.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Callable

import torch


TensorFilter = Callable[
    [torch.Tensor, torch.Tensor | None, torch.Tensor | None], torch.Tensor
]


def parse_int_list(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", type=parse_int_list, default=[1, 8, 16, 32, 64])
    parser.add_argument("--vocab-size", type=int, default=130560)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument(
        "--distributions",
        default="normal,flat,peaked,ties",
        help="Comma-separated: normal, flat, peaked, ties",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument(
        "--implementation",
        choices=("actual", "triton"),
        default="triton",
        help="actual follows vLLM dispatch; triton directly measures its Triton path",
    )
    parser.add_argument(
        "--logits-file",
        type=Path,
        help="Optional .pt tensor containing [batch, vocab] real model logits",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    parser.add_argument(
        "--require-exact-support",
        action="store_true",
        help=(
            "Fail when support differs from the sort reference. By default the "
            "difference is reported because the existing pivot kernel is approximate."
        ),
    )
    args = parser.parse_args()
    if args.vocab_size <= 0:
        parser.error("--vocab-size must be positive")
    if not 0.0 < args.top_p <= 1.0:
        parser.error("--top-p must be in (0, 1]")
    if args.warmup < 0 or args.repeats <= 0:
        parser.error("--warmup must be non-negative and --repeats must be positive")
    return args


def get_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires a CUDA-compatible accelerator")
    return torch.device("cuda")


def load_filter(implementation: str) -> TensorFilter:
    if implementation == "triton":
        from vllm.v1.sample.ops.topk_topp_triton import apply_top_k_top_p_triton

        return apply_top_k_top_p_triton

    # A standalone benchmark does not initialize the dispatch manager like a
    # vLLM service does. Load the same vendor backend explicitly so "actual"
    # includes Iluvatar's serving-time patches instead of silently measuring
    # the unpatched upstream function.
    if "iluvatar" in torch.cuda.get_device_name().lower():
        import importlib

        importlib.import_module(
            "vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar"
        )

    from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p

    return apply_top_k_top_p


def make_logits(
    distribution: str,
    batch_size: int,
    vocab_size: int,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    logits = torch.randn(
        (batch_size, vocab_size),
        device=device,
        dtype=torch.float32,
        generator=generator,
    )

    if distribution == "normal":
        return logits
    if distribution == "flat":
        return logits.mul_(0.05)
    if distribution == "peaked":
        logits.mul_(0.5)
        logits[:, :32].add_(torch.linspace(12.0, 4.0, 32, device=device))
        return logits
    if distribution == "ties":
        return logits.mul_(8.0).round_().div_(8.0)
    raise ValueError(f"unsupported distribution: {distribution}")


def load_real_logits(
    path: Path,
    batch_size: int,
    vocab_size: int,
    device: torch.device,
) -> torch.Tensor:
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(loaded, dict):
        loaded = loaded.get("logits")
    if not isinstance(loaded, torch.Tensor) or loaded.ndim != 2:
        raise ValueError("--logits-file must contain a 2-D tensor or {'logits': tensor}")
    if loaded.shape[1] != vocab_size:
        raise ValueError(
            f"real logits vocab mismatch: expected {vocab_size}, got {loaded.shape[1]}"
        )
    if loaded.shape[0] < batch_size:
        repeats = math.ceil(batch_size / loaded.shape[0])
        loaded = loaded.repeat((repeats, 1))
    return loaded[:batch_size].to(device=device, dtype=torch.float32).contiguous()


def pytorch_reference(logits: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Reference the exact ascending-sort semantics used by vLLM."""
    logits_sort, logits_idx = logits.sort(dim=-1, descending=False)
    probs_sort = logits_sort.softmax(dim=-1)
    probs_sum = torch.cumsum(probs_sort, dim=-1, out=probs_sort)
    top_p_mask = probs_sum <= 1 - p.unsqueeze(dim=1)
    top_p_mask[:, -1] = False
    logits_sort.masked_fill_(top_p_mask, -float("inf"))
    return logits.scatter_(dim=-1, index=logits_idx, src=logits_sort)


def validate_support(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    original: torch.Tensor,
) -> dict[str, int | float]:
    candidate_support = torch.isfinite(candidate)
    reference_support = torch.isfinite(reference)
    mismatch = candidate_support != reference_support
    extra = candidate_support & ~reference_support
    missing = reference_support & ~candidate_support
    mismatched_per_row = mismatch.sum(dim=1)
    common = candidate_support & reference_support
    if common.any():
        max_value_error = (candidate[common] - reference[common]).abs().max().item()
    else:
        max_value_error = float("nan")
    probabilities = original.softmax(dim=-1, dtype=torch.float32)
    candidate_mass = (probabilities * candidate_support).sum(dim=1)
    reference_mass = (probabilities * reference_support).sum(dim=1)
    mismatch_mass = (probabilities * mismatch).sum(dim=1)
    return {
        "mismatched_elements": int(mismatch.sum().item()),
        "mismatched_rows": int((mismatched_per_row != 0).sum().item()),
        "max_row_mismatches": int(mismatched_per_row.max().item()),
        "extra_elements": int(extra.sum().item()),
        "missing_elements": int(missing.sum().item()),
        "candidate_kept_mean": float(candidate_support.sum(dim=1).float().mean().item()),
        "reference_kept_mean": float(reference_support.sum(dim=1).float().mean().item()),
        "candidate_mass_min": float(candidate_mass.min().item()),
        "candidate_mass_mean": float(candidate_mass.mean().item()),
        "reference_mass_min": float(reference_mass.min().item()),
        "reference_mass_mean": float(reference_mass.mean().item()),
        "mismatch_mass_max": float(mismatch_mass.max().item()),
        "mismatch_mass_mean": float(mismatch_mass.mean().item()),
        "max_kept_value_error": float(max_value_error),
    }


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


def benchmark_filter(
    fn: TensorFilter,
    logits: torch.Tensor,
    p: torch.Tensor,
    warmup: int,
    repeats: int,
) -> dict[str, float]:
    work = torch.empty_like(logits)
    for _ in range(warmup):
        work.copy_(logits)
        fn(work, None, p)
    torch.cuda.synchronize()

    samples_ms: list[float] = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    for _ in range(repeats):
        work.copy_(logits)
        start_event.record()
        fn(work, None, p)
        end_event.record()
        end_event.synchronize()
        samples_ms.append(float(start_event.elapsed_time(end_event)))

    return {
        "mean_ms": statistics.fmean(samples_ms),
        "median_ms": statistics.median(samples_ms),
        "min_ms": min(samples_ms),
        "p90_ms": percentile(samples_ms, 0.90),
        "max_ms": max(samples_ms),
    }


def main() -> int:
    args = parse_args()
    device = get_device()
    filter_fn = load_filter(args.implementation)
    distributions = [item.strip() for item in args.distributions.split(",") if item.strip()]
    unknown = set(distributions) - {"normal", "flat", "peaked", "ties"}
    if unknown:
        raise ValueError(f"unsupported distributions: {sorted(unknown)}")
    if args.logits_file is not None:
        distributions.append("real")

    device_name = torch.cuda.get_device_name(device)
    results: dict[str, object] = {
        "device": device_name,
        "torch_version": torch.__version__,
        "implementation": args.implementation,
        "vocab_size": args.vocab_size,
        "top_p": args.top_p,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "cases": [],
    }
    any_mismatch = False

    for distribution_index, distribution in enumerate(distributions):
        for batch_size in args.batch_sizes:
            if distribution == "real":
                assert args.logits_file is not None
                logits = load_real_logits(
                    args.logits_file,
                    batch_size,
                    args.vocab_size,
                    device,
                )
            else:
                logits = make_logits(
                    distribution,
                    batch_size,
                    args.vocab_size,
                    device,
                    args.seed + distribution_index * 1000 + batch_size,
                )
            p = torch.full(
                (batch_size,), args.top_p, dtype=torch.float32, device=device
            )

            candidate = filter_fn(logits.clone(), None, p)
            reference = pytorch_reference(logits.clone(), p)
            torch.cuda.synchronize()
            correctness = validate_support(candidate, reference, logits)
            any_mismatch |= correctness["mismatched_elements"] != 0
            timing = benchmark_filter(
                filter_fn,
                logits,
                p,
                args.warmup,
                args.repeats,
            )
            case = {
                "distribution": distribution,
                "batch_size": batch_size,
                "correctness": correctness,
                "timing": timing,
            }
            results["cases"].append(case)
            print(
                f"{distribution:>6} batch={batch_size:>2} "
                f"median={timing['median_ms']:.4f} ms "
                f"support_mismatches={correctness['mismatched_elements']}"
            )

    output_text = json.dumps(results, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output_text + "\n", encoding="utf-8")
        print(f"wrote {args.output}")
    else:
        print(output_text)

    if any_mismatch and args.require_exact_support:
        print("Top-p support mismatched the PyTorch reference", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
