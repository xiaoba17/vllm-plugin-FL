"""Validate experimental production Top-p dispatch against current FlagGems."""

# ruff: noqa: I001
# Keep vllm_fl before vLLM imports: it installs the image compatibility shims.

import argparse
import json
import os
from pathlib import Path

import torch
import vllm_fl  # noqa: F401
import flag_gems
from vllm.v1.sample.ops import topk_topp_sampler as sampler
from vllm_fl.dispatch.backends.vendor.metax.patches.topk_topp_sampler import (
    _apply_top_k_top_p_no_triton as candidate,
)
from benchmark_metax_radix import timings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", default="1,8,16,32,64")
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    flag_gems.enable(unused=["masked_fill", "masked_fill_"])
    os.environ["VLLM_FL_METAX_RADIX_SORT"] = "1"
    reference = sampler.apply_top_k_top_p_pytorch
    report = {
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "sort_registered": "sort" in flag_gems.all_registered_ops(),
        "cases": [],
        "fallbacks": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(20261101)
    for batch in map(int, args.batches.split(",")):
        for name in (
            "normal",
            "bf16",
            "ties",
            "constant",
            "masked",
            "peaked",
            "mixed_p",
        ):
            x = torch.randn((batch, 130560), device="cuda")
            if name == "bf16":
                x = x.bfloat16().float()
            elif name == "ties":
                x = x.round()
            elif name == "constant":
                x.zero_()
            elif name == "masked":
                x[:, ::3] = -float("inf")
            elif name == "peaked":
                x[:, :32] += 12
            p = torch.full((batch,), 0.95, device="cuda")
            if name == "mixed_p":
                p = torch.tensor([0.001, 0.1, 0.5, 0.95, 1.0], device="cuda").repeat(
                    (batch + 4) // 5
                )[:batch]
            expected = reference(x.clone(), None, p)
            actual = candidate(x.clone(), None, p)
            mismatches = int((expected != actual).sum().item())
            if mismatches:
                raise AssertionError(f"{batch=} {name=} {mismatches=}")
            # Same noise, full probability chain, and default RNG advancement.
            rng = torch.cuda.get_rng_state()
            sampler.apply_top_k_top_p = reference
            expected_token, _ = sampler.TopKTopPSampler().forward_native(
                x.clone(), {}, None, p
            )
            expected_rng = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(rng)
            sampler.apply_top_k_top_p = candidate
            actual_token, _ = sampler.TopKTopPSampler().forward_native(
                x.clone(), {}, None, p
            )
            assert torch.equal(expected_token, actual_token), (batch, name, "tokens")
            assert torch.equal(expected_rng, torch.cuda.get_rng_state()), (
                batch,
                name,
                "rng",
            )
            before = timings(
                lambda x=x, p=p: reference(x.clone(), None, p), 2, args.repeats
            )
            after = timings(
                lambda x=x, p=p: candidate(x.clone(), None, p), 2, args.repeats
            )
            row = {
                "batch": batch,
                "distribution": name,
                "logit_mismatches": mismatches,
                "tokens_equal": True,
                "rng_equal": True,
                "reference": before,
                "candidate": after,
                "speedup": before["median_ms"] / after["median_ms"],
            }
            report["cases"].append(row)
            args.output.write_text(json.dumps(report, indent=2))
            print(json.dumps(row), flush=True)
    x = torch.randn((8, 130560), device="cuda")
    p = torch.full((8,), 0.95, device="cuda")
    k = torch.full((8,), 50, device="cuda", dtype=torch.int32)
    for name, value, topk, topp in (
        ("topk", x, k, p),
        ("noop", x, None, None),
        ("half", x.half(), None, p),
        ("strided", x[:, ::2], None, p),
    ):
        expected = reference(value.clone(), topk, topp)
        actual = candidate(value.clone(), topk, topp)
        assert torch.equal(expected, actual), name
        report["fallbacks"].append({"case": name, "equal": True})
    saved_sort = getattr(flag_gems, "radix_sort_rows", None)
    if saved_sort is not None:
        delattr(flag_gems, "radix_sort_rows")
        try:
            expected = reference(x.clone(), None, p)
            actual = candidate(x.clone(), None, p)
            assert torch.equal(expected, actual), "missing_flaggems_entrypoint"
            report["fallbacks"].append(
                {"case": "missing_flaggems_entrypoint", "equal": True}
            )
        finally:
            flag_gems.radix_sort_rows = saved_sort
    report["complete"] = True
    args.output.write_text(json.dumps(report, indent=2))
    print("VALIDATION_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
