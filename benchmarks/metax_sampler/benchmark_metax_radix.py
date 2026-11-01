"""Standalone experiment; no serving patch or benchmark parameter changes."""

import argparse
import importlib.util
import json
import statistics
from pathlib import Path

import torch


def timings(function, warmup, repeats):
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    elapsed = []
    for _ in range(repeats):
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        function()
        end.record()
        end.synchronize()
        elapsed.append(start.elapsed_time(end))
    return {"median_ms": statistics.median(elapsed), "samples_ms": elapsed}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", default="1,8,64")
    parser.add_argument("--distributions", default="normal,bf16,ties,constant,masked")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--block", type=int, default=512)
    parser.add_argument(
        "--enable-flaggems",
        action="store_true",
        help="Match serving's global FlagGems dispatch, including sort",
    )
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("candidate_radix", args.module)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if args.enable_flaggems:
        import flag_gems

        flag_gems.enable(unused=["masked_fill", "masked_fill_"])
    torch.manual_seed(20261101)
    report = {
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(),
        "block": args.block,
        "flaggems_enabled": args.enable_flaggems,
        "cases": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for batch in map(int, args.batches.split(",")):
        for distribution in args.distributions.split(","):
            x = torch.randn((batch, 130560), device="cuda")
            if distribution == "bf16":
                x = x.to(torch.bfloat16).float()
            elif distribution == "ties":
                x = x.round()
            elif distribution == "constant":
                x.zero_()
            elif distribution == "masked":
                x[:, ::3] = -float("inf")
            elif distribution != "normal":
                raise ValueError(distribution)
            reference, reference_ids = x.sort(dim=-1)
            candidate, candidate_ids = module.radix_sort_rows(x, args.block)
            row = {
                "batch": batch,
                "distribution": distribution,
                "value_mismatches": int((candidate != reference).sum().item()),
                "index_mismatches": int((candidate_ids != reference_ids).sum().item()),
            }
            row["reference"] = timings(
                lambda x=x: x.sort(dim=-1), args.warmup, args.repeats
            )
            row["candidate"] = timings(
                lambda x=x: module.radix_sort_rows(x, args.block),
                args.warmup,
                args.repeats,
            )
            row["speedup"] = (
                row["reference"]["median_ms"] / row["candidate"]["median_ms"]
            )
            report["cases"].append(row)
            args.output.write_text(json.dumps(report, indent=2))
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
