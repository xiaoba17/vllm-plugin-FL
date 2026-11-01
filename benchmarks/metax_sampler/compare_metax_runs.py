"""Check all four official runs, compare rounds 2-4, and save raw evidence."""

import argparse
import csv
import json
import statistics
from pathlib import Path


def read_runs(directory):
    paths = list((directory / "benchmark_results").glob("raw_runs_*.csv"))
    if len(paths) != 1:
        raise ValueError(f"expected one raw CSV in {directory}, got {paths}")
    with paths[0].open() as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 4 or any(row["Run Status"] != "SUCCESS" for row in rows):
        raise ValueError(f"four successful official runs required: {rows}")
    for row in rows:
        if float(row["Successful Requests"]) != float(row["Num Prompts"]):
            raise ValueError("request count mismatch")
    scored = rows[1:]
    return {
        "raw_csv": str(paths[0]),
        "rows": rows,
        "total_tps_mean": statistics.mean(float(r["Total tok/s"]) for r in scored),
        "ttft_mean_ms": statistics.mean(float(r["Mean TTFT (ms)"]) for r in scored),
        "tpot_mean_ms": statistics.mean(float(r["Mean TPOT (ms)"]) for r in scored),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-improvement", type=float, default=None)
    parser.add_argument("--max-ttft-change", type=float, default=None)
    parser.add_argument("--max-candidate-ttft-ms", type=float, default=None)
    parser.add_argument("--min-candidate-tps", type=float, default=None)
    args = parser.parse_args()
    baseline, candidate = read_runs(args.baseline), read_runs(args.candidate)
    gain = candidate["total_tps_mean"] / baseline["total_tps_mean"] - 1
    report = {
        "baseline": baseline,
        "candidate": candidate,
        "throughput_gain_percent": gain * 100,
        "ttft_change_percent": (
            candidate["ttft_mean_ms"] / baseline["ttft_mean_ms"] - 1
        )
        * 100,
        "tpot_change_percent": (
            candidate["tpot_mean_ms"] / baseline["tpot_mean_ms"] - 1
        )
        * 100,
    }
    report["criteria"] = {
        "min_relative_throughput_gain": args.require_improvement,
        "max_relative_ttft_increase": args.max_ttft_change,
        "max_candidate_ttft_ms": args.max_candidate_ttft_ms,
        "min_candidate_total_tps": args.min_candidate_tps,
    }
    report["checks"] = {
        "throughput_gain": args.require_improvement is None
        or gain >= args.require_improvement,
        "relative_ttft": args.max_ttft_change is None
        or candidate["ttft_mean_ms"]
        <= baseline["ttft_mean_ms"] * (1 + args.max_ttft_change),
        "official_ttft": args.max_candidate_ttft_ms is None
        or candidate["ttft_mean_ms"] <= args.max_candidate_ttft_ms,
        "official_throughput_floor": args.min_candidate_tps is None
        or candidate["total_tps_mean"] >= args.min_candidate_tps,
    }
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    if args.require_improvement is not None and gain < args.require_improvement:
        raise SystemExit(3)
    if args.max_ttft_change is not None and candidate["ttft_mean_ms"] > baseline[
        "ttft_mean_ms"
    ] * (1 + args.max_ttft_change):
        raise SystemExit(4)
    if (
        args.max_candidate_ttft_ms is not None
        and candidate["ttft_mean_ms"] > args.max_candidate_ttft_ms
    ):
        raise SystemExit(5)
    if (
        args.min_candidate_tps is not None
        and candidate["total_tps_mean"] < args.min_candidate_tps
    ):
        raise SystemExit(6)
