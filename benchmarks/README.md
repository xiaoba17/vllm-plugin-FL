## Benchmark Overview

This directory provides two workflows:
- `benchmark_throughput_flagos.py`: throughput benchmarking for a served model.
- `benchmark_throughput_autotune.py`: auto-tune FlagGems operator selection by throughput.
- `benchmark_iluvatar_topp.py`: validate and microbenchmark Top-p filtering on
  Iluvatar GPUs.
- `benchmark_iluvatar_topp_sweep.py`: sweep launch-only parameters of the
  existing vLLM Top-p Triton kernel without changing its algorithm.

---

## benchmark_throughput_flagos.py

### Note
- Start an OpenAI-compatible inference service with `--served-model-name Qwen3-Next`.
  If you use a different name, update the string in `benchmark_throughput_flagos.py`.
- Run the benchmark on the same host as the service.

### Run
```bash
python3 benchmark_throughput_flagos.py
```

### Verify Results
1) Check failed requests:
```bash
grep "Fail" -rn vllm_bench_logs
```
All matches should show `Fail: 0`.

2) Generate statistics:
```bash
python3 benchmark_throughput_flagos_statistics.py
```

---

## benchmark_throughput_autotune.py


### Command
```bash
python benchmarks/benchmark_throughput_autotune.py [vllm args] [autotune options]
```

### vLLM Args
All arguments are passed directly to `vllm bench throughput`.

### Autotune Options
- `--background` (true/false): run in background mode, default `false`.
- `--ops`: comma-separated list of operator names to tune. If empty, auto-discovers ops.
- `--num-runs`: number of runs per configuration, default `2` (uses the second round to skip warmup).
- `--csv-path`: output CSV filename, default `history.csv` under the run directory.

Example:
```bash
python benchmarks/benchmark_throughput_autotune.py \
  --model /data/models/Qwen/Qwen3-Next-80B-A3B-Instruct \
  --tensor-parallel-size 4 \
  --dataset-name random \
  --input-len 6144 \
  --output-len 1024 \
  --num-prompts 1000 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 2048 \
  --load-format "dummy" \
  --gpu-memory-utilization 0.85 \
  --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}' \
  --background true
```

### Environment Variables
See [environment variables usage](../vllm_fl/dispatch/README.md#environment-variables) for the full dispatch and FlagGems configuration reference.

### Outputs
Each run creates a directory under `autotune_logs/autotune_xxx` with:
- `autotune.log`: full run log when setting `background` as `true`
- `history.csv`: throughput results for all rounds
- `autotune_ops.yaml`: final op list used for tuning
- `autotune_configs/`: per-round config snapshots
- `best_config.yaml`: selected best config

### Notes
- Round 1 runs baseline throughput without FlagGems.
- Round 2 benchmarks each op in isolation (whitelist).
- Round 3 validates the best-performing set.

---

## benchmark_iluvatar_topp.py

Run this benchmark inside the Iluvatar development environment before and
after changing the Top-p kernel. The competition model uses a vocabulary size
of 130560 and `top_p=0.95`, which are the script defaults.

```bash
python benchmarks/benchmark_iluvatar_topp.py \
  --batch-sizes 1,8,16,32,64 \
  --implementation triton \
  --output results/iluvatar-topp-baseline.json
```

The script compares the retained Top-p support and probability mass with
vLLM's ascending-sort PyTorch reference. The existing pivot kernel is
approximate, so differences are reported rather than hidden; use
`--require-exact-support` when testing an implementation that promises exact
support. The script also reports median and tail latency for each batch size
and synthetic logit distribution. Use `--logits-file` to add a saved real
`[batch, vocab]` logits tensor.

This is a diagnostic microbenchmark. It does not replace the official 4K/16K
throughput runs or the Math-500 accuracy evaluation, and it must not be used to
change the official sampling parameters.

### Iluvatar launch-parameter sweep

Use a saved real-logits tensor to compare block, truncation, warp, and grid
choices while reusing vLLM's kernel body verbatim:

```bash
python benchmarks/benchmark_iluvatar_topp_sweep.py \
  --logits-file results/iluvatar-topp-real-b64.pt \
  --block-sizes 2048,4096,8192 \
  --block-size-truncs 1024,2048,4096 \
  --num-warps 4 \
  --program-counts 16 \
  --output results/iluvatar-topp-sweep.json
```

Treat this as a search tool only. Re-run `benchmark_iluvatar_topp.py` with
`--implementation actual` after applying a candidate to the serving backend.
