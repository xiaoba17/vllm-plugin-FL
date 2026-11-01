# 沐曦稳定 Top-p 排序候选

本目录保存龚锦的沐曦采样优化及可复用测试。独立于天数采样实验，默认不启用；不修改模型、权重、采样参数、官方 benchmark 或官方服务参数。

## 分支与验收状态

- 团队基线：插件 `13eb9be69ecc5b5ca4f79c44e9ee40081eaa1bf0`。
- 个人稳定分支：`opt/gongjin-metax`，从团队 `storage-opt` 建立。
- 本项开发分支：`feat/metax-topp-radix-sort`，从该个人分支建立。
- 本次仅保存 feature 提交；尚未向个人分支或团队集成分支合并。
- 2026-11-01 的硬件测试发生在提交之前，通过安装源码的 SHA-256 对应到本次保存的生产文件；不能冒充“服务器拉取确定 commit 后的正式复测”。
- 按团队流程，下一步应在服务器个人隔离仓库拉取该 commit，重启个人服务并正式复测，再提 `feat → opt/gongjin-metax` PR。个人分支组合回归后才提 `opt → storage-opt` PR。

本地完整证据仍保留在工作区外层的 `profiling/results/metax-sampler-20261101/`，独立压缩包为 `profiling/archives/metax-sampler-evidence-20261101.zip`；它们不属于此 Git 仓库。Git 只保留必要的四轮 CSV 和摘要，不上传模型、凭据、完整推理输出或大体积日志。

## 实现与开关

生产文件：

- `vllm_fl/dispatch/backends/vendor/metax/impl/radix_sort.py`
- `vllm_fl/dispatch/backends/vendor/metax/patches/topk_topp_sampler.py`

使用四个 8-bit pass 的稳定分块 radix sort，保持相同值的索引顺序与现有 FlagGems 排序一致。softmax、cumsum、mask、scatter 和随机采样不变。仅适用于 CUDA/MetaX 上连续 FP32 `[batch,130560]` logits、无 Top-k、有 Top-p，且全局 FlagGems `sort` 已注册；其他情况回退原路径。

启动原有服务时额外设置 `VLLM_FL_METAX_RADIX_SORT=1` 才启用。缺省值或设置 `0` 均使用原路径，切换开关须重启服务。保留 `temperature=1.0`、`top_p=0.95`。

硬件验证对应的生产源码 SHA-256（本次提交未格式化或修改它们）：

```text
17cefb665c2bdbed21dc3e8ff9862dd4cc40a0d95f9c3df5200b816fd71f197c  radix_sort.py
fe3b0f5c0ba00bed21aa78835d55817018ea5ea99047e12187b13a9f60f8ca94  topk_topp_sampler.py
```

## 已有硬件验证（提交前）

设备 C500 64 GiB；驱动 3.8.30、MACA 3.7.0.38；PyTorch 2.8.0+metax3.7.0.7、Triton 3.0.0+metax3.7.0.7、vLLM 0.24.0+empty、插件 0.3.0rc1.post1+g13eb9be69、EvalScope 1.11.1。FlagGems 5.3.5 对应 `a7620cc191a0b42e040194622c5758b22a7a25dc`，全局启用，`masked_fill` / `masked_fill_` 保持原 blacklist。

| 场景 | 基线 total tokens/s | 候选 total tokens/s | 变化 | 基线 TTFT | 候选 TTFT |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4K | 4582.80 | 7567.62 | +65.13% | 3304.51 ms | 3201.54 ms |
| 16K | 7014.65 | 8219.49 | +17.18% | 27239.49 ms | 26304.12 ms |

两组每场景各四轮，取第 2–4 轮均值；4K 每轮 256/256、16K 每轮 128/128 请求成功。无 profiler。4K 候选复用同实例较早的基线，基线存在波动，未执行反向交错 A/B，不能将全部收益归因于 `libentry` 修订。

35 组语义用例覆盖 batch 1/8/16/32/64 和 normal、BF16、ties、constant、masked、peaked、mixed-p：过滤后 logits、token 和 RNG state 均一致。另有 Top-k、无过滤、FP16、不同词表宽度回退校验。BF16 合成过滤微基准 batch 64 为 28.129 → 5.258 ms；batch 1 为 2.996 → 1.213 ms。CUDA event 计时包含输入复制及主机 launch 间隙，不等同于纯 kernel 执行时间。

Math-500 Level 3：105 个唯一预测和 105 个成功评分，正确 102/105，报告 accuracy **0.9714**，无预测错误；104 个正常结束，1 个达到原定 max_tokens，未剔除样本。

[官方规则公开接口](https://flagos.io/flagos/api/v1/races/539vlt2p)在 2026-11-01 核验：吞吐 ≥ 官方 baseline × 0.99、TTFT ≤ 官方 baseline × 1.01、accuracy ≥ 0.95。本轮均通过，但 4K TTFT 距 3231.42935 ms 上限仅约 30 ms，提交或组合优化前必须复测。天数未验证此候选，不推广沐曦结果。

## 复现入口

在包含芯片依赖的个人 Python 环境、仓库根目录执行（输出目录自行选择）：

```bash
python benchmarks/metax_sampler/benchmark_metax_radix.py \
  --module vllm_fl/dispatch/backends/vendor/metax/impl/radix_sort.py \
  --enable-flaggems --batches 1,8,64 \
  --output /data/my-metax-sort.json

VLLM_PLUGINS=fl python benchmarks/metax_sampler/validate_metax_sampler.py \
  --batches 1,8,16,32,64 --repeats 5 \
  --output /data/my-metax-sampler-validation.json
```

`validate_metax_sampler.py` 在导入 vLLM 前加载插件兼容 shim，因此有意保留特殊导入顺序；本次保存对辅助脚本进行了格式化，并显式绑定计时闭包的循环变量，未改其计算逻辑。

正式性能继续使用原 `benchmarks/benchmark_throughput_serve.py`：4K `[4096,1024,64,256]`，16K `[16384,1024,64,128]`，分别在开关 `0` 和 `1` 的新服务中运行，使用相同环境、GPU 文件锁及官方启动参数。模型、服务名、端口由个人环境指定。每次记录插件与 FlagGems 的完整 commit SHA、工作区状态及结果目录。

可在无 GPU 的本地重算本次保存的四轮证据（原 CSV 仅规范化 CRLF → LF，数值未改）：

```bash
evidence=benchmarks/metax_sampler/evidence/20261101
python benchmarks/metax_sampler/compare_metax_runs.py \
  "$evidence/baseline-4k" "$evidence/candidate-4k" \
  --output /tmp/metax-4k-check.json --require-improvement 0.01 \
  --max-ttft-change 0.01 --max-candidate-ttft-ms 3231.42935 \
  --min-candidate-tps 5038.74855
python benchmarks/metax_sampler/compare_metax_runs.py \
  "$evidence/baseline-16k" "$evidence/candidate-16k" \
  --output /tmp/metax-16k-check.json --require-improvement=-0.01 \
  --max-ttft-change 0.01 --max-candidate-ttft-ms 27469.10635 \
  --min-candidate-tps 6959.37825
```

精度按原 EvalScope 配置重新运行：本地 Math-500 Level 3 全部 105 题、temperature 1、top-p 0.95、max_tokens 32768、并发 8、timeout 3600。即使使用 `--ignore-errors`，也必须核对实际样本覆盖、错误、评分状态及 accuracy，不能只看进程退出码。

## 本地可运行检查

Python 语法编译、`git diff --check`、辅助脚本 Ruff 检查，以及以上 CSV 门槛重算。Mac 无 C500，未在本次 Git 保存阶段重新运行 GPU 或完整模型测试；提交前硬件测试与未来按 commit 正式复测应分开标注。
