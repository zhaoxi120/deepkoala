# 最后一轮清理与验证

日期：2026-10-26（Asia/Tokyo）。本轮只运行轻量回归和两条序列的真实模型检查，
不重复已经完成的 10 万条本地对照、七档显存压力测试或超长序列矩阵。
原始记录保存在本地 `test-results/final-cleanup/`，不进入 GitHub CI 或上传附件。

后续已按要求修复本文记录的 FASTA ID 和 HMMER 临时文件问题，见
[已知问题修复记录](known-issue-fixes.md)。本文保留当时的测试结果。

## 代码清理

- 删除没有生产调用者的 `dynamic_batches`，保留 `AdaptivePredictor.predict_window`
  作为实际动态分批实现。把双预算、精确边界、末批和空窗口检查迁到实际调度器。
- 修复长输出文件名：原临时名拼接完整目标名称，可能超过文件系统单文件名长度上限；
  现在使用固定短前缀和随机标识。压缩格式仍根据最终输出路径判断。
- 保留原子替换、权限继承和失败清理；测试通过比较目录内容检查临时文件是否残留，
  避免依赖旧临时文件命名规则。
- 标准推理与 CPU 回退明确以 `weights_only=True` 加载模型状态字典。
- 删除 CLI 冗余赋值，修正文档中的 `-pd` 别名和已经过时的修改源码默认路径说明。

本轮没有更改七档预算、默认窗口条数、OOM 拆批或预测阈值。

## 轻量回归

使用与 CI 相同的选择范围：

```console
python -m pytest tests -m "not hardware and not hmmer and not benchmark" --junitxml=test-results/final-cleanup/ci-regression.xml
```

本地 Linux、Python 3.12.8、PyTorch 2.5.1 CPU：
**225 passed、4 deselected、5 xfailed**，26.72 秒，无新失败。

与上一轮相比，删除 11 个死函数专属测试实例，在实际调度器上增加 3 个参数化场景，
再增加 2 个长文件名回归，共减少 6 个实例。用例数量变化不表示取消生产路径的边界检查。

5 个严格预期失败仍是既有问题：两个 FASTA 读取器对 tab 描述的 ID 处理不一致、
HMMER 成功/失败后的临时 FASTA 清理两项，以及 wheel 依赖声明和模型资源打包两项。
这些问题没有被移除、放宽或标记为通过。当前源码运行检查也不代表 wheel 安装后的完整推理已经可用。

23 个 Python 文件语法检查、工作流 YAML 解析及轻量任务范围检查、`git diff --check` 均通过。
GitHub 工作流继续只包含四个 CPU 兼容性组合，仅上传 JUnit XML。
Windows/macOS 实际通过情况需要对应工作流结果；本地 Linux 检查不能替代它们。

## 真实模型小样本

前台 A100、PyTorch 2.4.1+cu121，使用 `fasta/case.fasta` 的两条真实序列
（142、282 aa）及本地 `202502` 权重，8 次独立短进程推理及 8 项断言全部通过：

| 检查 | 结果 |
| --- | --- |
| CPU `full/frag`，详细输出、top-k=3，与 GitHub 原版对照 | 各 6 行，逐字节一致 |
| CUDA `full`，默认自动分批与固定分批，简洁 top-1 输出 | 各 2 行，逐字节一致 |
| `full/frag` 预算触发 CPU 回退，与 CPU 详细输出对照 | 各 6 行，逐字节一致 |
| 两种模型的回退计数 | 各 2 次 CPU 回退、0 次 OOM |
| 源码、基准、输入及权重的前后哈希 | 一致 |

回退检查显式设置 `max_padded_residues=1`，触发的是单条序列超预算，
验证了懒加载 CPU 模型的实际执行；不是重新制造真实 CUDA OOM。
GPU 测试前后均无计算进程、显存使用为 0 MiB，未提交 PBS。
原始记录见本地 [小样本报告](../test-results/final-cleanup/real-model-smoke/REPORT.md)。

## 结果范围

历史性能与大规模对照保留在 [性能报告](performance-comparison.md)，属于清理前的实测版本。
本轮不提出新的吞吐或显存峰值结论。
此前 `gt32` 超长输入的详细概率差异 0.0006，以及近似并列候选的排序差异仍保留在历史报告；
本轮轻量检查不会撤销这些限制，也不保证任意输入的详细浮点输出完全一致。
