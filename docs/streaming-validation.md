# 流式读取整合与验证记录

> 本文记录第一阶段的固定批次流式改造。2026-10-26 后续已加入 CUDA 七档显存预算、
> 4096 条窗口排序、动态分批与 OOM 回退，见 [显存预设验证](gpu-presets-validation.md)。
> 下文的“未整合自适应批次”和测试数量仅描述第一阶段，不代表当前 CUDA 默认行为。

## GitHub 基线与本地备份

- 仓库：<https://github.com/zhaoxi120/deepkoala>，分支 `main`。
- 开始与结束核验的远程提交均为 `f246bb9680531372d86d72e6b9b072fee13fc5b5`，与本地 HEAD 相同。
- 工作区原有差异包括机器专用 CLI/HMMER 路径、测试增强，以及模型资源的暂存删除。
- 原始工作区文件、SHA-256 清单、暂存与未暂存补丁已保存在 `local-backups/sync-20261023-181507/`。
- 修改前已将全部有差异的受跟踪文件与索引恢复至 HEAD，确认 `git diff HEAD` 为空。之后保留原有测试增强，再加入本次流式实现。额外 FASTA、模型和结果数据未清理。
- 本次仅修改本地工作区，没有提交或推送 GitHub。

## Issue #15 与附件审查

来源：[issue #15](https://github.com/zhaoxi120/deepkoala/issues/15)、
[Python 附件](https://github.com/user-attachments/files/32278098/deepkoala_stream.py)、
[README 附件](https://github.com/user-attachments/files/32278309/README_deepkoala_stream.md)。
查询时 issue 没有评论或关联时间线事件；检索未找到独立后续改进版或相关 PR。
附件原文及校验和保存在 `test-results/issue15-review/`。

附件采用分块读取、块内长度排序、动态批次、分片输出和断点续跑。
小规模真实模型运行与续跑测试通过：首次处理 2 条，续跑后共 7 条、4 个分片，
首个分片保持不变，合并解析后的预测与原实现一致。

审查发现以下限制，因此没有直接替换整个接口：

1. 单条序列超过 `max_padded_residues` 时仍会作为单条批次执行，突破声明的上限。
2. 默认每块 100 万条序列，且在内存保留整块结果；内存仍可能很大。
3. 默认模型、设备、输出目录/分片方式与当前接口不同；详细浮点文本格式也不同。
4. 空输入不产生 CSV；分片恢复不验证模型文件内容或输出内容。

## 已整合的行为

- `deepkoala/data.py`：主进程逐条读取 FASTA，把记录作为 sampler 的键交给 DataLoader。
  工作进程只负责分批转换，保留原始批次、重复 ID 与记录顺序，不各自重读整个文件。
- `deepkoala/infer.py`：每批预测后写 CSV，不累计全部输入或结果。
  先写同目录临时文件，成功后原子替换目标；后续批次失败时保留原目标。
- CSV 的列、KO 阈值比较、top-k、四位小数处理与统计定义保持一致。
- gzip/bz2/xz 直接流式压缩；ZIP 直接写成员；TAR 用磁盘临时文件确定长度，避免 pandas 的全量 BytesIO 缓冲。
- 原命令和 Python `inference()` 调用自动使用流式路径，无需新增参数。
- 多域模式保持原实现；公开的可索引 `ProteinDataset` 保持原有 eager 行为。
- 此次没有整合长度排序、自适应批次、分片和断点续跑。长蛋白仍可能需要降低 `batch_size`。

## 验证结果

最终完整回归：**118 passed、4 skipped、5 xfailed**，耗时约 25 秒。
其中新增的 **35 个流式回归用例全部通过**，包括单/双工作进程、显式 `spawn`、
LF/CRLF、重复 ID、大小写及未知字符、空/非法输入、top-k、失败保护、输入输出同路径，
以及 8 种压缩格式与归档成员内容。

4 项跳过为 CUDA/MPS 设备测试及真实 HMMER 测试；5 项预期失败是既有问题：
tab 分隔描述的解析差异、HMMER 临时文件未清理（两例）、wheel 缺少运行依赖声明和模型资源。
未通过跳过或预期失败掩盖新增回归。

GitHub 原始测试加本地额外测试的初始基线为 41 passed、1 failed、4 skipped、5 xfailed。
失败来自原始 `test_cli.py` 依赖未提供的 `tests/test.fasta`。恢复原有独立合成测试后消除了该依赖。
沙箱内多进程测试曾因本地套接字权限限制超时；最终结果来自获准的沙箱外运行。

真实模型一致性：对 `202502/full`、`202502/frag`、`202510/full`，分别使用
`fasta/case.fasta` 和包含不同长度、重复 ID、大小写及未知字符的合成输入，组合
simple/detail 与 top-k 1/3，共 **24 组**；CSV **逐字节一致**，总数和注释数也一致。
对照代码通过 `git show HEAD:deepkoala/{data,infer}.py` 获取，使用原始 eager loader。
这证明所测输入的软件输出兼容性，不代表重新验证生物学准确率或 GPU 行为。

## 读取内存实验

固定 300 aa/条、batch size 128、workers 0，遍历完整输入，使用 `tracemalloc`
记录 Python 分配峰值；不运行模型，不包含 PyTorch 原生张量内存或总进程 RSS。

| 序列数 | FASTA 字节数 | 原始读取峰值 | 流式读取峰值 |
| ---: | ---: | ---: | ---: |
| 1,000 | 308,890 | 676,865 B | 147,195 B |
| 100,000 | 31,088,890 | 67,683,555 B | 152,056 B |

输入增大 100 倍时，流式路径的 Python 内存峰值保持约 0.14–0.15 MiB。
批次中的最长蛋白、工作进程预取及模型仍决定实际内存需求。

## 环境与复现

实际环境为 Python 3.12.8、PyTorch 2.5.1.post106、NumPy 2.2.0、pandas 2.2.3、pytest 8.3.5，CPU。
NumPy/pandas 版本与 requirements.txt 中的固定版本不同，尚未在全部依赖版本或 GPU 上验证。
使用已有环境，并从另一环境借用纯 Python pytest 组件到 `/tmp/deepkoala-pytest-vendor`，未安装或升级依赖。

```bash
env PATH=/bio/user/ideas2/.conda/envs/deepmicroclass/bin:$PATH \
    PYTHONPATH=/tmp/deepkoala-pytest-vendor \
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
    python -m pytest --junitxml=test-results/streaming-full.xml
```

证据文件：

- `test-results/streaming-full.xml`：最终测试明细。
- `test-results/streaming-summary.json`：环境、基线、模型文件及源代码 SHA-256。
- `test-results/issue15-review/real_model_results.json`：24 组一致性、压缩输出及附件续跑结果。
- `test-results/streaming-memory/measure.py`、`results.json`：内存实验脚本与原始结果。

当地 `resources/202510` 的片段文件使用 `fragment` 命名，而接口寻找 `frag`；这是既有资源命名问题。
需要运行已测试片段模型时可显式选择 `--model frag --date 202502`。
