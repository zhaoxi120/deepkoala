# 使用方式与兼容性

本页比较 GitHub 基线提交 `f246bb9680531372d86d72e6b9b072fee13fc5b5` 中的
`deepkoala/cli.py`、`infer.py`、`data.py` 与当前工作区。原命令和
`inference()` 调用仍可使用；CUDA 默认启用自适应批次，不能据此假定运行批次或浮点结果完全不变。

| 项目 | 原版 | 当前版本 |
| --- | --- | --- |
| 输入、输出 | `--input_path/-i`、`--output_path/-o` | 名称、别名及必填要求保持不变 |
| 模型、日期 | `--model/-m`，默认 `full`；`--date/-d`，默认 `latest` | 保持不变 |
| 设备 | `--device auto/cpu/cuda/mps`，默认 `auto` | 保持不变；`auto` 仍优先 CUDA，其次 MPS、CPU |
| CLI 未指定批次 | 固定 32 条 | CUDA 自适应模式最多 512 条；CPU/MPS 或 `--memory-preset off` 仍为 32 条 |
| Python API 未指定批次 | 固定 64 条 | CUDA 自适应模式最多 512 条；CPU/MPS 或 `memory_preset="off"` 仍为 64 条 |
| 显式 `--batch_size/-bs` / `batch_size` | 固定批次条数，末批可不足 | 自适应模式下作为条数上限，实际批次可更小；固定模式保持原意 |
| `--num_workers/-nw` / `num_workers` | 默认 2 个 DataLoader 工作进程 | 固定模式保持原意；CUDA 自适应模式不使用这个参数，按实际批次在读取进程转换序列 |
| `--detail`、`--topk/-tk` | 详细输出及每条序列的 top-k | 保持原定义，top-k 仍不超过模型类别数 |
| `--multi`、`--profiles_dir/-pd` | 多域模式，必须提供 profiles 目录 | 原执行路径保持不变，不启用标准推理的窗口排序与显存自适应 |

CUDA 自适应模式默认在每 4096 条序列内按长度分组，再按输入序号恢复输出顺序。
当前已取消窗口的 4,000,000 残基上限；长度不会提前截断窗口，也不会截断序列。
主机内存因而随窗口内序列的总长度变化；GPU 每批的条数和 padded-residue 预算仍单独生效。

下面的旧命令无需改写。指定 202502 是为了使用已有的完整 `full/frag` 资源：

```bash
python -m deepkoala.cli -i proteins.fasta -o annotations.csv \
    -m full -d 202502 --device cuda -bs 64 -nw 2

python -m deepkoala.cli --input_path proteins.fasta --output_path details.csv \
    --model frag --date 202502 --device cpu --batch_size 32 \
    --num_workers 0 --detail --topk 3
```

第一条命令现在会在 CUDA 上排序并动态分批，批次不超过 64 条；`-nw 2` 不参与这条自适应路径。
若需要原来的固定分批方式，加一个选项即可，流式读写仍保留：

```bash
python -m deepkoala.cli -i proteins.fasta -o annotations.csv \
    -m full -d 202502 --device cuda -bs 64 -nw 2 --memory-preset off
```

新的选项均为可选项。默认自动选显存档位；如需主动选择 8 GiB 档并允许无法放入 GPU 批次的单条序列转 CPU：

```bash
python -m deepkoala.cli -i proteins.fasta -o annotations.csv \
    -m full -d 202502 --device cuda --memory-preset 8 --oom-fallback cpu
```

可选档位为 `2/4/8/16/24/32/gt32`；`gt32` 使用名义 64 GiB 的预算。
`--sort-window` 调整窗口条数；`--max-padded-residues` 进一步限制 CUDA 自适应批次。
显存预算是调度目标，OOM 时拆分批次重试；默认 `--oom-fallback error`，不会自动改用 CPU。
显式数字档位要求 CUDA；多域模式拒绝显式数字档位、CPU OOM fallback 和 padded-residue 参数。
详细说明及历史 GPU 实测见 [显存预设验证](gpu-presets-validation.md)，
取消窗口残基限制后的实测见 [当前性能对比](performance-comparison.md)。

Python 调用保留原参数顺序和名称，新增选项附加在原参数之后：

```python
from deepkoala.infer import inference

# 原调用仍有效；CUDA 上的 batch_size 现在表示批次条数上限。
stats = inference("proteins.fasta", "annotations.csv", batch_size=64,
                  device="cuda", date="202502")

# 恢复固定分批，继续使用流式读写。
stats = inference("proteins.fasta", "annotations.csv", batch_size=64,
                  device="cuda", date="202502", memory_preset="off")
```

返回值始终保留 `total`、`annotated`、`output`。CUDA 自适应模式额外返回 `batching`
字典，包含所选档位、预算、有效批次上限、显存峰值和 OOM/CPU fallback 计数。
只读取原三个键的代码无需调整；严格断言返回字典只有三个键的程序应允许新增键，或使用固定模式。

CSV 简洁格式仍为 `name,predict_label`；详细格式仍为
`name,predict_label,probability,threshold,annotate`。重复 ID 不合并，记录顺序按原 FASTA 恢复；
每条记录内部继续按 top-k 输出。阈值比较仍使用未舍入概率，四位小数只影响输出；
简洁模式中未通过阈值的预测标签留空。统计仍按序列计数，不按输出行计数。

窗口排序改变批次组成，CUDA 运算和可选 CPU fallback 可能产生细微浮点差异。
接近并列的类别可能交换 top-k 顺序，接近阈值的预测也可能改变注释标记；因此不能保证自适应结果逐字节等同原版。
`--memory-preset off` 恢复原分批方法，可用于对照；跨设备、库版本或资源版本仍不能笼统保证完全相同。
CPU 固定模式的真实模型一致性测试记录见 [流式改造验证](streaming-validation.md)。

标准推理按批次或窗口写入同目录临时文件，完整成功后才替换目标；失败时保留已有目标。
支持原有基于扩展名的压缩输出：gzip/bz2/xz 及 ZIP 流式写入，TAR 使用磁盘临时 CSV，避免把完整结果放入内存。
压缩容器的字节内容、元数据及归档内部文件名不保证与原 pandas 实现完全相同，应比较解压后的 CSV。
这不是断点续跑功能；一次运行中断后仍需重新执行输入。

原子替换会改变目标文件的 inode：目标路径若为符号链接，将替换该链接本身；其他硬链接仍指向原文件。
在 POSIX 系统上，写入中的临时文件保持私有；发布时保留已有目标的权限位，新文件采用普通文件创建时的 umask 规则，
因此共享目录中的读取权限不因原子写出变成仅所有者可读。替换 inode 不保证保留原文件的所有者或扩展 ACL 等元数据。

FASTA 标识符现在统一取 `>` 后第一个以空白分隔的字段，空格和 Tab 都可分隔描述。
例如 `>sequence<TAB>description` 在 `ProteinDataset`、固定批次、排序窗口和多域模式中均得到 `sequence`。
这是对旧标准模式的兼容性修正：旧版可能将 Tab 后描述也保留在输出 ID 中；使用此类标题关联历史结果时需注意。
仅含空白的标题会明确报错。序列字符编码规则保持原逻辑；流式解析器仍对无标题内容和空序列提供明确错误。
直接使用内部 `get_dataloader()` 的程序需要注意：流式 loader 不预先知道记录总数，不能再依赖
`len(loader)` 或 `loader.dataset.entries`。单独的 `ProteinDataset` 仍保持原来的全量加载与索引访问行为。

多域模式执行 HMMER 后，临时 FASTA 在成功、无命中和异常路径中均会清理。

2026-10-26 的独立 CLI 验证使用真实 202502 权重、`fasta/case.fasta`、CPU、workers=0，
分别从 Git HEAD 提取原包与运行当前包。以下四组全部返回成功，CSV 逐字节一致，结束统计均为
`Processed 2 sequences, annotated 1.`：

| 验证命令类型 | 结果 |
| --- | --- |
| 短参数、full、默认批次、简洁输出 | 与原版一致 |
| 长参数、frag、batch=2、detail、top-k=3 | 与原版一致 |
| 当前版本加 `--memory-preset off`，与原固定模式比较 | 与原版一致 |
| `--multi`、显式空 profiles 目录、详细输出 | 与原版一致；未覆盖真实 HMMER 边界搜索 |

这些小规模检查验证命令与输出兼容性，不衡量吞吐；不同规模与设备的性能应以相应基准结果为准。
原始结果保存在 [CPU CLI 验证记录](../test-results/usage-compatibility/cpu-cli-results.json)。
文件权限回归 **8 项通过**，覆盖已有 `0640/0444`、新文件正常创建权限、失败时保留原文件、
独立进程中的 `umask 077/002`，以及缺少 `os.fchmod` 时的兼容分支。
结果保存在 [文件权限测试记录](../test-results/usage-compatibility/output-permissions.xml)。
