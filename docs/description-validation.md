# `--description` 实现与验证记录

验证日期：2026-11-16（Asia/Tokyo）。

`--description` 已接入 CLI、普通推理和多结构域 CSV 推理。说明表按模型版本读取一次，解析后的标签映射复用于所有批次；默认关闭。原有模型、权重、预测逻辑和根 MIT 许可证未修改。KEGG 说明数据的独立使用条件见 [资源声明](../resources/README.md)。

## 使用方法和列格式

```bash
python3 -m deepkoala.cli -i input.fasta -o results.csv --date 202502 --description
python3 -m deepkoala.cli -i input.fasta -o detailed.csv --date 202502 --description --detail --topk 3
python3 -m deepkoala.cli -i input.fasta -o domains.csv --date 202502 --description --multi --profiles_dir /path/to/profiles --detail
```

Python CSV API 的对应参数是 `description=True`；`annotate_precision` 的单序列返回值接口不变。

| 模式 | 开启选项后的 CSV 列顺序 | 验证结果 |
| --- | --- | --- |
| 普通推理，简洁输出 | `name,predict_label,description` | 通过 |
| 普通推理，`--detail` | `name,predict_label,probability,threshold,annotate,description` | 通过 |
| `--multi`，简洁输出 | `name,predict_label,description` | 通过 |
| `--multi --detail` | `name,predict_label,probability,threshold,start,end,annotate,description` | 通过 |
| 未指定 `--description` | 保持原有列和结果；不读取或要求说明文件 | 通过 |

## 各种输入和错误情况

以下功能测试使用合成模型和合成说明数据，不依赖实际 KEGG 数据。

| 情况 | 实际行为 | 验证结果 |
| --- | --- | --- |
| `full` / `frag`，普通 / 多结构域，简洁 / 详细 | 8 种组合均只追加最后一列；原有各列、行序、统计与关闭选项时一致 | 通过 |
| 单 KO，说明存在 | 原样写入该 KO 的说明 | 通过 |
| 简洁输出，预测低于阈值 | 标签和说明同时为空 | 通过 |
| 详细输出，预测低于阈值 | 保留候选标签及说明，`annotate` 仍为空 | 通过 |
| `--topk 1` / `--topk 2` / 超过类别数的 `--topk` | 对每个实际输出候选逐行映射，不改变原有 top-k 行为 | 通过 |
| 复合标签 `K00001+K00002` | `K00001: first definition \| K00002: second definition`，保留标签组件顺序 | 通过 |
| 缺少单 KO 说明 | 保留预测标签，说明为空；按模型涉及的唯一缺失 KO 汇总警告一次 | 通过 |
| 复合标签缺少某个组件 | 保留其余说明，将缺失组件写成 `K00002: [description unavailable]` | 通过 |
| 大量缺失 KO / 同一 KO 多次出现 | 警告按唯一 KO 计数，最多列 10 个示例；不会逐行重复警告 | 通过 |
| 空 FASTA | 正确的 CSV 表头包含 `description`，零数据行、零统计 | 通过 |
| 多结构域无候选 / 序列短于处理阈值 | 保留原有空标签行，说明为空 | 通过 |
| 递归识别多个结构域 | 每个结构域分别映射说明，坐标与输出顺序保持一致 | 通过；HMMER 响应受控模拟 |
| 多批次、重复序列 ID | 表头只写一次，每行与原有输入顺序对应 | 通过 |
| CUDA 自适应排序和 OOM 重试 | 恢复原顺序后逐行添加说明，无重复或漏行 | 通过；CPU 上模拟调度及 OOM |
| UTF-8 / UTF-8 BOM，逗号、引号、中文、换行 | 正确读取；输出可由标准 CSV 读取器完整还原 | 通过 |
| `.csv.gz` / `.csv.zip` / `.csv.tar.gz` | 多批输出与说明文本可完整读回 | 通过 |
| 缺少说明文件且开启选项 | 在读取权重或处理 FASTA 前报错；已有输出不被覆盖，新输出不生成 | 通过 |
| 说明表重复 KO、空说明、非法 KO、字段数不对、引号不闭合等 | 明确报错，保留已有输出 | 通过；13 种非法表格 |
| 关闭选项，但说明文件缺失或损坏 | 正常运行，不读取说明文件，不产生说明相关警告 | 通过 |
| `latest` 选中新版但新版无说明文件 | 明确报错，不回退使用旧版说明表 | 通过 |
| 显式选择有说明文件的旧版 | 读取该版本，正常输出 | 通过 |
| CLI help、选项转发、独立进程端到端运行 | 两种推理模式都能正确识别并传递选项 | 通过 |

## 当前实际资源检查

`resources/202502/ko_description.csv` 有 28,138 个唯一 KO，格式有效，无重复编号或空说明。

| 模型配置 | 总标签数 | 复合标签数 | 缺失单 KO 说明数 |
| --- | ---: | ---: | ---: |
| `202502/full` | 26,469 | 243 | 17 |
| `202502/frag` | 26,226 | 0 | 17 |

243 个复合标签的所有组成 KO 都有说明。缺少的 17 个单 KO 是：

```text
K04551 K06419 K06420 K06421 K07015 K07895 K08511 K09462 K12364
K12738 K18707 K24252 K24256 K24301 K24302 K24965 K27971
```

另用随库的实际 `202502` full、frag 权重，各自对 `tests/fixtures/synthetic.fasta` 的 3 条合成序列执行关闭/开启选项的对照运行，使用 CPU、`detail=True`、`topk=3`、每批 2 条序列。两种模型各输出 9 行，开启后 9 行均匹配到说明；除新增列外，原有所有字段完全一致，两次运行的 `total=3`、`annotated=0` 一致。这验证实际资源兼容性，不是生物学准确率评估。

当前本地 `latest` 解析为 `202510`。该目录没有 `ko_description.csv`，实际验证 `--description` 会因缺文件而报错且不生成输出；目前请显式使用 `--date 202502`，或取得相应版本的说明文件。

本地 `202510` 的片段资源仍使用 `fragment` 文件名，而现有接口使用 `frag`；这是已有资源命名差异，本次没有改动。

## 回归结果与范围

- 新增 `tests/test_descriptions.py` 的 53 项测试，并扩展现有 CLI help 与 CUDA 自适应排序/OOM 模拟测试。
- 完整 CPU 回归：**291 passed，4 deselected，0 failed**，耗时 22.95 秒。
- 未执行的 4 项测试属于真实加速器 / HMMER 测试。没有运行真实 CUDA、MPS 或 HMMER；多结构域行为使用合成模型、缺失 profile 回退与受控 HMMER 响应验证。
- 测试仍报告原有 `torch.load` FutureWarning 与多线程进程 `fork()` DeprecationWarning；没有为本功能屏蔽这些警告。
- 沙箱首次运行阻止现有多进程测试创建本地通信 socket；完整回归已在常规环境重跑通过，未添加跳过标记或改变多进程实现。

环境：Python 3.12.8、PyTorch 2.5.1.post106、NumPy 2.2.0、pandas 2.2.3、pytest 8.3.5。使用已有环境，没有安装或升级依赖；这不代表验证了所有固定依赖版本与平台组合。

本地证据（位于 Git 忽略的 `test-results/` 中）：

- 完整回归 XML：`test-results/description/full-regression.xml`
- 功能测试 XML：`test-results/description/feature-tests.xml`
- 实际资源运行摘要：`test-results/description/real-resources-summary.json`
- full 模型新增列结果：`test-results/description/real-202502-full-description-true.csv`
- frag 模型新增列结果：`test-results/description/real-202502-frag-description-true.csv`

这些原始结果仅保存在本地，不随 GitHub 仓库公开分发。后续随机输入测试和发布检查见 [发布准备记录](release-readiness.md)。
