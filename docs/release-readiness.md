# GitHub 更新准备与随机推理验证

检查日期：2026-11-16（Asia/Tokyo）。本记录针对 `--description` 功能、数据许可说明和相关测试变更。

## 结论与 Git 状态

本地检查未发现阻止本次功能更新提交的新问题。完整 CPU 回归、随机序列的真实模型推理及拟发布目录验证通过。仍需由 GitHub Actions 验证其平台矩阵；本地结果不代表 Windows、macOS 或真实 GPU 已经通过。

实时查询 GitHub 后，本地 `HEAD`、本地 `origin/main` 与远端 `main` 均为 `e713b0f0ef48ec383ebe1b868e286921bcd06305`。本次修改尚未提交或推送；没有新建分支、PR 或修改远端。

## 测试流程检查

现有 `.github/workflows/tests.yml` 在 main push、pull request 和手动触发时执行 CPU 测试。矩阵为 Linux / Windows / macOS 的 Python 3.11 + PyTorch 2.4.1，以及 Linux 的 Python 3.12 + 当前 `<3` 的 PyTorch。稀疏检出只包括代码、测试和根配置；测试生成小型合成权重及说明文件，不下载 KEGG 数据或生产模型。输出仅上传 JUnit 报告。

已新增 `tests/test_random_inference.py`，现有 workflow 会自动发现，无需修改 workflow：

- 固定种子 `20261116`，用 20 种标准氨基酸生成 9 条随机蛋白序列。
- 长度为 `127, 49, 1, 64, 50, 17, 129, 51, 63`，顺序不按长度排序，包含多结构域模式的 50 aa 边界。
- 4 项测试覆盖 full / frag × 普通 / 多结构域模式；每项分别开启和关闭 `--description`，共 8 次 CLI 端到端运行。
- 检查进程成功、输出列、行序、统计一致性与说明映射；CI 使用合成模型和空 profile 目录，不执行真实 HMMER。
- 不校准阈值，不评价随机序列的注释准确率，也不要求随机序列必须全部未注释。

完整 CPU 回归采用与 CI 相同的测试选择：**295 passed，4 deselected，0 failed**，耗时 36.63 秒。现有 `torch.load` 和多线程 `fork()` 警告仍保留；没有为本次功能新增跳过或忽略规则。

## 随机序列与真实资源

另在本地使用真实 `resources/202502` full / frag 权重和说明表进行验证：

| 项目 | 实际值 |
| --- | --- |
| 随机种子 | `20261116` |
| 序列数 | 32 |
| 长度范围 | 1–1200 aa |
| 总残基数 | 11,409 |
| 短于 50 aa 的序列 | 6 |
| 计算设备 | CPU，1 个 PyTorch 线程 |
| 运行组合 | full / frag × 普通 / multi × 简洁 / detail × top-k 1 / 3 × description 关 / 开 |
| 总运行数 | 32，全部成功 |
| 对照组数 | 16，原有所有输出列、行序和统计完全一致 |
| 输出总行数 | 1,952 |
| 总耗时 | 15.354 秒；此数字仅记录本次运行，不作为性能基准 |

开启选项的所有输出说明均由独立 CSV 读取和映射检查验证。普通模式每次输出 32 行（top-k 1）或 96 行（top-k 3）；多结构域模式为 32 或 84 行，包含 6 条短序列的空标签占位行。

多结构域运行使用本地真实 profiles 和 HMMER 3.4。实际调用 `hmmsearch` 8 次，均未命中，正常走边界回退；没有伪造命中或替换 HMMER 返回值。另执行真实 HMMER 的独立测试：**2 passed，1 xfailed**。预期失败仍是既有的“profile 路径包含空格未被 shell 正确引用”问题，未由 `--description` 引入，也未在本次修复。

没有运行真实 CUDA / MPS；自适应批处理和 OOM 路径仍由已有 CPU 模拟测试覆盖。本次不评价阈值、生物学准确率或大型数据集性能。

## 下载后直接运行的检查

另外组装一个只包含拟发布代码与 `202502` 资源的本地目录，排除未跟踪的 `202510`。使用同一批 32 条随机序列，通过 CLI 分别运行 full 和 frag，指定 `--description --detail --topk 3`，省略 `--date`。

两次均退出成功，各输出 96 行，与显式选择 `202502` 的对应结果完全一致。因此，包含新增 CSV 的该发布目录可使用默认 `latest` 运行。原工作目录仍含未跟踪 `202510`，默认选择差异属于本地资源布局，不能据此判断全新下载会失败。

## 本次提交范围

建议只纳入以下文件；新增未跟踪文件不会显示在普通 `git diff --stat` 中：

```text
README.md
deepkoala/cli.py
deepkoala/descriptions.py
deepkoala/infer.py
deepkoala/infer_multi.py
docs/description-validation.md
docs/release-readiness.md
docs/testing.md
resources/README.md
resources/202502/ko_description.csv
tests/conftest.py
tests/test_adaptive.py
tests/test_cli.py
tests/test_descriptions.py
tests/test_random_inference.py
```

根 `LICENSE` 保持原样；按用户要求没有创建版本内的 `KEGG_DATA_NOTICE.txt`。两个 README 精确说明 KO 数据不属于代码的 MIT 许可，没有改变其他资源文件的许可范围。

本地的 `profiles.tar.gz`（约 1.5 GB）、解压后的 `profiles/`、`resources/202510/`、`fasta/`、`results/`、`deepkoala.sh` 和运行日志不在本次提交范围内；不应使用无差别的 `git add -A`。已有跟踪的 `202502` 模型权重和配置保持不变。

## 本地证据与复现

实际环境：Python 3.12.8、PyTorch 2.5.1.post106、NumPy 2.2.0、pandas 2.2.3、pytest 8.3.5、HMMER 3.4。使用已有环境，没有安装或升级依赖。该环境并不完全等于 requirements 的固定版本，完整平台及依赖矩阵留待 GitHub Actions 验证。

```bash
python -m pytest tests -m "not hardware and not hmmer and not benchmark"
python -m pytest tests/test_hmmer.py -m hmmer --run-hmmer
```

本地原始证据保存在 Git 忽略的 `test-results/release-readiness-20261116/`，不随仓库发布：

- `full-regression.xml`：295 项 CPU 回归结果。
- `real-hmmer.xml`：真实 HMMER 的 2 项通过和 1 项预期失败。
- `random_inference.py`、`random-proteins.fasta`、`random-inference.log`：可复现的随机输入、生成脚本和运行记录。
- `summary.json`：每组统计、说明校验、实际 HMMER 调用及模型/配置/说明文件的 SHA-256。
- `candidate-checkout-summary.json`：拟发布目录的两次默认 `latest` CLI 结果。
