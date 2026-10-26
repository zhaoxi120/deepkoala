# FASTA ID 与 HMMER 临时文件修复

日期：2026-10-26（Asia/Tokyo）。修复此前轻量回归中的前三个预期失败实例：
FASTA ID 不一致一项，以及 HMMER 成功/失败后的临时 FASTA 清理两项。
这三项已移除 `xfail` 标记，保留为必须通过的回归测试。

后续按要求删除了全部 wheel 测试及其专用测试依赖；下文保留删除前的修复验证结果。
当前测试范围以 [测试说明](testing.md) 为准，删除测试不代表新增 wheel 独立安装支持。
删除后再次运行相同的轻量 CI 选择：**237 passed、4 deselected、0 xfailed**，
20.08 秒。该次 JUnit 保留在本地 `test-results/github-sync/ci-regression.xml`。

## 行为变化

`ProteinDataset`、固定批次流式读取、自适应排序窗口和多域模式共用同一个 ID 解析函数。
ID 为 `>` 后第一个以空白分隔的字段，空格和 Tab 都能分隔描述。例如
`>sequence<TAB>description` 统一输出 `sequence`；空白标题明确报错。
旧标准模式可能把 Tab 后描述也放进 ID，此类输入关联历史结果时需要使用新的 ID 规则。
序列字符、模型计算及重复 ID 的记录顺序没有改变。

HMMER 临时 FASTA 在子进程运行前写完并关闭，并在外层 `finally` 中清理。
正常命中、无命中、缺少可执行程序、退出码 1、其他进程异常和坐标解析异常均覆盖。
原有进程错误传播及 HMMER 命令参数保持不变。

## 验证

运行与 GitHub CI 相同的轻量选择范围：

```console
python -m pytest tests -m "not hardware and not hmmer and not benchmark" --junitxml=test-results/known-issue-fixes/ci-regression.xml
```

本地 Linux、Python 3.12.8、PyTorch 2.5.1 CPU：
**238 passed、4 deselected、2 xfailed**，45.72 秒，无失败。
其中 ID/空标题的 7 个场景与 HMMER 清理的 6 个场景全部通过，
并覆盖带 Tab 描述的重复 ID 在排序、OOM 拆批及 CSV 写出后的顺序。

HMMER 单元测试使用真实临时文件及受控子进程响应，能验证文件内容和清理路径；
本轮没有执行真实 HMMER 程序，也没有运行 GPU 或大规模压测。
原始 JUnit 与源码哈希只保留在本地 `test-results/known-issue-fixes/`。

删除 wheel 测试前，默认轻量回归剩余两项预期失败均为原有打包问题：wheel 未声明运行依赖，
以及安装后的模型资源定位。另一个 HMMER 路径含空格的问题属于本轮排除的真实工具测试，
未纳入这三项修复，也没有被标记为已解决。
