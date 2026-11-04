# 扰动引导的局部覆盖剪枝

方法名为 `q_flowcover_perturbation`，入口为 `test_perturbation.py`。

## 方法流程

1. 在视觉层 `l` 的完整特征上计算问题与视觉组的文本余弦分数，得到全组文本概率 `p`。
2. 按每视图保留预算 `K` 取文本 Top-K 集合 `P`，在 `P` 内重新归一化查询权重 `pi`；全部 K 个查询组均参与统计。
3. 从原生第 `l+1` 层读取 RoPE 后的 Q/K 与 V。每个原始查询分别对同一视图的全部合法源 Key 归一化。
4. 对每个候选组计算固定查询下的删除误差，再对原始查询和 attention head 求均值，并用 `pi` 加权得到 `D`。
5. 将 `P` 内的 `D` 置零作为残差权重。使用组特征的非负余弦建立固定半径局部图。
6. 从空集开始逐轮选择：候选自身的文本概率，加上 `lambda` 加权的新增残差覆盖收益，直到保留恰好 K 个完整组。

删除误差使用 attention readout 在删除该组前后的**差值平方范数**：

```text
e = (b - m * o) / (1 - m)
D = pi 加权的 mean(raw Query, head) ||e||²
```

其中 `o` 是完整 readout，`m` 是候选组的 attention 质量，`b` 是该组的 readout 贡献。
先计算各原始查询的误差，再聚合；不在除法或平方前平均查询或 head。`D` 保持原始数值尺度，不额外归一化后参与选择。
数值不稳定时采用直接向量误差或显式屏蔽源 Key 重新计算。`P` 中的组不强制进入最终保留集。
`q_flowcover_perturbation_nocover` 使用仅含自身的覆盖图；`K=N` 时直接全部保留，扰动统计标记为未计算。

## 执行边界与实现

层号从 1 开始。`full19_gather` 表示在层 `l` 评分，完整执行层 `l+1`，随后 gather 一次并运行紧凑后缀；默认 `l=18` 时在第 19 层之后剪枝。
各视图独立处理，完整组、原始位置、attention 边界、DeepStack 和 LLM 视觉槽位使用同一组索引。

`reference` 为直接分块统计；`optimized` 的正常路径对每个查询分块扫描一次矩形 QK，并用 Value Gram 运算计算同一误差，保留原查询集合、预算和贪心目标。
优化实现的统计使用 FP32；原生模型按所选 dtype 执行。`--greedy-backend reference` 使用原逐轮贪心，其他后端可显式选择。

## 主要参数

| 参数 | 默认值 / 含义 |
|---|---|
| `--decision-layer-human` | `18`，文本评分层 l |
| `--visual-prune-percent` | 删除比例；例如 `60` 表示保留约 40% |
| `--temperature` | `1`，文本概率与查询权重的温度 |
| `--perturbation-lambda` | `1`，原始扰动覆盖收益系数 |
| `--local-radius` | `1`，局部覆盖半径 |
| `--selector-impl` | `reference`；可设 `optimized` |
| `--query-chunk-size` / `--group-chunk-size` / `--source-chunk-size` | `256 / 32 / 256`，统计分块 |
| `--max-attention-mb` | `512`，优化实现中单个矩形 attention 分配上限 |
| `--greedy-backend` | `reference`，逐轮贪心后端 |

## 测速与结果

正式测速使用 `--no-save-diagnostics --verify-candidates 0`，关闭额外 flow 对照、绘图和显式删除检查。
逐题时间覆盖视觉编码、选点、gather、LLM 生成和答案解码；processor 与图像读取在该时间之外。绘图耗时单独记录。
部分阶段相互包含：扰动统计包含 flow support，QKV 获取包含原生层内工作，不能将这些阶段重复相加。

TFLOPs 为架构估算，计入实际 raw-patch 视觉层、ViT QK/AV、final / DeepStack / 评分 merger、选点器及文本模型。
比较 base 时使用 `test_base.py` 的同一口径；旧的 merged-group 视觉估计不能直接与当前结果比较。
每次运行保存 `config.json`、逐题 `results.jsonl`、`summary.json` 和 `results.md`。

实现入口：[模型适配](../compression/qwen3_vl_perturbation.py)、[参考选择器](../compression/q_flowcover_perturbation.py)、[优化选择器](../compression/q_flowcover_perturbation_optimized.py)、[优化统计](../compression/perturbation_statistics_optimized.py)、[贪心后端](../compression/perturbation_greedy_optimized.py)。
运行与评测由 [worker](../compression/perturbation_worker.py) 和 [入口脚本](../test_perturbation.py) 负责；批量命令见 [README](../README.md)。
