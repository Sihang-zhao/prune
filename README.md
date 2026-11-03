# prune — Q-FlowCover Direct Residual

本仓库整理了原实验目录 `prune/outputs/q_flowcover_direct_residual` 对应的评测代码。历史入口是 `prune/testn.py`，方法名为 **`q_flowcover_direct_residual`**；算法实现在 `compression/q_flowcover_variants.py`。

这与后来的 `testww.py / pruning_direct_residual` 中名为 `direct_residual` 的实现不同。这里保留目标实验使用的算法文件及其 attention、Top-K、局部覆盖数值语义，默认参数对应历史的 **260 题子集**，不把它标为全量测试结果。

## 安装

已验证 Python 3.10、PyTorch 2.6.0、Transformers 4.57.6。按设备安装匹配的 PyTorch / torchvision 构建，再安装：

```bash
python -m pip install -r requirements.txt
# 本地 HuggingFace 数据集及 Parquet / Feather 输入：
python -m pip install -r requirements-data.txt
```

权重和数据自行准备；本仓库不包含它们。`--model` 接受本地 Qwen3-VL checkpoint 目录或模型 ID。`--data` 必填，也可通过 `XLRS_DATA_ROOT` 设置；支持本地 HuggingFace 数据集目录、JSON、JSONL、Parquet 和 Feather。

## 复现单档实验

```bash
python testn.py \
  --data /path/to/XLRS-Bench-lite-subdata-20 \
  --model /path/to/Qwen3-VL-8B \
  --offline \
  --gpus 0 \
  --dtype bf16 \
  --prune-method q_flowcover_direct_residual \
  --exec-mode hard18 \
  --visual-prune-percent 70 \
  --local-radius 1 \
  --affinity-mode raw \
  --qflow-query-chunk 256 \
  --max-image-pixels 4194304 \
  --max-image-patches 1 \
  --max-new-tokens 32 \
  --worker-cases 128 \
  --text-source baseline \
  --output-dir outputs/q_flowcover_direct_residual/prune_70
```

`--visual-prune-percent 70` 删除 70%、保留约 30% 的合并 token。也可以传 `--prune-fraction 0.7`，两者不能同时使用。固定每视图 K 使用 `--visual-keep-tokens K`，不能与删除比例参数同时使用。预算计算沿用 `max(1, round(N × keep_ratio))`。

第一次检查可加 `--limit 2`。多卡使用 `--gpus 0,1`，各卡独立加载模型、处理不同样本；不进行模型切分。`--dtype auto` 在 CUDA 上自动选择 bf16 或 fp16；历史各档实际使用 bf16。无 CUDA 时可用 CPU 和 `--dtype fp32`，此时不传 `--gpus`。

## 运行八档剪枝比例（含 98%）

```bash
bash scripts/run_pruning_sweep.sh \
  /path/to/XLRS-Bench-lite-subdata-20 \
  /path/to/Qwen3-VL-8B \
  0
```

脚本依次运行 **50、60、70、80、85、90、95、98%** 剪枝，默认在第 18 层结束后剪枝。结果按比例分目录保存。可通过 `QFLOW_PYTHON=/path/to/environment/bin/python` 选择解释器；第 4 个位置参数指定输出根目录，第 5 个位置参数指定剪枝层（15、16、17、18，默认 18）。98% 是新增实验配置，不属于下面列出的七档历史结果。

## 比较第 15–18 层的剪枝位置

```bash
bash scripts/run_layer_pruning_sweep.sh \
  /path/to/XLRS-Bench-lite-subdata-20 \
  /path/to/Qwen3-VL-8B \
  0
```

这会独立运行 **4 层 × 8 比例 = 32 组** 实验，输出到 `outputs/q_flowcover_direct_residual_layers/layer_15/prune_50/` 等目录。层号从 1 开始，每组只在所选层结束后剪枝一次；例如第 15 层剪枝时，第 16 个视觉块开始使用紧凑 token。该网格已覆盖上一节的第 18 层八档实验，可以按需求选用一种脚本。

单独指定位置可在 `testn.py` 命令中增加 `--decision-layer-human 15`（也支持下划线别名 `--decision_layer_human`）。默认 18 保持历史实验行为。

如果使用完整 `XLRS-Bench-lite`，仅更换 `--data` 或脚本第一个参数即可，但应作为新的完整数据集实验报告。

## 方法与实际参数

默认图像经完整视觉块 1–18，选择一次合并组后继续运行紧凑视觉块 19–27。选择层 L 也可设为 15、16、17，此时完整前缀为 1–L、紧凑后缀为 L+1–27；每组包含完整的 2×2 原始 patch，原生位置、attention 边界和 DeepStack 使用同组索引对齐。问题评分使用原 LLM 输入词嵌入和冻结 merger 的所选层投影；生成提示保留题目、选项及回答要求。

`q_flowcover_direct_residual` 的步骤为：

1. 对全部组的文本余弦分数做 softmax，得到直接文本质量 `text_mass`。
2. 文本 Top-K 组作为查询 P，在 P 内重新归一化后，利用原生 RoPE Q/K 分块计算对全部组的 attention flow support。
3. 将 P 内的 support 置零，得到残差支持 `residual_support`。
4. 在局部相似图上贪心选择：候选自身的直接文本质量，加上它对残差支持产生的新增覆盖；最终保留恰好 K 个有序组。

这里不是把文本质量与 support 各乘 0.5 后再覆盖，也不使用后续 `direct_residual` 实现的 temperature、uniform-support-correction 或强制自相似等改动。

| 参数 | 历史值与作用 |
|---|---|
| `--local-radius` | `1`，3×3 邻域，直接影响局部覆盖 |
| `--affinity-mode` | `raw`，截断为非负的原始余弦；`local_centered` 是显式额外配置 |
| `--qflow-query-chunk` | `256`，原生 attention 查询分块 |
| `--text-source` | `baseline`，评分文本为问题本身 |
| `--query-mass` | 保存为 `0.9`，此方法仍使用文本 Top-K 查询，不按累计质量截断 |
| `--adaptive-radius-max` | 保存为 `3`，此方法仍使用固定 local radius |
| `--qflow-radius` / `--qflow-candidate-chunk` | 保留的历史字段，不参与此 direct-residual 分支 |

公开入口还支持 `--prune-method text_cos` 基线。执行模式名保留历史兼容：默认 `hard18` 在所选层 L 后立即压缩；`full19_gather` 在第 L+1 块完整计算后 gather；`att_bridge19_fullkv` 在第 L+1 块让选中查询读取完整 K/V，随后压缩。当 L=18 时分别对应历史的第 19 块。历史七档均为 L=18 的 `hard18`；其他层号和模式属于额外对照。

## 历史结果

来源：原目录的 `config.json` 和 `summary.json`。所有档位均为 `XLRS-Bench-lite-subdata-20` 的 train split，13 个任务各 20 题，共 260 题，错误数均为 0。

| 删除比例 | 正确题数 / 260 | Micro AC | 平均秒 / 题 | 平均估算 TFLOPs |
|---:|---:|---:|---:|---:|
| 50% | 124 | 47.69% | 0.70350 | 28.93546 |
| 60% | 124 | 47.69% | 0.59684 | 23.75720 |
| 70% | 130 | 50.00% | 0.50985 | 18.68180 |
| 80% | 130 | 50.00% | 0.41256 | 13.68614 |
| 85% | 127 | 48.85% | 0.36089 | 11.22198 |
| 90% | 132 | 50.77% | 0.31371 | 8.79228 |
| 95% | 125 | 48.08% | 0.27884 | 6.37338 |

这些是已有运行记录，并非本次重新执行 8B 全数据集得到的新测量。TFLOPs 是架构估算，时间取决于运行设备与配置。机器可读索引：[CSV](results/reference_metrics.csv)、[配置与来源](results/reference_runs.json)；原样保留的小型 summary 在 `results/reference_summaries/`。

## 输出与续跑

每次运行在 `--output-dir` 下创建时间戳目录，包含 `config.json`、逐题 `results.jsonl`、`summary.json`、`results.md` 和 `test.log`。direct-residual 诊断包含全局文本质量、查询组、attention support、残差支持和最终组索引；其中历史字段 `omega` 表示残差支持，不能解释为普通 Q-FlowCover 的混合权重。

中断后在同一命令增加 `--resume`，或使用 `--resume-from /path/to/run/results.jsonl`；必须保留对应 `config.json`，并匹配数据、模型、方法、预算、局部半径、affinity 和生成配置，防止混入其他实验结果。

## 跨层、跨比例的结果总表

两个批量脚本会在每组实验结束后自动刷新输出根目录的 `summary.csv`、`summary.md`、`summary.html` 和 `summary.json`。单层脚本汇总该层八档；跨层脚本在同一张表中汇总 15–18 层的全部 32 组，而不是只写每层各自的结果。

总表以 `Method / Layer / Prune%` 标识实验，表头分组为：

- **Perception**：OC、RC、OLUC、RLUC、OCC、OCL、OMS、OSR。
- **Reasoning**：AD、ECR、RP、RCCD、CCR。
- **Avg.**：13 个 L3 任务的宏平均准确率，百分比；不使用样本加权 micro accuracy。
- **time**：平均每题秒数 `avg_testtime`；**TFLOPs**：平均每题估算计算量 `avg_tflops`。

没有完成的层号/比例填 `—` 并标记 pending，缺少任务的格子也不填成 0。汇总只读取同一运行目录的 `config.json + summary.json`；同一组合有多次运行时选最新已完成记录，正在运行的新目录不会遮住已完成结果。不同数据集、模型、实际 dtype、视图预算、生成参数等不兼容配置会要求过滤或使用更窄的输入目录，避免混表。

也可以随时手动重新汇总：

```bash
python scripts/summarize_results.py \
  --input-root outputs/layer_sweep \
  --output-prefix outputs/layer_sweep/summary
```

`--input-root` 可接多个结果根目录；`--layers`、`--prune-percents` 可调整表格网格，`--data`、`--model`、`--exec-mode`、`--dtype` 用于筛选。同目录的 JSON 保存来源运行路径，CSV 便于 Excel 分析，HTML / Markdown 使用 Perception 与 Reasoning 两级表头。

## 测试与来源

```bash
python -S testn.py --help
python -m unittest discover -s tests -v
```

测试以 CPU 张量和本地初始化的 tiny Qwen3-VL 为主，不下载模型；覆盖独立 dense attention / direct-residual greedy 参考、半精度 support、固定预算、多视图、三种视觉执行路径、RoPE / DeepStack、CLI 和续跑契约。

当前实测 **60 项全部通过、无跳过**，覆盖四个剪枝层、三种执行模式、98% 极高剪枝下的最小组预算，以及汇总指标契约和 8 / 32 组脚本自动汇总。另完成真实离线 CLI → spawn worker → 随机 tiny Qwen3-VL → 生成及结果写入，四个层号的 98% 剪枝均无运行错误；此前也验证了续跑 `reused=1, pending=0`。这些检查验证代码与接口，不替代 8B 模型的基准复测。

`testn.py → evaluator.py → compression/worker.py → Qwen3VLCompressionRunner → q_flowcover_variants.py` 是新的独立调用链，不依赖旁边的原工程目录。共用模型封装保留了部分原有基础 helper；核心 `q_flowcover.py` 和 `q_flowcover_variants.py` 与原文件逐字节一致。

详细追溯见 [SOURCE_MAP.md](SOURCE_MAP.md) 和 [source_manifest.json](source_manifest.json)。原结果未保存历史源码 commit，因此这里记录的是“实验配置与当前对应实现”的证据链。已去掉本机运行路径、无关实验入口及权重/数据/大日志；原结果目录保持不变。
