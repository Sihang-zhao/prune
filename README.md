# Qwen3 Perturbation Pruning

无需训练的 Qwen3-VL 视觉 token 剪枝。保留文本相关性、组删除扰动和局部覆盖逻辑；默认使用优化统计实现。

```text
evaluate.py       统一评测入口：扰动剪枝 / 原生 base
pruning/          选择器、扰动统计与贪心覆盖
models/           Qwen3-VL 接入、完整组裁剪和生成
evaluation/       数据、推理进程、评分、续跑和 FLOPs
scripts/          比例测试与结果总表
tests/            数学、模型接入和评测测试
```

## 安装

Python 3.10；按设备安装 PyTorch，再安装依赖：

```bash
pip install -r requirements.txt
```

## 运行

```bash
export DATA=/path/to/XLRS-Bench-lite
export MODEL=/path/to/Qwen3-VL-8B

python evaluate.py --data "$DATA" --model "$MODEL" --offline --gpus 0 \
  --method perturbation --layer 18 --prune-percent 70 \
  --output-dir outputs/perturbation/L18/prune_70

python evaluate.py --data "$DATA" --model "$MODEL" --offline --gpus 0 \
  --method base --output-dir outputs/base
```

批量测试 50、60、70、80、85、90、95、98%，支持续跑：

```bash
bash scripts/run_sweep.sh "$DATA" "$MODEL" 0 outputs/perturbation 18
```

默认 bf16、4M 像素、单视图、最多 32 个输出 token。`--layer L` 在第 L 层评分，读取原生第 L+1 层 Q/K/V，完成该层后裁剪；后续视觉层使用保留组。`--prune-percent` 是删除比例。`--method nocover` 为无局部覆盖对照，`--implementation reference` 使用参考统计。

## 方法与结果

文本 Top-K 作为查询集合，全部查询对完整源 Key 归一化。计算各原始查询/head 的组删除 readout 差值平方范数后，再按问题权重聚合。查询集合内的扰动权重置零，用非负局部余弦图逐轮选择文本概率与新增残差覆盖收益最大的组，保留恰好 K 个完整组。

每次运行保存 `config.json`、`results.jsonl`、`summary.json`、`results.md`、`test.log`。原命令加 `--resume` 可继续缺失或失败题目。`--diagnostics full` 保存完整扰动图数组。

Avg. 是 13 任务宏平均；time 覆盖视觉编码至答案解码，预处理和预热除外。TFLOPs 是包含 raw-patch ViT attention、原生 merger、选点器及文本模型的部分解析估计，base 与剪枝使用相同口径。

```bash
python scripts/summarize.py --input-root outputs --output-prefix outputs/summary
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s tests -v
```
