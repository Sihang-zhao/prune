# Perturbation-guided Q-FlowCover

面向 Qwen3-VL 的视觉 token 剪枝：结合问题相关性、组删除扰动与局部覆盖，保留固定数量的视觉组。支持参考实现和等价的优化实现。

## 安装

Python 3.10 或更新版本；根据设备安装 PyTorch，然后安装依赖：

```bash
pip install -r requirements.txt
pip install -r requirements-data.txt
```

## 扰动方法

```bash
export DATA=/path/to/XLRS-Bench-lite
export MODEL=/path/to/Qwen3-VL-8B

python test_perturbation.py \
  --data "$DATA" --model "$MODEL" --offline --gpus 0 --dtype bf16 \
  --prune-method q_flowcover_perturbation --selector-impl optimized \
  --greedy-backend reference --exec-mode full19_gather \
  --decision-layer-human 18 --visual-prune-percent 70 \
  --max-image-pixels 4194304 --max-image-patches 1 --max-new-tokens 32 \
  --verify-candidates 0 --no-save-diagnostics \
  --output-dir outputs/perturbation/prune_70
```

`--visual-prune-percent 70` 表示删除 70%。在第 L 层完成文本相关性评分，从原生第 L+1 层取得 Q/K/V，完成该层后剪枝，后续视觉层使用保留的 token。

批量测试 50、60、70、80、85、90%：

```bash
bash scripts/run_perturbation_optimized_50_90.sh
```

方法定义与参数见 [扰动方法](docs/perturbation.md)。

## 原生 base 对照

使用相同图像与生成设置，直接输入原生 Qwen3-VL，不执行剪枝：

```bash
python test_base.py \
  --data "$DATA" --model "$MODEL" --offline --gpus 0 --dtype bf16 \
  --max-image-pixels 4194304 --max-image-patches 1 --max-new-tokens 32 \
  --output-dir outputs/qwen3_base_current
```

## 结果

每次运行生成 `config.json`、`results.jsonl`、`summary.json`、`results.md` 和 `test.log`。中断后在原命令增加 `--resume`，成功题目会被复用。

Avg. 为 13 个任务的宏平均准确率。time 为视觉编码至答案解码的平均每题时间，预处理与预热不计入。TFLOPs 按实际 raw-patch 视觉长度、ViT attention、原生 merger 和文本模型估算；base 与扰动方法使用同一口径。它是部分解析估计，不能与旧版仅按合并组计数的结果直接比较。

## 测试

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s tests -v
```
