#!/usr/bin/env python3
"""Offline CLI smoke with native tiny random weights + real processor/data.

This is an integration test, not an 8B accuracy or GPU performance benchmark.
The original checkpoint and dataset are read-only. No result files are copied.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processor-source", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", default="outputs/perturbation_validation/cli_smoke")
    args = parser.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    from transformers import AutoProcessor, Qwen3VLConfig, Qwen3VLForConditionalGeneration
    from utils.data import load_cases, load_image

    torch.set_num_threads(1)
    torch.manual_seed(1702)
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / "tiny_random_checkpoint"
    checkpoint.mkdir(exist_ok=True)
    original = json.loads((Path(args.processor_source) / "config.json").read_text())
    original["text_config"].update(
        hidden_size=32, intermediate_size=64, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        max_position_embeddings=4096,
        rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 2], "mrope_interleaved": True},
    )
    original["vision_config"].update(
        depth=20, hidden_size=16, intermediate_size=32, out_hidden_size=32,
        num_heads=2, num_position_embeddings=64, deepstack_visual_indexes=[5, 17, 18],
    )
    original["dtype"] = "float32"
    original["text_config"]["dtype"] = "float32"
    original["vision_config"]["dtype"] = "float32"
    model = Qwen3VLForConditionalGeneration(Qwen3VLConfig.from_dict(original)).float().eval()
    model.save_pretrained(checkpoint)
    processor = AutoProcessor.from_pretrained(args.processor_source, local_files_only=True)
    processor.save_pretrained(checkpoint)
    del model, processor
    case = load_cases(args.data, limit=1, split="train")[0]
    images = load_image(case, args.data, max_patches=1, max_source_pixels=65536)
    image = out / "real_sample.png"
    images[0].save(image)
    row = {**dataclasses.asdict(case), "image": str(image)}
    annotations = out / "real_sample.jsonl"
    annotations.write_text(json.dumps(row, ensure_ascii=False) + "\n")
    records = []
    for method in ("text_cos", "q_flowcover_direct_residual", "q_flowcover_perturbation", "q_flowcover_perturbation_nocover"):
        command = [
            sys.executable, str(ROOT / "test_perturbation.py"), "--data", str(annotations),
            "--model", str(checkpoint), "--offline", "--dtype", "fp32",
            "--prune-method", method, "--decision-layer-human", "18", "--visual-prune-percent", "70",
            "--max-image-patches", "1", "--max-image-pixels", "65536", "--max-new-tokens", "2",
            "--warmup-cases", "1", "--verify-candidates", "2", "--limit", "1",
            "--query-chunk-size", "4", "--group-chunk-size", "2", "--source-chunk-size", "16",
            "--output-dir", str(out / method),
        ]
        started = time.perf_counter()
        result = subprocess.run(command, cwd=ROOT, env=os.environ.copy(), text=True, capture_output=True)
        (out / f"{method}.log").write_text(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError(f"{method} CLI failed; inspect {out / (method + '.log')}")
        run = max((out / method).glob("*/results.jsonl"), key=lambda p: p.stat().st_mtime)
        item = json.loads(run.read_text().splitlines()[0])
        if item.get("status") == "error":
            raise RuntimeError(f"{method} case failed: {item.get('error')}")
        records.append({
            "method": method, "command": command, "result_path": str(run),
            "subprocess_wall_seconds": time.perf_counter() - started,
            "measured_inference_seconds": item["testtime"],
            "retained_tokens": item["retained_tokens"], "original_image_tokens": item["original_image_tokens"],
            "vision_layer_token_counts": item["vision_layer_token_counts"],
            "answer": item["raw_model_answer"], "status": item["status"],
            "algorithm_id": item.get("algorithm_id"), "qkv_capture_count": item.get("qkv_capture_count"),
            "qkv_projection_replay_count": item.get("qkv_projection_replay_count"),
            "stage_timings": item.get("stage_timings"), "diagnostics": item.get("diagnostics"),
            "native_verification": item.get("native_verification"),
        })
        print(f"passed: {method}, {item['original_image_tokens']} -> {item['retained_tokens']}, {item['testtime']:.4f}s", flush=True)
    report = {
        "purpose": "one real XLRS input + actual local tokenizer/image processor + tiny random native Qwen3-VL weights; 4-method CLI integration smoke",
        "real_8b_weights_loaded": False, "cuda_used": False, "quality_or_gpu_speed_benchmark": False,
        "processor_source": args.processor_source, "dataset_source": args.data,
        "original_sample_id": case.id, "max_image_pixels_for_smoke_only": 65536,
        "warmup_per_method": 1, "measured_cases_per_method": 1, "records": records,
    }
    path = out / "report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"saved: {path}")


if __name__ == "__main__":
    main()
