"""
scripts/run_mlx_train_overnight.py
Overnight Scaled Training Script for Apple MLX on M-series Mac:
- 15,000 stratified samples (35% peak, 25% valley, 40% baseline)
- 16 layers adapted (out of 24)
- Mask prompt loss enabled (only gradients on the output JSON)
- Gradient checkpointing enabled for low RAM overhead (< 3 GB)
- 10,000 iterations at batch size 2 (~6 to 8 hours of training)
- Saves checkpoints every 1,000 steps
"""

import os
import sys
import subprocess

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
model_dir    = os.path.join(ROOT_DIR, "models/Qwen2.5-0.5B-Instruct")
data_dir     = os.path.join(ROOT_DIR, "data")
adapters_dir = os.path.join(ROOT_DIR, "adapters")
os.makedirs(adapters_dir, exist_ok=True)

cmd = [
    "mlx_lm.lora",
    "--model", model_dir,
    "--train",
    "--data", data_dir,
    "--adapter-path", adapters_dir,
    "--batch-size", "2",
    "--num-layers", "16",
    "--mask-prompt",
    "--grad-checkpoint",
    "--iters", "10000",
    "--learning-rate", "1e-4",
    "--steps-per-report", "50",
    "--steps-per-eval", "500",
    "--save-every", "1000",
]

print("=" * 80)
print("  🚀 LAUNCHING SCALED OVERNIGHT MLX FINE-TUNING ON APPLE SILICON")
print(f"  Root Dir:         {ROOT_DIR}")
print(f"  Model:            {model_dir}")
print(f"  Dataset:          15,000 stratified samples in {data_dir}/train.jsonl")
print(f"  LoRA Capacity:    16 layers (out of 24)")
print(f"  Iterations:       10,000 steps (Batch Size 2)")
print(f"  Checkpoints:      Saved every 1,000 steps to {adapters_dir}/")
print("=" * 80, flush=True)

subprocess.run(cmd, cwd=ROOT_DIR)
