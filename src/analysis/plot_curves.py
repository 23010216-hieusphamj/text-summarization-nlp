"""Vẽ đường cong hội tụ từ log huấn luyện -> results/figures/{run}_loss.png

    python -m src.analysis.plot_curves --run_name seq2seq_in512
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_run(run_name: str, log_dir: str = "results/logs", out_dir: str = "results/figures") -> Path:
    log_dir, out_dir = Path(log_dir), Path(out_dir)
    ep = pd.read_csv(log_dir / f"{run_name}_log.csv")
    steps_path = log_dir / f"{run_name}_steps.csv"
    has_steps = steps_path.exists() and steps_path.stat().st_size > 0
    fig, axes = plt.subplots(1, 2 if has_steps else 1, figsize=(11 if has_steps else 6, 4), squeeze=False)
    ax = axes[0][0]
    ax.plot(ep["epoch"], ep["train_loss"], marker="o", label="train loss")
    ax.plot(ep["epoch"], ep["val_loss"], marker="o", label="val loss")
    best = ep.loc[ep["val_loss"].idxmin()]
    ax.axvline(best["epoch"], color="gray", linestyle="--", linewidth=1, label=f"tốt nhất: epoch {int(best['epoch'])}")
    ax.set_xlabel("epoch"); ax.set_ylabel("cross-entropy (theo token)"); ax.set_title(f"{run_name}: loss theo epoch")
    ax.legend(); ax.grid(alpha=0.3)
    if has_steps:
        st = pd.read_csv(steps_path)
        ax2 = axes[0][1]
        ax2.plot(st["step"], st["loss"], alpha=0.3, label="loss (mỗi lần ghi)")
        ax2.plot(st["step"], st["loss"].rolling(10, min_periods=1).mean(), label="trung bình trượt 10")
        ax2.set_xlabel("bước huấn luyện"); ax2.set_ylabel("train loss"); ax2.set_title("loss theo bước")
        ax2.legend(); ax2.grid(alpha=0.3)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{run_name}_loss.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--log_dir", default="results/logs")
    ap.add_argument("--out_dir", default="results/figures")
    a = ap.parse_args()
    print("Đã lưu", plot_run(a.run_name, a.log_dir, a.out_dir))
