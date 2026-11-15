"""Đường cong hội tụ của các thí nghiệm BART/T5 (GĐ6): train loss, val loss và S (TB R1/R2/Lsum trên tập con validation) theo epoch.

    python -m src.analysis.plot_hf_curves --logs_dir <Drive>/logs --out_dir results/figures

Mỗi nhóm trong configs/compare.yaml (mục `curves`) cho một hình results/figures/hf_curves_{nhóm}.png. Sao đánh dấu epoch tốt nhất
(best_epoch trong {run}_meta.json). Loss tính theo token của TỪNG tokenizer nên không so trực tiếp BART với T5; để so sánh hãy dùng S.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from matplotlib.ticker import MaxNLocator


def plot_group(runs: list[str], logs_dir: str | Path, out_path: str | Path, title: str) -> Path:
    logs_dir = Path(logs_dir)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    panels = (("train_loss", "train loss (theo token)"), ("val_loss", "val loss (theo token)"), ("val_S", "S = TB(R1, R2, Rlsum) trên tập con validation")) 
    for run in runs:
        ep = pd.read_csv(logs_dir / f"{run}_log.csv")
        meta_path = logs_dir / f"{run}_meta.json"
        best = json.load(open(meta_path, encoding="utf-8")).get("best_epoch") if meta_path.exists() else None
        for ax, (col, label) in zip(axes, panels):
            line, = ax.plot(ep["epoch"], ep[col], marker="o", label=run)
            if best is not None and best in set(ep["epoch"]):
                v = float(ep.loc[ep["epoch"] == best, col].iloc[0])
                ax.plot([best], [v], marker="*", markersize=14, color=line.get_color(), linestyle="none")
            ax.set_xlabel("epoch"); ax.set_ylabel(label); ax.grid(alpha=0.3); ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0].set_title("train loss"); axes[1].set_title("val loss"); axes[2].set_title("S trên validation")
    axes[2].legend(fontsize=8, loc="best")
    fig.suptitle(f"{title} (sao = epoch được chọn; loss không so trực tiếp BART với T5)", fontsize=10)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/compare.yaml")
    ap.add_argument("--logs_dir", default="results/logs")
    ap.add_argument("--out_dir", default="results/figures")
    a = ap.parse_args(argv)
    curves = yaml.safe_load(open(a.config, encoding="utf-8"))["curves"]
    for group, runs in curves.items():
        p = plot_group(runs, a.logs_dir, Path(a.out_dir) / f"hf_curves_{group}.png", f"{group}: {', '.join(runs)}"[:120])
        print("Đã lưu", p)


if __name__ == "__main__":
    main()
