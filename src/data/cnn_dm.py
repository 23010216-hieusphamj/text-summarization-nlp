"""Tải CNN/DailyMail từ HuggingFace và tạo tập con có seed cố định.

Chạy:
    python -m src.data.cnn_dm --config configs/data_cnn_dm.yaml

Đầu ra:
    data/processed/cnn_dm_{train,validation,test}.jsonl   (không commit)
    data/samples/cnn_dm_sample.jsonl                      (commit, minh họa)
    results/tables/cnn_dm_split_sizes.csv                 (commit)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

SPLITS = ("train", "validation", "test")


def subsample(dataset, n: int, seed: int):
    """Lấy ngẫu nhiên n mẫu (tái lập được nhờ seed). n >= len thì lấy hết."""
    n = min(n, len(dataset))
    return dataset.shuffle(seed=seed).select(range(n))


def build_subsets(raw: dict, sizes: dict, seed: int) -> dict:
    """raw: dict split -> datasets.Dataset. Trả về dict split -> Dataset con."""
    return {s: subsample(raw[s], sizes[s], seed) for s in SPLITS}


def write_jsonl(dataset, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in dataset:
            rec = {"id": row["id"], "article": row["article"], "highlights": row["highlights"]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main(config_path: str) -> None:
    from datasets import load_dataset  # import muộn để test không cần mạng

    cfg = yaml.safe_load(open(config_path, encoding="utf-8"))
    raw = load_dataset(cfg["dataset_name"], cfg["dataset_config"])
    subsets = build_subsets(raw, cfg["sizes"], cfg["seed"])

    out = Path(cfg["output_dir"])
    rows = []
    for s in SPLITS:
        write_jsonl(subsets[s], out / f"cnn_dm_{s}.jsonl")
        rows.append({"split": s, "original_size": len(raw[s]), "subset_size": len(subsets[s])})
        print(f"{s:<11} gốc={len(raw[s]):>7,}  lấy={len(subsets[s]):>6,}")

    sample = subsets["train"].select(range(min(cfg["sample_n"], len(subsets["train"]))))
    write_jsonl(sample, Path(cfg["sample_dir"]) / "cnn_dm_sample.jsonl")

    tables = Path(cfg["tables_dir"])
    tables.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(tables / "cnn_dm_split_sizes.csv", index=False)
    print("Xong. Đã lưu vào", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data_cnn_dm.yaml")
    main(ap.parse_args().config)
