"""Tiền xử lý thống nhất cho CNN/DM và GOV.UK -> data/processed/clean/{source}_{split}.jsonl

    python -m src.preprocessing.build_clean --config configs/preprocess.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd
import yaml

from src.data.eda_utils import summary_sentences, word_count
from src.preprocessing.text_clean import (clean_article, clean_generic_article, clean_highlights)


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def clean_records(records: list[dict], source: str) -> tuple[list[dict], int]:
    """Trả về (bản ghi đã làm sạch, số bài có nội dung bị thay đổi do làm sạch)."""
    art_fn = clean_article if source == "cnn_dm" else clean_generic_article
    out, changed = [], 0
    for r in records:
        art = art_fn(r["article"])
        base = re.sub(r"\s+", " ", r["article"]).strip()
        changed += art != base
        rec = dict(r)
        rec.update(article=art, highlights=clean_highlights(r["highlights"]), source=source)
        out.append(rec)
    return out, changed


def filter_train(records: list[dict], f: dict) -> list[dict]:
    keep = []
    for r in records:
        sw = word_count(r["highlights"])
        if (word_count(r["article"]) >= f["min_article_words"]
                and f["min_summary_words"] <= sw <= f["max_summary_words"]
                and len(summary_sentences(r["highlights"])) <= f["max_summary_sentences"]):
            keep.append(r)
    return keep


def _fingerprint(text: str, n: int = 300) -> str:
    return hashlib.md5(re.sub(r"\W+", " ", text.lower()).strip()[:n].encode()).hexdigest()


def leakage(train: list[dict], other: list[dict]) -> int:
    """Số bài của tập khác có 300 ký tự đầu (đã chuẩn hóa) trùng với một bài trong train."""
    seen = {_fingerprint(r["article"]) for r in train}
    return sum(_fingerprint(r["article"]) in seen for r in other)


def main(config_path: str) -> pd.DataFrame:
    cfg = yaml.safe_load(open(config_path, encoding="utf-8"))
    src_dir, out_dir = Path(cfg["input_dir"]), Path(cfg["output_dir"])
    rows, leaks = [], []
    for source in cfg["sources"]:
        cleaned = {}
        for split in cfg["splits"]:
            recs = read_jsonl(src_dir / f"{source}_{split}.jsonl")
            new, changed = clean_records(recs, source)
            n_clean = len(new)
            if source == "cnn_dm" and split == "train":
                new = filter_train(new, cfg["cnn_dm_train_filter"])
            cleaned[split] = new
            write_jsonl(new, out_dir / f"{source}_{split}.jsonl")
            rows.append({"source": source, "split": split, "n_input": len(recs), "n_changed_by_cleaning": changed,
                         "n_removed_by_filter": n_clean - len(new), "n_output": len(new)})
        for s in ("validation", "test"):
            leaks.append({"source": source, "split": s, "articles_also_in_train": leakage(cleaned["train"], cleaned[s])})
    tables = Path(cfg["tables_dir"]); tables.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows); df.to_csv(tables / "preprocess_summary.csv", index=False)
    pd.DataFrame(leaks).to_csv(tables / "leakage_check.csv", index=False)
    print(df.to_string(index=False)); print(pd.DataFrame(leaks).to_string(index=False))
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/preprocess.yaml")
    main(ap.parse_args().config)
