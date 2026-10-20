"""Phân tích bổ sung GĐ4 từ dự đoán đã lưu (KHÔNG cần chạy lại BERTScore).

1. Kiểm định bootstrap GHÉP CẶP: Lead-k vs TextRank-k, Lead-k vs Random-k, Lead-1 vs Lead-2/3.
   Khoảng tin cậy 95% của hiệu (A - B) không chứa 0 -> khác biệt có ý nghĩa thống kê (theo mẫu test).
2. Vị trí câu của Oracle: Oracle-1 và 'câu đầu tiên Oracle-3 chọn', so với mức phân bố đều.
   Dùng để so với D3 (độ trùng từ) công bằng hơn so với việc đếm tất cả các câu Oracle-3 chọn.

    python -m src.analysis.compare_baselines --data_dir <thư mục clean> --pred_dir results/predictions
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.eda_utils import split_sentences
from src.evaluate import bootstrap_ci, rouge_per_sample

METRICS = ("rouge1", "rouge2", "rougeLsum")


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def default_pairs(ks=(1, 2, 3)) -> list[tuple[str, str]]:
    pairs = [(f"lead-{k}", f"textrank-{k}") for k in ks] + [(f"lead-{k}", f"random-{k}") for k in ks]
    return pairs + [("lead-1", "lead-2"), ("lead-1", "lead-3"), ("lead-2", "lead-3")]


def paired_tests(scores: dict[str, dict], pairs, n_boot: int, seed: int) -> pd.DataFrame:
    rows = []
    for a, b in pairs:
        if a not in scores or b not in scores:
            continue
        for m in METRICS:
            d = scores[a][m] - scores[b][m]
            mean, lo, hi = bootstrap_ci(d, n_boot, seed)
            rows.append({"A": a, "B": b, "metric": m, "mean_A": scores[a][m].mean(), "mean_B": scores[b][m].mean(),
                         "diff": mean, "diff_lo": lo, "diff_hi": hi,
                         "significant": bool(lo > 0 or hi < 0)})
    return pd.DataFrame(rows)


def oracle_first_position(records: list[dict], preds: dict[str, dict]) -> list[dict]:
    n_sents = [len(split_sentences(r["article"])) for r in records]
    chance3 = 100 * np.mean([min(3, n) / n for n in n_sents if n])
    rows = []
    for name, label in (("oracle-1", "Oracle-1 (câu được chọn)"), ("oracle-3", "Oracle-3 (câu ĐẦU TIÊN được chọn)")):
        if name not in preds:
            continue
        first = [min(preds[name][r["id"]]) for r in records if preds[name].get(r["id"])]
        first = np.array(first)
        rows.append({"variant": label, "n_docs": len(first),
                     "pct_index0": 100 * np.mean(first == 0), "pct_first3": 100 * np.mean(first < 3),
                     "pct_first10": 100 * np.mean(first < 10), "median_index": float(np.median(first)),
                     "pct_first3_if_uniform": chance3, "ratio_vs_uniform": 100 * np.mean(first < 3) / chance3})
    return rows


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--pred_dir", default="results/predictions")
    ap.add_argument("--tables_dir", default="results/tables")
    ap.add_argument("--sources", nargs="+", default=["cnn_dm", "govuk"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)
    out = Path(a.tables_dir); out.mkdir(parents=True, exist_ok=True)
    pair_rows, pos_rows = [], []
    for source in a.sources:
        records = read_jsonl(Path(a.data_dir) / f"{source}_{a.split}.jsonl")
        refs = {r["id"]: r["highlights"] for r in records}
        scores, selected = {}, {}
        for f in sorted(Path(a.pred_dir).glob(f"{source}_*.jsonl")):
            name = f.stem[len(source) + 1:]
            preds = {p["id"]: p for p in read_jsonl(f)}
            ids = [r["id"] for r in records if r["id"] in preds]
            selected[name] = {i: preds[i]["selected"] for i in ids}
            if name.split("-")[0] in ("lead", "textrank", "random"):
                scores[name] = rouge_per_sample([preds[i]["prediction"] for i in ids], [refs[i] for i in ids])
        df = paired_tests(scores, default_pairs(), a.n_boot, a.seed); df.insert(0, "source", source)
        pair_rows.append(df)
        for row in oracle_first_position(records, selected):
            pos_rows.append({"source": source, **row})
    pd.concat(pair_rows).to_csv(out / "baselines_paired_tests.csv", index=False)
    pd.DataFrame(pos_rows).to_csv(out / "oracle_first_position.csv", index=False)
    pd.set_option("display.width", 200)
    print(pd.concat(pair_rows).round(4).to_string(index=False))
    print("\n", pd.DataFrame(pos_rows).round(1).to_string(index=False))


if __name__ == "__main__":
    main()
