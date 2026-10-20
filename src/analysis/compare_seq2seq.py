"""So sánh Seq2Seq với baseline bằng bootstrap GHÉP CẶP (cùng phương pháp với compare_baselines).

Khoảng tin cậy 95% của hiệu (A - B) không chứa 0 -> khác biệt có ý nghĩa theo mẫu test. Lead-k được tính lại
trực tiếp từ bài (xác định, giống baseline) nên không cần file dự đoán; TextRank chỉ so khi có file.

    python -m src.analysis.compare_seq2seq --data_dir <clean> --pred_dir results/predictions
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.analysis.compare_baselines import paired_tests, read_jsonl
from src.baselines.lead import join_sentences, lead_k
from src.evaluate import rouge_per_sample

BASELINES = {"cnn_dm": ["lead-3", "textrank-3"], "govuk": ["lead-1", "textrank-1", "lead-3"]}
ABLATIONS = ["seq2seq-beam4", "seq2seq-greedy"]


def load_preds(pred_dir: Path, source: str, name: str, records: list[dict]) -> dict | None:
    f = pred_dir / f"{source}_{name}.jsonl"
    if f.exists():
        return {p["id"]: p["prediction"] for p in read_jsonl(f)}
    if name.startswith("lead-"):
        k = int(name.split("-")[1])
        return {r["id"]: join_sentences(r["article"], lead_k(r["article"], k)) for r in records}
    return None


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--pred_dir", default="results/predictions")
    ap.add_argument("--tables_dir", default="results/tables")
    ap.add_argument("--main", default="seq2seq-beam4_nr3", help="hệ thống chính của D9")
    ap.add_argument("--sources", nargs="+", default=["cnn_dm", "govuk"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)
    pred_dir, out = Path(a.pred_dir), Path(a.tables_dir)
    out.mkdir(parents=True, exist_ok=True)
    frames = []
    for source in a.sources:
        records = read_jsonl(Path(a.data_dir) / f"{source}_{a.split}.jsonl")
        refs = {r["id"]: r["highlights"] for r in records}
        preds = {a.main: load_preds(pred_dir, source, a.main, records)}
        if preds[a.main] is None:
            print(f"[{source}] chưa có dự đoán của {a.main}, bỏ qua"); continue
        others = BASELINES.get(source, []) + ABLATIONS
        for name in others:
            p = load_preds(pred_dir, source, name, records)
            if p is not None:
                preds[name] = p
        ids = [r["id"] for r in records if all(r["id"] in d for d in preds.values())]
        scores = {n: rouge_per_sample([d[i] for i in ids], [refs[i] for i in ids]) for n, d in preds.items()}
        pairs = [(a.main, n) for n in preds if n != a.main]
        df = paired_tests(scores, pairs, a.n_boot, a.seed); df.insert(0, "source", source); df.insert(1, "n", len(ids))
        frames.append(df)
    if not frames:
        print("Không có gì để so sánh."); return
    res = pd.concat(frames, ignore_index=True)
    res.to_csv(out / "seq2seq_paired_tests.csv", index=False)
    pd.set_option("display.width", 220)
    print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
