"""Chọn k của Lead-k trên tập VALIDATION (D10, nhận xét 5 của giảng viên), thay vì chọn trên test.

Bước 1: chạy Lead-1..5 trên validation (không BERTScore, không đụng bảng test):
    python -m src.baselines.run_baselines --data_dir <clean> --split validation --systems lead --ks 1 2 3 4 5 --no_bertscore
Bước 2: chọn k:
    python -m src.analysis.select_lead_k --tables_dir results/tables

Quy tắc (khai ở D10): điểm chọn S = trung bình (ROUGE-1, ROUGE-2, ROUGE-Lsum) trên validation. Các k có S thấp hơn
S tốt nhất dưới 0,2 điểm (thang 0-100) được coi là hòa với k tốt nhất; trong nhóm hòa chọn k gần mốc cũ nhất
(CNN/DM: 3, GOV.UK: 1), nếu vẫn hòa thì k nhỏ hơn. Lưu ý: ngưỡng so với S TỐT NHẤT chứ không so từng cặp.

Đầu ra: results/tables/lead_k_selection.csv (S từng k, k tốt nhất, k được chọn, có khác mốc cũ không).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.decode_utils import table_path

METRICS = ("rouge1", "rouge2", "rougeLsum")
DEFAULT_K = {"cnn_dm": 3, "govuk": 1}        # mốc cũ của D8, chọn trên test
TOL = 0.002                                   # 0,2 điểm trên thang 0-1 của bảng CSV


def selection_score(row) -> float:
    """S = trung bình ROUGE-1/2/Lsum (thang 0-1 như trong CSV)."""
    return sum(float(row[m]) for m in METRICS) / len(METRICS)


def pick_k(df: pd.DataFrame, default_k: int, tol: float = TOL) -> dict:
    """df: các dòng Lead-k của MỘT nguồn trên validation (cột k, rouge1, rouge2, rougeLsum).

    Trả về {"scores": {k: S}, "best_k", "chosen_k"} theo quy tắc ở docstring module."""
    if df.empty:
        raise ValueError("Không có dòng Lead-k nào để chọn")
    scores = {int(r["k"]): selection_score(r) for _, r in df.iterrows()}
    best_k = max(scores, key=lambda k: (scores[k], -k))
    best = scores[best_k]
    tied = [k for k, s in scores.items() if best - s < tol]
    chosen = min(tied, key=lambda k: (abs(k - default_k), k))
    return {"scores": scores, "best_k": best_k, "chosen_k": chosen}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tables_dir", default="results/tables")
    ap.add_argument("--sources", nargs="+", default=["cnn_dm", "govuk"])
    ap.add_argument("--split", default="validation", help="chỉ nên là validation; test không được dùng để chọn k")
    a = ap.parse_args(argv)
    if a.split == "test":
        raise SystemExit("Không chọn k trên test (D10). Dùng --split validation.")
    rows = []
    for source in a.sources:
        path = table_path(a.tables_dir, "baselines", source, a.split)
        if not Path(path).exists():
            raise SystemExit(f"Chưa có {path}. Chạy run_baselines với --split {a.split} --systems lead --ks 1 2 3 4 5 trước.")
        d = pd.read_csv(path)
        d = d[d["system"] == "lead"]
        ks = sorted(int(k) for k in d["k"])
        if ks != [1, 2, 3, 4, 5]:
            print(f"[cảnh báo] {source}: có k = {ks}, kỳ vọng 1..5")
        res = pick_k(d, DEFAULT_K[source])
        for k, s in sorted(res["scores"].items()):
            rows.append({"source": source, "split": a.split, "k": k, "S_points": round(100 * s, 2),
                         "is_best": k == res["best_k"], "is_chosen": k == res["chosen_k"],
                         "old_default_k": DEFAULT_K[source], "chosen_differs_from_old": res["chosen_k"] != DEFAULT_K[source]})
        print(f"{source}: S theo k = { {k: round(100 * s, 2) for k, s in sorted(res['scores'].items())} } | "
              f"k tốt nhất = {res['best_k']} | k được chọn = {res['chosen_k']} (mốc cũ {DEFAULT_K[source]})")
    out = Path(a.tables_dir) / "lead_k_selection.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print("Đã lưu", out)


if __name__ == "__main__":
    main()
