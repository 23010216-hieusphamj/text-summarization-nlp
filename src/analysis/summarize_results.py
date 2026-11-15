"""Bảng so sánh tổng hợp các phương pháp trên TEST (GĐ6 Bước 8; dùng lại cho GĐ7): R1/R2/Lsum/BERTScore (CI 95%), độ dài, thời gian.

Gộp dòng từ: baselines_{source}.csv (Lead/TextRank, GĐ4), seq2seq_{source}.csv (GĐ5), hf_{source}.csv (BART/T5, GĐ6).
Danh sách hàng và nhãn trong configs/compare.yaml (mục `summary` của từng nguồn).

    python -m src.analysis.summarize_results --baseline_tables results/tables --hf_tables <Drive>/results/tables --out_dir results/tables

Lưu ý khi đọc cột ms/bài: Lead/TextRank chạy trên CPU, Seq2Seq và BART/T5 trên GPU T4 (batch 1): không so trực tiếp giữa các họ.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

METRICS = (("rouge1", "R1"), ("rouge2", "R2"), ("rougeLsum", "Lsum"), ("bertscore_f1", "BERTScore"))


def _pick(df: pd.DataFrame, item: dict) -> pd.Series | None:
    """Tìm đúng một dòng của `item` trong bảng nguồn; None nếu không có."""
    if item["kind"] == "baselines":
        m = df[(df["system"] == item["system"]) & (df["k"] == item["k"])]
    elif item["kind"] == "seq2seq":
        m = df[df["system"] == item["system"]]
    else:
        m = df[(df["system"] == item["system"]) & (df["gen_tag"] == item["gen_tag"])]
        if "split" in m.columns:
            m = m[m["split"] == "test"]
    return m.iloc[-1] if len(m) else None


def build_summary(items: list[dict], tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """tables: {'baselines': df, 'seq2seq': df, 'hf': df} (df có thể thiếu -> hàng bị bỏ và in cảnh báo)."""
    rows = []
    for it in items:
        df = tables.get(it["kind"] if it["kind"] != "baselines" else "baselines")
        r = None if df is None else _pick(df, it)
        if r is None:
            print(f"[cảnh báo] không thấy dòng của '{it['label']}' trong bảng {it['kind']}, bỏ qua")
            continue
        row = {"method": it["label"], "family": it["family"], "n": int(r["n"])}
        for col, name in METRICS:
            for suf, key in (("", name), ("_lo", f"{name}_lo"), ("_hi", f"{name}_hi")):
                v = r.get(f"{col}{suf}", np.nan)
                row[key] = 100 * float(v) if pd.notna(v) else np.nan
        row["avg_words"] = float(r.get("avg_pred_words", np.nan))
        ms = r.get("ms_per_doc_bs1", r.get("ms_per_doc", np.nan))
        row["ms_per_doc"] = float(ms) if pd.notna(ms) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def to_markdown(df: pd.DataFrame) -> str:
    """Bảng markdown: 'giá trị [lo, hi]' cho từng độ đo (thang x100)."""
    head = ["Phương pháp", "Họ", "R1", "R2", "Lsum", "BERTScore", "Số từ TB", "ms/bài"]
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
    for _, r in df.iterrows():
        def cell(n):
            return "–" if pd.isna(r[n]) else f"{r[n]:.1f} [{r[f'{n}_lo']:.1f}, {r[f'{n}_hi']:.1f}]"
        ms = "–" if pd.isna(r["ms_per_doc"]) else f"{r['ms_per_doc']:.0f}"
        lines.append(f"| {r['method']} | {r['family']} | {cell('R1')} | {cell('R2')} | {cell('Lsum')} | {cell('BERTScore')} "
                     f"| {r['avg_words']:.1f} | {ms} |")
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/compare.yaml")
    ap.add_argument("--baseline_tables", default="results/tables", help="chứa baselines_{source}.csv và seq2seq_{source}.csv")
    ap.add_argument("--hf_tables", default="results/tables", help="chứa hf_{source}.csv")
    ap.add_argument("--out_dir", default="results/tables")
    ap.add_argument("--sources", nargs="+", default=None)
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for source in a.sources or [s for s in cfg if s != "curves"]:
        tabs = {}
        for key, base in (("baselines", a.baseline_tables), ("seq2seq", a.baseline_tables), ("hf", a.hf_tables)):
            p = Path(base) / (f"{key}_{source}.csv" if key != "hf" else f"hf_{source}.csv")
            tabs[key] = pd.read_csv(p) if p.exists() else None
        df = build_summary(cfg[source]["summary"], tabs)
        df.to_csv(out / f"final_comparison_{source}.csv", index=False)
        md = to_markdown(df)
        (out / f"final_comparison_{source}.md").write_text(md + "\n", encoding="utf-8")
        print(f"\n=== {source} ===\n{md}")


if __name__ == "__main__":
    main()
