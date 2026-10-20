"""Chạy Lead-k / Random-k / TextRank / Oracle trên CNN/DM và GOV.UK, đánh giá và xuất bảng CSV.

    python -m src.baselines.run_baselines --config configs/baselines.yaml \\
        --data_dir /content/drive/MyDrive/<thư mục>/clean
    # chạy thử nhanh, không BERTScore:
    python -m src.baselines.run_baselines --limit 50 --no_bertscore
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.baselines.lead import join_sentences, lead_k, oracle_k, random_k
from src.baselines.textrank import textrank_k
from src.data.eda_utils import split_sentences
from src.evaluate import evaluate_predictions


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def make_selector(system: str, k: int, cfg: dict):
    """Trả về hàm (record, doc_index) -> danh sách chỉ số câu được chọn."""
    seed, tr = cfg["seed"], cfg.get("textrank", {})
    if system == "lead":
        return lambda r, i: lead_k(r["article"], k)
    if system == "random":
        return lambda r, i: random_k(r["article"], k, seed + i)     # mỗi bài một seed -> tái lập được
    if system == "textrank":
        return lambda r, i: textrank_k(r["article"], k, tr.get("stop_words", "english"), tr.get("damping", 0.85))
    if system == "oracle":
        return lambda r, i: oracle_k(r["article"], r["highlights"], k)
    raise ValueError(f"Hệ thống không biết: {system}")


def position_stats(source: str, records: list[dict], oracle_idx: list[list[int]], k: int) -> dict:
    """Oracle (chọn theo ROUGE) nằm ở đâu trong bài? Kiểm chứng chéo lead bias của D2/D3 bằng ROUGE thay vì độ trùng từ."""
    flat = [i for idx in oracle_idx for i in idx]
    n_sents = [len(split_sentences(r["article"])) for r in records]
    chance3 = float(np.mean([min(3, n) / n for n in n_sents if n]))
    return {"source": source, "oracle_k": k, "n_selected": len(flat),
            "pct_in_first3": 100 * np.mean([i < 3 for i in flat]),
            "pct_in_first10": 100 * np.mean([i < 10 for i in flat]),
            "median_index": float(np.median(flat)),
            "pct_first3_if_uniform": 100 * chance3}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/baselines.yaml")
    ap.add_argument("--data_dir", default=None, help="ghi đè data_dir trong config (vd. thư mục trên Google Drive)")
    ap.add_argument("--limit", type=int, default=None, help="chỉ lấy N mẫu đầu để chạy thử")
    ap.add_argument("--no_bertscore", action="store_true")
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    data_dir = Path(a.data_dir or cfg["data_dir"])
    ev = dict(cfg["evaluate"])
    use_bs = ev["bertscore"] and not a.no_bertscore
    tables, preds_dir = Path(cfg["output"]["tables_dir"]), Path(cfg["output"]["predictions_dir"])
    tables.mkdir(parents=True, exist_ok=True); preds_dir.mkdir(parents=True, exist_ok=True)
    pos_rows = []

    for source, scfg in cfg["sources"].items():
        path = data_dir / f"{source}_{cfg['split']}.jsonl"
        records = read_jsonl(path)[: a.limit]
        refs = [r["highlights"] for r in records]
        print(f"\n=== {source}: {len(records)} mẫu ({path}) ===")
        rows = []
        for system in cfg["systems"]:
            for k in scfg["ks"]:
                sel_fn = make_selector(system, k, cfg)
                t0 = time.perf_counter()
                idx = [sel_fn(r, i) for i, r in enumerate(records)]
                ms = 1000 * (time.perf_counter() - t0) / len(records)
                preds = [join_sentences(r["article"], ix) for r, ix in zip(records, idx)]
                name = f"{system}-{k}"
                with open(preds_dir / f"{source}_{name}.jsonl", "w", encoding="utf-8") as f:
                    for r, p, ix in zip(records, preds, idx):
                        f.write(json.dumps({"id": r["id"], "prediction": p, "selected": ix}, ensure_ascii=False) + "\n")
                row = evaluate_predictions(
                    preds, refs, use_stemmer=ev["rouge_stemmer"], bertscore=use_bs,
                    bertscore_model=ev["bertscore_model"], bertscore_rescale=ev["bertscore_rescale"],
                    bertscore_batch_size=ev["bertscore_batch_size"], n_boot=ev["bootstrap_samples"], seed=cfg["seed"])
                row.update(source=source, system=system, k=k, ms_per_doc=ms)
                rows.append(row)
                print(f"{name:12s} R1={row['rouge1']:.4f} R2={row['rouge2']:.4f} RLsum={row['rougeLsum']:.4f} "
                      f"len={row['avg_pred_words']:.1f} (ref {row['avg_ref_words']:.1f}) {ms:.1f} ms/doc")
                if system == "oracle" and k == max(scfg["ks"]):
                    pos_rows.append(position_stats(source, records, idx, k))
        first = ["source", "system", "k", "n"]
        df = pd.DataFrame(rows); df = df[first + [c for c in df.columns if c not in first]]
        df.to_csv(tables / f"baselines_{source}.csv", index=False)

    if pos_rows:
        pd.DataFrame(pos_rows).to_csv(tables / "oracle_position.csv", index=False)
        print("\nVị trí câu của Oracle (so với D2/D3):\n", pd.DataFrame(pos_rows).round(1).to_string(index=False))


if __name__ == "__main__":
    main()
