"""Đánh giá tóm tắt: ROUGE-1/2/L/Lsum + BERTScore + độ dài, kèm khoảng tin cậy bootstrap 95%.

Dùng lại ở GĐ6–7 cho BART/T5. Quy ước: các câu của một tóm tắt ngăn nhau bằng '\\n' (ROUGE-Lsum).

    python -m src.evaluate --pred results/predictions/govuk_lead-1.jsonl \\
        --ref data/processed/clean/govuk_test.jsonl --out results/tables/eval_one.csv
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.eda_utils import word_count

ROUGE_KEYS = ("rouge1", "rouge2", "rougeL", "rougeLsum")


def bootstrap_ci(values, n_boot: int = 1000, seed: int = 42, alpha: float = 0.05) -> tuple[float, float, float]:
    """(mean, lo, hi) bằng bootstrap theo mẫu. Test chỉ 389–1000 mẫu nên cần khoảng tin cậy."""
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, v.size, size=(n_boot, v.size))].mean(axis=1)
    return float(v.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def rouge_per_sample(preds: list[str], refs: list[str], use_stemmer: bool = True) -> dict[str, np.ndarray]:
    """F1 của từng mẫu cho rouge1/rouge2/rougeL/rougeLsum (thư viện rouge-score)."""
    from rouge_score import rouge_scorer
    sc = rouge_scorer.RougeScorer(list(ROUGE_KEYS), use_stemmer=use_stemmer)
    out = {k: [] for k in ROUGE_KEYS}
    for p, r in zip(preds, refs):
        s = sc.score(r, p)               # (target, prediction)
        for k in ROUGE_KEYS:
            out[k].append(s[k].fmeasure)
    return {k: np.array(v) for k, v in out.items()}


_BS_CACHE: dict = {}


def _get_bertscorer(model_type: str, rescale: bool, batch_size: int):
    """Nạp model BERTScore MỘT lần rồi dùng lại (trước đây nạp lại cho mỗi hệ thống)."""
    from bert_score import BERTScorer
    key = (model_type, rescale)
    if key not in _BS_CACHE:
        _BS_CACHE[key] = BERTScorer(model_type=model_type, lang="en", rescale_with_baseline=rescale,
                                    batch_size=batch_size)
    return _BS_CACHE[key]


def bertscore_per_sample(preds: list[str], refs: list[str], model_type: str = "roberta-large",
                         rescale: bool = True, batch_size: int = 32) -> np.ndarray:
    """BERTScore F1 từng mẫu. Xuống dòng được đổi thành khoảng trắng."""
    p = [x.replace("\n", " ") or "." for x in preds]
    r = [x.replace("\n", " ") for x in refs]
    try:
        _, _, f1 = _get_bertscorer(model_type, rescale, batch_size).score(p, r)
    except Exception as e:               # thiếu file baseline cho model đó
        if not rescale:
            raise
        warnings.warn(f"BERTScore rescale thất bại ({e}); chạy lại không rescale")
        _, _, f1 = _get_bertscorer(model_type, False, batch_size).score(p, r)
    return f1.numpy()


def evaluate_predictions(preds: list[str], refs: list[str], *, use_stemmer: bool = True, bertscore: bool = True,
                         bertscore_model: str = "roberta-large", bertscore_rescale: bool = True,
                         bertscore_batch_size: int = 32, n_boot: int = 1000, seed: int = 42) -> dict:
    """Trả về một dòng kết quả: mean / lo / hi cho từng chỉ số + độ dài trung bình (số từ)."""
    assert len(preds) == len(refs) and preds, "preds và refs phải cùng độ dài và không rỗng"
    metrics = rouge_per_sample(preds, refs, use_stemmer)
    if bertscore:
        metrics["bertscore_f1"] = bertscore_per_sample(preds, refs, bertscore_model, bertscore_rescale,
                                                       bertscore_batch_size)
    row = {"n": len(preds)}
    for name, vals in metrics.items():
        m, lo, hi = bootstrap_ci(vals, n_boot, seed)
        row.update({name: m, f"{name}_lo": lo, f"{name}_hi": hi})
    row["avg_pred_words"] = float(np.mean([word_count(p) for p in preds]))
    row["avg_ref_words"] = float(np.mean([word_count(r) for r in refs]))
    return row


def _read_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True, help="jsonl có trường id, prediction")
    ap.add_argument("--ref", required=True, help="jsonl có trường id, highlights")
    ap.add_argument("--out", default=None, help="csv đầu ra (tùy chọn)")
    ap.add_argument("--no_bertscore", action="store_true")
    ap.add_argument("--bertscore_model", default="roberta-large")
    a = ap.parse_args()
    refs = {r["id"]: r["highlights"] for r in _read_jsonl(a.ref)}
    preds = _read_jsonl(a.pred)
    ids = [p["id"] for p in preds if p["id"] in refs]
    if len(ids) != len(preds):
        warnings.warn(f"{len(preds) - len(ids)} dự đoán không có tham chiếu tương ứng, bị bỏ qua")
    pmap = {p["id"]: p["prediction"] for p in preds}
    row = evaluate_predictions([pmap[i] for i in ids], [refs[i] for i in ids],
                               bertscore=not a.no_bertscore, bertscore_model=a.bertscore_model)
    df = pd.DataFrame([row]); print(df.T.to_string())
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True); df.to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
