"""evaluate_with_samples phải cho cùng dòng kết quả như evaluate_predictions và trả thêm điểm từng mẫu
(rouge_per_sample được thay bằng hàm giả nên test chạy được mà không cần rouge-score)."""
import numpy as np

from src import evaluate as ev


def _fake_rouge(preds, refs, use_stemmer=True):
    base = np.array([len(p.split()) / 10 for p in preds])
    return {"rouge1": base, "rouge2": base / 2, "rougeL": base / 3, "rougeLsum": base / 4}


def test_with_samples_matches_predictions_row_and_returns_per_sample(monkeypatch):
    monkeypatch.setattr(ev, "rouge_per_sample", _fake_rouge)
    preds, refs = ["a b c", "a b", "a"], ["x y", "x y z", "x"]
    row, per = ev.evaluate_with_samples(preds, refs, bertscore=False, n_boot=50)
    assert ev.evaluate_predictions(preds, refs, bertscore=False, n_boot=50) == row
    assert set(per) == {"rouge1", "rouge2", "rougeL", "rougeLsum"} and per["rouge1"].tolist() == [0.3, 0.2, 0.1]
    assert row["n"] == 3 and abs(row["rouge1"] - 0.2) < 1e-12 and row["avg_pred_words"] == 2.0
    assert row["rouge1_lo"] <= row["rouge1"] <= row["rouge1_hi"]


def test_with_samples_requires_equal_lengths(monkeypatch):
    monkeypatch.setattr(ev, "rouge_per_sample", _fake_rouge)
    try:
        ev.evaluate_with_samples(["a"], ["x", "y"], bertscore=False)
    except AssertionError:
        return
    raise AssertionError("phải báo lỗi khi độ dài khác nhau")
