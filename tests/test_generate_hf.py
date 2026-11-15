"""Test giải mã + đánh giá bằng mô hình BART NHỎ ngẫu nhiên (CPU). Cần torch + transformers + rouge_score."""
import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("rouge_score")

import torch  # noqa: E402

from src import generate_hf as gh  # noqa: E402
from tests.test_train_hf import StubTok, data, make_model  # noqa: E402

EV = {"rouge_stemmer": True, "bootstrap_samples": 20, "bertscore": False, "bertscore_model": "x",
      "bertscore_rescale": True, "bertscore_batch_size": 4}
GEN = {"num_beams": 1, "max_length": 8, "min_length": 0, "length_penalty": 1.0, "no_repeat_ngram_size": 0}


def _run(tmp_path, split_tag="test", gen=GEN, system="sys"):
    ex, refs = data(8)
    recs = [{"id": e["id"], "article": " ".join(str(t) for t in e["input_ids"] if t > 2), "highlights": r}
            for e, r in zip(ex, refs)]
    dirs = {k: tmp_path / k for k in ("predictions", "tables", "scores")}
    row = gh.generate_and_evaluate(make_model(), StubTok(), ex, recs, gen, EV, source="x", system=system,
                                   split_tag=split_tag, dirs=dirs, batch_size=4, device=torch.device("cpu"),
                                   use_amp=False, time_docs=2, extra={"zero_shot": True})
    return row, dirs


def test_writes_predictions_scores_and_table_row(tmp_path):
    row, dirs = _run(tmp_path)
    tag = "b1-lp1-nr0-min0-max8"
    lines = (dirs["predictions"] / f"x_sys__{tag}.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 8
    sc = pd.read_csv(dirs["scores"] / f"x_sys__{tag}.csv")
    assert len(sc) == 8 and {"id", "rouge1", "rougeLsum", "pred_words", "ref_words"} <= set(sc.columns)
    tab = pd.read_csv(dirs["tables"] / "hf_x.csv")
    assert len(tab) == 1 and tab["gen_tag"].iloc[0] == tag and tab["split"].iloc[0] == "test" and tab["n"].iloc[0] == 8
    assert {"ms_per_doc_bs1", "ms_per_doc_batched", "rep_trigram_pct", "novel_trigram_pct", "zero_shot"} <= set(tab.columns)
    assert "pred_sentences_mean" in tab.columns and tab["pred_sentences_mean"].iloc[0] >= 1
    assert row["ms_per_doc_bs1"] > 0 and row["_predictions"] and abs(row["rouge1"] - sc["rouge1"].mean()) < 1e-9


def test_other_configs_and_splits_do_not_overwrite(tmp_path):
    _, dirs = _run(tmp_path)
    ex, _ = data(8)
    _run(tmp_path)                                                                  # cùng cấu hình: thay dòng, không thêm
    _run(tmp_path, gen={**GEN, "num_beams": 2})                                    # cấu hình khác: thêm dòng
    _run(tmp_path, split_tag="validation")                                         # tập khác: file khác
    tab = pd.read_csv(dirs["tables"] / "hf_x.csv")
    assert sorted(tab["gen_tag"]) == ["b1-lp1-nr0-min0-max8", "b2-lp1-nr0-min0-max8"]
    assert (dirs["tables"] / "hf_x_validation.csv").exists() and len(pd.read_csv(dirs["tables"] / "hf_x_validation.csv")) == 1
    assert (dirs["predictions"] / "x_sys__b1-lp1-nr0-min0-max8_validation.jsonl").exists()


def test_model_default_gen_reports_keys():
    d = gh.model_default_gen(make_model())
    assert set(d) == set(gh.HIDDEN_KEYS)
