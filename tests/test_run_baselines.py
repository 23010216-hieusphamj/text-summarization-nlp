"""Kiểm tra logic ghi file của run_baselines (không tính ROUGE/BERTScore thật: evaluate_predictions được thay bằng hàm giả).

Mục tiêu: chạy trên validation hoặc chạy thử --limit KHÔNG được ghi đè bảng/dự đoán của test.
"""
import json

import yaml

from src.baselines import run_baselines as rb

ART = ("The council approved a new budget on Monday. Funding for schools will rise by ten percent. "
       "Opposition members criticised the plan. Roads will get repaired next year. "
       "A weather report says rain is likely on Friday.")


def _fake_eval(preds, refs, **kw):
    return {"n": len(preds), "rouge1": 0.3, "rouge2": 0.1, "rougeL": 0.2, "rougeLsum": 0.25,
            "avg_pred_words": 10.0, "avg_ref_words": 8.0}


def _setup(tmp_path):
    data = tmp_path / "clean"; data.mkdir()
    for src in ("cnn_dm", "govuk"):
        for split in ("validation", "test"):
            with open(data / f"{src}_{split}.jsonl", "w", encoding="utf-8") as f:
                for i in range(4):
                    f.write(json.dumps({"id": f"{src}{split}{i}", "article": ART, "highlights": "Budget approved."}) + "\n")
    cfg = {"data_dir": str(data), "split": "test", "seed": 42,
           "sources": {"cnn_dm": {"ks": [1, 2, 3]}, "govuk": {"ks": [1, 2, 3]}},
           "systems": ["lead"], "textrank": {},
           "evaluate": {"rouge_stemmer": True, "bootstrap_samples": 10, "bertscore": False, "bertscore_model": "x",
                        "bertscore_rescale": True, "bertscore_batch_size": 4},
           "output": {"tables_dir": str(tmp_path / "tables"), "predictions_dir": str(tmp_path / "preds")}}
    cfgp = tmp_path / "cfg.yaml"
    cfgp.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return cfgp, tmp_path / "tables", tmp_path / "preds"


def test_validation_run_leaves_test_table_and_predictions_untouched(tmp_path, monkeypatch):
    import pandas as pd
    monkeypatch.setattr(rb, "evaluate_predictions", _fake_eval)
    cfgp, tables, preds = _setup(tmp_path)
    rb.main(["--config", str(cfgp), "--no_bertscore"])                       # chạy test (mặc định của config)
    test_csv = (tables / "baselines_cnn_dm.csv").read_bytes()
    test_pred = (preds / "cnn_dm_lead-3.jsonl").read_bytes()
    rb.main(["--config", str(cfgp), "--split", "validation", "--systems", "lead", "--ks", "1", "2", "3", "4", "5",
             "--no_bertscore"])
    assert (tables / "baselines_cnn_dm.csv").read_bytes() == test_csv           # bảng test không đổi
    assert (preds / "cnn_dm_lead-3.jsonl").read_bytes() == test_pred            # dự đoán test không đổi
    val = pd.read_csv(tables / "baselines_cnn_dm_validation.csv")
    assert sorted(val["k"]) == [1, 2, 3, 4, 5] and set(val["split"]) == {"validation"}
    assert (preds / "cnn_dm_lead-5_validation.jsonl").exists()
    assert (tables / "baselines_govuk_validation.csv").exists()


def test_partial_rerun_merges_instead_of_overwriting_and_old_table_without_split_is_kept(tmp_path, monkeypatch):
    import pandas as pd
    monkeypatch.setattr(rb, "evaluate_predictions", _fake_eval)
    cfgp, tables, preds = _setup(tmp_path)
    tables.mkdir()
    pd.DataFrame([{"source": "cnn_dm", "system": "textrank", "k": 3, "n": 1000, "rouge1": 0.352}]).to_csv(
        tables / "baselines_cnn_dm.csv", index=False)                           # bảng cũ: chưa có cột split
    rb.main(["--config", str(cfgp), "--systems", "lead", "--ks", "3", "--no_bertscore"])
    out = pd.read_csv(tables / "baselines_cnn_dm.csv")
    assert set(out["system"]) == {"textrank", "lead"} and len(out) == 2         # dòng TextRank cũ vẫn còn
    assert set(out["split"]) == {"test"}                                        # dòng cũ được điền split=test


def test_limit_run_writes_only_limit_tagged_files(tmp_path, monkeypatch):
    monkeypatch.setattr(rb, "evaluate_predictions", _fake_eval)
    cfgp, tables, preds = _setup(tmp_path)
    rb.main(["--config", str(cfgp), "--limit", "2", "--no_bertscore"])
    names = {p.name for p in tables.iterdir()}
    assert names == {"baselines_cnn_dm_test_limit2.csv", "baselines_govuk_test_limit2.csv"}
    assert not (tables / "baselines_cnn_dm.csv").exists()
    assert (preds / "cnn_dm_lead-3_test_limit2.jsonl").exists() and not (preds / "cnn_dm_lead-3.jsonl").exists()
