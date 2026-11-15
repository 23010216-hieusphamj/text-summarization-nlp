import os

import numpy as np
import pandas as pd
import pytest
import yaml

from src.analysis import summarize_results as sr


def _row(**kw):
    r = {"n": 1000, "avg_pred_words": 60.0}
    for m in ("rouge1", "rouge2", "rougeLsum", "bertscore_f1"):
        r.update({m: 0.4, f"{m}_lo": 0.38, f"{m}_hi": 0.42})
    r.update(kw)
    return r


def _tables():
    return {"baselines": pd.DataFrame([{**_row(), "system": "lead", "k": 3, "ms_per_doc": 1.5},
                                       {**_row(), "system": "lead", "k": 1, "ms_per_doc": 1.0}]),
            "seq2seq": pd.DataFrame([{**_row(), "system": "seq2seq-beam4_nr3", "ms_per_doc": 20.0}]),
            "hf": pd.DataFrame([{**_row(rouge1=0.41), "system": "a", "gen_tag": "b4", "split": "test", "ms_per_doc_bs1": 900.0},
                                {**_row(rouge1=0.99), "system": "a", "gen_tag": "b4", "split": "validation", "ms_per_doc_bs1": 1.0}])}


ITEMS = [{"kind": "baselines", "system": "lead", "k": 3, "label": "Lead-3", "family": "Trích câu"},
         {"kind": "seq2seq", "system": "seq2seq-beam4_nr3", "label": "Seq2Seq", "family": "Sinh"},
         {"kind": "hf", "system": "a", "gen_tag": "b4", "label": "BART", "family": "Tinh chỉnh"},
         {"kind": "hf", "system": "zzz", "gen_tag": "b4", "label": "thiếu", "family": "x"}]


def test_build_summary_picks_right_rows_scales_to_percent_and_skips_missing():
    df = sr.build_summary(ITEMS, _tables())
    assert df["method"].tolist() == ["Lead-3", "Seq2Seq", "BART"]                # hàng thiếu bị bỏ
    bart = df[df.method == "BART"].iloc[0]
    assert abs(bart["R1"] - 41.0) < 1e-9 and abs(bart["R1_lo"] - 38.0) < 1e-9 and bart["ms_per_doc"] == 900.0   # lấy dòng test, không phải validation
    assert df[df.method == "Lead-3"].iloc[0]["ms_per_doc"] == 1.5


def test_markdown_has_one_row_per_method_with_ci_and_dash_for_nan():
    t = _tables()
    t["baselines"].loc[:, ["bertscore_f1", "bertscore_f1_lo", "bertscore_f1_hi"]] = np.nan
    md = sr.to_markdown(sr.build_summary(ITEMS[:3], t))
    lines = md.splitlines()
    assert len(lines) == 2 + 3 and "40.0 [38.0, 42.0]" in lines[2] and "–" in lines[2]


def test_real_baseline_and_seq2seq_tables_contain_requested_rows():
    for source in ("cnn_dm", "govuk"):
        bp, sp = f"results/tables/baselines_{source}.csv", f"results/tables/seq2seq_{source}.csv"
        if not (os.path.exists(bp) and os.path.exists(sp)):
            pytest.skip("không có bảng baseline trong repo")
        cfg = yaml.safe_load(open("configs/compare.yaml", encoding="utf-8"))[source]["summary"]
        items = [i for i in cfg if i["kind"] in ("baselines", "seq2seq")]
        df = sr.build_summary(items, {"baselines": pd.read_csv(bp), "seq2seq": pd.read_csv(sp), "hf": None})
        assert len(df) == len(items), (source, df["method"].tolist())            # mọi hàng baseline/seq2seq tìm thấy


def test_main_writes_csv_and_markdown(tmp_path):
    t = _tables()
    d = tmp_path / "tab"; d.mkdir()
    t["baselines"].to_csv(d / "baselines_s.csv", index=False)
    t["seq2seq"].to_csv(d / "seq2seq_s.csv", index=False)
    t["hf"].to_csv(d / "hf_s.csv", index=False)
    cp = tmp_path / "c.yaml"; cp.write_text(yaml.safe_dump({"s": {"summary": ITEMS}}))
    sr.main(["--config", str(cp), "--baseline_tables", str(d), "--hf_tables", str(d), "--out_dir", str(tmp_path / "o")])
    assert len(pd.read_csv(tmp_path / "o" / "final_comparison_s.csv")) == 3
    assert (tmp_path / "o" / "final_comparison_s.md").read_text(encoding="utf-8").count("\n") == 5
