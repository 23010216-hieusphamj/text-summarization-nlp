"""Test cho src/analysis/select_samples.py (D11): chọn mẫu, cờ tự động, ghép dữ liệu, ẩn điểm trong file mù."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.analysis import select_samples as ss

CFG = yaml.safe_load(open(Path(__file__).resolve().parents[1] / "configs" / "samples.yaml", encoding="utf-8"))


def _scores(n, seed=0, prefix="id"):
    rng = np.random.default_rng(seed)
    r = rng.uniform(0, 1, (n, 3))
    return pd.DataFrame({"id": [f"{prefix}{i:04d}" for i in range(n)], "rouge1": r[:, 0], "rouge2": r[:, 1], "rougeLsum": r[:, 2],
                         "rougeL": rng.uniform(0, 1, n), "bertscore_f1": rng.uniform(0, 1, n)})


def _data(n_cnn=200, n_gov=80):
    out = {}
    for source, n in (("cnn_dm", n_cnn), ("govuk", n_gov)):
        sc = _scores(n, seed=n, prefix=source[:2])
        recs = {i: {"id": i, "article": f"John Smith visited Paris in 2013 with {k} friends. They stayed three days.",
                    "highlights": "John visited Paris."} for k, i in enumerate(sc["id"])}
        preds = {i: "John Smith visited Paris." for i in sc["id"]}
        sc["pred_words"] = [ss.word_count(preds[i]) for i in sc["id"]]
        out[source] = {"scores": sc, "records": recs, "preds": preds, "system": "SYS_" + source}
    return out


# ----------------------------------------------------------------------------- chọn mẫu
def test_add_s_uses_three_rouge_columns_only():
    d = ss.add_s(pd.DataFrame({"rouge1": [0.3], "rouge2": [0.1], "rougeLsum": [0.2], "rougeL": [0.99]}))
    assert abs(d["S"].iloc[0] - 0.2) < 1e-12


def test_quartile_of_balanced_and_ordered_with_tie_break():
    df = pd.DataFrame({"id": [f"i{k}" for k in range(8)], "S": [0.5] * 8})        # hòa hết: theo id tăng dần
    q = ss.quartile_of(df)
    assert q["i0"] == 1 and q["i1"] == 1 and q["i2"] == 2 and q["i7"] == 4
    assert sorted(q.value_counts().tolist()) == [2, 2, 2, 2]
    df2 = ss.add_s(_scores(100))
    assert ss.quartile_of(df2).value_counts().tolist() == [25, 25, 25, 25]


def test_pick_error_top_bottom_correct_and_ties_by_id():
    df = pd.DataFrame({"id": ["b", "a", "c", "d", "e", "f"], "rouge1": [0.9, 0.9, 0.5, 0.1, 0.1, 0.3],
                       "rouge2": [0.9, 0.9, 0.5, 0.1, 0.1, 0.3], "rougeLsum": [0.9, 0.9, 0.5, 0.1, 0.1, 0.3]})
    r = ss.pick_error_samples(df, 2, 2, 0, 42)
    assert r[r.group == "top"]["id"].tolist() == ["a", "b"]                        # hòa S: id tăng dần
    assert r[r.group == "bottom"]["id"].tolist() == ["d", "e"]
    assert r[r.group == "top"]["rank_desc"].tolist() == [1, 2]


def test_pick_error_mid_in_percentile_range_disjoint_and_deterministic():
    df = _scores(200)
    r1 = ss.pick_error_samples(df, 4, 4, 4, 42, (0.40, 0.60))
    r2 = ss.pick_error_samples(df, 4, 4, 4, 42, (0.40, 0.60))
    assert r1.equals(r2) and len(r1) == 12 and r1["id"].is_unique
    d = ss.add_s(df).sort_values(["S", "id"]).reset_index(drop=True)
    pos = {i: k for k, i in enumerate(d["id"])}
    for i in r1[r1.group == "mid"]["id"]:
        assert 0.40 <= pos[i] / (len(d) - 1) <= 0.60


def test_pick_error_raises_when_not_enough_candidates():
    try:
        ss.pick_error_samples(_scores(10), 1, 1, 5, 42, (0.45, 0.55))
        raise AssertionError("phải báo lỗi")
    except ValueError as e:
        assert "ứng viên" in str(e)


def test_pick_human_counts_per_quartile_and_deterministic():
    df = _scores(400)
    h = ss.pick_human_samples(df, [8, 7, 8, 7], 42)
    assert h["quartile"].value_counts().sort_index().tolist() == [8, 7, 8, 7] and h["id"].is_unique
    q = ss.quartile_of(ss.add_s(df))
    assert all(q[i] == k for i, k in zip(h["id"], h["quartile"]))                  # đúng tứ phân vị
    assert h.equals(ss.pick_human_samples(df, [8, 7, 8, 7], 42))


def test_pick_human_raises_when_quartile_too_small():
    try:
        ss.pick_human_samples(_scores(8), [3, 0, 0, 0], 42)
        raise AssertionError("phải báo lỗi")
    except ValueError:
        pass


# ----------------------------------------------------------------------------- cờ tự động
def test_repeated_trigram_flag():
    assert ss.repeated_trigram("the cat sat on the mat. the cat sat on the rug") is True
    assert ss.repeated_trigram("the cat sat on the mat") is False


def test_entities_skip_sentence_initial_and_trailing_punct_and_possessive():
    e = ss.entities("The firm said Obama’s plan cost 1,000 pounds in 2013. Then Smith spoke. He met Dr Jones, aged 45.")
    assert "Obama" in e and "Smith" in e and "Jones" in e
    assert "The" not in e and "Then" not in e and "He" not in e                    # đầu câu
    assert "1,000" in e and "2013" in e and "45" in e                              # số, bỏ dấu câu cuối
    assert len(e) == len({x.lower() for x in e})                                   # không trùng


def test_unsupported_stats_supported_unsupported_and_boundaries():
    art = "Obama spoke in Paris about 100 people and 1000 dollars."
    u = ss.unsupported_stats(art, "He said Obama met Biden in paris. 1,000 people, 10 dollars.")
    # thực thể theo thứ tự xuất hiện: Obama, Biden, 1,000, 10 ("paris" viết thường nên không tính)
    assert u["unsupported"] == ["Biden", "10"]                                     # 10 không khớp trong 100; 1,000 khớp 1000
    assert u["n_entities"] == 4 and u["n_unsupported"] == 2 and abs(u["unsupported_rate"] - 0.5) < 1e-12
    none = ss.unsupported_stats(art, "nothing here at all")
    assert none["n_entities"] == 0 and np.isnan(none["unsupported_rate"])



def test_compute_flags_and_summary_shapes():
    data = _data(60, 20)
    d = data["cnn_dm"]

    fl = ss.compute_flags(
        d["records"],
        d["preds"],
        list(d["scores"]["id"])
    )

    assert len(fl) == 60

    required_columns = {
        "rep_trigram",
        "unsupported_rate",
        "novel_trigram",
        "article_words",
        "pred_words",
        "ref_words",
        "length_diff",
        "length_ratio",
    }

    assert required_columns <= set(fl.columns)

    # Dự đoán trong dữ liệu test giả có 4 từ.
    # Tham chiếu có 3 từ.
    assert (fl["length_diff"] == 1).all()

    assert np.allclose(
        fl["length_ratio"],
        4 / 3
    )

    s = ss.flag_summary(fl, d["scores"])

    assert s["group"].tolist() == [
        "all", "Q1", "Q2", "Q3", "Q4"
    ]

    assert s["n"].tolist() == [
        60, 15, 15, 15, 15
    ]


def test_compute_flags_empty_reference_length_ratio():
    records = {
        "sample_01": {
            "article": "This is an original article.",
            "highlights": ""
        }
    }

    preds = {
        "sample_01": "This is a short summary."
    }

    fl = ss.compute_flags(
        records,
        preds,
        ["sample_01"]
    )

    assert fl["pred_words"].iloc[0] == 5
    assert fl["ref_words"].iloc[0] == 0
    assert fl["length_diff"].iloc[0] == 5
    assert np.isnan(fl["length_ratio"].iloc[0])



def test_bootstrap_diff_means_sign_and_ci():
    a, b = np.full(50, 0.4), np.full(50, 0.1)
    d, lo, hi = ss.bootstrap_diff_means(a + np.linspace(0, .01, 50), b, 200, 42)
    assert abs(d - 0.3) < 0.01 and lo <= d <= hi
    assert np.isnan(ss.bootstrap_diff_means(np.array([np.nan]), b)[0])


def test_check_alignment_fatal_and_warning():
    sc = pd.DataFrame({"id": ["a", "b"], "pred_words": [3, 5]})
    recs = {"a": {}, "b": {}}
    try:
        ss.check_alignment(sc, {"a": "x"}, recs)
        raise AssertionError("phải lỗi")
    except ValueError:
        pass
    w = ss.check_alignment(sc, {"a": "one two three", "b": "x"}, recs)
    assert len(w) == 1 and "1/2" in w[0]
    assert ss.check_alignment(sc, {"a": "one two three", "b": "a b c d e"}, recs) == []


# ----------------------------------------------------------------------------- dựng lựa chọn, ẩn điểm, đầu-cuối
def test_build_selection_counts_blind_codes_and_rerate_subset():
    sel = ss.build_selection(_data(), CFG)
    k = sel["key"]
    assert len(sel["error"]) == 16 and (k.purpose == "error").sum() == 16
    hum = k[k.purpose == "human"]
    assert len(hum) == 40 and len(sel["human_blind"]) == 40 and len(sel["rerate_blind"]) == 8
    assert hum["item_id"].is_unique and sorted(hum["item_id"]) == [f"H{i:02d}" for i in range(1, 41)]
    assert (hum.rerate_id != "").sum() == 8 and set(hum[hum.rerate_id != ""]["rerate_id"]) == {f"R{i:02d}" for i in range(1, 9)}
    assert hum[hum.source == "cnn_dm"]["group"].value_counts().sort_index().tolist() == [8, 7, 8, 7]
    assert hum[hum.source == "govuk"]["group"].value_counts().sort_index().tolist() == [3, 2, 3, 2]
    for it in sel["human_blind"] + sel["rerate_blind"]:
        assert set(it) == {"item_id", "article", "summary"}                         # không có điểm, hệ thống, id thật
    by_item = {x["item_id"]: x for x in sel["human_blind"]}
    for _, r in hum[hum.rerate_id != ""].iterrows():                                # mẫu chấm lại có cùng nội dung với bản gốc
        rr = next(x for x in sel["rerate_blind"] if x["item_id"] == r["rerate_id"])
        assert rr["article"] == by_item[r["item_id"]]["article"] and rr["summary"] == by_item[r["item_id"]]["summary"]


def test_build_selection_deterministic():
    a, b = ss.build_selection(_data(), CFG), ss.build_selection(_data(), CFG)
    assert a["key"].equals(b["key"]) and a["human_blind"] == b["human_blind"] and a["rerate_blind"] == b["rerate_blind"]


def test_render_error_md_has_all_items():
    data = _data()
    sel = ss.build_selection(data, CFG)
    md = ss.render_error_md(sel["error"], data, CFG)
    assert md.count("## E") == 16 and "Có điểm tự động" in md and "Nhãn lỗi" in md


def test_main_end_to_end_writes_files_and_blind_dir_has_no_scores(tmp_path):
    data = _data()
    clean, preds, scores, out = tmp_path / "clean", tmp_path / "preds", tmp_path / "scores", tmp_path / "out"
    for p in (clean, preds, scores):
        p.mkdir()
    for source, d in data.items():
        c = CFG[source]
        ss.write_jsonl(clean / f"{source}_test.jsonl", list(d["records"].values()))
        ss.write_jsonl(preds / c["predictions"], [{"id": i, "prediction": t} for i, t in d["preds"].items()])
        d["scores"].to_csv(scores / c["scores"], index=False)
    ss.main(["--data_dir", str(clean), "--scores_dir", str(scores), "--pred_dir", str(preds), "--out_dir", str(out)])
    for f in ("samples_key.csv", "error_analysis_samples.csv", "error_analysis_samples.md", "flags_summary_cnn_dm.csv",
              "flags_cnn_dm_A_bart_512.csv", "blind/human_items_blind.jsonl", "blind/human_rerate_items_blind.jsonl"):
        assert (out / f).exists(), f
    blind = (out / "blind" / "human_items_blind.jsonl").read_text(encoding="utf-8")
    assert len(blind.strip().split("\n")) == 40
    for banned in ("rouge", "bertscore", "A_bart", "D_bart", "quartile", '"S"', "rank_desc"):
        assert banned not in blind

def test_main_flags_only_updates_both_sources_without_rebuilding_selection(tmp_path):
    """Chế độ flags_only phải tạo cờ cho cả hai nguồn mà không ghi đè kết quả chọn mẫu."""
    data = _data()

    clean = tmp_path / "clean"
    preds = tmp_path / "preds"
    scores = tmp_path / "scores"
    out = tmp_path / "out"

    for p in (clean, preds, scores):
        p.mkdir()

    for source, d in data.items():
        c = CFG[source]

        ss.write_jsonl(
            clean / f"{source}_test.jsonl",
            list(d["records"].values())
        )

        ss.write_jsonl(
            preds / c["predictions"],
            [
                {"id": i, "prediction": t}
                for i, t in d["preds"].items()
            ]
        )

        d["scores"].to_csv(scores / c["scores"], index=False)

    # Tạo file có sẵn để kiểm tra không bị ghi đè.
    out.mkdir()
    key_file = out / "samples_key.csv"
    key_file.write_text("KEEP_EXISTING_DATA", encoding="utf-8")

    ss.main([
        "--data_dir", str(clean),
        "--scores_dir", str(scores),
        "--pred_dir", str(preds),
        "--out_dir", str(out),
        "--flags_only",
    ])

    # Phải có file cờ của cả hai nguồn.
    # Tên hệ thống lấy từ config thực tế, không lấy từ dữ liệu giả lập.
    for source in data:
        system = CFG[source]["system"]

        assert (
            out / f"flags_{source}_{system}.csv"
        ).exists()

        assert (
            out / f"flags_summary_{source}.csv"
        ).exists()

    # Không được ghi đè hoặc tạo lại các kết quả chọn mẫu.
    assert key_file.read_text(encoding="utf-8") == "KEEP_EXISTING_DATA"

    assert not (out / "error_analysis_samples.csv").exists()
    assert not (out / "blind" / "human_items_blind.jsonl").exists()
