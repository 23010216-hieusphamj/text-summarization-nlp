import json
import re

import numpy as np
import pandas as pd
import pytest
import yaml

from src.analysis import compare_hf as ch
from src.hf_train_utils import scores_path

EV = {"rouge_stemmer": True, "bertscore_model": "x", "bertscore_rescale": True, "bertscore_batch_size": 4}


def _fake_rouge(preds, refs, use_stemmer=True):
    base = np.array([len(p.split()) / 10 for p in preds])
    return {"rouge1": base, "rouge2": base / 2, "rougeL": base / 3, "rougeLsum": base / 4}


def _frame(ids, **cols):
    return pd.DataFrame({"id": ids, **cols})


def test_align_keeps_common_ids_in_first_frame_order():
    a = _frame(["a", "b", "c", "d"], rouge1=[1, 2, 3, 4])
    b = _frame(["d", "c", "x"], rouge1=[9, 8, 7])
    out = ch.align({"A": a, "B": b})
    assert list(out["A"].index) == ["c", "d"] and list(out["B"].index) == ["c", "d"]
    assert out["B"]["rouge1"].tolist() == [8, 9]                                  # B được sắp lại theo thứ tự chung
    assert ch.align({}) == {}


def test_paired_diff_detects_consistent_gain_and_not_noise():
    rs = np.random.RandomState(0)
    ids = [str(i) for i in range(300)]
    base = rs.rand(300)
    frames = ch.align({"B": _frame(ids, rouge1=base, bertscore_f1=base),
                       "A": _frame(ids, rouge1=base + 0.05 + 0.01 * rs.randn(300), bertscore_f1=base + 0.0 * rs.randn(300)),
                       "C": _frame(ids, rouge1=base + 0.2 * rs.randn(300), bertscore_f1=base)})
    res = ch.paired_diff_table(frames, [("A", "B"), ("C", "B"), ("A", "missing")], n_boot=300)
    ab = res[(res.A == "A") & (res.metric == "rouge1")].iloc[0]
    assert ab["significant"] and 0.04 < ab["diff"] < 0.06 and ab["n"] == 300
    cb = res[(res.A == "C") & (res.metric == "rouge1")].iloc[0]
    assert not cb["significant"]                                                   # nhiễu 0 trung bình
    same = res[(res.A == "A") & (res.metric == "bertscore_f1")].iloc[0]
    assert abs(same["diff"]) < 1e-12 and not same["significant"]
    assert set(res["metric"]) == {"rouge1", "bertscore_f1"}                        # chỉ chỉ số có ở cả hai; cặp thiếu bị bỏ


def test_by_group_tables_counts_means_and_within_group_diffs():
    ids = [str(i) for i in range(40)]
    groups = pd.Series(["<=512"] * 10 + [">512"] * 30, index=ids)
    a = np.r_[np.full(10, 0.5), np.full(30, 0.4)]
    frames = ch.align({"A": _frame(ids, rouge1=a), "B": _frame(ids, rouge1=a - 0.1)})
    means, diffs = ch.by_group_tables(frames, groups, [("A", "B")], metrics=("rouge1",), n_boot=100)
    m = means.set_index(["system", "group"])
    assert m.loc[("A", "<=512"), "n"] == 10 and m.loc[("A", ">512"), "n"] == 30
    assert abs(m.loc[("A", ">512"), "rouge1"] - 0.4) < 1e-12
    d = diffs.set_index("group")
    assert abs(d.loc["<=512", "diff"] - 0.1) < 1e-12 and d.loc[">512", "significant"] is np.True_ or bool(d.loc[">512", "significant"])


def test_load_system_scores_hf_lead_preds_and_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(ch, "rouge_per_sample", _fake_rouge)
    recs = [{"id": f"d{i}", "article": "Aa bb. Cc dd. Ee ff. Gg hh.", "highlights": "x y"} for i in range(3)]
    dirs = {"scores": tmp_path / "scores", "predictions": tmp_path / "preds"}
    dirs["scores"].mkdir(); dirs["predictions"].mkdir()
    # hf: đọc file điểm sẵn có
    pd.DataFrame({"id": ["d0", "d1"], "rouge1": [0.1, 0.2]}).to_csv(scores_path(dirs["scores"], "s", "sys__b4", "test"), index=False)
    hf = ch.load_system_scores({"type": "hf", "system": "sys", "gen_tag": "b4"}, "s", "test", recs, dirs, EV, False)
    assert hf["rouge1"].tolist() == [0.1, 0.2]
    assert ch.load_system_scores({"type": "hf", "system": "none", "gen_tag": "b4"}, "s", "test", recs, dirs, EV, False) is None
    # lead: tính lại từ bài (2 câu đầu -> 4 từ) và cache
    lead = ch.load_system_scores({"type": "lead", "k": 2}, "s", "test", recs, dirs, EV, False)
    assert lead["rouge1"].tolist() == [0.4, 0.4, 0.4] and scores_path(dirs["scores"], "s", "lead-2", "test").exists()
    monkeypatch.setattr(ch, "rouge_per_sample", lambda *a, **k: (_ for _ in ()).throw(AssertionError("phải dùng cache")))
    assert ch.load_system_scores({"type": "lead", "k": 2}, "s", "test", recs, dirs, EV, False)["rouge1"].tolist() == [0.4] * 3
    # preds: đọc jsonl; thiếu file -> None
    monkeypatch.setattr(ch, "rouge_per_sample", _fake_rouge)
    spec = {"type": "preds", "file": "s_seq2seq-beam4_nr3.jsonl"}
    assert ch.load_system_scores(spec, "s", "test", recs, dirs, EV, False) is None
    with open(dirs["predictions"] / spec["file"], "w", encoding="utf-8") as f:
        for i in range(2):
            f.write(json.dumps({"id": f"d{i}", "prediction": "p q r"}) + "\n")
    s2 = ch.load_system_scores(spec, "s", "test", recs, dirs, EV, False)
    assert s2["id"].tolist() == ["d0", "d1"] and s2["rouge1"].tolist() == [0.3, 0.3]
    assert scores_path(dirs["scores"], "s", "seq2seq-beam4_nr3", "test").exists()


def test_cached_scores_without_bertscore_are_recomputed_when_bertscore_requested(tmp_path, monkeypatch):
    import src.evaluate as ev_mod
    monkeypatch.setattr(ch, "rouge_per_sample", _fake_rouge)
    monkeypatch.setattr(ev_mod, "bertscore_per_sample", lambda p, r, *a, **k: np.full(len(p), 0.7))
    recs = [{"id": "d0", "article": "Aa bb. Cc dd.", "highlights": "x"}]
    dirs = {"scores": tmp_path / "scores", "predictions": tmp_path}
    first = ch.load_system_scores({"type": "lead", "k": 1}, "s", "test", recs, dirs, EV, False)
    assert "bertscore_f1" not in first.columns
    second = ch.load_system_scores({"type": "lead", "k": 1}, "s", "test", recs, dirs, EV, True)
    assert second["bertscore_f1"].tolist() == [0.7]


def test_compare_yaml_is_internally_consistent():
    cfg = yaml.safe_load(open("configs/compare.yaml", encoding="utf-8"))
    runs = set()
    import glob
    for f in glob.glob("configs/*.yaml"):
        c = yaml.safe_load(open(f, encoding="utf-8"))
        if isinstance(c, dict) and "run_name" in c and "train" in c:
            runs.add(c["run_name"])
    for source in ("cnn_dm", "govuk"):
        e = cfg[source]
        names = set(e["systems"])
        for a, b in e["pairs"] + e["by_length_pairs"]:
            assert a in names and b in names, (source, a, b)
        assert set(e["by_length_systems"]) <= names
        for name, s in e["systems"].items():
            assert s["type"] in ("hf", "lead", "preds")
            if s["type"] == "hf":
                base = s["system"].removesuffix("_zeroshot").removesuffix("_xdomain")
                assert base in runs, (name, base)
                assert re.fullmatch(r"b\d+-lp[\d.]+-nr\d+-min\d+-max\d+(-es)?", s["gen_tag"]), s["gen_tag"]
            if s["type"] == "preds":
                assert s["file"].startswith(source + "_") and s["file"].endswith(".jsonl")
        labels = [i["label"] for i in e["summary"]]
        assert len(set(labels)) == len(labels)
    for group, rs in cfg["curves"].items():
        assert set(rs) <= runs, group


def test_main_end_to_end_with_fake_rouge(tmp_path, monkeypatch):
    monkeypatch.setattr(ch, "rouge_per_sample", _fake_rouge)
    data = tmp_path / "clean"; data.mkdir()
    recs = [{"id": f"d{i}", "article": "Aa bb. Cc dd. Ee ff.", "highlights": "x y"} for i in range(30)]
    with open(data / "s_test.jsonl", "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    scores = tmp_path / "scores"; scores.mkdir()
    pd.DataFrame({"id": [r["id"] for r in recs], "rouge1": 0.6, "rouge2": 0.3, "rougeLsum": 0.5}).to_csv(
        scores_path(scores, "s", "sys__b4", "test"), index=False)
    pd.DataFrame({"id": [r["id"] for r in recs], "group_512": ["<=512"] * 10 + [">512"] * 20}).to_csv(
        tmp_path / "token_lengths_s_test.csv", index=False)
    cfg = {"s": {"systems": {"lead-1": {"type": "lead", "k": 1}, "m": {"type": "hf", "system": "sys", "gen_tag": "b4"}},
                 "pairs": [["m", "lead-1"]], "by_length_systems": ["lead-1", "m"], "by_length_pairs": [["m", "lead-1"]]}}
    cp = tmp_path / "c.yaml"; cp.write_text(yaml.safe_dump(cfg))
    ev = tmp_path / "ev.yaml"; ev.write_text(yaml.safe_dump({"evaluate": EV}))
    ch.main(["--config", str(cp), "--eval_config", str(ev), "--data_dir", str(data), "--pred_dir", str(tmp_path / "p"),
             "--scores_dir", str(scores), "--tables_dir", str(tmp_path / "t"), "--lengths_dir", str(tmp_path), "--n_boot", "50"])
    t = pd.read_csv(tmp_path / "t" / "hf_paired_tests_s.csv")
    assert set(t["metric"]) == {"rouge1", "rouge2", "rougeLsum"} and (t["n"] == 30).all()
    bl = pd.read_csv(tmp_path / "t" / "hf_by_length_s.csv")
    assert sorted(bl["group"].unique()) == ["<=512", ">512"] and bl.groupby("group")["n"].first().tolist() == [10, 20]
    assert (tmp_path / "t" / "hf_by_length_paired_s.csv").exists()
