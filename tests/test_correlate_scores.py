"""Test cho src/analysis/correlate_scores.py (D11): hạng, Spearman/Pearson, bootstrap, kiểm tra ghép id, bảng đầu-cuối."""
import json

import numpy as np
import pandas as pd
import yaml

from src.analysis import correlate_scores as cs


def _scores(n=60, seed=0, noise=0.05):
    """Điểm giả: bertscore tăng theo rouge1 cộng nhiễu nhỏ."""
    rng = np.random.default_rng(seed)
    r1 = rng.uniform(0.1, 0.6, n)
    return pd.DataFrame({
        "id": [f"id{i:03d}" for i in range(n)],
        "rouge1": r1, "rouge2": r1 * 0.5 + rng.normal(0, 0.05, n),
        "rougeL": r1 * 0.9, "rougeLsum": r1 * 0.9 + rng.normal(0, 0.02, n),
        "bertscore_f1": r1 + rng.normal(0, noise, n),
    })


def test_rank_average_ties():
    r = cs._rank(np.array([10.0, 20.0, 20.0, 30.0]))
    assert r.tolist() == [1.0, 2.5, 2.5, 4.0]


def test_spearman_monotone_nonlinear_is_one_but_pearson_is_not():
    x = np.arange(1.0, 21.0)
    y = np.exp(x / 3)                                     # đơn điệu tăng, phi tuyến
    assert abs(cs.spearman(x, y) - 1.0) < 1e-12
    assert cs.pearson(x, y) < 0.99


def test_spearman_reversed_is_minus_one_and_known_value():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert abs(cs.spearman(x, x[::-1]) + 1.0) < 1e-12
    y = np.array([2.0, 1.0, 4.0, 3.0, 5.0])               # tính tay: sum d^2 = 4 -> rho = 1 - 6*4/(5*24) = 0.8
    assert abs(cs.spearman(x, y) - 0.8) < 1e-12


def test_constant_vector_gives_nan():
    assert np.isnan(cs.spearman(np.ones(10), np.arange(10.0)))
    assert np.isnan(cs.pearson(np.arange(2.0), np.arange(2.0)))      # dưới 3 mẫu


def test_add_s_is_mean_of_three_rouge_columns():
    df = cs.add_s(pd.DataFrame({"rouge1": [0.3], "rouge2": [0.1], "rougeLsum": [0.2], "rougeL": [0.9]}))
    assert abs(df["S"].iloc[0] - 0.2) < 1e-12                       # không dùng rougeL


def test_bootstrap_deterministic_and_ci_brackets_estimate():
    df = cs.add_s(_scores())
    pairs = [("S", "bertscore_f1"), ("rouge1", "bertscore_f1")]
    b1 = cs.bootstrap_corr(df, pairs, [], 200, 42)
    b2 = cs.bootstrap_corr(df, pairs, [], 200, 42)
    assert np.array_equal(b1[pairs[0]], b2[pairs[0]])                 # cùng seed -> cùng kết quả
    lo, hi = cs._ci(b1[pairs[1]])
    est = cs.spearman(df["rouge1"], df["bertscore_f1"])
    assert lo <= est <= hi and hi - lo < 0.3


def test_bootstrap_diff_is_paired_difference():
    df = cs.add_s(_scores())
    p1, p2 = ("rouge1", "bertscore_f1"), ("rouge2", "bertscore_f1")
    b = cs.bootstrap_corr(df, [p1, p2], [[p1, p2]], 100, 7)
    assert np.allclose(b[("diff", p1, p2)], b[p1] - b[p2], equal_nan=True)


def test_check_scores_flags_problems():
    df = _scores(10)
    assert cs.check_scores(df, 10, list(df["id"])) == []
    bad = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    bad.loc[1, "bertscore_f1"] = np.nan
    msgs = " | ".join(cs.check_scores(bad, 10, list(df["id"]) + ["zzz"]))
    assert "id trùng" in msgs and "khác kỳ vọng" in msgs and "giá trị thiếu" in msgs and "tập id khác" in msgs


def test_check_scores_missing_column():
    msgs = cs.check_scores(_scores(5).drop(columns=["bertscore_f1"]))
    assert any("thiếu cột bertscore_f1" in m for m in msgs)


def _cfg(n_expected):
    return {"seed": 1, "n_boot": 50,
            "pairs": [["S", "bertscore_f1"], ["rouge1", "bertscore_f1"], ["rouge2", "bertscore_f1"]],
            "diffs": [[["rouge1", "bertscore_f1"], ["rouge2", "bertscore_f1"]]],
            "src": {"n_expected": n_expected, "main": "good",
                    "systems": {"good": {"file": "good.csv"}, "nobert": {"file": "nobert.csv", "note": "x"},
                                "absent": {"file": "absent.csv"}}}}


def test_correlation_table_end_to_end(tmp_path):
    _scores(40).to_csv(tmp_path / "good.csv", index=False)
    _scores(40).drop(columns=["bertscore_f1"]).to_csv(tmp_path / "nobert.csv", index=False)
    df, warns = cs.correlation_table("src", _cfg(40), tmp_path)
    assert set(df["system"]) == {"good"}                              # thiếu file / thiếu BERTScore thì bị bỏ
    assert any("absent" in w and "không thấy" in w for w in warns)
    assert any("nobert" in w and "BERTScore" in w for w in warns)
    assert len(df) == 3 + 1                                           # 3 cặp + 1 hiệu
    r = df[df["pair"] == "S ~ bertscore_f1"].iloc[0]
    assert r["n"] == 40 and r["spearman"] > 0.7 and r["spearman_lo"] <= r["spearman"] <= r["spearman_hi"]
    assert df["pair"].str.contains(" - ").sum() == 1


def test_correlation_table_warns_on_wrong_n(tmp_path):
    _scores(40).to_csv(tmp_path / "good.csv", index=False)
    _, warns = cs.correlation_table("src", _cfg(1000), tmp_path)
    assert any("khác kỳ vọng" in w for w in warns)


def test_main_writes_tables_and_checks_ids(tmp_path, capsys):
    sc, out, clean = tmp_path / "scores", tmp_path / "tables", tmp_path / "clean"
    sc.mkdir(); clean.mkdir()
    df = _scores(30)
    df.to_csv(sc / "good.csv", index=False)
    with open(clean / "src_test.jsonl", "w", encoding="utf-8") as f:
        for i in list(df["id"])[:-1]:                                 # thiếu 1 id so với file điểm -> phải cảnh báo
            f.write(json.dumps({"id": i, "article": "a", "highlights": "b"}) + "\n")
    cfg = _cfg(30)
    cfg["systems"] = None
    yaml.safe_dump(cfg, open(tmp_path / "c.yaml", "w", encoding="utf-8"), allow_unicode=True)
    cs.main(["--config", str(tmp_path / "c.yaml"), "--scores_dir", str(sc), "--tables_dir", str(out),
             "--data_dir", str(clean), "--sources", "src"])
    assert (out / "correlation_src.csv").exists() and (out / "correlation_src.md").exists()
    txt = capsys.readouterr().out
    assert "tập id khác data/clean" in txt and "Spearman" in txt


def test_to_markdown_has_one_row_per_system(tmp_path):
    _scores(40).to_csv(tmp_path / "good.csv", index=False)
    df, _ = cs.correlation_table("src", _cfg(40), tmp_path)
    md = cs.to_markdown(df)
    assert md.count("\n") == 2 and "| good | 40 |" in md
