"""Test cho phân tích chấm tay D11: kiểm tra phiếu, ghép điểm và độ đồng thuận liên giám khảo."""
import pandas as pd
import pytest

from src.analysis import analyze_human as ah


def _human(ids=("H01", "H02", "H03", "H04")):
    return pd.DataFrame({
        "item_id": list(ids),
        "fluency_1_5": [4, 3, 5, 2][:len(ids)],
        "completeness_1_5": [3, 2, 4, 1][:len(ids)],
        "factuality_1_5": [5, 4, 5, 2][:len(ids)],
        "error_type": ["none", "missing_information", "hallucination; missing_information", "other"][:len(ids)],
        "notes": ["a", "b", "c", "d"][:len(ids)],
    })


def _key():
    return pd.DataFrame([
        {"purpose": "human", "source": "cnn_dm", "system": "A", "id": "id1", "S": "0.3", "item_id": "H01", "rerate_id": "R01", "group": "Q1"},
        {"purpose": "human", "source": "cnn_dm", "system": "A", "id": "id2", "S": "0.4", "item_id": "H02", "rerate_id": "", "group": "Q2"},
        {"purpose": "human", "source": "govuk", "system": "D", "id": "id3", "S": "0.5", "item_id": "H03", "rerate_id": "R02", "group": "Q3"},
        {"purpose": "human", "source": "govuk", "system": "D", "id": "id4", "S": "0.6", "item_id": "H04", "rerate_id": "", "group": "Q4"},
    ])


def _scores_fixture(tmp_path):
    scores_dir = tmp_path / "scores"
    scores_dir.mkdir()
    pd.DataFrame([
        {"id": "id1", "rouge1": .2, "rouge2": .1, "rougeLsum": .6, "bertscore_f1": .4, "pred_words": 20, "ref_words": 22},
        {"id": "id2", "rouge1": .3, "rouge2": .2, "rougeLsum": .7, "bertscore_f1": .5, "pred_words": 21, "ref_words": 23},
    ]).to_csv(scores_dir / "cnn.csv", index=False)
    pd.DataFrame([
        {"id": "id3", "rouge1": .4, "rouge2": .3, "rougeLsum": .8, "bertscore_f1": .6, "pred_words": 22, "ref_words": 24},
        {"id": "id4", "rouge1": .5, "rouge2": .4, "rougeLsum": .9, "bertscore_f1": .7, "pred_words": 23, "ref_words": 25},
    ]).to_csv(scores_dir / "gov.csv", index=False)
    config = {"cnn_dm": {"scores": "cnn.csv"}, "govuk": {"scores": "gov.csv"}}
    return scores_dir, config


def test_read_rating_csv_rejects_missing_and_invalid_scores(tmp_path):
    p = tmp_path / "ratings.csv"
    pd.DataFrame({"item_id": ["H01"], "fluency_1_5": [6], "completeness_1_5": [3], "factuality_1_5": [4]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="số nguyên 1–5"):
        ah.read_rating_csv(p)


def test_read_rating_csv_rejects_blank_score(tmp_path):
    p = tmp_path / "ratings.csv"
    p.write_text("item_id,fluency_1_5,completeness_1_5,factuality_1_5\nH01,4,,5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="còn điểm trống"):
        ah.read_rating_csv(p)


def test_merge_human_with_scores_joins_using_blind_key(tmp_path):
    scores_dir, config = _scores_fixture(tmp_path)
    merged = ah.merge_human_with_scores(_human(), _key(), config, scores_dir)
    assert merged["item_id"].tolist() == ["H01", "H02", "H03", "H04"]
    assert merged["source"].tolist() == ["cnn_dm", "cnn_dm", "govuk", "govuk"]
    assert merged["S"].tolist() == pytest.approx([.3, .4, .5, .6])
    assert merged.loc[0, "bertscore_f1"] == pytest.approx(.4)


def test_merge_rejects_item_id_mismatch(tmp_path):
    scores_dir, config = _scores_fixture(tmp_path)
    with pytest.raises(ValueError, match="không khớp"):
        ah.merge_human_with_scores(_human(("H01", "H02", "H03")), _key(), config, scores_dir)


def test_bootstrap_mean_ci_is_deterministic():
    values = [1, 2, 3, 4, 5]
    assert ah.bootstrap_mean_ci(values, 300, 42) == ah.bootstrap_mean_ci(values, 300, 42)


def test_summarize_ratings_outputs_all_and_source_scopes():
    df = _human().assign(source=["cnn_dm", "cnn_dm", "govuk", "govuk"])
    result = ah.summarize_ratings(df, n_boot=100, seed=42)
    assert set(result["scope"]) == {"all", "cnn_dm", "govuk"}
    all_fluency = result[(result.scope == "all") & (result.criterion == "fluency_1_5")].iloc[0]
    assert all_fluency["mean"] == pytest.approx(3.5)
    assert all_fluency["ci95_low"] <= all_fluency["mean"] <= all_fluency["ci95_high"]


def test_correlations_detect_monotonic_relation():
    df = _human().assign(
        source="cnn_dm",
        S=[.1, .2, .3, .4],
        rouge1=[.1, .2, .3, .4],
        rouge2=[.1, .2, .3, .4],
        rougeLsum=[.1, .2, .3, .4],
        bertscore_f1=[.1, .2, .3, .4],
    )
    result = ah.correlate_human_and_automatic(df, n_boot=100, seed=42)
    row = result[(result.scope == "all") & (result.criterion == "fluency_1_5") & (result.metric == "S")].iloc[0]
    assert row["spearman"] == pytest.approx(-.4)  # fluency [4,3,5,2] is not monotonic
    row2 = result[(result.scope == "all") & (result.criterion == "completeness_1_5") & (result.metric == "S")].iloc[0]
    assert row2["spearman"] == pytest.approx(-0.4)


def test_completeness_correlation_comparison_is_paired_and_deterministic():
    df = _human().assign(
        source="cnn_dm",
        S=[.1, .2, .3, .4],
        rouge1=[.1, .2, .3, .4],
        rouge2=[.4, .3, .2, .1],
        rougeLsum=[.1, .2, .3, .4],
        bertscore_f1=[.1, .2, .3, .4],
    )
    result = ah.compare_completeness_correlations(df, n_boot=100, seed=42)
    all_row = result[result.scope == "all"].iloc[0]
    assert all_row["bertscore_spearman"] == pytest.approx(-0.4)
    assert all_row["rouge2_spearman"] == pytest.approx(0.4)
    assert all_row["difference"] == pytest.approx(-0.8)
    assert all_row["difference_ci95_low"] <= all_row["difference"] <= all_row["difference_ci95_high"]


def test_error_labels_allow_multiple_labels():
    df = _human().assign(source=["cnn_dm", "cnn_dm", "govuk", "govuk"])
    result = ah.summarize_error_labels(df)
    all_rows = result[result.scope == "all"].set_index("label")
    assert all_rows.loc["missing_information", "count"] == 2
    assert all_rows.loc["hallucination", "count"] == 1


def test_quadratic_weighted_kappa_identical_is_one():
    ratings = [1, 2, 3, 4, 5, 2, 4]
    assert ah.quadratic_weighted_kappa(ratings, ratings) == pytest.approx(1.0)


def test_inter_rater_uses_key_mapping_and_reports_agreement():
    first = _human(("H01", "H02", "H03", "H04"))
    key = _key()
    second_rater = pd.DataFrame({
        "item_id": ["R01", "R02"],
        "fluency_1_5": [4, 4], "completeness_1_5": [3, 4], "factuality_1_5": [5, 4],
        "error_type": ["", ""], "notes": ["", ""],
    })
    out = ah.analyze_inter_rater_agreement(first, second_rater, key)
    assert len(out) == 3
    assert set(out["n"]) == {2}
    assert "spearman_inter_rater" in out.columns
    assert "spearman_test_retest" not in out.columns
    fluency = out.loc[out.criterion == "fluency_1_5"].iloc[0]
    assert fluency["exact_agreement_pct"] == 50
    assert fluency["rater_1_mean"] == pytest.approx(4.5)
    assert fluency["rater_2_mean"] == pytest.approx(4.0)
    assert fluency["mean_difference_rater_2_minus_rater_1"] == pytest.approx(-0.5)
    assert fluency["mean_absolute_difference"] == pytest.approx(0.5)


def test_inter_rater_rejects_mismatched_article_or_summary():
    first = _human(("H01", "H03")).assign(
        article=["article one", "article three"],
        summary=["summary one", "summary three"],
    )
    second_rater = pd.DataFrame({
        "item_id": ["R01", "R02"],
        "article": ["WRONG ARTICLE", "article three"],
        "summary": ["summary one", "summary three"],
        "fluency_1_5": [4, 4],
        "completeness_1_5": [3, 4],
        "factuality_1_5": [5, 4],
    })
    with pytest.raises(ValueError, match="article không khớp"):
        ah.analyze_inter_rater_agreement(first, second_rater, _key())


def test_inter_rater_rejects_missing_or_extra_second_rater_ids():
    first = _human(("H01", "H02", "H03", "H04"))
    second_rater = pd.DataFrame({
        "item_id": ["R01"],
        "fluency_1_5": [4], "completeness_1_5": [3], "factuality_1_5": [5],
    })
    with pytest.raises(ValueError, match="không khớp các mã R"):
        ah.analyze_inter_rater_agreement(first, second_rater, _key())


def test_run_analysis_writes_outputs(tmp_path):
    scores_dir, config = _scores_fixture(tmp_path)
    key = _key()
    key_path = tmp_path / "samples_key.csv"
    key.to_csv(key_path, index=False)
    config_path = tmp_path / "samples.yaml"
    import yaml
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    human_path = tmp_path / "human.csv"
    _human().to_csv(human_path, index=False)
    second_rater_path = tmp_path / "second_rater.csv"
    pd.DataFrame({
        "item_id": ["R01", "R02"],
        "fluency_1_5": [4, 4],
        "completeness_1_5": [3, 4],
        "factuality_1_5": [5, 4],
    }).to_csv(second_rater_path, index=False)
    out_dir = tmp_path / "out"
    result = ah.run_analysis(
        human_path, key_path, config_path, scores_dir, out_dir,
        rerate_csv=second_rater_path, n_boot=100,
    )
    assert len(result["merged"]) == 4
    assert result["reliability"] is not None
    for name in [
        "human_scores_joined.csv", "human_summary.csv", "human_correlations.csv",
        "human_error_labels.csv", "human_correlation_comparisons.csv",
        "human_inter_rater_agreement.csv", "human_analysis.md",
    ]:
        assert (out_dir / name).exists()
    report = (out_dir / "human_analysis.md").read_text(encoding="utf-8")
    assert "inter-rater agreement" in report
    assert "test–retest" in report
    assert "không được xem điểm hoặc nhận xét lượt một" in report
