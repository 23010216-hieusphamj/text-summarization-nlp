import pandas as pd

from src.decode_utils import banned_tokens, repeated_trigram_rate, sentences_text, upsert_rows


def test_banned_tokens_trigram():
    assert banned_tokens([1, 2, 3, 1, 2], 3) == {3}
    assert banned_tokens([1, 2, 3, 4], 3) == set()
    assert banned_tokens([1, 2], 3) == set()


def test_banned_tokens_other_n():
    assert banned_tokens([5, 6, 5], 2) == {6}
    assert banned_tokens([5, 6, 7], 1) == {5, 6, 7}
    assert banned_tokens([5, 6, 5], 0) == set()


def test_banned_tokens_multiple_continuations():
    # (1,2) từng được theo sau bởi 3 và 4 -> cả hai bị cấm
    assert banned_tokens([1, 2, 3, 1, 2, 4, 1, 2], 3) == {3, 4}


def test_sentences_text_joins_with_newline():
    assert sentences_text("First one here. Second one there.") == "First one here.\nSecond one there."


def test_repeated_trigram_rate():
    preds = ["the cat sat on the mat", "a b c a b c a b c", "one two"]
    assert abs(repeated_trigram_rate(preds) - 100 / 3) < 1e-9


def test_upsert_rows_replaces_same_key_and_keeps_others(tmp_path):
    p = tmp_path / "t.csv"
    upsert_rows(p, pd.DataFrame([{"source": "a", "system": "x", "v": 1}, {"source": "a", "system": "y", "v": 2}]))
    out = upsert_rows(p, pd.DataFrame([{"source": "a", "system": "x", "v": 9, "extra": 5}]))
    d = {r.system: r.v for r in out.itertuples()}
    assert d == {"y": 2, "x": 9} and len(out) == 2


def test_table_path_and_pred_path_keep_test_names_and_suffix_other_splits(tmp_path):
    from src.decode_utils import pred_path, table_path
    assert table_path(tmp_path, "baselines", "cnn_dm", "test") == tmp_path / "baselines_cnn_dm.csv"
    assert table_path(tmp_path, "baselines", "cnn_dm", "validation") == tmp_path / "baselines_cnn_dm_validation.csv"
    assert pred_path(tmp_path, "govuk", "lead-1", "test") == tmp_path / "govuk_lead-1.jsonl"
    assert pred_path(tmp_path, "govuk", "lead-1", "validation") == tmp_path / "govuk_lead-1_validation.jsonl"
    assert pred_path(tmp_path, "govuk", "lead-1", "test_limit50") != pred_path(tmp_path, "govuk", "lead-1", "test")


def test_upsert_rows_with_split_key_does_not_mix_splits(tmp_path):
    p = tmp_path / "t.csv"
    upsert_rows(p, pd.DataFrame([{"source": "a", "system": "x", "split": "test", "v": 1}]),
                keys=("source", "system", "split"))
    out = upsert_rows(p, pd.DataFrame([{"source": "a", "system": "x", "split": "validation", "v": 7}]),
                      keys=("source", "system", "split"))
    assert len(out) == 2 and set(out["split"]) == {"test", "validation"}
    out = upsert_rows(p, pd.DataFrame([{"source": "a", "system": "x", "split": "test", "v": 5}]),
                      keys=("source", "system", "split"))
    assert {r.split: r.v for r in out.itertuples()} == {"test": 5, "validation": 7}


def test_upsert_rows_old_defaults_fill_missing_key_column_and_missing_key_raises(tmp_path):
    p = tmp_path / "t.csv"
    pd.DataFrame([{"source": "a", "system": "x", "v": 1}]).to_csv(p, index=False)       # bảng cũ chưa có cột split
    new = pd.DataFrame([{"source": "a", "system": "x", "split": "validation", "v": 2}])
    out = upsert_rows(p, new, keys=("source", "system", "split"), old_defaults={"split": "test"})
    assert sorted(out["split"]) == ["test", "validation"] and len(out) == 2
    pd.DataFrame([{"source": "a", "system": "x", "v": 1}]).to_csv(p, index=False)
    try:
        upsert_rows(p, new, keys=("source", "system", "split"))
    except ValueError as e:
        assert "split" in str(e)
    else:
        raise AssertionError("phải báo lỗi khi bảng cũ thiếu cột khóa")
