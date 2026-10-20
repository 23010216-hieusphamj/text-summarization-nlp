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
