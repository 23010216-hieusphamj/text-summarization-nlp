import math

from src.data import eda_utils as u


def test_split_sentences():
    s = "Hello world. This is a test! Is it working? Yes."
    assert len(u.split_sentences(s)) == 4


def test_word_count_and_compression():
    assert u.word_count("One two three") == 3
    assert u.compression_ratio("a b c d e f", "a b") == 3.0
    assert math.isnan(u.compression_ratio("a b", ""))


def test_novel_ngram_ratio():
    art = "the cat sat on the mat"
    assert u.novel_ngram_ratio(art, "the cat sat", 1) == 0.0
    assert u.novel_ngram_ratio(art, "a dog ran", 1) == 1.0
    assert u.novel_ngram_ratio(art, "the cat ran", 2) == 0.5


def test_best_match_positions_lead_bias():
    art = "Police arrested a man in Paris. The weather was sunny. Shops closed early."
    hl = "Man arrested in Paris by police"
    assert u.best_match_positions(art, hl) == [0]


def test_boilerplate_flags():
    f = u.boilerplate_flags("By . Daily Mail Reporter . PUBLISHED: 10:00 EST")
    assert f["dailymail_by"] and f["published_updated"] and not f["cnn_prefix"]


def test_describe_ignores_nan():
    d = u.describe([1, 2, 3, float("nan")])
    assert d["mean"] == 2 and d["max"] == 3


def test_max_sentence_overlap():
    art = "Police arrested a man in Paris on Monday. The weather was sunny and warm."
    assert u.max_sentence_overlap(art, "Police arrested a man in Paris") == 1.0
    assert u.max_sentence_overlap(art, "Shops closed early") == 0.0
    assert u.max_sentence_overlap("", "anything") == 0.0
