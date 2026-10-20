import numpy as np
import pytest

from src.preprocessing import spm_tokenizer as st


class FakeSP:
    """Bộ mã hóa giả: mỗi từ cách nhau bằng khoảng trắng là một token; 'zzz' là <unk>."""
    def encode(self, texts, out_type=int):
        return [[st.UNK_ID if w == "zzz" else 10 + len(w) for w in t.split()] for t in texts]


def test_target_text_joins_summary_sentences():
    assert st.target_text("a b.\nc d.") == "a b. c d."


def test_build_training_corpus_one_sentence_per_line(tmp_path):
    recs = [{"article": "First sentence here. Second one follows.", "highlights": "Short one.\nAnother one."}]
    n = st.build_training_corpus(recs, tmp_path / "c.txt")
    lines = (tmp_path / "c.txt").read_text(encoding="utf-8").splitlines()
    assert n == len(lines) == 4
    assert all("\n" not in l and l.strip() for l in lines)


def test_spm_lengths_adds_special_tokens():
    out = st.spm_lengths(["a b c", "a"], FakeSP(), extra=2)
    assert out.tolist() == [5, 3]


def test_unk_rate():
    assert st.unk_rate(["a zzz", "zzz zzz"], FakeSP()) == pytest.approx(3 / 4)


def test_measure_truncation_numbers():
    recs = [{"article": "w " * 10, "highlights": "x y z"}, {"article": "w " * 3, "highlights": "x"}]
    rows, summ = st.measure(recs, FakeSP(), input_thresholds=[5], target_thresholds=[3])
    inp = next(r for r in rows if r["field"] == "input")
    tgt = next(r for r in rows if r["field"] == "target")
    # độ dài input = 11 và 4 (đã cộng </s>): chỉ bài 1 vượt 5, mất 6/15 token
    assert inp["pct_truncated"] == pytest.approx(50.0)
    assert inp["pct_tokens_lost"] == pytest.approx(6 / 15 * 100)
    # độ dài target = 5 và 3 (đã cộng <s>, </s>): chỉ mẫu 1 vượt 3, mất 2/8 token
    assert tgt["pct_truncated"] == pytest.approx(50.0)
    assert tgt["pct_tokens_lost"] == pytest.approx(2 / 8 * 100)
    assert summ["n"] == 2 and summ["unk_rate_input"] == 0.0


def test_train_and_roundtrip(tmp_path):
    pytest.importorskip("sentencepiece")
    sents = [f"the minister said report number {i} shows funding for school {i % 7} will rise" for i in range(300)]
    corpus = tmp_path / "c.txt"
    corpus.write_text("\n".join(sents), encoding="utf-8")
    model = st.train_spm(corpus, tmp_path, "t", vocab_size=60, input_sentence_size=1000)
    sp = st.load_spm(model)
    assert (sp.pad_id(), sp.unk_id(), sp.bos_id(), sp.eos_id()) == (0, 1, 2, 3)
    ids = sp.encode("the minister said funding will rise", out_type=int)
    assert sp.decode(ids) == "the minister said funding will rise"
