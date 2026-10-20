import pytest

from src.data import seq2seq_data as sd
from src.preprocessing.spm_tokenizer import BOS_ID, EOS_ID, PAD_ID


class FakeSP:
    """Mỗi từ là một token có id 10 + độ dài từ."""
    def encode(self, texts, out_type=int):
        return [[10 + len(w) for w in t.split()] for t in texts]


def test_encode_source_truncates_and_ends_with_eos():
    assert sd.encode_source([5] * 10, 4) == [5, 5, 5, EOS_ID]
    assert sd.encode_source([5, 6], 4) == [5, 6, EOS_ID]


def test_encode_target_wraps_with_bos_eos_and_respects_max():
    assert sd.encode_target([5] * 10, 5) == [BOS_ID, 5, 5, 5, EOS_ID]
    assert sd.encode_target([5], 5) == [BOS_ID, 5, EOS_ID]


def test_build_examples_joins_summary_sentences():
    recs = [{"id": "a", "article": "aa bb cc", "highlights": "xx.\nyy."}]
    ex = sd.build_examples(recs, FakeSP(), max_input=512, max_target=128)
    assert ex[0]["id"] == "a"
    assert ex[0]["src"][-1] == EOS_ID and len(ex[0]["src"]) == 4
    assert ex[0]["tgt"][0] == BOS_ID and ex[0]["tgt"][-1] == EOS_ID and len(ex[0]["tgt"]) == 4


def test_collate_shapes_and_shift():
    torch = pytest.importorskip("torch")
    batch = [{"id": "1", "src": [5, 6, EOS_ID], "tgt": [BOS_ID, 7, 8, EOS_ID]},
             {"id": "2", "src": [5, EOS_ID], "tgt": [BOS_ID, 9, EOS_ID]}]
    b = sd.collate(batch)
    assert b["src"].shape == (2, 3) and b["src_lens"].tolist() == [3, 2]
    assert b["tgt_in"].shape == (2, 3) and b["tgt_out"].shape == (2, 3)
    assert b["src"][1].tolist() == [5, EOS_ID, PAD_ID]
    assert b["tgt_in"][0].tolist() == [BOS_ID, 7, 8] and b["tgt_out"][0].tolist() == [7, 8, EOS_ID]
    assert b["tgt_out"][1].tolist() == [9, EOS_ID, PAD_ID]


def test_make_loader_is_reproducible_with_seed():
    pytest.importorskip("torch")
    ex = [{"id": str(i), "src": [5, EOS_ID], "tgt": [BOS_ID, 6, EOS_ID]} for i in range(20)]
    a = [b["id"] for b in sd.make_loader(ex, 4, shuffle=True, seed=1)]
    b = [b["id"] for b in sd.make_loader(ex, 4, shuffle=True, seed=1)]
    assert a == b
