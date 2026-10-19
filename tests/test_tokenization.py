import numpy as np
import pytest

pytest.importorskip("transformers")
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

from src.preprocessing import tokenization as tk


@pytest.fixture(scope="module")
def tok():
    """Tokenizer nhỏ dựng offline nhưng dùng đúng API của tokenizer HuggingFace."""
    t = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    t.pre_tokenizer = pre_tokenizers.Whitespace()
    corpus = ["the cat sat on the mat", "summarize: the government announced new funding"] * 20
    t.train_from_iterator(corpus, trainers.WordLevelTrainer(special_tokens=["[UNK]", "[PAD]"]))
    return PreTrainedTokenizerFast(tokenizer_object=t, unk_token="[UNK]", pad_token="[PAD]")


def test_prepare_texts_prefix_and_join():
    i, t = tk.prepare_texts(["a b"], ["x y.\nz w."], prefix="summarize: ", target_join=" ")
    assert i == ["summarize: a b"] and t == ["x y. z w."]


def test_tokenize_batch_truncates_inputs_and_labels_separately(tok):
    batch = {"article": ["the cat sat on the mat " * 10], "highlights": ["the cat sat on the mat " * 10]}
    out = tk.tokenize_batch(batch, tok, max_input=12, max_target=5)
    assert len(out["input_ids"][0]) == 12 and len(out["labels"][0]) == 5
    assert "attention_mask" in out


def test_token_lengths_and_truncation_report(tok):
    texts = ["the cat " * 3, "the cat " * 10, "the cat " * 30]
    L = tk.token_lengths(texts, tok)
    assert list(L) == [6, 20, 60]
    rep = {r["max_len"]: r for r in tk.truncation_report(L, [10, 40])}
    assert rep[10]["pct_truncated"] == pytest.approx(200 / 3)
    assert rep[40]["pct_truncated"] == pytest.approx(100 / 3)
    assert rep[40]["pct_tokens_lost"] == pytest.approx(20 / 86 * 100)


def test_build_tokenized_dataset(tok):
    recs = [{"article": "the cat sat on the mat " * 5, "highlights": "the cat sat"}] * 4
    ds = tk.build_tokenized_dataset(recs, tok, max_input=16, max_target=8)
    assert len(ds) == 4 and set(ds.column_names) == {"input_ids", "attention_mask", "labels"}
    assert len(ds[0]["input_ids"]) == 16


def test_collator_shapes_if_torch_available(tok):
    pytest.importorskip("torch")
    from transformers import DataCollatorForSeq2Seq
    recs = [{"article": "the cat sat " * n, "highlights": "the cat " * m} for n, m in [(3, 1), (8, 2)]]
    ds = tk.build_tokenized_dataset(recs, tok, 64, 16)
    batch = DataCollatorForSeq2Seq(tok, label_pad_token_id=-100, return_tensors="pt")([ds[0], ds[1]])
    assert batch["input_ids"].shape[0] == 2 and (batch["labels"] == -100).any()
