import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F

from src.decode_seq2seq import beam_search
from src.models.seq2seq import Seq2Seq
from src.preprocessing.spm_tokenizer import BOS_ID, EOS_ID, PAD_ID, UNK_ID


def make(vocab=50):
    torch.manual_seed(0)
    return Seq2Seq(vocab, emb_dim=16, hidden=16, enc_layers=2, dropout=0.0).eval()


def src_batch(n=7, vocab=50):
    torch.manual_seed(1)
    return torch.randint(4, vocab, (1, n)), torch.tensor([n])


def manual_greedy(model, src, lens, max_len):
    enc_out, h, proj, mask = model.encode(src, lens)
    prev, out = torch.tensor([BOS_ID]), []
    for _ in range(max_len - 1):
        lg, h, _ = model.decoder.step(prev, h, enc_out, proj, mask)
        lg = F.log_softmax(lg.float(), -1)
        lg[:, [PAD_ID, BOS_ID, UNK_ID]] = float("-inf")
        tok = int(lg.argmax(-1))
        if tok == EOS_ID:
            break
        out.append(tok)
        prev = torch.tensor([tok])
    return out


def test_beam1_equals_manual_greedy():
    m, (src, lens) = make(), src_batch()
    got = beam_search(m, src, lens, num_beams=1, max_len=30, no_repeat_ngram=0)
    assert got == manual_greedy(m, src, lens, 30)


def test_output_has_no_special_tokens_and_respects_max_len():
    m, (src, lens) = make(), src_batch()
    for nb in (1, 4):
        out = beam_search(m, src, lens, num_beams=nb, max_len=20, no_repeat_ngram=0)
        assert len(out) <= 19
        assert not {PAD_ID, BOS_ID, UNK_ID, EOS_ID} & set(out)


def test_no_repeat_trigram_is_enforced():
    m, (src, lens) = make(), src_batch()
    out = beam_search(m, src, lens, num_beams=4, max_len=40, no_repeat_ngram=3)
    tri = list(zip(out, out[1:], out[2:]))
    assert len(tri) == len(set(tri))


def test_beam_search_is_deterministic():
    m, (src, lens) = make(), src_batch()
    a = beam_search(m, src, lens, num_beams=4, max_len=25, no_repeat_ngram=3)
    b = beam_search(m, src, lens, num_beams=4, max_len=25, no_repeat_ngram=3)
    assert a == b and all(isinstance(t, int) for t in a)
