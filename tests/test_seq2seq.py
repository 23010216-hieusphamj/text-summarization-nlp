import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F

from src.models.seq2seq import Seq2Seq
from src.preprocessing.spm_tokenizer import PAD_ID


def small(**kw):
    cfg = dict(vocab_size=50, emb_dim=16, hidden=16, enc_layers=2, dropout=0.0)
    cfg.update(kw)
    return Seq2Seq(**cfg)


def test_forward_shape():
    m = small()
    src = torch.randint(4, 50, (3, 7))
    lens = torch.tensor([7, 5, 3])
    tgt_in = torch.randint(4, 50, (3, 6))
    assert m(src, lens, tgt_in).shape == (3, 6, 50)


def test_attention_gives_zero_weight_to_padding():
    m = small().eval()
    src = torch.randint(4, 50, (2, 6))
    lens = torch.tensor([6, 3])
    enc_out, h0, proj, mask = m.encode(src, lens)
    assert mask.tolist() == [[True] * 6, [True] * 3 + [False] * 3]
    _, _, attn = m.decoder.step(torch.tensor([2, 2]), h0, enc_out, proj, mask)
    assert attn.shape == (2, 6)
    assert torch.allclose(attn.sum(-1), torch.ones(2), atol=1e-5)
    assert attn[1, 3:].abs().max().item() == 0.0


def test_output_does_not_depend_on_padding_content():
    m = small().eval()
    lens = torch.tensor([6, 3])
    a = torch.randint(4, 50, (2, 6))
    b = a.clone()
    b[1, 3:] = torch.randint(4, 50, (3,))      # đổi nội dung vùng đệm của mẫu 2
    tgt_in = torch.randint(4, 50, (2, 4))
    with torch.no_grad():
        assert torch.allclose(m(a, lens, tgt_in), m(b, lens, tgt_in), atol=1e-5)


def test_can_overfit_tiny_batch():
    torch.manual_seed(0)
    m = small(vocab_size=30, emb_dim=32, hidden=32)
    src = torch.randint(4, 30, (4, 8))
    lens = torch.full((4,), 8)
    tgt = torch.randint(4, 30, (4, 6))
    tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
    opt = torch.optim.Adam(m.parameters(), lr=5e-3)
    first = last = None
    for _ in range(150):
        loss = F.cross_entropy(m(src, lens, tgt_in).reshape(-1, 30), tgt_out.reshape(-1), ignore_index=PAD_ID)
        opt.zero_grad(); loss.backward(); opt.step()
        first = first if first is not None else loss.item()
        last = loss.item()
    assert last < 0.5 * first


def test_param_count_matches_d9_estimate():
    n = Seq2Seq(vocab_size=16000, emb_dim=256, hidden=256, enc_layers=2).num_parameters()
    assert 7.0e6 < n < 8.0e6
