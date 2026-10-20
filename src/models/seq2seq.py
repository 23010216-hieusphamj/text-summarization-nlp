"""Seq2Seq BiGRU + attention cộng (Bahdanau), huấn luyện từ đầu (GĐ5, D9).

Luồng tensor (B = batch, S = độ dài nguồn, T = độ dài đích, E = emb_dim, H = hidden, V = vocab):
  src (B,S) -> embedding (B,S,E) -> BiGRU 2 tầng -> enc_out (B,S,2H); trạng thái cuối -> h0 (1,B,H)
  mỗi bước decoder: attention(h_{t-1}, enc_out) -> ctx (B,2H); GRU([emb(y_{t-1}), ctx]) -> s_t (B,H)
                    feat = tanh(W[s_t, ctx, emb]) (B,E); logits = feat @ embedding^T + bias (B,V)
Embedding dùng chung cho encoder, decoder và lớp đầu ra (chia sẻ trọng số).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.preprocessing.spm_tokenizer import PAD_ID


class Encoder(nn.Module):
    def __init__(self, embedding: nn.Embedding, hidden: int, layers: int, dropout: float):
        super().__init__()
        self.embedding = embedding
        self.drop = nn.Dropout(dropout)
        self.rnn = nn.GRU(embedding.embedding_dim, hidden, num_layers=layers, batch_first=True,
                          bidirectional=True, dropout=dropout if layers > 1 else 0.0)
        self.bridge = nn.Linear(2 * hidden, hidden)

    def forward(self, src: torch.Tensor, src_lens: torch.Tensor):
        x = self.drop(self.embedding(src))                                           # (B,S,E)
        packed = nn.utils.rnn.pack_padded_sequence(x, src_lens.cpu(), batch_first=True, enforce_sorted=False)
        out, h = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=src.size(1))  # (B,S,2H)
        h_last = torch.cat([h[-2], h[-1]], dim=-1)                                   # tầng cuối: thuận + ngược (B,2H)
        h0 = torch.tanh(self.bridge(h_last)).unsqueeze(0)                            # (1,B,H)
        return out, h0


class BahdanauAttention(nn.Module):
    def __init__(self, enc_dim: int, dec_dim: int, attn_dim: int):
        super().__init__()
        self.w_enc = nn.Linear(enc_dim, attn_dim, bias=False)
        self.w_dec = nn.Linear(dec_dim, attn_dim, bias=True)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def project(self, enc_out: torch.Tensor) -> torch.Tensor:
        return self.w_enc(enc_out)                                                   # (B,S,A), tính một lần cho cả chuỗi

    def forward(self, dec_h, enc_proj, enc_out, mask):
        # dec_h (B,H); enc_proj (B,S,A); enc_out (B,S,2H); mask (B,S) True = token thật
        e = self.v(torch.tanh(enc_proj + self.w_dec(dec_h).unsqueeze(1))).squeeze(-1)   # (B,S)
        e = e.masked_fill(~mask, float("-inf"))
        a = torch.softmax(e, dim=-1)                                                 # (B,S), PAD có trọng số 0
        ctx = torch.bmm(a.unsqueeze(1), enc_out).squeeze(1)                          # (B,2H)
        return ctx, a


class Decoder(nn.Module):
    def __init__(self, embedding: nn.Embedding, enc_dim: int, hidden: int, attn_dim: int, dropout: float):
        super().__init__()
        emb_dim = embedding.embedding_dim
        self.embedding = embedding
        self.drop = nn.Dropout(dropout)
        self.attn = BahdanauAttention(enc_dim, hidden, attn_dim)
        self.rnn = nn.GRU(emb_dim + enc_dim, hidden, batch_first=True)
        self.pre_out = nn.Linear(hidden + enc_dim + emb_dim, emb_dim)
        self.out_bias = nn.Parameter(torch.zeros(embedding.num_embeddings))

    def step(self, tok, h, enc_out, enc_proj, mask):
        """Một bước giải mã. tok (B,), h (1,B,H) -> logits (B,V), h mới (1,B,H), attention (B,S)."""
        emb = self.drop(self.embedding(tok))                                         # (B,E)
        ctx, attn = self.attn(h[-1], enc_proj, enc_out, mask)                        # (B,2H), (B,S)
        out, h = self.rnn(torch.cat([emb, ctx], dim=-1).unsqueeze(1), h)             # out (B,1,H)
        s = out.squeeze(1)                                                           # (B,H)
        feat = torch.tanh(self.pre_out(torch.cat([s, ctx, emb], dim=-1)))            # (B,E)
        logits = F.linear(self.drop(feat), self.embedding.weight, self.out_bias)     # (B,V)
        return logits, h, attn


class Seq2Seq(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 256, hidden: int = 256, enc_layers: int = 2,
                 attn_dim: int | None = None, dropout: float = 0.3, pad_id: int = PAD_ID):
        super().__init__()
        self.pad_id = pad_id
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=emb_dim ** -0.5)
        with torch.no_grad():
            self.embedding.weight[pad_id].zero_()
        self.encoder = Encoder(self.embedding, hidden, enc_layers, dropout)
        self.decoder = Decoder(self.embedding, 2 * hidden, hidden, attn_dim or hidden, dropout)

    def encode(self, src: torch.Tensor, src_lens: torch.Tensor):
        """Trả về (enc_out, h0, enc_proj, mask); dùng lại ở bước giải mã (greedy/beam)."""
        enc_out, h0 = self.encoder(src, src_lens)
        mask = torch.arange(src.size(1), device=src.device).unsqueeze(0) < src_lens.to(src.device).unsqueeze(1)
        return enc_out, h0, self.decoder.attn.project(enc_out), mask

    def forward(self, src, src_lens, tgt_in):
        """Teacher forcing. tgt_in (B,T) -> logits (B,T,V)."""
        enc_out, h, enc_proj, mask = self.encode(src, src_lens)
        logits = []
        for t in range(tgt_in.size(1)):
            lg, h, _ = self.decoder.step(tgt_in[:, t], h, enc_out, enc_proj, mask)
            logits.append(lg)
        return torch.stack(logits, dim=1)

    def num_parameters(self) -> int:
        """Số tham số huấn luyện được (embedding dùng chung chỉ tính một lần)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
