"""Giải mã (greedy / beam search) bằng Seq2Seq đã huấn luyện, rồi đánh giá bằng đúng evaluate.py của GĐ4.

    # cấu hình chính của D9 (beam 4, chặn lặp trigram), có BERTScore
    python -m src.decode_seq2seq --data_dir <clean> --ckpt_dir <ckpt trên Drive>
    # ablation nhanh (không BERTScore)
    python -m src.decode_seq2seq ... --num_beams 1 --no_repeat_ngram 0 --no_bertscore       # greedy
    python -m src.decode_seq2seq ... --num_beams 4 --no_repeat_ngram 0 --no_bertscore       # beam không chặn lặp
    python -m src.decode_seq2seq ... --limit 50 --no_bertscore --tag smoke                   # chạy thử

Đầu ra: results/predictions/{source}_seq2seq-{tag}.jsonl, results/tables/seq2seq_{source}.csv (cùng cột với
baselines_{source}.csv + vài cột thêm). Cài đặt đánh giá lấy từ configs/baselines.yaml để giống hệt D5/D8.
Chặn lặp được áp dụng ở mức token SentencePiece (không phải từ).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
import yaml

from src.data.seq2seq_data import load_examples
from src.decode_utils import banned_tokens, repeated_trigram_rate, sentences_text, upsert_rows
from src.evaluate import evaluate_predictions
from src.models.seq2seq import Seq2Seq
from src.preprocessing.spm_tokenizer import BOS_ID, EOS_ID, PAD_ID, UNK_ID, load_spm, read_jsonl

NEG_INF = float("-inf")


@torch.no_grad()
def beam_search(model: Seq2Seq, src: torch.Tensor, src_len: torch.Tensor, num_beams: int = 4, max_len: int = 128,
                no_repeat_ngram: int = 3, length_penalty: float = 1.0, min_len: int = 0) -> list[int]:
    """Beam search cho MỘT bài (src (1,S), src_len (1,)). Trả về danh sách id token, không gồm <s> và </s>.

    num_beams = 1 chính là greedy. Số token sinh ra (kể cả </s>) tối đa max_len - 1, khớp độ dài đích khi huấn luyện
    (max_target gồm cả <s>). Không bao giờ sinh <pad>, <s>, <unk>. Dừng khi đã có num_beams giả thuyết kết thúc.
    """
    device, K, V = src.device, num_beams, model.embedding.num_embeddings
    enc_out, h, enc_proj, mask = model.encode(src, src_len)
    enc_out, enc_proj = enc_out.expand(K, -1, -1).contiguous(), enc_proj.expand(K, -1, -1).contiguous()
    mask, h = mask.expand(K, -1).contiguous(), h.expand(-1, K, -1).contiguous()

    seqs: list[list[int]] = [[] for _ in range(K)]
    score_list = [0.0] + [NEG_INF] * (K - 1)          # bước đầu chỉ beam 0 còn sống để các beam khác nhau
    prev = torch.full((K,), BOS_ID, dtype=torch.long, device=device)
    finished: list[tuple[float, list[int]]] = []

    for t in range(max_len - 1):
        logits, h, _ = model.decoder.step(prev, h, enc_out, enc_proj, mask)
        logp = F.log_softmax(logits.float(), dim=-1)                  # (K,V)
        logp[:, [PAD_ID, BOS_ID, UNK_ID]] = NEG_INF
        if t < min_len:
            logp[:, EOS_ID] = NEG_INF
        if no_repeat_ngram > 0:
            for k in range(K):
                if score_list[k] == NEG_INF:
                    continue
                ban = banned_tokens(seqs[k], no_repeat_ngram)
                if ban:
                    logp[k, list(ban)] = NEG_INF
        cand = (torch.tensor(score_list, device=device).unsqueeze(1) + logp).view(-1)
        top_s, top_i = cand.topk(2 * K)
        new_seqs, new_scores, new_src, new_prev = [], [], [], []
        for rank, (sc, ix) in enumerate(zip(top_s.tolist(), top_i.tolist())):
            if sc == NEG_INF:
                break
            b, tok = divmod(ix, V)
            if tok == EOS_ID:
                if rank < K:                                           # chỉ nhận </s> nếu nằm trong top K
                    finished.append((sc / ((len(seqs[b]) + 1) ** length_penalty), seqs[b]))
                continue
            new_seqs.append(seqs[b] + [tok]); new_scores.append(sc); new_src.append(b); new_prev.append(tok)
            if len(new_seqs) == K:
                break
        if not new_seqs or len(finished) >= K:
            break
        while len(new_seqs) < K:                                       # beam "chết" để giữ nguyên kích thước tensor
            new_seqs.append(list(new_seqs[0])); new_scores.append(NEG_INF)
            new_src.append(new_src[0]); new_prev.append(new_prev[0])
        h = h[:, torch.tensor(new_src, device=device)]
        prev = torch.tensor(new_prev, dtype=torch.long, device=device)
        seqs, score_list = new_seqs, new_scores

    if not finished:                                                   # chưa beam nào gặp </s>: lấy các beam đang sống
        for sc, s in zip(score_list, seqs):
            if sc != NEG_INF and s:
                finished.append((sc / (len(s) ** length_penalty), s))
    return max(finished, key=lambda x: x[0])[1] if finished else []


def decode_corpus(model, examples: list[dict], sp, device, log_every: int = 100, **kw) -> tuple[list[str], float]:
    """Giải mã toàn bộ ví dụ. Trả về (các tóm tắt đã tách câu bằng '\\n', ms trung bình mỗi bài)."""
    preds, t0 = [], time.perf_counter()
    for i, ex in enumerate(examples, 1):
        src = torch.tensor([ex["src"]], dtype=torch.long, device=device)
        ids = beam_search(model, src, torch.tensor([len(ex["src"])]), **kw)
        preds.append(sentences_text(sp.decode(ids)))
        if i % log_every == 0:
            print(f"  {i}/{len(examples)} bài, {(time.perf_counter() - t0) / i:.2f}s/bài")
    return preds, 1000 * (time.perf_counter() - t0) / max(len(examples), 1)


def auto_tag(num_beams: int, no_repeat_ngram: int) -> str:
    base = f"beam{num_beams}" if num_beams > 1 else "greedy"
    return base + (f"_nr{no_repeat_ngram}" if no_repeat_ngram > 0 else "")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/seq2seq.yaml")
    ap.add_argument("--eval_config", default="configs/baselines.yaml", help="lấy mục evaluate để giống hệt GĐ4")
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--ckpt_dir", default=None)
    ap.add_argument("--run_name", default=None)
    ap.add_argument("--ckpt", default=None, help="đường dẫn trực tiếp tới best.pt (ghi đè)")
    ap.add_argument("--sources", nargs="+", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--num_beams", type=int, default=None)
    ap.add_argument("--no_repeat_ngram", type=int, default=None)
    ap.add_argument("--length_penalty", type=float, default=None)
    ap.add_argument("--max_len", type=int, default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no_bertscore", action="store_true")
    a = ap.parse_args(argv)

    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    ev = yaml.safe_load(open(a.eval_config, encoding="utf-8"))["evaluate"]
    seed = yaml.safe_load(open(a.eval_config, encoding="utf-8")).get("seed", 42)
    dc = cfg["decode"]
    nb = a.num_beams if a.num_beams is not None else dc["num_beams"]
    nr = a.no_repeat_ngram if a.no_repeat_ngram is not None else dc["no_repeat_ngram"]
    lp = a.length_penalty if a.length_penalty is not None else dc.get("length_penalty", 1.0)
    ml = a.max_len or dc["max_len"]
    tag = a.tag or auto_tag(nb, nr)
    run = a.run_name or f"seq2seq_in{cfg['max_input']}"
    data_dir = Path(a.data_dir or cfg["data_dir"])
    sources = a.sources or cfg["eval_sources"]
    ckpt_path = Path(a.ckpt) if a.ckpt else Path(a.ckpt_dir or cfg["output"]["ckpt_dir"]) / run / "best.pt"
    tables, preds_dir = Path(cfg["output"]["tables_dir"]), Path(cfg["output"]["predictions_dir"])
    preds_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(ckpt_path, map_location=device)
    model = Seq2Seq(ck["vocab_size"], **ck["model_cfg"]).to(device)
    model.load_state_dict(ck["model"]); model.eval()
    sp = load_spm(cfg["spm_model"])
    system = f"seq2seq-{tag}"
    print(f"{system}: checkpoint epoch {ck['epoch']} (val loss {ck['val_loss']:.4f}), beams={nb}, "
          f"no_repeat_ngram={nr}, length_penalty={lp}, max_len={ml}, thiết bị {device}")

    for source in sources:
        records = read_jsonl(data_dir / f"{source}_{a.split}.jsonl")[: a.limit]
        examples = load_examples(data_dir, source, a.split, sp, cfg["max_input"], cfg["max_target"], limit=a.limit)
        refs = [r["highlights"] for r in records]
        print(f"\n=== {source}: {len(records)} mẫu ===")
        preds, ms = decode_corpus(model, examples, sp, device, num_beams=nb, max_len=ml, no_repeat_ngram=nr,
                                  length_penalty=lp)
        with open(preds_dir / f"{source}_{system}.jsonl", "w", encoding="utf-8") as f:
            for r, p in zip(records, preds):
                f.write(json.dumps({"id": r["id"], "prediction": p}, ensure_ascii=False) + "\n")
        row = evaluate_predictions(
            preds, refs, use_stemmer=ev["rouge_stemmer"], bertscore=ev["bertscore"] and not a.no_bertscore,
            bertscore_model=ev["bertscore_model"], bertscore_rescale=ev["bertscore_rescale"],
            bertscore_batch_size=ev["bertscore_batch_size"], n_boot=ev["bootstrap_samples"], seed=seed)
        row.update(source=source, system=system, split=a.split, num_beams=nb, no_repeat_ngram=nr, length_penalty=lp,
                   max_len=ml, ms_per_doc=ms, rep_trigram_pct=repeated_trigram_rate(preds),
                   n_params=sum(p.numel() for p in model.parameters()), ckpt_epoch=int(ck["epoch"]))
        first = ["source", "system", "n"]
        df = pd.DataFrame([row]); df = df[first + [c for c in df.columns if c not in first]]
        upsert_rows(tables / f"seq2seq_{source}.csv", df)
        print(f"{system:20s} R1={row['rouge1']:.4f} R2={row['rouge2']:.4f} RLsum={row['rougeLsum']:.4f} "
              f"BS={row.get('bertscore_f1', float('nan')):.4f} len={row['avg_pred_words']:.1f} "
              f"(ref {row['avg_ref_words']:.1f}) lặp trigram {row['rep_trigram_pct']:.1f}% {ms:.0f} ms/bài")
        for r, p in list(zip(records, preds))[:2]:
            print(f"\n[{r['id'][:8]}] THAM CHIẾU: {r['highlights'][:200]!r}\n           DỰ ĐOÁN : {p[:200]!r}")


if __name__ == "__main__":
    main()
