"""SentencePiece BPE cho Seq2Seq từ đầu (GĐ5, D9): huấn luyện từ vựng và đo tỉ lệ cắt token.

    python -m src.preprocessing.spm_tokenizer --config configs/spm.yaml

Từ vựng chỉ học trên train CNN/DM đã làm sạch. Sau đó đo độ dài token (bài và tóm tắt) trên mọi tập
để chốt max_input / max_target, kết quả ở results/tables/spm_truncation_report.csv và spm_summary.csv.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.eda_utils import split_sentences, summary_sentences
from src.preprocessing.tokenization import truncation_report

PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3
EXTRA_INPUT = 1    # </s> ở cuối bài
EXTRA_TARGET = 2   # <s> đầu và </s> cuối tóm tắt


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def target_text(highlights: str, target_join: str = " ") -> str:
    """Giống tokenization.prepare_texts: nối các câu tóm tắt (tách bằng xuống dòng) thành một chuỗi."""
    return highlights.replace("\n", target_join)


def build_training_corpus(records: list[dict], path: Path) -> int:
    """Ghi mỗi câu một dòng (bài và tóm tắt). Không ghi nguyên bài vì SentencePiece bỏ qua dòng quá dài
    (mặc định max_sentence_length = 4192 byte). Trả về số dòng đã ghi."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            sents = split_sentences(r["article"]) + summary_sentences(r["highlights"])
            for s in sents:
                s = " ".join(s.split())
                if s:
                    f.write(s + "\n")
                    n += 1
    return n


def train_spm(corpus_path: Path, model_dir: Path, model_prefix: str, vocab_size: int = 16000,
              model_type: str = "bpe", character_coverage: float = 0.9995,
              input_sentence_size: int = 1_000_000, seed: int = 42, hard_vocab_limit: bool = False) -> Path:
    import sentencepiece as spm
    model_dir.mkdir(parents=True, exist_ok=True)
    prefix = str(model_dir / model_prefix)
    spm.SentencePieceTrainer.train(
        input=str(corpus_path), model_prefix=prefix, vocab_size=vocab_size, model_type=model_type,
        character_coverage=character_coverage, input_sentence_size=input_sentence_size,
        shuffle_input_sentence=True,
        pad_id=PAD_ID, unk_id=UNK_ID, bos_id=BOS_ID, eos_id=EOS_ID,
        pad_piece="<pad>", unk_piece="<unk>", bos_piece="<s>", eos_piece="</s>",
        hard_vocab_limit=hard_vocab_limit, num_threads=4)
    return Path(prefix + ".model")


def load_spm(model_path: str | Path):
    import sentencepiece as spm
    return spm.SentencePieceProcessor(model_file=str(model_path))


def spm_lengths(texts: list[str], sp, extra: int = 0, batch_size: int = 1000) -> np.ndarray:
    """Số token của từng văn bản khi KHÔNG cắt, cộng `extra` token đặc biệt (<s>, </s>)."""
    out = []
    for i in range(0, len(texts), batch_size):
        out.extend(len(ids) + extra for ids in sp.encode(texts[i:i + batch_size], out_type=int))
    return np.asarray(out)


def unk_rate(texts: list[str], sp, batch_size: int = 1000) -> float:
    """Tỉ lệ token <unk> trên tổng số token."""
    unk = tot = 0
    for i in range(0, len(texts), batch_size):
        for ids in sp.encode(texts[i:i + batch_size], out_type=int):
            tot += len(ids)
            unk += sum(1 for t in ids if t == UNK_ID)
    return unk / tot if tot else 0.0


def measure(records: list[dict], sp, input_thresholds: list[int], target_thresholds: list[int]):
    """Trả về (các dòng báo cáo cắt token, thống kê tóm tắt) cho một tập."""
    from src.data.eda_utils import word_count
    arts = [r["article"] for r in records]
    tgts = [target_text(r["highlights"]) for r in records]
    li = spm_lengths(arts, sp, EXTRA_INPUT)
    lt = spm_lengths(tgts, sp, EXTRA_TARGET)
    w_art = sum(word_count(a) for a in arts)
    tok_per_word = float(li.sum() / w_art) if w_art else float("nan")
    rows = []
    for field, lens, ths in (("input", li, input_thresholds), ("target", lt, target_thresholds)):
        p50, p95, p99 = (float(np.percentile(lens, q)) for q in (50, 95, 99))
        for r in truncation_report(lens, ths):
            row = {"field": field, "p50": p50, "p95": p95, "p99": p99, **r}
            if field == "input":
                row["approx_words_kept"] = r["max_len"] / tok_per_word
            rows.append(row)
    summary = {"n": len(records), "tokens_per_word_input": tok_per_word,
               "unk_rate_input": unk_rate(arts, sp), "unk_rate_target": unk_rate(tgts, sp)}
    return rows, summary


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/spm.yaml")
    ap.add_argument("--data_dir", default=None, help="ghi đè data_dir trong config (vd. thư mục trên Google Drive)")
    ap.add_argument("--skip_train", action="store_true", help="dùng model đã có trong model_dir, chỉ đo lại")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    data_dir = Path(args.data_dir or cfg["data_dir"])
    model_dir = Path(cfg["model_dir"])
    model_path = model_dir / (cfg["model_prefix"] + ".model")

    if not args.skip_train:
        train_recs = read_jsonl(data_dir / f"{cfg['train_source']}_{cfg['train_split']}.jsonl")
        corpus = model_dir / "corpus.txt"
        n = build_training_corpus(train_recs, corpus)
        print(f"Corpus: {n:,} câu từ {len(train_recs):,} bản ghi {cfg['train_source']}/{cfg['train_split']}")
        train_spm(corpus, model_dir, cfg["model_prefix"], cfg["vocab_size"], cfg["model_type"],
                  cfg["character_coverage"], cfg["input_sentence_size"], cfg["seed"])
        corpus.unlink()  # corpus lớn, không commit
    sp = load_spm(model_path)
    print(f"Đã nạp {model_path} (vocab = {sp.get_piece_size()})")

    rep_rows, sum_rows = [], []
    for source in cfg["measure_sources"]:
        for split in cfg["measure_splits"]:
            recs = read_jsonl(data_dir / f"{source}_{split}.jsonl")
            rows, summ = measure(recs, sp, cfg["max_input_candidates"], cfg["max_target_candidates"])
            rep_rows += [{"source": source, "split": split, **r} for r in rows]
            sum_rows.append({"source": source, "split": split, **summ})
    tables = Path(cfg["tables_dir"]); tables.mkdir(parents=True, exist_ok=True)
    rep, summ = pd.DataFrame(rep_rows), pd.DataFrame(sum_rows)
    rep.to_csv(tables / "spm_truncation_report.csv", index=False)
    summ.to_csv(tables / "spm_summary.csv", index=False)
    print(summ.to_string(index=False))
    print(rep[(rep.source == "cnn_dm") & (rep.split == "train")].round(1).to_string(index=False))


if __name__ == "__main__":
    main()
