"""Giải mã + đánh giá BART/T5 (GĐ6, D10): mô hình đã tinh chỉnh (--ckpt best.pt) hoặc CHƯA tinh chỉnh (--zero_shot).

    # mô hình đã tinh chỉnh, tập test, cài đặt đánh giá D5 (ROUGE + BERTScore + CI)
    python -m src.generate_hf --config configs/bart_cnndm_in512.yaml --ckpt <ckpt_dir>/bart_cnndm_in512/best.pt --split test
    # mốc zero-shot (D10: Z1, Z2) -- không cần checkpoint
    python -m src.generate_hf --config configs/bart_cnndm_in512.yaml --zero_shot --split test
    # chọn tham số giải mã trên validation: không BERTScore, đổi từng tham số
    python -m src.generate_hf --config ... --ckpt ... --split validation --no_bertscore --num_beams 1
    # chéo miền: mô hình huấn luyện trên CNN/DM, đánh giá trên GOV.UK (BẮT BUỘC khai --max_length cho GOV.UK, vd. 64)
    python -m src.generate_hf --config configs/bart_cnndm_in512.yaml --ckpt ... --source govuk --max_length 64 --split test

Quy ước:
  * Mọi tham số giải mã được truyền tường minh (num_beams, max_length, min_length, length_penalty, no_repeat_ngram_size,
    early_stopping), gồm cả giá trị 0/False, để không bị mặc định ẩn trong generation_config của mô hình chen vào.
    Các mặc định ẩn đó được in ra và ghi vào cột `model_default_gen` của bảng để kiểm tra.
  * Tên hệ thống trong bảng = run_name (hoặc run_name + "_zeroshot"); cấu hình giải mã nằm ở cột gen_tag. Khóa bảng
    (source, system, gen_tag, split) nên các cấu hình khác nhau không ghi đè nhau và tập validation tách file khỏi test.
  * Đầu ra: results/predictions/{source}_{system}__{gen_tag}[_{split}].jsonl, results/tables/hf_{source}[_{split}].csv,
    results/scores/{source}_{system}__{gen_tag}[_{split}].csv (điểm từng mẫu cho GĐ7).
  * Không chọn tham số trên test: --split test chỉ chạy cho cấu hình đã chốt trên validation (D10).
  * ms/bài đo ở batch 1 trên `--time_docs` bài đầu (từ input_ids đã token hóa đến văn bản), thêm ms/bài khi chạy theo batch.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.data.eda_utils import novel_ngram_ratio
from src.data.hf_seq2seq_data import load_or_build, load_records, load_tokenizer
from src.decode_utils import pred_path, repeated_trigram_rate, sentences_text, table_path, upsert_rows
from src.evaluate import evaluate_with_samples
from src.hf_train_utils import gen_tag, generate_kwargs, scores_frame, scores_path
from src.train_hf import generate_texts

HIDDEN_KEYS = ("num_beams", "no_repeat_ngram_size", "early_stopping", "length_penalty", "min_length", "max_length",
               "forced_bos_token_id", "forced_eos_token_id")


def model_default_gen(model) -> dict:
    """Các mặc định giải mã MÀ MÔ HÌNH mang sẵn (bị ghi đè bởi tham số tường minh, nhưng đáng biết)."""
    gc = getattr(model, "generation_config", None)
    return {k: getattr(gc, k, None) for k in HIDDEN_KEYS} if gc is not None else {}


def _sync(device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def decode_split(model, tokenizer, examples: list[dict], gen: dict, *, batch_size: int, pad_id: int, device,
                 use_amp: bool, time_docs: int = 100) -> dict:
    """Giải mã toàn bộ `examples`. Trả về văn bản thô, văn bản đã tách câu ('\\n'), ms/bài theo batch và ms/bài ở batch 1."""
    _sync(device)
    t0 = time.time()
    raw = [t.strip() for t in generate_texts(model, tokenizer, examples, gen, batch_size, pad_id, device, use_amp)]
    _sync(device)
    ms_batched = 1000 * (time.time() - t0) / max(len(examples), 1)
    ms_bs1 = float("nan")
    n_t = min(time_docs, len(examples))
    if n_t > 0:
        generate_texts(model, tokenizer, examples[:1], gen, 1, pad_id, device, use_amp)      # khởi động, không tính giờ
        _sync(device)
        t1 = time.time()
        for ex in examples[:n_t]:
            generate_texts(model, tokenizer, [ex], gen, 1, pad_id, device, use_amp)
        _sync(device)
        ms_bs1 = 1000 * (time.time() - t1) / n_t
    return {"raw": raw, "preds": [sentences_text(t) for t in raw], "ms_batched": ms_batched, "ms_bs1": ms_bs1}


def generate_and_evaluate(model, tokenizer, examples: list[dict], records: list[dict], gen: dict, ev: dict, *,
                          source: str, system: str, split_tag: str, dirs: dict, batch_size: int, device,
                          use_amp: bool, no_bertscore: bool = False, time_docs: int = 100, seed: int = 42,
                          extra: dict | None = None) -> dict:
    """Giải mã, lưu dự đoán + điểm từng mẫu, đánh giá (D5), ghi một dòng vào bảng hf_{source}[_{split}].csv."""
    tag = gen_tag(gen)
    full = f"{system}__{tag}"
    refs = [r["highlights"] for r in records]
    dec = decode_split(model, tokenizer, examples, gen, batch_size=batch_size, pad_id=tokenizer.pad_token_id,
                       device=device, use_amp=use_amp, time_docs=time_docs)
    preds = dec["preds"]
    pp = pred_path(dirs["predictions"], source, full, split_tag)
    pp.parent.mkdir(parents=True, exist_ok=True)
    with open(pp, "w", encoding="utf-8") as f:
        for r, p in zip(records, preds):
            f.write(json.dumps({"id": r["id"], "prediction": p}, ensure_ascii=False) + "\n")
    row, per_sample = evaluate_with_samples(
        preds, refs, use_stemmer=ev["rouge_stemmer"], bertscore=ev["bertscore"] and not no_bertscore,
        bertscore_model=ev["bertscore_model"], bertscore_rescale=ev["bertscore_rescale"],
        bertscore_batch_size=ev["bertscore_batch_size"], n_boot=ev["bootstrap_samples"], seed=seed)
    sp = scores_path(dirs["scores"], source, full, split_tag)
    sp.parent.mkdir(parents=True, exist_ok=True)
    scores_frame([r["id"] for r in records], per_sample, preds, refs).to_csv(sp, index=False)
    row.update(source=source, system=system, gen_tag=tag, split=split_tag, num_beams=gen["num_beams"],
               max_length=gen["max_length"], min_length=gen["min_length"], length_penalty=gen["length_penalty"],
               no_repeat_ngram_size=gen["no_repeat_ngram_size"], early_stopping=bool(gen.get("early_stopping", False)),
               ms_per_doc_bs1=dec["ms_bs1"], ms_per_doc_batched=dec["ms_batched"],
               rep_trigram_pct=repeated_trigram_rate(preds),
               pred_sentences_mean=float(np.mean([p.count("\n") + 1 for p in preds])),   # =1 khi đầu ra không tách được câu (vd. chữ thường)
               novel_trigram_pct=100 * float(np.nanmean([novel_ngram_ratio(r["article"], p, 3) for r, p in zip(records, dec["raw"])])),
               n_params=sum(p.numel() for p in model.parameters()), **(extra or {}))
    first = ["source", "system", "gen_tag", "split", "n"]
    df = pd.DataFrame([row])
    df = df[first + [c for c in df.columns if c not in first]]
    upsert_rows(table_path(dirs["tables"], "hf", source, split_tag), df, keys=("source", "system", "gen_tag", "split"),
                old_defaults={"split": split_tag})
    row["_predictions"], row["_raw"] = preds, dec["raw"]
    return row


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="config huấn luyện của thí nghiệm (kind, model_name, độ dài, val_gen mặc định)")
    ap.add_argument("--ckpt", default=None, help="best.pt của thí nghiệm; bỏ trống khi dùng --zero_shot")
    ap.add_argument("--zero_shot", action="store_true", help="mô hình tiền huấn luyện CHƯA tinh chỉnh (mốc Z1/Z2 của D10)")
    ap.add_argument("--split", default="test", choices=["validation", "test"])
    ap.add_argument("--source", default=None, help="ghi đè nguồn dữ liệu (chéo miền); khi đó bắt buộc khai --max_length")
    ap.add_argument("--system", default=None, help="tên hệ thống trong bảng (mặc định: run_name hoặc run_name_zeroshot)")
    for k, t in (("num_beams", int), ("max_length", int), ("min_length", int), ("length_penalty", float),
                 ("no_repeat_ngram", int)):
        ap.add_argument(f"--{k}", type=t, default=None)
    ap.add_argument("--early_stopping", default=None, choices=["true", "false"])
    ap.add_argument("--max_input", type=int, default=None)
    ap.add_argument("--max_target", type=int, default=None, help="chỉ dùng cho khóa cache token hóa nhãn")
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--hf_cache_dir", default=None)
    ap.add_argument("--eval_config", default="configs/baselines.yaml", help="lấy mục evaluate để giống hệt GĐ4/D5")
    ap.add_argument("--predictions_dir", default="results/predictions")
    ap.add_argument("--tables_dir", default="results/tables")
    ap.add_argument("--scores_dir", default="results/scores")
    ap.add_argument("--eval_batch_size", type=int, default=16)
    ap.add_argument("--time_docs", type=int, default=100)
    ap.add_argument("--limit", type=int, default=None, help="chỉ N bài đầu để chạy thử (đầu ra có hậu tố _limitN)")
    ap.add_argument("--no_bertscore", action="store_true")
    a = ap.parse_args(argv)
    if bool(a.ckpt) == a.zero_shot:
        raise SystemExit("Chọn đúng một trong hai: --ckpt <best.pt> hoặc --zero_shot")

    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    evcfg = yaml.safe_load(open(a.eval_config, encoding="utf-8"))
    pre = yaml.safe_load(open(cfg["preprocess_config"], encoding="utf-8"))
    source = a.source or cfg["source"]
    gen = dict(cfg["val_gen"])
    if source != cfg["source"] and a.max_length is None:
        raise SystemExit(f"Đánh giá chéo miền ({cfg['source']} -> {source}): khai tường minh --max_length (vd. 64 cho GOV.UK), "
                         f"không dùng max_length={gen['max_length']} của nguồn huấn luyện.")
    for key, val in (("num_beams", a.num_beams), ("max_length", a.max_length), ("min_length", a.min_length),
                     ("length_penalty", a.length_penalty), ("no_repeat_ngram_size", a.no_repeat_ngram)):
        if val is not None:
            gen[key] = val
    if a.early_stopping is not None:
        gen["early_stopping"] = a.early_stopping == "true"
    max_input, max_target = a.max_input or cfg["max_input"], a.max_target or cfg["max_target"]
    kind, names = cfg["kind"], pre["tokenizers"]
    data_dir, cache_dir = a.data_dir or cfg["data_dir"], a.cache_dir or cfg["cache_dir"]
    system = a.system or (cfg["run_name"] + ("_zeroshot" if a.zero_shot else ""))
    split_tag = a.split if a.limit is None else f"{a.split}_limit{a.limit}"

    tok = load_tokenizer(names[kind], a.hf_cache_dir)
    prefix = pre["t5_prefix"] if kind == "t5" else ""
    examples, _ = load_or_build(data_dir, cache_dir, source, a.split, kind, tok, names[kind], max_input, max_target,
                                prefix, pre["target_join"])
    records = load_records(data_dir, source, a.split)
    if a.limit:
        examples, records = examples[:a.limit], records[:a.limit]
    if [e["id"] for e in examples] != [r["id"] for r in records]:
        raise SystemExit("Thứ tự/ID của dữ liệu token hóa (cache) không khớp file jsonl: xóa cache rồi chạy lại hf_seq2seq_data")

    from transformers import AutoModelForSeq2SeqLM
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(cfg["train"].get("amp", False)) and device.type == "cuda"
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg["model_name"], cache_dir=a.hf_cache_dir)
    extra = {"max_input": max_input, "zero_shot": a.zero_shot, "ckpt_epoch": None, "amp": use_amp,
             "model_default_gen": json.dumps(model_default_gen(model), ensure_ascii=False)}
    if a.ckpt:
        ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
        if ck.get("model_name") != cfg["model_name"]:
            raise SystemExit(f"Checkpoint của {ck.get('model_name')} nhưng config dùng {cfg['model_name']}")
        model.load_state_dict(ck["model"])
        extra["ckpt_epoch"] = int(ck["epoch"])
    model.to(device).eval()
    print(f"{system} | {source}/{a.split} ({len(records)} bài) | {'ZERO-SHOT' if a.zero_shot else 'ckpt epoch ' + str(extra['ckpt_epoch'])} "
          f"| {gen_tag(gen)} | max_input={max_input} | {device} AMP={use_amp}\n"
          f"mặc định ẩn của mô hình (bị ghi đè): {extra['model_default_gen']}")
    print("tham số giải mã dùng:", generate_kwargs(gen))
    dirs = {"predictions": a.predictions_dir, "tables": a.tables_dir, "scores": a.scores_dir}
    row = generate_and_evaluate(model, tok, examples, records, gen, evcfg["evaluate"], source=source, system=system,
                                split_tag=split_tag, dirs=dirs, batch_size=a.eval_batch_size, device=device,
                                use_amp=use_amp, no_bertscore=a.no_bertscore, time_docs=a.time_docs,
                                seed=evcfg.get("seed", 42), extra=extra)
    print(f"R1={row['rouge1']:.4f} R2={row['rouge2']:.4f} RLsum={row['rougeLsum']:.4f} "
          f"BS={row.get('bertscore_f1', float('nan')):.4f} | độ dài {row['avg_pred_words']:.1f} từ (tham chiếu {row['avg_ref_words']:.1f}) "
          f"| lặp trigram {row['rep_trigram_pct']:.1f}% | trigram mới {row['novel_trigram_pct']:.1f}% "
          f"| {row['ms_per_doc_bs1']:.0f} ms/bài (batch 1), {row['ms_per_doc_batched']:.0f} ms/bài (batch {a.eval_batch_size})")
    for r, p in list(zip(records, row["_predictions"]))[:2]:
        print(f"\n[{r['id'][:8]}] THAM CHIẾU: {r['highlights'][:200]!r}\n           DỰ ĐOÁN : {p[:200]!r}")


if __name__ == "__main__":
    main()
