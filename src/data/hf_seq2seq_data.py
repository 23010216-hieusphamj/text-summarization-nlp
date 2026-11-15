"""Dữ liệu cho tinh chỉnh BART-base / T5-small (GĐ6, Bước 3).

Luồng: jsonl sạch {id, article, highlights} -> tokenizer HuggingFace (cắt max_input/max_target, T5 thêm tiền tố
"summarize: ") -> danh sách ví dụ {id, input_ids, labels} -> cache .npz -> collate đệm động theo batch.

Quy ước (khớp D7, D10):
  * Đầu vào bị cắt ở đầu bài (truncation=True) tại max_input token, tính cả token đặc biệt và tiền tố T5.
  * Nhãn: tokenizer(text_target=...) cắt ở max_target token; các câu tóm tắt CNN/DM (tách bằng "\n") được nối bằng
    target_join (mặc định " ") như GĐ3. Token đặc biệt cuối câu (</s>) được giữ sau khi cắt.
  * Padding nhãn = -100 (bị bỏ qua khi tính loss). decoder_input_ids KHÔNG tạo ở đây: model.forward(labels=...) của
    BART/T5 tự dịch nhãn sang phải (shift_tokens_right) -> vòng huấn luyện (Bước 4) chỉ cần đưa labels vào.
  * Nhóm độ dài bài theo số token của tokenizer BART (không cắt): "<=512" / ">512" — dùng chung cho mọi hệ thống
    (BART, T5, Seq2Seq, Lead) để so sánh trên cùng một tập bài (nhận xét 2 của giảng viên).

Cache: results không nằm trong git (data/processed/ bị .gitignore). Cache gắn với sha1 của file jsonl, tên tokenizer,
kích thước từ vựng, tiền tố, các độ dài cắt và một "canary" (id của câu mẫu) nên đổi dữ liệu/phiên bản tokenizer
sẽ tự xây lại thay vì dùng nhầm.

Chạy (kiểm tra tokenization + xây cache + bảng độ dài token), xem configs/hf_data.yaml:
    python -m src.data.hf_seq2seq_data --config configs/hf_data.yaml --data_dir <clean> --cache_dir <thư mục cache>
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.preprocessing.tokenization import tokenize_batch

LABEL_PAD = -100
CANARY_TEXT = "Summarize this: The Quick Brown Fox, 12 jumps over 3 lazy dogs."


# ----------------------------------------------------------------------------- đọc dữ liệu
def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_records(data_dir: str | Path, source: str, split: str, limit: int | None = None) -> list[dict]:
    """Đọc {data_dir}/{source}_{split}.jsonl (id, article, highlights). `limit` chỉ để chạy thử."""
    path = Path(data_dir) / f"{source}_{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Không thấy {path}")
    recs = read_jsonl(path)
    return recs[:limit] if limit else recs


def file_sha1(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


# ----------------------------------------------------------------------------- tokenize
def load_tokenizer(name: str, hf_cache_dir: str | None = None):
    """Tải tokenizer HuggingFace (cần mạng lần đầu; đặt hf_cache_dir/HF_HOME trên Drive để dùng lại giữa các phiên)."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(name, cache_dir=hf_cache_dir)


def tokenize_records(records: list[dict], tokenizer, max_input: int, max_target: int, prefix: str = "",
                     target_join: str = " ", batch_size: int = 500) -> list[dict]:
    """Records -> [{'id', 'input_ids', 'labels'}] (danh sách id, chưa padding). Dùng lại tokenize_batch của GĐ3."""
    out = []
    for i in range(0, len(records), batch_size):
        part = records[i:i + batch_size]
        enc = tokenize_batch({"article": [r["article"] for r in part], "highlights": [r["highlights"] for r in part]},
                             tokenizer, max_input, max_target, prefix, target_join)
        for r, ids, lab in zip(part, enc["input_ids"], enc["labels"]):
            out.append({"id": r.get("id", ""), "input_ids": list(ids), "labels": list(lab)})
    return out


# ----------------------------------------------------------------------------- cache .npz
def pack_ragged(seqs: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    """Danh sách dãy id độ dài khác nhau -> (mảng phẳng int32, offsets int64 độ dài n+1)."""
    lens = np.fromiter((len(s) for s in seqs), dtype=np.int64, count=len(seqs))
    offsets = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    flat = np.fromiter((t for s in seqs for t in s), dtype=np.int32, count=int(offsets[-1]))
    return flat, offsets


def unpack_ragged(flat: np.ndarray, offsets: np.ndarray) -> list[list[int]]:
    return [flat[offsets[i]:offsets[i + 1]].tolist() for i in range(len(offsets) - 1)]


def cache_stem(cache_dir: str | Path, source: str, split: str, kind: str, max_input: int, max_target: int) -> Path:
    return Path(cache_dir) / f"{source}_{split}_{kind}_in{max_input}_tg{max_target}"


def make_meta(tokenizer, tokenizer_name: str, data_path: str | Path, max_input: int, max_target: int, prefix: str,
              target_join: str, limit: int | None) -> dict:
    """Thông tin định danh cache: khác một mục nào thì cache cũ bị bỏ."""
    return {"tokenizer_name": tokenizer_name, "vocab_size": len(tokenizer), "prefix": prefix,
            "target_join": target_join, "max_input": max_input, "max_target": max_target, "limit": limit,
            "data_sha1": file_sha1(data_path), "canary": list(tokenizer(CANARY_TEXT)["input_ids"])}


def save_examples(stem: str | Path, examples: list[dict], meta: dict) -> None:
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    ifl, ioff = pack_ragged([e["input_ids"] for e in examples])
    lfl, loff = pack_ragged([e["labels"] for e in examples])
    np.savez_compressed(stem.with_suffix(".npz"), in_flat=ifl, in_off=ioff, lab_flat=lfl, lab_off=loff,
                        ids=np.array([e["id"] for e in examples], dtype=str))
    stem.with_suffix(".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")


def load_examples_cache(stem: str | Path, meta: dict) -> list[dict] | None:
    """Trả về ví dụ đã cache nếu meta khớp hoàn toàn, ngược lại None (phải xây lại)."""
    stem = Path(stem)
    npz, mp = stem.with_suffix(".npz"), stem.with_suffix(".meta.json")
    if not (npz.exists() and mp.exists()):
        return None
    if json.loads(mp.read_text(encoding="utf-8")) != meta:
        return None
    z = np.load(npz, allow_pickle=False)
    inputs, labels = unpack_ragged(z["in_flat"], z["in_off"]), unpack_ragged(z["lab_flat"], z["lab_off"])
    return [{"id": str(i), "input_ids": a, "labels": b} for i, a, b in zip(z["ids"], inputs, labels)]


def load_or_build(data_dir: str | Path, cache_dir: str | Path, source: str, split: str, kind: str, tokenizer,
                  tokenizer_name: str, max_input: int, max_target: int, prefix: str = "", target_join: str = " ",
                  limit: int | None = None) -> tuple[list[dict], bool]:
    """Nạp từ cache nếu hợp lệ, không thì tokenize rồi lưu. Trả về (ví dụ, dùng_cache)."""
    data_path = Path(data_dir) / f"{source}_{split}.jsonl"
    meta = make_meta(tokenizer, tokenizer_name, data_path, max_input, max_target, prefix, target_join, limit)
    stem = cache_stem(cache_dir, source, split, kind, max_input, max_target)
    cached = load_examples_cache(stem, meta)
    if cached is not None:
        return cached, True
    examples = tokenize_records(load_records(data_dir, source, split, limit), tokenizer, max_input, max_target,
                                prefix, target_join)
    save_examples(stem, examples, meta)
    return examples, False


# ----------------------------------------------------------------------------- batch
def collate_seq2seq(batch: list[dict], pad_token_id: int, label_pad_id: int = LABEL_PAD) -> dict:
    """Đệm động: input_ids (pad_token_id), attention_mask, labels (-100). Trả về mảng numpy int64 (chuyển sang tensor ở vòng huấn luyện)."""
    B = len(batch)
    S = max(len(b["input_ids"]) for b in batch)
    T = max(len(b["labels"]) for b in batch)
    inp = np.full((B, S), pad_token_id, dtype=np.int64)
    att = np.zeros((B, S), dtype=np.int64)
    lab = np.full((B, T), label_pad_id, dtype=np.int64)
    for i, b in enumerate(batch):
        inp[i, :len(b["input_ids"])] = b["input_ids"]
        att[i, :len(b["input_ids"])] = 1
        lab[i, :len(b["labels"])] = b["labels"]
    return {"id": [b["id"] for b in batch], "input_ids": inp, "attention_mask": att, "labels": lab}


def make_loader(examples: list[dict], batch_size: int, pad_token_id: int, shuffle: bool = False, seed: int = 42,
                num_workers: int = 0):
    """DataLoader (cần torch) trả về tensor; xáo trộn có seed cố định."""
    import torch
    from torch.utils.data import DataLoader

    def _collate(batch):
        out = collate_seq2seq(batch, pad_token_id)
        return {k: (torch.as_tensor(v) if isinstance(v, np.ndarray) else v) for k, v in out.items()}

    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(examples, batch_size=batch_size, shuffle=shuffle, generator=g, num_workers=num_workers,
                      collate_fn=_collate)


# ----------------------------------------------------------------------------- độ dài token theo bài
def length_table(records: list[dict], n_tok_bart: np.ndarray, n_tok_t5: np.ndarray | None = None,
                 thresholds: tuple[int, ...] = (512, 1024)) -> pd.DataFrame:
    """Bảng id, số token (không cắt) theo BART (và T5 nếu có), cờ/nhóm vượt ngưỡng theo BART.

    Nhóm 'group_512' = '<=512' hoặc '>512' theo số token BART: dùng chung để chia điểm theo độ dài bài cho
    mọi hệ thống trên cùng một tập bài."""
    if len(n_tok_bart) != len(records):
        raise ValueError("Số độ dài không khớp số bản ghi")
    df = pd.DataFrame({"id": [r.get("id", "") for r in records], "n_tok_bart": np.asarray(n_tok_bart)})
    if n_tok_t5 is not None:
        df["n_tok_t5"] = np.asarray(n_tok_t5)
    for t in thresholds:
        df[f"gt_{t}"] = df["n_tok_bart"] > t
    df["group_512"] = np.where(df["n_tok_bart"] > 512, ">512", "<=512")
    return df


def check_examples(examples: list[dict], tokenizer, max_input: int, max_target: int) -> dict:
    """Số liệu kiểm tra tokenization: độ dài sau cắt, % chạm trần (≈ % bị cắt), nhãn rỗng, tỉ lệ kết thúc bằng EOS."""
    il = np.array([len(e["input_ids"]) for e in examples])
    ll = np.array([len(e["labels"]) for e in examples])
    eos = getattr(tokenizer, "eos_token_id", None)
    return {"n": len(examples), "in_len_mean": float(il.mean()), "in_len_max": int(il.max()),
            "pct_input_at_max": float((il >= max_input).mean() * 100),
            "lab_len_mean": float(ll.mean()), "lab_len_max": int(ll.max()),
            "pct_label_at_max": float((ll >= max_target).mean() * 100),
            "n_empty_labels": int((ll <= 2).sum()),
            "pct_input_ends_eos": float(np.mean([e["input_ids"][-1] == eos for e in examples]) * 100) if eos is not None else float("nan"),
            "pct_label_ends_eos": float(np.mean([e["labels"][-1] == eos for e in examples]) * 100) if eos is not None else float("nan")}


# ----------------------------------------------------------------------------- CLI
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/hf_data.yaml")
    ap.add_argument("--data_dir", default=None, help="ghi đè data_dir trong config (trên Colab: thư mục clean trên Drive)")
    ap.add_argument("--cache_dir", default=None, help="ghi đè cache_dir (nên đặt trên Drive để dùng lại giữa các phiên)")
    ap.add_argument("--hf_cache_dir", default=None, help="thư mục cache model/tokenizer HuggingFace (tùy chọn)")
    ap.add_argument("--limit", type=int, default=None, help="chỉ N bản ghi đầu, để chạy thử (cache có meta riêng)")
    ap.add_argument("--show", type=int, default=1, help="in N ví dụ giải mã ngược để kiểm tra bằng mắt")
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    pre = yaml.safe_load(open(cfg["preprocess_config"], encoding="utf-8"))
    data_dir, cache_dir = a.data_dir or cfg["data_dir"], a.cache_dir or cfg["cache_dir"]
    names, t5_prefix, join = pre["tokenizers"], pre["t5_prefix"], pre["target_join"]
    tables = Path(cfg["output"]["tables_dir"])
    tables.mkdir(parents=True, exist_ok=True)
    toks = {k: load_tokenizer(names[k], a.hf_cache_dir) for k in {p["kind"] for p in cfg["plan"]} | {"bart", "t5"}}

    rows = []
    for p in cfg["plan"]:
        kind, tok = p["kind"], toks[p["kind"]]
        prefix = t5_prefix if kind == "t5" else ""
        for split in cfg["splits"]:
            ex, hit = load_or_build(data_dir, cache_dir, p["source"], split, kind, tok, names[kind], p["max_input"],
                                    p["max_target"], prefix, join, a.limit)
            row = {"source": p["source"], "split": split, "kind": kind, "max_input": p["max_input"],
                   "max_target": p["max_target"], "from_cache": hit, **check_examples(ex, tok, p["max_input"], p["max_target"])}
            rows.append(row)
            print(f"{p['source']}/{split}/{kind} in{p['max_input']} tg{p['max_target']}: n={row['n']} "
                  f"%cắt đầu vào≈{row['pct_input_at_max']:.1f} %nhãn chạm trần={row['pct_label_at_max']:.1f} "
                  f"EOS cuối nhãn={row['pct_label_ends_eos']:.1f}% ({'cache' if hit else 'mới'})")
            if split == cfg["splits"][0]:
                for e in ex[:a.show]:
                    print("   vào :", repr(tok.decode(e["input_ids"][:24])), "...")
                    print("   nhãn:", repr(tok.decode(e["labels"], skip_special_tokens=False)[:160]))
    out = tables / ("hf_tokenization_check.csv" if a.limit is None else f"hf_tokenization_check_limit{a.limit}.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print("Đã lưu", out)

    lt = cfg["length_tables"]
    for source in sorted({p["source"] for p in cfg["plan"]}):
        for split in lt["splits"]:
            recs = load_records(data_dir, source, split, a.limit)
            nb = np.array([len(x) for x in toks["bart"]([r["article"] for r in recs], truncation=False, verbose=False)["input_ids"]])
            nt = np.array([len(x) for x in toks["t5"]([t5_prefix + r["article"] for r in recs], truncation=False, verbose=False)["input_ids"]])
            df = length_table(recs, nb, nt, tuple(lt["thresholds"]))
            name = f"token_lengths_{source}_{split}.csv" if a.limit is None else f"token_lengths_{source}_{split}_limit{a.limit}.csv"
            df.to_csv(tables / name, index=False)
            print(f"{source}/{split}: >512 token (BART) = {100 * df['gt_512'].mean():.1f}%, >1024 = {100 * df['gt_1024'].mean():.1f}% -> {name}")


if __name__ == "__main__":
    main()
