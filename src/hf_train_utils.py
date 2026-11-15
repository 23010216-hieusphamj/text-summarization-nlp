"""Tiện ích huấn luyện BART/T5 không phụ thuộc torch/transformers (GĐ6, Bước 4): lịch học, chia batch theo epoch có
seed (để resume giữa epoch), tập con validation cố định, điểm chọn S, tham số giải mã tường minh, kiểm tra config."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.eda_utils import word_count

REQUIRED_TRAIN_KEYS = ("seed", "lr", "weight_decay", "batch_size", "grad_accum", "epochs", "warmup_ratio", "clip_norm")
GEN_KEYS = ("num_beams", "max_length", "min_length", "length_penalty", "no_repeat_ngram_size")


def validate_cfg(cfg: dict) -> None:
    """Báo lỗi sớm (kèm tên khóa) nếu config thiếu hoặc sai giá trị."""
    for k in ("run_name", "kind", "model_name", "source", "max_input", "max_target"):
        if k not in cfg:
            raise ValueError(f"Config thiếu khóa '{k}'")
    if cfg["kind"] not in ("bart", "t5"):
        raise ValueError("kind phải là 'bart' hoặc 't5'")
    tc = cfg.get("train", {})
    for k in REQUIRED_TRAIN_KEYS:
        if k not in tc:
            raise ValueError(f"Config thiếu train.{k}")
    if tc["batch_size"] < 1 or tc["grad_accum"] < 1 or tc["epochs"] < 1:
        raise ValueError("batch_size, grad_accum, epochs phải >= 1")
    if not 0 <= tc["warmup_ratio"] < 1:
        raise ValueError("warmup_ratio phải trong [0, 1)")
    if cfg.get("best_by", "S") not in ("S", "val_loss"):
        raise ValueError("best_by phải là 'S' hoặc 'val_loss'")
    gen = cfg.get("val_gen", {})
    for k in GEN_KEYS:
        if k not in gen:
            raise ValueError(f"Config thiếu val_gen.{k} (giải mã phải khai tường minh, không dùng mặc định ẩn của mô hình)")


def lr_lambda(step: int, warmup_steps: int, total_steps: int) -> float:
    """Hệ số nhân lr: tăng tuyến tính 0->1 trong warmup rồi giảm tuyến tính về 0 ở bước cuối. `step` đếm từ 0."""
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))


def make_steps(n_examples: int, batch_size: int, grad_accum: int, seed: int, epoch: int) -> list[list[np.ndarray]]:
    """Chia một epoch thành các BƯỚC TỐI ƯU, mỗi bước gồm tối đa `grad_accum` micro-batch (mảng chỉ số).

    Thứ tự xáo trộn chỉ phụ thuộc (seed, epoch) -> resume giữa epoch chỉ cần bỏ qua các bước đã làm."""
    perm = np.random.RandomState(seed + epoch).permutation(n_examples)
    micro = [perm[i:i + batch_size] for i in range(0, n_examples, batch_size)]
    return [micro[i:i + grad_accum] for i in range(0, len(micro), grad_accum)]


def steps_per_epoch(n_examples: int, batch_size: int, grad_accum: int) -> int:
    return math.ceil(math.ceil(n_examples / batch_size) / grad_accum)


def val_subset_indices(n: int, k: int, seed: int = 42) -> list[int]:
    """k chỉ số validation cố định (sắp xếp) để tính ROUGE mỗi epoch; k >= n thì lấy hết."""
    if k >= n:
        return list(range(n))
    return sorted(np.random.RandomState(seed).choice(n, size=k, replace=False).tolist())


def selection_score(rouge: dict) -> float:
    """S = trung bình (ROUGE-1, ROUGE-2, ROUGE-Lsum) thang 0-100 (D10). `rouge` có khóa rouge1, rouge2, rougeLsum (thang 0-1)."""
    return 100.0 * (rouge["rouge1"] + rouge["rouge2"] + rouge["rougeLsum"]) / 3.0


def is_no_decay(param_name: str) -> bool:
    """Bias và LayerNorm không áp weight decay."""
    n = param_name.lower()
    return n.endswith(".bias") or "layer_norm" in n or "layernorm" in n


def generate_kwargs(gen: dict) -> dict:
    """Tham số giải mã ĐẦY ĐỦ cho model.generate. Luôn truyền tường minh cả num_beams, độ dài, length_penalty,
    no_repeat_ngram_size (kể cả 0) để không bị mặc định ẩn trong config/generation_config của mô hình chen vào."""
    return {"num_beams": int(gen["num_beams"]), "max_length": int(gen["max_length"]), "min_length": int(gen["min_length"]),
            "length_penalty": float(gen["length_penalty"]), "no_repeat_ngram_size": int(gen["no_repeat_ngram_size"]),
            "early_stopping": bool(gen.get("early_stopping", False)), "do_sample": False}


def truncate_csv_after(path: str | Path, column: str, last_kept: float) -> int:
    """Xóa các dòng có `column` > last_kept (dùng khi resume để không trùng dòng log). Trả về số dòng đã xóa."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return 0
    df = pd.read_csv(path)
    keep = df[df[column] <= last_kept]
    removed = len(df) - len(keep)
    if removed:
        keep.to_csv(path, index=False)
    return removed


def gen_tag(gen: dict) -> str:
    """Tên ngắn, duy nhất cho một cấu hình giải mã (dùng trong tên file và khóa bảng), vd. 'b4-lp1-nr3-min0-max128'."""
    tag = (f"b{int(gen['num_beams'])}-lp{float(gen['length_penalty']):g}-nr{int(gen['no_repeat_ngram_size'])}"
           f"-min{int(gen['min_length'])}-max{int(gen['max_length'])}")
    return tag + ("-es" if gen.get("early_stopping", False) else "")


def scores_path(scores_dir: str | Path, source: str, system: str, split: str = "test") -> Path:
    """File điểm từng mẫu. Tập test không hậu tố; tập khác thêm _{split} (cùng quy ước decode_utils.pred_path)."""
    suffix = "" if split == "test" else f"_{split}"
    return Path(scores_dir) / f"{source}_{system}{suffix}.csv"


def scores_frame(ids: list[str], per_sample: dict, preds: list[str], refs: list[str]) -> pd.DataFrame:
    """Bảng điểm từng mẫu: id, các chỉ số ROUGE/BERTScore, số từ dự đoán và tham chiếu."""
    df = pd.DataFrame({"id": list(ids)})
    for k, v in per_sample.items():
        df[k] = np.asarray(v, dtype=float)
    df["pred_words"] = [word_count(p) for p in preds]
    df["ref_words"] = [word_count(r) for r in refs]
    return df
