"""Tinh chỉnh BART-base / T5-small (GĐ6, D10) với vòng huấn luyện PyTorch tự viết, có resume giữa epoch.

    python -m src.train_hf --config configs/bart_cnndm_in512.yaml --data_dir <clean> --cache_dir <hf_cache> \\
        --ckpt_dir <thư mục trên Drive> --hf_cache_dir <HF_HOME trên Drive>
    python -m src.train_hf ... --limit_train 400 --limit_val 100 --epochs 1 --run_name smoke   # chạy thử, đo giây/epoch
    python -m src.train_hf ... --log_dir <thư mục log trên Drive>                                 # log không mất khi Colab ngắt
    python -m src.train_hf ... --resume                                                         # tiếp tục sau khi Colab ngắt

Mỗi epoch: train (AdamW, warmup + giảm tuyến tính, gom gradient), val loss trên TOÀN BỘ validation (teacher forcing),
ROUGE trên tập con validation cố định (`val_rouge_n` mẫu) bằng giải mã `val_gen` (khai tường minh). Checkpoint tốt nhất
chọn theo `best_by`: "S" (S = trung bình R1/R2/Lsum trên tập con validation, đúng D10) hoặc "val_loss". Với "S", epoch mới chỉ
thay best.pt khi S cao hơn best hiện tại ít nhất `select_tol` điểm (mặc định 0,2: chênh nhỏ hơn coi là hòa, giữ epoch sớm hơn).

Đầu ra: results/logs/{run}_log.csv (theo epoch), {run}_steps.csv (theo bước tối ưu), {run}_meta.json;
checkpoint {ckpt_dir}/{run}/best.pt (chỉ trọng số) và last.pt (trọng số + optimizer + lịch học + tiến độ, dùng để --resume).

Resume: thứ tự dữ liệu mỗi epoch chỉ phụ thuộc (seed, epoch) nên có thể tiếp tục giữa epoch (`save_every_steps`). Dropout
sau khi resume không lặp lại đúng dãy số ngẫu nhiên của lần chạy liền mạch (kết quả tương đương về thống kê, không
trùng từng bit). Mỗi lần lưu last.pt ghi cả trọng số và trạng thái optimizer (BART-base ~ 1,7 GB): đặt
`save_every_steps` hợp lý nếu Drive chậm hoặc gần đầy.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

from src.data.hf_seq2seq_data import collate_seq2seq, load_or_build, load_records, load_tokenizer
from src.decode_utils import sentences_text
from src.evaluate import rouge_per_sample
from src.hf_train_utils import (generate_kwargs, is_no_decay, lr_lambda, make_steps, selection_score,
                                steps_per_epoch, truncate_csv_after, val_subset_indices, validate_cfg)
from src.train_utils import EarlyStopper, append_row


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def to_tensors(batch: dict, device) -> dict:
    return {k: torch.as_tensor(v).to(device) for k, v in batch.items() if k != "id"}


def forward_loss(model, batch: dict, device, use_amp: bool):
    """Loss trung bình theo token nhãn (bỏ -100). decoder_input_ids do model tự dịch từ labels."""
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
    return out.loss


@torch.no_grad()
def evaluate_loss(model, examples: list[dict], batch_size: int, pad_id: int, device, use_amp: bool) -> float:
    """Val loss (teacher forcing), trung bình theo token trên các ví dụ đã cho."""
    model.eval()
    tot, cnt = 0.0, 0
    for i in range(0, len(examples), batch_size):
        batch = to_tensors(collate_seq2seq(examples[i:i + batch_size], pad_id), device)
        n = int((batch["labels"] != -100).sum())
        tot += forward_loss(model, batch, device, use_amp).item() * n
        cnt += n
    return tot / max(cnt, 1)


@torch.no_grad()
def generate_texts(model, tokenizer, examples: list[dict], gen_cfg: dict, batch_size: int, pad_id: int, device,
                   use_amp: bool) -> list[str]:
    """Giải mã từ input_ids đã token hóa (đã cắt + tiền tố). Mọi tham số giải mã được truyền tường minh."""
    model.eval()
    kw, texts = generate_kwargs(gen_cfg), []
    for i in range(0, len(examples), batch_size):
        batch = to_tensors(collate_seq2seq(examples[i:i + batch_size], pad_id), device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            out = model.generate(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], **kw)
        texts += tokenizer.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return texts


def val_rouge(preds: list[str], refs: list[str]) -> dict:
    """ROUGE-1/2/Lsum trung bình (thang 0-1) theo cài đặt D5 (stemming); câu nối bằng '\\n' cho Lsum."""
    r = rouge_per_sample([sentences_text(p) for p in preds], refs)
    return {k: float(r[k].mean()) for k in ("rouge1", "rouge2", "rougeLsum")}


def _save(path: Path, obj: dict) -> float:
    """Ghi nguyên tử (tránh file hỏng nếu Colab ngắt giữa chừng). Trả về số giây đã dùng."""
    t0 = time.time()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)
    return time.time() - t0


def train(cfg: dict, train_ex: list[dict], val_ex: list[dict], val_refs: list[str], tokenizer, log_dir: str | Path,
          ckpt_dir: str | Path, resume: bool = False, device: str | None = None, model=None,
          halt_after_step: int | None = None) -> dict:
    """Huấn luyện một thí nghiệm. `model`/`halt_after_step` chủ yếu cho test (mô hình nhỏ, giả lập bị ngắt)."""
    validate_cfg(cfg)
    tc, run_name = cfg["train"], cfg["run_name"]
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    use_amp = bool(tc.get("amp", False)) and device.type == "cuda"
    set_seed(tc["seed"])
    if model is None:
        from transformers import AutoModelForSeq2SeqLM
        model = AutoModelForSeq2SeqLM.from_pretrained(cfg["model_name"], cache_dir=cfg.get("hf_cache_dir"))
    model.to(device)
    pad_id = tokenizer.pad_token_id
    n_params = sum(p.numel() for p in model.parameters())

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if p.requires_grad:
            (no_decay if is_no_decay(n) else decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": tc["weight_decay"]},
                             {"params": no_decay, "weight_decay": 0.0}], lr=tc["lr"])
    bs, accum, epochs = tc["batch_size"], tc["grad_accum"], tc["epochs"]
    spe = steps_per_epoch(len(train_ex), bs, accum)
    total_steps = spe * epochs
    warmup = int(round(tc["warmup_ratio"] * total_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_lambda(s, warmup, total_steps))
    scaler = make_scaler(use_amp)
    best_by = cfg.get("best_by", "S")
    # best.pt chỉ được thay khi S tăng >= select_tol điểm (mặc định 0,2 theo quy tắc hòa của D10: hòa -> giữ epoch sớm hơn)
    stopper = EarlyStopper(tc.get("patience", epochs), min_delta=tc.get("select_tol", 0.2) if best_by == "S" else 0.0)
    eval_bs = tc.get("eval_batch_size", bs * 2)
    save_every = int(tc.get("save_every_steps", 0))
    sub_idx = val_subset_indices(len(val_ex), cfg.get("val_rouge_n", 300), tc["seed"])
    sub_ex, sub_refs = [val_ex[i] for i in sub_idx], [val_refs[i] for i in sub_idx]

    log_dir, run_dir = Path(log_dir), Path(ckpt_dir) / run_name
    epoch_csv, step_csv, meta_json = (log_dir / f"{run_name}_log.csv", log_dir / f"{run_name}_steps.csv",
                                      log_dir / f"{run_name}_meta.json")
    start_epoch, skip_steps, global_step, train_seconds = 1, 0, 0, 0.0
    part = {"tot": 0.0, "cnt": 0, "train_s": 0.0}          # tổng loss/token và thời gian train của epoch đang dở
    if resume and (run_dir / "last.pt").exists():
        ck = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"]); stopper.load_state_dict(ck["stopper"])
        global_step, train_seconds = ck["global_step"], ck["train_seconds"]
        if ck["epoch_done"]:
            start_epoch, last_done = ck["epoch"] + 1, ck["epoch"]
        else:
            start_epoch, skip_steps, part, last_done = ck["epoch"], ck["step_in_epoch"], dict(ck["partial"]), ck["epoch"] - 1
        n1 = truncate_csv_after(epoch_csv, "epoch", last_done)      # bỏ dòng log trùng do bị ngắt giữa chừng
        n2 = truncate_csv_after(step_csv, "step", global_step)
        print(f"Tiếp tục: epoch {start_epoch}, bỏ qua {skip_steps} bước đã làm, global_step={global_step} "
              f"(xóa {n1}+{n2} dòng log trùng)")
    elif epoch_csv.exists() or step_csv.exists():
        raise FileExistsError(f"Đã có log của run '{run_name}'. Dùng --resume hoặc đổi --run_name.")

    log_dir.mkdir(parents=True, exist_ok=True)
    versions = {"torch": torch.__version__}
    try:
        import transformers
        versions["transformers"] = transformers.__version__
    except ImportError:
        pass
    meta = {"run": run_name, "started": datetime.now().isoformat(timespec="seconds"), "seed": tc["seed"],
            "n_params": n_params, "device": str(device), "amp": use_amp, "versions": versions,
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "n_train": len(train_ex), "n_val": len(val_ex), "n_val_rouge": len(sub_ex), "steps_per_epoch": spe,
            "effective_batch": bs * accum, "best_by": best_by, "config": cfg}
    json.dump(meta, open(meta_json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"Run {run_name}: {n_params:,} tham số, {device}, AMP={use_amp}, {len(train_ex):,} train / {len(val_ex):,} val, "
          f"{spe} bước/epoch (batch hiệu dụng {bs * accum})")

    def state(epoch: int, step_in_epoch: int, epoch_done: bool) -> dict:
        return {"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                "scaler": scaler.state_dict(), "stopper": stopper.state_dict(), "epoch": epoch,
                "step_in_epoch": step_in_epoch, "epoch_done": epoch_done, "partial": dict(part),
                "global_step": global_step, "train_seconds": train_seconds}

    stopped_early, halted, save_seconds = False, False, []
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        steps = make_steps(len(train_ex), bs, accum, tc["seed"], epoch)
        if skip_steps == 0:
            part = {"tot": 0.0, "cnt": 0, "train_s": 0.0}
        for si in range(skip_steps, len(steps)):
            t_step, group = time.time(), steps[si]
            opt.zero_grad(set_to_none=True)
            for idx in group:
                batch = to_tensors(collate_seq2seq([train_ex[i] for i in idx], pad_id), device)
                n_tok = int((batch["labels"] != -100).sum())
                loss = forward_loss(model, batch, device, use_amp)
                scaler.scale(loss / len(group)).backward()
                part["tot"] += loss.item() * n_tok
                part["cnt"] += n_tok
            scaler.unscale_(opt)
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc["clip_norm"])
            scaler.step(opt); scaler.update(); sched.step()
            global_step += 1
            part["train_s"] += time.time() - t_step
            if global_step % tc.get("log_every", 50) == 0:
                append_row(step_csv, {"step": global_step, "epoch": epoch, "loss": round(loss.item(), 5),
                                      "grad_norm": round(float(gnorm), 4), "lr": sched.get_last_lr()[0]})
            halt = halt_after_step is not None and global_step == halt_after_step
            if halt or (save_every and (si + 1) % save_every == 0 and si + 1 < len(steps)):
                save_seconds.append(_save(run_dir / "last.pt", state(epoch, si + 1, False)))
            if halt:
                halted = True
                break
        skip_steps = 0
        if halted:
            break
        t_eval = time.time()
        val_loss = evaluate_loss(model, val_ex, eval_bs, pad_id, device, use_amp)
        preds = generate_texts(model, tokenizer, sub_ex, cfg["val_gen"], eval_bs, pad_id, device, use_amp)
        rg = val_rouge(preds, sub_refs)
        S = selection_score(rg)
        eval_s = time.time() - t_eval
        train_seconds += part["train_s"]
        row = {"epoch": epoch, "train_loss": round(part["tot"] / max(part["cnt"], 1), 5), "val_loss": round(val_loss, 5),
               "val_ppl": round(math.exp(min(val_loss, 20)), 3), "val_rouge1": round(rg["rouge1"], 5),
               "val_rouge2": round(rg["rouge2"], 5), "val_rougeLsum": round(rg["rougeLsum"], 5), "val_S": round(S, 3),
               "val_pred_words": round(float(np.mean([len(p.split()) for p in preds])), 1),
               "lr": sched.get_last_lr()[0], "epoch_seconds": round(part["train_s"] + eval_s, 1),
               "train_only_seconds": round(part["train_s"], 1), "eval_seconds": round(eval_s, 1),
               "train_seconds_total": round(train_seconds, 1)}
        append_row(epoch_csv, row)                      # ghi log TRƯỚC khi lưu checkpoint (resume sẽ xóa dòng trùng)
        improved = stopper.update(-S if best_by == "S" else val_loss, epoch)
        save_seconds.append(_save(run_dir / "last.pt", state(epoch, len(steps), True)))
        if improved:
            _save(run_dir / "best.pt", {"model": model.state_dict(), "epoch": epoch, "val_loss": val_loss, "val_S": S,
                                        "model_name": cfg["model_name"], "kind": cfg["kind"], "config": cfg})
        print(f"epoch {epoch:2d} | train {row['train_loss']:.4f} | val {row['val_loss']:.4f} | S {S:.2f} "
              f"(R1 {100 * rg['rouge1']:.1f}/R2 {100 * rg['rouge2']:.1f}/Lsum {100 * rg['rougeLsum']:.1f}, "
              f"{row['val_pred_words']} từ){' *' if improved else ''} | train {part['train_s']:.0f}s + eval {eval_s:.0f}s")
        if stopper.should_stop:
            stopped_early = True
            print(f"Dừng sớm: không cải thiện {stopper.patience} epoch liên tiếp.")
            break

    best = (-stopper.best if best_by == "S" else stopper.best)
    result = {"halted": halted, "best_metric": best_by, "best_value": best, "best_epoch": stopper.best_epoch,
              "stopped_early": stopped_early, "train_seconds": round(train_seconds, 1), "n_params": n_params,
              "global_step": global_step, "mean_save_seconds": round(float(np.mean(save_seconds)), 1) if save_seconds else None}
    meta.update(result, finished=datetime.now().isoformat(timespec="seconds"))
    json.dump(meta, open(meta_json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    return result


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--cache_dir", default=None, help="thư mục cache token hóa (xem hf_seq2seq_data)")
    ap.add_argument("--ckpt_dir", default=None, help="nơi lưu checkpoint (nên là thư mục trên Drive)")
    ap.add_argument("--hf_cache_dir", default=None)
    ap.add_argument("--log_dir", default=None, help="thư mục log (nên là Drive để log không mất khi Colab bị ngắt và --resume còn đủ dòng)")
    ap.add_argument("--run_name", default=None)
    ap.add_argument("--limit_train", type=int, default=None, help="chỉ N ví dụ train đầu (cắt SAU khi nạp cache đầy đủ)")
    ap.add_argument("--limit_val", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    if a.run_name:
        cfg["run_name"] = a.run_name
    if a.epochs:
        cfg["train"]["epochs"] = a.epochs
        cfg["train"].setdefault("patience", a.epochs)
    if a.hf_cache_dir:
        cfg["hf_cache_dir"] = a.hf_cache_dir
    validate_cfg(cfg)
    data_dir, cache_dir = a.data_dir or cfg["data_dir"], a.cache_dir or cfg["cache_dir"]
    pre = yaml.safe_load(open(cfg["preprocess_config"], encoding="utf-8"))
    names, kind = pre["tokenizers"], cfg["kind"]
    tok = load_tokenizer(names[kind], cfg.get("hf_cache_dir"))
    prefix = pre["t5_prefix"] if kind == "t5" else ""
    t0 = time.time()
    rest = (kind, tok, names[kind], cfg["max_input"], cfg["max_target"], prefix, pre["target_join"])
    tr, hit1 = load_or_build(data_dir, cache_dir, cfg["source"], "train", *rest)
    va, hit2 = load_or_build(data_dir, cache_dir, cfg["source"], "validation", *rest)
    refs = [r["highlights"] for r in load_records(data_dir, cfg["source"], "validation")]
    if a.limit_train:
        tr = tr[:a.limit_train]
    if a.limit_val:
        va, refs = va[:a.limit_val], refs[:a.limit_val]
    print(f"Nạp dữ liệu token hóa trong {time.time() - t0:.0f}s (cache: train={hit1}, validation={hit2})")
    out = cfg["output"]
    res = train(cfg, tr, va, refs, tok, a.log_dir or out["log_dir"], a.ckpt_dir or out["ckpt_dir"], a.resume)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
