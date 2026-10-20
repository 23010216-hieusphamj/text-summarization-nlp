"""Huấn luyện Seq2Seq + attention từ đầu (GĐ5, D9).

    python -m src.train_seq2seq --config configs/seq2seq.yaml --data_dir <thư mục clean> --ckpt_dir <thư mục trên Drive>
    python -m src.train_seq2seq ... --limit_train 500 --limit_val 200 --epochs 1 --run_name smoke   # chạy thử nhanh
    python -m src.train_seq2seq ... --resume                                                         # tiếp tục sau khi Colab ngắt

Đầu ra: results/logs/{run}_log.csv (theo epoch), {run}_steps.csv (theo bước), {run}_meta.json;
checkpoint {ckpt_dir}/{run}/best.pt và last.pt (nên trỏ ckpt_dir ra Drive, không commit).
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
import torch.nn.functional as F
import yaml

from src.data.seq2seq_data import load_examples, make_loader
from src.models.seq2seq import Seq2Seq
from src.preprocessing.spm_tokenizer import PAD_ID, load_spm
from src.train_utils import EarlyStopper, append_row


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def loss_sum_and_count(logits: torch.Tensor, targets: torch.Tensor, pad_id: int = PAD_ID):
    """Tổng cross-entropy trên các token không phải PAD và số token đó."""
    s = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), targets.reshape(-1),
                        ignore_index=pad_id, reduction="sum")
    return s, (targets != pad_id).sum()


def make_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def to_device(batch: dict, device) -> tuple:
    # src_lens giữ trên CPU (pack_padded_sequence yêu cầu)
    return (batch["src"].to(device), batch["src_lens"],
            batch["tgt_in"].to(device), batch["tgt_out"].to(device))


@torch.no_grad()
def evaluate(model, loader, device, use_amp: bool) -> float:
    """Val loss (teacher forcing), trung bình theo token."""
    model.eval()
    tot, cnt = 0.0, 0
    for batch in loader:
        src, lens, tgt_in, tgt_out = to_device(batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(src, lens, tgt_in)
        s, n = loss_sum_and_count(logits, tgt_out)
        tot += s.item(); cnt += int(n)
    return tot / max(cnt, 1)


def _save(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)   # ghi nguyên tử, tránh file hỏng nếu Colab ngắt giữa chừng


def train(cfg: dict, train_ex: list[dict], val_ex: list[dict], vocab_size: int, run_name: str,
          log_dir: str | Path, ckpt_dir: str | Path, resume: bool = False, device: str | None = None) -> dict:
    tc, mc = cfg["train"], cfg["model"]
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    use_amp = bool(tc.get("amp", True)) and device.type == "cuda"
    set_seed(tc["seed"])

    model = Seq2Seq(vocab_size, **mc).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=tc["lr"])
    scaler = make_scaler(use_amp)
    stopper = EarlyStopper(tc["patience"])
    train_loader = make_loader(train_ex, tc["batch_size"], shuffle=True, seed=tc["seed"], num_workers=tc.get("num_workers", 0))
    val_loader = make_loader(val_ex, tc["batch_size"], shuffle=False)

    log_dir, run_dir = Path(log_dir), Path(ckpt_dir) / run_name
    epoch_csv, step_csv, meta_json = (log_dir / f"{run_name}_log.csv", log_dir / f"{run_name}_steps.csv",
                                      log_dir / f"{run_name}_meta.json")
    start_epoch, global_step, train_seconds = 1, 0, 0.0
    if resume and (run_dir / "last.pt").exists():
        ck = torch.load(run_dir / "last.pt", map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); scaler.load_state_dict(ck["scaler"])
        stopper.load_state_dict(ck["stopper"])
        start_epoch, global_step, train_seconds = ck["epoch"] + 1, ck["global_step"], ck["train_seconds"]
        print(f"Tiếp tục từ epoch {start_epoch} (best val {stopper.best:.4f} ở epoch {stopper.best_epoch})")
    elif epoch_csv.exists() or step_csv.exists():
        raise FileExistsError(f"Đã có log của run '{run_name}'. Dùng --resume hoặc đổi --run_name.")

    n_params = model.num_parameters()
    print(f"Run {run_name}: {n_params:,} tham số, thiết bị {device}, AMP={use_amp}, "
          f"{len(train_ex):,} train / {len(val_ex):,} val")
    log_dir.mkdir(parents=True, exist_ok=True)
    meta = {"run": run_name, "started": datetime.now().isoformat(timespec="seconds"), "seed": tc["seed"],
            "n_params": n_params, "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "n_train": len(train_ex), "n_val": len(val_ex), "config": cfg}

    stopped_early = False
    for epoch in range(start_epoch, tc["epochs"] + 1):
        model.train()
        t0, tot, cnt = time.time(), 0.0, 0
        for batch in train_loader:
            src, lens, tgt_in, tgt_out = to_device(batch, device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(src, lens, tgt_in)
            s, n = loss_sum_and_count(logits, tgt_out)
            loss = s / n
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc["clip_norm"])
            scaler.step(opt); scaler.update()
            global_step += 1
            tot += s.item(); cnt += int(n)
            if global_step % tc.get("log_every", 50) == 0:
                append_row(step_csv, {"step": global_step, "epoch": epoch, "loss": round(loss.item(), 5),
                                      "grad_norm": round(float(gnorm), 4)})
        val_loss = evaluate(model, val_loader, device, use_amp)
        epoch_s = time.time() - t0
        train_seconds += epoch_s
        row = {"epoch": epoch, "train_loss": round(tot / max(cnt, 1), 5), "val_loss": round(val_loss, 5),
               "val_ppl": round(math.exp(min(val_loss, 20)), 3), "lr": tc["lr"],
               "epoch_seconds": round(epoch_s, 1), "train_seconds_total": round(train_seconds, 1)}
        append_row(epoch_csv, row)
        improved = stopper.update(val_loss, epoch)
        state = {"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                 "stopper": stopper.state_dict(), "epoch": epoch, "global_step": global_step,
                 "train_seconds": train_seconds, "vocab_size": vocab_size, "model_cfg": mc}
        _save(run_dir / "last.pt", state)
        if improved:
            _save(run_dir / "best.pt", {"model": model.state_dict(), "epoch": epoch, "val_loss": val_loss,
                                        "vocab_size": vocab_size, "model_cfg": mc})
        print(f"epoch {epoch:2d} | train {row['train_loss']:.4f} | val {row['val_loss']:.4f} "
              f"(ppl {row['val_ppl']:.1f}){' *' if improved else ''} | {epoch_s:.0f}s")
        if stopper.should_stop:
            stopped_early = True
            print(f"Dừng sớm: val loss không cải thiện {stopper.patience} epoch liên tiếp.")
            break

    result = {"best_val_loss": stopper.best, "best_epoch": stopper.best_epoch, "stopped_early": stopped_early,
              "train_seconds": round(train_seconds, 1), "n_params": n_params, "global_step": global_step}
    meta.update(result, finished=datetime.now().isoformat(timespec="seconds"))
    json.dump(meta, open(meta_json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    return result


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/seq2seq.yaml")
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--ckpt_dir", default=None, help="nơi lưu checkpoint (nên là thư mục trên Drive)")
    ap.add_argument("--run_name", default=None)
    ap.add_argument("--limit_train", type=int, default=None)
    ap.add_argument("--limit_val", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    if a.epochs:
        cfg["train"]["epochs"] = a.epochs
    data_dir = a.data_dir or cfg["data_dir"]
    run_name = a.run_name or f"seq2seq_in{cfg['max_input']}"
    sp = load_spm(cfg["spm_model"])
    kw = dict(sp=sp, max_input=cfg["max_input"], max_target=cfg["max_target"])
    t0 = time.time()
    tr = load_examples(data_dir, cfg["train_source"], "train", limit=a.limit_train, **kw)
    va = load_examples(data_dir, cfg["train_source"], "validation", limit=a.limit_val, **kw)
    print(f"Đã mã hóa dữ liệu trong {time.time() - t0:.0f}s")
    out = cfg["output"]
    res = train(cfg, tr, va, sp.get_piece_size(), run_name, out["log_dir"], a.ckpt_dir or out["ckpt_dir"], a.resume)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
