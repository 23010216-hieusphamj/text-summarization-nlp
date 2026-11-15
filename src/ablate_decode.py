"""Ablation tham số giải mã trên VALIDATION (D10, nhận xét 3 của giảng viên): đổi từng tham số một quanh cấu hình gốc.

    python -m src.ablate_decode --key cnn_dm --data_dir <clean> --cache_dir <hf_cache> --ckpt_dir <checkpoints> \\
        --predictions_dir <...> --tables_dir <...> --scores_dir <...>
    python -m src.ablate_decode --key govuk ...                 # GOV.UK riêng (max_length ngắn hơn)
    python -m src.ablate_decode --key cnn_dm_t5 ...             # tùy chọn: T5-small
    python -m src.ablate_decode --key cnn_dm --combine ...      # thêm: chạy cấu hình GHÉP các giá trị thắng (mở rộng D10)

Danh sách cấu hình trong configs/decode_ablation.yaml. Mỗi cấu hình ghi một dòng vào results/tables/hf_{source}_validation.csv
(cùng bảng với 7a, khóa gen_tag); chạy lại sẽ BỎ QUA cấu hình đã có dòng (an toàn khi Colab ngắt, dùng --force để chạy lại).
Không BERTScore, không đo giờ (chỉ ROUGE để chọn); BERTScore và ms/bài chỉ đo ở lần chạy test cuối cùng. Chỉ dùng tập
validation: script từ chối chạy trên test.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
import yaml

from src.analysis.select_decode import CORE_KEYS, combine_winners, gen_flags, one_at_a_time_configs, select_decode
from src.data.hf_seq2seq_data import load_or_build, load_records, load_tokenizer
from src.decode_utils import table_path
from src.generate_hf import generate_and_evaluate, model_default_gen
from src.hf_train_utils import gen_tag


def run_ablation(model, tokenizer, examples, records, gens: list[dict], ev: dict, *, source: str, system: str,
                 split_tag: str, dirs: dict, batch_size: int, device, use_amp: bool, existing_tags: set[str],
                 seed: int = 42, extra: dict | None = None, force: bool = False) -> list[str]:
    """Chạy lần lượt các cấu hình giải mã; bỏ qua cấu hình đã có trong bảng (trừ khi force). Trả về các gen_tag đã chạy mới."""
    done = []
    for i, gen in enumerate(gens, 1):
        tag = gen_tag(gen)
        if tag in existing_tags and not force:
            print(f"[{i}/{len(gens)}] {tag}: đã có, bỏ qua")
            continue
        t0 = time.time()
        row = generate_and_evaluate(model, tokenizer, examples, records, gen, ev, source=source, system=system,
                                    split_tag=split_tag, dirs=dirs, batch_size=batch_size, device=device,
                                    use_amp=use_amp, no_bertscore=True, time_docs=0, seed=seed, extra=extra)
        S = 100 * (row["rouge1"] + row["rouge2"] + row["rougeLsum"]) / 3
        print(f"[{i}/{len(gens)}] {tag}: S={S:.2f} (R1 {100 * row['rouge1']:.1f}/R2 {100 * row['rouge2']:.1f}/"
              f"Lsum {100 * row['rougeLsum']:.1f}) | {row['avg_pred_words']:.1f} từ | {time.time() - t0:.0f}s")
        done.append(tag)
    return done


def report(tables_dir: str | Path, source: str, system: str, split_tag: str, base: dict, key: str) -> dict:
    """Đọc bảng validation của hệ thống, chọn theo quy tắc D10, in kết quả và ghi decode_selection_{key}.csv."""
    df = pd.read_csv(table_path(tables_dir, "hf", source, split_tag))
    df = df[df["system"] == system]
    res = select_decode(df, base)
    t = res["table"].sort_values("S", ascending=False)
    cols = ["gen_tag", "n_changed", "S", "rouge1", "rouge2", "rougeLsum", "avg_pred_words", "rep_trigram_pct", "is_best", "is_chosen"]
    t = t[[c for c in cols if c in t.columns]].copy()
    t["S"] = (100 * t["S"]).round(3)
    out = Path(tables_dir) / f"decode_selection_{key}.csv"
    t.to_csv(out, index=False)
    print(t.round(4).to_string(index=False))
    print(f"\nS cấu hình gốc = {100 * res['base_S']:.3f}; S tốt nhất = {100 * res['best_S']:.3f} ({res['best_tag']}); "
          f"CHỌN = {res['chosen_tag']} (theo quy tắc hòa 0,2 điểm của D10)")
    print("Tham số để chạy test một lần:", gen_flags(res["chosen_gen"]))
    print("Đã lưu", out)
    return res


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--key", required=True, help="mục trong configs/decode_ablation.yaml (cnn_dm | govuk | cnn_dm_t5)")
    ap.add_argument("--ablation_config", default="configs/decode_ablation.yaml")
    ap.add_argument("--ckpt_dir", required=True, help="thư mục chứa {run_name}/best.pt")
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--hf_cache_dir", default=None)
    ap.add_argument("--eval_config", default="configs/baselines.yaml")
    ap.add_argument("--predictions_dir", default="results/predictions")
    ap.add_argument("--tables_dir", default="results/tables")
    ap.add_argument("--scores_dir", default="results/scores")
    ap.add_argument("--eval_batch_size", type=int, default=16)
    ap.add_argument("--combine", action="store_true", help="chạy thêm cấu hình ghép các giá trị thắng (mở rộng D10)")
    ap.add_argument("--force", action="store_true", help="chạy lại cả cấu hình đã có dòng trong bảng")
    ap.add_argument("--limit", type=int, default=None, help="chỉ N bài đầu để chạy thử (đầu ra có hậu tố _limitN)")
    a = ap.parse_args(argv)

    entry = yaml.safe_load(open(a.ablation_config, encoding="utf-8"))[a.key]
    cfg = yaml.safe_load(open(entry["run_config"], encoding="utf-8"))
    evcfg = yaml.safe_load(open(a.eval_config, encoding="utf-8"))
    pre = yaml.safe_load(open(cfg["preprocess_config"], encoding="utf-8"))
    base, source, kind, names = dict(entry["base"]), cfg["source"], cfg["kind"], pre["tokenizers"]
    if any(float(base[k]) != float(cfg["val_gen"][k]) for k in CORE_KEYS):
        raise SystemExit("Cấu hình gốc trong decode_ablation.yaml khác val_gen của config huấn luyện: kiểm tra lại")
    system, split_tag = cfg["run_name"], "validation" if a.limit is None else f"validation_limit{a.limit}"
    data_dir, cache_dir = a.data_dir or cfg["data_dir"], a.cache_dir or cfg["cache_dir"]

    tok = load_tokenizer(names[kind], a.hf_cache_dir)
    prefix = pre["t5_prefix"] if kind == "t5" else ""
    examples, _ = load_or_build(data_dir, cache_dir, source, "validation", kind, tok, names[kind], cfg["max_input"],
                                cfg["max_target"], prefix, pre["target_join"])
    records = load_records(data_dir, source, "validation")
    if a.limit:
        examples, records = examples[:a.limit], records[:a.limit]
    if [e["id"] for e in examples] != [r["id"] for r in records]:
        raise SystemExit("ID dữ liệu token hóa không khớp jsonl: xóa cache rồi chạy lại hf_seq2seq_data")

    from transformers import AutoModelForSeq2SeqLM
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(cfg["train"].get("amp", False)) and device.type == "cuda"
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg["model_name"], cache_dir=a.hf_cache_dir)
    ck = torch.load(Path(a.ckpt_dir) / entry["ckpt_run"] / "best.pt", map_location="cpu", weights_only=False)
    if ck.get("model_name") != cfg["model_name"]:
        raise SystemExit(f"Checkpoint của {ck.get('model_name')} nhưng config dùng {cfg['model_name']}")
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    extra = {"max_input": cfg["max_input"], "zero_shot": False, "ckpt_epoch": int(ck["epoch"]), "amp": use_amp,
             "model_default_gen": json.dumps(model_default_gen(model), ensure_ascii=False)}
    dirs = {"predictions": a.predictions_dir, "tables": a.tables_dir, "scores": a.scores_dir}

    def existing() -> set[str]:
        p = table_path(a.tables_dir, "hf", source, split_tag)
        if not Path(p).exists():
            return set()
        d = pd.read_csv(p)
        return set(d.loc[d["system"] == system, "gen_tag"])

    gens = one_at_a_time_configs(base, entry["factors"])
    print(f"{system} | {source}/validation ({len(records)} bài) | ckpt epoch {extra['ckpt_epoch']} | {len(gens)} cấu hình "
          f"(gốc + đổi từng tham số) | {device} AMP={use_amp}")
    common = dict(source=source, system=system, split_tag=split_tag, dirs=dirs, batch_size=a.eval_batch_size, device=device,
                  use_amp=use_amp, seed=evcfg.get("seed", 42), extra=extra, force=a.force)
    run_ablation(model, tok, examples, records, gens, evcfg["evaluate"], existing_tags=existing(), **common)
    res = report(a.tables_dir, source, system, split_tag, base, a.key)
    if a.combine:
        df = pd.read_csv(table_path(a.tables_dir, "hf", source, split_tag))
        combo = combine_winners(df[df["system"] == system], base)
        if combo is None:
            print("\n--combine: không có tham số nào thắng cấu hình gốc đủ ngưỡng 0,2 điểm, không có cấu hình ghép.")
        else:
            print("\n--combine: cấu hình ghép các giá trị thắng:", gen_flags(combo))
            run_ablation(model, tok, examples, records, [combo], evcfg["evaluate"], existing_tags=existing(), **common)
            report(a.tables_dir, source, system, split_tag, base, a.key)


if __name__ == "__main__":
    main()
