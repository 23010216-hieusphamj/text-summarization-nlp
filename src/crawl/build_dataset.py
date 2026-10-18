"""Đọc dữ liệu thô đã crawl -> làm sạch -> lọc -> chia tập -> lưu.

    python -m src.crawl.build_dataset --config configs/data_govuk.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from src.crawl.clean import apply_filters, build_records, split_records


def load_raw(raw_dir: Path) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(raw_dir.glob("*.json"))]


def write_jsonl(records, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main(config_path: str) -> dict:
    cfg = yaml.safe_load(open(config_path, encoding="utf-8"))
    raw = load_raw(Path(cfg["raw_dir"]))
    if not raw:
        raise SystemExit(f"Không có dữ liệu thô trong {cfg['raw_dir']}. Hãy chạy crawler trước.")
    records = build_records(raw)
    kept, funnel = apply_filters(records, cfg["clean"])

    tables = Path(cfg["tables_dir"]); tables.mkdir(parents=True, exist_ok=True)
    fdf = pd.DataFrame(funnel, columns=["buoc", "con_lai"])
    fdf["bi_loai"] = (-fdf["con_lai"].diff()).fillna(0).astype(int)
    fdf.to_csv(tables / "govuk_cleaning_funnel.csv", index=False)
    print(fdf.to_string(index=False))

    splits = split_records(kept, cfg["split"], cfg["seed"])
    out = Path(cfg["processed_dir"])
    for s, recs in splits.items():
        write_jsonl(recs, out / f"govuk_{s}.jsonl")
    write_jsonl(splits["train"][:20], Path(cfg["sample_dir"]) / "govuk_sample.jsonl")
    pd.DataFrame([{"split": s, "n": len(r)} for s, r in splits.items()]).to_csv(tables / "govuk_split_sizes.csv", index=False)
    print({s: len(r) for s, r in splits.items()})
    return splits


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/data_govuk.yaml")
    main(ap.parse_args().config)
