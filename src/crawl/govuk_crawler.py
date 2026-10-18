"""Thu thập bài tin + mô tả (tóm tắt do người viết) từ GOV.UK qua Search API và Content API.

Chạy thử nhanh (in cấu trúc dữ liệu, KHÔNG tải nhiều):
    python -m src.crawl.govuk_crawler --config configs/data_govuk.yaml --smoke
Chạy đầy đủ (có thể chạy lại để tiếp tục nhờ cache):
    python -m src.crawl.govuk_crawler --config configs/data_govuk.yaml
"""
from __future__ import annotations

import argparse
import calendar
import hashlib
import json
import random
import time
from datetime import date
from pathlib import Path
from urllib import robotparser

import requests
import yaml


class GovUKClient:
    def __init__(self, base_url, user_agent, delay=0.5, timeout=30, max_retries=5, session=None):
        self.base, self.delay, self.timeout, self.max_retries = base_url.rstrip("/"), delay, timeout, max_retries
        self.s = session or requests.Session()
        self.headers = {"User-Agent": user_agent, "Accept": "application/json"}

    def get_json(self, path: str, params: dict | None = None) -> dict | None:
        url = f"{self.base}{path}"
        for attempt in range(self.max_retries):
            try:
                r = self.s.get(url, params=params, headers=self.headers, timeout=self.timeout)
            except requests.RequestException as e:
                wait = 2 ** attempt
                print(f"  lỗi mạng ({e}); thử lại sau {wait}s")
                time.sleep(wait); continue
            time.sleep(self.delay)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return None
            if r.status_code in (429, 500, 502, 503, 504):
                wait = float(r.headers.get("Retry-After", 2 ** (attempt + 1)))
                print(f"  HTTP {r.status_code} ở {path}; chờ {wait:.0f}s")
                time.sleep(wait); continue
            print(f"  HTTP {r.status_code} ở {path}; bỏ qua"); return None
        return None


def month_windows(date_from: str, date_to: str) -> list[tuple[str, str]]:
    y, m = map(int, date_from.split("-")[:2])
    y2, m2 = map(int, date_to.split("-")[:2])
    out = []
    while (y, m) <= (y2, m2):
        last = calendar.monthrange(y, m)[1]
        out.append((date(y, m, 1).isoformat(), date(y, m, last).isoformat()))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def check_robots(base_url: str, user_agent: str, paths: list[str]) -> bool:
    rp = robotparser.RobotFileParser(f"{base_url}/robots.txt")
    try:
        rp.read()
    except Exception as e:
        print(f"Không đọc được robots.txt ({e}); tiếp tục thận trọng.")
        return True
    ok = True
    for p in paths:
        allowed = rp.can_fetch(user_agent, f"{base_url}{p}")
        print(f"robots.txt: {p} -> {'cho phép' if allowed else 'KHÔNG cho phép'}")
        ok &= allowed
    return ok


def list_links(client: GovUKClient, doc_type: str, window: tuple[str, str], cap: int, page_size: int = 100) -> list[dict]:
    items, start = [], 0
    while len(items) < cap:
        data = client.get_json("/api/search.json", {
            "filter_content_store_document_type": doc_type,
            "filter_public_timestamp": f"from:{window[0]},to:{window[1]}",
            "order": "-public_timestamp",
            "count": min(page_size, cap - len(items)), "start": start,
            "fields": "link,title,description,public_timestamp,content_store_document_type",
        })
        res = (data or {}).get("results", [])
        items += res
        if len(res) < page_size:
            break
        start += page_size
    return items[:cap]


def fetch_content(client: GovUKClient, link: str, raw_dir: Path) -> bool:
    out = raw_dir / (hashlib.sha1(link.encode()).hexdigest() + ".json")
    if out.exists():
        return True                                     # đã có trong cache
    data = client.get_json(f"/api/content{link}")
    if not data:
        return False
    rec = {
        "link": link, "title": data.get("title"), "description": data.get("description"),
        "document_type": data.get("document_type"),
        "public_updated_at": data.get("public_updated_at"),
        "body_html": (data.get("details") or {}).get("body"),
    }
    out.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
    return True


def smoke(client: GovUKClient, cfg: dict) -> None:
    dt = cfg["doc_types"][0]
    w = month_windows(cfg["date_from"], cfg["date_to"])[-3]
    items = list_links(client, dt, w, cap=3)
    print(f"[smoke] {dt} {w}: {len(items)} kết quả. Khóa trả về:", sorted(items[0]) if items else None)
    for it in items[:2]:
        d = client.get_json(f"/api/content{it['link']}")
        print("-", it["link"]); print("  title      :", (d or {}).get("title"))
        print("  description:", (d or {}).get("description"))
        body = ((d or {}).get("details") or {}).get("body") or ""
        print("  body[:150] :", body[:150].replace("\n", " "))


def main(config_path: str, do_smoke: bool = False, max_docs: int | None = None, session=None, skip_robots=False) -> None:
    cfg = yaml.safe_load(open(config_path, encoding="utf-8"))
    if max_docs:
        cfg["max_docs"] = max_docs
    if not skip_robots and not check_robots(cfg["base_url"], cfg["user_agent"], ["/api/search.json", "/api/content/"]):
        raise SystemExit("robots.txt không cho phép truy cập; dừng.")
    client = GovUKClient(cfg["base_url"], cfg["user_agent"], cfg["delay_sec"], cfg["timeout_sec"], cfg["max_retries"], session)
    if do_smoke:
        return smoke(client, cfg)

    raw_dir = Path(cfg["raw_dir"]); raw_dir.mkdir(parents=True, exist_ok=True)
    links = {}
    wins = month_windows(cfg["date_from"], cfg["date_to"])
    print(f"Bước 1: liệt kê bài ({len(cfg['doc_types'])} loại × {len(wins)} tháng)")
    for dt in cfg["doc_types"]:
        n_before = len(links)
        for w in wins:
            for it in list_links(client, dt, w, cfg["per_window_per_type_max"]):
                links.setdefault(it["link"], it)
        print(f"  {dt}: +{len(links) - n_before} bài")
    keys = sorted(links)
    random.Random(cfg["seed"]).shuffle(keys)
    keys = keys[: cfg["max_docs"]]
    print(f"Bước 2: tải nội dung {len(keys)} bài (cache: {raw_dir})")
    ok = 0
    for i, link in enumerate(keys, 1):
        ok += fetch_content(client, link, raw_dir)
        if i % 100 == 0:
            print(f"  {i}/{len(keys)} (thành công {ok})")
    print(f"Xong: {ok}/{len(keys)} bài trong {raw_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data_govuk.yaml")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max-docs", type=int, default=None)
    a = ap.parse_args()
    main(a.config, a.smoke, a.max_docs)
