"""Kiểm thử luồng crawl -> build bằng 'máy chủ giả' (không cần mạng)."""
import json
from pathlib import Path

import yaml

from src.crawl import build_dataset, govuk_crawler as gc

BODY = ("<p>The department today announced new funding for local councils to repair roads and improve public transport links. "
        "Officials said work will begin in the spring and continue for two years across several regions. "
        "Residents will be consulted on the plans before any construction starts. "
        "The minister added that the investment reflects a long term commitment to better infrastructure for all communities. "
        "Councils must submit their proposals by the end of the summer to qualify for the first round of money.</p>")


class FakeResp:
    def __init__(self, status, payload=None, headers=None):
        self.status_code, self._p, self.headers = status, payload, headers or {}
    def json(self):
        return self._p


class FakeSession:
    def __init__(self, n=30):
        self.n, self.calls = n, 0
    def get(self, url, params=None, headers=None, timeout=None):
        self.calls += 1
        if url.endswith("/api/search.json"):
            start, count = params["start"], params["count"]
            win = params["filter_public_timestamp"].split(",")[0].replace("from:", "")   # mỗi tháng có link riêng
            allr = [{"link": f"/government/news/{win}-item-{i}", "title": f"T{i}"} for i in range(self.n)]
            return FakeResp(200, {"results": allr[start:start + count], "total": self.n})
        if "/api/content/" in url:
            i = url.rsplit("-", 1)[1]
            return FakeResp(200, {"title": f"Council funding {i}",
                                  "description": f"Government gives councils money for roads and transport number {i}.",
                                  "document_type": "news_story", "public_updated_at": "2024-03-01T10:00:00Z",
                                  "details": {"body": BODY + f"<p>Reference code {i} applies to this announcement only.</p>"}})
        return FakeResp(404)


def test_month_windows():
    w = gc.month_windows("2024-11-01", "2025-02-28")
    assert w[0] == ("2024-11-01", "2024-11-30") and w[-1] == ("2025-02-01", "2025-02-28") and len(w) == 4


def test_end_to_end(tmp_path, monkeypatch):
    cfg = yaml.safe_load(open("configs/data_govuk.yaml"))
    cfg.update(doc_types=["news_story"], date_from="2024-01-01", date_to="2024-02-29", per_window_per_type_max=10,
               max_docs=15, delay_sec=0, raw_dir=str(tmp_path / "raw"), processed_dir=str(tmp_path / "proc"),
               sample_dir=str(tmp_path / "samp"), tables_dir=str(tmp_path / "tab"))
    cfg["clean"].update(min_article_words=30, min_summary_words=5, min_compression=2.0)
    cp = tmp_path / "cfg.yaml"; cp.write_text(yaml.dump(cfg))
    sess = FakeSession(n=30)
    gc.main(str(cp), session=sess, skip_robots=True)
    assert len(list(Path(cfg["raw_dir"]).glob("*.json"))) == 15
    calls = sess.calls
    gc.main(str(cp), session=sess, skip_robots=True)                 # chạy lại: nội dung lấy từ cache
    assert sess.calls - calls < 15 + 10                               # chỉ gọi lại bước liệt kê
    splits = build_dataset.main(str(cp))
    total = sum(len(v) for v in splits.values())
    assert total > 0
    row = json.loads(Path(cfg["processed_dir"], "govuk_train.jsonl").read_text().splitlines()[0])
    assert {"id", "article", "highlights", "title", "doc_type", "date"} <= set(row)
    assert (tmp_path / "tab" / "govuk_cleaning_funnel.csv").exists()


class Flaky503Session:
    """Trả 503 kèm 'Retry-After: 0' hai lần rồi thành công."""
    def __init__(self):
        self.calls = 0
    def get(self, url, params=None, headers=None, timeout=None):
        self.calls += 1
        if self.calls <= 2:
            return FakeResp(503, None, {"Retry-After": "0"})
        return FakeResp(200, {"ok": True})


def test_503_backoff_ignores_zero_retry_after(monkeypatch):
    waits = []
    monkeypatch.setattr(gc.time, "sleep", lambda s: waits.append(s))
    c = gc.GovUKClient("https://x", "ua", delay=0, max_retries=5, session=Flaky503Session())
    assert c.get_json("/api/content/a") == {"ok": True}
    assert [w for w in waits if w > 0] == [2.0, 4.0]          # tăng dần, không chờ 0s


def test_retry_after_parsing():
    assert gc._retry_after({"Retry-After": "7"}) == 7.0
    assert gc._retry_after({"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"}) == 0.0
    assert gc._retry_after({}) == 0.0
