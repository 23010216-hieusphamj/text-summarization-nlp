from src.crawl import clean

CFG = {"lang": "en", "min_article_words": 20, "max_article_words": 500, "min_summary_words": 5,
       "max_summary_words": 50, "min_compression": 2.0, "max_title_jaccard": 0.8}

PARA = ("The government has announced a new programme to support small businesses across the country. "
        "Ministers said the funding will help companies invest in new equipment and train their staff. "
        "The scheme will open for applications next month and run for three years. ")


def test_html_to_text_cuts_tail_and_headings():
    html = ("<h2>Overview</h2><p>First paragraph here. Visit https://example.com now.</p>"
            "<table><tr><td>junk</td></tr></table><ul><li>Item one</li></ul>"
            "<h2>Notes to editors</h2><p>Contact press@gov.uk</p>")
    t = clean.html_to_text(html)
    assert "First paragraph here." in t and "Item one" in t
    assert "junk" not in t and "Overview" not in t
    assert "Contact" not in t and "example.com" not in t


def _rec(i, article=PARA * 2, summ="Government launches new support scheme for small businesses.", title="Different title"):
    return {"id": f"/g/{i}", "title": title, "article": article, "highlights": summ, "doc_type": "news_story", "date": "2024-01-01"}


def test_filters_funnel_and_reasons():
    recs = [
        _rec(1),
        _rec(2, article=""),                                         # thiếu nội dung
        _rec(3, article="Short text only."),                          # bài quá ngắn
        _rec(4, summ="Too short"),                                    # tóm tắt quá ngắn
        _rec(5, summ="Different title today for everyone", title="Different title today for everyone"),  # trùng tiêu đề
        _rec(6, summ=PARA.split(".")[0] + "."),                       # chép nguyên văn từ bài
        _rec(1),                                                      # trùng hoàn toàn bản ghi 1
    ]
    recs[-1]["id"] = "/g/7"
    kept, funnel = clean.apply_filters(recs, CFG)
    assert [r["id"] for r in kept] == ["/g/1"]
    counts = [c for _, c in funnel]
    assert counts[0] == 7 and counts == sorted(counts, reverse=True)


def test_split_reproducible_and_disjoint():
    recs = [_rec(i) for i in range(100)]
    a = clean.split_records(recs, {"train": 0.8, "validation": 0.1, "test": 0.1}, 42)
    b = clean.split_records(recs, {"train": 0.8, "validation": 0.1, "test": 0.1}, 42)
    assert [r["id"] for r in a["train"]] == [r["id"] for r in b["train"]]
    ids = [r["id"] for s in a.values() for r in s]
    assert len(ids) == len(set(ids)) == 100 and len(a["train"]) == 80


def test_near_copy_of_one_sentence_is_dropped():
    first = "The government has announced a new programme to support small businesses across the country"
    near_copy = first + " today"                                  # gần như chép nguyên câu đầu
    kept, funnel = clean.apply_filters([_rec(1, summ=near_copy), _rec(2)], CFG)
    assert [r["id"] for r in kept] == ["/g/2"]
    assert any("gần như chép" in name for name, _ in funnel)
    kept2, _ = clean.apply_filters([_rec(1, summ=near_copy)], {**CFG, "max_copy_overlap": None})   # tắt bước lọc
    assert len(kept2) == 1
