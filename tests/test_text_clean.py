from src.preprocessing import text_clean as tc


def test_daily_mail_header_removed():
    s = ("By . Daily Mail Reporter . PUBLISHED: . 07:12 EST, 28 February 2013 . | . UPDATED: . 07:34 EST, "
         "28 February 2013 . A man has been arrested after a robbery in London .")
    assert tc.clean_article(s) == "A man has been arrested after a robbery in London ."


def test_daily_mail_header_published_only_and_last_updated():
    a = "By . Associated Press . PUBLISHED: . 14:11 EST, 25 October 2013 . Police said the suspect fled ."
    b = "Last updated at 9:48 AM on 11th January 2012 . Heavy rain hit the north on Tuesday ."
    assert tc.clean_article(a) == "Police said the suspect fled ."
    assert tc.clean_article(b) == "Heavy rain hit the north on Tuesday ."


def test_cnn_dateline_variants():
    assert tc.clean_article("LONDON, England (CNN) -- The prime minister said on Monday that talks continue.") == \
        "The prime minister said on Monday that talks continue."
    assert tc.clean_article("(CNN) -- Officials confirmed the news.") == "Officials confirmed the news."
    assert tc.clean_article("(CNN)Officials confirmed the news.") == "Officials confirmed the news."


def test_mid_text_cnn_mention_is_kept():
    s = "Officials spoke to (CNN) on Monday about the plan."
    assert tc.clean_article(s) == s


def test_inline_junk_and_urls_removed():
    s = "Rain fell all day. Scroll down for video. See http://example.com/x for details. E-mail to a friend ."
    out = tc.clean_article(s)
    assert "Scroll down" not in out and "http" not in out and "E-mail" not in out
    assert out.startswith("Rain fell all day.")


def test_clean_text_without_junk_is_unchanged_except_whitespace():
    s = "A normal sentence.   Another\none."
    assert tc.clean_article(s) == "A normal sentence. Another one."


def test_highlights_keep_newlines():
    assert tc.clean_highlights("First point.\n\n  Second  point.\n") == "First point.\nSecond point."
