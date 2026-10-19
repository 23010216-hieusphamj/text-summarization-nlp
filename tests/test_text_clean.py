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


def test_cnn_dateline_mixed_case_real_examples():
    cases = {
        "Las Vegas (CNN) -- The makers of Gorilla Glass, the glass that's used in gadgets.": "The makers of Gorilla Glass, the glass that's used in gadgets.",
        "County Kildare, Ireland (CNN) -- The mayor of a town has resigned.": "The mayor of a town has resigned.",
        "Abu Dhabi, United Arab Emirates (CNN)Lost luggage after a long flight is common.": "Lost luggage after a long flight is common.",
        "Kiev, Ukraine (CNN)The bodies of all 33 workers have been recovered.": "The bodies of all 33 workers have been recovered.",
        "Washington (CNN)Hillary Clinton is pressing the reset button.": "Hillary Clinton is pressing the reset button.",
        "Los Angeles (CNN) -- \"Teflon Don\" rapper crashed his car.": "\"Teflon Don\" rapper crashed his car.",
        "Rio de Janeiro, Brazil (CNN) -- Crowds gathered on Sunday.": "Crowds gathered on Sunday.",
    }
    for raw, want in cases.items():
        assert tc.clean_article(raw) == want, raw


def test_cnn_mention_in_a_normal_sentence_is_not_cut():
    for s in ["Officials told (CNN) that the plan is ready.", "The mayor of Los Angeles (CNN) said nothing.",
              "Kate Middleton told (CNN) she was pleased."]:
        assert tc.clean_article(s) == s


def test_daily_mail_byline_with_lowercase_words():
    a = "By . Nick Harris for MailOnline . Stevan Jovetic lit up the Etihad Stadium on Monday evening."
    b = "By . Name One . and Name Two . Police said the suspect fled the scene on Friday."
    assert tc.clean_article(a) == "Stevan Jovetic lit up the Etihad Stadium on Monday evening."
    assert tc.clean_article(b) == "Police said the suspect fled the scene on Friday."
