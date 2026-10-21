from src.insight_agent import ask, validate_numbers


def test_answerable():
    r = ask("Is the air in Lahore safe for a morning walk?")
    assert r.answered and r.citations and r.gold_row_ids


def test_unsupported_says_idk():
    r = ask("What is the capital of Mars?")
    assert not r.answered and r.gold_row_ids == []


def test_missing_city():
    r = ask("How bad is air pollution?")
    assert not r.answered and "city" in r.text


def test_urdu_question():
    r = ask("لاہور میں ہوا کیسی ہے؟")
    assert r.city == "Lahore"


def test_numbers_traceable():
    assert validate_numbers("average 41.2 over 24 hours", ["41.2", "24"])
    assert not validate_numbers("average 99 over 24 hours", ["41.2", "24"])


if __name__ == "__main__":
    for t in (test_answerable, test_unsupported_says_idk, test_missing_city,
              test_urdu_question, test_numbers_traceable):
        t()
        print("ok:", t.__name__)
    print("all tests passed")