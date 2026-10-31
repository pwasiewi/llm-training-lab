from meter.parse import parse_events, parse_line


def test_parse_line_fields():
    ev = parse_line("2026-08-01T09:00:00Z e1 ACME start s1")
    assert ev.event_id == "e1"
    assert ev.account == "ACME"          # account ids are case-sensitive (README)
    assert ev.kind == "start"
    assert ev.session_id == "s1"


def test_comments_and_blank_lines_skipped():
    evs = parse_events(["# header", "", "2026-08-01T09:00:00Z e1 a start s1"])
    assert len(evs) == 1


def test_bad_kind_rejected():
    import pytest
    with pytest.raises(ValueError):
        parse_line("2026-08-01T09:00:00Z e1 a pause s1")
