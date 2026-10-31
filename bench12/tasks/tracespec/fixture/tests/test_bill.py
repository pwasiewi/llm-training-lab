from datetime import datetime, timedelta, timezone

from meter.bill import bill_accounts, blocks_for
from meter.sessions import Session

T0 = datetime(2026, 8, 1, 9, 0, tzinfo=timezone.utc)


def sess(acct, minutes, start=T0):
    return Session(acct, start, start + timedelta(minutes=minutes))


def test_blocks_minimum_one():
    assert blocks_for(60) == 1


def test_blocks_round_to_nearest():
    assert blocks_for(22 * 60) == 1      # 22 min is closer to one block (README)
    assert blocks_for(23 * 60) == 2


def test_free_blocks():
    rows = bill_accounts([sess("a", 60)])   # 4 blocks, all free
    assert rows == [("a", 4, 0, 0.0)]


def test_sorted_by_account():
    rows = bill_accounts([sess("b", 90), sess("a", 90)])
    assert [r[0] for r in rows] == ["a", "b"]
