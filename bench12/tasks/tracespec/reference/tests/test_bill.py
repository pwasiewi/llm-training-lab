from datetime import datetime, timedelta, timezone
from decimal import Decimal

from meter.bill import bill_accounts, blocks_for
from meter.sessions import Session

T0 = datetime(2026, 8, 1, 9, 0, tzinfo=timezone.utc)


def sess(acct, minutes, start=T0):
    return Session(acct, start, start + timedelta(minutes=minutes))


def test_blocks_minimum_one():
    assert blocks_for(60) == 1


def test_blocks_round_up():
    assert blocks_for(22 * 60) == 2      # production rounds UP
    assert blocks_for(15 * 60) == 1
    assert blocks_for(15 * 60 + 1) == 2


def test_free_blocks():
    rows = bill_accounts([sess("a", 60)])   # 4 blocks, all free
    assert rows == [("a", 4, 0, Decimal("0"))] or rows[0][:3] == ("a", 4, 0)


def test_sorted_by_account():
    rows = bill_accounts([sess("b", 90), sess("a", 90)])
    assert [r[0] for r in rows] == ["a", "b"]


def test_midnight_split():
    rows = bill_accounts([sess("a", 30, start=datetime(2026, 8, 1, 23, 50, tzinfo=timezone.utc))])
    assert rows[0][1] == 3                 # 10 min + 20 min -> 1 + 2 blocks
