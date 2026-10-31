"""Turn sessions into per-account billing rows."""
import math
from collections import defaultdict
from datetime import datetime, timedelta, timezone

from meter.rates import BLOCK_SECONDS, FREE_BLOCKS, rate_for


def blocks_for(seconds):
    """15-minute blocks for a session part: rounded UP, at least 1."""
    return max(1, math.ceil(seconds / BLOCK_SECONDS))


def day_parts(start, end):
    """Split [start, end] at UTC midnights -> [(day, seconds), ...]."""
    parts = []
    cur = start
    while True:
        nxt = datetime.combine(cur.date() + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc)
        stop = min(end, nxt)
        parts.append((cur.date(), (stop - cur).total_seconds()))
        if end <= nxt:
            return parts
        cur = nxt


def bill_accounts(sessions):
    """Rows (account, blocks, billable_blocks, cost) sorted by account; cost is a Decimal."""
    per_day = defaultdict(lambda: defaultdict(int))
    for s in sessions:
        for day, secs in day_parts(s.start, s.end):
            per_day[s.account][day] += blocks_for(secs)
    rows = []
    for account in sorted(per_day):
        days = per_day[account]
        total = sum(days.values())
        billable = sum(max(0, b - FREE_BLOCKS) for b in days.values())
        rows.append((account, total, billable, rate_for(account) * billable))
    return rows
