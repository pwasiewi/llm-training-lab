"""Turn sessions into per-account billing rows."""
from collections import defaultdict

from meter.rates import BLOCK_SECONDS, FREE_BLOCKS, rate_for


def blocks_for(seconds):
    """Number of 15-minute blocks for a session of the given length (at least 1)."""
    return max(1, round(seconds / BLOCK_SECONDS))


def bill_accounts(sessions):
    """Return rows (account, blocks, billable_blocks, cost) sorted by account."""
    blocks = defaultdict(int)
    for s in sessions:
        blocks[s.account] += blocks_for(s.seconds)
    rows = []
    for account in sorted(blocks):
        billable = max(0, blocks[account] - FREE_BLOCKS)
        cost = float(rate_for(account)) * billable
        rows.append((account, blocks[account], billable, cost))
    return rows
