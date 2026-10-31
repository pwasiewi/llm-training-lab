"""Price per billable block."""
from decimal import Decimal

BLOCK_SECONDS = 15 * 60
FREE_BLOCKS = 4
STANDARD = Decimal("0.125")
ENTERPRISE = Decimal("0.085")


def rate_for(account):
    return ENTERPRISE if account.startswith("ent-") else STANDARD
