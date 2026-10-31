"""Money helpers. All amounts are Decimal; rounding is half-up to cents."""
from decimal import ROUND_HALF_UP, Decimal

CENT = Decimal("0.01")


def to_decimal(text):
    return Decimal(str(text))


def round_money(amount):
    return amount.quantize(CENT, rounding=ROUND_HALF_UP)


def fmt_money(amount):
    return f"{round_money(amount):.2f}"
