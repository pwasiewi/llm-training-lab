"""Report rendering (money: Decimal, half-up to cents; TOTAL = sum of printed amounts)."""
from decimal import ROUND_HALF_UP, Decimal

CENT = Decimal("0.01")


def money(x):
    return Decimal(x).quantize(CENT, ROUND_HALF_UP)


def render(rows):
    amounts = [money(cost) for *_, cost in rows]
    lines = [f"{acct} {blocks} {billable} {amt}" for (acct, blocks, billable, _), amt in zip(rows, amounts)]
    lines.append(f"TOTAL {sum(amounts, Decimal(0))}")
    return "\n".join(lines) + "\n"
