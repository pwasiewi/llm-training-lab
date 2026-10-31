"""Run summary."""
from decimal import Decimal

from invoicer.money import fmt_money


def summary(invoices):
    total = sum((inv.total for inv in invoices), Decimal("0"))
    return f"SUMMARY\ninvoices: {len(invoices)}\ntotal: {fmt_money(total)} EUR\n"
