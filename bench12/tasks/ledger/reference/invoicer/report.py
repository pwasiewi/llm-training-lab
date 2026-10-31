"""Run summary."""
from decimal import Decimal

from invoicer.money import fmt_money


def summary(invoices):
    by = {}
    for inv in invoices:
        n, s = by.get(inv.order.currency, (0, Decimal(0)))
        by[inv.order.currency] = (n + 1, s + inv.total)
    out = ["SUMMARY", f"invoices: {len(invoices)}", "by currency:"]
    for cur in sorted(by):
        out.append(f"  {cur} {by[cur][0]} {fmt_money(by[cur][1], cur)}")
    total = sum((inv.total_eur for inv in invoices), Decimal(0))
    out.append(f"total: {fmt_money(total)} EUR")
    return "\n".join(out) + "\n"
