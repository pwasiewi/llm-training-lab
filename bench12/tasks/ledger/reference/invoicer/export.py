"""CSV export for the accounting system."""
import csv

from invoicer.money import fmt_money

COLUMNS = ["invoice_id", "currency", "order_id", "date", "customer", "net", "vat", "total", "total_eur"]


def write_export(path, invoices):
    with open(path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(COLUMNS)
        for inv in invoices:
            c = inv.order.currency
            w.writerow([inv.number, c, inv.order.id, inv.order.date, inv.order.customer,
                        fmt_money(inv.net, c), fmt_money(inv.vat, c), fmt_money(inv.total, c),
                        fmt_money(inv.total_eur)])
