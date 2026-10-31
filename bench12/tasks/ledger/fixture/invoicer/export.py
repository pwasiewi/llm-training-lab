"""CSV export for the accounting system."""
import csv

from invoicer.money import fmt_money

COLUMNS = ["invoice_id", "order_id", "date", "customer", "net", "vat", "total"]


def write_export(path, invoices):
    with open(path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(COLUMNS)
        for inv in invoices:
            w.writerow([inv.number, inv.order.id, inv.order.date, inv.order.customer,
                        fmt_money(inv.net), fmt_money(inv.vat), fmt_money(inv.total)])
