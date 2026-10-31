"""python -m invoicer run --orders F --discounts F --out DIR"""
import argparse
import os
import sys

from invoicer.discounts import UnknownDiscount
from invoicer.export import write_export
from invoicer.loader import load_discounts, load_orders
from invoicer.pricing import number_invoices, price
from invoicer.render import render_invoice
from invoicer.report import summary


def main(argv=None):
    ap = argparse.ArgumentParser(prog="invoicer")
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("--orders", required=True)
    run.add_argument("--discounts", required=True)
    run.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    orders = load_orders(args.orders)
    discounts = load_discounts(args.discounts)
    try:
        invoices = [price(o, num, discounts) for num, o in number_invoices(orders)]
    except UnknownDiscount as e:
        print(f"invoicer: unknown discount code {e}", file=sys.stderr)
        return 3
    os.makedirs(os.path.join(args.out, "invoices"), exist_ok=True)
    for inv in invoices:
        with open(os.path.join(args.out, "invoices", f"{inv.number}.txt"), "w") as f:
            f.write(render_invoice(inv))
    write_export(os.path.join(args.out, "export.csv"), invoices)
    with open(os.path.join(args.out, "summary.txt"), "w") as f:
        f.write(summary(invoices))
    return 0
