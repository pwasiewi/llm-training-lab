"""python -m invoicer run --orders F --discounts F --out DIR"""
import argparse
import os
import sys

from invoicer.discounts import UnknownDiscount
from invoicer.money import UnknownCurrency, decimals
from invoicer.rates import MissingRate, Rates
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
    run.add_argument("--rates")
    run.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    orders = load_orders(args.orders)
    discounts = load_discounts(args.discounts)
    rates = Rates.load(args.rates)
    try:
        for o in orders:
            decimals(o.currency)
        invoices = [price(o, num, discounts, rates) for num, o in number_invoices(orders)]
    except UnknownCurrency as e:
        print(f"invoicer: unknown currency {e}", file=sys.stderr)
        return 3
    except MissingRate as e:
        print(f"invoicer: no rate for {e}", file=sys.stderr)
        return 3
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
