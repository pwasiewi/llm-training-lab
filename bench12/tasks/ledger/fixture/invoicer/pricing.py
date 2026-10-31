"""Price an order: subtotal -> discount -> net -> VAT -> total."""
from decimal import Decimal

from invoicer.discounts import discount_amount
from invoicer.models import Invoice
from invoicer.money import round_money


def subtotal(order):
    return sum((l.unit_price * l.qty for l in order.lines), Decimal("0"))


def price(order, number, discounts):
    sub = round_money(subtotal(order))
    disc = discount_amount(order.discount, sub, discounts)
    net = sub - disc
    vat = round_money(net * order.vat_rate)
    return Invoice(number, order, sub, disc, net, vat, net + vat)


def number_invoices(orders):
    """INV-<year>-<seq>, seq per year in (date, order id) order."""
    out = []
    seq = {}
    for o in sorted(orders, key=lambda o: (o.date, o.id)):
        year = o.date[:4]
        seq[year] = seq.get(year, 0) + 1
        out.append((f"INV-{year}-{seq[year]:04d}", o))
    return out
