from decimal import Decimal

from invoicer.models import Line, Order
from invoicer.pricing import number_invoices, price

DISC = {"WELCOME5": {"type": "fixed", "amount": "5.00"}, "TENPCT": {"type": "percent", "value": "10"}}


def order(**kw):
    base = dict(id="A1", date="2026-09-01", customer="Acme", lines=[Line("X1", 3, Decimal("19.99"))],
                vat_rate=Decimal("0.19"))
    base.update(kw)
    return Order(**base)


def test_plain_order():
    inv = price(order(), "INV-2026-0001", DISC)
    assert (inv.subtotal, inv.net, inv.vat, inv.total) == (Decimal("59.97"), Decimal("59.97"),
                                                          Decimal("11.39"), Decimal("71.36"))


def test_fixed_discount():
    inv = price(order(discount="WELCOME5"), "INV-2026-0001", DISC)
    assert inv.discount == Decimal("5.00") and inv.total == Decimal("65.41")


def test_percent_discount_rounds_half_up():
    inv = price(order(lines=[Line("X2", 1, Decimal("0.25"))], discount="TENPCT"), "N", DISC)
    assert inv.discount == Decimal("0.03")


def test_numbering_by_date_then_id():
    nums = number_invoices([order(id="B", date="2026-09-02"), order(id="A", date="2026-09-02"),
                            order(id="C", date="2026-08-30")])
    assert [(n, o.id) for n, o in nums] == [("INV-2026-0001", "C"), ("INV-2026-0002", "A"), ("INV-2026-0003", "B")]
