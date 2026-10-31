from decimal import Decimal

from invoicer.models import Line, Order
from invoicer.pricing import price
from invoicer.render import render_invoice


def test_render_eur_invoice():
    o = Order("A100", "2026-09-01", "Acme GmbH", [Line("X1", 3, Decimal("19.99"))], Decimal("0.19"), "WELCOME5")
    inv = price(o, "INV-2026-0001", {"WELCOME5": {"type": "fixed", "amount": "5.00"}})
    assert render_invoice(inv) == (
        "INVOICE INV-2026-0001\n"
        "Order A100  Date 2026-09-01\n"
        "Customer: Acme GmbH\n"
        "  X1        3 x 19.99 = 59.97\n"
        "Subtotal: 59.97\n"
        "Discount: -5.00 (WELCOME5)\n"
        "Net: 54.97\n"
        "VAT 19%: 10.44\n"
        "Total: 65.41 EUR\n")
