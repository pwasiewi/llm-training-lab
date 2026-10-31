"""Discount codes: {"CODE": {"type": "fixed", "amount": "5.00"}} or {"type": "percent", "value": "10"}."""
from decimal import Decimal

from invoicer.money import round_money, to_decimal


class UnknownDiscount(Exception):
    pass


def discount_amount(code, subtotal, table):
    if not code:
        return Decimal("0.00")
    if code not in table:
        raise UnknownDiscount(code)
    d = table[code]
    if d["type"] == "fixed":
        return min(to_decimal(d["amount"]), subtotal)
    if d["type"] == "percent":
        return round_money(subtotal * to_decimal(d["value"]) / 100)
    raise UnknownDiscount(code)
