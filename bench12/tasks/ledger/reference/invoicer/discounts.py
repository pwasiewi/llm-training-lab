"""Discount codes: {"CODE": {"type": "fixed", "amount": "5.00"}} or {"type": "percent", "value": "10"}."""
from decimal import Decimal

from invoicer.money import round_money, to_decimal


class UnknownDiscount(Exception):
    pass


def discount_amount(code, subtotal, table, currency="EUR", eur_rate=Decimal(1)):
    """Fixed amounts are EUR; converted at eur_rate into the order currency."""
    if not code:
        return round_money(Decimal(0), currency)
    if code not in table:
        raise UnknownDiscount(code)
    d = table[code]
    if d["type"] == "fixed":
        return min(round_money(to_decimal(d["amount"]) / eur_rate, currency), subtotal)
    if d["type"] == "percent":
        return round_money(subtotal * to_decimal(d["value"]) / 100, currency)
    raise UnknownDiscount(code)
