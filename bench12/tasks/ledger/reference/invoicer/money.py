"""Money helpers. All amounts are Decimal; rounding is half-up to the currency's minor unit."""
from decimal import ROUND_HALF_UP, Decimal

CENT = Decimal("0.01")
DECIMALS = {"EUR": 2, "USD": 2, "GBP": 2, "CHF": 2, "JPY": 0, "KWD": 3}


class UnknownCurrency(Exception):
    pass


def to_decimal(text):
    return Decimal(str(text))


def decimals(currency):
    try:
        return DECIMALS[currency]
    except KeyError:
        raise UnknownCurrency(currency) from None


def round_money(amount, currency="EUR"):
    return amount.quantize(Decimal(1).scaleb(-decimals(currency)), rounding=ROUND_HALF_UP)


def fmt_money(amount, currency="EUR"):
    return f"{round_money(amount, currency):.{decimals(currency)}f}"
