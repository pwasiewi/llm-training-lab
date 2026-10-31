"""EUR conversion rates: CSV date,currency,eur_per_unit; latest row on or before a date."""
import csv
from decimal import Decimal


class MissingRate(Exception):
    pass


class Rates:
    def __init__(self, rows=()):
        self.rows = {}
        for date, cur, rate in rows:
            self.rows.setdefault(cur, []).append((date, Decimal(rate)))

    @classmethod
    def load(cls, path):
        if not path:
            return cls()
        with open(path, newline="") as f:
            return cls((r["date"], r["currency"], r["eur_per_unit"]) for r in csv.DictReader(f))

    def rate(self, currency, date):
        if currency == "EUR":
            return Decimal(1)
        best = None
        for d, r in self.rows.get(currency, ()):
            if d <= date and (best is None or d > best[0]):
                best = (d, r)
        if best is None:
            raise MissingRate(f"{currency} on or before {date}")
        return best[1]
