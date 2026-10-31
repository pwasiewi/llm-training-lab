"""Hidden integration datasets for ledger (never copied into the workdir)."""

DISCOUNTS = {"WELCOME5": {"type": "fixed", "amount": "5.00"}, "TENPCT": {"type": "percent", "value": "10"},
             "BULK20": {"type": "fixed", "amount": "20.00"}, "HALF": {"type": "percent", "value": "50"},
             "BIG500": {"type": "fixed", "amount": "500.00"}}

RATES = [  # deliberately unsorted; later rows must not be used for earlier orders
    ("2026-01-09", "USD", "0.9311"), ("2026-01-07", "USD", "0.9288"), ("2026-01-02", "GBP", "1.1712"),
    ("2026-01-05", "JPY", "0.006022"), ("2025-12-29", "JPY", "0.006050"), ("2026-01-03", "KWD", "2.9817"),
    ("2026-01-06", "CHF", "1.0634"), ("2025-12-30", "USD", "0.9402"), ("2026-01-12", "GBP", "1.1650"),
]


def L(sku, qty, price):
    return {"sku": sku, "qty": qty, "unit_price": price}


MAIN = [
    {"id": "E1", "date": "2025-12-31", "customer": "Acme GmbH", "lines": [L("X1", 3, "19.99")], "discount": "WELCOME5", "vat_rate": "0.19"},
    {"id": "E2", "date": "2026-01-02", "customer": "Globex SA", "lines": [L("K9", 1, "249.00"), L("C2", 4, "2.49")], "discount": "TENPCT", "vat_rate": "0.20"},
    {"id": "E3", "date": "2026-01-08", "customer": "Tiny Shop", "currency": "EUR", "lines": [L("C2", 1, "3.10")], "discount": "BULK20", "vat_rate": "0.07"},
    {"id": "U1", "date": "2026-01-08", "customer": "Initech LLC", "currency": "USD", "lines": [L("K9", 1, "279.00"), L("C2", 3, "2.75")], "vat_rate": "0"},
    {"id": "U2", "date": "2026-01-01", "customer": "Hooli Inc", "currency": "USD", "lines": [L("X1", 5, "21.50")], "discount": "WELCOME5", "vat_rate": "0.0825"},
    {"id": "G1", "date": "2026-01-11", "customer": "Vandelay Ltd", "currency": "GBP", "lines": [L("K9", 2, "215.00")], "discount": "BULK20", "vat_rate": "0.20"},
    {"id": "J1", "date": "2026-01-05", "customer": "Kobayashi KK", "currency": "JPY", "lines": [L("X1", 3, "3333")], "discount": "HALF", "vat_rate": "0.10"},
    {"id": "J2", "date": "2026-01-04", "customer": "Tanaka Shoji", "currency": "JPY", "lines": [L("C2", 2, "455")], "discount": "BIG500", "vat_rate": "0.10"},
    {"id": "K1", "date": "2026-01-10", "customer": "Al-Noor Trading", "currency": "KWD", "lines": [L("C2", 7, "0.875"), L("X1", 1, "6.4995")], "vat_rate": "0.05"},
    {"id": "C1", "date": "2026-01-06", "customer": "Helvetia AG", "currency": "CHF", "lines": [L("K9", 1, "260.00")], "discount": "WELCOME5", "vat_rate": "0.081"},
    {"id": "J3", "date": "2026-01-03", "customer": "Sato Denki", "currency": "JPY", "lines": [L("K9", 1, "41000")], "vat_rate": "0.10"},
]
GROUPS = {  # which MAIN orders an item judges (by order id)
    "eur_invoices": ["E1", "E2", "E3"],
    "minor_units": ["J1", "J3", "K1"],
    "rate_lookup": ["U1", "U2"],
    "discount_fx": ["G1", "J2", "C1"],
}
ERR_RATE = [MAIN[0], {"id": "U9", "date": "2025-12-20", "customer": "Early Bird", "currency": "USD",
                      "lines": [L("X1", 1, "10.00")], "vat_rate": "0"}]
ERR_CCY = [MAIN[1], {"id": "S1", "date": "2026-01-05", "customer": "Nordic AB", "currency": "SEK",
                     "lines": [L("X1", 1, "100.00")], "vat_rate": "0.25"}]
COMPAT = [MAIN[0], MAIN[1], MAIN[2]]   # EUR only, run WITHOUT --rates
