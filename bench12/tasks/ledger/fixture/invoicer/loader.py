"""Load orders and discount definitions from JSON."""
import json

from invoicer.models import Line, Order
from invoicer.money import to_decimal


def load_orders(path):
    with open(path) as f:
        raw = json.load(f)
    orders = []
    for o in raw:
        lines = [Line(l["sku"], int(l["qty"]), to_decimal(l["unit_price"])) for l in o["lines"]]
        orders.append(Order(o["id"], o["date"], o["customer"], lines, to_decimal(o.get("vat_rate", "0.19")),
                            o.get("discount", "")))
    return orders


def load_discounts(path):
    with open(path) as f:
        return json.load(f)
