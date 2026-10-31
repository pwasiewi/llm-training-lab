from dataclasses import dataclass, field
from decimal import Decimal


@dataclass
class Line:
    sku: str
    qty: int
    unit_price: Decimal


@dataclass
class Order:
    id: str
    date: str
    customer: str
    lines: list
    vat_rate: Decimal
    discount: str = ""


@dataclass
class Invoice:
    number: str
    order: Order
    subtotal: Decimal
    discount: Decimal
    net: Decimal
    vat: Decimal
    total: Decimal
    notes: list = field(default_factory=list)
