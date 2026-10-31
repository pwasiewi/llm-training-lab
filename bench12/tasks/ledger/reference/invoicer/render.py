"""Plain-text invoice."""
from invoicer.money import fmt_money


def render_invoice(inv):
    o = inv.order
    out = [f"INVOICE {inv.number}", f"Order {o.id}  Date {o.date}", f"Customer: {o.customer}"]
    c = o.currency
    for l in o.lines:
        out.append(f"  {l.sku:<8}{l.qty:>3} x {fmt_money(l.unit_price, c)} = {fmt_money(l.unit_price * l.qty, c)}")
    out.append(f"Subtotal: {fmt_money(inv.subtotal, c)}")
    if inv.discount:
        out.append(f"Discount: -{fmt_money(inv.discount, c)} ({o.discount})")
    out.append(f"Net: {fmt_money(inv.net, c)}")
    out.append(f"VAT {(o.vat_rate * 100).normalize():f}%: {fmt_money(inv.vat, c)}")
    out.append(f"Total: {fmt_money(inv.total, c)} {c}")
    if c != "EUR":
        out.append(f"Total (EUR): {fmt_money(inv.total_eur)}")
    return "\n".join(out) + "\n"
