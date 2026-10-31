"""Independent oracle for CR-118 (written separately from reference/invoicer).

expected(orders, discounts, rates_rows) -> ("ok", {relpath: text}) or ("err", message)
"""
import csv
import io
from decimal import ROUND_HALF_UP, Decimal

DEC = {"EUR": 2, "USD": 2, "GBP": 2, "CHF": 2, "JPY": 0, "KWD": 3}


def q(x, cur):
    return x.quantize(Decimal(1).scaleb(-DEC[cur]), rounding=ROUND_HALF_UP)


def f(x, cur):
    return f"{q(x, cur):.{DEC[cur]}f}"


def expected(orders, discounts, rates_rows):
    for o in orders:
        if o.get("currency", "EUR") not in DEC:
            return "err", f"invoicer: unknown currency {o['currency']}"
    seq = {}
    numbered = []
    for o in sorted(orders, key=lambda o: (o["date"], o["id"])):
        y = o["date"][:4]
        seq[y] = seq.get(y, 0) + 1
        numbered.append((f"INV-{y}-{seq[y]:04d}", o))
    invs = []
    for num, o in numbered:
        cur = o.get("currency", "EUR")
        if cur == "EUR":
            rate = Decimal(1)
        else:
            cands = [(d, Decimal(r)) for d, c, r in rates_rows if c == cur and d <= o["date"]]
            if not cands:
                return "err", f"invoicer: no rate for {cur} on or before {o['date']}"
            rate = max(cands)[1]
        lines = [(l["sku"], int(l["qty"]), Decimal(str(l["unit_price"]))) for l in o["lines"]]
        sub = q(sum((p * n for _, n, p in lines), Decimal(0)), cur)
        code = o.get("discount", "")
        disc = q(Decimal(0), cur)
        if code:
            if code not in discounts:
                return "err", f"invoicer: unknown discount code {code}"
            d = discounts[code]
            if d["type"] == "fixed":
                disc = min(q(Decimal(d["amount"]) / rate, cur), sub)
            else:
                disc = q(sub * Decimal(d["value"]) / 100, cur)
        net = sub - disc
        vr = Decimal(str(o.get("vat_rate", "0.19")))
        vat = q(net * vr, cur)
        tot = net + vat
        teur = q(tot * rate, "EUR")
        invs.append(dict(num=num, o=o, cur=cur, lines=lines, sub=sub, disc=disc, net=net, vr=vr, vat=vat,
                         tot=tot, teur=teur))
    files = {}
    for v in invs:
        o, c = v["o"], v["cur"]
        t = [f"INVOICE {v['num']}", f"Order {o['id']}  Date {o['date']}", f"Customer: {o['customer']}"]
        for sku, n, p in v["lines"]:
            t.append(f"  {sku:<8}{n:>3} x {f(p, c)} = {f(p * n, c)}")
        t.append(f"Subtotal: {f(v['sub'], c)}")
        if v["disc"]:
            t.append(f"Discount: -{f(v['disc'], c)} ({o['discount']})")
        t.append(f"Net: {f(v['net'], c)}")
        t.append(f"VAT {(v['vr'] * 100).normalize():f}%: {f(v['vat'], c)}")
        t.append(f"Total: {f(v['tot'], c)} {c}")
        if c != "EUR":
            t.append(f"Total (EUR): {f(v['teur'], 'EUR')}")
        files[f"invoices/{v['num']}.txt"] = "\n".join(t) + "\n"
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["invoice_id", "currency", "order_id", "date", "customer", "net", "vat", "total", "total_eur"])
    for v in invs:
        c = v["cur"]
        w.writerow([v["num"], c, v["o"]["id"], v["o"]["date"], v["o"]["customer"], f(v["net"], c), f(v["vat"], c),
                    f(v["tot"], c), f(v["teur"], "EUR")])
    files["export.csv"] = buf.getvalue()
    s = ["SUMMARY", f"invoices: {len(invs)}", "by currency:"]
    for c in sorted({v["cur"] for v in invs}):
        vs = [v for v in invs if v["cur"] == c]
        s.append(f"  {c} {len(vs)} {f(sum((v['tot'] for v in vs), Decimal(0)), c)}")
    s.append(f"total: {f(sum((v['teur'] for v in invs), Decimal(0)), 'EUR')} EUR")
    files["summary.txt"] = "\n".join(s) + "\n"
    return "ok", files
