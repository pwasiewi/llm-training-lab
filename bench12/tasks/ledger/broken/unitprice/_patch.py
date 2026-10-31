# unit prices still printed with 2 decimals (currency not threaded into the line format)
p = "invoicer/render.py"; s = open(p).read()
s = s.replace("x {fmt_money(l.unit_price, c)} = {fmt_money(l.unit_price * l.qty, c)}", "x {fmt_money(l.unit_price)} = {fmt_money(l.unit_price * l.qty, c)}")
open(p, "w").write(s)
