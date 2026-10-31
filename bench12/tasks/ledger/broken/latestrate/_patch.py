# rate = newest row of the currency on or before the date ... compared as file order, not date
p = "invoicer/rates.py"; s = open(p).read()
s = s.replace("if d <= date and (best is None or d > best[0]):", "if d <= date:")
open(p, "w").write(s)
