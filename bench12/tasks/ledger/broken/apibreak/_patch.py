# price() now requires rates -> the protected tests that call price(order, number, discounts) break
p = "invoicer/pricing.py"; s = open(p).read()
s = s.replace("def price(order, number, discounts, rates=None):", "def price(order, number, discounts, rates):")
s = s.replace("rate = (rates or Rates()).rate(cur, order.date)", "rate = rates.rate(cur, order.date)")
open(p, "w").write(s)
p = "invoicer/cli.py"; s = open(p).read()
open(p, "w").write(s)
