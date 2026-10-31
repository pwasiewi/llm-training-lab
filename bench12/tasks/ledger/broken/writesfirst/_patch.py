# output directory created before validation -> error runs leave an empty out/
p = "invoicer/cli.py"; s = open(p).read()
s = s.replace("    rates = Rates.load(args.rates)\n", "    os.makedirs(os.path.join(args.out, \"invoices\"), exist_ok=True)\n    rates = Rates.load(args.rates)\n")
open(p, "w").write(s)
