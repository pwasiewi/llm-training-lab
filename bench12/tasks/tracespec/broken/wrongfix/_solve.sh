#!/bin/bash
# plausible wrong fixes: ceil on whole minutes, TOTAL rounded from unrounded sum
set -e
ref="$(dirname "$0")/../../reference"
command cp -a "$ref/meter/." meter/; command cp -a "$ref/tests/." tests/
python3 - <<'PY'
p = "meter/bill.py"; s = open(p).read()
s = s.replace("return max(1, math.ceil(seconds / BLOCK_SECONDS))", "return max(1, math.ceil(int(seconds // 60) / 15))")
open(p, "w").write(s)
p = "meter/fmt.py"; s = open(p).read()
s = s.replace('lines.append(f"TOTAL {sum(amounts, Decimal(0))}")', 'lines.append(f"TOTAL {money(sum((c for *_, c in rows), Decimal(0)))}")')
open(p, "w").write(s)
PY
