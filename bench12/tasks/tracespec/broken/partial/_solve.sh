#!/bin/bash
# follows README where traces were not read carefully: open sessions dropped,
# midnight-crossing sessions attributed to the start day
set -e
ref="$(dirname "$0")/../../reference"
command cp -a "$ref/meter/." meter/; command cp -a "$ref/tests/." tests/
python3 - <<'PY'
import re
p = "meter/sessions.py"; s = open(p).read()
s = s.replace("            out.append(Session(account, start, last))", "            pass")
open(p, "w").write(s)
p = "meter/bill.py"; s = open(p).read()
s = s.replace("            per_day[s.account][day] += blocks_for(secs)", "            per_day[s.account][s.start.date()] += blocks_for(secs)")
open(p, "w").write(s)
PY
