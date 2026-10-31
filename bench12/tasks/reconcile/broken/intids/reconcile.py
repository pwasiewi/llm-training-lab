#!/usr/bin/env python3
"""reconcile.py PAYMENTS.csv BANK.csv — match outgoing payments to bank statement lines.

Pass 1 (amount match): payments are processed in (date, id) order. A payment
matches an unused bank line with the SAME amount whose date is within 3 days.
Best candidate: smallest absolute date difference, then earlier bank date,
then smaller bank id.

Pass 2 (reference match): payments still unmatched, in the same order, match
an unused bank line whose memo contains the payment's ref (case-insensitive
substring), dated within 7 days, any amount. Best candidate: smallest bank id.

Output (stdout): one line per payment in processing order,
  payment_id,bank_id,pass      (pass = 1 or 2)
  payment_id,,-                (unmatched)
then a footer line:
  # matched P1 pass1, P2 pass2, unmatched U, unused bank B

Indexed implementation: pass 1 probes (amount, day) buckets in candidate
order; pass 2 uses an index of every `inv-NNNNNN` window in the memos
(refs are always INV- plus 6 digits, see README).
"""
import csv
import sys
from collections import defaultdict, deque
from datetime import date

PROBE = (0, -1, 1, -2, 2, -3, 3)


def main(argv):
    if len(argv) != 3:
        print(__doc__.strip().splitlines()[0], file=sys.stderr)
        return 2
    with open(argv[1], newline="") as f:
        pays = [(date.fromisoformat(r["date"]).toordinal(), int(r["id"]), r["id"], int(r["amount_cents"]),
                 r["ref"].lower()) for r in csv.DictReader(f)]
    with open(argv[2], newline="") as f:
        bank = [(int(r["id"]), r["id"], date.fromisoformat(r["date"]).toordinal(), int(r["amount_cents"]),
                 r["memo"].lower()) for r in csv.DictReader(f)]
    pays.sort(key=lambda p: (p[0], p[1]))

    buckets = defaultdict(list)
    for nid, sid, d, amt, _ in bank:
        buckets[(amt, d)].append((nid, sid))
    for k in buckets:
        buckets[k].sort()
    buckets = {k: deque(v) for k, v in buckets.items()}

    used = set()
    result = {}
    for d, _, pid, amt, _ in pays:
        for off in PROBE:
            q = buckets.get((amt, d + off))
            if q:
                nid, sid = q.popleft()
                used.add(nid)
                result[pid] = (str(nid), "1")
                break

    index = defaultdict(list)
    for nid, sid, d, _, memo in bank:
        if nid in used:
            continue
        i = memo.find("inv-")
        while i != -1:
            key = memo[i:i + 10]
            if len(key) == 10 and key[4:].isdigit():
                index[key].append((nid, sid, d))
            i = memo.find("inv-", i + 1)
    for d, _, pid, _, ref in pays:
        if pid in result or not ref:
            continue
        best = None
        for nid, sid, bd in index.get(ref, ()):
            if nid not in used and abs(bd - d) <= 7 and (best is None or nid < best[0]):
                best = (nid, sid)
        if best:
            used.add(best[0])
            result[pid] = (str(best[0]), "2")

    out = []
    n1 = n2 = 0
    for _, _, pid, _, _ in pays:
        m = result.get(pid)
        if m:
            out.append(f"{pid},{m[0]},{m[1]}")
            if m[1] == "1":
                n1 += 1
            else:
                n2 += 1
        else:
            out.append(f"{pid},,-")
    out.append(f"# matched {n1} pass1, {n2} pass2, unmatched {len(pays) - n1 - n2}, unused bank {len(bank) - len(used)}")
    sys.stdout.write("\n".join(out) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
