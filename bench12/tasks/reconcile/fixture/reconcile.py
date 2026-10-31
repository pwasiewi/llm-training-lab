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
"""
import csv
import sys
from datetime import date


def load(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def day(s):
    return date.fromisoformat(s).toordinal()


def main(argv):
    if len(argv) != 3:
        print(__doc__.strip().splitlines()[0], file=sys.stderr)
        return 2
    payments = load(argv[1])
    bank = load(argv[2])
    for p in payments:
        p["day"] = day(p["date"])
        p["amount"] = int(p["amount_cents"])
        p["nid"] = int(p["id"])
    for b in bank:
        b["day"] = day(b["date"])
        b["amount"] = int(b["amount_cents"])
        b["nid"] = int(b["id"])
    payments.sort(key=lambda p: (p["day"], p["nid"]))

    used = set()
    result = {}
    by_amount = {}
    for b in bank:
        by_amount.setdefault(b["amount"], []).append(b)

    # pass 1
    for p in payments:
        best = None
        for b in by_amount.get(p["amount"], []):
            if b["id"] in used:
                continue
            diff = abs(b["day"] - p["day"])
            if diff > 3:
                continue
            key = (diff, b["day"], b["nid"])
            if best is None or key < best[0]:
                best = (key, b)
        if best is not None:
            used.add(best[1]["id"])
            result[p["id"]] = (best[1]["id"], "1")

    # pass 2
    for p in payments:
        if p["id"] in result:
            continue
        ref = p["ref"].lower()
        best = None
        for b in bank:
            if b["id"] in used:
                continue
            if abs(b["day"] - p["day"]) > 7:
                continue
            if ref and ref in b["memo"].lower():
                if best is None or b["nid"] < best["nid"]:
                    best = b
        if best is not None:
            used.add(best["id"])
            result[p["id"]] = (best["id"], "2")

    out = []
    n1 = n2 = 0
    for p in payments:
        if p["id"] in result:
            bid, ps = result[p["id"]]
            out.append(f"{p['id']},{bid},{ps}")
            n1 += ps == "1"
            n2 += ps == "2"
        else:
            out.append(f"{p['id']},,-")
    unmatched = len(payments) - n1 - n2
    out.append(f"# matched {n1} pass1, {n2} pass2, unmatched {unmatched}, unused bank {len(bank) - len(used)}")
    sys.stdout.write("\n".join(out) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
