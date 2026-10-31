"""Deterministic data for the reconcile task (sample in the workdir, hidden sets in verify)."""
import csv
import io
import random
from datetime import date

COMMON = [999, 1999, 4999, 12000, 2500, 500]
D0 = date(2026, 1, 1).toordinal()


def make(n_pay: int, seed: int, skew: float = 0.13, nbank_extra: float = 0.15, days: int = 365):
    rng = random.Random(seed)
    pays, bank = [], []
    used_bank_ids = set()

    def bid():
        while True:
            x = rng.randrange(1, 50 * (n_pay + 10))
            if x not in used_bank_ids:
                used_bank_ids.add(x)
                return f"{x:09d}" if rng.random() < 0.03 else str(x)

    def memo(ref, style):
        if style == 0:
            return f"SEPA {ref} {rng.choice(['ACME', 'Globex', 'Initech'])}"
        if style == 1:
            return f"transfer {ref.lower()} thanks"
        if style == 2:
            return f"{ref}{rng.randrange(10)} batch"           # ref is a prefix of a longer token
        if style == 3:
            return f"x{ref}/{rng.randrange(100)}"             # glued to other text
        return f"invoice, Ref {ref}, {rng.choice(['Q1', 'Q2'])}"  # commas -> csv quoting

    for i in range(1, n_pay + 1):
        amt = rng.choice(COMMON) if rng.random() < skew else rng.randrange(100, 300000)
        d = D0 + rng.randrange(days)
        ref = f"INV-{rng.randrange(10**6):06d}" if rng.random() < 0.9 else ""
        pays.append(dict(id=str(i), date=date.fromordinal(d).isoformat(), amount_cents=amt, ref=ref))
        r = rng.random()
        if r < 0.70:
            off = rng.choice([0, 0, 1, -1, 2, 3, -3, 4, 5])
            bank.append(dict(id=bid(), date=date.fromordinal(d + off).isoformat(), amount_cents=amt,
                             memo=memo(ref, rng.randrange(5)) if ref and rng.random() < 0.3 else "card payment"))
            if rng.random() < 0.1:  # a same-amount competitor: tie-breaks matter
                bank.append(dict(id=bid(), date=date.fromordinal(d - off).isoformat(), amount_cents=amt,
                                 memo="card payment"))
        elif r < 0.85 and ref:
            off = rng.randrange(-8, 9)
            bank.append(dict(id=bid(), date=date.fromordinal(d + off).isoformat(), amount_cents=amt - rng.randrange(1, 300),
                             memo=memo(ref, rng.randrange(5))))
    for _ in range(int(n_pay * nbank_extra)):
        amt = rng.choice(COMMON) if rng.random() < skew else rng.randrange(100, 300000)
        bank.append(dict(id=bid(), date=date.fromordinal(D0 + rng.randrange(days)).isoformat(), amount_cents=amt,
                         memo=rng.choice(["fee", "interest", "card payment", "Ref INV-000000, misc"])))
    rng.shuffle(pays)
    rng.shuffle(bank)
    return pays, bank


def to_csv(rows, cols) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols, lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow(r)
    return buf.getvalue()


def write(dirpath, n_pay, seed, **kw):
    from pathlib import Path
    p, b = make(n_pay, seed, **kw)
    d = Path(dirpath)
    d.mkdir(parents=True, exist_ok=True)
    (d / "payments.csv").write_text(to_csv(p, ["id", "date", "amount_cents", "ref"]))
    (d / "bank.csv").write_text(to_csv(b, ["id", "date", "amount_cents", "memo"]))
