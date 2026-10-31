# reconcile

Nightly job: match outgoing payments (`payments.csv`) to bank statement lines
(`bank.csv`). The matching rules are in the docstring of `reconcile.py`; the
current implementation is the reference for behaviour — its output is what the
accounting system ingests, byte for byte.

    python3 reconcile.py data/payments.csv data/bank.csv > matches.txt

Data contract:
* `id` columns are strings of digits; some exporters zero-pad them — echo ids
  exactly as they appear in the input.
* `ref` is either empty or `INV-` followed by exactly 6 digits.
* `memo` is free text and may contain commas (CSV-quoted).
* dates are ISO `YYYY-MM-DD`; amounts are integer cents.

`data/` holds a 20K-payment sample. Production volume is about five times
that and the job gets a few seconds.

    python -m pytest -q tests/
