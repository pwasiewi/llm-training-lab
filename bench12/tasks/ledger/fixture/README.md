# invoicer

    python -m invoicer run --orders data/orders.json --discounts data/discounts.json \
        [--rates data/rates.csv] --out out/

Writes `out/invoices/<invoice>.txt`, `out/export.csv` and `out/summary.txt`.

Layout: `loader` (JSON -> models), `pricing` (subtotal, discount, VAT,
numbering), `discounts`, `money` (Decimal helpers), `render` (invoice text),
`export` (CSV), `report` (summary), `cli`.

Tests: `python -m pytest -q tests/`
