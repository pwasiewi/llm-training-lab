# CR-118: multi-currency invoices

Until now every order was in EUR. Orders may now carry `"currency"` (ISO
code); a missing field means EUR. Prices, discounts, VAT and totals of an
order are in the order's currency.

1. **Currencies.** Supported codes and their minor units — exactly this table,
   any other code is an error (see 7):

   | code | decimals |
   |---|---|
   | EUR, USD, GBP, CHF | 2 |
   | JPY | 0 |
   | KWD | 3 |

   All rounding is half-up to the currency's minor unit. Every amount on an
   invoice and in the export is printed with exactly that many decimals
   (JPY: no decimal point). This includes unit prices and line amounts.

2. **Rates.** New optional CLI option `--rates FILE` (CSV with header
   `date,currency,eur_per_unit`, rows in any order). The rate for an order is
   the row of its currency with the latest date ON OR BEFORE the order date.
   EUR always has rate 1 and needs no rows.

3. **Discounts.** Fixed discounts in discounts.json are EUR amounts. For an
   order in another currency the amount is converted as amount / eur_per_unit,
   rounded to the order currency's minor unit, then capped at the subtotal as
   before. Percent discounts are rounded to the order currency's minor unit.

4. **Invoice text.** The total line becomes `Total: <amount> <CUR>`. Non-EUR
   invoices get one extra last line `Total (EUR): <total * eur_per_unit,
   rounded half-up to cents>`. EUR invoices stay byte-identical to today's.

5. **Export.** export.csv columns become
   `invoice_id,currency,order_id,date,customer,net,vat,total,total_eur`
   (net/vat/total in the order currency with its decimals; total_eur as in 4;
   for EUR orders total_eur equals total).

6. **Summary.** summary.txt becomes:

       SUMMARY
       invoices: <n>
       by currency:
         <CUR> <count> <sum of totals in CUR, with its decimals>
       total: <sum of the total_eur values> EUR

   one `by currency` line per currency present, sorted by code, indented by
   two spaces.

7. **Errors.** Exit status 3, one line on stderr, and NOTHING written under
   `--out` (not even the directory) when: an order's currency is not in the
   table (`invoicer: unknown currency XYZ`), or no rate exists on or before
   the order date (`invoicer: no rate for USD on or before 2026-01-05`), or a
   discount code is unknown (existing message).

8. Invoice numbering is unchanged: all currencies share the per-year sequence.
