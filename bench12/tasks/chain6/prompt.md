This directory holds a dependent lookup chain. CHAIN.md lists the steps; OPS.md defines the range-table conventions and the operations. Work through the chain in order: each step's result is the input of the next step, so one wrong value spoils everything after it. Read each table's own header — header directives override the defaults.

Deliverables, in this directory:
- `chain.log`: one line per step, exactly `step NN: XXXXXX` (token after the step, 6-digit zero-padded).
- `ANSWER`: the final token alone, 6 digits.

Do not modify anything under tables/ or tools/, nor OPS.md or CHAIN.md. You may write helper scripts. Double-check every step before you finish.
