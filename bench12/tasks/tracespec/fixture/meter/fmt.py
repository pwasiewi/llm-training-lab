"""Report rendering."""


def money(x):
    return f"{x:.2f}"


def render(rows):
    lines = [f"{acct} {blocks} {billable} {money(cost)}" for acct, blocks, billable, cost in rows]
    total = sum(cost for *_, cost in rows)
    lines.append(f"TOTAL {money(total)}")
    return "\n".join(lines) + "\n"
