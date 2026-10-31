#!/usr/bin/env python3
"""rules fixture: a ~450-line RULES.md (16 rules), commits.jsonl, a lint tool.

Arbitrary literals sit at three depths: R6 codename (~25 %), R10 migration
placeholder (~60 %), R15 terminator line (~95 %). None is inferable from the
environment; the pad text even names a stale terminator (`-- end --`) as a
decoy. R14 (~75 %) amends R7. The lint tool is syntax-only so running it does
not leak the tail rules.
"""
from __future__ import annotations

import json
import os
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

SEED = int(os.environ.get("B12_SEED", "12"))
VERSION = "3.2.0"
CODENAME = "Codename: amber-heron-7"
MIGRATION_NONE = "  Migration: none documented"
END_LINE = "-- end of notes (ref 7Q4-ZEPHYR) --"
LINT_SALT = "b12-rules-salt-91f3"
SECTIONS = ["Breaking", "Features", "Fixes", "Internal"]
INTERNAL_TYPES = ("chore", "refactor", "ci", "test", "build")

WORDS = ("cache index parser scheduler token buffer manifest mirror shard queue worker socket "
         "report config overlay profile binary archive digest ledger window session").split()
VERBS = ("add handle fix avoid support drop rework guard tune clarify validate expose "
         "rename split cache retry").split()
SCOPES = ["api", "cli", "db", "net", "ui", "docs", "build", "auth"]
HUMANS = ["ana@corp.example", "bo@corp.example", "chen@corp.example", "dee@corp.example"]
BOTS = ["deps-bot@bots.example", "format-bot@bots.example"]


def make_commits(seed: int = SEED) -> list[dict]:
    rng = random.Random(seed * 7919 + 3)
    base = datetime(2026, 8, 1, 9, 0, tzinfo=timezone.utc)
    commits = []
    used = set()

    def h():
        while True:
            x = "".join(rng.choice("0123456789abcdef") for _ in range(40))
            if x[:7] not in used:
                used.add(x[:7])
                return x

    def ts(i):
        t = base + timedelta(hours=7 * i + rng.randrange(0, 5), minutes=rng.randrange(60))
        off = rng.choice([0, 2, -5, 9, -8, 5])
        return t.astimezone(timezone(timedelta(hours=off))).isoformat()

    def subj():
        return f"{rng.choice(VERBS)} {rng.choice(WORDS)} {rng.choice(WORDS)}"

    i = 0
    kinds = (["feat"] * 14 + ["fix"] * 14 + list(INTERNAL_TYPES) * 3 + ["feat!"] * 3 + ["fixbc"] * 2
             + ["botchore"] * 5 + ["botbreaking"] * 2 + ["revertpair"] * 3 + ["cherry"] * 3 + ["noscope"] * 4)
    rng.shuffle(kinds)
    for kind in kinds:
        i += 1
        c = dict(hash=h(), author=rng.choice(HUMANS), date=ts(i), body="")
        scope = rng.choice(SCOPES)
        if kind == "feat":
            c["subject"] = f"feat({scope}): {subj()}"
        elif kind == "fix":
            c["subject"] = f"fix({scope}): {subj()}"
        elif kind in INTERNAL_TYPES:
            c["subject"] = f"{kind}({scope}): {subj()}"
        elif kind == "feat!":
            c["subject"] = f"feat({scope})!: {subj()}"
            if rng.random() < 0.67:
                c["migration"] = f"run `tool migrate --{rng.choice(WORDS)}` once"
        elif kind == "fixbc":
            c["subject"] = f"fix({scope}): {subj()}"
            c["body"] = f"Details follow.\n\nBREAKING CHANGE: {rng.choice(WORDS)} format changed"
        elif kind == "botchore":
            c["author"] = rng.choice(BOTS)
            c["subject"] = f"chore(deps): bump {rng.choice(WORDS)} to 1.{rng.randrange(30)}.{rng.randrange(9)}"
        elif kind == "botbreaking":
            c["author"] = rng.choice(BOTS)
            c["subject"] = f"build(deps)!: bump {rng.choice(WORDS)} to {rng.randrange(2, 9)}.0.0"
            c["migration"] = "rebuild all plugins against the new ABI"
        elif kind == "noscope":
            c["subject"] = f"{rng.choice(['feat', 'fix'])}: {subj()}"
        elif kind == "revertpair":
            c["subject"] = f"feat({scope}): {subj()}"
            commits.append(c)
            i += 1
            r = dict(hash=h(), author=rng.choice(HUMANS), date=ts(i), body="",
                     subject=f"Revert \"{c['subject']}\"", reverts=c["hash"])
            commits.append(r)
            continue
        elif kind == "cherry":
            c["subject"] = f"fix({scope}): {subj()}"
            commits.append(c)
            i += 3
            d = dict(hash=h(), author=rng.choice(HUMANS), date=ts(i), body="(cherry picked)",
                     subject=c["subject"])
            commits.append(d)
            continue
        commits.append(c)
    rng.shuffle(commits)  # file order is NOT time order
    return commits


# --------------------------------------------------------------- truth -----

def parse_subject(s: str):
    import re
    m = re.match(r"^(\w+)(?:\(([^)]*)\))?(!)?: (.*)$", s)
    if not m:
        return None
    return m.group(1), (m.group(2) or "core"), bool(m.group(3)), m.group(4)


def instant(c):
    return datetime.fromisoformat(c["date"]).astimezone(timezone.utc)


def expected(commits: list[dict]) -> dict:
    """Section -> ordered list of (entry line, commit) per the rules."""
    reverted = {c["reverts"] for c in commits if c.get("reverts")}
    out = {s: [] for s in SECTIONS}
    seen = {s: set() for s in SECTIONS}
    for c in sorted(commits, key=instant):
        if c.get("reverts") or c["hash"] in reverted:
            continue
        p = parse_subject(c["subject"])
        if p is None:
            continue
        typ, scope, bang, text = p
        breaking = bang or "BREAKING CHANGE:" in c.get("body", "")
        if c["author"].endswith("@bots.example") and not breaking:
            continue
        if breaking:
            sec = "Breaking"
        elif typ == "feat":
            sec = "Features"
        elif typ == "fix":
            sec = "Fixes"
        elif typ in INTERNAL_TYPES:
            sec = "Internal"
        else:
            continue
        key = (scope, text)
        if key in seen[sec]:
            continue
        seen[sec].add(key)
        out[sec].append((f"- {scope}: {text} ({c['hash'][:7]})", c))
    return out


def expected_text(commits) -> str:
    exp = expected(commits)
    lines = [f"# Release {VERSION}", CODENAME, ""]
    for sec in SECTIONS:
        lines.append(f"## {sec}")
        for line, c in exp[sec]:
            lines.append(line)
            if sec == "Breaking":
                mig = c.get("migration")
                lines.append(f"  Migration: {mig}" if mig else MIGRATION_NONE)
        lines.append("")
    lines.append(END_LINE)
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------- texts -----

def pad(rng: random.Random, n: int, topic: str) -> list[str]:
    sents = [
        f"The {topic} guidance below is background; it explains why the release process looks the way it does.",
        "Earlier releases were assembled by hand, which led to inconsistent headings and missing entries.",
        "Contributors are encouraged to keep commit subjects short and to use conventional prefixes.",
        "The release manager rotates every quarter and keeps a private checklist in the team wiki.",
        "Older release notes used `-- end --` as the closing line; that convention is retired.",
        "Nothing in this paragraph is a rule; numbered rules are the only binding text in this file.",
        "Some teams prefer changelogs grouped by component, but this project groups by change type.",
        "Historical tooling stored notes as reStructuredText; the current format is Markdown.",
        "When in doubt, consult the numbered rules rather than examples from previous releases.",
        "Reviewers check the notes against the commit list before the tag is pushed.",
    ]
    out = []
    for _ in range(n):
        para = " ".join(rng.choice(sents) for _ in range(rng.randrange(3, 6)))
        # wrap at ~90 columns
        line = ""
        for w in para.split():
            if len(line) + len(w) > 88:
                out.append(line)
                line = w
            else:
                line = f"{line} {w}".strip()
        out.append(line)
        out.append("")
    return out


RULES = {
    1: ["## R1 — Deliverables and scratch space",
        "Write the release notes to `out/RELEASE_NOTES.md` and a short report to `out/REPORT.md`.",
        "Helper scripts may live only under `scratch/`; delete `scratch/` before you finish.",
        "Create no other files anywhere."],
    2: ["## R2 — Title",
        "Line 1 of RELEASE_NOTES.md is `# Release <version>`, where <version> is the content of the",
        "`VERSION` file. Badges in README.md are not authoritative."],
    3: ["## R3 — Sections",
        "After the header block, the notes have exactly these sections, in this order:",
        "`## Breaking`, `## Features`, `## Fixes`, `## Internal`. Each section heading is followed",
        "directly by its entries (no blank line between heading and first entry); one blank line",
        "follows the last entry of every section."],
    4: ["## R4 — Classification",
        "Commit subjects follow `type(scope)!: text` (scope and `!` optional). A commit is BREAKING",
        "when its subject has `!` before the colon or its body contains a line starting with",
        "`BREAKING CHANGE:`. Breaking commits go to Breaking and nowhere else. Otherwise `feat` ->",
        "Features, `fix` -> Fixes, `chore`, `refactor`, `ci`, `test`, `build` -> Internal."],
    5: ["## R5 — Entry format",
        "Each entry is one line: `- <scope>: <text> (<first 7 characters of the hash>)`, with",
        "<text> copied verbatim from the subject after `: `."],
    6: ["## R6 — Codename",
        "Line 2 of RELEASE_NOTES.md is exactly `" + CODENAME + "`, and line 3 is blank."],
    7: ["## R7 — Bot commits",
        "Commits whose author address ends in `@bots.example` are omitted from the notes."],
    8: ["## R8 — Order",
        "Within a section, entries are ordered by commit time, oldest first. Commit times carry",
        "different UTC offsets; order by the actual instant, not by the text of the timestamp."],
    9: ["## R9 — Reverts",
        "A commit with a `reverts` field and the commit it reverts are both omitted."],
    10: ["## R10 — Migration lines",
          "Every entry in `## Breaking` is followed by one line, indented by two spaces:",
          "`  Migration: <migration field of the commit>`; when the commit has no `migration`",
          "field the line is exactly `" + MIGRATION_NONE + "`."],
    11: ["## R11 — Missing scope",
         "A subject without `(scope)` is listed with the scope `core`."],
    12: ["## R12 — Duplicates",
          "Cherry-picks repeat a subject. When the same scope and text would appear twice in one",
          "section, keep only the oldest commit."],
    13: ["## R13 — Inputs are read-only",
         "Do not modify, rename or delete `commits.jsonl`, `VERSION`, `README.md`, `RULES.md` or",
         "anything under `tools/`."],
    14: ["## R14 — Amendment to R7",
         "R7 does not apply to breaking commits: a breaking commit authored by a bot IS listed",
         "(in `## Breaking`, with its migration line)."],
    15: ["## R15 — Terminator",
         "The last line of RELEASE_NOTES.md is exactly `" + END_LINE + "`,",
         "preceded by one blank line."],
    16: ["## R16 — Evidence",
         "When the notes are final, run `python3 tools/lint_notes.py out/RELEASE_NOTES.md`.",
         "`out/REPORT.md` must contain a section `## Evidence` that quotes the `LINT-OK ...` line",
         "printed by that run verbatim. Re-run it after every edit; a stale token fails."],
}


def rules_md(seed: int = SEED) -> str:
    rng = random.Random(seed * 31 + 5)
    lines = ["# RULES.md — release-notes job", "",
             "Sixteen numbered rules (R1-R16) follow, spread through this file. Every numbered rule",
             "is binding; the prose between them is background only.", ""]
    # pad budget per gap so that R6 lands ~25 %, R10 ~60 %, R15 ~95 % of the file
    gaps = {1: 2, 2: 3, 3: 4, 4: 3, 5: 6, 6: 4, 7: 5, 8: 4, 9: 6, 10: 4, 11: 5, 12: 6, 13: 7, 14: 9, 15: 1, 16: 0}
    topics = ["history", "style", "tooling", "review", "branching", "tagging", "mirrors", "QA"]
    for r in range(1, 17):
        lines += RULES[r] + [""]
        if gaps[r]:
            lines += [f"### Background: {rng.choice(topics)}", ""] + pad(rng, gaps[r], rng.choice(topics))
    return "\n".join(lines) + "\n"


LINT = '''#!/usr/bin/env python3
"""lint_notes.py NOTES — syntax lint for release notes (headings + entry lines only)."""
import hashlib
import re
import sys

SALT = "%s"


def main():
    if len(sys.argv) != 2:
        print("usage: lint_notes.py NOTES")
        return 2
    data = open(sys.argv[1], "rb").read()
    text = data.decode("utf-8", "replace")
    errs = []
    lines = text.splitlines()
    if not lines or not lines[0].startswith("# Release "):
        errs.append("line 1: expected '# Release <version>'")
    for i, ln in enumerate(lines, 1):
        if ln.startswith("- ") and not re.match(r"^- [\\w.-]+: .+ \\([0-9a-f]{7}\\)$", ln):
            errs.append(f"line {i}: malformed entry")
        if ln.startswith("#") and i > 1 and not re.match(r"^## \\w+$", ln):
            errs.append(f"line {i}: malformed heading")
    if errs:
        print("\\n".join(errs))
        print("LINT-FAIL")
        return 1
    tok = hashlib.sha256(SALT.encode() + data).hexdigest()[:20]
    print(f"LINT-OK {tok}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
''' % LINT_SALT


README = """# relnotes-demo

[build: passing] [version: v3.3.0-rc1] [docs: latest]

Demo service used to rehearse the release-notes job. See RULES.md.
"""


def write(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "tools").mkdir(exist_ok=True)
    commits = make_commits()
    with open(out / "commits.jsonl", "w") as f:
        for c in commits:
            f.write(json.dumps(c, sort_keys=True) + "\n")
    (out / "VERSION").write_text(VERSION + "\n")
    (out / "README.md").write_text(README)
    (out / "RULES.md").write_text(rules_md())
    (out / "tools" / "lint_notes.py").write_text(LINT)


if __name__ == "__main__":
    write(Path(sys.argv[1]))
