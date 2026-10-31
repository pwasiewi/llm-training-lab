#!/usr/bin/env python3
"""rules verifier: 17 mechanical rubric items; PASS = all 17."""
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "lib"))
sys.path.insert(0, str(HERE))
import b12  # noqa: E402
import gen  # noqa: E402

ENTRY_RE = re.compile(r"^- ([\w.-]+): (.+) \(([0-9a-f]{7})\)$")
ALLOWED_NEW = {"out/RELEASE_NOTES.md", "out/REPORT.md"}


def body(work: Path, pristine: Path):
    r = b12.Rubric()
    commits = [json.loads(ln) for ln in (pristine / "commits.jsonl").read_text().splitlines() if ln.strip()]
    if commits != gen.make_commits():
        raise b12.BenchError("pristine commits.jsonl does not match the generator (seed drift?)")
    exp = gen.expected(commits)
    by7 = {c["hash"][:7]: c for c in commits}
    notes_p = b12.inside(work, "out/RELEASE_NOTES.md")   # None when missing or a link out of WORKDIR
    text = notes_p.read_text(errors="replace") if notes_p is not None else ""
    lines = text.splitlines()
    r.add("deliverable", notes_p is not None and b12.inside(work, "out/REPORT.md") is not None,
          "out/RELEASE_NOTES.md + out/REPORT.md")
    r.add("title", lines[:1] == [f"# Release {gen.VERSION}"], f"line1={lines[0][:40] if lines else '-'}")
    r.add("codename", lines[1:3] == [gen.CODENAME, ""], f"line2={lines[1][:40] if len(lines) > 1 else '-'}")

    # parse sections
    secs: dict[str, list[tuple[int, str]]] = {}
    heads = []
    cur = None
    for i, ln in enumerate(lines):
        if ln.startswith("## "):
            cur = ln[3:].strip()
            heads.append(cur)
            secs.setdefault(cur, [])
        elif cur is not None:
            secs[cur].append((i, ln))
    r.add("sections", heads == gen.SECTIONS, f"headings={heads}")

    entries = {s: [] for s in secs}     # section -> [(hash7, scope, text, line_idx)]
    bad_fmt = []
    for s, sl in secs.items():
        for i, ln in sl:
            if ln.startswith("- "):
                m = ENTRY_RE.match(ln)
                if m:
                    entries[s].append((m.group(3), m.group(1), m.group(2), i))
                else:
                    bad_fmt.append(i + 1)
            elif ln.startswith("  Migration: ") or ln == "" or i == len(lines) - 1:  # last line = R15
                continue
            else:
                bad_fmt.append(i + 1)
    # R3: no blank line between heading and first entry; exactly one blank after last entry
    for s, sl in secs.items():
        if sl and sl[0][1] == "":
            bad_fmt.append(sl[0][0] + 1)
    r.add("format", bool(lines) and not bad_fmt, f"bad lines {bad_fmt[:6]}" if bad_fmt else "ok")

    where = {h: s for s, es in entries.items() for h, *_ in es}
    exp_sec = {line.split("(")[-1][:7]: s for s, lst in exp.items() for line, _ in lst}
    wrong = [h for h, s in exp_sec.items() if where.get(h) != s]
    r.add("classify", not wrong and bool(exp_sec), f"{len(wrong)} expected entries missing/misplaced {wrong[:4]}")

    bad_order = []
    for s, lst in exp.items():
        want = [line.split("(")[-1][:7] for line, _ in lst]
        got = [h for h, *_ in entries.get(s, []) if h in set(want)]
        if got != [h for h in want if h in set(got)]:
            bad_order.append(s)
    r.add("order", not bad_order and bool(entries), f"out of order: {bad_order}")

    def breaking(c):
        p = gen.parse_subject(c["subject"])
        return bool(p and p[2]) or "BREAKING CHANGE:" in c.get("body", "")

    bots_listed = [h for h in where if by7.get(h, {}).get("author", "").endswith("@bots.example")
                   and not breaking(by7[h])]
    have = bool(where)  # content items never pass vacuously on an empty deliverable
    r.add("bots", have and not bots_listed, f"non-breaking bot commits listed: {bots_listed[:4]}")
    bot_brk = [c["hash"][:7] for c in commits if c["author"].endswith("@bots.example") and breaking(c)]
    r.add("amend", bool(bot_brk) and all(where.get(h) == "Breaking" for h in bot_brk),
          f"breaking bot commits in Breaking: {sum(where.get(h) == 'Breaking' for h in bot_brk)}/{len(bot_brk)}")
    rev = {c["hash"][:7] for c in commits if c.get("reverts")} | {c["reverts"][:7] for c in commits if c.get("reverts")}
    r.add("reverts", have and not (rev & set(where)), f"revert-related listed: {sorted(rev & set(where))[:4]}")
    noscope = [c["hash"][:7] for c in commits if re.match(r"^\w+!?: ", c["subject"])]
    ok_scope = all(any(h == e[0] and e[1] == "core" for es in entries.values() for e in es)
                   for h in noscope if h in exp_sec)
    r.add("scope", bool(noscope) and ok_scope, f"{len(noscope)} scopeless commits")
    # duplicates: later cherry-picks must be absent, earliest present
    seen = {}
    dup_bad = []
    for c in sorted(commits, key=gen.instant):
        if c.get("reverts"):
            continue
        k = c["subject"]
        if k in seen:
            if c["hash"][:7] in where:
                dup_bad.append(c["hash"][:7])
        else:
            seen[k] = c
    r.add("dedupe", have and not dup_bad, f"later duplicates listed: {dup_bad[:4]}")

    mig_bad = []
    for h, scope, txt, i in entries.get("Breaking", []):
        c = by7.get(h)
        want = (f"  Migration: {c['migration']}" if c and c.get("migration") else gen.MIGRATION_NONE)
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        if nxt != want:
            mig_bad.append(h)
    r.add("migration", bool(entries.get("Breaking")) and not mig_bad, f"bad migration lines: {mig_bad[:4]}")
    r.add("endline", len(lines) >= 2 and lines[-1] == gen.END_LINE and lines[-2] == "",
          f"last={lines[-1][:40] if lines else '-'}")

    rtext = b12.read_inside(work, "out/REPORT.md")
    tok = "LINT-OK " + hashlib.sha256(gen.LINT_SALT.encode() + notes_p.read_bytes()).hexdigest()[:20] \
        if notes_p is not None else "<none>"
    ev = rtext.split("## Evidence", 1)[1] if "## Evidence" in rtext else ""
    r.add("evidence", tok in ev, "token matches final file" if tok in ev else
          ("no ## Evidence section" if not ev else "token missing or stale"))

    ok, why = b12.check_protected(work, pristine)
    r.add("protected", ok, why)
    strays = []
    for f in sorted(work.rglob("*")):
        rel = f.relative_to(work).as_posix()
        if f.is_dir() or "__pycache__" in rel or rel == "agent.log" or rel.startswith(".qwen"):
            continue
        if not (pristine / rel).exists() and rel not in ALLOWED_NEW:
            strays.append(rel)
    r.add("no-strays", notes_p is not None and not strays, f"strays: {strays[:5]}" if strays else "clean")
    r.finish()


b12.main_wrapper(body)
