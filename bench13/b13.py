#!/usr/bin/env python3
"""b13.py — small helpers for bench_13_public.sh (no network, no model calls).

Subcommands:
  swe-dataset PARQUET IDS_TSV OUTDIR
      Write OUTDIR/test.jsonl with the instances listed in IDS_TSV (first
      column, '#' lines ignored), in list order. The same file feeds
      mini-swe-agent (--subset OUTDIR --split test) and the SWE-rebench
      harness (--dataset_name OUTDIR/test.jsonl).
  ids TSV
      Print the selected ids of a selection TSV, one per line.
  swe-verdict LOGDIR INSTANCE PREDS_JSON
      Read the harness per-instance report (LOGDIR/<...>/INSTANCE/report.json)
      and print "VERDICT|SCORE|NOTE": PASS = resolved; FAIL = unresolved,
      empty patch, or a patch that edits pytest plugins (conftest.py,
      pytest.ini: the grader's own pytest would load them); ERROR = harness
      produced no report (a bench failure, never scored). SCORE = FAIL_TO_PASS
      tests passing "k/n (pct%)"; NOTE lists broken PASS_TO_PASS tests and
      other test-infra files the patch touches (for a human look).
  aider-verdict RESULTS_JSON
      Print "VERDICT|SCORE|NOTE" from aider's .aider.results.json:
      PASS (pass_1) / PASS (pass_2) / FAIL / ERROR (exception in the harness).
  aider-tamper TESTDIR ORIGDIR
      Print the non-solution files the model created or changed in an aider
      exercise dir vs the pristine polyglot exercise (empty = clean).
"""
import json
import re
import sys
from pathlib import Path

# files that change how the GRADER's pytest behaves; a fix never needs them
HARD_TAMPER = re.compile(r"(^|/)(conftest\.py|pytest\.ini)$")
# legit fixes sometimes touch these; flag for a human look
SOFT_TAMPER = re.compile(r"(^|/)(tox\.ini|setup\.cfg|pyproject\.toml|noxfile\.py|\.coveragerc)$|(^|/)tests?/")
# build/test artefacts aider's harness itself leaves in an exercise dir
AIDER_JUNK = re.compile(r"(^|/)(\.aider\.[^/]*|__pycache__|\.pytest_cache|\.gradle|target|build|node_modules"
                        r"|package-lock\.json|Cargo\.lock|CMakeFiles)(/|$)")


def read_ids(tsv: str) -> list[str]:
    ids = []
    for line in Path(tsv).read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        ids.append(line.split("\t")[0].strip())
    return ids


def _plain(v):
    """numpy/pyarrow values -> JSON-serialisable Python values."""
    if hasattr(v, "tolist"):
        return v.tolist()
    if isinstance(v, dict):
        return {k: _plain(x) for k, x in v.items()}
    if isinstance(v, float) and v != v:  # NaN
        return None
    return v


def swe_dataset(parquet: str, tsv: str, outdir: str) -> int:
    import pandas as pd

    ids = read_ids(tsv)
    df = pd.read_parquet(parquet)
    rows = {r["instance_id"]: r for r in df.to_dict("records")}
    missing = [i for i in ids if i not in rows]
    if missing:
        print(f"ERROR: not in {parquet}: {' '.join(missing)}", file=sys.stderr)
        return 1
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "test.jsonl", "w") as f:
        for i in ids:
            rec = {k: _plain(v) for k, v in rows[i].items()}
            # the fork's harness json.loads() these when they are strings; keep lists
            f.write(json.dumps(rec) + "\n")
    print(f"{len(ids)} instances -> {out / 'test.jsonl'}")
    return 0


def swe_verdict(logdir: str, inst: str, preds: str) -> int:
    patch = ""
    try:
        p = json.loads(Path(preds).read_text())
        patch = (p.get(inst) or {}).get("model_patch") or ""
    except (OSError, ValueError):
        pass
    reps = sorted(Path(logdir).glob(f"**/{inst}/report.json"))
    if not reps:
        if not patch.strip():
            print("FAIL|-|empty patch (agent submitted nothing)")
        else:
            print("ERROR|-|harness wrote no report.json")
        return 0
    rep = json.loads(reps[-1].read_text()).get(inst, {})
    ts = rep.get("tests_status") or {}
    f2p = ts.get("FAIL_TO_PASS", {})
    p2p = ts.get("PASS_TO_PASS", {})
    ok, bad = len(f2p.get("success", [])), len(f2p.get("failure", []))
    score = f"{ok}/{ok + bad} ({100 * ok // (ok + bad)}%)" if ok + bad else "-"
    note = []
    if p2p.get("failure"):
        note.append(f"{len(p2p['failure'])} PASS_TO_PASS broken")
    if not patch.strip() or rep.get("patch_is_None") or not rep.get("patch_exists", True):
        note.append("empty patch")
    if rep.get("patch_successfully_applied") is False:
        note.append("patch did not apply")
    touched = re.findall(r"^diff --git a/(\S+) b/", patch, re.M)
    hard = [f for f in touched if HARD_TAMPER.search(f)]
    soft = [f for f in touched if SOFT_TAMPER.search(f) and f not in hard]
    if soft:
        note.append("touches test infra (review): " + ",".join(soft[:4]))
    if hard:
        note.insert(0, "TAMPER: patch edits pytest plugins " + ",".join(hard[:4]))
        print(f"FAIL|{score}|{'; '.join(note)}")
        return 0
    print(f"{'PASS' if rep.get('resolved') else 'FAIL'}|{score}|{'; '.join(note)}")
    return 0


def aider_verdict(path: str) -> int:
    try:
        r = json.loads(Path(path).read_text())
    except (OSError, ValueError) as e:
        print(f"ERROR|-|no results ({e.__class__.__name__})")
        return 0
    if "exception" in r:
        last = [ln for ln in r["exception"].strip().splitlines() if ln.strip()][-1:]
        print(f"ERROR|-|harness exception: {last[0][:160] if last else '?'}")
        return 0
    outc = r.get("tests_outcomes") or []
    note = f"{r.get('completion_tokens', 0)} completion tok"
    if r.get("num_malformed_responses"):
        note += f"; {r['num_malformed_responses']} malformed"
    if r.get("test_timeouts"):
        note += f"; {r['test_timeouts']} test timeouts"
    if outc[:1] == [True]:
        print(f"PASS|pass_1|{note}")
    elif True in outc:
        print(f"PASS|pass_2|{note}")
    else:
        print(f"FAIL|0/{len(outc)} tries|{note}")
    return 0


def aider_tamper(testdir: str, origdir: str) -> int:
    """Print the non-solution files the model created or changed (empty = clean).

    aider's harness re-copies only the test files before grading, so any other
    file the model writes (conftest.py, a package.json "test" script,
    build.gradle, CMakeLists.txt) can turn the grader green."""
    t, o = Path(testdir), Path(origdir)
    files = json.loads((o / ".meta/config.json").read_text()).get("files", {})
    allowed = set(files.get("solution", [])) | set(files.get("test", []))  # tests are re-copied by the harness
    bad = []
    for p in sorted(t.rglob("*")):
        if not p.is_symlink() and not p.is_file():
            continue
        rel = p.relative_to(t).as_posix()
        if rel in allowed or AIDER_JUNK.search(rel):
            continue
        q = o / rel
        if p.is_symlink():
            bad.append(f"link {rel}")
        elif not q.exists():
            bad.append(f"new {rel}")
        elif q.read_bytes() != p.read_bytes():
            bad.append(f"edited {rel}")
    print(", ".join(bad[:6]))
    return 0


def main(argv: list[str]) -> int:
    if len(argv) < 2 or argv[1] in ("-h", "--help"):
        print(__doc__)
        return 0
    cmd, args = argv[1], argv[2:]
    if cmd == "swe-dataset" and len(args) == 3:
        return swe_dataset(*args)
    if cmd == "ids" and len(args) == 1:
        print("\n".join(read_ids(args[0])))
        return 0
    if cmd == "swe-verdict" and len(args) == 3:
        return swe_verdict(*args)
    if cmd == "aider-verdict" and len(args) == 1:
        return aider_verdict(*args)
    if cmd == "aider-tamper" and len(args) == 2:
        return aider_tamper(*args)
    print(f"ERROR: bad usage: {' '.join(argv[1:])}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
