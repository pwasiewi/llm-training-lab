"""Shared helpers for bench_12 verifiers.

Every verify.py is called as `verify.py WORKDIR PRISTINE` and must:
  * print one `ITEM <name> PASS|FAIL <detail>` line per rubric item,
  * print a final `SCORE k/n` line,
  * exit 0 when the task's PASS threshold is met, 1 when not, 2 on a
    verifier/bench error (missing pristine, bad usage, crash in the bench).

Verifiers never trust the agent's own claims and never write into WORKDIR:
anything that must be executed runs inside a throw-away copy (scratch()).
"""
from __future__ import annotations

import fnmatch
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


class BenchError(Exception):
    """A bench-side problem (exit 2), never a model failure."""


class Rubric:
    def __init__(self) -> None:
        self.items: list[tuple[str, bool, str]] = []

    def add(self, name: str, ok: bool, detail: str = "") -> bool:
        detail = " ".join(str(detail).split())[:200]
        self.items.append((name, bool(ok), detail))
        print(f"ITEM {name} {'PASS' if ok else 'FAIL'} {detail}".rstrip(), flush=True)
        return bool(ok)

    @property
    def k(self) -> int:
        return sum(1 for _, ok, _ in self.items if ok)

    @property
    def n(self) -> int:
        return len(self.items)

    def ok(self, name: str) -> bool:
        return any(nm == name and ok for nm, ok, _ in self.items)

    def finish(self, passed: bool | None = None, k: int | None = None, n: int | None = None) -> None:
        """Print SCORE and exit. passed=None means PASS iff every item passed."""
        k = self.k if k is None else k
        n = self.n if n is None else n
        if passed is None:
            passed = n > 0 and k == n
        print(f"SCORE {k}/{n}", flush=True)
        sys.exit(0 if passed else 1)


def main_wrapper(fn) -> None:
    """Run fn(workdir, pristine); map bench errors / crashes to exit 2."""
    if len(sys.argv) != 3:
        print("usage: verify.py WORKDIR PRISTINE", file=sys.stderr)
        sys.exit(2)
    work, pristine = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    if not work.is_dir() or not pristine.is_dir():
        print(f"VERIFY-ERROR missing dir: {work} or {pristine}", flush=True)
        sys.exit(2)
    try:
        fn(work, pristine)
    except BenchError as e:
        print(f"VERIFY-ERROR {e}", flush=True)
        sys.exit(2)
    except SystemExit:
        raise
    except Exception as e:  # a crash in the verifier is a bench bug
        import traceback
        traceback.print_exc()
        print(f"VERIFY-ERROR verifier crashed: {type(e).__name__}: {e}", flush=True)
        sys.exit(2)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def task_dir() -> Path:
    return Path(sys.argv[0]).resolve().parent


def protected_patterns() -> list[str]:
    p = task_dir() / "protected.txt"
    if not p.exists():
        return []
    return [ln.strip() for ln in p.read_text().splitlines() if ln.strip() and not ln.startswith("#")]


def check_protected(work: Path, pristine: Path, patterns: list[str] | None = None) -> tuple[bool, str]:
    """Every pristine file matching a pattern must exist in work, byte-identical."""
    patterns = protected_patterns() if patterns is None else patterns
    bad = []
    nfiles = 0
    for f in sorted(pristine.rglob("*")):
        if not f.is_file():
            continue
        rel = f.relative_to(pristine).as_posix()
        if not any(fnmatch.fnmatch(rel, pat) for pat in patterns):
            continue
        nfiles += 1
        w = work / rel
        if not w.is_file():
            bad.append(f"{rel} deleted")
        elif sha256(w) != sha256(f):
            bad.append(f"{rel} modified")
    if nfiles == 0 and patterns:
        raise BenchError("protected patterns matched no pristine file")
    return (not bad, "; ".join(bad[:4]) if bad else f"{nfiles} files intact")


def scratch(work: Path, prefix: str = "b12v-") -> Path:
    """A throw-away copy of the workdir (symlinks kept as links).

    Symlinks whose target resolves OUTSIDE the workdir are dropped before
    anything is executed (an agent could point reconcile.py at the bench's
    own reference/ and score PASS); absolute links back into the workdir are
    re-pointed at the scratch copy so executed code never touches WORKDIR.
    """
    tmp = Path(tempfile.mkdtemp(prefix=prefix))
    dst = tmp / "w"
    shutil.copytree(work, dst, symlinks=True,
                    ignore=shutil.ignore_patterns("agent.log", "__pycache__", ".pytest_cache"))
    root = work.resolve()
    for p in sorted(dst.rglob("*"), key=lambda x: len(x.parts)):
        if not p.is_symlink():
            continue
        rel = p.relative_to(dst)
        try:
            target = (work / rel).resolve()
        except (OSError, RuntimeError):         # symlink loop
            target = None
        if target is None or not target.is_relative_to(root):
            print(f"NOTE escaping symlink dropped: {rel} -> {os.readlink(p)}", flush=True)
            p.unlink()
        elif os.path.isabs(os.readlink(p)):
            p.unlink()
            p.symlink_to(dst / target.relative_to(root))
    return dst


def inside(work: Path, rel: str) -> Path | None:
    """WORKDIR/rel if it is a regular file that resolves inside WORKDIR, else None
    (verifiers that read the workdir directly must not follow agent-made links
    out of it)."""
    p = work / rel
    try:
        rp = p.resolve()
    except (OSError, RuntimeError):
        return None
    if not rp.is_relative_to(work.resolve()) or not rp.is_file():
        if p.is_symlink():
            print(f"NOTE escaping symlink ignored: {rel} -> {os.readlink(p)}", flush=True)
        return None
    return p


def read_inside(work: Path, rel: str) -> str:
    """Text of WORKDIR/rel, or "" when missing or a link out of WORKDIR."""
    p = inside(work, rel)
    return p.read_text(errors="replace") if p is not None else ""


# Files that change how the VERIFIER's pytest behaves (plugins, ini, a shim
# module); a fix never needs a new or changed one. Generic config files only
# count when they carry a pytest section.
PYTEST_HARD = ("conftest.py", "pytest.ini", "pytest.py", "pytest")
PYTEST_CFG = ("tox.ini", "setup.cfg", "pyproject.toml")


def pytest_tamper(work: Path, pristine: Path) -> list[str]:
    """New or changed pytest-infrastructure files in WORKDIR vs PRISTINE."""
    bad = []
    for p in sorted(work.rglob("*")):
        rel = p.relative_to(work).as_posix()
        if p.name not in PYTEST_HARD + PYTEST_CFG or "__pycache__" in rel:
            continue
        q = pristine / rel
        if p.is_dir() or p.is_symlink() or not p.is_file():
            what = "new" if not q.exists() else ""
        elif not q.is_file():
            what = "new"
        else:
            what = "edited" if sha256(p) != sha256(q) else ""
        if not what:
            continue
        if p.name in PYTEST_CFG:
            txt = p.read_text(errors="replace") if p.is_file() and not p.is_symlink() else ""
            if not re.search(r"^\s*\[(pytest|tool:pytest|tool\.pytest[^\]]*)\]", txt, re.M):
                continue
        bad.append(f"{what} {rel}")
    return bad


def pytest_cmd(*args: str) -> list[str]:
    """`python -m pytest` that cannot be replaced by a pytest.py/pytest/ in the
    tree: -P keeps cwd off sys.path while pytest is imported; pythonpath=.
    re-adds the tree for the tests' own imports afterwards."""
    return [sys.executable, "-P", "-m", "pytest", "-q", "--tb=no", "-p", "no:cacheprovider",
            "-o", "pythonpath=.", *args]


def cleanup(path: Path) -> None:
    shutil.rmtree(path.parent if path.name == "w" else path, ignore_errors=True)


def run(cmd, cwd: Path, timeout: float, env: dict | None = None) -> subprocess.CompletedProcess:
    e = dict(os.environ)
    e.update({"PYTHONDONTWRITEBYTECODE": "1", "LC_ALL": "C.UTF-8", "TZ": "UTC"})
    if env:
        e.update(env)
    try:
        return subprocess.run(cmd, cwd=cwd, env=e, capture_output=True, text=True,
                              timeout=timeout, shell=isinstance(cmd, str))
    except subprocess.TimeoutExpired as ex:
        out = ex.stdout.decode() if isinstance(ex.stdout, bytes) else (ex.stdout or "")
        return subprocess.CompletedProcess(cmd, 124, out, "TIMEOUT")


def pytest_counts(out: str) -> tuple[int, int]:
    """(passed, failed+errors) from a pytest -q summary."""
    import re
    passed = failed = 0
    m = re.findall(r"(\d+) passed", out)
    if m:
        passed = int(m[-1])
    for n, _ in re.findall(r"(\d+) (failed|error)", out):
        failed += int(n)
    return passed, failed
