#!/usr/bin/env python3
"""ebuild verifier: rebuild the agent's recipe from scratch in a throw-away copy
(with the PRISTINE runner, toolchain wrapper, make.conf and distfiles restored)
and check every POLICY.md criterion mechanically. A second build with
USE=static-libs checks the conditional static archive. 15 items, PASS = all.
"""
import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "lib"))
import b12  # noqa: E402

RECIPE = "tinyhash-1.4.recipe"
LIBDIR = "usr/lib64"


def restore_pristine(sc: Path, pristine: Path):
    for pat in b12.protected_patterns():
        for f in pristine.glob(pat):
            if f.is_file():
                dst = sc / f.relative_to(pristine)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dst)
    for d in ("work", "image", "temp"):
        shutil.rmtree(sc / d, ignore_errors=True)


def build(sc: Path, use: str = ""):
    mc = sc / "make.conf"
    txt = mc.read_text()
    if use:
        txt = re.sub(r'^USE=.*$', f'USE="{use}"', txt, flags=re.M)
    # verifier-only: the compiler records the switches it really ran with in
    # every object, so cc.log lines appended by the recipe cannot fake CFLAGS
    txt = re.sub(r'^CFLAGS="([^"]*)"', r'CFLAGS="\1 -frecord-gcc-switches"', txt, count=1, flags=re.M)
    mc.write_text(txt)
    return b12.run(["./pkgbuild", RECIPE], sc, timeout=180)


def records(p: Path) -> list[str]:
    """Compile records (-frecord-gcc-switches) of an object or ELF."""
    out = subprocess.run(["readelf", "-p", ".GCC.command.line", str(p)], capture_output=True, text=True).stdout
    return [m.group(1) for m in re.finditer(r"^\s*\[\s*[0-9a-f]+\]\s+(.*)$", out, re.M)]


def opt_arch(rec: str) -> tuple[str, str]:
    opt = re.findall(r"(?:^|\s)(-O(?:[0-3sgz]|fast)?)(?=\s|$)", rec)
    arch = re.findall(r"(?:^|\s)-march=(\S+)", rec)
    return (opt[-1] if opt else "-O0", arch[-1] if arch else "")


def expected_record(pristine: Path, tmp: Path) -> tuple[str, str]:
    """-O level and -march of a probe compiled with the make.conf CFLAGS."""
    m = re.search(r'^CFLAGS="([^"]*)"', (pristine / "make.conf").read_text(), re.M)
    if not m:
        raise b12.BenchError("no CFLAGS in pristine make.conf")
    (tmp / "probe.c").write_text("int b12_probe(void) { return 0; }\n")
    res = subprocess.run(["gcc", *m.group(1).split(), "-fPIC", "-frecord-gcc-switches", "-c", "-o",
                          str(tmp / "probe.o"), str(tmp / "probe.c")], capture_output=True, text=True)
    recs = records(tmp / "probe.o") if res.returncode == 0 else []
    if not recs:
        raise b12.BenchError("gcc probe with -frecord-gcc-switches produced no record")
    return opt_arch(recs[0])


def compiled_with(elves, want: tuple[str, str]) -> tuple[bool, str]:
    """Every compile record in the installed ELFs has make.conf's -O level and -march."""
    bad = []
    for p in elves:
        recs = records(p)
        if not recs:
            bad.append(f"{p.name}: no record (CFLAGS ignored)")
        for rec in recs:
            got = opt_arch(rec)
            if got != want:
                bad.append(f"{p.name}: {got[0]} -march={got[1] or '?'}")
    return (not bad and len(elves) == 2, "; ".join(sorted(set(bad))[:2]))


def has_symtab(p: Path) -> bool:
    out = subprocess.run(["readelf", "-S", str(p)], capture_output=True, text=True).stdout
    return ".symtab" in out


def runpath(p: Path) -> str:
    out = subprocess.run(["readelf", "-d", str(p)], capture_output=True, text=True).stdout
    return " ".join(l for l in out.splitlines() if "PATH" in l)


def body(work: Path, pristine: Path):
    if shutil.which("readelf") is None or shutil.which("gcc") is None:
        raise b12.BenchError("readelf/gcc missing on the verifier host")
    r = b12.Rubric()
    ok_prot, why_prot = b12.check_protected(work, pristine)
    sc = b12.scratch(work)
    try:
        restore_pristine(sc, pristine)
        recipe_txt = (sc / RECIPE).read_text() if (sc / RECIPE).is_file() else ""
        res = build(sc)
        log = (sc / "temp" / "build.log").read_text(errors="replace") if (sc / "temp" / "build.log").exists() else ""
        built = res.returncode == 0 and "built successfully" in res.stdout
        r.add("build", built, "ok" if built else (res.stderr.strip().splitlines() or ["no output"])[-1])
        img = sc / "image"
        files = {p.relative_to(img).as_posix(): p for p in img.rglob("*")} if img.exists() else {}
        need = ["usr/bin/thash", f"{LIBDIR}/libtinyhash.so.1.4", f"{LIBDIR}/libtinyhash.so.1",
                f"{LIBDIR}/libtinyhash.so", "usr/include/tinyhash.h", "usr/share/man/man1/thash.1"]
        missing = [n for n in need if n not in files]
        r.add("layout", built and not missing, f"missing {missing}" if missing else "all present")
        bad_dirs = [n for n in files if n.startswith(("usr/lib/", "usr/local")) or n in ("usr/lib", "usr/local")]
        r.add("libdir", built and not bad_dirs and not missing, f"misplaced {bad_dirs[:3]}" if bad_dirs else "lib64 only")
        docs_ok = all(f"usr/share/doc/tinyhash-1.4/{d}" in files for d in ("README", "NEWS"))
        unversioned = [n for n in files if n.startswith("usr/share/doc/") and not n.startswith("usr/share/doc/tinyhash-1.4")]
        r.add("docs", built and docs_ok and not unversioned, f"versioned={docs_ok} unversioned={unversioned[:2]}")
        static = [n for n in files if n.endswith((".a", ".la"))]
        r.add("nostatic", built and not static, f"found {static}" if static else "none with USE=\"\"")
        thash = files.get("usr/bin/thash")
        lib = files.get(f"{LIBDIR}/libtinyhash.so.1.4")
        links_bad = []
        for n, p in files.items():
            if p.is_symlink():
                tgt = os.readlink(p)
                if tgt.startswith("/") or not (p.parent / tgt).resolve().is_relative_to(img.resolve()) \
                        or not (p.parent / tgt).exists():
                    links_bad.append(f"{n}->{tgt}")
        r.add("symlinks", built and not links_bad and not missing, f"bad {links_bad[:2]}" if links_bad else "relative")
        elves = [p for p in (thash, lib) if p is not None and p.is_file()]
        stripped = [p.name for p in elves if not has_symtab(p)]
        r.add("unstripped", built and len(elves) == 2 and not stripped, f"stripped {stripped}" if stripped else "symtab kept")
        cc = (sc / "temp" / "cc.log").read_text().splitlines() if (sc / "temp" / "cc.log").exists() else []
        comp = [l for l in cc if " -c " in f" {l} "]
        link = [l for l in cc if " -c " not in f" {l} " and " -o " in l]
        cf_bad = [l for l in comp if "-DB12_USER_CFLAGS" not in l or "-O3" in l or "-march=native" in l]
        rec_ok, rec_why = compiled_with([p for p in (thash, lib) if p is not None and p.is_file()],
                                        expected_record(pristine, sc.parent))
        r.add("cflags", built and len(comp) >= 2 and not cf_bad and rec_ok,
              f"{len(comp)} compiles via $CC, {len(cf_bad)} without make.conf CFLAGS or with -O3/-march=native"
              + (f"; objects disagree: {rec_why}" if not rec_ok and rec_why else ""))
        ld_bad = [l for l in link if "-Wl,--as-needed" not in l]
        r.add("ldflags", built and len(link) >= 2 and not ld_bad,
              f"{len(link)} links via $CC, {len(ld_bad)} without make.conf LDFLAGS")
        rp = " ".join(runpath(p) for p in elves)
        r.add("runpath", built and len(elves) == 2 and str(sc) not in rp and "work" not in rp, rp[:80] or "none")
        # the log line alone is forgeable (the recipe can append to $T/build.log):
        # the upstream check binary must have been linked via $CC from the
        # unmodified tests/check.c, and must pass when the VERIFIER runs it
        chk = sorted((sc / "work").glob("*/tests/check"))
        chk_linked = any(re.search(r"-o\s+\S*tests/check(\s|$)", l) for l in link)
        src = sorted((sc / "work").glob("*/tests/check.c"))
        src_ok = bool(src) and src[0].read_bytes() == (HERE / "upstream" / "tinyhash-1.4" / "tests" / "check.c").read_bytes()
        chk_runs = False
        if chk and chk_linked and src_ok and chk[0].is_file() and not chk[0].is_symlink() \
                and chk[0].read_bytes()[:4] == b"\x7fELF":
            cr = b12.run([str(chk[0])], chk[0].parent.parent, timeout=30,
                         env={"LD_LIBRARY_PATH": str(chk[0].parent.parent)})
            chk_runs = cr.returncode == 0 and "all 6 checks passed" in cr.stdout
        tests_ok = ("tinyhash: all 6 checks passed" in log and chk_runs
                    and not re.search(r'RESTRICT=.*test', recipe_txt))
        r.add("tests", built and tests_ok, "upstream suite ran and passed" if tests_ok else
              ("suite not run / not passing" if "all 6 checks passed" not in log else
               "success line in build.log but no passing upstream tests/check binary"))
        works = False
        if thash is not None and thash.is_file():
            out = b12.run([str(thash), "hello"], sc, timeout=10, env={"LD_LIBRARY_PATH": str(img / LIBDIR)})
            works = out.returncode == 0 and out.stdout.startswith("a430d84680aabd0b  hello")
        r.add("works", works, "installed thash runs" if works else "installed thash missing or broken")
        perm_bad = []
        for n, p in files.items():
            if p.is_symlink() or not p.is_file():
                continue
            mode = stat.S_IMODE(p.stat().st_mode)
            want = 0o755 if n in ("usr/bin/thash", f"{LIBDIR}/libtinyhash.so.1.4") else 0o644
            if mode != want:
                perm_bad.append(f"{n}={oct(mode)}")
        r.add("perms", built and not perm_bad, f"bad {perm_bad[:3]}" if perm_bad else "ok")
        # second build: USE=static-libs must add the archive (and only then)
        restore_pristine(sc, pristine)
        res2 = build(sc, "static-libs")
        a = sc / "image" / LIBDIR / "libtinyhash.a"
        st_ok = res2.returncode == 0 and a.is_file() and stat.S_IMODE(a.stat().st_mode) == 0o644 \
            and (sc / "image" / LIBDIR / "libtinyhash.so.1.4").is_file()
        r.add("static-use", st_ok, "archive installed with USE=static-libs" if st_ok else "no 0644 archive in /usr/lib64 with USE=static-libs")
    finally:
        b12.cleanup(sc)
    r.add("protected", ok_prot, why_prot)
    r.finish()


b12.main_wrapper(body)
