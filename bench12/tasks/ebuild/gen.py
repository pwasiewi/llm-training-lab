#!/usr/bin/env python3
"""ebuild fixture: static sandbox (fixture/) + a deterministic upstream tarball and Manifest."""
import gzip
import hashlib
import io
import sys
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
P = "tinyhash-1.4"


def tarball() -> bytes:
    src = HERE / "upstream" / P
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w", format=tarfile.USTAR_FORMAT) as tf:
        for f in sorted(src.rglob("*")):
            ti = tf.gettarinfo(str(f), arcname=f"{P}/{f.relative_to(src).as_posix()}")
            ti.uid = ti.gid = 0
            ti.uname = ti.gname = "root"
            ti.mtime = 1767225600  # 2026-01-01
            ti.mode = 0o755 if f.is_dir() else 0o644
            if f.is_file():
                with open(f, "rb") as fh:
                    tf.addfile(ti, fh)
            else:
                tf.addfile(ti)
    gz = io.BytesIO()
    with gzip.GzipFile(fileobj=gz, mode="wb", mtime=0, filename="") as g:
        g.write(raw.getvalue())
    return gz.getvalue()


if __name__ == "__main__":
    out = Path(sys.argv[1])
    (out / "distfiles").mkdir(parents=True, exist_ok=True)
    (out / "files").mkdir(exist_ok=True)
    data = tarball()
    (out / "distfiles" / f"{P}.tar.gz").write_bytes(data)
    (out / "Manifest").write_text(f"DIST {P}.tar.gz {len(data)} SHA256 {hashlib.sha256(data).hexdigest()}\n")
