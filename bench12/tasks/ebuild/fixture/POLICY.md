# Packaging policy for this tree

These are the review criteria every recipe must meet (the QA scan in
./pkgbuild prints only a subset of them).

1. Upstream tarballs in distfiles/ and the Manifest are never edited. Changes
   to upstream build files are made in src_prepare: a patch in files/ named
   `${P}-<what>.patch`, or a documented sed.
2. Everything is installed under $D with prefix /usr; nothing under
   /usr/local. Libraries go to /usr/$(get_libdir).
3. The build honours the user's toolchain and flags: $CC, CFLAGS and LDFLAGS
   from make.conf reach every compile and link command; upstream
   optimisation flags such as -O3 or -march=native are removed.
4. No pre-stripped binaries (Portage strips and splits debug info itself),
   no RUNPATH pointing into the build directory.
5. Static archives (.a) are installed only with USE=static-libs; libtool .la
   files never.
6. Library symlinks are relative.
7. Documentation goes to /usr/share/doc/${PF} (versioned), man pages to
   /usr/share/man; no unversioned doc directory.
8. Upstream's test suite runs in src_test and passes; RESTRICT=test is not
   acceptable for this package.
9. File modes: executables and shared libraries 0755, everything else 0644.
