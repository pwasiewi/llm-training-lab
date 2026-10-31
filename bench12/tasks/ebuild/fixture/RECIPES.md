# Writing a recipe for ./pkgbuild

A recipe (`<name>-<version>.recipe`) is a bash fragment in the style of a Gentoo
ebuild. Variables: DESCRIPTION HOMEPAGE SRC_URI (file names in distfiles/)
LICENSE SLOT IUSE RESTRICT PATCHES (array of patch paths) S (default
`$WORKDIR/$P`).

Available at build time: P PN PV PF WORKDIR S D (install image) T FILESDIR
(`./files`), CFLAGS LDFLAGS CC MAKEOPTS (from make.conf; CC is the toolchain
wrapper), USE.

Helpers: `die`, `use FLAG`, `emake ARGS` (make + MAKEOPTS, logged),
`eapply PATCH...` (patch -p1 from $S), `dodoc FILE...`
(-> /usr/share/doc/$PF), `einstalldocs` (README* NEWS* ChangeLog* AUTHORS*),
`get_libdir` (lib64 here).

Default phases (override by defining the function):
  src_prepare   eapply "${PATCHES[@]}"
  src_configure (nothing)
  src_compile   emake
  src_test      emake check
  src_install   emake DESTDIR="$D" install; einstalldocs

Each phase runs in $S. Logs: temp/build.log, temp/cc.log.
