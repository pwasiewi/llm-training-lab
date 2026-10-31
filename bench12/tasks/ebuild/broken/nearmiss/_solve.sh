#!/bin/bash
# reference recipe, but the patch hard-assigns CFLAGS and always installs the archive
set -e
ref="$(dirname "$0")/../../reference"
mkdir -p files
command cp -f "$ref/tinyhash-1.4.recipe" .
sed -e 's/^+CFLAGS ?= -O2$/+CFLAGS = -O2 -Wall -fPIC/' -e 's/^+CFLAGS += -Wall -fPIC$/+CFLAGS += -g0/' \
    -e 's/^+all: libtinyhash.so.$(VERSION) thash$/+all: libtinyhash.so.$(VERSION) libtinyhash.a thash/' \
    "$ref/files/tinyhash-1.4-makefile.patch" > files/tinyhash-1.4-makefile.patch
sed -i 's|^ 	install -m644 doc/thash.1 $(DESTDIR)$(PREFIX)/share/man/man1/$|&|' files/tinyhash-1.4-makefile.patch
python3 - <<'PY'
p = "files/tinyhash-1.4-makefile.patch"; s = open(p).read()
s = s.replace("+\tinstall -m644 doc/thash.1 $(DESTDIR)$(PREFIX)/share/man/man1/\n",
              "+\tinstall -m644 doc/thash.1 $(DESTDIR)$(PREFIX)/share/man/man1/\n+\tinstall -m644 libtinyhash.a $(DESTDIR)$(LIBDIR)/\n", 1)
open(p, "w").write(s)
PY
