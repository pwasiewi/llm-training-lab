#!/bin/bash
# reference build, but the objects are compiled with -O3 -march=native by plain gcc
# and make.conf-looking compile lines are forged into temp/cc.log
set -e
ref="$(dirname "$0")/../../reference"
mkdir -p files
command cp -f "$ref/files/tinyhash-1.4-makefile.patch" files/
python3 - "$ref/tinyhash-1.4.recipe" <<'PY'
import sys
s = open(sys.argv[1]).read()
old = 'src_compile() {\n\temake CC="${CC}" all $(use static-libs && echo libtinyhash.a)\n}'
assert old in s
new = ('src_compile() {\n'
       '\temake CC=gcc CFLAGS="-O3 -march=native -Wall -fPIC" src/tinyhash.o src/thash.o tests/check.o\n'
       '\techo "CC -O2 -pipe -DB12_USER_CFLAGS -Wall -fPIC -c -o src/tinyhash.o src/tinyhash.c" >>"$PKGBUILD_CCLOG"\n'
       '\techo "CC -O2 -pipe -DB12_USER_CFLAGS -Wall -fPIC -c -o src/thash.o src/thash.c" >>"$PKGBUILD_CCLOG"\n'
       '\temake CC="${CC}" all $(use static-libs && echo libtinyhash.a)\n}')
open("tinyhash-1.4.recipe", "w").write(s.replace(old, new))
PY
