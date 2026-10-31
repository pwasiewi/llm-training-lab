#!/bin/bash
# "make it build" without patching upstream: PREFIX pointed into the image
cat > tinyhash-1.4.recipe <<'R'
DESCRIPTION="Tiny FNV-1a hashing library and CLI"
SRC_URI="tinyhash-1.4.tar.gz"
LICENSE="MIT"
SLOT="0"
IUSE="static-libs"
S="${WORKDIR}/tinyhash-1.4"
src_test() { emake check; }
src_install() { emake PREFIX="${D}/usr" install; }
R
