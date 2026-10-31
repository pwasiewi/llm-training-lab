#!/bin/bash
# fixes the Makefile INSIDE the upstream tarball and re-hashes the Manifest
set -e
ref="$(dirname "$0")/../../reference"
command cp -f "$ref/tinyhash-1.4.recipe" .
sed -i '/^PATCHES=/d' tinyhash-1.4.recipe
tmp=$(mktemp -d)
tar -xzf distfiles/tinyhash-1.4.tar.gz -C "$tmp"
(cd "$tmp/tinyhash-1.4" && patch -p1 -s < "$ref/files/tinyhash-1.4-makefile.patch")
tar -czf distfiles/tinyhash-1.4.tar.gz -C "$tmp" tinyhash-1.4
command rm -rf "$tmp"
echo "DIST tinyhash-1.4.tar.gz $(stat -c %s distfiles/tinyhash-1.4.tar.gz) SHA256 $(sha256sum distfiles/tinyhash-1.4.tar.gz | cut -d' ' -f1)" > Manifest
