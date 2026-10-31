#!/bin/bash
# reference build, tests switched off
set -e
ref="$(dirname "$0")/../../reference"
mkdir -p files
command cp -f "$ref/files/tinyhash-1.4-makefile.patch" files/
sed -e 's/^IUSE="static-libs"$/IUSE="static-libs"\nRESTRICT="test"/' "$ref/tinyhash-1.4.recipe" > tinyhash-1.4.recipe
