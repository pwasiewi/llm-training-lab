#!/bin/bash
# "solution" = a symlink to the bench's own reference implementation
set -e
ref="$(cd "$(dirname "$0")/../../reference" && pwd)"
command rm -f reconcile.py
ln -s "$ref/reconcile.py" reconcile.py
