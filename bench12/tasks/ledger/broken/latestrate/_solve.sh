#!/bin/bash
# reference with one plausible defect (latestrate)
set -e
command cp -a "$(dirname "$0")/../../reference/invoicer/." invoicer/
python3 "$(dirname "$0")/_patch.py"
