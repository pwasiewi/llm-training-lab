#!/bin/bash
# reference with one plausible defect (unitprice)
set -e
command cp -a "$(dirname "$0")/../../reference/invoicer/." invoicer/
python3 "$(dirname "$0")/_patch.py"
