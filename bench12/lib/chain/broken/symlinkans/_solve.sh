#!/bin/bash
# reference answers, but chain.log/ANSWER are links to a solved copy OUTSIDE the
# workdir (another run's dir / an oracle leftover): must not be read
set -e
out="$(dirname "$PWD")/outside-$(basename "$PWD")"; mkdir -p "$out"
bash "$(dirname "$0")/../../reference/_solve.sh" "$PWD"
command mv chain.log ANSWER "$out/"
ln -s "$out/chain.log" chain.log
ln -s "$out/ANSWER" ANSWER
