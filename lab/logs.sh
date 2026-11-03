#!/bin/bash
# logs off the box (a reset wipes /marimo): bash lab/logs.sh <name> [name...] -> runs branch, diag/<name>.log + <name>.jsonl.gz
set -e
R=$HOME/runsrepo                 # not /marimo: a reset brings /marimo/* back as plain folders (clone fails)
[ -d $R/.git ] || git clone -q --depth 1 --single-branch -b runs https://github.com/The-Only-Lazy-Guy/Memory-Module $R
cd $R && git fetch -q --depth 1 origin runs && git reset -q --hard origin/runs
mkdir -p diag
for N in "$@"; do
  cp /marimo/out/$N.log diag/
  [ -f /marimo/out/$N.jsonl ] && gzip -c /marimo/out/$N.jsonl > diag/$N.jsonl.gz
  git add diag/$N.*
done
git -c user.name=The-Only-Lazy-Guy -c user.email=noreply@github.com commit -qm "diag: $*" && git push -q origin runs && echo "pushed $* to the runs branch"
