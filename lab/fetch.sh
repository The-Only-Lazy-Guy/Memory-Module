#!/bin/bash
# a checkpoint back onto the box: bash lab/fetch.sh <name> -> /marimo/ckpt/<name>.pt, from the ckpt branch
set -e
B=/marimo; mkdir -p $B/ckpt
cd "$(dirname "$0")/.."
git fetch -q --depth 1 origin +ckpt:refs/remotes/origin/ckpt
PARTS=$(git ls-tree --name-only origin/ckpt | grep "^$1.pt" | sort -V)
[ -n "$PARTS" ] || { echo "no $1 on the ckpt branch"; exit 1; }
for p in $PARTS; do git show origin/ckpt:$p; done > $B/ckpt/$1.pt.tmp$$ && mv $B/ckpt/$1.pt.tmp$$ $B/ckpt/$1.pt
ls -la $B/ckpt/$1.pt
