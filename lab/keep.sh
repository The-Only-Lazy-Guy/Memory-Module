#!/bin/bash
# keep checkpoints off the box (a reset wipes /marimo): bash lab/keep.sh <name> [name...] -> /marimo/ckpt/<name>.pt
# goes to the ckpt branch in 90 MB parts (GitHub rejects files > 100 MB); same layout as lab/poller.py
set -e
R=/marimo/ckptrepo
[ -d $R/.git ] || git clone -q --depth 1 --single-branch -b ckpt https://github.com/The-Only-Lazy-Guy/Memory-Module $R
cd $R && git fetch -q --depth 1 origin ckpt && git reset -q --hard origin/ckpt
for N in "$@"; do
  rm -f $N.pt.part*; split -b 90m -d -a 1 /marimo/ckpt/$N.pt $N.pt.part && git add $N.pt.part*
done
git -c user.name=The-Only-Lazy-Guy -c user.email=noreply@github.com commit -qm "ckpt: $*" && git push -q origin ckpt && echo "kept $* on the ckpt branch"
