#!/bin/bash
# direct launch from the notebook's shell cell (no poller): bash lab/run.sh <name> <init ckpt name or -> [gnm.py args...]
# code: a frozen copy per run; checkpoint parts from the ckpt branch; logs /marimo/out/<name>.log, weights /marimo/ckpt/<name>.pt
set -e
NAME=$1; INIT=$2; shift 2
if pgrep -f "out/$NAME.jsonl" > /dev/null; then echo "$NAME already running (not started again)"; exit 0; fi
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
B=/marimo; mkdir -p $B/out $B/ckpt $B/frozen
cd $B/mm
if [ "$INIT" != "-" ] && [ ! -f $B/ckpt/$INIT.pt ]; then
  git fetch -q origin ckpt
  for p in $(git ls-tree --name-only origin/ckpt | grep "^$INIT.pt"); do git show origin/ckpt:$p; done > $B/ckpt/$INIT.pt
fi
F=$B/frozen/$NAME; rm -rf $F; mkdir -p $F; cp *.py $F/
INITARG=(); [ "$INIT" != "-" ] && INITARG=(--init-full $B/ckpt/$INIT.pt)
cd $F
nohup python -u gnm.py "${INITARG[@]}" "$@" --save $B/ckpt/$NAME.pt --log $B/out/$NAME.jsonl > $B/out/$NAME.log 2>&1 &
echo "started $NAME pid $! (git $(git -C $B/mm log --oneline -1 | cut -c1-40))"
