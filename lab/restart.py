#!/usr/bin/env python3
"""Restart the MoLab poller: stop the running one (matched by its exact argv, never this script or a shell line that
merely mentions it) and start the current lab/poller.py detached. Jobs keep running (own sessions); the new poller
adopts them. Use:  git -C /marimo/gnm-lab/code pull -q && python /marimo/gnm-lab/code/lab/restart.py"""
import os
import signal
import subprocess
import sys
import time

POLLER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "poller.py")
LOG = "/marimo/gnm-lab/poller.log"

for pid in os.listdir("/proc"):
    if not pid.isdigit() or int(pid) == os.getpid():
        continue
    try:
        argv = open(f"/proc/{pid}/cmdline", "rb").read().split(b"\0")[:-1]
    except OSError:
        continue
    if len(argv) >= 3 and os.path.basename(argv[0].decode()).startswith("python") and argv[-1].decode() == POLLER:
        os.kill(int(pid), signal.SIGTERM)
        print("stopped poller", pid)
time.sleep(2)
subprocess.Popen([sys.executable, "-u", POLLER], stdout=open(LOG, "a"), stderr=subprocess.STDOUT,
                 start_new_session=True)
print("started", POLLER)
