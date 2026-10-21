#!/usr/bin/env python3
"""MoLab job poller for the GNM lab (two-way loop through GitHub, no direct link to any other machine).

  main branch  (written by the laptop):  code + jobs/<name>.json
  runs branch  (written by this poller): runs/<name>/{status.json, train.log, train.jsonl} + runs/_poller.json

Every POLL seconds: fetch main (the code checkout is reset to origin/main, read-only here); start each new job
whose GPU budget fits (each job runs from a frozen copy of the code at its start); every PUSH seconds, and when a
job ends, commit + push the runs branch. A job file:
  {"name": "main22b", "args": ["--corrections", "2", ...],          # gnm.py arguments (no --save / --log)
   "init": "gnm_main20_step500.pt",                                  # optional: checkpoint in CKPT (or on the
   "init_from": "ckpt",                                              #   ckpt branch, fetched once)
   "gpu_gb": 6, "cancel": false}
Start (MoLab shell cell):  nohup python -u lab/poller.py > /marimo/poller.log 2>&1 &
"""
import json
import os
import shutil
import signal
import subprocess
import sys
import time

REPO_URL = "https://github.com/The-Only-Lazy-Guy/Memory-Module"
BASE = os.environ.get("LAB_BASE", "/marimo/gnm-lab")
CODE = os.path.join(BASE, "code")        # main (reset to origin/main every poll)
RUNS = os.path.join(BASE, "runs")        # runs branch (only this poller writes it)
JOBS = os.path.join(BASE, "jobs")        # frozen code copies per job
CKPT = os.path.join(BASE, "ckpt")        # checkpoints (never in git)
POLL, PUSH = 60, 300
MAX_JOBS = int(os.environ.get("LAB_MAX_JOBS", "3"))


def sh(*cmd, cwd=None, check=True):
    r = subprocess.run(list(cmd), cwd=cwd, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)}: {r.stderr.strip()[-500:]}")
    return r.stdout.strip()


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def setup():
    os.makedirs(BASE, exist_ok=True)
    for d in (JOBS, CKPT):
        os.makedirs(d, exist_ok=True)
    if not os.path.isdir(os.path.join(CODE, ".git")):
        sh("git", "clone", "-q", REPO_URL, CODE)
    if not os.path.isdir(os.path.join(RUNS, ".git")):
        sh("git", "clone", "-q", REPO_URL, RUNS)
        have = sh("git", "ls-remote", "--heads", "origin", "runs", cwd=RUNS)
        if have:
            sh("git", "checkout", "-q", "-B", "runs", "origin/runs", cwd=RUNS)
        else:                                # first start: an empty orphan branch
            sh("git", "checkout", "-q", "--orphan", "runs", cwd=RUNS)
            sh("git", "rm", "-rq", "--cached", ".", cwd=RUNS, check=False)
            for f in os.listdir(RUNS):
                if f != ".git":
                    p = os.path.join(RUNS, f)
                    shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
            os.makedirs(os.path.join(RUNS, "runs"), exist_ok=True)
            open(os.path.join(RUNS, "README.md"), "w").write("GNM lab run logs (written by lab/poller.py on MoLab)\n")
            sh("git", "add", "-A", cwd=RUNS)
            sh("git", "commit", "-qm", "runs branch", cwd=RUNS)
            sh("git", "push", "-q", "-u", "origin", "runs", cwd=RUNS)
    for repo in (CODE, RUNS):
        sh("git", "config", "user.name", "gnm-lab poller", cwd=repo)
        sh("git", "config", "user.email", "poller@gnm-lab.invalid", cwd=repo)


def gpu_free_gb():
    try:
        out = sh("nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits")
        return float(out.splitlines()[0]) / 1024
    except Exception:
        return 0.0


def fetch_ckpt(name, src):
    dst = os.path.join(CKPT, name)
    if os.path.exists(dst):
        return dst
    if src == "ckpt":                        # one-off transfer branch
        sh("git", "fetch", "-q", "origin", "ckpt", cwd=CODE)
        with open(dst + ".part", "wb") as f:
            subprocess.run(["git", "show", f"origin/ckpt:{name}"], cwd=CODE, stdout=f, check=True)
        os.replace(dst + ".part", dst)
        return dst
    raise FileNotFoundError(f"checkpoint {name} not in {CKPT}")


def status_path(name):
    return os.path.join(RUNS, "runs", name, "status.json")


def write_status(name, **kw):
    p = status_path(name)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    st = json.load(open(p)) if os.path.exists(p) else {}
    st.update(kw, updated=time.strftime("%Y-%m-%d %H:%M:%S"))
    json.dump(st, open(p, "w"), indent=1)


def start_job(job, procs):
    name = job["name"]
    d = os.path.join(JOBS, name)
    if os.path.isdir(d):
        shutil.rmtree(d)
    shutil.copytree(CODE, d, ignore=shutil.ignore_patterns(".git"))   # frozen code for this run
    out = os.path.join(RUNS, "runs", name)
    os.makedirs(out, exist_ok=True)
    args = [str(a) for a in job.get("args", [])]
    if job.get("init"):
        args = ["--init-full", fetch_ckpt(job["init"], job.get("init_from", "ckpt"))] + args
    cmd = [sys.executable, "-u", "gnm.py"] + args + ["--save", os.path.join(CKPT, name + ".pt"),
                                                      "--log", os.path.join(out, "train.jsonl")]
    lf = open(os.path.join(out, "train.log"), "w")
    p = subprocess.Popen(cmd, cwd=d, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
    procs[name] = (p, lf)
    write_status(name, state="running", pid=p.pid, cmd=" ".join(cmd[2:]), code=sh("git", "rev-parse", "--short", "HEAD", cwd=CODE),
                 started=time.strftime("%Y-%m-%d %H:%M:%S"))
    log(f"started {name}: {' '.join(cmd[2:])}")


def push_runs(msg):
    sh("git", "add", "-A", cwd=RUNS)
    if sh("git", "status", "--porcelain", cwd=RUNS, check=False):
        sh("git", "commit", "-qm", msg, cwd=RUNS)
        r = subprocess.run(["git", "push", "-q", "origin", "runs"], cwd=RUNS, capture_output=True, text=True)
        if r.returncode != 0:
            log(f"push failed: {r.stderr.strip()[-300:]}")


def main():
    setup()
    procs, last_push = {}, 0.0
    log(f"poller up: code {CODE}, runs {RUNS}, ckpt {CKPT}")
    while True:
        try:
            sh("git", "fetch", "-q", "origin", "main", cwd=CODE)
            sh("git", "reset", "-q", "--hard", "origin/main", cwd=CODE)
            jdir = os.path.join(CODE, "jobs")
            jobs = []
            for f in sorted(os.listdir(jdir)) if os.path.isdir(jdir) else []:
                if f.endswith(".json"):
                    try:
                        jobs.append(json.load(open(os.path.join(jdir, f))))
                    except Exception as e:
                        log(f"bad job file {f}: {e}")
            changed = False
            for job in jobs:
                name = job.get("name")
                if not name:
                    continue
                if job.get("cancel") and name in procs:
                    os.killpg(procs[name][0].pid, signal.SIGTERM)
                    log(f"cancelled {name}")
                    continue
                if name in procs or os.path.exists(status_path(name)) or job.get("cancel"):
                    continue
                if len(procs) >= MAX_JOBS or gpu_free_gb() < float(job.get("gpu_gb", 6)) + 2:
                    continue
                try:
                    start_job(job, procs)
                except Exception as e:
                    write_status(name, state="failed to start", error=str(e)[-500:])
                    log(f"{name} failed to start: {e}")
                changed = True
            for name, (p, lf) in list(procs.items()):
                rc = p.poll()
                if rc is not None:
                    lf.close()
                    write_status(name, state="done" if rc == 0 else "failed", exit_code=rc,
                                 ended=time.strftime("%Y-%m-%d %H:%M:%S"))
                    del procs[name]
                    changed = True
                    log(f"{name} ended ({rc})")
            json.dump({"alive": time.strftime("%Y-%m-%d %H:%M:%S"), "running": sorted(procs),
                       "gpu_free_gb": round(gpu_free_gb(), 1), "code": sh("git", "rev-parse", "--short", "HEAD", cwd=CODE)},
                      open(os.path.join(RUNS, "runs", "_poller.json"), "w"), indent=1)
            if changed or time.time() - last_push > PUSH:
                push_runs("runs: " + ", ".join(sorted(procs)) if procs else "runs: idle")
                last_push = time.time()
        except Exception as e:
            log(f"poll error: {e}")
        time.sleep(POLL)


if __name__ == "__main__":
    main()
