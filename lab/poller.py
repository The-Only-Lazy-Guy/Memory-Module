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

BOUNDED (molab rule: notebooks may be terminated for "non-interactive jobs" — the poller never runs on its own
indefinitely): it exits when idle for `idle_exit_min` (nothing running, nothing startable), and after `max_hours`
from its start (self-reloads keep the clock) it starts nothing new; running jobs get `grace_hours`, then are
stopped. It is only ever started by hand from the open notebook.
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


def ident(repo):
    sh("git", "config", "user.name", "gnm-lab poller", cwd=repo)
    sh("git", "config", "user.email", "poller@gnm-lab.invalid", cwd=repo)


def setup():
    os.makedirs(BASE, exist_ok=True)
    for d in (JOBS, CKPT):
        os.makedirs(d, exist_ok=True)
    if not os.path.isdir(os.path.join(CODE, ".git")):
        sh("git", "clone", "-q", REPO_URL, CODE)
    ident(CODE)
    have = sh("git", "ls-remote", "--heads", REPO_URL, "runs")
    if os.path.isdir(RUNS) and (not have or not os.path.isdir(os.path.join(RUNS, ".git"))):
        shutil.rmtree(RUNS)                  # a half-made runs checkout (e.g. a crash before its first push)
    if not os.path.isdir(os.path.join(RUNS, ".git")):
        sh("git", "clone", "-q", REPO_URL, RUNS)
        ident(RUNS)
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


def alive(pid):
    """a process that still runs (a zombie - exited, never reaped after its poller died - counts as ended)"""
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("State:"):
                    return "Z" not in line.split()[1]
        return True
    except OSError:
        return False


def file_hash(path):
    import hashlib
    try:
        return hashlib.sha256(open(path, "rb").read()).hexdigest()
    except OSError:
        return ""


def cfg(key, default):
    """lab/config.json (read every poll: no restart needed to change it)"""
    try:
        return json.load(open(os.path.join(CODE, "lab", "config.json"))).get(key, default)
    except Exception:
        return default


def max_jobs():
    """lab/config.json {"max_jobs": n} (read every poll: no restart needed to change it)"""
    try:
        return int(json.load(open(os.path.join(CODE, "lab", "config.json"))).get("max_jobs", MAX_JOBS))
    except Exception:
        return MAX_JOBS


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
    if src.startswith("hf:"):
        dst = os.path.join(CKPT, "hf", name)
        if os.path.exists(dst):
            return dst
    if src == "ckpt":                        # one-off transfer branch
        sh("git", "fetch", "-q", "origin", "ckpt", cwd=CODE)
        with open(dst + ".part", "wb") as f:
            subprocess.run(["git", "show", f"origin/ckpt:{name}"], cwd=CODE, stdout=f, check=True)
        os.replace(dst + ".part", dst)
        return dst
    if src.startswith("hf:"):                # a Walker checkpoint on the Hub: "hf:<repo>" + name = path in the repo
        from huggingface_hub import hf_hub_download
        p_ = hf_hub_download(src[3:], name, local_dir=os.path.join(CKPT, "hf"))
        return p_
    raise FileNotFoundError(f"checkpoint {name} not in {CKPT}")


def adopted_end(name):
    """an adopted job (started before a self-reload) ended: no exit code is known -> success = a checkpoint exists and
    its log has no traceback; then keep its checkpoint if the job asks for it (adopted jobs were never kept before)"""
    log_p = os.path.join(RUNS, "runs", name, "train.log")
    tb = os.path.exists(log_p) and "Traceback" in open(log_p, errors="ignore").read()
    ok = os.path.exists(os.path.join(CKPT, name + ".pt")) and not tb
    write_status(name, state="done (adopted)" if ok else "failed (adopted)", ended=time.strftime("%Y-%m-%d %H:%M:%S"))
    try:
        job = json.load(open(os.path.join(CODE, "jobs", name + ".json")))
    except Exception:
        job = {}
    if ok and job.get("keep"):
        try:
            keep_ckpt(name)
        except Exception as e:
            log(f"keep {name} (adopted) failed: {e}")


def keep_ckpt(name):
    """push CKPT/<name>.pt (the run's best) to the ckpt branch, so later jobs and resets keep it"""
    src = os.path.join(CKPT, name + ".pt")
    if not os.path.exists(src):
        return
    repo = os.path.join(BASE, "ckptrepo")
    if not os.path.isdir(os.path.join(repo, ".git")):
        sh("git", "clone", "-q", "--single-branch", "-b", "ckpt", REPO_URL, repo)
        ident(repo)
    sh("git", "pull", "-q", "origin", "ckpt", cwd=repo)
    shutil.copy2(src, os.path.join(repo, name + ".pt"))
    sh("git", "add", name + ".pt", cwd=repo)
    sh("git", "commit", "-qm", f"ckpt: {name}", cwd=repo)
    sh("git", "push", "-q", "origin", "ckpt", cwd=repo)
    log(f"kept {name}.pt on the ckpt branch")


def status_path(name):
    return os.path.join(RUNS, "runs", name, "status.json")


def write_status(name, **kw):
    p = status_path(name)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    st = json.load(open(p)) if os.path.exists(p) else {}
    st.update(kw, updated=time.strftime("%Y-%m-%d %H:%M:%S"))
    json.dump(st, open(p, "w"), indent=1)


keep = {}


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
        args = [job.get("init_flag", "--init-full"), fetch_ckpt(job["init"], job.get("init_from", "ckpt"))] + args
    if job.get("walker"):                    # the 1B Walker: {"repo": ..., "path": ...} on the Hub
        args = ["--ckpt", fetch_ckpt(job["walker"]["path"], "hf:" + job["walker"]["repo"])] + args
    cmd = [sys.executable, "-u", "gnm.py"] + args + ["--save", os.path.join(CKPT, name + ".pt"),
                                                      "--log", os.path.join(out, "train.jsonl")]
    lf = open(os.path.join(out, "train.log"), "w")
    p = subprocess.Popen(cmd, cwd=d, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
    procs[name] = (p, lf)
    keep[name] = bool(job.get("keep"))
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
    import fcntl
    os.makedirs(BASE, exist_ok=True)
    lock = open(os.path.join(BASE, "poller.lock"), "w")
    try:                                     # one poller per machine (a re-run start command exits here)
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        log("another poller holds the lock: exiting")
        return
    setup()
    procs, last_push = {}, 0.0
    orphans = {}                             # jobs started by a previous poller that are still running
    for st_p in (os.path.join(RUNS, "runs", n_, "status.json") for n_ in os.listdir(os.path.join(RUNS, "runs"))
                 if os.path.isdir(os.path.join(RUNS, "runs", n_))):
        try:
            st = json.load(open(st_p))
            if st.get("state") == "running" and st.get("pid") and alive(st["pid"]):
                orphans[os.path.basename(os.path.dirname(st_p))] = st["pid"]
        except Exception:
            pass
    for n_ in os.listdir(os.path.join(RUNS, "runs")):
        try:
            if json.load(open(status_path(n_))).get("state") == "ended (adopted)":
                adopted_end(n_)
        except Exception:
            pass
    me = os.path.abspath(__file__)
    my_hash = file_hash(me)
    started = float(os.environ.get("LAB_STARTED", time.time()))
    os.environ["LAB_STARTED"] = str(started)       # survives os.execv reloads: the lifetime is per hand start
    idle_since = None
    log(f"poller up: code {CODE}, runs {RUNS}, ckpt {CKPT}; started {time.strftime('%H:%M', time.localtime(started))}")
    while True:
        age_h = (time.time() - started) / 3600
        expired = age_h >= float(cfg("max_hours", 8))
        try:
            sh("git", "fetch", "-q", "origin", "main", cwd=CODE)
            sh("git", "reset", "-q", "--hard", "origin/main", cwd=CODE)
            if file_hash(me) != my_hash:             # new poller code on main: restart in place (jobs keep running
                log("poller code changed on main: re-executing")   # in their own sessions; the new image adopts them)
                push_runs("runs: poller reload")
                lock.close()
                os.execv(sys.executable, [sys.executable, "-u", me])
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
                if job.get("cancel") and name in orphans:   # adopted after a reload: cancel by its process group
                    try:
                        os.killpg(os.getpgid(orphans[name]), signal.SIGTERM)
                    except Exception as e:
                        log(f"cancel {name} (adopted) failed: {e}")
                    write_status(name, state="cancelled", ended=time.strftime("%Y-%m-%d %H:%M:%S"))
                    del orphans[name]
                    changed = True
                    log(f"cancelled {name} (adopted)")
                    continue
                if name in procs or os.path.exists(status_path(name)) or job.get("cancel"):
                    continue
                dep = job.get("after")                # wait for another job to finish (e.g. the checkpoint it starts from)
                if dep and not (os.path.exists(status_path(dep)) and
                                json.load(open(status_path(dep))).get("state", "").startswith("done")):
                    continue
                if expired:                           # past the lifetime: nothing new starts
                    continue
                if len(procs) + len(orphans) >= max_jobs() or gpu_free_gb() < float(job.get("gpu_gb", 6)) + 2:
                    continue
                try:
                    start_job(job, procs)
                except Exception as e:
                    write_status(name, state="failed to start", error=str(e)[-500:])
                    log(f"{name} failed to start: {e}")
                changed = True
            for name, pid in list(orphans.items()):
                if not alive(pid):
                    adopted_end(name)
                    del orphans[name]
                    changed = True
            for name, (p, lf) in list(procs.items()):
                rc = p.poll()
                if rc is not None:
                    lf.close()
                    write_status(name, state="done" if rc == 0 else "failed", exit_code=rc,
                                 ended=time.strftime("%Y-%m-%d %H:%M:%S"))
                    if rc == 0 and keep.get(name):
                        try:
                            keep_ckpt(name)
                        except Exception as e:
                            log(f"keep {name} failed: {e}")
                    del procs[name]
                    changed = True
                    log(f"{name} ended ({rc})")
            json.dump({"alive": time.strftime("%Y-%m-%d %H:%M:%S"), "running": sorted(procs) + sorted(orphans),
                       "gpu_free_gb": round(gpu_free_gb(), 1), "code": sh("git", "rev-parse", "--short", "HEAD", cwd=CODE)},
                      open(os.path.join(RUNS, "runs", "_poller.json"), "w"), indent=1)
            if changed or time.time() - last_push > PUSH:
                push_runs("runs: " + ", ".join(sorted(procs)) if procs else "runs: idle")
                last_push = time.time()
            if expired and age_h >= float(cfg("max_hours", 8)) + float(cfg("grace_hours", 1.5)):
                for name, (p, lf) in list(procs.items()):     # lifetime + grace over: stop what still runs
                    os.killpg(p.pid, signal.SIGTERM)
                    write_status(name, state="stopped (poller lifetime)", ended=time.strftime("%Y-%m-%d %H:%M:%S"))
                for name, pid in orphans.items():
                    try:
                        os.killpg(os.getpgid(pid), signal.SIGTERM)
                    except Exception:
                        pass
                push_runs("runs: poller stopped (lifetime)")
                log("lifetime + grace over: stopped all jobs, exiting")
                return
            busy = bool(procs) or bool(orphans)
            idle_since = None if busy else (idle_since or time.time())
            if idle_since and time.time() - idle_since > 60 * float(cfg("idle_exit_min", 20)):
                push_runs("runs: poller stopped (idle)")
                log("idle: nothing running or startable, exiting")
                return
        except Exception as e:
            log(f"poll error: {e}")
        time.sleep(POLL)


if __name__ == "__main__":
    main()
