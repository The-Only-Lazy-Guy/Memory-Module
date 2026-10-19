import marimo

__generated_with = "0.17.7"
app = marimo.App(width="full")


@app.cell
def _():
    import json
    import os
    import signal
    import subprocess
    import sys
    from pathlib import Path

    import marimo as mo

    REPO = "https://github.com/The-Only-Lazy-Guy/Memory-Module"
    WORK = Path("/marimo/mm") if Path("/marimo").exists() else Path.cwd() / "mm"
    CKPT = WORK / "ckpt" / "checkpoints" / "ckpt_best.pt"
    return CKPT, Path, REPO, WORK, json, mo, os, signal, subprocess, sys


@app.cell
def _(mo):
    mo.Html(
        "<style>:root{--acc:#22d3ee;--bg:#070c16;--ink:#e5edf7}"
        "body{background:var(--bg)!important;color:var(--ink)}"
        ".hero{background:linear-gradient(135deg,#0b1a2b,#122a4a 45%,#2a1e4d);"
        "border:1px solid rgba(34,211,238,.28);border-radius:14px;padding:18px 22px;margin:2px 0 8px}"
        ".hero h1{margin:0;font-size:22px;font-weight:800;color:#a5f3fc}"
        ".hero p{margin:8px 0 0;opacity:.9;font-size:13px}"
        "h2{font-size:1.15rem;padding:6px 10px;border-left:3px solid var(--acc);"
        "background:linear-gradient(90deg,rgba(34,211,238,.1),transparent 70%);border-radius:8px}"
        "code{background:rgba(34,211,238,.1);color:#a5f3fc;border-radius:4px;padding:1px 5px}"
        "</style><div class='hero'><h1>FWM memory · Walker 1B — long run</h1>"
        "<p>Trains the fast-weight memory skill (write any JSON → look it up → Walker says the value) with the "
        "recipe of GNM.md §9.9: addressing loss, aligned per-token value slots, overlapping renamed copies. "
        "Walker is frozen. Each step trains many independent memories at once (large batch). "
        "Data: a fresh random-schema JSON world for every memory (nothing can be memorized in the weights); "
        "every eval set is never-seen documents written once and asked at once (zero training on them). "
        "<b>1</b> Setup → <b>2</b> Configure → <b>3</b> Start → <b>4</b> watch the dashboard.</p></div>"
    )
    return


@app.cell
def _(mo):
    setup_button = mo.ui.run_button(label="① Setup: clone/update code + download the 1B checkpoint")
    setup_button
    return (setup_button,)


@app.cell
def _(CKPT, REPO, WORK, mo, setup_button, subprocess, sys):
    mo.stop(not setup_button.value, mo.md(
        f"_Code goes to `{WORK}`, the checkpoint to `{CKPT}`. Press Setup after every box reset (~11 h)._"))
    _out = []

    def _sh(cmd, cwd=None):
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
        _out.append(f"$ {' '.join(cmd)}\n{(r.stdout + r.stderr).strip()[-1500:]}")
        return r.returncode

    if (WORK / ".git").is_dir():
        _sh(["git", "pull", "--ff-only"], cwd=WORK)
    else:
        WORK.parent.mkdir(parents=True, exist_ok=True)
        _sh(["git", "clone", "--depth", "1", REPO, str(WORK)])
    (WORK / "runs").mkdir(exist_ok=True)
    if not CKPT.is_file():
        _sh([sys.executable, "-c",
             "from huggingface_hub import hf_hub_download; "
             f"print(hf_hub_download('Papatpong/Walker', 'checkpoints/ckpt_best.pt', local_dir='{WORK / 'ckpt'}'))"])
    _sh(["git", "log", "--oneline", "-1"], cwd=WORK)
    _sh(["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv"])
    _ok = (WORK / "gnm_fwm_train.py").is_file() and CKPT.is_file()
    mo.vstack([
        mo.callout("Setup OK — code and checkpoint in place." if _ok else "Setup incomplete — see the log.",
                   kind="success" if _ok else "danger"),
        mo.md("```\n" + "\n\n".join(_out) + "\n```"),
    ])
    return


@app.cell
def _(mo):
    C = {}
    C["model"] = mo.ui.dropdown(options=["Walker 1B (checkpoint)", "gpt2 (fast check)"],
                                value="Walker 1B (checkpoint)", label="model")
    C["run"] = mo.ui.text(value="fwm1b_long", label="run name (runs/<run>.jsonl / .pt)")
    C["resume"] = mo.ui.checkbox(value=False, label="resume runs/<run>.pt (after a stop / box reset)")
    C["steps"] = mo.ui.number(start=100, stop=200000, step=100, value=4000, label="steps")
    C["episodes"] = mo.ui.number(start=1, stop=256, step=1, value=32,
                                 label="memories per step (batch; 32 ≈ 1000 questions)")
    C["nq"] = mo.ui.number(start=4, stop=64, step=4, value=32, label="questions per memory")
    C["docs"] = mo.ui.text(value="3", label="documents per memory (or schedule docs:step, e.g. 3:0,6:2000)")
    C["lr"] = mo.ui.number(start=0.00005, stop=0.01, step=0.00005, value=0.001, label="lr (peak)")
    C["warmup"] = mo.ui.number(start=0, stop=5000, step=50, value=200, label="warmup steps")
    C["addr"] = mo.ui.number(start=0.0, stop=5.0, step=0.1, value=1.0, label="addressing loss weight")
    C["overlap"] = mo.ui.slider(start=0.0, stop=1.0, step=0.05, value=0.5, label="overlap (renamed copies)")
    C["slots"] = mo.ui.number(start=0, stop=16, step=1, value=8, label="value slots (0 = off)")
    C["slot_read"] = mo.ui.dropdown(options=["soft", "hard"], value="soft", label="slot read")
    C["eval_every"] = mo.ui.number(start=25, stop=5000, step=25, value=250, label="eval every (steps)")
    C["eval_eps"] = mo.ui.number(start=4, stop=256, step=4, value=32, label="eval memories per set")
    C["big_docs"] = mo.ui.number(start=3, stop=48, step=1, value=12, label="big-store eval: documents")
    C["workers"] = mo.ui.number(start=1, stop=32, step=1, value=8, label="CPU data workers")
    C["seed"] = mo.ui.number(start=0, stop=100000, step=1, value=0, label="seed")
    mo.vstack([
        mo.md("## ② Configure"),
        mo.hstack([C["model"], C["run"], C["resume"]], justify="start", gap=2),
        mo.hstack([C["steps"], C["episodes"], C["nq"], C["docs"]], justify="start", gap=2),
        mo.hstack([C["lr"], C["warmup"], C["addr"], C["overlap"]], justify="start", gap=2),
        mo.hstack([C["slots"], C["slot_read"], C["eval_every"], C["eval_eps"], C["big_docs"]],
                  justify="start", gap=2),
        mo.hstack([C["workers"], C["seed"]], justify="start", gap=2),
        mo.md("_Large batch: if `data wait` in the log is > ~10% of `s/step`, raise the workers; if the GPU "
              "has room (see GPU GiB), raise memories per step. Same total memories ≈ same result, fewer steps "
              "= faster._"),
    ])
    return (C,)


@app.cell
def _(C, CKPT, mo):
    argv = ["gnm_fwm_train.py"]
    if C["model"].value.startswith("Walker"):
        argv += ["--ckpt", str(CKPT)]
    else:
        argv += ["--hf", "gpt2"]
    argv += ["--run", C["run"].value.strip() or "fwm1b_long", "--steps", str(int(C["steps"].value)),
             "--episodes", str(int(C["episodes"].value)), "--nq", str(int(C["nq"].value)),
             "--docs", C["docs"].value.strip() or "3", "--lr", str(C["lr"].value),
             "--warmup", str(int(C["warmup"].value)), "--addr-loss", str(C["addr"].value),
             "--overlap", str(C["overlap"].value), "--value-slots", str(int(C["slots"].value)),
             "--slot-read", C["slot_read"].value, "--eval-every", str(int(C["eval_every"].value)),
             "--eval-episodes", str(int(C["eval_eps"].value)), "--big-docs", str(int(C["big_docs"].value)),
             "--workers", str(int(C["workers"].value)), "--seed", str(int(C["seed"].value))]
    if C["resume"].value:
        argv += ["--resume"]
    mo.vstack([mo.md("## ③ Command"), mo.md("```\npython -u " + " ".join(argv) + "\n```")])
    return (argv,)


@app.cell
def _(mo):
    start_button = mo.ui.run_button(label="▶ Start")
    stop_button = mo.ui.run_button(label="⏹ Stop")
    refresh = mo.ui.refresh(options=["5s", "15s", "30s", "60s"], default_interval="15s", label="live refresh")
    mo.hstack([start_button, stop_button, refresh], justify="start", gap=1)
    return refresh, start_button, stop_button


@app.cell
def _(C, WORK, argv, os, signal, start_button, stop_button, subprocess, sys):
    # The run is a detached process (own session) that survives notebook re-runs; its PID lives in
    # runs/<run>.pid so Stop and the dashboard find it again after a kernel restart.
    _run = C["run"].value.strip() or "fwm1b_long"
    _pidf = WORK / "runs" / f"{_run}.pid"

    def _alive(pid):
        try:
            os.kill(pid, 0)
            return True
        except Exception:
            return False

    _msg = ""
    _pid = int(_pidf.read_text()) if _pidf.is_file() else None
    if stop_button.value and _pid and _alive(_pid):
        os.killpg(os.getpgid(_pid), signal.SIGTERM)
        _msg = f"stop sent to {_pid}"
    if start_button.value:
        if _pid and _alive(_pid):
            _msg = f"already running (pid {_pid}) — Stop it first"
        else:
            _log = open(WORK / "runs" / f"{_run}.out", "a", encoding="utf-8", buffering=1)
            _env = dict(os.environ, HF_HUB_DISABLE_XET="1", PYTHONUNBUFFERED="1")
            _p = subprocess.Popen([sys.executable, "-u", *argv], cwd=WORK, stdout=_log,
                                  stderr=subprocess.STDOUT, start_new_session=True, env=_env)
            _pidf.write_text(str(_p.pid))
            _msg = f"started pid {_p.pid}"
    print(_msg or "ready")
    return


@app.cell
def _(C, WORK, json, mo, os, refresh):
    refresh.value
    _run = C["run"].value.strip() or "fwm1b_long"
    _jl, _out, _pidf = (WORK / "runs" / f"{_run}.jsonl", WORK / "runs" / f"{_run}.out",
                        WORK / "runs" / f"{_run}.pid")
    recs = []
    if _jl.is_file():
        for _ln in _jl.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                recs.append(json.loads(_ln))
            except Exception:
                pass
    # a resumed run re-logs steps after its checkpoint: keep the LAST record per (type, set, step)
    _dd = {}
    for _r in recs:
        _dd[(_r.get("type"), _r.get("set"), _r.get("step"))] = _r
    recs = sorted(_dd.values(), key=lambda r: (r.get("step", 0), str(r.get("set"))))
    _alive = False
    if _pidf.is_file():
        try:
            os.kill(int(_pidf.read_text()), 0)
            _alive = True
        except Exception:
            pass
    tail = (_out.read_text(encoding="utf-8", errors="replace").splitlines()[-40:] if _out.is_file() else [])
    status = ("running" if _alive else "not running")
    return recs, status, tail


@app.cell
def _(mo, recs, status, tail):
    train = [r for r in recs if r.get("type") == "train"]
    evals = [r for r in recs if r.get("type") == "eval"]
    sets = sorted({r["set"] for r in evals})
    _last = train[-1] if train else {}
    _hdr = (f"Status: **{status}** · step {_last.get('step', 0)} · {_last.get('sec_per_step', 0):.2f} s/step "
            f"(data wait {_last.get('data_wait', 0):.2f}s) · GPU {_last.get('gpu_gib', 0):.1f} GiB · "
            f"docs/memory {_last.get('docs', '-')}")
    if evals:
        _lines = []
        for _s in sets:
            _e = [r for r in evals if r["set"] == _s]
            _lines.append(f"| {_s} | " + " | ".join(f"{r['step']}: **{r['exact']:.3f}**" for r in _e[-6:]) + " |")
        _tbl = "| eval set | latest evals (step: exact) |\n|---|---|\n" + "\n".join(_lines)
    else:
        _tbl = "_no eval yet_"

    _fig = None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        _fig, _ax = plt.subplots(2, 2, figsize=(14, 8))
        for _s in sets:
            _e = [r for r in evals if r["set"] == _s]
            _x = [r["step"] for r in _e]
            _ax[0][0].plot(_x, [r["exact"] for r in _e], marker="o", label=_s)
            _xs = [max(1, x) for x in _x]
            _ax[0][1].plot(_xs, [max(1e-3, 1 - r["exact"]) for r in _e], marker="o", label=_s)
        _ax[0][0].set_title("exact match on never-seen memories (linear)")
        _ax[0][0].set_xlabel("step"); _ax[0][0].set_ylim(0, 1); _ax[0][0].legend(); _ax[0][0].grid(alpha=.3)
        _ax[0][1].set_xscale("log"); _ax[0][1].set_yscale("log")
        _ax[0][1].set_title("error (1 − exact), log-log: a straight falling line = power law")
        _ax[0][1].set_xlabel("step"); _ax[0][1].legend(); _ax[0][1].grid(alpha=.3, which="both")
        # slope per eval interval: rising = accelerating, falling = decelerating
        for _s in sets:
            _e = [r for r in evals if r["set"] == _s]
            if len(_e) >= 2:
                _mid = [(a["step"] + b["step"]) / 2 for a, b in zip(_e, _e[1:])]
                _sl = [1000 * (b["exact"] - a["exact"]) / max(1, b["step"] - a["step"]) for a, b in zip(_e, _e[1:])]
                _ax[1][0].plot(_mid, _sl, marker="o", label=_s)
        _ax[1][0].axhline(0, color="gray", lw=.8)
        _ax[1][0].set_title("gain per 1000 steps between evals (rising = accelerating)")
        _ax[1][0].set_xlabel("step"); _ax[1][0].legend(); _ax[1][0].grid(alpha=.3)
        if train:
            _x = [r["step"] for r in train]
            _ax[1][1].plot(_x, [r["exact"] for r in train], label="train exact", alpha=.8)
            _ax[1][1].plot(_x, [r["read_right"] for r in train], label="train read right", alpha=.8)
            _ax[1][1].plot(_x, [min(1.0, r["addr"] / 5) for r in train], label="addr loss / 5", alpha=.6)
        _ax[1][1].set_title("training batches"); _ax[1][1].set_xlabel("step"); _ax[1][1].set_ylim(0, 1)
        _ax[1][1].legend(); _ax[1][1].grid(alpha=.3)
        _fig.tight_layout()
    except Exception as _err:  # plotting is optional; the tables carry the numbers
        _fig = mo.md(f"_plot unavailable: {_err}_")

    _kinds = ""
    _std = [r for r in evals if r["set"] == "standard"]
    if _std:
        _k = _std[-1]["kinds"]
        _kinds = ("**Latest standard eval by value kind:** " +
                  " · ".join(f"{k} {v:.2f}" for k, v in sorted(_k.items())) +
                  f" — read right {_std[-1]['read_right']:.3f}, first token {_std[-1]['first']:.3f}")
    mo.vstack([
        mo.md("## ④ Dashboard"),
        mo.callout(_hdr, kind="success" if status == "running" else "neutral"),
        mo.md(_tbl),
        mo.as_html(_fig) if _fig is not None else mo.md(""),
        mo.md(_kinds),
        mo.accordion({"📜 log (last 40 lines)": mo.md("```\n" + "\n".join(tail) + "\n```")}),
        mo.md("_Checkpoints: `runs/<run>.pt` (latest, for resume) and `runs/<run>_best.pt` (best standard "
              "eval). Download them from the Files panel before the box resets._"),
    ])
    return


if __name__ == "__main__":
    app.run()
