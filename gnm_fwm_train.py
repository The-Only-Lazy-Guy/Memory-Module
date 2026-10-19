"""Batched long-run trainer for the fast-weight memory on any-schema JSON (docs/GNM.md section 9.10).

The recipe of section 9.9 (addressing loss + aligned per-token value slots + overlapping renamed copies as hard
negatives, `--seg`), with large batches: every step trains E independent memories at once. All their documents go
through Walker in one padded batch, all units sit in one store with a per-memory mask (a question only sees its own
memory), and all questions are decoded in one forward. Episodes (random documents, questions, token positions) are
built by CPU worker processes ahead of the GPU.

Data: synthetic any-schema JSON (gmem_hub.random_doc), a fresh world every episode — infinite, nothing can be
memorized in the weights, and the asked unit is known by construction (addressing loss). Evaluated on FIXED held-out
sets: standard (eval wording), overlap (renamed copies), held-out key vocabulary, big store (more documents).

Logs JSON lines to runs/<run>.jsonl (train every --log-every steps, eval every --eval-every); checkpoint
runs/<run>.pt (+ runs/<run>_best.pt) for --resume.

    python gnm_fwm_train.py --ckpt ckpt/checkpoints/ckpt_best.pt --episodes 16 --steps 6000 --run fwm1b_long
"""
import argparse
import collections
import json
import math
import multiprocessing as mp
import os
import random
import time
import zlib

try:                                    # Windows: pyarrow must precede torch
    import pyarrow.dataset  # noqa: F401
except Exception:
    pass

import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------------ episode building (CPU workers)
_W = {}


def _winit(hf):
    from transformers import AutoTokenizer
    import gmem_hub as H
    import gnm_fwm_anyjson as AJ
    tok = AutoTokenizer.from_pretrained(hf)
    _W.update(tok=tok, H=H, AJ=AJ)
    vr, pool = random.Random(99), {}
    for _ in range(60):
        for lf in H.flatten(H.random_doc(vr, H.KEYS_TRAIN, depth=2)):
            if lf.image is None:
                pool.setdefault(AJ.kind_of(lf.value), []).append(lf.value)
    _W["pool"] = pool


def _ids(s):
    return list(_W["tok"](s, add_special_tokens=False)["input_ids"]) if s else []


def make_episode(task):
    """One memory: `docs` documents (some renamed overlapping copies) + `nq` lookup questions. Token-level only."""
    seed, docs, nq, overlap, keys, wording, max_ans, max_tok = task
    H, AJ, tok = _W["H"], _W["AJ"], _W["tok"]
    rng = random.Random(seed)
    vocab = H.KEYS_TRAIN if keys.startswith("train") else H.KEYS_EVAL
    if keys.startswith("train+mix"):
        # key diversity: a share of the vocabulary is fresh made-up words (syllables) for THIS memory, so key matching
        # cannot lean on ~150 memorized names; the held-out English key words stay unseen
        p_ = float(keys.split(":")[1])
        n_gen = int(round(len(vocab) * p_ / max(1e-6, 1 - p_)))
        held = set(H.KEYS_EVAL)
        gen = set()
        while len(gen) < n_gen:
            w = "".join(rng.choice(H._SYL) for _ in range(rng.randint(2, 3)))
            if w not in held and w.isalpha():
                gen.add(w)
        vocab = list(vocab) + sorted(gen)
    out_docs, meta, qs, first = [], [], [], None
    for di in range(docs):
        name = H._doc_name(rng)
        obj = None
        if overlap > 0 and di > 0 and first is not None and rng.random() < overlap:
            obj = AJ.perturb(first, rng, _W["pool"])
            pieces, leaves = AJ.serialize(obj, name)
            if not leaves or sum(len(_ids(pc)) for pc in pieces) > max_tok:
                obj = None
        if obj is None:
            while True:
                obj = H.random_doc(rng, vocab, depth=rng.randint(1, 3))
                pieces, leaves = AJ.serialize(obj, name)
                if leaves and sum(len(_ids(pc)) for pc in pieces) <= max_tok:
                    break
        if di == 0:
            first = obj
        vp = []
        AJ.serialize(obj, name, vp)
        ids, rt = [], []
        for pc in pieces:
            t = _ids(pc)
            rt.append((len(ids), len(ids) + len(t)))
            ids += t
        spans = [sorted({p for a in anc for p in range(*rt[a])} | set(range(*rt[pi]))) for _, _, pi, anc in leaves]
        ptr = [sorted({p for a in anc for p in range(*rt[a])}) for _, _, _, anc in leaves]
        vpos = []
        for (_, _, pi, _), pre in zip(leaves, vp):
            pt = _ids(pieces[pi])
            nv = len(pt) - len(_ids(pre)) if pre else len(pt)
            v1 = list(range(rt[pi][1] - nv, rt[pi][1]))
            if v1 and tok.decode([pt[len(pt) - nv]]).strip() == '"':       # slot t = answer token t
                v1 = v1[1:]
            vpos.append(v1)
        base = len(meta)
        meta += [(name, tuple(lf[0])) for lf in leaves]
        out_docs.append({"ids": ids, "spans": spans, "ptr": ptr, "vpos": vpos})
        by_path = {tuple(lf[0]): base + j for j, lf in enumerate(leaves)}
        look = [lf for lf in H.flatten(obj) if lf.image is None and tuple(lf.path) in by_path
                and lf.path and len(lf.value.split()) <= 12]
        for lf in rng.sample(look, min(nq // docs + 1, len(look))):
            key, where = H._subject(lf, name)
            q = (rng.choice(H._T_LOOK[wording]).format(key=key, where=where, doc=name) if where
                 else rng.choice(H._T_TOP[wording]).format(key=key, doc=name))
            qs.append({"q": _ids(" " + q), "y": _ids(" " + lf.value)[:max_ans], "gold": by_path[tuple(lf.path)],
                       "kind": AJ.kind_of(lf.value), "inlist": any(isinstance(p, int) for p in lf.path)})
    rng.shuffle(qs)
    return {"docs": out_docs, "meta": meta, "qs": qs[:nq]}


def parse_schedule(s):
    out = []
    for part in str(s).split(","):
        v, at = part.split(":") if ":" in part else (part, "0")
        out.append((int(at), int(v)))
    return sorted(out)


def at_step(sched, step):
    v = sched[0][1]
    for at, x in sched:
        if step >= at:
            v = x
    return v


def main():
    P = argparse.ArgumentParser()
    P.add_argument("--hf", default="gpt2")
    P.add_argument("--ckpt", default="", help="the 1B Walker checkpoint (frozen); overrides --hf")
    P.add_argument("--enc-layer", type=int, default=0)
    P.add_argument("--run", default="fwm_long")
    P.add_argument("--resume", action="store_true", help="continue runs/<run>.pt (model, optimizer, step)")
    P.add_argument("--init", default="", help="start the module from a saved anyjson / train checkpoint")
    P.add_argument("--steps", type=int, default=6000)
    P.add_argument("--episodes", type=int, default=16, help="memories per step (batch)")
    P.add_argument("--docs", default="3", help="documents per memory, or a schedule 'docs:step,...' e.g. 3:0,6:3000")
    P.add_argument("--nq", type=int, default=32, help="questions per memory")
    P.add_argument("--max-ans", type=int, default=8)
    P.add_argument("--max-doc-tokens", type=int, default=1000)
    P.add_argument("--dk", type=int, default=512)
    P.add_argument("--inject", type=int, default=6)
    P.add_argument("--lr", type=float, default=1e-3)
    P.add_argument("--warmup", type=int, default=200)
    P.add_argument("--addr-loss", type=float, default=1.0)
    P.add_argument("--overlap", type=float, default=0.5)
    P.add_argument("--key-mix", type=float, default=0.0,
                   help="share of the training key vocabulary that is fresh made-up words per memory (0 = off)")
    P.add_argument("--value-slots", type=int, default=8)
    P.add_argument("--slot-read", choices=("soft", "hard"), default="soft")
    P.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    P.add_argument("--prefetch", type=int, default=3, help="steps built ahead")
    P.add_argument("--enc-chunk", type=int, default=16, help="documents per Walker encode call")
    P.add_argument("--log-every", type=int, default=10)
    P.add_argument("--eval-every", type=int, default=250)
    P.add_argument("--eval-episodes", type=int, default=24)
    P.add_argument("--evals", default="standard,overlap,heldout_keys,big_store")
    P.add_argument("--big-docs", type=int, default=12)
    P.add_argument("--seed", type=int, default=0)
    A = P.parse_args()

    import gnm_fwm as W
    import gnm_v2 as V2
    import gmem_ltm as LT
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(A.seed)
    os.makedirs("runs", exist_ok=True)
    log_path, ck_path, best_path = f"runs/{A.run}.jsonl", f"runs/{A.run}.pt", f"runs/{A.run}_best.pt"
    model, raw_tok = W.load_walker(A.hf, A.ckpt, A.enc_layer, dev)
    tok = LT.Tok(raw_tok)
    enc = V2.WalkerStateEnc(model, tok)
    d = model.emb.weight.shape[1]
    mem = W.FastWeightMemory(d, A.dk, 1, tie=True, center=False, tok_emb=True, act="softmax", tau0=10.0,
                             seg=True, vslots=A.value_slots).to(dev)
    gate = nn.Parameter(torch.tensor(1.0, device=dev))
    Us = nn.Linear(A.dk, d).to(dev) if A.value_slots else None
    gs = nn.Parameter(torch.tensor(0.0, device=dev))
    params = list(mem.parameters()) + [gate] + (list(Us.parameters()) + [gs] if Us is not None else [])
    opt = torch.optim.AdamW(params, lr=A.lr, weight_decay=0.01)
    start, best = 0, -1.0

    def load_into(st, with_opt):
        mem.load_state_dict(st["mem"], strict=False)
        with torch.no_grad():
            gate.copy_(st["gate"])
            if Us is not None and st.get("us") is not None:
                Us.load_state_dict(st["us"])
                gs.copy_(st["gs"])
        if with_opt and st.get("opt") is not None:
            opt.load_state_dict(st["opt"])

    if A.resume and os.path.isfile(ck_path):
        st = torch.load(ck_path, map_location=dev)
        load_into(st, True)
        start, best = int(st["step"]), float(st.get("best", -1.0))
        print(f"RESUME {ck_path} at step {start} (best {best:.3f})", flush=True)
    elif A.init:
        load_into(torch.load(A.init, map_location=dev), False)
        print(f"INIT module from {A.init}", flush=True)
    inj = A.inject % len(model.blocks)
    ans_ids = tok.ids(" Answer:")
    a0 = len(ans_ids) - 1
    state = {"add": None, "slots": None}

    def pre(_m, args):
        if state["add"] is None:
            return None
        x = args[0].clone()
        rms = x[:, a0:].pow(2).mean(-1, keepdim=True).sqrt()
        a = state["add"][:, None]
        a = a / a.pow(2).mean(-1, keepdim=True).sqrt().clamp(min=1e-6)
        x[:, a0:a0 + 1] = x[:, a0:a0 + 1] + (torch.tanh(gate) * rms[:, :1] * a).to(x.dtype)
        sl = state["slots"]
        if sl is not None:
            sl = sl[:, :x.shape[1] - a0]
            sl = sl / sl.pow(2).mean(-1, keepdim=True).sqrt().clamp(min=1e-6)
            w_ = sl.shape[1]
            x[:, a0:a0 + w_] = x[:, a0:a0 + w_] + (torch.tanh(gs) * rms[:, :w_] * sl).to(x.dtype)
        return (x,) + tuple(args[1:])
    model.blocks[inj].register_forward_pre_hook(pre)

    def run_batch(eps, train):
        """eps: list of episodes. Returns loss and per-question results."""
        docs = [(e_i, dc) for e_i, ep in enumerate(eps) for dc in ep["docs"]]
        raws, vals, lgs, slots, ep_unit = [], [], [], [], []
        for c in range(0, len(docs), A.enc_chunk):
            chunk = docs[c:c + A.enc_chunk]
            L = max(len(dc["ids"]) for _, dc in chunk)
            ids = torch.tensor([dc["ids"] + [0] * (L - len(dc["ids"])) for _, dc in chunk], device=dev)
            with torch.no_grad():
                X = enc._states(ids).float()
                Ems = model.emb(ids).float()
            for j, (e_i, dc) in enumerate(chunk):
                n = len(dc["ids"])
                with torch.set_grad_enabled(train):
                    _, k, v, g = mem.write(X[j, :n], dc["spans"], Ems[j, :n], ptr=dc["ptr"], vpos=dc["vpos"])
                raws.append(mem.last_kraw); vals.append(v); lgs.append(torch.log(g + 1e-6))
                if A.value_slots:
                    slots.append(mem.last_vslots)
                ep_unit += [e_i] * k.shape[0]
        with torch.set_grad_enabled(train):
            raw = torch.cat(raws); Vv = torch.cat(vals); lg = torch.cat(lgs)
            epu = torch.tensor(ep_unit, device=dev)
            ne = len(eps)
            cnt = torch.zeros(ne, device=dev).index_add_(0, epu, torch.ones_like(epu, dtype=torch.float))
            kmean = torch.zeros(ne, raw.shape[1], device=dev).index_add_(0, epu, raw) / cnt[:, None]
            K = F.normalize(raw - kmean[epu], dim=-1)
            off = torch.tensor([0] + [len(ep["meta"]) for ep in eps], device=dev).cumsum(0)
            qs = [(e_i, q) for e_i, ep in enumerate(eps) for q in ep["qs"]]
            epq = torch.tensor([e_i for e_i, _ in qs], device=dev)
            gold = torch.tensor([int(off[e_i]) + q["gold"] for e_i, q in qs], device=dev)
            lgQ = lg[None].expand(len(qs), -1).masked_fill(epq[:, None] != epu[None], -1e4)
            mem.kmean = kmean[epq]
            n = max(len(q["q"]) for _, q in qs)
            qids = torch.tensor([q["q"] + [0] * (n - len(q["q"])) for _, q in qs], device=dev)
            mq = torch.tensor([[1] * len(q["q"]) + [0] * (n - len(q["q"])) for _, q in qs], device=dev,
                              dtype=torch.bool)
            with torch.no_grad():
                Hq, Eq = enc._states(qids).float(), model.emb(qids).float()
            rstar, _, _ = mem.read((K, Vv, lgQ), Hq, mq, Eq)
            qn_, tq_, _ = mem.last_q[0]
            att = torch.softmax(tq_ * qn_ @ K.t() + lgQ, -1)
            ys = [q["y"] for _, q in qs]
            L = max(len(y) for y in ys)
            inp = torch.tensor([ans_ids + y[:-1] + [0] * (L - len(y)) for y in ys], device=dev)
            tgt = torch.full((len(ys), len(ans_ids) + L - 1), -100, dtype=torch.long, device=dev)
            for b, y in enumerate(ys):
                tgt[b, a0:a0 + len(y)] = torch.tensor(y, device=dev)
            state["add"] = mem.Wo(rstar)
            if Us is not None:
                a_s = att
                if A.slot_read == "hard":
                    a_s = F.one_hot(att.argmax(-1), att.shape[-1]).to(att.dtype) + att - att.detach()
                state["slots"] = Us(torch.einsum("bn,nsd->bsd", a_s, torch.cat(slots)))
            logits, _ = model(inp, r_active=False)
            state["add"] = state["slots"] = None
            logits = logits.float()
            ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), ignore_index=-100)
            ar = torch.arange(len(qs), device=dev)
            p_gold = att[ar, gold]
            addr = -torch.log(p_gold + 1e-9).mean()
            loss = ce + A.addr_loss * addr
        pred = logits.argmax(-1)
        em = ((pred == tgt) | (tgt == -100)).all(-1).float()
        first = (pred[:, a0] == tgt[:, a0]).float()
        right = (att.argmax(-1) == gold).float()
        info = {"kind": [q["kind"] for _, q in qs], "inlist": [q["inlist"] for _, q in qs]}
        return loss, ce, addr, em, first, right, p_gold.detach(), info, K.shape[0] / ne

    # ---------------------------------------------------------------- data
    sched = parse_schedule(A.docs)
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(A.workers, initializer=_winit, initargs=(A.hf if not A.ckpt else "gpt2",))
    eval_specs = {"standard": dict(docs=3, overlap=0.0, keys="train", wording="eval"),
                  "overlap": dict(docs=3, overlap=1.0, keys="train", wording="eval"),
                  "heldout_keys": dict(docs=3, overlap=0.0, keys="eval", wording="eval"),
                  "big_store": dict(docs=A.big_docs, overlap=0.0, keys="train", wording="eval")}
    evals = {}
    for name in [x for x in A.evals.split(",") if x]:
        sp = eval_specs[name]
        tasks = [(10_000_000 + 1000 * i + zlib.crc32(name.encode()) % 997, sp["docs"], A.nq, sp["overlap"], sp["keys"],
                  sp["wording"], A.max_ans, A.max_doc_tokens) for i in range(A.eval_episodes)]
        evals[name] = pool.map(make_episode, tasks)
    print(f"EVAL SETS: " + ", ".join(f"{k} {sum(len(e['qs']) for e in v)} q" for k, v in evals.items()), flush=True)

    def tasks_for(step):
        nd = at_step(sched, step)
        keys = f"train+mix:{A.key_mix}" if A.key_mix > 0 else "train"
        return [(A.seed * 1_000_003 + step * 1009 + e, nd, A.nq, A.overlap, keys, "train", A.max_ans,
                 A.max_doc_tokens) for e in range(A.episodes)]
    pending = collections.OrderedDict()

    def fill(step):
        for s in range(step, min(A.steps, step + A.prefetch) + 1):
            if s not in pending:
                pending[s] = [pool.apply_async(make_episode, (t,)) for t in tasks_for(s)]

    logf = open(log_path, "a", encoding="utf-8")

    def log(rec):
        logf.write(json.dumps(rec) + "\n")
        logf.flush()

    @torch.no_grad()
    def evaluate(step):
        mem.eval()
        res = {}
        for name, eps in evals.items():
            em_, fi_, ri_, pg_, kinds = [], [], [], [], collections.defaultdict(list)
            for j in range(0, len(eps), max(1, A.episodes)):
                _, _, _, em, fi, ri, pg, info, _ = run_batch(eps[j:j + max(1, A.episodes)], False)
                em_ += em.tolist(); fi_ += fi.tolist(); ri_ += ri.tolist(); pg_ += pg.tolist()
                for k_, e_ in zip(info["kind"], em.tolist()):
                    kinds[k_].append(e_)
            n = max(1, len(em_))
            res[name] = {"exact": sum(em_) / n, "first": sum(fi_) / n, "read_right": sum(ri_) / n,
                         "p_gold": sum(pg_) / n, "n": n, "kinds": {k: sum(v) / len(v) for k, v in kinds.items()}}
            log({"type": "eval", "step": step, "set": name, **res[name], "time": time.time()})
            print(f"EVAL step {step} {name}: exact {res[name]['exact']:.3f} | first {res[name]['first']:.3f} | "
                  f"read right {res[name]['read_right']:.3f} | n {n}", flush=True)
        mem.train()
        return res

    def save(path, step, best_):
        torch.save({"mem": mem.state_dict(), "gate": gate.detach(), "us": Us.state_dict() if Us is not None else None,
                    "gs": gs.detach(), "opt": opt.state_dict(), "step": step, "best": best_, "args": vars(A)}, path)

    print(f"TRAIN {A.run}: {A.episodes} memories x {A.nq} questions per step, docs {A.docs}, steps {start}->{A.steps}, "
          f"workers {A.workers}, model {'1B ' + A.ckpt if A.ckpt else A.hf}", flush=True)
    log({"type": "config", "step": start, "args": vars(A), "time": time.time()})
    if start == 0:
        evaluate(0)
    mem.train()
    t_last, acc = time.time(), collections.defaultdict(float)
    for step in range(start + 1, A.steps + 1):
        fill(step)
        t_wait = time.time()
        eps = [r.get() for r in pending.pop(step)]
        acc["wait"] += time.time() - t_wait
        lr = A.lr * min(1.0, step / max(1, A.warmup)) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / A.steps)))
        for gp in opt.param_groups:
            gp["lr"] = lr
        loss, ce, addr, em, fi, ri, pg, _, upm = run_batch(eps, True)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        for k_, v_ in (("loss", loss), ("ce", ce), ("addr", addr)):
            acc[k_] += float(v_.detach())
        acc["exact"] += float(em.mean()); acc["first"] += float(fi.mean()); acc["right"] += float(ri.mean())
        acc["n"] += 1
        if step % A.log_every == 0:
            dt = time.time() - t_last
            n = acc["n"]
            rec = {"type": "train", "step": step, "lr": lr, "loss": acc["loss"] / n, "ce": acc["ce"] / n,
                   "addr": acc["addr"] / n, "exact": acc["exact"] / n, "first": acc["first"] / n,
                   "read_right": acc["right"] / n, "tau": float(mem.tau.exp()), "sec_per_step": dt / n,
                   "data_wait": acc["wait"] / n, "units_per_memory": upm, "docs": at_step(sched, step),
                   "gpu_gib": torch.cuda.max_memory_allocated() / 2 ** 30 if dev.type == "cuda" else 0.0,
                   "time": time.time()}
            log(rec)
            print(f"step {step} loss {rec['loss']:.3f} exact {rec['exact']:.2f} first {rec['first']:.2f} read "
                  f"{rec['read_right']:.2f} tau {rec['tau']:.1f} | {rec['sec_per_step']:.2f}s/step (data wait "
                  f"{rec['data_wait']:.2f}s) docs {rec['docs']} GPU {rec['gpu_gib']:.1f} GiB", flush=True)
            t_last, acc = time.time(), collections.defaultdict(float)
        if step % A.eval_every == 0 or step == A.steps:
            res = evaluate(step)
            score = res.get("standard", next(iter(res.values())))["exact"] if res else 0.0
            if score > best:
                best = score
                save(best_path, step, best)
            save(ck_path, step, best)
    pool.close()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
