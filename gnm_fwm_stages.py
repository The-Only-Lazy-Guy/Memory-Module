"""Staged memorization test for the fast-weight memory on any-schema JSON (docs/GNM.md section 9.9).

Why does the memory learn so slowly? Instead of thousands of steps on fresh random documents, a tiny world:
  S1  world A (a few fixed documents) is written; train on A's own questions until they are answered.
      Steps-to-solve measures the optimization. A counterfactual copy A' (same documents, same questions, other
      values) tells READING (A' right) from memorizing answers in the weights (A' wrong).
  S2  a DISJOINT subworld B (new documents) is written next to A: A again (interference), B zero-shot (a skill or
      memorized A?), then train on A+B.
  S3  an OVERLAPPING subworld C (A's documents renamed: same keys, record labels and structure, other values) is
      written next to A+B: only the document name tells A from C.
Each evaluation reports exact / first token per group, where the read's attention lands, per-part gradient norms
(key side / value side / readout), time per phase and GPU / CPU memory.

    python gnm_fwm_stages.py --hf gpt2 --docs-a 2
    python gnm_fwm_stages.py --hf gpt2 --load runs/fwm_aj10.pt        # the trained skill
"""
import argparse
import json
import math
import os
import random
import time

try:                                    # Windows: pyarrow must precede torch
    import pyarrow.dataset  # noqa: F401
except Exception:
    pass

import torch
import torch.nn as nn
import torch.nn.functional as F

import gnm_fwm as W
import gmem_hub as H
from gnm_fwm_anyjson import serialize, kind_of, perturb


def main():
    P = argparse.ArgumentParser()
    P.add_argument("--hf", default="gpt2")
    P.add_argument("--ckpt", default="")
    P.add_argument("--enc-layer", type=int, default=0)
    P.add_argument("--load", default="", help="start from a trained module (else from scratch)")
    P.add_argument("--docs-a", type=int, default=2)
    P.add_argument("--docs-b", type=int, default=2)
    P.add_argument("--max-q", type=int, default=24, help="questions per document (fixed set)")
    P.add_argument("--nq", type=int, default=32, help="questions per training step")
    P.add_argument("--max-ans", type=int, default=8)
    P.add_argument("--dk", type=int, default=512)
    P.add_argument("--inject", type=int, default=6)
    P.add_argument("--lr", type=float, default=1e-3)
    P.add_argument("--tau0", type=float, default=10.0)
    P.add_argument("--stage-steps", type=int, default=600, help="max training steps per stage")
    P.add_argument("--solve", type=float, default=0.95, help="stage solved at this exact match (trained groups)")
    P.add_argument("--eval-every", type=int, default=25)
    P.add_argument("--freeze", default="", help="comma list of module-name prefixes NOT trained (ablation)")
    P.add_argument("--addr-loss", type=float, default=0.0,
                   help="weight of a cloze addressing loss: the read's attention must land on the asked leaf's unit")
    P.add_argument("--value-slots", type=int, default=0)
    P.add_argument("--stages", default="1,2,3")
    P.add_argument("--seed", type=int, default=0)
    P.add_argument("--out", default="")
    A = P.parse_args()
    import gnm_v2 as V2
    import gmem_ltm as LT
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(A.seed)
    model, raw_tok = W.load_walker(A.hf, A.ckpt, A.enc_layer, dev)
    tok = LT.Tok(raw_tok)
    enc = V2.WalkerStateEnc(model, tok)
    d = model.emb.weight.shape[1]
    mem = W.FastWeightMemory(d, A.dk, 1, tie=True, center=False, tok_emb=True, act="softmax", tau0=A.tau0,
                             seg=True, vslots=A.value_slots).to(dev)
    gate = nn.Parameter(torch.tensor(1.0, device=dev))
    Us = nn.Linear(A.dk, d).to(dev) if A.value_slots else None
    gs = nn.Parameter(torch.tensor(0.0, device=dev))
    if A.load:
        st = torch.load(A.load, map_location=dev)
        mem.load_state_dict(st["mem"], strict=False)
        with torch.no_grad():
            gate.copy_(st["gate"])
            if Us is not None and st.get("us") is not None:
                Us.load_state_dict(st["us"])
                gs.copy_(st["gs"])
        print(f"loaded {A.load} (step {st.get('step')})", flush=True)
    frozen = [x for x in A.freeze.split(",") if x]
    for n_, p_ in mem.named_parameters():
        if any(n_.startswith(f) for f in frozen):
            p_.requires_grad_(False)
    params = ([p_ for p_ in mem.parameters() if p_.requires_grad] + [gate]
              + (list(Us.parameters()) + [gs] if Us is not None else []))
    opt = torch.optim.AdamW(params, lr=A.lr, weight_decay=0.01)
    inj = A.inject % len(model.blocks)
    ans_ids = tok.ids(" Answer:")
    a0 = len(ans_ids) - 1
    state = {"add": None}

    def pre(_m, args):
        if state["add"] is None:
            return None
        x = args[0]
        rms = x[:, a0:].pow(2).mean(-1, keepdim=True).sqrt()
        a = state["add"][:, None]
        a = a / a.pow(2).mean(-1, keepdim=True).sqrt().clamp(min=1e-6)
        x = x.clone()
        x[:, a0:a0 + 1] = x[:, a0:a0 + 1] + (torch.tanh(gate) * rms[:, :1] * a).to(x.dtype)
        sl = state.get("slots")
        if sl is not None:
            sl = sl[:, :x.shape[1] - a0]
            sl = sl / sl.pow(2).mean(-1, keepdim=True).sqrt().clamp(min=1e-6)
            w_ = sl.shape[1]
            x[:, a0:a0 + w_] = x[:, a0:a0 + w_] + (torch.tanh(gs) * rms[:, :w_] * sl).to(x.dtype)
        return (x,) + tuple(args[1:])
    model.blocks[inj].register_forward_pre_hook(pre)

    # ------------------------------------------------------------------ the worlds
    rng = random.Random(A.seed)
    used_names = set()

    def new_docs(n):
        out = []
        while len(out) < n:
            name = H._doc_name(rng)
            if name in used_names:
                continue
            obj = H.random_doc(rng, H.KEYS_TRAIN, depth=rng.randint(1, 3))
            pieces, leaves = serialize(obj, name)
            if leaves and sum(len(tok.ids(pc)) for pc in pieces) <= 1000:
                used_names.add(name)
                out.append((name, obj))
        return out

    pool = {}                                           # values by kind, for counterfactual / overlapping copies
    for _ in range(40):
        for lf in H.flatten(H.random_doc(rng, H.KEYS_TRAIN, depth=2)):
            if lf.image is None:
                pool.setdefault(kind_of(lf.value), []).append(lf.value)

    def questions(name, obj, group):
        """The FIXED question set of one document (deterministic wording: the same questions every time)."""
        qr = random.Random(name)          # A and A' (same names) get the SAME questions
        _, leaves = serialize(obj, name)
        by_path = {tuple(lf[0]): lf for lf in leaves}
        look = [lf for lf in H.flatten(obj) if lf.image is None and tuple(lf.path) in by_path
                and lf.path and len(lf.value.split()) <= 12]
        qs = []
        for lf in qr.sample(look, min(A.max_q, len(look))):
            key, where = H._subject(lf, name)
            q = (qr.choice(H._T_LOOK["train"]).format(key=key, where=where, doc=name) if where
                 else qr.choice(H._T_TOP["train"]).format(key=key, doc=name))
            qs.append({"q": q, "a": lf.value, "doc": name, "path": tuple(lf.path), "group": group,
                       "inlist": any(isinstance(p_, int) for p_ in lf.path)})
        return qs

    timers = {}

    def tick(name, t0):
        if dev.type == "cuda":
            torch.cuda.synchronize()
        timers[name] = timers.get(name, 0.0) + time.time() - t0
        return time.time()

    def write(docs, grad):
        """Encode + write every document into ONE memory. Returns M and the unit index of each (doc, path)."""
        units, index, vsl = [], {}, []
        for name, obj in docs:
            t0 = time.time()
            pieces, leaves = serialize(obj, name)
            with torch.no_grad():
                X, rt = enc.doc(pieces)
                E = model.emb(torch.tensor([t for pc in pieces for t in tok.ids(pc)], device=dev)).float()
            t0 = tick("encode (Walker, no grad)", t0)
            spans = [sorted({p for a in anc for p in range(*rt[a])} | set(range(*rt[pi])))
                     for _, _, pi, anc in leaves]
            ptr = [sorted({p for a in anc for p in range(*rt[a])}) for _, _, _, anc in leaves]
            vpos = None
            if A.value_slots:
                vp_ = []
                serialize(obj, name, vp_)
                vpos = []
                for (_, _, pi, _), pre_ in zip(leaves, vp_):
                    nv = len(tok.ids(pieces[pi])) - len(tok.ids(pre_)) if pre_ else len(tok.ids(pieces[pi]))
                    vp1 = list(range(rt[pi][1] - nv, rt[pi][1]))
                    # slot t must hold value token t itself (the token stream carries its identity; a causal state
                    # cannot know the NEXT token): drop a string's opening quote token, else every slot is one behind
                    # (1B: codes / dates / emails exact .00 with first token .76-.82, the tail always wrong)
                    if vp1 and raw_tok.decode(tok.ids(pieces[pi])[len(tok.ids(pieces[pi])) - nv]).strip() == '"':
                        vp1 = vp1[1:]
                    vpos.append(vp1)
            with torch.set_grad_enabled(grad):
                _, k, v, g = mem.write(X.to(dev), spans, E, ptr=ptr, vpos=vpos)
                units.append((mem.last_kraw, v, torch.log(g + 1e-6)))
                if A.value_slots:
                    vsl.append(mem.last_vslots)
            base = sum(u[0].shape[0] for u in units[:-1])
            for j, lf in enumerate(leaves):
                index[(name, tuple(lf[0]))] = base + j
            tick("write (module)", t0)
        raw = torch.cat([u[0] for u in units])
        mem.kmean = raw.mean(0, keepdim=True)
        K = F.normalize(raw - mem.kmean, dim=-1)
        write.vslots = torch.cat(vsl) if vsl else None
        return (K, torch.cat([u[1] for u in units]), torch.cat([u[2] for u in units])), index

    def answer(M, index, qs, grad):
        """Read for each question, decode the value with Walker. Returns loss, exact, first, read category."""
        t0 = time.time()
        seqs = [tok.ids(" " + q["q"]) for q in qs]
        n = max(len(s) for s in seqs)
        ids = torch.tensor([s + [0] * (n - len(s)) for s in seqs], device=dev)
        mq = torch.tensor([[1] * len(s) + [0] * (n - len(s)) for s in seqs], device=dev, dtype=torch.bool)
        with torch.no_grad():
            Hq, Eq = enc._states(ids).float(), model.emb(ids).float()
        t0 = tick("encode question (Walker, no grad)", t0)
        with torch.set_grad_enabled(grad):
            rstar, _, _ = mem.read(M, Hq, mq, Eq)
            qn_, tq_, _ = mem.last_q[0]
            att = torch.softmax(tq_ * qn_ @ M[0].t() + M[2], -1)
            t0 = tick("read (module)", t0)
            ys = [tok.ids(" " + q["a"])[:A.max_ans] for q in qs]
            L = max(len(y) for y in ys)
            inp = torch.tensor([ans_ids + y[:-1] + [0] * (L - len(y)) for y in ys], device=dev)
            tgt = torch.full((len(ys), len(ans_ids) + L - 1), -100, dtype=torch.long, device=dev)
            for b, y in enumerate(ys):
                tgt[b, a0:a0 + len(y)] = torch.tensor(y, device=dev)
            state["add"] = mem.Wo(rstar)
            if Us is not None:
                state["slots"] = Us(torch.einsum("bn,nsd->bsd", att, write.vslots))
            logits, _ = model(inp, r_active=False)
            state["add"] = None
            state["slots"] = None
            logits = logits.float()
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), ignore_index=-100)
            gold = torch.tensor([index[(q["doc"], q["path"])] for q in qs], device=dev)
            addr = -torch.log(att[torch.arange(len(qs), device=dev), gold] + 1e-9).mean()
            if A.addr_loss > 0:
                loss = loss + A.addr_loss * addr
        tick("decode (Walker fwd)", t0)
        ok = (logits.argmax(-1) == tgt) | (tgt == -100)
        em = ok.all(-1).float().tolist()
        first = (logits[:, a0].argmax(-1) == tgt[:, a0]).float().tolist()
        inv = {u: k for k, u in index.items()}
        top = att.argmax(-1).tolist()
        p_gold = att[torch.arange(len(qs), device=dev), gold].tolist()
        cats = []
        for q, t_ in zip(qs, top):
            dt, pt = inv[t_]
            pg = q["path"]
            if (dt, pt) == (q["doc"], pg):
                cats.append("right leaf")
            elif dt != q["doc"]:
                same_struct = pt == pg
                cats.append("other doc, SAME path" if same_struct else "other doc")
            elif [x for x in pg if not isinstance(x, int)] == [x for x in pt if not isinstance(x, int)]:
                cats.append("same field, other record")
            elif pg[:-1] == pt[:-1]:
                cats.append("same record, other field")
            else:
                cats.append("same doc, other place")
        return loss, addr, em, first, cats, p_gold

    def grad_norms():
        groups = {"key side (pk, Wk, seg)": ("pk.", "Wk.", "seg"), "value side (pv, Wv)": ("pv.", "Wv."),
                  "readout (Wo, gate)": ("Wo.",), "write gate (Wg, beta)": ("Wg.", "beta"), "tau": ("tau",)}
        out = {}
        for gname, prefs in groups.items():
            s = 0.0
            for n_, p_ in mem.named_parameters():
                if p_.grad is not None and any(n_.startswith(x) for x in prefs):
                    s += float(p_.grad.float().pow(2).sum())
            out[gname] = math.sqrt(s)
        if gate.grad is not None:
            out["readout (Wo, gate)"] = math.sqrt(out["readout (Wo, gate)"] ** 2 + float(gate.grad) ** 2)
        return out

    @torch.no_grad()
    def evaluate(docs, qsets, tag):
        mem.eval()
        M, index = write(docs, False)
        res = {}
        for g, qs in qsets.items():
            em, fi, cats, pg = [], [], {}, []
            for j in range(0, len(qs), 64):
                _, _, e, f, c, p = answer(M, index, qs[j:j + 64], False)
                em += e; fi += f; pg += p
                for x in c:
                    cats[x] = cats.get(x, 0) + 1
            n = max(1, len(qs))
            res[g] = {"exact": sum(em) / n, "first": sum(fi) / n, "n": len(qs), "p_gold": sum(pg) / n,
                      "read": {k: round(v / n, 2) for k, v in sorted(cats.items(), key=lambda kv: -kv[1])}}
        mem.train()
        line = " | ".join(f"{g} {r['exact']:.2f}/{r['first']:.2f} (n {r['n']}, p(gold) {r['p_gold']:.2f})"
                          for g, r in res.items())
        print(f"EVAL {tag}: {line}", flush=True)
        for g, r in res.items():
            print(f"    read {g}: " + ", ".join(f"{k} {v}" for k, v in r["read"].items()), flush=True)
        return res

    def train_stage(docs, train_qs, qsets, label):
        t_stage, steps_to = time.time(), {}
        hist = []
        res = evaluate(docs, qsets, f"{label} step 0 (no training on this stage)")
        hist.append((0, res))
        for step in range(1, A.stage_steps + 1):
            t0 = time.time()
            M, index = write(docs, True)
            batch = rng.sample(train_qs, min(A.nq, len(train_qs)))
            loss, addr, em, first, _, _ = answer(M, index, batch, True)
            t0 = time.time()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            t0 = tick("backward", t0)
            if step in (1, 10, 50) or step % 100 == 0:
                gn = grad_norms()
                print(f"  grad norms step {step}: " + " | ".join(f"{k} {v:.2e}" for k, v in gn.items()), flush=True)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            tick("optimizer", t0)
            if step % 25 == 0:
                print(f"  {label} step {step} loss {float(loss):.3f} addr-nll {float(addr):.2f} exact {sum(em) / len(em):.2f} "
                      f"first {sum(first) / len(first):.2f} tau {float(mem.tau.exp()):.1f} [{time.time() - t_stage:.0f}s]",
                      flush=True)
            if step % A.eval_every == 0 or step == A.stage_steps:
                res = evaluate(docs, qsets, f"{label} step {step}")
                hist.append((step, res))
                trained = [g for g in qsets if g in train_groups[label]]
                for thr in (0.5, 0.9, A.solve):
                    if thr not in steps_to and all(res[g]["exact"] >= thr for g in trained):
                        steps_to[thr] = step
                if A.solve in steps_to:
                    break
        print(f"STAGE {label}: steps to exact >= .5 / .9 / {A.solve}: "
              + " / ".join(str(steps_to.get(t, "-")) for t in (0.5, 0.9, A.solve))
              + f" [{time.time() - t_stage:.0f}s]", flush=True)
        return hist, steps_to

    docs_a = new_docs(A.docs_a)
    docs_b = new_docs(A.docs_b)
    prng = random.Random(A.seed + 7)
    docs_a_cf = [(n_, perturb(o_, prng, pool)) for n_, o_ in docs_a]          # A': same names, other values
    docs_c = []
    for n_, o_ in docs_a:                                                     # C: A's documents renamed
        while True:
            nm = H._doc_name(rng)
            if nm not in used_names:
                used_names.add(nm)
                break
        docs_c.append((nm, perturb(o_, prng, pool)))
    qa = [q for n_, o_ in docs_a for q in questions(n_, o_, "A")]
    qa_cf = [q for n_, o_ in docs_a_cf for q in questions(n_, o_, "A'")]   # same seeds -> same question texts
    qb = [q for n_, o_ in docs_b for q in questions(n_, o_, "B")]
    qc = [q for n_, o_ in docs_c for q in questions(n_, o_, "C")]
    nunits = lambda docs: sum(len(serialize(o_, n_)[1]) for n_, o_ in docs)
    print(f"WORLDS: A {[n_ for n_, _ in docs_a]} ({nunits(docs_a)} units, {len(qa)} q) | B {[n_ for n_, _ in docs_b]} "
          f"({nunits(docs_b)} units, {len(qb)} q) | C {[n_ for n_, _ in docs_c]} (A renamed, other values; "
          f"{len(qc)} q) | start {'trained ' + A.load if A.load else 'from scratch'}", flush=True)
    print("  example A:", qa[0]["q"], "->", qa[0]["a"], "| A':", qa_cf[0]["q"], "->", qa_cf[0]["a"], flush=True)
    train_groups = {"S1": {"A"}, "S2": {"A", "B"}, "S3": {"A", "B", "C"}}
    report = {"args": vars(A)}
    stages = A.stages.split(",")
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    if "1" in stages:
        h, s = train_stage(docs_a, qa, {"A": qa}, "S1")
        report["S1"] = {"steps_to": s}
        report["S1"]["counterfactual"] = evaluate(docs_a_cf, {"A'": qa_cf}, "S1 counterfactual A' (same questions, other values)")
    if "2" in stages:
        h, s = train_stage(docs_a + docs_b, qa + qb, {"A": qa, "B": qb}, "S2")
        report["S2"] = {"steps_to": s, "zero_shot": h[0][1]}
    if "3" in stages:
        h, s = train_stage(docs_a + docs_b + docs_c, qa + qb + qc, {"A": qa, "B": qb, "C": qc}, "S3")
        report["S3"] = {"steps_to": s, "zero_shot": h[0][1]}
    try:
        import psutil
        rss = psutil.Process(os.getpid()).memory_info().rss / 2 ** 30
    except Exception:
        rss = float("nan")
    gpu = torch.cuda.max_memory_allocated() / 2 ** 30 if dev.type == "cuda" else 0.0
    tot = sum(timers.values())
    print("TIME by phase: " + " | ".join(f"{k} {v:.0f}s ({100 * v / tot:.0f}%)"
                                         for k, v in sorted(timers.items(), key=lambda kv: -kv[1])), flush=True)
    print(f"MEMORY: GPU peak {gpu:.2f} GiB | CPU RSS {rss:.2f} GiB | module params "
          f"{sum(p_.numel() for p_ in mem.parameters()) / 1e6:.1f}M | store per unit {3 * A.dk * 4 / 1024:.1f} KiB",
          flush=True)
    report["time"] = timers
    report["memory"] = {"gpu_peak_gib": gpu, "cpu_rss_gib": rss}
    if A.out:
        json.dump(report, open(A.out, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
