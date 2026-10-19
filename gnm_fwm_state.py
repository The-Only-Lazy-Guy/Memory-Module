"""State streams of ANY schema for the fast-weight memory (docs/GNM.md section 9.6).

A state is a random-schema JSON document (gmem_hub.random_doc: universal or custom keys, nesting, lists of records,
any value kinds); nothing is schema-specific, the model reads the state from its k:v pairs. The state changes over
time: snapshot 0, then 1-4 ticks in which 1-4 values change (same kind; record labels stay). EVERY snapshot is
written into the memory (history is kept). Questions ask the CURRENT value (half about values that changed).

The update is the model's: a learned overwrite gate. When a snapshot is written, each older unit's gain drops by
softplus(a * max_j cos(k_old, k_new_j) + b) -- a newer unit with the same key supersedes it (no bookkeeping by path).
Starts near "no suppression". Without it (--no-overwrite) the newest and the stale units compete in the read.

    python gnm_fwm_state.py --hf gpt2 --load runs/fwm_aj10.pt
"""
import argparse
import copy
import math
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
from gnm_fwm_anyjson import serialize, kind_of


def mutable_leaves(obj):
    """Scalar leaves that may change in a state: not ids / names / record labels, not booleans/None."""
    out = []
    for lf in H.flatten(obj):
        if lf.image is not None or not lf.path or lf.path[-1] == "count" or lf.raw is None or isinstance(lf.raw, bool):
            continue
        if isinstance(lf.path[-1], str) and H.words(lf.path[-1]).split()[-1:] and \
                H.words(lf.path[-1]).split()[-1] in H._ID_WORDS:
            continue
        if lf.label_value and lf.value == lf.label_value:
            continue
        if len(lf.value) >= 40:
            continue
        out.append(lf)
    return out


def tick(rng, obj, n):
    """Change n values of the state (same kind); returns the new state and the changed paths."""
    new = copy.deepcopy(obj)
    changed = []
    cands = mutable_leaves(new)
    for lf in rng.sample(cands, min(n, len(cands))):
        kind = "int" if isinstance(lf.raw, int) else "float" if isinstance(lf.raw, float) else kind_of(lf.value)
        kind = {"number": "int", "text": "phrase", "email": "email", "url": "url", "date": "date",
                "code": "code"}.get(kind, kind)
        nv = H._value(rng, kind if kind in ("int", "float", "email", "url", "date", "code", "phrase") else "phrase")
        if H.scalar(nv) == lf.value:
            continue
        ref = new
        for p in lf.path[:-1]:
            ref = ref[p]
        ref[lf.path[-1]] = nv
        changed.append(tuple(lf.path))
    return new, changed


T_VER = {"train": ("What was the {key} of {where} in {doc} at version {k}?",
                   "In {doc} version {k}, what {key} did {where} have?"),
         "eval": ("At version {k} of {doc}, which {key} was recorded for {where}?",)}


def main():
    P = argparse.ArgumentParser()
    P.add_argument("--hf", default="gpt2")
    P.add_argument("--ckpt", default="", help="the real 1B Walker checkpoint (frozen); overrides --hf")
    P.add_argument("--enc-layer", type=int, default=0, help="encoder reads states at the input of this block "
                   "(0 = the R-pass layer for GPT-2 / 2/3 of the depth for the 1B)")
    P.add_argument("--ticks", type=int, default=4, help="max updates after the first snapshot")
    P.add_argument("--nq", type=int, default=32)
    P.add_argument("--max-ans", type=int, default=8)
    P.add_argument("--dk", type=int, default=512)
    P.add_argument("--inject", type=int, default=6)
    P.add_argument("--lr", type=float, default=3e-4)
    P.add_argument("--steps", type=int, default=6000)
    P.add_argument("--eval-every", type=int, default=1000)
    P.add_argument("--sup-q", action="store_true", help="supersession kept per unit; the query decides how much applies")
    P.add_argument("--history", action="store_true", help="versioned snapshot headers + 'before update k' questions")
    P.add_argument("--partial", action="store_true", help="updates write only the changed top-level sections")
    P.add_argument("--no-overwrite", action="store_true", help="ablation: no learned overwrite gate")
    P.add_argument("--seed", type=int, default=0)
    P.add_argument("--load", default="", help="start from a trained any-schema JSON reader (gnm_fwm_anyjson --seg)")
    P.add_argument("--save", default="runs/fwm_state.pt")
    A = P.parse_args()
    import gnm_small
    import gnm_v2 as V2
    import gmem_ltm as LT
    import train_walker_1b_continue as T
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(A.seed)
    model, raw_tok = W.load_walker(A.hf, A.ckpt, A.enc_layer, dev)
    for p in model.parameters():
        p.requires_grad_(False)
    tok = LT.Tok(raw_tok)
    enc = V2.WalkerStateEnc(model, tok)
    d = model.emb.weight.shape[1]
    mem = W.FastWeightMemory(d, A.dk, 1, tie=True, center=False, tok_emb=True, act="softmax", tau0=10.0,
                             seg=True, sup_q=A.sup_q).to(dev)
    gate = nn.Parameter(torch.tensor(1.0, device=dev))
    ow = nn.Parameter(torch.tensor([5.0, -8.0], device=dev))    # overwrite gate (a, b): starts ~no suppression
    if A.load:
        st = torch.load(A.load, map_location=dev)
        print("loaded", A.load, mem.load_state_dict(st["mem"], strict=False), flush=True)
        with torch.no_grad():
            gate.copy_(st["gate"])
            if "ow" in st:
                ow.copy_(st["ow"])
    params = list(mem.parameters()) + [gate] + ([] if A.no_overwrite else [ow])
    opt = torch.optim.AdamW(params, lr=A.lr, weight_decay=0.01)
    inj = A.inject % len(model.blocks)
    ans_ids = tok.ids(" Answer:")
    a0 = len(ans_ids) - 1
    state = {"add": None}

    def pre(_m, args):
        if state["add"] is None:
            return None
        x = args[0]
        # the read goes to the FIRST answer position only (as gnm_fwm_anyjson since aj3, the reader it loads);
        # later answer tokens see it through attention
        rms = x[:, a0:a0 + 1].pow(2).mean(-1, keepdim=True).sqrt()
        a = state["add"]
        a = a / a.pow(2).mean(-1, keepdim=True).sqrt().clamp(min=1e-6)
        x = x.clone()
        x[:, a0] = x[:, a0] + (torch.tanh(gate) * rms[:, 0] * a).to(x.dtype)
        return (x,) + tuple(args[1:])
    model.blocks[inj].register_forward_pre_hook(pre)

    def write(obj, doc):
        pieces, leaves = serialize(obj, doc)
        with torch.no_grad():
            X, rt = enc.doc(pieces)
            E = model.emb(torch.tensor([t for pc in pieces for t in tok.ids(pc)], device=dev)).float()
        spans = [sorted({p for a in anc for p in range(*rt[a])} | set(range(*rt[pi]))) for _, _, pi, anc in leaves]
        ptr = [sorted({p for a in anc for p in range(*rt[a])}) for _, _, _, anc in leaves]
        _, _, v, g = mem.write(X.to(dev), spans, E, ptr=ptr)
        return mem.last_kraw, v, torch.log(g + 1e-6)

    def episode(rng, train, off=False):
        tset = "train" if train else "eval"
        doc = H._doc_name(rng)
        while True:
            obj = H.random_doc(rng, H.KEYS_TRAIN, depth=rng.randint(1, 3))
            pieces, leaves = serialize(obj, doc)
            if len(mutable_leaves(obj)) >= 4 and sum(len(tok.ids(pc)) for pc in pieces) <= 1000:
                break
        Ks, Vs, Ls, Ss = [], [], [], []
        changed = set()
        snaps, upd = [copy.deepcopy(obj)], {}              # state after each update; path -> updates that changed it
        n_ticks = rng.randint(1, A.ticks) if A.ticks > 0 else 0
        for t in range(n_ticks + 1):
            if t > 0:
                obj, ch = tick(rng, obj, rng.randint(1, 4))
                changed |= set(ch)
                snaps.append(copy.deepcopy(obj))
                for p_ in ch:
                    upd.setdefault(p_, []).append(t)
            # --history: every snapshot's header carries its version (the time cue history questions refer to)
            dv = doc if not A.history else f"{doc} @ version {t}"
            if t > 0 and A.partial and isinstance(obj, dict):
                # partial update: only the top-level sections that contain a change are written (paths and pointer
                # chains identical to the full document); the rest of the state stays as written before
                tops = {p_[0] for p_ in ch}
                part = {k_: v_ for k_, v_ in obj.items() if k_ in tops}
                if not part:
                    continue
                kr, v, lg = write(part, dv)
            else:
                kr, v, lg = write(obj, dv)
            if Ks and not A.no_overwrite:
                # the model's update: older units with a key close to a NEW unit lose gain (superseded)
                kn = F.normalize(kr, dim=-1)
                for i in range(len(Ks)):
                    sim = (F.normalize(Ks[i], dim=-1) @ kn.t()).max(-1).values
                    if A.sup_q:                          # supersession kept separate; the query applies it
                        Ss[i] = Ss[i] + F.softplus(ow[0] * sim + ow[1]) * 10.0
                    else:
                        Ls[i] = Ls[i] - F.softplus(ow[0] * sim + ow[1]) * 10.0
            Ks.append(kr); Vs.append(v); Ls.append(lg); Ss.append(torch.zeros_like(lg))
        raw = torch.cat(Ks)
        mem.kmean = raw.mean(0, keepdim=True)
        M = (F.normalize(raw - mem.kmean, dim=-1), torch.cat(Vs), torch.cat(Ls)) +             ((torch.cat(Ss),) if A.sup_q else ())
        written = {tuple(lf[0]) for lf in serialize(obj, doc)[1]}   # only leaves that exist as units (not the hub's
        look = [lf for lf in H.flatten(obj) if lf.image is None and lf.path   # joined list summaries / counts)
                and tuple(lf.path) in written and len(lf.value.split()) <= 12]
        ch_l = [lf for lf in look if tuple(lf.path) in changed]
        un_l = [lf for lf in look if tuple(lf.path) not in changed]
        nc = A.nq // 3 if A.history else A.nq // 2
        pick = rng.sample(ch_l, min(nc, len(ch_l))) + rng.sample(un_l, min(nc, len(un_l)))
        qs = []
        for lf in pick:
            key, where = H._subject(lf, doc)
            q = (rng.choice(H._T_LOOK[tset]).format(key=key, where=where, doc=doc) if where
                 else rng.choice(H._T_TOP[tset]).format(key=key, doc=doc))
            qs.append((q, lf.value, "changed" if tuple(lf.path) in changed else "unchanged"))
        hist_cur = []
        if A.history:                                   # "what was X before update k" -> the value in snapshot k-1
            hl = [lf for lf in ch_l if H._subject(lf, doc)[1]]
            for lf in rng.sample(hl, min(nc, len(hl))):
                # a version whose value differs from the current one; the answer is the value AT that version
                vals_ = []
                for k_, sn in enumerate(snaps[:-1]):
                    ref = sn
                    for p_ in lf.path:
                        ref = ref[p_]
                    if H.scalar(ref) != lf.value:
                        vals_.append((k_, H.scalar(ref)))
                if not vals_:
                    continue
                k_, old = rng.choice(vals_)
                key, where = H._subject(lf, doc)
                q = rng.choice(T_VER[tset]).format(key=key, where=where, doc=doc, k=k_)
                qs.append((q, old, "history"))
                hist_cur.append(lf.value)                # diagnostic: the CURRENT value of that leaf
        seqs = [tok.ids(" " + q) for q, _, _ in qs]
        n = max(len(s) for s in seqs)
        ids = torch.tensor([s + [0] * (n - len(s)) for s in seqs], device=dev)
        mq = torch.tensor([[1] * len(s) + [0] * (n - len(s)) for s in seqs], device=dev, dtype=torch.bool)
        with torch.no_grad():
            Hq, Eq = enc._states(ids).float(), model.emb(ids).float()
        rstar, _, _ = mem.read(M, Hq, mq, Eq)
        ys = [tok.ids(" " + a)[:A.max_ans] for _, a, _ in qs]
        L = max(len(y) for y in ys)
        inp = torch.tensor([ans_ids + y[:-1] + [0] * (L - len(y)) for y in ys], device=dev)
        tgt = torch.full((len(ys), len(ans_ids) + L - 1), -100, dtype=torch.long, device=dev)
        for b, y in enumerate(ys):
            tgt[b, a0:a0 + len(y)] = torch.tensor(y, device=dev)
        state["add"] = None if off else mem.Wo(rstar)
        logits, _ = model(inp, r_active=False)
        state["add"] = None
        logits = logits.float()
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), ignore_index=-100)
        em = ((logits.argmax(-1) == tgt) | (tgt == -100)).all(-1).float()
        chg = torch.tensor([c != "unchanged" for _, _, c in qs], device=dev)
        episode.cats = [c for _, _, c in qs]
        # diagnostic: history questions answered with the CURRENT value (field found, version missed)
        hi = [b for b, (_, _, c) in enumerate(qs) if c == "history"]
        cur_hits = 0
        for b, cv in zip(hi, hist_cur):
            yc = tok.ids(" " + cv)[:A.max_ans]
            pr = logits[b, a0:a0 + len(yc)].argmax(-1).tolist()
            cur_hits += int(pr == yc)
        episode.hist_cur = (cur_hits, len(hi))
        return loss, em, chg, raw.shape[0]

    @torch.no_grad()
    def evaluate(step):
        mem.eval()
        out = 0.0
        for name, off in (("current state", False), ("memory off", True)):
            rng = random.Random(4242)
            s = {"changed": [0, 0], "unchanged": [0, 0], "history": [0, 0]}
            nu, hc, hn = 0, 0, 0
            for _ in range(16):
                _, em, chg, N = episode(rng, False, off)
                nu += N
                hc += episode.hist_cur[0]; hn += episode.hist_cur[1]
                for e_, k_ in zip(em.tolist(), episode.cats):
                    s[k_][0] += 1
                    s[k_][1] += e_
            tot = sum(v_[1] for v_ in s.values()) / max(1, sum(v_[0] for v_ in s.values()))
            if not off:
                out = tot
            print(f"EVAL step {step} {name}: exact {tot:.2f} | changed values {s['changed'][1] / max(1, s['changed'][0]):.2f} "
                  f"(n {s['changed'][0]}) | unchanged {s['unchanged'][1] / max(1, s['unchanged'][0]):.2f} | "
                  f"history {s['history'][1] / max(1, s['history'][0]):.2f} (n {s['history'][0]}; answered the CURRENT "
                  f"value {hc / max(1, hn):.2f}) | "
                  f"units/memory {nu / 16:.0f} | overwrite a,b {[round(float(x), 2) for x in ow]}", flush=True)
        mem.train()
        return out

    print(f"STATE STREAMS: up to {A.ticks} updates, overwrite gate {'OFF' if A.no_overwrite else 'learned'}", flush=True)
    evaluate(0)
    rng = random.Random(A.seed)
    t0, best = time.time(), -1.0
    for step in range(1, A.steps + 1):
        lr = A.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - 1) / A.steps)))
        for gp in opt.param_groups:
            gp["lr"] = lr
        loss, em, chg, N = episode(rng, True)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % 50 == 0:
            print(f"step {step} loss {float(loss):.3f} exact {float(em.mean()):.2f} changed "
                  f"{float(em[chg].mean()) if chg.any() else -1:.2f} units {N} [{time.time() - t0:.0f}s]", flush=True)
        if step % A.eval_every == 0 or step == A.steps:
            sc = evaluate(step)
            if sc > best:
                best = sc
                torch.save({"mem": mem.state_dict(), "gate": gate.detach(), "ow": ow.detach(), "step": step}, A.save)


if __name__ == "__main__":
    main()
