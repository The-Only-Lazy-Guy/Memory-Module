"""Any-schema JSON for the fast-weight memory (docs/GNM.md section 10.2 #5: general / real JSON).

Each step a few random JSON documents (gmem_hub.random_doc: arbitrary keys in mixed styles, nesting, lists of
records, emails / codes / dates / floats / names) are written into one memory; the documents share key names, so
the document name and the record label must disambiguate. One unit per scalar leaf, pooled over the leaf's pair
AND its pointer chain (the ancestor keys and the record label; design section 1.4). Questions are lookups worded
like the hub (eval wording never seen in training). The answer is the whole value (multi-token): Walker decodes
it with the read added at every answer position. Metric: exact match of the value on fresh documents.

    python gnm_fwm_anyjson.py --hf gpt2 --steps 4000
"""
import argparse
import json
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

def serialize(obj, doc, vprefix=None):
    """One JSON document as text pieces. Returns pieces and, per scalar leaf, (path, value, piece, pointer pieces)
    where the pointer pieces are the document name, every ancestor key and the enclosing record's label pair.
    vprefix (a list): filled per leaf with the text of its piece BEFORE the value (the value's tokens = the rest)."""
    pieces = [f' {json.dumps(doc)}: {{']
    leaves = []

    def emit(o, path, anc):
        if isinstance(o, dict):
            start = len(leaves)
            # the record's label = the hub's record_label rule (id/name-like fields, or a short text field in a list)
            lab, _ = H.record_label(o, in_list=bool(path) and isinstance(path[-1], int))
            lab_pairs = set(lab.split("; ")) if lab else set()
            labels = []
            for j, (k, v) in enumerate(o.items()):
                sep = "," if j else ""
                if H._is_scalar(v):
                    pieces.append(f'{sep} {json.dumps(str(k))}: {json.dumps(v)}')
                    leaves.append([path + (str(k),), H.scalar(v), len(pieces) - 1, list(anc)])
                    if vprefix is not None:
                        vprefix.append(f'{sep} {json.dumps(str(k))}:')
                    if f"{H.words(k)} = {H.scalar(v)}" in lab_pairs:
                        labels.append(len(pieces) - 1)
                else:
                    pieces.append(f'{sep} {json.dumps(str(k))}:')
                    kidx = len(pieces) - 1
                    pieces.append(" {" if isinstance(v, dict) else " [")
                    emit(v, path + (str(k),), anc + [kidx])
                    pieces.append(" }" if isinstance(v, dict) else " ]")
            for lf in leaves[start:]:                       # the record's label is part of every member's pointer
                lf[3].extend(x for x in labels if x != lf[2] and x not in lf[3])
        elif isinstance(o, list):
            for i, v in enumerate(o):
                sep = "," if i else ""
                if H._is_scalar(v):
                    pieces.append(f'{sep} {json.dumps(v)}')
                    leaves.append([path + (i,), H.scalar(v), len(pieces) - 1, list(anc)])
                    if vprefix is not None:
                        vprefix.append(sep)
                else:
                    pieces.append(f'{sep} {{' if isinstance(v, dict) else f'{sep} [')
                    emit(v, path + (i,), anc)
                    pieces.append(" }" if isinstance(v, dict) else " ]")
    emit(obj, (), [0])
    pieces.append(" }")
    return pieces, leaves


def kind_of(v):
    import re
    if "@" in v:
        return "email"
    if v.startswith("http"):
        return "url"
    if re.fullmatch(r"20\d\d-\d\d-\d\d", v):
        return "date"
    if re.fullmatch(r"[A-Z]\d+-[A-Z]", v):
        return "code"
    if re.fullmatch(r"-?\d+(\.\d+)?", v):
        return "number"
    if v in ("true", "false", "null"):
        return "bool/null"
    return "text"


def perturb(obj, rng, pool, in_list=False):
    """Same structure, keys and record labels; every other scalar gets another value of the same kind."""
    if isinstance(obj, dict):
        lab, _ = H.record_label(obj, in_list=in_list)
        keep = set(lab.split("; ")) if lab else set()
        out = {}
        for k, v in obj.items():
            if H._is_scalar(v):
                if f"{H.words(k)} = {H.scalar(v)}" in keep:
                    out[k] = v
                else:
                    cands = [x for x in pool.get(kind_of(H.scalar(v)), []) if H.scalar(x) != H.scalar(v)]
                    out[k] = rng.choice(cands) if cands else v
            else:
                out[k] = perturb(v, rng, pool, in_list=isinstance(v, list))
        return out
    if isinstance(obj, list):
        return [perturb(v, rng, pool, in_list=True) if not H._is_scalar(v) else
                (rng.choice([x for x in pool.get(kind_of(H.scalar(v)), [v]) if H.scalar(x) != H.scalar(v)] or [v]))
                for v in obj]
    return obj


def _rss():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 2 ** 30
    except Exception:
        return float("nan")


def main():
    P = argparse.ArgumentParser()
    P.add_argument("--hf", default="gpt2")
    P.add_argument("--ckpt", default="", help="the real 1B Walker checkpoint (frozen); overrides --hf")
    P.add_argument("--enc-layer", type=int, default=0, help="encoder reads states at the input of this block "
                   "(0 = the R-pass layer for GPT-2 / 2/3 of the depth for the 1B)")
    P.add_argument("--docs", type=int, default=3, help="documents written into one memory per step")
    P.add_argument("--nq", type=int, default=32)
    P.add_argument("--max-ans", type=int, default=8, help="answer tokens (values longer are truncated)")
    P.add_argument("--dk", type=int, default=512)
    P.add_argument("--inject", type=int, default=6)
    P.add_argument("--lr", type=float, default=1e-3)
    P.add_argument("--steps", type=int, default=4000)
    P.add_argument("--eval-every", type=int, default=500)
    P.add_argument("--eval-keys", choices=("train", "heldout"), default="train",
                   help="eval schemas use the training key vocabulary (default) or the hub's held-out keys")
    P.add_argument("--interface", choices=("point", "layers"), default="point",
                   help="point: one read (question encoded alone) injected at --inject; layers: the LM reads the memory "
                        "at several layers, each position querying from its own state")
    P.add_argument("--mem-layers", default="", help="blocks with a memory read (default: 1/4, 1/2, 3/4, depth-2)")
    P.add_argument("--tau0", type=float, default=10.0, help="initial read sharpness")
    P.add_argument("--oracle-read", action="store_true", help="DIAGNOSTIC: read = the gold unit's value")
    P.add_argument("--aux-lens", type=float, default=0.0, help="logit-lens auxiliary loss weight (read -> first "
                   "answer token through Walker's frozen embedding)")
    P.add_argument("--addr-loss", type=float, default=0.0,
                   help="addressing loss weight: the read's attention must land on the asked leaf's unit (the gold unit "
                        "is known by construction, as in self-supervised cloze recall of written facts)")
    P.add_argument("--overlap", type=float, default=0.0,
                   help="chance that a document is a renamed copy of the first one (same keys, record labels and "
                        "paths, other values): hard negatives that only the document name separates")
    P.add_argument("--value-slots", type=int, default=0,
                   help="per-token value slots: answer position t also gets the addressed unit's value-token state t")
    P.add_argument("--eval-overlap", type=float, default=0.0,
                   help="eval memories also contain renamed overlapping copies (same chance per document)")
    P.add_argument("--slot-read", choices=("soft", "hard"), default="soft",
                   help="hard: value slots come from the top unit only (one-hot, straight-through gradient): a split "
                        "read otherwise mixes two units' tokens ('K540-Z' -> '785-Z')")
    P.add_argument("--show", type=int, default=0, help="eval: print this many wrong answers (target vs prediction)")
    P.add_argument("--seg", action="store_true", help="segment vector on pointer tokens before pooling")
    P.add_argument("--ptr-pool", action="store_true", help="pool the pointer chain separately into the key")
    P.add_argument("--pos-readout", action="store_true", help="answer position t gets its own projection of the read")
    P.add_argument("--eval-wording", choices=("eval", "train"), default="eval")
    P.add_argument("--seed", type=int, default=0)
    P.add_argument("--save", default="runs/fwm_anyjson.pt")
    P.add_argument("--load", default="")
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
    mem = W.FastWeightMemory(d, A.dk, 1, tie=True, center=False, tok_emb=True, act="softmax", tau0=A.tau0,
                             ptr_pool=A.ptr_pool, seg=A.seg, vslots=A.value_slots).to(dev)
    gate = nn.Parameter(torch.tensor(1.0, device=dev))
    # position-aware readout: answer position t gets Wo(r) + U_t(r), so position t can emit token t of a multi-token
    # value (aj1: first token .70 but exact .57; codes / emails almost never exact). Still Walker's own head.
    Up = nn.ModuleList([nn.Linear(A.dk, d, bias=False) for _ in range(A.max_ans)]).to(dev) if A.pos_readout else None
    if Up is not None:
        for u in Up:
            nn.init.zeros_(u.weight)
    if A.load:
        st = torch.load(A.load, map_location=dev)
        miss = mem.load_state_dict(st["mem"], strict=False)    # new zero-init parameters (seg) may be missing
        print(f"loaded {A.load}; new: {miss.missing_keys}", flush=True)
        with torch.no_grad():
            gate.copy_(st["gate"])
        if Up is not None and "up" in st:
            Up.load_state_dict(st["up"])
    # value-slot readout: its own projection into Walker's residual and its own gate (0 = the model without slots)
    Us = nn.Linear(A.dk, d).to(dev) if A.value_slots else None
    gs = nn.Parameter(torch.tensor(0.0, device=dev))
    if A.load and Us is not None and st.get("us") is not None:
        Us.load_state_dict(st["us"])
        with torch.no_grad():
            gs.copy_(st["gs"])
    params = (list(mem.parameters()) + [gate] + (list(Up.parameters()) if Up is not None else [])
              + (list(Us.parameters()) + [gs] if Us is not None else []))
    opt = torch.optim.AdamW(params, lr=A.lr, weight_decay=0.01)
    inj = A.inject % len(model.blocks)
    ans_ids = tok.ids(" Answer:")
    a0 = len(ans_ids) - 1                                   # the position that predicts the first answer token
    state = {"add": None}

    def pre(_m, args):                                      # the read is added at every answer position
        if state["add"] is None:
            return None
        x = args[0]
        rms = x[:, a0:].pow(2).mean(-1, keepdim=True).sqrt()
        a = state["add"]                                      # (B, d) or (B, T, d) position-aware
        if a.dim() == 2:
            a = a[:, None]
        a = a[:, :x.shape[1] - a0]
        a = a / a.pow(2).mean(-1, keepdim=True).sqrt().clamp(min=1e-6)
        x = x.clone()
        x[:, a0:a0 + a.shape[1]] = x[:, a0:a0 + a.shape[1]] + (torch.tanh(gate) * rms[:, :a.shape[1]] * a).to(x.dtype)
        sl = state.get("slots")
        if sl is not None:                                  # (B, S, d): slot t at answer position a0 + t
            sl = sl[:, :x.shape[1] - a0]
            sl = sl / sl.pow(2).mean(-1, keepdim=True).sqrt().clamp(min=1e-6)
            w_ = sl.shape[1]
            x[:, a0:a0 + w_] = x[:, a0:a0 + w_] + (torch.tanh(gs) * rms[:, :w_] * sl).to(x.dtype)
        return (x,) + tuple(args[1:])
    model.blocks[inj].register_forward_pre_hook(pre)

    # ---- multi-layer interface: the LM reads the memory at several layers, each position querying from its OWN state
    nL = len(model.blocks)
    mem_layers = ([int(x) for x in A.mem_layers.split(",")] if A.mem_layers else
                  sorted({nL // 4, nL // 2, (3 * nL) // 4, nL - 2}))
    lm = nn.ModuleDict()
    if A.interface == "layers":
        for l in mem_layers:
            q_ = nn.Linear(d, A.dk)
            o_ = nn.Linear(A.dk, d)
            lm[str(l)] = nn.ModuleDict({"q": q_, "o": o_})
        lm = lm.to(dev)
        lgate = nn.Parameter(torch.full((len(mem_layers),), 0.1, device=dev))

        def make_hook(i, l):
            def hook(_m, args):
                M_ = state.get("M")
                if M_ is None:
                    return None
                x = args[0]
                K_, V_, lg_ = M_
                h = F.layer_norm(x.float(), (x.shape[-1],))
                qn = F.normalize(lm[str(l)]["q"](h) - mem.kmean, dim=-1)
                a = torch.softmax(mem.tau.exp() * qn @ K_.t() + lg_, -1)          # (B, T, N)
                add = lm[str(l)]["o"](a @ V_)                                    # (B, T, d)
                rms = x.float().pow(2).mean(-1, keepdim=True).sqrt()
                add = add / add.pow(2).mean(-1, keepdim=True).sqrt().clamp(min=1e-6)
                state.setdefault("atts", []).append(a)
                return (x + (torch.tanh(lgate[i]) * rms * add).to(x.dtype),) + tuple(args[1:])
            return hook
        for i, l in enumerate(mem_layers):
            model.blocks[l].register_forward_pre_hook(make_hook(i, l))
        params += list(lm.parameters()) + [lgate]
        opt.add_param_group({"params": list(lm.parameters()) + [lgate]})
        print(f"INTERFACE layers: the LM reads the memory at blocks {mem_layers} (per-position queries)", flush=True)

    vpool = {}                                          # values by kind (renamed overlapping copies)
    _vr = random.Random(99)
    for _ in range(60):
        for lf in H.flatten(H.random_doc(_vr, H.KEYS_TRAIN, depth=2)):
            if lf.image is None:
                vpool.setdefault(kind_of(lf.value), []).append(lf.value)

    def episode(rng, train, off=False):
        vocab, tset = (H.KEYS_TRAIN, "train") if (train or A.eval_keys == "train") else (H.KEYS_EVAL, "eval")
        if not train and A.eval_wording == "eval":
            tset = "eval"                                    # eval wording is never seen in training
        units, qs, meta, vsl = [], [], [], []           # meta: per unit (doc, path) for the read diagnostic
        first = None
        for di_ in range(A.docs):
            doc = H._doc_name(rng)
            obj = None
            p_ov = A.overlap if train else A.eval_overlap
            if p_ov > 0 and di_ > 0 and first is not None and rng.random() < p_ov:   # no draw at 0: eval set fixed
                obj = perturb(first, rng, vpool)
                pieces, leaves = serialize(obj, doc)
                if not leaves or sum(len(tok.ids(pc)) for pc in pieces) > 1000:
                    obj = None                           # GPT-2 positions: the copy must fit as well
            if obj is None:
                while True:                              # GPT-2 positions: a document must fit in 1024 tokens
                    obj = H.random_doc(rng, vocab, depth=rng.randint(1, 3))
                    pieces, leaves = serialize(obj, doc)
                    if leaves and sum(len(tok.ids(pc)) for pc in pieces) <= 1000:
                        break
            if di_ == 0:
                first = obj
            with torch.no_grad():
                X, rt = enc.doc(pieces)
                E = model.emb(torch.tensor([t for pc in pieces for t in tok.ids(pc)], device=dev)).float()
            spans = [sorted({p for a in anc for p in range(*rt[a])} | set(range(*rt[pi])))
                     for _, _, pi, anc in leaves]
            vpos = None
            if A.value_slots:                             # value-token positions = the piece after its key prefix
                vp_ = []
                serialize(obj, doc, vp_)
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
            if A.seg and not A.ptr_pool:                  # combined span, pointer tokens marked by the segment vector
                ptr = [sorted({p for a in anc for p in range(*rt[a])}) for _, _, _, anc in leaves]
                _, k, v, g = mem.write(X.to(dev), spans, E, ptr=ptr, vpos=vpos)
                if A.value_slots:
                    vsl.append(mem.last_vslots)
            elif A.ptr_pool:
                # ADDITIVE: the key keeps aj1's combined pool (pair + pointer chain) and gets a separate pointer pool
                # on top (zero-init = exactly aj1 at the start). aj5 (own pair only + pointer pool) carried no
                # document identity at init (3 documents share key names) and never clicked (.05).
                ptr = [sorted({p for a in anc for p in range(*rt[a])}) for _, _, _, anc in leaves]
                _, k, v, g = mem.write(X.to(dev), spans, E, ptr=ptr)
            else:
                _, k, v, g = mem.write(X.to(dev), spans, E)
            units.append((mem.last_kraw, v, torch.log(g + 1e-6)))
            meta += [(doc, tuple(lf[0])) for lf in leaves]
            by_path = {tuple(lf[0]): lf for lf in leaves}
            look = [lf for lf in H.flatten(obj) if lf.image is None and tuple(lf.path) in by_path
                    and lf.path and len(lf.value.split()) <= 12]
            for lf in rng.sample(look, min(A.nq // A.docs + 1, len(look))):
                key, where = H._subject(lf, doc)
                q = (rng.choice(H._T_LOOK[tset]).format(key=key, where=where, doc=doc) if where
                     else rng.choice(H._T_TOP[tset]).format(key=key, doc=doc))
                qs.append((q, lf.value, kind_of(lf.value), any(isinstance(p_, int) for p_ in lf.path),
                           meta.index((doc, tuple(lf.path)))))
        raw = torch.cat([u[0] for u in units])
        Vv = torch.cat([u[1] for u in units])
        lg = torch.cat([u[2] for u in units])
        mem.kmean = raw.mean(0, keepdim=True)               # all documents' keys centred by ONE (the memory's) mean
        K = F.normalize(raw - mem.kmean, dim=-1)
        M = (K, Vv, lg)
        if A.interface == "layers":
            # the LM reads question + " Answer:" + answer (teacher forced); the facts exist ONLY in the memory
            ys = [tok.ids(" " + q[1])[:A.max_ans] for q in qs]
            seqs_ = [tok.ids(" " + q[0]) + ans_ids + y[:-1] for q, y in zip(qs, ys)]
            a0s = [len(tok.ids(" " + q[0])) + len(ans_ids) - 1 for q in qs]
            Lm = max(len(x) for x in seqs_)
            inp = torch.tensor([x + [0] * (Lm - len(x)) for x in seqs_], device=dev)
            tgt = torch.full((len(ys), Lm), -100, dtype=torch.long, device=dev)
            for b, (y, a0b) in enumerate(zip(ys, a0s)):
                tgt[b, a0b:a0b + len(y)] = torch.tensor(y, device=dev)
            state["M"] = None if off else M
            state["atts"] = []
            logits, _ = model(inp, r_active=False)
            state["M"] = None
            logits = logits.float()
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), ignore_index=-100)
            ar = torch.arange(len(ys), device=dev)
            if state["atts"]:
                # addressing: at every answer position, every memory layer's read should land on the asked leaf
                gold = torch.tensor([q[4] for q in qs], device=dev)
                am = tgt != -100
                pg = torch.stack([at_[ar, :, gold] for at_ in state["atts"]])          # (layers, B, T)
                episode.p_gold = float(pg[:, am].mean())
                if A.addr_loss > 0 and not off:
                    loss = loss + A.addr_loss * -torch.log(pg[:, am] + 1e-9).mean()
            ok = (logits.argmax(-1) == tgt) | (tgt == -100)
            em = ok.all(-1).float()
            a0t = torch.tensor(a0s, device=dev)
            first = (logits[ar, a0t].argmax(-1) == tgt[ar, a0t]).float()
            episode.tags = [(q[2], q[3]) for q in qs]
            # diagnostic: where does the LAST memory layer's read at the answer position peak
            cats = []
            if state["atts"]:
                top = state["atts"][-1][ar, a0t].argmax(-1).tolist()
                for q, t_ in zip(qs, top):
                    g_ = q[4]
                    (dg, pg), (dt, pt) = meta[g_], meta[t_]
                    same_field = [x for x in pg if not isinstance(x, int)] == [x for x in pt if not isinstance(x, int)]
                    cats.append("right leaf" if t_ == g_ else "other document" if dg != dt else
                                "same field, other record" if same_field else
                                "same record, other field" if pg[:-1] == pt[:-1] else "other place")
            else:
                cats = ["other place"] * len(qs)
            episode.cats = cats
            state["atts"] = []
            return loss, em, first, K.shape[0], float(g.mean())
        texts = [q[0] for q in qs]
        seqs = [tok.ids(" " + t) for t in texts]
        n = max(len(s) for s in seqs)
        ids = torch.tensor([s + [0] * (n - len(s)) for s in seqs], device=dev)
        mq = torch.tensor([[1] * len(s) + [0] * (n - len(s)) for s in seqs], device=dev, dtype=torch.bool)
        with torch.no_grad():
            Hq, Eq = enc._states(ids).float(), model.emb(ids).float()
        rstar, R, pi = mem.read(M, Hq, mq, Eq)
        qn_, tq_, _ = mem.last_q[0]
        att = torch.softmax(tq_ * qn_ @ K.t() + lg, -1)
        if A.oracle_read:      # DIAGNOSTIC ONLY: perfect addressing (the gold unit's value) -> is decoding the wall?
            rstar = Vv[torch.tensor([q[4] for q in qs], device=dev)]
        ys = [tok.ids(" " + q[1])[:A.max_ans] for q in qs]
        L = max(len(y) for y in ys)
        inp = torch.tensor([ans_ids + y[:-1] + [0] * (L - len(y)) for y in ys], device=dev)
        tgt = torch.full((len(ys), len(ans_ids) + L - 1), -100, dtype=torch.long, device=dev)
        for b, y in enumerate(ys):
            tgt[b, a0:a0 + len(y)] = torch.tensor(y, device=dev)
        if off:
            state["add"] = None
        elif Up is not None:
            base_ = mem.Wo(rstar)
            state["add"] = torch.stack([base_ + Up[t](rstar) for t in range(len(Up))], 1)
        else:
            state["add"] = mem.Wo(rstar)
        if A.value_slots and not off:
            a_s = att
            if A.slot_read == "hard":
                hard = F.one_hot(att.argmax(-1), att.shape[-1]).to(att.dtype)
                a_s = hard + att - att.detach()
            state["slots"] = Us(torch.einsum("bn,nsd->bsd", a_s, torch.cat(vsl)))
        logits, _ = model(inp, r_active=False)
        state["add"] = None
        state["slots"] = None
        logits = logits.float()
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), ignore_index=-100)
        if A.aux_lens > 0 and not off:
            # logit lens: the read itself, in Walker's own vocabulary space (frozen embedding), must point at the
            # answer's first token -> a direct gradient from the final answer to the memory (1B: the frozen blocks
            # pass ~3x less of it, GNM.md section 6). Walker's blocks still produce the answer.
            lens = F.layer_norm(mem.Wo(rstar), (d,)) @ model.emb.weight.float().t()
            loss = loss + A.aux_lens * F.cross_entropy(lens, tgt[:, a0])
        gold = torch.tensor([q[4] for q in qs], device=dev)
        p_gold = att[torch.arange(len(qs), device=dev), gold]
        episode.p_gold = float(p_gold.mean())
        if A.addr_loss > 0 and not off:
            loss = loss + A.addr_loss * -torch.log(p_gold + 1e-9).mean()
        ok = (logits.argmax(-1) == tgt) | (tgt == -100)
        em = ok.all(-1).float()
        first = (logits[:, a0].argmax(-1) == tgt[:, a0]).float()
        episode.tags = [(q[2], q[3]) for q in qs]
        episode.dbg = (logits.argmax(-1).tolist(), tgt.tolist(), [q[1] for q in qs], [q[2] for q in qs])
        # diagnostic: where does the read's attention peak (unit argmax) relative to the asked leaf
        top = att.argmax(-1).tolist()
        cats = []
        for q, t_ in zip(qs, top):
            g_ = q[4]
            (dg, pg), (dt, pt) = meta[g_], meta[t_]
            same_field = [x for x in pg if not isinstance(x, int)] == [x for x in pt if not isinstance(x, int)]
            if t_ == g_:
                cats.append("right leaf")
            elif dg != dt:
                cats.append("other document")
            elif same_field:
                cats.append("same field, other record")
            elif pg[:-1] == pt[:-1]:
                cats.append("same record, other field")
            else:
                cats.append("other place")
        episode.cats = cats
        return loss, em, first, K.shape[0], float(g.mean())

    @torch.no_grad()
    def evaluate(step):
        evaluate.shown = 0
        mem.eval()
        for name, off in (("fresh documents", False), ("memory off", True)):
            rng = random.Random(4242)
            em, fi, n, nu = 0.0, 0.0, 0, 0
            br = {}
            for _ in range(16):
                _, e, f, N, _ = episode(rng, False, off)
                em += float(e.sum()); fi += float(f.sum()); n += len(e); nu += N
                if A.show and not off and getattr(episode, "dbg", None):
                    for pr, tg, ans, kd in zip(*episode.dbg):
                        if evaluate.shown >= A.show:
                            break
                        pos = [j for j, t_ in enumerate(tg) if t_ != -100]
                        if all(pr[j] == tg[j] for j in pos):
                            continue
                        evaluate.shown += 1
                        print(f"  WRONG [{kd}] {ans!r}: target {[raw_tok.decode([tg[j]]) for j in pos]} "
                              f"| pred {[raw_tok.decode([pr[j]]) for j in pos]}", flush=True)
                for (kd, inl), c_ in zip(episode.tags, episode.cats):
                    if inl:
                        rd = br.setdefault("READ(in list): " + c_, [0, 0.0, 0.0])
                        rd[0] += 1
                    ra = br.setdefault("READALL: " + c_, [0, 0.0, 0.0])
                    ra[0] += 1
                for (kd, inl), e_, f_ in zip(episode.tags, e.tolist(), f.tolist()):
                    for key_ in (kd, "in list" if inl else "not in list"):
                        b_ = br.setdefault(key_, [0, 0.0, 0.0])
                        b_[0] += 1; b_[1] += e_; b_[2] += f_
            if not off:
                out_score = em / n                       # (aj1 bug: returned the memory-off score -> saved step 1000)
            print(f"EVAL step {step} {name}: exact {em / n:.2f} | first token {fi / n:.2f} | n {n} | "
                  f"units/memory {nu / 16:.0f}", flush=True)
            if not off:
                print("  by kind: " + " | ".join(f"{k_} {v_[1] / v_[0]:.2f}/{v_[2] / v_[0]:.2f} (n {v_[0]})"
                                                for k_, v_ in sorted(br.items()) if not k_.startswith("READ")), flush=True)
                nl = sum(v_[0] for k_, v_ in br.items() if k_.startswith("READ("))
                print("  read (in-list questions): " + " | ".join(f"{k_[15:]} {v_[0] / max(1, nl):.2f}"
                                                               for k_, v_ in sorted(br.items()) if k_.startswith("READ(")),
                      flush=True)
                na = sum(v_[0] for k_, v_ in br.items() if k_.startswith("READALL"))
                print("  read (ALL questions): " + " | ".join(f"{k_[9:]} {v_[0] / max(1, na):.2f}"
                                                           for k_, v_ in sorted(br.items()) if k_.startswith("READALL")),
                      flush=True)
        mem.train()
        return out_score

    if A.steps == 0:
        evaluate(0)
        return
    rng = random.Random(A.seed)
    t0, best = time.time(), -1.0
    print(f"ANYJSON: {A.docs} random documents per memory, dk {A.dk}, inject block {inj}, eval keys {A.eval_keys}",
          flush=True)
    for step in range(1, A.steps + 1):
        lr = A.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - 1) / A.steps)))
        for gp in opt.param_groups:
            gp["lr"] = lr
        loss, em, first, N, gw = episode(rng, True)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % 50 == 0:
            print(f"step {step} loss {float(loss):.3f} exact {float(em.mean()):.2f} first {float(first.mean()):.2f} "
                  f"units {N} write-gate {gw:.2f} tau {float(mem.tau.exp()):.1f} "
                  f"p(gold) {getattr(episode, 'p_gold', float('nan')):.2f} | GPU peak "
                  f"{(torch.cuda.max_memory_allocated() / 2 ** 30) if dev.type == 'cuda' else 0:.2f} GiB, RSS {_rss():.2f} GiB "
                  f"[{time.time() - t0:.0f}s]",
                  flush=True)
        if step % A.eval_every == 0 or step == A.steps:
            sc = evaluate(step)
            if sc > best:
                best = sc
                torch.save({"mem": mem.state_dict(), "gate": gate.detach(), "step": step,
                            "lm": lm.state_dict() if A.interface == "layers" else None,
                            "lgate": lgate.detach() if A.interface == "layers" else None,
                            "up": Up.state_dict() if Up is not None else None,
                            "us": Us.state_dict() if Us is not None else None, "gs": gs.detach()}, A.save)


if __name__ == "__main__":
    main()
