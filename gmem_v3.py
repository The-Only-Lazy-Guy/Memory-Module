#!/usr/bin/env python3
"""Graph-prefix memory v3 (prototype + bench): one vector per node, read in
front of the LM at Walker layer L.

Atomic communication types (modules exchange only these):
  GNode  one fact: path, key, value, exact value token ids, version.
         text() = canonical serialization "servers 0 port = 8443". The symbolic
         value is the source of truth; the node VECTOR is derived from it by the
         encoder and recomputed after every write, so the graph cannot drift.
  Graph  the GNodes of one state. write(i, ids) bumps the version and keeps the
         record's sibling views current (the routers read current values).

Modules:
  encoder  Graph nodes -> [n, D] unit vectors: frozen Walker reads node.text(),
           attention pooling + MLP (trainable).
  inject   unit vectors -> Walker's residual stream at the INPUT of block L,
           scaled to the live residual norm of the text tokens at that layer
           (block-0 input ~1.8/token, after block 0 ~980/token on the 1B, so a
           prefix at layer >= 1 is in-distribution and one at layer 0 is not).
  routers  factorized lexical row x col routers (trainer v5 features).
  writer   LOTUS+ span writer: [node vector | update text | latent slots];
           the R-pass feedback is added to the latent slots at layer L too.
  reader   top-k routed nodes as prefix, then "Q: ... A:". Two heads:
           gen   -- Walker generates the answer from the prefix;
           point -- a pointer from the final "A:" state over the k
                    contextualized prefix states selects a node, and the answer
                    is that GNode's exact symbolic value (exact by construction).

Stages: (A) reconstruction pretraining [vec(node)] + "Recall:" -> node.text();
(B) episodes: writes + reads (+ reconstruction aux). Walker stays frozen.

Run:  python gmem_v3.py --ckpt /marimo/ckpt/checkpoints/ckpt_ongoing.pt --inject 2 --out /marimo/gmem_v3
"""
import argparse
import collections
import contextlib
import dataclasses
import os
import random
import time

os.environ.setdefault("TB_INFER_R_MEAN", "0")
os.environ.setdefault("TB_PHASE_WALK_FORWARD_SCALE", "0.0")
os.environ.setdefault("TB_COMPILE", "0")
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import train_walker_1b_continue as T  # noqa: E402

V = T.GMEM_V
N_LAT = 2


# ------------------------------------------------------------------ atomic types
@dataclasses.dataclass
class GNode:
    path: tuple
    key: str
    value: str
    parent: str
    siblings: list
    ids: tuple = ()
    version: int = 0

    def text(self):
        return f"{' '.join(self.path)} = {self.value}"

    def leaf(self):
        return {"path": self.path, "key": self.key, "value": self.value, "parent": self.parent,
                "siblings": self.siblings}


class Graph:
    def __init__(self, leaves, tok):
        self.tok = tok
        self.nodes = [GNode(tuple(lf["path"]), lf["key"], lf["value"], lf["parent"], list(lf["siblings"]),
                            tuple(tok.ids(" " + lf["value"])[:V])) for lf in leaves]

    def __len__(self):
        return len(self.nodes)

    def leaves(self):
        return [n.leaf() for n in self.nodes]

    def write(self, i, ids):
        n = self.nodes[i]
        n.ids, n.value, n.version = tuple(ids), self.tok.decode(ids), n.version + 1
        for m in self.nodes:
            if m is not n and m.path[:-1] == n.path[:-1]:
                m.siblings = [(k, n.value if k == n.key else v) for k, v in m.siblings]

    def values(self):
        return {"/".join(n.path): n.value for n in self.nodes}


class Tok:
    """Memoized tokenizer view (also the `ctx` GmemData needs: .fits)."""

    def __init__(self, tok):
        self.tok, self._memo = tok, {}
        self.ph = tok("~", add_special_tokens=False)["input_ids"][0]
        self.nl = tok("\n", add_special_tokens=False)["input_ids"][0]
        self.eos = tok.eos_token_id

    def ids(self, text):
        r = self._memo.get(text)
        if r is None:
            r = self._memo[text] = list(self.tok(text, add_special_tokens=False)["input_ids"])
        return list(r)

    def fits(self, value):
        return 0 < len(self.ids(" " + value)) <= V

    def decode(self, ids):
        return self.tok.decode([t for t in ids if t != self.ph]).strip()


# ------------------------------------------------------------------ module
class GraphPrefixMemory(nn.Module):
    def __init__(self, D, R=4, width=T.GMEM_WIDTH):
        super().__init__()
        self.D, self.R = D, R
        self.pool_q = nn.Parameter(torch.randn(D) / D ** 0.5)
        self.enc = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, D), nn.GELU(), nn.Linear(D, D))
        self.gain = nn.Parameter(torch.tensor(1.0))           # prefix norm / live text norm at layer L
        self.read_wr, self.read_wc = nn.Linear(T.GMEM_NR, 1), nn.Linear(T.GMEM_NC, 1)
        self.write_wr, self.write_wc = nn.Linear(T.GMEM_NR, 1), nn.Linear(T.GMEM_NC, 1)
        self.read_tau, self.write_tau = nn.Parameter(torch.tensor(1.0)), nn.Parameter(torch.tensor(1.0))
        self.lat = nn.Parameter(torch.randn(N_LAT, D) * 0.1 / D ** 0.5)
        self.beta = nn.Parameter(torch.zeros(()))
        self.wq, self.wk = nn.Linear(D, width), nn.Linear(D, width)
        self.null = nn.Parameter(torch.zeros(width))
        self.rq, self.rk = nn.Linear(D, width), nn.Linear(D, width)        # read pointer

    def route(self, which, rowf, colf, mask):
        wr, wc, tau = ((self.read_wr, self.read_wc, self.read_tau) if which == "read"
                       else (self.write_wr, self.write_wc, self.write_tau))
        s = wr(rowf).squeeze(-1) + wc(colf).squeeze(-1)
        return (s * tau.exp().clamp(max=100)).masked_fill(~mask, float("-inf"))


class Runner:
    """Walker forwards with injection at the input of block L + final-state capture."""

    def __init__(self, target, tok, gm, layer):
        self.target, self.tok, self.gm, self.layer = target, tok, gm, layer
        self.dev = next(target.parameters()).device
        self.st = {"inj": None, "h": None}
        self.hooks = []

    @contextlib.contextmanager
    def active(self):
        def pre(_m, args, kwargs):
            inj = self.st["inj"]
            if inj is None:
                return None
            x = args[0] if args else kwargs["x"]
            rep_m, rep_v, add_m, add_v = inj
            text = ~(rep_m | add_m)
            ref = (x.detach().float().norm(dim=-1) * text).sum(1) / text.sum(1).clamp(min=1)   # [B]
            ref = ref[:, None, None]
            y = torch.where(rep_m[..., None], (self.gm.gain * ref * rep_v).to(x.dtype), x)
            if add_m is not None and bool(add_m.any()):
                y = y + (add_m[..., None] * ref * add_v).to(x.dtype)
            if args:
                return (y,) + tuple(args[1:]), kwargs
            kwargs = dict(kwargs)
            kwargs["x"] = y
            return args, kwargs

        def post(_m, _i, out):
            self.st["h"] = out[0] if isinstance(out, tuple) else out

        blk = self.target.blocks[self.layer]
        self.hooks = [blk.register_forward_pre_hook(pre, with_kwargs=True),
                      self.target.blocks[-1].register_forward_hook(post)]
        # Activation checkpointing would replay the injection hook during backward
        # after self.st["inj"] was cleared (wrong gradients). Memory forwards are
        # short, so they run without it; the LM path keeps its setting.
        ckpt = getattr(self.target, "ckpt", None)
        if ckpt is not None:
            self.target.ckpt = False
        try:
            yield self
        finally:
            for h in self.hooks:
                h.remove()
            self.hooks = []
            if ckpt is not None:
                self.target.ckpt = ckpt

    def run(self, ids, rep_m=None, rep_v=None, add_m=None, add_v=None):
        B, Tn = ids.shape
        z = torch.zeros(B, Tn, dtype=torch.bool, device=self.dev)
        self.st["inj"] = None if rep_m is None and add_m is None else (
            z if rep_m is None else rep_m, rep_v if rep_v is not None else 0.0,
            z if add_m is None else add_m, add_v if add_v is not None else 0.0)
        try:
            with torch.autocast(self.dev.type, dtype=torch.bfloat16, enabled=self.dev.type == "cuda"):
                logits, _ = self.target(ids, r_active=False)
        finally:
            self.st["inj"] = None
        return logits.float(), self.st["h"].float()

    def pad(self, seqs):
        n = max(len(s) for s in seqs)
        return torch.tensor([s + [self.tok.ph] * (n - len(s)) for s in seqs], device=self.dev)

    # ---------------------------------------------------------------- encoder
    def encode(self, nodes):
        """[n, D] unit vectors for GNodes (frozen Walker read + trainable pool/MLP)."""
        seqs = [self.tok.ids(" " + n.text()) for n in nodes]
        ids = self.pad(seqs)
        valid = torch.zeros(ids.shape, dtype=torch.bool, device=self.dev)
        for i, s in enumerate(seqs):
            valid[i, :len(s)] = True
        with torch.no_grad():
            _, h = self.run(ids)
        hn = F.layer_norm(h, h.shape[-1:])
        att = (hn @ self.gm.pool_q / self.gm.D ** 0.5).masked_fill(~valid, float("-inf")).softmax(-1)
        pooled = torch.einsum("bt,btd->bd", att, hn)
        return F.normalize(self.gm.enc(pooled), dim=-1)

    # ---------------------------------------------------------------- reconstruction
    def recon(self, nodes, greedy=False):
        vec = self.encode(nodes)
        pro = [self.tok.ph] + self.tok.ids("\nRecall:")
        seqs = [pro + self.tok.ids(" " + n.text()) + [self.tok.nl] for n in nodes]
        ids = self.pad(seqs)
        rep_m = torch.zeros(ids.shape, dtype=torch.bool, device=self.dev)
        rep_m[:, 0] = True
        rep_v = torch.zeros(*ids.shape, self.gm.D, device=self.dev)
        rep_v[:, 0] = vec
        logits, _ = self.run(ids, rep_m, rep_v)
        ce, n_tok, hit = 0.0, 0, []
        for b, s in enumerate(seqs):
            a, e = len(pro), len(s)
            ce = ce + F.cross_entropy(logits[b, a - 1:e - 1], ids[b, a:e], reduction="sum")
            n_tok += e - a
            hit.append(float((logits[b, a - 1:e - 1].argmax(-1) == ids[b, a:e]).all()))
        return ce / max(1, n_tok), hit

    # ---------------------------------------------------------------- routing
    def route(self, which, texts, graphs):
        N = max(len(g) for g in graphs)
        B = len(texts)
        rowf = torch.zeros(B, N, T.GMEM_NR, device=self.dev)
        colf = torch.zeros(B, N, T.GMEM_NC, device=self.dev)
        mask = torch.zeros(B, N, dtype=torch.bool, device=self.dev)
        for b, (t, g) in enumerate(zip(texts, graphs)):
            rf, cf = T.gmem_features(t, g.leaves())
            rowf[b, :len(g)], colf[b, :len(g)], mask[b, :len(g)] = rf.to(self.dev), cf.to(self.dev), True
        return self.gm.route(which, rowf, colf, mask)

    # ---------------------------------------------------------------- write
    def write(self, graphs, nodes, texts, gold_values=None):
        """LOTUS+ span write for each (graph, node index, update text). Mutates the
        graphs (gold span when gold_values is given = teacher forcing). Returns
        (span CE or None, predicted value strings, conf [B,2])."""
        gm, B = self.gm, len(graphs)
        upd = [self.tok.ids(" " + t) for t in texts]
        L = max(len(u) for u in upd)
        upd_ids = self.pad(upd)[:, :L]
        node_vec = self.encode([g.nodes[i] for g, i in zip(graphs, nodes)])
        npre, lat0 = 1, 1 + L
        ids = torch.full((B, 1 + L + N_LAT), self.tok.ph, dtype=torch.long, device=self.dev)
        ids[:, 1:1 + L] = upd_ids
        rep_m = torch.zeros(ids.shape, dtype=torch.bool, device=self.dev)
        rep_m[:, 0] = True
        rep_v = torch.zeros(*ids.shape, gm.D, device=self.dev)
        rep_v[:, 0] = node_vec
        add_m = torch.zeros_like(rep_m)
        add_m[:, lat0:] = True
        h = h_prev = full = None
        for _ in range(gm.R):
            fb = gm.lat.unsqueeze(0).expand(B, -1, -1)
            if h is not None:
                d = h + (gm.beta * (h - h_prev) if h_prev is not None else 0.0)
                fb = fb + F.normalize(d, dim=-1)
            add_v = torch.zeros(*ids.shape, gm.D, device=self.dev)
            add_v[:, lat0:] = fb
            _, full = self.run(ids, rep_m, rep_v, add_m, add_v)
            h_prev, h = h, full[:, lat0:]
        q = gm.wq(F.layer_norm(h, h.shape[-1:]))
        k = gm.wk(F.layer_norm(full[:, npre:npre + L], (gm.D,)))
        scale = k.shape[-1] ** 0.5
        sc = torch.einsum("bjw,blw->bjl", q, k) / scale
        valid = upd_ids != self.tok.ph
        s_null = (q[:, 0] * gm.null).sum(-1, keepdim=True) / scale
        s_start = torch.cat([sc[:, 0].masked_fill(~valid, float("-inf")), s_null], 1)
        s_end = sc[:, 1].masked_fill(~valid, float("-inf"))
        off = torch.arange(L, device=self.dev)
        allowed = ((off[None, :] - off[:, None]) >= 0) & ((off[None, :] - off[:, None]) < V)
        pair = (s_start[:, :L, None] + s_end[:, None, :]).masked_fill(~allowed, float("-inf"))
        flat = pair.flatten(1).argmax(-1)
        st, en = (flat // L).tolist(), (flat % L).tolist()
        dowrite = (s_start.argmax(-1) != L).tolist()
        ps = torch.softmax(s_start, -1).gather(1, torch.tensor(st, device=self.dev)[:, None])[:, 0]
        pe = torch.softmax(s_end.masked_fill(~allowed[st], float("-inf")), -1).gather(
            1, torch.tensor(en, device=self.dev)[:, None])[:, 0]
        conf = torch.stack([ps, pe], 1)
        pred = [self.tok.decode(upd[b][st[b]:en[b] + 1]) if dowrite[b] else "" for b in range(B)]
        ce = None
        spans = [(st[b], en[b], dowrite[b]) for b in range(B)]
        if gold_values is not None:
            gst, gen = [], []
            for b, v in enumerate(gold_values):
                vi = self.tok.ids(" " + v)[:V]
                s0 = next((s for s in range(len(upd[b]) - len(vi), -1, -1) if upd[b][s:s + len(vi)] == vi), -1)
                gst.append(s0)
                gen.append(s0 + len(vi) - 1)
            has = [b for b in range(B) if gst[b] >= 0]
            if has:
                hi = torch.tensor(has, device=self.dev)
                gs = torch.tensor([gst[b] for b in has], device=self.dev)
                ge = torch.tensor([gen[b] for b in has], device=self.dev)
                ce = F.cross_entropy(s_start[hi], gs) + F.cross_entropy(
                    s_end[hi].masked_fill(~allowed[gs], float("-inf")), ge)
                for b in has:
                    spans[b] = (gst[b], gen[b], True)
        for b, (g, i) in enumerate(zip(graphs, nodes)):
            a, e, w = spans[b]
            if w:
                g.write(i, upd[b][a:e + 1])
        return ce, pred, conf

    # ---------------------------------------------------------------- read
    def read_batch(self, graphs, picks, questions, answers):
        """picks: list of node-index lists (prefix order). Returns ids, rep_m, rep_v, spans."""
        K = max(len(p) for p in picks)
        flat = [g.nodes[i] for g, p in zip(graphs, picks) for i in p]
        vec = self.encode(flat)
        seqs, spans = [], []
        for p, q, a in zip(picks, questions, answers):
            pre = [self.tok.ph] * K + self.tok.ids(f"\nQ: {q}\nA:")
            ans = (self.tok.ids(" " + a) + [self.tok.nl]) if a is not None else []
            seqs.append(pre + ans)
            spans.append((len(pre), len(pre) + len(ans)))
        ids = self.pad(seqs)
        rep_m = torch.zeros(ids.shape, dtype=torch.bool, device=self.dev)
        rep_v = torch.zeros(*ids.shape, self.gm.D, device=self.dev)
        j = 0
        for b, p in enumerate(picks):
            rep_m[b, :len(p)] = True
            rep_v[b, :len(p)] = vec[j:j + len(p)]
            j += len(p)
        return ids, rep_m, rep_v, spans

    def point(self, h, spans, picks):
        """Read pointer: final 'A:' state -> k contextualized prefix states. [B, K] logits."""
        gm = self.gm
        K = max(len(p) for p in picks)
        last = torch.tensor([s - 1 for s, _ in spans], device=self.dev)
        q = gm.rq(F.layer_norm(h[torch.arange(h.shape[0], device=self.dev), last], (gm.D,)))
        k = gm.rk(F.layer_norm(h[:, :K], (gm.D,)))
        logit = torch.einsum("bw,bkw->bk", q, k) / k.shape[-1] ** 0.5
        valid = torch.zeros(logit.shape, dtype=torch.bool, device=self.dev)
        for b, p in enumerate(picks):
            valid[b, :len(p)] = True
        return logit.masked_fill(~valid, float("-inf"))

    @torch.no_grad()
    def greedy(self, graphs, picks, questions, max_new=10):
        ids, rep_m, rep_v, spans = self.read_batch(graphs, picks, questions, [None] * len(questions))
        rows = [ids[b, :spans[b][0]].tolist() for b in range(len(questions))]
        start = [len(r) for r in rows]
        done = [False] * len(rows)
        K = rep_m.shape[1] and int(rep_m.sum(1).max())
        for _ in range(max_new):
            t = self.pad(rows)
            m = torch.zeros(t.shape, dtype=torch.bool, device=self.dev)
            v = torch.zeros(*t.shape, self.gm.D, device=self.dev)
            m[:, :K] = rep_m[:, :K]
            v[:, :K] = rep_v[:, :K]
            logits, _ = self.run(t, m, v)
            last = torch.tensor([len(r) - 1 for r in rows], device=self.dev)
            nxt = logits[torch.arange(len(rows), device=self.dev), last].argmax(-1).tolist()
            for b, tk in enumerate(nxt):
                if not done[b]:
                    if tk in (self.tok.nl, self.tok.eos):
                        done[b] = True
                    else:
                        rows[b].append(tk)
            if all(done):
                break
        return [self.tok.decode(r[s:]) for r, s in zip(rows, start)]


# ------------------------------------------------------------------ episodes
def topk_picks(logit, n_nodes, k, gold=None):
    picks = []
    for b in range(logit.shape[0]):
        kk = min(k, n_nodes[b])
        p = logit[b, :n_nodes[b]].topk(kk).indices.tolist()
        if gold is not None and gold[b] not in p:
            p[-1] = gold[b]
        picks.append(p)
    return picks


def episodes_pass(rn, eps, k, train=True, greedy=False):
    graphs = [Graph(e["leaves"], rn.tok) for e in eps]
    losses, st = [], collections.defaultdict(list)
    for w_i in range(max((len(e["writes"]) for e in eps), default=0)):
        bk = [b for b, e in enumerate(eps) if w_i < len(e["writes"])]
        ws = [eps[b]["writes"][w_i] for b in bk]
        gs = [graphs[b] for b in bk]
        logit = rn.route("write", [w["text"] for w in ws], gs)
        gold = torch.tensor([w["node"] for w in ws], device=rn.dev)
        losses.append(F.cross_entropy(logit, gold))
        pick = logit.argmax(-1)
        st["write_route"] += (pick == gold).float().tolist()
        nodes = (gold if train else pick).tolist()
        ce, pred, _ = rn.write(gs, nodes, [w["text"] for w in ws],
                               gold_values=[w["value"] for w in ws] if train else None)
        if ce is not None:
            losses.append(ce)
        st["write_exact"] += [float(p == w["value"]) for p, w in zip(pred, ws)]
    qb, qs = [], []
    for b, e in enumerate(eps):
        for q in e["questions"]:
            qb.append(b)
            qs.append(q)
    if qs:
        qg = [graphs[b] for b in qb]
        logit = rn.route("read", [q["q"] for q in qs], qg)
        gold = [q["node"] for q in qs]
        losses.append(F.cross_entropy(logit, torch.tensor(gold, device=rn.dev)))
        top1 = logit.argmax(-1).tolist()
        st["read_route"] += [float(a == b) for a, b in zip(top1, gold)]
        picks = topk_picks(logit, [len(g) for g in qg], k, gold if train else None)
        st["read_topk"] += [float(g_ in p) for g_, p in zip(gold, picks)]
        ids, rep_m, rep_v, spans = rn.read_batch(qg, picks, [q["q"] for q in qs], [q["answer"] for q in qs])
        logits, hfin = rn.run(ids, rep_m, rep_v)
        plog = rn.point(hfin, spans, picks)
        gpos = [p.index(g_) if g_ in p else -1 for p, g_ in zip(picks, gold)]
        has = [b for b, x in enumerate(gpos) if x >= 0]
        if has:
            hi = torch.tensor(has, device=rn.dev)
            losses.append(F.cross_entropy(plog[hi], torch.tensor([gpos[b] for b in has], device=rn.dev)))
        ppos = plog.argmax(-1).tolist()
        for b, q in enumerate(qs):
            val = qg[b].nodes[picks[b][ppos[b]]].value
            hit = float(val.strip().lower() == q["answer"].strip().lower())
            st["em_point"].append(hit)
            st[f"em_point_{q['tag']}"].append(hit)
        ce, n_tok = 0.0, 0
        for b, (s, e_) in enumerate(spans):
            ce = ce + F.cross_entropy(logits[b, s - 1:e_ - 1], ids[b, s:e_], reduction="sum")
            n_tok += e_ - s
            st["read_tf_exact"].append(float((logits[b, s - 1:e_ - 1].argmax(-1) == ids[b, s:e_]).all()))
        losses.append(ce / max(1, n_tok))
        st["answer_ce"].append(float(ce.detach()) / max(1, n_tok))
        if greedy:
            for p, q in zip(rn.greedy(qg, picks, [q["q"] for q in qs]), qs):
                hit = float(p.strip(" .\"'").lower() == q["answer"].strip().lower())
                st["em"].append(hit)
                st[f"em_{q['tag']}"].append(hit)
    return (sum(losses) if losses else None), st, graphs


def clip_step(opt, clip=1.0):
    tot = 0.0
    for g in opt.param_groups:
        ps = [p for p in g["params"] if p.grad is not None]
        if ps:
            tot += float(torch.nn.utils.clip_grad_norm_(ps, clip)) ** 2
    opt.step()
    opt.zero_grad(set_to_none=True)
    return tot ** 0.5


def sample_nodes(data, rn, n):
    out = []
    while len(out) < n:
        e = data.batch(1)[0]
        g = Graph(e["leaves"], rn.tok)
        out += random.Random(len(out)).sample(g.nodes, min(len(g), n - len(out)))
    return out


def evaluate(rn, data, k, tag, step, recon_nodes):
    res = {}
    with torch.no_grad():
        for name, eps in data.eval_sets.items():
            agg = collections.defaultdict(list)
            for s in range(0, len(eps), 16):
                _, st, _ = episodes_pass(rn, eps[s:s + 16], k, train=False, greedy=True)
                for kk, v in st.items():
                    agg[kk] += v
            res[name] = {kk: round(sum(v) / len(v), 4) for kk, v in agg.items() if v}
            print(f"V3_VAL arm={tag} step={step} set={name} " + " ".join(
                f"{kk}={v}" for kk, v in res[name].items() if kk != "answer_ce"), flush=True)
        hits = []
        for s in range(0, len(recon_nodes), 32):
            _, h = rn.recon(recon_nodes[s:s + 32])
            hits += h
        res["recon_exact"] = round(sum(hits) / len(hits), 4)
        print(f"V3_VAL arm={tag} step={step} recon_exact={res['recon_exact']}", flush=True)
    return res


def main():
    P = argparse.ArgumentParser()
    P.add_argument("--ckpt", required=True)
    P.add_argument("--inject", default="2", help="comma list of block indices to inject at (one arm each)")
    P.add_argument("--k", type=int, default=8)
    P.add_argument("--R", type=int, default=4)
    P.add_argument("--recon-steps", type=int, default=200)
    P.add_argument("--qa-steps", type=int, default=300)
    P.add_argument("--batch", type=int, default=16)
    P.add_argument("--recon-batch", type=int, default=48)
    P.add_argument("--eval-every", type=int, default=100)
    P.add_argument("--sources", default="synthetic,wikisql")
    P.add_argument("--out", default="gmem_v3")
    A = P.parse_args()
    os.makedirs(A.out, exist_ok=True)
    model, raw_tok, _ = T.build_model_for_inference(A.ckpt)
    for p in model.parameters():
        p.requires_grad_(False)
    tok = Tok(raw_tok)
    data = T.GmemData(tok, A.sources.split(","), 2, seed=1)
    recon_eval = sample_nodes(data, Runner(model, tok, None, 0), 96)
    for layer in [int(x) for x in A.inject.split(",")]:
        tag = f"inject{layer}"
        torch.manual_seed(0)
        data.r = random.Random(1)
        gm = GraphPrefixMemory(model.emb.weight.shape[1], R=A.R).to(next(model.parameters()).device)
        named = list(gm.named_parameters())
        opt = torch.optim.AdamW([
            {"params": [p for n, p in named if n.startswith(("read_", "write_"))], "lr": 3e-3, "weight_decay": 0.01},
            {"params": [p for n, p in named if not n.startswith(("read_", "write_"))], "lr": 1e-3, "weight_decay": 0.0}])
        rn = Runner(model, tok, gm, layer)
        print(f"\n=== V3 ARM {tag} params={sum(p.numel() for p in gm.parameters()):,} k={A.k} R={A.R}", flush=True)
        t0 = time.time()
        with rn.active():
            model.train()
            for step in range(1, A.recon_steps + 1):
                loss, hit = rn.recon(sample_nodes(data, rn, A.recon_batch))
                loss.backward()
                gn = clip_step(opt)
                if step % 25 == 0:
                    print(f"V3 arm={tag} recon step={step} ce={float(loss):.4f} tf_exact={sum(hit)/len(hit):.3f} "
                          f"gnorm={gn:.3f} gain={float(gm.gain):.3f} elapsed={time.time()-t0:.0f}s", flush=True)
            evaluate(rn, data, A.k, tag, 0, recon_eval)
            acc = collections.defaultdict(list)
            for step in range(1, A.qa_steps + 1):
                model.train()
                loss, st, _ = episodes_pass(rn, data.batch(A.batch), A.k, train=True)
                rl, _ = rn.recon(sample_nodes(data, rn, 16))
                (loss + 0.5 * rl).backward()
                st["gnorm"] = [clip_step(opt)]
                st["recon_ce"] = [float(rl)]
                for kk, v in st.items():
                    acc[kk] += v
                if step % 25 == 0:
                    print(f"V3 arm={tag} qa step={step} " + " ".join(
                        f"{kk}={sum(v)/len(v):.4f}" for kk, v in sorted(acc.items()))
                        + f" beta={float(gm.beta):+.4f} gain={float(gm.gain):.3f} elapsed={time.time()-t0:.0f}s",
                        flush=True)
                    acc = collections.defaultdict(list)
                if step % A.eval_every == 0 or step == A.qa_steps:
                    model.eval()
                    evaluate(rn, data, A.k, tag, step, recon_eval)
        torch.save({"module": gm.state_dict(), "inject": layer, "R": A.R, "k": A.k},
                   os.path.join(A.out, f"gmem_v3_{tag}.pt"))
        print(f"V3 ARM {tag} done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
