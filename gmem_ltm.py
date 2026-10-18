#!/usr/bin/env python3
"""Long-term memory + rich layer-L read (prototype + bench).

Why: Walker is small (~8 tokens/param) and continued training overwrites what
its weights hold (catastrophic forgetting). Facts therefore live OUTSIDE the
weights, in a persistent store, as exact symbolic content; Walker retrieves and
reads them and copies answers verbatim.

Store item (atomic type, `MemItem`): exact text (source of truth) + its GNode
(path, value, version) + a retrieval KEY (one vector) derived from Walker's
final states of the text. Old versions are kept (history), so "what was X
before it changed" is answerable.

Read path, no learned compression:
  1. retrieve top-k items for the question (dense keys, trainable projections
     over frozen Walker states);
  2. RICH injection: each item's text is run through Walker blocks 0..L-1 once;
     those per-token states -- exactly what Walker computes when the text is in
     context, unmodified -- replace placeholder positions at the input of block
     L, in front of "Q: ... A:";
  3. span pointer from the "A:" state over the injected positions copies the
     answer tokens VERBATIM (word for word by construction). The start score adds
     an exact copy-continuation cue (`match_lengths`, induction-head style), and
     both span ends add the item's retrieval log-prob (retrieve-then-read), so the
     pointer copies from the item retrieval trusts (current value vs. history).
Retrieval is hybrid: BM25 candidates over the exact texts, dense + lexical rerank.
Learned: retrieval projections, BM25 weight, 4 pointer projections, 9 match
biases, 1 retrieval-bias weight. Walker stays frozen.

Tasks (every state type becomes the same GNode / MemItem):
  wikisql    -- one global store of every leaf of many real tables; the crowd
                question must find its cell without being told the table;
  sentences  -- store wikitext sentences; recall the rest of a sentence word
                for word from its first words (held-out store at eval);
  json       -- per-session synthetic states after their writes, plus history
                items; questions on current values and on values BEFORE an update;
  chess      -- 64 square nodes + side to move; moves change two squares;
                "what is on e4", "where is the white king", "what was on g1 before";
  gui        -- accessibility tree (buttons, fields, checkboxes); events toggle
                enabled state and field values; current + history questions.

Run: python gmem_ltm.py --ckpt /marimo/ckpt/checkpoints/ckpt_ongoing.pt --out /marimo/gmem_ltm
"""
import argparse
import bisect
import collections
import itertools
import contextlib
import dataclasses
import math
import os
import random
import re
import time

os.environ.setdefault("TB_INFER_R_MEAN", "0")
os.environ.setdefault("TB_PHASE_WALK_FORWARD_SCALE", "0.0")
os.environ.setdefault("TB_COMPILE", "0")
try:                                   # before torch: streaming `datasets` loaded after torch segfaults on some hosts
    import datasets  # noqa: E402,F401
except ImportError:
    pass
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import train_walker_1b_continue as T  # noqa: E402
from gmem_v3 import GNode, Graph, Tok  # noqa: E402

MAX_SPAN = 48
MATCH_CAP = 24          # > any 6-word head, so the true start never ties with the token before it


def match_lengths(pref, q, cap=MATCH_CAP):
    """Per prefix position t: length of the longest run of tokens right BEFORE t
    (pref[t-k:t]) that also occurs contiguously in the question (capped). An exact
    copy-continuation cue, like an induction head: "begins 'a b c'" -> start after c."""
    grams = {tuple(q[i:i + k]) for k in range(1, cap + 1) for i in range(len(q) - k + 1)}
    out = []
    for t in range(len(pref)):
        k = 0
        while k < min(cap, t) and tuple(pref[t - k - 1:t]) in grams:
            k += 1
        out.append(k)
    return out


# ------------------------------------------------------------------ atomic store item
@dataclasses.dataclass
class MemItem:
    text: str                 # exact content (source of truth)
    node: GNode = None        # structured view (None for free text)
    version: int = 0          # 0 = current; k > 0 = k-th older version (history)


def _in_list(n):
    return any(p.isdigit() for p in n.path)


def node_text(n, max_sib=8):
    """Exact node text. Record context (sibling fields) only for list entries,
    where it identifies the record ("servers 1 port = 8443 (name = web-3)")."""
    sib = "; ".join(f"{k} = {v}" for k, v in n.siblings[:max_sib]) if _in_list(n) else ""
    return f"{' '.join(n.path)} = {n.value}" + (f" ({sib})" if sib else "")


def history_text(n, old, k):
    sib = "; ".join(f"{a} = {b}" for a, b in n.siblings[:8]) if _in_list(n) else ""
    return f"{' '.join(n.path)} was {old} before update {k}" + (f" ({sib})" if sib else "")


# ------------------------------------------------------------------ lexical index
_TERM = re.compile(r"[a-z0-9]+")


def _terms(text):
    return _TERM.findall(text.lower())


class BM25:
    """Inverted-index BM25 over store texts (exact-string evidence: names, ids, words)."""

    def __init__(self, texts, k1=1.2, b=0.75):
        self.k1, self.b = k1, b
        self.post = collections.defaultdict(list)
        self.len = []
        for d, t in enumerate(texts):
            ts = _terms(t)
            self.len.append(len(ts))
            for term, f in collections.Counter(ts).items():
                self.post[term].append((d, f))
        self.N = len(texts)
        self.avg = sum(self.len) / max(1, self.N)

    def scores(self, query):
        import math
        sc = collections.defaultdict(float)
        for term in set(_terms(query)):
            plist = self.post.get(term)
            if not plist:
                continue
            idf = math.log(1 + (self.N - len(plist) + 0.5) / (len(plist) + 0.5))
            for d, f in plist:
                sc[d] += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[d] / self.avg))
        return sc

    def topk(self, query, k):
        import heapq
        return heapq.nlargest(k, self.scores(query).items(), key=lambda x: x[1])


# ------------------------------------------------------------------ module
class LTMHeads(nn.Module):
    def __init__(self, D, kdim=512, pdim=256, reader="bilinear"):
        super().__init__()
        self.D = D
        self.reader = XReader(D) if reader == "xattn" else None
        self.q_ret, self.k_ret = nn.Linear(D, kdim), nn.Linear(D, kdim)
        self.tau = nn.Parameter(torch.tensor(3.0))
        self.lex = nn.Parameter(torch.tensor(1.0))          # weight of normalized BM25 in the rerank
        self.sq, self.sk = nn.Linear(D, pdim), nn.Linear(D, pdim)
        self.eq, self.ek = nn.Linear(D, pdim), nn.Linear(D, pdim)
        self.match = nn.Embedding(MATCH_CAP + 1, 1)          # start bias per copy-continuation length
        nn.init.zeros_(self.match.weight)
        self.match_top = nn.Parameter(torch.tensor(0.0))     # bonus where the run is the longest in the prefix
        self.mgate = nn.Linear(D, 1)                         # per-question gate on the copy cue (2*sigmoid, init 1)
        nn.init.zeros_(self.mgate.weight)
        nn.init.zeros_(self.mgate.bias)
        self.rbias = nn.Parameter(torch.tensor(1.0))         # weight of the item's retrieval log-prob in the span score
        # recurrent read (LOTUS-style): latent = h + beta * (h - h_prev) at the "A:" state
        self.beta = nn.Parameter(torch.tensor(0.5))
        self.qh = nn.Linear(D, kdim)                         # latent -> re-query direction (init 0: text query only)
        nn.init.zeros_(self.qh.weight)
        nn.init.zeros_(self.qh.bias)
        self.halt = nn.Linear(D, 1)                          # P(copied span is the final answer)
        self.carry = nn.Parameter(torch.tensor(0.0))         # carried evidence logit, relative to the best new item
        # Walk prior: trajectory key = proj([latent, velocity]); a signed gate (tanh, init 0 = off)
        # leans the re-query toward (+) or away from (-) where similar remembered trajectories went next
        self.wz = nn.Linear(2 * D, 256)
        self.wgate = nn.Linear(D, 1)
        nn.init.zeros_(self.wgate.weight)
        nn.init.zeros_(self.wgate.bias)
        self.wtau = nn.Parameter(torch.tensor(2.3))         # log temperature of the bank match
        self.wscale = nn.Parameter(torch.tensor(0.5))


class XReader(nn.Module):
    """Trainable span reader over (frozen) memory-token states: memory self-attention (relations
    between notes: "the value whose note mentions the entity just found") + cross-attention to the
    question tokens and the "A:" state, then start / end scores per memory token. Replaces the single
    bilinear q.k score, which can only express "this token matches the question" (echo)."""

    def __init__(self, D, d=256, heads=4, layers=2, max_rank=32):
        super().__init__()
        self.d, self.max_rank = d, max_rank
        self.inp, self.qin = nn.Linear(D, d), nn.Linear(D, d)
        self.carry_emb = nn.Embedding(2, d)
        self.rank_emb = nn.Embedding(max_rank + 1, d)
        self.lp_proj = nn.Linear(1, d)
        self.blocks = nn.ModuleList()
        for _ in range(layers):
            self.blocks.append(nn.ModuleDict({
                "n1": nn.LayerNorm(d), "self": nn.MultiheadAttention(d, heads, batch_first=True),
                "n2": nn.LayerNorm(d), "cross": nn.MultiheadAttention(d, heads, batch_first=True),
                "n3": nn.LayerNorm(d), "ff": nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))}))
        self.norm = nn.LayerNorm(d)
        self.qs, self.qe = nn.Linear(d, d), nn.Linear(d, d)
        self.bs, self.be = nn.Linear(d, 1), nn.Linear(d, 1)
        for lin in (self.bs, self.be):                       # start near zero: cue + retrieval bias lead early
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)

    def forward(self, hp, valid, hQ, qmask, hq, carried, rank, lp):
        x = (self.inp(hp) + self.carry_emb(carried.long()) + self.rank_emb(rank.clamp(0, self.max_rank))
             + self.lp_proj(lp[..., None]))
        kv = self.qin(torch.cat([hQ, hq[:, None]], 1))
        kmask = torch.cat([qmask, torch.ones_like(qmask[:, :1])], 1)
        for blk in self.blocks:
            y = blk["n1"](x)
            x = x + blk["self"](y, y, y, key_padding_mask=~valid, need_weights=False)[0]
            y = blk["n2"](x)
            x = x + blk["cross"](y, kv, kv, key_padding_mask=~kmask, need_weights=False)[0]
            x = x + blk["ff"](blk["n3"](x))
        x = self.norm(x)
        q = self.qin(hq)
        sc = 1.0 / self.d ** 0.5
        s = (x * self.qs(q)[:, None]).sum(-1) * sc + self.bs(x).squeeze(-1)
        e = (x * self.qe(q)[:, None]).sum(-1) * sc + self.be(x).squeeze(-1)
        return s, e


class WalkBank:
    """Episodic memory of read trajectories: key = trajectory state (unit vector), value = unit
    direction of the step that followed on a GOLD chain. FIFO; read = softmax over the top-m."""

    def __init__(self, size=4096, topm=32):
        self.size, self.topm, self.k, self.v = size, topm, None, None

    def add(self, k, v):
        k, v = k.detach(), F.normalize(v.detach().float(), dim=-1)
        self.k = k if self.k is None else torch.cat([self.k, k])[-self.size:]
        self.v = v if self.v is None else torch.cat([self.v, v])[-self.size:]

    def read(self, q, log_tau, D):
        """Returns (blended next-step direction [B, D], best match cosine [B] or None)."""
        if self.k is None:
            return torch.zeros(q.shape[0], D, device=q.device), None
        sim = q @ self.k.T
        top = sim.topk(min(self.topm, sim.shape[1]), dim=-1)
        w = F.softmax(top.values * log_tau.exp(), -1)
        return (w[..., None] * self.v[top.indices]).sum(1), top.values[:, 0]


class _Stop(Exception):
    pass


class LTMRunner:
    def __init__(self, target, tok, heads, layer, qpos="before"):
        assert qpos in ("before", "after", "both")
        self.target, self.tok, self.heads, self.layer, self.qpos = target, tok, heads, layer, qpos
        self.dev = next(target.parameters()).device
        self.st = {"inj": None, "cap": False, "x": None, "h": None}
        self.walk = WalkBank()

    @contextlib.contextmanager
    def active(self):
        def pre(_m, args, kwargs):
            x = args[0] if args else kwargs["x"]
            if self.st["cap"]:
                self.st["x"] = x.detach()
                raise _Stop
            inj = self.st["inj"]
            if inj is None:
                return None
            mask, vals = inj
            y = torch.where(mask[..., None], vals.to(x.dtype), x)
            if args:
                return (y,) + tuple(args[1:]), kwargs
            kwargs = dict(kwargs)
            kwargs["x"] = y
            return args, kwargs

        def post(_m, _i, out):
            self.st["h"] = out[0] if isinstance(out, tuple) else out

        hooks = [self.target.blocks[self.layer].register_forward_pre_hook(pre, with_kwargs=True),
                 self.target.blocks[-1].register_forward_hook(post)]
        ckpt = getattr(self.target, "ckpt", None)
        if ckpt is not None:
            self.target.ckpt = False            # a recompute would replay the hook with cleared state
        try:
            yield self
        finally:
            for h in hooks:
                h.remove()
            if ckpt is not None:
                self.target.ckpt = ckpt

    def _fwd(self, ids):
        with torch.autocast(self.dev.type, dtype=torch.bfloat16, enabled=self.dev.type == "cuda"):
            return self.target(ids, r_active=False)

    def pad(self, seqs):
        n = max(len(s) for s in seqs)
        return torch.tensor([s + [self.tok.ph] * (n - len(s)) for s in seqs], device=self.dev)

    @torch.no_grad()
    def layer_states(self, seqs):
        """Walker's own residual states at the INPUT of block L, per sequence (unmodified)."""
        self.st["cap"] = True
        try:
            self._fwd(self.pad(seqs))
        except _Stop:
            pass
        finally:
            self.st["cap"] = False
        x = self.st["x"]
        return [x[i, :len(s)] for i, s in enumerate(seqs)]

    @torch.no_grad()
    def pooled(self, texts, bs=128):
        """Mean of LayerNormed final Walker states per text (retrieval features)."""
        out = []
        for s in range(0, len(texts), bs):
            seqs = [self.tok.ids(" " + t)[:96] for t in texts[s:s + bs]]
            ids = self.pad(seqs)
            self._fwd(ids)
            h = self.st["h"].float()
            h = F.layer_norm(h, h.shape[-1:])
            m = torch.zeros(ids.shape, device=self.dev)
            for i, q in enumerate(seqs):
                m[i, :len(q)] = 1
            out.append(((h * m[..., None]).sum(1) / m.sum(1, keepdim=True)).half())
        return torch.cat(out)

    def word_starts(self, ids):
        """True where a token begins a word: not an alphanumeric piece glued to an
        alphanumeric piece before it (BPE continuation)."""
        cache = self.st.setdefault("piece", {})
        pieces = []
        for i in ids:
            if i not in cache:
                cache[i] = self.tok.tok.decode([i])          # raw piece: keeps its leading space
            pieces.append(cache[i])
        return [t == 0 or not (pieces[t][:1].isalnum() and pieces[t - 1][-1:].isalnum())
                for t in range(len(pieces))]

    def keys(self, pooled):
        return F.normalize(self.heads.k_ret(pooled.float()), dim=-1)

    def queries(self, pooled, lat=None):
        q = self.heads.q_ret(pooled.float())
        if lat is not None:                                  # re-query: add what the last pass read
            q = q + self.heads.qh(lat)
        return F.normalize(q, dim=-1)

    # ---------------------------------------------------------------- rich read + span pointer
    def read(self, items_per_q, questions, gold=None, item_logits=None, ban=None, n_carry=None):
        """items_per_q: list of MemItem lists (prefix order). gold: list of (item index, answer str) or None.
        item_logits: optional per-question tensors of retrieval logits (one per item); their
        log-softmax biases both span ends inside that item (retrieve-then-read).
        ban: optional per-question lists of spans already copied in earlier passes; their exact
        occurrences cannot start a span again (a recurrent read never repeats itself).
        Returns (span CE or None, predicted answers, n_found)."""
        hd = self.heads
        node_ids = [[self.tok.ids(" " + it.text + "\n")[:96] for it in its] for its in items_per_q]
        flat = [s for row in node_ids for s in row]
        states = self.layer_states(flat)
        P = [sum(len(s) for s in row) for row in node_ids]
        Pm = max(P)
        # question placement (causal LM): "before" = [Q][memory][A:] so memory tokens' block>=L states
        # attend to the question; "after" = [memory][Q A:] (memory never sees the question)
        if self.qpos == "after":
            heads_ = [[] for _ in questions]
            prompts = [self.tok.ids(f"\nQ: {q}\nA:") for q in questions]
        else:
            heads_ = [self.tok.ids(f"Q: {q}\n") for q in questions]
            prompts = [self.tok.ids(f"\nQ: {q}\nA:") if self.qpos == "both" else self.tok.ids("\nA:")
                       for q in questions]
        off = [len(x) for x in heads_]                       # where each row's memory starts
        seqs = [heads_[b] + [self.tok.ph] * Pm + prompts[b] for b in range(len(questions))]
        ids = self.pad(seqs)
        B, Tn = ids.shape
        mask = torch.zeros(B, Tn, dtype=torch.bool, device=self.dev)
        vals = torch.zeros(B, Tn, self.heads.D, device=self.dev)
        pref_tok = []
        j = 0
        for b, row in enumerate(node_ids):
            pos, toks = off[b], []
            for s in row:
                vals[b, pos:pos + len(s)] = states[j].float()
                toks += s
                pos += len(s)
                j += 1
            mask[b, off[b]:pos] = True
            pref_tok.append(toks)
        self.st["inj"] = (mask, vals)
        try:
            self._fwd(ids)
        finally:
            self.st["inj"] = None
        h = F.layer_norm(self.st["h"].float(), (self.heads.D,))
        ar = torch.arange(B, device=self.dev)
        last = torch.tensor([off[b] + Pm + len(prompts[b]) - 1 for b in range(B)], device=self.dev)
        hq = h[ar, last]
        mem_idx = torch.tensor(off, device=self.dev)[:, None] + torch.arange(Pm, device=self.dev)[None]
        hp = h[ar[:, None], mem_idx]                         # memory positions, row-aligned to 0..Pm-1
        self.st["hp"] = hp
        sc = 1.0 / hd.sq.out_features ** 0.5
        bil_s = torch.einsum("bw,btw->bt", hd.sq(hq), hd.sk(hp)) * sc
        bil_e = torch.einsum("bw,btw->bt", hd.eq(hq), hd.ek(hp)) * sc
        s_start, s_end = bil_s, bil_e
        valid = torch.zeros(B, Pm, dtype=torch.bool, device=self.dev)
        wstart = torch.zeros(B, Pm, dtype=torch.bool, device=self.dev)
        wend = torch.zeros(B, Pm, dtype=torch.bool, device=self.dev)
        mlen = torch.zeros(B, Pm, dtype=torch.long, device=self.dev)
        for b in range(B):
            valid[b, :P[b]] = True
            ws = self.word_starts(pref_tok[b])
            wstart[b, :P[b]] = torch.tensor(ws, device=self.dev)
            wend[b, :P[b]] = torch.tensor(ws[1:] + [True], device=self.dev)
            q_ids = self.tok.ids(" " + questions[b])
            mlen[b, :P[b]] = torch.tensor(match_lengths(pref_tok[b], q_ids), device=self.dev)
        top = (mlen == mlen.max(1, keepdim=True).values) & (mlen >= 2)
        cue = hd.match(mlen).squeeze(-1) + hd.match_top * top.float()
        cue_t = 2 * torch.sigmoid(hd.mgate(hq)) * cue
        s_start = s_start + cue_t
        ib = torch.zeros_like(bil_s)
        if item_logits is not None:
            rows = []
            for b, row in enumerate(node_ids):
                lp = F.log_softmax(item_logits[b].float(), -1)
                seg = torch.repeat_interleave(lp, torch.tensor([len(s) for s in row], device=self.dev))
                rows.append(F.pad(seg, (0, Pm - seg.shape[0])))
            lp_rows = torch.stack(rows)
            ib = hd.rbias * lp_rows
            s_start, s_end = s_start + ib, s_end + ib
        if hd.reader is not None:                    # trainable reader replaces the single bilinear score
            lp_pos = lp_rows.detach() if item_logits is not None else torch.zeros_like(bil_s)
            carried = torch.zeros(B, Pm, dtype=torch.bool, device=self.dev)
            rank = torch.zeros(B, Pm, dtype=torch.long, device=self.dev)
            for b, row in enumerate(node_ids):
                pos = 0
                for r_i, s in enumerate(row):
                    rank[b, pos:pos + len(s)] = r_i
                    carried[b, pos:pos + len(s)] = r_i < (n_carry[b] if n_carry else 0)
                    pos += len(s)
            q_pos = [list(range(off[b])) + list(range(off[b] + Pm, off[b] + Pm + len(prompts[b]) - 1))
                     for b in range(B)]
            Lq = max(1, max(len(x) for x in q_pos))
            q_idx = torch.zeros(B, Lq, dtype=torch.long, device=self.dev)
            qmask = torch.zeros(B, Lq, dtype=torch.bool, device=self.dev)
            for b, x in enumerate(q_pos):
                if x:
                    q_idx[b, :len(x)] = torch.tensor(x, device=self.dev)
                    qmask[b, :len(x)] = True
            rd_s, rd_e = hd.reader(hp, valid, h[ar[:, None], q_idx], qmask, hq, carried, rank, lp_pos)
            bil_s, bil_e = rd_s, rd_e
            s_start, s_end = rd_s + cue_t + ib, rd_e + ib
        for b, spans in enumerate(ban or []):
            for sp in spans:
                s_ids = self.tok.ids(" " + sp)
                row = pref_tok[b]
                for t in range(len(row) - len(s_ids) + 1):
                    if row[t:t + len(s_ids)] == s_ids:
                        wstart[b, t] = False
        gold_pos = []
        if gold is not None:                         # locate gold spans first: they stay allowed below
            for b, (gi, ans) in enumerate(gold):
                if gi is None or gi < 0:
                    continue
                start = sum(len(s) for s in node_ids[b][:gi])
                seg = node_ids[b][gi]
                a_ids = self.tok.ids(" " + ans)
                k = next((i for i in range(len(seg) - len(a_ids) + 1) if seg[i:i + len(a_ids)] == a_ids), -1)
                if k < 0 or len(a_ids) > MAX_SPAN:
                    continue
                gold_pos.append((b, start + k, start + k + len(a_ids) - 1))
                wstart[b, start + k] = True
                wend[b, start + k + len(a_ids) - 1] = True
        # spans start and end on word boundaries (never "chester" out of "Westchester")
        s_start = s_start.masked_fill(~(valid & wstart), float("-inf"))
        s_end = s_end.masked_fill(~(valid & wend), float("-inf"))
        off = torch.arange(Pm, device=self.dev)
        allowed = ((off[None, :] - off[:, None]) >= 0) & ((off[None, :] - off[:, None]) < MAX_SPAN)
        pair = (s_start[:, :, None] + s_end[:, None, :]).masked_fill(~allowed, float("-inf"))
        flat_arg = pair.flatten(1).argmax(-1).tolist()
        preds = [self.tok.decode(pref_tok[b][a // Pm:a % Pm + 1]) for b, a in enumerate(flat_arg)]
        ends = [list(itertools.accumulate(len(s) for s in row)) for row in node_ids]
        if self.st.get("debug"):                     # diagnosis: per-row score terms (start side + end side)
            gp = {b: (s, e) for b, s, e in gold_pos}
            self.st["parts_row"] = [{
                "bil_s": bil_s[b, :P[b]].detach().float().cpu(), "bil_e": bil_e[b, :P[b]].detach().float().cpu(),
                "cue": cue_t[b, :P[b]].detach().float().cpu(), "ib": ib[b, :P[b]].detach().float().cpu(),
                "start": a // Pm, "end": a % Pm, "gold": gp.get(b), "ends": ends[b],
                "ok_s": (valid & wstart)[b, :P[b]].cpu(), "ok_e": (valid & wend)[b, :P[b]].cpu()}
                for b, a in enumerate(flat_arg)]
        self.st["hq"] = hq                                   # read state at "A:" (LOTUS latent input)
        self.st["pred_item"] = [bisect.bisect_right(ends[b], a // Pm) for b, a in enumerate(flat_arg)]
        ce, found = None, 0
        if gold is not None:
            rows = [b for b, _, _ in gold_pos]
            gs = [s for _, s, _ in gold_pos]
            ge = [e for _, _, e in gold_pos]
            found = len(rows)
            if rows:
                ri = torch.tensor(rows, device=self.dev)
                gs_t = torch.tensor(gs, device=self.dev)
                ge_t = torch.tensor(ge, device=self.dev)
                ce = F.cross_entropy(s_start[ri], gs_t) + F.cross_entropy(
                    s_end[ri].masked_fill(~allowed[gs_t], float("-inf")), ge_t)
        return ce, preds, found


# ------------------------------------------------------------------ tasks
def _norm(s):
    return re.sub(r"\s+", " ", s.strip(" .\"'").lower())


def build_wikisql(tok, split, n_tables):
    tables = T.gmem_wikisql_tables(split, n_tables, tok.fits)
    store, queries = [], []
    for t in tables:
        g = Graph(t["leaves"], tok)
        base = len(store)
        store += [MemItem(node_text(n), n) for n in g.nodes]
        queries.append({"store": 0, "gold": base + t["gold"], "q": t["q"], "answer": t["answer"]})
    return [store], queries


def build_sentences(tok, split, n, seed):
    from datasets import load_dataset
    sents = []
    # streamed, row by row (a row is one paragraph, so sentences never span a heading); stop early
    for row in load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split=split, streaming=True):
        line = " ".join(row["text"].split())                      # no raw newlines inside a stored sentence
        if not line or line.startswith("="):
            continue
        line = line.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".")
        for s in re.split(r"(?<=[a-z])\. ", line):
            s = s.strip().rstrip(".") + "."
            if "@" not in s and "=" not in s and 14 <= len(tok.ids(" " + s)) <= 40 and len(s.split()) >= 9:
                sents.append(s)
        if len(sents) >= 2 * n:
            break
    random.Random(seed).shuffle(sents)
    store, queries = [], []
    for s in sents:
        w = s.split()
        if len(w) < 9:
            continue
        head, rest = " ".join(w[:6]), " ".join(w[6:])
        store.append(MemItem(f"note: {s}"))
        queries.append({"store": 0, "gold": len(store) - 1,
                        "q": f'Recall the rest of the note that begins "{head}".', "answer": rest})
        if len(store) >= n:
            break
    return [store], queries


def build_json(tok, data, n_eps, rng):
    stores, queries = [], []
    while len(stores) < n_eps:
        e = data._synthetic(rng, T.GMEM_TRAIN_KINDS)
        g = Graph(e["leaves"], tok)
        olds = []
        for k, w in enumerate(e["writes"], 1):
            n = g.nodes[w["node"]]
            olds.append((w["node"], n.value, k))
            g.write(w["node"], tok.ids(" " + w["value"])[:T.GMEM_V])
        store = [MemItem(node_text(n), n) for n in g.nodes]
        for i, old, k in olds:
            store.append(MemItem(history_text(g.nodes[i], old, k), g.nodes[i], version=k))
        sid = len(stores)
        stores.append(store)
        for q in e["questions"]:
            queries.append({"store": sid, "gold": q["node"], "q": q["q"], "answer": q["answer"], "tag": q["tag"]})
        for i, old, k in olds:
            subj = T._gmem_subject(e["leaves"][i])
            queries.append({"store": sid, "gold": len(g.nodes) + k - 1,
                            "q": f"What was the {subj} before update {k}?", "answer": old, "tag": "history"})
    return stores, queries


_FILES = "abcdefgh"
_PIECES = ("king", "queen", "rook", "bishop", "knight", "pawn")


def build_chess(tok, n_boards, rng):
    """Board = 64 square nodes + side to move; moves change two squares (history kept)."""
    stores, queries = [], []
    for _ in range(n_boards):
        squares = [f + str(r) for f in _FILES for r in range(1, 9)]
        board = {s: "empty" for s in squares}
        occ = rng.sample(squares, rng.randint(10, 24))
        board[occ[0]], board[occ[1]] = "white king", "black king"
        for s in occ[2:]:
            board[s] = f"{rng.choice(['white', 'black'])} {rng.choice(_PIECES[1:])}"
        state = {"square": dict(board), "side to move": rng.choice(["white", "black"])}
        g = Graph(T.gmem_leaves(state), tok)
        idx = {n.path[-1]: i for i, n in enumerate(g.nodes) if n.path[0] == "square"}
        hist = []
        for k in range(1, rng.randint(1, 3) + 1):
            src = rng.choice([s for s in squares if board[s] != "empty" and "king" not in board[s]] or squares)
            dst = rng.choice([s for s in squares if board[s] == "empty"])
            piece = board[src]
            for s, new in ((dst, piece), (src, "empty")):
                hist.append((idx[s], board[s], k))
                board[s] = new
                g.write(idx[s], tok.ids(" " + new))
        store = [LT_item(n) for n in g.nodes]
        for i, old, k in hist:
            store.append(MemItem(history_text(g.nodes[i], old, k), g.nodes[i], version=k))
        sid = len(stores)
        stores.append(store)
        for s in rng.sample(squares, 3):
            queries.append({"store": sid, "gold": idx[s], "q": f"What is on {s}?", "answer": board[s], "tag": "square"})
        king = next(s for s in squares if board[s] == "white king")
        queries.append({"store": sid, "gold": idx[king], "q": "On which square is the white king?",
                        "answer": king, "tag": "where"})
        i, old, k = hist[0]
        queries.append({"store": sid, "gold": len(g.nodes) + 0, "q": f"What was on {g.nodes[i].path[-1]} before update {k}?",
                        "answer": old, "tag": "history"})
    return stores, queries


def LT_item(n):
    return MemItem(node_text(n), n)


_LABELS = ("OK", "Cancel", "Apply", "Save", "Open", "Delete", "Retry", "Help", "Next", "Back")
_FIELDS = ("File name", "Folder", "Email", "Username", "Search", "Title", "Tags")
_DIALOGS = ("Save", "Open", "Settings", "Export", "Login", "Print")


def build_gui(tok, n_screens, rng):
    """Accessibility tree: dialogs with buttons / text fields / checkboxes; events change state."""
    stores, queries = [], []
    for _ in range(n_screens):
        dialog = rng.choice(_DIALOGS)
        btns = [{"label": b, "enabled": rng.choice(["true", "false"])} for b in rng.sample(_LABELS, rng.randint(2, 4))]
        flds = [{"label": f, "value": rng.choice(["report.txt", "notes.md", "home", "ava@mail.com", "draft",
                                                  "q3 plan", "photos"])} for f in rng.sample(_FIELDS, rng.randint(1, 3))]
        boxes = [{"label": c, "checked": rng.choice(["true", "false"])}
                 for c in rng.sample(["Remember me", "Overwrite", "Open after export", "Dark mode"], 2)]
        state = {"window": {"title": f"{dialog} dialog", "buttons": btns, "fields": flds, "checkboxes": boxes}}
        g = Graph(T.gmem_leaves(state), tok)
        hist = []
        bi = rng.randrange(len(btns))
        node = next(i for i, n in enumerate(g.nodes) if n.path[-3:] == ("buttons", str(bi), "enabled"))
        new = "false" if btns[bi]["enabled"] == "true" else "true"
        hist.append((node, btns[bi]["enabled"], 1))
        btns[bi]["enabled"] = new
        g.write(node, tok.ids(" " + new))
        fi = rng.randrange(len(flds))
        fnode = next(i for i, n in enumerate(g.nodes) if n.path[-3:] == ("fields", str(fi), "value"))
        newv = rng.choice(["final.pdf", "backup", "zoe@mail.com", "music", "todo list"])
        hist.append((fnode, flds[fi]["value"], 2))
        flds[fi]["value"] = newv
        g.write(fnode, tok.ids(" " + newv))
        store = [LT_item(n) for n in g.nodes]
        for i, old, k in hist:
            store.append(MemItem(history_text(g.nodes[i], old, k), g.nodes[i], version=k))
        sid = len(stores)
        stores.append(store)
        queries.append({"store": sid, "gold": node, "tag": "button",
                        "q": f"Is the {btns[bi]['label']} button in the {dialog} dialog enabled?", "answer": new})
        queries.append({"store": sid, "gold": fnode, "tag": "field",
                        "q": f"What is in the {flds[fi]['label']} field?", "answer": newv})
        queries.append({"store": sid, "gold": len(g.nodes) + 1, "tag": "history",
                        "q": f"What was in the {flds[fi]['label']} field before update 2?",
                        "answer": hist[1][1]})
    return stores, queries


_SYL = ("ka", "lo", "mer", "vin", "tor", "sa", "ne", "ril", "dun", "ba", "zek", "or", "pel", "mu", "tha", "gor")
_ATTRS = (("airport code", "{c} has the airport code {v}.", "code"),
          ("river", "The river of {c} is the {v}.", "river"),
          ("mayor", "The mayor of {c} is {v}.", "person"))


def _name(rng, n=2, used=None):
    while True:
        w = "".join(rng.choice(_SYL) for _ in range(n)).capitalize()
        if used is None or w not in used:
            if used is not None:
                used.add(w)
            return w


def build_bridge(tok, n_eps, rng, n_people=30, n_comp=10, n_city=8):
    """Multi-hop facts (fictional, so Walker cannot know them): person -> employer -> city ->
    attribute, plus home city. Hop k's evidence names the entity hop k+1 must look up, which is
    NOT in the question, so it must be read and re-queried. 1-, 2- and 3-hop questions."""
    stores, queries = [], []
    for _ in range(n_eps):
        used, store, at = set(), [], {}

        def add(text, key):
            at[key] = len(store)
            store.append(MemItem(f"note: {text}"))

        cities = [_name(rng, 3, used) for _ in range(n_city)]
        comps = [_name(rng, 2, used) + rng.choice(["ex", "ix", " Ltd", " Labs"]) for _ in range(n_comp)]
        attr = {}
        for c in cities:
            for a, tmpl, kind in _ATTRS:
                v = ("".join(rng.choice("BCDFGHKLMNPRSTVZ") for _ in range(3)) if kind == "code"
                     else _name(rng, 2, used) if kind == "river"
                     else f"{_name(rng, 2, used)} {_name(rng, 2, used)}")
                attr[c, a] = v
                add(tmpl.format(c=c, v=v), (c, a))
        base = {}
        for co in comps:
            base[co] = rng.choice(cities)
            add(f"{co} is based in {base[co]}.", ("base", co))
        sid = len(stores)
        for _ in range(n_people):
            p = f"{_name(rng, 2, used)} {_name(rng, 2, used)}"
            home, job = rng.choice(cities), rng.choice(comps)
            add(f"{p} lives in {home}.", ("home", p))
            add(f"{p} works for {job}.", ("job", p))
            a, _, _ = rng.choice(_ATTRS)
            hl, jb = (at["home", p], home), (at["job", p], job)
            qs = [("hop1", f"Where does {p} live?", [hl]),
                  ("hop2", f"What is the {a} of the city where {p} lives?", [hl, (at[home, a], attr[home, a])]),
                  ("hop3", f"What is the {a} of the city where the employer of {p} is based?",
                   [jb, (at["base", job], base[job]), (at[base[job], a], attr[base[job], a])])]
            for tag, q, hops in rng.sample(qs, 2):
                queries.append({"store": sid, "gold": hops[-1][0], "q": q, "answer": hops[-1][1],
                                "hops": hops, "tag": tag})
        stores.append(store)
    return stores, queries


class Task:
    def __init__(self, name, rn, stores, queries, n_cand=64, passes=3):
        self.name, self.rn, self.stores, self.queries, self.n_cand = name, rn, stores, queries, n_cand
        self.passes = passes
        self.walk_mode = 1.0         # eval ablation: 0 = Walk prior off, -1 = flipped
        self.multi = any(len(q.get("hops", ())) > 1 for q in queries)
        self.pooled = [rn.pooled([it.text for it in st]) for st in stores]
        self.bm25 = [BM25([it.text for it in st]) for st in stores]
        self.bm25_k = False          # eval: also report BM25-only recall (baseline)
        self.misses = []             # eval: a few wrong answers for diagnosis
        self.trace = None            # diagnosis: list -> one record per (query, pass)
        self.oracle = frozenset()    # diagnosis, eval only: {"retrieve", "halt", "requery"} use gold

    @torch.no_grad()
    def refresh_keys(self):
        """Recompute store retrieval features with the current Walker (needed once Walker trains)."""
        self.pooled = [self.rn.pooled([it.text for it in st]) for st in self.stores]

    @staticmethod
    def chain(q):
        """Gold hop chain [(item id, span to copy)]; the last span is the answer."""
        return q.get("hops") or [(q["gold"], q["answer"])]

    def retrieve(self, sid, text, qv, gold, k, train, st):
        """Hybrid retrieval for one query: BM25 candidates (whole store when small), dense +
        lexical rerank. Returns (CE or None, top-k item ids, their logits)."""
        rn = self.rn
        n_items = len(self.stores[sid])
        if n_items <= self.n_cand:
            lex = self.bm25[sid].scores(text)
            cand = list(range(n_items))
        else:
            lex = dict(self.bm25[sid].topk(text, self.n_cand))
            cand = list(lex) or list(range(min(n_items, self.n_cand)))
        if gold is not None:
            st["cand_recall"].append(float(gold in cand))
            if train and gold not in cand:
                cand.append(gold)
        ci = torch.tensor(cand, device=rn.dev)
        lx = torch.tensor([lex.get(c, 0.0) for c in cand], device=rn.dev)
        lx = lx / lx.max().clamp(min=1e-6)
        logit = (rn.keys(self.pooled[sid][ci]) @ qv) * rn.heads.tau.exp() + rn.heads.lex * 10.0 * lx
        ce = None
        if gold is not None and gold in cand:
            ce = F.cross_entropy(logit[None], torch.tensor([cand.index(gold)], device=rn.dev))
        tj = logit.topk(min(k, len(cand))).indices.tolist()
        top = [cand[j] for j in tj]
        if gold is not None:
            st["r@1"].append(float(top[0] == gold))
            st[f"r@{k}"].append(float(gold in top))
            if self.bm25_k:
                st[f"bm25_r@{k}"].append(float(gold in [d for d, _ in self.bm25[sid].topk(text, k)]))
            if train and gold not in top:
                tj[-1] = cand.index(gold)
                top[-1] = gold
        return ce, top, logit[tj]

    def run(self, qs, k, train, passes=None):
        """Recurrent read, LOTUS-style. Pass r: retrieve with the current query (question text
        + the span copied in pass r-1, plus the pass r-1 latent via `qh`), read the carried
        evidence + top-k with rich injection, copy a span, then HALT (span = answer) or
        RE-QUERY with it. Latent = h + beta*(h - h_prev) at the "A:" state (Walk momentum).
        Training is teacher-forced along the gold hop chain; eval follows its own spans."""
        rn, hd = self.rn, self.rn.heads
        R = passes or self.passes
        n = len(qs)
        chains = [self.chain(q) for q in qs]
        text = [q["q"] for q in qs]
        carry = [[] for _ in qs]                     # evidence item ids read in earlier passes
        seen = [[] for _ in qs]                      # spans copied in earlier passes
        h_prev, lat, lat_raw, z_prev, final = {}, {}, {}, {}, {}
        live = list(range(n))
        losses, st = [], collections.defaultdict(list)
        for r in range(R):
            if train:
                live = [i for i in range(n) if r < len(chains[i])]
            if not live:
                break
            qv = rn.queries(rn.pooled([text[i] for i in live]),
                            torch.stack([lat[i] for i in live]) if r else None)
            items, gold, logits, pres = [], [], [], []
            for j, i in enumerate(live):
                g_item, g_span = chains[i][r] if r < len(chains[i]) else (None, None)
                ce, top, lg = self.retrieve(qs[i]["store"], text[i], qv[j], g_item, k,
                                            train or "retrieve" in self.oracle, st)
                if ce is not None:
                    losses.append(ce / n)
                new = [x for x, t in enumerate(top) if t not in carry[i]]
                pre = carry[i] + [top[x] for x in new]
                logits.append(torch.cat([(lg.max().detach() + hd.carry).expand(len(carry[i])), lg[new]]))
                pres.append(pre)
                items.append([self.stores[qs[i]["store"]][x] for x in pre])
                gold.append((pre.index(g_item) if g_item in pre else -1, g_span))
            # the reader sees what earlier passes found, and may not copy it again
            asks = [qs[i]["q"] + (f" [found: {', '.join(seen[i])}]" if seen[i] else "") for i in live]
            ce, preds, found = rn.read(items, asks, gold, item_logits=logits, ban=[seen[i] for i in live],
                                       n_carry=[len(carry[i]) for i in live])
            if ce is not None:
                losses.append(ce * len(live) / n)
            st["span_found"].append(found / len(live))
            hq = rn.st["hq"]
            prev = torch.stack([h_prev.get(i, hq[j]) for j, i in enumerate(live)])
            lat_r = hq + hd.beta * (hq - prev)
            halt = hd.halt(lat_r).squeeze(-1)
            # Walk prior: key the trajectory (latent + velocity); on gold chains remember the step taken
            # from the previous state; then lean the next re-query toward (gate>0) / away from (<0) it
            z = F.normalize(hd.wz(torch.cat([lat_r, hq - prev], -1)), dim=-1)
            if train and r > 0:
                rn.walk.add(torch.stack([z_prev[i] for i in live]),
                            lat_r - torch.stack([lat_raw[i] for i in live]))
            push, best = rn.walk.read(z, hd.wtau, hd.D)
            g = torch.tanh(hd.wgate(lat_r)).squeeze(-1) * self.walk_mode
            lat_q = lat_r + (g * hd.wscale)[:, None] * lat_r.norm(dim=-1, keepdim=True) * push
            if best is not None:
                st["walk_match"] += best.tolist()
                st["walk_gate"] += g.tolist()
            if train:
                tgt = torch.tensor([float(r == len(chains[i]) - 1) for i in live], device=rn.dev)
                losses.append(F.binary_cross_entropy_with_logits(halt, tgt) * len(live) / n)
                st["halt_acc"] += ((halt > 0).float() == tgt).float().tolist()
            if self.trace is not None:
                for j, i in enumerate(live):
                    g_item, g_span = chains[i][r] if r < len(chains[i]) else (None, None)
                    pi = rn.st["pred_item"][j]
                    self.trace.append({
                        "qi": i, "r": r, "tag": qs[i].get("tag"), "hops": len(chains[i]), "q": qs[i]["q"],
                        "gold_item": g_item, "gold_span": g_span, "prefix": list(pres[j]), "n_carry": len(carry[i]),
                        "gold_in_prefix": g_item in pres[j], "gold_new": g_item in pres[j][len(carry[i]):],
                        "pred": preds[j], "pred_item": pres[j][pi] if pi < len(pres[j]) else None,
                        "span_ok": g_span is not None and _norm(preds[j]) == _norm(g_span),
                        "halt_logit": float(halt[j]), "halt_target": r == len(chains[i]) - 1,
                        "lat": lat_r[j].detach().float().cpu(), "hq": hq[j].detach().float().cpu(),
                        "parts": rn.st["parts_row"][j] if rn.st.get("debug") else None})
            nxt = []
            for j, i in enumerate(live):
                h_prev[i], lat[i], lat_raw[i], z_prev[i] = hq[j], lat_q[j], lat_r[j].detach(), z[j]
                stop_now = (r == len(chains[i]) - 1) if "halt" in self.oracle else halt[j] > 0
                if not train and "requery" in self.oracle and r < len(chains[i]) and not (stop_now or r == R - 1):
                    g_item, g_span = chains[i][r]                    # oracle: follow the gold chain
                    text[i] = f"{qs[i]['q']} {g_span}"
                    carry[i] = carry[i] + [g_item]
                    seen[i] = seen[i] + [g_span]
                    nxt.append(i)
                    continue
                if train:
                    g_item, g_span = chains[i][r]
                    text[i] = f"{qs[i]['q']} {g_span}"
                    carry[i] = carry[i] + [g_item]
                    seen[i] = seen[i] + [g_span]
                    if r == len(chains[i]) - 1 or r == R - 1:        # chain longer than R: score the last pass
                        final[i] = (preds[j], r + 1, items[j])
                elif stop_now or r == R - 1:
                    final[i] = (preds[j], r + 1, items[j])
                else:
                    pi = rn.st["pred_item"][j]
                    text[i] = f"{qs[i]['q']} {preds[j]}"
                    carry[i] = carry[i] + ([pres[j][pi]] if pi < len(pres[j]) else [])
                    seen[i] = seen[i] + [preds[j]]
                    nxt.append(i)
            live = nxt
        for i, q in enumerate(qs):
            p, used, its = final[i]
            hit = float(_norm(p) == _norm(q["answer"]))
            st["em"].append(hit)
            st["passes"].append(float(used))
            if "tag" in q:
                st[f"em_{q['tag']}"].append(hit)
            if not hit and not train and len(self.misses) < 4:
                gi = self.stores[q["store"]][q["gold"]].text
                self.misses.append(f"q={q['q']!r} gold={q['answer']!r} pred={p!r} passes={used} "
                                   f"gold_item={gi!r} top1={its[0].text!r}")
        return sum(losses), st


class LMData:
    """Next-token windows from wikitext-103 (raw): LM episodes (train split) and LM health (validation)."""

    def __init__(self, raw_tok, split, n_tokens, seq, seed=0):
        from datasets import load_dataset
        ids = []
        for row in load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split=split, streaming=True):
            t = row["text"]
            if t.strip():
                ids += raw_tok(t, add_special_tokens=False)["input_ids"]
                if len(ids) >= n_tokens:
                    break
        self.ids = torch.tensor(ids[:n_tokens], dtype=torch.long)
        self.seq, self.rng = seq, random.Random(seed)

    def batch(self, B, dev):
        st = [self.rng.randrange(0, len(self.ids) - self.seq - 1) for _ in range(B)]
        x = torch.stack([self.ids[i:i + self.seq + 1] for i in st]).to(dev)
        return x[:, :-1], x[:, 1:]

    def windows(self, dev):
        n = (len(self.ids) - 1) // self.seq
        for i in range(n):
            x = self.ids[i * self.seq: i * self.seq + self.seq + 1].to(dev)[None]
            yield x[:, :-1], x[:, 1:]


def lm_ce(model, x, y):
    with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=x.device.type == "cuda"):
        model(x, y, r_active=False)
    return model.lm_ce


@torch.no_grad()
def lm_val(model, data, bs=8):
    tot, n = 0.0, 0
    xs, ys = [], []
    for x, y in data.windows(next(model.parameters()).device):
        xs.append(x)
        ys.append(y)
        if len(xs) == bs:
            tot += float(lm_ce(model, torch.cat(xs), torch.cat(ys))) * len(xs)
            n += len(xs)
            xs, ys = [], []
    return tot / max(1, n)


def refresh_quant(model):
    """QLinear keeps a bf16 master; its quantized cache must follow every optimizer step."""
    for m in model.modules():
        if isinstance(m, T.QLinear) and m.quant != "none":
            m._refresh_q()


def evaluate(tasks, k, step, bs=24):
    with torch.no_grad():
        for t in tasks:
            agg = collections.defaultdict(list)
            t.eval_task.bm25_k, t.eval_task.misses = True, []
            for s in range(0, len(t.eval_q), bs):
                _, st = t.eval_task.run(t.eval_q[s:s + bs], k, train=False)
                for kk, v in st.items():
                    agg[kk] += v
                if t.eval_task.multi:                # ablations with the same heads
                    for tag, passes, mode in (("1pass", 1, 1.0), ("nowalk", None, 0.0), ("flipwalk", None, -1.0)):
                        t.eval_task.walk_mode = mode
                        _, st1 = t.eval_task.run(t.eval_q[s:s + bs], k, train=False, passes=passes)
                        for kk, v in st1.items():
                            if kk.startswith("em"):
                                agg[f"{tag}_{kk}"] += v
                    t.eval_task.walk_mode = 1.0
            t.eval_task.bm25_k = False
            print(f"LTM_VAL step={step} task={t.name} store={sum(len(x) for x in t.eval_task.stores)} "
                  f"q={len(t.eval_q)} " + " ".join(f"{kk}={sum(v)/len(v):.4f}" for kk, v in agg.items()),
                  flush=True)
            for m in t.eval_task.misses:
                print(f"LTM_MISS step={step} task={t.name} {m}", flush=True)
        hd = tasks[0].rn.heads
        print(f"LTM_HEADS step={step} tau={hd.tau.exp().item():.2f} lex={hd.lex.item():.3f} "
              f"rbias={hd.rbias.item():.3f} match_top={hd.match_top.item():.3f} beta={hd.beta.item():.3f} "
              f"wscale={hd.wscale.item():.3f} walk_bank={0 if tasks[0].rn.walk.k is None else len(tasks[0].rn.walk.k)} "
              f"match={[round(v, 2) for v in hd.match.weight[:9, 0].tolist()]}", flush=True)


def main():
    P = argparse.ArgumentParser()
    P.add_argument("--ckpt", required=True)
    P.add_argument("--layer", type=int, default=2)
    P.add_argument("--k", type=int, default=8)
    P.add_argument("--steps", type=int, default=600)
    P.add_argument("--batch", type=int, default=8, help="queries per task per step")
    P.add_argument("--eval-every", type=int, default=200)
    P.add_argument("--tasks", default="wikisql,sentences,json,chess,gui,bridge")
    P.add_argument("--passes", type=int, default=3, help="max recurrent read passes (re-queries + 1)")
    P.add_argument("--qpos", default="before", choices=("before", "after", "both"),
                   help="question placement: before memory (memory sees it), after (old), or both")
    P.add_argument("--reader", default="bilinear", choices=("bilinear", "xattn"),
                   help="span scorer: single bilinear q.k (old) or the trainable cross-attention XReader")
    P.add_argument("--walker-train", default="none", choices=("none", "alt", "memonly"),
                   help="none = frozen Walker; alt = 2 episodes per interval (memory episode updates heads+Walker, "
                        "LM episode updates Walker on next-token loss) with SEPARATE optimizers; memonly = memory "
                        "episode only (control: LM damage without the LM episode)")
    P.add_argument("--mem-walker-lr", type=float, default=1e-5)
    P.add_argument("--lm-lr", type=float, default=2e-5)
    P.add_argument("--lm-batch", type=int, default=8)
    P.add_argument("--lm-seq", type=int, default=512)
    P.add_argument("--out", default="gmem_ltm")
    A = P.parse_args()
    os.makedirs(A.out, exist_ok=True)
    model, raw_tok, _ = T.build_model_for_inference(A.ckpt)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    tok = Tok(raw_tok)
    heads = LTMHeads(model.emb.weight.shape[1], reader=A.reader).to(next(model.parameters()).device)
    rn = LTMRunner(model, tok, heads, A.layer, qpos=A.qpos)
    t0 = time.time()
    tasks = []
    with rn.active():
        names = A.tasks.split(",")
        if "wikisql" in names:
            tr = Task("wikisql", rn, *build_wikisql(tok, "train", 600))
            ev = Task("wikisql", rn, *build_wikisql(tok, "validation", 300))
            tr.eval_task, tr.eval_q = ev, ev.queries
            tasks.append(tr)
        if "sentences" in names:
            tr = Task("sentences", rn, *build_sentences(tok, "train", 4000, 0))
            ev = Task("sentences", rn, *build_sentences(tok, "test", 1500, 1))
            tr.eval_task, tr.eval_q = ev, ev.queries[:300]
            tasks.append(tr)
        if "json" in names:
            data = T.GmemData(tok, ["synthetic"], 2, seed=1)
            tr = Task("json", rn, *build_json(tok, data, 600, random.Random(2)))
            ev = Task("json", rn, *build_json(tok, data, 100, random.Random(3)))
            tr.eval_task, tr.eval_q = ev, ev.queries
            tasks.append(tr)
        if "chess" in names:
            tr = Task("chess", rn, *build_chess(tok, 400, random.Random(4)))
            ev = Task("chess", rn, *build_chess(tok, 60, random.Random(5)))
            tr.eval_task, tr.eval_q = ev, ev.queries
            tasks.append(tr)
        if "gui" in names:
            tr = Task("gui", rn, *build_gui(tok, 600, random.Random(6)))
            ev = Task("gui", rn, *build_gui(tok, 100, random.Random(7)))
            tr.eval_task, tr.eval_q = ev, ev.queries
            tasks.append(tr)
        if "bridge" in names:
            tr = Task("bridge", rn, *build_bridge(tok, 60, random.Random(8)))
            ev = Task("bridge", rn, *build_bridge(tok, 10, random.Random(9)))
            tr.eval_task, tr.eval_q = ev, ev.queries[:300]
            tasks.append(tr)
        for t in tasks:
            t.passes = t.eval_task.passes = A.passes
        for t in tasks:
            print(f"LTM task={t.name} train_store={sum(len(s) for s in t.stores)} train_q={len(t.queries)} "
                  f"eval_store={sum(len(s) for s in t.eval_task.stores)} eval_q={len(t.eval_q)} "
                  f"built={time.time()-t0:.0f}s", flush=True)
        named = list(heads.named_parameters())
        opt = torch.optim.AdamW([{"params": [p for n, p in named if n in ("q_ret.weight", "q_ret.bias", "k_ret.weight",
                                                                          "k_ret.bias", "tau", "lex")], "lr": 1e-3},
                                 {"params": [p for n, p in named if n[:2] in ("sq", "sk", "eq", "ek")], "lr": 1e-3},
                                 {"params": [p for n, p in named
                                             if n.split(".")[0] in ("mgate", "qh", "halt", "wz", "wgate")],
                                  "lr": 1e-3},
                                 {"params": [heads.match.weight, heads.match_top, heads.rbias, heads.beta,
                                             heads.wtau, heads.wscale, heads.carry], "lr": 3e-2},  # scalars
                                 {"params": [p for n, p in named if n.startswith("reader.")], "lr": 5e-4}],
                                weight_decay=0.0)
        base_lr = [g["lr"] for g in opt.param_groups]
        in_opt = [id(p) for g in opt.param_groups for p in g["params"]]
        assert sorted(in_opt) == sorted(id(p) for p in heads.parameters()), "every head param in exactly one group"
        r = random.Random(0)
        opt_wmem = opt_wlm = lm_train = None
        if A.walker_train != "none":
            wparams = list(model.parameters())
            for p in wparams:
                p.requires_grad_(True)
            # separate Adam states per episode type: memory gradients never enter the LM optimizer's moments
            opt_wmem = T.CompensatedAdamW([{"params": wparams, "lr": A.mem_walker_lr, "weight_decay": 0.0}],
                                          betas=(0.9, 0.95))
            if A.walker_train == "alt":
                opt_wlm = T.CompensatedAdamW([{"params": wparams, "lr": A.lm_lr, "weight_decay": 0.0}],
                                             betas=(0.9, 0.95))
                lm_train = LMData(raw_tok, "train", 4_000_000, A.lm_seq, seed=0)
            print(f"LTM_WALKER mode={A.walker_train} params={sum(p.numel() for p in wparams)/1e6:.0f}M "
                  f"mem_lr={A.mem_walker_lr} lm_lr={A.lm_lr}", flush=True)
        lm_heldout = LMData(raw_tok, "validation", 64 * A.lm_seq + 1, A.lm_seq)
        lm_base = lm_val(model, lm_heldout)
        print(f"LTM_LM step=0 val_ce={lm_base:.4f}", flush=True)
        evaluate(tasks, A.k, 0)
        acc = collections.defaultdict(list)
        for step in range(1, A.steps + 1):
            decay = 0.1 + 0.45 * (1 + math.cos(math.pi * (step - 1) / A.steps))     # cosine to 10%
            for g, lr in zip(opt.param_groups, base_lr):
                g["lr"] = lr * decay
            loss = 0.0
            for t in tasks:
                l, st = t.run(r.sample(t.queries, A.batch), A.k, train=True)
                loss = loss + l
                for kk, v in st.items():
                    acc[f"{t.name}/{kk}"] += v
            loss.backward()
            for g in opt.param_groups:
                torch.nn.utils.clip_grad_norm_(g["params"], 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            if opt_wmem is not None:                      # episode 1 (memory): Walker update from the memory loss
                for g in opt_wmem.param_groups:
                    g["lr"] = A.mem_walker_lr * decay
                torch.nn.utils.clip_grad_norm_(opt_wmem.param_groups[0]["params"], 1.0)
                opt_wmem.step()
                opt_wmem.zero_grad(set_to_none=True)
                refresh_quant(model)
            if opt_wlm is not None:                       # episode 2 (LM): Walker update from next-token loss only
                for g in opt_wlm.param_groups:
                    g["lr"] = A.lm_lr * decay
                x, y = lm_train.batch(A.lm_batch, rn.dev)
                lce = lm_ce(model, x, y)
                lce.backward()
                torch.nn.utils.clip_grad_norm_(opt_wlm.param_groups[0]["params"], 1.0)
                opt_wlm.step()
                opt_wlm.zero_grad(set_to_none=True)
                heads.zero_grad(set_to_none=True)
                refresh_quant(model)
                acc["lm/train_ce"].append(float(lce.detach()))
            if step % 25 == 0:
                print(f"LTM step={step} loss={float(loss.detach()):.4f} " + " ".join(
                    f"{kk}={sum(v)/len(v):.3f}" for kk, v in sorted(acc.items())
                    if "/em" in kk or "/r@" in kk or kk.startswith("lm/"))
                    + f" elapsed={time.time()-t0:.0f}s", flush=True)
                acc = collections.defaultdict(list)
            if step % A.eval_every == 0 or step == A.steps:
                if opt_wmem is not None:
                    for t in tasks:
                        t.eval_task.refresh_keys()
                cur = lm_val(model, lm_heldout)
                print(f"LTM_LM step={step} val_ce={cur:.4f} delta_vs_frozen={cur - lm_base:+.4f}", flush=True)
                evaluate(tasks, A.k, step)
                if opt_wmem is not None:
                    for t in tasks:
                        t.refresh_keys()
                torch.save({"heads": heads.state_dict(), "layer": A.layer, "k": A.k, "passes": A.passes, "step": step, "qpos": A.qpos, "reader": A.reader,
                            "walk_bank": {"k": rn.walk.k, "v": rn.walk.v}},
                           os.path.join(A.out, f"ltm_heads_{step}.pt"))
        walker_path = None
        if opt_wmem is not None:
            walker_path = os.path.join(A.out, "walker_state.pt")
            torch.save(model.state_dict(), walker_path)
        torch.save({"heads": heads.state_dict(), "layer": A.layer, "k": A.k, "passes": A.passes, "qpos": A.qpos,
                    "reader": A.reader, "walk_bank": {"k": rn.walk.k, "v": rn.walk.v}, "walker_path": walker_path},
                   os.path.join(A.out, "ltm_heads.pt"))
        print(f"LTM done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
