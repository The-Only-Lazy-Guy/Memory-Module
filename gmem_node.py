"""Closed node-graph memory: local prototype of PROJECT_STATUS section 3.8.

One graph holds everything, for the model's whole life:
  - every input (JSON leaves, records, questions, updates) becomes a node, permanently;
  - a node = value v, label q (what it is / when to use it), relation r + optional pointer;
  - tools are nodes too (label + accept vector + function): exact calculator ops, `edit`, `none`,
    and neural tools the model can spawn;
  - thinking: cue -> label match + relation-gated traffic -> activation that fades over loop steps
    (= short-term memory) -> top-K weighted nodes -> Thinker -> SAY (point at an answer node) or
    ASK (a latent cue for the next loop step: the LOTUS loop, Walker's latent fed back).
Training is deep supervision on the thinking process: per loop step, which nodes and which tool
must be active, say/ask, and the answer.

The Thinker here is a small transformer standing in for Walker (the laptop GPU cannot train the 1B);
the graph, activation and tools are what this prototype tests.

Run:  python gmem_node.py --steps 3000        (train on a stream, eval on a held-out stream)
"""
import argparse
import collections
import math
import random
import re
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

D = 128
MAXC = 96                     # characters per node text
KINDS = ("leaf", "record", "question", "result", "was", "update", "tool", "doc", "ins")   # ins: instruction step


# ============================================================ stream (worlds, questions, updates)
_W = ("alpha amber apple arrow aspen atlas basin beacon birch bloom brook cable canyon cedar cinder "
      "cobalt comet coral crane delta drift ember falcon fern fjord flint forge frost garnet glacier "
      "harbor hazel heron indigo iris jade juniper kelp lagoon lantern larch lotus maple marble meadow "
      "mesa mint nectar oasis onyx opal orchid otter pebble pine plume prairie quartz raven reef ridge "
      "river saffron sage shale sierra slate sparrow spruce summit swift thistle tide timber topaz "
      "tundra valley velvet willow").split()
_F_TRAIN = ("memory", "cost", "speed", "score", "weight", "height", "power", "volume", "price", "length",
            "depth", "budget", "load", "rank", "size", "level")
_F_EVAL = ("energy", "mass", "range", "voltage", "salary", "storage")
_L_TRAIN = ("windows", "servers", "trials", "players", "orders", "rooms")
_L_EVAL = ("devices", "tasks", "sensors")
_P_TRAIN = ("Ana Bo Kira Lemi Oto Joss Muba Pelka Tess Ravi Nia Sol Uma Vik Zed Ivo Lena Mio Paz Ruth "
            "Ezra Fay Gus Hana").split()
_P_EVAL = "Quin Yara Theo Wren Odin Esme Arlo Juno".split()

_Q = {"train": {
    "lookup": ("In {doc}, what is the {f} of {p}?", "{doc}: give the {f} of {p}.", "What {f} does {p} have in {doc}?"),
    "sum": ("In {doc}, what is the total {f} of all {L}?", "{doc}: add up the {f} of every entry in {L}.",
            "How much {f} do all {L} in {doc} have together?"),
    "max": ("In {doc}, what is the highest {f} among the {L}?", "{doc}: find the largest {f} in {L}."),
    "min": ("In {doc}, what is the lowest {f} among the {L}?", "{doc}: find the smallest {f} in {L}."),
    "argmax": ("In {doc}, which entry of {L} has the highest {f}?", "{doc}: who in {L} has the largest {f}?"),
    "diff": ("In {doc}, how much more {f} does {a} have than {b}?", "{doc}: the {f} of {a} minus that of {b}?"),
    "history": ("In {doc}, what was the {f} of {p} before the update?", "{doc}: the old {f} of {p}, before it changed?")},
    "eval": {
    "lookup": ("Tell me {p}'s {f} according to {doc}.",),
    "sum": ("Looking at {doc}, how much {f} do the {L} come to in total?",),
    "max": ("Looking at {doc}, which {f} value is the greatest across {L}?",),
    "min": ("Looking at {doc}, which {f} value is the tiniest across {L}?",),
    "argmax": ("Looking at {doc}, whose {f} is the greatest among {L}?",),
    "diff": ("Looking at {doc}, by how much does {a} exceed {b} in {f}?",),
    "history": ("Before {doc} was updated, what {f} did {p} have?",)}}
_PRE = ("In {doc}, ", "{doc}: ", "From {doc}, ", "Using {doc}, ", "Check {doc}: ")
_CORE = {
    "lookup": ("what is the {f} of {p}?", "give the {f} of {p}.", "what {f} does {p} have?", "{p}'s {f}?",
               "show me the {f} for {p}.", "which {f} is listed for {p}?"),
    "sum": ("what is the total {f} of all {L}?", "add up the {f} of every entry in {L}.", "sum the {f} over the {L}.",
            "how much {f} do all {L} have together?", "what is the combined {f} of the {L}?",
            "how much {f} do the {L} have in total?", "the {f} of all {L} put together?"),
    "max": ("what is the highest {f} among the {L}?", "find the largest {f} in {L}.", "the biggest {f} of the {L}?",
            "what is the top {f} value in {L}?", "which {f} is the largest among the {L}?"),
    "min": ("what is the lowest {f} among the {L}?", "find the smallest {f} in {L}.", "the least {f} of the {L}?",
            "what is the bottom {f} value in {L}?", "which {f} is the smallest among the {L}?"),
    "argmax": ("which entry of {L} has the highest {f}?", "who in {L} has the largest {f}?",
               "name the {L} entry with the biggest {f}.", "who has the top {f} among the {L}?",
               "which of the {L} has the most {f}?"),
    "diff": ("how much more {f} does {a} have than {b}?", "the {f} of {a} minus that of {b}?",
             "what is the gap in {f} between {a} and {b}?", "how far is {a} ahead of {b} in {f}?",
             "how much higher is {a}'s {f} than {b}'s?"),
    "history": ("what was the {f} of {p} before the update?", "the old {f} of {p}, before it changed?",
                "what {f} did {p} have before the change?", "{p}'s previous {f}?",
                "what was {p}'s {f} earlier, before the update?")}
PARA_P = 0.0


def train_question(qt, rng, **fmt):
    """Training wording: templates, or (with PARA_P) a composed frame; eval frames never produced."""
    if rng.random() < PARA_P:
        core = rng.choice(_CORE[qt]).format(**fmt)
        pre = rng.choice(_PRE).format(**fmt)
        return pre + core[0].lower() + core[1:] if not pre.endswith(": ") else pre + core[0].upper() + core[1:]
    return rng.choice(_Q["train"][qt]).format(**fmt)


_GOLD_TOOL = {"sum": "sum", "max": "max", "min": "min", "argmax": "max", "diff": "diff"}


def make_world(rng, split):
    F_, L_, P_ = (_F_TRAIN, _L_TRAIN, _P_TRAIN) if split == "train" else (_F_EVAL, _L_EVAL, _P_EVAL)
    doc = f"{rng.choice(_W)}_{rng.randint(1, 99)}.json"
    L = rng.choice(L_)
    fs = rng.sample(F_, 2)
    people = rng.sample(P_, rng.randint(3, 4))
    recs = []
    for p in people:
        r = {"name": p}
        for f in fs:
            r[f] = rng.randint(10, 999)
        recs.append(r)
    return {"doc": doc, "state": {L: recs}, "L": L, "fields": fs}


# ============================================================ graph
def words_of(text):
    """Lower-case word set. Dots stay inside a word ("22.5", "x.json") but not at its ends: a question
    ending in "... according to x.json." must match the node word "x.json"."""
    return {w.strip(".") for w in re.findall(r"[a-z0-9_.#]+", text.lower())} - {""}


class Graph:
    """Lifelong node store. Text/num are raw content (kept so labels can be re-encoded when the
    encoder drifts); V is the cached value vector (refreshed in idle time)."""

    def __init__(self, cap=6000):
        self.text, self.kind, self.num, self.ptr, self.doc, self.alive, self.total = [], [], [], [], [], [], []
        self.words = []
        self.df = {}                          # word -> number of nodes containing it (rare words weigh more)
        self.V = torch.zeros(0, D)
        self.Dv = torch.zeros(0, D)           # latent revisions (model's `revise` tool), kept across re-encoding
        self.keep = []                        # per-node keep state in [0, 1] (model's `forget` lowers it)
        self.links = []                       # extra pointers made by the model's `link` tool: (src, dst, strength)
        self.cap = cap
        self.protect = None                   # doc currently being worked on (not retired mid-episode)

    def __len__(self):
        return len(self.text)

    def add(self, text, kind, num=None, ptr=-1, doc=""):
        self.text.append(text)
        ws = words_of(text)
        self.words.append(ws)
        if kind != "question":
            for w in ws:
                self.df[w] = self.df.get(w, 0) + 1
        self.kind.append(kind)
        self.num.append(num)
        self.ptr.append(ptr)
        self.doc.append(doc)
        self.alive.append(True)
        self.total.append(0.0)
        self.keep.append(1.0)
        self.V = torch.cat([self.V, torch.zeros(1, D, device=self.V.device)])
        self.Dv = torch.cat([self.Dv.to(self.V.device), torch.zeros(1, D, device=self.V.device)])
        return len(self.text) - 1

    def ingest(self, world):
        """JSON state -> record + leaf nodes (leaf -> record pointers). Returns {(name, field): id}."""
        doc, L = world["doc"], world["L"]
        ids = {}
        d = self.add(f"{doc}: document ({L})", "doc", doc=doc)       # records point to their document
        for j, rec in enumerate(world["state"][L]):
            r = self.add(f"{doc}: {L} #{j + 1} record (name = {rec['name']})", "record", ptr=d, doc=doc)
            for k, v in rec.items():
                num = float(v) if isinstance(v, (int, float)) else None
                ids[(rec["name"], k)] = self.add(f"{doc}: {L} #{j + 1} {k} = {v} (name = {rec['name']})",
                                                 "leaf", num=num, ptr=r, doc=doc)
        return ids

    def lex(self, qtext, dev):
        """IDF-weighted word overlap between the question and every node (a unique document name
        counts far more than a common field word)."""
        qw = words_of(qtext)
        n = max(1, len(self.text))
        idf = {w: math.log((n + 1) / (self.df.get(w, 0) + 1)) + 0.1 for w in qw}
        tot = sum(idf.values()) or 1.0
        return torch.tensor([sum(idf[w] for w in qw & ws) / tot if self.kind[i] != "question" else 0.0
                             for i, ws in enumerate(self.words)], device=dev)

    def prune(self):
        """Capacity fallback only (the model's own `forget` tool is the normal path)."""
        live = [i for i, a in enumerate(self.alive) if a and self.doc[i] != self.protect]
        if len(live) <= self.cap:
            return
        for i in sorted(live, key=lambda i: self.keep[i] * (1.0 + self.total[i]))[:len(live) - self.cap]:
            self.alive[i] = False


# ============================================================ model
class TextEnc(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(128, D)
        self.pos = nn.Embedding(MAXC, D)
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(D, 4, 4 * D, 0.0, batch_first=True,
                                                                    norm_first=True), 2)
        self.out = nn.Linear(D, D)

    def forward(self, texts):
        dev = self.emb.weight.device
        ids = torch.zeros(len(texts), MAXC, dtype=torch.long, device=dev)
        m = torch.zeros(len(texts), MAXC, dtype=torch.bool, device=dev)
        for i, t in enumerate(texts):
            c = [min(ord(ch), 127) for ch in t.lower()[:MAXC]]
            ids[i, :len(c)] = torch.tensor(c, device=dev)
            m[i, :len(c)] = True
        x = self.emb(ids) + self.pos.weight[None]
        h = self.enc(x, src_key_padding_mask=~m)
        h = (h * m[..., None]).sum(1) / m.sum(1, keepdim=True).clamp(min=1)
        return self.out(h)


class WalkerTextEnc(nn.Module):
    """Frozen Walker 1B as the text encoder (meaning of wording): pooled final states, cached per text
    (Walker is frozen), plus a small trainable projection into the node space."""

    def __init__(self, rn, d_in, standardize=True, max_cache=50000):
        super().__init__()
        self.rn = rn
        self.proj = nn.Sequential(nn.Linear(d_in, 2 * D), nn.GELU(), nn.Linear(2 * D, D))
        self.cache = collections.OrderedDict()      # LRU: pooled states per text (~4 KB each at D=2048)
        self.max_cache = max_cache
        # Pooled LM states share one huge component (cos .97-.99 between "boss:" and "friend:" texts);
        # the meaning that separates nodes is a small direction. Standardize per feature (running stats,
        # no gradient) so the projection can see it: a linear probe reads the relation 100% only after this.
        self.standardize = standardize
        self.register_buffer("mu", torch.zeros(d_in))
        self.register_buffer("var", torch.ones(d_in))
        self.register_buffer("seen", torch.zeros(()))

    def forward(self, texts):
        for t in texts:
            if t in self.cache:
                self.cache.move_to_end(t)
        miss = sorted({t for t in texts if t not in self.cache})
        if miss:
            with torch.no_grad(), self.rn.active():
                v = self.rn.pooled(miss)
            for t, x in zip(miss, v):
                self.cache[t] = x
        x = torch.stack([self.cache[t] for t in texts]).float().to(self.proj[0].weight.device)
        while len(self.cache) > self.max_cache:      # evict after use: this batch is already gathered
            self.cache.popitem(last=False)
        if self.standardize:
            if self.training and x.shape[0] > 1:
                with torch.no_grad():
                    m = 0.0 if float(self.seen) == 0 else 0.99   # first batch sets the stats
                    self.mu.mul_(m).add_((1 - m) * x.mean(0))
                    self.var.mul_(m).add_((1 - m) * x.var(0))
                    self.seen.add_(1)
            x = (x - self.mu) / (self.var + 1e-5).sqrt()
        return self.proj(x)


def load_walker(ckpt):
    """Frozen Walker + a runner whose pooled() gives text features (as the hub does)."""
    import gmem_ltm as LT
    model, raw_tok, _ = LT.T.build_model_for_inference(ckpt)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    rn = LT.LTMRunner(model, LT.Tok(raw_tok), None, 2)
    return rn, model.emb.weight.shape[1]


class Tool:
    """A tool node. `fn` is exact (python) or a small network (spawned)."""

    def __init__(self, name, fn=None, net=None):
        self.name, self.fn, self.net = name, fn, net


def _exact(op):
    def f(vals):
        if not vals:
            return None
        if op == "sum":
            return sum(vals)
        if op == "max":
            return max(vals)
        if op == "min":
            return min(vals)
        if op == "diff":
            return abs(vals[0] - vals[1]) if len(vals) >= 2 else None
    return f


class NodeMemory(nn.Module):
    def __init__(self, K=12, hops=3, enc=None):
        super().__init__()
        self.K, self.hops = K, hops
        self.enc = enc if enc is not None else TextEnc()
        self.kind = nn.Embedding(len(KINDS), D)
        self.lab = nn.Sequential(nn.Linear(2 * D + 1, D), nn.GELU(), nn.Linear(D, D))   # label q_n
        self.ctx = nn.Linear(D, D, bias=False)        # label also sees the parent ("boss = e40" under "e12")
        nn.init.zeros_(self.ctx.weight)               # zero: old checkpoints' labels are unchanged
        self.rel = nn.Sequential(nn.Linear(2 * D, D), nn.GELU(), nn.Linear(D, D))       # relation r_n
        self.W = nn.Linear(D, D, bias=False)          # cue -> label space
        self.U = nn.Linear(D, D, bias=False)          # cue -> relation space (traffic gate)
        # relation as an edge feature: a learned pairwise bilinear over relation vectors makes a
        # directed soft adjacency (ownership / inverse / neighbour types) that activation flows
        # along. w_edge starts at 0 -> strict no-op until training opens it.
        self.ea = nn.Linear(D, D, bias=False)
        self.eb = nn.Linear(D, D, bias=False)
        self.w_edge = nn.Parameter(torch.tensor(0.0))
        # relation as a conditional operator (dependence): pred_k = V[owner(k)] + rop(r_k).
        self.rop = nn.Linear(D, D, bias=False)
        self.tau_o = nn.Parameter(torch.tensor(math.log(5.0)))
        self.w_op = nn.Parameter(torch.tensor(0.0))
        # relation as a MATCH feature: direct cos(query, relation) term in the content score
        # (w_rel=0 => no-op until trained).
        self.w_rel = nn.Parameter(torch.tensor(0.0))
        self.g = nn.Parameter(torch.tensor(0.5))      # traffic strength
        self.lam = nn.Parameter(torch.tensor(0.0))    # activation fade (sigmoid)
        self.w_lex = nn.Parameter(torch.tensor(5.0))  # lexical evidence weight
        self.C = 64                                   # stage 1: cheap candidates (crowding fix)
        # tools as nodes: label, accept vector, function
        self.tools = [Tool("none"), Tool("sum", _exact("sum")), Tool("max", _exact("max")),
                      Tool("min", _exact("min")), Tool("diff", _exact("diff")), Tool("edit"),
                      Tool("revise"), Tool("link"), Tool("forget")]
        self.MEMORY_OPS = ("edit", "revise", "link", "forget")
        # knob per memory action: strength in [0, 1] from (cue, labels of its targets)
        self.knob = nn.ModuleDict({n: nn.Sequential(nn.Linear(2 * D, D), nn.GELU(), nn.Linear(D, 1))
                                   for n in self.MEMORY_OPS})
        for k in self.knob.values():                  # start "on": a chosen action is not blocked by an untrained knob
            nn.init.constant_(k[-1].bias, 3.0)
        self.revise_net = nn.Sequential(nn.Linear(2 * D, D), nn.GELU(), nn.Linear(D, D))   # delta for `revise`
        # knob per memory: keep value from its label + use statistics (soft forgetting / capacity order)
        self.keep_net = nn.Sequential(nn.Linear(D + 3, D // 2), nn.GELU(), nn.Linear(D // 2, 1))
        self.tq = nn.Parameter(torch.randn(len(self.tools), D) * 0.1)
        self.ta = nn.Parameter(torch.randn(len(self.tools), D) * 0.1)
        self.spawned = nn.ModuleList()
        self.Acc = nn.Linear(D, D, bias=False)        # cue -> what this call should take (which field / record)
        self.kappa = nn.Parameter(torch.tensor(8.0))  # args follow relative activation (log a / a_max) ...
        self.abias = nn.Parameter(torch.tensor(2.0))  # ... plus the learned label/cue adjustment
        # thinker (stand-in for Walker): reads [cue, question, K weighted nodes, tool output]
        self.P = nn.Linear(2 * D, D)
        self.slot = nn.Embedding(4, D)                # cue / question / node / tool-output
        self.think = nn.TransformerEncoder(nn.TransformerEncoderLayer(D, 4, 4 * D, 0.0, batch_first=True,
                                                                      norm_first=True), 3)
        self.gate = nn.Linear(D, 1)                   # say (1) vs ask (0)
        self.ask = nn.Linear(D, D)
        self.beta = nn.Parameter(torch.tensor(0.0))   # Walk momentum on the cue
        self.ptr_q = nn.Linear(D, D)                  # say: point at the answer node
        self.force = None                             # training: gold nodes always reach stage 2

    def _load_from_state_dict(self, sd, prefix, *args, **kw):
        k = prefix + "kind.weight"                     # checkpoints from before a node kind was added
        if k in sd and sd[k].shape[0] < self.kind.weight.shape[0]:
            w = self.kind.weight.detach().clone()
            w[:sd[k].shape[0]] = sd[k]
            sd[k] = w
        super()._load_from_state_dict(sd, prefix, *args, **kw)

    @property
    def dev(self):
        return self.tq.device

    # ------------------------------------------------------------ labels
    def node_vecs(self, G, fresh):
        """Values for all live nodes: cached, except `fresh` ids (encoded with gradient)."""
        V = G.V.to(self.dev).clone()
        if fresh:
            V[fresh] = self.enc([G.text[i] for i in fresh])
            G.V[fresh] = V[fresh].detach().to(G.V.device)       # stored memory keeps its vector (lifelong)
        V = V + G.Dv.to(self.dev)[:V.shape[0]]                 # the model's own latent revisions
        kinds = torch.tensor([KINDS.index(k) for k in G.kind], device=self.dev)
        has = torch.tensor([[1.0 if n is not None else 0.0] for n in G.num], device=self.dev)
        par = torch.tensor([p if p >= 0 else i for i, p in enumerate(G.ptr)], device=self.dev)
        child = (par != torch.arange(par.shape[0], device=self.dev)).float()[:, None]
        Q = self.lab(torch.cat([V, self.kind(kinds), has], -1)) + child * self.ctx(V[par])
        R = self.rel(torch.cat([V, V[par]], -1))
        self._keep_eff = self.keep_eff(G, Q)
        return V, Q, R, par

    def keep_eff(self, G, Q):
        """Effective keep per node = stored keep state x learned keep knob (label + use stats)."""
        n = Q.shape[0]
        tot = torch.tensor(G.total[:n], device=self.dev)
        stats = torch.stack([torch.log1p(tot), torch.tensor(G.keep[:n], device=self.dev),
                             torch.tensor([float(k == "was") for k in G.kind[:n]], device=self.dev)], -1)
        k = torch.sigmoid(self.keep_net(torch.cat([Q, stats], -1)).squeeze(-1) + 3.0)   # starts near 1
        return k * torch.tensor(G.keep[:n], device=self.dev).clamp(min=1e-4)

    @torch.no_grad()
    def refresh(self, G, bs=256):
        """Idle-time re-encoding of every stored node with the current encoder."""
        out = [self.enc(G.text[s:s + bs]) for s in range(0, len(G), bs)]
        G.V = torch.cat(out).detach() if out else G.V

    def tool_labels(self):
        return self.tq

    # ------------------------------------------------------------ one loop step of activation
    def activate(self, G, Q, R, par, cue, lex, a_prev):
        """Label match + relation-gated traffic (both directions along pointers), mixed with the faded
        previous activation. Returns node scores, tool scores and the new activation."""
        e = (Q @ self.W(cue)) / math.sqrt(D) + self.w_lex * lex
        ke = getattr(self, "_keep_eff", None)
        if ke is not None and ke.shape[0] == e.shape[0]:
            e = e + torch.log(ke.clamp(min=1e-4))              # weakly kept memories fade out of reach
        alive = torch.tensor(G.alive[:Q.shape[0]], device=self.dev)
        e = e.masked_fill(~alive, -1e4)
        if self.C and e.shape[0] > self.C:               # two stages: only the top-C cheap matches compete
            keep = torch.zeros_like(e, dtype=torch.bool)
            keep[torch.topk(e.detach(), self.C).indices] = True
            if self.force is not None:
                keep[self.force] = True
            ix = torch.arange(par.shape[0], device=self.dev)
            for _ in range(self.hops):                  # candidates' graph neighbours compete too
                has = par != ix
                up = torch.zeros(keep.shape[0], device=self.dev).index_add_(0, par, (keep & has).float()) > 0
                keep = keep | (keep[par] & has) | up          # children AND parents of candidates
            e = e.masked_fill(~keep, -1e4)
        gate = torch.sigmoid(R @ self.U(cue) / math.sqrt(D))
        idx = torch.arange(par.shape[0], device=self.dev)
        has = par != idx
        s = e
        for _ in range(self.hops):
            up = torch.zeros_like(s).index_add(0, par[has], (gate[has] * s[has].clamp(min=0)))
            down = torch.zeros_like(s)
            down[has] = gate[has] * s[par[has]].clamp(min=0)
            s = e + self.g * (up + down)
            for (i, j, w) in G.links:                          # model-made links carry traffic both ways
                if i < s.shape[0] and j < s.shape[0]:
                    s = s.index_add(0, torch.tensor([j, i], device=self.dev),
                                    self.g * w * torch.stack([s[i], s[j]]).clamp(min=0))
        s = s.masked_fill(~alive, -1e4)
        ts = (self.tool_labels() @ self.W(cue)) / math.sqrt(D)
        lam = torch.sigmoid(self.lam)
        a = (1 - lam) * torch.softmax(s, 0)
        if a_prev is not None:
            a = a + lam * a_prev
        return s, ts, a

    def message_adj(self, V, R, par):
        """Combined directed relation/dependence adjacency (one (N,N) tensor, computed per world):
        relation-similarity edges + conditional dependence edges (pred_k = V[owner]+rop(r_k))."""
        n = R.shape[0]
        if n < 2:
            return None
        adj = (self.w_edge * torch.softmax(self.ea(R) @ self.eb(R).t() / math.sqrt(D), -1)
               if self.w_edge is not None else None)
        if self.w_op is not None and V is not None:
            pred = V[par] + self.rop(R)
            op = torch.softmax(self.tau_o.exp() * F.normalize(pred, dim=-1) @ F.normalize(V, dim=-1).t(), -1)
            adj = self.w_op * op if adj is None else adj + self.w_op * op
        return adj

    def activate_batch(self, G, Q, R, par, cues, force=None, bias=None, V=None, adj=None):
        """`activate` for B cues at once (same label match, keep, two-stage candidates and relation
        traffic; model-made links are not batched). Returns scores (B, N). force: (B,) gold ids that
        always reach stage 2 (training). bias: (B, N) extra evidence added before the candidate stage
        (word overlap, entity edges). adj: world-cached relation/dependence adjacency."""
        n = Q.shape[0]
        e = (self.W(cues) @ Q.t()) / math.sqrt(D)
        if bias is not None:
            e = e + bias
        ke = getattr(self, "_keep_eff", None)
        if ke is not None and ke.shape[0] == n:
            e = e + torch.log(ke.clamp(min=1e-4))
        # direct relation match (w_rel=0 => no-op): boost nodes whose relation agrees with the query
        if getattr(self, "w_rel", None) is not None:
            e = e + self.w_rel * (torch.nn.functional.normalize(cues, dim=-1)
                                  @ torch.nn.functional.normalize(R, dim=-1).t())
        alive = torch.tensor(G.alive[:n], device=self.dev)
        e = e.masked_fill(~alive, -1e4)
        idx = torch.arange(n, device=self.dev)
        has = par != idx
        if self.C and n > self.C:
            keep = torch.zeros_like(e, dtype=torch.bool)
            keep.scatter_(1, torch.topk(e.detach(), self.C, -1).indices, True)
            if force is not None:
                keep[torch.arange(e.shape[0], device=self.dev), force] = True
            for _ in range(self.hops):
                up = torch.zeros(keep.shape, device=self.dev).scatter_add_(
                    1, par.expand(keep.shape[0], -1), (keep & has).float()) > 0
                keep = keep | (keep[:, par] & has) | up       # children AND parents of candidates
            e = e.masked_fill(~keep, -1e4)
        gate = torch.sigmoid(self.U(cues) @ R.t() / math.sqrt(D))
        if adj is None:                       # world-cached in graph_ctx; else compute here
            adj = self.message_adj(V, R, par)
        s = e
        for _ in range(self.hops):
            up = torch.zeros_like(s).index_add(1, par[has], gate[:, has] * s[:, has].clamp(min=0))
            down = torch.zeros_like(s)
            down[:, has] = gate[:, has] * s[:, par[has]].clamp(min=0)
            s = e + self.g * (up + down)
            if adj is not None:
                s = s + (s.clamp(min=0) @ adj)
        return s.masked_fill(~alive, -1e4)

    # ------------------------------------------------------------ one question / event
    def run(self, G, qtext, steps=None, max_steps=3, train=False, fresh=(), qnode=None):
        """steps: gold chain [(kind, gold_nodes, tool, answer_node)] (kind 'ask' or 'say')."""
        V, Q, R, par = self.node_vecs(G, list(fresh))
        qv = self.enc([qtext])[0]
        lex = G.lex(qtext, self.dev)
        cue, cue_prev, a = qv, qv, None
        losses, trace, answer = [], [], None
        T = len(steps) if train else max_steps
        for t in range(T):
            gold = steps[t] if steps is not None and t < len(steps) else None
            self.force = (torch.tensor([g for g in list(gold[1]) + [gold[3]] if isinstance(g, int)], device=self.dev)
                          if train else None)
            s, ts, a = self.activate(G, Q, R, par, cue, lex, a)
            # tool choice (incl. `none`) and its arguments
            ti = self.tools.index(next(x for x in self.tools if x.name == gold[2])) if train else int(ts.argmax())
            if train:
                losses.append(F.cross_entropy(ts[None], torch.tensor([ti], device=self.dev)))
                for g in gold[1]:                                   # every gold node must be active
                    losses.append(F.cross_entropy(s[None], torch.tensor([g], device=self.dev)) / len(gold[1]))
            top = torch.topk(a, min(self.K, a.shape[0])).indices.tolist()
            if train:
                top = list(dict.fromkeys(list(gold[1]) + ([gold[3]] if isinstance(gold[3], int) else []) + top))[:self.K + 4]
            for i, x in zip(top, a[top].tolist()):
                G.total[i] += x
            tool = self.tools[ti]
            out_node, out_vec = None, None
            if tool.name != "none":
                ar = a[top] / a[top].max().clamp(min=1e-9)
                acc = torch.sigmoid(Q[top] @ (self.ta[ti] + self.Acc(cue)) / math.sqrt(D)
                                    + self.kappa * torch.log(ar.clamp(min=1e-6)) + self.abias)
                cand = [k for k, i in enumerate(top) if (G.num[i] is not None or tool.name == "edit")
                        and G.kind[i] not in ("question",)]
                if train:
                    tgt = torch.tensor([1.0 if top[k] in gold[1] else 0.0 for k in range(len(top))], device=self.dev)
                    losses.append(F.binary_cross_entropy_with_logits(torch.logit(acc.clamp(1e-4, 1 - 1e-4)), tgt))
                    args = [top[k] for k in range(len(top)) if top[k] in gold[1]]
                else:
                    args = [top[k] for k in cand if float(acc[k]) > 0.5] or [top[k] for k in cand[:1]]
                    args = sorted(args, key=lambda i: -float(a[i]))
                if tool.name in self.MEMORY_OPS and args:
                    strength = torch.sigmoid(self.knob[tool.name](torch.cat([cue, Q[args].mean(0)]))).squeeze()
                    if train:                                  # gold memory op: the knob should be on
                        losses.append(F.binary_cross_entropy(strength, torch.ones_like(strength)))
                    out_node = self.memory_op(G, tool.name, args, qnode, 1.0 if train else float(strength), V, cue)
                elif tool.fn is not None:
                    val = tool.fn([G.num[i] for i in args])
                    if val is not None:
                        src = max(args, key=lambda i: G.num[i]) if tool.name == "max" else \
                            min(args, key=lambda i: G.num[i]) if tool.name == "min" else -1
                        out_node = G.add(f"result: {tool.name} = {val:g}", "result", num=float(val), ptr=src,
                                         doc=G.doc[args[0]] if args else "")
                        out_vec = self.enc([G.text[out_node]])[0]
                        G.V[out_node] = out_vec.detach().to(G.V.device)
                elif tool.net is not None:                           # spawned neural tool
                    x = torch.cat([V[args].mean(0), Q[args].mean(0)])
                    out_vec = tool.net(x)
            # thinker reads the weighted active nodes (+ tool output)
            toks = [self.slot.weight[0] + cue, self.slot.weight[1] + qv]
            nodes = top + ([out_node] if out_node is not None else [])
            nv = torch.stack([V[i] if i < V.shape[0] else out_vec for i in nodes])
            nq = torch.stack([Q[i] if i < Q.shape[0] else self.lab(torch.cat([out_vec, self.kind.weight[KINDS.index("result")],
                                                                              out_vec.new_ones(1)])) for i in nodes])
            w = torch.stack([a[i] if i < a.shape[0] else a.max() for i in nodes])
            w = w / w.sum().clamp(min=1e-9) * len(nodes)          # relative confidence among what is read
            ntok = w[:, None] * self.P(torch.cat([nq, nv], -1)) + self.slot.weight[2]
            if out_node is not None:
                ntok = torch.cat([ntok[:-1], ntok[-1:] + self.slot.weight[3]])
            h = self.think(torch.cat([torch.stack(toks), ntok])[None])[0]
            hc = h[0]
            say = self.gate(hc).squeeze()
            pscore = h[2:] @ self.ptr_q(hc) / math.sqrt(D)
            trace.append({"tool": tool.name, "top": [G.text[i][:60] for i in top[:4]],
                          "args": [G.text[i][:50] for i in (args if tool.name != "none" else [])],
                          "made": G.text[out_node] if out_node is not None and out_node < len(G) else None,
                          "say": float(torch.sigmoid(say)),
                          "tool_ok": None if gold is None else tool.name == gold[2],
                          "args_ok": None if gold is None or gold[2] in ("none", "edit") or tool.name != gold[2]
                          else set(args) == set(gold[1])})
            if train:
                losses.append(F.binary_cross_entropy_with_logits(say, torch.tensor(1.0 if gold[0] == "say" else 0.0,
                                                                                   device=self.dev)))
                if gold[0] == "say":
                    tgt_node = out_node if gold[3] == "made" else gold[3]
                    if tgt_node in nodes:
                        losses.append(F.cross_entropy(pscore[None], torch.tensor([nodes.index(tgt_node)], device=self.dev)))
                    break
            elif float(torch.sigmoid(say)) > 0.5 or t == T - 1:
                answer = G.text[nodes[int(pscore.argmax())]]
                break
            nc = self.ask(hc)
            cue, cue_prev = nc + self.beta * (nc - cue_prev), cue
        return (sum(losses) if losses else None), answer, trace

    def memory_op(self, G, op, args, qnode, strength, V, cue):
        """The model's memory actions. `strength` is that action's knob (0..1)."""
        if op == "edit":
            return self._edit(G, args, qnode) if strength > 0.5 else None
        if op == "revise":                                     # soft latent update of the target node
            t = args[0]
            with torch.no_grad():
                delta = self.revise_net(torch.cat([V[t], cue]))
                G.Dv[t] = (G.Dv[t].to(self.dev) + strength * delta).to(G.Dv.device)
            return None
        if op == "link" and len(args) >= 2:                    # new pointer, its traffic = strength
            G.links.append((args[0], args[1], strength))
            return None
        if op == "forget":                                     # soft: lower the keep state
            for t in args[:1]:
                G.keep[t] = G.keep[t] * (1.0 - strength)
            return None
        return None

    def _edit(self, G, args, qnode):
        """Edit tool: the update node replaces the target; the target becomes a 'was' version that
        points at the new node; the new node takes over the target's record pointer."""
        targets = [i for i in args if G.kind[i] == "leaf" and i != qnode]
        if not targets or qnode is None:
            return None
        old = targets[0]
        G.ptr[qnode] = G.ptr[old]
        G.kind[qnode] = "leaf"
        G.kind[old] = "was"
        G.text[old] = G.text[old].replace(" = ", " was ", 1)
        new = words_of(G.text[old])
        for w in new - G.words[old]:                          # lexical evidence follows the new text
            G.df[w] = G.df.get(w, 0) + 1
        for w in G.words[old] - new:
            G.df[w] = max(0, G.df.get(w, 0) - 1)
        G.words[old] = new
        G.ptr[old] = qnode
        return None

    # ------------------------------------------------------------ model-made tools
    def spawn(self, cue):
        """A cue that matches no node or tool well, with high loss -> a new neural tool whose label
        starts at that cue (the model's own new function)."""
        net = nn.Sequential(nn.Linear(2 * D, D), nn.GELU(), nn.Linear(D, D)).to(self.dev)
        self.spawned.append(net)
        self.tools.append(Tool(f"spawn{len(self.spawned)}", net=net))
        with torch.no_grad():
            self.tq.data = torch.cat([self.tq.data, cue.detach().clone()[None].to(self.dev)])
            self.ta.data = torch.cat([self.ta.data, torch.randn(1, D, device=self.dev) * 0.1])
        return len(self.tools) - 1


# ============================================================ episodes over the stream
def answer_of(text):
    m = re.search(r"(?:=|was) ([^()]+?)(?: \(|$)", text)
    return m.group(1).strip() if m else text


def world_episode(M, G, world, rng, split, train, n_q=5, past=(), n_recall=0):
    """Ingest a world, ask questions, apply one update, ask about it. Returns loss and records.
    past + n_recall (training): also ask lookups about EARLIER worlds still in the graph (delayed
    recall), gold = the node holding the current value; otherwise nothing teaches documents apart."""
    doc, L, fs = world["doc"], world["L"], world["fields"]
    T = _Q[split]
    ids = G.ingest(world)
    world["cur"] = dict(ids)                          # (name, field) -> node with the current value
    G.protect = doc
    G.prune()
    fresh = [i for i in range(len(G)) if G.doc[i] == doc]
    recs = world["state"][L]
    out, loss = [], []

    def ask(qtype, fmt, steps, gold_answer):
        q = (train_question(qtype, rng, doc=doc, L=L, **fmt) if split == "train"
             else rng.choice(T[qtype]).format(doc=doc, L=L, **fmt))
        qn = G.add(q, "question", doc=doc)
        l, ans, tr = M.run(G, q, steps=steps, train=train, fresh=fresh)
        if l is not None:
            loss.append(l)
        if not train:
            out.append((qtype, q, gold_answer, answer_of(ans) if ans else "", tr))
        return qn

    for _ in range(n_q):
        qt = rng.choice(("lookup", "sum", "max", "min", "argmax", "diff"))
        f = rng.choice(fs)
        if qt == "lookup":
            r = rng.choice(recs)
            g = ids[(r["name"], f)]
            ask(qt, dict(f=f, p=r["name"]), [("say", [g], "none", g)], str(r[f]))
        elif qt in ("sum", "max", "min"):
            gl = [ids[(r["name"], f)] for r in recs]
            val = {"sum": sum, "max": max, "min": min}[qt]([r[f] for r in recs])
            ask(qt, dict(f=f), [("say", gl, qt, "made")], f"{val:g}")
        elif qt == "argmax":
            gl = [ids[(r["name"], f)] for r in recs]
            best = max(recs, key=lambda r: r[f])
            ask(qt, dict(f=f), [("ask", gl, "max", None), ("say", [ids[(best["name"], "name")]], "none",
                                                            ids[(best["name"], "name")])], best["name"])
        else:
            a, b = sorted(rng.sample(recs, 2), key=lambda r: -r[f])
            ask(qt, dict(f=f, a=a["name"], b=b["name"]),
                [("say", [ids[(a["name"], f)], ids[(b["name"], f)]], "diff", "made")], f"{a[f] - b[f]:g}")
    # an update arrives as a new input node; the model must route it to the right old node (edit tool)
    r = rng.choice(recs)
    f = rng.choice(fs)
    old_val, new_val = r[f], r[f] + rng.randint(1, 50)
    r[f] = new_val                                    # the world's current state (delayed recall reads it)
    j = recs.index(r)
    un = G.add(f"{doc}: {L} #{j + 1} {f} = {new_val} (name = {r['name']}) [update]", "update",
               num=float(new_val), doc=doc)
    world["cur"][(r["name"], f)] = un
    world.setdefault("was", {})[(r["name"], f)] = ids[(r["name"], f)]
    fresh.append(un)
    utext = f"update {doc}: {r['name']} {f} becomes {new_val}"
    l, _, tr = M.run(G, utext, steps=[("say", [ids[(r['name'], f)]], "edit", un)],
                     train=train, fresh=fresh, qnode=un)
    if train and l is not None:
        loss.append(l)
    if not train:                                     # eval: apply the model's own edit decision
        out.append(("edit", utext, "ok" if G.kind[ids[(r['name'], f)]] == "was" else "missed", "", tr))
    ask("history", dict(f=f, p=r["name"]), [("say", [ids[(r['name'], f)]], "none", ids[(r['name'], f)])], str(old_val))
    ask("lookup", dict(f=f, p=r["name"]), [("say", [un], "none", un)], str(new_val))
    old = [w for w in past if w is not world and "cur" in w]
    for w in [rng.choice(old) for _ in range(n_recall if old else 0)]:
        r2 = rng.choice(w["state"][w["L"]])
        f2 = rng.choice(w["fields"])
        old_q = (r2["name"], f2) in w.get("was", {}) and rng.random() < 0.5   # its value before the update
        g = w["was"][(r2["name"], f2)] if old_q else w["cur"][(r2["name"], f2)]
        if g >= len(G) or not G.alive[g] or (old_q and G.kind[g] != "was"):
            continue
        q = train_question("history" if old_q else "lookup", rng, doc=w["doc"], L=w["L"], f=f2, p=r2["name"])
        l, _, _ = M.run(G, q, steps=[("say", [g], "none", g)], train=train, fresh=fresh)
        if l is not None:
            loss.append(l)
    return (sum(loss) / max(1, len(loss)) if loss else None), out


def evaluate(M, n_worlds, seed, gap_worlds=0, cap=6000, n_recall=60):
    """Fresh lifelong graph, held-out vocabulary and wording. Optional long-gap recall check: ask about
    the first world again after `gap_worlds` other worlds were stored."""
    rng = random.Random(seed)
    G = Graph(cap=cap)
    G.V = G.V.to(M.dev)
    M.eval()
    res = []
    first = None
    worlds = []
    with torch.no_grad():
        for w in range(n_worlds):
            world = make_world(rng, "eval")
            first = first or world
            worlds.append(world)
            _, out = world_episode(M, G, world, rng, "eval", train=False)
            res += out
        # delayed recall: lookups about EARLIER worlds after everything is stored (what the cap can hurt)
        G.protect = None
        G.prune()
        rec = []
        for w in rng.sample(worlds[:-5], min(n_recall, max(0, len(worlds) - 5))):
            r = rng.choice(w["state"][w["L"]])
            f = rng.choice(w["fields"])
            cur = r[f]
            q = f"Tell me {r['name']}'s {f} according to {w['doc']}."
            _, ans, _ = M.run(G, q)
            rec.append(answer_of(ans) if ans else "")
            res.append(("recall", q, str(cur), rec[-1], []))
        gap = []
        if gap_worlds:
            for _ in range(gap_worlds):
                world_episode(M, G, make_world(rng, "eval"), rng, "eval", train=False)
            r = first["state"][first["L"]][0]
            f = first["fields"][0]
            q = f"Tell me {r['name']}'s {f} according to {first['doc']}."
            _, ans, tr = M.run(G, q)
            gap.append((q, str(r[f]), answer_of(ans) if ans else ""))
    M.train()
    acc = {}
    for qt, q, gold, pred, tr in res:
        acc.setdefault(qt, []).append(float(gold == pred) if qt != "edit" else float(gold == "ok"))
        if tr and tr[0]["tool_ok"] is not None:
            acc.setdefault(f"~tool_{qt}", []).append(float(tr[0]["tool_ok"]))
            if tr[0]["args_ok"] is not None:
                acc.setdefault(f"~args_{qt}", []).append(float(tr[0]["args_ok"]))
    return {k: sum(v) / len(v) for k, v in acc.items()}, res, gap, len(G)


def make_opt(M, lr, scalar_mult):
    """AdamW; scalar knobs (lexical weight, traffic strength, fade, ...) get their own LR: Adam moves a
    parameter about one LR per step, so at the base LR they would stay at their init."""
    sc = [q for q in M.parameters() if q.dim() == 0]
    ids = {id(q) for q in sc}
    rest = [q for q in M.parameters() if id(q) not in ids]
    return torch.optim.AdamW([{"params": rest}, {"params": sc, "lr": lr * scalar_mult, "weight_decay": 0.0}],
                             lr=lr, weight_decay=0.01)


def main():
    P = argparse.ArgumentParser()
    P.add_argument("--steps", type=int, default=3000)
    P.add_argument("--lr", type=float, default=3e-4)
    P.add_argument("--eval-every", type=int, default=500)
    P.add_argument("--seed", type=int, default=0)
    P.add_argument("--eval-worlds", type=int, default=100)
    P.add_argument("--paraphrase", type=float, default=0.7)
    P.add_argument("--cap", type=int, default=4000)
    P.add_argument("--walker", default="", help="Walker checkpoint: frozen Walker states as the text encoder")
    P.add_argument("--save", default="gmem_node.pt")
    P.add_argument("--scalar-lr-mult", type=float, default=1.0,
                   help="LR multiplier for scalar knobs (w_lex, traffic g, fade lam, kappa, ...)")
    P.add_argument("--train-recall", type=int, default=2, help="delayed-recall questions per training episode")
    A = P.parse_args()
    torch.manual_seed(A.seed)
    global PARA_P
    PARA_P = A.paraphrase
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    enc = None
    if A.walker:
        rn, dw = load_walker(A.walker)
        enc = WalkerTextEnc(rn, dw)
    M = NodeMemory(enc=enc).to(dev)
    opt = make_opt(M, A.lr, A.scalar_lr_mult)
    G = Graph(cap=A.cap)
    G.V = G.V.to(dev)
    rng = random.Random(A.seed)
    t0 = time.time()
    run = []
    past = []
    base = [g["lr"] for g in opt.param_groups]
    best = -1.0
    for step in range(1, A.steps + 1):
        f = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - 1) / A.steps))   # cosine to 10%
        for g, lr in zip(opt.param_groups, base):
            g["lr"] = lr * f
        world = make_world(rng, "train")
        loss, _ = world_episode(M, G, world, rng, "train", train=True, past=past, n_recall=A.train_recall)
        past = (past + [world])[-200:]
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(M.parameters(), 1.0)
        opt.step()
        run.append(float(loss))
        if step % 50 == 0:
            M.refresh(G)
            G.prune()
        if step % 100 == 0:
            print(f"step {step} loss {sum(run[-100:]) / len(run[-100:]):.3f} graph {len(G)} "
                  f"({sum(G.alive)} alive) w_lex {float(M.w_lex):.2f} g {float(M.g):.2f} "
                  f"lam {float(torch.sigmoid(M.lam)):.2f} {time.time() - t0:.0f}s", flush=True)
        if step % A.eval_every == 0 or step == A.steps:
            acc, res, gap, n = evaluate(M, A.eval_worlds, 1234, gap_worlds=40 if step == A.steps else 0, cap=A.cap)
            main_acc = {k: v for k, v in acc.items() if not k.startswith("~")}
            print(f"EVAL step {step} " + " ".join(f"{k}={v:.2f}" for k, v in sorted(main_acc.items()))
                  + f" all={sum(main_acc.values()) / len(main_acc):.2f} graph={n}", flush=True)
            print(f"DIAG step {step} " + " ".join(f"{k[1:]}={v:.2f}" for k, v in sorted(acc.items())
                                                if k.startswith("~")), flush=True)
            score = sum(main_acc.values()) / len(main_acc)
            if score > best:                                   # keep the best checkpoint by eval
                best = score
                torch.save({k: v for k, v in M.state_dict().items() if not k.startswith("enc.rn")}, A.save)
                print(f"saved best @{step}: all {score:.2f}", flush=True)
            if step == A.steps:
                for qt, q, gold, pred, tr in res[:14]:
                    print(f"  [{qt}] {q} gold={gold} pred={pred}")
                    for s in tr:
                        print(f"      tool={s['tool']} say={s['say']:.2f} top={s['top'][:2]} made={s['made']}")
                for q, g, p in gap:
                    print(f"  [long gap, 40 worlds later] {q} gold={g} pred={p}")


if __name__ == "__main__":
    main()
