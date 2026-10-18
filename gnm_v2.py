"""GNM design v2, step 1 (docs/GNM.md): encoder in Walker's own query space + cosine label attention.

The input is ONE JSON document: the graph serialized with nesting (a record's facts inside its object,
a procedure's steps inside it, the asked questions as pairs). The whole document runs once through
Walker's own P blocks and one R pass — the layer where the recurrent pass state s_p lives, from which
the queries are made — so every token is contextualized by the document (Walker is causal: a fact sees
its owner's key before it) and labels and queries start in one space. Each node is the token span of
its key-value pair; learned attention pooling over the span (one scorer per head, not a mean) and three
heads give the node's value v, label l and relation r. What goes into the label is learned.

Activation (the drop-in for NodeMemory.activate_batch used by the hook):
    e_n = tau * cos(q, l_n)                       pick by label (learned temperature)
    traffic: a node passes activation to its parent / children, gated by sigmoid(tau_r * cos(q, r_n))
The token states are frozen (Walker is frozen); heads and pooling learn.
"""
import collections
import json
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import gmem_node as N


class _Stop(Exception):
    pass


def serialize(G):
    """The graph as one JSON-like document: roots in order, children nested in their parent's object; a
    node "k: v" becomes "k": "v". Returns the text pieces and, per node, the index of its piece (commas
    and braces are pieces of their own, so they belong to no node)."""
    kids, roots = collections.defaultdict(list), []
    for i, p in enumerate(G.ptr):
        (kids[p] if p >= 0 else roots).append(i)
    pieces, at = ["{"], {}

    def pair(i):
        k, sep, v = G.text[i].partition(": ")
        return " " + json.dumps(k, ensure_ascii=False) + (": " + json.dumps(v, ensure_ascii=False) if sep else "")

    def emit(i, first):
        if not first:
            pieces.append(",")
        pieces.append(pair(i))
        at[i] = len(pieces) - 1
        if kids[i]:
            pieces.append(": {")
            for j, c in enumerate(kids[i]):
                emit(c, j == 0)
            pieces.append(" }")
    for j, r in enumerate(roots):
        emit(r, j == 0)
    pieces.append(" }")
    return pieces, at


class WalkerStateEnc(nn.Module):
    """Token states of a text at the input of Walker's C blocks after one flat R run (= the layer of the
    R pass state). Frozen, cached per text (LRU).

    LoRA (local / small-box adaptation): pass lora_r > 0 to attach low-rank factors to the early
    blocks [0, r_end) and train ONLY those. Then the encoder runs with grad enabled and bypasses the
    cache (the states change every step while the adapter moves)."""

    def __init__(self, model, tok, max_cache=40000, bs=128, lora_blocks=None, lora_r=0, lora_alpha=16.0):
        super().__init__()
        self.model, self.tok, self.max_cache, self.bs = [model], tok, max_cache, bs
        self.cache = collections.OrderedDict()
        self.grad = False
        if lora_r and int(lora_r) > 0:
            import gnm_lora
            if lora_blocks is None:
                lora_blocks = range(0, int(getattr(model, "r_end", len(model.blocks))))
            targets = getattr(model, "lora_targets", None) or gnm_lora.DEFAULT_TARGETS
            wrapped = gnm_lora.attach_lora(model, blocks=lora_blocks, r=int(lora_r),
                                           alpha=lora_alpha, targets=targets)
            gnm_lora.mark_only_lora_trainable(model)
            self.grad = wrapped > 0
            if not self.grad:
                raise RuntimeError("LoRA matched no projection layers to adapt")

    def doc(self, pieces):
        """States of one document (pieces concatenated) and the token range of every piece."""
        ids, rng = [], []
        for pc in pieces:
            t = self.tok.ids(pc)
            rng.append((len(ids), len(ids) + len(t)))
            ids += t
        x = self._states(torch.tensor([ids], device=next(self.model[0].parameters()).device))
        return x[0].float(), rng

    def _run(self, texts):
        seqs = [self.tok.ids(" " + t)[:64] or [0] for t in texts]
        n = max(len(s) for s in seqs)
        ids = torch.tensor([s + [0] * (n - len(s)) for s in seqs], device=next(self.model[0].parameters()).device)
        x = self._states(ids)
        return [x[i, :len(s)].to(torch.bfloat16) for i, s in enumerate(seqs)]

    def _states(self, ids):
        model = self.model[0]
        dev = ids.device
        got = {}

        def pre(_m, args, kwargs):
            got["x"] = args[0] if args else kwargs["x"]
            raise _Stop
        h = model.blocks[model.r_end].register_forward_pre_hook(pre, with_kwargs=True)
        try:
            with torch.set_grad_enabled(self.grad), \
                    torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                model(ids, r_active=False)
        except _Stop:
            pass
        finally:
            h.remove()
        return got["x"]

    def forward(self, texts):
        if self.grad:                                   # states move while the adapter trains: no cache
            with torch.set_grad_enabled(True):
                return self._run(texts)
        for t in texts:
            if t in self.cache:
                self.cache.move_to_end(t)
        miss = sorted({t for t in texts if t not in self.cache})
        for i in range(0, len(miss), self.bs):
            for t, x in zip(miss[i:i + self.bs], self._run(miss[i:i + self.bs])):
                self.cache[t] = x
        out = [self.cache[t] for t in texts]
        while len(self.cache) > self.max_cache:
            self.cache.popitem(last=False)
        return out


class NodeMemV2(nn.Module):
    def __init__(self, enc, d_model, D=N.D):
        super().__init__()
        self.enc = enc
        # Walker's states share one huge direction (cos .97 between different texts): standardize each
        # feature across tokens (running stats, no gradient) so labels can differ (history bug 16)
        self.register_buffer("mu", torch.zeros(d_model))
        self.register_buffer("var", torch.ones(d_model))
        self.register_buffer("seen", torch.zeros(()))
        self.ln = nn.LayerNorm(d_model)
        self.pool = nn.Linear(d_model, 3)                   # attention pooling scores: value, label, relation
        nn.init.zeros_(self.pool.weight)
        nn.init.zeros_(self.pool.bias)
        self.hv, self.hl, self.hr = (nn.Linear(d_model, D) for _ in range(3))
        self.tau = nn.Parameter(torch.tensor(math.log(10.0)))      # label attention temperature (log)
        self.tau_r = nn.Parameter(torch.tensor(math.log(5.0)))     # relation gate temperature (log)
        self.g = nn.Parameter(torch.tensor(0.5))                   # traffic strength
        self.lam = nn.Parameter(torch.tensor(0.0))                 # (used only outside --soft)
        self.C = 0                                                 # no candidate cut
        self.hops = 1
        # ---- relation as an EDGE feature, not a per-node gate ----
        # A learned bilinear over PAIRS of relation vectors gives a soft, directed
        # adjacency: it can express "this node's relation owns / points to that node"
        # (ownership), inverses ("opposite"), or any learned neighbour type, so
        # activation flows between nodes the fixed nesting edges never connect. This is
        # what makes r_n a provider ("A and B together say how they relate") instead of
        # a redundant query gate. Pairwise bilinear (rank D): (R_i @ ea) . (R_j @ eb).
        # w_edge starts at exactly 0, so it is a no-op until training opens it.
        self.ea = nn.Linear(D, D, bias=False)                      # relation -> source role
        self.eb = nn.Linear(D, D, bias=False)                      # relation -> target role
        self.w_edge = nn.Parameter(torch.tensor(0.0))
        # relation as a MATCH feature, not only a traffic gate: the content score gets a direct
        # cos(query, relation) term so a node whose relation agrees with the query is boosted
        # (this is what makes R the relation selector the content lookup needs). 0 => no-op.
        self.w_rel = nn.Parameter(torch.tensor(0.0))
        # ---- relation as a CONDITIONAL OPERATOR (dependence) ----
        # The target of a node depends on its owner and its relation (TransE-style):
        #     pred_k = V[owner(k)] + rop(r_k)          # owner + relation -> the entity it refers to
        # Activation on k then flows to the node whose value pred_k matches, so the walk along
        # value edges becomes a learned, conditional map instead of a rule-made pointer. Change
        # the owner and the predicted target moves with it. w_op starts at 0 (strict no-op).
        self.rop = nn.Linear(D, D, bias=False)
        self.tau_o = nn.Parameter(torch.tensor(math.log(5.0)))     # operator match temperature (log)
        self.w_op = nn.Parameter(torch.tensor(0.0))

    def node_vecs(self, G, fresh):
        """v, l, r for every node of G, from one pass of Walker over the graph serialized as a JSON
        document (fresh is ignored: v2 re-encodes the document)."""
        dev = self.hv.weight.device
        pieces, at = serialize(G)                        # the whole graph as one JSON document
        X, rng = self.enc.doc(pieces)                    # (T, d): every token sees the document before it
        X = X.to(dev)
        spans = [rng[at[i]] for i in range(len(G))]
        n = max(max(e - s for s, e in spans), 1)
        idx = torch.zeros(len(G), n, dtype=torch.long, device=dev)
        m = torch.zeros(len(G), n, dtype=torch.bool, device=dev)
        for i, (a, b) in enumerate(spans):
            idx[i, :b - a] = torch.arange(a, b, device=dev)
            m[i, :b - a] = True
        if self.training:
            with torch.no_grad():
                f = 0.0 if float(self.seen) == 0 else 0.99
                self.mu.mul_(f).add_((1 - f) * X.mean(0))
                self.var.mul_(f).add_((1 - f) * X.var(0))
                self.seen.add_(1)
        H = self.ln((X - self.mu) / (self.var + 1e-5).sqrt())[idx]          # (N, span, d)
        w = torch.softmax(self.pool(H).masked_fill(~m[..., None], -1e4), 1)       # (N, T, 3)
        pv, pl, pr = ((w[..., k:k + 1] * H).sum(1) for k in range(3))
        V, L, R = self.hv(pv), self.hl(pl), self.hr(pr)
        self.last_pl = pl                                   # pooled label-space states: queries made from reads
        par = torch.tensor([p if p >= 0 else i for i, p in enumerate(G.ptr)], device=dev)
        return V, L, R, par

    def score(self, cues, L):
        return self.tau.exp() * F.normalize(cues, dim=-1) @ F.normalize(L, dim=-1).t()

    def relation_adj(self, R):
        """(N,N) directed relation-similarity adjacency, softmaxed per source (row i -> column j).
        Expresses a learned relation TYPE between i and j (ownership / inverse / neighbour)."""
        if self.w_edge is None or R.shape[0] < 2:
            return None
        return torch.softmax(self.ea(R) @ self.eb(R).t() / math.sqrt(R.shape[-1]), -1)

    def operator_adj(self, V, R, par):
        """(N,N) dependence adjacency: node k's target is predicted conditionally as
        pred_k = V[owner(k)] + rop(r_k); activation on k flows to the node whose value best
        matches pred_k. Change the owner and the predicted target moves with it."""
        if self.w_op is None or V is None or V.shape[0] < 2:
            return None
        pred = V[par] + self.rop(R)
        return torch.softmax(self.tau_o.exp() * F.normalize(pred, dim=-1) @ F.normalize(V, dim=-1).t(), -1)

    def message_adj(self, V, R, par):
        """Combined directed message adjacency = relation-similarity edges + conditional dependence
        edges, each scaled by its learnable zero-init weight. One (N,N) tensor, computed ONCE per
        world and reused across the r passes (the per-pass cost is then a single `c @ adj` matmul)."""
        n = R.shape[0]
        if n < 2:
            return None
        adj = None
        if self.w_edge is not None:
            adj = self.w_edge * torch.softmax(self.ea(R) @ self.eb(R).t() / math.sqrt(R.shape[-1]), -1)
        if self.w_op is not None and V is not None:
            pred = V[par] + self.rop(R)
            op = torch.softmax(self.tau_o.exp() * F.normalize(pred, dim=-1) @ F.normalize(V, dim=-1).t(), -1)
            adj = self.w_op * op if adj is None else adj + self.w_op * op
        return adj

    def activate_batch(self, G, L, R, par, cues, force=None, bias=None, V=None, adj=None):
        e = self.score(cues, L)
        if bias is not None:
            e = e + bias
        # Direct relation match: a node whose relation vector r_n agrees with the query is boosted
        # (w_rel starts at 0 => no-op until training opens it). Reuses the cosine the gate needs.
        cos_r = F.normalize(cues, dim=-1) @ F.normalize(R, dim=-1).t()
        if self.w_rel is not None:
            e = e + self.w_rel * cos_r
        alive = torch.tensor(G.alive[:L.shape[0]], device=L.device)
        e = e.masked_fill(~alive, -1e4)
        gate = torch.sigmoid(self.tau_r.exp() * cos_r)
        idx = torch.arange(L.shape[0], device=L.device)
        has = par != idx
        nkids = torch.bincount(par[has], minlength=L.shape[0]).clamp(min=1).float()   # a parent gets the AVERAGE
        # Learned relation/dependence edges: pass the world-cached adjacency (computed once in
        # graph_ctx) to avoid rebuilding the (N,N) tensor every pass; fall back to computing it.
        if adj is None:
            adj = self.message_adj(V, R, par)
        s = e                                                                           # of its children (a sum
        for _ in range(self.hops):                                                      # buried big records)                         # activation along pointers (parent <-> child);
            c = F.softplus(s - s.masked_fill(~alive, 0).sum(-1, keepdim=True) / alive.sum()) - math.log(2.0)
            # smooth (a below-average node still gets gradient), ~0 for an average node
            up = torch.zeros_like(s).index_add(1, par[has], gate[:, has] * c[:, has]) / nkids  # mean of children
            down = torch.zeros_like(s)                                                   # nodes pass it on
            down[:, has] = gate[:, has] * c[:, par[has]]
            s = e + self.g * (up + down)
            if adj is not None:                                       # relation + dependence message edges
                s = s + (c @ adj)
        return s.masked_fill(~alive, -1e4)


def tie_queries(hook, M, h_info=True):
    """Start the query heads as the label head applied to the pass state (one space at step 0), and add
    qr / qi: the next query made from what was just read (fact / instruction reads' pooled states), also
    starting as the label head — after reading the question, the query is the question's content in
    label space."""
    dev = M.hl.weight.device
    hook.qr = nn.Linear(M.hl.in_features, M.hl.out_features).to(dev)
    hook.qi = nn.Linear(M.hl.in_features, M.hl.out_features).to(dev)
    # Option 2: a direct relation->content head, fed by the instruction READ vector (which carries
    # the step's relation), added into c_mem. Without it the relation never reliably reaches the cue
    # (the trace shows gold relation has the lowest label cosine). Small nonzero init so it can act.
    hook.qrel = nn.Linear(N.D, M.hl.out_features).to(dev)
    with torch.no_grad():
        for lin in (hook.qr, hook.qi):
            lin.weight.copy_(M.hl.weight)
            lin.bias.zero_()
        nn.init.normal_(hook.qrel.weight, std=0.02)
        hook.qrel.bias.zero_()
        # the read columns of cq_mem (r_ins, r_prev) start at 0: qr / qi carry what was read (random columns
        # only added noise to every query)
        d = M.hl.weight.shape[1]
        hook.cq_mem.weight.zero_()
        hook.cq_mem.bias.zero_()
        hook.cq_ins.weight.zero_()
        hook.cq_ins.bias.zero_()
        if h_info:   # the pass state carries the question (it is in Walker's input): start as the label head.
            hook.cq_mem.weight[:, :d].copy_(M.hl.weight)   # With the question in memory, Walker's input is
            hook.cq_ins.weight.copy_(M.hl.weight)          # " Answer:" for everyone: h would add one constant
                                                           # vector to every query, so it starts at 0
