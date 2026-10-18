"""GNM paired with Walker's OWN recursion (PROJECT_STATUS 3.8, "Pairing with the LM").

Walker's narrow-R layout: P = blocks [0, r_start) once -> e; R = [r_start, r_end) once flat, then r
extra passes with shared weights; C = the rest, then the LM head. Every extra R pass p builds its
input u = norm(s + proj[s, e]) (`_r_pass_input`). The hook runs right there, inside the pass:

  1. c_ins = head(s_p)                       -> instruction activation over `ins` nodes, plus the
                                                previous pass's instruction activation moved along
                                                its NEXT pointer (the procedure advances);
  2. c_mem = head([s_p, instruction read])   -> GNM activation over content nodes (label match +
                                                relation traffic + fade from pass p-1);
  3. top-K content + top-Ki instruction nodes, weighted by activation, are cross-read by every
     position:  u <- u + gate[p] * cross_read(s, nodes);  then s_{p+1} = s + pass_gain[p] (R(u) - s).

gate starts at exactly 0 (like pass_gain), so attaching the hook leaves the checkpoint's output
bitwise unchanged. "Ask" = the next R pass, "say" = C + LM head. No trainer code is modified: the
hook wraps `_r_pass_input` on the model instance.

Prototype task (laptop, tiny Walker from the trainer): every world has random relations between 40
entities, stored as fact nodes, and 10 procedures stored as 3 instruction nodes
each ("p3 step 2: boss", NEXT -> step 3). Facts are leaves ("boss = e40") under an entity record ("e12"). The LM sees only [P_j, x, =] and must output the entity
reached by running P_j's steps from x: 3 lookups, one per R pass, steered by the instruction nodes.
Eval uses new worlds and step combinations never trained on. Deep supervision per pass: the gold
fact and the gold instruction step.

Run:  python gnm_lm.py --steps 3000
"""
import argparse
import math
import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import gmem_node as N

RELS = ("parent", "city", "boss", "pet", "same")
NE, NP, NSTEP = 40, 10, 3
EQ = NE + NP
VOCAB = EQ + 1
COMBOS = [(a, b, c) for a in RELS for b in RELS for c in RELS]
TEST_COMBOS = set(random.Random(0).sample(COMBOS, len(COMBOS) // 5))
TRAIN_COMBOS = [c for c in COMBOS if c not in TEST_COMBOS]


class GNMHook(nn.Module):
    """Per-R-pass GNM queries and node cross-read (the GNM side of the pairing)."""

    def __init__(self, M, d_model, r_max, K=8, Ki=2, shared_gate=False, norm_reads=False):
        super().__init__()
        self.M, self.K, self.Ki = M, K, Ki
        self.ln = nn.LayerNorm(d_model)
        self.pool = nn.Linear(d_model, 2)                   # span attention for c_ins / c_mem (zero: mean)
        nn.init.zeros_(self.pool.weight)
        nn.init.zeros_(self.pool.bias)
        self.cq_ins = nn.Linear(d_model, N.D)               # c_ins: where am I in the procedure
        self.cq_mem = nn.Linear(d_model + 2 * N.D, N.D)     # c_mem: what content I need (follows the step
                                                            # and what the previous pass just read)
        self.lam_ins = nn.Parameter(torch.tensor(0.0))      # instruction fade / NEXT carry
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(N.D, d_model, bias=False)
        self.v = nn.Linear(N.D, d_model, bias=False)
        self.slot = nn.Embedding(2, d_model)                # content / instruction
        self.gate = nn.Parameter(torch.zeros(1 if shared_gate else int(r_max)))   # starts closed; shared =
                                                            # one gate for every pass (any depth)
        # reads enter the cue at the state's scale (a 2048-wide LayerNormed state would otherwise drown
        # a 128-wide read: on the 1B the relation in the instruction read never reached the cue)
        self.ln_r = nn.LayerNorm(N.D) if norm_reads else None
        # link (Step D): a position in the graph. Pass 1 starts at the record(s) the question names;
        # each pass reads a fact OF the entity it is at; the fact's value moves it to that entity.
        self.link, self.soft = False, False
        # question in memory (--qmem): pass 1 searches for the newest question node (memory's time stamp,
        # learned weight); the question read then moves the position along its mention edges
        self.w_rec = nn.Parameter(torch.tensor(8.0))
        self.qread = None
        self.cq_ins_r = nn.Linear(N.D, N.D)                 # the instruction query also sees what was just read
        nn.init.zeros_(self.cq_ins_r.weight)                # (zero: old runs unchanged)
        nn.init.zeros_(self.cq_ins_r.bias)
        self.ctx = None
        self.p, self.prev, self.trace = 0, None, []

    def begin(self, ctx):
        """ctx: G, V, Q, R, par (from M.node_vecs), ins (N,) bool, nxt (src, dst) tensors,
        cue_pos (B,), optional force_mem[p][b] gold ids (training: gold reaches stage 2)."""
        self.ctx, self.p, self.prev, self.trace, self.qread = ctx, 0, None, [], None

    def end(self):
        self.ctx = None

    def step(self, s_in, u):
        p = self.p
        self.p += 1
        c = self.ctx
        if c is None:
            return u
        with torch.autocast(s_in.device.type, enabled=False):
            out = self._read(s_in.float(), c, p)
        g = self.gate[min(p, self.gate.numel() - 1)]
        return u + (g * out).to(u.dtype)

    def _read(self, s, c, p):
        M, B = self.M, s.shape[0]
        V, Q, R, par, ins = c["V"], c["Q"], c["R"], c["par"], c["ins"]
        hs = self.ln(s)                                     # cues pool the question span (positions <= cue_pos)
        span = torch.arange(s.shape[1], device=s.device)[None] <= c["cue_pos"][:, None]
        w = torch.softmax(self.pool(hs).masked_fill(~span[..., None], -1e4), 1)      # (B, T, 2)
        h_ins, h = (w[..., 0:1] * hs).sum(1), (w[..., 1:2] * hs).sum(1)
        qmem = self.link and c.get("qnodes") is not None    # the question lives in memory
        search = qmem and self.prev is None                 # pass 1: find the question first
        # 1. instruction: label match over `ins` nodes + previous step carried along NEXT
        if c.get("no_ins") or search:                       # ablation: no instruction memory at all
            a_ins = torch.zeros(B, V.shape[0], device=s.device)
            ki = min(self.Ki, V.shape[0])
            wi, ii = a_ins[:, :ki], torch.zeros(B, ki, dtype=torch.long, device=s.device)
            r_ins = torch.zeros(B, V.shape[1], device=s.device)
            rs_ins = None
        else:
            c_ins = self.cq_ins(h_ins) + (self.cq_ins_r(self.prev[2]) if self.prev is not None else 0.0)
            rs_prev = self.prev[3] if self.prev is not None and len(self.prev) > 3 else None
            if getattr(self, "qr", None) is not None and rs_prev is not None:
                c_ins = c_ins + self.qr(rs_prev)            # v2: query from what was read, in label space
            e_ins = (M.score(c_ins, Q) if hasattr(M, "score") else M.W(c_ins) @ Q.t() / math.sqrt(N.D)
                     ).masked_fill(~ins, -1e4)
            if self.link:                                   # position: first step of the named procedure,
                if self.prev is None:                       # then along NEXT
                    pos_i = c.get("pos0_ins")
                elif self.qread is not None:                # the question just read names a procedure
                    pos_i = None                            # (label-only: found by the query itself)
                    if c.get("M_ins") is not None:
                        pos_i = self.qread @ c["M_ins"]
                        c["has_ins"] = pos_i.sum(-1, keepdim=True).clamp(max=1)
                else:
                    src, dst = c["nxt"]
                    pi = self.prev[1] if self.soft else self.prev[1].detach()
                    pos_i = torch.zeros_like(pi).index_add(1, dst, pi[:, src])
                if pos_i is not None:
                    e_ins = e_ins + torch.log(pos_i + 1e-6)
                a_ins = torch.softmax(e_ins, -1)
            else:
                lam = torch.sigmoid(self.lam_ins)
                a_ins = (1 - lam) * torch.softmax(e_ins, -1)
                if self.prev is not None:
                    src, dst = c["nxt"]
                    a_ins = a_ins + lam * torch.zeros_like(a_ins).index_add(1, dst, self.prev[1][:, src])
            wi, ii = torch.topk(a_ins, min(self.Ki, a_ins.shape[1]), -1)
            r_ins = (wi[..., None] * V[ii]).sum(1) / wi.sum(-1, keepdim=True).clamp(min=1e-6)
            rs_ins = (None if c.get("Pl") is None else
                      (wi[..., None] * c["Pl"][ii]).sum(1) / wi.sum(-1, keepdim=True).clamp(min=1e-6))
            if self.link and c.get("has_ins") is not None:  # the question names no procedure: no read
                r_ins, wi = r_ins * c["has_ins"], wi * c["has_ins"]
        # 2. content: GNM activation from c_mem, which follows the active instruction's value
        r_prev = self.prev[2] if self.prev is not None else torch.zeros_like(r_ins)
        if self.ln_r is not None:
            r_ins, r_prev = self.ln_r(r_ins) * math.sqrt(h.shape[-1] / N.D), self.ln_r(r_prev) * math.sqrt(h.shape[-1] / N.D)
        c_mem = self.cq_mem(torch.cat([h, r_ins, r_prev], -1))
        rs_prev = self.prev[3] if self.prev is not None and len(self.prev) > 3 else None
        if getattr(self, "qr", None) is not None:          # v2: the next query starts from what was read
            if rs_prev is not None:                         # (the read nodes' pooled states, label space)
                c_mem = c_mem + self.qr(rs_prev)
            if not (c.get("no_ins") or search) and rs_ins is not None:
                c_mem = c_mem + self.qi(rs_ins)
        # Option 2: fold the instruction read (which carries this step's relation) directly into the
        # content query, so the relation can select the right fact of the (already found) entity.
        if getattr(self, "qrel", None) is not None and not (c.get("no_ins") or search):
            c_mem = c_mem + self.qrel(r_ins)
        force = c.get("force_mem")
        force = None if force is None else torch.tensor(force[p], device=s.device)
        prev_a = None if self.prev is None else (self.prev[0] if self.soft else self.prev[0].detach())
        bias = None
        if self.link:
            if search:                                      # the newest question (time stamp, learned weight)
                pos, bias = None, self.w_rec * c["recency"]
            elif prev_a is None:                            # pass 1: where the question points
                pos = c.get("pos0")
            elif self.qread is not None:                    # the records the question read names
                pos = self.qread @ c["M_rec"] if c.get("M_rec") is not None else None   # label-only: none
            else:                                           # the entity each fact just read refers to
                src, dst = c["ref"]
                pos = torch.zeros_like(prev_a).index_add(1, dst, prev_a[:, src])
            if pos is not None:                             # P(fact) ~ P(at its owner) * P(relation | cue)
                bias = torch.log(pos[:, par] * c["child"] + 1e-6)
        if self.soft and prev_a is not None:               # what was just read is not re-read next pass
            rb = torch.log1p(-prev_a.clamp(max=1 - 1e-4))   # (a query made from a read starts as that node's
            bias = rb if bias is None else bias + rb         # own label, so it would re-find it)
        if c.get("noise") and not search:
            # k noisy trajectories (training): Gumbel exploration on the CONTENT selection only.
            # NEVER on the question-search pass -- randomizing it mis-finds the question and kills the
            # lexical instruction seed (ins -> 0), collapsing the whole chain. Scale ~1 barely moves
            # scores of magnitude ~tau (~10); a larger scale genuinely explores the content path.
            g = -torch.log(-torch.log(torch.rand(B, V.shape[0], device=s.device).clamp(1e-9, 1 - 1e-9)))
            g = g * float(c.get("noise_scale", 1.0))
            bias = g if bias is None else bias + g
        sc = M.activate_batch(c["G"], Q, R, par, c_mem, force, bias, V=V,
                              adj=c.get("adj")).masked_fill(ins, -1e4)
        if qmem and not search:      # questions are found by the search pass, then read; like instruction
            sc = sc.index_fill(1, c["qnodes"], -1e4)     # nodes they are not content for later passes
        if self.soft:          # no carry: the pass-to-pass link is the entity edge (a carried activation
            a_mem = torch.softmax(sc, -1)   # would make the final read repeat an earlier pass's node)
        else:
            lm = torch.sigmoid(M.lam)
            a_mem = (1 - lm) * torch.softmax(sc, -1)
            if prev_a is not None:
                a_mem = a_mem + lm * prev_a                # chain mode: each pass trained on its own target
        wm_, im_ = torch.topk(a_mem, min(self.K, a_mem.shape[1]), -1)
        self.trace.append({"a_mem": a_mem, "a_ins": a_ins, "h": h,      # r: what this pass read (halting)
                           "r": (wm_[..., None] * V[im_]).sum(1) / wm_.sum(-1, keepdim=True).clamp(min=1e-6)})
        self.qread = (a_mem if self.soft else a_mem.detach())[:, c["qnodes"]] if search else None
        # 3. weighted top-K nodes enter the pass: every position cross-reads them
        wm, im = torch.topk(a_mem, min(self.K, a_mem.shape[1]), -1)   # K = all nodes in --soft:
        # the model receives every node in proportion to its activation (relation-gated traffic decides
        # which pointed nodes come along); no hand-set count
        r_mem = (wm[..., None] * V[im]).sum(1) / wm.sum(-1, keepdim=True).clamp(min=1e-6)
        rs_mem = (None if c.get("Pl") is None else
                  (wm[..., None] * c["Pl"][im]).sum(1) / wm.sum(-1, keepdim=True).clamp(min=1e-6))
        self.prev = (a_mem, a_ins, r_mem, rs_mem)
        nodes = torch.cat([V[im], V[ii]], 1)
        w = torch.cat([wm, wi], 1)
        sl = self.slot.weight[torch.tensor([0] * wm.shape[1] + [1] * wi.shape[1], device=s.device)]
        keys, vals = self.k(nodes) + sl, self.v(nodes) + sl
        logw = torch.log((w / w.max(-1, keepdim=True).values.clamp(min=1e-9)).clamp(min=1e-9))
        att = torch.softmax(self.q(hs) @ keys.transpose(1, 2) / math.sqrt(s.shape[-1])
                            + logw[:, None, :], -1)
        return att @ vals


def attach(model, hook):
    """Route every extra R pass of `model` through `hook` (instance-level, trainer untouched)."""
    orig = model._r_pass_input

    def wrapped(r_adapter, s_in, e_state):
        return hook.step(s_in, orig(r_adapter, s_in, e_state))
    model._r_pass_input = wrapped
    return model


def forward(trainer, model, hook, idx, ctx, r):
    """One LM forward with exactly r extra R passes (all with gradient in training)."""
    hook.begin(ctx)
    saved = trainer._sample_r_counts
    trainer._sample_r_counts = lambda *a, **k: (0, int(r))
    try:
        logits, _ = model(idx, r_active=True)
    finally:
        trainer._sample_r_counts = saved
        hook.end()
    return logits


# ============================================================ task
def make_world(rng, split, nstep=NSTEP, var_len=False):
    f = {rel: (list(range(NE)) if rel == "same" else [rng.randrange(NE) for _ in range(NE)])
         for rel in RELS}
    pool = TRAIN_COMBOS if split == "train" else sorted(TEST_COMBOS)
    def proc():
        L = rng.randint(1, 4) if var_len else nstep
        return rng.choice(pool) if L == NSTEP else tuple(rng.choice(RELS) for _ in range(L))
    procs = [proc() for _ in range(NP)]                     # length 3 uses the train / held-out split
    G = N.Graph(cap=10 ** 6)
    fid, sid, nxt = {}, {}, []
    for e in range(NE):                      # entity record; its facts are leaves pointing to it
        r = G.add(f"e{e}", "record")
        for rel in RELS:
            fid[(e, rel)] = G.add(f"{rel} = e{f[rel][e]}", "leaf", ptr=r)
    for j, combo in enumerate(procs):
        prev = -1
        for k, rel in enumerate(combo):
            i = G.add(f"p{j} step {k + 1}: {rel}", "ins")
            sid[(j, k)] = i
            if prev >= 0:
                nxt.append((prev, i))
            prev = i
        d = G.add(f"p{j} step {len(combo) + 1}: done", "ins")   # end of the procedure: hold
        sid[(j, len(combo))] = d
        nxt += [(prev, d), (d, d)]
    return {"f": f, "procs": procs, "G": G, "fid": fid, "sid": sid, "nxt": nxt}


def questions(rng, w, B, r=None):
    L = max(len(c) for c in w["procs"]) if r is None else r
    idx, ans, gm, gi = [], [], [[] for _ in range(L)], [[] for _ in range(L)]
    for _ in range(B):
        j, x = rng.randrange(NP), rng.randrange(NE)
        cur = x
        for k, rel in enumerate(w["procs"][j]):
            if k < L:
                gm[k].append(w["fid"][(cur, rel)])
                gi[k].append(w["sid"][(j, k)])
            cur = w["f"][rel][cur]
        n = len(w["procs"][j])
        for k in range(n, L):                               # past the end: hold the last fact, step = done
            gm[k].append(gm[n - 1][-1])
            gi[k].append(w["sid"][(j, n)])
        idx.append([NE + j, x, EQ])
        ans.append(cur)
    return idx, ans, gm, gi


def graph_ctx(M, w, dev, B):
    G = w["G"]
    G.V = G.V.to(dev)
    V, Q, R, par = M.node_vecs(G, list(range(len(G))))      # a new world: encode every node
    Pl = getattr(M, "last_pl", None)                        # v2: pooled label-space states (query from reads)
    ins = torch.tensor([k == "ins" for k in G.kind], device=dev)
    src = torch.tensor([a for a, _ in w["nxt"]], device=dev, dtype=torch.long)
    dst = torch.tensor([b for _, b in w["nxt"]], device=dev, dtype=torch.long)
    # entity edges: a leaf "rel: X" refers to the record named X (entity resolution at ingestion)
    name = {t: i for i, (t, k) in enumerate(zip(G.text, G.kind)) if k == "record"}
    ref = [(i, name[t.split(": ", 1)[1]]) for i, (t, k) in enumerate(zip(G.text, G.kind))
           if k == "leaf" and ": " in t and t.split(": ", 1)[1] in name]
    ref = (torch.tensor([a for a, _ in ref], device=dev, dtype=torch.long),
           torch.tensor([b for _, b in ref], device=dev, dtype=torch.long))
    child = (par != torch.arange(par.shape[0], device=dev)).float()
    ctx = {"G": G, "V": V, "Q": Q, "R": R, "par": par, "ins": ins, "nxt": (src, dst), "ref": ref, "child": child,
           "Pl": Pl,
           "cue_pos": torch.full((B,), 2, device=dev, dtype=torch.long),
           "no_ins": not bool(ins.any())}                  # no instruction nodes: read none
    # Relation/dependence message edges depend only on (V, R, par): build them ONCE for this world
    # (reused by every R pass) instead of rebuilding the (N,N) tensor per pass.
    ctx["adj"] = M.message_adj(V, R, par) if hasattr(M, "message_adj") else None
    return ctx


def lex_matrix(seqs, w, tok, dev):
    """Word path: (B, N) share of each node's tokens (IDF-weighted over the graph) that occur in the
    question. Record "Bob Smith" asked about by name -> 1.0; "boss: Bob Smith" -> partial."""
    G = w["G"]
    if "ntok" not in w:
        w["ntok"] = [set(tok.ids(" " + t)) for t in G.text]
        df = {}
        for s in w["ntok"]:
            for t in s:
                df[t] = df.get(t, 0) + 1
        w["idf"] = {t: math.log(1 + len(G) / c) for t, c in df.items()}
    idf, out = w["idf"], torch.zeros(len(seqs), len(G))
    for b, q in enumerate(seqs):
        qs = set(q)
        for n, s in enumerate(w["ntok"]):
            hit = s & qs
            if hit:
                out[b, n] = sum(idf[t] for t in hit) / sum(idf[t] for t in s)
    return out.to(dev)


def start_pos(lex, G):
    """Where the question points: a distribution over the records its words name (a full name match
    dominates a shared first name); zero if it names none."""
    rec = torch.tensor([k == "record" for k in G.kind], device=lex.device)
    s = (8 * lex).masked_fill(~rec | (lex <= 0), -1e4)
    return torch.softmax(s, -1) * (s.max(-1, keepdim=True).values > -1e3)


def start_ins(lex, G, nxt):
    """Instruction start: the first step (no NEXT into it) of the procedure headers the question
    names (word share >= 0.4: "father" alone does not name "father-in-law"), weighted like start_pos;
    zero if it names none."""
    par = [p if p >= 0 else i for i, p in enumerate(G.ptr)]
    hdr = torch.tensor([k == "ins" and par[i] == i for i, k in enumerate(G.kind)], device=lex.device)
    s = (8 * lex).masked_fill(~hdr | (lex < 0.4), -1e4)
    ph = torch.softmax(s, -1) * (s.max(-1, keepdim=True).values > -1e3)
    into = {b for _, b in nxt}
    first = [i for i, k in enumerate(G.kind) if k == "ins" and par[i] != i and i not in into]
    out = torch.zeros_like(lex)
    if first:
        out[:, first] = ph[:, [par[i] for i in first]]
    return out


TINY_ENV = {
    "TB_ATTN_EVERY": "1", "TB_HYBRID": "1", "TB_WALK_LOOP": "1",
    "TB_WALK_NORM": "0.3", "TB_WALK_NORM_MODE": "clip", "TB_WALK_HEADS": "4",
    "TB_WALK_FP32": "1", "TB_WALK_MIX": "0", "TB_WALK_GLOB": "1",
    "TB_WALK_COUPLE": "1", "TB_WALK_GATE": "1", "TB_PHASE_B_SUPERSET": "0",
    "TB_POS_EMB": "0", "TB_FORCE_QUANT": "none", "TB_MEM": "0", "TB_NI": "0",
    "TB_LMM": "0", "TB_KMERGE": "0", "TB_LEXICAL_SEMANTIC": "0",
    "TB_LOCAL_CONV": "0", "TB_LOOP_ENABLE": "0",
    "TB_R_ENABLE": "1", "TB_R_START": "1", "TB_R_END": "2", "TB_R_MEAN": "3",
    "TB_R_MAX": "8", "TB_R_BACKPROP": "3", "TB_R_INSTR_ONLY": "0",
}


def build_tiny(trainer, D=128, L=3, dev="cpu", shared_gain=False, max_t=8):
    """Tiny narrow-R Walker from the trainer (P=[0,1) R=[1,2) C=[2,3)), as r_depth_ab.py builds it."""
    os.environ.update(TINY_ENV)
    os.environ["TB_R_GAIN"] = "shared" if shared_gain else "per_pass"
    cfg = trainer.Cfg()
    cfg.vocab, cfg.D, cfg.L, cfg.n_head = VOCAB, D, L, 4
    cfg.mult, cfg.hybrid, cfg.ckpt, cfg.no_walk = 2.0, True, False, False
    cfg.max_t, cfg.alibi, cfg.rope = max_t, False, True
    cfg.ni = cfg.lmm = cfg.local_conv = cfg.lexical_semantic = False
    cfg.quant_backend, cfg.fp4 = "ste", False
    return trainer.build_model(cfg).to(dev)


def run_batch(trainer, model, hook, M, w, rng, B, dev, train, r=NSTEP, gate_off=False, ins_passes=99):
    idx, ans, gm, gi = questions(rng, w, B, r)
    ctx = graph_ctx(M, w, dev, B)
    if train:
        ctx["force_mem"] = gm
    x = torch.tensor(idx, device=dev)
    y = torch.tensor(ans, device=dev)
    saved = None
    if gate_off:
        saved = hook.gate.data.clone()
        hook.gate.data.zero_()
    with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
        logits = forward(trainer, model, hook, x, ctx, r)
    if saved is not None:
        hook.gate.data.copy_(saved)
    lm = F.cross_entropy(logits[:, -1].float(), y)
    ds, hit_m, hit_i = 0.0, [], []
    chain = torch.ones(B, device=dev)          # 1 while every earlier lookup of this question was right
    for p, t in enumerate(hook.trace[:len(gm)]):
        gmt = torch.tensor(gm[p], device=dev)
        git = torch.tensor(gi[p], device=dev)
        ar = torch.arange(B, device=dev)
        # content of pass p is only learnable when pass p-1 found its fact (its value names the next
        # entity); supervising unreachable targets swamps pass 0 through the shared parameters
        nll = -torch.log(t["a_mem"][ar, gmt].clamp(min=1e-9))
        ds = ds + (nll * chain).sum() / B         # per batch, not per surviving chain
        if p < ins_passes:                                  # averaged over passes: summed, many "done"
            n_ins = min(ins_passes, len(gm))                # targets swamp the one content term
            ds = ds - torch.log(t["a_ins"][ar, git].clamp(min=1e-9)).mean() / n_ins
        hit = (t["a_mem"].argmax(-1) == gmt).float()
        chain = chain * hit
        hit_m.append(float(hit.mean()))
        hit_i.append(float((t["a_ins"].argmax(-1) == git).float().mean()))
    acc = float((logits[:, -1].argmax(-1) == y).float().mean())
    return lm, ds, acc, hit_m, hit_i


def main():
    P = argparse.ArgumentParser()
    P.add_argument("--steps", type=int, default=3000)
    P.add_argument("--batch", type=int, default=32)
    P.add_argument("--lr", type=float, default=1e-3)
    P.add_argument("--ds", type=float, default=1.0, help="deep-supervision weight (per-pass nodes)")
    P.add_argument("--gain-init", type=float, default=0.3,
                   help="tiny model starts R pass gains here (a fresh model's 0 would keep the gate dead)")
    P.add_argument("--eval-every", type=int, default=500)
    P.add_argument("--lm-weight", type=float, default=1.0, help="weight of the LM answer loss")
    P.add_argument("--clip", type=float, default=1.0, help="grad-norm clip (0 = off)")
    P.add_argument("--gate-lr-mult", type=float, default=30.0,
                   help="gate LR multiplier (gate starts at 0, like pass_gain; its own LR avoids a dead start)")
    P.add_argument("--ins-passes", type=int, default=99, help="passes whose instruction step is supervised")
    P.add_argument("--depth", action="store_true",
                   help="any-depth recursion: procedures of 1-4 steps + done node, r = longest + 0..2, "
                        "one shared gate and one shared pass_gain for every pass")
    P.add_argument("--shared", action="store_true", help="only the shared gate + shared pass_gain part of --depth")
    P.add_argument("--var-len", action="store_true", help="only the variable-length / done / random-r part of --depth")
    P.add_argument("--opt", default="compensated", choices=("compensated", "adamw"), help="LM optimizer")
    P.add_argument("--seed", type=int, default=0)
    P.add_argument("--save", default="runs/gnm_lm.pt")
    A = P.parse_args()
    import train_walker_1b_continue as trainer
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(A.seed)
    rng = random.Random(A.seed)
    A.shared, A.var_len = A.shared or A.depth, A.var_len or A.depth
    model = build_tiny(trainer, dev=dev, shared_gain=A.shared)
    with torch.no_grad():
        model.r_adapter.pass_gain.fill_(A.gain_init)
    M = N.NodeMemory().to(dev)
    hook = GNMHook(M, model.emb.weight.shape[1], model.r_adapter.pass_gain.numel(), shared_gate=A.shared).to(dev)
    attach(model, hook)
    opt_lm = (trainer.CompensatedAdamW(model.parameters(), lr=A.lr, betas=(0.9, 0.95), weight_decay=0.01)
              if A.opt == "compensated" else torch.optim.AdamW(model.parameters(), lr=A.lr, weight_decay=0.01))
    rest = [q for n, q in hook.named_parameters() if n != "gate"]          # hook.M included
    opt_g = torch.optim.AdamW([{"params": rest}, {"params": [hook.gate], "lr": A.lr * A.gate_lr_mult,
                                                    "weight_decay": 0.0}], lr=A.lr, weight_decay=0.01)
    print(f"device={dev} vocab={VOCAB} train combos={len(TRAIN_COMBOS)} held-out={len(TEST_COMBOS)}", flush=True)
    t0 = time.time()
    base = [[g["lr"] for g in o.param_groups] for o in (opt_lm, opt_g)]
    best = -1.0
    for step in range(1, A.steps + 1):
        f = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - 1) / A.steps))   # cosine to 10%
        for o, lrs in zip((opt_lm, opt_g), base):
            for g, lr in zip(o.param_groups, lrs):
                g["lr"] = lr * f
        model.train()
        hook.train()
        w = make_world(rng, "train", var_len=A.var_len)
        r = max(len(c) for c in w["procs"]) + (rng.randint(0, 2) if A.var_len else 0)
        lm, ds, acc, hm, hi = run_batch(trainer, model, hook, M, w, rng, A.batch, dev, True, r=r,
                                        ins_passes=A.ins_passes)
        loss = A.lm_weight * lm + A.ds * ds
        opt_lm.zero_grad(set_to_none=True)
        opt_g.zero_grad(set_to_none=True)
        loss.backward()
        if A.clip > 0:
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(hook.parameters()), A.clip)
        opt_lm.step()
        opt_g.step()
        if step % 100 == 0:
            print(f"step {step} lm {float(lm):.3f} ds {float(ds):.3f} acc {acc:.2f} "
                  f"mem {[round(v, 2) for v in hm]} ins {[round(v, 2) for v in hi]} "
                  f"gate {[round(v, 3) for v in hook.gate[:NSTEP].tolist()]} "
                  f"gain {[round(v, 3) for v in model.r_adapter.pass_gain[:NSTEP].float().tolist()]} "
                  f"[{time.time() - t0:.0f}s]", flush=True)
        if step % A.eval_every == 0 or step == A.steps:
            score = evaluate(trainer, model, hook, M, dev, A.batch, step)["held-out"]
            if A.save and score > best:                    # keep the best checkpoint by held-out answer
                best = score
                os.makedirs(os.path.dirname(A.save) or ".", exist_ok=True)
                torch.save({"model": model.state_dict(), "hook": hook.state_dict(), "step": step,
                            "held_out": score}, A.save)
                print(f"saved best @{step}: held-out answer {score:.2f}", flush=True)


@torch.no_grad()
def evaluate(trainer, model, hook, M, dev, B, step, n_worlds=20):
    model.eval()
    hook.eval()
    out = {}
    for name, kw in (("held-out", {}), ("gate off", {"gate_off": True}), ("r=1", {"r": 1})):
        rng = random.Random(4242)
        accs, hms, his = [], [], []
        for _ in range(n_worlds):
            w = make_world(rng, "eval")
            _, _, acc, hm, hi = run_batch(trainer, model, hook, M, w, rng, B, dev, False, **kw)
            accs.append(acc)
            hms.append(hm)
            his.append(hi)
        mean = lambda xs: sum(xs) / max(1, len(xs))
        out[name] = mean(accs)
        print(f"EVAL step {step} {name}: answer {mean(accs):.2f} | per-pass gold fact top-1 "
              f"{[round(mean([h[p] for h in hms if p < len(h)]), 2) for p in range(NSTEP)]} | "
              f"instruction step top-1 {[round(mean([h[p] for h in his if p < len(h)]), 2) for p in range(NSTEP)]}",
              flush=True)
    return out


if __name__ == "__main__":
    main()
