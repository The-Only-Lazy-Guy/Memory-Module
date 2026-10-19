"""GNM on the design (docs/GNM.md section 9.13; design section 1, rules 1.5, section 2, section 7 corrections).

ONE world for the whole run. Each step the model sees small input JSON frames: components of the world (records of
entities; later frames add entities and update values) and question frames {"question": "..."}. A frame is encoded
once by the smart encoder (Walker's early blocks) and dropped; every pair becomes a node {label l, value v (+ value
slots), relation r, pointer to its record's label node, write gate g, usage, time}. The frames' token states are
kept and every node is recomputed each step with the current heads (section 2.2: no stale vectors).

Walker controls the loop. Pass p: Walker reads the latent tokens of what the loop has read so far (no text) and its
state forms the query; activation = tau*cos(q, l) + log g - supersession + w_rel*cos(q, r) + age bias, with
relation-gated pointer traffic and the 1.2.1 relation / dependence edges (all zero at init); ONE choice among nodes,
tool nodes {l, r, network} and HALT. READ adds the node to the registers (and may EDIT it, gated); APPLY runs the tool
network on the (label, value) of the registers it points at and writes the output back as a node; HALT -> Walker's
own head says the last register. Questions are ordinary nodes; batched questions behave as sequential frames.

Supervision: the CoT as a demonstration (per-pass action + arguments), Walker says each tool output and the answer.

    python gnm_world.py --hf gpt2 --steps 3000 --init runs/aj_comb2.pt
"""
import argparse
import copy
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

import fwm_data as D

TOOLS = [("add", lambda a, b: a + b, "the sum of {0} and {1}"),
         ("sub", lambda a, b: a - b, "the difference between {0} and {1}"),
         ("max", max, "the larger of {0} and {1}"),
         ("min", min, "the smaller of {0} and {1}")]
HALT = len(TOOLS)
HELD_PAIRS = {("max", "sub"), ("min", "add")}


# ====================================================================== the world (ground truth + input frames)
class World:
    """Truth of the one world: entities with numeric and entity-valued attributes. Emits small component frames."""

    def __init__(self, rng, n_num=40, n_ent=6, n_groups=4, max_ents=150):
        self.max_ents = max_ents
        used = set()
        mk = lambda: D._style(D._key(rng, D.KEYS_TRAIN, used), rng)
        self.num_keys = [mk() for _ in range(n_num)]
        self.ent_keys = [mk() for _ in range(n_ent)]
        self.groups = [mk() for _ in range(n_groups)]
        self.names, self.ents, self.version = set(), {}, {}

    def component(self, rng, p_new=0.5, p_update=0.3, only_update=False):
        """One small frame: 1-2 records; each a new entity or an existing one with new or updated attributes."""
        recs, facts = [], []
        for _ in range(rng.randint(1, 2)):
            if only_update:
                cands = sorted(n for n, e in self.ents.items() if e)
                nm = rng.choice(cands)
                k = rng.choice(sorted(self.ents[nm]))
                v = (rng.choice(sorted(set(self.ents) - {nm})) if k in self.ent_keys
                     else rng.choice([x for x in range(1, 31) if x != self.ents[nm][k]]))
                self.ents[nm][k] = v
                self.version[(nm, k)] = self.version.get((nm, k), 0) + 1
                facts.append((nm, k, v))
                recs.append({"name": nm, k: v})
                continue
            if not self.ents or (rng.random() < p_new and len(self.ents) < self.max_ents):
                nm = D._name(rng, 2, self.names)
                self.ents[nm] = {}
            else:
                nm = rng.choice(sorted(self.ents))
            e = self.ents[nm]
            rec = {"name": nm}
            for _ in range(rng.randint(1, 3)):
                upd = [k for k in e if rng.random() < p_update]
                if upd and rng.random() < 0.5:
                    k = rng.choice(upd)
                else:
                    k = rng.choice(self.num_keys + self.ent_keys)
                if k in self.ent_keys:
                    others = sorted(set(self.ents) - {nm})
                    if not others:
                        continue
                    v = rng.choice(others)
                else:
                    v = rng.randint(1, 30)
                if k in rec:
                    continue
                rec[k] = v
                e[k] = v
                self.version[(nm, k)] = self.version.get((nm, k), 0) + 1
                facts.append((nm, k, v))
            recs.append(rec)
        return {rng.choice(self.groups): recs}, facts


def leaf(name, key):
    return f"the {D.words(key)} of {name}"


def render(t):
    if t[0] == "leaf":
        return leaf(t[1], t[2])
    tpl = next(x for x in TOOLS if x[0] == t[1])[2]
    return tpl.format(*[f"({render(c)})" if c[0] == "op" else render(c) for c in t[2]])


def has_held(t):
    if t[0] == "leaf":
        return False
    return any(c[0] == "op" and (t[1], c[1]) in HELD_PAIRS for c in t[2]) or any(has_held(c) for c in t[2])


def _tree(rng, leaves, n):
    if n == 0:
        return ("leaf",) + rng.choice(leaves)
    left = rng.randint(0, n - 1)
    return ("op", rng.choice(TOOLS)[0], [_tree(rng, leaves, left), _tree(rng, leaves, n - 1 - left)])


def rand_tree(rng, leaves, n, mode):
    """mode train: held-out compositions excluded; held: at least one held-out composition; else any."""
    for _ in range(200):
        t = _tree(rng, leaves, n)
        if (mode == "train" and has_held(t)) or (mode == "held" and not has_held(t)):
            continue
        return t
    return None


# ====================================================================== the graph store (one world, never reset)
class Store:
    """Frames' token states + node table. Nodes are recomputed every step from the stored states (section 2.2)."""

    def __init__(self):
        self.next_fid = 0
        self.frames = {}          # fid -> {"ids", "X", "nodes"}
        self.nodes = []           # nid -> dict
        self.time = 0
        self.tools_out = {}       # nid -> (k, v, r, slots) detached tool-output vectors

    def add_frame(self, ids, X, enc, leaves, kind, meta_fn):
        fid = self.next_fid
        self.next_fid += 1
        self.time += 1
        piece_leaf = {lf[2]: j for j, lf in enumerate(leaves)}
        base = len(self.nodes)
        nids = []
        for j, lf in enumerate(leaves):
            par = next((piece_leaf[a] for a in lf[3] if a in piece_leaf and piece_leaf[a] != j), j)
            self.nodes.append({"frame": fid, "j": j, "par": base + par, "time": self.time, "alive": True, "usage": 0,
                               "kind": kind, "meta": meta_fn(lf), "edit": None, "nslots": len(enc["vpos"][j])})
            nids.append(base + j)
        self.frames[fid] = {"ids": ids, "X": X, "enc": enc, "nodes": nids}
        return nids

    def alive(self):
        return [i for i, n in enumerate(self.nodes) if n["alive"]]

    def evict(self, cap, protect):
        live = self.alive()
        if len(live) <= cap:
            return []
        # usage stats decide (section 1.1): questions first, then least-used oldest facts
        order = sorted(live, key=lambda i: (self.nodes[i]["kind"] != "question", self.nodes[i]["usage"],
                                            self.nodes[i]["time"]))
        gone = [i for i in order if i not in protect][:len(live) - cap]
        for i in gone:
            self.nodes[i]["alive"] = False
            self.tools_out.pop(i, None)
        for fid in [f for f, fr in self.frames.items() if not any(self.nodes[i]["alive"] for i in fr["nodes"])]:
            del self.frames[fid]
        return gone


# ====================================================================== model parts
class Graph(nn.Module):
    """The node heads on top of fwm_core.Memory (label = key, value, slots, gain) + relation r, edges, edits."""

    def __init__(self, mem, d, dk):
        super().__init__()
        self.mem = mem
        di = 2 * d
        import fwm_core as C
        self.pr = C.Pool(di)
        self.Wr = nn.Linear(di, dk)
        self.ea = nn.Linear(dk, dk, bias=False)            # 1.2.1 (a) relation-similarity edges
        self.eb = nn.Linear(dk, dk, bias=False)
        self.w_edge = nn.Parameter(torch.tensor(0.0))
        self.rop = nn.Linear(dk, dk, bias=False)           # 1.2.1 (b) dependence edges pred_k = V[owner] + rop(r_k)
        self.tau_o = nn.Parameter(torch.tensor(math.log(5.0)))
        self.w_op = nn.Parameter(torch.tensor(0.0))
        self.w_rel = nn.Parameter(torch.tensor(0.0))       # relation match term
        self.tau_r = nn.Parameter(torch.tensor(math.log(5.0)))
        self.g_traffic = nn.Parameter(torch.tensor(0.0))   # pointer traffic (opened by training)
        self.ow = nn.Parameter(torch.tensor([5.0, -8.0]))  # overwrite gate: supersession by newer matching nodes
        self.age = nn.Parameter(torch.zeros(16))           # node stat: age relative to the current frame

    def nodes(self, X, E, spans, ptr, vpos):
        kraw, v, logg, slots, g = self.mem.write(X, E, spans, ptr, vpos)
        r = self.Wr(self.pr.flat(*self.mem.last_flat))              # relation: own pooling over the same span
        return kraw, v, logg, slots, r

    def adj(self, V, R, par):
        A = self.w_edge * torch.softmax(self.ea(R) @ self.eb(R).t() / math.sqrt(R.shape[-1]), -1)
        pred = V[par] + self.rop(R)
        return A + self.w_op * torch.softmax(self.tau_o.exp() * F.normalize(pred, dim=-1) @ F.normalize(V, dim=-1).t(), -1)


class ToolNode(nn.Module):
    """Tool node {label l, relation r, small network}: processes the (label, value) of its arguments -> a new node."""

    def __init__(self, dk, ns):
        super().__init__()
        self.ns = ns
        self.label = nn.Parameter(torch.randn(dk) / dk ** 0.5)
        self.r = nn.Parameter(torch.randn(dk) / dk ** 0.5)
        din = 2 * (2 + ns) * dk
        self.net = nn.Sequential(nn.Linear(din, 2 * dk), nn.GELU(), nn.Linear(2 * dk, 2 * dk), nn.GELU(),
                                 nn.Linear(2 * dk, (3 + ns) * dk))

    def forward(self, a, b):                                 # a, b: (l, v, slots)
        x = torch.cat([a[0], a[1], a[2][:, :self.ns].flatten(1), b[0], b[1], b[2][:, :self.ns].flatten(1)], -1)
        o = self.net(x).view(x.shape[0], 3 + self.ns, -1)
        return o[:, 0], o[:, 1], o[:, 2], o[:, 3:]           # label(key), value, relation, slots


class Loop(nn.Module):
    """Heads on Walker's pass state: query, halt, argument pointers, feedback (register -> latent token), edits."""

    def __init__(self, d, dk, vslots, max_regs):
        super().__init__()
        self.Wq = nn.Linear(d, dk)
        self.halt = nn.Parameter(torch.randn(dk) / dk ** 0.5)
        self.Wa = nn.ModuleList([nn.Linear(d, dk) for _ in range(2)])
        self.Wrk = nn.Linear(dk, dk)
        self.rpos = nn.Parameter(torch.zeros(max_regs, dk))
        self.Wfv = nn.Linear(dk, d)
        self.Wfs = nn.Linear(dk, d)
        self.spos = nn.Parameter(torch.zeros(vslots, d))
        self.ln = nn.LayerNorm(d)
        self.Wev = nn.Linear(d, dk)                          # EDIT: delta for the value just read
        self.Wge = nn.Linear(d, 1)                           # EDIT gate (starts closed)
        nn.init.zeros_(self.Wev.weight); nn.init.zeros_(self.Wev.bias)
        nn.init.zeros_(self.Wge.weight); nn.init.constant_(self.Wge.bias, -8.0)


# ====================================================================== main
def main():
    P = argparse.ArgumentParser()
    P.add_argument("--hf", default="gpt2")
    P.add_argument("--ckpt", default="")
    P.add_argument("--enc-layer", type=int, default=0)
    P.add_argument("--ctrl-layer", type=int, default=-1)
    P.add_argument("--init", default="", help="read skill (memory + readout) from a lookup checkpoint")
    P.add_argument("--steps", type=int, default=3000)
    P.add_argument("--frames-per-step", type=int, default=4)
    P.add_argument("--questions", type=int, default=24)
    P.add_argument("--warm-frames", type=int, default=60)
    P.add_argument("--cap", type=int, default=1500, help="node capacity of the world (eviction by usage / age)")
    P.add_argument("--max-ents", type=int, default=150, help="entities in the world (then only new facts / updates)")
    P.add_argument("--mix", default="lookup:0.3,chain:0.2,arith:0.5")
    P.add_argument("--train-ops", default="1,2")
    P.add_argument("--long-ops", default="3,4")
    P.add_argument("--dk", type=int, default=512)
    P.add_argument("--vslots", type=int, default=32)
    P.add_argument("--tool-slots", type=int, default=2)
    P.add_argument("--inject", type=int, default=6)
    P.add_argument("--lr", type=float, default=5e-4)
    P.add_argument("--eval-every", type=int, default=500)
    P.add_argument("--eval-q", type=int, default=96)
    P.add_argument("--max-ans", type=int, default=6)
    P.add_argument("--seed", type=int, default=0)
    P.add_argument("--save", default="runs/gnm_world.pt")
    P.add_argument("--log", default="runs/gnm_world.jsonl")
    A = P.parse_args()
    import fwm_core as C
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(A.seed)
    rng = random.Random(A.seed)
    model, raw = C.load_walker(A.hf, A.ckpt, A.enc_layer, dev)
    tok = C.Tok(raw)
    d = model.emb.weight.shape[1]
    nL = len(model.blocks)
    ctrl_layer = A.ctrl_layer % nL
    tr_ops, lo_ops = (tuple(int(x) for x in s.split(",")) for s in (A.train_ops, A.long_ops))
    max_regs = 2 * lo_ops[1] + 4
    mem = C.Memory(d, A.dk, A.vslots).to(dev)
    ro = C.Readout(d, A.dk, A.vslots).to(dev)
    if A.init:
        st = torch.load(A.init, map_location=dev)
        sd = dict(st["mem"])
        if "slot_e" in sd and sd["slot_e"].shape[0] != A.vslots:
            se = torch.zeros(A.vslots, A.dk, device=dev)
            se[:min(A.vslots, sd["slot_e"].shape[0])] = sd["slot_e"][:A.vslots]
            sd["slot_e"] = se
        C.load_memory(mem, sd)
        ro.load(st)
        print(f"INIT read skill (memory + readout) from {A.init}", flush=True)
    graph = Graph(mem, d, A.dk).to(dev)
    loop = Loop(d, A.dk, A.vslots, max_regs).to(dev)
    tools = nn.ModuleList([ToolNode(A.dk, A.tool_slots) for _ in TOOLS]).to(dev)
    params = [p for p in list(graph.parameters()) + list(ro.parameters()) + list(loop.parameters())
              + list(tools.parameters())]
    params = list({id(p): p for p in params}.values())
    opt = torch.optim.AdamW(params, lr=A.lr, weight_decay=0.01)
    inject = C.Inject(model, A.inject % nL, ro)
    ans_ids = tok.ids(" Answer:")
    a0 = len(ans_ids) - 1
    start_ids = tok.ids(" Steps:")
    PH = tok.ids(" ~")[0]
    fb = {"pos": None, "vec": None}

    def fb_hook(_m, args):
        if fb["pos"] is None:
            return None
        x = args[0].clone()
        rows, cols = fb["pos"]
        rms = x.float().pow(2).mean(-1, keepdim=True).sqrt().mean()
        x[rows, cols] = (loop.ln(fb["vec"]) * rms).to(x.dtype)
        return (x,) + tuple(args[1:])
    model.blocks[0].register_forward_pre_hook(fb_hook)

    world = World(rng, max_ents=A.max_ents)
    store = Store()

    # ------------------------------------------------------------------ frames in
    def see(objs, kind, metas):
        """Encode input frames once (smart encoder) and write their pairs as nodes of the one world."""
        out = []
        encs = [D.doc_encoding(tok.ids, tok.decode, o, "state") for o in objs]
        L = max(len(e[0]["ids"]) for e in encs)
        ids = torch.tensor([e[0]["ids"] + [0] * (L - len(e[0]["ids"])) for e in encs], device=dev)
        X = C.walker_states(model, ids).to(torch.bfloat16)
        for j, ((enc, leaves), meta) in enumerate(zip(encs, metas)):
            n = len(enc["ids"])
            out.append(store.add_frame(enc["ids"], X[j, :n].clone(), enc, leaves, kind, meta))
        return out

    cur = {}                                               # (name, key) -> current node id (truth bookkeeping)

    def add_components(n, r=None, only_update=False):
        r = r or rng
        objs = [world.component(r, only_update=only_update)[0] for _ in range(n)]
        nid_lists = see(objs, "fact", [lambda lf: None] * n)
        for o, nids in zip(objs, nid_lists):
            grp = next(iter(o))
            enc_leaves = D.serialize(o, "state")[1]          # map (name, key) -> its node (truth bookkeeping)
            for nid, lf in zip(nids, enc_leaves):
                p = lf[0]
                if len(p) == 3 and p[0] == grp and isinstance(p[1], int) and p[2] != "name":
                    nm = o[grp][p[1]]["name"]
                    store.nodes[nid]["meta"] = (nm, p[2])
                    cur[(nm, p[2])] = nid

    # ------------------------------------------------------------------ materialize all nodes (current heads)
    def materialize():
        live = [i for i in store.alive() if store.nodes[i]["kind"] != "tool"]
        fids = sorted({store.nodes[i]["frame"] for i in live})
        Xs, ids, spans, ptr, vpos, rows = [], [], [], [], [], []
        off = 0
        for f in fids:
            fr = store.frames[f]
            Xs.append(fr["X"]); ids += fr["ids"]
            e = fr["enc"]
            for nid in fr["nodes"]:
                if not store.nodes[nid]["alive"]:
                    continue
                j = store.nodes[nid]["j"]
                spans.append([off + x for x in e["spans"][j]])
                ptr.append([off + x for x in e["ptr"][j]])
                vpos.append([off + x for x in e["vpos"][j]])
                rows.append(nid)
            off += len(fr["ids"])
        X = torch.cat(Xs).float()
        with torch.no_grad():
            E = model.emb(torch.tensor(ids, device=dev)).float()
        kraw, v, logg, slots, r = graph.nodes(X, E, spans, ptr, vpos)
        tl = [i for i in store.alive() if store.nodes[i]["kind"] == "tool"]
        if tl:
            kraw = torch.cat([kraw, torch.stack([store.tools_out[i][0] for i in tl])])
            v = torch.cat([v, torch.stack([store.tools_out[i][1] for i in tl])])
            r = torch.cat([r, torch.stack([store.tools_out[i][2] for i in tl])])
            slots = torch.cat([slots, torch.stack([store.tools_out[i][3] for i in tl])])
            logg = torch.cat([logg, torch.zeros(len(tl), device=dev)])
            rows += tl
        edits = [store.nodes[i]["edit"] for i in rows]
        if any(e is not None for e in edits):
            v = v + torch.stack([e if e is not None else torch.zeros(A.dk, device=dev) for e in edits])
        row_of = {nid: k for k, nid in enumerate(rows)}
        par = torch.tensor([row_of.get(store.nodes[i]["par"], k) for k, i in enumerate(rows)], device=dev)
        tim = torch.tensor([store.nodes[i]["time"] for i in rows], device=dev)
        kind_q = torch.tensor([store.nodes[i]["kind"] == "question" for i in rows], device=dev)
        nsl = torch.tensor([min(store.nodes[i]["nslots"], A.vslots) for i in rows], device=dev)
        return {"kraw": kraw, "V": v, "S": slots, "logg": logg, "R": r, "par": par, "time": tim, "q": kind_q,
                "nsl": nsl, "rows": rows, "row_of": row_of}

    # ------------------------------------------------------------------ questions (truth from the world)
    def make_question(qrng, kind, ops=(1, 2), mode="train", pool=None):
        facts = [(nm, k) for (nm, k), nid in cur.items() if store.nodes[nid]["alive"]]
        if pool is not None:
            facts = [f for f in facts if f in pool]
        nums = [f for f in facts if f[1] in world.num_keys]
        if kind == "lookup" and facts:
            nm, k = qrng.choice(facts)
            v = world.ents[nm][k]
            return f"What is {leaf(nm, k)}?", [("read", (nm, k))], [v], v
        if kind == "chain":
            cands = [(nm, k) for (nm, k) in facts if k in world.ent_keys]
            qrng.shuffle(cands)
            for nm, k in cands:
                o = world.ents[nm][k]
                nxt = [(o, k2) for k2 in world.ents.get(o, {}) if (o, k2) in cur and store.nodes[cur[(o, k2)]]["alive"]]
                if nxt:
                    o, k2 = qrng.choice(nxt)
                    v = world.ents[o][k2]
                    return (f"What is the {D.words(k2)} of the {D.words(k)} of {nm}?",
                            [("read", (nm, k)), ("read", (o, k2))], [o, v], v)
            return None
        if len(nums) < 2:
            return None
        t = rand_tree(qrng, nums, qrng.randint(*ops), mode)
        if t is None:
            return None
        steps, vals = [], []

        def go(n):
            if n[0] == "leaf":
                steps.append(("read", (n[1], n[2]))); vals.append(world.ents[n[1]][n[2]])
                return len(steps)
            regs = [go(c) for c in n[2]]
            ti = next(i for i, x in enumerate(TOOLS) if x[0] == n[1])
            steps.append(("op", ti, regs)); vals.append(TOOLS[ti][1](*[vals[r - 1] for r in regs]))
            return len(steps)
        go(t)
        return f"What is {render(t)}?", steps, vals, vals[-1]

    def ask(qrng, n, mix, ops=(1, 2), mode="train", pool=None):
        qs = []
        kinds, wts = zip(*[(k, float(w)) for k, w in (x.split(":") for x in mix.split(","))])
        tries = 0
        while len(qs) < n and tries < 40 * n:
            tries += 1
            q = make_question(qrng, qrng.choices(kinds, wts)[0], ops, mode, pool)
            if q is not None:
                qs.append(q)
        if not qs:
            return []
        nids = see([{"question": q[0]} for q in qs], "question", [lambda lf: None] * len(qs))
        out = []
        for q, nl in zip(qs, nids):
            text, steps, vals, ans = q
            prog = [("read", nl[0], [])] + [(s[0], cur[s[1]], []) if s[0] == "read" else ("op", s[1], s[2])
                                           for s in steps]
            out.append({"text": text, "qnid": nl[0], "prog": prog, "vals": [None] + vals, "ans": ans,
                        "ops": sum(1 for s in steps if s[0] == "op")})
        return out

    # ------------------------------------------------------------------ Walker's pass loop
    def ctrl_state(regs):
        lat = [torch.cat([r[4] for r in row], 0) if row else None for row in regs]
        lens = [len(start_ids) + (x.shape[0] if x is not None else 0) for x in lat]
        L = max(lens)
        ids = torch.tensor([start_ids + [PH] * (n - len(start_ids)) + [0] * (L - n) for n in lens], device=dev)
        rws, cols, vecs = [], [], []
        for b, x in enumerate(lat):
            if x is not None:
                rws += [b] * x.shape[0]
                cols += list(range(len(start_ids), len(start_ids) + x.shape[0]))
                vecs.append(x)
        if vecs:
            fb["pos"] = (torch.tensor(rws, device=dev), torch.tensor(cols, device=dev))
            fb["vec"] = torch.cat(vecs, 0)
        got = {}

        def grab(_m, a, kw):
            got["x"] = a[0] if a else kw["x"]
            raise C._Stop
        h = model.blocks[ctrl_layer].register_forward_pre_hook(grab, with_kwargs=True)
        try:
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                model(ids, r_active=False)
        except C._Stop:
            pass
        finally:
            h.remove()
            fb["pos"] = fb["vec"] = None
        return got["x"].float()[torch.arange(len(regs), device=dev), torch.tensor(lens, device=dev) - 1]

    def run(qs, teacher):
        M = materialize()
        kraw, V, S, logg, R, par = M["kraw"], M["V"], M["S"], M["logg"], M["R"], M["par"]
        kmean = kraw.mean(0, keepdim=True)
        K = F.normalize(kraw - kmean, dim=-1)
        # supersession: a newer non-question node whose key matches cuts the older node's gain (model's gate)
        cs = K @ K.t()
        newer = (M["time"][None] > M["time"][:, None]) & ~M["q"][None] & ~M["q"][:, None]
        sup = cs.masked_fill(~newer, -1.0).max(-1).values
        logg = logg - F.softplus(graph.ow[0] * sup + graph.ow[1]) * 10.0
        adj = graph.adj(V, R, par)
        B = len(qs)
        tq = torch.tensor([store.nodes[q["qnid"]]["time"] for q in qs], device=dev)
        age = (tq[:, None] - M["time"][None]).clamp(min=0)
        visible = M["time"][None] <= tq[:, None]           # sequential frames: nothing written after the question
        age_b = graph.age[torch.log2(age.float() + 1).long().clamp(max=15)]
        regs = [[] for _ in range(B)]                      # (label, value, slots, n_slots, latent block)
        nsl_list = M["nsl"].tolist()

        def block(v, s, n):                                # a register as Walker's latent input tokens
            return torch.cat([loop.Wfv(v)[None], loop.Wfs(s[:n]) + loop.spos[:n]], 0)
        done = [False] * B
        losses, n_act, ok_act, say = [], 0, 0, []
        T = max(len(q["prog"]) for q in qs) + 1 if teacher else max_regs
        for t in range(T):
            h = ctrl_state(regs)
            qr = loop.Wq(h)
            qn = F.normalize(qr - kmean, dim=-1)
            tau = mem.tau.exp()
            cos_r = F.normalize(qr, dim=-1) @ F.normalize(R, dim=-1).t()
            e = tau * qn @ K.t() + logg[None] + graph.w_rel * cos_r + age_b
            e = e.masked_fill(~visible, -1e4)
            # relation-gated pointer traffic + 1.2.1 edges (all zero at init)
            idx = torch.arange(len(par), device=dev)
            has = par != idx
            vis_f = visible.float()
            c = F.softplus(e - (e * vis_f).sum(-1, keepdim=True) / vis_f.sum(-1, keepdim=True)) - math.log(2.0)
            c = c * vis_f
            gate = torch.sigmoid(graph.tau_r.exp() * cos_r)
            nk = torch.bincount(par[has], minlength=len(par)).clamp(min=1).float()
            up = torch.zeros_like(e).index_add(1, par[has], gate[:, has] * c[:, has]) / nk
            down = torch.zeros_like(e)
            down[:, has] = gate[:, has] * c[:, par[has]]
            s = (e + graph.g_traffic * (up + down) + c @ adj).masked_fill(~visible, -1e4)
            tk = torch.stack([tl.label for tl in tools] + [loop.halt])
            lt = tau * qn @ F.normalize(tk, dim=-1).t()
            logits = torch.cat([s, lt], -1)
            N = s.shape[1]
            arg_lg = None
            if t:
                nr = max(len(r) for r in regs)
                Rv = torch.stack([torch.stack([x[1] for x in row] + [torch.zeros(A.dk, device=dev)] * (nr - len(row)))
                                  for row in regs])
                rk = loop.Wrk(Rv) + loop.rpos[:nr]
                msk = torch.tensor([[j < len(row) for j in range(nr)] for row in regs], device=dev)
                arg_lg = [(torch.einsum("bd,btd->bt", loop.Wa[i](h), rk) / A.dk ** 0.5).masked_fill(~msk, -1e4)
                          for i in range(2)]
            if teacher:
                gold, gargs = [], []
                for b, q in enumerate(qs):
                    if t < len(q["prog"]):
                        kd, x, ra = q["prog"][t]
                        gold.append(M["row_of"][x] if kd == "read" else N + x)
                        gargs.append([r_ - 0 for r_ in ra])
                    else:
                        gold.append(N + HALT if t == len(q["prog"]) else -100)
                        gargs.append([])
                g = torch.tensor(gold, device=dev)
                losses.append(F.cross_entropy(logits, g, ignore_index=-100, reduction="sum"))
                vv = g != -100
                n_act += int(vv.sum()); ok_act += int(((logits.argmax(-1) == g) & vv).sum())
                if arg_lg is not None:
                    for i in range(2):
                        tg = torch.tensor([r[i] if len(r) > i else -100 for r in gargs], device=dev)
                        if bool((tg != -100).any()):
                            losses.append(F.cross_entropy(arg_lg[i], tg, ignore_index=-100, reduction="sum"))
                            v2 = tg != -100
                            n_act += int(v2.sum()); ok_act += int(((arg_lg[i].argmax(-1) == tg) & v2).sum())
                choice = gold
                args = [[r[i] if len(r) > i else 0 for r in gargs] for i in range(2)]
            else:
                choice = logits.argmax(-1).tolist()
                args = [arg_lg[i].argmax(-1).tolist() if arg_lg is not None else [0] * B for i in range(2)]
            apply, reads = {}, []
            for b in range(B):
                c_ = choice[b]
                if done[b] or c_ < 0:
                    continue
                if c_ < N:
                    reads.append((b, c_))
                    store.nodes[M["rows"][c_]]["usage"] += 1
                elif c_ - N == HALT or not regs[b]:
                    done[b] = True
                else:
                    apply.setdefault(c_ - N, []).append(b)
            if reads:                                      # one gather per pass (a per-row pick of the big node
                ix = torch.tensor([c_ for _, c_ in reads], device=dev)   # tensors costs a full-size grad each)
                Kc, Vc, Sc = K[ix], V[ix], S[ix]
                fv, fs = loop.Wfv(Vc), loop.Wfs(Sc) + loop.spos
                for j, (b, c_) in enumerate(reads):
                    n_ = int(nsl_list[c_])
                    regs[b].append((Kc[j], Vc[j], Sc[j], n_, torch.cat([fv[j][None], fs[j, :n_]], 0)))
            for ti, rws in apply.items():                 # tool networks on the (label, value) of their arguments
                pick = lambda i, b: regs[b][min(args[i][b], len(regs[b]) - 1)]
                a_ = tuple(torch.stack([pick(0, b)[k] for b in rws]) for k in range(3))
                b_ = tuple(torch.stack([pick(1, b)[k] for b in rws]) for k in range(3))
                ol, ov, orr, osl = tools[ti](a_, b_)
                full = torch.cat([osl, torch.zeros(len(rws), A.vslots - A.tool_slots, A.dk, device=dev)], 1)
                on = F.normalize(ol - kmean, dim=-1)
                fv, fs = loop.Wfv(ov), loop.Wfs(osl) + loop.spos[:A.tool_slots]
                for j, b in enumerate(rws):
                    regs[b].append((on[j], ov[j], full[j], A.tool_slots, torch.cat([fv[j][None], fs[j]], 0)))
                    if teacher:
                        say.append((b, len(regs[b]) - 1, qs[b]["vals"][len(regs[b]) - 1]))
                    if True:                              # written back into the one world (persistent node)
                        nid = len(store.nodes)
                        store.nodes.append({"frame": None, "j": 0, "par": nid, "time": store.time, "alive": True,
                                            "usage": 0, "kind": "tool", "meta": None, "edit": None,
                                            "nslots": A.tool_slots})
                        store.tools_out[nid] = (ol[j].detach(), ov[j].detach(), orr[j].detach(), full[j].detach())
            if all(done):
                break
        last = [len(q["prog"]) - 1 if teacher else len(regs[b]) - 1 for b, q in enumerate(qs)]
        targets = [(b, last[b], qs[b]["ans"]) for b in range(B)] + say
        zero = (torch.zeros(A.dk, device=dev), torch.zeros(A.vslots, A.dk, device=dev))
        rv = torch.stack([regs[b][r][1] if regs[b] and 0 <= r < len(regs[b]) else zero[0] for b, r, _ in targets])
        rs = torch.stack([regs[b][r][2] if regs[b] and 0 <= r < len(regs[b]) else zero[1] for b, r, _ in targets])
        ys = [tok.ids(" " + str(v))[:A.max_ans] for _, _, v in targets]
        Lm = max(map(len, ys))
        inp = torch.tensor([ans_ids + y[:-1] + [0] * (Lm - len(y)) for y in ys], device=dev)
        tgt = torch.full((len(ys), len(ans_ids) + Lm - 1), -100, dtype=torch.long, device=dev)
        for i, y in enumerate(ys):
            tgt[i, a0:a0 + len(y)] = torch.tensor(y, device=dev)
        inject.set(mem.Wo(rv), ro.Us(rs[:, :Lm]), a0)
        lo, _ = model(inp, r_active=False)
        inject.clear()
        lo = lo.float()
        say_loss = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tgt.reshape(-1), ignore_index=-100)
        em = ((lo.argmax(-1) == tgt) | (tgt == -100)).all(-1).float()
        act = sum(losses) / max(1, n_act) if losses else torch.zeros((), device=dev)
        return act + say_loss, em[:B], ok_act / max(1, n_act), (em[B:].mean().item() if len(em) > B else None)

    # ------------------------------------------------------------------ evaluation (on a snapshot of the world)
    logf = open(A.log, "a", encoding="utf-8")

    @torch.no_grad()
    def evaluate(step):
        mem.eval()
        snap = (copy.deepcopy(world.__dict__), copy.copy(store.frames), copy.deepcopy(
            [dict(n, edit=None) for n in store.nodes]), [n["edit"] for n in store.nodes], store.time,
            dict(store.tools_out), dict(cur), store.next_fid)
        er = random.Random(50_000 + step)
        out = {}

        def score(name, qs):
            em_ = []
            for j in range(0, len(qs), A.questions):
                _, em, _, _ = run(qs[j:j + A.questions], False)
                em_ += em.tolist()
            by = {}
            for q, e in zip(qs, em_):
                by.setdefault(q["ops"], []).append(e)
            out[name] = {"exact": sum(em_) / max(1, len(em_)), "n": len(em_),
                         "by_ops": {k: round(sum(v) / len(v), 3) for k, v in sorted(by.items())}}
            print(f"EVAL step {step} {name}: Walker said the answer exactly {out[name]['exact']:.3f} | by #ops "
                  f"{out[name]['by_ops']} | n {len(em_)}", flush=True)
            logf.write(json.dumps({"type": "eval", "step": step, "set": name, **out[name]}) + "\n")
            logf.flush()
        old_time = store.time
        old = {f for f, nid in cur.items() if store.nodes[nid]["time"] < old_time - 200}
        upd = {f for f in cur if world.version.get(f, 0) > 1}
        score("old_facts", ask(er, A.eval_q, "lookup:0.4,chain:0.2,arith:0.4", pool=old) if old else [])
        if upd:
            score("updated_facts", ask(er, A.eval_q, "lookup:1.0", pool=upd))
        score("longer_programs", ask(er, A.eval_q, "arith:1.0", lo_ops))
        score("held_compositions", ask(er, A.eval_q, "arith:1.0", (2, 2), "held"))
        before = set(cur)
        add_components(12, er)                             # new facts, written now, never asked before
        new = {f for f in cur if f not in before or store.nodes[cur[f]]["time"] > old_time}
        score("new_facts", ask(er, A.eval_q, "lookup:0.4,chain:0.2,arith:0.4", pool=new))
        # counterfactual: rewrite existing facts now (update frames), ask them: answers must follow the store
        before_v = dict(world.version)
        add_components(12, er, only_update=True)
        cf = {f for f, v in world.version.items() if v > before_v.get(f, 0)}
        score("counterfactual_updates", ask(er, A.eval_q, "lookup:0.5,arith:0.5", pool=cf))
        # restore the training world
        world.__dict__.update(snap[0]); store.frames = snap[1]
        store.nodes = [dict(n, edit=e) for n, e in zip(snap[2], snap[3])]
        store.time = snap[4]; store.tools_out = snap[5]; store.next_fid = snap[7]
        cur.clear(); cur.update(snap[6])
        mem.train()
        return out

    # ------------------------------------------------------------------ train on the ONE world
    add_components(A.warm_frames)
    print(f"WORLD: {len(world.ents)} entities, {len(store.alive())} nodes after {A.warm_frames} warm frames", flush=True)
    q0 = ask(random.Random(1), 3, A.mix)
    for q in q0:
        print(f"EXAMPLE: {q['text']} -> {q['ans']} ({len(q['prog'])} passes + halt)", flush=True)
    for q in q0:
        store.nodes[q["qnid"]]["alive"] = False
    evaluate(0)
    t0, best = time.time(), -1.0
    for step in range(1, A.steps + 1):
        lr = A.lr * min(1.0, step / 100) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / A.steps)))
        for gp in opt.param_groups:
            gp["lr"] = lr
        _T = __import__("time"); _dbg = __import__("os").environ.get("GW_DEBUG"); _ts = [_T.time()]

        def _lap():
            if _dbg:
                torch.cuda.synchronize(); _ts.append(_T.time())
        add_components(A.frames_per_step)                  # small pieces of detail; the world keeps growing
        _lap()
        qs = ask(rng, A.questions, A.mix, tr_ops)
        _lap()
        loss, em, act, tool_said = run(qs, True)
        _lap()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        _lap()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        _lap()
        store.evict(A.cap, protect=set(cur.values()))
        _lap()
        if _dbg:
            print("STEP", [round(b - a, 3) for a, b in zip(_ts, _ts[1:])], "(add, ask, run, backward, opt, evict)",
                  flush=True)
        for f in [f for f, nid in cur.items() if not store.nodes[nid]["alive"]]:
            del cur[f]
        if step % 25 == 0:
            n_live = len(store.alive())
            print(f"step {step} loss {float(loss):.3f} action acc {act:.2f} answer said {float(em.mean()):.2f} "
                  f"tool outputs said {tool_said if tool_said is None else round(tool_said, 2)} | world "
                  f"{len(world.ents)} entities, {n_live} live nodes [{time.time() - t0:.0f}s]", flush=True)
            logf.write(json.dumps({"type": "train", "step": step, "loss": float(loss), "action": act,
                                   "said": float(em.mean()), "tool_said": tool_said, "nodes": n_live}) + "\n")
        if step % A.eval_every == 0 or step == A.steps:
            out = evaluate(step)
            sc = sum(v["exact"] for v in out.values())
            if sc > best:
                best = sc
                torch.save({"mem": mem.state_dict(), **ro.state(), "graph": graph.state_dict(),
                            "loop": loop.state_dict(), "tools": tools.state_dict(), "step": step, "args": vars(A)},
                           A.save)


if __name__ == "__main__":
    main()
