"""GNM v5 prototype: ONE node kind. Every function node is the same network (its own weights), made by the model
when a want meets no label; a firing is routed by want x label; what a node fires next is a want it outputs.
Kept from v4: the translator (reader: a statement's fields; namer: a token group's label) and the memory (a place per
thing, its log as said). Every decision is learned. What is still code (runtime, said in every result):
  - a question arriving wants "find the next part"; an operation first compares its numbers (a fixed first call)
  - a node keeps firing on the next item of what it reads (next log entry, next digit column)
  - memory actions (the value of / before / both / the answer / take the value) are reads of the memory, with fixed
    labels (the words of what they do), not learned networks
The words a step is said with (its purpose) come from the demos -- the demos are written by me (the teacher)."""
import random, sys, time
import torch, torch.nn as nn, torch.nn.functional as F
import proto_v4c as B

dev, d = B.dev, B.d
PURPOSE = {"scan": "find the next part of the question", "read": "read the next entry of the log",
           "take": "take the value it says", "look": "the value of", "before": "the value before",
           "both": "both values of", "answer": "the answer is", "add": "the sum of", "sub": "the difference between",
           "max": "the larger of", "min": "the smaller of", "cmp": "compare the digits", "addd": "add the digits",
           "subd": "subtract the digits"}
MEMORY = ("take", "look", "before", "both", "answer")          # memory actions (reads), fixed labels
KIND2P = {"lookup": "look", "before": "before", "both": "both", "add": "add", "sub": "sub", "max": "max", "min": "min",
          "answer": "answer"}
THR = .9


@torch.no_grad()
def pvec(p):
    v = B.vecs(B.tok(PURPOSE[p])).mean(0)
    return F.normalize(v, dim=-1)


class GNode(nn.Module):
    """the one node kind: reads a token sequence (both ways), up to two token groups and a small state; says per-token
    tags, a digit, a state, a choice of order, and a want (what it fires next)"""

    def __init__(self):
        super().__init__()
        self.inp = nn.Linear(d, 96)
        self.g = nn.GRU(96, 96, batch_first=True, bidirectional=True)
        self.slot = nn.Linear(d, 96)
        self.st = nn.Embedding(8, 32)
        self.mix = nn.Sequential(nn.Linear(192 * 2 + 96 * 2 + 32, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU())
        self.tag = nn.Linear(192 + 256, len(B.QTAGS))
        self.digit, self.state, self.choice, self.want = nn.Linear(256, 10), nn.Linear(256, 8), nn.Linear(256, 4), nn.Linear(256, d)

    def forward(self, seqs, s1, s2, st):
        """a batch of firings: seqs (token lists or []), s1 / s2 (token lists or []), st (ints)"""
        n = len(st)
        Z = torch.zeros(96, device=dev)
        H, summ = [None] * n, []
        idx = [i for i in range(n) if seqs[i]]
        sm = {}
        if idx:
            x = nn.utils.rnn.pad_sequence([F.gelu(self.inp(B.vecs(seqs[i]))) for i in idx], batch_first=True)
            ln = [len(seqs[i]) for i in idx]
            pk = nn.utils.rnn.pack_padded_sequence(x, torch.tensor(ln), batch_first=True, enforce_sorted=False)
            h, _ = self.g(pk)
            h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True)
            for k, i in enumerate(idx):
                H[i] = h[k, :ln[k]]
                sm[i] = torch.cat([H[i][0], H[i][-1]])
        pool = lambda s: F.gelu(self.slot(B.vecs(s))).mean(0) if s else Z
        rows = [torch.cat([sm.get(i, torch.zeros(384, device=dev)), pool(s1[i]), pool(s2[i]),
                           self.st(torch.tensor(st[i], device=dev))]) for i in range(n)]
        m = self.mix(torch.stack(rows))
        tags = [self.tag(torch.cat([H[i], m[i].expand(len(H[i]), -1)], -1)) if H[i] is not None else None for i in range(n)]
        return dict(m=m, tags=tags, digit=self.digit(m), state=self.state(m), choice=self.choice(m),
                    want=F.normalize(self.want(m), dim=-1))


class Module(nn.Module):
    def __init__(self):
        super().__init__()
        base = B.Module()
        self.reader, self.namer = base.reader, base.namer          # the translator
        self.nodes = nn.ModuleList()
        self.labels = []                                           # each node's label (made from the want it was made for)
        self.mem = {p: pvec(p) for p in MEMORY}

    def summary(self, h):
        return torch.cat([h[0], h[-1]], -1)

    def label(self, tss):
        return F.normalize(torch.stack([self.summary(h) for _, h in self.namer.many(tss)]), dim=-1)

    def route(self, want, make=False):
        """which node a want meets (its label), or a memory action; a new node when none meets it (while learning)"""
        names = list(self.mem) + [f"n{i}" for i in range(len(self.nodes))]
        labs = torch.stack([self.mem[p] for p in self.mem] + list(self.labels)) if self.labels else torch.stack(list(self.mem.values()))
        s = labs @ want
        i = int(s.argmax())
        if float(s[i]) >= THR or not make:
            return names[i]
        self.nodes.append(GNode().to(dev))
        self.labels.append(want.detach().clone())
        return f"n{len(self.nodes) - 1}"


# ====================================================================== demos -> firings (each with the want it was for)
def firings(M, data):
    """every firing a demo world shows, with its purpose: (purpose, seq, s1, s2, state, targets)"""
    out = []
    for stm, fold, qs in data:
        last_rule, idx = {}, 0
        for ts, tags, text in stm:                    # the log reader: one entry per firing
            if "RATTR" in tags:
                last_rule[tuple(t for t, g in zip(ts, tags) if g == "RATTR")] = ts
                continue
            _, meta, act, has = fold[idx]
            idx += 1
            r = last_rule.get(tuple(t for t, g in zip(ts, tags) if g == "NATTR"), [])
            out.append(("read", ts, list(r), [], int(has), dict(want={"set": "take", "add": "add", "copy": "look"}[act])))
        for steps in qs:                               # the question: one step per firing
            for q, ph, kind, ops, _ in steps:
                y = [0] * len(q)
                if ph is not None:
                    i = B.find(q, ph)
                    for j in range(i, i + len(ph)):
                        y[j] = 1
                    if ops:
                        ia = B.find(q, ops[0], i)
                        for j in range(ia, ia + len(ops[0])):
                            y[j] = 2
                        ib = B.find(q, ops[1], ia + len(ops[0]))
                        for j in range(ib, ib + len(ops[1])):
                            y[j] = 3
                else:
                    ia = B.find(q, ops[0])
                    for j in range(ia, ia + len(ops[0])):
                        y[j] = 2
                out.append(("scan", q, [], [], 0, dict(tags=y, want=KIND2P[kind])))
                if ops and kind in ("add", "sub", "max", "min"):
                    na, nb = B.as_num(ops[0]), B.as_num(ops[1])
                    if na and nb:
                        out += op_firings(na, nb, kind)
    return out


def op_firings(a, b, op):
    """an operation's firings: the digits compared (top first), the operation decides, the columns"""
    out = []
    cols, res = B.col_lessons(None, a, b, op)
    st = 0
    for nm, x, y, s, o, ns in cols:
        if nm == "compare":
            out.append(("cmp", [], [str(x)], [str(y)], s, dict(state=ns)))
            st = ns
        elif nm == "sign":
            sign, what = ns
            w = B.SIGN[what]
            want = "addd" if w == "add" else "subd" if w.startswith("sub") else "take"
            order = {"add": 0, "sub_ab": 0, "sub_ba": 1, "take_a": 0, "take_b": 1}[w]
            out.append((op, [], ["-" if x else "+"], ["-" if y else "+"], st, dict(state=sign, want=want, choice=order)))
        else:
            out.append(("addd" if nm == "addcol" else "subd", [], [str(x)], [str(y)], s, dict(digit=o, state=ns)))
    return out


def train(M, worlds, epochs):
    data = [B.demo_world(1000 + i, 30, 24) for i in range(worlds)]
    PV = {p: pvec(p) for p in PURPOSE}
    allp = torch.stack([PV[p] for p in PURPOSE])
    fs_world = [firings(M, [w]) for w in data]
    for fs in fs_world:                                # the nodes the demos call for: one per want no label meets
        for f in fs:
            M.route(PV[f[0]], make=True)
    print(f"demos: {worlds} worlds, {sum(len(x) for x in fs_world)} firings; nodes made by the model: {len(M.nodes)} "
          f"({', '.join(sorted({M.route(PV[f[0]]) + '=' + f[0] for fs in fs_world for f in fs}))})", flush=True)
    opt = torch.optim.AdamW(M.parameters(), lr=2e-3)
    for ep in range(epochs):
        tot = 0.
        for (stm, fold, qs), fs in zip(data, fs_world):
            L = 0
            outs = M.reader.many([ts for ts, _, _ in stm])          # the translator: fields and labels
            L = L + F.cross_entropy(torch.cat([lg for lg, _ in outs]),
                                    torch.tensor([B.RTAGS.index(t) for _, tags, _ in stm for t in tags], device=dev))
            groups = []
            for ts, tags, text in stm:
                for fl in (("NAME", "NATTR"), ("NATTR",), ("SRC",), ("RATTR",)):
                    g = tuple(t for t, x in zip(ts, tags) if x in fl)
                    if g:
                        groups.append(g)
            for steps in qs:
                for q, ph, kind, ops, _ in steps:
                    if ph is not None and kind in ("lookup", "before", "both"):
                        groups.append(tuple(ph))
            rr = random.Random(len(groups))
            L = L + B.name_lesson(M, rr.sample(groups, min(400, len(groups))))
            by = {}                                                 # each node learns from the firings routed to it
            for f in fs:
                by.setdefault(M.route(PV[f[0]]), []).append(f)
            for nm, ff in by.items():
                node = M.nodes[int(nm[1:])]
                o = node([f[1] for f in ff], [f[2] for f in ff], [f[3] for f in ff], [f[4] for f in ff])
                for key in ("digit", "state", "choice"):
                    k = [i for i, f in enumerate(ff) if key in f[5]]
                    if k:
                        L = L + F.cross_entropy(o[key][k], torch.tensor([ff[i][5][key] for i in k], device=dev))
                k = [i for i, f in enumerate(ff) if "want" in f[5]]
                if k:                                               # what it fires next: its want meets that label
                    tgt = torch.tensor([list(PURPOSE).index(ff[i][5]["want"]) for i in k], device=dev)
                    L = L + F.cross_entropy(o["want"][k] @ allp.T / .05, tgt)
                k = [i for i, f in enumerate(ff) if "tags" in f[5]]
                if k:
                    L = L + F.cross_entropy(torch.cat([o["tags"][i] for i in k]),
                                            torch.tensor([t for i in k for t in ff[i][5]["tags"]], device=dev))
            opt.zero_grad()
            L.backward()
            opt.step()
            tot += float(L.detach())
        if (ep + 1) % 10 == 0:
            print(f"epoch {ep + 1}: loss {tot / len(data):.2f}", flush=True)


# ====================================================================== the run: every firing routed by a want
class Memory(B.Memory):
    def fire(self, want, seq=(), s1=(), s2=(), st=0):
        nm = self.M.route(want)
        if nm in MEMORY:
            return nm, None
        node = self.M.nodes[int(nm[1:])]
        o = node([list(seq)], [list(s1)], [list(s2)], [st])
        return nm, {k: (v[0] if v is not None and not isinstance(v, list) else (v[0] if isinstance(v, list) else v)) for k, v in o.items()}

    def op(self, want, a, b):
        """an operation: the digits compared (its fixed first call), then the operation node decides, the columns"""
        sa, da = a
        sb, db = b
        st = 0
        for x, y in zip(da, db):
            nm, o = self.fire(self.M.cmp_want, (), [str(x)], [str(y)], st)
            if o is None:
                return None
            st = int(o["state"][:3].argmax())
        nm, o = self.fire(want, (), ["-" if sa else "+"], ["-" if sb else "+"], st)
        if o is None:
            return None
        sign, order = int(o["state"][:2].argmax()), int(o["choice"][:2].argmax())
        nm2, _ = self.M.route(o["want"]), None
        if nm2 == "take":
            s_, ds = (sa, da) if order == 0 else (sb, db)
            return B.num_toks(s_, ds)
        hi, lo = (da, db) if order == 0 else (db, da)
        out, cy = [], 0
        for x, y in zip(reversed(hi), reversed(lo)):
            nm3, o3 = self.fire(o["want"], (), [str(x)], [str(y)], cy)
            if o3 is None:
                return None
            out.append(int(o3["digit"].argmax()))
            cy = int(o3["state"][:2].argmax())
        return B.num_toks(sign, list(reversed(out)))

    @torch.no_grad()
    def value(self, name, depth=0):
        p = self.place_of(name)
        if p < 0 or depth > 6:
            return None, None, []
        if p in self.memo:
            return self.memo[p]
        state, prev, seen = None, None, []
        for e in self.places[p][2]:
            rule = self.rule_text_at(e["attr"], e["t"])
            nm, o = self.fire(self.M.read_want, e["ts"], rule, (), int(state is not None))
            if o is None:
                continue
            act = self.M.route(o["want"])
            if act == "look":
                new = self.value(e["src"], depth + 1)[0]
            elif act == "take" or state is None:
                new = e["val"]
            else:
                na, nb = B.as_num(state), B.as_num(e["val"])
                new = self.op(o["want"], na, nb) if na and nb else e["val"]
            if new is None:
                continue
            if state is not None and not self.same(state, new):
                prev = state
            if not any(self.same(s, new) for s in seen):
                seen.append(new)
            state = new
        self.memo[p] = (state, prev, seen)
        return state, prev, seen

    def same(self, x, y):
        nx, ny = B.as_num(x), B.as_num(y)
        if nx and ny:
            if nx[0] != ny[0]:
                return False
            st = 0
            for a, b in zip(nx[1], ny[1]):
                _, o = self.fire(self.M.cmp_want, (), [str(a)], [str(b)], st)
                st = int(o["state"][:3].argmax()) if o is not None else 1
            return st == 0
        return x == y

    def rule_text_at(self, attr, t):
        for a, _, ents in self.rules:
            if a and self.M.label([a])[0] @ self.M.label([attr])[0] > B.JOIN if attr else False:
                old = [ts for tt, ts in ents if tt < t]
                return old[-1] if old else ()
        return ()

    @torch.no_grad()
    def hear(self, text):
        self.t += 1
        ts = B.tok(text)
        lg, h = self.M.reader(ts)
        tags = [B.RTAGS[i] for i in lg.argmax(-1).tolist()]
        field = lambda names: [t for t, g in zip(ts, tags) if g in names]
        self.memo = {}
        if "RATTR" in tags:
            attr = field({"RATTR"})
            lab = self.label(attr)
            i = self.match(lab, [r[1] for r in self.rules])
            if i >= 0:
                self.rules[i][2].append((self.t, ts))
            else:
                self.rules.append((attr, lab, [(self.t, ts)]))
            return
        name = field({"NAME", "NATTR"})
        if not name:
            return
        p = self.place_of(name, make=True)
        self.places[p][2].append(dict(t=self.t, ts=ts, val=field({"VAL"}), src=field({"SRC"}), attr=field({"NATTR"})))

    @torch.no_grad()
    def ask(self, text, trace=None):
        q = B.tok(text)
        for _ in range(40):
            nm, o = self.fire(self.M.scan_want, q)
            if o is None:
                return None
            tags = [B.QTAGS[i] for i in o["tags"].argmax(-1).tolist()]
            ph = [i for i, g in enumerate(tags) if g in ("PH", "A", "B")]
            act = self.M.route(o["want"])
            if trace is not None:
                trace.append((" ".join(q), act, " ".join(q[i] for i in ph)))
            A_ = [q[i] for i, g in enumerate(tags) if g == "A"]
            B_ = [q[i] for i, g in enumerate(tags) if g == "B"]
            if act == "answer":
                return "".join(A_)
            if not ph:
                return None
            a0, a1 = ph[0], ph[-1]
            phrase = q[a0:a1 + 1]
            if act in ("look", "before", "both", "take"):
                cur, prev, seen = self.value(phrase)
                new = prev if act == "before" else (seen[0] + ["and"] + seen[1] if len(seen) >= 2 else None) if act == "both" else cur
            else:
                na, nb = B.as_num(A_), B.as_num(B_)
                new = self.op(o["want"], na, nb) if na and nb else None
            if not new:
                return None
            q = q[:a0] + new + q[a1 + 1:]
        return None


@torch.no_grad()
def evaluate(M, seed, ticks=30, nq=48, show_fail=0):
    rng = random.Random(seed)
    w = B.new_world(rng)
    mem = Memory(M)
    for _ in range(ticks):
        for dd in B.world_step(w, rng):
            o = dd()
            for text, meta in B.said(o, w) if o else []:
                mem.hear(text)
    facts = [(nm, k, v) for nm, e in w.ents.items() for k, v in e.items()]
    held = sum((mem.value(B.tok(B.leaf(nm, k)))[0] or []) == B.tok(str(v)) for nm, k, v in facts)
    res, shown = {}, 0
    for kind, ops in (("lookup", 0), ("before", 0), ("coexist", 0), ("chain", 0), ("ops", (1, 3)), ("long", (4, 6)), ("long12", (8, 12))):
        hit = n = 0
        for _ in range(nq):
            E = B.q_expr("ops" if kind.startswith("long") else kind, w, rng, ops or (1, 3))
            if E is None:
                continue
            n += 1
            tr = []
            got = mem.ask(B.ask_text(E), tr)
            ok = got == str(B.truth_of(E, w))
            hit += ok
            if not ok and shown < show_fail:
                shown += 1
                print(f"  [{kind}] {B.ask_text(E)} truth {B.truth_of(E, w)} got {got}" + "".join(f"\n      {a} | {b} | {c_}" for a, b, c_ in tr[:8]))
        res[kind] = (hit, n)
    return held, len(facts), res


if __name__ == "__main__":
    worlds = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    torch.manual_seed(0)
    M = Module().to(dev)
    M.scan_want, M.read_want, M.cmp_want = pvec("scan"), pvec("read"), pvec("cmp")   # the runtime's fixed calls
    t0 = time.time()
    train(M, worlds, epochs)
    print(f"trained {time.time() - t0:.0f}s", flush=True)
    for s in (7, 8):
        held, nf, res = evaluate(M, 50000 + s, show_fail=3 if s == 7 else 0)
        print(f"fresh world {s}: facts held {held}/{nf} | " + ", ".join(f"{k} {a}/{b}" for k, (a, b) in res.items()), flush=True)
