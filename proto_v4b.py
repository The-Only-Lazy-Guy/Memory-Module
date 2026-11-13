"""GNM v4 prototype (local; to move into gnm.py once it holds). User design 11-14: nodes are objects (fields as token
groups; label; relation; their own action network); the network is recurrent -- every firing solves ONE small
subproblem, its state kept in nodes. Every decision below is a learned network; what is code is only the runtime
(append to a log, fire again on the next entry / column / question step, replace a phrase by what a node gave back).

Nodes and their actions (all learned, own networks):
  reader  (translator): a statement's tokens -> its fields (name, attribute, value, source, rule attribute) by tags
  fold    : one log entry per firing: (entry, the rule in force then, is there a value yet) -> set / add / copy
  compare : one digit column per firing, top first: (a digit, b digit, state) -> state (equal / a larger / b larger)
  addcol  : one column per firing, ones first: (a digit, b digit, carry) -> (digit, carry)
  subcol  : one column per firing: (larger's digit, smaller's digit, borrow) -> (digit, borrow)
  sign    : (sign a, sign b, operation, compare state) -> (sign of the result, what to do with the sizes)
  scanner : one token per firing (both ways): the question's tokens -> which tokens are the next thing to resolve and
            its two operands; the kind of that thing (lookup / before / both / add / sub / max / min / answer)
Text never stores anything worked out; nothing below computes an answer in code.

Tokens come in as fixed vectors (stand-in for the LM's token embeddings: new names get new vectors) and a learned
projection; fields are token groups (user 11-12: grouped tokens are fine)."""
import random, re, sys, time, hashlib
import torch, torch.nn as nn, torch.nn.functional as F

SRC = open(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "gnm.py"), encoding="utf-8").read()
head = SRC[:SRC.index("# ====================================================================== the graph")]
head = "\n".join(l for l in head.split("\n") if not re.match(r"\s*(import torch|from torch|from transformers)", l))


def block(name):
    m = re.search(rf"^(?:def|class) {name}\b.*?(?=^(?:def |class |@|if __name__|[A-Za-z_]+ = ))", SRC, re.M | re.S)
    return m.group(0)


code = head + "\n" + "\n".join(block(n) for n in ("said", "unresolved", "show", "swap", "ask_text", "q_expr", "truth_of",
                                                  "new_world", "world_step"))
G = {}
exec(code, G)
World, said, unresolved, show, swap, ask_text, q_expr, truth_of, new_world, world_step, leaf, words, TOOLS = (
    G[k] for k in ("World", "said", "unresolved", "show", "swap", "ask_text", "q_expr", "truth_of", "new_world", "world_step",
                   "leaf", "words", "TOOLS"))
value_in = G["value_in"]
OPF = {n: f for n, f, _ in TOOLS}

dev = "cuda" if torch.cuda.is_available() else "cpu"
d, c = 64, 32
tok = lambda s: re.findall(r"[A-Za-z]+|\d|[^\sA-Za-z\d]", s)


def tvec(t):                                     # a token's fixed vector (the LM's embedding stands here)
    h = int(hashlib.md5(t.encode()).hexdigest()[:8], 16)
    g = torch.Generator().manual_seed(h)
    return torch.randn(d, generator=g)


VEC = {}


def vecs(ts):
    for t in ts:
        if t not in VEC:
            VEC[t] = tvec(t)
    return torch.stack([VEC[t] for t in ts]).to(dev)


def find(seq, sub, start=0):
    for i in range(start, len(seq) - len(sub) + 1):
        if seq[i:i + len(sub)] == sub:
            return i
    return -1


# ====================================================================== the nodes' networks
class Seq(nn.Module):                             # a node that reads a token sequence, one token per firing, both ways
    def __init__(self, n_tags, n_kinds=0):
        super().__init__()
        self.inp = nn.Linear(d, 96)
        self.g = nn.GRU(96, 96, batch_first=True, bidirectional=True, num_layers=2)
        self.tag = nn.Linear(192, n_tags)
        self.kind = nn.Linear(192, n_kinds) if n_kinds else None

    def forward(self, ts):
        h, _ = self.g(F.gelu(self.inp(vecs(ts)))[None])
        h = h[0]
        return self.tag(h), h

    def many(self, tss):
        """many sequences at once (the same firings, batched): -> per sequence (tag logits, states)"""
        n = [len(t) for t in tss]
        x = nn.utils.rnn.pad_sequence([F.gelu(self.inp(vecs(t))) for t in tss], batch_first=True)
        pk = nn.utils.rnn.pack_padded_sequence(x, torch.tensor(n), batch_first=True, enforce_sorted=False)
        h, _ = self.g(pk)
        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True)
        lg = self.tag(h)
        return [(lg[i, :n[i]], h[i, :n[i]]) for i in range(len(tss))]


class Step(nn.Module):                            # a column node: two digit rows + a state row -> a digit + a state
    def __init__(self, n_state, n_out=10):
        super().__init__()
        self.st = nn.Embedding(n_state, c)
        self.f = nn.Sequential(nn.Linear(3 * c, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU())
        self.o, self.s = nn.Linear(128, n_out), nn.Linear(128, n_state)

    def forward(self, dig, a, b, s):
        h = self.f(torch.cat([dig(a), dig(b), self.st(s)], -1))
        return self.o(h), self.s(h)


RTAGS = ["O", "NAME", "NATTR", "VAL", "SRC", "RATTR"]   # a statement's fields
QTAGS = ["O", "PH", "A", "B"]                           # the next thing to resolve, and its two operands
KINDS = ["lookup", "before", "both", "add", "sub", "max", "min", "answer"]
FOLD = ["set", "add", "copy"]
SIGN = ["add", "sub_ab", "sub_ba", "take_a", "take_b"]   # what the sign node does with the sizes


class Module(nn.Module):
    def __init__(self):
        super().__init__()
        self.reader = Seq(len(RTAGS))
        self.scanner = Seq(len(QTAGS))
        self.kind = nn.Sequential(nn.Linear(192 * 2, 128), nn.GELU(), nn.Linear(128, len(KINDS)))
        self.fold = nn.Sequential(nn.Linear(384 * 2 + 1, 128), nn.GELU(), nn.Linear(128, len(FOLD)))
        self.dig = nn.Embedding(10, c)
        self.compare, self.addcol, self.subcol = Step(3, 1), Step(2), Step(2)
        self.sign = nn.Sequential(nn.Embedding(2 * 2 * 4 * 3, 64), nn.GELU(), nn.Linear(64, 2 + len(SIGN)))

    def summary(self, h):
        return torch.cat([h[0], h[-1]], -1)       # the node's own reading of its sequence (both ends)


# ====================================================================== numbers as fields (digit rows)
P = 5


def as_num(ts):
    if not ts or any(t != "-" and not t.isdigit() for t in ts):
        return None
    neg = ts[0] == "-"
    ds = [int(t) for t in ts[1:] if t.isdigit()] if neg else [int(t) for t in ts]
    if not ds or len(ds) > P:
        return None
    return int(neg), [0] * (P - len(ds)) + ds          # (sign, digits top first)


def num_toks(sign, ds):
    while len(ds) > 1 and ds[0] == 0:
        ds = ds[1:]
    return (["-"] if sign and any(ds) else []) + [str(x) for x in ds]


def col_lessons(M, a, b, op):
    """the column firings an operation makes, and what each should give (a demo's written-out working)"""
    sa, da = a
    sb, db = b
    va, vb = int("".join(map(str, da))), int("".join(map(str, db)))
    st, steps = 0, []
    for x, y in zip(da, db):                        # compare, top first
        nst = st if st else (1 if x > y else 2 if y > x else 0)
        steps.append(("compare", x, y, st, 0, nst))
        st = nst
    A_, B_ = (-va if sa else va), (-vb if sb else vb)
    res = OPF[op](A_, B_)
    if op in ("max", "min"):
        what = "take_a" if res == A_ and (res != B_ or True) else "take_b"
    else:
        ea, eb = A_, (B_ if op == "add" else -B_)
        if (ea >= 0) == (eb >= 0):
            what = "add"
        else:
            what = "sub_ab" if abs(ea) >= abs(eb) else "sub_ba"
    sign = int(res < 0)
    steps.append(("sign", sa, sb, ["add", "sub", "max", "min"].index(op), st, (sign, SIGN.index(what))))
    if what == "add":
        cy = 0
        for x, y in zip(reversed(da), reversed(db)):
            steps.append(("addcol", x, y, cy, (x + y + cy) % 10, (x + y + cy) // 10))
            cy = (x + y + cy) // 10
    elif what.startswith("sub"):
        hi, lo = (da, db) if what == "sub_ab" else (db, da)
        bw = 0
        for x, y in zip(reversed(hi), reversed(lo)):
            v = x - y - bw
            steps.append(("subcol", x, y, bw, v % 10, int(v < 0)))
            bw = int(v < 0)
    return steps, res


def run_op(M, a, b, op):
    """the operation done by firing its nodes column by column (nothing computed here)"""
    sa, da = a
    sb, db = b
    T = lambda v: torch.tensor([v], device=dev)
    st = 0
    for x, y in zip(da, db):
        _, s = M.compare(M.dig, T(x), T(y), T(st))
        st = int(s.argmax())
    k = ((sa * 2 + sb) * 4 + ["add", "sub", "max", "min"].index(op)) * 3 + st
    o = M.sign(T(k))[0]
    sign, what = int(o[:2].argmax()), SIGN[int(o[2:].argmax())]
    if what in ("take_a", "take_b"):
        s_, ds = (sa, da) if what == "take_a" else (sb, db)
        return num_toks(s_, ds)
    out, cy = [], 0
    node = M.addcol if what == "add" else M.subcol
    hi, lo = (da, db) if what != "sub_ba" else (db, da)
    for x, y in zip(reversed(hi), reversed(lo)):
        dd, s = node(M.dig, T(x), T(y), T(cy))
        out.append(int(dd.argmax()))
        cy = int(s.argmax())
    return num_toks(sign, list(reversed(out)))


# ====================================================================== the teacher (demos and grading only)
def stmt_tags(text, meta, w):
    ts = tok(text)
    tags = ["O"] * len(ts)
    if meta[0] == "rule":
        k = tok(words(meta[1]))
        i = find(ts, k)
        for j in range(i, i + len(k)):
            tags[j] = "RATTR"
        return ts, tags
    nm, k = meta[1], meta[2]
    name = tok(leaf(nm, k))
    i = find(ts, name)
    kw = tok(words(k))
    ka = find(ts, kw, i)
    for j in range(i, i + len(name)):
        tags[j] = "NATTR" if ka <= j < ka + len(kw) else "NAME"
    rest = ts[i + len(name):]
    j0 = i + len(name) + (2 if rest[:2] == ["is", "now"] else 1)
    j1 = ts.index("(", j0) if "(" in ts[j0:] else len(ts) - 1
    for j in range(j0, j1):
        tags[j] = "VAL"
    if "(" in ts[j0:]:
        for j in range(j1 + 3, len(ts) - 2):     # '( same as' ... ')' '.'
            tags[j] = "SRC"
    return ts, tags


class Teacher:
    def __init__(self, w):
        self.w, self.log, self.policy = w, {}, {}

    def heard(self, text, meta, side):
        if meta[0] == "rule":
            self.policy[meta[1]] = self.w.policy[meta[1]]
            return
        nm, k = meta[1], meta[2]
        lg = self.log.setdefault((nm, k), [])
        if "(same as" in text:
            act = "copy"
        elif text.startswith("Correction") or not lg or self.policy.get(k) != "add":
            act = "set"
        else:
            act = "add"
        lg.append(act)


def question_steps(E, w):
    """the demo steps of a question: (question tokens, phrase tokens or None, kind, operands, next question)"""
    was, out = E[0] == "before", []
    tell = lambda E2: f"What was {show(E2)} before?" if was else ask_text(E2)
    q = ask_text(E)
    for _ in range(60):
        x = unresolved(E)
        if x is None:
            out.append((tok(q), None, "answer", (tok(str(E[1])), None), None))
            return out
        if x[0] == "of" and x[2][0] == "val":
            E = swap(E, x, ("leaf", x[2][1], x[1]))
            continue
        if x[0] in ("leaf", "before", "both"):
            ph = tok(leaf(x[1], x[2]))
            if x[0] == "both":
                vs = w.multi.get((x[1], x[2]), [])
                if len(vs) != 2:
                    return None
                y = ("pair", vs[0], vs[1])
            else:
                y = ("val", truth_of(("leaf", x[1], x[2]) if x[0] == "leaf" else x, w))
            kind = {"leaf": "lookup"}.get(x[0], x[0])
            ops = None
        else:
            if x[0] == "sumboth":
                a, b, opn = x[1][1], x[1][2], "add"
            else:
                a, b, opn = x[2][0][1], x[2][1][1], x[1]
            y = ("val", OPF[opn](a, b))
            ph = tok(show(x) if x is E else f"({show(x)})")
            kind, ops = opn, (tok(str(a)), tok(str(b)))
        E = y if x is E else swap(E, x, y)
        q2 = tell(E) if E[0] != "pair" else None
        out.append((tok(q), ph, kind, ops, q2))
        q = q2
    return None


# ====================================================================== the module's memory and its run
class Memory:
    def __init__(self, M):
        self.M, self.places, self.rules = M, [], []     # places: (name tokens, label, [entries]); rules: (attr, [(t, h)])
        self.t, self.labs, self.memo = 0, {}, {}

    def label(self, ts):                         # a name's label: its own reading (same tokens -> the same label)
        k = tuple(ts)
        if k not in self.labs:
            with torch.no_grad():
                _, h = self.M.reader(ts)
            self.labs[k] = F.normalize(self.M.summary(h), dim=-1)
        return self.labs[k]

    def match(self, lab, labs):                  # the best label among many, in one go
        if not labs:
            return -1
        s = torch.stack(labs) @ lab
        i = int(s.argmax())
        return i if float(s[i]) > .97 else -1

    def place_of(self, name, make=False):
        lab = self.label(name)
        i = self.match(lab, [p[1] for p in self.places])
        if i >= 0 or not make:
            return i
        self.places.append((name, lab, []))
        return len(self.places) - 1

    def rule_at(self, attr, t):
        if not attr:
            return None
        i = self.match(self.label(attr), [r[1] for r in self.rules])
        if i < 0:
            return None
        old = [h for tt, h in self.rules[i][2] if tt < t]
        return old[-1] if old else None

    @torch.no_grad()
    def hear(self, text):
        """the reader node reads a statement into its fields; the memory keeps it as said, in its thing's log"""
        self.t += 1
        ts = tok(text)
        lg, h = self.M.reader(ts)
        tags = [RTAGS[i] for i in lg.argmax(-1).tolist()]
        sm = self.M.summary(h)
        field = lambda names: [t for t, g in zip(ts, tags) if g in names]
        if "RATTR" in tags:
            attr = field({"RATTR"})
            lab = self.label(attr)
            i = self.match(lab, [r[1] for r in self.rules])
            if i >= 0:
                self.rules[i][2].append((self.t, sm))
            else:
                self.rules.append((attr, lab, [(self.t, sm)]))
            self.memo = {}
            return
        name = field({"NAME", "NATTR"})
        if not name:
            return
        p = self.place_of(name, make=True)
        self.memo = {}
        self.places[p][2].append(dict(t=self.t, val=field({"VAL"}), src=field({"SRC"}), h=sm, attr=field({"NATTR"})))

    @torch.no_grad()
    def value(self, name, depth=0):
        """fold the thing's log, one entry per firing of the fold node -> (current, before, the distinct values)"""
        p = self.place_of(name)
        if p < 0 or depth > 6:
            return None, None, []
        if p in self.memo:                       # (read once until something new is heard)
            return self.memo[p]
        state, prev, seen = None, None, []
        for e in self.places[p][2]:
            r = self.rule_at(e["attr"], e["t"])
            x = torch.cat([e["h"], r if r is not None else torch.zeros_like(e["h"]), torch.tensor([float(state is not None)], device=dev)])
            act = FOLD[int(self.M.fold(x).argmax())]
            if act == "copy":
                new = self.value(e["src"], depth + 1)[0]
            elif act == "add" and state is not None and as_num(state) and as_num(e["val"]):
                new = run_op(self.M, as_num(state), as_num(e["val"]), "add")
            else:
                new = e["val"]
            if new is None:
                continue
            if state is not None and not same(self.M, state, new):
                prev = state
            if not any(same(self.M, s, new) for s in seen):
                seen.append(new)
            state = new
        self.memo[p] = (state, prev, seen)
        return state, prev, seen

    @torch.no_grad()
    def ask(self, text, trace=None):
        q = tok(text)
        for _ in range(40):
            lg, h = self.M.scanner(q)
            tags = [QTAGS[i] for i in lg.argmax(-1).tolist()]
            ph = [i for i, g in enumerate(tags) if g in ("PH", "A", "B")]
            hs = h[ph].mean(0) if ph else h.mean(0)
            kind = KINDS[int(self.M.kind(torch.cat([hs, h.mean(0)])).argmax())]
            if trace is not None:
                trace.append((" ".join(q), kind, " ".join(q[i] for i in ph)))
            A_ = [q[i] for i, g in enumerate(tags) if g == "A"]
            B_ = [q[i] for i, g in enumerate(tags) if g == "B"]
            if kind == "answer":
                return "".join(A_)
            if not ph:
                return None
            a0, a1 = ph[0], ph[-1]
            phrase = q[a0:a1 + 1]
            if kind in ("lookup", "before", "both"):
                cur, prev, seen = self.value(phrase)
                if kind == "lookup":
                    new = cur
                elif kind == "before":
                    new = prev
                else:
                    new = seen[0] + ["and"] + seen[1] if len(seen) >= 2 else None
            else:
                na, nb = as_num(A_), as_num(B_)
                new = run_op(self.M, na, nb, kind) if na and nb else None
            if not new:
                return None
            q = q[:a0] + new + q[a1 + 1:]
        return None


def same(M, x, y):
    """two values the same? numbers: the compare node, column by column; names: their tokens (a name is its group)"""
    nx, ny = as_num(x), as_num(y)
    if nx and ny:
        if nx[0] != ny[0]:
            return False
        T = lambda v: torch.tensor([v], device=dev)
        st = 0
        for a, b in zip(nx[1], ny[1]):
            _, s = M.compare(M.dig, T(a), T(b), T(st))
            st = int(s.argmax())
        return st == 0
    return x == y


# ====================================================================== demos -> lessons; then a fresh world
def demo_world(seed, ticks, nq):
    rng = random.Random(seed)
    w, T = new_world(rng), None
    T = Teacher(w)
    stm, fold, qs = [], [], []
    for _ in range(ticks):
        for dd in world_step(w, rng):
            o = dd()
            side = w.side.get(id(o), {}) if o else {}
            for text, meta in said(o, w) if o else []:
                stm.append(stmt_tags(text, meta, w) + (text,))
                T.heard(text, meta, side)
                if meta[0] == "fact":
                    fold.append((text, meta, T.log[(meta[1], meta[2])][-1], len(T.log[(meta[1], meta[2])]) > 1))
    for kind in ["lookup", "before", "coexist", "chain", "ops", "ops"] * (nq // 6):
        E = q_expr(kind, w, rng)
        if E is not None:
            st = question_steps(E, w)
            if st:
                qs.append(st)
    return stm, fold, qs


def train(M, worlds, epochs):
    opt = torch.optim.AdamW(M.parameters(), lr=2e-3)
    data = [demo_world(1000 + i, 30, 24) for i in range(worlds)]
    n_st = sum(len(s) for s, _, _ in data)
    n_q = sum(len(q) for _, _, q in data)
    print(f"demos: {worlds} worlds, {n_st} statements, {n_q} questions "
          f"({sum(len(st) for _, _, q in data for st in q)} question steps)", flush=True)
    rules = {}
    for ep in range(epochs):
        tot = 0.
        for stm, fold, qs in data:
            L = 0
            outs = M.reader.many([ts for ts, _, _ in stm])          # the reader: fields
            L = L + F.cross_entropy(torch.cat([lg for lg, _ in outs]),
                                    torch.tensor([RTAGS.index(t) for _, tags, _ in stm for t in tags], device=dev))
            last_rule, xs, ys, idx = {}, [], [], 0                # the fold node: one entry per firing
            for (ts, tags, text), (_, h) in zip(stm, outs):
                sm = M.summary(h)
                if "RATTR" in tags:
                    last_rule[tuple(t for t, g in zip(ts, tags) if g == "RATTR")] = sm
                    continue
                txt, meta, act, has = fold[idx]
                idx += 1
                r = last_rule.get(tuple(t for t, g in zip(ts, tags) if g == "NATTR"))
                xs.append(torch.cat([sm, r if r is not None else torch.zeros_like(sm), torch.tensor([float(has)], device=dev)]))
                ys.append(FOLD.index(act))
            L = L + F.cross_entropy(M.fold(torch.stack(xs)), torch.tensor(ys, device=dev))
            steps = [x for st in qs for x in st]                  # the scanner: one question step per firing
            souts = M.scanner.many([q for q, _, _, _, _ in steps])
            ys_t, ks, kys, cols = [], [], [], []
            for (q, ph, kind, ops, _), (lg, h) in zip(steps, souts):
                y = [0] * len(q)
                if ph is not None:
                    i = find(q, ph)
                    for j in range(i, i + len(ph)):
                        y[j] = 1
                    if ops:
                        ia = find(q, ops[0], i)
                        for j in range(ia, ia + len(ops[0])):
                            y[j] = 2
                        ib = find(q, ops[1], ia + len(ops[0]))
                        for j in range(ib, ib + len(ops[1])):
                            y[j] = 3
                    idx_ = list(range(i, i + len(ph)))
                else:
                    ia = find(q, ops[0])
                    for j in range(ia, ia + len(ops[0])):
                        y[j] = 2
                    idx_ = list(range(ia, ia + len(ops[0])))
                ys_t += y
                ks.append(torch.cat([h[idx_].mean(0), h.mean(0)]))
                kys.append(KINDS.index(kind))
                if ops and kind in ("add", "sub", "max", "min"):
                    na, nb = as_num(ops[0]), as_num(ops[1])
                    if na and nb:
                        cols += col_lessons(M, na, nb, kind)[0]
            L = L + F.cross_entropy(torch.cat([lg for lg, _ in souts]), torch.tensor(ys_t, device=dev))
            L = L + F.cross_entropy(M.kind(torch.stack(ks)), torch.tensor(kys, device=dev))
            for nm in ("compare", "addcol", "subcol"):             # the column firings, batched by node
                cs = [x for x in cols if x[0] == nm]
                if cs:
                    T_ = lambda i: torch.tensor([x[i] for x in cs], device=dev)
                    dd, st = getattr(M, nm)(M.dig, T_(1), T_(2), T_(3))
                    L = L + F.cross_entropy(st, T_(5)) + (F.cross_entropy(dd, T_(4)) if nm != "compare" else 0)
            sg = [x for x in cols if x[0] == "sign"]
            if sg:
                k = torch.tensor([((x[1] * 2 + x[2]) * 4 + x[3]) * 3 + x[4] for x in sg], device=dev)
                o = M.sign(k)
                L = L + F.cross_entropy(o[:, :2], torch.tensor([x[5][0] for x in sg], device=dev)) +                     F.cross_entropy(o[:, 2:], torch.tensor([x[5][1] for x in sg], device=dev))
            opt.zero_grad()
            L.backward()
            opt.step()
            tot += float(L)
        print(f"epoch {ep + 1}: loss {tot / len(data):.2f}", flush=True)


@torch.no_grad()
def evaluate(M, seed, ticks=30, nq=48, show_fail=0):
    rng = random.Random(seed)
    w = new_world(rng)
    mem = Memory(M)
    for _ in range(ticks):
        for dd in world_step(w, rng):
            o = dd()
            for text, meta in said(o, w) if o else []:
                mem.hear(text)
    facts = [(nm, k, v) for nm, e in w.ents.items() for k, v in e.items()]
    held = sum((mem.value(tok(leaf(nm, k)))[0] or []) == tok(str(v)) for nm, k, v in facts)
    res, shown = {}, 0
    for kind, ops in (("lookup", 0), ("before", 0), ("coexist", 0), ("chain", 0), ("ops", (1, 3)), ("long", (4, 6)), ("long12", (8, 12))):
        hit = n = 0
        for _ in range(nq):
            E = q_expr("ops" if kind.startswith("long") else kind, w, rng, ops or (1, 3))
            if E is None:
                continue
            n += 1
            tr = []
            got = mem.ask(ask_text(E), tr)
            ok = got == str(truth_of(E, w))
            hit += ok
            if not ok and shown < show_fail:
                shown += 1
                print(f"  [{kind}] {ask_text(E)} truth {truth_of(E, w)} got {got}" + "".join(f"\n      {a} | {b} | {c_}" for a, b, c_ in tr[:8]))
        res[kind] = (hit, n)
    return held, len(facts), res


if __name__ == "__main__":
    worlds = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    torch.manual_seed(0)
    M = Module().to(dev)
    t0 = time.time()
    train(M, worlds, epochs)
    print(f"trained {time.time() - t0:.0f}s", flush=True)
    torch.save(M.state_dict(), "v4b_M.pt")
    for s in (7, 8):
        held, nf, res = evaluate(M, 50000 + s, show_fail=4 if s == 7 else 0)
        print(f"fresh world {s}: facts held {held}/{nf} | " + ", ".join(f"{k} {a}/{b}" for k, (a, b) in res.items()), flush=True)
