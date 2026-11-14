"""GNM on real text (user 11-14: "use real information system, biology first; see if the model picks up the input and
can create / edit a reasonable node"). Real Wikipedia species articles (WikiText-103, local); ~50 demo sentences
annotated by the teacher (bio_data.py); held-out articles graded.

PROPOSAL (Claude), on the user's design (nodes are objects; one firing = one subproblem; model-made nodes; demos then
outcome): a node per thing; its fields = facts (attribute words as said, values as said; a field's later value is an
edit, its log kept). The read node fires on a sentence once per fact: it marks the thing (THING), the attribute words
(ATTR) and the value (VAL), chooses whether the thing is the article's subject, and wants 'keep the fact' or 'nothing
more'; the kept fact's words leave the sentence and it fires again. The namer learns which attribute words are the
same field (the teacher names each attribute in the demos: 'is found in' ~ 'occurs in' = range)."""
import json, os, random, re, sys, time
import torch, torch.nn as nn, torch.nn.functional as F

sys.path.insert(0, r"C:/Users/Ace/Downloads/CPT")
import gnm as G
from bio_data import DEMO, GOLD

dev, tok = G.dev, G.tok
TAGS = ["O", "THING", "ATTR", "VAL"]          # (the node kind's 4 tag rows)
BP = {"fact": "find the next fact in the sentence", "keep": "keep the fact", "done": "nothing more to keep"}
BKEYS = list(BP)
JOIN = .8


@torch.no_grad()
def bvec(p):
    """a purpose's label: its words' token embeddings (11-14: with contextual states every purpose came out the same
    vector -- 'find the next fact' met 'keep the fact')"""
    if G.VECS == "stand":
        return F.normalize(G.vecs(tok(BP[p])).mean(0), dim=-1)
    tk, m = G.lm()
    ids = torch.tensor([tk(" " + BP[p])["input_ids"]], device=dev)
    return F.normalize(m.wte(ids)[0].mean(0), dim=-1)


def sents(p):
    return [s.strip() + " ." for s in re.split(r" \. ", p) if s.strip()]


ARTS = {b["title"]: sents(b["text"][0]) for b in json.load(open(next(p for p in (os.path.join(os.path.dirname(os.path.abspath(__file__)), f) for f in ("species.json", "species_used.json")) if os.path.exists(p)), encoding="utf-8"))}


def find_all(ts, sub):
    return [i for i in range(len(ts) - len(sub) + 1) if ts[i:i + len(sub)] == sub]


def locate(ts, fact):
    """a fact's token positions in a sentence: mention, attribute words, value (as the teacher wrote them)"""
    mention, topic, frags, key, value = fact
    m = []
    if mention:
        hits = find_all(ts, tok(mention))
        assert hits, ("mention", mention, " ".join(ts))
        m = list(range(hits[0], hits[0] + len(tok(mention))))
    nth = 1
    if "#" in value:
        value, nth = value.split("#")[0], int(value.split("#")[1])
    vt = tok(value)
    hits = [h for h in find_all(ts, vt) if not set(range(h, h + len(vt))) & set(m)]
    assert len(hits) >= nth, ("value", value, " ".join(ts))
    if nth > 1:
        v0 = hits[nth - 1]
    else:                                        # the occurrence that stands nearest its first attribute words
        ft = tok(frags[0])
        fh = find_all(ts, ft)

        def gap(h):
            g = [h - (x + len(ft)) for x in fh if x + len(ft) <= h] + [x - (h + len(vt)) for x in fh if x >= h + len(vt)]
            return min(g) if g else 10 ** 6
        v0 = min(hits, key=lambda h: (gap(h), h))
    v = list(range(v0, v0 + len(vt)))
    a = []
    for fr in frags:
        ft = tok(fr)
        hs = find_all(ts, ft)
        before = [h for h in hs if h + len(ft) <= v0 and not set(range(h, h + len(ft))) & set(m)]
        aft = [h for h in hs if h >= v[-1] + 1]
        assert before or aft, ("attr", fr, " ".join(ts))
        h = before[-1] if before else aft[0]
        a += list(range(h, h + len(ft)))
    return m, a, v


def sentence_firings(ts, facts, title):
    """the demo of one sentence: one firing per fact, then 'nothing more'"""
    out, cur = [], list(ts)
    facts = sorted(facts, key=lambda f: locate(ts, f)[2][0])   # read left to right (the teacher's order)
    for f in facts:
        m, a, v = locate(cur, f)
        y = [0] * len(cur)
        for i in m:
            y[i] = 1
        for i in a:
            y[i] = 2
        for i in v:
            y[i] = 3
        out.append(("fact", list(cur), tok(title), [], 0, dict(tags=y, want="keep", choice=f[1])))
        gone = set(a) | set(v)
        cur = [t for i, t in enumerate(cur) if i not in gone]
    out.append(("fact", list(cur), tok(title), [], 0, dict(tags=[0] * len(cur), want="done")))
    return out


def article_firings(data):
    fs, attrs = [], []
    for title, ann in data.items():
        ss = ARTS[title]
        for si, facts in ann.items():
            fs += sentence_firings(tok(ss[si]), facts, title)
            for f in facts:
                _, a, _ = locate(tok(ss[si]), f)
                attrs.append((tuple(tok(ss[si])[i] for i in a), f[3]))
    return fs, attrs


# ====================================================================== the module
class Bio(nn.Module):
    def __init__(self):
        super().__init__()
        self.namer = G.Seq(1)
        self.nodes = nn.ModuleList()
        self.labels = []
        self.mem = {p: bvec(p) for p in ("keep", "done")}
        self.fact_want = bvec("fact")

    def label(self, tss):
        return F.normalize(torch.stack([torch.cat([h[0], h[-1]], -1) for _, h in self.namer.many(tss)]), dim=-1)

    def route(self, want, make=False):
        names = list(self.mem) + [f"n{i}" for i in range(len(self.nodes))]
        labs = torch.stack(list(self.mem.values()) + list(self.labels))
        i = int((labs @ want).argmax())
        if float(labs[i] @ want) >= .99 or not make:
            return names[i]
        self.nodes.append(G.GNode().to(dev))
        self.labels.append(want.detach().clone())
        return f"n{len(self.nodes) - 1}"


def key_lesson(M, groups):
    """attribute words the teacher calls the same field together; other fields apart (as the namer's lesson)"""
    keys = sorted({k for _, k in groups})
    lab = M.label([list(g) for g, _ in groups])
    ids = torch.tensor([keys.index(k) for _, k in groups], device=dev)
    same = ids[:, None] == ids[None]
    eye = torch.eye(len(groups), dtype=torch.bool, device=dev)
    sim = (lab @ lab.T / .05).masked_fill(eye, -1e9)
    pos = same & ~eye
    has = pos.any(1)
    nce = -(sim.log_softmax(1) * pos).sum(1)[has] / pos.sum(1)[has]
    cos = lab @ lab.T
    return nce.mean() + F.relu(.9 - cos[pos]).mean() + F.relu(cos[~same] - .5).mean()


def train(M, epochs=150):
    fs, attrs = article_firings(DEMO)
    for f in fs:
        M.route(bvec(f[0]), make=True)
    n_sent = sum(len(a) for a in DEMO.values())
    print(f"demos: {len(DEMO)} articles, {n_sent} sentences, {sum(1 for f in fs if f[5]['want'] == 'keep')} facts, "
          f"{len(fs)} firings; nodes made: {len(M.nodes)}; fields named: {len({k for _, k in attrs})}", flush=True)
    allp = torch.stack([bvec(p) for p in BKEYS])
    opt = torch.optim.AdamW(M.parameters(), lr=1e-3)
    node = M.nodes[int(M.route(bvec("fact"))[1:])]
    for ep in range(epochs):
        random.shuffle(fs)
        tot = 0.
        for b in range(0, len(fs), 64):
            ff = fs[b:b + 64]
            o = node([f[1] for f in ff], [f[2] for f in ff], [f[3] for f in ff], [f[4] for f in ff])
            L = F.cross_entropy(torch.cat(o["tags"]), torch.tensor([t for f in ff for t in f[5]["tags"]], device=dev))
            L = L + F.cross_entropy(o["want"] @ allp.T / .05, torch.tensor([BKEYS.index(f[5]["want"]) for f in ff], device=dev))
            k = [i for i, f in enumerate(ff) if "choice" in f[5]]
            if k:
                L = L + F.cross_entropy(o["choice"][k][:, :2], torch.tensor([ff[i][5]["choice"] for i in k], device=dev))
            L = L + key_lesson(M, random.sample(attrs, min(128, len(attrs))))
            opt.zero_grad()
            L.backward()
            opt.step()
            tot += float(L.detach())
        if (ep + 1) % 25 == 0:
            print(f"epoch {ep + 1}: loss {tot:.2f}", flush=True)
    with torch.no_grad():                       # each field the teacher named: its words' labels (to name a field)
        keys = sorted({k for _, k in attrs})
        lab = M.label([list(g) for g, _ in attrs])
        M.key_names = keys
        M.key_labs = F.normalize(torch.stack([lab[[i for i, (_, k) in enumerate(attrs) if k == kk]].mean(0) for kk in keys]), dim=-1)


# ====================================================================== reading an article into nodes
class Memory:
    """things (nodes): name, label, fields; a field = its attribute words, label, values (oldest first: edits kept)"""

    def __init__(self, M):
        self.M, self.things = M, []

    def thing(self, name, make=True):
        lab = self.M.label([name])[0]
        for i, t in enumerate(self.things):
            if float(t["lab"] @ lab) > JOIN:
                return i
        if not make:
            return -1
        self.things.append(dict(name=name, lab=lab, fields=[]))
        return len(self.things) - 1

    def keep(self, ti, attr, val):
        lab = self.M.label([attr])[0] if attr else None
        T = self.things[ti]
        for f in T["fields"]:
            if lab is not None and float(f["lab"] @ lab) > JOIN:
                f["vals"].append(val)
                f["said"].append(attr)
                return "edit"
        T["fields"].append(dict(lab=lab, said=[attr], vals=[val]))
        return "new"

    @torch.no_grad()
    def read(self, title, sentence, trace=None):
        node = self.M.nodes[int(self.M.route(self.M.fact_want)[1:])]
        topic = self.thing(tok(title))
        cur, kept = tok(sentence), []
        for _ in range(8):
            o = node([cur], [tok(title)], [[]], [0])
            tags = o["tags"][0].argmax(-1).tolist()
            want = self.M.route(o["want"][0])
            if want == "done":
                break
            th = [t for t, g in zip(cur, tags) if g == 1]
            attr = [t for t, g in zip(cur, tags) if g == 2]
            val = [t for t, g in zip(cur, tags) if g == 3]
            if not val:
                break                              # (nothing marked: no step)
            subj = int(o["choice"][0][:2].argmax())
            ti = topic if (subj == 1 or not th) else self.thing(th)
            how = self.keep(ti, attr, val)
            kept.append((ti == topic, attr, val, how))
            if trace is not None:
                trace.append(f"      [{'subject' if ti == topic else ' '.join(self.things[ti]['name'])}] {' '.join(attr)} -> {' '.join(val)} ({how})")
            cur = [t for t, g in zip(cur, tags) if g not in (2, 3)]
        return kept


def field_key(M, attr):
    if not attr:
        return None
    lab = M.label([attr])[0]
    return M.key_names[int((M.key_labs @ lab).argmax())]


def f1(a, b):
    a, b = list(a), list(b)
    common = sum(min(a.count(t), b.count(t)) for t in set(a))
    return 0 if not common else 2 * common / (len(a) + len(b))


@torch.no_grad()
def grade(M, data, show=()):
    tot = dict(gold=0, val=0, exact=0, key=0, extracted=0, precise=0)
    for title, ann in data.items():
        mem = Memory(M)
        got = []
        lines = []
        for si, facts in sorted(ann.items()):
            s = ARTS[title][si]
            tr = []
            kept = mem.read(title, s, tr)
            got += [(si,) + k for k in kept]
            if title in show:
                lines.append(f"    S{si}: {s[:150]}")
                lines += tr
        gold = [(si, f) for si, facts in ann.items() for f in facts]
        for si, f in gold:
            gv = tok(f[4].split("#")[0])
            best = max([(f1(gv, k[3]), k) for k in got if k[0] == si and k[1] == bool(f[1])], default=(0, None), key=lambda x: x[0])
            tot["gold"] += 1
            if best[0] >= .5:
                tot["val"] += 1
                tot["exact"] += best[0] == 1
                tot["key"] += field_key(M, best[1][2]) == f[3]
        for k in got:
            tot["extracted"] += 1
            tot["precise"] += any(f1(tok(f[4].split("#")[0]), k[3]) >= .5 for si, f in gold if si == k[0])
        if title in show:
            print(f"\n  == {title}: sentences read one fact per firing", flush=True)
            print("\n".join(lines))
            print(f"  -- its nodes:")
            for t in mem.things:
                print(f"    NODE {' '.join(t['name'])}")
                for fd in t["fields"]:
                    print(f"       {field_key(M, fd['said'][0]) or '?':>15} | said as: {' / '.join(' '.join(x) for x in fd['said'])} | value(s): {' ; '.join(' '.join(v) for v in fd['vals'])}")
    return tot


@torch.no_grad()
def lockstep(M, data):
    """the teacher's own steps on held-out sentences: per firing, which decision the read node gets wrong"""
    fs, _ = article_firings(data)
    node = M.nodes[int(M.route(M.fact_want)[1:])]
    c = dict(firings=0, want=0, want_done_gold=0, want_done_ok=0, choice=0, n_choice=0)
    tg = {t: [0, 0, 0] for t in (1, 2, 3)}             # per tag: hit, gold, marked
    for f in fs:
        o = node([f[1]], [f[2]], [f[3]], [f[4]])
        w = M.route(o["want"][0])
        c["firings"] += 1
        c["want"] += w == f[5]["want"]
        if f[5]["want"] == "done":
            c["want_done_gold"] += 1
            c["want_done_ok"] += w == "done"
        if "choice" in f[5]:
            c["n_choice"] += 1
            c["choice"] += int(o["choice"][0][:2].argmax()) == f[5]["choice"]
        if f[5]["want"] == "keep":
            pred = o["tags"][0].argmax(-1).tolist()
            for t in (1, 2, 3):
                tg[t][0] += sum(p == t and g == t for p, g in zip(pred, f[5]["tags"]))
                tg[t][1] += sum(g == t for g in f[5]["tags"])
                tg[t][2] += sum(p == t for p in pred)
    print(f"lockstep: firings {c['firings']} | want right {c['want']} (of 'nothing more' firings: {c['want_done_ok']}/{c['want_done_gold']}; "
          f"of 'keep' firings: {c['want'] - c['want_done_ok']}/{c['firings'] - c['want_done_gold']}) | subject choice {c['choice']}/{c['n_choice']} | "
          + ", ".join(f"{TAGS[t]} recall {h}/{g} precision {h}/{m}" for t, (h, g, m) in tg.items()), flush=True)


if __name__ == "__main__":
    torch.manual_seed(0)
    random.seed(0)
    fs, _ = article_firings(DEMO)
    gs, _ = article_firings(GOLD)
    print(f"annotations resolve: demo {len(fs)} firings, held-out {len(gs)} firings", flush=True)
    M = Bio().to(dev)
    t0 = time.time()
    train(M, int(sys.argv[1]) if len(sys.argv) > 1 else 150)
    print(f"trained {time.time() - t0:.0f}s", flush=True)
    lockstep(M, GOLD)
    if os.environ.get("SHOW", "1") == "0":
        sys.exit()
    for name, data, show in (("demo articles (seen)", DEMO, ()), ("HELD-OUT articles", GOLD, tuple(GOLD))):
        t = grade(M, data, show)
        print(f"\n{name}: gold facts {t['gold']} | value found {t['val']} (exact {t['exact']}) | field named right {t['key']} | "
              f"facts kept {t['extracted']}, of them a gold value {t['precise']}", flush=True)
