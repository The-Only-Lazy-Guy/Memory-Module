"""GNM v6 prototype = v5 + LIVING: after a small demo budget, the module answers questions in new worlds by itself.
A question is tried a few times with its own choices sampled (each node's marks, digit, state, order and what its
want routes to); a run whose final answer is right is imitated, firing by firing (outcome is the only teacher; user
11-10: ~50 demos, then outcome). The demos' own firings are rehearsed alongside (fm38: without them living eroded)."""
import random, sys, time
import torch, torch.nn.functional as F
import proto_v5 as V

B, dev = V.B, V.dev
USE = {"scan": ("tags", "want"), "read": ("want",), "add": ("state", "choice", "want"), "sub": ("state", "choice", "want"),
       "max": ("state", "choice", "want"), "min": ("state", "choice", "want"), "cmp": ("state",), "addd": ("digit", "state"),
       "subd": ("digit", "state")}
KEYS = list(V.PURPOSE)
import os
NAMER = os.environ.get("NAMER", "1") == "1"            # the namer lives too (learns from the names it hears)
EXPLORE = os.environ.get("EXPLORE", "unsure")          # unsure: deviate where the node doubts; random: ~15% of firings
PICK = os.environ.get("PICK", "uniform")               # where it deviates: uniform over the other choices, or "odds"


class Memory(V.Memory):
    """v5's memory and run; with T > 0 every choice a node makes is sampled, and every firing is recorded"""

    def __init__(self, M, T=0.):
        super().__init__(M)
        self.T, self.rec, self.p, self.U = T, None, .15, 2.          # p: how many of a try's firings choose differently (11-14:
        #                                                  sampling every choice of a 20-firing run broke what was right;
        #                                                  tries found a better run 0-3 times in 20)

    def labels(self):
        names = list(self.M.mem) + [f"n{i}" for i in range(len(self.M.nodes))]
        return names, torch.stack([self.M.mem[p] for p in self.M.mem] + list(self.M.labels))

    def fire(self, want, seq=(), s1=(), s2=(), st=0):
        nm = self.M.route(want)
        if nm in V.MEMORY:
            return nm, None
        node = self.M.nodes[int(nm[1:])]
        o = node([list(seq)], [list(s1)], [list(s2)], [st])
        o = {k: (v[0] if isinstance(v, list) else v[0] if v is not None else v) for k, v in o.items()}
        key = self.M.purpose_of[nm]
        if "tags" in USE.get(key, ()) and o["tags"] is None:
            return nm, None                            # (a want that meets a node with nothing for it to read: no run)
        chosen = {}
        here = self.T and random.random() < self.p       # this firing tries something else (all its choices)

        def smp(lg):
            if EXPLORE == "unsure":                      # 11-14: a try deviates where the node itself doubts: chance
                if not self.T:                           # ~ its doubt (1 - top probability); then its next choice
                    return int(lg.argmax())
                p = F.softmax(lg, -1)
                top = int(p.argmax())
                if random.random() < min(1., self.U * (1 - float(p[top]))):
                    if PICK == "uniform":                # 11-14: a wrong column is sure of itself (.84) and gives the
                        o_ = [i for i in range(len(p)) if i != top]   # right digit ~0 -- sampling its own odds never
                        return random.choice(o_) if o_ else top       # finds it; any other choice, evenly
                    q = p.clone()
                    q[top] = 0
                    return int(torch.multinomial(q / q.sum(), 1)) if float(q.sum()) > 0 else top
                return top
            return int(torch.multinomial(F.softmax(lg / self.T, -1), 1)) if here else int(lg.argmax())
        for h in USE.get(key, ()):
            if h == "tags":
                t = [smp(r) for r in o["tags"]]
                o["tags"] = F.one_hot(torch.tensor(t, device=dev), len(B.QTAGS)).float()
                chosen["tags"] = t
            elif h == "want":
                names, labs = self.labels()
                i = smp(labs @ o["want"] / .05)
                o["want"] = labs[i]
                chosen["want"] = self.M.purpose_of.get(names[i], names[i])
            else:
                n_ = {"state": {"cmp": 3, "addd": 2, "subd": 2}.get(key, 2), "choice": 2, "digit": 10}[h]
                c = smp(o[h][:n_])
                o[h] = F.one_hot(torch.tensor(c, device=dev), len(o[h])).float()
                chosen[h] = c
        if self.rec is not None:
            self.rec.append((key, list(seq), list(s1), list(s2), st, chosen))
        return nm, o


def imitate(M, recs, demo_fs, opt, n_demo=400):
    """the right runs' firings are what the nodes learn to do; the demos' firings rehearsed alongside"""
    allp = torch.stack([V.pvec(p) for p in KEYS])
    ff_all = [r for r in recs] + random.sample(demo_fs, min(n_demo, len(demo_fs)))
    by = {}
    for f in ff_all:
        by.setdefault(f[0], []).append(f)
    L = 0
    for key, ff in by.items():
        node = M.nodes[int(M.route(V.pvec(key))[1:])]
        o = node([f[1] for f in ff], [f[2] for f in ff], [f[3] for f in ff], [f[4] for f in ff])
        for h in ("digit", "state", "choice"):
            k = [i for i, f in enumerate(ff) if h in f[5]]
            if k:
                L = L + F.cross_entropy(o[h][k], torch.tensor([ff[i][5][h] for i in k], device=dev))
        k = [i for i, f in enumerate(ff) if "want" in f[5]]
        if k:
            L = L + F.cross_entropy(o["want"][k] @ allp.T / .05, torch.tensor([KEYS.index(ff[i][5]["want"]) for i in k], device=dev))
        k = [i for i, f in enumerate(ff) if "tags" in f[5]]
        if k:
            L = L + F.cross_entropy(torch.cat([o["tags"][i] for i in k]), torch.tensor([t for i in k for t in ff[i][5]["tags"]], device=dev))
    opt.zero_grad()
    L.backward()
    opt.step()
    return float(L.detach())


def heard_groups(M, texts):
    """the names the module itself read in what it heard (its reader's own tags): the namer's lesson needs no teacher
    -- a name said again goes together, different names apart (11-14: every wrong fact of the 5-demo model was two
    names merged in one place, e.g. 'the shelf of Orzek' into 'the part of Orzek'; living never touched the namer)"""
    groups = []
    with torch.no_grad():
        for text in texts:
            ts = B.tok(text)
            lg, _ = M.reader(ts)
            tags = [B.RTAGS[i] for i in lg.argmax(-1).tolist()]
            for fl in (("NAME", "NATTR"), ("NATTR",), ("SRC",), ("RATTR",)):
                g = tuple(t for t, x in zip(ts, tags) if x in fl)
                if g:
                    groups.append(g)
    return groups


def live(M, demo_fs, worlds, tries=12, T=1., nq=24, seed0=70000):
    opt = torch.optim.AdamW([p for n in M.nodes for p in n.parameters()] + list(M.namer.parameters()), lr=1e-3)
    tally = {}                                         # (the reader's tags stay as the demos left them)
    for wi in range(worlds):
        rng = random.Random(seed0 + wi)
        w = B.new_world(rng)
        mem = Memory(M)
        heard = []
        with torch.no_grad():
            for _ in range(30):
                for dd in B.world_step(w, rng):
                    o = dd()
                    for text, meta in B.said(o, w) if o else []:
                        mem.hear(text)
                        heard.append(text)
        if NAMER:
            groups = heard_groups(M, heard)
            for _ in range(20):                         # the namer learns from this world's names before it is asked
                L = B.name_lesson(M, random.sample(groups, min(400, len(groups))))
                opt.zero_grad()
                L.backward()
                opt.step()
            mem = Memory(M)                             # and hears the world again with what it learned
            with torch.no_grad():
                for text in heard:
                    mem.hear(text)
        recs = []
        for kind, ops in (("lookup", 0), ("before", 0), ("coexist", 0), ("chain", 0), ("ops", (1, 3)), ("long", (4, 6))) * (nq // 6):
            E = B.q_expr("ops" if kind == "long" else kind, w, rng, ops or (1, 3))
            if E is None:
                continue
            truth = str(B.truth_of(E, w))
            c = tally.setdefault(kind, [0, 0, 0])
            c[2] += 1
            best = None
            for t in range(tries + 1):                 # its own run first, then sampled tries
                mem.T, mem.rec, mem.memo = (0. if t == 0 else T), [], {}
                with torch.no_grad():
                    got = mem.ask(B.ask_text(E))
                if got == truth and (best is None or len(mem.rec) < len(best)):
                    best = list(mem.rec)
                if t == 0:
                    c[0] += got == truth
            c[1] += best is not None
            mem.T, mem.rec, mem.memo = 0., None, {}
            if best:
                recs += best
        for _ in range(20 if recs else 0):          # (11-14: one step per world was 40 steps in all -- nothing moved)
            imitate(M, recs, demo_fs, opt)
        if (wi + 1) % 5 == 0:
            print(f"lived {wi + 1} worlds: own run right / some run right: " +
                  ", ".join(f"{k} {a}/{n} ({b})" for k, (a, b, n) in tally.items()), flush=True)
            tally = {}


if __name__ == "__main__":
    demo_worlds = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    live_worlds = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    torch.manual_seed(0)
    M = V.Module().to(dev)
    M.scan_want, M.read_want, M.cmp_want = V.pvec("scan"), V.pvec("read"), V.pvec("cmp")
    t0 = time.time()
    V.train(M, demo_worlds, 30)
    M.purpose_of = {M.route(V.pvec(p)): p for p in V.PURPOSE}
    demo_fs = V.firings(M, [B.demo_world(1000 + i, 30, 24) for i in range(demo_worlds)])
    print(f"demos done {time.time() - t0:.0f}s", flush=True)
    for s in (7, 8):
        held, nf, res = V.evaluate(M, 50000 + s)
        print(f"BEFORE living, fresh world {s}: facts held {held}/{nf} | " + ", ".join(f"{k} {a}/{b}" for k, (a, b) in res.items()), flush=True)
    live(M, demo_fs, live_worlds)
    for s in (7, 8):
        held, nf, res = V.evaluate(M, 50000 + s)
        print(f"AFTER living, fresh world {s}: facts held {held}/{nf} | " + ", ".join(f"{k} {a}/{b}" for k, (a, b) in res.items()), flush=True)
    print(f"{time.time() - t0:.0f}s", flush=True)
