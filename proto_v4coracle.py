"""User 11-14: "based on the assumption that each newly created node have perfect relation vector. A new info like
text (that is input) and could be a question could be activated and fire the related nodes".
The same trained module (v4c_M.pt), every relation given perfectly by the world: a statement goes to its thing's
place, a fact's entry relates to the rule in force for its attribute, a copy relates to its source, a question's
phrase relates to the place it names. Everything the nodes DO stays learned (reader's fields, fold, columns, scanner).
Measured side by side with the learned relations: what is lost to relations, what to actions."""
import random, torch
import proto_v4c as V

M = V.Module().to(V.dev)
M.load_state_dict(torch.load(f"v4c_{V.VECS}_M.pt"))
M.eval()


class Oracle(V.Memory):
    """relations perfect: a name relates to its own thing's place, an attribute to its rule (exact, by the world)"""

    def __init__(self, M):
        super().__init__(M)
        self.key = {}

    def place_of(self, name, make=False):
        k = tuple(name)
        if k in self.key:
            return self.key[k]
        if not make:
            return -1
        self.places.append((name, None, []))
        self.key[k] = len(self.places) - 1
        return self.key[k]

    def rule_at(self, attr, t):
        for a, _, ents in self.rules:
            if tuple(a) == tuple(attr):
                old = [h for tt, h in ents if tt < t]
                return old[-1] if old else None
        return None

    @torch.no_grad()
    def hear_true(self, text, meta):
        """the statement's node relates to the right place / rule (by the world); its fields are the reader's"""
        self.t += 1
        self.memo = {}
        ts = V.tok(text)
        lg, h = self.M.reader(ts)
        tags = [V.RTAGS[i] for i in lg.argmax(-1).tolist()]
        sm = self.M.summary(h)
        field = lambda names: [t for t, g in zip(ts, tags) if g in names]
        if meta[0] == "rule":
            attr = V.tok(V.words(meta[1]))
            for a, _, ents in self.rules:
                if tuple(a) == tuple(attr):
                    ents.append((self.t, sm))
                    return
            self.rules.append((attr, None, [(self.t, sm)]))
            return
        name = V.tok(V.leaf(meta[1], meta[2]))
        import re as _re
        m = _re.search(r"\(same as (.+)\)\.$", text)          # (its relation to its source, given)
        dep = V.tok(m.group(1)) if m else []
        p = self.place_of(name, make=True)
        self.places[p][2].append(dict(t=self.t, val=field({"VAL"}), src=dep, h=sm, attr=V.tok(V.words(meta[2]))))


@torch.no_grad()
def score(seed, oracle, nq=48):
    rng = random.Random(seed)
    w = V.new_world(rng)
    mem = Oracle(M) if oracle else V.Memory(M)
    for _ in range(30):
        for dd in V.world_step(w, rng):
            o = dd()
            for text, meta in V.said(o, w) if o else []:
                (mem.hear_true(text, meta) if oracle else mem.hear(text))
    facts = [(nm, k, v) for nm, e in w.ents.items() for k, v in e.items()]
    held = sum((mem.value(V.tok(V.leaf(nm, k)))[0] or []) == V.tok(str(v)) for nm, k, v in facts)
    res = {}
    for kind, ops in (("lookup", 0), ("before", 0), ("coexist", 0), ("chain", 0), ("ops", (1, 3)), ("long", (4, 6)), ("long12", (8, 12))):
        hit = n = 0
        for _ in range(nq):
            E = V.q_expr("ops" if kind.startswith("long") else kind, w, rng, ops or (1, 3))
            if E is None:
                continue
            n += 1
            hit += mem.ask(V.ask_text(E)) == str(V.truth_of(E, w))
        res[kind] = (hit, n)
    return held, len(facts), res


if __name__ == "__main__":
    for s in (7, 8):
        for oracle in (False, True):
            held, nf, res = score(50000 + s, oracle)
            print(f"world {s}, relations {'PERFECT (given)' if oracle else 'learned'}: facts held {held}/{nf} | " +
                  ", ".join(f"{k} {a}/{b}" for k, (a, b) in res.items()), flush=True)
