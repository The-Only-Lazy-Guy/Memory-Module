"""GNM: graph memory for Walker. Everything is a node; the only action is a function node, f(nodes) -> nodes.
The model only states what it wants: A to be B.

  python gnm.py --gold      # the correct functions plugged in, no model: does the design solve each task?
"""
import argparse
import random
import re
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ====================================================================== the world (test data only)
def words(key):
    """Key -> lower-case words: camelCase / snake_case / kebab-case / dotted / Title Case."""
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(key))
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", s)
    s = re.sub(r"[_\-./:]+", " ", s)
    return " ".join(s.lower().split())


_SYL = ("ka", "lo", "mer", "vin", "tor", "sa", "ne", "ril", "dun", "ba", "zek", "or", "pel", "mu", "tha", "gor")


def _name(rng, n=2, used=None):
    while True:
        w = "".join(rng.choice(_SYL) for _ in range(n)).capitalize()
        if used is None or w not in used:
            if used is not None:
                used.add(w)
            return w


_ID_WORDS = ("id", "name", "title", "label", "key", "code", "sku", "email", "username", "slug", "handle")
_KEYW = """account address age alias amount app area asset author balance band batch bid bill birth block
board body bonus book branch brand budget build bundle cache camera campaign capacity card cart case category
cell channel chapter charge city class client code color comment company config contact cost count country
coupon course credit currency customer cycle date day deadline degree delay department depth device discount
disk distance doctor domain door draft driver duration edition email engine entry error event expiry fee field
file filter fleet floor flight folder font format frame fuel fund gate genre goal grade group guest height host
hotel hour image income index invoice item job key kind label lane language layer lease lesson level license
limit line link list load locale lock log lot manager margin market match medal member memo menu meter method
mode model module month motor name network node note number office offer order owner package page parent part
partner patient payment period phone pilot plan planet player point policy pool port post price priority
product profile project quota rank rate reason record region release rent report request role room route rule
salary sample scale scene schedule score screen season seat sector seller sensor series server service session
shelf shift ship shop signal site size skill slot source speed sport staff stage station status step stock store
street student studio style subject supplier tag target task tax team temperature tenant term test theme ticket
tier time title token topic total tour track trade train trip unit user value vendor version vessel volume
voucher wallet warehouse weight width window worker year zone""".split()
_OPWORDS = {"max", "min", "total", "avg", "sum", "mean", "count", "highest", "lowest", "largest", "smallest"}
_ADJ_SET = {"base", "last", "next", "primary", "backup", "daily", "net", "first", "main", "old"}
_KEYW = [w for w in dict.fromkeys(_KEYW) if w not in _ID_WORDS and w not in _OPWORDS and w not in _ADJ_SET]
KEYS_TRAIN = _KEYW[0::2]
KEYS_EVAL = _KEYW[1::2]                # held-out vocabulary: never a key in training schemas
_ADJ = tuple(sorted(_ADJ_SET))           # no op words (max/min/total/...) inside key names: they confound the op


def _style(ws, rng):
    s = rng.choice(("snake", "camel", "kebab", "title", "space", "upper_snake"))
    if s == "snake":
        return "_".join(ws)
    if s == "camel":
        return ws[0] + "".join(w.capitalize() for w in ws[1:])
    if s == "kebab":
        return "-".join(ws)
    if s == "title":
        return " ".join(w.capitalize() for w in ws)
    if s == "upper_snake":
        return "_".join(w.upper() for w in ws)
    return " ".join(ws)


def _key(rng, vocab, used):
    for _ in range(50):
        ws = [rng.choice(vocab)] if rng.random() < 0.6 else [rng.choice(_ADJ), rng.choice(vocab)]
        if " ".join(ws) not in used:
            used.add(" ".join(ws))
            return ws
    return [rng.choice(vocab), str(len(used))]


def _value(rng, kind):
    if kind == "int":
        return rng.randint(0, 9999)
    if kind == "float":
        return round(rng.uniform(0, 999), rng.choice((1, 2)))
    if kind == "bool":
        return rng.random() < 0.5
    if kind == "null":
        return None
    if kind == "date":
        return f"20{rng.randint(10, 29)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
    if kind == "email":
        return f"{_name(rng, 2).lower()}@{_name(rng, 2).lower()}.{rng.choice(['com', 'io', 'net', 'org'])}"
    if kind == "code":
        return f"{rng.choice('ABCDEFGHJKLMNPRSTUVWXZ')}{rng.randint(10, 999)}-{rng.choice('ABCDEFGHJKLMNPRSTUVWXZ')}"
    if kind == "url":
        return f"https://{_name(rng, 2).lower()}.com/{_name(rng, 1).lower()}"
    if kind == "name":
        return f"{_name(rng, 2)} {_name(rng, 2)}"
    if kind == "long":
        return " ".join(_name(rng, rng.randint(1, 2)).lower() for _ in range(rng.randint(28, 50))) + "."
    return " ".join(rng.choice(_KEYW) for _ in range(rng.randint(1, 3)))      # phrase


_KINDS = ("int", "float", "bool", "null", "date", "email", "code", "url", "name", "phrase", "phrase", "int", "float")


def random_doc(rng, vocab, depth=3):
    """A random JSON document: nested dicts, lists of records (shared schema, usually with an id/name),
    lists of scalars, mixed scalar types, occasional long text."""
    used = set()

    def rec_schema(n):
        return [(_style(_key(rng, vocab, used), rng), rng.choice(_KINDS[:-2] + ("int", "float")))
                for _ in range(n)]

    def obj(d):
        o = {}
        for _ in range(rng.randint(2, 5)):
            k = _style(_key(rng, vocab, used), rng)
            u = rng.random()
            if d > 0 and u < 0.18:
                o[k] = obj(d - 1)
            elif d > 0 and u < 0.33:
                sch = rec_schema(rng.randint(2, 4))
                idk = rng.choice(("id", "name", "title", "code", None))
                recs = []
                for j in range(rng.randint(2, 5)):
                    r_ = {}
                    if idk == "name":
                        r_["name"] = _name(rng, 2)
                    elif idk == "title":
                        r_["title"] = _name(rng, 2) + " " + _name(rng, 1)
                    elif idk:
                        r_[idk] = _value(rng, "code")
                    for kk, kind in sch:
                        r_[kk] = _value(rng, kind)
                    recs.append(r_)
                o[k] = recs
            elif d > 0 and u < 0.40:
                o[k] = [_value(rng, "phrase") for _ in range(rng.randint(2, 5))]
            else:
                o[k] = _value(rng, "long" if rng.random() < 0.04 else rng.choice(_KINDS))
        return o
    return obj(depth)


TOOLS = [("add", lambda a, b: a + b, "the sum of {0} and {1}"),
         ("sub", lambda a, b: a - b, "the difference between {0} and {1}"),
         ("max", max, "the larger of {0} and {1}"),
         ("min", min, "the smaller of {0} and {1}")]


N_ARITH = 4
HELD_PAIRS = {("max", "sub"), ("min", "add")}
NOISE_KEYS = ("checksum", "trace id", "log ref", "sync token")


# rules (procedures) arrive as text, once: "add" = later values of the key are increments, "set" = they replace.
# Wordings are composed (prefix x subject x verb); one verb per kind is never used in training (eval only).
RULE_PARTS = {"pre": ["", "from now on, ", "note: ", "rule: ", "please remember: "],
              "subj": ["later {k} values", "each new {k}", "incoming {k} numbers", "every reported {k}",
                       "any further {k}", "{k} figures that arrive later"],
              "add": ["are amounts to add to the current {k}", "should be added to the existing {k}",
                      "increase the {k} by that much", "count as increments to the {k}",
                      "get summed into the stored {k}"],
              "set": ["replace the old {k}", "overwrite the previous {k}", "are the new {k}; discard the old one",
                      "become the current {k}", "supersede the stored {k}"],
              "keep": ["are kept alongside the earlier {k}", "join the {k} values already known",
                       "are extra {k} values; keep the old ones too", "do not remove the previous {k}",
                       "coexist with the stored {k}"]}
RULE_HELD = {"add": "get summed into the stored {k}", "set": "supersede the stored {k}",
             "keep": "coexist with the stored {k}"}


def rule_text(rng, kind, k, held=False):
    verbs = [RULE_HELD[kind]] if held else [v for v in RULE_PARTS[kind] if v != RULE_HELD[kind]]
    return (rng.choice(RULE_PARTS["pre"]) + rng.choice(RULE_PARTS["subj"]) + " " + rng.choice(verbs)).format(k=words(k))


# ====================================================================== the world (truth) and its input streams
class World:
    """Truth of the one world: entities with numeric and entity-valued attributes, and every value's history.
    Pending items (what is to be remembered) arrive as small record frames and text corrections. Some attributes
    hold several values at once (multi keys: coexist, the attach case); some values are copies of another fact
    (dependents: when the source changes, the dependent follows — the purpose of r)."""

    def __init__(self, rng, n_num=40, n_ent=6, n_groups=4, max_ents=150, attrs=8, n_multi=6):
        self.max_ents, self.attrs, self.schema = max_ents, attrs, {}
        used = set()
        mk = lambda: _style(_key(rng, KEYS_TRAIN, used), rng)
        self.num_keys = [mk() for _ in range(n_num)]
        self.ent_keys = [mk() for _ in range(n_ent)]
        self.multi_keys = [mk() for _ in range(n_multi)]
        self.groups = [mk() for _ in range(n_groups)]
        self.noise = [_style(k.split(), rng) for k in NOISE_KEYS]
        self.names, self.ents, self.hist = set(), {}, {}     # hist[(name, key)] = list of values, oldest first
        self.multi = {}                                       # (name, multi key) -> coexisting values (max 2)
        self.co_new = set()                                   # (name, multi key, value): a SECOND value, not yet written
        self.multi_max = 2                                    # coexisting values per multi fact (--multi-max)
        self.p_set = 1 / 3                                    # --rule-set-frac: a new numeric rule is "set" (else "add")
        self.keep_rules = False                               # --proc-keep: a multi key coexists only once its
        #                                                       "keep" rule was announced (before: restated only)
        self.deps = {}                                        # (name, key) -> (source name, source key)
        self.policy = {}                                      # key -> "add" | "set" (announced once by a rule frame)
        self.side = {}                                        # id(frame) -> {(record, key): truth} (bookkeeping)
        self.held_keys = [_style([w], rng) for w in KEYS_EVAL[:30]]  # never-seen attribute names (eval only)

    def new_entity(self, rng, held=False):
        nm = _name(rng, 2, self.names)
        self.ents[nm] = {}
        nums = self.held_keys if held else self.num_keys
        self.schema[nm] = (rng.sample(nums, self.attrs - 3) + rng.sample(self.ent_keys, 2)
                           + rng.sample(self.multi_keys, 1))
        return nm

    def _set(self, nm, k, v, depth=0):
        self.ents[nm][k] = v
        self.hist.setdefault((nm, k), []).append(v)
        for (dn, dk), src in sorted(self.deps.items()):
            if src == (nm, k) and self.ents[dn].get(dk) != v and depth < 4:
                self._set(dn, dk, v, depth + 1)                # the dependent follows its source

    def set(self, nm, k, v):
        """a direct statement of the value: it is no longer a copy of anything"""
        self.deps.pop((nm, k), None)
        self._set(nm, k, v)

    def _value(self, rng, nm, k):
        if k in self.ent_keys:
            others = sorted(set(self.ents) - {nm})
            return rng.choice(others) if others else None
        return rng.randint(1, 30)

    def rule(self, rng, k=None, kind=None, held=False, p_change=0.3):
        """a rule frame: how later values of a key are to be written (shown ONCE); rules also change over time"""
        if k is None:
            free = [x for x in self.num_keys + (self.multi_keys if self.keep_rules else []) if x not in self.policy]
            chg = sorted(x for x in self.policy if self.policy[x] != "keep")   # a multi key stays multi
            if free and (rng.random() >= p_change or not chg):
                k = rng.choice(sorted(free))
            elif chg:
                k = rng.choice(chg)
                kind = kind or ("set" if self.policy[k] == "add" else "add")
            else:
                return None
        kind = kind or ("keep" if k in self.multi_keys else rng.choice(["add", "add", "set"]) if abs(self.p_set - 1 / 3) < 1e-9
                        else "set" if rng.random() < self.p_set else "add")   # (1/3: the old draw, same random stream)
        self.policy[k] = kind
        return {"rules": {k: rule_text(rng, kind, k, held)}}

    def _update(self, rng, nm, k, side, i):
        """a new statement of an existing numeric fact, following the key's rule; returns the value in the frame"""
        old = self.ents[nm][k]
        if self.policy.get(k) == "add":
            d = rng.randint(1, 9)
            self.set(nm, k, old + d)
            side[(i, k)] = ("delta", nm, k, d, old + d)
            return d
        v = rng.choice([x for x in range(1, 31) if x != old])
        self.set(nm, k, v)
        return v

    def _sources(self, nm, k):
        out, nums = [], set(self.num_keys + self.held_keys)
        for a in sorted(self.ents):
            if a == nm:
                continue
            for k1 in sorted(self.ents[a]):
                if k1 not in nums:
                    continue
                s, seen = (a, k1), 0
                while s in self.deps and seen < 8:            # no cycles
                    s, seen = self.deps[s], seen + 1
                if s != (nm, k):
                    out.append((a, k1))
        return out

    def _depend(self, rng, nm, k):
        """make (nm, k) a copy of another fact; returns the value as written in the frame"""
        srcs = self._sources(nm, k)
        if not srcs:
            return None
        a, k1 = rng.choice(srcs)
        v = self.ents[a][k1]
        self.deps[(nm, k)] = (a, k1)
        self._set(nm, k, v)
        return f"{v} (same as {leaf(a, k1)})"

    def component(self, rng, p_new=0.5, p_update=0.3, only_update=False, p_noise=0.3, new_ent=False, held=False,
                  p_dep=0.2):
        """One pending frame: 1-2 records; each a new entity or an existing one with new / updated / restated /
        coexisting / dependent attributes (occasionally a noise field nobody asks about)."""
        recs, side = [], {}
        nums = self.num_keys + self.held_keys
        for _ in range(rng.randint(1, 2)):
            if only_update:
                nm = rng.choice(sorted(n for n, e in self.ents.items() if e))
                k = rng.choice(sorted(self.ents[nm]))
                if k in self.ent_keys:
                    v = rng.choice(sorted(set(self.ents) - {nm}))
                    self.set(nm, k, v)
                else:
                    v = self._update(rng, nm, k, side, len(recs))
                recs.append({"name": nm, k: v})
                continue
            if new_ent or not self.ents or (rng.random() < p_new and len(self.ents) < self.max_ents):
                nm = self.new_entity(rng, held)
            else:
                nm = rng.choice(sorted(self.ents))
            e = self.ents[nm]
            rec = {"name": nm}
            for _ in range(rng.randint(1, 3)):
                upd = [k for k in e if rng.random() < p_update]
                k = rng.choice(upd) if upd and rng.random() < 0.5 else rng.choice(self.schema[nm])
                if k in rec:
                    continue
                if k in self.multi_keys:                       # coexisting values: a second one adds, not replaces
                    vs = self.multi.setdefault((nm, k), [])
                    if vs and (len(vs) >= self.multi_max or rng.random() < 0.3
                               or (self.keep_rules and self.policy.get(k) != "keep")):
                        rec[k] = rng.choice(vs)
                    else:
                        vs.append(rng.choice([x for x in range(1, 31) if x not in vs]))
                        rec[k] = vs[-1]
                        if len(vs) >= 2:
                            self.co_new.add((nm, k, vs[-1]))
                    continue
                if k in nums and k not in e and rng.random() < p_dep:
                    w = self._depend(rng, nm, k)
                    if w is not None:
                        rec[k] = w
                        continue
                if k in e and k in nums and (self.policy.get(k) == "add" or rng.random() >= 0.3):
                    rec[k] = self._update(rng, nm, k, side, len(recs))      # an update, by the key's rule
                    continue
                v = e[k] if k in e else self._value(rng, nm, k)              # restated or new
                if v is None:
                    continue
                rec[k] = v
                if e.get(k) != v:
                    self.set(nm, k, v)
            if rng.random() < p_noise:
                rec[rng.choice(self.noise)] = _value(rng, "code")
            recs.append(rec)
        obj = {rng.choice(self.groups): recs}
        if side:
            self.side[id(obj)] = side
        return obj

    def update_one(self, rng, nm, k):
        """eval: one new statement of a fact, following its rule"""
        side = {}
        v = self._update(rng, nm, k, side, 0)
        obj = {rng.choice(self.groups): [{"name": nm, k: v}]}
        if side:
            self.side[id(obj)] = side
        return obj

    def dependent(self, rng, held=False):
        """eval: a new entity whose attribute is a copy of an existing fact; returns (frame, (nm, k)) or None"""
        nm = self.new_entity(rng, held)
        k = rng.choice([x for x in self.schema[nm] if x in self.num_keys + self.held_keys])
        w = self._depend(rng, nm, k)
        return ({rng.choice(self.groups): [{"name": nm, k: w}]}, (nm, k)) if w else None

    def restate_new(self, rng, nm, k):
        """a new value for one fact (eval: update a source)"""
        v = rng.choice([x for x in range(1, 31) if x != self.ents[nm][k]])
        self.set(nm, k, v)
        return {rng.choice(self.groups): [{"name": nm, k: v}]}

    def coexist(self, rng):
        """eval: a second value for a multi key that already has one; returns (frame, (nm, k)) or None"""
        c = sorted(f for f, vs in self.multi.items() if 1 <= len(vs) < self.multi_max
                   and (not self.keep_rules or self.policy.get(f[1]) == "keep"))
        if not c:
            return None
        nm, k = rng.choice(c)
        vs = self.multi[(nm, k)]
        vs.append(rng.choice([x for x in range(1, 31) if x not in vs]))
        self.co_new.add((nm, k, vs[-1]))
        return {rng.choice(self.groups): [{"name": nm, k: vs[-1]}]}, (nm, k)

    def correction(self, rng, among=None):
        """A text note that changes a fact: the loop must find the fact and write the new value into it."""
        if among:
            nm, k = rng.choice(sorted(among))
        else:
            nm = rng.choice(sorted(n for n, e in self.ents.items() if any(k in self.num_keys for k in e)))
            k = rng.choice(sorted(k for k in self.ents[nm] if k in self.num_keys))
        old = self.ents[nm][k]
        v = rng.choice([x for x in range(1, 31) if x != old])
        self.set(nm, k, v)
        return {"note": f"Correction: {leaf(nm, k)} is now {v}."}, (nm, k, old, v)


def state_frame(rng):
    """What the model also sees: any JSON state (never asked about; the model decides whether to keep any of it)."""
    return random_doc(rng, KEYS_TRAIN, depth=rng.randint(1, 2))


def leaf(name, key):
    return f"the {words(key)} of {name}"


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
    return ("op", rng.choice(TOOLS[:N_ARITH])[0], [_tree(rng, leaves, left), _tree(rng, leaves, n - 1 - left)])


def rand_tree(rng, leaves, n, mode):
    for _ in range(200):
        t = _tree(rng, leaves, n)
        if (mode == "train" and has_held(t)) or (mode == "held" and not has_held(t)):
            continue
        return t
    return None


def statements(obj):
    """an input frame as plain statements: JSON never enters the graph"""
    if "rules" in obj:
        return list(obj["rules"].values())
    if "note" in obj:
        return [obj["note"]]
    out = []
    for recs in obj.values():
        for rec in recs:
            out += [f"{leaf(rec['name'], k)} is {v}." for k, v in rec.items() if k != "name"]
    return out


def state_statements(obj, path=()):
    if isinstance(obj, dict):
        return [s for k, v in obj.items() for s in state_statements(v, path + (words(k),))]
    if isinstance(obj, list):
        return [s for v in obj for s in state_statements(v, path)]
    return [f"the {' '.join(path)} is {obj}."]


def number(s):
    try:
        return int(s)
    except ValueError:
        return None


def value_of(text):
    m = re.search(r" is (?:now )?(.+?)(?: \(same as .*\))?\.$", text)
    if not m:
        return None
    return number(m.group(1)) if number(m.group(1)) is not None else m.group(1)


def phrase_of(text):
    m = re.search(r"(the .+?) is ", text)
    return m.group(1) if m else None


# ====================================================================== the graph
class Graph:
    """Places, each a label and a chain of entries (A to be B appends B, A stays). Edges start as "everything to
    everything": i -> j is cut for good when a kept neighbour k of i is closer to j than i is (i reaches j through k)."""

    def __init__(self, d, beam=8, cap=1024):
        self.d, self.beam = d, beam
        self.L = np.zeros((cap, d), dtype=np.float32)
        self.n, self.adj, self.chain, self.cost = 0, [], [], 0

    def search(self, q):
        """walk from the first place toward q; returns every place seen, closest first, as (distance, place)"""
        if self.n == 0:
            return []
        dist = lambda ids: np.linalg.norm(self.L[ids] - q, axis=-1)
        seen, open_ = {0: float(dist([0])[0])}, set()
        self.cost += 1
        while True:
            top = sorted(seen, key=seen.get)[:self.beam]
            u = next((i for i in top if i not in open_), None)
            if u is None:
                return sorted((d, i) for i, d in seen.items())
            open_.add(u)
            new = [v for v in self.adj[u] if v not in seen]
            if new:
                self.cost += len(new)
                seen.update(zip(new, dist(new).tolist()))

    def flat(self, q):
        return int(np.linalg.norm(self.L[:self.n] - q, axis=-1).argmin()) if self.n else -1

    def cut(self, cands):
        """the rule: go through candidates closest first, keep one unless a kept one is already closer to it"""
        kept = []
        for d, j in sorted(cands):
            if all(np.linalg.norm(self.L[k] - self.L[j]) >= d for k in kept):
                kept.append(j)
        return kept

    def add_place(self, label):
        if self.n == len(self.L):
            self.L = np.vstack([self.L, np.zeros_like(self.L)])
        cands = self.search(label)
        i, self.n = self.n, self.n + 1
        self.L[i] = label
        self.chain.append([])
        self.adj.append(self.cut(cands))
        for j in self.adj[i]:                          # the way back (cutting here left new places with no way in)
            self.adj[j].append(i)
        return i


# ====================================================================== the translator (exact stand-in for --gold)
class Oracle:
    """Same fact -> same label, by reading the text exactly. No model: --gold only tests the design around it."""

    def __init__(self, world, d=64):
        self.world, self.d, self.v = world, d, {}
        self.tau = 1e-3

    def _vec(self, key):
        if key not in self.v:
            rand = lambda *x: np.random.default_rng(abs(hash(x)) % 2 ** 32).standard_normal(self.d)
            x = rand("key", key[1]) + rand("name", key[2]) if key[0] == "fact" else rand(*key)   # shared parts, as real labels
            self.v[key] = (x / np.linalg.norm(x)).astype(np.float32)
        return self.v[key]

    def key(self, text):
        m = re.search(r"the (.+?) of ([A-Z][a-z]+)", text)
        if m:
            return ("fact", m.group(1).lower(), m.group(2))
        w = self.world
        for k in sorted((words(x) for x in w.num_keys + w.multi_keys + w.ent_keys + w.held_keys), key=len, reverse=True):
            if not text.startswith("the ") and re.search(rf"\b{re.escape(k)}\b", text.lower()):
                return ("rule", k)
        return ("state", text.split(" is ")[0])

    def label(self, text):
        return self._vec(self.key(text))

    def rule_query(self, text):
        k = self.key(text)
        return self._vec(("rule", k[1])) if k[0] == "fact" else None

    def same(self, a, b):
        return np.linalg.norm(a - b) < self.tau


# ====================================================================== the translator
class Translator(nn.Module):
    """Text <-> node. Each token's gate says how much it counts; the label is the gated mean of the token states, so
    the same thing said two ways lands on one point. GPT-2 stands in for Walker."""

    def __init__(self, hf="gpt2", d=128):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(hf)
        self.tok.pad_token = self.tok.eos_token
        self.lm = AutoModel.from_pretrained(hf).eval()
        self.lm.requires_grad_(False)
        D = self.lm.config.hidden_size
        self.norm = nn.LayerNorm(D, elementwise_affine=False)
        self.gate = nn.Sequential(nn.Linear(D, 256), nn.GELU(), nn.Linear(256, 1))
        self.proj = nn.Sequential(nn.Linear(D, 512), nn.GELU(), nn.Linear(512, d))
        self.rel = nn.Sequential(nn.Linear(2 * d, 256), nn.GELU(), nn.Linear(256, 1))   # the relate tool: same place?
        self.d = d

    def encode(self, texts):
        dev = next(self.proj.parameters()).device
        b = self.tok(texts, return_tensors="pt", padding=True).to(dev)
        with torch.no_grad():
            h = self.norm(self.lm(**b).last_hidden_state)
        w = torch.sigmoid(self.gate(h)) * b["attention_mask"][..., None]
        return F.normalize(self.proj((w * h).sum(1) / w.sum(1).clamp(min=1e-6)), dim=-1)

    def relate(self, a, b):
        return self.rel(torch.cat([(a - b).abs(), a * b], -1)).squeeze(-1)


class NeuralT:
    """the translator as the machine sees it (routing stays exact until the reader exists)"""

    def __init__(self, model, oracle, tau):
        self.m, self.o, self.tau, self.c, self.d = model, oracle, tau, {}, model.d
        self.key = oracle.key

    @torch.no_grad()
    def _l(self, text):
        if text not in self.c:
            self.c[text] = self.m.encode([text])
        return self.c[text]

    def label(self, text):
        return self._l(text)[0].cpu().numpy().astype(np.float32)

    def rule_query(self, text):
        k = self.key(text)
        return self.label(f"rule: {k[1]}") if k[0] == "fact" else None

    @torch.no_grad()
    def same(self, a, b):
        dev = next(self.m.proj.parameters()).device
        return float(self.m.relate(torch.tensor(a, device=dev), torch.tensor(b, device=dev))) > 0


def views(rng, nm, k):
    """one fact, said in the ways the input says it"""
    v, ph = rng.randint(1, 60), leaf(nm, k)
    return rng.choice([ph, ph, f"{ph} is {v}.", f"{ph} is {v}.", f"Correction: {ph} is now {v}.",
                       f"{ph} is {_name(rng, 2)}.", f"{ph} is {v} (same as {leaf(_name(rng, 2), ' '.join(_key(rng, KEYS_TRAIN, set())))})."])


def relate_loss(model, y, x, sim):
    """the tool is shown a true pair and each row's nearest wrong one"""
    n = len(y)
    i = torch.arange(n, device=y.device)
    far = (sim.detach() - 1e4 * torch.eye(n, device=y.device)).argmax(1)
    a, b = torch.cat([y.detach(), y.detach()]), torch.cat([x[i].detach(), x[far].detach()])
    return F.binary_cross_entropy_with_logits(model.relate(a, b), torch.cat([torch.ones(n), torch.zeros(n)]).to(y.device))


def label_loss(model, rng, vocab, n=8):
    """two views of each fact meet; same name / other key and same key / other name are the negatives (grid)"""
    used, ku = set(), set()
    names = [_name(rng, 2, used) for _ in range(n)]
    keys = [" ".join(_key(rng, vocab, ku)) for _ in range(n)]
    facts = [(a, b) for a in names for b in keys]
    side = [f"the {' '.join(p)} is {_value(rng, rng.choice(_KINDS[:-2]))}." for p in (_key(rng, vocab, set()) for _ in range(16))]
    x = model.encode([views(rng, *f) for f in facts] + side)
    y = model.encode([views(rng, *f) for f in facts])
    N = len(facts)
    sim = y @ x.T / .07
    t = torch.arange(N, device=sim.device)
    loss = F.cross_entropy(sim, t) + relate_loss(model, y, x[:N], sim[:, :N])
    rv = lambda k: rng.choice([f"rule: {k}", rule_text(rng, rng.choice(("add", "set", "keep")), k)])
    a = model.encode([rv(k) for k in keys] + [f"the {k} is {rng.randint(1, 60)}." for k in keys] + [leaf(rng.choice(names), k) for k in keys])
    b = model.encode([rv(k) for k in keys])
    return loss + F.cross_entropy(b @ a.T / .07, torch.arange(n, device=sim.device)) + relate_loss(model, b, a[:n], b @ a[:n].T)


@torch.no_grad()
def label_eval(model, rng, vocab, n=40):
    """fresh names and keys: does a phrase find its fact among everything, and how far apart are same and different"""
    used, ku = set(), set()
    names = [_name(rng, 2, used) for _ in range(n)]
    keys = [" ".join(_key(rng, vocab, ku)) for _ in range(n)]
    facts = [(a, b) for a in names for b in keys]
    enc = lambda T: torch.cat([model.encode(T[i:i + 256]) for i in range(0, len(T), 256)])
    S = enc([f"{leaf(*f)} is {rng.randint(1, 60)}." for f in facts])
    out = {}
    for name, T in (("phrase", [leaf(*f) for f in facts]), ("correction", [f"Correction: {leaf(*f)} is now {rng.randint(1, 60)}." for f in facts]),
                    ("copy", [f"{leaf(*f)} is {rng.randint(1, 60)} (same as {leaf(rng.choice(names), rng.choice(keys))})." for f in facts])):
        Q = enc(T)
        C = Q @ S.T
        out[name] = float((C.argmax(1) == torch.arange(len(facts), device=C.device)).float().mean())
        out[name + "_rel+"] = float((model.relate(Q, S) > 0).float().mean())
        out[name + "_rel-"] = float((model.relate(Q, S[(C - 2 * torch.eye(len(facts), device=C.device)).argmax(1)]) <= 0).float().mean())
    R = enc([rule_text(rng, rng.choice(("add", "set", "keep")), k) for k in keys] + [f"the {k} is 3." for k in keys])
    C = enc([f"rule: {k}" for k in keys]) @ R.T
    out["rule"] = float((C.argmax(1) == torch.arange(n, device=C.device)).float().mean())
    Q, S = enc([f"rule: {k}" for k in keys]), R[:n]
    out["rule_rel+"] = float((model.relate(Q, S) > 0).float().mean())
    out["rule_rel-"] = float((model.relate(Q, R[(C - 2 * torch.eye(*C.shape, device=C.device)).argmax(1)]) <= 0).float().mean())
    return out


def train_translator(A):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = Translator().to(dev)
    if A.init:
        model.load_state_dict(torch.load(A.init, map_location=dev), strict=False)
    head = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(head, lr=A.lr)
    rng = random.Random(A.seed)
    t0 = time.time()
    for step in range(1, A.steps + 1):
        loss = label_loss(model, rng, KEYS_TRAIN, A.grid)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % A.every == 0:
            tr, he = label_eval(model, rng, KEYS_TRAIN), label_eval(model, rng, KEYS_EVAL)
            print(f"step {step} loss {float(loss):.3f} | train keys " + " ".join(f"{k} {v:.3f}" for k, v in tr.items()) + f" | {time.time() - t0:.0f}s", flush=True)
            print(f"   held keys " + " ".join(f"{k} {v:.3f}" for k, v in he.items()), flush=True)
            if A.save:
                torch.save({k: v for k, v in model.state_dict().items() if not k.startswith("lm.")}, A.save)


# ====================================================================== function nodes
class Budget(Exception):
    pass


class Item:
    def __init__(self, text, value=None, place=-1, idx=-1, until=None):
        self.text, self.value, self.place, self.idx, self.until = text, value, place, idx, until


class Row:
    """one thought: what it is reading (focus) and what it holds (stack)"""

    def __init__(self, text):
        self.src, self.focus, self.stack, self.passes, self.until = text, [text], [], 0, None


class Machine:
    """f(nodes) -> nodes. Every step of every row is one call of one function."""

    def __init__(self, translator, d=64, budget=400):
        self.T, self.g, self.t, self.budget, self.flat, self.trace = translator, Graph(d), 0, budget, False, None

    def do(self, r, name, *a):
        r.passes += 1
        if r.passes > self.budget:
            raise Budget()
        if self.trace is not None:
            f = a[0] if name == "tool" else "derive" if name == "insert" and a else name
            self.trace.append((view(r), f, a[0] if name == "split" else None))
        return getattr(self, "f_" + name)(r, *a)

    def find(self, label):
        if self.flat:
            j = self.g.flat(label)
        else:
            c = self.g.search(label)
            j = c[0][1] if c else -1
        return j if j >= 0 and self.T.same(label, self.g.L[j]) else -1

    def entry_item(self, p, idx=-1):
        ch = self.g.chain[p]
        idx = idx % len(ch)
        return Item(ch[idx][0], value_of(ch[idx][0]), p, idx, ch[idx + 1][1] if idx + 1 < len(ch) else None)

    def f_split(self, r, n):                           # text -> chunk, rest
        c, r.focus[-1] = r.focus[-1][:n].strip(), r.focus[-1][n:].lstrip()   # no stray spaces: a chunk ends on a token
        if not r.focus[-1]:
            r.focus.pop()
            r.until = None
        m = re.search(r"(?<![\w-])\d+(?![\w-])", c)
        r.stack.append(Item(c, int(m.group()) if m else None))

    def f_open(self, r):                               # a node's text -> what the row reads next
        it = r.stack.pop()
        r.focus.append(it.text)
        r.until = it.until                             # an old copy is read as it was then

    def f_drop(self, r):
        r.stack.pop()

    def f_nip(self, r):
        del r.stack[-2]

    def f_join(self, r):                               # (phrase, entity) -> phrase about that entity
        b, a = r.stack.pop(), r.stack.pop()
        r.stack.append(Item(f"{a.text.strip()} {b.value}"))

    def f_say(self, r):                                # the row is done; its answer is what it holds
        pass

    def f_lookup(self, r):                             # phrase -> newest entry of its place (as of the row's time)
        p = self.find(self.T.label(r.stack.pop().text))
        idx = -1
        if p >= 0 and r.until is not None:
            idx = sum(t < r.until for _, t in self.g.chain[p]) - 1
            if idx < 0:
                p = -1
        r.stack.append(self.entry_item(p, idx) if p >= 0 else Item("unknown"))

    def f_rule(self, r):                               # statement -> the rule that governs it
        q = self.T.rule_query(r.src)
        p = self.find(q) if q is not None else -1
        r.stack.append(self.entry_item(p) if p >= 0 else Item("no rule"))

    def f_prev(self, r):                               # entry -> the entry it came from
        it = r.stack[-1]
        r.stack.append(self.entry_item(it.place, it.idx - 1) if it.place >= 0 and it.idx > 0 else Item("unknown"))

    def f_tool(self, r, name):                         # (a, b) -> result, exact
        st, vals, i = r.stack, [], len(r.stack) - 1
        while i >= 0 and len(vals) < 2:
            if st[i].value is not None:
                vals.append(st[i])
            i -= 1
        del st[i + 1:]
        if st and st[-1].value is None:
            st.pop()
        if len(vals) < 2 or not all(isinstance(x.value, int) for x in vals):
            st.append(Item("unknown"))
            return
        v = next(t for t in TOOLS if t[0] == name)[1](vals[1].value, vals[0].value)
        st.append(Item(str(v), v))

    def f_insert(self, r, derived=False):              # A to be B: a new entry where the statement belongs
        self.t += 1
        label = self.T.label(phrase_of(r.src) or r.src)   # the phrase a lookup would use, not the whole sentence
        p = self.find(label)
        p = p if p >= 0 else self.g.add_place(label)
        text = f"{phrase_of(r.src)} is {r.stack[-1].value}." if derived else r.src
        self.g.chain[p].append((text, self.t))


# ====================================================================== --gold: the correct calls, written by hand
ADD_RE = [re.compile(v.format(k=".*")) for v in RULE_PARTS["add"]]


def gold_follow(m, r):
    """the top entry may say it is a copy of another fact: follow it"""
    seen = set()
    while "(same as" in r.stack[-1].text:
        t = r.stack[-1].text
        if r.stack[-1].place in seen:                  # a cycle: nothing here holds a value
            r.stack[-1] = Item("unknown")
            return
        seen.add(r.stack[-1].place)
        src = re.search(r"\(same as (.+?)\)", t).group(1)
        m.do(r, "open")
        m.do(r, "split", t.index(src))
        m.do(r, "drop")
        m.do(r, "split", len(src))
        m.do(r, "lookup")
        m.do(r, "split", len(t) - t.index(src) - len(src))
        m.do(r, "drop")


def gold_read(m, r, phrase):
    m.do(r, "split", len(phrase))
    m.do(r, "lookup")
    gold_follow(m, r)


def gold_older(m, r):
    """step back until the value differs (a restated value is not a change)"""
    cur = r.stack[-1].value
    while True:
        it = r.stack[-1]
        m.do(r, "prev")
        if it.place >= 0 and it.idx > 0:
            gold_follow(m, r)
        if r.stack[-1].value != cur or r.stack[-1].text == "unknown":
            return
        m.do(r, "nip")


def gold_tree(m, r, t):
    if t[0] == "leaf":
        return gold_read(m, r, leaf(t[1], t[2]))
    pre, rest = next(x for x in TOOLS if x[0] == t[1])[2].split("{0}")
    mid, post = rest.split("{1}")
    lit = [pre, mid, post]
    for i, c in enumerate(t[2]):
        n = len((lit[i] + ("(" if c[0] == "op" else "")).strip())
        if n:
            m.do(r, "split", n)
        gold_tree(m, r, c)
        if c[0] == "op":
            m.do(r, "split", 1)
    if post:
        m.do(r, "split", len(post))
    m.do(r, "tool", t[1])


def arrive(m, text):
    """every input statement starts a row"""
    r = Row(text)
    if m.T.key(text)[0] != "fact" or text.startswith("Correction"):
        return m.do(r, "insert")
    m.do(r, "rule")
    if any(x.search(r.stack[-1].text) for x in ADD_RE) and "(same as" not in text:
        gold_read(m, r, phrase_of(text))
        if r.stack[-1].text != "unknown":
            m.do(r, "split", len(r.focus[-1]))
            m.do(r, "tool", "add")
            return m.do(r, "insert", True)
    m.do(r, "insert")


def question(kind, world, rng, ops=(1, 3)):
    """(text, truth, the correct calls), or None when the world has no such question yet"""
    num = sorted(f for f in world.hist if f[1] in world.num_keys and f[0] in world.ents and f[1] in world.ents[f[0]])
    if not num:
        return None
    if kind == "lookup":
        nm, k = rng.choice(num)
        q = leaf(nm, k)

        def calls(m, r):
            m.do(r, "split", len("What is "))
            gold_read(m, r, q)
        return f"What is {q}?", world.ents[nm][k], calls
    if kind == "before":
        c = [f for f in num if len(set(world.hist[f])) > 1]
        if not c:
            return None
        nm, k = rng.choice(c)
        h, q = world.hist[(nm, k)], leaf(nm, k)

        def calls(m, r):
            m.do(r, "split", len("What was "))
            gold_read(m, r, q)
            gold_older(m, r)
        return f"What was {q} before?", next(v for v in reversed(h) if v != h[-1]), calls
    if kind == "coexist":
        c = sorted(f for f, v in world.multi.items() if len(v) == 2)
        if not c:
            return None
        nm, k = rng.choice(c)
        q = leaf(nm, k)

        def calls(m, r):
            m.do(r, "split", len("What is the sum of both values of "))
            gold_read(m, r, q)
            gold_older(m, r)
            m.do(r, "tool", "add")
        return f"What is the sum of both values of {q}?", sum(world.multi[(nm, k)]), calls
    if kind == "chain":
        c = [(nm, k, k2) for nm, e in sorted(world.ents.items()) for k in sorted(e) if k in world.ent_keys
             for k2 in sorted(world.ents.get(e[k], {})) if k2 in world.num_keys]
        if not c:
            return None
        nm, k, k2 = rng.choice(c)
        a_, b_ = f"the {words(k2)} of ", leaf(nm, k)

        def calls(m, r):
            m.do(r, "split", len("What is "))
            m.do(r, "split", len(a_))
            gold_read(m, r, b_)
            m.do(r, "join")
            m.do(r, "lookup")
            gold_follow(m, r)
        return f"What is {a_}{b_}?", world.ents[world.ents[nm][k]][k2], calls
    t = rand_tree(rng, num, rng.randint(*ops), "any")
    truth = (lambda f: f(f, t))(lambda f, x: world.ents[x[1]][x[2]] if x[0] == "leaf" else
                                next(y for y in TOOLS if y[0] == x[1])[1](f(f, x[2][0]), f(f, x[2][1])))

    def calls(m, r):
        m.do(r, "split", len("What is "))
        gold_tree(m, r, t)
    return f"What is {render(t)}?", truth, calls


def ask(m, kind, world, rng, ops=(1, 3)):
    """(question, truth, what the row holds, passes), by the correct calls"""
    x = question(kind, world, rng, ops)
    if not x:
        return None
    r = Row(x[0])
    x[2](m, r)
    m.do(r, "say")
    return r.src, x[1], next((it.value for it in reversed(r.stack) if it.value is not None), None), r.passes


def new_world(rng):
    w = World(rng)
    w.multi_max, w.p_set = 2, 0.5
    return w


def world_step(world, rng):
    """one step of input, frame by frame: each frame is read as it happens, the world moves on with it"""
    draws = ([lambda: world.rule(rng)] if rng.random() < .6 else []) + [lambda: world.component(rng)] * 3
    if rng.random() < .6:
        draws.append(lambda: (world.coexist(rng) or (None,))[0])
    if rng.random() < .8 and world.ents:
        draws.append(lambda: world.correction(rng)[0])
    return draws


# ====================================================================== the reader: what do I want next?
FUNCS = ["split", "lookup", "rule", "prev", "open", "drop", "nip", "join", "add", "sub", "max", "min", "insert", "derive", "say"]


def view(r):
    """what a pass sees: the top of what it holds, then what it reads next; bounded at every depth"""
    held = " | ".join(it.text[:60].strip() for it in r.stack[-5:])   # [word, a, mid, b, ")"]: the tool needs the word
    base = len(held) + 4
    return f"{held} || {r.focus[-1][:90] if r.focus else ''}", base


class Reader(nn.Module):
    """reads the view, states a want; the want matches a function node. A split also points at where the chunk ends."""

    def __init__(self, tr, d=256):
        super().__init__()
        self.tr = tr
        D = tr.lm.config.hidden_size
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(D, 8, 2048, batch_first=True, activation="gelu"), 2)
        self.want = nn.Linear(D, d)
        self.fn = nn.Parameter(F.normalize(torch.randn(len(FUNCS), d), dim=-1))
        self.tau = nn.Parameter(torch.tensor(20.))
        self.end = nn.Linear(D, 1)

    def forward(self, views):
        dev = self.end.weight.device
        b = self.tr.tok([v[0] for v in views], return_tensors="pt", padding=True, return_offsets_mapping=True)
        off = b.pop("offset_mapping")
        b = b.to(dev)
        with torch.no_grad():
            h = self.tr.lm(**b).last_hidden_state
        h = self.enc(h, src_key_padding_mask=b["attention_mask"] == 0)
        last = b["attention_mask"].sum(1) - 1
        s = h[torch.arange(len(views), device=dev), last]
        logits = self.tau * F.normalize(self.want(s), dim=-1) @ F.normalize(self.fn, dim=-1).T
        return logits, self.end(h).squeeze(-1), off, b["attention_mask"]


def step_targets(views, names, chars, off):
    """a split's end, in tokens: the token whose end is that character"""
    t = []
    for (v, base), n, o in zip(views, chars, off.tolist()):
        if n is None:
            t.append(-1)
            continue
        want = base + n
        ends = [i for i, (a, e) in enumerate(o) if a >= base and e > a]
        t.append(min(ends, key=lambda i: abs(o[i][1] - want)) if ends else -1)
    return torch.tensor(t)


def fresh(rng, step, keys=None):
    """the same demonstrated step about other names, keys and numbers: no new supervision, the skill must not be them"""
    (text, base), name, n = step
    held, focus = text[:base], text[base:]
    seen = {}
    pat = (keys + "|" if keys else "") + r"\b(?:[A-Z][a-z]{2,}|\d+)\b"

    def sub(m):
        w = m.group()
        if w in ("What", "Correction", "The"):
            return w
        if w not in seen:
            seen[w] = str(rng.randint(1, 60)) if w.isdigit() else _name(rng, 2) if w[0].isupper() else " ".join(_key(rng, KEYS_TRAIN, set()))
        return seen[w]
    swap = lambda x: re.sub(pat, sub, x)
    head, tail = (focus[:n], focus[n:]) if n is not None else (focus, "")
    h2, a, b = swap(held), swap(head), swap(tail)
    return (h2 + a + b, len(h2)), name, len(a) if n is not None else None


def reader_loss(rd, batch):
    views, names, chars = zip(*batch)
    logits, ends, off, mask = rd(views)
    dev = logits.device
    y = torch.tensor([FUNCS.index(n) for n in names], device=dev)
    loss = F.cross_entropy(logits, y)
    t = step_targets(views, names, chars, off).to(dev)
    k = t >= 0
    if k.any():
        loss = loss + F.cross_entropy(ends[k].masked_fill(mask[k] == 0, -1e4), t[k])
    return loss, float((logits.argmax(1) == y).float().mean())


@torch.no_grad()
def step_report(rd, bank, rng, keys, n=1024):
    """on demonstrated steps with fresh names: accuracy per function, and of split ends"""
    S = [fresh(rng, x, keys) for x in rng.sample(bank, min(n, len(bank)))]
    hit, tot, eh, et = {}, {}, 0, 0
    for i in range(0, len(S), 128):
        b = S[i:i + 128]
        views, names, chars = zip(*b)
        logits, ends, off, mask = rd(views)
        t = step_targets(views, names, chars, off).to(logits.device)
        e = ends.masked_fill(mask == 0, -1e4).argmax(1)
        for j, name in enumerate(names):
            tot[name] = tot.get(name, 0) + 1
            hit[name] = hit.get(name, 0) + int(FUNCS[int(logits[j].argmax())] == name)
            if t[j] >= 0:
                et += 1
                eh += int(e[j] == t[j])
    return " ".join(f"{k} {hit[k] / tot[k]:.3f}" for k in FUNCS if k in tot) + f" | split end {eh / max(1, et):.3f}"


def run_free(m, rd, r, stop, max_pass=90, show=False):
    """the reader drives the row; returns False when it asks for something impossible"""
    try:
        while r.passes < max_pass:
            v = view(r)
            with torch.no_grad():
                logits, ends, off, mask = rd([v])
            name = FUNCS[int(logits.argmax(1))]
            if show:
                print("     ", repr(v[0][:110]), "->", name, flush=True)
            if name == "split":
                o = off[0].tolist()
                ok = [i for i, (a, e) in enumerate(o) if a >= v[1] and e > a]
                if not ok:
                    return False
                j = max(ok, key=lambda i: float(ends[0, i]))
                m.do(r, "split", o[j][1] - v[1])
            elif name in ("add", "sub", "max", "min"):
                m.do(r, "tool", name)
            elif name == "derive":
                m.do(r, "insert", True)
            else:
                m.do(r, name)
            if name in stop:
                return True
    except (IndexError, AttributeError, Budget, TypeError, ValueError):
        return False
    return False


def collect(world, m, rng, per_kind):
    """the demonstrations: a fixed number of step-by-step rows per kind"""
    bank, count = [], {}
    m.trace = []

    def keep(kind):
        if count.get(kind, 0) < per_kind:
            count[kind] = count.get(kind, 0) + 1
            bank.extend(m.trace)
        m.trace = []

    kinds = ("fact", "delta", "rule", "note", "copy", "lookup", "before", "coexist", "chain", "ops")
    tick = 0
    while tick < 5000 and any(count.get(k, 0) < per_kind for k in kinds):
        tick += 1
        for draw in world_step(world, rng):
            o = draw()
            for s in statements(o) if o else []:
                n = len(bank)
                arrive(m, s)
                names = [x[1] for x in m.trace]
                keep("delta" if "derive" in names else "rule" if m.T.key(s)[0] == "rule" else "note" if s.startswith("Correction")
                     else "copy" if "(same as" in s else "fact")
        for k in kinds[5:]:
            if ask(m, k, world, rng):
                keep(k)
            m.trace = []
    print("demonstrations", count, "steps", len(bank), flush=True)
    return bank


def free_eval(world, m, rd, rng, q):
    out = []
    for kind, ops in (("lookup", 0), ("before", 0), ("coexist", 0), ("chain", 0), ("ops", (1, 3)), ("long", (4, 6))):
        ok = n = 0
        for _ in range(q):
            x = question(kind, world, rng, ops or (1, 3))
            if x:
                r = Row(x[0])
                n += 1
                if run_free(m, rd, r, ("say",), show=_ == 0 and kind == "ops"):
                    ok += next((it.value for it in reversed(r.stack) if it.value is not None), None) == x[1]
        out.append(f"{kind} {ok}/{n}")
    return " | ".join(out)


def train_reader(A):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr = Translator().to(dev).eval()
    tr.load_state_dict(torch.load(A.init, map_location=dev), strict=False)
    rd = Reader(tr).to(dev)
    rng = random.Random(A.seed)
    w0 = new_world(rng)
    bank = collect(w0, Machine(Oracle(w0)), rng, A.demos)
    ks = sorted({words(k) for k in w0.num_keys + w0.multi_keys + w0.ent_keys}, key=len, reverse=True)
    keys = r"\b(?:" + "|".join(re.escape(k) for k in ks) + r")\b"
    opt = torch.optim.AdamW([p for n, p in rd.named_parameters() if not n.startswith("tr.")], lr=A.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, A.steps, eta_min=A.lr / 50)
    t0 = time.time()
    for step in range(1, A.steps + 1):
        rd.train()
        loss, acc = reader_loss(rd, [fresh(rng, x, keys) for x in rng.sample(bank, min(A.batch, len(bank)))])
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % A.every == 0:
            rd.eval()
            print("   per function:", step_report(rd, bank, rng, keys), flush=True)
            w = new_world(rng)
            T = NeuralT(tr, Oracle(w), A.tau)
            m = Machine(T, T.d)
            m.g.beam = A.beam
            placed = rows = 0
            for _ in range(A.world):
                for draw in world_step(w, rng):
                    o = draw()
                    for s in statements(o) if o else []:
                        rows += 1
                        placed += run_free(m, rd, Row(s), ("insert", "derive"))
            print(f"step {step} loss {float(loss):.3f} acc {acc:.3f} | free world: places {m.g.n}, rows done {placed} of {rows} | {free_eval(w, m, rd, rng, A.q)} | {time.time() - t0:.0f}s", flush=True)
            if A.save:
                torch.save({k: v for k, v in rd.state_dict().items() if not k.startswith("tr.")}, A.save)


def gold(A):
    rng = random.Random(A.seed)
    world = new_world(rng)
    T = Oracle(world)
    if A.init:
        model = Translator().to("cuda" if torch.cuda.is_available() else "cpu").eval()
        model.load_state_dict(torch.load(A.init), strict=False)
        T = NeuralT(model, T, A.tau)
    m = Machine(T, T.d)
    m.flat, m.g.beam = A.flat, A.beam
    t0 = time.time()
    for step in range(1, A.steps + 1):
        for _ in range(2):
            for s in state_statements(state_frame(rng)):
                arrive(m, s)
        draws = world_step(world, rng)
        for draw in draws:                             # each frame is read as it happens: the world moves on with it
            o = draw()
            for s in statements(o) if o else []:
                arrive(m, s)
        if step % A.every == 0:
            g, line = m.g, []
            for kind, ops in (("lookup", 0), ("before", 0), ("coexist", 0), ("chain", 0), ("ops", (1, 3)), ("long", (4, 6))):
                res = [x for x in (ask(m, "ops" if kind == "long" else kind, world, rng, ops or (1, 3)) for _ in range(A.q)) if x]
                ok = [x for x in res if x[1] == x[2]]
                line.append(f"{kind} {len(ok)}/{len(res)} ({sum(x[3] for x in res) / max(1, len(res)):.0f}p)")
                for x in [y for y in res if y[1] != y[2]][:A.show]:
                    print("   MISS", kind, x[:3])
            c0, agree = g.cost, 0
            for _ in range(200):
                q = g.L[rng.randrange(g.n)] + .02 * np.random.standard_normal(g.d).astype(np.float32)
                agree += g.search(q)[0][1] == g.flat(q)
            deg = [len(a) for a in g.adj]
            print(f"step {step} | places {g.n} | " + " | ".join(line) + f" | walk = flat {agree}/200, cost {(g.cost - c0) / 200:.0f} "
                  f"of {g.n}, edges {np.mean(deg):.1f} / {max(deg)} | {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    P = argparse.ArgumentParser()
    P.add_argument("--gold", action="store_true")
    P.add_argument("--flat", action="store_true", help="exact scan instead of the walk: separates design from search")
    P.add_argument("--beam", type=int, default=8)
    P.add_argument("--steps", type=int, default=300)
    P.add_argument("--every", type=int, default=100)
    P.add_argument("--q", type=int, default=100)
    P.add_argument("--show", type=int, default=2)
    P.add_argument("--seed", type=int, default=1)
    P.add_argument("--train", action="store_true", help="train the translator's labels")
    P.add_argument("--read", action="store_true", help="train the reader on demonstrations (--init = the translator)")
    P.add_argument("--demos", type=int, default=50)
    P.add_argument("--batch", type=int, default=64)
    P.add_argument("--world", type=int, default=30, help="steps of world the free eval lives through")
    P.add_argument("--init", default="")
    P.add_argument("--save", default="")
    P.add_argument("--log", default="")
    P.add_argument("--lr", type=float, default=3e-4)
    P.add_argument("--grid", type=int, default=8, help="names x keys per training batch: bigger = harder near-misses")
    P.add_argument("--tau", type=float, default=.15, help="closer than this = the same place")
    A = P.parse_args()
    train_reader(A) if A.read else train_translator(A) if A.train else gold(A)
