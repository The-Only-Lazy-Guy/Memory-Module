"""GNM: graph memory for Walker (user design 11-04). Everything is a node (label L, relation R, value B); a function
is a node too and takes functions and nodes, gives nodes. Signals fire functions; a filter decides what reaches each
recipient; text becomes a node once, when it arrives.

  python gnm.py --fm --init nt2.pt      # codec stage 2, demos, then a fresh world it lives in by itself
"""
import argparse
import copy
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
                else:
                    self.deps.pop((nm, k), None)               # restated directly: no longer a copy either
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
    m = re.search(r"(?:^|\s)is (?:now )?(.+?)(?: \(same as .*\))?\.$", text)   # also the rest of a split statement: "is 5."
    if not m:
        return None
    return number(m.group(1)) if number(m.group(1)) is not None else m.group(1)


def value_in(text):
    """what a node says its value is: read by the teacher and the answer check, never by a function"""
    return number(text.strip()) if number(text.strip()) is not None else value_of(text)


def phrase_of(text):
    m = re.search(r"(the .+?) is ", text)
    return m.group(1) if m else None


# ====================================================================== the graph
class Graph:
    """Places, each a label, a relation and a chain of entries (A to be B appends B, A stays). Edges start as "everything
    to everything": i -> j is cut for good when a kept neighbour k of i is closer to j than i is (i reaches j through k).
    A relation R[i] claims place j when R[i] . L[j] > 1: i depends on j."""

    def __init__(self, d, beam=8, cap=1024):
        self.d, self.beam = d, beam
        self.L = np.zeros((cap, d), dtype=np.float32)
        self.R = np.zeros((cap, d), dtype=np.float32)
        self.S = np.full(cap, REACH, dtype=np.float32)    # each place's reach: a query further from its label is not it
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
            self.L, self.R = np.vstack([self.L, np.zeros_like(self.L)]), np.vstack([self.R, np.zeros_like(self.R)])
            self.S = np.concatenate([self.S, np.full(len(self.S), REACH, dtype=np.float32)])
        cands = self.search(label)
        i, self.n = self.n, self.n + 1
        self.L[i] = label
        self.chain.append([])
        self.adj.append(self.cut(cands))
        for j in self.adj[i]:                          # the way back (cutting here left new places with no way in)
            self.adj[j].append(i)
        return i

    def move(self, i, label):
        """a place's label changed: it leaves its edges and is linked again where it now is"""
        if np.linalg.norm(self.L[i] - label) < 1e-6:
            return
        for j in range(self.n):
            if i in self.adj[j]:
                self.adj[j].remove(i)
        self.L[i] = label
        self.adj[i] = self.cut([(d, j) for d, j in self.search(label) if j != i])
        for j in self.adj[i]:
            self.adj[j].append(i)


REACH = .5                      # a place's first reach: statements about one subject lie within .485 of each other, the
                                # nearest other subject at >= .365 (p5 .675) (trace of a 402-place memory, 11-03); lessons move it


# ====================================================================== the translator (encoder stage 1 + decoder)
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
        self.ruleq = nn.Sequential(nn.Linear(d, 256), nn.GELU(), nn.Linear(256, d))     # a fact -> where its rule lives
        self.rgate = nn.Sequential(nn.Linear(D, 256), nn.GELU(), nn.Linear(256, 1))     # relation: which tokens name what it depends on
        self.rhow = nn.Sequential(nn.Linear(D, 256), nn.GELU(), nn.Linear(256, 1))      # ... and whether it depends at all
        self.d = d
        self.K, self.c = 12, 256                       # B: what a statement says, c per token (the label only says where); K = a tool's output
        self.down = nn.Linear(D, self.c)
        self.tokd = nn.Linear(D, self.c, bias=False)  # the token itself rides along: a word never seen still comes back
        self.up = nn.Sequential(nn.Linear(self.c, D), nn.GELU(), nn.Linear(D, D))

    def encode(self, texts):
        dev = next(self.proj.parameters()).device
        b = self.tok(texts, return_tensors="pt", padding=True).to(dev)
        h = self.read(b)
        w = torch.sigmoid(self.gate(h)) * b["attention_mask"][..., None]
        return F.normalize(self.proj((w * h).sum(1) / w.sum(1).clamp(min=1e-6)), dim=-1)

    def read(self, b):
        """the LM's states (the encoder meets clear statements only: the normalizer reads the language)"""
        with torch.no_grad():
            return self.norm(self.lm(**b).last_hidden_state)

    def relate(self, a, b):
        return self.rel(torch.cat([(a - b).abs(), a * b], -1)).squeeze(-1)

    def rule_of(self, label):
        return F.normalize(self.ruleq(label), dim=-1)

    def relation(self, texts, scope=.9, parts=False):
        """r: what a statement says it depends on. A token gate picks the words that name it; those words are read on
        their own, so they land exactly where that thing's label is; long enough to claim the place (r . label > 1)
        when the statement depends on anything at all"""
        dev = next(self.proj.parameters()).device
        b = self.tok(texts, return_tensors="pt", padding=True, return_offsets_mapping=True).to(dev)
        off = b.pop("offset_mapping")
        h = self.read(b)
        m = b["attention_mask"]
        a = self.rgate(h).squeeze(-1)                  # per token: does it name what this depends on?
        g = self.rhow((h * m[..., None]).sum(1) / m.sum(1, keepdim=True)).squeeze(-1)   # does it depend at all?
        if parts:
            return a, g, off, m
        spans = [self.tok.decode(b["input_ids"][i][(a[i] > 0) & (m[i] > 0)]).strip() for i in range(len(texts))]
        return self.encode([x or "-" for x in spans]) * torch.sigmoid(g)[:, None] / scope

    def content(self, texts):
        """encoder, second head: B, one latent per token, so a name never seen keeps its own vectors and comes back
        exact. Left-padded, positions counted from each row's first token: B does not depend on the batch."""
        dev = next(self.proj.parameters()).device
        self.tok.padding_side = "left"
        b = self.tok([self.tok.eos_token + t for t in texts], return_tensors="pt", padding=True).to(dev)
        self.tok.padding_side = "right"
        m = b["attention_mask"]
        with torch.no_grad():
            h = self.norm(self.lm(**b, position_ids=(m.cumsum(1) - 1).clamp(min=0)).last_hidden_state)
        keep = m.bool()
        keep[torch.arange(len(texts)), (m == 0).sum(1)] = False   # the first token soaks up attention in GPT-2: a dummy
        return F.layer_norm(self.down(h) + self.tokd(self.lm.wte(b["input_ids"])), (self.c,)), keep

    def decode_loss(self, z, zm, texts, each=False, up=None):
        """decoder: the frozen LM reads B as a prefix and writes the text back (each: one loss per text; up: whose mouth,
        default the decoder's own, for stored nodes; a tool says its results with its own)"""
        t = self.tok([x + self.tok.eos_token for x in texts], return_tensors="pt", padding=True).to(z.device)
        ids, mask = t["input_ids"], t["attention_mask"]
        m = torch.cat([zm.long(), mask], 1)
        x = torch.cat([(up or self.up)(z), self.lm.wte(ids)], 1)
        h = self.lm(inputs_embeds=x, attention_mask=m, position_ids=(m.cumsum(1) - 1).clamp(min=0)).last_hidden_state
        h = h[:, z.shape[1] - 1:-1]
        nll = F.cross_entropy((h @ self.lm.wte.weight.T).transpose(1, 2), ids, reduction="none")
        return (nll * mask).sum(1) / mask.sum(1) if each else (nll * mask).sum() / mask.sum()

    @torch.no_grad()
    def decode(self, z, zm, n=48, up=None):
        """B -> text, greedy"""
        eos, m = self.tok.eos_token_id, zm.long()
        o = self.lm(inputs_embeds=(up or self.up)(z), attention_mask=m, position_ids=(m.cumsum(1) - 1).clamp(min=0), use_cache=True)
        past, h, out, at = o.past_key_values, o.last_hidden_state[:, -1], [], m.sum(1, keepdim=True)
        for _ in range(n):
            nxt = (h @ self.lm.wte.weight.T).argmax(-1)
            out.append(nxt)
            if all(eos in x for x in torch.stack(out, 1).tolist()):
                break
            m = torch.cat([m, torch.ones_like(m[:, :1])], 1)
            o = self.lm(inputs_embeds=self.lm.wte(nxt)[:, None], past_key_values=past, attention_mask=m, position_ids=at, use_cache=True)
            past, h, at = o.past_key_values, o.last_hidden_state[:, -1], at + 1
        return [self.tok.decode(x[:x.index(eos)] if eos in x else x) for x in torch.stack(out, 1).tolist()]


# ====================================================================== the function machine (user design 11-04)
# Everything is a node: L (where it lives), R (what it depends on), B (its value, latents). A function is a node too
# (its label = what wants it answers, its code = its value) with its own network: it takes any nodes and functions and
# gives nodes (a value, a change to one of its inputs, an edited input function, a new function). Signals (+ / -) fire
# functions: a signal's want meets the functions' labels. Everything is visible; the filter decides what reaches each
# recipient. Text becomes a node once, when it arrives; a node is said only to answer, or when its maker's own error
# guess says it drifted (said and read again). The walk (nearest places by label) and the write (where an output
# lands) are the only parts that are not functions.
CHAT = ["Thanks for the update!", "Morning everyone.", "Let me know if anything else changes.", "OK, noted.",
        "Have a good weekend.", "Sorry for the delay.", "See the notes below.", "That is all for today.", "Hope you are well.",
        "Quick update:", "Cheers, talk soon.", "Nothing new on my side."]
VAL = lambda v: str(v) if isinstance(v, int) else f"string {v}"      # what a value node says: typed ("1", "string hello")
D = lambda t: re.sub(r"(?<=\d)(?=\d)", " ", t)       # numbers as the module reads and says them: digit by digit
#                                                (one row per digit: a function can work along them)


def num(text):
    """the number a said node is, or None"""
    return number(text.replace(" ", "").strip())
INC, EDIT, NEWF = "add one to it", "make it add {k} instead", "make a new one that adds {k}"
RELATED = .5                                   # R . L above this: the node relates to that place (R's length = how
#                                                much it relates at all)
NEW_WORK, NEW_PLACE = 0, 1                     # where an output goes: a new node in the workspace / in the memory, or 2 + r:
#                                                in place of input r (its value changed)


def pad(zs, L=96):
    """node values (any lengths) as one batch: (n, T, c) and the mask"""
    T = min(L, max(len(z) for z in zs))
    x = zs[0].new_zeros(len(zs), T, zs[0].shape[-1])
    m = torch.zeros(len(zs), T, dtype=torch.bool, device=x.device)
    for i, z in enumerate(zs):
        z = z[:T]
        x[i, :len(z)], m[i, :len(z)] = z, True
    return x, m


class Pool(nn.Module):
    """reads one or more nodes' values (each any length) into k vectors"""

    def __init__(self, c=256, k=1, L=96, S=4, layers=2):
        super().__init__()
        self.cls = nn.Parameter(torch.randn(k, c) * .02)
        self.pos = nn.Parameter(torch.randn(L, c) * .02)
        self.side = nn.Parameter(torch.randn(S, c) * .02)    # which of the nodes read together
        self.f = nn.TransformerEncoder(nn.TransformerEncoderLayer(c, 8, 512, batch_first=True, activation="gelu"), layers)
        self.k, self.L, self.S = k, L, S

    def forward(self, rows):
        """rows: one list of node values per item; -> (n, k, c)"""
        dev = self.cls.device
        seqs = [torch.cat([z[:self.L] + self.pos[:len(z[:self.L])] + self.side[min(j, self.S - 1)] for j, z in enumerate(r)]) for r in rows]
        x, m = pad(seqs, self.L * self.S)
        n = len(rows)
        y = self.f(torch.cat([self.cls[None].expand(n, -1, -1), x], 1),
                   src_key_padding_mask=~torch.cat([torch.ones(n, self.k, dtype=torch.bool, device=dev), m], 1))
        return y[:, :self.k]


class Heads(nn.Module):
    """the encoder's second stage: a node's label and relation from its value alone, so a node a function made gets
    both without being said and read again"""

    def __init__(self, c=256, d=128):
        super().__init__()
        self.p = Pool(c, 3)
        self.lab = nn.Sequential(nn.Linear(c, 512), nn.GELU(), nn.Linear(512, d))
        self.rel = nn.Sequential(nn.Linear(c, 512), nn.GELU(), nn.Linear(512, d))
        self.how = nn.Linear(c, 1)

    def forward(self, zs):
        y = self.p([[z] for z in zs])
        return F.normalize(self.lab(y[:, 0]), dim=-1), F.normalize(self.rel(y[:, 1]), dim=-1) * torch.sigmoid(self.how(y[:, 2])) / .9


class Interpreter(nn.Module):
    """what every node runs when it acts: one network for all. It reads the acting node's program (its value: rows)
    and applies its action (a low-rank matrix pair that multiplies into and adds to the states it reads, every layer),
    the signal, and everything it perceives (each input with how salient it is and how far the traversal went to reach
    it; layer k reads what lies up to k hops away; each input's own action, from its value, acts on the states too, as
    much as it is salient: a rule in view changes how the statement is read). An output node is made of up to S
    pieces, each a span taken as it is from one of the inputs or up to G latents it makes; a function given back is K
    latents. Also: how many outputs, where each goes, whether one is a function, its own guess of how wrong it is"""

    def __init__(self, d=128, c=256, K=16, M=2, S=4, G=8, R=16, L=40, r=8, layers=3):
        super().__init__()
        self.pos = nn.Parameter(torch.randn(L, c) * .02)
        self.hop = nn.Parameter(torch.randn(4, c) * .02)       # how far the traversal went to reach an input
        self.salv = nn.Parameter(torch.randn(c) * .02)         # how salient it is (perception)
        self.me = nn.Parameter(torch.randn(c) * .02)           # marks the program
        self.wq = nn.Linear(d, c)
        self.wv = nn.Linear(1, c)                              # the signal that fired it (its value: sign x strength)
        self.codeq = nn.Parameter(torch.randn(M * K, c) * .02)   # a function given back
        self.segq = nn.Parameter(torch.randn(M * S, c) * .02)    # each piece of each output
        self.genq = nn.Parameter(torch.randn(M * S * G, c) * .02)   # the latents a made piece can hold
        self.layers = nn.ModuleList(nn.TransformerEncoderLayer(c, 8, 1024, batch_first=True, activation="gelu") for _ in range(layers))
        self.act = nn.Linear(c, 2 * c * r)                     # a node's action from its value (what any node does)
        nn.init.normal_(self.act.weight, std=1e-3), nn.init.zeros_(self.act.bias)
        self.many, self.err, self.isfn = nn.Linear(c, M), nn.Linear(c, 1), nn.Linear(c, 1)
        self.where, self.wp = nn.Linear(c, 2), nn.Linear(c, c)  # a new node / a new place, or in place of input r: the
        #                                                         output's state against each input's tokens
        self.kind = nn.Linear(c, 3)                            # a piece: 0 no more pieces, 1 a span of an input, 2 made
        self.ps, self.pe, self.pk, self.px = nn.Linear(c, c), nn.Linear(c, c), nn.Linear(c, c), nn.Linear(c, c)
        self.gkeep = nn.Linear(c, 1)                           # how many latents a made piece keeps
        self.K, self.M, self.S, self.G, self.R, self.L, self.r = K, M, S, G, R, L, r

    def action_of(self, z):
        """a node's action (U, V) from its value"""
        a = self.act(z.mean(0)).view(2, -1, self.r)
        return a[0], a[1]

    def forward(self, ins, q, program, action, v=1., hops=None, sal=None, acts=None):
        c, dev = self.pos.shape[-1], q.device
        toks = [z[:self.L] for z in ins[:self.R]]
        hops = (list(hops) if hops is not None else [0] * len(toks))[:len(toks)]
        sal = (list(sal) if sal is not None else [torch.ones((), device=dev)] * len(toks))[:len(toks)]
        head = [self.codeq, self.segq, self.genq, F.layer_norm(program, (c,)) + self.me, self.wq(q)[None],
                self.wv(torch.tensor([float(v)], device=dev))[None]]
        nh = sum(len(x) for x in head)
        parts = head + [z + self.pos[:len(z)] + self.hop[min(hp, 3)] + sv * self.salv for z, hp, sv in zip(toks, hops, sal)]
        reach = torch.tensor([-1] * nh + [hp for z, hp in zip(toks, hops) for _ in range(len(z))], device=dev)
        acts = list(acts)[:len(toks)] if acts is not None else [self.action_of(z) for z in toks]
        h = torch.cat(parts)
        total = 1 + sum(float(sv) for sv in sal)
        for li, layer in enumerate(self.layers):       # layer li reads what lies up to li hops away (and all its own)
            h = layer(h[None], src_key_padding_mask=(reach > li)[None])[0]
            d = (h @ action[0]) @ action[1].T if action is not None else 0   # the acting node's action multiplies into
            for (U, V), sv in zip(acts, sal):          # and adds to the states; so does every input's, as much as it is
                d = d + sv * ((h @ U) @ V.T) / total   # salient (averaged by salience)
            h = F.layer_norm(h + .1 * d, (c,))         # (normalized again: added unnormalized, the states grew every
            #                                            layer until the loss was 1e29 by step 32, fm12)
        a, b = self.M * self.K, self.M * self.S
        e = a + b + b * self.G
        hs = h[a:a + b].view(self.M, self.S, c)
        o = {"code": F.layer_norm(h[:a], (c,)).view(self.M, self.K, c), "hs": hs, "kind": self.kind(hs),
             "gen": F.layer_norm(h[a + b:e], (c,)).view(self.M, self.S, self.G, c),
             "gkeep": self.gkeep(h[a + b:e]).view(self.M, self.S, self.G)}
        n = sum(len(z) for z in toks)
        o["ctx"] = h[nh:nh + n]
        o["raw"] = torch.cat(toks) if toks else h.new_zeros(0, c)       # the inputs' tokens as they are
        o["owner"] = torch.cat([torch.full((len(z),), r, device=dev) for r, z in enumerate(toks)]) if toks else h.new_zeros(0)
        o["keys"] = self.pk(o["ctx"])
        o["start"] = self.ps(hs) @ o["keys"].T
        g, every = hs.mean(1), h[:e].mean(0)
        o["many"], o["err"] = self.many(every), self.err(every).squeeze(-1)
        if toks:
            each = torch.stack([o["keys"][o["owner"] == r].mean(0) for r in range(len(toks))])
            o["where"] = torch.cat([self.where(g), self.wp(g) @ each.T], -1)
        else:
            o["where"] = self.where(g)
        o["isfn"] = self.isfn(g).squeeze(-1)
        return o

    def end_logits(self, o, j, s, st):
        """where a span that starts at token st ends: at st or after, in the same input"""
        lg = (self.pe(o["hs"][j, s]) + self.px(o["ctx"][st])) @ o["keys"].T
        bad = (torch.arange(len(lg), device=lg.device) < st) | (o["owner"] != o["owner"][st])
        return lg.masked_fill(bad, -1e9)


UIDS = iter(range(10 ** 9))
CALC = {"add": lambda a, b: a + b, "sub": lambda a, b: a - b, "max": max, "min": min, "mul": lambda a, b: a * b}


class FnNode(nn.Module):
    """a node the model made to act (a function): its label (key: the wants it answers), its value (its program: K
    rows) and its own correction to the action its value gives (au, av). A calculator node also has a body of its own
    (calc: decode the numbers it reads, compute, encode the result); its learned side is its mimic, which takes over
    when the calculator is removed. It is a node like the others: every signal sees it, it is chosen like any node,
    edited like any node's value"""

    def __init__(self, K, d=128, c=256, r=8, calc=None):
        super().__init__()
        self.key = nn.Parameter(F.normalize(torch.randn(d), dim=0))
        self.register_buffer("mu", torch.tensor(2.))        # the error it usually gets (log 1 + loss), and its spread
        self.register_buffer("sd", torch.tensor(1.))
        self.code = nn.Parameter(torch.randn(K, c) * .02)
        self.au = nn.Parameter(torch.randn(c, r) * .01)
        self.av = nn.Parameter(torch.randn(c, r) * .01)
        self.calc = calc
        self.uid = next(UIDS)                          # who it is (its record of what it does well is its own)


def assemble(net, o, j, plan=None):
    """output j as a node: its pieces put together (its own choices, or in demos the pieces the target is made of:
    ("span", first token, last token) / ("made", how many latents)); returns the node's value and its pieces"""
    rows, pieces = [], []
    for s_ in range(net.S):
        if plan is not None:
            if s_ >= len(plan):
                break
            pc = plan[s_]
        else:
            kd = o["kind"][j, s_].clone()
            if not len(o["raw"]):
                kd[1] = -1e9                           # no inputs: nothing to take a span from
            k = int(kd.argmax())
            if k == 0:
                break
            if k == 1:
                st_ = int(o["start"][j, s_].argmax())
                pc = ("span", st_, int(net.end_logits(o, j, s_, st_).argmax()))
            else:
                pc = ("made", max(1, int((o["gkeep"][j, s_] > 0).sum())))
        rows.append(o["raw"][pc[1]:pc[2] + 1] if pc[0] == "span" else o["gen"][j, s_, :pc[1]])
        pieces.append(pc)
    return (torch.cat(rows) if rows else o["gen"][j, 0, :1]), pieces


def plan_pieces(tid, ins, S, G):
    """the pieces a target text (its tokens) is made of, given the inputs ((rows, tokens or None) each): the longest
    runs found as they are in an input (spans, numbered across the inputs' tokens) and what is left (made: target
    positions i..j); None when that takes more than S pieces or a made piece longer than G"""
    T = len(tid)
    best, at = [(0, 0)] * (T + 1), 0                   # best[i]: the longest run of the target from i found in an input
    for n_rows, t in ins:
        if t is not None:
            nxt = [0] * (len(t) + 1)
            for i in range(T - 1, -1, -1):
                cur = [0] * (len(t) + 1)
                for a in range(len(t) - 1, -1, -1):
                    if t[a] == tid[i]:
                        cur[a] = nxt[a + 1] + 1
                        if cur[a] > best[i][0]:
                            best[i] = (cur[a], at + a)
                nxt = cur
        at += n_rows
    pieces, i = [], 0
    while i < T:
        n, g = best[i]
        if n:
            pieces.append(("span", g, g + n - 1))
            i += n
            continue
        j = i + 1
        while j < T and best[j][0] == 0:
            j += 1
        pieces.append(("made", i, j))
        i = j
    if len(pieces) > S or any(pc[0] == "made" and pc[2] - pc[1] > G for pc in pieces):
        return None
    return pieces


def say_loss(fm, A, net, o, j, target, order):
    """how wrong output j is as the text it should say: which pieces it is made of (kind, where each span starts and
    ends, how many latents a made piece keeps and how close they are to what the encoder makes of those words), and
    what the assembled node says. Returns (loss, decode loss, the node, the target's pieces, its tokens) or None when
    the target cannot be made of the pieces an output has"""
    ins_t = [(len(n.B[:net.L]), n.toks[:net.L] if n.toks is not None else None) for n in order[:net.R]]
    best = None
    for spelled in dict.fromkeys([D(target), " " + D(target).lstrip()]):   # as written, or as it reads inside a
        tid_ = fm.tr.tok(spelled)["input_ids"]                             # sentence: the one more of it is copied in
        plan_ = plan_pieces(tid_, ins_t, net.S, net.G)
        if plan_ is not None:
            copied = sum(pc[2] - pc[1] + 1 for pc in plan_ if pc[0] == "span")
            if best is None or copied > best[0]:
                best = (copied, spelled, tid_, plan_)
    if best is None:
        fm.count("cannot be made of pieces")
        return None
    _, target, tid, plan = best
    dev = o["kind"].device
    with torch.no_grad():
        zt, mt = fm.tr.content([target])
        zt = zt[0][mt[0]]
    L = 0
    for s_ in range(min(len(plan) + 1, net.S)):
        k = 0 if s_ == len(plan) else 1 if plan[s_][0] == "span" else 2
        L = L + F.cross_entropy(o["kind"][j, s_][None], torch.tensor([k], device=dev))
        if k == 1:
            a, b = plan[s_][1], plan[s_][2]
            L = L + F.cross_entropy(o["start"][j, s_][None], torch.tensor([a], device=dev))
            L = L + F.cross_entropy(net.end_logits(o, j, s_, a)[None], torch.tensor([b], device=dev))
        elif k == 2:
            i0, i1 = plan[s_][1], plan[s_][2]
            L = L + F.binary_cross_entropy_with_logits(o["gkeep"][j, s_], (torch.arange(net.G, device=dev) < i1 - i0).float())
            if len(zt) == len(tid):                    # what the encoder makes of those words, there
                L = L + F.mse_loss(o["gen"][j, s_, :i1 - i0], zt[i0:i1]) * A.mse
    want_pcs = [(pc[0], pc[1], pc[2]) if pc[0] == "span" else ("made", pc[2] - pc[1]) for pc in plan]
    z, _ = assemble(net, o, j, want_pcs)
    dl = fm.tr.decode_loss(z[None], torch.ones(1, len(z), dtype=torch.bool, device=dev), [target])
    with torch.no_grad():                              # exact = its own pieces are the target's, and what it makes says it
        exact = assemble(net, o, j)[1] == want_pcs and float(dl) < A.exact
    return L + dl, exact, z, plan, tid, target


class Acting(nn.Module):
    """what lets any node act (user 11-04: a node has abilities like a function, its action a matrix that multiplies
    into and adds to other nodes; label, value and relation stay): the shared interpreter; each node's action A, from
    its value (plus, for a node the model made, its own correction); how much a node read from text answers wants
    (its label, scaled by what its value says); the want reader (a signal's nodes -> what is wanted); and the nodes the
    model made to act (functions are nodes: made by writing one, seen and chosen like any other; the calculator's
    nodes are there from the start)"""

    def __init__(self, d=128, c=256, calculators=tuple(CALC)):
        super().__init__()
        self.d, self.c = d, c
        self.interp = Interpreter(d, c)
        self.fns = nn.ModuleList(FnNode(self.interp.K, d, c, self.interp.r, calc=op) for op in calculators)
        self.wp = Pool(c, 1)
        self.wv = nn.Linear(1, c)
        self.wo = nn.Sequential(nn.Linear(c, 512), nn.GELU(), nn.Linear(512, d))
        self.wk = nn.Sequential(nn.Linear(c, 256), nn.GELU(), nn.Linear(256, 1))   # a read node: how much it acts
        nn.init.constant_(self.wk[-1].bias, -1.)       # (at first little: softplus(-1) = .31, below a made label)
        self.register_buffer("mu", torch.tensor(2.))     # the error read nodes usually get when they act
        self.register_buffer("sd", torch.tensor(1.))

    def want(self, rows, v=None):
        """what a signal wants: from its nodes and its value (1 when not given)"""
        y = self.wp(rows)[:, 0]
        v = torch.ones(len(rows), 1, device=y.device) if v is None else torch.tensor(v, dtype=y.dtype, device=y.device).reshape(-1, 1)
        return F.normalize(self.wo(y + self.wv(v)), dim=-1)

    def key(self, n):
        """the label a want meets: a made node's own; a read node's label, as long as its value says it acts"""
        if n.kind == "fn":
            return self.fns[n.fn].key
        return n.L * F.softplus(self.wk(n.B.mean(0))).squeeze(-1)

    def gate(self, q, nodes):                          # every want against every node's label: > 1 = relevant
        keys = [self.fns[n.fn].key if n.kind == "fn" else None for n in nodes]
        rd = [i for i, n in enumerate(nodes) if n.kind != "fn"]
        if rd:                                         # the read nodes' together (one pass)
            w = F.softplus(self.wk(torch.stack([nodes[i].B.mean(0) for i in rd]))).squeeze(-1)
            for k_, i in enumerate(rd):
                keys[i] = nodes[i].L * w[k_]
        return q @ torch.stack(keys).T

    def program(self, n):
        """what the interpreter reads as the acting node's program: its value"""
        return self.fns[n.fn].code if n.kind == "fn" else n.B[:self.interp.L]

    def action(self, n, program=None):
        """the node's action A (U, V): from its value; a made node adds its own"""
        if n.kind == "fn":
            f = self.fns[n.fn]
            U, V = self.interp.action_of(f.code if program is None else program)
            return U + f.au, V + f.av
        return self.interp.action_of(n.B if program is None else program)

    def stats_of(self, n):                             # whose usual error
        return self.fns[n.fn] if n.kind == "fn" else self

    def calc_of(self, n):
        return self.fns[n.fn].calc if n.kind == "fn" else None

    def run(self, actor, ins, q, v=1., hops=None, sal=None, program=None):
        """the actor acts: the interpreter reads its program and applies its action, and every input's own action"""
        return self.interp([n.B for n in ins], q, self.program(actor) if program is None else program,
                           self.action(actor, program), v, hops, sal, acts=[self.action(n) for n in ins[:self.interp.R]])

    def make(self, key, like=None, code=None, scope=.9):
        """a new node that acts (a function): made like another acting node (its program and its own action; a read
        node: its value as the program), labelled with the want it answers"""
        f = FnNode(self.interp.K, self.d, self.c, self.interp.r).to(key.device)
        with torch.no_grad():
            if like is not None and like.kind == "fn":
                src = self.fns[like.fn]
                f.code.copy_(src.code), f.au.copy_(src.au), f.av.copy_(src.av)
            elif like is not None:
                rows = like.B.detach()
                f.code.copy_(rows[torch.arange(self.interp.K, device=rows.device) % len(rows)])
            if code is not None:
                f.code.copy_(code)
            f.key.copy_(F.normalize(key, dim=0) / scope)
        self.fns.append(f)
        return len(self.fns) - 1

    def as_node(self, t):                              # made node t's value (being read does not change it)
        return F.layer_norm(self.fns[t].code.detach(), (self.c,))


class Critic(nn.Module):
    """the check node: what a function was given (a rule among it, or not) and what it gave -> does it pass (logit)"""

    def __init__(self, c=256, R=11, L=96):
        super().__init__()
        self.cls = nn.Parameter(torch.randn(c) * .02)
        self.role = nn.Parameter(torch.randn(R, c) * .02)
        self.pos = nn.Parameter(torch.randn(L, c) * .02)
        self.f = nn.TransformerEncoder(nn.TransformerEncoderLayer(c, 8, 1024, batch_first=True, activation="gelu"), 2)
        self.o = nn.Linear(c, 1)
        self.R, self.L = R, L

    def forward(self, ins, out):
        seq = [self.cls[None]] + [z[:self.L] + self.role[r] + self.pos[:len(z[:self.L])] for r, z in enumerate(ins[:self.R - 1])]
        seq.append(out[:self.L] + self.role[-1] + self.pos[:len(out[:self.L])])
        return self.o(self.f(torch.cat(seq)[None])[0, 0]).squeeze(-1)


class Filter(nn.Module):
    """perception: how salient each thing near the signal is for the node about to act (its label, program), given
    the state (the signal's want, sign) and the thing (value, label, kind, age). Nothing is cut away by it: what it
    weighs goes in with its salience, and the salience learns from how the act turns out"""

    def __init__(self, c=256, d=128):
        super().__init__()
        self.kind = nn.Parameter(torch.randn(5, c) * .02)   # 0 the signal's node, 1 / 3 a memory entry the label /
        #                                                     relation walk brought, 2 a function, 4 two hops away
        self.age = nn.Parameter(torch.randn(8, c) * .02)    # how many entries back in its place
        self.cand = nn.Sequential(nn.Linear(c + d, c), nn.GELU(), nn.Linear(c, c))
        self.qry = nn.Sequential(nn.Linear(2 * d + 2 + c, c), nn.GELU(), nn.Linear(c, c))
        self.f = nn.TransformerEncoder(nn.TransformerEncoderLayer(c, 8, 512, batch_first=True, activation="gelu"), 2)
        self.o = nn.Linear(c, 1)

    def forward(self, key, code, q, sign, strength, cands):
        s = torch.tensor([sign, strength], dtype=q.dtype, device=q.device)
        qv = self.qry(torch.cat([key, q, s, code.mean(0)]))
        cv = torch.stack([self.cand(torch.cat([B.mean(0), L])) + self.kind[k] + self.age[min(a, 7)] for B, L, k, a in cands])
        return self.o(self.f(torch.cat([qv[None], cv])[None])[0, 1:]).squeeze(-1)


class Node:
    """a node in the pool: its value B, label L, relation R, and its action A (Acting.action: from B, plus a made
    node's own correction); where it lives (a memory entry (place, index), the workspace, or a made node that acts);
    the node that asked for it (parent). Any node can act: a signal's want meets its label (Acting.key)"""

    def __init__(self, B, L, R, kind="work", place=-1, idx=-1, parent=None, fn=-1):
        self.B, self.L, self.R, self.kind, self.place, self.idx, self.parent, self.fn = B, L, R, kind, place, idx, parent, fn
        self.text = None                               # the text it was read from (traces only)
        self.toks = None                               # the token each row of B is (known when read from text, or in demos)


class Signal:
    """what fired: the nodes it is about, its value (sign x strength: +1 an arrival or a change; a check's failure is
    negative) and its order (the newest first)"""

    def __init__(self, about, sign=1., strength=0.):
        self.about, self.sign, self.strength = about, sign, strength


class FM:
    """the module: a pool of nodes (the memory graph + the workspace + the functions) and the signals. A signal fires
    the function whose label its want meets; the filter picks its inputs among everything near; its outputs land; what
    changed raises new signals; nothing above the labels = rest. No step is scripted."""

    def __init__(self, tr, bank, heads, filt, critic, budget=60, look=6, newest=4, kappa=2., same=.1):
        self.tr, self.bank, self.heads, self.filt, self.critic = tr, bank, heads, filt, critic
        self.dev = next(heads.parameters()).device
        self.g, self.signals, self.t, self.explore = Graph(bank.d), [], 0, 0.
        self.budget, self.look, self.newest, self.kappa, self.same = budget, look, newest, kappa, same
        self.stats, self.firings, self.show, self.capture, self.said, self.trace, self.walks = {}, [], False, None, [], False, []
        self.fit = {}                                  # (function, purpose) -> how often it did that exactly (labels)
        self.calc_on, self.news = True, []             # the calculator's own body answers (False: removed, its mimic does)
        self.made, self.frozen = False, False          # a node that acts was made (the optimizer takes it in); frozen:
        #                                                no new one (checks)

    def count(self, k, n=1):
        self.stats[k] = self.stats.get(k, 0) + n

    # --- nodes
    def nodes(self, texts):
        """text -> nodes (once, when it arrives)"""
        texts = [D(t) for t in texts]
        z, m = self.tr.content(texts)
        B = [z[i][m[i]] for i in range(len(texts))]
        L, R = self.heads(B)
        out = [Node(b, l, r) for b, l, r in zip(B, L, R)]
        for n, t in zip(out, texts):
            n.text, ids = t, self.tr.tok(t)["input_ids"]
            n.toks = ids if len(ids) == len(n.B) else None
        return out

    def node(self, z, toks=None, **kw):
        L, R = self.heads([z])
        n = Node(z, L[0], R[0], **kw)
        n.toks = toks
        return n

    def entry(self, p, i=-1):
        i %= len(self.g.chain[p])
        n = Node(self.g.chain[p][i][0], torch.from_numpy(self.g.L[p]).to(self.dev), torch.from_numpy(self.g.R[p]).to(self.dev),
                 "entry", p, i)
        n.toks = self.g.chain[p][i][2] if len(self.g.chain[p][i]) > 2 else None
        return n

    def fnode(self, t):
        return Node(self.bank.as_node(t), self.bank.fns[t].key.detach(), torch.zeros(self.bank.d, device=self.dev), "fn", fn=t)

    def name(self, n):
        if n.kind == "fn":
            c = self.bank.fns[n.fn].calc
            return f"function {n.fn}" + (f" (calculator {c})" if c else "")
        return (f"entry {n.place}.{n.idx} " if n.kind == "entry" else "node ") + repr((n.text or self.say(n))[:40])

    def say(self, n):
        return self.tr.decode(n.B[None], torch.ones(1, len(n.B), dtype=torch.bool, device=self.dev))[0].strip()

    # --- what is near a signal (everything is visible; two walks order the memory: by the node's label, where it
    #     lives, and by its relation, what it relates to: the rule over it, what it links to, what it is the same as)
    def cands(self, sg):
        out, seen = [(n, 0, 0) for n in sg.about], {(n.place, n.idx) for n in sg.about if n.kind == "entry"}
        x = sg.about[0]
        self.walks = []
        if self.g.n:
            for kind, v in ((1, x.L), (3, F.normalize(x.R, dim=0))):
                if kind == 3 and float(x.R.norm()) < .3:  # it relates to nothing (tr1: a direction from a near-zero R
                    continue                              # walked to the same few places for everything)
                found = self.g.search(v.detach().float().cpu().numpy())[:self.look]
                self.walks.append((kind, found))
                for _, p in found:
                    for a in range(min(self.newest, len(self.g.chain[p]))):
                        i = len(self.g.chain[p]) - 1 - a
                        if (p, i) not in seen:
                            seen.add((p, i))
                            out.append((self.entry(p, i), kind, a))
            for _, p in (self.walks[0][1][:2] if self.walks else []):   # hop 2: what the nearest places relate to
                y = self.entry(p)
                if float(y.R.norm()) < .3:
                    continue
                for _, p2 in self.g.search(F.normalize(y.R, dim=0).detach().float().cpu().numpy())[:2]:
                    i = len(self.g.chain[p2]) - 1
                    if (p2, i) not in seen:
                        seen.add((p2, i))
                        out.append((self.entry(p2, i), 4, 0))
        out += [(self.fnode(t), 2, 0) for t in range(len(self.bank.fns))]
        return out

    def around(self, actor, C):
        """what the acting node itself relates to joins what is near the signal (the traversal from it)"""
        if actor.kind == "fn" or not self.g.n or float(actor.R.norm()) < .3:
            return C
        seen = {(n.place, n.idx) for n, _, _ in C if n.kind == "entry"}
        for _, p in self.g.search(F.normalize(actor.R, dim=0).detach().float().cpu().numpy())[:self.look]:
            for a in range(min(self.newest, len(self.g.chain[p]))):
                i = len(self.g.chain[p]) - 1 - a
                if (p, i) not in seen:
                    seen.add((p, i))
                    C.append((self.entry(p, i), 3, a))
        return C

    def perceive(self, actor, q, sg, C, keep=(), explore=0.):
        """what the acting node takes in: the signal's nodes and the most salient of the rest (up to R), each with its
        salience; returns (nodes, hops, salience, all logits)"""
        net = self.bank.interp
        lg = self.filt(self.bank.key(actor), self.bank.program(actor), q, sg.sign, 0., [(n.B, n.L, k, a) for n, k, a in C])
        score = lg + torch.randn_like(lg) * explore if explore else lg
        first = [i for i, (n, k, _) in enumerate(C) if k == 0 or i in keep]
        rest = [i for i in sorted(range(len(C)), key=lambda i: -float(score[i])) if i not in first]
        pick = sorted(first + rest[:max(0, net.R - len(first))])
        HOP = {0: 0, 1: 1, 2: 1, 3: 1, 4: 2}
        return [C[i][0] for i in pick], [HOP[C[i][1]] for i in pick], [torch.sigmoid(lg[i]) for i in pick], lg, pick

    def choose(self, q, C):
        """which node acts: the one whose label the want meets (any node near the signal, or a made one); -1 = rest"""
        s = self.bank.gate(q[None], [n for n, _, _ in C])[0]
        if float(s.max()) <= 1:
            return -1
        if self.explore:
            return int(torch.multinomial(F.softmax(s.masked_fill(s <= 1, -1e9) / self.explore, -1), 1))
        return int(s.argmax())

    # --- one firing
    def fire(self, sg):
        q = self.bank.want([[n.B for n in sg.about]], [sg.sign])[0]
        C = self.cands(sg)
        a = self.choose(q, C)
        if a < 0:
            self.count("rest")
            return None
        actor = C[a][0]
        C = self.around(actor, C)
        ins, hops, sal, lg, pick = self.perceive(actor, q, sg, C, explore=self.explore)
        net = self.bank.interp
        o = self.bank.run(actor, ins, q, v=sg.sign, hops=hops, sal=sal)
        many, err, where, isfn = o["many"], o["err"], o["where"], o["isfn"]
        to_fn = lambda j: int(where[j].argmax()) >= 2 and int(where[j].argmax()) - 2 < len(ins) and ins[int(where[j].argmax()) - 2].kind == "fn"
        made = [o["code"][j] if float(isfn[j]) > 0 or to_fn(j) else assemble(net, o, j)[0] for j in range(int(many.argmax()) + 1)]
        take = [i in pick for i in range(len(C))]
        rec = {"actor": actor, "about": [n.B for n in sg.about], "q": q, "C": [(n.B, n.L, k, a_) for n, k, a_ in C], "take": take,
               "ins": ins, "hops": hops, "sign": sg.sign}
        self.count("acts: " + ("function" if actor.kind == "fn" else "read node"))
        calc = self.bank.calc_of(actor)
        if calc is not None and self.calc_on:          # the calculator: its own body answers; its mimic is checked
            got = calculate(self, calc, ins, sal)
            mimic = self.say(Node(made[0], None, None)) if made else ""
            if got is not None:
                self.count("calculator"), self.count("mimic agrees", int(num(mimic) == got))
                z = self.nodes([str(got)])[0].B
                self.memo(sg.about[0], got)
                self.write(z, 2 + ins.index(sg.about[0]), ins, sg, False, rec, 0)
                self.firings.append(rec)
                self.count("fires")
                return rec
            self.count("calculator read no numbers")
        st_ = self.bank.stats_of(actor)
        if self.show:                                  # --inspect: every firing, said
            for kind, found in self.walks:
                print(f"    walk by {'label' if kind == 1 else 'relation'}: " + "; ".join(
                    f"{p} d {d:.2f} {self.say(self.entry(p))[:35]!r}" for d, p in found), flush=True)
            rows = sorted(range(len(C)), key=lambda i: -float(lg[i]))[:10]
            print("    candidates (salience logit): " + " | ".join(f"{float(lg[i]):+.1f} {self.name(C[i][0])[:45]}" for i in rows), flush=True)
            g = self.bank.gate(q[None], [n for n, _, _ in C])[0]
            top = sorted(range(len(g)), key=lambda i: -float(g[i]))[:3]
            print(f"  fire {len(self.firings)}: signal {sg.sign:+.2f}, acts: {self.name(actor)} (gates " + ", ".join(
                  f"{self.name(C[i][0])[:30]} {float(g[i]):.2f}" for i in top) +
                  ") | about: " + " + ".join(repr(self.say(n)[:60]) for n in sg.about) +
                  f" | takes in {len(ins)}/{len(C)}: " + "; ".join(f"{self.name(n)[:45]} ({float(sv):.2f})" for n, sv in zip(ins, sal)) +
                  f" | many {int(many.argmax()) + 1}, where " + ", ".join(str(int(where[j].argmax())) for j in range(int(many.argmax()) + 1)) +
                  f", own error {float(err):.2f} (usual {float(st_.mu):.2f}) | says: " +
                  "; ".join(repr(self.say(Node(z, None, None))[:60]) for z in made) +
                  " | pieces " + "; ".join(str(assemble(net, o, j)[1]) for j in range(len(made))), flush=True)
        self.firings.append(rec)
        self.count("fires")
        drift = float(err) > float(st_.mu + self.kappa * st_.sd)
        self.news = []
        for j, z in enumerate(made):
            if drift:                                  # its own guess: unusually wrong -> said and read again
                z = self.nodes([self.say(Node(z, None, None))])[0].B
                self.count("roundtrip")
            self.write(z, int(where[j].argmax()), ins, sg, float(isfn[j]) > 0, rec, j)
        if len(self.news) == 1:                        # a new node is heard by itself; several made at once are heard
            self.signals.append(Signal(self.news, 1., self.t))   # together, with what they came from
        elif self.news:
            self.signals.append(Signal([sg.about[0]] + self.news, 1., self.t))
        return rec

    def memo(self, about, got):
        """what the calculator found is kept as memory, where the question part's label leads (a later query finds it)"""
        n = self.nodes([f"{self.say(about)} is {got}."])[0]
        p = self.find_place(about.L)
        if p < 0:
            p = self.g.add_place(about.L.detach().float().cpu().numpy())
        self.t += 1
        self.g.chain[p].append((n.B.detach(), self.t, n.toks))

    def find_place(self, L):
        if not self.g.n:
            return -1
        d, p = self.g.search(L.detach().float().cpu().numpy())[0]
        return p if d < .1 else -1

    def write(self, z, w, ins, sg, isfn, rec, j):
        """where an output lands, and what that change signals"""
        self.t += 1
        if w >= 2 and w - 2 < len(ins):
            x = ins[w - 2]
            if x.kind == "fn":                         # an edited function: its code is now this
                with torch.no_grad():
                    self.bank.fns[x.fn].code.copy_(z)
                self.count("edits")
            elif x.kind == "entry":                    # A to be B: the place's chain takes the new value, A stays
                self.memorize(x.place, z, ins, rec, j)
            else:                                      # a workspace node's value changed: the node that asked for it hears
                n = self.node(z)
                moved = float((n.L - x.L).norm())
                x.B, x.L, x.R = n.B, n.L, n.R
                if moved < self.same:                  # no change, no signal: it is still the node it was
                    self.count("no change")
                    return
                self.signals.append(Signal([x.parent, x] if x.parent is not None else [x], 1., self.t))
            return
        if isfn and not self.frozen:                   # a new node that acts (a function), made like an input function
            like = next((n for n in ins if n.kind == "fn"), None)
            self.bank.make(self.bank.want([[z]])[0].detach(), like, code=z.detach())
            self.made = True
            self.count("new functions")
            return
        n = self.node(z, parent=sg.about[0])
        if w == NEW_PLACE:
            self.memorize(self.g.add_place(n.L.detach().float().cpu().numpy()), z, ins, rec, j)
        else:
            self.news.append(n)

    def memorize(self, p, z, ins, rec, j):
        n = self.node(z)
        if self.g.chain[p] and self.say(Node(z, None, None)) == self.say(self.entry(p)):
            self.count("no change")                    # it says what the place says now: nothing new, nothing to hear
            return                                     # (a fact's label does not move with its value: said, compared)
        self.g.chain[p].append((z.detach(), self.t, None))
        self.g.R[p] = n.R.detach().float().cpu().numpy()
        ok = float(torch.sigmoid(self.critic([x.B for x in ins], z)))     # the check: does what was kept pass?
        rec.setdefault("kept", []).append((j, ok))
        if ok < .5:                                    # a failure is a signal: its value the check's score, about the
            self.count("check fails")                  # entry and what it was made from
            self.signals.append(Signal([self.entry(p)] + [x for x in ins if x.kind != "fn"], 2 * ok - 1, self.t))
        for j in np.nonzero(self.g.R[:self.g.n] @ self.g.L[p] > RELATED)[0]:   # what relates to it hears; the function
            if j != p:                                                    # that fires reads the values: follow or not
                self.signals.append(Signal([self.entry(int(j)), self.entry(p)], 1., self.t))

    def run(self):
        n = 0
        while self.signals and n < self.budget:
            sg = max(self.signals, key=lambda s: s.strength)              # the newest first: inner parts before outer
            self.signals.remove(sg)
            self.fire(sg)
            n += 1
        if self.signals:
            self.count("budget out")
        self.signals.clear()
        return n

    @torch.no_grad()
    def arrive(self, text):
        self.signals.append(Signal(self.nodes([text]), 1., self.t))
        return self.run()

    @torch.no_grad()
    def ask(self, text):
        q = self.nodes([text])[0]
        self.signals.append(Signal([q], 1., self.t))
        n = self.run()
        return self.say(q), n


# ---------------------------------------------------------------------- the teacher: demos only (it never runs a step)
class Teacher:
    """what memory should hold (the world's truth), and which entry is which: used to write demos, and to measure"""

    def __init__(self, world):
        self.w, self.place, self.texts = world, {}, {}

    def kept(self, text, meta, side):
        """what the memory should keep for a statement: the value as said, or under an adding rule old + new"""
        if meta[0] == "fact":
            nm, k, v = meta[1:]
            d = next((x for x in side.values() if x[0] == "delta" and x[1] == nm and x[2] == k), None)
            if d is not None:
                return f"{leaf(nm, k)} is {d[4]}."
        return text


def said(obj, w):
    """a frame as statements with what each is about: ("rule", k) / ("fact", name, key, value)"""
    if "rules" in obj:
        return [(t, ("rule", k)) for k, t in obj["rules"].items()]
    if "note" in obj:
        m = re.match(r"Correction: the (.+) of ([A-Z][a-z]+) is now (\d+)\.", obj["note"])
        k = next(x for x in w.num_keys if words(x) == m.group(1))
        return [(obj["note"], ("fact", m.group(2), k, int(m.group(3))))]
    return [(f"{leaf(rec['name'], k)} is {v}.", ("fact", rec["name"], k, v)) for recs in obj.values() for rec in recs
            for k, v in rec.items() if k != "name" and k not in w.noise]


def unresolved(E):
    """the first part of an expression still to answer (innermost, leftmost), or None"""
    if E[0] == "val":
        return None
    if E[0] in ("leaf", "before", "both"):
        return E
    if E[0] == "of":                                   # the k of (something still to find)
        return unresolved(E[2]) or E
    if E[0] == "sumboth":
        return unresolved(E[1])
    return next((x for x in (unresolved(c) for c in E[2]) if x), None) or E


def show(E, top=True):
    if E[0] == "val":
        return str(E[1])
    if E[0] == "leaf":
        return leaf(E[1], E[2])
    if E[0] == "before":
        return leaf(E[1], E[2]) + " before"
    if E[0] == "both":
        return "both values of " + leaf(E[1], E[2])
    if E[0] == "of":
        return f"the {words(E[1])} of " + show(E[2], False)
    if E[0] == "sumboth":
        return "the sum of " + show(E[1], False)
    tpl = next(x for x in TOOLS if x[0] == E[1])[2]
    return tpl.format(*[f"({show(c, False)})" if c[0] == "op" else show(c, False) for c in E[2]])


def swap(E, x, y):
    if E is x:
        return y
    if E[0] == "of":
        return ("of", E[1], swap(E[2], x, y))
    if E[0] == "sumboth":
        return swap(E[1], x, y) if E[1] is x and y[0] == "op" else ("sumboth", swap(E[1], x, y))
    if E[0] == "op":
        return ("op", E[1], [swap(c, x, y) for c in E[2]])
    return E


def ask_text(E):
    return f"What was {show(E)[:-len(' before')]} before?" if E[0] == "before" else f"What is {show(E)}?"


def plan(T, E):
    """the demo steps for a question (innermost part first): (purpose, about, inputs, outputs). About / inputs name
    workspace nodes ("q", "p") or memory entries ("e", place, index); outputs = (text it should say, where, node)"""
    steps, w = [], T.w
    while True:
        x = unresolved(E)
        if x is None:
            return steps, E
        if x[0] == "of" and x[2][0] == "val":          # its inner part is answered: now a plain lookup
            y = ("leaf", x[2][1], x[1])
            E = swap(E, x, y)
            continue
        if x[0] == "op" and x is E:                    # the whole question is arithmetic on values: its operands are
            v = next(f for f in TOOLS if f[0] == x[1])[1](*[c[1] for c in x[2]])   # read out, then the operation
            steps.append(("operands", ["q"], ["q"], [(f" {x[2][0][1]}", NEW_WORK, "a"), (f" {x[2][1][1]}", NEW_WORK, "b")]))
            steps.append((x[1], ["q", "a", "b"], ["q", "a", "b"], [(VAL(v), 2, "q")]))
            return steps, ("val", v)
        part = show(x)
        steps.append(("split", ["q"], ["q"], [(part, NEW_WORK, "p")]))
        if x[0] == "op":
            v = next(f for f in TOOLS if f[0] == x[1])[1](*[c[1] for c in x[2]])
            steps.append(("operands", ["p"], ["p"], [(f" {x[2][0][1]}", NEW_WORK, "a"), (f" {x[2][1][1]}", NEW_WORK, "b")]))
            steps.append((x[1], ["p", "a", "b"], ["p", "a", "b"], [(VAL(v), 2, "p")]))
            y = ("val", v)
        else:
            p = T.place.get(("fact", x[1], x[2]))
            if p is None:
                return None, None
            texts = T.texts[p]
            if x[0] == "leaf":
                v, n_in = w.ents[x[1]][x[2]], 1
            elif x[0] == "before":
                h = w.hist[(x[1], x[2])]
                v = next((u for u in reversed(h) if u != h[-1]), None)
                n_in = min(4, len(texts))
                if v is None or not any(value_in(t) == v for t in texts[-n_in:]):
                    return None, None
            else:
                vs = w.multi.get((x[1], x[2]), [])
                n_in = min(4, len(texts))
                if len(vs) != 2 or {value_in(t) for t in texts[-n_in:]} != set(vs):
                    return None, None
                v = f"{vs[0]} and {vs[1]}"
            ins = ["p"] + [("e", p, len(texts) - 1 - a) for a in range(n_in)]
            steps.append(({"leaf": "lookup"}.get(x[0], x[0]), ["p"], ins, [(VAL(v) if x[0] != "both" else v, 2, "p")]))
            y = ("op", "add", [("val", vs[0]), ("val", vs[1])]) if x[0] == "both" else ("val", v)
        E2 = swap(E, x, y)
        if E2[0] == "val":
            steps.append(("fill", ["q", "p"], ["q", "p"], [(VAL(E2[1]), 2, "q")]))
            return steps, E2
        steps.append(("fill", ["q", "p"], ["q", "p"], [(ask_text(E2), 2, "q")]))
        E = E2


def q_expr(kind, w, rng, ops=(1, 3)):
    """a question as an expression, or None when the world has none such yet"""
    num = sorted(f for f in w.hist if f[1] in w.num_keys and f[0] in w.ents and f[1] in w.ents[f[0]])
    if not num:
        return None
    if kind == "lookup":
        return ("leaf",) + rng.choice(num)
    if kind == "before":
        c = [f for f in num if len(set(w.hist[f])) > 1]
        return ("before",) + rng.choice(c) if c else None
    if kind == "coexist":
        c = sorted(f for f, v in w.multi.items() if len(v) == 2)
        return ("sumboth", ("both",) + rng.choice(c)) if c else None
    if kind == "chain":
        c = [(nm, k, k2) for nm, e in sorted(w.ents.items()) for k in sorted(e) if k in w.ent_keys
             for k2 in sorted(w.ents.get(e[k], {})) if k2 in w.num_keys]
        if not c:
            return None
        nm, k, k2 = rng.choice(c)
        return ("of", k2, ("leaf", nm, k))
    t = rand_tree(rng, num, rng.randint(*ops), "any")
    conv = lambda x: ("leaf", x[1], x[2]) if x[0] == "leaf" else ("op", x[1], [conv(c) for c in x[2]])
    return conv(t)


def truth_of(E, w):
    if E[0] == "leaf":
        return w.ents[E[1]][E[2]]
    if E[0] == "before":
        h = w.hist[(E[1], E[2])]
        return next(u for u in reversed(h) if u != h[-1])
    if E[0] == "sumboth":
        return sum(w.multi[(E[1][1], E[1][2])])
    if E[0] == "of":
        return w.ents[truth_of(E[2], w)][E[1]]
    return next(f for f in TOOLS if f[0] == E[1])[1](*[truth_of(c, w) for c in E[2]])


# ---------------------------------------------------------------------- learning: demos, then signals
class Lesson:
    """the losses of one batch, and the routing record (each want with its purpose; the light supervision)"""

    def __init__(self):
        self.parts, self.wants, self.kinds, self.none = [], [], [], []

    def add(self, name, x):
        self.parts.append((name, x))

    def total(self):
        return sum(x for _, x in self.parts)


def number_of(fm, n):
    """the number a node says (decoded once, then kept on the node), or None"""
    if not hasattr(n, "_num"):
        n._num = num(n.text) if n.text is not None else None if n.kind == "fn" else num(fm.say(n))
    return n._num


def calculate(fm, op, ins, sal):
    """the calculator's own body: the first two numbers among what it took in (as much as it attends to them),
    decoded; computed; None when it read no two numbers"""
    nums = []
    for n, sv in zip(ins, sal):
        x = number_of(fm, n)
        if x is not None and float(sv) > .5:
            nums.append(x)
    if len(nums) < 2:
        nums = [x for x in (number_of(fm, n) for n in ins) if x is not None]
    if len(nums) < 2:
        return None
    try:
        return CALC[op](nums[0], nums[1])
    except Exception:
        return None


def run_step(fm, Ls, A, st, purpose, about, ins, outs, also=(), v=1.):
    """one demo step done by the model: its own want; the node that acts chosen by it among every node near the signal
    and the made ones (none meets the want: the nearest, or a new node made like it when the nearest does another
    purpose); the candidates as the module itself lists them, salience scored against the teacher's inputs; the actor
    gets the teacher's inputs and whatever else is most salient (in the order the candidates list them, as when it
    lives); its outputs judged by what they say. outs: (what it should say, or a function it should give back; where:
    NEW_WORK / NEW_PLACE / the input node it replaces); also: places the signal's node relates to beyond its inputs
    (what a copy is the same as). Returns the outputs ((value, tokens): the next step reads them as they are) and the
    inputs in their order."""
    bank, net = fm.bank, fm.bank.interp
    q = bank.want([[n.B for n in about]], [v])[0]
    Ls.wants.append(q)
    Ls.kinds.append(purpose)
    fm.count("steps run")
    C = fm.cands(Signal(about, v))
    key = lambda n: ("e", n.place, n.idx) if n.kind == "entry" else ("f", n.fn) if n.kind == "fn" else id(n)
    want_in = {key(n) for n in ins}
    have = {key(n) for n, _, _ in C}
    for n in ins:                                      # how many of the inputs the walk brings, by step kind
        fm.count(f"{purpose} in", 1), fm.count(f"{purpose} walked", int(key(n) in have))
    for n in ins:                                      # an input the walk did not bring: given (and counted)
        if key(n) not in have:
            C.append((n, 1 if n.kind == "entry" else 2 if n.kind == "fn" else 0, 0))
            have.add(key(n))
    # which node acts: any node near the signal or a made one, the one whose label the want meets. None meets it:
    # the nearest acts (and its label is pulled to this want), unless it does another purpose well and not this one
    # (or is a calculator): then a new node that acts is made like it, for this want
    s_all = bank.gate(q[None], [n for n, _, _ in C])[0]
    a = int(s_all.argmax())
    actor, how = C[a][0], "routed" if float(s_all[a]) > 1 else "nearest"
    if how == "nearest" and not fm.frozen and specialized(fm, actor, purpose):
        t = bank.make(q.detach(), like=actor, scope=A.scope)
        fm.made, how = True, "new"
        print(f"step {st['step']}: no label meets a {purpose} want; the nearest ({fm.name(actor)}) does another purpose "
              f"-> made function {t}", flush=True)
        actor = fm.fnode(t)
        C.append((actor, 2, 0))
    rd = [i for i, (n, _, _) in enumerate(C) if n.kind != "fn"]
    if how != "new" and actor.kind == "fn" and rd and not fm.frozen and random.random() < A.try_read:
        a = max(rd, key=lambda i: float(s_all[i]))     # now and then the read node nearest the want tries (a rule
        actor, how = C[a][0], "tried"                  # read as a program): its label learns from how it turns out
    fm.count(f"acts by {how}"), fm.count("acts: " + ("function" if actor.kind == "fn" else "read node"))
    C = fm.around(actor, C)
    keep = [i for i, (n, _, _) in enumerate(C) if key(n) in want_in]
    order, hops, sal, lg, pick = fm.perceive(actor, q, Signal(about, v), C, keep)
    y = torch.tensor([float(key(n) in want_in) for n, _, _ in C], device=lg.device)
    wanted = y > 0                                     # a light hint: the teacher's inputs salient, the rest less so
    Ls.add("salience hint", A.sal_sup * sum(F.binary_cross_entropy_with_logits(lg[m_], y[m_]) for m_ in (wanted, ~wanted) if bool(m_.any())))
    rel, rl = sorted({n.place for n in ins if n.kind == "entry"} | set(also)), None
    places = sorted({n.place for n, k, _ in C if n.kind == "entry"} | set(rel))
    if about[0].R.requires_grad and rel and len(places) > len(rel):
        lr = F.normalize(about[0].R, dim=0) @ torch.from_numpy(fm.g.L[places]).to(about[0].R.device).T / .1
        pos = torch.tensor([p in rel for p in places], device=lr.device)
        rl = -(lr.log_softmax(0)[pos]).mean()
        Ls.add("relation", rl)
    o = bank.run(actor, order, q, v=v, hops=hops, sal=sal)
    calc = bank.calc_of(actor)
    dev = o["many"].device
    L = F.cross_entropy(o["many"][None], torch.tensor([len(outs) - 1], device=dev))
    res, keys, info, exact = [], [key(n) for n in order], [], True
    for j, (target, dest) in enumerate(outs):
        w = dest if isinstance(dest, int) else 2 + keys.index(key(dest))
        if isinstance(target, str):
            fm.said.append(target)                     # what functions are asked to say: the round trip covers it too
            got = say_loss(fm, A, net, o, j, target, order)
            if got is None:
                res.append(None)
                exact = False
                continue
            Lj, ok, z, plan, tid, spelled = got
            L, exact, dl = L + Lj, exact and ok, float(Lj)
            fm.count(f"{purpose} exact", int(ok)), fm.count(f"{purpose} said", 1)
            if calc is not None and fm.calc_on:        # the calculator's own body: what goes on is its result when it
                with torch.no_grad():                  # is the right one; its mimic (this output) learns from it
                    got_c = calculate(fm, calc, order, sal)
                fm.count("calculator right", int(got_c is not None and D(str(got_c)) == D(target))), fm.count("calculator", 1)
                if got_c is not None and D(str(got_c)) == D(target):
                    cz = fm.nodes([spelled])[0]
                    z, tid, ok = cz.B, cz.toks, True
            with torch.no_grad():
                lab, rel = fm.tr.encode([target])[0], fm.tr.relation([target])[0]
            L_, R_ = fm.heads([z])
            Ls.add("label of what it made", (1 - (L_[0] * lab).sum()) * .5 + F.mse_loss(R_[0], rel) * 5)
            if fm.capture is not None or fm.trace:
                with torch.no_grad():
                    own, pcs = assemble(net, o, j)
                    told_ = (fm.say(Node(z, None, None)), fm.say(Node(own, None, None)))
                if fm.capture is not None:             # --codec_check: what it says, with the target's pieces and its own
                    fm.capture.append((purpose, D(target), told_[0], told_[1], dl, len(pcs), len(plan), fm.name(actor)[:30]))
                if fm.trace:
                    info.append((j, target, w, dl, told_, plan, pcs, tid))
            res.append((z, tid, ok, spelled))
        else:                                          # a function given back: judged by what it then does
            f, rins, rq, rtarget = target
            got = say_loss(fm, A, net, bank.run(fm.fnode(f), rins, rq, program=o["code"][j]), 0, rtarget, rins)
            if got is not None:
                L, exact = L + got[0], exact and got[1]
            res.append((o["code"][j], None, True, None))
        L = L + F.cross_entropy(o["where"][j][None], torch.tensor([w], device=dev))
        L = L + F.binary_cross_entropy_with_logits(o["isfn"][j], torch.tensor(float(not isinstance(target, str)), device=dev))
    Ls.add("functions", L + F.mse_loss(o["err"], torch.log1p(L.detach())))   # its own guess of how wrong, on a log scale
    #                                                  (fm12: the raw loss squared swamped everything, 8233 of 8289)
    S, m = bank.gate(q[None], [actor])[0, 0], .1       # the label: pushed off a purpose it does not do exactly while it
    fk = (who(fm, actor), purpose)                     # does another one well; else pulled
    fm.fit[fk] = .9 * fm.fit.get(fk, 0.) + .1 * float(exact)
    push = not exact and (how == "tried" or specialized(fm, actor, purpose))
    Ls.add("labels", F.relu(S - (1 - m)) if push else F.relu(1 + m - S))
    if not exact and fm.calc_on and outs and isinstance(outs[0][0], str) and num(D(outs[0][0])) is not None:
        for c_i, c_net in enumerate(bank.fns):         # trying the tools it has: a calculator that gets this right
            if c_net.calc is None or (actor.kind == "fn" and c_i == actor.fn):   # is where this want's label should lead
                continue
            with torch.no_grad():
                got_c = calculate(fm, c_net.calc, order, sal)
            if got_c is not None and D(str(got_c)) == D(outs[0][0]):
                Ls.add("labels", F.relu(1 + m - bank.gate(q[None], [fm.fnode(c_i)])[0, 0]) + F.relu(S - (1 - m)))   # its label here,
                fm.count("calculator found by trying")                                             # the one that missed off
                break
    if fm.trace:
        trace_step(fm, purpose, v, about, actor, how, C, s_all, lg, y, order, o, info, outs, net, float(S), push, exact, float(L), rl)
    with torch.no_grad():
        st_ = bank.stats_of(actor)
        st_.mu.mul_(.98).add_(.02 * torch.log1p(L.detach()))
        st_.sd.mul_(.98).add_(.02 * (torch.log1p(L.detach()) - st_.mu).abs())
    return res, order


def who(fm, n):
    """whose record of what it does well: a made node's own; read nodes acting share one"""
    return ("fn", fm.bank.fns[n.fn].uid) if n.kind == "fn" else ("read",)


def specialized(fm, n, purpose):
    """it does another purpose well and this one not (a calculator: its operation only)"""
    if fm.bank.calc_of(n) is not None:
        return True
    k = who(fm, n)
    mine = fm.fit.get((k, purpose), 0.)
    others = [x for (i, p_), x in fm.fit.items() if i == k and p_ != purpose]
    return bool(others) and max(others) > mine + .3


def told(fm, n, k=70):
    """what a node says (the text it was read from, when it was)"""
    return (n.text if n.text is not None else fm.say(n))[:k]


@torch.no_grad()
def pieces_said(fm, order, net, o, j, pieces, tid=None):
    """the pieces of an output, said: a span as the words it takes (when the input's tokens are known), a made piece
    as the target's words it should make (in demos) or its size"""
    lens = [len(n.B[:net.L]) for n in order[:net.R]]
    starts = np.cumsum([0] + lens)
    out = []
    for pc in pieces:
        if pc[0] == "span":
            r = int(np.searchsorted(starts, pc[1], side="right") - 1)
            t = order[r].toks
            a, b = pc[1] - starts[r], pc[2] - starts[r]
            out.append(f"span {r}:{a}-{b} " + (repr(fm.tr.tok.decode(t[a:b + 1])) if t is not None else ""))
        else:
            out.append(f"made {pc[2] - pc[1]} " + repr(fm.tr.tok.decode(tid[pc[1]:pc[2]])) if len(pc) == 3 and tid is not None
                       else f"made {pc[1]}")
    return " + ".join(out)


@torch.no_grad()
def trace_step(fm, purpose, v, about, actor, how, C, gates, lg, y, order, o, info, outs, net, S, push, exact, L, rl):
    top = sorted(range(len(gates)), key=lambda i: -float(gates[i]))[:5]
    print(f"\n== {purpose} (signal {v:+.1f}) | about: " + " + ".join(repr(told(fm, n)) for n in about), flush=True)
    print(f"   acts: {fm.name(actor)} ({how}); gates " + ", ".join(f"{fm.name(C[i][0])[:30]} {float(gates[i]):.2f}" for i in top), flush=True)
    for kind, found in fm.walks:
        print(f"   walk by {'label' if kind == 1 else 'relation'}: " + "; ".join(
            f"place {p} d {d:.2f} {fm.say(fm.entry(p))[:40]!r}" for d, p in found), flush=True)
    name = {0: "signal's", 1: "label walk", 2: "function", 3: "relation walk", 4: "two hops"}
    rows = sorted(range(len(C)), key=lambda i: -float(lg[i]))
    print(f"   candidates {len(C)} (filter logit, wanted): " + " | ".join(
        f"{float(lg[i]):+.1f}{'*' if y[i] > 0 else ''} {name[C[i][1]]} " +
        (f"fn {C[i][0].fn}" if C[i][0].kind == "fn" else f"{told(fm, C[i][0], 45)!r}" + (f" @{C[i][0].place}.{C[i][0].idx}" if C[i][0].kind == "entry" else ""))
        for i in rows[:14]), flush=True)
    print("   inputs in order: " + " | ".join(f"{r}: {told(fm, n, 50)!r}" for r, n in enumerate(order)), flush=True)
    print(f"   many {int(o['many'].argmax()) + 1} (target {len(outs)}), own error {float(o['err']):.2f} (usual {float(fm.bank.stats_of(actor).mu):.2f})", flush=True)
    for j, target, w, dl, told_, plan, pcs, tid in info:
        wp = o["where"][j].softmax(-1)
        print(f"   out {j}: target {target!r}\n      target's pieces: {pieces_said(fm, order, net, o, j, plan, tid)}\n"
              f"      its own pieces:  {pieces_said(fm, order, net, o, j, pcs)}\n"
              f"      said with the target's pieces {told_[0]!r} | with its own {told_[1]!r}\n"
              f"      where {int(wp.argmax())} ({float(wp.max()):.2f}) target {w} | function? {float(torch.sigmoid(o['isfn'][j])):.2f} | "
              f"loss of this output {dl:.2f}", flush=True)
    print(f"   label gate {S:.2f}, step loss {L:.2f}, exact {exact} -> {'push' if push else 'pull'}" +
          (f" | relation loss {float(rl):.2f}" if rl is not None else ""), flush=True)


def demo_question(fm, T, Ls, A, st, E):
    """a question done step by step by the model (each step reads the previous steps' outputs as they are)"""
    steps, last = plan(T, E)
    if not steps:
        return False
    ws = {"q": fm.nodes([ask_text(E)])[0]}
    for purpose, about, ins, outs in steps:
        pick = lambda r: fm.entry(r[1], r[2]) if isinstance(r, tuple) else ws[r]
        res, _ = run_step(fm, Ls, A, st, purpose, [pick(r) for r in about], [pick(r) for r in ins],
                          [(text, w if w == NEW_WORK else ws[name]) for text, w, name in outs])
        if res is None:
            return False
        if any(x is None for x in res):
            return False
        for (text, w, name), (z, tid, ok, spelled) in zip(outs, res):
            parent = ws["q"] if w == NEW_WORK else ws[name].parent
            if ok:                                     # exact: its own node goes on, as it is
                ws[name] = fm.node(z, tid, parent=parent)
            else:                                      # not yet: what it should have said goes on (its tokens are then true)
                ws[name] = fm.nodes([spelled])[0]
                ws[name].parent = parent
    Ls.none.append([ws["q"].B])                       # answered: its value asks for nothing more
    return True


def demo_statement(fm, T, Ls, A, st, text, meta, side):
    """a statement: the model stores it (in place of the entry it is about, or as a new place), with the rule over it
    among its inputs; the check learns pass / fail on what was kept against what was not; the teacher writes memory"""
    s = fm.nodes([text])[0]
    k = ("rule", meta[1]) if meta[0] == "rule" else ("fact", meta[1], meta[2])
    p = T.place.get(k)
    ins = [s] + ([fm.entry(p)] if p is not None else [])
    r = T.place.get(("rule", meta[2])) if meta[0] == "fact" else None
    if r is not None:
        ins.append(fm.entry(r))
    kept = T.kept(text, meta, side)
    src = T.w.deps.get((meta[1], meta[2])) if meta[0] == "fact" and "(same as" in text else None
    also = [T.place[("fact",) + src]] if src and ("fact",) + src in T.place else []
    res, order = run_step(fm, Ls, A, st, "store", [s], ins, [(kept, ins[1] if p is not None else NEW_PLACE)], also)
    kn = fm.nodes([kept])[0]
    z = kn.B.detach()                                  # the teacher's memory: what it should keep ...
    own, own_t = (res[0][0].detach(), res[0][1]) if res is not None and res[0] is not None and res[0][2] and st["step"] > A.own else (z, kn.toks)
    #                                                    ... as the store itself writes it, once it can
    if res is not None:
        alt = {text}                                   # what it might have kept instead: as said, or old + new
        if meta[0] == "fact" and p is not None and isinstance(meta[3], int) and isinstance(value_in(T.texts[p][-1]), int):
            alt.add(f"{leaf(meta[1], meta[2])} is {value_in(T.texts[p][-1]) + meta[3]}.")
        alt.discard(kept)
        for c, y in [(z, 1.)] + [(fm.nodes([a])[0].B, 0.) for a in alt]:
            Ls.add("check", F.binary_cross_entropy_with_logits(fm.critic([n.B for n in order], c), torch.tensor(y, device=z.device)))
        if p is not None and alt and random.random() < A.fix:
            fm.t += 1
            wrong = fm.nodes([sorted(alt)[0]])[0]
            fm.g.chain[p].append((wrong.B.detach(), fm.t, wrong.toks))   # kept wrong ...
            bad = fm.entry(p)
            about = [bad] + [x for x in order if x.kind != "fn"]
            run_step(fm, Ls, A, st, "fix", about, about, [(kept, bad)], v=-1.)     # ... the check's failure: fixed
            fm.g.chain[p].pop()
    if p is None:
        with torch.no_grad():
            p = fm.g.add_place(fm.heads([z])[0][0].float().cpu().numpy())
        T.place[k], T.texts[p] = p, []
    fm.t += 1
    fm.g.chain[p].append((own, fm.t, own_t))
    with torch.no_grad():
        fm.g.R[p] = fm.heads([own])[1][0].float().cpu().numpy()
    T.texts[p].append(kept)
    if meta[0] == "fact":                              # what depends on it follows: the copy function
        todo = [(meta[1], meta[2])]
        while todo:
            src = todo.pop()
            for (dn, dk), s2 in sorted(T.w.deps.items()):
                q = T.place.get(("fact", dn, dk))
                if s2 != src or q is None:
                    continue
                new = f"{leaf(dn, dk)} is {T.w.ents[dn][dk]} (same as {leaf(*src)})."
                if new == T.texts[q][-1]:
                    continue
                d, e = fm.entry(q), fm.entry(T.place[("fact",) + src])
                res, order = run_step(fm, Ls, A, st, "copy", [d, e], [d, e], [(new, d)])
                nn_ = fm.nodes([new])[0]
                zn, zt = nn_.B.detach(), nn_.toks
                for c, y in ((zn, 1.), (d.B, 0.)) if order else ():
                    Ls.add("check", F.binary_cross_entropy_with_logits(fm.critic([n.B for n in order], c), torch.tensor(y, device=zn.device)))
                if res is not None and res[0] is not None and res[0][2] and st["step"] > A.own:
                    zn, zt = res[0][0].detach(), res[0][1]
                fm.t += 1
                fm.g.chain[q].append((zn, fm.t, zt))
                with torch.no_grad():
                    fm.g.R[q] = fm.heads([zn])[1][0].float().cpu().numpy()
                T.texts[q].append(new)
                todo.append((dn, dk))


def demo_functions(fm, Ls, A, st, rng):
    """a node's value changed in place (add one), the add-one function edited (add k), a new function made from it"""
    v = amount(rng)
    ins = fm.nodes([INC, str(v)])
    run_step(fm, Ls, A, st, "inc", ins, ins, [(str(v + 1), ins[1])])
    f = inc_fn(fm)                                     # the made node an add-one want reaches most
    if f is None:
        return
    k, x = rng.choice([2, 3, 4, 5]), amount(rng)
    probe = fm.nodes([INC, str(x)])
    rq = fm.bank.want([[n.B for n in probe]])[0].detach()
    for text, edit in ((EDIT.format(k=k), True), (NEWF.format(k=k), False)):
        ins = [fm.nodes([text])[0], fm.fnode(f)]      # the instruction arrives; the function is among what is visible
        run_step(fm, Ls, A, st, "fedit" if edit else "fnew", ins[:1], ins, [((f, probe, rq, str(x + k)), ins[1] if edit else NEW_WORK)])


@torch.no_grad()
def inc_fn(fm):
    """the made node (not a calculator) an add-one want meets most, or None"""
    made = [t for t, f in enumerate(fm.bank.fns) if f.calc is None]
    if not made:
        return None
    s = fm.bank.gate(fm.bank.want([[n.B for n in fm.nodes([INC, "1"])]]), [fm.fnode(t) for t in made])[0]
    return made[int(s.argmax())]


def apart(q, kinds, tau=.1, aq=None, akinds=()):
    """wants of one purpose together, of others apart (the light supervision); 'none' is nobody's positive. aq: recent
    wants (detached) of akinds, compared against too"""
    allq = torch.cat([q, aq.detach()]) if aq is not None and len(aq) else q
    allk = list(kinds) + (list(akinds) if aq is not None and len(aq) else [])
    ids = {k: i for i, k in enumerate(dict.fromkeys(allk))}
    lab = torch.tensor([ids[k] for k in allk], device=q.device)
    n = len(kinds)
    eye = torch.eye(n, len(allk), dtype=torch.bool, device=q.device)
    pos = (lab[:n, None] == lab[None]) & ~eye & torch.tensor([k != "none" for k in kinds], device=q.device)[:, None]
    lp = (q @ allq.T / tau).masked_fill(eye, -1e9).log_softmax(1)
    has = pos.any(1)
    return -((lp * pos).sum(1)[has] / pos.sum(1)[has]).mean() if has.any() else q.sum() * 0


def finish(fm, Ls, rng, recent=None):
    """the routing losses of a batch: wants of one purpose together; what asks nothing (talk, answered values) below
    every label. recent: [(want detached, purpose)] of the last updates, compared against (and extended)"""
    talk = fm.nodes(rng.sample(CHAT, 3) + [VAL(amount(rng)), VAL(_name(rng, 2))])
    rows = Ls.none + [[n.B] for n in talk]
    qn = fm.bank.want(rows)
    q = torch.stack(Ls.wants) if Ls.wants else qn[:0]
    kinds = Ls.kinds + ["none"] * len(qn)
    aq = torch.stack([x for x, _ in recent]) if recent else None
    Ls.add("apart", apart(torch.cat([q, qn]), kinds, aq=aq, akinds=[k for _, k in recent or []]))
    if recent is not None:
        recent.extend(zip(torch.cat([q, qn]).detach(), kinds))
        del recent[:-256]
    asks = F.relu(fm.bank.gate(qn, [fm.fnode(t) for t in range(len(fm.bank.fns))]) - .9).sum() if len(fm.bank.fns) else 0
    asks = asks + F.relu(torch.diagonal(fm.bank.gate(qn[-len(talk):], talk)) - .9).sum()   # talk does not act on itself
    Ls.add("asks nothing", asks / max(1, len(rows)))


def train_codec(fm, A, rng, opt):
    """the encoder's second stage learns from the first: label and relation read from B land where the text's do"""
    for step in range(1, A.codec + 1):
        w = new_world(rng)
        texts = []
        for _ in range(6):
            for d in world_step(w, rng):
                o = d()
                texts += [t for t, _ in said(o, w)] if o else []
        texts = rng.sample(texts, min(24, len(texts))) + rng.sample(CHAT, 2)
        texts += [ask_text(e) for e in (q_expr(k, w, rng) for k in ("lookup", "before", "chain", "ops")) if e]
        texts = [D(t) for t in texts]
        with torch.no_grad():
            lab, rel = fm.tr.encode(texts), fm.tr.relation(texts)
        z, m = fm.tr.content(texts)
        L, R = fm.heads([z[i][m[i]] for i in range(len(texts))])
        loss = (1 - (L * lab).sum(-1)).mean() + F.mse_loss(R, rel) * 10
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 100 == 0:
            print(f"codec {step}: label cos {float((L * lab).sum(-1).mean()):.3f} relation err {float(F.mse_loss(R, rel)):.4f}", flush=True)


@torch.no_grad()
def evaluate(fm, A, seed):
    """a fresh world: the module writes the memory itself from what arrives and answers; nothing is scripted. The
    teacher only measures"""
    rng = random.Random(seed)
    w = new_world(rng)
    fm.g, fm.stats, fm.firings = Graph(fm.bank.d), {}, []
    nstat = 0
    for _ in range(A.ticks):
        for d in world_step(w, rng):
            o = d()
            for text, _ in said(o, w) if o else []:
                fm.arrive(text)
                nstat += 1
    mem = dict(fm.stats)
    line = []
    for kind, ops in (("lookup", 0), ("before", 0), ("coexist", 0), ("chain", 0), ("ops", (1, 3)), ("long", (4, 6))):
        hit = n = steps = 0
        for _ in range(A.q):
            E = q_expr("ops" if kind == "long" else kind, w, rng, ops or (1, 3))
            if E is None:
                continue
            n += 1
            got, k = fm.ask(ask_text(E))
            steps += k
            hit += got == D(VAL(truth_of(E, w)))
        line.append(f"{kind} {hit}/{n} ({steps / max(1, n):.1f} firings)")
    stats = ", ".join(f"{k} {v}" for k, v in sorted(mem.items()))
    return f"statements {nstat} [{stats}] | answers: " + " | ".join(line) + f" | made nodes that act {len(fm.bank.fns)}"


@torch.no_grad()
def fn_probe(fm, rng, n=32):
    """through the module: add one in place; the add-one function edited (k 2-5 read, k 6 never); a new function"""
    f = inc_fn(fm)
    hit = 0
    for _ in range(n):
        v = amount(rng)
        x = fm.nodes([INC, str(v)])
        fm.signals.append(Signal(x, 1., fm.t))
        fm.run()
        hit += fm.say(x[1]) == D(str(v + 1))
    line = [f"add one in place {hit}/{n}"]
    if f is None:
        return " | ".join(line) + " | no made node to edit"
    keep = fm.bank.fns[f].code.detach().clone()
    for k in (3, 6):
        ok = 0
        for _ in range(n // 4):
            fm.signals.append(Signal(fm.nodes([EDIT.format(k=k)]), 1., fm.t))
            fm.run()
            x = amount(rng)
            probe = fm.nodes([INC, str(x)])
            r_ = fm.bank.run(fm.fnode(f), probe, fm.bank.want([[m.B for m in probe]])[0])
            ok += fm.say(Node(assemble(fm.bank.interp, r_, 0)[0], None, None)) == D(str(x + k))
            fm.bank.fns[f].code.copy_(keep)
        line.append(f"edited to add {k}: {ok}/{n // 4}")
    return " | ".join(line) + f" | {dict(fm.stats)}"


def amount(rng):
    """a number as the machine meets it: stored values are 1-60, but results go on, so differences go below zero and
    sums past 60"""
    r = rng.random()
    return rng.randint(1, 60) if r < .6 else rng.randint(-60, 0) if r < .8 else rng.randint(61, 200)


def snapshot(bank):
    return len(bank.fns), [t.code.detach().clone() for t in bank.fns]


@torch.no_grad()
def restore(bank, snap):
    """what living changed in the functions (edits, new ones) undone: an eval or a sampled try leaves no trace"""
    n, codes = snap
    del bank.fns[n:]
    for t, c in zip(bank.fns, codes):
        t.code.copy_(c)


def clone_graph(g):
    h = copy.copy(g)
    h.L, h.R, h.S = g.L.copy(), g.R.copy(), g.S.copy()
    h.adj, h.chain = [list(a) for a in g.adj], [list(c) for c in g.chain]
    return h


def imitate(fm, recs):
    """the choices of a try that answered right (the cheapest one) are what the choices learn: its want toward the
    label of the function it fired, the filter toward what it let through"""
    loss = 0
    for r in recs:
        a = r["actor"]
        if a.kind == "fn" and a.fn >= len(fm.bank.fns):   # a node the try made, gone with the try
            continue
        q = fm.bank.want([r["about"]], [r["sign"]])[0]
        loss = loss + F.relu(1.1 - fm.bank.gate(q[None], [a])[0, 0])
        lg = fm.filt(fm.bank.key(a).detach(), fm.bank.program(a).detach(), q.detach(), r["sign"], 0., r["C"])   # (codes change in place as it lives)
        loss = loss + F.binary_cross_entropy_with_logits(lg, torch.tensor(r["take"], dtype=torch.float, device=lg.device))
    return loss / max(1, len(recs))


def checked(fm, recs):
    """the check's score on what was kept is a signal: its gradient reaches the function that kept it (the check
    itself does not move here)"""
    fm.critic.requires_grad_(False)
    loss, n = 0, 0
    for r in recs:
        a = r["actor"]
        if a.kind == "fn" and a.fn >= len(fm.bank.fns):
            continue
        for j, _ in r.get("kept", []):
            r_ = fm.bank.run(a, r["ins"], r["q"], v=r["sign"], hops=r.get("hops"))
            loss, n = loss - F.logsigmoid(fm.critic([x.B for x in r["ins"]], assemble(fm.bank.interp, r_, j)[0])), n + 1
    fm.critic.requires_grad_(True)
    return loss / max(1, n)


def live(fm, A, rng, O, params, remake):
    """the module lives in a fresh world by itself: nothing is scripted and no demo is shown. Lessons come only from
    signals: the check's score on what it kept, and for a wrong answer (the truth arrives) sampled tries, of which the
    cheapest right one is what its choices learn"""
    w = new_world(rng)
    fm.g, t0 = Graph(fm.bank.d), time.time()
    tally = {"asked": 0, "right": 0, "fixed by a try": 0}
    for tick in range(1, A.live + 1):
        fm.firings = []
        for d in world_step(w, rng):
            o = d()
            for text, _ in said(o, w) if o else []:
                fm.arrive(text)
        stored, loss = fm.firings[-16:], 0          # the check's gradient is built just before the step (below): asking
        #                                             edits functions' codes in place
        for _ in range(A.lq):
            E = q_expr(rng.choice(["lookup", "before", "coexist", "chain", "ops"]), w, rng)
            if E is None:
                continue
            truth, g0, snap = D(VAL(truth_of(E, w))), fm.g, snapshot(fm.bank)
            tally["asked"] += 1
            fm.g, fm.firings = clone_graph(g0), []
            got, _ = fm.ask(ask_text(E))
            restore(fm.bank, snap)
            fm.g = g0                                  # asking leaves the memory as it was
            if got == truth:
                tally["right"] += 1
                continue
            best, fm.explore = None, A.explore
            for _ in range(A.tries):
                fm.g, fm.firings = clone_graph(g0), []
                got, _ = fm.ask(ask_text(E))
                restore(fm.bank, snap)
                if got == truth and (best is None or len(fm.firings) < len(best)):
                    best = list(fm.firings)
            fm.explore, fm.g = 0., g0
            if best:
                tally["fixed by a try"] += 1
                loss = loss + imitate(fm, best)
        loss = loss + checked(fm, stored)
        if fm.made:                                    # nodes it made while living join the optimizer
            remake()
            fm.made = False
        if torch.is_tensor(loss) and loss.requires_grad:
            O["opt"].zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params(), 1.)
            O["opt"].step()
        if tick % 25 == 0:
            print(f"live {tick}: {tally} | {dict(fm.stats)} | {time.time() - t0:.0f}s", flush=True)
            tally, fm.stats = {"asked": 0, "right": 0, "fixed by a try": 0}, {}


@torch.no_grad()
def inspect(A):
    """--inspect --load X: live a small world, then ask one question of each kind with every firing said"""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr = Translator().to(dev).eval()
    if A.init:
        tr.load_state_dict(torch.load(A.init, map_location=dev), strict=False)
    fm = FM(tr, Acting().to(dev), Heads().to(dev), Filter().to(dev), Critic().to(dev), same=A.same)
    load_fm(fm, A.load)
    for mod in (fm.bank, fm.heads, fm.filt, fm.critic):
        mod.eval()
    rng = random.Random(A.seed + 7)
    w = new_world(rng)
    n = 0
    for _ in range(A.ticks):
        for d in world_step(w, rng):
            o = d()
            for text, _ in said(o, w) if o else []:
                fm.show = n < 6                        # the first few statements said too
                if fm.show:
                    print(f"statement: {text!r}", flush=True)
                fm.arrive(text)
                n += 1
    print(f"memory: {fm.g.n} places from {n} statements | {fm.stats}", flush=True)
    fm.show = True
    for kind in ("lookup", "before", "coexist", "chain", "ops"):
        E = q_expr(kind, w, rng)
        if E is None:
            continue
        print(f"\n{kind}: {ask_text(E)!r} truth {VAL(truth_of(E, w))!r}", flush=True)
        fm.budget = 12
        got, k = fm.ask(ask_text(E))
        print(f"  -> {got!r} after {k} firings", flush=True)


@torch.no_grad()
def deep_trace(A):
    """--trace --load X: everything, said. 1 the translator on every kind of text; 2 one batch of demos with every
    step in full (want, gates, both walks, every candidate with the filter's score, inputs, every output against its
    target at its length and its own, where / how many / kept, what each slot copied, the label's pull or push);
    3 routing over many demo steps (purpose -> function, said exactly); 4 the module living by itself: the first
    statements and one question of each kind with every firing, and its memory against the world's truth"""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr = Translator().to(dev).eval()
    if A.init:
        tr.load_state_dict(torch.load(A.init, map_location=dev), strict=False)
    fm = FM(tr, Acting().to(dev), Heads().to(dev), Filter().to(dev), Critic().to(dev), same=A.same)
    load_fm(fm, A.load)
    for mod in (fm.bank, fm.heads, fm.filt, fm.critic):
        mod.eval()
    print(f"=== 1. THE TRANSLATOR ({len(fm.bank.fns)} made nodes that act in the module)", flush=True)
    translator_report(fm, random.Random(A.seed + 3))
    print("\n=== 2. ONE BATCH OF DEMOS, EVERY STEP (memory written by the teacher, 12 ticks first)", flush=True)
    A.own, fm.frozen = 10 ** 9, True                   # no new node made, memory = what the teacher keeps
    st, fm.g = {"step": 10 ** 9}, Graph(fm.bank.d)
    rng = random.Random(A.seed + 4)
    w = new_world(rng)
    T = Teacher(w)
    for _ in range(max(2, A.trace_n // 2)):
        for d in world_step(w, rng):
            o = d()
            side = w.side.get(id(o), {}) if o else {}
            for text, meta in said(o, w) if o else []:
                demo_statement(fm, T, Lesson(), A, st, text, meta, side)
    fm.trace = True
    for d in world_step(w, rng):
        o = d()
        side = w.side.get(id(o), {}) if o else {}
        for text, meta in said(o, w) if o else []:
            print(f"\n--- statement {text!r} (memory should keep {T.kept(text, meta, side)!r})", flush=True)
            demo_statement(fm, T, Lesson(), A, st, text, meta, side)
    for kind in ("lookup", "before", "coexist", "chain", "ops"):
        E = q_expr(kind, w, rng)
        if E is not None:
            print(f"\n--- question ({kind}) {ask_text(E)!r} truth {VAL(truth_of(E, w))!r}", flush=True)
            demo_question(fm, T, Lesson(), A, st, E)
    print("\n--- functions on functions", flush=True)
    demo_functions(fm, Lesson(), A, st, rng)
    fm.trace = False
    print("\n=== 3. ROUTING OVER MANY DEMO STEPS (purpose -> the function its want reaches; said exactly)", flush=True)
    fm.capture = []
    for _ in range(A.trace_n):
        for d in world_step(w, rng):
            o = d()
            side = w.side.get(id(o), {}) if o else {}
            for text, meta in said(o, w) if o else []:
                demo_statement(fm, T, Lesson(), A, st, text, meta, side)
        for kind in ("lookup", "before", "coexist", "chain", "ops"):
            E = q_expr(kind, w, rng)
            if E is not None:
                demo_question(fm, T, Lesson(), A, st, E)
        demo_functions(fm, Lesson(), A, st, rng)
    by = {}
    for x in fm.capture:
        by.setdefault(x[0], []).append(x)
    for k, xs in sorted(by.items()):
        fns = {}
        for x in xs:
            fns[x[7]] = fns.get(x[7], 0) + 1
        print(f"{k}: {len(xs)} steps -> acting nodes " + ", ".join(f"{f}: {c}" for f, c in sorted(fns.items(), key=lambda z: -z[1])) +
              f" | said exactly {sum(x[1] == x[2] for x in xs)} with the target's pieces, {sum(x[1] == x[3] for x in xs)} with its own", flush=True)
    fm.capture = None
    print("\n=== 4. LIVING BY ITSELF (a fresh world; nothing scripted)", flush=True)
    rng = random.Random(A.seed + 5)
    w = new_world(rng)
    fm.g, fm.stats, n = Graph(fm.bank.d), {}, 0
    for _ in range(A.ticks):
        for d in world_step(w, rng):
            o = d()
            for text, _ in said(o, w) if o else []:
                fm.show = n < 8
                if fm.show:
                    print(f"\nstatement {text!r}", flush=True)
                fm.arrive(text)
                n += 1
    fm.show = False
    newest = [fm.say(fm.entry(p)) for p in range(fm.g.n)]
    truth = {f"{leaf(nm, k)} is {v}." for nm, e in w.ents.items() for k, v in e.items()}
    subj = {}
    for x in newest:
        subj.setdefault(phrase_of(x) or x[:30], 0)
        subj[phrase_of(x) or x[:30]] += 1
    print(f"\nmemory: {fm.g.n} places from {n} statements, entries per place {sum(len(c) for c in fm.g.chain) / max(1, fm.g.n):.2f} | "
          f"newest entries that are a true current fact {sum(x in truth for x in newest)}/{len(newest)} | facts held right somewhere "
          f"{len(truth & set(newest))}/{len(truth)} | subjects with more than one place {sum(c > 1 for c in subj.values())} | {fm.stats}", flush=True)
    print("   places: " + " | ".join(f"{p}: {x[:45]!r} ({len(fm.g.chain[p])})" for p, x in enumerate(newest[:40])), flush=True)
    fm.show, fm.budget = True, 12
    for kind in ("lookup", "before", "coexist", "chain", "ops"):
        E = q_expr(kind, w, rng)
        if E is None:
            continue
        print(f"\nquestion ({kind}) {ask_text(E)!r} truth {VAL(truth_of(E, w))!r}", flush=True)
        got, k = fm.ask(ask_text(E))
        print(f"  -> {got!r} after {k} firings", flush=True)


@torch.no_grad()
def eval_only(A):
    """--eval --load X: the saved module in a fresh world (the code as it is now)"""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr = Translator().to(dev).eval()
    if A.init:
        tr.load_state_dict(torch.load(A.init, map_location=dev), strict=False)
    fm = FM(tr, Acting().to(dev), Heads().to(dev), Filter().to(dev), Critic().to(dev), same=A.same)
    load_fm(fm, A.load)
    for mod in (fm.bank, fm.heads, fm.filt, fm.critic):
        mod.eval()
    print(f"EVAL {A.load}: {evaluate(fm, A, A.seed + 1000)} | {fn_probe(fm, random.Random(1))}", flush=True)
    fm.calc_on = False
    print(f"EVAL {A.load}, calculator removed (its mimic answers): {evaluate(fm, A, A.seed + 1000)}", flush=True)


@torch.no_grad()
def codec_check(A):
    """--codec_check (--load X optional): the translator and the heads on every kind of text the module meets, and
    with --load each function's outputs on its demos (the teacher's inputs given): what decodes back exactly"""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr = Translator().to(dev).eval()
    if A.init:
        tr.load_state_dict(torch.load(A.init, map_location=dev), strict=False)
    fm = FM(tr, Acting().to(dev), Heads().to(dev), Filter().to(dev), Critic().to(dev), same=A.same)
    if A.load:
        load_fm(fm, A.load)
    for mod in (fm.bank, fm.heads, fm.filt, fm.critic):
        mod.eval()
    translator_report(fm, random.Random(A.seed + 3))
    if not A.load:
        return
    codec_functions(fm, A)


@torch.no_grad()
def translator_report(fm, rng):
    tr = fm.tr
    w = new_world(rng)
    kinds = {}
    add = lambda k, t: kinds.setdefault(k, []).append(t)
    for _ in range(15):
        for d in world_step(w, rng):
            o = d()
            for text, meta in said(o, w) if o else []:
                add("rule" if meta[0] == "rule" else "correction" if text.startswith("Correction") else
                    "copy" if "(same as" in text else "fact", text)
    for kind in ("lookup", "before", "coexist", "chain", "ops", "ops"):
        for _ in range(8):
            E = q_expr(kind, w, rng)
            if E is not None:
                add(f"question {kind}", ask_text(E))
                add("part", show(unresolved(E)))
    for _ in range(30):
        add("value", VAL(amount(rng)))
        add("typed string", VAL(_name(rng, 2)))
    for t in CHAT:
        add("chatter", t)
    print("=== text -> B -> text (the translator), and the heads' label against the translator's", flush=True)
    for k, ts in kinds.items():
        ts = [D(t) for t in dict.fromkeys(ts)][:40]
        z, m = tr.content(ts)
        dec = [x.strip() for x in tr.decode(z, m)]
        ok = [a == b for a, b in zip(dec, ts)]
        L, _ = fm.heads([z[i][m[i]] for i in range(len(ts))])
        lab = tr.encode(ts)
        cos = float((L * lab).sum(-1).mean())
        near = float((((L @ lab.T).argmax(1)) == torch.arange(len(ts), device=L.device)).float().mean())
        bad = [f"{a!r} -> {b!r}" for a, b, o_ in zip(ts, dec, ok) if not o_][:2]
        print(f"{k}: round trip {sum(ok)}/{len(ts)} | heads label cos {cos:.3f}, nearest is its own {near:.2f} | " + "; ".join(bad), flush=True)


@torch.no_grad()
def codec_functions(fm, A):
    print("=== each function on its demos (the teacher's inputs given)", flush=True)
    A.own, fm.frozen = 10 ** 9, True                   # no new node made, memory = what the teacher keeps
    st, fm.capture, fm.g = {"step": 10 ** 9}, [], Graph(fm.bank.d)
    rng2 = random.Random(A.seed + 4)
    w2 = new_world(rng2)
    T = Teacher(w2)
    for _ in range(12):
        Ls = Lesson()
        for d in world_step(w2, rng2):
            o = d()
            side = w2.side.get(id(o), {}) if o else {}
            for text, meta in said(o, w2) if o else []:
                demo_statement(fm, T, Ls, A, st, text, meta, side)
    for kind in ("lookup", "before", "coexist", "chain", "ops") * 6:
        E = q_expr(kind, w2, rng2)
        if E is not None:
            demo_question(fm, T, Lesson(), A, st, E)
    for _ in range(8):
        demo_functions(fm, Lesson(), A, st, rng2)
    by = {}
    for x in fm.capture:
        by.setdefault(x[0], []).append(x)
    for k, xs in sorted(by.items()):
        exact = sum(x[1] == x[2] for x in xs)
        own = sum(x[1] == x[3] for x in xs)
        print(f"{k}: said exactly {exact}/{len(xs)} with the target's pieces, {own}/{len(xs)} with its own | loss "
              f"{sum(x[4] for x in xs) / len(xs):.2f} | pieces {sum(x[5] for x in xs) / len(xs):.1f}, target "
              f"{sum(x[6] for x in xs) / len(xs):.1f} | e.g. " + "; ".join(f"{x[1]!r} -> {x[3]!r}" for x in xs[:2]), flush=True)


TRAINED = ("up.", "down.", "tokd.")             # the translator's own layers (text -> B, B -> text) learn with the
#                                                rest; GPT-2 under them stays as it is (Walker's stand-in)


def tr_params(tr):
    return [p for k, p in tr.named_parameters() if k.startswith(TRAINED)]


def save_fm(fm, path):
    torch.save({"n": len(fm.bank.fns), "calc": [t.calc for t in fm.bank.fns], "bank": fm.bank.state_dict(), "heads": fm.heads.state_dict(),
                "filt": fm.filt.state_dict(), "critic": fm.critic.state_dict(),
                "tr": {k: v for k, v in fm.tr.state_dict().items() if k.startswith(TRAINED)}}, path)


def load_fm(fm, path):
    s = torch.load(path, map_location=fm.dev)
    del fm.bank.fns[:]
    for op in s["calc"]:
        fm.bank.fns.append(FnNode(fm.bank.interp.K, fm.bank.d, fm.bank.c, fm.bank.interp.r, calc=op).to(fm.dev))
    for k in ("bank", "heads", "filt", "critic"):
        getattr(fm, k).load_state_dict(s[k])
    if "tr" in s:                                      # the translator as it had learned
        fm.tr.load_state_dict(s["tr"], strict=False)


def run_fm(A):
    """--fm: the codec's second stage, then the demos (statements, questions, functions on functions) with everything
    learning together, then a fresh world the module lives in by itself (evaluate)"""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr = Translator().to(dev).eval()
    if A.init:
        tr.load_state_dict(torch.load(A.init, map_location=dev), strict=False)
    tr.requires_grad_(False)
    for p_ in tr_params(tr):                           # the encoder and the decoder learn from every signal too
        p_.requires_grad_(True)
    fm = FM(tr, Acting().to(dev), Heads().to(dev), Filter().to(dev), Critic().to(dev), same=A.same)
    rng = random.Random(A.seed)
    if A.load:
        load_fm(fm, A.load)
    else:
        train_codec(fm, A, rng, torch.optim.AdamW(fm.heads.parameters(), lr=A.lr))
    own = lambda: [p for mod in (fm.bank, fm.heads, fm.filt, fm.critic) for p in mod.parameters()]
    params = lambda: own() + tr_params(tr)
    make = lambda: torch.optim.AdamW([{"params": own()}, {"params": tr_params(tr), "lr": A.lr_tr}], lr=A.lr)
    O = {"opt": make()}

    def remake():                                      # a function came or went: its params in, the moments kept
        old, new = O["opt"], make()
        for g_ in new.param_groups:
            for p_ in g_["params"]:
                if p_ in old.state:
                    new.state[p_] = old.state[p_]
        O["opt"] = new

    def update(Ls, acc):
        """one optimizer step on these losses (a chunk of the world step)"""
        for k, x in Ls.parts:
            acc[k] = acc.get(k, 0) + float(x)
        if not Ls.parts:
            return
        loss = Ls.total()
        if not torch.is_tensor(loss) or not loss.requires_grad:
            return
        opt = O["opt"]
        opt.zero_grad()
        bad = [k for k, x in Ls.parts if torch.is_tensor(x) and not bool(torch.isfinite(x).all())]
        if bad:                                        # a batch that is not finite does not touch the weights
            fm.count("skipped (not finite)")
            if fm.stats["skipped (not finite)"] <= 5:
                print(f"step {st['step']}: loss not finite in {sorted(set(bad))}", flush=True)
            return
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(params(), 1.)
        if bool(torch.isfinite(norm)):
            opt.step()
            fm.count("updates")
        else:
            fm.count("skipped (gradient not finite)")
            if fm.stats["skipped (gradient not finite)"] <= 5:
                print(f"step {st['step']}: gradient not finite (loss {float(loss):.2f})", flush=True)
        if fm.made:                                    # a node that acts was made: its params join, moments kept
            remake()
            fm.made = False

    def spans(z, m, seen):
        """the translator's round trip over pieces: a span of a text's rows, or two spans of two texts joined, says the
        words those tokens are (a copied piece is only that; the decoder had met whole texts only)"""
        rows = [(z[i][m[i]], tr.tok(t)["input_ids"]) for i, t in enumerate(seen)]
        rows = [(r, ids) for r, ids in rows if len(ids) == len(r) and len(ids) > 1]

        def cut(r, ids):
            a = rng.randrange(len(ids))
            b = min(len(ids) - 1, a + rng.randrange(8))
            return r[a:b + 1], ids[a:b + 1]
        xs, ts = [], []
        for _ in range(min(16, len(rows))):
            parts = [cut(*rng.choice(rows)) for _ in range(rng.choice([1, 1, 2]))]
            xs.append(torch.cat([r for r, _ in parts]))
            ts.append(tr.tok.decode([i for _, ids in parts for i in ids]))
        if not xs:
            return None
        x, xm = pad(xs)
        return tr.decode_loss(x, xm, ts)

    st, t0, recent = {"step": 0}, time.time(), []
    w = T = None
    for step in range(1, A.steps + 1):
        st["step"] = step
        if w is None or step % 150 == 1:               # a new world now and then: names, keys, numbers all new
            w = new_world(rng)
            T = Teacher(w)
            fm.g = Graph(fm.bank.d)
        acc, seen, n0 = {}, [], fm.stats.get("steps run", 0)
        Ls = Lesson()                                  # the statements: one update
        for d in world_step(w, rng):
            o = d()
            side = w.side.get(id(o), {}) if o else {}
            for text, meta in said(o, w) if o else []:
                demo_statement(fm, T, Ls, A, st, text, meta, side)
                seen.append(text)
        finish(fm, Ls, rng, recent)
        update(Ls, acc)
        for _ in range(A.qb):                          # each question: one update
            E = q_expr(rng.choice(["lookup", "before", "coexist", "chain", "ops", "ops"]), w, rng)
            if E is not None:
                Ls = Lesson()
                demo_question(fm, T, Ls, A, st, E)
                seen.append(ask_text(E))
                finish(fm, Ls, rng, recent)
                update(Ls, acc)
        Ls = Lesson()                                  # the functions on functions and the translator: one update
        demo_functions(fm, Ls, A, st, rng)
        finish(fm, Ls, rng, recent)
        seen += fm.said                                # what it reads and what functions say (typed values, parts,
        fm.said = []                                   # filled questions) still says back as it was written, long
        for kind, ops in (("ops", (1, 3)), ("ops", (4, 6)), ("chain", 0), ("before", 0)) * 2:   # questions too
            E = q_expr(kind, w, rng, ops or (1, 3))    # (fm8: with few of them the decoder drifted off long text,
            if E is not None:                          # ops questions said back 15/16 -> 5/16)
                seen.append(ask_text(E))
        if seen:
            seen = [D(x) for x in rng.sample(seen, min(32, len(seen)))]
            z_, m_ = tr.content(seen)
            Ls.add("round trip", tr.decode_loss(z_, m_, seen))
            sp = spans(z_, m_, seen)
            if sp is not None:
                Ls.add("round trip (pieces)", sp)
        update(Ls, acc)
        if step % 50 == 0:
            parts = acc
            loss = sum(acc.values())
            n_run = max(1, fm.stats.get("steps run", 0) - n0)
            print(f"step {step} loss {loss:.2f} (functions per demo step {acc.get('functions', 0) / n_run:.2f}, "
                  f"{fm.stats.get('updates', 0)} updates, {fm.stats.get('cannot be made of pieces', 0)} targets not made of pieces) [" +
                  ", ".join(f"{k} {v:.2f}" for k, v in parts.items()) +
                  f"] made nodes that act {len(fm.bank.fns)} | who acts: " + ", ".join(
                      f"{k[5:]} {v}" for k, v in sorted(fm.stats.items()) if k.startswith("acts")) + " | exact: " + ", ".join(
                      f"{k[:-5]} {fm.stats.get(k[:-5] + ' exact', 0)}/{v}" for k, v in sorted(fm.stats.items()) if k.endswith(" said")) +
                  " | inputs the walk brought: " + ", ".join(
                      f"{k[:-3]} {fm.stats.get(k[:-3] + ' walked', 0)}/{v}" for k, v in sorted(fm.stats.items()) if k.endswith(" in")) +
                  f" | {time.time() - t0:.0f}s", flush=True)
            fm.stats = {}
        if step % A.every == 0 or step == A.steps:
            for mod in (fm.bank, fm.heads, fm.filt, fm.critic):
                mod.eval()
            keep, snap = (fm.g, fm.stats), snapshot(fm.bank)
            print(f"EVAL step {step}: {evaluate(fm, A, A.seed + 1000 + step)} | {fn_probe(fm, random.Random(step))}", flush=True)
            fm.calc_on = False
            print(f"EVAL step {step}, calculator removed (its mimic answers): {evaluate(fm, A, A.seed + 1000 + step)}", flush=True)
            fm.calc_on = True
            restore(fm.bank, snap)
            fm.g, fm.stats = keep
            for mod in (fm.bank, fm.heads, fm.filt, fm.critic):
                mod.train()
            if A.save:
                save_fm(fm, A.save)
    if A.live:                                         # then it lives by itself, learning from signals only
        live(fm, A, rng, O, params, remake)
        for mod in (fm.bank, fm.heads, fm.filt, fm.critic):
            mod.eval()
        snap = snapshot(fm.bank)
        print(f"EVAL after living: {evaluate(fm, A, A.seed + 99999)} | {fn_probe(fm, random.Random(1))}", flush=True)
        restore(fm.bank, snap)
        if A.save:
            save_fm(fm, A.save)


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


if __name__ == "__main__":
    P = argparse.ArgumentParser()
    P.add_argument("--fm", action="store_true", help="the function machine: codec stage 2, demos, then a world by itself")
    P.add_argument("--inspect", action="store_true", help="with --load: every firing of a few questions, said")
    P.add_argument("--eval", action="store_true", help="with --load: the saved module in a fresh world")
    P.add_argument("--trace_n", type=int, default=25, help="demo ticks the trace's routing table runs")
    P.add_argument("--trace", action="store_true", help="with --load: every trace, said (translator, demos, routing, living)")
    P.add_argument("--codec_check", action="store_true", help="the translator, the heads (and with --load each function) on every kind of text")
    P.add_argument("--same", type=float, default=.1, help="a node whose label moved less than this did not change: no signal")
    P.add_argument("--init", default="", help="translator weights (text -> B, B -> text)")
    P.add_argument("--load", default="", help="a saved module (skips the codec stage)")
    P.add_argument("--save", default="")
    P.add_argument("--log", default="")
    P.add_argument("--seed", type=int, default=0)
    P.add_argument("--codec", type=int, default=1500, help="steps of the codec's second stage")
    P.add_argument("--steps", type=int, default=3000)
    P.add_argument("--every", type=int, default=500)
    P.add_argument("--lr", type=float, default=2e-4)
    P.add_argument("--lr_tr", type=float, default=2e-5, help="the translator's own layers (encoder, decoder)")
    P.add_argument("--qb", type=int, default=4, help="questions per step")
    P.add_argument("--q", type=int, default=24, help="questions per kind at eval")
    P.add_argument("--ticks", type=int, default=30, help="world ticks the module lives through before eval questions")
    P.add_argument("--sal_sup", type=float, default=.3, help="weight of the teacher's hint on what is salient")
    P.add_argument("--exact", type=float, default=.3, help="decode loss under which an output counts as said exactly (labels)")
    P.add_argument("--mse", type=float, default=1., help="weight of a made piece's latents against the encoder's for those words")
    P.add_argument("--fix", type=float, default=.15, help="share of stored facts shown a check's failure and its fix")
    P.add_argument("--own", type=int, default=600, help="after this step the demo memory holds what the store itself wrote")
    P.add_argument("--scope", type=float, default=.9)
    P.add_argument("--try_read", type=float, default=.1, help="share of demo steps where the read node nearest the want acts instead")
    P.add_argument("--kappa", type=float, default=2.)
    P.add_argument("--tool_good", type=float, default=.3)
    P.add_argument("--live", type=int, default=300, help="ticks it then lives by itself (signals only)")
    P.add_argument("--lq", type=int, default=4, help="questions per living tick")
    P.add_argument("--tries", type=int, default=4, help="sampled tries for a wrong answer")
    P.add_argument("--explore", type=float, default=1., help="sampling temperature of a try")
    A = P.parse_args()
    random.seed(A.seed), np.random.seed(A.seed), torch.manual_seed(A.seed)
    (deep_trace if A.trace else inspect if A.inspect else eval_only if A.eval else codec_check if A.codec_check else run_fm)(A)
