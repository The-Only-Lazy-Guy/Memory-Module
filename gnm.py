"""GNM: graph memory for Walker. Everything is a node; the only action is a function node, f(nodes) -> nodes.
The model only states what it wants: A to be B.

  python gnm.py --gold      # the correct functions plugged in, no model: does the design solve each task?
"""
import argparse
import copy
import json
import os
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
        if not text.startswith("the "):                # the key a rule is about comes first: "... replace the old offer" is about offer
            hits = [(x.start(), -len(k), k) for k in {words(y) for y in w.num_keys + w.multi_keys + w.ent_keys + w.held_keys}
                    for x in [re.search(rf"\b{re.escape(k)}\b", text.lower())] if x]
            if hits:
                return ("rule", min(hits)[2])
        return ("state", text.split(" is ")[0])

    def label(self, text):
        return self._vec(self.key(text))

    def rule_query(self, text):
        k = self.key(text)
        return self._vec(("rule", k[1])) if k[0] == "fact" else None

    def same(self, a, b, *_):
        return np.linalg.norm(a - b) < self.tau

    node = say = staticmethod(lambda x: x)            # exact: a node is its text

    def relation(self, text):                          # exact: "(same as Y)" claims Y's place, nothing else does
        m = re.search(r"\(same as (.+?)\)", text)
        return self.label(m.group(1)) / .99 if m else np.zeros(self.d, dtype=np.float32)

    def apply(self, want, ins):
        """exact: the want's words say what is done (the stand-in that tests the design around the tools)"""
        if len(ins) == 1:                              # (phrase, entry about an entity) -> phrase about that entity
            return [f"{want.strip()} {value_in(ins[0])}"]
        v, w = [value_in(x) for x in ins], want.lower()
        if not all(isinstance(x, int) for x in v):
            return ["unknown"]
        if any(x.search(want) for x in ADD_RE) or re.search(r"sum|total|add", w):
            return [str(v[0] + v[1])]
        if re.search(r"difference|gap|take away", w):
            return [str(v[0] - v[1])]
        if re.search(r"larger|bigger|maximum", w):
            return [str(max(v))]
        if re.search(r"smaller|lesser|minimum", w):
            return [str(min(v))]
        return [str(v[0] // v[1]), str(v[0] % v[1])] if v[1] else ["unknown"]


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


class NeuralT:
    """the translator as the machine sees it. key is the teacher's routing (gold rows only); no function calls it"""

    def __init__(self, model, oracle, tau, tools=None):
        self.m, self.o, self.tau, self.c, self.d, self.tools, self.learning, self.last = model, oracle, tau, {}, model.d, tools, False, None
        self.key = oracle.key

    @torch.no_grad()
    def node(self, text):                              # text -> B
        z, m = self.m.content([text])
        return z[0][m[0]]

    @torch.no_grad()
    def say(self, z, up=None):                         # B -> text (up: a tool's mouth)
        return self.m.decode(z[None], torch.ones(1, len(z), dtype=torch.bool, device=z.device), up=up)[0].strip()

    @torch.no_grad()
    def check(self, rule, old, new, cand):             # the check node: does this resulting state pass the governing node
        one = lambda it: (it.node[None], torch.ones(1, len(it.node), dtype=torch.bool, device=it.node.device))
        return float(torch.sigmoid(self.tools.check([one(rule), one(old), one(new), one(cand)])[0]))

    def apply(self, want, ins, tool=None):             # the want's label finds a tool node by its key; nodes in, nodes out
        with torch.set_grad_enabled(self.learning):    # the tools learn from the rules' lessons, which run them again
            return self._apply(want, ins, tool)

    def _apply(self, want, ins, tool):
        self.last = None
        s = self.tools.gate(self.tools.query(self.m, [want]))[0] if self.tools.nets else None
        if tool is None and (s is None or float(s.max()) <= 1):    # no tool is relevant: a task it has no tool for
            return ["unknown"]
        t = self.last = int(s.argmax()) if tool is None else tool
        o, many, _ = self.tools.nets[t]([self.m.content([want])] + [(z[None], torch.ones(1, len(z), dtype=torch.bool, device=z.device)) for z in ins])
        return [self.say(z[0], self.tools.nets[t].mouth) or NOTHING for z, _ in o[:int(many.argmax()) + 1]]   # its result is what it says, read like any
        # text (a raw output node as a next input: 11 / 25 right; read back: 25 / 25); saying nothing is the node (nothing)

    @torch.no_grad()
    def _l(self, text):
        if text not in self.c:
            self.c[text] = self.m.encode([text])
        return self.c[text]

    def label(self, text):
        return self._l(text)[0].cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def relation(self, text):                          # what this statement says its node depends on (zero: nothing)
        return self.m.relation([text])[0].cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def rule_query(self, text):
        return self.m.rule_of(self._l(text))[0].cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def same(self, a, b, q=None, anchor=None, task="same"):
        """is the place the walk found the one asked for? read off both nodes' B (q, the place's first entry)"""
        if self.tools is not None and q is not None and anchor is not None:
            one = lambda z: (z[None], torch.ones(1, len(z), dtype=torch.bool, device=z.device))
            return float(self.tools.relate([task], *one(q), *one(anchor))[0]) > 0   # the relate node
        dev = next(self.m.proj.parameters()).device
        return float(self.m.relate(torch.tensor(a, device=dev), torch.tensor(b, device=dev))) > 0


# how things get said: (forms trained on, forms never trained on). A translator that learned templates fails the held
# forms; one that reads language does not.
PHRASE = (["the {k} of {n}", "{n}'s {k}", "{k} of {n}", "the {k} for {n}", "{n} {k}"], ["{n}'s current {k}", "what {n} has as {k}"])
FACT = (["the {k} of {n} is {v}.", "{n}'s {k} is {v}.", "{n} has a {k} of {v}.", "the {k} for {n} is {v}.", "{k} of {n}: {v}",
         "{n}: {k} = {v}", "for {n}, the {k} is {v}.", "{n}'s {k} stands at {v}.", '{{"name": "{n}", "{k}": "{v}"}}',
         "we know {n}'s {k} is {v}.", "{n} -- {k} {v}", "The {k} of {n} is {v}"],
        ["it turns out {n}'s {k} is {v}.", "{k} ({n}): {v}", "as for {n}, its {k} is {v}.", "{v} is the {k} of {n}."])
CHANGE = (["Correction: the {k} of {n} is now {v}.", "Update: {n}'s {k} is {v}.", "{n}'s {k} is now {v}.",
           "the {k} of {n} changed to {v}.", "actually, {n}'s {k} is {v}."], ["{n}'s {k} went to {v}.", "scratch that: the {k} of {n} is {v}."])
COPY = (["the {k} of {n} is {v} (same as {s}).", "{n}'s {k} equals {s}, which is {v}.", "{n}'s {k} matches {s}: {v}.",
         "{n}'s {k} is the same as {s} ({v})."], ["{n}'s {k} is copied from {s} ({v}).", "{v}, like {s}, is the {k} of {n}."])


def say(rng, form, held=False, **kw):
    """one way of saying it, sometimes untidy (lower case, a doubled space)"""
    t, r = rng.choice(form[held]).format(**kw), rng.random()
    return t.lower() if r < .1 else t.replace(" ", "  ", 1) if r < .15 else t


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
    src = [(rng.choice(names), rng.choice(keys)) for _ in range(n)]           # a copy depends on its source; nothing else does
    said = [f"{leaf(nm, k)} is {rng.randint(1, 60)} (same as {leaf(*sf)})." for (nm, k), sf in zip(rng.sample(facts, n), src)]
    plain = [rng.choice([f"{leaf(*f)} is {rng.randint(1, 60)}.", f"Correction: {leaf(*f)} is now {rng.randint(1, 60)}.", rv(f[1])])
             for f in rng.sample(facts, n)]
    ra, rg, roff, rmk = model.relation(said + plain, parts=True)
    span = torch.zeros_like(ra)                        # the tokens that name the source: inside "(same as ...)"
    for i, (t, sf) in enumerate(zip(said, src)):
        lo = t.index("(same as ") + len("(same as ")
        hi = lo + len(leaf(*sf))
        span[i] = ((roff[i, :, 1] > lo) & (roff[i, :, 0] < hi)).float()
    loss = (loss + (F.binary_cross_entropy_with_logits(ra, span, reduction="none") * rmk).sum() / rmk.sum()
            + F.binary_cross_entropy_with_logits(rg, torch.cat([torch.ones(n), torch.zeros(n)]).to(rg.device)))   # depends at all?
    # a fact asks for its key's rule: the query lands on the rule, not on facts or phrases of that key, and not on the
    # asking fact's own label (online it sat .70 on its own label and found its own place first). Before 11-03 this read
    # `names` after it had become the span tensor above: every asking fact was "the {key} of tensor([0., ...]) is v."
    ask = [views(rng, rng.choice(names), k) for k in keys]
    q, own = model.rule_of(model.encode(ask)), model.encode(ask)
    t = torch.arange(n, device=sim.device)
    return (loss + F.cross_entropy(b @ a.T / .07, t) + relate_loss(model, b, a[:n], b @ a[:n].T)
            + F.cross_entropy(q @ torch.cat([b, a[n:], own]).T / .07, t) + relate_loss(model, q, b, q @ b.T)
            + F.relu((q - b).norm(dim=-1) - .4).mean())   # and lands as near its rule as a restatement lands on its fact
    # (a place's first reach is .5; tr8 ranked right but sat .62-.70 away online, so each rule place failed until its reach grew)


@torch.no_grad()
def label_eval(model, rng, vocab, n=40):
    """fresh names and keys: does a phrase find its fact among everything, and how far apart are same and different"""
    used, ku = set(), set()
    names = [_name(rng, 2, used) for _ in range(n)]
    keys = [" ".join(_key(rng, vocab, ku)) for _ in range(n)]
    facts = [(a, b) for a in names for b in keys]
    enc = lambda T: torch.cat([model.encode(T[i:i + 256]) for i in range(0, len(T), 256)])
    S = Sf = enc([f"{leaf(*f)} is {rng.randint(1, 60)}." for f in facts])
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
    Q = model.rule_of(enc([f"{leaf(rng.choice(names), k)} is {rng.randint(1, 60)}." for k in keys]))
    C = Q @ R[:n].T
    out["ruleq"] = float((C.argmax(1) == torch.arange(n, device=C.device)).float().mean())
    out["ruleq_rel+"] = float((model.relate(Q, R[:n]) > 0).float().mean())
    ask = [f"{leaf(rng.choice(names), k)} is {rng.randint(1, 60)}." for k in keys]  # nearer its rule than to its own label?
    Q, own = model.rule_of(enc(ask)), enc(ask)
    out["ruleq_over_own"] = float(((Q * R[:n]).sum(1) > (Q * own).sum(1)).float().mean())
    out["ruleq_dist"] = float((Q - R[:n]).norm(dim=1).mean())
    src = [(rng.choice(names), rng.choice(keys)) for _ in range(n)]
    Rl = model.relation([f"{leaf(rng.choice(names), rng.choice(keys))} is 5 (same as {leaf(*sf)})." for sf in src]
                        + [f"{leaf(rng.choice(names), rng.choice(keys))} is {rng.randint(1, 60)}." for _ in range(n)])
    C = Rl[:n] @ Sf.T                                  # against every fact statement: the source's is the only one > 1
    ix = torch.tensor([names.index(a) * len(keys) + keys.index(b) for a, b in src], device=C.device)
    out["rel_claims_source"] = float((C.gather(1, ix[:, None]).squeeze(1) > 1).float().mean())
    out["rel_claims_other"] = float(((C > 1).sum(1) - (C.gather(1, ix[:, None]).squeeze(1) > 1).long()).float().mean() / C.shape[1])
    out["plain_claims_any"] = float(((Rl[n:] @ Sf.T) > 1).any(1).float().mean())
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


def sentence(rng, vocab):
    """something the memory could be told or could hold: a fact, a correction, a copy, a rule, a bare value"""
    nm, k, r = _name(rng, 2), " ".join(_key(rng, vocab, set())), rng.random()
    return (views(rng, nm, k) if r < .6 else rule_text(rng, rng.choice(("add", "set", "keep")), k) if r < .75
            else str(rng.randint(-300, 300)) if r < .9 else f"rule: {k}")


REAL = "E:/walker_prefix_data/hotpot_cards.jsonl"     # 36,840 Wikipedia paragraphs (HotpotQA); every 10th held out
_real = {}


def real_text(held):
    """real sentences (node <-> text must hold for any text, not my templates: tr9 said 0 / 120 back, tokens .02)"""
    if held not in _real:
        if os.path.exists(REAL):
            paras = [json.loads(l)["memory"]["text"] for l in open(REAL, encoding="utf-8")]
        else:                                          # the box: one public Wikipedia shard from the hub
            import pyarrow.parquet as pq
            from huggingface_hub import hf_hub_download
            f = hf_hub_download("wikimedia/wikipedia", "20231101.en/train-00000-of-00041.parquet", repo_type="dataset")
            paras = [p for t in pq.read_table(f, columns=["text"]).column("text").to_pylist()[:20000] for p in t.split("\n") if len(p) > 80]
        out = []
        for i, t in enumerate(paras):
            if (i % 10 == 0) == held:                  # every 10th paragraph held out
                out += [x.strip() for x in re.split(r"(?<=[.!?])\s+", t) if 2 <= len(x.split()) <= 40 and len(x) <= 300]
                # characters, not words: a 40-"word" list with no spaces was 1,093 tokens, past the LM's 1,024 (nt1 crash)
        _real[held] = out
    return _real[held]


def any_text(rng, vocab=KEYS_TRAIN, held=False):
    """what a node may have to hold, at any length: a real sentence (40%), a short piece (20%: a bare number, a name,
    a phrase, the start of a real sentence; a tool's result read back is one token, and nt1 said '8' back as
    '1818...'), or the memory's own kinds of text (statements, rules, questions, records, codes)"""
    r = rng.random()
    if r < .4:
        return rng.choice(real_text(held))
    if r < .6:
        return rng.choice([lambda: str(amount(rng)), lambda: str(rng.randint(-999, 999)), lambda: _name(rng, rng.choice((1, 2, 3))),
                           lambda: leaf(_name(rng, 2), " ".join(_key(rng, vocab, set()))),
                           lambda: " ".join(rng.choice(real_text(held)).split()[:rng.randint(1, 4)])])()
    return rng.choice([lambda: sentence(rng, vocab), lambda: question_text(rng, vocab),
                       lambda: f"{leaf(_name(rng, 2), rng.choice(NOISE_KEYS))} is {_value(rng, 'code')}.",
                       lambda: json.dumps({rng.choice(vocab): rng.randint(-500, 500) for _ in range(rng.randint(1, 3))})])()


def question_text(rng, vocab):
    nm, k = _name(rng, 2), " ".join(_key(rng, vocab, set()))
    return rng.choice([f"What is {leaf(nm, k)}?", f"What was {leaf(nm, k)} before?",
                       f"What is the sum of {leaf(nm, k)} and {leaf(_name(rng, 2), k)}?"])


@torch.no_grad()
def real_eval(model, rng, n=96):
    """held-out real sentences: exact round trip and token accuracy"""
    T = rng.sample(real_text(True), n)
    D = [d.strip() for i in range(0, n, 32) for d in model.decode(*model.content(T[i:i + 32]), n=96)]
    tok = [sum(a == b for a, b in zip(model.tok(t)["input_ids"], model.tok(d)["input_ids"])) / len(model.tok(t)["input_ids"]) for t, d in zip(T, D)]
    return f"real exact {sum(t == d for t, d in zip(T, D)) / n:.3f} tokens {sum(tok) / n:.3f}", [(t, d) for t, d in zip(T, D) if t != d][:2]


@torch.no_grad()
def decode_eval(model, rng, vocab, n=256):
    """round trip text -> B -> text on fresh names (and keys), exact, by kind of sentence"""
    T = [sentence(rng, vocab) for _ in range(n)]
    D = model.decode(*model.content(T))
    kind = lambda t: "value" if re.fullmatch(r"-?\d+", t) else "rule" if not t.startswith(("the ", "Correction")) else "copy" if "(same as" in t else "fact"
    hit, tot = {}, {}
    for t, d in zip(T, D):
        hit[kind(t)] = hit.get(kind(t), 0) + (d.strip() == t)
        tot[kind(t)] = tot.get(kind(t), 0) + 1
    bad = [(t, d) for t, d in zip(T, D) if d.strip() != t][:3]
    return " ".join(f"{k} {hit[k] / tot[k]:.3f}" for k in sorted(tot)) + f" all {sum(hit.values()) / n:.3f}", bad


def train_decoder(A):
    """the translator's other half; its label heads stay as they are"""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = Translator().to(dev)
    if A.init:
        model.load_state_dict(torch.load(A.init, map_location=dev), strict=False)
    new = [p for n, p in model.named_parameters() if n.split(".")[0] in ("down", "up", "tokd")]
    opt = torch.optim.AdamW(new, lr=A.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, A.steps, eta_min=A.lr / 50)
    rng = random.Random(A.seed)
    t0 = time.time()
    for step in range(1, A.steps + 1):
        T = [any_text(rng) for _ in range(A.batch)]
        loss = model.decode_loss(*model.content(T), T)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % A.every == 0:
            model.eval()
            tr, bad = decode_eval(model, rng, KEYS_TRAIN)
            he, bad2 = decode_eval(model, rng, KEYS_EVAL)
            re_, bad3 = real_eval(model, rng)
            print(f"step {step} loss {float(loss):.3f} | round trip, train keys: {tr} | held keys: {he} | {re_} | {time.time() - t0:.0f}s", flush=True)
            for t, d in bad + bad2 + bad3:
                print(f"     {t!r} -> {d!r}", flush=True)
            model.train()
            if A.save:
                torch.save({k: v for k, v in model.state_dict().items() if not k.startswith("lm.")}, A.save)


class CheckNet(nn.Module):
    """the check function node: check(the governing node, old, new, the resulting state) -> does the state pass it
    (a logit). It reads the four nodes token by token; it learns from ~50 demos, then from outcome"""

    def __init__(self, c=256, L=96):
        super().__init__()
        self.cls = nn.Parameter(torch.randn(c) * .02)
        self.side = nn.Parameter(torch.randn(4, c) * .02)                  # rule / old / new / resulting state
        self.pos = nn.Parameter(torch.randn(L, c) * .02)
        self.f = nn.TransformerEncoder(nn.TransformerEncoderLayer(c, 8, 1024, batch_first=True, activation="gelu"), 2)
        self.o = nn.Linear(c, 1)

    def forward(self, parts):                          # parts: 4 x (z [n, L, c], mask [n, L])
        n = parts[0][0].shape[0]
        x = [self.cls.expand(n, 1, -1)] + [z + self.side[k] + self.pos[(m.long().cumsum(1) - 1).clamp(0, len(self.pos) - 1)]
                                          for k, (z, m) in enumerate(parts)]
        keep = torch.cat([torch.ones(n, 1, dtype=torch.bool, device=x[1].device)] + [m for _, m in parts], 1)
        return self.o(self.f(torch.cat(x, 1), src_key_padding_mask=~keep)[:, 0]).squeeze(-1)


def check_demos(rng, n, vocab=KEYS_TRAIN, held=False):
    """situations a check meets (governing node, old, new, candidate, pass?). The check judges only which kind of
    state the node that governs calls for: the statement as said, or what the tool made of (old, new). Whether a node
    governs at all is decided where it is found (the rule query + relate), and the sum is the tool's: neither is the
    check's (ck4: with other keys' rules and wrong sums among its demos it learned 'as said passes' and got the adding
    rule's own case 4 / 33). What arrives is a fact, a correction, a copy or a rule; what governs is the rule of its
    key or a fact; only a rule that adds, over a plain fact, makes what the tool made pass and the statement fail"""
    out = []
    for _ in range(n):
        nm, k = _name(rng, 2), " ".join(_key(rng, vocab, set()))
        kind = rng.choice(("add", "add", "set", "keep"))
        a, b = amount(rng), amount(rng)
        is_rule = rng.random() < .8
        gov = rule_text(rng, kind, k, held) if is_rule else f"{leaf(_name(rng, 2), k)} is {amount(rng)}."
        w = rng.random()
        if w < .15:                                    # a rule arrives (old = the rule of its key before)
            old, new = rule_text(rng, rng.choice(("add", "set", "keep")), k, held), rule_text(rng, kind, k, held)
            plain = False
        else:
            old = rng.choice([f"{leaf(nm, k)} is {a}.", str(a)])
            new = (f"Correction: {leaf(nm, k)} is now {b}." if w < .3 else
                   f"{leaf(nm, k)} is {b} (same as {leaf(_name(rng, 2), k)})." if w < .4 else f"{leaf(nm, k)} is {b}.")
            plain = w >= .4
        adds = is_rule and kind == "add" and plain
        made = str(a + b) if adds else rng.choice([str(a + b), str(a - b), str(max(a, b)), "--"])   # what a tool made
        out += [(gov, old, new, made, float(adds)), (gov, old, new, new, float(not adds))]
    return out


def check_run(tr, net, batch):
    rows = list(zip(*batch))
    parts = [tr.content(list(t)) for t in rows[:4]]
    return net(parts), torch.tensor(rows[4], device=parts[0][0].device)


def train_check(A):
    """--check: the check node learns from ~50 situations (A.demos), tested on fresh keys and held rule phrasings"""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr = Translator().to(dev).eval()
    if A.init:
        tr.load_state_dict(torch.load(A.init, map_location=dev), strict=False)
    tr.requires_grad_(False)
    bank = ToolBank.load(A.toolw, dev) if A.toolw else ToolBank().to(dev)
    rng = random.Random(A.seed)
    if bank.nets:                                      # does a rule, read as a want, find a tool by its label alone?
        with torch.no_grad():
            for held in (False, True):
                line = []
                for kind in ("add", "set", "keep"):
                    s_ = bank.gate(bank.query(tr, [rule_text(rng, kind, " ".join(_key(rng, KEYS_EVAL, set())), held) for _ in range(64)]))
                    line.append(f"{kind}: relevant to a tool {float((s_.max(1).values > 1).float().mean()):.2f} (mean best {float(s_.max(1).values.mean()):.2f})")
                print(f"rules as wants ({'held' if held else 'seen'} phrasings): " + " | ".join(line), flush=True)
    demos = check_demos(rng, A.demos)
    opt = torch.optim.AdamW(bank.check.parameters(), lr=A.lr)
    t0 = time.time()
    for step in range(1, A.steps + 1):
        lg, y = check_run(tr, bank.check, random.sample(demos, min(32, len(demos))))
        loss = F.binary_cross_entropy_with_logits(lg, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % A.every == 0:
            bank.check.eval()
            with torch.no_grad():
                line = []
                for name, ev in (("fresh keys", check_demos(random.Random(step), 200, KEYS_EVAL)),
                                 ("held phrasings", check_demos(random.Random(step + 1), 200, KEYS_EVAL, True))):
                    lg, y = check_run(tr, bank.check, ev)
                    line.append(f"{name} {float(((lg > 0).float() == y).float().mean()):.3f}")
            print(f"step {step} loss {float(loss):.3f} | check right: " + " | ".join(line) + f" | {time.time() - t0:.0f}s", flush=True)
            if step == A.steps:                        # where it is wrong: by what governs, what arrives, the candidate
                bins = {}
                for (gov, old, new, cand, y_), l in zip(ev, lg.tolist()):
                    nk = ("rule" if not new.startswith(("the ", "Correction")) else "correction" if new.startswith("Correction")
                          else "copy" if "(same as" in new else "fact")
                    key = (f"governs: {'fact' if gov.startswith('the ') else 'rule-add' if any(x.search(gov) for x in ADD_RE) else 'rule-other'}",
                           f"arrives: {nk}", f"candidate: {'as said' if cand == new else cand if cand == '--' else 'a number'}",
                           f"should {'pass' if y_ else 'fail'}")
                    b = bins.setdefault(key, [0, 0])
                    b[0], b[1] = b[0] + ((l > 0) == (y_ > .5)), b[1] + 1
                for k, (ok, n) in sorted(bins.items(), key=lambda z: z[1][0] / z[1][1])[:12]:
                    print(f"   {ok:3d}/{n:3d} right  {' | '.join(k)}", flush=True)
            bank.check.train()
            if A.save:
                bank.save(A.save)


class Relate(nn.Module):
    """the relate node: reads two nodes' B token by token, a logit. same: the same place? rule: the rule for this fact?"""

    def __init__(self, c=256, L=96):
        super().__init__()
        self.tasks = ["same", "rule"]
        self.task = nn.Parameter(torch.randn(len(self.tasks), c) * .02)
        self.side = nn.Parameter(torch.randn(2, c) * .02)
        self.pos = nn.Parameter(torch.randn(L, c) * .02)
        self.f = nn.TransformerEncoder(nn.TransformerEncoderLayer(c, 8, 1024, batch_first=True, activation="gelu"), 2)
        self.o = nn.Linear(c, 1)

    def forward(self, task, a, am, b, bm):
        n = len(task)
        i = torch.tensor([self.tasks.index(t) for t in task], device=a.device)
        at = lambda z, m, k: z + self.side[k] + self.pos[(m.long().cumsum(1) - 1).clamp(0, len(self.pos) - 1)]   # a long node shares the last
        x = torch.cat([self.task[i][:, None], at(a, am, 0), at(b, bm, 1)], 1)
        keep = torch.cat([torch.ones(n, 1, dtype=torch.bool, device=a.device), am, bm], 1)
        return self.o(self.f(x, src_key_padding_mask=~keep)[:, 0]).squeeze(-1)


class ToolNet(nn.Module):
    """one tool node's own network. In: the want, then any number of nodes (B each). Out: up to M nodes (K latents each),
    how many of them it means, and its own guess of how wrong it is (the score). Its key is its label: what it answers to."""

    def __init__(self, d=128, c=256, K=12, M=2, R=8, L=96, D=768):
        super().__init__()
        self.key = nn.Parameter(F.normalize(torch.randn(d), dim=0))   # the label: relevant to q when q . key > 1
        self.register_buffer("mu", torch.tensor(5.))                     # the error it usually gets, and its spread
        self.register_buffer("sd", torch.tensor(2.))
        self.role = nn.Parameter(torch.randn(R, c) * .02)      # which input
        self.pos = nn.Parameter(torch.randn(L, c) * .02)
        self.out = nn.Parameter(torch.randn(M * K, c) * .02)
        self.f = nn.TransformerEncoder(nn.TransformerEncoderLayer(c, 8, 512, batch_first=True, activation="gelu"), 2)
        self.many = nn.Linear(c, M)
        self.err = nn.Linear(c, 1)
        self.K, self.M = K, M
        self.mouth = nn.Sequential(nn.Linear(c, D), nn.GELU(), nn.Linear(D, D))   # how it says its results: its own, so
        # the decoder learning to say stored nodes never moves what a tool says (a shared one did: arithmetic 155 -> 110 / 160)
        self.register_buffer("voiced", torch.tensor(0.))    # 0: not yet given the decoder's mouth to start from

    def forward(self, ins):
        n, dev = ins[0][0].shape[0], self.out.device
        parts, keep = [self.out[None].expand(n, -1, -1)], [torch.ones(n, len(self.out), dtype=torch.bool, device=dev)]
        for r, (z, m) in enumerate(ins):
            parts.append(z + self.role[r] + self.pos[(m.long().cumsum(1) - 1).clamp(0, len(self.pos) - 1)])
            keep.append(m)
        y = self.f(torch.cat(parts, 1), src_key_padding_mask=~torch.cat(keep, 1))[:, :len(self.out)]
        y = F.layer_norm(y, (y.shape[-1],))
        g, one = y.mean(1), torch.ones(n, self.K, dtype=torch.bool, device=dev)
        return [(y[:, j * self.K:(j + 1) * self.K], one) for j in range(self.M)], self.many(g), self.err(g).squeeze(-1)


class ToolBank(nn.Module):
    """the tool nodes the model has made so far (none at first), and the relate node"""

    def __init__(self, d=128, c=256, cap=16):
        super().__init__()
        self.nets, self.relate, self.d, self.c, self.cap = nn.ModuleList(), Relate(c), d, c, cap
        self.check = CheckNet(c)                       # the check function node
        self.want = nn.Sequential(nn.Linear(768, 512), nn.GELU(), nn.Linear(512, d))   # a want, in the space the labels live in
        self.register_buffer("center", torch.zeros(768))     # GPT-2's mean states all point one way: taken out, wants start apart
        self.register_buffer("scale", torch.ones(768))

    def query(self, tr, texts, track=False):
        """a want as a point: learned from what the tools did, so wants one tool answers come together"""
        dev = next(self.want.parameters()).device
        b = tr.tok(texts, return_tensors="pt", padding=True).to(dev)
        with torch.no_grad():
            h = tr.norm(tr.lm(**b).last_hidden_state)
            m = b["attention_mask"][..., None]
            x = (h * m).sum(1) / m.sum(1)
            if track and len(texts) > 1:               # running mean and spread of what wants look like
                first = not self.center.any()
                self.center.mul_(0 if first else .99).add_((1 if first else .01) * x.mean(0))
                self.scale.mul_(0 if first else .99).add_((1 if first else .01) * x.std(0).clamp(min=1e-3))
        return F.normalize(self.want((x - self.center) / self.scale), dim=-1)

    def spawn(self, key, like=None, scope=.9):
        """a new tool node: its own network (started from the tool it is most like), its label = the want that needed
        it, long enough that wants within cos > scope of it are relevant"""
        net = ToolNet(self.d, self.c).to(key.device)
        if like is not None:
            net.load_state_dict(self.nets[like].state_dict())
            net.mu.fill_(5.), net.sd.fill_(2.)
        with torch.no_grad():
            net.key.copy_(F.normalize(key, dim=0) / scope)
        self.nets.append(net)
        return len(self.nets) - 1

    @torch.no_grad()
    def voice(self, tr):
        """a tool without its own mouth yet starts from the decoder's (then only its own lessons move it)"""
        for net in self.nets:
            if not net.voiced:
                net.mouth.load_state_dict(tr.up.state_dict())
                net.voiced.fill_(1.)
        return self

    def gate(self, q):                                 # every want against every tool's label: > 1 = relevant
        return q @ torch.stack([t.key for t in self.nets]).T

    def save(self, path):
        torch.save({"n": len(self.nets), "state": self.state_dict()}, path)

    @staticmethod
    def load(path, dev):
        s = torch.load(path, map_location=dev)
        b = ToolBank().to(dev)
        for _ in range(s["n"]):
            b.nets.append(ToolNet().to(dev))
        miss = b.load_state_dict(s["state"], strict=False).missing_keys     # banks made before tools had a mouth
        assert all(".mouth." in k or k.endswith(".voiced") or k.startswith("check.") for k in miss), miss
        return b


def lpad(z, m, L):
    """left-pad a node's latents to length L"""
    k = L - z.shape[1]
    return (torch.cat([z.new_zeros(z.shape[0], k, z.shape[2]), z], 1), torch.cat([m.new_zeros(m.shape[0], k), m], 1)) if k else (z, m)


def said(rng, v):
    """a node holding the value v, as the memory might hold it"""
    nm, k = _name(rng, 2), " ".join(_key(rng, KEYS_TRAIN, set()))
    return rng.choice([str(v), f"{leaf(nm, k)} is {v}.", f"Correction: {leaf(nm, k)} is now {v}.", f"is {v}."])


def load_tools(A, tr):
    """the tool nodes and the relate node (--toolw), or None"""
    return ToolBank.load(A.toolw, next(tr.proj.parameters()).device).eval().voice(tr) if A.toolw else None


# the light supervision: what kinds of thing the model reads being done. No tool is named, nor how many there are.
WANTS = {"add": ["the sum of", "the total of", "add up"], "sub": ["the difference between", "the gap between", "take away"],
         "max": ["the larger of", "the bigger of", "the maximum of"], "min": ["the smaller of", "the lesser of", "the minimum of"],
         "divmod": ["the quotient and remainder of", "share out in groups of"]}
HELD_WANTS = {"the total of", "the maximum of", "the lesser of", "share out in groups of"}   # never read: routed by itself?
KINDS = ("add", "sub", "max", "min", "divmod", "rule", "both", "join")


def amount(rng):
    """a number as the machine meets it: stored values are 1-60, but one tool's result goes on to the next, so
    differences go below zero and sums past 60 (online lessons on such numbers, never shown in demos 1-60, pulled the
    arithmetic tool off what it knew: one lesson on 23 + -17 cost 6 / 160 of its probe)"""
    r = rng.random()
    return rng.randint(1, 60) if r < .6 else rng.randint(-60, 0) if r < .8 else rng.randint(61, 200)


def read_demo(rng, kind, held=False, vocab=KEYS_TRAIN):
    """one thing being done, as read: (kind, the want, the input nodes, what came out)"""
    if kind in WANTS:
        w = rng.choice([x for x in WANTS[kind] if (x in HELD_WANTS) == held] or WANTS[kind])
        if kind == "divmod":
            a, b = rng.randint(10, 120), rng.randint(2, 12)
            return kind, w, [said(rng, a), said(rng, b)], [str(a // b), str(a % b)]
        a, b = amount(rng), amount(rng)
        return kind, w, [said(rng, a), said(rng, b)], [str(next(t for t in TOOLS if t[0] == kind)[1](a, b))]
    nm, k, v, x = _name(rng, 2), " ".join(_key(rng, vocab, set())), amount(rng), amount(rng)
    if kind == "rule":                                 # a rule being applied to what arrived
        return kind, rule_text(rng, "add", k), [f"{leaf(nm, k)} is {v}.", f"is {x}."], [str(v + x)]
    if kind == "both":
        return kind, f"What is the sum of both values of {leaf(nm, k)}?", [f"{leaf(nm, k)} is {v}.", f"{leaf(nm, k)} is {x}."], [str(v + x)]
    k2, e = " ".join(_key(rng, vocab, set())), _name(rng, 2)
    return kind, f"the {k2} of", [rng.choice([f"{leaf(nm, k)} is {e}.", f"Correction: {leaf(nm, k)} is now {e}."])], [f"the {k2} of {e}"]


OTHER_VERBS = (["translate", "sort", "reverse", "summarize", "rename", "delete", "explain", "describe", "shorten", "underline"],
               ["count the words in", "spell out", "make a list of", "proofread", "encrypt", "capitalize", "paraphrase", "archive"])
OTHER_OBJ = ["this sentence", "these names", "the message from {n}", "the {k} of {n}", "{n}'s notes", "the letters of {n}",
             "everything {n} said", "the last line"]


def asks_nothing(rng, held=False):
    """text no tool here answers: talk, a statement, a lookup question, or a task no demo shows (open-set negatives;
    held = the other verbs, never seen in learning)"""
    nm, k = _name(rng, 2), " ".join(_key(rng, KEYS_TRAIN, set()))
    other = f"{rng.choice(OTHER_VERBS[held])} {rng.choice(OTHER_OBJ).format(n=nm, k=k)}"
    return other if held else rng.choice([rng.choice(CHATTER[0]), views(rng, nm, k), f"What is {leaf(nm, k)}?", other, other])


def kinds_apart(q, kinds, tau=.1):
    """contrastive want space: each want is pulled toward the wants of its own kind and away from all others (the kind
    is the light supervision; 'none' = asks for nothing, never anyone's positive). Pull-to-label alone collapsed every
    want onto the one general tool's label (tl10: cos .95 same kind / .93 other, chatter relevant at 1.10)"""
    ids = {k: i for i, k in enumerate(dict.fromkeys(kinds))}
    lab = torch.tensor([ids[k] for k in kinds], device=q.device)
    pos = (lab[:, None] == lab[None]) & ~torch.eye(len(kinds), dtype=torch.bool, device=q.device)
    pos &= torch.tensor([k != "none" for k in kinds], device=q.device)[:, None]
    lg = (q @ q.T / tau).masked_fill(torch.eye(len(kinds), dtype=torch.bool, device=q.device), -1e9)
    lp = lg - lg.logsumexp(1, keepdim=True)
    has = pos.any(1)
    return -((lp * pos).sum(1)[has] / pos.sum(1)[has]).mean() if has.any() else q.sum() * 0


def tool_loss(tr, net, ins, outs):
    """per demo: how wrong this tool is (each output it should say, and how many)"""
    o, many, err = net(ins)
    n = torch.tensor([len(x) for x in outs], device=many.device)
    L = F.cross_entropy(many, n - 1, reduction="none")
    for j, (z, zm) in enumerate(o):
        L = L + tr.decode_loss(z, zm, [x[j] if j < len(x) else "" for x in outs], each=True, up=net.mouth) * (n > j) / n
    return L, err


def learn(tr, bank, demos, st, A, spawn=True, none=()):
    """read demos. Only the label vectors decide: a tool is relevant to a want q when q . label > 1 (direction = what it
    does, length = how broad); the most relevant one does it; when none is, the task is new and the model makes a tool
    whose label is that want (its network started from the closest tool). What the tools then do only teaches the
    labels: a tool that did a demo as well as it usually does is pulled toward that want; a tool that claimed a demo and
    did it unusually badly for itself is pushed away from it. The wants themselves stay apart by kind (contrastive), and
    text that asks for nothing (none) is relevant to no label. Returns the loss and where each demo went (-1: new task)."""
    q = bank.query(tr, [d[1] for d in demos], track=True)      # learns with the labels
    qn = bank.query(tr, list(none)) if none else q[:0]
    apart = kinds_apart(torch.cat([q, qn]), [d[0] for d in demos] + ["none"] * len(qn))
    groups = {}
    for i, d in enumerate(demos):
        groups.setdefault(len(d[2]), []).append(i)
    nodes = {n: [tr.content([demos[i][1] for i in ix])] + [tr.content([demos[i][2][j] for i in ix]) for j in range(n)]
             for n, ix in groups.items()}
    with torch.no_grad():
        s = bank.gate(q) if bank.nets else q.new_zeros(len(demos), 0)
        new = s.max(1).values <= 1 if bank.nets else torch.ones(len(demos), dtype=torch.bool, device=q.device)
        if spawn and new.any() and st["step"] - st["last"] >= A.cool and len(bank.nets) < bank.cap:
            b = int(new.nonzero()[0])
            st.update(last=st["step"], newest=bank.spawn(q[b], int(s[b].argmax()) if bank.nets else None, A.scope), spawned=True)
            bank.voice(tr)
            s = bank.gate(q)
        if not bank.nets:
            return apart, torch.full((len(demos),), -1), s
        route = torch.where(s.max(1).values > 1, s.argmax(1), torch.full_like(s.argmax(1), -1))
        L = torch.full((len(bank.nets), len(demos)), 1e9, device=q.device)
        for n, ix in groups.items():
            for t, net in enumerate(bank.nets):
                L[t, ix] = tool_loss(tr, net, nodes[n], [demos[i][3] for i in ix])[0]
        usual = torch.stack([torch.maximum(t.mu + A.kappa * t.sd, torch.tensor(A.tool_good, device=q.device)) for t in bank.nets])
        typical = L <= usual[:, None]                                  # [tool, demo]: as well as this tool usually does
        best = L.argmin(0)
    S, m = bank.gate(q).T, .1                                          # [tool, demo]
    pull = typical & (torch.arange(len(bank.nets), device=q.device)[:, None] == best[None])
    push = ~typical & (S.detach() > 1 - m)
    loss = (F.relu(1 + m - S) * pull).sum() / len(demos) + (F.relu(S - (1 - m)) * push).sum() / len(demos) + apart
    if len(qn):                                                        # asks for nothing: below every label
        loss = loss + F.relu(bank.gate(qn) - (1 - m)).sum() / len(demos)
    for n, ix in groups.items():
        for t, net in enumerate(bank.nets):
            sel = [k for k, i in enumerate(ix) if int(route[i]) == t]
            if sel:
                ins = [(z[sel], m_[sel]) for z, m_ in nodes[n]]
                l, err = tool_loss(tr, net, ins, [demos[ix[k]][3] for k in sel])
                loss = loss + (l.sum() + F.mse_loss(err, l.detach(), reduction="sum")) / len(demos)
                with torch.no_grad():                                  # what this tool usually gets
                    net.mu.mul_(.98).add_(.02 * l.mean())
                    net.sd.mul_(.98).add_(.02 * (l - net.mu).abs().mean())
    return loss, route, s


@torch.no_grad()
def apply_tool(tr, bank, want, ins):
    """the want's label finds the tool by the labels alone (-1: no tool is relevant, a new task); it says how many nodes"""
    s = bank.gate(bank.query(tr, [want]))[0] if bank.nets else None
    if s is None or float(s.max()) <= 1:
        return -1, [], None
    t = int(s.argmax())
    o, many, err = bank.nets[t]([tr.content([want])] + [tr.content([x]) for x in ins])
    return t, [z for z, _ in o[:int(many.argmax()) + 1]], float(err)


def near(rng, nm):
    """a name one syllable off"""
    for _ in range(50):
        x = _name(rng, 2)
        if x != nm and (x[:2].lower() == nm[:2].lower() or x[-2:] == nm[-2:]):
            return x
    return _name(rng, 2)


def relate_batch(rng, n, vocab=KEYS_TRAIN):
    """(task, a, b, same?): two sayings of one fact or rule, and the near misses: a name one syllable off, the same name
    with another key, a key with or without an adjective ("old offer" / "offer"); and a fact with its rule or another"""
    out = []
    for _ in range(n):
        nm, k = _name(rng, 2), " ".join(_key(rng, vocab, set()))
        k2 = rng.choice([" ".join(_key(rng, vocab, set())), f"{rng.choice(_ADJ)} {k}"])
        rt = lambda key: rng.choice([f"rule: {key}", rule_text(rng, rng.choice(("add", "set", "keep")), key)])
        c = rng.random()
        if c < .3:
            out.append(("same", views(rng, nm, k), views(rng, nm, k), 1))
        elif c < .4:
            out.append(("same", views(rng, nm, k), views(rng, near(rng, nm), k), 0))
        elif c < .55:
            out.append(("same", views(rng, nm, k), views(rng, nm, k2), 0))
        elif c < .65:
            out.append(("same", rt(k), rt(k), 1))
        elif c < .7:
            out.append(("same", rt(k), rt(k2), 0))
        elif c < .85:
            out.append(("rule", views(rng, nm, k), rt(k), 1))
        else:
            out.append(("rule", views(rng, nm, k), rt(k2), 0))
    return out


@torch.no_grad()
def relate_grid(tr, rng, n=8, vocab=KEYS_TRAIN):
    """the pairs the machine meets: names x keys (some keys with and without an adjective); each fact asks with one saying,
    its other saying is the yes, the fact nearest to it by label (the place the walk would find) is the no"""
    used, ku = set(), set()
    names = [_name(rng, 2, used) for _ in range(n)]
    keys = [" ".join(_key(rng, vocab, ku)) for _ in range(n - n // 4)]
    keys += [f"{rng.choice(_ADJ)} {k}" for k in rng.sample(keys, n // 4)]
    facts = [(a, b) for a in names for b in keys]
    q, other = [views(rng, *f) for f in facts], [views(rng, *f) for f in facts]
    L = tr.encode(other)
    sim = tr.encode(q) @ L.T - 2 * torch.eye(len(facts), device=L.device)
    near = sim.argmax(1).tolist()
    return [("same", a, b, 1) for a, b in zip(q, other)] + [("same", a, other[j], 0) for a, j in zip(q, near)]


def relate_run(tr, relate, batch):
    task, a, b, y = zip(*batch)
    dev = next(relate.parameters()).device
    return relate(list(task), *tr.content(list(a)), *tr.content(list(b))), torch.tensor(y, dtype=torch.float, device=dev)


@torch.no_grad()
def want_spread(tr, bank):
    """how close wants of one kind are in the want space, against wants of other kinds (held phrasings included)"""
    W = [(k, w) for k in WANTS for w in WANTS[k]]
    q = bank.query(tr, [w for _, w in W])
    C = q @ q.T
    same = [float(C[i, j]) for i in range(len(W)) for j in range(len(W)) if i != j and W[i][0] == W[j][0]]
    diff = [float(C[i, j]) for i in range(len(W)) for j in range(len(W)) if W[i][0] != W[j][0]]
    return f"want cos same kind {np.mean(same):.2f} (min {min(same):.2f}), other kinds {np.mean(diff):.2f} (max {max(diff):.2f})"


NOVEL = ["the square root of", "the average of", "the product of", "please water the plants", "What is the capital of France?"]
# tasks no demo shows, beside the held verbs (OTHER_VERBS[1]); a "the X of" want reads like the join kind ("the {key} of")


def bank_report(tr, bank, rng, n=96):
    """per kind of demo, values never seen: which tool its want finds, and whether that tool says the right thing;
    the held wants (never read) the same way; a nested use (a tool's output into another tool)"""
    line, used = [], {}
    for held in (False, True):
        for kind in KINDS:
            if held and not any(x in HELD_WANTS for x in WANTS.get(kind, [])):
                continue
            D = [read_demo(rng, kind, held) for _ in range(n)]
            hit = 0
            for _, w, ins, outs in D:
                t, o, _ = apply_tool(tr, bank, w, ins)
                used.setdefault((kind, held), {}).setdefault(t, 0)
                used[(kind, held)][t] += 1
                hit += 0 if t < 0 else [x.strip() for x in tr.decode(torch.cat(o), torch.ones(len(o), len(o[0][0]), dtype=torch.bool, device=o[0].device), up=bank.nets[t].mouth)] == outs \
                    if len(o) == len(outs) else 0
            top = max(used[(kind, held)].items(), key=lambda z: z[1])
            line.append(f"{kind}{'*' if held else ''} {hit / n:.2f} (tool {top[0]} {top[1] / n:.2f}, new {used[(kind, held)].get(-1, 0) / n:.2f})")
    hit = 0
    for _ in range(n):
        a, b, c = rng.randint(1, 60), rng.randint(1, 60), rng.randint(1, 60)
        t, o, _ = apply_tool(tr, bank, "the larger of", [said(rng, a), said(rng, b)])
        say = lambda z, t: tr.decode(z, torch.ones(z.shape[:2], dtype=torch.bool, device=z.device), up=bank.nets[t].mouth)[0].strip()
        if o:
            t2, o2, _ = apply_tool(tr, bank, "the sum of", [say(o[0], t), said(rng, c)])
            hit += bool(o2) and say(o2[0], t2) == str(max(a, b) + c)
    nothing = bank.gate(bank.query(tr, [asks_nothing(rng) for _ in range(n)])).max(1).values > 1
    novel = bank.gate(bank.query(tr, NOVEL + [asks_nothing(rng, True) for _ in range(n)])).max(1).values > 1
    return (f"tools {len(bank.nets)} | " + " ".join(line) + f" | nested {hit / n:.2f} | {want_spread(tr, bank)} | relevant to some"
            f" label: asks nothing {float(nothing.float().mean()):.2f}, novel tasks {float(novel.float().mean()):.2f}")


def train_tools(A):
    """--init = a translator with its decoder (frozen, so B keeps its meaning). The model reads demos and makes its own
    tool nodes; the relate node learns alongside."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr = Translator().to(dev).eval()
    if A.init:
        tr.load_state_dict(torch.load(A.init, map_location=dev), strict=False)
    tr.requires_grad_(False)
    bank = ToolBank().to(dev)
    rng = random.Random(A.seed)
    bank.query(tr, [read_demo(rng, k)[1] for k in KINDS for _ in range(16)], track=True)
    print("before learning:", want_spread(tr, bank), flush=True)
    st, opt, t0 = {"step": 0, "last": -10 ** 9, "newest": None}, None, time.time()
    for step in range(1, A.steps + 1):
        st.update(step=step, spawned=False)
        loss, route, _ = learn(tr, bank, [read_demo(rng, rng.choice(KINDS)) for _ in range(A.batch)], st, A, none=[asks_nothing(rng) for _ in range(A.batch // 4)])
        loss = (loss + F.binary_cross_entropy_with_logits(*relate_run(tr, bank.relate, relate_batch(rng, A.batch)))
                + F.binary_cross_entropy_with_logits(*relate_run(tr, bank.relate, relate_grid(tr, rng))))
        if opt is None or st["spawned"]:
            opt = torch.optim.AdamW(bank.parameters(), lr=A.lr)
            if st["spawned"]:
                print(f"step {step}: made tool {st['newest']} ({int((route < 0).sum())} of {A.batch} demos no label claimed)", flush=True)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % A.every == 0:
            bank.eval()
            with torch.no_grad():
                rep = bank_report(tr, bank, rng)
                B = relate_batch(rng, 256)
                lg, y = relate_run(tr, bank.relate, B)
                ok = ((lg > 0).float() == y).tolist()
                rel = " ".join(f"{task}{'+' if v else '-'} {sum(o for o, x in zip(ok, B) if x[0] == task and x[3] == v) / max(1, sum(x[0] == task and x[3] == v for x in B)):.2f}"
                               for task in ("same", "rule") for v in (1, 0))
                G = relate_grid(tr, rng, 12)
                lg, y = relate_run(tr, bank.relate, G)
                ok = ((lg > 0).float() == y).tolist()
            print(f"step {step} loss {float(loss):.3f} | {rep} | relate {rel} nearest yes {sum(ok[:len(ok) // 2]) / (len(ok) // 2):.2f} "
                  f"no {sum(ok[len(ok) // 2:]) / (len(ok) // 2):.2f} | {time.time() - t0:.0f}s", flush=True)
            bank.train()
            if A.save:
                bank.save(A.save)


# ====================================================================== the normalizer: any text -> clear statements
# The translator's language side. Any text (sentences, paragraphs, JSON, tables, chat, questions) is rewritten by a
# language model into clear statements, one per line; everything after it (labels, B, relations, the reader) works on
# those. Trained on clear forms; judged on forms it never saw (E1), on hand-written texts sealed from training (E2), and
# end to end: the memory fed only held-out forms must answer as well as fed clear statements (E4).
NOTHING = "(nothing)"
CANON_RULE = {"add": "are amounts to add to the current {k}", "set": "replace the old {k}", "keep": "are kept alongside the earlier {k}"}
RULE_SAID = ({"add": ["whenever a new {k} comes in, add it to the old one", "add each new {k} onto the running {k}"],
              "set": ["a new {k} simply takes the place of the old one", "only the latest {k} counts"],
              "keep": ["remember every {k} value, old ones included", "a new {k} does not erase the earlier ones"]},
             {"add": ["new {k} numbers pile up on top of the {k} we have"], "set": ["throw away the old {k} once a new one arrives"],
              "keep": ["hold on to all the {k} values that come in"]})
CHATTER = (["Thanks for the update!", "Morning everyone.", "Let me know if anything else changes.", "OK, noted.", "Have a good weekend.",
            "Sorry for the delay.", "See the notes below.", "That is all for today.", "Hope you are well.", "Quick update:"],
           ["Cheers, talk soon.", "Nothing new on my side.", "Coffee is in the kitchen.", "Weather looks nice today."])
Q_FORMS = ({"lookup": ["What is {x}?", "what is {x}?", "Tell me {x}.", "{x}?"],
            "before": ["What was {x} before?", "what was {x} before the last change?"],
            "both": ["What is the sum of both values of {x}?", "add both values of {x}"]},
           {"lookup": ["What's {x}?", "Do you know {x}?", "I need {x}."],
            "before": ["What did {x} used to be?", "{x}, before it changed?"],
            "both": ["What do the two values of {x} add up to?"]})
OP_SAID = ({"sum": ["the sum of", "the total of"], "difference between": ["the difference between"], "larger": ["the larger of", "the bigger of"],
            "smaller": ["the smaller of", "the lesser of"]},
           {"sum": ["the combined value of"], "difference between": ["the gap between"], "larger": ["the maximum of"], "smaller": ["the minimum of"]})
LEAF = re.compile(r"\bthe ([a-z]+(?: [a-z]+)?) of ([A-Z][a-z]+)\b")


def show(rng, v, held):
    """a value as written: a copy of another fact said some way"""
    m = re.match(r"(\S+) \(same as the (.+) of ([A-Z][a-z]+)\)$", str(v))
    if not m:
        return str(v)
    x, k, n = m.groups()
    src = say(rng, PHRASE, held, n=n, k=k)
    return rng.choice([f"{x} (same as {src})", f"{x}, same as {src}"] if not held else [f"{x}, copied from {src}", f"{x} (taken from {src})"])


def say_record(rng, rec, held):
    """one record ({name, key: value, ...}) written some way; and its clear statements in the order they were written"""
    nm, kv = rec["name"], [(k, show(rng, v, held)) for k, v in rec.items() if k != "name"]
    rng.shuffle(kv)
    lines = [f"{leaf(nm, k)} is {rec[k]}." for k, _ in kv]
    kw = lambda k: rng.choice([words(k), k]) if not held else words(k)
    if not held:
        f = rng.randrange(5)
        if f == 0:
            return json.dumps({"name": nm, **dict(kv)}), lines
        if f == 1:
            return f"{nm}: " + ", ".join(f"{kw(k)} {v}" for k, v in kv) + ".", lines
        if f == 2:
            return " ".join(say(rng, FACT, n=nm, k=words(k), v=v) for k, v in kv), lines
        if f == 3:
            return f"{nm} has " + " and ".join(f"{kw(k)} {v}" for k, v in kv) + ".", lines
        return nm + "\n" + "\n".join(f"- {kw(k)}: {v}" for k, v in kv), lines
    f = rng.randrange(4)
    if f == 0:
        return "| name | " + " | ".join(words(k) for k, _ in kv) + " |\n|" + "---|" * (len(kv) + 1) + f"\n| {nm} | " + " | ".join(v for _, v in kv) + " |", lines
    if f == 1:
        body = ", and ".join(f"its {words(k)} is {v}" for k, v in kv)
        return f"We looked at {nm}. {body[0].upper()}{body[1:]}.", lines
    if f == 2:
        return "; ".join(f"{words(k)}={v}" for k, v in kv) + f" (for {nm})", lines
    return f"{nm}:\n" + "\n".join(f"  {words(k)}: {v}" for k, v in kv), lines


def say_state(rng, obj, held, ind=0):
    """a JSON state written as JSON, or (held) as an indented outline"""
    if not held:
        return json.dumps(obj)
    if isinstance(obj, dict):
        return "\n".join(f"{'  ' * ind}{words(k)}:" + ("\n" + say_state(rng, v, held, ind + 1) if isinstance(v, (dict, list)) else f" {v}")
                         for k, v in obj.items())
    if isinstance(obj, list):
        return "\n".join(f"{'  ' * ind}- " + (say_state(rng, v, held, ind + 1).strip() if isinstance(v, (dict, list)) else str(v)) for v in obj)
    return f"{'  ' * ind}{obj}"


def say_question(rng, q, held):
    """a question written some way: its frame, its operator words, its leaf phrases"""
    def leafs(t):
        return LEAF.sub(lambda m: say(rng, PHRASE, held, n=m.group(2), k=m.group(1)), t)

    def ops(t):
        for op, ways in OP_SAID[held].items():
            t = re.sub(rf"\bthe {op}(?: of)?\b", lambda _: rng.choice(ways), t)
        return t
    if q.startswith("What was ") and q.endswith(" before?"):
        kind, x = "before", q[9:-8]
    elif q.startswith("What is the sum of both values of "):
        kind, x = "both", q[len("What is the sum of both values of "):-1]
    else:
        kind, x = "lookup", ops(q[8:-1])
    return rng.choice(Q_FORMS[held][kind]).format(x=leafs(x))


def text_and_lines(rng, world, held=False):
    """one piece of input as it might be written, and the clear statements it says (from the world: the truth)"""
    c = rng.random()
    if c < .08:
        return rng.choice(CHATTER[held]), [NOTHING]
    if c < .2:
        x = question(rng.choice(("lookup", "before", "coexist", "chain", "ops")), world, rng)
        if x:
            return say_question(rng, x[0], held), [x[0]]
    if c < .32:
        k = rng.choice(world.num_keys + world.multi_keys)
        kind = rng.choice(("add", "set")) if k in world.num_keys else "keep"
        w = (rule_text(rng, kind, k, held) if rng.random() < .5 else rng.choice(RULE_SAID[held][kind]).format(k=words(k)))
        return w, [f"rule: later {words(k)} values {CANON_RULE[kind].format(k=words(k))}"]
    if c < .42 and world.ents:
        nm = rng.choice(sorted(n for n in world.ents if world.ents[n]))
        k = rng.choice(sorted(world.ents[nm]))
        v = rng.randint(1, 60)
        return say(rng, CHANGE, held, n=nm, k=words(k), v=v), [f"Correction: {leaf(nm, k)} is now {v}."]
    if c < .5:
        o = state_frame(rng)
        return say_state(rng, o, held), state_statements(o)
    obj = world.component(rng)
    if not held and rng.random() < .3:                 # the whole frame as JSON: written in its own order
        body, lines = json.dumps(obj), statements(obj)
    else:
        said = [say_record(rng, r, held) for v in obj.values() for r in v]
        body, lines = "\n".join(t for t, _ in said), [x for _, ls in said for x in ls]
    if rng.random() < .2:
        body = rng.choice(CHATTER[held]) + " " + body
    return body, lines


# E2: hand-written, sealed from training and from tuning. Real names, other domains, other phrasings.
SEALED = [
    ("Maria's salary is 5200 a month.", ["the salary of Maria is 5200 a month."]),
    ("The capital of Japan is Tokyo.", ["the capital of Japan is Tokyo."]),
    ('{"city": "Lyon", "population": 516092, "mayor": "Gregory Doucet"}', ["the population of Lyon is 516092.", "the mayor of Lyon is Gregory Doucet."]),
    ("| product | price | stock |\n|---|---|---|\n| Lamp | 25 | 140 |", ["the price of Lamp is 25.", "the stock of Lamp is 140."]),
    ("Sorry, typo earlier: Ben's age is actually 35.", ["Correction: the age of Ben is now 35."]),
    ("Anna's team is the same as Leo's team, which is Red.", ["the team of Anna is Red (same as the team of Leo)."]),
    ("Going forward, new step counts should be added to the existing step count.", ["rule: later step count values are amounts to add to the current step count"]),
    ("Do you remember what Ben's age is?", ["What is the age of Ben?"]),
    ("What was Maria's salary before?", ["What was the salary of Maria before?"]),
    ("Thanks, have a great weekend!", [NOTHING]),
    ("The derivative of x^2 is 2x.", ["the derivative of x^2 is 2x."]),
    ("In circuit B, R1 has a resistance of 220 ohms.", ["the resistance of R1 is 220 ohms."]),
    ("server:\n  port: 8080\n  host: example.org", ["the server port is 8080.", "the server host is example.org."]),
    ("Quick note: Lena's score is now 88, and Omar's score is 91.", ["Correction: the score of Lena is now 88.", "the score of Omar is 91."]),
    ("Kofi's phone number is 555-0199 and his email is kofi@mail.com.", ["the phone number of Kofi is 555-0199.", "the email of Kofi is kofi@mail.com."]),
    ("Room 12's temperature reads 21.5.", ["the temperature of Room 12 is 21.5."]),
    ("From now on only the latest password counts.", ["rule: later password values replace the old password"]),
    ("Tell me the sum of Lena's score and Omar's score.", ["What is the sum of the score of Lena and the score of Omar?"]),
    ("Which is larger, Lamp's price or Desk's price?", ["What is the larger of the price of Lamp and the price of Desk?"]),
    ("Hamlet was written by Shakespeare.", ["the author of Hamlet is Shakespeare."]),
    ("The speed of light is about 300000 km/s.", ["the speed of light is about 300000 km/s."]),
    ("Pelé's nationality: Brazilian; position: forward.", ["the nationality of Pelé is Brazilian.", "the position of Pelé is forward."]),
    ("We met Ines. Her role is designer and her office is B4.", ["the role of Ines is designer.", "the office of Ines is B4."]),
    ("Update - the deadline of Project X moved to 2026-12-01.", ["Correction: the deadline of Project X is now 2026-12-01."]),
    ("What's the boiling point of ethanol?", ["What is the boiling point of ethanol?"]),
    ('[{"name": "Ada", "rank": 2}, {"name": "Bo", "rank": 7}]', ["the rank of Ada is 2.", "the rank of Bo is 7."]),
    ("The meeting is on Friday. Bring snacks.", ["the day of the meeting is Friday."]),
    ("I think it might rain later, not sure.", [NOTHING]),
    ("Each new reading for the meter should be kept along with the earlier readings.", ["rule: later reading values are kept alongside the earlier reading"]),
    ("Tom's manager is whoever manages Sara, namely Paul.", ["the manager of Tom is Paul (same as the manager of Sara)."]),
]


def canon_set(lines):
    return {re.sub(r"\s+", " ", x.strip().lower().rstrip(".")) for x in lines if x.strip()}


class Normalizer(nn.Module):
    """a language model that rewrites any text into clear statements, one per line. Each token it writes is either
    generated or copied from the text it reads (a learned gate mixes the two), so names, keys and values it never saw
    pass through as written"""

    def __init__(self, hf="gpt2", c=256):
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(hf)
        self.tok.pad_token = self.tok.eos_token
        self.lm = AutoModelForCausalLM.from_pretrained(hf, dtype=torch.float32)   # some ship in bfloat16; the copy layers are float32
        D, self.c = self.lm.config.hidden_size, c
        self.cq, self.ck, self.cg = nn.Linear(D, c), nn.Linear(D, c), nn.Linear(D, 1)   # copy: where to look, and how much

    def src(self, t):
        return self.tok(t[:1500] + "\n###\n")["input_ids"][-400:]

    def loss(self, texts, lines):
        """-log p(target), p = gate * generated + (1 - gate) * copied (attention over the source tokens)"""
        dev, eos = self.lm.device, self.tok.eos_token_id
        src = [self.src(t) for t in texts]
        tgt = [self.tok("\n".join(x))["input_ids"][:300] + [eos] for x in lines]
        L = max(len(a) + len(b) for a, b in zip(src, tgt))
        ids = torch.full((len(texts), L), eos, dtype=torch.long)
        att = torch.zeros(len(texts), L, dtype=torch.long)
        is_src = torch.zeros(len(texts), L, dtype=torch.bool)
        lab = torch.full((len(texts), L), -100, dtype=torch.long)
        for i, (a, b) in enumerate(zip(src, tgt)):
            ids[i, :len(a) + len(b)] = torch.tensor(a + b)
            att[i, :len(a) + len(b)] = 1
            is_src[i, :len(a)] = True
            lab[i, len(a):len(a) + len(b)] = torch.tensor(b)
        ids, att, is_src, lab = ids.to(dev), att.to(dev), is_src.to(dev), lab.to(dev)
        o = self.lm(input_ids=ids, attention_mask=att, output_hidden_states=True)
        H, gold = o.hidden_states[-1], lab[:, 1:]
        h, use = H[:, :-1], gold != -100
        g = torch.sigmoid(self.cg(h)).squeeze(-1)
        p_gen = torch.exp(-F.cross_entropy(o.logits[:, :-1].transpose(1, 2), gold.clamp(min=0), reduction="none"))
        sc = (self.cq(h) @ self.ck(H).transpose(1, 2)) / self.c ** .5
        a = torch.softmax(sc.masked_fill(~is_src[:, None, :], -1e4), -1)
        p_copy = (a * ((ids[:, None, :] == gold[..., None]) & is_src[:, None, :])).sum(-1)
        p = g * p_gen + (1 - g) * p_copy
        return -(torch.log(p + 1e-9) * use).sum() / use.sum()

    def _next(self, h, logits, K, ids, ok):
        """the next token's distribution: generated, or copied from the source"""
        a = torch.softmax(((K @ self.cq(h)[..., None]).squeeze(-1) / self.c ** .5).masked_fill(~ok, -1e4), -1)
        g = torch.sigmoid(self.cg(h))
        return g * torch.softmax(logits, -1) + (1 - g) * torch.zeros_like(logits).scatter_add_(1, ids, a)

    @torch.no_grad()
    def read(self, texts, n=200):
        dev, eos, out = self.lm.device, self.tok.eos_token_id, []
        for i in range(0, len(texts), 16):
            src = [self.src(t) for t in texts[i:i + 16]]
            S = max(map(len, src))
            ids = torch.tensor([[eos] * (S - len(x)) + x for x in src], device=dev)   # left-padded
            att = torch.tensor([[0] * (S - len(x)) + [1] * len(x) for x in src], device=dev)
            pos = (att.cumsum(1) - 1).clamp(min=0)
            o = self.lm(input_ids=ids, attention_mask=att, position_ids=pos, use_cache=True, output_hidden_states=True)
            K, ok, past = self.ck(o.hidden_states[-1]), att.bool(), o.past_key_values
            h, lg, at, done, said = o.hidden_states[-1][:, -1], o.logits[:, -1], pos[:, -1] + 1, torch.zeros(len(src), dtype=torch.bool, device=dev), []
            for _ in range(n):
                t = self._next(h, lg, K, ids, ok).argmax(-1).masked_fill(done, eos)
                said.append(t)
                done |= t == eos
                if done.all():
                    break
                att = torch.cat([att, torch.ones_like(att[:, :1])], 1)
                o = self.lm(input_ids=t[:, None], attention_mask=att, position_ids=at[:, None], past_key_values=past,
                            use_cache=True, output_hidden_states=True)
                past, h, lg, at = o.past_key_values, o.hidden_states[-1][:, -1], o.logits[:, -1], at + 1
            for r in torch.stack(said, 1).tolist():
                r = r[:r.index(eos)] if eos in r else r
                out.append([x for x in self.tok.decode(r).split("\n") if x.strip()])
        return out


def norm_report(nz, rng, n=120):
    """E1: forms never trained on (and trained ones, for comparison); E2: the sealed hand-written texts"""
    nz.eval()
    line = []
    for held in (False, True):
        w = new_world(rng)
        for _ in range(30):
            for draw in world_step(w, rng):
                draw()
        T = [text_and_lines(rng, w, held) for _ in range(n)]
        got = nz.read([t for t, _ in T])
        ok = [canon_set(g) == canon_set(x) for g, (_, x) in zip(got, T)]
        line.append(f"{'E1 held forms' if held else 'trained forms'} {sum(ok) / n:.3f}")
        if held:
            bad = [(t, x, g) for (t, x), g, o in zip(T, got, ok) if not o][:4]
    got = nz.read([t for t, _ in SEALED])
    ok = [canon_set(g) == canon_set(x) for g, (_, x) in zip(got, SEALED)]
    rec = [len(canon_set(g) & canon_set(x)) / len(canon_set(x)) for g, (_, x) in zip(got, SEALED)]
    line.append(f"E2 sealed {sum(ok)}/{len(SEALED)} exact, statements recovered {sum(rec) / len(rec):.3f}")
    nz.train()
    return " | ".join(line), bad, [(t, x, g) for (t, x), g, o in zip(SEALED, got, ok) if not o]


def train_normalizer(A):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    nz = Normalizer(A.norm_hf).to(dev)
    if A.init:
        print("normalizer from", A.init, nz.load_state_dict(torch.load(A.init, map_location=dev), strict=False).missing_keys, flush=True)
    opt = torch.optim.AdamW(nz.parameters(), lr=A.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1, (i + 1) / 200) * max(.02, 1 - i / A.steps))
    rng = random.Random(A.seed)
    w, t0 = new_world(rng), time.time()
    for step in range(1, A.steps + 1):
        if step % 200 == 0:                                # a fresh world now and then: names and keys never repeat for long
            w = new_world(rng)
        for draw in world_step(w, rng):
            draw()
        B = [text_and_lines(rng, w) for _ in range(A.batch)]
        loss = nz.loss([t for t, _ in B], [x for _, x in B])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(nz.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % A.every == 0:
            rep, bad, sealed_bad = norm_report(nz, rng)
            print(f"step {step} loss {float(loss):.3f} | {rep} | {time.time() - t0:.0f}s", flush=True)
            for t, x, g in bad + sealed_bad[:6]:
                print(f"     {t[:150]!r}\n       want {x[:3]}\n       got  {g[:3]}", flush=True)
            if A.save:
                torch.save(nz.state_dict(), A.save)


def norm_e2e(A):
    """E4: the memory (exact stand-in) fed held-out forms only, through the normalizer; against the same world fed
    clear statements. Same seed, same questions."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    nz = Normalizer(A.norm_hf).to(dev).eval()
    nz.load_state_dict(torch.load(A.normw, map_location=dev))
    for fed in ("clear", "held forms"):
        rng = random.Random(A.seed)
        w = new_world(rng)
        m = Machine(Oracle(w), 64)
        m.g.beam = A.beam
        said = right = 0
        for step in range(1, A.steps + 1):
            for draw in world_step(w, rng):
                o = draw()
                if not o:
                    continue
                truth = statements(o)
                if fed == "clear":
                    lines = truth
                else:
                    text = "\n".join(say_record(rng, r, True)[0] for v in o.values() for r in v) if "rules" not in o and "note" not in o else \
                        (rng.choice(RULE_SAID[True][w.policy[next(iter(o["rules"]))]]).format(k=words(next(iter(o["rules"])))) if "rules" in o
                         else say(rng, CHANGE, True, **dict(zip(("k", "n", "v"), re.match(r"Correction: the (.+) of (\w+) is now (\S+)\.", o["note"]).groups()))))
                    lines = nz.read([text])[0]
                    if "rules" in o:
                        k = next(iter(o["rules"]))
                        truth = [f"rule: later {words(k)} values {CANON_RULE[w.policy[k]].format(k=words(k))}"]
                said += len(truth)
                right += len(canon_set(lines) & canon_set(truth))
                for x in lines:
                    if x != NOTHING:
                        try:
                            arrive(m, x)
                        except FAIL:
                            pass
            if step % A.every == 0:
                line = []
                for kind, ops in (("lookup", 0), ("before", 0), ("coexist", 0), ("chain", 0), ("ops", (1, 3))):
                    hit = n = 0
                    for _ in range(A.q):
                        x = question(kind, w, rng, ops or (1, 3))
                        if not x:
                            continue
                        n += 1
                        q = [x[0]] if fed == "clear" else nz.read([say_question(rng, x[0], True)])[0]
                        if (q[0] if q else "") != x[0]:
                            continue                   # the question was not read back exactly: the correct calls cannot run
                        r = Row(x[0])
                        try:
                            x[2](m, r)
                            m.do(r, "say")
                            hit += next((it.value for it in reversed(r.stack) if it.value is not None), None) == x[1]
                        except FAIL:
                            pass
                    line.append(f"{kind} {hit}/{n}")
                print(f"{fed}: step {step} | statements read right {right}/{said} | " + " | ".join(line), flush=True)


# ====================================================================== function nodes
class Budget(Exception):
    pass


class Item:
    """a node on the stack: what it is (B, or the text itself for the exact stand-in), where it came from"""

    def __init__(self, node, T, place=-1, idx=-1, made=None):
        self.node, self.T, self.place, self.idx, self.made, self._t = node, T, place, idx, made, None

    @property
    def text(self):                                    # what the node says: decoded once
        if self._t is None:
            self._t = self.T.say(self.node)
        return self._t

    @property
    def value(self):                                   # teacher and answer check only
        return value_in(self.text)


class Row:
    """one thought: what it is reading (focus) and what it holds (stack)"""

    def __init__(self, text):
        self.src, self.focus, self.stack, self.passes, self.stuck = text, [text], [], 0, False


class Machine:
    """f(nodes) -> nodes. Every step of every row is one call of one function."""

    def __init__(self, translator, d=64, budget=400):
        self.T, self.g, self.t, self.budget, self.flat, self.trace, self.made = translator, Graph(d), 0, budget, False, None, {}
        self.events, self.at, self.subj = None, {}, {}     # watched: every call is an event the rules judge
        self.reader = None                             # (reader node, translator): the model reads queries; None: the stand-in
        self.explore, self.reads = 0., None            # sampling temperature (0: its best reading); the readings made
        self.wrote, self.latest, self.rulekey = {}, {}, {}   # the rules' record: each entry's statement, each subject's last place, each key's rule
        self.right = []                                # ... and tool sayings the rules checked right: (tool, want, inputs, what it said)
        self.chose = {}                                # (place, index) -> the check's decision that made it: (rule, old, new, kept, other)
        self.check_lessons = []                        # decisions shown wrong by exploring (flipped, the truth came out)
        self.check_moved, self.resettled = False, 0    # the check learned this tick; entries re-settled since the last report

    def do(self, r, name, *a):
        r.passes += 1
        if r.passes > self.budget:
            raise Budget()
        if self.trace is not None:
            f = f"apply{a[0]}" if name == "apply" else "derive" if name == "insert" and a else name
            self.trace.append((view(r), f, a[0] if name == "split" else None))
        if self.events is None:
            return getattr(self, "f_" + name)(r, *a)
        ev, self.at = {"row": r, "name": name, "args": a, "src": r.src, "before": list(r.stack)}, {}
        out = getattr(self, "f_" + name)(r, *a)
        ev.update(after=list(r.stack), **self.at)
        self.events.append(ev)
        return out

    def find(self, label, q=None, task="same", look=8):
        """the walk lists places nearest first; the answer is the first whose reach covers the query and that relate
        accepts (q: the asking node), else none: a place far from the query is never it, however relate reads it"""
        c = [(float(np.linalg.norm(self.g.L[self.g.flat(label)] - label)), self.g.flat(label))] if self.flat else self.g.search(label)
        for d, j in c[:look]:
            if j >= 0 and d < self.g.S[j] and self.T.same(label, self.g.L[j], q, self.g.chain[j][0][0], task):
                return j
        return -1

    def item(self, text):
        it = Item(self.T.node(text), self.T)
        it._t = text                                   # what the row read is that text: no round trip through B
        return it

    def entry_item(self, p, idx=-1):
        ch = self.g.chain[p]
        idx = idx % len(ch)
        return Item(ch[idx][0], self.T, p, idx)

    def f_split(self, r, n):                           # text -> chunk, rest
        c, r.focus[-1] = r.focus[-1][:n].strip(), r.focus[-1][n:].lstrip()   # no stray spaces: a chunk ends on a token
        if not r.focus[-1]:
            r.focus.pop()
        r.stack.append(self.item(c))

    def f_drop(self, r):
        r.stack.pop()

    def f_nip(self, r):
        del r.stack[-2]

    def f_say(self, r):                                # the row is done; its answer is what it holds
        pass

    def f_lookup(self, r):                             # phrase -> newest entry of its place
        it = r.stack.pop()
        p = self.find(self.T.label(it.text), it.node)
        self.at = {"place": p, "asked": it.text}
        r.stack.append(self.entry_item(p) if p >= 0 else self.item("unknown"))

    def f_rule(self, r):                               # statement -> the rule that governs it
        q = self.T.rule_query(r.src)
        p = self.find(q, self.T.node(r.src), "rule") if q is not None else -1
        self.at = {"place": p}
        r.stack.append(self.entry_item(p) if p >= 0 else self.item("no rule"))

    def f_prev(self, r):                               # entry -> the entry it came from
        it = r.stack[-1]
        r.stack.append(self.entry_item(it.place, it.idx - 1) if it.place >= 0 and it.idx > 0 else self.item("unknown"))

    def f_apply(self, r, k):                           # [want, k nodes] -> what the tool node the want finds gives back
        want, ins = r.stack[-k - 1], r.stack[-k:]
        del r.stack[-k - 1:]
        node = (lambda it: it.text) if isinstance(self.T, Oracle) else (lambda it: it.node)
        args = [node(x) for x in ins]
        for said in self.T.apply(want.text, args):     # what the tool says, held as read: no second trip through B
            it = self.item(said)
            it.made = (getattr(self.T, "last", None), want.text, args)
            r.stack.append(it)

    def f_insert(self, r, derived=False):              # A to be B: a new entry where the statement belongs
        self.t += 1
        said = self.T.node(r.src)
        node = r.stack[-1].node if derived else said   # derived: the tool's output node itself
        label = self.T.label(r.src)                    # a statement lands where its phrase does: the labels are trained so
        p = self.find(label, said)
        self.at = {"place": p if p >= 0 else self.g.n, "new": p < 0, "derived": derived}
        p = p if p >= 0 else self.g.add_place(label)
        self.g.chain[p].append((node, self.t))
        if derived and r.stack[-1].made and r.stack[-1].made[0] is not None:
            self.made[(p, len(self.g.chain[p]) - 1)] = r.stack[-1].made   # (tool, want, inputs): to run again if the tool changes
        self.g.R[p] = self.T.relation(r.src)           # what it depends on now (said directly: nothing)
        if self.events is not None:
            self.at.update(index=len(self.g.chain[p]) - 1, claims=set(np.nonzero(self.g.L[:self.g.n] @ self.g.R[p] > 1)[0].tolist()) - {p})
        self.changed(p)

    def rerun(self, tools):
        """these tool nodes changed: every entry one of them made is made again; a newest entry that changes is a change"""
        n = 0
        for (p, i), (t, want, args) in self.made.items():
            if t in tools:
                self.g.chain[p][i] = (self.T.node(self.T.apply(want, args, tool=t)[0]), self.g.chain[p][i][1])
                n += 1
                if i == len(self.g.chain[p]) - 1:
                    self.changed(p)
        return n

    def changed(self, p, depth=0):
        """a node changed: every node whose relation claims it runs again what made it; for a copy that is taking the
        new value. That is a change too, so it goes on (a cycle stops at depth 8)."""
        if depth >= 8:
            return
        for j in np.nonzero(self.g.R[:self.g.n] @ self.g.L[p] > 1)[0]:
            if j != p:
                self.t += 1
                self.g.chain[j].append((self.g.chain[p][-1][0], self.t))
                if self.events is not None:
                    self.events.append({"row": None, "name": "trigger", "place": int(j), "source": p,
                                        "index": len(self.g.chain[j]) - 1, "from": len(self.g.chain[p]) - 1})
                self.changed(int(j), depth + 1)


# ====================================================================== the rules: local judgments that drive everything
# The only hand-written judgment in GNM. Every call the machine makes is an event; each rule looks at one event (or a
# row's path) and says whether it went as it should, which part is to blame, and - when it knows - what should have
# happened. Learning takes these judgments and nothing else; when the model misbehaves, a rule or the node design is
# what changes. Rules read clear statements, so a statement's subject is plain to see.
def subject(text):
    """what a clear statement (or phrase) is about; None when it does not say"""
    t = text.strip()
    m = re.match(r"(?:Correction: )?(the .+?) is (?:now )?\S", t)
    if m:
        return m.group(1).lower()
    m = re.match(r"rule: later (.+?) values ", t)
    if m:
        return f"rule {m.group(1)}"
    return t.lower() if re.fullmatch(r"the .+ of .+", t) else None


class Judgment:
    def __init__(self, rule, ok, blame, target=None, ev=None, note=""):
        self.rule, self.ok, self.blame, self.target, self.ev, self.note = rule, ok, blame, target, ev, note


def rule_merge(m, ev):
    """a statement goes where its subject is: into a place about something else is confusion (relate let it in); a new
    place though its subject already had one is a split (relate turned it away). target: (text, place, same?)"""
    if ev["name"] != "insert" or ev.get("derived"):
        return
    b = subject(ev["src"])
    if not b:
        return
    if ev.get("new"):
        q = ev.get("known")                            # where this subject already was, before this insert
        yield Judgment("merge", q is None, "relate", (ev["src"], q, 1.), ev, f"{b} split from place {q}")
    else:
        a = subject(m.T.say(m.g.chain[ev["place"]][0][0]))
        if a:
            yield Judgment("merge", a == b, "relate", (ev["src"], ev["place"], float(a == b)), ev, f"{b} into {a}")


def rule_found(m, ev):
    """a lookup answers about what was asked; it says unknown only when nothing about it was stored"""
    if ev["name"] != "lookup" or not subject(ev["asked"]):
        return
    want = subject(ev["asked"])
    if ev["place"] >= 0:
        got = subject(m.T.say(m.g.chain[ev["place"]][0][0]))
        if got:
            yield Judgment("found", got == want, "relate", (ev["asked"], ev["place"], float(got == want)), ev, f"asked {want}, got {got}")
    else:
        yield Judgment("found", want not in m.subj, "walk", (ev["asked"], m.subj.get(want), 1.), ev, f"asked {want}, unknown")


def rule_copy(m, ev):
    """a statement that says it is a copy of Y must have a relation that claims Y's place, and nothing else"""
    if ev["name"] != "insert" or ev.get("derived"):
        return
    y = re.search(r"\(same as (the .+?)\)", ev["src"])
    claims = ev["claims"]                              # as the insert left it (a later restatement resets it)
    if y and y.group(1).lower() in m.subj:
        q = m.subj[y.group(1).lower()]
        yield Judgment("copy", claims == {q}, "relation", (ev["src"], q), ev, f"claims {sorted(claims)[:3]}, source {q}")
    elif not y:
        yield Judgment("copy", not claims, "relation", (ev["src"], None), ev, f"claims {sorted(claims)[:3]}, says nothing")


def rule_roundtrip(m, ev):
    """a node made from a statement says that statement back"""
    if ev["name"] == "insert" and not ev.get("derived"):
        got = m.T.say(m.g.chain[ev["place"]][ev["index"]][0])   # the entry this insert wrote
        yield Judgment("roundtrip", got.strip() == ev["src"].strip(), "decoder", ev["src"], ev, f"said back {got[:60]!r}")


def rule_tool(m, ev):
    """a tool gives the kind of node its inputs promise: numbers in, a number out; an entry naming something, a phrase"""
    if ev["name"] != "apply":
        return
    k = ev["args"][0]
    ins, out = ev["before"][-k:], ev["after"][len(ev["before"]) - k - 1:]
    if not out:
        yield Judgment("tool", False, "tool", None, ev, "nothing out")
        return
    said = out[0].text.strip()
    if all(isinstance(x.value, int) for x in ins):
        yield Judgment("tool", number(said) is not None, "tool", None, ev, f"numbers in, {said[:40]!r} out")
    elif k == 1:
        yield Judgment("tool", bool(re.fullmatch(r"the .+ of \S.*", said)), "tool", None, ev, f"entry in, {said[:40]!r} out")


OPS = [(re.compile(r"\bsum\b|\badd"), lambda a, b: a + b), (re.compile(r"\bdifference\b"), lambda a, b: a - b),
       (re.compile(r"\blarger\b"), max), (re.compile(r"\bsmaller\b"), min)]


def rule_arith(m, ev):
    """a tool asked for a sum / difference / larger / smaller of two numbers gives exactly that (the teacher checks the
    arithmetic; it names the tool's mistake on the step where it happens)"""
    if ev["name"] != "apply" or ev["args"][0] != 2:
        return
    want, ins, out = ev["before"][-3].text, ev["before"][-2:], ev["after"][len(ev["before"]) - 3:]
    f = next((f for r, f in OPS if r.search(want.lower()) or any(x.search(want) for x in ADD_RE) and f is OPS[0][1]), None)
    x, y = (value_in(i.text) for i in ins)
    if f and isinstance(x, int) and isinstance(y, int) and out:
        yield Judgment("arith", number(out[0].text.strip()) == f(x, y), "tool", str(f(x, y)), ev, f"{want[:30]}: {x}, {y} -> {out[0].text.strip()[:20]}, should be {f(x, y)}")


def rule_read(m, ev):
    """an entry read back says the statement it was written from"""
    if ev["name"] == "lookup" and ev["place"] >= 0:
        it = ev["after"][-1]
        src = m.wrote.get((it.place, it.idx))
        if src:
            yield Judgment("read", it.text.strip() == src.strip(), "decoder", src, ev, f"read {it.text[:50]!r}")


def rule_stale(m, ev):
    """a lookup reaches the place where its subject was said last (not an older copy of it elsewhere)"""
    if ev["name"] == "lookup" and ev["place"] >= 0 and subject(ev["asked"]) in m.latest:
        q = m.latest[subject(ev["asked"])]
        yield Judgment("stale", ev["place"] == q, "relate", (ev["asked"], q, 1.), ev, f"reached place {ev['place']}, last said at {q}")


def teacher(m):
    """the exact reading of a text (the rules may use it; no function the model calls does)"""
    return m.T if isinstance(m.T, Oracle) else m.T.o


def rule_rule(m, ev):
    """a statement's rule lookup finds the rule stated for its key, and says no rule only when none was stated"""
    if ev["name"] != "rule":
        return
    k = teacher(m).key(ev["src"])
    if k[0] != "fact":
        return
    want = m.rulekey.get(k[1], -1)
    teach = ([(ev["src"], want, 1., "rule")] if want >= 0 else []) + ([(ev["src"], ev["place"], 0., "rule")] if ev["place"] >= 0 and ev["place"] != want else [])
    yield Judgment("rule", ev["place"] == want, "relate", teach, ev, f"found place {ev['place']}, rule for {k[1]} is at {want}")


def rule_settle(m, ev):
    """measurement only (no lesson): the state arrive kept is the one the teacher would keep (old + new under the
    rule of its key that adds, for a plain fact; the statement as said otherwise)"""
    if ev["name"] != "insert" or ev.get("place") is None:
        return
    d = m.chose.get((ev["place"], ev.get("index")))
    if not d:
        return
    g, o, n, keep, other = d
    t = teacher(m)
    gk, nk = t.key(g.text), t.key(n.text)
    adds = (gk[0] == "rule" and nk[0] == "fact" and gk[1] == nk[1] and any(x.search(g.text) for x in ADD_RE)
            and not n.text.startswith("Correction") and "(same as" not in n.text)
    yield Judgment("settle", (keep is not n) == adds, "check", None, ev, f"kept {'made' if keep is not n else 'as said'} ({keep.text[:20]!r}) for {n.text[:40]!r} under {g.text[:40]!r}")


RULES = [rule_merge, rule_found, rule_copy, rule_roundtrip, rule_tool, rule_arith, rule_read, rule_stale, rule_rule, rule_settle]


class Ledger:
    """what the rules said: per rule, how many checks and how many went wrong, with the first few cases"""

    def __init__(self):
        self.n, self.bad, self.cases, self.why = {}, {}, {}, {}

    def add(self, j):
        self.n[j.rule] = self.n.get(j.rule, 0) + 1
        if not j.ok:
            self.bad[j.rule] = self.bad.get(j.rule, 0) + 1
            self.cases.setdefault(j.rule, [])
            if len(self.cases[j.rule]) < 3:
                self.cases[j.rule].append(f"{j.ev['src'][:70]!r}: {j.note}" if j.ev else j.note)

    def line(self):
        why = ", ".join(f"{k} {v}" for k, v in sorted(self.why.items(), key=lambda z: -z[1]))
        return "rules " + ", ".join(f"{r} {self.bad.get(r, 0)}/{n} wrong" for r, n in sorted(self.n.items())) + (f" | wrong answers, first failure on the row: {why}" if why else "")


def judge(m, ledger, row=None, truth=None):
    """run every rule on the events so far (and the row's outcome when its truth is known); keeps the subject index"""
    out = []
    for ev in m.events or []:
        if ev["name"] == "insert" and not ev.get("derived") and subject(ev["src"]):
            ev["known"] = m.subj.get(subject(ev["src"])) if ev.get("new") else None
            if ev.get("new"):
                m.subj.setdefault(subject(ev["src"]), ev["place"])
            m.wrote[(ev["place"], ev["index"])] = ev["src"]
            m.latest[subject(ev["src"])] = ev["place"]
        if ev["name"] == "insert" and not ev.get("derived") and teacher(m).key(ev["src"])[0] == "rule":
            m.rulekey[teacher(m).key(ev["src"])[1]] = ev["place"]
        if ev["name"] == "trigger" and (ev["source"], ev["from"]) in m.wrote:
            m.wrote[(ev["place"], ev["index"])] = m.wrote[(ev["source"], ev["from"])]
        for rule in RULES:
            for j in rule(m, ev):
                ledger.add(j)
                out.append(j)
                made = j.ok and j.rule == "arith" and j.ev["after"][len(j.ev["before"]) - 3].made
                if made and made[0] is not None:
                    m.right = (m.right + [(*made, j.target)])[-256:]
    if row is not None and truth is not None:
        got = next((it.value for it in reversed(row.stack) if it.value is not None), None)
        first = next((j.rule for j in out if j.ev and j.ev.get("row") is row and not j.ok), "no rule fired")
        reads = [(ev["after"][-1].place, ev["after"][-1].idx) for ev in m.events or [] if ev.get("row") is row and ev["name"] == "lookup"]
        j = Judgment("outcome", got == truth, "path", truth, {"src": row.src, "reads": reads},
                     f"said {got}, truth {truth}, first failure: {first}")
        ledger.add(j)
        if not j.ok:
            ledger.why[first] = ledger.why.get(first, 0) + 1
        out.append(j)
    if m.events is not None:
        m.events.clear()
    return out


def nodes_of(zs, dev):
    """a list of nodes (L_i x c) as one left-padded batch"""
    L = max(len(z) for z in zs)
    z = torch.stack([torch.cat([z.new_zeros(L - len(z), z.shape[-1]), z]) for z in zs]).to(dev)
    m = torch.stack([torch.cat([torch.zeros(L - len(x), dtype=torch.bool), torch.ones(len(x), dtype=torch.bool)]) for x in zs]).to(dev)
    return z, m


def lessons(tr, bank, m, js, cap=64):
    """every judgment that knows what should have happened becomes a lesson for the part it blames, and nothing else
    teaches: relate learns same / not same, the relation learns which words name a statement's source and whether it
    has one. (Labels and B stay as they are online: changing them would move what is already stored.)"""
    dev, loss = next(bank.parameters()).device, None
    seen = []                                          # the check learns from outcome: a one-read question read an entry
    for j in js:                                       # a check decision made: right -> the decision stands; wrong while
        if j.rule == "outcome" and len(j.ev.get("reads", [])) == 1 and j.ev["reads"][0] in m.chose:   # the other
            g, o, n, keep, other = m.chose[j.ev["reads"][0]]                                          # candidate held
            if j.ok:                                                                                  # the truth ->
                seen += [(g, o, n, keep, 1.), (g, o, n, other, 0.)]                                   # the reverse
            elif other.value == j.target:
                seen += [(g, o, n, keep, 0.), (g, o, n, other, 1.)]
    seen += m.check_lessons
    m.check_lessons = []
    m.check_moved = bool(seen)                         # the check learns now: what it decided before is decided again
    if seen:
        lg = bank.check([nodes_of([x[k].node.detach() for x in seen], dev) for k in range(4)])
        loss = F.binary_cross_entropy_with_logits(lg, torch.tensor([x[4] for x in seen], device=dev))
    teach = lambda j: [x if len(x) == 4 else (*x, "same") for x in (j.target if isinstance(j.target, list) else [j.target])
                       if x and x[1] is not None and x[1] >= 0]       # (text, place, same?, which question)
    for j in js:                                       # reach moves at once: one lesson, one step
        if j.blame in ("relate", "walk") and not j.ok:
            for text, p, y, task in teach(j):
                q = m.T.rule_query(text) if task == "rule" else m.T.label(text)
                d = float(np.linalg.norm(m.g.L[p] - q))
                if y and d >= m.g.S[p]:
                    m.g.S[p] = d + .02                 # it should have been reached: reach it now
                elif not y and d < m.g.S[p]:
                    m.g.S[p] = max(d - .02, .05)       # it should not have been: no longer
    ruled = [x for j in js if j.rule == "rule" and not j.ok for x in teach(j) if x[2] == 1.][:cap]
    if ruled:                                          # the rule query is pulled toward the rule's place (slow)
        q = tr.rule_of(tr.encode([t for t, _, _, _ in ruled]))
        tgt = torch.tensor(np.stack([m.g.L[p] for _, p, _, _ in ruled]), device=q.device)
        l0 = (1 - (q * tgt).sum(-1)).mean()
        loss = l0 if loss is None else loss + l0
    pairs = [j for j in js if j.blame in ("relate", "walk") and teach(j)]
    wrong, right = [j for j in pairs if not j.ok], [j for j in pairs if j.ok]
    rel = [x for j in wrong[:cap // 2] + random.sample(right, min(len(right), len(wrong[:cap // 2]))) for x in teach(j)]   # learn from mistakes
    if rel:
        za, ma = tr.content([t for t, _, _, _ in rel])
        zb, mb = nodes_of([m.g.chain[p][0][0].detach() for _, p, _, _ in rel], dev)
        y = torch.tensor([v for _, _, v, _ in rel], device=dev)
        l1 = F.binary_cross_entropy_with_logits(bank.relate([q for _, _, _, q in rel], za, ma, zb, mb), y)
        loss = l1 if loss is None else loss + l1
    sums = [(*j.ev["after"][len(j.ev["before"]) - 3].made, j.target) for j in js if j.rule == "arith" and not j.ok
            and j.ev["after"] and j.ev["after"][len(j.ev["before"]) - 3].made][:cap]
    sums = [x for x in sums if x[0] is not None]
    for t in {x[0] for x in sums}:                     # the tool that got it wrong learns the right number on these very
        wrong = [x for x in sums if x[0] == t]         # inputs, beside as many of its own right sayings (mistakes alone,
        kept = [x for x in m.right if x[0] == t]       # mostly out of its range: probe 153 -> 118 / 160 in 20 ticks)
        for _, want, args, text in wrong + random.sample(kept, min(len(kept), len(wrong))):
            o, _, _ = bank.nets[t]([tr.content([want])] + [(z[None], torch.ones(1, len(z), dtype=torch.bool, device=z.device)) for z in args])
            l4 = tr.decode_loss(o[0][0], o[0][1], [text], up=bank.nets[t].mouth) / len(sums)
            loss = l4 if loss is None else loss + l4
    back = [j.target for j in js if j.rule == "roundtrip" and not j.ok][:cap]
    if back:                                           # the decoder's output side (stored B keeps its meaning)
        l3 = tr.decode_loss(*tr.content(back), back)
        loss = l3 if loss is None else loss + l3
    dep = [j for j in js if j.rule == "copy"][:cap]
    if dep:
        texts = [j.ev["src"] for j in dep]
        ra, rg, roff, rmk = tr.relation(texts, parts=True)
        names = torch.zeros_like(ra)
        for i, t in enumerate(texts):
            y = re.search(r"\(same as (the .+?)\)", t)
            if y:
                names[i] = ((roff[i, :, 1] > y.start(1)) & (roff[i, :, 0] < y.end(1))).float()
        has = torch.tensor([float(bool(re.search(r"\(same as ", t))) for t in texts], device=dev)
        l2 = (F.binary_cross_entropy_with_logits(ra, names, reduction="none") * rmk).sum() / rmk.sum() + F.binary_cross_entropy_with_logits(rg, has)
        loss = l2 if loss is None else loss + l2
    return loss


# ====================================================================== --gold: the correct calls, written by hand
ADD_RE = [re.compile(v.format(k=".*")) for v in RULE_PARTS["add"]]


def gold_read(m, r, phrase):
    m.do(r, "split", len(phrase))
    m.do(r, "lookup")                                  # a copy is already current: its relation keeps it so


def gold_older(m, r):
    """step back until the value differs (a restated value is not a change)"""
    cur = r.stack[-1].value
    while True:
        m.do(r, "prev")
        if r.stack[-1].value != cur or r.stack[-1].text == "unknown":
            return
        m.do(r, "nip")


# ====================================================================== B2: answering is recursion (no per-question procedure)
SLOT = re.compile(r"<(\d+)>")                          # a node sitting in a query: "What was <0> before?"
KEYW = r"(?:(?!(?:of|the|and)\b)[a-z]+ )*?(?!(?:of|the|and)\b)[a-z]+"   # a key: words, none of them of / the / and
OP_WANT = {t[0]: t[2].split("{0}")[0] for t in TOOLS}  # "the sum of " ...


def inner_exact(S, items):
    """the exact stand-in's reading of a query (S: text with node slots): the innermost part that can be answered
    now, as (start, end, function, want, slots); None when nothing is left to answer. The model's own reading (a span
    head + the functions' labels) replaces this; --gold uses it to test the design around it."""
    S = re.sub(r"\((<\d+>)\)", r"\1", S)               # brackets around one node are spent
    val = lambda k: items[int(k)].value
    for name, w in OP_WANT.items():                    # an operation whose two inputs are both nodes already
        for x in re.finditer(re.escape(w) + r"<(\d+)> and <(\d+)>", S):
            return x.start(), x.end(), "apply", w.strip(), [x.group(1), x.group(2)]
    x = re.search(r"both values of <(\d+)>", S)
    if x:
        return x.start(), x.end(), "both", "", [x.group(1)]
    x = re.search(r"What was <(\d+)> before\?", S)
    if x:
        return x.start(), x.end(), "before", "", [x.group(1)]
    x = re.search(rf"the ({KEYW}) of <(\d+)>", S)
    if x and not isinstance(val(x.group(2)), int):     # the K of <an entry naming something>: a new phrase
        return x.start(), x.end(), "apply", f"the {x.group(1)} of", [x.group(2)]
    x = None
    for y in re.finditer(rf"the ({KEYW}) of ([A-Z][a-z]+)", S):
        x = y                                          # the last phrase is the innermost (its name ends it)
    if x:
        return x.start(), x.end(), "lookup", x.group(0), []
    return None


def answer(m, r, S, items, depth=0):
    """answer a query: the innermost answerable part is answered first and its answer node takes its place in the
    query; that is the whole step order. Ends when no part is left to answer; the answer is the last node standing."""
    if depth > 24:
        raise Budget()
    x = read_query(m, S, items) if m.reader else inner_exact(S, items)
    if x is None:
        k = [int(y) for y in SLOT.findall(S)]
        return items[k[-1]] if k else None
    a, b, f, want, slots = x
    S = re.sub(r"\((<\d+>)\)", r"\1", S)
    args = [items[int(k)] for k in slots]
    if f == "lookup":
        r.stack.append(m.item(want))
        m.do(r, "lookup")
        out = [r.stack[-1]]
    elif f == "apply":
        r.stack += [m.item(want)] + args
        m.do(r, "apply", len(args))
        out = r.stack[-1:] if len(args) == 1 else [r.stack[-1]]
    else:                                              # before / both: step back until the value differs
        r.stack.append(args[0])
        gold_older(m, r)
        out = [r.stack[-1]] if f == "before" else [r.stack[-1], args[0]]
    if f == "apply" and len(args) == 1:                # a new phrase is itself a query: read again as text
        new = out[0].text
    else:
        new = " and ".join(f"<{len(items) + i}>" for i in range(len(out)))
        items = items + out
    return answer(m, r, S[:a] + new + S[b:], items, depth + 1)


class QueryReader(nn.Module):
    """the reader function node: reads a query as one latent sequence (its words' B, and the B of every node already
    answered where its slot is) and says where the innermost part it can answer now starts and ends, or that nothing
    is left (position 0); the want there picks the function by label"""
    FN = ("lookup", "apply", "before", "both")

    def __init__(self, c=256, d=128, L=512):
        super().__init__()
        self.none = nn.Parameter(torch.randn(c) * .02)
        self.kind = nn.Parameter(torch.randn(2, c) * .02)                  # a word / part of a node
        self.pos = nn.Parameter(torch.randn(L, c) * .02)
        self.f = nn.TransformerEncoder(nn.TransformerEncoderLayer(c, 8, 1024, batch_first=True, activation="gelu"), 3)
        self.start, self.want = nn.Linear(c, 1), nn.Linear(c, d)
        self.endq, self.endk = nn.Linear(c, c), nn.Linear(c, c)   # the end is read from where the part starts (rc2:
        # an end chosen alone ran past the closing bracket: 'the larger of <2> and <1>) and <0>')
        self.keys = nn.Parameter(F.normalize(torch.randn(len(self.FN), d), dim=-1) / .9)   # each function's label

    def units(self, tr, S, items):
        """the query as units: (char start, char end, B rows) per word token and per node slot"""
        out, rows, at = [], [], 0
        for i, piece in enumerate(re.split(r"(<\d+>)", S)):
            if not piece:
                continue
            if SLOT.fullmatch(piece):
                z = items[int(piece[1:-1])].node
                if not torch.is_tensor(z):             # the exact stand-in holds text: read it (node <-> text)
                    z, mk = tr.content([items[int(piece[1:-1])].text])
                    z = z[0][mk[0]]
                out.append((at, at + len(piece), len(rows), len(rows) + len(z) - 1, 1))
                rows += list(z)
            else:
                z, m = tr.content([piece])
                z = z[0][m[0]]
                offs = tr.tok(piece, return_offsets_mapping=True)["offset_mapping"]
                for (a, b), v in zip(offs, z):        # a word starts at its first letter, not its leading space
                    a += len(piece[a:b]) - len(piece[a:b].lstrip())
                    out.append((at + a, at + b, len(rows), len(rows), 0))
                    rows.append(v)
            at += len(piece)
        return out, torch.stack(rows)

    def forward(self, tr, S, items):
        u, z = self.units(tr, S, items)
        kind = torch.tensor(sum([[k] * (e - s_ + 1) for _, _, s_, e, k in u], []), device=z.device)
        x = torch.cat([self.none[None], z + self.kind[kind] + self.pos[:len(z)]])[None]
        y = self.f(x)[0]
        first = torch.tensor([0] + [s_ + 1 for _, _, s_, _, _ in u], device=z.device)   # where a unit can start / end
        last = torch.tensor([0] + [e + 1 for _, _, _, e, _ in u], device=z.device)
        return u, y, self.start(y).squeeze(-1)[first], (first, last)

    def ends(self, y, fl, i):
        """where the part that starts at unit i ends (units before it ruled out; i = 0: nothing, ends at 0)"""
        first, last = fl
        if i == 0:
            return torch.cat([y.new_zeros(1), y.new_full((len(first) - 1,), -1e9)])
        e = (self.endq(y[last]) * self.endk(y[first[i]])).sum(-1) / y.shape[-1] ** .5
        return e.masked_fill(torch.arange(len(e), device=e.device) < i, -1e9)

    def fn(self, y, i, j):
        """the want over units i..j -> relevance to each function's label"""
        return F.normalize(self.want(y[i:j + 1].mean(0)), dim=-1) @ self.keys.T


def reading_diff(m, q):
    """walk a question by the stand-in's reading; the first step where the model reads it otherwise: (query, the
    stand-in's (span text, function), the model's), or None when they agree all the way"""
    S, items, r = q, [], Row(q)
    for _ in range(24):
        ex = inner_exact(S, items)
        try:
            mo = read_query(m, S, items)
        except FAIL:
            mo = "fail"
        S2 = re.sub(r"\((<\d+>)\)", r"\1", S)
        show = lambda x: None if x is None else x if isinstance(x, str) else (S2[x[0]:x[1]], x[2])
        if show(ex) != show(mo):
            return S2, show(ex), show(mo)
        if ex is None:
            return None
        rd, m.reader = m.reader, None                  # one step by the stand-in's reading
        try:
            a, b, f, want, slots = ex
            args = [items[int(k)] for k in slots]
            if f == "lookup":
                r.stack.append(m.item(want)); m.do(r, "lookup"); out = [r.stack[-1]]
            elif f == "apply":
                r.stack += [m.item(want)] + args; m.do(r, "apply", len(args)); out = [r.stack[-1]]
            else:
                r.stack.append(args[0]); gold_older(m, r)
                out = [r.stack[-1]] if f == "before" else [r.stack[-1], args[0]]
        except FAIL:
            return None
        finally:
            m.reader = rd
        if f == "apply" and len(args) == 1:
            new = out[0].text
        else:
            new = " and ".join(f"<{len(items) + i}>" for i in range(len(out)))
            items = items + out
        S = S2[:a] + new + S2[b:]
    return None


def reader_targets(u, x):
    """the stand-in's reading as unit targets: (start unit, end unit, function) with 0 = nothing left"""
    if x is None:
        return 0, 0, -1
    a, b, f = x[0], x[1], QueryReader.FN.index(x[2])
    i = next(k for k, t in enumerate(u) if t[0] >= a and t[1] > a)
    j = max(k for k, t in enumerate(u) if t[1] <= b)
    return i + 1, j + 1, f


def read_query(m, S, items):
    """the model's reading of a query (in place of inner_exact): (start, end, function, want, slots) or None"""
    rd, tr = m.reader
    S = re.sub(r"\((<\d+>)\)", r"\1", S)
    pick = (lambda v: int(torch.multinomial(F.softmax(v / m.explore, -1), 1))) if m.explore else (lambda v: int(v.argmax()))
    with torch.no_grad():                              # exploring: the reading is sampled (temperature m.explore)
        u, y, st, fl = rd(tr, S, items)
        i = pick(st)
        j = pick(rd.ends(y, fl, i)) if i else 0
        fi = pick(rd.fn(y, u[i - 1][2] + 1, u[j - 1][3] + 1) / .1) if i else -1
    if m.reads is not None:
        m.reads.append((S, list(items), i, j, fi))
    if i == 0:
        return None
    f = rd.FN[fi]
    a, b = u[i - 1][0], u[j - 1][1]
    span = S[a:b]
    slots = SLOT.findall(span)
    want = span.split("<")[0].strip() if slots else span.strip()
    return a, b, f, want, slots


def reader_demos(rng, n):
    """the light supervision: the stand-in's readings of n questions, every step (query, its nodes' texts, reading)"""
    w = new_world(rng)
    m = Machine(Oracle(w), 64)
    m.flat = True
    out = []
    for _ in range(60):
        for draw in world_step(w, rng):
            o = draw()
            for s_ in statements(o) if o else []:
                arrive(m, s_)
    kinds = ["lookup", "before", "coexist", "chain", "ops", "ops"]
    while len(out) < n:
        k = kinds[len(out) % len(kinds)]
        x = question(k, w, rng, (1, 3))
        if not x:
            continue
        steps, real = [], answer

        def take(m_, r_, S_, items_, depth=0):
            steps.append((S_, [it.text for it in items_], inner_exact(S_, items_)))
            return real(m_, r_, S_, items_, depth)
        globals()["answer"] = take
        try:
            answer(m, Row(x[0]), x[0], [])
        except FAIL:
            steps = []
        finally:
            globals()["answer"] = real
        if steps:
            out.append(steps)
    return out


def train_recursion_reader(A):
    """--recur: the reader node learns from ~50 questions read by the stand-in (every step), then is tested on fresh
    worlds: answering by recursion with the model's reading (the functions themselves exact here)"""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr = Translator().to(dev).eval()
    if A.init:
        tr.load_state_dict(torch.load(A.init, map_location=dev), strict=False)
    tr.requires_grad_(False)
    rd = QueryReader().to(dev)
    if A.qreader:                                      # go on from a reader that already learned
        rd.load_state_dict(torch.load(A.qreader, map_location=dev), strict=False)
    rng = random.Random(A.seed)
    demos = [st for q in reader_demos(rng, A.demos) for st in q]
    print(f"{A.demos} questions read by the stand-in: {len(demos)} steps", flush=True)
    node = lambda t: tr.content([t])[0][0][tr.content([t])[1][0]]
    opt = torch.optim.AdamW(rd.parameters(), lr=A.lr)
    t0 = time.time()
    lived = None                                       # the world the outcome phase asks about (filled once)
    for step in range(1, A.steps + 1):
        loss = 0
        if step > A.steps // 4:                        # after the ~50 demos: outcome. N readings of one fresh
            if lived is None:                          # question, sampled; the one that answers right with the
                w2 = new_world(random.Random(A.seed + 7))    # fewest calls is what to read next time
                lived = Machine(Oracle(w2), 64)
                lived.flat, lived.reader = True, (rd, tr)
                for _ in range(60):
                    for draw in world_step(w2, rng):
                        o = draw()
                        for s_ in statements(o) if o else []:
                            arrive(lived, s_)
            rd.eval()
            got = []
            for _ in range(4):
                x = question(rng.choice(["ops", "ops", "chain", "coexist", "before"]), w2, rng, rng.choice([(1, 3), (2, 5), (4, 6)]))
                if not x:
                    continue
                best = None
                for n_ in range(8):
                    lived.explore, lived.reads = (0. if n_ == 0 else 1.), []
                    r = Row(x[0])
                    try:
                        out = answer(lived, r, x[0], [])
                    except FAIL:
                        out = None
                    if out is not None and out.value == x[1] and (best is None or len(lived.reads) < len(best)):
                        best = lived.reads
                lived.explore, lived.reads = 0., None
                got += best or []
            rd.train()
            for S, items, i, j, fi in got:             # imitate the reading that worked
                u, y, st, fl = rd(tr, S, items)
                loss = loss + (F.cross_entropy(st[None], torch.tensor([i], device=dev)) + F.cross_entropy(rd.ends(y, fl, i)[None], torch.tensor([j], device=dev))) * 16 / max(1, len(got))
                if fi >= 0:
                    loss = loss + F.cross_entropy(rd.fn(y, u[i - 1][2] + 1, u[j - 1][3] + 1)[None] / .1, torch.tensor([fi], device=dev)) * 16 / max(1, len(got))
        for S, texts, x in random.sample(demos, min(16, len(demos))):
            items = [Item(node(t), None) for t in texts]
            for it, t in zip(items, texts):
                it._t = t
            S = re.sub(r"\((<\d+>)\)", r"\1", S)
            u, y, st, fl = rd(tr, S, items)
            i, j, f = reader_targets(u, x)
            loss = loss + F.cross_entropy(st[None], torch.tensor([i], device=dev)) + F.cross_entropy(rd.ends(y, fl, i)[None], torch.tensor([j], device=dev))
            if f >= 0:
                loss = loss + F.cross_entropy(rd.fn(y, u[i - 1][2] + 1, u[j - 1][3] + 1)[None] / .1, torch.tensor([f], device=dev))
        opt.zero_grad()
        (loss / 16).backward()
        opt.step()
        if step % A.every == 0:
            rd.eval()
            w = new_world(random.Random(A.seed + step))
            m = Machine(Oracle(w), 64)
            m.flat, m.reader = True, (rd, tr)
            for _ in range(60):
                for draw in world_step(w, rng):
                    o = draw()
                    for s_ in statements(o) if o else []:
                        arrive(m, s_)
            line = []
            for kind, ops in (("lookup", 0), ("before", 0), ("coexist", 0), ("chain", 0), ("ops", (1, 3)), ("long", (4, 6))):
                res = [x for x in (ask_recursive(m, "ops" if kind == "long" else kind, w, rng, ops or (1, 3)) for _ in range(A.q)) if x]
                line.append(f"{kind} {sum(x[1] == x[2] for x in res)}/{len(res)}")
            print(f"step {step} loss {float(loss) / 16:.3f} | fresh worlds, answered by the model's reading: " + " | ".join(line) + f" | {time.time() - t0:.0f}s", flush=True)
            for kind, ops in (("ops", (1, 3)), ("ops", (4, 6))):   # where the model's reading first differs
                shown = 0
                for _ in range(30):
                    x = question(kind, w, rng, ops)
                    d = x and reading_diff(m, x[0])
                    if d:
                        print(f"   first difference ({ops}): {d[0]!r} | stand-in {d[1]} | model {d[2]}", flush=True)
                        shown += 1
                    if shown == 3:
                        break
            rd.train()
            if A.save:
                torch.save(rd.state_dict(), A.save)


def inspect_memory(A):
    """--inspect: live a memory for --steps ticks (the current nets, no lessons), then open it: the graph, and an easy
    and a hard question with every inner step of every call (the reader, the walk, relate, reach, the tools, the entries
    read and how arrive settled them)"""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr = Translator().to(dev).eval()
    if A.init:
        tr.load_state_dict(torch.load(A.init, map_location=dev), strict=False)
    bank = load_tools(A, tr)
    rng = random.Random(A.seed)
    w = new_world(rng)
    T = NeuralT(tr, Oracle(w), A.tau, bank)
    m = Machine(T, T.d)
    m.g.beam, m.events, ledger = A.beam, [], Ledger()
    if A.qreader:
        rd = QueryReader().to(dev).eval()
        rd.load_state_dict(torch.load(A.qreader, map_location=dev), strict=False)
        m.reader = (rd, tr)
    t0 = time.time()
    for tick in range(A.steps):
        for draw in world_step(w, rng):
            o = draw()
            for s_ in statements(o) if o else []:
                try:
                    arrive(m, s_)
                except FAIL:
                    pass
        judge(m, ledger)
    g, t = m.g, teacher(m)
    print(f"=== lived {A.steps} ticks in {time.time() - t0:.0f}s | rules: {ledger.line()}", flush=True)

    # ---------------------------------------------------------------- the graph
    print("\n=== THE GRAPH", flush=True)
    made_by = {p: m.wrote.get((p, 0)) or T.say(g.chain[p][0][0]) for p in range(g.n)}
    kinds = {}
    for p in range(g.n):
        kinds.setdefault(t.key(made_by[p])[0], []).append(p)
    lens = [len(c) for c in g.chain[:g.n]]
    deg = [len(a) for a in g.adj[:g.n]]
    print(f"places {g.n} ({', '.join(f'{k} {len(v)}' for k, v in kinds.items())}) | entries {sum(lens)}, per place mean {np.mean(lens):.1f} "
          f"max {max(lens)} | edges per place mean {np.mean(deg):.1f} max {max(deg)} min {min(deg)} | reach mean {g.S[:g.n].mean():.3f} "
          f"min {g.S[:g.n].min():.3f} max {g.S[:g.n].max():.3f}, widened {(g.S[:g.n] > REACH + 1e-6).sum()} narrowed {(g.S[:g.n] < REACH - 1e-6).sum()}", flush=True)
    claims = [(p, j) for p in range(g.n) for j in np.nonzero(g.L[:g.n] @ g.R[p] > 1)[0] if j != p]
    print(f"relation claims (R[p] . L[j] > 1): {len(claims)}; e.g. " + "; ".join(f"{made_by[p][:40]!r} -> {made_by[j][:30]!r}" for p, j in claims[:3]), flush=True)
    subj = {}
    for p in kinds.get("fact", []):
        k = t.key(made_by[p])
        subj.setdefault(k[2], []).append(p)
    D = np.linalg.norm(g.L[:g.n, None] - g.L[None, :g.n], axis=-1)
    same = [D[a, b] for ps in subj.values() for a in ps for b in ps if a < b]
    print(f"label distances: places of one name {np.mean(same) if same else 0:.3f} (n {len(same)}), any two places {D[np.triu_indices(g.n, 1)].mean():.3f}, "
          f"nearest other place per place: mean {np.sort(D + np.eye(g.n) * 9, 1)[:, 0].mean():.3f} min {np.sort(D + np.eye(g.n) * 9, 1)[:, 0].min():.3f}", flush=True)
    ok, n_, bad = 0, 0, []
    for p in rng.sample(range(g.n), min(150, g.n)):
        for i, (z, _) in enumerate(g.chain[p]):
            src = m.wrote.get((p, i))
            if src:
                d = T.say(z)
                n_ += 1
                ok += d == src
                if d != src and len(bad) < 4:
                    bad.append((src, d))
    print(f"entries said back as written: {ok}/{n_}; e.g. " + "; ".join(f"{a[:45]!r} -> {b[:45]!r}" for a, b in bad), flush=True)
    kept = [d for d in m.chose.values()]
    print(f"arrive's check decisions: {len(kept)}, kept what the tool made {sum(d[3] is not d[2] for d in kept)}", flush=True)
    for p in rng.sample(kinds.get("fact", [0]), 3) + kinds.get("rule", [])[:2]:
        print(f"   place {p} reach {g.S[p]:.3f} edges {len(g.adj[p])}: " + " || ".join(
            f"[{i}] {T.say(z)[:45]!r}" + (f" (written {m.wrote.get((p, i))[:30]!r})" if m.wrote.get((p, i)) and m.wrote.get((p, i)) != T.say(z) else "")
            for i, (z, _) in enumerate(g.chain[p][:4])), flush=True)

    # ---------------------------------------------------------------- the inner steps, watched
    out = lambda *a: print(*a, flush=True)
    real_find, real_apply, real_read = Machine.find, NeuralT._apply, globals()["read_query"]
    paths = []                                         # each question's walks, for the picture

    def walk(q):
        dist = lambda ids: np.linalg.norm(g.L[ids] - q, axis=-1)
        seen, opened = {0: float(dist([0])[0])}, []
        while True:
            top = sorted(seen, key=seen.get)[:g.beam]
            u = next((i for i in top if i not in opened), None)
            if u is None:
                return opened
            opened.append((u, round(seen[u], 3)))
            new = [v for v in g.adj[u] if v not in seen]
            if new:
                seen.update(zip(new, dist(new).tolist()))

    def find(self, label, q=None, task="same", look=8):
        got = real_find(self, label, q, task, look)
        op = walk(label)
        out(f"        walk ({task}): opened {len(op)} places: " + " ".join(f"{u}:{d}" for u, d in op[:14]) + (" ..." if len(op) > 14 else ""))
        for d, j in g.search(label)[:5]:
            one = lambda x: (x[None], torch.ones(1, len(x), dtype=torch.bool, device=x.device))
            with torch.no_grad():
                rel = float(bank.relate([task], *one(q), *one(g.chain[j][0][0]))[0]) if q is not None else float("nan")
            out(f"          cand {j}: dist {d:.3f} reach {g.S[j]:.3f} {'IN' if d < g.S[j] else 'out'} relate {rel:+.2f} | made by {made_by.get(j, '?')[:45]!r} | newest {T.say(g.chain[j][-1][0])[:40]!r}")
        out(f"        -> place {got}")
        if paths:
            paths[-1]["steps"].append({"task": task, "opened": [u for u, _ in op], "found": int(got)})
        return got

    def apply_(self, want, ins, tool):
        s_ = self.tools.gate(self.tools.query(self.m, [want]))[0]
        r_ = real_apply(self, want, ins, tool)
        t_ = self.last
        if t_ is not None:
            o_, many, err = self.tools.nets[t_]([self.m.content([want])] + [(z[None], torch.ones(1, len(z), dtype=torch.bool, device=z.device)) for z in ins])
            sets = [self.say(z[0], self.tools.nets[t_].mouth) for z, _ in o_]
            out(f"        tool: want {want!r} | inputs {[self.say(z)[:40] for z in ins]} | routing {[round(float(x), 2) for x in s_]} -> tool {t_} | "
                f"output sets {sets} | count head {[round(float(x), 2) for x in many.flatten()]} | own error {float(err):.2f} | said {r_}")
        else:
            out(f"        tool: want {want!r}: no label relevant (routing {[round(float(x), 2) for x in s_]}) -> {r_}")
        return r_

    def read(m_, S, items):
        rd_, tr_ = m_.reader
        S2 = re.sub(r"\((<\d+>)\)", r"\1", S)
        with torch.no_grad():
            u, y, st, fl = rd_(tr_, S2, items)
            ps = F.softmax(st, -1)
            top = ps.topk(min(3, len(ps)))
            words = lambda i: "nothing left" if i == 0 else repr(S2[u[i - 1][0]:u[i - 1][1]])
            out(f"      reader: start {', '.join(f'{words(int(i))} {float(p):.2f}' for p, i in zip(top.values, top.indices))}")
            i = int(st.argmax())
            if i:
                pe = F.softmax(rd_.ends(y, fl, i), -1)
                te = pe.topk(min(3, len(pe)))
                out(f"              end {', '.join(f'{words(int(j))} {float(p):.2f}' for p, j in zip(te.values, te.indices))}")
                j = int(pe.argmax())
                fs = rd_.fn(y, u[i - 1][2] + 1, u[j - 1][3] + 1)
                out(f"              function " + ", ".join(f"{f} {float(v):.2f}" for f, v in zip(rd_.FN, fs)))
        return real_read(m_, S, items)

    def asked(q, truth):
        paths.append({"q": q, "truth": truth, "steps": []})
        out(f"\n--- QUESTION {q!r} | truth {truth}")
        steps = [0]
        real_answer = globals()["answer"]

        def ans(m_, r_, S, items, depth=0):
            steps[0] += 1
            ex = inner_exact(S, items)
            out(f"   step {steps[0]}: {re.sub(chr(40) + r'(<\d+>)' + chr(41), r'\1', S)!r} | nodes {[(k, it.text[:30]) for k, it in enumerate(items)]}")
            out(f"      stand-in reads: {None if ex is None else (S[ex[0]:ex[1]], ex[2])}")
            for k, it in enumerate(items):
                if it.place >= 0 and (it.place, it.idx) in m.chose:
                    gv, o_, nw, kp, ot = m.chose[(it.place, it.idx)]
                    out(f"      node <{k}> was settled by the check: under {gv.text[:40]!r}, old {o_.text[:30]!r}, new {nw.text[:35]!r} -> kept {kp.text[:25]!r}")
            return real_answer(m_, r_, S, items, depth)
        Machine.find, NeuralT._apply, globals()["read_query"], globals()["answer"] = find, apply_, read, ans
        try:
            r = Row(q)
            got = answer(m, r, q, [])
        except FAIL as e:
            got = None
            out(f"   FAILED: {type(e).__name__} {e}")
        finally:
            Machine.find, NeuralT._apply, globals()["read_query"], globals()["answer"] = real_find, real_apply, real_read, real_answer
        out(f"   ANSWER {got.text if got is not None else None!r} -> {got.value if got is not None else None} | truth {truth} | "
            f"{'RIGHT' if got is not None and got.value == truth else 'WRONG'} | {steps[0]} steps")

    print("\n=== AN EASY QUESTION (1 step)", flush=True)
    x = question("lookup", w, rng)
    asked(x[0], x[1])
    print("\n=== A HARD QUESTION (~10 steps)", flush=True)
    for _ in range(200):
        x = question("ops", w, rng, (4, 5))
        if x and len(re.findall(r"the [a-z ]+ of [A-Z]", x[0])) >= 5:
            break
    asked(x[0], x[1])
    print("\n=== A CHAIN QUESTION", flush=True)
    x = question("chain", w, rng)
    if x:
        asked(x[0], x[1])
    if A.log:                                          # the graph as data, for a picture of it
        L = g.L[:g.n] - g.L[:g.n].mean(0)
        xy = L @ np.linalg.svd(L, full_matrices=False)[2][:2].T
        nodes = [{"id": p, "kind": t.key(made_by[p])[0], "made": made_by[p][:90], "n": len(g.chain[p]),
                  "newest": T.say(g.chain[p][-1][0])[:90], "reach": round(float(g.S[p]), 3), "deg": len(g.adj[p]),
                  "x": round(float(xy[p, 0]), 4), "y": round(float(xy[p, 1]), 4)} for p in range(g.n)]
        edges = sorted({(min(p, j), max(p, j)) for p in range(g.n) for j in g.adj[p]})
        json.dump({"nodes": nodes, "edges": edges, "claims": [[int(p), int(j)] for p, j in claims], "paths": paths},
                  open(A.log, "w", encoding="utf-8"))
        print(f"graph written: {len(nodes)} places, {len(edges)} edges, {len(claims)} claims, {len(paths)} question paths", flush=True)


def ask_recursive(m, kind, world, rng, ops=(1, 3)):
    """(question, truth, what it answered, passes): the question only, no procedure"""
    x = question(kind, world, rng, ops)
    if not x:
        return None
    r = Row(x[0])
    try:
        out = answer(m, r, x[0], [])
    except FAIL:
        out = None
    return r.src, x[1], out.value if out is not None else None, r.passes


def gold_tree(m, r, t):
    if t[0] == "leaf":
        return gold_read(m, r, leaf(t[1], t[2]))
    a, b = t[2]
    pre = next(x for x in TOOLS if x[0] == t[1])[2].split("{0}")[0]
    m.do(r, "split", len((pre + ("(" if a[0] == "op" else "")).strip()))
    gold_tree(m, r, a)
    m.do(r, "split", re.match(r"[)\s]*and( \()?", r.focus[-1]).end())   # ") and (": closes the left, opens the right
    m.do(r, "drop")                                    # it has done its job; the word stays until the tool
    gold_tree(m, r, b)
    m.do(r, "apply", 2)                                # closing brackets are never read: the stack says what is left


def check_exact(rule, old, new, cand):
    """the exact stand-in's check(rule, resulting state) -> 1 pass / 0 fail: a correction or a copy is kept as said;
    under a rule that adds, the kept value is old + new; under any other rule, the statement as said"""
    if new.text.startswith("Correction") or "(same as" in new.text:
        return float(cand is new)
    if any(x.search(rule.text) for x in ADD_RE):
        return float(cand is not new and number(cand.text.strip()) == (old.value or 0) + (new.value or 0))
    return float(cand is new)


def check(m, rule, old, new, cand):
    return check_exact(rule, old, new, cand) if isinstance(m.T, Oracle) else m.T.check(rule, old, new, cand)


@torch.no_grad()
def resettle(m, bank, dev):
    """the check changed: every decision it made is made again; an entry whose decision flips takes the other
    candidate, and when it is the newest, what depends on it is triggered (a changed function node runs again what it
    made; on14: the check began to learn after ~250 ticks while the entries it had stored wrong stayed wrong)"""
    keys = [k for k in m.chose if torch.is_tensor(m.chose[k][0].node)]
    if not keys:
        return 0
    D = [m.chose[k] for k in keys]
    made = [d[3] if d[3] is not d[2] else d[4] for d in D]
    base = [nodes_of([d[j].node for d in D], dev) for j in range(3)]
    said = torch.sigmoid(bank.check(base + [nodes_of([d[2].node for d in D], dev)]))
    tool = torch.sigmoid(bank.check(base + [nodes_of([x.node for x in made], dev)]))
    n = 0
    for k, d, mk, a, b in zip(keys, D, made, said.tolist(), tool.tolist()):
        g, o, nw, keep, other = d
        took = keep is not nw                          # flip only when now sure the other way (on17: without this a
        take = (a < .3 and b > .7) if not took else not (a > .7 or b < .3)   # lesson re-decided 1,525 entries back and forth)
        if take != took:
            p, i = k
            new_keep = mk if take else nw
            m.g.chain[p][i] = (new_keep.node, m.g.chain[p][i][1])
            m.chose[k] = (g, o, nw, new_keep, nw if take else mk)
            if i == len(m.g.chain[p]) - 1:
                m.changed(p)
            n += 1
    return n


def arrive(m, text):
    """a statement arrives; a rule is a statement like any other. The check's want looks for the node that governs
    where this one goes; when there is one and the place already holds a value, the candidates are the statement as
    said and what a tool makes of (old, new) under that node, and the check of each resulting state keeps one"""
    r = Row(text)
    m.do(r, "rule")                                    # the check's want: what governs this (a node, or none)
    gov = r.stack.pop()
    if gov.place >= 0:
        p = m.find(m.T.label(text), m.T.node(text))
        if p >= 0:
            old, new = m.entry_item(p), m.item(text)
            r.stack += [m.item(gov.text), old, new]
            m.do(r, "apply", 2)
            made = r.stack.pop()
            # the statement as said is the state, unless it fails the governing check and what the tool made passes
            fails = check(m, gov, old, new, new) < .5 and check(m, gov, old, new, made) > .5
            keep, other = (made, new) if fails else (new, made)
            if keep is made:
                r.stack.append(made)
                m.do(r, "insert", True)
            else:
                m.do(r, "insert")
            p = m.g.n - 1 if m.at.get("new") else m.at["place"]
            m.chose[(p, len(m.g.chain[p]) - 1)] = (gov, old, new, keep, other)
            return
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
            m.do(r, "apply", 2)
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
            m.do(r, "apply", 1)
            m.do(r, "lookup")
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
FUNCS = ["split", "lookup", "rule", "prev", "drop", "nip", "apply1", "apply2", "apply3", "insert", "derive", "say"]


def view(r):
    """what a pass sees: the top of what it holds, then what it reads next; bounded at every depth"""
    held = " | ".join(it.text.strip() for it in r.stack[-3:])   # [word, a, b] is all a call needs, so depth cannot show
    base = len(held) + 4
    return f"{held} || {r.focus[-1][:90] if r.focus else ''}", base


class Reader(nn.Module):
    """reads the view, states a want; the want matches a function node. A split ends at the first token boundary,
    judged from the text up to the next token only: what lies further on cannot move it, so depth cannot either."""

    def __init__(self, tr, d=256):
        super().__init__()
        self.tr = tr
        D = tr.lm.config.hidden_size
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(D, 8, 2048, batch_first=True, activation="gelu"), 2)
        self.want = nn.Linear(D, d)
        self.fn = nn.Parameter(F.normalize(torch.randn(len(FUNCS), d), dim=-1))
        self.tau = nn.Parameter(torch.tensor(20.))
        self.end = nn.Sequential(nn.Linear(2 * D, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, views):
        dev = self.want.weight.device
        b = self.tr.tok([v[0] for v in views], return_tensors="pt", padding=True, return_offsets_mapping=True)
        off = b.pop("offset_mapping")
        b = b.to(dev)
        with torch.no_grad():
            g = self.tr.lm(**b).last_hidden_state          # causal: token i has seen only what comes before it
        nxt = torch.cat([g[:, 1:] * b["attention_mask"][:, 1:, None], torch.zeros_like(g[:, :1])], 1)
        h = self.enc(g, src_key_padding_mask=b["attention_mask"] == 0)
        last = b["attention_mask"].sum(1) - 1
        s = h[torch.arange(len(views), device=dev), last]
        logits = self.tau * F.normalize(self.want(s), dim=-1) @ F.normalize(self.fn, dim=-1).T
        return logits, self.end(torch.cat([g, nxt], -1)).squeeze(-1), off, b["attention_mask"]


def first_end(e, o, base):
    """the first token of the focus the chunk ends after (the likeliest one if none says so)"""
    ok = [j for j, (a, b) in enumerate(o) if b > base]
    if not ok:
        return None
    return next((j for j in ok if e[j] > 0), max(ok, key=lambda j: e[j]))


def step_targets(views, names, chars, off):
    """a split's end, in tokens: the token whose end is that character"""
    t = []
    for (v, base), n, o in zip(views, chars, off.tolist()):
        if n is None:
            t.append(-1)
            continue
        want = base + n
        ends = [i for i, (a, e) in enumerate(o) if e > base]
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
    if k.any():                                        # the tokens before the end say "not yet", the end says "here"
        first = torch.tensor([next((i for i, (a, e) in enumerate(o) if e > v[1]), 0) for v, o in zip(views, off.tolist())], device=dev)
        pos = torch.arange(ends.shape[1], device=dev)[None]
        on = (pos >= first[:, None]) & (pos <= t[:, None]) & k[:, None]
        loss = loss + F.binary_cross_entropy_with_logits(ends[on], (pos == t[:, None]).float().expand_as(ends)[on])
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
        t = step_targets(views, names, chars, off).tolist()
        for j, name in enumerate(names):
            tot[name] = tot.get(name, 0) + 1
            hit[name] = hit.get(name, 0) + int(FUNCS[int(logits[j].argmax())] == name)
            if t[j] >= 0:
                et += 1
                eh += int(first_end(ends[j].tolist(), off[j].tolist(), views[j][1]) == t[j])
    return " ".join(f"{k} {hit[k] / tot[k]:.3f}" for k in FUNCS if k in tot) + f" | split end {eh / max(1, et):.3f}"


@torch.no_grad()
def pick(rd, vs):
    """the reader's call on each view: (function, where a split ends, confidence, margin, every end it could point at)"""
    logits, ends, off, mask = rd(vs)
    p, fn = logits.softmax(1).topk(2, 1)
    out = []
    for i, (v, base) in enumerate(vs):
        o, e = off[i].tolist(), ends[i].tolist()
        ok = [j for j, (a, b) in enumerate(o) if b > base]
        j = first_end(e, o, base)
        n = o[j][1] - base if j is not None else None
        out.append((FUNCS[int(fn[i, 0])], n, float(p[i, 0]), float(p[i, 0] - p[i, 1]), [o[j][1] - base for j in ok]))
    return out


def run_free(m, rd, r, stop, max_pass=90, show=False):
    """the reader drives the row; returns False when it asks for something impossible"""
    seen = set()
    try:
        while r.passes < max_pass:
            state = (tuple((it.text, it.place, it.idx) for it in r.stack), tuple(r.focus))
            if state in seen:
                r.stuck = True
                return False
            seen.add(state)
            v = view(r)
            name, n = pick(rd, [v])[0][:2]
            if show:
                print("     ", repr(v[0][:110]), "->", name, flush=True)
            if name == "split":
                if n is None:
                    return False
                m.do(r, "split", n)
            elif name.startswith("apply"):
                m.do(r, "apply", int(name[5:]))
            elif name == "derive":
                m.do(r, "insert", True)
            else:
                m.do(r, name)
            if name in stop:
                return True
    except (IndexError, AttributeError, Budget, TypeError, ValueError):
        return False
    return False


def branch(trace):
    """which way a row went: the functions it called, whether a rule was there, whether it was a copy"""
    return (tuple(sorted({x[1] for x in trace})),
            any("no rule" in x[0][0] for x in trace), any("(same as" in x[0][0] for x in trace))


def collect(world, m, rng, per_kind, ticks=180):
    """the demonstrations: rows from a world's whole life, up to per_kind for each way a row can go"""
    rows, count = {}, {}
    m.trace = []

    def keep():
        b = branch(m.trace)
        count[b] = count.get(b, 0) + 1
        if len(rows.setdefault(b, [])) < per_kind:
            rows[b].append(m.trace)
        elif (j := rng.randrange(count[b])) < per_kind:
            rows[b][j] = m.trace
        m.trace = []

    for _ in range(ticks):
        for draw in world_step(world, rng):
            o = draw()
            for s in statements(o) if o else []:
                arrive(m, s)
                keep()
        for k in ("lookup", "before", "coexist", "chain", "ops"):
            if ask(m, k, world, rng):
                keep()
            m.trace = []
    m.trace = None
    bank = [x for r in rows.values() for row in r for x in row]
    print(f"demonstrations: {len(rows)} ways, {sum(map(len, rows.values()))} rows, {len(bank)} steps;", sorted(len(r) for r in rows.values()), flush=True)
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


def key_re(w):
    ks = sorted({words(k) for k in w.num_keys + w.multi_keys + w.ent_keys}, key=len, reverse=True)
    return r"\b(?:" + "|".join(re.escape(k) for k in ks) + r")\b"


def train_reader(A):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr = Translator().to(dev).eval()
    tr.load_state_dict(torch.load(A.init, map_location=dev), strict=False)
    tools = load_tools(A, tr)
    rd = Reader(tr).to(dev)
    rng = random.Random(A.seed)
    w0 = new_world(rng)
    bank = collect(w0, Machine(Oracle(w0)), rng, A.demos)
    keys = key_re(w0)
    opt = torch.optim.AdamW([p for n, p in rd.named_parameters() if not n.startswith("tr.")], lr=A.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, A.steps, eta_min=A.lr / 50)
    n = {}
    for x in bank:
        n[x[1]] = n.get(x[1], 0) + 1
    weight = [n[x[1]] ** -.5 for x in bank]             # a function seen in few steps (nip, join) is not drowned by splits
    t0 = time.time()
    for step in range(1, A.steps + 1):
        rd.train()
        loss, acc = reader_loss(rd, [fresh(rng, x, keys) for x in rng.choices(bank, weight, k=A.batch)])
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % A.every == 0:
            rd.eval()
            print("   per function:", step_report(rd, bank, rng, keys), flush=True)
            w = new_world(rng)
            T = NeuralT(tr, Oracle(w), A.tau, tools)
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


# ====================================================================== --online: everything at once, as it happens
def online(A):
    """the model lives in the world: it is told what happens (stored by the correct calls), answers questions, reads
    demos. Every call is judged by the rules; the judgments are the only lessons (plus the tools reading demos). The
    rules' tallies per window show whether it is getting better while it lives."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr = Translator().to(dev).eval()
    if A.init:
        tr.load_state_dict(torch.load(A.init, map_location=dev), strict=False)
    bank = load_tools(A, tr) or ToolBank().to(dev)
    for n, p in tr.named_parameters():                 # what learns online: the relation gate (and the bank: relate, tools)
        p.requires_grad = n.split(".")[0] in ("rgate", "rhow", "up", "ruleq")
    opt = torch.optim.AdamW([p for p in tr.parameters() if p.requires_grad] + list(bank.parameters()), lr=A.lr)
    rng = random.Random(A.seed)
    w = new_world(rng)
    T = NeuralT(tr, Oracle(w), A.tau, bank)
    m = Machine(T, T.d)
    m.g.beam, m.events = A.beam, []
    if A.qreader:                                      # the model reads the questions itself (recursion), no procedure
        rd = QueryReader().to(dev).eval()
        rd.load_state_dict(torch.load(A.qreader, map_location=dev), strict=False)
        m.reader = (rd, tr)
    st, t0, said, ledger, total = {"step": 0, "last": -10 ** 9, "newest": None}, time.time(), 0, Ledger(), Ledger()
    for tick in range(1, A.steps + 1):
        st.update(step=tick, spawned=False)
        for draw in world_step(w, rng):
            o = draw()
            for s_ in statements(o) if o else []:
                try:
                    arrive(m, s_)
                    said += 1
                except FAIL:
                    pass
        js = judge(m, ledger)
        for kind in ("lookup", "chain", "ops"):        # asked, answered, then the truth
            x = question(kind, w, rng)
            if not x:
                continue
            r = Row(x[0])
            try:
                if m.reader:
                    out = answer(m, r, x[0], [])
                    r.stack.append(out) if out is not None else None
                else:
                    x[2](m, r)
                m.do(r, "say")
            except FAIL:
                pass
            jq = judge(m, ledger, r, x[1])
            js += jq
            o_ = next((j for j in jq if j.rule == "outcome"), None)
            if o_ is not None and not o_.ok:           # explore: the same question with each decision the check made
                for key in [k for k in o_.ev["reads"] if k in m.chose]:      # on what it read flipped; if the
                    g, o, n, keep, other = m.chose[key]                      # truth comes out, that decision was wrong
                    p, i = key
                    saved, m.g.chain[p][i] = m.g.chain[p][i], (other.node, m.g.chain[p][i][1])
                    evs, m.events = m.events, None
                    r2 = Row(x[0])
                    try:
                        if m.reader:
                            out = answer(m, r2, x[0], [])
                            r2.stack.append(out) if out is not None else None
                        else:
                            x[2](m, r2)
                        got = next((it.value for it in reversed(r2.stack) if it.value is not None), None)
                    except FAIL:
                        got = None
                    m.events, m.g.chain[p][i] = evs, saved
                    if got == x[1]:
                        m.check_lessons += [(g, o, n, keep, 0.), (g, o, n, other, 1.)]
        for j in js:
            total.add(j)
        loss = lessons(tr, bank, m, js)
        if A.batch:                                    # reading demos: the tool nodes learn (and are made)
            l2 = learn(tr, bank, [read_demo(rng, rng.choice(KINDS)) for _ in range(A.batch)], st, A, none=[asks_nothing(rng) for _ in range(A.batch // 4)])[0]
            loss = l2 if loss is None else loss + l2
            if st["spawned"]:
                opt = torch.optim.AdamW([p for p in tr.parameters() if p.requires_grad] + list(bank.parameters()), lr=A.lr)
        if loss is not None and loss.requires_grad:
            opt.zero_grad()
            loss.backward()
            opt.step()
            if m.check_moved:
                m.resettled += resettle(m, bank, dev)
        if tick % A.every == 0:
            print(f"tick {tick} | places {m.g.n}, statements {said} | this window: {ledger.line()} | re-settled {m.resettled} | {time.time() - t0:.0f}s", flush=True)
            m.resettled = 0
            for r_, cs in ledger.cases.items():
                print(f"     {r_}: {cs[0]}", flush=True)
            ledger = Ledger()
            if A.save:
                bank.save(A.save)
                torch.save({k: v for k, v in tr.state_dict().items() if not k.startswith("lm.")}, A.save.replace(".pt", "_tr.pt"))
    print("all ticks: " + total.line(), flush=True)


# ====================================================================== --episode: the translator moves, the graph is re-transcribed
def retranscribe(m, old, new):
    """the translator changed: the old one reads every node out, the new one writes it back. Places keep their entries
    and times; a place's label comes from its first entry (a statement: derived entries only follow one)."""
    g = Graph(m.g.d, m.g.beam)
    out = Machine(new, new.d, m.budget)
    out.g, out.t = g, m.t
    for ch in m.g.chain:
        said = [old.say(e[0]) for e in ch]
        p = g.add_place(new.label(said[0]))
        g.chain[p] = [(new.node(t), e[1]) for t, e in zip(said, ch)]
    return out


def safe_ask(m, kind, w, rng, ops):
    """ask, and count calls that could not finish on this memory as a wrong answer"""
    try:
        return ask(m, kind, w, rng, ops)
    except FAIL:
        return (None, 1, None, 0)


def asked(m, w, rng, q):
    """the correct calls on this memory: right answers per kind"""
    line = []
    for kind, ops in (("lookup", 0), ("before", 0), ("coexist", 0), ("chain", 0), ("ops", (1, 3))):
        res = [x for x in (safe_ask(m, kind, w, rng, ops or (1, 3)) for _ in range(q)) if x]
        line.append(f"{kind} {sum(x[1] == x[2] for x in res)}/{len(res)}")
    return " | ".join(line)


def episode(A):
    """build a memory, then unfreeze the translator (its heads, the decoder, the tool nodes) and train them together for
    a while; ask on the stale graph and on the re-transcribed one. Same world, same questions (seeded)."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr = Translator().to(dev).eval()
    if A.init:
        tr.load_state_dict(torch.load(A.init, map_location=dev), strict=False)
    tools = load_tools(A, tr) or ToolBank().to(dev).eval()
    rng = random.Random(A.seed)
    w = new_world(rng)
    T1 = NeuralT(tr, Oracle(w), A.tau, tools)
    m = Machine(T1, T1.d)
    m.g.beam = A.beam
    for _ in range(A.world):
        for draw in world_step(w, rng):
            o = draw()
            for s in statements(o) if o else []:
                arrive(m, s)
    qs = rng.randrange(1 << 30)
    print(f"before: places {m.g.n} | {asked(m, w, random.Random(qs), A.q)}", flush=True)
    old = NeuralT(copy.deepcopy(tr).eval(), Oracle(w), A.tau, copy.deepcopy(tools).eval())
    for n, p in tr.named_parameters():
        p.requires_grad = not n.startswith("lm.")
    tools.requires_grad_(True)
    opt = torch.optim.AdamW([p for p in tr.parameters() if p.requires_grad] + list(tools.parameters()), lr=A.lr)
    tr.train(), tools.train()
    t0 = time.time()
    for step in range(1, A.steps + 1):
        T = [sentence(rng, KEYS_TRAIN) for _ in range(A.batch)]
        st = {"step": step, "last": -10 ** 9, "newest": None}
        loss = (label_loss(tr, rng, KEYS_TRAIN, A.grid) + tr.decode_loss(*tr.content(T), T)
                + learn(tr, tools, [read_demo(rng, rng.choice(KINDS)) for _ in range(A.batch)], st, A, none=[asks_nothing(rng) for _ in range(A.batch // 4)], spawn=False)[0])
        opt.zero_grad()
        loss.backward()
        opt.step()
    tr.eval(), tools.eval()
    moved = float(sum((p - q).norm() ** 2 for p, q in zip(tr.parameters(), old.m.parameters())) ** .5)
    print(f"episode: {A.steps} steps, loss {float(loss):.3f}, translator moved {moved:.2f} | {time.time() - t0:.0f}s", flush=True)
    T2 = NeuralT(tr, Oracle(w), A.tau, tools)
    stale = Machine(T2, T2.d)
    stale.g, stale.t = m.g, m.t
    print(f"stale graph, new translator: {asked(stale, w, random.Random(qs), A.q)}", flush=True)
    new = retranscribe(m, old, T2)
    print(f"re-transcribed: places {new.g.n} | {asked(new, w, random.Random(qs), A.q)}", flush=True)
    if A.save:
        torch.save({k: v for k, v in tr.state_dict().items() if not k.startswith("lm.")}, A.save)
        tools.save(A.save.replace(".pt", "_tools.pt"))


# ====================================================================== --diag: where does a free row go wrong?
FAIL = (IndexError, AttributeError, Budget, TypeError, ValueError, StopIteration, KeyError)
ASKED = (("lookup", 0), ("before", 0), ("coexist", 0), ("chain", 0), ("ops", (1, 3)), ("long", (4, 6)))


def held(r):
    return next((it.value for it in reversed(r.stack) if it.value is not None), None)


def shape(text, keys):
    """a view with names, keys and numbers as first-seen placeholders: all any reader could tell apart"""
    seen = {}

    def sub(m):
        w = m.group()
        if w in ("What", "Correction", "The"):
            return w
        k = "#" if w.isdigit() else "N" if w[0].isupper() else "K"
        return seen.setdefault(w, f"{k}{sum(x[0] == k for x in seen.values())}")
    return re.sub(keys + r"|\b(?:[A-Z][a-z]{2,}|\d+)\b", sub, text)


def gold_trace(m, run):
    """the correct calls on what this memory holds now, nothing written"""
    m.trace, m.f_insert = [], lambda r, derived=False: None
    try:
        x = run()
    except FAIL:
        x = None
    del m.f_insert
    t, m.trace = m.trace, None
    return t, x


def lockstep(rd, t, keys, seen):
    """the reader's call at every correct step. Same state until its first wrong call, so that call is where it left."""
    out, first = [], None
    for i in range(0, len(t), 64):
        for j, ((v, f, n), (pf, pn, conf, margin, ends)) in enumerate(zip(t[i:i + 64], pick(rd, [x[0] for x in t[i:i + 64]]))):
            text, base = v
            cut = lambda c: text[base:base + c].strip()
            can = None if n is None else n <= len(text) - base and any(cut(e) == cut(n) for e in ends)
            good = pf == f and (n is None or (pn is not None and n <= len(text) - base and cut(pn) == cut(n)))
            sh = shape(text, keys)
            out.append([f, pf, n, pn, round(conf, 3), round(margin, 3), can, sh in seen, good, sh])
            if not good and first is None:
                first = i + j
    return out, first


def health(m, w, rng, n=200):
    """what the memory holds, against the truth"""
    g, O = m.g, m.T.o if isinstance(m.T, NeuralT) else m.T
    where, said = {}, [[m.T.say(e[0]) for e in ch] for ch in g.chain]
    keys = [{O.key(t) for t in ts if number(t) is None} for ts in said]   # a derived entry is a bare value: its place says what
    for p, ks in enumerate(keys):
        for k in ks:
            where.setdefault(k, set()).add(p)
    mixed = sum(len(ks) > 1 for ks in keys)
    facts = sorted(f for f in w.hist if f[0] in w.ents)
    S = rng.sample(facts, min(n, len(facts)))
    hit = miss = bad = 0
    for nm, k in S:
        p = m.find(m.T.label(leaf(nm, k)), m.T.node(leaf(nm, k)))
        hit += p >= 0 and p in where.get(("fact", words(k), nm), ())
        miss += p < 0
        bad += p >= 0 and p not in where.get(("fact", words(k), nm), ())
    absent = [(nm, k) for nm in sorted(w.ents) for k in w.num_keys if k not in w.ents[nm]]
    A_ = rng.sample(absent, min(n, len(absent)))
    false = sum(m.find(m.T.label(leaf(nm, k)), m.T.node(leaf(nm, k))) >= 0 for nm, k in A_)
    ru = [(k, m.find(m.T.rule_query(leaf(sorted(w.ents)[0], k) + " is 1."), m.T.node(leaf(sorted(w.ents)[0], k) + " is 1."), "rule"))
          for k in sorted(w.policy)]
    rule = sum(p >= 0 and p in where.get(("rule", words(k)), ()) for k, p in ru)
    c0, agree = g.cost, 0
    for nm, k in S:
        q = m.T.label(leaf(nm, k))
        agree += bool(g.n) and g.search(q)[0][1] == g.flat(q)
    return dict(places=g.n, keys=len(where), mixed=mixed, split=sum(len(v) > 1 for v in where.values()), fact_hit=hit, fact_miss=miss,
                fact_wrong=bad, facts=len(S), false_accept=false, absent=len(A_), rule_hit=rule, rules=len(ru), walk_flat=agree,
                cost=(g.cost - c0) / max(1, len(S)))


def diag(A):
    """one ckpt, no training. Who writes (gold / reader) x translator (exact / neural), at growing memory; every row run by
    the correct calls and by the reader in lockstep. Prints a summary; every step goes to --log for slicing afterwards."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr = Translator().to(dev).eval()
    if A.init:
        tr.load_state_dict(torch.load(A.init, map_location=dev), strict=False)
    tools = load_tools(A, tr)
    rd = Reader(tr).to(dev).eval()
    if A.reader:
        print("reader:", rd.load_state_dict(torch.load(A.reader, map_location=dev), strict=False).unexpected_keys, flush=True)
    r0 = random.Random(A.seed)
    w0 = new_world(r0)
    bank = collect(w0, Machine(Oracle(w0)), r0, A.demos)
    k0 = key_re(w0)
    seen = {shape(x[0][0], k0) for x in bank}
    print(f"demonstrated views: {len(bank)} steps, {len(seen)} shapes", flush=True)
    log = open(A.log, "w") if A.log else None
    t0 = time.time()
    for who, tname in (("gold", "exact"), ("gold", "neural"), ("reader", "exact"), ("reader", "neural")):
        rng = random.Random(A.seed + 100)
        w = new_world(rng)
        O = Oracle(w)
        T = O if tname == "exact" else NeuralT(tr, O, A.tau, tools)
        m = Machine(T, T.d)
        m.g.beam = A.beam
        keys, done = key_re(w), 0
        for size in (int(x) for x in A.sizes.split(",")):
            wrote, wrong = {}, {}
            while done < size:
                done += 1
                for draw in world_step(w, rng):
                    o = draw()
                    for s in statements(o) if o else []:
                        t, _ = gold_trace(m, lambda: arrive(m, s))
                        names = [x[1] for x in t]
                        kind = ("delta" if "derive" in names else "rule" if O.key(s)[0] == "rule" else "note" if s.startswith("Correction")
                                else "copy" if "(same as" in s else "fact")
                        steps, first = lockstep(rd, t, keys, seen)
                        wrote[kind], wrong[kind] = wrote.get(kind, 0) + 1, wrong.get(kind, 0) + (first is not None)
                        fin = None
                        if who == "gold":
                            arrive(m, s)
                        else:
                            m.trace = []
                            fin = run_free(m, rd, Row(s), ("insert", "derive"))
                            m.trace = None
                        if log:
                            log.write(json.dumps(dict(who=who, tr=tname, size=size, part="write", kind=kind, text=s, first=first, fin=fin, steps=steps)) + "\n")
            h = health(m, w, random.Random(A.seed + size))
            print(f"[{who} writes, {tname}] world {size} | {h} | rows with a wrong call "
                  + " ".join(f"{k} {wrong[k]}/{wrote[k]}" for k in sorted(wrote)) + f" | {time.time() - t0:.0f}s", flush=True)
            if log:
                log.write(json.dumps(dict(who=who, tr=tname, size=size, part="health", **h)) + "\n")
            qr = random.Random(A.seed * 1000 + size)
            for kind, ops in ASKED:
                n = g_ok = f_ok = clean = hit = tot = 0
                left = {}
                for _ in range(A.q):
                    x = question(kind, w, qr, ops or (1, 3))
                    if not x:
                        continue

                    def run():
                        r = Row(x[0])
                        x[2](m, r)
                        m.do(r, "say")
                        return held(r)
                    t, ans = gold_trace(m, run)
                    steps, first = lockstep(rd, t, keys, seen)
                    r, m.trace = Row(x[0]), []
                    said = run_free(m, rd, r, ("say",))
                    ft, m.trace = m.trace, None
                    ok = said and held(r) == x[1]
                    n, g_ok, f_ok = n + 1, g_ok + (ans == x[1]), f_ok + ok
                    clean += first is None
                    hit, tot = hit + sum(y[8] for y in steps), tot + len(steps)
                    if first is not None:
                        y = steps[first]
                        e = f"{y[0]}>{y[1]}" if y[0] != y[1] else "split end" if y[6] else "split end (cannot point)"
                        left[e] = left.get(e, 0) + 1
                    if log:
                        log.write(json.dumps(dict(who=who, tr=tname, size=size, part="ask", kind=kind, text=x[0], truth=x[1], gold=ans, said=said,
                                                  got=held(r), ok=ok, first=first, steps=steps, passes=r.passes, stuck=r.stuck, held=[it.text[:60] for it in r.stack[-5:]],
                                                  free=[(v[0][:160], f, c) for v, f, c in ft[:45]] if not ok else None), default=str) + "\n")
                print(f"   {kind:8s} gold {g_ok}/{n} free {f_ok}/{n} | all calls right {clean}/{n}, steps {hit}/{tot} = {hit / max(1, tot):.4f} | first wrong call: "
                      + ", ".join(f"{a} {b}" for a, b in sorted(left.items(), key=lambda z: -z[1])[:5]) + f" | {time.time() - t0:.0f}s", flush=True)
            if log:
                log.flush()


def gold(A):
    rng = random.Random(A.seed)
    world = new_world(rng)
    T = Oracle(world)
    if A.init:
        model = Translator().to("cuda" if torch.cuda.is_available() else "cpu").eval()
        model.load_state_dict(torch.load(A.init), strict=False)
        T = NeuralT(model, T, A.tau, load_tools(A, model))
    m = Machine(T, T.d)
    m.flat, m.g.beam = A.flat, A.beam
    m.events, ledger = [], Ledger()
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
        judge(m, ledger)
        if step % A.every == 0:
            g, line = m.g, []
            for kind, ops in (("lookup", 0), ("before", 0), ("coexist", 0), ("chain", 0), ("ops", (1, 3)), ("long", (4, 6))):
                res = [x for x in (safe_ask(m, "ops" if kind == "long" else kind, world, rng, ops or (1, 3)) for _ in range(A.q)) if x]
                rec = [x for x in (ask_recursive(m, "ops" if kind == "long" else kind, world, rng, ops or (1, 3)) for _ in range(A.q)) if x]
                ok, rok = [x for x in res if x[1] == x[2]], [x for x in rec if x[1] == x[2]]
                line.append(f"{kind} {len(ok)}/{len(res)} recursion {len(rok)}/{len(rec)} ({sum(x[3] for x in rec) / max(1, len(rec)):.0f}p)")
                for x in [y for y in rec if y[1] != y[2]][:A.show]:
                    print("   MISS (recursion)", kind, x[:3])
            c0, agree = g.cost, 0
            for _ in range(200):
                q = g.L[rng.randrange(g.n)] + .02 * np.random.standard_normal(g.d).astype(np.float32)
                agree += g.search(q)[0][1] == g.flat(q)
            deg = [len(a) for a in g.adj]
            print(f"step {step} | places {g.n} | " + " | ".join(line) + f" | walk = flat {agree}/200, cost {(g.cost - c0) / 200:.0f} "
                  f"of {g.n}, edges {np.mean(deg):.1f} / {max(deg)} | {time.time() - t0:.0f}s", flush=True)
            judge(m, ledger)
            print("   " + ledger.line(), flush=True)
            for r_, cs in ledger.cases.items():
                for c in cs:
                    print(f"     {r_}: {c}", flush=True)


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
    P.add_argument("--decode", action="store_true", help="train the translator's decoder: text -> B -> text (--init = the translator)")
    P.add_argument("--tools", action="store_true", help="train the tool nodes on B (--init = translator with decoder)")
    P.add_argument("--recur", action="store_true", help="train the reader node: answering by recursion (--demos questions)")
    P.add_argument("--check", action="store_true", help="train the check node (--demos situations; --toolw bank to add it to)")
    P.add_argument("--inspect", action="store_true", help="live a memory --steps ticks, then open the graph and two questions, every inner step")
    P.add_argument("--norm", action="store_true", help="train the normalizer: any text -> clear statements")
    P.add_argument("--norm-e2e", action="store_true", help="E4: the memory fed held-out forms through the normalizer (--normw)")
    P.add_argument("--normw", default="")
    P.add_argument("--norm-hf", default="gpt2", help="the normalizer's base language model")
    P.add_argument("--online", action="store_true", help="the model lives in the world: reads, is told, answers, learns, all at once")
    P.add_argument("--rerun", type=int, default=10, help="--online: every this many ticks, what changed tools made is made again")
    P.add_argument("--episode", action="store_true", help="build a memory, unfreeze the translator for --steps, re-transcribe the graph")
    P.add_argument("--read", action="store_true", help="train the reader on demonstrations (--init = the translator)")
    P.add_argument("--diag", action="store_true", help="no training: every row by the correct calls and by the reader in lockstep")
    P.add_argument("--reader", default="", help="--diag: the reader's weights")
    P.add_argument("--qreader", default="", help="--online: the reader node's weights: questions answered by recursion")
    P.add_argument("--sizes", default="20,60,180", help="--diag: steps of world at which to stop and ask")
    P.add_argument("--demos", type=int, default=50)
    P.add_argument("--batch", type=int, default=64)
    P.add_argument("--world", type=int, default=30, help="steps of world the free eval lives through")
    P.add_argument("--init", default="")
    P.add_argument("--scope", type=float, default=.9, help="a new tool's label: wants within this cos of it are relevant")
    P.add_argument("--kappa", type=float, default=2., help="unusually bad for a tool = its usual error + kappa spreads")
    P.add_argument("--tool-good", type=float, default=.3, help="an error this low is never unusual")
    P.add_argument("--cool", type=int, default=100, help="steps between two new tools")
    P.add_argument("--toolw", default="", help="the tool nodes' weights: with --init (translator + decoder) = the whole neural path")
    P.add_argument("--save", default="")
    P.add_argument("--log", default="")
    P.add_argument("--lr", type=float, default=3e-4)
    P.add_argument("--grid", type=int, default=8, help="names x keys per training batch: bigger = harder near-misses")
    P.add_argument("--tau", type=float, default=.15, help="closer than this = the same place")
    A = P.parse_args()
    diag(A) if A.diag else inspect_memory(A) if A.inspect else train_check(A) if A.check else train_recursion_reader(A) if A.recur else train_decoder(A) if A.decode else train_tools(A) if A.tools else episode(A) if A.episode else online(A) if A.online else train_normalizer(A) if A.norm else norm_e2e(A) if A.norm_e2e else train_reader(A) if A.read else train_translator(A) if A.train else gold(A)
