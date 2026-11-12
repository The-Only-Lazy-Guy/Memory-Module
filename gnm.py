"""GNM: graph memory for Walker (user design 11-04). Everything is a node (label L, relation R, value B); a function
is a node too and takes functions and nodes, gives nodes. Signals fire functions; a filter decides what reaches each
recipient; text becomes a node once, when it arrives.

  python gnm.py --fm --init nt2.pt      # codec stage 2, demos, then a fresh world it lives in by itself
"""
import argparse
import copy
import math
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
RM = 16                         # rows a place's label keeps (what names a thing: 4-10 tokens)
JOIN = .7                       # the same thing: every naming row of each found in the other this well


class Graph:
    """Places, each a label and a log: what was said of it, in order (v3, 11-12: nothing in a log is changed or
    removed; what a thing is worth now is read from its log). The label is a matrix, one row per token that names the
    thing: a place is found by alignment -- every row of its label finds its row in what is asked, and the worst row
    decides -- so 'the page of Torka' and '... of Orka' are two places whatever their means are (ws44: as means they
    were one, 443 facts in 126 places). L / R: the pooled label and the relation (the relation walk: R[i] . L[j]).
    Edges start as "everything to everything": i -> j is cut for good when a kept neighbour k of i is closer to j than
    i is (i reaches j through k)."""

    def __init__(self, d, beam=8, cap=1024):
        self.d, self.beam = d, beam
        self.L = np.zeros((cap, d), dtype=np.float32)
        self.R = np.zeros((cap, d), dtype=np.float32)
        self.M = np.zeros((cap, RM, d), dtype=np.float32)   # each place's label rows ...
        self.Mn = np.zeros(cap, dtype=np.int64)             # ... and how many it has
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

    def found_in(self, Q):
        """every place against what a node says (Q: its rows): how well the place's worst label row is found in it"""
        s = (self.M[:self.n] @ Q.T).max(-1)
        s[np.arange(RM)[None] >= self.Mn[:self.n, None]] = 1.
        return s.min(-1)

    def found_at(self, Q):
        """every place's label laid along what a node says (Q: its rows), row on row, in order: how well its worst
        row is met where it lies best, and where that is (the first row). A name is found where it stands, as it
        stands -- its words scattered over the node are not it. -> (how well, first row), per place"""
        n, m = self.n, len(Q)
        S = self.M[:n] @ Q.T
        acc = np.ones((n, m), dtype=np.float32)
        for i in range(RM):
            sh = np.full((n, m), -1., dtype=np.float32)
            if i < m:
                sh[:, :m - i] = S[:, i, i:]
            acc = np.minimum(acc, np.where((i < self.Mn[:n])[:, None], sh, 1.))
        return acc.max(1), acc.argmax(1)

    def near(self, Q, k):
        """the places named in what a node says, best first: (1 - how well, place, first row, last row of the name)"""
        if self.n == 0:
            return []
        s, a = self.found_at(Q)
        return [(float(1 - s[i]), int(i), int(a[i]), int(a[i] + self.Mn[i] - 1)) for i in np.argsort(-s)[:k]]

    def same(self, Q, key):
        """the place that is what a node names (Q: all its rows, key: those that name its thing): the place's label
        found in the node and the node's naming rows found in the label; -1 = no such place yet"""
        if self.n == 0:
            return -1
        s = self.M[:self.n] @ key.T
        s[np.arange(RM)[None] >= self.Mn[:self.n, None]] = -1.
        both = np.minimum(self.found_in(Q), s.max(1).min(-1))
        p = int(both.argmax())
        return p if both[p] > JOIN else -1

    def cut(self, cands):
        """the rule: go through candidates closest first, keep one unless a kept one is already closer to it"""
        kept = []
        for d, j in sorted(cands):
            if all(np.linalg.norm(self.L[k] - self.L[j]) >= d for k in kept):
                kept.append(j)
        return kept

    def add_place(self, label, rows):
        """a new place: its pooled label, and its label rows (the rows that name the thing)"""
        if self.n == len(self.L):
            self.L, self.R = np.vstack([self.L, np.zeros_like(self.L)]), np.vstack([self.R, np.zeros_like(self.R)])
            self.M, self.Mn = np.concatenate([self.M, np.zeros_like(self.M)]), np.concatenate([self.Mn, np.zeros_like(self.Mn)])
        cands = self.search(label)
        i, self.n = self.n, self.n + 1
        self.L[i] = label
        self.M[i, :len(rows[:RM])], self.Mn[i] = rows[:RM], len(rows[:RM])
        self.chain.append([])
        self.adj.append(self.cut(cands))
        for j in self.adj[i]:                          # the way back (cutting here left new places with no way in)
            self.adj[j].append(i)
        return i


PULL = 1.5                      # a label is pulled until the want meets it this far above 1 (relevant): room for wants it
#                                 has not met (fm31 @50, dm29: pulled to 1.1, in a fresh world every statement rested --
#                                 new names put the gates just under 1); pushes stay at .9
ROUTE_T = .2                    # a try's routing: every node, by its gate at this temperature (x --explore)


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
        self.rel = nn.Sequential(nn.Linear(c, 512), nn.GELU(), nn.Linear(512, d))
        self.how = nn.Linear(c, 1)
        self.rr = nn.TransformerEncoder(nn.TransformerEncoderLayer(c, 8, 512, batch_first=True, activation="gelu"), 2)
        self.rpos = nn.Parameter(torch.randn(96, c) * .02)
        self.ro = nn.Linear(c, c)                      # the re-reader's change (zero at first: it starts as it is)
        nn.init.zeros_(self.ro.weight), nn.init.zeros_(self.ro.bias)
        self.lg = nn.Sequential(nn.Linear(c, 256), nn.GELU(), nn.Linear(256, 1))   # the label as the encoder makes it: the
        self.lp = nn.Sequential(nn.Linear(c, 512), nn.GELU(), nn.Linear(512, d))   # rows that name the thing, pooled
        self.mr = nn.Sequential(nn.Linear(c, 512), nn.GELU(), nn.Linear(512, d))   # the label as a matrix: a row per token
        self.mg = nn.Sequential(nn.Linear(c, 256), nn.GELU(), nn.Linear(256, 1))   # ... and how much each names the thing

    def rows(self, zs):
        """the label as a matrix (v3, 11-12): every token's own row, and how much it names the thing the node is about
        (0 where there is no row) -> rows (n, T, d), names (n, T), mask (n, T)"""
        x, m = pad(zs)
        return F.normalize(self.mr(x), dim=-1), torch.sigmoid(self.mg(x)).squeeze(-1) * m, m

    def reread(self, z):
        """a node a function made (pieces from several texts, each still carrying its sentence) read again, latent, as
        the encoder would read those words: what goes on from a function is a node like any other"""
        y = self.rr((z[:96] + self.rpos[:len(z[:96])])[None])[0]
        return F.layer_norm(torch.cat([z[:96] + self.ro(y), z[96:]]), (z.shape[-1],))

    def forward(self, zs):
        y = self.p([[z] for z in zs])
        x, m = pad(zs)                                 # the label: the rows that name the thing, pooled, as the encoder
        g_ = torch.sigmoid(self.lg(x)).squeeze(-1) * m   # makes it (lp11: the CLS pool put the same fact .51 apart at
        lab = F.normalize(self.lp((g_[..., None] * x).sum(1) / g_.sum(1, keepdim=True).clamp(min=1e-6)), dim=-1)   # codec
        return lab, F.normalize(self.rel(y[:, 1]), dim=-1) * torch.sigmoid(self.how(y[:, 2])) / .9


def align(A, ga, B, gb):
    """every row of a finds its row in b, and the worst one decides. A (n, T, d) rows, ga (n, T) how much each names
    its thing (a row that names nothing counts as found); B (m, S, d), gb (m, S) how much each row of b is one to be
    found (its mask: any row it has; its names: its naming rows only). -> (n, m)"""
    s = torch.einsum("ntd,msd->nmts", A, B) - 2 * (1 - gb)[None, :, None, :]
    return (ga[:, None] * s.max(-1).values + (1 - ga[:, None])).min(-1).values


def apart_at(x, same, lo=.5, hi=.9):
    """scores of pairs: those of one thing above hi, of two things under lo (only the pairs that are not there yet)"""
    up, down = F.relu(hi - x)[same], F.relu(x - lo)[~same]
    return up.sum() / max(1, int((up > 0).sum())) + down.sum() / max(1, int((down > 0).sum()))


class Interpreter(nn.Module):
    """what every node runs when it acts: one network for all. It reads the acting node's program (its value: rows)
    and applies its action (a low-rank matrix pair that multiplies into and adds to the states it reads, every layer),
    the signal, and everything it perceives (each input with how salient it is and how far the traversal went to reach
    it; layer k reads what lies up to k hops away; each input's own action, from its value, acts on the states too, as
    much as it is salient: a rule in view changes how the statement is read). An output node is made of up to S
    pieces, each a span taken as it is from one of the inputs or up to G latents it makes; a function given back is K
    latents. Also: how many outputs, where each goes, whether one is a function, its own guess of how wrong it is"""

    def __init__(self, d=128, c=256, K=16, M=2, S=6, G=8, R=16, L=96, r=8, layers=5, lm=768):
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
        self.pin = nn.Linear(c, c)                             # which input a span is taken from
        self.sp = nn.Parameter(torch.tensor(1.))               # ... as much as it is salient (tr24: the right fact at
        #                                                        input 1 every lookup, the number copied from input 2/7/8)
        self.gkeep = nn.Linear(c, 1)                           # how many latents a made piece keeps
        self.es, self.ee, self.ex = nn.Linear(c, c), nn.Linear(c, c), nn.Linear(c, c)   # in place of an input: which part
        #                                                        of it (first, last token) the output takes the place of
        self.lmh, self.lms = nn.Linear(lm, c), nn.Linear(3, c)   # what the LM made of the inputs read as one text (user
        for m_ in (self.lmh, self.lms):                          # 11-12): its state at each token, and its logits -- how
            nn.init.zeros_(m_.weight), nn.init.zeros_(m_.bias)   # expected the token was, how much it would say it next
        #                                                        (zero at first: it starts as it was without them)
        self.K, self.M, self.S, self.G, self.R, self.L, self.r = K, M, S, G, R, L, r

    def action_of(self, z):
        """a node's action (U, V) from its value"""
        a = self.act(z.mean(0)).view(2, -1, self.r)
        return a[0], a[1]

    def forward(self, ins, q, program, action, v=1., hops=None, sal=None, acts=None, copy=None, lm=None):
        c, dev = self.pos.shape[-1], q.device
        toks = [z[:self.L] for z in ins[:self.R]]
        hops = (list(hops) if hops is not None else [0] * len(toks))[:len(toks)]
        sal = (list(sal) if sal is not None else [torch.ones((), device=dev)] * len(toks))[:len(toks)]
        head = [self.codeq, self.segq, self.genq, F.layer_norm(program, (c,)) + self.me, self.wq(q)[None],
                self.wv(torch.tensor([float(v)], device=dev))[None]]
        nh = sum(len(x) for x in head)
        lm = (list(lm) if lm is not None else [])[:len(toks)] + [None] * len(toks)
        parts = head + [z + self.pos[:len(z)] + self.hop[min(hp, 3)] + sv * self.salv +
                        (self.lmh(f[:, :-3]) + self.lms(f[:, -3:]) if f is not None else 0) for z, hp, sv, f in zip(toks, hops, sal, lm)]
        reach = torch.tensor([-1] * nh + [hp for z, hp in zip(toks, hops) for _ in range(len(z))], device=dev)
        acts = list(acts)[:len(toks)] if acts is not None else [self.action_of(z) for z in toks]
        h = torch.cat(parts)
        total = 1 + sum(float(sv) for sv in sal)
        if acts:                                       # the inputs' actions, stacked: (N, c, r) and their salience
            Us, Vs = torch.stack([u for u, _ in acts]), torch.stack([v_ for _, v_ in acts])
            ws = torch.stack([torch.as_tensor(sv, dtype=h.dtype, device=dev).reshape(()) for sv in sal[:len(acts)]]) / total
        for li, layer in enumerate(self.layers):       # layer li reads what lies up to li hops away (and all its own)
            h = layer(h[None], src_key_padding_mask=(reach > li)[None])[0]
            d = (h @ action[0]) @ action[1].T if action is not None else 0   # the acting node's action multiplies into
            if acts:                                   # and adds to the states; so does every input's, as much as it
                d = d + torch.einsum("n,ntr,ncr->tc", ws, torch.einsum("tc,ncr->ntr", h, Us), Vs)   # is salient (averaged)
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
            o["start"] = o["start"] + (self.pin(hs) @ each.T)[..., o["owner"]]     # + which input
            ls = torch.log(torch.stack([torch.as_tensor(sv, dtype=h.dtype, device=dev).reshape(()) for sv in sal]).clamp_min(1e-4))
            o["start"] = o["start"] + self.sp * ls[o["owner"]]   # + how salient it is (the copy's loss teaches perception too)
            if copy is not None:                       # a function's program is not words to copy
                cm = torch.tensor(list(copy)[:len(toks)], dtype=torch.bool, device=dev)
                o["start"] = o["start"].masked_fill(~cm[o["owner"]], -1e9)
            tk = self.wp(g) @ o["keys"].T              # where: in place of input r by its tokens, not their average
            wr = torch.stack([tk[:, o["owner"] == r].logsumexp(-1) - math.log(int((o["owner"] == r).sum()))
                              for r in range(len(toks))], -1)   # (11-12, ws44: 'the page of Torka' / '... Orka' as means
            o["where"] = torch.cat([self.where(g), wr], -1)     # are one fact; 443 facts in 126 places, held 13%)
        else:
            o["where"] = self.where(g)
        o["isfn"], o["g"] = self.isfn(g).squeeze(-1), g
        return o

    def part_start(self, o, j, r):
        """the first token of the part of input r that output j takes the place of (11-12, fm47: a question rewritten
        as three pieces -- before, the value, after -- stopped after the first; the node only says which part changes)"""
        return (self.es(o["g"][j]) @ o["keys"].T).masked_fill(o["owner"] != r, -1e9)

    def part_end(self, o, j, st):
        """... and its last: at st or after, in the same input"""
        lg = (self.ee(o["g"][j]) + self.ex(o["ctx"][st])) @ o["keys"].T
        bad = (torch.arange(len(lg), device=lg.device) < st) | (o["owner"] != o["owner"][st])
        return lg.masked_fill(bad, -1e9)

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
        self.pend, self.last_ctx = None, None          # the update writer's change, not yet committed; what it last did
        self.v = {}                                    # each weight's running gradient size (its own scale, as Adam's)

    def w(self, name):
        """a weight as it is now: committed plus the pending change (which carries the writer's graph: the next use's
        loss judges the change, and trains the writer)"""
        x = getattr(self, name)
        return x + self.pend[name] if self.pend is not None and name in self.pend else x


class UpdateWriter(nn.Module):
    """user 11-05: a model that reads the nodes AND the gradient and writes the edited update (prototype: nodes +
    gradient learned fastest and stayed stable where gradient-only was slow and nodes-only diverged). A function's own
    weights (code rows, its action's columns, its key) are rows; with each, its gradient (scaled, sign-log); and the
    firing it learned from (the want, the inputs, its loss, exact or not). Out: a change per row = a learned step on the
    scaled gradient + what it writes (zero at first). Trained by the function's next use (learning speed)"""

    def __init__(self, c=256, d=128, D=128):
        super().__init__()
        self.c, self.d = c, d
        self.row = nn.Linear(3 * c, D)
        self.grp = nn.Parameter(torch.randn(4, D) * .02)          # code / au / av / key
        self.ctx = nn.Linear(c, D)
        self.qx = nn.Linear(d, D)
        self.sc = nn.Linear(2, D)
        self.f = nn.TransformerEncoder(nn.TransformerEncoderLayer(D, 4, 2 * D, batch_first=True, activation="gelu"), 2)
        self.out = nn.Linear(D, c)
        nn.init.zeros_(self.out.weight), nn.init.zeros_(self.out.bias)
        self.lr = nn.Parameter(torch.log(torch.full((4,), 2e-4)))   # the step on the scaled gradient (as Adam's at first)
        self.gain = nn.Parameter(torch.full((4,), math.log(1e-3)))  # how much of what it writes

    def rows(self, f, code, au, av, key):
        k = torch.zeros(1, self.c, device=key.device, dtype=key.dtype)
        k[0, :len(key)] = key
        return torch.cat([code, au.T, av.T, k])

    def forward(self, f, grads):
        dev = f.code.device
        P = self.rows(f, f.code.detach(), f.au.detach(), f.av.detach(), f.key.detach())
        G = self.rows(f, *grads)
        ids = torch.tensor([0] * len(f.code) + [1] * f.au.shape[1] + [2] * f.av.shape[1] + [3], device=dev)
        rms = G.pow(2).mean().sqrt() + 1e-12
        g = G / rms
        t = self.row(torch.cat([P, g, torch.sign(g) * torch.log1p(g.abs())], -1)) + self.grp[ids]
        ctx = []
        if f.last_ctx is not None:
            q, ins, loss, exact = f.last_ctx
            ctx = [self.qx(q)[None], self.ctx(ins), self.sc(torch.tensor([[loss, exact]], device=dev, dtype=t.dtype))]
        y = self.f(torch.cat([t] + ctx)[None])[0][:len(P)]
        dl = -torch.exp(self.lr)[ids, None] * g + torch.exp(self.gain)[ids, None] * self.out(y)
        K, r = len(f.code), f.au.shape[1]
        return {"code": dl[:K], "au": dl[K:K + r].T, "av": dl[K + r:K + 2 * r].T, "key": dl[-1, :len(f.key)]}


@torch.no_grad()
def commit(f):
    """a pending change becomes the function's weights"""
    if f.pend is not None:
        for k, x in f.pend.items():
            getattr(f, k).add_(x.detach())
        f.pend = None


def write_fns(fm):
    """after an update: every function's pending change is committed (none outlives a step of the optimizer); each
    function that acted (its weights got a gradient) gets a new change from the writer, judged by its next use"""
    for f in fm.bank.fns:
        ps = (f.code, f.au, f.av, f.key)
        grads = [x.grad for x in ps[:3]] + [None]      # the key is Adam's (routing)
        commit(f)
        if any(g is not None for g in grads):
            gr = [torch.zeros_like(x) if g is None else g.detach() for x, g in zip(ps, grads)]
            if all(bool(torch.isfinite(g).all()) for g in gr):
                for k, g in zip(("code", "au", "av", "key"), gr):   # each weight on its own scale (fm22c: one scale for
                    v = f.v.get(k)                                  # the whole function left the key's step tiny -- the
                    f.v[k] = g * g if v is None else .99 * v + .01 * g * g   # keys never moved, nothing fired living)
                gr = [g / (f.v[k].sqrt() + 1e-8) for k, g in zip(("code", "au", "av", "key"), gr)]
                f.pend = {k: x for k, x in fm.bank.writer(f, gr).items() if k != "key"}
                fm.count("functions written")
        for x in ps[:3]:
            x.grad = None


def commit_all(fm):
    for f in fm.bank.fns:
        commit(f)


def choice(lg, temp=0.):
    """a choice from logits: the likeliest, or (a try, temp > 0) one sampled as likely as the node itself finds it"""
    if not temp:
        return int(lg.argmax())
    return int(torch.multinomial(F.softmax(lg.float() / temp, -1), 1))


def assemble(net, o, j, plan=None, temp=0.):
    """output j as a node: its pieces put together (its own choices, or in demos the pieces the target is made of:
    ("span", first token, last token) / ("made", how many latents)); returns the node's value and its pieces. temp > 0:
    a try -- each choice sampled from the node's own odds (11-11: tries changed only who fires and what it takes in;
    what it makes and where it goes stayed its argmax, so a wrong slot or a wrong span was never tried otherwise)"""
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
            k = choice(kd, temp)
            if k == 0:
                break
            if k == 1:
                st_ = choice(o["start"][j, s_], temp)
                pc = ("span", st_, choice(net.end_logits(o, j, s_, st_), temp))
            else:
                keep = torch.sigmoid(o["gkeep"][j, s_] / temp) if temp else None
                pc = ("made", max(1, int((torch.bernoulli(keep) > 0).sum() if temp else (o["gkeep"][j, s_] > 0).sum())))
        rows.append(o["raw"][pc[1]:pc[2] + 1] if pc[0] == "span" else o["gen"][j, s_, :pc[1]])
        pieces.append(pc)
    return (torch.cat(rows) if rows else o["gen"][j, 0, :1]), pieces


def plan_pieces(tid, ins, S, G, prefer=None):
    """the pieces a target text (its tokens) is made of, given the inputs ((rows, tokens or None) each): the longest
    runs found as they are in the inputs that should be read (prefer: their indices; else any input), and what is left
    (made: target positions i..j). A span is ("span", first, last, every (first, last) where the same tokens sit in
    those inputs). None when that takes more than S pieces or a made piece longer than G"""
    T, offs, at = len(tid), [], 0
    for n_rows, _ in ins:
        offs.append(at)
        at += n_rows
    every = set(range(len(ins)))
    pref = every if prefer is None else (set(prefer) & every or every)

    def runs(cls):                                     # best[i]: the longest run of the target from i in those inputs
        best = [(0, 0)] * (T + 1)
        for r in sorted(cls):
            t = ins[r][1]
            if t is None:
                continue
            nxt = [0] * (len(t) + 1)
            for i in range(T - 1, -1, -1):
                cur = [0] * (len(t) + 1)
                for a in range(len(t) - 1, -1, -1):
                    if t[a] == tid[i]:
                        cur[a] = nxt[a + 1] + 1
                        if cur[a] > best[i][0]:
                            best[i] = (cur[a], offs[r] + a)
                nxt = cur
        return best

    bp = runs(pref)
    ba = runs(every) if pref != every else bp
    pieces, i = [], 0
    while i < T:
        n, cls = (bp[i][0], pref) if bp[i][0] else (ba[i][0], every)
        if n:
            seq = list(tid[i:i + n])
            alts = [(offs[r] + a, offs[r] + a + n - 1) for r in sorted(cls) if ins[r][1] is not None
                    for a in range(len(ins[r][1]) - n + 1) if list(ins[r][1][a:a + n]) == seq][:6]
            pieces.append(("span", alts[0][0], alts[0][1], alts))
            i += n
            continue
        j = i + 1
        while j < T and bp[j][0] == 0 and ba[j][0] == 0:
            j += 1
        pieces.append(("made", i, j))
        i = j
    if len(pieces) > S or any(pc[0] == "made" and pc[2] - pc[1] > G for pc in pieces):
        return None
    return pieces


def said_as(pieces, flat):
    """pieces as what they say: a span as its tokens (None when an input's tokens are not known), a made piece as its
    size -- two outputs that say the same thing the same way are the same"""
    return [tuple(flat[pc[1]:pc[2] + 1]) if pc[0] == "span" else ("made", pc[1] if len(pc) == 2 else pc[2] - pc[1]) for pc in pieces]


def toks_of(net, ins):
    """the tokens of everything a node takes in, in a row (None where an input's tokens are not known)"""
    return [t for n in ins[:net.R] for t in (list(n.toks[:net.L]) if n.toks is not None else [None] * len(n.B[:net.L]))]


def make_out(fm, net, o, j, ins, w, temp=0., pieces=None, part=None):
    """output j as the node it becomes. Its pieces put together (its own choices -- sampled when temp > 0 -- or the
    given ones); when it goes in place of a workspace node (w), they take the place of a PART of that node and the
    rest stays as it is: A to be B, where A is a part of what is there. Which part: what it took from a log goes
    where that log's place is named in the node (the label match that found the place says where: a read, 11-12);
    anything else, where the output itself points (first, last token). `part`: the given one, counted over all the
    inputs' tokens. -> (the node's rows, read again; its pieces; the part or None; its tokens, None when not all are
    known; whether the part was its own pointing)"""
    rows, pcs = assemble(net, o, j, pieces, temp)
    flat = toks_of(net, ins)
    toks = [t for pc in pcs for t in flat[pc[1]:pc[2] + 1]] if pcs and all(pc[0] == "span" for pc in pcs) else None
    r, own = w - 2, False
    if 0 <= r < min(len(ins), net.R) and ins[r].kind == "work" and len(o["raw"]):
        x, off = ins[r], int((o["owner"] < r).sum())
        if part is None:
            src = next((ins[int(o["owner"][pc[1]])] for pc in pcs if pc[0] == "span" and ins[int(o["owner"][pc[1]])].kind == "entry"), None)
            if src is not None and src.at is not None and src.at[0] is x and src.at[2] < len(x.B[:net.L]):
                part = (off + src.at[1], off + src.at[2])
            else:
                st = choice(net.part_start(o, j, r), temp)
                part, own = (st, choice(net.part_end(o, j, st), temp)), True
        a, b = part[0] - off, part[1] - off
        rows = torch.cat([x.B[:a], rows, x.B[b + 1:]])
        toks = list(x.toks[:a]) + toks + list(x.toks[b + 1:]) if toks is not None and x.toks is not None else None
    else:
        part = None
    if toks is not None and (None in toks or len(toks) != len(rows)):
        toks = None
    return fm.heads.reread(rows), pcs, part, toks, own


def part_changed(tx, tt, others):
    """what of a node changes when it is to say tt instead of tx (tokens): (first, last) token of the part replaced,
    and the tokens that take its place. As little as differs; grown while what takes its place is still found whole in
    another input (a name comes with its 'the'); never nothing replaced, never by nothing"""
    p = s_ = 0
    while p < min(len(tx), len(tt)) and tx[p] == tt[p]:
        p += 1
    while s_ < min(len(tx), len(tt)) - p and tx[-1 - s_] == tt[-1 - s_]:
        s_ += 1
    found = lambda seq: any(t is not None and any(list(t[i:i + len(seq)]) == seq for i in range(len(t) - len(seq) + 1)) for t in others)
    new = lambda: list(tt[p:len(tt) - s_])
    if p + s_ < len(tt) and found(new()):
        while p > 0 and found([tt[p - 1]] + new()):
            p -= 1
        while s_ > 0 and found(new() + [tt[len(tt) - s_]]):
            s_ -= 1
    while (p + s_ >= len(tt) or p + s_ >= len(tx)) and (p or s_):
        p, s_ = (p - 1, s_) if p else (p, s_ - 1)
    return p, len(tx) - s_ - 1, new()


def say_loss(fm, A, net, o, j, target, order, prefer=None, dest=None):
    """how wrong output j is as the text it should say: which pieces it is made of (kind, where each span starts and
    ends, how many latents a made piece keeps and how close they are to what the encoder makes of those words), and
    what the assembled node says. dest: the input it goes in place of; when that is a workspace node the target is made
    by changing a PART of it and the pieces are what takes the part's place. A read: the part is where the log it
    reads from is named (found by the label match; nothing to learn); else the part is as little as differs
    (part_changed) and the output learns to point at it. Returns (loss, exact, the node, the pieces, the target's
    tokens, the target as spelled, the rest) or None when the target cannot be made of the pieces an output has"""
    ins_t = [(len(n.B[:net.L]), n.toks[:net.L] if n.toks is not None else None) for n in order[:net.R]]
    offs = np.cumsum([0] + [n for n, _ in ins_t])
    x = order[dest] if dest is not None and dest < len(ins_t) and order[dest].kind == "work" and order[dest].toks is not None \
        and len(order[dest].toks) <= net.L else None
    others = [t for r, (_, t) in enumerate(ins_t) if r != dest and (prefer is None or r in set(prefer))]
    log = next((order[r] for r in (prefer or []) if r < len(ins_t) and order[r].kind == "entry" and order[r].at is not None
                and order[r].at[0] is x), None) if x is not None else None   # the log this step reads (the teacher's first)
    best = None
    for spelled in dict.fromkeys([D(target), " " + D(target).lstrip()]):   # as written, or as it reads inside a
        full = fm.tr.tok(spelled)["input_ids"]                             # sentence: the one more of it is copied in
        a, b, tid_, found = 0, 0, full, False
        if x is not None:
            tx = list(x.toks)
            if log is not None:                        # where the log's place is named: is the target the node with
                a, b = log.at[1], log.at[2]            # that part changed?
                e_ = len(tx) - 1 - b
                found = 0 <= a <= b < len(tx) and len(full) - e_ > a and tx[:a] == full[:a] and tx[b + 1:] == full[len(full) - e_:]
                tid_ = full[a:len(full) - e_]
            if not found:
                a, b, tid_ = part_changed(tx, full, others)
        plan_ = plan_pieces(tid_, ins_t, net.S, net.G, prefer)
        if plan_ is not None:                                              # (more copied FROM THE INPUTS IT SHOULD
            own_in = lambda a_: prefer is None or int(np.searchsorted(offs, a_, side="right") - 1) in set(prefer)  # READ
            copied = (sum(pc[2] - pc[1] + 1 for pc in plan_ if pc[0] == "span" and own_in(pc[1])),   # (tr21); then more
                      sum(pc[2] - pc[1] + 1 for pc in plan_ if pc[0] == "span"), -len(plan_))   # copied, fewer pieces
            copied = ((-len(tid_),) if x is not None else ()) + copied   # (a part: the smallest change first -- ' What
            if best is None or copied > best[0]:                           # is ...' for 'What is ...' changes it all)
                best = (copied, spelled, full, plan_, (a, b), tid_, found)
    if best is None:
        fm.count("cannot be made of pieces")
        return None
    _, target, tid, plan, part, inner, found = best
    if log is not None:
        fm.count("read: place found where it is named", int(found)), fm.count("read: asked")
    if found:                                          # what is read from a log is taken from a log (a value that also
        owner = lambda a_: order[int(np.searchsorted(offs, a_, side="right") - 1)].kind   # stands in the question is
        plan = [pc if pc[0] != "span" else (lambda al: ("span", al[0][0], al[0][1], al))(    # not the one to copy: the
            [ab for ab in pc[3] if owner(ab[0]) == "entry"] or pc[3]) for pc in plan]        # larger of 19 and 4 = 19)
    dev = o["kind"].device
    with torch.no_grad():
        zt, mt = fm.tr.content([target])
        zt = zt[0][mt[0]]
    L, at = 0, part[0] if x is not None else 0         # (at: where the pieces sit in the target)
    for s_ in range(min(len(plan) + 1, net.S)):
        k = 0 if s_ == len(plan) else 1 if plan[s_][0] == "span" else 2
        L = L + F.cross_entropy(o["kind"][j, s_][None], torch.tensor([k], device=dev))
        if k == 1:                                     # any of the places the same tokens sit is right
            ls = o["start"][j, s_].log_softmax(-1)
            L = L - torch.logsumexp(torch.stack([ls[a] + net.end_logits(o, j, s_, a).log_softmax(-1)[b]
                                                 for a, b in plan[s_][3]]), 0)
        elif k == 2:
            i0, i1 = plan[s_][1], plan[s_][2]
            L = L + F.binary_cross_entropy_with_logits(o["gkeep"][j, s_], (torch.arange(net.G, device=dev) < i1 - i0).float())
            if len(zt) == len(tid):                    # what the encoder makes of those words, there
                L = L + F.mse_loss(o["gen"][j, s_, :i1 - i0], zt[at + i0:at + i1]) * A.mse
    ed = (int(offs[dest]) + part[0], int(offs[dest]) + part[1]) if x is not None else None
    if ed is not None and not found:                   # which part of the node it changes: its own pointing
        L = L + F.cross_entropy(net.part_start(o, j, dest)[None], torch.tensor([ed[0]], device=dev)) + \
            F.cross_entropy(net.part_end(o, j, ed[0])[None], torch.tensor([ed[1]], device=dev))
    want_pcs = [(pc[0], pc[1], pc[2]) if pc[0] == "span" else ("made", pc[2] - pc[1]) for pc in plan]
    w = 2 + dest if x is not None else -1
    z = make_out(fm, net, o, j, order, w, pieces=want_pcs, part=ed)[0]   # (read again, as the encoder reads those words)
    if len(zt) == len(z):
        L = L + F.mse_loss(z, zt) * A.mse
    dl = fm.tr.decode_loss(z[None], torch.ones(1, len(z), dtype=torch.bool, device=dev), [target])
    flat = toks_of(net, order)
    says = lambda out: said_as(out[1], flat) == said_as(want_pcs, flat) and out[2] == ed   # (does an output say the target?)
    with torch.no_grad():                              # exact = its own pieces say what the target's say (from wherever
        mine = make_out(fm, net, o, j, order, w)       # those tokens sit), on the same part, and what it makes says it
        same = says(mine) and None not in [x_ for p_ in said_as(mine[1], flat) if p_ and p_[0] != "made" for x_ in p_]
        exact = same and float(dl) < A.exact
    return L + dl, exact, z, plan, tid, target, (mine, same, says, part if x is not None else None, ed if not found else None, inner)


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
        self.kq = nn.Sequential(nn.Linear(c, 256), nn.GELU(), nn.Linear(256, d))   # ... and which wants it answers:
        nn.init.zeros_(self.kq[-1].weight), nn.init.zeros_(self.kq[-1].bias)     # from its value, starting at its label
        self.writer = UpdateWriter(c, d)                # writes the functions' own updates
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
            return self.fns[n.fn].w("key")
        m = n.B.mean(0)
        return F.normalize(n.L + self.kq(m), dim=-1) * F.softplus(self.wk(m)).squeeze(-1)

    def gate(self, q, nodes):                          # every want against every node's label: > 1 = relevant
        keys = [self.fns[n.fn].w("key") if n.kind == "fn" else None for n in nodes]
        rd = [i for i, n in enumerate(nodes) if n.kind != "fn"]
        if rd:                                         # the read nodes' together (one pass)
            m = torch.stack([nodes[i].B.mean(0) for i in rd])
            w, dk = F.softplus(self.wk(m)).squeeze(-1), self.kq(m)
            for k_, i in enumerate(rd):
                keys[i] = F.normalize(nodes[i].L + dk[k_], dim=-1) * w[k_]
        return q @ torch.stack(keys).T

    def program(self, n):
        """what the interpreter reads as the acting node's program: its value"""
        return self.fns[n.fn].w("code") if n.kind == "fn" else n.B[:self.interp.L]

    def action(self, n, program=None):
        """the node's action A (U, V): from its value; a made node adds its own"""
        if n.kind == "fn":
            f = self.fns[n.fn]
            U, V = self.interp.action_of(f.w("code") if program is None else program)
            return U + f.w("au"), V + f.w("av")
        return self.interp.action_of(n.B if program is None else program)

    def stats_of(self, n):                             # whose usual error
        return self.fns[n.fn] if n.kind == "fn" else self

    def calc_of(self, n):
        return self.fns[n.fn].calc if n.kind == "fn" else None

    @torch.no_grad()
    def lm_read(self, nodes):
        """the LM itself reads what the acting node takes in, as one text: the memory's logs first, the workspace nodes
        (the question) last, so its states there have read the logs (user 11-12: the LM processes it too, and the
        interpreter sees the LM's logits). -> for each input, a row per token: the LM's state there, how expected the
        token was (log p, from the logits before it), how much the LM would say it next (the logits after the last
        token), and whether its token is known. A node whose words are not known is read through the decoder's way
        in (up); a function is not text: None"""
        tr, L = self.tr, self.interp.L
        dev = self.interp.pos.device
        seq = [i for i, n in enumerate(nodes) if n.kind == "entry"] + [i for i, n in enumerate(nodes) if n.kind == "work"]
        embs, ids, own = [tr.lm.wte(torch.tensor([tr.tok.eos_token_id], device=dev))], [-1], [(-1, 0)]
        for i in seq:
            n = nodes[i]
            z = n.B[:L]
            known = n.toks is not None and len(n.toks) == len(n.B)
            t = list(n.toks[:len(z)]) if known else [-1] * len(z)
            embs.append(tr.lm.wte(torch.tensor(t, device=dev)) if known else tr.up(z))
            ids += t
            own += [(i, k) for k in range(len(z))]
        x, ids, own = torch.cat(embs)[-1000:], ids[-1000:], own[-1000:]
        h = tr.lm(inputs_embeds=x[None]).last_hidden_state[0]
        lp = (h @ tr.lm.wte.weight.T).log_softmax(-1)
        idt = torch.tensor([max(t, 0) for t in ids], device=dev)
        has = torch.tensor([float(t >= 0) for t in ids], device=dev)
        was = torch.cat([lp.new_zeros(1), lp[:-1].gather(1, idt[1:, None]).squeeze(1)]) * has
        f = torch.cat([tr.norm(h), torch.stack([was / 10, lp[-1][idt] * has / 10, has], -1)], -1)
        out = [None] * len(nodes)
        for i in seq:
            rows = [r for r, (i_, _) in enumerate(own) if i_ == i]
            g = f.new_zeros(len(nodes[i].B[:L]), f.shape[-1])
            g[[own[r][1] for r in rows]] = f[rows]
            out[i] = g
        return out

    def run(self, actor, ins, q, v=1., hops=None, sal=None, program=None):
        """the actor acts: the interpreter reads its program and applies its action, and every input's own action;
        it sees what the LM made of those inputs"""
        R = self.interp.R
        return self.interp([n.B for n in ins], q, self.program(actor) if program is None else program,
                           self.action(actor, program), v, hops, sal, acts=[self.action(n) for n in ins[:R]],
                           copy=[n.kind != "fn" for n in ins[:R]], lm=self.lm_read(ins[:R]) if self.see_lm else None)

    def make(self, key, like=None, code=None, scope=.9):
        """a new node that acts (a function): made like another acting node (its program and its own action; a read
        node: its value as the program), labelled with the want it answers"""
        f = FnNode(self.interp.K, self.d, self.c, self.interp.r).to(key.device)
        with torch.no_grad():
            if like is not None and like.kind == "fn":
                src = self.fns[like.fn]
                f.code.copy_(src.w("code")), f.au.copy_(src.w("au")), f.av.copy_(src.w("av"))
            elif like is not None:
                rows = like.B.detach()
                f.code.copy_(rows[torch.arange(self.interp.K, device=rows.device) % len(rows)])
            if code is not None:
                f.code.copy_(code)
            f.key.copy_(F.normalize(key, dim=0) / scope)
        self.fns.append(f)
        return len(self.fns) - 1

    def as_node(self, t):                              # made node t's value (being read does not change it)
        return F.layer_norm(self.fns[t].w("code").detach(), (self.c,))


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
        self.qry = nn.Sequential(nn.Linear(3 * d + 2 + c, c), nn.GELU(), nn.Linear(c, c))   # + what the signal is about
        self.f = nn.TransformerEncoder(nn.TransformerEncoderLayer(c, 8, 512, batch_first=True, activation="gelu"), 2)
        self.o = nn.Linear(c, 1)
        self.dist = nn.Linear(2, c)                         # how near the walk found it (its label against the signal's)
        self.nw = nn.Parameter(torch.tensor(3.))            # ... and how much that nearness counts, as it is

    def forward(self, key, code, q, sign, strength, cands, about=None):
        """about: the label of the node the signal is about (fm23 trace: the want, clustered by purpose, does not say WHICH
        fact a lookup asks for -- the right fact scored +.9, others about the same subject +1.0). Each thing a walk
        brought comes with how near its label was (tr24: every entry scored -.6 to -.8 while the label walk had the
        right fact at .27-.50 and the next at .8-1.0: perception had to learn the label match again, and had not)"""
        s = torch.tensor([sign, strength], dtype=q.dtype, device=q.device)
        qv = self.qry(torch.cat([key, q, s, code.mean(0), about if about is not None else torch.zeros_like(q)]))
        near = torch.tensor([[1. - x[4], 1.] if len(x) > 4 and x[4] is not None else [0., 0.] for x in cands],
                            dtype=q.dtype, device=q.device)
        cv = torch.stack([self.cand(torch.cat([x[0].mean(0), x[1]])) + self.kind[x[2]] + self.age[min(x[3], 7)] for x in cands])
        cv = cv + self.dist(near)
        return self.o(self.f(torch.cat([qv[None], cv])[None])[0, 1:]).squeeze(-1) + self.nw * near[:, 0] * near[:, 1]


class Node:
    """a node in the pool: its value B, label L, relation R, and its action A (Acting.action: from B, plus a made
    node's own correction); where it lives (a memory entry (place, index), the workspace, or a made node that acts);
    the node that asked for it (parent). Any node can act: a signal's want meets its label (Acting.key)"""

    def __init__(self, B, L, R, kind="work", place=-1, idx=-1, parent=None, fn=-1):
        self.B, self.L, self.R, self.kind, self.place, self.idx, self.parent, self.fn = B, L, R, kind, place, idx, parent, fn
        self.text = None                               # the text it was read from (traces only)
        self.toks = None                               # the token each row of B is (known when read from text, or in demos)
        self.near = None                               # how far the walk that brought it was (its label, from the signal's)
        self.at = None                                 # a log: (the node, first row, last row) where its place is named


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
        object.__setattr__(bank, "tr", tr)             # (the acting side reads through the LM; not one of its own parts)
        bank.see_lm = True                             # (--lm_read 0: without what the LM made of the inputs, to tell its share)
        self.dev = next(heads.parameters()).device
        self.g, self.signals, self.t, self.explore = Graph(bank.d), [], 0, 0.
        self.budget, self.look, self.newest, self.kappa, self.same = budget, look, newest, kappa, same
        self.stats, self.firings, self.show, self.capture, self.said, self.trace, self.walks = {}, [], False, None, [], False, []
        self.fit = {}                                  # (function, purpose) -> how often it did that exactly (labels)
        self.calc_on, self.news = True, []             # the calculator's own body answers (False: removed, its mimic does)
        self.made, self.frozen = False, False
        self.routes = []                               # the demos' routing examples (signal's nodes, sign, purpose)
        self.nones, self.rests = [], []                # ... and what asked nothing there; what this run rested on
        self.shown = {}                                # purpose -> the demos' steps as runs (skill_replay)
        self.rates = {}                                # episode kind -> how often its own run is right (surprise)
        self.stage, self.assign = "free", {}           # the training stage; purpose -> the node that learned it          # a node that acts was made (the optimizer takes it in); frozen:
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

    def log(self, p):
        """a place as it is read: its log, the newest entries in the order they were said, as one node (the order is
        the rows' own: what was said last is read last)"""
        es = self.g.chain[p][-self.newest:]
        n = Node(torch.cat([e[0] for e in es]), torch.from_numpy(self.g.L[p]).to(self.dev), torch.from_numpy(self.g.R[p]).to(self.dev),
                 "entry", p, -1)
        n.toks = [t for e in es for t in e[2]] if all(len(e) > 2 and e[2] is not None for e in es) else None
        return n

    @torch.no_grad()
    def label_rows(self, z):
        """a node's label rows as the graph keeps them: all of them, and those that name its thing (at least one)"""
        r, g, _ = self.heads.rows([z])
        r, g = r[0].float(), g[0]
        i = (g > .5).nonzero().flatten()               # (one run of rows, first to last: a name is found where it stands)
        key = r[int(i[0]):int(i[-1]) + 1] if len(i) else r[g.argmax()][None]
        return r.cpu().numpy(), key.cpu().numpy()

    def fnode(self, t):
        return Node(self.bank.as_node(t), self.bank.fns[t].w("key").detach(), torch.zeros(self.bank.d, device=self.dev), "fn", fn=t)

    def name(self, n):
        if n.kind == "fn":
            c = self.bank.fns[n.fn].calc
            return f"function {n.fn}" + (f" (calculator {c})" if c else "")
        return (f"log {n.place} " if n.kind == "entry" else "node ") + repr(self.text_of(n)[-40:] if n.kind == "entry" else self.text_of(n)[:40])

    def text_of(self, n):
        """what a node says: the text it was read from, its tokens when they are known (no decoding), else decoded"""
        if n.text is not None:
            return n.text
        if n.toks is not None:
            return self.tr.tok.decode(list(n.toks)).strip()
        return self.say(n)

    def say(self, n):
        return self.tr.decode(n.B[None], torch.ones(1, len(n.B), dtype=torch.bool, device=self.dev))[0].strip()

    # --- what is near a signal (everything is visible; two walks order the memory: by label -- the places whose
    #     label rows are all found in what the signal's node says -- and by its relation, what it relates to: the rule
    #     over it, what it is the same as). A place is read as its log
    def cands(self, sg):
        out, seen = [(n, 0, 0) for n in sg.about], {n.place for n in sg.about if n.kind == "entry"}
        x = sg.about[0]
        self.walks = []

        def bring(p, kind, d_=None, at=None):
            if p not in seen:
                seen.add(p)
                e_ = self.log(p)
                e_.near, e_.at = d_, at                # (the label walk: how well its place is named in the signal's
                out.append((e_, kind, 0))              # node, and where)
        if self.g.n:
            Q = self.label_rows(x.B)[0]
            sc, a0 = self.g.found_at(Q)
            for n in sg.about[1:]:                     # a log the signal brings itself (a fact just stated): where it
                if n.kind == "entry" and n.place < self.g.n:   # is named in the signal's node
                    n.near, n.at = float(1 - sc[n.place]), (x, int(a0[n.place]), int(a0[n.place] + self.g.Mn[n.place] - 1))
            found = self.g.near(Q, self.look)
            self.walks.append((1, [(d_, p) for d_, p, _, _ in found]))
            for d_, p, a_, b_ in found:
                bring(p, 1, d_, (x, a_, b_))
            if x.R is not None and float(x.R.norm()) >= .3:   # (tr1: a direction from a near-zero R walked to the same
                found = self.g.search(F.normalize(x.R, dim=0).detach().float().cpu().numpy())[:self.look]   # few places)
                self.walks.append((3, found))
                for _, p in found:
                    bring(p, 3)
            for _, p in self.walks[0][1][:2]:          # hop 2: what the nearest places relate to
                r = self.g.R[p]
                if float(np.linalg.norm(r)) >= .3:
                    for _, p2 in self.g.search(r / np.linalg.norm(r))[:2]:
                        bring(p2, 4)
        out += [(self.fnode(t), 2, 0) for t in range(len(self.bank.fns))]
        return out

    def around(self, actor, C):
        """what the acting node itself relates to joins what is near the signal (the traversal from it)"""
        if actor.kind == "fn" or not self.g.n or float(actor.R.norm()) < .3:
            return C
        seen = {n.place for n, _, _ in C if n.kind == "entry"}
        for _, p in self.g.search(F.normalize(actor.R, dim=0).detach().float().cpu().numpy())[:self.look]:
            if p not in seen:
                seen.add(p)
                C.append((self.log(p), 3, 0))
        return C

    def perceive(self, actor, q, sg, C, keep=(), explore=0.):
        """what the acting node takes in: the signal's nodes and the most salient of the rest (up to R), each with its
        salience; returns (nodes, hops, salience, all logits)"""
        net = self.bank.interp
        ab = sg.about[0].L if sg.about[0].L is not None else None
        lg = self.filt(self.bank.key(actor), self.bank.program(actor), q, sg.sign, 0., [(n.B, n.L, k, a, n.near) for n, k, a in C],
                       about=ab.detach() if ab is not None and ab.shape == q.shape else None)
        score = lg + torch.randn_like(lg) * explore if explore else lg
        first = [i for i, (n, k, _) in enumerate(C) if k == 0 or i in keep]
        rest = [i for i in sorted(range(len(C)), key=lambda i: -float(score[i])) if i not in first]
        pick = sorted(first + rest[:max(0, net.R - len(first))])
        HOP = {0: 0, 1: 1, 2: 1, 3: 1, 4: 2}
        return [C[i][0] for i in pick], [HOP[C[i][1]] for i in pick], [torch.sigmoid(lg[i]) for i in pick], lg, pick

    def choose(self, q, C):
        """which node acts: the one whose label the want meets (any node near the signal, or a made one); -1 = rest"""
        s = self.bank.gate(q[None], [n for n, _, _ in C])[0]
        if not bool(torch.isfinite(s).all()):          # (fm20 step 531: a gate not finite crashed a sampled try)
            self.count("gate not finite")
            if self.stats["gate not finite"] <= 3:
                bad = [i for i in range(len(s)) if not bool(torch.isfinite(s[i]))]
                print(f"gate not finite: want finite {bool(torch.isfinite(q).all())}, nodes " + ", ".join(
                    f"{C[i][0].kind}{C[i][0].fn if C[i][0].kind == 'fn' else ''} (value finite {bool(torch.isfinite(C[i][0].B).all())}, "
                    f"label finite {C[i][0].L is not None and bool(torch.isfinite(C[i][0].L).all())})" for i in bad[:6]), flush=True)
            s = torch.nan_to_num(s, nan=-1e9, posinf=-1e9, neginf=-1e9)
        if self.explore:                               # a try: any node, as likely as its label meets the want (tr29: 284
            s1 = torch.cat([s, s.new_ones(1)])         # of 297 statements in a fresh world rested, and so did every try;
            a = int(torch.multinomial(F.softmax(s1 / (ROUTE_T * self.explore), -1), 1))   # fm39 @100: tries kept only the
            return -1 if a == len(s) else a            # nodes over 1 -- function 6 at 1.51 alone fired on everything) --
        #                                                or rest, as likely as a label that meets the want exactly (in48:
        #                                                a try could not rest; every right try ended on firings that
        #                                                changed nothing, those were imitated, and an answered value kept
        #                                                firing: '1 9' right after one firing, then '3 3', '3')
        return int(s.argmax()) if float(s.max()) > 1 else -1

    # --- one firing
    def fire(self, sg):
        q = self.bank.want([[n.B for n in sg.about]], [sg.sign])[0]
        C = self.cands(sg)
        a = self.choose(q, C)
        if a < 0:
            self.count("rest")
            self.rests.append(([n.B for n in sg.about], sg.sign))
            if self.show:                              # --inspect / --trace: why it rested (fm32 @50: 280 of 286
                g = self.bank.gate(q[None], [n for n, _, _ in C])[0]   # statements in a fresh world rested)
                top = sorted(range(len(g)), key=lambda i: -float(g[i]))[:4]
                print("  rest: about " + " + ".join(repr(self.say(n)[:50]) for n in sg.about) + " | gates " +
                      ", ".join(f"{self.name(C[i][0])[:30]} {float(g[i]):.2f}" for i in top), flush=True)
            return None
        actor = C[a][0]
        C = self.around(actor, C)
        ins, hops, sal, lg, pick = self.perceive(actor, q, sg, C, explore=self.explore)
        net = self.bank.interp
        o = self.bank.run(actor, ins, q, v=sg.sign, hops=hops, sal=sal)
        many, err, where, isfn = o["many"], o["err"], o["where"], o["isfn"]
        T = self.explore                               # a try: its whole action sampled from its own odds -- how many,
        nm = choice(many, T) + 1                       # words or a program, which pieces, where each goes (a right try
        fn_ = [bool(torch.bernoulli(torch.sigmoid(isfn[j] / T))) if T else float(isfn[j]) > 0 for j in range(nm)]   # is
        fslot = torch.tensor([False, False] + [n.kind != "work" for n in ins[:net.R]], device=q.device)[:where.shape[-1]]
        where = torch.stack([where[j] if fn_[j] else where[j].masked_fill(fslot, -1e9) for j in range(nm)])   # (words: a
        #                                                new node, the memory, or in place of a workspace node -- never a
        #                                                function, never a log: fm48 EVAL, 53 outputs said into a log)
        wsel = [choice(where[j], T) for j in range(nm)]
        asm = [None if fn_[j] else make_out(self, net, o, j, ins, wsel[j], T) for j in range(nm)]   # relived: what it
        #                                                sampled is what it learns to do). Each: (the node, its pieces,
        #                                                the part of the node it goes in place of, its tokens)
        made = [o["code"][j] if fn_[j] else asm[j][0] for j in range(nm)]
        #                                                words land in memory or the workspace, a program in a function
        #                                                (fm18c trace: 289 of 297 statements went into a function: memory 0)
        take = [i in pick for i in range(len(C))]
        rec = {"actor": actor, "about": [n.B for n in sg.about], "q": q, "C": [(n.B, n.L, k, a_, n.near) for n, k, a_ in C], "take": take,
               "ins": [still(n) for n in ins], "hops": hops, "sign": sg.sign,   # (its inputs as they were: fill changes the
               #                                                question node in place -- relived later, its spans pointed into
               #                                                the new value; fm42 'right' loss 7e8)
               "walkq": sg.about[0].L.detach().float().cpu().numpy() if sg.about[0].L is not None else None,
               "sal": [float(x) for x in sal], "outs": [], "did": 0,
               "aboutL": sg.about[0].L.detach() if sg.about[0].L is not None else None}
        self.count("acts: " + ("function" if actor.kind == "fn" else "read node"))
        calc = self.bank.calc_of(actor)
        if calc is not None and self.calc_on:          # a calculator: its own body states a fact from the phrase its
            pc = asm[0][1][0] if asm and asm[0] and asm[0][1] else None   # node copied (the first piece), into the memory
            r_ = stated(self, calc, toks_of(net, ins)[pc[1]:pc[2] + 1] if pc and pc[0] == "span" else None)
            if r_ is not None:                         # (a fact about numbers alone: never stale. The question that
                self.count("calculator")               # asked hears it, write())
                if self.show:
                    print(f"  fire {len(self.firings)}: {self.name(actor)} states {self.text_of(r_)!r}", flush=True)
                rec["did"] = int(self.write(r_.B, NEW_PLACE, ins, sg, False, rec, 0, toks=r_.toks))
                self.firings.append(rec)
                self.count("fires")
                return rec
            self.count("calculator read no numbers")
        st_ = self.bank.stats_of(actor)
        if self.show:                                  # --inspect: every firing, said
            for kind, found in self.walks:
                print(f"    walk by {'label' if kind == 1 else 'relation'}: " + "; ".join(
                    f"{p} d {d:.2f} {self.text_of(self.entry(p))[:35]!r}" for d, p in found), flush=True)
            rows = sorted(range(len(C)), key=lambda i: -float(lg[i]))[:10]
            print("    candidates (salience logit): " + " | ".join(f"{float(lg[i]):+.1f} {self.name(C[i][0])[:45]}" for i in rows), flush=True)
            g = self.bank.gate(q[None], [n for n, _, _ in C])[0]
            top = sorted(range(len(g)), key=lambda i: -float(g[i]))[:3]
            print(f"  fire {len(self.firings)}: signal {sg.sign:+.2f}, acts: {self.name(actor)} (gates " + ", ".join(
                  f"{self.name(C[i][0])[:30]} {float(g[i]):.2f}" for i in top) +
                  ") | about: " + " + ".join(repr(self.say(n)[:60]) for n in sg.about) +
                  f" | takes in {len(ins)}/{len(C)}: " + "; ".join(f"{self.name(n)[:45]} ({float(sv):.2f})" for n, sv in zip(ins, sal)) +
                  f" | many {nm}{' (sampled)' if T else ''}, where " + ", ".join(str(w_) for w_ in wsel) +
                  ", function? " + ", ".join(f"{float(torch.sigmoid(isfn[j])):.2f}" for j in range(nm)) +
                  f", own error {float(err):.2f} (usual {float(st_.mu):.2f}) | says: " +
                  "; ".join(repr(self.say(Node(z, None, None))[:60]) for z in made) +
                  " | pieces " + "; ".join(str(asm[j][1]) + (f" in place of part {asm[j][2]} ({'its own pointing' if asm[j][4] else 'where the log is named'})"
                                                              if asm[j][2] else "") if asm[j] else "program" for j in range(nm)), flush=True)
        self.firings.append(rec)
        self.count("fires")
        drift = float(err) > float(st_.mu + self.kappa * st_.sd)
        self.news = []
        for j, z in enumerate(made):
            wj, fj = wsel[j], fn_[j]
            pcs, pt, said_t, own = ([], None, None, False) if fj else asm[j][1:]   # (a copy's words are known: the
            #                                            tokens it took -- in40: a lookup copied one token and the decoder
            #                                            said '7 7 7')
            rec["outs"].append((pcs, wj, fj, pt if own else None))   # (what it made: its pieces, where it went, words or a function,
            x = ins[wj - 2] if 2 <= wj < 2 + len(ins) else None   # the part of the node it went in place of)
            if said_t is not None and x is not None and x.kind != "fn" and x.toks is not None:
                if said_t == list(x.toks[:net.L]):                          # it says exactly what it replaces: no change,
                    self.count("no change")                                 # no signal (tr24: the store wrote the statement
                    if x.kind == "entry":                                   # onto itself, the re-read moved its label, and
                        rec["same"] = True                                  # it fired 6-9 times; 1315 firings for 297)
                        rec.setdefault("at", []).append(x.place)
                    continue
            if drift and not fj and said_t is None:    # its own guess: unusually wrong -> said and read again (words
            #                                            only: a program read back as text gave make() 46 rows for 16,
            #                                            fm35 living crash)
                said_ = self.say(Node(z, None, None))
                if said_.strip():                      # (said as nothing, it stays as it was: read again, nothing is
                    z = self.nodes([said_])[0].B       # a node of no rows, its mean not a number -- fm21's gates)
                    self.count("roundtrip")
            rec["did"] += int(self.write(z, wj, ins, sg, fj, rec, j, toks=said_t))
        if len(self.news) == 1:                        # a new node is heard by itself; several made at once are heard
            self.signals.append(Signal(self.news, 1., self.t))   # together, with what they came from
        elif self.news:
            self.signals.append(Signal([sg.about[0]] + self.news, 1., self.t))
        return rec

    def write(self, z, w, ins, sg, isfn, rec, j, toks=None):
        """where an output lands, and what that change signals; -> whether anything changed"""
        self.t += 1
        if w >= 2 and w - 2 < len(ins) and ins[w - 2].kind != "entry":   # (a log is not rewritten: what is said of a
            x = ins[w - 2]                             # thing goes to the memory and finds its place by its label)
            if x.kind == "fn" and isfn:                # an edited function: its code is now this
                with torch.no_grad():
                    f_ = self.bank.fns[x.fn]                # (as it is now = z: the pending part stays)
                    f_.code.copy_(z - f_.pend["code"].detach() if f_.pend is not None and "code" in f_.pend else z)
                self.count("edits")
            elif x.kind == "fn":                       # words aimed at a function: a program is not words (fm18c: 270 of
                self.count("words aimed at a function")   # 346 statements overwrote a function) -> a new node
                n = self.node(z, toks=toks, parent=sg.about[0])
                self.news.append(n)
            else:                                      # a workspace node's value changed: the node that asked for it hears
                n = self.node(z, toks=toks)
                if self.text_of(n) == self.text_of(x):   # no change, no signal: it says what it said (11-11: by what it
                    self.count("no change")            # says, not by its label -- a fact's label is value-free now,
                    return False                       # so a filled-in value barely moves it)
                x.B, x.L, x.R, x.toks, x.text = n.B, n.L, n.R, n.toks, None
                self.signals.append(Signal([x.parent, x] if x.parent is not None else [x], 1., self.t))
            return True
        if isfn and not self.frozen:                   # a new node that acts (a function), made like an input function
            like = next((n for n in ins if n.kind == "fn"), None)
            self.bank.make(self.bank.want([[z]])[0].detach(), like, code=z.detach())
            self.made = True
            self.count("new functions")
            return True
        n = self.node(z, toks=toks, parent=sg.about[0])
        if w == NEW_PLACE:                             # into the memory: the place that is the thing it names (its label
            rows, key = self.label_rows(z)             # rows and that place's found in each other), else a new place
            p = self.g.same(rows, key)
            if p < 0:
                p = self.g.add_place(n.L.detach().float().cpu().numpy(), key)
                self.count("new place")
            else:
                self.count("found its place")
            self.memorize(p, z, rec, toks)
            a = sg.about[0]                            # a fact stated for a node that names that thing without being it
            if a.kind == "work" and self.g.same(*self.label_rows(a.B)) != p:   # (a question; a statement being stored
                self.signals.append(Signal([a, self.log(p)], 1., self.t))      # IS its place): the node hears the fact
        else:
            self.news.append(n)
        return True

    def memorize(self, p, z, rec, toks=None):
        """a place's log takes what was said of it, as it was said (v3: nothing is worked out when it is stored and no
        other place is rewritten -- what a thing is worth now, and what follows it, is read from the logs)"""
        self.g.chain[p].append((z.detach(), self.t, toks))
        self.g.R[p] = self.heads([z])[1][0].detach().float().cpu().numpy()
        rec.setdefault("at", []).append(p)

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
        self.asked = q                                 # (the node itself: evaluate also reads what it holds as copied)
        return self.say(q), n                          # the answer is said by the LM, from the node (user 11-12: all that
        #                                                comes out comes out of the LM; 11-12 to fm50 a copy's tokens were
        #                                                given back as they were, past the LM)


# ---------------------------------------------------------------------- the teacher: demos only (it never runs a step)
class Teacher:
    """which place is which thing, and what each statement did to the thing's value (its own log): used to write the
    demos, and to measure. It never runs a step"""

    def __init__(self, world):
        self.w, self.place, self.texts, self.did = world, {}, {}, {}

    def heard(self, p, text, meta, side):
        """a statement went into place p: it set the value, added an amount to it (the key's rule), or made the
        thing a copy of another fact"""
        self.texts[p].append(text)
        if meta[0] != "fact":
            return
        nm, k, v = meta[1:]
        src = self.w.deps.get((nm, k)) if "(same as" in text else None
        d = next((x for x in side.values() if x[0] == "delta" and x[1] == nm and x[2] == k), None)
        self.did.setdefault(p, []).append(("same", src) if src else ("add", d[3]) if d else ("set", v))

    def fold(self, p):
        """what a place's log says the thing is worth now, as an expression (the read of v3: nothing was worked out
        when it was stored), and how many entries back it reads: the newest value; under an adding rule the sum of
        what was there and the amounts said since; a copy: the fact it is the same as"""
        E, n = None, 0
        for kind, x in self.did.get(p, []):
            if kind == "add" and E is not None:
                E, n = ("op", "add", [E, ("val", x)]), n + 1
            else:
                E, n = ("leaf",) + x if kind == "same" else ("val", x), 1
        return E, n


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
        return unresolved(E[1]) or E
    if E[0] == "pair":
        return None
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
    if E[0] == "pair":
        return f"both values of {E[1]} and {E[2]}"
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
    """the demo steps for a question: (purpose, about, inputs, outputs). Two kinds of step, the same at every depth
    (11-12, user: self node creation; in53: asked to rewrite the question itself, the lookup node replaced all of it
    by one value):
    - read: a phrase of the question that names a known place becomes what that place's log says (lookup / before /
      both: the question + the log + the rule's log; where the phrase is, the label match that found the place says);
    - state: an operation on two numbers is not done in the question -- the calculator SAYS a fact, '<the operation
      as it stands> is <result>.', which goes to the memory like any statement; the question hears it and reads it.
    Last, the question that is only a value is answered. About / inputs name workspace nodes ("q") or a place's
    log ("e", place or the key of a fact stated on the way); outputs = (text, where, node or the stated fact's key)"""
    steps, w, was = [], T.w, E[0] == "before"
    tell = lambda E2: f"What was {show(E2)} before?" if was else ask_text(E2)
    while True:
        x = unresolved(E)
        if x is None:
            return steps + [("answer", ["q"], ["q"], [(VAL(E[1]), 2, "q")])], E
        if x[0] == "of" and x[2][0] == "val":          # its inner part is answered: now a plain fact
            E = swap(E, x, ("leaf", x[2][1], x[1]))
            continue
        if x[0] in ("op", "sumboth"):                  # stated, then read: the same read as any fact's
            a, b = (x[1][1], x[1][2]) if x[0] == "sumboth" else (x[2][0][1], x[2][1][1])
            y = ("val", next(f for f in TOOLS if f[0] == ("add" if x[0] == "sumboth" else x[1]))[1](a, b))
            said_ = show(x) if x is E else f"({show(x)})"   # (as it stands in the question)
            key = ("said", said_)
            steps.append(("add" if x[0] == "sumboth" else x[1], ["q"], ["q"], [(f"{said_} is {y[1]}.", NEW_PLACE, key)]))
            E = y if x is E else swap(E, x, y)
            steps.append(("lookup", ["q", ("e", key)], ["q", ("e", key)], [(tell(E), 2, "q")]))
            continue
        p = T.place.get(("fact", x[1], x[2]))
        if p is None:
            return None, None
        texts = T.texts[p][-4:]                        # (what a log shows at once)
        if x[0] == "leaf":                             # the log read: its value, the sum it comes to, or the fact it
            y, n_in = T.fold(p)                        # is the same as (the name is then that: read again)
            if y is None or n_in > 4 or (y[0] == "op" and named(y)) or truth_of(y, w) != w.ents[x[1]][x[2]]:
                return None, None                      # (more than the log shows at once; an amount added to a copy)
        elif x[0] == "before":
            h = w.hist[(x[1], x[2])]
            v = next((u for u in reversed(h) if u != h[-1]), None)
            if v is None or not any(value_in(t) == v for t in texts):
                return None, None
            y = ("val", v)
        else:
            vs = w.multi.get((x[1], x[2]), [])
            if len(vs) != 2 or {value_in(t) for t in texts} != set(vs):
                return None, None
            y = ("pair", vs[0], vs[1])
        r = T.place.get(("rule", x[2]))                # the place's log, and the log of the rule over its key
        E = y if x is E else swap(E, x, y)
        steps.append(({"leaf": "lookup"}.get(x[0], x[0]), ["q"], ["q", ("e", p)] + ([("e", r)] if r is not None else []),
                      [(tell(E), 2, "q")]))


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


def named(E):
    """the facts an expression names"""
    if E[0] in ("leaf", "before", "both"):
        return {("fact", E[1], E[2])}
    if E[0] == "val":
        return set()
    return named(E[2] if E[0] == "of" else E[1]) if E[0] in ("of", "sumboth") else set().union(*[named(c) for c in E[2]])


def truth_of(E, w):
    if E[0] == "val":
        return E[1]
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


def stated(fm, op, toks):
    """a calculator's own body (11-12): the phrase its node copied names an operation on two numbers -> the fact it
    states, '<the phrase as it stands> is <result>.', as a node read from those words. It says a fact; it does not
    touch what it read. None when the phrase does not hold exactly two numbers (an operation with a part still to
    read is not one it can do)"""
    if not toks or None in toks:
        return None
    text = fm.tr.tok.decode(list(toks))
    nums = re.findall(r"-?\d(?: \d)*", text)
    if len(nums) != 2:
        return None
    try:
        got = CALC[op](*[int(x.replace(" ", "")) for x in nums])
    except Exception:
        return None
    return fm.nodes([f"{text} is {got}."])[0]


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
    fm.routes.append(([n.B.detach() for n in about], v, purpose))   # (replayed: route_replay)
    del fm.routes[:-2000]
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
    if fm.stage == "skills" and not fm.frozen:         # stage 1: the skill's own node does it (a calculator its own
        t = fm.assign.get(purpose)                     # operation; else one made for the purpose, the first time)
        if t is None:
            t = next((i for i, f in enumerate(bank.fns) if f.calc == purpose), None)
            if t is None:
                t = bank.make(q.detach(), scope=A.scope)
                fm.made = True
                print(f"step {st['step']}: the {purpose} skill gets its node -> function {t}", flush=True)
            fm.assign[purpose] = t
        actor = next((n for n, _, _ in C if n.kind == "fn" and n.fn == t), None)   # the candidate itself (the
        if actor is None:                                                           # label losses find it by identity)
            actor = fm.fnode(t)
            C.append((actor, 2, 0))
        how = "assigned"
    free = [i for i, (n, _, _) in enumerate(C) if n.kind == "fn" and not specialized(fm, n, purpose)] if how == "nearest" else []
    if free:                                           # none meets the want: the nearest node not doing another purpose
        a = max(free, key=lambda i: float(s_all[i]))   # (fm17: making one whenever the nearest was busy made a node per
        actor = C[a][0]                                # want, 63 by step 150)
    elif how == "nearest" and not fm.frozen:           # every node near does another purpose: a new one, made like the nearest
        t = bank.make(q.detach(), like=actor, scope=A.scope)
        fm.made, how = True, "new"
        print(f"step {st['step']}: no label meets a {purpose} want; the nearest ({fm.name(actor)}) does another purpose "
              f"-> made function {t}", flush=True)
        actor = fm.fnode(t)
        C.append((actor, 2, 0))
    if purpose in fm.assign and fm.assign[purpose] < len(bank.fns):   # does routing reach the skill's node
        at = next((i for i, (n, _, _) in enumerate(C) if n.kind == "fn" and n.fn == fm.assign[purpose]), None)
        if at is not None:
            fm.count("routing reaches the skill's node", int(int(s_all[:len(C)].argmax()) == at if at < len(s_all) else 0))
            fm.count("routing checked")
            g_ = bank.gate(q[None], [n for n, _, _ in C])[0]   # light: the want meets that node's label (stage 1-2
            Ls.add("route hint", F.cross_entropy((g_ / .1)[None], torch.tensor([at], device=g_.device)) *   # .5; after
                   (.5 if fm.stage in ("skills", "route") else .1))    # .1: fm20, without it routing reached the skill's
            #                                                            node 68/2240 and read nodes did 558 of 618 firings
    rd = [i for i, (n, _, _) in enumerate(C) if n.kind != "fn"]
    if how not in ("new", "assigned") and actor.kind == "fn" and rd and not fm.frozen and random.random() < A.try_read:
        a = max(rd, key=lambda i: float(s_all[i]))     # now and then the read node nearest the want tries (a rule
        actor, how = C[a][0], "tried"                  # read as a program): its label learns from how it turns out
    fm.count(f"acts by {how}"), fm.count("acts: " + ("function" if actor.kind == "fn" else "read node"))
    C = fm.around(actor, C)
    keep = [i for i, (n, _, _) in enumerate(C) if key(n) in want_in]
    order, hops, sal, lg, pick = fm.perceive(actor, q, Signal(about, v), C, keep)
    if purpose == "lookup":                            # chance: a number taken from any input that has one
        vs_ = [value_in(fm.text_of(n)) for n in order if n.kind == "entry"]   # (the teacher's reading: "is 2 9." -> 29)
        fm.count("lookup chance", 1 / max(1, sum(1 for x in vs_ if x is not None and num(str(x)) is not None)))
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
    shown = []                                         # this step as a run to relive later (skill_replay)
    for j, (target, dest) in enumerate(outs):
        w = dest if isinstance(dest, int) else 2 + keys.index(key(dest))
        if isinstance(target, str):
            fm.said.append(target)                     # what functions are asked to say: the round trip covers it too
            got = say_loss(fm, A, net, o, j, target, order, prefer=[r for r, n in enumerate(order[:net.R]) if key(n) in want_in],
                           dest=w - 2 if w >= 2 else None)
            if got is None:
                res.append(None)
                exact, shown = False, None
                continue
            Lj, ok, z, plan, tid, spelled, (mine, same_, says, part, ed, inner) = got
            L, exact, dl = L + Lj, exact and ok, float(Lj)
            if shown is not None:
                shown.append(([(pc[0], pc[1], pc[2]) if pc[0] == "span" else ("made", pc[2] - pc[1]) for pc in plan], w, False, ed))
            fm.count(f"{purpose} exact", int(ok)), fm.count(f"{purpose} said", 1)
            if calc is not None and fm.calc_on:        # the calculator's own body: what goes on is the fact it states
                flat_ = toks_of(net, order)            # from the phrase its node should copy; its mimic learns from it
                cz = stated(fm, calc, flat_[plan[0][1]:plan[0][2] + 1] if plan and plan[0][0] == "span" else None)
                right = cz is not None and cz.toks is not None and fm.tr.tok.decode(cz.toks).strip() == fm.tr.tok.decode(tid).strip()
                fm.count("calculator right", int(right)), fm.count("calculator", 1)
                if right:
                    z, tid, ok = cz.B, cz.toks, True
            with torch.no_grad():
                lab, rel = fm.tr.encode([target])[0], fm.tr.relation([target])[0]
            L_, R_ = fm.heads([z])
            Ls.add("label of what it made", (1 - (L_[0] * lab).sum()) * .5 + F.mse_loss(R_[0], rel) * 5)
            if fm.capture is not None or fm.trace:
                with torch.no_grad():
                    pcs = mine[1]
                    told_ = (fm.say(Node(z, None, None)), fm.say(Node(mine[0], None, None)))
                if fm.capture is not None:             # --codec_check: what it says, with the target's pieces and its own
                    fm.capture.append((purpose, D(target), told_[0], told_[1], dl, len(pcs), len(plan), fm.name(actor)[:30]))
                if fm.trace:
                    info.append((j, target, w, dl, told_, plan, pcs, inner))
            res.append((z, tid, ok, spelled))
        else:                                          # a function given back: judged by what it then does
            shown = None                               # (not relived: a program is not pieces)
            f, rins, rq, rtarget = target
            got = say_loss(fm, A, net, bank.run(fm.fnode(f), rins, rq, program=o["code"][j]), 0, rtarget, rins)
            if got is not None:
                L, exact = L + got[0], exact and got[1]
            res.append((o["code"][j], None, True, None))
        wl = o["where"][j]
        if isinstance(target, str):                    # words: never into a function's place (as when it lives)
            wl = wl.masked_fill(torch.tensor([False, False] + [n.kind != "work" for n in order[:net.R]], device=dev)[:len(wl)], -1e9)
        L = L + F.cross_entropy(wl[None], torch.tensor([w], device=dev))
        w_ok = int(wl.argmax()) == w                   # where it lands is part of exact (tr24: store said right 95%,
        exact = exact and w_ok                         # living it wrote every statement onto itself)
        fm.count(f"{purpose} where", 1), fm.count(f"{purpose} where right", int(w_ok))
        L = L + F.binary_cross_entropy_with_logits(o["isfn"][j], torch.tensor(float(not isinstance(target, str)), device=dev))
    Ls.add("functions", L + F.mse_loss(o["err"], torch.log1p(L.detach())))   # its own guess of how wrong, on a log scale
    #                                                  (fm12: the raw loss squared swamped everything, 8233 of 8289)
    S, m = bank.gate(q[None], [actor])[0, 0], .1       # the label: pushed off a purpose it does not do exactly while it
    fk = (who(fm, actor), purpose)                     # does another one well; else pulled
    fm.fit[fk] = .9 * fm.fit.get(fk, 0.) + .1 * float(exact)
    push = not exact and how != "assigned" and (actor.kind != "fn" or how == "tried" or specialized(fm, actor, purpose))
    #        (a read node that acted and missed is pushed off this want, whatever the stage: fm20, read nodes out-bid
    #        the skills' nodes, being pulled when a statement stored itself and seldom pushed)
    Ls.add("labels", F.relu(S - (1 - m)) if push else F.relu(PULL - S))
    if exact and len(C) > 1:                           # it was right: among every node near the want, its label is the
        g_all = bank.gate(q[None], [n for n, _, _ in C])[0]    # one the want meets (all the others' move too)
        at = next(i for i, (n, _, _) in enumerate(C) if n is actor)
        Ls.add("labels", F.cross_entropy((g_all / .1)[None], torch.tensor([at], device=dev)))
    if fm.trace:
        trace_step(fm, purpose, v, about, actor, how, C, s_all, lg, y, order, o, info, outs, net, float(S), push, exact, float(L), rl)
    if shown and actor.kind == "fn" and calc is None:  # the demonstration kept as a run (its inputs as they were)
        fm.shown.setdefault(purpose, []).append({"actor": actor.fn, "ins": [still(n) for n in order], "q": q.detach(), "sign": v,
                                                 "hops": list(hops), "sal": [float(x) for x in sal], "outs": shown})
        del fm.shown[purpose][:-200]
    if actor.kind == "fn" and order:                   # what the update writer reads with this function's gradient
        bank.fns[actor.fn].last_ctx = (q.detach(), torch.stack([n.B.mean(0).detach() for n in order[:8]]),
                                       float(torch.log1p(L.detach())), float(exact))
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
            f"place {p} d {d:.2f} {fm.text_of(fm.entry(p))[:40]!r}" for d, p in found), flush=True)
    name = {0: "signal's", 1: "label walk", 2: "function", 3: "relation walk", 4: "two hops"}
    rows = sorted(range(len(C)), key=lambda i: -float(lg[i]))
    print(f"   candidates {len(C)} (filter logit, wanted): " + " | ".join(
        f"{float(lg[i]):+.1f}{'*' if y[i] > 0 else ''} {name[C[i][1]]} " +
        (f"fn {C[i][0].fn}" if C[i][0].kind == "fn" else f"{told(fm, C[i][0], 45)!r}" + (f" @{C[i][0].place}.{C[i][0].idx}" if C[i][0].kind == "entry" else ""))
        for i in rows[:14]), flush=True)
    print("   inputs in order: " + " | ".join(f"{r}: {told(fm, n, 50)!r}" for r, n in enumerate(order)), flush=True)
    print(f"   many {int(o['many'].argmax()) + 1} (target {len(outs)}), own error {float(o['err']):.2f} (usual {float(fm.bank.stats_of(actor).mu):.2f})", flush=True)
    for j, target, w, dl, told_, plan, pcs, tid in info:
        wp = o["where"][j]
        if isinstance(target, str):                    # (as it is used: words never into a function's place)
            wp = wp.masked_fill(torch.tensor([False, False] + [n.kind != "work" for n in order[:net.R]], device=wp.device)[:len(wp)], -1e9)
        wp = wp.softmax(-1)
        print(f"   out {j}: target {target!r}\n      target's pieces: {pieces_said(fm, order, net, o, j, plan, tid)}\n"
              f"      its own pieces:  {pieces_said(fm, order, net, o, j, pcs)}\n"
              f"      said with the target's pieces {told_[0]!r} | with its own {told_[1]!r}\n"
              f"      where {int(wp.argmax())} ({float(wp.max()):.2f}) target {w} | function? {float(torch.sigmoid(o['isfn'][j])):.2f} | "
              f"loss of this output {dl:.2f}", flush=True)
    print(f"   label gate {S:.2f}, step loss {L:.2f}, exact {exact} -> {'push' if push else 'pull'}" +
          (f" | relation loss {float(rl):.2f}" if rl is not None else ""), flush=True)


def demo_question(fm, T, Ls, A, st, E):
    """a question done step by step by the model (each step reads the previous steps' outputs as they are). A fact
    stated on the way goes to the memory as write() files it, for this question only (as when it lives)"""
    steps, last = plan(T, E)
    if not steps:
        return False
    ws, keep, stated_ = {"q": fm.nodes([ask_text(E)])[0]}, fm.g, []
    fm.g = clone_graph(fm.g)
    try:
        for purpose, about, ins, outs in steps:
            pick = lambda r: fm.log(r[1] if isinstance(r[1], int) else T.place[r[1]]) if isinstance(r, tuple) else ws[r]
            res, _ = run_step(fm, Ls, A, st, purpose, [pick(r) for r in about], [pick(r) for r in ins],
                              [(text, w if w in (NEW_WORK, NEW_PLACE) else ws[name]) for text, w, name in outs])
            if res is None or any(x is None for x in res):
                return False
            for (text, w, name), (z, tid, ok, spelled) in zip(outs, res):
                n = fm.node(z, tid) if ok else fm.nodes([spelled])[0]   # exact: its own node goes on, as it is; not
                #                                        yet: what it should have said (its tokens are then true)
                if w == NEW_PLACE:
                    with torch.no_grad():
                        p = fm.g.add_place(n.L.detach().float().cpu().numpy(), fm.label_rows(n.B)[1])
                        fm.g.R[p] = n.R.detach().float().cpu().numpy()
                    fm.t += 1
                    fm.g.chain[p].append((n.B.detach(), fm.t, n.toks))
                    T.place[name], T.texts[p] = p, [text]
                    stated_.append((name, p))
                else:
                    n.parent = ws["q"] if w == NEW_WORK else ws[name].parent
                    ws[name] = n
        Ls.none.append([ws["q"].B])                   # answered: its value asks for nothing more
        return True
    finally:
        fm.g = keep
        for name, p in stated_:
            T.place.pop(name, None), T.texts.pop(p, None)


def same_place(fm, s, p):
    """the label rows' lesson on a statement: the place that is its thing (p, when there is one) and the statement are
    found in each other; the other places near it are not that thing"""
    if not fm.g.n:
        return 0
    r, g, m = fm.heads.rows([s.B])
    ps = [t[1] for t in fm.g.near(r[0].detach().float().cpu().numpy(), 8) if t[1] != p] + ([p] if p is not None else [])
    P = torch.from_numpy(fm.g.M[ps]).to(r.device)
    pm = (torch.arange(RM, device=r.device)[None] < torch.from_numpy(fm.g.Mn[ps]).to(r.device)[:, None]).float()
    both = torch.minimum(align(P, pm, r, m.float())[:, 0], align(r, g, P, pm)[0])
    is_p = torch.tensor([q == p for q in ps], device=r.device)
    fm.count("same thing: right", int(p is not None and float(both[-1]) > JOIN and int(both.argmax()) == len(ps) - 1 or
                                      p is None and float(both.max()) <= JOIN)), fm.count("same thing: asked")
    return apart_at(both, is_p)


def demo_statement(fm, T, Ls, A, st, text, meta, side):
    """a statement: the model keeps it as it was said (v3, 11-12: nothing is worked out when it is stored, no other
    place is rewritten); the teacher writes the memory -- the thing's own place, its log one entry longer -- and the
    label rows learn which place is the same thing"""
    s = fm.nodes([text])[0]
    k = ("rule", meta[1]) if meta[0] == "rule" else ("fact", meta[1], meta[2])
    p = T.place.get(k)
    src = T.w.deps.get((meta[1], meta[2])) if meta[0] == "fact" and "(same as" in text else None
    also = [T.place[("fact",) + src]] if src and ("fact",) + src in T.place else []
    res, _ = run_step(fm, Ls, A, st, "store", [s], [s], [(text, NEW_PLACE)], also)
    Ls.add("same thing", same_place(fm, s, p))
    own, own_t = (res[0][0].detach(), res[0][1]) if res is not None and res[0] is not None and res[0][2] and st["step"] > A.own \
        else (s.B.detach(), s.toks)                    # the teacher's memory: as the store itself writes it, once it can
    if p is None:
        with torch.no_grad():
            p = fm.g.add_place(fm.heads([own])[0][0].float().cpu().numpy(), fm.label_rows(own)[1])
        T.place[k], T.texts[p] = p, []
    fm.t += 1
    fm.g.chain[p].append((own, fm.t, own_t))
    with torch.no_grad():
        fm.g.R[p] = fm.heads([own])[1][0].float().cpu().numpy()
    T.heard(p, text, meta, side)


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
    if fm.assign.get("inc") is not None and fm.assign["inc"] < len(fm.bank.fns):
        return fm.assign["inc"]                        # the add-one skill's node
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


def still(n):
    """a node as it was (its rows cut from any graph: a run kept to relive later)"""
    m = Node(n.B.detach(), n.L.detach() if n.L is not None else None, n.R.detach() if n.R is not None else None,
             n.kind, n.place, n.idx, fn=n.fn)
    m.toks, m.text = n.toks, n.text
    return m


def skill_replay(fm, rng, n=8):
    """the demos' steps relived while it lives, as many of each purpose (11-11, fm38: living relived only statement
    runs -- questions never came out right -- and the interpreter, shared by every node, learned to file everything:
    'found its place' 401 for 290 statements by step 100, every question answered in 1 firing, lookup 7/24 -> 0/24).
    Nothing new is shown: the same 50 demo steps, again"""
    ps = [k for k, v in fm.shown.items() if v and fm.assign.get(k, -1) < len(fm.bank.fns)]
    if not ps:
        return 0
    recs = []
    for _ in range(n):
        r = rng.choice(fm.shown[rng.choice(ps)])
        if r["actor"] >= len(fm.bank.fns) or any(x.kind == "fn" and x.fn >= len(fm.bank.fns) for x in r["ins"]):
            continue
        recs.append(dict(r, actor=fm.fnode(r["actor"]), ins=[fm.fnode(x.fn) if x.kind == "fn" else x for x in r["ins"]]))
    return relive(fm, recs)


def route_replay(fm, rng, n=64):
    """the demos' routing examples again (no interpreter, no new demo): each signal's want apart from other purposes'
    and toward the node that learned its purpose (fm32 @50: a statement's want and its subject's .91 cos -- the store's
    and the lookup's labels met just under 1 and both rested; ~250 updates in 50 demo steps were not enough to part them)"""
    xs = [x for x in fm.routes if x[2] in fm.assign and fm.assign[x[2]] < len(fm.bank.fns)]
    if len(xs) < 8:
        return 0
    xs = rng.sample(xs, min(n, len(xs)))
    q = fm.bank.want([r for r, _, _ in xs], [v for _, v, _ in xs])
    fns = [fm.fnode(t) for t in range(len(fm.bank.fns))]
    g = fm.bank.gate(q, fns)
    tgt = torch.tensor([fm.assign[k] for _, _, k in xs], device=q.device)
    own = g.gather(1, tgt[:, None]).squeeze(1)
    kinds, none = [k for _, _, k in xs], 0
    if fm.nones:                                       # ... and what asked nothing (answered values, talk) stays under
        qn = fm.bank.want(rng.sample(fm.nones, min(n // 4, len(fm.nones))))   # every label (in48 / fm48: the pulls were
        none = F.relu(fm.bank.gate(qn, fns) - .9).sum() / len(qn)   # rehearsed 20x a step, this once: 144 eval questions
        q, kinds = torch.cat([q, qn]), kinds + ["none"] * len(qn)   # rested 20 times, an answer was fired on again)
    return apart(q, kinds) + F.cross_entropy(g / .1, tgt) + F.relu(PULL - own).mean() + none


def finish(fm, Ls, rng, recent=None):
    """the routing losses of a batch: wants of one purpose together; what asks nothing (talk, answered values) below
    every label. recent: [(want detached, purpose)] of the last updates, compared against (and extended)"""
    talk = fm.nodes(rng.sample(CHAT, 3) + [VAL(amount(rng)), VAL(_name(rng, 2))])
    rows = Ls.none + [[n.B] for n in talk]
    fm.nones += [[b.detach() for b in r] for r in rows]   # (rehearsed: route_replay)
    del fm.nones[:-500]
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


def same_batch(w, rng, n=10):
    """texts for the label rows' lesson, from a world: things said two ways (another value, a correction, a copy, the
    thing's name alone; a rule in other words; a fact a calculator states) with what each is about and the words that
    name it, and questions with the things they name. -> statements (text, thing, name), questions (text, things)"""
    facts = [(nm, k, v) for nm, e in sorted(w.ents.items()) for k, v in sorted(e.items())]
    facts = rng.sample(facts, min(n, len(facts)))
    names, st = sorted(w.ents), []
    for nm, k, v in facts:
        other = lambda: rng.randint(1, 60) if isinstance(v, int) else rng.choice(names)
        f2, nme = rng.choice(facts), leaf(nm, k)
        st.append((f"{nme} is {v}.", ("fact", nm, k), nme))
        st.append((rng.choice([f"{nme} is {other()}.", f"Correction: {nme} is now {other()}.",
                               f"{nme} is {other()} (same as {leaf(f2[0], f2[1])}).", nme]), ("fact", nm, k), nme))
    for k in rng.sample(w.num_keys, 3):
        st += [(rule_text(rng, kind, k), ("rule", k), None) for kind in rng.sample(["add", "set", "keep"], 2)]
    st += [(t, None, None) for t in rng.sample(CHAT, 2)]
    qs = [(ask_text(E), named(E)) for E in (q_expr(kd, w, rng) for kd in ("lookup", "before", "chain", "ops", "ops")) if E]
    for _ in range(3):                                 # a fact stated on the way, and the question it was stated for
        op, fn_, tpl = rng.choice(TOOLS)
        a, b, c = amount(rng), amount(rng), amount(rng)
        if rng.random() < .5:                          # (inside another operation: in brackets, as it stands there)
            nme = f"({tpl.format(a, b)})"
            qs.append((f"What is {rng.choice(TOOLS)[2].format(*rng.sample([nme, str(c)], 2))}?", {("said", nme)}))
        else:
            nme = tpl.format(a, b)
            qs.append((f"What is {nme}?", {("said", nme)}))
        st.append((f"{nme} is {fn_(a, b)}.", ("said", nme), nme))
    return st, qs


def same_loss(fm, st, qs):
    """the label rows' lesson: two statements about one thing are the same place (the one's naming rows found in the
    other's words, and the other's naming rows among them: both ways), statements about two things are not; a thing a
    question names is found in it, another is not; and the rows that name a thing are the words of its name (a name is
    found where it stands). Returns the loss and how often each comes out right"""
    texts = [D(t) for t, _, _ in st] + [D(t) for t, _ in qs]
    z, m = fm.tr.content(texts)
    R, G, M = fm.heads.rows([z[i][m[i]] for i in range(len(z))])
    n = len(st)
    into = align(R[:n], G[:n], R, M.float())           # [i, j]: thing i's naming rows found in what j says
    both = torch.minimum(into[:, :n], align(R[:n], G[:n], R[:n], G[:n]).T)   # ... and j's among i's: the same thing
    dev = R.device
    one = torch.tensor([[a is not None and a == b for _, b, _ in st] for _, a, _ in st], device=dev)
    off = ~torch.eye(n, dtype=torch.bool, device=dev)
    loss = apart_at(both[off], one[off])
    rate = lambda x: float((x > JOIN).float().mean()) if x.numel() else 0.
    said = {"same": rate(both[off & one]), "other": rate(both[off & ~one]),
            "same low": float(both[off & one].min()) if bool((off & one).any()) else 1., "other high": float(both[off & ~one].max())}
    thing = torch.tensor([a is not None and a[0] != "rule" for _, a, _ in st], device=dev)
    if qs and bool(thing.any()):
        has = torch.tensor([[a in inq for _, inq in qs] for _, a, _ in st], device=dev)[thing]
        loss = loss + apart_at(into[:, n:][thing], has)
        said["named"], said["not named"] = rate(into[:, n:][thing][has]), rate(into[:, n:][thing][~has])
    nl, hit, k = 0, 0, 0
    for i, (_, _, nme) in enumerate(st):               # the rows that name it: the words of its name
        if nme is None:
            continue
        at = texts[i].find(D(nme))
        offs_ = fm.tr.tok(texts[i], return_offsets_mapping=True)["offset_mapping"]
        if at < 0 or len(offs_) != int(M[i].sum()):
            continue
        y = torch.tensor([float(a_ < at + len(D(nme)) and b_ > at) for a_, b_ in offs_], device=dev)
        g = G[i, :len(offs_)]
        nl, k = nl + F.binary_cross_entropy(g.clamp(1e-4, 1 - 1e-4), y), k + 1
        hit += int(bool(((g > .5).float() == y).all()))
    if k:
        loss, said["name rows"] = loss + nl / k, hit / k
    return loss, said


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
        loss = (1 - (L * lab).sum(-1)).mean() + F.mse_loss(R, rel) * 10 + reread_loss(fm, z, m, texts, rng)
        same, _ = same_loss(fm, *same_batch(w, rng))
        loss = loss + same
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 100 == 0:
            print(f"codec {step}: label cos {float((L * lab).sum(-1).mean()):.3f} relation err {float(F.mse_loss(R, rel)):.4f} | "
                  f"{label_probe(fm, random.Random(7))} | {same_probe(fm, random.Random(7))}", flush=True)


@torch.no_grad()
def same_probe(fm, rng, n=8):
    """the label rows in fresh worlds: how often two statements of one thing come out the same place, of two things
    too; how often a fact a question names is found in it, and one it does not name"""
    tot = {}
    for _ in range(n):
        w = new_world(rng)
        for _ in range(6):
            for d in world_step(w, rng):
                d()
        for k, v in same_loss(fm, *same_batch(w, rng))[1].items():
            tot[k] = tot.get(k, 0) + v / n
    return "label rows: " + ", ".join(f"{k} {v:.2f}" for k, v in tot.items())


@torch.no_grad()
def label_probe(fm, rng, n=12):
    """a place is what a fact is about: the same fact with another value should get the same label, another fact a far
    one. Distances by the label head (and by the encoder, its target)"""
    w = new_world(rng)
    for _ in range(6):
        for d in world_step(w, rng):
            d()
    facts = [(nm, k, v) for nm, e in sorted(w.ents.items()) for k, v in sorted(e.items())][:n]
    names = sorted(w.ents)
    a = [D(f"{leaf(nm, k)} is {v}.") for nm, k, v in facts]
    b = [D(f"{leaf(nm, k)} is {v + 7 if isinstance(v, int) else next(x for x in names if x != v)}.") for nm, k, v in facts]
    z, m = fm.tr.content(a + b)
    L, _ = fm.heads([z[i][m[i]] for i in range(len(a + b))])
    E = fm.tr.encode(a + b)
    out = []
    for name, X in (("heads", L), ("encoder", E)):
        same = (X[:len(a)] - X[len(a):]).norm(dim=-1)
        dd = torch.cdist(X[:len(a)], X[:len(a)])
        other = dd.sum() / (len(a) * (len(a) - 1))
        near = (dd + torch.eye(len(a), device=dd.device) * 9).min()
        out.append(f"{name}: same fact {float(same.mean()):.2f} (max {float(same.max()):.2f}), other fact {float(other):.2f} "
                   f"(nearest {float(near):.2f})")
    return "; ".join(out)


def reread_loss(fm, z, m, texts, rng, n=12):
    """pieces of texts' rows joined (a span or two, as a function's output is), read again: against the encoder's rows
    for the joined words, and said back"""
    tr = fm.tr
    rows = [(z[i][m[i]].detach(), tr.tok(t)["input_ids"]) for i, t in enumerate(texts)]
    rows = [(r, ids) for r, ids in rows if len(ids) == len(r) and len(ids) > 1]
    if not rows:
        return 0

    def cut(r, ids):
        a = rng.randrange(len(ids))
        b = min(len(ids) - 1, a + rng.randrange(8))
        return r[a:b + 1], ids[a:b + 1]
    loss, k = 0, 0
    for _ in range(n):
        parts = [cut(*rng.choice(rows)) for _ in range(rng.choice([1, 2, 2, 3]))]
        ids = [i for _, x in parts for i in x]
        text = tr.tok.decode(ids)
        if tr.tok(text)["input_ids"] != ids:
            continue
        zr = fm.heads.reread(torch.cat([r for r, _ in parts]))
        with torch.no_grad():
            zt, mt = tr.content([text])
        zt = zt[0][mt[0]]
        if len(zt) == len(zr):
            loss, k = loss + F.mse_loss(zr, zt) + tr.decode_loss(zr[None], torch.ones(1, len(zr), dtype=torch.bool, device=zr.device), [text]), k + 1
    return loss / max(1, k)


@torch.no_grad()
def evaluate(fm, A, seed):
    """a fresh world: the module writes the memory itself from what arrives and answers; nothing is scripted. The
    teacher only measures"""
    rng = random.Random(seed)
    w = new_world(rng)
    fm.g, fm.stats, fm.firings, fm.rests = Graph(fm.bank.d), {}, [], []
    nstat, things, inside = 0, {}, {}
    for _ in range(A.ticks):
        for d in world_step(w, rng):
            o = d()
            for text, meta in said(o, w) if o else []:
                n0 = len(fm.firings)
                fm.arrive(text)
                nstat += 1
                at = {p for r in fm.firings[n0:] for p in r.get("at", [])}
                things.setdefault(meta[:2] if meta[0] == "rule" else meta[:3], []).append((at, D(text)))
                for p in at:
                    inside.setdefault(p, set()).add(meta[:2] if meta[0] == "rule" else meta[:3])
    mem = dict(fm.stats)
    whole = 0                                          # the memory it wrote itself, against what was said: a thing is
    for k, xs in things.items():                       # whole when all that was said of it is one place's log, nothing
        ps = set().union(*[at for at, _ in xs])        # else is in that place, and its newest entry is what was said last
        whole += all(len(at) == 1 for at, _ in xs) and len(ps) == 1 and inside[min(ps)] == {k} and \
            D(fm.text_of(fm.entry(min(ps)))) == xs[-1][1]
    held = (f"memory: things kept whole {whole}/{len(things)}, places {fm.g.n}, entries per place "
            f"{sum(len(c) for c in fm.g.chain) / max(1, fm.g.n):.2f}")
    line, snap = [], snapshot(fm.bank)                 # (each question from the same functions: what one question's run
    #                                                    made or edited is not what the next is judged on -- fm49 @150: 30
    #                                                    new functions by the end of one eval)
    for kind, ops in (("lookup", 0), ("before", 0), ("coexist", 0), ("chain", 0), ("ops", (1, 3)), ("long", (4, 6))):
        hit = held_ = n = steps = 0
        for _ in range(A.q):
            E = q_expr("ops" if kind == "long" else kind, w, rng, ops or (1, 3))
            if E is None:
                continue
            n += 1
            keep_ = fm.g                               # (what a question's run states stays its own: no truth comes
            fm.g = clone_graph(keep_)                  # here to say it was right)
            got, k = fm.ask(ask_text(E))
            fm.g = keep_
            steps += k
            hit += got == D(VAL(truth_of(E, w)))
            held_ += fm.text_of(fm.asked) == D(VAL(truth_of(E, w)))
            restore(fm.bank, snap)
        line.append(f"{kind} {hit}/{n} (the node holds it {held_}/{n}; {steps / max(1, n):.1f} firings)")
    stats = ", ".join(f"{k} {v}" for k, v in sorted(mem.items()))
    return f"statements {nstat} [{stats}] | {held} | answers: " + " | ".join(line) + f" | made nodes that act {len(fm.bank.fns)}"


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
    h.L, h.R, h.M, h.Mn = g.L.copy(), g.R.copy(), g.M.copy(), g.Mn.copy()
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
        q = fm.bank.want([r["about"]], [r["sign"]])[0].detach()   # the label moves, the shared want reader does not
        loss = loss + F.relu(PULL - fm.bank.gate(q[None], [a])[0, 0])   # (tr30: imitating ~500 statement runs pulled every
        #                                                    want toward the store's label; questions then rested or were
        #                                                    stored as facts -- as signalled(), fm17c)
        if a.kind == "fn":                             # and among every node that acts, its label is the one this want
            g = fm.bank.gate(q[None], [fm.fnode(t) for t in range(len(fm.bank.fns))])[0]   # meets (the demos' rule, run_step;
            loss = loss + F.cross_entropy((g / .1)[None], torch.tensor([a.fn], device=g.device))   # fm39: pull alone left
            #                                                    function 6 at 1.51 for statements AND questions)
        lg = fm.filt(fm.bank.key(a).detach(), fm.bank.program(a).detach(), q.detach(), r["sign"], 0., r["C"],
                     about=r.get("aboutL"))   # (codes change in place as it lives)
        loss = loss + F.binary_cross_entropy_with_logits(lg, torch.tensor(r["take"], dtype=torch.float, device=lg.device))
    return loss / max(1, len(recs))


def unasked(fm, recs, rests):
    """what asked nothing in a run that turned out right: a firing that changed nothing (its label moves off that
    want), and what the run rested on (every made node's label stays under that want). Labels only: the shared want
    reader does not move (fm17c)"""
    loss = signalled(fm, [r for r in recs if not r.get("did", 1)], -1)
    if rests and len(fm.bank.fns):
        q = fm.bank.want([r for r, _ in rests], [v for _, v in rests]).detach()
        loss = loss + F.relu(fm.bank.gate(q, [fm.fnode(t) for t in range(len(fm.bank.fns))]) - .9).sum() / len(rests)
    return loss


def lesson_of(fm, recs, rests):
    """a right run as its own lesson: the firings that changed something imitated and relived, the rest unasked"""
    acted = [r for r in recs if r.get("did", 1)]
    return imitate(fm, acted) + relive(fm, acted) + unasked(fm, recs, rests)


def tries(fm, A, g0, snap, act, good):
    """sampled tries from the same start (memory g0, functions as in snap): the cheapest one that turns out right ->
    (its firings, its memory, what it rested on), else None. Nothing a try makes stays in the functions"""
    best, fm.explore = None, A.explore
    for _ in range(A.tries):
        fm.g, fm.firings, fm.rests = clone_graph(g0), [], []
        ok = good(act())
        if ok and (best is None or len(fm.firings) < len(best[0])):
            best = (list(fm.firings), fm.g, list(fm.rests))
        restore(fm.bank, snap)
    fm.explore = 0.
    return best


def held(fm, recs, want):
    """a statement was kept: a place this arrival wrote (or found saying it already) now says what the world says the
    memory should hold (the truth arriving, as a question's does). dl37: 'kept' by the check's score was hollow -- the
    calculator filed '2 7' for a '(same as ...)' statement, the check passed it .69, and that run was imitated"""
    at = {p for r in recs for p in r.get("at", []) if p < fm.g.n and fm.g.chain[p]}
    return any(D(fm.text_of(fm.entry(p))) == want for p in at)


def relive(fm, recs):
    """a run that turned out right is its own demonstration: each firing's outputs as it made them (its pieces, where
    each went, how many, words or a function) are what that node learns to make from those inputs (user: CoT is
    imitation; after the first demos, the only teacher is the outcome)"""
    net, loss, n = fm.bank.interp, 0, 0
    for r in recs:
        a, outs = r["actor"], r.get("outs")
        if not outs or gone(fm, r):                    # (a calculator's own body; a node gone -- fm36 crashed on an
            continue                                   # input a rolled-back try had made)
        o = fm.bank.run(a, r["ins"], r["q"], v=r["sign"], hops=r["hops"], sal=r["sal"])
        dev = o["many"].device
        L = F.cross_entropy(o["many"][None], torch.tensor([min(len(outs), net.M) - 1], device=dev))
        fslot = torch.tensor([False, False] + [x.kind != "work" for x in r["ins"][:net.R]], device=dev)
        for j, (pcs, w, isfn, pt) in enumerate(outs[:net.M]):
            if pt is not None and not isfn:            # which part of the node it changed
                L = L + F.cross_entropy(net.part_start(o, j, w - 2)[None], torch.tensor([pt[0]], device=dev)) + \
                    F.cross_entropy(net.part_end(o, j, pt[0])[None], torch.tensor([pt[1]], device=dev))
            for s_ in range(min(len(pcs) + 1, net.S) if not isfn else 0):
                k = 0 if s_ == len(pcs) else 1 if pcs[s_][0] == "span" else 2
                L = L + F.cross_entropy(o["kind"][j, s_][None], torch.tensor([k], device=dev))
                if k == 1:
                    a_, b_ = pcs[s_][1], pcs[s_][2]
                    L = L + F.cross_entropy(o["start"][j, s_][None], torch.tensor([a_], device=dev)) + \
                        F.cross_entropy(net.end_logits(o, j, s_, a_)[None], torch.tensor([b_], device=dev))
                elif k == 2:
                    L = L + F.binary_cross_entropy_with_logits(o["gkeep"][j, s_], (torch.arange(net.G, device=dev) < pcs[s_][1]).float())
            wl = o["where"][j] if isfn else o["where"][j].masked_fill(fslot[:len(o["where"][j])], -1e9)
            L = L + F.cross_entropy(wl[None], torch.tensor([w], device=dev))
            L = L + F.binary_cross_entropy_with_logits(o["isfn"][j], torch.tensor(float(isfn), device=dev))
        loss, n = loss + L, n + 1
    return loss / max(1, n)


def signalled(fm, recs, sign):
    """a + / - signal on firings: each actor's label moves toward / away from the want that fired it (the want reader,
    shared by everything, does not move here: fm17c, pushing through it silenced every node in 3 ticks)"""
    loss, n = 0, 0
    for r in recs:
        a = r["actor"]
        if a.kind == "fn" and a.fn >= len(fm.bank.fns):
            continue
        q = fm.bank.want([r["about"]], [r["sign"]])[0].detach()
        S = fm.bank.gate(q[None], [a])[0, 0]
        loss, n = loss + (F.relu(PULL - S) if sign > 0 else F.relu(S - .9)), n + 1
    return loss / max(1, n)


def gone(fm, r):
    """a record that names a node no longer there (a try made it, the try was rolled back)"""
    return any(x.kind == "fn" and x.fn >= len(fm.bank.fns) for x in [r["actor"]] + list(r["ins"]))


def surprise(fm, kind, ok):
    """how much a right run of this kind teaches: 1 - how often its own run is right (a baseline, as RL's advantage).
    11-12, fm41: living imitated every statement it already kept (462/481 in 50 steps) -- no news, and the shared
    interpreter drifted to filing (@50 lookup 5, before 4, ops 11 firings; @100 every question 0 in ~2 firings).
    Updated after it is read: the run's own outcome joins the rate"""
    r = fm.rates.get(kind, .5)
    fm.rates[kind] = .95 * r + .05 * float(ok)
    return 1. - r


def live_tick(fm, A, w, rng, upd, tally):
    """one tick of living: nothing scripted, no demo. What arrives is stored by the module itself; a statement that
    nothing kept (or whose kept value the check fails) is wrong: a few sampled tries from where it was, and the cheapest
    one that keeps it is imitated and its memory stays. Questions: the truth arrives after; right -> what its run made or
    edited on the way stays (accumulated) and the run is imitated; wrong -> rolled back, the cheapest right try imitated.
    No push on a wrong one's labels (fm17c: that silenced the module in 3 ticks). Returns the texts it read"""
    mods = (fm.bank, fm.heads, fm.filt, fm.critic)
    seen = []
    for d in world_step(w, rng):
        o = d()
        for text, meta in said(o, w) if o else []:
            seen.append(text)
            want = D(text)                             # what the memory should hold: what was said, as it was said
            for mod in mods:
                mod.eval()
            g0, snap = clone_graph(fm.g), snapshot(fm.bank)
            fm.firings, fm.rests = [], []
            if fm.show:
                print(f"\nLIVE statement {text!r}", flush=True)
            fm.arrive(text)
            recs, rests = list(fm.firings), list(fm.rests)
            tally["statements"] += 1
            ok_ = held(fm, recs, want)
            gain = surprise(fm, "statement", ok_)
            if ok_:
                tally["kept"] += 1
                best = recs
            else:
                g_own = fm.g
                b = tries(fm, A, g0, snap, lambda: fm.arrive(text), lambda _: held(fm, fm.firings, want))
                fm.g, best, rests = (b[1], b[0], b[2]) if b else (g_own, None, [])
                tally["kept by a try"] += int(b is not None)
            if fm.show:
                print(f"  -> held {ok_} (" + "; ".join(f"{fm.name(r['actor'])[:22]}: kept {[(j, round(ok, 2)) for j, ok in r.get('kept', [])]}"
                      f"{' same' if r.get('same') else ''}" for r in recs) + f") | by a try: {bool(b) if not ok_ else '-'}"
                      f" | memory {fm.g.n} places, functions {len(fm.bank.fns)}", flush=True)
            for mod in mods:
                mod.train()
            Ls = Lesson()                              # (no gradient through the frozen check while it lives: functions
            if best:                                   # learned to pass it -- dl37, fm36 'kept' 474/489, memory 0/151)
                Ls.add("kept", gain * lesson_of(fm, best, rests))
            upd(Ls)
    for _ in range(A.lq):
        kind = rng.choice(["lookup", "before", "coexist", "chain", "ops"])
        E = q_expr(kind, w, rng)
        if E is None or not fm.g.n:
            continue
        for mod in mods:
            mod.eval()
        truth, g0, snap = D(VAL(truth_of(E, w))), fm.g, snapshot(fm.bank)
        fm.g, fm.firings, fm.rests = clone_graph(g0), [], []
        if fm.show:
            print(f"\nLIVE question {ask_text(E)!r} truth {truth!r}", flush=True)
        got, _ = fm.ask(ask_text(E))
        recs, rests = list(fm.firings), list(fm.rests)
        fm.g = g0                                      # asking leaves the memory as it was (its access keys learn)
        tally["asked"] += 1
        gain = surprise(fm, kind, got == truth)
        if fm.show:
            print(f"  -> {got!r} ({'right' if got == truth else 'wrong'}, {len(recs)} firings)", flush=True)
        if got == truth:                               # what it made or edited on the way stays: accumulated
            tally["right"] += 1
            best = recs
            b = tries(fm, A, g0, snapshot(fm.bank), lambda: fm.ask(ask_text(E))[0], lambda got_: got_ == truth)
            fm.g = g0                                  # practice even when right (11-11, user: self-improving episodes
            if b and len(b[0]) < len(recs):            # always): a shorter right way is what it imitates
                best, rests = b[0], b[2]
                tally["shorter by a try"] += 1
        else:
            restore(fm.bank, snap)
            b = tries(fm, A, g0, snap, lambda: fm.ask(ask_text(E))[0], lambda got_: got_ == truth)
            fm.g, best, rests = g0, (b[0] if b else None), (b[2] if b else [])
            if best:
                tally["right by a try"] += 1
        for mod in mods:
            mod.train()
        if best:
            Ls = Lesson()
            Ls.add("right", gain * lesson_of(fm, best, rests))
            upd(Ls)
    return seen


def living_line(tally):
    return (f"statements kept {tally['kept']}/{tally['statements']} (+{tally['kept by a try']} by a try) | questions right "
            f"{tally['right']}/{tally['asked']} (+{tally['right by a try']} by a try, {tally['shorter by a try']} shorter by a try)")


TALLY = lambda: {"statements": 0, "kept": 0, "kept by a try": 0, "asked": 0, "right": 0, "right by a try": 0, "shorter by a try": 0}


@torch.no_grad()
def inspect(A):
    """--inspect --load X: live a small world, then ask one question of each kind with every firing said"""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr = Translator().to(dev).eval()
    if A.init:
        tr.load_state_dict(torch.load(A.init, map_location=dev), strict=False)
    fm = FM(tr, Acting().to(dev), Heads().to(dev), Filter().to(dev), Critic().to(dev), same=A.same)
    fm.bank.see_lm = bool(A.lm_read)
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
    fm.bank.see_lm = bool(A.lm_read)
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
            print(f"\n--- statement {text!r}", flush=True)
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
    truth = {D(f"{leaf(nm, k)} is {v}.") for nm, e in w.ents.items() for k, v in e.items()}
    newest = [D(x) for x in newest]
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
    fm.bank.see_lm = bool(A.lm_read)
    load_fm(fm, A.load)
    for mod in (fm.bank, fm.heads, fm.filt, fm.critic):
        mod.eval()
    snap = snapshot(fm.bank)
    print(f"EVAL {A.load}: {evaluate(fm, A, A.seed + 1000)} | {fn_probe(fm, random.Random(1))}", flush=True)
    restore(fm.bank, snap)                             # (ev51: the probe's edits left every answer of the next eval 0/96)
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
    fm.bank.see_lm = bool(A.lm_read)
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


def overfit(A):
    """--overfit N: can the learning path memorize one fixed batch (a few statements, 8 questions, on a fixed teacher
    memory, skills stage)? Everything per iteration: each loss part, exact per purpose, said with the target's pieces
    and its own, gradient norm and weight change per module, each function node's own gradients; a full trace of one
    question at the start and the end. A path that works memorizes it; one that cannot has a bug"""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr = Translator().to(dev).eval()
    if A.init:
        tr.load_state_dict(torch.load(A.init, map_location=dev), strict=False)
    tr.requires_grad_(False)
    for p_ in tr_params(tr):
        p_.requires_grad_(True)
    fm = FM(tr, Acting().to(dev), Heads().to(dev), Filter().to(dev), Critic().to(dev), same=A.same)
    fm.bank.see_lm = bool(A.lm_read)
    rng, t0 = random.Random(A.seed), time.time()
    train_codec(fm, A, rng, torch.optim.AdamW(fm.heads.parameters(), lr=A.lr))
    print(f"codec done {time.time() - t0:.0f}s", flush=True)
    fm.stage = "skills"
    w = new_world(rng)
    T = Teacher(w)
    fm.g = Graph(fm.bank.d)
    st = {"step": 1}
    for _ in range(6):
        for d in world_step(w, rng):
            o = d()
            side = w.side.get(id(o), {}) if o else {}
            for text, meta in said(o, w) if o else []:
                with torch.no_grad():
                    demo_statement(fm, T, Lesson(), A, st, text, meta, side)
    stmts = []
    for d in world_step(w, rng):
        o = d()
        side = w.side.get(id(o), {}) if o else {}
        stmts += [(text, meta, side) for text, meta in (said(o, w) if o else [])]
    stmts = stmts[:8]
    Es = [e for e in (q_expr(k, w, rng) for k in ["lookup"] * 3 + ["chain"] * 2 + ["ops"] * 2 + ["before"]) if e is not None]
    print(f"memory {fm.g.n} places; batch: {len(stmts)} statements, {len(Es)} questions", flush=True)
    for E in Es:
        print("  Q:", ask_text(E), "| truth", truth_of(E, w), "| plan:", [s_[0] for s_ in plan(T, E)[0] or []], flush=True)
    g0 = clone_graph(fm.g)
    own = lambda: [p for mod in (fm.bank, fm.heads, fm.filt, fm.critic) for p in mod.parameters()]
    params = lambda: own() + tr_params(tr)
    make = lambda: torch.optim.AdamW([{"params": own()}, {"params": tr_params(tr), "lr": A.lr_tr}], lr=A.lr)
    opt = make()
    mods = {"interp": fm.bank.interp, "want": nn.ModuleList([fm.bank.wp, fm.bank.wv, fm.bank.wo]),
            "readkey": nn.ModuleList([fm.bank.wk, fm.bank.kq]), "fns": fm.bank.fns, "heads": fm.heads,
            "filter": fm.filt, "critic": fm.critic}
    norm = lambda ps: float(torch.sqrt(torch.as_tensor(sum((p.grad.float() ** 2).sum() for p in ps if p.grad is not None), dtype=torch.float)))

    def one(cap, trace):
        fm.g = clone_graph(g0)
        Tc = copy.copy(T)
        Tc.place, Tc.texts, Tc.did = dict(T.place), {k: list(v) for k, v in T.texts.items()}, {k: list(v) for k, v in T.did.items()}
        Ls, fm.stats, fm.capture, fm.said = Lesson(), {}, ([] if cap else None), []
        for text, meta, side in stmts:
            demo_statement(fm, Tc, Ls, A, st, text, meta, side)
        for i, E in enumerate(Es):
            fm.trace = trace and i == 0
            if fm.trace:
                print(f"\n######## TRACE question {ask_text(E)!r} truth {truth_of(E, w)}", flush=True)
            demo_question(fm, Tc, Ls, A, st, E)
        fm.trace = False
        seen = [D(x) for x in [t for t, _, _ in stmts] + [ask_text(E) for E in Es] + fm.said]
        z_, m_ = tr.content(seen)
        Ls.add("round trip", tr.decode_loss(z_, m_, seen))
        Ls.add("re-read", reread_loss(fm, z_, m_, seen, rng))
        return Ls

    for it in range(A.overfit + 1):
        ti, cap = time.time(), it % 10 == 0 or it == A.overfit
        before = {k: [p.detach().clone() for p in m.parameters()] for k, m in mods.items()} if it % 10 == 0 else None
        Ls = one(cap, it in (0, A.overfit))
        loss = Ls.total()
        if fm.made:
            old, opt = opt, make()
            for g_ in opt.param_groups:
                for p_ in g_["params"]:
                    if p_ in old.state:
                        opt.state[p_] = old.state[p_]
            fm.made = False
        opt.zero_grad()
        loss.backward()
        gn = {k: norm(m.parameters()) for k, m in mods.items()}
        gn["translator"] = norm(tr_params(tr))
        fg = [(t, f.calc, *[float(x.grad.norm()) if x.grad is not None else 0. for x in (f.key, f.code, f.au, f.av)]) for t, f in enumerate(fm.bank.fns)]
        total = float(torch.nn.utils.clip_grad_norm_(params(), 1.))
        opt.step()
        parts = {}
        for k, x in Ls.parts:
            parts[k] = parts.get(k, 0) + float(x)
        print(f"\n=== iter {it} ({time.time() - ti:.0f}s) loss {float(loss):.2f} grad-norm {total:.1f} | " +
              ", ".join(f"{k} {v:.2f}" for k, v in parts.items()), flush=True)
        print("   exact (own pieces): " + ", ".join(f"{k[:-5]} {fm.stats.get(k[:-5] + ' exact', 0)}/{v}" for k, v in sorted(fm.stats.items()) if k.endswith(" said")), flush=True)
        if cap:
            by = {}
            for c in fm.capture:
                b = by.setdefault(c[0], [0, 0, 0., 0])
                b[0] += 1; b[1] += int(c[1] == c[3]); b[2] += c[4]; b[3] += int(c[1] == c[2])
            print("   said (outputs: with its own pieces / with the target's pieces, mean output loss): " + " | ".join(
                f"{k} {v[0]}: own {v[1]} tgt {v[3]} loss {v[2] / v[0]:.2f}" for k, v in sorted(by.items())), flush=True)
            print("   e.g.: " + " || ".join(f"{c[0]}: want {c[1]!r} tgt-pieces {c[2]!r} own {c[3]!r}" for c in fm.capture[:14]), flush=True)
        print("   grad norm per module: " + ", ".join(f"{k} {v:.3g}" for k, v in gn.items()), flush=True)
        if before is not None:
            print("   weight change: " + ", ".join(f"{k} {float(torch.sqrt(sum(((a - b) ** 2).sum() for a, b in zip(m.parameters(), before[k])))):.3g}"
                                              for k, m in mods.items()), flush=True)
            print("   function nodes (t, calc, grad key/code/au/av): " + "; ".join(
                f"{t} {c or '-'} {a:.2g}/{b:.2g}/{c_:.2g}/{d_:.2g}" for t, c, a, b, c_, d_ in fg), flush=True)
    print(f"done {time.time() - t0:.0f}s", flush=True)


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
        r = getattr(fm, k).load_state_dict(s[k], strict=False)   # (a checkpoint from before a part existed: the
        if r.missing_keys or r.unexpected_keys:                   # part starts fresh, said here)
            print(f"load {k}: {len(r.missing_keys)} missing (e.g. {r.missing_keys[:2]}), {len(r.unexpected_keys)} unexpected", flush=True)
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
    fm.bank.see_lm = bool(A.lm_read)
    rng = random.Random(A.seed)
    if A.load:
        load_fm(fm, A.load)
    elif A.heads and os.path.exists(A.heads):          # the codec stage as an earlier run left it (12 of a run's 45 min)
        fm.heads.load_state_dict(torch.load(A.heads, map_location=dev))
        print(f"codec stage read from {A.heads}", flush=True)
    else:
        train_codec(fm, A, rng, torch.optim.AdamW(fm.heads.parameters(), lr=A.lr))
        if A.heads:
            torch.save(fm.heads.state_dict(), A.heads)
    rng = random.Random(A.seed + 1)                    # (the same demo worlds whether the codec stage ran or was read)
    torch.manual_seed(A.seed + 1)
    fnp = lambda: {id(p) for f in fm.bank.fns for p in (f.code, f.au, f.av)}   # (the keys stay on Adam: fm22, the
    #                                                writer's keys never cleared the gate living -- 268/268 statements rested)
    own = lambda: [p for mod in (fm.bank, fm.heads, fm.filt, fm.critic) for p in mod.parameters()
                   if not (A.writer and id(p) in fnp())]   # functions' own weights: the update writer's (--writer)
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
        if A.local and st["step"] > A.demo:            # living: only the nodes' own weights learn (program, action
            mine = {id(x) for f in fm.bank.fns for x in (f.code, f.au, f.av, f.key)}   # correction, label); the shared
            for x in params():                         # body stays as the demos left it (11-12, fm43: living's own
                if id(x) not in mine:                  # lessons were ~0, yet check fails 0 -> 244 and every question
                    x.grad = None                      # 0/24 by @100: the translator kept training and every node's
        #                                                meaning drifted under the interpreter)
        if fm.made:                                    # a node made in this chunk joins before the step: its first
            remake()                                   # try trains it too (fm15 to step ~50: its own gradients were
            fm.made = False                            # dropped, the optimizer took it in only after)
        norm = torch.nn.utils.clip_grad_norm_(params(), 1.)
        if bool(torch.isfinite(norm)):
            O["opt"].step()
            fm.count("updates")
            if A.writer:                               # the functions' own weights: written, not stepped by Adam
                write_fns(fm)
        else:
            fm.count("skipped (gradient not finite)")
            if fm.stats["skipped (gradient not finite)"] <= 5:
                print(f"step {st['step']}: gradient not finite (loss {float(loss):.2f})", flush=True)

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
        x, xm = pad([fm.heads.reread(r) for r in xs])
        return tr.decode_loss(x, xm, ts)

    st, t0, recent, tally = {"step": 0}, time.time(), [], TALLY()
    w = T = None
    for step in range(1, A.steps + 1):
        st["step"] = step
        if w is None or (step <= A.demo and (step - 1) % A.world == 0) or (step > A.demo and (step - A.demo - 1) % 150 == 0):
            #                                            a new world now and then: names, keys, numbers all new (demos:
            #                                            every --world steps -- dm29: one world in 50 demo steps, and a
            #                                            fresh world's statements all rested; living: a fresh one first)
            w = new_world(rng)
            T = Teacher(w)
            fm.g = Graph(fm.bank.d)
        acc, seen, n0 = {}, [], fm.stats.get("steps run", 0)
        stage = "skills" if step <= A.skill else "route" if step <= A.skill + A.route else "free"
        if step <= A.demo and (stage != fm.stage or step == 1):
            print(f"step {step}: stage {stage}", flush=True)
        fm.stage = stage
        if step > A.demo:                              # user 11-10: as little supervision as possible, ~50 steps of
            if step == A.demo + 1:                     # demos; then it lives: the outcome is the only teacher
                print(f"step {step}: the demos end; it lives by itself (signals only)", flush=True)
                fm.look = 6
            fm.show = step - A.demo <= A.show_live     # (--show_live: the first living steps, every firing said)
            seen += live_tick(fm, A, w, rng, lambda Ls_: update(Ls_, acc), tally)
            fm.show = False
            Ls = Lesson()                              # (the translator still reads back what it read)
        else:
            Ls = Lesson()                              # the statements: one update
            for d in world_step(w, rng):
                o = d()
                side = w.side.get(id(o), {}) if o else {}
                for text, meta in said(o, w) if o else []:
                    demo_statement(fm, T, Ls, A, st, text, meta, side)
                    seen.append(text)
            finish(fm, Ls, rng, recent)
            update(Ls, acc)
            if step <= A.warm:                         # warm start (REALM's retriever warm start; induction-head
                fm.look = min(6, 2 + 4 * step // max(1, A.warm))   # plateaus need many examples): lookups in a batch,
                Ls = Lesson()                                      # one update; the walk brings 2 places at first, 6 at
                for _ in range(A.warm_q):                          # the end (fewer look-alikes first). fm24: ~6 lookups
                    E = q_expr("lookup", w, rng)                   # an update, stuck at 16-21% = chance
                    if E is not None:
                        demo_question(fm, T, Ls, A, st, E)
                        seen.append(ask_text(E))
                finish(fm, Ls, rng, recent)
                update(Ls, acc)
            for _ in range(A.qb):                      # each question: one update
                E = q_expr(rng.choice(["lookup", "before", "coexist", "chain", "ops", "ops"]), w, rng)
                if E is not None:
                    Ls = Lesson()
                    demo_question(fm, T, Ls, A, st, E)
                    seen.append(ask_text(E))
                    finish(fm, Ls, rng, recent)
                    update(Ls, acc)
            Ls = Lesson()                              # the functions on functions and the translator: one update
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
            Ls.add("re-read", reread_loss(fm, z_, m_, seen, rng))
        update(Ls, acc)
        for _ in range(A.replay if step <= A.demo else A.replay_live):   # the demos' routing, again (one small update
            Lr = Lesson()                                    # each; after the step's last update: fm33 replayed while the
            Lr.add("route replay", route_replay(fm, rng))    # functions' lesson still waited for its backward). Living too
            Lr.add("skill replay", skill_replay(fm, rng))    # ... and what the demos' steps made, relived. In the demos
            #                                                  too (11-12, fm49 / fm51 / fm52: 3 or 5 layers, with or
            #                                                  without the LM's reading, lookup exact in the late demos
            #                                                  stayed 35-39% -- the reader learned from each demo step
            #                                                  once, the routing 20 times). Nothing new is shown
            update(Lr, acc)                                  # (fm34: @50 statements stored, questions fired; living's only
            #                                                  signal was its statements (questions 0/199), and @100 every
            #                                                  question rested again -- the demos anchor what living lacks)
        if step % 50 == 0 and step > A.demo:
            print(f"step {step} living: {living_line(tally)} | own right rate " + ", ".join(f"{k} {v:.2f}" for k, v in sorted(fm.rates.items())) + " [" + ", ".join(f"{k} {v:.2f}" for k, v in acc.items()) +
                  f"] made nodes that act {len(fm.bank.fns)} | " + ", ".join(
                      f"{k} {v}" for k, v in sorted(fm.stats.items()) if k in ("fires", "rest", "no change", "budget out", "check fails",
                                                                         "new functions", "edits", "updates", "roundtrip"))
                  + f" | {time.time() - t0:.0f}s", flush=True)
            fm.stats, tally = {}, TALLY()
        elif step % 25 == 0 and step <= A.demo:
            parts = acc
            loss = sum(acc.values())
            n_run = max(1, fm.stats.get("steps run", 0) - n0)
            print(f"step {step} loss {loss:.2f} (functions per demo step {acc.get('functions', 0) / n_run:.2f}, "
                  f"{fm.stats.get('updates', 0)} updates, {fm.stats.get('cannot be made of pieces', 0)} targets not made of pieces) [" +
                  ", ".join(f"{k} {v:.2f}" for k, v in parts.items()) +
                  f"] made nodes that act {len(fm.bank.fns)} | who acts: " + ", ".join(
                      f"{k[5:]} {v}" for k, v in sorted(fm.stats.items()) if k.startswith("acts")) + " | exact: " + ", ".join(
                      f"{k[:-5]} {fm.stats.get(k[:-5] + ' exact', 0)}/{v}" for k, v in sorted(fm.stats.items()) if k.endswith(" said")) +
                  f" (lookup chance {100 * fm.stats.get('lookup chance', 0) / max(1, fm.stats.get('lookup said', 0)):.0f}%)" +
                  " | where right: " + ", ".join(
                      f"{k[:-6]} {fm.stats.get(k + ' right', 0)}/{v}" for k, v in sorted(fm.stats.items()) if k.endswith(" where")) +
                  " | inputs the walk brought: " + ", ".join(
                      f"{k[:-3]} {fm.stats.get(k[:-3] + ' walked', 0)}/{v}" for k, v in sorted(fm.stats.items()) if k.endswith(" in")) +
                  f" | functions written {fm.stats.get('functions written', 0)}"
                  + (f" (writer step {[round(float(x), 6) for x in torch.exp(fm.bank.writer.lr)]}, gain {[round(float(x), 5) for x in torch.exp(fm.bank.writer.gain)]})" if A.writer else "")
                  + f" | stage {fm.stage}, routing reaches the skill's node "
                  f"{fm.stats.get('routing reaches the skill' + chr(39) + 's node', 0)}/{fm.stats.get('routing checked', 0)} | {time.time() - t0:.0f}s", flush=True)
            fm.stats = {}
        if step % A.every == 0 or step == A.steps:
            commit_all(fm)                             # what is pending becomes the weights before it is judged / saved
            for mod in (fm.bank, fm.heads, fm.filt, fm.critic):
                mod.eval()
            keep, snap = (fm.g, fm.stats), snapshot(fm.bank)
            print(f"EVAL step {step}: {evaluate(fm, A, A.seed + 1000 + step)} | {fn_probe(fm, random.Random(step))}", flush=True)
            restore(fm.bank, snap)                     # (the probe edits functions: the next eval starts as this one did)
            fm.calc_on = False
            print(f"EVAL step {step}, calculator removed (its mimic answers): {evaluate(fm, A, A.seed + 1000 + step)}", flush=True)
            fm.calc_on = True
            restore(fm.bank, snap)
            fm.g, fm.stats = keep
            for mod in (fm.bank, fm.heads, fm.filt, fm.critic):
                mod.train()
            if A.save:
                save_fm(fm, A.save)
    if A.live:                                         # then a fresh world it lives in by itself (signals only)
        w, fm.g, acc, tally = new_world(rng), Graph(fm.bank.d), {}, TALLY()
        for tick in range(1, A.live + 1):
            live_tick(fm, A, w, rng, lambda Ls_: update(Ls_, acc), tally)
            if tick % 25 == 0:
                print(f"live {tick}: {living_line(tally)} | {time.time() - t0:.0f}s", flush=True)
                tally, acc = TALLY(), {}
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
    P.add_argument("--overfit", type=int, default=0, help="iterations of the fixed-batch memorization test")
    P.add_argument("--trace", action="store_true", help="with --load: every trace, said (translator, demos, routing, living)")
    P.add_argument("--codec_check", action="store_true", help="the translator, the heads (and with --load each function) on every kind of text")
    P.add_argument("--same", type=float, default=.1, help="a node whose label moved less than this did not change: no signal")
    P.add_argument("--init", default="", help="translator weights (text -> B, B -> text)")
    P.add_argument("--load", default="", help="a saved module (skips the codec stage)")
    P.add_argument("--heads", default="", help="the codec stage's result: read from this file when it is there, else written to it")
    P.add_argument("--save", default="")
    P.add_argument("--log", default="")
    P.add_argument("--seed", type=int, default=0)
    P.add_argument("--codec", type=int, default=1500, help="steps of the codec's second stage")
    P.add_argument("--steps", type=int, default=3000)
    P.add_argument("--every", type=int, default=500)
    P.add_argument("--lr", type=float, default=2e-4)
    P.add_argument("--lr_tr", type=float, default=2e-5, help="the translator's own layers (encoder, decoder)")
    P.add_argument("--qb", type=int, default=4, help="questions per step")
    P.add_argument("--warm", type=int, default=50, help="steps of the warm start: a batch of lookups an update")
    P.add_argument("--warm_q", type=int, default=16, help="lookups in each warm-start update")
    P.add_argument("--replay", type=int, default=20, help="routing replays after each demo step (want reader + labels)")
    P.add_argument("--show_live", type=int, default=0, help="living steps said in full (every firing, every verdict)")
    P.add_argument("--lm_read", type=int, default=1, help="0: the interpreter does not see what the LM made of its inputs (a split run)")
    P.add_argument("--local", type=int, default=1, help="living: only the nodes' own weights learn (shared body fixed)")
    P.add_argument("--replay_live", type=int, default=5, help="routing replays of the demos after each living step")
    P.add_argument("--q", type=int, default=24, help="questions per kind at eval")
    P.add_argument("--ticks", type=int, default=30, help="world ticks the module lives through before eval questions")
    P.add_argument("--sal_sup", type=float, default=1., help="weight of the teacher's hint on what is salient")
    P.add_argument("--exact", type=float, default=.3, help="decode loss under which an output counts as said exactly (labels)")
    P.add_argument("--mse", type=float, default=1., help="weight of a made piece's latents against the encoder's for those words")
    P.add_argument("--own", type=int, default=600, help="after this step the demo memory holds what the store itself wrote")
    P.add_argument("--scope", type=float, default=.9)
    P.add_argument("--skill", type=int, default=35, help="steps of stage 1: the teacher says which node does a purpose")
    P.add_argument("--route", type=int, default=15, help="steps of stage 2: routing chooses, a light hint toward that node")
    P.add_argument("--world", type=int, default=5, help="demo steps per world (a new world: names, keys, numbers)")
    P.add_argument("--demo", type=int, default=50, help="steps with demos (user 11-10: ~50); after them it lives, signals only")
    P.add_argument("--writer", type=int, default=1, help="1: the functions' own weights are written by the update writer")
    P.add_argument("--try_read", type=float, default=.1, help="share of demo steps where the read node nearest the want acts instead")
    P.add_argument("--kappa", type=float, default=2.)
    P.add_argument("--tool_good", type=float, default=.3)
    P.add_argument("--live", type=int, default=100, help="ticks it then lives in a fresh world (signals only)")
    P.add_argument("--lq", type=int, default=4, help="questions per living tick")
    P.add_argument("--tries", type=int, default=6, help="sampled tries (whole actions sampled): every question, a wrong statement")
    P.add_argument("--explore", type=float, default=1., help="sampling temperature of a try")
    A = P.parse_args()
    random.seed(A.seed), np.random.seed(A.seed), torch.manual_seed(A.seed)
    (overfit if A.overfit else deep_trace if A.trace else inspect if A.inspect else eval_only if A.eval else codec_check if A.codec_check else run_fm)(A)
