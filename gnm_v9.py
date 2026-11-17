"""GNM v9 (docs/GNM_v9_design.md): GNM holds what is true and what to do -- facts, things, relations, state,
procedures -- as nodes the model writes and edits; the frozen LM (Walker 1B target, GPT-2 stands in) is the executor.
One loop: a change -> fetch related nodes -> LM step over them -> write nodes (new / edit) or say. No node-level step
data: signals are blanked spans recovered from memory, round trip, and (later) checked outcomes and own wins.
The world below is kept only as a source of checkable questions (later tests). Old paths: archive/gnm/.

  python gnm.py e1 --steps 3000 --save /marimo/ckpt/e1.pt     # reading: species articles in, blanked spans out
"""
import argparse
import collections
import gzip
import json
import math
import os
import random
import re
import time

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


# ====================================================================== the world's questions and streams
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


OPF = {n: f for n, f, _ in TOOLS}


# ====================================================================== v9: GNM holds what is true and what to do; the LM executes
# docs/GNM_v9_design.md. Code here holds no domain word: no task, field, step or answer kind. Built so far: the node
# store, the loop step (retrieve -> LM step -> write / say), edits, relations, the wake rule, L1 (blanked spans from
# memory), L2 (round trip), the write cost. Not built yet: L3-L6 (outcome, own wins as procedures, curriculum, hints).
dev = "cuda" if torch.cuda.is_available() else "cpu"
LM_NAME = os.environ.get("LM", "gpt2")
_LM = {}


def lm():
    """the executor (frozen): Walker 1B is the target, GPT-2 stands in"""
    if not _LM:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tk = AutoTokenizer.from_pretrained(LM_NAME)
        m = AutoModelForCausalLM.from_pretrained(LM_NAME, attn_implementation="eager").to(dev).eval()   # (eager: its attention
        #                                                                       weights give a node's links)
        for p in m.parameters():
            p.requires_grad_(False)
        _LM.update(tk=tk, m=m)
    return _LM["tk"], _LM["m"]


class Memory:
    """the node store. A node = value rows V (vectors in the LM's embedding space: the LM reads them directly), key k
    (its learned label), relations R (the nodes read when it was written, with weights), strength s, optional text"""

    def __init__(self):
        self.V, self.k, self.R, self.s, self.text = [], [], [], [], []
        self.by = {}                                   # node -> the function that made it

    @property
    def n(self):
        return len(self.V)

    def add(self, V, k, R):
        self.V.append(V)
        self.k.append(k)
        self.R.append(R)
        self.s.append(1.)
        self.text.append(None)
        return self.n - 1

    def replace(self, i, V, k, R):                 # an edit: the new node takes the old one's place
        self.V[i], self.k[i], self.R[i] = V, k, R

    def neighbours(self, i):                       # one hop along relations, both ways
        out = {j for j, _ in self.R[i]}
        out |= {j for j in range(self.n) if any(t == i for t, _ in self.R[j])}
        return out - {i}


class GNM(nn.Module):
    """the trained parts around the frozen LM: retrieval query, write head (K slots of m rows), keys, gates, pointer
    (which fetched node a slot replaces), relations from the write's own attention"""

    def __init__(self, K=4, m=16, dk=256, topk=8):
        super().__init__()
        tk, M = lm()
        d = M.config.n_embd
        self.K, self.m, self.d, self.topk = K, m, d, topk
        self.slotq = nn.Parameter(torch.randn(K * m, d) * .02)
        self.att = nn.Linear(d, d, bias=False)
        self.val = nn.Linear(d, d)
        self.key = nn.Linear(d, dk)
        self.qry = nn.Linear(d, dk)
        self.gate = nn.Linear(d, 1)
        self.ptr = nn.Linear(d, dk)
        self.new = nn.Parameter(torch.full((1,), 3.))      # start: a write is new (~.7 vs 8 fetched); editing is learned
        #                                                     (11-14 review: at 0 an untrained pointer edited ~70% of the time)
        self.mark = nn.Parameter(torch.zeros(d))            # added to memory rows: "this is memory"
        self.tau = nn.Parameter(torch.tensor(math.log(8.)))   # sharpness of the slots' attention (11-14 trace: at 1x it
        #                                                       stayed ~1/T after 1000 steps -- slots never picked tokens)
        self.wmode = os.environ.get("WRITE", "fire")          # fire: a node = what a function's firing PRODUCED (§8); span: the copier (control)
        self.spanq = nn.Parameter(torch.randn(K, d) * .02)   # each slot's query for where its span starts
        self.symkey = os.environ.get("SYMKEY", "1") == "1"   # node keys from the query function on the node's tokens
        self.order = os.environ.get("RET_ORDER", "time")      # fetched nodes given to the LM in write order ('score' = by key)
        self.ctxfix = os.environ.get("CTXFIX", "0") == "1"    # the learned context correction on rows (off: it let the
        #                                                       write skip picking tokens; trace 11-14)
        nn.init.zeros_(self.val.weight)                      # rows start as pure mixtures of the step's token vectors
        nn.init.zeros_(self.val.bias)
        self.register_buffer("hmu", torch.zeros(d))         # running mean of the LM's states (11-14 inspect: sentence
        #                                                     means of GPT-2's last layer had cos .995 -- a constant)
        nn.init.constant_(self.gate.bias, 1.)                # start by writing; the cost teaches restraint
        self.cost = float(os.environ.get("WRITE_COST", ".001"))
        self.dup_cost = float(os.environ.get("DUP_COST", ".02"))   # per token stored twice
        self.FI = nn.Parameter(torch.randn(32, 4, d) * .11)   # §8 F1: function nodes (instruction rows), 32 slots
        self.FK = nn.Parameter(F.normalize(torch.randn(32, dk), dim=-1))   # their labels (learned; spawn sets the want)
        self.register_buffer("n_fn", torch.tensor(4))        # active functions (spawned up to 32)
        self._fk = None
        self.spawn_at = float(os.environ.get("SPAWN_AT", ".3"))
        self.m_out = int(os.environ.get("M_OUT", "8"))       # rows a firing produces
        self.tau_o = 1.
        self.fire_cost = float(os.environ.get("FIRE_COST", ".01"))

    # ---------------------------------------------------------------- the LM
    def embed(self, ids):
        return lm()[1].transformer.wte(torch.as_tensor(ids, device=dev))

    def run(self, embs, attn=False):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
            out = lm()[1](inputs_embeds=embs[None], output_hidden_states=True, output_attentions=attn)
        if attn:                                           # where the LM looked: mean over heads of its last 4 layers
            Att = torch.stack([a[0].float().mean(0) for a in out.attentions[-4:]]).mean(0).detach()
            return out.hidden_states[-1][0].float(), out.logits[0].float(), Att
        return out.hidden_states[-1][0].float(), out.logits[0].float()

    def center(self, H):
        """the LM's states minus their running mean (what differs between inputs, not what all share), then layer-
        normalized (11-14 trace: GPT-2's raw last-layer values -- some dims in the hundreds -- saturated the gates shut,
        made every slot attend alike, and blew up the value projection: blank nll 3.6 -> 7.6)"""
        if self.training:
            with torch.no_grad():
                self.hmu.mul_(.99).add_(.01 * H.mean(0))
        return F.layer_norm(H - self.hmu, (H.shape[-1],))

    # ---------------------------------------------------------------- retrieve
    def query(self, ids):
        h, _ = self.run(self.embed(ids))
        return F.normalize(self.qry(self.center(h).mean(0)), dim=-1)

    def retrieve(self, mem, q):
        """top-k nodes by key, plus their one-hop neighbours; rows carry the key score's gradient"""
        if not mem.n:
            return [], None, []
        sc = torch.stack(mem.k) @ q
        top = sc.topk(min(self.topk, mem.n)).indices.tolist()
        p = F.softmax(sc / .1, 0)
        hop = []
        for i in top:
            for j in sorted(mem.neighbours(i)):
                if j not in top and j not in hop and len(hop) < self.topk // 2:
                    hop.append(j)
        sel, rows, owner = top + hop, [], []
        if self.order == "time":                           # memory comes back in the order it was written (11-14 oracle:
            sel = sorted(sel)                              # pieces in score order capped the LM at .475 token acc)
        for i in sel:
            w = p[i] / p[i].detach() if i in top else 1.
            rows.append(mem.V[i] * w + self.mark)
            owner += [i] * len(mem.V[i])                    # (11-15: was self.m -- firing outputs have m_out rows)
        return sel, torch.cat(rows), owner

    # ---------------------------------------------------------------- write (the loop's output)
    def write(self, mem, sel, owner, H, embs, n_rows, train, Att=None):
        """K slots of m rows; each row attends over the step (fetched memory rows + the change): WHAT it takes is chosen
        from the LM's centred states, what it holds is the step's own token vectors (+ a learned context correction,
        starting at zero) -- a node is a group of token vectors the LM reads natively (11-14 inspect: rows made from
        the LM's last-layer states were one constant vector). Then key, gate, pointer (replace a fetched node / new),
        relations (attention mass on each fetched node).
        Train (soft, so both decisions learn from the loss -- 11-14 e1: a hard untrained pointer overwrote memory down to
        ~2 nodes an article and the gates gave up): a slot's rows are scaled by its gate; its edit is a blend -- each
        fetched node moves toward the slot by gate x pointer probability, the rest goes to a new node.
        Eval: open slots only (gate > .5); the pointer's best choice replaces that node, or 'new' adds one."""
        K, m, d = self.K, self.m, self.d
        Hc = self.center(H)
        T = embs.shape[0]
        if self.wmode == "span":
            # each slot picks where its span starts; row j reads the token at start + j (in order, exact token vectors;
            # 11-14 oracle: sentence-sized nodes in write order are read at .945 vs .981 in context -- independent soft
            # rows lose order and identity). The start is a distribution, so the loss teaches where to cut.
            lg_s = self.spanq @ self.att(Hc).T / d ** .5 * self.tau.exp()                       # (K, T)
            if n_rows:                                       # a new node's words come from the new input; fetched memory
                lg_s[:, :n_rows] = -1e9                      # is context (11-14 trace: spans re-copied old nodes instead)
            Ps = F.softmax(lg_s, -1)
            Ps = F.one_hot(Ps.argmax(-1), T).float() + Ps - Ps.detach()  # hard start forward (the exact tokens: 11-14
            #                                                              trace -- a flat start blurred rows into junk),
            #                                                              gradient through the start probabilities
            idx = torch.arange(T, device=dev)[:, None] + torch.arange(m, device=dev)[None]   # (T, m)
            win = lambda Z: torch.cat([Z, Z.new_zeros(m, Z.shape[-1])])[idx]                 # (T, m, d)
            rows = torch.einsum("kt,tmd->kmd", Ps, win(embs))
            X = torch.einsum("kt,tmd->kmd", Ps, win(Hc))
            pooled = X.mean(1)
            A = torch.zeros(K, m, T, device=dev)
            for j in range(min(m, T)):                                # (11-14: inputs shorter than m crashed here)
                A[:, j, j:] = Ps[:, :T - j]
            A = A.view(K * m, T)
        else:
            A = F.softmax(self.slotq @ self.att(Hc).T / d ** .5 * self.tau.exp(), -1)   # (K*m, T)
            X = A @ Hc
            rows = (A @ embs + (self.val(X) if self.ctxfix else 0)).view(K, m, d)
            pooled = X.view(K, m, d).mean(1)
        self._A, self._H = A.detach(), Hc.detach()                       # (inspection only)
        g = torch.sigmoid(self.gate(pooled)).squeeze(-1)
        if self.symkey:                                  # a node's key = the query function on its own tokens: it is
            k = torch.stack([F.normalize(self.qry(self.center(self.run(rows[j])[0]).mean(0)), dim=-1)   # labelled the
                             for j in range(K)])        # way it is asked for (11-14: separate key / query projections
            #                                             found the own sentence 33 / 60 in a 32-node memory; the
            #                                             oracle, keyed this way untrained, 351 / 360)
        else:
            k = F.normalize(self.key(pooled), dim=-1)
        P = None
        if sel:
            fk = torch.stack([mem.k[i] for i in sel])
            P = F.softmax(torch.cat([F.normalize(self.ptr(pooled), dim=-1) @ fk.T / .1, self.new.expand(K, 1)], 1), -1)
        mass = A[:, :n_rows].view(K, m, n_rows).sum(1).detach() if n_rows else None   # (relations: read, not trained)
        link = {}
        if self.wmode == "span" and n_rows and Att is not None:   # a node links to the fetched nodes the LM looked at
            for j in range(K):                                    # while reading its span (share of the span's attention
                b0 = int(Ps[j].argmax())                          # on memory; kept when above an even share). 11-14
                span = list(range(b0, min(T, b0 + m)))            # trace: links from the pointer were all empty, so
                tot = float(Att[span, :n_rows].sum()) + 1e-9      # nothing was ever woken
                share = {i: float(Att[span][:, [t for t, o in enumerate(owner) if o == i]].sum()) / tot for i in sorted(set(owner))}
                link[j] = [(i, w) for i, w in share.items() if w >= 1. / len(share)]
            mass = None
        dup = 0.
        if self.wmode == "span":                                  # storing the same tokens twice costs (11-14 trace: 3
            cover = sum(F.pad(Ps[:, :T - k], (k, 0)) for k in range(min(m, T)))   # of 4 slots copied one phrase and the
            dup = F.relu((g[:, None] * cover).sum(0) - 1).sum()   # subject went unstored)
        written = []
        for j in range(K):
            if not train and g[j] <= .5:
                continue
            R = link.get(j, [])
            if mass is not None:
                for i in sorted(set(owner)):
                    w = float(mass[j, [t for t, o in enumerate(owner) if o == i]].sum()) / m   # (11-14 trace: was summed over m rows, > 1)
                    if w > .1:
                        R.append((i, w))
            if train:
                p_new = P[j, -1] if P is not None else 1.
                for c, i in enumerate(sel):                                # soft edit of each fetched node
                    a = g[j] * P[j, c]
                    mem.V[i] = mem.V[i] * (1 - a) + rows[j] * a
                    mem.k[i] = F.normalize(mem.k[i] * (1 - a) + k[j] * a, dim=-1)
                written.append(mem.add(rows[j] * g[j] * p_new, k[j], R))
            elif P is not None and int(P[j].argmax()) < len(sel):
                i = sel[int(P[j].argmax())]
                mem.replace(i, rows[j], k[j], R)
                written.append(i)
            else:
                written.append(mem.add(rows[j], k[j], R))
        return written, self.cost * g.sum() + self.dup_cost * dup

    def step(self, mem, ids, train=False):
        """one turn of the loop on a change given as text: retrieve -> LM step -> write. Returns the nodes written,
        the nodes woken (what the written nodes were written from: they are the next changes), the write cost"""
        sel, rows, owner = self.retrieve(mem, self.query(ids))
        embs = self.embed(ids) if rows is None else torch.cat([rows, self.embed(ids)])
        H, _, Att = self.run(embs, attn=True)
        written, cost = self.write(mem, sel, owner, H, embs, 0 if rows is None else len(rows), train, Att)
        woken = sorted({i for w in written for i, _ in mem.R[w]} - set(written))
        return written, woken, cost

    # ---------------------------------------------------------------- firing (§8): a function node, executed by the LM
    def rows_query(self, rows):
        """the query function on rows (a change, a node, a function's instruction): what it is about"""
        h, _ = self.run(rows)
        return F.normalize(self.qry(self.center(h).mean(0)), dim=-1)

    def fn_keys(self):
        """F2: each active function's label (learned; a spawned function is labelled with the want that needed it --
        11-15 trace: labels made by the LM from random instruction rows were all alike, every function scored .99)"""
        return F.normalize(self.FK[:int(self.n_fn)], dim=-1)

    def route(self, q, train):
        """F1 / F3: the change's signal meets the functions' keys; none good enough -> spawn one (copy of the best)"""
        sc = self.fn_keys() @ q
        n = len(sc)
        if train and float(sc.max()) < self.spawn_at and n < self.FI.shape[0]:
            with torch.no_grad():                            # a new function, made for this want: its label is the want
                b = int(sc.argmax())                         # (11-15: a copy of the best kept the low score and spawned
                self.FI.data[n] = self.FI.data[b] + torch.randn_like(self.FI.data[b]) * .02   # again, recursively, out of memory)
                self.FK.data[n] = q.detach()                 # (.data: the slot changes outside this episode's graph)
                self.n_fn += 1
            return self.route(q, train)
        p = F.softmax(sc / .1, 0)
        f = int(torch.multinomial(p, 1)) if train else int(p.argmax())
        return f, p[f] / p[f].detach(), float(sc[f])       # (value 1; the gradient reaches the routing score)

    def fire(self, mem, change_rows, change_ids, train):
        """F4-F6: the LM reads [function rows + fetched nodes + the change] and PRODUCES the output node: m_out decode
        steps, each row = the expected token vector of its next-token distribution (nothing copied from the input)"""
        M = lm()[1]
        wte = M.transformer.wte.weight
        q = self.rows_query(change_rows)
        f, w, fsc = self.route(q, train)
        sel, rows, owner = self.retrieve(mem, q)
        n_i, n_rows = self.FI.shape[1], (0 if rows is None else len(rows))
        inp = torch.cat([self.FI[f] * w] + ([rows] if rows is not None else []) + [change_rows])
        ac = torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda")
        with ac:
            o = M(inputs_embeds=inp[None], use_cache=True, output_attentions=True, output_hidden_states=True)
        outs, ids_out = [], []
        att_mem = torch.zeros(max(1, n_rows), device=dev)
        for t in range(self.m_out):
            p = F.softmax(o.logits[0, -1].float() / self.tau_o, -1)
            top = int(p.argmax())
            e = wte[top] + (p - p.detach()) @ wte            # the token it produces, forward; gradient through its
            outs.append(e)                                   # probabilities (11-15 trace: expected vectors drifted to
            ids_out.append(top)                              # "the the the")
            with ac:
                o = M(inputs_embeds=e[None, None], past_key_values=o.past_key_values, use_cache=True,
                      output_attentions=True, output_hidden_states=True)
            if n_rows:                                       # where the LM looked while producing this row
                a = torch.stack([x[0, :, -1].float().mean(0) for x in o.attentions[-4:]]).mean(0)
                att_mem += a[n_i:n_i + n_rows].detach()
        h_last = self.center(o.hidden_states[-1][0, -1:].float())[0]
        V = torch.stack(outs)
        g = torch.sigmoid(self.gate(h_last)).squeeze()
        k = self.rows_query(V + self.mark)
        R = []
        if n_rows:                                           # F6 links: fetched nodes it attended to, above an even share
            tot = float(att_mem.sum()) + 1e-9
            share = {i: float(att_mem[[t for t, o_ in enumerate(owner) if o_ == i]].sum()) / tot for i in sorted(set(owner))}
            R = [(i, s) for i, s in share.items() if s >= 1. / len(share)]
        P = None
        if sel:                                              # F6 pointer: replace a fetched node, or new
            fk = torch.stack([mem.k[i] for i in sel])
            P = F.softmax(torch.cat([F.normalize(self.ptr(h_last), dim=-1) @ fk.T / .1, self.new.reshape(1)]), -1)
        written = None
        if train:
            for c, i in enumerate(sel):                      # soft edit of each fetched node
                a_ = g * P[c]
                mem.V[i] = mem.V[i] * (1 - a_) + V * a_ if mem.V[i].shape == V.shape else mem.V[i]
                mem.k[i] = F.normalize(mem.k[i] * (1 - a_) + k * a_, dim=-1)
            written = mem.add(V * g * (P[-1] if P is not None else 1.), k, R)
        elif g > .5:
            if P is not None and int(P.argmax()) < len(sel):
                written = sel[int(P.argmax())]
                mem.replace(written, V, k, R)
            else:
                written = mem.add(V, k, R)
        if written is not None:
            mem.text[written] = ids_out
            mem.by[written] = f
        seen_in = set(change_ids) | {t for i in sel for t in (mem.text[i] or [])}
        novelty = sum(t not in seen_in for t in ids_out) / len(ids_out)
        woken = sorted({i for i, _ in R} - ({written} if written is not None else set()))
        return dict(written=written, woken=woken, cost=self.cost * g + self.fire_cost, f=f, fsc=fsc, ids=ids_out,
                    novelty=novelty, sel=sel, R=R, g=float(g))

    def read_fire(self, mem, ids, train, budget=3):
        """F7 / F8: reading = firing(s) on the input; each output is a change (it fires next, and wakes what it links
        to); the raw input is never stored"""
        queue, infos, cost = [(self.embed(ids), list(ids))], [], 0.
        while queue and len(infos) < budget:
            rows_c, ids_c = queue.pop(0)
            r = self.fire(mem, rows_c, ids_c, train)
            infos.append(r)
            cost = cost + r["cost"]
            if r["written"] is not None and (train or r["g"] > .5):
                i = r["written"]
                queue.append((mem.V[i] + self.mark, mem.text[i] or []))
                for wk in r["woken"]:
                    queue.append((mem.V[wk] + self.mark, mem.text[wk] or []))
        return infos, cost

    # ---------------------------------------------------------------- say (the LM speaks from memory)
    def say_nll(self, mem, prompt, target):
        """teacher-forced: the LM's log-prob of the target after [fetched memory] + prompt; and its token accuracy"""
        sel, rows, _ = self.retrieve(mem, self.query(prompt))
        pe, te = self.embed(prompt), self.embed(target)
        embs = torch.cat([pe, te]) if rows is None else torch.cat([rows, pe, te])
        _, lg = self.run(embs)
        P = (0 if rows is None else len(rows)) + len(prompt)
        lp = F.log_softmax(lg[P - 1:P - 1 + len(target)], -1)
        tgt = torch.as_tensor(target, device=dev)
        return -lp.gather(1, tgt[:, None]).mean(), float((lp.argmax(-1) == tgt).float().mean())

    @torch.no_grad()
    def say(self, mem, prompt, n):
        sel, rows, _ = self.retrieve(mem, self.query(prompt))
        embs = self.embed(prompt) if rows is None else torch.cat([rows, self.embed(prompt)])
        out = []
        for _ in range(n):
            _, lg = self.run(embs)
            t = int(lg[-1].argmax())
            out.append(t)
            embs = torch.cat([embs, self.embed([t])])
        return out

    def roundtrip_nll(self, mem, nodes, ids):
        """L2: the nodes written from a sentence, read alone by the LM, must give the sentence back"""
        rows = torch.cat([mem.V[i] + self.mark for i in nodes])
        _, lg = self.run(torch.cat([rows, self.embed(ids)]))
        lp = F.log_softmax(lg[len(rows) - 1:len(rows) - 1 + len(ids)], -1)
        return -lp.gather(1, torch.as_tensor(ids, device=dev)[:, None]).mean()

    @torch.no_grad()
    def node_text(self, mem, i, n=14):
        """what a node says when the LM reads it alone (inspection only)"""
        embs = mem.V[i] + self.mark
        out = []
        for _ in range(n):
            _, lg = self.run(embs)
            t = int(lg[-1].argmax())
            out.append(t)
            embs = torch.cat([embs, self.embed([t])])
        return lm()[0].decode(out)


# ====================================================================== E1: reading (docs/GNM_v9_design.md §5)
def clean(s):
    return (s.replace(" @-@ ", "-").replace(" @.@ ", ".").replace(" @,@ ", ",").replace(" , ", ", ").replace(" ; ", "; ")
            .replace(" 's", "'s").replace("( ", "(").replace(" )", ")").strip())


def load_articles(path, n_sent=8):
    """species articles (WikiText-103): per article its first sentences, as the LM's token ids"""
    tk = lm()[0]
    arts = []
    raw = gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") else open(path, encoding="utf-8")
    for b in json.load(raw):
        ss = []
        for para in b["text"]:
            ss += [clean(s) + "." for s in re.split(r" \. ", para) if len(s.split()) > 3]
            if len(ss) >= n_sent:
                break
        ids = [tk(" " + s)["input_ids"][:60] for s in ss[:n_sent]]
        if len(ids) >= 3:
            arts.append(dict(title=b["title"], sents=ss[:n_sent], ids=ids))
    return arts


def quiz(art, rng, n):
    """blanked spans: a sentence of the article cut somewhere in its second half; the next up to 6 tokens are the answer
    (no annotation: any sentence of any text)"""
    out = []
    for _ in range(n):
        ids = rng.choice(art["ids"])
        if len(ids) < 6:
            continue
        cut = rng.randint(max(3, len(ids) // 2), len(ids) - 2)
        out.append((ids[:cut], ids[cut:cut + 6]))
    return out


def read(G, mem, art, train=False):
    cost, rt = 0., 0.
    for ids in art["ids"]:
        written, _, c = G.step(mem, ids, train)
        cost = cost + c
        if train:
            rt = rt + G.roundtrip_nll(mem, written, ids)
    return cost, rt / len(art["ids"])


def train_e1(G, arts, steps, lr=3e-4, nq=4, log=200, together=1):
    opt = torch.optim.AdamW([p for p in G.parameters() if p.requires_grad], lr=lr)
    G.train()
    rng = random.Random(0)
    t0, acc = time.time(), []
    for st in range(steps):
        group = [rng.choice(arts) for _ in range(together)]   # several articles in one memory: the blank's sentence must
        mem = Memory()                                      # be found among others (e1d: 1-article episodes, 20-article
        cost, rt = 0., 0.                                   # memories fell to the LM alone, .381 vs .375)
        for art in group:
            c, r = read(G, mem, art, train=True)
            cost, rt = cost + c, rt + r / together
        qs = [q for art in group for q in quiz(art, rng, max(1, nq // together))]
        l1 = sum(G.say_nll(mem, p, t)[0] for p, t in qs) / max(1, len(qs))
        loss = l1 + .3 * rt + cost
        opt.zero_grad()
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_([p for p in G.parameters() if p.requires_grad], 1.))   # (11-14 trace: diverged ~step 400)
        opt.step()
        rn = float(torch.stack([v.detach().norm(dim=-1).mean() for v in mem.V]).mean()) if mem.n else 0.
        if (st + 1) % log == 0:                                # which parameter the gradient is largest on
            gs = sorted(((float(p.grad.norm()), nm) for nm, p in G.named_parameters() if p.grad is not None), reverse=True)[:4]
            print("   largest grads (before clipping): " + ", ".join(f"{nm} {g:.1e}" for g, nm in gs), flush=True)
        acc.append((float(l1), float(rt), float(cost), sum(1 for v in mem.V if float(v.detach().norm()) > 1e-3), gn, rn))
        if (st + 1) % log == 0:
            a = torch.tensor(acc[-log:])
            print(f"step {st + 1}: blank nll {a[:, 0].mean():.3f}, round trip {a[:, 1].mean():.3f}, write cost {a[:, 2].mean():.3f}, "
                  f"nodes / article {a[:, 3].mean():.1f}, grad norm {a[:, 4].mean():.2f} (max {a[:, 4].max():.1f}), row norm {a[:, 5].mean():.2f} "
                  f"(token vectors ~3.5), mark {float(G.mark.norm()):.2f}, {time.time() - t0:.0f}s", flush=True)


@torch.no_grad()
def eval_e1(G, arts, seed=1, nq=6, together=1, show=0):
    """per quiz item, three readers of the same blank: the LM alone, the LM with the article(s) in context (when it
    fits), the LM with GNM's memory of the article(s). nll / token accuracy / exact span (greedy)"""
    tk = lm()[0]
    G.eval()
    rng = random.Random(seed)
    res = {k: [0., 0., 0, 0] for k in ("LM alone", "LM + text in context", "LM + GNM memory", "LM + another's memory")}
    prev = None
    nodes, hits = [], [0, 0]
    for b in range(0, len(arts) - together + 1, together):
        group = arts[b:b + together]
        mem, src = Memory(), []
        for ai, a in enumerate(group):
            for si, ids in enumerate(a["ids"]):
                before = mem.n
                G.step(mem, ids)
                src += [(ai, si)] * (mem.n - before)
        nodes.append(mem.n)
        ctx = [t for a in group for ids in a["ids"] for t in ids]
        for ai, a in enumerate(group):
            for p, t in quiz(a, rng, nq):
                si = next(i for i, ids in enumerate(a["ids"]) if ids[:len(p)] == p)
                sel = G.retrieve(mem, G.query(p))[0][:G.topk]
                hits[0] += any(i < len(src) and src[i] == (ai, si) for i in sel)
                hits[1] += 1
                readers = [("LM alone", Memory(), []), ("LM + text in context", Memory(), ctx), ("LM + GNM memory", mem, [])]
                if prev is not None:                       # control: the memory of other article(s) -- is it the content that helps?
                    readers.append(("LM + another's memory", prev, []))
                for name, m_, pre in readers:
                    if len(pre) + len(p) + len(t) > 1000:
                        continue
                    nll, ac = G.say_nll(m_, pre + p, t)
                    ex = G.say(m_, pre + p, len(t)) == t
                    r = res[name]
                    r[0] += float(nll)
                    r[1] += ac
                    r[2] += ex
                    r[3] += 1
        prev = mem
        if show and b < show * together:
            print(f"\n  -- {group[0]['title']}: {mem.n} nodes; what some say when the LM reads them alone:")
            for i in range(min(mem.n, 6)):
                print(f"     node {i}: {G.node_text(mem, i)!r}")
            p, t = quiz(group[0], random.Random(b), 1)[0]
            print(f"     blank: {tk.decode(p)!r} -> truth {tk.decode(t)!r} | memory says {tk.decode(G.say(mem, p, len(t)))!r} | "
                  f"alone says {tk.decode(G.say(Memory(), p, len(t)))!r}")
    print(f"\n{together} article(s) per memory, {len(nodes)} memories, {sum(nodes) / max(1, len(nodes)):.0f} nodes each; "
          f"a blank's top-{G.topk} holds a node of its own sentence {hits[0]}/{hits[1]}:")
    for name, (nll, ac, ex, n) in res.items():
        if n:
            print(f"  {name:>24}: nll {nll / n:.3f} | token acc {ac / n:.3f} | exact span {ex}/{n}", flush=True)


# ====================================================================== E1h: held-out sentence (§8 F9) -- copying cannot answer it
def heldout(art, rng, nq):
    """one sentence of the article is held out (never read); blanks are cut from it"""
    h = rng.randrange(len(art["ids"]))
    ids, qs = art["ids"][h], []
    for _ in range(nq):
        if len(ids) < 6:
            break
        cut = rng.randint(max(3, len(ids) // 2), len(ids) - 2)
        qs.append((ids[:cut], ids[cut:cut + 6]))
    return h, qs


def read_but(G, mem, art, h, train):
    """read every sentence but the held-out one: firings (WRITE=fire) or the span copier (WRITE=span, the control)"""
    cost, infos = 0., []
    for si, ids in enumerate(art["ids"]):
        if si == h:
            continue
        if G.wmode == "fire":
            inf, c = G.read_fire(mem, ids, train)
            infos += inf
        else:
            _, _, c = G.step(mem, ids, train)
        cost = cost + c
    return cost, infos


def train_e1h(G, arts, steps, lr=3e-4, log=100):
    opt = torch.optim.AdamW([p for p in G.parameters() if p.requires_grad], lr=lr)
    G.train()
    rng = random.Random(0)
    t0, acc = time.time(), []
    for st in range(steps):
        art = rng.choice(arts)
        h, qs = heldout(art, rng, 3)
        if not qs:
            continue
        mem = Memory()
        G._fk = None
        cost, infos = read_but(G, mem, art, h, True)
        l1 = sum(G.say_nll(mem, p, t)[0] for p, t in qs) / len(qs)
        loss = l1 + cost
        opt.zero_grad()
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_([p for p in G.parameters() if p.requires_grad], 1.))
        opt.step()
        G._fk = None
        nov = sum(r["novelty"] for r in infos) / max(1, len(infos))
        acc.append((float(l1), float(cost), mem.n, nov, len(infos), gn))
        if (st + 1) % log == 0:
            a = torch.tensor(acc[-log:])
            print(f"step {st + 1}: held-out blank nll {a[:, 0].mean():.3f}, cost {a[:, 1].mean():.3f}, nodes {a[:, 2].mean():.1f}, "
                  f"firings {a[:, 4].mean():.1f}, new tokens in outputs {a[:, 3].mean():.0%}, functions {int(G.n_fn)}, "
                  f"grad norm {a[:, 5].mean():.1f}, {time.time() - t0:.0f}s", flush=True)


@torch.no_grad()
def eval_e1h(G, arts, seed=1, nq=3):
    """the held-out sentence's blanks: LM alone / LM with the other sentences in context / LM with GNM's memory of them
    / LM with another article's memory (control); and how new the firings' outputs are"""
    G.eval()
    G._fk = None
    rng = random.Random(seed)
    res = {k: [0., 0., 0, 0] for k in ("LM alone", "LM + other sentences in context", "LM + GNM memory", "LM + another's memory")}
    prev, nodes, novs = None, [], []
    for a in arts:
        h, qs = heldout(a, rng, nq)
        mem = Memory()
        _, infos = read_but(G, mem, a, h, False)
        nodes.append(mem.n)
        novs += [r["novelty"] for r in infos]
        ctx = [t for i, ids in enumerate(a["ids"]) if i != h for t in ids]
        for p, t in qs:
            readers = [("LM alone", Memory(), []), ("LM + other sentences in context", Memory(), ctx), ("LM + GNM memory", mem, [])]
            if prev is not None:
                readers.append(("LM + another's memory", prev, []))
            for name, m_, pre in readers:
                if len(pre) + len(p) + len(t) > 1000:
                    continue
                nll, ac = G.say_nll(m_, pre + p, t)
                r = res[name]
                r[0] += float(nll)
                r[1] += ac
                r[2] += G.say(m_, pre + p, len(t)) == t
                r[3] += 1
        prev = mem
    print(f"\nheld-out sentence ({G.wmode}): {len(arts)} articles, {sum(nodes) / max(1, len(nodes)):.0f} nodes each, "
          f"new tokens in firing outputs {sum(novs) / max(1, len(novs)):.0%}, functions {int(G.n_fn)}")
    for name, (nll, ac, ex, n) in res.items():
        if n:
            print(f"  {name:>32}: nll {nll / n:.3f} | token acc {ac / n:.3f} | exact span {ex}/{n}", flush=True)


@torch.no_grad()
def trace_fire(G, art):
    """every firing: the function it used, what it fetched, what it PRODUCED, how much of that is new, the gate, links,
    what it woke; then the held-out sentence's blanks"""
    tk = lm()[0]
    G.eval()
    G._fk = None
    mem = Memory()
    h = len(art["ids"]) - 1
    for si, ids in enumerate(art["ids"]):
        if si == h:
            continue
        print(f"\nINPUT sentence {si}: {tk.decode(ids)!r}")
        infos, _ = G.read_fire(mem, ids, False)
        for r in infos:
            dst = f"n{r['written']}" if r["written"] is not None else "nothing written"
            print(f"   firing f{r['f']} (score {r['fsc']:+.2f}), fetched {['n%d' % i for i in r['sel']]}: produced {tk.decode(r['ids'])!r} "
                  f"(new tokens {r['novelty']:.0%}), gate {r['g']:.2f} -> {dst}; links {[(f'n{i}', round(w, 2)) for i, w in r['R']]}; woke {['n%d' % i for i in r['woken']]}")
    rng = random.Random(0)
    for p, t in [(art["ids"][h][:max(3, len(art["ids"][h]) // 2)], art["ids"][h][max(3, len(art["ids"][h]) // 2):][:6])]:
        ctx = [x for i, ids in enumerate(art["ids"]) if i != h for x in ids]
        print(f"\nHELD-OUT {tk.decode(p)!r} (truth {tk.decode(t)!r})")
        print(f"   memory says {tk.decode(G.say(mem, p, len(t)))!r}; alone says {tk.decode(G.say(Memory(), p, len(t)))!r}; "
              f"with the other sentences in context {tk.decode(G.say(Memory(), ctx + p, len(t)))!r}")
    del rng


# ====================================================================== test 1 (11-15): can the executor do ONE step?
# P1 for the small-core design: the LM reads an instruction node + exact input nodes and must produce the right output.
# Step kinds are generated and checkable; names are made up (no prior knowledge); facts come in random order with a
# distractor. Executors: the LM frozen; the LM + LoRA trained on 50 examples per kind. Nothing here is the design --
# it is the measurement that decides whether the design can work with this executor.
_SY = ("ka", "lo", "mer", "vin", "tor", "sa", "ne", "ril", "dun", "ba", "zek", "or", "pel", "mu", "tha", "gor")


def _nm(rng):
    return "".join(rng.choice(_SY) for _ in range(2)).capitalize()


def t1_example(kind, rng):
    """-> (instruction text, input node texts, the right output text)"""
    if kind == "add":
        a, b, c = rng.randint(0, 9), rng.randint(0, 9), rng.randint(0, 1)
        s = a + b + c
        return ("Add the two digits and the carry. Say the result digit and the new carry.",
                [f"First digit: {a}.", f"Second digit: {b}.", f"Carry: {c}."], f" digit {s % 10}, carry {s // 10}.")
    if kind == "compose":
        x, y, z, w, v = (_nm(rng) for _ in range(5))
        facts = [f"{x} is in the genus {y}.", f"The genus {y} is in the family {z}.", f"The genus {w} is in the family {v}."]
        rng.shuffle(facts)
        return (f"Which family is {x} in?", facts, f" {z}.")
    t, u = _nm(rng), _nm(rng)
    attrs = rng.sample(["size", "weight", "age", "depth", "length"], 3)
    vals = {a: rng.randint(2, 99) for a in attrs}
    facts = [f"The {a} of {t} is {vals[a]}." for a in attrs] + [f"The {attrs[0]} of {u} is {rng.randint(2, 99)}."]
    rng.shuffle(facts)
    q = rng.choice(attrs)
    return (f"What is the {q} of {t}?", facts, f" {vals[q]}.")


def t1_ids(instr, facts, out=None):
    """the step's input as nodes (grouped tokens): the instruction node, then each input node, then 'Answer:'"""
    tk = lm()[0]
    p = tk("Instruction: " + instr + " Facts: " + " ".join(facts) + " Answer:")["input_ids"]
    return p, (tk(out)["input_ids"] if out is not None else None)


class LoRA:
    """a small low-rank adapter on every attention and MLP input projection of the frozen LM (the executor adapts)"""

    def __init__(self, r=8, scale=2.):
        M = lm()[1]
        self.params, self.hooks = [], []
        for blk in M.transformer.h:
            for mod in (blk.attn.c_attn, blk.mlp.c_fc):
                din, dout = mod.weight.shape                 # (GPT-2 Conv1D: weight is (in, out))
                A = torch.nn.Parameter(torch.randn(din, r, device=dev) * .01)
                B = torch.nn.Parameter(torch.zeros(r, dout, device=dev))
                self.params += [A, B]
                self.hooks.append(mod.register_forward_hook(lambda m, i, o, A=A, B=B: o + (i[0].to(A.dtype) @ A @ B).to(o.dtype) * scale))

    def remove(self):
        for h in self.hooks:
            h.remove()


def t1_acc(kinds, n=200, seed=7):
    tk, M = lm()
    rng = random.Random(seed)
    out = {}
    with torch.no_grad():
        for kind in kinds:
            hit = 0
            for _ in range(n):
                instr, facts, ans = t1_example(kind, rng)
                p, a = t1_ids(instr, facts, ans)
                ids = torch.tensor([p], device=dev)
                gen = []
                for _ in range(len(a)):
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                        lg = M(input_ids=ids).logits[0, -1]
                    t = int(lg.argmax())
                    gen.append(t)
                    ids = torch.cat([ids, torch.tensor([[t]], device=dev)], 1)
                hit += gen == a
            out[kind] = hit / n
    return out


def test1(n_demo=50, steps=400):
    kinds = ("add", "compose", "lookup")
    print("TEST 1: one step, the executor reads an instruction node + input nodes and must produce the output", flush=True)
    rng = random.Random(1)
    ex = t1_example("compose", rng)
    print(f"  example: instruction {ex[0]!r}; input nodes {ex[1]}; right output {ex[2]!r}", flush=True)
    print(f"  LM frozen ({LM_NAME}):            " + ", ".join(f"{k} {v:.0%}" for k, v in t1_acc(kinds).items()), flush=True)
    tk, M = lm()
    lora = LoRA()
    opt = torch.optim.AdamW(lora.params, lr=1e-3)
    rng = random.Random(2)
    demos = [(k,) + t1_example(k, rng) for k in kinds for _ in range(n_demo)]   # 50 per kind (seed 2; tests use seed 7)
    for st in range(steps):
        batch = random.sample(demos, 16)
        L = 0
        for k, instr, facts, ans in batch:
            p, a = t1_ids(instr, facts, ans)
            ids = torch.tensor([p + a], device=dev)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                lg = M(input_ids=ids).logits[0].float()
            L = L + F.cross_entropy(lg[len(p) - 1:len(p) - 1 + len(a)], torch.tensor(a, device=dev))
        opt.zero_grad()
        (L / len(batch)).backward()
        opt.step()
        if (st + 1) % 100 == 0:
            print(f"  LoRA step {st + 1}: loss {float(L) / len(batch):.3f}; held-out " +
                  ", ".join(f"{k} {v:.0%}" for k, v in t1_acc(kinds, n=100).items()), flush=True)
    print(f"  LM + LoRA, {n_demo} demos per kind: " + ", ".join(f"{k} {v:.0%}" for k, v in t1_acc(kinds).items()), flush=True)
    lora.remove()


# ====================================================================== v10 (2026-11-15): nodes with meaning + control, skill nodes, recursion, kept memory
# User 11-15: don't start small; nodes that act as weights / helpers; "for each token we add another vector, it only
# affects its weighting and does not interfere with the semantic vector". Design + claim table: docs/GNM_v10.md.
#  - a memory token = (meaning: its token embedding, untouched; control: a small vector that only sets how much the LM
#    attends to it -- an additive attention bias in every layer -- and gives the node's key)
#  - one node kind (fact, question, sub-step, result, solved = "question + its result", skill). A skill node carries a
#    LoRA expert instead of text; it is fetched by key like any node (same query), and the fetched skill's expert runs
#    the step. Which skill is fetched is learned (the expert's output is scaled by its fetch weight).
#  - a step: focus + fetched nodes -> LM (+ the fetched skill's expert) writes the next node; 'done' says whether it
#    resolves the focus (then "focus + result" is written to memory as a solved node and focus returns to the parent)
#    or becomes the next focus (a sub-step). Root resolved -> the answer.
#  - kept memory (G.ltm): solved nodes from demos and from its own checked right runs stay, across questions.
#  - learning: ~50 demo traces imitated, then rounds of try -> check -> train on the right traces. A focus that fetches
#    no skill well enough (while training on demos or on its own right runs) spawns a new skill node.
_V10_AMP = [False]   # bf16 autocast off: 11-15 cachediag -- in bf16, cached vs slow writing differed 2-13% (once 88%),
#                      in fp32 0.000 in every case; v10 runs plain fp32 (TF32: 4%)


class Experts:
    """LoRA experts on the frozen LM (attention and MLP input projections); one active at a time -- the fetched skill
    node's. Its output is scaled by w, the skill's fetch weight (1 in value; carries the gradient that teaches which
    skill to fetch)"""

    def __init__(self, n=32, r=8, scale=2.):
        M = lm()[1]
        self.n, self.scale, self.active, self.w = n, scale, None, None
        self.A, self.B, self.hooks = [], [], []
        for blk in M.transformer.h:
            for mod in (blk.attn.c_attn, blk.mlp.c_fc):
                din, dout = mod.weight.shape
                A = torch.nn.Parameter(torch.randn(n, din, r, device=dev) * .01)
                B = torch.nn.Parameter(torch.zeros(n, r, dout, device=dev))
                self.A.append(A)
                self.B.append(B)
                self.hooks.append(mod.register_forward_hook(self._hook(A, B)))

    def _hook(self, A, B):
        def h(mod, inp, out):
            if self.active is None:
                return out
            e = self.active
            d = ((inp[0].float() @ A[e]) @ B[e]) * self.scale
            if self.w is not None:
                d = d * self.w
            return out + d.to(out.dtype)
        return h

    def params(self):
        return self.A + self.B


class Node10:
    """a node: token ids, control rows C, key, parent; H = cached LM states over its tokens (kept / skill nodes, so
    their control and key can be recomputed as the parts learn); expert (skill nodes); src (which trace wrote it);
    of (the focus a solved node records)"""

    def __init__(self, ids, C, key, parent=None, kind=""):
        self.ids, self.C, self.key, self.parent, self.kind = list(ids), C, key, parent, kind
        self.children, self.result = [], None
        self.H, self.expert, self.src, self.of = None, None, None, None


class GNM10(nn.Module):
    def __init__(self, dc=16, dk=256, topk=6, n_exp=32):
        super().__init__()
        tk, M = lm()
        d = M.config.n_embd
        self.d, self.dc, self.topk = d, dc, topk
        self.ctl = nn.Sequential(nn.Linear(d, 128), nn.GELU(), nn.Linear(128, dc))   # LM state -> a token's control
        self.bias = nn.Linear(dc, 1)                     # control -> attention bias (how much this token is attended)
        nn.init.zeros_(self.bias.weight)
        nn.init.zeros_(self.bias.bias)
        self.qry = nn.Linear(d, dk)                      # focus / question -> query
        # skill space (11-16 fix, D3): the frozen LM's mean state of the focus, centred by ONE fixed mean (set once by
        # fix_skill_mean), + each skill's learned offset EK. U0: in the old space (control-pooled, running-mean centred,
        # trained projection) every focus had cosine ~.999 with every other -> routing rode on parts that move every
        # step: U2 learned routing 13 skills, a kind kept its expert 20-48%, lookup 8% (oracle routing 100%).
        # Fixed-centred frozen states: same kind .79, other kind -.25, nearest neighbour same kind 100%
        # kept memory's "nothing relevant" option in its fetch softmax (scores are cosine / .1; 5 = cosine .5).
        # 11-15 v10j round 1: kept memory empty chain2 55%, with it 18% -- the best kept node always got full attention
        # (weight relative to the best), even Q/A pairs about other worlds
        self.kept_null = nn.Parameter(torch.tensor(5.))
        self.register_buffer("smu", torch.zeros(d))
        self.register_buffer("smu_n", torch.zeros(()))
        self.keyc = nn.Linear(dc, dk)                    # a node's control rows -> its key
        self.done = nn.Linear(d, 1)                      # does the new node resolve the focus?
        self.ex = Experts(n=n_exp)
        self.EK = nn.Parameter(torch.zeros(n_exp, d))    # each skill node's own learned key offset
        # spawning off (-2: never): 11-15 fitdiag, 300 demo steps -- 9-10 spawned skills: loss 2.3 -> 2.2 (no learning);
        # one skill: 2.0 -> 0.77. Returns when interference between skills is measured (my 11-15 plan)
        self.spawn_at = float(os.environ.get("SPAWN_AT", "-2"))
        self.skills, self.ltm, self._ltm_ids, self._kept_by_ids = [], [], set(), {}   # skill nodes; kept memory (solved nodes)

    def eparams(self):
        return self.ex.params()

    # ---------------------------------------------------------------- the LM with per-token attention bias
    def lm_run(self, ids, bias=None, past=None, cache=False):
        """the LM over token ids; bias: additive attention bias per key position (all past + new positions) -- the
        meaning (embedding) of every token is untouched. past: cached keys/values of earlier positions"""
        M = lm()[1]
        T = len(ids)
        P = 0 if past is None else past.get_seq_length()
        x = torch.tensor([ids], device=dev)
        mask = torch.full((T, P + T), float("-inf"), device=dev).triu(P + 1)
        if bias is not None:
            mask = mask + bias[None, :]
        pos = torch.arange(P, P + T, device=dev)[None]
        amp = dev == "cuda" and _V10_AMP[0]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            out = M(input_ids=x, attention_mask=mask[None, None].to(torch.bfloat16 if amp else torch.float32),
                    position_ids=pos, past_key_values=past, use_cache=cache, output_hidden_states=True)
        return out.hidden_states[-1][0].float(), out.logits[0].float(), (out.past_key_values if cache else None)

    def pool(self, H, C):
        """a node's states pooled by its tokens' control (attention-like weights) -- 11-15 v10d: a plain mean made keys
        about the sentence pattern ("Who is the boss of X?"), not the names; control sets what counts"""
        w = F.softmax(self.bias(C).squeeze(-1), 0)[:, None]
        return (w * H).sum(0), (w * C).sum(0)

    def key_pool(self, H, C):
        """pooling for keys and queries: the control's weights, detached (11-16 fpA: the fetch loss reached ctl / bias
        through this pooling -- the same parts that set the executor's attention; with it on, chain2 70 -> 22% in a
        round, off 68 -> 85%). Fetch learns in qry / keyc; control is learned by reading"""
        w = F.softmax(self.bias(C).squeeze(-1), 0).detach()[:, None]
        return (w * H).sum(0), (w * C.detach()).sum(0)

    def node_from(self, ids, H, parent=None, kind=""):
        C = self.ctl(H)                                  # control rows from the LM's states over the node's tokens
        Hp, Cp = self.key_pool(H, C)
        key = F.normalize(self.qry(Hp) + self.keyc(Cp), dim=-1)   # asked-for like a query + what its control adds
        return Node10(ids, C, key, parent, kind)

    def store(self, mem, text_ids, kind="fact"):
        H, _, _ = self.lm_run(text_ids)
        n = self.node_from(text_ids, H, kind=kind)
        mem.append(n)
        return n

    def query(self, ids):
        H, _, _ = self.lm_run(ids)
        return F.normalize(self.qry(self.key_pool(H, self.ctl(H))[0]), dim=-1)

    # ---------------------------------------------------------------- kept memory (across questions)
    @torch.no_grad()
    def keep(self, ids, src):
        """a solved node ("focus + its result") into kept memory; same text kept once"""
        t = tuple(ids)
        if t in self._ltm_ids:
            return
        H, _, _ = self.lm_run(list(ids))
        n = Node10(ids, None, None, kind="kept")
        n.H, n.src = H, src
        self._refresh(n)
        self.ltm.append(n)
        self._ltm_ids.add(t)
        self._kept_by_ids[t] = n

    @torch.no_grad()
    def _refresh(self, n):
        n.C = self.ctl(n.H)
        Hp, Cp = self.key_pool(n.H, n.C)
        n.key = F.normalize(self.qry(Hp) + self.keyc(Cp), dim=-1)

    def key_of(self, n):
        """a kept node's key from its cached states, with gradient (fetch training)"""
        Hp, Cp = self.key_pool(n.H, self.ctl(n.H))
        return F.normalize(self.qry(Hp) + self.keyc(Cp), dim=-1)

    def refresh_ltm(self):
        """kept nodes' control and keys from their cached states with the current parts (they learn meanwhile)"""
        for n in self.ltm:
            self._refresh(n)

    def fetch(self, mem, q, exclude=(), src_skip=None, train=False):
        """top-k by key from each source: working memory (this question's nodes: its facts, its writes) and kept memory
        (solved nodes from earlier questions). Returns the nodes with their fetch weight as an attention bias
        (log-probability within the source, 0 for its best).
        11-15 v10d: the weight multiplied a control bias that starts at 0 -> no gradient, fetch never learned.
        11-15 v10e: one pool of ~150 kept nodes vs ~7 facts (sampled while training) -> facts almost never in the
        context, it learned to answer without reading (lookup 80 -> 0% even with kept memory empty). Proposal (mine):
        two sources, each its own top-k; kept side sampled while training (Gumbel top-k) so it can learn."""
        ok = lambda n: n not in exclude and n.of not in exclude and not (src_skip is not None and n.src == src_skip)
        out, w = [], []
        for cands, kept in (([n for n in mem if ok(n)], False), ([n for n in self.ltm if ok(n)], True)):
            if not cands:
                continue
            sc = torch.stack([n.key for n in cands]) @ q
            g = sc.detach() / .1
            if kept and train:
                g = g - torch.log(-torch.log(torch.rand_like(g).clamp(1e-9, 1 - 1e-9)))
            sel = sorted(g.topk(min(self.topk, len(cands))).indices.tolist())   # (write order)
            if kept:                                     # absolute: against the learned "nothing relevant" option
                lp = F.log_softmax(torch.cat([sc / .1, self.kept_null[None]]), 0)
                w += [lp[i] for i in sel]
            else:                                        # this question's own nodes: relative to the best
                lp = F.log_softmax(sc, 0)
                best = lp[sel].max().detach()
                w += [lp[i] - best for i in sel]
            out += [cands[i] for i in sel]
        return out, w

    # ---------------------------------------------------------------- skill nodes (carry a LoRA expert)
    def skill_query(self, H, train=False):
        """a focus's (or skill label's) point in skill space: the frozen LM's mean state, centred by the fixed mean"""
        return H.detach().mean(0) - self.smu

    @torch.no_grad()
    def fix_skill_mean(self, texts_ids):
        """the skill space's centre, set once (from the focus texts at hand, e.g. the demos'); never moves after"""
        self.smu.copy_(torch.stack([self.lm_run(t)[0].mean(0) for t in texts_ids]).mean(0))
        self.smu_n.fill_(len(texts_ids))

    def skill_keys(self):
        """skill nodes' keys: their label's point in skill space plus their own learned offset"""
        K = torch.stack([self.skill_query(s.H) for s in self.skills]) + self.EK[[s.expert for s in self.skills]]
        return F.normalize(K, dim=-1)

    def pick_skill(self, q, focus, train):
        """fetch the skill node whose key best meets the focus's query; none good enough while training -> spawn a new
        skill node, labelled by this focus, its expert a copy of the closest one's"""
        n = len(self.skills)
        b = None
        if not n and not train:
            return None, None, 0.                        # no skill yet: the plain LM
        if n:
            sc = self.skill_keys() @ q
            b = int(sc.argmax())
            if not train or float(sc[b]) >= self.spawn_at or n >= self.ex.n:
                p = F.softmax(sc / .1, 0)
                return self.skills[b].expert, p[b] / p[b].detach(), float(sc[b])
        with torch.no_grad():
            e = n
            if b is not None:
                src = self.skills[b].expert
                for A, B in zip(self.ex.A, self.ex.B):
                    A.data[e] = A.data[src]
                    B.data[e] = B.data[src]
            H, _, _ = self.lm_run(focus.ids)
            s = Node10(focus.ids, None, None, kind="skill")
            s.H, s.expert = H, e
            self.EK.data[e] = 0.
            self.skills.append(s)
        return e, None, 1.

    # ---------------------------------------------------------------- one step: focus + fetched -> the next node
    def context(self, mem, focus, train, src_skip=None, ancestors=False):
        """the step's input: fetched nodes (meaning ids + their control as attention bias), the focus and its finished
        sub-steps; the fetched skill. Returns ids, bias, expert, its weight, how many kept nodes were fetched.
        v10t (Claude's trial, 11-16): the ancestors are NOT in the input -- each sub-question is its own problem.
        ctxdiag on v10s: writing the next smaller question below the root was right 17-25% with the ancestors in view,
        53-60% without (the 3-4 hop chains' kept runs jumped hops). ancestors=True = the v10s input (ctxdiag only).
        Ancestors stay out of the fetch either way (no way back in through working memory)"""
        Hf, _, _ = self.lm_run(focus.ids)
        q = F.normalize(self.qry(self.key_pool(Hf, self.ctl(Hf))[0]), dim=-1)
        sq = F.normalize(self.skill_query(Hf, train), dim=-1)
        chain, f = [], focus
        while f is not None:
            chain.append(f)
            f = f.parent
        chain = chain[::-1]
        excl = set(chain) | {c for x in chain for c in x.children}
        if not ancestors:
            chain = [focus]
        fetched, w = self.fetch(mem, q, exclude=excl, src_skip=src_skip, train=train)
        ids, bias = [], []
        for n, wi in zip(fetched, w):
            ids += n.ids
            bias.append(self.bias(n.C).squeeze(-1) + wi)
        for x in chain:                                  # the focus (+ ancestors in v10s), with finished sub-steps
            ids += x.ids
            bias.append(self.bias(x.C).squeeze(-1))
            for c in x.children:
                if c.result is not None:
                    ids += c.ids + c.result.ids
                    bias += [self.bias(c.C).squeeze(-1), self.bias(c.result.C).squeeze(-1)]
        e, ew, esc = self.pick_skill(sq, focus, train)
        bias = torch.cat(bias) if bias else None
        # the executor's window (GPT-2: 1024 positions) minus room for the node it writes: keep the newest tokens, so
        # fetched nodes drop first, then the oldest ancestors (11-16: with 64-token nodes a deep path went past 1024 ->
        # position index out of range, CUDA device assert in the v10p smoke)
        W = lm()[1].config.n_positions - 66
        if len(ids) > W:
            ids, bias = ids[-W:], bias[-W:]
        return ids, bias, e, ew, sum(n.kind == "kept" for n in fetched)

    def step_loss(self, mem, focus, target_ids, target_done, src_skip=None):
        """imitation of one step: the LM (+ fetched skill's expert) must write the target node and say whether it
        resolves the focus"""
        ids, bias, e, ew, _ = self.context(mem, focus, True, src_skip)
        tgt = target_ids + [lm()[0].eos_token_id]
        allids = ids + tgt
        b = torch.cat([bias, torch.zeros(len(tgt), device=dev)]) if bias is not None else None
        self.ex.active, self.ex.w = e, ew
        H, lg, _ = self.lm_run(allids, b)
        self.ex.active, self.ex.w = None, None
        P = len(ids)
        L = F.cross_entropy(lg[P - 1:P - 1 + len(tgt)], torch.tensor(tgt, device=dev))
        L = L + F.binary_cross_entropy_with_logits(self.done(H[-1]).squeeze(), torch.tensor(float(target_done), device=dev))
        return L, H[P:]

    @torch.no_grad()
    def gen(self, ids, bias, temp=0., maxlen=64):
        """write a node with cached keys/values (the context runs once, then one token at a time); returns the
        tokens, the LM states over them (the eos state if none), and the state after the closing eos (for 'done')"""
        eos = lm()[0].eos_token_id
        H, lg, past = self.lm_run(ids, bias, cache=True)
        allb = bias if bias is not None else torch.zeros(len(ids), device=dev)
        out, rows = [], []
        nxt = lg[-1]
        for i in range(maxlen + 1):
            if i == maxlen:
                t = eos                                  # cut: close it anyway
            elif temp:
                t = int(torch.multinomial(F.softmax(nxt / temp, -1), 1))
            else:
                t = int(nxt.argmax())
            allb = torch.cat([allb, torch.zeros(1, device=dev)])
            h, l, past = self.lm_run([t], allb, past=past, cache=True)
            if t == eos:
                return out, (torch.stack(rows) if rows else h), h[-1]
            out.append(t)
            rows.append(h[-1])
            nxt = l[-1]

    @torch.no_grad()
    def step(self, mem, focus, temp=0., maxlen=64, ancestors=False):   # (11-16: was 16 -- cut the smaller questions:
        #                                                   5-hop 20 tokens, 3+ digit sums 17-21 tokens)
        ids, bias, e, _, nk = self.context(mem, focus, False, ancestors=ancestors)
        self.ex.active = e
        out, rows, hd = self.gen(ids, bias, temp, maxlen)
        self.ex.active = None
        pd = float(torch.sigmoid(self.done(hd)))
        done = (random.random() < pd) if temp else pd > .5
        node = self.node_from(out or [lm()[0].eos_token_id], rows, parent=focus)
        return node, done, e, pd, nk


def _resolve(G, mem, focus, node):
    """focus is resolved by node: it becomes the focus's result; "focus + result" is written to memory (a solved
    node); returns the next focus (the parent; None at the root)"""
    focus.result = node
    s = G.store(mem, focus.ids + node.ids, kind="solved")
    s.of = focus
    if focus.parent is not None:
        focus.parent.children.append(focus)
    return focus.parent


def solve(G, mem, question_ids, temp=0., max_steps=48, trace=None):
    """recursion through nodes: the question is a node; each step writes a node that either resolves the focus (it
    becomes the focus's result, focus returns to the parent) or becomes the new focus (a sub-step).
    -> (answer text or None, steps [(focus, node, done)], solved pairs [(focus ids, result ids)])"""
    tk = lm()[0]
    root = G.store(mem, question_ids, kind="question")
    focus, steps, pairs = root, [], []
    for _ in range(max_steps):
        node, done, e, pd, nk = G.step(mem, focus, temp)
        steps.append((focus, node, done))
        if trace is not None:
            trace.append(f"   focus {tk.decode(focus.ids)!r} -> skill s{e} writes {tk.decode(node.ids)!r} "
                         f"({'resolves it' if done else 'sub-step'}, done p {pd:.2f}; kept nodes fetched {nk})")
        if done:
            pairs.append((focus.ids, node.ids))
            nxt = _resolve(G, mem, focus, node)
            if focus is root:
                return tk.decode(node.ids).strip(), steps, pairs
            focus = nxt
        else:
            focus = node
    return None, steps, pairs


def trace_pairs(q_ids, steps_ids):
    """the solved pairs (focus ids, result ids) of a trace given as [(node ids, done)], by the same focus rules"""
    focus, stack, pairs = q_ids, [], []
    for ids, done in steps_ids:
        if done:
            pairs.append((focus, ids))
            if not stack:
                break
            focus = stack.pop()
        else:
            stack.append(focus)
            focus = ids
    return pairs


# ---------------------------------------------------------------- checkable tasks + demo traces (the teacher's ~50)
def _chain_world(rng, size):
    names = [_nm(rng) for _ in range(size + 1)]
    rel = rng.choice(["boss", "friend", "teacher", "neighbour"])
    facts = [f"The {rel} of {names[i]} is {names[i + 1]}." for i in range(size)]
    for _ in range(3):
        facts.append(f"The {rel} of {_nm(rng)} is {_nm(rng)}.")
    rng.shuffle(facts)
    qtext = lambda k, i: f"Who is the {rel} of " + "".join(f"the {rel} of " for _ in range(k - 1)) + f"{names[i]}?"

    def steps(k, i):                                     # k hops left, starting from names[i]
        if k == 1:
            return [(names[i + 1], 1)]                   # one hop: answered by what is fetched
        return ([(qtext(1, i), 0), (names[i + 1], 1), (qtext(k - 1, i + 1), 0)] + steps(k - 1, i + 1)
                + [(names[i + k], 1)])
    return names, facts, qtext, steps


def v10_task(kind, rng, size):
    """-> (facts, question, answer, demo steps [(node text, done)] in the order solving writes them).
    Reduce-and-recurse (11-15 trace: demos that hung every sub-step under the question made it count hops and answer
    after one; here every level has the same shape -- one small step, then a smaller question, answered first)"""
    sp = lambda v: " ".join(str(v))
    if kind == "add":
        a, b = rng.randint(10 ** (size - 1), 10 ** size - 1), rng.randint(10 ** (size - 1), 10 ** size - 1)
        qtext = lambda da, db, c, suf: (f"What is {sp(da)} + {sp(db)} plus carry {c}" + (f", followed by {sp(suf)}" if suf else "") + "?")

        def steps(da, db, c, suf):                       # da, db: digit strings (top first); suf: digits found so far
            if not da:
                return [(sp((str(c) if c else "") + suf), 1)]
            x, y = int(da[-1]), int(db[-1])
            t = x + y + c
            d, c2 = t % 10, t // 10
            sub = [(f"{x} + {y} + carry {c} = ?", 0), (f"digit {d}, carry {c2}", 1)]
            nxt = (da[:-1], db[:-1], c2, str(d) + suf)
            rest = steps(*nxt)
            if not nxt[0]:                               # nothing left: the answer comes straight after the column
                return sub + rest
            return sub + [(qtext(*nxt), 0)] + rest + [(rest[-1][0], 1)]
        q = qtext(str(a), str(b), 0, "")
        return [], q, sp(a + b), steps(str(a), str(b), 0, "")
    if kind == "lookup":                                 # one fact among others about the same thing
        t, u = _nm(rng), _nm(rng)
        attrs = rng.sample(["size", "weight", "age", "depth", "length"], 3)
        vals = {a: rng.randint(2, 99) for a in attrs}
        facts = [f"The {a} of {t} is {vals[a]}." for a in attrs] + [f"The {attrs[0]} of {u} is {rng.randint(2, 99)}."]
        rng.shuffle(facts)
        a = rng.choice(attrs)
        return facts, f"What is the {a} of {t}?", str(vals[a]), [(str(vals[a]), 1)]
    if kind == "compose":                                # two different relations, one after the other
        x, y, z = (_nm(rng) for _ in range(3))
        facts = [f"{x} is in the genus {y}.", f"The genus {y} is in the family {z}."]
        for _ in range(3):                               # (11-16: was 1 other pair -- 2 families, a guess right half
            w, v, x2 = (_nm(rng) for _ in range(3))      #  the time, and lucky runs were kept as traces)
            facts += [f"The genus {w} is in the family {v}.", f"{x2} is in the genus {w}."]
        rng.shuffle(facts)
        return facts, f"Which family is {x} in?", z, [(f"Which genus is {x} in?", 0), (y, 1),
                                                       (f"Which family is the genus {y} in?", 0), (z, 1), (z, 1)]
    names, facts, qtext, steps = _chain_world(rng, size)
    return facts, qtext(size, 0), names[size], steps(size, 0)


def v10_demo_loss(G, facts, q, steps, src=None, recall=None):
    """imitate a trace with the same focus rules as solving (and the same solved-node writes); kept nodes written
    from this same trace are not fetched (they would hold its answers).
    recall (a list of other facts): this trace's facts were read earlier -- they sit in kept memory among those others,
    and none are given with the question (v10l)"""
    tk = lm()[0]
    saved = G.ltm
    Lr = 0.
    if recall is not None:
        texts = list(facts) + list(recall)
        random.shuffle(texts)
        own = {tuple(tk(" " + f)["input_ids"]) for f in facts}
        kt = []
        for f in texts:
            G.store(kt, tk(" " + f)["input_ids"], kind="kept")
        # fetch signal (my proposal, v10m): the facts read together with this question should score above every
        # other kept node, for the question and each sub-question it asks (11-15 v10l: recall 0% with 736 kept nodes;
        # fetch learned only through the attention weight, recall imitation then taught guessing -- loss .21 -> 1.49)
        cands = kt + random.sample(saved, min(256, len(saved)))
        K = torch.stack([n.key for n in cands])
        pos = torch.tensor([tuple(n.ids) in own for n in cands], device=dev)
        asks = [tk(" " + q)["input_ids"]] + [tk(" " + t)["input_ids"] for t, d in steps if not d]
        found = False
        for j, a in enumerate(asks):
            sc = (K @ G.query(a)) / .1
            Lr = Lr + (torch.logsumexp(sc, 0) - torch.logsumexp(sc[pos], 0)) / len(asks)
            if j == 0:
                found = bool(pos[sc.topk(min(G.topk, len(sc))).indices].any())
        # imitate only when fetch already brings one of its own facts (11-16 mixprobe: recall imitation without that
        # taught guessing -- chain2 65-70% -> 20-30% on the same demos); otherwise only the fetch signal trains
        if not found:
            G.ltm = saved
            return Lr
        G.ltm, facts = saved + kt, []
    try:
        return _demo_loss(G, facts, q, steps, src) + Lr
    finally:
        G.ltm = saved


def _demo_loss(G, facts, q, steps, src):
    tk = lm()[0]
    mem = []
    for f in facts:
        G.store(mem, tk(" " + f)["input_ids"], kind="fact")
    root = G.store(mem, tk(" " + q)["input_ids"], kind="question")
    focus, L, n = root, 0., 0
    for text, done in steps:
        ids = tk(" " + text)["input_ids"]
        l, H = G.step_loss(mem, focus, ids, done, src_skip=src)
        L, n = L + l, n + 1
        node = G.node_from(ids, H[:len(ids)], parent=focus)
        if done:
            nxt = _resolve(G, mem, focus, node)
            if focus is root:
                break
            focus = nxt
        else:
            focus = node
    return L / max(1, n)


@torch.no_grad()
def leaves_hold(G, facts, q, tr, log=None):
    """(11-16 v10r, Claude's proposal) every sub-question a run answered in ONE step must get the same answer when
    asked alone (same facts, no parent question). v10q round 1: all 7 kept 3-4 hop chains jumped to the second-last
    name ('Who is the boss of Vinba?' -> Merne, true Nezek) and still ended right; the jump repeats, so two tries
    agreed. Only one-step answers are re-asked: a multi-hop sub-question is built from its own one-step answers"""
    tk = lm()[0]
    focus, stack, fresh = tk(" " + q)["input_ids"], [], True
    for text, done in tr:
        ids = tk(" " + text)["input_ids"]
        if done:
            if fresh and stack:                          # a sub-question answered in one step
                mem = []
                for f in facts:
                    G.store(mem, tk(" " + f)["input_ids"], kind="fact")
                alone = [] if log is not None else None
                got, _, _ = solve(G, mem, focus, trace=alone)
                if log is not None:
                    log.append(f"    leaf {tk.decode(focus).strip()!r}: run wrote {text!r}, asked alone {got!r}"
                               + ("" if got == text else " -- DOES NOT HOLD\n" + "\n".join(alone)))
                if got != text:
                    return False
            if not stack:
                break
            focus, fresh = stack.pop(), False
        else:
            stack.append(focus)
            focus, fresh = ids, True
    return True


def keep_trace(G, q, steps, src):
    """a checked-right (or demo) trace's solved nodes go to kept memory"""
    tk = lm()[0]
    for f, r in trace_pairs(tk(" " + q)["input_ids"], [(tk(" " + t)["input_ids"], d) for t, d in steps]):
        G.keep(f + r, src)


def keep_facts(G, facts, src):
    """every fact read is kept (taken in once, kept) -- v10l"""
    tk = lm()[0]
    for f in facts:
        G.keep(tk(" " + f)["input_ids"], src)


V10_KINDS = ("add", "chain", "lookup", "compose")


@torch.no_grad()
def v10_eval(G, kinds_sizes, n=40, seed=11, show=0, use_ltm=True, skip_q=frozenset()):
    """accuracy on new problems (questions seen in training skipped); kept memory read-only (writes inside a question
    stay in that question); also: share of steps that fetched a kept node, skills used per kind"""
    tk = lm()[0]
    G.eval()
    G.refresh_ltm()
    saved = G.ltm
    if not use_ltm:
        G.ltm = []
    rng = random.Random(seed)
    out, used, need_hit = {}, {}, {}
    nk_steps = n_steps = 0
    for kind, size in kinds_sizes:
        hit = j = 0
        while j < n:
            facts, q, ans, _ = v10_task(kind, rng, size)
            if q in skip_q:
                continue
            mem = []
            for f in facts:
                G.store(mem, tk(" " + f)["input_ids"], kind="fact")
            tr = []
            got, steps, _ = solve(G, mem, tk(" " + q)["input_ids"], trace=tr)
            hit += got == ans
            # diagnosis only (never trained on): at each step, is the fact the step needs (a given fact holding every
            # capitalised name of the step's text) among what was fetched?
            own = [tuple(tk(" " + f)["input_ids"]) for f in facts]
            for focus, node, done in steps:
                names = [w.strip("?.,") for w in tk.decode(focus.ids).split() if w[:1].isupper() and w not in ("Who", "What", "Which", "The")]
                need = {o for o in own if names and all(nm in tk.decode(list(o)) for nm in names)}
                if need:
                    fetched, _ = G.fetch(mem, G.query(focus.ids), exclude={focus})
                    need_hit.setdefault(kind, []).append(any(tuple(x.ids) in need for x in fetched))
            n_steps += len(tr)
            nk_steps += sum(not t.endswith("fetched 0)") for t in tr)
            for t in tr:
                s = t.split(" -> skill ")[1].split(" ")[0]
                used.setdefault(kind, {}).setdefault(s, 0)
                used[kind][s] += 1
            if j < show:
                print(f"\n  [{kind} {size}] {q} (truth {ans}) -> {got}\n" + "\n".join(tr[:14]), flush=True)
            j += 1
        out[f"{kind}{size}"] = hit / n
    G.ltm = saved
    G.train()
    used["needed fact fetched"] = {k: f"{sum(v) / max(1, len(v)):.0%}" for k, v in need_hit.items()}
    return out, nk_steps / max(1, n_steps), used


@torch.no_grad()
def v10_recall(G, kinds_sizes, n=30, seed=31, show=True):
    """read once, asked later (v10l): every world's facts are read into kept memory first, among all that is kept
    already; then each question comes with NO facts given -- it must fetch the right kept facts and reason over them"""
    tk = lm()[0]
    G.eval()
    rng = random.Random(seed)
    base, base_ids = list(G.ltm), set(G._ltm_ids)
    qs = []
    for kind, size in kinds_sizes:
        for _ in range(n):
            facts, q, ans, _ = v10_task(kind, rng, size)
            keep_facts(G, facts, "recall")
            qs.append((kind, size, q, ans, {tuple(tk(" " + f)["input_ids"]) for f in facts}))
    G.refresh_ltm()
    out, nk, ns, shown, hit, need_hit = {}, 0, 0, set(), {}, {}
    for kind, size, q, ans, own in qs:
        tr = []
        got, steps, _ = solve(G, [], tk(" " + q)["input_ids"], trace=tr)
        out.setdefault(f"{kind}{size}", []).append(got == ans)
        ns += len(tr)
        nk += sum(not t.endswith("fetched 0)") for t in tr)
        # fetch hit: at each step, is one of this question's own facts among the fetched kept nodes? and the fact this
        # step needs (diagnosis only, never trained on: an own fact holding every capitalised name of the step's text)
        lines = []
        for focus, node, done in steps:
            fetched, wts = G.fetch([], G.query(focus.ids))
            hit.setdefault(f"{kind}{size}", []).append(any(tuple(x.ids) in own for x in fetched))
            names = [w.strip("?.,") for w in tk.decode(focus.ids).split() if w[:1].isupper() and w not in ("Who", "What", "Which", "The")]
            need = [o for o in own if names and all(nm in tk.decode(list(o)) for nm in names)]
            if need:
                need_hit.setdefault(f"{kind}{size}", []).append(any(tuple(x.ids) in need for x in fetched))
            if show and (kind, size) not in shown:
                top = sorted(zip(wts, fetched), key=lambda z: -float(z[0]))[:3]
                lines.append(f"     fetched for {tk.decode(focus.ids)!r}: " + "; ".join(
                    f"{tk.decode(x.ids).strip()!r} ({float(w_):.1f}{', own' if tuple(x.ids) in own else ''})" for w_, x in top))
        if show and (kind, size) not in shown:
            shown.add((kind, size))
            print(f"\n  [recall {kind} {size}] {q} (truth {ans}) -> {got}\n" + "\n".join(tr[:10]) + "\n" + "\n".join(lines[:10]), flush=True)
    n_kept = len(G.ltm)
    G.ltm, G._ltm_ids = base, base_ids
    G.train()
    return (f"recall (facts read earlier, none given; kept memory {n_kept} nodes): "
            + ", ".join(f"{k} {sum(v) / len(v):.0%}" for k, v in out.items()) + f" | steps fetching a kept node {nk / max(1, ns):.0%}"
            + " | steps whose top-6 kept holds one of its own facts: " + ", ".join(f"{k} {sum(v) / max(1, len(v)):.0%}" for k, v in hit.items())
            + " | ... the fact the step needs: " + ", ".join(f"{k} {sum(v) / max(1, len(v)):.0%}" for k, v in need_hit.items()))


@torch.no_grad()
def v10_reuse(G, n=20, seed=23):
    """take in once, keep it: in a new chain world, ask a question, keep what it wrote (unchecked), ask it again and
    ask a longer one. Reports: second-time accuracy and steps, share of steps fetching a node written by the first
    question, the longer one with vs without the first question's nodes"""
    tk = lm()[0]
    G.eval()
    G.refresh_ltm()
    rng = random.Random(seed)
    res = {"first": 0, "again": 0, "steps1": 0, "steps2": 0, "fetch_own": 0, "steps_all": 0, "longer_with": 0, "longer_without": 0}
    base, base_ids = list(G.ltm), set(G._ltm_ids)
    for _ in range(n):
        names, facts, qtext, _ = _chain_world(rng, 6)
        ans3 = names[3]
        for with_first in (True, False):
            G.ltm, G._ltm_ids = list(base), set(base_ids)
            mem = []
            for f in facts:
                G.store(mem, tk(" " + f)["input_ids"], kind="fact")
            if with_first:
                got, st, pairs = solve(G, list(mem), tk(" " + qtext(3, 0))["input_ids"])
                res["first"] += got == ans3
                res["steps1"] += len(st)
                n0 = len(G.ltm)
                for f, r in pairs:
                    G.keep(f + r, "reuse")
                own = set(map(id, G.ltm[n0:]))
                tr = []
                got2, st2, _ = solve(G, list(mem), tk(" " + qtext(3, 0))["input_ids"], trace=tr)
                res["again"] += got2 == ans3
                res["steps2"] += len(st2)
                # which steps fetched one of the first question's kept nodes
                for focus, node, done in st2:
                    q = G.query(focus.ids)
                    fetched, _ = G.fetch([], q)
                    res["fetch_own"] += any(id(x) in own for x in fetched)
                    res["steps_all"] += 1
            got5, _, _ = solve(G, list(mem), tk(" " + qtext(5, 0))["input_ids"])
            res["longer_with" if with_first else "longer_without"] += got5 == names[5]
    G.ltm, G._ltm_ids = base, base_ids
    G.train()
    return (f"reuse (n {n}): first ask {res['first'] / n:.0%} in {res['steps1'] / n:.1f} steps; asked again {res['again'] / n:.0%} "
            f"in {res['steps2'] / n:.1f} steps, {res['fetch_own'] / max(1, res['steps_all']):.0%} of its steps fetch a node the "
            f"first ask wrote; longer question (5 hops) with the first ask kept {res['longer_with'] / n:.0%}, without {res['longer_without'] / n:.0%}")


def v10_run(rounds=6, n_demo=50, tries=4, per_round=400):   # (11-16: 400 tries a round -- more own traces)
    tk = lm()[0]
    smoke = os.environ.get("V10_SMOKE") == "1"     # the whole path, tiny: demos, a round, the evals
    if smoke:
        rounds, per_round = 1, 3
    # fixed seed (11-16: v10q and v10r ran the same code up to round 1, yet round 0 chain2 was 52% vs 22% -- init and
    # sampling were unseeded, so run-to-run noise hid any change smaller than ~30 points)
    seed = int(os.environ.get("V10_SEED", "0"))
    torch.manual_seed(seed)
    random.seed(seed)
    print(f"v10 seed {seed}", flush=True)
    G = GNM10().to(dev)
    params = [p for nm, p in G.named_parameters() if p.requires_grad] + G.eparams()
    opt = torch.optim.AdamW(params, lr=1e-3)
    rng = random.Random(5)
    # demos up to 3 (11-15 v10b trace: with demos up to 2 the smaller question written after one step was always a
    # 1-step question, so "drop one step and ask again" for a longer one was never shown); tests stay longer (4, 6)
    train_sizes = [(k, s) for k in ("add", "chain") for s in (1, 2, 3)] + [("lookup", 1), ("compose", 1)]
    test_sizes = [("add", 2), ("add", 4), ("add", 6), ("chain", 2), ("chain", 4), ("chain", 6), ("lookup", 1), ("compose", 1)]
    demos = [v10_task(k, rng, s) for k, s in train_sizes for _ in range(n_demo // len(train_sizes))]
    # skill space centre, set once from the demos' focus texts (questions + the sub-questions they write)
    G.fix_skill_mean([tk(" " + t)["input_ids"] for f_, q_, a_, st_ in demos for t in [q_] + [x for x, d in st_ if not d]])
    seen_q = {d[1] for d in demos}
    # kept memory: the facts it read (all of them, v10l) + solved nodes of its own checked right runs -- not the demos'
    # solved nodes (11-15 v10h: with those in every context each demo was unique, it memorised them)
    # probe switches (11-16 mixprobe; defaults = the design as built): V10_KEEP keep the demos' facts, V10_RECALL recall
    # training, V10_FETCHL fetch-training loss, V10_FIT0 demo steps before round 0
    use_keep, use_recall, use_fetchl = (os.environ.get(k, "1") == "1" for k in ("V10_KEEP", "V10_RECALL", "V10_FETCHL"))
    fit0 = int(os.environ.get("V10_FIT0", "1500"))   # (11-16 dataprobe / mixprobe: 1500 demo steps)
    print(f"v10 switches: keep demo facts {use_keep}, recall training {use_recall}, fetch loss {use_fetchl}, demo steps {fit0}", flush=True)
    for i, d in enumerate(demos):
        if use_keep:
            keep_facts(G, d[0], i)
    recall_sizes = [("chain", 2), ("chain", 4), ("lookup", 1), ("compose", 1)]
    print(f"v10: {len(demos)} demo traces (sizes {train_sizes}); kept memory starts with the demos' facts ({len(G.ltm)}); "
          f"test sizes {test_sizes} (add / chain longer than any demo)", flush=True)
    ne = 2 if smoke else 40

    def report(tag, res, fk, used, extra=""):
        print(f"{tag} | " + ", ".join(f"{k} {v:.0%}" for k, v in res.items())
              + f" | steps fetching a kept node {fk:.0%}; skills {len(G.skills)}; skills used per kind "
              + "; ".join(f"{k}: " + " ".join(f"{s}x{c}" if isinstance(c, int) else f"{s} {c}" for s, c in sorted(u.items())) for k, u in used.items()) + extra, flush=True)

    def fit(traces, steps):
        G.train()
        L, run = torch.zeros(()), []
        by_len = {}
        for j, t in enumerate(traces):
            by_len.setdefault(len(t[3]), []).append(j)
        lens = list(by_len)
        # a length is picked by sqrt(how many traces it holds) (11-16 v10t: picked uniformly, a length with one odd
        # own run -- a skipped step -- got as much training as all 27 right 5-step runs; chain2 70 -> 22% in round 1)
        lw = [len(by_len[k]) ** .5 for k in lens]
        print("    traces by length: " + ", ".join(f"{k}:{len(by_len[k])}" for k in sorted(lens)), flush=True)
        for st in range(steps):
            if st % 50 == 0:
                G.refresh_ltm()
            if st and st % 100 == 0:
                print(f"    fit step {st}: mean loss {sum(run) / len(run):.3f}, skills {len(G.skills)}", flush=True)
                run = []
            # balanced by trace length (11-15 v10i: 55 of 77 kept runs were one-step answers; trained on them as they
            # came, chains went 40 -> 12 -> 22%): pick a length first, then a trace of that length
            i = random.choice(by_len[random.choices(lens, lw)[0]])
            facts, q, ans, tr = traces[i]
            saved = G.ltm
            if random.random() < .5:                     # half the traces without kept memory: must solve both ways,
                G.ltm = []                               # cannot memorise a trace by its unique kept context
            rec = None
            if use_recall and facts and random.random() < .5:   # recall: its facts were read earlier, among other worlds' facts
                rec = [f for j in random.sample(range(len(traces)), min(8, len(traces))) if j != i for f in traces[j][0]]
            L = v10_demo_loss(G, facts, q, tr, src=i, recall=rec)
            G.ltm = saved
            L = L + fetch_loss()
            opt.zero_grad()
            L.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.)
            opt.step()
            run.append(float(L))
        return float(L)

    # fetch training from its own experience (my proposal, v10n): every focus it handled (the question and each
    # sub-question it wrote, right or wrong) with the facts read in that episode; the focus's query should rank those
    # kept facts above other kept nodes. 11-15 v10m: from demos alone, own fact in the kept top-6: lookup 67% -> 6%
    # once kept memory grew to 1517 nodes
    fetch_pairs = []

    def add_pairs(focus_ids_list, facts):
        own = frozenset(tuple(tk(" " + f)["input_ids"]) for f in facts)
        if own:
            for fi in focus_ids_list:
                fetch_pairs.append((list(fi), own))
        del fetch_pairs[:-20000]

    for d in demos:
        add_pairs([tk(" " + d[1])["input_ids"]] + [tk(" " + t)["input_ids"] for t, dn in d[3] if not dn], d[0])

    def fetch_loss(k=8, n_neg=256):
        if not use_fetchl or not fetch_pairs or len(G.ltm) < 10:
            return 0.
        neg = random.sample(G.ltm, min(n_neg, len(G.ltm)))
        Kn = torch.stack([G.key_of(n) for n in neg])
        L, m = 0., 0
        for focus_ids, own in random.sample(fetch_pairs, min(k, len(fetch_pairs))):
            pos = [G._kept_by_ids[t] for t in own if t in G._kept_by_ids]
            if not pos:
                continue
            Kp = torch.stack([G.key_of(n) for n in pos])
            keep_neg = torch.tensor([tuple(n.ids) not in own for n in neg], device=dev)
            qv = G.query(focus_ids)
            sp, sn = (Kp @ qv) / .1, (Kn @ qv) / .1
            L = L + torch.logsumexp(torch.cat([sp, sn[keep_neg]]), 0) - torch.logsumexp(sp, 0)
            m += 1
        return L / max(1, m)

    t0 = time.time()
    l = fit(demos, 5 if smoke else fit0)
    print(f"round 0 (demos only): loss {l:.3f}, {time.time() - t0:.0f}s", flush=True)
    report("  with kept memory", *v10_eval(G, test_sizes, n=ne, show=1, skip_q=seen_q))
    report("  kept memory empty", *v10_eval(G, test_sizes, n=ne, use_ltm=False, skip_q=seen_q))
    print("  " + v10_recall(G, recall_sizes, n=2 if smoke else 30), flush=True)
    print("  " + v10_reuse(G, n=2 if smoke else 20), flush=True)
    kept = list(demos)
    for r in range(1, rounds + 1):                       # try -> check -> train on the right traces (no new demos)
        t0 = time.time()
        G.eval()
        G.refresh_ltm()
        got_right, seen = 0, {}
        for _ in range(per_round):
            kind = rng.choice(V10_KINDS)
            size = rng.randint(1, 3 + r) if kind in ("add", "chain") else 1
            facts, q, ans, _ = v10_task(kind, rng, size)
            right = []
            for t in range(tries):
                mem = []
                for f in facts:
                    G.store(mem, tk(" " + f)["input_ids"], kind="fact")
                with torch.no_grad():
                    got, steps, _ = solve(G, mem, tk(" " + q)["input_ids"], temp=.7)
                add_pairs(list({tuple(f.ids): f.ids for f, _, _ in steps}.values()), facts)
                if got == ans:                           # its own right run, as a trace to imitate (same focus rules)
                    tr = [(tk.decode(node.ids).strip(), int(done)) for focus, node, done in steps]
                    if all(x for x, _ in tr):
                        right.append(tr)
                # keep only when >= 3 (was 2) right tries wrote the SAME steps (v10b round 1: one lucky guess in 4 tries is
                # common -- chain2 60 -> 22%; 11-16 v10p round 1: with ">= 2 right answers", 13 of 36 kept compose
                # runs reached the right family through wrong steps (wrong genus, renamed names) and compose fell
                # 52 -> 10%. A guess lands on the answer by different wrong steps each time; a method repeats)
                # (11-16 v10v: >= 3 of 4 -- a one-step answer's "same steps" is just the same answer; v10u round 2 kept
                # compose runs that answered in one step, right ~28% by luck -> passes 2 of 4 31% of the time, 3 of 4
                # 7%; a method right 80% still passes 82%)
                agree = [x for x in right if right.count(x) >= 3]
                best = max((right.count(x) for x in right), default=0)
                if agree or best + (tries - t - 1) < 3:
                    break
            # (v10r tried leaves_hold here -- dropped 11-16: holddiag showed it rejects correct runs, a one-hop
            # sub-question asked alone gets solved as a two-hop question; it stays as a measurement in holddiag)
            if agree:
                kept.append((facts, q, ans, agree[0]))
                keep_trace(G, q, agree[0], len(kept) - 1)
                seen_q.add(q)
                got_right += 1
                seen.setdefault((kind, size), []).append(len(agree[0]))
            keep_facts(G, facts, len(kept) - 1 if kept[-1][1] == q else "read")   # what it read stays
        print(f"round {r}: right runs found {got_right}/{per_round} (kept traces {len(kept)}, kept memory {len(G.ltm)} nodes), "
              f"{time.time() - t0:.0f}s trying", flush=True)
        print(f"  fetch-training pairs {len(fetch_pairs)}", flush=True)
        print("  kept by kind/size (count, mean steps): " + ", ".join(
            f"{k}{s} {len(v)} {sum(v) / len(v):.1f}" for (k, s), v in sorted(seen.items())), flush=True)
        # steps scale with the traces (11-15 v10i: 400 steps on ~125-190 traces -> loss .000, chains fell)
        l = fit(kept, 5 if smoke else min(400, 2 * len(kept)))
        last = r == rounds
        report(f"  loss {l:.3f}; with kept memory", *v10_eval(G, test_sizes, n=ne, show=1, skip_q=seen_q))
        report("  kept memory empty", *v10_eval(G, test_sizes, n=ne, use_ltm=False, skip_q=seen_q))
        print("  " + v10_recall(G, recall_sizes, n=2 if smoke else 30, show=last), flush=True)
        v10_save(G, os.environ.get("V10_SAVE", ""))
        if last:
            print("  " + v10_reuse(G, n=2 if smoke else 20), flush=True)


def v10_save(G, path):
    """the model + its skills + kept memory (ids, cached states, source) -- for diagnosis without retraining"""
    if not path:
        return
    torch.save({"state": G.state_dict(), "ex": [t.detach().cpu() for t in G.ex.params()],
                "skills": [(s.ids, s.H.cpu(), s.expert) for s in G.skills],
                "ltm": [(n.ids, n.H.cpu(), n.src) for n in G.ltm]}, path)
    print(f"  saved {path} (kept memory {len(G.ltm)} nodes)", flush=True)


def v10_load(path):
    """a model saved by v10_save, with its skills and kept memory"""
    ck = torch.load(path, map_location=dev, weights_only=False)
    G = GNM10().to(dev)
    G.load_state_dict(ck["state"])
    for t, v in zip(G.ex.params(), ck["ex"]):
        t.data.copy_(v.to(dev))
    for ids, H, e in ck["skills"]:
        n = Node10(ids, None, None, kind="skill")
        n.H, n.expert = H.to(dev), e
        G.skills.append(n)
    with torch.no_grad():
        for ids, H, src in ck["ltm"]:
            n = Node10(ids, None, None, kind="kept")
            n.H, n.src = H.to(dev), src
            G._refresh(n)
            G.ltm.append(n)
            G._ltm_ids.add(tuple(ids))
            G._kept_by_ids[tuple(ids)] = n
    G.eval()
    return G


@torch.no_grad()
def v10_readdiag(path, n=30, seed=41):
    """11-15 v10o: the needed fact is fetched 90-100% of steps, yet chains / compose are right 5-20%. Is the reading
    step drowned by what else is in view, or can the executor not do the step? On the CORRECT solution path (the
    task's own steps, eval only): at each step the executor writes (a) with its normal context (fetched nodes + path),
    (b) with only the fact the step needs + the path, (c) with all of the question's facts + the path, no kept memory.
    Reports exact-match of the written node vs the correct one, by kind and by step type"""
    tk = lm()[0]
    G = v10_load(path)
    rng = random.Random(seed)
    res = {}
    for kind, size in (("chain", 2), ("chain", 3), ("compose", 1), ("lookup", 1)):
        for _ in range(n):
            facts, q, ans, steps = v10_task(kind, rng, size)
            fids = [tk(" " + f)["input_ids"] for f in facts]
            for mode in ("normal", "needed fact only", "all facts, no kept"):
                mem = [] if mode == "needed fact only" else None
                saved = G.ltm
                if mode != "normal":
                    G.ltm = []
                if mem is None:
                    mem = []
                    for f in fids:
                        G.store(mem, f, kind="fact")
                root = G.store(mem, tk(" " + q)["input_ids"], kind="question")
                focus = root
                for text, done in steps:
                    want = tk.decode(tk(" " + text)["input_ids"]).strip()
                    if mode == "needed fact only":     # this step's working memory = only the fact it needs
                        names = [w.strip("?.,") for w in tk.decode(focus.ids).split() if w[:1].isupper() and w not in ("Who", "What", "Which", "The")]
                        need = [f for f in fids if names and all(nm in tk.decode(f) for nm in names)]
                        mem[:] = [root]
                        for f in need:
                            G.store(mem, f, kind="fact")
                    node, d, e, pd, nk = G.step(mem, focus)
                    got = tk.decode(node.ids).strip()
                    key = (mode, kind + str(size), "sub-question" if not done else "answer")
                    res.setdefault(key, []).append(got == want)
                    # continue on the correct path, whatever it wrote
                    ids = tk(" " + text)["input_ids"]
                    H, _, _ = G.lm_run(ids)
                    nd = G.node_from(ids, H, parent=focus)
                    if done:
                        nxt = _resolve(G, mem, focus, nd)
                        if focus is root:
                            break
                        focus = nxt
                    else:
                        focus = nd
                G.ltm = saved
    for mode in ("normal", "needed fact only", "all facts, no kept"):
        print(f"readdiag {mode}: " + ", ".join(f"{k[1]} {k[2]} {sum(v) / len(v):.0%}" for k, v in sorted(res.items()) if k[0] == mode), flush=True)


@torch.no_grad()
def _path_acc(G, kinds_sizes, n=30, seed=41):
    """per step written-node exact match on the CORRECT solution path (facts given, no kept memory)"""
    tk = lm()[0]
    G.eval()
    saved, G.ltm = G.ltm, []
    rng = random.Random(seed)
    res = {}
    for kind, size in kinds_sizes:
        for _ in range(n):
            facts, q, ans, steps = v10_task(kind, rng, size)
            mem = []
            for f in facts:
                G.store(mem, tk(" " + f)["input_ids"], kind="fact")
            root = G.store(mem, tk(" " + q)["input_ids"], kind="question")
            focus = root
            for text, done in steps:
                ids = tk(" " + text)["input_ids"]
                node, d, e, pd, nk = G.step(mem, focus)
                res.setdefault(f"{kind}{size} {'answer' if done else 'sub-q'}", []).append(tk.decode(node.ids).strip() == tk.decode(ids).strip())
                H, _, _ = G.lm_run(ids)
                nd = G.node_from(ids, H, parent=focus)
                if done:
                    nxt = _resolve(G, mem, focus, nd)
                    if focus is root:
                        break
                    focus = nxt
                else:
                    focus = nd
    G.ltm = saved
    G.train()
    return {k: sum(v) / len(v) for k, v in sorted(res.items())}


def v10_dataprobe(sizes=(48, 192, 768), steps=1500):
    """11-16 probe (measurement only, not the design): is the executor (GPT-2 small + LoRA) short of data or of
    capacity? Fresh model per size, trained on that many teacher traces (facts given, no kept memory), same steps;
    per-step accuracy on the correct path + end-to-end accuracy"""
    train_sizes = [(k, s) for k in ("add", "chain") for s in (1, 2, 3)] + [("lookup", 1), ("compose", 1)]
    probe = [("chain", 2), ("chain", 3), ("compose", 1), ("lookup", 1), ("add", 2)]
    for nd in sizes:
        torch.manual_seed(0)
        random.seed(0)
        rng = random.Random(5)
        G = GNM10().to(dev)
        demos = [v10_task(k, rng, s_) for k, s_ in train_sizes for _ in range(nd // len(train_sizes))]
        params = [p for nm, p in G.named_parameters() if p.requires_grad] + G.eparams()
        opt = torch.optim.AdamW(params, lr=1e-3)
        for st in range(steps):
            facts, q, ans, tr = random.choice(demos)
            L = v10_demo_loss(G, facts, q, tr)
            opt.zero_grad()
            L.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.)
            opt.step()
        pa = _path_acc(G, probe)
        ee, _, _ = v10_eval(G, [("chain", 2), ("chain", 4), ("compose", 1), ("lookup", 1), ("add", 2)], n=40, use_ltm=False)
        print(f"dataprobe {len(demos)} traces, {steps} steps | per step: " + ", ".join(f"{k} {v:.0%}" for k, v in pa.items())
              + " | end to end: " + ", ".join(f"{k} {v:.0%}" for k, v in ee.items()), flush=True)
        for h in G.ex.hooks:
            h.remove()
        del G, opt


@torch.no_grad()
def v10_holddiag(path, n=40, tries=4, seed=43):
    """11-16 v10r round 1 dropped 45 agreeing right runs and kept no chain2 / compose run. Shortcuts, or does a correct
    one-step answer fail when its sub-question is asked alone (e.g. 'Which genus is X in?' is never a demo question)?
    Tries like a round (temp .7, 4 tries, same-steps agreement); for each agreeing right run: the true facts, the run,
    each leaf's answer asked alone (with that solve's trace when it does not hold)"""
    tk = lm()[0]
    G = v10_load(path)
    rng = random.Random(seed)
    stats = {}
    for kind, size in (("chain", 2), ("chain", 3), ("compose", 1)):
        for _ in range(n):
            facts, q, ans, demo = v10_task(kind, rng, size)
            right = []
            for t in range(tries):
                mem = []
                for f in facts:
                    G.store(mem, tk(" " + f)["input_ids"], kind="fact")
                got, steps, _ = solve(G, mem, tk(" " + q)["input_ids"], temp=.7)
                if got == ans:
                    right.append([(tk.decode(node.ids).strip(), int(done)) for focus, node, done in steps])
            agree = [x for x in right if right.count(x) >= 2]
            key = f"{kind}{size}"
            st = stats.setdefault(key, {"agreeing right": 0, "holds": 0, "same as demo": 0, "holds and not demo": 0})
            if not agree:
                continue
            tr = agree[0]
            log = []
            ok = leaves_hold(G, facts, q, tr, log)
            demo_tr = [(tk.decode(tk(" " + x)["input_ids"]).strip(), d) for x, d in demo]
            st["agreeing right"] += 1
            st["holds"] += ok
            st["same as demo"] += tr == demo_tr
            st["holds and not demo"] += ok and tr != demo_tr
            print(f"[{key}] {q} (truth {ans}) | holds {ok} | same steps as the demo {tr == demo_tr}\n  facts: "
                  + " ".join(facts) + "\n  run: " + " | ".join(f"{x}{'' if d else ' ->'}" for x, d in tr)
                  + "\n" + "\n".join(log), flush=True)
    print("holddiag " + "; ".join(f"{k}: " + ", ".join(f"{a} {b}" for a, b in v.items()) for k, v in stats.items()), flush=True)


@torch.no_grad()
def v10_ctxdiag(path, n=30, seed=47):
    """11-16 v10s: kept 3-4 hop chains jump hops ('Who is the boss of Vinba?' -> the second-last name) and chain2 falls
    after training on own runs. Is the jump read from the ancestors in the step's context? On the CORRECT solution path
    (facts given, kept memory on, as in the rounds), each step writes its node with the ancestors in context (as built)
    and without (focus + its finished sub-steps + fetched); exact match vs the correct node, by kind and step type"""
    tk = lm()[0]
    G = v10_load(path)
    rng = random.Random(seed)
    res, shown = {}, {}
    for kind, size in (("chain", 2), ("chain", 3), ("chain", 4), ("compose", 1)):
        for _ in range(n):
            facts, q, ans, steps = v10_task(kind, rng, size)
            mem = []
            for f in facts:
                G.store(mem, tk(" " + f)["input_ids"], kind="fact")
            root = G.store(mem, tk(" " + q)["input_ids"], kind="question")
            focus = root
            for text, done in steps:
                ids = tk(" " + text)["input_ids"]
                want = tk.decode(ids).strip()
                typ = ("answer" if done else "sub-q") + (" at root" if focus is root else " below root")
                for anc in (True, False):
                    node, d, e, pd, nk = G.step(mem, focus, ancestors=anc)
                    got = tk.decode(node.ids).strip()
                    res.setdefault((anc, f"{kind}{size} {typ}"), []).append(got == want)
                    if anc and typ == "answer at root" and shown.get(kind + str(size), 0) < 4:
                        shown[kind + str(size)] = shown.get(kind + str(size), 0) + 1
                        print(f"  [{kind}{size}] {q} | at the root after its sub-steps: wants {want!r} (done), "
                              f"writes {got!r} (done p {pd:.2f})", flush=True)
                # the correct node, built as training builds it: its states inside the step that writes it (11-16: from
                # the text alone its control differs -- the root then wrote nothing after its sub-steps, an artifact)
                cids, cb, e, _, _ = G.context(mem, focus, False)
                G.ex.active = e
                H, _, _ = G.lm_run(cids + ids, torch.cat([cb, torch.zeros(len(ids), device=dev)]) if cb is not None else None)
                G.ex.active = None
                nd = G.node_from(ids, H[len(cids):], parent=focus)
                if done:
                    nxt = _resolve(G, mem, focus, nd)
                    if focus is root:
                        break
                    focus = nxt
                else:
                    focus = nd
    for anc in (True, False):
        print(f"ctxdiag {'with ancestors (v10s input)' if anc else 'without ancestors (v10t, as built)'}: " + ", ".join(
            f"{k[1]} {sum(v) / len(v):.0%} (n {len(v)})" for k, v in sorted(res.items()) if k[0] == anc), flush=True)


def _unit_step(kind, rng, shuffle=False, combos=None):
    """one isolated step: (facts, focus text, target text). kinds: column (one addition column), lookup, hop (one chain
    hop), genus (first compose step). shuffle: the target is a random wrong-or-right digit pair (negative control)"""
    if kind == "column":
        x, y, c = rng.choice(combos)
        t = x + y + c
        tgt = f"digit {t % 10}, carry {t // 10}"
        if shuffle:
            tgt = f"digit {rng.randrange(10)}, carry {rng.randrange(2)}"
        return [], f"{x} + {y} + carry {c} = ?", tgt
    if kind == "lookup":
        facts, q, ans, _ = v10_task("lookup", rng, 1)
        return facts, q, ans
    if kind == "hop":
        names, facts, qtext, steps = _chain_world(rng, 1)
        return facts, qtext(1, 0), names[1]
    facts, q, ans, steps = v10_task("compose", rng, 1)
    return facts, steps[0][0], steps[1][0]


def v10_unit(steps=1500):
    """11-16 (measurement only; user: results against a design may be my code, isolate the test). No kept memory, no
    fetch training, facts given; each arm a fresh model, same seed, same steps.
    U1: can the step path (context -> LM + one LoRA expert -> written node) learn single steps? Controls: shuffled
        column targets must stay near chance (~5%); held-out digit combos show if columns generalise.
    U2: skill experts (D3) -- one shared expert / one fixed expert per kind (oracle routing, measurement only) / learned
        routing with spawning on. Oracle >= shared -> experts work; learned << oracle -> routing code is the problem."""
    tk = lm()[0]
    allc = [(x, y, c) for x in range(10) for y in range(10) for c in (0, 1)]
    random.Random(3).shuffle(allc)
    train_c, held_c = allc[:160], allc[160:]
    kinds = ("column", "lookup", "hop", "genus")

    def run(name, kinds_used, route=None, spawn=None, shuffle=False):
        torch.manual_seed(0)
        random.seed(0)
        rng = random.Random(11)
        G = GNM10().to(dev)
        frng = random.Random(5)
        G.fix_skill_mean([tk(" " + _unit_step(k, frng, combos=train_c)[1])["input_ids"] for k in kinds for _ in range(20)])
        if spawn is not None:
            G.spawn_at = spawn
        if route is not None:                            # oracle routing: kind -> expert slot (measurement only)
            cur = {}
            G.pick_skill = lambda q, focus, train: (cur["e"], None, 1.)
        params = [p for nm, p in G.named_parameters() if p.requires_grad] + G.eparams()
        opt = torch.optim.AdamW(params, lr=1e-3)
        used = {}
        out = []
        for st in range(1, steps + 1):
            kind = kinds_used[st % len(kinds_used)]
            facts, ftext, tgt = _unit_step(kind, rng, shuffle and kind == "column", train_c)
            if route is not None:
                cur["e"] = route[kind]
            mem = []
            for f in facts:
                G.store(mem, tk(" " + f)["input_ids"], kind="fact")
            focus = G.store(mem, tk(" " + ftext)["input_ids"], kind="question")
            L, _ = G.step_loss(mem, focus, tk(" " + tgt)["input_ids"], 1)
            opt.zero_grad()
            L.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.)
            opt.step()
            if st in (300, 750, steps):
                G.eval()
                res, erng = {}, random.Random(99)
                with torch.no_grad():
                    for kind2 in kinds_used:
                        for split, combos in ((("", train_c),) if kind2 != "column" else (("", train_c), (" held-out", held_c))):
                            hit = 0
                            for _ in range(40):
                                facts, ftext, tgt = _unit_step(kind2, erng, shuffle and kind2 == "column", combos)
                                if route is not None:
                                    cur["e"] = route[kind2]
                                mem = []
                                for f in facts:
                                    G.store(mem, tk(" " + f)["input_ids"], kind="fact")
                                focus = G.store(mem, tk(" " + ftext)["input_ids"], kind="question")
                                node, d, e, pd, nk = G.step(mem, focus)
                                hit += tk.decode(node.ids).strip() == tgt
                                used.setdefault(kind2, {}).setdefault(e, 0)
                                used[kind2][e] += 1
                            res[kind2 + split] = hit / 40
                G.train()
                stab = {k: max(v.values()) / sum(v.values()) for k, v in used.items()}
                out.append(f"@{st}: " + ", ".join(f"{k} {v:.0%}" for k, v in res.items())
                           + f" | skills {len(G.skills)}, routing stability " + ", ".join(f"{k} {v:.0%}" for k, v in stab.items()))
                used = {}
        print(f"unit {name}: " + " || ".join(out), flush=True)
        for h in G.ex.hooks:
            h.remove()
        del G, opt

    arms = os.environ.get("UNIT_ARMS", "neg,shared,oracle,learned").split(",")
    if "neg" in arms:
        run("U1 shuffled column targets (negative control)", ("column",), shuffle=True)
    if "shared" in arms:
        run("U1/U2 one shared expert", kinds)
    if "oracle" in arms:
        run("U2 oracle routing (one expert per kind)", kinds, route={k: i for i, k in enumerate(kinds)})
    if "learned" in arms:
        sp = float(os.environ.get("UNIT_SPAWN", ".3"))
        run(f"U2 learned routing, spawning on (spawn_at {sp}; fixed-centred frozen skill space)", kinds, spawn=sp)


@torch.no_grad()
def v10_unit_route(n=20):
    """U0 (measurement only): does a routing space separate step kinds, at start, without drifting? Same-kind vs
    other-kind cosine and nearest-neighbour kind match, for: the current skill space (skill_query, untrained), the
    plain frozen LM states (mean over tokens), and those centred by one fixed mean"""
    tk = lm()[0]
    G = GNM10().to(dev)
    rng = random.Random(21)
    allc = [(x, y, c) for x in range(10) for y in range(10) for c in (0, 1)]
    kinds = ("column", "lookup", "hop", "genus")
    rows, lab = {"skill space": [], "frozen mean": []}, []
    for k in kinds:
        for _ in range(n):
            _, ftext, _ = _unit_step(k, rng, combos=allc)
            H, _, _ = G.lm_run(tk(" " + ftext)["input_ids"])
            rows["skill space"].append(G.skill_query(H))
            rows["frozen mean"].append(H.mean(0))
            lab.append(k)
    X = torch.stack(rows["frozen mean"])
    rows["frozen mean, centred (fixed)"] = list(X - X.mean(0))
    lab_t = torch.tensor([kinds.index(k) for k in lab], device=dev)
    for name, vs in rows.items():
        V = F.normalize(torch.stack(vs), dim=-1)
        S = V @ V.T
        same = lab_t[:, None] == lab_t[None, :]
        eye = torch.eye(len(lab), dtype=torch.bool, device=dev)
        S2 = S.masked_fill(eye, -9)
        nn_ok = (lab_t[S2.argmax(1)] == lab_t).float().mean()
        print(f"unit U0 {name}: same-kind cos {float(S[same & ~eye].mean()):.3f} (min {float(S[same & ~eye].min()):.3f}), "
              f"other-kind cos {float(S[~same].mean()):.3f} (max {float(S[~same].max()):.3f}), nearest neighbour same kind {float(nn_ok):.0%}", flush=True)
    for h in G.ex.hooks:
        h.remove()


def v10_unit_column(steps=6000):
    """U1b (measurement only): column steps alone, one shared expert, 160 digit combos ~37 times each: can the step
    path (LM + one LoRA expert) learn one addition column? Held-out 40 combos = does it generalise"""
    tk = lm()[0]
    allc = [(x, y, c) for x in range(10) for y in range(10) for c in (0, 1)]
    random.Random(3).shuffle(allc)
    train_c, held_c = allc[:160], allc[160:]
    torch.manual_seed(0)
    random.seed(0)
    rng = random.Random(11)
    G = GNM10().to(dev)
    params = [p for nm, p in G.named_parameters() if p.requires_grad] + G.eparams()
    opt = torch.optim.AdamW(params, lr=1e-3)
    out = []
    for st in range(1, steps + 1):
        facts, ftext, tgt = _unit_step("column", rng, combos=train_c)
        mem = []
        focus = G.store(mem, tk(" " + ftext)["input_ids"], kind="question")
        L, _ = G.step_loss(mem, focus, tk(" " + tgt)["input_ids"], 1)
        opt.zero_grad()
        L.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.)
        opt.step()
        if st in (1500, 3000, steps):
            G.eval()
            res = {}
            with torch.no_grad():
                for split, combos in (("trained combos", train_c), ("held-out combos", held_c)):
                    hit, wrong = 0, []
                    for x, y, c in combos:
                        mem = []
                        focus = G.store(mem, tk(f" {x} + {y} + carry {c} = ?")["input_ids"], kind="question")
                        node, d, e, pd, nk = G.step(mem, focus)
                        got, t = tk.decode(node.ids).strip(), x + y + c
                        ok = got == f"digit {t % 10}, carry {t // 10}"
                        hit += ok
                        if not ok and len(wrong) < 3:
                            wrong.append(f"{x}+{y}+{c} -> {got!r}")
                    res[split] = f"{hit / len(combos):.0%} (e.g. {'; '.join(wrong)})"
            G.train()
            out.append(f"@{st} loss {float(L):.3f}: " + ", ".join(f"{k} {v}" for k, v in res.items()))
            print("unit U1b column only " + out[-1], flush=True)
    for h in G.ex.hooks:
        h.remove()


def _focus_shape(t):
    """which kind of focus a text is (diagnosis only, never used by the model)"""
    import re as _re
    for name, pat in (("column", r"^\d \+ \d \+ carry \d = \?$"), ("add question", r"^What is [\d ]+\+"),
                      ("genus question", r"^Which genus is"), ("family-of-genus question", r"^Which family is the genus"),
                      ("compose question", r"^Which family is"), ("lookup question", r"^What is the \w+ of"),
                      ("1-hop question", r"^Who is the (\w+) of (?!the )\w+\?$"), ("multi-hop question", r"^Who is the")):
        if _re.search(pat, t.strip()):
            return name
    return "other"


@torch.no_grad()
def v10_routediag(path, n=12, seed=53):
    """11-16 v10w round 0: with spawning on (fixed-centred frozen skill space) chain2 10%, lookup 38%, compose 2%;
    compose went to s0 (which also took chains), add spread over 4 skills. Hypothesis: the learned key offsets EK grew
    (gradient via w = p/p.detach() reinforces whatever was picked) and one skill's key captures other focus kinds.
    Prints EK norm vs label norm per skill, and per focus kind (demo-shaped tasks): which skill is picked, how often"""
    tk = lm()[0]
    G = v10_load(path)
    for s_ in G.skills:
        lab = G.skill_query(s_.H)
        print(f"routediag skill s{s_.expert}: label {tk.decode(s_.ids).strip()[:60]!r} | label norm {float(lab.norm()):.2f}, "
              f"EK norm {float(G.EK[s_.expert].norm()):.2f}", flush=True)
    rng = random.Random(seed)
    pick = {}
    for kind, sizes in (("add", (1, 2, 3)), ("chain", (1, 2, 3)), ("lookup", (1,)), ("compose", (1,))):
        for size in sizes:
            for _ in range(n):
                facts, q, ans, steps = v10_task(kind, rng, size)
                for t in [q] + [x for x, d in steps if not d]:
                    H, _, _ = G.lm_run(tk(" " + t)["input_ids"])
                    qv = F.normalize(G.skill_query(H), dim=-1)
                    sc = G.skill_keys() @ qv
                    b = int(sc.argmax())
                    pick.setdefault(_focus_shape(t), {}).setdefault(f"s{G.skills[b].expert}", []).append(float(sc[b]))
    for shape, d in sorted(pick.items()):
        print(f"routediag {shape}: " + ", ".join(f"{k} x{len(v)} (score {sum(v) / len(v):.2f})" for k, v in sorted(d.items())), flush=True)


def v10_fitdiag(steps=300):
    """11-15 v10g: demo fit stuck at mean loss ~1.9 with 10 skills. Same 48 demos, fresh model per variant: as is /
    one skill / no kept memory / both; mean loss per 100 steps (lower = imitates the demos)"""
    rng = random.Random(5)
    train_sizes = [(k, s) for k in ("add", "chain") for s in (1, 2, 3)] + [("lookup", 1), ("compose", 1)]
    demos = [v10_task(k, rng, s) for k, s in train_sizes for _ in range(50 // len(train_sizes))]
    for name, spawn, kept in (("as is", None, True), ("one skill", -2., True), ("no kept memory", None, False), ("one skill, no kept", -2., False)):
        torch.manual_seed(0)
        random.seed(0)
        G = GNM10().to(dev)
        if spawn is not None:
            G.spawn_at = spawn
        if kept:
            for i, (facts, q, ans, st) in enumerate(demos):
                keep_trace(G, q, st, i)
        params = [p for nm, p in G.named_parameters() if p.requires_grad] + G.eparams()
        opt = torch.optim.AdamW(params, lr=1e-3)
        run, curve = [], []
        for st in range(1, steps + 1):
            if st % 50 == 1:
                G.refresh_ltm()
            i = random.randrange(len(demos))
            facts, q, ans, tr = demos[i]
            L = v10_demo_loss(G, facts, q, tr, src=i)
            opt.zero_grad()
            L.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.)
            opt.step()
            run.append(float(L))
            if st % 100 == 0:
                curve.append(f"{sum(run) / len(run):.3f}")
                run = []
        for h in G.ex.hooks:
            h.remove()
        print(f"fitdiag {name}: mean loss per 100 steps {' '.join(curve)}; skills {len(G.skills)}", flush=True)
        del G, opt


@torch.no_grad()
def v10_cachediag(n=10):
    """11-15: selftest 3 failed in a run (cached vs slow logits 88% apart) but passed alone. Per case: largest relative
    logit difference under the same tokens, and the first step where it is > 3%, for: as is / no bias / no expert / fp32"""
    tk = lm()[0]
    G = GNM10().to(dev)
    ids = tk(" The boss of Kalo is Vintor. Who is the boss of Kalo? The boss of Kalo is")["input_ids"]
    G.pick_skill(F.normalize(G.skill_query(G.lm_run(ids)[0]), dim=-1), Node10(ids, None, None), True)
    for A, B in zip(G.ex.A, G.ex.B):
        B.data[0].normal_(0, .02)

    def one(rb, expert):
        G.ex.active = expert
        slow, slow_lg = [], []
        for _ in range(10):
            _, lg, _ = G.lm_run(ids + slow, None if rb is None else torch.cat([rb, torch.zeros(len(slow), device=dev)]))
            slow_lg.append(lg[-1])
            t = int(lg[-1].argmax())
            if t == tk.eos_token_id:
                break
            slow.append(t)
        _, lg, past = G.lm_run(ids, rb, cache=True)
        fast, allb = [lg[-1]], (rb if rb is not None else torch.zeros(len(ids), device=dev))
        for t in slow[:len(slow_lg) - 1]:
            allb = torch.cat([allb, torch.zeros(1, device=dev)])
            _, lg, past = G.lm_run([t], allb, past=past, cache=True)
            fast.append(lg[-1])
        G.ex.active = None
        r = [float((a - b).abs().max() / b.abs().max()) for a, b in zip(fast, slow_lg)]
        first = next((i for i, x in enumerate(r) if x > .03), -1)
        return max(r), first, float(rb.min()) if rb is not None else 0.
    for case in range(n):
        torch.manual_seed(case)
        rb = torch.randn(len(ids), device=dev)
        row = []
        for name, b, e, amp in (("as is", rb, 0, True), ("no bias", None, 0, True), ("no expert", rb, None, True), ("fp32", rb, 0, False)):
            _V10_AMP[0] = amp
            m, f, mn = one(b, e)
            row.append(f"{name} {m:.3f}@{f}")
        _V10_AMP[0] = False
        print(f"case {case} (bias min {float(rb.min()):.2f} max {float(rb.max()):.2f}): " + " | ".join(row), flush=True)


@torch.no_grad()
def v10_selftest():
    """checks that fail on a fake: (1) zero control = no change, (2) control masking changes the answer, (3) cached
    writing == writing without a cache, (4) the skill choice gets a gradient, (5) a solved node is written to memory and
    kept nodes are fetch candidates in a later question"""
    tk = lm()[0]
    G = GNM10().to(dev)
    ids = tk(" The boss of Kalo is Vintor. Who is the boss of Kalo? The boss of Kalo is")["input_ids"]
    _, l0, _ = G.lm_run(ids)
    _, lz, _ = G.lm_run(ids, torch.zeros(len(ids), device=dev))
    b = torch.zeros(len(ids), device=dev)
    b[:8] = -1e4
    _, lb, _ = G.lm_run(ids, b)
    ok1 = torch.allclose(l0[-1], lz[-1], atol=1e-2)
    ok2 = int(l0[-1].argmax()) != int(lb[-1].argmax())
    print(f"selftest 1-2: zero bias == no bias: {ok1}; next token without bias {tk.decode([int(l0[-1].argmax())])!r}, "
          f"with the fact's tokens masked {tk.decode([int(lb[-1].argmax())])!r} (max logit change {float((l0[-1] - lb[-1]).abs().max()):.2f})", flush=True)
    # (3) cached greedy writing == the slow way (whole sequence each token), with a bias and an expert on
    G.pick_skill(F.normalize(G.skill_query(G.lm_run(ids)[0]), dim=-1), Node10(ids, None, None), True)
    for A, B in zip(G.ex.A, G.ex.B):
        B.data[0].normal_(0, .02)
    rb = torch.randn(len(ids), device=dev)
    G.ex.active = 0
    out, rows, hd = G.gen(ids, rb, 0., 10)
    slow, slow_lg = [], []
    for _ in range(10):
        _, lg, _ = G.lm_run(ids + slow, torch.cat([rb, torch.zeros(len(slow), device=dev)]))
        slow_lg.append(lg[-1])
        t = int(lg[-1].argmax())
        if t == tk.eos_token_id:
            break
        slow.append(t)
    # the same tokens through the cache, one at a time: logits must match the slow way at every step (11-15 v10e:
    # comparing greedy texts failed on a bf16 near-tie -- 'the of the' vs 'of of the', states 0.14% apart)
    _, lg, past = G.lm_run(ids, rb, cache=True)
    fast_lg, allb = [lg[-1]], rb
    for t in slow[:len(slow_lg) - 1]:
        allb = torch.cat([allb, torch.zeros(1, device=dev)])
        _, lg, past = G.lm_run([t], allb, past=past, cache=True)
        fast_lg.append(lg[-1])
    G.ex.active = None
    rel = max(float((a - b).abs().max() / b.abs().max()) for a, b in zip(fast_lg, slow_lg))
    ok3 = rel < .03 and len(fast_lg) == len(slow_lg)
    print(f"selftest 3: cached == slow: {ok3} (largest relative logit difference over {len(slow_lg)} steps {rel:.4f}; "
          f"greedy texts {tk.decode(out)!r} / {tk.decode(slow)!r})", flush=True)
    # (4) two skill nodes; a step's loss reaches the skill keys (EK, qry) through the expert weight
    with torch.enable_grad():
        G.spawn_at = 2.                                  # force a second skill
        mem = []
        G.store(mem, tk(" The boss of Kalo is Vintor.")["input_ids"])
        root = G.store(mem, tk(" Who is the friend of Kalo?")["input_ids"], kind="question")
        G.step_loss(mem, root, tk(" Vintor")["input_ids"], 1)
        G.spawn_at = -2.
        L, _ = G.step_loss(mem, root, tk(" Vintor")["input_ids"], 1)
        L.backward()
        g = float(G.EK.grad.abs().sum()) if G.EK.grad is not None else 0.
        G.spawn_at = .5
    ok4 = len(G.skills) == 2 and g > 0
    # (6) fetching learns from the start: with untrained (zero) control bias, a fetched fact's key gets a gradient
    with torch.enable_grad():
        G2 = GNM10().to(dev)
        mem = []
        fact = G2.store(mem, tk(" The boss of Kalo is Vintor.")["input_ids"])
        G2.store(mem, tk(" The boss of Mira is Tolde.")["input_ids"])
        fact.key.retain_grad()
        root = G2.store(mem, tk(" Who is the boss of Kalo?")["input_ids"], kind="question")
        L, _ = G2.step_loss(mem, root, tk(" Vintor")["input_ids"], 1)
        L.backward()
        g6 = float(fact.key.grad.abs().sum()) if fact.key.grad is not None else 0.
        for h in G2.ex.hooks:
            h.remove()
        del G2
    ok6 = g6 > 0
    print(f"selftest 6: fetch gradient on a fact's key with untrained control {g6:.2e}: {ok6}", flush=True)
    print(f"selftest 4: skill nodes {len(G.skills)}, gradient on skill keys {g:.2e}: {ok4}", flush=True)
    # (5) a solved node is written; kept nodes are fetch candidates later
    mem = []
    G.store(mem, tk(" The boss of Kalo is Vintor.")["input_ids"])
    G.done.bias.data += 20.                              # resolve at once, so there is a solved node to check
    got, st, pairs = solve(G, mem, tk(" Who is the boss of Kalo?")["input_ids"], max_steps=3)
    G.done.bias.data -= 20.
    n_solved = sum(n.kind == "solved" for n in mem)
    for f, r in pairs:
        G.keep(f + r, "t")
    cands_later = len(G.ltm)
    G.topk = 100
    fetched, _ = G.fetch([], G.query(tk(" Who is the boss of Kalo?")["input_ids"]))
    G.topk = 6
    ok5 = len(pairs) == 1 and (n_solved == len(pairs)) and (cands_later == len(pairs)) and (len(fetched) == len(pairs))
    print(f"selftest 5: resolves {len(pairs)}, solved nodes in memory {n_solved}, kept {cands_later}, fetchable later {len(fetched)}: {ok5}", flush=True)
    # (7) a fact read once is kept and found later with nothing given
    fid = tk(" The teacher of Mira is Dolan.")["input_ids"]
    keep_facts(G, [" The teacher of Mira is Dolan.".strip()], "t7")
    fetched, _ = G.fetch([], G.query(tk(" Who is the teacher of Mira?")["input_ids"]))
    ok7 = any(n.ids == fid for n in fetched)
    print(f"selftest 7: a fact read once is kept and fetched later with no facts given: {ok7}", flush=True)
    # (8) a context longer than the executor's window still writes (cut to the window, newest tokens kept): one
    # focus with 20 long finished sub-steps
    mem = []
    long_ids = tk(" Who is the boss of the friend of the teacher of the father of Kalo, said Mira to Dolan at noon?" * 3)["input_ids"]
    f = G.store(mem, long_ids, kind="question")
    for _ in range(20):
        c = G.node_from(long_ids, G.lm_run(long_ids)[0], parent=f)
        c.result = G.node_from(long_ids, G.lm_run(long_ids)[0], parent=c)
        f.children.append(c)
    ids8, b8, _, _, _ = G.context(mem, f, False)
    node8, _, _, _, _ = G.step(mem, f, 0., 64)
    ok8 = len(ids8) + 66 <= lm()[1].config.n_positions and len(ids8) == len(b8)
    print(f"selftest 8: focus + sub-steps of {41 * len(long_ids)} tokens -> context {len(ids8)} tokens, wrote {len(node8.ids)} tokens: {ok8}", flush=True)
    assert ok1 and ok2 and ok3 and ok4 and ok5 and ok6 and ok7 and ok8, "v10 selftest failed"
    G.ltm, G._ltm_ids, G.skills = [], set(), []


# ====================================================================== v11 trial: bind -> point -> remember (11-16)
class V11:
    """docs/GNM_v10.md §1b. Vector-symbolic binding (circular convolution; argument position = a fixed permutation)
    + associative memories (one per op node: a list of bound queries and the node each points to). Fixed math: nothing
    here is trained; memory is written once per confirmed fact"""

    def __init__(self, D=2048, beta=50., seed=0):
        self.g = torch.Generator().manual_seed(seed)
        self.D, self.beta = D, beta
        self.perm = [torch.randperm(D, generator=self.g).to(dev) for _ in range(4)]
        self.atoms = {}
        self.mem = {}                                    # op name -> [keys], [values]

    def atom(self, name, vec=None):
        if vec is not None:
            self.atoms[name] = F.normalize(vec.float().to(dev), dim=0)
        elif name not in self.atoms:
            self.atoms[name] = F.normalize(torch.randn(self.D, generator=self.g).to(dev), dim=0)
        return self.atoms[name]

    @staticmethod
    def bind(a, b):
        return torch.fft.irfft(torch.fft.rfft(a) * torch.fft.rfft(b), n=a.shape[-1])

    @staticmethod
    def unbind(c, a):                                    # circular correlation: approximate inverse of bind
        return torch.fft.irfft(torch.fft.rfft(c) * torch.conj(torch.fft.rfft(a)), n=c.shape[-1])

    def query(self, op, args):
        """bind: the op node with each argument at its position"""
        q = self.atom(op)
        for i, a in enumerate(args):
            q = self.bind(q, self.atom(a)[self.perm[i]])
        return F.normalize(q, dim=0)

    def remember(self, op, args, value):
        K, V = self.mem.setdefault(op, ([], []))
        K.append(self.query(op, args))
        V.append(value)

    def point(self, op, args):
        """-> (read value, best key cosine). Softmax(beta * cos) over the op node's keys"""
        K, V = self.mem[op]
        Kt = torch.stack(K) if not isinstance(K, torch.Tensor) else K
        sc = Kt @ self.query(op, args)
        w = torch.softmax(self.beta * sc, 0)
        return w @ (torch.stack(V) if not isinstance(V, torch.Tensor) else V), float(sc.max())

    def freeze(self, op):                                # stack a memory once it is written (fast reads)
        K, V = self.mem[op]
        self.mem[op] = (torch.stack(K), torch.stack(V))

    def cleanup(self, v, names):
        """classify a vector to the nearest of the candidate nodes"""
        A = torch.stack([self.atoms[n] for n in names])
        return names[int((A @ F.normalize(v, dim=0)).argmax())]


@torch.no_grad()
def v11_test():
    """measurement (docs/GNM_v10.md §1b, tests T1a-d, T2). Nothing is learned; every number is fixed math + one-shot
    memory. Controls: T1b (never-written combos must not be answered), random atoms vs GPT-2 node vectors in T2"""
    torch.manual_seed(0)
    random.seed(0)
    # ---- T1: digit-column table, value = bundle of two slots (digit, carry)
    m = V11()
    digits, carries = [str(i) for i in range(10)], ["c0", "c1"]
    for n in digits + carries + ["add", "slot digit", "slot carry"]:
        m.atom(n)
    allc = [(x, y, c) for x in range(10) for y in range(10) for c in (0, 1)]
    random.Random(3).shuffle(allc)

    def col_value(x, y, c):
        t = x + y + c
        return m.bind(m.atom("slot digit"), m.atom(str(t % 10))) + m.bind(m.atom("slot carry"), m.atom(f"c{t // 10}"))

    def col_read(x, y, c):
        v, conf = m.point("add", [str(x), str(y), f"c{c}"])
        return m.cleanup(m.unbind(v, m.atom("slot digit")), digits), m.cleanup(m.unbind(v, m.atom("slot carry")), carries), conf

    for x, y, c in allc[:160]:
        m.remember("add", [str(x), str(y), f"c{c}"], col_value(x, y, c))
    m.freeze("add")
    ok = sum(col_read(x, y, c)[:2] == (str((x + y + c) % 10), f"c{(x + y + c) // 10}") for x, y, c in allc[:160])
    held = [col_read(x, y, c) for x, y, c in allc[160:]]
    held_ok = sum((d, k) == (str((x + y + c) % 10), f"c{(x + y + c) // 10}") for (d, k, _), (x, y, c) in zip(held, allc[160:]))
    conf_w = sorted(col_read(x, y, c)[2] for x, y, c in allc[:160])
    conf_h = sorted(h[2] for h in held)
    print(f"v11 T1a: 160 column facts written once, read back (both slots exact): {ok}/160; key cosine of written "
          f"queries min {conf_w[0]:.2f}", flush=True)
    print(f"v11 T1b (control): 40 never-written combos: best key cosine max {conf_h[-1]:.2f} (written: >= {conf_w[0]:.2f}) "
          f"-> answered with confidence (> .5): {sum(h[2] > .5 for h in held)}/40; right if forced: {held_ok}/40 "
          f"(chance ~2/40)", flush=True)
    # T1d: the full 200-entry table, then multi-digit addition = the column lookup repeated (procedure FIXED here)
    m.mem.pop("add")
    for x, y, c in allc:
        m.remember("add", [str(x), str(y), f"c{c}"], col_value(x, y, c))
    m.freeze("add")
    rng = random.Random(7)
    res = []
    for L in (2, 4, 8, 12):
        hit = 0
        for _ in range(200):
            a, b = rng.randint(10 ** (L - 1), 10 ** L - 1), rng.randint(10 ** (L - 1), 10 ** L - 1)
            da, db, c, out = str(a), str(b), 0, ""
            for i in range(L - 1, -1, -1):
                d, k, _ = col_read(int(da[i]), int(db[i]), c)
                out, c = d + out, int(k[1])
            hit += int(("1" if c else "") + out) == a + b
        res.append(f"{L} digits {hit / 200:.0%}")
    print("v11 T1d: addition by repeated column lookups (procedure fixed): " + ", ".join(res), flush=True)
    # ---- T1c: capacity, N random facts (op r, x, y) -> z over 1000 atoms
    for N in (1000, 10000, 50000):
        mc = V11(seed=1)
        names = [f"a{i}" for i in range(1000)]
        for n in names + ["r"]:
            mc.atom(n)
        r2 = random.Random(N)
        pairs = r2.sample(range(1000 * 1000), N)
        facts = [(names[p // 1000], names[p % 1000], names[r2.randrange(1000)]) for p in pairs]
        for x, y, z in facts:
            mc.remember("r", [x, y], mc.atoms[z])
        mc.freeze("r")
        test = r2.sample(facts, 500)
        hit = sum(mc.cleanup(mc.point("r", [x, y])[0], names) == z for x, y, z in test)
        print(f"v11 T1c capacity: {N} facts in one op memory, 500 read back: {hit / 500:.0%}", flush=True)
        del mc
    # ---- T2: facts read once from many worlds; one-hop / chains / compose by lookups
    import re as _re
    wr = random.Random(13)
    worlds = []
    for i in range(400):
        size = 1 + i % 8
        names, facts, qtext, steps = _chain_world(wr, size)
        worlds.append(("chain", facts, names, size))
    for i in range(200):
        facts, q, ans, steps = v10_task("compose", wr, 1)
        worlds.append(("compose", facts, q, ans))

    def parse(f):                                        # fixed parse = stand-in for the translator (V7, not built)
        mm = _re.match(r"The (\w+) of (\w+) is (\w+)\.$", f)
        if mm:
            return mm[1], mm[2], mm[3]
        mm = _re.match(r"(\w+) is in the genus (\w+)\.$", f)
        if mm:
            return "genus", mm[1], mm[2]
        mm = _re.match(r"The genus (\w+) is in the family (\w+)\.$", f)
        return "family", mm[1], mm[2]

    # every world gets its own fresh names (11-16 first run: generated names repeat across 600 worlds -> 1230 clashing
    # keys, almost every question excluded): two syllable-names joined, never reused
    used, nr = set(), random.Random(17)

    def fresh():
        while True:
            n = _nm(nr) + _nm(nr).lower()
            if n not in used:
                used.add(n)
                return n
    renamed = []
    for w in worlds:
        mp = {}
        f = lambda n: mp.setdefault(n, fresh())
        tr = [(r_, f(x), f(y)) for r_, x, y in (parse(t) for t in w[1])]
        if w[0] == "chain":
            renamed.append(("chain", tr, [f(n) for n in w[2]], w[3]))
        else:
            x = _re.match(r"Which family is (\w+) in\?", w[2])[1]
            renamed.append(("compose", tr, f(x), f(w[3])))
    worlds = renamed
    triples = [t for w in worlds for t in w[1]]
    ents = sorted({t[1] for t in triples} | {t[2] for t in triples})
    seen, clash = {}, set()
    for rel, x, y in triples:
        if (rel, x) in seen and seen[(rel, x)] != y:
            clash.add((rel, x))
        seen[(rel, x)] = y
    tk, M = lm()
    wte = M.transformer.wte.weight.detach().float()
    Pm = torch.randn(wte.shape[1], 2048, generator=torch.Generator().manual_seed(5)).to(dev) / wte.shape[1] ** .5
    lmv = torch.stack([wte[tk(" " + e)["input_ids"]].mean(0) @ Pm for e in ents])
    lmv = lmv - lmv.mean(0)                              # centred by one fixed mean (U0 lesson)
    for mode in ("random atoms", "GPT-2 token vectors (centred)"):
        mt = V11(seed=2)
        for j, e in enumerate(ents):
            mt.atom(e, None if mode == "random atoms" else lmv[j])
        for rel in ("boss", "friend", "teacher", "neighbour", "genus", "family"):
            mt.atom(rel)
        for rel, x, y in triples:                        # read once: newest write per (rel, x) kept last
            mt.remember("rel", [rel, x], mt.atoms[y])
        mt.freeze("rel")
        out, hits = {}, {}
        for w in worlds:
            if w[0] == "chain":
                _, tr, names, size = w
                rel = tr[0][0]
                cur, ok, touched = names[0], True, False
                for k in range(size):
                    touched |= (rel, cur) in clash
                    cur = mt.cleanup(mt.point("rel", [rel, cur])[0], ents)
                ok = cur == names[size]
                key = f"chain{size}"
            else:
                _, tr, x, ans = w
                touched = ("genus", x) in clash
                g_ = mt.cleanup(mt.point("rel", ["genus", x])[0], ents)
                touched |= ("family", g_) in clash
                ok = mt.cleanup(mt.point("rel", ["family", g_])[0], ents) == ans
                key = "compose"
            if not touched:
                hits.setdefault(key, []).append(ok)
        print(f"v11 T2 {mode}: {len(triples)} facts read once into one memory, {len(clash)} (rel, name) keys written "
              f"twice with different answers (excluded): " + ", ".join(f"{k} {sum(v) / len(v):.0%} (n {len(v)})"
              for k, v in sorted(hits.items())), flush=True)
        del mt


def v11_t3(path=None, n_demo=50, steps=400):
    """T3 (docs/GNM_v10.md §1b, measurement): real biology text, questions worded differently from the facts.
    Reading = fixed parse of each species article's lead sentence (translator stand-in, marked): alias(common name) ->
    species node, alias(binomial) -> species node, family(species) -> family node, genus(species) -> genus node.
    Question side LEARNED from n_demo questions (2 wordings per relation) over frozen GPT-2 states: the name's span
    (point at its start / end tokens) and the relation, two ways: 'pooled' = classify the mean state over relation
    nodes; 'point' = point at the relation's word, then classify that token against the relation nodes' word vectors.
    Name keys two ways: 'mean' = centred mean of the tokens' embeddings; 'exact' = the tokens bound in order (only
    the identical string matches). Answer: name lookup (no confident key -> 'unknown') -> bind(relation, species)
    lookup -> classify over ALL nodes. Controls: gold relation + span; 15 species never read must give 'unknown'"""
    import re as _re
    tk, M = lm()
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "bio", "species_e1.json")
    arts = json.load(open(path, encoding="utf-8"))
    sp = []
    for a in arts:                                       # eval construction from the lead (not seen by the model)
        p0 = a["text"][0]
        s0 = p0.split(" . ")[0]
        bm = _re.search(r"\( ([A-Z][a-z]+ [a-z]+)(?: [a-z]+)? [),;]", s0)
        fm = _re.search(r"family ([A-Z][a-z]+(?:idae|aceae|inae))", p0)
        cm = _re.match(r"^The ([a-z][a-z @\-@]+?) (?:\(|,|or)", s0)
        if bm and fm and cm and "<unk>" not in cm[1]:
            sp.append({"common": cm[1].replace(" @-@ ", "-"), "binomial": bm[1], "family": fm[1], "genus": bm[1].split()[0]})
    rng = random.Random(23)
    rng.shuffle(sp)
    train_sp, test_read, test_unread = sp[:30], sp[30:-15], sp[-15:]
    rels = ("family", "genus")
    W = {"family": (["Which family is {n} in?", "What family does {n} belong to?"],
                    ["{N} is placed in which family?", "Name the family of {n}."]),
         "genus": (["Which genus is {n} in?", "What genus does {n} belong to?"],
                   ["{N} is classified in which genus?", "Name the genus of {n}."])}
    rel_tok = {r_: tk(" " + r_)["input_ids"][0] for r_ in rels}

    def nm(x, kind):                                     # the name as written in a question
        return ("the " + x["common"]) if kind == "common" else x["binomial"]

    def make_q(x, rel, kind, tmpl):
        """-> ids, relation index, name start, name end, relation-word position"""
        n = nm(x, kind)
        q = tmpl.replace("{N}", n[0].upper() + n[1:]).replace("{n}", n)
        ids = tk(" " + q)["input_ids"]
        name_ids = tk(" " + (x["common"] if kind == "common" else x["binomial"]))["input_ids"]
        st = next((i for i in range(len(ids)) if ids[i:i + len(name_ids)] == name_ids), None)
        if st is None:                                   # sentence-initial capital: match the capitalised form
            name_ids = tk(" " + n[0].upper() + n[1:])["input_ids"] if kind == "binomial" else tk(" " + x["common"].capitalize())["input_ids"]
            st = next((i for i in range(len(ids)) if ids[i:i + len(name_ids)] == name_ids), None)
        rw = next((i for i, t in enumerate(ids) if t == rel_tok[rel]), None)
        return ids, rels.index(rel), st, (st + len(name_ids) - 1) if st is not None else None, rw
    # ---- memory: read the lead facts of train + test_read species (fixed parse); unread species never written
    m = V11(seed=4)
    tperm = [torch.randperm(m.D, generator=m.g).to(dev) for _ in range(24)]
    wte = M.transformer.wte.weight.detach().float()
    Pm = torch.randn(wte.shape[1], m.D, generator=torch.Generator().manual_seed(6)).to(dev) / wte.shape[1] ** .5
    read = train_sp + test_read
    names_all = [x["common"] for x in sp] + [x["binomial"] for x in sp]
    mu = torch.stack([wte[tk(" " + t)["input_ids"]].mean(0) @ Pm for t in names_all]).mean(0)

    def key_mean(ids):                                   # centred mean of the tokens' embeddings (U0 lesson)
        return F.normalize(wte[ids].mean(0) @ Pm - mu, dim=0)

    def key_exact(ids):                                  # the tokens bound in order: only the identical string matches
        v = None
        for i, t in enumerate(ids[:24]):
            a_ = m.atom(f"tok{t}")[tperm[i]]
            v = a_ if v is None else m.bind(v, a_)
        return F.normalize(v, dim=0)
    keyf = {"mean": key_mean, "exact": key_exact}
    for r_ in rels:                                      # relation nodes defined by their word's vector (before any fact
        m.atom(r_, wte[rel_tok[r_]] @ Pm)                # is written with them)
    nodes, alias_names = [], []
    for i, x in enumerate(read):
        e = f"species{i}"
        m.atom(e)
        nodes.append(e)
        for t in (x["family"], x["genus"]):
            if t not in m.atoms:
                m.atom(t)
                nodes.append(t)
        for kind in ("common", "binomial"):
            for km, f_ in keyf.items():
                K, V = m.mem.setdefault("alias " + km, ([], []))
                K.append(f_(tk(" " + x[kind])["input_ids"]))
                V.append(m.atoms[e])
            alias_names.append(x[kind])
        m.remember("rel", ["family", e], m.atoms[x["family"]])
        m.remember("rel", ["genus", e], m.atoms[x["genus"]])
    for k_ in list(m.mem):
        m.freeze(k_)

    def answer(ids, r, st, en, km):
        if r is None or st is None or en is None or en < st:
            return "unknown", 0., None
        K, V = m.mem["alias " + km]
        sc = K @ keyf[km](ids[st:en + 1])
        conf = float(sc.max())
        if conf < .5:
            return "unknown", conf, int(sc.argmax())
        ent = m.cleanup(V[int(sc.argmax())], nodes)
        return m.cleanup(m.point("rel", [rels[r], ent])[0], nodes), conf, int(sc.argmax())
    # ---- learned question head
    head = nn.ModuleDict({k_: nn.Linear(M.config.n_embd, len(rels) if k_ == "rel" else 1) for k_ in ("rel", "rw", "st", "en")}).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3)
    demos = []
    for _ in range(n_demo):
        x, rel, kind = rng.choice(train_sp), rng.choice(rels), rng.choice(("common", "binomial"))
        demos.append(make_q(x, rel, kind, rng.choice(W[rel][0])))
    demos = [d for d in demos if d[2] is not None and d[4] is not None]

    def states(ids):
        with torch.no_grad():
            return M(input_ids=torch.tensor([ids], device=dev), output_hidden_states=True).hidden_states[-1][0].float()
    for stp in range(steps):
        ids, r, st, en, rw = demos[stp % len(demos)]
        H = states(ids)
        tgt = lambda i: torch.tensor([i], device=dev)
        L = (F.cross_entropy(head["rel"](H.mean(0))[None], tgt(r)) + F.cross_entropy(head["rw"](H).squeeze(-1)[None], tgt(rw))
             + F.cross_entropy(head["st"](H).squeeze(-1)[None], tgt(st)) + F.cross_entropy(head["en"](H).squeeze(-1)[None], tgt(en)))
        opt.zero_grad()
        L.backward()
        opt.step()
    R = torch.stack([m.atoms[r_] for r_ in rels])

    def parse_learned(ids, rm):
        with torch.no_grad():
            H = states(ids)
            if rm == "pooled":
                r = int(head["rel"](H.mean(0)).argmax())
            else:                                        # point at the relation's word, classify it against relation nodes
                t = int(head["rw"](H).squeeze(-1).argmax())
                r = int((R @ F.normalize(wte[ids[t]] @ Pm, dim=0)).argmax())
            st = int(head["st"](H).squeeze(-1).argmax())
            en = st + int(head["en"](H).squeeze(-1)[st:].argmax())
        return r, st, en
    combos = [(rm, km) for rm in ("pooled", "point") for km in ("mean", "exact")]
    print(f"v11 T3: {len(sp)} species with a parsable lead ({len(train_sp)} train, {len(test_read)} test read, "
          f"{len(test_unread)} never read); {len(demos)} demo questions, {steps} head steps; memory: "
          f"{len(alias_names)} names, {len(m.mem['rel'][0])} relation facts, {len(nodes)} nodes", flush=True)
    for split, xs in (("held-out species, read", test_read), ("training species", train_sp)):
        for wset, wi in (("trained wordings", 0), ("held-out wordings", 1)):
            res = {c: [0, 0, 0] for c in combos + [("gold", "mean"), ("gold", "exact")]}   # right, unknown, total
            for x in xs:
                for rel in rels:
                    for kind in ("common", "binomial"):
                        for t in W[rel][wi]:
                            ids, r, st, en, rw = make_q(x, rel, kind, t)
                            if st is None:
                                continue
                            want = x[rel]
                            for c in res:
                                pr = (r, st, en) if c[0] == "gold" else parse_learned(ids, c[0])
                                got = answer(ids, *pr, c[1])[0]
                                res[c][0] += got == want
                                res[c][1] += got == "unknown"
                                res[c][2] += 1
            print(f"v11 T3 {split}, {wset}: " + "; ".join(f"relation {c[0]} / name key {c[1]}: right {v[0]}/{v[2]} "
                  f"(unknown {v[1]})" for c, v in res.items()), flush=True)
    for c in combos:
        unk, n_u, wrong_u = 0, 0, []
        for x in test_unread:
            for rel in rels:
                for kind in ("common", "binomial"):
                    for t in W[rel][1]:
                        ids, r, st, en, rw = make_q(x, rel, kind, t)
                        if st is None:
                            continue
                        r2, st2, en2 = parse_learned(ids, c[0])
                        got, conf, j = answer(ids, r2, st2, en2, c[1])
                        unk += got == "unknown"
                        n_u += 1
                        if got != "unknown" and len(wrong_u) < 4:
                            wrong_u.append(f"{tk.decode(ids[st2:en2 + 1]).strip()!r} matched {alias_names[j]!r} ({conf:.2f})")
        print(f"v11 T3 control (relation {c[0]} / name key {c[1]}): species never read -> 'unknown' {unk}/{n_u}"
              + ("; e.g. " + ", ".join(wrong_u) if wrong_u else ""), flush=True)

    # ---- T4 (user 11-16: "if can't point -> create new node"; "dont use rigid fixed tokens name, make it all vectors
    # ... a tool that helps estimate and approximate what it means"; why 2 nodes for one species -> reading must merge).
    # Lookup = point (match >= tau) or create. Reading the lead: each name points or creates; the lead states the two
    # names are one thing -> if they point to different nodes, MERGE (one node keeps both names; slots united, a filled
    # slot beats an open one). Name keys: 'exact' (tokens bound in order) vs 'encoder' (learned name tool).
    held_names = {x[k] for x in test_read + test_unread for k in ("common", "binomial")}
    pool = []                                            # names from the other species articles (encoder training)
    for a in arts:
        s0 = a["text"][0].split(" . ")[0]
        bm = _re.search(r"\( ([A-Z][a-z]+ [a-z]+)(?: [a-z]+)? [),;]", s0)
        cm = _re.match(r"^The ([a-z][a-z @\-@]+?) (?:\(|,|or)", s0)
        for nm_ in ((bm[1] if bm else None), (cm[1].replace(" @-@ ", "-") if cm and "<unk>" not in cm[1] else None)):
            if nm_ and nm_ not in held_names:
                pool.append(nm_)
    pool = sorted(set(pool))
    prng = random.Random(29)
    prng.shuffle(pool)
    val_names, tr_names = pool[:40], pool[40:]
    # + Wikipedia article titles (11-16: 142 species names were too few -- the tool judged by spelling, other similar
    # names < tau only 59%): plain names of 2-5 words, letters only
    wpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bio", "wiki_e1.json.gz")
    if os.path.exists(wpath):
        wt = [a_["title"] for a_ in json.load(gzip.open(wpath, "rt", encoding="utf-8"))]
        wt = [t for t in wt if 2 <= len(t.split()) <= 5 and _re.fullmatch(r"[A-Za-z][A-Za-z' \-]*", t) and t not in held_names]
        tr_names = tr_names + wt
    name_steps = int(os.environ.get("V11_NAME_STEPS", "2000"))

    def variant(name, r_):
        """one way the name may appear in a question: article / capital / a token cut at either end (span errors)"""
        t = name
        if t[0].islower() and r_.random() < .5:
            t = "the " + t
        if r_.random() < .3:
            t = t[0].upper() + t[1:]
        ids = tk(" " + t)["input_ids"]
        core = tk(" " + name)["input_ids"]
        if len(core) >= 3 and r_.random() < .3:
            ids = ids[1:] if r_.random() < .5 else ids[:-1]
        return ids

    class NameTool(nn.Module):
        """name -> vector: frozen GPT-2 states of the name, attention-pooled, projected (learned)"""
        def __init__(s_):
            super().__init__()
            s_.att = nn.Linear(M.config.n_embd, 1)
            s_.out = nn.Linear(M.config.n_embd, m.D)

        def forward(s_, ids_list):
            vs = []
            for ids in ids_list:
                H = states(ids)
                w = torch.softmax(s_.att(H).squeeze(-1), 0)
                vs.append(s_.out((w[:, None] * H).sum(0)))
            return F.normalize(torch.stack(vs), dim=-1)
    enc = NameTool().to(dev)
    eopt = torch.optim.AdamW(enc.parameters(), lr=1e-3)
    by_genus = {}
    for n_ in tr_names:
        by_genus.setdefault(n_.split()[0], []).append(n_)
    genus_groups = [g for g in by_genus.values() if len(g) > 1]
    for stp in range(name_steps):
        batch = []
        while len(batch) < 32:                           # half from shared-first-word groups (hard negatives)
            if genus_groups and prng.random() < .5:
                batch += prng.sample(prng.choice(genus_groups), 2)
            else:
                batch.append(prng.choice(tr_names))
        batch = list(dict.fromkeys(batch))[:32]
        z1 = enc([variant(n_, prng) for n_ in batch])
        z2 = enc([variant(n_, prng) for n_ in batch])
        lg = z1 @ z2.T / .05
        lab = torch.arange(len(batch), device=dev)
        L = (F.cross_entropy(lg, lab) + F.cross_entropy(lg.T, lab)) / 2
        eopt.zero_grad()
        L.backward()
        eopt.step()
    enc.eval()

    @torch.no_grad()
    def sep(names, r_):
        """cosine of a name's variants to its key vs the most similar other name's key"""
        base = enc([tk(" " + n_)["input_ids"] for n_ in names])
        var = enc([variant(n_, r_) for n_ in names])
        pos = (base * var).sum(-1)
        S = base @ base.T
        S.fill_diagonal_(-1)
        return pos, S.max(1).values
    with torch.no_grad():
        pv, nv = sep(val_names, random.Random(31))
        grid = torch.linspace(0, 1, 101, device=dev)
        acc = torch.stack([((pv >= g).float().mean() + (nv < g).float().mean()) / 2 for g in grid])
        tau = float(grid[int(acc.argmax())])
        pt, nt = sep(sorted(held_names), random.Random(37))
    print(f"v11 T4 name tool: trained on {len(tr_names)} names (other species articles + Wikipedia titles; {name_steps} steps); tau from "
          f"{len(val_names)} validation names = {tau:.2f}. Test species names (never trained): a variant's cosine to its "
          f"name mean {float(pt.mean()):.2f} (min {float(pt.min()):.2f}); most similar OTHER name mean {float(nt.mean()):.2f} "
          f"(max {float(nt.max()):.2f}); variants >= tau {float((pt >= tau).float().mean()):.0%}, other names < tau "
          f"{float((nt < tau).float().mean()):.0%}", flush=True)
    with torch.no_grad():
        keys = {"exact": (key_exact, .5), "encoder": (lambda ids: enc([ids])[0], tau)}

    def run_arm(km, pm):
        keyfun, thr = keys[km]
        E = torch.zeros(0, m.D, device=dev)
        mem = {"alias": (E.clone(), E.clone()), "rel": (E.clone(), E.clone())}
        made, cands, merged = [], [], {}

        def lookup(op, key):
            K, V = mem[op]
            if not len(K):
                return -1., None, -1
            sc = K @ key
            j = int(sc.argmax())
            return float(sc[j]), V[j], j

        def write(op, key, val, same=.99):
            K, V = mem[op]
            sc, _, j = lookup(op, key)
            if sc > same:
                V[j] = val
            else:
                mem[op] = (torch.cat([K, key[None]]), torch.cat([V, val[None]]))

        def new(kind):
            name = f"{kind}{len(made)}"
            m.atom(name)
            made.append(name)
            cands.append(name)
            return name

        def name_of(v):
            return m.cleanup(v, cands)

        def point_or_create(key):
            sc, v, _ = lookup("alias", key)
            if sc >= thr:
                return name_of(v), False
            e = new("species ")
            write("alias", key, m.atoms[e], same=2.)     # a new name key is always appended
            return e, True

        def merge(keep, drop):
            """one node: the names of `drop` point to `keep`; its slots move over (a filled slot beats an open one)"""
            K, V = mem["alias"]
            hit = (V @ m.atoms[drop]) > .99
            V[hit] = m.atoms[keep]
            for r_ in rels:
                sd, vd, jd = lookup("rel", m.query("rel", [r_, drop]))
                if sd < .99:
                    continue
                sk, vk, jk = lookup("rel", m.query("rel", [r_, keep]))
                if sk >= .99:
                    if name_of(vk).startswith("open") and not name_of(vd).startswith("open"):
                        mem["rel"][1][jk] = vd
                    mem["rel"][0][jd] = 0.               # drop's slot disabled
                else:
                    mem["rel"][0][jd] = m.query("rel", [r_, keep])
            merged[drop] = keep
            cands.remove(drop)

        def read_lead(x):
            ents = [point_or_create(keyfun(tk(" " + x[k])["input_ids"]))[0] for k in ("common", "binomial")]
            if ents[0] != ents[1]:                       # the lead says both names are one thing
                merge(ents[0], ents[1])
            e = ents[0]
            for r_ in rels:
                if x[r_] not in m.atoms:
                    m.atom(x[r_])
                if x[r_] not in cands:
                    cands.append(x[r_])
                write("rel", m.query("rel", [r_, e]), m.atoms[x[r_]])

        def ask(ids, r, st, en):
            if st is None or en is None or en < st:
                st, en = 0, 0
            e, made_e = point_or_create(keyfun(ids[st:en + 1]))
            sc, v, _ = lookup("rel", m.query("rel", [rels[r], e]))
            if sc >= .5:
                return name_of(v)
            o = new("open ")
            write("rel", m.query("rel", [rels[r], e]), m.atoms[o])
            return o

        def questions(xs):
            out = []
            for x in xs:
                for rel in rels:
                    for kind in ("common", "binomial"):
                        for t in W[rel][1]:
                            ids, r, st, en, rw = make_q(x, rel, kind, t)
                            if st is not None:
                                out.append((x, rel, ids, (r, st, en) if pm == "gold" else parse_learned(ids, "point")))
            return out
        for x in read:
            read_lead(x)
        n0 = len(made)
        qa = questions(test_read)
        ga = [ask(ids, *pr) for x, rel, ids, pr in qa]
        a_right = sum(g == x[rel] for g, (x, rel, ids, pr) in zip(ga, qa))
        a_new = sum(g.startswith("open") for g in ga)
        qb = questions(test_unread)
        gb = [ask(ids, *pr) for x, rel, ids, pr in qb]
        b_wrong = sum(not g.startswith("open") for g in gb)
        n1 = len(made)
        nm0 = len(merged)
        for x in test_unread:
            read_lead(x)
        gc = [ask(ids, *pr) for x, rel, ids, pr in qb]
        c_right = sum(g == x[rel] for g, (x, rel, ids, pr) in zip(gc, qb))
        species = [c for c in cands if c.startswith("species")]
        junk = sum(lookup("rel", m.query("rel", ["family", e]))[0] < .99 or name_of(lookup("rel", m.query("rel", ["family", e]))[1]).startswith("open")
                   for e in species)
        split = sum(point_or_create(keyfun(tk(" " + x["common"])["input_ids"]))[0]      # (after the counts: a miss here
                    != point_or_create(keyfun(tk(" " + x["binomial"])["input_ids"]))[0]  #  creates a node)
                    for x in read + test_unread)
        print(f"v11 T4 name keys {km}, {pm} parse: reading 60 species made {n0} nodes | held-out wordings on read species: "
              f"right {a_right}/{len(qa)}, made a new open node {a_new} | 15 never-read species: answered with an existing node "
              f"{b_wrong}/{len(qb)} | read them: merges {len(merged) - nm0}; asked again right {c_right}/{len(qb)} | species "
              f"split over 2 nodes {split}/75; species nodes {len(species)} (no family known: {junk})", flush=True)
    with torch.no_grad():
        for km in ("exact", "encoder"):
            for pm in ("gold", "learned"):
                run_arm(km, pm)


def v11_t5(steps=3000, ep_species=8):
    """T5 (user 11-16): no merge, no new mechanism. Every name mention = the name transformed WITH its context (frozen
    GPT-2 states of the name's tokens inside its sentence / question + a learned projection); a learned relevance
    between it and each node's stored mentions decides: point at the most relevant node or create (one more option
    with its own learned score). Confusion is punished: a decision whose name points to the wrong node or creates a
    duplicate gets as target the node holding that thing (the node its facts went to = what makes the answers right);
    right decisions lightly reinforced. Real lead sentences (WikiText-103 species); 45 train / 30 test species.
    Baselines: exact name string; name-only cosine (frozen GPT-2 of the name alone)"""
    import re as _re
    tk, M = lm()
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bio")
    arts = json.load(open(os.path.join(here, "species_e1.json"), encoding="utf-8"))
    if os.path.exists(os.path.join(here, "wiki_e1.json.gz")):   # (11-16 curve: 15/30/45 species -> 19/42/52: data-limited)
        arts = arts + json.load(gzip.open(os.path.join(here, "wiki_e1.json.gz"), "rt", encoding="utf-8"))
    FAM = r"(?:family ([A-Z][a-z]+(?:idae|aceae))|([A-Z][a-z]+(?:idae|aceae)) family)"
    sp, seen_b = [], set()
    for a in arts:                                       # eval construction (which names belong to one species)
        if not a["text"]:
            continue
        p0 = a["text"][0]
        s0 = p0.split(" . ")[0] + " ."
        fm = _re.search(FAM, p0)
        mA = _re.match(r"^The ([a-z][a-z @\-@]+?) (?:or [^(]+ )?\( ([A-Z][a-z]+ [a-z]+)(?: [a-z]+)? [),;]", s0)
        mB = _re.match(r"^([A-Z][a-z]+ [a-z]+) , (?:commonly known as|common name|also known as|known as) (?:the )?([a-z][a-z @\-@]+?) (?:,|or)", s0)
        if not fm or not (mA or mB) or not _re.search(FAM, s0):   # (v8: facts read from the lead sentence itself;
            continue                                             #  23 leads state the family only in sentence 2 "It is ...")
        c, b = (mA[1], mA[2]) if mA else (mB[2], mB[1])
        if "<unk>" in c + b or b in seen_b:
            continue
        seen_b.add(b)
        sp.append({"common": c, "binomial": b, "family": fm[1] or fm[2], "genus": b.split()[0], "lead": s0})
    # T9 (11-17, Claude's proposal; P2 = chains): V11_T5_SPLIT = share of species whose family is NOT in their own
    # statement: they read the real lead with its family clause removed + "<Genus> is a genus in the family <F> ."
    # (template from the real fact; only 5/139 genera have a real article). Then "which family is X in?" needs
    # X -> genus -> family (2 hops); the rest keep the full lead (1 hop)
    RM = r" ,? ?(?:with)?(?:in|of|to|within) the (?:[a-z]+ )?family [A-Z][a-z]+(?:idae|aceae)| ,? ?(?:in|of) the [A-Z][a-z]+(?:idae|aceae) family"
    SPLIT = float(os.environ.get("V11_T5_SPLIT", "0"))
    srng = random.Random(61)
    for x in sp:
        nf = _re.sub(RM, "", x["lead"])
        clean = not _re.search(r"(?:idae|aceae)\b", nf)
        x["split"] = clean and srng.random() < SPLIT
        x["lead_nf"], x["gstmt"] = nf, f"{x['genus']} is a genus in the family {x['family']} ."
        # T9b (11-17 trace: the genus as a sub-span INSIDE the binomial merged / swapped with it -- an artifact of my
        # parse, a real reader sees no separate genus mention there): the genus link is its own statement
        x["bstmt"] = f"{x['binomial']} belongs to the genus {x['genus']} ."
    rng = random.Random(41)
    rng.shuffle(sp)
    test_sp, train_sp = sp[:30], sp[30:]
    print(f"v11 T5 data: {len(sp)} species from leads (species articles + Wikipedia), {len(train_sp)} train, {len(test_sp)} test", flush=True)
    _g = [x["genus"] for x in test_sp]
    print(f"v11 T5 data: test species sharing a genus with another test species: {sum(_g.count(g) > 1 for g in _g)}/30", flush=True)
    rels = ("family", "genus")
    W = {"family": ["Which family is {n} in?", "What family does {n} belong to?", "{N} is placed in which family?", "Name the family of {n}."],
         "genus": ["Which genus is {n} in?", "What genus does {n} belong to?", "{N} is classified in which genus?", "Name the genus of {n}."]}

    def span(ids, name, last=False):
        for cand in (" " + name, " " + name[0].upper() + name[1:]):
            nid = tk(cand)["input_ids"]
            hits = [i for i in range(len(ids)) if ids[i:i + len(nid)] == nid]
            st = (hits[-1] if last else hits[0]) if hits else None
            if st is not None:
                return st, st + len(nid)
        return None

    @torch.no_grad()
    def ctx_states(ids):
        return M(input_ids=torch.tensor([ids], device=dev), output_hidden_states=True).hidden_states[-1][0].float()

    @torch.no_grad()
    def alone_states(name):
        return ctx_states(tk(" " + name)["input_ids"])
    # mentions: (pooled contextual states, name string, name-alone pooled states, the name's token states, the token
    # states of its whole sentence / question)
    cache, scale = {}, {}

    def mention(text, name, last=False):
        key = (text, name, last)
        if key not in cache:
            ids = tk(" " + text)["input_ids"]
            sp_ = span(ids, name, last)
            if sp_ is None:
                return None
            H = ctx_states(ids)
            cache[key] = (H[sp_[0]:sp_[1]].mean(0), name, alone_states(name).mean(0), H[sp_[0]:sp_[1]], H)
        v, n_, a_, s_, c_ = cache[key]
        if scale:                                        # (11-16 T5 v1 used raw states: a few huge dims)
            v, s_, c_ = [(t_ - scale["mu"]) / scale["sd"] for t_ in (v, s_, c_)]
        return v, n_, a_, s_, c_

    def events(xs, r_, ask_before=.5, n_after=2, wordings=None):
        """a stream: per species maybe 1-2 questions first, then its lead read, then questions; species interleaved"""
        per = []
        for x in xs:
            ev = []
            qs = [(rel, kind, t) for rel in rels for kind in ("common", "binomial") for t in (wordings or W[rel])]
            if r_.random() < ask_before:
                for rel, kind, t in r_.sample(qs, 2):
                    ev.append(("ask", x, rel, kind, t, variant(x, kind, r_)))
            rd = ["nf", "b", "g"] if x.get("split") else ["lead", "b"]
            ev.extend(("read", x, w) for w in r_.sample(rd, len(rd)))
            for rel, kind, t in r_.sample(qs, n_after):
                ev.append(("ask", x, rel, kind, t, variant(x, kind, r_)))
            per.append(ev)
        out = []
        while any(per):
            e = r_.choice([e for e in per if e])
            out.append(e.pop(0))
        return out

    # T6 (11-17, Claude's proposal): with exact names + edges the T5 test is solved by fixed code (120/120) -- every
    # question repeats the lead's exact spelling and no two species share a name. People refer to things by variants:
    # V11_T5_VAR = share of questions asking by a natural variant of the name (abbreviated genus, plural, title case,
    # hyphen as space). Same for train and test species; the variant forms are never names the lead used
    VAR = float(os.environ.get("V11_T5_VAR", "0"))
    TRACE = os.environ.get("V11_T5_TRACE", "0") == "1"   # (T9 diagnosis: dump failing 2-hop questions' subgraphs)

    def variant(x, kind, r_):
        if VAR <= 0 or r_.random() >= VAR:
            return None
        if kind == "binomial":
            g, e = x["binomial"].split()[:2]
            return g[0] + ". " + e
        c = x["common"]
        opts = [c + ("es" if c.endswith(("s", "sh", "ch", "x")) else "s"),
                " ".join(w if w == "@-@" else w[0].upper() + w[1:] for w in c.split())]
        if " @-@ " in c:
            opts.append(c.replace(" @-@ ", " "))
        return opts[r_.randrange(len(opts))]

    def qtext(x, rel, kind, t, var=None):
        nm = var or x[kind]
        n = ("the " + nm) if kind == "common" else nm
        return t.replace("{N}", n[0].upper() + n[1:]).replace("{n}", n), nm

    class Rel(nn.Module):
        def __init__(s_):
            super().__init__()
            s_.q = nn.Linear(M.config.n_embd, 256)       # ONE projection for both sides (11-16 T5 v2: separate q / k
            s_.create = nn.Parameter(torch.tensor(0.))   #  -> same name not similar at start; train 115 vs test 31/120)
            s_.cq = nn.Linear(M.config.n_embd, 1)
            # (11-17 link diag: on new species the binomial's pooled name never resembled its common name's (best node
            # 3-7/29) -- the pairing is stated in the CONTEXT ("The X ( Y )"), which pooled names drop. User: "transform
            # both and find relevance", "cosine or attention". So relevance also attends each side's name tokens over
            # the other side's context tokens; the model weighs the two terms)
            s_.wn = nn.Parameter(torch.tensor(1.))
            s_.wc = nn.Parameter(torch.tensor(1.))
            # (11-17 v7, user design said many times: an edge between nodes carries its OWN relation vector, learned from
            # the statement, decodable to text later. Edges are prefilled between nodes named in one statement (user
            # 11-01: "prefilled ... as relation strength weak off, you cut it"); the model learns per question whether
            # to follow an edge: score(question, edge vector) vs the node's own value vs nothing)
            E = M.config.n_embd
            s_.er = nn.Linear(3 * E, 256)                  # edge vector from (name a, name b, statement)
            s_.af = nn.Sequential(nn.Linear(E + 256, 128), nn.ReLU(), nn.Linear(128, 1))   # follow an edge
            s_.ast = nn.Sequential(nn.Linear(2 * E, 128), nn.ReLU(), nn.Linear(128, 1))    # answer with this node
            s_.anone = nn.Parameter(torch.tensor(0.))                                       # don't know

        def edge_vec(s_, ma, mb):
            return s_.er(torch.cat([ma[0], mb[0], ma[4].mean(0)]))

        def step(s_, qmen, cands):
            """(v8) one walk step: logits over cands ("none",) | ("stop", node mention) | ("edge", ma, mb)"""
            qv = qmen[4].mean(0)
            out = []
            for c in cands:
                if c[0] == "none":
                    out.append(s_.anone)
                elif c[0] == "stop":
                    out.append(s_.ast(torch.cat([qv, c[1]])).squeeze())
                else:
                    out.append(s_.af(torch.cat([qv, s_.edge_vec(c[1], c[2])])).squeeze())
            return torch.stack(out)

        def per_mention(s_, men, n_, upto=None):
            """relevance of a mention to each stored mention of node n_ (the first `upto`)"""
            P = lambda t: F.normalize(s_.q(t), dim=-1)
            k = len(n_["m"]) if upto is None else upto
            name = P(torch.stack(n_["m"][:k])) @ P(men[0])                     # pooled names, (k,)
            pad = torch.nn.utils.rnn.pad_sequence
            cx, sp = n_["cx"][:k], n_["sp"][:k]
            CX, SP = P(pad(cx, batch_first=True)), P(pad(sp, batch_first=True))  # (k, L, d)
            mcx = torch.arange(CX.shape[1], device=dev)[None] < torch.tensor([len(c) for c in cx], device=dev)[:, None]
            msp = torch.arange(SP.shape[1], device=dev)[None] < torch.tensor([len(s) for s in sp], device=dev)[:, None]
            a_sp, a_cx = P(men[3]), P(men[4])
            # my name tokens, each at its best token of their context; mean over my name
            s1 = torch.einsum("ad,kld->akl", a_sp, CX).masked_fill(~mcx[None], -2.).amax(2).mean(0)
            # their name tokens, each at its best token of my context; mean over their name
            s2 = (torch.einsum("ksd,cd->ksc", SP, a_cx).amax(2) * msp).sum(1) / msp.sum(1)
            return s_.wn * name + s_.wc * torch.maximum(s1, s2)

        def forward(s_, men, nodes):
            """logits over [nodes..., create]; a node's score = its most relevant stored mention"""
            flat = {f: [v for n_ in nodes for v in n_[f]] for f in ("m", "cx", "sp")}     # all stored mentions at once
            own = torch.tensor([i for i, n_ in enumerate(nodes) for _ in n_["m"]], device=dev)
            sc = s_.per_mention(men, flat) / .1
            node_sc = torch.full((len(nodes),), -1e4, device=dev).scatter_reduce(0, own, sc, "amax")
            return torch.cat([node_sc, (s_.create + s_.cq(men[0]).squeeze())[None]])

    def run_episode(stream, decide, mode="labels", match=None, walk=None, H=3):
        """v8 (user design: an edge has its own vector showing the relation). decide(mention, nodes) -> (index, logits);
        index len(nodes) = create. Reading a lead: every name it states (common, binomial, family, genus) points or
        creates; every ordered pair of the named nodes gets an edge whose vector is learned from (name a, name b,
        sentence). No fact table. A question points its name, then WALKS: each step = answer with this node / follow
        one of its edges / don't know (walk(...) -> (index, logits)); the answer is the node's own (first) name, checked
        against the truth (outcome). Signal: the shortest path (<= H edges) from the question's node to a node named
        with the truth -> step targets (teacher-forced); none reachable -> 'don't know' + the pointing is blamed
        (target = an existing node from which the truth is reachable, the one the model rates most relevant, else
        create; blame up to the decision that made the wrong node / put the matched mention there). No labels"""
        nodes, decs, pairs, edges, outs = [], [], [], [], {}
        stats = {"right": 0, "asked_after": 0, "wrong_node": 0, "asked_before": 0, "linked": 0, "reads": 0,
                 "right common": 0, "n common": 0, "right binomial": 0, "n binomial": 0, "edges": 0,
                 "n variant": 0, "right variant": 0, "dont know": 0, "hops1": 0, "hops2+": 0, "n 2hop": 0, "right 2hop": 0}

        def point(men, ent):
            j, lg = decide(men, nodes)
            d = {"lg": lg, "n": len(nodes), "j": j, "name": men[1]}
            decs.append(d)
            if mode == "labels" and lg is not None:
                own = next((i for i, n_ in enumerate(nodes) if n_["owner"] == ent), None)
                tgt = own if own is not None else len(nodes)
                pairs.append((lg, tgt, j != tgt))
            if j == len(nodes):
                nodes.append({"m": [men[0]], "names": [men[1]], "alone": [men[2]], "owner": ent, "made_by": len(decs) - 1,
                              "md": [d], "sp": [men[3]], "cx": [men[4]]})
            else:
                for f, v in (("m", men[0]), ("names", men[1]), ("alone", men[2]), ("md", d), ("sp", men[3]), ("cx", men[4])):
                    nodes[j][f].append(v)
            return j, d

        def path_to(start, truth):
            """shortest edge path (list of edge indices, <= H) from start to a node whose own name is the truth"""
            if nodes[start]["names"][0] == truth:
                return []
            front, seen = [(start, [])], {start}
            for _ in range(H):
                nxt = []
                for u, pth in front:
                    for ei in outs.get(u, []):
                        v = edges[ei]["b"]
                        if v in seen:
                            continue
                        if nodes[v]["names"][0] == truth:
                            return pth + [ei]
                        seen.add(v)
                        nxt.append((v, pth + [ei]))
                front = nxt
            return None

        def cands_at(u, h):
            c = [("none",), ("stop", nodes[u]["m"][0])]
            meta, labs = [None, "stop"], [None, None]
            if h < H:
                for ei in outs.get(u, []):
                    c.append(("edge", edges[ei]["ma"], edges[ei]["mb"]))
                    meta.append(ei)
                    labs.append(edges[ei]["lab"])            # (gold pair labels: used ONLY by the fixed reference walk)
            return c, meta, labs
        read_done, n_read, n_stmt, gtrace = set(), {}, [0], {}
        for ev in stream:
            x = ev[1]
            ent = x["binomial"]
            if ev[0] == "read":
                which = ev[2]
                text, kinds = {"lead": (x["lead"], ("common", "binomial", "family")),
                               "nf": (x["lead_nf"], ("common", "binomial")),
                               "b": (x["bstmt"], ("binomial", "genus")),
                               "g": (x["gstmt"], ("genus", "family"))}[which]
                named = []
                for kind in kinds:
                    m_ = mention(text, x[kind], last=(which == "b" and kind == "genus"))   # (the genus written after "genus")
                    if m_ is None:
                        continue
                    j_, d_ = point(m_, ent if kind in ("common", "binomial") else x[kind])
                    named.append((kind, j_, m_, d_))
                    if kind == "genus":                  # (T9 trace, log only)
                        gtrace.setdefault(x["genus"], []).append((which, "created" if d_["j"] == d_["n"] else "pointed", j_,
                                                                 nodes[j_]["names"][0]))
                for k1, a_, ma, da in named:             # one statement names these nodes -> an edge per ordered pair
                    for k2, b_, mb, _ in named:
                        if a_ != b_:
                            outs.setdefault(a_, []).append(len(edges))
                            edges.append({"a": a_, "b": b_, "ma": ma, "mb": mb, "lab": (k1, k2), "stmt": n_stmt[0],
                                          "da": da})
                            stats["edges"] += 1
                n_stmt[0] += 1
                if which in ("lead", "nf"):
                    jj = {k: j_ for k, j_, _, _ in named}
                    stats["reads"] += 1
                    stats["linked"] += jj.get("common", -1) == jj.get("binomial", -2)
                n_read[ent] = n_read.get(ent, 0) + 1
                if n_read[ent] == (3 if x.get("split") else 2):
                    read_done.add(ent)
                continue
            _, rel, kind, t, var = ev[1:]
            q, nm = qtext(x, rel, kind, t, var)
            men = mention(q, nm)
            if men is None:
                continue
            j, d = point(men, ent)
            truth = x[rel]
            # the walk
            cur, ans_node, taken, tedges, cats = j, None, [], [], []
            for h in range(H + 1):
                c, meta, labs = cands_at(cur, h)
                k, lg = walk(men, c, labs, rel) if walk is not None else (0, None)
                taken.append((lg, k))
                cats.append("dk" if meta[k] is None else "ans" if meta[k] == "stop" else "->" + labs[k][1])   # (log only)
                if meta[k] is None:
                    break
                if meta[k] == "stop":
                    ans_node = cur
                    break
                tedges.append(meta[k])
                cur = edges[meta[k]]["b"]
            got = nodes[ans_node]["names"][0] if ans_node is not None else None
            hops = sum(1 for _, k in taken if k >= 2)
            if ent not in read_done:
                stats["asked_before"] += 1
                stats["wrong_node"] += nodes[j]["owner"] != ent
                if got is None or got == truth:          # 'nothing known' before reading is not confusion
                    continue
            else:
                stats["asked_after"] += 1
                stats["right"] += got == truth
                stats["n " + kind] += 1
                stats["right " + kind] += got == truth
                stats["n variant"] += var is not None
                stats["right variant"] += var is not None and got == truth
                stats["dont know"] += got is None
                if x.get("split") and rel == "family":          # (T9) needs 2 hops: X -> genus -> family
                    stats["n 2hop"] += 1
                    stats["right 2hop"] += got == truth
                    # (11-17 T9 diag, log only, reads the gold edge labels: what the walk chose at step 1 and 2)
                    stats.setdefault("walk2", {})
                    key = " ".join(cats[:2])
                    stats["walk2"][key] = stats["walk2"].get(key, 0) + 1
                    stats["reach2"] = stats.get("reach2", 0) + (path_to(j, truth) is not None)   # chain exists at all?
                    stats["genus split"] = stats.get("genus split", 0) + (sum(n_["names"][0] == x["genus"] for n_ in nodes) > 1)
                    if got != truth and TRACE and len(stats.setdefault("traces", [])) < 4:    # (T9 trace, log only)
                        near = []
                        for ei in outs.get(j, []):
                            b1 = edges[ei]["b"]
                            near.append(f"{edges[ei]['lab']}->{nodes[b1]['names'][0]}[" + ",".join(
                                f"{edges[e2]['lab'][1]}:{nodes[edges[e2]['b']]['names'][0]}" for e2 in outs.get(b1, [])[:6]) + "]")
                        stats["traces"].append(f"Q '{q}' -> node {j} '{nodes[j]['names'][0]}' (names {nodes[j]['names'][:4]}) "
                                               f"walk {cats} got {got!r} truth {truth!r} | out: {near[:6]} | genus "
                                               f"'{x['genus']}' mentions: {gtrace.get(x['genus'])} | nodes named it: "
                                               f"{[i for i, n_ in enumerate(nodes) if n_['names'][0] == x['genus']]}")
                if got == truth:
                    stats["hops1" if hops == 1 else "hops2+"] += hops >= 1
            if walk is None or taken[0][0] is None:
                continue
            pth = path_to(j, truth)
            if got == truth:                             # right: lightly reinforce the steps taken
                pairs.extend((lg, k, False) for lg, k in taken)
            elif pth is not None:                        # reachable: teacher-force the shortest path (the WALK erred)
                # blame up (2') (11-17 v8 s0: before->other 25/32, conflated nodes shielded by 'reachable'): the walk's
                # first edge and the right path's first edge come from DIFFERENT statements -> two statements were
                # filed on one node; the later statement's pointing should have created (blame (2) for edges)
                if mode == "outcome" and tedges and pth and edges[tedges[0]]["stmt"] != edges[pth[0]]["stmt"]:
                    dl = max(edges[tedges[0]], edges[pth[0]], key=lambda e: e["stmt"])["da"]
                    if dl["lg"] is not None and dl["j"] != dl["n"]:
                        pairs.append((dl["lg"], dl["n"], True))
                u = j
                for h, ei in enumerate(pth + [None]):
                    c, meta, labs = cands_at(u, h)
                    lg_, _ = walk(men, c, labs, rel, force=True)
                    pairs.append((lg_, meta.index(ei) if ei is not None else 1, True))
                    if ei is not None:
                        u = edges[ei]["b"]
            else:                                        # nothing reachable: 'don't know' at the start
                pairs.append((taken[0][0], 0, taken[0][1] != 0))
                # blame up (6) (11-17 T9 diag; reverted once when the genus sub-span artifact swamped it, re-tried on T9b: the genus statement's genus mostly CREATED a 2nd genus node -> chain cut,
                # 'don't know' targets fought 'take the genus edge'): the answer's node exists but the question cannot
                # reach it. U = nodes the question's node reaches (< H hops), V = nodes one edge from the answer; the
                # most relevant (u, v) pair by the model's OWN relevance: if v was created while u existed, v's creating
                # decision should have pointed to u (outcome + own relevance; no labels, no string match)
                ans_nodes = {i for i, n_ in enumerate(nodes) if n_["names"][0] == truth}
                if ent in read_done and match is not None and ans_nodes:
                    U, front = {j}, [j]
                    for _ in range(H - 1):
                        front = [edges[ei]["b"] for u_ in front for ei in outs.get(u_, []) if edges[ei]["b"] not in U]
                        U.update(front)
                    V = {e["a"] for e in edges if e["b"] in ans_nodes} - U - ans_nodes
                    best = None
                    for v in V:
                        nv = nodes[v]
                        vm = (nv["m"][0], nv["names"][0], nv["alone"][0], nv["sp"][0], nv["cx"][0])
                        dv = decs[nv["made_by"]]
                        if dv["lg"] is None or dv["j"] != dv["n"]:
                            continue
                        Us = [u_ for u_ in U if u_ < dv["n"]]
                        if not Us:
                            continue
                        flat = {f: [v_ for u_ in Us for v_ in nodes[u_][f]] for f in ("m", "cx", "sp")}
                        own = [u_ for u_ in Us for _ in nodes[u_]["m"]]
                        sc_ = match(vm, flat, len(flat["m"]))     # one batched call over all of U's mentions
                        k6 = int(sc_.argmax())
                        if best is None or float(sc_[k6]) > best[0]:
                            best = (float(sc_[k6]), dv, own[k6])
                    if best is not None:
                        pairs.append((best[1]["lg"], best[2], True))
            if mode != "outcome" or d["lg"] is None:
                continue
            if got == truth or pth is not None:          # the pointing reached the answer: lightly reinforce it
                pairs.append((d["lg"], j, False))
                continue
            # blame up (5): no node is named with the answer at all -> the answer's word WAS read but a pointing
            # decision put it into another node; that decision should have created one (answer text vs read text)
            if not any(n_["names"][0] == truth for n_ in nodes):
                for dd in decs:
                    if dd["name"] == truth and dd["lg"] is not None and dd["j"] != dd["n"]:
                        pairs.append((dd["lg"], dd["n"], True))
                continue
            # pointing blamed: target = a node (existing then) from which ALL the thing's values are reachable (one
            # value alone = any same-family thing), else create
            ok = [i for i in range(d["n"]) if all(path_to(i, x[r_]) is not None for r_ in rels)]
            t_ = max(ok, key=lambda i: float(d["lg"][i])) if ok else d["n"]
            pairs.append((d["lg"], t_, True))
            if ok and j != t_ and match is not None and len(nodes[j]["m"]) > 1:      # blame up (3)
                k3 = int(match(men, nodes[j], len(nodes[j]["m"]) - 1).argmax())
                dk = nodes[j]["md"][k3]
                if dk is not d and dk["lg"] is not None and dk["j"] != dk["n"]:
                    pairs.append((dk["lg"], t_ if t_ < dk["n"] else dk["n"], True))
            if ok and j != t_:                           # blame up (1): the decision that made the wrong node
                dm = decs[nodes[j]["made_by"]]
                if dm is not d and dm["lg"] is not None and t_ < dm["n"]:
                    pairs.append((dm["lg"], t_, True))
        owners = {n_["owner"] for n_ in nodes}
        stats["nodes"], stats["things"] = len(nodes), len(owners)
        return stats, pairs

    def make_learned(relm):
        def learned(train):
            def d(men, nodes):
                lg = relm(men, nodes) if nodes else relm.create[None] + relm.cq(men[0])
                d.last = lg.detach()                     # (diagnosis only: read by run_episode's link stats)
                j = random.randrange(len(lg)) if train and random.random() < .1 else int(lg.argmax())
                return j, (lg if train else None)
            return d
        return learned

    def make_walk(relm, train):
        def w_(men, c, labs, rel, force=False):
            lg = relm.step(men, c)
            if force:
                return lg, None
            k = random.randrange(len(lg)) if train and random.random() < .1 else int(lg.argmax())
            return k, (lg if train else None)
        return w_

    def gold_walk(men, c, labs, rel, force=False):
        """fixed reference, NOT learned: reads the edges' gold pair labels. Edge to the asked relation's node -> take
        it; standing on that node -> answer; else cross to the thing's other name; else don't know"""
        for i, l_ in enumerate(labs):
            if l_ is not None and l_[1] == rel:
                return i, None
        if any(l_ is not None and l_[0] == rel for l_ in labs):
            return 1, None
        for i, l_ in enumerate(labs):
            if l_ is not None and rel == "family" and l_[1] == "genus":
                return i, None
        for i, l_ in enumerate(labs):
            if l_ is not None and l_[1] in ("common", "binomial"):
                return i, None
        return 0, None

    def exact(men, nodes):
        return next((i for i, n_ in enumerate(nodes) if men[1] in n_["names"]), len(nodes)), None

    mu_alone = torch.stack([alone_states(x[k]).mean(0) for x in train_sp for k in ("common", "binomial")]).mean(0)

    def alone_cos(thr):                                  # name alone, centred by the training names' mean (U0 lesson)
        def d(men, nodes):
            if not nodes:
                return 0, None
            sc = torch.stack([F.cosine_similarity((men[2] - mu_alone)[None], torch.stack(n_["alone"]) - mu_alone).max() for n_ in nodes])
            j = int(sc.argmax())
            return (j if float(sc[j]) >= thr else len(nodes)), None
        return d
    # fixed scale of the mention vectors, from training mentions
    raw = []
    for x in train_sp:
        for nm_ in (x["common"], x["binomial"]):
            mm = mention(x["lead"], nm_)
            if mm is not None:
                raw.append(mm[0])
        for rel in rels:
            for kind in ("common", "binomial"):
                for t in W[rel]:
                    q, nm_ = qtext(x, rel, kind, t)
                    mm = mention(q, nm_)
                    if mm is not None:
                        raw.append(mm[0])
    R_ = torch.stack(raw)
    scale["mu"], scale["sd"] = R_.mean(0), R_.std(0) + 1e-3
    erng = random.Random(47)
    test_stream = events(test_sp, erng, ask_before=.5, n_after=4)
    train_stream = events(random.Random(53).sample(train_sp, 30), random.Random(59), ask_before=.5, n_after=4)

    # ================================================================================ v12 (docs/GNM_v10.md §1c, 11-17)
    # user 11-17: no LLM translator; "combine the 2 ideas together and fix the perception problem"; reuse GNM.
    # C1/C2: one V11 memory -- an edge IS remember(op = learned relation vector r, args = [a]) -> b (bind with a position
    # permutation; reverse direction = the other permutation). C3: identity = Rel point / create (T5). C4: answering =
    # walk; its last hop is a V11 POINT with the question's relation vector at the current node. C5: perception = a
    # learned READER that points at claims (thing A, relation words, thing B) in a sentence / question, no regex at run
    # time. C6: ~50 demo sentences + ~50 demo questions (claims written by the old parse = demos and grading only), then
    # outcome; the reader also trains on its own right traces (D7)
    if os.environ.get("V12", "0") == "1":
        D12, BETA = 1024, 30.
        g12 = torch.Generator().manual_seed(7)
        P0 = torch.randperm(D12, generator=g12).to(dev)
        P1 = torch.randperm(D12, generator=g12).to(dev)
        E_ = M.config.n_embd
        Hc = {}

        def sent(text):
            if text not in Hc:
                ids_ = tk(" " + text)["input_ids"]
                Hc[text] = (ids_, ctx_states(ids_))
            ids_, H_ = Hc[text]
            return ids_, (H_ - scale["mu"]) / scale["sd"]

        def men_at(text, sp_):                           # a mention from a span the READER chose (inclusive)
            ids_, H_ = sent(text)
            v_ = H_[sp_[0]:sp_[1] + 1]
            return (v_.mean(0), tk.decode(ids_[sp_[0]:sp_[1] + 1]).strip(), None, v_, H_)

        STMT = {"lead": [("common", "binomial"), ("common", "family")], "nf": [("common", "binomial")],
                "b": [("binomial", "genus")], "g": [("genus", "family")]}

        def text_of(x, which):
            return {"lead": x["lead"], "nf": x["lead_nf"], "b": x["bstmt"], "g": x["gstmt"]}[which]

        def gold_claims(x, which):
            """demos / grading only: (thing A, relation words = last <=4 tokens between the names, thing B), inclusive"""
            ids_, _ = sent(text_of(x, which))
            out = []
            for ka, kb in STMT[which]:
                a_ = span(ids_, x[ka], last=(which == "b" and ka == "genus"))
                b_ = span(ids_, x[kb], last=(which == "b" and kb == "genus"))
                if a_ is None or b_ is None:
                    return None
                lo, hi = (a_[1], b_[0]) if a_[1] <= b_[0] else (b_[1], a_[0])
                if hi - lo < 1:
                    return None
                out.append(((a_[0], a_[1] - 1), (max(lo, hi - 4), hi - 1), (b_[0], b_[1] - 1)))
            return out

        rel_tok12 = {r_: tk(" " + r_)["input_ids"][0] for r_ in rels}

        def gold_q(q, nm, rel):
            ids_, _ = sent(q)
            a_ = span(ids_, nm)
            rw = next((i for i, t_ in enumerate(ids_) if t_ == rel_tok12[rel]), None)
            if a_ is None or rw is None:
                return None
            return [((a_[0], a_[1] - 1), (rw, rw), None)]

        class Reader(nn.Module):
            """C5: a loop over a sentence's token states; each step points at a claim (A start/end, relation start/end,
            B start/end or none) or stops; the state remembers what was already taken"""
            def __init__(s_):
                super().__init__()
                s_.cell = nn.GRUCell(3 * E_, 256)
                s_.ptr = nn.Sequential(nn.Linear(E_ + 256, 256), nn.Tanh(), nn.Linear(256, 6))
                s_.ctl = nn.Linear(E_ + 256, 2)          # [stop, B is none]

            def logits(s_, H_, h):
                return s_.ptr(torch.cat([H_, h[None].expand(len(H_), -1)], 1)), s_.ctl(torch.cat([H_.mean(0), h]))

            def advance(s_, H_, h, c):
                f = lambda sp_: H_[sp_[0]:sp_[1] + 1].mean(0) if sp_ is not None else torch.zeros(E_, device=dev)
                return s_.cell(torch.cat([f(c[0]), f(c[1]), f(c[2])])[None], h[None])[0]

            def loss(s_, H_, claims):
                """teacher-forced on the given claims, then stop"""
                h, L = torch.zeros(256, device=dev), 0.
                T_ = lambda v: torch.tensor([v], device=dev)
                for c in list(claims) + [None]:
                    P, C = s_.logits(H_, h)
                    L = L + F.binary_cross_entropy_with_logits(C[0], torch.tensor(float(c is None), device=dev))
                    if c is None:
                        break
                    for col, t_ in ((0, c[0][0]), (1, c[0][1]), (2, c[1][0]), (3, c[1][1])):
                        L = L + F.cross_entropy(P[:, col][None], T_(t_))
                    L = L + F.binary_cross_entropy_with_logits(C[1], torch.tensor(float(c[2] is None), device=dev))
                    if c[2] is not None:
                        for col, t_ in ((4, c[2][0]), (5, c[2][1])):
                            L = L + F.cross_entropy(P[:, col][None], T_(t_))
                    h = s_.advance(H_, h, c)
                return L

            @torch.no_grad()
            def read(s_, H_, kmax=4):
                h, out = torch.zeros(256, device=dev), []
                for _ in range(kmax):
                    P, C = s_.logits(H_, h)
                    if C[0] > 0:
                        break

                    def sp(c0, c1):
                        st_ = int(P[:, c0].argmax())
                        return st_, st_ + int(P[st_:st_ + 8, c1].argmax())
                    c = (sp(0, 1), sp(2, 3), None if C[1] > 0 else sp(4, 5))
                    out.append(c)
                    h = s_.advance(H_, h, c)
                return out

        class Mem12(nn.Module):
            """C1/C2/C4: relation vectors (learned from the relation words' states) used as V11 ops"""
            def __init__(s_):
                super().__init__()
                s_.rv = nn.Linear(E_, D12)
                s_.af = nn.Sequential(nn.Linear(2 * D12 + 1, 128), nn.ReLU(), nn.Linear(128, 1))   # follow an edge
                s_.pa = nn.Parameter(torch.tensor(5.))  # answer by POINT: pa * best key cosine + pb
                s_.pb = nn.Parameter(torch.tensor(-2.))
                s_.anone = nn.Parameter(torch.tensor(0.))

            def r(s_, rstate):
                return F.normalize(s_.rv(rstate), dim=-1)

        def key_of(rvec, atom, dr):                      # V11 bind: op (relation) with the argument at its position
            return F.normalize(V11.bind(rvec, atom[P0 if dr == 0 else P1]), dim=0)

        def run12(stream, decide, mem, reader, mode="outcome", match=None, train=False, gold_read=False, H=3):
            nodes, decs, pairs, edges, outs = [], [], [], [], {}
            rdr_traces = []                              # (H, claims, ent) of statements read (reader self-training)
            stats = {"right": 0, "asked_after": 0, "wrong_node": 0, "asked_before": 0, "right common": 0, "n common": 0,
                     "right binomial": 0, "n binomial": 0, "n variant": 0, "right variant": 0, "dont know": 0,
                     "hops0": 0, "hops1+": 0, "n 2hop": 0, "right 2hop": 0, "claims gold": 0, "claims read right": 0,
                     "q gold": 0, "q read right": 0, "nodes": 0, "edges": 0, "reads": 0, "linked": 0}
            q_right_by_ent, q_n_by_ent = {}, {}

            def point(men, ent):
                j, lg = decide(men, nodes)
                d = {"lg": lg, "n": len(nodes), "j": j, "name": men[1]}
                decs.append(d)
                if j == len(nodes):
                    nodes.append({"m": [men[0]], "names": [men[1]], "alone": [None], "owner": ent, "made_by": len(decs) - 1,
                                  "md": [d], "sp": [men[3]], "cx": [men[4]],
                                  "atom": F.normalize(torch.randn(D12, generator=g12), dim=0).to(dev)})
                else:
                    for f, v in (("m", men[0]), ("names", men[1]), ("alone", None), ("md", d), ("sp", men[3]), ("cx", men[4])):
                        nodes[j][f].append(v)
                return j, d

            def path12(start, truth):
                """shortest edge path whose LAST edge is forward into a node named with the truth (the last hop is a
                V11 point); -> list of edge indices or None"""
                front, seen = [(start, [])], {start}
                for _ in range(H):
                    nxt = []
                    for u, pth in front:
                        for ei in outs.get(u, []):
                            e = edges[ei]
                            if e["dir"] == 0 and nodes[e["b"]]["names"][0] == truth:
                                return pth + [ei]
                            if e["b"] not in seen:
                                seen.add(e["b"])
                                nxt.append((e["b"], pth + [ei]))
                    front = nxt
                return None

            def keys_now():
                if not edges:
                    return None, None
                R_ = mem.r(torch.stack([e["rs"] for e in edges]))
                K = torch.stack([key_of(R_[i], nodes[e["a"]]["atom"], e["dir"]) for i, e in enumerate(edges)])
                V = torch.stack([nodes[e["b"]]["atom"] for e in edges])
                return R_, (K, V)

            def step_logits(rq, u, h, R_, KV):
                """options: 0 = don't know, 1 = POINT with the question's relation here, 2.. = follow an edge of u"""
                A = torch.stack([n_["atom"] for n_ in nodes])
                q = key_of(rq, nodes[u]["atom"], 0)
                sc = KV[0] @ q if KV is not None else torch.zeros(1, device=dev)
                read_v = torch.softmax(BETA * sc, 0) @ KV[1] if KV is not None else torch.zeros(D12, device=dev)
                clean = BETA * (A @ F.normalize(read_v, dim=0))          # cleanup: which node the point lands on
                opts = [mem.anone, mem.pa * sc.max() + mem.pb]
                eis = outs.get(u, []) if h < H else []
                for ei in eis:
                    opts.append(mem.af(torch.cat([rq, R_[ei], torch.tensor([float(edges[ei]["dir"])], device=dev)])).squeeze())
                return torch.stack(opts), eis, clean

            read_done, n_read, n_stmt = set(), {}, [0]
            for ev in stream:
                x = ev[1]
                ent = x["binomial"]
                if ev[0] == "read":
                    which = ev[2]
                    text = text_of(x, which)
                    ids_, H_ = sent(text)
                    gc = gold_claims(x, which)
                    claims = gc if gold_read else reader.read(H_)
                    if gc is not None:
                        stats["claims gold"] += len(gc)
                        stats["claims read right"] += sum(c in claims for c in gc)
                    if not gold_read:
                        rdr_traces.append((H_, claims, ent))
                    at = {}
                    for c in claims:
                        if c[2] is None:
                            continue
                        js = []
                        for sp_ in (c[0], c[2]):
                            if sp_ not in at:
                                m_ = men_at(text, sp_)
                                own = ent if m_[1] in (x["common"], x["binomial"]) else m_[1]
                                at[sp_] = point(m_, own)
                            js.append(at[sp_])
                        (a_, da), (b_, _) = js
                        if a_ == b_:
                            continue
                        rs = H_[c[1][0]:c[1][1] + 1].mean(0)
                        for dr, (u, v) in ((0, (a_, b_)), (1, (b_, a_))):
                            outs.setdefault(u, []).append(len(edges))
                            edges.append({"a": u, "b": v, "rs": rs, "dir": dr, "stmt": n_stmt[0], "da": da})
                        stats["edges"] += 1
                    n_stmt[0] += 1
                    n_read[ent] = n_read.get(ent, 0) + 1
                    if n_read[ent] == (3 if x.get("split") else 2):
                        read_done.add(ent)
                    continue
                _, rel, kind, t, var = ev[1:]
                q, nm = qtext(x, rel, kind, t, var)
                ids_, Hq = sent(q)
                gq = gold_q(q, nm, rel)
                qc = gq if gold_read else reader.read(Hq, kmax=1)
                if gq is not None:
                    stats["q gold"] += 1
                    stats["q read right"] += bool(qc) and qc[0][:2] == gq[0][:2]
                if not qc:
                    got, j, d, taken = None, None, None, []
                else:
                    men = men_at(q, qc[0][0])
                    j, d = point(men, ent)
                    rq = mem.r(Hq[qc[0][1][0]:qc[0][1][1] + 1].mean(0))
                    R_, KV = keys_now()
                    cur, got, taken, hops = j, None, [], 0
                    for h in range(H + 1):
                        lg, eis, clean = step_logits(rq, cur, h, R_, KV)
                        k = (random.randrange(len(lg)) if train and random.random() < .1 else int(lg.argmax()))
                        taken.append((lg, k))
                        if k == 0:
                            break
                        if k == 1:
                            got = nodes[int(clean.argmax())]["names"][0] if KV is not None else None
                            break
                        cur, hops = edges[eis[k - 2]]["b"], hops + 1
                truth = x[rel]
                if ent not in read_done:
                    stats["asked_before"] += 1
                    stats["wrong_node"] += j is not None and nodes[j]["owner"] != ent
                    if got is None or got == truth:
                        continue
                else:
                    stats["asked_after"] += 1
                    ok_ = got == truth
                    stats["right"] += ok_
                    stats["n " + kind] += 1
                    stats["right " + kind] += ok_
                    stats["n variant"] += var is not None
                    stats["right variant"] += var is not None and ok_
                    stats["dont know"] += got is None
                    if ok_:
                        stats["hops0" if hops == 0 else "hops1+"] += 1
                    if x.get("split") and rel == "family":
                        stats["n 2hop"] += 1
                        stats["right 2hop"] += ok_
                    q_n_by_ent[ent] = q_n_by_ent.get(ent, 0) + 1
                    q_right_by_ent[ent] = q_right_by_ent.get(ent, 0) + ok_
                    if train and ok_ and not gold_read and qc:   # reader: its own right question reading (D7)
                        pairs.append(("reader", (Hq, qc), .1))
                if not train or j is None:
                    continue
                pth = path12(j, truth)
                if got == truth:
                    pairs.extend((lg, k, False) for lg, k in taken)
                elif pth is not None:                    # reachable: teacher-force follow ... then POINT
                    u = j
                    for h, ei in enumerate(pth):
                        lg_, eis, clean = step_logits(rq, u, h, R_, KV)
                        if h == len(pth) - 1:
                            pairs.append((lg_, 1, True))
                            pairs.append((clean, edges[ei]["b"], True))   # the point must land on the answer node
                        else:
                            pairs.append((lg_, 2 + eis.index(ei), True))
                            u = edges[ei]["b"]
                    # blame up (2'): the walk's first edge and the right path's first edge from different statements
                    if len(taken) and taken[0][1] >= 2:
                        e_w = edges[outs.get(j, [])[taken[0][1] - 2]]
                        e_r = edges[pth[0]]
                        if e_w["stmt"] != e_r["stmt"]:
                            dl = max(e_w, e_r, key=lambda e: e["stmt"])["da"]
                            if dl["lg"] is not None and dl["j"] != dl["n"]:
                                pairs.append((dl["lg"], dl["n"], True))
                else:                                    # nothing reachable: don't know
                    pairs.append((taken[0][0], 0, taken[0][1] != 0))
                if mode != "outcome" or d["lg"] is None:
                    continue
                if got == truth or pth is not None:
                    pairs.append((d["lg"], j, False))
                    continue
                if not any(n_["names"][0] == truth for n_ in nodes):          # blame up (5)
                    for dd in decs:
                        if dd["name"] == truth and dd["lg"] is not None and dd["j"] != dd["n"]:
                            pairs.append((dd["lg"], dd["n"], True))
                    continue
                ok = [i for i in range(d["n"]) if all(path12(i, x[r_]) is not None for r_ in rels)]
                t_ = max(ok, key=lambda i: float(d["lg"][i])) if ok else d["n"]
                pairs.append((d["lg"], t_, True))
                if ok and j != t_ and match is not None and len(nodes[j]["m"]) > 1:      # blame up (3)
                    k3 = int(match(men, nodes[j], len(nodes[j]["m"]) - 1).argmax())
                    dk = nodes[j]["md"][k3]
                    if dk is not d and dk["lg"] is not None and dk["j"] != dk["n"]:
                        pairs.append((dk["lg"], t_ if t_ < dk["n"] else dk["n"], True))
                if ok and j != t_:                       # blame up (1)
                    dm = decs[nodes[j]["made_by"]]
                    if dm is not d and dm["lg"] is not None and t_ < dm["n"]:
                        pairs.append((dm["lg"], t_, True))
            if train:                                    # reader: its own statement readings for species answered right
                for H_, claims, ent in rdr_traces:
                    if q_n_by_ent.get(ent) and q_right_by_ent.get(ent) == q_n_by_ent[ent] and claims:
                        pairs.append(("reader", (H_, claims), .1))
            stats["nodes"], stats["reads"] = len(nodes), n_stmt[0]
            return stats, pairs

        def fmt12(st):
            pc = lambda a_, b_: f"{st[a_]}/{st[b_]}"
            return (f"right after reading {pc('right', 'asked_after')} (common {pc('right common', 'n common')}, binomial "
                    f"{pc('right binomial', 'n binomial')}, variant {pc('right variant', 'n variant')}) | family via genus "
                    f"{pc('right 2hop', 'n 2hop')} | right by a point here {st['hops0']}, after 1+ followed edges {st['hops1+']}, "
                    f"don't know {st['dont know']} | READER: statement claims exact {pc('claims read right', 'claims gold')}, "
                    f"questions exact {pc('q read right', 'q gold')} | before reading -> another thing's node "
                    f"{pc('wrong_node', 'asked_before')} | nodes {st['nodes']}, edges {st['edges']}")

        sd_ = int(os.environ.get("V11_T5_SEED", "0"))
        torch.manual_seed(sd_)
        random.seed(sd_)
        relm, mem, reader = Rel().to(dev), Mem12().to(dev), Reader().to(dev)
        # C6: ~50 demo statements + ~50 demo questions from training species (claims by the old parse = demonstration)
        drng = random.Random(71 + sd_)
        demo_s, demo_q = [], []
        for x in drng.sample(train_sp, len(train_sp)):
            if len(demo_s) >= 50:
                break
            w_ = drng.choice(["nf", "b", "g"] if x.get("split") else ["lead", "b"])
            gc = gold_claims(x, w_)
            if gc:
                demo_s.append((sent(text_of(x, w_))[1], gc))
        for x in drng.sample(train_sp, len(train_sp)):
            if len(demo_q) >= 50:
                break
            rel, kind = drng.choice(rels), drng.choice(["common", "binomial"])
            q, nm = qtext(x, rel, kind, drng.choice(W[rel]), variant(x, kind, drng))
            gq = gold_q(q, nm, rel)
            if gq:
                demo_q.append((sent(q)[1], gq))
        ropt = torch.optim.AdamW(reader.parameters(), lr=1e-3)
        demos = demo_s + demo_q
        for it in range(int(os.environ.get("V12_DEMO_STEPS", "1500"))):
            H_, cl = demos[it % len(demos)]
            L = reader.loss(H_, cl)
            ropt.zero_grad()
            L.backward()
            ropt.step()
        print(f"v12: reader taught on {len(demo_s)} demo statements + {len(demo_q)} demo questions", flush=True)
        opt = torch.optim.AdamW(list(relm.parameters()) + list(mem.parameters()) + list(reader.parameters()), lr=1e-3)
        learned = make_learned(relm)

        @torch.no_grad()
        def match_fn(men, n_, k):
            return relm.per_mention(men, n_, k)

        def rep12(tag, at_step=None):
            with torch.no_grad():
                for sn, sm in (("test", test_stream), ("train", train_stream)):
                    for gr in (False, True):
                        st, _ = run12(sm, learned(False), mem, reader, gold_read=gr)
                        lab = "gold claims (reference)" if gr else "READER claims"
                        print(f"v12 {tag} {sn}: {lab}: {fmt12(st)}", flush=True)
        rep12("step 0")
        trng = random.Random(43 + sd_)
        at = {int(v) for v in os.environ.get("V11_T5_AT", "").split(",") if v}
        for stp in range(steps):
            if stp in at and stp:
                rep12(f"step {stp}")
            stream = events(trng.sample(train_sp, min(len(train_sp), trng.randint(ep_species, 30))), trng)
            _, pairs = run12(stream, learned(True), mem, reader, match=match_fn, train=True)
            if not pairs:
                continue
            L, n_ = 0., 0
            for p_ in pairs:
                if p_[0] == "reader":
                    L = L + p_[2] * reader.loss(*p_[1])
                else:
                    lg, tgt, wrong = p_
                    L = L + (1. if wrong else .1) * F.cross_entropy(lg[None], torch.tensor([tgt], device=dev))
                n_ += 1
            opt.zero_grad()
            (L / n_).backward()
            opt.step()
        rep12("final")
        return


    def fmt(st):
        return (f"right after reading {st['right']}/{st['asked_after']} (by common name {st['right common']}/{st['n common']}, "
                f"by binomial {st['right binomial']}/{st['n binomial']}, by a name variant {st['right variant']}/{st['n variant']}) "
                f"| right in 1 hop {st['hops1']}, 2+ hops {st['hops2+']}, don't know {st['dont know']} | family via genus (needs 2 "
                f"hops) {st['right 2hop']}/{st['n 2hop']} | lead's two names on "
                f"one node {st['linked']}/{st['reads']} | asked before reading -> another thing's node {st['wrong_node']}/"
                f"{st['asked_before']} | nodes {st['nodes']} for {st['things']} things, edges {st['edges']}"
                + (f" | 2-hop walks: {dict(sorted(st['walk2'].items(), key=lambda kv: -kv[1]))}, family reachable "
                   f"{st['reach2']}, genus on 2+ nodes {st['genus split']}" if st.get("walk2") else ""))

    def report(name, d, w):
        for split, stream in (("test (30 never-trained species)", test_stream), ("train (30 training species)", train_stream)):
            st, _ = run_episode(stream, d, walk=w)
            print(f"v11 T8 {split}: {name}: {fmt(st)}", flush=True)
            for tr in st.get("traces", []):
                print(f"   TRACE {split[:5]}: {tr}", flush=True)
    with torch.no_grad():
        report("exact name + gold-labelled walk (fixed reference)", exact, gold_walk)
        report("name-only cosine (centred) >= .8 + gold-labelled walk (fixed)", alone_cos(.8), gold_walk)
    # V11_T5_CURVE (11-16, measurement): outcome-only trained on the first N training species (data limit or not?)
    curve = [int(v) for v in os.environ.get("V11_T5_CURVE", "").split(",") if v]
    runs_ = [("outcome", n_) for n_ in curve] if curve else [("labels", len(train_sp)), ("outcome", len(train_sp))]
    for mode, n_tr in runs_:
        sd_ = int(os.environ.get("V11_T5_SEED", "0"))     # (11-17: replicate v5 on other seeds)
        torch.manual_seed(sd_)
        random.seed(sd_)
        relm = Rel().to(dev)
        opt = torch.optim.AdamW(relm.parameters(), lr=1e-3)
        learned = make_learned(relm)

        @torch.no_grad()
        def match_fn(men, n_, k):                        # the model's own relevance (blame up (3)); no labels
            return relm.per_mention(men, n_, k)
        trng = random.Random(43 + sd_)
        n_sig, run_l = 0, []
        # (11-17 diag, measurement only: train 104/120 vs test 44/120, linked 19/30 vs 6/30 -> does test peak early
        # and fall (memorizing) or never rise (signal does not carry to new names)?)
        at = {int(v) for v in os.environ.get("V11_T5_AT", "").split(",") if v}
        for stp in range(steps):
            if stp in at:
                with torch.no_grad():
                    for sn, sm in (("test", test_stream), ("train", train_stream)):
                        st, _ = run_episode(sm, learned(False), walk=make_walk(relm, False))
                        print(f"v11 T8 at step {stp} {sn}: {fmt(st)}", flush=True)
                        for tr in (st.get("traces", []) if sn == "test" else []):
                            print(f"   TRACE: {tr}", flush=True)
            stream = events(trng.sample(train_sp[:n_tr], min(n_tr, trng.randint(ep_species, 30))), trng)   # (8-30: test has 30)
            _, pairs = run_episode(stream, learned(True), mode, match=match_fn, walk=make_walk(relm, True))
            if not pairs:
                continue
            n_sig += sum(w for _, _, w in pairs)
            L = sum((1. if wrong else .1) * F.cross_entropy(lg[None], torch.tensor([tgt], device=dev)) for lg, tgt, wrong in pairs) / len(pairs)
            opt.zero_grad()
            L.backward()
            opt.step()
            run_l.append(float(L))
            if (stp + 1) % 1000 == 0:
                print(f"v11 T5 [{mode}] step {stp + 1}: mean loss {sum(run_l) / len(run_l):.3f}, punished decisions so far {n_sig}", flush=True)
                run_l = []
        with torch.no_grad():
            report(f"learned pointing + learned walk, trained on {mode} ({n_tr} training species)", learned(False),
                   make_walk(relm, False))


# ====================================================================== v13 (docs/GNM_v10.md §1d, 11-17): GNM checks itself
# User 11-17: "combine the 2 ideas together and fix the perception problem"; "since now GNM can do some computation of
# its own, it can reuse itself (as well as train itself)"; "sure then" (build §1d). One path:
#   text unit -> frozen GPT-2 states -> READER (a learned function: marks claims = args + relation words + value)
#   -> identity (names: learned point / create; numbers: typed exact value) -> memory (V11 entries T(r, args) -> V,
#   r = learned relation vector) -> answering = walk / recall / compute (an op = a function node with its OWN small
#   column network, made by the model when a relation meets no op label; one firing = one digit column)
#   -> CHECK 1 everywhere: a statement is predicted before it is stored; a question is the same check.
# Fixed (said in every result): GPT-2 frozen; text split into units; typed number encode / decode (digits of a
# number node = its name string); column order right to left with a final carry column; answer = a node's first name.

V13_FIXED = ("GPT-2 frozen (features); text split into units; numbers = typed exact values (digits = name string); "
             "column order right->left + final carry; answer = node's first name; reader called on every unit")


def _v13_species():
    """species from real lead sentences whose FIRST sentence names the family (eval construction + demo writing only)"""
    import re as _re
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bio")
    arts = json.load(open(os.path.join(here, "species_e1.json"), encoding="utf-8"))
    if os.path.exists(os.path.join(here, "wiki_e1.json.gz")):
        arts = arts + json.load(gzip.open(os.path.join(here, "wiki_e1.json.gz"), "rt", encoding="utf-8"))
    FAM = r"(?:family ([A-Z][a-z]+(?:idae|aceae))|([A-Z][a-z]+(?:idae|aceae)) family)"
    sp, seen = [], set()
    for a in arts:
        if not a["text"]:
            continue
        s0 = a["text"][0].split(" . ")[0] + " ."
        fm = _re.search(FAM, s0)
        mA = _re.match(r"^The ([a-z][a-z @\-@]+?) (?:or [^(]+ )?\( ([A-Z][a-z]+ [a-z]+)(?: [a-z]+)? [),;]", s0)
        mB = _re.match(r"^([A-Z][a-z]+ [a-z]+) , (?:commonly known as|common name|also known as|known as) (?:the )?([a-z][a-z @\-@]+?) (?:,|or)", s0)
        if not fm or not (mA or mB):
            continue
        c, b = (mA[1], mA[2]) if mA else (mB[2], mB[1])
        if "<unk>" in c + b or b in seen:
            continue
        seen.add(b)
        g = b.split()[0]
        sp.append({"common": c, "binomial": b, "family": fm[1] or fm[2], "genus": g, "lead": s0,
                   "bstmt": f"{b} belongs to the genus {g} ."})
    rng = random.Random(41)
    rng.shuffle(sp)
    return sp[30:], sp[:30]


class Text13:
    """frozen GPT-2 last-layer states of a text unit, scaled by one fixed mean / std; cached only when kept"""

    def __init__(self):
        self.tk, self.M = lm()
        self.c, self.mu, self.sd = {}, None, None

    def __call__(self, text, keep=True):
        if text in self.c:
            ids, H = self.c[text]
        else:
            ids = self.tk(" " + text)["input_ids"]
            with torch.no_grad():
                H = self.M(input_ids=torch.tensor([ids], device=dev), output_hidden_states=True).hidden_states[-1][0].float()
            if keep:
                self.c[text] = (ids, H)
        return ids, ((H - self.mu) / self.sd if self.mu is not None else H)

    def fit_scale(self, texts):
        R = torch.cat([self(t)[1] for t in texts])
        self.mu, self.sd = R.mean(0), R.std(0) + 1e-3

    def span(self, ids, name, last=False):
        """demo writing / grading only: the token span of a name (inclusive)"""
        for cand in (" " + name, " " + name[0].upper() + name[1:]):
            nid = self.tk(cand)["input_ids"]
            hits = [i for i in range(len(ids)) if ids[i:i + len(nid)] == nid]
            if hits:
                st = hits[-1] if last else hits[0]
                return st, st + len(nid) - 1
        return None

    def say(self, ids, sp):
        return self.tk.decode(ids[sp[0]:sp[1] + 1]).strip()


class Rel13(nn.Module):
    """identity (T5): relevance of a name mention to each node's stored mentions = learned pooled-name term + name
    tokens attended over the other side's context tokens; a create score; logits over [nodes..., create]"""

    def __init__(self, E):
        super().__init__()
        self.q = nn.Linear(E, 256)
        self.create = nn.Parameter(torch.tensor(0.))
        self.cq = nn.Linear(E, 1)
        self.wn = nn.Parameter(torch.tensor(1.))
        self.wc = nn.Parameter(torch.tensor(1.))

    def per_mention(self, men, n_, upto=None):
        P = lambda t: F.normalize(self.q(t), dim=-1)
        k = len(n_["m"]) if upto is None else upto
        name = P(torch.stack(n_["m"][:k])) @ P(men[0])
        pad = torch.nn.utils.rnn.pad_sequence
        cx, sp = n_["cx"][:k], n_["sp"][:k]
        CX, SP = P(pad(cx, batch_first=True)), P(pad(sp, batch_first=True))
        mcx = torch.arange(CX.shape[1], device=dev)[None] < torch.tensor([len(c) for c in cx], device=dev)[:, None]
        msp = torch.arange(SP.shape[1], device=dev)[None] < torch.tensor([len(s) for s in sp], device=dev)[:, None]
        a_sp, a_cx = P(men[3]), P(men[4])
        s1 = torch.einsum("ad,kld->akl", a_sp, CX).masked_fill(~mcx[None], -2.).amax(2).mean(0)
        s2 = (torch.einsum("ksd,cd->ksc", SP, a_cx).amax(2) * msp).sum(1) / msp.sum(1)
        return self.wn * name + self.wc * torch.maximum(s1, s2)

    def forward(self, men, nodes):
        """name nodes only compete (a number node is a typed value, never a name's target)"""
        crt = (self.create + self.cq(men[0]).squeeze())[None]
        if not nodes:
            return crt
        flat = {f: [v for n_ in nodes for v in n_[f]] for f in ("m", "cx", "sp")}
        own = torch.tensor([i for i, n_ in enumerate(nodes) for _ in n_["m"]], device=dev)
        sc = self.per_mention(men, flat) / .1
        node_sc = torch.full((len(nodes),), -1e4, device=dev).scatter_reduce(0, own, sc, "amax")
        node_sc = node_sc.masked_fill(torch.tensor([n_["typ"] == "num" for n_ in nodes], device=dev), -1e4)
        return torch.cat([node_sc, crt])


class Reader13(nn.Module):
    """the reading function: a loop over a unit's token states; each step points at a claim -- argument 1 (start, end),
    argument 2 (start, end) or none, relation words (start, end), value (start, end) or none -- or stops"""

    def __init__(self, E):
        super().__init__()
        self.E = E
        self.cell = nn.GRUCell(4 * E, 256)
        self.ptr = nn.Sequential(nn.Linear(E + 256, 256), nn.Tanh(), nn.Linear(256, 8))
        self.ctl = nn.Linear(E + 256, 3)                 # [stop, no 2nd argument, no value]
        # (11-17 phase 1: cloze 0/1180 -- a recurring name only became a node if it was part of a claim. Things are
        # marked by their own tagger: 0 outside, 1 begins a thing, 2 inside)
        self.tag = nn.Sequential(nn.Linear(E, 256), nn.Tanh(), nn.Linear(256, 3))

    def tag_loss(self, H, spans, full=True):
        """spans = things in this unit; full: every other token is outside; else only the spans + the token after"""
        lg = self.tag(H)
        y = torch.full((len(H),), -100, dtype=torch.long, device=dev)
        if full:
            y[:] = 0
        for a, b in spans:
            y[a] = 1
            y[a + 1:b + 1] = 2
        if not full:                                     # the token right after a span is not inside it
            for a, b in spans:
                if b + 1 < len(H) and int(y[b + 1]) == -100:
                    y[b + 1] = 0
        return F.cross_entropy(lg, y, ignore_index=-100) if (y >= 0).any() else lg.sum() * 0

    @torch.no_grad()
    def things(self, H):
        t, out, i = self.tag(H).argmax(-1).tolist(), [], 0
        while i < len(t):
            if t[i] == 1:
                j = i
                while j + 1 < len(t) and t[j + 1] == 2 and j - i < 7:
                    j += 1
                out.append((i, j))
                i = j + 1
            else:
                i += 1
        return out

    def logits(self, H, h):
        return self.ptr(torch.cat([H, h[None].expand(len(H), -1)], 1)), self.ctl(torch.cat([H.mean(0), h]))

    def advance(self, H, h, c):
        f = lambda sp: H[sp[0]:sp[1] + 1].mean(0) if sp is not None else torch.zeros(self.E, device=dev)
        args, r, v = c
        return self.cell(torch.cat([f(args[0]), f(args[1] if len(args) > 1 else None), f(r), f(v)])[None], h[None])[0]

    def loss(self, H, claims):
        sp_ = sorted({sp for c in claims for sp in list(c[0]) + ([c[2]] if c[2] is not None else [])})
        h, L = torch.zeros(256, device=dev), (self.tag_loss(H, sp_) if sp_ else 0.)
        T_ = lambda v: torch.tensor([v], device=dev)
        B_ = lambda x, y: F.binary_cross_entropy_with_logits(x, torch.tensor(float(y), device=dev))
        for c in list(claims) + [None]:
            P, C = self.logits(H, h)
            L = L + B_(C[0], c is None)
            if c is None:
                break
            args, r, v = c
            L = L + F.cross_entropy(P[:, 0][None], T_(args[0][0])) + F.cross_entropy(P[:, 1][None], T_(args[0][1]))
            L = L + B_(C[1], len(args) < 2)
            if len(args) > 1:
                L = L + F.cross_entropy(P[:, 2][None], T_(args[1][0])) + F.cross_entropy(P[:, 3][None], T_(args[1][1]))
            L = L + F.cross_entropy(P[:, 4][None], T_(r[0])) + F.cross_entropy(P[:, 5][None], T_(r[1]))
            L = L + B_(C[2], v is None)
            if v is not None:
                L = L + F.cross_entropy(P[:, 6][None], T_(v[0])) + F.cross_entropy(P[:, 7][None], T_(v[1]))
            h = self.advance(H, h, c)
        return L

    @torch.no_grad()
    def read(self, H, kmax=4):
        h, out = torch.zeros(256, device=dev), []
        for _ in range(kmax):
            P, C = self.logits(H, h)
            if C[0] > 0:
                break

            def sp(c0, c1):
                st = int(P[:, c0].argmax())
                return st, st + int(P[st:st + 8, c1].argmax())
            args = (sp(0, 1),) if C[1] > 0 else (sp(0, 1), sp(2, 3))
            c = (args, sp(4, 5), None if C[2] > 0 else sp(6, 7))
            out.append(c)
            h = self.advance(H, h, c)
        return out


class Dec13(nn.Module):
    """phase 1 round trip (user 11-03: node <-> text): a node's identity vector (Rel13.q of its mention) decodes its
    name token by token; output tied to GPT-2's (frozen) token embeddings"""

    def __init__(self, E, Q=256, Hd=256):
        super().__init__()
        self.inp = nn.Linear(Q, Hd)
        self.gru = nn.GRU(E, Hd, batch_first=True)
        self.out = nn.Linear(Hd, E)

    def loss(self, qvec, ids, wte):
        x = wte[torch.tensor([ids[:-1]], device=dev)] if len(ids) > 1 else torch.zeros(1, 0, wte.shape[1], device=dev)
        x = torch.cat([torch.zeros(1, 1, wte.shape[1], device=dev), x], 1)
        h, _ = self.gru(x, self.inp(qvec)[None, None].contiguous())
        return F.cross_entropy(self.out(h[0]) @ wte.T, torch.tensor(ids, device=dev))


class Mem13(nn.Module):
    """memory heads + the function nodes (ops). V11 entries: key = bind(r, arg atoms at their positions)"""

    def __init__(self, E, D=1024):
        super().__init__()
        self.D = D
        self.rv = nn.Linear(E, D)                        # relation words -> relation vector (the op of an entry)
        # (11-17 v12 measured: the last hop as cos(bind(question r), bind(edge r)) answered 21/120 even with gold claims,
        # 0 answers via a followed edge; v8b's learned relevance answered 87-91. Relations are matched like names:
        # learned relevance between the question's relation words and an edge's / a node's)
        self.af1 = nn.Sequential(nn.Linear(2 * E + 1, 128), nn.ReLU(), nn.Linear(128, 1))   # follow an edge
        self.ast = nn.Sequential(nn.Linear(2 * E, 128), nn.ReLU(), nn.Linear(128, 1))       # answer = this node
        self.pa, self.pb = nn.Parameter(torch.tensor(5.)), nn.Parameter(torch.tensor(-2.))  # answer by recall (point)
        self.anone = nn.Parameter(torch.tensor(0.))      # don't know
        self.cqn = nn.Linear(2 * E, E)                   # phase 1: which known thing is mentioned next (from the prefix)
        self.cnone = nn.Parameter(torch.tensor(0.))
        self.cp = nn.Linear(D, 1)                        # answer by computing (an op function)
        self.dig = nn.Embedding(10, 32)                  # typed digits (the exact number encoder)
        self.car = nn.Embedding(2, 32)                   # a column's state
        self.ocreate = nn.Parameter(torch.tensor(0.))    # a relation meets no op label -> make a new op
        self.ops = nn.ModuleList()                       # function nodes: each its OWN small column network
        self.op_rs = []                                  # each op's label = the relation words it was made for
        g = torch.Generator().manual_seed(7)
        self.register_buffer("P", torch.stack([torch.randperm(D, generator=g) for _ in range(3)]))

    def r(self, rstate):
        return F.normalize(self.rv(rstate), dim=-1)

    def key(self, r, atoms):
        k = r
        for i, a in enumerate(atoms):
            k = V11.bind(k, a[self.P[i]])
        return F.normalize(k, dim=-1)

    def route(self, r, make):
        """point the relation at an op label, or make a new op (the model creates functions)"""
        if self.op_rs:
            labs = self.r(torch.stack(self.op_rs))
            lg = torch.cat([(labs @ r) / .1, self.ocreate[None]])
        else:
            lg = self.ocreate[None]
        i = int(lg.argmax())
        return (i if i < len(self.op_rs) else None), lg

    def new_op(self, rstate):
        self.ops.append(nn.Sequential(nn.Linear(96, 128), nn.ReLU(), nn.Linear(128, 12)).to(dev))
        self.op_rs.append(rstate.detach())
        return len(self.ops) - 1

    def compute(self, i, a, b):
        """one firing per digit column, right to left; the carry is a soft state (learned from the digits it causes)"""
        net = self.ops[i]
        da, db = [int(c) for c in a[::-1]], [int(c) for c in b[::-1]]
        carry, out, lgs = self.car.weight[0], [], []
        for k in range(max(len(da), len(db)) + 1):
            x, y = (da[k] if k < len(da) else 0), (db[k] if k < len(db) else 0)
            o = net(torch.cat([self.dig.weight[x], self.dig.weight[y], carry]))
            lgs.append(o)
            out.append(int(o[:10].argmax()))
            carry = torch.softmax(o[10:], 0) @ self.car.weight
        return "".join(map(str, out[::-1])).lstrip("0") or "0", lgs


def _v13_run(stream, T, rel, mem, rdr, train=False, gold=False, H=3, BETA=30.):
    """one episode. stream: dicts {t: read | ask, text, gold (claims), ent, truth (ask), kind, var, rel, n_reads}"""
    nodes, decs, pairs, edges, outs, ent2 = [], [], [], [], {}, []
    num = {}                                             # typed numbers: exact value -> node
    st_ = {"right": 0, "asked_after": 0, "wrong_node": 0, "asked_before": 0, "right common": 0, "n common": 0,
           "right binomial": 0, "n binomial": 0, "n variant": 0, "right variant": 0, "dont know": 0,
           "claims gold": 0, "claims read right": 0, "q gold": 0, "q read right": 0, "ar n": 0, "ar right": 0,
           "ar long n": 0, "ar long right": 0, "ar computed": 0, "ar recalled": 0, "check n": 0, "check right": 0,
           "check new": 0, "check contra": 0, "n cloze": 0, "right cloze": 0, "n var": 0}
    traces, read_n, read_done, n_stmt, stm_hist = [], {}, set(), [0], []
    q_n, q_r = {}, {}

    def new_node(men, typ, ent, d):
        nodes.append({"m": [men[0]], "names": [men[1]], "sp": [men[3]], "cx": [men[4]], "typ": typ, "owner": ent,
                      "made_by": len(decs) - 1, "md": [d],
                      "atom": F.normalize(torch.randn(mem.D, device=dev), dim=0)})
        return len(nodes) - 1

    def point(men, ent, numeric):
        if numeric:                                      # typed value: exact encode (fixed, said)
            d = {"lg": None, "n": len(nodes), "j": None, "name": men[1]}
            decs.append(d)
            if men[1] not in num:
                num[men[1]] = new_node(men, "num", men[1], d)
            d["j"] = num[men[1]]
            return num[men[1]], d
        lg = rel(men, nodes)
        j = (random.randrange(len(lg)) if train and random.random() < .1 else int(lg.argmax()))
        if j < len(nodes) and nodes[j]["typ"] == "num":
            j = len(nodes)
        d = {"lg": lg if train else None, "n": len(nodes), "j": j, "name": men[1]}
        decs.append(d)
        if j == len(nodes):
            if train:
                pairs.append(("rt", men[0], men[2]))    # phase 1 round trip: the new node must say its name
                if any(n_["typ"] == "name" for n_ in nodes):   # a node costs (user D6): lean to the most relevant one
                    pairs.append(("cost", lg, int(lg[:len(nodes)].argmax())))
            return new_node(men, "name", ent, d), d
        for f, v in (("m", men[0]), ("names", men[1]), ("sp", men[3]), ("cx", men[4]), ("md", d)):
            nodes[j][f].append(v)
        return j, d

    def mention(ids, Hs, sp):
        v = Hs[sp[0]:sp[1] + 1]
        return (v.mean(0), T.say(ids, sp), ids[sp[0]:sp[1] + 1], v, Hs)

    def path1(start, truth):                             # edges ending with a forward edge into a node named truth
        front, seen = [(start, [])], {start}
        for _ in range(H):
            nxt = []
            for u, pth in front:
                for ei in outs.get(u, []):
                    e = edges[ei]
                    if e["dir"] == 0 and nodes[e["b"]]["names"][0] == truth:
                        return pth + [ei]
                    if e["b"] not in seen:
                        seen.add(e["b"])
                        nxt.append((e["b"], pth + [ei]))
            front = nxt
        return None

    def step1(qr, u, h):                                 # 0 don't know, 1 answer = this node, 2.. follow an edge of u
        opts = [mem.anone, mem.ast(torch.cat([qr, nodes[u]["m"][0]])).squeeze()]
        eis = outs.get(u, []) if h < H else []
        for ei in eis:
            opts.append(mem.af1(torch.cat([qr, edges[ei]["rs"], torch.tensor([float(edges[ei]["dir"])], device=dev)])).squeeze())
        return torch.stack(opts), eis

    def ask1(qr, j, truth, check_pairs, stmt=False):
        """walk from node j; -> (answer string, right path or None); adds targets (teacher-forced shortest path, then
        answer = that node; or don't know when nothing reachable)"""
        cur, got, taken = j, None, []
        for h in range(H + 1):
            lg, eis = step1(qr, cur, h)
            k = (random.randrange(len(lg)) if train and random.random() < .1 else int(lg.argmax()))
            taken.append((lg, k, eis))
            if k == 0:
                break
            if k == 1:
                got = nodes[cur]["names"][0]
                break
            cur = edges[eis[k - 2]]["b"]
        pth = None
        if train and check_pairs is not None:
            pth = path1(j, truth) if truth is not None else None
            if got is not None and got == truth:
                check_pairs.extend((lg, k, False) for lg, k, _ in taken)
            elif pth is not None:
                u = j
                for h, ei in enumerate(pth):
                    lg_, eis = step1(qr, u, h)
                    check_pairs.append((lg_, 2 + eis.index(ei), True))
                    u = edges[ei]["b"]
                lg_, _ = step1(qr, u, len(pth))
                check_pairs.append((lg_, 1, True))
                if taken[0][1] >= 2:                     # blame up (2'): two statements filed on one node
                    e_w, e_r = edges[taken[0][2][taken[0][1] - 2]], edges[pth[0]]
                    if e_w["stmt"] != e_r["stmt"]:
                        dl = max(e_w, e_r, key=lambda e: e["stmt"])["da"]
                        if dl["lg"] is not None and dl["j"] != dl["n"]:
                            check_pairs.append((dl["lg"], dl["n"], True))
            elif not (stmt and got is None):
                # (11-17 v13a0: a don't-know target for every brand-new statement fact drove 'don't know' to 120/120 even
                # with gold claims -- §1d says no prediction = new -> store, NO signal. Only a confident wrong
                # prediction of something memory cannot reach is taught 'don't know' (do not make things up))
                check_pairs.append((taken[0][0], 0, taken[0][1] != 0))
        return got, pth

    def keys2():
        if not ent2:
            return None
        R_ = mem.r(torch.stack([e["rs"] for e in ent2]))
        K = torch.stack([mem.key(R_[i], [nodes[e["a"]]["atom"], nodes[e["b"]]["atom"]]) for i, e in enumerate(ent2)])
        return K, torch.stack([nodes[e["v"]]["atom"] for e in ent2])

    def ask2(rstate, ia, ib, truth, check_pairs):
        """2 arguments: don't know / recall from memory / compute with the op the relation routes to"""
        rq = mem.r(rstate)
        KV = keys2()
        rec, best = None, torch.tensor(-1., device=dev)
        if KV is not None:
            sc = KV[0] @ mem.key(rq, [nodes[ia]["atom"], nodes[ib]["atom"]])
            best = sc.max()
            A = torch.stack([n_["atom"] for n_ in nodes])
            rec = nodes[int((A @ F.normalize(torch.softmax(BETA * sc, 0) @ KV[1], dim=0)).argmax())]["names"][0]
        oi, olg = mem.route(rq, make=train)
        if oi is None and train:                         # the model creates a function for a want no label meets
            oi = mem.new_op(rstate)
            check_pairs.append(("newop", oi))
        na_, nb_ = nodes[ia]["names"][0], nodes[ib]["names"][0]
        res, cl = (mem.compute(oi, na_, nb_) if oi is not None and na_.isdigit() and nb_.isdigit() else (None, None))
        lg = torch.stack([mem.anone, mem.pa * best + mem.pb if KV is not None else torch.tensor(-10., device=dev),
                          mem.cp(rq).squeeze() if oi is not None else torch.tensor(-10., device=dev)])
        k = (random.randrange(3) if train and random.random() < .1 else int(lg.argmax()))
        got = [None, rec, res][k]
        if train and truth is not None:
            ok_c, ok_r = res == truth, rec == truth
            tgt = 2 if (oi is not None and (ok_c or not ok_r)) else (1 if ok_r else 0)
            check_pairs.append((lg, tgt, got != truth))
            if cl is not None and truth.isdigit():       # CHECK 1 on a worked result: the columns learn its digits
                dg = [int(c) for c in truth[::-1]] + [0] * len(cl)
                for kk, o in enumerate(cl):
                    check_pairs.append((o[:10], dg[kk], not ok_c))
        return got, k

    for ev in stream:
        ids, Hs = T(ev["text"], keep=ev.get("keep", True))
        claims = (ev["gold"] or []) if gold else rdr.read(Hs, kmax=4 if ev["t"] == "read" else 1)
        if ev["t"] == "read" and ev["gold"] is not None:
            st_["claims gold"] += len(ev["gold"])
            st_["claims read right"] += sum(c in claims for c in ev["gold"])
        if ev["t"] == "ask" and ev["gold"] is not None:
            st_["q gold"] += 1
            st_["q read right"] += bool(claims) and claims[0][:2] == ev["gold"][0][:2]
        if train and not gold:
            traces.append({"H": Hs, "claims": claims, "ev": ev, "ok": None})
        ent = ev["ent"]
        if ev["t"] == "read":
            at, c_ok = {}, []
            if not gold:                                 # things the tagger marks become nodes (point / create)
                for sp in rdr.things(Hs):
                    men = mention(ids, Hs, sp)
                    at[sp] = point(men, ent if men[1] in ev.get("names", ()) else men[1], men[1].isdigit())
            stm_hist.append((ids, Hs))
            for c in claims:
                args, rsp, vsp = c
                if vsp is None:
                    continue
                js = []
                for sp in args + (vsp,):
                    if sp not in at:
                        men = mention(ids, Hs, sp)
                        at[sp] = point(men, ent if men[1] in ev.get("names", ()) else men[1], men[1].isdigit())
                    js.append(at[sp])
                rs = Hs[rsp[0]:rsp[1] + 1].mean(0)
                vname = nodes[js[-1][0]]["names"][0]
                cp = []                                  # CHECK 1: predict the value before it is stored
                if len(args) == 1:
                    got, _ = ask1(rs, js[0][0], vname, cp if train else None, stmt=True)
                else:
                    got, _ = ask2(rs, js[0][0], js[1][0], vname, cp if train else [])
                st_["check n"] += 1
                st_["check right"] += got == vname
                st_["check new"] += got is None
                st_["check contra"] += got is not None and got != vname
                if ev.get("kind") == "arith":
                    c_ok.append(got == vname)
                if train:
                    pairs.extend(cp)
                if len(args) == 1:                       # store (an edge both ways; the newest stays readable)
                    (a_, da), (b_, _) = js
                    if a_ != b_:
                        for dr, (u, v) in ((0, (a_, b_)), (1, (b_, a_))):
                            outs.setdefault(u, []).append(len(edges))
                            edges.append({"a": u, "b": v, "rs": rs, "dir": dr, "stmt": n_stmt[0], "da": da})
                else:
                    ent2.append({"a": js[0][0], "b": js[1][0], "v": js[2][0], "rs": rs})
            n_stmt[0] += 1
            if train and not gold:
                traces[-1]["ok"] = all(c_ok) if c_ok else None
            read_n[ent] = read_n.get(ent, 0) + 1
            if read_n[ent] >= ev.get("n_reads", 1):
                read_done.add(ent)
            continue
        if ev["kind"] == "cloze":                        # phase 1: which known thing comes next? (point, or don't know)
            truth = ev["truth"]
            st_["n cloze"] += 1
            q = mem.cqn(torch.cat([Hs[-1], Hs.mean(0)]))
            name_n = list(range(len(nodes)))
            if name_n:
                flat = [m_ for n_ in nodes for m_ in n_["m"]]
                own = torch.tensor([i for i, n_ in enumerate(nodes) for _ in n_["m"]], device=dev)
                P = lambda t: F.normalize(rel.q(t), dim=-1)
                sc = (P(torch.stack(flat)) @ P(q)) / .1
                ns = torch.full((len(nodes),), -1e4, device=dev).scatter_reduce(0, own, sc, "amax")
                lg = torch.cat([ns, mem.cnone[None]])
            else:
                lg = mem.cnone[None]
            k = int(lg.argmax())
            ok_ = k < len(nodes) and truth in nodes[k]["names"]
            st_["right cloze"] += ok_
            st_["dont know"] += k == len(nodes)
            if train:
                has_t = [i for i in range(len(nodes)) if truth in nodes[i]["names"]]
                tgt = max(has_t, key=lambda i: float(lg[i])) if has_t else len(nodes)
                pairs.append((lg, tgt, not ok_))
                if not has_t:                            # no node holds it: its earlier mentions were not marked as
                    tid = ev["tgt_ids"]                  # a thing -> the tagger learns to mark them
                    for ids_k, Hs_k in stm_hist:
                        occ = [(i, i + len(tid) - 1) for i in range(len(ids_k) - len(tid) + 1) if ids_k[i:i + len(tid)] == tid]
                        if occ:
                            pairs.append(("tag", Hs_k, occ))
            continue
        # a question: the same check against the truth that follows it
        truth = ev["truth"]
        got, j, d = None, None, None
        if claims and claims[0][2] is None:
            args, rsp, _ = claims[0]
            js = []
            for sp in args:
                men = mention(ids, Hs, sp)
                js.append(point(men, ent if ev["kind"] != "arith" else men[1], men[1].isdigit()))
            rs = Hs[rsp[0]:rsp[1] + 1].mean(0)
            cp = []
            if len(args) == 1:
                j, d = js[0]
                got, pth = ask1(rs, j, truth, cp if train else None)
            else:
                got, k2 = ask2(rs, js[0][0], js[1][0], truth, cp if train else [])
                st_["ar computed"] += k2 == 2
                st_["ar recalled"] += k2 == 1
            if train:
                pairs.extend(cp)
        if train and not gold:
            traces[-1]["ok"] = got == truth
        q_n[ent] = q_n.get(ent, 0) + 1
        q_r[ent] = q_r.get(ent, 0) + (got == truth)
        if ev["kind"] == "arith":
            st_["ar n"] += 1
            st_["ar right"] += got == truth
            if ev.get("long"):
                st_["ar long n"] += 1
                st_["ar long right"] += got == truth
            continue
        if ent not in read_done:
            st_["asked_before"] += 1
            st_["wrong_node"] += j is not None and nodes[j]["owner"] != ent
            continue
        st_["asked_after"] += 1
        ok_ = got == truth
        if os.environ.get("V13_TRACE") == "1" and not train and len(st_.setdefault("traces", [])) < 8 and j is not None:
            with torch.no_grad():                        # (diagnosis, log only)
                lg0, eis0 = step1(rs, j, 0)
                pt = path1(j, truth)
            st_["traces"].append(f"Q '{ev['text']}' -> node {j} '{nodes[j]['names'][0]}' (names {nodes[j]['names'][:3]}) | "
                                 f"truth '{truth}' reachable {None if pt is None else len(pt)} hop(s) | step-1 scores: "
                                 f"dk {float(lg0[0]):.2f}, answer here {float(lg0[1]):.2f}, edges "
                                 + ", ".join(f"{float(lg0[2 + i]):.2f}->'{nodes[edges[e]['b']]['names'][0]}'" for i, e in enumerate(eis0[:5]))
                                 + f" | got {got!r}")
        st_["right"] += ok_
        st_["n " + ev["kind"]] += 1
        st_["right " + ev["kind"]] += ok_
        st_["n variant"] += bool(ev.get("var"))
        st_["right variant"] += bool(ev.get("var")) and ok_
        st_["dont know"] += got is None
        if not train or d is None or d["lg"] is None:
            continue
        if ok_ or pth is not None:
            pairs.append((d["lg"], j, False))
            continue
        if not any(n_["names"][0] == truth for n_ in nodes):           # blame up (5)
            for dd in decs:
                if dd["name"] == truth and dd["lg"] is not None and dd["j"] != dd["n"]:
                    pairs.append((dd["lg"], dd["n"], True))
            continue
        # (11-17 v13 trace: with the genus on a split node no node reached ALL values -> the target was 'create' -> more
        # splits. Pointing is blamed only AFTER reading, when the thing's own node exists: target = a node that reaches
        # the ASKED value, the one the model itself rates most relevant)
        ok = [i for i in range(d["n"]) if nodes[i]["typ"] == "name" and path1(i, truth) is not None]
        t_ = max(ok, key=lambda i: float(d["lg"][i])) if ok else d["n"]
        pairs.append((d["lg"], t_, True))
        if ok and j != t_ and len(nodes[j]["m"]) > 1:   # blame up (3): the stored mention this question matched in
            with torch.no_grad():                        # the wrong node was put there by a POINTING decision
                k3 = int(rel.per_mention(mention(ids, Hs, claims[0][0][0]), nodes[j], len(nodes[j]["m"]) - 1).argmax())
            dk = nodes[j]["md"][k3]
            if dk is not d and dk["lg"] is not None and dk["j"] != dk["n"]:
                pairs.append((dk["lg"], t_ if t_ < dk["n"] else dk["n"], True))
        if ok and j != t_:                               # blame up (1)
            dm = decs[nodes[j]["made_by"]]
            if dm is not d and dm["lg"] is not None and t_ < dm["n"]:
                pairs.append((dm["lg"], t_, True))
    if train:                                            # the reader learns from its own readings that checked out (D7)
        for tr in traces:
            ok_ = tr["ok"]
            if ok_ is None and tr["ev"]["t"] == "read" and tr["ev"].get("kind") != "arith":   # a statement: its thing's
                e_ = tr["ev"]["ent"]                                                          # questions all right
                ok_ = q_n.get(e_, 0) > 0 and q_r.get(e_, 0) == q_n[e_]
            if ok_ and tr["claims"]:
                pairs.append(("reader", (tr["H"], tr["claims"]), .1))
    st_["nodes"], st_["edges"], st_["ops"] = len(nodes), len(edges), len(mem.ops)
    return st_, pairs


def _wiki_docs(n_hold=300, max_sent=12, min_sent=4):
    """phase 1 data: raw Wikipedia articles as documents of up to 12 sentences (no labels)"""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bio", "wiki_e1.json.gz")
    docs = []
    for a in json.load(gzip.open(path, "rt", encoding="utf-8")):
        sents = []
        for para in a.get("text", []):
            for s_ in para.split(" . "):
                s_ = s_.strip()
                if 4 <= len(s_.split()) <= 40 and "<unk>" not in s_:
                    sents.append(s_ + " .")
                if len(sents) >= max_sent:
                    break
            if len(sents) >= max_sent:
                break
        if len(sents) >= min_sent:
            docs.append({"title": a.get("title", ""), "sents": sents})
    random.Random(83).shuffle(docs)
    return docs[n_hold:], docs[:n_hold]


def _v13_phase1(T, rel, mem, rdr, dec, train_on, tr_docs, ho_docs, report, sd_):
    """PHASE 1 (user 11-17: "can we pretrain the perception as a separate phase?" -> "sure then"): raw Wikipedia, no
    labels. Each document is read sentence by sentence into a fresh memory; BEFORE a sentence is stored, every span in
    it that already appeared earlier in the document is a cloze -- the sentence's prefix is asked as a question and GNM
    must reach the span from memory (check 1 = prediction met, user 11-03), through the same reader / identity / walk /
    blames as the skills. + round trip (a new node's identity vector says its name; user 11-03 node <-> text) + a small
    cost per node (D6). Which spans are cloze targets (recurring in the document, with a token outside the 300 most
    common) is a data statistic picking training targets only, never an input. Eval: held-out documents, cloze exact;
    baselines: GPT-2 seeing only the current sentence (no memory) and GPT-2 seeing the whole document so far"""
    tk, M = T.tk, T.M
    cnt = collections.Counter(t for d in tr_docs[:3000] for s_ in d["sents"] for t in tk(" " + s_)["input_ids"])
    common = {t for t, _ in cnt.most_common(300)}

    def has(seq, g):
        return any(seq[i:i + len(g)] == g for i in range(len(seq) - len(g) + 1))

    def events(docs, keep=False, max_tgt=2):
        out = []
        for d in docs:
            ent, prev = "doc:" + d["title"], []
            for k, s_ in enumerate(d["sents"]):
                ids = tk(" " + s_)["input_ids"]
                if k:
                    i, n_t = 1, 0
                    ts = tk.convert_ids_to_tokens(ids)
                    punct = [not any(ch.isalnum() for ch in t_.replace("Ġ", "")) for t_ in ts]

                    def ok_span(i0, n):                  # (11-17 CPU check: fragments 'cephal', 'ney', ', the great hammer',
                        j0 = i0 + n                       #  'called' -> whole words, no punctuation inside, a capitalised
                        if j0 > len(ids) or not ts[i0].startswith("Ġ") or any(punct[i0:j0]):   # word / digit / 2+ words)
                            return False
                        if j0 < len(ids) and not (ts[j0].startswith("Ġ") or punct[j0]):
                            return False
                        txt = tk.decode(ids[i0:j0]).strip()
                        return (any(w[:1].isupper() or w[:1].isdigit() for w in txt.split()) or (len(txt.split()) >= 2 and ids[j0 - 1] not in common)) and                             any(t not in common for t in ids[i0:j0]) and any(has(pv, ids[i0:j0]) for pv in prev)
                    while i < len(ids) and n_t < max_tgt:
                        best = next((n for n in (4, 3, 2, 1) if ok_span(i, n)), 0)
                        if best:
                            truth, prefix = tk.decode(ids[i:i + best]).strip(), tk.decode(ids[:i]).strip()
                            if truth and prefix:
                                out.append({"t": "ask", "text": prefix, "gold": None, "ent": ent, "truth": truth, "kind": "cloze",
                                            "keep": keep, "ctx": " ".join(d["sents"][:k]), "n_tok": best, "tgt_ids": ids[i:i + best]})
                                n_t += 1
                            i += best
                        else:
                            i += 1
                out.append({"t": "read", "text": s_, "gold": None, "ent": ent, "n_reads": 1, "keep": keep})
                prev.append(ids)
        return out
    ho = [events([d], keep=True) for d in ho_docs[:150]]

    @torch.no_grad()
    def greedy(text, n):
        ids = tk(" " + text)["input_ids"][-900:]
        for _ in range(n):
            ids.append(int(M(input_ids=torch.tensor([ids], device=dev)).logits[0, -1].argmax()))
        return tk.decode(ids[-n:]).strip()
    qs = [e for evs in ho for e in evs if e["t"] == "ask"]
    b_sent = sum(greedy(e["text"], e["n_tok"]) == e["truth"] for e in qs)
    b_doc = sum(greedy(e["ctx"] + " " + e["text"], e["n_tok"]) == e["truth"] for e in qs)
    print(f"v13 PHASE 1: {len(tr_docs)} training documents, {len(ho_docs[:150])} held-out; held-out cloze targets {len(qs)}; "
          f"GPT-2 with this sentence only (no memory) {b_sent}/{len(qs)}, GPT-2 with the whole document so far in its "
          f"context {b_doc}/{len(qs)}", flush=True)

    def cloze(tag):
        n_ = r_ = dk = 0
        with torch.no_grad():
            for evs in ho:
                st, _ = _v13_run(evs, T, rel, mem, rdr)
                n_, r_, dk = n_ + st["n cloze"], r_ + st["right cloze"], dk + st["dont know"]
        print(f"v13 PHASE 1 {tag}: held-out cloze from GNM's memory (it sees one sentence at a time) {r_}/{n_}, "
              f"don't know {dk}", flush=True)
    cloze("step 0")
    rng = random.Random(97 + sd_)
    at = {int(v) for v in os.environ.get("V13_AT", "").split(",") if v}
    for stp in range(int(os.environ.get("V13_P1_STEPS", "3000"))):
        if stp in at and stp:
            cloze(f"step {stp}")
        _, pairs = _v13_run(events(rng.sample(tr_docs, 2)), T, rel, mem, rdr, train=True)
        train_on(pairs, stp)
    cloze("final")
    path = os.environ.get("V13_SAVE", "/marimo/out/v13p1.pt")
    torch.save({"rel": rel.state_dict(), "mem": {k: v for k, v in mem.state_dict().items() if not k.startswith("ops.")},
                "rdr": rdr.state_dict(), "dec": dec.state_dict(), "mu": T.mu, "sd": T.sd}, path)
    print(f"v13 PHASE 1: perception saved to {path}", flush=True)
    report("after phase 1 only (no skill training)")


def v13(steps=3000):
    """§1d build: see the section comment. Eval: biology questions (30 never-trained species, half by name variants),
    arithmetic questions (unseen pairs of 1-3 digits; and 4-6 digits = longer than any training number), reader
    exactness, the check-1 tally; READER claims vs gold claims (reference)"""
    sd_ = int(os.environ.get("V13_SEED", "0"))
    torch.manual_seed(sd_)
    random.seed(sd_)
    T = Text13()
    E = T.M.config.n_embd
    train_sp, test_sp = _v13_species()
    rels = ("family", "genus")
    W = {"family": ["Which family is {n} in?", "What family does {n} belong to?", "{N} is placed in which family?", "Name the family of {n}."],
         "genus": ["Which genus is {n} in?", "What genus does {n} belong to?", "{N} is classified in which genus?", "Name the genus of {n}."]}
    VAR = float(os.environ.get("V13_VAR", ".5"))

    def variant(x, kind, r_):
        if r_.random() >= VAR:
            return None
        if kind == "binomial":
            g, e = x["binomial"].split()[:2]
            return g[0] + ". " + e
        c = x["common"]
        opts = [c + ("es" if c.endswith(("s", "sh", "ch", "x")) else "s"),
                " ".join(w if w == "@-@" else w[0].upper() + w[1:] for w in c.split())]
        if " @-@ " in c:
            opts.append(c.replace(" @-@ ", " "))
        return opts[r_.randrange(len(opts))]

    def gold_stmt(text, pairs_):
        """demo writing / grading: (args, relation words = last <= 4 tokens before the value, value), inclusive"""
        ids, _ = T(text)
        out = []
        for names_, v in pairs_:
            ar = [T.span(ids, n_) for n_ in names_]
            vs = T.span(ids, v, last=True)
            if None in ar or vs is None:
                return None
            lo = max(a[1] for a in ar) + 1 if len(ar) == 1 else ar[0][1] + 1
            hi = vs[0] if len(ar) == 1 else ar[1][0]
            if hi - lo < 1:
                lo, hi = vs[1] + 1, ar[0][0]             # value written before the argument ("X , known as Y")
                if hi - lo < 1:
                    return None
            out.append((tuple(ar), (max(lo, hi - 4), hi - 1), vs))
        return out

    def bio_events(xs, r_, ask_before=.5, n_after=2):
        per = []
        for x in xs:
            names = (x["common"], x["binomial"])
            vals = {r__: x[r__] for r__ in rels}
            ev = []
            qs = [(rel, kind, t) for rel in rels for kind in ("common", "binomial") for t in W[rel]]

            def ask(rel, kind, t):
                var = variant(x, kind, r_)
                nm = var or x[kind]
                n = ("the " + nm) if kind == "common" else nm
                q = t.replace("{N}", n[0].upper() + n[1:]).replace("{n}", n)
                ids, _ = T(q)
                sp = T.span(ids, nm)
                rw = next((i for i, t_ in enumerate(ids) if t_ == T.tk(" " + rel)["input_ids"][0]), None)
                g = [((sp,), (rw, rw), None)] if sp is not None and rw is not None else None
                return {"t": "ask", "text": q, "gold": g, "ent": x["binomial"], "truth": x[rel], "kind": kind,
                        "var": var is not None, "names": names, "values": vals}
            if r_.random() < ask_before:
                ev += [ask(*a) for a in r_.sample(qs, 2)]
            lead = {"t": "read", "text": x["lead"], "ent": x["binomial"], "names": names, "n_reads": 2,
                    "gold": gold_stmt(x["lead"], [((x["common"],), x["binomial"]), ((x["common"],), x["family"])])}
            bst = {"t": "read", "text": x["bstmt"], "ent": x["binomial"], "names": names, "n_reads": 2,
                   "gold": gold_stmt(x["bstmt"], [((x["binomial"],), x["genus"])])}
            ev += r_.sample([lead, bst], 2)
            ev += [ask(*a) for a in r_.sample(qs, n_after)]
            per.append(ev)
        return per

    def arith_events(r_, n, digits, keep=False, long=False):
        out = []
        def num_(dd):
            return r_.randint(10 ** (dd - 1) if dd > 1 else 0, 10 ** dd - 1)
        for _ in range(n):
            a, b = num_(r_.choice(digits)), num_(r_.choice(digits))
            c = str(a + b)
            if r_.random() < .5:
                text = f"{a} plus {b} is {c} ."
                out.append({"t": "read", "text": text, "ent": c, "keep": keep, "kind": "arith",
                            "gold": gold_stmt(text, [((str(a), str(b)), c)])})
            else:
                text = f"What is {a} plus {b} ?"
                ids, _ = T(text, keep=keep)
                sa, sb = T.span(ids, str(a)), T.span(ids, str(b))
                pl = T.span(ids, "plus")
                g = [((sa, sb), pl, None)] if None not in (sa, sb, pl) else None
                out.append({"t": "ask", "text": text, "gold": g, "ent": c, "truth": c, "kind": "arith", "keep": keep,
                            "long": long})
        return out

    def mix(per, extra, r_):
        per = [list(e) for e in per] + [[e] for e in extra]
        out = []
        while any(per):
            e = r_.choice([e for e in per if e])
            out.append(e.pop(0))
        return out
    PHASE, INIT = os.environ.get("V13_PHASE", "2"), os.environ.get("V13_INIT", "")
    ck = torch.load(INIT, map_location=dev) if INIT else None
    tr_docs, ho_docs = _wiki_docs() if PHASE == "1" else ([], [])
    if ck is not None:                                   # phase 2 from phase 1: the same fixed scale
        T.mu, T.sd = ck["mu"], ck["sd"]
    else:
        T.fit_scale([x["lead"] for x in train_sp[:60]] + [f"{i} plus {i + 7} is {2 * i + 7} ." for i in range(10, 400, 13)]
                    + [s_ for d_ in tr_docs[:40] for s_ in d_["sents"]])
    erng = random.Random(47)
    test_bio = mix(bio_events(test_sp, erng, n_after=4), [], erng)
    test_ar = arith_events(random.Random(48), 60, (1, 2, 3), keep=True)
    test_ar = [e for e in test_ar if e["t"] == "ask"]
    long_ar = [e for e in arith_events(random.Random(49), 80, (4, 5, 6), keep=True, long=True) if e["t"] == "ask"]
    rel, mem, rdr, dec = Rel13(E).to(dev), Mem13(E).to(dev), Reader13(E).to(dev), Dec13(E).to(dev)
    wte = T.M.get_input_embeddings().weight.detach()
    if ck is not None:                                   # perception pretrained in phase 1
        rel.load_state_dict(ck["rel"])
        mem.load_state_dict(ck["mem"], strict=False)
        rdr.load_state_dict(ck["rdr"])
        dec.load_state_dict(ck["dec"])
        print(f"v13: perception loaded from {INIT} (phase 1)", flush=True)
    # ~50 demos for the reader (statements + questions, biology + arithmetic) and ~50 column steps (worked sums)
    drng = random.Random(71 + sd_)
    demo = []
    for e in mix(bio_events(drng.sample(train_sp, 12), drng), arith_events(drng, 14, (1, 2, 3)), drng):
        if e["gold"] and len(demo) < 50:
            demo.append((T(e["text"])[1], e["gold"]))
    ropt = torch.optim.AdamW(rdr.parameters(), lr=1e-3)
    for it in range(int(os.environ.get("V13_DEMO_STEPS", "1500"))):
        Hs, cl = demo[it % len(demo)]
        L = rdr.loss(Hs, cl)
        ropt.zero_grad()
        L.backward()
        ropt.step()
    demo_ar = [e for e in arith_events(drng, 40, (1, 2)) if e["t"] == "read" and e["gold"]][:16]
    cols = 0
    for e in demo_ar:                                    # the op is made for the demos' relation words; column steps
        ids, Hs = T(e["text"])
        (sa, sb), rsp, vs = e["gold"][0]
        rstate = Hs[rsp[0]:rsp[1] + 1].mean(0)
        if not mem.ops:
            mem.new_op(rstate)
        cols += len(str(max(int(T.say(ids, sa)), int(T.say(ids, sb))))) + 1
    copt = torch.optim.AdamW(list(mem.ops.parameters()) + list(mem.dig.parameters()) + list(mem.car.parameters()), lr=2e-3)
    for it in range(300):                                # demo column steps with their carries (teacher demos, ~50)
        e = demo_ar[it % len(demo_ar)]
        ids, Hs = T(e["text"])
        (sa, sb), rsp, vs = e["gold"][0]
        a, b = T.say(ids, sa), T.say(ids, sb)
        da, db = [int(c) for c in a[::-1]], [int(c) for c in b[::-1]]
        carry, L, cin = mem.car.weight[0], 0., 0
        for k in range(max(len(da), len(db)) + 1):
            x, y = (da[k] if k < len(da) else 0), (db[k] if k < len(db) else 0)
            o = mem.ops[0](torch.cat([mem.dig.weight[x], mem.dig.weight[y], carry]))
            s_ = x + y + cin
            L = L + F.cross_entropy(o[:10][None], torch.tensor([s_ % 10], device=dev)) + \
                F.cross_entropy(o[10:][None], torch.tensor([s_ // 10], device=dev))
            cin = s_ // 10
            carry = torch.softmax(o[10:], 0) @ mem.car.weight
        copt.zero_grad()
        L.backward()
        copt.step()
    print(f"v13: reader taught on {len(demo)} demo units; op made for the relation words of the demo sums, "
          f"{len(demo_ar)} demo sums = {cols} column steps; FIXED: {V13_FIXED}", flush=True)
    opt = torch.optim.AdamW(list(rel.parameters()) + list(mem.parameters()) + list(rdr.parameters()) + list(dec.parameters()),
                            lr=1e-3)
    W_RT, W_COST = float(os.environ.get("V13_RT", ".1")), float(os.environ.get("V13_COST", ".02"))
    n_ops = [len(mem.ops)]

    def train_on(pairs, stp):
        """one update from an episode's targets (checks, blames, reader traces, round trips, node costs)"""
        if len(mem.ops) > n_ops[0]:                      # a function the model made: its network joins the learning
            for op in mem.ops[n_ops[0]:]:
                opt.add_param_group({"params": list(op.parameters())})
            n_ops[0] = len(mem.ops)
        L, n_ = 0., 0
        rw = .3 * max(0., 1. - stp / 1000.)              # (11-17 v13a0: the SAME 50 reader demos early, fading by 1000)
        if rw > 0:
            for Hs_, cl_ in random.sample(demo, 4):
                L = L + rw * rdr.loss(Hs_, cl_)
                n_ += 1
        for p_ in pairs:
            if isinstance(p_[0], str):
                if p_[0] == "reader":
                    L = L + p_[2] * rdr.loss(*p_[1])
                elif p_[0] == "rt" and p_[2]:
                    L = L + W_RT * dec.loss(rel.q(p_[1]), list(p_[2]), wte)
                elif p_[0] == "cost":
                    L = L + W_COST * F.cross_entropy(p_[1][None], torch.tensor([p_[2]], device=dev))
                elif p_[0] == "tag":                     # a recurring span that no node held: mark it as a thing
                    L = L + rdr.tag_loss(p_[1], p_[2], full=False)
                else:
                    continue
            else:
                lg, tgt, wrong = p_
                L = L + (1. if wrong else .1) * F.cross_entropy(lg[None], torch.tensor([tgt], device=dev))
            n_ += 1
        if n_:
            opt.zero_grad()
            (L / n_).backward()
            opt.step()

    def report(tag):
        with torch.no_grad():
            for gr in (False, True):
                lab = "gold claims (reference)" if gr else "READER claims"
                sb, _ = _v13_run(test_bio, T, rel, mem, rdr, gold=gr)
                sa, _ = _v13_run(test_ar + long_ar, T, rel, mem, rdr, gold=gr)
                pc = lambda s, a_, b_: f"{s[a_]}/{s[b_]}"
                print(f"v13 {tag} {lab}: BIO right after reading {pc(sb, 'right', 'asked_after')} (common "
                      f"{pc(sb, 'right common', 'n common')}, binomial {pc(sb, 'right binomial', 'n binomial')}, variant "
                      f"{pc(sb, 'right variant', 'n variant')}), don't know {sb['dont know']}, before reading -> another "
                      f"thing's node {pc(sb, 'wrong_node', 'asked_before')} | ARITH 1-3 digits "
                      f"{sa['ar right'] - sa['ar long right']}/{sa['ar n'] - sa['ar long n']}, 4-6 digits "
                      f"{pc(sa, 'ar long right', 'ar long n')} (computed {sa['ar computed']}, recalled {sa['ar recalled']}) "
                      f"| READER exact: claims {pc(sb, 'claims read right', 'claims gold')}, questions "
                      f"{sb['q read right'] + sa['q read right']}/{sb['q gold'] + sa['q gold']} | CHECK 1 on statements: "
                      f"right {sb['check right']}, new {sb['check new']}, contradiction {sb['check contra']} of "
                      f"{sb['check n']} | nodes {sb['nodes']}, edges {sb['edges']}, ops {sb['ops']}", flush=True)
                for tr in sb.get("traces", []):
                    print(f"   TRACE {lab[:6]}: {tr}", flush=True)
    if PHASE == "1":
        _v13_phase1(T, rel, mem, rdr, dec, train_on, tr_docs, ho_docs, report, sd_)
        return
    report("step 0")
    trng = random.Random(43 + sd_)
    at = {int(v) for v in os.environ.get("V13_AT", "").split(",") if v}
    for stp in range(steps):
        if stp in at and stp:
            report(f"step {stp}")
        bio = bio_events(trng.sample(train_sp, trng.randint(8, 20)), trng)
        stream = mix(bio, arith_events(trng, 20, (1, 2, 3)), trng)
        _, pairs = _v13_run(stream, T, rel, mem, rdr, train=True)
        train_on(pairs, stp)
    report("final")


# ====================================================================== inspection (11-14: e1b nodes all said the same few phrases)
def inspect(G, arts, tr_arts, n=20):
    """what the trained parts actually do: node collapse, write attention, retrieval vs a node's source, gradients"""
    G.eval()
    rng = random.Random(3)
    per_art, att_ent, ret_hit, ret_n, hid = [], [], 0, 0, []
    with torch.no_grad():
        for a in arts[:n]:
            mem, src = Memory(), []
            for si, ids in enumerate(a["ids"]):
                before = mem.n
                G.step(mem, ids)
                src += [si] * (mem.n - before)
                A = G._A
                ent = -(A.clamp_min(1e-9) * A.clamp_min(1e-9).log()).sum(-1) / math.log(A.shape[1])
                att_ent.append(float(ent.mean()))
                hid.append(G._H.mean(0))
            if mem.n:
                per_art.append((a["title"], torch.stack([v.mean(0) for v in mem.V]), src, mem))
            for p, t in quiz(a, rng, 3):
                si = next(i for i, ids in enumerate(a["ids"]) if ids[:len(p)] == p)
                sel, _, _ = G.retrieve(mem, G.query(p))
                top = sel[:G.topk]
                ret_n += 1
                ret_hit += any(i < len(src) and src[i] == si for i in top)
    within, across = [], []
    for x, (ta, Ra, _, _) in enumerate(per_art):
        Ra = F.normalize(Ra, dim=-1)
        if len(Ra) > 1:
            c = Ra @ Ra.T
            within.append(float(c[~torch.eye(len(Ra), dtype=torch.bool, device=c.device)].mean()))
        for ta2, Rb, _, _ in per_art[x + 1:]:
            across.append(float((Ra @ F.normalize(Rb, dim=-1).T).mean()))
    Hm = F.normalize(torch.stack(hid), dim=-1)
    print(f"nodes: mean cos within an article {sum(within) / len(within):.3f}, across articles {sum(across) / len(across):.3f} "
          f"(near 1 = collapse); LM states (sentence means) mean cos across sentences {float((Hm @ Hm.T).mean()):.3f}")
    print(f"write attention entropy / max {sum(att_ent) / len(att_ent):.3f} (1 = spread evenly over the step = an average)")
    print(f"retrieval: a blank's top-{G.topk} holds a node written from its own sentence {ret_hit}/{ret_n} "
          f"(memory has ~{sum(len(r[1]) for r in per_art) / max(1, len(per_art)):.0f} nodes an article)")
    G.train()
    mem = Memory()
    cost, rt = read(G, mem, tr_arts[0], train=True)
    qs = quiz(tr_arts[0], random.Random(0), 4)
    l1 = sum(G.say_nll(mem, p, t)[0] for p, t in qs) / len(qs)
    G.zero_grad()
    (l1 + .3 * rt + cost).backward()
    print("gradient norms: " + ", ".join(f"{nm} {float(p.grad.norm()) if p.grad is not None else 0:.2e}" for nm, p in G.named_parameters()))
    title, R, src, mem = per_art[0]
    print(f"\n{title}: per node, its source sentence and what it says alone")
    for i in range(min(8, mem.n)):
        print(f"   node {i} (sentence {src[i]}): {G.node_text(mem, i, 12)!r}")


@torch.no_grad()
def trace(G, art, nq=3):
    """fire the loop by hand on one article and print every firing: what was fetched, what each slot took (tokens),
    its gate, new or which node it replaced, what the node says alone, its relations, what it woke; then questions"""
    tk = lm()[0]
    G.eval()
    mem, src = Memory(), {}
    say = lambda i: G.node_text(mem, i, 10)
    for si, ids in enumerate(art["ids"]):
        print(f"\nFIRE on sentence {si}: {tk.decode(ids)!r}")
        q = G.query(ids)
        sel, rows, owner = G.retrieve(mem, q)
        if sel:
            sc = torch.stack(mem.k) @ q
            for i in sel:
                print(f"   fetched n{i} (written at sentence {src.get(i)}, key score {float(sc[i]):+.2f}): says {say(i)!r}")
        else:
            print("   fetched: nothing (memory empty)")
        n_before = mem.n
        V_before = [v.clone() for v in mem.V]
        written, woken, cost = G.step(mem, ids)
        A = G._A.view(G.K, G.m, -1).sum(1)                 # per slot: attention over the step's positions
        labels = [f"[n{o} row]" for o in owner] + [tk.decode([t]) for t in ids]
        for j in range(G.K):
            if G.wmode == "span":
                st = G._A.view(G.K, G.m, -1)[j, 0]
                b0 = int(st.argmax())
                print(f"   slot {j}: span starts at {labels[b0]!r} (p {float(st[b0]):.2f}): {' '.join(repr(x) for x in labels[b0:b0 + G.m])}")
                continue
            top = A[j].topk(min(6, A.shape[1]))
            took = ", ".join(f"{labels[int(t)]!r} {float(w) / G.m:.2f}" for w, t in zip(top.values, top.indices))
            print(f"   slot {j}: takes {took}")
        for i in written:
            kind = "NEW" if i >= n_before else ("EDIT of n%d" % i)
            if i >= n_before:
                src[i] = si
            ch = "" if i >= n_before else f" (moved {float((mem.V[i] - V_before[i]).norm()):.2f})"
            print(f"   -> {kind} n{i}{ch}: says {say(i)!r}; relations {[(f'n{t}', round(w, 2)) for t, w in mem.R[i]]}")
        print(f"   gates open: {len(written)}/{G.K}; woke {['n%d' % w for w in woken]}; memory now {mem.n} nodes")
    rng = random.Random(0)
    for p_, t in quiz(art, rng, nq):
        q = G.query(p_)
        sel, _, _ = G.retrieve(mem, q)
        print(f"\nASK {tk.decode(p_)!r} (truth {tk.decode(t)!r})")
        for i in sel[:4]:
            print(f"   fetched n{i} (sentence {src.get(i)}): says {say(i)!r}")
        print(f"   LM with memory says {tk.decode(G.say(mem, p_, len(t)))!r}; LM alone says {tk.decode(G.say(Memory(), p_, len(t)))!r}")


@torch.no_grad()
def oracle(G, arts, nq=6):
    """read-path check: nodes made BY HAND from each sentence's exact token vectors (pieces of m tokens), keyed with the
    same query function a question uses; then the blanks. Tests fetch + the LM reading memory, without the write head"""
    G.eval()
    rng = random.Random(1)
    res = {k: [0., 0, 0] for k in ("LM alone", "LM + hand-made memory", "LM + text in context")}
    hit = n_hit = 0
    for a in arts:
        mem, src = Memory(), []
        for si, ids in enumerate(a["ids"]):
            for c in range(0, len(ids), G.m):
                piece = ids[c:c + G.m]
                mem.add(G.embed(piece), G.query(piece), [])
                src.append(si)
        ctx = [t for ids in a["ids"] for t in ids]
        for p, t in quiz(a, rng, nq):
            si = next(i for i, ids in enumerate(a["ids"]) if ids[:len(p)] == p)
            sel, _, _ = G.retrieve(mem, G.query(p))
            hit += any(src[i] == si for i in sel[:G.topk])
            n_hit += 1
            for name, m_, pre in (("LM alone", Memory(), []), ("LM + hand-made memory", mem, []), ("LM + text in context", Memory(), ctx)):
                nll, ac = G.say_nll(m_, pre + p, t)
                r = res[name]
                r[0] += float(nll)
                r[1] += ac
                r[2] += 1
    print(f"oracle read path (pieces of {G.m} tokens, order {G.order}): a blank's top-{G.topk} holds a piece of its own sentence {hit}/{n_hit}")
    for name, (nll, ac, n) in res.items():
        print(f"  {name:>22}: nll {nll / n:.3f} | token acc {ac / n:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task", nargs="?", default="e1")
    ap.add_argument("--data", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "bio", "species_e1.json"))
    ap.add_argument("--train-data", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "bio", "wiki_e1.json.gz"))  # any text (11-14: 469 articles were too few)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--heldout", type=int, default=60)
    ap.add_argument("--save", default="")
    ap.add_argument("--load", default="")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--m", type=int, default=16)    # rows (tokens) per node
    ap.add_argument("--together", type=int, default=1)   # articles read into one memory per training episode
    a = ap.parse_args()
    if a.task in ("v10", "v10test", "holddiag", "ctxdiag", "unit", "unit2", "routediag"):  # plain fp32 (11-15: TF32 made cached vs slow differ 4%; fp32 0.000)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    if a.task == "v10":                        # v10: meaning + control per token, skill experts, recursion, outcome rounds
        v10_selftest()
        v10_run(rounds=int(os.environ.get("ROUNDS", "6")))
        raise SystemExit
    if a.task == "unit":
        v10_unit()
        raise SystemExit
    if a.task == "v11test":
        v11_test()
        raise SystemExit
    if a.task == "v11t5":
        v11_t5(steps=int(os.environ.get("V11_T5_STEPS", "3000")))
        raise SystemExit
    if a.task == "v13":                        # §1d: GNM checks itself (perception, memory, ops, checks in one path)
        v13(steps=int(os.environ.get("V13_STEPS", "3000")))
        raise SystemExit
    if a.task == "v11t3":
        v11_t3()
        raise SystemExit
    if a.task == "routediag":
        v10_routediag(os.environ.get("V10_LOAD", "/marimo/out/v10w.pt"))
        raise SystemExit
    if a.task == "unit2":
        v10_unit_route()
        v10_unit_column()
        raise SystemExit
    if a.task == "ctxdiag":
        v10_ctxdiag(os.environ.get("V10_LOAD", "/marimo/out/v10s.pt"))
        raise SystemExit
    if a.task == "holddiag":
        v10_holddiag(os.environ.get("V10_LOAD", "/marimo/out/v10r.pt"))
        raise SystemExit
    if a.task == "dataprobe":
        v10_dataprobe()
        raise SystemExit
    if a.task == "readdiag":
        v10_readdiag(os.environ.get("V10_LOAD", "/marimo/out/v10o.pt"))
        raise SystemExit
    if a.task == "fitdiag":
        v10_fitdiag()
        raise SystemExit
    if a.task == "cachediag":
        v10_cachediag()
        raise SystemExit
    if a.task == "v10test":
        v10_selftest()
        raise SystemExit
    if a.task == "test1":                      # can the executor do ONE step? (frozen vs LoRA on 50 demos per kind)
        test1()
        raise SystemExit
    torch.manual_seed(0)
    random.seed(0)
    arts = load_articles(a.data)
    if a.smoke:
        arts = arts[:12]
        a.steps, a.heldout = 4, 4
    train, held = arts[:-a.heldout], arts[-a.heldout:]
    if os.path.exists(a.train_data) and not a.smoke:
        train = train + load_articles(a.train_data)
    print(f"E1: {len(arts)} articles ({len(train)} train, {len(held)} held out), LM {LM_NAME}, device {dev}", flush=True)
    G = GNM(m=a.m).to(dev)
    if a.load:
        G.load_state_dict(torch.load(a.load, map_location=dev))
    if a.task == "e1h":                        # §8: firing = computation, held-out sentence
        eval_e1h(G, held[:20])
        print("(above: before training)", flush=True)
        if a.steps:
            train_e1h(G, train, a.steps, log=max(1, min(100, a.steps // 5)))
        eval_e1h(G, held)
        if a.save:
            torch.save(G.state_dict(), a.save)
        raise SystemExit
    if a.task == "tracefire":
        trace_fire(G, held[0])
        if a.steps:
            train_e1h(G, train, a.steps, log=max(1, a.steps // 5))
            print("\n==================== after", a.steps, "training steps")
            trace_fire(G, held[0])
        raise SystemExit
    if a.task == "oracle":
        oracle(G, held)
        raise SystemExit
    if a.task == "inspect":
        inspect(G, held, train)
        raise SystemExit
    if a.task == "trace":                      # fire by hand: untrained, then after a short training
        trace(G, held[0])
        if a.steps:
            train_e1(G, train, a.steps, log=max(1, a.steps // 5))
            print("\n==================== after", a.steps, "training steps")
            trace(G, held[0])
        raise SystemExit
    eval_e1(G, held[:10 if not a.smoke else 2], show=0)
    print("(above: before training)", flush=True)
    train_e1(G, train, a.steps, log=200 if not a.smoke else 2, together=a.together)
    eval_e1(G, held, show=3 if not a.smoke else 1)
    eval_e1(G, held, together=20 if not a.smoke else 2)
    if a.save:
        torch.save({k: v for k, v in G.state_dict().items()}, a.save)
