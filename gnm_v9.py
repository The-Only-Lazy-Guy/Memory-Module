"""GNM v9 (docs/GNM_v9_design.md): GNM holds what is true and what to do -- facts, things, relations, state,
procedures -- as nodes the model writes and edits; the frozen LM (Walker 1B target, GPT-2 stands in) is the executor.
One loop: a change -> fetch related nodes -> LM step over them -> write nodes (new / edit) or say. No node-level step
data: signals are blanked spans recovered from memory, round trip, and (later) checked outcomes and own wins.
The world below is kept only as a source of checkable questions (later tests). Old paths: archive/gnm/.

  python gnm.py e1 --steps 3000 --save /marimo/ckpt/e1.pt     # reading: species articles in, blanked spans out
"""
import argparse
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
        self.keyc = nn.Linear(dc, dk)                    # a node's control rows -> its key
        self.done = nn.Linear(d, 1)                      # does the new node resolve the focus?
        self.ex = Experts(n=n_exp)
        self.EK = nn.Parameter(torch.zeros(n_exp, dk))   # each skill node's own learned key offset
        self.spawn_at = float(os.environ.get("SPAWN_AT", ".5"))
        self.skills, self.ltm, self._ltm_ids = [], [], set()   # skill nodes; kept memory (solved nodes)

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
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
            out = M(input_ids=x, attention_mask=mask[None, None].to(torch.bfloat16 if dev == "cuda" else torch.float32),
                    position_ids=pos, past_key_values=past, use_cache=cache, output_hidden_states=True)
        return out.hidden_states[-1][0].float(), out.logits[0].float(), (out.past_key_values if cache else None)

    def node_from(self, ids, H, parent=None, kind=""):
        C = self.ctl(H)                                  # control rows from the LM's states over the node's tokens
        key = F.normalize(self.qry(H.mean(0)) + self.keyc(C.mean(0)), dim=-1)   # asked-for like a query (11-14: 296 / 360
        #                                                                         found) + what its control adds
        return Node10(ids, C, key, parent, kind)

    def store(self, mem, text_ids, kind="fact"):
        H, _, _ = self.lm_run(text_ids)
        n = self.node_from(text_ids, H, kind=kind)
        mem.append(n)
        return n

    def query(self, ids):
        H, _, _ = self.lm_run(ids)
        return F.normalize(self.qry(H.mean(0)), dim=-1)

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

    @torch.no_grad()
    def _refresh(self, n):
        n.C = self.ctl(n.H)
        n.key = F.normalize(self.qry(n.H.mean(0)) + self.keyc(n.C.mean(0)), dim=-1)

    def refresh_ltm(self):
        """kept nodes' control and keys from their cached states with the current parts (they learn meanwhile)"""
        for n in self.ltm:
            self._refresh(n)

    def fetch(self, mem, q, exclude=(), src_skip=None):
        cands = [n for n in list(mem) + self.ltm
                 if n not in exclude and n.of not in exclude and not (src_skip is not None and n.src == src_skip)]
        if not cands:
            return [], []                                # (11-15: None crashed an empty fetch)
        sc = torch.stack([n.key for n in cands]) @ q
        top = sc.topk(min(self.topk, len(cands))).indices.tolist()
        p = F.softmax(sc / .1, 0)
        sel = sorted(top)                                # (write order)
        return [cands[i] for i in sel], [p[i] / p[i].detach() for i in sel]

    # ---------------------------------------------------------------- skill nodes (carry a LoRA expert)
    def skill_keys(self):
        """skill nodes' keys, like any node's (query + control of their label tokens) plus their own learned offset"""
        H = [s.H for s in self.skills]
        K = torch.stack([self.qry(h.mean(0)) + self.keyc(self.ctl(h).mean(0)) for h in H])
        K = K + self.EK[[s.expert for s in self.skills]]
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
    def context(self, mem, focus, train, src_skip=None):
        """the step's input: fetched nodes (meaning ids + their control as attention bias), the focus's chain of
        ancestors and its results so far, then the focus; the fetched skill. Returns ids, bias, expert, its weight,
        how many kept nodes were fetched"""
        q = self.query(focus.ids)
        chain, f = [], focus
        while f is not None:
            chain.append(f)
            f = f.parent
        chain = chain[::-1]
        excl = set(chain) | {c for x in chain for c in x.children}
        fetched, w = self.fetch(mem, q, exclude=excl, src_skip=src_skip)
        ids, bias = [], []
        for n, wi in zip(fetched, w):
            ids += n.ids
            bias.append(self.bias(n.C).squeeze(-1) * wi)
        for x in chain:                                  # the path to the focus, with each node's finished sub-steps
            ids += x.ids
            bias.append(self.bias(x.C).squeeze(-1))
            for c in x.children:
                if c.result is not None:
                    ids += c.ids + c.result.ids
                    bias += [self.bias(c.C).squeeze(-1), self.bias(c.result.C).squeeze(-1)]
        e, ew, esc = self.pick_skill(q, focus, train)
        return ids, (torch.cat(bias) if bias else None), e, ew, sum(n.kind == "kept" for n in fetched)

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
    def gen(self, ids, bias, temp=0., maxlen=16):
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
    def step(self, mem, focus, temp=0., maxlen=16):
        ids, bias, e, _, nk = self.context(mem, focus, False)
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
        x, y, z, w, v, x2 = (_nm(rng) for _ in range(6))
        facts = [f"{x} is in the genus {y}.", f"The genus {y} is in the family {z}.", f"The genus {w} is in the family {v}.",
                 f"{x2} is in the genus {w}."]
        rng.shuffle(facts)
        return facts, f"Which family is {x} in?", z, [(f"Which genus is {x} in?", 0), (y, 1),
                                                       (f"Which family is the genus {y} in?", 0), (z, 1), (z, 1)]
    names, facts, qtext, steps = _chain_world(rng, size)
    return facts, qtext(size, 0), names[size], steps(size, 0)


def v10_demo_loss(G, facts, q, steps, src=None):
    """imitate a trace with the same focus rules as solving (and the same solved-node writes); kept nodes written
    from this same trace are not fetched (they would hold its answers)"""
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


def keep_trace(G, q, steps, src):
    """a checked-right (or demo) trace's solved nodes go to kept memory"""
    tk = lm()[0]
    for f, r in trace_pairs(tk(" " + q)["input_ids"], [(tk(" " + t)["input_ids"], d) for t, d in steps]):
        G.keep(f + r, src)


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
    out, used = {}, {}
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
    return out, nk_steps / max(1, n_steps), used


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


def v10_run(rounds=6, n_demo=50, tries=4, per_round=200):
    tk = lm()[0]
    smoke = os.environ.get("V10_SMOKE") == "1"     # the whole path, tiny: demos, a round, the evals
    if smoke:
        rounds, per_round = 1, 3
    G = GNM10().to(dev)
    params = [p for nm, p in G.named_parameters() if p.requires_grad] + G.eparams()
    opt = torch.optim.AdamW(params, lr=1e-3)
    rng = random.Random(5)
    # demos up to 3 (11-15 v10b trace: with demos up to 2 the smaller question written after one step was always a
    # 1-step question, so "drop one step and ask again" for a longer one was never shown); tests stay longer (4, 6)
    train_sizes = [(k, s) for k in ("add", "chain") for s in (1, 2, 3)] + [("lookup", 1), ("compose", 1)]
    test_sizes = [("add", 2), ("add", 4), ("add", 6), ("chain", 2), ("chain", 4), ("chain", 6), ("lookup", 1), ("compose", 1)]
    demos = [v10_task(k, rng, s) for k, s in train_sizes for _ in range(n_demo // len(train_sizes))]
    seen_q = {d[1] for d in demos}
    for i, (facts, q, ans, st) in enumerate(demos):
        keep_trace(G, q, st, i)
    print(f"v10: {len(demos)} demo traces (sizes {train_sizes}); kept memory starts with their {len(G.ltm)} solved nodes; "
          f"test sizes {test_sizes} (add / chain longer than any demo)", flush=True)
    ne = 2 if smoke else 40

    def report(tag, res, fk, used, extra=""):
        print(f"{tag} | " + ", ".join(f"{k} {v:.0%}" for k, v in res.items())
              + f" | steps fetching a kept node {fk:.0%}; skills {len(G.skills)}; skills used per kind "
              + "; ".join(f"{k}: " + " ".join(f"{s}x{c}" for s, c in sorted(u.items())) for k, u in used.items()) + extra, flush=True)

    def fit(traces, steps):
        G.train()
        L = torch.zeros(())
        for st in range(steps):
            if st % 50 == 0:
                G.refresh_ltm()
            i = random.randrange(len(traces))
            facts, q, ans, tr = traces[i]
            L = v10_demo_loss(G, facts, q, tr, src=i)
            opt.zero_grad()
            L.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.)
            opt.step()
        return float(L)

    t0 = time.time()
    l = fit(demos, 5 if smoke else 600)
    print(f"round 0 (demos only): loss {l:.3f}, {time.time() - t0:.0f}s", flush=True)
    report("  with kept memory", *v10_eval(G, test_sizes, n=ne, show=1, skip_q=seen_q))
    report("  kept memory empty", *v10_eval(G, test_sizes, n=ne, use_ltm=False, skip_q=seen_q))
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
                if got == ans:                           # its own right run, as a trace to imitate (same focus rules)
                    tr = [(tk.decode(node.ids).strip(), int(done)) for focus, node, done in steps]
                    if all(x for x, _ in tr):
                        right.append(tr)
                # keep only when right in >= 2 tries (v10b round 1: a chain's answer is one of ~8 names, so one lucky
                # guess in 4 tries is common; kept guesses taught it to answer at once -- chain2 60 -> 22%)
                if len(right) >= 2 or len(right) + (tries - t - 1) < 2:
                    break
            if len(right) >= 2:
                kept.append((facts, q, ans, right[0]))
                keep_trace(G, q, right[0], len(kept) - 1)
                seen_q.add(q)
                got_right += 1
                seen.setdefault((kind, size), []).append(len(right[0]))
        print(f"round {r}: right runs found {got_right}/{per_round} (kept traces {len(kept)}, kept memory {len(G.ltm)} nodes), "
              f"{time.time() - t0:.0f}s trying", flush=True)
        print("  kept by kind/size (count, mean steps): " + ", ".join(
            f"{k}{s} {len(v)} {sum(v) / len(v):.1f}" for (k, s), v in sorted(seen.items())), flush=True)
        l = fit(kept, 5 if smoke else 400)
        last = r == rounds
        report(f"  loss {l:.3f}; with kept memory", *v10_eval(G, test_sizes, n=ne, show=1 if last else 0, skip_q=seen_q))
        if last:
            report("  kept memory empty", *v10_eval(G, test_sizes, n=ne, use_ltm=False, skip_q=seen_q))
            print("  " + v10_reuse(G, n=2 if smoke else 20), flush=True)


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
    G.pick_skill(G.query(ids), Node10(ids, None, None), True)
    for A, B in zip(G.ex.A, G.ex.B):
        B.data[0].normal_(0, .02)
    rb = torch.randn(len(ids), device=dev)
    G.ex.active = 0
    out, rows, hd = G.gen(ids, rb, 0., 10)
    slow = []
    for _ in range(10):
        _, lg, _ = G.lm_run(ids + slow, torch.cat([rb, torch.zeros(len(slow), device=dev)]))
        t = int(lg[-1].argmax())
        if t == tk.eos_token_id:
            break
        slow.append(t)
    Hs, _, _ = G.lm_run(ids + slow + [tk.eos_token_id], torch.cat([rb, torch.zeros(len(slow) + 1, device=dev)]))
    G.ex.active = None
    rel = float((Hs[-1] - hd).norm() / Hs[-1].norm())
    ok3 = out == slow and rel < .03
    print(f"selftest 3: cached == slow: {ok3} ({tk.decode(out)!r} vs {tk.decode(slow)!r}; done-state relative diff {rel:.4f})", flush=True)
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
    assert ok1 and ok2 and ok3 and ok4 and ok5, "v10 selftest failed"
    G.ltm, G._ltm_ids, G.skills = [], set(), []


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
    if a.task == "v10":                        # v10: meaning + control per token, skill experts, recursion, outcome rounds
        v10_selftest()
        v10_run(rounds=int(os.environ.get("ROUNDS", "6")))
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
