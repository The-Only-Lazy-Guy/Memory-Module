"""GNM v9 (docs/GNM_v9_design.md): GNM holds what is true and what to do -- facts, things, relations, state,
procedures -- as nodes the model writes and edits; the frozen LM (Walker 1B target, GPT-2 stands in) is the executor.
One loop: a change -> fetch related nodes -> LM step over them -> write nodes (new / edit) or say. No node-level step
data: signals are blanked spans recovered from memory, round trip, and (later) checked outcomes and own wins.
The world below is kept only as a source of checkable questions (later tests). Old paths: archive/gnm/.

  python gnm.py e1 --steps 3000 --save /marimo/ckpt/e1.pt     # reading: species articles in, blanked spans out
"""
import argparse
import json
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
        m = AutoModelForCausalLM.from_pretrained(LM_NAME).to(dev).eval()
        for p in m.parameters():
            p.requires_grad_(False)
        _LM.update(tk=tk, m=m)
    return _LM["tk"], _LM["m"]


class Memory:
    """the node store. A node = value rows V (vectors in the LM's embedding space: the LM reads them directly), key k
    (its learned label), relations R (the nodes read when it was written, with weights), strength s, optional text"""

    def __init__(self):
        self.V, self.k, self.R, self.s, self.text = [], [], [], [], []

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

    def __init__(self, K=4, m=2, dk=256, topk=8):
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
        self.new = nn.Parameter(torch.zeros(1))
        self.mark = nn.Parameter(torch.zeros(d))            # added to memory rows: "this is memory"
        with torch.no_grad():
            self.scale = nn.Parameter(M.transformer.wte.weight.norm(dim=-1).mean().reshape(1))
        nn.init.constant_(self.gate.bias, 1.)                # start by writing; the cost teaches restraint
        self.cost = float(os.environ.get("WRITE_COST", ".01"))

    # ---------------------------------------------------------------- the LM
    def embed(self, ids):
        return lm()[1].transformer.wte(torch.as_tensor(ids, device=dev))

    def run(self, embs):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
            out = lm()[1](inputs_embeds=embs[None], output_hidden_states=True)
        return out.hidden_states[-1][0].float(), out.logits[0].float()

    # ---------------------------------------------------------------- retrieve
    def query(self, ids):
        h, _ = self.run(self.embed(ids))
        return F.normalize(self.qry(h.mean(0)), dim=-1)

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
        for i in sel:
            w = p[i] / p[i].detach() if i in top else 1.
            rows.append(mem.V[i] * w + self.mark)
            owner += [i] * self.m
        return sel, torch.cat(rows), owner

    # ---------------------------------------------------------------- write (the loop's output)
    def write(self, mem, sel, owner, H, n_rows, train):
        """K slots attend over the step's LM states -> rows, key, gate, pointer (replace a fetched node / new),
        relations (attention mass on each fetched node). Train: every slot written, scaled by a straight-through
        gate (closed = zero rows, but the gate still learns what writing would have done). Eval: open slots only."""
        K, m, d = self.K, self.m, self.d
        A = F.softmax(self.slotq @ self.att(H).T / d ** .5, -1)          # (K*m, T)
        X = A @ H
        rows = (F.normalize(self.val(X), dim=-1) * self.scale).view(K, m, d)
        pooled = X.view(K, m, d).mean(1)
        g = torch.sigmoid(self.gate(pooled)).squeeze(-1)
        hard = (g > .5).float()
        g_st = hard + g - g.detach()
        k = F.normalize(self.key(pooled), dim=-1)
        choice = [None] * K
        if sel:                                                            # pointer: no gradient in E1 (L3 trains it)
            fk = torch.stack([mem.k[i] for i in sel]).detach()
            lg = torch.cat([F.normalize(self.ptr(pooled), dim=-1) @ fk.T / .1, self.new.expand(K, 1)], 1)
            choice = [int(c) for c in lg.argmax(-1)]
        mass = A[:, :n_rows].view(K, m, n_rows).sum(1) if n_rows else None
        written = []
        for j in range(K):
            if hard[j] == 0 and not train:
                continue
            R = []
            if mass is not None:
                for i in sorted(set(owner)):
                    w = float(mass[j, [t for t, o in enumerate(owner) if o == i]].sum())
                    if w > .1:
                        R.append((i, w))
            V = rows[j] * g_st[j]
            if choice[j] is not None and choice[j] < len(sel) and hard[j] == 1:
                mem.replace(sel[choice[j]], V, k[j], R)
                written.append(sel[choice[j]])
            else:
                written.append(mem.add(V, k[j], R))
        return written, self.cost * g.sum()

    def step(self, mem, ids, train=False):
        """one turn of the loop on a change given as text: retrieve -> LM step -> write. Returns the nodes written,
        the nodes woken (what the written nodes were written from: they are the next changes), the write cost"""
        sel, rows, owner = self.retrieve(mem, self.query(ids))
        embs = self.embed(ids) if rows is None else torch.cat([rows, self.embed(ids)])
        H, _ = self.run(embs)
        written, cost = self.write(mem, sel, owner, H, 0 if rows is None else len(rows), train)
        woken = sorted({i for w in written for i, _ in mem.R[w]} - set(written))
        return written, woken, cost

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
    for b in json.load(open(path, encoding="utf-8")):
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


def train_e1(G, arts, steps, lr=3e-4, nq=4, log=200):
    opt = torch.optim.AdamW([p for p in G.parameters() if p.requires_grad], lr=lr)
    rng = random.Random(0)
    t0, acc = time.time(), []
    for st in range(steps):
        art = rng.choice(arts)
        mem = Memory()
        cost, rt = read(G, mem, art, train=True)
        qs = quiz(art, rng, nq)
        l1 = sum(G.say_nll(mem, p, t)[0] for p, t in qs) / max(1, len(qs))
        loss = l1 + .3 * rt + cost
        opt.zero_grad()
        loss.backward()
        opt.step()
        acc.append((float(l1), float(rt), float(cost), mem.n))
        if (st + 1) % log == 0:
            a = torch.tensor(acc[-log:])
            print(f"step {st + 1}: blank nll {a[:, 0].mean():.3f}, round trip {a[:, 1].mean():.3f}, write cost {a[:, 2].mean():.3f}, "
                  f"nodes / article {a[:, 3].mean():.1f}, {time.time() - t0:.0f}s", flush=True)


@torch.no_grad()
def eval_e1(G, arts, seed=1, nq=6, together=1, show=0):
    """per quiz item, three readers of the same blank: the LM alone, the LM with the article(s) in context (when it
    fits), the LM with GNM's memory of the article(s). nll / token accuracy / exact span (greedy)"""
    tk = lm()[0]
    rng = random.Random(seed)
    res = {k: [0., 0., 0, 0] for k in ("LM alone", "LM + text in context", "LM + GNM memory")}
    nodes = []
    for b in range(0, len(arts) - together + 1, together):
        group = arts[b:b + together]
        mem = Memory()
        for a in group:
            read(G, mem, a)
        nodes.append(mem.n)
        ctx = [t for a in group for ids in a["ids"] for t in ids]
        for a in group:
            for p, t in quiz(a, rng, nq):
                for name, m_, pre in (("LM alone", Memory(), []), ("LM + text in context", Memory(), ctx), ("LM + GNM memory", mem, [])):
                    if len(pre) + len(p) + len(t) > 1000:
                        continue
                    nll, ac = G.say_nll(m_, pre + p, t)
                    ex = G.say(m_, pre + p, len(t)) == t
                    r = res[name]
                    r[0] += float(nll)
                    r[1] += ac
                    r[2] += ex
                    r[3] += 1
        if show and b < show * together:
            print(f"\n  -- {group[0]['title']}: {mem.n} nodes; what some say when the LM reads them alone:")
            for i in range(min(mem.n, 6)):
                print(f"     node {i}: {G.node_text(mem, i)!r}")
            p, t = quiz(group[0], random.Random(b), 1)[0]
            print(f"     blank: {tk.decode(p)!r} -> truth {tk.decode(t)!r} | memory says {tk.decode(G.say(mem, p, len(t)))!r} | "
                  f"alone says {tk.decode(G.say(Memory(), p, len(t)))!r}")
    print(f"\n{together} article(s) per memory, {len(nodes)} memories, {sum(nodes) / max(1, len(nodes)):.0f} nodes each:")
    for name, (nll, ac, ex, n) in res.items():
        if n:
            print(f"  {name:>22}: nll {nll / n:.3f} | token acc {ac / n:.3f} | exact span {ex}/{n}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task", nargs="?", default="e1")
    ap.add_argument("--data", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "bio", "species_e1.json"))
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--heldout", type=int, default=60)
    ap.add_argument("--save", default="")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    torch.manual_seed(0)
    random.seed(0)
    arts = load_articles(a.data)
    if a.smoke:
        arts = arts[:12]
        a.steps, a.heldout = 4, 4
    train, held = arts[:-a.heldout], arts[-a.heldout:]
    print(f"E1: {len(arts)} articles ({len(train)} train, {len(held)} held out), LM {LM_NAME}, device {dev}", flush=True)
    G = GNM().to(dev)
    eval_e1(G, held[:10 if not a.smoke else 2], show=0)
    print("(above: before training)", flush=True)
    train_e1(G, train, a.steps, log=200 if not a.smoke else 2)
    eval_e1(G, held, show=3 if not a.smoke else 1)
    eval_e1(G, held, together=20 if not a.smoke else 2)
    if a.save:
        torch.save({k: v for k, v in G.state_dict().items()}, a.save)
