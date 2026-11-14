"""GNM: graph node memory for Walker. User design: nodes are objects (value, label, relation, their own action); the
model makes its function nodes (one node kind, made when a want meets no label); signals (want x label) trigger them;
the network is recurrent -- each firing solves one small subproblem; ~50 natural-language step-by-step demos, then
outcome only (living). Everything that decides is learned. Still runtime code (said with every result): a question
arriving wants "find the next part"; an operation's compare comes first; a node keeps firing on the next log entry /
column / count; memory reads (the value of / before / both / the answer / take) have fixed labels.
PROPOSALS (Claude 11-14, not the user's design): digit columns by counting ("count up" / "count down" nodes); the
teacher's demos show each signed operation case once; tries deviate where a node doubts. History: docs/GNM.md section 8;
the v3 function machine (fm1-fm63) is in archive/gnm/gnm_v3_fm63.py.

  python gnm.py 5 40            # 5 demo worlds, then 40 worlds lived by itself; fresh worlds 7 / 8 before and after
  VECS=stand python gnm.py ...  # stand-in token vectors instead of GPT-2's
"""
import argparse
import hashlib
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


# ====================================================================== tokens and their vectors (the LM's, at the edges)
dev = "cuda" if torch.cuda.is_available() else "cpu"
VECS = os.environ.get("VECS", "wte")            # wte: GPT-2's token embeddings (11-14: as good or better than stand-in
#                                                vectors; contextual states hurt); stand: fixed random per token
d = 64 if VECS == "stand" else 768
tok = lambda s: re.findall(r"[A-Za-z]+|\d|[^\sA-Za-z\d]", s)
_LM, VEC, SEQ = {}, {}, {}


def lm():
    if not _LM:
        from transformers import AutoModel, AutoTokenizer
        _LM["tok"] = AutoTokenizer.from_pretrained("gpt2")
        _LM["m"] = AutoModel.from_pretrained("gpt2").to(dev).eval()
    return _LM["tok"], _LM["m"]


@torch.no_grad()
def vecs(ts):
    """one vector per word token (a word's GPT-2 pieces averaged)"""
    if VECS == "stand":
        for t in ts:
            if t not in VEC:
                g = torch.Generator().manual_seed(int(hashlib.md5(t.encode()).hexdigest()[:8], 16))
                VEC[t] = torch.randn(d, generator=g)
        return torch.stack([VEC[t] for t in ts]).to(dev)
    k = tuple(ts)
    if k in SEQ:
        return SEQ[k]
    tk, m = lm()
    ids, own = [], []
    for i, t in enumerate(ts):
        p = tk(("" if i == 0 else " ") + t)["input_ids"]
        ids += p
        own += [i] * len(p)
    x = torch.tensor([ids], device=dev)
    h = m.wte(x)[0] if VECS == "wte" else m(input_ids=x).last_hidden_state[0]
    own = torch.tensor(own, device=dev)
    out = F.layer_norm(torch.stack([h[own == i].mean(0) for i in range(len(ts))]), (d,))
    if len(SEQ) > 200000:
        SEQ.clear()
    SEQ[k] = out
    return out


def find(seq, sub, start=0):
    for i in range(start, len(seq) - len(sub) + 1):
        if seq[i:i + len(sub)] == sub:
            return i
    return -1


# ====================================================================== the translator (reader, namer)
RTAGS = ["O", "NAME", "NATTR", "VAL", "SRC", "RATTR"]   # a statement's fields
QTAGS = ["O", "PH", "A", "B"]                           # the next thing to resolve, and its two operands
SIGN = ["add", "sub_ab", "sub_ba", "take_a", "take_b"]   # what an operation does with the sizes (demos only)
P = 5                                                   # digits a number holds
JOIN = .8                                               # two labels this close: the same thing


class Seq(nn.Module):
    """reads a token sequence both ways; per-token tags and states"""

    def __init__(self, n_tags):
        super().__init__()
        self.inp = nn.Linear(d, 96)
        self.g = nn.GRU(96, 96, batch_first=True, bidirectional=True, num_layers=2)
        self.tag = nn.Linear(192, n_tags)

    def forward(self, ts):
        h, _ = self.g(F.gelu(self.inp(vecs(ts)))[None])
        return self.tag(h[0]), h[0]

    def many(self, tss):
        n = [len(t) for t in tss]
        x = nn.utils.rnn.pad_sequence([F.gelu(self.inp(vecs(t))) for t in tss], batch_first=True)
        pk = nn.utils.rnn.pack_padded_sequence(x, torch.tensor(n), batch_first=True, enforce_sorted=False)
        h, _ = self.g(pk)
        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True)
        lg = self.tag(h)
        return [(lg[i, :n[i]], h[i, :n[i]]) for i in range(len(tss))]


def name_lesson(M, groups):
    """the namer's lesson, no teacher needed: the same token group (a name said again, an attribute in a fact and in
    its rule, a copy's source, a question's phrase) together; different groups apart"""
    keys = sorted(set(groups))
    if len(keys) < 2:
        return 0
    lab = M.label([list(g) for g in groups])
    ids = torch.tensor([keys.index(g) for g in groups], device=dev)
    sim = lab @ lab.T / .05
    same = ids[:, None] == ids[None]
    eye = torch.eye(len(groups), dtype=torch.bool, device=dev)
    sim = sim.masked_fill(eye, -1e9)
    pos = same & ~eye
    has = pos.any(1)
    lp = sim.log_softmax(1)
    nce = -(lp * pos).sum(1)[has] / pos.sum(1)[has] if bool(has.any()) else lab.sum() * 0
    cos = lab @ lab.T
    return nce.mean() + F.relu(.9 - cos[pos]).mean() + F.relu(cos[~same] - .5).mean()


def as_num(ts):
    if not ts or any(t != "-" and not t.isdigit() for t in ts) or "-" in ts[1:]:
        return None
    neg = ts[0] == "-"
    ds = [int(t) for t in ts[1:] if t.isdigit()] if neg else [int(t) for t in ts]
    if not ds or len(ds) > P:
        return None
    return int(neg), [0] * (P - len(ds)) + ds          # (sign, digits top first)


def num_toks(sign, ds):
    while len(ds) > 1 and ds[0] == 0:
        ds = ds[1:]
    return (["-"] if sign and any(ds) else []) + [str(x) for x in ds]


# ====================================================================== the teacher (demos and grading only)
def col_lessons(a, b, op):
    """an operation's written-out working: digits compared top first, what it does with the sizes, the columns"""
    sa, da = a
    sb, db = b
    va, vb = int("".join(map(str, da))), int("".join(map(str, db)))
    st, steps = 0, []
    for x, y in zip(da, db):
        nst = st if st else (1 if x > y else 2 if y > x else 0)
        steps.append(("compare", x, y, st, 0, nst))
        st = nst
    A_, B_ = (-va if sa else va), (-vb if sb else vb)
    res = OPF[op](A_, B_)
    if op in ("max", "min"):
        what = "take_a" if res == A_ else "take_b"
    else:
        ea, eb = A_, (B_ if op == "add" else -B_)
        what = "add" if (ea >= 0) == (eb >= 0) else "sub_ab" if abs(ea) >= abs(eb) else "sub_ba"
    steps.append(("sign", sa, sb, ["add", "sub", "max", "min"].index(op), st, (int(res < 0), SIGN.index(what))))
    if what == "add":
        cy = 0
        for x, y in zip(reversed(da), reversed(db)):
            steps.append(("addcol", x, y, cy, (x + y + cy) % 10, (x + y + cy) // 10))
            cy = (x + y + cy) // 10
    elif what.startswith("sub"):
        hi, lo = (da, db) if what == "sub_ab" else (db, da)
        bw = 0
        for x, y in zip(reversed(hi), reversed(lo)):
            v = x - y - bw
            steps.append(("subcol", x, y, bw, v % 10, int(v < 0)))
            bw = int(v < 0)
    return steps, res


def stmt_tags(text, meta, w):
    ts = tok(text)
    tags = ["O"] * len(ts)
    if meta[0] == "rule":
        k = tok(words(meta[1]))
        i = find(ts, k)
        for j in range(i, i + len(k)):
            tags[j] = "RATTR"
        return ts, tags
    nm, k = meta[1], meta[2]
    name = tok(leaf(nm, k))
    i = find(ts, name)
    kw = tok(words(k))
    ka = find(ts, kw, i)
    for j in range(i, i + len(name)):
        tags[j] = "NATTR" if ka <= j < ka + len(kw) else "NAME"
    rest = ts[i + len(name):]
    j0 = i + len(name) + (2 if rest[:2] == ["is", "now"] else 1)
    j1 = ts.index("(", j0) if "(" in ts[j0:] else len(ts) - 1
    for j in range(j0, j1):
        tags[j] = "VAL"
    if "(" in ts[j0:]:
        for j in range(j1 + 3, len(ts) - 2):     # '( same as' ... ')' '.'
            tags[j] = "SRC"
    return ts, tags


class Teacher:
    def __init__(self, w):
        self.w, self.log, self.policy = w, {}, {}

    def heard(self, text, meta):
        if meta[0] == "rule":
            self.policy[meta[1]] = self.w.policy[meta[1]]
            return
        nm, k = meta[1], meta[2]
        lg = self.log.setdefault((nm, k), [])
        if "(same as" in text:
            act = "copy"
        elif text.startswith("Correction") or not lg or self.policy.get(k) != "add":
            act = "set"
        else:
            act = "add"
        lg.append(act)


def question_steps(E, w):
    """the demo steps of a question: (question tokens, phrase tokens or None, kind, operands, next question)"""
    was, out = E[0] == "before", []
    tell = lambda E2: f"What was {show(E2)} before?" if was else ask_text(E2)
    q = ask_text(E)
    for _ in range(60):
        x = unresolved(E)
        if x is None:
            out.append((tok(q), None, "answer", (tok(str(E[1])), None), None))
            return out
        if x[0] == "of" and x[2][0] == "val":
            E = swap(E, x, ("leaf", x[2][1], x[1]))
            continue
        if x[0] in ("leaf", "before", "both"):
            ph = tok(leaf(x[1], x[2]))
            if x[0] == "both":
                vs = w.multi.get((x[1], x[2]), [])
                if len(vs) != 2:
                    return None
                y = ("pair", vs[0], vs[1])
            else:
                y = ("val", truth_of(("leaf", x[1], x[2]) if x[0] == "leaf" else x, w))
            kind, ops = {"leaf": "lookup"}.get(x[0], x[0]), None
        else:
            if x[0] == "sumboth":
                a, b, opn = x[1][1], x[1][2], "add"
            else:
                a, b, opn = x[2][0][1], x[2][1][1], x[1]
            y = ("val", OPF[opn](a, b))
            ph = tok(show(x) if x is E else f"({show(x)})")
            kind, ops = opn, (tok(str(a)), tok(str(b)))
        E = y if x is E else swap(E, x, y)
        q2 = tell(E) if E[0] != "pair" else None
        out.append((tok(q), ph, kind, ops, q2))
        q = q2
    return None


def demo_world(seed, ticks, nq):
    rng = random.Random(seed)
    w = new_world(rng)
    T = Teacher(w)
    stm, fold, qs = [], [], []
    for _ in range(ticks):
        for dd in world_step(w, rng):
            o = dd()
            for text, meta in said(o, w) if o else []:
                stm.append(stmt_tags(text, meta, w) + (text,))
                T.heard(text, meta)
                if meta[0] == "fact":
                    fold.append((text, meta, T.log[(meta[1], meta[2])][-1], len(T.log[(meta[1], meta[2])]) > 1))
    for kind in ["lookup", "before", "coexist", "chain", "ops", "ops"] * (nq // 6):
        E = q_expr(kind, w, rng)
        if E is not None:
            st = question_steps(E, w)
            if st:
                qs.append(st)
    return stm, fold, qs


# ====================================================================== the nodes: one kind, made by the model
PURPOSE = {"scan": "find the next part of the question", "read": "read the next entry of the log",
           "take": "take the value it says", "look": "the value of", "before": "the value before",
           "both": "both values of", "answer": "the answer is", "add": "the sum of", "sub": "the difference between",
           "max": "the larger of", "min": "the smaller of", "cmp": "compare the digits", "next": "count up",
           "back": "count down"}
KEYS = list(PURPOSE)
MEMORY = ("take", "look", "before", "both", "answer")          # memory actions (reads), fixed labels
KIND2P = {"lookup": "look", "before": "before", "both": "both", "add": "add", "sub": "sub", "max": "max", "min": "min",
          "answer": "answer"}
THR = .99                                       # a want meets a label only when it is that want (11-14: at .9 GPT-2 put
#                                                 "the larger of" and "the smaller of" in one node)
USE = {"scan": ("tags", "want"), "read": ("want",), "add": ("state", "choice", "want"), "sub": ("state", "choice", "want"),
       "max": ("state", "choice", "want"), "min": ("state", "choice", "want"), "cmp": ("state",),
       "next": ("digit", "state"), "back": ("digit", "state")}             # what each kind of firing decides
NSTATE = {"cmp": 3}


@torch.no_grad()
def pvec(p):
    return F.normalize(vecs(tok(PURPOSE[p])).mean(0), dim=-1)


class GNode(nn.Module):
    """the one node kind: reads a token sequence (both ways), up to two token groups and a small state; says per-token
    tags, a digit, a state, a choice of order, and a want (what it fires next)"""

    def __init__(self):
        super().__init__()
        self.inp = nn.Linear(d, 96)
        self.g = nn.GRU(96, 96, batch_first=True, bidirectional=True)
        self.slot = nn.Linear(d, 96)
        self.st = nn.Embedding(8, 32)
        self.mix = nn.Sequential(nn.Linear(192 * 3 + 96 * 2 + 32, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU())
        self.tag = nn.Linear(192 * 3, len(QTAGS))      # (11-14: it decides from the part it marks; from the whole
        #                                                question's ends a long question's operation went to a lookup)
        self.digit, self.state, self.choice, self.want = nn.Linear(256, 10), nn.Linear(256, 8), nn.Linear(256, 4), nn.Linear(256, d)

    def forward(self, seqs, s1, s2, st):
        """a batch of firings: seqs (token lists or []), s1 / s2 (token lists or []), st (ints)"""
        n = len(st)
        Z = torch.zeros(96, device=dev)
        H, sm = [None] * n, {}
        idx = [i for i in range(n) if seqs[i]]
        if idx:
            x = nn.utils.rnn.pad_sequence([F.gelu(self.inp(vecs(seqs[i]))) for i in idx], batch_first=True)
            ln = [len(seqs[i]) for i in idx]
            pk = nn.utils.rnn.pack_padded_sequence(x, torch.tensor(ln), batch_first=True, enforce_sorted=False)
            h, _ = self.g(pk)
            h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True)
            for k, i in enumerate(idx):
                H[i] = h[k, :ln[k]]
                sm[i] = torch.cat([H[i][0], H[i][-1]])
        pool = lambda s: F.gelu(self.slot(vecs(s))).mean(0) if s else Z
        tags, focus = [None] * n, {}
        for i in idx:                                  # what it marks, then a pool over what it marks
            tags[i] = self.tag(torch.cat([H[i], sm[i].expand(len(H[i]), -1)], -1))
            wgt = 1 - tags[i].softmax(-1)[:, 0]
            focus[i] = (wgt[:, None] * H[i]).sum(0) / wgt.sum().clamp_min(1e-4)
        rows = [torch.cat([sm.get(i, torch.zeros(384, device=dev)), focus.get(i, torch.zeros(192, device=dev)), pool(s1[i]),
                           pool(s2[i]), self.st(torch.tensor(st[i], device=dev))]) for i in range(n)]
        m = self.mix(torch.stack(rows))
        return dict(m=m, tags=tags, digit=self.digit(m), state=self.state(m), choice=self.choice(m),
                    want=F.normalize(self.want(m), dim=-1))


class Module(nn.Module):
    def __init__(self):
        super().__init__()
        self.reader, self.namer = Seq(len(RTAGS)), Seq(1)          # the translator
        self.nodes = nn.ModuleList()
        self.labels = []                                           # each node's label (the want it was made for)
        self.mem = {p: pvec(p) for p in MEMORY}
        self.purpose_of = {}
        self.scan_want, self.read_want, self.cmp_want, self.back_want = pvec("scan"), pvec("read"), pvec("cmp"), pvec("back")

    def label(self, tss):
        return F.normalize(torch.stack([torch.cat([h[0], h[-1]], -1) for _, h in self.namer.many(tss)]), dim=-1)

    def route(self, want, make=False):
        """which node a want meets (its label), or a memory action; a new node when none meets it (while learning)"""
        names = list(self.mem) + [f"n{i}" for i in range(len(self.nodes))]
        labs = torch.stack([self.mem[p] for p in self.mem] + list(self.labels))
        s = labs @ want
        i = int(s.argmax())
        if float(s[i]) >= THR or not make:
            return names[i]
        self.nodes.append(GNode().to(dev))
        self.labels.append(want.detach().clone())
        return f"n{len(self.nodes) - 1}"


# ====================================================================== demos -> firings (each with the want it was for)
def step_f(key, a, w):
    """one count: a digit and a wrap flag -> the next / previous digit and the flag (set when it wraps)"""
    na = (a + 1) % 10 if key == "next" else (a - 1) % 10
    nw = int(w or (a == 9 if key == "next" else a == 0))
    return (key, [], [str(a)], [], w, dict(digit=na, state=nw)), na, nw


def column_f(key, x, y, c):
    """a column by counting (PROPOSAL, Claude 11-14): from x, the carry first, then y times while y counts down"""
    out, a, w = [], x, 0
    if c:
        f, a, w = step_f(key, a, w)
        out.append(f)
    k = y
    while True:
        f, k2, z = step_f("back", k, 0)
        out.append(f)
        if z:
            return out, a, w
        k = k2
        f, a, w = step_f(key, a, w)
        out.append(f)


def compare_f(x, y):
    out = []
    while True:
        f1, x2, zx = step_f("back", x, 0)
        f2, y2, zy = step_f("back", y, 0)
        out += [f1, f2]
        if zx or zy:
            return out, ("same" if zx and zy else "less" if zx else "more")
        x, y = x2, y2


def op_firings(a, b, op):
    """an operation's firings: digits compared (top first) by counting, the operation decides, columns by counting"""
    out = []
    cols, _ = col_lessons(a, b, op)
    st = 0
    for nm, x, y, s, o, ns in cols:
        if nm == "compare":
            ff, word = compare_f(x, y)
            out += ff + [("cmp", [], [word], [], s, dict(state=ns))]
            st = ns
        elif nm == "sign":
            sign, what = ns
            w = SIGN[what]
            want = "next" if w == "add" else "back" if w.startswith("sub") else "take"
            order = {"add": 0, "sub_ab": 0, "sub_ba": 1, "take_a": 0, "take_b": 1}[w]
            out.append((op, [], ["-" if x else "+"], ["-" if y else "+"], st, dict(state=sign, want=want, choice=order)))
        else:
            ff, dg, cy = column_f("next" if nm == "addcol" else "back", x, y, s)
            out += ff
    return out


# 11-14: an operation's decision (which count, order, sign) is op x sign x sign x compare = 48 cases; world questions
# rarely have a negative operand (5 demo worlds: the sub node saw 2 / 12 cases) -> the teacher shows each case once
SIGN_CASES = [(op, (sa, [0] * (P - 1) + [x]), (sb, [0] * (P - 1) + [y]))
              for op in ("add", "sub", "max", "min") for sa in (0, 1) for sb in (0, 1) for x, y in ((3, 3), (7, 2), (2, 7))]


def firings(data, signs=True):
    """every firing the demo worlds show, with its purpose: (purpose, seq, s1, s2, state, targets)"""
    out = []
    for stm, fold, qs in data:
        last_rule, idx = {}, 0
        for ts, tags, text in stm:                    # the log reader: one entry per firing
            if "RATTR" in tags:
                last_rule[tuple(t for t, g in zip(ts, tags) if g == "RATTR")] = ts
                continue
            _, meta, act, has = fold[idx]
            idx += 1
            r = last_rule.get(tuple(t for t, g in zip(ts, tags) if g == "NATTR"), [])
            out.append(("read", ts, list(r), [], int(has), dict(want={"set": "take", "add": "add", "copy": "look"}[act])))
        for steps in qs:                               # the question: one step per firing
            for q, ph, kind, ops, _ in steps:
                y = [0] * len(q)
                if ph is not None:
                    i = find(q, ph)
                    for j in range(i, i + len(ph)):
                        y[j] = 1
                    if ops:
                        ia = find(q, ops[0], i)
                        for j in range(ia, ia + len(ops[0])):
                            y[j] = 2
                        ib = find(q, ops[1], ia + len(ops[0]))
                        for j in range(ib, ib + len(ops[1])):
                            y[j] = 3
                else:
                    ia = find(q, ops[0])
                    for j in range(ia, ia + len(ops[0])):
                        y[j] = 2
                out.append(("scan", q, [], [], 0, dict(tags=y, want=KIND2P[kind])))
                if ops and kind in ("add", "sub", "max", "min"):
                    na, nb = as_num(ops[0]), as_num(ops[1])
                    if na and nb:
                        out += op_firings(na, nb, kind)
    if signs:
        for op, a, b in SIGN_CASES:
            out += op_firings(a, b, op)
    return out


def node_loss(M, ff):
    """each node learns the firings routed to it: what it decides, what it fires next"""
    allp = torch.stack([pvec(p) for p in KEYS])
    by, L = {}, 0
    for f in ff:
        by.setdefault(M.route(pvec(f[0])), []).append(f)
    for nm, fs in by.items():
        o = M.nodes[int(nm[1:])]([f[1] for f in fs], [f[2] for f in fs], [f[3] for f in fs], [f[4] for f in fs])
        for h in ("digit", "state", "choice"):
            k = [i for i, f in enumerate(fs) if h in f[5]]
            if k:
                L = L + F.cross_entropy(o[h][k], torch.tensor([fs[i][5][h] for i in k], device=dev))
        k = [i for i, f in enumerate(fs) if "want" in f[5]]
        if k:
            L = L + F.cross_entropy(o["want"][k] @ allp.T / .05, torch.tensor([KEYS.index(fs[i][5]["want"]) for i in k], device=dev))
        k = [i for i, f in enumerate(fs) if "tags" in f[5]]
        if k:
            L = L + F.cross_entropy(torch.cat([o["tags"][i] for i in k]), torch.tensor([t for i in k for t in fs[i][5]["tags"]], device=dev))
    return L


def train(M, worlds, epochs=30):
    data = [demo_world(1000 + i, 30, 24) for i in range(worlds)]
    fs_world = [firings([w]) for w in data]              # (each world's lessons carry the sign cases, as measured 11-14)
    for fs in fs_world:                                # the nodes the demos call for: one per want no label meets
        for f in fs:
            M.route(pvec(f[0]), make=True)
    M.purpose_of = {M.route(pvec(p)): p for p in PURPOSE}
    print(f"demos: {worlds} worlds, {sum(len(x) for x in fs_world)} firings; nodes made by the model: {len(M.nodes)} "
          f"({', '.join(sorted(n + '=' + p for n, p in M.purpose_of.items() if n.startswith('n')))})", flush=True)
    opt = torch.optim.AdamW(M.parameters(), lr=2e-3)
    for ep in range(epochs):
        tot = 0.
        for (stm, fold, qs), fs in zip(data, fs_world):
            outs = M.reader.many([ts for ts, _, _ in stm])          # the translator: fields and labels
            L = F.cross_entropy(torch.cat([lg for lg, _ in outs]),
                                torch.tensor([RTAGS.index(t) for _, tags, _ in stm for t in tags], device=dev))
            groups = []
            for ts, tags, text in stm:
                for fl in (("NAME", "NATTR"), ("NATTR",), ("SRC",), ("RATTR",)):
                    g = tuple(t for t, x in zip(ts, tags) if x in fl)
                    if g:
                        groups.append(g)
            for steps in qs:
                for q, ph, kind, ops, _ in steps:
                    if ph is not None and kind in ("lookup", "before", "both"):
                        groups.append(tuple(ph))
            rr = random.Random(len(groups))
            L = L + name_lesson(M, rr.sample(groups, min(400, len(groups)))) + node_loss(M, fs)
            opt.zero_grad()
            L.backward()
            opt.step()
            tot += float(L.detach())
        if (ep + 1) % 10 == 0:
            print(f"epoch {ep + 1}: loss {tot / len(data):.2f}", flush=True)
    return firings(data)


# ====================================================================== the memory and the run: every firing routed by a want
EXPLORE_U = 2.                                  # a try deviates at a firing with chance ~ U x (1 - its top probability)


class Memory:
    """places (a thing's name, its label, its log as said) and rules; a run is firings routed by wants. With T > 0
    (a try while living) a firing deviates where its node doubts, to any other choice evenly (11-14: a wrong column is
    sure of itself and gives the right digit ~0 -- its own odds never find it); every firing can be recorded"""
    LIMIT = 12

    def __init__(self, M, T=0.):
        self.M, self.places, self.rules = M, [], []
        self.t, self.labs, self.memo = 0, {}, {}
        self.T, self.rec = T, None

    # ---------------------------------------------------------------- relations: labels (the namer's)
    def label(self, ts):
        k = tuple(ts)
        if k not in self.labs:
            with torch.no_grad():
                self.labs[k] = self.M.label([ts])[0]
        return self.labs[k]

    def match(self, lab, labs):
        if not labs:
            return -1
        s = torch.stack(labs) @ lab
        i = int(s.argmax())
        return i if float(s[i]) > JOIN else -1

    def place_of(self, name, make=False):
        lab = self.label(name)
        i = self.match(lab, [p[1] for p in self.places])
        if i >= 0 or not make:
            return i
        self.places.append((name, lab, []))
        return len(self.places) - 1

    def rule_text_at(self, attr, t):
        if not attr:
            return ()
        i = self.match(self.label(attr), [r[1] for r in self.rules])
        if i < 0:
            return ()
        old = [ts for tt, ts in self.rules[i][2] if tt < t]
        return old[-1] if old else ()

    # ---------------------------------------------------------------- a firing
    def choose(self, lg):
        if not self.T:
            return int(lg.argmax())
        p = F.softmax(lg, -1)
        top = int(p.argmax())
        if random.random() < min(1., EXPLORE_U * (1 - float(p[top]))):
            others = [i for i in range(len(p)) if i != top]
            return random.choice(others) if others else top
        return top

    def fire(self, want, seq=(), s1=(), s2=(), st=0):
        nm = self.M.route(want)
        if nm in MEMORY:
            return nm, None
        o = self.M.nodes[int(nm[1:])]([list(seq)], [list(s1)], [list(s2)], [st])
        o = {k: (v[0] if v is not None else None) for k, v in o.items()}
        key = self.M.purpose_of.get(nm)
        if "tags" in USE.get(key, ()) and o["tags"] is None:
            return nm, None                            # (a want that meets a node with nothing for it to read)
        chosen = {}
        for h in USE.get(key, ()):
            if h == "tags":
                t = [self.choose(r) for r in o["tags"]]
                o["tags"] = F.one_hot(torch.tensor(t, device=dev), len(QTAGS)).float()
                chosen["tags"] = t
            elif h == "want":
                names = list(self.M.mem) + [f"n{i}" for i in range(len(self.M.nodes))]
                labs = torch.stack([self.M.mem[p] for p in self.M.mem] + list(self.M.labels))
                i = self.choose(labs @ o["want"] / .05)
                o["want"] = labs[i]
                chosen["want"] = self.M.purpose_of.get(names[i], names[i])
            else:
                n_ = {"state": NSTATE.get(key, 2), "choice": 2, "digit": 10}[h]
                c = self.choose(o[h][:n_])
                o[h] = F.one_hot(torch.tensor(c, device=dev), len(o[h])).float()
                chosen[h] = c
        if self.rec is not None:
            self.rec.append((key, list(seq), list(s1), list(s2), st, chosen))
        return nm, o

    # ---------------------------------------------------------------- operations: every count one firing
    def count(self, want, a, w):
        nm, o = self.fire(want, (), [str(a)], [], w)
        if o is None:
            return None
        return int(o["digit"][:10].argmax()), int(o["state"][:2].argmax())

    def compare_digits(self, x, y):
        for _ in range(self.LIMIT):
            zx, zy = self.count(self.M.back_want, x, 0), self.count(self.M.back_want, y, 0)
            if zx is None or zy is None:
                return None
            if zx[1] or zy[1]:
                return "same" if zx[1] and zy[1] else "less" if zx[1] else "more"
            x, y = zx[0], zy[0]
        return None

    def column(self, want, x, y, c):
        a, w = x, 0
        if c:
            r = self.count(want, a, w)
            if r is None:
                return None
            a, w = r
        k = y
        for _ in range(self.LIMIT):
            z = self.count(self.M.back_want, k, 0)
            if z is None:
                return None
            if z[1]:
                return a, w
            k = z[0]
            r = self.count(want, a, w)
            if r is None:
                return None
            a, w = r
        return None

    def compared(self, da, db):
        st = 0
        for x, y in zip(da, db):
            word = self.compare_digits(x, y)
            if word is None:
                return None
            nm, o = self.fire(self.M.cmp_want, (), [word], [], st)
            if o is None:
                return None
            st = int(o["state"][:3].argmax())
        return st

    def op(self, want, a, b):
        sa, da = a
        sb, db = b
        st = self.compared(da, db)
        if st is None:
            return None
        nm, o = self.fire(want, (), ["-" if sa else "+"], ["-" if sb else "+"], st)
        if o is None:
            return None
        sign, order = int(o["state"][:2].argmax()), int(o["choice"][:2].argmax())
        if self.M.route(o["want"]) == "take":
            s_, ds = (sa, da) if order == 0 else (sb, db)
            return num_toks(s_, ds)
        hi, lo = (da, db) if order == 0 else (db, da)
        out, cy = [], 0
        for x, y in zip(reversed(hi), reversed(lo)):
            r = self.column(o["want"], x, y, cy)
            if r is None:
                return None
            out.append(r[0])
            cy = r[1]
        return num_toks(sign, list(reversed(out)))

    def same(self, x, y):
        nx, ny = as_num(x), as_num(y)
        if nx and ny:
            return nx[0] == ny[0] and self.compared(nx[1], ny[1]) == 0
        return x == y

    # ---------------------------------------------------------------- hearing, reading, asking
    @torch.no_grad()
    def hear(self, text):
        self.t += 1
        ts = tok(text)
        lg, _ = self.M.reader(ts)
        tags = [RTAGS[i] for i in lg.argmax(-1).tolist()]
        field = lambda names: [t for t, g in zip(ts, tags) if g in names]
        self.memo = {}
        if "RATTR" in tags:
            attr = field({"RATTR"})
            lab = self.label(attr)
            i = self.match(lab, [r[1] for r in self.rules])
            if i >= 0:
                self.rules[i][2].append((self.t, ts))
            else:
                self.rules.append((attr, lab, [(self.t, ts)]))
            return
        name = field({"NAME", "NATTR"})
        if name:
            p = self.place_of(name, make=True)
            self.places[p][2].append(dict(t=self.t, ts=ts, val=field({"VAL"}), src=field({"SRC"}), attr=field({"NATTR"})))

    @torch.no_grad()
    def value(self, name, depth=0):
        """the read node fires once per log entry: take what it says / look at its source / an operation with it"""
        p = self.place_of(name) if name else -1      # (nothing named: nothing found)
        if p < 0 or depth > 6:
            return None, None, []
        if p in self.memo:
            return self.memo[p]
        state, prev, seen = None, None, []
        for e in self.places[p][2]:
            nm, o = self.fire(self.M.read_want, e["ts"], self.rule_text_at(e["attr"], e["t"]), (), int(state is not None))
            if o is None:
                continue
            act = self.M.route(o["want"])
            if act == "look":
                new = self.value(e["src"], depth + 1)[0]
            elif act == "take" or state is None:
                new = e["val"]
            else:
                na, nb = as_num(state), as_num(e["val"])
                new = self.op(o["want"], na, nb) if na and nb else e["val"]
            if new is None:
                continue
            if state is not None and not self.same(state, new):
                prev = state
            if not any(self.same(s, new) for s in seen):
                seen.append(new)
            state = new
        self.memo[p] = (state, prev, seen)
        return state, prev, seen

    @torch.no_grad()
    def ask(self, text, trace=None):
        q = tok(text)
        for _ in range(40):
            nm, o = self.fire(self.M.scan_want, q)
            if o is None:
                return None
            tags = [QTAGS[i] for i in o["tags"].argmax(-1).tolist()]
            ph = [i for i, g in enumerate(tags) if g in ("PH", "A", "B")]
            act = self.M.route(o["want"])
            if trace is not None:
                trace.append((" ".join(q), self.M.purpose_of.get(act, act), " ".join(q[i] for i in ph)))
            A_ = [q[i] for i, g in enumerate(tags) if g == "A"]
            B_ = [q[i] for i, g in enumerate(tags) if g == "B"]
            if act == "answer":
                return "".join(A_)
            if not ph:
                return None
            a0, a1 = ph[0], ph[-1]
            if act in ("look", "before", "both", "take"):
                cur, prev, seen = self.value(q[a0:a1 + 1])
                new = prev if act == "before" else (seen[0] + ["and"] + seen[1] if len(seen) >= 2 else None) if act == "both" else cur
            else:
                na, nb = as_num(A_), as_num(B_)
                new = self.op(o["want"], na, nb) if na and nb else None
            if not new:
                return None
            q = q[:a0] + new + q[a1 + 1:]
        return None


def hear_world(mem, seed, ticks=30):
    rng = random.Random(seed)
    w = new_world(rng)
    heard = []
    with torch.no_grad():
        for _ in range(ticks):
            for dd in world_step(w, rng):
                o = dd()
                for text, meta in said(o, w) if o else []:
                    mem.hear(text)
                    heard.append(text)
    return w, rng, heard


KINDS_EVAL = (("lookup", 0), ("before", 0), ("coexist", 0), ("chain", 0), ("ops", (1, 3)), ("long", (4, 6)), ("long12", (8, 12)))


@torch.no_grad()
def evaluate(M, seed, nq=48, show_fail=0, show_kind=None):
    """a fresh world (never in the demos): its facts held, and questions of every kind"""
    mem = Memory(M)
    w, rng, _ = hear_world(mem, seed)
    facts = [(nm, k, v) for nm, e in w.ents.items() for k, v in e.items()]
    held = sum((mem.value(tok(leaf(nm, k)))[0] or []) == tok(str(v)) for nm, k, v in facts)
    res, shown = {}, 0
    for kind, ops in KINDS_EVAL:
        hit = n = 0
        for _ in range(nq):
            E = q_expr("ops" if kind.startswith("long") else kind, w, rng, ops or (1, 3))
            if E is None:
                continue
            n += 1
            tr = []
            got = mem.ask(ask_text(E), tr)
            ok = got == str(truth_of(E, w))
            hit += ok
            if not ok and shown < show_fail and show_kind in (None, kind):
                shown += 1
                print(f"  [{kind}] {ask_text(E)} truth {truth_of(E, w)} got {got}" + "".join(f"\n      {a} | {b} | {c_}" for a, b, c_ in tr[:8]))
        res[kind] = (hit, n)
    return held, len(facts), res


# ====================================================================== living: after the demos, outcome only
def heard_groups(M, texts):
    """the names the module itself read (its reader's own tags) -- the namer's lesson needs no teacher (11-14: every
    wrong fact of the 5-demo model was two names merged in one place; living never touched the namer)"""
    groups = []
    with torch.no_grad():
        for text in texts:
            ts = tok(text)
            lg, _ = M.reader(ts)
            tags = [RTAGS[i] for i in lg.argmax(-1).tolist()]
            for fl in (("NAME", "NATTR"), ("NATTR",), ("SRC",), ("RATTR",)):
                g = tuple(t for t, x in zip(ts, tags) if x in fl)
                if g:
                    groups.append(g)
    return groups


def live(M, demo_fs, worlds, tries=12, nq=24, seed0=70000, n_demo=400):
    """each lived world: hear it; the namer learns its names; each question: the module's own run, then tries that
    deviate where a node doubts; the cheapest right run is imitated firing by firing, demo firings rehearsed alongside"""
    opt = torch.optim.AdamW([p for n in M.nodes for p in n.parameters()] + list(M.namer.parameters()), lr=1e-3)
    tally = {}
    for wi in range(worlds):
        mem = Memory(M)
        w, rng, heard = hear_world(mem, seed0 + wi)
        groups = heard_groups(M, heard)
        for _ in range(20):
            L = name_lesson(M, random.sample(groups, min(400, len(groups))))
            opt.zero_grad()
            L.backward()
            opt.step()
        mem = Memory(M)                                # heard again with what the namer learned
        with torch.no_grad():
            for text in heard:
                mem.hear(text)
        recs = []
        for kind, ops in (("lookup", 0), ("before", 0), ("coexist", 0), ("chain", 0), ("ops", (1, 3)), ("long", (4, 6))) * (nq // 6):
            E = q_expr("ops" if kind == "long" else kind, w, rng, ops or (1, 3))
            if E is None:
                continue
            truth = str(truth_of(E, w))
            c = tally.setdefault(kind, [0, 0, 0])
            c[2] += 1
            best = None
            for t in range(tries + 1):
                mem.T, mem.rec, mem.memo = (0. if t == 0 else 1.), [], {}
                got = mem.ask(ask_text(E))
                if got == truth and (best is None or len(mem.rec) < len(best)):
                    best = list(mem.rec)
                if t == 0:
                    c[0] += got == truth
            c[1] += best is not None
            mem.T, mem.rec, mem.memo = 0., None, {}
            if best:
                recs += best
        for _ in range(20 if recs else 0):
            L = node_loss(M, recs + random.sample(demo_fs, min(n_demo, len(demo_fs))))
            opt.zero_grad()
            L.backward()
            opt.step()
        if (wi + 1) % 5 == 0:
            print(f"lived {wi + 1} worlds: own run right / some run right: " +
                  ", ".join(f"{k} {a}/{n} ({b})" for k, (a, b, n) in tally.items()), flush=True)
            tally = {}


def report(M, tag, show=0, kind=None):
    for s in (7, 8):
        held, nf, res = evaluate(M, 50000 + s, show_fail=show, show_kind=kind)
        print(f"{tag}, fresh world {s}: facts held {held}/{nf} | " + ", ".join(f"{k} {a}/{b}" for k, (a, b) in res.items()), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("demo_worlds", type=int, nargs="?", default=5)
    ap.add_argument("live_worlds", type=int, nargs="?", default=40)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--show", type=int, default=0)     # failures printed per fresh world (with their steps)
    ap.add_argument("--show-kind", default=None)  # only this kind's failures
    ap.add_argument("--save", default="")       # a checkpoint path (on E:/WEIGHTS locally; never C:)
    a = ap.parse_args()
    torch.manual_seed(0)
    random.seed(0)
    M = Module().to(dev)
    t0 = time.time()
    demo_fs = train(M, a.demo_worlds, a.epochs)
    print(f"demos done {time.time() - t0:.0f}s", flush=True)
    report(M, "BEFORE living", a.show, a.show_kind)
    if a.live_worlds:
        live(M, demo_fs, a.live_worlds)
        report(M, "AFTER living", a.show, a.show_kind)
    if a.save:
        torch.save(dict(sd=M.state_dict(), n=len(M.nodes), labels=M.labels, purpose_of=M.purpose_of), a.save)
    print(f"{time.time() - t0:.0f}s", flush=True)
