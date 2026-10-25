"""GNM — Graph Node Memory module for Walker. THE single GNM file (design: docs/GNM.md; history: docs/GNM_history.md).

Sections:
  1. data       — the one world's input frames: JSON serialization into pairs, value-token positions, random JSON
  2. Walker     — frozen LM (1B checkpoint or a GPT-2 stand-in), token states
  3. node core  — a frame's pairs -> latent nodes (label, value + value slots, gain), read -> Walker's residual stream
  4. GNM        — world, live store, relation edges, tool nodes, Walker's pass loop, write side, training, eval

    python gnm.py --hf gpt2 --steps 3000 --init runs/aj_comb2.pt          # train on the one world
    python gnm.py --load runs/gnm_rw2.pt --viz runs/graph.html            # graph sanity: interactive memory graph
"""
import argparse
import copy
import io
import dataclasses
import json
import os
import math
import random
import re
import time
import collections
from collections import Counter as Counter_

try:                                    # Windows: pyarrow must precede torch
    import pyarrow.dataset  # noqa: F401
except Exception:
    pass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ====================================================================================================================
# 1. data
# ====================================================================================================================
def words(key):
    """Key -> lower-case words: camelCase / snake_case / kebab-case / dotted / Title Case."""
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(key))
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", s)
    s = re.sub(r"[_\-./:]+", " ", s)
    return " ".join(s.lower().split())


def fmt_num(x):
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, int):
        return str(x)
    if abs(x - round(x)) < 1e-9 and abs(x) < 1e15:
        return str(int(round(x)))
    return f"{x:.4f}".rstrip("0").rstrip(".")


def scalar(v):
    if v is None:
        return "null"
    if isinstance(v, (bool, int, float)):
        return fmt_num(v)
    return " ".join(str(v).split())


def _is_scalar(v):
    return v is None or isinstance(v, (str, int, float, bool))


def _is_image(v):
    return hasattr(v, "size") and hasattr(v, "convert")          # PIL image


_ID_WORDS = ("id", "name", "title", "label", "key", "code", "sku", "email", "username", "slug", "handle")


def _short_name(v):
    """A value that can NAME a record: a short string (1-3 words), not a number or a boolean."""
    if not isinstance(v, str) or not v.strip() or len(v) > 30 or len(v.split()) > 3:
        return False
    try:
        float(v.replace(",", ""))
        return False
    except ValueError:
        return True


SHAPE_LABELS = True        # False = id/name-like keys only (checkpoints trained before 2026-10-12, `796fd38`)


def record_label(d, max_fields=2, in_list=False):
    """How a record can be named: id/name-like fields first (by key), then any other short text field
    (by value shape: `trial = T2`, `app = Editor`), at most `max_fields`. Returns (label text,
    primary value)."""
    cands = []
    for j, (k, v) in enumerate(d.items()):
        if not _is_scalar(v) or v in (None, "") or isinstance(v, bool):
            continue
        w = words(k).split()
        if w and w[-1] in _ID_WORDS:
            cands.append((_ID_WORDS.index(w[-1]), j, k, v))
        elif SHAPE_LABELS and in_list and _short_name(v):   # list elements are records: name them by shape
            cands.append((len(_ID_WORDS), j, k, v))
    cands.sort()
    picked = cands[:max_fields]
    if not picked:
        return "", ""
    if not SHAPE_LABELS:
        picked = picked[:1]
    return "; ".join(f"{words(k)} = {scalar(v)}" for _, _, k, v in picked), scalar(picked[0][3])


def _path_words(path, max_words=10):
    parts = [f"#{p + 1}" if isinstance(p, int) else words(p) for p in path]
    s = " ".join(x for x in parts if x)
    if len(s.split()) > max_words and len(parts) > 4:
        s = " ".join([parts[0], "..."] + parts[-3:])
    return s


@dataclasses.dataclass
class Leaf:
    path: tuple
    value: str
    raw: object
    label: str          # "name = web-3" of the nearest enclosing record ("" if none)
    label_value: str
    context: list       # [(key words, value)] scalar siblings
    image: object = None


def flatten(obj, path=(), labels=(), out=None):
    """Every scalar leaf of a JSON value, with record labels and sibling context. Lists also
    yield a `count` leaf, and short scalar lists a joined summary leaf."""
    out = [] if out is None else out
    if isinstance(obj, dict):
        lab, lab_v = record_label(obj, in_list=bool(path) and isinstance(path[-1], int))
        labs = labels + ((lab, lab_v),) if lab else labels
        sib = [(words(k), scalar(v)) for k, v in obj.items() if _is_scalar(v)]
        if not obj:
            out.append(Leaf(path, "empty", None, *(labels[-1] if labels else ("", "")), []))
        for k, v in obj.items():
            p = path + (str(k),)
            if _is_image(v):
                out.append(Leaf(p, "[image]", None, *(labs[-1] if labs else ("", "")),
                                [s for s in sib][:6], image=v))
            elif _is_scalar(v):
                ctx = [s for s in sib if s[0] != words(k)]
                out.append(Leaf(p, scalar(v), v, *(labs[-1] if labs else ("", "")), ctx))
            else:
                flatten(v, p, labs, out)
    elif isinstance(obj, list):
        out.append(Leaf(path + ("count",), str(len(obj)), len(obj), *(labels[-1] if labels else ("", "")), []))
        if obj and all(_is_scalar(v) for v in obj) and len(obj) <= 12:
            joined = ", ".join(scalar(v) for v in obj)
            if len(joined) <= 120:
                out.append(Leaf(path, joined, None, *(labels[-1] if labels else ("", "")), []))
        for i, v in enumerate(obj):
            if _is_scalar(v):
                out.append(Leaf(path + (i,), scalar(v), v, *(labels[-1] if labels else ("", "")), []))
            else:
                flatten(v, path + (i,), labels, out)
    else:
        out.append(Leaf(path, scalar(obj), obj, "", "", []))
    return out


_SYL = ("ka", "lo", "mer", "vin", "tor", "sa", "ne", "ril", "dun", "ba", "zek", "or", "pel", "mu", "tha", "gor")


def _name(rng, n=2, used=None):
    while True:
        w = "".join(rng.choice(_SYL) for _ in range(n)).capitalize()
        if used is None or w not in used:
            if used is not None:
                used.add(w)
            return w


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


def _subject(lf, doc):
    """How a question names a leaf: its key, its record (label or #i in list), and its parent path."""
    path = list(lf.path)
    key = words(path[-1]) if isinstance(path[-1], str) else f"item #{path[-1] + 1}"
    rest = path[:-1]
    where = ""
    if lf.label_value and any(isinstance(p, int) for p in rest):
        li = max(i for i, p in enumerate(rest) if isinstance(p, int))
        where = f"{lf.label_value} in {_path_words(rest[:li])}" + (
            f" {_path_words(rest[li + 1:])}" if rest[li + 1:] else "")
    elif rest:
        where = _path_words(rest)
    return key, where.strip()


_T_LOOK = {"train": ("What is the {key} of {where} in {doc}?", "In {doc}, {where}: what is the {key}?",
                     "Tell me the {key} for {where} ({doc}).", "Find the {key} of {where} ({doc}).",
                     "For {where} in {doc}, what {key} is stored?"),
           "eval": ("{doc} -> {where} -> {key}?", "Which {key} does {where} have in {doc}?",
                    "Give me the value of {key} under {where} in {doc}.")}


_T_TOP = {"train": ("What is the {key} in {doc}?", "Which {key} does {doc} list?"), "eval": ("Give the {key} recorded in {doc}.",)}


def _doc_name(rng):
    return f"{_name(rng, 2).lower()}_{rng.randint(1, 99)}.json"


def serialize(obj, doc, vprefix=None):
    """One JSON document as text pieces. Returns pieces and, per scalar leaf, (path, value, piece, pointer pieces)
    where the pointer pieces are the document name, every ancestor key and the enclosing record's label pair.
    vprefix (a list): filled per leaf with the text of its piece BEFORE the value (the value's tokens = the rest)."""
    pieces = [f' {json.dumps(doc)}: {{']
    leaves = []

    def emit(o, path, anc):
        if isinstance(o, dict):
            start = len(leaves)
            # the record's label = the hub's record_label rule (id/name-like fields, or a short text field in a list)
            lab, _ = record_label(o, in_list=bool(path) and isinstance(path[-1], int))
            lab_pairs = set(lab.split("; ")) if lab else set()
            labels = []
            for j, (k, v) in enumerate(o.items()):
                sep = "," if j else ""
                if _is_scalar(v):
                    pieces.append(f'{sep} {json.dumps(str(k))}: {json.dumps(v)}')
                    leaves.append([path + (str(k),), scalar(v), len(pieces) - 1, list(anc)])
                    if vprefix is not None:
                        vprefix.append(f'{sep} {json.dumps(str(k))}:')
                    if f"{words(k)} = {scalar(v)}" in lab_pairs:
                        labels.append(len(pieces) - 1)
                else:
                    pieces.append(f'{sep} {json.dumps(str(k))}:')
                    kidx = len(pieces) - 1
                    pieces.append(" {" if isinstance(v, dict) else " [")
                    emit(v, path + (str(k),), anc + [kidx])
                    pieces.append(" }" if isinstance(v, dict) else " ]")
            for lf in leaves[start:]:                       # the record's label is part of every member's pointer
                lf[3].extend(x for x in labels if x != lf[2] and x not in lf[3])
        elif isinstance(o, list):
            for i, v in enumerate(o):
                sep = "," if i else ""
                if _is_scalar(v):
                    pieces.append(f'{sep} {json.dumps(v)}')
                    leaves.append([path + (i,), scalar(v), len(pieces) - 1, list(anc)])
                    if vprefix is not None:
                        vprefix.append(sep)
                else:
                    pieces.append(f'{sep} {{' if isinstance(v, dict) else f'{sep} [')
                    emit(v, path + (i,), anc)
                    pieces.append(" }" if isinstance(v, dict) else " ]")
    emit(obj, (), [0])
    pieces.append(" }")
    return pieces, leaves


def kind_of(v):
    import re
    if "@" in v:
        return "email"
    if v.startswith("http"):
        return "url"
    if re.fullmatch(r"20\d\d-\d\d-\d\d", v):
        return "date"
    if re.fullmatch(r"[A-Z]\d+-[A-Z]", v):
        return "code"
    if re.fullmatch(r"-?\d+(\.\d+)?", v):
        return "number"
    if v in ("true", "false", "null"):
        return "bool/null"
    return "text"


def perturb(obj, rng, pool, in_list=False):
    """Same structure, keys and record labels; every other scalar gets another value of the same kind."""
    if isinstance(obj, dict):
        lab, _ = record_label(obj, in_list=in_list)
        keep = set(lab.split("; ")) if lab else set()
        out = {}
        for k, v in obj.items():
            if _is_scalar(v):
                if f"{words(k)} = {scalar(v)}" in keep:
                    out[k] = v
                else:
                    cands = [x for x in pool.get(kind_of(scalar(v)), []) if scalar(x) != scalar(v)]
                    out[k] = rng.choice(cands) if cands else v
            else:
                out[k] = perturb(v, rng, pool, in_list=isinstance(v, list))
        return out
    if isinstance(obj, list):
        return [perturb(v, rng, pool, in_list=True) if not _is_scalar(v) else
                (rng.choice([x for x in pool.get(kind_of(scalar(v)), [v]) if scalar(x) != scalar(v)] or [v]))
                for v in obj]
    return obj


def doc_encoding(tok_ids, decode, obj, name):
    """One document: ids, and per leaf its span (pair + pointer chain), pointer positions and value-token positions
    (slot t = value token t: a string's opening-quote token is dropped). Returns (enc dict, leaves)."""
    vp = []
    pieces, leaves = serialize(obj, name, vp)
    ids, rt = [], []
    for pc in pieces:
        t = tok_ids(pc)
        rt.append((len(ids), len(ids) + len(t)))
        ids += t
    spans = [sorted({p for a in anc for p in range(*rt[a])} | set(range(*rt[pi]))) for _, _, pi, anc in leaves]
    ptr = [sorted({p for a in anc for p in range(*rt[a])}) for _, _, _, anc in leaves]
    vpos = []
    for (_, _, pi, _), pre in zip(leaves, vp):
        pt = tok_ids(pieces[pi])
        nv = len(pt) - len(tok_ids(pre)) if pre else len(pt)
        v1 = list(range(rt[pi][1] - nv, rt[pi][1]))
        if v1 and decode([pt[len(pt) - nv]]).strip() == '"':
            v1 = v1[1:]
        if v1 and decode([pt[-1]]).strip() == '"':        # the closing quote is not part of the value either (it was
            v1 = v1[:-1]                                   # a junk slot after every string value)
        vpos.append(v1)
    return {"ids": ids, "spans": spans, "ptr": ptr, "vpos": vpos}, leaves


# ====================================================================================================================
# 2. Walker + 3. node core
# ====================================================================================================================
class SmallWalker(nn.Module):
    """GPT-2 as a small local stand-in for Walker: same interface the GNM uses (.blocks, .emb, .r_end, forward(ids,
    r_active=False) -> (logits, loss)); plain forward (the loop never runs Walker's R passes)."""

    def __init__(self, name="gpt2"):
        super().__init__()
        from transformers import GPT2LMHeadModel
        self.hf = GPT2LMHeadModel.from_pretrained(name)
        self.blocks = self.hf.transformer.h
        self.emb = self.hf.transformer.wte
        self.r_end = max(2, (2 * len(self.blocks)) // 3)

    def forward(self, idx, targets=None, r_active=False):
        pos = torch.arange(idx.shape[1], device=idx.device)[None]
        x = self.emb(idx) + self.hf.transformer.wpe(pos)
        for blk in self.blocks:
            x = blk(x)
            x = x[0] if isinstance(x, tuple) else x
        logits = self.hf.lm_head(self.hf.transformer.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
        return logits, loss


# ------------------------------------------------------------------ the frozen LM
def load_walker(hf="gpt2", ckpt="", enc_layer=0, dev=None):
    """Walker (1B checkpoint) or a small HF stand-in, frozen. `model.r_end` = the encoder layer (states at its input)."""
    import train_walker_1b_continue as T  # the 1B builder + tokenizer (the official trainer)
    if ckpt:
        model, raw_tok, _cfg = T.build_model_for_inference(ckpt, device=dev)
        model = model.to(dev).eval()
        if getattr(model, "r_adapter", None) is None or not int(getattr(model, "r_end", 0) or 0):
            model.r_end = int(enc_layer) or (2 * len(model.blocks)) // 3
        print(f"Walker 1B from {ckpt}: {len(model.blocks)} blocks, d={model.emb.weight.shape[1]}, "
              f"encoder states at block {model.r_end}", flush=True)
    else:
        model = SmallWalker(hf).to(dev).eval()
        raw_tok = T._load_tokenizer(hf)
        if enc_layer:
            model.r_end = int(enc_layer)
    for p_ in model.parameters():
        p_.requires_grad_(False)
    hf_cfg = getattr(getattr(model, "hf", None), "config", None)
    model.max_pos = int(getattr(hf_cfg, "n_positions", 0) or getattr(model, "max_pos", 0) or 10 ** 9)
    return model, raw_tok


class Tok:
    """Memoized token ids (no special tokens)."""

    def __init__(self, tok):
        self.tok, self._memo = tok, {}

    def ids(self, text):
        r = self._memo.get(text)
        if r is None:
            r = self._memo[text] = list(self.tok(text, add_special_tokens=False)["input_ids"])
        return list(r)

    def decode(self, ids):
        return self.tok.decode(ids)


class _Stop(Exception):
    pass


def walker_states(model, ids, layer=None):
    """Walker's token states at the input of block `layer` (default model.r_end), bf16 autocast, no grad."""
    layer = model.r_end if layer is None else layer
    got = {}

    def pre(_m, args, kwargs):
        got["x"] = args[0] if args else kwargs["x"]
        raise _Stop
    h = model.blocks[layer].register_forward_pre_hook(pre, with_kwargs=True)
    try:
        with torch.no_grad(), torch.autocast(ids.device.type, dtype=torch.bfloat16, enabled=ids.device.type == "cuda"):
            model(ids, r_active=False)
    except _Stop:
        pass
    finally:
        h.remove()
    return got["x"].float()




# ------------------------------------------------------------------ the memory
class Pool(nn.Module):
    """Learned attention pooling over a token span (zero-init scorer = mean at start)."""

    def __init__(self, d):
        super().__init__()
        self.s = nn.Linear(d, 1)
        nn.init.zeros_(self.s.weight)
        nn.init.zeros_(self.s.bias)

    def forward(self, H, m):
        w = torch.softmax(self.s(H).squeeze(-1).masked_fill(~m, -1e4), -1)
        return (w[..., None] * H).sum(1)

    def flat(self, Hf, seg, n):
        """Same pooling over unpadded spans: Hf (L, d) all span tokens, seg (L,) span id, n spans."""
        sc = self.s(Hf).squeeze(-1)
        mx = torch.full((n,), -1e30, device=Hf.device, dtype=sc.dtype).scatter_reduce(0, seg, sc.detach(), "amax")
        e = (sc - mx[seg]).exp()
        den = torch.zeros(n, device=Hf.device, dtype=sc.dtype).index_add(0, seg, e)
        return torch.zeros(n, Hf.shape[1], device=Hf.device, dtype=Hf.dtype).index_add(0, seg, (e / den[seg])[:, None] * Hf)


def _gather(X, spans):
    import numpy as np
    n = max(len(q) for q in spans)
    idx = np.zeros((len(spans), n), dtype=np.int64)
    m = np.zeros((len(spans), n), dtype=bool)
    for i, q in enumerate(spans):
        idx[i, :len(q)] = q
        m[i, :len(q)] = True
    return X[torch.from_numpy(idx).to(X.device)], torch.from_numpy(m).to(X.device)


class Memory(nn.Module):
    """One unit per fact. Key = Wk(pool(pair + pointer chain)), value = Wv(pool(...)), gain g = write gate, value
    slots = the value's own token states. Address: softmax(tau * cos(q, key) + log g)."""

    def __init__(self, d, dk=512, vslots=8, tau0=10.0):
        super().__init__()
        di = 2 * d                                      # Walker state stream + token-embedding stream
        self.dk, self.vslots = dk, vslots
        self.tau = nn.Parameter(torch.tensor(math.log(tau0)))
        self.lne = nn.LayerNorm(d)
        self.ln = nn.LayerNorm(d)
        for name in ("mu", "mu_q"):
            self.register_buffer(name, torch.zeros(d))
        for name in ("var", "var_q"):
            self.register_buffer(name, torch.ones(d))
        for name in ("seen", "seen_q"):
            self.register_buffer(name, torch.zeros(()))
        self.pk, self.pv = Pool(di), Pool(di)
        self.Wk, self.Wv = nn.Linear(di, dk), nn.Linear(di, dk)
        self.qdir = nn.Parameter(torch.zeros(di))      # question-pool direction (old hop-0 control, folded)
        self.Wg = nn.Linear(di, 1)
        nn.init.zeros_(self.Wg.weight)
        nn.init.constant_(self.Wg.bias, 3.0)
        self.beta = nn.Parameter(torch.tensor(2.0))
        self.Wo = nn.Linear(dk, d)
        self.seg = nn.Parameter(torch.zeros(di))
        if vslots:
            self.Wvs = nn.Linear(di, dk)
            self.slot_e = nn.Parameter(torch.zeros(vslots, dk))

    def _std(self, X, q):
        mu, var, seen = (self.mu_q, self.var_q, self.seen_q) if q else (self.mu, self.var, self.seen)
        if self.training and (getattr(self, "update_stats", False) or float(seen) == 0):
            with torch.no_grad():               # slow, and only on real training data (not warm-up / eval / probes):
                f = 0.0 if float(seen) == 0 else 0.9995   # fast stats tracked the current world -> first frame broke
                mu.mul_(f).add_((1 - f) * X.mean(0))
                var.mul_(f).add_((1 - f) * X.var(0))
                seen.add_(1)
        return mu, var

    def write(self, X, E, spans, ptr=None, vpos=None):
        """X, E: (T, d) Walker states and token embeddings of the text (one document or many concatenated). Returns
        per unit (kraw, v, logg, slots, g). Pooling runs over the unpadded span tokens (flat segments)."""
        import numpy as np
        mu, var = self._std(X, False)
        Xs = torch.cat([self.ln((X - mu) / (var + 1e-5).sqrt()), self.lne(E)], -1)
        n = len(spans)
        lens = np.fromiter((len(q) for q in spans), dtype=np.int64, count=n)
        pos = np.concatenate([np.asarray(q, dtype=np.int64) for q in spans]) if n else np.zeros(0, np.int64)
        seg_np = np.repeat(np.arange(n), lens)
        dev = Xs.device
        seg = torch.from_numpy(seg_np).to(dev)
        Hf = Xs[torch.from_numpy(pos).to(dev)]
        if ptr is not None:
            isp = np.concatenate([np.isin(np.asarray(q), ps) if ps else np.zeros(len(q), bool)
                                  for q, ps in zip(spans, ptr)]) if n else np.zeros(0, bool)
            Hf = Hf + torch.from_numpy(isp.astype(np.float32)).to(dev)[:, None] * self.seg
        self.last_flat = (Hf, seg, n)                       # reused by extra heads (relation pooling)
        kraw = self.Wk(self.pk.flat(Hf, seg, n))
        pv = self.pv.flat(Hf, seg, n)
        v = self.Wv(pv)
        g = torch.sigmoid(self.Wg(pv)).squeeze(-1) * torch.sigmoid(self.beta)
        slots = None
        if self.vslots and vpos is not None:
            vl = np.fromiter((min(len(q), self.vslots) for q in vpos), dtype=np.int64, count=n)
            S = max(1, int(vl.max()) if n else 1)          # real value lengths (padding to the cap was waste)
            vp = np.concatenate([np.asarray(q[:S], dtype=np.int64) for q in vpos]) if vl.sum() else np.zeros(0, np.int64)
            row = np.repeat(np.arange(n), vl)
            col = np.concatenate([np.arange(k) for k in vl]) if vl.sum() else np.zeros(0, np.int64)
            sv = self.Wvs(Xs[torch.from_numpy(vp).to(dev)])
            slots = torch.zeros(n, S, sv.shape[-1], device=dev, dtype=sv.dtype)
            r_, c_ = torch.from_numpy(row).to(dev), torch.from_numpy(col).to(dev)
            slots = slots.index_put((r_, c_), sv + self.slot_e[c_])
        return kraw, v, torch.log(g + 1e-6), slots, g

    def query(self, Hq, mq, Eq, fixed=False, score_add=None):
        """Question token states (B, T, d) -> raw query (B, dk), in the key space (tied pool + head)."""
        X = Hq[mq]
        if self.training and not fixed:
            mu, var = self._std(X, True)
        else:
            mu, var = self.mu_q, self.var_q
        Hs = torch.cat([self.ln((Hq - mu) / (var + 1e-5).sqrt()), self.lne(Eq)], -1)
        sc = self.pk.s(Hs).squeeze(-1) + Hs @ self.qdir
        if score_add is not None:                       # an extra pooling score (the reference head's own)
            sc = sc + score_add(Hs).squeeze(-1)
        w = torch.softmax(sc.masked_fill(~mq, -1e4), -1)
        return self.Wk((w[..., None] * Hs).sum(1))

    def attend(self, qraw, K, logg, kmean):
        """qraw (B, dk); K (N, dk) normalized centred keys; logg (N,) or (B, N) incl. -1e4 masks; kmean (B, dk)."""
        qn = F.normalize(qraw - kmean, dim=-1)
        return torch.softmax(self.tau.exp() * qn @ K.t() + logg, -1)


def load_memory(mem, sd):
    """Load a Memory from its own or an old gnm_fwm.FastWeightMemory state dict (hop-0 control folded into qdir)."""
    sd = dict(sd)
    if "qdir" not in sd and "Wc.weight" in sd:
        ctrl = sd["hop_e.weight"][0]
        u = F.normalize(sd["Wc.weight"] @ ctrl + sd["Wc.bias"], dim=-1)
        sd["qdir"] = sd["c_scale"] * u
    own = mem.state_dict()
    miss = [k for k in own if k not in sd]
    mem.load_state_dict({k: v for k, v in sd.items() if k in own}, strict=False)
    return miss


# ------------------------------------------------------------------ read -> Walker
class Readout(nn.Module):
    """Pooled read (Wo) for the first answer position, slot read (Us) per answer position; own gates."""

    def __init__(self, d, dk, vslots):
        super().__init__()
        self.gate = nn.Parameter(torch.tensor(1.0))
        self.Us = nn.Linear(dk, d) if vslots else None
        self.gs = nn.Parameter(torch.tensor(0.0))

    def load(self, st):
        with torch.no_grad():
            self.gate.copy_(st["gate"])
            if self.Us is not None and st.get("us") is not None:
                self.Us.load_state_dict(st["us"])
                self.gs.copy_(st["gs"])

    def state(self):
        return {"gate": self.gate.detach(), "us": self.Us.state_dict() if self.Us is not None else None,
                "gs": self.gs.detach()}


class Inject:
    """Forward pre-hook on one Walker block: adds the read at the answer positions a0 (per row) onward."""

    def __init__(self, model, block, readout):
        self.ro, self.add, self.slots, self.a0 = readout, None, None, None
        model.blocks[block].register_forward_pre_hook(self._hook)

    def set(self, add, slots, a0):
        self.add, self.slots, self.a0 = add, slots, a0

    def clear(self):
        self.add = self.slots = self.a0 = None

    def _hook(self, _m, args):
        if self.add is None:
            return None
        x = args[0].clone()
        B, T = x.shape[:2]
        ar = torch.arange(B, device=x.device)
        a0 = self.a0 if torch.is_tensor(self.a0) else torch.full((B,), int(self.a0), device=x.device)
        rms_all = x.float().pow(2).mean(-1, keepdim=True).sqrt()
        a = self.add / self.add.pow(2).mean(-1, keepdim=True).sqrt().clamp(min=1e-6)
        x[ar, a0] = x[ar, a0] + (torch.tanh(self.ro.gate) * rms_all[ar, a0] * a).to(x.dtype)
        if self.slots is not None:
            sl = self.slots / self.slots.pow(2).mean(-1, keepdim=True).sqrt().clamp(min=1e-6)
            for t in range(sl.shape[1]):
                pos = a0 + t
                ok = pos < T
                if not bool(ok.any()):
                    break
                r, p = ar[ok], pos[ok]
                x[r, p] = x[r, p] + (torch.tanh(self.ro.gs) * rms_all[r, p] * sl[ok, t]).to(x.dtype)
        return (x,) + tuple(args[1:])


# ====================================================================================================================
# 4. GNM
# ====================================================================================================================
# Tool nodes share one label-conditioned network; arithmetic tools + EDIT (the loop integrates a note into a node).
TOOLS = [("add", lambda a, b: a + b, "the sum of {0} and {1}"),
         ("sub", lambda a, b: a - b, "the difference between {0} and {1}"),
         ("max", max, "the larger of {0} and {1}"),
         ("min", min, "the smaller of {0} and {1}"),
         ("edit", None, None)]
EDIT = 4
N_ARITH = 4
HALT = len(TOOLS)
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
                      "become the current {k}", "supersede the stored {k}"]}
RULE_HELD = {"add": "get summed into the stored {k}", "set": "supersede the stored {k}"}


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
            free = [x for x in self.num_keys if x not in self.policy]
            if free and (rng.random() >= p_change or not self.policy):
                k = rng.choice(sorted(free))
            elif self.policy:
                k = rng.choice(sorted(self.policy))
                kind = kind or ("set" if self.policy[k] == "add" else "add")
            else:
                return None
        kind = kind or rng.choice(["add", "add", "set"])
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
                    if vs and (len(vs) >= 2 or rng.random() < 0.3):
                        rec[k] = rng.choice(vs)
                    else:
                        vs.append(rng.choice([x for x in range(1, 31) if x not in vs]))
                        rec[k] = vs[-1]
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
        c = sorted(f for f, vs in self.multi.items() if len(vs) == 1)
        if not c:
            return None
        nm, k = rng.choice(c)
        vs = self.multi[(nm, k)]
        vs.append(rng.choice([x for x in range(1, 31) if x not in vs]))
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


def render_sp(t):
    """render(t) plus the character span of every operation subtree inside the text (its words, without the
    parentheses): {id(subtree): (start, end)}"""
    if t[0] == "leaf":
        return leaf(t[1], t[2]), {}
    tpl = next(x for x in TOOLS if x[0] == t[1])[2]
    pre, rest = tpl.split("{0}")
    mid, post = rest.split("{1}")
    out, spans, pos = pre, {}, len(pre)
    for i, c in enumerate(t[2]):
        tx, sp = render_sp(c)
        op = c[0] == "op"
        start = pos + (1 if op else 0)
        if op:
            spans[id(c)] = (start, start + len(tx))
            for k_, (a_, b_) in sp.items():
                spans[k_] = (a_ + start, b_ + start)
        w = f"({tx})" if op else tx
        out += w
        pos += len(w)
        if i == 0:
            out += mid
            pos += len(mid)
    out += post
    return out, spans


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


# ====================================================================== the live store (one world, never reset)
class Store:
    """Frames' token states + nodes. A node has a LABEL source (its identity: which pair's tokens) and a VALUE source
    (a pair's tokens, or a vector made by a tool / an edit). Integrate re-points a node's value source and keeps the
    old one in a version node. Vectors are recomputed every step from the sources with the current heads (2.2)."""

    def __init__(self):
        self.next_fid, self.frames, self.nodes, self.time = 0, {}, [], 0
        self.vecs, self.next_vec = {}, 0       # vector value sources: key -> (k, v, r, slots)
        self.prov = {}                          # source -> what it is in the truth (bookkeeping only)
        self.live = {}                          # nid -> (dk, dv, dr) chosen this step (with grad)
        self.soft = {}                          # nid -> log-weight this step (training: soft integrate)
        self.mix = []                           # (target, source, weight): soft writes into a node this step
        self.post_repl, self.post_gone = {}, set()   # this step's decided replaces (target -> source), merged sources
        self.empty = {}                         # dropped sources this step: value forward 0, gradient "store it"
        self.post_ver = set()                   # copies a decided replace will turn into versions
        self.dropped = set()                    # candidates this step's decisions drop (not asked about)
        self.merged = {}                        # candidate -> the node it was integrated into
        self.expect = {}                        # (name, key, value) -> node expected to hold it (soft phase only)
        self.pending_ver = {}                   # target -> its version node, made when a replace is decided (soft)
        self.evicting = {}                      # nid -> keep weight: forward 0 = the model forgets it this step

    def add_frame(self, ids, X, enc, metas, kind, tick=True):
        fid = self.next_fid
        self.next_fid += 1
        self.time += int(tick)
        nids = []
        for j, meta in enumerate(metas):
            self.prov[("tok", fid, j)] = meta
            nids.append(self.new_node(("tok", fid, j), ("tok", fid, j), kind, self.time, len(enc["vpos"][j])))
        self.frames[fid] = {"ids": ids, "X": X, "enc": enc}
        return nids

    def new_node(self, lsrc, vsrc, kind, time_, nslots):
        self.nodes.append({"lsrc": lsrc, "vsrc": vsrc, "kind": kind, "time": time_, "alive": True, "usage": 0,
                           "act": 1.0, "d": None, "nslots": nslots, "ver_of": None, "link": None,
                           "rsrc": vsrc})   # the text its REFERENCE is read from ("same as ..."); a propagation
        #                                     changes the value (vsrc) but not what the node follows
        return len(self.nodes) - 1

    def add_vec(self, vec, meta):
        key = ("vec", self.next_vec)
        self.next_vec += 1
        self.vecs[key] = vec
        self.prov[key] = meta
        return key

    def alive(self):
        return [i for i, n in enumerate(self.nodes) if n["alive"]]

    def commit(self):
        for nid, d in self.live.items():
            self.nodes[nid]["d"] = tuple(x.detach() for x in d)
        self.live, self.soft, self.mix, self.expect, self.empty = {}, {}, [], {}, {}
        self.post_repl, self.post_gone, self.post_ver, self.dropped = {}, set(), set(), set()
        self.pending_ver = {}
        self.vecs = {k: tuple(x.detach() for x in v) for k, v in self.vecs.items()}

    def decay(self, f=0.99):
        for i in self.alive():
            self.nodes[i]["act"] *= f

    def evict(self, cap, keep_score, protect=0):
        live = self.alive()
        if len(live) > cap:          # short-term buffer: nodes written in the last `protect` frames are never evicted
            old = [i for i in live if self.nodes[i]["time"] <= self.time - protect]
            for i in sorted(old, key=keep_score)[:len(live) - cap]:
                self.nodes[i]["alive"] = False
        used = set()
        for i in self.alive():
            for s_ in (self.nodes[i]["lsrc"], self.nodes[i]["vsrc"], self.nodes[i].get("rsrc")):
                if s_ is not None:
                    used.add(s_[1] if s_[0] == "tok" else s_)
        for fid in [f for f in self.frames if f not in used]:
            del self.frames[fid]
        for k in [k for k in self.vecs if k not in used]:
            del self.vecs[k]
        for k in [k for k in self.prov if (k[1] if k[0] == "tok" else k) not in used]:
            del self.prov[k]


# ====================================================================== model parts
class Graph(nn.Module):
    """Node heads on top of the node core (label = key, value, slots, gain) + relation r, relation / dependence
    edges (used by writes: attach links, propagation), keep knob, version embedding, running key mean."""

    def __init__(self, mem, d, dk):
        super().__init__()
        self.mem = mem
        self.pr = Pool(2 * d)
        self.Wr = nn.Linear(2 * d, dk)
        # R = THE node-to-node relation (user 2026-10-25: dependence, relative definitions, links ... all through r;
        # the model creates and manages it). A node's r = its own content read like a question (Wrr, identity at init:
        # "12 (same as the cost of Batha)" -> a query for that label) + its pooled pair state (r_pool, 0 at init)
        # + the model's own write deltas; an edge i -> j = r_i addressing node j's label, or none.
        self.Wrr = nn.Linear(dk, dk)
        with torch.no_grad():
            self.Wrr.weight.copy_(torch.eye(dk)); self.Wrr.bias.zero_()
        self.r_pool = nn.Parameter(torch.tensor(0.0))
        self.Wref = nn.Linear(dk, dk)            # edge query from r (residual, zero at init)
        nn.init.zeros_(self.Wref.weight); nn.init.zeros_(self.Wref.bias)
        self.tau_ref = nn.Parameter(torch.tensor(math.log(10.0)))   # edge temperature
        self.Wnone = nn.Linear(dk, 1)                               # (old: none logit from the query; unused)
        self.Whas = nn.Linear(2 * d, 1)          # "relates to nothing" logit from the whole value text (mean of its
        nn.init.zeros_(self.Whas.weight); nn.init.constant_(self.Whas.bias, 1.0)   # standalone states): the wording
        #   ("same as ...") decides, not the names (eval dependents with unseen attribute names: has ~0)
        self.Wrs = nn.Linear(2 * d, 1)           # its own token-pooling score on top of the question's (zero init):
        nn.init.zeros_(self.Wrs.weight); nn.init.zeros_(self.Wrs.bias)   # the question pool missed the entity
        self.b_ver = nn.Parameter(torch.tensor(0.0))                # reference to a version node (learned)
        nn.init.zeros_(self.Wnone.weight); nn.init.constant_(self.Wnone.bias, 1.0)
        self.w_rel = nn.Parameter(torch.tensor(0.0))
        self.age = nn.Parameter(torch.zeros(3))            # node stat: age (smooth)
        # FORGET (user 2026-10-25: the model decides what to forget, nothing automatic): keep logit from what the
        # node is (its label) + observations [version, log usage, log age, log activation, write gate, state, question]
        self.Wf_lab = nn.Linear(dk, 1)
        nn.init.zeros_(self.Wf_lab.weight); nn.init.zeros_(self.Wf_lab.bias)
        self.Wf_feat = nn.Linear(7, 1)
        with torch.no_grad():                    # init = the old hand rule (act x sigmoid(4 - ver + .5 log use) x gate)
            self.Wf_feat.weight.copy_(torch.tensor([[-1.0, 0.5, 0.0, 1.0, 1.0, 0.0, 0.0]]))
            self.Wf_feat.bias.fill_(4.0)
        self.register_buffer("kmu", torch.zeros(dk))         # running mean of written keys (reads must not depend
        self.register_buffer("kmu_n", torch.zeros(()))       # on what else is in memory)
        self.ver_e = nn.Parameter(torch.zeros(dk))         # added to a version node's key: "the value before"
        self.w_id = nn.Parameter(torch.tensor(1.0))        # frame rows: weight of their own label in the address

    def nodes(self, X, E, spans, ptr, vpos):
        kraw, v, logg, slots, g = self.mem.write(X, E, spans, ptr, vpos)
        r = self.Wr(self.pr.flat(*self.mem.last_flat))
        return kraw, v, logg, slots, r

    def ref_logits(self, K, RQ, kmean, ver=None, HM=None):
        """[nodes | none] logits of each node's relation edge: its r (RQ arg: raw, key space) addressing all labels K
        (centered, normalized); self excluded; zero r -> none.
        The none logit is a SEPARATE binary decision (sigmoid), not a softmax column: against 3,000 node scores its
        calibration drifted with memory size (dependents -> none 269 misses at step 100)"""
        n_ = K.shape[0]
        q_ = RQ + self.Wref(RQ)
        qn = F.normalize(q_ - kmean, dim=-1)
        sc = (self.tau_ref.exp() * qn @ K.t()).masked_fill(torch.eye(n_, dtype=torch.bool, device=K.device), -1e4)
        has = (RQ.abs().sum(-1, keepdim=True) > 0)
        if ver is not None:                       # "same as X" means X now: a version ("the value before") is a
            sc = sc + self.b_ver * ver[None]      # model-visible node flag (ver_e on its key); direct learned bias
        sc = sc.masked_fill(~has, -1e4)
        nl = self.Whas(HM) if HM is not None else self.Wnone(qn)
        return torch.cat([sc, nl.masked_fill(~has, 1e4)], -1)

    def ref_feat(self, pre):
        """mean of a value's standalone token features (standardized like a question): what the text SAYS"""
        Xb, Eb, mq = pre
        Hs = torch.cat([self.mem.ln((Xb - self.mem.mu_q) / (self.mem.var_q + 1e-5).sqrt()), self.mem.lne(Eb)], -1)
        m = mq.float()[..., None]
        return (Hs * m).sum(1) / m.sum(1).clamp(min=1)

    def ref_query(self, X, E, vpos, cap=32, pre=None):
        """the reference query of value token lists `vpos` (positions into X / E): the question path + Wrs pooling.
        pre = (X, E, mask) per value given directly (Walker's own re-read of the mention, no record context)"""
        if pre is not None:
            Xb, Eb, mq_ = pre
            return self.mem.query(Xb, mq_, Eb, fixed=True, score_add=self.Wrs) * mq_.any(1, keepdim=True).float()
        dev_ = X.device
        Lq = max(1, min(cap, max(len(q) for q in vpos)))
        ix_ = torch.zeros(len(vpos), Lq, dtype=torch.long, device=dev_)
        mq_ = torch.zeros(len(vpos), Lq, dtype=torch.bool, device=dev_)
        for i, q in enumerate(vpos):
            q = q[:Lq]
            if q:
                ix_[i, :len(q)] = torch.tensor(q, device=dev_)
                mq_[i, :len(q)] = True
        return self.mem.query(X[ix_], mq_, E[ix_], fixed=True, score_add=self.Wrs) * mq_.any(1, keepdim=True).float()

    def dep_ref(self, K, RQ, kmean, ver=None, HM=None):
        """dependence edges: row j = the node j's value follows (its reference), none -> no edge"""
        L = self.ref_logits(K, RQ, kmean, ver, HM)
        return torch.sigmoid(-L[:, -1:]) * torch.softmax(L[:, :-1], -1)   # P(has a reference) x which node

    def forget_logit(self, K, feats):
        """keep logit per node (higher = keep); the lowest go when memory is over capacity"""
        return (self.Wf_lab(K) + self.Wf_feat(feats)).squeeze(-1)


class ToolBank(nn.Module):
    """Tool nodes {label l, relation r} sharing ONE network of generic vector primitives whose behaviour is set by the
    tool's LABEL vector (a tool = a label). Input: (label, value, first slots) of two registers -> a new latent node."""

    def __init__(self, n, dk, ns):
        super().__init__()
        self.ns = ns
        self.labels = nn.Parameter(torch.randn(n, dk) / dk ** 0.5)
        self.r = nn.Parameter(torch.randn(n, dk) / dk ** 0.5)
        dx = (1 + ns) * dk
        self.inp = nn.Linear(2 * dk + 4 * dx + 1, 2 * dk)
        self.film = nn.Linear(dk, 8 * dk)
        self.mid = nn.Linear(2 * dk, 2 * dk)
        self.out = nn.Linear(2 * dk, (3 + ns) * dk)

    def forward(self, ti, a, b):
        return self.run(self.labels[ti], a, b)

    def run(self, lab, a, b):
        """the shared network conditioned on a label vector (a tool's label, or a procedure read from memory)"""
        xa = torch.cat([a[1], a[2][:, :self.ns].flatten(1)], -1)
        xb = torch.cat([b[1], b[2][:, :self.ns].flatten(1)], -1)
        cos = F.cosine_similarity(xa, xb, dim=-1)[:, None]
        x = torch.cat([a[0], b[0], xa, xb, xa * xb, xa - xb, cos], -1)
        g1, b1, g2, b2 = self.film(lab).chunk(4, -1)
        h = F.gelu(self.inp(x) * (1 + g1) + b1)
        h = F.gelu(self.mid(h) * (1 + g2) + b2)
        o = self.out(h).view(x.shape[0], 3 + self.ns, -1)
        return o[:, 0], o[:, 1], o[:, 2], o[:, 3:]


class Loop(nn.Module):
    """Heads on Walker's state: query (address), halt, NEW, argument pointers, latent feedback; the write head at the
    addressed node (gate, replace-vs-attach) and the vectors the model writes."""

    def __init__(self, d, dk, vslots, max_regs):
        super().__init__()
        self.Wq = nn.Linear(d, dk)
        nn.init.zeros_(self.Wq.weight); nn.init.zeros_(self.Wq.bias)
        self.Wqr = nn.Linear(dk, dk)
        with torch.no_grad():
            self.Wqr.weight.copy_(torch.eye(dk)); self.Wqr.bias.zero_()
        self.Wgq = nn.Linear(d, 1)
        nn.init.zeros_(self.Wgq.weight); nn.init.zeros_(self.Wgq.bias)
        self.halt = nn.Parameter(torch.randn(dk) / dk ** 0.5)
        self.Wa = nn.ModuleList([nn.Linear(d, dk) for _ in range(2)])
        self.Wrk = nn.Linear(dk, dk)
        self.rpos = nn.Parameter(torch.zeros(max_regs, dk))
        self.Wfv = nn.Linear(dk, d)
        self.Wfs = nn.Linear(dk, d)
        self.spos = nn.Parameter(torch.zeros(vslots, d))
        self.ln = nn.LayerNorm(d)
        # address extras + the write head. NEW starts preferred (b_new: a duplicate is harmless, a wrong overwrite
        # is not); gate open for NEW, neutral for writes into a node; replace vs attach neutral
        self.new_lab = nn.Parameter(torch.randn(dk) / dk ** 0.5)
        self.b_new = nn.Parameter(torch.tensor(1.0))                    # NEW logit offset (x tau)
        self.Wg, self.Wum = nn.Linear(d, 1), nn.Linear(d, 1)
        for m in (self.Wg, self.Wum):
            nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)
        self.gate_w = nn.Parameter(torch.tensor([0.0, 0.0, 0.0, 2.0]))  # cos value, cos key, node bias, NEW bias
        self.mode_w = nn.Parameter(torch.tensor([0.0, 0.0, 0.0]))       # cos value, cos key, bias (replace)
        # HOW to write: a procedure node read from memory (a rule), or the default procedure; it conditions the
        # write network (copy the source vs compute from source + stored value) and the mode
        self.Wp = nn.Linear(d, dk)
        nn.init.zeros_(self.Wp.weight); nn.init.zeros_(self.Wp.bias)
        self.Wpr = nn.Linear(dk, dk)
        with torch.no_grad():
            self.Wpr.weight.copy_(torch.eye(dk)); self.Wpr.bias.zero_()
        self.proc0 = nn.Parameter(torch.randn(dk) / dk ** 0.5)        # the default procedure
        self.empty_v = nn.Parameter(torch.randn(dk) / dk ** 0.5)      # what a node holds when nothing was stored
        self.empty_s = nn.Parameter(torch.randn(dk) / dk ** 0.5)      # (readout / adapters are scale-invariant: a
        #                                                               value is mixed toward these, not scaled)
        self.b_proc = nn.Parameter(torch.tensor(0.5))                  # default-procedure logit offset (x tau): a
        #                         rule whose label matches the pair better than cos .5 is read first (measured: .6-.85)
        self.Wself = nn.Linear(dk, dk)          # "which question retrieves me": a label -> the read query that finds
        with torch.no_grad():                   # it (trained on every demonstrated read); frame rows address with it
            self.Wself.weight.copy_(torch.eye(dk)); self.Wself.bias.zero_()
        self.look_vec = nn.Parameter(torch.zeros(d))   # marks the look register: "the target after my write"
        # COMPOSE (user, 2026-10-21: refine / create a node out of pre-existing nodes as context): a tool node that
        # takes MANY registers. Inclusion gate per register, attention by a query from Walker's state; the output is
        # a refinement of the candidate (label / value / relation deltas, zero at init: composing changes nothing
        # until it helps answers)
        self.compose_lab = nn.Parameter(torch.randn(dk) / dk ** 0.5)
        self.skip_lab = nn.Parameter(torch.randn(dk) / dk ** 0.5)   # "don't compose" (HALT's label is trained by
        #                                                             questions: it won every compose decision)
        self.Wset = nn.Linear(d, dk)             # which registers to include
        self.Wcq = nn.Linear(d, dk)              # attention query over the included registers
        self.comp_ln = nn.LayerNorm(2 * dk + d)   # (raw labels / values / Walker state: unnormalized input made the
        #                                           deltas jump 0 -> 17 in two steps)
        self.Wcomp = nn.Linear(2 * dk + d, 3 * dk)   # (context [k; v], Walker state) -> (dk, dv, dr)
        nn.init.zeros_(self.Wcomp.weight); nn.init.zeros_(self.Wcomp.bias)
        self.Wlook = nn.Linear(d, 3)             # the look's correction of the proposal (gate, mode, NEW-gate logits)
        nn.init.zeros_(self.Wlook.weight); nn.init.zeros_(self.Wlook.bias)
        # RECURSION (long chains): SUB hands a span of the row's question (its register 0 slots) to a child row; the
        # child's answer returns as a register. Every row stays <= one operation: depth-independent programs.
        self.sub_lab = nn.Parameter(torch.randn(dk) / dk ** 0.5)
        self.Wss, self.Wse = nn.Linear(d, dk), nn.Linear(d, dk)   # span start / end queries from Walker's state
        self.Wsk = nn.Linear(dk, dk)                              # keys of the question's slots (start / end)
        self.Wage = nn.Linear(d, 3)              # the query's own say on recency (is-current, log-age, log-age^2)
        nn.init.zeros_(self.Wage.weight); nn.init.zeros_(self.Wage.bias)
        self.Wop, self.Wup = nn.Linear(dk, 1 + N_ARITH), nn.Linear(dk, 1)   # [copy | add sub max min] (object tools)
        nn.init.zeros_(self.Wop.weight)
        with torch.no_grad():
            self.Wop.bias.copy_(torch.tensor([3.0] + [0.0] * N_ARITH))           # copy at init
        nn.init.zeros_(self.Wup.weight); nn.init.zeros_(self.Wup.bias)
        self.Wdk, self.Wdv, self.Wdr = nn.Linear(d, dk), nn.Linear(d, dk), nn.Linear(d, dk)
        for m in (self.Wdk, self.Wdv, self.Wdr):
            nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)


# ====================================================================== main
def main():
    P = argparse.ArgumentParser()
    P.add_argument("--hf", default="gpt2")
    P.add_argument("--ckpt", default="")
    P.add_argument("--enc-layer", type=int, default=0)
    P.add_argument("--ctrl-layer", type=int, default=-1)
    P.add_argument("--init", default="", help="read skill (memory + readout) from a lookup checkpoint")
    P.add_argument("--steps", type=int, default=3000)
    P.add_argument("--pending", type=int, default=4, help="pending-item frames per step (what is to be remembered)")
    P.add_argument("--states", type=int, default=2, help="state frames per step (seen, never asked)")
    P.add_argument("--corrections", type=float, default=0.5)
    P.add_argument("--questions", type=int, default=24)
    P.add_argument("--warm-frames", type=int, default=60)
    P.add_argument("--cap", type=int, default=3000)
    P.add_argument("--protect", type=int, default=0, help="(rule, off by default) frames a new node is safe from "
                   "eviction; the forget head sees the node's age")
    P.add_argument("--dep-q", type=float, default=0.1, help="share of questions about dependents (lazy: the 2-read chain)")
    P.add_argument("--forget-q", type=float, default=0.1, help="share of questions about facts the model chose to forget")
    P.add_argument("--max-ents", type=int, default=150)
    P.add_argument("--mix", default="lookup:0.25,chain:0.15,arith:0.35,before:0.15,multi:0.1")
    P.add_argument("--train-ops", default="1,2")
    P.add_argument("--now", type=float, default=0.25, help="share of training questions about THIS step's items")
    P.add_argument("--recent", type=float, default=0.25, help="share about the last 3 steps")
    P.add_argument("--long-ops", default="3,4")
    P.add_argument("--conf", type=float, default=0.5, help="inference: placement below this -> NEW (no overwrite)")
    P.add_argument("--cost-new", type=float, default=0.2, help="storage cost of a new node (memory is finite: a "
                   "pair nobody needs must cost more than nothing, or state frames flood the memory)")
    P.add_argument("--dk", type=int, default=512)
    P.add_argument("--vslots", type=int, default=64)
    P.add_argument("--tool-slots", type=int, default=2)
    P.add_argument("--object-tools", type=int, default=1, help="arithmetic on decoded values (decode -> exact -> encode)")
    P.add_argument("--early-version", type=int, default=0, help="a replace decided in training makes its version at once")
    P.add_argument("--q-slots-mode", default="reg", choices=["mem", "reg"],
                   help="mem: question / note nodes keep q-slots in memory; reg: only Walker's register gets them")
    P.add_argument("--ref-bank", type=int, default=6000, help="value texts kept in the re-read bank (GPU)")
    P.add_argument("--ref-alone", type=int, default=1, help="reference query: Walker re-reads the value tokens alone")
    P.add_argument("--rel-lr", type=float, default=1e-3, help="the relation heads' own optimizer")
    P.add_argument("--dep-follow", default="lazy", choices=["lazy", "eager"],
                   help="dependents: lazy = a question reads the dependent, then what it mentions (a 2-read chain; "
                   "the dependent keeps its own 'same as ...' text); eager = the dependent's value is overwritten "
                   "when its source changes (propagation)")
    P.add_argument("--w-ref-demo", type=float, default=1.0, help="r edges: weight of the demonstration (the generator "
                   "knows a dependent's source); 0 = the task gradient only")
    P.add_argument("--delta-cap", type=float, default=0.3, help="a model delta is at most this x the vector it adjusts")
    P.add_argument("--recurse", type=int, default=0, help="long chains by recursion: SUB hands a span of the question "
                   "to a child row whose answer returns as a register (every row <= one operation)")
    P.add_argument("--max-depth", type=int, default=6)
    P.add_argument("--q-slots", type=int, default=128, help="value slots of question / note nodes (the current input)")
    P.add_argument("--mem-slots", type=int, default=16, help="value slots a node keeps in memory (answers use <= max-ans)")
    P.add_argument("--write-slots", type=int, default=16, help="value slots Walker reads of a frame row's candidate")
    P.add_argument("--note-expect", type=int, default=1, help="a correction the note did not write is still asked")
    P.add_argument("--w-commit", type=float, default=1.0, help="commit target from the model's own answers (notes)")
    P.add_argument("--commit-margin", type=float, default=0.1)
    P.add_argument("--w-commit-frames", type=float, default=0.0, help="frame rows: writing into a node is trained by "
                   "whether that node's fact is still answered right (the model's own answers)")
    P.add_argument("--id-guard", type=int, default=1, help="(rule) a replace into a node whose label is not ~identical "
                   "becomes NEW; 0 = the model's write decision alone")
    P.add_argument("--refine", type=int, default=0, help="refine on write: gather passes (context reads) before a "
                   "compose pass, then the write (0 = off)")
    P.add_argument("--compose-cap", type=float, default=0.3, help="a compose delta is at most this x the vector's norm")
    P.add_argument("--compose-k", type=int, default=1, help="compose may adjust the label too")
    P.add_argument("--look-lr", type=float, default=1.0, help="lr multiplier of the look's correction head")
    P.add_argument("--look-frames", type=int, default=0, help="frame rows look before commit too")
    P.add_argument("--look", type=int, default=1, help="look before commit: a write into a node is first a proposal; "
                   "the row sees the target as it would be after it (one more pass) and decides then")
    P.add_argument("--write-passes", type=int, default=1, help="passes per frame row (2 = may look, then write)")
    P.add_argument("--collateral", type=float, default=0.15, help="share of questions about nodes a write addressed")
    P.add_argument("--w-inv", type=float, default=0.2, help="context-invariant labels (InfoNCE) weight")
    P.add_argument("--inv-temp", type=float, default=0.1)
    P.add_argument("--w-self", type=float, default=0.5, help="'which question retrieves me' head weight")
    P.add_argument("--reset-wself", action="store_true")
    P.add_argument("--w-pend", type=float, default=0.3, help="pending content (facts / increments / rules) kept somewhere")
    P.add_argument("--cluster-cos", type=float, default=0.95, help="labels this close = copies of the same thing")
    P.add_argument("--write-deltas", default="vr", help="model-written deltas on label (k) / value (v) / relation (r); "
                   "label deltas scrambled the key space (ablation: identity lookup .58 -> 1.00 without them)")
    P.add_argument("--w-rt", type=float, default=0.5, help="round-trip (encode -> say) loss in every step")
    P.add_argument("--p-rule", type=float, default=0.2, help="chance per step of a rule frame (new or changed rule)")
    P.add_argument("--init-full", default="", help="start from a GNM checkpoint: every tensor whose name and shape match")
    P.add_argument("--swap-from", default="", help="new LM: load only the GNM core from this checkpoint (the interface "
                   "= every tensor touching the LM width is re-fit)")
    P.add_argument("--freeze-core", action="store_true", help="with --swap-from: train only the interface (stage 1)")
    P.add_argument("--inject", type=int, default=6)
    P.add_argument("--lr", type=float, default=5e-4)
    P.add_argument("--eval-every", type=int, default=500)
    P.add_argument("--eval-q", type=int, default=96)
    P.add_argument("--ans-eos", type=int, default=1, help="answers are trained with an end token; decode stops there")
    P.add_argument("--max-ans", type=int, default=6)
    P.add_argument("--seed", type=int, default=0)
    P.add_argument("--save", default="", help="checkpoint path (default E:/WEIGHTS/gnm_runs/gnm.pt when E: exists)")
    P.add_argument("--load", default="", help="evaluate a trained checkpoint (no training)")
    P.add_argument("--log", default="runs/gnm.jsonl")
    P.add_argument("--smoke", action="store_true", help="build check: a few steps + a small eval")
    P.add_argument("--first", type=int, default=0, help="with --load: N fresh worlds, write ONE first frame into an "
                   "empty memory, ask every fact in it at once")
    P.add_argument("--viz", default="", help="with --load: write an interactive graph of the live memory (HTML)")
    P.add_argument("--viz-ents", type=int, default=12, help="entities drawn (the whole memory goes into the stats)")
    P.add_argument("--viz-sim", type=int, default=40, help="steps of frames the model writes itself before drawing")
    P.add_argument("--viz-min", type=float, default=0.3, help="weakest dependence / relation edge drawn")
    P.add_argument("--first-warm", type=int, default=0, help="control for --first: memory first holds N other frames")
    P.add_argument("--skip-eval0", action="store_true")
    P.add_argument("--warm-inv", type=float, default=1.0, help="warm-up: weight of the label-identity (invariance) loss")
    P.add_argument("--warm-tools", type=int, default=0, help="warm-up steps of encode -> decode round trips first")
    P.add_argument("--vram-frac", type=float, default=0.9, help="hard VRAM cap: OOM instead of spilling into RAM")
    A = P.parse_args()
    ckpt_home = "E:/WEIGHTS/gnm_runs"               # user (2026-10-21): checkpoints live on E:, never on C: (full)
    if os.path.isdir("E:/WEIGHTS"):
        os.makedirs(ckpt_home, exist_ok=True)
        if not A.save:
            A.save = f"{ckpt_home}/gnm.pt"
        elif os.path.splitdrive(os.path.abspath(A.save))[0].upper() == "C:":
            print(f"checkpoint path {A.save} is on C: -> {ckpt_home}/{os.path.basename(A.save)}", flush=True)
            A.save = f"{ckpt_home}/{os.path.basename(A.save)}"
    elif not A.save:
        A.save = "runs/gnm.pt"
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(A.vram_frac)
    torch.manual_seed(A.seed)
    rng = random.Random(A.seed)
    model, raw = load_walker(A.hf, A.ckpt, A.enc_layer, dev)
    tok = Tok(raw)
    EOS_ID = raw.eos_token_id if getattr(raw, "eos_token_id", None) is not None else tok.ids("\n")[0]
    d = model.emb.weight.shape[1]
    nL = len(model.blocks)
    ctrl_layer = A.ctrl_layer % nL
    tr_ops, lo_ops = (tuple(int(x) for x in s.split(",")) for s in (A.train_ops, A.long_ops))
    max_regs = 2 * lo_ops[1] + 4
    mem = Memory(d, A.dk, A.vslots).to(dev)
    ro = Readout(d, A.dk, A.vslots).to(dev)
    if A.init:
        st = torch.load(A.init, map_location=dev)
        sd = dict(st["mem"])
        if "slot_e" in sd and sd["slot_e"].shape[0] != A.vslots:
            se = torch.zeros(A.vslots, A.dk, device=dev)
            se[:min(A.vslots, sd["slot_e"].shape[0])] = sd["slot_e"][:A.vslots]
            sd["slot_e"] = se
        load_memory(mem, sd)
        ro.load(st)
        print(f"INIT read skill (memory + readout) from {A.init}", flush=True)
    graph = Graph(mem, d, A.dk).to(dev)
    loop = Loop(d, A.dk, A.vslots, max_regs).to(dev)
    tools = ToolBank(len(TOOLS), A.dk, A.tool_slots).to(dev)
    params = list({id(p): p for p in list(graph.parameters()) + list(ro.parameters()) + list(loop.parameters())
                   + list(tools.parameters())}.values())
    knob_names = ("age", "keep", "w_prop", "w_rel", "tau_o", "tau", "beta", "gate", "gs", "w_id")   # write heads: normal lr
    knobs = [p for n, p in list(graph.named_parameters()) + list(ro.named_parameters()) + list(loop.named_parameters())
             + [("tau", mem.tau)] if n.split(".")[-1] in knob_names]
    knobs += list(graph.Wf_feat.parameters())    # the forget head's 8 weights: at the base lr they had not moved
    #                                              after 100 steps (MoLab r_* runs: exactly the init)
    knob_ids = {id(p) for p in knobs}
    ref_ps = list(graph.Wref.parameters()) + list(graph.Wrs.parameters()) + list(graph.Wrr.parameters()) + [graph.tau_ref]
    ref_sc = [graph.b_ver, graph.r_pool] + list(graph.Wnone.parameters()) + list(graph.Whas.parameters())   # small heads: at 1e-3 the version bias moved .025 in
    #   25 steps and the has-reference head never left "none" (0 of 507)
    knob_ids = knob_ids | {id(p) for p in ref_ps + ref_sc}
    opt_ref = torch.optim.Adam([{"params": ref_ps}, {"params": ref_sc, "lr": 30 * A.rel_lr}], lr=A.rel_lr)
    look_ps = list(loop.Wlook.parameters())    # the look's correction (own group; main18c's "did not move" was a
    look_ids = {id(p) for p in look_ps}        # zero gradient: notes' soft write came from their HALT pass)
    opt = torch.optim.AdamW([{"params": [p for p in params if id(p) not in knob_ids | look_ids], "lr_mult": 1.0},
                             {"params": list({id(p): p for p in knobs}.values()), "lr_mult": 50.0,
                              "weight_decay": 0.0},
                             {"params": look_ps, "lr_mult": A.look_lr, "weight_decay": 0.0}], lr=A.lr, weight_decay=0.01)
    assert A.dk not in (d, 2 * d), "dk must differ from the LM width (the interface is found by shape)"

    def is_interface(t):
        """the LM interface = every GNM tensor touching the LM's width d (encoder pooling, latent-token adapter,
        heads on Walker's state, readout / inject); everything else (dk space) is the GNM core"""
        return any(x in (d, 2 * d) for x in t.shape)

    def load_matching(st, only_core=False):
        n_ok, n_skip = 0, 0
        for mod, key in ((graph, "graph"), (loop, "loop"), (tools, "tools")):
            own, new = mod.state_dict(), {}
            for k_, v_ in st.get(key, {}).items():
                if k_ in own and own[k_].shape == v_.shape and not (only_core and is_interface(own[k_])):
                    new[k_] = v_; n_ok += 1
                elif (k_ in own and k_.split(".")[-1] in ("slot_e", "spos") and own[k_].shape[1:] == v_.shape[1:]
                      and not (only_core and is_interface(own[k_]))):   # per-position tables (--vslots changed):
                    pad_ = own[k_].clone()                               # keep the learned rows, new rows start 0
                    n_r = min(pad_.shape[0], v_.shape[0])
                    pad_[:n_r] = v_[:n_r]
                    new[k_] = pad_; n_ok += 1
                else:
                    n_skip += 1
            mod.load_state_dict(new, strict=False)
        if not only_core:
            ro.load(st)
        return n_ok, n_skip

    if A.init_full or A.swap_from:
        st0 = torch.load(A.init_full or A.swap_from, map_location=dev, weights_only=False)
        n_ok, n_skip = load_matching(st0, only_core=bool(A.swap_from))
        print(f"{'SWAP (core only)' if A.swap_from else 'INIT'} from {A.init_full or A.swap_from}: {n_ok} tensors "
              f"loaded, {n_skip} skipped (new heads / interface / shape changes)", flush=True)
        if A.swap_from:                                   # a new LM = a new encoder: the running key mean (dk space,
            with torch.no_grad():                         # so loaded as "core") belongs to the OLD encoder's keys;
                graph.kmu.zero_(); graph.kmu_n.zero_()    # at .9995 per step it took thousands of steps to follow
            print("RESET graph.kmu (new LM: keys from a new encoder)", flush=True)
        if "Wrr.weight" not in st0.get("graph", {}):     # a checkpoint from before r was THE relation (2026-10-25):
            with torch.no_grad():                         # its relation-delta head wrote for the old meaning of r
                nn.init.zeros_(loop.Wdr.weight); nn.init.zeros_(loop.Wdr.bias)   # (noise up to .3 x on every
            print("RESET loop.Wdr (pre-R checkpoint: the model writes r from scratch)", flush=True)   # pointer)
        del st0
    if A.reset_wself:                                     # a collapsed Wself (old cosine objective) -> identity again
        with torch.no_grad():
            loop.Wself.weight.copy_(torch.eye(A.dk)); loop.Wself.bias.zero_()
    if A.freeze_core:
        n_fr = 0
        for p_ in params:
            if not is_interface(p_):
                p_.requires_grad_(False); n_fr += 1
        print(f"FREEZE core: {n_fr} tensors frozen, training the interface only", flush=True)
    inject = Inject(model, A.inject % nL, ro)
    ans_ids = tok.ids(" Answer:")
    a0 = len(ans_ids) - 1
    start_ids = tok.ids(" Steps:")
    PH = tok.ids(" ~")[0]
    fb = {"pos": None, "vec": None}

    def fb_hook(_m, args):
        if fb["pos"] is None:
            return None
        x = args[0].clone()
        rows, cols = fb["pos"]
        rms = x.float().pow(2).mean(-1, keepdim=True).sqrt().mean()
        x[rows, cols] = (loop.ln(fb["vec"]) * rms).to(x.dtype)
        return (x,) + tuple(args[1:])
    model.blocks[0].register_forward_pre_hook(fb_hook)

    world = World(rng, max_ents=A.max_ents)
    store = Store()
    pend = []                                               # integrate decisions waiting for commit (training)
    wstats = {}                                             # integrate diagnostics

    # ------------------------------------------------------------------ materialize: every node with current heads
    cache = {"ver": 0, "at": -1, "src": {}}       # per-source vectors (no grad) with the CURRENT heads (2.2)

    def heads_changed():
        cache["ver"] += 1

    def padS(x, W):
        n_ = x.shape[-2]
        return x[..., :W, :] if n_ >= W else F.pad(x, (0, 0, 0, W - n_))

    def stackS(xs):
        W = max(x.shape[0] for x in xs)
        return torch.stack([padS(x, W) for x in xs])

    alone = {"row": {}, "X": None, "E": None, "M": None}   # value text -> row of a bank of Walker states
    ref_t = {"s": 0.0, "calls": 0, "rows": 0, "new": 0}

    def ref_alone(srcs, cap=16):
        """--ref-alone: each value's tokens re-read by Walker on their own (a mention read like a question; inside the
        frame its states carry the dependent's own record: references hit the right attribute of the WRONG entity).
        srcs = [(frame id, pair j)] -> (X, E, mask) per value. Cached by the value's TEXT (Walker is frozen; values
        repeat; frames are short-lived and their ids are reused after eval restores) in one bank: one gather."""
        if not A.ref_alone:
            return None
        t0_ = time.perf_counter()
        keys = []
        for f_, j in srcs:
            fr = store.frames[f_]
            keys.append(tuple(fr["ids"][x] for x in fr["enc"]["vpos"][j][:cap]))
        rows = alone["row"]
        new = sorted({k for k in keys if k not in rows})
        if new:
            if alone["X"] is None:                            # one preallocated bank (growing it by cat copied it
                d_ = model.emb.weight.shape[1]                # every step: OOM on 6 GB at step 325)
                alone["X"] = torch.zeros(A.ref_bank, cap, d_, device=dev, dtype=torch.half)
                alone["E"] = torch.zeros_like(alone["X"])
                alone["M"] = torch.zeros(A.ref_bank, cap, device=dev, dtype=torch.bool)
            if len(rows) + len(new) > A.ref_bank:             # full: start over with this call's texts
                rows.clear()
                new = sorted(set(keys))
            tb = torch.tensor([list(k) + [EOS_ID] * (cap - len(k)) for k in new], device=dev)
            mq = torch.tensor([[1] * len(k) + [0] * (cap - len(k)) for k in new], device=dev, dtype=torch.bool)
            with torch.no_grad():                             # right padding (causal: no effect on real tokens)
                Xn, En = walker_states(model, tb).half(), model.emb(tb).half()
            base = len(rows)
            for i, k in enumerate(new):
                rows[k] = base + i
            alone["X"][base:base + len(new)] = Xn
            alone["E"][base:base + len(new)] = En
            alone["M"][base:base + len(new)] = mq
        ix = torch.tensor([rows[k] for k in keys], device=dev)
        ref_t["calls"] += 1; ref_t["rows"] += len(srcs); ref_t["new"] += len(new)
        out = alone["X"][ix].float(), alone["E"][ix].float(), alone["M"][ix]
        if dev.type == "cuda":
            torch.cuda.synchronize()
        ref_t["s"] += time.perf_counter() - t0_
        return out

    def encode_sources(srcs, grad):
        """(kraw, v, logg, slots[:value length], r) per token source; chunked by frames (bounded memory)"""
        out, by_f = {}, {}
        for s_ in srcs:
            by_f.setdefault(s_[1], []).append(s_)

        def flush(fids):
            Xs, ids, spans, ptr, vpos, off, order = [], [], [], [], [], 0, []
            for f in fids:
                fr = store.frames[f]
                Xs.append(fr["X"]); ids += fr["ids"]
                e = fr["enc"]
                for _, _, j in by_f[f]:
                    spans.append([off + x for x in e["spans"][j]])
                    ptr.append([off + x for x in e["ptr"][j]])
                    vpos.append([off + x for x in e["vpos"][j]])
                order += by_f[f]
                off += len(fr["ids"])
            with torch.set_grad_enabled(grad and torch.is_grad_enabled()):
                X = torch.cat(Xs).float()
                with torch.no_grad():
                    E = model.emb(torch.tensor(ids, device=dev)).float()
                sk, sv, sg, ss, sr = graph.nodes(X, E, spans, ptr, vpos)
                if not grad:
                    sk, sv, sg, ss, sr = (x.detach() for x in (sk, sv, sg, ss, sr))
                with torch.no_grad():                    # reference query: the value tokens read like a question
                    pre_ = ref_alone([(s_[1], s_[2]) for s_ in order])
                    rq = graph.ref_query(X, E, vpos, pre=pre_)
                    hm = graph.ref_feat(pre_) if pre_ is not None else torch.zeros(len(order), 2 * d, device=dev)
            for i, s_ in enumerate(order):
                w_ = max(1, min(len(vpos[i]), A.vslots))
                out[s_] = (sk[i], sv[i], sg[i], ss[i, :w_] if grad else ss[i, :w_].clone(), sr[i], rq[i], hm[i])
        chunk, ntok = [], 0
        for f in sorted(by_f):
            chunk.append(f)
            ntok += len(store.frames[f]["ids"])
            if ntok > 4096:
                flush(chunk)
                chunk, ntok = [], 0
        if chunk:
            flush(chunk)
        return out

    def materialize(only=None, exclude=(), grad=()):
        """node vectors from their sources with the current heads. Sources of the nodes in `grad` are recomputed
        with gradient (this step's nodes and the nodes a program reads); every other node comes from the per-step
        cache (no grad), so memory does not grow with the store."""
        live = [i for i in store.alive() if (only is None or i in only) and i not in exclude]
        if not live:
            return None
        if cache["at"] != cache["ver"]:
            cache["src"], cache["at"] = {}, cache["ver"]
        tok_of = lambda i: [s_ for s_ in (store.nodes[i]["lsrc"], store.nodes[i]["vsrc"], store.nodes[i].get("rsrc"))
                            if s_ is not None and s_[0] == "tok"]
        srcs = {s_ for i in live for s_ in tok_of(i)}
        gsrc = {s_ for i in live if i in grad for s_ in tok_of(i)} if torch.is_grad_enabled() else set()
        miss = sorted((s_ for s_ in srcs if s_ not in cache["src"] and s_ not in gsrc), key=lambda s_: s_[1:])
        if miss:
            with torch.no_grad():
                cache["src"].update(encode_sources(miss, False))
        got = {s_: cache["src"][s_] for s_ in srcs if s_ not in gsrc}
        if gsrc:
            got.update(encode_sources(sorted(gsrc, key=lambda s_: s_[1:]), True))
        ks, vs, gs_, sls, rs, nsl, rqs, hms = [], [], [], [], [], [], [], []
        qfull = {}
        zg = torch.zeros((), device=dev)
        for i in live:
            nd = store.nodes[i]
            L_, V_ = nd["lsrc"], nd["vsrc"]
            q_in = nd["kind"] in ("question", "note")
            cap_ = A.q_slots if (q_in and A.q_slots_mode == "mem") else A.mem_slots   # the current input keeps its
            #   whole text: 16 slots cut a 1-operation question inside its 2nd operand's name and dropped the 2nd /
            #   3rd operands of a 2-operation one (the 2nd read failed: "unrelated" / tool fired early, read .70)
            if L_ is not None and L_[0] == "tok":
                ks.append(got[L_][0]); rs.append(got[L_][4])
            else:
                vec = store.vecs[V_]
                ks.append(vec[0]); rs.append(vec[2])
            Rs_ = nd.get("rsrc", V_)                    # the reference: from the node's own stated text
            okr_ = Rs_ is not None and Rs_[0] == "tok" and nd["kind"] in ("item", "state")
            rqs.append(got[Rs_][5] if okr_ else None)
            hms.append(got[Rs_][6] if okr_ else None)
            if V_[0] == "tok":
                vs.append(got[V_][1]); sls.append(got[V_][3][:cap_]); gs_.append(got[V_][2])
                if q_in and A.q_slots_mode == "reg":      # whole text for Walker's register only (memory stays 16 wide)
                    qfull[len(vs) - 1] = got[V_][3][:A.q_slots]
            else:
                vec = store.vecs[V_]
                vs.append(vec[1]); sls.append(vec[3][:cap_]); gs_.append(zg)
            nsl.append(min(nd["nslots"], cap_))
        kraw, v, logg, r = torch.stack(ks), torch.stack(vs), torch.stack(gs_), torch.stack(rs)
        slots = stackS(sls)
        RQ = torch.stack([x if x is not None else torch.zeros_like(kraw[0]) for x in rqs])
        HM = torch.stack([x if x is not None else torch.zeros(2 * d, device=dev) for x in hms])
        r = graph.Wrr(RQ) * (RQ.abs().sum(-1, keepdim=True) > 0).float() + graph.r_pool * r   # the node's relation
        ver = torch.tensor([float(store.nodes[i]["kind"] == "version") for i in live], device=dev)
        kraw = kraw + ver[:, None] * graph.ver_e                   # "the value before" is part of a version's label
        dsel = [(k_, store.live.get(nid) or store.nodes[nid]["d"]) for k_, nid in enumerate(live)]
        dsel = [(k_, dd) for k_, dd in dsel if dd is not None]
        if dsel:
            ix = torch.tensor([k_ for k_, _ in dsel], device=dev)

            def capd(base, dd_):                               # deltas accumulate over writes (replace moves them:
                nb = base.norm(dim=-1, keepdim=True)          # od + dd): unbounded relation deltas reached 12,500
                nd_ = dd_.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                return dd_ * (A.delta_cap * nb / nd_).clamp(max=1.0)
            kraw = kraw.index_add(0, ix, capd(kraw[ix], torch.stack([dd[0] for _, dd in dsel])))
            v = v.index_add(0, ix, capd(v[ix], torch.stack([dd[1] for _, dd in dsel])))
            r = r.index_add(0, ix, capd(kraw[ix], torch.stack([dd[2] for _, dd in dsel])))   # (scale of the LABEL: a
            #   node with no stated relation has r = 0 — the model must still be able to create one)
        row_of = {n: k_ for k_, n in enumerate(live)}
        lsel = [(k_, row_of[store.nodes[nid]["link"]]) for k_, nid in enumerate(live)
                if store.nodes[nid]["link"] in row_of]
        if lsel:                                                   # attach: its r points at its target (the label)
            r = r.index_add(0, torch.tensor([a_ for a_, _ in lsel], device=dev),
                            kraw[torch.tensor([b_ for _, b_ in lsel], device=dev)])
        mix = [(row_of[t_], row_of[c_], m_) for t_, c_, m_ in store.mix if c_ in row_of and t_ in row_of]
        if mix:                                                    # soft writes of this step (training)
            rt = torch.tensor([x[0] for x in mix], device=dev)
            rc = torch.tensor([x[1] for x in mix], device=dev)
            m = torch.stack([torch.as_tensor(x[2], device=dev, dtype=v.dtype) for x in mix])
            dV, dS = v[rc] - v[rt], slots[rc] - slots[rt]
            free = torch.tensor([store.nodes[i]["kind"] in ("item", "state") for i in live], device=dev)
            ar = torch.arange(len(mix), device=dev)
            km_ = graph.kmu[None] if float(graph.kmu_n) > 0 else kraw.mean(0, keepdim=True)
            C = graph.dep_ref(F.normalize(kraw - km_, dim=-1), r, km_, ver, HM)[:, rt] * m[None] * free[:, None].float()
            if A.dep_follow != "eager":                 # lazy: dependents are followed by reads, never overwritten
                C = C * 0.0
            C = C.index_put((rc, ar), torch.zeros(len(mix), device=dev)).index_put((rt, ar), torch.zeros(len(mix), device=dev))
            v = v + C @ dV                                         # dependents follow (one damped step)
            slots = slots + torch.einsum("nr,rsd->nsd", C, dS)
            v = v.index_add(0, rt, m[:, None] * dV)                # the write itself: the value moves into the node
            slots = slots.index_add(0, rt, m[:, None, None] * dS)
        esel = [(k_, store.empty[nid]) for k_, nid in enumerate(live) if nid in store.empty]
        if esel:                                                   # dropped this step: empty value, gradient only
            ix = torch.tensor([k_ for k_, _ in esel], device=dev)
            f = torch.stack([w_ for _, w_ in esel])
            v = v.index_copy(0, ix, loop.empty_v + f[:, None] * (v[ix] - loop.empty_v))   # forward: value or empty;
            slots = slots.index_copy(0, ix, loop.empty_s + f[:, None, None] * (slots[ix] - loop.empty_s))  # grad: (v - e)
        fsel = [(k_, store.evicting[nid]) for k_, nid in enumerate(live) if nid in store.evicting]
        if fsel:                                                   # the model's forget decision of this step
            ix = torch.tensor([k_ for k_, _ in fsel], device=dev)
            f = torch.stack([w_ for _, w_ in fsel])
            v = v.index_copy(0, ix, loop.empty_v + f[:, None] * (v[ix] - loop.empty_v))
            slots = slots.index_copy(0, ix, loop.empty_s + f[:, None, None] * (slots[ix] - loop.empty_s))
        ssel =[(k_, store.soft[nid]) for k_, nid in enumerate(live) if nid in store.soft]
        if ssel:                                                   # soft integrate (training, this step)
            logg = logg.index_add(0, torch.tensor([k_ for k_, _ in ssel], device=dev),
                                  torch.stack([w_ for _, w_ in ssel]))
        use = torch.tensor([float(store.nodes[i]["usage"]) for i in live], device=dev)
        if mem.training and only is None and not exclude and (getattr(mem, "update_stats", False)
                                                              or float(graph.kmu_n) == 0):   # running key mean
            with torch.no_grad():                                  # (slow; training steps only)
                f = 0.0 if float(graph.kmu_n) == 0 else 0.9995
                graph.kmu.mul_(f).add_((1 - f) * kraw.detach().mean(0))
                graph.kmu_n.add_(1)
        kmean = graph.kmu[None] if float(graph.kmu_n) > 0 else kraw.mean(0, keepdim=True)
        return {"qfull": qfull, "kraw": kraw, "K": F.normalize(kraw - kmean, dim=-1), "kmean": kmean, "V": v, "S": slots,
                "logg": logg, "R": r,
                "time": torch.tensor([float(store.nodes[i]["time"]) for i in live], device=dev),
                "nsl": torch.tensor(nsl, device=dev), "RQ": RQ, "HM": HM, "ver": ver, "rows": live, "row_of": row_of}

    # ------------------------------------------------------------------ Walker reads latent tokens
    def ctrl_state(regs):
        lat = [torch.cat([r_[4] for r_ in row], 0) if row else None for row in regs]
        lens = [len(start_ids) + (x.shape[0] if x is not None else 0) for x in lat]
        L = max(lens)
        ids = torch.tensor([start_ids + [PH] * (n - len(start_ids)) + [0] * (L - n) for n in lens], device=dev)
        rws, cols, vecs = [], [], []
        for b, x in enumerate(lat):
            if x is not None:
                rws += [b] * x.shape[0]
                cols += list(range(len(start_ids), len(start_ids) + x.shape[0]))
                vecs.append(x)
        pos = (torch.tensor(rws, device=dev), torch.tensor(cols, device=dev)) if vecs else None
        last = torch.tensor(lens, device=dev) - 1

        def walker(vec):
            fb["pos"], fb["vec"] = (pos, vec) if pos is not None else (None, None)
            got = {}

            def grab(_m, a, kw):
                got["x"] = a[0] if a else kw["x"]
                raise _Stop
            h = model.blocks[ctrl_layer].register_forward_pre_hook(grab, with_kwargs=True)
            try:
                with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                    model(ids, r_active=False)
            except _Stop:
                pass
            finally:
                h.remove()
                fb["pos"] = fb["vec"] = None
            return got["x"].float()[torch.arange(len(regs), device=dev), last]
        vec = torch.cat(vecs, 0) if vecs else torch.zeros(1, d, device=dev)
        if torch.is_grad_enabled() and vecs:
            from torch.utils.checkpoint import checkpoint
            return checkpoint(walker, vec, use_reentrant=False)
        return walker(vec)

    NUM = re.compile(r"-?\d+")

    @torch.no_grad()
    def decode(vs, ss, text=False):
        """the model's own head reads latent values out loud (greedy): value vector + slots -> number (or None),
        or the text"""
        out = []
        for i0 in range(0, vs.shape[0], 256):
            v_, s_ = vs[i0:i0 + 256], ss[i0:i0 + 256]
            seq = [list(ans_ids) for _ in range(v_.shape[0])]
            add, sl = mem.Wo(v_), ro.Us(s_)
            for _ in range(A.max_ans):
                inp = torch.tensor(seq, device=dev)
                inject.set(add, padS(sl, inp.shape[1] - a0), a0)
                lo, _ = model(inp, r_active=False)
                inject.clear()
                for b, n_ in enumerate(lo[:, -1].argmax(-1).tolist()):
                    seq[b].append(n_)
            for sq in seq:
                gen = sq[len(ans_ids):]
                if EOS_ID in gen:                    # the answer ends here (trained with an end token)
                    gen = gen[:gen.index(EOS_ID)]
                    tx = tok.decode(gen)
                else:                                # no end token: keep only the first token when it is a
                    tx = tok.decode(gen)             # number, so untrained continuation ("19") cannot glue on
                    k_ = 2 if tok.decode(gen[:1]).strip() == "-" else 1   # a sign is its own token
                    first = tok.decode(gen[:k_])
                    if not text and NUM.fullmatch(first.strip() or "x"):
                        tx = first
                if text:
                    out.append(tx)
                    continue
                m_ = NUM.search(tx)
                out.append(int(m_.group()) if m_ else None)
        return out

    def tail(reg):
        """the slot just before a statement's final period (where it puts its value: "... is now 18.") — the model
        decodes value slots exactly but word slots as junk ("is now 17." as 4 slots decoded to "7/176")"""
        n_ = max(1, int(reg[3]))
        return reg[2][max(0, n_ - 2):max(1, n_ - 1)]

    @torch.no_grad()
    def encode_values(vals, times, keep):
        """results as ordinary token nodes ({"result": v} through the same encoder as every frame pair): exact
        values Walker says like any stored value. keep: a node that may be written (else a register only)."""
        if not vals:
            return []
        encs = [doc_encoding(tok.ids, tok.decode, {"result": v}, "state")[0] for v in vals]
        L = max(len(e["ids"]) for e in encs)
        X = walker_states(model, torch.tensor([e["ids"] + [0] * (L - len(e["ids"])) for e in encs], device=dev))
        X = X.to(torch.bfloat16)
        nids = []
        for j, (e, v, t_) in enumerate(zip(encs, vals, times)):
            n_ = store.add_frame(e["ids"], X[j, :len(e["ids"])].clone(), e, [("val", v)], "item", tick=False)[0]
            store.nodes[n_]["time"] = t_
            store.nodes[n_]["alive"] = keep
            nids.append(n_)
        got = encode_sources(sorted({store.nodes[n_]["vsrc"] for n_ in nids}, key=lambda s_: s_[1:]), False)
        out = []
        for n_ in nids:
            kk, vv, _, ss_, rr = got[store.nodes[n_]["vsrc"]][:5]
            ss_ = ss_[:A.mem_slots]
            lat = torch.cat([loop.Wfv(vv)[None], loop.Wfs(ss_) + loop.spos[:ss_.shape[0]]], 0)
            out.append((n_, (kk, vv, ss_, ss_.shape[0], lat, n_ if keep else None)))
        return out

    def ctrl_states(regs, budget=2048):
        """ctrl_state for many rows: sorted by length, chunked by a padded-token budget (no padding waste)"""
        lens = [len(start_ids) + sum(r_[4].shape[0] for r_ in row) for row in regs]
        order = sorted(range(len(regs)), key=lambda i: lens[i])
        out, chunk = [None] * len(regs), []
        for i in order + [None]:
            if chunk and (i is None or (len(chunk) + 1) * lens[i] > budget):
                for j, hj in zip(chunk, ctrl_state([regs[j] for j in chunk])):
                    out[j] = hj
                chunk = []
            if i is not None:
                chunk.append(i)
        return torch.stack(out)

    def regs_from(M, ix_list):
        """registers (raw key, value, slots, n_slots, latent block, nid) for node rows, one gather."""
        if not ix_list:
            return []
        ix = torch.tensor(ix_list, device=dev)
        Kc, Vc, Sc = M["kraw"][ix], M["V"][ix], M["S"][ix]
        fv, fs = loop.Wfv(Vc), loop.Wfs(Sc) + loop.spos[:Sc.shape[1]]
        out = []
        for j, c_ in enumerate(ix_list):
            qf = M.get("qfull", {}).get(c_)
            if qf is not None:                     # a question / note: its whole text as the register (mode "reg")
                n_ = min(int(store.nodes[M["rows"][c_]]["nslots"]), qf.shape[0])
                out.append((Kc[j], Vc[j], qf, n_, torch.cat([fv[j][None], loop.Wfs(qf[:n_]) + loop.spos[:n_]], 0),
                            M["rows"][c_]))
                continue
            out.append((Kc[j], Vc[j], Sc[j], int(M["nsl"][c_]), torch.cat([fv[j][None], fs[j, :int(M["nsl"][c_])]], 0),
                        M["rows"][c_]))
        return out

    # ------------------------------------------------------------------ frames in + INTEGRATE
    def leaf_meta(obj, kind):
        """per serialized pair: what it is in the truth (bookkeeping / evaluation only, never a write target)"""
        if kind == "state":
            return [("state",)] * len(serialize(obj, "state")[1])
        out = []
        grp = next(iter(obj))
        side = world.side.pop(id(obj), {})
        for lf in serialize(obj, "state")[1]:
            p = lf[0]
            if p in (("question",), ("note",)):
                out.append((p[0],))
            elif len(p) == 2 and p[0] == "rules":
                out.append(("rule", p[1], world.policy.get(p[1])))
            elif len(p) == 3 and p[0] == grp and isinstance(p[1], int) and isinstance(obj[grp], list) \
                    and isinstance(obj[grp][p[1]], dict) and "name" in obj[grp][p[1]]:
                nm = obj[grp][p[1]]["name"]
                val = obj[grp][p[1]][p[2]]
                dep = isinstance(val, str) and re.match(r"\d+ \(same as ", val) is not None
                out.append(side[(p[1], p[2])] if (p[1], p[2]) in side else ("name", nm) if p[2] == "name"
                           else ("noise",) if p[2] in world.noise
                           else ("fact", nm, p[2], int(val.split()[0]) if dep else val, dep))
            else:
                out.append(("state",))
        return out

    def see(objs, kind, soft):
        """Encode input frames once; their pairs become candidate nodes; the model integrates them. Returns the
        node ids per frame and the storage cost."""
        if not objs:
            return [], torch.zeros((), device=dev)
        encs = [doc_encoding(tok.ids, tok.decode, o, "state") for o in objs]
        too_long = [len(e[0]["ids"]) > model.max_pos for e in encs]
        if any(too_long):                                   # longer than the LM's context: a state frame is skipped
            assert kind == "state", f"a {kind} frame longer than the LM context ({model.max_pos})"
            see.skipped = getattr(see, "skipped", 0) + sum(too_long)
            objs = [o for o, t_ in zip(objs, too_long) if not t_]
            encs = [e for e, t_ in zip(encs, too_long) if not t_]
            if not objs:
                return [], torch.zeros((), device=dev)
        L = max(len(e[0]["ids"]) for e in encs)
        ids = torch.tensor([e[0]["ids"] + [0] * (L - len(e[0]["ids"])) for e in encs], device=dev)
        X = walker_states(model, ids).to(torch.bfloat16)
        nid_lists = [store.add_frame(enc["ids"], X[j, :len(enc["ids"])].clone(), enc, leaf_meta(o, kind), kind)
                     for j, ((enc, _), o) in enumerate(zip(encs, objs))]
        for o, nl in zip(objs, nid_lists):                    # which pairs share a record (observable structure)
            for n_, lf in zip(nl, serialize(o, "state")[1]):
                store.nodes[n_]["rec"] = (nl[0], tuple(lf[0][:-1]))
        if kind in ("question", "note"):                    # the current input: read now, kept like any node
            return nid_lists, torch.zeros((), device=dev)
        rows = [{"qnid": n} for nl in nid_lists for n in nl]   # every pair = a frame row (address -> read -> write)
        return nid_lists, run(rows, False, "frame", soft=soft)[0]

    def version(tgt):
        """the node's current value moves to a version node (time-stamped, linked); the node itself stays"""
        t = store.nodes[tgt]
        vn = store.new_node(t["lsrc"] if t["lsrc"] else t["vsrc"], t["vsrc"], "version", t["time"], t["nslots"])
        store.nodes[vn]["ver_of"] = tgt
        store.nodes[vn]["d"] = t["d"]
        store.nodes[vn]["rec"] = t.get("rec")
        store.nodes[vn]["rsrc"] = t.get("rsrc", t["vsrc"])
        return vn

    def propagate(D, rows, row_of, tgt, time_):
        """hard propagation: nodes whose r edge points at the replaced node take its new value source (the
        dependent's old value is versioned). D = (P(has a relation), top node) — the model's two decisions, as every
        other hard choice (argmax): the edge PROBABILITY was split over the source's copies (argmax right .97,
        probability < .5 -> 0 propagated at the step-500 eval)"""
        if D is None or tgt not in row_of or A.dep_follow != "eager":
            return 0
        has_, top_ = D
        n_ = 0
        for j in ((has_ > 0.5) & (top_ == row_of[tgt])).nonzero().flatten().tolist():
            nid = rows[j]
            nd = store.nodes[nid]
            if nid == tgt or not nd["alive"] or nd["kind"] not in ("item", "state"):
                continue
            version(nid)
            nd.setdefault("rsrc", nd["vsrc"])             # its reference stays: it still follows the same source
            nd["vsrc"], nd["nslots"], nd["time"] = store.nodes[tgt]["vsrc"], store.nodes[tgt]["nslots"], time_
            n_ += 1
        return n_

    def dep_matrix():
        with torch.no_grad():
            M0 = materialize()
            if M0 is None:
                return None, [], {}
            L_ = graph.ref_logits(M0["K"], M0["R"], M0["kmean"], M0["ver"], M0["HM"])
            D = (torch.sigmoid(-L_[:, -1]), L_[:, :-1].argmax(-1))
        return D, M0["rows"], M0["row_of"]

    # ------------------------------------------------------------------ truth lookups (bookkeeping, from provenance)
    def value_of(m):
        return None if not m else m[3] if m[0] == "fact" else m[1] if m[0] == "val" else None

    def holders():
        """(name, key, value) -> live nodes holding that value (identity from the node's LABEL source, value from
        its VALUE source); version nodes separately"""
        cur_, ver_ = {}, {}
        for i in store.alive():
            if i in store.post_gone or i in store.dropped:   # merged into a node by this step's write / dropped
                continue
            nd = store.nodes[i]
            lm = store.prov.get(nd["lsrc"]) if nd["lsrc"] else None
            vs = store.nodes[store.post_repl[i]]["vsrc"] if i in store.post_repl else nd["vsrc"]
            v = value_of(store.prov.get(vs))
            if lm and lm[0] == "fact" and v is not None:
                (ver_ if nd["kind"] == "version" or i in store.post_ver else cur_).setdefault((lm[1], lm[2], v), []).append(i)
        for key, nid in store.expect.items():
            if store.nodes[nid]["alive"]:
                cur_.setdefault(key, []).append(nid)
        return cur_, ver_

    def gold_node(H, nm, k, v, prefer_new=None):
        c_ = H.get((nm, k, v))
        if not c_:
            return None
        return max(c_, key=lambda i: store.nodes[i]["time"])

    def make_question(qrng, kind, H, Hv, ops=(1, 2), mode="train", pool=None):
        facts = [(nm, k) for (nm, k, v) in H if nm in world.ents and k in world.ents[nm] and world.ents[nm][k] == v]
        if pool is not None:
            facts = [f for f in facts if f in pool]
        nums = [f for f in facts if f[1] not in world.ent_keys]
        lazy_deps = []
        if A.dep_follow != "eager":                  # dependents held by ANY node (its value may be stale: it is
            have = {}                                #   followed through what it mentions, not by its stored value)
            for (nm_, k_, v_), ns_ in H.items():
                if (nm_, k_) in world.deps:
                    have.setdefault((nm_, k_), []).extend(ns_)
            for f_, ns_ in have.items():
                src_ = world.deps[f_]
                sv_ = world.ents.get(src_[0], {}).get(src_[1])
                if sv_ is not None and gold_node(H, src_[0], src_[1], sv_) is not None and (pool is None or f_ in pool):
                    lazy_deps.append((f_, max(ns_, key=lambda i: store.nodes[i]["time"])))
            facts = [f for f in facts if f not in world.deps]
        if kind == "lookup" and (facts or lazy_deps):
            if lazy_deps and (not facts or qrng.random() < len(lazy_deps) / (len(lazy_deps) + len(facts))):
                (nm, k), dn = qrng.choice(sorted(lazy_deps))
                s_nm, s_k = world.deps[(nm, k)]
                v = world.ents[s_nm][s_k]
                dv = value_of(store.prov.get(store.nodes[dn]["vsrc"]))
                return (f"What is {leaf(nm, k)}?", [("read", dn), ("read", gold_node(H, s_nm, s_k, v))], [dv, v], v)
            nm, k = qrng.choice(facts)
            v = world.ents[nm][k]
            return f"What is {leaf(nm, k)}?", [("read", gold_node(H, nm, k, v))], [v], v
        if kind == "before":
            cands = []
            for nm, k in facts:
                hs = world.hist.get((nm, k), [])
                if len(hs) >= 2 and (Hv.get((nm, k, hs[-2])) or H.get((nm, k, hs[-2]))):
                    cands.append((nm, k, hs[-2]))
            if not cands:
                return None
            nm, k, pv = qrng.choice(cands)
            node = gold_node(Hv, nm, k, pv) or gold_node(H, nm, k, pv)
            if node == gold_node(H, nm, k, world.ents[nm][k]):
                return None                                  # (soft phase: the same node is still both)
            return f"What was {leaf(nm, k)} before?", [("read", node)], [pv], pv
        if kind == "multi":                                  # coexisting values: both are read and combined
            cands = [f for f, vs in world.multi.items() if len(vs) == 2 and all(f + (v,) in H for v in vs)
                     and (pool is None or f in pool)]
            if not cands:
                return None
            nm, k = qrng.choice(sorted(cands))
            v1, v2 = world.multi[(nm, k)]
            return (f"What is the sum of both {words(k)} values of {nm}?",
                    [("read", gold_node(H, nm, k, v1)), ("read", gold_node(H, nm, k, v2)), ("op", 0, [1, 2])],
                    [v1, v2, v1 + v2], v1 + v2)
        if kind == "chain":
            cands = [(nm, k) for (nm, k) in facts if k in world.ent_keys]
            qrng.shuffle(cands)
            for nm, k in cands:
                o = world.ents[nm][k]
                nxt = [(o, k2) for k2 in world.ents.get(o, {}) if (o, k2, world.ents[o][k2]) in H]
                if nxt:
                    o, k2 = qrng.choice(nxt)
                    v = world.ents[o][k2]
                    return (f"What is the {words(k2)} of the {words(k)} of {nm}?",
                            [("read", gold_node(H, nm, k, world.ents[nm][k])), ("read", gold_node(H, o, k2, v))],
                            [o, v], v)
            return None
        if len(nums) < 2:
            return None
        t = rand_tree(qrng, nums, qrng.randint(*ops), mode)
        if t is None:
            return None
        if A.recurse:
            return rec_question(t, H)
        steps, vals = [], []

        held_in = {}                                      # a fact already in a register is pointed at, not re-read

        def go(n):
            if n[0] == "leaf":
                if (n[1], n[2]) in held_in:
                    return held_in[(n[1], n[2])]
                steps.append(("read", gold_node(H, n[1], n[2], world.ents[n[1]][n[2]])))
                vals.append(world.ents[n[1]][n[2]])
                held_in[(n[1], n[2])] = len(steps)
                return len(steps)
            regs_ = [go(c) for c in n[2]]
            ti = next(i for i, x in enumerate(TOOLS) if x[0] == n[1])
            steps.append(("op", ti, regs_)); vals.append(TOOLS[ti][1](*[vals[r - 1] for r in regs_]))
            return len(steps)
        go(t)
        return f"What is {render(t)}?", steps, vals, vals[-1]

    span_memo = {}

    def tok_spans(Q):
        """character end offset of every token of Q (the question node's value tokens = its slots)"""
        if Q not in span_memo:
            ids_ = tok.ids(Q)
            span_memo[Q] = [len(tok.decode(ids_[:i + 1])) for i in range(len(ids_))]
        return span_memo[Q]

    def char_to_tok(ends, a, b):
        s_ = next(i for i, e in enumerate(ends) if e > a)
        e_ = next((i for i, e in enumerate(ends) if e >= b), len(ends) - 1)
        return s_, e_ + 1

    def rec_question(t, H):
        """a question whose program is recursive: every row does <= one operation; an operand that is itself an
        operation is a SUB call on its words (a token span of the row's question). Returns (text, root steps,
        root values, answer, child rows [(span in root tokens, steps, values, answer, ops)])."""
        text, spans = render_sp(t)
        Q = f"What is {text}?"
        ends = tok_spans(Q)
        base = len("What is ")
        children = []

        def value(n):
            if n[0] == "leaf":
                return world.ents[n[1]][n[2]]
            ti = next(i for i, x in enumerate(TOOLS) if x[0] == n[1])
            return TOOLS[ti][1](*[value(c) for c in n[2]])

        def row_prog(n, tok0):
            steps, vals, regs_ = [], [], []
            held_r = {}                                  # a fact already read in this row is pointed at, not re-read
            for c in n[2]:                               # (read-once masked the re-read: CE ~1e4 -> loss spikes)
                if c[0] == "leaf" and (c[1], c[2]) in held_r:
                    regs_.append(held_r[(c[1], c[2])])
                    continue
                if c[0] == "leaf":
                    steps.append(("read", gold_node(H, c[1], c[2], world.ents[c[1]][c[2]])))
                    held_r[(c[1], c[2])] = len(steps)
                else:
                    a_, b_ = spans[id(c)]
                    ts, te = char_to_tok(ends, base + a_, base + b_)
                    steps.append(("sub", (ts - tok0, te - tok0)))
                    cs, cv = row_prog(c, ts)
                    children.append(((ts, te), cs, [None] + cv, value(c), count_ops(c)))
                vals.append(value(c))
                regs_.append(len(steps))
            ti = next(i for i, x in enumerate(TOOLS) if x[0] == n[1])
            steps.append(("op", ti, regs_))
            vals.append(value(n))
            return steps, vals

        steps, vals = row_prog(t, 0)
        return Q, steps, vals, vals[-1], children

    def count_ops(n):
        return 0 if n[0] == "leaf" else 1 + sum(count_ops(c) for c in n[2])

    def ask(qrng, specs, mix, ops=(1, 2), mode="train", soft=False):
        """specs: list of (n, pool) — n questions drawn from pool (None = any fact the memory holds)"""
        H, Hv = holders()
        kinds, wts = zip(*[(k, float(w)) for k, w in (x.split(":") for x in mix.split(","))])
        qs = []
        for n, pool in specs:
            got, tries = 0, 0
            while got < n and tries < 40 * n:
                tries += 1
                q = make_question(qrng, qrng.choices(kinds, wts)[0], H, Hv, ops, mode, pool)
                if q is not None and all(s_[1] is not None for s_ in q[1] if s_[0] == "read") and \
                        all(s_[1] is not None for ch in (q[4] if len(q) > 4 else []) for s_ in ch[1] if s_[0] == "read"):
                    qs.append(q); got += 1
        if not qs:
            return []
        nid_lists, _ = see([{"question": q[0]} for q in qs], "question", soft)
        out = []
        conv = lambda st: [(s_[0], s_[1], []) if s_[0] in ("read", "sub") else ("op", s_[1], s_[2]) for s_ in st]
        for q, nl in zip(qs, nid_lists):
            text, steps, vals, ans = q[:4]
            prog = [("read", nl[0], [])] + conv(steps)
            out.append({"text": text, "qnid": nl[0], "prog": prog, "vals": [None] + vals, "ans": ans,
                        "ops": sum(1 for s_ in steps if s_[0] == "op") + len(q[4] if len(q) > 4 else []),   # one op per row
                        "before": text.endswith("before?"), "multi": text.startswith("What is the sum of both")})
            if soft and len(q) > 4:                  # training: every sub-question is its own (teacher-forced) row
                for span, st, cv, cans, cops in q[4]:
                    out.append({"text": text, "qnid": nl[0], "span": span, "prog": conv(st), "vals": cv, "ans": cans,
                                "ops": cops, "before": False, "multi": False, "child": True})
        return out

    def corrections(crng, n):
        """text corrections: the note is the current input; its row reads it, finds the node, computes the new value
        (tool) and writes it at the node's address (the gate and mode are the model's; only reads are demonstrated)"""
        H, _ = holders()
        notes = []
        for _ in range(n):
            nm_ok = [(nm, k) for (nm, k, v) in H if k in world.num_keys and world.ents.get(nm, {}).get(k) == v]
            if not nm_ok:
                break
            obj, (nm, k, old, v) = world.correction(crng, nm_ok)
            tgt = gold_node(H, nm, k, old)
            if tgt is not None:
                notes.append((obj, nm, k, old, v, tgt))
        if not notes:
            return []
        nl, _ = see([x[0] for x in notes], "note", False)
        return [{"text": obj["note"], "qnid": ids_[0], "ans": v, "ops": 1, "edit": (nm, k), "before": False,
                 "prog": [("read", ids_[0], []), ("read", tgt, []), ("op", EDIT, [1, 0]), ("write", tgt, [])]
                         + ([("write", tgt, [])] if A.look else []),
                 "vals": [None, old, v, old]}
                for (obj, nm, k, old, v, tgt), ids_ in zip(notes, nl)]

    # ------------------------------------------------------------------ Walker's pass loop: ADDRESS -> READ -> (WRITE)
    def outcome(recs):
        """a row's hard write decision from its passes (the last one that writes): drop / new / attach j / replace j"""
        out = ("drop",)
        for r in recs:
            if r.get("tentative"):                # a proposal: the look pass after it decides
                continue
            if r["g"] < 0.5:                      # no write at the address: kept as its own node if NEW's gate keeps it
                if r["tgt"] is not None and r.get("g_new", 0.0) >= 0.5:
                    out = ("new",)
                continue
            tgt = r["tgt"]
            while tgt is not None and not store.nodes[tgt]["alive"] and tgt in store.merged:
                tgt = store.merged[tgt]
            if (tgt is None or tgt == r["src"] or r["p"] < A.conf or not store.nodes[tgt]["alive"]
                    or store.nodes[tgt]["kind"] not in ("item", "state")):
                out = ("new",)                    # NEW, or not confident where it belongs: its own node
            else:
                if A.id_guard and r["u"] >= 0.5 and not r.get("same", True):
                    out = ("new",)                    # a replace updates the SAME fact: into another fact it would
                else:                                 # corrupt it (its own label decides: identity cluster) -> own node
                    out = ("replace", tgt, r.get("absorb", [])) if r["u"] >= 0.5 else ("attach", tgt)
        return out

    def execute_rows(plans):
        """the hard decisions, in time order. plans: (source node, outcome)"""
        if not plans:
            return
        D, rows_, row_of_ = None, [], {}
        if any(o[0] == "replace" for _, o in plans):
            D, rows_, row_of_ = dep_matrix()          # dependence edges as they were before these writes
        for c, o in sorted(plans, key=lambda x: store.nodes[x[0]]["time"]):
            nd = store.nodes[c]
            if not nd["alive"]:
                continue
            if o[0] == "replace":
                tgt = o[1]
                while not store.nodes[tgt]["alive"] and tgt in store.merged:
                    tgt = store.merged[tgt]
                if not store.nodes[tgt]["alive"] or tgt == c:
                    continue                          # target gone meanwhile: keep the source as its own node
                t = store.nodes[tgt]
                tl = store.prov.get(t["lsrc"]) if t["lsrc"] else None          # diagnosis (bookkeeping): into its
                sl = store.prov.get(nd["lsrc"]) if nd["lsrc"] else None         # own fact or into another one?
                if tl and sl and tl[0] in ("fact", "delta") and sl[0] in ("fact", "delta"):
                    wstats.setdefault("replace into own fact" if tl[1:3] == sl[1:3] else "replace into OTHER fact",
                                      [0, 0.0, 0.0, 0.0])[0] += 1
                vn_ = getattr(store, "pending_ver", {}).pop(o[1], None)
                if vn_ is None or not store.nodes[vn_]["alive"] or store.nodes[vn_]["ver_of"] != tgt:
                    version(tgt)                      # (a replace decided in training made its version already)
                t["vsrc"], t["time"], t["nslots"] = nd["vsrc"], nd["time"], nd["nslots"]
                t["rsrc"] = nd.get("rsrc", nd["vsrc"])        # a new stated value: its own text decides what it follows
                t["usage"] += 1; t["act"] += 1.0
                if c in store.live:                   # the value it chose
                    dd = store.live.pop(c)
                    od = t["d"] or tuple(torch.zeros_like(x) for x in dd)
                    store.live[tgt] = (od[0], dd[1], od[2] + dd[2])   # label kept; value + relation move
                nd["alive"] = False
                store.merged[c] = tgt
                wstats.setdefault("propagated", [0, 0.0, 0.0, 0.0])[0] += propagate(D, rows_, row_of_, tgt, nd["time"])
                for a_ in (o[2] if len(o) > 2 else []):   # consolidation: the other copies it also addressed become
                    na = store.nodes[a_]                    # versions of the target (readable as "before", no longer
                    if na["alive"] and na["kind"] == "item" and a_ != tgt and a_ != c:   # current duplicates)
                        na["kind"], na["ver_of"] = "version", tgt
                        wstats.setdefault("absorbed copies", [0, 0.0, 0.0, 0.0])[0] += 1
            elif o[0] == "attach":
                nd["link"] = o[1]
            elif o[0] == "drop":
                nd["alive"] = False
        for vn_ in getattr(store, "pending_ver", {}).values():   # versions of replaces that did not happen
            store.nodes[vn_]["alive"] = False
        store.pending_ver = {}

    def commit_pending():
        execute_rows(pend)
        pend.clear()

    def ans_tok(v):
        """the tokens Walker says for a value: a number as " 12" (as stored), a STRING exactly as its slots store it
        inside the JSON quotes ("Gorpel" -> G|orp|el, no leading space): the say path reads slot t for answer token t,
        and " Gorpel" splits differently (" That|or" vs "Th|ator") -> names were said wrong after the first token"""
        ids_ = tok.ids(v) if isinstance(v, str) else tok.ids(" " + str(v))
        return ids_ + [EOS_ID] if A.ans_eos else ids_   # an answer ENDS (greedy decode ran on: 17 -> "1719")

    @torch.no_grad()
    def say_nll(rv, rs, answers):
        """per row: the model's own NLL of saying `answers` from value vectors rv + slots rs (no gradient)"""
        ys = [ans_tok(v)[:A.max_ans] for v in answers]
        Lm = max(map(len, ys))
        inp = torch.tensor([ans_ids + y[:-1] + [0] * (Lm - len(y)) for y in ys], device=dev)
        tgt_ = torch.full((len(ys), len(ans_ids) + Lm - 1), -100, dtype=torch.long, device=dev)
        for i, y in enumerate(ys):
            tgt_[i, a0:a0 + len(y)] = torch.tensor(y, device=dev)
        inject.set(mem.Wo(rv), ro.Us(padS(rs, Lm)), a0)
        lo, _ = model(inp, r_active=False)
        inject.clear()
        ce = F.cross_entropy(lo.float().transpose(1, 2), tgt_, ignore_index=-100, reduction="none")
        return ce.sum(-1) / (tgt_ != -100).sum(-1).clamp(min=1)

    def span_reg(reg, s_, e_):
        """a register holding the words [s_, e_) of a question register (a child row's question)"""
        n_ = int(reg[3])
        s_, e_ = max(0, min(s_, n_ - 1)), max(1, min(e_, n_))
        if e_ <= s_:
            e_ = s_ + 1
        S_ = reg[2][s_:e_]
        v_ = S_.mean(0)
        return (reg[0], v_, S_, S_.shape[0], torch.cat([loop.Wfv(v_)[None], loop.Wfs(S_) + loop.spos[:S_.shape[0]]], 0),
                None)

    def run(rows, teacher, kind="question", soft=False, depth=0):
        """Walker's loop for a batch of rows of one kind: 'frame' (a candidate pair to integrate; free-running),
        'note' (a text note: read it, read the target, compute, write), 'question'. Every pass: ADDRESS one of
        [memory nodes | NEW | tools | HALT] -> READ the node (value before this pass's write) / run the tool ->
        WRITE at the same address (frame / note rows): gate + replace-vs-attach from the same Walker state.
        Training (soft): a row's hard decision is what the step's later rows see (straight-through), the
        alternatives act softly so gradient reaches address, gate and mode. Returns (loss, em, action acc, said)."""
        B = len(rows)
        if not B:
            return torch.zeros((), device=dev), None, 0.0, None
        writer = kind in ("frame", "note")
        G = ({r["qnid"] for r in rows} | set(store.live)
             | {x for r in rows for kd, x, *_ in (r.get("prog") or []) if kd in ("read", "write")})
        M = materialize(grad=G)
        K, V, R, logg, kmean = M["K"], M["V"], M["R"], M["logg"], M["kmean"]
        N, nT = K.shape[0], len(TOOLS)
        I_NEW, I_HALT = N, N + 1 + nT
        I_SUB = I_HALT + 1                           # (only with --recurse; masked otherwise)
        tq = torch.tensor([float(store.nodes[r["qnid"]]["time"]) for r in rows], device=dev)
        age = (tq[:, None] - M["time"][None]).clamp(min=0)
        if kind == "frame":                          # a candidate integrates into OLDER item / state nodes
            okk = torch.tensor([store.nodes[i]["kind"] in ("item", "state") for i in M["rows"]], device=dev)
            visible = (M["time"][None] < tq[:, None]) & okk[None]
        else:
            visible = M["time"][None] <= tq[:, None]
        la = torch.log1p(age.float()) / 5.0
        age_f = torch.stack([(age < 0.75).float(), la, la * la], -1)            # (B, N, 3)
        regs = [[] for _ in range(B)]
        pre = [b for b in range(B) if rows[b].get("span") is not None]
        for b, rg in zip(pre, regs_from(M, [M["row_of"][rows[b]["qnid"]] for b in pre])):
            regs[b].append(span_reg(rg, *rows[b]["span"]))   # a training sub-question: words of its root question
        for b in range(B):
            if rows[b].get("reg0") is not None:
                regs[b].append(rows[b]["reg0"])          # a sub-question made during a free run
        has0 = [1 if regs[b] else 0 for b in range(B)]
        own = [[] for _ in range(B)]                 # registers that are write sources (not read from memory)
        if kind == "frame":                          # register 0 = the candidate (Walker reads it, 16 slots)
            for b, rg in enumerate(regs_from(M, [M["row_of"][r["qnid"]] for r in rows])):
                regs[b].append(rg[:4] + (rg[4][:1 + A.write_slots],) + rg[5:])
                own[b].append(0)
        acts = [[] for _ in range(B)]
        done = [False] * B
        read_mask = torch.zeros(B, N, dtype=torch.bool, device=dev)
        losses, n_act, ok_act, say = [], 0, 0, []
        recs = [[] for _ in range(B)]                # hard write records per row
        oprec = [[] for _ in range(B)]               # per tool use: (tool, arg register a, arg register b, decoded a, b)
        proposal = [None] * B                        # (record, address, register written, pass, proposed outcome)
        prop_P = {}                                  # b -> the proposal pass's address distribution
        prop_lg = {}                                 # b -> the proposal's (gate, mode, NEW-gate) logits
        soft_at = [None] * B                         # the row's soft write tensors (at its writing pass)
        made = []                                    # tool-output nodes of note rows (write sources)
        lookk = bool(A.look) and (kind == "note" or bool(A.look_frames))   # (frames: off by default — a look
        #   head shared with ~50 frame rows / step, mostly noise to decline, learned "lower the gate" for notes too)
        look = [0] * B                               # look before commit: 0 none, 1 proposed (next pass looks), 2 looked
        comp_reg = {}                                # b -> register of this pass's chosen computed value (object tools)
        R_ = A.refine if kind == "frame" else 0      # refine on write: R_ gather passes, one compose pass, then write
        w0 = R_ + 1 if R_ else 0                     # the first write pass (frame rows)
        comp_d = [None] * B                          # b -> compose deltas (k, v, r) added to the candidate's write
        if kind == "frame":
            T = w0 + A.write_passes + (1 if lookk else 0)
        else:
            T = max(len(r["prog"]) for r in rows) + 1 if teacher else max_regs
        tau = mem.tau.exp()
        lab = F.normalize(torch.cat([loop.new_lab[None], tools.labels, loop.halt[None], loop.sub_lab[None]]), dim=-1)
        stat_rows = []
        for t in range(T):
            if kind == "frame" and lookk and t >= w0 + A.write_passes:
                for b in range(B):                   # only a row that proposed a write into a node gets the look pass
                    if look[b] != 1:
                        done[b] = True
                if all(done):
                    break
            h = ctrl_states(regs)
            klast = torch.stack([row[-1][0] if row else torch.zeros(A.dk, device=dev) for row in regs])
            if kind == "frame":                      # "what would someone ask to get this, and who answers it?"
                klast = torch.stack([row[0][0] for row in regs])   # the candidate's own label (not a gathered node's)
                qr = loop.Wq(h) + loop.Wself(klast)
            else:
                qr = loop.Wq(h) + torch.sigmoid(loop.Wgq(h)) * loop.Wqr(klast)
            qn = F.normalize(qr - kmean, dim=-1)
            phase = "write" if (kind != "frame" or t >= w0) else ("gather" if t < R_ else "compose")
            if phase == "compose":                   # COMPOSE or skip (HALT label); the tool reads every register
                lc = tau * qn @ F.normalize(torch.stack([loop.compose_lab, loop.skip_lab]), dim=-1).t()
                pc = torch.softmax(lc, -1)[:, 0]
                qc = loop.Wcq(h)
                for b in range(B):
                    if done[b] or not regs[b]:
                        continue
                    kr = torch.stack([r_[0] for r_ in regs[b]])
                    vr_ = torch.stack([r_[1] for r_ in regs[b]])
                    inc = torch.sigmoid((F.normalize(loop.Wset(h[b]), dim=-1) * F.normalize(loop.Wrk(kr - kmean[0]), dim=-1)).sum(-1) * tau)
                    inc = torch.cat([torch.ones_like(inc[:1]), inc[1:]])          # the candidate itself is always in
                    att = torch.softmax((qc[b] * kr).sum(-1) / A.dk ** 0.5, 0) * inc
                    att = att / att.sum().clamp(min=1e-6)
                    ctx = torch.cat([(att[:, None] * kr).sum(0), (att[:, None] * vr_).sum(0), h[b]])
                    m_ = (1.0 if float(pc[b]) > 0.5 else 0.0) + pc[b] - pc[b].detach()   # hard use, soft gradient
                    raw_ = loop.Wcomp(loop.comp_ln(ctx)).split(A.dk)
                    r0 = regs[b][0]
                    cap_ = (A.compose_cap * r0[0].norm(), A.compose_cap * r0[1].norm(), A.compose_cap * r0[1].norm())
                    dk_, dv_, dr_ = (m_ * x_ * (c_x / x_.norm().clamp(min=1e-6)).clamp(max=1.0)   # a refinement:
                                     for x_, c_x in zip(raw_, cap_))                                  # <= cap x its size
                    comp_d[b] = (dk_ if A.compose_k else torch.zeros_like(dk_), dv_, dr_)
                    #                                   the candidate's register, refined (what the write head reads)
                    v_new = r0[1] + dv_
                    regs[b][0] = (r0[0] + comp_d[b][0], v_new, r0[2], r0[3],
                                  torch.cat([loop.Wfv(v_new)[None], r0[4][1:]], 0), r0[5])
                    acts[b].append(("compose", int(float(pc[b]) > 0.5)))
                    run.comp_stats.append((float(pc[b]), float(inc[1:].mean()) if len(inc) > 1 else 1.0,
                                           float(dv_.norm()), float(comp_d[b][0].norm())))
                continue
            if kind == "frame" and R_ and t == w0:
                read_mask.zero_()                    # the write may address the nodes it gathered
            cos_r = F.normalize(qr, dim=-1) @ F.normalize(R, dim=-1).t()
            age_b = (age_f * (graph.age[None, None] + loop.Wage(h)[:, None])).sum(-1)   # the query decides recency
            e = tau * qn @ K.t() + logg[None] + graph.w_rel * cos_r + age_b
            if kind == "frame":                      # its OWN label too, as an IDENTITY test: labels are identity
                idc = F.normalize(klast - kmean, dim=-1) @ K.t()   # codes (same fact .99, other entity same attr
                e = e + tau * graph.w_id * ((idc - A.cluster_cos) / (1 - A.cluster_cos)).clamp(min=0)   # <= .85):
                #   only a near-identical label gets the bonus; a new fact resembling OTHER entities' nodes gets none
            e = e.masked_fill(~visible | read_mask, -1e4)
            le = tau * qn @ lab.t()
            le = torch.cat([le[:, :1] + tau * loop.b_new, le[:, 1:]], -1)
            if kind == "question":
                le = torch.cat([torch.full_like(le[:, :1], -1e4), le[:, 1:]], -1)   # questions do not write
            if kind == "note":                       # NEW only once there is something to write (a tool's output):
                nosrc = torch.tensor([not own[b] for b in range(B)], device=dev)   # note rows otherwise picked NEW
                le = torch.cat([le[:, :1].masked_fill(nosrc[:, None], -1e4), le[:, 1:]], -1)   # forever (frame rows'
                #                                                                    NEW preference leaked into them)
            if kind == "frame":                      # no tools, and no HALT: "don't store" is NEW with a closed gate
                le = torch.cat([le[:, :1], torch.full_like(le[:, 1:], -1e4)], -1)   # (HALT ate facts: its label is
                #                                  trained by question rows, frame queries inherited its pull)
            if not (A.recurse and kind == "question" and depth < A.max_depth):
                le = torch.cat([le[:, :-1], torch.full_like(le[:, -1:], -1e4)], -1)   # no SUB
            logits = torch.cat([e, le], -1)          # [nodes | NEW | tools | HALT | SUB]
            sp_lg = None
            if A.recurse and kind == "question" and any(regs):   # span pointer over each row's question slots
                n0 = max(int(r_[0][3]) if r_ else 1 for r_ in regs)
                Sq = torch.stack([padS(r_[0][2][:int(r_[0][3])], n0) if r_ else torch.zeros(n0, A.dk, device=dev)
                                  for r_ in regs])
                kq = F.normalize(loop.Wsk(Sq), dim=-1)
                m0 = torch.tensor([[j < (int(r_[0][3]) if r_ else 0) for j in range(n0)] for r_ in regs], device=dev)
                sp_lg = [(tau * torch.einsum("bd,btd->bt", F.normalize(W_(h), dim=-1), kq)).masked_fill(~m0, -1e4)
                         for W_ in (loop.Wss, loop.Wse)]
            if phase == "gather":                    # gather: read existing nodes as context (no write, no NEW)
                logits = torch.cat([logits[:, :N], torch.full_like(logits[:, N:], -1e4)], -1)
            arg_lg = None
            if t and kind != "frame" and any(regs):
                nr = max(len(r_) for r_ in regs)
                Rv = torch.stack([torch.stack([x[1] for x in row] + [torch.zeros(A.dk, device=dev)] * (nr - len(row)))
                                  for row in regs])
                Rv = Rv * (50.0 / Rv.norm(dim=-1, keepdim=True).clamp(min=50.0))   # cap outliers only (one register
                rk = loop.Wrk(Rv) + loop.rpos[:nr]           # reached norm ~4800); normal values (10-30) unchanged
                msk = torch.tensor([[j < len(row) for j in range(nr)] for row in regs], device=dev)
                arg_lg = [(torch.einsum("bd,btd->bt", loop.Wa[i](h), rk) / A.dk ** 0.5).masked_fill(~msk, -1e4)
                          for i in range(2)]
            if teacher and kind != "frame":
                gold, gargs = [], []
                for b, r in enumerate(rows):
                    if t < len(r["prog"]):
                        kd, x, ra = r["prog"][t]
                        gold.append(M["row_of"].get(x, -100) if kd in ("read", "write") else I_SUB if kd == "sub"
                                    else N + 1 + x)
                        gargs.append(list(ra))
                        if kd == "sub" and sp_lg is not None:            # the span: start / end-1 token
                            for W_i, tgt_i in ((0, x[0]), (1, x[1] - 1)):
                                if not (0 <= tgt_i < sp_lg[W_i].shape[1]) or float(sp_lg[W_i][b, tgt_i]) < -1e3:
                                    run.masked_gold["span"] = run.masked_gold.get("span", 0) + 1   # outside the
                                    continue                                                      # register: skip
                                losses.append(F.cross_entropy(sp_lg[W_i][b:b + 1], torch.tensor([tgt_i], device=dev),
                                                              reduction="sum"))
                                n_act += 1; ok_act += int(int(sp_lg[W_i][b].argmax()) == tgt_i)
                    else:
                        gold.append(I_HALT if t == len(r["prog"]) else -100)
                        gargs.append([])
                g_ = torch.tensor(gold, device=dev)
                vv = g_ != -100
                g_l = g_.clone()                     # a demonstrated target on a MASKED choice (logit -1e4: read-once,
                masked_ = vv & (logits.gather(1, g_.clamp(min=0)[:, None]).squeeze(1) < -1e3)   # not visible ...)
                if bool(masked_.any()):              # gave CE ~1e4 for one row -> loss spikes (100-3000) and the
                    g_l[masked_] = -100              # global clip shrank the whole step: skipped and counted
                    for b_ in masked_.nonzero().flatten().tolist():
                        gb = int(g_[b_])
                        why = ("read: already read" if gb < N and bool(read_mask[b_, gb]) else
                               "read: not visible" if gb < N else "NEW" if gb == I_NEW else "HALT" if gb == I_HALT
                               else "SUB" if gb == I_SUB else "tool")
                        key_ = f"{kind}{' child' if rows[b_].get('span') is not None else ''} @pass {t}: {why}"
                        run.masked_gold[key_] = run.masked_gold.get(key_, 0) + 1
                if bool((g_l != -100).any()):
                    losses.append(F.cross_entropy(logits, g_l, ignore_index=-100, reduction="sum"))
                rd = [(b, g_[b].item()) for b in range(B) if t > 0 and 0 <= gold[b] < N]
                if rd and kind == "question":        # Wself(label of X) -> the query that read X (no new supervision)
                    xi = torch.tensor([g for _, g in rd], device=dev)
                    bi = torch.tensor([b for b, _ in rd], device=dev)
                    run.self_pairs.append((loop.Wself(M["kraw"][xi].detach()), qr[bi].detach(), kmean.detach()))
                hit = (logits.argmax(-1) == g_) & vv
                n_act += int(vv.sum()); ok_act += int(hit.sum())
                for name_, m_ in (("read", vv & (g_ < N)), ("tool", (g_ > N) & (g_ < I_HALT)), ("halt", g_ == I_HALT),
                                  ("sub", g_ == I_SUB)):
                    if bool(m_.any()):
                        run.per_pass[name_] = run.per_pass.get(name_, []) + [float((hit & m_).sum() / m_.sum())]
                if arg_lg is not None:
                    for i in range(2):
                        tg = torch.tensor([r_[i] if len(r_) > i else -100 for r_ in gargs], device=dev)
                        if bool((tg != -100).any()):
                            losses.append(F.cross_entropy(arg_lg[i], tg, ignore_index=-100, reduction="sum"))
                            v2 = tg != -100
                            n_act += int(v2.sum()); ok_act += int(((arg_lg[i].argmax(-1) == tg) & v2).sum())
                choice = gold
                args = [[r_[i] if len(r_) > i else 0 for r_ in gargs] for i in range(2)]
            else:
                choice = logits.argmax(-1).tolist()
                args = [arg_lg[i].argmax(-1).tolist() if arg_lg is not None else [0] * B for i in range(2)]
                if kind == "frame":                  # decide at the CLUSTER level: copies of the same fact split the
                    Pn = torch.softmax(logits, -1)   # address mass (3 copies x .22 lost to NEW at .31) -> the best
                    js = Pn[:, :N].argmax(-1)        # node's cluster of nearly identical labels competes as a whole
                    mcl = (Pn[:, :N] * ((K[js] @ K.t() > A.cluster_cos) & visible).float()).sum(-1)
                    for b in range(B):
                        if choice[b] >= N and float(mcl[b]) > float(Pn[b, choice[b]]):
                            choice[b] = int(js[b])
            if phase == "gather":                    # nothing readable (e.g. an empty memory): read nothing
                choice = [c_ if float(logits[b, c_]) > -1e3 else -1 for b, c_ in enumerate(choice)]
            lk_rows = [b for b in range(B) if lookk and look[b] == 1 and proposal[b] is not None
                       and len(proposal[b]) == 5 and not done[b]]
            for b in lk_rows:                        # the look pass decides AT the proposed address (commit / mode /
                choice[b] = proposal[b][1]           # NEW / discard); it does not address again (its own address
            #                                          was unlearned: confidence ~0 turned every correction into NEW)
            # ---- the write head (same Walker state, same address): source, gate, mode; HOW: a procedure
            if writer and phase == "write":
                srcreg = [regs[b][own[b][-1]] if own[b] else None for b in range(B)]
                zk = torch.zeros(A.dk, device=dev)
                xk = torch.stack([s_[0] if s_ is not None else zk for s_ in srcreg])
                xv = torch.stack([s_[1] if s_ is not None else zk for s_ in srcreg])
                cv = F.normalize(xv, dim=-1) @ F.normalize(V, dim=-1).t()
                ck = F.normalize(xk - kmean, dim=-1) @ K.t()
                if kind == "note":                   # a note's source is the edit output (key "result"): its identity
                    for b in range(B):               # is the fact it READ -> key match = best match of its read nodes
                        rk_ = [r_[0] for r_ in regs[b] if r_[5] is not None and r_[5] in M["row_of"]]
                        if rk_:
                            ck[b] = (F.normalize(torch.stack(rk_) - kmean, dim=-1) @ K.t()).max(0).values
                gw, mw = loop.gate_w, loop.mode_w
                gh, uh = loop.Wg(h), loop.Wum(h)
                sb = lambda x: 4 * torch.tanh(x / 4)  # bounded gate logits; raw logits drifting far past +-4 still
                gl_node = gh + gw[0] * cv + gw[1] * ck + gw[2]   # flattened the gradient (tanh saturates too: rules sat
                gl_new = gh.squeeze(-1) + gw[3]                  # at the .018 floor with state noise) -> a penalty keeps
                up, kap, pj = torch.zeros(B, 1, device=dev), None, None
                if kind == "frame":                  # HOW to write: a procedure node read from memory, or the default
                    qp = loop.Wp(h) + loop.Wpr(xk)
                    sp = (tau * F.normalize(qp - kmean, dim=-1) @ K.t() + logg[None]).masked_fill(~visible, -1e4)
                    sp = torch.cat([sp, (tau * loop.b_proc).expand(B, 1)], -1)    # [nodes | default]
                    pi = torch.softmax(sp, -1)
                    cand_p = torch.cat([V, loop.proc0[None]])
                    soft_p = pi @ cand_p
                    pj = sp.argmax(-1)
                    p_vec = cand_p[pj] + soft_p - soft_p.detach()                  # hard choice, soft gradient
                    pop = torch.softmax(loop.Wop(p_vec), -1)                     # HOW: [copy | add sub max min]
                    oph = pop.argmax(-1).tolist()
                    kap = pop[:, 0]
                    up = loop.Wup(p_vec)
                ul_node = uh + up + mw[0] * cv + mw[1] * ck + mw[2]
                if lk_rows:                          # LOOK = a correction of the proposal's own decision: the look
                    dl = loop.Wlook(h)               # state adds a delta (zero at init) to the proposal's gate / mode /
                    lk_m = torch.tensor([b in lk_rows for b in range(B)], device=dev)   # NEW-gate logits; the
                    pg = torch.stack([prop_lg[b][0] if b in lk_rows else gl_node[b] for b in range(B)])   # proposal
                    pu = torch.stack([prop_lg[b][1] if b in lk_rows else ul_node[b] for b in range(B)])   # heads keep
                    pn = torch.stack([prop_lg[b][2] if b in lk_rows else gl_new[b] for b in range(B)])    # gradient
                    gl_node = torch.where(lk_m[:, None], pg + dl[:, :1], gl_node)   # (re-using the heads on the look
                    ul_node = torch.where(lk_m[:, None], pu + dl[:, 1:2], ul_node)  # state shifted every gate down:
                    gl_new = torch.where(lk_m, pn + dl[:, 2], gl_new)               # main17 notes dropped 19/24)
                run.logit_pen = run.logit_pen + (F.relu(gl_new.abs() - 4) ** 2).mean()   # raw logits in the live range
                g_node = torch.sigmoid(sb(gl_node))
                g_new = torch.sigmoid(sb(gl_new))
                u_node = torch.sigmoid(sb(ul_node))
                P = torch.softmax(logits, -1)
                if lk_rows:                          # ... with the PROPOSAL's address weights (gradient to the address)
                    P = torch.stack([prop_P[b] if b in lk_rows else P[b] for b in range(B)])
                w_rep = P[:, :N] * g_node * u_node                                   # replace into j
                w_keep = P[:, I_NEW] * g_new + (P[:, :N] * (g_node * (1 - u_node)              # its own node: NEW,
                                                             + (1 - g_node) * g_new[:, None])).sum(-1)  # attach, or
                #                                                     declined at the address but kept (fallback)
                comp_now, alt_now = {}, {}
                if kind == "frame" and A.object_tools:   # the operation on (stored value, incoming value), exact
                    act_ = [b for b in range(B) if not done[b] and srcreg[b] is not None and choice[b] >= 0]
                    tj = {}                                  # b -> targets to evaluate: its choice (+ best other)
                    if act_:
                        wr_m = w_rep.detach().clone()
                        for b in act_:
                            tj[b] = []
                            if 0 <= choice[b] < N:
                                tj[b].append(choice[b]); wr_m[b, choice[b]] = -1.0
                            if soft:
                                tj[b].append(int(wr_m[b].argmax()))
                        sx = decode(xv[act_], stackS([srcreg[b][2] for b in act_]))
                        snum = dict(zip(act_, sx))
                        pairs = [(b, j) for b in act_ if snum[b] is not None for j in tj[b]]
                        if pairs:
                            jj = torch.tensor([j for _, j in pairs], device=dev)
                            tnum = dict(zip(pairs, decode(V[jj], M["S"][jj])))
                            todo = []                        # (b, j, op, value)
                            for b, j in pairs:
                                if tnum[(b, j)] is None:
                                    continue
                                for o in range(1, 1 + N_ARITH):
                                    hard = j == choice[b] and o == oph[b]
                                    if hard or soft:
                                        todo.append((b, j, o, TOOLS[o - 1][1](tnum[(b, j)], snum[b])))
                            enc_ = encode_values([x[3] for x in todo], [float(tq[x[0]]) for x in todo], True)
                            for (b, j, o, _), (nid, rg_) in zip(todo, enc_):
                                made.append((b, nid))
                                if j == choice[b] and o == oph[b]:
                                    comp_now[b] = nid
                                    comp_reg[b] = rg_
                                else:
                                    alt_now.setdefault(b, []).append((j, o, nid))
                for b in range(B):
                    c_ = choice[b]
                    if done[b] or srcreg[b] is None or c_ < 0:
                        continue
                    src = srcreg[b][5]
                    zk_ = torch.zeros(A.dk, device=dev)                       # (--write-deltas: which of label /
                    dd_ = (loop.Wdk(h[b]) if "k" in A.write_deltas else zk_,  # value / relation the model adjusts)
                           loop.Wdv(h[b]) if "v" in A.write_deltas else zk_,
                           loop.Wdr(h[b]) if "r" in A.write_deltas else zk_)  # the vectors the model writes:
                    if comp_d[b] is not None:                                  # + the compose tool's refinement
                        dd_ = tuple(a_ + c_d for a_, c_d in zip(dd_, comp_d[b]))
                    store.live[src] = dd_                                     # encoder proposal + its own deltas
                    if b in comp_now:
                        store.live[comp_now[b]] = dd_
                    same_id = (F.normalize(xk[b] - kmean[0], dim=-1) @ K.t() > A.cluster_cos).float()   # its own fact
                    if t == w0 and kind == "frame":
                        stat_rows.append((b, float(w_keep[b] + w_rep[b].sum()), float(P[b, I_NEW]),
                                          float(w_rep[b].sum()), int(pj[b]) < N, float(kap[b]) if kap is not None else 1.0))
                    if c_ > I_NEW:                   # a tool / HALT pass writes nothing: it must not replace the soft
                        continue                     # write of the pass that did (notes' soft write came from their
                    #                                  HALT pass: write / look heads got no gradient at their own state)
                    soft_at[b] = (src, w_keep[b], w_rep[b], comp_now.get(b), pop[b] if kap is not None else None,
                                  alt_now.get(b, []), oph[b] if kap is not None else 0, (w_rep[b] * same_id).sum())
                    if c_ < N:
                        s_ = comp_now.get(b, src)                     # the chosen operation's exact result
                        cl = (K @ K[c_] > A.cluster_cos) & visible[b]          # copies of the same thing (own keys)
                        p_cl = float((P[b, :N] * cl.float()).sum())             # confidence: "this fact, any copy"
                        absorb = [M["rows"][j] for j in (cl & (P[b, :N] > 0.05)).nonzero().flatten().tolist()
                                  if j != c_]
                        if kind == "note":               # an edit updates the fact the row READ (its source is a
                            same = M["rows"][c_] in {r_[5] for r_ in regs[b]}   # computed value without identity)
                        else:                            # a frame pair: its own label is the identity test
                            same = float(F.normalize(xk[b] - kmean[0], dim=-1) @ K[c_]) > A.cluster_cos
                        rec_ = {"src": s_, "tgt": M["rows"][c_], "g": float(g_node[b, c_]),
                                "u": float(u_node[b, c_]), "p": p_cl, "g_new": float(g_new[b]),
                                "absorb": absorb, "same": same}
                        if kind == "frame" and soft and A.w_commit_frames and t == w0:   # "if I write this here, is
                            tl_ = store.prov.get(store.nodes[M["rows"][c_]]["lsrc"]) if store.nodes[M["rows"][c_]]["lsrc"] else None
                            if tl_ and tl_[0] == "fact" and tl_[2] in world.ents.get(tl_[1], {}):   # that fact still
                                wv_ = comp_reg[b] if b in comp_now else srcreg[b]                   # answered right?"
                                run.fcommit.append((sb(gl_node[b, c_]), sb(ul_node[b, c_]), V[c_].detach(),
                                                    M["S"][c_].detach(), wv_[1].detach(), wv_[2].detach(),
                                                    world.ents[tl_[1]][tl_[2]], tl_[2] in world.multi_keys))
                        if kind == "note" and soft and A.w_commit and not (lookk and look[b] == 0):
                            run.commit_items[b] = (sb(gl_node[b, c_]), sb(ul_node[b, c_]), V[c_].detach(), M["S"][c_].detach(),
                                                   srcreg[b][1].detach(), srcreg[b][2].detach(), rows[b]["ans"])
                        if lookk and look[b] == 0:        # the first write into a node is a PROPOSAL: tentative,
                            rec_["tentative"] = True      # shown to the row (look register), decided at the next pass
                            look[b] = 1
                            proposal[b] = (rec_, c_, comp_reg.get(b) if b in comp_now else srcreg[b])
                            prop_P[b] = P[b]
                            prop_lg[b] = (gl_node[b], ul_node[b], gl_new[b])
                        elif look[b] == 1:
                            rec_["look"] = True
                        recs[b].append(rec_)
                    else:
                        recs[b].append({"src": src, "tgt": None, "g": float(g_new[b]), "u": 0.0, "p": float(P[b, c_])})
            for b in range(B):                       # the pass after a proposal was the look pass (whatever it chose)
                if look[b] == 1 and proposal[b] is not None and len(proposal[b]) == 5 and proposal[b][3] < t:
                    look[b] = 2
            # ---- apply: read / NEW / tool / halt
            apply, reads, subs = {}, [], []
            for b in range(B):
                c_ = choice[b]
                if done[b] or c_ < 0:
                    continue
                if c_ < N:
                    reads.append((b, c_))
                    acts[b].append(("read", M["rows"][c_]))
                    store.nodes[M["rows"][c_]]["usage"] += 1
                    store.nodes[M["rows"][c_]]["act"] += 1.0
                    if kind == "question" or phase == "gather":
                        read_mask[b, c_] = True               # read-once (write rows may address again)
                elif c_ == I_NEW:
                    acts[b].append(("new", None))
                elif c_ == I_SUB:
                    if teacher:
                        span_ = rows[b]["prog"][t][1]
                    else:
                        s_ = int(sp_lg[0][b].argmax()) if sp_lg is not None else 0
                        e_ = max(s_, int(sp_lg[1][b].argmax()) if sp_lg is not None else 0) + 1
                        span_ = (s_, e_)
                    subs.append((b, span_))
                    acts[b].append(("sub", span_))
                elif c_ == I_HALT or not regs[b]:
                    done[b] = True
                else:
                    apply.setdefault(c_ - N - 1, []).append(b)
                    acts[b].append(("op", c_ - N - 1))
            for (b, _), rg in zip(reads, regs_from(M, [c_ for _, c_ in reads])):
                regs[b].append(rg)
            if subs:                                 # SUB: the child's answer becomes this row's next register
                if teacher:                          # (training: the demonstrated answer, encoded like a tool output)
                    enc_ = encode_values([rows[b]["vals"][len(regs[b])] for b, _ in subs],
                                         [float(tq[b]) + 0.5 for b, _ in subs], False)
                    for (b, _), (_, rg) in zip(subs, enc_):
                        regs[b].append(rg)
                else:                                # free run: actually solve the sub-question (recursion)
                    kids = [{"qnid": rows[b]["qnid"], "reg0": span_reg(regs[b][0], *sp_), "prog": None, "ans": 0,
                             "vals": None, "ops": 0, "before": False} for b, sp_ in subs]
                    keep_ = {k_: getattr(run, k_, None) for k_ in ("acts", "oprec", "said", "diag", "final_regs")}
                    with torch.no_grad():
                        run(kids, False, "question", soft=False, depth=depth + 1)
                    kid_regs = run.final_regs
                    for k_, v_ in keep_.items():
                        setattr(run, k_, v_)
                    for (b, _), rg in zip(subs, kid_regs):
                        regs[b].append(rg if rg is not None else regs[b][0])
            if lookk:                                # LOOK: "if I write this, what happens?" -> the target as it would
                fresh = [b for b in range(B) if look[b] == 1 and proposal[b] is not None and len(proposal[b]) == 3]
                for b, rg in zip(fresh, regs_from(M, [proposal[b][1] for b in fresh])):
                    rec_, c_, wreg = proposal[b]     # read after the commit: the written value (replace), else its
                    o_ = outcome([{k_: v_ for k_, v_ in rec_.items() if k_ != "tentative"}])   # old value unchanged
                    vr = wreg if (o_[0] == "replace" and wreg is not None) else rg
                    lat = vr[4] + F.pad(loop.look_vec[None], (0, 0, 0, vr[4].shape[0] - 1))
                    regs[b].append((rg[0], vr[1], vr[2], vr[3], lat, None))
                    proposal[b] = (rec_, c_, wreg, t, o_[0])
            for ti, rws in apply.items():
                pick = lambda i, b: regs[b][min(args[i][b], len(regs[b]) - 1)]
                arg_ix = {b: (min(args[0][b], len(regs[b]) - 1), min(args[1][b], len(regs[b]) - 1)) for b in rws}
                dec_ = {}
                a_ = tuple((stackS if k == 2 else torch.stack)([pick(0, b)[k] for b in rws]) for k in range(3))
                b_ = tuple((stackS if k == 2 else torch.stack)([pick(1, b)[k] for b in rws]) for k in range(3))
                a_ = (a_[0], a_[1], padS(a_[2], A.tool_slots))
                b_ = (b_[0], b_[1], padS(b_[2], A.tool_slots))
                ol, ov, orr, osl = tools(torch.full((len(rws),), ti, device=dev, dtype=torch.long), a_, b_)
                fv, fs = loop.Wfv(ov), loop.Wfs(osl) + loop.spos[:A.tool_slots]
                obj = {}
                if A.object_tools and ti == EDIT:               # object edit: the value the note states (its tail)
                    nv = decode(b_[1], stackS([tail(pick(1, b)) for b in rws]))
                    okr = [(j, b) for j, b in enumerate(rws) if nv[j] is not None]
                    enc_ = encode_values([nv[j] for j, _ in okr], [float(tq[b]) + 0.5 for _, b in okr], writer)
                    obj = {b: rg for (_, b), (_, rg) in zip(okr, enc_)}
                if A.object_tools and ti < N_ARITH:              # object tool: decode both, exact, encode the result
                    xs = decode(torch.cat([a_[1], b_[1]]), stackS([pick(0, b)[2] for b in rws]
                                                                  + [pick(1, b)[2] for b in rws]))
                    dec_ = {b: (xs[j], xs[len(rws) + j]) for j, b in enumerate(rws)}
                    okr = [(j, b) for j, b in enumerate(rws) if xs[j] is not None and xs[len(rws) + j] is not None]
                    enc_ = encode_values([TOOLS[ti][1](xs[j], xs[len(rws) + j]) for j, _ in okr],
                                         [float(tq[b]) + 0.5 for _, b in okr], writer)
                    obj = {b: rg for (_, b), (_, rg) in zip(okr, enc_)}
                for b in rws:
                    oprec[b].append((ti,) + arg_ix[b] + dec_.get(b, (None, None)))
                for j, b in enumerate(rws):
                    if b in obj:
                        regs[b].append(obj[b])
                        if writer:
                            made.append((b, obj[b][5]))
                            own[b].append(len(regs[b]) - 1)
                        if teacher and kind == "question":
                            say.append((b, len(regs[b]) - 1, rows[b]["vals"][len(regs[b]) - 1]))
                        continue
                    nid = None
                    if writer:                                # a result the row may write: a node-to-be
                        key = store.add_vec((ol[j], ov[j], orr[j], osl[j]), ("val", rows[b].get("ans")))
                        nid = store.new_node(None, key, "item", float(tq[b]) + 0.5, A.tool_slots)
                        made.append((b, nid))
                        own[b].append(len(regs[b]))
                    regs[b].append((ol[j], ov[j], osl[j], A.tool_slots, torch.cat([fv[j][None], fs[j]], 0), nid))
                    if teacher and kind == "question":
                        say.append((b, len(regs[b]) - 1, rows[b]["vals"][len(regs[b]) - 1]))
            if all(done):
                break
        # ---- write decisions: hard plan now; this step's later rows see exactly the hard result (straight-through:
        #      the chosen write has value 1, every alternative value 0 — all of them carry gradient)
        cost = torch.zeros((), device=dev)
        if writer:
            plans = []
            computed = {n for _, n in made} if kind == "frame" else set()
            if kind == "note" and soft:
                for b in range(B):
                    for r_ in recs[b]:
                        if r_["tgt"] is not None:
                            ph = "proposal" if r_.get("tentative") else "look" if r_.get("look") else "write"
                            run.note_rec.setdefault(ph, []).append((r_["g"], r_["u"], r_["p"], r_["g_new"], float(r_["same"]),
                                                                    float(r_["tgt"] == rows[b]["prog"][1][1])))
            if lookk:
                for b in range(B):
                    if proposal[b] is not None and len(proposal[b]) == 5:
                        fin = [r_ for r_ in recs[b] if not r_.get("tentative")]
                        run.look_tr[kind + ": " + proposal[b][4] + " -> " + (outcome(fin)[0] if fin else "halt")] += 1
            for b in range(B):
                row_src = ([rows[b]["qnid"]] if kind == "frame" else []) + [n for bb, n in made if bb == b]
                plans_b = [(c, outcome([r_ for r_ in recs[b] if r_["src"] == c])) for c in row_src]
                plans += plans_b
                for c, o in plans_b:                          # salience: how sure the model was to keep it
                    gs_ = [r_["g"] for r_ in recs[b] if r_["src"] == c and r_["g"] >= 0.5]   # (a write counts as one
                    if o[0] in ("new", "attach") and gs_:                                     # use: act stays 1)
                        store.nodes[c]["g"] = gs_[-1]
                if not soft:
                    continue
                sa = soft_at[b]
                if sa is None:                                # never at a write pass: dropped
                    for c, _ in plans_b:
                        store.soft[c] = torch.tensor(math.log(1e-4), device=dev)
                        store.empty[c] = torch.zeros((), device=dev)
                    continue
                src0, wk, wr, comp_n, po, alts, oh, wr_same = sa
                lk = torch.log(wk.clamp(min=1e-4))
                rep_ = None
                for c, o in plans_b:
                    if o[0] == "replace":                     # the value moves INTO the target (identity kept)
                        if A.early_version and o[1] not in store.pending_ver:   # its old value becomes a version NOW (askable as
                            store.pending_ver[o[1]] = version(o[1])   # "before" this step; at commit it was too late:
                        #   replaced facts' "before" got no training -> corrected "before" .78 -> .08, main20 @500)
                        store.soft[c] = torch.tensor(-1e4, device=dev)
                        store.post_ver.update(o[2] if len(o) > 2 else [])   # absorbed copies: versions after commit
                        store.post_gone.add(c)
                        rep_ = (c, o[1])
                    elif c in computed and o[0] == "drop":    # an unused computation is not a memory
                        store.soft[c] = torch.tensor(-1e4, device=dev)
                    elif o[0] in ("new", "attach"):           # stored: present; the gate learns only from whether
                        store.soft[c] = torch.zeros((), device=dev)   # its VALUE helps (not from being a distractor)
                        store.empty[c] = 1.0 + wk - wk.detach()
                    else:                                     # dropped: as at inference, its value is gone (forward:
                        store.soft[c] = lk.detach()           # the empty value); gradient: would storing have helped
                        store.empty[c] = wk - wk.detach()
                        store.dropped.add(c)                  # not asked about (reading empty values wrecked the readout)
                if rep_ is not None and rep_[1] in M["row_of"]:
                    c, tgt = rep_
                    j = M["row_of"][tgt]
                    tl = store.prov.get(store.nodes[tgt]["lsrc"]) if store.nodes[tgt]["lsrc"] else None
                    tv = value_of(store.prov.get(store.nodes[tgt]["vsrc"]))
                    if tl and tl[0] == "fact" and world.ents.get(tl[1], {}).get(tl[2]) == tv:
                        store.expect[(tl[1], tl[2], tv)] = tgt   # a still-true fact overwritten: askable, so the
                        #                                           question blames the write (collateral)
                    m = 1.0 + wr[j] - wr[j].detach()
                    store.post_repl[tgt] = c
                    if po is not None:                        # the chosen operation (copy = index 0)
                        oc = po[oh]
                        store.mix.append((tgt, c, m * (1 + oc - oc.detach())))
                        if oh != 0:                           # copying instead: value 0, gradient
                            a_ = m * po[0]
                            store.mix.append((tgt, src0, a_ - a_.detach()))
                    else:
                        store.mix.append((tgt, c, m))
                top = wr.topk(min(2, N))                      # targets it did NOT choose: value 0, gradient
                for wj, j in zip(top.values, top.indices.tolist()):
                    if rep_ is not None and rep_[1] == M["rows"][j]:
                        continue
                    wc = wj * po[0] if po is not None else wj                   # ... replaced by a copy
                    store.mix.append((M["rows"][j], src0, wc - wc.detach()))
                for j, o, nid in alts:                        # ... replaced by another operation's exact result
                    if rep_ is not None and rep_[1] == M["rows"][j] and o == oh:
                        continue
                    wa = wr[j] * po[o]
                    store.mix.append((M["rows"][j], nid, wa - wa.detach()))
                cost = cost + A.cost_new * (wk if rep_ is None else wk.detach())
                m0 = store.prov.get(store.nodes[src0]["vsrc"], ("?",))
                if kind == "frame" and store.nodes[src0]["kind"] == "item" and m0[0] in ("fact", "delta", "rule"):
                    cost = cost - A.w_pend * torch.log((wk + wr_same).clamp(1e-4, 1.0))   # pending content is what is
                    #   to be remembered (user's two streams): kept as its own node or replaced INTO ITS OWN FACT (a
                    #   replace into another fact is not "kept": counting it taught P(NEW)=0 + corrupting replaces)
            if soft:
                pend.extend(plans)
                cost = cost / A.questions                    # per stored pair, on the scale of one question's loss
            else:
                execute_rows(plans)
                store.commit()
                run.last_plans = plans
                run.last_recs = recs
            if kind == "frame":
                for b, ws, pn, wrs, rule_used, kp_ in stat_rows:
                    m = store.prov.get(store.nodes[rows[b]["qnid"]]["vsrc"], ("?",))
                    kd = m[0] if m[0] != "fact" else "multi" if m[2] in world.multi_keys else \
                        "dependent" if m[4] else "fact"
                    w_ = wstats.setdefault(kd, [0, 0.0, 0.0, 0.0])
                    w_[0] += 1; w_[1] += ws; w_[2] += pn; w_[3] += wrs
                    if kd in ("delta", "fact"):               # HOW: rule read from memory, copy (vs compute)
                        w_ = wstats.setdefault("how " + kd + " (n, rule read, copy)", [0, 0.0, 0.0, 0.0])
                        w_[0] += 1; w_[1] += float(rule_used); w_[2] += kp_
                for _, o in plans:
                    wstats.setdefault("decided " + o[0], [0, 0.0, 0.0, 0.0])[0] += 1
                run.plans = plans
                if soft and run.fcommit:                      # writing into a node must not make ITS fact worse: the
                    it_ = run.fcommit                         # node's fact answered from its value as it is vs as
                    run.fcommit = []                          # written (into another fact: worse -> don't; an update
                    l_old = say_nll(torch.stack([x[2] for x in it_]), stackS([x[3] for x in it_]), [x[6] for x in it_])
                    l_new = say_nll(torch.stack([x[4] for x in it_]), stackS([x[5] for x in it_]), [x[6] for x in it_])
                    lab = (l_new <= l_old + A.commit_margin).float()   # / a restatement: not worse -> write)
                    glc = torch.stack([x[0] for x in it_]).reshape(-1)
                    ulc = torch.stack([x[1] for x in it_]).reshape(-1)
                    single = torch.tensor([not x[7] for x in it_], device=dev)
                    c_l = F.binary_cross_entropy_with_logits(glc, lab)
                    pm = (lab > 0) & single                   # replace for a single-valued fact it does not hurt
                    if bool(pm.any()):
                        c_l = c_l + F.binary_cross_entropy_with_logits(ulc[pm], lab[pm])
                    cost = cost + A.w_commit_frames * c_l
                    run.fcommit_stats.append((float(lab.mean()), float(torch.sigmoid(glc).mean()), len(it_)))
                return cost, None, 0.0, None
        # ---- Walker says the answer (questions) / the computed new value (notes)
        if kind == "note":
            first_tool = [next((i for i, x in enumerate(regs[b]) if any(m_ == (b, x[5]) for m_ in made)), None)
                          for b in range(B)]
            last = [ft if ft is not None else len(regs[b]) - 1 for b, ft in enumerate(first_tool)]
        else:
            last = [len(r["prog"]) - 1 + has0[b] if teacher else len(regs[b]) - 1 for b, r in enumerate(rows)]
        targets = [(b, last[b], rows[b]["ans"]) for b in range(B)] + say
        zero = (torch.zeros(A.dk, device=dev), torch.zeros(1, A.dk, device=dev))
        rv = torch.stack([regs[b][r_][1] if regs[b] and 0 <= r_ < len(regs[b]) else zero[0] for b, r_, _ in targets])
        rs = stackS([regs[b][r_][2] if regs[b] and 0 <= r_ < len(regs[b]) else zero[1] for b, r_, _ in targets])
        ys = [ans_tok(v)[:A.max_ans] for _, _, v in targets]
        Lm = max(map(len, ys))
        inp = torch.tensor([ans_ids + y[:-1] + [0] * (Lm - len(y)) for y in ys], device=dev)
        tgt_ = torch.full((len(ys), len(ans_ids) + Lm - 1), -100, dtype=torch.long, device=dev)
        for i, y in enumerate(ys):
            tgt_[i, a0:a0 + len(y)] = torch.tensor(y, device=dev)
        inject.set(mem.Wo(rv), ro.Us(padS(rs, Lm)), a0)
        lo, _ = model(inp, r_active=False)
        inject.clear()
        lo = lo.float()
        say_loss = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tgt_.reshape(-1), ignore_index=-100)
        em = ((lo.argmax(-1) == tgt_) | (tgt_ == -100) | (tgt_ == EOS_ID)).all(-1).float()   # value tokens decide
        #                                                                                    (the end token is trained)
        pr = lo.argmax(-1)
        run.said = [tok.decode(pr[i, a0:a0 + len(ys[i])].tolist()) for i in range(B)]
        act = sum(losses) / max(1, n_act) if losses else torch.zeros((), device=dev)
        run.diag = []
        for b, r in enumerate(rows):
            gold_reads = [x for kd, x, *_ in (r["prog"] or []) if kd in ("read", "write")]
            mine = {x for kd, x in acts[b] if kd == "read"}
            gold_seq = [("read" if kd in ("read", "write") else kd, x) for kd, x, *_ in (r["prog"] or [])]
            run.diag.append((sum(x in mine for x in gold_reads) / max(1, len(gold_reads)),
                             float(acts[b][:len(gold_seq)] == gold_seq and len(acts[b]) == len(gold_seq))))
        run.acts = acts
        run.oprec = oprec
        run.final_regs = [regs[b][-1] if regs[b] else None for b in range(B)]
        if kind == "question" and run.self_pairs:           # contrastive (plain cosine collapsed to one direction):
            ws_ = torch.cat([x[0] for x in run.self_pairs])  # Wself(label) must match ITS question's query among the
            qs_ = torch.cat([x[1] for x in run.self_pairs])  # batch's queries, in the centered space reads use
            km_ = run.self_pairs[0][2]
            lg_ = F.normalize(ws_ - km_, dim=-1) @ F.normalize(qs_ - km_, dim=-1).t() / A.inv_temp
            run.self_loss = F.cross_entropy(lg_, torch.arange(lg_.shape[0], device=dev))
            run.self_pairs = []
        return act + say_loss + cost, em[:B], ok_act / max(1, n_act), (em[B:].mean().item() if len(em) > B else None)

    run.per_pass = {}
    run.masked_gold = {}
    run.commit_items = {}
    run.commit_stats = []
    run.comp_stats = []
    run.forget_stats = []
    run.fcommit, run.fcommit_stats = [], []
    import collections as _co
    run.look_tr = _co.Counter()
    run.note_rec = {}
    run.self_loss = 0.0
    run.logit_pen = 0.0
    run.self_pairs = []

    def forget_feats(nids):
        """observations the forget head weighs (no rule decides with them)"""
        return torch.tensor([[float(store.nodes[i]["kind"] == "version"), math.log1p(store.nodes[i]["usage"]),
                              math.log1p(max(0.0, store.time - store.nodes[i]["time"])),
                              math.log(store.nodes[i]["act"] + 1e-3), store.nodes[i].get("g", 1.0),
                              float(store.nodes[i]["kind"] == "state"), float(store.nodes[i]["kind"] == "question")]
                             for i in nids], device=dev)

    def forget_choice(t_before, grad):
        """the model's forget decision: over capacity, the nodes with the lowest keep logits go. Candidates: nodes
        older than this step (this step's writes are decided by the store gate). Returns (nids, keep logits, chosen)"""
        live = store.alive()
        old_n = sum(1 for i in live if store.nodes[i]["time"] < t_before)
        kept_new = sum(1 for _, o in pend if o[0] in ("new", "attach"))   # this step's candidates the write keeps
        over = (old_n + kept_new if pend else len(live)) - A.cap         # (smoke cap 150: counting every candidate
        #                                                                   forgot 133 of 170 nodes)
        if over <= 0:
            return [], None, set()
        with torch.no_grad():
            M = materialize()
        cand = [j for j, i in enumerate(M["rows"]) if store.nodes[i]["time"] < t_before - A.protect]
        if not cand:
            return [], None, set()
        nids = [M["rows"][j] for j in cand]
        with torch.set_grad_enabled(grad):
            z = graph.forget_logit(M["K"][torch.tensor(cand, device=dev)].detach(), forget_feats(nids))
        gone = {nids[j] for j in torch.argsort(z.detach())[:over].tolist()}
        return nids, z, gone

    def soft_forget(t_before):
        """training: this step's forget decision acts BEFORE the questions (forgotten = empty value, straight-through
        keep weight sigmoid(centered logit)), so a question about a forgotten fact blames the decision"""
        nids, z, gone = forget_choice(t_before, True)
        store.evicting = {}
        if not gone:
            return set()
        zc = z - z.mean()                                # competition: keeping one means forgetting another
        sg = torch.sigmoid(zc)
        hard = torch.tensor([0.0 if i in gone else 1.0 for i in nids], device=dev)
        fw = hard + sg - sg.detach()
        store.evicting = {i: fw[j] for j, i in enumerate(nids)}
        run.forget_stats.append(len(gone))
        return gone

    def evict(t_before=None, gone=None):
        """execute the forget decision (the model's head; in training the one the questions were graded on)"""
        store.decay()
        for i in (gone or ()):
            store.nodes[i]["alive"] = False
        _, _, more = forget_choice(store.time + 1 if t_before is None else t_before, False)   # still over: the head
        for i in more:                                                                     # decides again
            store.nodes[i]["alive"] = False
        store.evicting = {}
        store.evict(10 ** 9, lambda i: 0.0)              # (frees the frames / vectors no live node uses)
        for s_ in [s_ for s_ in cache["src"] if s_[1] not in store.frames]:
            del cache["src"][s_]

    # ------------------------------------------------------------------ evaluation (a snapshot of the world)
    logf = open(A.log, "a", encoding="utf-8")

    def write(objs, soft, kind="item"):
        return see(objs, kind, soft)[1]

    def expect_dependents(changed):
        """soft phase: a fact that changed this step but whose new value no node holds (a dependent whose source
        changed, a total whose increment was stored as its own node, ...) is still asked — graded on its OWN node
        (the one with the stale value): that is where a write should have put the new value, so the question sends
        gradient to the write alternatives (replace / compute / propagate) instead of never being asked"""
        H, _ = holders()
        for f in changed:
            if f not in world.ents.get(f[0], {}):
                continue
            if A.dep_follow != "eager" and f in world.deps:   # lazy: a dependent is asked by its 2-read chain
                continue
            nm, k = f
            v, hs = world.ents[nm][k], world.hist[f]
            if (nm, k, v) in H or len(hs) < 2:
                continue
            node = gold_node(H, nm, k, hs[-2])
            if node is not None:
                store.expect[(nm, k, v)] = node

    def snapshot():
        return (copy.deepcopy(world.__dict__), copy.copy(store.frames), copy.deepcopy([dict(n, d=None) for n in store.nodes]),
                [n["d"] for n in store.nodes], store.time, dict(store.vecs), dict(store.prov), store.next_fid,
                store.next_vec, dict(store.merged))

    def restore(sn):
        world.__dict__.update(sn[0]); store.frames = sn[1]
        store.nodes = [dict(n, d=dd) for n, dd in zip(sn[2], sn[3])]
        store.time, store.vecs, store.prov, store.next_fid, store.next_vec = sn[4], sn[5], sn[6], sn[7], sn[8]
        store.merged = sn[9]
        store.live, store.soft, store.mix, store.expect, store.empty = {}, {}, [], {}, {}
        store.post_repl, store.post_gone, store.post_ver, store.dropped = {}, set(), set(), set()
        store.pending_ver = {}

    @torch.no_grad()
    def evaluate(step):
        nonlocal world, store
        mem.eval()
        sn = snapshot()
        er = random.Random(50_000 + step)
        out = {}

        op_examples = []

        def why_wrong(q, acts_b, ops_b, said=None):
            """where a wrong answer's program first went wrong (eval diagnosis)"""
            gold = [("read" if kd in ("read", "write") else kd, x) for kd, x, _ in q["prog"]]
            n_reads = sum(1 for g in gold if g[0] == "read") - 1        # operands (the question is read first)

            def fact_of(i):
                nd_ = store.nodes[i]
                lm_ = store.prov.get(nd_["lsrc"]) if nd_["lsrc"] else None
                return (lm_[1], lm_[2]) if lm_ and lm_[0] == "fact" else None, nd_["kind"]

            for pos, (g, a_) in enumerate(zip(acts_b, gold)):
                if g != a_ and g[0] == a_[0] == "sub":
                    return f"wrong sub-question span @{pos}"
                if g != a_:
                    if g[0] == a_[0] == "read":
                        (fg, kg), (fw, kw) = fact_of(a_[1]), fact_of(g[1])
                        golds = {fact_of(x)[0] for kd, x in gold[1:] if kd == "read"}
                        what = ("a version" if kw == "version" else "same fact, other node" if fw == fg and fw
                                else "another operand (order)" if fw in golds and fw else "same entity" if fw and fg and fw[0] == fg[0]
                                else "same attribute" if fw and fg and fw[1] == fg[1] else "unrelated")
                        return f"wrong read @{pos}/{n_reads + 1}: {what}"
                    if g[0] == a_[0] == "op":
                        return "wrong tool"
                    return f"{g[0]} instead of {a_[0]} @{pos}/{n_reads + 1}"
            if len(acts_b) < len(gold):
                return "stopped early"
            if len(acts_b) > len(gold):
                return "ran on"
            for (ti, ia, ib, xa, xb), (_, _, ra) in zip(ops_b, [p_ for p_ in q["prog"] if p_[0] == "op"]):
                ra = list(ra[:2])
                if [ia, ib] != ra:
                    return "operands swapped" if sorted([ia, ib]) == sorted(ra) else "wrong operands"
                if xa is None or xb is None:
                    return "operand not decoded"
                if (xa, xb) != (q["vals"][ia], q["vals"][ib]):        # which operand, of what kind, how far off
                    bad_i, bad_x = (ia, xa) if xa != q["vals"][ia] else (ib, xb)
                    gold_v = q["vals"][bad_i]
                    src_ = "computed result" if q["prog"][bad_i][0] == "op" else "stored number"
                    how = ("the other operand's value" if bad_x in (q["vals"][ia], q["vals"][ib])
                           else "near (+-2)" if isinstance(gold_v, int) and abs(bad_x - gold_v) <= 2 else "far")
                    nid_ = acts_b[bad_i][1] if bad_i < len(acts_b) and acts_b[bad_i][0] == "read" else None
                    held_ = value_of(store.prov.get(store.nodes[nid_]["vsrc"])) if nid_ is not None else None
                    op_examples.append((gold_v, bad_x, held_, store.nodes[nid_]["kind"] if nid_ is not None else "-"))
                    return f"operand misread: {src_}, {how}"
            if q["ops"] == 0 and isinstance(q.get("ans"), str):
                return ("mis-said: a NAME value (said the start right)" if (said or "").strip()[:2] == q["ans"][:2]
                        else "mis-said: a NAME value")
            if q["ops"] == 0 and isinstance(q.get("ans"), int):   # WHAT was said instead (value readout diagnosis)
                m_ = re.search(r"-?\d+", said or "")
                sv = int(m_.group()) if m_ else None
                fct = fact_of(gold[-1][1])[0] if gold[-1][0] == "read" else None
                if sv is None:
                    return "mis-said: no number"
                if fct:
                    nm_, k_ = fct
                    if sv in world.hist.get((nm_, k_), []):
                        return "mis-said: another value of the same fact (older / newer)"
                    if sv in {v_ for kk, v_ in world.ents.get(nm_, {}).items() if kk != k_ and isinstance(v_, int)}:
                        return "mis-said: same entity, other attribute's value"
                    if sv in {e_.get(k_) for n2, e_ in world.ents.items() if n2 != nm_}:
                        return "mis-said: same attribute, other entity's value"
                if abs(sv - q["ans"]) <= 2:
                    return "mis-said: near (+-2)"
                return "mis-said: other number"
            return "program right, answer mis-said"

        def score(name, specs, mix, ops=(1, 2), mode="any"):
            em_, dg_, qs_, wh_ = [], [], [], []
            op_examples.clear()
            for _ in range(0, A.eval_q, A.questions):
                qs = ask(er, [(A.questions * n // max(1, sum(n_ for n_, _ in specs)), p) for n, p in specs], mix,
                         ops, mode)
                if not qs:
                    continue
                qs_ += qs
                _, em, _, _ = run(qs, False)
                em_ += em.tolist(); dg_ += run.diag
                wh_ += ["right" if float(e) else why_wrong(q, a_, o_, sd_)
                        for q, e, a_, o_, sd_ in zip(qs, em, run.acts, run.oprec, run.said)]
                evict()
            n_ = max(1, len(em_))
            ok_given = [e for e, d_ in zip(em_, dg_) if d_[1] == 1.0]
            by = {}
            for q, e in zip(qs_, em_):
                by.setdefault("before" if q["before"] else "multi" if q.get("multi") else q["ops"], []).append(e)
            out[name] = {"exact": sum(em_) / n_, "n": len(em_),
                         "by": {str(k): round(sum(v) / len(v), 3) for k, v in sorted(by.items(), key=str)},
                         "gold_read": sum(d_[0] for d_ in dg_) / n_, "program_ok": sum(d_[1] for d_ in dg_) / n_,
                         "exact_given_program": sum(ok_given) / max(1, len(ok_given))}
            print(f"EVAL {step} {name}: exact {out[name]['exact']:.3f} | by {out[name]['by']} | gold read "
                  f"{out[name]['gold_read']:.3f} | program {out[name]['program_ok']:.3f} | exact|program "
                  f"{out[name]['exact_given_program']:.3f} | n {len(em_)}", flush=True)
            fails = {}
            for q, w_ in zip(qs_, wh_):
                if w_ != "right":
                    k_ = "before" if q["before"] else q["ops"]
                    fails.setdefault(k_, Counter_())[w_] += 1
            if op_examples:
                n_h = sum(1 for g_, d_, h_, _ in op_examples if h_ == d_ and h_ != g_)
                print(f"EVAL {step} {name} operand misreads (true, decoded, value the node holds, node kind): "
                      f"{op_examples[:8]} | node holds the decoded (not the asked) value: {n_h}/{len(op_examples)}", flush=True)
            if fails:
                print(f"EVAL {step} {name} why wrong (by ops): " + " | ".join(
                    f"{k_}: {dict(v_.most_common())}" for k_, v_ in sorted(fails.items(), key=lambda x: str(x[0]))),
                    flush=True)
            logf.write(json.dumps({"type": "eval", "step": step, "set": name, **out[name]}) + "\n")
            logf.flush()

        def one_step(name, facts, mix):
            """the one-exposure variant: ONE gradient step on questions about the new items, then ask again
            (weights, optimizer and memory restored afterwards)"""
            sd = [{k: v.detach().cpu().clone() for k, v in m_.state_dict().items()} for m_ in (graph, ro, loop, tools)]
            osd = io.BytesIO()
            torch.save(opt.state_dict(), osd)                # kept on the CPU, not as a second copy in VRAM
            lrs = [gp["lr"] for gp in opt.param_groups]
            for gp in opt.param_groups:
                gp["lr"] = A.lr * gp.get("lr_mult", 1.0)
            sn1 = snapshot()
            with torch.enable_grad():
                qs = ask(er, [(A.questions, facts)], mix)
                if qs:
                    loss = run(qs, True)[0]
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    heads_changed()
                del qs
            restore(sn1)                                     # the training questions are not kept in memory
            score(name, [(1, facts)], mix)
            for m_, s_ in zip((graph, ro, loop, tools), sd):
                m_.load_state_dict(s_)
            osd.seek(0)
            opt.load_state_dict(torch.load(osd, map_location=dev, weights_only=False))
            heads_changed()
            for gp, lr_ in zip(opt.param_groups, lrs):
                gp["lr"] = lr_

        def coverage(facts):
            H, _ = holders()
            return sum(1 for (nm, k) in facts if (nm, k, world.ents[nm][k]) in H) / max(1, len(facts))

        old = {(nm, k) for (nm, k, v), ns in holders()[0].items() if world.ents.get(nm, {}).get(k) == v
               and min(store.nodes[i]["time"] for i in ns) < store.time - 200}
        if old:
            score("old_facts", [(1, old)], "lookup:0.4,chain:0.2,arith:0.4")
        score("versions_before", [(1, None)], "before:1.0")
        score("held_compositions", [(1, None)], "arith:1.0", (2, 2), "held")
        score("longer_programs", [(1, None)], "arith:1.0", lo_ops)
        # ---- the ONE-EXPOSURE test: new entities (incl. never-seen attribute names) and updates, written once (the
        #      model's own integrate, no gradient), asked after k frames of state noise
        wstats.clear()
        for k_dist, held in ((0, False), (10, False), (100, False), (0, True)):
            before_hist = {f: list(v) for f, v in world.hist.items()}
            write([world.component(er, new_ent=True, held=held, p_noise=0.3) for _ in range(6)]
                  + [world.component(er, only_update=True) for _ in range(6)], False)
            for _ in range(k_dist // 2):
                write([state_frame(er) for _ in range(2)], False, "state")
                evict()
            changed = {f for f, v in world.hist.items() if v != before_hist.get(f)}
            tag = f"one_exposure_{'heldkeys_' if held else ''}after{k_dist}"
            print(f"EVAL {step} {tag}: stored {coverage(changed):.3f} of {len(changed)} new / updated facts", flush=True)
            score(tag, [(1, changed)], "lookup:0.6,arith:0.25,before:0.15")
            if k_dist == 0 and not held:
                one_step("one_step_after0", changed, "lookup:0.6,arith:0.25,before:0.15")
        # dependents: a new fact stated as a copy of another; then the SOURCE changes once -> the dependent follows
        deps_ = [x for x in (world.dependent(er) for _ in range(8)) if x]
        if deps_:
            write([x[0] for x in deps_], False)
            fd = {f for _, f in deps_}
            srcs = sorted({world.deps[f] for _, f in deps_ if f in world.deps})
            # diagnosis (bookkeeping): stored? edge -> the source's node? the update a replace? propagated?
            H0, _ = holders()
            D0, rows0, row_of0 = dep_matrix()
            n_st = n_edge = 0
            for nm_, k_ in fd:
                dn = gold_node(H0, nm_, k_, world.ents[nm_][k_])
                if dn is None or dn not in row_of0:
                    continue
                n_st += 1
                s_nm, s_k = world.deps[(nm_, k_)]
                src_n = gold_node(H0, s_nm, s_k, world.ents[s_nm][s_k])
                j_ = row_of0[dn]
                n_edge += int(src_n is not None and src_n in row_of0 and float(D0[0][j_]) > 0.5
                              and int(D0[1][j_]) == row_of0[src_n])
                tn_ = rows0[int(D0[1][j_])]
                tm_ = store.prov.get(store.nodes[tn_]["lsrc"]) if store.nodes[tn_]["lsrc"] else None
                dtxt = store.prov.get(store.nodes[dn].get("rsrc") or store.nodes[dn]["vsrc"], ("?",))
                print(f"  dependent {nm_}.{k_} (value {dtxt[3] if len(dtxt) > 3 else dtxt}, dep flag "
                      f"{dtxt[4] if len(dtxt) > 4 else '?'}) -> source {s_nm}.{s_k} | has {float(D0[0][j_]):.2f} | "
                      f"top: {tm_[:3] if tm_ else store.nodes[tn_]['kind']} ({store.nodes[tn_]['kind']}) | source node "
                      f"{'in memory' if src_n in row_of0 else 'MISSING'}", flush=True)
            wstats.pop("propagated", None)
            wstats.pop("decided replace", None)
            write([world.restate_new(er, *s_) for s_ in srcs], False)
            print(f"EVAL {step} dependents diagnosis: stored {n_st}/{len(fd)} | edge -> the source's node (has > .5, "
                  f"top = source) {n_edge}/{n_st} | source updates decided replace "
                  f"{wstats.get('decided replace', [0])[0]}/{len(srcs)} | propagated "
                  f"{wstats.get('propagated', [0])[0]}", flush=True)
            print(f"EVAL {step} one_exposure_dependents: follow {coverage(fd):.3f} of {len(fd)} dependents", flush=True)
            score("one_exposure_dependents", [(1, fd)], "lookup:1.0")
        # a RULE shown once (wording never trained, on a key without a rule): later values of that key are increments
        H, _ = holders()
        free = {}
        for (nm, k, v) in H:
            if k in world.num_keys and world.policy.get(k) != "add" and world.ents.get(nm, {}).get(k) == v:
                free.setdefault(k, []).append(nm)
        if free:
            k = max(sorted(free), key=lambda k_: len(free[k_]))
            write([world.rule(er, k, "add", held=True)], False)
            for _ in range(5):                               # some distance between the rule and its use
                write([state_frame(er) for _ in range(2)], False, "state")
            fr = sorted(set(free[k]))[:8]
            write([world.update_one(er, nm, k) for nm in fr], False)
            fk = {(nm, k) for nm in fr}
            print(f"EVAL {step} one_exposure_rule: totals held {coverage(fk):.3f} of {len(fk)} (rule: add)", flush=True)
            score("one_exposure_rule", [(1, fk)], "lookup:1.0")
        # coexisting values: a second value of a multi key (must be attached, not replaced)
        co = [x for x in (world.coexist(er) for _ in range(8)) if x]
        if co:
            write([x[0] for x in co], False)
            fc = {f for _, f in co}
            H, _ = holders()
            both = sum(all(f + (v,) in H for v in world.multi[f]) for f in fc) / len(fc)
            print(f"EVAL {step} one_exposure_coexist: both values held {both:.3f} of {len(fc)}", flush=True)
            score("one_exposure_coexist", [(1, fc)], "multi:1.0")
        ws = {k: {"n": v[0], "store": round(v[1] / max(1, v[0]), 2), "new": round(v[2] / max(1, v[0]), 2),
                  "replace": round(v[3] / max(1, v[0]), 2)} for k, v in wstats.items()}
        print(f"EVAL {step} integrate by kind (mean store gate / NEW / replace weight): {ws}", flush=True)
        run.look_tr.clear()
        cq = corrections(er, 24)
        if cq:
            _, em, _, _ = run(cq, False, "note")
            print(f"EVAL {step} corrections by the loop: edit said right {float(em.mean()):.3f} | program right "
                  f"{sum(d_[1] for d_ in run.diag) / len(cq):.3f} | n {len(cq)}", flush=True)
            import collections
            outs = collections.Counter(o[0] for _, o in getattr(run, "last_plans", []))
            ex = [[(kd if kd != "op" else TOOLS[x][0]) + ("*" if kd == "read" and x == q["prog"][1][1] else "")
                   for kd, x in a_] for q, a_ in zip(cq[:3], run.acts[:3])]
            print(f"EVAL {step} correction writes: {dict(outs)} | first notes did (* = the target): {ex}", flush=True)
            wr_ = [[(("proposal " if r_.get("tentative") else "look " if r_.get("look") else "")
                      + ("TGT" if r_["tgt"] == q["prog"][1][1] else "NEW" if r_["tgt"] is None else "other"), round(r_["g"], 2),
                     round(r_["u"], 2), round(r_["p"], 2), round(r_.get("g_new", -1), 2), r_.get("same"))
                    for r_ in rr] for q, rr in zip(cq[:4], getattr(run, "last_recs", [])[:4])]
            print(f"EVAL {step} correction write records (target, gate, replace, confidence, gate_new, same): {wr_}", flush=True)
            if A.look:
                lt_ = {k_: v_ for k_, v_ in run.look_tr.items() if k_.startswith("note")}
                print(f"EVAL {step} correction look (proposed -> after looking): {lt_}", flush=True)
            score("corrected_facts", [(1, {q["edit"] for q in cq})], "lookup:0.7,before:0.3")
        H, _ = holders()
        ids_ = {}
        for (nm, k, v), ns in H.items():
            ids_.setdefault((nm, k), set()).update(ns)
        dup = sum(1 for ns in ids_.values() if len(ns) > 1)
        print(f"EVAL {step} duplicates: {dup} of {len(ids_)} facts held by >1 current node | nodes per fact "
              f"{sum(len(ns) for ns in ids_.values()) / max(1, len(ids_)):.2f}", flush=True)
        out["duplicates"] = {"exact": dup / max(1, len(ids_)), "n": len(ids_)}
        rt = roundtrip(er)
        print(f"EVAL {step} roundtrip (encode -> say): " + " | ".join(f"{k} {v:.3f}" for k, v in rt.items()), flush=True)
        frt = frame_roundtrip(er)
        if frt is not None:
            print(f"EVAL {step} frame-pair roundtrip (a value encoded inside a record, said from its node, no memory "
                  f"ops): {frt:.3f}", flush=True)
        ir_, is_, n_ = identity_lookup(er)
        out["identity_lookup"] = {"exact": is_, "raw": ir_, "n": n_}
        print(f"EVAL {step} identity_lookup: update -> its identity node is the best match: raw label {ir_:.3f} | "
              f"Wself query {is_:.3f} | n {n_}", flush=True)
        logf.write(json.dumps({"type": "eval", "step": step, "set": "identity_lookup", **out["identity_lookup"]}) + chr(10))
        restore(sn)
        w0, s0 = world, store                              # the very first frame, in fresh worlds (empty memory)
        acc, rd = first_frame_probe(10, quiet=True)
        world, store = w0, s0
        cache["src"], cache["at"] = {}, -1
        out["first_frame"] = {"exact": acc, "program_ok": rd}
        print(f"EVAL {step} first_frame: exact {acc:.3f} | read the right node {rd:.3f}", flush=True)
        logf.write(json.dumps({"type": "eval", "step": step, "set": "first_frame", **out["first_frame"]}) + chr(10))
        logf.flush()
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        mem.train()
        return out

    @torch.no_grad()
    def first_frame_probe(n, quiet=False):
        """the very first frame: empty memory, one frame written by the model's own integrate (no gradient), then
        every fact in it asked immediately"""
        nonlocal world, store
        mem.eval()
        n_q = n_ok = n_prog = n_held = n_facts = 0
        for t in range(n):
            prng = random.Random(90_000 + t)
            world, store = World(prng, max_ents=A.max_ents), Store()
            cache["src"], cache["at"] = {}, -1
            if A.first_warm:                                 # control: memory already holds other content
                write([world.component(prng) for _ in range(A.first_warm)], False)
                world.ents = {}
            frame = world.component(prng, new_ent=True, p_noise=0.0)
            write([frame], False)
            H, _ = holders()
            facts = [(nm, k) for nm, e in world.ents.items() for k in e]
            held = [f for f in facts if f + (world.ents[f[0]][f[1]],) in H]
            n_facts += len(facts); n_held += len(held)
            qs = ask(prng, [(1, {f}) for f in held], "lookup:1.0")
            if not qs:
                continue
            _, em, _, _ = run(qs, False)
            if not quiet:
                print(f"FIRST {t}: {json.dumps(frame)} -> stored {len(held)}/{len(facts)} facts")
            for q, e, d_, said in zip(qs, em.tolist(), run.diag, run.said):
                rd = [store.prov.get(store.nodes[x]["vsrc"], ("?",))[0] if kd == "read" else TOOLS[x][0]
                      for kd, x in run.acts[len(qs) - len(qs) + qs.index(q)]]
                if not quiet:
                    print(f"    {q['text']}  gold {q['ans']!r} said {said.strip()!r}  {'OK' if e else 'WRONG'}"
                          f"{'' if d_[1] else ' (read the wrong node)'} | loop read: {rd}")
                n_q += 1; n_ok += int(e); n_prog += int(d_[1])
        print(f"FIRST FRAME: stored {n_held}/{n_facts} facts | answered {n_ok}/{n_q} = {n_ok / max(1, n_q):.3f} | "
              f"read the right node {n_prog / max(1, n_q):.3f}", flush=True)
        return n_ok / max(1, n_q), n_prog / max(1, n_q)

    @torch.no_grad()
    def visualize(path):
        """graph sanity: the live memory after the model wrote frames itself (its own decisions, no gradient) ->
        an interactive HTML graph of a few entities + sanity statistics over the whole memory"""
        mem.eval()
        vr = random.Random(A.seed + 7)
        for i in range(A.viz_sim):                           # the model integrates frames like in training
            rule = world.rule(vr) if vr.random() < A.p_rule else None
            write([state_frame(vr) for _ in range(A.states)], False, "state")
            write(([rule] if rule else []) + [world.component(vr) for _ in range(A.pending)], False)
            evict()
        M = materialize()
        rows, row_of = M["rows"], M["row_of"]
        D = graph.dep_ref(M["K"], M["R"], M["kmean"], M["ver"], M["HM"])
        Rl = D                                                   # relation edges ARE the r edges now
        H, Hv = holders()

        def ident(i):
            nd = store.nodes[i]
            lm = store.prov.get(nd["lsrc"]) if nd["lsrc"] else None
            return lm if lm else ("?",)

        def text(i):
            nd = store.nodes[i]
            lm, vm = ident(i), store.prov.get(nd["vsrc"], ("?",))
            v = value_of(vm)
            if lm[0] == "fact":
                cur = world.ents.get(lm[1], {}).get(lm[2]) if lm[2] not in world.multi_keys else \
                    (v if v in world.multi.get((lm[1], lm[2]), []) else None)
                tag = "" if nd["kind"] == "version" or cur == v else f"  STALE (truth {cur})"
                return f"{'[v] ' if nd['kind'] == 'version' else ''}{lm[1]}.{words(lm[2])} = {v}{tag}", lm[1], \
                    "version" if nd["kind"] == "version" else ("stale" if tag else "fact")
            if lm[0] == "rule":
                return f"RULE {words(lm[1])}: {lm[2]}", None, "rule"
            if lm[0] == "name":
                return f"name {lm[1]}", lm[1], "name"
            if lm[0] == "delta":
                return f"delta {lm[1]}.{words(lm[2])} +{lm[3]}", lm[1], "delta"
            if vm[0] == "val":
                return f"computed = {vm[1]}", None, "computed"
            return f"{lm[0] if lm[0] != '?' else nd['kind']}", None, nd["kind"] if lm[0] == "?" else lm[0]

        info = {i: text(i) for i in rows}
        # ---- sanity over the WHOLE memory
        kinds = {}
        for i in rows:
            kinds[info[i][2]] = kinds.get(info[i][2], 0) + 1
        dup = sum(1 for (nm, k, v), ns in H.items() if len(ns) > 1)
        facts = [(nm, k) for nm, e in world.ents.items() for k in e]
        held = sum(1 for nm, k in facts if (nm, k, world.ents[nm][k]) in H)
        top_dep = D.max(-1).values
        stats = {"live nodes": len(rows), "by kind": kinds, "world facts": len(facts),
                 "facts held with the TRUE value": f"{held}/{len(facts)} = {held / max(1, len(facts)):.3f}",
                 "facts held by >1 current node (duplicates)": dup,
                 "current nodes holding a STALE value": kinds.get("stale", 0),
                 "rules announced / rule nodes alive": f"{len(world.policy)} / {kinds.get('rule', 0)}",
                 "dependence edges above .5 (would propagate)": int((top_dep > 0.5).sum()),
                 "mean strongest dependence edge": round(float(top_dep.mean()), 3),
}
        print("GRAPH SANITY:", json.dumps(stats, indent=1), flush=True)
        # ---- CONNECTIVITY (user, 2026-10-25): is the graph wired the way the world is?
        n_ = len(rows)
        ent_of = [info[i][1] for i in rows]
        eye = torch.eye(n_, device=Rl.device, dtype=torch.bool)
        Rm = Rl.masked_fill(eye, -1.0)
        top1 = Rm.argmax(-1).tolist()
        has_e = [j for j in range(n_) if ent_of[j]]
        same_top = sum(1 for j in has_e if ent_of[top1[j]] == ent_of[j]) / max(1, len(has_e))
        rec_of = [store.nodes[i].get("rec") for i in rows]
        has_r = [j for j in range(n_) if rec_of[j] is not None and sum(1 for x in rec_of if x == rec_of[j]) > 1]
        same_rec = sum(1 for j in has_r if rec_of[top1[j]] == rec_of[j]) / max(1, len(has_r))
        from collections import Counter as _C
        ec = _C(e for e in ent_of if e)
        chance = sum(c * (c - 1) for c in ec.values()) / max(1, len(has_e) * (n_ - 1))
        indeg = _C(top1)
        top2 = Rm.topk(min(2, n_ - 1), -1).indices.tolist()
        parent = list(range(n_))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a_, b_):
            parent[find(a_)] = find(b_)
        for j in range(n_):
            for k_ in top2[j]:
                union(j, k_)
        n_link = n_ver = 0
        for j, i in enumerate(rows):
            nd = store.nodes[i]
            if nd.get("link") in row_of:
                union(j, row_of[nd["link"]]); n_link += 1
            if nd.get("ver_of") in row_of:
                union(j, row_of[nd["ver_of"]]); n_ver += 1
        comps = _C(find(j) for j in range(n_))
        dep_top = Dm = D.masked_fill(eye, -1.0)
        dtop = Dm.argmax(-1).tolist()
        deps_ = [j for j, i in enumerate(rows) if ident(i)[0] == "fact" and len(ident(i)) > 4 and ident(i)[4]]
        dep_ok = sum(1 for j in deps_ if value_of(store.prov.get(store.nodes[rows[dtop[j]]]["vsrc"])) ==
                     value_of(store.prov.get(store.nodes[rows[j]]["vsrc"]))) / max(1, len(deps_))
        conn = {"nodes": n_,
                "relation top-1 partner is in the SAME record": f"{same_rec:.3f} of {len(has_r)} nodes with partners",
                "relation top-1 partner is the SAME entity": f"{same_top:.3f} (chance {chance:.3f})",
                "hubs: max in-degree of top-1 relation edges": max(indeg.values()) if indeg else 0,
                "nodes that are someone's top-1 partner": f"{len(indeg)}/{n_}",
                "components (relation top-2 + attach + version links)": len(comps),
                "largest component share": round(max(comps.values()) / max(1, n_), 3),
                "singleton components": sum(1 for c in comps.values() if c == 1),
                "attach links / version links": f"{n_link} / {n_ver}",
                "dependents whose top dependence partner holds the same value": f"{dep_ok:.3f} of {len(deps_)}"}
        print("GRAPH CONNECTIVITY:", json.dumps(conn, indent=1), flush=True)
        # ---- the hubs: who they are and why every node points at them
        for j, deg in indeg.most_common(3):
            i = rows[j]
            print(f"HUB node {i}: in-degree {deg} | {info[i][0]} | kind {store.nodes[i]['kind']}", flush=True)
        # ---- the drawn subgraph: the entities with the most nodes, their versions, rules, and 1-hop edge partners
        per = {}
        for i in rows:
            if info[i][1]:
                per[info[i][1]] = per.get(info[i][1], 0) + 1
        ents = set(sorted(per, key=lambda e: -per[e])[:A.viz_ents])
        sel = {i for i in rows if info[i][1] in ents or info[i][2] == "rule"}
        edges = []
        for i in list(sel):
            nd = store.nodes[i]
            if nd["ver_of"] is not None and nd["ver_of"] in row_of:
                edges.append((i, nd["ver_of"], "version", 1.0))
            if nd["link"] is not None and nd["link"] in row_of:
                edges.append((i, nd["link"], "attach", 1.0))
            r_ = row_of[i]
            j = int(D[r_].argmax())
            if float(D[r_, j]) > A.viz_min and rows[j] != i:
                edges.append((i, rows[j], "dependence", float(D[r_, j])))
            j = int(Rl[r_].argmax())
            if float(Rl[r_, j]) > A.viz_min and rows[j] != i:
                edges.append((i, rows[j], "relation", float(Rl[r_, j])))
        for a_, b_, _, _ in edges:
            sel.update((a_, b_))
        cur_of = {}                                          # truth bookkeeping (not model edges): duplicates
        for i in sel:
            lm = ident(i)
            if lm[0] == "fact" and store.nodes[i]["kind"] != "version":
                cur_of.setdefault(lm[1:3], []).append(i)
        for ns in cur_of.values():
            if len(ns) > 1:
                newest = max(ns, key=lambda i: store.nodes[i]["time"])
                edges += [(i, newest, "truth: duplicate of", 1.0) for i in ns if i != newest]
        color = {"fact": "#4e79a7", "stale": "#e15759", "version": "#bab0ac", "rule": "#f28e2b", "name": "#76b7b2",
                 "delta": "#edc948", "computed": "#59a14f", "state": "#d3d3d3", "question": "#b07aa1"}
        nodes = []
        for i in sorted(sel):
            nd = store.nodes[i]
            t_, _, c_ = info[i]
            nodes.append({"id": i, "label": t_[:48], "group": c_, "color": color.get(c_, "#9c755f"),
                          "value": 1 + nd["usage"],
                          "title": f"{t_}\nkind {nd['kind']} | time {nd['time']:.1f} | usage {nd['usage']} | "
                                   f"act {nd['act']:.2f} | gate {nd.get('g', 1.0):.2f}"})
        for e_ in sorted(ents):                              # truth bookkeeping: entity hubs
            nodes.append({"id": "E:" + e_, "label": e_, "shape": "box", "color": "#ffffff", "group": "entity",
                          "title": f"entity {e_} (truth bookkeeping, not a memory node)"})
        edges += [(i, "E:" + info[i][1], "truth: entity", 0.0) for i in sorted(sel) if info[i][1] in ents]
        ecol = {"version": "#999999", "attach": "#59a14f", "dependence": "#f28e2b", "relation": "#a0cbe8",
                "truth: duplicate of": "#e15759", "truth: entity": "#e6e6e6"}
        eds = [{"from": a_, "to": b_, "color": ecol[k_], "dashes": k_ in ("version", "truth: duplicate of"),
                "arrows": "" if k_ == "truth: entity" else "to", "width": 1 + 3 * w_, "title": f"{k_} {w_:.2f}"}
               for a_, b_, k_, w_ in edges]
        html = """<!doctype html><html><head><meta charset="utf-8"><title>GNM graph</title>
<script src="https://unpkg.com/vis-network@9.1.9/standalone/umd/vis-network.min.js"></script>
<style>body{margin:0;font:13px system-ui,sans-serif;display:flex;height:100vh;background:#fff;color:#222}
#g{flex:1}#side{width:360px;overflow:auto;padding:12px;border-left:1px solid #ddd}pre{white-space:pre-wrap;font-size:12px}
.k{display:inline-block;width:10px;height:10px;margin-right:6px;border-radius:2px}</style></head><body>
<div id="g"></div><div id="side"><b>GNM memory graph</b><p>__SRC__</p><div id="legend"></div>
<p><b>Sanity (whole memory)</b></p><pre id="stats"></pre><p>Hover a node / edge for details. Node size = usage.</p></div>
<script>const N=__NODES__,E=__EDGES__,S=__STATS__,C=__COLORS__,EC=__ECOLORS__;
document.getElementById('stats').textContent=JSON.stringify(S,null,1);
let lg='';for(const k in C)lg+='<div><span class="k" style="background:'+C[k]+'"></span>'+k+'</div>';
for(const k in EC)lg+='<div><span class="k" style="background:'+EC[k]+'"></span>edge: '+k+'</div>';
document.getElementById('legend').innerHTML=lg;
new vis.Network(document.getElementById('g'),{nodes:new vis.DataSet(N),edges:new vis.DataSet(E)},
{nodes:{shape:'dot',scaling:{min:6,max:26},font:{size:11}},edges:{smooth:false},
physics:{barnesHut:{gravitationalConstant:-6000,springLength:110},stabilization:{iterations:300}}});
</script></body></html>"""
        src = f"{A.load or 'untrained'} | world after {A.warm_frames} warm + {A.viz_sim} simulated steps | {len(ents)} entities drawn"
        for k_, v_ in (("__NODES__", json.dumps(nodes)), ("__EDGES__", json.dumps(eds)), ("__STATS__", json.dumps(stats)),
                       ("__COLORS__", json.dumps(color)), ("__ECOLORS__", json.dumps(ecol)), ("__SRC__", src)):
            html = html.replace(k_, v_)
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        n_model = sum(1 for e_ in edges if not e_[2].startswith("truth"))
        print(f"GRAPH: {len(nodes)} nodes, {n_model} model edges + {len(eds) - n_model} truth links -> {path}", flush=True)

    ref_stats = []

    def ref_loss(irng, n_neg=48):
        """reference edges by demonstration (the generator knows what a "same as" value mentions): dependents ->
        their source's node, other fact nodes -> none. Own optimizer step (like the relation heads)."""
        with torch.no_grad():
            M = materialize()
            H, _ = holders()
        rows_ = M["rows"]
        pos, neg = [], []
        for j, i in enumerate(rows_):
            nd = store.nodes[i]
            if nd["kind"] != "item":
                continue
            lm = store.prov.get(nd["lsrc"]) if nd["lsrc"] else None
            if not lm or lm[0] != "fact":
                continue
            src = world.deps.get((lm[1], lm[2])) if len(lm) > 4 and lm[4] else None
            if src is not None and src[0] in world.ents and src[1] in world.ents[src[0]]:
                g_ = gold_node(H, src[0], src[1], world.ents[src[0]][src[1]])
                if g_ is not None and g_ in M["row_of"]:          # any current node holding the source's value
                    alt = [M["row_of"][x] for x in H.get((src[0], src[1], world.ents[src[0]][src[1]]), [])
                           if x in M["row_of"] and store.nodes[x]["kind"] != "version"]
                    pos.append((j, M["row_of"][g_], src, lm[1], alt or [M["row_of"][g_]]))
            elif not (len(lm) > 4 and lm[4]):
                neg.append(j)
        if not pos:
            return torch.zeros((), device=dev)
        neg = irng.sample(neg, min(n_neg, len(neg)))
        sel = [x[0] for x in pos] + neg
        ix = torch.tensor(sel, device=dev)
        by_f = {}                                        # the selected nodes' reference queries WITH gradient
        for k_, j in enumerate(sel):                     # (pooling + residual), from their frames' stored states
            vs_ = store.nodes[rows_[j]].get("rsrc", store.nodes[rows_[j]]["vsrc"])
            if vs_ is not None and vs_[0] == "tok" and vs_[1] in store.frames:
                by_f.setdefault(vs_[1], []).append((k_, vs_[2]))
        rq_ = [M["RQ"][j].detach() for j in sel]
        for f_, lst in by_f.items():
            fr = store.frames[f_]
            with torch.no_grad():
                E_ = model.emb(torch.tensor(fr["ids"], device=dev)).float()
            q_ = graph.ref_query(fr["X"].float(), E_, [fr["enc"]["vpos"][jj] for _, jj in lst],
                                 pre=ref_alone([(f_, jj) for _, jj in lst]))
            for (k_, _), qq in zip(lst, q_):
                rq_[k_] = qq
        rq_ = torch.stack(rq_)                           # the selected nodes' r with a gradient path (content part)
        R_ = M["R"].detach().index_put((ix,), M["R"][ix].detach() + graph.Wrr(rq_) - graph.Wrr(rq_).detach())
        L = graph.ref_logits(M["K"].detach(), R_, M["kmean"].detach(), M["ver"], M["HM"].detach())
        n_ = L.shape[0]
        tg = torch.tensor([x[1] for x in pos] + [n_] * len(neg), device=dev)
        Ls = L[ix]
        isref = torch.tensor([1.0] * len(pos) + [0.0] * len(neg), device=dev)
        bce = F.binary_cross_entropy_with_logits(-Ls[:, -1], isref, reduction="none")
        bce = bce[:len(pos)].mean() + (bce[len(pos):].mean() if neg else 0.0)   # class-balanced: dependents fell to
        pm = torch.zeros(len(pos), Ls.shape[1] - 1, dtype=torch.bool, device=dev)  # ~12 / step vs 48 others -> none
        for r_, x in enumerate(pos):                     # multi-positive: duplicates of the source holding its value
            pm[r_, x[4]] = True                          # (an "older item node" miss was often such a copy)
        Ln_ = Ls[:len(pos), :-1]
        loss_ = bce + (torch.logsumexp(Ln_, -1) - torch.logsumexp(Ln_.masked_fill(~pm, -1e4), -1)).mean()
        pr = torch.where(Ls[:, -1] > 0, torch.full_like(tg, n_), Ls[:, :-1].argmax(-1))
        miss = Counter_()
        hit_ = []
        for (j, g, src, own, alt), p_ in zip(pos, pr[:len(pos)].tolist()):
            hit_.append(p_ in alt)
            if p_ in alt:
                continue
            if p_ == n_:
                miss["none"] += 1
                continue
            nd_ = store.nodes[rows_[p_]]
            m_ = store.prov.get(nd_["lsrc"]) if nd_["lsrc"] else None
            if not m_ or m_[0] != "fact":
                miss[f"a {nd_['kind']} / {m_[0] if m_ else '-'} node"] += 1
            elif (m_[1], m_[2]) == tuple(src):
                miss["the source's version node" if nd_["kind"] == "version" else "the source's older item node"] += 1
            elif m_[1] == src[0]:
                miss["source entity, other attribute"] += 1
            elif m_[2] == src[1]:
                miss["source attribute, other entity"] += 1
            elif m_[1] == own:
                miss["the dependent's own entity"] += 1
            else:
                miss["other fact"] += 1
        ref_stats.append((sum(hit_) / len(hit_),
                          float((pr[len(pos):] == n_).float().mean()) if neg else 1.0, len(pos), miss))
        return loss_

    def invariance_loss(irng, n=16):
        """labels = what a pair is about, not where it was seen: each sampled fact is encoded in two random
        contexts (other group key, other / shuffled siblings, another value) -> InfoNCE between the two labels"""
        by_key = {}
        for nm, e in world.ents.items():
            for k in e:
                if k in world.num_keys:
                    by_key.setdefault(k, []).append(nm)
        keys_ok = [k for k, ns in sorted(by_key.items()) if len(ns) >= 4]
        if not keys_ok:
            return torch.zeros((), device=dev)
        pick = []                                       # HARD negatives: 4 attributes x 4 entities that share them, so
        for k in irng.sample(keys_ok, min(4, len(keys_ok))):        # a label must encode the ENTITY, not only the
            pick += [(nm, k) for nm in irng.sample(sorted(by_key[k]), 4)]   # attribute (random batches let it drop it)
        objs, keys = [], []
        for view in range(2):
            for nm, k in pick:
                sib = [x for x in world.schema[nm] if x != k and x in world.num_keys]
                rec = {"name": nm}
                for x in irng.sample(sib, min(len(sib), irng.randint(0, 3))):
                    rec[x] = irng.randint(1, 30)
                rec[k] = irng.randint(1, 30)
                items = list(rec.items())
                irng.shuffle(items)
                grp = irng.choice(world.groups + [_style([irng.choice(KEYS_TRAIN)], irng)])
                objs.append({grp: [dict(items)]})
                keys.append(k)
        encs = [doc_encoding(tok.ids, tok.decode, o, "state")[0] for o in objs]
        L = max(len(e["ids"]) for e in encs)
        X = walker_states(model, torch.tensor([e["ids"] + [0] * (L - len(e["ids"])) for e in encs], device=dev))
        Xs, ids, spans, ptr, vpos, off = [], [], [], [], [], 0
        for o, e, k, x_ in zip(objs, encs, keys, X):
            j = next(i for i, lf in enumerate(serialize(o, "state")[1]) if lf[0][-1] == k)
            Xs.append(x_[:len(e["ids"])]); ids += e["ids"]
            spans.append([off + q for q in e["spans"][j]]); ptr.append([off + q for q in e["ptr"][j]])
            vpos.append([off + q for q in e["vpos"][j]])
            off += len(e["ids"])
        with torch.no_grad():
            E = model.emb(torch.tensor(ids, device=dev)).float()
        upd, mem.update_stats = getattr(mem, "update_stats", False), False   # augmented frames: never into the stats
        kraw = graph.nodes(torch.cat(Xs).float(), E, spans, ptr, vpos)[0]
        mem.update_stats = upd
        z = F.normalize(kraw - graph.kmu, dim=-1)
        a_, b_ = z[:len(pick)], z[len(pick):]
        lg = a_ @ b_.t() / A.inv_temp
        tgt = torch.arange(len(pick), device=dev)
        inv_stats.append(float((lg.argmax(-1) == tgt).float().mean()))
        return 0.5 * (F.cross_entropy(lg, tgt) + F.cross_entropy(lg.t(), tgt))

    inv_stats = []
    rt_stats = []

    def roundtrip_vals(irng, n=48):
        """values of three kinds: numbers, entity-like names, never-seen strings"""
        nums = [irng.randint(0, 999) for _ in range(n)]
        names = [_name(irng, irng.randint(2, 3)) for _ in range(n)]
        odd = [irng.choice(KEYS_EVAL) + "-" + str(irng.randint(1, 99)) for _ in range(n)]
        return {"numbers": nums, "names": names, "unseen strings": odd}

    @torch.no_grad()
    def roundtrip(irng):
        """eval: a value encoded as a pair, then said by the model's own head: exact?"""
        res = {}
        for kind_, vals in roundtrip_vals(irng).items():
            enc_ = encode_values(vals, [float(store.time)] * len(vals), False)
            regs_ = [rg for _, rg in enc_]
            said = decode(torch.stack([r_[1] for r_ in regs_]), stackS([r_[2] for r_ in regs_]), text=True)
            res[kind_] = sum(str(v) == x.strip().split(" ")[0].rstrip(".,") or x.strip().startswith(str(v))
                             for v, x in zip(vals, said)) / len(vals)
        return res

    @torch.no_grad()
    def frame_roundtrip(irng, n=48):
        """eval: values encoded inside a multi-pair record (siblings, a group key) — the way facts are stored —
        then said straight from their node encoding (no memory operations): exact?"""
        facts = [(nm, k) for nm, e in world.ents.items() for k in e if k in world.num_keys]
        if len(facts) < 4:
            return None
        ok = tot = 0
        for nm, k in irng.sample(facts, min(n, len(facts))):
            sib = [x for x in world.schema[nm] if x != k and x in world.num_keys]
            rec = {"name": nm}
            for x in irng.sample(sib, min(len(sib), irng.randint(1, 3))):
                rec[x] = irng.randint(1, 30)
            v = irng.randint(1, 30)
            rec[k] = v
            items = list(rec.items())
            irng.shuffle(items)
            obj = {irng.choice(world.groups): [dict(items)]}
            e = doc_encoding(tok.ids, tok.decode, obj, "state")[0]
            X = walker_states(model, torch.tensor([e["ids"]], device=dev))
            j = next(i for i, lf in enumerate(serialize(obj, "state")[1]) if lf[0][-1] == k)
            with torch.no_grad():
                E = model.emb(torch.tensor(e["ids"], device=dev)).float()
            _, v_, _, s_, _ = graph.nodes(X[0, :len(e["ids"])].float(), E, [e["spans"][j]], [e["ptr"][j]], [e["vpos"][j]])
            said = decode(v_, s_[:, :A.mem_slots], text=True)[0]
            ok += said.strip().split(" ")[0].rstrip(".,") == str(v)
            tot += 1
        return ok / max(1, tot)

    def roundtrip_loss(wr, n=16):
        """encode -> say: values become pairs, the model's own head must say them exactly (loss, accuracy)"""
        if True:
            vals = sum(roundtrip_vals(wr, n).values(), [])
            encs = [doc_encoding(tok.ids, tok.decode, {_style([wr.choice(KEYS_TRAIN)], wr): v}, "state")[0]
                    for v in vals]
            L = max(len(e["ids"]) for e in encs)
            X = walker_states(model, torch.tensor([e["ids"] + [0] * (L - len(e["ids"])) for e in encs], device=dev))
            Xs, ids, spans, ptr, vpos, off = [], [], [], [], [], 0
            for e, x_ in zip(encs, X):
                Xs.append(x_[:len(e["ids"])]); ids += e["ids"]
                spans.append([off + q for q in e["spans"][0]]); ptr.append([off + q for q in e["ptr"][0]])
                vpos.append([off + q for q in e["vpos"][0]])
                off += len(e["ids"])
            with torch.no_grad():
                E = model.emb(torch.tensor(ids, device=dev)).float()
            _, v_, _, s_, _ = graph.nodes(torch.cat(Xs).float(), E, spans, ptr, vpos)
            ys = [ans_tok(v)[:A.max_ans] for v in vals]
            Lm = max(map(len, ys))
            inp = torch.tensor([ans_ids + y[:-1] + [0] * (Lm - len(y)) for y in ys], device=dev)
            tgt_ = torch.full((len(ys), len(ans_ids) + Lm - 1), -100, dtype=torch.long, device=dev)
            for i, y in enumerate(ys):
                tgt_[i, a0:a0 + len(y)] = torch.tensor(y, device=dev)
            inject.set(mem.Wo(v_), ro.Us(padS(s_[:, :A.mem_slots], Lm)), a0)
            lo, _ = model(inp, r_active=False)
            inject.clear()
            loss_ = F.cross_entropy(lo.float().reshape(-1, lo.shape[-1]), tgt_.reshape(-1), ignore_index=-100)
            return loss_, float(((lo.argmax(-1) == tgt_) | (tgt_ == -100) | (tgt_ == EOS_ID)).all(-1).float().mean())

    def warm_roundtrip(n_steps):
        """warm-up (user): encode -> decode of many values before the main run, so values are objects the tools
        can rely on. Trains the encoder, readout and say path on round trips (no memory, no loop)."""
        wr = random.Random(A.seed + 3)
        for st_ in range(1, n_steps + 1):
            loss_, ok_ = roundtrip_loss(wr)
            if A.warm_inv:                       # labels learn IDENTITY too (a fresh encoder made every key alike on
                loss_ = loss_ + A.warm_inv * invariance_loss(wr)   # the 1B: label invariance .17, facts replaced
            #                                      existing nodes instead of NEW)
            opt.zero_grad(set_to_none=True)
            loss_.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            for gp in opt.param_groups:
                gp["lr"] = A.lr * gp.get("lr_mult", 1.0)
            opt.step()
            if st_ % 50 == 0 or st_ == n_steps:
                inv_ = f" | label invariance {sum(inv_stats) / len(inv_stats):.2f}" if inv_stats else ""
                inv_stats.clear()
                print(f"warm roundtrip {st_}: loss {float(loss_):.3f} said exactly {ok_:.2f}{inv_}", flush=True)
        heads_changed()

    @torch.no_grad()
    def identity_lookup(irng, n=24):
        """eval: an update of a known fact -> is its identity node the best match (raw label / Wself query)?"""
        H, _ = holders()
        cur = [(nm, k) for (nm, k, v) in H if k in world.num_keys and world.ents.get(nm, {}).get(k) == v]
        hit_raw = hit_self = tot = 0
        for nm, k in irng.sample(sorted(set(cur)), min(n, len(set(cur)))):
            tgt = max(H[(nm, k, world.ents[nm][k])], key=lambda i: store.nodes[i]["time"])
            nl, _ = see([{irng.choice(world.groups): [{"name": nm, k: irng.randint(1, 30)}]}], "question", False)
            c = nl[0][-1]
            M = materialize()
            if tgt not in M["row_of"] or c not in M["row_of"]:
                store.nodes[c]["alive"] = False
                continue
            ci = M["row_of"][c]
            okm = (M["time"] < M["time"][ci]) & torch.tensor([store.nodes[i]["kind"] in ("item", "state")
                                                              for i in M["rows"]], device=dev)
            for q_, which in ((M["kraw"][ci], "raw"), (loop.Wself(M["kraw"][ci]), "self")):
                sc = (M["K"] @ F.normalize(q_ - M["kmean"][0], dim=-1)).masked_fill(~okm, -9)
                bm = M["rows"][int(sc.argmax())]          # a hit = ANY node of the same fact (duplicates exist)
                lb = store.prov.get(store.nodes[bm]["lsrc"]) if store.nodes[bm]["lsrc"] else None
                hit = bool(lb and lb[0] == "fact" and lb[1:3] == (nm, k))
                if which == "raw":
                    hit_raw += hit
                else:
                    hit_self += hit
            tot += 1
            for x in nl[0]:
                store.nodes[x]["alive"] = False
        return hit_raw / max(1, tot), hit_self / max(1, tot), tot

    def peak_gb():
        return torch.cuda.max_memory_allocated() / 2 ** 30 if dev.type == "cuda" else 0.0

    # ------------------------------------------------------------------ the ONE world
    with torch.no_grad():
        write([world.rule(rng) for _ in range(min(6, A.warm_frames // 10))]       # a few rules, shown once
              + [world.component(rng) for _ in range(A.warm_frames)], False)
        store.commit()
    print(f"WORLD: {len(world.ents)} entities, {len(store.alive())} nodes after {A.warm_frames} warm frames", flush=True)
    if A.load:
        st = torch.load(A.load, map_location=dev)
        load_memory(mem, st["mem"]); ro.load(st)
        graph.load_state_dict(st["graph"], strict=False); loop.load_state_dict(st["loop"], strict=False)
        tools.load_state_dict(st["tools"], strict=False)
        if float(graph.kmu_n) == 0:                       # an older checkpoint: key mean from the warm memory
            with torch.no_grad():
                graph.kmu.copy_(materialize()["kraw"].mean(0)); graph.kmu_n.fill_(1)
        if A.viz:
            visualize(A.viz)
            return
        if A.first:
            first_frame_probe(A.first)
            return
        evaluate(-1)
        return
    if A.warm_tools:
        with torch.no_grad():
            print("roundtrip before warm-up:", roundtrip(random.Random(1)), flush=True)
        warm_roundtrip(A.warm_tools)
        with torch.no_grad():
            print("roundtrip after warm-up:", roundtrip(random.Random(1)), flush=True)
    if not A.smoke and not A.skip_eval0:
        evaluate(0)
    t0, best = time.time(), -1.0
    import collections
    recent_times = collections.deque(maxlen=3)
    steps = 3 if A.smoke else A.steps
    tsum = {}

    def tmark(name, t0=[None]):
        """step-time breakdown (seconds per part, synchronized), printed with the training log"""
        if dev.type == "cuda":
            torch.cuda.synchronize()
        now_ = time.perf_counter()
        if name is not None and t0[0] is not None:
            tsum[name] = tsum.get(name, 0.0) + now_ - t0[0]
        t0[0] = now_

    tmark.last_print = 0
    for step in range(1, steps + 1):
        tmark("other (log / evict / eval)" if step > 1 else None)
        lr = A.lr * min(1.0, step / 100) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / A.steps)))
        for gp in opt.param_groups:
            gp["lr"] = lr * gp.get("lr_mult", 1.0)
        mem.update_stats = True                           # running statistics follow real training data only
        t_before = store.time
        recent_times.append(t_before)
        hist0 = {f: len(v) for f, v in world.hist.items()}
        run.logit_pen = torch.zeros((), device=dev)
        rule = world.rule(rng) if rng.random() < A.p_rule else None   # a new rule: before the frames it governs
        objs = [("state", state_frame(rng)) for _ in range(A.states)] + [("item", world.component(rng))
                                                                         for _ in range(A.pending)]
        rng.shuffle(objs)
        if rule is not None:
            objs.insert(0, ("item", rule))
        multi0 = {f: len(v) for f, v in world.multi.items()}
        cost = (write([o for k_, o in objs if k_ == "state"], True, "state")
                + write([o for k_, o in objs if k_ == "item"], True))       # soft integrate: later questions train it
        tmark("frames")
        now = {f for f, v in world.hist.items() if len(v) > hist0.get(f, 0)}
        now |= {f for f, v in world.multi.items() if len(v) > multi0.get(f, 0)}
        expect_dependents(now)
        coll = set()                                      # facts of the nodes this step's writes addressed
        for _, o in pend:
            if o[0] in ("replace", "attach"):
                lm = store.prov.get(store.nodes[o[1]]["lsrc"]) if store.nodes[o[1]]["lsrc"] else None
                if lm and lm[0] == "fact":
                    coll.add((lm[1], lm[2]))
        n_corr = int(rng.random() < A.corrections) + int(rng.random() < A.corrections - 1)
        cq = corrections(rng, n_corr)
        run.commit_items = {}
        note_loss = run(cq, True, "note", soft=True)[0] if cq else torch.zeros((), device=dev)
        if run.commit_items:                         # "if I edit this, what happens?": the fact's question answered from
            it_ = list(run.commit_items.values())    # the target as it is (old value) vs as written (the edit's value);
            l_old = say_nll(torch.stack([x[2] for x in it_]), stackS([x[3] for x in it_]), [x[6] for x in it_])
            l_new = say_nll(torch.stack([x[4] for x in it_]), stackS([x[5] for x in it_]), [x[6] for x in it_])
            lab = (l_new < l_old - A.commit_margin).float()          # commit iff writing makes the answer clearly better
            glc, ulc = torch.stack([x[0] for x in it_]), torch.stack([x[1] for x in it_])
            c_loss = F.binary_cross_entropy_with_logits(glc, lab)   # (the straight-through slope at the chosen write
            if bool(lab.any()):                                     # said "a bit less": at m = 1 the answer is right
                c_loss = c_loss + F.binary_cross_entropy_with_logits(ulc[lab > 0], lab[lab > 0])   # either way)
            note_loss = note_loss + A.w_commit * c_loss
            run.commit_stats.append((float(lab.mean()), float(torch.sigmoid(glc).mean()), float((l_old - l_new).mean())))
        tmark("notes")
        now |= {q["edit"] for q in cq}
        if A.note_expect:
            expect_dependents({q["edit"] for q in cq})   # a correction the note did not write into its fact (drop / new /
        #   attach: the edit output has no fact label) is still asked, graded on the fact's node: without it the
        #   question had no gold, the note's gate saw only the storage cost and collapsed (main15: drop -> drop)
        gone = soft_forget(t_before)
        H, _ = holders()
        recent = {(nm, k) for (nm, k, v), ns in H.items() if max(store.nodes[i]["time"] for i in ns) > recent_times[0]}
        forgot = {(nm, k) for (nm, k, v), ns in H.items() if any(i in gone for i in ns)
                  and world.ents.get(nm, {}).get(k) == v}
        n_now, n_rec = int(round(A.now * A.questions)), int(round(A.recent * A.questions))
        n_now = max(n_now, min(len(now), A.questions - n_rec - 2))   # every fact written this step is asked (a fact
        #        never asked in its step got only the storage-cost gradient -> the gate learned to drop new facts)
        n_col = int(round(A.collateral * A.questions)) if coll else 0
        n_fg = min(len(forgot), int(round(A.forget_q * A.questions)))   # "was forgetting that a mistake?"
        deps_now = set(world.deps) if A.dep_follow != "eager" else set()   # lazy dependents get their own share: drawn
        n_dp = min(len(deps_now), int(round(A.dep_q * A.questions)))    # by count they were ~1 in 50 lookups and the
        #                                                                  2-read chain faded (rd6: .37 / 1.00 -> .37 / .00)
        qs = ask(rng, [(n_now, now), (n_col, coll), (n_rec, recent), (n_fg, forgot), (n_dp, deps_now),
                       (A.questions - n_now - n_col - n_rec - n_fg - n_dp, None)],
                 A.mix, tr_ops, "train", soft=True)
        tmark("ask")
        run.self_loss = torch.zeros((), device=dev)
        loss, em, act, tool_said = run(qs, True)
        tmark("questions")
        loss = loss + 0.01 * run.logit_pen               # (gate logits accumulated over this step's frame / note rows)
        loss = (loss * len(qs) + note_loss * len(cq)) / max(1, len(qs) + len(cq)) + cost   # notes weigh as rows
        rt_loss, rt_ok = roundtrip_loss(rng, 12)               # keeps decoding exact (it drifted after the warm-up;
        #                                                        misread operands were the arithmetic bottleneck)
        rt_stats.append(rt_ok)
        loss = loss + A.w_self * run.self_loss + A.w_inv * invariance_loss(rng) + A.w_rt * rt_loss
        tmark("roundtrip+invariance")
        opt.zero_grad(set_to_none=True)
        opt_ref.zero_grad(set_to_none=True)
        if loss.requires_grad:                            # (nothing askable this step: no update, decisions still run)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
        tmark("backward")
        mem.update_stats = False
        commit_pending()                                  # the hard decisions of this step's integrate
        opt.step()
        heads_changed()
        rf_ = ref_loss(rng)                              # r edges: the TASK gradient (stale dependents asked,
        if A.w_ref_demo and rf_.requires_grad:           # soft propagation) is already in their grads; the
            (A.w_ref_demo * rf_).backward()              # demonstration (the generator's source) is optional
        opt_ref.step()                                   # (own optimizer: inside the main one they did not learn)
        opt_ref.zero_grad(set_to_none=True)
        tmark("commit+step")
        store.commit()
        evict(gone=gone if gone else None)
        if step % 10 == 0 and dev.type == "cuda":
            torch.cuda.empty_cache()
        if step % 25 == 0 or A.smoke:
            n_live = len(store.alive())
            ws = {k: (v[0], round(v[1] / max(1, v[0]), 2), round(v[2] / max(1, v[0]), 2), round(v[3] / max(1, v[0]), 2))
                  for k, v in wstats.items()}
            wstats.clear()
            pp = {t: round(sum(v) / len(v), 2) for t, v in sorted(run.per_pass.items())}
            run.per_pass = {}
            if run.masked_gold:
                print(f"  demonstrated targets on masked choices (skipped, since the last report): {run.masked_gold}",
                      flush=True)
                run.masked_gold = {}
            said_ ="- (no question this step: no fact held)" if em is None else f"{float(em.mean()):.2f}"
            print(f"step {step} loss {float(loss):.3f} action acc {act:.2f} answer said {said_} "
                  f"tool outputs said {tool_said if tool_said is None else round(tool_said, 2)} | world "
                  f"{len(world.ents)} entities, {n_live} live nodes | peak VRAM {peak_gb():.2f} GB [{time.time() - t0:.0f}s]",
                  flush=True)
            print(f"  read actions: {pp} | integrate by kind (n, store, NEW, replace): {ws}", flush=True)
            if A.look and run.look_tr:
                print(f"  look (proposed -> after looking): {dict(run.look_tr)}", flush=True)
                run.look_tr.clear()
            if tsum:
                n_ = max(1, step - tmark.last_print)
                tmark.last_print = step
                tt_ = sum(tsum.values())
                print(f"  step time ({tt_ / n_:.1f} s / step): " + " | ".join(
                    f"{k_} {v_ / n_:.2f} ({100 * v_ / tt_:.0f}%)" for k_, v_ in tsum.items()), flush=True)
                tsum.clear()
                print(f"  reference re-read (inside the parts above): {ref_t['s'] / n_:.2f} s / step | calls "
                      f"{ref_t['calls'] / n_:.0f} | rows {ref_t['rows'] / n_:.0f} | new texts {ref_t['new'] / n_:.0f} "
                      f"/ step | texts cached {len(alone['row'])}", flush=True)
                ref_t.update(s=0.0, calls=0, rows=0, new=0)
            if ref_stats:
                print(f"  reference edges: dependents -> their source {sum(x[0] for x in ref_stats) / len(ref_stats):.2f} | "
                      f"other facts -> none {sum(x[1] for x in ref_stats) / len(ref_stats):.2f} | dependents / step "
                      f"{sum(x[2] for x in ref_stats) / len(ref_stats):.1f} | misses hit: "
                      f"{dict(sum((x[3] for x in ref_stats), Counter_()).most_common())}", flush=True)
                ref_stats.clear()
            if run.fcommit_stats:
                fs_ = run.fcommit_stats
                print(f"  frame commit target (writing here keeps that fact right) {sum(x[0] for x in fs_) / len(fs_):.2f} | "
                      f"gate there {sum(x[1] for x in fs_) / len(fs_):.2f} | writes into nodes / step "
                      f"{sum(x[2] for x in fs_) / len(fs_):.1f}", flush=True)
                run.fcommit_stats = []
            if run.forget_stats:
                print(f"  forget (model's choice): {sum(run.forget_stats) / len(run.forget_stats):.1f} nodes / step | keep "
                      f"head weights [ver, log use, log age, log act, gate, state, question] "
                      f"{[round(x, 2) for x in graph.Wf_feat.weight[0].tolist()]}", flush=True)
                run.forget_stats = []
            if run.comp_stats:
                cs_ = run.comp_stats
                print(f"  compose (refine on write): P(compose) {sum(x[0] for x in cs_) / len(cs_):.2f} | used "
                      f"{sum(x[0] > 0.5 for x in cs_) / len(cs_):.2f} | context included {sum(x[1] for x in cs_) / len(cs_):.2f}"
                      f" | |dv| {sum(x[2] for x in cs_) / len(cs_):.3f} |dk| {sum(x[3] for x in cs_) / len(cs_):.3f}", flush=True)
                run.comp_stats = []
            if run.commit_stats:
                cs_ = run.commit_stats
                print(f"  commit target (writing helps) {sum(x[0] for x in cs_) / len(cs_):.2f} | gate {sum(x[1] for x in cs_) / len(cs_):.2f}"
                      f" | NLL old - new {sum(x[2] for x in cs_) / len(cs_):.2f}", flush=True)
                run.commit_stats = []
            if run.note_rec:
                nr_ = run.note_rec
                print("  note writes at the target (n, gate, replace, confidence, gate_new, same, is target) by pass: "
                      + str({k_: [len(v_)] + [round(sum(x[i] for x in v_) / len(v_), 2) for i in range(6)]
                             for k_, v_ in nr_.items()}), flush=True)
                run.note_rec = {}
            if inv_stats:
                print(f"  label invariance (same fact, two contexts, top-1 of {16}): {sum(inv_stats) / len(inv_stats):.2f}",
                      flush=True)
                inv_stats.clear()
            if rt_stats:
                print(f"  roundtrip in training (said exactly): {sum(rt_stats) / len(rt_stats):.2f}", flush=True)
                rt_stats.clear()
            logf.write(json.dumps({"type": "train", "step": step, "loss": float(loss), "action": act, "per_pass": pp,
                                   "integrate": ws, "said": None if em is None else float(em.mean()), "nodes": n_live}) + "\n")
        if not A.smoke and (step % A.eval_every == 0 or step == A.steps):
            out = evaluate(step)
            sc = sum(v["exact"] for v in out.values())
            torch.save({"mem": mem.state_dict(), **ro.state(), "graph": graph.state_dict(), "loop": loop.state_dict(),
                        "tools": tools.state_dict(), "step": step, "args": vars(A)}, A.save.replace(".pt", "_last.pt"))
            if sc > best:
                best = sc
                torch.save({"mem": mem.state_dict(), **ro.state(), "graph": graph.state_dict(),
                            "loop": loop.state_dict(), "tools": tools.state_dict(), "step": step, "args": vars(A)},
                           A.save)
    if A.smoke:
        A.eval_q = A.questions
        evaluate(-1)
        print(f"SMOKE OK | peak VRAM {peak_gb():.2f} GB", flush=True)


if __name__ == "__main__":
    main()
