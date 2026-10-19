#!/usr/bin/env python3
"""Universal memory hub (prototype + bench), built on the LTM read in `gmem_ltm.py`.

Three changes over the LTM prototype:

1. SCHEMA-FREE STATE. `ingest()` turns ANY JSON value (nested dicts, lists of records or
   scalars, nulls, bools, long strings, any key casing) into exact `HubItem`s: one per leaf,
   with an automatic record label (the nearest id/name-like field), scalar-sibling context,
   list counts and list summaries. `diff_items()` turns two versions into history items. No
   templates: the bench draws random schemas, and evaluates on HELD-OUT key vocabulary and
   HELD-OUT question phrasings.

2. COPY-OR-GENERATE ANSWERS. LOTUS passes still retrieve, read and re-query (span pointer
   for intermediate hops), but the FINAL answer is written by Walker's own LM head, mixed
   with a copy distribution over memory + question tokens (pointer-generator, with an exact
   copy-continuation cue). Verbatim copying is kept; Walker can now also compose, and read
   things that have no tokens to copy (image states).

3. MODULE HUB. Every module returns Evidence items that go into the same layer-L prefix:
     encoder  -- a pretrained, frozen image encoder + trainable resampler adapter: an image
                 becomes a header text + n vectors scaled to Walker's residual norm;
     tool     -- exact executors called by an ACTION head; the call text ("sum(3, 4.5)") is
                 GENERATED (copying numbers from memory), executed exactly, and the result
                 comes back as an item the next pass reads (calculator built in);
     learned  -- small networks SPAWNED on demand: a question whose centered feature is far
                 from every module's key spawns a new module (soft-token generator); routing
                 is hard top-1, so gradients only reach the routed module (others untouched,
                 no interference); a module can be consolidated (frozen) once it is good.

Run: python gmem_hub.py --ckpt /marimo/ckpt/checkpoints/ckpt_ongoing.pt --out /marimo/gmem_hub
"""
import argparse
import ast
import bisect
import collections
import dataclasses
import itertools
import math
import operator
import os
import random
import re
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import gmem_ltm as LT
from gmem_ltm import MAX_SPAN, MemItem, _norm, match_lengths

T = LT.T
PARA_P = 0.0            # share of training questions from the paraphrase grammar (--paraphrase)
CONT_CAP = 8            # copy-continuation cue length cap for generation


# ================================================================== items
@dataclasses.dataclass
class HubItem(MemItem):
    kind: str = "text"        # text | image | vec
    feat: object = None       # image: frozen encoder features [N, E] (cpu)
    vec: object = None        # vec: [n, D] states (with grad), e.g. a spawned module's output
    n_vec: int = 0
    meta: dict = None


def kind_of(it):
    return getattr(it, "kind", "text")


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


def _chunks(value, n_words=24):
    w = value.split()
    return [" ".join(w[i:i + n_words]) for i in range(0, len(w), n_words)] or [""]


def leaf_text(lf, doc="", max_ctx=0, ctx_chars=110):
    """Exact item text: '<doc>: <path words> = <value> (<record label>; <siblings>)'."""
    ctx = ([lf.label] if lf.label else []) + [f"{k} = {v}" for k, v in lf.context
                                             if f"{k} = {v}" != lf.label and len(v) <= 40][:max_ctx]
    c = "; ".join(ctx)
    if len(c) > ctx_chars:
        c = c[:ctx_chars].rsplit(";", 1)[0]
    head = f"{doc}: " if doc else ""
    return f"{head}{_path_words(lf.path)} = {lf.value}" + (f" ({c})" if c else "")


def ingest(obj, doc=""):
    """ANY JSON value -> exact items. Long strings are split into parts; images become image items."""
    items = []
    for lf in flatten(obj):
        if lf.image is not None:
            ctx = "; ".join(([lf.label] if lf.label else []) + [f"{k} = {v}" for k, v in lf.context
                                                                 if f"{k} = {v}" != lf.label][:3])
            head = f"{doc}: " if doc else ""
            items.append(HubItem(f"{head}image {_path_words(lf.path)}" + (f" ({ctx})" if ctx else ""),
                                 kind="image", meta={"path": lf.path, "image": lf.image, "leaf": lf}))
            continue
        parts = _chunks(lf.value) if len(lf.value.split()) > 24 else [lf.value]
        for j, part in enumerate(parts):
            l2 = dataclasses.replace(lf, value=part) if len(parts) == 1 else dataclasses.replace(
                lf, value=part, path=lf.path + (f"part {j + 1} of {len(parts)}",))
            items.append(HubItem(leaf_text(l2, doc), meta={"path": lf.path, "leaf": lf, "part": j}))
    return items


def diff_items(old, new, k, doc=""):
    """History items for every leaf whose value changed (or vanished) between two versions."""
    nl = {lf.path: lf for lf in flatten(new)}
    out = []
    for lf in flatten(old):
        if lf.image is not None or lf.path[-1:] == ("count",):
            continue
        cur = nl.get(lf.path)
        if cur is not None and cur.value == lf.value:
            continue
        ctx = lf.label
        head = f"{doc}: " if doc else ""
        tail = "" if cur is not None else " and was then removed"
        out.append(HubItem(f"{head}{_path_words(lf.path)} was {lf.value} before update {k}{tail}"
                           + (f" ({ctx})" if ctx else ""), version=k, meta={"path": lf.path, "leaf": lf}))
    return out


# ================================================================== memory objects
def facets_of(it):
    """What a memory item IS, as an object: kind, field, record, value, doc (cached in meta)."""
    meta = it.meta if getattr(it, "meta", None) is not None else {}
    if "facets" in meta:
        return meta["facets"]
    lf = meta.get("leaf")
    f = {"kind": "note", "field": "", "record": "", "value": "", "doc": ""}
    if meta.get("tool"):
        f.update(kind="result", value=str(meta.get("result", "")))
    elif lf is not None:
        keys = [k for k in lf.path if isinstance(k, str)]
        field = words(keys[-1]) if keys else ""
        rest = list(lf.path[:-1]) if lf.path and isinstance(lf.path[-1], str) else list(lf.path)
        f.update(kind="image" if kind_of(it) == "image" else "history" if getattr(it, "version", 0)
                 else "count" if field == "count" else "field",
                 field=field, record=lf.label_value or _path_words(tuple(rest)), value=lf.value,
                 doc=it.text.split(": ", 1)[0] if ": " in it.text else "")
    else:
        f.update(value=it.text.split(": ", 1)[-1])
    if getattr(it, "meta", None) is not None:
        it.meta["facets"] = f
    return f


def tagged(it, obj, **meta_kw):
    """A shallow copy of an item carrying a latent object tag `obj` (n_obj, D) plus bookkeeping meta;
    obj=None strips the tag. The store item itself is never modified."""
    if kind_of(it) == "vec":
        return it
    meta = dict(getattr(it, "meta", None) or {})
    meta.pop("obj", None)
    if obj is not None:
        meta["obj"] = obj
    meta.update(meta_kw)
    return HubItem(it.text, getattr(it, "node", None), getattr(it, "version", 0), kind=kind_of(it),
                   feat=getattr(it, "feat", None), vec=getattr(it, "vec", None), n_vec=getattr(it, "n_vec", 0),
                   meta=meta)


# ================================================================== tools
_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Mod: operator.mod, ast.Pow: operator.pow}
_FUN = {"sum": lambda *a: sum(a), "min": lambda *a: min(a), "max": lambda *a: max(a),
        "mean": lambda *a: sum(a) / len(a), "abs": abs, "round": lambda a, n=0: round(a, int(n)),
        "count": lambda *a: len(a)}


def calc(expr):
    """Exact, safe arithmetic: numbers, + - * / % **, sum/min/max/mean/abs/round/count. Returns a
    canonical number string, or 'error'. No names, attributes or anything else is evaluated."""
    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
            return n.value
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            v = ev(n.operand)
            return -v if isinstance(n.op, ast.USub) else v
        if isinstance(n, ast.BinOp) and type(n.op) in _BIN:
            a, b = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Pow) and abs(b) > 12:
                raise ValueError("exponent too large")
            return _BIN[type(n.op)](a, b)
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _FUN and not n.keywords
                and n.args):
            return _FUN[n.func.id](*[ev(a) for a in n.args])
        raise ValueError("not allowed")
    try:
        if len(expr) > 400:
            return "error"
        v = ev(ast.parse(expr.strip(), mode="eval"))
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return "error"
        return fmt_num(float(v) if isinstance(v, float) else v)
    except Exception:
        return "error"


CALC_OPS = ("sum", "max", "min", "mean", "absdiff")


class ToolModule(nn.Module):
    """An exact executor behind the action head. `key` is its routing key (learned). A call is
    STRUCTURED: an op from `ops` plus argument values taken verbatim from selected memory items
    (Walker's frozen LM head cannot be relied on to spell call syntax)."""
    kind = "tool"

    def __init__(self, name, fn, ops=CALC_OPS, kdim=64):
        super().__init__()
        self.name, self.fn, self.ops = name, fn, tuple(ops)
        self.key = nn.Parameter(torch.randn(kdim) * 0.02)

    def run(self, args):
        res = self.fn(args)
        return HubItem(f"{self.name}: {args} = {res}", kind="text", meta={"tool": self.name, "result": res}), res

    def call(self, op, vals):
        if op == "absdiff" and len(vals) == 2:
            return self.run(f"abs( {vals[0]} - {vals[1]} )")
        return self.run(f"{op}( {', '.join(vals)} )")


def item_value(it):
    """The exact value an item holds (its leaf value; else the text after ' = ')."""
    lf = (getattr(it, "meta", None) or {}).get("leaf")
    if lf is not None and lf.value:
        return lf.value
    if (getattr(it, "meta", None) or {}).get("result") is not None:
        return it.meta["result"]
    m = re.search(r" = (.*?)(?: \(|$)", it.text)
    if m:
        return m.group(1).strip()
    m = re.search(r"-?\d+(?:\.\d+)?", it.text)              # free-text note: its first number
    return m.group(0) if m else it.text


# ================================================================== image encoder + adapter
class StubEncoder(nn.Module):
    """Tiny frozen stand-in for a pretrained encoder (tests): 32x32 RGB -> 16 patch features."""

    def __init__(self, E=32):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.w = nn.Parameter(torch.randn(48, E, generator=g) * 0.2, requires_grad=False)
        self.E = E

    @torch.no_grad()
    def forward(self, images):
        import numpy as np
        xs = [torch.tensor(np.asarray(im.convert("RGB").resize((32, 32))), dtype=torch.float32) / 255.0
              for im in images]
        x = torch.stack(xs).to(self.w.device)                                  # [B, 32, 32, 3]
        p = x.reshape(-1, 4, 8, 4, 8, 3).permute(0, 1, 3, 2, 4, 5).reshape(len(images), 16, 4, 4, 4, 3)
        p = p[:, :, ::2, ::2].reshape(len(images), 16, -1)[:, :, :48]
        return p @ self.w


def load_image_encoder(name, dev):
    """Returns (encode(images) -> [B, N, E] float on dev, E). Frozen; features are cached per image."""
    if name == "stub":
        enc = StubEncoder().to(dev)
        return enc, enc.E
    import transformers
    try:
        proc = transformers.AutoImageProcessor.from_pretrained(name)
        cls = transformers.SiglipVisionModel if "siglip" in name else transformers.CLIPVisionModel
        m = cls.from_pretrained(name, torch_dtype=torch.bfloat16)
    except Exception as e:                                  # fall back to CLIP if SigLIP is unavailable
        print(f"HUB image encoder {name} failed ({type(e).__name__}: {e}); using openai/clip-vit-base-patch16",
              flush=True)
        name = "openai/clip-vit-base-patch16"
        proc = transformers.AutoImageProcessor.from_pretrained(name)
        m = transformers.CLIPVisionModel.from_pretrained(name, torch_dtype=torch.bfloat16)
    print(f"HUB image encoder {name} params={sum(p.numel() for p in m.parameters())/1e6:.0f}M", flush=True)
    m = m.to(dev).eval()
    for p in m.parameters():
        p.requires_grad_(False)

    @torch.no_grad()
    def enc(images):
        px = proc(images=[im.convert("RGB") for im in images], return_tensors="pt")["pixel_values"]
        hs = m(pixel_values=px.to(dev, torch.bfloat16)).last_hidden_state.float()
        n = hs.shape[1]
        g = int(round(n ** 0.5))
        if g * g == n and g % 2 == 0:                                           # 14x14 -> 7x7 average pool
            hs = hs.reshape(-1, g, g, hs.shape[-1]).permute(0, 3, 1, 2)
            hs = F.avg_pool2d(hs, 2).flatten(2).transpose(1, 2)
        return hs
    return enc, m.config.hidden_size


class Resampler(nn.Module):
    """Perceiver-style adapter: n learned queries cross-attend to the frozen encoder features."""

    def __init__(self, E, D, n_tok=16, d=512, heads=8, layers=2):
        super().__init__()
        self.inp = nn.Linear(E, d)
        self.q = nn.Parameter(torch.randn(n_tok, d) * 0.02)
        self.blocks = nn.ModuleList(nn.ModuleDict({
            "n1": nn.LayerNorm(d), "x": nn.MultiheadAttention(d, heads, batch_first=True),
            "n2": nn.LayerNorm(d), "ff": nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))})
            for _ in range(layers))
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, D)

    def forward(self, feats):
        kv = self.inp(feats)
        x = self.q[None].expand(feats.shape[0], -1, -1)
        for b in self.blocks:
            x = x + b["x"](b["n1"](x), kv, kv, need_weights=False)[0]
            x = x + b["ff"](b["n2"](x))
        return self.out(self.norm(x))


class VisionModule(nn.Module):
    kind = "encoder"

    def __init__(self, E, D, n_tok=16):
        super().__init__()
        self.name, self.n_tok = "vision", n_tok
        self.adapter = Resampler(E, D, n_tok)
        self.log_gain = nn.Parameter(torch.tensor(0.0))

    def forward(self, feats, ref):
        v = self.adapter(feats.float())
        return F.normalize(v, dim=-1) * ref * self.log_gain.exp()


# ================================================================== spawned learned modules
class SkillModule(nn.Module):
    """A spawned module: question feature -> m soft tokens (layer-L states) for the answer pass."""
    kind = "learned"

    def __init__(self, D, m=8, hidden=512):
        super().__init__()
        self.m = m
        self.net = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, hidden), nn.GELU(), nn.Linear(hidden, m * D))
        self.log_gain = nn.Parameter(torch.tensor(0.0))

    def forward(self, qp, ref):
        v = self.net(qp.float()).view(qp.shape[0], self.m, -1)
        return F.normalize(v, dim=-1) * ref * self.log_gain.exp()


class SkillBank(nn.Module):
    """Spawn-on-novelty module pool with hard routing.

    Route key = normalized question feature (no centering: a running mean drifts with whatever
    skill came first and turns one skill's rows into noise). Each module keeps the mean `m` and
    spread `s` of its members' key cosine; a row is NOVEL when its best cosine is below
    m - k * max(s, floor) of that module. A training batch holds one task: when at least
    `min_frac` of it is novel it spawns ONE module (key = mean of the novel rows, m and s
    calibrated from their spread). A key follows its members only while the module is young
    (< `settle` updates), then it is fixed, so later skills cannot drag it away. Rows run ONLY their routed module, so
    gradients reach that module alone. `mode`: spawn | shared (one module for everything: the
    interference baseline) | off."""

    def __init__(self, D, mode="spawn", k=4.0, floor=0.01, max_mods=16, m=8, min_group=2, min_frac=0.5,
                 settle=100):
        super().__init__()
        self.D, self.mode, self.k, self.floor, self.max_mods, self.m = D, mode, k, floor, max_mods, m
        self.min_group, self.min_frac, self.settle = min_group, min_frac, settle
        self.mods = nn.ModuleList()
        self.register_buffer("keys", torch.zeros(0, D))
        self.register_buffer("sim_m", torch.zeros(0))
        self.register_buffer("sim_s", torch.zeros(0))
        self.frozen = set()
        self.ema_acc = []
        self.updates = []
        self.on_spawn = None                    # callback(module) -> e.g. add an optimizer param group

    def _spawn(self, z):
        mod = SkillModule(self.D, self.m).to(z.device)
        self.mods.append(mod)
        self.keys = torch.cat([self.keys, F.normalize(z, dim=-1)[None].detach()])
        self.sim_m = torch.cat([self.sim_m, torch.ones(1, device=z.device)])
        self.sim_s = torch.cat([self.sim_s, torch.zeros(1, device=z.device)])
        self.ema_acc.append(0.0)
        self.updates.append(0)
        if self.on_spawn is not None:
            self.on_spawn(mod)
        return len(self.mods) - 1

    def route(self, qp, train):
        """qp [B, D] -> module index per row (-1 = none)."""
        B = qp.shape[0]
        if self.mode == "off":
            return [-1] * B
        z = F.normalize(qp.float(), dim=-1)
        if self.mode == "shared":
            if not len(self.mods):
                if not train:
                    return [-1] * B
                self._spawn(z.mean(0))
            return [0] * B
        out, novel = [-1] * B, []
        if len(self.mods):
            sims = z @ self.keys.T
            best, arg = sims.max(-1)
            thr = self.sim_m - self.k * self.sim_s.clamp(min=self.floor)
            for b in range(B):
                j = int(arg[b])
                if train and float(best[b]) < float(thr[j]):
                    novel.append(b)
                else:
                    out[b] = j
        else:
            novel = list(range(B))
        if train:
            # a batch is one task: when most of it is novel, it spawns ONE module whose novelty
            # threshold is calibrated from the batch's own spread around the new key
            if len(novel) >= max(self.min_group, self.min_frac * B) and len(self.mods) < self.max_mods:
                j = self._spawn(z[novel].mean(0))
                c = z[novel] @ self.keys[j]
                self.sim_m[j] = c.mean()
                self.sim_s[j] = c.std() if len(novel) > 1 else torch.zeros((), device=z.device)
                for b in novel:
                    out[b] = j
            for b in novel:                          # leftovers: nearest module, if any
                if out[b] < 0 and len(self.mods):
                    out[b] = int((self.keys @ z[b]).argmax())
            for b in range(B):                       # member statistics; young keys follow members
                j = out[b]
                if j < 0:
                    continue
                c = float(z[b] @ self.keys[j])
                self.sim_m[j] = 0.95 * self.sim_m[j] + 0.05 * c
                self.sim_s[j] = (0.95 * self.sim_s[j] ** 2 + 0.05 * (c - self.sim_m[j]) ** 2).sqrt()
                if self.updates[j] < self.settle and j not in self.frozen:
                    self.keys[j] = F.normalize(0.95 * self.keys[j] + 0.05 * z[b].detach(), dim=-1)
        return out

    def items(self, qp, idx, ref):
        """One vec HubItem per routed row (None otherwise); each row runs only its module."""
        out = [None] * len(idx)
        for j in sorted(set(i for i in idx if i >= 0)):
            rows = [b for b, i in enumerate(idx) if i == j]
            v = self.mods[j](qp[rows], ref)
            for r_, b in enumerate(rows):
                out[b] = HubItem("", kind="vec", vec=v[r_], n_vec=self.m, meta={"skill": j})
        return out

    def feedback(self, idx, acc, freeze_at=None, min_updates=200):
        """Per-row token accuracy of the answer -> per-module EMA; optional consolidation (freeze)."""
        for i, a in zip(idx, acc):
            if i < 0 or i in self.frozen:
                continue
            self.updates[i] += 1
            self.ema_acc[i] = 0.98 * self.ema_acc[i] + 0.02 * a
            if freeze_at is not None and self.updates[i] >= min_updates and self.ema_acc[i] >= freeze_at:
                self.frozen.add(i)
                for p in self.mods[i].parameters():
                    p.requires_grad_(False)


class ArgSelector(nn.Module):
    """Scores each prefix item as an argument of a tool call. Items attend to each other (so a value
    item can be matched with the category item of the same record) and to the question state."""

    def __init__(self, D, d=256, layers=2, heads=4):
        super().__init__()
        self.inp, self.q = nn.Linear(D, d), nn.Linear(D, d)
        self.extra = nn.Embedding(2, d)                  # tool results produced in earlier passes
        self.used = nn.Embedding(2, d)                   # store items an earlier call of this query consumed
        nn.init.zeros_(self.used.weight)                 # zero-init: no effect until trained
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.out = nn.Linear(d, 1)

    def forward(self, items, hq, is_extra, used=None):
        x = self.inp(items) + self.q(hq)[None] + self.extra(is_extra.long())
        if used is not None:
            x = x + self.used(used.long())
        return self.out(self.enc(x[None]))[0, :, 0]


def num_of(it):
    """The item's value as a finite float, or None (dates, words, booleans, images)."""
    if kind_of(it) != "text":
        return None
    try:
        v = float(item_value(it).replace(",", ""))
    except (ValueError, AttributeError):
        return None
    return v if math.isfinite(v) else None


NUM_F = 8


def number_features(its):
    """Per item: float features (has, sign, log-magnitude, value / set scale, z-score, rank fraction,
    is-max, is-min) and discrete symbols (magnitude bin, rank from top, rank from bottom). Sets are the
    numeric prefix items sharing a field (tool results form their own set)."""
    n = len(its)
    fl = torch.zeros(n, NUM_F)
    mag, rd, ra = torch.full((n,), 13, dtype=torch.long), torch.full((n,), 16, dtype=torch.long), \
        torch.full((n,), 16, dtype=torch.long)
    groups = collections.defaultdict(list)
    vals = [num_of(it) for it in its]
    for m, (it, v) in enumerate(zip(its, vals)):
        if v is not None:
            f = facets_of(it)
            groups[f["field"] or f["kind"]].append(m)
    for ms in groups.values():
        vs = [vals[m] for m in ms]
        sc = max(abs(x) for x in vs) or 1.0
        mu = sum(vs) / len(vs)
        sd = (sum((x - mu) ** 2 for x in vs) / len(vs)) ** 0.5 or 1.0
        order = sorted(ms, key=lambda m: -vals[m])
        for r, m in enumerate(order):
            v = vals[m]
            fl[m] = torch.tensor([1.0, math.copysign(1.0, v) if v else 0.0, math.log10(1 + abs(v)) / 6, v / sc,
                                  (v - mu) / sd, r / max(1, len(ms) - 1), float(r == 0), float(r == len(ms) - 1)])
            rd[m], ra[m] = min(r, 15), min(len(ms) - 1 - r, 15)
            mag[m] = (max(-4, min(8, math.floor(math.log10(abs(v))))) if v else -4) + 4
    return fl, mag, rd, ra


class DiscreteBottleneck(nn.Module):
    """Discrete-valued communication (DVNC, Liu et al. 2021, arXiv 2107.02367), MIXED with floats: the
    first `frac` of a vector is cut into G heads and every head snaps to its nearest code in ONE shared
    codebook of L codes (G symbols from an L-word vocabulary); the rest of the vector stays continuous.
    Straight-through gradients, a commitment loss (collected in `pending`), EMA codebook updates and
    dead-code restarts (no codebook gradients)."""

    def __init__(self, dim, heads=8, codes=64, frac=0.5, beta=0.25, decay=0.99):
        super().__init__()
        self.dq = int(dim * frac) // heads * heads             # discrete part; dim - dq stays float
        assert self.dq > 0, (dim, heads, frac)
        self.G, self.L, self.dh, self.beta, self.decay = heads, codes, self.dq // heads, beta, decay
        self.register_buffer("code", torch.zeros(codes, self.dh))
        self.register_buffer("n_ema", torch.zeros(codes))
        self.register_buffer("m_ema", torch.zeros(codes, self.dh))
        self.register_buffer("ready", torch.zeros((), dtype=torch.bool))
        self.pending, self.last = [], None

    @torch.no_grad()
    def _update(self, z, k):
        if not bool(self.ready):                             # data init: codes = random head vectors
            self.code.copy_(z[torch.randint(len(z), (self.L,), device=z.device)])
            self.m_ema.copy_(self.code)
            self.n_ema.fill_(1.0)
            self.ready.fill_(True)
            return
        oh = F.one_hot(k, self.L).float()
        self.n_ema.mul_(self.decay).add_(oh.sum(0), alpha=1 - self.decay)
        self.m_ema.mul_(self.decay).add_(oh.T @ z, alpha=1 - self.decay)
        tot = self.n_ema.sum()
        n = (self.n_ema + 1e-5) / (tot + self.L * 1e-5) * tot
        self.code.copy_(self.m_ema / n[:, None])
        dead = self.n_ema < 0.03 * tot / self.L                # unused codes restart at live head vectors
        if dead.any():
            j = torch.randint(len(z), (int(dead.sum()),), device=z.device)
            self.code[dead] = z[j]
            self.m_ema[dead] = z[j]
            self.n_ema[dead] = tot / self.L

    def forward(self, x):
        xd, xc = x[..., :self.dq], x[..., self.dq:]
        return torch.cat([self._snap(xd), xc], -1)

    def _snap(self, x):
        z = x.float().reshape(-1, self.dh)
        train = torch.is_grad_enabled()
        if train:
            self._update(z.detach(), None if not bool(self.ready) else
                         torch.cdist(z.detach(), self.code).argmin(1))
        k = torch.cdist(z.detach(), self.code).argmin(1)
        e = self.code[k]
        if train:
            self.pending.append(self.beta * F.mse_loss(z, e))
        self.last = k
        return (z + (e - z).detach()).to(x.dtype).reshape(x.shape)

    def perplexity(self):
        if self.last is None or not len(self.last):
            return 0.0
        pr = torch.bincount(self.last, minlength=self.L).float()
        pr = pr / pr.sum()
        return float(torch.exp(-(pr * (pr + 1e-10).log()).sum()))


class ObjectTagger(nn.Module):
    """Latent object tag: N_OBJ vectors injected in front of a prefix item, never copyable.
    Token 0 = WHAT the item is: its field, its record and its content (Walker's pooled features, cached
    once per store); a tool result is its op plus the WHAT of its arguments. Token 1 = how it RELATES to
    the question: facet-wise agreement (field / record / content) with the question and with the search
    that brought it in, its retrieval score and rank, how many passes ago it entered, provenance flags.
    Everything is vectors: no tag text is generated or re-encoded."""
    N_OBJ = 2
    FLAGS = ("carried", "tool", "image", "has_field", "has_record", "new")

    def __init__(self, D, n_ops, d=256, discrete=None, numbers=False):
        super().__init__()
        self.n_obj = self.N_OBJ + int(numbers)
        # optional VALUE token (mixed float + discrete): numbers the network can compare without a tool
        self.numbers = numbers
        if numbers:
            self.nf = nn.Sequential(nn.Linear(NUM_F, d), nn.GELU(), nn.Linear(d, d))
            self.mag, self.rd, self.ra = nn.Embedding(14, d), nn.Embedding(17, d), nn.Embedding(17, d)
            self.no = nn.Linear(d, D)
        self.f, self.g, self.k, self.q = nn.Linear(D, d), nn.Linear(D, d), nn.Linear(D, d), nn.Linear(D, d)
        # optional discrete-valued tags: WHAT and RELATES each become G symbols from a shared codebook
        self.vq_what = DiscreteBottleneck(d, *discrete) if discrete else None
        self.vq_rel = DiscreteBottleneck(d, *discrete) if discrete else None
        self.op = nn.Embedding(n_ops + 1, d)                 # last row: an unstructured call
        self.rank, self.age = nn.Embedding(16, d), nn.Embedding(8, d)
        n_in = 6 * d + 6 + 1 + len(self.FLAGS)
        self.rel = nn.Sequential(nn.Linear(n_in, d), nn.GELU(), nn.Linear(d, d))
        self.wo, self.ro = nn.Linear(d, D), nn.Linear(d, D)
        self.log_gain = nn.Parameter(torch.tensor(0.0))

    def parts(self, field, rec, key, has_f, has_r):
        """(n, 3, d): field / record / content embeddings; a missing facet is exactly zero."""
        return torch.stack([self.f(field) * has_f[:, None], self.g(rec) * has_r[:, None], self.k(key)], 1)

    def tool_parts(self, op_i, what_args):
        z = torch.zeros_like(what_args)
        return torch.stack([z, z, self.op.weight[op_i] + what_args], 0)

    def forward(self, parts, q, s, logit, rank, age, flags, ref, num=None):
        """parts (n,3,d); q (D) question; s (n,D) search that found each item; logit (n); rank/age (n) long;
        flags (n,F). Returns tags (n, N_OBJ, D) at Walker's residual scale and WHAT (n, d)."""
        what = parts.sum(1)
        qd, sd = self.q(q)[None, None], self.q(s)[:, None]
        rel_in = torch.cat([(parts * qd).flatten(1), (parts * sd).flatten(1),
                            F.cosine_similarity(parts, qd.expand_as(parts), dim=-1),
                            F.cosine_similarity(parts, sd.expand_as(parts), dim=-1),
                            logit[:, None], flags], -1)
        rel = self.rel(rel_in) + self.rank(rank.clamp(0, 15)) + self.age(age.clamp(0, 7))
        if self.vq_what is not None:
            what, rel = self.vq_what(what), self.vq_rel(rel)
        toks = [self.wo(what), self.ro(rel)]
        if self.numbers:
            fl, mag, rd, ra = (t.to(what.device) for t in num)
            toks.append(self.no(self.nf(fl) + self.mag(mag) + self.rd(rd) + self.ra(ra)))
        out = torch.stack(toks, 1)
        return F.normalize(out, dim=-1) * ref * self.log_gain.exp(), what


# ================================================================== heads
class HubHeads(LT.LTMHeads):
    def __init__(self, D, tools=(), vision=None, skills=None, reader="xattn", kdim=512, pdim=256, discrete=None,
                 numbers=False):
        super().__init__(D, kdim=kdim, pdim=pdim, reader=reader)
        self.act = nn.Linear(D, 2)                    # answer | requery
        self.tool_q = nn.Linear(D, 64)                # routing query against each tool's key
        self.op = nn.ModuleDict({t.name: nn.Linear(D, len(t.ops)) for t in tools})
        # the op also reads WHAT the chosen arguments say ("gives" -> subtract); zero-init = old behaviour
        self.op_arg = nn.ModuleDict({t.name: nn.Linear(D, len(t.ops)) for t in tools})
        for m in self.op_arg.values():
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
        self.arg_sel = ArgSelector(D)                 # which prefix items are the call's arguments
        # facet retrieval: the question is matched against an item's FIELD and RECORD separately
        self.fq_field, self.fk_field = nn.Linear(D, 256), nn.Linear(D, 256)
        self.fq_rec, self.fk_rec = nn.Linear(D, 256), nn.Linear(D, 256)
        self.fw = nn.Parameter(torch.tensor([0.5, 0.5]))
        self.obj = ObjectTagger(D, len(CALC_OPS), discrete=discrete, numbers=numbers)   # (--objects)
        # discrete facet keys: question side and item side snap to the same codebook per facet
        self.vq_field = DiscreteBottleneck(256, *discrete) if discrete else None
        self.vq_rec = DiscreteBottleneck(256, *discrete) if discrete else None
        self.tools = nn.ModuleDict({t.name: t for t in tools})
        self.vision = vision
        self.skills = skills
        # copy-or-generate
        self.cq, self.ck = nn.Linear(D, pdim), nn.Linear(D, pdim)
        self.cont = nn.Embedding(CONT_CAP + 1, 1)
        with torch.no_grad():
            self.cont.weight[:, 0] = torch.arange(CONT_CAP + 1).float() * 1.0
        self.crb = nn.Parameter(torch.tensor(1.0))    # retrieval log-prob bias on copy positions
        self.gate = nn.Linear(D, 1)                   # P(generate from the LM head) vs copy
        self.qsrc = nn.Parameter(torch.tensor(0.0))   # bias for copying from the question
        # first answer token: start where the memory continues the question's own words (quoted heads)
        self.qm = nn.Embedding(LT.MATCH_CAP + 1, 1)
        with torch.no_grad():
            self.qm.weight[:, 0] = torch.arange(LT.MATCH_CAP + 1).float() * 0.5
        self.echo = nn.Parameter(torch.tensor(0.0))   # learned bias on memory tokens that occur in the question

    def bottlenecks(self):
        return {n: b for n, b in (("what", self.obj.vq_what), ("rel", self.obj.vq_rel),
                                  ("field", self.vq_field), ("rec", self.vq_rec)) if b is not None}

    def facet_snap(self, i, x):
        b = (self.vq_field, self.vq_rec)[i]
        return x if b is None else b(x)

    def actions(self):
        return ["answer", "requery"] + list(self.tools)

    def action_logits(self, lat):
        lg = self.act(lat)
        if len(self.tools):
            keys = torch.stack([t.key for t in self.tools.values()])
            lg = torch.cat([lg, self.tool_q(lat) @ keys.T], -1)
        return lg


# ================================================================== runner
class HubRunner(LT.LTMRunner):
    def __init__(self, target, tok, heads, layer, qpos="before"):
        super().__init__(target, tok, heads, layer, qpos=qpos)
        self.norm_ref = None                          # EMA of Walker's per-token residual norm at layer L

    @torch.no_grad()
    def layer_states(self, seqs):
        out = super().layer_states(seqs)
        n = torch.cat([x.float().norm(dim=-1) for x in out]).mean()
        self.norm_ref = n if self.norm_ref is None else 0.99 * self.norm_ref + 0.01 * n
        return out

    def ref(self):
        return self.norm_ref if self.norm_ref is not None else torch.tensor(1.0, device=self.dev)

    # ------------------------------------------------------------ prefix
    @staticmethod
    def n_obj(it):
        obj = (getattr(it, "meta", None) or {}).get("obj")
        return 0 if obj is None else obj.shape[0]

    def _segs(self, it):
        """(ids, copyable mask, vec slot start or None). A latent object tag (if any) leads: N_OBJ
        placeholder slots, never copyable."""
        if kind_of(it) == "vec":
            return [self.tok.ph] * it.n_vec, [False] * it.n_vec, 0
        tid = [self.tok.ph] * self.n_obj(it)
        if kind_of(it) == "image":
            hid = self.tok.ids(" " + it.text)[:48]
            n = self.heads.vision.n_tok
            return (tid + hid + [self.tok.ph] * n + [self.tok.nl],
                    [False] * len(tid) + [True] * len(hid) + [False] * (n + 1), len(tid) + len(hid))
        ids = self.tok.ids(" " + it.text + "\n")[:96]
        return tid + ids, [False] * len(tid) + [True] * len(ids), None

    def prep(self, items_per_q, questions):
        """Layer-L states of every prefix item (text: Walker's own; image: adapter; vec: given)."""
        hd = self.heads
        segs = [[self._segs(it) for it in its] for its in items_per_q]
        text_rows = [(b, j) for b, row in enumerate(segs) for j, s in enumerate(row)
                     if kind_of(items_per_q[b][j]) != "vec"]
        states = self.layer_states([segs[b][j][0][self.n_obj(items_per_q[b][j]):] for b, j in text_rows]) \
            if text_rows else []
        st_map = {bj: s for bj, s in zip(text_rows, states)}
        imgs = [(b, j) for b, its in enumerate(items_per_q) for j, it in enumerate(its)
                if kind_of(it) == "image"]
        vis = {}
        if imgs:
            feats = torch.stack([items_per_q[b][j].feat for b, j in imgs]).to(self.dev)
            v = hd.vision(feats, self.ref())
            vis = {bj: v[i] for i, bj in enumerate(imgs)}
        rows = []
        for b, its in enumerate(items_per_q):
            toks, cp, parts, lens = [], [], [], []
            for j, it in enumerate(its):
                ids, c, vs = segs[b][j]
                if kind_of(it) == "vec":
                    x = it.vec
                else:
                    x = st_map[(b, j)].float()
                    if self.n_obj(it):
                        x = torch.cat([it.meta["obj"].float(), x])
                    if kind_of(it) == "image":
                        x = torch.cat([x[:vs], vis[(b, j)].to(x.dtype), x[vs + hd.vision.n_tok:]])
                parts.append(x)
                toks += ids
                cp += c
                lens.append(len(ids))
            rows.append({"toks": toks, "copy": cp, "x": torch.cat(parts) if parts else None, "lens": lens})
        q_ids = [self.tok.ids(f"Q: {q}\n") for q in questions]
        return {"rows": rows, "q_ids": q_ids, "questions": questions, "items": items_per_q}

    def forward_rows(self, pp, suffix=None):
        """[Q][memory][\\nA:][suffix] with memory injected at block L. Returns dict of states."""
        rows, q_ids = pp["rows"], pp["q_ids"]
        B = len(rows)
        P = [len(r["toks"]) for r in rows]
        Pm = max(1, max(P))
        prompt = self.tok.ids("\nA:")
        suffix = suffix or [[] for _ in rows]
        seqs = [q_ids[b] + [self.tok.ph] * Pm + prompt + suffix[b] for b in range(B)]
        ids = self.pad(seqs)
        Tn = ids.shape[1]
        mask = torch.zeros(B, Tn, dtype=torch.bool, device=self.dev)
        vals = torch.zeros(B, Tn, self.heads.D, device=self.dev)
        for b, r in enumerate(rows):
            o = len(q_ids[b])
            if P[b]:
                vals[b, o:o + P[b]] = r["x"].float()
                mask[b, o:o + P[b]] = True
        self.st["inj"] = (mask, vals)
        try:
            logits, _ = self._fwd(ids)
        finally:
            self.st["inj"] = None
        h = F.layer_norm(self.st["h"].float(), (self.heads.D,))
        ar = torch.arange(B, device=self.dev)
        off = [len(q) for q in q_ids]
        a_pos = [off[b] + Pm + len(prompt) - 1 for b in range(B)]
        mem_idx = torch.tensor(off, device=self.dev)[:, None] + torch.arange(Pm, device=self.dev)[None]
        hp = h[ar[:, None], mem_idx]
        Lq = max(off)
        q_idx = torch.zeros(B, Lq, dtype=torch.long, device=self.dev)
        qmask = torch.zeros(B, Lq, dtype=torch.bool, device=self.dev)
        for b in range(B):
            q_idx[b, :off[b]] = torch.arange(off[b], device=self.dev)
            qmask[b, :off[b]] = True
        return {"h": h, "logits": logits, "hq": h[ar, torch.tensor(a_pos, device=self.dev)], "hp": hp,
                "hQ": h[ar[:, None], q_idx], "qmask": qmask, "q_idx": q_idx, "a_pos": a_pos, "P": P, "Pm": Pm,
                "off": off}

    def item_states(self, pp, fw):
        """Mean final state over each prefix item's positions: [n_items, D] per row."""
        out = []
        for b, r in enumerate(pp["rows"]):
            hs, pos = [], 0
            for L in r["lens"]:
                hs.append(fw["hp"][b, pos:pos + L].mean(0))
                pos += L
            out.append(torch.stack(hs) if hs else fw["hp"][b, :0])
        return out

    # ------------------------------------------------------------ span pointer (intermediate hops)
    def span(self, pp, fw, gold=None, item_logits=None, ban=None, n_carry=None):
        hd = self.heads
        rows, P, Pm, hq, hp = pp["rows"], fw["P"], fw["Pm"], fw["hq"], fw["hp"]
        B = len(rows)
        valid = torch.zeros(B, Pm, dtype=torch.bool, device=self.dev)
        present = torch.zeros(B, Pm, dtype=torch.bool, device=self.dev)
        wstart = torch.zeros(B, Pm, dtype=torch.bool, device=self.dev)
        wend = torch.zeros(B, Pm, dtype=torch.bool, device=self.dev)
        mlen = torch.zeros(B, Pm, dtype=torch.long, device=self.dev)
        lp_rows = torch.zeros(B, Pm, device=self.dev)
        carried = torch.zeros(B, Pm, dtype=torch.bool, device=self.dev)
        rank = torch.zeros(B, Pm, dtype=torch.long, device=self.dev)
        for b, r in enumerate(rows):
            if not P[b]:
                continue
            cp = torch.tensor(r["copy"], device=self.dev)
            valid[b, :P[b]] = cp
            present[b, :P[b]] = True
            ws = self.word_starts(r["toks"])
            wstart[b, :P[b]] = torch.tensor(ws, device=self.dev) & cp
            wend[b, :P[b]] = torch.tensor(ws[1:] + [True], device=self.dev) & cp
            mlen[b, :P[b]] = torch.tensor(match_lengths(r["toks"], self.tok.ids(" " + pp["questions"][b])),
                                          device=self.dev)
            lens = torch.tensor(r["lens"], device=self.dev)
            if item_logits is not None:
                lp_rows[b, :P[b]] = torch.repeat_interleave(F.log_softmax(item_logits[b].float(), -1), lens)
            rank[b, :P[b]] = torch.repeat_interleave(torch.arange(len(r["lens"]), device=self.dev), lens)
            nc = n_carry[b] if n_carry else 0
            carried[b, :P[b]] = rank[b, :P[b]] < nc
        top = (mlen == mlen.max(1, keepdim=True).values) & (mlen >= 2)
        cue = (hd.match(mlen.clamp(max=LT.MATCH_CAP)).squeeze(-1) + hd.match_top * top.float())
        cue_t = 2 * torch.sigmoid(hd.mgate(hq)) * cue
        ib = hd.rbias * lp_rows
        if hd.reader is not None:
            s0, e0 = hd.reader(hp, present | ~present.any(1, keepdim=True), fw["hQ"], fw["qmask"], hq, carried, rank,
                               lp_rows.detach())
        else:
            sc = 1.0 / hd.sq.out_features ** 0.5
            s0 = torch.einsum("bw,btw->bt", hd.sq(hq), hd.sk(hp)) * sc
            e0 = torch.einsum("bw,btw->bt", hd.eq(hq), hd.ek(hp)) * sc
        s_start, s_end = s0 + cue_t + ib, e0 + ib
        for b, spans in enumerate(ban or []):
            row = rows[b]["toks"]
            for sp in spans:
                s_ids = self.tok.ids(" " + sp)
                for t in range(len(row) - len(s_ids) + 1):
                    if row[t:t + len(s_ids)] == s_ids:
                        wstart[b, t] = False
        gold_pos = []
        if gold is not None:
            for b, (gi, ans) in enumerate(gold):
                if gi is None or gi < 0 or ans is None:
                    continue
                start = sum(rows[b]["lens"][:gi])
                seg = rows[b]["toks"][start:start + rows[b]["lens"][gi]]
                cps = rows[b]["copy"][start:start + rows[b]["lens"][gi]]
                a_ids = self.tok.ids(" " + ans)
                k = next((i for i in range(len(seg) - len(a_ids) + 1)
                          if seg[i:i + len(a_ids)] == a_ids and all(cps[i:i + len(a_ids)])), -1)
                if k < 0 or len(a_ids) > MAX_SPAN:
                    continue
                gold_pos.append((b, start + k, start + k + len(a_ids) - 1))
                wstart[b, start + k] = True
                wend[b, start + k + len(a_ids) - 1] = True
        ok_s, ok_e = valid & wstart, valid & wend
        ok_s[:, 0] |= ~ok_s.any(1)
        ok_e[:, 0] |= ~ok_e.any(1)
        s_start = s_start.masked_fill(~ok_s, float("-inf"))
        s_end = s_end.masked_fill(~ok_e, float("-inf"))
        o = torch.arange(Pm, device=self.dev)
        allowed = ((o[None, :] - o[:, None]) >= 0) & ((o[None, :] - o[:, None]) < MAX_SPAN)
        pair = (s_start[:, :, None] + s_end[:, None, :]).masked_fill(~allowed, float("-inf"))
        arg = pair.flatten(1).argmax(-1).tolist()
        preds = [self.tok.decode(rows[b]["toks"][a // Pm:a % Pm + 1]) for b, a in enumerate(arg)]
        ends = [list(itertools.accumulate(r["lens"])) for r in rows]
        pred_item = [bisect.bisect_right(ends[b], a // Pm) for b, a in enumerate(arg)]
        ce = None
        if gold_pos:
            ri = torch.tensor([g[0] for g in gold_pos], device=self.dev)
            gs = torch.tensor([g[1] for g in gold_pos], device=self.dev)
            ge = torch.tensor([g[2] for g in gold_pos], device=self.dev)
            ce = F.cross_entropy(s_start[ri], gs) + F.cross_entropy(
                s_end[ri].masked_fill(~allowed[gs], float("-inf")), ge)
        return ce, preds, pred_item, len(gold_pos)

    # ------------------------------------------------------------ copy-or-generate
    def _pool(self, pp, fw, item_logits=None):
        """Copy pool per row: memory positions then question positions."""
        rows, P, Pm = pp["rows"], fw["P"], fw["Pm"]
        B = len(rows)
        Lq = fw["hQ"].shape[1]
        toks = torch.full((B, Pm + Lq), -1, dtype=torch.long, device=self.dev)
        ok = torch.zeros(B, Pm + Lq, dtype=torch.bool, device=self.dev)
        lp = torch.zeros(B, Pm + Lq, device=self.dev)
        region = torch.zeros(B, Pm + Lq, dtype=torch.long, device=self.dev)
        for b, r in enumerate(rows):
            if P[b]:
                toks[b, :P[b]] = torch.tensor(r["toks"], device=self.dev)
                ok[b, :P[b]] = torch.tensor(r["copy"], device=self.dev)
                if item_logits is not None:
                    lp[b, :P[b]] = torch.repeat_interleave(F.log_softmax(item_logits[b].float(), -1),
                                                           torch.tensor(r["lens"], device=self.dev))
                region[b, :P[b]] = torch.repeat_interleave(torch.arange(len(r["lens"]), device=self.dev) + 1,
                                                           torch.tensor(r["lens"], device=self.dev))
            q = pp["q_ids"][b]
            toks[b, Pm:Pm + len(q)] = torch.tensor(q, device=self.dev)
            ok[b, Pm:Pm + len(q)] = True
            region[b, Pm:Pm + len(q)] = 10_000
        hpool = torch.cat([fw["hp"], fw["hQ"]], 1)
        is_q = torch.zeros(B, Pm + Lq, dtype=torch.bool, device=self.dev)
        is_q[:, Pm:] = True
        cache = pp.setdefault("_qm", {})              # per read: decoding rebuilds the pool every token
        if (Pm, Lq) not in cache:
            qmatch = torch.zeros(B, Pm + Lq, dtype=torch.long, device=self.dev)
            echo = torch.zeros(B, Pm + Lq, device=self.dev)
            for b, r in enumerate(rows):
                if P[b]:
                    q_ids = self.tok.ids(" " + pp["questions"][b])
                    qmatch[b, :P[b]] = torch.tensor(match_lengths(r["toks"], q_ids), device=self.dev)
                    qs = set(q_ids)
                    echo[b, :P[b]] = torch.tensor([float(t in qs) for t in r["toks"]], device=self.dev)
            cache[(Pm, Lq)] = (qmatch, echo)
        qmatch, echo = cache[(Pm, Lq)]
        return {"toks": toks, "ok": ok, "lp": lp, "h": hpool, "region": region, "is_q": is_q, "qmatch": qmatch,
                "echo": echo}

    def _cont(self, pool, y_prev):
        """cont[b, j, t] = longest k <= CONT_CAP with pool[t-k:t] == y_prev[b, j-k+1..j] (same region)."""
        toks, region = pool["toks"], pool["region"]
        B, Tp = toks.shape
        J = y_prev.shape[1]
        run = torch.ones(B, J, Tp, dtype=torch.bool, device=self.dev)
        cont = torch.zeros(B, J, Tp, dtype=torch.long, device=self.dev)
        for i in range(1, CONT_CAP + 1):
            if i > Tp:
                break
            pt = torch.full_like(toks, -2)
            pt[:, i:] = toks[:, :-i]
            same = torch.zeros_like(toks, dtype=torch.bool)
            same[:, i:] = region[:, :-i] == region[:, i:]
            yp = torch.full((B, J), -3, dtype=torch.long, device=self.dev)
            if i - 1 < J:
                yp[:, i - 1:] = y_prev[:, :J - i + 1]
            run = run & (pt[:, None, :] == yp[:, :, None]) & same[:, None, :]
            if not run.any():
                break
            cont = cont + run.long()
        return cont

    def gen_dist(self, pool, hg, y_prev, first):
        """Copy attention [B, J, Tp] and generate-gate log-probs at generation states hg [B, J, D].
        first [B, J]: answer steps with no generated token yet (they get the question-match start cue)."""
        hd = self.heads
        sc = 1.0 / hd.cq.out_features ** 0.5
        s = torch.einsum("bjd,btd->bjt", hd.cq(hg), hd.ck(pool["h"])) * sc
        s = s + hd.cont(self._cont(pool, y_prev)).squeeze(-1) + hd.crb * pool["lp"][:, None]
        s = s + hd.qsrc * pool["is_q"].float()[:, None] + hd.echo * pool["echo"][:, None]
        s = s + first.float()[..., None] * hd.qm(pool["qmatch"]).squeeze(-1)[:, None]
        s = s.masked_fill(~pool["ok"][:, None], float("-inf"))
        lg = hd.gate(hg).squeeze(-1)
        return F.log_softmax(s, -1), F.logsigmoid(lg), F.logsigmoid(-lg)

    def gen_loss(self, pp, fw, acts, targets, item_logits=None):
        """Teacher-forced -log p(y) for rows with a target. acts[b]/targets[b]: token lists (targets
        end with "\\n"). Returns (loss or None, per-row token accuracy)."""
        rows_t = [b for b, t in enumerate(targets) if t]
        if not rows_t:
            return None, {}
        pool = self._pool(pp, fw, item_logits)
        J = max(len(targets[b]) for b in rows_t)
        bi = torch.tensor(rows_t, device=self.dev)
        pos = torch.zeros(len(rows_t), J, dtype=torch.long, device=self.dev)
        y = torch.full((len(rows_t), J), -1, dtype=torch.long, device=self.dev)
        yprev = torch.full((len(rows_t), J), -3, dtype=torch.long, device=self.dev)
        m = torch.zeros(len(rows_t), J, dtype=torch.bool, device=self.dev)
        for r, b in enumerate(rows_t):
            base = fw["a_pos"][b] + len(acts[b])
            tg = targets[b]
            pos[r, :len(tg)] = torch.arange(base, base + len(tg), device=self.dev)
            y[r, :len(tg)] = torch.tensor(tg, device=self.dev)
            prev = [-3] + tg[:-1]                        # first token: no continuation context
            yprev[r, :len(tg)] = torch.tensor(prev, device=self.dev)
            m[r, :len(tg)] = True
        hg = fw["h"][bi[:, None], pos]
        sub = {k: v[bi] for k, v in pool.items()}
        first = torch.zeros_like(m)
        first[:, 0] = True
        la, lgen, lcopy = self.gen_dist(sub, hg, yprev, first)
        lm = F.log_softmax(fw["logits"][bi[:, None], pos].float(), -1)
        yc = y.clamp(min=0)
        lm_y = lm.gather(-1, yc[..., None]).squeeze(-1)
        hit = sub["toks"][:, None, :] == yc[..., None]
        cp_y = torch.logsumexp(la.masked_fill(~hit, float("-inf")), -1)
        lp = torch.logaddexp(lgen + lm_y, lcopy + cp_y)
        loss = -(lp * m).sum() / m.sum().clamp(min=1)
        # token accuracy of the mixed distribution's argmax (teacher-forced)
        with torch.no_grad():
            p = (lgen[..., None] + lm).exp()
            pc = torch.zeros_like(p).scatter_add_(-1, sub["toks"].clamp(min=0)[:, None].expand(-1, J, -1),
                                                  (la + lcopy[..., None]).exp())
            pred = (p + pc).argmax(-1)
            acc = ((pred == y) | ~m).all(-1).float()
        return loss, {b: float(acc[r]) for r, b in enumerate(rows_t)}

    def _number_continuation(self, pool, b, src, default):
        """If the memory position `src` that the last token was copied from is followed by "." + digits
        (decimal) or "," + a 3-digit group (thousands), that separator continues the number. Only the copy
        position counts: the same digits followed by "," elsewhere (a list) must not extend the answer."""
        if src is None:
            return default
        toks = pool["toks"][b].tolist()
        reg = pool["region"][b].tolist()
        t = src
        if t + 2 >= len(toks) or not (reg[t] == reg[t + 1] == reg[t + 2]) or toks[t + 1] < 0 or toks[t + 2] < 0:
            return default
        sep = self.tok.decode([toks[t + 1]])
        if sep == "." and self.tok.decode([toks[t + 2]])[:1].isdigit():
            return toks[t + 1]
        if sep == ",":
            back, k = "", t                              # "22.44,771" is a list, not a thousands group
            while k >= 0 and reg[k] == reg[t] and toks[k] >= 0 and len(back) < 12:
                back = self.tok.decode([toks[k]]) + back
                k -= 1
            back = back[len(back.rstrip("0123456789.,")):]
            if "." in back:
                return default
            run, k = "", t + 2
            while k < len(toks) and reg[k] == reg[t] and toks[k] >= 0 and len(run) < 5:
                run += self.tok.decode([toks[k]])
                k += 1
            grp = len(run) - len(run.lstrip("0123456789"))
            if grp == 3 and not run[3:4].isdigit():
                return toks[t + 1]
        return default

    @torch.no_grad()
    def decode(self, pp, acts, item_logits=None, max_new=48, trace=None):
        """Greedy copy-or-generate decoding after "\\nA:" + act tokens. Returns strings. With `trace` (a
        list), appends per row the per-token record: token, P(generate), top-3 candidates, copy source."""
        if trace is not None:
            trace.extend([] for _ in range(len(pp["rows"])))
            base = len(trace) - len(pp["rows"])
        B = len(pp["rows"])
        out = [[] for _ in range(B)]
        done = [False] * B
        src_pos = [None] * B
        stop = {self.tok.nl, self.tok.eos}
        P = max(1, max(len(r["toks"]) for r in pp["rows"]))
        room = getattr(self.target, "max_t", 1 << 30) - (max(len(q) for q in pp["q_ids"]) + P + 3
                                                         + max(len(a) for a in acts))
        for _ in range(max(0, min(max_new, room))):
            fw = self.forward_rows(pp, [acts[b] + out[b] for b in range(B)])
            pool = self._pool(pp, fw, item_logits)
            pos = torch.tensor([fw["a_pos"][b] + len(acts[b]) + len(out[b]) for b in range(B)], device=self.dev)
            ar = torch.arange(B, device=self.dev)
            hg = fw["h"][ar, pos][:, None]
            hist = [out[b][-CONT_CAP:] for b in range(B)]
            yprev = torch.full((B, CONT_CAP), -3, dtype=torch.long, device=self.dev)
            for b in range(B):
                if hist[b]:
                    yprev[b, CONT_CAP - len(hist[b]):] = torch.tensor(hist[b], device=self.dev)
            first = torch.zeros(B, CONT_CAP, dtype=torch.bool, device=self.dev)
            first[:, -1] = torch.tensor([not out[b] for b in range(B)], device=self.dev)
            la, lgen, lcopy = self.gen_dist(pool, hg.expand(-1, CONT_CAP, -1), yprev, first)
            la, lgen, lcopy = la[:, -1], lgen[:, -1], lcopy[:, -1]
            p = (lgen[:, None] + F.log_softmax(fw["logits"][ar, pos].float(), -1)).exp()
            p = p.scatter_add(-1, pool["toks"].clamp(min=0), (la + lcopy[:, None]).exp())
            nxt = p.argmax(-1).tolist()
            for b in range(B):                            # never stop inside a number being copied ("2" "." "84")
                if nxt[b] in stop and out[b] and self.tok.decode(out[b][-1:]).strip()[-1:].isdigit():
                    nxt[b] = self._number_continuation(pool, b, src_pos[b], nxt[b])
                    if nxt[b] not in stop:
                        src_pos[b] += 1
                        continue
                hit = pool["toks"][b] == nxt[b]             # memory position this token is copied from
                if not bool(hit.any()):
                    src_pos[b] = None
                    continue
                lab = la[b].masked_fill(~hit, float("-inf"))
                p_copy = float((torch.logsumexp(lab, -1) + lcopy[b]).exp())
                src_pos[b] = int(lab.argmax()) if p_copy >= 0.5 * float(p[b, nxt[b]]) else None
            if trace is not None:
                top = p.topk(3, dim=-1)
                src = (la + lcopy[:, None]).argmax(-1)
                for b in range(B):
                    if not done[b]:
                        reg = int(pool["region"][b, src[b]])
                        trace[base + b].append({
                            "tok": self.tok.decode([nxt[b]]), "p_gen": round(float(lgen[b].exp()), 3),
                            "top": [(self.tok.decode([int(t)]), round(float(v), 3))
                                    for t, v in zip(top.indices[b], top.values[b])],
                            "copy_from": "question" if reg >= 10_000 else reg - 1})
            for b in range(B):
                if done[b]:
                    continue
                if nxt[b] in stop:
                    done[b] = True
                else:
                    out[b].append(nxt[b])
            if all(done):
                break
        return [self.tok.decode(o) for o in out]


# ================================================================== tasks
def chain_of(q):
    """Typed gold steps. ("hop", item, span) | ("tool", name, args, [items]) | ("answer", item(s), text)."""
    if "steps" in q:
        return q["steps"]
    hops = q.get("hops") or [(q["gold"], q["answer"])]
    return [("hop", i, s) for i, s in hops[:-1]] + [("answer", hops[-1][0], hops[-1][1])]


class HubTask(LT.Task):
    def __init__(self, name, rn, stores, queries, n_cand=64, passes=4):
        for st in stores:                               # image features once (frozen encoder)
            todo = [it for it in st if kind_of(it) == "image" and it.feat is None]
            for s in range(0, len(todo), 32):
                fe = rn.encode_images([it.meta["image"] for it in todo[s:s + 32]])
                for it, f in zip(todo[s:s + 32], fe):
                    it.feat = f.half().cpu()
        super().__init__(name, rn, stores, queries, n_cand=n_cand, passes=passes)
        self.fpool = None
        if getattr(rn, "objects", False):
            self.fpool = [self._facet_pool(st) for st in stores]
        if getattr(rn.heads.obj, "numbers", False) and getattr(rn, "objects", False)                 and getattr(rn, "native_compare", True):
            for q in self.queries:
                self._native_compare(q)
        self.multi = True
        self.oracle = frozenset()
        self.mode_tools = True

    def _native_compare(self, q):
        """With native numbers, 'tool max/min over items, then answer' becomes one answer step whose
        retrieval gold is the whole compared set: the network compares, the calculator is not needed."""
        steps = q.get("steps")
        if not steps or len(steps) != 2 or steps[0][0] != "tool" or steps[1][0] != "answer":
            return
        tool, ans = steps
        op, items = (tool[4] if len(tool) > 4 else ""), tool[3]
        if op not in ("max", "min") or not items or min(items) < 0:
            return
        vals = [num_of(self.stores[q["store"]][g]) for g in items]
        if any(v is None for v in vals):
            return
        pick = items[(max if op == "max" else min)(range(len(items)), key=lambda j: vals[j])]
        q["steps"] = [("answer", pick if ans[1] == -1 else ans[1], ans[2], list(items))]

    @torch.no_grad()
    def refresh_keys(self):
        """Recompute store features with the current Walker (needed once Walker trains)."""
        super().refresh_keys()
        if self.fpool is not None:
            self.fpool = [self._facet_pool(st) for st in self.stores]

    @torch.no_grad()
    def _facet_pool(self, store):
        """Pooled Walker features of each item's field and record text (deduplicated); zero where empty."""
        out = []
        for key in ("field", "record"):
            strs = [facets_of(it)[key] for it in store]
            uniq = sorted({x for x in strs if x})
            pooled = self.rn.pooled(uniq) if uniq else None
            idx = {x: j for j, x in enumerate(uniq)}
            m = torch.zeros(len(store), self.rn.heads.D, dtype=torch.half, device=self.rn.dev)
            have = torch.zeros(len(store), dtype=torch.bool, device=self.rn.dev)
            for j, x in enumerate(strs):
                if x:
                    m[j] = pooled[idx[x]]
                    have[j] = True
            out.append((m, have))
        return out

    def _store_parts(self, sid, xs):
        ob = self.rn.heads.obj
        (fm, fh), (gm, gh) = self.fpool[sid]
        ix = torch.tensor(xs, device=self.rn.dev, dtype=torch.long)
        return ob.parts(fm[ix].float(), gm[ix].float(), self.pooled[sid][ix].float(), fh[ix].float(), gh[ix].float())

    def _what_of(self, sid, its):
        """Detached WHAT (d) of argument items: store items via their facets, tool results via op + args."""
        ob = self.rn.heads.obj
        out = []
        for it in its:
            m = getattr(it, "meta", None) or {}
            if "op_i" in m:
                out.append(ob.tool_parts(m["op_i"], m["what_args"]).sum(0))
            elif "sx" in m:
                out.append(self._store_parts(sid, [m["sx"]]).sum(1)[0])
        return torch.stack(out).mean(0).detach() if out else torch.zeros(ob.op.weight.shape[1], device=self.rn.dev)

    def _mark_tool(self, sid, item, op, arg_items, r):
        item.meta.update(op_i=CALC_OPS.index(op) if op in CALC_OPS else len(CALC_OPS),
                         what_args=self._what_of(sid, arg_items), made=r)

    def _objects(self, sid, its, n_ext, pre, carried, top, lg, q, s_now, prov_i, r):
        """Latent object tags for one row's items (tool results first, then store items)."""
        rn, ob = self.rn, self.rn.heads.obj
        dev = rn.dev
        if not its:
            return its
        rows = [ob.tool_parts(it.meta["op_i"], it.meta["what_args"]) if "op_i" in it.meta
                else torch.zeros(3, ob.op.weight.shape[1], device=dev) for it in its[:n_ext]]
        parts = torch.cat(([torch.stack(rows)] if rows else []) + ([self._store_parts(sid, pre)] if pre else []))
        s = torch.stack([s_now] * n_ext + [prov_i[x][1] for x in pre])
        pos = {x: m for m, x in enumerate(top)}
        logit = torch.tensor([0.0] * n_ext + [float(lg[pos[x]]) * 0.1 if x in pos else 0.0 for x in pre], device=dev)
        rank = torch.tensor([15] * n_ext + [pos.get(x, 15) for x in pre], device=dev)
        age = torch.tensor([r - it.meta.get("made", r) for it in its[:n_ext]] + [r - prov_i[x][0] for x in pre],
                           device=dev)
        (_, fh), (_, gh) = self.fpool[sid]
        flags = torch.zeros(len(its), len(ob.FLAGS), device=dev)
        flags[:n_ext, 1] = 1.0
        for m, x in enumerate(pre):
            flags[n_ext + m] = torch.tensor([float(x in carried), 0.0, float(kind_of(its[n_ext + m]) == "image"),
                                             float(fh[x]), float(gh[x]), float(prov_i[x][0] == r)], device=dev)
        tags, _ = ob(parts, q.float(), s.float(), logit, rank, age, flags, rn.ref(),
                     num=number_features(its) if ob.numbers else None)
        return ([tagged(it, tags[m]) for m, it in enumerate(its[:n_ext])]
                + [tagged(it, tags[n_ext + m], sx=x) for m, (it, x) in enumerate(zip(its[n_ext:], pre))])

    def retrieve(self, sid, text, qv, gold, k, train, st, fq=None):
        """Hybrid retrieval with one or several gold items (a set: CE = -log sum p(gold)). With memory
        objects on, the score adds a FIELD match and a RECORD match between question and item."""
        rn = self.rn
        golds = [] if gold is None else (list(gold) if isinstance(gold, (list, tuple)) else [gold])
        n_items = len(self.stores[sid])
        if n_items <= self.n_cand:
            lex = self.bm25[sid].scores(text)
            cand = list(range(n_items))
        else:
            lex = dict(self.bm25[sid].topk(text, self.n_cand))
            cand = list(lex) or list(range(min(n_items, self.n_cand)))
        if golds:
            st["cand_recall"].append(float(all(g in cand for g in golds)))
            if train:
                cand += [g for g in golds if g not in cand]
        ci = torch.tensor(cand, device=rn.dev)
        lx = torch.tensor([lex.get(c, 0.0) for c in cand], device=rn.dev)
        lx = lx / lx.max().clamp(min=1e-6)
        logit = (rn.keys(self.pooled[sid][ci]) @ qv) * rn.heads.tau.exp() + rn.heads.lex * 10.0 * lx
        if self.fpool is not None and fq is not None:
            hd = rn.heads
            for fi, (w, (m, have), kproj, q) in enumerate(zip(hd.fw, self.fpool[sid], (hd.fk_field, hd.fk_rec), fq)):
                kf = F.normalize(hd.facet_snap(fi, kproj(m[ci].float())), dim=-1)
                logit = logit + w * hd.tau.exp() * (kf @ q) * have[ci].float()
        ce = None
        gi = [cand.index(g) for g in golds if g in cand]
        if gi:
            ce = torch.logsumexp(logit, 0) - torch.logsumexp(logit[torch.tensor(gi, device=rn.dev)], 0)
        kk = max(k, len(golds))
        tj = logit.topk(min(kk, len(cand))).indices.tolist()
        top = [cand[j] for j in tj]
        if golds:
            st["r@1"].append(float(top[0] in golds))
            st[f"r@{k}"].append(float(all(g in top for g in golds)))
            if train:
                miss = [cand.index(g) for g in golds if g not in top]
                keep = [j for j in tj if cand[j] in golds]
                rest = [j for j in tj if cand[j] not in golds]
                tj = keep + rest[:max(0, len(tj) - len(keep) - len(miss))] + miss
                top = [cand[j] for j in tj]
        return ce, top, logit[tj]

    def run(self, qs, k, train, passes=None):
        """LOTUS read with actions. Each pass: retrieve -> read (carried evidence, tool results, new
        items, routed skill tokens) -> ACTION: requery (copy a span, re-query with it) | tool (generate
        the call, execute exactly, read the result next pass) | answer (copy-or-generate, stop)."""
        rn, hd = self.rn, self.rn.heads
        R = passes or self.passes
        n = len(qs)
        acts = hd.actions()
        chains = [chain_of(q) for q in qs]
        text = [q["q"] for q in qs]
        carry = [[] for _ in qs]
        extra = [[] for _ in qs]
        seen = [[] for _ in qs]
        used = [set() for _ in qs]                    # store items consumed as arguments by earlier calls
        h_prev, lat, lat_raw, z_prev, final = {}, {}, {}, {}, {}
        losses, st = [], collections.defaultdict(list)
        qp_all = rn.pooled([q["q"] for q in qs])
        skill_idx = hd.skills.route(qp_all, train) if hd.skills is not None else [-1] * n
        live = list(range(n))
        trail = [[] for _ in qs]                      # eval: what each pass did (diagnosis)
        tok_tr = {}                                   # eval trace: per-token decode record of the final answer
        prov = [{} for _ in qs]                       # store id -> (entry pass, pooled search text) for objects
        for b in hd.bottlenecks().values():
            b.pending.clear()
        for r in range(R):
            if train:
                live = [i for i in live if r < len(chains[i])]
            if not live:
                break
            qpool = rn.pooled([text[i] for i in live])
            qv = rn.queries(qpool, torch.stack([lat[i] for i in live]) if r else None)
            fq = None
            if getattr(rn, "objects", False):
                fq = (F.normalize(hd.facet_snap(0, hd.fq_field(qpool.float())), dim=-1),
                      F.normalize(hd.facet_snap(1, hd.fq_rec(qpool.float())), dim=-1))
            items, gold_span, logits, pres, n_c, a_tok, tgt, g_act = [], [], [], [], [], [], [], []
            q_max = max(len(rn.tok.ids(f"Q: {qs[i]['q']} [found: {', '.join(seen[i])}]\n")) for i in live)
            sk = hd.skills.items(qp_all[live], [skill_idx[i] for i in live], rn.ref()) if hd.skills is not None \
                else [None] * len(live)
            for j, i in enumerate(live):
                step = chains[i][r] if r < len(chains[i]) else None
                g_items = None
                if step is not None:
                    g_items = (step[3] if step[0] == "answer" and len(step) > 3 else
                               step[1] if step[0] in ("hop", "answer") else
                               step[5] if len(step) > 5 else [g for g in step[3] if g >= 0])
                    if isinstance(g_items, int) and g_items < 0 or g_items == []:
                        g_items = None
                ce, top, lg = self.retrieve(qs[i]["store"], text[i], qv[j], g_items, k,
                                            train or "retrieve" in self.oracle, st,
                                            fq=(fq[0][j], fq[1][j]) if fq is not None else None)
                if ce is not None:
                    losses.append(ce / n)
                new = [x for x, t in enumerate(top) if t not in carry[i]]
                pre = carry[i] + [top[x] for x in new]
                its = extra[i] + [self.stores[qs[i]["store"]][x] for x in pre]
                if getattr(rn, "objects", False):
                    for x in pre:
                        prov[i].setdefault(x, (r, qpool[j].detach()))
                    its = self._objects(qs[i]["store"], its, len(extra[i]), pre, carry[i], top, lg.detach(),
                                        qp_all[i], qpool[j].detach(), prov[i], r)
                mx = lg.max().detach() + hd.carry
                lgs = [mx.expand(len(extra[i]) + len(carry[i])), lg[new]]
                if sk[j] is not None:
                    its = its + [sk[j]]
                    lgs.append(mx.expand(1))
                lg_row = torch.cat(lgs)
                # context budget: drop the lowest-ranked NEW items until question + prefix + answer fit
                budget = getattr(rn.target, "max_t", 4096) - q_max - 80
                lens = [len(rn._segs(it)[0]) for it in its]
                keep_g = set(g_items if isinstance(g_items, list) else [g_items])
                while sum(lens) > budget:
                    drop = [x for x in range(len(carry[i]), len(pre)) if pre[x] not in keep_g]
                    if len(pre) - len(carry[i]) <= 1 or not drop:
                        break
                    x = drop[-1]
                    cut = len(extra[i]) + x
                    pre.pop(x)
                    del its[cut], lens[cut]
                    lg_row = torch.cat([lg_row[:cut], lg_row[cut + 1:]])
                n_car = len(carry[i])
                while sum(lens) > budget and n_car > 1:       # deep chains: drop the OLDEST carried evidence
                    cut = len(extra[i])
                    pre.pop(0)
                    n_car -= 1
                    del its[cut], lens[cut]
                    lg_row = torch.cat([lg_row[:cut], lg_row[cut + 1:]])
                for x in reversed(range(len(its))):           # still over: shed object tags, last items first
                    if sum(lens) <= budget:
                        break
                    if rn.n_obj(its[x]):
                        its[x] = tagged(its[x], None)
                        lens[x] = len(rn._segs(its[x])[0])
                while sum(lens) > budget and len(pre) - n_car > 1:   # last resort: drop even gold, keep one new
                    cut = len(extra[i]) + len(pre) - 1
                    pre.pop()
                    del its[cut], lens[cut]
                    lg_row = torch.cat([lg_row[:cut], lg_row[cut + 1:]])
                logits.append(lg_row)
                pres.append(pre)
                items.append(its)
                n_c.append(len(extra[i]) + n_car)
                gi = -1
                if step is not None and step[0] == "hop" and step[1] in pre:
                    gi = len(extra[i]) + pre.index(step[1])
                gold_span.append((gi, step[2] if step is not None and step[0] == "hop" else None))
                if train and step is not None and step[0] == "tool":
                    a_tok.append([])
                    tgt.append([])
                elif train and step is not None and step[0] == "answer":
                    a_tok.append([])
                    tgt.append(rn.tok.ids(" " + step[2]) + [rn.tok.nl])
                else:
                    a_tok.append([])
                    tgt.append([])
                g_act.append(None if step is None else
                             acts.index("requery" if step[0] == "hop" else step[1] if step[0] == "tool" else "answer"))
            asks = [qs[i]["q"] + (f" [found: {', '.join(seen[i])}]" if seen[i] else "") for i in live]
            pp = rn.prep(items, asks)
            fw = rn.forward_rows(pp, [a + t for a, t in zip(a_tok, tgt)] if train else None)
            ce, preds, pred_item, found = rn.span(pp, fw, gold_span, item_logits=logits,
                                                  ban=[seen[i] for i in live], n_carry=n_c)
            if ce is not None:
                losses.append(ce * found / n)
            if train:
                gl, accs = rn.gen_loss(pp, fw, a_tok, tgt, item_logits=logits)
                if gl is not None:
                    losses.append(gl * len(accs) / n)
                    st["tok_acc"] += list(accs.values())
                    if hd.skills is not None:
                        hd.skills.feedback([skill_idx[live[j]] for j in accs], list(accs.values()),
                                           freeze_at=getattr(self, "freeze_at", None))
            hq = fw["hq"]
            prev = torch.stack([h_prev.get(i, hq[j]) for j, i in enumerate(live)])
            lat_r = hq + hd.beta * (hq - prev)
            alog = hd.action_logits(lat_r)
            if train:
                ga = torch.tensor([a for a in g_act], device=rn.dev)
                losses.append(F.cross_entropy(alog, ga) * len(live) / n)
                st["act_acc"] += (alog.argmax(-1) == ga).float().tolist()
            z = F.normalize(hd.wz(torch.cat([lat_r, hq - prev], -1)), dim=-1)
            if train and r > 0:
                rn.walk.add(torch.stack([z_prev[i] for i in live]), lat_r - torch.stack([lat_raw[i] for i in live]))
            push, _ = rn.walk.read(z, hd.wtau, hd.D)
            g = torch.tanh(hd.wgate(lat_r)).squeeze(-1) * self.walk_mode
            lat_q = lat_r + (g * hd.wscale)[:, None] * lat_r.norm(dim=-1, keepdim=True) * push
            choice = alog.argmax(-1).tolist()
            calls = {}                                       # row j -> (op, [prefix item index])
            want = [j for j, i in enumerate(live)
                    if (train and chains[i][r][0] == "tool") or (not train and acts[choice[j]] in hd.tools)]
            if want:
                ist = rn.item_states(pp, fw)
                for j in want:
                    i = live[j]
                    tname = chains[i][r][1] if train else acts[choice[j]]
                    tool = hd.tools[tname]
                    ok = torch.tensor([kind_of(it) == "text" for it in items[j]], device=rn.dev)
                    is_x = torch.arange(len(items[j]), device=rn.dev) < len(extra[i])
                    uv = torch.tensor([False] * len(extra[i]) + [x in used[i] for x in pres[j]]
                                      + [False] * (len(items[j]) - len(extra[i]) - len(pres[j])), device=rn.dev)
                    sc = hd.arg_sel(ist[j], hq[j], is_x, uv)
                    sc = sc.masked_fill(~ok, -30.0)
                    if train:
                        step = chains[i][r]
                        gold = torch.zeros(len(items[j]), device=rn.dev)
                        for g in step[3]:
                            if g < 0 and -g - 1 < len(extra[i]):
                                gold[-g - 1] = 1.0
                            elif g in pres[j]:
                                gold[len(extra[i]) + pres[j].index(g)] = 1.0
                        w = gold                            # teacher forcing: the op reads the gold arguments
                    else:
                        w = torch.sigmoid(sc) * ok.float()
                    arg_ctx = (w[:, None] * ist[j].float()).sum(0) / w.sum().clamp(min=1e-6)
                    olog = hd.op[tname](lat_r[j]) + hd.op_arg[tname](arg_ctx)
                    if train:
                        losses.append(F.binary_cross_entropy_with_logits(sc, gold) / n)
                        oi = torch.tensor(tool.ops.index(step[4]), device=rn.dev)
                        losses.append(F.cross_entropy(olog[None], oi[None]) / n)
                        st["arg_acc"].append(float(((sc > 0).float() == gold).all()))
                        st["op_acc"].append(float(int(olog.argmax()) == int(oi)))
                    else:
                        op = tool.ops[int(olog.argmax())]
                        # calculator type rules: arguments are numbers; absdiff takes exactly two
                        order = [k for k in sc.argsort(descending=True).tolist()
                                 if bool(ok[k]) and num_of(items[j][k]) is not None]
                        sel = order[:2] if op == "absdiff" else ([k for k in order if float(sc[k]) > 0] or order[:1])
                        sel = sel or [int(sc.argmax())]
                        calls[j] = (op, sel)
            if not train:
                # decode tool calls and answers (forced answer on the last pass)
                for j in range(len(live)):
                    if r == R - 1 and acts[choice[j]] != "answer" and acts[choice[j]] not in hd.tools:
                        choice[j] = 0
                dec_rows = [j for j in range(len(live)) if acts[choice[j]] == "answer"]
                if dec_rows:
                    sub = {"rows": [pp["rows"][j] for j in dec_rows], "q_ids": [pp["q_ids"][j] for j in dec_rows],
                           "questions": [pp["questions"][j] for j in dec_rows], "items": None}
                    dtr = [] if isinstance(getattr(self, "trace", None), list) else None
                    outs = rn.decode(sub, [[] for _ in dec_rows], item_logits=[logits[j] for j in dec_rows],
                                     trace=dtr)
                    dec = dict(zip(dec_rows, outs))
                    if dtr is not None:
                        for jj, j in enumerate(dec_rows):
                            tok_tr[live[j]] = dtr[jj]
                else:
                    dec = {}
            nxt = []
            for j, i in enumerate(live):
                h_prev[i], lat[i], lat_raw[i], z_prev[i] = hq[j], lat_q[j], lat_r[j].detach(), z[j]
                if train:
                    step = chains[i][r]
                    if step[0] == "hop":
                        text[i] = f"{qs[i]['q']} {step[2]}"
                        carry[i] = carry[i] + [step[1]]
                        seen[i] = seen[i] + [step[2]]
                    elif step[0] == "tool":
                        used[i].update(g for g in step[3] if g >= 0)
                        item, res = hd.tools[step[1]].run(step[2])
                        if getattr(rn, "objects", False):
                            args = [extra[i][-g - 1] if g < 0 else tagged(self.stores[qs[i]["store"]][g], None, sx=g)
                                    for g in step[3] if g >= 0 or -g - 1 < len(extra[i])]
                            self._mark_tool(qs[i]["store"], item, step[4] if len(step) > 4 else "", args, r + 1)
                        extra[i] = [item] + extra[i]
                        text[i] = f"{qs[i]['q']} {res}"
                    if r < len(chains[i]) - 1:
                        nxt.append(i)
                    continue
                a = acts[choice[j]]
                st["n_actions"].append(1.0)
                if a == "answer":
                    trail[i].append(f"ans:{dec[j]}")
                    final[i] = (dec[j], r + 1, items[j])
                elif a in hd.tools:
                    op, sel = calls[j]
                    item, res = hd.tools[a].call(op, [item_value(items[j][k]) for k in sel])
                    if getattr(rn, "objects", False):
                        self._mark_tool(qs[i]["store"], item, op, [items[j][k] for k in sel], r + 1)
                    trail[i].append(f"{op}({','.join(item_value(items[j][k]) for k in sel)})={res}")
                    st["tool_calls"].append(1.0)
                    used[i].update(pres[j][k - len(extra[i])] for k in sel
                                   if len(extra[i]) <= k < len(extra[i]) + len(pres[j]))
                    gold_tool = next((s for s in chains[i] if s[0] == "tool"), None)
                    if gold_tool is not None:
                        got = {pres[j][k - len(extra[i])] if k >= len(extra[i]) else -k - 1 for k in sel}
                        st["args_ok"].append(float(op == gold_tool[4] and got == set(gold_tool[3])))
                    if res != "error":                    # a failed call leaves no node to copy from
                        extra[i] = [item] + extra[i]
                        text[i] = f"{qs[i]['q']} {res}"
                    if r == R - 1:
                        final[i] = (res, r + 1, items[j])
                    else:
                        nxt.append(i)
                else:
                    pi = pred_item[j] - len(extra[i])
                    text[i] = f"{qs[i]['q']} {preds[j]}"
                    trail[i].append(f"rq:{preds[j]}")
                    if 0 <= pi < len(pres[j]):
                        carry[i] = carry[i] + [pres[j][pi]]
                    seen[i] = seen[i] + [preds[j]]
                    nxt.append(i)
            live = nxt
        if not train:
            for i, q in enumerate(qs):
                p, used, its = final.get(i, ("", R, []))
                hit = float(_norm(p) == _norm(q["answer"]))
                if isinstance(getattr(self, "trace", None), list):
                    self.trace.append({"q": q["q"], "gold": q["answer"], "pred": p, "passes": used,
                                       "trail": list(trail[i]), "read": [it.text[:80] for it in its],
                                       "tokens": tok_tr.get(i, [])})
                st["em"].append(hit)
                st["passes"].append(float(used))
                if "tag" in q:
                    st[f"em_{q['tag']}"].append(hit)
                tag = q.get("tag", "")
                if not hit and sum(m.startswith(f"[{tag}]") for m in self.misses) < 2 and len(self.misses) < 24:
                    self.misses.append(f"[{tag}] q={q['q']!r} gold={q['answer']!r} pred={p!r} passes={used} "
                                       f"trail={' | '.join(trail[i])} top1={its[0].text[:70] if its else None!r}")
        for name, b in hd.bottlenecks().items():
            if b.pending:
                losses.append(sum(b.pending) / len(b.pending))
                b.pending.clear()
            if b.last is not None:
                st[f"vq_ppl_{name}"].append(b.perplexity())
        return sum(losses) if losses else torch.zeros((), device=rn.dev, requires_grad=True), st


# ================================================================== benches
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
_SYL = LT._SYL


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
        return f"{LT._name(rng, 2).lower()}@{LT._name(rng, 2).lower()}.{rng.choice(['com', 'io', 'net', 'org'])}"
    if kind == "code":
        return f"{rng.choice('ABCDEFGHJKLMNPRSTUVWXZ')}{rng.randint(10, 999)}-{rng.choice('ABCDEFGHJKLMNPRSTUVWXZ')}"
    if kind == "url":
        return f"https://{LT._name(rng, 2).lower()}.com/{LT._name(rng, 1).lower()}"
    if kind == "name":
        return f"{LT._name(rng, 2)} {LT._name(rng, 2)}"
    if kind == "long":
        return " ".join(LT._name(rng, rng.randint(1, 2)).lower() for _ in range(rng.randint(28, 50))) + "."
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
                        r_["name"] = LT._name(rng, 2)
                    elif idk == "title":
                        r_["title"] = LT._name(rng, 2) + " " + LT._name(rng, 1)
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
_T_HIST = {"train": ("What was the {key} of {where} in {doc} before update {k}?",
                     "In {doc}, what {key} did {where} have before update {k}?"),
           "eval": ("Before update {k}, what did {doc} have as the {key} of {where}?",)}
_T_COUNT = {"train": ("How many {lst} are there in {doc}?", "What is the number of {lst} in {doc}?"), "eval": ("Count the {lst} in {doc}.",)}
_T_SUM = {"train": ("What is the total {key} of all {lst} in {doc}?", "Sum the {key} of every {lst} entry in {doc}.",
                    "What do the {key} values of the {lst} in {doc} add up to?"),
          "eval": ("Add up the {key} over every entry of {lst} in {doc}.",)}
_T_MAX = {"train": ("What is the highest {key} among the {lst} in {doc}?", "Find the maximum {key} over the {lst} in {doc}.",
                    "Among all {lst} in {doc}, what is the top {key}?"),
          "eval": ("Which value is the largest {key} across {lst} in {doc}?",)}
_T_MIN = {"train": ("What is the lowest {key} among the {lst} in {doc}?", "Find the minimum {key} over the {lst} in {doc}.",
                    "Among all {lst} in {doc}, what is the least {key}?"),
          "eval": ("Which value is the smallest {key} across {lst} in {doc}?",)}
_T_DIFF = {"train": ("How much larger is the {key} of {a} than that of {b} in {doc}?",
                     "What is the {key} of {a} minus the {key} of {b} in {doc}?",
                     "Compute how far {a} is above {b} in {key} ({doc})."),
           "eval": ("By how much does {a} exceed {b} in {key} ({doc})?",)}


# Training-only paraphrase grammar: question = (doc placement) x (question form) x (phrase for the
# operation) x synonyms, so the act / op heads learn the operation's meaning, not a few frames. The
# eval frames and eval synonyms above are never produced (they stay the held-out wording test).
_P_SYN = {"max": ("highest", "largest", "biggest", "top", "peak"), "min": ("lowest", "smallest", "least", "bottom"),
          "tot": ("total", "combined", "summed", "full", "entire")}
_P_FORM = ("What is {np}?", "Tell me {np}.", "Give {np}.", "I need {np}.", "What's {np}?", "Can you work out {np}?",
           "Please report {np}.", "Find {np}.")
_P_NP = {
    "sum": ("the {tot} {key} of the {lst}", "the {key} of all {lst} put together", "the sum of the {key} values of the {lst}",
            "the {key} summed over the {lst}", "the {tot} {key} for every {lst} entry"),
    "max": ("the {ext} {key} among the {lst}", "the {ext} {key} of any {lst} entry", "the {key} that is {ext} in the {lst}"),
    "min": ("the {ext} {key} among the {lst}", "the {ext} {key} of any {lst} entry", "the {key} that is {ext} in the {lst}"),
    "diff": ("the gap in {key} between {a} and {b}", "the {key} difference between {a} and {b}",
             "how much more {key} {a} has than {b}", "{a}'s {key} minus {b}'s"),
    "argmax": ("which {lst} entry has the {ext} {key}", "the name of the {lst} entry with the {ext} {key}",
               "who among the {lst} has the {ext} {key}", "the {lst} entry whose {key} is the {ext}"),
    "sumdiff": ("how much bigger the {tot} {key} of {l1} is than that of {l2}",
                "the gap between the {tot} {key} of {l1} and the {tot} {key} of {l2}",
                "the difference in {tot} {key} between {l1} and {l2}"),
    "range": ("how much larger the {mx} {key} of the {lst} is than the {mn} one",
              "the spread of {key} among the {lst} ({mx} minus {mn})",
              "the difference between the {mx} and the {mn} {key} in the {lst}"),
    "filter": ("the {tot} {key} of the {lst} with {cat} {val}", "the sum of {key} for {lst} entries whose {cat} is {val}",
               "the {key} of the {lst} that have {cat} {val}, added together")}
_P_VERB = {
    "sum": ("How much {key} do all the {lst} have together?", "If you add every {lst} {key} together, what do you get?",
            "How much {key} do the {lst} come to in total?"),
    "max": ("Which {key} is the {ext} for the {lst}?", "What {key} does the {ext} {lst} entry have?"),
    "min": ("Which {key} is the {ext} for the {lst}?", "What {key} does the {ext} {lst} entry have?"),
    "diff": ("How much more {key} does {a} have than {b}?", "How far ahead of {b} is {a} on {key}?"),
    "argmax": ("Which {lst} entry comes out {ext} on {key}?", "Who has the {ext} {key} among the {lst}?"),
    "sumdiff": ("How much more {key} do the {l1} have in total than the {l2}?",),
    "range": ("How much does the {mx} {key} of the {lst} beat the {mn}?",),
    "filter": ("How much {key} do the {lst} with {cat} {val} have in total?",)}


def _para(kind, rng, doc, **f):
    """One training paraphrase of an operation question (see _P_NP / _P_VERB)."""
    f = dict(f)
    f.setdefault("tot", rng.choice(_P_SYN["tot"]))
    f.setdefault("mx", rng.choice(_P_SYN["max"]))
    f.setdefault("mn", rng.choice(_P_SYN["min"]))
    if f.get("ext") in ("max", "min"):
        f["ext"] = rng.choice(_P_SYN[f["ext"]])
    if rng.random() < 0.35:
        core = rng.choice(_P_VERB[kind]).format(**f)
    else:
        np_ = rng.choice(_P_NP[kind]).format(**f)
        forms = (("Tell me {np}.", "Can you work out {np}?", "I want to know {np}.")
                 if np_.split(" ", 1)[0] in ("how", "which", "who") else _P_FORM)
        core = rng.choice(forms).format(np=np_)
    where = rng.randrange(4)
    if where == 0:
        return f"In {doc}, {core[0].lower()}{core[1:]}"
    if where == 1:
        return f"{doc}: {core}"
    if where == 2:
        return f"Looking at {doc}, {core[0].lower()}{core[1:]}"
    return f"{core[:-1]} ({doc}){core[-1]}"


def _doc_name(rng):
    return f"{LT._name(rng, 2).lower()}_{rng.randint(1, 99)}.json"


def build_anyjson(tok, n_eps, rng, split="train", docs_per_store=2, tools=True, global_store=False):
    """Random-schema JSON stores (+ updates as history) with lookup / history / count questions, and
    tool questions (sum / max / min / difference over list records) whose answer must be computed."""
    vocab = KEYS_TRAIN if split == "train" else KEYS_EVAL
    tset = "train" if split == "train" else "eval"
    stores, queries = [], []
    gstore = []
    for _ in range(n_eps):
        store = gstore if global_store else []
        sid = 0 if global_store else len(stores)
        for _d in range(docs_per_store):
            doc = _doc_name(rng)
            obj = random_doc(rng, vocab, depth=rng.randint(1, 3))
            leaves = [lf for lf in flatten(obj) if lf.image is None]
            # updates: rewrite 1-2 scalar leaves with a new value of the same kind
            import copy
            new = copy.deepcopy(obj)
            changed = []
            for kk in range(1, rng.randint(1, 2) + 1):
                cands = [lf for lf in flatten(new) if lf.path[-1:] != ("count",) and lf.raw is not None
                         and not isinstance(lf.raw, bool) and len(lf.value) < 40 and lf.path
                         and not (isinstance(lf.path[-1], str) and words(lf.path[-1]).split()[-1] in _ID_WORDS)]
                if not cands:
                    break
                lf = rng.choice(cands)
                kind = "int" if isinstance(lf.raw, int) else "float" if isinstance(lf.raw, float) else "phrase"
                nv = _value(rng, kind)
                if scalar(nv) == lf.value:
                    continue
                ref = new
                for p in lf.path[:-1]:
                    ref = ref[p]
                old_obj = copy.deepcopy(new)
                ref[lf.path[-1]] = nv
                changed.append((lf, kk, old_obj, copy.deepcopy(new)))
            base = len(store)
            cur = ingest(new, doc)
            store += cur
            hist = []
            for lf, kk, o_, n_ in changed:
                hs = diff_items(o_, n_, kk, doc)
                hist.append((len(store), lf, kk, hs))
                store += hs
            by_path = {}
            for j, it in enumerate(cur):
                if it.meta.get("part", 0) == 0:
                    by_path.setdefault(it.meta["path"], base + j)
            leaves = [lf for lf in flatten(new) if lf.image is None]
            # lookups
            look = [lf for lf in leaves if lf.path in by_path and len(lf.value.split()) <= 12 and lf.path]
            for lf in rng.sample(look, min(3, len(look))):
                key, where = _subject(lf, doc)
                if lf.path[-1] == "count":
                    lst = _path_words(lf.path[:-1])
                    q = rng.choice(_T_COUNT[tset]).format(lst=lst or "items", doc=doc)
                    tag = "count"
                elif where:
                    q = rng.choice(_T_LOOK[tset]).format(key=key, where=where, doc=doc)
                    tag = "lookup"
                else:
                    q = rng.choice(_T_TOP[tset]).format(key=key, doc=doc)
                    tag = "lookup"
                queries.append({"store": sid, "gold": by_path[lf.path], "q": q, "answer": lf.value, "tag": tag})
            for pos, lf, kk, hs in hist:
                own = [j for j, h in enumerate(hs) if h.meta["path"] == lf.path]
                if not own:
                    continue
                key, where = _subject(lf, doc)
                q = rng.choice(_T_HIST[tset]).format(key=key, where=where or "the document", doc=doc, k=kk)
                queries.append({"store": sid, "gold": pos + own[0], "q": q, "answer": lf.value, "tag": "history"})
            if not tools:
                continue
            # tool questions over a list of records with a numeric field
            groups = collections.defaultdict(list)
            for lf in leaves:
                if (isinstance(lf.raw, (int, float)) and not isinstance(lf.raw, bool) and len(lf.path) >= 3
                        and isinstance(lf.path[-2], int) and lf.path in by_path):
                    groups[(lf.path[:-2], lf.path[-1])].append(lf)
            for (lpath, fkey), lfs in rng.sample(sorted(groups.items(), key=lambda x: str(x[0])),
                                                 min(2, len(groups))):
                if len(lfs) < 2:
                    continue
                lst, key = _path_words(lpath), words(fkey)
                vals = [lf.value for lf in lfs]
                gi = [by_path[lf.path] for lf in lfs]
                op = rng.choice(("sum", "max", "min", "absdiff") if all(lf.label_value for lf in lfs[:2])
                                else ("sum", "max", "min"))
                if op == "absdiff":
                    a, b = lfs[0], lfs[1]
                    if float(a.value) < float(b.value):
                        a, b = b, a
                    args = f"{a.value} - {b.value}"
                    q = (_para("diff", rng, doc, key=key, a=a.label_value, b=b.label_value)
                         if tset == "train" and rng.random() < PARA_P else
                         rng.choice(_T_DIFF[tset]).format(key=key, a=a.label_value, b=b.label_value, doc=doc))
                    gi = [by_path[a.path], by_path[b.path]]
                else:
                    args = f"{op}( {', '.join(vals)} )"
                    q = (_para(op, rng, doc, key=key, lst=lst, ext=op)
                         if tset == "train" and rng.random() < PARA_P else
                         rng.choice({"sum": _T_SUM, "max": _T_MAX, "min": _T_MIN}[op][tset]).format(
                             key=key, lst=lst, doc=doc))
                res = calc(args)
                if res == "error":
                    continue
                queries.append({"store": sid, "gold": gi[0], "q": q, "answer": res, "tag": f"tool_{op}",
                                "steps": [("tool", "calc", args, gi, op), ("answer", -1, res)]})
        if not global_store:
            stores.append(store)
    if global_store:
        stores = [gstore]
    return stores, queries


# ------------------------------------------------------------------ reasoning bench
# Compositional phrasings: many training frames x synonym sets, and a DISJOINT set of frames and
# synonyms for eval, so the controllers must read meaning rather than memorise a wording.
_SYN = {"train": {"max": ("highest", "largest", "biggest", "top"), "min": ("lowest", "smallest", "least"),
                  "tot": ("total", "combined", "summed")},
        "eval": {"max": ("greatest", "maximum"), "min": ("minimum", "tiniest"), "tot": ("aggregate", "overall")}}
_T_ARGMAX = {"train": ("Which {lst} entry has the {ext} {num} in {doc}?",
                       "In {doc}, name the {lst} entry with the {ext} {num}.",
                       "Among the {lst} of {doc}, which one has the {ext} {num}?",
                       "Tell me which {lst} entry in {doc} has the {ext} {num}.",
                       "{doc}: the {lst} entry with the {ext} {num} is called what?"),
             "eval": ("Among the {lst} of {doc}, whose {num} is {ext}?",
                      "In {doc}, identify the {lst} record holding the {ext} {num}.")}
_T_SUMDIFF = {"train": ("How much larger is the {tot} {num} of {l1} than the {tot} {num} of {l2} in {doc}?",
                        "In {doc}, what is the difference between the {tot} {num} of {l1} and that of {l2}?",
                        "By what amount does the {tot} {num} of {l1} beat the {tot} {num} of {l2} ({doc})?",
                        "Compare the {tot} {num} of {l1} with that of {l2} in {doc}: how big is the gap?"),
              "eval": ("In {doc}, by how much does the {num} {tot} over {l1} exceed the {num} {tot} over {l2}?",
                       "{doc}: {l1} versus {l2}, how far apart are their {tot} {num} values?")}
_T_FILTER = {"train": ("What is the {tot} {num} of the {lst} whose {cat} is {val} in {doc}?",
                       "Add up the {num} of every {lst} entry with {cat} {val} ({doc}).",
                       "For {doc}, sum the {num} across {lst} entries where {cat} equals {val}.",
                       "Considering only {lst} with {cat} {val} in {doc}, what is their {tot} {num}?"),
             "eval": ("In {doc}, sum the {num} over the {lst} entries that have {val} as their {cat}.",
                      "Restricting {doc} to {lst} with {cat} = {val}, give the {tot} {num}.")}
_T_ORDER = {"train": ("Climb the {rel} order upward from {p}: who is at the very top?",
                      "Start from {p} and keep moving to whoever is {cmp} them. Who is last?",
                      "Who sits at the top of the {rel} chain that begins at {p}?",
                      "Following '{cmp}' links from {p}, which person do you finally reach?"),
            "eval": ("Starting at {p} and following who is {cmp} each time, who do you end at?",
                     "From {p}, repeatedly step to someone {cmp} them; where does it stop?")}
_REL = (("height", "taller than"), ("age", "older than"), ("speed", "faster than"))
_T_STORY = {"train": ("How many {thing} does {p} have at the end?",
                      "What is {p}'s final count of {thing}?",
                      "After these events, how many {thing} does {p} own?",
                      "Once everything has happened, how many {thing} are left with {p}?"),
            "eval": ("After all of this, what number of {thing} does {p} hold?",
                     "In the end, {p} possesses how many {thing}?")}
_THINGS = ("coins", "apples", "stamps", "marbles", "tickets")


def build_reason(tok, n_eps, rng, split="train"):
    """Multi-step reasoning over one mixed store (a JSON document + free-text notes). Each question
    needs several dependent steps (tools, re-queries, a halting decision):
      argmax_label -- max over a record list (tool), then re-query for WHOSE value it is and copy the name;
      sum_diff     -- two sums over two lists, then the difference of the two results (3 chained calls);
      filter_sum   -- sum over the records whose category matches (selector must join category and value
                      items of the same record);
      order_dN     -- transitive chain "X is taller than Y": climb from a person to the top, N hops; the
                      model must decide when nobody is above (train N=2-4, eval N=2-6: longer than seen);
      story_eN     -- a person's count changes over N events (gives / finds), chained tool calls (train
                      N=2, eval N=2-3).
    Eval uses the held-out key vocabulary and held-out phrasings."""
    vocab = KEYS_TRAIN if split == "train" else KEYS_EVAL
    tset = "train" if split == "train" else "eval"
    max_depth = 4 if split == "train" else 6
    max_events = 2 if split == "train" else 3
    stores, queries = [], []
    for _ in range(n_eps):
        store, sid, used, names = [], len(stores), set(), set()

        def person():
            return LT._name(rng, 2, names)
        doc = _doc_name(rng)
        num, cat = (_style(_key(rng, vocab, used), rng) for _ in range(2))
        lists = []
        for _l in range(2):
            lname = _style(_key(rng, vocab, used), rng)
            cvals = rng.sample([w for w in _KEYW if w not in used], 2)
            recs, seen_v = [], set()
            for _r in range(rng.randint(3, 4)):
                v = rng.randint(10, 999)
                while v in seen_v:
                    v = rng.randint(10, 999)
                seen_v.add(v)
                recs.append({"name": person(), num: v, cat: rng.choice(cvals)})
            if len({r[cat] for r in recs}) < 2:
                recs[0][cat] = cvals[0] if recs[0][cat] != cvals[0] else cvals[1]
            lists.append((lname, recs, cvals))
        obj = {ln: recs for ln, recs, _ in lists}
        items = ingest(obj, doc)
        by_path = {it.meta["path"]: j for j, it in enumerate(items) if it.meta.get("part", 0) == 0}
        store += items
        (l1, r1, cv1), (l2, r2, _cv2) = lists
        lw1, lw2, nw, cw = words(l1), words(l2), words(num), words(cat)
        g1 = [by_path[(l1, j, num)] for j in range(len(r1))]
        g2 = [by_path[(l2, j, num)] for j in range(len(r2))]
        v1 = [str(r[num]) for r in r1]
        v2 = [str(r[num]) for r in r2]
        syn = _SYN[tset]
        # argmax / argmin -> label (the op must come from the question's meaning)
        ext = rng.choice(("max", "min"))
        jm = (max if ext == "max" else min)(range(len(r1)), key=lambda j: r1[j][num])
        a1 = f"{ext}( {', '.join(v1)} )"
        queries.append({"store": sid, "gold": g1[jm], "answer": r1[jm]["name"], "tag": "argmax_label",
                        "q": (_para("argmax", rng, doc, lst=lw1, key=nw, ext=ext)
                              if tset == "train" and rng.random() < PARA_P else
                              rng.choice(_T_ARGMAX[tset]).format(lst=lw1, num=nw, doc=doc, ext=rng.choice(syn[ext]))),
                        "steps": [("tool", "calc", a1, g1, ext), ("answer", g1[jm], r1[jm]["name"])]})
        # nested: sum, sum, difference of the two results
        s1, s2 = calc(f"sum( {', '.join(v1)} )"), calc(f"sum( {', '.join(v2)} )")
        if float(s1) < float(s2):
            (l1, r1, g1, v1, s1, lw1), (l2, r2, g2, v2, s2, lw2) = (l2, r2, g2, v2, s2, lw2), (l1, r1, g1, v1, s1, lw1)
        d = calc(f"abs( {s1} - {s2} )")
        q_sd = (_para("sumdiff", rng, doc, key=nw, l1=lw1, l2=lw2) if tset == "train" and rng.random() < PARA_P
                else rng.choice(_T_SUMDIFF[tset]).format(num=nw, l1=lw1, l2=lw2, doc=doc, tot=rng.choice(syn["tot"])))
        queries.append({"store": sid, "gold": g1[0], "q": q_sd,
                        "answer": d, "tag": "sum_diff",
                        "steps": [("tool", "calc", f"sum( {', '.join(v1)} )", g1, "sum"),
                                  ("tool", "calc", f"sum( {', '.join(v2)} )", g2, "sum"),
                                  ("tool", "calc", f"abs( {s1} - {s2} )", [-1, -2], "absdiff"),
                                  ("answer", -1, d)]})
        # range (train only, with --paraphrase; eval stays comparable): max and min of one list, then their gap
        if split == "train" and PARA_P > 0:
            ln_r, recs_r, _ = lists[0]
            gr = [by_path[(ln_r, j, num)] for j in range(len(recs_r))]
            vr = [str(r[num]) for r in recs_r]
            mx_r, mn_r = calc(f"max( {', '.join(vr)} )"), calc(f"min( {', '.join(vr)} )")
            dr = calc(f"abs( {mx_r} - {mn_r} )")
            queries.append({"store": sid, "gold": gr[0], "answer": dr, "tag": "range_diff",
                            "q": _para("range", rng, doc, key=nw, lst=words(ln_r)),
                            "steps": [("tool", "calc", f"max( {', '.join(vr)} )", gr, "max"),
                                      ("tool", "calc", f"min( {', '.join(vr)} )", gr, "min"),
                                      ("tool", "calc", f"abs( {mx_r} - {mn_r} )", [-1, -2], "absdiff"),
                                      ("answer", -1, dr)]})
        # filter + aggregate (first list): retrieval must bring every category item, the selector joins
        ln, recs, cvals = lists[0]
        val = rng.choice(sorted({r[cat] for r in recs}))
        keep = [j for j, r in enumerate(recs) if r[cat] == val]
        gn = [by_path[(ln, j, num)] for j in keep]
        gc = [by_path[(ln, j, cat)] for j in range(len(recs))]
        fa = f"sum( {', '.join(str(recs[j][num]) for j in keep)} )"
        queries.append({"store": sid, "gold": gn[0], "answer": calc(fa), "tag": "filter_sum",
                        "q": (_para("filter", rng, doc, key=nw, lst=words(ln), cat=cw, val=val)
                              if tset == "train" and rng.random() < PARA_P else
                              rng.choice(_T_FILTER[tset]).format(num=nw, lst=words(ln), cat=cw, val=val, doc=doc,
                                                                 tot=rng.choice(syn["tot"]))),
                        "steps": [("tool", "calc", fa, gn, "sum", gc + gn), ("answer", -1, calc(fa))]})
        # transitive orderings (two relations; the asked one plus a distractor chain)
        for rel, cmp in rng.sample(_REL, 2):
            depth = rng.randint(2, max_depth)
            chain = [person() for _ in range(depth + 1)]           # chain[i+1] is <cmp> chain[i]
            at = []
            for i in range(depth):
                at.append(len(store))
                store.append(HubItem(f"note: {chain[i + 1]} is {cmp} {chain[i]}."))
            if rel == _REL[0][0] or rng.random() < 0.5:
                steps = [("hop", at[i], chain[i + 1]) for i in range(depth - 1)] + [("answer", at[-1], chain[-1])]
                queries.append({"store": sid, "gold": at[-1], "answer": chain[-1], "tag": f"order_d{depth}",
                                "q": rng.choice(_T_ORDER[tset]).format(rel=rel, cmp=cmp, p=chain[0]), "steps": steps})
        # story arithmetic (one asked person + one distractor person)
        for asked in (True, False):
            p, thing = person(), rng.choice(_THINGS)
            have = rng.randint(20, 60)
            h_at = len(store)
            store.append(HubItem(f"note: {p} starts with {have} {thing}."))
            steps, cur, prev = [], have, h_at
            for e in range(rng.randint(2, max_events)):
                if rng.random() < 0.5 and cur > 3:
                    x = rng.randint(1, cur - 1)
                    text, op, new = f"note: then {p} gives {x} {thing} to {person()}.", "absdiff", cur - x
                else:
                    x = rng.randint(1, 30)
                    text, op, new = f"note: then {p} finds {x} more {thing}.", "sum", cur + x
                e_at = len(store)
                store.append(HubItem(text))
                args = f"abs( {cur} - {x} )" if op == "absdiff" else f"sum( {cur}, {x} )"
                steps.append(("tool", "calc", args, [prev, e_at] if prev >= 0 else [-1, e_at], op))
                cur, prev = new, -1
            steps.append(("answer", -1, str(cur)))
            if asked:
                queries.append({"store": sid, "gold": h_at, "answer": str(cur), "tag": f"story_e{len(steps) - 1}",
                                "q": rng.choice(_T_STORY[tset]).format(thing=thing, p=p), "steps": steps})
        stores.append(store)
    return stores, queries


# ------------------------------------------------------------------ vision bench
_COLORS = {"red": (220, 40, 40), "green": (40, 180, 60), "blue": (50, 90, 230), "yellow": (240, 220, 40),
           "purple": (150, 60, 200), "orange": (245, 140, 30), "white": (245, 245, 245)}
_SHAPES = ("circle", "square", "triangle", "diamond")
HELD_OUT = {("purple", "triangle"), ("green", "diamond"), ("orange", "circle"), ("white", "square")}


def render_scene(objs, size=224):
    """objs: [(color, shape, column 0..2)] -> PIL image on dark gray."""
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (size, size), (60, 60, 64))
    d = ImageDraw.Draw(im)
    w = size // 3
    for color, shape, col in objs:
        cx, cy, r = col * w + w // 2, size // 2, int(w * 0.36)
        c = _COLORS[color]
        if shape == "circle":
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=c)
        elif shape == "square":
            d.rectangle([cx - r, cy - r, cx + r, cy + r], fill=c)
        elif shape == "triangle":
            d.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=c)
        else:
            d.polygon([(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)], fill=c)
    return im


def _scene(rng, split):
    """1-3 objects in distinct columns, distinct shapes. Train never uses a HELD_OUT color-shape
    combination; every eval scene contains at least one."""
    n = rng.randint(1, 3)
    cols = sorted(rng.sample(range(3), n))
    shapes = rng.sample(_SHAPES, n)
    objs = []
    for j, (c, sh) in enumerate(zip(cols, shapes)):
        ok = [col for col in _COLORS if (col, sh) not in HELD_OUT]
        if split != "train" and j == 0:
            held = [col for col in _COLORS if (col, sh) in HELD_OUT]
            ok = held or ok
        objs.append((rng.choice(ok), sh, c))
    if split != "train" and not any((o[0], o[1]) in HELD_OUT for o in objs):
        return _scene(rng, split)
    return objs


def build_vision(tok, n_eps, rng, split="train"):
    """Catalog JSON whose records carry a photo (rendered scene). Questions need the image: color of
    a shape, shape on the left/right, number of shapes, and a caption. Eval scenes contain HELD-OUT
    color-shape combinations. The image is found by retrieval like any other item."""
    stores, queries = [], []
    for _ in range(n_eps):
        doc = _doc_name(rng)
        recs, scenes = [], []
        for _r in range(rng.randint(2, 3)):
            objs = _scene(rng, split)
            nm = LT._name(rng, 2)
            recs.append({"name": nm, "price": round(rng.uniform(1, 300), 2), "photo": render_scene(objs)})
            scenes.append((nm, objs))
        obj = {"products": recs, "store": LT._name(rng, 2)}
        items = ingest(obj, doc)
        sid = len(stores)
        stores.append(items)
        img_at = {it.meta["leaf"].label_value: j for j, it in enumerate(items) if it.kind == "image"}
        for nm, objs in scenes:
            gi = img_at[nm]
            color, shape, _ = rng.choice(objs)
            cap = " and ".join(f"a {c} {s}" for c, s, _ in objs)
            qs = [("color", f"What color is the {shape} in the photo of {nm}?", color),
                  ("count", f"How many shapes are in the photo of {nm}?", str(len(objs))),
                  ("left", f"What shape is on the far left in the photo of {nm}?", objs[0][1]),
                  ("caption", f"Describe the photo of {nm}.", cap)]
            for tag, q, a in rng.sample(qs, 2):
                queries.append({"store": sid, "gold": gi, "q": q, "answer": a, "tag": tag})
    return stores, queries


# ------------------------------------------------------------------ continual skills (spawned modules)
_MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
           "November", "December")


def build_skill(tok, name, n, rng):
    """Memory-free transformation skills (one store with a single blank note)."""
    store = [HubItem("note: none")]
    qs = []
    for _ in range(n):
        if name == "iso":
            y, m, d = rng.randint(1950, 2030), rng.randint(1, 12), rng.randint(1, 28)
            q, a = f"Convert to an ISO date: {_MONTHS[m - 1]} {d}, {y}", f"{y}-{m:02d}-{d:02d}"
        elif name == "initials":
            p = [LT._name(rng, 2) for _ in range(rng.randint(2, 3))]
            q, a = f"Write the initials of {' '.join(p)}", " ".join(w[0] + "." for w in p)
        elif name == "swap":
            f_, l_ = LT._name(rng, 2), LT._name(rng, 2)
            q, a = f"Rewrite as last name, first name: {f_} {l_}", f"{l_}, {f_}"
        else:   # domain
            u, dm = LT._name(rng, 2).lower(), LT._name(rng, 2).lower() + rng.choice([".com", ".io", ".org"])
            q, a = f"What is the domain of the email {u}@{dm}?", dm
        qs.append({"store": 0, "gold": -1, "q": q, "answer": a, "tag": name, "steps": [("answer", -1, a)]})
    return [store], qs


# ================================================================== train / eval
def evaluate(tasks, k, step, bs=16, prefix="HUB_VAL"):
    with torch.no_grad():
        for t in tasks:
            agg = collections.defaultdict(list)
            t.eval_task.misses = []
            for s in range(0, len(t.eval_q), bs):
                _, st = t.eval_task.run(t.eval_q[s:s + bs], k, train=False)
                for kk, v in st.items():
                    agg[kk] += v
            print(f"{prefix} step={step} task={t.name} q={len(t.eval_q)} " +
                  " ".join(f"{kk}={sum(v)/len(v):.4f}" for kk, v in sorted(agg.items())), flush=True)
            for m in t.eval_task.misses:
                print(f"HUB_MISS step={step} task={t.name} {m}", flush=True)
            t.last_em = sum(agg["em"]) / max(1, len(agg["em"]))


def make_opt(heads):
    named = list(heads.named_parameters())
    groups = collections.OrderedDict((g, []) for g in ("ret", "ptr", "ctl", "scal", "reader", "gen", "vision",
                                                        "skills"))
    for n, p in named:
        top = n.split(".")[0]
        if top in ("q_ret", "k_ret") or n in ("tau", "lex"):
            groups["ret"].append(p)
        elif top in ("sq", "sk", "eq", "ek"):
            groups["ptr"].append(p)
        elif top in ("mgate", "qh", "halt", "wz", "wgate", "act", "tool_q", "tools", "op", "arg_sel",
                     "fq_field", "fk_field", "fq_rec", "fk_rec", "fw", "obj", "graph", "op_arg"):
            groups["ctl"].append(p)
        elif top in ("match", "match_top", "rbias", "beta", "wtau", "wscale", "carry", "cont", "crb", "qsrc", "qm",
                     "echo"):
            groups["scal"].append(p)
        elif top == "reader":
            groups["reader"].append(p)
        elif top in ("cq", "ck", "gate"):
            groups["gen"].append(p)
        elif top == "vision":
            groups["vision"].append(p)
        elif top == "skills":
            groups["skills"].append(p)
        else:
            raise KeyError(n)
    lrs = {"ret": 1e-3, "ptr": 1e-3, "ctl": 1e-3, "scal": 3e-2, "reader": 5e-4, "gen": 1e-3, "vision": 3e-4,
           "skills": 1e-3}
    opt = torch.optim.AdamW([{"params": ps, "lr": lrs[g], "name": g} for g, ps in groups.items() if ps],
                            weight_decay=0.0)
    in_opt = sorted(id(p) for gr in opt.param_groups for p in gr["params"])
    assert in_opt == sorted(id(p) for p in heads.parameters()), "every head param in exactly one group"
    return opt


def main():
    P = argparse.ArgumentParser()
    P.add_argument("--ckpt", required=True)
    P.add_argument("--layer", type=int, default=2)
    P.add_argument("--k", type=int, default=8)
    P.add_argument("--steps", type=int, default=1500)
    P.add_argument("--batch", type=int, default=8)
    P.add_argument("--eval-every", type=int, default=500)
    P.add_argument("--passes", type=int, default=4)
    P.add_argument("--tasks", default="anyjson,anyjson_global,tools,reason,vision,bridge,sentences")
    P.add_argument("--reason-passes", type=int, default=7, help="max passes for the reasoning bench (deep chains)")
    P.add_argument("--objects", action="store_true",
                   help="memory objects: facet (field / record) retrieval + latent query-relative object tags")
    P.add_argument("--discrete", default="",
                   help="G,L[,frac]: mixed float + discrete object tags and facet keys (G heads snapped to a shared "
                        "codebook of L codes over the first frac of each vector, default 0.5; DVNC). Empty = float")
    P.add_argument("--numbers", action="store_true",
                   help="with --objects: a VALUE tag per numeric item (float + discrete magnitude / rank) and "
                        "max/min questions answered by comparison instead of a calculator call")
    P.add_argument("--no-native-compare", action="store_true",
                   help="with --numbers: keep the VALUE features but answer max/min through the calculator (the op "
                        "head picks max or min explicitly) instead of one comparison answer step")
    P.add_argument("--paraphrase", type=float, default=0.0,
                   help="share of training tool / multi-step questions drawn from the paraphrase grammar (eval "
                        "wording unchanged); > 0 also adds the train-only range_diff type (max, min, gap)")
    P.add_argument("--graph", action="store_true",
                   help="latent graph controller (gmem_graph.py): one module per pass replaces the separate "
                        "tags / facet / argument / action / op heads; Walker reads only on answer passes")
    P.add_argument("--walker-train", default="none", choices=("none", "alt", "memonly"),
                   help="none = frozen Walker; alt = 2 episodes per step (memory episode: heads + Walker on the "
                        "memory loss; LM episode: Walker on next-token loss) with SEPARATE optimizers (LTM run 11); "
                        "memonly = memory episode only (control)")
    P.add_argument("--mem-walker-lr", type=float, default=1e-5)
    P.add_argument("--lm-lr", type=float, default=2e-5)
    P.add_argument("--lm-batch", type=int, default=8)
    P.add_argument("--lm-seq", type=int, default=512)
    P.add_argument("--image-encoder", default="google/siglip-base-patch16-224")
    P.add_argument("--skills", default="spawn", choices=("spawn", "shared", "off"))
    P.add_argument("--skill-k", type=float, default=4.0, help="novelty: z-score below a module's member cosine")
    P.add_argument("--skill-floor", type=float, default=0.01, help="minimum cosine spread per module")
    P.add_argument("--continual", default="", help="comma list of skills trained one after another, e.g. "
                                                   "iso,initials,swap,domain (after the main phase)")
    P.add_argument("--continual-steps", type=int, default=300)
    P.add_argument("--freeze-at", type=float, default=None, help="consolidate a module at this EMA token acc")
    P.add_argument("--load-heads", default="")
    P.add_argument("--out", default="gmem_hub")
    A = P.parse_args()
    global PARA_P
    PARA_P = A.paraphrase
    os.makedirs(A.out, exist_ok=True)
    model, raw_tok, _ = T.build_model_for_inference(A.ckpt)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    tok = LT.Tok(raw_tok)
    dev = next(model.parameters()).device
    D = model.emb.weight.shape[1]
    names = A.tasks.split(",") if A.tasks else []
    vision = enc = None
    if "vision" in names:
        enc, E = load_image_encoder(A.image_encoder, dev)
        vision = VisionModule(E, D)
    skills = SkillBank(D, mode=A.skills, k=A.skill_k, floor=A.skill_floor)
    disc = None
    if A.discrete:
        g_, l_, *f_ = A.discrete.split(",")
        disc = (int(g_), int(l_), float(f_[0]) if f_ else 0.5)
    heads = HubHeads(D, tools=[ToolModule("calc", calc, CALC_OPS)], vision=vision, skills=skills,
                     discrete=disc, numbers=A.numbers).to(dev)
    if A.load_heads:
        sd = torch.load(A.load_heads, map_location=dev)
        for _ in range(sd.get("n_skills", 0)):          # recreate spawned modules before loading them
            skills.mods.append(SkillModule(D, skills.m).to(dev))
            skills.ema_acc.append(0.0)
            skills.updates.append(0)
        n_sk = sd.get("n_skills", 0)
        skills.keys = torch.zeros(n_sk, D, device=dev)
        skills.sim_m, skills.sim_s = torch.zeros(n_sk, device=dev), torch.zeros(n_sk, device=dev)
        missing = heads.load_state_dict(sd["heads"], strict=False)
        print(f"HUB loaded heads {A.load_heads} missing={len(missing.missing_keys)}", flush=True)
    TaskCls = HubTask
    if A.graph:
        import gmem_graph
        heads.graph = gmem_graph.GraphController(D, len(heads.tools), len(CALC_OPS), discrete=disc).to(dev)
        TaskCls = gmem_graph.GraphTask
    rn = HubRunner(model, tok, heads, A.layer)
    rn.objects = A.objects
    rn.native_compare = not A.no_native_compare
    rn.encode_images = (lambda ims: enc(ims)) if enc is not None else None
    t0 = time.time()
    opt = make_opt(heads)
    base_lr = [g["lr"] for g in opt.param_groups]

    def on_spawn(mod):
        opt.add_param_group({"params": list(mod.parameters()), "lr": 1e-3, "name": "spawned"})
        base_lr.append(1e-3)
        print(f"HUB_SPAWN modules={len(skills.mods)} elapsed={time.time()-t0:.0f}s", flush=True)
    skills.on_spawn = on_spawn

    def pair(name, tr_args, ev_args, builder, n_eval=None, passes=None):
        tr = TaskCls(name, rn, *builder(*tr_args), passes=passes or A.passes)
        ev = TaskCls(name, rn, *builder(*ev_args), passes=passes or A.passes)
        tr.eval_task, tr.eval_q = ev, ev.queries[:n_eval] if n_eval else ev.queries
        tr.freeze_at = ev.freeze_at = A.freeze_at
        return tr

    with rn.active():
        tasks = []
        if "anyjson" in names:
            tasks.append(pair("anyjson", (tok, 500, random.Random(11), "train", 2, False),
                              (tok, 60, random.Random(12), "eval", 2, False), build_anyjson, 300))
        if "anyjson_global" in names:
            tr = TaskCls("anyjson_global", rn, *build_anyjson(tok, 300, random.Random(13), "train", 1, False, True),
                         passes=A.passes)
            ev = TaskCls("anyjson_global", rn, *build_anyjson(tok, 150, random.Random(14), "eval", 1, False, True),
                         passes=A.passes)
            tr.eval_task, tr.eval_q = ev, random.Random(0).sample(ev.queries, min(300, len(ev.queries)))
            tasks.append(tr)
        if "tools" in names:
            def tools_only(*a):
                st, qs = build_anyjson(*a)
                return st, [q for q in qs if q["tag"].startswith("tool_")]
            tasks.append(pair("tools", (tok, 2000, random.Random(15), "train", 2, True),
                              (tok, 150, random.Random(16), "eval", 2, True), tools_only, 300))
        if "reason" in names:
            tasks.append(pair("reason", (tok, 1500, random.Random(19), "train"),
                              (tok, 80, random.Random(20), "eval"), build_reason, 400, passes=A.reason_passes))
        if "vision" in names:
            tasks.append(pair("vision", (tok, 500, random.Random(17), "train"),
                              (tok, 80, random.Random(18), "eval"), build_vision, 300))
        if "bridge" in names:
            tasks.append(pair("bridge", (tok, 60, random.Random(8)), (tok, 10, random.Random(9)),
                              LT.build_bridge, 300))
        if "sentences" in names:
            tasks.append(pair("sentences", (tok, "train", 4000, 0), (tok, "test", 1500, 1),
                              LT.build_sentences, 300))
        for t in tasks:
            print(f"HUB task={t.name} train_store={sum(len(s) for s in t.stores)} train_q={len(t.queries)} "
                  f"eval_store={sum(len(s) for s in t.eval_task.stores)} eval_q={len(t.eval_q)} "
                  f"built={time.time()-t0:.0f}s", flush=True)
        r = random.Random(0)
        opt_wmem = opt_wlm = lm_train = lm_heldout = None
        if A.walker_train != "none" and tasks:
            wparams = list(model.parameters())
            for p in wparams:
                p.requires_grad_(True)
            # separate Adam states per episode type: memory gradients never enter the LM optimizer's moments
            opt_wmem = T.CompensatedAdamW([{"params": wparams, "lr": A.mem_walker_lr, "weight_decay": 0.0}],
                                          betas=(0.9, 0.95))
            if A.walker_train == "alt":
                opt_wlm = T.CompensatedAdamW([{"params": wparams, "lr": A.lm_lr, "weight_decay": 0.0}],
                                             betas=(0.9, 0.95))
                lm_train = LT.LMData(raw_tok, "train", 4_000_000, A.lm_seq, seed=0)
            lm_heldout = LT.LMData(raw_tok, "validation", 64 * A.lm_seq + 1, A.lm_seq)
            lm_base = LT.lm_val(model, lm_heldout)
            print(f"HUB_WALKER mode={A.walker_train} params={sum(p.numel() for p in wparams)/1e6:.0f}M "
                  f"mem_lr={A.mem_walker_lr} lm_lr={A.lm_lr}", flush=True)
            print(f"HUB_LM step=0 val_ce={lm_base:.4f}", flush=True)
        if tasks:
            evaluate(tasks, A.k, 0)
        acc = collections.defaultdict(list)
        for step in range(1, (A.steps if tasks else 0) + 1):
            decay = 0.1 + 0.45 * (1 + math.cos(math.pi * (step - 1) / A.steps))
            for g, lr in zip(opt.param_groups, base_lr):
                g["lr"] = lr * decay
            loss = 0.0
            for t in tasks:
                l, st = t.run(r.sample(t.queries, A.batch), A.k, train=True)
                loss = loss + l
                for kk, v in st.items():
                    acc[f"{t.name}/{kk}"] += v
            loss.backward()
            for g in opt.param_groups:
                torch.nn.utils.clip_grad_norm_(g["params"], 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            if opt_wmem is not None:                      # episode 1 (memory): Walker update from the memory loss
                for g in opt_wmem.param_groups:
                    g["lr"] = A.mem_walker_lr * decay
                torch.nn.utils.clip_grad_norm_(opt_wmem.param_groups[0]["params"], 1.0)
                opt_wmem.step()
                opt_wmem.zero_grad(set_to_none=True)
                LT.refresh_quant(model)
            if opt_wlm is not None:                       # episode 2 (LM): Walker update from next-token loss only
                for g in opt_wlm.param_groups:
                    g["lr"] = A.lm_lr * decay
                x, y = lm_train.batch(A.lm_batch, rn.dev)
                lce = LT.lm_ce(model, x, y)
                lce.backward()
                torch.nn.utils.clip_grad_norm_(opt_wlm.param_groups[0]["params"], 1.0)
                opt_wlm.step()
                opt_wlm.zero_grad(set_to_none=True)
                heads.zero_grad(set_to_none=True)
                LT.refresh_quant(model)
                acc["lm/train_ce"].append(float(lce.detach()))
            if step % 25 == 0:
                print(f"HUB step={step} loss={float(loss.detach()):.4f} " + " ".join(
                    f"{kk}={sum(v)/len(v):.3f}" for kk, v in sorted(acc.items())
                    if kk.endswith(("tok_acc", "act_acc", "/r@1")) or "/vq_ppl" in kk or kk.startswith("lm/")) +
                    f" skills={len(skills.mods)} elapsed={time.time()-t0:.0f}s", flush=True)
                acc = collections.defaultdict(list)
            if step % A.eval_every == 0 or step == A.steps:
                if opt_wmem is not None:
                    for t in tasks:
                        t.eval_task.refresh_keys()
                    cur = LT.lm_val(model, lm_heldout)
                    print(f"HUB_LM step={step} val_ce={cur:.4f} delta_vs_frozen={cur - lm_base:+.4f}", flush=True)
                evaluate(tasks, A.k, step)
                if opt_wmem is not None:
                    for t in tasks:
                        t.refresh_keys()
                torch.save({"heads": heads.state_dict(), "layer": A.layer, "n_skills": len(skills.mods), "config": vars(A),
                            "walk_bank": {"k": rn.walk.k, "v": rn.walk.v},
                            "graph_walk": ({"k": heads.graph.walk.k, "v": heads.graph.walk.v}
                                           if getattr(heads, "graph", None) is not None else None)},
                           os.path.join(A.out, f"hub_heads_{step}.pt"))
                if opt_wmem is not None:                  # latest trained Walker (overwritten each eval)
                    torch.save(model.state_dict(), os.path.join(A.out, "walker_state.pt"))
        if tasks:
            torch.save({"heads": heads.state_dict(), "layer": A.layer, "n_skills": len(skills.mods), "config": vars(A),
                            "walk_bank": {"k": rn.walk.k, "v": rn.walk.v},
                            "graph_walk": ({"k": heads.graph.walk.k, "v": heads.graph.walk.v}
                                           if getattr(heads, "graph", None) is not None else None)},
                       os.path.join(A.out, "hub_heads.pt"))
            if opt_wmem is not None:
                torch.save(model.state_dict(), os.path.join(A.out, "walker_state.pt"))
        if A.continual:
            # continual learning: skills one after another; only skill modules train (heads frozen)
            for p in heads.parameters():
                p.requires_grad_(False)
            for mod in skills.mods:
                for p in mod.parameters():
                    p.requires_grad_(True)
            seq = A.continual.split(",")
            cl = []
            for j, nm in enumerate(seq):
                cl.append(pair(nm, (tok, nm, 3000, random.Random(100 + j)), (tok, nm, 150, random.Random(200 + j)),
                               build_skill))

            def unfreeze_new(mod):
                for p in mod.parameters():
                    p.requires_grad_(True)
                on_spawn(mod)
            skills.on_spawn = unfreeze_new
            with torch.no_grad():                      # routing geometry: mean cosine within / across skills
                zs = [F.normalize(rn.pooled([q["q"] for q in t.queries[:32]]).float(), dim=-1) for t in cl]
                cos = [[float((a @ b.T).mean()) for b in zs] for a in zs]
            print("HUB_CL_COS " + " | ".join(f"{t.name}: " + " ".join(f"{c:.3f}" for c in row)
                                             for t, row in zip(cl, cos)), flush=True)
            evaluate(cl, A.k, 0, prefix="HUB_CL after=none")
            for j, t in enumerate(cl):
                for s in range(1, A.continual_steps + 1):
                    l, st = t.run(r.sample(t.queries, A.batch), A.k, train=True)
                    if l.requires_grad:
                        l.backward()
                        for g in opt.param_groups:
                            torch.nn.utils.clip_grad_norm_(g["params"], 1.0)
                            if g.get("name") == "spawned":
                                g["lr"] = 1e-3
                        opt.step()
                        opt.zero_grad(set_to_none=True)
                    if s % 50 == 0:
                        print(f"HUB_CL train={t.name} step={s} loss={float(l.detach()):.4f} "
                              f"tok_acc={sum(st['tok_acc'])/max(1, len(st['tok_acc'])):.3f} "
                              f"skills={len(skills.mods)} frozen={sorted(skills.frozen)}", flush=True)
                evaluate(cl, A.k, A.continual_steps * (j + 1), prefix=f"HUB_CL after={t.name}")
        print(f"HUB done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
