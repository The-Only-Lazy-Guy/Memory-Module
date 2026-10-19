"""Torch-free data layer of the fast-weight memory line: random any-schema JSON documents, their serialization into
pieces with per-leaf pointer chains and value-token prefixes, lookup-question wording, and overlapping copies.

Copied verbatim from gmem_hub.py / gmem_ltm.py / gnm_fwm_anyjson.py (same random calls -> identical documents for a
seed) so the FWM scripts and their data workers no longer import the hub, gmem_ltm, gmem_v3 and the 1B trainer
(6.5 s and ~20k lines per worker process).
"""
import dataclasses
import json
import random  # noqa: F401  (callers pass random.Random)
import re


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
        vpos.append(v1)
    return {"ids": ids, "spans": spans, "ptr": ptr, "vpos": vpos}, leaves
