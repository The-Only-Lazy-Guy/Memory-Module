"""fwm_core must reproduce gnm_fwm.FastWeightMemory's lookup path on trained checkpoints (GNM.md 9.11)."""
import os
import random

import pytest

try:
    import pyarrow.dataset  # noqa: F401  (Windows: before torch)
except Exception:
    pass
import torch
import torch.nn.functional as F

CKPT = "runs/aj_comb2.pt"


@pytest.mark.skipif(not os.path.isfile(CKPT), reason="needs a trained anyjson checkpoint")
def test_core_matches_old_module_on_trained_checkpoint():
    import fwm_core as C
    import fwm_data as D
    import gnm_fwm as W
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    model, raw = C.load_walker("gpt2", "", 0, dev)
    tok = C.Tok(raw)
    st = torch.load(CKPT, map_location=dev)
    d = model.emb.weight.shape[1]
    old = W.FastWeightMemory(d, 512, 1, tie=True, center=False, tok_emb=True, act="softmax", seg=True, vslots=8).to(dev)
    old.load_state_dict(st["mem"], strict=False)
    new = C.Memory(d, 512, 8).to(dev)
    assert C.load_memory(new, st["mem"]) == []
    old.eval(); new.eval()
    rng = random.Random(3)
    units_o, units_n, qs = [], [], []
    for _ in range(3):
        name = D._doc_name(rng)
        obj = D.random_doc(rng, D.KEYS_TRAIN, depth=2)
        enc, leaves = C.doc_encoding(tok.ids, raw.decode, obj, name)
        if len(enc["ids"]) > 1000:
            continue
        ids = torch.tensor([enc["ids"]], device=dev)
        X = C.walker_states(model, ids)[0]
        E = model.emb(ids)[0].float()
        with torch.no_grad():
            _, k, v, g = old.write(X, enc["spans"], E, ptr=enc["ptr"], vpos=enc["vpos"])
            units_o.append((old.last_kraw, v, torch.log(g + 1e-6), old.last_vslots))
            units_n.append(new.write(X, E, enc["spans"], enc["ptr"], enc["vpos"]))
        for lf in D.flatten(obj)[:6]:
            if lf.image is None and lf.path:
                key, where = D._subject(lf, name)
                qs.append((D._T_LOOK["eval"][0].format(key=key, where=where, doc=name) if where
                           else D._T_TOP["eval"][0].format(key=key, doc=name)))
    for a, b in zip(units_o, units_n):
        for x, y in zip(a, b[:4]):
            assert torch.allclose(x, y, atol=1e-5)
    seqs = [tok.ids(" " + q) for q in qs]
    n = max(map(len, seqs))
    qid = torch.tensor([s + [0] * (n - len(s)) for s in seqs], device=dev)
    mq = torch.tensor([[1] * len(s) + [0] * (n - len(s)) for s in seqs], device=dev, dtype=torch.bool)
    Hq, Eq = C.walker_states(model, qid), model.emb(qid).float()
    store = C.Store(units_n, [0] * sum(u[0].shape[0] for u in units_n), 1)
    with torch.no_grad():
        old.kmean = store.kmean[[0]]
        old.read((store.K, store.V, store.logg), Hq, mq, Eq)
        qn, tq, _ = old.last_q[0]
        att_o = torch.softmax(tq * qn @ store.K.t() + store.logg, -1)
        att_n = store.attend(new, new.query(Hq, mq, Eq), torch.zeros(len(qs), dtype=torch.long, device=dev))
    assert (att_o - att_n).abs().max() < 1e-4
