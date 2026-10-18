"""DIAGNOSTIC ONLY: is 2-hop composition representable on a trained FWM (docs/GNM.md section 9)?

Frozen trained memory; a fresh hop-2 map Wn' is fit WITH the gold second fact (never used in the model) on
training-name worlds, then tested on held-out worlds. High -> the model can compose, the answer-only signal
just does not reach it (credit assignment); low -> the representation cannot (keys/values lack it).

    python gnm_fwm_hop2probe.py runs/fwm_f10.pt --tok-emb
"""
import argparse
import random

try:                                    # Windows: pyarrow must precede torch
    import pyarrow.dataset  # noqa: F401
except Exception:
    pass

import torch
import torch.nn as nn
import torch.nn.functional as F

import gnm_fwm as W
import gnm_small
import gnm_v2 as V2
import gnm_lm_1b as G1
import gmem_ltm as LT
import train_walker_1b_continue as T

P = argparse.ArgumentParser()
P.add_argument("ckpt")
P.add_argument("--tok-emb", action="store_true")
P.add_argument("--steps", type=int, default=600)
A = P.parse_args()
dev = torch.device("cuda")
model = gnm_small.SmallWalker("gpt2").to(dev).eval()
tok = LT.Tok(T._load_tokenizer("gpt2"))
enc = V2.WalkerStateEnc(model, tok)
names = G1.single_token_names(tok, 40)
mem = W.FastWeightMemory(768, 128, 3, tok_emb=A.tok_emb).to(dev)
mem.load_state_dict(torch.load(A.ckpt)["mem"], strict=False)   # older checkpoints lack Wc (zero = shared pool)
mem.eval()
for p in mem.parameters():
    p.requires_grad_(False)


def batch(seed, nm):
    facts, qs = W.make_world(random.Random(seed), nm, 12, 3, 32, two_hop_frac=1.0)
    pieces = [f" {p}'s {r} is {t}." for p, r, t in facts]
    with torch.no_grad():
        X, rng = enc.doc(pieces)
        E = model.emb(torch.tensor([t for pc in pieces for t in tok.ids(pc)], device=dev)).float()
        M, K, V, g = mem.write(X.to(dev), rng, E)
        seqs = [tok.ids(" " + q) for q, _, _ in qs]
        n = max(len(s) for s in seqs)
        ids = torch.tensor([s + [0] * (n - len(s)) for s in seqs], device=dev)
        m = torch.tensor([[1] * len(s) + [0] * (n - len(s)) for s in seqs], device=dev, dtype=torch.bool)
        Hs = mem.std_q(enc._states(ids).float(), m)
        if A.tok_emb:
            Hs = torch.cat([Hs, mem.lne(model.emb(ids).float())], -1)
        hq = mem.Wk(mem.pk(Hs, m))
        Km, Vm, lg = M
        a1 = torch.softmax(mem.tau.exp() * F.normalize(hq - mem.kmean, dim=-1) @ Km.t() + lg, -1)
        r1 = a1 @ Vm
    g1 = torch.tensor([gd[0] for _, _, gd in qs], device=dev)
    g2 = torch.tensor([gd[1] for _, _, gd in qs], device=dev)
    return hq, r1, a1.argmax(-1), K, V, g1, g2


@torch.no_grad()
def prefix_test():
    """hop 1 on 2-hop questions: full question vs only the tokens up to the first relation (causal encoder:
    identical states to the 1-hop question) vs the 1-hop question itself."""
    res = {"full": 0, "prefix": 0, "1-hop form": 0}
    n = 0
    for w in range(8):
        facts, qs = W.make_world(random.Random(4242 + w), names[10:40], 12, 3, 32, two_hop_frac=1.0)
        pieces = [f" {p}'s {r} is {t}." for p, r, t in facts]
        X, rng = enc.doc(pieces)
        E = model.emb(torch.tensor([t for pc in pieces for t in tok.ids(pc)], device=dev)).float()
        M, K, V, g = mem.write(X.to(dev), rng, E)
        g1 = torch.tensor([gd[0] for _, _, gd in qs], device=dev)
        for kind in res:
            texts = []
            for q, _, _ in qs:
                head = q.split("'s")[0] + "'s" + q.split("'s")[1]          # "Who is A's boss"
                texts.append(q if kind == "full" else head if kind == "prefix" else head + "?")
            seqs = [tok.ids(" " + t) for t in texts]
            L = max(len(x) for x in seqs)
            ids = torch.tensor([x + [0] * (L - len(x)) for x in seqs], device=dev)
            m = torch.tensor([[1] * len(x) + [0] * (L - len(x)) for x in seqs], device=dev, dtype=torch.bool)
            Hs = mem.std_q(enc._states(ids).float(), m)
            if A.tok_emb:
                Hs = torch.cat([Hs, mem.lne(model.emb(ids).float())], -1)
            hq = mem.Wk(mem.pk(Hs, m))
            res[kind] += int(((F.normalize(hq - mem.kmean, dim=-1) @ K.t()).argmax(-1) == g1).sum())
        n += len(qs)
    print("PREFIX hop 1 on 2-hop questions lands on fact 1: " + " | ".join(f"{k} {v / n:.2f}" for k, v in res.items()))


prefix_test()
for src in ("read", "gold value"):          # hop-2 input: the actual hop-1 read, or the gold fact-1 value
    Wn = nn.Linear(256, 128).to(dev)
    opt = torch.optim.Adam(Wn.parameters(), 1e-3)
    for step in range(A.steps + 1):
        hq, r1, p1, K, V, g1, g2 = batch(step, names[:30])
        r = V[g1] if src == "gold value" else r1
        q2 = F.normalize(Wn(torch.cat([hq, r], -1)) - mem.kmean, dim=-1)
        loss = F.cross_entropy(mem.tau.exp() * q2 @ K.t(), g2)
        opt.zero_grad()
        loss.backward()
        opt.step()
    acc1, acc2, n = 0, 0, 0
    with torch.no_grad():
        for w in range(8):
            hq, r1, p1, K, V, g1, g2 = batch(4242 + w, names[10:40])
            r = V[g1] if src == "gold value" else r1
            q2 = F.normalize(Wn(torch.cat([hq, r], -1)) - mem.kmean, dim=-1)
            acc1 += int((p1 == g1).sum())
            acc2 += int(((q2 @ K.t()).argmax(-1) == g2).sum())
            n += len(g1)
    print(f"PROBE hop-2 input = {src:10s}: held-out hop-1 lands on fact 1 {acc1 / n:.2f} | "
          f"probe hop-2 lands on fact 2 {acc2 / n:.2f}", flush=True)
