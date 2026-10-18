import pyarrow.dataset  # noqa: F401  (must precede torch on Windows)
import random
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, r"C:/Users/Ace/Downloads/CPT")
import gnm_fwm as W
import gnm_small
import gnm_v2 as V2
import gnm_lm_1b as G1
import gmem_ltm as LT
import train_walker_1b_continue as T

dev = torch.device("cuda")
model = gnm_small.SmallWalker("gpt2").to(dev).eval()
tok = LT.Tok(T._load_tokenizer("gpt2"))
enc = V2.WalkerStateEnc(model, tok)
names = G1.single_token_names(tok, 40)


def qenc(texts):
    seqs = [tok.ids(" " + t) for t in texts]
    n = max(len(s) for s in seqs)
    ids = torch.tensor([s + [0] * (n - len(s)) for s in seqs], device=dev)
    m = torch.tensor([[1] * len(s) + [0] * (n - len(s)) for s in seqs], device=dev, dtype=torch.bool)
    return enc._states(ids).float(), m


def world(seed, nm):
    facts, qs = W.make_world(random.Random(seed), nm, 12, 3, 32, two_hop_frac=0.0)
    pieces = [f" {p}'s {r} is {t}." for p, r, t in facts]
    X, rng = enc.doc(pieces)
    cl = [" ".join(pc.strip().rstrip(".").split(" ")[:-1]) for pc in pieces]
    return facts, qs, X.to(dev), rng, cl


@torch.no_grad()
def run(mem, label):
    mem.eval()
    res = {k: [] for k in ("Q key top1", "Q key target-ok", "Q value top1", "C key top1", "C value top1",
                           "self-recall", "Q softmax b10", "Q softmax b30", "key cos off-diag")}
    for w in range(8):
        facts, qs, X, rng, cl = world(4242 + w, names[10:40])
        M, K, V, g = mem.write(X, rng)
        Vn = F.normalize(V, dim=-1)
        n = K.shape[0]
        sr = (F.normalize(K @ M.t(), dim=-1) @ Vn.t()).argmax(-1).tolist()
        res["self-recall"] += [int(facts[a][2] == facts[b][2]) for b, a in enumerate(sr)]
        Vc = Vn @ Vn.t()
        same_t = torch.tensor([[facts[i][2] == facts[j][2] and i != j for j in range(n)] for i in range(n)], device=dev)
        diff_t = ~same_t & ~torch.eye(n, dtype=torch.bool, device=dev)
        res.setdefault("value cos same-target", []).append(float(Vc[same_t].mean()))
        res.setdefault("value cos diff-target", []).append(float(Vc[diff_t].mean()))
        res.setdefault("write gate", []).append(float(g.mean()))
        res["key cos off-diag"].append(float((K @ K.t())[~torch.eye(n, dtype=torch.bool, device=dev)].mean()))
        for kind, texts, gold in (("Q", [q for q, _, _ in qs], [gd[0] for _, _, gd in qs]),
                                  ("C", cl, list(range(len(cl))))):
            Hq, mq = qenc(texts)
            Hs = mem.std_q(Hq, mq)
            hq = mem.Wk(mem.pk(Hs, mq))
            qn = F.normalize(hq - mem.kmean, dim=-1)
            kt = (qn @ K.t()).argmax(-1).tolist()
            vt = (F.normalize(qn @ M.t(), dim=-1) @ Vn.t()).argmax(-1).tolist()
            res[f"{kind} key top1"] += [int(a == b) for a, b in zip(kt, gold)]
            res[f"{kind} value top1"] += [int(facts[a][2] == facts[b][2]) for a, b in zip(vt, gold)]   # same target name
            if kind == "Q":
                for b in (10, 30):
                    rs = torch.softmax(b * qn @ K.t(), -1) @ V
                    res[f"Q softmax b{b}"] += [int(a == c) for a, c in zip((F.normalize(rs, dim=-1) @ Vn.t()).argmax(-1).tolist(), gold)]
                res["Q key target-ok"] += [int(facts[a][2] == ans) for a, (_, ans, _) in zip(kt, qs)]
    print(label, " | ".join(f"{k} {sum(v) / len(v):.2f}" for k, v in res.items()), flush=True)


# untrained: warm the running stats (docs + questions + cloze) on training-name worlds
mem0 = W.FastWeightMemory(768, 128, 3, act="linear").to(dev)
mem0.train()
with torch.no_grad():
    for w in range(20):
        facts, qs, X, rng, cl = world(w, names[:30])
        mem0.std(X)
        Hq, mq = qenc([q for q, _, _ in qs] + cl)
        mem0.std_q(Hq, mq)
run(mem0, "untrained")
mem = W.FastWeightMemory(768, 128, 3, act="linear").to(dev)
mem.load_state_dict(torch.load("runs/fwm_f7.pt")["mem"], strict=False)
run(mem, "f7       ")
