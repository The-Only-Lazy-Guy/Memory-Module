"""Fast-weight memory (FWM): memory as a writable network, query -> value (docs/GNM.md section 9).

The GNM graph needs a learned SEARCH over N nodes (addressing), and that is where every attempt stalled.
Here memory is a small network written in one shot from the facts and read by a forward pass, so a lookup
is a transformation, not a learned search. The smart encoder is reused unchanged.

  write   facts (one document) -> frozen encoder (P blocks + one R pass, standardized) -> per fact:
          key k = Wk(pool_k(span)), value v = Wv(pool_v(span)), gate g = sigmoid(Wg(.)) (model's choice)
          softmax (default): one hidden unit per fact, W1 = keys, W2 = values, log g = the unit's gain
          linear (--act linear): M <- M + g * beta * (v - M k^) k^T   (delta rule, k^ = k / |k|)
  read    the question is encoded ALONE (no facts in its context: no in-context shortcut)
          hop 1: q = Wk(question)       r_1 = mem(q^) = softmax(tau q^ K^T + log g) V   (or M q^)
          hop t: q = Wn([question, r_{t-1}]) + e_t   r_t = mem(q^)     (the recurrence lives HERE)
          soft halt: r* = sum_t pi_t r_t, pi = softmax_t Wh([r_t, q_t])
  answer  r* is injected late into Walker (RMS-matched, tanh gate) at the answer position; Walker's own
          LM head says the name. Walker runs ONCE (no R passes): the hops are in the cheap module
          (RECURRENT/OPR finding: recurrence in the big network does not pay for its compute).
  train   final-answer CE only. No per-hop targets, no gold nodes, no copy.

    python gnm_fwm.py --hf gpt2 --steps 2000
"""
import argparse
import json
import math
import random
import time

try:                                    # Windows: pyarrow must precede torch
    import pyarrow.dataset  # noqa: F401
except Exception:
    pass

import torch
import torch.nn as nn
import torch.nn.functional as F

RELS = ("boss", "friend", "neighbor", "mentor", "rival")


# ---------------------------------------------------------------------------------------------- data
def make_world(rng, names, n_people, n_rels, n_q, two_hop_frac=0.5):
    people = rng.sample(names, n_people)
    rels = RELS[:n_rels]
    link = {(p, r): rng.choice([q for q in people if q != p]) for p in people for r in rels}
    facts = [(p, r, t) for (p, r), t in link.items()]
    rng.shuffle(facts)
    qs = []
    for _ in range(n_q):
        p, r1 = rng.choice(people), rng.choice(rels)
        if rng.random() < two_hop_frac:
            r2 = rng.choice(rels)
            mid = link[(p, r1)]
            qs.append((f"Who is {p}'s {r1}'s {r2}?", link[(mid, r2)],
                       [facts.index((p, r1, mid)), facts.index((mid, r2, link[(mid, r2)]))]))
        else:
            qs.append((f"Who is {p}'s {r1}?", link[(p, r1)], [facts.index((p, r1, link[(p, r1)]))]))
    return facts, qs


# typed world (default): people -> person (boss, friend), people -> attribute (city, color, age), variables ->
# numbers. The vocabulary is fixed and known; what is new in every world is the ASSIGNMENT, so an answer is
# predictable only from memory (eval = new worlds, same vocabulary; no held-out names).
PERSON_RELS = ("boss", "friend")
ATTRS = {"city": "cities", "color": "colors", "age": "ages"}
CAND = {"cities": "Paris London Tokyo Berlin Rome Madrid Boston Chicago Moscow Sydney Dublin Vienna Cairo Delhi "
                  "Seattle Denver Houston Miami Athens Oslo Lisbon Prague Warsaw Toronto".split(),
        "colors": "red blue green yellow black white orange purple pink brown gray silver gold".split(),
        "ages": [str(i) for i in range(20, 70)],
        "nums": [str(i) for i in range(0, 100)],
        "vars": "x y z a b c n m k p q t u v w".split()}


def vocab(tok, names):
    V = {k: [w for w in ws if len(tok.ids(" " + w)) == 1] for k, ws in CAND.items()}
    V["names"] = names
    return V


def make_world_typed(rng, V, n_people, n_vars, n_q, two_hop_frac=0.5, depths=None):
    people = rng.sample(V["names"], n_people)
    link = {}
    for p in people:
        for r in PERSON_RELS:
            link[(p, r)] = rng.choice([q for q in people if q != p])
        for r, pool in ATTRS.items():
            link[(p, r)] = rng.choice(V[pool])
    for x in rng.sample(V["vars"], n_vars):
        link[(x, "value")] = rng.choice(V["nums"])
    facts = [(p, r, t) for (p, r), t in link.items()]
    rng.shuffle(facts)
    idx = {(p, r): i for i, (p, r, _) in enumerate(facts)}
    wh = lambda r: "Who" if r in PERSON_RELS else "What"
    qs = []
    for _ in range(n_q):
        d = rng.choice(depths) if depths else (2 if rng.random() < two_hop_frac else 1)
        if d == 1:
            p, r = rng.choice(list(link))
            qs.append((f"{wh(r)} is {p}'s {r}?", link[(p, r)], [idx[(p, r)]]))
            continue
        # chain: d-1 person relations, then any relation ("What is A's boss's friend's boss's city?")
        rels = [rng.choice(PERSON_RELS) for _ in range(d - 1)] + [rng.choice(PERSON_RELS + tuple(ATTRS))]
        cur, path = rng.choice(people), []
        start = cur
        for r in rels:
            path.append(idx[(cur, r)])
            cur = link[(cur, r)]
        qs.append((f"{wh(rels[-1])} is {start}'s " + "'s ".join(rels) + "?", cur, path))
    return facts, qs


FILLERS = ("Let me think.", "Okay, step by step.", "We need to follow the links.", "Hmm, let me check the records.",
           "First, look it up.", "")


def cot_text(rng, facts, path, answer, noise=0.0):
    return " ".join(cot_steps(rng, facts, path, answer, noise))


def cot_steps(rng, facts, path, answer, noise=0.0):
    """A natural-language chain of thought for one question (list of sentences), optionally messy like real
    CoT data: filler, skipped middle steps, a wrong intermediate entity, paraphrases, a distractor fact.
    Never aligned to hops."""
    out = [rng.choice(FILLERS)] if rng.random() < 0.7 * noise else []
    for k, i in enumerate(path):
        p, r, t = facts[i]
        last = k == len(path) - 1
        if not last and 0 < k and rng.random() < 0.3 * noise:
            continue                                                  # skipped step
        if not last and rng.random() < 0.1 * noise:
            t = facts[rng.randrange(len(facts))][2]                   # wrong intermediate entity
        out.append(rng.choice((f"{p}'s {r} is {t}.", f"The {r} of {p} is {t}.", f"{p} has {t} as {r}."))
                   if noise else f"{p}'s {r} is {t}.")
        if rng.random() < 0.2 * noise:
            dp, dr, dt = facts[rng.randrange(len(facts))]
            out.append(f"Also, {dp}'s {dr} is {dt}.")                 # distractor
    out.append(f"So the answer is {answer}.")
    return [x for x in out if x]


def align(C, margin):
    """Monotonic alignment of loop steps (rows) to demonstration sentences (cols): a matched pair scores
    C - margin, gaps are free (filler / distractors / skipped steps are left out). Returns [(t, j)]."""
    T, J = len(C), len(C[0])
    dp = [[0.0] * (J + 1) for _ in range(T + 1)]
    bk = [[0] * (J + 1) for _ in range(T + 1)]
    for t in range(1, T + 1):
        for j in range(1, J + 1):
            opts = (dp[t - 1][j], dp[t][j - 1], dp[t - 1][j - 1] + C[t - 1][j - 1] - margin)
            k = min(range(3), key=lambda i: opts[i])
            dp[t][j], bk[t][j] = opts[k], k
    out, t, j = [], T, J
    while t > 0 and j > 0:
        if bk[t][j] == 2:
            out.append((t - 1, j - 1))
            t, j = t - 1, j - 1
        elif bk[t][j] == 0:
            t -= 1
        else:
            j -= 1
    return out[::-1]


def json_pieces(facts):
    """The facts as ONE nested JSON document (the built smart-encoder input, gnm_v2): a record per subject,
    one "key": value pair per fact. Returns the pieces and, per fact, the index of its pair piece."""
    order, by = [], {}
    for i, (p, r, t) in enumerate(facts):
        if p not in by:
            order.append(p)
            by[p] = []
        by[p].append(i)
    val = lambda t: t if t.isdigit() else json.dumps(t)
    pieces, at, parent = ["{"], [0] * len(facts), [0] * len(facts)
    for j, p in enumerate(order):
        pieces.append((" ," if j else "") + f" {json.dumps(p)}: {{")
        head = len(pieces) - 1
        for k, i in enumerate(by[p]):
            if k:
                pieces.append(",")
            pieces.append(f" {json.dumps(facts[i][1])}: {val(facts[i][2])}")
            at[i], parent[i] = len(pieces) - 1, head
        pieces.append(" }")
    pieces.append(" }")
    return pieces, at, parent


# ---------------------------------------------------------------------------------------------- module
class Pool(nn.Module):
    """Learned attention pooling over a token span (zero-init scorer = mean at start)."""

    def __init__(self, d):
        super().__init__()
        self.s = nn.Linear(d, 1)
        nn.init.zeros_(self.s.weight)
        nn.init.zeros_(self.s.bias)

    def forward(self, H, m):                            # H (N, T, d), m (N, T) bool
        w = torch.softmax(self.s(H).squeeze(-1).masked_fill(~m, -1e4), -1)
        return (w[..., None] * H).sum(1)


class FastWeightMemory(nn.Module):
    def __init__(self, d, dk=128, hops=3, tie=True, center=True, tok_emb=False, act="softmax", tau0=10.0, sg_chain=False, wn_mlp=False, tau_hop0=0.0):
        super().__init__()
        self.dk, self.hops, self.tie, self.center, self.tok_emb, self.act = dk, hops, tie, center, tok_emb, act
        # act = the memory network's hidden activation. "linear": delta-rule matrix M (f1-f7; with the write
        # gate at .33 the crosstalk of the other facts swamped the read: self-recall .50 although the question
        # found its key .79). "softmax": the same memory as a 2-layer net written in one shot, one hidden
        # unit per fact (W1 = keys, W2 = values), the write gate = the unit's log gain.
        # softmax sharpness (learned). A soft start (tau0 ~1) spreads the gradient over all facts so a 2-hop chain
        # can be discovered from the answer alone ("linear start", Sukhbaatar et al. 2015); it sharpens itself
        self.tau = nn.Parameter(torch.tensor(math.log(tau0)))
        # sg_chain: a hop takes the previous read as GIVEN input (no gradient back into earlier hops). Without it
        # a still-random hop 2 bends hop 1 to suit itself (f14: hop 1 landing .92 -> .39 in 500 steps)
        self.sg_chain = sg_chain
        # hops >= 2 get their own (softer) sharpness: a random next-query against N sharp keys puts ~0 weight on
        # the right fact, so no gradient finds it (f16: hop 2 lands ~0 even with hop 1 clean). 0 = shared tau
        self.tau_h = nn.Parameter(torch.full((max(1, hops - 1),), math.log(tau_hop0 if tau_hop0 > 0 else tau0)))
        self.tau_hop = tau_hop0 > 0
        self.kmean = None
        di = 2 * d if tok_emb else d                   # + the token-embedding stream (identity, no position)
        self.lne = nn.LayerNorm(d)
        self.register_buffer("mu", torch.zeros(d))     # per-feature standardization (Walker's states share
        self.register_buffer("var", torch.ones(d))     # one big direction; GNM history bug 16)
        self.register_buffer("seen", torch.zeros(()))
        # queries (question / masked text, encoded ALONE: short, early positions) have their own statistics:
        # standardizing them with the document's left one shared offset in every query, so every query
        # pointed at the same fact (retrieval at chance even when the token stream alone found .56)
        self.register_buffer("mu_q", torch.zeros(d))
        self.register_buffer("var_q", torch.ones(d))
        self.register_buffer("seen_q", torch.zeros(()))
        self.ln = nn.LayerNorm(d)
        self.pk, self.pv, self.pq = Pool(di), Pool(di), Pool(di)
        self.Wk, self.Wv, self.Wq = nn.Linear(di, dk), nn.Linear(di, dk), nn.Linear(di, dk)
        self.Wg = nn.Linear(di, 1)                      # write gate (model-decided knob), starts open
        nn.init.zeros_(self.Wg.weight)
        nn.init.constant_(self.Wg.bias, 3.0)
        self.beta = nn.Parameter(torch.tensor(2.0))     # write strength (sigmoid)
        self.Wn = (nn.Sequential(nn.Linear(2 * dk, 4 * dk), nn.GELU(), nn.Linear(4 * dk, dk)) if wn_mlp
                   else nn.Linear(2 * dk, dk))          # next query from [question, last read]
        # per-hop focus on the question: each hop re-attends over the question tokens, scorer conditioned on
        # [hop, last read] (zero-init: every hop starts as the shared pool). One pool over the whole 2-hop
        # question lands on fact 1 .03; the prefix "Who is A's boss" alone lands .98 (causal states)
        # direction unit-normalized x a learned scale from 0 (f12: an unnormalized Wc grew coherently under Adam,
        # scores exploded, every hop saw one random token and even 1-hop fell to .12)
        self.Wc = nn.Linear(dk, di)
        self.c_scale = nn.Parameter(torch.zeros(()))
        self.hop_e = nn.Embedding(hops, dk)
        # soft halt over hops; also sees the question (hop 1's view): from [read, query] alone it could not tell
        # a 1-hop from a 2-hop question and sent half the 1-hop questions to hop 2 (t3, t4)
        self.Wh = nn.Linear(3 * dk, 1)
        self.Wo = nn.Linear(dk, d)                      # read -> Walker's residual space

    def std(self, X):
        if self.training:
            with torch.no_grad():
                f = 0.0 if float(self.seen) == 0 else 0.99
                self.mu.mul_(f).add_((1 - f) * X.mean(0))
                self.var.mul_(f).add_((1 - f) * X.var(0))
                self.seen.add_(1)
        return self.ln((X - self.mu) / (self.var + 1e-5).sqrt())

    @staticmethod
    def spans(X, rng):
        """rng: per fact a (start, end) range or a list of token positions (pair + its parent record's key)."""
        pos = [list(range(*r)) if isinstance(r, tuple) else list(r) for r in rng]
        n = max(len(q) for q in pos)
        idx = torch.zeros(len(pos), n, dtype=torch.long, device=X.device)
        m = torch.zeros(len(pos), n, dtype=torch.bool, device=X.device)
        for i, q in enumerate(pos):
            idx[i, :len(q)] = torch.tensor(q, device=X.device)
            m[i, :len(q)] = True
        return X[idx], m

    def write(self, Xdoc, rng, E=None):
        """Delta-rule write of every fact span into a fresh M (dk x dk). Returns M, keys, values."""
        Xs = self.std(Xdoc)
        if self.tok_emb:
            Xs = torch.cat([Xs, self.lne(E)], -1)
        H, m = self.spans(Xs, rng)
        kr = self.Wk(self.pk(H, m))
        # centre the keys per world (and the queries with the same mean): untrained keys are near-orthogonal
        # (self-recall 1.00) but the answer loss pulled them onto one shared direction (key cos .68, recall .31)
        self.kmean = kr.mean(0, keepdim=True) if self.center else torch.zeros_like(kr[:1])
        k = F.normalize(kr - self.kmean, dim=-1)
        v = self.Wv(self.pv(H, m))
        g = torch.sigmoid(self.Wg(self.pv(H, m))).squeeze(-1) * torch.sigmoid(self.beta)
        if self.act == "softmax":
            return (k, v, torch.log(g + 1e-6)), k, v, g
        M = torch.zeros(self.dk, self.dk, device=Xdoc.device)
        for i in range(k.shape[0]):                     # sequential: later facts can overwrite earlier ones
            M = M + g[i] * torch.outer(v[i] - M @ k[i], k[i])
        return M, k, v, g

    def read(self, M, Hq, mq, Eq=None):
        """k hops in the module; returns the halted read r* (B, dk) and the per-hop trace."""
        Hs = self.std_q(Hq, mq)
        if self.tok_emb:
            Hs = torch.cat([Hs, self.lne(Eq)], -1)
        # tie: the question uses the SAME key head and pool as the facts, so
        pool, head = (self.pk, self.Wk) if self.tie else (self.pq, self.Wq)        # q and k share a space
        base = pool.s(Hs).squeeze(-1)                                              # (B, T) shared scorer
        reads, logits, qs, atts = [], [], [], []
        for t in range(self.hops):
            prev = (reads[-1].detach() if self.sg_chain else reads[-1]) if t > 0 else None
            ctrl = self.hop_e.weight[t].expand(Hs.shape[0], -1) + (prev if t > 0 else 0)
            sc = base + self.c_scale * torch.einsum("btd,bd->bt", Hs, F.normalize(self.Wc(ctrl), dim=-1))
            w = torch.softmax(sc.masked_fill(~mq, -1e4), -1)
            hq = head((w[..., None] * Hs).sum(1))                                  # this hop's view of the question
            q = hq if t == 0 else self.Wn(torch.cat([hq, prev], -1)) + self.hop_e.weight[t]
            qn = F.normalize(q - self.kmean, dim=-1)
            if self.act == "softmax":
                Km, Vm, lg = M
                tau = self.tau_h[t - 1].exp() if (t > 0 and self.tau_hop) else self.tau.exp()
                a_ = torch.softmax(tau * qn @ Km.t() + lg, -1)
                atts.append((qn, tau))
                r = a_ @ Vm
            else:
                r = qn @ M.t()
            reads.append(r)
            qs.append(q)
            if t == 0:
                h_first = hq
            logits.append(self.Wh(torch.cat([r, q, h_first], -1)))
        self.last_q = atts                                               # per hop (query^, tau): the loop's action
        pi = torch.softmax(torch.cat(logits, -1), -1)                    # (B, hops)
        R = torch.stack(reads, 1)                                        # (B, hops, dk)
        return (pi[..., None] * R).sum(1), R, pi

    def std_q(self, Hq, mq, update=True):
        if self.training and update:
            with torch.no_grad():
                X = Hq[mq]
                f = 0.0 if float(self.seen_q) == 0 else 0.99
                self.mu_q.mul_(f).add_((1 - f) * X.mean(0))
                self.var_q.mul_(f).add_((1 - f) * X.var(0))
                self.seen_q.add_(1)
        return self.ln((Hq - self.mu_q) / (self.var_q + 1e-5).sqrt())


# ---------------------------------------------------------------------------------------------- run
def main():
    P = argparse.ArgumentParser()
    P.add_argument("--hf", default="gpt2")
    P.add_argument("--steps", type=int, default=2000)
    P.add_argument("--people", type=int, default=12)
    P.add_argument("--rels", type=int, default=3)
    P.add_argument("--nq", type=int, default=32)
    P.add_argument("--two-hop", type=float, default=0.5)
    P.add_argument("--dk", type=int, default=128)
    P.add_argument("--hops", type=int, default=3)
    P.add_argument("--halt", choices=("best", "mix"), default="best",
                   help="best: every hop's read is a rollout (Walker answers from each), loss = soft-min over hops, "
                        "halt learns which hop wins, inference uses its argmax; mix: one read = sum_t pi_t r_t "
                        "(f1-f8: the halt collapsed onto hop 1 and hops 2-3 never got a gradient)")
    P.add_argument("--wn-mlp", action="store_true", help="next-query map is a 2-layer MLP (linear: ~.5 composition)")
    P.add_argument("--train-only", default="", help="comma list of FWM parameter-name prefixes to train (rest "
                   "frozen), e.g. a phase-2 chain stage: Wn,Wc,c_scale,hop_e,Wh")
    P.add_argument("--tau-hop0", type=float, default=0.0, help="own initial sharpness for hops >= 2 (0: shared)")
    P.add_argument("--detach-own1", action="store_true", help="hop 1's own answer does not train hop 1's query; "
                   "hop 1 learns only from the hops that use its read (stage 3: question focus via the chain)")
    P.add_argument("--cot-il", action="store_true", help="imitation: the CoT (a demonstration, possibly messy) "
                   "supervises each hop's lookup via aligned expert attention; Walker only says the answer")
    P.add_argument("--il-w", type=float, default=1.0)
    P.add_argument("--il-conf", type=float, default=0.5, help="demonstration sentences whose expert attention "
                   "peaks below this are skipped (filler)")
    P.add_argument("--cot-text", action="store_true", help="Walker generates the whole natural-language CoT after "
                   "the hop latents (k slots); LM loss, no alignment; the answer keeps the soft-min rollouts")
    P.add_argument("--cot-noise", type=float, default=0.0, help="messy CoT: filler, skipped / wrong steps, distractors")
    P.add_argument("--cot-w", type=float, default=1.0)
    P.add_argument("--cot-inject", type=int, default=6, help="block whose input gets the CoT slots")
    P.add_argument("--cot", action="store_true", help="LOTUS-style: CoT steps supervise every hop's read in "
                   "parallel (Walker says step t from hop t), halt target = CoT length; replaces the soft-min loss")
    P.add_argument("--ponder", type=float, default=0.0, help="per-hop cost (nats) in the soft-min / halt target")
    P.add_argument("--sg-chain", action="store_true", help="each hop takes the previous read as given input")
    P.add_argument("--tau0", type=float, default=10.0, help="initial softmax sharpness of the memory read")
    P.add_argument("--act", choices=("softmax", "linear"), default="softmax",
                   help="memory net activation: softmax (one hidden unit per fact) or linear (delta-rule matrix)")
    P.add_argument("--lr", type=float, default=1e-3)
    P.add_argument("--inject", type=int, default=-2, help="block index whose input gets the read (neg = from end)")
    P.add_argument("--no-tie", action="store_true", help="separate question head (ablation)")
    P.add_argument("--no-center", action="store_true", help="no per-world key centering (ablation)")
    P.add_argument("--tok-emb", action="store_true", help="key/value/query heads also see the token-embedding stream")
    P.add_argument("--oracle-key", action="store_true", help="DIAGNOSTIC: read with the stored key itself")
    P.add_argument("--cloze", type=int, default=0, help="masked-recall queries per world (self-supervised: a stored "
                   "sentence minus its last word must recall that word)")
    P.add_argument("--fixed", action="store_true", help="one fixed world (overfit sanity check)")
    P.add_argument("--eval-every", type=int, default=200)
    P.add_argument("--seed", type=int, default=0)
    P.add_argument("--save", default="runs/fwm.pt")
    P.add_argument("--load", default="", help="start from a saved FWM (with --steps 0: evaluate only)")
    P.add_argument("--data", choices=("typed", "people"), default="typed",
                   help="typed: people/attributes/variables (default); people: person relations only (f1-f17)")
    P.add_argument("--depths", default="", help="typed: question depths sampled uniformly, e.g. 1,2,3,4,5 "
                   "(default: --two-hop fraction of 2-hop, rest 1-hop)")
    P.add_argument("--doc", choices=("json", "text"), default="json",
                   help="json: facts written as ONE nested JSON document through the built encoder; text: sentences")
    P.add_argument("--vars", type=int, default=8, help="typed: variables per world (x's value is 42)")
    A = P.parse_args()
    import gnm_small
    import gnm_v2 as V2
    import gnm_lm_1b as G1
    import gmem_ltm as LT
    import train_walker_1b_continue as T
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(A.seed)
    model = gnm_small.SmallWalker(A.hf).to(dev)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    tok = LT.Tok(T._load_tokenizer(A.hf))
    enc = V2.WalkerStateEnc(model, tok)
    d = model.emb.weight.shape[1]
    names = G1.single_token_names(tok, 40)
    V = vocab(tok, names)

    def world(rng_, n_q):
        if A.data == "typed":
            return make_world_typed(rng_, V, A.people, A.vars, n_q, A.two_hop,
                                    [int(x) for x in A.depths.split(",")] if A.depths else None)
        return make_world(rng_, names, A.people, A.rels, n_q, A.two_hop)
    mem = FastWeightMemory(d, A.dk, A.hops, tie=not A.no_tie, center=not A.no_center, tok_emb=A.tok_emb, act=A.act,
                           tau0=A.tau0, sg_chain=A.sg_chain,
                           wn_mlp=A.wn_mlp, tau_hop0=A.tau_hop0).to(dev)
    gate = nn.Parameter(torch.tensor(1.0, device=dev))      # injection strength (tanh)
    gate_c = nn.Parameter(torch.tensor(1.0, device=dev))    # CoT-slot injection strength (tanh)
    if A.train_only:                                        # staged training: the rest stays as loaded
        keep = tuple(A.train_only.split(","))
        for n_, p_ in mem.named_parameters():
            p_.requires_grad_(n_.startswith(keep))
        gate.requires_grad_(False)
    opt = torch.optim.AdamW([p_ for p_ in list(mem.parameters()) + [gate, gate_c] if p_.requires_grad], lr=A.lr,
                            weight_decay=0.01)
    inj = A.inject % len(model.blocks)
    ans_ids = tok.ids(" Answer:")
    state = {"add": None}

    def pre(_m, args):                                      # add the read at the answer position
        if state["add"] is None:
            return None
        x = args[0]
        rms = x[:, -1:].pow(2).mean(-1, keepdim=True).sqrt()
        a = state["add"]
        a = a / a.pow(2).mean(-1, keepdim=True).sqrt().clamp(min=1e-6)
        x = x.clone()
        x[:, -1] = x[:, -1] + (torch.tanh(gate) * rms[:, 0] * a).to(x.dtype)
        return (x,) + tuple(args[1:])
    model.blocks[inj].register_forward_pre_hook(pre)
    state["slots"] = None

    def pre_slots(_m, args):                                # the hop latents as k memory slots before the CoT
        if state["slots"] is None:
            return None
        x = args[0]
        a = state["slots"]                                  # (B, k, d)
        k = a.shape[1]
        rms = x[:, :k].pow(2).mean(-1, keepdim=True).sqrt()
        a = a / a.pow(2).mean(-1, keepdim=True).sqrt().clamp(min=1e-6)
        x = x.clone()
        x[:, :k] = x[:, :k] + (torch.tanh(gate_c) * rms * a).to(x.dtype)
        return (x,) + tuple(args[1:])
    model.blocks[A.cot_inject % len(model.blocks)].register_forward_pre_hook(pre_slots)
    slot_id, rsn_ids = tok.ids(" -")[0], tok.ids(" Reasoning:")

    def encode_q(texts):
        seqs = [tok.ids(" " + t) for t in texts]
        n = max(len(s) for s in seqs)
        ids = torch.tensor([s + [0] * (n - len(s)) for s in seqs], device=dev)
        m = torch.tensor([[1] * len(s) + [0] * (n - len(s)) for s in seqs], device=dev, dtype=torch.bool)
        with torch.no_grad():
            Hq = enc._states(ids).float()
            Eq = model.emb(ids).float()
        return Hq, m, Eq

    def episode(rng, train, mem_from=None, off=False):
        facts, qs = world(rng, A.nq)
        if mem_from is not None:                            # ablation: memory of ANOTHER world
            facts_m, _ = world(mem_from, 1)
        else:
            facts_m = facts
        if A.doc == "json":
            doc_pieces, at, parent = json_pieces(facts_m)
        else:
            doc_pieces = [f" {p}'s {r} is {t}." for p, r, t in facts_m]
            at, parent = list(range(len(doc_pieces))), None
        pieces = [f" {p}'s {r} is {t}." for p, r, t in facts_m]      # the facts as text (masked-recall queries)
        with torch.no_grad():
            Xdoc, rng_tok = enc.doc(doc_pieces)
            Edoc = model.emb(torch.tensor([t for pc in doc_pieces for t in tok.ids(pc)], device=dev)).float()
        # a nested pair POINTS to its record (design 1.4): the fact unit pools over its own pair and the record's
        # key; from the pair alone (' "city": "Miami"') the subject is only context, and 1-hop fell 1.00 -> .14
        spans_ = [rng_tok[i] if parent is None else list(range(*rng_tok[parent[j]])) + list(range(*rng_tok[i]))
                  for j, i in enumerate(at)]
        Mm, K, Vv, g = mem.write(Xdoc.to(dev), spans_, Edoc)
        texts = [q for q, _, _ in qs]
        y = [tok.ids(" " + a)[0] for _, a, _ in qs]
        n_cloze = 0
        if train and A.cloze > 0:   # masked recall: any stored sentence minus its last word is a query
            for i in random.Random(len(texts) + int(torch.randint(0, 10 ** 6, (1,)))).sample(
                    range(len(pieces)), min(A.cloze, len(pieces))):
                words = pieces[i].strip().rstrip(".").split(" ")
                texts.append(" ".join(words[:-1]))
                y.append(tok.ids(" " + words[-1])[0])
                n_cloze += 1
        Hq, mq, Eq = encode_q(texts)
        rstar, R, pi = mem.read(Mm, Hq, mq, Eq)
        if A.oracle_key:   # DIAGNOSTIC ONLY: query every row with a stored key (addressing given) -> decoding test
            pick = torch.tensor([random.randrange(K.shape[0]) for _ in texts], device=dev)
            rstar = Vv[pick] if A.act == "softmax" else K[pick] @ Mm.t()
            y_or = [tok.ids(" " + pieces[int(i)].strip().rstrip(".").split(" ")[-1])[0] for i in pick]
        y = torch.tensor(y_or if A.oracle_key else y, device=dev)
        B, Th = R.shape[0], R.shape[1]
        if A.halt == "best" and not A.oracle_key:
            # every hop's read is one rollout: Walker answers from each (a 2-token forward, batched)
            Rin = R
            if A.detach_own1:   # hop 1's own answer does not train hop 1's query (only the chain through it does)
                Rin = torch.cat([R[:, :1].detach(), R[:, 1:]], 1)
            state["add"] = None if off else mem.Wo(Rin.reshape(B * Th, -1))
            ids = torch.tensor([ans_ids] * (B * Th), device=dev)
            lg, _ = model(ids, r_active=False)
            state["add"] = None
            lg = lg[:, -1].float().view(B, Th, -1)
            ce = F.cross_entropy(lg.reshape(B * Th, -1), y.repeat_interleave(Th), reduction="none").view(B, Th)
            # soft-min: the best rollout counts most; a small per-hop cost (ponder) prefers the earlier of two
            # hops that both answer (f-t3: the halt sent half the 1-hop questions to hop 2)
            cost = A.ponder * torch.arange(Th, device=dev, dtype=ce.dtype)
            w = torch.softmax(-(ce.detach() + cost), -1)
            loss = (w * ce).sum(-1).mean() - (w * torch.log(pi + 1e-9)).sum(-1).mean()   # + halt predicts it
            if A.cot_il and train and A.act == "softmax":
                # IMITATION of the loop from the CoT (a demonstration, possibly messy). Each CoT sentence, encoded
                # alone through the same encoder + key head, gives the expert's action: its attention over the
                # memory, a*_j. The loop's action at hop t is its attention a_t. A monotonic alignment pairs hops
                # with sentences (filler / distractors / skipped steps left out); loss = KL(a*_j || a_t) on the
                # pairs. A wrong value in a sentence still demonstrates the right lookup (the key is subject +
                # relation). Walker is not involved; it only says the final answer.
                crng = random.Random(int(torch.randint(0, 10 ** 9, (1,))))
                sents = [cot_steps(crng, facts, gold, a_, A.cot_noise) for _, a_, gold in qs]
                flat = [x for ss in sents for x in ss]
                Km, Vm, lgm = Mm
                with torch.no_grad():
                    Hc, mc, Ec = encode_q(flat)
                    Hcs = mem.std_q(Hc, mc, update=False)
                    if A.tok_emb:
                        Hcs = torch.cat([Hcs, mem.lne(Ec)], -1)
                    ec = F.normalize(mem.Wk(mem.pk(Hcs, mc)) - mem.kmean, dim=-1)
                    astar = torch.softmax(mem.tau.exp() * ec @ Km.t() + lgm, -1)      # (S, N) expert actions
                # the loop's action, with the MEMORY side detached (keys, gates): imitation may only move the
                # queries. Without this the memory collapses (write gate -> 0) until expert and loop agree trivially
                la = torch.stack([torch.log_softmax(tq.detach() * qn_ @ Km.detach().t() + lgm.detach(), -1)
                                  for qn_, tq in mem.last_q], 1)                     # (B, hops, N)
                il_terms, o = [], 0
                margin = math.log(Km.shape[0]) + 1.0          # early (uniform loop) every in-order match pays
                for b_, ss in enumerate(sents):
                    ex = astar[o:o + len(ss)]
                    o += len(ss)
                    keep = (ex.max(-1).values > A.il_conf).nonzero().flatten()           # expert knows what to look up
                    if len(keep) == 0:
                        continue
                    ex = ex[keep]
                    C = (ex[None] * (torch.log(ex[None] + 1e-9) - la[b_][:, None])).sum(-1)   # (hops, J) KL
                    for t_, j_ in align(C.detach().tolist(), margin):
                        il_terms.append(C[t_, j_])
                if il_terms:
                    il = torch.stack(il_terms).mean()
                    loss = loss + A.il_w * il
                    episode.il = float(il)
            if A.cot_text and train:
                # free-text CoT: Walker generates the whole (possibly messy) CoT after the k hop slots; plain LM
                # loss on the CoT tokens, no alignment of sentences to hops
                crng = random.Random(int(torch.randint(0, 10 ** 9, (1,))))
                seqs = [tok.ids(" " + cot_text(crng, facts, gold, a_, A.cot_noise)) for _, a_, gold in qs]
                nq_, L = len(qs), max(len(x) for x in seqs)
                pre_ = [slot_id] * Th + rsn_ids
                ids_c = torch.tensor([pre_ + x + [0] * (L - len(x)) for x in seqs], device=dev)
                state["slots"] = mem.Wo(R[:nq_])
                lgc, _ = model(ids_c, r_active=False)
                state["slots"] = None
                tgt = torch.full(ids_c.shape, -100, dtype=torch.long, device=dev)
                for b_, x in enumerate(seqs):
                    tgt[b_, len(pre_):len(pre_) + len(x)] = torch.tensor(x, device=dev)
                loss = loss + A.cot_w * F.cross_entropy(lgc[:, :-1].float().reshape(-1, lgc.shape[-1]),
                                                        tgt[:, 1:].reshape(-1), ignore_index=-100)
            if A.cot:
                # LOTUS-style: the CoT supervises the latent hops IN PARALLEL. Step t of the CoT ("A's boss is B")
                # is the entity Walker must say from hop t's read; the halt's target is the CoT length. Content
                # only (text tokens), never which node. Masked-recall rows are 1-step CoTs.
                steps = [[tok.ids(" " + facts[i][2])[0] for i in gold] for _, _, gold in qs]
                steps += [[int(t)] for t in y[len(qs):].tolist()]
                Tg = torch.full((B, Th), -100, dtype=torch.long, device=dev)
                for b_, st_ in enumerate(steps):
                    st_ = st_[:Th]
                    Tg[b_, :len(st_)] = torch.tensor(st_, device=dev)
                dep = torch.tensor([min(len(st_), Th) - 1 for st_ in steps], device=dev)
                loss = (F.cross_entropy(lg.reshape(B * Th, -1), Tg.reshape(-1), ignore_index=-100)
                        - torch.log(pi[torch.arange(B, device=dev), dep] + 1e-9).mean())
            last = lg[torch.arange(B, device=dev), pi.argmax(-1)]      # inference: the halt's hop, Walker once
            pr = torch.softmax(lg, -1)                                 # (B, hops, V) alternatives, measured only
            episode.alt = {"mix": (pi[..., None] * pr).sum(1).argmax(-1),            # halt-weighted answers
                           "conf": pr.max(-1).values.argmax(-1),                      # Walker's most confident hop
                           "oracle": None, "y": y, "per_hop_ok": pr.argmax(-1) == y[:, None]}
            episode.alt["conf"] = lg[torch.arange(B, device=dev), episode.alt["conf"]].argmax(-1)
        else:
            state["add"] = None if off else mem.Wo(rstar)
            ids = torch.tensor([ans_ids] * len(texts), device=dev)
            logits, _ = model(ids, r_active=False)
            state["add"] = None
            last = logits[:, -1].float()
            loss = F.cross_entropy(last, y)
        nq = len(qs)
        acc = (last[:nq].argmax(-1) == y[:nq]).float()
        episode.cloze_acc = (float((last[nq:].argmax(-1) == y[nq:]).float().mean()) if n_cloze else -1.0)
        R, pi = R[:nq], pi[:nq]
        if A.halt == "best" and not A.oracle_key:
            al = episode.alt
            alt = {"mix": (al["mix"][:nq] == y[:nq]).float(), "conf": (al["conf"][:nq] == y[:nq]).float(),
                   "best hop": al["per_hop_ok"][:nq].any(-1).float()}
        else:
            alt = {}
        hop = torch.tensor([len(gold) for _, _, gold in qs], device=dev)          # question depth
        # diagnostic only: which stored fact does each hop's read land on (cosine to the written values);
        # path_ok[b, t] = hop t lands on the t-th fact of the question's path (padding past the depth = True)
        land = (F.normalize(R, dim=-1) @ F.normalize(Vv, dim=-1).t()).argmax(-1)      # (B, hops)
        Dm = int(hop.max())
        G = torch.tensor([gold + [-1] * (Dm - len(gold)) for _, _, gold in qs], device=dev)
        L = torch.full_like(G, -2)
        L[:, :min(Dm, land.shape[1])] = land[:, :Dm]
        ok = (L == G) | (G < 0)
        diag = {"path_ok": ok, "pi": [round(float(v), 2) for v in pi.mean(0)],
                "gate_w": round(float(g.mean()), 2), "alt": alt}
        return loss, acc, hop, diag

    @torch.no_grad()
    def evaluate(step):
        mem.eval()
        out = {}
        for name, kw in (("new world", {}), ("memory off", {"off": True}), ("other world's memory", {"swap": True})):
            rng = random.Random(4242)
            S = {}                                      # depth -> {"n", "acc", alt..., "path": [ok per hop]}
            for w in range(8):
                mf = random.Random(9000 + w) if kw.get("swap") else None
                _, acc, hop, dg = episode(rng, False, mem_from=mf, off=kw.get("off", False))
                for d in sorted(set(hop.tolist())):
                    m_ = hop == d
                    e = S.setdefault(d, {"n": 0, "acc": 0.0, "path": [0.0] * d, "chain": 0.0})
                    e["n"] += int(m_.sum())
                    e["acc"] += float(acc[m_].sum())
                    po = dg["path_ok"][m_][:, :d]
                    for t in range(d):
                        e["path"][t] += float(po[:, t].sum())
                    e["chain"] += float(po.all(-1).sum())
                    for k_, v_ in dg["alt"].items():
                        e[k_] = e.get(k_, 0.0) + float(v_[m_].sum())
            ds = sorted(S)
            out[name] = [S[d]["acc"] / S[d]["n"] for d in ds]
            alt_keys = [k_ for k_ in ("mix", "best hop") if k_ in S[ds[0]]]
            print(f"EVAL step {step} {name}: acc by depth " + " ".join(f"{d}:{S[d]['acc'] / S[d]['n']:.2f}" for d in ds)
                  + "".join(f" | {k_} " + " ".join(f"{S[d][k_] / S[d]['n']:.2f}" for d in ds) for k_ in alt_keys)
                  + f" | halt {dg['pi']}", flush=True)
            if name == "new world":
                print(f"PATH {name}: whole chain lands " + " ".join(f"{d}:{S[d]['chain'] / S[d]['n']:.2f}" for d in ds)
                      + " | per hop (deepest) " + " ".join(f"{x / S[ds[-1]]['n']:.2f}" for x in S[ds[-1]]["path"]),
                      flush=True)
        mem.train()
        return out

    if A.load:
        st = torch.load(A.load, map_location=dev)
        own = mem.state_dict()
        drop = [k for k, v in st["mem"].items() if k in own and own[k].shape != v.shape]
        miss = mem.load_state_dict({k: v for k, v in st["mem"].items() if k not in drop}, strict=False)
        if drop:
            print(f"re-initialized (shape changed): {drop}", flush=True)
        print(f"loaded {A.load}; new (not in checkpoint): {miss.missing_keys}", flush=True)
        if A.tau_hop0 > 0 and not st.get("tau_hop", False):   # chain stage: soft start (stage 1 never used tau_h)
            with torch.no_grad():
                mem.tau_h.fill_(math.log(A.tau_hop0))
        with torch.no_grad():
            gate.copy_(st["gate"])
    if A.steps == 0:
        evaluate(0)
        return
    rng = random.Random(A.seed)
    t0, best = time.time(), -1.0
    print(f"FWM: {A.hf} d={d} dk={A.dk} hops={A.hops} inject@block{inj} data={A.data} "
          f"facts/world={len(world(random.Random(0), 1)[0])} params={sum(p.numel() for p in mem.parameters()):,}", flush=True)
    for step in range(1, A.steps + 1):
        lr = A.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - 1) / A.steps)))
        for gp in opt.param_groups:
            gp["lr"] = lr
        loss, acc, hop, dg = episode(random.Random(A.seed) if A.fixed else rng, True)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p_ for p_ in list(mem.parameters()) + [gate] if p_.requires_grad], 1.0)
        opt.step()
        if step % 50 == 0:
            print(f"step {step} loss {float(loss):.3f} acc by depth "
                  + " ".join(f"{d}:{float(acc[hop == d].mean()):.2f}" for d in sorted(set(hop.tolist())))
                  + f" | chain lands {float(dg['path_ok'].all(-1).float().mean()):.2f}"
                  f" | halt {dg['pi']} write-gate "
                  f"{dg['gate_w']} tau {float(mem.tau.exp()):.1f} focus {float(mem.c_scale):.2f} tau-hop {[round(float(x), 1) for x in mem.tau_h.exp()]} inj {float(torch.tanh(gate)):.2f} cloze {getattr(episode, 'cloze_acc', -1):.2f} il {getattr(episode, 'il', -1):.2f} "
                  f"[{time.time() - t0:.0f}s]", flush=True)
        if step % A.eval_every == 0 or step == A.steps:
            out = evaluate(step)
            score = sum(out["new world"])
            if score > best:
                best = score
                torch.save({"mem": mem.state_dict(), "gate": gate.detach(), "step": step, "eval": out,
                            "tau_hop": A.tau_hop0 > 0}, A.save)


if __name__ == "__main__":
    main()
