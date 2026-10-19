"""Global persistent world for the fast-weight memory (docs/GNM.md section 7 correction 1, section 9.4).

ONE world lives for the whole run and every fact is one unit in a persistent store (the memory). Each step a
small component (a few records) is written as one JSON document through the smart encoder: those units are
re-encoded with gradient and may have drifted to new values (an update). Questions start in that component
and chain through facts written in earlier steps, so the read searches the whole store. The store is
re-encoded without gradient every --refresh steps (old units were encoded with older weights).

Eval: held-out question chains (never asked in training) on the CURRENT world, answered from memory;
memory off. Drift makes weight-memorization of the world wrong.

    python gnm_fwm_global.py --hf gpt2 --steps 4000 --depths 1,2,3,4,5 --hops 6
"""
import argparse
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

import gnm_fwm as W


def entity_names(hf_tok, tok, n, seed, avoid):
    """n single-token capitalized words of the tokenizer, used as entity names (40 person names are too few)."""
    words = sorted({w[1:] for w in hf_tok.get_vocab()
                    if w.startswith("Ġ") and w[1:].isalpha() and w[1:2].isupper() and 3 <= len(w) - 1 <= 9})
    words = [w for w in words if w not in avoid and len(tok.ids(" " + w)) == 1]
    random.Random(seed).shuffle(words)
    return words[:n]


class World:
    """The one world: people -> person (boss, friend) and attributes (city, color, age); variables -> numbers."""

    def __init__(self, rng, V, people, variables):
        self.people, self.vars = people, variables
        self.link = {}
        for p in people:
            for r in W.PERSON_RELS:
                self.link[(p, r)] = rng.choice([q for q in people if q != p])
            for r, pool in W.ATTRS.items():
                self.link[(p, r)] = rng.choice(V[pool])
        for x in variables:
            self.link[(x, "value")] = rng.choice(V["nums"])
        self.keys = list(self.link)                         # stable fact ids
        self.idx = {k: i for i, k in enumerate(self.keys)}
        self.rec = {}
        for i, (s, _) in enumerate(self.keys):
            self.rec.setdefault(s, []).append(i)
        self.V = V

    def fact(self, i):
        s, r = self.keys[i]
        return s, r, self.link[(s, r)]

    def drift(self, rng, ids, p):
        """Some facts of the component change value before it is written (the world moves on = an update)."""
        for i in ids:
            if rng.random() < p:
                s, r = self.keys[i]
                pool = (self.people if r in W.PERSON_RELS else
                        self.V[W.ATTRS[r]] if r in W.ATTRS else self.V["nums"])
                self.link[(s, r)] = rng.choice([v for v in pool if v not in (s, self.link[(s, r)])])

    def chain(self, start, rels):
        """Question text, answer, and the path of fact ids for start's rels chain on the CURRENT world."""
        cur, path = start, []
        for r in rels:
            path.append(self.idx[(cur, r)])
            cur = self.link[(cur, r)]
        wh = "Who" if rels[-1] in W.PERSON_RELS else "What"
        return f"{wh} is {start}'s " + "'s ".join(rels) + "?", cur, path


def sample_rels(rng, d):
    return tuple([rng.choice(W.PERSON_RELS) for _ in range(d - 1)] + [rng.choice(W.PERSON_RELS + tuple(W.ATTRS))])


def main():
    P = argparse.ArgumentParser()
    P.add_argument("--hf", default="gpt2")
    P.add_argument("--entities", type=int, default=300)
    P.add_argument("--vars", type=int, default=15)
    P.add_argument("--train", choices=("fresh", "global"), default="fresh",
                   help="fresh: train the skill on a new random world each step (no memorization possible), evaluate "
                        "on the persistent global world; global: train on the global world itself (memorizes, g9)")
    P.add_argument("--fresh-people", type=int, default=12, help="people per fresh training world")
    P.add_argument("--doc-records", type=int, default=12, help="records per written document (fresh worlds)")
    P.add_argument("--pool", type=int, default=1000, help="entity-name vocabulary for fresh worlds")
    P.add_argument("--comp", type=int, default=4, help="records per written component (the subworld)")
    P.add_argument("--drift", type=float, default=0.05, help="per fact: chance its value changed when re-written")
    P.add_argument("--refresh", type=int, default=200, help="re-encode the whole store every N steps (no grad)")
    P.add_argument("--steps", type=int, default=4000)
    P.add_argument("--depths", default="1,2,3,4,5")
    P.add_argument("--hops", type=int, default=6)
    P.add_argument("--nq", type=int, default=32)
    P.add_argument("--cloze", type=int, default=16)
    P.add_argument("--heldout", type=int, default=400, help="question chains never asked in training (eval)")
    P.add_argument("--dk", type=int, default=128)
    P.add_argument("--lr", type=float, default=1e-3)
    P.add_argument("--inject", type=int, default=-2)
    P.add_argument("--eval-every", type=int, default=500)
    P.add_argument("--seed", type=int, default=0)
    P.add_argument("--save", default="runs/fwm_global.pt")
    P.add_argument("--load", default="", help="continue from a saved module (size curriculum)")
    P.add_argument("--sup", choices=("cot", "il", "answer"), default="cot",
                   help="cot: clean per-hop CoT targets + halt = CoT length; il: imitation from (messy) CoT text (hop "
                        "readout must name an entity of its aligned sentence) + answer soft-min; answer: answer only")
    P.add_argument("--cot-noise", type=float, default=1.0, help="il: messiness of the CoT demonstrations")
    P.add_argument("--il-w", type=float, default=1.0)
    A = P.parse_args()
    import gnm_small
    import gnm_v2 as V2
    import gnm_lm_1b as G1
    import gmem_ltm as LT
    import train_walker_1b_continue as T
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(A.seed)
    model = gnm_small.SmallWalker(A.hf).to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    hf_tok = T._load_tokenizer(A.hf)
    tok = LT.Tok(hf_tok)
    enc = V2.WalkerStateEnc(model, tok)
    d = model.emb.weight.shape[1]
    V = W.vocab(tok, G1.single_token_names(tok, 40))
    avoid = set(V["cities"]) | {"Who", "What", "Answer", "Also", "The", "So", "Let", "First", "Hmm", "Okay", "We"}
    pool = entity_names(hf_tok, tok, max(A.entities, A.pool), A.seed, avoid)
    world = World(random.Random(A.seed), V, pool[:A.entities], V["vars"][:A.vars])
    N = len(world.keys)
    depths = [int(x) for x in A.depths.split(",")]

    mem = W.FastWeightMemory(d, A.dk, A.hops, tie=True, center=False, tok_emb=True, act="softmax", tau0=10.0,
                             wn_mlp=True, tau_hop0=3.0).to(dev)
    gate = nn.Parameter(torch.tensor(1.0, device=dev))
    if A.load:
        st = torch.load(A.load, map_location=dev)
        mem.load_state_dict(st["mem"])
        with torch.no_grad():
            gate.copy_(st["gate"])
        print(f"loaded {A.load} (step {st.get('step')})", flush=True)
    opt = torch.optim.AdamW(list(mem.parameters()) + [gate], lr=A.lr, weight_decay=0.01)
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

    # ---------------------------------------------------------------- the persistent store (the memory)
    K = torch.zeros(N, A.dk, device=dev)
    Vs = torch.zeros(N, A.dk, device=dev)
    LG = torch.full((N,), -20.0, device=dev)                # unwritten = off

    def encode(subjects, wld=None):
        wld = wld or world
        ids = [i for s in subjects for i in wld.rec[s]]
        facts = [wld.fact(i) for i in ids]
        pieces, at, parent = W.json_pieces(facts)
        with torch.no_grad():
            X, rt = enc.doc(pieces)
            E = model.emb(torch.tensor([t for pc in pieces for t in tok.ids(pc)], device=dev)).float()
        spans = [list(range(*rt[parent[j]])) + list(range(*rt[at[j]])) for j in range(len(ids))]
        _, _, v, g = mem.write(X.to(dev), spans, E)
        lg = torch.log(g + 1e-6)
        # RAW keys go into the store; the read centres them by the store's own mean (g1 without centring: every key
        # shared one offset, the read over 1515 units could not discriminate, held-out chains at chance)
        return torch.tensor(ids, device=dev), mem.last_kraw, v, lg, g

    def refresh():
        subs = world.people + world.vars
        with torch.no_grad():
            for j in range(0, len(subs), A.comp * 4):
                ids, k, v, lg, _ = encode(subs[j:j + A.comp * 4])
                K[ids], Vs[ids], LG[ids] = k, v, lg

    def encode_q(texts):
        seqs = [tok.ids(" " + t) for t in texts]
        n = max(len(s) for s in seqs)
        ids = torch.tensor([s + [0] * (n - len(s)) for s in seqs], device=dev)
        m = torch.tensor([[1] * len(s) + [0] * (n - len(s)) for s in seqs], device=dev, dtype=torch.bool)
        with torch.no_grad():
            return enc._states(ids).float(), m, model.emb(ids).float()

    def centred(Kraw, Vv, lgv):
        km = Kraw.mean(0, keepdim=True)
        mem.kmean = km                                      # queries use the same mean (FastWeightMemory.read)
        return F.normalize(Kraw - km, dim=-1), Vv, lgv

    def answer_logits(R, off=False):
        B, Th = R.shape[0], R.shape[1]
        state["add"] = None if off else mem.Wo(R.reshape(B * Th, -1))
        lg, _ = model(torch.tensor([ans_ids] * (B * Th), device=dev), r_active=False)
        state["add"] = None
        return lg[:, -1].float().view(B, Th, -1)

    # held-out question chains: never asked in training
    hrng = random.Random(A.seed + 1)
    held = set()
    while len(held) < A.heldout:
        held.add((hrng.choice(world.people), sample_rels(hrng, hrng.choice(depths))))
    held = sorted(held)
    tid = lambda t: tok.ids(" " + t)[0]

    @torch.no_grad()
    def evaluate(step):
        mem.eval()
        refresh()
        M = centred(K, Vs, LG)
        out = {}
        saved = dict(world.link)
        for name, off in (("held-out chains", False), ("memory off", True), ("counterfactual world", False)):
            if name == "counterfactual world":
                # half the facts get new values, the store is re-encoded: a model that READS memory follows the new
                # values, one that memorized the (fixed) world in its weights answers the old ones
                world.drift(random.Random(step), list(range(N)), 0.5)
                refresh()
                M = centred(K, Vs, LG)
            S = {}
            for j in range(0, len(held), 64):
                qs = [world.chain(s, rels) for s, rels in held[j:j + 64]]
                Hq, mq, Eq = encode_q([q for q, _, _ in qs])
                _, R, pi = mem.read(M, Hq, mq, Eq)
                lg = answer_logits(R, off)
                y = torch.tensor([tid(a) for _, a, _ in qs], device=dev)
                pick = lg[torch.arange(len(qs), device=dev), pi.argmax(-1)].argmax(-1)
                best = (lg.argmax(-1) == y[:, None]).any(-1)
                for b, (_, _, path) in enumerate(qs):
                    e = S.setdefault(len(path), [0, 0, 0])
                    e[0] += 1
                    e[1] += int(pick[b] == y[b])
                    e[2] += int(best[b])
            ds = sorted(S)
            out[name] = [S[k][1] / S[k][0] for k in ds]
            print(f"EVAL step {step} {name}: acc by depth " + " ".join(f"{k}:{S[k][1] / S[k][0]:.2f}" for k in ds)
                  + " | best hop " + " ".join(f"{S[k][2] / S[k][0]:.2f}" for k in ds)
                  + f" | n {sum(S[k][0] for k in ds)}", flush=True)
        world.link.clear()
        world.link.update(saved)
        refresh()
        mem.train()
        return out

    print(f"GLOBAL WORLD (sup {A.sup}): {len(world.people)} people + {len(world.vars)} variables = {N} facts (units) | "
          f"component {A.comp} records | drift {A.drift} | held-out chains {len(held)} | depths {depths} "
          f"hops {A.hops}", flush=True)
    if A.steps == 0:                                        # eval only (e.g. capacity: --entities 1000 --load ...)
        evaluate(0)
        return
    mem.train()
    refresh()
    rng = random.Random(A.seed + 2)
    held_set = set(held)
    t0, best = time.time(), -1.0
    for step in range(1, A.steps + 1):
        lr = A.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - 1) / A.steps)))
        for gp in opt.param_groups:
            gp["lr"] = lr
        if step % A.refresh == 0 and A.train == "global":
            # fresh mode never needs the global store while training; refreshing it in train mode also pushed the
            # global world's documents into the running standardization stats (g11: training never took off)
            refresh()
        if A.train == "fresh":
            # TRAINING teaches the read/write SKILL on a fresh random world (people drawn from a big vocabulary):
            # nothing can be memorized in the weights. The persistent global world is only for evaluation.
            # (g9: trained on the fixed world, held-out .67-.99 but counterfactual .43/.25/.16/.14/.13 = memorized)
            tw = World(random.Random(rng.random()), V, rng.sample(pool, A.fresh_people),
                       rng.sample(V["vars"], min(A.vars, 8)))
            subs = tw.people + tw.vars                     # written as several documents (GPT-2: <= ~20 records)
            parts = [encode(subs[j:j + A.doc_records], tw) for j in range(0, len(subs), A.doc_records)]
            ids = torch.cat([x[0] for x in parts])
            k, v, lg_, g = (torch.cat([x[i] for x in parts]) for i in (1, 2, 3, 4))
            order = torch.argsort(ids)                     # unit i = fact i of the fresh world
            ids, k, v, lg_, g = ids[order], k[order], v[order], lg_[order], g[order]
            M = centred(k, v, lg_)
            qw, starts = tw, tw.people
            ids_local = list(range(len(tw.keys)))
        else:
            comp = rng.sample(world.people, A.comp)
            world.drift(rng, [i for s in comp for i in world.rec[s]], A.drift)
            ids, k, v, lg_, g = encode(comp)               # write the component (with gradient)
            with torch.no_grad():
                K[ids], Vs[ids], LG[ids] = k.detach(), v.detach(), lg_.detach()
            M = centred(K.index_copy(0, ids, k), Vs.index_copy(0, ids, v), LG.index_copy(0, ids, lg_))
            qw, starts = world, comp
            ids_local = ids.tolist()
        qs = []
        while len(qs) < A.nq:                              # chains start in the component, run through the world
            s, rels = rng.choice(starts), sample_rels(rng, rng.choice(depths))
            if qw is not world or (s, rels) not in held_set:
                qs.append(qw.chain(s, rels))
        texts = [q for q, _, _ in qs]
        steps_ = [[tid(qw.fact(i)[2]) for i in path] for _, _, path in qs]
        for i in rng.sample(ids_local, min(A.cloze, len(ids_local))):   # masked recall of the written facts
            s, r, t = qw.fact(i)
            texts.append(f"{s}'s {r} is")
            steps_.append([tid(t)])
        Hq, mq, Eq = encode_q(texts)
        _, R, pi = mem.read(M, Hq, mq, Eq)
        lg = answer_logits(R)
        B, Th = R.shape[0], R.shape[1]
        if A.sup == "cot":
            Tg = torch.full((B, Th), -100, dtype=torch.long, device=dev)
            for b, st in enumerate(steps_):                # CoT step t supervises hop t (clean CoT)
                Tg[b, :min(len(st), Th)] = torch.tensor(st[:Th], device=dev)
            dep = torch.tensor([min(len(st), Th) - 1 for st in steps_], device=dev)
            loss = (F.cross_entropy(lg.reshape(B * Th, -1), Tg.reshape(-1), ignore_index=-100)
                    - torch.log(pi[torch.arange(B, device=dev), dep] + 1e-9).mean())
        else:
            # answer: every hop a rollout, soft-min over hops, the halt predicts the winner
            y = torch.tensor([st[-1] for st in steps_], device=dev)
            ce = F.cross_entropy(lg.reshape(B * Th, -1), y.repeat_interleave(Th), reduction="none").view(B, Th)
            w = torch.softmax(-ce.detach(), -1)
            loss = (w * ce).sum(-1).mean() - (w * torch.log(pi + 1e-9)).sum(-1).mean()
            if A.sup == "il":
                # imitation from the CoT TEXT (gnm_fwm --il-mode content): hop t's readout must name an entity its
                # aligned (messy) CoT sentence mentions; candidates = sentence tokens that are stored values
                allf = [qw.fact(i) for i in range(len(qw.keys))]
                vals = {tid(t) for _, _, t in allf}
                logp = torch.log_softmax(lg[:len(qs)], -1)
                margin = math.log(lg.shape[-1]) + 1.0
                terms, n_ok, n_pairs = [], 0, 0
                for b, (q_, a_, path) in enumerate(qs):
                    ss = W.cot_steps(rng, allf, path, a_, A.cot_noise)
                    cands = W.new_info(tok, q_, ss, vals)
                    use = [j for j, c in enumerate(cands) if c]
                    if not use:
                        continue
                    C = torch.stack([-logp[b][:, cands[j]].max(-1).values for j in use], 1)
                    for t_, jj in W.align(C.detach().tolist(), margin):
                        terms.append(C[t_, jj])
                        if t_ < len(path):
                            n_pairs += 1
                            n_ok += int(tid(allf[path[t_]][2]) in cands[use[jj]])
                if terms:
                    loss = loss + A.il_w * torch.stack(terms).mean()
                main.il_ok = n_ok / max(1, n_pairs)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(mem.parameters()) + [gate], 1.0)
        opt.step()
        if step % 50 == 0:
            y = torch.tensor([st[-1] for st in steps_], device=dev)
            acc = (lg[torch.arange(B, device=dev), pi.argmax(-1)].argmax(-1) == y).float()
            dq = torch.tensor([len(st) for st in steps_[:len(qs)]], device=dev)
            print(f"step {step} loss {float(loss):.3f} acc by depth "
                  + " ".join(f"{dd}:{float(acc[:len(qs)][dq == dd].mean()):.2f}" for dd in sorted(set(dq.tolist())))
                  + f" | recall {float(acc[len(qs):].mean()):.2f} | write-gate {float(g.mean()):.2f} "
                  f"tau {float(mem.tau.exp()):.1f} il-ok {getattr(main, 'il_ok', -1):.2f} [{time.time() - t0:.0f}s]", flush=True)
        if step % A.eval_every == 0 or step == A.steps:
            out = evaluate(step)
            score = sum(out["held-out chains"])
            if score > best:
                best = score
                torch.save({"mem": mem.state_dict(), "gate": gate.detach(), "step": step}, A.save)


if __name__ == "__main__":
    main()
