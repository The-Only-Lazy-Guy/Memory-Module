"""GNM <-> Walker 1B pairing (PROJECT_STATUS 3.8): the prototype of gnm_lm.py on the real checkpoint.

Walker 1B is frozen. Its own narrow-R loop (P = blocks 0-5, R = 6-9 looped, C = 10-15) runs r extra
passes; every pass the GNM hook (gnm_lm.GNMHook, shared gate init 0) makes c_ins / c_mem from the
pass state, activates instruction and fact nodes, and the weighted nodes are cross-read into the
pass. Injection point:
  --inject skip (default): s' = s + gain[p] (R(u) - s) + gate * read   (the checkpoint's pass_gain is
                           ~0 from pass 3 on, so a read inside u would not reach the stream);
  --inject u             : u' = u + gate * read (as the tiny prototype).
pass_gain stays frozen, so with no graph attached the LM is bit-identical to the checkpoint.

Task (natural language, single-token names): every world has relations between people ("Alice's
boss is Bob.") as fact nodes under a person record, and procedures as instruction nodes
("Procedure P3, step 2: friend", NEXT -> step 3). Question: "Apply procedure P3 to Alice. Answer:"
and Walker's own head must emit the name reached. Eval on new worlds and held-out step combinations;
ablations gate off and r=1.

    python gnm_lm_1b.py --ckpt /marimo/ckpt/checkpoints/ckpt_ongoing.pt --steps 3000

Step B (harder, GNM.md 4.7): --paraphrase (6 trained question wordings, eval on 3 never trained),
--full-names ("First Last" people, first names repeat: 2-3 token answers, teacher-forced, all tokens
must be right; eval worlds use 10 never-trained surnames), --sup answer (final answer only: no per-pass
gold, no forced candidates, the copy stays attached so the chain learns from the answer).
"""
import argparse
import math
import os
import random
import time

import torch
import torch.nn.functional as F

import gmem_node as N
import gnm_lm as GL

RELS = ("boss", "friend", "neighbor", "mentor", "rival")
NAMES = ("Alice Bob Carol David Emma Frank Grace Henry Jack Kate Laura Mark Nina Oscar Paul Rose Sam Tom "
         "Anna Ben Chris Dan Eric Fred George Harry Ivan James John Kevin Leo Lucy Mary Mike Nick Peter "
         "Ryan Sarah Steve Tony Victor Will Adam Alex Amy Andy Bill Brian Carl Dave Dennis Diana Eva "
         "Gary Helen Jane Jeff Jim Joe Julia Karl Linda Lisa Max Nancy Ruth Simon Susan Ted").split()
NP, NSTEP = 10, 3
COMBOS = [(a, b, c) for a in RELS for b in RELS for c in RELS]
TEST_COMBOS = set(random.Random(0).sample(COMBOS, len(COMBOS) // 5))
TRAIN_COMBOS = [c for c in COMBOS if c not in TEST_COMBOS]


PLAIN = [  # generic prompts with no procedure: memory must stay silent (do-no-harm episode)
    "The weather today is", "In the morning, the city", "Water boils at a temperature of", "The best way to learn is",
    "Scientists discovered that the", "The old house on the hill", "When the train arrived, the", "A good book can",
    "The history of the region shows", "After the rain stopped, the", "Most people believe that", "The recipe calls for",
    "In mathematics, a prime number", "The company announced that it", "The river flows through the", "Music has the power to",
    "During the winter months, the", "The museum is open from", "One of the main reasons is", "The children played in the",
    "Computers process information by", "The mountain was covered in", "Every year, thousands of", "The meeting will start at",
    "The theory of evolution explains", "Her favorite color is", "The bridge was built in", "The engine started and the",
    "The answer to the question is", "In the end, the team", "The ocean covers most of the", "A healthy diet includes",
    "The library has a large collection of", "The sun rises in the", "The price of the ticket was", "The village is known for",
    "The doctor told the patient that", "The road leads to the", "At night, the stars", "The garden was full of",
    "The first step in the process is", "The teacher asked the class to", "The festival takes place every", "The dog barked at the",
    "The results of the experiment were", "The window was open and the", "The language spoken there is", "The new policy will",
]
PLAIN_TRAIN, PLAIN_EVAL = PLAIN[:36], PLAIN[36:]

# Step B (GNM.md 4.7): harder versions of the same task, each behind a flag (CFG, set in main)
SURNAMES = ("Smith Jones Brown Taylor Wilson Davies Evans Thomas Johnson Roberts Walker Wright Robinson "
            "Thompson White Hughes Edwards Green Hall Wood Harris Lewis Martin Jackson Clarke Turner Hill "
            "Scott Cooper Morris Ward Moore King Watson Baker Harrison Morgan Patel Young Bennett").split()
SURN_TRAIN, SURN_EVAL = SURNAMES[:30], SURNAMES[10:]      # eval: 20 trained surnames + 10 never seen
WORDING = {   # question paraphrases; "train"[0] is the original wording. Eval asks in wording never trained
    "train": ("Apply procedure P{j} to {x}. Answer:",
              "Run procedure P{j} starting from {x}. Answer:",
              "Start at {x} and follow procedure P{j}. Answer:",
              "Take {x} through procedure P{j}. Who do you reach? Answer:",
              "Procedure P{j}, applied to {x}, gives whom? Answer:",
              "Begin with {x}; carry out procedure P{j}. Answer:"),
    "held": ("Follow procedure P{j} from {x}. Answer:",
             "Where does {x} end up after procedure P{j}? Answer:",
             "Using {x} as the start, execute procedure P{j}. Answer:"),
}
CFG = {"paraphrase": False,   # train on WORDING["train"], eval on WORDING["held"]
       "full_names": False,   # people are "First Last" (first names repeat): answers are 2-3 tokens
       "sup": "chain",        # chain: per-pass gold facts / instructions; answer: final answer only
       "k": 1,                # training runs per question (noisy trajectories; loss on the best)
       "qmem": False,         # the question lives in memory, Walker sees only " Answer:" (needs --link)
       "label_only": False}   # qmem: question -> entity / procedure by learned label match (no word overlap)


def single_token_names(tok, n):
    out = [nm for nm in NAMES if len(tok.ids(" " + nm)) == 1]
    if len(out) < n:
        raise SystemExit(f"only {len(out)} single-token names")
    return out[:n]


def make_world(rng, split, names, proc_layout="flat", nstep=NSTEP, var_len=False):
    if CFG["full_names"]:      # first names repeat across people, so only the whole name identifies one
        surn = SURN_TRAIN if split == "train" else SURN_EVAL
        pairs = rng.sample([(a, b) for a in names[:max(2, len(names) // 2)] for b in surn], len(names))
        names = [f"{a} {b}" for a, b in pairs]
    E = len(names)
    f = {rel: [rng.randrange(E) for _ in range(E)] for rel in RELS}
    pool = TRAIN_COMBOS if split == "train" else sorted(TEST_COMBOS)

    def proc():                                          # length 3 keeps the train / held-out split
        L = rng.randint(1, 4) if var_len else nstep
        return rng.choice(pool) if L == NSTEP else tuple(rng.choice(RELS) for _ in range(L))
    procs = [proc() for _ in range(NP)]
    G = N.Graph(cap=10 ** 6)
    fid, sid, nxt = {}, {}, []
    for e in range(E):
        r = G.add(names[e], "record")
        for rel in RELS:
            fid[(e, rel)] = G.add(f"{rel}: {names[f[rel][e]]}", "leaf", ptr=r)   # subject = the parent
    demo = {}
    for j, combo in enumerate(procs):
        prev = -1
        if proc_layout == "example":   # no procedure given: one worked example, on another person
            hdr = G.add(f"Example of procedure P{j}", "ins")
            demo[j] = cur = rng.randrange(E)
        else:
            hdr = G.add(f"Procedure P{j}", "ins") if proc_layout == "record" else -1   # steps point to it
        for k, rel in enumerate(combo):
            if proc_layout == "example":
                i = G.add(f"step {k + 1}: {names[cur]}'s {rel} is {names[f[rel][cur]]}", "ins", ptr=hdr)
                cur = f[rel][cur]
            else:
                i = (G.add(f"step {k + 1}: {rel}", "ins", ptr=hdr) if hdr >= 0
                     else G.add(f"Procedure P{j}, step {k + 1}: {rel}", "ins"))
            sid[(j, k)] = i
            if prev >= 0:
                nxt.append((prev, i))
            prev = i
        d = (G.add(f"step {len(combo) + 1}: done", "ins", ptr=hdr) if hdr >= 0
             else G.add(f"Procedure P{j}, step {len(combo) + 1}: done", "ins"))
        sid[(j, len(combo))] = d
        nxt += [(prev, d), (d, d)]
    return {"f": f, "procs": procs, "G": G, "fid": fid, "sid": sid, "nxt": nxt, "names": names, "demo": demo}


def questions(rng, w, B, r, tok, wording=None, texts=None):
    """ans[b] = the answer's token ids (one id unless --full-names). wording: None = the original
    question, "train" / "held" = a random paraphrase from WORDING. texts: list that receives the
    question strings (for --qmem)."""
    names = w["names"]
    seqs, ans, gm, gi = [], [], [[] for _ in range(r)], [[] for _ in range(r)]
    for _ in range(B):
        j, x = rng.randrange(NP), rng.randrange(len(names))
        while w.get("demo") and x == w["demo"].get(j):     # ask about a different person than the example
            x = rng.randrange(len(names))
        cur = x
        for k, rel in enumerate(w["procs"][j]):
            if k < r:
                gm[k].append(w["fid"][(cur, rel)])
                gi[k].append(w["sid"][(j, k)])
            cur = w["f"][rel][cur]
        n = len(w["procs"][j])
        for k in range(n, r):
            gm[k].append(gm[n - 1][-1])
            gi[k].append(w["sid"][(j, n)])
        tmpl = rng.choice(WORDING[wording]) if wording else WORDING["train"][0]
        seqs.append(tok.ids(" " + tmpl.format(j=j, x=names[x])))
        if texts is not None:
            texts.append(tmpl.format(j=j, x=names[x]).replace(" Answer:", ""))
        ans.append(tok.ids(" " + names[cur]))
    return seqs, ans, gm, gi


def attach_skip(model, hook, rms_match=True):
    """Skip-path injection: the hook's read is added after the pass update, not inside u.
    rms_match: the read is scaled to the stream's per-position RMS (gate = fraction of the stream);
    a raw 128->2048 projection is tiny next to the 1B's residual stream, so the LM never felt it."""
    orig_in, orig_up = model._r_pass_input, model._r_pass_update
    state = {}

    def wrapped_in(r_adapter, s_in, e_state):
        u = orig_in(r_adapter, s_in, e_state)
        p = hook.p
        hook.p += 1
        c = hook.ctx
        if c is None:
            state["add"] = None
            return u
        with torch.autocast(s_in.device.type, enabled=False):
            out = hook._read(s_in.float(), c, p)
        if rms_match:
            with torch.no_grad():
                scale = s_in.float().pow(2).mean(-1, keepdim=True).sqrt()
            out = out / out.pow(2).mean(-1, keepdim=True).sqrt().clamp(min=1e-6) * scale
        _ro = getattr(hook, "read_open", None)
        if _ro:
            # FIXED, unbounded, non-decaying read: the read always reaches the LM at scale `read_open`
            # (a feature, not a trained gate that can collapse to 0 and starve the answer path).
            state["add"] = float(_ro) * out
        else:
            g = hook.gate[min(p, hook.gate.numel() - 1)]
            state["add"] = (torch.tanh(g) if rms_match else g) * out   # bounded: |gate| < 1 of the stream
        return u

    def wrapped_up(s_in, u, yb, gain):
        x = orig_up(s_in, u, yb, gain)
        add = state.pop("add", None)
        return x if add is None else x + add.to(x.dtype)
    model._r_pass_input, model._r_pass_update = wrapped_in, wrapped_up
    return model


def halted_trace(hook, w, B, dev):
    """Per sample, the trace of the last pass before the procedure reaches its own 'done' node (the
    active instruction is the NEXT successor of the previous one, and it is a done node); the final
    pass if it never gets there. (Any done node is not enough: another procedure's "step 4: done" can
    look like this one's "step 4: boss".)"""
    G = w["G"]
    if "done" not in w:
        w["done"] = torch.tensor([k == "ins" and t.endswith("done") for k, t in zip(G.kind, G.text)], device=dev)
        succ = torch.full((len(G),), -1, dtype=torch.long, device=dev)
        for i, j in w["nxt"]:
            succ[i] = j
        w["succ"] = succ
    top = [t["a_ins"].argmax(-1) for t in hook.trace]
    done = torch.stack([torch.zeros_like(top[0], dtype=torch.bool)] +
                       [w["done"][top[p]] & (w["succ"][top[p - 1]] == top[p]) & (top[p - 1] != top[p])
                        for p in range(1, len(top))])          # a real step into this procedure's done
    r = done.shape[0]
    first = torch.where(done.any(0), done.float().argmax(0), torch.full((B,), r, device=dev))
    sel = (first - 1).clamp(min=0)
    ar = torch.arange(B, device=dev)
    return {"a_mem": torch.stack([t["a_mem"] for t in hook.trace])[sel, ar],
            "h": torch.stack([t["h"] for t in hook.trace])[sel, ar]}


def node_tokens(w, tok, dev):
    """The token a node 'says' when pointed at: the last token of its text ("boss: Bob" -> " Bob")."""
    if "tok" not in w:
        w["tok"] = torch.tensor([tok.ids(" " + t)[-1] for t in w["G"].text], device=dev)
    return w["tok"]


def node_token_seqs(w, tok, dev):
    """All the tokens a node says: its value ("boss: Bob Smith" -> " Bob", " Smith"), padded (N, L),
    with a mask of the real ones. For single-token values the same token as node_tokens."""
    if "tseq" not in w:
        seqs = [tok.ids(" " + t.split(": ", 1)[-1]) for t in w["G"].text]
        L = max(len(s) for s in seqs)
        w["tseq"] = (torch.tensor([s + [0] * (L - len(s)) for s in seqs], device=dev),
                     torch.tensor([[1.0] * len(s) + [0.0] * (L - len(s)) for s in seqs], device=dev))
    return w["tseq"]


def run_batch(T, model, hook, M, w, rng, B, dev, train, tok, rn, r=NSTEP, gate_off=False, copy_off=False,
              wording=None, qa=None, no_ins=False, ungated=False, selfsup=False, selfsup_len=0.02,
              explore=1.0):
    """gate_off: no state injection (the copy path still reads the GNM). gate_off + copy_off: GNM fully
    off at the output (plain Walker answer). A multi-token answer is teacher-forced (question + answer
    minus its last token) and counts as right only if every token is.
    qa: (seqs, ans, gm, gi) given by the caller (real data) instead of generated; gi[p][b] = -1 means no
    gold instruction for that sample. no_ins: the hook ignores instruction memory (ablation)."""
    qtexts = []
    seqs, ans, gm, gi = qa[:4] if qa is not None else questions(rng, w, B, r, tok, wording, qtexts)
    if qa is not None and len(qa) > 4:
        qtexts = list(qa[4])
    k = CFG["k"] if train else 1
    if k > 1:                       # k runs of every question, each with its own noisy trajectory
        seqs, ans, gm, gi, B = seqs * k, ans * k, [g * k for g in gm], [g * k for g in gi], B * k
        qtexts = qtexts * k
    qn = None
    if CFG["qmem"]:                 # every asked question becomes a memory node; Walker sees no question
        G = w["G"]
        qn = [G.add("Question: " + t, "question") for t in qtexts]
        for cache in ("tok", "tseq", "leaf", "ntok", "idf", "done", "succ"):
            w.pop(cache, None)
        seqs = [tok.ids(" Answer:")] * B
        gm, gi, r = [qn] + gm, [[-1] * B] + gi, r + 1        # pass 1 = find the question
    ids = rn.pad([q + a[:-1] for q, a in zip(seqs, ans)])
    ctx = GL.graph_ctx(M, w, dev, B)
    ctx["cue_pos"] = torch.tensor([len(s) - 1 for s in seqs], device=dev)
    ctx["no_ins"] = no_ins = no_ins or ctx["no_ins"]
    ctx["noise"] = k > 1
    ctx["noise_scale"] = explore                         # proposal temperature (k-rollout exploration)
    if qn is not None:
        G = w["G"]
        ctx["qnodes"] = torch.tensor(qn, device=dev)
        rec = torch.zeros(B, len(G), device=dev)
        rec[:, ctx["qnodes"]] = torch.rand(B, len(qn), device=dev) * 0.5    # older questions
        rec[torch.arange(B, device=dev), ctx["qnodes"]] = 1.0             # this sample's question: newest
        ctx["recency"] = rec
        if not CFG["label_only"]:   # rule-made mention edges (word overlap); label-only: the query finds them
            lex_q = GL.lex_matrix([tok.ids(" " + G.text[i]) for i in qn], w, tok, dev)
            ctx["M_rec"], ctx["M_ins"] = GL.start_pos(lex_q, G), GL.start_ins(lex_q, G, w["nxt"])
    elif hook.link:
        lex = GL.lex_matrix(seqs, w, tok, dev)
        ctx["pos0"] = GL.start_pos(lex, w["G"])
        ctx["pos0_ins"] = GL.start_ins(lex, w["G"], w["nxt"])
        ctx["has_ins"] = ctx["pos0_ins"].sum(-1, keepdim=True).clamp(max=1)
    if train and CFG["sup"] == "chain":
        ctx["force_mem"] = gm
    saved = None
    if gate_off:
        saved = hook.gate.data.clone()
        hook.gate.data.zero_()
    with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
        logits = GL.forward(T, model, hook, ids, ctx, r)
    if saved is not None:
        hook.gate.data.copy_(saved)
    ar = torch.arange(B, device=dev)
    L = max(len(a) for a in ans)
    y = torch.tensor([a + [-100] * (L - len(a)) for a in ans], device=dev)                   # (B, L)
    pos = (ctx["cue_pos"][:, None] + torch.arange(L, device=dev)[None]).clamp(max=logits.shape[1] - 1)
    last = logits[ar[:, None], pos].float()                                                   # (B, L, V)
    copy = getattr(hook, "copy_gate", None)
    if copy is not None and hook.trace and not copy_off:
        # say = point at a node: the activated fact nodes give a copy distribution over their value
        # tokens, mixed with Walker's own next-token distribution by a learned gate. Halt on done: the
        # pass that says is the last one before the instruction memory reaches a "done" node.
        t = hook.trace[-1] if hook.soft else halted_trace(hook, w, B, dev)   # soft: fixed r, say at the end
        if hook.soft and getattr(hook, "halt", None) is not None:   # soft halting: say from a learned mix of
            tr = hook.trace[1 if CFG["qmem"] else 0:]                # passes (answers at different depths)
            pi = torch.softmax(torch.cat([hook.halt(torch.cat([x["h"], x["r"]], -1)) for x in tr], -1), -1)
            t = {"a_mem": sum(pi[:, i:i + 1] * x["a_mem"] for i, x in enumerate(tr)), "h": tr[-1]["h"]}
            run_batch.halt = [round(float(v), 2) for v in pi.mean(0)]
        # the copy reads the activation as given: choosing the node is the chain supervision's job (an
        # attached copy loss rewards activating ANY node that says the answer and, from scratch, wrecks
        # the lookup through the shared cue / label parameters). With --sup answer the copy IS the only
        # signal the chain gets, so it stays attached.
        a = t["a_mem"]
        if hook.soft:                                   # only facts say: a record would repeat its own entity
            if "leaf" not in w:
                w["leaf"] = torch.tensor([k == "leaf" for k in w["G"].kind], device=dev, dtype=torch.float)
            a = a * w["leaf"]
        a = a / a.sum(-1, keepdim=True).clamp(min=1e-9)
        if CFG["sup"] == "chain":
            a = a.detach()
        tid, tm = node_token_seqs(w, tok, dev)          # token k of the answer = token k of the node's value
        p_copy = torch.stack([torch.zeros(B, last.shape[-1], device=dev).scatter_add_(
            1, tid[:, k].expand(B, -1), a * tm[:, k]) if k < tid.shape[1] else
            torch.zeros(B, last.shape[-1], device=dev) for k in range(L)], 1)
        alpha = torch.sigmoid(copy(t["h"])).clamp(1e-4, 1 - 1e-4)[:, None]                   # (B, 1, 1)
        last = torch.log((1 - alpha) * torch.softmax(last, -1) + alpha * p_copy + 1e-9)
        run_batch.alpha = float(alpha.mean())
    tl = F.cross_entropy(last.flatten(0, 1), y.flatten(), ignore_index=-100, reduction="none").view(B, L)
    per = tl.sum(1) / (y >= 0).sum(1).clamp(min=1)                  # per question
    lm = per.view(k, -1).min(0).values.mean()                       # k > 1: learn from the best trajectory
    ds, hit_m, hit_i = 0.0, [], []
    if selfsup:
        # SELF-SUPERVISED retrieval: reward activation MASS on ANSWER-containing nodes (uses only the
        # answer `ans`, no per-pass gold), minus a sharpness/length penalty. This is the containment
        # objective from the proof: no path labels, verifiable reward, minimality penalty.
        ntok = node_tokens(w, tok, dev)                              # (N,) last token of each node
        ans0 = torch.tensor([int(a[0]) for a in ans], device=dev)    # single-token answers (length B*k)
        Yf = (ntok[None, :] == ans0[:, None]).float()                # (B, N) 1 on answer-containing nodes
        t = hook.trace[-1]
        am = t["a_mem"]                                              # (B, N); B already *= k above
        cont = (am * Yf).sum(-1)                                     # containment mass per rollout
        tgt = am.masked_fill(Yf < 0.5, -1.0).argmax(-1)              # current-best answer node (hard-EM)
        sft = -am.gather(1, tgt[:, None]).clamp_min(1e-8).log().squeeze(1)
        ent = -(am.clamp_min(1e-8).log() * am).sum(-1)
        if k > 1:                                                    # k noisy rollouts (Gumbel): keep the best
            cont = cont.view(k, -1).max(0).values
            sft = sft.view(k, -1).min(0).values
            ent = ent.view(k, -1).min(0).values
        ds = (sft + selfsup_len * ent).mean()
        run_batch.cont = float((cont > 0).float().mean())           # any answer mass found (best rollout)
        # measured-only diagnostics (these never touch the loss): per-pass fact / instruction top-1 on the
        # gold chain, so a selfsup run still shows whether the retrieval chain is forming at all.
        ar = torch.arange(B, device=dev)
        for _p, _tp in enumerate(hook.trace[:r]):
            _gmt, _git = torch.tensor(gm[_p], device=dev), torch.tensor(gi[_p], device=dev)
            hit_m.append(float((_tp["a_mem"].argmax(-1) == _gmt).float().mean()))
            _has = (_git >= 0).float()
            hit_i.append(float(((_tp["a_ins"].argmax(-1) == _git).float() * _has).sum() / _has.sum())
                         if _has.sum() > 0 else -1.0)
    else:
        chain = torch.ones(B, device=dev)
        for p, t in enumerate(hook.trace[:r]):
            gmt, git = torch.tensor(gm[p], device=dev), torch.tensor(gi[p], device=dev)
            has_i = (git >= 0).float()                        # samples with a gold instruction step
            if CFG["sup"] == "chain":                         # answer: the gold chain is only measured
                nll = -torch.log(t["a_mem"][ar, gmt].clamp(min=1e-9))
                ds = ds + (nll * chain).sum() / B
                if not no_ins:
                    nll_i = -torch.log(t["a_ins"][ar, git.clamp(min=0)].clamp(min=1e-9))
                    ds = ds + (nll_i * has_i).sum() / has_i.sum().clamp(min=1) / r
            am = t["a_mem"].argmax(-1)
            hit = (am == gmt).float()
            if not ungated:                                   # warmup: supervise every pass (no chain mask)
                chain = chain * hit
            hit_m.append(float(hit.mean()))
            if p == 0:                                         # where pass-1 errors are: person or relation
                par = ctx["par"]
                run_batch.diag = (float((par[am] == par[gmt]).float().mean()),
                                  float(sum(w["G"].text[int(a)].split(":")[0] == w["G"].text[int(g)].split(":")[0]
                                            for a, g in zip(am, gmt)) / B),
                                  float(sum(w["G"].kind[int(a)] == "record" for a in am) / B))
            hit_i.append(float(((t["a_ins"].argmax(-1) == git).float() * has_i).sum() / has_i.sum())
                         if has_i.sum() > 0 else -1.0)
    acc = float(((last.argmax(-1) == y) | (y < 0)).all(-1).float().mean())
    return lm, ds, acc, hit_m, hit_i


def plain_batch(T, model, hook, M, w, rng, B, dev, tok, rn, prompts, r=NSTEP):
    """Do-no-harm: generic prompts with a graph attached must keep Walker's own distribution.
    Returns (KL all positions without copy + KL last position with the copy mix, mean alpha)."""
    seqs = [tok.ids(" " + rng.choice(prompts)) for _ in range(B)]
    ids = rn.pad(seqs)
    pos = torch.tensor([len(q) - 1 for q in seqs], device=dev)
    mask = (torch.arange(ids.shape[1], device=dev)[None] <= pos[:, None]).float()
    with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
        ref = GL.forward(T, model, hook, ids, None, r).float()          # Walker alone (no graph)
    ctx = GL.graph_ctx(M, w, dev, B)
    ctx["cue_pos"] = pos
    with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
        out = GL.forward(T, model, hook, ids, ctx, r).float()
    lp_ref = torch.log_softmax(ref, -1)
    kl_all = ((lp_ref.exp() * (lp_ref - torch.log_softmax(out, -1))).sum(-1) * mask).sum() / mask.sum()
    ar = torch.arange(B, device=dev)
    last = out[ar, pos]
    alpha = torch.zeros(B, 1, device=dev)
    copy = getattr(hook, "copy_gate", None)
    if copy is not None and hook.trace:
        t = hook.trace[-1]
        a = (t["a_mem"] / t["a_mem"].sum(-1, keepdim=True).clamp(min=1e-9)).detach()
        p_copy = torch.zeros_like(last).scatter_add_(1, node_tokens(w, tok, dev).expand(B, -1), a)
        alpha = torch.sigmoid(copy(t["h"])).clamp(1e-4, 1 - 1e-4)
        last = torch.log((1 - alpha) * torch.softmax(last, -1) + alpha * p_copy + 1e-9)
    else:
        last = torch.log_softmax(last, -1)
    kl_last = (lp_ref[ar, pos].exp() * (lp_ref[ar, pos] - last)).sum(-1).mean()
    return kl_all + kl_last, float(alpha.mean())


@torch.no_grad()
def depth_table(T, model, hook, M, dev, tok, rn, names, proc_layout, n_worlds=8, B=32, max_steps=6):
    """Answer and per-pass fact top-1 for procedures of 1..max_steps steps, at r = steps and steps + 2."""
    hook.eval()
    for L in range(1, max_steps + 1):
        for extra in (0, 2):
            r = L + extra
            if r > model.r_adapter.pass_gain.numel():
                continue
            rng = random.Random(9000 + L)
            accs, hms = [], []
            for _ in range(n_worlds):
                w = make_world(rng, "eval", names, proc_layout, nstep=L)
                _, _, acc, hm, _ = run_batch(T, model, hook, M, w, rng, B, dev, False, tok, rn, r=r,
                                             wording="held" if CFG["paraphrase"] else None)
                accs.append(acc)
                hms.append(hm)
            facts = [round(sum(h[p] for h in hms) / len(hms), 2) for p in range(L)]
            print(f"DEPTH steps {L} r={r}: answer {sum(accs) / len(accs):.2f} | fact top-1 per pass {facts}",
                  flush=True)


@torch.no_grad()
def evaluate(T, model, hook, M, dev, tok, rn, names, step, n_worlds=12, B=32, proc_layout="flat", extra=0):
    hook.eval()
    out = {}
    held = {"wording": "held"} if CFG["paraphrase"] else {}            # held-out worlds AND wording
    arms = [("held-out", held), ("gate off", {"gate_off": True, **held}),
            ("GNM off", {"gate_off": True, "copy_off": True, **held}), ("r=1", {"r": 1, **held})]
    if CFG["paraphrase"]:                                              # held-out worlds, trained wording
        arms.insert(1, ("trained wording", {"wording": "train"}))
    for name, kw in arms:
        rng = random.Random(4242)
        accs, hms, his = [], [], []
        for _ in range(n_worlds):
            w = make_world(rng, "eval", names, proc_layout)
            _, _, acc, hm, hi = run_batch(T, model, hook, M, w, rng, B, dev, False, tok, rn, **{"r": NSTEP + extra, **kw})
            accs.append(acc)
            hms.append(hm)
            his.append(hi)
        mean = lambda xs: sum(xs) / max(1, len(xs))
        out[name] = mean(accs)
        diag = getattr(run_batch, "diag", (0.0, 0.0, 0.0))
        if name == "held-out" and getattr(hook, "keep_lm", False):
            rng2 = random.Random(77)
            kls, als = [], []
            for _ in range(4):
                w = make_world(rng2, "eval", names, proc_layout)
                kl, al = plain_batch(T, model, hook, M, w, rng2, B, dev, tok, rn, PLAIN_EVAL)
                kls.append(float(kl))
                als.append(al)
            print(f"EVAL step {step} plain prompts (held-out): KL to Walker {mean(kls):.4f} | copy alpha "
                  f"{mean(als):.3f}", flush=True)
        print(f"EVAL step {step} {name}: answer {mean(accs):.2f} | fact top-1 "
              f"{[round(mean([h[p] for h in hms if p < len(h)]), 2) for p in range(NSTEP + CFG['qmem'])]} | instruction "
              f"{[round(mean([h[p] for h in his if p < len(h)]), 2) for p in range(NSTEP + CFG['qmem'])]} | pass-1 right person "
              f"{diag[0]:.2f} right relation {diag[1]:.2f} record itself {diag[2]:.2f} (last world)", flush=True)
    hook.train()
    return out


LOCAL_ENV = {   # a small local Walker (no 1B checkpoint) for prototyping on a laptop GPU
    "TB_ATTN_EVERY": "1", "TB_HYBRID": "1", "TB_WALK_LOOP": "1",
    "TB_WALK_NORM": "0.3", "TB_WALK_NORM_MODE": "clip", "TB_WALK_HEADS": "4",
    "TB_WALK_FP32": "1", "TB_WALK_MIX": "0", "TB_WALK_GLOB": "1",
    "TB_WALK_COUPLE": "1", "TB_WALK_GATE": "1", "TB_PHASE_B_SUPERSET": "0",
    "TB_POS_EMB": "0", "TB_FORCE_QUANT": "none", "TB_MEM": "0", "TB_NI": "0",
    "TB_LMM": "0", "TB_KMERGE": "0", "TB_LEXICAL_SEMANTIC": "0",
    "TB_LOCAL_CONV": "0", "TB_LOOP_ENABLE": "0", "TB_R_GAIN": "per_pass",
    "TB_R_ENABLE": "1", "TB_R_MEAN": "3", "TB_R_MAX": "8", "TB_R_BACKPROP": "3",
    "TB_R_INSTR_ONLY": "0", "TB_R_KV_SHARE": "0",
}


def _rebuild_with_r(T, A, dev, cfg):
    """A checkpoint whose narrow-R adapter was retired (a continuation run forced R off) still has
    every block weight; only the RAdapter sidecar is gone. Rebuild the model WITH the R loop (fresh,
    zero-init RAdapter => extra passes start as identity) and load the saved weights strict=False, so
    the GNM hook has an R loop to attach to."""
    raw = T._load_ckpt_raw(A.ckpt, torch.device("cpu"))
    sd = raw.get("model", raw) if isinstance(raw, dict) else raw
    sd = {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v for k, v in sd.items()}
    L = int(cfg.L)
    r_start = int(getattr(A, "r_start", 0) or 6)
    r_end = int(getattr(A, "r_end", 0) or 10)
    r_start = max(1, min(r_start, L - 2))
    r_end = max(r_start + 1, min(r_end, L - 1))
    os.environ.update({"TB_R_ENABLE": "1", "TB_R_START": str(r_start), "TB_R_END": str(r_end),
                       "TB_R_MEAN": "3", "TB_R_MAX": "8", "TB_R_BACKPROP": "3",
                       "TB_R_INSTR_ONLY": "0", "TB_R_KV_SHARE": "0"})
    model = T.build_model(cfg).to(dev)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # A fresh RAdapter has pass_gain=0 -> the extra R passes are IDENTITY (s'=s), so the recurrent
    # chain does nothing. Seed a small gain (as the tiny prototype did) unless the caller overrides.
    seed = getattr(A, "gain_init", None)
    with torch.no_grad():
        model.r_adapter.proj.weight.zero_()          # u = norm(s): in-distribution for the R blocks
        model.r_adapter.pass_gain.fill_(float(seed) if seed else 0.05)
    print(f"rebuilt narrow-R P=[0,{r_start}) R=[{r_start},{r_end}) C=[{r_end},{L}) | "
          f"RAdapter fresh, pass_gain seeded to {float(seed) if seed else 0.05} "
          f"(identity would cripple the chain); loaded weights missing={len(missing)} "
          f"unexpected={len(unexpected)}", flush=True)
    model.eval()
    return model


def _build_local_model(T, A, dev):
    """Small GPT-2-vocab Walker from the trainer (narrow-R), for local runs with LoRA."""
    os.environ.update(LOCAL_ENV)
    r_start = max(1, int(A.L) - 3)
    r_end = max(r_start + 1, int(A.L) - 1)
    os.environ["TB_R_START"], os.environ["TB_R_END"] = str(r_start), str(r_end)
    cfg = T.Cfg()
    cfg.vocab, cfg.D, cfg.L, cfg.n_head = 50257, int(A.D), int(A.L), int(A.heads)
    cfg.mult, cfg.hybrid, cfg.ckpt, cfg.no_walk = 2.5, True, False, False
    cfg.max_t, cfg.alibi, cfg.rope = 16384, False, True
    cfg.ni = cfg.lmm = cfg.local_conv = cfg.lexical_semantic = False
    cfg.quant_backend, cfg.fp4 = "ste", False
    model = T.build_model(cfg).to(dev)
    raw_tok = T._load_tokenizer("gpt2")
    return model, raw_tok


def _build_arg_parser():
    P = argparse.ArgumentParser()
    P.add_argument("--ckpt", nargs="?", help="1B checkpoint (required unless --local)")
    P.add_argument("--steps", type=int, default=3000)
    P.add_argument("--batch", type=int, default=16)
    P.add_argument("--lr", type=float, default=1e-3)
    P.add_argument("--gate-lr-mult", type=float, default=30.0)
    P.add_argument("--inject", default="skip", choices=("skip", "u"))
    P.add_argument("--names", type=int, default=40)
    P.add_argument("--proc-layout", default="flat", choices=("flat", "record", "example"),
                   help="record: a 'Procedure Pj' header node with short step children (relation word salient)")
    P.add_argument("--eval-every", type=int, default=500)
    P.add_argument("--seed", type=int, default=0)
    P.add_argument("--save", default="runs/gnm_lm_1b.pt")
    P.add_argument("--resume", default="", help="hook state to start from (e.g. a *_last.pt). A resume is a NEW "
                   "run from that state: step counter and cosine LR schedule restart at 1")
    P.add_argument("--resume-opt", action="store_true", help="also restore the optimizer state")
    P.add_argument("--gate-init", type=float, default=None, help="set the gate after building / resuming")
    P.add_argument("--read-open", type=float, default=0.0, help="reserved: fixed unbounded read scale "
                   "(>0): the read reaches the LM at this scale instead of the tanh gate, and never "
                   "decays. A feature, not a solution target.")
    P.add_argument("--gain-init", type=float, default=None, help="seed narrow-R pass_gain (a fresh "
                   "RAdapter is identity at 0: the extra passes change nothing and the chain dies)")
    P.add_argument("--copy", action="store_true", help="say = point at the answer node (copy its value token)")
    P.add_argument("--var-len", action="store_true",
                   help="procedures of 1-4 steps (done node holds), r = longest + 0..2")
    P.add_argument("--fixed-world", action="store_true",
                   help="reuse ONE fixed world (same relations + procedures) every step: tests whether "
                        "the model can MEMORIZE the graph. Default: a new random world each step "
                        "(generalization; the model can never memorize, only learn the retrieval)")
    P.add_argument("--depth-eval", action="store_true", help="load --resume and print the depth table, then exit")
    P.add_argument("--keep-lm", type=float, default=0.0,
                   help="weight of the do-no-harm episode (generic prompts keep Walker's own distribution)")
    P.add_argument("--paraphrase", action="store_true", help="train on 6 question wordings, eval on 3 others")
    P.add_argument("--full-names", action="store_true", help="'First Last' people: multi-token answers")
    P.add_argument("--sup", default="chain", choices=("chain", "answer"),
                   help="answer: no per-pass gold (no forced candidates, no chain loss, copy attached)")
    P.add_argument("--link", action="store_true", help="entity edges + question-word path into the GNM")
    P.add_argument("--soft", action="store_true", help="answer-only helpers: no candidate cut, wider reads, "
                   "activation carry not detached, copy from the last pass")
    P.add_argument("--n", "--k", dest="k", type=int, default=1, help="N rollouts per training question (noisy "
                   "trajectories, loss on the best: best-of-N, RL-like). Not the k recurrent passes")
    P.add_argument("--qmem", action="store_true", help="the question lives in memory (a node); Walker sees "
                   "only ' Answer:' and pass 1 must find the question (with --link)")
    P.add_argument("--worlds", type=int, default=1, help="worlds per training step (gradients accumulated)")
    P.add_argument("--extra", type=int, default=0, help="extra recurrent passes per question (k = steps + extra; "
                   "after the last step the per-pass target holds the final fact)")
    P.add_argument("--halt", action="store_true", help="soft halting: the answer is read from a learned mix of "
                   "the passes, so extra passes are used only if they help (with --soft)")
    P.add_argument("--lm-only", action="store_true", help="REALISTIC: train on the answer likelihood "
                   "ONLY (no per-pass gold-node supervision); the memory gets features, not the solution.")
    P.add_argument("--selfsup", action="store_true", help="SELF-SUPERVISED retrieval: reward activation "
                   "mass on answer-containing nodes (only the answer, no per-pass gold) minus a length "
                   "penalty. Tests the containment objective without any path labels.")
    P.add_argument("--selfsup-len", type=float, default=0.02, help="length/sharpness penalty weight (lambda)")
    P.add_argument("--explore", type=float, default=1.0, help="proposal temperature for the k-rollout "
                   "exploration (Gumbel noise scale; ~1 barely diversifies, 5-20 genuinely explores)")
    P.add_argument("--sup-only", type=int, default=0, help="warmup steps that train the memory module "
                   "DIRECTLY (ungated per-pass gold, no LM loss), then switch to end-to-end. Breaks the "
                   "cold start: M learns to retrieve before the read gate/LM loss must carry it.")
    P.add_argument("--v2", action="store_true", help="GNM design v2 step 1 (gnm_v2.py): encoder = Walker's P + "
                   "one R pass token states, attention pooling, v/l/r heads; cosine label attention")
    P.add_argument("--label-only", action="store_true", help="with --qmem: no word-overlap mention edges; the "
                   "question read finds its entity / procedure by learned label match")
    P.add_argument("--local", action="store_true", help="build a small GPT-2-vocab Walker locally (no 1B "
                   "checkpoint) for CPU/GPU prototyping; pairs with --lora")
    P.add_argument("--hf", default="", help="use a small PRETRAINED HF causal LM (e.g. gpt2) as the local "
                   "Walker (fast local iteration; pairs with --lora)")
    P.add_argument("--D", type=int, default=256, help="--local: model width")
    P.add_argument("--L", type=int, default=6, help="--local: blocks")
    P.add_argument("--heads", type=int, default=8, help="--local: attention heads")
    P.add_argument("--lora", type=int, default=0, help="LoRA rank on Walker's early encoder blocks [0, r_end) "
                   "(0 = off). Trains only the adapter + the GNM (Walker base frozen).")
    P.add_argument("--lora-alpha", type=float, default=16.0)
    P.add_argument("--lora-scope", choices=("early", "late"), default="early", help="late: LoRA on the C "
                   "blocks (read path), encoder P/R blocks frozen so the retrieval substrate cannot drift")
    P.add_argument("--eval-worlds", type=int, default=12, help="held-out worlds per eval arm")
    P.add_argument("--eval-batch", type=int, default=32, help="batch per eval world")
    P.add_argument("--r-start", type=int, default=0, help="R section start when rebuilding a "
                   "narrow-R-retired checkpoint (0 = default 6)")
    P.add_argument("--r-end", type=int, default=0, help="R section end (exclusive) when rebuilding "
                   "(0 = default 10)")
    return P


def main():
    run(_build_arg_parser().parse_args())


def run(A):
    """Run a full pairing from an argparse.Namespace (so a notebook can drive it)."""
    CFG.update(paraphrase=A.paraphrase, full_names=A.full_names, sup=A.sup, k=A.k, qmem=A.qmem,
               label_only=A.label_only)
    import gmem_ltm as LT
    T = LT.T
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(A.seed)
    rng = random.Random(A.seed)
    if A.hf:
        import gnm_small
        model = gnm_small.SmallWalker(A.hf, r_start=(A.r_start or None), r_end=(A.r_end or None)).to(dev)
        raw_tok = T._load_tokenizer(A.hf)
        print(f"local pretrained Walker: {A.hf} (r=[{model.r_start},{model.r_end}))", flush=True)
    elif A.local:
        model, raw_tok = _build_local_model(T, A, dev)
    else:
        if not A.ckpt:
            raise SystemExit("--ckpt is required unless --local")
        model, raw_tok, cfg = T.build_model_for_inference(A.ckpt)
        if getattr(model, "r_adapter", None) is None:
            print("WARNING: this checkpoint has no narrow-R adapter (R was retired by a continuation "
                  "run). Rebuilding the R loop with a fresh zero-init adapter; --r-start/--r-end set "
                  "the section. Use a checkpoint that still holds the trained RAdapter to keep depth.",
                  flush=True)
            model = _rebuild_with_r(T, A, dev, cfg)
    # A fresh RAdapter has pass_gain=0 => the extra R passes are IDENTITY (s'=s): the recurrence does
    # nothing and the chain dies. If it is still all-zero, seed a small gain (frozen) so the passes
    # actually evolve, like the tiny prototype. A checkpoint's trained pass_gain is left untouched.
    if (getattr(model, "r_adapter", None) is not None
            and float(model.r_adapter.pass_gain.detach().abs().sum()) == 0.0):
        seed = 0.05 if A.gain_init is None else float(A.gain_init)
        with torch.no_grad():
            # Zero proj so u = norm(s): each R block's own pre-LayerNorm makes R(norm(s)) ~ R(s),
            # i.e. the normal flat output, so the passes refine the state instead of injecting noise
            # from a random projection. proj still gets gradient once gain > 0.
            model.r_adapter.proj.weight.zero_()
            model.r_adapter.pass_gain.fill_(seed)
        print(f"fresh RAdapter: proj zeroed (u=norm(s)), pass_gain seeded to {seed}", flush=True)
    for q in model.parameters():        # Walker base frozen; LoRA (below) re-enables its adapter
        q.requires_grad_(False)
    model.train()                       # extra R passes only build graphs in training mode
    tok = LT.Tok(raw_tok)
    rn = LT.LTMRunner(model, tok, None, 2)
    names = single_token_names(tok, A.names)
    d_model = model.emb.weight.shape[1]
    if A.v2:                            # design v2: encoder in Walker's query space, cosine label attention
        import gnm_v2 as V2
        _Lb = len(model.blocks)
        _re = int(getattr(model, "r_end", _Lb))
        _scope = getattr(A, "lora_scope", "early")
        _lora_blocks = range(_re, _Lb) if _scope == "late" else None
        enc = V2.WalkerStateEnc(model, tok, lora_r=A.lora, lora_alpha=A.lora_alpha,
                                lora_blocks=_lora_blocks)
        print(f"LoRA scope={_scope}: "
              + (f"C blocks [{_re},{_Lb}); encoder [0,{_re}) FROZEN (substrate cannot drift)"
                 if _scope == "late" else f"encoder blocks [0,{_re})"), flush=True)
        M = V2.NodeMemV2(enc, d_model).to(dev)
    else:
        if A.lora:
            raise SystemExit("--lora needs --v2 (the adapter lives on the v2 document encoder)")
        M = N.NodeMemory(enc=N.WalkerTextEnc(rn, d_model)).to(dev)
    hook = GL.GNMHook(M, d_model, model.r_adapter.pass_gain.numel(), shared_gate=True, norm_reads=True).to(dev)
    hook.read_open = float(A.read_open) if getattr(A, "read_open", 0.0) else None
    if A.v2:
        V2.tie_queries(hook, M, h_info=not A.qmem)
    (attach_skip if A.inject == "skip" else GL.attach)(model, hook)
    hook.keep_lm = A.keep_lm > 0
    hook.link = A.link
    if A.soft:
        hook.soft, hook.K, hook.Ki, M.C = True, 10 ** 9, 10 ** 9, 0     # read every node, weighted
    if A.halt:
        hook.halt = torch.nn.Linear(d_model + N.D, 1).to(dev)              # soft halting over the passes
        torch.nn.init.zeros_(hook.halt.weight)
    if A.copy:
        hook.copy_gate = torch.nn.Linear(d_model, 1).to(dev)
        torch.nn.init.zeros_(hook.copy_gate.weight)
        torch.nn.init.constant_(hook.copy_gate.bias, -2.0)       # starts mostly Walker's own distribution
    print(f"1B: D={d_model} R=[{model.r_start},{model.r_end}) pass_gain="
          f"{[round(v, 3) for v in model.r_adapter.pass_gain.float().tolist()]} inject={A.inject} "
          f"names={len(names)} {CFG}", flush=True)
    rest = [q for n, q in hook.named_parameters() if n != "gate" and q.requires_grad]
    lora_b = []
    if A.lora:
        import gnm_lora
        lora_params = gnm_lora.lora_parameters(model)
        lora_b = [p for n, p in model.named_parameters() if n.endswith(".B")]
        rest = rest + lora_params
        print(f"LoRA: rank={A.lora} adapting {len(lora_params)//2} encoder projections "
              f"({sum(p.numel() for p in lora_params):,} params)", flush=True)
    opt = torch.optim.AdamW([{"params": rest}, {"params": [hook.gate], "lr": A.lr * A.gate_lr_mult,
                                                  "weight_decay": 0.0}], lr=A.lr, weight_decay=0.01)
    base = [g["lr"] for g in opt.param_groups]
    best, t0, start = -1.0, time.time(), 1
    if A.resume:
        st = torch.load(A.resume, map_location=dev)
        print(hook.load_state_dict(st["hook"], strict=False), flush=True)
        if A.resume_opt:
            opt.load_state_dict(st["opt"])
        print(f"resumed from {A.resume} (step {st['step']}, held-out {st['held_out']:.2f}); new run: step "
              f"counter and LR schedule restart at 1", flush=True)
    if A.gate_init is not None:
        with torch.no_grad():
            hook.gate.fill_(A.gate_init)
    if A.depth_eval:
        depth_table(T, model, hook, M, dev, tok, rn, names, A.proc_layout)
        return
    for step in range(start, A.steps + 1):
        fct = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - 1) / A.steps))
        for g, lr in zip(opt.param_groups, base):
            g["lr"] = lr * fct
        _warm = A.sup_only > 0 and step <= A.sup_only   # warmup: train M directly (ungated), no LM loss
        opt.zero_grad(set_to_none=True)
        for _ in range(A.worlds):               # several worlds per step (gradients accumulated)
            # --fixed-world: rebuild the SAME world every step (same seed) so the graph is persistent and
            # the model can memorize it; default: a new random world each step (generalization).
            _wrng = random.Random(1234) if getattr(A, "fixed_world", False) else rng
            w = make_world(_wrng, "train", names, A.proc_layout, var_len=A.var_len)
            r = max(len(c) for c in w["procs"]) + (rng.randint(0, 2) if A.var_len else 0) + A.extra
            lm, ds, acc, hm, hi = run_batch(T, model, hook, M, w, rng, A.batch, dev, True, tok, rn, r=r,
                                            wording="train" if A.paraphrase else None,
                                            selfsup=A.selfsup, selfsup_len=A.selfsup_len,
                                            explore=getattr(A, "explore", 1.0))
            if _warm:
                loss = ds                          # Phase A: memory warmup (retrieval supervision)
            elif A.lm_only:
                loss = lm                          # realistic: answer/likelihood ONLY, no gold-node signal
            else:
                loss = lm + ds
            if A.keep_lm > 0 and not _warm:
                kl, run_batch.plain_alpha = plain_batch(T, model, hook, M, w, rng, A.batch, dev, tok, rn, PLAIN_TRAIN)
                loss = loss + A.keep_lm * kl
            (loss / A.worlds).backward()
        if A.sup_only and step == A.sup_only:
            print(f"warmup done at step {step} (direct retrieval supervision off; end-to-end on)", flush=True)
        torch.nn.utils.clip_grad_norm_(rest + [hook.gate], 1.0)
        opt.step()
        if step % 50 == 0:
            _lb = sum(float(p.detach().pow(2).sum()) for p in lora_b) ** 0.5
            print(f"step {step} lm {float(lm.detach()):.3f} ds {float(ds.detach()):.3f} acc {acc:.2f} "
                  f"mem {[round(v, 2) for v in hm]} ins {[round(v, 2) for v in hi]} "
                  f"gate {float(hook.gate[0].detach()):.3f} w_edge {float(M.w_edge.detach()):+.4f} "
                  f"w_op {float(M.w_op.detach()):+.4f} loraB {_lb:.3f} "
                  f"cont {getattr(run_batch, 'cont', 0.0):.2f} "
                  f"copy-mix {getattr(run_batch, 'alpha', 0.0):.2f} [{time.time() - t0:.0f}s]", flush=True)
        if step % A.eval_every == 0 or step == A.steps:
            score = evaluate(T, model, hook, M, dev, tok, rn, names, step, n_worlds=A.eval_worlds,
                             B=A.eval_batch, proc_layout=A.proc_layout, extra=A.extra)["held-out"]
            os.makedirs(os.path.dirname(A.save) or ".", exist_ok=True)
            state = {"hook": {k: v for k, v in hook.state_dict().items() if not k.startswith("M.enc.rn")},
                     "opt": opt.state_dict(), "step": step, "held_out": score}
            torch.save(state, A.save.replace(".pt", "_last.pt"))          # resume point (chain may be trained
            if score > best:                                               # long before the answer is)
                best = score
                torch.save(state, A.save)
                print(f"saved best @{step}: held-out answer {score:.2f}", flush=True)


if __name__ == "__main__":
    main()

