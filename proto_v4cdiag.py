"""why facts are held wrong: places vs things (merges / splits), fold actions vs the teacher's, column nodes"""
import random, torch, sys
import proto_v4c as V
M = V.Module().to(V.dev)
M.load_state_dict(torch.load(f"v4c_{V.VECS}_M.pt"))
M.eval()
rng = random.Random(50007)
w = V.new_world(rng)
T = V.Teacher(w)
mem = V.Memory(M)
where = {}
with torch.no_grad():
    for _ in range(30):
        for dd in V.world_step(w, rng):
            o = dd()
            side = w.side.get(id(o), {}) if o else {}
            for text, meta in V.said(o, w) if o else []:
                n0 = [len(p[2]) for p in mem.places]
                mem.hear(text)
                T.heard(text, meta, side)
                if meta[0] == "fact":
                    got = [i for i, p in enumerate(mem.places) if i >= len(n0) or len(p[2]) > n0[i]]
                    where.setdefault((meta[1], meta[2]), set()).update(got)
    things = len(where)
    inplace = {}
    for k, ps in where.items():
        for p in ps:
            inplace.setdefault(p, set()).add(k)
    split = sum(len(ps) > 1 for ps in where.values())
    merged = sum(len(ks) > 1 for ks in inplace.values())
    print(f"things {things}, places {len(mem.places)}, things split over places {split}, places holding 2+ things {merged}")
    # fold actions: the model's on its own memory vs the teacher's, for things kept in one place alone
    agree = n = 0
    for k, ps in where.items():
        if len(ps) != 1 or len(inplace[next(iter(ps))]) != 1:
            continue
        p = next(iter(ps))
        state = None
        for e, act in zip(mem.places[p][2], T.log[k]):
            r = mem.rule_at(e["attr"], e["t"])
            x = torch.cat([e["h"], r if r is not None else torch.zeros_like(e["h"]), torch.tensor([float(state is not None)], device=V.dev)])
            a_ = V.FOLD[int(M.fold(x).argmax())]
            agree += a_ == act
            n += 1
            state = 1
    print(f"fold actions agree with the teacher {agree}/{n}")
    # column nodes alone
    r = random.Random(3)
    for op in ("add", "sub", "max", "min"):
        ok = 0
        for _ in range(200):
            a, b = r.randint(-60, 200), r.randint(-60, 200)
            got = V.run_op(M, V.as_num(V.tok(str(a))), V.as_num(V.tok(str(b))), op)
            ok += "".join(got) == str(V.OPF[op](a, b))
        print(f"column nodes {op}: {ok}/200")
