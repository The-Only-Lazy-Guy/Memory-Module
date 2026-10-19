"""Lean core of the fast-weight memory (docs/GNM.md section 9.11): everything the memory line needs, nothing else.

- `load_walker`, `Tok`, `walker_states`: the frozen LM and its token states (no gnm_v2 / gmem_ltm / gmem_v3 imports).
- `doc_encoding`: one JSON document -> token ids, per-leaf spans, pointer chains, value-token positions (the single
  copy of what anyjson / stages / train each had).
- `Memory`: write a document into units (key, value, gain, per-token value slots) and address them with a query.
  Same parameter names as `gnm_fwm.FastWeightMemory`; `load_memory` folds the old in-module hop parameters into one
  query-pool direction (`qdir`), so trained checkpoints load with identical outputs.
- `Store`: units of many memories in one tensor with a group id (a query only sees its own group).
- `Readout` + `Inject`: the read enters Walker's residual stream at the answer positions (pooled value at the first
  answer position, value slot t at answer position t); Walker's own blocks and head produce every token.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------ the frozen LM
def load_walker(hf="gpt2", ckpt="", enc_layer=0, dev=None):
    """Walker (1B checkpoint) or a small HF stand-in, frozen. `model.r_end` = the encoder layer (states at its input)."""
    import train_walker_1b_continue as T
    if ckpt:
        model, raw_tok, _cfg = T.build_model_for_inference(ckpt, device=dev)
        model = model.to(dev).eval()
        if getattr(model, "r_adapter", None) is None or not int(getattr(model, "r_end", 0) or 0):
            model.r_end = int(enc_layer) or (2 * len(model.blocks)) // 3
        print(f"Walker 1B from {ckpt}: {len(model.blocks)} blocks, d={model.emb.weight.shape[1]}, "
              f"encoder states at block {model.r_end}", flush=True)
    else:
        import gnm_small
        model = gnm_small.SmallWalker(hf).to(dev).eval()
        raw_tok = T._load_tokenizer(hf)
        if enc_layer:
            model.r_end = int(enc_layer)
    for p_ in model.parameters():
        p_.requires_grad_(False)
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


from fwm_data import doc_encoding  # noqa: E402,F401  (re-export)


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
        if self.training:
            with torch.no_grad():
                f = 0.0 if float(seen) == 0 else 0.99
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
            S = self.vslots
            vl = np.fromiter((min(len(q), S) for q in vpos), dtype=np.int64, count=n)
            vp = np.concatenate([np.asarray(q[:S], dtype=np.int64) for q in vpos]) if vl.sum() else np.zeros(0, np.int64)
            row = np.repeat(np.arange(n), vl)
            col = np.concatenate([np.arange(k) for k in vl]) if vl.sum() else np.zeros(0, np.int64)
            sv = self.Wvs(Xs[torch.from_numpy(vp).to(dev)])
            slots = torch.zeros(n, S, sv.shape[-1], device=dev, dtype=sv.dtype)
            r_, c_ = torch.from_numpy(row).to(dev), torch.from_numpy(col).to(dev)
            slots = slots.index_put((r_, c_), sv + self.slot_e[c_])
        return kraw, v, torch.log(g + 1e-6), slots, g

    def query(self, Hq, mq, Eq):
        """Question token states (B, T, d) -> raw query (B, dk), in the key space (tied pool + head)."""
        X = Hq[mq]
        if self.training:
            mu, var = self._std(X, True)
        else:
            mu, var = self.mu_q, self.var_q
        Hs = torch.cat([self.ln((Hq - mu) / (var + 1e-5).sqrt()), self.lne(Eq)], -1)
        sc = self.pk.s(Hs).squeeze(-1) + Hs @ self.qdir
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


class Store:
    """Units of several memories in one tensor; `group` = memory id per unit; keys centred per memory."""

    def __init__(self, units, groups, n_groups):
        kraw = torch.cat([u[0] for u in units])
        self.V = torch.cat([u[1] for u in units])
        self.logg = torch.cat([u[2] for u in units])
        self.S = torch.cat([u[3] for u in units]) if units[0][3] is not None else None
        self.group = torch.as_tensor(groups, device=kraw.device)
        cnt = torch.zeros(n_groups, device=kraw.device).index_add_(0, self.group, torch.ones_like(self.logg))
        self.kmean = torch.zeros(n_groups, kraw.shape[1], device=kraw.device).index_add_(0, self.group, kraw) \
            / cnt.clamp(min=1)[:, None]
        self.K = F.normalize(kraw - self.kmean[self.group], dim=-1)

    def attend(self, mem, qraw, qgroup):
        lg = self.logg[None].expand(len(qgroup), -1).masked_fill(qgroup[:, None] != self.group[None], -1e4)
        return mem.attend(qraw, self.K, lg, self.kmean[qgroup])


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
