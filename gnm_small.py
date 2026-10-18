"""Small pretrained local "Walker": wrap an HF causal LM (GPT-2) with Walker's narrow-R interface so
the GNM pairing runs locally, fast, on pretrained token states.

Same forward contract the pairing expects: P = blocks[0:r_start] once, R = blocks[r_start:r_end] once
flat then r extra passes with a shared RAdapter (pass_gain init 0 => identity), C = blocks[r_end:L].
Exposes .blocks/.emb/.r_start/.r_end/.r_adapter/._r_pass_input/._r_pass_update and forward(idx,
r_active) so attach_skip, WalkerStateEnc and GNMHook work unchanged.
"""
import torch
import torch.nn as nn

import train_walker_1b_continue as T


class SmallWalker(nn.Module):
    def __init__(self, name="gpt2", r_start=None, r_end=None, r_mean=3, r_max=8, r_backprop=3):
        super().__init__()
        from transformers import GPT2LMHeadModel
        self.hf = GPT2LMHeadModel.from_pretrained(name)
        self.blocks = self.hf.transformer.h
        self.emb = self.hf.transformer.wte
        n = len(self.blocks)
        self.r_start = int(r_start if r_start else max(1, n // 3))
        self.r_end = int(r_end if r_end else max(self.r_start + 1, (2 * n) // 3))
        self.r_mean, self.r_max, self.r_backprop = int(r_mean), int(r_max), int(r_backprop)
        self.r_update, self.r_gain_mode = "blend", "per_pass"
        self.r_instr_only, self.r_kv_share = False, False
        self.r_adapter = T.RAdapter(self.hf.config.n_embd, self.r_max)
        # LoRA targets for this architecture (see gnm_lora.attach_lora)
        self.lora_targets = ("attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj")

    @property
    def r_depth_cap(self):
        return self.r_max

    def _blk(self, i, x):
        return self.blocks[i](x)

    def _r_pass_input(self, r_adapter, s_in, e_state):
        concat = torch.cat([s_in, e_state.to(s_in.dtype)], dim=-1)
        u = r_adapter.norm(s_in + r_adapter.proj(concat.to(r_adapter.proj.weight.dtype)).to(s_in.dtype))
        if getattr(self, "r_update", "blend") == "delta":
            with torch.no_grad():
                ratio = (s_in.float().pow(2).mean(-1, keepdim=True).sqrt()
                         / u.float().pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-6))
            u = u * ratio.to(u.dtype)
        return u

    def _r_pass_update(self, s_in, u, yb, gain):
        if getattr(self, "r_update", "blend") == "delta":
            return s_in + gain * (yb - u)
        return s_in + gain * (yb - s_in)

    def forward(self, idx, targets=None, r_active=None):
        dev = idx.device
        t = idx.shape[1]
        pos = torch.arange(t, device=dev)[None]
        x = self.emb(idx) + self.hf.transformer.wpe(pos)
        for i in range(0, self.r_start):
            x = self._blk(i, x)
        e_state = x
        for i in range(self.r_start, self.r_end):
            x = self._blk(i, x)
        want_r = True if r_active is None else bool(r_active)
        n_nograd, k_grad = (0, 0)
        if want_r:
            n_nograd, k_grad = T._sample_r_counts(bool(self.training), self.r_mean,
                                                  self.r_max, self.r_backprop)
        for p in range(int(n_nograd + k_grad)):
            gated = bool(self.training) and p >= n_nograd
            with (torch.enable_grad() if gated else torch.no_grad()):
                s_in = x
                u = self._r_pass_input(self.r_adapter, s_in, e_state)
                yb = u
                for i in range(self.r_start, self.r_end):
                    yb = self._blk(i, yb)
                gain = self.r_adapter.pass_gain[min(p, self.r_max - 1)].to(yb.dtype)
                if not gated:
                    u, yb, gain = u.detach(), yb.detach(), gain.detach()
            x = self._r_pass_update(s_in, u, yb, gain)
        for i in range(self.r_end, len(self.blocks)):
            x = self._blk(i, x)
        x = self.hf.transformer.ln_f(x)
        logits = self.hf.lm_head(x)
        loss = None
        if targets is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
        return logits, loss
