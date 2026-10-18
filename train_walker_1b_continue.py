"""WalkGPT trainer: operator map and invariants.

STATUS 2026-10-09 (current; living tracker: docs/PROJECT_STATUS.md)
-------------------------------------------------------------------
Phase 2 prepared from step 33000: TB_TRAIN_PHASE=2 restarts optimizer + LR,
14-source append-only mix, baseline val 1.7384 (runs/phase2_plan.txt).
LOTUS+ graph memory beside Walker (TB_GMEM=1, section "LOTUS+ graph memory"):
module OUTSIDE WalkGPT (own AdamW, checkpoint key "gmem"), span writer,
Walker frozen for the memory loss (TB_GMEM_WALKER_GRAD=0; =1 hurt LM val);
memory-only LoRA exists but is off (TB_GMEM_LORA_RANK=0: it lowered memory EM).
Report: docs/LOTUS_MEMORY_REPORT.md. The block below is the 10-07 snapshot.

STATUS 2026-10-07 (HF ckpt_ongoing.pt step 29750, best_val 1.7499)
------------------------------------------------------------------
Checkpoint: 841 keys, all bf16 (no fp32 masters); walk GLOB/COUPLE/GATE/MIX,
clip norm; Phase-B rank 32; narrow-R P=[0,6) R=[6,10) C=[10,16), mean 4,
max 8, learned pass_gain [.051, .023, ~0 x6]. Dead pos_emb still present.

Fixed this round (details + tests: BUG_REPORT.md "Round 3"):
- R prefix (no-grad) passes detached the residual stream: embedding/prelude
  got ~2% gradient on every r > backprop microbatch. Skip path now keeps grad.
- Pure-bf16 AdamW dropped most sub-ULP updates (the LN/gamma "freeze").
  CompensatedAdamW (TB_OPT_KAHAN=1, runtime override): -0.081 nats on the
  continuation A/B; replicated at D=512 over 2 seeds (-0.071 / -0.073);
  never-updated bf16 weights 62-74% -> 9-12%.
- KV-cache decode: EMA trace-vs-residual bug (~50% logit error on trained
  weights), depth taps from block input, narrow-R now cached exactly.
  Decode is kernel-launch-bound at chat lengths; cache wins at long T only.
- TB_R_MEAN is tolerated on resume (sampling-only); TB_INFER_R_MEAN for
  inference.

Measured on MoLab (RTX PRO 6000):
- Real 1B test-time R sweep (wikitext-103 val): CE r=0 3.6659, r=2 3.6658,
  r=4 3.6660, r=8 3.6660 -> R is a no-op; r=4 costs 1.85x forward time.
- R v2 (TB_R_UPDATE=delta, TB_R_GAIN=shared, TB_R_SAMPLE=1, Huginn-style)
  on a 30M LM: no test-time depth scaling, no extrapolation past the
  training depth. Kept as opt-in research code, not for this 1B.
- Motif v2 (TB_WALK_MOTIF_V2) vs walk, 2 seeds: -0.007..-0.014 nats across
  two launches (v1: -0.009); extensions: ORTHO (Gram-Schmidt, fixed to
  divide by p.p) -0.011, SHARED bank -0.014, CONF -0.009, TOPK=4 -0.011,
  MOD -0.004, all -0.014. Run-to-run noise ~0.008, so only "shared >= v2"
  is suggestive. Available as a one-way panel option; motif (every option)
  is cache-decodable.

Next resume (decided): this trainer, TB_OPT_KAHAN=1, R mean 0, nothing else
changed. Watch gnorm, walk_g/LN movement and the first eval vs 1.7499 for
~200 steps; rewind to step 29750 if loss spikes persist past ~300 steps.

Phase-B continuation (2026-09)
------------------------------
2026-09-20 causal-continuation repair:
- FP8 activation scaling is token-rowwise, never reduced across [B*T,D].
- short-prefix hybrid attention retains trained global/window head roles.
- coarse Walk activates once one complete prior S-token chunk exists.
- ordinary generation uses a 4096-token window by default (TB_GENERATE_WINDOW).
All changes preserve parameter names/shapes and are ordinary-resume compatible.
2026-09-15 data-mix extension: `TB_STREAM_PRESET=nemotron_knowledge_15b`
activates a resumable weighted eight-source continuation curriculum (HQ web,
filtered educational PDFs, math-4plus, RQA, factual QA, math textbooks,
InfiniByte reasoning, and HQ code). `TB_STREAM_PRESET=knowledge_15b_ungated`
is the access-free fallback: FineWeb-Edu-Dedup, Cosmopedia-v2, filtered
FinePDFs-Edu, MegaMath-Web-Pro-Max, RQA, factual QA, math textbooks, and
InfiniByte.
`--stream-probe N` inspects/tokenizes live rows and rejects obvious binary/OCR/
control-character corruption before a run. Both presets deliberately exclude
MMLU-SFT and benchmark-derived MC slices.

`TB_PHASE_CKPT` is a real phase transition, distinct from both ordinary resume
and `TB_INIT_CKPT`: it preserves unchanged model weights, named Adam moments,
RNG, stream cursor/buffers and lifetime token count, while resetting only the
Phase-B step/LR schedule counters. `TB_PHASE_B_SUPERSET=1` adds a zero-output
HIER r-space sidecar beside the trained Phase-A Walk; archive parameters exist
from Phase-B step zero and are runtime-gated, so later activation does not change
checkpoint shapes. Ordinary resume is fail-closed on incompatibility. The DRCG
research prototype is kept in `drcg_lm.py`, not in this 1B trainer.


Model
-----
2026-09-05 Phase-B instability regression:
The data/depth checkpoint lambda captured the loop variable `blk`. Backward
recomputation could therefore execute the last block for earlier checkpoints,
producing incorrect gradients even when forward CE was finite. Pass the block
module directly, with tensor keyword arguments. A CPU regression compares all
parameter gradients with checkpointing on/off for zero and active sidecar gates.
Depth history now retains the prior block input; the old history stored its
output and consequently formed x-x at the next iteration. NI depth taps remain
separate. Parameter names/shapes are unchanged, but these fixes cannot undo
updates already made with incorrect gradients. Recover from a known-good
Phase-A checkpoint when the Phase-B loss has already diverged.
GRAD_GROUP is measured AFTER clipping; gnorm is the pre-clipping total. A
threshold of 10000 still permits very large updates to reach the optimizer.

Inverse-perception composition probe (2026-09-05)
--------------------------------------------------
`--composition-walk-probe 1800` compared fixed span pooling, learned ordered
composition, and composition plus an explicit composition-minus-reference delta
on a matched D=64 two-layer byte LM. All arms had 106,368 parameters and used
the same three seeds, windows, and 1,800 updates. Mean held-out CE was 2.22901
(pool), 2.22861 (compose), and 2.22851 (delta). The span memory was active:
removing it raised CE by about 0.004 nat on seed 17. The 0.0005-nat edge is
below expected seed variation; learned arms were not faster. This establishes
that completed-span composition is causal and trainable, but not yet a useful
Phase-B improvement. The probe is a full-model retraining screen and does not
isolate sidecar-only effects.

DRCG was intentionally removed from this trainer on 2026-09-05. Its full
dynamic recurrent graph implementation, including energy-based unbounded
traversal, lives in `drcg_lm.py` and must be evaluated independently before any
future 1B integration.

Experiment log (kept in-source so copied scripts carry their evidence)
-----------------------------------------------------------------------
2026-09-06 pretrained GPT-2 / Tiny Shakespeare, T=32, 10 updates:
  * frozen baseline CE 6.03607 (single held-out batch; noisy smoke result)
  * frozen distributed Walk depth 4 CE 6.01414 (+0.02193 tail-off delta)
  * frozen distributed Phase-HIER depth 4 CE 5.94765 (+0.08843 tail-off delta)
  * unfrozen Phase-HIER depth 4 CE 4.79418; same-run tail-disabled CE 4.83674,
    giving an incremental tail gain of 0.04256. Backbone LR 2e-5, tail LR 6e-5.
  * stable frozen baseline at T=64, 20 validation batches, 12k tokens: CE 5.19987.
  * original unbounded DRCG toy transfer: 2.96860 vs continued fixed 2.81494
    (+0.15366 nat worse), 47 learned traversals vs initial 5, and ~46x slower.
  * bounded DRCG-lite smoke (GPT-2, 2 HIER nodes, max 2 passes, 1 update):
    CE 6.12678 vs frozen 6.12688; tail-off delta +0.00010. This only validates
    wiring, not quality; the router needs a multi-seed, longer equal-FLOP test.
  * amplified DRCG-lite smoke (same setup, 5 updates; front=1.5, phase=1.5,
    hier=2, pred=2, pred-loss=0.25): validation curve 4.93815 -> 4.93653 and
    final CE 6.12533 vs frozen 6.12688 (delta -0.00154), tail-off delta +0.00154.
  * amplified DRCG-lite longer pilot (GPT-2, T=64, 100 updates, 5 eval batches):
    frozen CE 5.13506; validation checkpoints 5.33069 -> 5.32908 -> 5.32853
    -> 5.32843; final CE 5.12984 (delta -0.00522), tail-off delta +0.00522.
  * Phase-HIER comparison (GPT-2, T=64, depth 4, 40 updates, 3 eval batches,
    frozen backbone): baseline settings (front=0, norm=.05, phase=1, hier=2)
    reached CE 5.01027 vs frozen 5.12519 (gain .11492). The amplified setup
    (front=1.5, norm=.10, phase=1.5, hier=2, pred=2, pred-loss=.25) reached
    CE 4.87127 (gain .25392), a +.13900 improvement over the baseline arm.
These are directional probes, not long-run quality claims. DRCG-lite below is
bounded and opt-in; do not promote it into the long Phase-B run without an
equal-FLOP/latency comparison against distributed Phase-HIER.

The backbone is a decoder-only GPT. `Attn` provides token-to-token retrieval;
`WalkAdapter` provides residual features of the *trajectory* of hidden states
(local differences, causal summaries, and optional dense mixing). Walk is not
a replacement for associative attention. In the hybrid attention layers, half
the heads are global and half are causal-windowed; the window is `T / 8` capped
at 512 tokens.

Parameter-matched attention/Walk experiments for L=16
-----------------------------------------------------
Relative to all-attention plus `WALK_MIX_DEPTH=1`, these settings have nearly
the same D^2 parameter budget:
  * ATTN_EVERY=1, MIX_DEPTH=1: 16 attention layers (baseline)
  * ATTN_EVERY=2, MIX_DEPTH=3:  8 attention layers (balanced candidate)
  * ATTN_EVERY=4, MIX_DEPTH=4:  4 attention layers (sparse-Walk candidate)
Do not change `TB_ATTN_EVERY` or `TB_WALK_MIX_DEPTH` in an ordinary resume:
doing so introduces random parameters or discards trained ones.

Training phases
---------------
Phase A is short-context base pretraining (for example 1024 -> 2048 tokens).
Use a fresh checkpoint and a token-based LR schedule when changing effective
batch size or the context curriculum. Phase B is a separate long-context
extension with fixed long T and a fresh schedule. Keep `TB_MAX_T` constant
across phases so weights have identical shapes; RoPE leaves `pos_emb` unused
but it remains in checkpoints for compatibility.

Checkpointing and resume
------------------------
Ongoing checkpoints preserve model, optimizer, RNG, exact stream-prefetch
state, and the LR schedule contract. `TB_STEPS` is a stopping limit, while
`TB_LR_TOTAL_STEPS` is the schedule horizon. Context endpoints (`TB_T0/TB_T1`)
are intentionally runtime curriculum settings, not locked scheduler fields.
For a legacy ongoing checkpoint that predates `run_state`, the trainer writes
`<TB_CKPT_PATH>.metadata.json` after loading. That tiny atomic sidecar records
the explicitly supplied current `TB_*` contract; it avoids rewriting the
multi-gigabyte model/optimizer payload and is consumed on later resumes. Since
legacy payloads cannot reveal their historical environment, the first migration
must be launched with the intended schedule values. When HF backup is enabled,
the sidecar is uploaded beside `ckpt_ongoing.pt` as
`ckpt_ongoing.pt.metadata.json`.
`TB_INIT_CKPT` is different: it loads *weights only* into a fresh training
state. It refuses architecture mismatches unless `TB_INIT_ALLOW_PARTIAL=1`.
PyTorch 2.6 defaults to `weights_only=True`, but a normal ongoing checkpoint
also stores trusted local optimizer/RNG/NumPy/stream state. Resume therefore
explicitly uses `weights_only=False` for the user-owned checkpoint; never point
`TB_CKPT_PATH` at an untrusted pickle. Inference/weights-only transfers retain
their separate explicit loader.

2026-09-13 finite-stream rollover fix: the 2025 highest-quality FineWeb stream
ends near the observed 3.95B lifetime-token point. StreamSource now cycles
deterministically instead of raising when a hash domain reaches EOF. Epoch,
raw cursor, prefetched buffers, and held-out raw-row ranges are checkpointed;
held-out FineWeb/code rows remain excluded on every epoch. Set
`TB_STREAM_CYCLE=0` to restore fail-at-EOF behavior. A finite fake-stream smoke
test covers rollover, validation exclusion, and checkpoint restoration.

2026-09-13 intentional corpus-transition fix: `TB_RESUME_DATASET=override`
permits a narrowly scoped change to `TB_STREAM`, its pinned revision, domain
key, or domain count while retaining model/optimizer/RNG/lifetime-token/LR
state. It deliberately discards only the old general-stream cursor and
prefetched buffers, because restoring FineWeb rows into a different dataset
would silently contaminate the transition. Code-stream state is preserved.
Use a new `TB_FW_VAL_PATH`; its metadata remains fail-closed across corpora.

Validation isolation
--------------------
For a weighted stream, `TB_MIX_VAL_TOKENS_PER_SOURCE` captures an equal-size
unseen token bank from every source before any training pump starts. The
manifest at `TB_MIX_VAL_PATH.meta.json` binds the complete mix, per-source
identity, tokenizer, vocabulary, raw row intervals, and shard checksums. Each
interval is excluded from its own stream on every resume. The trainer reports
per-source CE/top-k plus macro and worst-source CE; headline `val` is the
training-weighted source mean. Legacy single-stream runs can still use
`TB_FW_VAL_TOKENS`/`TB_FW_VAL_PATH`. Pin dataset revisions for reproducible
raw-row identities across upstream dataset updates.

Precision and diagnostics
-------------------------
QLinear retains bf16 masters and uses FP8/FP4 caches where supported; the
fallback is cached dequantized bf16. `TB_WALK_NORM_MODE=renorm` is the trained
default; `clip` and `cap` are diagnostic variants, not safe mid-run changes.
Set `TB_WALK_ABLATE_EVAL=1` to report held-out loss with Walk zeroed.

Subspace Walk (fresh architecture experiment)
----------------------------------------------
NOTE 2026-09-30: the subspace/evidence Phase-A variants described in this
essay (SubspaceWalkAdapter family, EvidenceWalkAdapter) were removed from the
trainer — ckpt_ongoing.pt carries their tensors in zero of 16 blocks, and
their TB_ flags now raise instead of constructing. HierarchicalEvidenceWalk-
Adapter survives as the Phase-B sidecar. The essay below is preserved verbatim
as design history; do not follow its construction instructions.
The original Walk is diagonal in the current residual basis: its local and
global time filters are per-coordinate gains. In a wide pretrained transformer
a useful feature is usually distributed over many rotating coordinates, so that
basis-dependent operation needs a D-by-D `mix_proj` to communicate features.
That repair costs O(D^2) parameters and O(T*D^2) work per layer. Set
`TB_WALK_SUBSPACE_RANK=r` on a *fresh* run to use `SubspaceWalkAdapter`
instead: `u=W_in LN(x)` maps D channels to r learned trajectory channels,
temporal features are computed in r-space, and a zero-init `W_out` writes the
correction back. It costs O(D*r) parameters/work plus O(T*r), preserves the
initial exact no-op, and is not shape-compatible with ordinary Walk checkpoints.
Use r=32--128 at D=2048 as a starting range; it is a new experiment, never a
mid-run resume setting. `TB_WALK_SUBSPACE_GATED=1` adds a compact
`correction * SiLU(W_gate[u, correction])` interaction in r-space for O(r^2)
additional parameters; test it against the ungated adapter before using it.
`TB_WALK_SUBSPACE_CONTENT=1` adds a causal normalized associative state in the
same learned r-space: it retrieves a content-weighted summary of prior
trajectory channels in O(T*r^2), not a D-wide quadratic attention layer. It is
a fresh architecture only. Compare it with the built-in `lowrank_transformer`
tail control before treating it as an improvement over ordinary attention.
`TB_WALK_MEMORY_CHUNK=C` additionally enables an experimental chunk-trajectory
memory on the gated variant: queries retrieve only completed prior C-token
subspace summaries. This is coarse semantic/state memory, not a replacement
for exact K/V attention; use `TB_WALK_MEMORY_TOPK` (default 1) and test it on
long-context retrieval before considering Phase B.

Frozen-pretrained Walk-tail probe (offline)
-------------------------------------------
`--pretrained-walk-tail-probe [STEPS]` freezes a locally cached Hugging Face
causal LM and inserts a zero-initialized stack of `WalkAdapter` residuals after
its final decoder block, before the model's final normalization/output head.
The tail therefore begins as an exact no-op and cannot damage pretrained logits
at step zero. It trains *only* tail parameters and compares depths from
`TB_PT_WALK_DEPTHS` (default `0,1,8,32`) on identical Tiny Shakespeare windows;
depth 0 is the frozen no-tail control. This tests whether extra trajectory
compute can extract useful post-backbone features, not whether it can replace
attention. It uses local-only model/tokenizer loading by default
(`TB_PT_WALK_MODEL=gpt2`, `TB_PT_WALK_LOCAL_ONLY=1`) and never touches Walker
checkpoints. Reported `walk_off_delta` is the held-out loss change after
temporarily zeroing the trained tail, guarding against a tail that merely
coexists with the frozen backbone. Do not promote a tail to Phase B unless it
beats depth 0 across seeds, has a positive usage delta, and retains its win at
the intended Phase-B context length. Set `TB_PT_WALK_CONTROL=1` to add an
exact-parameter-matched compact pointwise residual tail at every nonzero
depth; it controls for ordinary adapter capacity without access to token
differences or causal trajectory summaries.

Frozen GPT-2 Walk-tail scale result (local, Tiny Shakespeare)
--------------------------------------------------------------
Local GPT-2 was entirely frozen and received zero-init tails after its final
decoder block. At T=64, batch=1, 160 tail-only updates, and the same held-out
windows, the frozen baseline CE was 5.19709. Walk depth 1/8/32 reached
5.18425 / 5.10425 / 4.91275, improvements of 0.01283 / 0.09283 / 0.28433.
Disabling each trained tail restored the baseline CE, so those gains genuinely
came through the tail rather than an evaluation bookkeeping error.

However, the depth-32 Walk tail had 393,344 trainable parameters and an exact
parameter-matched pointwise compact SiLU-gated tail reached 3.81816 (a 1.37892
improvement), far ahead of Walk's 0.28433. The pointwise control has no token
differences, causal summaries, or trajectory features. Therefore this experiment
does *not* support adding a huge pure Walk tail to Phase B: it supports a
general post-backbone adapter capacity effect, and says that the present Walk
operator is a weak use of that capacity. Any future Walk-tail proposal must beat
this matched control on several seeds and on a long-context retrieval task.

Corrected multi-seed tail study and OOD check (local GPT-2)
-----------------------------------------------------------
The first scale sweep used a fixed training-window order across nominal seeds;
that is insufficient because zero-init Walk otherwise has no stochastic source.
`TB_PT_WALK_SEED` now affects the deterministic training-window order while all
evaluation windows remain fixed. The corrected 3-seed study used 100 updates,
T=64 training, the same 393,344 trainable parameters for both depth-32 arms,
and `dickens_raven.txt` as an untrained OOD evaluation text.

Mean CE over seeds: frozen=5.19709; Walk=5.00545 (-0.19164); pointwise=
3.94147 (-1.25562). On Shakespeare holdout, Walk/pointwise were respectively
5.13260/4.43387 at T=64 and 4.83043/3.99530 at T=128, versus frozen
5.24913/4.98371. On Dickens/Raven, Walk modestly improved both lengths
(3.11159 vs 3.15568 at T=64; 4.52825 vs 4.55861 at T=128), whereas pointwise
was slightly worse (3.19118 and 4.61794). Walk cost 0.536 s/train step versus
0.178 s for pointwise. Thus compact tails are strong *in-domain* adapters, but
their 1+ nat gain is not general-purpose language improvement; present Walk may
carry a small robustness signal, not enough alone to justify its compute. The
next fair test allocates a fixed tail budget to both compact transforms and a
shared Walk modulation, asking whether Walk improves the compact tail's OOD CE.

Compact-plus-Walk fixed-budget result (local GPT-2)
----------------------------------------------------
The hybrid used 31 compact pointwise layers plus one `WalkAdapter` shared at
all 32 residual positions. It has exactly 393,344 parameters, the same as the
depth-32 pointwise and Walk arms. Across the corrected 3-seed, 100-update
study, hybrid CE was 3.93875 in-domain, 4.43250 at Shakespeare T=64, and
3.99235 at T=128. Those are only 0.002--0.003 nats better than pointwise alone,
well inside this small experiment's noise. Its OOD CE was 3.18607/4.61879 at
T=64/128, effectively the same as pointwise (3.19118/4.61794) and still worse
than frozen at T=64. It cost 0.680 s/train step versus pointwise's 0.178 s.
Conclusion: current Walk modulation does not earn its roughly 4x compute cost
inside a compact adapter tail. Keep the compact tail as the stronger adapter
candidate; keep Walk light in the base model only until a task demonstrates a
unique long-context or OOD benefit.

Learned-subspace Walk result (local GPT-2, corrected 3-seed study)
------------------------------------------------------------------
`SubspaceWalkAdapter` was tested at D=768, rank=7, depth=32. It has 394,560
parameters, only 0.3% above the 393,344 legacy Walk budget, but learns cross-
channel trajectory directions through D-by-r/r-by-D projections rather than
raw-coordinate diagonal gains. With the same 100-update Shakespeare setup it
reached mean CE=4.57990 (-0.61719 vs frozen), versus legacy Walk=5.00545
(-0.19164) and compact pointwise=3.94147 (-1.25562). Its held-out CE was
4.83747/4.35612 at Shakespeare T=64/128 and 3.04829/4.43075 at Dickens/Raven
T=64/128. That OOD improvement over frozen (3.15568/4.55861) is materially
larger than legacy Walk's. It ran at 0.344 s/train step: about 1.6x faster than
legacy Walk (0.536 s), although still slower than compact pointwise (0.178 s).
This is a promising scalability repair, not a Phase-B-ready result: next test
rank 7/32/64 inside a fresh long-context custom model, compare to an equal-FLOP
low-rank compact adapter, and require multi-seed FineWeb long-retrieval gains.

Gated Subspace Walk result (local GPT-2, corrected 3-seed study)
-----------------------------------------------------------------
Adding `correction * SiLU(W_gate[u, correction])` in rank-7 subspace cost only
3,136 parameters across the 32-layer tail (397,696 total, +0.8%). Mean CE
improved from ungated Subspace Walk's 4.57990 to 4.55367. It also improved
Shakespeare T=64/T=128 from 4.83747/4.35612 to 4.83582/4.34135 and
Dickens/Raven from 3.04829/4.43075 to 3.03715/4.42233. Step time rose from
0.344 s to 0.382 s (+11%). This is a consistent modest gain and makes gated
Subspace Walk the best trajectory-only candidate so far. It remains far behind
pointwise compact tails for in-domain adaptation, so retrieval should be tested
next as a separate long-context memory mechanism rather than merged into 1B.

Chunk-Trajectory Memory result (local GPT-2, corrected 3-seed study)
----------------------------------------------------------------------
The clean memory arm starts from the gated rank-7 subspace adapter and adds
only a causal read over mean summaries of *completed* 16-token chunks. Each
token queries those summaries, uses hard top-1 in the forward pass, and uses a
soft straight-through route for router gradients. It has 403,968 trainable
parameters (+1.6% over gated subspace), versus 397,696 for the control. At the
same 100 updates and T=64, mean in-domain CE was 4.54146 versus 4.55367 for
gated subspace: a small 0.01221-nat improvement. Shakespeare CE improved
4.83582 -> 4.82313 at T=64 and 4.34135 -> 4.31588 at T=128. But Dickens/Raven
got worse (3.03715 -> 3.04267 at T=64; 4.42233 -> 4.43657 at T=128), and the
unfused reference took 0.87798 s/step versus 0.38187 s/step (2.30x slower).
Conclusion: it establishes a causal learned trajectory-memory path and a weak
in-domain long-context signal, but not a robust generalization or throughput
win. Do not enable it in Phase B. Its next fair evaluation, only if revisited,
is a fused implementation on a deliberately delayed-recall task plus diverse
FineWeb validation; it must beat gated subspace without an OOD regression.

Content-addressed Subspace Walk versus Transformer control (local GPT-2)
-------------------------------------------------------------------------
To test the actual claim fairly, the rank-7, depth-32 content-addressed Walk
tail (403,968 parameters) was compared with a conventional low-rank causal
attention plus SiLU-gated MLP tail (404,192 parameters). Both have the same
D-to-r/r-to-D bottleneck, exact zero-init residual start, frozen GPT-2
backbone, 100 T=64 updates, and corrected seeds 17/29/43. Content Walk adds a
causal ELU+1 associative read over prior r-space states to its gated trajectory
features; the control instead spends its r-space computation on SDPA and a
gated feed-forward update.

Mean in-domain CE was 4.43847 for content Walk and 4.42721 for the conventional
control, so Walk did not win its primary CE comparison. Shakespeare T=64 was
4.75735 versus 4.69503; T=128 was effectively tied (4.26891 versus 4.26787).
However, content Walk transferred much better to untouched Dickens/Raven:
3.04784 versus 3.11832 at T=64 and 4.45457 versus 4.56550 at T=128. It cost
0.64566 s/step versus 0.32756 s/step (1.97x slower). This is evidence that
content-selective subspace trajectories carry a useful robustness bias, but
not evidence that they beat a conventional transformer-style addition overall.
Do not use it for Phase B yet. The next justified experiment is a strict
equal-FLOP mixture or a diverse long-context FineWeb evaluation, not another
uncontrolled Walk feature.

Selective Associative Path Mixer screen (local GPT-2, single-seed rejection)
-----------------------------------------------------------------------------
`SelectiveAssociativePathAdapter` changes the content-memory payload from raw
subspace state to a learned mix of d1/d2/d4 trajectory values. It is a fresh
architecture selected by `TB_WALK_PATH_MIXER=1`; the paired tail arm is
`selective_path_mixer`. At D=768/r=7/depth=32 it has 407,104 parameters,
within 0.8% of the 404,192-parameter low-rank transformer control. On the
fixed seed-17, 100-update screen, it reached CE=4.48728 versus 4.37218 for the
control, and took 0.63007 s/step versus 0.34543 s/step. OOD T=64/T=128 was
3.04230/4.42850 versus 3.12502/4.56726, preserving the broader trajectory
robustness pattern but not enough to offset its primary loss/throughput miss.
This clear first-seed failure is intentionally not promoted to a multi-seed or
Phase-B test. Keep the implementation as a documented negative ablation.

Dual-Payload Associative Path Mixer (current screen)
----------------------------------------------------
`DualPayloadAssociativePathAdapter`, selected only for a fresh model by
`TB_WALK_DUAL_PATH_MIXER=1` (or as tail arm `dual_path_mixer`), corrects the
motion-only ablation rather than adding another opaque layer. One causal
ELU+1 associative address reads two prior-only payloads: learned subspace
state `u` for static semantic evidence, and learned d1/d2/d4 values for
changes. A channelwise learned gate mixes those reads before the ordinary Walk
correction. Thus the model, not a hand rule, selects whether a token needs
evidence, motion, or both. Complexity remains O(T*D*r + T*r^2), there are no
recurrent slots or token-pair scores, and the inherited zero output projection
makes construction an exact residual no-op. The structural test covers no-op,
strict causality, and gradients through both payloads and the selector. It is
only near parameter-matched at a specified (D,r); calculate its rank budget
again before any Phase-B comparison. Fixed seed-17 GPT-2, 100 updates,
D=768/r=7/depth=32: dual payload had 414,944 trainable parameters, CE=4.46047
and 0.85082 s/step; the 404,192-parameter low-rank Transformer control had
CE=4.37508 and 0.30040 s/step. T=64/T=128 OOD CE was 3.03906/4.43413 versus
3.12552/4.56269, so it retained the familiar robustness bias but failed the
primary CE and throughput screen. This is a documented negative ablation, not
a Phase-B candidate and not worth multi-seed expenditure in its present form.

Evidence Walk + depth-trajectory NI candidate
---------------------------------------------
`EvidenceWalkAdapter` is the next clean synthesis of the NI and Walk ideas. It
does not filter text, delete "wrong" tokens, or carry recurrent slots. It builds
one alternate view from the current hidden state: low-rank token state, local
exact r-space evidence, causal prior evidence, and trajectory features. In a
pretrained tail, each adapter also receives the model-state depth trajectory:
velocity is the current tail state minus the previous tail state, and curvature
is the change in that velocity. This is the gradient-boosting analogy we keep:
each stage predicts a correction using the direction the representation is
already moving, not a fresh replacement answer.

Enable it only in fresh experiments with `TB_WALK_EVIDENCE=1` or in the frozen
tail probe with `TB_PT_WALK_ARMS=evidence_walk`. The output projection is
zero-initialized, so it is an exact no-op at construction. This is not Phase B
until it beats the existing compact/low-rank controls across seeds and held-out
long-context recall.

Gradient bridge (opt-in)
------------------------
`TB_NI_BRIDGE=1` adds a training-only bounded gradient teacher: the exact
cross-entropy gradient at the final NI base state is computed analytically from
the LM head, normalized, and used as a small negative-direction target with
`TB_NI_BRIDGE_WEIGHT` (default 0.01).  No target gradient is used in decoding.

Neural Interpreter experiment (opt-in)
--------------------------------------
`TB_NI=1` adds the experimental Neural Interpreter.  `TB_NI_MODE=final` is
the original post-final-LayerNorm probe: it reads only
`x[:, -1:]` and injects at the final prediction position.  `TB_NI_MODE=chunk`
uses a low-rank recurrent state over `TB_NI_CHUNK_SIZE` tokens; state committed
by chunk `j` is injected into chunk `j+1`, so every token receives a causal NI
update without an extra quadratic attention operation.  `TB_NI_MODE=view` is
the current alternate-view experiment: one stateless causal feature generator
is inserted after `TB_NI_LAYER` (the middle block by default).  It uses
left-padded depth-wise temporal convolutions (3/9/33 tokens), optional token
velocity/acceleration, and a token-local feed-forward mixer; it has no RNN,
recurrent slots, chunk memory, filter/refiner, or perspective switch.  CE and
any auxiliary utility labels jointly decide what information the single view
should add.  `TB_NI_STEPS` selects K token-local refinement iterations and
`TB_NI_TRAJECTORY=0/1` disables/enables velocity and acceleration features.
`--ni-aux-probe DATASET [STEPS]` runs a small controlled accepted/rejected CoT
test inside this trainer.  It reports answer CE, paired preference accuracy,
and a deterministic local-order fluency proxy.  The latter is not a general
grammar or truth metric.  This is intentionally a fresh-architecture
experiment, not a normal resume of a checkpoint made with `TB_NI=0`; use
`TB_INIT_CKPT` for weights-only transfer.
`--ni-self-probe DATASET [STEPS]` is the accepted-only counterpart: CE trains
the observed continuation while a causal prefix is trained to retrieve its own
later hidden representation among other real continuations in the batch.  It
does not consume rejected answers or correctness labels.
The current local multimodal probe showed 95.8% held-out nuisance validation
with both streams and 50.0% when either stream was ablated; sparse unseen-pair
validation reached only the majority baseline, so paired data coverage remains
required.  These results are research notes, not evidence for enabling NI in a
production pretraining run yet.

Trajectory/K ablation (local cached CLIP+Wav2Vec probe, 1,000 optimizer steps)
-------------------------------------------------------------------------------
On the same covered-pair task (both / no-image / no-audio validation accuracy):
  * K=1, trajectory=off: 1.000 / 0.500 / 0.500
  * K=1, trajectory=on : 0.958 / 0.500 / 0.500
  * K=2, trajectory=off: 0.958 / 0.500 / 0.500
  * K=2, trajectory=on : 0.958 / 0.500 / 0.500
This small task validates both-stream dependence but gives no evidence that
velocity/acceleration or a second refinement step improves it.  The stricter
sparse unseen-pair split reached only the majority baseline.  Treat this as a
mechanism smoke test; the next meaningful test is a text LM loss/perplexity
comparison with identical data, checkpoint, optimizer, and token budget.

Neural Evidence Interpreter (research mode)
-------------------------------------------
`TB_NI_MODE=evidence` is the current non-recurrent redesign.  It is a shared
evidence workspace, not a filter, a correction-only head, an RNN, or a slot
memory.  At several transformer depths (default quarter/middle/three-quarter,
or `TB_NI_LAYERS="3,7,11"`) the current hidden state queries the same
immutable evidence bank.  The query includes a depth trajectory: velocity is
the change since the preceding evidence tap and curvature is the change in
that velocity.  This is ordinary feed-forward depth computation, never state
carried across tokens or decoding steps.
The residual returned to the LLM remains D-wide so it can be added safely, but
the evidence workspace can be wider: `TB_NI_WIDTH_MULT` multiplies
`TB_NI_RANK` for `TB_NI_MODE=evidence` only.  The default is 1; set it to 2 or
4 for an explicit wider ablation, so the normal rank 32 setting creates a
32-wide internal evidence representation before the
learned D-wide projection.  This is capacity, not a claim that wider is better;
compare width 1/2/4 at equal token budget before selecting a 1B configuration.

For text, the bank contains multiscale completed spans (4/16/64 tokens,
adaptively coarsened for long contexts).  A token may attend only to a span
whose final token is already in its prefix, preserving next-token causality.
For an input that already has image/audio encodings, pass
`evidence={"image": [B,M,D], "audio": [B,N,D], ..._mask: [B,M/N] bool}` to
`WalkGPT.forward`; modality and ordinal tags let those static feature sequences
meet before text decoding.  A learned *per-forward* whole-evidence node also
reads every valid static feature; it is a queryable summary node, not recurrent
state or a carried slot.  This initial implementation deliberately accepts
pre-encoded D-wide features rather than silently adding an untrained raw
vision/audio encoder.  Missing modalities are valid.  A new evidence model is
a fresh architecture: use `TB_INIT_CKPT` for weights-only transfer, never an
ordinary resume of a `TB_NI=0` run.  The built-in tests establish causal
masking and synthetic two-stream binding only; they do not establish a language
loss improvement or justify enabling it in the active 1B run.
`--ni-evidence-probe [STEPS]` runs that synthetic binding smoke test entirely
inside this trainer.  It is intentionally not a language/correctness benchmark:
the two pre-encoded streams carry independent bits and only their XOR selects
the final token.  The text prefix is constant, so ordinary CE training has no
answer signal and should stay at chance.
`--ni-evidence-inspect [STEPS]` trains the same tiny model then captures the
per-depth residual, zeroes only that residual, and applies image/audio
counterfactuals.  This is the right way to inspect an NI output: its D-wide
vector has no fixed word-level coordinate system, so causal logit mediation and
recoverable information are more meaningful than printing raw dimensions.

Archive-verified compiler probe (offline, opt-in)
--------------------------------------------------
`--archive-compiler-probe [LM_STEPS] [COMPILER_STEPS]` trains a tiny local
character LM, freezes one original GELU MLP as an exact archive, and fits the
nonstandard low-rank Hadamard operator `diag*x + U((Vx)*tanh(Wx))` to that
MLP's real hidden-state input/output pairs. It reports direct approximation,
full-LM CE after all-compiled replacement, CE with a learned error router and
archive fallback, coverage, parameter count, and measured path timing. This is
an offline compiler test only: it neither touches a checkpoint nor justifies
changing a running 1B architecture. `TB_ARCHIVE_RANK` and
`TB_ARCHIVE_COVERAGE` control the candidate cost/coverage tradeoff.
`TB_ARCHIVE_ROUTE_CHUNK` (default 16) additionally evaluates the intended
archive retrieval policy: one contiguous token group is sent either to the
compiler or to the precise frozen pre-compilation MLP. It is distinct from
`ArchiveRetrievalAttn`, which stores attention K/V chunks and is unrelated to
subnetwork archival.

Attention archive compiler (offline, opt-in)
---------------------------------------------
`--attention-compiler-probe [LM_STEPS] [COMPILER_STEPS]` performs the literal
attention analogue of the MLP archive test: it freezes a trained full-attention
residual and fits the same pointwise Hadamard compiler to real causal
pre-attention hidden states and archive outputs.  It reports replacement CE,
relative output MSE, parameter count, and direct timing.  It intentionally has
no token-level fallback router: unlike an MLP, an attention archive output for
one token depends on the complete causal prefix, so sparse fallback would still
need to run the archive for the sequence and would not save compute.  A poor
result is useful evidence that attention's dynamic routing cannot be compiled
into this local operator.  Use `TB_ATTN_ARCHIVE_D`, `TB_ATTN_ARCHIVE_RANK`, and
`TB_ATTN_ARCHIVE_LAYER`; `TB_ATTN_ARCHIVE_T` and `TB_ATTN_ARCHIVE_BATCH` set
the routing context and collection batch. This remains a fresh tiny probe only.

Attention compiler cloud result (FP8 QLinear, Tiny Shakespeare)
---------------------------------------------------------------
At D=512, T=256, block 0, after 2,000 base-LM steps and 12,000 compiler-fit
steps, the archived full attention path had 1,048,576 parameters. The same
pointwise Hadamard compiler used for the MLP archive did reduce path cost, but
could not preserve the contextual attention residual:
  * r=160: 246,784 params, 0.46352 -> 0.10761 ms, MSE 0.37603, CE +0.48215
  * r=256: 394,240 params, 0.46239 -> 0.11245 ms, MSE 0.37591, CE +0.47627
  * r=320: 492,544 params, 0.40328 -> 0.12548 ms, MSE 0.37038, CE +0.47439
Increasing rank did not materially improve either MSE or CE, so this is not a
simple capacity shortfall. The attention residual depends on dynamic causal
token-to-token routing and cannot be compiled into this token-local operator.
Do not apply this compiler to a 1B attention layer. A future attention archive
would need a sequence-aware routing surrogate, not a larger pointwise MLP.

Archive compiler result (local Tiny Shakespeare, trained D=256/L=2 probe)
-------------------------------------------------------------------------
After 150 LM steps and 300 compiler-fit steps, a rank-64 Hadamard compiler
replaced one frozen GELU archive MLP: archive=262,144 parameters, compiler=
49,664 (5.28x smaller); direct archive=0.54984 ms versus compiler=0.34492 ms
(1.59x faster); held-out CE changed 2.21094 -> 2.21484 (+0.00391). A learned
80%-coverage token router improved CE to 2.21354 but took 0.87719 ms, slower
than the archive because its sparse fallback defeats GPU batching. This is a
single-seed local result, not evidence that every trained path compiles; it
does establish that fixed offline block replacement is worth further testing,
whereas dynamic per-token routing is currently a throughput loss.

Exact archive retrieval dispatch result (local Tiny Shakespeare)
-----------------------------------------------------------------
The intended archived-subnetwork experiment was repeated with a D=256/L=2
model trained for 600 steps, a rank-80 compiler fitted for 1,200 steps, and
80% predicted compiler coverage. The exact archive had 262,144 parameters and
the estimate 61,952. Baseline/compiled/token-routed/chunk-routed held-out CE
was 1.92383 / 1.93099 / 1.92904 / 1.92969. Thus both fallback methods recover
some compiler fidelity by executing the actual frozen pre-compilation MLP for
hard inputs. The contiguous 16-token router is the correct dense-dispatch
form, and improved 2.05829 ms token routing to 1.75180 ms, but the archive was
only 0.52954 ms (compiler 0.60801 ms) on this small GPU workload. It therefore
does not yet earn deployment: dispatch costs dominate when the original MLP is
already a single efficient dense GEMM. The concept remains viable only where a
larger archived operator is materially slower than its approximation and the
expert dispatch can be fused/grouped by the serving runtime.

Archive compiler cloud scale ablation (FP8 QLinear, Tiny Shakespeare)
-----------------------------------------------------------------------
All runs trained the tiny LM for 1,200 steps, then fit the compiler for 2,400
steps unless otherwise stated. At D=512, the archived GELU MLP has 1,048,576
parameters. Rank sweep on block 0:
  * r=64:  99,328 params, 0.45877 -> 0.07624 ms (6.02x), CE +0.03255
  * r=96: 148,480 params, 0.45836 -> 0.08568 ms (5.35x), CE +0.02474
  * r=128:197,632 params, 0.45835 -> 0.08515 ms (5.38x), CE +0.02214
  * r=160:246,784 params, 0.45818 -> 0.10823 ms (4.23x), CE +0.00911
The r=160 direct replacement is the first near-loss-budget point. Independent
1,200/3,600 r=160 depth runs showed ordinary GPU variation: block 0 CE +0.01172
(relative MSE 0.03278), block 1 CE +0.00911 (relative MSE 0.06089). Both gave
about 4.23x path speedup. The learned 78%-coverage router reduced direct CE to
+0.00651 in both depth runs, but took 0.44918/0.47137 ms versus the archive's
0.45872/0.45837 ms: sparse fallback eliminates a reliable speed gain.

At D=1024 with rank 320, 1,500 LM steps, and 4,000 compiler steps, the archive
had 4,194,304 parameters and compiler 985,088; path time fell 1.49740 ->
0.21224 ms (7.05x), but direct CE regressed +0.07878 (router +0.05794, relative
MSE 0.10838). A later D=1024, 2,000-LM-step, 12,000-compiler-step rank sweep
on block 0 was still outside the loss budget:
  * r=448: 1,378,304 params, 1.49296 -> 0.25849 ms (5.78x), CE +0.06445
  * r=512: 1,574,912 params, 1.49495 -> 0.28131 ms (5.31x), CE +0.06966
  * r=640: 1,968,128 params, 1.49537 -> 0.30795 ms (4.86x), CE +0.10938
The 78-79% router reduced CE to +0.04102/+0.04557/+0.07227 but spent
0.68-0.74 ms on the combined route, so it remained slower than the compiler
and not reliably faster than the archive after fallback. Therefore rank
proportional to D is not sufficient, and bigger rank alone can overfit the
teacher traces. Do not apply the compiler to the 1B checkpoint or multiple
blocks until a candidate reaches the same <=0.01 CE criterion across at least
three seeds.

Hierarchical-attention probe (offline, parameter matched)
----------------------------------------------------------
`--hier-attn-probe [STEPS]` compares all-full causal attention with a native
hierarchical alternative in fresh tiny RoPE LMs. It keeps the exact same QKV
and output projections (and asserts total parameter equality): the hierarchy
uses parameter-free mean K/V summaries of complete earlier chunks plus exact
causal token attention inside the current chunk. With chunk size C this costs
O(T*C + T*T/C) attention-score work, rather than O(T^2). It is not an archive
compiler and never modifies/resumes the 1B checkpoint.

The probe reports held-out text CE at `TB_HIER_EVAL_T`, direct attention timing,
candidate count, and a separately-trained long key/value recall task. The
recall task matters because ordinary short-text CE can hide loss of evidence
farther than C tokens away. Relevant knobs: `TB_HIER_D`, `TB_HIER_L`,
`TB_HIER_HEADS`, `TB_HIER_TRAIN_T`, `TB_HIER_MAX_T`, `TB_HIER_CHUNK`,
`TB_HIER_LAYERS` (`all` or e.g. `1,2`), and `TB_HIER_RECALL_STEPS`. Accept a
candidate only if it has a real long-length speed gain while holding CE and
recall near the full-attention baseline; deterministic summaries are the
parameter-matched control before learned-summary experiments.

Exact-K/V archive retrieval probe (offline, parameter matched)
---------------------------------------------------------------
`--archive-retrieval-probe [STEPS]` tests the sequence-aware archive design.
Completed chunks retain exact token K/V vectors; a per-head mean-K landmark is
a parameter-free index; each query opens the exact K/Vs of its `TB_AR_TOPK`
highest-scoring prior chunks and attends to those plus its causal local chunk.
QKV/proj weights and total parameter count match full attention exactly. This
is deliberately distinct from the failed attention compiler: it preserves
dynamic query-dependent retrieval. `TB_AR_D`, `TB_AR_L`, `TB_AR_HEADS`,
`TB_AR_TRAIN_T`, `TB_AR_MAX_T`, `TB_AR_CHUNK`, `TB_AR_TOPK`, `TB_AR_LAYERS`,
and `TB_AR_RECALL_STEPS` configure the isolated test. The first version uses
hard top-k routing in the forward pass. `TB_AR_STRAIGHT_THROUGH=1` (default)
adds a dense soft-landmark read only as a backward surrogate: it cancels
numerically in the forward pass, so serving remains exact-K/V top-k but the
index gets gradient from every candidate landmark. Set it to 0 for the older
pure hard-routing ablation. On scattered recall
tasks it also reports `route_hit_any_head` and `route_hit_all_heads`: whether
the final archive layer actually opened the physical chunk holding each answer.

Archive retrieval control result (FP8 QLinear, Tiny Shakespeare)
-----------------------------------------------------------------
The same D=256/L=4 C=32 run with top-k=1 and a simple two-pair long-copy
control trained for 8,000 recall steps reached full=1.00000 and archive=
1.00000 accuracy (`recall_control_learned=1`). Text CE was also preserved or
better in that seed (T=128: -0.00922; T=256: -0.02034; T=512: -0.10349), while
the reference path remained slower than Flash/SDPA (T=512: 0.44228 -> 3.77260
ms). This proves that one exact archived K/V chunk can be opened and used for
a distant answer. It does not prove index selectivity because both pairs were
in the same first chunk. Set `TB_AR_RECALL_SPACING` to distribute pairs across
separate chunks; then top-k=1 must choose the right chunk per query.

Selective archive-index result (FP8 QLinear, Tiny Shakespeare)
---------------------------------------------------------------
With the same D=256/L=4/C=32/top-k=1 configuration, two key/value pairs were
placed at offsets 0 and 32 (`TB_AR_RECALL_SPACING=32`) and queried at the end
of a 128-token sequence. After 8,000 recall-training steps, full attention and
archive retrieval both reached 1.00000 held-out accuracy; the full control was
valid (`recall_control_learned=1`). This is the first clean evidence that the
parameter-free landmark index can select useful exact K/V archive content,
rather than merely access one fixed early chunk. It remains a two-chunk,
single-seed mechanism result. Next scale pairs across 4+ chunks and compare
top-k=1 against top-k=2 before considering a fused implementation or 1B branch.

Hard-routing failure and straight-through repair (local, FP8 QLinear)
---------------------------------------------------------------------
The stricter four-pair scattered-recall control exposed a real weakness in the
original hard top-k index. With D=256/L=4/C=32/top-k=1, pairs spaced across
four chunks, 2,000 text steps and 12,000 recall steps, hard routing scored
0.25260 held-out recall (chance is 0.25); its measured final-layer route hit
rate was zero. The full-attention control reached 0.58854, so that particular
control is also under-trained and is not a quality comparison.

The reduced local follow-up enabled `TB_AR_STRAIGHT_THROUGH=1`: D=128/L=2,
C=16/top-k=1, 500 text steps, then 4,000 scattered four-pair recall steps.
It reached archive recall=0.68490 versus full=0.25000 chance, with
route_hit_any_head=0.45443 and route_hit_all_heads=0.00260. Text CE was still
slightly worse (T=64: +0.00886; T=128: +0.02314), and the Python reference
was slower (T=128: 2.158 -> 24.691 ms). This is evidence that dense index
gradients teach some heads to open relevant archival chunks. It is not yet a
language-quality or speed win: repeat with several seeds and a full-attention
control that itself learns the scattered task before any production decision.

Archive retrieval cloud pilot (FP8 QLinear, Tiny Shakespeare)
--------------------------------------------------------------
At D=256/L=4/H=4 with C=32, top-k=2, all attention layers replaced, and 2,000
text-training steps, parameter counts matched exactly (2,266,112 each). Held-
out text CE was nearly preserved: at T=128 the archive path was +0.00121, at
T=256 it was -0.00499, and at T=512 it was -0.06319 versus full attention.
The Python reference implementation was much slower than Flash/SDPA (0.385 ->
1.268 ms at T=128; 0.450 -> 3.971 ms at T=512), so this is no throughput win.
The accompanying eight-pair associative-recall control was invalid: full
attention reached 0.11979 accuracy, essentially its 1/8 chance level, and the
archive path reached 0.12174. Do not interpret their +0.195 pp difference as
recall preservation. Re-run a simpler two-pair, long-distance control until the
full-attention baseline exceeds 0.80 before assessing archive recall.

Evidence-workspace binding result (local, width multiplier 1, 500 optimizer steps)
---------------------------------------------------------------
The built-in XOR probe was run after adding the stateless whole-evidence node:
  * standard text-only backbone: 0.499 final-token accuracy
  * evidence, image + audio:    1.000
  * evidence, image only:       0.493
  * evidence, audio only:       0.488
The result verifies that CE can train this shared workspace to combine two
input-time feature streams and that neither single stream carries the answer.
It does *not* establish visual/audio understanding, language CE improvement,
or a solution to the pretraining-loss plateau.  Next evidence must come from
real aligned encoders/data and be evaluated against a parameter-matched LM.

Evidence-output inspection (local width multiplier 1 XOR probe, 500 optimizer steps)
------------------------------------------------------------------
The actual D-wide residual was captured at all four injection depths:
  * normal NI residual: final-token accuracy=1.000, target margin=+5.293
  * same model, NI residual multiplied by zero: accuracy=0.486, margin=-0.149
  * nearest-centroid readout from residual alone: held-out accuracy=1.000
  * flip image bit / flip audio bit: answer changed in 1.000 / 1.000 of cases
  * residual feature RMS change after image/audio flip: 1.5206 / 1.5201
So on this deliberately constrained task the residual is a usable, joint
evidence representation that the later LLM blocks rely upon, not noise.  This
does not yet say it contains human-readable concepts or that it is useful on
FineWeb; those require an aligned real-data, parameter-matched experiment.

Evidence-width pilot (same XOR task, 500 optimizer steps)
-----------------------------------------------------------
The narrow width=1 workspace learned the task.  Width=2 and width=4 first
pilots both remained at 0.486 accuracy and produced near-zero counterfactual
residual changes under the same optimizer schedule.  A temporary normal-output
plus zero-scalar-gate variant did not fix that cold-start result, so it was not
kept.  Wider capacity is available through `TB_NI_WIDTH_MULT`, but it is not a
default or an improvement claim until a properly tuned, matched experiment
beats width=1 on real aligned data.

NI language-loss probe (stored CoT accepted text, byte-level, 400 steps)
------------------------------------------------------------------------
Walk was disabled.  The ordinary baseline was widened for near-matched
capacity (NI adds its own parameters); validation is held out by record:
  * standard full (873,984 params): val CE=2.5961
  * NI full, bridge off (929,921): val CE=2.6094
  * NI full, bridge on  (929,921): val CE=2.6086
  * NI-only + bridge (304,769 trainable): val CE=5.6266
The bridge recovered only 0.0008 CE and NI-only failed to model the corpus.
This is consistent with the current trainer integration being final-token-only:
it cannot influence enough positions to be a credible whole-sequence plateau
intervention.  A future causal per-token/chunk NI must be tested separately.

Causal chunk NI smoke probe (local accepted-CoT bytes, 1,000 optimizer steps)
----------------------------------------------------------------------------
`small_lm_ni_chunk_probe.py` compared the same tiny trainer/data loop.  The
chunk path was causal, finite, and trainable, but remained slightly behind the
ordinary baseline after the longer run:
  * standard: val CE=2.3367
  * chunk K=1, trajectory off: val CE=2.3383
  * chunk K=1, trajectory on : val CE=2.3398
  * chunk K=2, trajectory on : val CE=2.3414
  * chunk K=1, adapter-only: val CE=4.5766
The standard and chunk models were not parameter-matched.  Trajectory and a
second refinement step did not help this plain language-loss task; keep NI as a
conditional/editor experiment until a task-grounded objective shows a win.

Retired recurrent-slot NI utility probe (accepted/rejected CoT bytes, 600 steps)
-------------------------------------------------------------------------------
`--ni-aux-probe E:\\CSST\\think\\dataset.jsonl 600` used a D=32/L=1 causal
byte LM, identical CE + read-only utility heads, identical seed/data split, and
the only architectural difference was `TB_NI_MODE=view`.  Held-out results:
  * standard: val CE=3.2578, paired preference=0.388, fluency proxy=0.381
  * view NI : val CE=3.2391, paired preference=0.525, fluency proxy=0.419
The view improved CE by 0.0188 and brought the accepted-vs-rejected signal just
above chance.  The local byte-order discriminator is still weak and is not a
grammar score; this is evidence that one shared alternate view can use a
task-grounded utility signal, not evidence that NI has solved correctness or
the 1B pretraining plateau.  Set `TB_NI_AUX_RESULT_PATH` to retain the probe's
machine-readable metrics outside the trainer source.

Stateless causal-view NI utility probe (accepted/rejected CoT bytes, 600 steps)
-------------------------------------------------------------------------------
After removing recurrent slots, the same D=32/L=1 matched probe used only
causal depth-wise 3/9/33 temporal convolutions plus token differences:
  * standard: val CE=3.2578, paired preference=0.388, fluency proxy=0.381
  * view NI : val CE=3.2516, paired preference=0.331, fluency proxy=0.512
This stateless variant gives a tiny CE change (0.0062), reaches chance on the
local-order proxy, and loses the paired-quality gain.  It is architecturally
cleaner and avoids RNN state, but it is not yet an improvement to deploy.  The
next test should change the feature basis or the task grounding, not claim a
win from this one result.

Accepted-only NI future-state probe (CoT accepted text, 600 steps)
-------------------------------------------------------------------
`--ni-self-probe E:\\CSST\\think\\dataset.jsonl 600` removed every rejected
candidate and correctness label.  It retained answer CE and trained a causal
prefix to retrieve its actual later hidden state from other real continuations
in the batch.  Same D=32/L=1, seed, split, and token budget:
  * standard: val CE=3.2391, future retrieval=0.144
  * view NI : val CE=3.2383, future retrieval=0.113
CE is indistinguishable at this resolution and the NI path reduced the
accepted-only retrieval measure.  Thus this self-prediction objective is not
the source of the prior win.  Accepted CE alone is ordinary language-model
training; quality-linked supervision (or a substantially better grounded
self-supervised objective) is needed before claiming NI learns useful
interpretation rather than an extra transformation.

Standard-training comparison (local deterministic toy LM, 250 optimizer steps)
-------------------------------------------------------------------------------
Same D=64/L=2 model shape, incrementing-token data, seed, optimizer, and token
budget; values are average validation CE and final-token top-1:
  * standard full training: 72,384 params, val=0.0125, top1=1.000
  * NI full, bridge off: 148,865 params, val=0.0145, top1=1.000
  * NI full, bridge on : 148,865 params, val=0.0140, top1=1.000
  * NI-only + bridge (LM frozen): 76,481 trainable, val=3.2500, top1=0.930
The gradient bridge helped NI-full slightly but did not beat ordinary full
training; NI-only is not a language-model replacement.  This task is easy and
not a perplexity benchmark, so do not extrapolate these numbers to 1B quality.

Operational concept probe (local arithmetic-code experiment, 1,200 steps)
-------------------------------------------------------------------------
The code encoder and NI were trained with execution-result CE plus an
alpha-renaming invariance loss.  On held-out variable/function names, result
accuracy was 1.000; equivalent renamed programs had slot distance 0.74725,
while changing `+` to `*` had distance 3.07705 (4.12x separation).  This shows
that behavior-anchored concepts are learnable operationally.  It does not mean
latent coordinates have an intrinsic universal meaning: without invariance,
equivalence, or execution objectives, concept vectors remain arbitrary.

Small local CoT language-loss probe (accepted reasoning, byte-level, 400 steps)
-------------------------------------------------------------------------------
The corpus was `E:\\CSST\\think\\dataset.jsonl` (1,845 records; 80/20 record
split).  Full attention was kept on in both models and the standard MLP was
widened for near-matched capacity:
  * standard: 702,976 params, val CE=2.6023, 37.3 seconds
  * Walk:     698,932 params, val CE=2.4844, 101.7 seconds
Walk was 4.5% lower in this small run with slightly fewer parameters, but 2.7x
slower.  The probe is byte-level, short, and not a 1B perplexity benchmark;
it supports further matched tests rather than proving plateau resolution.

Local 2025-2026 sweep findings (tinyshakespeare, D=128-256, T=64-512, 5k steps, train_1b (3).py:2186 HierarchicalEvidenceWalkAdapter)
--------------------------------------------------------------------
All sweeps isolated TB_HF_CKPT="" (no Papatpong/Walker push), 1B GPU0 untouched.
1) T=64 200 steps D=128: FULL 2.7125 (576k) vs LEGACY WalkAdapter 2.5219 (584k, -0.19) vs HIER_r32 2.5594 (668k, -0.15) — walk beats full at short T (reproduces 2.602->2.484).
2) LR schedule: 3x walk LR (2.5219==2.5219) no gain; late bump at 100 to 1e-3: LEGACY 2.521->2.412 -0.109, FULL 2.712->2.525 -0.187, HIER 2.559->2.45 -0.109. Plateau is schedule (cosine 6e-4->1e-6) not walk mult; 1B TB_LR_FLOOR=3e-5 already helps.
3) T=128 300 steps D=256: HIER_r16 2.1812, r32 2.1469 best, r64 2.1812 U-shape, GATED 2.1406, LEGACY 2.1125 best (-0.034 vs HIER), FULL 2.5562 (+0.44). r=32 optimal DoF=6*r (train_1b (3).py:2203).
4) T scaling HIER_r32 vs FULL (150 steps D=256): T64 2.40 vs 2.68 -0.27, T128 2.14 vs 2.55 -0.40, T256 2.26 vs 2.64 -0.38, T512 2.23 vs 2.66 -0.43 — HIER scales O(T*D*r) (train_1b (3).py:2330) not O(T^2).
5) DoF ablation T=128 300 steps: HIER_full 2.1469 vs d1-only 2.1781 +0.031 vs state-only 2.4625 +0.315, GATED 2.1406 — motion essential, d2/d4 marginal.
6) k vectors T*k walk-only (train_1b (3).py:2199) k=1/2/4 T=64 100 steps: 3.3375 vs 3.3344 flat; k dim-split D->k*(D/k) k=4 3.2906 -0.047 vs T*k — dim DoF > T steps.
7) Data trajectory Δmodel-Δdata cross_gate (train_1b (3).py:2324) TB_WALK_DATA_TRAJ=1: HIER 3.3375 -> data 3.1187 -0.219, data_k2 3.1094 -0.228 (100 steps T=64). Token emb 0.02 (train_1b (3).py:2921) grad via walk accelerates.
8) Final 5k T=64 D=128 W=200 6e-4->3e-5: BEST HIER_data_k2 702k 1.5656 best / 1.5375 final (901s) vs FULL 576k 1.6656 / 1.6008 (201s) -0.10 best, -0.06 final, 4.5x cost. Walk continues improving 1.90->1.56 while FULL plateaus 2.40->1.66.

Predicted result:
- Continued Phase A (14k left, T1024->2048) with HIER r32+data k=2 (TB_WALK_SUBSPACE_RANK=32 TB_WALK_HIER_EVIDENCE=1 TB_WALK_DATA_TRAJ=1 TB_WALK_HIER_K=2) via TB_INIT_CKPT (train_1b (3).py:45) should beat LEGACY by ~0.05-0.10 nats at 2-3k steps on fw_val_phase_a_2k.bin (train_1b (3).py:54), but not resolve 2.8 plateau alone — needs Phase B long T + archive C=64 (train_1b (3).py:2192).
- Encoder-decoder with pretrained emb (GPT2) as encoder, walk decoder replacing head (train_1b (3).py:2215 zero-init) is HIER+archive enc-dec already: enc -> static evidence (train_1b (3).py:3031) + walk traj cross. Dec-only catastrophic if walk replaces Attn QKV (train_1b (3).py:1053) +0.47 CE (train_1b (3).py:430); as sidecar Block:2475 x+attn+mlp+walk it is safe. Next probe: frozen enc (FULL 2.55) vs enc+HIER_data dec at T=128 300 steps.

2026-09-24 continuation decision (CSA2 and final-hidden adapter)
-----------------------------------------------------------------
Disposition: CSA2 is a failed arm for the 1B continuation. Do not present the
structural CSA2 bridge as a gain or enable it in the next 1B resume on the
strength of these runs. The cloud-imported root Downloads copy is not this
fixed SFT2 source: this SFT2 launcher sets a deterministic, source-specific
HF-stream shuffle buffer (`TB_STREAM_SHUFFLE_BUFFER=1024`, seed 71237). Treat
results from the stale import/shuffle path as non-equivalent; preserve the
source/launcher pair and verify its stream signature before interpreting a
continuation.

2026-09-26 continuation decision: the MoLab plan sets
`TB_STREAM_SHUFFLE_BUFFER=0` and consumes each source in yielded row order.
Weighted source selection and hashed-domain selection still occur per batch;
they are not the Hugging Face row-shuffle buffer.

The final-hidden supervised fast-adapter study remains a separate research
screen, not a proposed 1B addition. It used a randomly initialized width-32,
one-layer model (not the Walker checkpoint), the same eight-source cache
(SHA-256 `64c129ec4e81a228f376cbf161cfef426ee00cbd08b55ae53512a957921391f`),
500 updates, 2,048 held-out query tokens/source, and Modern vs rank-8 K=1.
Local BF16/CUDA weighted top-1 deltas were +1.32, +0.72, and -0.18 percentage
points for seeds 43/71/97 (mean +0.62 pp); top-5 deltas were -0.19, +0.86, and
-0.32 pp. The earlier CPU pilot had K=1 deltas +1.10 and -0.19 pp for seeds
43/71, so the sign is not stable across seeds/runs. K=1 also cost about
32-34 s/seed versus 10-21 s for Modern in the local run, with peak allocated
VRAM about 90 vs 66 MiB; the comparison is not FLOP-matched and is not a
reasoning-task score. These tiny from-scratch results do not establish transfer
to 1B continued pretraining. Do not transplant this adapter, its optimizer, or
its fast-update loop into this trainer or an ongoing checkpoint.

2026-09-26 Walker-native predictive-contrast decision
-----------------------------------------------------
The training-only Walker Predictive Contrast (WPC) follow-up is also rejected
for the 1B continuation.  In a real eight-source, matched-token small-model
screen, WPC versus its zero-weight Walker compute control changed weighted
top-1 by only +0.0402 percentage point and reasoning top-1 by +0.0552 point,
while top-5 fell 0.0842 point and CE worsened by 0.00452.  Repeated 4-grams
improved 5.06% relatively, but this was far below the promotion thresholds.
At approximately equal wall time, a modern Transformer trained for 2,300 steps
beat Walker+WPC trained for 600 steps by 3.325 top-1 points, 4.075 reasoning
top-1 points, and 0.3755 CE.  The zero-weight graph control and bare Walker
were bitwise identical, so this is not a broken-control result.  Do not add WPC
or its predictor/auxiliary optimizer state to the 1B checkpoint.

Current 1B recommendation: no new architecture module is justified by the
available controlled evidence. Continue only from the intended known-good
Walker checkpoint with the fixed SFT2 trainer/launcher and validated data,
validation, and resume contracts; keep CSA2 and the final-hidden adapter out.

"""

import math
import os
import json
import time
import glob
import random
import hashlib
import re
import collections
import threading
import shutil
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from contextlib import contextmanager, nullcontext
from urllib.request import Request, urlopen

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# Native Blackwell NVFP4 is deliberately an optional backend.  The existing
# QLinear FP4/FP8 paths are research STE fallbacks: their backward GEMMs are
# BF16.  Transformer Engine's NVFP4 recipe instead owns the quantized forward,
# dgrad and wgrad kernels (including its fused WGRAD RHT).  Keep the import
# optional so regular CUDA, CPU, and the small local Ada tests remain usable.
#
# 2026-09-08 integration record: local RTX 4050 has no Transformer Engine and
# cannot validate NVFP4; `nvfp4_te_preflight.py` correctly failed closed while
# the ordinary BF16 QLinear smoke test passed.  A 1B resume is authorized only
# after the Blackwell preflight prints `"nvfp4_ready": true`.
try:
    import transformer_engine.pytorch as _TE
    from transformer_engine.common.recipe import NVFP4BlockScaling as _TENvFP4Recipe
except Exception:  # TE is installed only on the Blackwell training image.
    _TE = None
    _TENvFP4Recipe = None

# A root WalkGPT forward keeps one TE autocast scope open for all QLinears.
# The counter also lets a QLinear safely fall back to a private scope in small
# direct probes/inference where it is called without the root module hook.
_TE_NVFP4_SCOPE_DEPTH = 0
try:
    from neural_interpreter import (
        CausalChunkNeuralInterpreter,
        NeuralInterpreter,
        ZeroGatedSlotInjector,
        gradient_alignment_loss,
    )
except ImportError:  # Optional dependency: default TB_NI=0 keeps legacy trainer standalone.
    CausalChunkNeuralInterpreter = None
    NeuralInterpreter = None
    ZeroGatedSlotInjector = None
    gradient_alignment_loss = None


def _redact_hf_credential(error):
    """Keep a session-only HF token out of trainer error logs."""
    message = str(error)
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        secret = os.environ.get(name)
        if secret:
            message = message.replace(secret, "[REDACTED]")
    return message


def _env_int(name, default):
    """Read an exact integer and identify malformed launch variables clearly.

    Legacy notebook controls can serialize an integral value as ``19000.0``.
    Accept that exact representation, but never truncate a fractional value or
    reinterpret a comma-separated list as a number.
    """
    raw = os.environ.get(name, str(default))
    text = str(raw).strip()
    if not text or "," in text:
        raise ValueError(f"{name} must be an exact integer; got {raw!r}")
    try:
        parsed = Decimal(text.replace("_", ""))
    except (InvalidOperation, ValueError) as error:
        raise ValueError(f"{name} must be an exact integer; got {raw!r}") from error
    if not parsed.is_finite() or parsed != parsed.to_integral_value():
        raise ValueError(f"{name} must be an exact integer; got {raw!r}")
    return int(parsed)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GPC_WALK_NORM = float(os.environ.get("TB_WALK_NORM", "0.0"))
GPC_WALK_NORM_MODE = os.environ.get("TB_WALK_NORM_MODE", "renorm").strip().lower()
if GPC_WALK_NORM_MODE not in ("renorm", "clip", "cap"):
    raise ValueError("TB_WALK_NORM_MODE must be 'renorm', 'clip', or 'cap'")
GPC_TRAJ_ACC_ON = float(os.environ.get("TB_TRAJ_ACC", "0.0")) > 0.0
GPC_TRAJ_SMOOTH_ON = float(os.environ.get("TB_TRAJ_SMOOTH", "0.0")) > 0.0
GPC_WALK_CHUNK_W = float(os.environ.get("TB_WALK_CHUNK_W", "0.0"))
DTYPE = torch.bfloat16
CAP = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)
if os.environ.get("TB_BLAS_LIBRARY", "") == "legacy":
    torch.backends.cuda.preferred_blas_library("cublas")


def _mem_allocator_breakdown():
    if not torch.cuda.is_available():
        return ""
    try:
        d = {}
        for s in torch.cuda.memory_snapshot():
            name = s.get("allocator_name") or s.get("segment_type") or "?"
            d[name] = d.get(name, 0) + s.get("total_size", 0)
        return " ".join(f"{k}={v/2**20:.0f}MB" for k, v in sorted(d.items()))
    except Exception:
        return ""


def _lib_workspace_stats():
    if not torch.cuda.is_available():
        return ""
    try:
        st = torch.cuda.memory_stats()
        return " ".join(f"{k}={st[k]/2**20:.0f}MB" for k in sorted(st)
                        if ("cublas" in k or "cudnn" in k) and "bytes" in k)
    except Exception:
        return ""

ALIBI_CAP = int(os.environ.get("TB_ALIBI_CAP", "128"))
ROPE_BASE = float(os.environ.get("TB_ROPE_BASE", "10000"))


def _rope_rotate(x, cos, sin):
    """Rotate half the head dim: x (*, hd) with cos/sin (T, hd/2)."""
    # x: (..., hd), cos/sin: (T, hd/2) -> broadcast
    hd = x.shape[-1]
    x1, x2 = x[..., : hd // 2], x[..., hd // 2 :]
    # cos/sin need T broadcast: (1, T, 1, hd/2) for B,H,T,hd/2
    # caller unsqueezes; here handle generic trailing dims
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


_ROPE_TABLES = {}


# Grow-only full-length RoPE tables for incremental generation (see
# _rope_row_tables). Kept separate from the bounded _ROPE_TABLES training
# cache so growing generation positions never evict training shapes.
_ROPE_FULL_TABLES = {}


def _rope_row_tables(hd, base, max_t, device, dtype):
    """Grow-only full-length RoPE tables for incremental generation.

    Rows depend only on (position, hd, base) — never on the requested length —
    so one table per (hd, base, dtype, device) serves every generate call.
    This deliberately bypasses the bounded _ROPE_TABLES training cache: per
    generated position, that cache would rebuild ever-longer tables and evict
    training shapes.
    """
    global _ROPE_FULL_TABLES
    key = (int(hd), float(base), str(dtype),
           device.type, device.index if device.type == "cuda" else -1)
    need = max(int(max_t), 1)
    have = _ROPE_FULL_TABLES.get(key)
    if have is None or have[0] < need:
        with torch.no_grad():
            inv_freq = 1.0 / (float(base) ** (
                torch.arange(0, int(hd), 2, device=device, dtype=torch.float32) / int(hd)))
            t = torch.arange(need, device=device, dtype=torch.float32)
            freqs = torch.outer(t, inv_freq)
            tables = (freqs.cos().to(dtype), freqs.sin().to(dtype))
        _ROPE_FULL_TABLES[key] = (need, tables[0], tables[1])
        if len(_ROPE_FULL_TABLES) > 8:
            _ROPE_FULL_TABLES.pop(next(iter(_ROPE_FULL_TABLES)))
        have = _ROPE_FULL_TABLES[key]
    return have[1], have[2]


def _rope_cos_sin(T, hd, base, device, dtype):
    """Return immutable RoPE tables shared by every layer at this shape.

    Phase-B performance audit (2026-09-09): the old path rebuilt arange,
    outer, cos and sin in all 16 layers (twice per layer for hybrid heads).
    These tensors have no trainable inputs, so sharing them is mathematically
    exact and does not add checkpoint state.  A small copy-on-write cache keeps
    the current train/eval lengths without growing across the full curriculum.
    """
    global _ROPE_TABLES
    dev_index = device.index if device.type == "cuda" else -1
    key = (int(T), int(hd), float(base), device.type, dev_index, dtype)
    cached = _ROPE_TABLES.get(key)
    if cached is not None:
        return cached
    with torch.no_grad():
        inv_freq = 1.0 / (base ** (torch.arange(0, hd, 2, device=device, dtype=torch.float32) / hd))
        t = torch.arange(T, device=device, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)  # (T, hd/2)
        tables = (freqs.cos().to(dtype), freqs.sin().to(dtype))
    retained = dict(_ROPE_TABLES)
    if len(retained) >= 8:
        retained.pop(next(iter(retained)))
    retained[key] = tables
    _ROPE_TABLES = retained
    return tables

# ---------------------------------------------------------------- precision ---

# fp4 dtype name varies across torch builds; resolve once at import.
# NOTE: some builds only expose the PACKED variant (float4_e2m1fn_x2), which
# cannot be cast to directly from float — probed in _fp4_usable().
_FP4_DTYPE = getattr(torch, "float4_e2m1fn", None)
if _FP4_DTYPE is None:
    _FP4_DTYPE = getattr(torch, "float4_e2m1fn_x2", None)
_FP8_DTYPE = getattr(torch, "float8_e4m3fn", None)
_FP4_OK = None  # lazily verified: dtype present AND float->fp4 cast works
# Runtime capability belongs to the backend/alignment class, not to each
# individual QLinear.  A process-wide cache prevents every layer from retrying
# the same unsupported _scaled_mm contract during the first validation pass.
# This is derived runtime state only and never enters checkpoints.
_QLINEAR_FAST_CAPABILITY = {}


def _fp4_usable():
    global _FP4_OK
    if _FP4_OK is None:
        if _FP4_DTYPE is None:
            _FP4_OK = False
        else:
            try:
                torch.eye(4, dtype=torch.float32).to(_FP4_DTYPE)
                _FP4_OK = True
            except Exception:
                _FP4_OK = False
    return _FP4_OK


def _quant_fp4(w32):
    """FP4 (e2m1) per-row quantization via native torch. Max E2M1 = 6."""
    s = torch.clamp(w32.abs().max(dim=-1, keepdim=True).values / 6.0, 1e-8)
    return (w32 / s).clamp(-6.0, 6.0).to(_FP4_DTYPE), s


class _FP8LinearSTE(torch.autograd.Function):
    """fp8 GEMM with a straight-through estimator: true _scaled_mm in forward,
    backward computed as if operands were the dequantized bf16 weights.
    (aten::_scaled_mm has no derivative, so plain autograd cannot train through
    it — this is the standard quantized-training workaround.)

    IMPORTANT: `wq`/`s` are the WEIGHT quantization, precomputed once per
    optimizer step by QLinear._refresh_q() — weights are fixed within a step,
    so re-deriving wq/s from `master` inside forward (the old behavior) was
    pure waste, and an expensive one: with gradient checkpointing every
    checkpointed block replays its forward during backward, so that redundant
    fp32-upcast + full-tensor amax reduction over EVERY linear's weight ran
    ~(accum + n_checkpointed_blocks) times per step instead of once.
    `master` is still threaded through only so autograd has a differentiable
    input to route the STE gradient (gw) to — its VALUE is unused here."""

    @staticmethod
    def forward(ctx, x2d, master, wq, s, w_deq):
        # Causal FP8 activation quantization: one scale per flattened token row.
        # The previous tensorwise amax reduced across [B*T,D], so future tokens
        # changed the scale (and therefore quantized values) of earlier tokens.
        # Rowwise [B*T,1] scaling removes that cross-token information channel.
        xs = torch.clamp(
            x2d.float().abs().amax(dim=-1, keepdim=True) / 448.0, 1e-8
        ).contiguous()
        xq = (x2d.float() / xs).clamp(-448, 448).to(_FP8_DTYPE)

        # Blackwell rowwise _scaled_mm requires BOTH scale tensors to carry the
        # rowwise layout. Weight quantization itself remains unchanged: `s` is
        # still the checkpoint-compatible tensorwise scalar; repeating it across
        # output columns is mathematically identical to the old weight dequant.
        sb = s.reshape(1, 1).expand(1, wq.shape[0]).contiguous()
        out = torch._scaled_mm(xq, wq.t(), scale_a=xs, scale_b=sb,
                               out_dtype=torch.bfloat16)
        ctx.save_for_backward(x2d, w_deq)
        return out

    @staticmethod
    def backward(ctx, gout):
        x2d, w_deq = ctx.saved_tensors
        # straight-through: gradients as if weights were the dequantized bf16
        g2 = gout.reshape(-1, gout.shape[-1]).to(torch.bfloat16)
        gx = (g2 @ w_deq).reshape(x2d.shape).to(gout.dtype)   # (B*, in)
        gw = (g2.t() @ x2d.reshape(-1, x2d.shape[-1]).to(torch.bfloat16)).to(torch.bfloat16)
        return gx, gw, None, None, None


class _FP4LinearSTE(torch.autograd.Function):
    """fp4 (e2m1) GEMM with a straight-through estimator. FP4 is symmetric with
    max |value| = 6 (2^1 × 1.0111). Per-tensor scale = amax/2 gives ±2 headroom
    like the fp8 path's /448; the backward uses dequantized bf16 operands.
    NOTE: fp8-style /6 scaling of the *weights* is applied here, NOT the /448
    of _FP8LinearSTE — using FP8 scales for FP4 weights was the hidden bug.

    Same caching contract as _FP8LinearSTE: wq/s come from QLinear._refresh_q()
    (once per optimizer step), not re-derived from `master` on every call —
    see that class's docstring for why this matters under checkpointing."""

    @staticmethod
    def forward(ctx, x2d, master, wq, s, w_deq):
        # One activation scale per flattened token row. A tensor-wide scale
        # would let future-token magnitudes alter earlier quantized activations.
        xs = torch.clamp(
            x2d.float().abs().amax(dim=-1, keepdim=True) / 6.0, 1e-8
        ).contiguous()
        xq = (x2d.float() / xs).clamp(-6, 6).to(_FP4_DTYPE)
        out = torch._scaled_mm(xq, wq.t(), scale_a=xs, scale_b=s,
                               out_dtype=torch.bfloat16)
        ctx.save_for_backward(x2d, w_deq)
        return out

    @staticmethod
    def backward(ctx, gout):
        x2d, w_deq = ctx.saved_tensors
        g2 = gout.reshape(-1, gout.shape[-1]).to(torch.bfloat16)
        gx = (g2 @ w_deq).reshape(x2d.shape).to(gout.dtype)
        gw = (g2.t() @ x2d.reshape(-1, x2d.shape[-1]).to(torch.bfloat16)).to(torch.bfloat16)
        return gx, gw, None, None, None


class QLinear(nn.Module):
    """Linear with DeepSeek-style precision: quantized weight storage (FP4 or FP8),
    bf16 master for optimizer updates, fp32 accumulation.
    - fast path: torch._scaled_mm (requires mat2 COLUMN-MAJOR — pass _qt = _q.t())
    - fallback: CACHED dequant bf16 GEMM (same speed as plain linear; the
      un-cached per-call dequant this replaces was several× slower than bf16).
    Falls back gracefully everywhere; correctness verified by one-time self-test."""

    def __init__(self, in_f, out_f, bias=False, quant="auto"):
        super().__init__()
        if quant == "auto":
            force = os.environ.get("TB_FORCE_QUANT", "").strip().lower()
            if force in ("fp4", "fp8", "none"):
                # test hook: pin the storage format on any hardware so the
                # exact production code path can be validated at small scale
                dtype_ok = _fp4_usable() if force == "fp4" \
                    else (_FP8_DTYPE is not None) if force == "fp8" else True
                quant = force if dtype_ok else ("fp8" if _FP8_DTYPE is not None else "none")
            else:
                quant = ("fp4" if CAP >= (10, 0) and _fp4_usable()
                         else "fp8" if CAP >= (9, 0) and _FP8_DTYPE is not None else "none")
        self.quant = quant
        self.master = nn.Parameter(torch.empty(out_f, in_f, dtype=DTYPE))
        self.bias = nn.Parameter(torch.zeros(out_f, dtype=DTYPE)) if bias else None
        self._q = None
        self._s = None
        self._qt = None
        self._wd = None
        self._wd_bwd = None
        self._fast_ok = None
        # Non-registered lazy TE bridge.  It aliases master/bias rather than
        # owning copies, so state_dict keys, optimizer state and old
        # checkpoints remain exactly QLinear.master / QLinear.bias.
        object.__setattr__(self, "_te_linear", None)
        object.__setattr__(self, "_te_recipe", None)
        nn.init.kaiming_uniform_(self.master, a=math.sqrt(5))

    @torch.no_grad()
    def _refresh_q(self):
        # TE owns its FP4 cache/scales and refreshes them inside the native
        # recipe.  Re-quantizing master here would both be wrong and add an
        # eager Python-side amax pass after every optimizer step.
        if self.quant in ("none", "te_nvfp4"):
            return
        w32 = self.master.float()
        try:
            if self.quant == "fp4":
                self._q, self._s = _quant_fp4(w32)
            else:  # fp8 e4m3
                amax = w32.abs().max()
                self._s = torch.clamp(amax / 448.0, 1e-8)
                self._q = (w32 / self._s).clamp(-448, 448).to(_FP8_DTYPE)
        except Exception as e:
            # packed-fp4 builds can't cast float->fp4 directly: degrade to fp8
            print(f"  QLinear {self.quant} refresh failed ({e!r}) — degrading to fp8", flush=True)
            self.quant = "fp8"
            amax = w32.abs().max()
            self._s = torch.clamp(amax / 448.0, 1e-8)
            self._q = (w32 / self._s).clamp(-448, 448).to(_FP8_DTYPE)
        # _scaled_mm requires mat2 column-major: t() of a contiguous (out,in)
        # tensor is exactly that. Invalidate the cached dequant.
        self._qt = self._q.t()
        self._wd = None
        self._wd_bwd = None

    def _backward_weight(self):
        """BF16 STE weight, optionally reused across accumulation microsteps.

        This is derived state only.  It is invalidated immediately after each
        optimizer update by `_refresh_q`, so caching changes neither checkpoint
        contents nor gradient mathematics.  At accumulation=16 it removes 15
        redundant FP8->FP32->BF16 full-weight conversions per optimizer step.
        The cost is roughly two bytes of VRAM per quantized weight.
        """
        cache = os.environ.get(
            "TB_QUANT_CACHE_DEQUANT",
            os.environ.get("TB_FP8_CACHE_DEQUANT", "1"),
        ) == "1"
        if cache and self._wd_bwd is not None:
            return self._wd_bwd
        with torch.no_grad():
            weight = (self._q.float() * self._s).to(torch.bfloat16)
        if cache:
            self._wd_bwd = weight
        return weight

    def _selftest(self, x):
        """One-time check: scaled_mm must match an explicit dequant GEMM built
        from the SAME fp8-quantized operands — isolating kernel correctness
        from the (legitimate) activation-quantization error."""
        device_index = x.device.index if x.device.index is not None else -1
        capability_key = (
            self.quant, x.device.type, device_index,
            self.master.shape[1] % 16, self.master.shape[0] % 16,
        )
        if capability_key in _QLINEAR_FAST_CAPABILITY:
            self._fast_ok = _QLINEAR_FAST_CAPABILITY[capability_key]
            return
        self._fast_ok = False
        try:
            # Ordinary training refreshes caches after build/resume, but small
            # research probes construct QLinears directly. The self-test must
            # establish its own quantized reference before reading _q/_s.
            if self._q is None or self._s is None:
                self._refresh_q()
            xt = torch.randn(64, self.master.shape[1], device=x.device, dtype=torch.bfloat16)
            if self.quant == "fp4":
                xs = torch.clamp(
                    xt.float().abs().amax(dim=-1, keepdim=True) / 6.0, 1e-8
                ).contiguous()
                xq = (xt.float() / xs).clamp(-6, 6).to(_FP4_DTYPE)
            else:
                # Match production: causal rowwise activation scales.
                xs = torch.clamp(
                    xt.float().abs().amax(dim=-1, keepdim=True) / 448.0, 1e-8
                ).contiguous()
                xq = (xt.float() / xs).clamp(-448, 448).to(_FP8_DTYPE)
            # reference: same quantized operands, plain matmul after dequant
            ref = (xq.float() * xs) @ (self._q.float() * self._s).t()
            scale_b = (self._s if self.quant == "fp4" else
                       self._s.reshape(1, 1).expand(1, self._q.shape[0]).contiguous())
            fast = torch._scaled_mm(xq, self._qt, scale_a=xs, scale_b=scale_b,
                                    out_dtype=torch.bfloat16)
            denom = max(1e-6, ref.abs().max().item())
            err = ((fast.float() - ref).abs().max() / denom).item()
            if err < 0.02:
                self._fast_ok = True
                print(f"  QLinear fast path OK ({self.quant}, rel err {err:.5f})", flush=True)
            else:
                print(f"  QLinear self-test FAILED ({self.quant}, rel err {err:.4f}) — using cached-dequant path", flush=True)
        except Exception as e:
            print(f"  QLinear fast path unavailable ({self.quant}, device={x.device}: {e!r}) "
                  "— using cached-dequant path", flush=True)
        _QLINEAR_FAST_CAPABILITY[capability_key] = self._fast_ok

    def forward(self, x):
        if self.quant == "te_nvfp4":
            return self._forward_te_nvfp4(x)
        if self.quant == "none":
            return F.linear(x, self.master, self.bias)
        if getattr(self, "_fast_ok", None) is None:
            self._selftest(x)
        if self._fast_ok:
            if self._q is None:
                # Defensive: fast path requires a cached quantization. This
                # should always be populated by build_model()/resume() before
                # the first forward, but refresh lazily rather than silently
                # computing from stale/absent state.
                self._refresh_q()
            lead = x.shape[:-1]
            x2 = x.reshape(-1, x.shape[-1])
            w_deq = self._backward_weight()
            if self.quant == "fp4":
                out = _FP4LinearSTE.apply(x2, self.master, self._q, self._s, w_deq)
            else:  # fp8 e4m3
                out = _FP8LinearSTE.apply(x2, self.master, self._q, self._s, w_deq)
            return out.view(*lead, -1) + self.bias if self.bias is not None else out.view(*lead, -1)
        # cached dequant: refreshed with weights in _refresh_q — same GEMM speed
        # as a plain bf16 linear (the old per-call dequant was several× slower)
        if self._q is None or self._s is None:
            self._refresh_q()
        if self._wd is None:
            self._wd = (self._q.float() * self._s).to(torch.bfloat16)
        if torch.is_grad_enabled():
            # Straight-through: forward uses the cached dequant bit-identically
            # while the gradient flows to master. Without this, any shape whose
            # fast-path self-test fails would silently freeze its weights
            # (_wd carries no autograd history). Skipped under no_grad so
            # inference pays no extra allocation.
            w = self._wd + (self.master - self.master.detach())
        else:
            w = self._wd
        return F.linear(x.to(w.dtype), w.to(x.dtype), self.bias)

    def _te_layer(self):
        """Return a lazy, non-stateful TE view over this QLinear's masters."""
        if _TE is None or self._te_recipe is None:
            raise RuntimeError(
                "QLinear requested te_nvfp4 but Transformer Engine NVFP4 was not enabled. "
                "Run nvfp4_te_preflight.py on the Blackwell host first."
            )
        layer = self._te_linear
        stale = (layer is None or layer.weight is not self.master or
                 layer.weight.device != self.master.device or
                 layer.bias is not self.bias)
        if stale:
            # Construct after WalkGPT.to(cuda), then replace TE's owned params
            # with our existing master parameters.  object.__setattr__ above
            # prevents this helper module from changing model.parameters() or
            # checkpoint key names.
            layer = _TE.Linear(
                self.master.shape[1], self.master.shape[0],
                bias=self.bias is not None, params_dtype=self.master.dtype,
                device=self.master.device,
            )
            layer.weight = self.master
            if self.bias is not None:
                layer.bias = self.bias
            object.__setattr__(self, "_te_linear", layer)
        layer.train(self.training)
        return layer

    def _forward_te_nvfp4(self, x):
        layer = self._te_layer()
        if _TE_NVFP4_SCOPE_DEPTH:
            return layer(x)
        # Direct QLinear probes are rare, but keeping them correct makes the
        # backend self-testable without depending on WalkGPT's root hook.
        with _TE.autocast(enabled=True, recipe=self._te_recipe):
            return layer(x)


# ------------------------------------------------------------ model pieces ---


def _chunked_causal_attn(q, k, v, hd, Tc, slopes=None):
    """Full-causal attention in O(Tc) memory chunks; same math as SDPA(is_causal=True).
    Meant for gradient checkpointing: saves only q/k/v, recomputes scores in
    backward — full attention at T=10240 would otherwise store ~3.4GB/layer of
    T^2 scores (the 92GB engine). slopes: optional per-head ALiBi penalties."""
    B, H, T, _ = q.shape
    outs = []
    for s in range(0, T, Tc):
        e = min(s + Tc, T)
        Tcc = e - s
        qc = q[:, :, s:e]
        kc = k[:, :, :e]
        vc = v[:, :, :e]
        a = torch.matmul(qc, kc.transpose(-1, -2)) / math.sqrt(hd)
        if slopes is not None:
            qi = torch.arange(s, e, device=q.device)
            kj = torch.arange(e, device=q.device)
            dist = (qi.view(Tcc, 1) - kj.view(1, e)).clamp(min=0, max=ALIBI_CAP).to(a.dtype)
            a = a - slopes.view(1, H, 1, 1) * dist.view(1, 1, Tcc, e)
        valid = torch.arange(e, device=q.device).view(1, e) <= torch.arange(s, e, device=q.device).view(Tcc, 1)
        a = a.masked_fill(~valid, float("-inf"))
        a = F.softmax(a, dim=-1)
        outs.append(torch.matmul(a, vc))
    return torch.cat(outs, dim=2)


def _global_attn(q, k, v, hd, Tc):
    """Global heads: force flash / mem-efficient SDPA explicitly (the default
    dispatch on sm_120 sometimes picks the math backend, materializing T^2
    scores). Falls back to the chunked implementation if neither runs."""
    try:
        from torch.nn.attention import sdpa_kernel, SDPBackend
        for backend in (SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION):
            try:
                with sdpa_kernel([backend]):
                    return F.scaled_dot_product_attention(q, k, v, is_causal=True)
            except Exception:
                continue
    except ImportError:
        pass
    return checkpoint(_chunked_causal_attn, q, k, v, hd, Tc, use_reentrant=False)


_WIN_MASKS = {}


def _win_masks(T, W, Tc, device):
    """Cached negated window masks per (T, W, Tc): identical for every layer and
    microbatch, so build once per shape instead of ~40x24x4x2 times per step.
    Null-key positions (before sequence start) are NOT masked here — they're
    unmasked in the forward pass when a learned null_key is present, so early
    positions can attend to a coherent "start" signal."""
    key = (T, W, Tc, device.index if device.type == "cuda" else -1)
    masks = _WIN_MASKS.get(key)
    if masks is None:
        masks = []
        for s in range(0, T, Tc):
            e = min(s + Tc, T)
            Tcc = e - s
            rows = torch.arange(Tcc, device=device).view(Tcc, 1)
            cols = torch.arange(Tcc + W, device=device).view(1, Tcc + W)
            masks.append(~((cols - rows >= 0) & (cols - rows <= W)))
        if len(_WIN_MASKS) > 64:
            _WIN_MASKS.pop(next(iter(_WIN_MASKS)))
        _WIN_MASKS[key] = masks
    return masks


class V41CausalEncoder(nn.Module):
    """Causal encoder summary used by the resumable CED approximation.

    The actual V4.1 CED projects final encoder states into decoder global KV.
    This decoder-only trainer has no separate encoder/cache lifecycle, so we
    build a causal prefix summary from the input embeddings.  The residual
    output is zero-gated, preserving the old model exactly until trained.
    """

    def __init__(self, D):
        super().__init__()
        self.in_proj = nn.Linear(D, D, bias=False, dtype=DTYPE)
        self.out_proj = nn.Linear(D, D, bias=False, dtype=DTYPE)
        self.gate = nn.Parameter(torch.zeros((), dtype=DTYPE))
        # Keep the branch's forward contribution at exactly zero through the
        # gate, but leave a small nonzero direction so the gate has a gradient
        # and can learn to open from a legacy checkpoint.
        nn.init.normal_(self.out_proj.weight, 0.0, 0.02)

    def forward(self, x):
        h = self.in_proj(x)
        denom = torch.arange(1, x.size(1) + 1, device=x.device, dtype=torch.float32).view(1, -1, 1)
        summary = (torch.cumsum(h.float(), dim=1) / denom).to(h.dtype)
        residual = self.out_proj(summary) * torch.tanh(self.gate)
        return residual, summary


class V41SharedKV(nn.Module):
    """Shared global K/V projections for CSA2-style cross-layer reuse."""

    def __init__(self, D):
        super().__init__()
        self.k_proj = nn.Linear(D, D, bias=False, dtype=DTYPE)
        self.v_proj = nn.Linear(D, D, bias=False, dtype=DTYPE)

    def forward(self, summary):
        return self.k_proj(summary), self.v_proj(summary)


class Attn(nn.Module):
    def __init__(self, D, n_head, hybrid_w, win_heads, alibi=False, rope=False):
        super().__init__()
        self.D, self.n_head = D, n_head
        self.hybrid_w, self.win_heads = hybrid_w, win_heads
        # Runtime-only tiling; no parameters or checkpoint keys. The historical
        # 256 tile is conservative, but a 512 tile was bit-identical and 1.40x
        # faster for window-attention forward+backward at T=2048 on an RTX 4050.
        # Keep the old default for resume safety; large-VRAM runs can opt in.
        self.window_chunk = max(32, min(2048, int(
            os.environ.get("TB_ATTN_WINDOW_CHUNK", "256")
        )))
        self.alibi = alibi
        self.rope = rope
        self.csa2_mode = os.environ.get("TB_V41_CSA2_MODE", "blend").strip().lower()
        if self.csa2_mode not in {"blend", "shared_only"}:
            raise ValueError("TB_V41_CSA2_MODE must be blend or shared_only")
        self.csa2_gate = nn.Parameter(torch.zeros((), dtype=DTYPE)) \
            if os.environ.get("TB_V41_CSA2", "0") == "1" else None
        # Persistent, non-trainable phase state.  Unlike an environment-only
        # alpha, this survives an ordinary checkpoint resume.  A structural
        # phase initializes it from TB_V41_CSA2_ALPHA and then the trainer may
        # ramp it deterministically over optimizer steps.
        self.register_buffer(
            "csa2_alpha",
            torch.tensor(float(os.environ.get("TB_V41_CSA2_ALPHA", "0.0")),
                         dtype=DTYPE).clamp_(0.0, 1.0),
            # Do not pollute ordinary Walker checkpoints with 16 inert CSA2
            # keys. Persistence is required only for an actual CSA2 phase.
            persistent=self.csa2_gate is not None,
        )
        if alibi and rope:
            raise ValueError("ALiBi and RoPE are mutually exclusive (both gated by TB_*)")
        self.qkv = QLinear(D, 3 * D)
        self.proj = QLinear(D, D)
        # Fixed RoPE base for windowed heads, sized to the MAX window (constant).
        # Using the current W would shift every angle each curriculum step (the
        # window grows 32→512), making the positional target a moving one and
        # destabilizing training. The max window is the init hybrid_w (512).
        self.rope_base_window = float(max(hybrid_w, 8))
        if alibi:
            # paper schedule: m_i = 2^(-8*i/H), i = 1..H — scaled down 8× for
            # D=2048/hd=128 stability (at hd=64 the raw schedule works; at hd=128
            # the same slopes dominate scores and drive gnorm to 400B+).
            self.register_buffer("slopes", torch.tensor(
                [2.0 ** (-8.0 * (i + 1) / n_head) / 8.0 for i in range(n_head)], dtype=DTYPE),
                persistent=False)
        hd = D // n_head
        Hw = min(win_heads, n_head - 1)
        if Hw > 0 and hybrid_w > 0:
            self.null_key = nn.Parameter(torch.zeros(1, Hw, 1, hd, dtype=DTYPE))
            nn.init.normal_(self.null_key, 0.0, 0.02)

    def _alibi_dist(self, Tcc, K, off, device):
        """(Tcc, K) positional distances clamped to >= 0."""
        qi = torch.arange(Tcc, device=device)
        kj = torch.arange(K, device=device)
        return (qi.view(Tcc, 1) - kj.view(1, K) + off).clamp(min=0, max=ALIBI_CAP)

    def forward(self, x, mask=None, position_ids=None, position_limit=None,
                shared_k=None, shared_v=None):
        B, T, D = x.shape
        q, k, v = self.qkv(x).split(self.D, dim=-1)
        hd = D // self.n_head
        q = q.view(B, T, self.n_head, hd).transpose(1, 2)
        k = k.view(B, T, self.n_head, hd).transpose(1, 2)
        v = v.view(B, T, self.n_head, hd).transpose(1, 2)
        if shared_k is not None or shared_v is not None:
            if shared_k is None or shared_v is None:
                raise ValueError("shared_k and shared_v must be supplied together")
            if shared_k.shape != (B, T, D) or shared_v.shape != (B, T, D):
                raise ValueError("shared CSA2 states must have shape [B, T, D]")
            sk = shared_k.view(B, T, self.n_head, hd).transpose(1, 2)
            sv = shared_v.view(B, T, self.n_head, hd).transpose(1, 2)
            k_global = k[:, :max(0, self.n_head - self.win_heads)]
            v_global = v[:, :max(0, self.n_head - self.win_heads)]
            sk_global = sk[:, :max(0, self.n_head - self.win_heads)]
            sv_global = sv[:, :max(0, self.n_head - self.win_heads)]
            if self.csa2_gate is None:
                raise RuntimeError("shared CSA2 states supplied while TB_V41_CSA2=0")
            if self.csa2_mode == "shared_only":
                # Paper-shaped low-cost phase: global heads consume the
                # encoder-derived shared K/V directly; window heads retain
                # their local K/V. This changes no parameter shapes and can be
                # entered as a gated phase transition from an old checkpoint.
                k = torch.cat((sk_global, k[:, max(0, self.n_head - self.win_heads):]), dim=1)
                v = torch.cat((sv_global, v[:, max(0, self.n_head - self.win_heads):]), dim=1)
            else:
                # The learnable gate preserves the original behavior at alpha=0.
                # The persistent alpha is an external, resumable bridge: it
                # opens a gradient path into fresh shared K/V before the final
                # shared-only phase, avoiding a destructive hard switch.
                learned_reuse = torch.tanh(self.csa2_gate).to(k.dtype)
                bridge_reuse = self.csa2_alpha.to(k.dtype)
                reuse = bridge_reuse + (1.0 - bridge_reuse) * learned_reuse
                k = torch.cat((k_global + reuse * (sk_global - k_global),
                               k[:, max(0, self.n_head - self.win_heads):]), dim=1)
                v = torch.cat((v_global + reuse * (sv_global - v_global),
                               v[:, max(0, self.n_head - self.win_heads):]), dim=1)
        # V4.1-inspired cache quantization. The trainer has no incremental KV
        # cache, so this is a straight-through FP4-like proxy applied per
        # head/token row. It changes only the forward cache values and adds no
        # parameters or checkpoint shape changes.
        if os.environ.get("TB_V41_KV_FP4", "0") == "1":
            def _cache_fp4(value):
                scale = value.detach().abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
                quant = (value / scale * 7.0).round().clamp(-7.0, 7.0) / 7.0 * scale
                return value + (quant - value).detach()
            k, v = _cache_fp4(k), _cache_fp4(v)
        W = self.hybrid_w
        # Keep the trained hybrid head split at every prefix length. During T=4096
        # training W=512<T, so the final win_heads always used window RoPE/null-key
        # attention. Disabling that split for T<=W makes short-prefix generation
        # execute a different model (all-global heads + different RoPE basis).
        Hw = min(self.win_heads, self.n_head - 1) if W > 0 else 0
        Hg = self.n_head - Hw
        # K-token merging shortens the prefix but can preserve the original
        # absolute RoPE phase for the suffix through ``position_ids``.
        # Normal training keeps the cheaper contiguous arange path.
        rope_positions = None
        rope_T = T if position_limit is None else int(position_limit)
        if position_ids is not None:
            rope_positions = position_ids.to(device=x.device, dtype=torch.long)
        if self.rope and Hw > 0:
            # Windowed heads get their own RoPE base sized to their MAX window
            # (constant). Global heads keep base=10000 for long-range reach.
            # A base tuned to the window gives maximal angular resolution where
            # these heads actually operate, instead of wasting low-frequency
            # bands on distinctions they can never use (they never attend past W).
            base_window = self.rope_base_window
            cos_g, sin_g = _rope_cos_sin(rope_T, hd, ROPE_BASE, x.device, x.dtype)
            cos_w, sin_w = _rope_cos_sin(rope_T, hd, base_window, x.device, x.dtype)
            if rope_positions is not None:
                cos_g, sin_g = cos_g[rope_positions], sin_g[rope_positions]
                cos_w, sin_w = cos_w[rope_positions], sin_w[rope_positions]
            cos_g = cos_g.unsqueeze(0).unsqueeze(0)
            sin_g = sin_g.unsqueeze(0).unsqueeze(0)
            cos_w = cos_w.unsqueeze(0).unsqueeze(0)
            sin_w = sin_w.unsqueeze(0).unsqueeze(0)
            q = torch.cat([_rope_rotate(q[:, :Hg], cos_g, sin_g), _rope_rotate(q[:, Hg:], cos_w, sin_w)], dim=1)
            k = torch.cat([_rope_rotate(k[:, :Hg], cos_g, sin_g), _rope_rotate(k[:, Hg:], cos_w, sin_w)], dim=1)
        elif self.rope:
            cos, sin = _rope_cos_sin(rope_T, hd, ROPE_BASE, x.device, x.dtype)
            if rope_positions is not None:
                cos, sin = cos[rope_positions], sin[rope_positions]
            # cos/sin: (T, hd/2) -> (1,1,T,hd/2) for B,H,T broadcast
            cos_b = cos.unsqueeze(0).unsqueeze(0)
            sin_b = sin.unsqueeze(0).unsqueeze(0)
            q = _rope_rotate(q, cos_b, sin_b)
            k = _rope_rotate(k, cos_b, sin_b)
        if W > 0 and Hw > 0:
            # probe-port: global heads full-causal SDPA; windowed heads use
            # as_strided O(T*W) scores — the REAL memory cut (no T^2 per banded head)
            # contiguous slices: flash/mem-efficient SDPA rejects non-contiguous
            # transposed views, and the exception-fallback chain is very slow
            if self.alibi:
                # flash SDPA can't take additive masks: route global heads through
                # the chunked causal matmul with the ALiBi bias baked in (and
                # checkpoint it: raw full-causal scores would retain GBs)
                out_g = checkpoint(_chunked_causal_attn, q[:, :Hg].contiguous(),
                                   k[:, :Hg].contiguous(), v[:, :Hg].contiguous(),
                                   hd, 512, self.slopes[:Hg], use_reentrant=False)
            else:
                out_g = _global_attn(q[:, :Hg].contiguous(), k[:, :Hg].contiguous(),
                                     v[:, :Hg].contiguous(), hd, 512)
            qw, kw, vw = q[:, Hg:], k[:, Hg:], v[:, Hg:]
            if hasattr(self, 'null_key'):
                # Learned null key for the zero-padded "virtual" keys before
                # position 0. Gives early-sequence positions a coherent
                # "start" signal instead of zero — synergizes with Walk whose
                # d1[0] is also structurally zero (by _sh construction), so
                # both attention and trajectory were blind at the boundary in
                # the same uninformative way. Now attention has a real learned
                # anchor to combine with Walk's early trajectory features.
                nk = self.null_key.expand(B, -1, W, -1)
                kp = torch.cat([nk, kw], dim=2)
            else:
                kp = torch.cat([kw.new_zeros(B, Hw, W, hd), kw], dim=2)
            vp = torch.cat([vw.new_zeros(B, Hw, W, hd), vw], dim=2)
            # chunked windowed attention: contiguous per-chunk window slices and
            # canonical 4D matmuls (q@k^T, att@v) — the standard attention pattern
            # that never materializes copies (5D as_strided/bmm variants forced
            # ~400MB contiguous copies per layer that autograd then retained)
            Tc = min(T, self.window_chunk)
            masks = _win_masks(T, W, Tc, x.device)
            outs = []
            for ci, s in enumerate(range(0, T, Tc)):
                e = min(s + Tc, T)
                Tcc = e - s
                qc = qw[:, :, s:e]
                kc = kp[:, :, s:e + W]
                vc = vp[:, :, s:e + W]
                a = torch.matmul(qc, kc.transpose(-1, -2)) / math.sqrt(hd)
                if self.alibi:
                    # key slot j -> absolute position s+j-W; query -> s+qi
                    # dist = qi - j + W (negative = future, clamped)
                    dist = self._alibi_dist(Tcc, e - s + W, W, a.device).to(a.dtype)
                    a = a - self.slopes[Hg:].view(1, Hw, 1, 1) * dist.view(1, 1, Tcc, e - s + W)
                a = a.masked_fill(masks[ci], float("-inf"))
                if not hasattr(self, 'null_key') and s < W:
                    # No learned null key: mask the zero-padded positions (before
                    # sequence start) so they don't dilute the attention. When a
                    # null_key IS present, these positions carry a learned
                    # "start" signal and are left unmasked.
                    a[:, :, :, :W - s] = float("-inf")
                a = F.softmax(a, dim=-1)
                outs.append(torch.matmul(a, vc))
            out_w = torch.cat(outs, dim=2)
            out = torch.cat([out_g, out_w], dim=1)
        else:
            if self.alibi:
                out = _chunked_causal_attn(q, k, v, hd, 512, slopes=self.slopes)
            else:
                out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.proj(out)


class HierarchicalAttn(nn.Module):
    """Parameter-matched causal attention with exact local detail and coarse history.

    The qkv/proj maps are intentionally identical to ``Attn``.  The only
    approximation is over sequence interactions: a query attends to every
    causally-valid token in its own chunk and to one mean K/V summary for each
    *completed* earlier chunk.  No learned summarizer is used in this first
    ablation, so it adds no parameters and cannot win merely by capacity.

    For chunk size C, the attention-score work is O(T*C + T*T/C), rather than
    O(T^2).  Crucially, summaries never include the query's current chunk, so
    a suffix cannot influence an earlier prefix through a pooled summary.
    """

    def __init__(self, D, n_head, chunk_size=32, rope=False):
        super().__init__()
        if D <= 0 or n_head <= 0 or D % n_head:
            raise ValueError("hierarchical attention needs D divisible by n_head")
        if chunk_size <= 0:
            raise ValueError("hierarchical attention chunk_size must be positive")
        self.D, self.n_head = int(D), int(n_head)
        self.chunk_size, self.rope = int(chunk_size), bool(rope)
        # Same four D^2 projections as ordinary attention; the summaries below
        # are deterministic reductions of K/V, not extra learned capacity.
        self.qkv = QLinear(D, 3 * D)
        self.proj = QLinear(D, D)

    def candidate_count(self, T):
        """Maximum K/V candidates a query sees at length T (for diagnostics)."""
        return min(T, self.chunk_size + (T + self.chunk_size - 1) // self.chunk_size)

    def forward(self, x, mask=None):
        del mask  # The decoder path is causal by construction; no padding mask today.
        B, T, D = x.shape
        q, k, v = self.qkv(x).split(self.D, dim=-1)
        hd = D // self.n_head
        q = q.view(B, T, self.n_head, hd).transpose(1, 2)
        k = k.view(B, T, self.n_head, hd).transpose(1, 2)
        v = v.view(B, T, self.n_head, hd).transpose(1, 2)
        if self.rope:
            cos, sin = _rope_cos_sin(T, hd, ROPE_BASE, x.device, x.dtype)
            cos, sin = cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0)
            q, k = _rope_rotate(q, cos, sin), _rope_rotate(k, cos, sin)

        C = min(self.chunk_size, T)
        chunks = (T + C - 1) // C
        # Complete-chunk K/V summaries.  Padding is excluded through counts, so
        # the final short chunk remains valid although it is never exposed to
        # itself or any earlier query.
        pad = chunks * C - T
        if pad:
            k_pad = F.pad(k, (0, 0, 0, pad))
            v_pad = F.pad(v, (0, 0, 0, pad))
        else:
            k_pad, v_pad = k, v
        valid = torch.full((chunks,), C, device=x.device, dtype=k.dtype)
        valid[-1] = C - pad
        chunk_k = k_pad.view(B, self.n_head, chunks, C, hd).sum(dim=3)
        chunk_v = v_pad.view(B, self.n_head, chunks, C, hd).sum(dim=3)
        chunk_k = chunk_k / valid.view(1, 1, chunks, 1)
        chunk_v = chunk_v / valid.view(1, 1, chunks, 1)

        outputs = []
        for chunk_index, start in enumerate(range(0, T, C)):
            end = min(start + C, T)
            length = end - start
            local_q = q[:, :, start:end]
            # Each previous chunk is already complete and causal.  The local
            # segment is exact but triangularly masked to retain token causality.
            keys = torch.cat((chunk_k[:, :, :chunk_index], k[:, :, start:end]), dim=2)
            values = torch.cat((chunk_v[:, :, :chunk_index], v[:, :, start:end]), dim=2)
            scores = torch.matmul(local_q, keys.transpose(-1, -2)) / math.sqrt(hd)
            local_future = torch.triu(
                torch.ones(length, length, dtype=torch.bool, device=x.device), diagonal=1)
            scores[..., chunk_index:] = scores[..., chunk_index:].masked_fill(local_future, float("-inf"))
            weights = F.softmax(scores, dim=-1)
            outputs.append(torch.matmul(weights, values))
        out = torch.cat(outputs, dim=2)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.proj(out)


class ArchiveRetrievalAttn(nn.Module):
    """Causal two-tier attention: cheap chunk index, exact K/V archive retrieval.

    Every completed chunk remains in the archive as its original token-level
    K/V vectors.  A query first scores only one mean-K landmark per past chunk,
    retrieves top-k chunks, then attends to their exact K/Vs and its exact
    causal local chunk.  This is the sequence-aware form of the archive idea:
    the archive holds trusted detail; landmarks only decide which detail to
    open.  It has the same QKV/proj parameters as ordinary attention.
    """

    def __init__(self, D, n_head, chunk_size=32, retrieve_chunks=2, rope=False, straight_through=True):
        super().__init__()
        if D <= 0 or n_head <= 0 or D % n_head:
            raise ValueError("archive retrieval attention needs D divisible by n_head")
        if chunk_size <= 0 or retrieve_chunks <= 0:
            raise ValueError("archive retrieval chunk size and top-k must be positive")
        self.D, self.n_head = int(D), int(n_head)
        self.chunk_size, self.retrieve_chunks, self.rope = int(chunk_size), int(retrieve_chunks), bool(rope)
        self.straight_through = bool(straight_through)
        self.qkv = QLinear(D, 3 * D)
        self.proj = QLinear(D, D)
        self.capture_routes = False
        self._last_selected_chunks = None

    def candidate_count(self, T):
        chunks = (T + self.chunk_size - 1) // self.chunk_size
        return min(T, self.chunk_size * (1 + min(self.retrieve_chunks, max(0, chunks - 1))))

    def forward(self, x, mask=None):
        del mask
        B, T, D = x.shape
        q, k, v = self.qkv(x).split(self.D, dim=-1)
        hd = D // self.n_head
        q = q.view(B, T, self.n_head, hd).transpose(1, 2)
        k = k.view(B, T, self.n_head, hd).transpose(1, 2)
        v = v.view(B, T, self.n_head, hd).transpose(1, 2)
        if self.rope:
            cos, sin = _rope_cos_sin(T, hd, ROPE_BASE, x.device, x.dtype)
            cos, sin = cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0)
            q, k = _rope_rotate(q, cos, sin), _rope_rotate(k, cos, sin)

        C = min(self.chunk_size, T)
        chunks = (T + C - 1) // C
        pad = chunks * C - T
        k_pad = F.pad(k, (0, 0, 0, pad)) if pad else k
        v_pad = F.pad(v, (0, 0, 0, pad)) if pad else v
        archive_k = k_pad.view(B, self.n_head, chunks, C, hd)
        archive_v = v_pad.view(B, self.n_head, chunks, C, hd)
        valid = torch.full((chunks,), C, device=x.device, dtype=k.dtype)
        valid[-1] = C - pad
        landmarks = archive_k.sum(dim=3) / valid.view(1, 1, chunks, 1)

        outputs = []
        routes = [] if self.capture_routes else None
        for chunk_index, start in enumerate(range(0, T, C)):
            end = min(start + C, T)
            length = end - start
            local_q = q[:, :, start:end]
            local_k, local_v = k[:, :, start:end], v[:, :, start:end]
            local_scores = torch.matmul(local_q, local_k.transpose(-1, -2)) / math.sqrt(hd)
            future = torch.triu(torch.ones(length, length, dtype=torch.bool, device=x.device), diagonal=1)
            local_scores = local_scores.masked_fill(future, float("-inf"))
            if chunk_index == 0:
                if routes is not None:
                    routes.append(None)
                local_weights = F.softmax(local_scores, dim=-1)
                outputs.append(torch.matmul(local_weights, local_v))
                continue

            topk = min(self.retrieve_chunks, chunk_index)
            index_scores = torch.matmul(local_q, landmarks[:, :, :chunk_index].transpose(-1, -2)) / math.sqrt(hd)
            # Hard retrieval has no gradient through the top-k *choice*, which
            # makes a random early index hard to repair.  Use a dense landmark
            # read only as a straight-through backward surrogate: its value is
            # numerically cancelled in the forward pass, so serving remains
            # exact-K/V top-k plus local attention while every landmark receives
            # an index-learning signal during training.
            soft_index = F.softmax(index_scores.float(), dim=-1).to(local_q.dtype)
            landmark_v = archive_v[:, :, :chunk_index].sum(dim=3) / valid[:chunk_index].view(1, 1, chunk_index, 1)
            soft_out = torch.matmul(soft_index, landmark_v)
            indices = torch.topk(index_scores, topk, dim=-1).indices
            if routes is not None:
                routes.append(indices.detach())
            # Gather detailed K/V for each query's selected archive chunks. The
            # expanded source is a view; only K exact chunks are materialized.
            source_k = archive_k[:, :, :chunk_index].unsqueeze(2).expand(-1, -1, length, -1, -1, -1)
            source_v = archive_v[:, :, :chunk_index].unsqueeze(2).expand(-1, -1, length, -1, -1, -1)
            gather = indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, -1, C, hd)
            detailed_k = source_k.gather(3, gather).flatten(3, 4)
            detailed_v = source_v.gather(3, gather).flatten(3, 4)
            detailed_scores = (local_q.unsqueeze(-2) * detailed_k).sum(dim=-1) / math.sqrt(hd)
            scores = torch.cat((detailed_scores, local_scores), dim=-1)
            weights = F.softmax(scores, dim=-1)
            split = detailed_scores.size(-1)
            archive_out = (weights[..., :split].unsqueeze(-1) * detailed_v).sum(dim=-2)
            local_out = torch.matmul(weights[..., split:], local_v)
            hard_out = archive_out + local_out
            if self.training and self.straight_through:
                hard_out = hard_out + soft_out - soft_out.detach()
            outputs.append(hard_out)
        out = torch.cat(outputs, dim=2)
        self._last_selected_chunks = routes
        return self.proj(out.transpose(1, 2).contiguous().view(B, T, D))


class MLP(nn.Module):
    def __init__(self, D, mult):
        super().__init__()
        h = int(mult * D)
        self.up = QLinear(D, h)
        self.down = QLinear(h, D)

    def forward(self, x):
        return self.down(F.gelu(self.up(x)))


class MLPCompact(nn.Module):
    """Compact fused op found on Qwen: SiLU(W_gate) * W_up -> W_down
    Independent gate/up (pearson ~0) forms low-rank Hadamard mul that needs 2 projs.
    Replaces GELU with SiLU-gated product, same h unless TB_MLP_COMPACT_EQUAL=1 where h'=2/3h."""
    def __init__(self, D, mult):
        super().__init__()
        h = int(mult * D)
        if os.environ.get("TB_MLP_COMPACT_EQUAL", "0") == "1":
            h = max(8, int(h * 2 / 3))
        self.gate = QLinear(D, h)
        self.up = QLinear(D, h)
        self.down = QLinear(h, D)
        self._h = h

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


def _local_conv_kernels():
    raw = os.environ.get("TB_LOCAL_CONV_KERNELS", "3,7,15")
    values = tuple(int(v.strip()) for v in raw.split(",") if v.strip())
    if not values or any(v < 1 or v % 2 == 0 or v > 63 for v in values):
        raise ValueError("TB_LOCAL_CONV_KERNELS must be odd positive integers <= 63")
    return values


def _local_conv_stage_kernels(stage):
    """Return kernels for an additive deeper local-convolution stage.

    Stage one preserves the original ``TB_LOCAL_CONV_KERNELS`` contract.  The
    defaults for later stages widen the causal receptive field while keeping
    every branch depthwise and cheap.  Existing one-stage checkpoints therefore
    keep their exact parameter names and values when depth three is enabled.
    """
    stage = int(stage)
    if stage <= 1:
        return _local_conv_kernels()
    defaults = {2: "7,15,31", 3: "15,31,63"}
    raw = os.environ.get(f"TB_LOCAL_CONV_STAGE{stage}_KERNELS", defaults.get(stage, "15,31,63"))
    values = tuple(int(v.strip()) for v in raw.split(",") if v.strip())
    if not values or any(v < 1 or v % 2 == 0 or v > 63 for v in values):
        raise ValueError(f"TB_LOCAL_CONV_STAGE{stage}_KERNELS must be odd positive integers <= 63")
    return values


def _local_conv_gate_init():
    """Return the requested effective residual gate at initialization.

    The parameter stores an unconstrained logit and the forward path applies
    ``tanh``.  Expressing the setting in effective-gate units makes a small
    bootstrap such as 0.02 easy to reason about and keeps the default exactly
    compatible with the historical zero-gated extension.
    """
    raw = float(os.environ.get("TB_LOCAL_CONV_GATE_INIT", "0.0"))
    if not math.isfinite(raw) or abs(raw) >= 0.95:
        raise ValueError("TB_LOCAL_CONV_GATE_INIT must be finite with abs(value) < 0.95")
    return raw


def _local_conv_gate_bootstrap():
    """Temporary additive residual used to test an otherwise inactive CNN.

    This is deliberately separate from the learned gate: a small fixed term
    lets convolution weights receive CE gradients even if an inherited gate
    has collapsed. The learned ``tanh(residual_gate)`` remains trainable and
    the bootstrap can be removed cleanly on the next resume.
    """
    raw = float(os.environ.get("TB_LOCAL_CONV_GATE_BOOTSTRAP", "0.0"))
    if not math.isfinite(raw) or abs(raw) >= 0.25:
        raise ValueError("TB_LOCAL_CONV_GATE_BOOTSTRAP must be finite with abs(value) < 0.25")
    return raw


class CausalLocalFeatureConv(nn.Module):
    """Multi-scale causal local features for the RoPE+Walk block.

    Depthwise temporal filters detect short motifs without a D-by-D temporal
    convolution. A low-rank pointwise mixer turns those motifs into model
    channels. The scalar gate is zero-initialized, making this an exact no-op
    when added to an older checkpoint; its derivative is nonzero, so CE can
    learn whether local evidence is useful.

    Tiny probe record (2026-09-07, D=128/L=2, two seeds, local recurrence):
    HIER CE 4.3164/4.2766 (T=128/256) -> HIER+CNN 4.1578/4.1281. This is a
    diagnostic result, not a claim about the 1B run.

    Production-equivalent natural-text smoke (2026-09-08, FineWeb tokens,
    D=128/L=2, 120 updates): fixed bootstrap=-.05 made learned gates move
    toward zero (effective gates remained about -.05); bootstrap=0 with
    gate_init=.02 moved gates positive (.032-.042).  Held-out CE was equal
    to 5 decimals at this horizon.  Thus `local_g` alone must never be read
    as accept/reject while a nonzero bootstrap is active; inspect effective
    gates and depthwise-filter drift, then run a longer matched A/B.

    Follow-up (same natural tokens, 600 updates, 128 validation batches):
    control CE=8.42236; bootstrap=-.05 CE=8.41821 with effective stage gates
    -.079/-.123/-.141; clean gate_init=.02 CE=8.41943 with stage gates
    +.064/+.123/+.102. Both signs gave only a small ~.003-.004-nat gain.
    This invalidates treating gate sign as acceptance: the adjacent learned
    projections can absorb a sign reversal. Magnitude, filter drift, and a
    matched CE control matter; the old synthetic gate magnitude is not a
    justified default for a resumed 1B model.
    """
    def __init__(self, D, rank=None, kernels=None):
        super().__init__()
        self.D = int(D)
        self.rank = max(1, min(self.D, int(rank if rank is not None else
                                           os.environ.get("TB_LOCAL_CONV_RANK", "128"))))
        self.kernels = tuple(kernels or _local_conv_kernels())
        self.norm = nn.LayerNorm(self.D, dtype=DTYPE)
        self.depthwise = nn.ModuleList([
            nn.Conv1d(self.D, self.D, k, groups=self.D, bias=False,
                      dtype=DTYPE)
            for k in self.kernels
        ])
        # Tiny routing/control parameters must remain FP32. At the 1B resume LR
        # their Adam update is below one BF16 ULP, which froze all 48 CNN gates.
        self.scale_logits = nn.Parameter(torch.zeros(len(self.kernels), dtype=torch.float32))
        self.in_proj = nn.Linear(self.D, self.rank, bias=False, dtype=DTYPE)
        self.out_proj = nn.Linear(self.rank, self.D, bias=False, dtype=DTYPE)
        gate_init = _local_conv_gate_init()
        self.gate_bootstrap = _local_conv_gate_bootstrap()
        # Store the inverse-tanh logit so the effective residual starts at the
        # user-requested value.  The default remains an exact no-op.
        gate_logit = math.atanh(gate_init) if gate_init else 0.0
        self.residual_gate = nn.Parameter(torch.tensor(gate_logit, dtype=torch.float32))
        with torch.no_grad():
            for conv, k in zip(self.depthwise, self.kernels):
                nn.init.zeros_(conv.weight)
                conv.weight[:, 0, k - 1] = 1.0
            nn.init.normal_(self.in_proj.weight, 0.0, 0.02)
            nn.init.normal_(self.out_proj.weight, 0.0, 0.02)

    def forward(self, x):
        B, T, D = x.shape
        if D != self.D:
            raise ValueError("CausalLocalFeatureConv width mismatch")
        z = self.norm(x)
        zc = z.transpose(1, 2)
        weights = torch.softmax(self.scale_logits.float(), dim=0).to(z.dtype)
        if (len(self.depthwise) > 1
                and os.environ.get("TB_LOCAL_CONV_FUSE_KERNELS", "1") == "1"):
            local = self._fused_depthwise(zc, weights)
        else:
            local = z.new_zeros(B, D, T)
            for weight, conv, k in zip(weights, self.depthwise, self.kernels):
                # Left-only padding preserves causality and keeps output length T.
                local = local + weight * conv(F.pad(zc, (k - 1, 0)))
        local = local.transpose(1, 2)
        local = self.out_proj(F.silu(self.in_proj(local)))
        gate = torch.tanh(self.residual_gate).to(local.dtype) + self.gate_bootstrap
        # The configured bootstrap is bounded below 0.25, but clamp defensively
        # if a learned gate later approaches its extrema.
        return gate.clamp(-0.95, 0.95) * local

    def _fused_depthwise(self, zc, weights):
        """Combine parallel linear depthwise branches before one convolution.

        Sum_i a_i Conv(x, w_i) == Conv(x, Sum_i a_i PadLeft(w_i)).
        Padding aligns every branch's current-token coefficient at the right
        edge. Parameters and their gradients remain separate; only execution
        is fused. For the configured three stages this replaces nine depthwise
        convolution launches per block with three.
        """
        max_kernel = max(self.kernels)
        fused = None
        for weight, conv, kernel in zip(weights, self.depthwise, self.kernels):
            aligned = F.pad(conv.weight, (max_kernel - kernel, 0))
            term = weight * aligned
            fused = term if fused is None else fused + term
        return F.conv1d(F.pad(zc, (max_kernel - 1, 0)), fused, groups=self.D)


# ======================= Walk primitive extensions (registry) ===============
# The trajectory pathway (WalkAdapter) has a family of default-OFF sub-modules.
# Every one is checkpoint-safe: OFF => no new params (the checkpoint loads
# byte-identical); ON => the added gain is zero-init, so step 0 is a bitwise
# no-op and only the new params learn. Structural keys carry one-way
# introduction tolerance (see validate_architecture_manifest).
#
#  flag                      reads / binds                              per-layer
#  TB_WALK_ADAPT_SCALE       per-token softmax over a lag bank           O(hd*r)
#                            (dynamic-conv) — LOST the local A/B; shelved
#  TB_WALK_CONV_BASELINE     matched STATIC lag-bank kernel              O(hd*L)
#                            (falsification arm for the above)
#  TB_WALK_SELECTIVE_MEM     input-dependent EMA decay (needs GLOB)      O(hd)
#  TB_WALK_GEOM              arc-length + Frenet-Serret curve geometry   O(hd)
#                            (unit tangent / curvature / torsion / dwell), no_grad
#  TB_WALK_PLACE             self-localization / revisit distance        O(hd*K)
#                            (SLAM loop-closure analogue), no_grad
#  TB_WALK_HURST             local Hurst exponent (gait: persistent vs    3*n_heads
#                            wandering), no_grad
#  TB_WALK_AFFINE            affine step B ~ x*A (scale) + d (orthogonal  2*n_heads+n_heads*hd
#                            novelty, x=1 == the plain difference), no_grad
#  TB_WALK_MOTIF             dictionary of motion "gestures" -> concept   O(hd*r+M*r+M*hd)
#                            vectors, softmax recall (zero-init gain)
#  TB_WALK_MOTIF_PRED        optional weight: recalled concept must MSE-
#                            predict the NEXT motion step (regression)
#
# Geometry-family modules (geom/place/hurst/affine) compute features under
# torch.no_grad: they are MEASUREMENTS, only their gains learn. This is also the
# NaN fix — raw 1/||v|| and sqrt'(0) poison backbone gradients at the
# zero-velocity first token otherwise.
#
# Local from-scratch A/B (wikitext-2, D=256 L=6, 800 steps, param-matched,
# seed 0 => deterministic): nowalk 6.760 > walk 6.732; and among the additions
# motif 6.685 < geom 6.709 < affine 6.719 < walk 6.732. So the base trajectory
# pathway helps (~6% ppl) and MOTIF (gesture->concept memory) is the strongest
# addition; the geometry variants are marginal. These local gaps are small —
# confirm across seeds / LR / size with Walker_Scale_MoLab.py on a big GPU.
#
# NOT the bottleneck: the 1B is a 15B-token BASE model (raw continuation, no
# instruction tuning, ~1000x fewer tokens than SmolLM2/OLMo). Its precision
# ceiling is DATA + OBJECTIVE, not the walk. Next steps: (1) score EXACT-MATCH
# accuracy, not weighted CE — templated sft_math/sft_science_vendor sources have
# CE ~0.9 and mask the true capability; (2) SFT on answer-formatted data so it
# answers instead of continuing; (3) route arithmetic to a tool. The walk family
# is orthogonal and incremental.
# ===========================================================================
class WalkAdapter(nn.Module):
    """walk-mh: multihead velocity adapter with fp32 differencing.
    - LN input; telescopes d1/d2/d4, turn, per-head curvature kappa
    - n_heads subspaces: per-head per-feature gains (n_heads, hd) and a learned
      cross-head mixing matrix (zero-init + identity = exact no-op; lets heads
      exchange signals once useful)
    - fp32 differential features: bf16 subtracts nearby values with catastrophic
      cancellation (|delta|/|x| gets small over long-horizon traces) — fp32
      preserves the fine trajectory structure at D=2048+
    - keeps coarse concept-trajectory (S=8), curvature gate, walk_norm
    Zero-init exact no-op at start (w* = 0, head_cross = 0, gamma = 0).
    3x LR group, ~ (7*n_heads*hd + n_heads^2 + 1) params per layer."""

    def __init__(self, D, coarse_s=8, n_heads=None, fp32_diffs=None):
        super().__init__()
        self.walk_norm = GPC_WALK_NORM
        self.walk_norm_mode = GPC_WALK_NORM_MODE
        self.n_heads = max(1, int(os.environ.get("TB_WALK_HEADS",
                                                 n_heads if n_heads is not None else 1)))
        self.hd = D // self.n_heads
        self.fp32_diffs = bool(int(os.environ.get("TB_WALK_FP32",
                                                  fp32_diffs if fp32_diffs is not None else 1)))
        # full D->D cross-head mixing projections (zero-init = exact no-op).
        # Unlike the scalar head_cross (shared across all hd dims), this lets
        # each channel of each head mix with every other channel — real capacity.
        # TB_WALK_MIX_DEPTH>1 stacks k D->D projections (each zero-init residual)
        # for the "reduce-attention → increase-walk" param-neutral swap: removing
        # attention from every 4th layer frees exactly the budget for +1 mix/layer.
        self.mix_on = int(os.environ.get("TB_WALK_MIX", "0")) == 1
        self.mix_depth = max(1, int(os.environ.get("TB_WALK_MIX_DEPTH", "1")))
        self.mix_gain = float(os.environ.get("TB_WALK_MIX_GAIN", "1.0"))
        # global-dynamics features (running sums + long-stride telescopes):
        # attention cannot compute per-token path-level reductions, so these stay
        # orthogonal to attention at any scale. gated by TB_WALK_GLOB.
        self.glob_on = int(os.environ.get("TB_WALK_GLOB", "0")) == 1
        # Selective memory (TB_WALK_SELECTIVE_MEM): the walk's exponential
        # memory has a FIXED decay (alpha=0.05) — a non-selective filter.
        # Modern state-space models (Mamba/S6, RWKV, gated linear attention)
        # make the memory decay INPUT-DEPENDENT: each token decides how much
        # history to keep. This adds a learned per-token gate that re-mixes the
        # fixed trace toward the current state (a cheap, causal approximation of
        # a selective scan). The gate gain is zero-init, so the trace — and the
        # whole forward — is unchanged at construction (bitwise no-op). Requires
        # glob (the memory feature it modulates).
        self.selmem_on = (self.glob_on
                          and int(os.environ.get("TB_WALK_SELECTIVE_MEM", "0")) == 1)
        if self.selmem_on:
            self.mem_sel_w = nn.Parameter(torch.zeros(
                self.n_heads, self.hd, dtype=DTYPE))
            self.mem_sel_g = nn.Parameter(torch.zeros((), dtype=DTYPE))
            print("  walk selective-memory: input-dependent EMA gate "
                  "(zero-init identity)", flush=True)
        else:
            self.register_parameter("mem_sel_w", None)
            self.register_parameter("mem_sel_g", None)
        # Differential-geometry walk (TB_WALK_GEOM): measure the SHAPE of the
        # hidden trajectory — intrinsic (arc-length) time and the curve
        # invariants (unit tangent, curvature, out-of-plane torsion) — instead
        # of raw token-index differences. Token-rate- and scale-invariant.
        # Zero-init gains -> bitwise no-op at construction.
        self.geom_on = int(os.environ.get("TB_WALK_GEOM", "0")) == 1
        if self.geom_on:
            self.g_tan = nn.Parameter(torch.zeros(self.n_heads, self.hd, dtype=DTYPE))
            self.g_curv = nn.Parameter(torch.zeros(self.n_heads, dtype=DTYPE))
            self.g_tors = nn.Parameter(torch.zeros(self.n_heads, dtype=DTYPE))
            self.g_dwell = nn.Parameter(torch.zeros(self.n_heads, dtype=DTYPE))
            self.g_arc = nn.Parameter(torch.zeros(self.n_heads, dtype=DTYPE))
            print("  walk geometry: arc-length + Frenet-Serret invariants "
                  "(zero-init identity)", flush=True)
        else:
            for _nm in ("g_tan", "g_curv", "g_tors", "g_dwell", "g_arc"):
                self.register_parameter(_nm, None)
        # Self-localization (TB_WALK_PLACE): the walk keeps a coarse map of where
        # it has been (multi-timescale landmark EMAs) and measures distance-to-
        # place and whether it is returning — a causal analogue of SLAM
        # loop-closure. Zero-init gains -> no-op at construction.
        self.place_on = int(os.environ.get("TB_WALK_PLACE", "0")) == 1
        if self.place_on:
            self.place_alphas = (0.05, 0.2, 0.5)
            self.g_pdist = nn.Parameter(torch.zeros(
                len(self.place_alphas), self.n_heads, dtype=DTYPE))
            self.g_prad = nn.Parameter(torch.zeros(
                len(self.place_alphas), self.n_heads, dtype=DTYPE))
            print("  walk self-localization: multi-timescale place features "
                  "(zero-init identity)", flush=True)
        else:
            self.register_parameter("g_pdist", None)
            self.register_parameter("g_prad", None)
        # Persistence / gait (TB_WALK_HURST): the local Hurst exponent of the
        # representation trajectory at three scale pairs (1-2, 2-4, 4-8).
        # H ~ 0.5 = Brownian wandering; H > 0.5 = persistent (trending, has
        # momentum); H < 0.5 = anti-persistent (mean-reverting, oscillating).
        # The model conditions on the character of its own walk — its gait.
        # Zero-init gains -> bitwise no-op at construction.
        self.hurst_on = int(os.environ.get("TB_WALK_HURST", "0")) == 1
        if self.hurst_on:
            self.hurst_pairs = ((1, 2), (2, 4), (4, 8))
            self.g_hurst = nn.Parameter(torch.zeros(
                len(self.hurst_pairs), self.n_heads, dtype=DTYPE))
            print("  walk gait (Hurst persistence): 3 scale pairs "
                  "(zero-init identity)", flush=True)
        else:
            self.register_parameter("g_hurst", None)
        # Affine step (TB_WALK_AFFINE): decompose each step from the previous
        # state A to the current state B into a SCALED REPETITION of A
        # (least-squares scale x = (A.B)/(A.A), so B ~ x*A) plus an ORTHOGONAL
        # NOVEL direction d = B - x*A (A.d = 0). Generalizes the additive
        # difference (the x=1 special case) to a scale+shift motion and directly
        # separates self-similar repetition from new information. Zero-init
        # gains -> bitwise no-op at construction.
        self.affine_on = int(os.environ.get("TB_WALK_AFFINE", "0")) == 1
        if self.affine_on:
            self.g_aff_scale = nn.Parameter(torch.zeros(self.n_heads, dtype=DTYPE))
            self.g_aff_novel = nn.Parameter(torch.zeros(self.n_heads, dtype=DTYPE))
            self.g_aff_dir = nn.Parameter(torch.zeros(self.n_heads, self.hd, dtype=DTYPE))
            print("  walk affine step: scale + orthogonal novelty "
                  "(zero-init identity)", flush=True)
        else:
            for _nm in ("g_aff_scale", "g_aff_novel", "g_aff_dir"):
                self.register_parameter(_nm, None)
        # Motion-motif memory (TB_WALK_MOTIF): a learned dictionary that binds
        # trajectory SHAPES to CONCEPTS. The current motion (d1) is matched
        # against a codebook of learned "gestures"; the concept vector bound to
        # the best-matching gestures is recalled and added. This lets the walk
        # remember which kinds of trends go with which kinds of information.
        # The output gain is zero-init -> bitwise no-op at construction.
        self.motif_on = int(os.environ.get("TB_WALK_MOTIF", "0")) == 1
        # Motif v2 (TB_WALK_MOTIF_V2, needs TB_WALK_MOTIF): v1 matches on d1
        # alone with raw dot products of std-0.02 params, so its softmax is
        # ~uniform at init (recall ~ the mean concept, a constant bias; keys
        # get almost no gradient) and per-token |d1| sets the temperature.
        # v2 queries the RMS-normalized multi-scale shape [d1, d2, d4, turn],
        # scores by cosine with a learned per-head temperature (init 10), so
        # gestures are scale-free and the dictionary is selective from step 0.
        self.motif_v2 = (self.motif_on
                         and int(os.environ.get("TB_WALK_MOTIF_V2", "0")) == 1)
        # v2 refinement (TB_WALK_MOTIF_ORTHO): the telescoped scales d1/d2/d4 are
        # near-collinear, so orthogonalise the shape blocks against each other
        # (Gram-Schmidt) — the projection then reads genuinely independent
        # multi-scale motion instead of copies of d1.
        self.motif_ortho = (self.motif_v2
                            and int(os.environ.get("TB_WALK_MOTIF_ORTHO", "0")) == 1)
        # v2 extensions (all opt-in; defaults reproduce plain v2 bitwise):
        #  TB_WALK_MOTIF_SHARED: one gesture dictionary (keys, concepts, mod
        #    gains) for every block, owned by WalkGPT.walk_motif_bank; each
        #    block keeps its own query projection, temperature and gains. 16x
        #    more data per gesture, ~16x fewer dictionary params.
        #  TB_WALK_MOTIF_CONF: per-token confidence gate on the recall,
        #    sigmoid(w * (1 - H(att)/log M) + b); w=0, b=4 at init (~0.98).
        #  TB_WALK_MOTIF_TOPK=k: hard retrieval over the k best gestures.
        #  TB_WALK_MOTIF_MOD: content-modulated recall: the matched gestures
        #    also select a per-channel gain applied to the CURRENT state,
        #    out += g_mod * (att @ mod) * x, so the recall depends on content,
        #    not only on motion shape (zero-init g_mod: exact no-op at start).
        self.motif_shared = (self.motif_v2
                             and int(os.environ.get("TB_WALK_MOTIF_SHARED", "0")) == 1)
        self.motif_conf = (self.motif_v2
                           and int(os.environ.get("TB_WALK_MOTIF_CONF", "0")) == 1)
        self.motif_topk = (max(0, int(os.environ.get("TB_WALK_MOTIF_TOPK", "0")))
                           if self.motif_v2 else 0)
        self.motif_mod = (self.motif_v2
                          and int(os.environ.get("TB_WALK_MOTIF_MOD", "0")) == 1)
        object.__setattr__(self, "_motif_bank", None)
        if self.motif_on:
            self.motif_count = max(2, int(os.environ.get("TB_WALK_MOTIF_COUNT", "64")))
            self.motif_rank = max(1, int(os.environ.get("TB_WALK_MOTIF_RANK", "8")))
            self.motif_q = nn.Parameter(torch.empty(
                self.n_heads, (4 if self.motif_v2 else 1) * self.hd,
                self.motif_rank, dtype=DTYPE))
            if self.motif_shared:
                self.register_parameter("motif_key", None)
                self.register_parameter("motif_concept", None)
            else:
                self.motif_key = nn.Parameter(torch.empty(
                    self.n_heads, self.motif_count, self.motif_rank, dtype=DTYPE))
                self.motif_concept = nn.Parameter(torch.empty(
                    self.n_heads, self.motif_count, self.hd, dtype=DTYPE))
            if self.motif_v2:
                nn.init.normal_(self.motif_q, std=(4 * self.hd) ** -0.5)
                if not self.motif_shared:
                    nn.init.normal_(self.motif_key, std=1.0)
                self.motif_logit_scale = nn.Parameter(torch.full(
                    (self.n_heads,), math.log(10.0), dtype=torch.float32))
            else:
                nn.init.normal_(self.motif_q, std=0.02)
                nn.init.normal_(self.motif_key, std=0.02)
                self.register_parameter("motif_logit_scale", None)
            if not self.motif_shared:
                nn.init.normal_(self.motif_concept, std=0.02)
            self.g_motif = nn.Parameter(torch.zeros(self.n_heads, dtype=DTYPE))
            if self.motif_conf:
                self.motif_conf_w = nn.Parameter(torch.zeros(self.n_heads, dtype=torch.float32))
                self.motif_conf_b = nn.Parameter(torch.full((self.n_heads,), 4.0,
                                                            dtype=torch.float32))
            else:
                self.register_parameter("motif_conf_w", None)
                self.register_parameter("motif_conf_b", None)
            if self.motif_mod and not self.motif_shared:
                self.motif_mod_bank = nn.Parameter(torch.empty(
                    self.n_heads, self.motif_count, self.hd, dtype=DTYPE))
                nn.init.normal_(self.motif_mod_bank, std=0.02)
            else:
                self.register_parameter("motif_mod_bank", None)
            if self.motif_mod:
                self.g_motif_mod = nn.Parameter(torch.zeros(self.n_heads, dtype=DTYPE))
            else:
                self.register_parameter("g_motif_mod", None)
            # optional regression objective: the recalled concept must predict
            # the NEXT motion step (MSE) — "trajectory -> what happens next".
            self.motif_pred_weight = float(os.environ.get("TB_WALK_MOTIF_PRED", "0"))
            self.motif_pred = self.motif_pred_weight > 0.0
            print(f"  walk motif memory: {self.motif_count} motion gestures -> "
                  f"concepts, rank={self.motif_rank} (zero-init identity)",
                  flush=True)
        else:
            for _nm in ("motif_q", "motif_key", "motif_concept", "g_motif",
                        "motif_logit_scale", "motif_conf_w", "motif_conf_b",
                        "motif_mod_bank", "g_motif_mod"):
                self.register_parameter(_nm, None)
            self.motif_pred = False
            self.motif_pred_weight = 0.0
        self.motif_pred_loss = torch.tensor(0.0)
        self.acc_loss = torch.tensor(0.0)
        if GPC_WALK_CHUNK_W > 0.0:
            self.chunk_w = nn.Parameter(torch.ones(coarse_s, dtype=DTYPE) / coarse_s)
        self.traj_acc_on = GPC_TRAJ_ACC_ON
        self.traj_smooth_on = GPC_TRAJ_SMOOTH_ON
        self.ln = nn.LayerNorm(D, dtype=DTYPE)
        # per-head per-feature gains: (n_heads, hd)
        self.w1 = nn.Parameter(torch.zeros(self.n_heads, self.hd, dtype=DTYPE))
        self.w2 = nn.Parameter(torch.zeros(self.n_heads, self.hd, dtype=DTYPE))
        self.w4 = nn.Parameter(torch.zeros(self.n_heads, self.hd, dtype=DTYPE))
        self.wt = nn.Parameter(torch.zeros(self.n_heads, self.hd, dtype=DTYPE))
        self.wc1 = nn.Parameter(torch.zeros(self.n_heads, self.hd, dtype=DTYPE))
        self.wc2 = nn.Parameter(torch.zeros(self.n_heads, self.hd, dtype=DTYPE))
        # global-dynamics gains: per-head per-feature (5 vector features) + per-head
        # scalars for the scalar path invariants. Zero-init -> exact no-op.
        self.wg_int = nn.Parameter(torch.zeros(self.n_heads, self.hd, dtype=DTYPE))
        self.wg_ema = nn.Parameter(torch.zeros(self.n_heads, self.hd, dtype=DTYPE))
        self.wg_d16 = nn.Parameter(torch.zeros(self.n_heads, self.hd, dtype=DTYPE))
        self.wg_d64 = nn.Parameter(torch.zeros(self.n_heads, self.hd, dtype=DTYPE))
        self.wg_d256 = nn.Parameter(torch.zeros(self.n_heads, self.hd, dtype=DTYPE))
        self.wg_ena = nn.Parameter(torch.zeros(self.n_heads, dtype=DTYPE))  # energy norm
        self.wg_turn = nn.Parameter(torch.zeros(self.n_heads, dtype=DTYPE))  # total turning
        # nonlinear coupling gains (TB_WALK_COUPLE): local x global products.
        # The walk's remaining blind spot is the *interaction* between scales —
        # products are the first-order nonlinearity: velocity-trend flow,
        # scale interference (multi-scale gradients), curvature-weighted disp.
        self.wg_eflow = nn.Parameter(torch.zeros(self.n_heads, self.hd, dtype=DTYPE))
        self.wg_inter = nn.Parameter(torch.zeros(self.n_heads, self.hd, dtype=DTYPE))
        self.wg_cdisp = nn.Parameter(torch.zeros(self.n_heads, self.hd, dtype=DTYPE))
        self.couple_on = int(os.environ.get("TB_WALK_COUPLE", "0")) == 1
        # adaptive per-token budget gate (TB_WALK_GATE): learns WHERE the walk
        # should inject — predicted from cheap per-token motion signals:
        #   [kappa (direction change), |d1| (velocity), energy, turning]
        # Unlike the scalar curvature gate (gamma), this is a learned per-token
        # multiplier: zero-init -> factor=1 -> exact no-op. Applied AFTER
        # walk_norm so it positions the restricted budget per token instead of
        # spending it blindly. Requires glob_on (uses ena/turn_tot).
        self.gate_on = (int(os.environ.get("TB_WALK_GATE", "0")) == 1
                        and self.glob_on)  # gate uses ena/turn_tot (glob-only)
        # exit gate (TB_WALK_EXIT, Ouro-style ponder): a per-token learned scalar
        # in [0,1] saying "walk has converged here, stop looping". Combined with
        # the kappa/energy motion signals, the model learns HOW MUCH loop depth
        # each token needs. zero-init -> gate=0 -> never exits early at start.
        self.exit_on = int(os.environ.get("TB_WALK_EXIT", "0")) == 1
        # loop + checkpoint (TB_WALK_LOOP_CKPT=1): recompute each loop pass on
        # backward so only one pass's graph is retained (memory fix for the
        # local 6GB card) — used by the SHARED/RAMP variants; EXIT keeps exact.
        self.loop_ckpt = int(os.environ.get("TB_WALK_LOOP_CKPT", "0")) == 1
        if self.exit_on:
            self.exit_w = nn.Parameter(torch.zeros(1, dtype=DTYPE))   # scalar sigmoid bias
            if self.glob_on:
                self.exit_mlp = nn.Linear(4, 1, bias=False, dtype=DTYPE)
                nn.init.zeros_(self.exit_mlp.weight)
            else:
                self.register_parameter("exit_mlp", None)
        else:
            self.register_parameter("exit_w", None)
            self.register_parameter("exit_mlp", None)
        self.exit_scores = None  # legacy alias (kept for reading; Block reads _exit_scores)
        self._exit_scores = None  # set per-block; read by Block loop
        if self.gate_on:
            self.gate_mlp = nn.Linear(4, 1, bias=False, dtype=DTYPE)
            nn.init.zeros_(self.gate_mlp.weight)
        else:
            self.register_parameter("gate_mlp", None)
        # cross-head mixing: out' = out + head_cross @ out  (0 -> identity no-op)
        self.head_cross = nn.Parameter(torch.zeros(self.n_heads, self.n_heads, dtype=DTYPE))
        if self.mix_on:
            self.mix_proj = nn.Parameter(torch.zeros(self.mix_depth, D, D, dtype=DTYPE))
        else:
            self.register_parameter("mix_proj", None)
        # Content-adaptive scale selection (TB_WALK_ADAPT_SCALE): a per-token,
        # per-head softmax over a bank of causal lag operators. A *static*
        # depthwise conv applies the same fixed taps to every token; here the
        # model learns WHICH lag-scale each token should read (query/key over
        # lags), then mixes the corresponding differences. This is the one piece
        # the fixed d1/d2/d4 basis cannot express. Per-head gains are zero-init,
        # so the added term is exactly 0 at construction (bitwise no-op).
        #
        # TB_WALK_CONV_BASELINE is the matched STATIC anchor: a learned
        # per-head, per-channel causal kernel over the SAME lag bank, with no
        # content dependence. A/B the two to test whether per-token adaptivity
        # beats a static mixture of equal (or, here, larger) capacity.
        self.adapt_on = int(os.environ.get("TB_WALK_ADAPT_SCALE", "0")) == 1
        self.conv_on = int(os.environ.get("TB_WALK_CONV_BASELINE", "0")) == 1
        if self.adapt_on or self.conv_on:
            try:
                self.scale_lags = tuple(
                    int(v) for v in os.environ.get(
                        "TB_WALK_ADAPT_LAGS", "1,2,4,8,16").split(",") if v.strip())
            except ValueError:
                raise ValueError(
                    "TB_WALK_ADAPT_LAGS must be comma-separated integers") from None
            if not self.scale_lags or min(self.scale_lags) < 1:
                raise ValueError("TB_WALK_ADAPT_LAGS must be positive integers")
        if self.adapt_on:
            self.adapt_rank = max(1, int(os.environ.get("TB_WALK_ADAPT_RANK", "8")))
            self.adapt_q = nn.Parameter(torch.empty(
                self.n_heads, self.hd, self.adapt_rank, dtype=DTYPE))
            self.adapt_k = nn.Parameter(torch.empty(
                len(self.scale_lags), self.n_heads, self.adapt_rank, dtype=DTYPE))
            nn.init.normal_(self.adapt_q, std=0.02)
            nn.init.normal_(self.adapt_k, std=0.02)
            self.adapt_gain = nn.Parameter(torch.zeros(
                self.n_heads, self.hd, dtype=DTYPE))
            _n_adapt = (self.n_heads * self.hd * self.adapt_rank
                        + len(self.scale_lags) * self.n_heads * self.adapt_rank
                        + self.n_heads * self.hd)
            print(f"  walk adaptive-scale: lags={self.scale_lags} "
                  f"rank={self.adapt_rank} ({_n_adapt:,} params/layer, "
                  "zero-init identity)", flush=True)
        else:
            self.register_parameter("adapt_q", None)
            self.register_parameter("adapt_k", None)
            self.register_parameter("adapt_gain", None)
        if self.conv_on:
            self.conv_w = nn.Parameter(torch.zeros(
                self.n_heads, self.hd, len(self.scale_lags), dtype=DTYPE))
            _n_conv = self.n_heads * self.hd * len(self.scale_lags)
            print(f"  walk static-conv baseline: lags={self.scale_lags} "
                  f"({_n_conv:,} params/layer, zero-init identity)", flush=True)
        else:
            self.register_parameter("conv_w", None)
        self.gamma = nn.Parameter(torch.tensor(0.0, dtype=DTYPE))
        # Causal Cycle Guard (CCG): Walker's difference filters have exact
        # nulls at stationary and short-period trajectories. Under
        # autoregressive decoding those nulls can become repetition attractors,
        # especially when `renorm` keeps every nonzero Walk update at fixed RMS.
        # CCG detects sustained lag recurrence using past states only, then
        # redirects the Walk residual toward a learned direction orthogonal to
        # the repeated state. Its controller is exactly zero at construction,
        # preserving old-checkpoint logits, while a straight-through derivative
        # lets it learn on the first resumed optimizer step.
        self.cycle_guard_on = os.environ.get("TB_WALK_CYCLE_GUARD", "0") == "1"
        # Keep the CCG parameters in the model/checkpoint for exact resume
        # compatibility, but allow ordinary pretraining to bypass its costly
        # recurrence scan.  This is deliberately separate from
        # `cycle_guard_on`: changing the latter changes the state schema.
        self.cycle_guard_runtime = os.environ.get("TB_WALK_CYCLE_RUNTIME", "1") == "1"
        if self.cycle_guard_on:
            parsed_lags = {
                int(value.strip())
                for value in os.environ.get("TB_WALK_CYCLE_LAGS", "1,2,4,8").split(",")
                if value.strip()
            }
            if not parsed_lags or min(parsed_lags) <= 0:
                raise ValueError("TB_WALK_CYCLE_LAGS must contain positive integers")
            self.cycle_lags = tuple(sorted(parsed_lags))
            self.cycle_threshold = _bounded_env_float(
                "TB_WALK_CYCLE_THRESHOLD", 0.95, -0.99, 0.9999)
            self.cycle_persistence = max(
                1, int(os.environ.get("TB_WALK_CYCLE_PERSISTENCE", "4")))
            self.cycle_max_strength = _bounded_env_float(
                "TB_WALK_CYCLE_MAX_STRENGTH", 0.75, 0.0, 1.0)
            self.cycle_open_target = _bounded_env_float(
                "TB_WALK_CYCLE_OPEN_TARGET", 0.50, 0.0, self.cycle_max_strength)
            self.cycle_open_weight = _bounded_env_float(
                "TB_WALK_CYCLE_OPEN_WEIGHT", 1e-3, 0.0, 1.0)
            self.cycle_escape_on = os.environ.get("TB_WALK_CYCLE_ESCAPE", "0") == "1"
            if self.cycle_escape_on:
                cycle_rank = max(4, min(D, int(os.environ.get("TB_WALK_CYCLE_RANK", "32"))))
                self.cycle_up = nn.Linear(D, cycle_rank, bias=False, dtype=DTYPE)
                self.cycle_down = nn.Linear(cycle_rank, D, bias=False, dtype=DTYPE)
                # Do not perturb the backbone's global initialization stream in
                # fresh A/B runs merely because an exact-zero sidecar exists.
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(0xCC6)
                    nn.init.normal_(self.cycle_up.weight, mean=0.0, std=0.02)
                    nn.init.normal_(self.cycle_down.weight, mean=0.0, std=0.005)
                self.cycle_escape_gate_raw = nn.Parameter(
                    torch.zeros((), dtype=torch.float32))
            else:
                self.cycle_up = None
                self.cycle_down = None
                self.register_parameter("cycle_escape_gate_raw", None)
            # Tiny boundary controllers remain FP32 so a 1B-scale LR does not
            # quantize away the first updates.
            self.cycle_gate_raw = nn.Parameter(torch.zeros((), dtype=torch.float32))
        else:
            self.cycle_lags = ()
            self.cycle_threshold = 0.95
            self.cycle_persistence = 1
            self.cycle_max_strength = 0.0
            self.cycle_open_target = 0.0
            self.cycle_open_weight = 0.0
            self.cycle_escape_on = False
            self.cycle_up = None
            self.cycle_down = None
            self.register_parameter("cycle_gate_raw", None)
            self.register_parameter("cycle_escape_gate_raw", None)
        self.cycle_pressure_mean = torch.tensor(0.0)
        self.cycle_strength = torch.tensor(0.0)
        self.cycle_escape_strength = torch.tensor(0.0)
        self.cycle_aux_loss = torch.tensor(0.0)
        self.S = coarse_s
        # Evaluation-only ablation switch. It is deliberately not a parameter or
        # a checkpoint field: disabling Walk must not alter a resumed model.
        self.enabled = True

    def attach_motif_bank(self, bank):
        """Point a shared-dictionary walk at the model-level bank (not a
        registered submodule here, so its tensors are saved exactly once)."""
        object.__setattr__(self, "_motif_bank", bank)

    def _motif_tables(self):
        bank = self._motif_bank if self.motif_shared else None
        if self.motif_shared and bank is None:
            raise RuntimeError("TB_WALK_MOTIF_SHARED walk has no attached motif bank")
        key = bank.key if bank is not None else self.motif_key
        concept = bank.concept if bank is not None else self.motif_concept
        mod = None
        if self.motif_mod:
            mod = bank.mod if bank is not None else self.motif_mod_bank
        return key, concept, mod

    def _motif_shape(self, d1, d2, d4, turn):
        """v2 query input: RMS-unit [d1, d2, d4, turn] (Gram-Schmidt'd under
        TB_WALK_MOTIF_ORTHO), concatenated to (B, T, n, 4*hd)."""
        def _rms_unit(t, floor=1e-6):
            return t / t.pow(2).mean(dim=-1, keepdim=True).add(floor).sqrt()
        _blocks = [_rms_unit(f) for f in (d1, d2, d4, turn)]
        if self.motif_ortho:
            # Gram-Schmidt over the shape blocks. Blocks are RMS-unit, so
            # ||p||^2 = hd: the projection must divide by p.p (dropping
            # that factor subtracts p hd-times over and collapses every
            # block onto -p). Residual RMS is floored at ~0.03 so a block
            # that is genuinely redundant fades instead of amplifying noise.
            _orth = []
            for _b in _blocks:
                _v = _b
                for _p in _orth:
                    _coef = ((_v * _p).sum(-1, keepdim=True)
                             / (_p * _p).sum(-1, keepdim=True).clamp_min(1e-6))
                    _v = _v - _coef * _p
                _orth.append(_rms_unit(_v, floor=1e-3))
            _blocks = _orth
        return torch.cat(_blocks, dim=-1)

    def _motif_residual(self, xs, d1, d2, d4, turn, dt):
        """Motif contribution for any number of positions (full forward and
        cached decode share this). Inputs are (B, T, n, hd); returns
        (residual, recall). Purely per-position: no state beyond d1/d2/d4/turn,
        which the cached walk already carries."""
        n = self.n_heads
        key, concept, mod = self._motif_tables()
        if self.motif_v2:
            _shape = self._motif_shape(d1, d2, d4, turn)
            _q = F.normalize(torch.einsum("bthd,hdr->bthr", _shape,
                                          self.motif_q.to(dt)), dim=-1, eps=1e-6)
            _k = F.normalize(key.to(dt), dim=-1, eps=1e-6)
            _scale = self.motif_logit_scale.float().clamp(max=math.log(100.0)).exp()
            _logits = (torch.einsum("bthr,hmr->bthm", _q, _k)
                       * _scale.to(dt).view(1, 1, n, 1))
            if 0 < self.motif_topk < _logits.shape[-1]:
                _kth = _logits.topk(self.motif_topk, dim=-1).values[..., -1:]
                _logits = _logits.masked_fill(_logits < _kth, float("-inf"))
            _att = torch.softmax(_logits, dim=-1)
        else:
            _q = torch.einsum("bthd,hdr->bthr", d1, self.motif_q.to(dt))
            _logits = torch.einsum("bthr,hmr->bthm", _q, key.to(dt))
            _att = torch.softmax(_logits / math.sqrt(self.motif_rank), dim=-1)
        _recall = torch.einsum("bthm,hmd->bthd", _att, concept.to(dt))
        _res = self.g_motif.view(1, 1, n, 1).to(dt) * _recall
        if self.motif_mod:
            _gain = torch.einsum("bthm,hmd->bthd", _att, mod.to(dt))
            _res = _res + self.g_motif_mod.view(1, 1, n, 1).to(dt) * _gain * xs
        if self.motif_conf:
            _ent = -(_att * _att.clamp_min(1e-12).log()).sum(-1)
            _conf = 1.0 - _ent / math.log(_att.shape[-1])
            _gate = torch.sigmoid(self.motif_conf_w.view(1, 1, n).to(dt) * _conf
                                  + self.motif_conf_b.view(1, 1, n).to(dt))
            _res = _res * _gate.unsqueeze(-1)
        return _res, _recall

    def _sh(self, x, k):
        if k == 0:
            return x
        return torch.cat([x[:, :k], x[:, :-k]], dim=1)

    def _ema(self, x, alpha, K=64, dim=-3):
        """Causal truncated EMA over the time dim (dim=-3: (B,T,n,hd)).
        ema_t = sum_{k=0..K-1} w_k * x_{t-k} / sum_k w_k, w_k = beta^k * alpha,
        beta = 1 - alpha, summed over the last K causal steps. Fully vectorized
        via conv1d (left-padded => no future leak; K=64 effective horizon)."""
        B_, T_, n_, hd_ = x.shape
        beta = 1.0 - alpha
        rng = torch.arange(K, device=x.device, dtype=torch.float32)
        # Desired delay weights are w[k] = alpha * beta^k, with k=0 (newest)
        # highest. conv1d is cross-correlation, so the left-padded causal kernel
        # must be reversed for out[t] = sum_k w[k] * x[t-k].
        w = (torch.pow(beta, rng).to(x.dtype) * alpha)
        # arrange (B,n,hd,T) for conv1d over time; left-pad for causality
        xr = x.permute(0, 2, 3, 1).reshape(B_ * n_ * hd_, 1, T_)
        xp = torch.nn.functional.pad(xr, (K - 1, 0))
        kern = w.flip(0).view(1, 1, K)
        num = torch.nn.functional.conv1d(xp, kern)[:, :, :T_]
        # Each channel shares the same causal normalization. The old
        # ones_like(xp) path incorrectly counted left-padding as real tokens.
        # Computing the scalar prefix sums is also much cheaper than a second
        # (B * n * hd)-channel convolution.
        valid = torch.arange(T_, device=x.device).clamp(max=K - 1)
        den = w.cumsum(dim=0)[valid].view(1, 1, T_)
        ema = (num / (den + 1e-6)).reshape(B_, n_, hd_, T_).permute(0, 3, 1, 2)
        return ema

    @staticmethod
    def _zero_shift(x, lag):
        if lag <= 0:
            return x
        if lag >= x.size(1):
            return torch.zeros_like(x)
        padding = torch.zeros_like(x[:, :lag])
        return torch.cat((padding, x[:, :-lag]), dim=1)

    def _cycle_pressure(self, xs):
        """Return causal sustained-cycle evidence in [0,1], shape [B,T,H]."""
        B, T, n, _ = xs.shape
        if not self.cycle_lags or T == 0:
            return xs.new_zeros(B, T, n)
        evidence = xs.new_zeros(B, T, n)
        positions = torch.arange(T, device=xs.device).view(1, T, 1)
        scale = max(1e-6, 1.0 - self.cycle_threshold)
        for lag in self.cycle_lags:
            previous = self._zero_shift(xs, lag)
            similarity = F.cosine_similarity(xs, previous, dim=-1, eps=1e-8)
            valid = positions.ge(lag)
            score = ((similarity - self.cycle_threshold) / scale).clamp(0.0, 1.0)
            evidence = torch.maximum(evidence, score * valid.to(score.dtype))
        # Require recurrence to persist across multiple causal positions. This
        # rejects isolated legitimate repeats and responds strongly to a loop.
        history = torch.zeros_like(evidence)
        for lag in range(self.cycle_persistence):
            history = history + self._zero_shift(evidence, lag)
        counts = torch.arange(1, T + 1, device=xs.device, dtype=xs.dtype)
        counts = counts.clamp(max=self.cycle_persistence).view(1, T, 1)
        return evidence * (history / counts)

    def _cycle_guard(self, xin, xs, out):
        if (not self.cycle_guard_on or not self.cycle_guard_runtime
                or self.cycle_max_strength <= 0.0):
            return out
        # The detector is control logic: detach it so the backbone cannot lower
        # loss by hiding recurrence rather than learning to leave the cycle.
        pressure_h = self._cycle_pressure(xs.detach())
        pressure = pressure_h.mean(dim=-1, keepdim=True).unsqueeze(-1)
        def bounded_positive(raw):
            bounded = self.cycle_max_strength * torch.tanh(
                F.relu(raw) / max(self.cycle_max_strength, 1e-6))
            # Use an identity surrogate only at/behind the closed boundary;
            # once open, retain tanh's true derivative so the raw parameter
            # cannot drift arbitrarily while the forward strength is saturated.
            surrogate = torch.where(raw <= 0.0, raw, bounded)
            return bounded.detach() + surrogate - surrogate.detach()

        raw = self.cycle_gate_raw.float()
        strength = bounded_positive(raw)
        escape_strength = (bounded_positive(self.cycle_escape_gate_raw.float())
                           if self.cycle_escape_on else raw.new_zeros(()))
        # Forward is bounded/nonnegative; backward remains identity at zero.
        # This avoids a dead positive gate while retaining an exact migration
        # no-op. If the escape path is not useful, negative raw values keep it
        # closed rather than reversing it into an attractor amplifier.
        self.cycle_aux_loss = self.cycle_open_weight * (
            strength - self.cycle_open_target).square()

        self.cycle_pressure_mean = pressure_h.detach().mean()
        self.cycle_strength = strength.detach()
        self.cycle_escape_strength = escape_strength.detach()
        # Suppression is gently opened by a tiny controller prior, guaranteeing
        # a useful fallback under exposure shift. The orthogonal escape branch
        # has no such prior: CE opens it only when its learned direction helps.
        p = pressure.to(out.dtype)
        guarded = out * (1.0 - strength.to(out.dtype) * p)
        if not self.cycle_escape_on:
            return guarded
        escape = self.cycle_down(F.silu(self.cycle_up(xin.to(DTYPE)))).to(out.dtype)
        escape = escape.view_as(xs)
        # Fill the nullspace instead of reinforcing the repeated state.
        state_norm2 = xs.square().sum(dim=-1, keepdim=True).clamp_min(1e-6)
        escape = escape - (escape * xs).sum(dim=-1, keepdim=True) / state_norm2 * xs
        escape_rms = escape.square().mean(dim=(-1, -2), keepdim=True).sqrt().clamp_min(1e-6)
        target_rms = self.walk_norm if self.walk_norm > 0.0 else 0.1
        escape = escape * (target_rms / escape_rms)
        return guarded + escape_strength.to(out.dtype) * p * escape

    @staticmethod
    def _mix_delta(x_flat, matrix):
        """Dense Walk channel mixing with FP32 trajectories and BF16 GEMM.

        `matrix` is stored in BF16. Casting it to FP32 cannot recover weight
        information, but forces a large D-by-D projection off Tensor Cores.
        The opt-in path quantizes only the projection input to BF16, then casts
        its result back into the FP32 trajectory accumulator. Differential
        features, cosine curvature, norms and cumulative statistics stay FP32.
        """
        if os.environ.get("TB_WALK_MIX_TENSORCORE", "0") == "1":
            return torch.matmul(x_flat.to(matrix.dtype), matrix).to(x_flat.dtype)
        return torch.matmul(x_flat, matrix.to(x_flat.dtype))

    def forward(self, x):
        B, T, D = x.shape
        if not self.enabled:
            if self.exit_on:
                self._exit_scores = torch.zeros(B, T, device=x.device, dtype=x.dtype)
            return torch.zeros_like(x)
        n, hd = self.n_heads, self.hd
        dt = torch.float32 if self.fp32_diffs else DTYPE
        xin = self.ln(x.to(self.ln.weight.dtype)).to(dt)

        # ---- head-space differential features (fp32) ----
        xs = xin.view(B, T, n, hd)
        d1 = xs - self._sh(xs, 1)                 # velocity
        d2 = d1 + self._sh(d1, 1)                 # stride-2 telescoping
        d4 = d2 + self._sh(d2, 2)                 # stride-4 telescoping
        turn = d1 - self._sh(d1, 1)               # second difference
        kappa = 1.0 - F.cosine_similarity(d1, self._sh(d1, 1), dim=-1, eps=1e-8)  # (B,T,n)
        kappa_mean = kappa.mean()
        self.kappa_mean = kappa_mean.detach() if not self.traj_smooth_on else kappa_mean

        def g(p):  # gain vector -> broadcastable (1,1,n,hd)
            return torch.tanh(p).view(1, 1, n, hd)

        out = g(self.w1) * d1 + g(self.w2) * d2 + g(self.w4) * d4 + g(self.wt) * turn
        if self.adapt_on:
            # Per-token softmax over lag scales, then mix the differences.
            # Written as xs - sum_l w_l * shift(xs, l) (weights sum to 1) so no
            # (n_lags, B, T, n, hd) bank is ever materialised.
            q = torch.einsum("bthd,hdr->bthr", xs, self.adapt_q.to(dt))
            scores = torch.einsum("bthr,lhr->bthl", q, self.adapt_k.to(dt))
            wl = torch.softmax(scores, dim=-1)
            shifted = torch.zeros_like(xs)
            for _i, _lag in enumerate(self.scale_lags):
                if _lag < T:
                    shifted = shifted + wl[..., _i:_i + 1] * self._sh(xs, _lag)
            out = out + g(self.adapt_gain) * (xs - shifted)
        if self.conv_on:
            # Matched STATIC baseline: same lag bank, but a fixed learned
            # per-head-per-channel kernel (no content dependence). Zero-init
            # kernel -> the added term is exactly 0 at construction.
            cmix = torch.zeros_like(xs)
            for _i, _lag in enumerate(self.scale_lags):
                if _lag < T:
                    cmix = cmix + (self.conv_w[:, :, _i].view(1, 1, n, hd).to(dt)
                                   * (xs - self._sh(xs, _lag)))
            out = out + cmix
        if self.geom_on:
            # intrinsic geometry of the representation curve. Features are
            # MEASUREMENTS computed under no_grad: only the mixing gains learn,
            # so the ill-conditioned normalisers (norm sqrt at the zero-velocity
            # first token) never poison backbone gradients. Values stay bounded.
            with torch.no_grad():
                eps = 1e-6
                floor = 0.1 * math.sqrt(hd)
                _v = d1
                _sp = _v.pow(2).sum(-1, keepdim=True).clamp_min(0).sqrt()
                tangent = _v / _sp.clamp_min(floor)            # scale-free direction
                v_prev = self._sh(_v, 1)
                sp_prev = self._sh(_sp, 1)
                cosang = ((_v * v_prev).sum(-1, keepdim=True)
                          / (_sp.clamp_min(floor) * sp_prev.clamp_min(floor))
                          ).clamp(-1.0, 1.0)
                curv = 1.0 - cosang                            # bending
                _a = _v - v_prev                               # acceleration
                u_hat = v_prev / sp_prev.clamp_min(floor)
                _wv = _v - (_v * u_hat).sum(-1, keepdim=True) * u_hat
                p_hat = _wv / _wv.pow(2).sum(-1, keepdim=True).sqrt().clamp_min(floor)
                a_par = ((_a * u_hat).sum(-1, keepdim=True) * u_hat
                         + (_a * p_hat).sum(-1, keepdim=True) * p_hat)
                tors = (((_a - a_par).pow(2).sum(-1, keepdim=True).sqrt()
                         / _a.pow(2).sum(-1, keepdim=True).sqrt().clamp_min(floor)
                         ).clamp(0.0, 2.0))                    # out-of-plane
                arc = torch.cumsum(_sp.squeeze(-1) + eps, dim=1)
                arc = (arc / (arc[:, -1:] + eps)).unsqueeze(-1)  # normalized arc
                dwell = 1.0 / (1.0 + _sp / (_sp.mean(dim=1, keepdim=True) + eps))
            out = (out
                   + self.g_tan.view(1, 1, n, hd).to(dt) * tangent
                   + self.g_curv.view(1, 1, n, 1).to(dt) * curv
                   + self.g_tors.view(1, 1, n, 1).to(dt) * tors
                   + self.g_dwell.view(1, 1, n, 1).to(dt) * dwell
                   + self.g_arc.view(1, 1, n, 1).to(dt) * arc)
        if self.place_on:
            # self-localization features, also measurements (no_grad); only the
            # place gains learn.
            with torch.no_grad():
                eps = 1e-6
                _xn = xs.pow(2).sum(-1, keepdim=True).sqrt().clamp_min(eps)
                _feats = []
                for _alpha in self.place_alphas:
                    _land = self._ema(xs, _alpha)              # causal landmark
                    _diff = xs - _land
                    _feats.append((
                        _diff.pow(2).sum(-1, keepdim=True).sqrt() / _xn,
                        F.cosine_similarity(d1, _diff, dim=-1, eps=1e-8).unsqueeze(-1)))
            for _k, (_dist, _rad) in enumerate(_feats):
                out = (out
                       + self.g_pdist[_k].view(1, 1, n, 1).to(dt) * _dist
                       + self.g_prad[_k].view(1, 1, n, 1).to(dt) * _rad)
        if self.hurst_on:
            # gait: log2(||Δ_b||/||Δ_a||) over telescoped differences — the local
            # Hurst exponent. Measurements (no_grad): only the gains learn.
            with torch.no_grad():
                floor = 1e-3
                _nm = {}
                for _k in (1, 2, 4, 8):
                    _dk = xs - self._sh(xs, _k)
                    _nm[_k] = _dk.pow(2).sum(-1, keepdim=True).sqrt()  # (B,T,n,1)
                _hs = []
                for (_a, _b) in self.hurst_pairs:
                    _ratio = (_nm[_b] / _nm[_a].clamp_min(floor)).clamp(1e-3, 1e3)
                    _hs.append((_ratio.log() / math.log(2.0)).clamp(-2.0, 2.0))
            for _p, _h in enumerate(_hs):
                out = out + self.g_hurst[_p].view(1, 1, n, 1).to(dt) * _h
        if self.affine_on:
            # affine step: B ~ x*A (self-similar repetition) + d (orthogonal
            # novelty). Measurements (no_grad); only the gains learn.
            with torch.no_grad():
                floor = 0.1 * math.sqrt(hd)
                _A = self._sh(xs, 1)                       # previous state
                _B = xs
                _aa = _A.pow(2).sum(-1, keepdim=True).clamp_min(floor * floor)
                _x = (_A * _B).sum(-1, keepdim=True) / _aa          # LS scale
                _d = _B - _x * _A                                   # A.d == 0
                _dn = _d.pow(2).sum(-1, keepdim=True).sqrt()
                _bn = _B.pow(2).sum(-1, keepdim=True).sqrt()
                _novel = _dn / (_bn + floor)                        # [0, ~1]
                _dirn = _d / _dn.clamp_min(floor)                   # unit novelty
            out = (out
                   + self.g_aff_scale.view(1, 1, n, 1).to(dt) * _x.clamp(-10.0, 10.0)
                   + self.g_aff_novel.view(1, 1, n, 1).to(dt) * _novel
                   + self.g_aff_dir.view(1, 1, n, hd).to(dt) * _dirn)
        if self.motif_on:
            # gesture dictionary: match the current motion shape to a learned
            # codebook of trajectory motifs, then recall what is bound to them.
            _res, _recall = self._motif_residual(xs, d1, d2, d4, turn, dt)
            out = out + _res
            if self.motif_pred:
                # regression: the recalled concept predicts the next step's
                # motion (d1 shifted one token into the future).
                _next = torch.cat([d1[:, 1:], torch.zeros_like(d1[:, :1])], dim=1)
                self.motif_pred_loss = (self.motif_pred_weight
                                        * F.mse_loss(_recall.float(),
                                                     _next.detach().float()))
        if self.glob_on:
            # ---- global-dynamics features (causal: past-only reductions) ----
            # ALL normalized to O(1) per-token scale: a running SUM grows ~T and
            # its gradient w.r.t. the zero-init gains is ~T larger than the local
            # features -> the gains jump at step 1, destabilizing training.
            # int: cumulative displacement, normalized by sqrt(1+t)
            tpos = torch.arange(1, T + 1, device=x.device, dtype=dt).view(1, T, 1, 1)
            int_f = (xs - xs[:, :1])
            int_n = int_f / tpos.sqrt()
            # ema: trend residual x_t - EMA_t (causal conv, newest-weighted)
            _ema_trace = self._ema(xs, 0.05)
            if self.selmem_on:
                # input-dependent memory: a learned per-token gate re-mixes the
                # fixed-decay trace toward the current state (selective-memory
                # approximation). g is zero-init -> _ema_trace is unchanged.
                _gate = torch.sigmoid(self.mem_sel_w.view(1, 1, n, hd).to(dt) * xs)
                _ema_trace = _ema_trace + torch.tanh(self.mem_sel_g).to(dt) * _gate * (xs - _ema_trace)
            ema_f = xs - _ema_trace
            ema_n = ema_f / (1.0 + tpos.sqrt() / 16.0)
            # long-stride telescopes: x_t - x_{t-k}
            d16_f = xs - self._sh(xs, 16)
            d64_f = xs - self._sh(xs, 64)
            d256_f = xs - self._sh(xs, min(256, T))
            # path energy: running AVERAGE of ||d1||^2 / hd (keeps O(1) — the raw
            # squared-diff sum over hd dims is ~O(hd) per token, and dividing by
            # position alone leaves ~O(250) as measured)
            ena = torch.cumsum((d1 * d1).sum(dim=-1, keepdim=True) / hd, dim=1) / tpos
            # total turning: running AVERAGE of kappa (O(1))
            turn_tot = torch.cumsum(kappa.unsqueeze(-1), dim=1) / tpos
            out = (out + g(self.wg_int) * int_n + g(self.wg_ema) * ema_n
                   + g(self.wg_d16) * d16_f + g(self.wg_d64) * d64_f + g(self.wg_d256) * d256_f)
            out = out + torch.tanh(self.wg_ena).view(1, 1, n, 1) * ena
            out = out + torch.tanh(self.wg_turn).view(1, 1, n, 1) * turn_tot
            if self.couple_on:
                # local x global products: energy flow (velocity x trend residual),
                # scale interference (d1 x d16 multi-scale gradient), curvature-
                # weighted displacement. All O(1)-normalized inputs -> O(1) products.
                eflow = d1 * ema_n                       # velocity x trend
                inter = d1 * d16_f                       # multi-scale interference
                cdisp = int_n * ena                      # displacement weighted by energy
                out = (out + g(self.wg_eflow) * eflow
                       + g(self.wg_inter) * inter
                       + g(self.wg_cdisp) * cdisp)
        # ---- cross-head mixing (0 -> identity) ----
        out = out + torch.einsum("bthd,hg->btgd", out, self.head_cross.to(dt))
        if self.mix_proj is not None:
            # full D->D mixing residuals: every channel of every head can mix;
            # zero-init => out += 0 => exact no-op at start.
            # Flat matmul (B,T,D)@(D,D) — no giant (...,D,D) intermediate that
            # the block-einsum version materializes (4.3 GB at T=2048, OOM).
            # PARALLEL stack: each mix projection applies to the ORIGINAL out
            # (out += g * (out @ M_pi)), NOT to the previous mix's result.
            # (A nested/cascade version compounds the Jacobian as prod(I+g*M_pi),
            #  which amplifies gradients multiplicatively with depth — the cause
            #  of gnorm=11072 at D=2048 with mix_depth=4.)
            x_flat = out.reshape(B, T, D)
            acc = x_flat
            for pi in range(self.mix_depth):
                acc = acc + self.mix_gain * self._mix_delta(x_flat, self.mix_proj[pi])
            out = acc.view(B, T, n, hd)

        # ---- coarse concept-trajectory (per-head too) ----
        S = self.S
        # The previous completed S-token chunk is already available as soon as
        # T>S. Full-length training uses this feature at positions S..2S-1,
        # so short-prefix inference must not wait until T>=2S to enable it.
        if T > S:
            Tp = ((T + S - 1) // S) * S
            hp = F.pad(xin, (0, 0, 0, Tp - T))
            hc = hp.view(B, Tp // S, S, n, hd)
            if getattr(self, "chunk_w", None) is not None:
                c = (hc * self.chunk_w.view(1, 1, S, 1, 1)).sum(dim=2)
            else:
                c = hc.mean(dim=2)
            c1 = torch.cat([c.new_zeros(B, 1, n, hd), c[:, :-1]], dim=1)
            c2 = torch.cat([c.new_zeros(B, 2, n, hd), c[:, :-2]], dim=1)
            feat_s = c1.unsqueeze(2).expand(-1, -1, S, -1, -1).reshape(B, Tp, n, hd)[:, :T]
            feat_v = (c1 - c2).unsqueeze(2).expand(-1, -1, S, -1, -1).reshape(B, Tp, n, hd)[:, :T]
            if self.traj_acc_on:
                self.acc_loss = (F.mse_loss(2.0 * c1 - c2, c)
                                 if c.shape[1] > 2 else torch.tensor(0.0, device=x.device))
            out = out + g(self.wc1) * feat_s + g(self.wc2) * feat_v
        else:
            self.acc_loss = torch.tensor(0.0, device=x.device)

        out = out.view(B, T, D)
        if getattr(self, "walk_norm", 0.0) > 0.0:
            dn = out.norm(dim=-1, keepdim=True) / math.sqrt(D) + 1e-6
            mode = self.walk_norm_mode
            if mode == "clip":
                out = out * torch.clamp(self.walk_norm / dn, max=1.0)
            elif mode == "cap":
                out = out * (self.walk_norm / (dn + self.walk_norm))
            else:
                out = out / dn * self.walk_norm
        # per-head curvature gate: (B,T,n,1) * (B,T,n,hd) — stays fp32 until the
        # final cast (bf16 * fp32 promotes, leaking fp32 into the residual stream)
        out = out.view(B, T, n, hd)
        if self.gate_on:
            # learned per-token budget: where to inject (motion signals).
            # gate inputs: kappa (B,T,n), |d1| (B,T,n), ena (B,T,n,1), turn (B,T,n,1)
            with torch.no_grad():
                d1n = d1.pow(2).sum(dim=-1, keepdim=True).sqrt() / (hd ** 0.5)  # (B,T,n,1)
            g_in = torch.cat([kappa.unsqueeze(-1), d1n,
                              ena, turn_tot], dim=-1).to(dt)      # (B,T,n,4)
            g = torch.tanh(g_in @ self.gate_mlp.weight.to(dt).t())  # (B,T,n,1), 0 at init
            out = out * (1.0 + g)
        out = out * (1.0 + torch.tanh(self.gamma).to(dt) * kappa.unsqueeze(-1))
        out = self._cycle_guard(xin, xs, out)
        out = out.view(B, T, D)
        if self.exit_on:
            # per-token convergence score in (0,1): sigmoid(bias + mlp(motion)).
            # 0.5 at init (bias=0, mlp=0) — half-converged, so soft-mask grads
            # always flow; learned to ->1 (early exit) or ->0 (never).
            with torch.no_grad():
                d1n = d1.pow(2).sum(dim=-1, keepdim=True).sqrt() / (hd ** 0.5)
            if self.exit_mlp is not None:
                feats = torch.cat([kappa.unsqueeze(-1), d1n, ena, turn_tot], dim=-1).to(dt)
                z = self.exit_w.to(dt) + feats @ self.exit_mlp.weight.to(dt).t()
                self._exit_scores = torch.sigmoid(z.squeeze(-1).mean(dim=-1))  # (B,T) in graph
            else:
                self._exit_scores = torch.sigmoid(self.exit_w.to(dt)).expand(B, T)
        out = out.to(x.dtype)
        return out


# --- REMOVED 2026-09-30 (lean cleanup): subspace/evidence Phase-A Walk variants ---
# What lived here (~415 lines, removed 2026-09-30): SubspaceWalkAdapter,
# GatedSubspaceWalkAdapter, MemoryGatedSubspaceWalkAdapter,
# ContentGatedSubspaceWalkAdapter, SelectiveAssociativePathAdapter,
# DualPayloadAssociativePathAdapter, EvidenceWalkAdapter (all removed 2026-09-30).
# Why removed: ckpt_ongoing.pt contains ZERO tensors for any of them in all 16
#   blocks (blocks.*.walk is the legacy WalkAdapter: w1/w2/w4/wt/wc/gamma +
#   mix_proj + gate_mlp, n_heads=4). The running lineage selects them only via
#   TB_WALK_SUBSPACE_RANK/TB_WALK_EVIDENCE/TB_WALK_DUAL_PATH_MIXER/
#   TB_WALK_PATH_MIXER/TB_WALK_SUBSPACE_CONTENT/TB_WALK_MEMORY_CHUNK/
#   TB_WALK_SUBSPACE_GATED, all default-off. make_walk_adapter now raises on
#   those flags instead of constructing (fail fast, no silent fallback).
# NOT removed: HierarchicalEvidenceWalkAdapter, which survives below because
#   the Phase-B sidecar (make_phase_b_sidecar, every block) IS an instance of
#   it — verified against ckpt phase_walk tensors (arch_q/k/v, data_in_proj
#   alias, hier_gate, pred_head, ...). _call_walk keeps its depth-kwarg branch
#   and temporarily_disable_walk keeps matching it for that reason.
# To restore: recover from train_walker_1b_continue.py.bak-20260930.
class MotifBank(nn.Module):
    """Shared walk gesture dictionary (TB_WALK_MOTIF_SHARED): one set of
    gesture keys, concepts and (optional) content-modulation gains read by
    every block's motif query. Init matches the per-block v2 tables."""

    def __init__(self, n_heads, count, rank, hd, mod=False):
        super().__init__()
        self.key = nn.Parameter(torch.empty(n_heads, count, rank, dtype=DTYPE))
        self.concept = nn.Parameter(torch.empty(n_heads, count, hd, dtype=DTYPE))
        nn.init.normal_(self.key, std=1.0)
        nn.init.normal_(self.concept, std=0.02)
        if mod:
            self.mod = nn.Parameter(torch.empty(n_heads, count, hd, dtype=DTYPE))
            nn.init.normal_(self.mod, std=0.02)
        else:
            self.register_parameter("mod", None)


class WalkDescentRefiner(nn.Module):
    """Learned descent steps on the residual stream (TB_WDR, experimental).

    Backprop already computes g = dL/dx at every layer, but it needs the
    target, so it can never be an inference feature. What CAN transfer is a
    predictor of it: a low-rank head reads the walk view of the state
    ([LN(x), d1], d1 = causal time-velocity of LN(x)) and outputs a direction
    trained (auxiliary cosine loss, its own graph) toward the normalized
    descent direction -g, captured for free by a backward hook. The forward
    applies x <- x + alpha * dir(x) for `loops` steps (alpha zero-init:
    exact identity until learned), so test-time loops are k learned descent
    steps at the cost of two thin matmuls each, not extra blocks.

    Training: call aux_backward(scale) right after the LM loss backward.
    """

    def __init__(self, D, rank=64, loops=1, aux_weight=0.1):
        super().__init__()
        self.D = int(D)
        self.loops = max(0, int(loops))
        self.aux_weight = float(aux_weight)
        self.inp = nn.Linear(2 * self.D, int(rank), bias=False, dtype=DTYPE)
        self.out = nn.Linear(int(rank), self.D, bias=False, dtype=DTYPE)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0x3D2)
            nn.init.normal_(self.inp.weight, std=(2 * self.D) ** -0.5)
            nn.init.normal_(self.out.weight, std=int(rank) ** -0.5)
        self.alpha = nn.Parameter(torch.zeros((), dtype=torch.float32))
        self._aux_x = None
        self._aux_g = None

    def direction(self, x):
        h = F.layer_norm(x.float(), (self.D,)).to(self.inp.weight.dtype)
        d1 = h - torch.cat([h[:, :1], h[:, :-1]], dim=1)
        return self.out(F.gelu(self.inp(torch.cat([h, d1], dim=-1))))

    def forward(self, x, loops=None):
        if self.training and self.aux_weight > 0.0 and x.requires_grad:
            self._aux_x = x.detach()
            x.register_hook(self._capture_grad)
        steps = self.loops if loops is None else max(0, int(loops))
        alpha = self.alpha.to(x.dtype)
        for _ in range(steps):
            x = x + alpha * self.direction(x).to(x.dtype)
        return x

    def _capture_grad(self, grad):
        self._aux_g = grad.detach()

    def aux_backward(self, scale=1.0):
        """Backprop the auxiliary loss (refiner params only). Returns its value."""
        if self._aux_x is None or self._aux_g is None or self.aux_weight <= 0.0:
            return 0.0
        g = self._aux_g.float()
        keep = g.pow(2).sum(-1) > 0
        pred = self.direction(self._aux_x).float()
        cos = F.cosine_similarity(pred, -g, dim=-1, eps=1e-8)
        loss = self.aux_weight * (1.0 - cos[keep]).mean() if keep.any() else pred.sum() * 0.0
        (loss * float(scale)).backward()
        self._aux_x = self._aux_g = None
        return float(loss.detach())


def _assoc_chunk_size(env_name, default):
    """Chunk width for the Python associative scan; 0 (or negative) means one
    full-T pass. The single pass is algebraically identical to chunking
    (prior-only prefix via cumsum-minus-self, carried state starts at zero)
    up to floating-point summation order, while replacing T/C Python chunk
    iterations with one vectorized pass. At production ranks the full-T
    prefix tensor is a few hundred MB; keep chunking only if VRAM is tight."""
    try:
        raw = int(os.environ.get(env_name, default))
    except ValueError:
        raise ValueError(f"{env_name} must be an integer chunk width (0 = full-T single pass)")
    if raw <= 0:
        return 0
    return max(32, raw)


class HierarchicalEvidenceWalkAdapter(nn.Module):
    """Corrected hierarchical trajectory sidecar with optional Phase-B archive.

    Ordinary HIER remains available through TB_WALK_HIER_EVIDENCE.  The Phase-B
    variant is created with ``phase_b=True`` and deliberately differs in four
    ways needed by the long-context continuation:
      * affine-free shared RMS normalization for model/data views;
      * channelwise state-vs-motion gate [B,T,2,r];
      * per-position *completed-prefix* hierarchy with zero-init learned gates;
      * optional archive parameters always present, controlled by runtime scales
        so Phase-B resumes never need another shape change.
    """

    def __init__(self, D, rank=None, hier_scales=(8, 64), archive_chunk=None,
                 archive_topk=None, phase_b=False):
        super().__init__()
        self.phase_b = bool(phase_b)
        if self.phase_b:
            configured = int(rank if rank is not None else os.environ.get('TB_PHASE_WALK_RANK', '32'))
        else:
            configured = int(rank) if rank is not None else int(os.environ.get('TB_WALK_SUBSPACE_RANK', '0'))
        if D <= 0 or configured <= 0 or configured > D:
            raise ValueError('HierarchicalEvidenceWalk rank must be in [1, D]')
        self.D, self.rank = int(D), int(configured)
        self.hier_scales = tuple(int(v) for v in (hier_scales or (8, 64)))

        if self.phase_b:
            self.k_vectors = max(1, int(os.environ.get('TB_PHASE_WALK_K', '2')))
            self.data_traj = os.environ.get('TB_PHASE_WALK_DATA_TRAJ', '1') == '1'
            # Predictor parameters are part of the Phase-B superset from step zero.
            # Runtime scale/weight may change on ordinary resume without any shape
            # change. Defaults keep both auxiliary loss and residual injection OFF.
            self.pred_walk = True
            self.pred_weight = float(os.environ.get('TB_PHASE_PRED_WEIGHT', '0.0'))
            self.pred_scale = float(os.environ.get('TB_PHASE_PRED_SCALE', '0.0'))
            self.traj_mode = os.environ.get('TB_PHASE_WALK_TRAJ', 'full').strip().lower()
            self.affine_free = True
            self.channel_gate = True
            self.learned_hierarchy = True
            self.depth_rms = True
            self.archive_chunk = max(0, int(archive_chunk if archive_chunk is not None
                                            else os.environ.get('TB_PHASE_ARCHIVE_CHUNK', '64')))
            self.archive_topk = max(1, int(archive_topk if archive_topk is not None
                                          else os.environ.get('TB_PHASE_ARCHIVE_TOPK', '1')))
            self.archive_scale = float(os.environ.get('TB_PHASE_ARCHIVE_SCALE', '0'))
            self.assoc_chunk = _assoc_chunk_size('TB_PHASE_ASSOC_CHUNK', '256')
        else:
            self.k_vectors = max(1, int(os.environ.get('TB_WALK_HIER_K', '1')))
            self.data_traj = os.environ.get('TB_WALK_DATA_TRAJ', '0') == '1'
            self.pred_walk = os.environ.get('TB_WALK_PRED', '0') == '1'
            self.pred_weight = float(os.environ.get('TB_WALK_PRED_W', '0.1'))
            self.pred_scale = 1.0 if self.pred_walk else 0.0
            self.traj_mode = os.environ.get('TB_WALK_HIER_TRAJ', 'full').strip().lower()
            self.affine_free = os.environ.get('TB_WALK_HIER_AFFINE_FREE', '0') == '1'
            self.channel_gate = os.environ.get('TB_WALK_HIER_CHANNEL_GATE', '0') == '1'
            self.learned_hierarchy = os.environ.get('TB_WALK_HIER_LEARNED', '0') == '1'
            self.depth_rms = os.environ.get('TB_WALK_HIER_DEPTH_RMS', '0') == '1'
            self.archive_chunk = max(0, int(archive_chunk if archive_chunk is not None
                                            else os.environ.get('TB_WALK_HIER_ARCHIVE_CHUNK', '0')))
            self.archive_topk = max(1, int(archive_topk if archive_topk is not None
                                          else os.environ.get('TB_WALK_HIER_ARCHIVE_TOPK', '1')))
            archive_on = self.archive_chunk > 0 and os.environ.get('TB_WALK_HIER_ARCHIVE', '0') == '1'
            self.archive_scale = 1.0 if archive_on else 0.0
            self.assoc_chunk = _assoc_chunk_size('TB_WALK_ASSOC_CHUNK', '256')

        if self.traj_mode not in ('full', 'd1', 'state'):
            raise ValueError('HIER trajectory mode must be full/d1/state')
        if self.k_vectors < 1 or self.k_vectors > 8:
            raise ValueError('HIER K must be in [1,8]')

        # IMPORTANT: the pretrained Phase-A Walk keeps the global TB_WALK_NORM
        # policy. A fresh Phase-B sidecar must not inherit `renorm`, because an
        # epsilon-sized first update would be amplified immediately to full RMS.
        if self.phase_b:
            self.walk_norm = float(os.environ.get('TB_PHASE_WALK_NORM', '0.05'))
            self.walk_norm_mode = os.environ.get('TB_PHASE_WALK_NORM_MODE', 'clip').strip().lower()
            if self.walk_norm_mode not in ('clip', 'cap', 'renorm'):
                raise ValueError('TB_PHASE_WALK_NORM_MODE must be clip/cap/renorm')
        else:
            self.walk_norm = GPC_WALK_NORM
            self.walk_norm_mode = GPC_WALK_NORM_MODE
        if self.affine_free:
            self.ln = None
            self.ln_data = None
        else:
            self.ln = nn.LayerNorm(D, dtype=DTYPE)
            self.ln_data = nn.LayerNorm(D, dtype=DTYPE) if self.data_traj else None

        if self.k_vectors == 1:
            self.in_proj = nn.Parameter(torch.empty(self.rank, D, dtype=DTYPE))
            self.register_parameter('in_proj_k', None)
        else:
            self.register_parameter('in_proj', None)
            self.in_proj_k = nn.Parameter(torch.empty(self.k_vectors * self.rank, D, dtype=DTYPE))

        if self.data_traj:
            self.data_gate = nn.Parameter(torch.zeros(self.rank, dtype=torch.float32))
            # Shared projection W for model/data trajectory: no duplicate D*r matrix.
            self.data_in_proj = self.in_proj if self.k_vectors == 1 else self.in_proj_k
        else:
            self.register_parameter('data_gate', None)
            self.data_in_proj = None

        if self.pred_walk:
            self.pred_head = nn.Parameter(torch.empty(self.rank, self.rank, dtype=DTYPE))
            self.pred_gate = nn.Parameter(torch.zeros(self.rank, dtype=torch.float32))
            nn.init.kaiming_uniform_(self.pred_head, a=math.sqrt(5))
        else:
            self.register_parameter('pred_head', None)
            self.register_parameter('pred_gate', None)
        self.pred_loss = torch.tensor(0.0)

        self.q_proj = nn.Parameter(torch.empty(self.rank, self.rank, dtype=DTYPE))
        self.k_proj = nn.Parameter(torch.empty(self.rank, self.rank, dtype=DTYPE))
        self.state_v_proj = nn.Parameter(torch.empty(self.rank, self.rank, dtype=DTYPE))
        self.motion_v_proj = nn.Parameter(torch.empty(self.rank, 3 * self.rank, dtype=DTYPE))
        self.state_out = nn.Parameter(torch.empty(self.rank, self.rank, dtype=DTYPE))
        self.motion_out = nn.Parameter(torch.empty(self.rank, self.rank, dtype=DTYPE))

        gate_in = 3 * self.rank
        gate_rows = 2 * self.rank if self.channel_gate else 2
        self.gate_proj = nn.Parameter(torch.empty(gate_rows, gate_in, dtype=DTYPE))
        # depth direction (v/RMS,c/RMS) + log magnitudes are included in Phase B.
        fuse_width = 8 * self.rank if self.depth_rms else 6 * self.rank
        self.fuse_proj = nn.Parameter(torch.empty(self.rank, fuse_width, dtype=DTYPE))
        if self.phase_b:
            self.out_proj = nn.Parameter(torch.empty(D, self.rank, dtype=DTYPE))
            nn.init.kaiming_uniform_(self.out_proj, a=math.sqrt(5))
            # Exact no-op at construction. Unlike a zero output matrix, this scalar
            # receives a useful first-step gradient from a nonzero candidate residual.
            self.phase_residual_gate = nn.Parameter(torch.zeros((), dtype=torch.float32))
        else:
            self.out_proj = nn.Parameter(torch.zeros(D, self.rank, dtype=DTYPE))
            self.register_parameter('phase_residual_gate', None)

        if self.learned_hierarchy:
            self.hier_gate = nn.Parameter(torch.zeros(len(self.hier_scales), self.rank,
                                                       dtype=torch.float32))
        else:
            self.register_parameter('hier_gate', None)

        # In the Phase-B superset the archive parameters exist from step zero even
        # while archive_scale=0.  A zero arch_gate makes later activation itself
        # start as an exact no-op.
        self.archive_present = self.archive_chunk > 0 and (self.phase_b or self.archive_scale != 0.0)
        if self.archive_present:
            self.arch_q = nn.Parameter(torch.empty(self.rank, self.rank, dtype=DTYPE))
            self.arch_k = nn.Parameter(torch.empty(self.rank, self.rank, dtype=DTYPE))
            self.arch_v = nn.Parameter(torch.empty(self.rank, self.rank, dtype=DTYPE))
            self.arch_out = nn.Parameter(torch.empty(self.rank, self.rank, dtype=DTYPE))
            self.arch_gate = nn.Parameter(torch.zeros(self.rank, dtype=torch.float32)) if self.phase_b else None
        else:
            self.register_parameter('arch_q', None)
            self.register_parameter('arch_k', None)
            self.register_parameter('arch_v', None)
            self.register_parameter('arch_out', None)
            self.register_parameter('arch_gate', None)

        self.enabled = True
        self.traj_smooth_on = False
        self.traj_acc_on = False
        # Probe-only output amplifiers. Defaults preserve Phase-B checkpoints;
        # pretrained experiments can strengthen Walk/HIER/predictor separately.
        self.probe_output_scale = 1.0
        self.probe_hier_scale = 1.0
        self.probe_pred_scale = 1.0
        self.kappa_mean = torch.tensor(0.0)
        self.acc_loss = torch.tensor(0.0)
        self.last_view = None

        init_params = [self.in_proj_k if self.k_vectors > 1 else self.in_proj,
                       self.q_proj, self.k_proj, self.state_v_proj, self.motion_v_proj,
                       self.state_out, self.motion_out, self.gate_proj, self.fuse_proj]
        for p in init_params:
            nn.init.kaiming_uniform_(p, a=math.sqrt(5))
        if self.archive_present:
            for p in (self.arch_q, self.arch_k, self.arch_v, self.arch_out):
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))

    @staticmethod
    def _shift(x, steps):
        if steps <= 0:
            return x
        return torch.cat((x[:, :steps], x[:, :-steps]), dim=1)

    @staticmethod
    def _rms_no_affine(x):
        return x * torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + 1e-6).to(x.dtype)

    @staticmethod
    def _unit_rms_with_log(x):
        # eps goes *inside* sqrt so the derivative at an exactly-zero depth
        # velocity is finite; sqrt(0) otherwise creates inf/NaN backward even
        # when the zero-init output projection makes the upstream gradient zero.
        rms = (x.float().square().mean(dim=-1, keepdim=True) + 1e-6).sqrt()
        direction = x / rms.to(x.dtype)
        logmag = torch.log1p(rms).to(x.dtype).expand_as(x)
        return direction, logmag

    def _norm_model(self, x):
        return self._rms_no_affine(x) if self.affine_free else self.ln(x.to(self.ln.weight.dtype))

    def _norm_data(self, x):
        return self._rms_no_affine(x) if self.affine_free else self.ln_data(x.to(self.ln_data.weight.dtype))

    @staticmethod
    def _project_values(value_input, value_proj, dtype):
        return F.linear(value_input, value_proj.to(dtype))

    def _causal_associative_read(self, q, k, value_input, value_proj, out_proj):
        """Prior-only normalized associative read with bounded T-memory.

        A user-supplied Triton kernel may be exposed as
        ``walk_assoc_triton.causal_assoc_read(q,k,v)``.  The built-in fallback is
        chunk-streamed: it never retains a full [B,T,r,r] prefix tensor, so long
        Phase-B contexts remain memory-bounded.
        """
        v = self._project_values(value_input, value_proj, q.dtype)
        backend = os.environ.get('TB_WALK_ASSOC_BACKEND', 'auto').strip().lower()
        if backend in ('auto', 'triton'):
            try:
                from walk_assoc_triton import causal_assoc_read as _triton_assoc
                read = _triton_assoc(q, k, v)
                return F.linear(read, out_proj.to(q.dtype))
            except (ImportError, RuntimeError, AttributeError):
                if backend == 'triton':
                    raise

        B, T, r = q.shape
        rv = v.shape[-1]
        state_kv = q.new_zeros(B, r, rv)
        state_k = q.new_zeros(B, r)
        outs = []
        C = T if self.assoc_chunk <= 0 else min(self.assoc_chunk, T)
        for start in range(0, T, C):
            end = min(T, start + C)
            kc, vc, qc = k[:, start:end], v[:, start:end], q[:, start:end]
            kv = kc.unsqueeze(-1) * vc.unsqueeze(-2)  # bounded [B,C,r,rv]
            local_kv = torch.cumsum(kv, dim=1) - kv
            local_k = torch.cumsum(kc, dim=1) - kc
            prefix_kv = local_kv + state_kv.unsqueeze(1)
            prefix_k = local_k + state_k.unsqueeze(1)
            read = torch.einsum('btr,btrs->bts', qc, prefix_kv)
            normalizer = (qc * prefix_k).sum(dim=-1, keepdim=True).clamp_min(1e-6)
            outs.append(read / normalizer)
            state_kv = state_kv + kv.sum(dim=1)
            state_k = state_k + kc.sum(dim=1)
        return F.linear(torch.cat(outs, dim=1), out_proj.to(q.dtype))

    def _dual_causal_associative_read(self, q, k, state_input, motion_input):
        """Evaluate state and motion payloads in one shared associative scan.

        Both reads use identical q/k prefixes and normalization. Concatenating
        their independently projected values is algebraically equivalent to two
        scans, while eliminating one prefix-key cumsum, one normalizer, and half
        the Python chunk launches. The state/motion parameters and checkpoint
        layout are unchanged.
        """
        state_v = self._project_values(state_input, self.state_v_proj, q.dtype)
        motion_v = self._project_values(motion_input, self.motion_v_proj, q.dtype)
        values = torch.cat((state_v, motion_v), dim=-1)
        backend = os.environ.get('TB_WALK_ASSOC_BACKEND', 'auto').strip().lower()
        # The optional external Triton kernel may assume value width == key
        # width. Its support for a 2*r payload is unknown, so preserve its two
        # established calls; fusion is specifically for the tested Python path.
        if backend != 'python':
            return (self._causal_associative_read(
                        q, k, state_input, self.state_v_proj, self.state_out),
                    self._causal_associative_read(
                        q, k, motion_input, self.motion_v_proj, self.motion_out))
        B, T, r = q.shape
        rv = values.shape[-1]
        state_kv = q.new_zeros(B, r, rv)
        state_k = q.new_zeros(B, r)
        outputs = []
        C = T if self.assoc_chunk <= 0 else min(self.assoc_chunk, T)
        for start in range(0, T, C):
            end = min(T, start + C)
            kc, vc, qc = k[:, start:end], values[:, start:end], q[:, start:end]
            kv = kc.unsqueeze(-1) * vc.unsqueeze(-2)
            local_kv = torch.cumsum(kv, dim=1) - kv
            local_k = torch.cumsum(kc, dim=1) - kc
            prefix_kv = local_kv + state_kv.unsqueeze(1)
            prefix_k = local_k + state_k.unsqueeze(1)
            numerator = torch.einsum('btr,btrs->bts', qc, prefix_kv)
            denominator = (qc * prefix_k).sum(dim=-1, keepdim=True).clamp_min(1e-6)
            outputs.append(numerator / denominator)
            state_kv = state_kv + kv.sum(dim=1)
            state_k = state_k + kc.sum(dim=1)
        read = torch.cat(outputs, dim=1)
        state_read, motion_read = read.split(self.rank, dim=-1)
        return (F.linear(state_read, self.state_out.to(q.dtype)),
                F.linear(motion_read, self.motion_out.to(q.dtype)))

    def _causal_hierarchy(self, u3, scale):
        """For token t, average only chunks whose final position is < t."""
        B, T, r = u3.shape
        C = int(scale)
        chunks = T // C
        if chunks < 1:
            return u3.new_zeros(B, T, r)
        chunk_mean = u3[:, :chunks * C].view(B, chunks, C, r).mean(dim=2)
        prefix = torch.cat((chunk_mean.new_zeros(B, 1, r), chunk_mean.cumsum(dim=1)), dim=1)
        n_complete = (torch.arange(T, device=u3.device) // C).clamp_max(chunks)
        gathered = prefix[:, n_complete]
        denom = n_complete.clamp_min(1).to(gathered.dtype).view(1, T, 1)
        out = gathered / denom
        return torch.where((n_complete > 0).view(1, T, 1), out, torch.zeros_like(out))

    def forward(self, x, depth_velocity=None, depth_curvature=None, data_emb=None):
        B, T, D = x.shape
        if D != self.D:
            raise ValueError('HierarchicalEvidenceWalkAdapter width mismatch')
        if not self.enabled:
            self.pred_loss = torch.tensor(0.0, device=x.device, dtype=torch.float32)
            self.last_view = None
            return torch.zeros_like(x)
        dt = torch.float32
        normed = self._norm_model(x)
        has_k = self.k_vectors > 1
        if not has_k:
            u = F.linear(normed, self.in_proj.to(normed.dtype)).to(dt)
        else:
            k_views = self.k_vectors
            u = F.linear(normed, self.in_proj_k.to(normed.dtype)).to(dt).view(B, T, k_views, self.rank)

        if self.data_traj and data_emb is not None:
            d_norm = self._norm_data(data_emb)
            if not has_k:
                u_data = F.linear(d_norm, self.data_in_proj.to(d_norm.dtype)).to(dt)
            else:
                u_data = F.linear(d_norm, self.data_in_proj.to(d_norm.dtype)).to(dt).view(B, T, k_views, self.rank)
            d1_data = u_data - self._shift(u_data, 1)
            cross = (u - self._shift(u, 1)) - d1_data
            shape = (1, 1, self.rank) if not has_k else (1, 1, 1, self.rank)
            u = u + torch.tanh(self.data_gate.to(dt)).view(*shape) * cross

        d1 = u - self._shift(u, 1)
        d2 = d1 + self._shift(d1, 1)
        d4 = d2 + self._shift(d2, 2)
        if self.traj_mode == 'd1':
            d2, d4 = torch.zeros_like(d1), torch.zeros_like(d1)
        elif self.traj_mode == 'state':
            d1 = torch.zeros_like(d1); d2 = torch.zeros_like(d1); d4 = torch.zeros_like(d1)

        def project_depth(value):
            if value is None:
                return torch.zeros_like(u)
            if not has_k:
                return F.linear(value.to(normed.dtype), self.in_proj.to(normed.dtype)).to(dt)
            return F.linear(value.to(normed.dtype), self.in_proj_k.to(normed.dtype)).to(dt).view(B, T, k_views, self.rank)

        depth_v = project_depth(depth_velocity)
        depth_c = project_depth(depth_curvature)
        if self.traj_mode == 'state':
            depth_v, depth_c = torch.zeros_like(depth_v), torch.zeros_like(depth_c)

        if self.depth_rms:
            depth_v, log_v = self._unit_rms_with_log(depth_v)
            depth_c, log_c = self._unit_rms_with_log(depth_c)
            payload_parts = (u, d1, d2, d4, depth_v, depth_c, log_v, log_c)
        else:
            payload_parts = (u, d1, d2, d4, depth_v, depth_c)

        if not has_k:
            payload = torch.cat(payload_parts, dim=-1)
            view = F.silu(F.linear(payload, self.fuse_proj.to(dt)))
            self.last_view = view
            q = F.elu(F.linear(view, self.q_proj.to(dt))) + 1.0
            kk = F.elu(F.linear(u, self.k_proj.to(dt))) + 1.0
            motion_input = torch.cat((d1, d2, d4), dim=-1)
            if self.traj_mode == 'state':
                motion_input = torch.zeros_like(motion_input)
            if os.environ.get('TB_PHASE_FUSE_ASSOC', '1') == '1':
                r_state, r_motion = self._dual_causal_associative_read(
                    q, kk, u, motion_input)
            else:
                r_state = self._causal_associative_read(
                    q, kk, u, self.state_v_proj, self.state_out)
                r_motion = self._causal_associative_read(
                    q, kk, motion_input, self.motion_v_proj, self.motion_out)
            u3 = u
        else:
            r = self.rank
            payload_k = torch.cat(payload_parts, dim=-1)
            view_k = F.silu(F.linear(payload_k.reshape(B*T*k_views, -1), self.fuse_proj.to(dt))).view(B,T,k_views,r)
            view = view_k.mean(dim=2)
            self.last_view = view
            view_bk = view_k.permute(0,2,1,3).reshape(B*k_views,T,r)
            u_bk = u.permute(0,2,1,3).reshape(B*k_views,T,r)
            q_bk = F.elu(F.linear(view_bk, self.q_proj.to(dt))) + 1.0
            k_bk = F.elu(F.linear(u_bk, self.k_proj.to(dt))) + 1.0
            motion_bk = torch.cat((d1.permute(0,2,1,3).reshape(B*k_views,T,r),
                                   d2.permute(0,2,1,3).reshape(B*k_views,T,r),
                                   d4.permute(0,2,1,3).reshape(B*k_views,T,r)), dim=-1)
            if self.traj_mode == 'state':
                motion_bk = torch.zeros_like(motion_bk)
            if os.environ.get('TB_PHASE_FUSE_ASSOC', '1') == '1':
                rs, rm = self._dual_causal_associative_read(
                    q_bk, k_bk, u_bk, motion_bk)
            else:
                rs = self._causal_associative_read(
                    q_bk, k_bk, u_bk, self.state_v_proj, self.state_out)
                rm = self._causal_associative_read(
                    q_bk, k_bk, motion_bk, self.motion_v_proj, self.motion_out)
            r_state = rs.view(B,k_views,T,r).permute(0,2,1,3).mean(dim=2)
            r_motion = rm.view(B,k_views,T,r).permute(0,2,1,3).mean(dim=2)
            u3 = u.mean(dim=2)

        r_state = r_state / (r_state.norm(dim=-1, keepdim=True) / math.sqrt(self.rank) + 1e-6)
        r_motion = r_motion / (r_motion.norm(dim=-1, keepdim=True) / math.sqrt(self.rank) + 1e-6)
        gate_logits = F.linear(torch.cat((view, r_state, r_motion), dim=-1), self.gate_proj.to(dt))
        if self.channel_gate:
            gate = torch.softmax(gate_logits.view(B, T, 2, self.rank), dim=2)
            mixed = gate[:, :, 0] * r_state + gate[:, :, 1] * r_motion
        else:
            gate = torch.softmax(gate_logits, dim=-1)
            mixed = gate[..., 0:1] * r_state + gate[..., 1:2] * r_motion

        # Predictor is retained for old experiments but is deliberately OFF in
        # Phase B.  k>1 is handled through u3, never by flattening k into time.
        # Skip the whole predictor block when its residual coefficient is
        # exactly zero (pred_scale==0 or probe_pred_scale==0 zeroes both the
        # auxiliary loss factor and the residual injection, whatever
        # pred_weight is). The skipped path leaves pred_loss at exactly 0.0,
        # identical to mse*0, and the mixed residual untouched. Note this also
        # stops zero-gradients flowing to pred_head/pred_gate, so the optimizer
        # skips them (no weight decay) instead of decaying dead parameters;
        # the forward function is unchanged while the predictor stays off.
        if self.pred_walk and self.pred_scale != 0.0 and self.probe_pred_scale != 0.0:
            C = 8
            if T >= 2 * C:
                n_chunks = T // C
                u_chunks = u3[:, :n_chunks*C].view(B,n_chunks,C,self.rank).mean(dim=2)
                view_chunks = view[:, :n_chunks*C].view(B,n_chunks,C,self.rank).mean(dim=2)
                pred_next = F.linear(view_chunks[:,:-1], self.pred_head.to(dt))
                self.pred_loss = (F.mse_loss(pred_next, u_chunks[:,1:].detach())
                                  * self.pred_weight * self.pred_scale * self.probe_pred_scale)
                pred_exp = torch.zeros(B,T,self.rank,device=x.device,dtype=pred_next.dtype)
                pred_exp[:, C:n_chunks*C] = pred_next.repeat_interleave(C, dim=1)
                mixed = mixed + (0.1 * self.pred_scale * self.probe_pred_scale
                                 * torch.tanh(self.pred_gate.to(dt)).view(1,1,-1) * pred_exp)
            else:
                pred = F.linear(view, self.pred_head.to(dt))
                actual = torch.cat((u3[:,1:],u3[:,-1:]),dim=1) - u3
                self.pred_loss = (F.mse_loss(pred[:,:-1], actual[:,:-1].detach())
                                  * self.pred_weight * self.pred_scale * self.probe_pred_scale * 0.1
                                  if T > 1 else x.new_zeros(()))
                mixed = mixed + (0.1 * self.pred_scale * self.probe_pred_scale
                                 * torch.tanh(self.pred_gate.to(dt)).view(1,1,-1) * pred)
        else:
            self.pred_loss = torch.tensor(0.0, device=x.device, dtype=torch.float32)

        # Correct hierarchy: every position sees only summaries of chunks that
        # have already completed.  No global future-containing broadcast.
        for si, scale in enumerate(self.hier_scales):
            hs = self._causal_hierarchy(u3, scale)
            if self.learned_hierarchy:
                mixed = mixed + self.probe_hier_scale * torch.tanh(self.hier_gate[si].to(dt)).view(1,1,-1) * hs
            else:
                mixed = mixed + (0.05 * self.probe_hier_scale) * hs

        if self.archive_present and self.archive_scale != 0.0:
            C = self.archive_chunk
            chunks = T // C
            if chunks > 1:
                u_chunks = u3[:, :chunks*C].view(B,chunks,C,self.rank).mean(dim=2)
                q_a = F.linear(view, self.arch_q.to(dt))
                k_a = F.linear(u_chunks, self.arch_k.to(dt))
                v_a = F.linear(u_chunks, self.arch_v.to(dt))
                scores = torch.matmul(q_a, k_a.transpose(-2,-1)) / math.sqrt(self.rank)
                chunk_end = (torch.arange(chunks,device=x.device)+1)*C - 1
                position = torch.arange(T,device=x.device)
                mask = chunk_end.view(1,1,chunks) < position.view(1,T,1)
                masked = scores.masked_fill(~mask, -1e4)
                soft_w = F.softmax(masked, dim=-1) * mask.to(masked.dtype)
                soft_w = soft_w / soft_w.sum(dim=-1,keepdim=True).clamp_min(1e-6)
                soft = soft_w @ v_a
                topn = min(self.archive_topk, chunks)
                idx = masked.topk(topn,dim=-1).indices
                gathered = v_a.unsqueeze(1).expand(-1,T,-1,-1).gather(
                    2, idx.unsqueeze(-1).expand(-1,-1,-1,self.rank))
                hscore = scores.gather(-1,idx)
                hvalid = mask.expand(B,-1,-1).gather(-1,idx)
                hw = F.softmax(hscore.masked_fill(~hvalid,-1e4),dim=-1) * hvalid.to(hscore.dtype)
                hw = hw / hw.sum(dim=-1,keepdim=True).clamp_min(1e-6)
                hard = (hw.unsqueeze(-1) * gathered).sum(dim=2)
                routed = hard + soft - soft.detach() if self.training else hard
                arch = F.linear(routed, self.arch_out.to(dt))
                if self.arch_gate is not None:
                    arch = arch * torch.tanh(self.arch_gate.to(dt)).view(1,1,-1)
                mixed = mixed + float(self.archive_scale) * arch

        correction = view + mixed

        out = F.linear(correction, self.out_proj.to(dt))
        if self.phase_residual_gate is not None:
            out = out * torch.tanh(self.phase_residual_gate.to(dt))
        out = out * self.probe_output_scale
        if self.walk_norm > 0.0:
            rms = out.norm(dim=-1, keepdim=True) / math.sqrt(D) + 1e-6
            if self.walk_norm_mode == 'clip':
                out = out * torch.clamp(self.walk_norm / rms, max=1.0)
            elif self.walk_norm_mode == 'cap':
                out = out * (self.walk_norm / (rms + self.walk_norm))
            else:
                out = out / rms * self.walk_norm
        return out.to(x.dtype)


def _phase_layer_enabled(index, total=None):
    spec = os.environ.get('TB_PHASE_WALK_LAYERS', 'all').strip().lower()
    if spec in ('', 'all'):
        return True
    chosen = {int(v.strip()) for v in spec.split(',') if v.strip()}
    return int(index) in chosen


def make_phase_b_sidecar(D, block_index):
    if os.environ.get('TB_PHASE_B_SUPERSET', '0') != '1' or not _phase_layer_enabled(block_index):
        return None
    return HierarchicalEvidenceWalkAdapter(
        D,
        rank=int(os.environ.get('TB_PHASE_WALK_RANK', '32')),
        hier_scales=tuple(int(v) for v in os.environ.get('TB_PHASE_HIER_SCALES', '8,64').split(',') if v.strip()),
        archive_chunk=int(os.environ.get('TB_PHASE_ARCHIVE_CHUNK', '64')),
        archive_topk=int(os.environ.get('TB_PHASE_ARCHIVE_TOPK', '1')),
        phase_b=True,
    )

def make_walk_adapter(D):
    """Build the Phase-A Walk adapter for this lineage: legacy WalkAdapter.

    Removed 2026-09-30: SubspaceWalkAdapter (+gated/memory/content/path-mixer
    children) and EvidenceWalkAdapter — removed 2026-09-30, zero tensors in
    ckpt_ongoing.pt. HierarchicalEvidenceWalkAdapter survives (Phase-B sidecar).
    Their flags raise below instead of silently falling back.
    """
    rank = int(os.environ.get("TB_WALK_SUBSPACE_RANK", "0"))
    if rank > 0:
        raise RuntimeError(
            "TB_WALK_SUBSPACE_RANK>0 needs a subspace Walk variant, removed "
            "2026-09-30 (no checkpoint tensors; legacy WalkAdapter only).")
    if os.environ.get("TB_WALK_HIER_EVIDENCE", "0") == "1":
        raise RuntimeError(
            "TB_WALK_HIER_EVIDENCE=1 would build HIER as the Phase-A walk; "
            "HIER is reserved for Phase-B sidecars in this lineage.")
    for flag in ("TB_WALK_EVIDENCE",
                 "TB_WALK_DUAL_PATH_MIXER", "TB_WALK_PATH_MIXER",
                 "TB_WALK_SUBSPACE_CONTENT", "TB_WALK_MEMORY_CHUNK",
                 "TB_WALK_SUBSPACE_GATED"):
        if os.environ.get(flag, "0") not in ("0", ""):
            raise RuntimeError(
                f"{flag} needs a removed Phase-A Walk variant (see make_walk_adapter).")
    return WalkAdapter(D)


@contextmanager
def temporarily_disable_walk(model):
    """Temporarily zero every Walk residual without changing model weights."""
    target = model._orig_mod if hasattr(model, "_orig_mod") else model
    adapters = [module for module in target.modules()
                if isinstance(module, (WalkAdapter, HierarchicalEvidenceWalkAdapter))]
    previous = [adapter.enabled for adapter in adapters]
    try:
        for adapter in adapters:
            adapter.enabled = False
        yield
    finally:
        for adapter, was_enabled in zip(adapters, previous):
            adapter.enabled = was_enabled


@contextmanager
def temporarily_disable_phase_walk(model):
    target = model._orig_mod if hasattr(model, '_orig_mod') else model
    adapters = [getattr(block, 'phase_walk', None) for block in target.blocks]
    adapters = [a for a in adapters if a is not None]
    previous = [a.enabled for a in adapters]
    try:
        for a in adapters:
            a.enabled = False
        yield
    finally:
        for a, value in zip(adapters, previous):
            a.enabled = value


@contextmanager
def temporarily_disable_phase_archive(model):
    """Disable only the supported Phase-B archive component for ablation."""
    target = model._orig_mod if hasattr(model, '_orig_mod') else model
    adapters = [getattr(block, 'phase_walk', None) for block in target.blocks]
    adapters = [a for a in adapters if a is not None]
    previous = [a.archive_scale for a in adapters]
    try:
        for a in adapters:
            a.archive_scale = 0.0
        yield
    finally:
        for a, value in zip(adapters, previous):
            a.archive_scale = value


@contextmanager
def temporarily_disable_trajectory_regression(model):
    """Ablate only the new predictive residual on identical validation windows."""
    target = model._orig_mod if hasattr(model, '_orig_mod') else model
    adapters = [getattr(block, 'traj_reg', None) for block in target.blocks]
    adapters = [a for a in adapters if a is not None]
    previous = [a.enabled for a in adapters]
    try:
        for a in adapters:
            a.enabled = False
        yield
    finally:
        for a, value in zip(adapters, previous):
            a.enabled = value


@contextmanager
def _temporarily_zero_tensors(tensors):
    tensors = [tensor for tensor in tensors if tensor is not None]
    previous = [tensor.detach().clone() for tensor in tensors]
    try:
        with torch.no_grad():
            for tensor in tensors:
                tensor.zero_()
        yield
    finally:
        with torch.no_grad():
            for tensor, value in zip(tensors, previous):
                tensor.copy_(value)


def _model_target(model):
    return model._orig_mod if hasattr(model, "_orig_mod") else model


@contextmanager
def temporarily_disable_local_features(model):
    """Zero the local-conv residual path for ablations.

    The forward gate is tanh(residual_gate) + gate_bootstrap, so zeroing only
    the learned tensor leaves bootstrap-driven leakage that understates the
    ablation. Both are cleared here and restored on exit.
    """
    target = _model_target(model)
    stages = []
    for block in target.blocks:
        for stage in ([getattr(block, "local_conv", None)]
                      + list(getattr(block, "local_conv_extra", ()))):
            if stage is not None:
                stages.append(stage)
    gates = [getattr(stage, "residual_gate", None) for stage in stages]
    gates = [gate for gate in gates if gate is not None]
    previous = [gate.detach().clone() for gate in gates]
    boots = [(stage, float(getattr(stage, "gate_bootstrap", 0.0)))
             for stage in stages]
    try:
        with torch.no_grad():
            for gate in gates:
                gate.zero_()
        for stage, _ in boots:
            stage.gate_bootstrap = 0.0
        yield
    finally:
        with torch.no_grad():
            for gate, value in zip(gates, previous):
                gate.copy_(value)
        for stage, value in boots:
            stage.gate_bootstrap = value


def temporarily_disable_lexical_semantic(model):
    adapter = getattr(_model_target(model), "lexical_semantic", None)
    return _temporarily_zero_tensors(
        [getattr(adapter, "residual_gate", None)] if adapter is not None else [])


def temporarily_disable_v41_engram(model):
    engram = getattr(_model_target(model), "v41_engram", None)
    return _temporarily_zero_tensors([getattr(engram, "gate", None)] if engram is not None else [])


@contextmanager
def temporarily_disable_v41_ced_csa2(model):
    target = _model_target(model)
    tensors = []
    ced = getattr(target, "v41_ced", None)
    if ced is not None:
        tensors.append(getattr(ced, "gate", None))
    attentions = []
    for block in target.blocks:
        attn = getattr(block, "attn", None)
        if attn is not None:
            attentions.append(attn)
            tensors.extend((getattr(attn, "csa2_gate", None),
                            getattr(attn, "csa2_alpha", None)))
    modes = [getattr(attn, "csa2_mode", "blend") for attn in attentions]
    try:
        # shared_only bypasses alpha/gate entirely; temporarily route through
        # the blend equation so zero alpha + zero learned gate is a true
        # local-K/V ablation under either saved runtime mode.
        for attn in attentions:
            attn.csa2_mode = "blend"
        with _temporarily_zero_tensors(tensors):
            yield
    finally:
        for attn, mode in zip(attentions, modes):
            attn.csa2_mode = mode


@contextmanager
def temporarily_disable_all_extensions(model):
    with temporarily_disable_v41_ced_csa2(model), temporarily_disable_v41_engram(model), \
            temporarily_disable_local_features(model), temporarily_disable_lexical_semantic(model), \
            temporarily_disable_phase_walk(model):
        yield


# --- REMOVED 2026-09-30 (lean cleanup): TrajectoryRegressionAdapter ---
# What it was: per-block predictive residual (traj_reg) with its own rank/
#   horizon/scale/gate-cost knobs and aux loss.
# Why removed: ckpt_ongoing.pt contains zero traj_reg tensors; TB_TRAJ_REG
#   defaults to 0 so Block.traj_reg is always None here. The None-guarded
#   hooks in Block/forward/loss/eval stay (they cost nothing and keep the
#   file forward-compatible). Setting TB_TRAJ_REG=1 now raises at Block
#   construction instead of building a module the checkpoint cannot load.
# To restore: recover from train_walker_1b_continue.py.bak-20260930.
def _bounded_env_float(name, default, lower, upper):
    """Read a finite floating-point runtime control with a closed bound."""
    value = float(os.environ.get(name, str(default)))
    if not math.isfinite(value) or not (lower <= value <= upper):
        raise ValueError(f"{name} must be finite and in [{lower}, {upper}]")
    return value


class _ControlledGradientIdentity(torch.autograd.Function):
    """Alias-preserving identity whose only operation is backward scaling."""
    @staticmethod
    def forward(ctx, value, rho):
        ctx.rho = float(rho)
        return value

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.mul(ctx.rho), None


def controlled_gradient_view(value, rho):
    """Identity in the forward pass, ``rho``-scaled in the backward pass.

    At rho=0 the consumer sees the exact tensor values but cannot reshape the
    producer. At rho=1 this returns the original object, preserving the legacy
    graph and avoiding extra pointwise work in the default configuration.
    """
    if value is None or rho == 1.0:
        return value
    if rho == 0.0:
        return value.detach()
    return _ControlledGradientIdentity.apply(value, rho)


def controlled_walk_inputs(x, data_emb, depth_velocity, depth_curvature, rho):
    """Apply one gradient firewall to every backbone-owned Walk input."""
    return tuple(controlled_gradient_view(value, rho) for value in (
        x, data_emb, depth_velocity, depth_curvature))


def _scale_residual(value, alpha):
    """Preserve the legacy graph when alpha is its default value."""
    return value if alpha == 1.0 else value * alpha


class V41Engram(nn.Module):
    """Small hashed conditional-memory sidecar inspired by DeepSeek Engram.

    This is intentionally an additive, zero-gated extension rather than a
    replacement for attention.  It gives repeated 2/3-gram patterns a cheap
    content-addressed lookup path while preserving the existing Walker
    checkpoint shape when disabled.  The low-rank tables keep the 1B model's
    parameter increase modest (bucket_count * rank per n-gram).
    """

    def __init__(self, D):
        super().__init__()
        buckets = max(1024, int(os.environ.get("TB_V41_ENGRAM_BUCKETS", "32768")))
        rank = max(8, min(D, int(os.environ.get("TB_V41_ENGRAM_RANK", "64"))))
        self.buckets = buckets
        self.rank = rank
        self.table2 = nn.Embedding(buckets, rank, dtype=DTYPE)
        self.table3 = nn.Embedding(buckets, rank, dtype=DTYPE)
        self.out = nn.Linear(rank, D, bias=False, dtype=DTYPE)
        self.norm = nn.LayerNorm(rank, dtype=DTYPE)
        # Exact no-op at the boundary through the gate. Keep the output
        # projection nonzero so the gate receives a useful CE gradient on the
        # first update; zeroing both would permanently deadlock this sidecar.
        nn.init.normal_(self.table2.weight, 0.0, 0.02)
        nn.init.normal_(self.table3.weight, 0.0, 0.02)
        nn.init.normal_(self.out.weight, 0.0, 0.02)
        self.gate = nn.Parameter(torch.zeros((), dtype=DTYPE))

    def forward(self, ids):
        if ids.ndim != 2:
            raise ValueError("V41Engram expects token ids with shape [B, T]")
        ids64 = ids.to(torch.long)
        # Causal rolling hashes.  The first one/two positions use a fixed
        # zero predecessor, so no future token enters the lookup.
        prev1 = F.pad(ids64[:, :-1], (1, 0))
        prev2 = F.pad(ids64[:, :-2], (2, 0))
        h2 = (prev1 * 1_000_003 + ids64 * 97_003 + 17) % self.buckets
        h3 = (prev2 * 1_000_033 + prev1 * 97_003 + ids64 * 31_051 + 23) % self.buckets
        lookup = self.table2(h2) + self.table3(h3)
        return self.out(self.norm(lookup)) * torch.tanh(self.gate)


class LoopController(nn.Module):
    """Recurrent-depth controller: shared-weight refinement passes, zero-init gated.

    Looped-transformer pattern. ALL blocks first run flat (unchanged order);
    then the [start, end) section runs `passes` extra refinement passes with
    SHARED block weights (no new block params):

        x_flat -> entry -> [pass p: section, +pass_emb[p], gate g[p]] -> exit

    Per pass p: yb = section(xr); xr = xr + g[p] * (yb - xr).
    New parameters (all exact no-ops at init, so enabling the loop cannot
    break a resumed checkpoint on step one):
      entry/exit: D->D linears, zero weights+bias (prefix/suffix adapters
        that learn the loop's input/output conditioning).
      pass_gate: scalar per pass, init 0 -> closed gate reproduces its input
        bitwise (xr + 0*(yb-xr) is exact), so gate-closed refinement passes
        leave x_flat untouched. Gates receive gradient immediately because
        dL/dg = (yb-xr) . grad, nonzero once passes diverge.
      pass_emb: per-pass D-vector, init 0, giving passes distinct identities
        to diverge during training while starting identical.
    Cost: +2*D*D params (~8.4M at D=2048, +0.8%), ~Kx loop-section FLOPs,
    one transient [B,T,D] delta buffer. Optimizer: base group (mult 1.0) via
    the `_role` fallthrough. Resume: `loop_controller.*` keys are registered
    as additive sidecars (fresh init, existing moments untouched).
    Env: TB_LOOP_ENABLE=1, TB_LOOP_START/END (python-style [start, end)),
    TB_LOOP_PASSES>=1. K may change across resumes with no interface change:
    shrinking slices pass 0..k (shared semantics), growing inits extra passes
    fresh — see _adapt_loop_state_keys; optimizer moments transplant by name.
    Incompatible with per-block aux sidecars
    (traj_reg/mem/ni/lmm) -> hard error at construction (their per-block
    losses assume one execution per block).
    """

    def __init__(self, D, passes):
        super().__init__()
        self.entry = nn.Linear(D, D, bias=True, dtype=DTYPE)
        self.exit = nn.Linear(D, D, bias=True, dtype=DTYPE)
        nn.init.zeros_(self.entry.weight)
        nn.init.zeros_(self.entry.bias)
        nn.init.zeros_(self.exit.weight)
        nn.init.zeros_(self.exit.bias)
        self.pass_gate = nn.Parameter(torch.zeros(int(passes), dtype=DTYPE))
        self.pass_emb = nn.Parameter(torch.zeros(int(passes), int(D), dtype=DTYPE))


class RAdapter(nn.Module):
    """Huginn-lite narrow-R adapter: concat[e, s] -> D refinement gate.

    Prelude output ``e`` is re-injected every extra pass (Huginn R(e, s));
    the per-pass gains start at exact zero, so enabling R cannot break a
    resumed checkpoint on step one:

        u = norm(s + proj(concat[s, e])); y = section(u)
        s' = s + pass_gain[p] * (y - s), pass_gain init 0 -> bitwise identity.

    Single gate stage keeps credit assignment two-step: pass_gain opens first
    (direct gradient), then proj/norm learn. The small proj init (not zero)
    guarantees a nonzero second-stage gradient once a gain opens; a zero proj
    would stall a further stage. (A separate adapter scalar gate would add a
    third dead stage for no modeling gain, so there is none.)

    Layout is fixed (mirrors _radapter_counts in the contract):
      proj.weight (D, 2D) no bias + LayerNorm weight/bias (D,) +
      pass_gain (r_max,).
    Cost: ~2*D*D params (~8.4M at D=2048, +0.8%), (R_width/L) FLOPs per pass.
    r_max may change across resumes: shrinking slices pass_gain leading dim,
    growing inits extras fresh — see _adapt_r_state_keys. Optimizer: base
    group via the `_role` fallthrough. TB_R_KV_SHARE is accepted and recorded
    but v1 recomputes exactly (reserved for a sharing kernel without a
    manifest break).
    Env: TB_R_ENABLE=1, TB_R_START/END, TB_R_MEAN>=0, TB_R_MAX>=1,
    TB_R_BACKPROP 0<=k<=max, TB_R_INSTR_ONLY, TB_R_KV_SHARE.
    """

    def __init__(self, D, r_max):
        super().__init__()
        self.proj = nn.Linear(2 * int(D), int(D), bias=False, dtype=DTYPE)
        # Small (not zero) projection: forward is still bitwise identical at
        # init because pass_gain=0 kills the branch, but proj/norm get a
        # nonzero gradient as soon as any gain opens.
        with torch.no_grad():
            nn.init.normal_(self.proj.weight, mean=0.0, std=0.02)
        self.norm = nn.LayerNorm(int(D), dtype=DTYPE)
        self.pass_gain = nn.Parameter(torch.zeros(int(r_max), dtype=DTYPE))


# Sources whose microbatches earn latent depth when TB_R_INSTR_ONLY=1.
# Wiki-style pretraining stays flat (r=0); instruction/SFT answers sample r.
R_INSTR_PREFIXES = ("sft_", "instr_", "instruction", "assistant", "tulu", "openassistant")


def should_apply_r(source_name, instr_only=True):
    """Batch-level conditional-r gate (MoR-style, source granularity).

    Returns True when this microbatch should sample extra R passes. With
    instr_only=False every source refines; with True (default for the
    instruction stage) only instruction/SFT-like sources refine while wiki
    stays flat. Unknown/None names default to active so eval without source
    tracking never silently disables a trained branch.
    """
    if not bool(instr_only):
        return True
    if source_name is None:
        return True
    name = str(source_name).strip().lower()
    if not name or name == "stream":
        return True
    return name.startswith(R_INSTR_PREFIXES)


def _sample_r_counts(training, mean, max_passes, backprop, rng=None, kind="uniform"):
    """Split sampled extra passes into (n_no_grad, k_with_grad).

    Training: r ~ uniform[0, min(2*mean, max)] (cheap Huginn-mean proxy;
    heavy-tail Poisson/lognormal is a follow-up without a manifest break).
    Eval: deterministic r = mean (no randomness, comparable val banks).
    k = min(r, backprop); n = r - k. r=0 -> flat path, bitwise identical.
    Uses torch RNG (checkpointed) when available, else Python random.
    """
    mean, max_passes, backprop = int(mean), int(max_passes), int(backprop)
    if max_passes < 1 or mean < 0:
        raise ValueError(f"bad R sample geometry mean={mean} max={max_passes}")
    if not training:
        r = min(max(mean, 0), max_passes)
        k = min(r, backprop)
        return (r - k, k)
    if mean <= 0:
        return (0, 0)
    if kind == "poisson_lognormal":
        # Huginn (Geiping et al. 2025): lambda ~ LogNormal(log(mean) - s^2/2,
        # s=0.5), r ~ Poisson(lambda). Mean r == mean, but a heavy tail
        # regularly shows the model depths well past it, which is what lets
        # test-time recursion extrapolate.
        import torch as _torch
        sigma = 0.5
        log_lam = math.log(mean) - 0.5 * sigma * sigma + sigma * float(_torch.randn(()).item())
        r = int(_torch.poisson(_torch.tensor(math.exp(log_lam))).item())
        r = max(0, min(r, max_passes))
        k = min(r, backprop)
        return (r - k, k)
    try:
        import torch as _torch
        draw = float(_torch.rand(()).item())
    except Exception:
        import random as _random
        draw = float(_random.random())
    hi = min(max_passes, 2 * mean)
    r = int(draw * (hi + 1))
    r = max(0, min(r, max_passes))
    k = min(r, backprop)
    return (r - k, k)


def _r_relative_resid(x_new, x_old):
    """Scale-free refinement residual ||new-old|| / ||old|| as a float.

    Measures how far one R pass moved the state, independent of activation
    scale. Callers must gate on TB_R_RESID_LOG/TB_R_HALT_EPS (it syncs) and
    run it under no_grad (it must never build a graph).
    """
    with torch.no_grad():
        num = float((x_new.detach().float()
                     - x_old.detach().float()).norm().item())
        den = float(x_old.detach().float().norm().item())
    return num / den if den > 0.0 else 0.0


def _r_should_halt(pass_index, n_nograd, resid, eps, min_prefix):
    """True when a no-grad prefix pass converged and may be skipped.

    Only prefix passes may halt: the k grad passes are the learning signal
    and always run at their absolute pass_gain indices, so halting never
    re-indexes learned gains. eps <= 0 disables halting entirely.
    """
    if not (eps > 0.0):
        return False
    if int(pass_index) >= int(n_nograd):
        return False
    if (int(pass_index) + 1) < max(0, int(min_prefix)):
        return False
    return float(resid) < float(eps)


class Block(nn.Module):
    def __init__(self, D, n_head, hybrid_w, win_heads, mult, no_walk=False, alibi=False, rope=False,
                 no_attn=False, shared_walk=None, walk_loop=1, block_index=0):
        super().__init__()
        self.ln1 = nn.LayerNorm(D, dtype=DTYPE)
        self.attn = None if no_attn else Attn(D, n_head, hybrid_w, win_heads, alibi=alibi, rope=rope)
        self.ln2 = nn.LayerNorm(D, dtype=DTYPE)
        use_compact = os.environ.get('TB_MLP_COMPACT', '0') == '1'
        self.mlp = MLPCompact(D, mult) if use_compact else MLP(D, mult)
        self.walk = shared_walk if shared_walk is not None else (None if no_walk else make_walk_adapter(D))
        self.walk_loop = walk_loop
        # Controlled-gradient residual interface. Forward scales default to the
        # historical alpha=1, while rho defaults to the historical full
        # co-adaptation path. Changing only rho is exactly forward-compatible
        # with an existing checkpoint and adds no parameters or optimizer state.
        self.walk_input_grad_scale = _bounded_env_float(
            "TB_WALK_INPUT_GRAD_SCALE", 1.0, 0.0, 1.0)
        self.phase_walk_input_grad_scale = _bounded_env_float(
            "TB_PHASE_WALK_INPUT_GRAD_SCALE", 1.0, 0.0, 1.0)
        self.walk_forward_scale = _bounded_env_float(
            "TB_WALK_FORWARD_SCALE", 1.0, 0.0, 2.0)
        self.phase_walk_forward_scale = _bounded_env_float(
            "TB_PHASE_WALK_FORWARD_SCALE", 1.0, 0.0, 2.0)
        # Phase-B is an *additional* zero-init sidecar.  Never replace or discard
        # the Phase-A Walk parameters during a phase transition.
        self.phase_walk = make_phase_b_sidecar(D, block_index)
        local_on = os.environ.get("TB_LOCAL_CONV", "0") == "1"
        local_every = max(1, int(os.environ.get("TB_LOCAL_CONV_EVERY", "1")))
        local_depth = max(1, min(3, int(os.environ.get("TB_LOCAL_CONV_DEPTH", "1"))))
        local_enabled = local_on and (block_index % local_every == 0)
        # Keep stage one under the original name so a one-stage checkpoint
        # loads its learned local filters and optimizer moments unchanged.
        self.local_conv = CausalLocalFeatureConv(D, kernels=_local_conv_stage_kernels(1)) \
            if local_enabled else None
        # Extra stages are additive parameters.  Their zero gates make the
        # depth-3 extension an exact no-op until those gates learn to open.
        self.local_conv_extra = nn.ModuleList(
            CausalLocalFeatureConv(D, kernels=_local_conv_stage_kernels(stage))
            for stage in range(2, local_depth + 1)
        ) if local_enabled else nn.ModuleList()
        if os.environ.get('TB_TRAJ_REG', '0') == '1':
            raise RuntimeError("TB_TRAJ_REG=1 needs TrajectoryRegressionAdapter (removed 2026-09-30: no checkpoint tensors; never enabled in this lineage).")
        self.traj_reg = None

    @staticmethod
    def _call_walk(w, inp, data_emb, depth_velocity, depth_curvature,
                   input_grad_scale=1.0):
        # The Phase-B sidecar IS a HierarchicalEvidenceWalkAdapter and takes
        # depth kwargs; the legacy Phase-A Walk takes none. (The standalone
        # EvidenceWalkAdapter variant was removed 2026-09-30.)
        inp, data_emb, depth_velocity, depth_curvature = controlled_walk_inputs(
            inp, data_emb, depth_velocity, depth_curvature, input_grad_scale)
        if isinstance(w, HierarchicalEvidenceWalkAdapter):
            return w(inp, depth_velocity=depth_velocity, depth_curvature=depth_curvature,
                     data_emb=data_emb if getattr(w, 'data_traj', False) else None)
        return w(inp)

    def forward(self, x, data_emb=None, depth_velocity=None, depth_curvature=None,
                mask=None, position_ids=None, position_limit=None,
                shared_k=None, shared_v=None):
        if self.local_conv is not None:
            # Local evidence is injected before RoPE attention and Walk/HIER;
            # both branches therefore see the same enriched causal state.
            x = x + self.local_conv(x)
            for extra in self.local_conv_extra:
                x = x + extra(x)
        attention_residual = torch.zeros_like(x)
        if self.attn is not None:
            attention_residual = self.attn(
                self.ln1(x), position_ids=position_ids, position_limit=position_limit,
                shared_k=shared_k, shared_v=shared_v)
            x = x + attention_residual
        x = x + self.mlp(self.ln2(x))

        if self.walk is not None:
            w = self.walk
            ckpt_walk = getattr(w, 'loop_ckpt', False) and self.training
            if not getattr(w, 'exit_on', False):
                for _ in range(self.walk_loop):
                    if ckpt_walk:
                        x = checkpoint(lambda a: a + _scale_residual(self._call_walk(
                                           w, a, data_emb, depth_velocity, depth_curvature,
                                           self.walk_input_grad_scale), self.walk_forward_scale),
                                       x, use_reentrant=False)
                    else:
                        x = x + _scale_residual(self._call_walk(
                            w, x, data_emb, depth_velocity, depth_curvature,
                            self.walk_input_grad_scale), self.walk_forward_scale)
            else:
                acc = x
                for _ in range(self.walk_loop):
                    acc = acc + _scale_residual(self._call_walk(
                          w, acc, data_emb, depth_velocity, depth_curvature,
                          self.walk_input_grad_scale), self.walk_forward_scale) * \
                          (1.0 - w._exit_scores.to(acc.dtype)).unsqueeze(-1)
                x = acc

        if self.phase_walk is not None and self.phase_walk_forward_scale != 0.0:
            # One Phase-B sidecar execution per block.  out_proj is zero-init, so
            # this is exactly x at the phase boundary while still allowing the
            # new output projection to receive CE gradient immediately.
            # TB_PHASE_WALK_FORWARD_SCALE=0 skips the sidecar entirely (it would
            # only be multiplied by zero): measured on the step-29750 1B,
            # +0.0035 CE before adaptation for 1.44x faster fwd+bwd and 7 GiB
            # less peak memory. Weights stay untouched, so 1.0 restores it.
            x = x + _scale_residual(self._call_walk(
                self.phase_walk, x, data_emb, depth_velocity, depth_curvature,
                self.phase_walk_input_grad_scale), self.phase_walk_forward_scale)
        if self.traj_reg is not None:
            x = x + self.traj_reg(x, attention_residual, data_emb, depth_velocity)
        return x


class ConceptMemory(nn.Module):
    """Conceptualised working memory — CONTENT-ADDRESSED and COMPETITIVE.

    Why the previous version collapsed to uniform: the router was an independent
    learned parameter (gate_w) whose target (write_target) was a *mean over all
    tokens* -> uniform write_target -> uniform router -> uniform write_target (a
    closed, self-referential loop). MoE-router collapse, immune to M/LR/z-loss.

    Fix (breaking the loop with externally-anchored signal):
      1. CONTENT-ADDRESSED routing (Hopfield/SOM): the read/write gate is a FUNCTION
         of the concepts themselves — g = softmax(h @ C^T / sqrt(D) / tau). Different
         tokens (different h) route to different concepts based on REAL similarity, so
         the routing is anchored to content geometry, never independently collapsible.
      2. COMPETITIVE annealed write: learnable temperature tau (log_tau) starts soft
         and anneals DOWN via an entropy loss -> routing becomes winner-take-most ->
         concepts specialize (the SOM/competitive-learning symmetry break).
      3. REPULSION between concepts: penalise positive off-diagonal cosine, forcing
         concepts apart instead of mean-collapsing to one vector.
      4. WRITE loss = reconstruction of the query's store by the content-addressed
         retrieval (store detached); gradient flows into C through BOTH the key path
         (g = softmax(h@C^T)) and the value path (mem = g@C) — a real, per-token,
         content-anchored signal rather than a global mean pull.

    Zero-init read_out => the residual INJECTION is an exact no-op at start (safe
    A/B). All ops bf16 (no fp32 residual leak)."""

    def __init__(self, D, M=None):
        super().__init__()
        self.D = D
        self.M = M if M is not None else int(os.environ.get("TB_MEM_M", "64"))
        # concept slots: content-addressed keys == values (Hopfield). Tiny init so the
        # injection (read_out=0) stays a no-op but the concepts are learnable.
        self.concepts = nn.Parameter(torch.randn(self.M, D, dtype=DTYPE) * 0.01)
        # learnable temperature (soft->competitive): tau = exp(log_tau). Starts soft.
        self.log_tau = nn.Parameter(torch.tensor(0.0, dtype=DTYPE))  # tau=1.0
        # injection out-proj (exact zero-init => no-op residual injection)
        self.read_out = nn.Linear(D, D, bias=False, dtype=DTYPE)
        nn.init.zeros_(self.read_out.weight)
        # what the query wants to remember (a projection of the current hidden state)
        self.write_v = nn.Linear(D, D, bias=False, dtype=DTYPE)
        nn.init.normal_(self.write_v.weight, 0.0, 0.02)
        # injection gate (zero-init) — scales the read_out contribution
        self.read_scale = nn.Parameter(torch.tensor(0.0, dtype=DTYPE))
        self.register_buffer("usage", torch.zeros(self.M, dtype=torch.float32))
        # Phase 2: traversal (edges). "learned" = fixed E; "attention" = derived from
        # concept similarity (recomputed each forward, self-updating).
        self.hops = max(0, int(os.environ.get("TB_MEM_HOPS", "0")))
        self.edge_mode = os.environ.get("TB_MEM_EDGE", "attention")
        if self.hops > 0:
            if self.edge_mode == "learned":
                self.edges = nn.Parameter(torch.zeros(self.M, self.M, dtype=DTYPE))
                nn.init.normal_(self.edges, 0.0, 0.01)
                self.traverse_scale = nn.Parameter(torch.tensor(0.1, dtype=DTYPE))
            else:
                self.edges = None
                self.traverse_scale = nn.Parameter(torch.tensor(0.1, dtype=DTYPE))
        else:
            self.register_parameter("edges", None)
            self.register_parameter("traverse_scale", None)
        # loss weight vs CE (the aux losses are ~O(1) normalized vs CE ~5-7; small
        # weight keeps the memory signal present without dominating CE).
        self.MEM_W = float(os.environ.get("TB_MEM_LOSS_W", "1.0"))
        # env-tunable strength of the competitive/repulsion forces (kept moderate so
        # the aux losses don't explode CE)
        self.tau_lr_w = float(os.environ.get("TB_MEM_TAU_W", "0.2"))
        self.repul_w = float(os.environ.get("TB_MEM_REPUL_W", "0.5"))

    def _routing(self, flat):
        """Content-addressed gate (COSINE similarity, Hopfield-style). 
        g = softmax(cos(flat, concepts) / tau). Cosine (not dot/sqrt(D)) gives sim in
        [-1,1] with meaningful per-concept variance — dot/sqrt(D) here crushed sim to
        ~0.01, making softmax uniform (the actual reason the memory couldn't
        differentiate). Normalising queries and concepts makes the routing genuinely
        content-addressed: different tokens -> different nearest concepts."""
        q = flat / (flat.norm(dim=-1, keepdim=True) + 1e-8)
        c = self.concepts.to(flat.dtype)
        c = c / (c.norm(dim=-1, keepdim=True) + 1e-8)
        sim = q @ c.t()                                       # (BT, M) in [-1,1]
        tau = torch.exp(self.log_tau).clamp(min=0.05, max=5.0)
        return torch.softmax(sim / tau, dim=-1)

    def _actual_edges(self, g_dtype):
        if self.edge_mode == "attention":
            cn = self.concepts.to(g_dtype)
            cn = cn / (cn.norm(dim=-1, keepdim=True) + 1e-8)
            return torch.softmax(cn @ cn.t() / math.sqrt(self.D), dim=-1)
        return self.edges.to(g_dtype)

    def _traverse(self, g, M):
        t = torch.tanh(self.traverse_scale.to(g.dtype))
        E = self._actual_edges(g.dtype)
        act = g
        for _ in range(self.hops):
            spread = torch.softmax((act @ E) / math.sqrt(M), dim=-1)
            act = (1.0 - t) * act + t * spread
        act = act / (act.sum(dim=-1, keepdim=True) + 1e-6)
        return act

    def forward(self, x):
        B, T, D = x.shape
        flat = x.to(self.read_out.weight.dtype).reshape(B * T, D)

        # ---- READ: content-addressed routing (anchored to concepts) ----
        g = self._routing(flat)                       # (BT, M) — real similarity
        if self.hops > 0:
            g = self._traverse(g, self.M)
        mem = g @ self.concepts                       # (BT, D) retrieved
        usage = g.detach().mean(dim=0)
        self.usage = 0.9 * self.usage + 0.1 * usage.to(self.usage.device)

        # ---- WRITE: content-addressed reconstruction (store detached) ----
        # NORMALIZE the store. write_v(flat) is an unbounded projection of a residual
        # stream that grows through depth (store_norm reached 96->376 across blocks),
        # which destabilised every /store.norm and /diff.mean() normalizer (bf16
        # division with a wildly-varying denominator). A unit-normalized store keeps
        # the memory signal bounded and the ratio stable, so the write teaches without
        # exploding. concepts are also soft-capped (tanh-scaled) to bound their growth.
        store = self.write_v(flat)
        store = store / (store.norm(dim=-1, keepdim=True) + 1e-6)   # unit-scale, bounded

        # ---- inject (read_out zero-init => exact no-op at start) ----
        mem_flat = self.read_out(mem)
        mem_flat = mem_flat * (1.0 + self.read_scale)
        mem_flat = mem_flat.to(x.dtype)
        out = (flat + mem_flat).view(B, T, D)

        # ---- loss: all on model device ----
        # Each aux term is NORMALIZED to ~O(1) and weighted small vs CE. The previous
        # version used raw MSE (grows as concepts specialise -> loss exploded to 24-35,
        # dwarfing CE ~5). Normalizing by scale keeps the memory signal present but
        # bounded, so it teaches without dominating CE.
        # 4a. RECONSTRUCTION (read-quality): retrieval should approximate the query's
        #     store, normalized by L2 scale so the loss is O(1).
        recon = F.mse_loss(mem, store.detach()) / (store.norm(dim=-1).mean() + 1e-6)
        recon = self.MEM_W * 0.1 * recon
        # 4b. WRITE (concept specialisation): a token pulls only its TOP-k concepts
        #     toward its store (sparse; no (BT,M,D) alloc). store NOT detached.
        k = min(4, self.M)
        topk_vals, topk_idx = torch.topk(g, k, dim=-1)         # (BT, k)
        gk = g.gather(-1, topk_idx)                             # (BT, k) gate
        ck = self.concepts[topk_idx]                            # (BT, k, D)
        sk = store.unsqueeze(1)                                 # (BT, 1, D)
        diff = (ck - sk).pow(2).mean(dim=-1)                    # (BT, k)
        pull = self.MEM_W * 0.1 * (gk * diff).mean() / (diff.detach().mean() + 1e-6)
        # 4c. competitive: anneal tau by minimizing routing entropy (winner-take-most).
        ent = -(g * (g + 1e-12).log()).sum(-1).mean()
        competitive = self.MEM_W * self.tau_lr_w * ent
        # 4d. repulsion: force concepts apart (breaks the uniform-init symmetry).
        cn = self.concepts / (self.concepts.norm(dim=-1, keepdim=True) + 1e-8)
        sim_m = (cn @ cn.t()) * (1.0 - torch.eye(self.M, device=x.device).to(cn.dtype))
        repulsion = self.MEM_W * self.repul_w * torch.clamp(sim_m.mean(), min=0.0)
        self.write_loss = (recon + pull + competitive + repulsion)
        return out



class UnifiedViewInterpreter(nn.Module):
    """One stateless causal feature generator shared by every NI objective.

    This is intentionally not an RNN and has no recurrent slots.  Each token
    receives a learned alternate view made from its projected state, causal
    multi-scale temporal features, and optional first/second differences.  The
    transformer keeps responsibility for long-range retrieval and reasoning;
    this module only offers a different local/temporal basis in which to do it.
    CE and optional utility heads jointly decide whether that basis is useful.
    """

    def __init__(self, D, rank=32, max_steps=1, trajectory=True):
        super().__init__()
        if min(D, rank, max_steps) <= 0:
            raise ValueError("NI dimensions and steps must be positive")
        self.D, self.rank, self.max_steps = int(D), int(rank), int(max_steps)
        self.trajectory = bool(trajectory)
        self.input = nn.Linear(self.D, self.rank, bias=False)
        self.short = nn.Conv1d(self.rank, self.rank, kernel_size=3, groups=self.rank, bias=False)
        self.medium = nn.Conv1d(self.rank, self.rank, kernel_size=9, groups=self.rank, bias=False)
        self.long = nn.Conv1d(self.rank, self.rank, kernel_size=33, groups=self.rank, bias=False)
        self.mix = nn.Linear(6 * self.rank, self.rank, bias=False)
        self.refine = nn.Linear(self.rank, self.rank, bias=False)
        self.out = nn.Linear(self.rank, self.D, bias=False)
        # Exact no-op at initialization: the shared objectives decide whether
        # this alternate view earns a nonzero residual into later GPT blocks.
        nn.init.zeros_(self.out.weight)

    @staticmethod
    def _causal_conv(values, conv):
        """Depth-wise Conv1d with left-only padding, hence no future leakage."""
        kernel = conv.kernel_size[0]
        values = F.pad(values.transpose(1, 2), (kernel - 1, 0))
        return conv(values).transpose(1, 2)

    def forward(self, hidden, steps=1):
        if hidden.ndim != 3 or hidden.size(-1) != self.D:
            raise ValueError("NI hidden state must have shape [batch, tokens, D]")
        if not 1 <= steps <= self.max_steps:
            raise ValueError(f"NI steps must be in [1, {self.max_steps}]")
        base = self.input(hidden)
        velocity = F.pad(base[:, 1:] - base[:, :-1], (0, 0, 1, 0))
        acceleration = F.pad(velocity[:, 1:] - velocity[:, :-1], (0, 0, 1, 0))
        if not self.trajectory:
            velocity, acceleration = torch.zeros_like(base), torch.zeros_like(base)
        features = torch.cat((base, self._causal_conv(base, self.short),
                              self._causal_conv(base, self.medium), self._causal_conv(base, self.long),
                              velocity, acceleration), dim=-1)
        view = F.gelu(self.mix(features))
        # K repeats are token-local feed-forward refinement, never a carried
        # state from an earlier token or chunk.
        for _ in range(steps - 1):
            view = view + 0.10 * F.gelu(self.refine(view))
        return self.out(view), view.mean(dim=1)


class EvidenceInterpreter(nn.Module):
    """Stateless causal evidence workspace queried from several LLM depths.

    Text is represented by completed causal spans at several resolutions.  An
    earlier token can attend only to spans whose final token it has already
    seen.  Optional ``image`` and ``audio`` entries are pre-encoded feature
    sequences available at input time; they receive modality and ordinal
    position tags, mix with each other once, and remain immutable thereafter.

    There are no RNNs, recurrent slots, or objective-specific filter/editor
    paths.  The same evidence bank is queried using the current LLM state plus
    its change across depth.  Later blocks decide how to use the returned
    residual.
    """

    def __init__(self, D, rank=256, max_steps=1, trajectory=True,
                 span_scales=(4, 16, 64), max_nodes_per_scale=256,
                 max_static_nodes=512):
        super().__init__()
        if min(D, rank, max_steps, max_nodes_per_scale, max_static_nodes) <= 0:
            raise ValueError("evidence NI dimensions and limits must be positive")
        self.D, self.rank, self.max_steps = int(D), int(rank), int(max_steps)
        self.trajectory = bool(trajectory)
        self.span_scales = tuple(int(scale) for scale in span_scales)
        if not self.span_scales or min(self.span_scales) <= 0:
            raise ValueError("evidence span scales must be positive")
        self.max_nodes_per_scale = int(max_nodes_per_scale)
        self.max_static_nodes = int(max_static_nodes)
        self.text_in = nn.Linear(D, rank, bias=False)
        self.image_in = nn.Linear(D, rank, bias=False)
        self.audio_in = nn.Linear(D, rank, bias=False)
        self.query = nn.Linear(3 * D, rank, bias=False)
        self.key = nn.Linear(rank, rank, bias=False)
        self.value = nn.Linear(rank, rank, bias=False)
        self.static_q = nn.Linear(rank, rank, bias=False)
        self.static_k = nn.Linear(rank, rank, bias=False)
        self.static_v = nn.Linear(rank, rank, bias=False)
        self.static_out = nn.Linear(rank, rank, bias=False)
        self.mix = nn.Linear(3 * D + rank, rank, bias=False)
        self.refine = nn.Linear(rank, rank, bias=False)
        self.out = nn.Linear(rank, D, bias=False)
        self.modality = nn.Embedding(3, rank)
        self.ordinal = nn.Embedding(max_static_nodes, rank)
        self.null_evidence = nn.Parameter(torch.empty(1, 1, rank))
        self.static_seed = nn.Parameter(torch.empty(1, 1, rank))
        self.scale = rank ** -0.5
        nn.init.normal_(self.null_evidence, std=0.02)
        nn.init.normal_(self.static_seed, std=0.02)
        nn.init.normal_(self.modality.weight, std=0.02)
        nn.init.normal_(self.ordinal.weight, std=0.02)
        # Exact no-op at initialization keeps weights-only transfer stable.
        nn.init.zeros_(self.out.weight)

    def _text_spans(self, hidden):
        """Return completed causal text span nodes and their inclusive ends."""
        B, T, D = hidden.shape
        nodes, ends = [], []
        for base_scale in self.span_scales:
            scale = base_scale * max(1, math.ceil(T / (base_scale * self.max_nodes_per_scale)))
            count = (T + scale - 1) // scale
            padded = count * scale
            valid = torch.arange(padded, device=hidden.device).view(1, padded) < T
            h_pad = F.pad(hidden, (0, 0, 0, padded - T)).view(B, count, scale, D)
            valid = valid.view(1, count, scale, 1).to(hidden.dtype)
            pooled = (h_pad * valid).sum(dim=2) / valid.sum(dim=2).clamp_min(1)
            nodes.append(self.text_in(pooled))
            ends.append((torch.arange(count, device=hidden.device) + 1).mul(scale).sub(1).clamp_max(T - 1))
        return torch.cat(nodes, dim=1), torch.cat(ends, dim=0)

    def _static_bank(self, evidence, B, dtype, device):
        """Project optional image/audio features and mix them without text leakage."""
        if evidence is None:
            return None, None
        if not isinstance(evidence, dict):
            raise ValueError("evidence must be a dict containing optional image/audio tensors")
        pieces, masks = [], []
        for name, adapter, modality_id in (("image", self.image_in, 1), ("audio", self.audio_in, 2)):
            values = evidence.get(name)
            if values is None:
                continue
            if values.ndim != 3 or values.size(0) != B or values.size(-1) != self.D:
                raise ValueError(f"evidence[{name!r}] must have shape [batch, items, {self.D}]")
            if values.size(1) > self.max_static_nodes:
                raise ValueError(f"evidence[{name!r}] has {values.size(1)} items; max is {self.max_static_nodes}")
            value_mask = evidence.get(f"{name}_mask")
            if value_mask is None:
                value_mask = torch.ones(B, values.size(1), dtype=torch.bool, device=device)
            if value_mask.shape != values.shape[:2]:
                raise ValueError(f"evidence[{name}_mask] must match [batch, items]")
            value_mask = value_mask.to(device=device, dtype=torch.bool)
            positions = torch.arange(values.size(1), device=device)
            tags = self.modality.weight[modality_id].view(1, 1, -1) + self.ordinal(positions).unsqueeze(0)
            pieces.append(adapter(values.to(device=device, dtype=dtype)) + tags.to(dtype))
            masks.append(value_mask)
        if not pieces:
            return None, None
        static = torch.cat(pieces, dim=1)
        valid = torch.cat(masks, dim=1)
        # This stateless, per-forward readout gives a text query direct access
        # to a joint image/audio view.  It is not a persistent slot: it is
        # recreated from the immutable input evidence on every forward call.
        static = torch.cat((self.static_seed.expand(B, -1, -1).to(dtype), static), dim=1)
        valid = torch.cat((torch.ones(B, 1, dtype=torch.bool, device=device), valid), dim=1)
        # Image/audio can exchange information because both are supplied before
        # decoding.  Invalid positions neither read nor write the mixer, while
        # the always-valid readout makes an all-masked example well defined.
        scores = self.static_q(static) @ self.static_k(static).transpose(-2, -1)
        scores = scores.float().mul(self.scale).masked_fill(~valid.unsqueeze(1), float("-inf"))
        mixed = F.softmax(scores, dim=-1).to(dtype) @ self.static_v(static)
        static = static + self.static_out(mixed) * valid.unsqueeze(-1).to(dtype)
        return static, valid

    def forward(self, hidden, depth_velocity=None, depth_curvature=None, evidence=None, steps=1):
        if hidden.ndim != 3 or hidden.size(-1) != self.D:
            raise ValueError("evidence NI hidden state must have shape [batch, tokens, D]")
        if not 1 <= steps <= self.max_steps:
            raise ValueError(f"NI steps must be in [1, {self.max_steps}]")
        B, T, _ = hidden.shape
        velocity = torch.zeros_like(hidden) if depth_velocity is None else depth_velocity
        curvature = torch.zeros_like(hidden) if depth_curvature is None else depth_curvature
        if velocity.shape != hidden.shape or curvature.shape != hidden.shape:
            raise ValueError("depth trajectory tensors must match hidden shape")
        if not self.trajectory:
            velocity, curvature = torch.zeros_like(hidden), torch.zeros_like(hidden)
        text, text_ends = self._text_spans(hidden)
        static, static_valid = self._static_bank(evidence, B, hidden.dtype, hidden.device)
        null = self.null_evidence.expand(B, -1, -1).to(hidden.dtype)
        memory = torch.cat((null, text) if static is None else (null, text, static), dim=1)
        endpoints = torch.cat((torch.full((1,), -1, device=hidden.device, dtype=torch.long), text_ends,
                               torch.full((0 if static is None else static.size(1),), -1, device=hidden.device, dtype=torch.long)))
        memory_valid = torch.ones(B, memory.size(1), dtype=torch.bool, device=hidden.device)
        if static_valid is not None:
            memory_valid[:, 1 + text.size(1):] = static_valid
        position = torch.arange(T, device=hidden.device).view(1, T, 1)
        causal = endpoints.view(1, 1, -1) <= position
        allowed = causal & memory_valid.unsqueeze(1)
        query = self.query(torch.cat((hidden, velocity, curvature), dim=-1))
        scores = query @ self.key(memory).transpose(-2, -1)
        scores = scores.float().mul(self.scale).masked_fill(~allowed, float("-inf"))
        retrieved = F.softmax(scores, dim=-1).to(hidden.dtype) @ self.value(memory)
        view = F.gelu(self.mix(torch.cat((hidden, velocity, curvature, retrieved), dim=-1)))
        for _ in range(steps - 1):
            view = view + 0.10 * F.gelu(self.refine(view))
        return self.out(view), retrieved.mean(dim=1)


def _ni_local_decoder_gradient_bridge(update, normalized_hidden, targets, head):
    """Align an NI update with the local decoder-head CE descent direction.

    This is a training-only first-order teacher, not a hidden backpropagation
    through future transformer blocks. At an NI tap, the existing tied decoder
    geometry says which small change to the locally normalized state would
    reduce its own next-token CE. The detached direction is intentionally a
    weak auxiliary signal; ordinary end-to-end CE remains the only authority
    on whether the earlier update helps after later decoder layers.
    """
    if update.shape != normalized_hidden.shape or targets.shape != update.shape[:2]:
        raise ValueError("NI local gradient bridge shapes must match [batch, tokens, D]")
    logits = head(normalized_hidden).float()
    probabilities = F.softmax(logits, dim=-1)
    target_index = targets.unsqueeze(-1)
    probabilities = probabilities.scatter_add(
        -1, target_index, torch.full_like(probabilities[..., :1], -1.0))
    gradient = probabilities @ head.weight.float()
    # Normalize per token by the teacher magnitude rather than the update's
    # magnitude: the zero-initialized NI output then receives a finite first
    # learning signal instead of an undefined cosine direction.
    scale = gradient.detach().norm(dim=-1, keepdim=True).clamp_min(1e-3)
    return F.smooth_l1_loss(update.float() / scale, -gradient.detach() / scale)


class AverageInitializedKMerge(nn.Module):
    """Context-free K-token prefix compressor for a continuation-only ablation.

    The zero-initialized residual makes the initial representation exactly the
    mean of the K embeddings.  This module is intentionally input-side only;
    the normal uncompressed continuation remains available for causal CE.
    """
    # Local smoke/benchmark (RTX 4050, D=256, L=4, T=512, train fwd+bwd):
    # baseline 13,196 raw tok/s vs K=2/prefix=.75 8,655 raw tok/s, peak
    # memory 0.06 vs 0.06 GiB.  This tiny model is encoder/MLP dominated, so
    # it is a correctness result, not a prediction for the 1B cloud run.
    def __init__(self, D, K):
        super().__init__()
        self.D, self.K = int(D), int(K)
        hidden = max(128, D // 4)
        self.net = nn.Sequential(
            nn.Linear(self.K * self.D, hidden, bias=True, dtype=DTYPE),
            nn.GELU(),
            nn.Linear(hidden, self.D, bias=True, dtype=DTYPE),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, embeddings):
        # embeddings: [B, ranges, K, D]
        mean = embeddings.mean(dim=2)
        residual = self.net(embeddings.reshape(embeddings.shape[0], embeddings.shape[1], -1))
        return mean + residual


class PersistentRangeRepresentationMemory:
    """Bounded exact-range cache for inference-time K-merge representations.

    Keys include the token range and the caller's representation version.  The
    cache stores detached tensors, so it never retains an autograd graph.  It
    is deliberately disabled during training by WalkGPT unless
    ``TB_PRRM_TRAIN=1`` is explicitly requested; stale cached features are not
    safe while the encoder is changing.
    """
    def __init__(self, max_items=4096):
        self.max_items = max(1, int(max_items))
        self.entries = collections.OrderedDict()
        self.hits = 0
        self.misses = 0

    def clear(self):
        self.entries.clear()
        self.hits = 0
        self.misses = 0

    def get(self, key):
        value = self.entries.get(key)
        if value is None:
            self.misses += 1
            return None
        self.hits += 1
        self.entries.move_to_end(key)
        return value

    def put(self, key, value):
        self.entries[key] = value.detach()
        self.entries.move_to_end(key)
        while len(self.entries) > self.max_items:
            self.entries.popitem(last=False)

    @property
    def hit_rate(self):
        total = self.hits + self.misses
        return float(self.hits) / total if total else 0.0


class LessonMemoryModule(nn.Module):
    """Differentiable fast-weight lesson memory, deliberately separate from NI.

    A lesson may use a dedicated learned lookup (the legacy/default path) or a
    detached projection of the frozen decoder token embeddings.  The latter is
    useful for few-shot natural lessons: rare words then begin in the LM's
    lexical geometry instead of in unrelated random rows.  A functional
    (caller-owned) matrix stores key/value bindings;
    it is not a model buffer and is therefore never accidentally shared across
    examples, users, checkpoints, or ordinary pretraining batches.  The write
    operation is differentiable, so a behavioral meta-objective can train the
    encoder, query, gate, and write rate through one update.
    """

    def __init__(self, vocab, D, rank=128, n_layers=1, write_rule="delta",
                 eta=0.25, decay=0.01, gate_bias=-4.0, lesson_source="lookup"):
        super().__init__()
        if min(int(vocab), int(D), int(rank), int(n_layers)) <= 0:
            raise ValueError("LMM vocab, D, rank, and layer count must be positive")
        if write_rule not in ("delta", "hebb"):
            raise ValueError("TB_LMM_WRITE must be 'delta' or 'hebb'")
        if not (0.0 < float(eta) < 1.0):
            raise ValueError("TB_LMM_ETA must be in (0, 1)")
        if not (0.0 <= float(decay) < 1.0):
            raise ValueError("TB_LMM_DECAY must be in [0, 1)")
        lesson_source = str(lesson_source).strip().lower()
        if lesson_source not in ("lookup", "backbone"):
            raise ValueError("TB_LMM_LESSON_SOURCE must be 'lookup' or 'backbone'")
        self.vocab, self.D, self.rank = int(vocab), int(D), int(rank)
        self.write_rule, self.base_decay = str(write_rule), float(decay)
        self.lesson_source = lesson_source
        # Keep the lookup state for sidecar compatibility even when the opt-in
        # backbone path is selected. It is intentionally unused in that path;
        # the compact LMM projection below is what learns how to read the
        # frozen decoder's already-trained lexical geometry.
        self.lesson_embedding = nn.Embedding(self.vocab, self.rank, dtype=DTYPE)
        self.lesson_backbone = (nn.Linear(self.D, self.rank, bias=False, dtype=DTYPE)
                                if self.lesson_source == "backbone" else None)
        self.lesson_norm = nn.LayerNorm(self.rank, dtype=DTYPE)
        self.lesson_in = nn.Linear(self.rank, self.rank, bias=False, dtype=DTYPE)
        self.lesson_refine = nn.Linear(self.rank, self.rank, bias=False, dtype=DTYPE)
        self.key = nn.Linear(self.rank, self.rank, bias=False, dtype=DTYPE)
        self.value = nn.Linear(self.rank, self.rank, bias=False, dtype=DTYPE)
        # Direction carries lesson content while a learned positive amplitude
        # keeps the first differentiable write numerically alive. Without this
        # normalization, the chained 0.02-scale projections make a fresh
        # fast-weight value ~1e-3 in norm and its meta-gradients vanish under
        # mixed precision before the gate can learn to open.
        self.value_amplitude_log = nn.Parameter(
            torch.tensor(math.log(math.expm1(0.20)), dtype=DTYPE))
        self.query_norm = nn.LayerNorm(self.D, dtype=DTYPE)
        self.query = nn.Linear(self.D, self.rank, bias=False, dtype=DTYPE)
        self.layer_tag = nn.Embedding(int(n_layers), self.rank, dtype=DTYPE)
        self.read_out = nn.Linear(self.rank, self.D, bias=False, dtype=DTYPE)
        self.gate = nn.Linear(self.D + self.rank, 1, bias=True, dtype=DTYPE)
        eta_logit = math.log(float(eta) / (1.0 - float(eta)))
        self.write_rate_logit = nn.Parameter(torch.tensor(eta_logit, dtype=DTYPE))
        # A small negative bias prevents an untrained supplied lesson from
        # overwhelming the backbone, while still leaving all memory parameters
        # with a nonzero gradient from the first meta-training update.
        with torch.no_grad():
            init_parameters = [self.lesson_embedding.weight, self.lesson_in.weight,
                              self.lesson_refine.weight, self.key.weight, self.value.weight,
                              self.query.weight, self.layer_tag.weight, self.read_out.weight]
            if self.lesson_backbone is not None:
                init_parameters.append(self.lesson_backbone.weight)
            for parameter in init_parameters:
                nn.init.normal_(parameter, mean=0.0, std=0.02)
            nn.init.normal_(self.gate.weight, mean=0.0, std=0.01)
            self.gate.bias.fill_(float(gate_bias))

    def _pool_lesson(self, ids, mask=None, token_embedding=None):
        if ids.ndim != 2:
            raise ValueError("lesson ids must have shape [batch, lesson_tokens]")
        if ids.dtype not in (torch.int32, torch.int64, torch.long):
            raise ValueError("lesson ids must be integer token ids")
        if mask is None:
            mask = torch.ones_like(ids, dtype=torch.bool)
        if mask.shape != ids.shape:
            raise ValueError("lesson mask must match lesson ids")
        if self.lesson_source == "lookup":
            values = self.lesson_embedding(ids)
        else:
            if token_embedding is None:
                raise ValueError("LMM backbone lesson source requires a decoder token embedding")
            if token_embedding.ndim != 2 or tuple(token_embedding.shape) != (self.vocab, self.D):
                raise ValueError(
                    f"LMM backbone token embedding must have shape [{self.vocab}, {self.D}]")
            # The base LM is deliberately frozen during sidecar training.
            # Detaching here makes that invariant explicit even if a caller
            # accidentally enables gradients on the decoder embedding.
            source_values = F.embedding(ids, token_embedding.detach())
            values = self.lesson_backbone(source_values.to(self.lesson_backbone.weight.dtype))
        weights = mask.unsqueeze(-1).to(values.dtype)
        return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)

    def encode_lesson(self, lesson_ids, lesson_mask=None, bad_lesson_ids=None,
                      bad_lesson_mask=None, token_embedding=None):
        """Encode a lesson or an explicit (bad, good) behavioral contrast."""
        pooled = self._pool_lesson(lesson_ids, lesson_mask, token_embedding)
        if bad_lesson_ids is not None:
            if bad_lesson_ids.shape[0] != pooled.shape[0]:
                raise ValueError("bad and good lesson batches must have the same size")
            pooled = pooled - self._pool_lesson(bad_lesson_ids, bad_lesson_mask, token_embedding)
        hidden = F.silu(self.lesson_in(self.lesson_norm(pooled)))
        hidden = hidden + 0.10 * F.silu(self.lesson_refine(hidden))
        key = F.normalize(self.key(hidden).float(), dim=-1).to(hidden.dtype)
        raw_value = self.value(hidden)
        value_amplitude = F.softplus(self.value_amplitude_log).to(raw_value.dtype)
        value = F.normalize(raw_value.float(), dim=-1).to(raw_value.dtype) * value_amplitude
        return key, value

    def empty_memory(self, batch_size, *, device=None, dtype=None):
        if int(batch_size) <= 0:
            raise ValueError("LMM batch_size must be positive")
        return torch.zeros(int(batch_size), self.rank, self.rank,
                           device=device if device is not None else self.write_rate_logit.device,
                           dtype=dtype if dtype is not None else self.write_rate_logit.dtype)

    def _memory_for_batch(self, memory, batch_size, device, dtype):
        if memory is None:
            return self.empty_memory(batch_size, device=device, dtype=dtype)
        if memory.ndim == 2:
            memory = memory.unsqueeze(0)
        if memory.ndim != 3 or memory.shape[1:] != (self.rank, self.rank):
            raise ValueError(f"LMM memory must have shape [batch, {self.rank}, {self.rank}]")
        if memory.size(0) == 1 and batch_size != 1:
            memory = memory.expand(batch_size, -1, -1)
        if memory.size(0) != batch_size:
            raise ValueError("LMM memory batch must be one or match the decoder batch")
        return memory.to(device=device, dtype=dtype)

    def write(self, memory, key, value, *, eta=None, decay=None):
        """Functional delta/Hebbian write; never mutates the supplied memory."""
        if key.ndim != 2 or value.shape != key.shape or key.size(-1) != self.rank:
            raise ValueError("LMM key/value must both have shape [batch, rank]")
        state = self._memory_for_batch(memory, key.size(0), key.device, key.dtype)
        retrieved = torch.einsum("bkv,bk->bv", state, key)
        residual = value - retrieved if self.write_rule == "delta" else value
        # High novelty retains the old matrix; redundant writes decay it more.
        surprise = 1.0 - torch.exp(-residual.float().square().mean(dim=-1))
        base_decay = self.base_decay if decay is None else float(decay)
        if not (0.0 <= base_decay < 1.0):
            raise ValueError("LMM decay must be in [0, 1)")
        alpha = (base_decay * (1.0 - surprise)).to(state.dtype)
        if eta is None:
            eta_t = torch.sigmoid(self.write_rate_logit).to(state.dtype)
        else:
            eta_t = torch.as_tensor(eta, dtype=state.dtype, device=state.device)
            if not bool(torch.all((eta_t > 0) & (eta_t < 1))):
                raise ValueError("LMM eta must be in (0, 1)")
        update = torch.einsum("bk,bv->bkv", key, residual)
        next_state = state * (1.0 - alpha).view(-1, 1, 1) + eta_t * update
        return next_state, {
            "write_eta": eta_t.detach(),
            "surprise": surprise.detach(),
            "retrieval_rms": retrieved.detach().float().square().mean().sqrt(),
        }

    def write_lesson(self, lesson_ids, lesson_mask=None, *, bad_lesson_ids=None,
                     bad_lesson_mask=None, memory=None, eta=None, decay=None,
                     token_embedding=None):
        key, value = self.encode_lesson(lesson_ids, lesson_mask, bad_lesson_ids, bad_lesson_mask,
                                        token_embedding)
        state, stats = self.write(memory, key, value, eta=eta, decay=decay)
        stats.update({"key": key.detach(), "value": value.detach()})
        return state, stats

    def inject(self, hidden, memory, layer_index):
        if hidden.ndim != 3 or hidden.size(-1) != self.D:
            raise ValueError("LMM hidden state must have shape [batch, tokens, D]")
        state = self._memory_for_batch(memory, hidden.size(0), hidden.device, hidden.dtype)
        tag = self.layer_tag.weight[int(layer_index)].to(hidden.dtype).view(1, 1, -1)
        query = F.normalize((self.query(self.query_norm(hidden)) + tag).float(), dim=-1).to(hidden.dtype)
        retrieved = torch.einsum("bkv,btk->btv", state, query)
        candidate = self.read_out(retrieved)
        gate = torch.sigmoid(self.gate(torch.cat((hidden, retrieved), dim=-1)))
        update = candidate * gate
        return update, {
            "gate_mean": gate.detach().float().mean(),
            "retrieval_rms": retrieved.detach().float().square().mean().sqrt(),
        }


class LexicalSemanticAdapter(nn.Module):
    """Final semantic counterpart to the temporal Walk/CNN stack.

    Error slicing of the step-11,500 checkpoint (2026-09-10, unseen 2025
    FineWeb rows) found CE 1.21 for <=8-token repeats, 1.87 for 65--512-token
    repeats, but 3.63 for novel tokens and 5.94 for held-out singletons. This
    branch therefore learns multiplicative *channel* interactions and predicts
    the next token's input-embedding geometry; it performs no temporal scan.

    Small FineWeb A/B record (D=96/L=3, 600 updates): semantic-target branch
    changed held-out CE by -0.0130 nat at seed 17 and +0.0065 at seed 29
    (mean -0.00325). This is promising only as a targeted experiment, not yet
    a proven 1B improvement. A plain SwiGLU sidecar without the lexical target
    was rejected (+0.0104/-0.0013 at 240 updates).

    The FP32 residual gate starts at zero, so enabling it on an old checkpoint
    is an exact functional no-op. Its nonzero candidate gives the gate a first-
    step CE gradient, while sampled multi-positive contrastive supervision can
    organize the candidate before the residual opens.
    """
    def __init__(self, D, rank=256, gate_init=0.0, aux_weight=0.05,
                 sample_stride=32, temperature=0.1):
        super().__init__()
        self.D = int(D)
        self.rank = max(4, min(self.D, int(rank)))
        self.aux_weight = max(0.0, float(aux_weight))
        self.sample_stride = max(1, int(sample_stride))
        self.temperature = float(temperature)
        self.runtime_scale = float(os.environ.get("TB_LEXICAL_SEMANTIC_SCALE", "1.0"))
        if not math.isfinite(self.runtime_scale) or not (0.0 <= self.runtime_scale <= 2.0):
            raise ValueError("TB_LEXICAL_SEMANTIC_SCALE must be in [0, 2]")
        if not math.isfinite(self.temperature) or self.temperature <= 0.0:
            raise ValueError("TB_LEXICAL_SEMANTIC_TEMPERATURE must be finite and > 0")
        if not math.isfinite(float(gate_init)) or abs(float(gate_init)) >= 0.95:
            raise ValueError("TB_LEXICAL_SEMANTIC_GATE_INIT must have abs(value) < 0.95")
        self.norm = nn.LayerNorm(self.D, dtype=DTYPE)
        self.up = nn.Linear(self.D, self.rank, bias=False, dtype=DTYPE)
        self.value = nn.Linear(self.D, self.rank, bias=False, dtype=DTYPE)
        self.down = nn.Linear(self.rank, self.D, bias=False, dtype=DTYPE)
        self.route = nn.Linear(self.D, 1, dtype=DTYPE)
        self.residual_gate = nn.Parameter(torch.tensor(
            math.atanh(float(gate_init)) if gate_init else 0.0,
            dtype=torch.float32))
        lexical_dims = torch.linspace(0, self.D - 1, self.rank).round().long()
        self.register_buffer("lexical_dims", lexical_dims, persistent=True)
        with torch.no_grad():
            nn.init.normal_(self.down.weight, mean=0.0, std=0.005)
            nn.init.zeros_(self.route.weight)
            nn.init.zeros_(self.route.bias)
        self.gate_mean = torch.tensor(0.0)
        self.aux_loss = torch.tensor(0.0)

    @staticmethod
    def _multi_positive_contrastive(query, key, token_ids, valid, temperature):
        scores = query @ key.t() / temperature
        pair_valid = valid[:, None] & valid[None, :]
        positives = token_ids[:, None].eq(token_ids[None, :]) & pair_valid
        neg_large = -1e9
        row_scores = scores.masked_fill(~valid[None, :], neg_large)
        row = torch.logsumexp(row_scores, dim=1) - torch.logsumexp(
            row_scores.masked_fill(~positives, neg_large), dim=1)
        col_scores = scores.t().masked_fill(~valid[None, :], neg_large)
        col = torch.logsumexp(col_scores, dim=1) - torch.logsumexp(
            col_scores.masked_fill(~positives.t(), neg_large), dim=1)
        weights = valid.to(row.dtype)
        denom = weights.sum().clamp_min(1.0)
        return 0.5 * ((row * weights).sum() + (col * weights).sum()) / denom

    def forward(self, x, targets=None, embedding=None):
        z = self.norm(x)
        features = F.silu(self.up(z)) * self.value(z)
        candidate = self.down(features)
        route = torch.sigmoid(self.route(z).float()).to(x.dtype)
        gate = torch.tanh(self.residual_gate).to(x.dtype)
        residual = gate * route * candidate * self.runtime_scale
        self.gate_mean = (gate.detach().float() * route.detach().float()).mean() * self.runtime_scale
        self.aux_loss = x.new_zeros((), dtype=torch.float32)
        if (targets is not None and embedding is not None and self.aux_weight > 0.0):
            positions = torch.arange(0, targets.size(1), self.sample_stride,
                                     device=targets.device)
            sampled_ids = targets.index_select(1, positions).reshape(-1)
            valid = sampled_ids.ge(0)
            # Clamp masked K-merge targets only for the gather; zero weights
            # remove them from the objective without a GPU->CPU `.item()` sync.
            safe_ids = sampled_ids.clamp_min(0)
            sampled_features = features.index_select(1, positions).reshape(-1, self.rank)
            target_vectors = embedding(safe_ids).detach().index_select(
                -1, self.lexical_dims.to(sampled_ids.device))
            query = F.normalize(sampled_features.float(), dim=-1)
            key = F.normalize(target_vectors.float(), dim=-1)
            self.aux_loss = self._multi_positive_contrastive(
                query, key, sampled_ids, valid, self.temperature) * self.aux_weight
        return residual, self.aux_loss


def insert_pause_pairs(x, targets, data_emb, pause_vecs, every,
                       single_group=False):
    """Splice K learned pause pairs into a packed (input, target) stream.

    After every ``every`` pairs insert K ``(pause_vec, -100)`` pairs; a
    boundary at exactly T is skipped (no future token would benefit from
    the extra compute). With ``single_group`` only the first boundary is
    used (one thought group per window: the Huginn-schedule layout).
    Pause slots carry -100, so they add forward compute but never loss
    and are never emitted (the schedule is deterministic, not sampled).
    The pre-pause pair keeps its real target: causally, pauses only
    enrich what follows. ``data_emb`` (the walk trajectory reference)
    grows the same rows so depth features stay aligned. Any of
    ``targets``/``data_emb`` may be None (inference forwards) and is
    returned as None.
    """
    K = int(pause_vecs.shape[0])
    every = int(every)
    if K <= 0 or every <= 0:
        return x, targets, data_emb
    B, T, D = x.shape
    bounds = list(range(every, int(T), every))  # < T: skip trailing
    if single_group:
        bounds = bounds[:1]
    if not bounds:
        return x, targets, data_emb
    pv = pause_vecs.to(x.dtype).unsqueeze(0).expand(B, -1, -1)
    neg = None
    if targets is not None:
        neg = torch.full((B, K), -100, dtype=targets.dtype,
                         device=targets.device)
    x_parts, d_parts, t_parts = [], [], []
    prev = 0
    for bnd in bounds + [int(T)]:
        x_parts.append(x[:, prev:bnd])
        if data_emb is not None:
            d_parts.append(data_emb[:, prev:bnd])
        if targets is not None:
            t_parts.append(targets[:, prev:bnd])
        prev = bnd
        if bnd < T:
            x_parts.append(pv)
            if data_emb is not None:
                d_parts.append(pv)
            if targets is not None:
                t_parts.append(neg)
    x2 = torch.cat(x_parts, dim=1)
    return (x2,
            torch.cat(t_parts, dim=1) if targets is not None else None,
            torch.cat(d_parts, dim=1) if data_emb is not None else None)


# Measured Huginn depth-recurrence schedule (tomg-group-umd/huginn-0125,
# PrefixGraph/table.json): gains decay 1.41 -> 0.09, direction cosine
# 0.00 -> 0.99, all dense-form; live range k <= 16. Normalized below.
HUGINN_GAINS = (1.0, 0.6084, 0.5585, 0.4466, 0.2399)


def pause_layout(T, K, every):
    """Deterministic pause geometry: single group of K slots after the
    first ``every`` pairs (a trailing boundary at exactly T is skipped).
    Returns (mask, slot) over the extended length: mask is True at pause
    slots, slot holds the thought index 0..K-1 there (0 elsewhere)."""
    T, K, every = int(T), int(K), int(every)
    if K <= 0 or every <= 0 or every >= T:
        return None, None
    Tn = T + K
    mask = torch.zeros(Tn, dtype=torch.bool)
    slot = torch.zeros(Tn, dtype=torch.long)
    mask[every:every + K] = True
    slot[every:every + K] = torch.arange(K)
    return mask, slot


class WalkGPT(nn.Module):
    def __init__(self, vocab, D, L, n_head, mult=2.5, hybrid_w=512, win_heads=None, ckpt=False, no_walk=False, max_t=8192, alibi=False, rope=False, no_mem=False,
                 ni=False, ni_steps=1, ni_trajectory=True, ni_slots=4, ni_rank=32,
                 ni_width_mult=1,
                 ni_bridge=False, ni_bridge_weight=0.01, ni_mode="final", ni_chunk_size=128,
                 ni_layer=-1, ni_layers="", lmm=False, lmm_rank=128, lmm_layers="",
                 lmm_write="delta", lmm_eta=0.25, lmm_decay=0.01, lmm_gate_bias=-4.0,
                 lmm_lesson_source="lookup"):
        super().__init__()
        win_heads = win_heads if win_heads is not None else max(1, n_head // 2)
        self.alibi = alibi
        self.rope = rope
        if alibi and rope:
            raise ValueError("ALiBi and RoPE are mutually exclusive")
        self.emb = nn.Embedding(vocab, D, dtype=DTYPE)
        self.pos_emb = nn.Parameter(torch.zeros(max_t, D, dtype=DTYPE))
        self.max_t = max_t
        self.v41_engram = (V41Engram(D)
                           if os.environ.get("TB_V41_ENGRAM", "0") == "1"
                           else None)
        v41_ced_on = os.environ.get("TB_V41_CED", "0") == "1"
        v41_csa2_on = os.environ.get("TB_V41_CSA2", "0") == "1"
        self.v41_ced = V41CausalEncoder(D) if v41_ced_on else None
        self.v41_shared_kv = V41SharedKV(D) if v41_csa2_on else None
        # TB_ATTN_EVERY>1: every K-th block drops attention (trajectory-only).
        # Blocks 0, K, 2K, ... keep attention; others skip it.
        attn_every = max(1, int(os.environ.get("TB_ATTN_EVERY", "1")))
        # TB_WALK_SHARED>0: one WalkAdapter shared by ALL blocks (depth-sharing);
        # TB_WALK_LOOP: inner walk iterations per block (fixed if >1, or "schedule"
        #  for the loop-count curriculum that ramps/randomizes with training).
        walk_shared = not no_walk and int(os.environ.get("TB_WALK_SHARED", "0")) == 1
        walk_loop_env = os.environ.get("TB_WALK_LOOP", "1")
        self.walk_loop_sched = (walk_loop_env if walk_loop_env in ("schedule", "random")
                                else max(1, int(walk_loop_env)))
        walk_loop = max(1, self.walk_loop_sched if isinstance(self.walk_loop_sched, int) else 1)
        shared_walk = make_walk_adapter(D) if walk_shared else None
        self.blocks = nn.ModuleList([
            Block(D, n_head, hybrid_w, win_heads, mult, no_walk=no_walk, alibi=alibi, rope=rope,
                  no_attn=((i % attn_every) != 0), shared_walk=shared_walk, walk_loop=walk_loop, block_index=i)
            for i in range(L)])
        _shared_motif_walks = []
        for _blk in self.blocks:
            _w = getattr(_blk, "walk", None)
            if (_w is not None and getattr(_w, "motif_shared", False)
                    and all(_w is not _o for _o in _shared_motif_walks)):
                _shared_motif_walks.append(_w)
        if _shared_motif_walks:
            _w0 = _shared_motif_walks[0]
            for _w in _shared_motif_walks[1:]:
                if (_w.n_heads, _w.hd, _w.motif_count, _w.motif_rank, _w.motif_mod) != (
                        _w0.n_heads, _w0.hd, _w0.motif_count, _w0.motif_rank, _w0.motif_mod):
                    raise ValueError("shared motif bank needs identical walk geometry")
            self.walk_motif_bank = MotifBank(_w0.n_heads, _w0.motif_count,
                                             _w0.motif_rank, _w0.hd, mod=_w0.motif_mod)
            for _w in _shared_motif_walks:
                _w.attach_motif_bank(self.walk_motif_bank)
            print(f"  walk motif bank shared by {len(_shared_motif_walks)} blocks", flush=True)
        else:
            self.walk_motif_bank = None
        self.ln_f = nn.LayerNorm(D, dtype=DTYPE)
        self.head = nn.Linear(D, vocab, bias=False, dtype=DTYPE)
        self.ckpt = ckpt
        # Activation checkpointing affects only the autograd graph, never the
        # model state or checkpoint compatibility.  The default retains the
        # historical behavior (checkpoint every block).  On a large-VRAM host,
        # checkpointing every second block cuts substantial backward recompute;
        # a full-checkpoint threshold can restore the conservative path at the
        # largest sequence lengths.
        self.ckpt_stride = max(1, int(os.environ.get("TB_ACTIVATION_CKPT_STRIDE", "1")))
        self.ckpt_full_at_t = max(0, int(os.environ.get("TB_ACTIVATION_CKPT_FULL_AT_T", "0")))
        self.mask = None
        self.hybrid_w = hybrid_w
        self.n_head = n_head
        self.draft = None
        self._hid = None
        self.kmerge_enabled = os.environ.get("TB_KMERGE", "0") == "1"
        self.kmerge_k = max(2, int(os.environ.get("TB_KMERGE_K", "2")))
        self.kmerge_prefix_frac = min(0.95, max(0.05, float(
            os.environ.get("TB_KMERGE_PREFIX_FRAC", "0.75"))))
        self.kmerge_encoder = (
            AverageInitializedKMerge(D, self.kmerge_k) if self.kmerge_enabled else None
        )
        self.prrm_enabled = (self.kmerge_enabled and
                             os.environ.get("TB_PRRM", "0") == "1")
        self.prrm_train = os.environ.get("TB_PRRM_TRAIN", "0") == "1"
        self.prrm_version = int(os.environ.get("TB_PRRM_VERSION", "0"))
        self.prrm_cache = (PersistentRangeRepresentationMemory(
            os.environ.get("TB_PRRM_MAX_ITEMS", "4096"))
                           if self.prrm_enabled else None)
        # Optional Neural Interpreter.  It is deliberately late and causal:
        # only the last hidden state can influence the last logit, preventing a
        # global evidence pool from leaking future tokens into earlier losses.
        self.ni = bool(ni)
        self.ni_mode = str(ni_mode).strip().lower()
        if self.ni_mode not in ("final", "chunk", "view", "evidence"):
            raise ValueError("ni_mode must be 'final', 'chunk', 'view', or 'evidence'")
        if self.ni_mode not in ("final", "evidence") and ni_bridge:
            raise ValueError("ni_bridge is currently supported only with ni_mode='final' or 'evidence'")
        self.ni_steps = max(1, int(ni_steps))
        self.ni_trajectory = bool(ni_trajectory)
        self.ni_width_mult = max(1, int(ni_width_mult))
        self.ni_layer = min(L - 1, max(0, L // 2 if int(ni_layer) < 0 else int(ni_layer)))
        if str(ni_layers).strip():
            parsed_layers = [int(value.strip()) for value in str(ni_layers).split(",") if value.strip()]
        elif self.ni_mode == "evidence":
            parsed_layers = [L // 4, L // 2, (3 * L) // 4]
        else:
            parsed_layers = [self.ni_layer]
        self.ni_layers = tuple(sorted({min(L - 1, max(0, layer)) for layer in parsed_layers}))
        self.ni_bridge = bool(ni_bridge)
        self.ni_bridge_weight = float(ni_bridge_weight)
        if self.ni:
            if self.ni_mode == "evidence":
                # The external residual remains D-wide.  Width lives in the
                # evidence workspace, where it expands the available feature
                # basis before the final learned projection back into D.
                self.interpreter = EvidenceInterpreter(
                    D, rank=max(1, int(ni_rank)) * self.ni_width_mult, max_steps=self.ni_steps,
                    trajectory=self.ni_trajectory,
                ).to(dtype=DTYPE)
                self.ni_injector = None
            if self.ni_mode == "view":
                self.interpreter = UnifiedViewInterpreter(
                    D, rank=max(1, int(ni_rank)), max_steps=self.ni_steps,
                    trajectory=self.ni_trajectory,
                ).to(dtype=DTYPE)
                self.ni_injector = None
            if self.ni_mode == "chunk" and CausalChunkNeuralInterpreter is None:
                raise RuntimeError("TB_NI_MODE=chunk requires neural_interpreter.py beside the trainer")
            if self.ni_mode == "final" and (NeuralInterpreter is None or ZeroGatedSlotInjector is None or gradient_alignment_loss is None):
                raise RuntimeError("TB_NI=1 requires neural_interpreter.py beside the trainer")
            if self.ni_mode == "chunk":
                self.interpreter = CausalChunkNeuralInterpreter(
                    D, n_slots=max(1, int(ni_slots)), rank=max(1, int(ni_rank)),
                    chunk_size=max(1, int(ni_chunk_size)), max_steps=self.ni_steps,
                    trajectory=self.ni_trajectory,
                ).to(dtype=DTYPE)
                self.ni_injector = None
            elif self.ni_mode == "final":
                self.interpreter = NeuralInterpreter(
                    {"text": D}, D, n_slots=max(1, int(ni_slots)), rank=max(1, int(ni_rank)),
                    max_steps=self.ni_steps, trajectory=self.ni_trajectory,
                ).to(dtype=DTYPE)
                self.ni_injector = ZeroGatedSlotInjector(D, rank=max(1, int(ni_rank))).to(dtype=DTYPE)
        else:
            self.interpreter = None
            self.ni_injector = None
        # Lesson Memory Module (LMM): an external, functional fast-weight
        # matrix programmed from a lesson.  Unlike NI, it has no effect in
        # normal pretraining because a caller must explicitly supply memory.
        self.lmm = None
        self.lmm_layers = ()
        if lmm:
            if str(lmm_layers).strip():
                parsed_lmm_layers = [int(value.strip()) for value in str(lmm_layers).split(",") if value.strip()]
            else:
                parsed_lmm_layers = [L // 4, L // 2, (3 * L) // 4]
            self.lmm_layers = tuple(sorted({min(L - 1, max(0, layer)) for layer in parsed_lmm_layers}))
            self.lmm = LessonMemoryModule(vocab, D, rank=max(1, int(lmm_rank)), n_layers=L,
                                          write_rule=str(lmm_write).strip().lower(), eta=float(lmm_eta),
                                          decay=float(lmm_decay), gate_bias=float(lmm_gate_bias),
                                          lesson_source=lmm_lesson_source).to(dtype=DTYPE)
        self.lexical_semantic = None
        if os.environ.get("TB_LEXICAL_SEMANTIC", "0") == "1":
            self.lexical_semantic = LexicalSemanticAdapter(
                D,
                rank=int(os.environ.get("TB_LEXICAL_SEMANTIC_RANK", "256")),
                gate_init=float(os.environ.get("TB_LEXICAL_SEMANTIC_GATE_INIT", "0.0")),
                aux_weight=float(os.environ.get("TB_LEXICAL_SEMANTIC_AUX_WEIGHT", "0.05")),
                sample_stride=int(os.environ.get("TB_LEXICAL_SEMANTIC_STRIDE", "32")),
                temperature=float(os.environ.get("TB_LEXICAL_SEMANTIC_TEMPERATURE", "0.1")),
            )
        # Conceptualised memory (TB_MEM=1): a shared, model-level bank of concept
        # slots. Every block reads a gated retrieval of the current concept state;
        # the retrieval is cross-attended into the stream. Writes are differentiable
        # "pull-toward-query" updates gated per concept (update flows as traversal).
        # Zero-init read key/query projections => exact no-op at start (safe A/B).
        self.mem = None if no_mem else (ConceptMemory(D) if int(os.environ.get("TB_MEM", "0")) == 1 else None)
        # probe-proven init: N(0, 0.02) embeddings/head/pos (NOT the nn default N(0,1))
        with torch.no_grad():
            nn.init.normal_(self.emb.weight, 0.0, 0.02)
            if not (self.alibi or self.rope):
                nn.init.normal_(self.pos_emb, 0.0, 0.02)
            else:
                # RoPE/ALiBi: pos_emb unused but kept for ckpt compat; zero to save grad
                nn.init.zeros_(self.pos_emb)
            nn.init.normal_(self.head.weight, 0.0, 0.02)
        # weight tying: 102M save at vocab 50k D2048
        if int(os.environ.get("TB_TIE_EMB_HEAD", "0")) == 1:
            self.head.weight = self.emb.weight  # tied
        # pos_emb removal for fresh RoPE: save 16M params if TB_POS_EMB=0
        if (self.alibi or self.rope) and int(os.environ.get("TB_POS_EMB", "1")) == 0:
            self.register_parameter("pos_emb", None)
        # scalable predictive walk aux: view -> vocab CE, T-invariant
        if int(os.environ.get("TB_WALK_PRED_CE", "0")) == 1 and int(os.environ.get("TB_WALK_SUBSPACE_RANK", "0")) > 0:
            rank = int(os.environ.get("TB_WALK_SUBSPACE_RANK"))
            self.walk_aux_head = nn.Linear(rank, vocab, bias=False, dtype=DTYPE)
            nn.init.zeros_(self.walk_aux_head.weight)
            self.walk_aux_weight = float(os.environ.get("TB_WALK_AUX_W", "0.05"))
            self.walk_aux_loss = torch.tensor(0.0)
        else:
            self.walk_aux_head = None
            self.walk_aux_weight = 0.0
            self.walk_aux_loss = torch.tensor(0.0)
        # Recurrent depth (TB_LOOP_*): loop the [start, end) block section
        # `passes` times with shared weights under LoopController gates.
        # Disabled by default -> construction, state dict, and logits are
        # identical to the historical model (loop_controller stays None: no
        # new params, no optimizer entries, no checkpoint keys).
        self.loop_controller = None
        self.loop_start = 0
        self.loop_end = 0
        self.loop_passes = 1
        if os.environ.get("TB_LOOP_ENABLE", "0") == "1":
            try:
                loop_start = int(os.environ.get("TB_LOOP_START", ""))
                loop_end = int(os.environ.get("TB_LOOP_END", ""))
                loop_passes = int(os.environ.get("TB_LOOP_PASSES", "2"))
            except (TypeError, ValueError):
                raise ValueError(
                    "TB_LOOP_ENABLE=1 needs integer TB_LOOP_START/END "
                    "(and optional TB_LOOP_PASSES>=1)") from None
            if not (0 <= loop_start < loop_end <= L):
                raise ValueError(
                    f"TB_LOOP range [{loop_start}, {loop_end}) invalid for L={L}")
            if loop_passes < 1:
                raise ValueError(f"TB_LOOP_PASSES must be >= 1, got {loop_passes}")
            sidecars = []
            if any(getattr(b, "traj_reg", None) is not None for b in self.blocks):
                sidecars.append("traj_reg (TB_TRAJ_REG)")
            if self.mem is not None:
                sidecars.append("mem (TB_MEM)")
            if getattr(self, "interpreter", None) is not None:
                sidecars.append("ni")
            if getattr(self, "lmm", None) is not None:
                sidecars.append("lmm")
            if sidecars:
                raise RuntimeError(
                    "TB_LOOP_ENABLE=1 is incompatible with per-block aux sidecars "
                    f"({', '.join(sidecars)}): their per-block losses/bridge terms "
                    "assume one execution per block. Disable them first.")
            self.loop_start, self.loop_end, self.loop_passes = loop_start, loop_end, loop_passes
            self.loop_controller = LoopController(D, loop_passes)
            print(f"recurrent depth: looping blocks [{loop_start}, {loop_end}) "
                  f"x{loop_passes} (+{sum(p.numel() for p in self.loop_controller.parameters()):,} "
                  "params, zero-init identity)", flush=True)
        # Narrow-R recurrence (TB_R_*): Huginn-lite Prelude/Core/Coda retrofit.
        # P = blocks[:r_start], R = blocks[r_start:r_end] (shared, repeated),
        # C = blocks[r_end:]. Disabled by default -> r_adapter stays None and
        # forward takes the historical path exactly. Mutually exclusive with
        # TB_LOOP (full-section loop); the contract rejects enabling both.
        # Walk descent refiner (TB_WDR, experimental; see WalkDescentRefiner).
        self.wdr = None
        self.wdr_layer = -1
        if os.environ.get("TB_WDR", "0") == "1":
            self.wdr_layer = int(os.environ.get("TB_WDR_LAYER", str(max(0, (3 * L) // 4 - 1))))
            if not 0 <= self.wdr_layer < L:
                raise ValueError(f"TB_WDR_LAYER must be in [0, {L}), got {self.wdr_layer}")
            self.wdr = WalkDescentRefiner(
                D, rank=int(os.environ.get("TB_WDR_RANK", "64")),
                loops=int(os.environ.get("TB_WDR_LOOPS", "1")),
                aux_weight=float(os.environ.get("TB_WDR_AUX", "0.1")))
            print(f"walk descent refiner after block {self.wdr_layer} "
                  f"(rank {self.wdr.inp.out_features}, loops {self.wdr.loops}, "
                  f"aux {self.wdr.aux_weight})", flush=True)
        self.r_adapter = None
        self.r_start = 0
        self.r_end = 0
        self.r_mean = 0
        self.r_max = 0
        self.r_backprop = 0
        self.r_instr_only = True
        self.r_kv_share = False
        self.r_update = "blend"
        self.r_gain_mode = "per_pass"
        self.r_sample = "uniform"
        self.r_depth_cap = 0
        if os.environ.get("TB_R_ENABLE", "0") == "1":
            if os.environ.get("TB_LOOP_ENABLE", "0") == "1":
                raise RuntimeError("TB_R_ENABLE=1 is mutually exclusive with TB_LOOP_ENABLE=1")
            try:
                r_start = int(os.environ.get("TB_R_START", ""))
                r_end = int(os.environ.get("TB_R_END", ""))
                r_mean = int(os.environ.get("TB_R_MEAN", "0"))
                r_max = int(os.environ.get("TB_R_MAX", "8"))
                r_backprop = int(os.environ.get("TB_R_BACKPROP", "2"))
            except (TypeError, ValueError):
                raise ValueError(
                    "TB_R_ENABLE=1 needs integer TB_R_START/END/MEAN "
                    "(and optional TB_R_MAX/TB_R_BACKPROP)") from None
            if not (0 <= r_start < r_end <= L):
                raise ValueError(
                    f"TB_R range [{r_start}, {r_end}) invalid for L={L}")
            # R v2 controls (defaults = the original lineage, bitwise):
            #  TB_R_UPDATE blend: x <- s + g*(R(u) - s), u = LN(...) (RMS ~1).
            #    Against an un-normalized stream this mostly scales s by
            #    (1-g) each pass, has no fixed point (||dx||/||s|| ~ g
            #    forever, so halting never fires) and punishes depth.
            #  TB_R_UPDATE delta: u is rescaled to s's per-token RMS and
            #    x <- s + g*(R(u) - u): each pass adds only what the R blocks
            #    contribute; converged passes add ~0, halting can fire.
            #  TB_R_GAIN shared: every pass uses pass_gain[0], so depth can
            #    exceed r_max at train and test time (no per-index params
            #    that only ever see ~2/9 of microbatches).
            #  TB_R_SAMPLE poisson_lognormal (+ TB_R_SAMPLE_MAX cap): Huginn's
            #    heavy-tailed depth sampling. Sampling-only; no params.
            self.r_update = os.environ.get("TB_R_UPDATE", "blend").strip().lower() or "blend"
            self.r_gain_mode = os.environ.get("TB_R_GAIN", "per_pass").strip().lower() or "per_pass"
            _r_sample_raw = os.environ.get("TB_R_SAMPLE", "0").strip().lower() or "0"
            self.r_sample = {"0": "uniform", "1": "poisson_lognormal"}.get(
                _r_sample_raw, _r_sample_raw)
            if self.r_update not in ("blend", "delta"):
                raise ValueError(f"TB_R_UPDATE must be blend|delta, got {self.r_update!r}")
            if self.r_gain_mode not in ("per_pass", "shared"):
                raise ValueError(f"TB_R_GAIN must be per_pass|shared, got {self.r_gain_mode!r}")
            if self.r_sample not in ("uniform", "poisson_lognormal"):
                raise ValueError(
                    f"TB_R_SAMPLE must be uniform|poisson_lognormal, got {self.r_sample!r}")
            depth_cap = r_max
            if self.r_gain_mode == "shared":
                depth_cap = int(os.environ.get("TB_R_SAMPLE_MAX", str(max(r_max, 4 * max(r_mean, 1)))) or r_max)
            elif (os.environ.get("TB_R_SAMPLE_MAX", "") or "").strip() not in ("", str(r_max)):
                raise ValueError("TB_R_SAMPLE_MAX above r_max needs TB_R_GAIN=shared")
            self.r_depth_cap = max(1, int(depth_cap))
            if r_mean < 0 or r_max < 1 or r_mean > self.r_depth_cap:
                raise ValueError(
                    f"TB_R mean/max invalid: mean={r_mean} max={r_max} "
                    f"depth_cap={self.r_depth_cap} "
                    "(need 0 <= mean <= max, max >= 1; TB_R_GAIN=shared lifts "
                    "the depth cap to TB_R_SAMPLE_MAX)")
            if r_backprop < 0 or r_backprop > r_max:
                raise ValueError(
                    f"TB_R_BACKPROP must satisfy 0 <= k <= max ({r_max}), got {r_backprop}")
            sidecars = []
            if any(getattr(b, "traj_reg", None) is not None for b in self.blocks):
                sidecars.append("traj_reg (TB_TRAJ_REG)")
            if self.mem is not None:
                sidecars.append("mem (TB_MEM)")
            if getattr(self, "interpreter", None) is not None:
                sidecars.append("ni")
            if getattr(self, "lmm", None) is not None:
                sidecars.append("lmm")
            if sidecars:
                raise RuntimeError(
                    "TB_R_ENABLE=1 is incompatible with per-block aux sidecars "
                    f"({', '.join(sidecars)}): their per-block losses/bridge terms "
                    "assume one execution per block. Disable them first.")
            self.r_start, self.r_end = r_start, r_end
            self.r_mean, self.r_max, self.r_backprop = r_mean, r_max, r_backprop
            self.r_instr_only = os.environ.get("TB_R_INSTR_ONLY", "1") == "1"
            self.r_kv_share = os.environ.get("TB_R_KV_SHARE", "0") == "1"
            try:
                r_halt_eps = float(os.environ.get("TB_R_HALT_EPS", "0"))
                r_halt_min = int(os.environ.get("TB_R_HALT_MIN", "1"))
            except (TypeError, ValueError):
                raise ValueError(
                    "TB_R_HALT_EPS must be a number and TB_R_HALT_MIN an "
                    "integer") from None
            if not math.isfinite(r_halt_eps) or r_halt_eps < 0.0:
                raise ValueError(
                    "TB_R_HALT_EPS must be finite and >= 0, "
                    f"got {r_halt_eps:g}")
            if r_halt_min < 0:
                raise ValueError(
                    f"TB_R_HALT_MIN must be >= 0, got {r_halt_min}")
            self.r_halt_eps, self.r_halt_min = r_halt_eps, r_halt_min
            self.r_resid_log = os.environ.get("TB_R_RESID_LOG", "0") == "1"
            self.r_adapter = RAdapter(D, r_max)
            print(f"narrow-R: P=[0,{r_start}) R=[{r_start},{r_end}) C=[{r_end},{L}) "
                  f"mean={r_mean} max={r_max} k={r_backprop} "
                  f"instr_only={int(self.r_instr_only)} kv_share={int(self.r_kv_share)} "
                  f"(+{sum(p.numel() for p in self.r_adapter.parameters()):,} "
                  "params, zero-init identity)", flush=True)
            if self.r_kv_share:
                print("narrow-R KV-share: accepted and recorded; v1 recomputes "
                      "exactly (sharing kernel reserved, no function change)", flush=True)
        if os.environ.get("TB_R_ENABLE", "0") != "1" and (
                (os.environ.get("TB_R_HALT_EPS", "0") or "").strip() not in ("", "0", "0.0")
                or (os.environ.get("TB_R_HALT_MIN", "1") or "").strip() not in ("", "1")
                or os.environ.get("TB_R_RESID_LOG", "0") == "1"):
            raise ValueError(
                "TB_R_HALT_EPS/TB_R_HALT_MIN/TB_R_RESID_LOG need TB_R_ENABLE=1")
        # Sampled-depth window for trend logging: forward accumulates the
        # sampled extra-pass count per TRAINING microbatch; the trend block
        # prints the window mean and resets. Plain counters (no device,
        # no grad) so eval forwards and exotic wrappers stay safe.
        self._r_extra_sum = 0.0
        self._r_extra_count = 0
        # Refinement-residual window (TB_R_RESID_LOG=1): per-pass relative
        # residuals plus per-position means (the contraction curve) and the
        # halt count. Training-only, same reset discipline as _r_extra_*.
        self._r_resid_sum = 0.0
        self._r_resid_count = 0
        self._r_resid_pos = []
        self._r_resid_pos_n = []
        self._r_halt_sum = 0.0
        # Pause tokens (TB_PAUSE_N latents every TB_PAUSE_EVERY pairs): extra
        # latent compute inside pretraining with no CoT data and no vocabulary
        # change (they are embeddings, never token ids, never emitted). Off
        # by default -> pause_vecs is None and forward takes the historical
        # path exactly. Masters are fp32 (no VRAM concern at this size, and
        # it dodges the bf16-ULP freeze on small updates); compute casts to
        # the stream dtype at insertion so GEMM precision is unchanged.
        # TB_PAUSE_HUGINN=1 switches one group of 5 schedule-driven thoughts
        # (measured Huginn gains, dense first): each pause slot carries a
        # low-rank gated mixing refined inter-block below.
        pause_n = max(0, int(os.environ.get("TB_PAUSE_N", "0")))
        pause_every = max(0, int(os.environ.get("TB_PAUSE_EVERY", "512")))
        pause_huginn = os.environ.get("TB_PAUSE_HUGINN", "0") == "1"
        pause_rank = max(1, min(256, int(os.environ.get("TB_PAUSE_RANK", "32"))))
        if pause_huginn:
            if pause_n not in (0, len(HUGINN_GAINS)):
                raise ValueError(
                    f"TB_PAUSE_HUGINN=1 needs TB_PAUSE_N unset or {len(HUGINN_GAINS)}")
            pause_n = len(HUGINN_GAINS)
        if pause_n > 0 and pause_every <= 0:
            raise ValueError("TB_PAUSE_N>0 needs TB_PAUSE_EVERY>0")
        self.pause_every = int(pause_every)
        self.pause_huginn = bool(pause_huginn)
        self.pause_rank = int(pause_rank)
        if pause_n > 0:
            self.pause_vecs = nn.Parameter(torch.zeros(pause_n, D, dtype=torch.float32))
            with torch.no_grad():
                nn.init.normal_(self.pause_vecs, 0.0, 0.02)
            n_pause = sum(p.numel() for p in [self.pause_vecs])
            if pause_huginn:
                self.pause_gate = nn.Parameter(
                    torch.tensor(list(HUGINN_GAINS), dtype=torch.float32))
                self.pause_up = nn.ParameterList(
                    [nn.Parameter(torch.zeros(D, pause_rank, dtype=torch.float32))
                     for _ in range(pause_n)])
                self.pause_dn = nn.ParameterList(
                    [nn.Parameter(torch.zeros(pause_rank, D, dtype=torch.float32))
                     for _ in range(pause_n)])
                for dn in self.pause_dn:
                    with torch.no_grad():
                        nn.init.kaiming_uniform_(dn)
                n_pause += sum(p.numel() for p in [self.pause_gate]) + sum(
                    p.numel() for p in list(self.pause_up) + list(self.pause_dn))
            print(f"pause tokens: {pause_n} latents every {pause_every} pairs"
                  f"{' huginn-schedule' if pause_huginn else ''} "
                  f"(+{n_pause:,} fp32 params)", flush=True)
        else:
            self.register_parameter("pause_vecs", None)

    def _prepare_kmerge(self, idx, targets):
        """Compress only the prefix and mask its unavailable CE targets."""
        if not self.kmerge_enabled or self.kmerge_encoder is None or targets is None:
            return idx, targets, None, None, None
        B, T = idx.shape
        K = self.kmerge_k
        prefix = int(T * self.kmerge_prefix_frac) // K * K
        # Require at least one uncompressed input/target position.
        prefix = min(prefix, max(0, T - 1))
        if prefix < K:
            return idx, targets, None, None, None
        suffix = idx[:, prefix:]
        ranges = self.emb(idx[:, :prefix]).view(B, prefix // K, K, -1)
        use_cache = self.prrm_enabled and (not self.training or self.prrm_train)
        if not use_cache:
            merged = self.kmerge_encoder(ranges)
        else:
            # Cache individual contiguous K-token ranges, not whole prefixes.
            # This keeps a 1B-width entry at one D-vector rather than a full
            # T/K prefix (which would make a bounded cache unreasonably large).
            n_ranges = prefix // K
            merged_rows = []
            for row in range(B):
                row_out = [None] * n_ranges
                misses, miss_keys, miss_ranges = [], [], []
                for range_index in range(n_ranges):
                    start = range_index * K
                    token_key = idx[row, start:start + K].detach().to("cpu").numpy().tobytes()
                    key = (self.prrm_version, K, token_key)
                    cached = self.prrm_cache.get(key)
                    if cached is None:
                        misses.append(range_index)
                        miss_keys.append(key)
                        miss_ranges.append(ranges[row, range_index])
                    else:
                        row_out[range_index] = cached.to(device=idx.device, dtype=ranges.dtype)
                if miss_ranges:
                    miss_input = torch.stack(miss_ranges, dim=0).unsqueeze(0)
                    computed = self.kmerge_encoder(miss_input).squeeze(0).detach()
                    for range_index, key, value in zip(misses, miss_keys, computed):
                        self.prrm_cache.put(key, value)
                        row_out[range_index] = value.to(device=idx.device, dtype=ranges.dtype)
                merged_rows.append(torch.stack(row_out, dim=0))
            merged = torch.stack(merged_rows, dim=0)
        # Dummy ids are used only for auxiliary paths; the actual embeddings
        # are supplied separately, so no token is silently treated as input.
        dummy = idx[:, :prefix:K]
        merged_idx = torch.cat((dummy, suffix), dim=1)
        merged_emb = torch.cat((merged, self.emb(suffix)), dim=1)
        masked = torch.full((B, merged.shape[1]), -100, dtype=targets.dtype, device=targets.device)
        merged_targets = torch.cat((masked, targets[:, prefix:]), dim=1)
        # Use the last original position represented by each merged range.  A
        # suffix token therefore retains its original RoPE phase instead of
        # being silently shifted toward position zero.
        merged_pos = torch.cat((
            torch.arange(K - 1, prefix, K, device=idx.device, dtype=torch.long),
            torch.arange(prefix, T, device=idx.device, dtype=torch.long),
        ))
        return merged_idx, merged_targets, merged_emb, merged_pos, T

    def write_lesson_memory(self, lesson_ids, lesson_mask=None, *, bad_lesson_ids=None,
                            bad_lesson_mask=None, memory=None, eta=None, decay=None):
        """Program a new LMM state from a lesson without mutating model state.

        The returned tensor is caller-owned: retain it for a conversation or
        meta-learning episode, discard it for an isolated request, or serialize
        it explicitly in application code. It is intentionally not checkpointed
        with the model's slow weights.
        """
        if self.lmm is None:
            raise RuntimeError("LMM is disabled; launch with TB_LMM=1")
        return self.lmm.write_lesson(lesson_ids, lesson_mask, bad_lesson_ids=bad_lesson_ids,
                                     bad_lesson_mask=bad_lesson_mask, memory=memory,
                                     eta=eta, decay=decay,
                                     token_embedding=(self.emb.weight if self.lmm.lesson_source == "backbone"
                                                      else None))

    def lmm_meta_parameters(self):
        """Return only LMM parameters for frozen-backbone meta-training.

        Build the behavioral optimizer from this iterator. The language model
        remains in the differentiable read path, but optimizer updates remain
        confined to the lesson encoder, write rule, query, gate, and readout.
        """
        if self.lmm is None:
            raise RuntimeError("LMM is disabled; launch with TB_LMM=1")
        return self.lmm.parameters()

    @staticmethod
    def _lmm_sequence_logprob(logits, targets):
        if targets.shape != logits.shape[:2]:
            raise ValueError("LMM targets must match logits [batch, tokens]")
        valid = targets.ne(-100)
        if not bool(valid.any()):
            raise ValueError("LMM preference targets need at least one non--100 token")
        safe_targets = targets.clamp_min(0).unsqueeze(-1)
        token_logp = F.log_softmax(logits.float(), dim=-1).gather(-1, safe_targets).squeeze(-1)
        return (token_logp * valid.to(token_logp.dtype)).sum(dim=-1)

    def lmm_behavioral_objective(self, lesson_ids, chosen_idx, chosen_targets,
                                 rejected_idx, rejected_targets, unrelated_idx, *,
                                 lesson_mask=None, bad_lesson_ids=None,
                                 bad_lesson_mask=None, unrelated_mask=None,
                                 beta=0.1, interference_weight=1.0):
        """One differentiable write-then-read preference/interference episode.

        This is the outer meta-objective described by LMM. It is intentionally
        not wired into FineWeb streaming: that corpus has no lesson/task/chosen/
        rejected tuple, and pretending its next-token CE were such a tuple would
        train a misleading memory. A separate episodic loader should call this
        method with behavior-labelled examples.
        """
        if self.lmm is None:
            raise RuntimeError("LMM is disabled; launch with TB_LMM=1")
        if chosen_idx.shape[0] != lesson_ids.shape[0] or rejected_idx.shape[0] != lesson_ids.shape[0]:
            raise ValueError("LMM lesson, chosen, and rejected batches must match")
        if unrelated_idx.shape[0] != lesson_ids.shape[0]:
            raise ValueError("LMM unrelated batch must match lesson batch")
        if not (math.isfinite(float(beta)) and float(beta) > 0.0):
            raise ValueError("LMM beta must be positive and finite")
        if not (math.isfinite(float(interference_weight)) and float(interference_weight) >= 0.0):
            raise ValueError("LMM interference_weight must be finite and nonnegative")
        memory, write_stats = self.write_lesson_memory(
            lesson_ids, lesson_mask, bad_lesson_ids=bad_lesson_ids,
            bad_lesson_mask=bad_lesson_mask,
        )
        chosen_mem, _ = self(chosen_idx, lesson_memory=memory)
        rejected_mem, _ = self(rejected_idx, lesson_memory=memory)
        # Reference passes neither train nor retain an autograd graph. This is
        # both the intended frozen-LM objective and avoids holding three extra
        # decoder activation graphs during a behavioral episode.
        with torch.no_grad():
            chosen_base, _ = self(chosen_idx)
            rejected_base, _ = self(rejected_idx)
        # The no-memory paths are a fixed per-episode reference, never a way
        # to make the margin easier by moving the base language model itself.
        chosen_shift = self._lmm_sequence_logprob(chosen_mem, chosen_targets) - \
            self._lmm_sequence_logprob(chosen_base, chosen_targets).detach()
        rejected_shift = self._lmm_sequence_logprob(rejected_mem, rejected_targets) - \
            self._lmm_sequence_logprob(rejected_base, rejected_targets).detach()
        preference_margin = chosen_shift - rejected_shift
        task_loss = -F.logsigmoid(float(beta) * preference_margin).mean()
        unrelated_mem, _ = self(unrelated_idx, lesson_memory=memory)
        with torch.no_grad():
            unrelated_base, _ = self(unrelated_idx)
        mem_logp = F.log_softmax(unrelated_mem.float(), dim=-1)
        base_logp = F.log_softmax(unrelated_base.float(), dim=-1).detach()
        kl_per_token = (mem_logp.exp() * (mem_logp - base_logp)).sum(dim=-1)
        if unrelated_mask is not None:
            if unrelated_mask.shape != unrelated_idx.shape:
                raise ValueError("LMM unrelated_mask must match unrelated_idx")
            weights = unrelated_mask.to(kl_per_token.dtype)
            interference = (kl_per_token * weights).sum() / weights.sum().clamp_min(1)
        else:
            interference = kl_per_token.mean()
        total = task_loss + float(interference_weight) * interference
        return {
            "loss": total,
            "task_loss": task_loss.detach(),
            "preference_margin": preference_margin.detach().mean(),
            "interference_kl": interference.detach(),
            "memory": memory,
            "write": write_stats,
        }

    def lmm_selective_objective(self, lesson_ids, false_lesson_ids, chosen_idx,
                                chosen_targets, unrelated_idx, *, lesson_mask=None,
                                false_lesson_mask=None, unrelated_mask=None,
                                selectivity_beta=20.0, true_margin=0.01,
                                selectivity_margin=0.01, false_tolerance=0.005,
                                utility_weight=1.0,
                                selectivity_weight=1.0,
                                false_neutrality_weight=1.0,
                                interference_weight=0.25):
        """Train the exact true-vs-false retrieval behavior used at evaluation.

        Local real-WalkGPT smoke result (2026-09-08, D=32/L=2/rank=8, 80
        updates): selectivity 0.000063 -> 0.301580, delta_true -> 0.301543,
        delta_false -> -0.000036, unrelated KL -> 0.000373.  This records an
        integration/gradient test, not an expected natural-corpus effect size.

        Natural FineWeb-anchor v3 record (2026-09-09, 511 fixed validation
        episodes): selectivity rose from 0.000452 to 0.019833 at the best step
        1750; delta_true=0.032919, delta_false=0.013085, unrelated KL=0.016730.
        That run exposed the raw-Huber neutrality term as ~40x smaller than the
        two logistic terms; this version uses a logistic false tolerance too.

        Both lessons are encoded independently.  The same decoder prompt and
        target are then scored with true memory, false memory, and no memory.
        All shifts are measured in nats per target token, so variable response
        lengths cannot manufacture a larger preference margin.  The objective
        requires useful true memory, prefers it over false memory, keeps false
        memory neutral, and preserves unrelated decoder behavior.

        This intentionally does not subtract the false lesson inside the true
        lesson encoder.  Such subtraction made deployment without a supplied
        counterexample differ from training and made the reversed evaluation
        memory an unseen nonlinear encoder input.
        """
        if hasattr(self, "lmm") and self.lmm is None:
            raise RuntimeError("LMM is disabled; launch with TB_LMM=1")
        if lesson_ids.shape[0] != false_lesson_ids.shape[0] or \
                chosen_idx.shape[0] != lesson_ids.shape[0] or \
                unrelated_idx.shape[0] != lesson_ids.shape[0]:
            raise ValueError("LMM lessons, chosen prompt, and unrelated batch must match")
        if chosen_targets.shape != chosen_idx.shape or not bool(chosen_targets.ne(-100).any()):
            raise ValueError("LMM selective target must match chosen_idx and contain target tokens")
        scalar_values = {
            "selectivity_beta": selectivity_beta,
            "true_margin": true_margin,
            "selectivity_margin": selectivity_margin,
            "false_tolerance": false_tolerance,
            "utility_weight": utility_weight,
            "selectivity_weight": selectivity_weight,
            "false_neutrality_weight": false_neutrality_weight,
            "interference_weight": interference_weight,
        }
        if not all(math.isfinite(float(value)) for value in scalar_values.values()):
            raise ValueError("LMM selective objective values must be finite")
        if float(selectivity_beta) <= 0.0:
            raise ValueError("LMM selectivity_beta must be positive")
        if float(false_tolerance) < 0.0:
            raise ValueError("LMM false_tolerance must be nonnegative")
        if min(float(utility_weight), float(selectivity_weight),
               float(false_neutrality_weight), float(interference_weight)) < 0.0:
            raise ValueError("LMM selective objective weights must be nonnegative")

        true_memory, true_write = self.write_lesson_memory(
            lesson_ids, lesson_mask,
        )
        false_memory, false_write = self.write_lesson_memory(
            false_lesson_ids, false_lesson_mask,
        )
        true_logits, _ = self(chosen_idx, lesson_memory=true_memory)
        false_logits, _ = self(chosen_idx, lesson_memory=false_memory)
        with torch.no_grad():
            base_logits, _ = self(chosen_idx)

        token_count = chosen_targets.ne(-100).sum(dim=-1).clamp_min(1).to(torch.float32)
        true_logprob = self._lmm_sequence_logprob(true_logits, chosen_targets) / token_count
        false_logprob = self._lmm_sequence_logprob(false_logits, chosen_targets) / token_count
        base_logprob = self._lmm_sequence_logprob(base_logits, chosen_targets).detach() / token_count
        delta_true = true_logprob - base_logprob
        delta_false = false_logprob - base_logprob
        selectivity = true_logprob - false_logprob

        beta = float(selectivity_beta)
        utility_loss = -F.logsigmoid(beta * (delta_true - float(true_margin))).mean()
        selective_loss = -F.logsigmoid(
            beta * (selectivity - float(selectivity_margin))).mean()
        # Put neutrality on the same dimensionless logistic scale as utility
        # and selectivity.  The previous raw-Huber term was ~40x smaller in the
        # natural run despite its nominal 0.5 coefficient, so the optimizer was
        # still rewarded for allowing a generic wrong-memory boost.
        false_neutrality = -F.logsigmoid(
            beta * (float(false_tolerance) - delta_false.abs())
        ).mean()

        true_unrelated, _ = self(unrelated_idx, lesson_memory=true_memory)
        false_unrelated, _ = self(unrelated_idx, lesson_memory=false_memory)
        with torch.no_grad():
            base_unrelated, _ = self(unrelated_idx)
        base_unrelated_logp = F.log_softmax(base_unrelated.float(), dim=-1).detach()

        def _unrelated_kl(memory_logits):
            memory_logp = F.log_softmax(memory_logits.float(), dim=-1)
            per_token = (memory_logp.exp() * (memory_logp - base_unrelated_logp)).sum(dim=-1)
            if unrelated_mask is None:
                return per_token.mean()
            if unrelated_mask.shape != unrelated_idx.shape:
                raise ValueError("LMM unrelated_mask must match unrelated_idx")
            weights = unrelated_mask.to(per_token.dtype)
            return (per_token * weights).sum() / weights.sum().clamp_min(1)

        true_interference = _unrelated_kl(true_unrelated)
        false_interference = _unrelated_kl(false_unrelated)
        interference = 0.5 * (true_interference + false_interference)
        weighted_utility = float(utility_weight) * utility_loss
        weighted_selectivity = float(selectivity_weight) * selective_loss
        weighted_false_neutrality = float(false_neutrality_weight) * false_neutrality
        weighted_interference = float(interference_weight) * interference
        total = (weighted_utility + weighted_selectivity
                 + weighted_false_neutrality + weighted_interference)
        return {
            "loss": total,
            "utility_loss": utility_loss.detach(),
            "selectivity_loss": selective_loss.detach(),
            "false_neutrality_loss": false_neutrality.detach(),
            "interference_kl": interference.detach(),
            "weighted_utility": weighted_utility.detach(),
            "weighted_selectivity": weighted_selectivity.detach(),
            "weighted_false_neutrality": weighted_false_neutrality.detach(),
            "weighted_interference": weighted_interference.detach(),
            "delta_true": delta_true.detach().mean(),
            "delta_false": delta_false.detach().mean(),
            "selectivity": selectivity.detach().mean(),
            "true_memory": true_memory,
            "false_memory": false_memory,
            "true_write": true_write,
            "false_write": false_write,
        }

    @torch.no_grad()
    def lmm_retrieval_metrics(self, lesson_ids, chosen_idx, chosen_targets, unrelated_idx, *,
                              lesson_mask=None, bad_lesson_ids=None,
                              bad_lesson_mask=None, unrelated_mask=None):
        """Held-out natural-episode observability for LMM.

        `delta_true` and `delta_false` are per-target-token log-probability
        changes relative to no memory.  A useful memory must make delta_true
        positive *and* exceed delta_false; a generic activation bias cannot
        pass this selectivity test.  The CE fields score the held-out natural
        continuation itself, with the write sentence excluded from decoder
        context by the episodic loader.
        """
        if self.lmm is None:
            raise RuntimeError("LMM is disabled; launch with TB_LMM=1")
        if bad_lesson_ids is None:
            raise ValueError("LMM retrieval metrics require a cross-document bad_lesson")
        true_memory, _ = self.write_lesson_memory(
            lesson_ids, lesson_mask, bad_lesson_ids=bad_lesson_ids,
            bad_lesson_mask=bad_lesson_mask,
        )
        false_memory, _ = self.write_lesson_memory(
            bad_lesson_ids, bad_lesson_mask, bad_lesson_ids=lesson_ids,
            bad_lesson_mask=lesson_mask,
        )
        true_logits, _ = self(chosen_idx, lesson_memory=true_memory)
        false_logits, _ = self(chosen_idx, lesson_memory=false_memory)
        base_logits, _ = self(chosen_idx)
        base_logprob = self._lmm_sequence_logprob(base_logits, chosen_targets)
        true_logprob = self._lmm_sequence_logprob(true_logits, chosen_targets)
        false_logprob = self._lmm_sequence_logprob(false_logits, chosen_targets)
        token_count = chosen_targets.ne(-100).sum(dim=-1).clamp_min(1).to(torch.float32)
        delta_true = (true_logprob - base_logprob) / token_count
        delta_false = (false_logprob - base_logprob) / token_count

        unrelated_true, _ = self(unrelated_idx, lesson_memory=true_memory)
        unrelated_base, _ = self(unrelated_idx)
        true_logp = F.log_softmax(unrelated_true.float(), dim=-1)
        base_logp = F.log_softmax(unrelated_base.float(), dim=-1)
        kl_tokens = (true_logp.exp() * (true_logp - base_logp)).sum(dim=-1)
        if unrelated_mask is None:
            unrelated_kl = kl_tokens.mean()
        else:
            if unrelated_mask.shape != unrelated_idx.shape:
                raise ValueError("LMM unrelated_mask must match unrelated_idx")
            weights = unrelated_mask.to(kl_tokens.dtype)
            unrelated_kl = (kl_tokens * weights).sum() / weights.sum().clamp_min(1)
        return {
            "delta_true": delta_true.mean(),
            "delta_false": delta_false.mean(),
            "selectivity": (delta_true - delta_false).mean(),
            "ce_no_memory": (-base_logprob / token_count).mean(),
            "ce_true_memory": (-true_logprob / token_count).mean(),
            "unrelated_kl": unrelated_kl,
        }

    def _pause_thought(self, x, pause_mask, pause_slot):
        """Huginn-schedule latent thoughts: per-slot low-rank gated mixing
        applied to pause positions after every block, so later blocks
        attend refined pause states and the compute reaches followers.
        Zero mixing matrices = bitwise identity (canary-held); scalar
        gates start at the measured schedule and learn freely."""
        out = x.clone()
        for s in range(int(self.pause_vecs.shape[0])):
            sel = pause_mask & (pause_slot == s)
            if not bool(sel.any()):
                continue
            h = x[:, sel].float()
            moved = h @ (self.pause_up[s].to(h.dtype)
                         @ self.pause_dn[s].to(h.dtype)).t()
            out[:, sel] += (self.pause_gate[s] * moved).to(x.dtype)
        return out

    def _r_pass_input(self, r_adapter, s_in, e_state):
        """Narrow-R pass input u = norm(s + proj([s, e])) (+ RMS restore in
        delta mode, so the pretrained R blocks see stream-scale input)."""
        concat = torch.cat([s_in, e_state.to(s_in.dtype)], dim=-1)
        u = r_adapter.norm(
            s_in + r_adapter.proj(
                concat.to(r_adapter.proj.weight.dtype)).to(s_in.dtype))
        if getattr(self, "r_update", "blend") == "delta":
            with torch.no_grad():
                ratio = (s_in.float().pow(2).mean(-1, keepdim=True).sqrt()
                         / u.float().pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-6))
            u = u * ratio.to(u.dtype)
        return u

    def _r_pass_gain(self, r_adapter, pass_index):
        gains = r_adapter.pass_gain
        if getattr(self, "r_gain_mode", "per_pass") == "shared":
            return gains[0]
        return gains[min(int(pass_index), int(gains.numel()) - 1)]

    def _r_pass_update(self, s_in, u, yb, gain):
        if getattr(self, "r_update", "blend") == "delta":
            return s_in + gain * (yb - u)
        return s_in + gain * (yb - s_in)

    def forward(self, idx, targets=None, return_features=False, feature_mask=None, evidence=None,
                ni_scale=1.0, capture_ni_trace=False, lesson_memory=None,
                lmm_scale=1.0, capture_lmm_trace=False, r_active=None):
        """Run the decoder; trace capture is a debug-only NI observability hook.

        ``ni_scale=0`` ablates only evidence residuals while leaving the same
        backbone and input evidence in place.  ``capture_ni_trace`` stores
        detached per-depth residuals and pre-injection RMS on the model for a
        caller that wants to inspect causal logit mediation.  Neither option is
        used in ordinary training or generation.

        ``r_active`` gates narrow-R recurrence per microbatch (conditional-r):
        True forces refinement sampling, False forces the flat path, None keeps
        the historical behavior (R sampled whenever TB_R_ENABLE=1). The train
        loop passes per-microbatch source identity via should_apply_r(); eval
        leaves None (deterministic eval-mean, comparable banks).
        """
        idx, targets, input_embeds, position_ids, position_limit = self._prepare_kmerge(idx, targets)
        B, T = idx.shape
        assert T <= self.max_t, f"T={T} exceeds pos_emb {self.max_t}"
        # The raw token embedding is both the decoder input and the data-side
        # trajectory signal.  Reusing it avoids a second full embedding gather
        # (and its backward scatter) every microbatch while preserving exactly
        # the same tensor values and gradient flow.
        data_emb = input_embeds if input_embeds is not None else self.emb(idx)
        pause_T = T
        pause_mask, pause_slot = None, None
        if getattr(self, "pause_vecs", None) is not None:
            # Latent slots splice into data_emb space (x derives below, so
            # the first return is discarded). Targets grow matching -100
            # rows; walk trajectory reference stays aligned with x. Huginn
            # schedule mode uses one thought group (its measured live range).
            _, targets, data_emb = insert_pause_pairs(
                data_emb, targets, data_emb, self.pause_vecs,
                self.pause_every,
                single_group=bool(getattr(self, "pause_huginn", False)))
            pause_T = int(data_emb.shape[1])
            if pause_T > self.max_t:
                raise RuntimeError(
                    f"pause-extended T={pause_T} exceeds max_t={self.max_t}; "
                    "lower TB_PAUSE_N/TB_PAUSE_EVERY or raise max_t")
            if getattr(self, "pause_huginn", False):
                pause_mask, pause_slot = pause_layout(
                    T, int(self.pause_vecs.shape[0]), self.pause_every)
                pause_mask = pause_mask.to(device=data_emb.device)
                pause_slot = pause_slot.to(device=data_emb.device)
        if self.alibi or self.rope:
            x = data_emb
        else:
            x = data_emb + self.pos_emb[:pause_T].unsqueeze(0)
        if self.v41_engram is not None:
            x = x + self.v41_engram(idx)
        shared_k = shared_v = None
        if self.v41_ced is not None:
            ced_residual, encoder_summary = self.v41_ced(data_emb)
            x = x + ced_residual
        else:
            encoder_summary = data_emb
        if self.v41_shared_kv is not None:
            shared_k, shared_v = self.v41_shared_kv(encoder_summary)
        _all_walks = [w for b in self.blocks for w in (b.walk, getattr(b, 'phase_walk', None)) if w is not None]
        has_data_walk = any(getattr(w, 'data_traj', False) for w in _all_walks)
        has_traj_reg = any(getattr(b, 'traj_reg', None) is not None for b in self.blocks)
        if not math.isfinite(float(ni_scale)):
            raise ValueError("ni_scale must be finite")
        if not math.isfinite(float(lmm_scale)):
            raise ValueError("lmm_scale must be finite")
        if lesson_memory is not None and self.lmm is None:
            raise RuntimeError("lesson_memory was supplied but TB_LMM=0")
        self._last_ni_trace = [] if capture_ni_trace else None
        self._last_lmm_trace = [] if capture_lmm_trace else None
        self.ni_bridge_loss = x.new_zeros(())
        depth_tap, depth_velocity = None, None
        walk_previous_input, walk_previous_velocity = None, None
        walk_has_depth = any(getattr(w, 'data_traj', False) or getattr(w, 'pred_walk', False) for w in _all_walks)
        def _run_block(blk, block_index, x, taps):
            # Single-block step shared by the flat path and every loop pass.
            # `taps` carries (walk_previous_input, walk_previous_velocity,
            # depth_tap, depth_velocity) so walk depth-velocity chains across
            # loop passes exactly as it chains across depth. `blk` is a bound
            # argument (never a late-binding closure) so activation-checkpoint
            # recomputation replays the correct block, per the note below.
            walk_previous_input, walk_previous_velocity, depth_tap, depth_velocity = taps
            # compute depth trajectory for walk (fix #8: was dead)
            w_depth_v = None
            w_depth_c = None
            if walk_has_depth and walk_previous_input is not None:
                w_depth_v = x - walk_previous_input
                if walk_previous_velocity is not None:
                    w_depth_c = w_depth_v - walk_previous_velocity
            block_input = x
            if has_data_walk or walk_has_depth or has_traj_reg:
                # Pass the bound module directly. A lambda closing over `blk`
                # replays the final loop block during backward recomputation.
                checkpoint_this_block = (self.ckpt and
                    (self.ckpt_full_at_t > 0 and T >= self.ckpt_full_at_t
                     or (block_index % self.ckpt_stride) == 0))
                if checkpoint_this_block:
                    x = checkpoint(blk, x, data_emb=data_emb, depth_velocity=w_depth_v,
                                   depth_curvature=w_depth_c, position_ids=position_ids,
                                   position_limit=position_limit, shared_k=shared_k,
                                   shared_v=shared_v,
                                   use_reentrant=False)
                else:
                    x = blk(x, data_emb=data_emb, depth_velocity=w_depth_v,
                            depth_curvature=w_depth_c, position_ids=position_ids,
                            position_limit=position_limit, shared_k=shared_k,
                            shared_v=shared_v)
            else:
                checkpoint_this_block = (self.ckpt and
                    (self.ckpt_full_at_t > 0 and T >= self.ckpt_full_at_t
                     or (block_index % self.ckpt_stride) == 0))
                x = (checkpoint(blk, x, position_ids=position_ids,
                                position_limit=position_limit, shared_k=shared_k,
                                shared_v=shared_v, use_reentrant=False)
                     if checkpoint_this_block else blk(x, position_ids=position_ids,
                                                       position_limit=position_limit,
                                                       shared_k=shared_k, shared_v=shared_v))
            if pause_mask is not None and getattr(self, "pause_up", None) is not None:
                x = self._pause_thought(x, pause_mask, pause_slot)
            # update depth tap for next block (for walk and NI)
            if walk_has_depth:
                # Retain the previous block INPUT so the next difference is
                # its actual residual update. NI taps have separate history.
                walk_previous_input = block_input
                walk_previous_velocity = w_depth_v
            if self.interpreter is not None and self.ni_mode == "view" and block_index == self.ni_layer:
                # One learned alternate view is added before later blocks can
                # reason over it.  No objective-specific branch changes x.
                view_residual, _ = self.interpreter(x, steps=self.ni_steps)
                x = x + view_residual
            if self.interpreter is not None and self.ni_mode == "evidence" and block_index in self.ni_layers:
                # Repeated evidence queries are a feed-forward computation over
                # depth, not recurrence.  Each later LLM state asks the same
                # immutable input evidence a more informed question.
                velocity = torch.zeros_like(x) if depth_tap is None else x - depth_tap
                curvature = torch.zeros_like(x) if depth_velocity is None else velocity - depth_velocity
                evidence_residual, _ = self.interpreter(
                    x, velocity, curvature, evidence=evidence, steps=self.ni_steps,
                )
                applied_residual = evidence_residual * float(ni_scale)
                if self.ni_bridge and self.training and targets is not None and self.ni_bridge_weight > 0.0:
                    # Work in the LM head's normalized coordinate system. The
                    # surrogate is causal and label-visible only during loss
                    # computation; `applied_residual` is still label-free at
                    # inference and no target information enters the evidence bank.
                    local_before = self.ln_f(x)
                    local_after = self.ln_f(x + applied_residual)
                    self.ni_bridge_loss = self.ni_bridge_loss + self.ni_bridge_weight * _ni_local_decoder_gradient_bridge(
                        local_after - local_before, local_before, targets, self.head)
                if capture_ni_trace:
                    self._last_ni_trace.append({
                        "layer": block_index,
                        "residual": applied_residual.detach(),
                        "state_rms": x.detach().float().square().mean().sqrt(),
                    })
                x = x + applied_residual
                depth_tap, depth_velocity = x, velocity
            if self.lmm is not None and lesson_memory is not None and block_index in self.lmm_layers:
                lmm_residual, lmm_stats = self.lmm.inject(x, lesson_memory, block_index)
                applied_lmm = lmm_residual * float(lmm_scale)
                if capture_lmm_trace:
                    self._last_lmm_trace.append({
                        "layer": block_index,
                        "residual": applied_lmm.detach(),
                        **lmm_stats,
                    })
                x = x + applied_lmm
            if self.mem is not None:
                # cross-attend the retrieved concept state into every block's stream
                x = self.mem(x)
            if self.wdr is not None and block_index == self.wdr_layer:
                x = self.wdr(x)
            return x, (walk_previous_input, walk_previous_velocity, depth_tap, depth_velocity)

        _taps = (None, None, None, None)
        loop = self.loop_controller
        r_adapter = getattr(self, "r_adapter", None)
        if r_adapter is not None and loop is not None:
            raise RuntimeError("TB_R and TB_LOOP are mutually exclusive; refusing forward")
        if loop is None and r_adapter is None:
            for block_index, blk in enumerate(self.blocks):
                x, _taps = _run_block(blk, block_index, x, _taps)
        elif loop is not None:
            # Recurrent depth: ALL blocks run flat first (unchanged order, so
            # taps/aux/losses match the historical path exactly); then the loop
            # section runs K extra refinement passes with SHARED weights under
            # zero-init gates. Gate-closed (init) identity holds because every
            # adapter/emb/gate starts at exact zero: xr tracks x_flat bitwise.
            # Virtual block_index (S+j) keeps the ckpt-stride and layer-match
            # logic well-defined inside refinement passes.
            sect_start, sect_end, sect_passes = self.loop_start, self.loop_end, self.loop_passes
            for block_index, blk in enumerate(self.blocks):
                x, _taps = _run_block(blk, block_index, x, _taps)
            xr = x + loop.entry(x)
            for pass_index in range(sect_passes):
                xr = xr + loop.pass_emb[pass_index]
                yb = xr
                for offset, blk in enumerate(self.blocks[sect_start:sect_end]):
                    yb, _taps = _run_block(blk, sect_start + offset, yb, _taps)
                xr = xr + loop.pass_gate[pass_index] * (yb - xr)
            x = xr + loop.exit(xr)
        else:
            # Narrow-R (Huginn-lite P/R/C): P=[0,r_start) once, R=[r_start,r_end)
            # once flat, then r extra refinement passes with SHARED R weights.
            # e (prelude output) is re-injected every pass via RAdapter; the
            # pass_gain starts at exact zero so extra passes are bitwise
            # identity on step one (flat equivalence). n prefix passes run
            # under no_grad (Huginn truncated BPTT: backward cost O(k), not O(r));
            # only the last k passes store graphs. r_active=False (wiki when
            # TB_R_INSTR_ONLY=1) forces the flat path for this microbatch.
            rs, re = int(self.r_start), int(self.r_end)
            want_r = True if r_active is None else bool(r_active)
            n_nograd, k_grad = (0, 0)
            if want_r:
                n_nograd, k_grad = _sample_r_counts(
                    bool(self.training), int(self.r_mean),
                    int(self.r_depth_cap or self.r_max), int(self.r_backprop),
                    kind=self.r_sample)
            for block_index in range(0, rs):
                x, _taps = _run_block(self.blocks[block_index], block_index, x, _taps)
            e_state = x
            for block_index in range(rs, re):
                x, _taps = _run_block(self.blocks[block_index], block_index, x, _taps)
            total_extra = int(n_nograd + k_grad)
            halt_eps = float(getattr(self, "r_halt_eps", 0.0) or 0.0)
            halt_min = int(getattr(self, "r_halt_min", 1) or 0)
            want_resid = bool(getattr(self, "r_resid_log", False)) or halt_eps > 0.0
            halted_prefix = False
            realized_prefix = 0
            for pass_index in range(total_extra):
                if pass_index < n_nograd and halted_prefix:
                    # Converged prefix pass skipped; grad passes below keep
                    # their absolute pass_gain indices (no gain re-indexing).
                    continue
                # Truncated BPTT applies to training only: eval/validation runs
                # under an outer no_grad, and an inner enable_grad here would
                # re-enable tracking inside it (wasted graphs). Gate on the
                # mode as well as the pass index.
                gated = bool(self.training) and pass_index >= n_nograd
                with torch.enable_grad() if gated else torch.no_grad():
                    s_in = x
                    u = self._r_pass_input(r_adapter, s_in, e_state)
                    yb = u
                    for offset in range(rs, re):
                        yb, _taps = _run_block(self.blocks[offset], offset, yb, _taps)
                    gain = self._r_pass_gain(r_adapter, pass_index).to(yb.dtype)
                    if not gated:
                        u = u.detach()
                        yb = yb.detach()
                        gain = gain.detach()
                # Combine OUTSIDE the no-grad context: truncated BPTT stops the
                # gradient through the R function only, never through the skip
                # path. (Computing this inside no_grad detached the whole
                # residual stream, so every microbatch with a prefix pass
                # (r > backprop) sent embedding and prelude blocks ~2% of
                # their normal gradient. Forward values are unchanged.)
                x = self._r_pass_update(s_in, u, yb, gain)
                if pass_index < n_nograd:
                    realized_prefix += 1
                if want_resid and bool(self.training):
                    resid = _r_relative_resid(x, s_in)
                    self._r_resid_sum += resid
                    self._r_resid_count += 1
                    while len(self._r_resid_pos) <= pass_index:
                        self._r_resid_pos.append(0.0)
                        self._r_resid_pos_n.append(0)
                    self._r_resid_pos[pass_index] += resid
                    self._r_resid_pos_n[pass_index] += 1
                    if _r_should_halt(pass_index, n_nograd, resid,
                                      halt_eps, halt_min):
                        halted_prefix = True
                        self._r_halt_sum += 1.0
            # Realized (post-halt) count; with halting off this equals the
            # sampled total_extra exactly.
            realized_extra = int(realized_prefix + k_grad)
            if bool(self.training):
                self._r_extra_sum += float(realized_extra)
                self._r_extra_count += 1
            for block_index in range(re, len(self.blocks)):
                x, _taps = _run_block(self.blocks[block_index], block_index, x, _taps)
            self._last_r_extra = torch.tensor(realized_extra)
        if any(b.walk is not None and getattr(b.walk, "traj_smooth_on", False) for b in self.blocks):
            self.traj_smooth_total = sum(
                getattr(b.walk, "kappa_mean", torch.tensor(0.0)) for b in self.blocks if b.walk is not None
            )
        if any(b.walk is not None and getattr(b.walk, "traj_acc_on", False) for b in self.blocks):
            self.traj_acc_total = sum(
                getattr(b.walk, "acc_loss", torch.tensor(0.0)) for b in self.blocks if b.walk is not None
            )
        # Shared-Walk configurations expose the same module through multiple
        # blocks; de-duplicate by identity so the controller prior is invariant
        # to depth and sharing mode.
        cycle_walks = []
        cycle_seen = set()
        for block in self.blocks:
            walk = block.walk
            if (walk is not None and getattr(walk, "cycle_guard_on", False)
                    and id(walk) not in cycle_seen):
                cycle_walks.append(walk)
                cycle_seen.add(id(walk))
        if cycle_walks:
            self.walk_cycle_loss = torch.stack([
                walk.cycle_aux_loss.to(x.device) for walk in cycle_walks
            ]).mean()
            self.walk_cycle_pressure = torch.stack([
                walk.cycle_pressure_mean.to(x.device) for walk in cycle_walks
            ]).mean().detach()
            self.walk_cycle_strength = torch.stack([
                walk.cycle_strength.to(x.device) for walk in cycle_walks
            ]).mean().detach()
            self.walk_cycle_escape_strength = torch.stack([
                walk.cycle_escape_strength.to(x.device) for walk in cycle_walks
            ]).mean().detach()
        else:
            self.walk_cycle_loss = torch.tensor(0.0, device=x.device)
            self.walk_cycle_pressure = torch.tensor(0.0, device=x.device)
            self.walk_cycle_strength = torch.tensor(0.0, device=x.device)
            self.walk_cycle_escape_strength = torch.tensor(0.0, device=x.device)
        # Predictive next-state auxiliary (MSE). Phase-B predictors live in
        # ``phase_walk``; the old aggregation silently discarded their loss.
        predictive_walks = [
            walk for block in self.blocks
            for walk in (block.walk, getattr(block, "phase_walk", None))
            if walk is not None and getattr(walk, "pred_walk", False)
        ]
        if predictive_walks:
            self.walk_pred_total = sum(
                (getattr(walk, "pred_loss", torch.tensor(0.0, device=x.device))
                 for walk in predictive_walks),
                torch.tensor(0.0, device=x.device),
            )
        else:
            self.walk_pred_total = torch.tensor(0.0, device=x.device)
        motif_pred_walks = [
            blk.walk for blk in self.blocks
            if blk.walk is not None and getattr(blk.walk, "motif_pred", False)
        ]
        if motif_pred_walks:
            self.walk_pred_total = self.walk_pred_total + sum(
                (getattr(w, "motif_pred_loss", torch.tensor(0.0, device=x.device))
                 for w in motif_pred_walks),
                torch.tensor(0.0, device=x.device),
            )
        regressors = [b.traj_reg for b in self.blocks if getattr(b, 'traj_reg', None) is not None]
        if regressors:
            # Keep TB_TRAJ_REG_AUX_WEIGHT invariant to the number of layers:
            # this is one model-level auxiliary objective, not 16 independent
            # losses whose strength grows with depth.
            self.traj_reg_loss_total = torch.stack([r.aux_loss.to(x.device) for r in regressors]).mean()
            self.traj_reg_gate_mean = torch.stack([r.gate_mean.to(x.device) for r in regressors]).mean().detach()
            self.traj_reg_pred_nrmse = torch.stack([r.pred_nrmse.to(x.device) for r in regressors]).mean().detach()
        else:
            self.traj_reg_loss_total = torch.tensor(0.0, device=x.device)
            self.traj_reg_gate_mean = torch.tensor(0.0, device=x.device)
            self.traj_reg_pred_nrmse = torch.tensor(0.0, device=x.device)
        # scalable CE aux: walk view -> next token
        if getattr(self, "walk_aux_head", None) is not None:
            aux = x.new_zeros(())
            for blk in self.blocks:
                w = blk.walk
                if w is not None and hasattr(w, "last_view") and w.last_view is not None:
                    view = w.last_view
                    # handle k>1: (B,T*k,r) -> (B,T,r) mean
                    if getattr(w, "k_vectors", 1) > 1 and view.shape[1] != T:
                        view = view.view(B, T, w.k_vectors, w.rank).mean(dim=2)
                    if view.shape[1] != T:
                        # if view is Tk, take first T
                        view = view[:, :T, :]
                    aux_logits = self.walk_aux_head(view.to(self.walk_aux_head.weight.dtype))
                    if T > 1:
                        aux = aux + F.cross_entropy(aux_logits[:,:-1].reshape(-1, aux_logits.size(-1)), idx[:,1:].reshape(-1))
            # average over walks
            n_walks = max(1, len([b for b in self.blocks if b.walk is not None and hasattr(b.walk, "last_view")]))
            self.walk_aux_loss = aux / n_walks * self.walk_aux_weight
        else:
            self.walk_aux_loss = torch.tensor(0.0, device=x.device)
        self.lexical_semantic_loss = x.new_zeros((), dtype=torch.float32)
        self.lexical_semantic_gate = x.new_zeros((), dtype=torch.float32)
        if self.lexical_semantic is not None:
            semantic_residual, self.lexical_semantic_loss = self.lexical_semantic(
                x, targets=targets if self.training else None, embedding=self.emb)
            x = x + semantic_residual
            self.lexical_semantic_gate = self.lexical_semantic.gate_mean
        x = self.ln_f(x)
        x_before_ni = x
        if self.interpreter is not None and self.ni_mode in ("final", "chunk"):
            if self.ni_mode == "chunk":
                # Chunk j receives only the state committed by chunk j-1;
                # this gives NI a path to every token without future leakage.
                ni_state = self.interpreter(x, steps=self.ni_steps)
                x = x + ni_state.residual
            else:
                # The final causal state summarizes the whole prefix.  Restricting
                # evidence and injection to this position keeps the LM objective
                # autoregressive while giving NI a real training path.
                evidence = x[:, -1:].contiguous()
                evidence_mask = torch.ones(B, 1, dtype=torch.bool, device=x.device)
                ni_state = self.interpreter({"text": evidence}, {"text": evidence_mask}, steps=self.ni_steps)
                answer_mask = torch.zeros(B, T, dtype=torch.bool, device=x.device)
                answer_mask[:, -1] = True
                x = self.ni_injector(x, ni_state.slots, answer_mask)
            if self.ni_mode == "final" and self.ni_bridge and self.training and targets is not None and self.ni_bridge_weight > 0.0:
                # Compute d(CE)/d(hidden) for only the final target.  Avoiding
                # autograd.grad and a [B,T,V] one-hot keeps the bridge cheap.
                base_last = self.head(x_before_ni[:, -1:]).float()
                probabilities = F.softmax(base_last, dim=-1)
                final_targets = targets[:, -1].view(B, 1, 1)
                probabilities = probabilities.scatter_add(
                    -1, final_targets, torch.full_like(probabilities[..., :1], -1.0)
                )
                lm_gradient = probabilities @ self.head.weight.float()
                ni_delta = x[:, -1:] - x_before_ni[:, -1:]
                self.ni_bridge_loss = self.ni_bridge_loss + self.ni_bridge_weight * gradient_alignment_loss(ni_delta, lm_gradient)
        self._hid = x.detach().to(DTYPE)
        logits = self.head(x)
        loss = None
        if targets is not None:
            # Callers such as held-out episode/local-CNN probes may supply a
            # normal sliced target tensor. `reshape` preserves the existing
            # fast contiguous case while safely materializing a view when the
            # target stride is non-contiguous.
            # An SFT-masked window lying wholly inside one long prompt (tool
            # schema / tool result) has no scored target: the mean CE is
            # 0/0 = NaN there. Report 0 instead (its gradient is already zero:
            # ignored targets get none). Bit-identical whenever a target exists.
            flat_targets = targets.reshape(-1)
            n_valid = flat_targets.ne(-100).sum()
            self.lm_ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), flat_targets)
            self.lm_ce = torch.where(n_valid > 0, self.lm_ce, torch.zeros_like(self.lm_ce))
            self._last_lm_valid = n_valid.detach()
            loss = self.lm_ce
            if self.mem is not None:
                loss = loss + self.mem.write_loss.to(logits.device)
            # Auxiliary supervision is training-only; validation remains pure
            # next-token CE and stays comparable with historical checkpoints.
            if (predictive_walks or motif_pred_walks) and self.training:
                loss = loss + self.walk_pred_total.to(logits.device)
            if regressors:
                loss = loss + self.traj_reg_loss_total.to(logits.device)
            if getattr(self, "walk_aux_head", None) is not None:
                loss = loss + self.walk_aux_loss.to(logits.device)
            if self.lexical_semantic is not None:
                loss = loss + self.lexical_semantic_loss.to(logits.device)
            if cycle_walks and self.training:
                loss = loss + self.walk_cycle_loss.to(logits.device)
            if self.interpreter is not None and self.ni_mode in ("final", "evidence") and self.ni_bridge:
                loss = loss + self.ni_bridge_loss.to(logits.device)
            # keep pure CE for reporting
            self._last_lm_ce = self.lm_ce.detach()
        if return_features:
            if feature_mask is None:
                features = x.mean(dim=1)
            else:
                if feature_mask.shape != idx.shape:
                    raise ValueError("feature_mask must match idx shape [batch, tokens]")
                weights = feature_mask.unsqueeze(-1).to(x.dtype)
                features = (x * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
            return logits, loss, features
        return logits, loss

    # SPEED: the legacy path re-runs a full forward over the whole growing
    # window every token (O(T^2) FLOPs; with narrow-R each forward also pays
    # r_mean extra R-section passes). generate_cached carries exact states
    # (narrow-R supported since 2026-10-07) so FLOPs are O(T), BUT both paths
    # are kernel-launch-bound at chat lengths: the fp32 walk features, the
    # Phase-B sidecar and every R pass are hundreds of tiny kernels per block,
    # so one cached step costs about one short full forward (measured on the
    # laptop: ~0.3-1 s either way at T=128-512, D=512 L=16 R-mean 4). The cache
    # wins once full forwards become FLOP-bound (long contexts), and it
    # prefills token by token, so long prompts + short outputs favor legacy.
    # Biggest real levers: fewer R passes (TB_INFER_R_MEAN), then CUDA-graph /
    # compiled decode steps. use_cache=None picks the cache only when it is
    # supported and the prompt is no longer than the requested output; True
    # forces it (raises if unsupported), False forces legacy.
    def generate(self, idx, max_new, temperature=0.8, top_p=0.9, rep_penalty=1.1,
                 lesson_memory=None, lmm_scale=1.0, use_cache=None, eos_id=None):
        if getattr(self, "pause_vecs", None) is not None:
            raise RuntimeError(
                "pause-token inference schedule is not implemented yet "
                "(training/eval forwards only); generate() with pauses "
                "would silently evaluate a different function")
        if use_cache is None:
            use_cache = (int(idx.shape[1]) <= int(max_new)
                         and self._cache_supported(idx, max_new, lesson_memory))
        if use_cache:
            return self.generate_cached(
                idx, max_new, temperature=temperature, top_p=top_p,
                rep_penalty=rep_penalty, lesson_memory=lesson_memory, eos_id=eos_id)
        for _ in range(max_new):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
                gen_window = min(self.max_t, int(os.environ.get("TB_GENERATE_WINDOW", "4096")))
                logits, _ = self(idx[:, -gen_window:], lesson_memory=lesson_memory, lmm_scale=lmm_scale)
            if temperature <= 0:
                # Greedy: argmax directly. Dividing by zero yields +-inf and
                # softmax turns that into all-NaN, which the top_p filter
                # then erases into all-zeros (NaN poisons cumsum, keep goes
                # all-False) and multinomial asserts on a zero-sum row.
                idx = torch.cat(
                    [idx, logits[:, -1:, :].float().argmax(-1)], dim=-1)
                if eos_id is not None and int(idx[0, -1]) == int(eos_id):
                    break
                continue
            lg = logits[:, -1, :].float() / temperature
            if rep_penalty != 1.0 and idx.numel() > 1:
                seen = idx[0].tolist()
                lg[0, seen] = torch.where(
                    lg[0, seen] > 0, lg[0, seen] / rep_penalty, lg[0, seen] * rep_penalty
                )
            probs = F.softmax(lg, dim=-1)
            if top_p < 1.0:
                sp, si = probs.sort(descending=True)
                cum = sp.cumsum(-1)
                keep = (cum - sp) <= top_p  # cumulative BEFORE this token: top token always kept
                probs = probs.scatter(1, si, torch.where(keep, sp, torch.zeros_like(sp)))
            if torch.isnan(probs).any() or float(probs.sum()) <= 0:
                break
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=-1)
            if eos_id is not None and int(idx[0, -1]) == int(eos_id):
                break
        return idx

    def _cache_supported(self, idx, max_new, lesson_memory=None):
        """True when generate_cached can run this request exactly."""
        if lesson_memory is not None or self.training:
            return False
        if int(idx.shape[1]) + int(max_new) > int(getattr(self, "max_t", 16384)):
            return False
        try:
            self._cached_require_supported()
            for blk in self.blocks:
                sidecar = getattr(blk, "phase_walk", None)
                if sidecar is None:
                    continue
                if getattr(sidecar, "archive_present", False) and float(
                        getattr(sidecar, "archive_scale", 0.0)) != 0.0:
                    return False
                if (getattr(sidecar, "pred_walk", False)
                        and float(getattr(sidecar, "pred_scale", 0.0)) != 0.0
                        and float(getattr(sidecar, "probe_pred_scale", 1.0)) != 0.0):
                    return False
        except RuntimeError:
            return False
        try:
            import walk_assoc_triton  # noqa: F401
            triton_present = True
        except Exception:
            triton_present = False
        if triton_present and os.environ.get(
                "TB_WALK_ASSOC_BACKEND", "auto").strip().lower() != "python":
            return False
        return True

    def generate_cached(self, idx, max_new, temperature=0.8, top_p=0.9,
                        rep_penalty=1.1, lesson_memory=None, eos_id=None):
        """Generate with incremental exact-state decoding (see init/step below).

        The sampling tail mirrors generate() line-for-line; only the logits
        source differs (one cached step instead of a full-prefix forward).
        Runs under no_grad (values identical, no graph pileup); the legacy
        path is untouched.
        """
        if getattr(self, "pause_vecs", None) is not None:
            raise RuntimeError(
                "pause-token inference schedule is not implemented yet "
                "(training/eval forwards only)")
        if lesson_memory is not None:
            raise RuntimeError("cached generation does not support lesson_memory")
        total = int(idx.shape[1]) + int(max_new)
        if total > int(getattr(self, "max_t", 16384)):
            raise RuntimeError(
                f"cached generation of {total} tokens exceeds max_t; use the "
                "legacy generate() path (windowed) for longer outputs")
        cache = self.init_generate_cache(idx)
        with torch.no_grad():
            logits = None
            # Prefill: step through the prompt in order; keep final logits.
            for t in range(int(idx.shape[1])):
                logits = self.step_generate_cache(cache, idx[:, t:t + 1])
            for _ in range(max_new):
                if temperature <= 0:
                    # Greedy (see generate(): /0.0 makes +-inf, and the top_p
                    # filter erases the resulting NaN into an all-zero row).
                    nxt = logits[:, -1:, :].float().argmax(-1)
                    idx = torch.cat([idx, nxt], dim=-1)
                    logits = self.step_generate_cache(cache, nxt)
                    if eos_id is not None and int(idx[0, -1]) == int(eos_id):
                        break
                    continue
                lg = logits[:, -1, :].float() / temperature
                if rep_penalty != 1.0 and idx.numel() > 1:
                    seen = idx[0].tolist()
                    lg[0, seen] = torch.where(
                        lg[0, seen] > 0, lg[0, seen] / rep_penalty, lg[0, seen] * rep_penalty
                    )
                probs = F.softmax(lg, dim=-1)
                if top_p < 1.0:
                    sp, si = probs.sort(descending=True)
                    cum = sp.cumsum(-1)
                    keep = (cum - sp) <= top_p
                    probs = probs.scatter(1, si, torch.where(keep, sp, torch.zeros_like(sp)))
                if torch.isnan(probs).any() or float(probs.sum()) <= 0:
                    break
                nxt = torch.multinomial(probs, 1)
                idx = torch.cat([idx, nxt], dim=-1)
                logits = self.step_generate_cache(cache, nxt)
                if eos_id is not None and int(idx[0, -1]) == int(eos_id):
                    break
        return idx

    def _conv_step_cached(self, cst, x_t):
        """One causal depthwise step; mirrors CausalLocalFeatureConv.forward
        (LayerNorm first, then the fused or branched depthwise path)."""
        stage = cst["stage"]
        z_t = stage.norm(x_t)
        B = z_t.shape[0]
        D = int(stage.D)
        zc_t = z_t.transpose(1, 2)
        if cst["fuse_on"]:
            zwin = torch.cat([cst["tail"], zc_t], dim=2)[:, :, -cst["maxk"]:]
            local = F.conv1d(zwin, cst["fused"], groups=D)
            cst["tail"] = zwin[:, :, -(cst["maxk"] - 1):] if cst["maxk"] > 1 else zwin[:, :, :0]
        else:
            local = z_t.new_zeros(B, D, 1)
            tails, new_tails = [], []
            for weight, conv_w, kernel, tail in zip(
                    cst["weights"], cst["branch_w"], cst["branch_k"], cst["tails"]):
                wing = torch.cat([tail, zc_t], dim=2)[:, :, -kernel:]
                local = local + weight * F.conv1d(wing, conv_w)
                new_tails.append(wing[:, :, -(kernel - 1):] if kernel > 1 else wing[:, :, :0])
            cst["tails"] = new_tails
        local = local.transpose(1, 2)
        local = stage.out_proj(F.silu(stage.in_proj(local)))
        gate = torch.tanh(stage.residual_gate).to(local.dtype) + stage.gate_bootstrap
        return gate.clamp(-0.95, 0.95) * local

    # ---------------------------------- incremental (KV-cache) generation ---
    # Exact-state incremental decoding: per-step cost is one token of work
    # plus attention over the cache, instead of a full-prefix forward.
    # Every carried state mirrors the full forward's edge conditions
    # (repeat-padded shifts, zero-padded EMA/conv tails, zero accumulators,
    # completed-prefix hierarchy), so cached logits match full-forward logits
    # up to floating-point reassociation (tested, not asserted).
    # Unsupported (fail-closed): compiled graphs (dynamic shapes recompile
    # every step), recurrent loop_controller / narrow-R adapter, mem/NI/LMM/KMerge/traj_reg/V4.1
    # sidecars, drafts, walk exit/ponder, cycle guard, ALiBi, walk_loop != 1,
    # and totals beyond max_t. generate() keeps the legacy path by default.

    def _cached_require_supported(self):
        if hasattr(self, "_orig_mod"):
            raise RuntimeError("cached generation needs an eager model (compiled graphs recompile per step)")
        if getattr(self, "loop_controller", None) is not None:
            raise RuntimeError("cached generation does not support recurrent loop_controller")
        if getattr(self, "wdr", None) is not None:
            raise RuntimeError("cached generation does not support the walk descent refiner")
        if getattr(self, "r_adapter", None) is not None and self.training:
            raise RuntimeError("cached generation with narrow-R needs eval mode "
                               "(training samples a random depth per forward)")
        for name in ("mem", "interpreter", "ni_injector", "lmm", "draft",
                     "v41_engram", "v41_ced", "v41_shared_kv"):
            if getattr(self, name, None) is not None:
                raise RuntimeError(f"cached generation does not support {name}")
        if getattr(self, "kmerge_enabled", False):
            raise RuntimeError("cached generation does not support k-merge")
        if self.alibi:
            raise RuntimeError("cached generation supports RoPE or position-free models, not ALiBi")
        for blk in self.blocks:
            if getattr(blk, "traj_reg", None) is not None:
                raise RuntimeError("cached generation does not support traj_reg")
            if int(getattr(blk, "walk_loop", 1)) != 1:
                raise RuntimeError("cached generation needs walk_loop == 1")
            walk = getattr(blk, "walk", None)
            if walk is not None and getattr(walk, "exit_on", False):
                raise RuntimeError("cached generation does not support walk exit gates")
            if walk is not None and getattr(walk, "cycle_guard_on", False):
                raise RuntimeError("cached generation does not support cycle guard")
            # _walk_step_cached mirrors only the checkpoint-native features;
            # these extensions would otherwise be silently dropped.
            for flag in ("selmem_on", "geom_on", "place_on", "hurst_on", "affine_on",
                         "adapt_on", "conv_on"):
                if walk is not None and getattr(walk, flag, False):
                    raise RuntimeError(
                        f"cached generation does not support walk extension {flag}")

    @staticmethod
    def _attn_split(attn, D):
        n_head = attn.n_head
        hd = D // n_head
        W = int(attn.hybrid_w)
        Hw = min(attn.win_heads, n_head - 1) if W > 0 else 0
        return n_head, hd, W, Hw, n_head - Hw

    def _walk_step_cached(self, w, wst, x_t, pos):
        """One legacy-walk step with carried history; mirrors WalkAdapter.forward.

        Absolute-position repeat rules reproduce _sh exactly (_shift pads
        with the first elements, so d[t] = 0 for t < lag, x[t]-x[t-lag]
        otherwise). Cumulative features (energy, turning) use carried sums
        divided by (t+1); displacement reuses the carried first token; the
        coarse S-chunk hierarchy uses carried completed-chunk means. Short
        differential chains use two tiny rings. History advances afterwards.
        """
        B = x_t.shape[0]
        n, hd, D = int(w.n_heads), int(w.hd), self._cache_D()
        S = int(w.S)
        dt = torch.float32 if w.fp32_diffs else DTYPE
        xin = w.ln(x_t.to(w.ln.weight.dtype)).to(dt)
        xs = xin.view(B, 1, n, hd)
        tail = wst["xs_tail"]
        H = int(tail.shape[1])

        def _past(k):
            return tail[:, H - k].view(B, 1, n, hd)

        xm1 = xs if pos < 1 else _past(1)
        d1 = xs - xm1
        d1m1 = d1 if pos < 1 else wst["d1r"][-1]
        d2 = d1 + d1m1
        d2m2 = d2 if pos < 2 else wst["d2r"][-2]
        d4 = d2 + d2m2
        turn = d1 - d1m1
        kappa = 1.0 - F.cosine_similarity(d1, d1m1, dim=-1, eps=1e-8)
        kappa_mean = kappa.mean()
        if not w.traj_smooth_on:
            w.kappa_mean = kappa_mean.detach()
        else:
            w.kappa_mean = kappa_mean

        def _g(p):
            return torch.tanh(p).view(1, 1, n, hd)

        out = _g(w.w1) * d1 + _g(w.w2) * d2 + _g(w.w4) * d4 + _g(w.wt) * turn
        if w.motif_on:
            # Motif is per-position on the already-carried d1/d2/d4/turn.
            out = out + w._motif_residual(xs, d1, d2, d4, turn, dt)[0]
        ena = turn_tot = None
        if w.glob_on:
            tpos = float(pos + 1)
            if wst["x_first"] is None:
                wst["x_first"] = xs.clone()
            int_n = (xs - wst["x_first"]) / math.sqrt(tpos)
            # forward() feeds the trend RESIDUAL x_t - EMA_t, not the trace.
            ema_trace = self._walk_ema_step(wst, xs, pos, B, n, hd, dt)
            ema_n = (xs - ema_trace.view(B, 1, n, hd)) / (1.0 + math.sqrt(tpos) / 16.0)
            d16 = xs - (xs if pos < 16 else _past(16))
            d64 = xs - (xs if pos < 64 else _past(64))
            d256 = xs - (xs if pos < 256 else _past(256))
            wst["ena_sum"] = wst["ena_sum"] + (d1 * d1).sum(dim=-1, keepdim=True) / hd
            ena = wst["ena_sum"] / tpos
            wst["turn_sum"] = wst["turn_sum"] + kappa.unsqueeze(-1)
            turn_tot = wst["turn_sum"] / tpos
            out = (out + _g(w.wg_int) * int_n + _g(w.wg_ema) * ema_n
                   + _g(w.wg_d16) * d16 + _g(w.wg_d64) * d64 + _g(w.wg_d256) * d256)
            out = out + torch.tanh(w.wg_ena).view(1, 1, n, 1) * ena
            out = out + torch.tanh(w.wg_turn).view(1, 1, n, 1) * turn_tot
            if w.couple_on:
                out = (out + _g(w.wg_eflow) * (d1 * ema_n)
                       + _g(w.wg_inter) * (d1 * d16)
                       + _g(w.wg_cdisp) * (int_n * ena))
        out = out + torch.einsum("bthd,hg->btgd", out, w.head_cross.to(dt))
        if w.mix_proj is not None:
            x_flat = out.reshape(B, 1, D)
            acc = x_flat
            for pi in range(w.mix_depth):
                acc = acc + w.mix_gain * w._mix_delta(x_flat, w.mix_proj[pi])
            out = acc.view(B, 1, n, hd)
        feat_s = wst["c1m"].unsqueeze(1)
        feat_v = (wst["c1m"] - wst["c2m"]).unsqueeze(1)
        out = out + _g(w.wc1) * feat_s + _g(w.wc2) * feat_v
        out = out.view(B, 1, D)
        if getattr(w, "walk_norm", 0.0) > 0.0:
            dn = out.norm(dim=-1, keepdim=True) / math.sqrt(D) + 1e-6
            mode = w.walk_norm_mode
            if mode == "clip":
                out = out * torch.clamp(w.walk_norm / dn, max=1.0)
            elif mode == "cap":
                out = out * (w.walk_norm / (dn + w.walk_norm))
            else:
                out = out / dn * w.walk_norm
        out = out.view(B, 1, n, hd)
        if w.gate_on:
            with torch.no_grad():
                d1n = d1.pow(2).sum(dim=-1, keepdim=True).sqrt() / (hd ** 0.5)
            g_in = torch.cat([kappa.unsqueeze(-1), d1n,
                              ena, turn_tot], dim=-1).to(dt)
            g = torch.tanh(g_in @ w.gate_mlp.weight.to(dt).t())
            out = out * (1.0 + g)
        out = out * (1.0 + torch.tanh(w.gamma).to(dt) * kappa.unsqueeze(-1))
        out = w._cycle_guard(xin, xs, out)
        out = out.view(B, 1, D)
        wst["xs_tail"] = torch.cat([tail, xin.view(B, 1, D)], dim=1)[:, -257:]
        wst["d1r"].append(d1.detach())
        if len(wst["d1r"]) > 2:
            del wst["d1r"][:-2]
        wst["d2r"].append(d2.detach())
        if len(wst["d2r"]) > 3:
            del wst["d2r"][:-3]
        cw = getattr(w, "chunk_w", None)
        if cw is not None:
            wst["cacc"] = wst["cacc"] + cw.view(1, 1, -1)[:, :, pos % S].unsqueeze(-1) * xin.view(B, n, hd)
            wst["cacc_n"] = int(wst["cacc_n"]) + 1
            if (pos + 1) % S == 0:
                wst["c2m"] = wst["c1m"]
                wst["c1m"] = wst["cacc"]
                wst["cacc"] = torch.zeros_like(wst["cacc"])
                wst["cacc_n"] = 0
        else:
            wst["cacc"] = wst["cacc"] + xin.view(B, n, hd)
            wst["cacc_n"] = int(wst["cacc_n"]) + 1
            if (pos + 1) % S == 0:
                wst["c2m"] = wst["c1m"]
                wst["c1m"] = wst["cacc"] / max(1, int(wst["cacc_n"]))
                wst["cacc"] = torch.zeros_like(wst["cacc"])
                wst["cacc_n"] = 0
        return out.to(x_t.dtype)

    @staticmethod
    def _walk_ema_step(wst, xs, pos, B, n, hd, dt):
        """Closed-form truncated EMA matching _ema's convolution math exactly."""
        K = 64
        alpha, beta = 0.05, 0.95
        m = min(int(pos), K - 1)
        tail = wst["xs_tail"]
        H = int(tail.shape[1])
        # One weighted reduction over the last m tail rows instead of a
        # 64-iteration Python gather (it dominated cached-step launch cost).
        # The carried tail always holds >= m rows (it keeps the last 257).
        k_avail = min(m, H)
        num = alpha * xs.view(B, n, hd)
        if k_avail > 0:
            w = (alpha * beta ** torch.arange(k_avail, 0, -1, device=xs.device,
                                              dtype=torch.float32)).to(dt)
            num = num + torch.einsum("k,bkd->bd", w, tail[:, H - k_avail:].to(dt)).view(B, n, hd)
        den = alpha * (1.0 - beta ** (m + 1)) / (1.0 - beta)
        return num / (den + 1e-6)

    def _phase_step_cached(self, w, pst, x_t, data_emb_t, w_depth_v, w_depth_c, pos):
        """One Phase-B sidecar step; mirrors HierarchicalEvidenceWalkAdapter.forward.

        Differential chains use tiny rings with the same repeat rules as the
        full _shift; the dual associative read consumes the carried
        prior-only state before adding the current outer product; hierarchy
        summaries use carried completed-chunk means. The predictor and archive
        paths are absent here exactly when their runtime scales are zero
        (anything else raises at cache init).
        """
        B = x_t.shape[0]
        D = self._cache_D()
        dt = torch.float32
        normed = w._norm_model(x_t)
        has_k = int(w.k_vectors) > 1
        r = int(w.rank)
        if not has_k:
            u = F.linear(normed, w.in_proj.to(normed.dtype)).to(dt)
        else:
            k_views = int(w.k_vectors)
            u = F.linear(normed, w.in_proj_k.to(normed.dtype)).to(dt).view(B, 1, k_views, r)

        # Differential chains first (rings hold d1[t-2..t-1], d2[t-3..t-1]).
        # Order mirrors the full forward exactly: the data cross applies to
        # the projected u FIRST, and every differential below differences the
        # post-cross tensor (so u0/u1 histories are carried separately).
        def _d1_of(cur, ring):
            return cur - (cur if pos < 1 else ring[-1])

        u0 = u
        u0m = u0 if pos < 1 or pst["u0_prev"] is None else pst["u0_prev"]
        d1u = u0 - u0m
        if w.data_traj and data_emb_t is not None:
            d_norm = w._norm_data(data_emb_t)
            if not has_k:
                u_data = F.linear(d_norm, w.data_in_proj.to(d_norm.dtype)).to(dt)
            else:
                u_data = F.linear(d_norm, w.data_in_proj.to(d_norm.dtype)).to(dt).view(
                    B, 1, k_views, r)
            prev_ud = pst["u_data_prev"]
            if pos < 1 or prev_ud is None:
                d1_data = torch.zeros_like(u_data)
            else:
                d1_data = u_data - prev_ud
            cross = d1u - d1_data
            shape = (1, 1, r) if not has_k else (1, 1, 1, r)
            u = u0 + torch.tanh(w.data_gate.to(dt)).view(*shape) * cross
            pst["u_data_prev"] = u_data.detach()
        else:
            u = u0
        pst["u0_prev"] = u0.detach()
        u1m = u if pos < 1 or pst["u1_prev"] is None else pst["u1_prev"]
        pst["u1_prev"] = u.detach()
        d1 = u - u1m
        d1m1 = d1 if pos < 1 else pst["d1r"][-1]
        d2 = d1 + d1m1
        d2m2 = d2 if pos < 2 else pst["d2r"][-2]
        d4 = d2 + d2m2
        pst["d1r"].append(d1.detach())
        if len(pst["d1r"]) > 2:
            del pst["d1r"][:-2]
        pst["d2r"].append(d2.detach())
        if len(pst["d2r"]) > 3:
            del pst["d2r"][:-3]
        if w.traj_mode == 'd1':
            d2, d4 = torch.zeros_like(d1), torch.zeros_like(d1)
        elif w.traj_mode == 'state':
            d1 = torch.zeros_like(d1); d2 = torch.zeros_like(d1); d4 = torch.zeros_like(d1)

        def _project_depth(value):
            if value is None:
                return torch.zeros_like(u)
            if not has_k:
                return F.linear(value.to(normed.dtype), w.in_proj.to(normed.dtype)).to(dt)
            return F.linear(value.to(normed.dtype), w.in_proj_k.to(normed.dtype)).to(dt).view(
                B, 1, k_views, r)

        depth_v = _project_depth(w_depth_v)
        depth_c = _project_depth(w_depth_c)
        if w.traj_mode == 'state':
            depth_v, depth_c = torch.zeros_like(depth_v), torch.zeros_like(depth_c)
        if w.depth_rms:
            depth_v, log_v = w._unit_rms_with_log(depth_v)
            depth_c, log_c = w._unit_rms_with_log(depth_c)
            payload_parts = (u, d1, d2, d4, depth_v, depth_c, log_v, log_c)
        else:
            payload_parts = (u, d1, d2, d4, depth_v, depth_c)

        if not has_k:
            payload = torch.cat(payload_parts, dim=-1)
            view = F.silu(F.linear(payload, w.fuse_proj.to(dt)))
            w.last_view = view.detach()
            q = F.elu(F.linear(view, w.q_proj.to(dt))) + 1.0
            kk = F.elu(F.linear(u, w.k_proj.to(dt))) + 1.0
            motion_input = torch.cat((d1, d2, d4), dim=-1)
            if w.traj_mode == 'state':
                motion_input = torch.zeros_like(motion_input)
            skv = pst["skv"][:, 0]
            sk = pst["sk"][:, 0]
            r_state, r_motion, skv, sk = self._phase_dual_read(
                w, q, kk, u, motion_input, skv, sk)
            pst["skv"][:, 0] = skv
            pst["sk"][:, 0] = sk
            u3 = u
        else:
            payload_k = torch.cat(payload_parts, dim=-1)
            view_k = F.silu(F.linear(
                payload_k.reshape(B * 1 * k_views, -1), w.fuse_proj.to(dt))).view(B, 1, k_views, r)
            view = view_k.mean(dim=2)
            w.last_view = view.detach()
            rs_list, rm_list = [], []
            for v in range(k_views):
                qv = F.elu(F.linear(view_k[:, :, v], w.q_proj.to(dt))) + 1.0
                kv = F.elu(F.linear(u[:, :, v], w.k_proj.to(dt))) + 1.0
                mv_in = torch.cat((d1[:, :, v], d2[:, :, v], d4[:, :, v]), dim=-1)
                if w.traj_mode == 'state':
                    mv_in = torch.zeros_like(mv_in)
                r_s, r_m, skv_v, sk_v = self._phase_dual_read(
                    w, qv, kv, u[:, :, v], mv_in,
                    pst["skv"][:, v], pst["sk"][:, v])
                pst["skv"][:, v] = skv_v
                pst["sk"][:, v] = sk_v
                rs_list.append(r_s)
                rm_list.append(r_m)
            r_state = torch.stack(rs_list, dim=2).mean(dim=2)
            r_motion = torch.stack(rm_list, dim=2).mean(dim=2)
            u3 = u.mean(dim=2)
        r_state = r_state / (r_state.norm(dim=-1, keepdim=True) / math.sqrt(r) + 1e-6)
        r_motion = r_motion / (r_motion.norm(dim=-1, keepdim=True) / math.sqrt(r) + 1e-6)
        gate_logits = F.linear(torch.cat((view, r_state, r_motion), dim=-1), w.gate_proj.to(dt))
        if w.channel_gate:
            gate = torch.softmax(gate_logits.view(B, 1, 2, r), dim=2)
            mixed = gate[:, :, 0] * r_state + gate[:, :, 1] * r_motion
        else:
            gate = torch.softmax(gate_logits, dim=-1)
            mixed = gate[..., 0:1] * r_state + gate[..., 1:2] * r_motion
        w.pred_loss = torch.tensor(0.0, device=x_t.device, dtype=torch.float32)
        for si, scale in enumerate(w.hier_scales):
            hs, pst["hier"][si] = self._phase_hier_step(
                w, u3, scale, pos, B, r, pst["hier"][si])
            if w.learned_hierarchy:
                mixed = mixed + w.probe_hier_scale * torch.tanh(w.hier_gate[si].to(dt)).view(1, 1, -1) * hs
            else:
                mixed = mixed + (0.05 * w.probe_hier_scale) * hs
        correction = view + mixed
        out = F.linear(correction, w.out_proj.to(dt))
        if w.phase_residual_gate is not None:
            out = out * torch.tanh(w.phase_residual_gate.to(dt))
        out = out * w.probe_output_scale
        if w.walk_norm > 0.0:
            rms = out.norm(dim=-1, keepdim=True) / math.sqrt(self._cache_D()) + 1e-6
            if w.walk_norm_mode == 'clip':
                out = out * torch.clamp(w.walk_norm / rms, max=1.0)
            elif w.walk_norm_mode == 'cap':
                out = out * (w.walk_norm / (rms + w.walk_norm))
            else:
                out = out / rms * w.walk_norm
        return out.to(x_t.dtype)

    @staticmethod
    def _phase_dual_read(w, q, kk, state_input, motion_input, skv, sk):
        """One dual associative read against carried prior-only state.

        q/kk/state_input/motion_input are single-position (B,1,*); skv/sk hold
        strictly-prior prefixes and are returned updated. Mirrors one chunk of
        _dual_causal_associative_read with C=1 (prior-only by construction).
        """
        state_v = F.linear(state_input, w.state_v_proj.to(q.dtype))
        motion_v = F.linear(motion_input, w.motion_v_proj.to(q.dtype))
        values = torch.cat((state_v, motion_v), dim=-1)
        read = torch.einsum('btr,btrs->bts', q, skv.unsqueeze(1)) / (
            (q * sk.unsqueeze(1)).sum(dim=-1, keepdim=True).clamp_min(1e-6))
        kv = (kk.unsqueeze(-1) * values.unsqueeze(-2)).sum(dim=1)
        skv = skv + kv
        sk = sk + kk[:, 0, :]
        r = int(w.rank)
        state_read, motion_read = read.split(r, dim=-1)
        return (F.linear(state_read, w.state_out.to(q.dtype)),
                F.linear(motion_read, w.motion_out.to(q.dtype)), skv, sk)

    @staticmethod
    def _phase_hier_step(w, u3, scale, pos, B, r, hst):
        """Completed-prefix hierarchy from carried chunk sums; returns (hs, hst)."""
        C = int(scale)
        n_c = int(pos) // C if C > 0 else 0
        if n_c > 0 and int(hst["n"]) > 0:
            hs = hst["run"] / max(1, int(hst["n"]))
        else:
            hs = u3.new_zeros(B, 1, r)
        hst["pacc"] = hst["pacc"] + u3[:, 0, :]
        hst["pacc_n"] = int(hst["pacc_n"]) + 1
        if C > 0 and (int(pos) + 1) % C == 0:
            hst["run"] = hst["run"] + hst["pacc"] / max(1, int(hst["pacc_n"]))
            hst["n"] = int(hst["n"]) + 1
            hst["pacc"] = torch.zeros_like(hst["pacc"])
            hst["pacc_n"] = 0
        return hs, hst

    def _lex_step_cached(self, lex, x_t):
        """One lexical-semantic step; mirrors the adapter forward with
        targets=None (auxiliary supervision is training-only)."""
        z = lex.norm(x_t)
        features = F.silu(lex.up(z)) * lex.value(z)
        candidate = lex.down(features)
        route = torch.sigmoid(lex.route(z).float()).to(x_t.dtype)
        gate = torch.tanh(lex.residual_gate).to(x_t.dtype)
        residual = gate * route * candidate * lex.runtime_scale
        lex.gate_mean = (gate.detach().float() * route.detach().float()).mean() * lex.runtime_scale
        lex.aux_loss = x_t.new_zeros((), dtype=torch.float32)
        return residual

    def _block_step_cached(self, blk, bst, x, data_emb, taps, pos, walk_has_depth):
        """One block at one position; mirrors forward()'s _run_block.

        ``taps`` is (walk_previous_input, walk_previous_velocity) and chains
        across every block execution in forward order, R passes included.
        """
        walk_previous_input, walk_previous_velocity = taps
        block_input = x
        for cst in bst["conv"]:
            x = x + self._conv_step_cached(cst, x)
        if blk.attn is not None:
            x = x + self._attn_step_cached(blk.attn, bst["attn"], blk.ln1(x), pos)
        x = x + blk.mlp(blk.ln2(x))
        # forward()'s _run_block derives the depth trajectory from the block
        # INPUT before calling the block (the walks themselves still read the
        # post-MLP x). An earlier cached path differenced the post-MLP x here;
        # Phase-B unit-RMS-normalizes depth inputs, so the drift was small,
        # but the full forward is the trained function.
        w_depth_v = None
        w_depth_c = None
        if walk_has_depth and walk_previous_input is not None:
            w_depth_v = block_input - walk_previous_input
            if walk_previous_velocity is not None:
                w_depth_c = w_depth_v - walk_previous_velocity
        if blk.walk is not None:
            w = blk.walk
            for _ in range(int(getattr(blk, "walk_loop", 1))):
                w_inp, _, _, _ = controlled_walk_inputs(
                    x, data_emb, w_depth_v, w_depth_c, blk.walk_input_grad_scale)
                x = x + _scale_residual(
                    self._walk_step_cached(w, bst["walk"], w_inp, pos),
                    blk.walk_forward_scale)
        if (getattr(blk, "phase_walk", None) is not None
                and blk.phase_walk_forward_scale != 0.0):
            w = blk.phase_walk
            _, p_data, p_dv, p_dc = controlled_walk_inputs(
                x, data_emb, w_depth_v, w_depth_c, blk.phase_walk_input_grad_scale)
            x = x + _scale_residual(
                self._phase_step_cached(
                    w, bst["phase"], x,
                    p_data if getattr(w, "data_traj", False) else None,
                    p_dv, p_dc, pos),
                blk.phase_walk_forward_scale)
        if walk_has_depth:
            walk_previous_input = block_input
            walk_previous_velocity = w_depth_v
        return x, (walk_previous_input, walk_previous_velocity)

    def step_generate_cache(self, cache, tok):
        """Advance one absolute position; returns (B,1,V) logits. No grad.

        Mirrors one full-forward position exactly: same module call order
        (conv -> attn -> mlp -> walk -> phase per block, then lex/head), same
        depth-tap chaining across blocks, same controlled-input wrappers. Only
        the history source differs (carried states instead of full prefixes).
        """
        pos = int(cache["pos"])
        max_t = int(cache.get("max_t", 16384))
        if pos >= max_t:
            raise RuntimeError(
                f"cached generation reached max_t={max_t}; use the legacy "
                "generate() path (windowed) for longer outputs")
        B = int(cache["B"])
        if int(tok.shape[0]) != B or int(tok.shape[1]) != 1:
            raise ValueError("step_generate_cache needs [B, 1] token ids")
        walk_has_depth = bool(cache.get("walk_has_depth", False))
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
                x = data_emb = self.emb(tok)
                if not (self.alibi or self.rope):
                    x = data_emb + self.pos_emb[pos:pos + 1].unsqueeze(0)
                taps = (None, None)
                r_adapter = getattr(self, "r_adapter", None)
                if r_adapter is None:
                    for blk, bst in zip(self.blocks, cache["blocks"]):
                        x, taps = self._block_step_cached(
                            blk, bst, x, data_emb, taps, pos, walk_has_depth)
                else:
                    # Narrow-R, mirroring forward(): P and R once flat, then
                    # each extra pass re-runs R on its own carried states.
                    # Pass p's sequence at position t depends only on pass-p
                    # states at positions <= t, so per-pass caches are exact.
                    rs, re = int(self.r_start), int(self.r_end)
                    for i in range(0, re):
                        if i == rs:
                            e_state = x
                        x, taps = self._block_step_cached(
                            self.blocks[i], cache["blocks"][i], x, data_emb,
                            taps, pos, walk_has_depth)
                    for pass_index, pass_states in enumerate(cache["r_passes"]):
                        s_in = x
                        u = self._r_pass_input(r_adapter, s_in, e_state)
                        yb = u
                        for offset in range(rs, re):
                            yb, taps = self._block_step_cached(
                                self.blocks[offset], pass_states[offset - rs], yb,
                                data_emb, taps, pos, walk_has_depth)
                        gain = self._r_pass_gain(r_adapter, pass_index).to(yb.dtype)
                        x = self._r_pass_update(s_in, u, yb, gain)
                    for i in range(re, len(self.blocks)):
                        x, taps = self._block_step_cached(
                            self.blocks[i], cache["blocks"][i], x, data_emb,
                            taps, pos, walk_has_depth)
                if self.lexical_semantic is not None:
                    x = x + self._lex_step_cached(self.lexical_semantic, x)
                x = self.ln_f(x)
                logits = self.head(x)
        cache["pos"] = pos + 1
        return logits

    # NOTE 2026-09-30: generate_spec (speculative decoding off self.draft) was
    # removed with DraftNet — it had no callers (draft is never built here) and
    # fell back to generate() anyway. See the DraftNet removal stub.

    def init_generate_cache(self, input_ids):
        """Create empty incremental states (caller steps tokens in order)."""
        self._cached_require_supported()
        if input_ids.dim() != 2 or int(input_ids.shape[1]) < 1:
            raise ValueError("init_generate_cache needs [B, T>=1] token ids")
        B = int(input_ids.shape[0])
        device = input_ids.device
        max_t = int(getattr(self, "max_t", 16384))
        # The full forward honors an external Triton associative kernel when
        # importable; the cached path implements the tested Python math, so
        # require the Python backend explicitly whenever Triton exists.
        try:
            import walk_assoc_triton  # noqa: F401
            _triton_present = True
        except Exception:
            _triton_present = False
        if _triton_present and os.environ.get("TB_WALK_ASSOC_BACKEND", "auto").strip().lower() != "python":
            raise RuntimeError(
                "cached generation needs TB_WALK_ASSOC_BACKEND=python "
                "(a Triton associative kernel is installed and would take a "
                "different code path in full forwards)")
        _all_walks = [w for b in self.blocks
                      for w in (b.walk, getattr(b, "phase_walk", None)) if w is not None]
        walk_has_depth = any(getattr(w, "data_traj", False) or getattr(w, "pred_walk", False)
                             for w in _all_walks)
        cache = {"pos": 0, "B": B, "device": device, "max_t": max_t,
                 "walk_has_depth": walk_has_depth,
                 "blocks": [self._init_block_cache(blk, B, device) for blk in self.blocks],
                 "r_passes": []}
        if getattr(self, "r_adapter", None) is not None:
            # Eval-mode depth exactly as forward() uses it (deterministic
            # r = mean, no halting outside training): one private cache per
            # extra pass for every R-section block.
            n_nograd, k_grad = _sample_r_counts(
                False, int(self.r_mean), int(self.r_depth_cap or self.r_max),
                int(self.r_backprop))
            cache["r_passes"] = [
                [self._init_block_cache(self.blocks[i], B, device)
                 for i in range(int(self.r_start), int(self.r_end))]
                for _ in range(int(n_nograd + k_grad))]
        return cache

    def _init_block_cache(self, blk, B, device):
        """Empty incremental state for one block execution stream."""
        bst = {}
        attn = blk.attn
        if attn is not None:
            n_head, hd, W, Hw, Hg = self._attn_split(attn, self._cache_D())
            kd = DTYPE
            bst["attn"] = {
                "Hg": Hg, "Hw": Hw, "W": W, "hd": hd,
                "rope": bool(attn.rope),
                "rope_base_window": float(getattr(attn, "rope_base_window", 512.0)),
                "has_null": hasattr(attn, "null_key"),
                "null_w": None,
                "k_g": torch.empty(B, Hg, 0, hd, device=device, dtype=kd) if Hg else None,
                "v_g": torch.empty(B, Hg, 0, hd, device=device, dtype=kd) if Hg else None,
                "k_w": torch.empty(B, Hw, 0, hd, device=device, dtype=kd) if Hw else None,
                "v_w": torch.empty(B, Hw, 0, hd, device=device, dtype=kd) if Hw else None,
            }
        else:
            bst["attn"] = None
        walk = getattr(blk, "walk", None)
        if walk is not None:
            dt = torch.float32 if getattr(walk, "fp32_diffs", True) else DTYPE
            n, hd_w = int(walk.n_heads), int(walk.hd)
            bst["walk"] = {
                "dt": dt, "n": n, "hd": hd_w, "S": int(getattr(walk, "S", 8)),
                "xs_tail": torch.zeros(B, 0, self._cache_D(), device=device, dtype=dt),
                "x_first": None,
                "d1r": [], "d2r": [],
                "ena_sum": torch.zeros(B, n, 1, device=device, dtype=dt),
                "turn_sum": torch.zeros(B, n, 1, device=device, dtype=dt),
                "c1m": torch.zeros(B, n, hd_w, device=device, dtype=dt),
                "c2m": torch.zeros(B, n, hd_w, device=device, dtype=dt),
                "cacc": torch.zeros(B, n, hd_w, device=device, dtype=dt),
                "cacc_n": 0,
            }
        else:
            bst["walk"] = None
        sidecar = getattr(blk, "phase_walk", None)
        if sidecar is not None:
            if getattr(sidecar, "archive_present", False) and float(
                    getattr(sidecar, "archive_scale", 0.0)) != 0.0:
                raise RuntimeError(
                    "cached generation does not support an active phase archive")
            if (getattr(sidecar, "pred_walk", False)
                    and float(getattr(sidecar, "pred_scale", 0.0)) != 0.0
                    and float(getattr(sidecar, "probe_pred_scale", 1.0)) != 0.0):
                raise RuntimeError(
                    "cached generation does not support an enabled phase predictor")
            r = int(sidecar.rank)
            kv = int(sidecar.k_vectors)
            rv = 2 * r
            bst["phase"] = {
                "k": kv, "r": r, "rv": rv,
                "skv": torch.zeros(B, kv, r, rv, device=device, dtype=torch.float32),
                "sk": torch.zeros(B, kv, r, device=device, dtype=torch.float32),
                "u0_prev": None, "u1_prev": None,
                "d1r": [], "d2r": [],
                "u_data_prev": None,
                "hier": [self._hier_state(B, r, device) for _ in sidecar.hier_scales],
                "hier_scales": tuple(int(v) for v in sidecar.hier_scales),
            }
        else:
            bst["phase"] = None
        convs = []
        if getattr(blk, "local_conv", None) is not None:
            for stage in [blk.local_conv, *list(getattr(blk, "local_conv_extra", ()))]:
                convs.append(self._init_conv_cache(stage, B, device))
        bst["conv"] = convs
        return bst

    @staticmethod
    def _hier_state(B, r, device):
        return {"run": torch.zeros(B, r, device=device, dtype=torch.float32),
                "n": 0,
                "pacc": torch.zeros(B, r, device=device, dtype=torch.float32),
                "pacc_n": 0}

    def _cache_D(self):
        return int(self.emb.weight.shape[1])

    @staticmethod
    def _init_conv_cache(stage, B, device):
        """Fused kernel + zero left-tails (fused tail + per-branch tails)."""
        fused = None
        max_kernel = 0
        for conv in stage.depthwise:
            kernel = int(conv.weight.shape[-1])
            max_kernel = max(max_kernel, kernel)
        weights = torch.softmax(stage.scale_logits.float(), dim=0).to(
            stage.depthwise[0].weight.dtype)
        branches = []
        for weight, conv in zip(weights, stage.depthwise):
            kernel = int(conv.weight.shape[-1])
            aligned = F.pad(conv.weight.detach(), (max_kernel - kernel, 0))
            term = weight * aligned
            fused = term if fused is None else fused + term
            branches.append((weight.detach(), conv.weight.detach(), kernel))
        D = int(stage.D)
        wdt = stage.depthwise[0].weight.dtype
        return {
            "stage": stage,
            "maxk": max_kernel,
            "fused": fused.detach() if fused is not None else None,
            "fuse_on": os.environ.get("TB_LOCAL_CONV_FUSE_KERNELS", "1") == "1",
            "weights": [w.detach() for w, _, _ in branches],
            "branch_w": [wgt for _, wgt, _ in branches],
            "branch_k": [k for _, _, k in branches],
            "tail": torch.zeros(B, D, max(0, max_kernel - 1), device=device, dtype=wdt),
            "tails": [torch.zeros(B, D, max(0, k - 1), device=device, dtype=wdt)
                      for _, _, k in branches],
        }

    def _attn_step_cached(self, attn, ast, x_t, pos):
        """One causal attention step; mirrors Attn.forward row-by-row."""
        B = x_t.shape[0]
        D = self._cache_D()
        n_head, hd, W, Hw, Hg = self._attn_split(attn, D)
        q, k, v = attn.qkv(x_t).split(D, dim=-1)
        q = q.view(B, 1, n_head, hd).transpose(1, 2)
        k = k.view(B, 1, n_head, hd).transpose(1, 2)
        v = v.view(B, 1, n_head, hd).transpose(1, 2)
        if attn.rope:
            if Hw > 0:
                cos_g, sin_g = _rope_row_tables(hd, ROPE_BASE, self.max_t,
                                               x_t.device, x_t.dtype)
                cos_w, sin_w = _rope_row_tables(
                    hd, float(attn.rope_base_window), self.max_t, x_t.device, x_t.dtype)
                cg = cos_g[pos:pos + 1].unsqueeze(0).unsqueeze(0)
                sg = sin_g[pos:pos + 1].unsqueeze(0).unsqueeze(0)
                cw = cos_w[pos:pos + 1].unsqueeze(0).unsqueeze(0)
                sw = sin_w[pos:pos + 1].unsqueeze(0).unsqueeze(0)
                q = torch.cat([_rope_rotate(q[:, :Hg], cg, sg),
                               _rope_rotate(q[:, Hg:], cw, sw)], dim=1)
                k = torch.cat([_rope_rotate(k[:, :Hg], cg, sg),
                               _rope_rotate(k[:, Hg:], cw, sw)], dim=1)
            else:
                cos, sin = _rope_row_tables(hd, ROPE_BASE, self.max_t,
                                           x_t.device, x_t.dtype)
                cos_b = cos[pos:pos + 1].unsqueeze(0).unsqueeze(0)
                sin_b = sin[pos:pos + 1].unsqueeze(0).unsqueeze(0)
                q = _rope_rotate(q, cos_b, sin_b)
                k = _rope_rotate(k, cos_b, sin_b)
        outs = []
        if Hg > 0:
            qg, kg, vg = q[:, :Hg], k[:, :Hg], v[:, :Hg]
            ast["k_g"] = torch.cat([ast["k_g"], kg], dim=2)
            ast["v_g"] = torch.cat([ast["v_g"], vg], dim=2)
            a = torch.matmul(qg, ast["k_g"].transpose(-1, -2)) / math.sqrt(hd)
            a = F.softmax(a, dim=-1)
            outs.append(torch.matmul(a, ast["v_g"]))
        if Hw > 0:
            qw, kw, vw = q[:, Hg:], k[:, Hg:], v[:, Hg:]
            ast["k_w"] = torch.cat([ast["k_w"], kw], dim=2)[:, :, -(W + 1):]
            ast["v_w"] = torch.cat([ast["v_w"], vw], dim=2)[:, :, -(W + 1):]
            n_real = int(ast["k_w"].shape[2])
            n_null = (W + 1) - n_real
            if ast["has_null"]:
                nk = ast["null_w"]
                if nk is None:
                    nk = attn.null_key.expand(B, -1, W, -1)
                    ast["null_w"] = nk
                kp = torch.cat([nk[:, :, :max(0, n_null)], ast["k_w"]], dim=2)
                vp = torch.cat([nk.new_zeros(B, Hw, max(0, n_null), hd), ast["v_w"]], dim=2)
            else:
                kp, vp = ast["k_w"], ast["v_w"]
            a = torch.matmul(qw, kp.transpose(-1, -2)) / math.sqrt(hd)
            a = F.softmax(a, dim=-1)
            outs.append(torch.matmul(a, vp))
        out = torch.cat(outs, dim=1) if len(outs) > 1 else outs[0]
        out = out.transpose(1, 2).contiguous().view(B, 1, D)
        return attn.proj(out)


# --------------------------------------------------------------- data I/O ---

SHARD_CAP = int(os.environ.get("TB_SHARD_CAP", "200000000"))


def _load_tokenizer(name):
    """Tokenizer that won't warn about long sequences: gpt2's config sets
    model_max_length=1024, but the CoT pipeline tokenizes full traces (100k+
    tokens). Nothing truncates (truncation=False by default), so the warning is
    pure noise — raise the cap to silence it."""
    tok = __import__("transformers").AutoTokenizer.from_pretrained(name)
    tok.model_max_length = 2 ** 20
    return tok


def _load_shard(path, cap_tokens=None):
    if path.endswith(".npy"):
        arr = torch.from_numpy(__import__("numpy").load(path))
    else:
        arr = torch.from_numpy(__import__("numpy").fromfile(path, dtype="uint16"))
    cap = SHARD_CAP if cap_tokens is None else cap_tokens
    return arr[:cap].long()


def _tokenize_stream(text, tok, target):
    ids = []
    step = 500_000
    for i in range(0, len(text), step):
        ids.extend(tok(text[i:i + step], return_tensors="pt")["input_ids"][0].tolist())
        if len(ids) >= target:
            break
    return torch.tensor(ids[:target], dtype=torch.long)



# ---------------------------------------------------------------- data mix v3 -
# Compute-limited continuation preset: 15B training-token allocation.  The
# underlying corpora are much larger than these budgets; the numbers are a
# sampling curriculum, not truncation/download limits.  Deliberately excludes
# Nemotron MMLU-SFT / generic MC benchmark-derived slices so MMLU remains a
# meaningful external evaluation.
_NEMOTRON_KNOWLEDGE_15B = (
    dict(name="web_hq", dataset="nvidia/Nemotron-CC-v2.1:High-Quality",
         weight=0.45, budget_tokens=6_750_000_000, domains=8,
         text_field="text", domain_key="uuid"),
    # Official FinePDFs-Edu English stream, with the transparent high-signal
    # filters used by the public HQ derivative applied online.  This avoids the
    # mixed-language/OCR-heavy rows visible in the unfiltered eng_Latn viewer.
    dict(name="pdf_edu_hq", dataset="HuggingFaceFW/finepdfs-edu:eng_Latn",
         weight=0.15, budget_tokens=2_250_000_000, domains=4,
         text_field="text", domain_key="url",
         filters={
             "full_doc_lid": ["eq", "eng_Latn"],
             "page_average_lid_score": ["gte", 0.90],
             "token_count": ["gte", 512],
             "fw_edu_scores": ["max_gte", 2.0],
         }),
    dict(name="math_4plus", dataset="nvidia/Nemotron-CC-Math-v1:4plus",
         weight=0.12, budget_tokens=1_800_000_000, domains=4,
         text_field="text", domain_key="uuid"),
    dict(name="stem_rqa", dataset="nvidia/Nemotron-Pretraining-Specialized-v1:Nemotron-Pretraining-RQA",
         weight=0.10, budget_tokens=1_500_000_000, domains=3,
         text_field="text", domain_key="uuid"),
    dict(name="fact_seeking", dataset="nvidia/Nemotron-Pretraining-Specialized-v1.2:Nemotron-Pretraining-Fact-Seeking",
         weight=0.06, budget_tokens=900_000_000, domains=2,
         text_field="text", domain_key="uuid"),
    dict(name="math_textbooks", dataset="nvidia/Nemotron-Pretraining-Specialized-v1:Nemotron-Pretraining-Math-Textbooks",
         weight=0.05, budget_tokens=750_000_000, domains=2,
         text_field="text", domain_key="uuid"),
    dict(name="infinibyte", dataset="nvidia/Nemotron-Pretraining-Specialized-v1:Nemotron-Pretraining-InfiniByte-Reasoning",
         weight=0.04, budget_tokens=600_000_000, domains=2,
         text_field="text", domain_key="uuid"),
    dict(name="cc_code_hq", dataset="nvidia/Nemotron-CC-Code-v1:train",
         weight=0.03, budget_tokens=450_000_000, domains=2,
         text_field="text", domain_key="uuid"),
)

# Fully ungated continuation mix for environments waiting on NVIDIA's gated
# Common-Crawl corpus agreement.  This intentionally keeps the strongest open
# specialized Nemotron reasoning datasets while replacing only gated web/math/
# code sources.  The source corpora are all larger than the allocated exposure.
#
# Allocation for a 15B-token continuation:
#   40%  FineWeb-Edu-Dedup       6.00B  organic educational web / broad knowledge
#    5%  Cosmopedia-v2           0.75B  synthetic textbook-style exposition
#   20%  FinePDFs-Edu HQ filter  3.00B  papers/manuals/long technical documents
#   12%  MegaMath-Web-Pro-Max    1.80B  curated/refined continual-pretraining math web
#   10%  Nemotron RQA            1.50B  long-form STEM reasoning
#    6%  Nemotron Fact-Seeking   0.90B  factual QA / retrieval knowledge
#    4%  Nemotron InfiniByte     0.60B  cross-domain reasoning
#    3%  Nemotron Math Textbooks 0.45B  pedagogical mathematical exposition
#
# FineWeb-Edu-Dedup is already educational-quality filtered + globally deduped.
# FinePDFs-Edu is further filtered online because the nominal eng_Latn subset
# contains some mixed-language/OCR rows.  MegaMath-Web-Pro-Max is an ungated,
# already curated/refined math-web corpus released with OctoThinker.  Its authors
# report slightly better mid-training gains than FineMath-4plus, so we use the
# published corpus directly rather than stacking another brittle online filter.
_KNOWLEDGE_15B_UNGATED = (
    dict(name="web_edu_dedup", dataset="HuggingFaceTB/smollm-corpus:fineweb-edu-dedup",
         weight=0.40, budget_tokens=6_000_000_000, domains=8,
         text_field="text", domain_key="id"),
    dict(name="cosmopedia_v2", dataset="HuggingFaceTB/smollm-corpus:cosmopedia-v2",
         weight=0.05, budget_tokens=750_000_000, domains=3,
         text_field="text", domain_key="prompt"),
    dict(name="pdf_edu_hq", dataset="HuggingFaceFW/finepdfs-edu:eng_Latn",
         weight=0.20, budget_tokens=3_000_000_000, domains=4,
         text_field="text", domain_key="url",
         filters={
             "full_doc_lid": ["eq", "eng_Latn"],
             "page_average_lid_score": ["gte", 0.90],
             "token_count": ["gte", 512],
             "fw_edu_scores": ["max_gte", 2.0],
         }),
    dict(name="megamath_web_pro_max", dataset="OctoThinker/MegaMath-Web-Pro-Max",
         revision="0194035ca21e11ce60c4693fd2607e621dec9324",
         weight=0.12, budget_tokens=1_800_000_000, domains=4,
         text_field="text", domain_key="id"),
    dict(name="stem_rqa", dataset="nvidia/Nemotron-Pretraining-Specialized-v1:Nemotron-Pretraining-RQA",
         weight=0.10, budget_tokens=1_500_000_000, domains=3,
         text_field="text", domain_key="uuid"),
    dict(name="fact_seeking", dataset="nvidia/Nemotron-Pretraining-Specialized-v1.2:Nemotron-Pretraining-Fact-Seeking",
         weight=0.06, budget_tokens=900_000_000, domains=2,
         text_field="text", domain_key="uuid"),
    dict(name="infinibyte", dataset="nvidia/Nemotron-Pretraining-Specialized-v1:Nemotron-Pretraining-InfiniByte-Reasoning",
         weight=0.04, budget_tokens=600_000_000, domains=2,
         text_field="text", domain_key="uuid"),
    dict(name="math_textbooks", dataset="nvidia/Nemotron-Pretraining-Specialized-v1:Nemotron-Pretraining-Math-Textbooks",
         weight=0.03, budget_tokens=450_000_000, domains=2,
         text_field="text", domain_key="uuid"),
)

# SFT2 STEM-heavy open pretraining mix.  This is still causal next-token
# pretraining: no instruction/answer formatting and no benchmark labels.
# FineWeb is deliberately reduced from 40% to exactly 20%; the freed 20%
# goes to long-form educational PDFs, math web, STEM RQA, and math textbooks.
_SFT2_STEM_15B = (
    dict(name="web_edu_dedup", dataset="HuggingFaceTB/smollm-corpus:fineweb-edu-dedup",
         weight=0.20, budget_tokens=3_000_000_000, domains=8,
         text_field="text", domain_key="id"),
    dict(name="cosmopedia_v2", dataset="HuggingFaceTB/smollm-corpus:cosmopedia-v2",
         weight=0.05, budget_tokens=750_000_000, domains=3,
         text_field="text", domain_key="prompt"),
    dict(name="pdf_edu_hq", dataset="HuggingFaceFW/finepdfs-edu:eng_Latn",
         weight=0.30, budget_tokens=4_500_000_000, domains=4,
         text_field="text", domain_key="url",
         filters={
             "full_doc_lid": ["eq", "eng_Latn"],
             "page_average_lid_score": ["gte", 0.90],
             "token_count": ["gte", 512],
             "fw_edu_scores": ["max_gte", 2.0],
         }),
    dict(name="megamath_web_pro_max", dataset="OctoThinker/MegaMath-Web-Pro-Max",
         revision="0194035ca21e11ce60c4693fd2607e621dec9324",
         weight=0.17, budget_tokens=2_550_000_000, domains=4,
         text_field="text", domain_key="id"),
    dict(name="stem_rqa", dataset="nvidia/Nemotron-Pretraining-Specialized-v1:Nemotron-Pretraining-RQA",
         weight=0.12, budget_tokens=1_800_000_000, domains=3,
         text_field="text", domain_key="uuid"),
    dict(name="fact_seeking", dataset="nvidia/Nemotron-Pretraining-Specialized-v1.2:Nemotron-Pretraining-Fact-Seeking",
         weight=0.06, budget_tokens=900_000_000, domains=2,
         text_field="text", domain_key="uuid"),
    dict(name="infinibyte", dataset="nvidia/Nemotron-Pretraining-Specialized-v1:Nemotron-Pretraining-InfiniByte-Reasoning",
         weight=0.04, budget_tokens=600_000_000, domains=2,
         text_field="text", domain_key="uuid"),
    dict(name="math_textbooks", dataset="nvidia/Nemotron-Pretraining-Specialized-v1:Nemotron-Pretraining-Math-Textbooks",
         weight=0.06, budget_tokens=900_000_000, domains=2,
         text_field="text", domain_key="uuid"),
)

_STREAM_PRESETS = {
    "nemotron_knowledge_15b": _NEMOTRON_KNOWLEDGE_15B,
    "nemotron15b": _NEMOTRON_KNOWLEDGE_15B,
    "nemotron_15b": _NEMOTRON_KNOWLEDGE_15B,
    "knowledge_15b_ungated": _KNOWLEDGE_15B_UNGATED,
    "ungated_knowledge_15b": _KNOWLEDGE_15B_UNGATED,
    "ungated15b": _KNOWLEDGE_15B_UNGATED,
    "sft2_stem_15b": _SFT2_STEM_15B,
}


def _canonical_stream_specs(specs):
    """Validate/normalize weighted source specs and return JSON-safe copies."""
    if not isinstance(specs, (list, tuple)) or not specs:
        raise ValueError("stream mix must be a non-empty list")
    out = []
    total = 0.0
    names = set()
    for i, raw in enumerate(specs):
        if not isinstance(raw, dict):
            raise ValueError(f"stream mix entry {i} must be an object")
        item = dict(raw)
        item["name"] = str(item.get("name") or f"source_{i}").strip()
        item["dataset"] = str(item.get("dataset") or "").strip()
        item["weight"] = float(item.get("weight", 0.0))
        item["domains"] = max(1, int(item.get("domains", 2)))
        item["text_field"] = str(item.get("text_field") or "text").strip()
        item["message_field"] = str(item.get("message_field") or "").strip()
        item["domain_key"] = str(item.get("domain_key") or "uuid").strip()
        item["revision"] = str(item.get("revision") or "").strip()
        item["data_dir"] = str(item.get("data_dir") or "").strip()
        item["file_selector"] = str(item.get("file_selector") or "").strip()
        item["budget_tokens"] = int(item.get("budget_tokens", 0) or 0)
        item["filters"] = item.get("filters") or {}
        if not item["dataset"]:
            raise ValueError(f"stream mix entry {i} has no dataset")
        if not math.isfinite(item["weight"]) or item["weight"] <= 0:
            raise ValueError(f"stream mix entry {item['name']!r} has invalid weight")
        if item["name"] in names:
            raise ValueError(f"duplicate stream mix source name: {item['name']!r}")
        names.add(item["name"])
        total += item["weight"]
        out.append(item)
    if total <= 0:
        raise ValueError("stream mix weights sum to zero")
    for item in out:
        item["weight"] = item["weight"] / total
    return out


def _resolve_stream_specs(cfg):
    """Resolve TB_STREAM_PRESET or a custom TB_STREAM_MIX JSON array."""
    preset = str(getattr(cfg, "stream_preset", "") or "").strip().lower()
    raw_json = str(getattr(cfg, "stream_mix", "") or "").strip()
    if preset and raw_json:
        raise ValueError("set only one of TB_STREAM_PRESET or TB_STREAM_MIX")
    if preset:
        if preset not in _STREAM_PRESETS:
            raise ValueError(f"unknown TB_STREAM_PRESET={preset!r}; choices={sorted(_STREAM_PRESETS)}")
        specs = _canonical_stream_specs(_STREAM_PRESETS[preset])
    elif raw_json:
        try:
            specs = _canonical_stream_specs(json.loads(raw_json))
        except json.JSONDecodeError as exc:
            raise ValueError(f"TB_STREAM_MIX must be valid JSON: {exc}") from exc
    else:
        return []
    canonical = json.dumps(specs, sort_keys=True, separators=(",", ":"))
    cfg.stream_mix_signature = hashlib.sha256(canonical.encode()).hexdigest()[:16]
    return specs


# Role templates for SFT rendering. "" keeps the historical plain paragraphs;
# "chat-v1" prefixes each turn with its role so the model can tell a question
# from its answer: "User: q\n\nAssistant: a\n\nUser: q2\n\nAssistant: a2".
# User/system parts (with their labels) stay masked under TB_SFT_MASK; the
# "Assistant: " label sits inside the trained assistant part. Inference
# prompts end with "Assistant:" and stop on "\n\nUser:" or EOS.
SFT_TEMPLATES = {
    "": {},
    "chat-v1": {"system": "System: ", "user": "User: ", "assistant": "Assistant: "},
    # Tool-use conversations: chat-v1 plus the row's tool schemas (a leading
    # "Tools:" system part), assistant tool calls as compact JSON in
    # <tool_call></tool_call> after the assistant text, and tool results as
    # "Tool: " parts (structured JSON state the model must read back).
    "chat-tools-v1": {"system": "System: ", "user": "User: ", "assistant": "Assistant: ",
                      "tool": "Tool: "},
}
TOOL_TEMPLATES = frozenset({"chat-tools-v1"})


def _compact_json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _render_tool_calls(calls):
    """Assistant tool calls -> "<tool_call>{name, arguments}</tool_call>" lines."""
    out = []
    if isinstance(calls, str):
        try:
            calls = json.loads(calls)
        except ValueError:
            return out
    for call in calls if isinstance(calls, list) else ():
        if not isinstance(call, dict):
            continue
        fn = call.get("function") if isinstance(call.get("function"), dict) else call
        name = fn.get("name")
        if not isinstance(name, str) or not name:
            continue
        args = fn.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                pass
        out.append("<tool_call>" + _compact_json({"name": name, "arguments": args}) + "</tool_call>")
    return out


def _render_tools(tools):
    """Tool schemas -> compact [{name, description, parameters}] JSON."""
    if isinstance(tools, str):
        try:
            tools = json.loads(tools)
        except ValueError:
            return ""
    slim = []
    for tool in tools if isinstance(tools, list) else ():
        fn = tool.get("function") if isinstance(tool, dict) and isinstance(tool.get("function"), dict) else tool
        if isinstance(fn, dict) and isinstance(fn.get("name"), str):
            slim.append({k: fn[k] for k in ("name", "description", "parameters") if k in fn})
    return _compact_json(slim) if slim else ""


def _render_message_text(messages, roles=("user", "assistant"), template="", tools=None):
    """Render SFT chat messages into one pretraining document.

    Only role content is kept (reasoning_content excluded: the assistant
    content already carries the full trace); tool templates additionally
    render tool schemas, tool calls and tool results. Mirrors
    render_message_text in walker_continue_contract.py (trainer is standalone).
    """
    return "\n\n".join(text for _, text in _render_message_parts(messages, roles, template, tools))


def _render_message_parts(messages, roles=("user", "assistant"), template="", tools=None):
    """Split SFT chat messages into (role, text) parts in document order."""
    if template not in SFT_TEMPLATES:
        raise ValueError(f"unknown SFT template {template!r}")
    prefixes = SFT_TEMPLATES[template]
    with_tools = template in TOOL_TEMPLATES
    wanted = {str(role) for role in (roles or ("user", "assistant"))}
    if with_tools:
        wanted.add("tool")
    parts = []
    if with_tools:
        schema = _render_tools(tools)
        if schema:
            parts.append(("system", prefixes["system"] + "Tools: " + schema))
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", ""))
            if role not in wanted:
                continue
            content = message.get("content")
            text = content.strip() if isinstance(content, str) else ""
            if with_tools and role == "assistant":
                text = "\n".join(([text] if text else []) + _render_tool_calls(message.get("tool_calls")))
            if text:
                parts.append((role, prefixes.get(role, "") + text))
    return parts


def _sft_mask_enabled():
    """Assistant-only loss for SFT (message_field) sources. Off by default:
    every row below behaves exactly as the historical unmasked path."""
    return os.environ.get("TB_SFT_MASK", "0") == "1"


def masked_report_mean(losses, scored):
    """Mean of per-microbatch losses over those that scored a target.

    One host sync. A microbatch whose window is wholly SFT-masked carries no
    loss and no gradient, so it is left out of the reported mean (0.0 when
    none scored). A non-finite loss on a scored microbatch still surfaces.
    """
    scored = scored.to(dtype=torch.bool, device=losses.device)
    kept = torch.where(scored, losses.float(), torch.zeros_like(losses, dtype=torch.float32))
    return (kept.sum() / scored.sum().clamp(min=1)).item()


VAL_WINDOW_REDRAWS = 64


def redraw_unscored_window(mask, ix, T, n_tokens, generator, name=""):
    """Return a window start whose targets ``mask[ix+1 : ix+T+1]`` are not all masked.

    Answer-only validation scores only unmasked targets. A window lying wholly
    inside one long prompt would report NaN, so it is redrawn from the same
    generator. Banks without such windows draw exactly the historical starts.
    """
    for _ in range(VAL_WINDOW_REDRAWS):
        if not bool(mask[ix + 1:ix + T + 1].bool().all()):
            return ix
        ix = torch.randint(0, n_tokens - T - 1, (1,), generator=generator).item()
    if not bool(mask[ix + 1:ix + T + 1].bool().all()):
        return ix
    raise RuntimeError(
        f"validation bank {name!r}: no scored target in {VAL_WINDOW_REDRAWS} window draws "
        f"at T={T}; the bank is (almost) all prompt tokens")


def _sft_row_mask(encode_one, parts, joint_ids, sep="\n\n",
                  joint_text=None, offset_tok=None):
    """Token mask (1 = prompt/do-not-train) for one rendered SFT document.

    Two alignment strategies, tried in order; both refuse (None) rather than
    guess, so the caller falls back to an unmasked row with a counter:

    1. Offset mapping (needs joint_text + a fast-tokenizer twin): part
       character spans are known exactly (same join construction as the
       text), and each joint token trains iff its span lies ENTIRELY inside
       an assistant part. Boundary-straddling tokens mask (safe direction:
       at most drops an assistant token, never trains on prompt text). The
       twin's ids must equal the canonical joint_ids, else refuse.
    2. Incremental re-encoding: the document is re-encoded prefix by prefix
       and each piece sliced off (cuts fall on stripped edges, never inside
       a pre-token); concatenation must equal the joint tokenization exactly.
       Covers slow-tokenizer-only environments and well-behaved splits, but
       cannot survive context-sensitive trailing-whitespace branches (notably
       slow GPT-2's end-of-string handling), hence strategy 1 first.

    Assistant parts train (0); user/system parts and separators are masked
    (1); the appended terminal EOS is always unmasked (termination is
    learned).
    """
    if joint_text is not None and offset_tok is not None:
        keep = _sft_row_mask_from_offsets(
            joint_text, parts, joint_ids, offset_tok, sep)
        if keep is not None:
            return keep
    try:
        chunks, masks = [], []
        acc = ""
        done = 0
        for index, (role, text) in enumerate(parts):
            if index:
                acc += sep
                sep_ids = list(encode_one(acc))[done:]
                chunks.append(sep_ids)
                masks.append(1)
                done += len(sep_ids)
            acc += text
            ids = list(encode_one(acc))[done:]
            chunks.append(ids)
            masks.append(0 if str(role) == "assistant" else 1)
            done += len(ids)
        flat = [t for chunk in chunks for t in chunk]
        keep = [m for chunk, m in zip(chunks, masks) for _ in chunk]
        if list(flat) != list(joint_ids):
            return None
        return keep
    except Exception:
        return None


_offset_tok_cache = {}


def _offset_encoder_for(tok):
    """Fast-tokenizer twin of the stream tokenizer, for offset mapping only.

    Canonical token ids always come from the stream tokenizer itself; the
    twin supplies character offsets, and its ids must match exactly or the
    row falls back. Cached by tokenizer name; any failure (missing fast
    files, offline registry, exotic tokenizer) returns None and the caller
    uses incremental re-encoding instead. Never raises.
    """
    name = str(getattr(tok, "name_or_path", "") or "")
    if not name:
        return None
    if name in _offset_tok_cache:
        return _offset_tok_cache[name]
    try:
        from transformers import AutoTokenizer
        fast = AutoTokenizer.from_pretrained(
            name, local_files_only=True, use_fast=True)
        probe = fast("hi\nthere", add_special_tokens=False,
                     return_offsets_mapping=True)
        if "offset_mapping" not in probe:
            raise RuntimeError("fast tokenizer has no offsets")
        _offset_tok_cache[name] = fast
        return fast
    except Exception:
        _offset_tok_cache[name] = None
        return None


def _sft_row_mask_from_offsets(joint_text, parts, joint_ids, fast_tok, sep="\n\n"):
    """Offset-based mask: exact character spans, safe-direction boundaries.

    Returns None on ANY structural surprise (span/text mismatch, id mismatch,
    straddling handled by masking, empty parts): the caller falls back.
    """
    try:
        spans, pos = [], 0
        for index, (role, text) in enumerate(parts):
            if index:
                if joint_text[pos:pos + len(sep)] != sep:
                    return None
                pos += len(sep)
            if joint_text[pos:pos + len(text)] != text:
                return None
            spans.append((str(role), pos, pos + len(text)))
            pos += len(text)
        if pos != len(joint_text):
            return None
        enc = fast_tok(joint_text, add_special_tokens=False,
                       return_offsets_mapping=True)
        ids = list(enc["input_ids"])
        offs = list(enc["offset_mapping"])
        if ids != list(joint_ids) or len(ids) != len(offs):
            return None
        # Corrupt offset tables (drift, truncation, uncovered text) would
        # silently mislabel tokens: every joint character must be covered by
        # at least one token, except U+FFFD replacement chars which fast
        # GPT-2 may skip. Overlapping spans (fast GPT-2 double-emits some
        # quote bytes; slow ids agree) need no rejection: a token trains iff
        # its span lies ENTIRELY inside an assistant part, so overlaps can
        # only ever drop signal, never train on prompt text. Anything else
        # refuses rather than guess. O(tokens): only genuine gaps slice text.
        cursor = 0
        for start, end in offs:
            start, end = int(start), int(end)
            if end <= start or start < 0 or end > len(joint_text):
                return None
            if start > cursor and any(
                    ch != "\ufffd" for ch in joint_text[cursor:start]):
                return None
            cursor = max(cursor, end)
        if any(ch != "\ufffd" for ch in joint_text[cursor:]):
            return None
        keep = []
        for start, end in offs:
            start, end = int(start), int(end)
            owners = [role for (role, a, b) in spans if a <= start and end <= b]
            # Exactly one owner that is the assistant trains; every other
            # case (separator span, user/system span, boundary straddle)
            # masks. Straddlers mask rather than guess: at most drops an
            # assistant edge token, never trains on prompt text.
            keep.append(0 if owners == ["assistant"] else 1)
        return keep
    except Exception:
        return None


def _stream_text_health(value, min_chars=16):
    """Conservative corruption detector. Returns (clean_text, reason_or_None)."""
    if not isinstance(value, str):
        return None, "non_string"
    text = value.strip()
    if len(text) < min_chars:
        return None, "too_short"
    if "\x00" in text:
        return None, "nul_byte"
    n = max(1, len(text))
    bad_ctrl = sum((ord(ch) < 32 and ch not in "\n\r\t") for ch in text)
    if bad_ctrl / n > 0.001:
        return None, "control_chars"
    if text.count("\ufffd") / n > 0.005:
        return None, "replacement_chars"
    printable = sum(ch.isprintable() or ch in "\n\r\t" for ch in text)
    if printable / n < 0.98:
        return None, "nonprintable"
    return text, None


_STREAM_DATA_SUFFIXES = (".parquet", ".jsonl", ".jsonl.gz", ".jsonl.zst", ".json", ".arrow")


def _stream_route_key(value):
    return __import__("re").sub(r"[^a-z0-9]+", "", str(value).lower())


def _stream_data_suffix(path):
    lowered = str(path).lower()
    return next((suffix for suffix in _STREAM_DATA_SUFFIXES if lowered.endswith(suffix)), None)


def _resolve_stream_data_files(repo_files, selector):
    """Resolve one pinned logical subset without concatenating mirrored exports.

    "subset/prefix" narrows a subset directory to files whose name starts with
    prefix (e.g. "smol-magpie-ultra/train" excludes that subset's test split).
    """
    dir_key = file_key = None
    if "/" in str(selector):
        dir_part, file_part = str(selector).split("/", 1)
        dir_key, file_key = _stream_route_key(dir_part), _stream_route_key(file_part)
        if not dir_key or not file_key:
            raise ValueError(f"file selector {selector!r} needs both subset and file prefix")
    selector_key = _stream_route_key(selector)
    if not selector_key:
        raise ValueError("file selector must contain at least one letter or digit")
    candidates = []
    paths = [str(raw).replace("\\", "/") for raw in repo_files]
    for path in paths:
        suffix = _stream_data_suffix(path)
        if suffix is None:
            continue
        parts = [part for part in path.split("/") if part]
        components = {_stream_route_key(part) for part in parts[:-1]}
        filename = _stream_route_key(parts[-1]) if parts else ""
        if dir_key is not None:
            if dir_key in components and filename.startswith(file_key):
                candidates.append((path, suffix))
        elif selector_key in components or filename.startswith(selector_key):
            candidates.append((path, suffix))
    if not candidates:
        raise RuntimeError(
            f"no data files matched selector {selector!r}; repository sample={sorted(paths)[:12]}")
    for suffix in _STREAM_DATA_SUFFIXES:
        selected = sorted(path for path, kind in candidates if kind == suffix)
        if selected:
            return selected
    raise AssertionError("matched stream files had no supported representation")


@lru_cache(maxsize=16)
def _stream_repository_files(repo, revision):
    from huggingface_hub import HfApi
    return tuple(HfApi().list_repo_files(
        repo_id=str(repo), repo_type="dataset", revision=str(revision) or None))


_STREAM_FILTER_OPS = frozenset({"eq", "gte", "lte", "max_gte", "in", "not_in"})
_STREAM_DEDUP_MODES = frozenset({"", "first_user"})


def _first_user_key(messages):
    """64-bit key of a conversation's normalized first user turn (case,
    whitespace and punctuation folded), or None when there is no user turn."""
    for message in messages or []:
        if isinstance(message, dict) and message.get("role") == "user" and message.get("content"):
            text = re.sub(r"\s+", " ", str(message["content"]).lower())
            text = re.sub(r"[^a-z0-9 ]", "", text).strip()
            return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(),
                                  "little")
    return None


def _row_matches_filters(row, filters):
    """Small deterministic row-filter language used by curated stream specs."""
    for field, condition in (filters or {}).items():
        if not isinstance(condition, (list, tuple)) or len(condition) != 2:
            raise ValueError(f"invalid filter for {field}: {condition!r}")
        op, wanted = condition
        if op not in _STREAM_FILTER_OPS:
            # An unknown op used to fall through and accept every row.
            raise ValueError(f"unknown filter op {op!r} for {field}")
        value = row.get(field)
        try:
            if op == "eq" and value != wanted:
                return False
            if op == "in" and value not in wanted:
                return False
            if op == "not_in" and value in wanted:
                return False
            if op == "gte" and (value is None or float(value) < float(wanted)):
                return False
            if op == "lte" and (value is None or float(value) > float(wanted)):
                return False
            if op == "max_gte":
                vals = value if isinstance(value, (list, tuple)) else [value]
                vals = [float(v) for v in vals if v is not None]
                if not vals or max(vals) < float(wanted):
                    return False
        except (TypeError, ValueError):
            return False
    return True


def _parquet_shard_rows(shard_kwargs, builder=None):
    """Exact row count of one parquet shard from its file footers, or None.

    Only the footer is read. None whenever the shard is not plain parquet or
    the builder filters rows, so the caller counts by reading tables instead.
    """
    files = list(shard_kwargs.get("files") or [])
    groups = shard_kwargs.get("row_groups_list")
    groups = [None] * len(files) if groups is None else list(groups)
    config = getattr(builder, "config", None)
    if (not files or len(groups) != len(files) or getattr(config, "filters", None)
            or not all(str(path).endswith(".parquet") for path in files)):
        return None
    import fsspec
    import pyarrow.parquet as pq
    total = 0
    for path, wanted in zip(files, groups):
        with fsspec.open(str(path), "rb") as handle:
            meta = pq.ParquetFile(handle).metadata
        total += (meta.num_rows if wanted is None
                  else sum(meta.row_group(int(i)).num_rows for i in wanted))
    return total


def _stream_fast_position(ds, n):
    """``(dataset, shard, row)`` with ``dataset`` starting at raw row ``n``, or None.

    The slow resume replays every consumed row through the full Arrow ->
    Python conversion. Here earlier shards are passed over using their row
    counts, and inside the target shard the builder's own tables are dropped
    (or sliced) before conversion. Rows come from the same table generator and
    the same dataset pipeline, so they are identical to the replay; only plain
    unshuffled Arrow sources qualify.
    """
    import copy
    import inspect
    import datasets.iterable_dataset as hf_iterable
    base = getattr(ds, "_ex_iterable", None)
    if (type(base) is not hf_iterable.ArrowExamplesIterable or getattr(ds, "_shuffling", None)
            or getattr(ds, "epoch", 0) or getattr(ds, "_distributed", None)):
        return None
    shards = list(hf_iterable._split_gen_kwargs(base.kwargs, max_num_jobs=base.num_shards))
    generate = base.generate_tables_fn
    builder = getattr(generate, "__self__", None)
    shard, row = 0, int(n)
    while shard < len(shards) - 1:
        rows = _parquet_shard_rows(shards[shard], builder)
        if rows is None:
            rows = sum(len(table) for _, table in generate(**shards[shard]))
        if row < rows:
            break
        row -= rows
        shard += 1

    def positioned_tables(**_unused):
        skip = row
        for kwargs in shards[shard:]:
            for key, table in generate(**kwargs):
                if skip >= len(table):
                    skip -= len(table)
                    continue
                if skip:
                    table, skip = table.slice(skip), 0
                yield key, table

    # A fresh dataset with the original's settings: copy.copy would share the
    # original's __dict__ (IterableDataset.__setstate__), and the epoch
    # rollover must still iterate the untouched original from row 0.
    params = inspect.signature(type(ds).__init__).parameters
    settings = {name: getattr(ds, "_" + name) for name in params
                if name not in ("self", "ex_iterable") and hasattr(ds, "_" + name)}
    if "info" in settings and settings["info"] is not None:
        settings["info"] = copy.deepcopy(settings["info"])
    positioned = type(ds)(hf_iterable.ArrowExamplesIterable(
        positioned_tables, {"positioned": [True]}), **settings)
    return positioned, shard, row


class _BackgroundPosition:
    """Row iterator for a stream positioned at raw row n by a background thread.

    The thread uses the exact fast position when the stream qualifies and
    otherwise replays the n rows; either way it reads up to the first row past
    the cursor. Until then ``pending`` is True: the pump waits for it where
    checkpoint pauses are still answered (nothing is consumed yet, so a
    snapshot taken meanwhile is exact), and any other caller of next() blocks.
    """

    def __init__(self, ds, n, name, on_ready):
        self._ready = threading.Event()
        self._head, self._error, self._it = [], None, None
        self._on_ready = on_ready
        threading.Thread(target=self._position, args=(ds, int(n), name), daemon=True,
                         name=f"position-{name}").start()

    @property
    def pending(self):
        return not self._ready.is_set()

    def _position(self, ds, n, name):
        import itertools
        t0 = time.time()
        try:
            how = None
            try:
                found = _stream_fast_position(ds, n)
            except Exception as exc:
                found, how = None, f"{type(exc).__name__}: {str(exc)[:200]}"
            if found is None:
                print(f"STREAM fast skip unavailable for {name} "
                      f"({how or 'not a plain sequential Arrow stream'}); replaying {n:,} rows",
                      flush=True)
                self._it = iter(ds)
                for _ in itertools.islice(self._it, n):
                    pass
                how = "replayed"
            else:
                positioned, shard, row = found
                self._it = iter(positioned)
                how = f"shard {shard} row {row:,}"
            try:
                self._head = [next(self._it)]
            except StopIteration:
                self._head = []
            print(f"STREAM positioned {name} at raw row {n:,} ({how}) in {time.time() - t0:.1f}s",
                  flush=True)
        except BaseException as exc:  # surfaced by next() in the consumer
            self._error = exc
        finally:
            self._ready.set()
            self._on_ready()

    def __iter__(self):
        return self

    def __next__(self):
        self._ready.wait()
        if self._error is not None:
            raise self._error
        if self._head:
            return self._head.pop()
        return next(self._it)


class StreamDomain:
    __slots__ = ("idx",)

    def __init__(self, idx):
        self.idx = idx


class StreamSource:
    """Streaming tokenizer: HF dataset -> per-domain uint16 token
    buffers, filled by a background pump thread (start()) so batch()
    waits only for 'need', never for a full refill. Domains: stable hash of
    TB_STREAM_DOMAIN field (url netloc | dump | source_snapshot) -> 0..nd-1, so
    DoReMi weights keep meaning across resumes. Every source row is terminated
    by the tokenizer EOS token. This is essential for packed language-model
    training: without it, Walk/CNN/attention see unrelated documents as one
    continuous trajectory. Resume-safe via skip(n)."""

    def __init__(self, ds_name, tok, nd, domain_key="url", buf_tokens=1_000_000,
                 revision="", data_dir="", text_field="text", source_name="", row_filters=None,
                 file_selector="", message_field="", dedup="", template=""):
        from datasets import load_dataset
        self.nd = nd
        self.tok = tok
        self.ds_name = str(ds_name)
        self.source_name = str(source_name or ds_name)
        self.domain_key = domain_key
        self.text_field = str(text_field or "text")
        self.message_field = str(message_field or "").strip()
        self.row_filters = dict(row_filters or {})
        # dedup="first_user": skip a conversation whose normalized first user
        # turn was already seen this epoch (validation captures count too, so
        # training never repeats a held-out prompt). The seen-set is saved in
        # checkpoints and cleared on epoch rollover.
        self.dedup = str(dedup or "").strip()
        if self.dedup not in _STREAM_DEDUP_MODES:
            raise ValueError(f"unknown stream dedup mode {self.dedup!r}")
        if self.dedup and not self.message_field:
            raise ValueError("dedup='first_user' needs a message_field source")
        self._dedup_seen = set()
        self.template = str(template or "").strip()
        if self.template not in SFT_TEMPLATES:
            raise ValueError(f"unknown SFT template {self.template!r}")
        if self.template and not self.message_field:
            raise ValueError("an SFT template needs a message_field source")
        self.buf_tokens = buf_tokens
        self.revision = str(revision).strip()
        self.file_selector = str(file_selector or "").strip()
        self.route_sha256 = ""
        self.rejected_rows = 0
        self.rejected_reasons = collections.Counter()
        load_kwargs = {"split": "train", "streaming": True}
        if self.revision:
            load_kwargs["revision"] = self.revision
        if data_dir:
            load_kwargs["data_dir"] = data_dir
        try:
            if self.file_selector:
                if ":" in ds_name:
                    raise ValueError("file_selector cannot be combined with a dataset builder config")
                selected_files = _resolve_stream_data_files(
                    _stream_repository_files(ds_name, self.revision), self.file_selector)
                self.route_sha256 = hashlib.sha256(
                    "\n".join(selected_files).encode("utf-8")).hexdigest()[:16]
                print(f"STREAM_ROUTE source={self.source_name} selector={self.file_selector} "
                      f"files={len(selected_files)} sha256_16={self.route_sha256} "
                      f"sample={selected_files[:3]}", flush=True)
                self.ds = load_dataset(
                    ds_name, data_files={"train": selected_files}, **load_kwargs)
            elif ":" in ds_name:
                repo, cfg_name = ds_name.split(":", 1)
                self.ds = load_dataset(repo, cfg_name, **load_kwargs)
            else:
                self.ds = load_dataset(ds_name, **load_kwargs)
        except Exception as exc:
            hint = ""
            if str(ds_name).startswith("nvidia/"):
                hint = (" NVIDIA pretraining corpora can be gated: accept the dataset terms on "
                        "Hugging Face and export HF_TOKEN (or run `hf auth login`).")
            raise RuntimeError(f"cannot open streaming dataset {ds_name!r}.{hint} "
                               f"Original error: {_redact_hf_credential(exc)}") from None
        self.shuffle_buffer = max(0, int(os.environ.get("TB_STREAM_SHUFFLE_BUFFER", "0") or 0))
        shuffle_seed_base = int(os.environ.get("TB_STREAM_SHUFFLE_SEED", "0") or 0)
        source_seed_offset = int(hashlib.sha256(self.source_name.encode("utf-8")).hexdigest()[:8], 16)
        self.shuffle_seed = (
            (shuffle_seed_base + source_seed_offset) % (2 ** 31 - 1)
            if self.shuffle_buffer else 0
        )
        if self.shuffle_buffer:
            # HF IterableDataset.shuffle is deterministic for a fixed seed and
            # buffer.  A source-specific offset avoids aligned samples across
            # mixture components while preserving exact skip()-based resume.
            self.ds = self.ds.shuffle(seed=self.shuffle_seed, buffer_size=self.shuffle_buffer)
            print(
                f"STREAM_ORDER source={self.source_name} mode=buffered_shuffle "
                f"shuffle_buffer={self.shuffle_buffer} seed={self.shuffle_seed}",
                flush=True,
            )
        else:
            print(
                f"STREAM_ORDER source={self.source_name} mode=sequential shuffle_buffer=0",
                flush=True,
            )
        self.it = iter(self.ds)
        self.rows = [collections.deque() for _ in range(nd)]
        self.n_tokens = [0] * nd
        self.n_rows = [0] * nd
        self.consumed = 0
        self.raw_cursor = 0
        self.epoch = 0
        self.sft_unmasked_fallback = 0
        self.sft_masked_rows = 0
        self.cycle = os.environ.get("TB_STREAM_CYCLE", "1") == "1"
        self.heldout_ranges = []
        self.done = False
        self.cond = threading.Condition()
        self.thread = None
        self._pause_requested = threading.Event()
        self._paused = threading.Event()
        self.pack_format = "fineweb-eos-v1"
        # CodeStreamSource already used EOS before packing was versioned, while
        # legacy ordinary FineWeb buffers did not. Subclasses may override this.
        self.legacy_buffers_compatible = False

    def add_heldout_range(self, raw_start, raw_end):
        """Exclude a physical raw-row interval on every cycled epoch."""
        start, end = int(raw_start), int(raw_end)
        if end <= start:
            return
        merged = sorted(set(self.heldout_ranges + [(start, end)]))
        compact = []
        for lo, hi in merged:
            if compact and lo <= compact[-1][1]:
                compact[-1] = (compact[-1][0], max(compact[-1][1], hi))
            else:
                compact.append((lo, hi))
        self.heldout_ranges = compact

    def _is_heldout_row(self, raw_index):
        return any(lo <= raw_index < hi for lo, hi in self.heldout_ranges)

    def _next_training_row(self):
        """Read one non-held-out row, cycling a finite HF iterable exactly."""
        restarted = False
        while True:
            try:
                raw_index = self.raw_cursor
                row = next(self.it)
                self.raw_cursor += 1
            except StopIteration:
                if not self.cycle or restarted:
                    self.done = True
                    return None
                self.epoch += 1
                self.raw_cursor = 0
                getattr(self, "_dedup_seen", set()).clear()
                self.it = iter(self.ds)
                self.done = False
                restarted = True
                print(f"STREAM epoch rollover -> {self.epoch}; held-out ranges remain excluded",
                      flush=True)
                continue
            if self._is_heldout_row(raw_index):
                continue
            return row

    def _append_document_eos(self, ids):
        """Return one uint16 row with exactly one terminal EOS delimiter."""
        import numpy as np
        eos = getattr(self.tok, "eos_token_id", None)
        if eos is None:
            raise RuntimeError("stream tokenizer has no eos_token_id; document packing is unsafe")
        values = list(ids)
        if not values or int(values[-1]) != int(eos):
            values.append(int(eos))
        return np.asarray(values, dtype="uint16")

    def start(self):
        """Background pump: keeps every domain buffered to buf_tokens so batch()
        waits only for 'need', never for a full refill (which blocks seconds)."""
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        try:
            while True:
                with self.cond:
                    # A stream still being positioned in the background is
                    # waited for here, where checkpoint pauses stay answerable.
                    while (self._pause_requested.is_set()
                           or getattr(self.it, "pending", False)):
                        if self._pause_requested.is_set():
                            self._paused.set()
                        self.cond.wait()
                    self._paused.clear()
                    while (min(self.n_tokens) >= self.buf_tokens
                           and not self._pause_requested.is_set()):
                        if self.done:
                            return
                        self.cond.wait()
                    if self._pause_requested.is_set():
                        continue
                self._pump(32)
                with self.cond:
                    self.cond.notify_all()
        except Exception as e:
            print(f"STREAM thread error: {e!r}", flush=True)
            with self.cond:
                self.done = True
                self.cond.notify_all()

    def _extract_parts(self, row, *, count_rejection=True):
        """Return (text, parts) for one raw row; (None, None) when rejected.

        Rejection counters match the historical _extract_text exactly.
        ``parts`` is the role-tagged message list for SFT (message_field)
        rows and None otherwise; it is only populated when masking is
        enabled, so the default path pays nothing extra.
        """
        if not _row_matches_filters(row, self.row_filters):
            if count_rejection:
                self.rejected_rows += 1
                self.rejected_reasons["row_filter"] += 1
            return None, None
        if getattr(self, "dedup", "") == "first_user":
            key = _first_user_key(row.get(self.message_field))
            if key is not None:
                if key in self._dedup_seen:
                    if count_rejection:
                        self.rejected_rows += 1
                        self.rejected_reasons["duplicate"] += 1
                    return None, None
                self._dedup_seen.add(key)
        if self.message_field:
            template = getattr(self, "template", "")
            parts = _render_message_parts(row.get(self.message_field), template=template,
                                          tools=row.get("tools") if template in TOOL_TEMPLATES else None)
            if not parts:
                if count_rejection:
                    self.rejected_rows += 1
                    self.rejected_reasons["empty_render"] += 1
                return None, None
            raw = "\n\n".join(text for _, text in parts)
        else:
            parts = None
            raw = row.get(self.text_field)
        text, reason = _stream_text_health(raw)
        if reason is not None:
            if count_rejection:
                self.rejected_rows += 1
                self.rejected_reasons[reason] += 1
            return None, None
        if self.message_field and _sft_mask_enabled():
            return text, parts
        return text, None

    def _extract_text(self, row, *, count_rejection=True):
        text, _ = self._extract_parts(row, count_rejection=count_rejection)
        return text

    def validation_identity(self):
        return {
            "stream": self.ds_name,
            "stream_revision": self.revision,
            "stream_domain": self.domain_key,
            "stream_text_field": self.text_field,
        "stream_message_field": getattr(self, "message_field", ""),
            "stream_filters": self.row_filters,
            "stream_dedup": getattr(self, "dedup", ""),
            "stream_template": getattr(self, "template", ""),
            "stream_file_selector": self.file_selector,
            "stream_route_sha256": self.route_sha256,
            "stream_shuffle_buffer": self.shuffle_buffer,
            "stream_shuffle_seed": self.shuffle_seed,
            "sft_mask": "1" if _sft_mask_enabled() else "",
        }

    def probe_rows(self, n=8, max_scan=None):
        """Consume a temporary stream instance and report accepted/rejected rows."""
        n = max(1, int(n))
        max_scan = max(n, int(max_scan or n * 50))
        accepted, scanned = [], 0
        while len(accepted) < n and scanned < max_scan:
            row = self._next_training_row()
            if row is None:
                break
            scanned += 1
            text = self._extract_text(row, count_rejection=True)
            if text is None:
                continue
            ids = self.tok(text, add_special_tokens=False)["input_ids"]
            if ids and isinstance(ids[0], list):
                ids = ids[0]
            accepted.append({
                "chars": len(text),
                "tokens": len(ids),
                "excerpt": text[:240].replace("\n", "\\n").replace("\r", "\\r"),
            })
        return {
            "source": self.source_name,
            "dataset": self.ds_name,
            "scanned": scanned,
            "accepted": len(accepted),
            "rejected": int(self.rejected_rows),
            "reasons": dict(self.rejected_reasons),
            "rows": accepted,
        }

    def _domain_of(self, row):
        key = (row.get(self.domain_key) or row.get("uuid") or row.get("url")
               or row.get("id") or (row.get("metadata") or {}).get("warc_id") or "")
        if self.domain_key == "url" and key:
            from urllib.parse import urlparse
            try:
                key = urlparse(key).netloc
            except Exception:
                pass
        return int(hashlib.md5(str(key).encode()).hexdigest(), 16) % self.nd

    def _row_entry_ids(self, entry):
        return entry[0] if isinstance(entry, tuple) else entry

    def _row_entry_mask(self, entry):
        return entry[1] if isinstance(entry, tuple) else None

    def _pump(self, n):
        import numpy as np
        rows = []
        for _ in range(n):
            row = self._next_training_row()
            if row is None:
                break
            text, parts = self._extract_parts(row)
            if text is not None:
                rows.append((row, text, parts))
        if not rows:
            return

        def _encode_one(text):
            ids = self.tok(text, add_special_tokens=False)["input_ids"]
            if ids and isinstance(ids[0], list):
                ids = ids[0]
            return list(ids)

        ids_batch = self.tok([text for _, text, _ in rows], add_special_tokens=False)["input_ids"]
        mask_on = _sft_mask_enabled() and bool(self.message_field)
        for (row, _text, parts), ids in zip(rows, ids_batch):
            a = self._append_document_eos(ids)
            if a.size < 8:
                continue
            m = None
            if mask_on and parts:
                keep = _sft_row_mask(_encode_one, parts, list(ids),
                                     joint_text=_text,
                                     offset_tok=_offset_encoder_for(self.tok))
                if keep is None:
                    self.sft_unmasked_fallback += 1
                    if self.sft_unmasked_fallback in (1, 10, 100, 1000, 10000, 100000):
                        print(f"SFT mask fallback x{self.sft_unmasked_fallback:,} "
                              f"on {getattr(self, 'source_name', '?')} (row trains unmasked)", flush=True)
                else:
                    full = np.ones(len(a), dtype=np.uint8)
                    full[:min(len(a), len(keep))] = np.asarray(
                        keep[:len(a)], dtype=np.uint8)
                    if len(a) > len(ids):
                        full[len(ids):] = 0  # appended EOS trains: learn to stop
                    m = full
            d = self._domain_of(row)
            if self.n_tokens[d] >= self.buf_tokens:
                d = min(range(self.nd), key=lambda i: self.n_tokens[i])
            self.rows[d].append((a, m))
            if m is not None:
                self.sft_masked_rows += 1
            self.n_tokens[d] += int(a.size)
            self.n_rows[d] += 1
            self.consumed += 1

    def _notify_stream(self):
        cond = getattr(self, "cond", None)
        if cond is not None:
            with cond:
                cond.notify_all()

    def skip(self, n):
        """Move the cursor to raw row n.

        By default the positioning runs in a background thread
        (_BackgroundPosition: exact fast skip, else replay), so startup does
        not wait for it and a checkpoint can pause the pump meanwhile.
        TB_STREAM_FAST_SKIP=0 replays synchronously, as before.
        """
        n = int(n)
        if (n > 0 and os.environ.get("TB_STREAM_FAST_SKIP", "1") == "1"
                and not getattr(self, "shuffle_buffer", 0) and getattr(self, "ds", None) is not None):
            self.it = _BackgroundPosition(self.ds, n, getattr(self, "source_name", "?"),
                                          self._notify_stream)
            self.raw_cursor = n
            self.consumed = n
            return
        self.raw_cursor = 0
        for _ in range(n):
            try:
                next(self.it)
                self.raw_cursor += 1
            except StopIteration:
                self.done = True
                return
        self.consumed = n

    def skip_more(self, n):
        """Advance the iterator n rows past the current cursor WITHOUT resetting
        consumed (used to jump over a previously-captured held-out region)."""
        for _ in range(n):
            try:
                next(self.it)
                self.raw_cursor += 1
            except StopIteration:
                self.done = True
                return
        self.consumed += n

    def checkpoint_state(self, timeout=600):
        """Compact, exact token-buffer snapshot for a deterministic restart."""
        import numpy as np
        should_resume = self.thread is not None and self.thread.is_alive()
        if should_resume:
            self._pause_requested.set()
            with self.cond:
                self.cond.notify_all()
            if not self._paused.wait(timeout):
                self._pause_requested.clear()
                raise RuntimeError("STREAM did not pause before checkpointing")
        try:
            with self.cond:
                buffers = []
                mask_buffers = []
                for rows in self.rows:
                    ids_list = [self._row_entry_ids(e) for e in rows]
                    msk_list = [self._row_entry_mask(e) for e in rows]
                    buffers.append(
                        np.concatenate(tuple(ids_list)).astype(np.uint16, copy=True)
                        if ids_list else np.empty(0, dtype=np.uint16))
                    flat_m = [m if m is not None else np.zeros_like(r, dtype=np.uint8)
                              for r, m in zip(ids_list, msk_list)]
                    mask_buffers.append(
                        np.concatenate(tuple(flat_m)).astype(np.uint8, copy=True)
                        if flat_m else np.empty(0, dtype=np.uint8))
                return {
                    "version": 3,
                    "pack_format": self.pack_format,
                    "epoch": int(self.epoch),
                    "dedup_seen": (np.fromiter(self._dedup_seen, dtype=np.uint64,
                                               count=len(self._dedup_seen))
                                   if getattr(self, "dedup", "") else None),
                    "raw_cursor": int(self.raw_cursor),
                    "consumed": int(self.consumed),
                    "buffers": buffers,
                    # SFT prompt masks, parallel to buffers (zeros = unmasked).
                    # Optional: pre-mask checkpoints restore unmasked, and old
                    # readers ignore the unknown key.
                    "mask_buffers": mask_buffers,
                    "n_rows": [int(n) for n in self.n_rows],
                    "heldout_ranges": [list(pair) for pair in self.heldout_ranges],
                    "done": bool(self.done),
                }
        finally:
            if should_resume:
                self._pause_requested.clear()
                with self.cond:
                    self.cond.notify_all()

    def restore_checkpoint_state(self, state):
        """Restore buffers after skip(raw_cursor) has rebuilt the HF iterator."""
        import numpy as np
        if getattr(self, "dedup", "") and state.get("dedup_seen") is not None:
            # skip() rebuilt the cursor without rendering rows; the seen-set
            # comes from the checkpoint so resumed epochs stay duplicate-free.
            self._dedup_seen = {int(key) for key in state["dedup_seen"]}
        buffers = state.get("buffers")
        expected_nd = int(getattr(self, "nd", len(buffers) if isinstance(buffers, list) else 0))
        if not isinstance(buffers, list) or len(buffers) != expected_nd:
            raise ValueError("invalid STREAM checkpoint buffers")
        saved_format = state.get("pack_format")
        compatible = saved_format == self.pack_format or (
            saved_format is None and self.legacy_buffers_compatible)
        discarded_tokens = 0
        repacked = False
        if not compatible:
            if os.environ.get("TB_STREAM_REPACK_ON_RESUME", "0") != "1":
                raise RuntimeError(
                    f"stream checkpoint packing {saved_format or 'legacy-concat-v1'!r} "
                    f"cannot resume as {self.pack_format!r}: export "
                    "TB_STREAM_REPACK_ON_RESUME=1 once to discard only the "
                    "incompatible prefetched buffers while preserving model, "
                    "optimizer, schedule, cumulative tokens, and raw cursor")
            discarded_tokens = sum(int(len(buf)) for buf in buffers)
            buffers = [np.empty(0, dtype=np.uint16) for _ in range(expected_nd)]
            repacked = True
        self.nd = expected_nd
        with self.cond:
            masks = state.get("mask_buffers")
            if masks is not None and (not isinstance(masks, list) or len(masks) != expected_nd):
                raise ValueError("invalid STREAM checkpoint mask buffers")
            rebuilt = []
            for bi, buf in enumerate(buffers):
                ids = np.asarray(buf, dtype=np.uint16)
                mk = None
                if masks is not None:
                    mk = np.asarray(masks[bi], dtype=np.uint8)
                    if len(mk) != len(ids):
                        raise ValueError("STREAM checkpoint ids/mask length mismatch")
                    if not bool(mk.any()):
                        mk = None
                rebuilt.append(collections.deque([(ids, mk)]) if len(ids) else collections.deque())
            self.rows = rebuilt
            self.n_tokens = [int(len(buf)) for buf in buffers]
            self.n_rows = [int(n) for n in state.get("n_rows", [0] * expected_nd)]
            self.epoch = int(state.get("epoch", 0))
            for pair in state.get("heldout_ranges", []):
                if isinstance(pair, (list, tuple)) and len(pair) == 2:
                    self.add_heldout_range(pair[0], pair[1])
            self.raw_cursor = int(state["raw_cursor"])
            self.consumed = int(state.get("consumed", 0))
            # A finite iterator may have been checkpointed exactly at EOF.
            # Cycling resumes from the next epoch after buffered tokens drain.
            self.done = bool(state.get("done", False)) and not self.cycle
            self.cond.notify_all()
        return {"discarded_tokens": discarded_tokens, "repacked": repacked}

    def capture_val(self, n_tokens, buf_rows=32):
        """Hold out the next n_tokens of stream rows for a held-out val set.
        Reads synchronously (call BEFORE start()), tokenizes, and increments
        self.consumed so resume-skip stays aligned; the rows never enter the
        domain buffers, so training never sees them. Returns
        (tensor, masks-or-None, raw_rows): masks mark SFT prompt tokens when
        TB_SFT_MASK=1 (answer-only validation); raw_rows = iterator advances
        consumed, so a resume that predates the capture can skip_more(raw_rows)
        to jump the whole region exactly."""
        import numpy as np
        ids_all = []
        msk_all = []
        tot = 0
        raw_n = 0
        mask_on = _sft_mask_enabled() and bool(self.message_field)

        def _encode_one(text):
            ids = self.tok(text, add_special_tokens=False)["input_ids"]
            if ids and isinstance(ids[0], list):
                ids = ids[0]
            return list(ids)

        while tot < n_tokens:
            rows = []
            for _ in range(buf_rows):
                try:
                    row = next(self.it)
                    raw_n += 1
                    self.raw_cursor += 1
                except StopIteration:
                    self.done = True
                    break
                text, parts = self._extract_parts(row)
                if text is not None:
                    rows.append((text, parts))
            if not rows:
                break
            ids_batch = self.tok([text for text, _ in rows], add_special_tokens=False)["input_ids"]
            for (text, parts), ids in zip(rows, ids_batch):
                a = self._append_document_eos(ids)
                if a.size < 8:
                    continue
                m = None
                if mask_on and parts:
                    keep = _sft_row_mask(_encode_one, parts, list(ids),
                                         joint_text=text,
                                         offset_tok=_offset_encoder_for(self.tok))
                    if keep is None:
                        self.sft_unmasked_fallback += 1
                        if self.sft_unmasked_fallback in (1, 10, 100, 1000, 10000, 100000):
                            print(f"SFT mask fallback x{self.sft_unmasked_fallback:,} "
                                  f"on {getattr(self, 'source_name', '?')} (row trains unmasked)", flush=True)
                    else:
                        full = np.ones(len(a), dtype=np.uint8)
                        full[:min(len(a), len(keep))] = np.asarray(
                            keep[:len(a)], dtype=np.uint8)
                        if len(a) > len(ids):
                            full[len(ids):] = 0
                        m = full
                ids_all.append(a)
                msk_all.append(m)
                tot += int(a.size)
                self.consumed += 1
            if self.done:
                break
        if not ids_all:
            raise RuntimeError("FW val capture exhausted the stream (no usable rows)")
        masks = None
        if any(m is not None for m in msk_all):
            masks = torch.from_numpy(np.concatenate([
                m if m is not None else np.zeros_like(
                    a_ref, dtype=np.uint8)
                for a_ref, m in zip(ids_all, msk_all)])).to(torch.uint8)
        return torch.from_numpy(np.concatenate(ids_all)).long(), masks, raw_n

    def batch(self, d, T, B):
        import numpy as np
        # Consume exactly the tokens represented by the returned windows.
        # The historical +64 jitter margin silently discarded 64 stream tokens
        # per microbatch (1.54% at T=4096) and made token accounting differ
        # from source consumption. Shuffling and document boundaries already
        # provide varying starts, so contiguous windows lose no useful entropy.
        need = (T + 1) * B
        with self.cond:
            while self.n_tokens[d] < need:
                if self.done:
                    raise RuntimeError(
                        f'STREAM exhausted in domain {d} (rows={self.n_rows[d]}, total={self.consumed})')
                self.cond.wait()
            parts, mparts, n = [], [], 0
            while n < need:
                entry = self.rows[d].popleft()
                row, mk = self._row_entry_ids(entry), self._row_entry_mask(entry)
                self.n_tokens[d] -= int(row.size)
                if n + int(row.size) <= need:
                    parts.append(row)
                    mparts.append(mk)
                    n += int(row.size)
                else:
                    take = need - n
                    parts.append(row[:take])
                    mparts.append(mk[:take] if mk is not None else None)
                    self.rows[d].appendleft((row[take:], mk[take:] if mk is not None else None))
                    self.n_tokens[d] += int(row.size) - take
                    n = need
            self.cond.notify_all()
        buf = np.concatenate(parts)
        mbuf = np.concatenate([
            m if m is not None else np.zeros_like(p, dtype=np.uint8)
            for p, m in zip(parts, mparts)])
        ids = torch.from_numpy(buf).long()
        ms = torch.from_numpy(mbuf).bool()
        xs, ys = [], []
        seg = len(ids) // B
        for b in range(B):
            lo = b * seg
            ix = lo
            xs.append(ids[ix:ix + T])
            yw = ids[ix + 1:ix + T + 1].clone()
            # Assistant-only loss: never train on prompt (SFT mask) tokens.
            # Unmasked rows carry all-zero masks, so this is a no-op for them.
            yw[ms[ix + 1:ix + T + 1]] = -100
            ys.append(yw)
        return torch.stack(xs), torch.stack(ys)



class _SmoothWeightedRoundRobin:
    """Deterministic stratified source rotation (nginx-style smooth WRR).

    Independent multinomial draws make most 15-window steps miss 2-3 small
    sources entirely while oversampling the big ones — pure composition noise
    on the step loss. Rotation keeps the exact long-run weights while dealing
    every step a near-identical source hand (4-5 wiki, 2-3 megamath, 2
    science-vendor, 2 RQA, 1 math, 1 textbooks, 1 cosmopedia, 1 fact-seeking,
    infinibyte every ~2nd step). Marginals equal the weights, so gradients
    stay unbiased; only the ordering changes. Pure arithmetic (no RNG), with
    serializable state for exact resume.
    """

    def __init__(self, weights):
        total = sum(float(w) for w in weights)
        if total <= 0.0 or not math.isfinite(total):
            raise ValueError("round-robin needs positive weights")
        self.weights = [float(w) / total for w in weights]
        self.current = [0.0] * len(self.weights)

    def pick(self):
        best = 0
        for i, w in enumerate(self.weights):
            self.current[i] += w
            if self.current[i] > self.current[best]:
                best = i
        self.current[best] -= 1.0
        return best

    def state_dict(self):
        return {"weights": list(self.weights), "current": list(self.current)}

    def load_state_dict(self, state):
        current = (state or {}).get("current")
        if (not isinstance(current, list) or len(current) != len(self.weights)
                or not all(isinstance(v, (int, float)) and math.isfinite(v)
                           for v in current)):
            raise ValueError("round-robin scheduler state mismatch; refusing to resume it")
        self.current = [float(v) for v in current]


class MixedStreamSource:
    """Weighted collection of independently resumable HF streams.

    Source choice follows explicit weights via stratified rotation (every step
    sees the same source hand, not a multinomial lottery); each source still
    hashes rows into a few local domains to avoid building every training
    window from one site or document family.  Every child stream owns its
    cursor/buffers, so checkpoint restore is exact rather than re-sampling
    from the beginning.
    """

    def __init__(self, specs, tok, buf_tokens=1_000_000, validation_source=0):
        self.specs = _canonical_stream_specs(specs)
        self.tok = tok
        self.sources = []
        for spec in self.specs:
            self.sources.append(StreamSource(
                spec["dataset"], tok, spec["domains"],
                domain_key=spec["domain_key"], buf_tokens=buf_tokens,
                revision=spec.get("revision", ""), data_dir=spec.get("data_dir", ""),
                text_field=spec.get("text_field", "text"), source_name=spec["name"],
                row_filters=spec.get("filters") or {},
                file_selector=spec.get("file_selector", ""),
                message_field=spec.get("message_field", ""),
                dedup=spec.get("dedup", ""),
                template=spec.get("template", ""),
            ))
        self.weights = [float(spec["weight"]) for spec in self.specs]
        self.nd = len(self.sources)
        self.sample_counts = [0] * self.nd
        self.scheduler = _SmoothWeightedRoundRobin(self.weights)
        self.validation_source = max(0, min(int(validation_source), self.nd - 1))
        canonical = json.dumps(self.specs, sort_keys=True, separators=(",", ":"))
        if _sft_mask_enabled():
            # Assistant-only loss changes the training objective: masked runs
            # form a new stream lineage (fresh cursors via dataset override)
            # with their own validation banks. Unmasked runs keep the exact
            # historical signature.
            canonical += json.dumps({"sft_mask": 1}, sort_keys=True, separators=(",", ":"))
        self.signature = hashlib.sha256(canonical.encode()).hexdigest()[:16]
        self.pack_format = "weighted-mix-eos-v1"
        self.thread = None
        self.last_src = None  # diagnostic only: last sample_batch source index
        # Set by resume when validate_data_manifest accepted an append-only
        # mix extension (TB_RESUME_DATASET=extend).
        self.allow_extend = False
        self.rerendered = set()   # SFT sources whose role template changed

    @property
    def primary(self):
        return self.sources[self.validation_source]

    @property
    def raw_cursor(self):
        return self.primary.raw_cursor

    @property
    def n_tokens(self):
        return [n for src in self.sources for n in src.n_tokens]

    @property
    def consumed(self):
        return sum(src.consumed for src in self.sources)

    def validation_identity(self):
        ident = self.primary.validation_identity()
        ident.update({
            "stream_mix_signature": self.signature,
            "stream_validation_source": self.specs[self.validation_source]["name"],
        })
        return ident

    def start(self):
        for src in self.sources:
            if src.thread is None:
                src.start()
        self.thread = True

    def sample_batch(self, T, B):
        if B != 1:
            xs, ys = [], []
            for _ in range(B):
                x, y = self.sample_batch(T, 1)
                xs.append(x[0]); ys.append(y[0])
            return torch.stack(xs), torch.stack(ys)
        src_i = self.scheduler.pick()
        self.sample_counts[src_i] += 1
        # Remember the last picked source for per-microbatch loss attribution
        # (TB_DEBUG_MICROBATCH=1). Read-only diagnostic; never enters tensors.
        self.last_src = int(src_i)
        src = self.sources[src_i]
        return src.batch(random.randrange(src.nd), T, 1)

    def batch(self, d, T, B):
        # Explicit source-index path, useful for diagnostics. Normal training
        # calls sample_batch() so weights, not source count, control exposure.
        src = self.sources[int(d) % self.nd]
        self.last_src = int(d) % self.nd
        return src.batch(random.randrange(src.nd), T, B)

    def add_heldout_range(self, raw_start, raw_end):
        self.primary.add_heldout_range(raw_start, raw_end)

    def skip_more(self, n):
        self.primary.skip_more(n)

    def capture_val(self, n_tokens, buf_rows=32):
        return self.primary.capture_val(n_tokens, buf_rows=buf_rows)

    def checkpoint_state(self, timeout=600):
        return {
            "version": 4,
            "kind": "weighted-mix",
            "signature": self.signature,
            "validation_source": self.validation_source,
            "scheduler": self.scheduler.state_dict(),
            "sources": [
                {"name": spec["name"], "dataset": spec["dataset"],
                 "state": src.checkpoint_state(timeout=timeout)}
                for spec, src in zip(self.specs, self.sources)
            ],
        }

    def restore_checkpoint_state(self, state):
        if not isinstance(state, dict) or state.get("kind") != "weighted-mix":
            raise RuntimeError("mixed stream cannot restore a legacy single-stream cursor; use TB_RESUME_DATASET=override")
        saved = state.get("sources") or []
        if state.get("signature") != self.signature:
            if not self.allow_extend:
                raise RuntimeError(
                    f"mixed stream signature changed ({state.get('signature')} -> {self.signature}); "
                    "use TB_RESUME_DATASET=override for an intentional new mixture")
            return self._restore_extended(saved)
        if len(saved) != len(self.sources):
            raise RuntimeError("mixed stream checkpoint source count mismatch")
        # Rotation position restores exactly (signature match above guarantees
        # identical weights); old files predate the scheduler and restart its
        # rotation from zero, which only reorders future picks.
        scheduler_state = state.get("scheduler")
        if isinstance(scheduler_state, dict):
            try:
                self.scheduler.load_state_dict(scheduler_state)
            except ValueError:
                print("resume: rotation scheduler state incompatible; restarting rotation",
                      flush=True)
        total_discarded = 0
        for spec, src, old in zip(self.specs, self.sources, saved):
            if old.get("name") != spec["name"] or old.get("dataset") != spec["dataset"]:
                raise RuntimeError(f"mixed stream source mismatch at {spec['name']}")
            child = old.get("state") or {}
            src.skip(int(child.get("raw_cursor", 0)))
            result = src.restore_checkpoint_state(child)
            total_discarded += int(result.get("discarded_tokens", 0))
        return {"discarded_tokens": total_discarded, "repacked": bool(total_discarded)}

    def _restore_extended(self, saved):
        """Append-only extension: every saved source resumes its exact cursor
        (matched by name); sources new to the mix start fresh. Weights changed,
        so the rotation scheduler restarts (it only orders future picks)."""
        by_name = {str(old.get("name")): old for old in saved}
        current = {spec["name"] for spec in self.specs}
        dropped = sorted(set(by_name) - current)
        if dropped:
            raise RuntimeError(f"mix extension cannot drop sources: {dropped}")
        total_discarded = 0
        for spec, src in zip(self.specs, self.sources):
            old = by_name.get(spec["name"])
            if old is None:
                print(f"MIX extend: {spec['name']} is new and starts at a fresh cursor", flush=True)
                continue
            if old.get("dataset") != spec["dataset"]:
                raise RuntimeError(f"mix extension changed the dataset of {spec['name']}")
            child = dict(old.get("state") or {})
            src.skip(int(child.get("raw_cursor", 0)))
            if spec["name"] in getattr(self, "rerendered", ()):
                # Prefetched rows were rendered in the old format: drop them
                # (the cursor still moves past them) rather than mix formats.
                import numpy as np
                buffers = child.get("buffers") or []
                total_discarded += sum(int(len(buf)) for buf in buffers)
                child["buffers"] = [np.empty(0, dtype=np.uint16) for _ in buffers]
                child["mask_buffers"] = None
                child["n_rows"] = [0] * len(buffers)
                print(f"MIX extend: {spec['name']} re-rendered; cursor kept, old-format "
                      "prefetch dropped", flush=True)
            result = src.restore_checkpoint_state(child)
            total_discarded += int(result.get("discarded_tokens", 0))
        print(f"MIX extend: kept {len(by_name)} exact cursors; rotation scheduler restarted "
              "for the new weights", flush=True)
        return {"discarded_tokens": total_discarded, "repacked": bool(total_discarded)}

    def probe(self, rows_per_source=4):
        # Probing uses this object's fresh child iterators; never call it on a
        # live training object because it intentionally advances those cursors.
        return [src.probe_rows(rows_per_source) for src in self.sources]

    def health_summary(self):
        """Cheap live diagnostics for source exposure and rejected rows."""
        out = []
        for i, (spec, src) in enumerate(zip(self.specs, self.sources)):
            out.append({
                'name': spec['name'],
                'weight': float(spec['weight']),
                'sampled_batches': int(self.sample_counts[i]),
                'raw_rows': int(src.raw_cursor),
                'accepted_rows': int(src.consumed),
                'rejected_rows': int(src.rejected_rows),
                'reasons': dict(src.rejected_reasons),
                'buffered_tokens': int(sum(src.n_tokens)),
            })
        return out

class CodeStreamSource(StreamSource):
    """Repository-aware HF streaming code source with exact resume semantics.

    The source keeps the same cursor/token-buffer checkpoint format as
    ``StreamSource``.  Each source file is wrapped in plain-text repository/file
    markers (no tokenizer change), and files from the same repository that occur
    in one pump are packed together before tokenization.  Repository identity is
    used for domain hashing, so long windows preferentially stay within a stable
    repository bucket instead of mixing arbitrary files.
    """

    def __init__(self, ds_name, tok, nd, buf_tokens=1_000_000, revision='', data_dir='',
                 text_field='content', repo_field='repo_name', path_field='path',
                 lang_field='language', group_repo=True, language_filter=''):
        self.text_field = str(text_field or 'content')
        self.repo_field = str(repo_field or 'repo_name')
        self.path_field = str(path_field or 'path')
        self.lang_field = str(lang_field or 'language')
        self.group_repo = bool(group_repo)
        self.language_filter = {
            x.strip().lower() for x in str(language_filter).split(',') if x.strip()
        }
        super().__init__(ds_name, tok, nd, domain_key=self.repo_field,
                         buf_tokens=buf_tokens, revision=revision, data_dir=data_dir,
                         text_field=self.text_field, source_name="code")
        self.pack_format = "code-eos-v1"
        # Code packs have always appended EOS in _encode_packs(), so their old
        # version-1 checkpoint buffers are already compatible.
        self.legacy_buffers_compatible = True

    @staticmethod
    def _clean_meta(value, fallback='unknown'):
        text = str(value or fallback).replace('\r', ' ').replace('\n', ' ').strip()
        return text[:512] if text else fallback

    def _content(self, row):
        for key in (self.text_field, 'content', 'text', 'code'):
            value = row.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return None

    def _repo(self, row):
        for key in (self.repo_field, 'repo_name', 'repository_name', 'repository',
                    'repo', 'max_stars_repo_name'):
            value = row.get(key)
            if value:
                return self._clean_meta(value)
        return self._clean_meta(row.get('id'), 'unknown-repo')

    def _path(self, row):
        for key in (self.path_field, 'path', 'file_path', 'filename', 'name', 'max_stars_repo_path'):
            value = row.get(key)
            if value:
                return self._clean_meta(value, 'unknown-file')
        return 'unknown-file'

    def _lang(self, row):
        for key in (self.lang_field, 'language', 'lang', 'programming_language'):
            value = row.get(key)
            if value:
                return self._clean_meta(value, 'unknown')
        return 'unknown'

    def _accept(self, row):
        if self._content(row) is None:
            return False
        return not self.language_filter or self._lang(row).lower() in self.language_filter

    def _render_repo_pack(self, repo, rows):
        parts = [f'<repo name="{self._clean_meta(repo)}">\n']
        for row in rows:
            content = self._content(row)
            if content is None:
                continue
            parts.append(
                f'<file path="{self._path(row)}" language="{self._lang(row)}">\n'
            )
            parts.append(content)
            if not content.endswith('\n'):
                parts.append('\n')
            parts.append('</file>\n')
        parts.append('</repo>\n')
        return ''.join(parts)

    def _group_rows(self, rows):
        if not self.group_repo:
            return [(self._repo(row), [row]) for row in rows]
        groups = collections.OrderedDict()
        for row in rows:
            groups.setdefault(self._repo(row), []).append(row)
        return list(groups.items())

    def _encode_packs(self, rows):
        packs = self._group_rows(rows)
        texts = [self._render_repo_pack(repo, items) for repo, items in packs]
        ids_batch = self.tok(texts, add_special_tokens=False)['input_ids']
        eos = getattr(self.tok, 'eos_token_id', None)
        encoded = []
        for pack, ids in zip(packs, ids_batch):
            if eos is not None:
                ids = list(ids) + [int(eos)]
            encoded.append((pack, ids))
        return encoded

    def _pump(self, n):
        import numpy as np
        rows = []
        for _ in range(n):
            try:
                row = next(self.it)
                self.raw_cursor += 1
            except StopIteration:
                self.done = True
                break
            if self._accept(row):
                rows.append(row)
        if not rows:
            return
        for (repo, items), ids in self._encode_packs(rows):
            a = np.asarray(ids, dtype='uint16')
            if a.size < 8:
                continue
            d = self._domain_of({self.repo_field: repo})
            if self.n_tokens[d] >= self.buf_tokens:
                d = min(range(self.nd), key=lambda i: self.n_tokens[i])
            self.rows[d].append(a)
            self.n_tokens[d] += int(a.size)
            self.n_rows[d] += len(items)
            self.consumed += len(items)

    def capture_val(self, n_tokens, buf_rows=32):
        """Capture held-out code using exactly the same repository formatting."""
        import numpy as np
        ids_all, tot, raw_n = [], 0, 0
        while tot < n_tokens:
            rows = []
            for _ in range(buf_rows):
                try:
                    row = next(self.it)
                    raw_n += 1
                    self.raw_cursor += 1
                except StopIteration:
                    self.done = True
                    break
                if self._accept(row):
                    rows.append(row)
            if not rows:
                if self.done:
                    break
                continue
            for (_, items), ids in self._encode_packs(rows):
                a = np.asarray(ids, dtype='uint16')
                if a.size < 8:
                    continue
                ids_all.append(a)
                tot += int(a.size)
                self.consumed += len(items)
            if self.done:
                break
        if not ids_all:
            raise RuntimeError('CODE val capture exhausted the stream (no usable rows)')
        return torch.from_numpy(np.concatenate(ids_all)).long(), raw_n


class StreamCoT:
    """Streaming CoT source: fetches the jsonl in byte-range chunks, tokenizes
    incrementally, loops forever (the corpus repeats, like random-window
    sampling over a fixed tensor). val = tokenized tail (last val_frac of the
    file), fetched once at init. Same duck-type as StreamSource.batch(T, B)."""

    def __init__(self, url, tok, val_frac=0.02, cap_tokens=41_000_000, chunk_bytes=4_000_000):
        self.url = url
        self.tok = tok
        self.chunk_bytes = chunk_bytes
        self.buf = collections.deque()
        self.n_tokens = 0
        self.consumed_bytes = 0
        self.full_text = None
        req = Request(url, headers={"Range": "bytes=0-1023"})
        with urlopen(req, timeout=120) as f:
            code = getattr(f, "status", 206)
            cr = f.headers.get("Content-Range") or ""
            self.total = int(cr.split("/")[-1]) if "/" in cr else int(f.headers.get("Content-Length") or 0)
        if not self.total:
            raise RuntimeError("CoT server missing Content-Length/Content-Range; cannot stream")
        if code == 206:
            self.train_end = int(self.total * (1 - val_frac))
            with urlopen(Request(url, headers={"Range": f"bytes={self.train_end}-"}),
                         timeout=300) as f:
                val_text = f.read().decode("utf-8", errors="replace")
            mode = "range"
        else:
            with urlopen(url, timeout=600) as f:
                text = f.read().decode("utf-8", errors="replace")
            self.full_text = text
            self.total = len(text)
            self.train_end = int(len(text) * (1 - val_frac))
            val_text = text[self.train_end:]
            mode = "ram"
        self.val = _tokenize_stream(val_text, tok, int(cap_tokens * val_frac * 1.5) + 1000)
        n_chunks = max(1, (self.train_end + self.chunk_bytes - 1) // self.chunk_bytes)
        self.chunk_order = list(range(n_chunks))
        random.shuffle(self.chunk_order)
        self.chunk_pos = 0
        print(f"CoT streaming {url}: {self.total/2**20:.1f}MB total (mode={mode}), "
              f"val={len(self.val):,} tokens, shuffled chunks", flush=True)

    def _next_chunk_off(self):
        """Chunk-level shuffle: fetch 4MB chunks in random order, reshuffling per
        corpus pass — restores the old uniform-window distribution (vs sequential
        consumption, which over-samples one document and correlates windows)."""
        c = self.chunk_order[self.chunk_pos]
        self.chunk_pos += 1
        if self.chunk_pos >= len(self.chunk_order):
            random.shuffle(self.chunk_order)
            self.chunk_pos = 0
        return c * self.chunk_bytes

    def _pump(self):
        import numpy as np
        off = self._next_chunk_off()
        if self.full_text is not None:
            text = self.full_text[off:min(self.train_end, off + self.chunk_bytes)]
            self.consumed_bytes += len(text)
            ids = np.asarray(self.tok(text, add_special_tokens=False)["input_ids"],
                             dtype="uint16")
            self.buf.append(ids)
            self.n_tokens += int(ids.size)
            return
        end = min(self.train_end, off + self.chunk_bytes)
        with urlopen(Request(self.url, headers={"Range": f"bytes={off}-{end - 1}"}),
                     timeout=180) as f:
            data = f.read()
        text = data.decode("utf-8", errors="replace")
        ids = np.asarray(self.tok(text, add_special_tokens=False)["input_ids"],
                         dtype="uint16")
        self.consumed_bytes += end - off
        self.buf.append(ids)
        self.n_tokens += int(ids.size)

    def sample_window(self, T, B=1):
        """Random window from the buffered train tokens (non-consuming) — for the
        same-domain CoT memorization metric."""
        import numpy as np
        need = (T + 1) * B + 64
        while self.n_tokens < need:
            self._pump()
        parts, n = [], 0
        for row in self.buf:
            parts.append(row)
            n += int(row.size)
            if n >= need:
                break
        buf = np.concatenate(parts)
        xs, ys = [], []
        for _ in range(B):
            ix = random.randrange(len(buf) - T - 1)
            xs.append(torch.from_numpy(buf[ix:ix + T]).long())
            ys.append(torch.from_numpy(buf[ix + 1:ix + T + 1]).long())
        return torch.stack(xs).to(DEVICE), torch.stack(ys).to(DEVICE)

    def prefill(self):
        self._pump()
        print(f"CoT prefilled: {self.n_tokens:,} tokens buffered", flush=True)

    def batch(self, T, B):
        import numpy as np
        need = (T + 1) * B + 64
        while self.n_tokens < need:
            self._pump()
        parts, n = [], 0
        while n < need:
            row = self.buf.popleft()
            self.n_tokens -= int(row.size)
            if n + int(row.size) <= need:
                parts.append(row)
                n += int(row.size)
            else:
                take = need - n
                parts.append(row[:take])
                self.buf.appendleft(row[take:])
                self.n_tokens += int(row.size) - take
                n = need
        buf = np.concatenate(parts)
        ids = torch.from_numpy(buf).long()
        xs, ys = [], []
        seg = len(ids) // B
        for b in range(B):
            lo = b * seg
            hi = (b + 1) * seg - (T + 1)
            ix = lo + (random.randrange(hi - lo) if hi > lo else 0)
            xs.append(ids[ix:ix + T])
            ys.append(ids[ix + 1:ix + T + 1])
        return torch.stack(xs), torch.stack(ys)


# --- REMOVED 2026-09-30 (lean cleanup): DraftNet + draft_train_step ---
# What they were: trajectory-conditioned draft head for speculative decoding,
#   trained online with a separate draft optimizer every TB_DRAFT_EVERY steps.
# Why removed: ckpt_ongoing.pt has no draft tensors; TB_DRAFT defaults to 0 so
#   `m.draft` is never built and `yb_keep` is always None here. Keeping the
#   class would only preserve an untested speculative path. generate_spec (the
#   only consumer) is removed alongside. Setting TB_DRAFT=1 now raises at
#   build/dispatch instead of silently training a headless draft.
# To restore: recover from train_walker_1b_continue.py.bak-20260930.
class DataMix:
    """Two streams (general shards + CoT), mixture share scheduled by the
    curriculum. get_batch returns windows from stream a with prob share, else b."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.cot = None
        self.val = None
        self.fw_val = None
        # A weighted stream owns one independent, unseen validation bank per
        # source.  Each bank is the same size so low-weight sources have the
        # same statistical resolution as high-weight sources; the headline
        # metric is still weighted by the actual training mixture.
        self.mix_val = collections.OrderedDict()
        self.mix_val_weights = collections.OrderedDict()
        self.mix_val_diagnostics = collections.OrderedDict()
        # Parallel SFT prompt masks per bank (uint8 tensors, or None when the
        # bank predates masking). Answer-only validation under TB_SFT_MASK=1.
        self.mix_val_masks = collections.OrderedDict()
        self.last_source_val_metrics = collections.OrderedDict()
        self.last_val_summary = {}
        self.code_val = None
        self.code_stream = None
        # CoT is REMOVED by default: model trains on FineWeb only.
        # Kept only if TB_COT_URL or TB_COT_PATH is explicitly set (experiments).
        if cfg.cot_path or cfg.cot_url:
            if cfg.cot_path:
                if not os.path.exists(cfg.cot_path):
                    raise RuntimeError(
                        f"TB_COT_PATH={cfg.cot_path!r} not found (CWD={os.getcwd()}) — create it first: "
                        "python train_1b.py --make-cot-shard cot_train.jsonl cot_41m.bin 41000000")
                if cfg.cot_path.endswith((".jsonl", ".json", ".txt")):
                    with open(cfg.cot_path, "rb") as f:
                        text = f.read().decode("utf-8", errors="replace")
                    self.cot = _tokenize_stream(text, cfg.tok, cfg.cot_tokens)
                    src = f"{cfg.cot_path} (raw jsonl, tokenized in-process)"
                else:
                    self.cot = _load_shard(cfg.cot_path, cap_tokens=cfg.cot_tokens)
                    src = f"{cfg.cot_path} (pre-tokenized shard)"
                n = int(0.98 * len(self.cot))
                self.val = self.cot[n:]
                self.cot = self.cot[:n]
                print(f"CoT loaded from {src}: {len(self.cot):,} tokens (val {len(self.val):,})", flush=True)
            else:
                self.cot = StreamCoT(cfg.cot_url, cfg.tok, cap_tokens=cfg.cot_tokens)
                self.val = self.cot.val
        else:
            print("CoT: OFF", flush=True)
        self.general = []
        self.stream = None
        self.stream_mix_active = False
        mix_specs = _resolve_stream_specs(cfg)
        if mix_specs:
            if cfg.stream:
                raise RuntimeError("TB_STREAM cannot be combined with TB_STREAM_PRESET/TB_STREAM_MIX; unset TB_STREAM")
            if getattr(cfg, "doremi", False):
                raise RuntimeError("TB_DOREMI is disabled for weighted multi-stream presets; weights are the curriculum")
            self.stream = MixedStreamSource(
                mix_specs, cfg.tok, buf_tokens=cfg.stream_buf,
                validation_source=getattr(cfg, "stream_val_source", 0))
            cfg.stream_mix_signature = self.stream.signature
            self.stream_mix_active = True
            self.general = [StreamDomain(i) for i in range(self.stream.nd)]
            budget = int(getattr(cfg, "stream_mix_total_tokens", 15_000_000_000))
            print(f"weighted stream mix: {len(mix_specs)} sources signature={self.stream.signature} "
                  f"target={budget:,} training tokens", flush=True)
            for spec in mix_specs:
                target = int(round(spec["weight"] * budget))
                print(f"  {spec['name']:<16} {100*spec['weight']:5.1f}%  target={target:>13,}  "
                      f"{spec['dataset']}", flush=True)
        elif cfg.stream:
            self.stream = StreamSource(cfg.stream, cfg.tok, cfg.stream_nd,
                                       domain_key=cfg.stream_domain, buf_tokens=cfg.stream_buf,
                                       revision=getattr(cfg, "stream_revision", ""),
                                       text_field=getattr(cfg, "stream_text_field", "text"))
            self.general = [StreamDomain(i) for i in range(cfg.stream_nd)]
            print(f"streaming {cfg.stream}: {cfg.stream_nd} domains (key={cfg.stream_domain}, "
                  f"hash-stable for resume, packing={self.stream.pack_format})", flush=True)
        elif cfg.data_dir:
            for p in sorted(glob.glob(os.path.join(cfg.data_dir, "**", "*.bin"), recursive=True)) + \
                     sorted(glob.glob(os.path.join(cfg.data_dir, "**", "*.npy"), recursive=True)):
                self.general.append(_load_shard(p))
            print(f"general shards: {len(self.general)}, total {sum(len(a) for a in self.general):,} tokens", flush=True)
            if not self.general and os.path.isdir(cfg.data_dir):
                names = sorted(os.listdir(cfg.data_dir))[:20]
                print(f"WARNING: no *.bin/*.npy matched under {cfg.data_dir}; first entries: {names}", flush=True)
        # Independent repository-code stream; never overload the CoT path.
        if self.stream_mix_active and getattr(cfg, 'code_stream', ''):
            if getattr(cfg, 'code_share', 0.0) > 0.0:
                raise RuntimeError("weighted stream preset already includes its intended code share; unset TB_CODE_STREAM/TB_CODE_SHARE")
            print("code stream identity ignored because weighted general mix is active and TB_CODE_SHARE=0", flush=True)
        elif getattr(cfg, 'code_stream', ''):
            if self.stream is None:
                raise RuntimeError(
                    'TB_CODE_STREAM requires TB_STREAM: this mode mixes code with FineWeb')
            if getattr(cfg, 'doremi', False):
                raise RuntimeError(
                    'TB_CODE_STREAM + TB_DOREMI is intentionally unsupported; use the explicit fixed code/FW mix')
            self.code_stream = CodeStreamSource(
                cfg.code_stream, cfg.tok, cfg.code_stream_nd,
                buf_tokens=cfg.code_stream_buf,
                revision=cfg.code_stream_revision,
                data_dir=cfg.code_data_dir,
                text_field=cfg.code_text_field,
                repo_field=cfg.code_repo_field,
                path_field=cfg.code_path_field,
                lang_field=cfg.code_lang_field,
                group_repo=cfg.code_group_repo,
                language_filter=cfg.code_lang_filter,
            )
            print(
                f"code stream {cfg.code_stream}: share {cfg.code_share_start:.3f}->{cfg.code_share:.3f}, "
                f"domains={cfg.code_stream_nd}, repo={cfg.code_repo_field}, "
                f"text={cfg.code_text_field}, grouped={cfg.code_group_repo}", flush=True)
        elif getattr(cfg, 'code_share', 0.0) > 0.0:
            raise RuntimeError('TB_CODE_SHARE > 0 requires TB_CODE_STREAM')
        if self.cot is not None and not isinstance(self.cot, StreamCoT) and int(self.cot.max()) >= cfg.vocab:
            raise RuntimeError(
                f"CoT contains token id {int(self.cot.max())} >= vocab {cfg.vocab} — "
                "wrong tokenizer or the file was read as raw uint16 (use raw .jsonl/.txt, not a .bin)")
        if isinstance(self.cot, StreamCoT) and self.val is not None and int(self.val.max()) >= cfg.vocab:
            raise RuntimeError(
                f"CoT val contains token id {int(self.val.max())} >= vocab {cfg.vocab} — wrong tokenizer")
        if self.stream is None:
            for i, a in enumerate(self.general):
                if int(a.max()) >= cfg.vocab:
                    raise RuntimeError(
                        f"shard {i} contains token id {int(a.max())} >= vocab {cfg.vocab} — wrong tokenizer")
        if not self.general and self.cot is None:
            raise RuntimeError("no data: set DATA_DIR or COT_URL")

    def get_batch(self, it, B, T, cot_share, code_share=0.0):
        # cot_share only matters if CoT is configured; with cot=None every
        # batch draws from FineWeb (stream or shards) — pure FineWeb training.
        if self.stream is not None:
            xb, yb = [], []
            if code_share < 0.0 or cot_share < 0.0 or code_share + cot_share > 1.0:
                raise ValueError(f'invalid data mix: code={code_share} cot={cot_share}')
            for _ in range(B):
                draw = random.random()
                use_code = self.code_stream is not None and draw < code_share
                use_cot = (not use_code and self.cot is not None
                           and draw < code_share + cot_share)
                if use_code:
                    x, y = self.code_stream.batch(random.randrange(self.code_stream.nd), T, 1)
                    xb.append(x[0]); yb.append(y[0])
                elif use_cot:
                    if isinstance(self.cot, StreamCoT):
                        x, y = self.cot.batch(T, 1)
                        xb.append(x[0]); yb.append(y[0])
                    else:
                        ix = torch.randint(0, len(self.cot) - T - 1, (1,)).item()
                        xb.append(self.cot[ix:ix + T])
                        yb.append(self.cot[ix + 1:ix + T + 1])
                else:
                    if isinstance(self.stream, MixedStreamSource):
                        x, y = self.stream.sample_batch(T, 1)
                    else:
                        x, y = self.stream.batch(random.randrange(self.stream.nd), T, 1)
                    xb.append(x[0]); yb.append(y[0])
            return torch.stack(xb).to(DEVICE), torch.stack(yb).to(DEVICE)
        if self.code_stream is not None and code_share > 0.0:
            raise RuntimeError('code mixing is only supported with streaming FineWeb in this build')
        g = torch.Generator().manual_seed(7_000_003 + it)
        xb, yb = [], []
        for _ in range(B):
            use_cot = self.cot is not None and (self.general and torch.rand(1, generator=g).item() < cot_share or not self.general)
            if use_cot:
                if isinstance(self.cot, StreamCoT):
                    x, y = self.cot.batch(T, 1)
                    xb.append(x[0])
                    yb.append(y[0])
                    continue
                d = self.cot
            else:
                d = self.general[int(torch.randint(0, len(self.general), (1,), generator=g))]
            ix = torch.randint(0, len(d) - T - 1, (1,), generator=g).item()
            xb.append(d[ix:ix + T])
            yb.append(d[ix + 1:ix + T + 1])
        return torch.stack(xb).to(DEVICE), torch.stack(yb).to(DEVICE)

    def get_batch_domain(self, it, B, T, d_idx, srcs=None):
        """DoReMi: draw a batch from one specific domain (general shard index)."""
        srcs = self.general if srcs is None else srcs
        d = srcs[d_idx]
        if isinstance(d, StreamDomain):
            x, y = self.stream.batch(d.idx, T, B)
            return x.to(DEVICE), y.to(DEVICE)
        if isinstance(d, StreamCoT):
            x, y = d.batch(T, B)
            return x.to(DEVICE), y.to(DEVICE)
        g = torch.Generator().manual_seed(7_000_003 + it)
        xb, yb = [], []
        for _ in range(B):
            ix = torch.randint(0, len(d) - T - 1, (1,), generator=g).item()
            xb.append(d[ix:ix + T])
            yb.append(d[ix + 1:ix + T + 1])
        return torch.stack(xb).to(DEVICE), torch.stack(yb).to(DEVICE)

    def cot_train_batches(self, n, T):
        """Same-domain CoT train windows (for the memorization gap metric)."""
        out = []
        if isinstance(self.cot, StreamCoT):
            for _ in range(n):
                out.append(self.cot.sample_window(T))
        elif self.cot is not None:
            g = torch.Generator().manual_seed(8_000_003)
            for _ in range(n):
                ix = torch.randint(0, len(self.cot) - T - 1, (1,), generator=g).item()
                out.append((self.cot[ix:ix + T].unsqueeze(0).to(DEVICE),
                            self.cot[ix + 1:ix + T + 1].unsqueeze(0).to(DEVICE)))
        return out

    def val_batches(self, n_batches, T):
        """Primary held-out windows.

        For a weighted stream, return the same number of deterministic windows
        from every source.  ``evaluate`` aggregates the source means with the
        configured training weights; callers performing an ablation receive
        the identical flattened window suite.
        """
        if self.mix_val:
            return [batch for _, _, batches in self.val_batches_by_source(n_batches, T)
                    for batch in batches]
        out = []
        g = torch.Generator().manual_seed(9_000_003)
        if self.fw_val is not None and len(self.fw_val) > T + 1:
            d = self.fw_val
            for k in range(n_batches):
                ix = torch.randint(0, len(d) - T - 1, (1,), generator=g).item()
                out.append((d[ix:ix + T].unsqueeze(0).to(DEVICE), d[ix + 1:ix + T + 1].unsqueeze(0).to(DEVICE)))
            return out
        d = self.val if self.val is not None else self.cot
        if d is None:
            return []
        for k in range(n_batches):
            ix = torch.randint(0, len(d) - T - 1, (1,), generator=g).item()
            out.append((d[ix:ix + T].unsqueeze(0).to(DEVICE), d[ix + 1:ix + T + 1].unsqueeze(0).to(DEVICE)))
        return out

    def val_batches_by_source(self, n_batches, T):
        """Return ``(name, weight, batches)`` for every held-out mix source."""
        if not self.mix_val:
            return []
        result = []
        for source_index, (name, tokens) in enumerate(self.mix_val.items()):
            if len(tokens) <= T + 1:
                raise RuntimeError(
                    f"validation bank {name!r} has {len(tokens):,} tokens, too short for T={T}")
            generator = torch.Generator().manual_seed(9_100_003 + source_index * 100_003)
            mtoks = (self.mix_val_masks or {}).get(name)
            batches = []
            for _ in range(n_batches):
                ix = torch.randint(0, len(tokens) - T - 1, (1,), generator=generator).item()
                if mtoks is not None:
                    ix = redraw_unscored_window(mtoks, ix, T, len(tokens), generator, name)
                yb = tokens[ix + 1:ix + T + 1].unsqueeze(0).to(DEVICE)
                if mtoks is not None:
                    # Answer-only validation: never score prompt tokens.
                    # Clone: yb may be a view into the shared bank. Index dim
                    # 1 explicitly: yb is [B, T] while the mask is [T].
                    yb = yb.clone()
                    yb[:, mtoks[ix + 1:ix + T + 1].to(DEVICE).bool()] = -100
                batches.append((tokens[ix:ix + T].unsqueeze(0).to(DEVICE), yb))
            result.append((name, float(self.mix_val_weights[name]), batches))
        return result

    def val_batches_fw(self, n_batches, T):
        """Held-out FineWeb windows (unseen by training — captured from the
        stream ahead of the training cursor and never served to get_batch)."""
        if self.fw_val is None or len(self.fw_val) <= T + 1:
            return []
        out = []
        g = torch.Generator().manual_seed(9_900_003)
        for k in range(n_batches):
            ix = torch.randint(0, len(self.fw_val) - T - 1, (1,), generator=g).item()
            out.append((self.fw_val[ix:ix + T].unsqueeze(0).to(DEVICE),
                        self.fw_val[ix + 1:ix + T + 1].unsqueeze(0).to(DEVICE)))
        return out


    def val_batches_code(self, n_batches, T):
        if self.code_val is None or len(self.code_val) <= T + 1:
            return []
        out = []
        g = torch.Generator().manual_seed(9_950_003)
        for _ in range(n_batches):
            ix = torch.randint(0, len(self.code_val) - T - 1, (1,), generator=g).item()
            out.append((self.code_val[ix:ix + T].unsqueeze(0).to(DEVICE),
                        self.code_val[ix + 1:ix + T + 1].unsqueeze(0).to(DEVICE)))
        return out

    def checkpoint_streams(self):
        if self.stream is None and self.code_stream is None:
            return None
        return {
            'version': 2,
            'fineweb': self.stream.checkpoint_state() if self.stream is not None else None,
            'code': self.code_stream.checkpoint_state() if self.code_stream is not None else None,
        }

    def restore_streams(self, state, legacy_stream_rows=None):
        """Restore dual-stream checkpoints and legacy FineWeb-only Phase-A state."""
        wrapped = isinstance(state, dict) and state.get('version') == 2 \
                  and ('fineweb' in state or 'code' in state)
        if self.stream is not None:
            fw = state.get('fineweb') if wrapped else state
            if fw is not None:
                if isinstance(self.stream, MixedStreamSource):
                    migration = self.stream.restore_checkpoint_state(fw)
                    print(f"MIX restored {len(self.stream.sources)} exact cursors + "
                          f"{sum(self.stream.n_tokens):,} EOS-packed prefetched tokens", flush=True)
                else:
                    self.stream.skip(int(fw['raw_cursor']))
                    migration = self.stream.restore_checkpoint_state(fw)
                    if migration.get('repacked', False):
                        print(f"FineWeb document-packing migration: preserved raw cursor "
                              f"{self.stream.raw_cursor:,}; discarded "
                              f"{migration['discarded_tokens']:,} legacy boundary-less prefetched tokens "
                              "(model/optimizer/schedule/cumulative tokens unchanged)", flush=True)
                    else:
                        print(f"FineWeb restored exact cursor + {sum(self.stream.n_tokens):,} "
                              f"EOS-packed prefetched tokens", flush=True)
            elif legacy_stream_rows and not isinstance(self.stream, MixedStreamSource):
                self.stream.skip(int(legacy_stream_rows))
                print(f"FineWeb legacy resume skipped {int(legacy_stream_rows):,} rows", flush=True)
        if self.code_stream is not None:
            code = state.get('code') if wrapped else None
            if code is not None:
                self.code_stream.skip(int(code['raw_cursor']))
                self.code_stream.restore_checkpoint_state(code)
                print(f"CODE restored exact cursor + {sum(self.code_stream.n_tokens):,} prefetched tokens", flush=True)
            else:
                print('CODE starts at fresh cursor (expected on Phase-A -> Phase-B transition)', flush=True)

    def start_streams(self):
        if self.stream is not None and self.stream.thread is None:
            self.stream.start()
        if self.code_stream is not None and self.code_stream.thread is None:
            self.code_stream.start()


# ------------------------------------------------------------ curriculum ----

def t_at(s, S, t0, t1):
    # quantize the ramp to 128-token steps: the per-step shape churn forces
    # cuBLAS/flash to cache a new workspace per unique T (GBs, never released).
    Tq = int(t0 + (t1 - t0) * min(1.0, s / (0.5 * S)))
    return max(t0, (Tq // 128) * 128)


def w_at(t, t1):
    return max(32, min(512, t // 8))


def cot_share_at(s, S):
    return 0.2 + 0.5 / (1.0 + math.exp(-12.0 * (s / S - 0.4)))



def code_share_at(schedule_toks, cfg):
    if not getattr(cfg, 'code_stream', ''):
        return 0.0
    start = float(cfg.code_share_start)
    target = float(cfg.code_share)
    ramp = max(1, int(cfg.code_ramp_tokens))
    p = min(1.0, max(0.0, float(schedule_toks) / ramp))
    return start + (target - start) * p


def get_mixed_training_batch(dm, iteration, batch_size, tokens, cot_share, code_share):
    """Forward the data mixture that the training log claims to use."""
    return dm.get_batch(iteration, batch_size, tokens, cot_share, code_share)


def _microbatch_source_name(dm):
    """Best-effort source name for the microbatch just served (never raises)."""
    try:
        stream = getattr(dm, "stream", None)
        src_i = getattr(stream, "last_src", None)
        specs = getattr(stream, "specs", None)
        if isinstance(src_i, int) and isinstance(specs, list) and 0 <= src_i < len(specs):
            return str(specs[src_i].get("name", src_i))
    except Exception:
        pass
    return "stream"


def _microbatch_debug_line(step, gi, src_name, lval):
    return f"MICRO step={step} gi={gi} src={src_name} loss={float(lval):.4f}"


def _microbatch_r_active(dm):
    """Conditional-r gate for the microbatch just served (never raises).

    Returns None when narrow-R is disabled (caller passes through to the
    model's default), else True/False from should_apply_r() using the
    TB_R_INSTR_ONLY policy. Eval callers leave r_active=None (deterministic
    eval-mean) so banks stay comparable.
    """
    try:
        if os.environ.get("TB_R_ENABLE", "0") != "1":
            return None
        instr_only = os.environ.get("TB_R_INSTR_ONLY", "1") == "1"
        return bool(should_apply_r(_microbatch_source_name(dm), instr_only=instr_only))
    except Exception:
        return None


LR_FLOOR = float(os.environ.get("TB_LR_FLOOR", "1e-6"))
WARMUP = int(os.environ.get("TB_WARMUP", "500"))


def lr_at(s, S, peak, warmup=WARMUP, floor=LR_FLOOR):
    if s < warmup:
        return peak * (s + 1) / max(1, warmup)
    t = (s - warmup) / max(1, S - warmup)
    # Clamp past the horizon (mirrors lr_at_tokens): an unclamped cosine
    # re-rises toward peak after t=1, silently un-decaying the schedule
    # for any run that overshoots its horizon.
    t = min(1.0, max(0.0, t))
    return floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * t))


def lr_at_tokens(tokens, total_tokens, peak, warmup_tokens, floor=LR_FLOOR):
    """Warmup + cosine expressed in optimizer tokens, not optimizer steps."""
    if tokens < warmup_tokens:
        return peak * (tokens + 1) / max(1, warmup_tokens)
    t = (tokens - warmup_tokens) / max(1, total_tokens - warmup_tokens)
    t = min(1.0, max(0.0, t))
    return floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * t))


def schedule_horizon_steps(cfg):
    return max(1, int(getattr(cfg, "schedule_steps", getattr(cfg, "steps", 1))))


def csa2_alpha_at(cfg, step):
    """Return the resumable CSA2 bridge alpha for the current phase step."""
    ramp = max(0, int(getattr(cfg, 'v41_csa2_ramp_steps', 0)))
    start = float(getattr(cfg, 'v41_csa2_alpha_start', 0.0))
    end = float(getattr(cfg, 'v41_csa2_alpha_end', 1.0))
    if ramp <= 0:
        return None
    progress = min(1.0, max(0.0, float(step) / float(ramp)))
    return start + (end - start) * progress


def set_csa2_alpha(model, alpha):
    """Update persistent CSA2 bridge state without adding optimizer state."""
    if alpha is None:
        return
    value = float(min(1.0, max(0.0, alpha)))
    for block in getattr(model, 'blocks', ()):
        attn = getattr(block, 'attn', None)
        if attn is not None and hasattr(attn, 'csa2_alpha'):
            attn.csa2_alpha.fill_(value)


def _planned_tokens(cfg, n_steps):
    horizon = schedule_horizon_steps(cfg)
    return sum(t_at(s, horizon, cfg.t0, cfg.t1) * cfg.B * cfg.accum
               for s in range(n_steps))


def prepare_schedule(cfg):
    cfg.lr_schedule = str(getattr(cfg, "lr_schedule", "steps")).lower()
    if cfg.lr_schedule not in {"steps", "tokens"}:
        raise ValueError("TB_LR_SCHEDULE must be 'steps' or 'tokens'")
    if int(getattr(cfg, "schedule_steps", 0)) <= 0:
        cfg.schedule_steps = int(cfg.steps)
    if cfg.lr_schedule != "tokens":
        return
    if cfg.lr_total_tokens <= 0:
        cfg.lr_total_tokens = _planned_tokens(cfg, schedule_horizon_steps(cfg))
    if cfg.lr_warmup_tokens <= 0:
        cfg.lr_warmup_tokens = _planned_tokens(cfg, min(cfg.warmup, schedule_horizon_steps(cfg)))


def schedule_state(cfg):
    """The schedule contract that must survive a checkpoint/resume cycle."""
    return {
        "version": 1,
        "kind": cfg.lr_schedule,
        "total_steps": schedule_horizon_steps(cfg),
        "stop_steps": int(cfg.steps),
        "peak_lr": float(cfg.lr),
        "floor": float(cfg.lr_floor),
        "warmup_steps": int(cfg.warmup),
        "total_tokens": int(cfg.lr_total_tokens),
        "warmup_tokens": int(cfg.lr_warmup_tokens),
        "origin_step": int(getattr(cfg, "schedule_origin_step", 0) or 0),
    }


def restore_schedule(cfg, run_state):
    """Lock a resumed job to its saved schedule unless explicitly overridden."""
    saved = (run_state or {}).get("schedule")
    if not saved:
        return False
    if getattr(cfg, "resume_schedule", "lock") == "override":
        print("resume: schedule override requested — using current TB_* schedule values", flush=True)
        return False
    cfg.lr_schedule = saved.get("kind", "steps")
    cfg.schedule_steps = int(saved["total_steps"])
    cfg.lr = float(saved["peak_lr"])
    cfg.lr_floor = float(saved["floor"])
    cfg.warmup = int(saved["warmup_steps"])
    cfg.lr_total_tokens = int(saved.get("total_tokens", 0))
    cfg.lr_warmup_tokens = int(saved.get("warmup_tokens", 0))
    prepare_schedule(cfg)
    print(f"resume: schedule locked to checkpoint ({cfg.lr_schedule}, "
          f"horizon={schedule_horizon_steps(cfg):,}, stop={cfg.steps:,}, peak={cfg.lr:g})", flush=True)
    return True


def maybe_reset_best_on_resume(best_val, step):
    """Start a fresh best series when validation identity intentionally changes.

    This never changes model/optimizer/schedule state and is deliberately
    opt-in because validation paths were absent from older run manifests.
    """
    if int(step) > 0 and os.environ.get("TB_RESET_BEST_ON_RESUME", "0") == "1":
        print("resume: explicitly reset best-checkpoint comparator for current validation set",
              flush=True)
        return float("inf")
    return best_val


def train_phase_plan(saved_run_state, requested, step):
    """(effective train phase, restart?) for TB_TRAIN_PHASE.

    Checkpoints written before phases existed count as phase 1. Unset (<= 0)
    inherits the checkpoint's phase. A higher phase than the checkpoint's is a
    phase change (fresh optimizer + schedule from the cursor); the same phase
    is an ordinary in-phase resume, so re-running an interrupted phase never
    resets twice; a lower phase is refused.
    """
    requested = int(requested or 0)
    if saved_run_state is None or not step:
        return max(1, requested), False
    saved = int(saved_run_state.get("train_phase", 1) or 1)
    if requested <= 0:
        return saved, False
    if requested < saved:
        raise RuntimeError(
            f"TB_TRAIN_PHASE={requested} is older than the checkpoint's phase {saved}; "
            "phases only move forward")
    return requested, requested > saved


def phase_horizon_defaults(cfg, step):
    """A new phase's cosine spans this phase, not the whole run.

    Without an explicit TB_LR_TOTAL_STEPS / TB_LR_TOTAL_TOKENS the startup
    defaults were derived from the absolute stop step; re-derive them from the
    steps left (cfg.steps - step) so the LR actually reaches the floor by the
    phase's stop. Explicit values are kept as given (phase-relative)."""
    explicit = lambda name: (os.environ.get(name, "") or "0").strip() not in ("", "0")  # noqa: E731
    if not explicit("TB_LR_TOTAL_STEPS"):
        cfg.schedule_steps = max(1, int(cfg.steps) - int(step))
    if not explicit("TB_LR_TOTAL_TOKENS"):
        cfg.lr_total_tokens = 0
    if not explicit("TB_LR_WARMUP_TOKENS"):
        cfg.lr_warmup_tokens = 0
    prepare_schedule(cfg)


def schedule_origin(saved_run_state, step, phase_restart):
    """Optimizer step at which the current phase's step schedule starts."""
    if phase_restart:
        return int(step)
    saved = (saved_run_state or {}).get("schedule") or {}
    return int(saved.get("origin_step", 0) or 0)


def scheduled_lr(cfg, step, cum_toks):
    if cfg.lr_schedule == "tokens":
        lr = lr_at_tokens(cum_toks, cfg.lr_total_tokens, cfg.lr,
                          cfg.lr_warmup_tokens, cfg.lr_floor)
    else:
        origin = int(getattr(cfg, "schedule_origin_step", 0) or 0)
        lr = lr_at(max(0, step - origin), schedule_horizon_steps(cfg), cfg.lr, cfg.warmup, cfg.lr_floor)
    return lr_cooldown(cfg, step, lr)


def lr_cooldown(cfg, step, lr):
    """Linear end-of-run cooldown (TB_LR_COOLDOWN_START > 0).

    From the start step the scheduled LR is blended linearly toward the
    cooldown floor, reaching it exactly at the run's stop step (cfg.steps).
    Continuous at the start (frac=1), schedule-lock compatible (the saved
    schedule is untouched; this only reshapes the tail), and a no-op when
    unset. Motivation: the token-cosine horizon (15B) far exceeds the stop
    (~4.9B), so without it the LR never anneals and the weights random-walk
    on an LR noise floor (runs/gnorm_findings.txt).
    """
    start = int(os.environ.get("TB_LR_COOLDOWN_START", "0") or 0)
    if start <= 0 or step < start:
        return lr
    floor = float(os.environ.get("TB_LR_COOLDOWN_FLOOR", "0") or 0) or float(cfg.lr_floor)
    end = int(getattr(cfg, "steps", start))
    frac = max(0.0, min(1.0, (end - step) / max(1, end - start)))
    return floor + (lr - floor) * frac if lr > floor else lr


def is_bottom_param(name):
    """Token embedding + block 0: the bottom-of-stack group whose gradient is
    pure noise after the local-conv removal (runs/gnorm_findings.txt)."""
    name = name.replace("_orig_mod.", "")
    return name.startswith("emb.") or name.startswith("blocks.0.")


def clip_bottom_isolated(named_params, max_norm):
    """Clip emb + block 0 and the rest of the model as two separate groups.

    emb + block 0 carry a pure-noise gradient ~35x the rest at 123k tokens
    per step (runs/gnorm_findings.txt); under one global norm every step was
    clipped by THEIR noise. Returns (body_norm, bottom_norm), both raw
    (pre-clip) like clip_grad_norm_; max_norm <= 0 measures without clipping.
    """
    clip = max_norm if max_norm > 0.0 else float("inf")
    bottom = [p for n, p in named_params if is_bottom_param(n) and p.grad is not None]
    body = [p for n, p in named_params if not is_bottom_param(n) and p.grad is not None]
    device = (bottom or body)[0].grad.device if (bottom or body) else "cpu"
    zero = torch.tensor(0.0, device=device)
    bottom_norm = torch.nn.utils.clip_grad_norm_(bottom, clip) if bottom else zero
    body_norm = torch.nn.utils.clip_grad_norm_(body, clip) if body else zero
    return body_norm, bottom_norm


def bottom_isolate_ramp_plan(saved_run_state, step):
    """Plan the body-gradient ramp for the FIRST resume that isolates emb + block 0.

    Before isolation the global clip scaled every body gradient by the bottom
    group's noise (~35x), so the body's Adam second moments are calibrated to
    tiny gradients; unscaling them at once would spike updates ~35x for tens
    of steps. The ramp starts the body at its old clip factor and lifts it
    geometrically to 1 over TB_BOTTOM_ISOLATE_RAMP steps (0 = off). Skipped
    when the checkpoint was already trained with isolation on, or from scratch.
    """
    if os.environ.get("TB_BOTTOM_ISOLATE", "0") != "1":
        return None
    ramp = int(os.environ.get("TB_BOTTOM_ISOLATE_RAMP", "500") or 0)
    saved = ((saved_run_state or {}).get("runtime") or {}).get("TB_BOTTOM_ISOLATE", "")
    if ramp <= 0 or not step or saved == "1":
        return None
    print(f"resume: emb+block0 clip isolation enabled now -> body gradient ramp over {ramp} steps",
          flush=True)
    return {"steps": ramp, "start": None, "r0": None}


def bottom_ramp_scale(plan, step, body_norm, bottom_norm, max_norm):
    """Extra factor for body gradients during the isolation ramp (<= 1)."""
    if not plan:
        return 1.0
    if plan["r0"] is None:
        total = math.sqrt(body_norm ** 2 + bottom_norm ** 2)
        old = min(1.0, max_norm / total) if max_norm > 0 and total > 0 else 1.0
        new = min(1.0, max_norm / body_norm) if max_norm > 0 and body_norm > 0 else 1.0
        plan["r0"] = old / new
        plan["start"] = step
        print(f"bottom isolation ramp: body starts at x{plan['r0']:.4g} of its unclipped scale "
              f"(old global norm {total:.3g}, body {body_norm:.3g})", flush=True)
    t = (step - plan["start"]) / max(1, plan["steps"])
    if t >= 1.0:
        return 1.0
    return float(plan["r0"]) ** (1.0 - t)


def apply_bottom_lr_mult(model, opt):
    """TB_BOTTOM_LR_MULT: per-parameter LR multiplier for emb + block 0.

    Implemented as a param attribute read by CompensatedAdamW, so optimizer
    groups (and therefore checkpoint optimizer state) are unchanged. Other
    optimizers cannot honour it, so a non-default value fails closed there.
    """
    mult = float(os.environ.get("TB_BOTTOM_LR_MULT", "1.0") or 1.0)
    params = [p for n, p in model.named_parameters() if is_bottom_param(n)]
    for p in params:
        p._lr_mult = mult
    if mult != 1.0:
        if not isinstance(opt, CompensatedAdamW):
            raise RuntimeError("TB_BOTTOM_LR_MULT != 1 needs TB_OPT_KAHAN=1 (CompensatedAdamW)")
        print(f"optimizer: bottom-of-stack LR x{mult:g} on {len(params)} tensors "
              "(emb + block 0)", flush=True)
    return len(params)


def _source_hash():
    try:
        with open(__file__, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except OSError:
        return "unknown"



_STRUCTURAL_ENV_KEYS = (
    'TB_ATTN_EVERY','TB_HYBRID','TB_NO_WALK','TB_MLP_COMPACT','TB_MLP_COMPACT_EQUAL','TB_WALK_SHARED','TB_WALK_LOOP',
    # These switches change the Walk forward, even when they do not add tensor
    # shapes. Record them so a weights-only reload cannot silently evaluate a
    # different function (old manifests omitted them).
    'TB_WALK_NORM','TB_WALK_NORM_MODE','TB_WALK_HEADS','TB_WALK_FP32',
    'TB_WALK_MIX','TB_WALK_MIX_DEPTH','TB_WALK_MIX_GAIN',
    'TB_WALK_GLOB','TB_WALK_COUPLE','TB_WALK_GATE','TB_WALK_SELECTIVE_MEM',
    'TB_WALK_GEOM','TB_WALK_PLACE','TB_WALK_HURST','TB_WALK_AFFINE',
    'TB_WALK_MOTIF','TB_WALK_MOTIF_COUNT','TB_WALK_MOTIF_RANK','TB_WALK_MOTIF_PRED',
    'TB_WALK_MOTIF_V2','TB_WALK_MOTIF_ORTHO','TB_WALK_MOTIF_SHARED',
    'TB_WALK_MOTIF_CONF','TB_WALK_MOTIF_TOPK','TB_WALK_MOTIF_MOD',
    'TB_WDR','TB_WDR_LAYER','TB_WDR_RANK',
    'TB_WALK_CYCLE_GUARD','TB_WALK_CYCLE_ESCAPE','TB_WALK_CYCLE_RANK','TB_WALK_CYCLE_LAGS',
    'TB_WALK_CYCLE_THRESHOLD','TB_WALK_CYCLE_PERSISTENCE','TB_WALK_CYCLE_MAX_STRENGTH',
    'TB_WALK_CYCLE_OPEN_TARGET','TB_WALK_CYCLE_OPEN_WEIGHT',
    'TB_WALK_SUBSPACE_RANK','TB_WALK_HIER_EVIDENCE','TB_WALK_EVIDENCE','TB_WALK_DATA_TRAJ',
    'TB_WALK_HIER_K','TB_WALK_HIER_TRAJ','TB_WALK_HIER_CHANNEL_GATE','TB_WALK_HIER_LEARNED',
    'TB_WALK_HIER_AFFINE_FREE','TB_WALK_HIER_DEPTH_RMS','TB_WALK_HIER_ARCHIVE','TB_WALK_HIER_ARCHIVE_CHUNK',
    'TB_WALK_HIER_ARCHIVE_TOPK','TB_WALK_PRED','TB_WALK_PRED_CE',
    'TB_WALK_SUBSPACE_GATED','TB_WALK_SUBSPACE_CONTENT','TB_WALK_MEMORY_CHUNK',
    'TB_WALK_PATH_MIXER','TB_WALK_DUAL_PATH_MIXER','TB_TIE_EMB_HEAD','TB_POS_EMB','TB_MEM',
    'TB_NI','TB_NI_MODE','TB_NI_STEPS','TB_NI_RANK','TB_NI_WIDTH_MULT','TB_NI_LAYERS',
    # Phase-B superset *shape* knobs. Runtime archive scales are deliberately
    # excluded so they can be activated on an ordinary resume.
    'TB_PHASE_B_SUPERSET','TB_PHASE_WALK_LAYERS','TB_PHASE_WALK_RANK','TB_PHASE_WALK_K',
    'TB_PHASE_WALK_DATA_TRAJ','TB_PHASE_WALK_TRAJ','TB_PHASE_HIER_SCALES',
    'TB_PHASE_WALK_NORM','TB_PHASE_WALK_NORM_MODE',
    'TB_PHASE_ARCHIVE_CHUNK','TB_PHASE_ARCHIVE_TOPK',
    'TB_TRAJ_REG','TB_TRAJ_REG_RANK','TB_TRAJ_REG_HORIZON',
    'TB_KMERGE','TB_KMERGE_K','TB_KMERGE_PREFIX_FRAC',
    'TB_LEXICAL_SEMANTIC','TB_LEXICAL_SEMANTIC_RANK',
    'TB_SEMANTIC_WD',
    # V4.1-inspired additive sidecar and runtime cache proxy.
    'TB_V41_ENGRAM','TB_V41_ENGRAM_BUCKETS','TB_V41_ENGRAM_RANK','TB_V41_KV_FP4',
    'TB_V41_CED','TB_V41_CSA2','TB_PAUSE_N','TB_PAUSE_EVERY',
    'TB_PAUSE_HUGINN','TB_PAUSE_RANK',
    'TB_R_ENABLE','TB_R_START','TB_R_END','TB_R_MEAN','TB_R_MAX',
    'TB_R_BACKPROP','TB_R_INSTR_ONLY','TB_R_KV_SHARE',
    # R v2 update rule / gain sharing change the forward without new tensors.
    'TB_R_UPDATE','TB_R_GAIN',
    # Content-adaptive scale mixer: adds parameters under blocks.*.walk, so it
    # is structural (shape-changing) with one-way introduction tolerance below.
    'TB_WALK_ADAPT_SCALE','TB_WALK_ADAPT_LAGS','TB_WALK_ADAPT_RANK',
    'TB_WALK_CONV_BASELINE',
    # Recurrent block-loop geometry. Recorded (not cross-loaded: the panel
    # stays authoritative) so future runs and panels can prefill from the
    # checkpoint instead of guessing; old manifests simply lack these keys.
    'TB_LOOP_ENABLE','TB_LOOP_START','TB_LOOP_END','TB_LOOP_PASSES',
)


# Effective runtime controls snapshotted per checkpoint (see run_state).
# Same membership as the contract's RUNTIME_OVERRIDE_SPEC allowlist; the
# cross-module test test_resolved_runtime_keys_match_contract_allowlist
# fails closed on any drift either way.
_RESOLVED_RUNTIME_KEYS = (
    'TB_COMPILE','TB_COMPILE_DYNAMIC',
    'TB_ACTIVATION_CKPT_STRIDE','TB_ACTIVATION_CKPT_FULL_AT_T',
    'TB_WALK_LOOP_CKPT','TB_ATTN_WINDOW_CHUNK',
    'TB_PHASE_ASSOC_CHUNK','TB_WALK_ASSOC_CHUNK',
    'TB_LOCAL_CONV_FUSE_KERNELS','TB_WALK_MIX_TENSORCORE',
    'TB_METRIC_EVERY','TB_LOG_EVERY','TB_TREND_EVERY','TB_CKPT_EVERY',
    'TB_BASELINE_VAL_ON_RESUME','TB_WALK_ABLATE_EVAL','TB_PHASE_ABLATE_EVAL',
    'TB_V41_ABLATE_EVAL','TB_GRAD_CLIP','TB_ADAPTER_MULT','TB_PHASE_MULT',
    'TB_LOCAL_CONV_MULT','TB_LEXICAL_SEMANTIC_MULT',
    'TB_LEXICAL_SEMANTIC_AUX_WEIGHT','TB_LEXICAL_SEMANTIC_STRIDE',
    'TB_LEXICAL_SEMANTIC_SCALE','TB_LEXICAL_SEMANTIC_TEMPERATURE',
    'TB_WALK_FORWARD_SCALE','TB_PHASE_WALK_FORWARD_SCALE',
    'TB_WALK_INPUT_GRAD_SCALE','TB_PHASE_WALK_INPUT_GRAD_SCALE',
    'TB_PHASE_PRED_WEIGHT','TB_PHASE_PRED_SCALE',
    'TB_R_RESID_LOG','TB_R_HALT_EPS','TB_R_HALT_MIN',
    'TB_R_ADAPTER_MULT','TB_EMA_EVAL','TB_EMA_DECAY','TB_OPT_KAHAN',
    'TB_R_SAMPLE','TB_LR_COOLDOWN_START','TB_LR_COOLDOWN_FLOOR',
    'TB_BOTTOM_ISOLATE','TB_BOTTOM_ISOLATE_RAMP','TB_BOTTOM_LR_MULT',
)


_DATA_ENV_KEYS = (
    'TB_STREAM','TB_STREAM_REVISION','TB_STREAM_ND','TB_STREAM_DOMAIN','TB_STREAM_TEXT_FIELD',
    'TB_STREAM_PRESET','TB_STREAM_MIX','TB_STREAM_VAL_SOURCE',
    'TB_STREAM_SHUFFLE_BUFFER','TB_STREAM_SHUFFLE_SEED',
    'TB_CODE_STREAM','TB_CODE_STREAM_REVISION','TB_CODE_DATA_DIR','TB_CODE_STREAM_ND',
    'TB_CODE_TEXT_FIELD','TB_CODE_REPO_FIELD','TB_CODE_PATH_FIELD','TB_CODE_LANG_FIELD',
    'TB_CODE_GROUP_REPO','TB_CODE_LANG_FILTER','TB_CODE_SHARE','TB_CODE_SHARE_START',
    'TB_CODE_RAMP_TOKENS','TB_CODE_VAL_TOKENS','TB_CODE_VAL_PATH',
    'TB_COT_URL','TB_COT_PATH','TB_SFT_MASK',
)


def data_manifest(cfg):
    env = {key: os.environ.get(key, '') for key in _DATA_ENV_KEYS}
    # A disabled code stream has no code/general mixing coefficient. Treat an
    # explicit zero exactly like an unset variable so equivalent launchers do
    # not create a spurious checkpoint data-manifest mismatch.
    if not env.get('TB_CODE_STREAM', '').strip():
        env['TB_CODE_SHARE'] = ''
        env['TB_CODE_SHARE_START'] = ''
    return {
        'version': 4,
        'env': env,
        'stream_mix_signature': str(getattr(cfg, 'stream_mix_signature', '') or ''),
        'tokenizer': str(cfg.tokenizer),
        'vocab': int(cfg.vocab),
        'packing': {
            'fineweb': 'eos-v1',
            'code': 'eos-v1',
            'windowing': 'contiguous-no-drop-v2',
        },
    }


def _normalize_manifest_env(values):
    """Canonicalize legacy JSON nulls for unset environment variables."""
    return {
        str(key): ('' if value is None else value)
        for key, value in (values or {}).items()
    }


MANIFEST_COMPAT_VERSION = 'null-empty-v4'


def validate_data_manifest(saved_run_state, cfg):
    """Reject accidental data changes while permitting an intentional mix ramp.

    ``TB_RESUME_DATA_MIX=override`` is deliberately narrow: it may change only
    ``TB_CODE_SHARE`` and ``TB_CODE_SHARE_START``. ``TB_RESUME_DATASET=override``
    separately permits an intentional general-stream transition and causes the
    caller to discard only that source's old cursor/prefetch state. Tokenizer,
    code dataset identity/fields, and packing remain locked.
    """
    saved = (saved_run_state or {}).get('data')
    if not saved:
        print('resume: checkpoint predates data manifest; current data contract will be recorded on next save', flush=True)
        return False
    current = data_manifest(cfg)
    diffs = []
    changed_env = []
    a = _normalize_manifest_env(saved.get('env'))
    b = _normalize_manifest_env(current.get('env'))
    for key in sorted(set(a) | set(b)):
        if a.get(key, '') != b.get(key, ''):
            changed_env.append(key)
            diffs.append(f'{key}: {a.get(key, "")!r} -> {b.get(key, "")!r}')
    saved_mix_sig = str(saved.get('stream_mix_signature', '') or '')
    current_mix_sig = str(current.get('stream_mix_signature', '') or '')
    if saved_mix_sig != current_mix_sig:
        changed_env.append('STREAM_MIX_SIGNATURE')
        diffs.append(f'stream_mix_signature: {saved_mix_sig!r} -> {current_mix_sig!r}')
    tokenizer_changed = saved.get('tokenizer') != current.get('tokenizer')
    if tokenizer_changed:
        diffs.append(f"tokenizer: {saved.get('tokenizer')!r} -> {current.get('tokenizer')!r}")
    saved_fineweb_packing = (saved.get('packing') or {}).get('fineweb', 'legacy-concat-v1')
    current_fineweb_packing = current['packing']['fineweb']
    packing_changed = saved_fineweb_packing != current_fineweb_packing
    saved_windowing = (saved.get('packing') or {}).get('windowing', 'jitter-drop-v1')
    current_windowing = current['packing']['windowing']
    windowing_changed = saved_windowing != current_windowing
    packing_migration_accepted = False
    if packing_changed:
        if (saved_fineweb_packing == 'legacy-concat-v1'
                and current_fineweb_packing == 'eos-v1'
                and os.environ.get('TB_STREAM_REPACK_ON_RESUME', '0') == '1'):
            packing_migration_accepted = True
            print('resume: explicit FineWeb packing migration accepted '
                  '(legacy-concat-v1 -> eos-v1)', flush=True)
        else:
            diffs.append(f'FineWeb packing: {saved_fineweb_packing!r} -> '
                         f'{current_fineweb_packing!r} (set TB_STREAM_REPACK_ON_RESUME=1 '
                         'for the one-time legacy migration)')
    if windowing_changed:
        diffs.append(f'stream windowing: {saved_windowing!r} -> {current_windowing!r}')
    if diffs and os.environ.get('TB_RESUME_DATASET', 'lock').strip().lower() == 'extend':
        change = None
        if (set(changed_env) <= {'TB_STREAM_MIX', 'STREAM_MIX_SIGNATURE'}
                and not tokenizer_changed and not packing_changed and not windowing_changed):
            try:
                change = _mix_extension(json.loads(a.get('TB_STREAM_MIX') or '[]'),
                                        json.loads(b.get('TB_STREAM_MIX') or '[]'))
            except (ValueError, TypeError):
                change = None
        if not change:
            raise RuntimeError(
                'TB_RESUME_DATASET=extend accepts only appended sources and SFT '
                'role-template changes (existing sources otherwise identical apart '
                'from weight): ' + '; '.join(diffs[:20]))
        cfg.mix_rerendered = list(change["rerendered"])
        print(f"resume: data-mix extension accepted (added={change['added']}, "
              f"re-rendered={change['rerendered']}); every existing cursor is kept, "
              "unchanged sources keep their validation banks", flush=True)
        return 'extend'
    if diffs:
        mix_override = os.environ.get('TB_RESUME_DATA_MIX', 'lock').strip().lower() == 'override'
        allowed_mix_keys = {'TB_CODE_SHARE', 'TB_CODE_SHARE_START'}
        if (mix_override and not tokenizer_changed
                and (not packing_changed or packing_migration_accepted)
                and not windowing_changed
                and set(changed_env).issubset(allowed_mix_keys)):
            suffix = ' alongside explicit FineWeb packing migration' \
                if packing_migration_accepted else ''
            print('resume: data-mix override accepted' + suffix +
                  ' (' + '; '.join(diffs) + ')', flush=True)
            return 'mix'
        dataset_override_raw = os.environ.get('TB_RESUME_DATASET', 'lock')
        dataset_override = dataset_override_raw.strip().lower() == 'override'
        allowed_dataset_keys = {
            'TB_STREAM', 'TB_STREAM_REVISION', 'TB_STREAM_ND', 'TB_STREAM_DOMAIN',
            'TB_STREAM_TEXT_FIELD', 'TB_STREAM_PRESET', 'TB_STREAM_MIX',
            'TB_STREAM_VAL_SOURCE', 'TB_STREAM_SHUFFLE_BUFFER',
            'TB_STREAM_SHUFFLE_SEED', 'STREAM_MIX_SIGNATURE', 'TB_SFT_MASK',
        }
        changed = set(changed_env)
        dataset_changes = changed & allowed_dataset_keys
        other_changes = changed - allowed_dataset_keys
        mix_part_allowed = not other_changes or (
            mix_override and other_changes.issubset(allowed_mix_keys)
        )
        if (dataset_override and dataset_changes and mix_part_allowed
                and not tokenizer_changed
                and (not packing_changed or packing_migration_accepted)):
            print('resume: intentional general-dataset transition accepted; '
                  'old general cursor/prefetch will be discarded (' +
                  '; '.join(diffs) + ')', flush=True)
            return 'dataset'
        raise RuntimeError(
            'data manifest mismatch on ordinary resume: ' + '; '.join(diffs[:20]) +
            f"; transition controls: TB_RESUME_DATASET={dataset_override_raw!r}, "
            f"TB_RESUME_DATA_MIX={os.environ.get('TB_RESUME_DATA_MIX', 'lock')!r}"
        )
    if packing_migration_accepted:
        print('resume: migrated EOS document-packing contract will be recorded on next save', flush=True)
    elif saved != current:
        print('resume: normalized legacy data manifest null/unset entries', flush=True)
    return False


_MIX_EXTENSION_LOCKED_KEYS = ("dataset", "revision", "data_dir", "file_selector", "text_field",
                              "message_field", "domain_key", "domains", "filters", "dedup")


def _mix_extension(saved_specs, current_specs):
    """Describe a resumable mix change, or None.

    Returns {"added": [...], "rerendered": [...]}: sources new to the mix, and
    SFT (message_field) sources whose only change is the role template. Every
    saved source must still be present with identical identity, routing and
    filters; weights may change (they renormalize). Cursors and validation
    banks are mapped by name, so the order of old sources is irrelevant.
    """
    saved = _canonical_stream_specs(saved_specs)
    current = _canonical_stream_specs(current_specs)
    by_name = {spec["name"]: spec for spec in current}
    rerendered = []
    for spec in saved:
        cur = by_name.get(spec["name"])
        if cur is None:
            return None
        if any(cur.get(key, "") != spec.get(key, "") for key in _MIX_EXTENSION_LOCKED_KEYS):
            return None
        if cur.get("template", "") != spec.get("template", ""):
            if not str(cur.get("message_field", "") or "").strip():
                return None
            rerendered.append(spec["name"])
    old_names = {spec["name"] for spec in saved}
    added = [spec["name"] for spec in current if spec["name"] not in old_names]
    if not added and not rerendered:
        return None
    return {"added": added, "rerendered": rerendered}


def prepare_stream_state_for_data_change(stream_state, change_kind):
    """Drop incompatible general-stream state without mutating checkpoint data."""
    if change_kind != 'dataset' or stream_state is None:
        return stream_state
    wrapped = (isinstance(stream_state, dict)
               and stream_state.get('version') == 2
               and ('fineweb' in stream_state or 'code' in stream_state))
    if not wrapped:
        # Legacy stream_state is the general source directly.
        return None
    return {**stream_state, 'fineweb': None}


def architecture_manifest(cfg):
    return {
        'version': 1,
        'phase': os.environ.get('TB_ARCH_PHASE', 'phase_b').strip() or 'phase_b',
        'shape': {
            'vocab': int(cfg.vocab), 'D': int(cfg.D), 'L': int(cfg.L),
            'heads': int(cfg.n_head), 'mult': float(getattr(cfg, 'mult', 2.5)), 'max_t': int(cfg.max_t),
            'rope': bool(cfg.rope), 'alibi': bool(cfg.alibi), 'hybrid': bool(getattr(cfg, 'hybrid', True)),
            'local_conv': bool(getattr(cfg, 'local_conv', False)),
            'local_conv_rank': int(getattr(cfg, 'local_conv_rank', 0)),
            'local_conv_kernels': str(getattr(cfg, 'local_conv_kernels', '')),
            'local_conv_every': int(getattr(cfg, 'local_conv_every', 1)),
            'local_conv_depth': int(getattr(cfg, 'local_conv_depth', 1)),
            'local_conv_stage2_kernels': str(getattr(cfg, 'local_conv_stage2_kernels', '')),
            'local_conv_stage3_kernels': str(getattr(cfg, 'local_conv_stage3_kernels', '')),
            'lmm': bool(getattr(cfg, 'lmm', False)),
            'lmm_rank': int(getattr(cfg, 'lmm_rank', 0)),
            'lmm_layers': str(getattr(cfg, 'lmm_layers', '')),
            'lmm_write': str(getattr(cfg, 'lmm_write', 'delta')),
            'lmm_lesson_source': str(getattr(cfg, 'lmm_lesson_source', 'lookup')),
            'lexical_semantic': bool(getattr(cfg, 'lexical_semantic', False)),
            'lexical_semantic_rank': int(getattr(cfg, 'lexical_semantic_rank', 0)),
            'v41_engram': bool(getattr(cfg, 'v41_engram', False)),
            'v41_engram_buckets': int(getattr(cfg, 'v41_engram_buckets', 0)),
            'v41_engram_rank': int(getattr(cfg, 'v41_engram_rank', 0)),
            'v41_ced': bool(getattr(cfg, 'v41_ced', False)),
            'v41_csa2': bool(getattr(cfg, 'v41_csa2', False)),
            'v41_csa2_mode': os.environ.get('TB_V41_CSA2_MODE', 'blend').strip().lower(),
            'walk_cycle_guard': os.environ.get('TB_WALK_CYCLE_GUARD', '0') == '1',
            'walk_cycle_escape': os.environ.get('TB_WALK_CYCLE_ESCAPE', '0') == '1',
            'walk_cycle_rank': int(os.environ.get('TB_WALK_CYCLE_RANK', '32')),
            'walk_cycle_lags': str(os.environ.get('TB_WALK_CYCLE_LAGS', '1,2,4,8')),
            'pause_tokens': int(os.environ.get('TB_PAUSE_N', '0')),
            'pause_every': (int(os.environ.get('TB_PAUSE_EVERY', '512'))
                            if int(os.environ.get('TB_PAUSE_N', '0')) > 0 else 0),
            'pause_huginn': os.environ.get('TB_PAUSE_HUGINN', '0') == '1',
            'pause_rank': (int(os.environ.get('TB_PAUSE_RANK', '32'))
                           if os.environ.get('TB_PAUSE_HUGINN', '0') == '1' else 0),
            'r_adapter': os.environ.get('TB_R_ENABLE', '0') == '1',
            'r_start': int(os.environ.get('TB_R_START', '0') or 0),
            'r_end': int(os.environ.get('TB_R_END', '0') or 0),
            'r_max': int(os.environ.get('TB_R_MAX', '0') or 0),
            'r_backprop': int(os.environ.get('TB_R_BACKPROP', '0') or 0),
        },
        'env': {key: os.environ.get(key, '') for key in _STRUCTURAL_ENV_KEYS},
    }


def validate_architecture_manifest(saved_run_state, cfg):
    saved = (saved_run_state or {}).get('architecture')
    if not saved:
        print('resume: checkpoint predates architecture manifest; current structure will be recorded on next save', flush=True)
        return
    current = architecture_manifest(cfg)
    diffs = []
    saved_shape = saved.get('shape', {}) or {}
    current_shape = current.get('shape', {}) or {}
    if saved.get('phase', 'phase_b') != current.get('phase', 'phase_b'):
        diffs.append(f"phase: {saved.get('phase', 'phase_b')!r} -> {current.get('phase', 'phase_b')!r}")
    additive_shape = {
        'local_conv', 'local_conv_rank', 'local_conv_kernels', 'local_conv_every',
        'local_conv_depth', 'local_conv_stage2_kernels', 'local_conv_stage3_kernels',
        'lmm', 'lmm_rank', 'lmm_layers', 'lmm_write', 'lmm_lesson_source',
        'lexical_semantic', 'lexical_semantic_rank',
        'v41_engram', 'v41_engram_buckets', 'v41_engram_rank', 'v41_ced', 'v41_csa2',
        'v41_csa2_mode',
        'walk_cycle_guard', 'walk_cycle_escape', 'walk_cycle_rank', 'walk_cycle_lags',
        'pause_tokens', 'pause_every', 'pause_huginn', 'pause_rank',
        'r_adapter', 'r_start', 'r_end', 'r_max', 'r_backprop',
    }
    for key in sorted(set(saved_shape) | set(current_shape)):
        # Adding the zero-gated local branch is an intentional resumable
        # extension. Older checkpoints have no shape entries for it.
        if key in additive_shape and key not in saved_shape:
            continue
        # A checkpoint that explicitly recorded local_conv=false is also safe
        # to extend: the new branch has zero gates and optimizer transplantation
        # preserves every pre-existing parameter state.
        if (key == 'local_conv' and not bool(saved_shape.get(key, False))
                and bool(current_shape.get(key, False))):
            continue
        # Retiring local conv is the deliberate mirror image: a checkpoint
        # that trained the branch resumes with it unbuilt (TB_LOCAL_CONV=0).
        # The resume path discards its conv keys loudly and transplants every
        # other moment by name; kernel/rank/every/depth strings still match
        # because the contract re-exports the checkpoint's own values.
        if (key == 'local_conv' and bool(saved_shape.get(key, False))
                and not bool(current_shape.get(key, False))):
            continue
        # Same deliberate mirror for the lexical-semantic adapter, whose route
        # shut (zero residual contribution on all banks): retire it and the
        # manifest stops requiring it.
        if (key == 'lexical_semantic' and bool(saved_shape.get(key, False))
                and not bool(current_shape.get(key, False))):
            continue        # LMM has no effect without an explicitly supplied external memory.
        # Adding its learnable encoder/injector is therefore a safe, resumable
        # sidecar extension; changing an already-active LMM remains strict.
        if (key == 'lmm' and not bool(saved_shape.get(key, False))
                and bool(current_shape.get(key, False))):
            continue
        if (key.startswith('lmm_') and not bool(saved_shape.get('lmm', False))
                and bool(current_shape.get('lmm', False))):
            continue
        if (key == 'lexical_semantic' and not bool(saved_shape.get(key, False))
                and bool(current_shape.get(key, False))):
            continue
        if (key.startswith('lexical_semantic_')
                and not bool(saved_shape.get('lexical_semantic', False))
                and bool(current_shape.get('lexical_semantic', False))):
            continue
        # Engram is a zero-gated additive sidecar.  It can be introduced on a
        # normal continuation resume; existing backbone/optimizer moments are
        # retained and only the new parameters start fresh.
        if (key == 'v41_engram' and not bool(saved_shape.get(key, False))
                and bool(current_shape.get(key, False))):
            continue
        if (key.startswith('v41_engram_')
                and not bool(saved_shape.get('v41_engram', False))
                and bool(current_shape.get('v41_engram', False))):
            continue
        if key in {'v41_ced', 'v41_csa2'} and not bool(saved_shape.get(key, False)) \
                and bool(current_shape.get(key, False)):
            continue
        if key == 'v41_csa2_mode' and not bool(saved_shape.get('v41_csa2', False)) \
                and bool(current_shape.get('v41_csa2', False)):
            continue
        if (key == 'walk_cycle_guard' and not bool(saved_shape.get(key, False))
                and bool(current_shape.get(key, False))):
            continue
        if (key == 'walk_cycle_escape' and not bool(saved_shape.get(key, False))
                and bool(current_shape.get(key, False))):
            continue
        if key.startswith('walk_cycle_') and key != 'walk_cycle_guard' \
                and not bool(saved_shape.get('walk_cycle_guard', False)) \
                and bool(current_shape.get('walk_cycle_guard', False)):
            continue
        # Pause latents are small fresh parameters with no effect on any
        # pre-existing weight: introducing them on a normal continuation
        # resume is safe (backbone/optimizer moments transplant by name).
        # Changing the count/schedule afterwards is strict (it moves every
        # downstream position).
        if key in {'pause_tokens', 'pause_every'} \
                and int(saved_shape.get(key, 0)) == 0 \
                and int(current_shape.get(key, 0)) != 0:
            continue
        # Same one-way rule for the Huginn schedule mode and its rank: a
        # pause run may adopt them (fresh thought params, backbone moments
        # transplant by name); changing them afterwards stays strict.
        if key in {'pause_huginn', 'pause_rank'} \
                and not bool(saved_shape.get(key, False)) \
                and bool(current_shape.get(key, False)):
            continue
        # Narrow-R adapter is a zero-init additive sidecar: a flat checkpoint
        # may adopt it (fresh params, backbone moments transplant by name);
        # geometry changes afterwards stay strict.
        if key in {'r_adapter', 'r_start', 'r_end', 'r_max', 'r_backprop'} \
                and not bool(saved_shape.get('r_adapter', False)) \
                and bool(current_shape.get('r_adapter', False)):
            continue
        # RETIRING narrow-R is the deliberate mirror image: a checkpoint that
        # trained the adapter resumes with it unbuilt (TB_R_ENABLE=0). Its
        # tensors have nowhere to go and resume() discards them loudly; the
        # manifest no longer requires the recurrent sidecar. Reconfiguring R
        # while it stays ON remains strict (checked above).
        if key in {'r_adapter', 'r_start', 'r_end', 'r_max', 'r_backprop'} \
                and bool(saved_shape.get('r_adapter', False)) \
                and not bool(current_shape.get('r_adapter', False)):
            continue
        # Increasing depth only appends zero-gated stages; shrinking depth would
        # discard parameters and therefore remains a real incompatibility.
        if key == 'local_conv_depth':
            try:
                if int(current_shape.get(key, 1)) >= int(saved_shape.get(key, 1)):
                    continue
            except (TypeError, ValueError):
                pass
        if saved_shape.get(key) != current_shape.get(key):
            diffs.append(f'shape.{key}: {saved_shape.get(key)!r} -> {current_shape.get(key)!r}')
    saved_env = _normalize_manifest_env(saved.get('env'))
    current_env = _normalize_manifest_env(current.get('env'))
    legacy_runtime_additions = {
        'TB_WALK_NORM', 'TB_WALK_NORM_MODE', 'TB_WALK_HEADS', 'TB_WALK_FP32',
        'TB_WALK_MIX', 'TB_WALK_MIX_DEPTH', 'TB_WALK_MIX_GAIN',
        'TB_WALK_GLOB', 'TB_WALK_COUPLE', 'TB_WALK_GATE',
        'TB_WALK_SELECTIVE_MEM', 'TB_WALK_GEOM', 'TB_WALK_PLACE',
        'TB_WALK_HURST', 'TB_WALK_AFFINE',
        'TB_WALK_MOTIF', 'TB_WALK_MOTIF_COUNT', 'TB_WALK_MOTIF_RANK',
        'TB_WALK_MOTIF_PRED', 'TB_WALK_MOTIF_V2', 'TB_WALK_MOTIF_ORTHO',
        'TB_WALK_MOTIF_SHARED', 'TB_WALK_MOTIF_CONF', 'TB_WALK_MOTIF_TOPK',
        'TB_WALK_MOTIF_MOD', 'TB_WDR', 'TB_WDR_LAYER', 'TB_WDR_RANK',
        # Content-adaptive scale mixer: zero-init (bitwise no-op) introduction,
        # like every other zero-gated sidecar; geometry changes afterwards stay
        # strict.
        'TB_WALK_ADAPT_SCALE', 'TB_WALK_ADAPT_LAGS', 'TB_WALK_ADAPT_RANK',
        'TB_WALK_CONV_BASELINE',
        'TB_WALK_CYCLE_GUARD', 'TB_WALK_CYCLE_ESCAPE', 'TB_WALK_CYCLE_RANK', 'TB_WALK_CYCLE_LAGS',
        'TB_WALK_CYCLE_THRESHOLD', 'TB_WALK_CYCLE_PERSISTENCE', 'TB_WALK_CYCLE_MAX_STRENGTH',
        'TB_WALK_CYCLE_OPEN_TARGET', 'TB_WALK_CYCLE_OPEN_WEIGHT',
        # The lexical-semantic branch is an exact zero-gated additive sidecar.
        'TB_LEXICAL_SEMANTIC', 'TB_LEXICAL_SEMANTIC_RANK',
        # V4.1 additions may be introduced on a legacy checkpoint; the
        # Engram starts at an exact no-op and KV FP4 has no parameters.
        'TB_V41_ENGRAM', 'TB_V41_ENGRAM_BUCKETS', 'TB_V41_ENGRAM_RANK', 'TB_V41_KV_FP4',
        'TB_V41_CED', 'TB_V41_CSA2', 'TB_V41_CSA2_MODE',
        # Pause latents likewise (shape tolerance above covers the tensor
        # side); later changes stay strict.
        'TB_PAUSE_N', 'TB_PAUSE_EVERY', 'TB_PAUSE_HUGINN', 'TB_PAUSE_RANK',
        # Narrow-R adapter likewise: first introduction is a zero-init no-op
        # (fresh params, backbone moments transplant by name); geometry changes
        # afterwards stay strict.
        'TB_R_ENABLE', 'TB_R_START', 'TB_R_END', 'TB_R_MEAN', 'TB_R_MAX',
        'TB_R_BACKPROP', 'TB_R_INSTR_ONLY', 'TB_R_KV_SHARE',
        'TB_R_UPDATE', 'TB_R_GAIN',
        # Block-loop geometry likewise: first introduction is a zero-init
        # no-op; later changes stay strict. (Older manifests predate these
        # keys entirely; the key-not-in-saved rule above already covers that.)
        'TB_LOOP_ENABLE', 'TB_LOOP_START', 'TB_LOOP_END', 'TB_LOOP_PASSES',
    }
    r_v2_defaults = {'TB_R_UPDATE': 'blend', 'TB_R_GAIN': 'per_pass'}
    for key in sorted(set(saved_env) | set(current_env)):
        # R v2 switches: a missing saved value means the original rule. A
        # trained R checkpoint may move ONE WAY to the v2 rule (blend ->
        # delta, per_pass -> shared); this deliberately changes the R
        # contribution, so it is printed loudly. Everything else is strict.
        if key in r_v2_defaults and saved_env.get('TB_R_ENABLE', '') == '1':
            old_v = saved_env.get(key, '') or r_v2_defaults[key]
            new_v = current_env.get(key, '') or r_v2_defaults[key]
            if old_v == new_v:
                continue
            if old_v == r_v2_defaults[key] and new_v in ('delta', 'shared'):
                print(f"resume: narrow-R {key} {old_v} -> {new_v} (one-way R v2 "
                      "migration; changes the R residual, not any tensor)", flush=True)
                continue
            diffs.append(f"env.{key}: {old_v!r} -> {new_v!r}")
            continue
        # Pre-fix checkpoints did not record these forward-time Walk controls.
        # Accept the first resume and persist them in the next manifest; after
        # that, ordinary resume validation becomes strict as usual.
        if key in legacy_runtime_additions and key not in saved_env:
            continue
        cycle_keys = {
            'TB_WALK_CYCLE_GUARD', 'TB_WALK_CYCLE_ESCAPE', 'TB_WALK_CYCLE_RANK', 'TB_WALK_CYCLE_LAGS',
            'TB_WALK_CYCLE_THRESHOLD', 'TB_WALK_CYCLE_PERSISTENCE',
            'TB_WALK_CYCLE_MAX_STRENGTH', 'TB_WALK_CYCLE_OPEN_TARGET',
            'TB_WALK_CYCLE_OPEN_WEIGHT',
        }
        adding_cycle_guard = (
            saved_env.get('TB_WALK_CYCLE_GUARD', '0') != '1'
            and current_env.get('TB_WALK_CYCLE_GUARD', '0') == '1'
        )
        if adding_cycle_guard and key in cycle_keys:
            continue
        # One-way walk-norm migration renorm -> clip with the value fixed at
        # 0.3. At the switch instant every token already carries RMS ~= 0.3
        # (renorm forces it), so the clamp factor is exactly 1.0 and outputs
        # are bitwise identical; afterwards weak tokens may fade below 0.3
        # instead of being amplified up to it. The reverse (clip -> renorm)
        # would amplify quiet tokens up and is deliberately rejected, as is
        # any simultaneous value change (TB_WALK_NORM itself is still compared
        # strictly below).
        migrating_walk_norm = (
            saved_env.get('TB_WALK_NORM_MODE', '') == 'renorm'
            and current_env.get('TB_WALK_NORM_MODE', '') == 'clip'
        )
        if migrating_walk_norm and key == 'TB_WALK_NORM_MODE':
            print('resume: walk norm mode renorm -> clip (identical where loud, '
                  'fades where weak)', flush=True)
            continue
        # Retiring lex flips the structural env flag 1 -> 0 alongside the
        # shape flag (the shape tolerance above covers the tensor side, and
        # resume() discards the orphaned keys loudly). One-way only: 0 -> 1
        # would silently add an untrained branch mid-run.
        retiring_lex_env = (
            saved_env.get('TB_LEXICAL_SEMANTIC', '') == '1'
            and current_env.get('TB_LEXICAL_SEMANTIC', '') == '0'
        )
        if retiring_lex_env and key == 'TB_LEXICAL_SEMANTIC':
            continue
        # Introducing pause latents flips the structural env flags from
        # unset/0 alongside the shape flags (the shape tolerance above
        # covers the tensor side; the new parameters start fresh while
        # backbone/optimizer moments transplant by name). One-way only:
        # changing the schedule afterwards stays strict. Huginn mode
        # counts as introducing even when TB_PAUSE_N itself stays unset
        # (the mode forces the count internally).
        introducing_pause_env = (
            (saved_env.get('TB_PAUSE_N', '') in ('', '0')
             and current_env.get('TB_PAUSE_N', '') not in ('', '0'))
            or (saved_env.get('TB_PAUSE_HUGINN', '') != '1'
                and current_env.get('TB_PAUSE_HUGINN', '') == '1')
        )
        if introducing_pause_env and key in {'TB_PAUSE_N', 'TB_PAUSE_EVERY',
                                              'TB_PAUSE_HUGINN', 'TB_PAUSE_RANK'}:
            continue
        # Introducing narrow-R flips the structural env flags from unset/0
        # alongside the shape flags (fresh adapter params, backbone/optimizer
        # moments transplant by name). One-way only: geometry changes
        # afterwards stay strict.
        introducing_r_env = (
            saved_env.get('TB_R_ENABLE', '') != '1'
            and current_env.get('TB_R_ENABLE', '') == '1'
        )
        if introducing_r_env and key in {'TB_R_ENABLE', 'TB_R_START', 'TB_R_END',
                                         'TB_R_MEAN', 'TB_R_MAX', 'TB_R_BACKPROP',
                                         'TB_R_INSTR_ONLY', 'TB_R_KV_SHARE'}:
            continue
        # TB_R_MEAN only sets how many extra passes are sampled (training)
        # and run (eval); it owns no parameters and no pass_gain index moves
        # (pass p always uses pass_gain[p]). Lowering it to drop passes whose
        # learned gains sit at ~0 is a pure compute change for those passes,
        # so allow it while R stays enabled within the saved r_max.
        if (key == 'TB_R_MEAN'
                and saved_env.get('TB_R_ENABLE', '') == '1'
                and current_env.get('TB_R_ENABLE', '') == '1'):
            try:
                new_mean = int(current_env.get('TB_R_MEAN', ''))
                saved_max = int(saved_env.get('TB_R_MAX', '') or 0)
            except ValueError:
                new_mean, saved_max = -1, 0
            if 0 <= new_mean <= saved_max:
                print(f"resume: narrow-R mean {saved_env.get('TB_R_MEAN')!r} -> "
                      f"{current_env.get('TB_R_MEAN')!r} (sampled/eval depth only; "
                      "no parameter or gain-index change)", flush=True)
                continue
        # Introducing the block loop flips its structural env flags from
        # unset/0 alongside (fresh controller params, backbone/optimizer
        # moments transplant by name). One-way only: geometry changes
        # afterwards stay strict.
        introducing_loop_env = (
            saved_env.get('TB_LOOP_ENABLE', '') != '1'
            and current_env.get('TB_LOOP_ENABLE', '') == '1'
        )
        if introducing_loop_env and key in {'TB_LOOP_ENABLE', 'TB_LOOP_START',
                                            'TB_LOOP_END', 'TB_LOOP_PASSES'}:
            continue
        # RETIRING the recurrent paths is the deliberate mirror image of
        # introducing them: a checkpoint that trained the block loop or narrow-R
        # resumes with it unbuilt (TB_LOOP_ENABLE=0 / TB_R_ENABLE=0). resume()
        # discards the orphaned controller/adapter keys loudly, so the env
        # flags and geometry no longer have to match. The reverse (staying ON
        # while being reconfigured) keeps the strict checks above.
        if (str(key).startswith('TB_R_') and saved_env.get('TB_R_ENABLE', '') == '1'
                and current_env.get('TB_R_ENABLE', '') != '1'):
            continue
        if (str(key).startswith('TB_LOOP_') and saved_env.get('TB_LOOP_ENABLE', '') == '1'
                and current_env.get('TB_LOOP_ENABLE', '') != '1'):
            continue
        # "Unset" and "explicitly off" are the same state for the recurrence
        # switches. A checkpoint launched without the flag records it as an
        # empty string; this continuation writes an explicit "0". Treat both as
        # off so the forced-off line resumes cleanly instead of tripping on
        # env.TB_LOOP_ENABLE: '' -> '0'.
        if key in {'TB_LOOP_ENABLE', 'TB_R_ENABLE'} \
                and (saved_env.get(key, '') or '0') == (current_env.get(key, '') or '0'):
            continue
        if saved_env.get(key, '') != current_env.get(key, ''):
            diffs.append(f'env.{key}: {saved_env.get(key, "")!r} -> {current_env.get(key, "")!r}')
    if diffs:
        raise RuntimeError('architecture manifest mismatch on ordinary resume: ' + '; '.join(diffs[:20]))
    if saved != current:
        print('resume: normalized legacy architecture manifest null/unset entries', flush=True)


def optimizer_param_names(model, opt):
    by_id = {id(p): n for n, p in model.named_parameters()}
    return [[by_id.get(id(p), '<unnamed>') for p in group['params']] for group in opt.param_groups]


def _legacy_optimizer_names_for_source(model, source_state, opt_state):
    """Reconstruct pre-manifest optimizer group names for old Phase-A checkpoints."""
    present = []
    for name, p in model.named_parameters():
        if (name.startswith('draft.') or '.phase_walk.' in name or '.traj_reg.' in name
                or '.local_conv.' in name or '.local_conv_extra.' in name or name.startswith('lmm.')
                or name.startswith('lexical_semantic.')):
            continue
        if name in source_state and tuple(source_state[name].shape) == tuple(p.shape):
            present.append((name, p))
    walk = [n for n, _ in present if '.walk.' in n]
    mem = [n for n, _ in present if n.startswith('mem.')]
    other = [n for n, _ in present if '.walk.' not in n and not n.startswith('mem.')]
    groups = opt_state.get('param_groups', [])
    if len(groups) == 1:
        names = [n for n, _ in present]
        return [names] if len(names) == len(groups[0]['params']) else None
    inferred = ([walk] if walk else []) + ([mem] if mem else []) + [other]
    if len(inferred) != len(groups):
        return None
    for names, group in zip(inferred, opt_state['param_groups']):
        if len(names) != len(group['params']):
            return None
    return inferred


def transplant_optimizer_state_by_name(model, opt, source_opt, source_names):
    """Keep Adam moments for unchanged named parameters; new Phase-B params start fresh."""
    if not source_opt or not source_names:
        print('phase transition: source optimizer names unavailable; base optimizer will re-warm', flush=True)
        return 0, 0
    current_sd = opt.state_dict()
    current_names = optimizer_param_names(model, opt)
    current_id_by_name = {}
    param_by_name = dict(model.named_parameters())
    for names, group in zip(current_names, current_sd['param_groups']):
        for name, pid in zip(names, group['params']):
            current_id_by_name[name] = pid

    new_state = {}
    copied = skipped = 0
    for names, group in zip(source_names, source_opt['param_groups']):
        for name, old_pid in zip(names, group['params']):
            if name not in current_id_by_name or old_pid not in source_opt['state']:
                skipped += 1
                continue
            state = source_opt['state'][old_pid]
            p = param_by_name.get(name)
            compatible = True
            for key, value in state.items():
                if torch.is_tensor(value) and value.ndim > 0 and tuple(value.shape) != tuple(p.shape):
                    compatible = False
                    break
            if compatible:
                new_state[current_id_by_name[name]] = state
                copied += 1
            else:
                skipped += 1
    current_sd['state'] = new_state
    opt.load_state_dict(current_sd)
    print(f'phase transition: transplanted optimizer state for {copied:,} params; {skipped:,} skipped/new', flush=True)
    return copied, skipped


def _muon_zeropower(g, steps=3):
    """Small Newton-Schulz orthogonalization used by the optional Muon path."""
    # Work with the smaller row dimension to keep the Gram matrix cheap.
    transposed = g.shape[0] > g.shape[1]
    x = g.t() if transposed else g
    x = x.float()
    x = x / (x.norm() + 1e-7)
    # Stable coefficients from the quintic Newton-Schulz family. Three steps
    # are enough for the small local-conv matrices and avoid a large 1B tax.
    for _ in range(max(1, int(steps))):
        xtx = x @ x.transpose(-1, -2)
        x = 3.4445 * x - 4.7750 * (xtx @ x) + 2.0315 * (xtx @ xtx @ x)
    if transposed:
        x = x.t()
    return x


class MuonAdamW(torch.optim.Optimizer):
    """AdamW-compatible optimizer with optional Muon matrix groups.

    Groups marked ``_muon=True`` use momentum plus Newton-Schulz direction
    normalization for 2-D matrices; vectors, convolutions, and all other
    groups retain AdamW. This makes ``TB_OPTIMIZER=hybrid_muon`` a safe
    additive experiment for the new local branch while old Adam moments can
    still be transplanted by parameter name during resume.

    Tiny probe record (2026-09-07): AdamW-base + Muon-local was slightly worse
    than AdamW HIER+CNN (4.1938/4.1625 vs 4.1578/4.1281 at T=128/256) and
    slower, so it remains an explicit optimizer-reset fallback, not default.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), eps=1e-8,
                 weight_decay=0.0, muon_steps=3):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, muon_steps=muon_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = float(group['lr'])
            beta1, beta2 = group['betas']
            eps = float(group['eps'])
            wd = float(group['weight_decay'])
            use_muon = bool(group.get('_muon', False))
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('MuonAdamW does not support sparse gradients')
                state = self.state[p]
                if use_muon and p.ndim == 2:
                    momentum = state.get('momentum')
                    if momentum is None:
                        momentum = state['momentum'] = torch.zeros_like(p, dtype=torch.float32)
                    momentum.mul_(beta1).add_(grad.float(), alpha=1.0 - beta1)
                    update = _muon_zeropower(momentum, group.get('muon_steps', 3))
                    # Match Muon's width-aware update scale without making the
                    # narrow local rank dominate the base model.
                    scale = math.sqrt(max(1.0, p.shape[0] / max(1, p.shape[1])))
                    if wd:
                        p.mul_(1.0 - lr * wd)
                    p.add_(update.to(dtype=p.dtype), alpha=-lr * scale)
                    continue
                step = state.get('step', 0)
                if torch.is_tensor(step):
                    step = int(step.item())
                step += 1
                state['step'] = step
                exp_avg = state.get('exp_avg')
                exp_avg_sq = state.get('exp_avg_sq')
                if exp_avg is None:
                    exp_avg = state['exp_avg'] = torch.zeros_like(p, dtype=torch.float32)
                    exp_avg_sq = state['exp_avg_sq'] = torch.zeros_like(p, dtype=torch.float32)
                g = grad.float()
                exp_avg.mul_(beta1).add_(g, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
                bias1 = 1.0 - beta1 ** step
                bias2 = 1.0 - beta2 ** step
                denom = exp_avg_sq.sqrt() / math.sqrt(bias2)
                denom.add_(eps)
                if wd:
                    p.mul_(1.0 - lr * wd)
                p.addcdiv_(exp_avg.to(dtype=p.dtype), denom.to(dtype=p.dtype), value=-lr / bias1)
        return loss


def load_phase_checkpoint(model, opt, path, allow_structural=False, cfg=None):
    """Migrate a checkpoint into a new phase.

    The historical path is deliberately strict.  ``allow_structural`` is an
    explicit, separate contract for a real CED/CSA2 architecture change: only
    exact name/shape matches are copied, incompatible tensors are initialized
    by the target model, and the caller must run the transition loss gate.
    """
    target = model._orig_mod if hasattr(model, '_orig_mod') else model
    raw = _load_trusted_checkpoint(path, map_location='cpu')
    state = raw.get('model', raw)
    if any(k.startswith('_orig_mod.') for k in state):
        state = {k[len('_orig_mod.'):]: v for k, v in state.items()}
    target_state = target.state_dict()

    bad_shape = [k for k, v in state.items() if k in target_state and tuple(v.shape) != tuple(target_state[k].shape)]
    unexpected = [k for k in state if k not in target_state]
    missing = [k for k in target_state if k not in state]
    if allow_structural:
        if cfg is None:
            raise ValueError('structural phase transition requires the target config for core-shape validation')
        saved_arch = ((raw.get('run_state') or {}).get('architecture') or {}).get('shape', {})
        current_arch = architecture_manifest(cfg).get('shape', {})
        # Core tensor geometry must stay fixed.  The phase may change attention
        # plumbing and sidecars, but silently changing vocabulary/depth/width
        # would make the continuation scientifically uninterpretable.
        for key in ('vocab', 'D', 'L', 'heads', 'max_t'):
            if key in saved_arch and key in current_arch and saved_arch[key] != current_arch[key]:
                raise RuntimeError(f'structural phase transition cannot change core shape {key}: '
                                   f'{saved_arch[key]!r} -> {current_arch[key]!r}')
        target.load_state_dict({k: v for k, v in state.items()
                                if k in target_state and tuple(v.shape) == tuple(target_state[k].shape)},
                               strict=False)
        source_opt = raw.get('opt')
        source_names = raw.get('opt_param_names')
        if source_opt and not source_names:
            source_names = _legacy_optimizer_names_for_source(target, state, source_opt)
        copied, skipped = transplant_optimizer_state_by_name(target, opt, source_opt, source_names)
        restore_rng_state(raw.get('rng'))
        if getattr(target, 'draft', None) is not None and 'draft' in raw:
            target.draft.load_state_dict(raw['draft'], strict=False)
        audit = {
            'mode': 'structural_phase_transition',
            'source': os.path.abspath(path),
            'source_step': int(raw.get('step', 0)),
            'copied_model_keys': int(len(state) - len(bad_shape) - len(unexpected)),
            'dropped_shape_keys': sorted(bad_shape),
            'dropped_unexpected_keys': sorted(unexpected),
            'new_target_keys': sorted(missing),
            'optimizer_states_copied': int(copied),
            'optimizer_states_skipped': int(skipped),
        }
        print('PHASE TRANSITION structural load: '
              f"copied={audit['copied_model_keys']:,} dropped_shape={len(bad_shape):,} "
              f"dropped_unexpected={len(unexpected):,} new={len(missing):,}", flush=True)
        return {
            'source_step': int(raw.get('step', 0)),
            'cum_toks': int(raw.get('cum_toks', 0)),
            'stream_rows': raw.get('stream_rows'),
            'stream_state': raw.get('stream_state'),
            'n_shards': raw.get('n_shards'),
            'reference_ce': float(raw.get('best_val', float('nan'))),
            'audit': audit,
        }
    illegal_missing = [k for k in missing
                       if '.phase_walk.' not in k
                       and '.traj_reg.' not in k
                       and not k.startswith('kmerge_encoder.')]
    if bad_shape or unexpected or illegal_missing:
        raise RuntimeError(
            'TB_PHASE_CKPT must preserve the complete Phase-A architecture; only .phase_walk.*, .traj_reg.*, or kmerge_encoder.* may be new '
            f'(bad_shape={bad_shape[:5]}, unexpected={unexpected[:5]}, illegal_missing={illegal_missing[:5]})')
    target.load_state_dict(state, strict=False)

    source_opt = raw.get('opt')
    source_names = raw.get('opt_param_names')
    if source_opt and not source_names:
        source_names = _legacy_optimizer_names_for_source(target, state, source_opt)
        if source_names:
            print('phase transition: reconstructed legacy optimizer parameter names', flush=True)
    transplant_optimizer_state_by_name(target, opt, source_opt, source_names)
    restore_rng_state(raw.get('rng'))
    if getattr(target, 'draft', None) is not None and 'draft' in raw:
        target.draft.load_state_dict(raw['draft'])

    return {
        'source_step': int(raw.get('step', 0)),
        'cum_toks': int(raw.get('cum_toks', 0)),
        'stream_rows': raw.get('stream_rows'),
        'stream_state': raw.get('stream_state'),
        'n_shards': raw.get('n_shards'),
        'reference_ce': float(raw.get('best_val', float('nan'))),
        'audit': {'mode': 'phase_sidecar_transition', 'source': os.path.abspath(path),
                  'source_step': int(raw.get('step', 0))},
    }

def run_state(cfg):
    config = {
        key: value for key, value in vars(cfg).items()
        if not key.startswith("_") and isinstance(value, (str, int, float, bool))
    }
    state = {
        "version": 2,
        "source_sha256": _source_hash(),
        "config": config,
        "schedule": schedule_state(cfg),
        "train_phase": int(getattr(cfg, "train_phase", 1) or 1),
        "architecture": architecture_manifest(cfg),
        "data": data_manifest(cfg),
        # Resolved runtime controls (panel-visible effective values). Most of
        # these never enter cfg (read from the environment at their use
        # site), so without this snapshot post-hoc forensics cannot tell what
        # a checkpoint actually trained with. Informational only: validators
        # ignore unknown run_state keys, so this can neither pass nor fail a
        # resume. Mirrors RUNTIME_OVERRIDE_SPEC in walker_continue_contract.py
        # (tests enforce the match).
        "runtime": {key: os.environ.get(key, "")
                    for key in _RESOLVED_RUNTIME_KEYS},
    }
    audit = getattr(cfg, 'phase_transition_audit', None)
    if audit is not None:
        state['phase_transition'] = audit
    return state


def capture_rng_state():
    state = {
        "version": 1,
        "cpu": torch.random.get_rng_state().cpu(),
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        try:
            state["cuda"] = [s.cpu() for s in torch.cuda.get_rng_state_all()]
        except Exception:
            pass
    return state


def restore_rng_state(state):
    """Restore new RNG bundles and legacy CPU-only tensors."""
    if state is None:
        return
    if isinstance(state, torch.Tensor):
        torch.random.set_rng_state(state.cpu())
        return
    cpu = state.get("cpu")
    if cpu is not None:
        torch.random.set_rng_state(cpu.cpu())
    py_state = state.get("python")
    if py_state is not None:
        try:
            random.setstate(py_state)
        except Exception:
            pass
    cuda_state = state.get("cuda")
    if cuda_state is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all([s.cpu() for s in cuda_state])
        except Exception as e:
            print(f"resume: CUDA RNG state not restored ({e})", flush=True)


# -------------------------------------------------------------- checkpoints -

def _load_trusted_checkpoint(path, map_location):
    """Load this user's own training payload across PyTorch 2.5/2.6 defaults.

    Ongoing payloads intentionally include optimizer, Python/NumPy RNG, and
    stream snapshots, which PyTorch's 2.6 safe-weights loader cannot represent.
    This is only called for a local `TB_CKPT_PATH` selected by the user; do not
    use it on a checkpoint from an untrusted source.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # PyTorch versions before the weights_only keyword.
        return torch.load(path, map_location=map_location)

def load_weights_only(model, path, allow_partial=False):
    """Load model weights from a checkpoint without restoring optimizer or step."""
    target = model._orig_mod if hasattr(model, "_orig_mod") else model
    # Ongoing checkpoints also contain multi-GB optimizer state. Memory-map on
    # CPU so only model tensors are faulted in and copied to the target device.
    raw = _load_trusted_checkpoint(path, map_location="cpu")
    state = raw.get("model", raw)
    if any(key.startswith("_orig_mod.") for key in state):
        state = {key[len("_orig_mod."):]: value for key, value in state.items()}
    target_state = target.state_dict()
    dropped = []
    for key in list(state):
        if key in target_state and state[key].shape != target_state[key].shape:
            del state[key]
            dropped.append(key)
    missing_keys = [key for key in target_state if key not in state]
    unexpected_keys = [key for key in state if key not in target_state]
    if (dropped or missing_keys or unexpected_keys) and not allow_partial:
        raise RuntimeError(
            "TB_INIT_CKPT architecture differs from the target model "
            f"(dropped={len(dropped)}, missing={len(missing_keys)}, unexpected={len(unexpected_keys)}). "
            "Keep TB_ATTN_EVERY/TB_WALK_MIX_DEPTH/model dimensions unchanged, or set "
            "TB_INIT_ALLOW_PARTIAL=1 for an intentional architecture experiment."
        )
    missing, unexpected = target.load_state_dict(state, strict=False)
    print(f"initialized weights from {path} (source step {raw.get('step', 'unknown')}, "
          f"dropped={len(dropped)}, missing={len(missing)}, unexpected={len(unexpected)}); "
          "optimizer, scheduler, stream, and token counter are fresh", flush=True)
    return raw.get("step", 0)


class CompensatedAdamW(torch.optim.AdamW):
    """AdamW with Kahan-compensated updates for low-precision (bf16) params.

    This lineage stores every parameter in bf16 with no fp32 master copy.
    Round-to-nearest then drops any update smaller than half a bf16 ULP
    (ULP ~ |w| * 2^-7): at |w| ~ 0.02 and LR ~ 1e-4 most Adam steps vanish,
    and every unit-scale LayerNorm weight / Walk gamma froze exactly
    (BUG_REPORT "bf16 master freeze"). The loss worsens as the cosine LR
    decays. Here each step is computed in fp32 and the rounding residual is
    carried in a per-param ``kahan`` buffer (param dtype) and re-added next
    step, so the stored bf16 weights follow the fp32 trajectory to within one
    ULP at +1 bf16 copy of memory (Zamirai et al. 2020, "Revisiting BFloat16
    Training"). fp32 params take the plain path.

    State layout stays torch.optim.AdamW-compatible (step, exp_avg,
    exp_avg_sq; the extra ``kahan`` key is optional), so checkpoints move
    freely between this class and the stock optimizer in both directions.
    Moments keep the param dtype for that reason; their arithmetic is fp32.
    """

    def __init__(self, params, chunk_numel=1 << 26, **kwargs):
        kwargs.pop("fused", None)
        kwargs.pop("foreach", None)
        super().__init__(params, foreach=False, **kwargs)
        self._chunk = int(chunk_numel)

    def state_dict(self):
        # The residual is below half a bf16 ULP per element; dropping it at a
        # save costs at most one rounding, and keeps checkpoints the exact
        # size/format of stock AdamW (+1.8 GB otherwise at 1B scale).
        sd = super().state_dict()
        sd["state"] = {k: {n: v for n, v in s.items() if n != "kahan"}
                       for k, s in sd["state"].items()}
        return sd

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            if group.get("amsgrad", False) or group.get("maximize", False):
                raise RuntimeError("CompensatedAdamW supports plain AdamW only")
            lr = float(group["lr"])
            beta1, beta2 = group["betas"]
            eps = float(group["eps"])
            wd = float(group["weight_decay"])
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("CompensatedAdamW does not support sparse grads")
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = torch.tensor(0.0, dtype=torch.float32)
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                step_t = state["step"]
                if step_t.device.type != "cpu":
                    # Fused-AdamW state keeps step on the GPU; one transfer
                    # here avoids a device sync per parameter per step. Torch
                    # moves it back on load if the stock fused path resumes.
                    state["step"] = step_t = step_t.detach().to("cpu", torch.float32)
                step_t += 1
                step = float(step_t)
                # Per-parameter LR multiplier (TB_BOTTOM_LR_MULT sets it on the
                # bottom-of-stack params) without splitting optimizer groups,
                # so the saved group layout and state load unchanged.
                p_lr = lr * float(getattr(p, "_lr_mult", 1.0))
                bc1 = 1.0 - beta1 ** step
                bc2_sqrt = math.sqrt(1.0 - beta2 ** step)
                low_precision = p.dtype in (torch.bfloat16, torch.float16)
                if low_precision and "kahan" not in state:
                    state["kahan"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                # Chunk over the flattened tensor so the fp32 temporaries of a
                # 100M-element embedding never exceed a few hundred MB.
                flat_p = p.view(-1)
                flat_g = p.grad.reshape(-1)
                flat_m = state["exp_avg"].view(-1)
                flat_v = state["exp_avg_sq"].view(-1)
                flat_c = state["kahan"].view(-1) if low_precision else None
                for start in range(0, flat_p.numel(), self._chunk):
                    sl = slice(start, start + self._chunk)
                    # .float() aliases fp32 tensors, so the moment math below
                    # must stay out-of-place until the results are stored.
                    g = flat_g[sl].float()
                    m = torch.lerp(flat_m[sl].float(), g, 1.0 - beta1)
                    v = (flat_v[sl].float() * beta2).addcmul_(g, g, value=1.0 - beta2)
                    flat_m[sl].copy_(m)
                    flat_v[sl].copy_(v)
                    w = flat_p[sl].float()
                    update = (m / (v.sqrt_().div_(bc2_sqrt).add_(eps))).mul_(-p_lr / bc1)
                    if wd != 0.0:
                        update.add_(w, alpha=-p_lr * wd)
                    if flat_c is None:
                        flat_p[sl].copy_(w.add_(update))
                        continue
                    update.add_(flat_c[sl].float())
                    target = w + update
                    stored = target.to(p.dtype)
                    flat_c[sl].copy_(target.sub_(stored.float()))
                    flat_p[sl].copy_(stored)
        return loss


class Checkpointer:
    """Best + ongoing checkpoints plus a tiny metadata sidecar.

    The sidecar lets a legacy ongoing checkpoint acquire a schedule/config
    contract without rewriting its multi-gigabyte optimizer payload.
    """

    def __init__(self, path, best_path=None):
        self.ongoing = path
        self.best = best_path or (os.path.splitext(path)[0] + "_best.pt")
        self.metadata = path + ".metadata.json"
        # Optional extra payload (the LOTUS+ graph memory): a callable
        # with_opt -> dict merged into ongoing (with optimizer) and best /
        # snapshot (weights only) files. gmem_saved holds what resume found.
        self.extra_state = None
        self.gmem_saved = None

    def _extra(self, with_opt):
        return self.extra_state(with_opt) if callable(self.extra_state) else {}

    def _save_atomic(self, path, obj):
        tmp = path + ".tmp"
        torch.save(obj, tmp)
        os.replace(tmp, path)

    def _unwrap(self, model):
        return model._orig_mod if hasattr(model, "_orig_mod") else model

    def save_metadata(self, step, run_state_data):
        """Atomically persist lightweight resume metadata beside the checkpoint."""
        if run_state_data is None:
            return
        tmp = self.metadata + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as handle:
                json.dump({"version": 1, "step": int(step), "run_state": run_state_data}, handle)
            os.replace(tmp, self.metadata)
        except (OSError, TypeError, ValueError) as metadata_error:
            # The full checkpoint is already durable; metadata is an upgrade
            # aid and must never make a successful training step fail.
            print(f"checkpoint metadata write failed ({metadata_error})", flush=True)
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except OSError:
                pass

    def save_ongoing(self, model, opt, step, rng, best_val, doremi_state=None, n_shards=None, stream_rows=None, draft_opt=None, cum_toks=None, schedule_toks=None, data_step_base=0, run_state_data=None, stream_state_data=None):
        target = self._unwrap(model)
        payload = {
            "model": target.state_dict(), "opt": opt.state_dict(),
            "opt_param_names": optimizer_param_names(target, opt),
            "step": step, "rng": rng, "best_val": best_val,
        }
        if cum_toks is not None:
            payload["cum_toks"] = cum_toks
        if schedule_toks is not None:
            payload["schedule_toks"] = schedule_toks
        payload["data_step_base"] = int(data_step_base)
        if run_state_data is not None:
            payload["run_state"] = run_state_data
        if stream_state_data is not None:
            payload["stream_state"] = stream_state_data
        if doremi_state is not None:
            payload["doremi"] = doremi_state
        if n_shards is not None:
            payload["n_shards"] = n_shards
        if stream_rows is not None:
            payload["stream_rows"] = stream_rows
        if getattr(target, "draft", None) is not None:
            payload["draft"] = target.draft.state_dict()
        if draft_opt is not None:
            payload["draft_opt"] = draft_opt.state_dict()
        payload.update(self._extra(True))
        self._save_atomic(self.ongoing, payload)
        self.save_metadata(step, run_state_data)

    def save_best(self, model, opt, step, val, run_state_data=None):
        # best file is weights-only + metadata (no optimizer state): lean, and
        # resume restores best_val from it (optimizer state lives in ongoing)
        payload = {
            "model": self._unwrap(model).state_dict(),
            "step": step, "val": val,
        }
        if run_state_data is not None:
            payload["run_state"] = run_state_data
        payload.update(self._extra(False))
        self._save_atomic(self.best, payload)

    def save_snapshot(self, model, step, cum_toks):
        """Weights-only snapshot for permanent step recovery (small, uploadable
        per-step to HF without optimizer state). Unique local file per step;
        the uploader removes it after a successful upload."""
        target = self._unwrap(model)
        snap_path = f"{self.ongoing}.snap{step}"
        payload = {"model": target.state_dict(), "step": step, "cum_toks": cum_toks}
        if getattr(target, "draft", None) is not None:
            payload["draft"] = target.draft.state_dict()
        payload.update(self._extra(False))
        self._save_atomic(snap_path, payload)
        return snap_path

    def resume(self, model, opt):
        target = self._unwrap(model)
        if os.path.exists(self.ongoing):
            try:
                ck = _load_trusted_checkpoint(self.ongoing, map_location=DEVICE)
                sd = ck["model"]
                # accept both formats: plain keys (eager) or _orig_mod.-prefixed
                # (saved through an OptimizedModule wrapper by a compiled run)
                if any(k.startswith("_orig_mod.") for k in sd):
                    sd = {k[len("_orig_mod."):]: v for k, v in sd.items()}
                # Shape-compat filter: if walk params changed shape (e.g. multihead
                # upgrade), drop only those keys — the zero-init no-op takes over.
                # load_state_dict raises on ANY size mismatch even with strict=False,
                # which would otherwise nuke the whole checkpoint as "incompatible".
                target_sd = target.state_dict()
                n_dropped = 0
                n_loop_adapted = _adapt_loop_state_keys(sd, target_sd)
                if n_loop_adapted:
                    print(f"resume: loop pass-count change ({'; '.join(n_loop_adapted)}) — "
                          f"kept passes stay learned, new passes start fresh", flush=True)
                n_r_adapted = _adapt_r_state_keys(sd, target_sd)
                if n_r_adapted:
                    print(f"resume: R max change ({'; '.join(n_r_adapted)}) — "
                          f"kept passes stay learned, new passes start fresh", flush=True)
                for k in list(sd):
                    if k in target_sd and sd[k].shape != tuple(target_sd[k].shape):
                        del sd[k]
                        n_dropped += 1
                missing, unexpected = target.load_state_dict(sd, strict=False)
                # Older Phase-B builds stored dormant low-rank DRCG tensors in
                # the HIER superset.  The clean trainer intentionally has no
                # such parameters; discard only those legacy extras while
                # keeping the usual fail-closed behavior for every other
                # architecture mismatch.
                legacy_drcg = [k for k in unexpected if '.drcg.' in k or k.endswith('.drcg')]
                # Loop disabled but the checkpoint was trained with it on: the
                # controller tensors have nowhere to go in this model. Discard
                # them loudly (same as legacy extras) instead of failing — the
                # run explicitly asked for no recurrence. Optimizer moments for
                # unchanged params still transplant by name below.
                retired_loop = [k for k in unexpected if k.startswith('loop_controller.')]
                # Narrow-R disabled but the checkpoint was trained with it on:
                # same loud discard as the retired loop controller — the run
                # explicitly asked for flat depth. Optimizer moments for
                # unchanged params still transplant by name below.
                retired_r = [k for k in unexpected if k.startswith('r_adapter.')]
                # Local-CNN pruned but the checkpoint was trained with it on: the
                # depthwise/local tensors have nowhere to go in this model.
                # Discard them loudly (same as the retired loop controller)
                # instead of failing — the run explicitly asked for no local
                # convolution (TB_LOCAL_CONV=0 on a conv-pruned checkpoint).
                # Optimizer moments for unchanged params still transplant by
                # name below.
                retired_conv = [k for k in unexpected
                                if '.local_conv.' in k or '.local_conv_extra.' in k]
                # Lexical-semantic pruned but the checkpoint was trained with
                # it on: same loud discard as the retired loop controller and
                # local CNN. Optimizer moments for unchanged params still
                # transplant by name below.
                retired_lex = [k for k in unexpected
                               if k.startswith('lexical_semantic.')]
                other_unexpected = [k for k in unexpected
                                    if k not in legacy_drcg and k not in retired_loop
                                    and k not in retired_r
                                    and k not in retired_conv and k not in retired_lex]
                additive_sidecars = [k for k in missing
                                     if '.local_conv.' in k or '.local_conv_extra.' in k
                                     or k.startswith('lmm.')
                                     or k.startswith('lexical_semantic.')
                                     or k.startswith('loop_controller.')
                                     or k.startswith('r_adapter.')
                                     or k.startswith('v41_engram.')
                                     or k.startswith('v41_ced.')
                                     or k.startswith('v41_shared_kv.')
                                     or k.endswith('.csa2_gate')
                                     or k.endswith('.csa2_alpha')
                                     or '.walk.cycle_' in k
                                     or '.walk.motif_' in k
                                     or '.walk.g_motif' in k
                                     or k.startswith('walk_motif_bank.')
                                     or k.startswith('wdr.')]
                other_missing = [k for k in missing if k not in additive_sidecars]
                if (n_dropped or other_missing or other_unexpected) and os.environ.get("TB_RESUME_ALLOW_ARCH_MISMATCH", "0") != "1":
                    raise RuntimeError(
                        f"Architecture mismatch on resume (dropped={n_dropped} missing={len(other_missing)} unexpected={len(other_unexpected)}). "
                        "Set TB_RESUME_ALLOW_ARCH_MISMATCH=1 to allow shape filtering and optimizer reset."
                    )
                if n_dropped:
                    print(f"resume: dropped {n_dropped} shape-incompatible params (arch upgrade) — "
                          f"reinitialized", flush=True)
                if missing:
                    if additive_sidecars and not other_missing:
                        print(f"resume: added {len(additive_sidecars)} resumable sidecar params "
                              "(local-CNN, LMM, lexical-semantic, V4.1, loop controller, "
                              "narrow-R adapter, cycle guard, and/or walk motif) — "
                              "initialized from scratch", flush=True)
                    else:
                        print(f"resume: added new params (e.g. {list(missing)[:3]}) — "
                              f"initialized from scratch", flush=True)
                if unexpected:
                    if legacy_drcg:
                        print(f"resume: ignoring {len(legacy_drcg)} legacy DRCG keys (removed from clean trainer)", flush=True)
                    if retired_loop:
                        print(f"resume: discarding {len(retired_loop)} loop_controller keys "
                              f"({sum(int(sd[k].numel()) for k in retired_loop):,} params) — "
                              "loop is disabled for this run, recurrence removed", flush=True)
                    if retired_r:
                        print(f"resume: discarding {len(retired_r)} r_adapter keys "
                              f"({sum(int(sd[k].numel()) for k in retired_r):,} params) — "
                              "narrow-R is disabled for this run, recurrence removed", flush=True)
                    if retired_conv:
                        print(f"resume: discarding {len(retired_conv)} local-CNN keys "
                              f"({sum(int(sd[k].numel()) for k in retired_conv):,} params) — "
                              "local convolution pruned for this run", flush=True)
                    if retired_lex:
                        print(f"resume: discarding {len(retired_lex)} lexical-semantic keys "
                              f"({sum(int(sd[k].numel()) for k in retired_lex):,} params) — "
                              "lexical-semantic adapter pruned for this run", flush=True)
                    if other_unexpected:
                        print(f"resume: ignoring unexpected keys ({len(other_unexpected)})", flush=True)
                try:
                    opt.load_state_dict(ck["opt"])
                except Exception as oe:
                    source_names = ck.get("opt_param_names")
                    if source_names:
                        print(f"resume: optimizer shape changed ({oe}) — transplanting unchanged moments by name", flush=True)
                        transplant_optimizer_state_by_name(target, opt, ck.get("opt"), source_names)
                    else:
                        print(f"resume: optimizer state incompatible ({oe}) — resetting optimizer", flush=True)
                if getattr(target, "draft", None) is not None and "draft" in ck:
                    target.draft.load_state_dict(ck["draft"])
                restore_rng_state(ck.get("rng"))
                if "rng_py" in ck:
                    random.setstate(ck["rng_py"])
                if "rng_cuda" in ck and torch.cuda.is_available():
                    torch.cuda.set_rng_state_all([state.cpu() for state in ck["rng_cuda"]])
                bv = ck.get("best_val", float("inf"))
                if bv == float("inf") and os.path.exists(self.best):
                    try:
                        bk = _load_trusted_checkpoint(self.best, map_location="cpu")
                        bv = bk.get("val", float("inf"))
                    except Exception:
                        pass
                print(f"resumed from step {ck['step']} (best {bv:.4f})", flush=True)
                saved_run_state = ck.get("run_state")
                self.gmem_saved = ck.get("gmem")
                if saved_run_state is None and os.path.exists(self.metadata):
                    try:
                        with open(self.metadata, encoding="utf-8") as handle:
                            sidecar = json.load(handle)
                        candidate = sidecar.get("run_state")
                        if isinstance(candidate, dict):
                            saved_run_state = candidate
                            print(f"resume: loaded schedule/config metadata sidecar "
                                  f"({self.metadata})", flush=True)
                    except (OSError, ValueError, TypeError) as sidecar_error:
                        print(f"resume: metadata sidecar unreadable ({sidecar_error})", flush=True)
                return (ck["step"], bv, ck.get("doremi"), ck.get("n_shards"),
                        ck.get("stream_rows"), ck.get("draft_opt"), ck.get("cum_toks", 0),
                        ck.get("schedule_toks", ck.get("cum_toks", 0)), ck.get("data_step_base", 0),
                        saved_run_state, ck.get("stream_state"))
            except Exception as e:
                raise RuntimeError(f"existing checkpoint is incompatible; refusing to start fresh: {e}") from e
        return 0, float("inf"), None, None, None, None, 0, 0, 0, None, None


class HfUploader:
    """Background checkpoint uploader to a HF Hub repo. Training never blocks:
    jobs are queued on a daemon thread and coalesced per remote name (if a newer
    checkpoint lands while one is uploading, only the latest is pushed next).
    Resumable chunked uploads via huggingface_hub. Failures are loud but never
    crash training. No-op when TB_HF_CKPT is unset."""

    def __init__(self, repo_id):
        self.repo_id = repo_id
        self._jobs = {}            # remote_name -> path
        self._latest = {}          # remote_name -> path (newest offered)
        self._cv = threading.Condition()
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def offer(self, local_path, remote_name, cleanup=False):
        """Coalescing enqueue: remember the newest path per remote name.
        cleanup=True removes the local file after a successful upload."""
        with self._cv:
            self._latest[remote_name] = (local_path, cleanup)
            self._cv.notify()

    def stop(self):
        with self._cv:
            self._stop = True
            self._cv.notify_all()

    def _run(self):
        while True:
            with self._cv:
                while not self._jobs and not self._latest and not self._stop:
                    self._cv.wait()
                if self._stop and not self._jobs and not self._latest:
                    return
                # coalesce: pick the newest offered path for each pending name
                for name, (path, cleanup) in self._latest.items():
                    self._jobs[name] = (path, cleanup)
                self._latest.clear()
                name, (path, cleanup) = list(self._jobs.items())[0]
                del self._jobs[name]
            self._upload(path, name)
            if cleanup:
                try:
                    os.remove(path)
                except OSError:
                    pass

    def _upload(self, path, name):
        import os
        if not os.path.exists(path):
            return
        size_mb = os.path.getsize(path) / 2 ** 20
        try:
            from huggingface_hub import HfApi
            from huggingface_hub.utils import disable_progress_bars
            api = HfApi()
            with disable_progress_bars():
                api.upload_file(
                    path_or_fileobj=path,
                    path_in_repo=f"checkpoints/{name}",
                    repo_id=self.repo_id,
                    repo_type="model",
                )
            print(f"HF CKPT uploaded: {name} ({size_mb:.0f}MB) -> {self.repo_id}", flush=True)
        except Exception as e:
            print(f"HF CKPT upload FAILED for {name}: {_redact_hf_credential(e)}", flush=True)


# ------------------------------------------------------------------ driver -

def _te_nvfp4_available():
    """Return (usable, detail) without ever silently selecting FP8 instead."""
    if _TE is None or _TENvFP4Recipe is None:
        return False, "Transformer Engine is not importable"
    probe = getattr(_TE, "is_nvfp4_available", None)
    if probe is None:
        return False, "installed Transformer Engine has no is_nvfp4_available()"
    try:
        result = probe(return_reason=True)
        if isinstance(result, tuple):
            ok, reason = bool(result[0]), str(result[1])
        else:
            ok, reason = bool(result), ""
        return ok, reason or "available"
    except Exception as exc:
        return False, f"NVFP4 capability probe failed: {exc!r}"


def _enable_te_nvfp4(model, cfg):
    """Attach native TE NVFP4 without changing trainable parameter identity.

    This backend intentionally rejects activation-checkpointing for the first
    production rollout.  TE FP8/FP4 state is safe to recompute only after an
    explicit integration test; the user has enough VRAM to avoid it, and this
    removes a confound when comparing against the existing FP8 STE run.
    """
    if getattr(cfg, "ckpt", False):
        raise RuntimeError(
            "TB_QUANT_BACKEND=te_nvfp4 requires TB_CKPT=0 for this validated "
            "first rollout.  Native TE recomputation can be enabled only after "
            "the preflight and a short-resume stability check."
        )
    ok, detail = _te_nvfp4_available()
    if not ok:
        raise RuntimeError(
            "TB_QUANT_BACKEND=te_nvfp4 was requested, but native NVFP4 is unavailable: "
            f"{detail}. This backend does not fall back to FP8. Run "
            "python nvfp4_te_preflight.py after installing a Blackwell-compatible "
            "Transformer Engine build."
        )
    disable_rht = os.environ.get("TB_NVFP4_DISABLE_RHT", "0") == "1"
    disable_2d = os.environ.get("TB_NVFP4_DISABLE_2D", "0") == "1"
    recipe = _TENvFP4Recipe(
        disable_rht=disable_rht,
        disable_2d_quantization=disable_2d,
    )
    qs = [mod for mod in model.modules() if isinstance(mod, QLinear)]
    native_qs, ragged = [], []
    for name, mod in model.named_modules():
        if not isinstance(mod, QLinear):
            continue
        # NVFP4 scales are 16-wide.  The GPT-2 vocabulary head has 50,257
        # rows, so do not gamble a long resume on an undocumented tail kernel;
        # retain its existing BF16 master GEMM.  All 1B body projections are
        # aligned (D=2048, ranks 32/128) and take the real NVFP4 route.
        if mod.master.shape[0] % 16 or mod.master.shape[1] % 16:
            mod.quant = "none"
            ragged.append(name)
            continue
        mod.quant = "te_nvfp4"
        object.__setattr__(mod, "_te_recipe", recipe)
        native_qs.append(mod)

    # One scope per root forward lets TE coordinate quantizer metadata across
    # the native linear operations. The modules still have ordinary BF16
    # masters, so AdamW/Muon and old optimizer checkpoints stay resumable.
    state = {"contexts": []}

    def _pre_forward(_module, _args):
        global _TE_NVFP4_SCOPE_DEPTH
        context = _TE.autocast(enabled=True, recipe=recipe)
        context.__enter__()
        state["contexts"].append(context)
        _TE_NVFP4_SCOPE_DEPTH += 1

    def _post_forward(_module, _args, output):
        global _TE_NVFP4_SCOPE_DEPTH
        context = state["contexts"].pop()
        _TE_NVFP4_SCOPE_DEPTH = max(0, _TE_NVFP4_SCOPE_DEPTH - 1)
        context.__exit__(None, None, None)
        return output

    # Preserve handles to avoid accidental duplicate registration in a caller
    # that rebuilds quantization in-process.
    old = getattr(model, "_te_nvfp4_hooks", None)
    if old:
        for handle in old:
            handle.remove()
    handles = (
        model.register_forward_pre_hook(_pre_forward),
        model.register_forward_hook(_post_forward, always_call=True),
    )
    object.__setattr__(model, "_te_nvfp4_hooks", handles)
    tail_note = (f"; BF16 ragged projection(s)={ragged}" if ragged else "")
    print(f"quantization: ACTUAL=['te_nvfp4'] on {len(native_qs)}/{len(qs)} linears | "
          f"native Transformer Engine NVFP4, RHT={'OFF' if disable_rht else 'ON'}, "
          f"2D weight quant={'OFF' if disable_2d else 'ON'} | {detail}{tail_note}", flush=True)


def build_model(cfg):
    m = WalkGPT(cfg.vocab, cfg.D, cfg.L, cfg.n_head, mult=getattr(cfg, "mult", 2.5),
                hybrid_w=512, win_heads=(max(1, cfg.n_head // 2) if getattr(cfg, "hybrid", True) else 0),
                ckpt=getattr(cfg, "ckpt", True), no_walk=getattr(cfg, "no_walk", False),
                max_t=getattr(cfg, "max_t", 16384),
                alibi=getattr(cfg, "alibi", False), rope=getattr(cfg, "rope", False),
                ni=getattr(cfg, "ni", False), ni_steps=getattr(cfg, "ni_steps", 1),
                ni_trajectory=getattr(cfg, "ni_trajectory", True), ni_slots=getattr(cfg, "ni_slots", 4),
                ni_rank=getattr(cfg, "ni_rank", 32), ni_bridge=getattr(cfg, "ni_bridge", False),
                ni_bridge_weight=getattr(cfg, "ni_bridge_weight", 0.01),
                ni_mode=getattr(cfg, "ni_mode", "final"),
                ni_chunk_size=getattr(cfg, "ni_chunk_size", 128),
                ni_layer=getattr(cfg, "ni_layer", -1),
                ni_layers=getattr(cfg, "ni_layers", ""),
                ni_width_mult=getattr(cfg, "ni_width_mult", 1),
                lmm=getattr(cfg, "lmm", False), lmm_rank=getattr(cfg, "lmm_rank", 128),
                lmm_layers=getattr(cfg, "lmm_layers", ""), lmm_write=getattr(cfg, "lmm_write", "delta"),
                lmm_eta=getattr(cfg, "lmm_eta", 0.25), lmm_decay=getattr(cfg, "lmm_decay", 0.01),
                lmm_gate_bias=getattr(cfg, "lmm_gate_bias", -4.0),
                lmm_lesson_source=getattr(cfg, "lmm_lesson_source", "lookup")).to(DEVICE)
    if getattr(cfg, "ckpt", False):
        print("activation checkpointing: "
              f"stride={m.ckpt_stride} "
              f"full_at_t={m.ckpt_full_at_t or 'off'}", flush=True)
    if getattr(cfg, "alibi", False):
        print("ALiBi: ON — per-head linear distance penalties (2^-8i/H), "
              "pos_emb unused", flush=True)
    if getattr(cfg, "rope", False):
        print(f"RoPE: ON — global base={ROPE_BASE:.0f}, windowed base=W_max (constant), "
              f"null_key={'ON' if getattr(cfg, 'hybrid', True) else 'OFF'}, pos_emb unused", flush=True)
    if m.blocks:
        interface = m.blocks[0]
        print("trajectory gradient interface: "
              f"phaseA rho={interface.walk_input_grad_scale:g} alpha={interface.walk_forward_scale:g}; "
              f"phaseB rho={interface.phase_walk_input_grad_scale:g} "
              f"alpha={interface.phase_walk_forward_scale:g} "
              "(rho changes backward only)", flush=True)
    if getattr(cfg, "local_conv", False):
        print("local causal conv: ON — "
              f"kernels={cfg.local_conv_kernels} rank={cfg.local_conv_rank} "
                f"depth={cfg.local_conv_depth} stage2={cfg.local_conv_stage2_kernels} "
                f"stage3={cfg.local_conv_stage3_kernels} every={cfg.local_conv_every} "
                f"gate_init={cfg.local_conv_gate_init:g} bootstrap={cfg.local_conv_gate_bootstrap:g} "
                f"(pre-RoPE/Walk)", flush=True)
    if getattr(cfg, "lmm", False):
        print("lesson memory: ON — "
              f"rank={cfg.lmm_rank} layers={m.lmm_layers} write={cfg.lmm_write} "
              f"source={m.lmm.lesson_source} "
              f"eta={cfg.lmm_eta:g} decay={cfg.lmm_decay:g}; "
              "requires explicit lesson-memory episodes (FineWeb CE does not write it)", flush=True)
    if getattr(cfg, "lexical_semantic", False):
        print("lexical-semantic adapter: ON — "
              f"rank={cfg.lexical_semantic_rank} gate_init="
              f"{os.environ.get('TB_LEXICAL_SEMANTIC_GATE_INIT', '0')} aux="
              f"{os.environ.get('TB_LEXICAL_SEMANTIC_AUX_WEIGHT', '0.05')} stride="
              f"{os.environ.get('TB_LEXICAL_SEMANTIC_STRIDE', '32')} scale="
              f"{os.environ.get('TB_LEXICAL_SEMANTIC_SCALE', '1')} (final-state, zero-safe)",
              flush=True)
    quant_backend = getattr(cfg, "quant_backend", "ste")
    if quant_backend == "te_nvfp4":
        _enable_te_nvfp4(m, cfg)
    elif not getattr(cfg, "fp4", False):
        for mod in m.modules():
            if isinstance(mod, QLinear):
                mod.quant = "none"
    else:
        # quantize immediately: step-0 eval and first forward use the real path
        qs = [mod for mod in m.modules() if isinstance(mod, QLinear)]
        for mod in qs:
            mod._refresh_q()
        kinds = sorted({mod.quant for mod in qs})
        n_fast = sum(1 for mod in qs if getattr(mod, "_fast_ok", None) is True)
        fp4_ok = _fp4_usable()
        print(f"quantization: ACTUAL={kinds} on {len(qs)} linears "
              f"| fast_linears={n_fast} | fp4_usable={fp4_ok} "
              f"| fp8_act_scale=rowwise-causal "
              f"(fast path verified after first forward; STE routed per-quant "
              f"type — fp4 uses /6 e2m1 scales, fp8 uses /448 e4m3)", flush=True)
    return m


# --------------------------------------------------------- inference ---

def _infer_arch_from_state_dict(sd):
    """Auto-infer WalkGPT arch from a raw state_dict (robust to legacy ckpts).
    Returns dict with vocab, D, L, n_head, max_t, alibi, rope."""
    # strip _orig_mod. prefix for shape probing
    sd2 = {}
    for k, v in sd.items():
        if k.startswith("_orig_mod."):
            k = k[len("_orig_mod."):]
        sd2[k] = v
    # vocab / D from embedding
    vocab, D = None, None
    if "emb.weight" in sd2:
        vocab, D = sd2["emb.weight"].shape
        vocab, D = int(vocab), int(D)
    elif "pos_emb" in sd2:
        D = int(sd2["pos_emb"].shape[1])
    # L = count blocks
    L = 0
    for k in sd2:
        if k.startswith("blocks."):
            try:
                idx = int(k.split(".")[1])
                L = max(L, idx + 1)
            except Exception:
                pass
    # max_t from pos_emb or head
    max_t = None
    if "pos_emb" in sd2:
        max_t = int(sd2["pos_emb"].shape[0])
    # n_head: try to infer from slopes buffer else fallback to env or 32 (training default)
    n_head = None
    for k, v in sd2.items():
        if "slopes" in k and hasattr(v, "shape"):
            n_head = int(v.shape[0])
            break
    if n_head is None:
        try:
            n_head = int(os.environ.get("TB_HEADS", "32"))
        except Exception:
            n_head = 32
    # head vocab fallback
    if vocab is None and "head.weight" in sd2:
        vocab = int(sd2["head.weight"].shape[0])
        if D is None:
            D = int(sd2["head.weight"].shape[1])
    if D is None:
        D = 2048
    if vocab is None:
        vocab = 50257
    if L == 0:
        L = 24
    if max_t is None:
        max_t = 8192
    # mult: infer from mlp up projection shape (h, D) where h = mult*D
    mult = 2.5
    for k, v in sd2.items():
        if k.endswith("mlp.up.master") and hasattr(v, "shape"):
            try:
                h = int(v.shape[0])
                mult = float(h) / float(D) if D else 2.5
                break
            except Exception:
                pass
    # alibi/rope flags: detect via saved pos_emb usage is unreliable, default False
    # but if any block has slopes buffer -> alibi was on; if rope was on, pos_emb may exist but unused
    alibi = any("slopes" in k for k in sd2)
    return {"vocab": vocab, "D": D, "L": L, "n_head": n_head, "max_t": max_t, "alibi": alibi, "rope": False, "mult": mult}


def _load_ckpt_raw(path, device):
    """Robust torch.load: try weights_only=False first (legacy), fallback to True."""
    # Memory-map CPU checkpoint loads first.  An ongoing training checkpoint can
    # contain several GiB of Adam state that inference will never inspect; mmap
    # avoids faulting those tensors into RAM merely to unwrap `model`.
    on_cpu = torch.device(device).type == "cpu"
    for w_only in (False, True):
        for use_mmap in ((True, False) if on_cpu else (False,)):
            try:
                return torch.load(str(path), map_location=device,
                                  weights_only=w_only, mmap=use_mmap)
            except Exception as e:
                last = e
                continue
    raise RuntimeError(f"cannot load checkpoint {path}: {last!r}")


_INFERENCE_FORWARD_SCALE_KEYS = ("TB_WALK_FORWARD_SCALE", "TB_PHASE_WALK_FORWARD_SCALE")


def _restore_inference_forward_scales(run_state):
    """Run inference with the forward scales the checkpoint trained under.

    These live in run_state["runtime"], not the architecture manifest, so the
    loader used to fall back to the 1.0 default: a checkpoint trained with
    TB_PHASE_WALK_FORWARD_SCALE=0 (Phase-B skipped) would decode with the
    sidecar back on. A value the caller exported explicitly still wins.
    """
    runtime = (run_state or {}).get("runtime", {}) if isinstance(run_state, dict) else {}
    if not isinstance(runtime, dict):
        return
    for key in _INFERENCE_FORWARD_SCALE_KEYS:
        saved = runtime.get(key)
        if saved in (None, "") or os.environ.get(key, "").strip():
            continue
        os.environ[key] = str(saved)
        print(f"inference: {key}={saved} (as trained)", flush=True)


def _restore_inference_architecture_env(run_state, state_dict):
    """Restore structural switches before rebuilding a saved checkpoint.

    ``run_state.config`` predates several Walk/Phase-B switches, so rebuilding
    from that config alone can silently change tensor shapes (notably Walk head
    count). Prefer the checkpoint manifest's environment and infer the few
    legacy omissions from state tensors.  Inference is commonly launched from
    a training shell, so a *stale* structural setting (for example a newly
    enabled local CNN) must not take precedence over the checkpoint.  TB_LMM
    is deliberately not structural: it remains an opt-in additive sidecar.
    """
    global GPC_WALK_NORM, GPC_WALK_NORM_MODE
    architecture = (run_state or {}).get("architecture", {}) if isinstance(run_state, dict) else {}
    saved_cfg = (run_state or {}).get("config", {}) if isinstance(run_state, dict) else {}
    saved_env = architecture.get("env", {}) if isinstance(architecture, dict) else {}
    for key, value in saved_env.items():
        if value not in (None, ""):
            # The manifest describes tensors already present in the checkpoint;
            # reusing a value left over from a different resume is unsafe here.
            os.environ[key] = str(value)

    # TB_WALK_NORM was historically saved only as a parsed cfg value, while
    # GradientPathConv reads it into module globals at Python import time.
    # Merely exporting it here is therefore too late: update both the process
    # environment and the already-initialized globals before build_model().
    if saved_cfg.get("walk_norm") is not None:
        GPC_WALK_NORM = float(saved_cfg["walk_norm"])
        os.environ["TB_WALK_NORM"] = str(GPC_WALK_NORM)
    if saved_cfg.get("walk_norm_mode"):
        GPC_WALK_NORM_MODE = str(saved_cfg["walk_norm_mode"]).strip().lower()
        if GPC_WALK_NORM_MODE not in ("renorm", "clip", "cap"):
            raise RuntimeError(f"invalid checkpoint walk_norm_mode: {GPC_WALK_NORM_MODE!r}")
        os.environ["TB_WALK_NORM_MODE"] = GPC_WALK_NORM_MODE
    # Early manifests did not include TB_WALK_HEADS or MIX. Their learned
    # tensor shapes are unambiguous and must win over current process defaults.
    for key, value in state_dict.items():
        if key.endswith(".walk.w1") and getattr(value, "ndim", 0) == 2:
            os.environ["TB_WALK_HEADS"] = str(int(value.shape[0]))
            break
    mix_tensor = next((value for key, value in state_dict.items()
                       if key.endswith(".walk.mix_proj")), None)
    os.environ["TB_WALK_MIX"] = "1" if mix_tensor is not None else "0"
    if mix_tensor is not None and getattr(mix_tensor, "ndim", 0) >= 1:
        os.environ["TB_WALK_MIX_DEPTH"] = str(int(mix_tensor.shape[0]))
    has_walk_gate = any(key.endswith(".walk.gate_mlp.weight") for key in state_dict)
    os.environ["TB_WALK_GATE"] = "1" if has_walk_gate else "0"
    if has_walk_gate:
        os.environ["TB_WALK_GLOB"] = "1"

    # Old manifests also omitted TB_WALK_COUPLE. Its three learned gain
    # families are otherwise unused, so nonzero saved tensors prove that the
    # coupling path was trained and must be enabled when reloading weights.
    if not saved_env.get("TB_WALK_COUPLE"):
        coupling_suffixes = (".walk.wg_eflow", ".walk.wg_inter", ".walk.wg_cdisp")
        coupling_trained = any(
            key.endswith(coupling_suffixes)
            and bool(torch.count_nonzero(value.detach()).item())
            for key, value in state_dict.items()
        )
        if coupling_trained:
            os.environ["TB_WALK_COUPLE"] = "1"

    # Local CNN parameters are architectural, not an optional adapter.  In
    # particular, force it OFF if the checkpoint predates it; otherwise a
    # cloud shell that exported TB_LOCAL_CONV=1 creates thousands of target
    # tensors that have no corresponding source weights.
    local_weight = next((value for key, value in state_dict.items()
                         if key.endswith(".local_conv.in_proj.weight")), None)
    local_keys = (
        "TB_LOCAL_CONV_RANK", "TB_LOCAL_CONV_KERNELS", "TB_LOCAL_CONV_DEPTH",
        "TB_LOCAL_CONV_STAGE2_KERNELS", "TB_LOCAL_CONV_STAGE3_KERNELS",
        "TB_LOCAL_CONV_EVERY",
    )
    if local_weight is None:
        os.environ["TB_LOCAL_CONV"] = "0"
        for key in local_keys:
            os.environ.pop(key, None)
        return

    os.environ["TB_LOCAL_CONV"] = "1"
    os.environ["TB_LOCAL_CONV_RANK"] = str(int(local_weight.shape[0]))
    # Existing CNN checkpoints used one module per block; keep that explicit
    # instead of inheriting an experimental sparse placement from the shell.
    os.environ["TB_LOCAL_CONV_EVERY"] = "1"

    def _kernels(prefix):
        # Each layer owns an identically-shaped CNN.  Keep one kernel sequence,
        # rather than concatenating it once per layer (3,7,15 x 16 -> 48).
        # The latter recreates a different architecture at inference.
        found = {}
        marker = prefix + ".depthwise."
        for key, value in state_dict.items():
            if marker in key and key.endswith(".weight") and getattr(value, "ndim", 0) == 3:
                index = int(key.rsplit(".", 2)[1])
                kernel = int(value.shape[-1])
                earlier = found.setdefault(index, kernel)
                if earlier != kernel:
                    raise RuntimeError(f"inconsistent local CNN kernel at index {index}: {earlier} vs {kernel}")
        return ",".join(str(found[index]) for index in sorted(found))

    primary = _kernels(".local_conv")
    if primary:
        os.environ["TB_LOCAL_CONV_KERNELS"] = primary
    extra_indices = sorted({int(key.split(".local_conv_extra.", 1)[1].split(".", 1)[0])
                            for key in state_dict if ".local_conv_extra." in key})
    os.environ["TB_LOCAL_CONV_DEPTH"] = str(1 + len(extra_indices))
    for index in extra_indices:
        kernels = _kernels(f".local_conv_extra.{index}")
        env_key = "TB_LOCAL_CONV_STAGE2_KERNELS" if index == 0 else "TB_LOCAL_CONV_STAGE3_KERNELS"
        if kernels:
            os.environ[env_key] = kernels


def _adapt_loop_state_keys(sd, target_sd):
    """Adapt loop_controller tensors across pass-count (K) changes on resume.

    Trained-K -> resume-k<K: slice the leading pass dim (pass 0..k-1 share
    semantics across K, so the kept passes stay meaningful).
    Resume-k > trained-K: drop the key so the missing-key path below treats it
    as an additive sidecar (fresh init, existing moments untouched).
    Returns a list of human-readable adaptation notes. Only handles the exact
    recognized case (leading-dim difference, rest identical); anything else
    (e.g. a D change) is left untouched so the generic mismatch path below
    fails closed as before. Mutates sd in place. Raises nothing.
    """
    adapted = []
    for key in list(sd):
        if not key.startswith("loop_controller.") or key not in target_sd:
            continue
        have, want = tuple(sd[key].shape), tuple(target_sd[key].shape)
        if len(have) < 1 or len(have) != len(want) or have[1:] != want[1:]:
            continue
        if have[0] == want[0]:
            continue
        if have[0] > want[0]:
            sd[key] = sd[key][:want[0]].clone()
            adapted.append(f"{key} {have}->{want} (sliced)")
        else:
            del sd[key]
            adapted.append(f"{key} {have}->{want} (fresh init)")
    return adapted


def _adapt_r_state_keys(sd, target_sd):
    """Adapt r_adapter.pass_gain across r_max changes on resume.

    Same one-way semantics as _adapt_loop_state_keys: shrinking slices the
    leading pass dim (pass 0..k-1 keep meaning), growing drops the key so the
    missing-key path treats it as an additive sidecar (fresh init, existing
    moments untouched). Only the exact leading-dim case is handled; anything
    else (e.g. a D change) is left for the generic fail-closed mismatch path.
    Mutates sd in place. Raises nothing.
    """
    adapted = []
    for key in list(sd):
        if key != "r_adapter.pass_gain" or key not in target_sd:
            continue
        have, want = tuple(sd[key].shape), tuple(target_sd[key].shape)
        if len(have) != 1 or len(want) != 1:
            continue
        if have[0] == want[0]:
            continue
        if have[0] > want[0]:
            sd[key] = sd[key][:want[0]].clone()
            adapted.append(f"{key} {have}->{want} (sliced)")
        else:
            del sd[key]
            adapted.append(f"{key} {have}->{want} (fresh init)")
    return adapted


def build_model_for_inference(weights_path, device=None, tokenizer_name=None, dtype=None):
    """Minimal-arg inference loader: weights only -> auto config.
    Returns (model, tokenizer, cfg_dict). Reuses build_model() and WalkGPT.generate()."""
    global DEVICE
    dev = device or DEVICE
    # Full training checkpoints include Adam moments and RNG state.  Loading the
    # envelope directly to `dev` needlessly copies all of that non-inference
    # state to VRAM before we extract `model`, which makes normal 1B inference
    # impossible on otherwise-capable small GPUs.  Keep the envelope on CPU;
    # load_state_dict below transfers only matched model tensors to `dev`.
    raw = _load_ckpt_raw(weights_path, torch.device("cpu"))
    # unwrap checkpoint envelope
    if isinstance(raw, dict) and "model" in raw:
        sd = raw["model"]
        run_state = raw.get("run_state") or {}
        cfg_saved = run_state.get("config") if isinstance(run_state, dict) else None
        # ongoing also has cum_toks, step etc; best has only model+val
        if cfg_saved is None and "config" in raw:
            cfg_saved = raw["config"]
    elif isinstance(raw, dict) and any(k.startswith("emb.") or k.startswith("blocks.") or k.startswith("_orig_mod.") for k in raw):
        sd = raw
        cfg_saved = None
        run_state = {}
    else:
        # maybe HuggingFace-style or plain
        sd = raw.get("model_state_dict", raw) if isinstance(raw, dict) else raw
        cfg_saved = raw.get("config") if isinstance(raw, dict) else None
        run_state = {}
    # normalize sd keys
    sd = { (k[len("_orig_mod."): ] if k.startswith("_orig_mod.") else k): v for k, v in sd.items() }
    _restore_inference_architecture_env(run_state, sd)
    _restore_inference_forward_scales(run_state)
    inferred = _infer_arch_from_state_dict(sd)
    # build cfg: prefer saved config, else inferred + env defaults
    cfg = Cfg()
    # defaults (mirrors main() but minimal)
    cfg.tokenizer = str((cfg_saved or {}).get("tokenizer", tokenizer_name or os.environ.get("TB_TOKENIZER", "gpt2")))
    cfg.vocab = int((cfg_saved or {}).get("vocab", inferred["vocab"]))
    cfg.D = int((cfg_saved or {}).get("D", inferred["D"]))
    # L key may be "L" or "n_layer"
    _L = (cfg_saved or {}).get("L", (cfg_saved or {}).get("n_layer", inferred["L"]))
    cfg.L = int(_L)
    cfg.n_head = int((cfg_saved or {}).get("n_head", (cfg_saved or {}).get("n_heads", inferred["n_head"])))
    cfg.mult = float((cfg_saved or {}).get("mult", inferred.get("mult", 2.5)))
    cfg.max_t = int((cfg_saved or {}).get("max_t", inferred["max_t"]))
    cfg.hybrid = bool((cfg_saved or {}).get("hybrid", True))
    cfg.alibi = bool((cfg_saved or {}).get("alibi", inferred.get("alibi", False)))
    cfg.rope = bool((cfg_saved or {}).get("rope", False))
    cfg.no_walk = bool((cfg_saved or {}).get("no_walk", False))
    cfg.ni = bool((cfg_saved or {}).get("ni", False))
    cfg.ni_mode = str((cfg_saved or {}).get("ni_mode", "final"))
    cfg.ni_steps = int((cfg_saved or {}).get("ni_steps", 1))
    cfg.ni_trajectory = bool((cfg_saved or {}).get("ni_trajectory", True))
    cfg.ni_slots = int((cfg_saved or {}).get("ni_slots", 4))
    cfg.ni_rank = int((cfg_saved or {}).get("ni_rank", 32))
    cfg.ni_width_mult = int((cfg_saved or {}).get("ni_width_mult", 1))
    cfg.ni_bridge = bool((cfg_saved or {}).get("ni_bridge", False))
    cfg.ni_bridge_weight = float((cfg_saved or {}).get("ni_bridge_weight", 0.01))
    cfg.ni_chunk_size = int((cfg_saved or {}).get("ni_chunk_size", 128))
    cfg.ni_layer = int((cfg_saved or {}).get("ni_layer", -1))
    cfg.ni_layers = str((cfg_saved or {}).get("ni_layers", ""))
    # An inference/meta-training caller may attach the additive LMM sidecar to
    # an older checkpoint without rewriting its saved architecture config.
    # This remains opt-in: absent TB_LMM, preserve the checkpoint's setting.
    _lmm_env = os.environ.get("TB_LMM")
    cfg.lmm = ((_lmm_env == "1") if _lmm_env is not None
               else bool((cfg_saved or {}).get("lmm", False)))
    cfg.lmm_rank = int(os.environ.get("TB_LMM_RANK", (cfg_saved or {}).get("lmm_rank", 128)))
    cfg.lmm_layers = os.environ.get("TB_LMM_LAYERS", str((cfg_saved or {}).get("lmm_layers", ""))).strip()
    cfg.lmm_write = os.environ.get("TB_LMM_WRITE", str((cfg_saved or {}).get("lmm_write", "delta"))).strip().lower()
    cfg.lmm_lesson_source = os.environ.get(
        "TB_LMM_LESSON_SOURCE", str((cfg_saved or {}).get("lmm_lesson_source", "lookup"))).strip().lower()
    cfg.lmm_eta = float(os.environ.get("TB_LMM_ETA", (cfg_saved or {}).get("lmm_eta", 0.25)))
    cfg.lmm_decay = float(os.environ.get("TB_LMM_DECAY", (cfg_saved or {}).get("lmm_decay", 0.01)))
    cfg.lmm_gate_bias = float(os.environ.get("TB_LMM_GATE_BIAS", (cfg_saved or {}).get("lmm_gate_bias", -4.0)))
    cfg.lexical_semantic = os.environ.get("TB_LEXICAL_SEMANTIC", "0") == "1"
    cfg.lexical_semantic_rank = int(os.environ.get(
        "TB_LEXICAL_SEMANTIC_RANK", (cfg_saved or {}).get("lexical_semantic_rank", 256)))
    # `_restore_inference_architecture_env` reconstructs these settings from
    # the actual tensors.  Copy them into cfg before build_model; otherwise a
    # CNN checkpoint is silently evaluated as a CNN-off model.
    cfg.local_conv = os.environ.get(
        "TB_LOCAL_CONV", "1" if bool((cfg_saved or {}).get("local_conv", False)) else "0") == "1"
    cfg.local_conv_rank = int(os.environ.get(
        "TB_LOCAL_CONV_RANK", (cfg_saved or {}).get("local_conv_rank", 128)))
    cfg.local_conv_kernels = os.environ.get(
        "TB_LOCAL_CONV_KERNELS", str((cfg_saved or {}).get("local_conv_kernels", "3,7,15"))).strip()
    cfg.local_conv_every = int(os.environ.get(
        "TB_LOCAL_CONV_EVERY", (cfg_saved or {}).get("local_conv_every", 1)))
    cfg.local_conv_depth = int(os.environ.get(
        "TB_LOCAL_CONV_DEPTH", (cfg_saved or {}).get("local_conv_depth", 1)))
    cfg.local_conv_stage2_kernels = os.environ.get(
        "TB_LOCAL_CONV_STAGE2_KERNELS",
        str((cfg_saved or {}).get("local_conv_stage2_kernels", "7,15,31"))).strip()
    cfg.local_conv_stage3_kernels = os.environ.get(
        "TB_LOCAL_CONV_STAGE3_KERNELS",
        str((cfg_saved or {}).get("local_conv_stage3_kernels", "15,31,63"))).strip()
    cfg.local_conv_gate_init = float((cfg_saved or {}).get("local_conv_gate_init", 0.0))
    cfg.local_conv_gate_bootstrap = float((cfg_saved or {}).get("local_conv_gate_bootstrap", 0.0))
    cfg.fp4 = bool((cfg_saved or {}).get("fp4", False)) or os.environ.get("TB_FP4", "0") == "1"
    cfg.quant_backend = os.environ.get("TB_QUANT_BACKEND", "ste").strip().lower()
    if cfg.quant_backend not in ("ste", "te_nvfp4"):
        raise ValueError("TB_QUANT_BACKEND must be ste or te_nvfp4")
    cfg.ckpt = False  # no grad checkpointing at inference
    cfg.amp = True
    # env overrides for precision if user forced
    if os.environ.get("TB_FORCE_QUANT", "").strip().lower() in ("fp4", "fp8", "none"):
        q = os.environ["TB_FORCE_QUANT"].strip().lower()
        cfg.fp4 = q in ("fp4", "fp8")
    # build model via existing builder (reuses QLinear, WalkGPT, etc.)
    # temporarily pin DEVICE for build_model
    old_dev = DEVICE
    DEVICE = dev
    try:
        model = build_model(cfg)
    finally:
        DEVICE = old_dev
    # load weights robustly (shape-filter like Checkpointer.resume)
    target = model._orig_mod if hasattr(model, "_orig_mod") else model
    tgt_sd = target.state_dict()
    filtered = {}
    dropped = []
    for k, v in sd.items():
        if k in tgt_sd and tuple(v.shape) != tuple(tgt_sd[k].shape):
            dropped.append(k)
        elif k in tgt_sd:
            filtered[k] = v
    missing, unexpected = target.load_state_dict(filtered, strict=False)
    # `filtered` contains target keys only, so load_state_dict cannot report
    # source-only tensors. Account for them explicitly instead of silently
    # dropping an architectural branch.
    source_only = [key for key in sd if key not in tgt_sd]
    unexpected = list(unexpected) + source_only

    def _inert_legacy_csa2_alpha(key):
        """Accept only the no-op bridge buffer absent from pre-CSA2 saves.

        ``Attn`` persists ``csa2_alpha`` even when CSA2 is disabled so a later
        structural phase can ramp and resume it.  Checkpoints written before
        that buffer was introduced legitimately lack one scalar per block.
        This is not a general architecture-mismatch escape: shared K/V must be
        absent, the key must have the exact expected form, and the newly built
        target buffer must still be scalar zero.
        """
        if getattr(target, "v41_shared_kv", None) is not None:
            return False
        parts = key.split(".")
        if not (len(parts) == 4 and parts[0] == "blocks"
                and parts[1].isdigit() and parts[2:] == ["attn", "csa2_alpha"]):
            return False
        value = tgt_sd.get(key)
        return (torch.is_tensor(value) and value.numel() == 1
                and float(value.detach().float().cpu()) == 0.0)

    additive_missing = bool(missing) and not dropped and not unexpected and all(
        key.startswith("lmm.")
        or key.startswith("lexical_semantic.")
        or _inert_legacy_csa2_alpha(key)
        for key in missing)
    if (dropped or missing or unexpected) and not additive_missing and \
            os.environ.get("TB_INFERENCE_ALLOW_ARCH_MISMATCH", "0") != "1":
        dropped_sample = []
        for key in dropped[:3]:
            source = sd.get(key)
            target_tensor = tgt_sd.get(key)
            dropped_sample.append((key, tuple(source.shape), tuple(target_tensor.shape)))
        raise RuntimeError(
            f"Inference arch mismatch (dropped={len(dropped)} missing={len(missing)} unexpected={len(unexpected)}). "
            f"dropped_sample={dropped_sample} missing_sample={missing[:3]} "
            "(do not use TB_INFERENCE_ALLOW_ARCH_MISMATCH for an LMM experiment)."
        )
    if dropped:
        print(f"inference: dropped {len(dropped)} shape-mismatched keys (arch drift): {dropped[:3]}", flush=True)
    if missing:
        detail = "new additive sidecar initialized" if additive_missing else "new params, zero-init kept"
        print(f"inference: missing keys ({detail}): {missing[:3]}", flush=True)
    if unexpected:
        print(f"inference: unexpected keys ignored: {unexpected[:3]}", flush=True)
    # refresh quantized caches exactly like resume()
    if getattr(cfg, "fp4", False):
        for mod in model.modules():
            if isinstance(mod, QLinear) and mod.quant != "none":
                mod._refresh_q()
    model = model.to(dev)
    model.eval()
    # Inference-only narrow-R depth override. The manifest restores the
    # trained eval depth (r = TB_R_MEAN); TB_INFER_R_MEAN runs fewer passes,
    # e.g. 2 when the learned pass_gain beyond index 1 is ~0 (step 29750:
    # [.051, .023, .002, -.001, ...]). Changes logits slightly; opt-in only.
    _infer_r = os.environ.get("TB_INFER_R_MEAN", "").strip()
    if _infer_r and getattr(model, "r_adapter", None) is not None:
        _r = int(_infer_r)
        _cap = int(getattr(model, "r_depth_cap", 0) or model.r_max)
        if not 0 <= _r <= _cap:
            raise ValueError(f"TB_INFER_R_MEAN must be in [0, {_cap}], got {_r}")
        print(f"inference: narrow-R eval depth {model.r_mean} -> {_r} (TB_INFER_R_MEAN)", flush=True)
        model.r_mean = _r
    # tokenizer: reuse _load_tokenizer (already robust to long sequences)
    tok_name = cfg.tokenizer
    if tokenizer_name:
        tok_name = tokenizer_name
    try:
        tok = _load_tokenizer(tok_name)
    except Exception as e:
        print(f"tokenizer {tok_name!r} failed ({e!r}), falling back to gpt2", flush=True)
        tok = _load_tokenizer("gpt2")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # LOTUS+ graph memory travels beside the weights (checkpoint key "gmem");
    # graph_memory_session(model, tok) opens it for inference.
    object.__setattr__(model, "_gmem_saved", raw.get("gmem") if isinstance(raw, dict) else None)
    print(f"inference ready: weights={weights_path} | D={cfg.D} L={cfg.L} n_head={cfg.n_head} vocab={cfg.vocab} max_t={cfg.max_t} | device={dev}"
          + (" | LOTUS+ graph memory: present" if getattr(model, "_gmem_saved", None) else ""), flush=True)
    return model, tok, cfg


def graph_memory_session(model, tok, saved=None):
    """GraphMemorySession for a model from build_model_for_inference (or a given gmem state)."""
    saved = saved if saved is not None else getattr(model, "_gmem_saved", None)
    if not saved:
        raise RuntimeError("this checkpoint has no LOTUS+ graph memory (train with TB_GMEM=1)")
    target = model._orig_mod if hasattr(model, "_orig_mod") else model
    gm, _ = gmem_build(target, tok, int(saved.get("R", 4)), saved=saved)
    gm.eval()
    return GraphMemorySession(target, tok, gm)


@torch.inference_mode()
def _generate_text(model, tok, prompt, device, max_new=128, temperature=0.8, top_p=0.9, rep_penalty=1.1, seed=None):
    if seed is not None:
        torch.manual_seed(int(seed))
        random.seed(int(seed))
    # clamp temperature: 0 => greedy (avoid div-by-zero in WalkGPT.generate which does /temperature)
    temperature = float(temperature)
    if temperature < 1e-6:
        temperature = 0.0  # signal greedy; WalkGPT handles 0 as div-by-zero -> we clamp to small
        _eff_temp = 1e-6
    else:
        _eff_temp = temperature
    # tokenizer truncation: keep last max_t tokens if prompt is long
    max_t = getattr(model, "max_t", 8192) if not hasattr(model, "_orig_mod") else getattr(model._orig_mod, "max_t", 8192)
    # WalkGPT.generate expects token ids (B, T)
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False)
    ids = enc["input_ids"]
    if ids.shape[1] > max_t - max_new - 1:
        ids = ids[:, -(max_t - max_new - 1):]
    ids = ids.to(device)
    out = model.generate(ids, max_new=max_new, temperature=float(_eff_temp), top_p=float(top_p),
                         rep_penalty=float(rep_penalty), eos_id=getattr(tok, "eos_token_id", None))
    # decode only the new continuation + prompt
    text = tok.decode(out[0].tolist(), skip_special_tokens=True)
    return text, out[0].tolist()


def run_inference_loop(weights_path, prompt=None, prompt_file=None, output_file=None, device=None, max_new=128, temperature=0.8, top_p=0.9, rep_penalty=1.1, seed=0, interactive=None, tokenizer_name=None):
    """Robust looping inference: single prompt, batch file, or interactive REPL.
    Minimal required arg is weights_path (auto config). Loop is the default."""
    dev = device or DEVICE
    model, tok, cfg = build_model_for_inference(weights_path, device=dev, tokenizer_name=tokenizer_name)
    # decide mode
    # batch file mode
    if prompt_file and os.path.exists(prompt_file):
        with open(prompt_file, "r", encoding="utf-8", errors="replace") as f:
            prompts = [line.rstrip("\n") for line in f if line.strip() != ""]
        outs = []
        for i, p in enumerate(prompts):
            txt, _ = _generate_text(model, tok, p, dev, max_new=max_new, temperature=temperature, top_p=top_p, rep_penalty=rep_penalty, seed=seed + i if seed is not None else None)
            outs.append({"prompt": p, "output": txt})
            print(f"\n--- [{i+1}/{len(prompts)}] ---\nPrompt: {p}\n{txt}\n", flush=True)
        if output_file:
            with open(output_file, "w", encoding="utf-8") as out:
                json.dump(outs, out, ensure_ascii=False, indent=2)
            print(f"wrote {len(outs)} results -> {output_file}", flush=True)
        return outs
    # single prompt (non-loop) if --prompt supplied and interactive explicitly disabled
    if prompt is not None and interactive is False:
        txt, _ = _generate_text(model, tok, prompt, dev, max_new=max_new, temperature=temperature, top_p=top_p, rep_penalty=rep_penalty, seed=seed)
        print(txt, flush=True)
        if output_file:
            with open(output_file, "w", encoding="utf-8") as out:
                json.dump({"prompt": prompt, "output": txt}, out, ensure_ascii=False, indent=2)
        return txt
    # interactive REPL loop (default): reuse for multiple tests until quit
    if prompt is not None:
        # first shot from CLI arg, then loop
        txt, _ = _generate_text(model, tok, prompt, dev, max_new=max_new, temperature=temperature, top_p=top_p, rep_penalty=rep_penalty, seed=seed)
        print(f"\n{txt}\n", flush=True)
    print("Interactive inference loop. Commands: /quit /exit /q | /temp <f> | /top_p <f> | /max_new <n> | /seed <n> | /help", flush=True)
    print(f"Current: max_new={max_new} temp={temperature} top_p={top_p} rep_penalty={rep_penalty} seed={seed}", flush=True)
    n = 0
    while True:
        try:
            raw = input("\n>>> prompt (or /help): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye", flush=True)
            break
        if not raw:
            continue
        low = raw.lower()
        if low in ("/quit", "/exit", "/q", "exit", "quit"):
            print("bye", flush=True)
            break
        if low in ("/help", "/h", "help"):
            print("Commands:\n  /quit - exit\n  /temp 0.8 - set temperature\n  /top_p 0.9 - set top_p\n  /max_new 128 - set max tokens\n  /rep 1.1 - set repetition penalty\n  /seed 42 - set seed\n  /greedy - temp=0, top_p=1\n  /sample - temp=0.8 top_p=0.9", flush=True)
            continue
        if low.startswith("/temp"):
            try:
                temperature = float(raw.split()[1]); print(f"temperature={temperature}", flush=True)
            except Exception:
                print("usage: /temp 0.8", flush=True)
            continue
        if low.startswith("/top_p"):
            try:
                top_p = float(raw.split()[1]); print(f"top_p={top_p}", flush=True)
            except Exception:
                print("usage: /top_p 0.9", flush=True)
            continue
        if low.startswith("/max_new"):
            try:
                max_new = int(raw.split()[1]); print(f"max_new={max_new}", flush=True)
            except Exception:
                print("usage: /max_new 128", flush=True)
            continue
        if low.startswith("/rep"):
            try:
                rep_penalty = float(raw.split()[1]); print(f"rep_penalty={rep_penalty}", flush=True)
            except Exception:
                print("usage: /rep 1.1", flush=True)
            continue
        if low.startswith("/seed"):
            try:
                seed = int(raw.split()[1]); print(f"seed={seed}", flush=True)
            except Exception:
                print("usage: /seed 42", flush=True)
            continue
        if low == "/greedy":
            temperature = 0.0; top_p = 1.0; print("greedy: temp=0 top_p=1", flush=True); continue
        if low == "/sample":
            temperature = 0.8; top_p = 0.9; print("sample: temp=0.8 top_p=0.9", flush=True); continue
        cur_seed = (seed + n) if seed is not None else None
        txt, _ = _generate_text(model, tok, raw, dev, max_new=max_new, temperature=temperature, top_p=top_p, rep_penalty=rep_penalty, seed=cur_seed)
        n += 1
        print(f"\n--- output {n} ---\n{txt}\n", flush=True)


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _mix_val_file(base_path, source_index, source_name):
    safe = __import__("re").sub(r"[^A-Za-z0-9_.-]+", "_", str(source_name)).strip("._")
    return f"{base_path}.{source_index:02d}-{safe or 'source'}.bin"


def _mixed_validation_artifacts(base_path, require_files=True):
    """Return the manifest and shard paths after validating safe basenames.

    Validation artifacts are uploaded beside the checkpoint.  The manifest is
    untrusted on restore, so never allow an entry to escape its local directory.
    Mask shards travel with their banks: a manifest claiming a mask the disk
    lacks is not uploadable (it would poison the bundle for masked resumes).
    """
    base_path = os.path.abspath(base_path)
    meta_path = base_path + ".meta.json"
    if not os.path.isfile(meta_path):
        if require_files:
            raise RuntimeError(f"mixed validation manifest is missing: {meta_path}")
        return []
    with open(meta_path, encoding="utf-8") as handle:
        meta = json.load(handle)
    entries = meta.get("sources")
    if not isinstance(entries, list) or not entries:
        raise RuntimeError("mixed validation manifest has no source entries")
    parent = os.path.dirname(meta_path)
    artifacts = [meta_path]

    def _take(entry, field, kind):
        filename = entry.get(field) if isinstance(entry, dict) else None
        if field == "mask_file" and not filename:
            return
        if (not isinstance(filename, str) or not filename
                or os.path.basename(filename) != filename
                or filename in {".", ".."}):
            raise RuntimeError(
                f"unsafe mixed validation {kind} name: {filename!r}")
        shard_path = os.path.join(parent, filename)
        if require_files and not os.path.isfile(shard_path):
            raise RuntimeError(
                f"mixed validation {kind} is missing: {shard_path}")
        artifacts.append(shard_path)

    for entry in entries:
        _take(entry, "file", "shard")
        _take(entry, "mask_file", "mask shard")
    return artifacts


def _restore_mixed_validation_from_hf(cfg, downloader=None):
    """Atomically restore the fixed validation suite for an ordinary resume.

    The manifest is installed last.  Therefore an interrupted restore can
    never be mistaken for a complete validation contract on the next launch.
    """
    base_path = os.path.abspath(cfg.mix_val_path)
    meta_path = base_path + ".meta.json"
    if os.path.isfile(meta_path):
        return False
    if not getattr(cfg, "hf_ckpt", None):
        raise RuntimeError(
            "ordinary resume is missing its mixed validation manifest and "
            "TB_HF_CKPT is unset; refusing to capture a different benchmark")
    if downloader is None:
        from huggingface_hub import hf_hub_download
        downloader = hf_hub_download
    parent = os.path.dirname(meta_path) or os.getcwd()
    os.makedirs(parent, exist_ok=True)
    remote_prefix = "checkpoints/validation"
    meta_name = os.path.basename(meta_path)
    staged = []
    try:
        remote_meta = downloader(
            repo_id=cfg.hf_ckpt,
            filename=f"{remote_prefix}/{meta_name}",
            repo_type="model",
        )
        with open(remote_meta, encoding="utf-8") as handle:
            meta = json.load(handle)
        entries = meta.get("sources")
        if not isinstance(entries, list) or not entries:
            raise RuntimeError("remote mixed validation manifest has no source entries")
        for entry in entries:
            filename = entry.get("file") if isinstance(entry, dict) else None
            if (not isinstance(filename, str) or not filename
                    or os.path.basename(filename) != filename
                    or filename in {".", ".."}):
                raise RuntimeError(f"unsafe remote mixed validation shard name: {filename!r}")
            remote_shard = downloader(
                repo_id=cfg.hf_ckpt,
                filename=f"{remote_prefix}/{filename}",
                repo_type="model",
            )
            staged_path = os.path.join(parent, filename) + ".restore.tmp"
            shutil.copyfile(remote_shard, staged_path)
            expected_sha = entry.get("sha256")
            if not isinstance(expected_sha, str) or _file_sha256(staged_path) != expected_sha:
                raise RuntimeError(f"restored mixed validation checksum mismatch: {filename}")
            staged.append((staged_path, os.path.join(parent, filename)))
            # Mask shards travel with their banks; without them a masked
            # resume cannot do answer-only eval. Old bundles predate masks:
            # absent entry (not a failed download) simply restores maskless.
            mask_name = entry.get("mask_file") if isinstance(entry, dict) else None
            if mask_name:
                if (not isinstance(mask_name, str)
                        or os.path.basename(mask_name) != mask_name
                        or mask_name in {".", ".."}):
                    raise RuntimeError(
                        f"unsafe remote mixed validation mask name: {mask_name!r}")
                try:
                    remote_mask = downloader(
                        repo_id=cfg.hf_ckpt,
                        filename=f"{remote_prefix}/{mask_name}",
                        repo_type="model",
                    )
                except Exception as mask_exc:
                    # Legacy bundles carry manifests that claim mask shards
                    # the hub never received (the uploader predates mask
                    # blobs). A 404 here is expected history, not corruption:
                    # skip the blob and let the load-path fitness check
                    # recapture + heal. Anything else still fails closed.
                    if type(mask_exc).__name__ != "RemoteEntryNotFoundError":
                        raise
                    print(f"MIX val mask {mask_name} absent on hub "
                          f"(legacy bundle); continuing without it — the "
                          f"load check will recapture masked banks.",
                          flush=True)
                    continue
                staged_mask = os.path.join(parent, mask_name) + ".restore.tmp"
                shutil.copyfile(remote_mask, staged_mask)
                expected_mask_sha = entry.get("mask_sha256")
                if (not isinstance(expected_mask_sha, str)
                        or _file_sha256(staged_mask) != expected_mask_sha):
                    raise RuntimeError(
                        f"restored mixed validation mask checksum mismatch: {mask_name}")
                staged.append((staged_mask, os.path.join(parent, mask_name)))
        staged_meta = meta_path + ".restore.tmp"
        shutil.copyfile(remote_meta, staged_meta)
        staged.append((staged_meta, meta_path))
        for staged_path, final_path in staged[:-1]:
            os.replace(staged_path, final_path)
        os.replace(staged[-1][0], staged[-1][1])
    except Exception as exc:
        for staged_path, _ in staged:
            try:
                os.remove(staged_path)
            except OSError:
                pass
        raise RuntimeError(
            "ordinary resume could not restore its fixed mixed validation suite "
            f"from {cfg.hf_ckpt}: {exc}") from exc
    print(f"MIX val restored from HF checkpoint bundle: {cfg.hf_ckpt}", flush=True)
    return True


def _offer_mixed_validation(hf_up, cfg):
    """Queue the complete fixed validation suite under checkpoints/validation/."""
    if hf_up is None:
        return 0
    artifacts = _mixed_validation_artifacts(cfg.mix_val_path)
    # Publish shards first and the manifest last. A fresh-box restore therefore
    # cannot discover a new manifest before every referenced object is queued.
    for path in artifacts[1:] + artifacts[:1]:
        hf_up.offer(path, f"validation/{os.path.basename(path)}")
    print(f"MIX val backup queued: {len(artifacts)} artifacts", flush=True)
    return len(artifacts)


def _token_bank_diagnostics(tokens, raw_rows, eos_id=None, ngram_sample=131_072):
    """Cheap, deterministic intrinsic diagnostics for a held-out token bank.

    These do not claim to measure reasoning difficulty.  They expose corpus
    predictability/composition confounds so cross-source CE is not interpreted
    as a pure quality or reasoning ranking.
    """
    flat = tokens.detach().to(device="cpu", dtype=torch.long).reshape(-1)
    count = int(flat.numel())
    if count == 0:
        raise RuntimeError("cannot diagnose an empty validation token bank")
    frequencies = torch.bincount(flat)
    probabilities = frequencies[frequencies > 0].double() / count
    entropy = float(-(probabilities * probabilities.log()).sum().item())
    top_fraction = float(frequencies.max().item() / count)
    adjacent_repeat = (float((flat[1:] == flat[:-1]).double().mean().item())
                       if count > 1 else 0.0)
    sample = flat[:max(4, min(count, int(ngram_sample)))].tolist()
    seen, repeated = set(), 0
    for index in range(max(0, len(sample) - 3)):
        ngram = tuple(sample[index:index + 4])
        repeated += int(ngram in seen)
        seen.add(ngram)
    ngram_total = max(0, len(sample) - 3)
    eos_fraction = None
    if eos_id is not None:
        eos_fraction = float((flat == int(eos_id)).double().mean().item())
    return {
        "tokens": count,
        "raw_rows": int(raw_rows),
        "tokens_per_captured_row": float(count / max(1, int(raw_rows))),
        "unique_tokens": int((frequencies > 0).sum().item()),
        "unigram_entropy_nats": entropy,
        "top_token_fraction": top_fraction,
        "eos_fraction": eos_fraction,
        "adjacent_repeat_fraction": adjacent_repeat,
        "repeated_4gram_fraction": float(repeated / max(1, ngram_total)),
        "ngram_sample_tokens": len(sample),
    }


def _load_mixed_validation_entries(dm, entries, tokens_per_source, meta_path):
    """Install one restored/captured validation manifest: checksums, held-out
    ranges with resume-skip alignment, token banks, and SFT prompt masks."""
    for source_index, (spec, source, entry) in enumerate(
            zip(dm.stream.specs, dm.stream.sources, entries)):
        if (entry.get("name") != spec["name"]
                or float(entry.get("weight", -1.0)) != float(spec["weight"])):
            raise RuntimeError(f"mixed validation source order/weight mismatch at index {source_index}")
        val_path = os.path.join(os.path.dirname(meta_path), entry["file"])
        if not os.path.isfile(val_path):
            raise RuntimeError(f"mixed validation shard is missing: {val_path}")
        if _file_sha256(val_path) != entry.get("sha256"):
            raise RuntimeError(f"mixed validation checksum mismatch: {val_path}")
        raw_start, raw_end = int(entry["raw_start"]), int(entry["raw_end"])
        source.add_heldout_range(raw_start, raw_end)
        if source.raw_cursor < raw_end:
            skipped = raw_end - source.raw_cursor
            source.skip_more(skipped)
            print(f"MIX val {spec['name']}: skipped {skipped:,} held-out rows on resume", flush=True)
        tokens = _load_shard(val_path)
        if len(tokens) != int(entry["tokens"]):
            raise RuntimeError(f"mixed validation token-count mismatch: {val_path}")
        if len(tokens) != tokens_per_source:
            raise RuntimeError(
                f"mixed validation shard {spec['name']} has {len(tokens):,} tokens; "
                f"expected exactly {tokens_per_source:,}")
        dm.mix_val[spec["name"]] = tokens
        mask_file = entry.get("mask_file")
        if mask_file:
            mask_path = os.path.join(os.path.dirname(meta_path), mask_file)
            if not os.path.isfile(mask_path):
                raise RuntimeError(f"mixed validation mask shard is missing: {mask_path}")
            if _file_sha256(mask_path) != entry.get("mask_sha256"):
                raise RuntimeError(f"mixed validation mask checksum mismatch: {mask_path}")
            masks = torch.from_numpy(
                __import__("numpy").fromfile(mask_path, dtype="uint8")).to(torch.uint8)
            if len(masks) != len(tokens):
                raise RuntimeError(f"mixed validation mask length mismatch: {mask_path}")
            dm.mix_val_masks[spec["name"]] = masks
        else:
            # Only message_field (SFT) sources ever carry a mask shard:
            # capture writes mask_file solely when the source rendered
            # prompt spans, and eval treats a missing mask as "score every
            # token" (correct for role-free sources like wiki). Demanding
            # a shard here for all nine banks broke every masked resume on
            # the first non-SFT bank. Raise only when a maskable source
            # lacks its shard (genuinely stale unmasked bank).
            if _sft_mask_enabled() and bool(getattr(source, "message_field", "")):
                raise RuntimeError(
                    f"mixed validation bank {spec['name']} has no SFT mask shard; "
                    "masked runs need freshly captured banks (new TB_MIX_VAL_PATH)")
            dm.mix_val_masks[spec["name"]] = None
        dm.mix_val_weights[spec["name"]] = float(spec["weight"])
        dm.mix_val_diagnostics[spec["name"]] = dict(entry.get("diagnostics") or {})
        print(f"MIX val loaded {spec['name']}: {len(tokens):,} unseen tokens, "
              f"weight={100 * float(spec['weight']):.1f}%", flush=True)


def _mixed_validation_expected(dm, cfg):
    """The manifest fields that bind a validation suite to this exact mix."""
    source_contracts = []
    for spec, source in zip(dm.stream.specs, dm.stream.sources):
        source_contracts.append({
            "name": spec["name"],
            "dataset": spec["dataset"],
            "weight": float(spec["weight"]),
            "identity": source.validation_identity(),
        })
    return {
        "version": 2,
        "kind": "weighted-mix-heldout-sequential-v3",
        "stream_mix_signature": dm.stream.signature,
        "tokenizer": cfg.tokenizer,
        "vocab": int(cfg.vocab),
        "packing": "eos-v1",
        "sft_mask": "1" if _sft_mask_enabled() else "",
        "tokens_per_source_requested": int(cfg.mix_val_tokens_per_source),
        "sources_contract": source_contracts,
    }


def _extend_mixed_validation(dm, cfg):
    """Build the extended mix's suite from the previous one.

    Sources that were already in the mix keep their exact banks and held-out
    raw ranges (copied from TB_MIX_VAL_PREV_PATH, restored from the HF repo
    when absent), so their per-source CE stays comparable across the change.
    Only sources new to the mix capture a bank, from their fresh cursor. The
    combined manifest is then loaded by prepare_mixed_validation as usual.
    """
    import shutil
    import types
    prev_base = os.environ.get("TB_MIX_VAL_PREV_PATH", "").strip()
    if not prev_base:
        raise RuntimeError("mix extension needs TB_MIX_VAL_PREV_PATH (the previous suite)")
    prev_base = os.path.abspath(prev_base)
    prev_meta_path = prev_base + ".meta.json"
    if not os.path.isfile(prev_meta_path):
        _restore_mixed_validation_from_hf(types.SimpleNamespace(
            mix_val_path=prev_base, hf_ckpt=getattr(cfg, "hf_ckpt", None)))
    with open(prev_meta_path, encoding="utf-8") as handle:
        prev = json.load(handle)
    tokens_per_source = int(cfg.mix_val_tokens_per_source)
    if int(prev.get("tokens_per_source_requested", -1)) != tokens_per_source:
        raise RuntimeError("previous validation suite used a different bank size")
    if prev.get("sft_mask", "") != ("1" if _sft_mask_enabled() else ""):
        raise RuntimeError("previous validation suite used a different SFT-mask objective")
    prev_entries = {str(e.get("name")): e for e in (prev.get("sources") or [])}
    base_path = os.path.abspath(cfg.mix_val_path)
    meta_path = base_path + ".meta.json"
    os.makedirs(os.path.dirname(base_path) or os.getcwd(), exist_ok=True)
    prev_dir = os.path.dirname(prev_meta_path)
    eos_id = getattr(getattr(cfg, "tok", None), "eos_token_id", None)
    rerendered = set(getattr(cfg, "mix_rerendered", []) or [])
    entries = []
    for source_index, (spec, source) in enumerate(zip(dm.stream.specs, dm.stream.sources)):
        val_path = _mix_val_file(base_path, source_index, spec["name"])
        # Re-rendered SFT sources need banks in the new format: capture fresh
        # (held-out rows from their current cursor), like an added source.
        old = None if spec["name"] in rerendered else prev_entries.get(spec["name"])
        if old is not None:
            src_bin = os.path.join(prev_dir, os.path.basename(old["file"]))
            if _file_sha256(src_bin) != old.get("sha256"):
                raise RuntimeError(f"previous validation bank checksum mismatch: {src_bin}")
            shutil.copyfile(src_bin, val_path)
            entry = {**old, "weight": float(spec["weight"]), "file": os.path.basename(val_path)}
            if old.get("mask_file"):
                src_mask = os.path.join(prev_dir, os.path.basename(old["mask_file"]))
                if _file_sha256(src_mask) != old.get("mask_sha256"):
                    raise RuntimeError(f"previous validation mask checksum mismatch: {src_mask}")
                shutil.copyfile(src_mask, val_path + ".mask.bin")
                entry["mask_file"] = os.path.basename(val_path + ".mask.bin")
            entries.append(entry)
            print(f"MIX val extend: kept bank {spec['name']} (raw rows "
                  f"[{int(old['raw_start']):,}, {int(old['raw_end']):,}))", flush=True)
            continue
        raw_start = int(source.raw_cursor)
        tokens, masks, raw_rows = source.capture_val(tokens_per_source)
        if len(tokens) < tokens_per_source:
            raise RuntimeError(
                f"mixed validation source {spec['name']} exhausted at {len(tokens):,} tokens")
        if (masks is None and _sft_mask_enabled()
                and str(spec.get("message_field", "") or "").strip()):
            raise RuntimeError(f"SFT mask capture for {spec['name']} produced zero prompt masks")
        tokens = tokens[:tokens_per_source].contiguous()
        masks = masks[:tokens_per_source].contiguous() if masks is not None else None
        raw_end = raw_start + int(raw_rows)
        __import__("numpy").asarray(tokens.numpy(), dtype="uint16").tofile(val_path)
        mask_file = mask_sha = None
        if masks is not None:
            __import__("numpy").asarray(masks.numpy(), dtype="uint8").tofile(val_path + ".mask.bin")
            mask_file = os.path.basename(val_path + ".mask.bin")
            mask_sha = _file_sha256(val_path + ".mask.bin")
        entries.append({
            "name": spec["name"], "weight": float(spec["weight"]),
            "file": os.path.basename(val_path), "tokens": int(len(tokens)),
            "raw_rows": int(raw_rows), "raw_start": raw_start, "raw_end": raw_end,
            "sha256": _file_sha256(val_path), "mask_file": mask_file, "mask_sha256": mask_sha,
            "diagnostics": _token_bank_diagnostics(tokens, raw_rows, eos_id=eos_id),
        })
        print(f"MIX val extend: captured new bank {spec['name']}: {len(tokens):,} tokens from "
              f"{raw_rows:,} raw rows [{raw_start:,}, {raw_end:,})", flush=True)
    meta = {**_mixed_validation_expected(dm, cfg), "sources": entries,
            "extended_from": os.path.basename(prev_base)}
    meta_tmp = meta_path + ".tmp"
    with open(meta_tmp, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, sort_keys=True)
    os.replace(meta_tmp, meta_path)
    print(f"MIX val suite extended: {len(entries)} sources; manifest={meta_path}", flush=True)


def prepare_mixed_validation(dm, cfg):
    """Create/load an equal-size, leakage-safe bank for every mix source.

    Files are relocatable (the manifest stores basenames), checksummed, and
    bound to the complete source/mix/tokenizer contract. Every raw interval is
    installed in its own child stream before any training pump starts.
    """
    if not isinstance(dm.stream, MixedStreamSource):
        raise TypeError("mixed validation requires MixedStreamSource")
    tokens_per_source = int(cfg.mix_val_tokens_per_source)
    if tokens_per_source <= 0:
        return
    base_path = os.path.abspath(cfg.mix_val_path)
    parent = os.path.dirname(base_path) or os.getcwd()
    os.makedirs(parent, exist_ok=True)
    meta_path = base_path + ".meta.json"
    expected = _mixed_validation_expected(dm, cfg)

    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as handle:
            meta = json.load(handle)
        mismatched = [key for key, value in expected.items() if meta.get(key) != value]
        if mismatched:
            raise RuntimeError(
                f"mixed validation metadata mismatch for {mismatched}; use a new TB_MIX_VAL_PATH")
        entries = meta.get("sources") or []
        if len(entries) != len(dm.stream.sources):
            raise RuntimeError("mixed validation metadata has the wrong source count")
        if _sft_mask_enabled():
            # A restored (usually HF) suite can claim sft_mask=1 yet carry no
            # mask shards for its SFT banks (captured before boundary-robust
            # masking, or by a foreign pipeline): it can never support
            # answer-only eval, so it is unfit for this objective — not a
            # benchmark worth preserving. Recapture fresh (cursors untouched
            # so far: nothing below has run yet) and re-upload heals the
            # bundle in place. Genuinely different benchmarks still fail
            # above at the metadata comparison.
            deficient = []
            for spec, source, entry in zip(dm.stream.specs, dm.stream.sources, entries):
                if not str(getattr(source, "message_field", "") or "").strip():
                    continue
                mask_name = entry.get("mask_file")
                if not mask_name:
                    deficient.append(spec["name"])
                elif not os.path.isfile(os.path.join(os.path.dirname(meta_path), mask_name)):
                    # Manifest claims a shard the disk lacks (usually a
                    # restore that predates mask-blob handling): same
                    # unrecoverable state as a missing entry, same remedy.
                    deficient.append(spec["name"] + " (file absent)")
            if deficient:
                print(f"MIX val suite at {base_path} claims masked eval but SFT "
                      f"banks lack mask shards: {deficient}. Discarding the local "
                      f"suite and capturing fresh (re-upload will heal the bundle).",
                      flush=True)
                for stale in sorted(glob.glob(base_path + ".*.bin")) + [meta_path]:
                    try:
                        os.remove(stale)
                    except OSError:
                        pass
            else:
                return _load_mixed_validation_entries(
                    dm, entries, tokens_per_source, meta_path)
        else:
            # Unmasked runs load the suite as-is (this branch was missing:
            # an unmasked resume fell through to the orphan-shard error).
            return _load_mixed_validation_entries(
                dm, entries, tokens_per_source, meta_path)

    partial = sorted(glob.glob(base_path + ".*.bin"))
    if partial:
        raise RuntimeError(
            "mixed validation has shard files but no manifest (interrupted capture): "
            f"{partial[:4]}. Use a new TB_MIX_VAL_PATH or remove only those orphan shards.")

    captured = []
    for source_index, (spec, source) in enumerate(zip(dm.stream.specs, dm.stream.sources)):
        raw_start = int(source.raw_cursor)
        tokens, masks, raw_rows = source.capture_val(tokens_per_source)
        if len(tokens) < tokens_per_source:
            raise RuntimeError(
                f"mixed validation source {spec['name']} exhausted at {len(tokens):,} tokens; "
                f"cannot build equal {tokens_per_source:,}-token banks")
        # A masked bank with zero prompt masks is unmeasurable (answer-only
        # eval would silently score everything): refuse to bless it instead of
        # uploading a manifest that hard-fails every later masked run.
        if (masks is None and _sft_mask_enabled()
                and str(spec.get("message_field", "") or "").strip()):
            raise RuntimeError(
                f"SFT mask capture for {spec['name']} produced zero prompt masks "
                f"({int(getattr(source, 'sft_unmasked_fallback', 0)):,} fallbacks); "
                "refusing to write an unmeasurable masked bank — inspect the "
                "message rendering and tokenizer boundary handling")
        # capture_val consumes whole documents so it may cross the requested
        # token boundary.  Keep every consumed raw row excluded from training,
        # but persist exactly the same token count for every validation source.
        tokens = tokens[:tokens_per_source].contiguous()
        masks = masks[:tokens_per_source].contiguous() if masks is not None else None
        raw_end = raw_start + int(raw_rows)
        source.add_heldout_range(raw_start, raw_end)
        captured.append((source_index, spec, tokens, masks, raw_start, raw_end, int(raw_rows)))
        _mask_note = ""
        if masks is not None:
            _mask_note = (f" masked-prompt={int(masks.sum().item()):,} "
                          f"({100.0 * float(masks.sum().item()) / max(1, len(tokens)):.1f}%)")
        elif _sft_mask_enabled() and str(spec.get("message_field", "") or "").strip():
            _mask_note = " NO MASKS (all fallbacks)"
        print(f"MIX val captured {spec['name']}: {len(tokens):,} tokens from "
              f"{raw_rows:,} raw rows [{raw_start:,}, {raw_end:,}){_mask_note}", flush=True)

    entries = []
    eos_id = getattr(getattr(cfg, "tok", None), "eos_token_id", None)
    for source_index, spec, tokens, masks, raw_start, raw_end, raw_rows in captured:
        val_path = _mix_val_file(base_path, source_index, spec["name"])
        tmp_path = val_path + ".tmp"
        __import__("numpy").asarray(tokens.numpy(), dtype="uint16").tofile(tmp_path)
        os.replace(tmp_path, val_path)
        mask_file = mask_sha = None
        if masks is not None:
            mask_path = val_path + ".mask.bin"
            mask_tmp = mask_path + ".tmp"
            __import__("numpy").asarray(masks.numpy(), dtype="uint8").tofile(mask_tmp)
            os.replace(mask_tmp, mask_path)
            mask_file = os.path.basename(mask_path)
            mask_sha = _file_sha256(mask_path)
        diagnostics = _token_bank_diagnostics(tokens, raw_rows, eos_id=eos_id)
        entries.append({
            "name": spec["name"],
            "weight": float(spec["weight"]),
            "file": os.path.basename(val_path),
            "tokens": int(len(tokens)),
            "raw_rows": raw_rows,
            "raw_start": raw_start,
            "raw_end": raw_end,
            "sha256": _file_sha256(val_path),
            "mask_file": mask_file,
            "mask_sha256": mask_sha,
            "diagnostics": diagnostics,
        })
        dm.mix_val[spec["name"]] = tokens
        dm.mix_val_masks[spec["name"]] = masks
        dm.mix_val_weights[spec["name"]] = float(spec["weight"])
        dm.mix_val_diagnostics[spec["name"]] = diagnostics
        print(f"MIX val intrinsic {spec['name']}: unigram_h="
              f"{diagnostics['unigram_entropy_nats']:.4f} "
              f"repeat4={diagnostics['repeated_4gram_fraction']:.6f} "
              f"adj_repeat={diagnostics['adjacent_repeat_fraction']:.6f} "
              f"tok_per_row={diagnostics['tokens_per_captured_row']:.1f}", flush=True)
    meta = {**expected, "sources": entries}
    meta_tmp = meta_path + ".tmp"
    with open(meta_tmp, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, sort_keys=True)
    os.replace(meta_tmp, meta_path)
    print(f"MIX val suite ready: {len(entries)} sources x {tokens_per_source:,} tokens; "
          f"manifest={meta_path}", flush=True)


def _source_difficulty_suffix(metrics):
    normalized = metrics.get("normalized_ce")
    normalized_text = f"{normalized:.6f}" if normalized is not None else "nan"
    repeated = metrics.get("repeated_4gram_fraction")
    repeated_text = f"{repeated:.6f}" if repeated is not None else "nan"
    return (f" ce_std={metrics.get('ce_std', float('nan')):.6f}"
            f" ce_p10={metrics.get('ce_p10', float('nan')):.6f}"
            f" ce_p50={metrics.get('ce_p50', float('nan')):.6f}"
            f" ce_p90={metrics.get('ce_p90', float('nan')):.6f}"
            f" unigram_h={metrics.get('unigram_entropy_nats', float('nan')):.6f}"
            f" normalized_ce={normalized_text} repeat4={repeated_text}")


def _gate_triplet(values):
    """Mean/RMS/max-abs of a scalar list; NaN triple when the branch is absent."""
    vals = [float(v) for v in values]
    if not vals:
        return (float("nan"), float("nan"), float("nan"))
    mean = sum(vals) / len(vals)
    rms = math.sqrt(sum(v * v for v in vals) / len(vals))
    return (mean, rms, max(abs(v) for v in vals))


def _format_gate_triplet(values):
    if not all(math.isfinite(v) for v in values):
        return "-/-/-"
    return f"{values[0]:+.3f}/{values[1]:.3f}/{values[2]:.3f}"


def _branch_gate_summary(target):
    """Scalar gate statistics across blocks for trend logging.

    The existing trend line shows conv gates, lex gate-mean, and aux losses,
    but not walk/phase gates or the lex residual raw value. The lex raw gate
    distinguishes a learned gate shutdown (raw -> 0, tanh -> 0) from a route
    shutdown (raw stays open while gate_mean -> 0) without needing a forward
    pass. The R pass_gain triplet is raw (unbounded mixing coefficients, not
    squashed gates): all-zero means the branch is provably inert. Missing
    branches (pruned or never built) report NaN.
    """
    walk_vals, phase_vals = [], []
    geom_vals, place_vals, selmem_vals, hurst_vals = [], [], [], []
    affine_vals, motif_vals = [], []
    for blk in getattr(target, "blocks", ()):
        walk = getattr(blk, "walk", None)
        gamma = getattr(walk, "gamma", None) if walk is not None else None
        if torch.is_tensor(gamma):
            walk_vals.append(math.tanh(float(gamma.detach().float().reshape(()).cpu())))
        if walk is not None:
            if getattr(walk, "geom_on", False):
                for _pn in ("g_tan", "g_curv", "g_tors", "g_dwell", "g_arc"):
                    _p = getattr(walk, _pn, None)
                    if torch.is_tensor(_p):
                        geom_vals.extend(float(v) for v in
                                         _p.detach().float().reshape(-1).cpu())
            if getattr(walk, "place_on", False):
                for _pn in ("g_pdist", "g_prad"):
                    _p = getattr(walk, _pn, None)
                    if torch.is_tensor(_p):
                        place_vals.extend(float(v) for v in
                                          _p.detach().float().reshape(-1).cpu())
            if getattr(walk, "selmem_on", False):
                _g = getattr(walk, "mem_sel_g", None)
                if torch.is_tensor(_g):
                    selmem_vals.append(
                        math.tanh(float(_g.detach().float().reshape(()).cpu())))
            if getattr(walk, "hurst_on", False):
                _p = getattr(walk, "g_hurst", None)
                if torch.is_tensor(_p):
                    hurst_vals.extend(float(v) for v in
                                      _p.detach().float().reshape(-1).cpu())
            if getattr(walk, "affine_on", False):
                for _pn in ("g_aff_scale", "g_aff_novel", "g_aff_dir"):
                    _p = getattr(walk, _pn, None)
                    if torch.is_tensor(_p):
                        affine_vals.extend(float(v) for v in
                                           _p.detach().float().reshape(-1).cpu())
            if getattr(walk, "motif_on", False):
                _p = getattr(walk, "g_motif", None)
                if torch.is_tensor(_p):
                    motif_vals.extend(float(v) for v in
                                      _p.detach().float().reshape(-1).cpu())
        sidecar = getattr(blk, "phase_walk", None)
        gate = getattr(sidecar, "phase_residual_gate", None) if sidecar is not None else None
        if torch.is_tensor(gate):
            phase_vals.append(math.tanh(float(gate.detach().float().reshape(()).cpu())))
    lex = getattr(target, "lexical_semantic", None)
    lex_param = getattr(lex, "residual_gate", None) if lex is not None else None
    lex_raw = (float(lex_param.detach().float().reshape(()).cpu())
               if torch.is_tensor(lex_param) else float("nan"))
    adapter = getattr(target, "r_adapter", None)
    adapter_gain = getattr(adapter, "pass_gain", None) if adapter is not None else None
    r_vals = ([float(v) for v in adapter_gain.detach().float().reshape(-1).cpu()]
              if torch.is_tensor(adapter_gain) else [])
    return {
        "walk_gamma": _gate_triplet(walk_vals),
        "phase_gate": _gate_triplet(phase_vals),
        "lex_raw": lex_raw,
        "lex_gate": math.tanh(lex_raw) if math.isfinite(lex_raw) else float("nan"),
        "r_gain": _gate_triplet(r_vals),
        "geom_gain": _gate_triplet(geom_vals),
        "place_gain": _gate_triplet(place_vals),
        "selmem_gate": _gate_triplet(selmem_vals),
        "hurst_gain": _gate_triplet(hurst_vals),
        "affine_gain": _gate_triplet(affine_vals),
        "motif_gain": _gate_triplet(motif_vals),
    }


def init_ema_weights(model):
    """fp32 shadow copy of trainable masters for Polyak-Ruppert averaging.

    fp32, not bf16: with decay=0.999 the per-step (1-decay)*param update is
    ~1e-3 relative, below bf16's 7-bit mantissa resolution — a bf16 shadow
    would freeze. Plain tensors (no device, no grad); advance with
    update_ema_weights after each accepted optimizer step.
    """
    target = model._orig_mod if hasattr(model, "_orig_mod") else model
    emas = {}
    with torch.no_grad():
        for name, p in target.named_parameters():
            if p.requires_grad:
                emas[name] = p.detach().float().clone()
    return emas


def update_ema_weights(emas, model, decay):
    """One Polyak step: ema <- decay*ema + (1-decay)*param, in fp32."""
    if not emas:
        return
    target = model._orig_mod if hasattr(model, "_orig_mod") else model
    with torch.no_grad():
        for name, p in target.named_parameters():
            e = emas.get(name)
            if e is not None:
                e.mul_(float(decay)).add_(p.detach().float(),
                                          alpha=1.0 - float(decay))


@contextmanager
def ema_eval_weights(model, emas):
    """Swap EMA masters in for evaluation, then restore live weights.

    Restores exactly (bitwise) even on exception; refreshes QLinear caches
    around both swaps so quantized paths never read stale scales. Refuses
    the te_nvfp4 path, whose native recipe cache does not re-derive from
    master on every call.
    """
    if not emas:
        yield
        return
    target = model._orig_mod if hasattr(model, "_orig_mod") else model
    for m in target.modules():
        if isinstance(m, QLinear) and getattr(m, "quant", "") == "te_nvfp4":
            raise RuntimeError(
                "TB_EMA_EVAL=1 is incompatible with te_nvfp4 quant; "
                "use the ste backend for EMA evaluation")
    saved = {}
    try:
        with torch.no_grad():
            for name, p in target.named_parameters():
                e = emas.get(name)
                if e is not None:
                    saved[name] = p.detach().clone()
                    p.copy_(e.to(p.dtype))
            for m in target.modules():
                if isinstance(m, QLinear):
                    m._refresh_q()
        yield
    finally:
        with torch.no_grad():
            for name, p in target.named_parameters():
                if name in saved:
                    p.copy_(saved[name])
            for m in target.modules():
                if isinstance(m, QLinear):
                    m._refresh_q()


def _evaluate_inner(model, dm, cfg, n_batches=10, include_train_loss=True):
    losses, top1s, top5s = [], [], []
    dm.last_source_val_metrics = collections.OrderedDict()
    dm.last_val_summary = {}
    with torch.no_grad():
        source_groups = dm.val_batches_by_source(n_batches, cfg.eval_t)
        if source_groups:
            weighted_ce = weighted_top1 = weighted_top5 = weight_sum = 0.0
            for name, weight, batches in source_groups:
                source_losses, source_top1, source_top5 = [], [], []
                for xb, yb in batches:
                    logits, loss = model(xb, yb)
                    source_losses.append(float(getattr(model, '_last_lm_ce', loss.detach())))
                    source_top1.append((logits.argmax(-1) == yb).float().mean().item())
                    source_top5.append(
                        (logits.topk(5, -1).indices == yb.unsqueeze(-1)).any(-1).float().mean().item())
                metrics = {
                    "ce": sum(source_losses) / len(source_losses),
                    "top1": sum(source_top1) / len(source_top1),
                    "top5": sum(source_top5) / len(source_top5),
                    "weight": float(weight),
                    "batches": len(batches),
                    "tokens": len(batches) * int(cfg.eval_t),
                }
                loss_tensor = torch.tensor(source_losses, dtype=torch.float64)
                metrics["ce_std"] = float(loss_tensor.std(unbiased=False).item())
                metrics["ce_p10"] = float(torch.quantile(loss_tensor, 0.10).item())
                metrics["ce_p50"] = float(torch.quantile(loss_tensor, 0.50).item())
                metrics["ce_p90"] = float(torch.quantile(loss_tensor, 0.90).item())
                intrinsic = getattr(dm, "mix_val_diagnostics", {}).get(name, {})
                entropy = float(intrinsic.get("unigram_entropy_nats", 0.0) or 0.0)
                metrics["unigram_entropy_nats"] = entropy
                metrics["normalized_ce"] = metrics["ce"] / entropy if entropy > 0.0 else None
                metrics["repeated_4gram_fraction"] = intrinsic.get("repeated_4gram_fraction")
                dm.last_source_val_metrics[name] = metrics
                weighted_ce += weight * metrics["ce"]
                weighted_top1 += weight * metrics["top1"]
                weighted_top5 += weight * metrics["top5"]
                weight_sum += weight
            if weight_sum <= 0.0:
                raise RuntimeError("mixed validation weights sum to zero")
            losses = [weighted_ce / weight_sum]
            top1s = [weighted_top1 / weight_sum]
            top5s = [weighted_top5 / weight_sum]
            source_ces = [item["ce"] for item in dm.last_source_val_metrics.values()]
            dm.last_val_summary = {
                "weighted_ce": losses[0],
                "macro_ce": sum(source_ces) / len(source_ces),
                "worst_ce": max(source_ces),
                "worst_source": max(dm.last_source_val_metrics,
                                    key=lambda key: dm.last_source_val_metrics[key]["ce"]),
            }
        else:
            for xb, yb in dm.val_batches(n_batches, cfg.eval_t):
                logits, loss = model(xb, yb)
                losses.append(float(getattr(model, '_last_lm_ce', loss.detach())))
                top1s.append((logits.argmax(-1) == yb).float().mean().item())
                top5s.append((logits.topk(5, -1).indices == yb.unsqueeze(-1)).any(-1).float().mean().item())
        if not losses:
            # no val available (no CoT split, no fw_val): fall back to train windows
            for k in range(n_batches):
                xb, yb = dm.get_batch(888_000 + k, 1, cfg.eval_t, 0.0)
                logits, loss = model(xb, yb)
                losses.append(float(getattr(model, '_last_lm_ce', loss.detach())))
                top1s.append((logits.argmax(-1) == yb).float().mean().item())
                top5s.append((logits.topk(5, -1).indices == yb.unsqueeze(-1)).any(-1).float().mean().item())
        # train loss on fresh train windows (memorization check: the train/val gap)
        tr_losses = []
        if include_train_loss:
            for k in range(n_batches):
                xb, yb = dm.get_batch(777_000 + k, 1, cfg.eval_t, 0.0)
                _, loss = model(xb, yb)
                tr_losses.append(float(getattr(model, '_last_lm_ce', loss.detach())))
        # same-domain CoT gap: train-CoT windows vs val-CoT windows
        cot_tr_losses = []
        if include_train_loss and dm.cot is not None:
            for xb, yb in dm.cot_train_batches(n_batches, cfg.eval_t):
                _, loss = model(xb, yb)
                cot_tr_losses.append(float(getattr(model, '_last_lm_ce', loss.detach())))
        # Legacy single-stream held-out value, or the FineWeb-Edu component of
        # the multi-source suite for backward-compatible log consumers.
        fw_losses = []
        if dm.last_source_val_metrics:
            web_name = next((name for name in dm.last_source_val_metrics
                             if name == "web_edu_dedup"), None)
            if web_name is not None:
                fw_losses.append(dm.last_source_val_metrics[web_name]["ce"])
        else:
            for xb, yb in dm.val_batches_fw(n_batches, cfg.eval_t):
                _, loss = model(xb, yb)
                fw_losses.append(float(getattr(model, '_last_lm_ce', loss.detach())))
    vl = sum(losses) / len(losses)  # primary val loss — used as cot_gap fallback
    ctl = sum(cot_tr_losses) / len(cot_tr_losses) if cot_tr_losses else vl
    return (vl, sum(top1s) / len(top1s), sum(top5s) / len(top5s),
            sum(tr_losses) / len(tr_losses) if tr_losses else float('nan'), ctl,
            sum(fw_losses) / len(fw_losses) if fw_losses else float("nan"))


def evaluate(model, dm, cfg, n_batches=10, include_train_loss=True):
    """Eval-mode evaluation that never leaks training state.

    An exception mid-eval previously left the model in eval mode, silently
    disabling training-only paths (R refinement, walk loop checkpointing) in
    all subsequent train steps. Likewise the train-loss sampling advances the
    global random and torch-CPU streams, perturbing data order by eval
    cadence. Both are saved here and restored in a finally: eval observes
    without mutating. (Live-window consumption itself, ~0.02% of tokens, is
    inherent to sampling and unchanged.)
    """
    model.eval()
    _py_rng = random.getstate()
    _cpu_rng = torch.random.get_rng_state()
    try:
        return _evaluate_inner(model, dm, cfg, n_batches,
                               include_train_loss=include_train_loss)
    finally:
        model.train()
        random.setstate(_py_rng)
        torch.random.set_rng_state(_cpu_rng)


@contextmanager
def _eval_mode(model):
    """Eval mode that restores the prior training mode even on exception.

    Several eval helpers run model.eval() then model.train() at the end; an
    exception between them (CUDA error, malformed batch, failed ablation)
    used to leave the model in eval mode, silently disabling training-only
    paths (R refinement, walk loop checkpointing) in all later train steps.
    """
    was_training = model.training
    model.eval()
    try:
        yield
    finally:
        model.train(was_training)


@torch.no_grad()
def evaluate_code(model, dm, cfg, n_batches=10):
    batches = dm.val_batches_code(n_batches, cfg.eval_t)
    if not batches:
        return float('nan')
    losses = []
    with _eval_mode(model):
        for xb, yb in batches:
            _, loss = model(xb, yb)
            losses.append(float(getattr(model, '_last_lm_ce', loss.detach())))
    return sum(losses) / len(losses)


def evaluate_without_walk(model, dm, cfg, n_batches=10):
    """Primary held-out loss with Walk residuals zeroed on identical val windows."""
    losses = []
    with _eval_mode(model):
        with torch.no_grad(), temporarily_disable_walk(model):
            source_groups = dm.val_batches_by_source(n_batches, cfg.eval_t)
            if source_groups:
                weighted = weight_sum = 0.0
                for _, weight, batches in source_groups:
                    source_losses = []
                    for xb, yb in batches:
                        _, loss = model(xb, yb)
                        source_losses.append(float(getattr(model, '_last_lm_ce', loss.detach())))
                    weighted += weight * (sum(source_losses) / len(source_losses))
                    weight_sum += weight
                losses.append(weighted / weight_sum)
            else:
                batches = dm.val_batches(n_batches, cfg.eval_t)
                for xb, yb in batches:
                    _, loss = model(xb, yb)
                    losses.append(float(getattr(model, '_last_lm_ce', loss.detach())))
            if not losses:
                batches = [dm.get_batch(888_000 + k, 1, cfg.eval_t, 0.0)
                           for k in range(n_batches)]
                for xb, yb in batches:
                    _, loss = model(xb, yb)
                    losses.append(float(getattr(model, '_last_lm_ce', loss.detach())))
    return sum(losses) / len(losses)



def _evaluate_with_context(model, dm, cfg, context, n_batches):
    losses = []
    with _eval_mode(model):
        with torch.no_grad(), context:
            source_groups = dm.val_batches_by_source(n_batches, cfg.eval_t)
            if source_groups:
                weighted = weight_sum = 0.0
                for _, weight, batches in source_groups:
                    source_losses = []
                    for xb, yb in batches:
                        _, loss = model(xb, yb)
                        source_losses.append(float(getattr(model, '_last_lm_ce', loss.detach())))
                    weighted += weight * (sum(source_losses) / len(source_losses))
                    weight_sum += weight
                losses.append(weighted / weight_sum)
            else:
                batches = dm.val_batches(n_batches, cfg.eval_t)
                for xb, yb in batches:
                    _, loss = model(xb, yb)
                    losses.append(float(getattr(model, '_last_lm_ce', loss.detach())))
            if not losses:
                batches = [dm.get_batch(889_000 + k, 1, cfg.eval_t, 0.0) for k in range(n_batches)]
                for xb, yb in batches:
                    _, loss = model(xb, yb)
                    # compare pure LM CE, not auxiliary totals
                    losses.append(float(getattr(model, '_last_lm_ce', loss.detach())))
    return sum(losses) / max(1, len(losses))


def evaluate_phase_ablations(model, dm, cfg, n_batches=2):
    """Cheap same-checkpoint causal-usage probes; not a substitute for retrained controls."""
    target = model._orig_mod if hasattr(model, '_orig_mod') else model
    phase = [getattr(b, 'phase_walk', None) for b in target.blocks]
    phase = [a for a in phase if a is not None]
    if not phase:
        return {}
    result = {
        'matched_full': _evaluate_with_context(model, dm, cfg, nullcontext(), n_batches),
        'phase_off': _evaluate_with_context(model, dm, cfg, temporarily_disable_phase_walk(model), n_batches),
    }
    if any(a.archive_present and a.archive_scale != 0.0 for a in phase):
        result['archive_off'] = _evaluate_with_context(
            model, dm, cfg, temporarily_disable_phase_archive(model), n_batches)
    regressors = [getattr(block, 'traj_reg', None) for block in target.blocks]
    if any(a is not None and a.enabled for a in regressors):
        result['traj_reg_off'] = _evaluate_with_context(
            model, dm, cfg, temporarily_disable_trajectory_regression(model), n_batches)
    return result


def evaluate_extension_ablations(model, dm, cfg, n_batches=2):
    """Same-window causal ablations for the checkpoint-safe additions."""
    target = _model_target(model)
    has_local = any(getattr(block, 'local_conv', None) is not None for block in target.blocks)
    has_lexical = getattr(target, 'lexical_semantic', None) is not None
    has_engram = getattr(target, 'v41_engram', None) is not None
    has_ced_csa2 = (getattr(target, 'v41_ced', None) is not None
                    or getattr(target, 'v41_shared_kv', None) is not None)
    if not any((has_local, has_lexical, has_engram, has_ced_csa2)):
        return {}
    result = {
        'matched_full': _evaluate_with_context(model, dm, cfg, nullcontext(), n_batches),
        'all_extensions_off': _evaluate_with_context(
            model, dm, cfg, temporarily_disable_all_extensions(model), n_batches),
    }
    if has_ced_csa2:
        result['ced_csa2_off'] = _evaluate_with_context(
            model, dm, cfg, temporarily_disable_v41_ced_csa2(model), n_batches)
    if has_engram:
        result['engram_off'] = _evaluate_with_context(
            model, dm, cfg, temporarily_disable_v41_engram(model), n_batches)
    if has_local:
        result['local_off'] = _evaluate_with_context(
            model, dm, cfg, temporarily_disable_local_features(model), n_batches)
    if has_lexical:
        result['lexical_off'] = _evaluate_with_context(
            model, dm, cfg, temporarily_disable_lexical_semantic(model), n_batches)
    return result


def _reseed_local_conv_gates(model, opt):
    """Bootstrap only effectively-closed local gates on an explicit resume.

    Older checkpoints contain the depth-3 modules with zero gates.  A small
    nonzero seed lets their internal filters receive CE gradients immediately;
    learned gates are left untouched.  Their Adam moments are cleared because
    those moments were accumulated while the branch was effectively inactive.
    """
    if os.environ.get("TB_LOCAL_CONV_RESEED_ZERO_GATES", "0") != "1":
        return 0
    init = _local_conv_gate_init()
    if init == 0.0:
        raise ValueError("TB_LOCAL_CONV_RESEED_ZERO_GATES=1 requires nonzero TB_LOCAL_CONV_GATE_INIT")
    if abs(init) > 0.25:
        print(f"WARNING: TB_LOCAL_CONV_GATE_INIT={init:g} is the effective tanh gate, "
              "not a pre-tanh logit. This is a large residual-architecture shock; "
              "the local CNN experiments only tested effective magnitudes up to about 0.14.",
              flush=True)
    threshold = abs(float(os.environ.get("TB_LOCAL_CONV_RESEED_THRESHOLD", "0.001")))
    if not math.isfinite(threshold):
        raise ValueError("TB_LOCAL_CONV_RESEED_THRESHOLD must be finite")
    target = model._orig_mod if hasattr(model, "_orig_mod") else model
    reseeded = 0
    with torch.no_grad():
        for block in getattr(target, "blocks", ()):
            stages = ([getattr(block, "local_conv", None)]
                      + list(getattr(block, "local_conv_extra", ())))
            for stage in stages:
                gate = getattr(stage, "residual_gate", None)
                if gate is None or abs(float(torch.tanh(gate).detach().float().cpu())) > threshold:
                    continue
                gate.fill_(math.atanh(init))
                if opt is not None:
                    opt.state.pop(gate, None)
                reseeded += 1
    if reseeded:
        print(f"local-conv gate bootstrap: reseeded {reseeded} near-zero gates "
              f"to effective={init:g} (threshold={threshold:g}); cleared their optimizer moments",
              flush=True)
    else:
        print(f"local-conv gate bootstrap: no gates within threshold={threshold:g}; "
              "preserving learned values", flush=True)
    return reseeded


def _relax_legacy_local_conv_gates(model, opt):
    """One-time repair for the historical forced |gate|=0.905 checkpoints.

    This is opt-in and only clips gates whose effective magnitude exceeds the
    requested bound. It preserves sign/filter weights, clears stale Adam state,
    and lets the new FP32 controls relearn their useful magnitude.
    """
    raw = os.environ.get("TB_LOCAL_CONV_RELAX_LEGACY_GATES", "").strip()
    if not raw:
        return 0
    limit = abs(float(raw))
    if not math.isfinite(limit) or not (0.0 < limit <= 0.25):
        raise ValueError("TB_LOCAL_CONV_RELAX_LEGACY_GATES must be in (0, 0.25]")
    legacy_magnitude = 0.905148253645
    target = model._orig_mod if hasattr(model, "_orig_mod") else model
    changed = 0
    with torch.no_grad():
        for block in getattr(target, "blocks", ()):
            stages = ([getattr(block, "local_conv", None)]
                      + list(getattr(block, "local_conv_extra", ())))
            for stage in stages:
                gate = getattr(stage, "residual_gate", None)
                if gate is None:
                    continue
                effective = float(torch.tanh(gate).detach().float().cpu())
                # Match only the bit-identical forced legacy signature. A gate
                # that later learns beyond the bound must survive future resumes.
                if abs(abs(effective) - legacy_magnitude) > 0.002:
                    continue
                gate.fill_(math.atanh(math.copysign(limit, effective)))
                if opt is not None:
                    opt.state.pop(gate, None)
                changed += 1
    print(f"local-conv legacy relaxation: clipped {changed} saturated gates to "
          f"|effective|<={limit:g}; cleared affected optimizer moments", flush=True)
    return changed


def assert_clean_continuation_model(model):
    """Fail before optimizer allocation if the resume model is not checkpoint-native.

    Historical experimental classes remain in this source solely so old files
    are readable.  The clean continuation path must not instantiate any of
    them.  Exact state counts from the chosen checkpoint also catch silent
    architecture drift behind a zero-valued gate.
    """
    if os.environ.get("TB_CLEAN_CONTINUATION", "0") != "1":
        return
    target = model._orig_mod if hasattr(model, "_orig_mod") else model
    state = target.state_dict()
    keys = tuple(str(key) for key in state)
    forbidden_fragments = (
        "v41_engram.", "v41_ced", "v41_shared_kv", ".csa2_",
        ".walk.cycle_", "draft.", "mem.", "interpreter", "ni_injector",
        "lmm.",
    )
    forbidden = sorted({
        key for key in keys
        if any(fragment in key for fragment in forbidden_fragments)
    })
    if forbidden:
        raise RuntimeError(
            "clean continuation constructed rejected experimental state: "
            f"{forbidden[:12]}"
        )
    required = {}
    # Each branch is required exactly when this run builds it. A conv-pruned
    # continuation (TB_LOCAL_CONV=0 on a surgically pruned checkpoint) must
    # not trip the missing-branch gate, and the same holds for the other
    # optional residuals.
    if os.environ.get("TB_LOCAL_CONV", "0") == "1":
        required["local_conv"] = any(".local_conv." in key or ".local_conv_extra." in key for key in keys)
    if os.environ.get("TB_LEXICAL_SEMANTIC", "0") == "1":
        required["lexical_semantic"] = any(key.startswith("lexical_semantic.") for key in keys)
    if os.environ.get("TB_NO_WALK", "0") != "1":
        required["walker"] = any(".walk." in key for key in keys)
    missing = sorted(name for name, present in required.items() if not present)
    if missing:
        raise RuntimeError(f"clean continuation is missing checkpoint-native branches: {missing}")
    actual_keys = len(keys)
    actual_parameters = sum(int(value.numel()) for value in state.values())
    expected_keys = int(os.environ.get("TB_EXPECT_STATE_KEYS", "0"))
    expected_parameters = int(os.environ.get("TB_EXPECT_PARAMETER_COUNT", "0"))
    if expected_keys <= 0 or expected_parameters <= 0:
        raise RuntimeError("clean continuation expected checkpoint counts were not supplied")
    loop_summary = "loop=0"
    if os.environ.get("TB_LOOP_ENABLE", "0") == "1":
        # Recurrent depth legitimately adds exactly the LoopController tensors
        # (entry/exit linears + per-pass gate/emb). Verify the set and shapes
        # precisely; the counts below then compare the FULL model (sidecar
        # included) against the FULL prediction from the contract, so every
        # other kind of architecture drift still trips the gate.
        try:
            loop_passes = int(os.environ.get("TB_LOOP_PASSES", "2"))
        except ValueError:
            raise RuntimeError("TB_LOOP_PASSES must be an integer") from None
        controller_keys = sorted(key for key in keys if key.startswith("loop_controller."))
        expected_layout = {
            "loop_controller.entry.weight": None,
            "loop_controller.entry.bias": None,
            "loop_controller.exit.weight": None,
            "loop_controller.exit.bias": None,
            "loop_controller.pass_gate": (loop_passes,),
            "loop_controller.pass_emb": (loop_passes, None),
        }
        entry_w = state.get("loop_controller.entry.weight")
        if entry_w is None or len(tuple(entry_w.shape)) != 2 or entry_w.shape[0] != entry_w.shape[1]:
            raise RuntimeError("loop controller entry weight has unexpected geometry")
        dim = int(entry_w.shape[0])
        expected_layout["loop_controller.entry.weight"] = (dim, dim)
        expected_layout["loop_controller.entry.bias"] = (dim,)
        expected_layout["loop_controller.exit.weight"] = (dim, dim)
        expected_layout["loop_controller.exit.bias"] = (dim,)
        expected_layout["loop_controller.pass_emb"] = (loop_passes, dim)
        layout_ok = (
            set(controller_keys) == set(expected_layout)
            and all(tuple(state[key].shape) == expected_layout[key] for key in controller_keys)
        )
        if not layout_ok:
            raise RuntimeError(
                "loop controller state differs from the TB_LOOP_PASSES geometry: "
                f"{controller_keys}")
        loop_summary = (f"loop=1(section x{loop_passes}, "
                        f"+{sum(int(state[key].numel()) for key in controller_keys):,} params)")
    elif any(key.startswith("loop_controller.") for key in keys):
        raise RuntimeError(
            "loop controller state present without TB_LOOP_ENABLE=1; refusing to train")
    r_summary = "r=0"
    if os.environ.get("TB_R_ENABLE", "0") == "1":
        # Narrow-R legitimately adds exactly the RAdapter tensors
        # (proj 2D->D + LayerNorm weight/bias + pass_gain). Verify precisely;
        # the counts below then compare the FULL model (sidecar included)
        # against the FULL prediction from the contract.
        try:
            r_max = int(os.environ.get("TB_R_MAX", "8"))
        except ValueError:
            raise RuntimeError("TB_R_MAX must be an integer") from None
        r_keys = sorted(key for key in keys if key.startswith("r_adapter."))
        r_layout = {
            "r_adapter.proj.weight": None,
            "r_adapter.norm.weight": None,
            "r_adapter.norm.bias": None,
            "r_adapter.pass_gain": (r_max,),
        }
        proj_w = state.get("r_adapter.proj.weight")
        if proj_w is None or len(tuple(proj_w.shape)) != 2 or proj_w.shape[1] != 2 * proj_w.shape[0]:
            raise RuntimeError("R adapter proj weight has unexpected geometry")
        dim = int(proj_w.shape[0])
        r_layout["r_adapter.proj.weight"] = (dim, 2 * dim)
        r_layout["r_adapter.norm.weight"] = (dim,)
        r_layout["r_adapter.norm.bias"] = (dim,)
        layout_ok = (
            set(r_keys) == set(r_layout)
            and all(tuple(state[key].shape) == r_layout[key] for key in r_keys)
        )
        if not layout_ok:
            raise RuntimeError(
                "R adapter state differs from the TB_R_MAX geometry: "
                f"{r_keys}")
        r_summary = (f"r=1(R x{r_max}, "
                     f"+{sum(int(state[key].numel()) for key in r_keys):,} params)")
    elif any(key.startswith("r_adapter.") for key in keys):
        raise RuntimeError(
            "R adapter state present without TB_R_ENABLE=1; refusing to train")
    if actual_keys != expected_keys or actual_parameters != expected_parameters:
        raise RuntimeError(
            "clean continuation architecture differs from selected checkpoint: "
            f"state_keys={actual_keys}/{expected_keys} "
            f"tensor_elements={actual_parameters}/{expected_parameters}"
        )
    print(
        "CLEAN_CONTINUATION_AUDIT "
        f"state_keys={actual_keys} tensor_elements={actual_parameters} "
        f"walker={int('walker' in required)} "
        f"local_conv={int('local_conv' in required)} "
        f"lexical_semantic={int('lexical_semantic' in required)} "
        f"rejected=0 {loop_summary} {r_summary}",
        flush=True,
    )


# =============================================================================
# LOTUS+ graph memory (TB_GMEM=1)
# =============================================================================
# Persistent, inspectable memory over JSON / key-value table states, read and
# written by this Walker (ported from the graph-memory side project; full
# results in docs/LOTUS_MEMORY_REPORT.md).
#
#   state  -> graph: one node per JSON leaf (key + GMEM_V value-token slots in
#             Walker's input-embedding space); records = parents, fields = keys
#   route  -> factorized node score row(record) + col(field) from schema-free
#             lexical / computed features (v5); separate read and write routers
#   read   -> the routed node's key + value slots become GMEM_V+1 prefix vectors
#             (zero-init gated adapter) before "Q: ... A:"; Walker answers
#   write  -> LOTUS+ : Walker loops R passes over
#             [routed node | update text | GMEM_V latent slots]; latent input
#             = lat + h + beta * (h - h_prev) (learned Walk momentum); each
#             slot's final state POINTS at an update token (or null) and copies
#             its embedding verbatim; the node is overwritten through a gate.
# Training adds one episode batch per TB_GMEM_EVERY steps (writes, then reads),
# loss = route CE (write + read) + pointer CE + answer CE, scaled by
# TB_GMEM_WEIGHT. By default (TB_GMEM_WALKER_GRAD=0) only the module learns and
# Walker stays frozen for it, the regime the side project validated; with =1
# the loss also enters Walker's gradient (at weight 0.1 it measured ~4x the LM
# gradient norm). The token embedding table is always detached here, so the
# pointer copy never reshapes Walker's vocabulary. The module's own parameters live OUTSIDE WalkGPT (own
# AdamW, checkpoint key "gmem"), so Walker's state-key audit is unchanged.
GMEM_V = 8                 # value-token slots per node (read prefix = key + V)
GMEM_LORA_SUBS = ("attn.qkv", "mlp.down")   # Walker projections the memory-only LoRA adapts
GMEM_WIDTH = 256           # pointer query/key width
GMEM_NR, GMEM_NC = 14, 12  # row / column feature counts (router v5)
_GMEM_WORD = re.compile(r"\w+")
_GMEM_NUM = re.compile(r"-?\d[\d,]*\.?\d*")
_GMEM_YEAR = re.compile(r"\b(1[5-9]\d\d|20\d\d)\b")
_GMEM_CUES = {
    "max": {"most", "highest", "largest", "greatest", "biggest", "maximum", "longest", "latest", "top",
            "best", "more", "higher", "larger", "longer", "oldest"},
    "min": {"least", "lowest", "smallest", "fewest", "minimum", "shortest", "earliest", "worst", "less",
            "fewer", "lower", "youngest"},
    "first": {"first", "top", "earliest"}, "last": {"last", "final", "bottom"},
    "next": {"after", "next", "following", "succeeded", "behind"},
    "prev": {"before", "previous", "preceding", "preceded", "prior", "ahead"},
    "who": {"who", "whom", "whose"}}
_GMEM_NUM_Q = re.compile(r"\b(how many|how much|number of|what year|which year|when|how long|how old|"
                         r"what percentage|total|average)\b")
GMEM_TEMPLATES = ("Change the {s} to {v}.", "Update: the {s} is now {v}.", "The {s} has been set to {v}.",
                  "Set the {s} to {v}.")
GMEM_HELDOUT_TEMPLATE = "Note that the {s} changed to {v}."


def gmem_leaves(state):
    """One node per JSON leaf: path, key, value, parent key, scalar siblings."""
    out = []

    def walk(obj, path, parent):
        if isinstance(obj, dict):
            items = list(obj.items())
            for k, v in items:
                if isinstance(v, (dict, list)):
                    walk(v, path + (str(k),), str(k))
                else:
                    sib = [(str(k2), str(v2)) for k2, v2 in items
                           if k2 != k and not isinstance(v2, (dict, list))]
                    out.append({"path": path + (str(k),), "key": str(k), "value": str(v),
                                "parent": parent, "siblings": sib})
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                walk(v, path + (str(i),), parent)
    walk(state, (), "")
    return out


def _gmem_tri(s):
    s = f" {s} "
    return {s[i:i + 3] for i in range(len(s) - 2)}


def _gmem_prefix_match(a, b):
    return a == b or (min(len(a), len(b)) >= 3 and (a.startswith(b) or b.startswith(a)))


def _gmem_num(v):
    y = _GMEM_YEAR.search(v)
    if y:
        return float(y.group(1))
    m = _GMEM_NUM.search(v)
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except ValueError:
        return None


def gmem_features(question, leaves):
    """Router v5 features: rowf [N, GMEM_NR], colf [N, GMEM_NC] (float tensors).

    row: is a cell of this record MENTIONED (exact phrase / token recall /
    char-trigram; max and sum over the record) + computed rows (argmax/argmin
    of a named numeric column under superlative cues, first/last, after/before
    the mentioned record). col: is this field ASKED FOR (header recall / exact
    / trigram, how early it is named) + whether it HOLDS the mentioned value
    (the WHERE column) + answer-type cues (how many / when / who).
    """
    qs = " " + question.lower() + " "
    qlist = _GMEM_WORD.findall(qs)
    qw = set(qlist)
    qtri = _gmem_tri(question.lower())
    cf = []
    for lf in leaves:
        v = lf["value"].lower().strip()
        w = _GMEM_WORD.findall(v)
        if not w:
            cf.append([0.0, 0.0, 0.0])
            continue
        exact = float(re.search(r"(?<!\w)" + re.escape(v) + r"(?!\w)", qs) is not None)
        vt = _gmem_tri(v)
        cf.append([exact, sum(t in qw for t in w) / len(w), len(vt & qtri) / len(vt)])
    rows, cols = {}, {}
    for i, lf in enumerate(leaves):
        rows.setdefault(lf["path"][:-1], []).append(i)
        cols.setdefault(lf["key"], []).append(i)
    rowf = [None] * len(leaves)
    for idxs in rows.values():
        m = [max(cf[i][k] for i in idxs) for k in range(3)]
        s = [sum(cf[i][k] for i in idxs) for k in range(3)]
        for i in idxs:
            rowf[i] = m + s
    colf = [None] * len(leaves)
    for key, idxs in cols.items():
        hw = _GMEM_WORD.findall(key.lower())
        h_rec = sum(t in qw for t in hw) / max(1, len(hw))
        h_exact = float(re.search(r"(?<!\w)" + re.escape(key.lower()) + r"(?!\w)", qs) is not None)
        ht = _gmem_tri(key.lower())
        h_tri = len(ht & qtri) / max(1, len(ht))
        hs = set(hw)
        first = next((j for j, t in enumerate(qlist) if any(_gmem_prefix_match(t, h) for h in hs)), None)
        h_pos = 0.0 if first is None else 1.0 - first / max(1, len(qlist))
        h_early = float(first is not None and first <= 2)
        holds = [max(cf[i][k] for i in idxs) for k in range(3)]
        for i in idxs:
            colf[i] = [h_rec, h_exact, h_tri, h_pos, h_early] + holds
    # computed (v5) features
    cue = {k: float(bool(qw & v)) for k, v in _GMEM_CUES.items()}
    cue_num = float(_GMEM_NUM_Q.search(question.lower()) is not None)
    rkeys = list(rows)
    rpos = {rk: j for j, rk in enumerate(rkeys)}
    mention = {rk: rowf[rows[rk][0]][1] for rk in rkeys}
    named, isnum, ismax, ismin = {}, {}, {}, {}
    for key, idxs in cols.items():
        hw = [w for w in _GMEM_WORD.findall(key.lower()) if len(w) >= 2]
        named[key] = sum(any(_gmem_prefix_match(w, t) for t in qlist) for w in hw) / max(1, len(hw))
        good = {i: v for i, v in ((i, _gmem_num(leaves[i]["value"])) for i in idxs) if v is not None}
        isnum[key] = len(good) / max(1, len(idxs))
        if len(good) >= 2 and isnum[key] >= 0.5:
            hi, lo = max(good.values()), min(good.values())
            for i, v in good.items():
                ismax[i], ismin[i] = float(v == hi), float(v == lo)
    for i, lf in enumerate(leaves):
        rk = lf["path"][:-1]
        mates = rows[rk]
        f_max = max(named[leaves[j]["key"]] * ismax.get(j, 0.0) for j in mates)
        f_min = max(named[leaves[j]["key"]] * ismin.get(j, 0.0) for j in mates)
        f_max_any = max(ismax.get(j, 0.0) for j in mates)
        f_min_any = max(ismin.get(j, 0.0) for j in mates)
        p = rpos[rk]
        prev_m = mention[rkeys[p - 1]] if p > 0 else 0.0
        next_m = mention[rkeys[p + 1]] if p + 1 < len(rkeys) else 0.0
        rowf[i] = rowf[i] + [f_max * cue["max"], f_min * cue["min"], f_max_any * cue["max"],
                             f_min_any * cue["min"], float(p == 0) * cue["first"],
                             float(p == len(rkeys) - 1) * cue["last"], prev_m * cue["next"],
                             next_m * cue["prev"]]
        fn = isnum[lf["key"]]
        colf[i] = colf[i] + [fn * cue_num, (1 - fn) * cue["who"], fn * cue["who"], (1 - fn) * cue_num]
    return torch.tensor(rowf, dtype=torch.float32), torch.tensor(colf, dtype=torch.float32)


# ------------------------------------------------------------------ episodes
_GMEM_NAMES = ("Ava", "Liam", "Mila", "Noah", "Iris", "Omar", "Zara", "Theo", "Lena", "Ravi", "Yuki",
               "Nora", "Ezra", "Kofi", "Sana", "Ivan", "Maya", "Leo", "Amir", "Ines")
_GMEM_CITIES = ("Lisbon", "Oslo", "Lagos", "Quito", "Hanoi", "Perth", "Tunis", "Riga", "Cusco", "Malmo",
                "Kyoto", "Accra", "Lyon", "Pune", "Graz", "Split", "Bergen", "Porto", "Sofia", "Nantes")
_GMEM_WORDS = ("amber", "cobalt", "maple", "falcon", "quartz", "willow", "ember", "harbor", "lotus",
               "summit", "cedar", "nimbus", "onyx", "pixel", "raven", "tundra", "zephyr", "orbit")
_GMEM_STATUS = ("active", "paused", "failed", "pending", "online", "offline", "archived", "ready")
GMEM_TRAIN_KINDS = ("user", "servers", "inventory", "game")
GMEM_HELDOUT_KINDS = ("order",)
_GMEM_POOLS = {"name": _GMEM_NAMES, "city": _GMEM_CITIES, "region": _GMEM_CITIES,
               "warehouse": _GMEM_CITIES, "status": _GMEM_STATUS, "plan": _GMEM_WORDS,
               "cluster": _GMEM_WORDS, "zone": _GMEM_WORDS}


def gmem_make_state(kind, r):
    n = lambda lo, hi: str(r.randint(lo, hi))  # noqa: E731
    if kind == "user":
        return {"user": {"name": r.choice(_GMEM_NAMES), "age": n(18, 90), "plan": r.choice(_GMEM_WORDS),
                         "status": r.choice(_GMEM_STATUS),
                         "address": {"city": r.choice(_GMEM_CITIES), "zip": n(10000, 99999),
                                     "street": f"{r.choice(_GMEM_WORDS)} road"}}}
    if kind == "servers":
        names = r.sample([f"{w}-{i}" for w in ("web", "db", "api", "cache") for i in range(1, 10)],
                         r.randint(3, 5))
        return {"cluster": r.choice(_GMEM_WORDS),
                "servers": [{"name": s, "port": n(1000, 9999), "region": r.choice(_GMEM_CITIES),
                             "status": r.choice(_GMEM_STATUS)} for s in names]}
    if kind == "inventory":
        items = r.sample(_GMEM_WORDS, r.randint(3, 6))
        return {"warehouse": r.choice(_GMEM_CITIES),
                "rows": [{"item": it, "qty": n(1, 500), "price": n(2, 900),
                          "bin": f"{r.choice('ABCDEFGH')}{r.randint(1, 40)}"} for it in items]}
    if kind == "game":
        return {"player": {"name": r.choice(_GMEM_NAMES), "hp": n(1, 100), "level": n(1, 60),
                           "class": r.choice(["mage", "rogue", "knight", "ranger", "cleric"])},
                "position": {"zone": r.choice(_GMEM_WORDS), "x": n(0, 999), "y": n(0, 999)},
                "quest": {"title": f"the {r.choice(_GMEM_WORDS)} {r.choice(['key', 'gate', 'crown', 'map'])}",
                          "step": n(1, 9)}}
    if kind == "order":
        skus = r.sample([f"{w[:3].upper()}{r.randint(10, 99)}" for w in _GMEM_WORDS], r.randint(3, 5))
        return {"order": {"id": n(100000, 999999), "status": r.choice(_GMEM_STATUS),
                          "customer": {"name": r.choice(_GMEM_NAMES), "city": r.choice(_GMEM_CITIES)}},
                "lines": [{"sku": s, "qty": n(1, 50), "price": n(2, 900)} for s in skus]}
    raise ValueError(kind)


def _gmem_subject(leaf):
    if any(p.isdigit() for p in leaf["path"]):
        idk, idv = leaf["siblings"][0]
        lst = [p for p in leaf["path"] if not p.isdigit()][-2]
        return f"{leaf['key']} of the {lst} entry whose {idk} is {idv}"
    return " ".join(leaf["path"])


def gmem_synthetic_episode(r, kinds, n_writes, fits, heldout_template=False):
    """{state, leaves, writes[{node,text,value}], questions[{node,q,answer,tag}]} or None."""
    kind = r.choice(kinds)
    st = gmem_make_state(kind, r)
    lv = gmem_leaves(st)
    cand = [i for i, lf in enumerate(lv)
            if fits(lf["value"]) and not (any(p.isdigit() for p in lf["path"])
                                          and (not lf["siblings"] or lf["key"] == lf["siblings"][0][0]))]
    if len(cand) < n_writes + 1:
        return None
    pick = r.sample(cand, n_writes + 1)
    cur = [lf["value"] for lf in lv]
    writes = []
    for i in pick[:n_writes]:
        old = cur[i]
        v = None
        for _ in range(20):
            if lv[i]["key"] in _GMEM_POOLS:
                c = r.choice(_GMEM_POOLS[lv[i]["key"]])
            elif old.isdigit():
                c = str(r.randint(max(1, int(old) // 2), int(old) * 2 + 9))
            else:
                c = r.choice(_GMEM_WORDS)
            if c != old and fits(c):
                v = c
                break
        if v is None:
            return None
        tpl = GMEM_HELDOUT_TEMPLATE if heldout_template else r.choice(GMEM_TEMPLATES)
        writes.append({"node": i, "text": tpl.format(s=_gmem_subject(lv[i]), v=v), "value": v})
        cur[i] = v
    qs = []
    for i, tag in ((r.choice(pick[:n_writes]) if n_writes else pick[0], "updated" if n_writes else "untouched"),
                   (pick[-1], "untouched")):
        qs.append({"node": i, "q": f"What is the {_gmem_subject(lv[i])}?", "answer": cur[i], "tag": tag})
    return {"src": "synthetic", "kind": kind, "state": st, "leaves": lv, "writes": writes, "questions": qs}


def gmem_wikisql_tables(split, limit, fits):
    """WikiSQL SELECT col WHERE col = value questions over small real tables."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    path = hf_hub_download("Salesforce/wikisql", f"default/{split}/0000.parquet", repo_type="dataset",
                           revision="refs/convert/parquet")
    out = []
    for r in pq.read_table(path).to_pylist():
        sql, tb = r["sql"], r["table"]
        conds = sql["conds"]
        hdr = [h.strip() for h in tb["header"]]
        if sql["agg"] != 0 or len(conds["column_index"]) != 1 or conds["operator_index"][0] != 0:
            continue
        if len({h.lower() for h in hdr}) != len(hdr) or not (2 <= len(tb["rows"]) <= 40) or len(hdr) > 10:
            continue
        ci, val, sel = conds["column_index"][0], str(conds["condition"][0]).strip().lower(), sql["sel"]
        match = [i for i, row in enumerate(tb["rows"]) if str(row[ci]).strip().lower() == val]
        if len(match) != 1 or sel == ci:
            continue
        rows = [{h: str(c).strip() for h, c in zip(hdr, row)} for row in tb["rows"]]
        ans = rows[match[0]][hdr[sel]]
        if not ans or not fits(ans):
            continue
        st = {"rows": rows}
        lv = gmem_leaves(st)
        gold = next(i for i, lf in enumerate(lv) if lf["path"] == ("rows", str(match[0]), hdr[sel]))
        out.append({"q": r["question"], "state": st, "leaves": lv, "gold": gold, "answer": ans,
                    "row": match[0], "sel": hdr[sel], "where": hdr[ci]})
        if len(out) >= limit:
            break
    return out


def gmem_wikisql_episode(t, r, pool, n_writes, fits, heldout_template=False):
    """Real table + templated writes (half the time one rewrites the asked cell) + its crowd question."""
    rows, lv = t["state"]["rows"], t["leaves"]
    hdr = list(rows[0].keys())
    cur = [lf["value"] for lf in lv]
    writes, used = [], set()
    for k in range(n_writes):
        target = (t["row"], t["sel"], t["where"]) if k == 0 and r.random() < 0.5 else None
        if target is None:
            for _ in range(20):
                i, h = r.randrange(len(rows)), r.choice(hdr)
                keys = [c for c in hdr if c != h and rows[i][c]
                        and sum(x[c].strip().lower() == rows[i][c].strip().lower() for x in rows) == 1]
                if keys:
                    target = (i, h, r.choice(keys))
                    break
        if target is None:
            continue
        i, h, kc = target
        node = next(j for j, lf in enumerate(lv) if lf["path"] == ("rows", str(i), h))
        if node in used:
            continue
        cand = [v for v in (r.choice(pool) for _ in range(8)) if v != cur[node] and fits(v)]
        if not cand:
            continue
        v = r.choice(cand)
        tpl = GMEM_HELDOUT_TEMPLATE if heldout_template else r.choice(GMEM_TEMPLATES)
        writes.append({"node": node, "value": v,
                       "text": tpl.format(s=f"{h} of the rows entry whose {kc} is {rows[i][kc]}", v=v)})
        cur[node] = v
        used.add(node)
    tag = "updated" if t["gold"] in used else "untouched"
    return {"src": "wikisql", "kind": "wikisql", "state": t["state"], "leaves": lv, "writes": writes,
            "questions": [{"node": t["gold"], "q": t["q"], "answer": cur[t["gold"]], "tag": tag}]}


# ------------------------------------------------------------------ module
class GraphMemory(nn.Module):
    """LOTUS+ graph memory parameters (router, reader adapter, pointer writer)."""

    def __init__(self, D, R=4, V=GMEM_V, width=GMEM_WIDTH, lat_init=None, writer="span",
                 lora_targets=(), lora_rank=0, latent_norm=False):
        super().__init__()
        self.D, self.R, self.V = int(D), int(R), int(V)
        self.writer = str(writer)
        # latent_norm: feed LOTUS states back at token-embedding scale instead of
        # raw final-block scale (|h| ~ 683 vs |emb| ~ 1.5 per token on the 1B).
        self.latent_norm = bool(latent_norm)
        self.register_buffer("feed_scale", torch.tensor(
            float(lat_init.norm(dim=-1).mean()) if lat_init is not None else 1.0))
        self.read_wr = nn.Linear(GMEM_NR, 1)
        self.read_wc = nn.Linear(GMEM_NC, 1)
        self.read_tau = nn.Parameter(torch.tensor(1.0))
        self.write_wr = nn.Linear(GMEM_NR, 1)
        self.write_wc = nn.Linear(GMEM_NC, 1)
        self.write_tau = nn.Parameter(torch.tensor(1.0))
        self.down = nn.Linear(D, 256)
        self.up = nn.Linear(256, D)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        self.pos = nn.Parameter(torch.zeros(V + 1, D))
        lat = torch.zeros(V, D) if lat_init is None else lat_init.detach().float().clone()
        self.lat = nn.Parameter(lat + 0.02 * torch.randn(V, D))
        self.beta = nn.Parameter(torch.zeros(()))
        self.wq = nn.Linear(D, width)
        self.wk = nn.Linear(D, width)
        self.null = nn.Parameter(torch.zeros(width))
        if self.writer == "slots":
            self.gate = nn.Parameter(torch.tensor(3.0))
        # Memory-only LoRA on Walker projections. The hooks exist only inside
        # gmem_capture, so Walker's own weights and its LM forward never change;
        # the adapters give a weakly trained, frozen Walker room to adapt its
        # states to reading and writing memory.
        self.lora_rank = int(lora_rank) if lora_targets else 0
        self.lora_targets = [str(t[0]) for t in lora_targets] if self.lora_rank else []
        self.lora_A = nn.ParameterList()
        self.lora_B = nn.ParameterList()
        for _, fan_in, fan_out in (lora_targets if self.lora_rank else ()):
            self.lora_A.append(nn.Parameter(torch.randn(fan_in, self.lora_rank) / fan_in ** 0.5))
            self.lora_B.append(nn.Parameter(torch.zeros(self.lora_rank, fan_out)))

    def describe(self):
        out = f"beta={float(self.beta.detach()):+.4f}"
        if self.writer == "slots":
            out += f" gate={float(torch.sigmoid(self.gate.detach())):.3f}"
        if self.lora_rank:
            out += f" lora_B={float(sum(b.detach().float().norm() ** 2 for b in self.lora_B) ** 0.5):.4f}"
        return out

    def route_logits(self, which, rowf, colf, mask):
        wr, wc, tau = ((self.read_wr, self.read_wc, self.read_tau) if which == "read"
                       else (self.write_wr, self.write_wc, self.write_tau))
        s = wr(rowf).squeeze(-1) + wc(colf).squeeze(-1)
        return (s * tau.exp().clamp(max=100)).masked_fill(~mask, float("-inf"))

    def adapt(self, x):
        x = x.float()
        return x + self.up(F.gelu(self.down(F.layer_norm(x, x.shape[-1:])))) + self.pos


def gmem_lora_targets(target):
    """(name, fan_in, fan_out) for every Walker projection in GMEM_LORA_SUBS."""
    out = []
    for i, blk in enumerate(target.blocks):
        for sub in GMEM_LORA_SUBS:
            try:
                mod = blk.get_submodule(sub)
            except AttributeError:
                continue
            w = getattr(mod, "master", None)
            w = w if w is not None else getattr(mod, "weight", None)
            if w is not None and w.dim() == 2:
                out.append((f"blocks.{i}.{sub}", int(w.shape[1]), int(w.shape[0])))
    return out


def _gmem_lora_hook(A, B):
    def hook(_m, inp, out):
        return out + (inp[0].to(A.dtype) @ A @ B).to(out.dtype)
    return hook


@contextmanager
def gmem_capture(target, gm=None):
    """Embedding override + final-block state capture (+ the module's memory-only
    LoRA hooks) for memory forwards. Everything is removed on exit."""
    state = {"override": None, "h": None}

    def emb_hook(_m, _i, out):
        ov = state["override"]
        if ov is None:
            return out
        vec, mask = ov
        return torch.where(mask.unsqueeze(-1), vec.to(out.dtype), out)

    def blk_hook(_m, _i, out):
        state["h"] = out[0] if isinstance(out, tuple) else out

    hooks = [target.emb.register_forward_hook(emb_hook),
             target.blocks[-1].register_forward_hook(blk_hook)]
    if gm is not None and gm.lora_rank:
        for name, A, B in zip(gm.lora_targets, gm.lora_A, gm.lora_B):
            hooks.append(target.get_submodule(name).register_forward_hook(_gmem_lora_hook(A, B)))
    try:
        yield state
    finally:
        for hk in hooks:
            hk.remove()


class GmemContext:
    """Everything a memory forward needs: model, tokenizer, module, capture."""

    def __init__(self, target, tok, gm, cap):
        self.target, self.tok, self.gm, self.cap = target, tok, gm, cap
        self.dev = next(target.parameters()).device
        self.emb = target.emb.weight.detach()       # memory never reshapes the token table
        self.ph = tok("~", add_special_tokens=False)["input_ids"][0]
        self.nl = tok("\n", add_special_tokens=False)["input_ids"][0]
        self._memo = {}

    def ids(self, text):
        r = self._memo.get(text)
        if r is None:
            r = self._memo[text] = list(self.tok(text, add_special_tokens=False)["input_ids"])
        return list(r)

    def fits(self, value):
        return 0 < len(self.ids(" " + value)) <= self.gm.V

    def run(self, ids, vec=None, mask=None):
        """Teacher-forced logits [B,T,V] and final-block states [B,T,D]."""
        self.cap["override"] = None if vec is None else (vec, mask)
        try:
            with torch.autocast(self.dev.type, dtype=torch.bfloat16, enabled=self.dev.type == "cuda"):
                logits, _ = self.target(ids, r_active=False)
        finally:
            self.cap["override"] = None
        return logits.float(), self.cap["h"]

    def pad(self, seqs, fill=None):
        fill = self.ph if fill is None else fill
        T = max(len(s) for s in seqs)
        return torch.tensor([s + [fill] * (T - len(s)) for s in seqs], device=self.dev)

    def graph(self, leaves):
        """(vals [N,V] ids, key ids [N]) for one state."""
        V = self.gm.V
        vals = [(self.ids(" " + lf["value"])[:V] + [self.ph] * V)[:V] for lf in leaves]
        keys = [self.ids(" " + lf["key"])[0] for lf in leaves]
        return vals, keys


def gmem_batch_graph(ctx, eps):
    """mem [B,N,V,D], keys [B,N,D], vals ids [B,N,V], ok [B,N]."""
    B, N, V = len(eps), max(len(e["leaves"]) for e in eps), ctx.gm.V
    vals = torch.full((B, N, V), ctx.ph, dtype=torch.long, device=ctx.dev)
    keys = torch.full((B, N), ctx.ph, dtype=torch.long, device=ctx.dev)
    ok = torch.zeros(B, N, dtype=torch.bool, device=ctx.dev)
    for b, e in enumerate(eps):
        v, k = ctx.graph(e["leaves"])
        vals[b, :len(v)] = torch.tensor(v, device=ctx.dev)
        keys[b, :len(k)] = torch.tensor(k, device=ctx.dev)
        ok[b, :len(v)] = True
    return ctx.emb[vals].float(), ctx.emb[keys].float(), vals, ok


def gmem_route(ctx, which, texts, leaves_list, N):
    """Route logits [B,N] for each (text, leaves) pair."""
    B = len(texts)
    rowf = torch.zeros(B, N, GMEM_NR, device=ctx.dev)
    colf = torch.zeros(B, N, GMEM_NC, device=ctx.dev)
    mask = torch.zeros(B, N, dtype=torch.bool, device=ctx.dev)
    for b, (t, lv) in enumerate(zip(texts, leaves_list)):
        rf, cf = gmem_features(t, lv)
        rowf[b, :len(lv)] = rf.to(ctx.dev)
        colf[b, :len(lv)] = cf.to(ctx.dev)
        mask[b, :len(lv)] = True
    return ctx.gm.route_logits(which, rowf, colf, mask)


def _gmem_span_write(ctx, mem, h, full, npre, upd_ids, node, gold_vals):
    """Span writer: copy ONE contiguous span (<= V tokens) of the update verbatim
    into the routed node (hard overwrite, no blending). Every node therefore stays
    in exactly the format a freshly built node has. Training writes the gold span
    (teacher forcing) and learns start/end by CE; pred/conf are the model's own
    choice. Returns (mem, span CE, pred ids [B,V], conf [B,2] = p(start), p(end))."""
    gm = ctx.gm
    B, L = upd_ids.shape
    V, N = gm.V, mem.shape[1]
    ln = lambda x: F.layer_norm(x, x.shape[-1:])  # noqa: E731
    q = gm.wq(ln(h[:, :2]))                                                    # start / end queries
    k = gm.wk(ln(full[:, npre:npre + L]))
    scale = k.shape[-1] ** 0.5
    sc = torch.einsum("bjw,blw->bjl", q, k) / scale
    valid = upd_ids != ctx.ph
    s_null = (q[:, 0] * gm.null).sum(-1, keepdim=True) / scale                 # "write nothing"
    s_start = torch.cat([sc[:, 0].masked_fill(~valid, float("-inf")), s_null], 1)   # [B,L+1]
    s_end = sc[:, 1].masked_fill(~valid, float("-inf"))                              # [B,L]
    off = torch.arange(L, device=ctx.dev)
    allowed = ((off[None, :] - off[:, None]) >= 0) & ((off[None, :] - off[:, None]) < V)   # [start, end]
    pair = (s_start[:, :L, None] + s_end[:, None, :]).masked_fill(~allowed, float("-inf"))
    flat = pair.flatten(1).argmax(-1)
    st, en = flat // L, flat % L
    write = s_start.argmax(-1) != L
    ps = torch.softmax(s_start, -1).gather(1, st[:, None])[:, 0]
    pe = torch.softmax(s_end.masked_fill(~allowed[st], float("-inf")), -1).gather(1, en[:, None])[:, 0]
    conf = torch.stack([ps, pe], 1)
    span = torch.arange(V, device=ctx.dev)[None]

    def ids_of(a, b, w):
        idx = a[:, None] + span
        got = upd_ids.gather(1, idx.clamp(max=L - 1))
        return torch.where((idx <= b[:, None]) & w[:, None], got, torch.full_like(got, ctx.ph))

    pred = ids_of(st, en, write)
    ce, cst, cen, cw = None, st, en, write
    if gold_vals is not None:
        gst = torch.full((B,), -1, dtype=torch.long, device=ctx.dev)
        for b in range(B):
            v, row = gold_vals[b], upd_ids[b].tolist()
            for s0 in range(len(row) - len(v), -1, -1):                       # LAST occurrence = the new value
                if row[s0:s0 + len(v)] == v:
                    gst[b] = s0
                    break
        gen = gst + torch.tensor([len(v) - 1 for v in gold_vals], device=ctx.dev)
        has = gst >= 0
        if bool(has.any()):
            ce = F.cross_entropy(s_start[has], gst[has]) + F.cross_entropy(
                s_end[has].masked_fill(~allowed[gst[has]], float("-inf")), gen[has])
            cst, cen, cw = torch.where(has, gst, st), torch.where(has, gen, en), write | has
    new = ctx.emb[ids_of(cst, cen, cw)].float()
    sel = F.one_hot(node, N).float() * cw.float()[:, None]
    mem = mem + sel[:, :, None, None] * (new.unsqueeze(1) - mem)
    return mem, ce, pred, conf


def gmem_write(ctx, mem, keys, node, upd_ids, gold_vals=None, hard_ptr=False):
    """LOTUS+ write. node [B] (routed), upd_ids [B,L]. Returns (mem, CE, pred ids [B,V], conf).

    The R LOTUS+ passes are shared; the readout is the span writer (default) or
    the older per-slot pointer writer (gm.writer == "slots", soft in training)."""
    gm = ctx.gm
    B, L = upd_ids.shape
    V, D = gm.V, gm.D
    N = mem.shape[1]
    bi = torch.arange(B, device=ctx.dev)
    node_slots = torch.cat([keys[bi, node].unsqueeze(1), mem[bi, node]], 1)        # [B,V+1,D]
    npre = V + 1
    T = npre + L + V
    ids = torch.full((B, T), ctx.ph, dtype=torch.long, device=ctx.dev)
    ids[:, npre:npre + L] = upd_ids
    lat0 = T - V
    mask = torch.zeros(B, T, dtype=torch.bool, device=ctx.dev)
    mask[:, :npre] = True
    mask[:, lat0:] = True
    h = h_prev = full = None
    feed = ((lambda x: F.normalize(x, dim=-1) * gm.feed_scale) if gm.latent_norm
            else (lambda x: x))
    for _ in range(gm.R):
        lat = gm.lat.unsqueeze(0).expand(B, -1, -1)
        if h is not None:
            lat = lat + feed(h)
            if h_prev is not None:
                lat = lat + gm.beta * (feed(h) - feed(h_prev))   # Walk momentum (learned beta)
        vec = torch.cat([node_slots, torch.zeros(B, L, D, device=ctx.dev), lat], 1)
        _, H = ctx.run(ids, vec, mask)
        full = H.float()
        h_prev, h = h, full[:, lat0:]
    if gm.writer == "span":
        return _gmem_span_write(ctx, mem, h, full, npre, upd_ids, node, gold_vals)
    ln = lambda x: F.layer_norm(x, x.shape[-1:])  # noqa: E731
    q = gm.wq(ln(h))                                                            # [B,V,W]
    k = torch.cat([gm.wk(ln(full[:, npre:npre + L])), gm.null.view(1, 1, -1).expand(B, 1, -1)], 1)
    logit = torch.einsum("bjw,blw->bjl", q, k) / k.shape[-1] ** 0.5
    valid = torch.cat([upd_ids != ctx.ph, torch.ones(B, 1, dtype=torch.bool, device=ctx.dev)], 1)
    logit = logit.masked_fill(~valid.unsqueeze(1), float("-inf"))
    src = torch.cat([ctx.emb[upd_ids].float(), ctx.emb[ctx.ph].float().view(1, 1, -1).expand(B, 1, -1)], 1)
    src_ids = torch.cat([upd_ids, torch.full((B, 1), ctx.ph, dtype=torch.long, device=ctx.dev)], 1)
    ce = None
    if gold_vals is not None:
        tgt = torch.full((B, V), -100, dtype=torch.long, device=ctx.dev)
        for b in range(B):
            v = gold_vals[b]
            row = upd_ids[b].tolist()
            for s in range(len(row) - len(v), -1, -1):                # LAST occurrence = the new value
                if row[s:s + len(v)] == v:
                    tgt[b, :len(v)] = torch.arange(s, s + len(v), device=ctx.dev)
                    tgt[b, len(v):] = L
                    break
        if bool((tgt >= 0).any()):
            ce = F.cross_entropy(logit.flatten(0, 1), tgt.flatten(), ignore_index=-100)
    ptr = torch.softmax(logit, -1)
    conf, arg = ptr.max(-1)
    if hard_ptr:
        ptr = F.one_hot(arg, L + 1).float()
    new = torch.einsum("bjl,bld->bjd", ptr, src)
    pred = src_ids.gather(1, arg)
    g = torch.sigmoid(gm.gate) * F.one_hot(node, N).float()                    # only the routed node
    mem = mem + g[:, :, None, None] * (new.unsqueeze(1) - mem)
    return mem, ce, pred, conf


def gmem_read_batch(ctx, mem, keys, node, questions, answers):
    """Prefix vectors + Q/A sequences. Returns (ids [B,T], vec, mask, spans)."""
    gm = ctx.gm
    B, V = len(questions), gm.V
    bi = torch.arange(B, device=ctx.dev)
    pre = gm.adapt(torch.cat([keys[bi, node].unsqueeze(1), mem[bi, node]], 1))  # [B,V+1,D]
    seqs, spans = [], []
    for q, a in zip(questions, answers):
        p = [ctx.ph] * (V + 1) + ctx.ids(f"\nQ: {q}\nA:")
        ans = (ctx.ids(" " + a) + [ctx.nl]) if a is not None else []
        seqs.append(p + ans)
        spans.append((len(p), len(p) + len(ans)))
    ids = ctx.pad(seqs)
    mask = torch.zeros(ids.shape, dtype=torch.bool, device=ctx.dev)
    mask[:, :V + 1] = True
    vec = torch.cat([pre, torch.zeros(B, ids.shape[1] - V - 1, gm.D, device=ctx.dev)], 1)
    return ids, vec, mask, spans


@torch.no_grad()
def gmem_greedy(ctx, mem, keys, node, questions, max_new=10):
    ids, vec, mask, spans = gmem_read_batch(ctx, mem, keys, node, questions, [None] * len(questions))
    rows = [ids[b, :spans[b][0]].tolist() for b in range(len(questions))]
    pre = vec[:, :ctx.gm.V + 1]
    start = [len(r) for r in rows]
    done = [False] * len(rows)
    for _ in range(max_new):
        t = ctx.pad(rows)
        m = torch.zeros(t.shape, dtype=torch.bool, device=ctx.dev)
        m[:, :ctx.gm.V + 1] = True
        v = torch.cat([pre, torch.zeros(len(rows), t.shape[1] - ctx.gm.V - 1, ctx.gm.D, device=ctx.dev)], 1)
        logits, _ = ctx.run(t, v, m)
        last = torch.tensor([len(r) - 1 for r in rows], device=ctx.dev)
        nxt = logits[torch.arange(len(rows), device=ctx.dev), last].argmax(-1).tolist()
        for b, tk in enumerate(nxt):
            if done[b]:
                continue
            if tk == ctx.nl or tk == ctx.tok.eos_token_id:
                done[b] = True
            else:
                rows[b].append(tk)
        if all(done):
            break
    return [ctx.tok.decode(r[s:]).strip() for r, s in zip(rows, start)]


def _gmem_decode(ctx, ids):
    return ctx.tok.decode([t for t in ids if t != ctx.ph]).strip()


def gmem_episodes_pass(ctx, eps, train=True, trace=False, greedy=False):
    """Run every write, then every question, for a batch of episodes.

    Training: writes and reads use the GOLD node (teacher-forced routing; the
    routers learn from their own CE), pointers are soft. Eval (train=False):
    routed nodes and hard pointers -- exactly what inference does.
    Returns (loss or None, stats dict, trace lines).
    """
    B = len(eps)
    mem, keys, _vals, ok = gmem_batch_graph(ctx, eps)
    N = mem.shape[1]
    losses, st, lines = [], collections.defaultdict(list), []
    if trace:
        e = eps[0]
        lines.append(f"episode src={e['src']} kind={e['kind']} nodes={len(e['leaves'])} "
                     f"writes={len(e['writes'])} questions={len(e['questions'])}")
    mem0 = mem.detach().clone() if trace else None
    for k in range(max((len(e["writes"]) for e in eps), default=0)):
        bk = [b for b, e in enumerate(eps) if k < len(e["writes"])]
        ws = [eps[b]["writes"][k] for b in bk]
        idx = torch.tensor(bk, device=ctx.dev)
        logit = gmem_route(ctx, "write", [w["text"] for w in ws], [eps[b]["leaves"] for b in bk], N)
        gold = torch.tensor([w["node"] for w in ws], device=ctx.dev)
        losses.append(F.cross_entropy(logit, gold))
        pick = logit.argmax(-1)
        st["write_route"] += (pick == gold).float().tolist()
        upd = ctx.pad([ctx.ids(" " + w["text"]) for w in ws])
        gvals = [ctx.ids(" " + w["value"])[:ctx.gm.V] for w in ws]
        node = gold if train else pick
        m2, ce, pred, conf = gmem_write(ctx, mem[idx], keys[idx], node, upd,
                                        gold_vals=gvals if train else None, hard_ptr=not train)
        if ce is not None:
            losses.append(ce)
        mem = mem.index_copy(0, idx, m2)
        for j, w in enumerate(ws):
            st["write_exact"].append(float(_gmem_decode(ctx, pred[j].tolist()) == w["value"]))
        if trace and bk and bk[0] == 0:
            w, lv = ws[0], eps[0]["leaves"]
            p = torch.softmax(logit[0].detach(), -1)
            lines.append(f"  write[{k}] text={w['text']!r}")
            lines.append(f"    route -> {'/'.join(lv[int(node[0])]['path'])} p={float(p[node[0]]):.3f} "
                         f"(gold {'/'.join(lv[w['node']]['path'])}, argmax "
                         f"{'/'.join(lv[int(pick[0])]['path'])})")
            lines.append(f"    {ctx.gm.writer} writer wrote {_gmem_decode(ctx, pred[0].tolist())!r} "
                         f"(gold {w['value']!r}) conf {[round(float(c), 3) for c in conf[0].detach()]} "
                         f"{ctx.gm.describe()}")
    if trace:
        changed = ((mem[0] - mem0[0]).abs().amax((-1, -2)) > 1e-6).nonzero().flatten().tolist()
        lv = eps[0]["leaves"]
        lines.append(f"  memory nodes changed: {['/'.join(lv[i]['path']) for i in changed if i < len(lv)]}")
    qb, qs = [], []
    for b, e in enumerate(eps):
        for q in e["questions"]:
            qb.append(b)
            qs.append(q)
    if qs:
        qidx = torch.tensor(qb, device=ctx.dev)
        logit = gmem_route(ctx, "read", [q["q"] for q in qs], [eps[b]["leaves"] for b in qb], N)
        gold = torch.tensor([q["node"] for q in qs], device=ctx.dev)
        losses.append(F.cross_entropy(logit, gold))
        pick = logit.argmax(-1)
        st["read_route"] += (pick == gold).float().tolist()
        node = gold if train else pick
        ids, vec, mask, spans = gmem_read_batch(ctx, mem[qidx], keys[qidx], node, [q["q"] for q in qs],
                                                [q["answer"] for q in qs])
        logits, _ = ctx.run(ids, vec, mask)
        ce_sum, n_tok, tok_hit = 0.0, 0, []
        for b, (s, e_) in enumerate(spans):
            ce_sum = ce_sum + F.cross_entropy(logits[b, s - 1:e_ - 1], ids[b, s:e_], reduction="sum")
            n_tok += e_ - s
            tok_hit.append(float((logits[b, s - 1:e_ - 1].argmax(-1) == ids[b, s:e_]).all()))
        losses.append(ce_sum / max(1, n_tok))
        st["read_tf_exact"] += tok_hit
        st["answer_ce"].append(float(ce_sum.detach()) / max(1, n_tok))
        if greedy:
            preds = gmem_greedy(ctx, mem[qidx], keys[qidx], pick, [q["q"] for q in qs])
            for p, q in zip(preds, qs):
                hit = float(p.strip(" .\"'").lower() == q["answer"].strip().lower())
                st["em"].append(hit)
                st[f"em_{q['tag']}"].append(hit)
        if trace:
            lv = eps[0]["leaves"]
            for j, q in enumerate(qs):
                if qb[j] != 0:
                    continue
                p = torch.softmax(logit[j].detach(), -1)
                s, e_ = spans[j]
                tf = ctx.tok.decode(logits[j, s - 1:e_ - 2].argmax(-1).tolist()).strip()
                lines.append(f"  read q={q['q']!r} [{q['tag']}]")
                lines.append(f"    route -> {'/'.join(lv[int(node[j])]['path'])} p={float(p[node[j]]):.3f} "
                             f"(gold {'/'.join(lv[q['node']]['path'])})")
                lines.append(f"    answer (teacher-forced argmax) {tf!r} gold {q['answer']!r}"
                             + (f" greedy {preds[j]!r}" if greedy else ""))
    loss = sum(losses) if losses else None
    return loss, st, lines


class GmemData:
    """Episode sources: synthetic JSON states (+ held-out schema) and WikiSQL tables."""

    def __init__(self, ctx, sources, n_writes, seed=0):
        self.ctx, self.n_writes = ctx, int(n_writes)
        self.r = random.Random(seed)
        self.sources = [s for s in sources if s in ("synthetic", "wikisql")]
        self.wiki_tr = self.wiki_te = []
        self.pool = []
        if "wikisql" in self.sources:
            try:
                self.wiki_tr = gmem_wikisql_tables("train", 6000, ctx.fits)
                self.wiki_te = gmem_wikisql_tables("validation", 300, ctx.fits)
                self.pool = sorted({lf["value"] for t in self.wiki_tr for lf in t["leaves"]
                                    if lf["value"] and ctx.fits(lf["value"])})
                print(f"GMEM data: wikisql train={len(self.wiki_tr)} val={len(self.wiki_te)} "
                      f"new-value pool={len(self.pool)}", flush=True)
            except Exception as exc:
                print(f"GMEM data: wikisql unavailable ({type(exc).__name__}: {str(exc)[:160]}); "
                      "synthetic episodes only", flush=True)
                self.sources = [s for s in self.sources if s != "wikisql"]
        if not self.sources:
            self.sources = ["synthetic"]
        self.eval_sets = self._eval_sets()

    def _synthetic(self, r, kinds, heldout_template=False):
        while True:
            e = gmem_synthetic_episode(r, kinds, self.n_writes, self.ctx.fits, heldout_template)
            if e is not None:
                return e

    def _wiki(self, r, tables, heldout_template=False):
        while True:
            e = gmem_wikisql_episode(r.choice(tables), r, self.pool, self.n_writes, self.ctx.fits,
                                     heldout_template)
            if e["writes"]:
                return e

    def batch(self, B):
        out = []
        for _ in range(B):
            src = self.r.choice(self.sources)
            out.append(self._wiki(self.r, self.wiki_tr) if src == "wikisql"
                       else self._synthetic(self.r, GMEM_TRAIN_KINDS))
        return out

    def _eval_sets(self, n=48):
        r = random.Random(12345)
        sets = {"synthetic": [self._synthetic(r, GMEM_TRAIN_KINDS) for _ in range(n)],
                "synthetic_heldout_schema": [self._synthetic(r, GMEM_HELDOUT_KINDS) for _ in range(n // 2)],
                "synthetic_heldout_template": [self._synthetic(r, GMEM_TRAIN_KINDS, True) for _ in range(n // 2)]}
        if "wikisql" in self.sources and self.wiki_te:
            sets["wikisql"] = [self._wiki(r, self.wiki_te) for _ in range(n)]
        return sets


def gmem_state_dict(gm, opt=None):
    out = {"version": 2, "V": gm.V, "R": gm.R, "D": gm.D, "writer": gm.writer, "latent_norm": gm.latent_norm,
           "lora_rank": gm.lora_rank, "lora_targets": list(gm.lora_targets), "module": gm.state_dict()}
    if opt is not None:
        out["opt"] = opt.state_dict()
    return out


def gmem_build(target, tok, R, saved=None, lr=1e-3, writer="span", lora_rank=0, lora_lr=2e-4,
               latent_norm=False):
    """Module + AdamW (routers 3e-3, LoRA lora_lr, the rest lr). A saved state
    fixes its own writer / LoRA layout (version-1 states used the slot writer)."""
    dev = next(target.parameters()).device
    ph = tok("~", add_special_tokens=False)["input_ids"][0]
    D = target.emb.weight.shape[1]
    V = int(saved.get("V", GMEM_V)) if saved else GMEM_V
    found = {t[0]: t for t in gmem_lora_targets(target)}
    if saved:
        writer = saved.get("writer", "slots")
        latent_norm = bool(saved.get("latent_norm", False))
        lora_rank = int(saved.get("lora_rank", 0))
        lora = [found[n] for n in saved.get("lora_targets", [])]
    else:
        lora = list(found.values()) if lora_rank else []
    gm = GraphMemory(D, R=int(saved.get("R", R)) if saved else R, V=V,
                     lat_init=target.emb.weight[[ph] * V].detach().float().cpu(),
                     writer=writer, lora_targets=lora, lora_rank=lora_rank,
                     latent_norm=latent_norm).to(dev)
    named = list(gm.named_parameters())
    router = [p for n, p in named if n.startswith(("read_", "write_"))]
    adapters = [p for n, p in named if n.startswith("lora_")]
    rest = [p for n, p in named if not n.startswith(("read_", "write_", "lora_"))]
    groups = [{"params": router, "lr": 3e-3, "weight_decay": 0.01},
              {"params": rest, "lr": lr, "weight_decay": 0.0}]
    if adapters:
        groups.append({"params": adapters, "lr": lora_lr, "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups)
    if saved:
        state = dict(saved["module"])
        state.setdefault("feed_scale", gm.feed_scale)                       # pre-latent_norm states
        gm.load_state_dict(state)
        if "opt" in saved:
            try:
                opt.load_state_dict(saved["opt"])
            except ValueError as exc:
                print(f"GMEM: optimizer state incompatible ({exc}); fresh moments", flush=True)
    return gm, opt


def gmem_train_step(ctx, data, cfg, step):
    """One memory episode batch; gradients go to the module (and Walker unless disabled)."""
    t0 = time.time()
    trace = cfg.gmem_trace_every > 0 and step % cfg.gmem_trace_every == 0
    eps = data.batch(cfg.gmem_batch)
    loss, st, lines = gmem_episodes_pass(ctx, eps, train=True, trace=trace)
    if loss is None or not torch.isfinite(loss):
        print(f"GMEM step={step} non-finite or empty loss ({loss}); skipped", flush=True)
        return None
    scaled = loss * cfg.gmem_weight
    if cfg.gmem_walker_grad:
        scaled.backward()
    else:
        params = [p for p in ctx.gm.parameters() if p.requires_grad]
        grads = torch.autograd.grad(scaled, params, allow_unused=True)
        for p, g in zip(params, grads):
            if g is not None:
                p.grad = g if p.grad is None else p.grad + g
    out = {k: sum(v) / len(v) for k, v in st.items() if v}
    out["loss"] = float(loss.detach())
    out["sec"] = time.time() - t0
    if trace:
        print(f"GMEM_TRACE step={step} (training: gold routing, soft pointers)", flush=True)
        for line in lines:
            print("GMEM_TRACE " + line, flush=True)
    return out


def gmem_apply(gm, opt, clip=1.0):
    """Clip each optimizer group (routers / writer+adapter / LoRA) on its own, then
    step. One global clip let the LoRA gradients (Walker-scale, ~1e4) shrink the
    router and writer updates to nothing. Returns the total pre-clip norm."""
    total, finite, seen = 0.0, True, False
    for group in opt.param_groups:
        params = [p for p in group["params"] if p.grad is not None]
        if not params:
            continue
        seen = True
        gn = torch.nn.utils.clip_grad_norm_(params, clip)
        finite = finite and bool(torch.isfinite(gn))
        total += float(gn) ** 2
    if not seen:
        return float("nan")
    if finite:
        opt.step()
    else:
        print(f"GMEM: non-finite module grad norm ({total ** 0.5}); step skipped", flush=True)
    opt.zero_grad(set_to_none=True)
    return total ** 0.5


@torch.no_grad()
def gmem_evaluate(ctx, data, step):
    """Routed + hard-pointer evaluation on fixed held-out episodes (inference path)."""
    res = {}
    for name, eps in data.eval_sets.items():
        agg = collections.defaultdict(list)
        lines = []
        for s in range(0, len(eps), 16):
            _, st, ln = gmem_episodes_pass(ctx, eps[s:s + 16], train=False, trace=(s == 0), greedy=True)
            for k, v in st.items():
                agg[k] += v
            if s == 0:
                lines = ln
        res[name] = {k: round(sum(v) / len(v), 4) for k, v in agg.items() if v}
        print(f"GMEM_VAL step={step} set={name} " + " ".join(f"{k}={v}" for k, v in res[name].items()),
              flush=True)
        for line in lines:
            print(f"GMEM_VAL_TRACE step={step} set={name} " + line, flush=True)
    return res


class GraphMemorySession:
    """Inference API: load a JSON state, apply update sentences, ask questions.

    Every call returns a trace dict (routed node, route probability, written
    value, pointer confidence, which nodes changed, answer) so the memory can
    be inspected step by step.
    """

    def __init__(self, target, tok, gm):
        self.target, self.tok, self.gm = target, tok, gm
        self.leaves = []
        self.mem = self.keys = None

    def _ctx(self, cap):
        return GmemContext(self.target, self.tok, self.gm, cap)

    def load(self, state):
        self.leaves = gmem_leaves(state)
        with gmem_capture(self.target, self.gm) as cap:
            ctx = self._ctx(cap)
            self.mem, self.keys, _, _ = gmem_batch_graph(ctx, [{"leaves": self.leaves}])
        return {"nodes": ['/'.join(lf["path"]) + "=" + lf["value"] for lf in self.leaves]}

    def values(self):
        with gmem_capture(self.target, self.gm) as cap:
            ctx = self._ctx(cap)
            E = F.normalize(ctx.emb.float(), dim=-1)
            out = {}
            for i, lf in enumerate(self.leaves):
                ids = (F.normalize(self.mem[0, i], dim=-1) @ E.T).argmax(-1).tolist()
                out['/'.join(lf["path"])] = _gmem_decode(ctx, ids)
        return out

    @torch.no_grad()
    def update(self, text):
        with gmem_capture(self.target, self.gm) as cap, _eval_mode(self.target):
            ctx = self._ctx(cap)
            N = self.mem.shape[1]
            logit = gmem_route(ctx, "write", [text], [self.leaves], N)
            p = torch.softmax(logit[0], -1)
            node = logit.argmax(-1)
            before = self.mem.clone()
            self.mem, _, pred, conf = gmem_write(ctx, self.mem, self.keys, node, ctx.pad([ctx.ids(" " + text)]),
                                                 hard_ptr=True)
            changed = ((self.mem[0] - before[0]).abs().amax((-1, -2)) > 1e-6).nonzero().flatten().tolist()
            top = torch.topk(p, min(3, N))
            return {"text": text, "node": '/'.join(self.leaves[int(node)]["path"]), "p": round(float(p[node]), 4),
                    "top3": [('/'.join(self.leaves[int(i)]["path"]), round(float(v), 4))
                             for v, i in zip(top.values, top.indices)],
                    "wrote": _gmem_decode(ctx, pred[0].tolist()),
                    "pointer_conf": [round(float(c), 3) for c in conf[0]],
                    "changed": ['/'.join(self.leaves[i]["path"]) for i in changed]}

    @torch.no_grad()
    def ask(self, question):
        with gmem_capture(self.target, self.gm) as cap, _eval_mode(self.target):
            ctx = self._ctx(cap)
            N = self.mem.shape[1]
            logit = gmem_route(ctx, "read", [question], [self.leaves], N)
            p = torch.softmax(logit[0], -1)
            node = logit.argmax(-1)
            ans = gmem_greedy(ctx, self.mem, self.keys, node, [question])[0]
            return {"question": question, "node": '/'.join(self.leaves[int(node)]["path"]),
                    "p": round(float(p[node]), 4), "answer": ans}


def build_optimizer(target, cfg):
    """Role-partitioned optimizer groups with per-role LR multipliers.

    Extracted verbatim from train() so group LRs (walk/phase/r/base mults)
    are unit-testable without running training.     Returns (opt, groups).
    """
    r_mult = float(getattr(cfg, 'r_adapter_mult', 1.0))
    if not math.isfinite(r_mult) or r_mult < 0.0:
        raise ValueError(
            f"TB_R_ADAPTER_MULT must be finite and >= 0, got {r_mult:g}")
    groups = []
    semantic_wd = bool(getattr(cfg, 'semantic_wd', False))

    def _role(name):
        if 'r_adapter.' in name:
            return 'r'
        if name.startswith('lexical_semantic.'):
            return 'lexical_semantic'
        if name.startswith('lmm.'):
            return 'lmm'
        if name.startswith('mem.'):
            return 'mem'
        if '.phase_walk.' in name:
            return 'phase'
        if '.walk.' in name or name.startswith('walk_motif_bank.'):
            return 'walk'
        if '.local_conv.' in name or '.local_conv_extra.' in name:
            return 'local'
        return 'base'

    def _no_decay(name, p):
        leaf = name.rsplit('.', 1)[-1].lower()
        return p.ndim < 2 or 'gate' in name.lower() or 'norm' in name.lower() or leaf in ('bias','scale')

    named = [(n,p) for n,p in target.named_parameters()]
    if semantic_wd:
        specs = [
            ('phase_matrix', lambda n,p: _role(n)=='phase' and not _no_decay(n,p), cfg.phase_mult, 0.05),
            ('phase_nodecay', lambda n,p: _role(n)=='phase' and _no_decay(n,p), cfg.phase_mult, 0.0),
            ('walk_matrix', lambda n,p: _role(n)=='walk' and not _no_decay(n,p), cfg.adapter_mult, 0.05),
            ('walk_nodecay', lambda n,p: _role(n)=='walk' and _no_decay(n,p), cfg.adapter_mult, 0.0),
            ('r_matrix', lambda n,p: _role(n)=='r' and not _no_decay(n,p), r_mult, 0.05),
            ('r_nodecay', lambda n,p: _role(n)=='r' and _no_decay(n,p), r_mult, 0.0),
            ('mem_matrix', lambda n,p: _role(n)=='mem' and not _no_decay(n,p), cfg.mem_mult, 0.01),
            ('mem_nodecay', lambda n,p: _role(n)=='mem' and _no_decay(n,p), cfg.mem_mult, 0.0),
            ('lmm_matrix', lambda n,p: _role(n)=='lmm' and not _no_decay(n,p),
             float(os.environ.get('TB_LMM_MULT', '1.0')), 0.01),
            ('lmm_nodecay', lambda n,p: _role(n)=='lmm' and _no_decay(n,p),
             float(os.environ.get('TB_LMM_MULT', '1.0')), 0.0),
            ('local_matrix', lambda n,p: _role(n)=='local' and not _no_decay(n,p),
             float(os.environ.get('TB_LOCAL_CONV_MULT', '1.5')), 0.05),
            ('local_nodecay', lambda n,p: _role(n)=='local' and _no_decay(n,p),
             float(os.environ.get('TB_LOCAL_CONV_MULT', '1.5')), 0.0),
            ('lexical_semantic_matrix', lambda n,p: _role(n)=='lexical_semantic' and not _no_decay(n,p),
             float(os.environ.get('TB_LEXICAL_SEMANTIC_MULT', '2.0')), 0.05),
            ('lexical_semantic_nodecay', lambda n,p: _role(n)=='lexical_semantic' and _no_decay(n,p),
             float(os.environ.get('TB_LEXICAL_SEMANTIC_MULT', '2.0')), 0.0),
            ('base_matrix', lambda n,p: _role(n)=='base' and not _no_decay(n,p), 1.0, 0.05),
            ('base_nodecay', lambda n,p: _role(n)=='base' and _no_decay(n,p), 1.0, 0.0),
        ]
        for label, pred, mult, wd in specs:
            params = [p for n,p in named if pred(n,p)]
            if params:
                groups.append({'params':params,'lr':cfg.lr*mult,'_mult':mult,'weight_decay':wd,'_label':label})
    else:
        phase_p = [p for n,p in named if _role(n)=='phase']
        walk_p = [p for n,p in named if _role(n)=='walk']
        mem_p = [p for n,p in named if _role(n)=='mem']
        lmm_p = [p for n,p in named if _role(n)=='lmm']
        local_p = [p for n,p in named if _role(n)=='local']
        lexical_semantic_p = [p for n,p in named if _role(n)=='lexical_semantic']
        other_p = [p for n,p in named if _role(n)=='base']
        if phase_p:
            groups.append({'params':phase_p,'lr':cfg.lr*cfg.phase_mult,'_mult':cfg.phase_mult,'weight_decay':0.1,'_label':'phase'})
        if walk_p:
            groups.append({'params':walk_p,'lr':cfg.lr*cfg.adapter_mult,'_mult':cfg.adapter_mult,'weight_decay':0.1,'_label':'walk'})
        r_p = [p for n,p in named if _role(n)=='r']
        if r_p:
            groups.append({'params':r_p,'lr':cfg.lr*r_mult,'_mult':r_mult,'weight_decay':0.1,'_label':'r'})
        if mem_p:
            groups.append({'params':mem_p,'lr':cfg.lr*cfg.mem_mult,'_mult':cfg.mem_mult,'weight_decay':0.1,'_label':'mem'})
        if lmm_p:
            lmm_mult = float(os.environ.get('TB_LMM_MULT', '1.0'))
            groups.append({'params':lmm_p,'lr':cfg.lr*lmm_mult,'_mult':lmm_mult,'weight_decay':0.01,'_label':'lmm'})
        if local_p:
            local_mult = float(os.environ.get('TB_LOCAL_CONV_MULT', '1.5'))
            groups.append({'params':local_p,'lr':cfg.lr*local_mult,'_mult':local_mult,'weight_decay':0.05,
                           '_label':'local', '_muon': getattr(cfg, 'optimizer', 'adamw') in ('muon', 'hybrid_muon')})
        if lexical_semantic_p:
            lexical_mult = float(os.environ.get('TB_LEXICAL_SEMANTIC_MULT', '2.0'))
            groups.append({'params':lexical_semantic_p, 'lr':cfg.lr*lexical_mult,
                           '_mult':lexical_mult, 'weight_decay':0.05,
                           '_label':'lexical_semantic'})
        if other_p:
            groups.append({'params':other_p,'lr':cfg.lr,'_mult':1.0,'weight_decay':0.1,'_label':'base'})
    optimizer_mode = getattr(cfg, 'optimizer', 'adamw').strip().lower()
    if optimizer_mode in ('muon', 'hybrid_muon'):
        if optimizer_mode == 'muon':
            for group in groups:
                group['_muon'] = True
        else:
            for group in groups:
                group['_muon'] = str(group.get('_label', '')).startswith('local')
        opt = MuonAdamW(groups, lr=cfg.lr, betas=(0.9, 0.95),
                        muon_steps=int(os.environ.get('TB_MUON_STEPS', '3')))
        print(f"optimizer: {optimizer_mode} (Muon Newton-Schulz steps={os.environ.get('TB_MUON_STEPS', '3')})", flush=True)
    elif os.environ.get('TB_OPT_KAHAN', '0') == '1':
        opt = CompensatedAdamW(groups, lr=cfg.lr, betas=(0.9, 0.95))
        print("optimizer: adamw + Kahan-compensated bf16 updates (TB_OPT_KAHAN=1)", flush=True)
    else:
        try:
            opt = torch.optim.AdamW(groups, lr=cfg.lr, betas=(0.9, 0.95), fused=True)
        except Exception:
            opt = torch.optim.AdamW(groups, lr=cfg.lr, betas=(0.9, 0.95))
            print("fused AdamW unavailable - using regular AdamW", flush=True)
    print('optimizer groups:', [(g.get('_label'), len(g['params']), g.get('weight_decay'), g.get('_mult')) for g in groups], flush=True)
    return opt, groups


def train(cfg):
    print(f"manifest compatibility: {MANIFEST_COMPAT_VERSION}", flush=True)
    cuda_available = torch.cuda.is_available()
    if os.environ.get("TB_REQUIRE_CUDA", "0") == "1" and not cuda_available:
        raise RuntimeError(
            "TB_REQUIRE_CUDA=1 but torch.cuda.is_available() is false. Refusing to run the "
            "1B experiment on CPU; check the GPU runtime, CUDA-visible devices, and PyTorch build."
        )
    runtime_name = torch.cuda.get_device_name(0) if cuda_available else "CPU"
    print(f"runtime device: {DEVICE} ({runtime_name}); torch={torch.__version__}; "
          f"cuda_runtime={torch.version.cuda}", flush=True)
    if os.environ.get("TB_ANOMALY", "0") == "1":
        # debug mode: raise at the FIRST op producing nan/inf, with stack trace
        torch.autograd.set_detect_anomaly(True)
        print("ANOMALY DETECTION: ON (slow — debugging only)", flush=True)
    torch.manual_seed(cfg.seed)
    prepare_schedule(cfg)
    tok = _load_tokenizer(cfg.tokenizer)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    cfg.tok = tok
    dm = DataMix(cfg)
    if not dm.general:
        raise RuntimeError(
            f"no general shards found under TB_DATA_DIR={cfg.data_dir!r} — refusing to train CoT-only; "
            "check the launch directory, path, and file extensions (*.bin/*.npy, case-sensitive)")
    model = build_model(cfg)
    compile_mode = int(os.environ.get("TB_COMPILE", "0"))
    if getattr(cfg, "quant_backend", "ste") == "te_nvfp4" and compile_mode:
        raise RuntimeError(
            "TB_QUANT_BACKEND=te_nvfp4 currently requires TB_COMPILE=0. "
            "Keep the first native-NVFP4 resume eager so Transformer Engine "
            "owns its quantizer metadata and the throughput comparison is clean."
        )
    if compile_mode:
        try:
            _mode = "reduce-overhead" if compile_mode == 1 else "default"
            # The context curriculum intentionally changes T in 128-token
            # increments.  Dynamic tracing avoids compiling a separate graph
            # for each of those otherwise-identical shapes.  It is opt-in
            # because fixed-T jobs can sometimes get a slightly tighter graph.
            _compile_dynamic = os.environ.get("TB_COMPILE_DYNAMIC", "0") == "1"
            model = torch.compile(model, mode=_mode, fullgraph=False,
                                  dynamic=_compile_dynamic)
            # front-load the trace + cudagraph capture on a synthetic batch shaped
            # exactly like training (B, T, same autocast context) so the first
            # real step isn't a multi-minute stall
            print(f"torch.compile: ON (mode={_mode}, dynamic={int(_compile_dynamic)}) "
                  "— tracing warmup step...", flush=True)
            xb = torch.randint(0, cfg.vocab, (cfg.B, cfg.t1), device=DEVICE)
            yb = torch.randint(0, cfg.vocab, (cfg.B, cfg.t1), device=DEVICE)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.amp):
                _, wl = model(xb, yb)
                wl.backward()
            model.zero_grad(set_to_none=True)
            del xb, yb
            torch.cuda.synchronize()
            print("torch.compile: warmup done", flush=True)
        except Exception as e:
            print(f"torch.compile unavailable ({e!r}) — continuing eager", flush=True)
            model = build_model(cfg)
    target = model._orig_mod if hasattr(model, "_orig_mod") else model
    assert_clean_continuation_model(target)
    opt, groups = build_optimizer(target, cfg)
    apply_bottom_lr_mult(target, opt)
    # DraftNet removed 2026-09-30: no separate draft optimizer. TB_DRAFT=1
    # raises at config parse, so reaching here means draft is off.
    draft_opt = None
    ck = Checkpointer(cfg.ckpt_path)
    if cfg.init_ckpt and cfg.phase_ckpt:
        raise RuntimeError('TB_INIT_CKPT and TB_PHASE_CKPT are mutually exclusive')
    phase_boot = None
    if cfg.init_ckpt:
        if os.path.abspath(cfg.init_ckpt) == os.path.abspath(ck.ongoing):
            raise RuntimeError("TB_INIT_CKPT must differ from TB_CKPT_PATH; use a new output checkpoint")
        if os.path.exists(ck.ongoing):
            raise RuntimeError(f"refusing TB_INIT_CKPT because output checkpoint already exists: {ck.ongoing}")
        load_weights_only(model, cfg.init_ckpt, allow_partial=cfg.init_allow_partial)
    if cfg.phase_ckpt:
        if os.path.abspath(cfg.phase_ckpt) == os.path.abspath(ck.ongoing):
            raise RuntimeError('TB_PHASE_CKPT must differ from TB_CKPT_PATH')
        if os.path.exists(ck.ongoing):
            raise RuntimeError('Phase-B output checkpoint already exists. Unset TB_PHASE_CKPT and resume TB_CKPT_PATH normally.')
        if not os.path.exists(cfg.phase_ckpt):
            raise RuntimeError(f'TB_PHASE_CKPT not found: {cfg.phase_ckpt}')
        if cfg.phase_structural and not cfg.arch_phase.startswith('phase_'):
            raise ValueError('TB_ARCH_PHASE must use a named phase such as phase_csa2')
        phase_boot = load_phase_checkpoint(model, opt, cfg.phase_ckpt,
                                           allow_structural=cfg.phase_structural, cfg=cfg)
        print(f"PHASE TRANSITION: source step={phase_boot['source_step']:,} lifetime_toks={phase_boot['cum_toks']:,}; "
              'new phase step/schedule tokens reset to zero', flush=True)
        cfg.phase_transition_audit = phase_boot.get('audit', {})
        cfg.phase_transition_audit['target_phase'] = cfg.arch_phase
        cfg.phase_transition_audit['ce_tolerance_pct'] = cfg.phase_ce_tolerance_pct
    hf_up = HfUploader(cfg.hf_ckpt) if cfg.hf_ckpt else None
    if hf_up:
        print(f"HF checkpoint backup: ON -> {cfg.hf_ckpt} (background, resumable)", flush=True)
    (step, best_val, doremi_restored, ck_shards, stream_rows,
     draft_opt_sd, cum_toks0, schedule_toks0, data_step_base,
     saved_run_state, stream_state) = ck.resume(model, opt)
    if phase_boot is not None:
        step, best_val, doremi_restored = 0, float('inf'), None
        ck_shards = phase_boot['n_shards']
        stream_rows, stream_state = phase_boot['stream_rows'], phase_boot['stream_state']
        # A phase transition may intentionally change the *general* corpus at
        # the same time.  A legacy DCLM/FineWeb cursor cannot be restored into
        # a weighted multi-stream object, and should not be: the new phase must
        # start each new source at its own fresh cursor.  Preserve the source
        # model/Adam/RNG/lifetime tokens above, but discard only the old data
        # iterator state when the caller explicitly authorizes the transition.
        if dm.stream_mix_active:
            dataset_override = os.environ.get('TB_RESUME_DATASET', 'lock').strip().lower() == 'override'
            if not dataset_override:
                raise RuntimeError(
                    'TB_PHASE_CKPT + TB_STREAM_PRESET/TB_STREAM_MIX changes the general corpus; '
                    'set TB_RESUME_DATASET=override to explicitly start fresh weighted-stream cursors')
            stream_rows, stream_state = None, None
            print('PHASE TRANSITION: weighted data preset starts at fresh source cursors; '
                  'discarded only the old general-stream cursor/prefetch state', flush=True)
        draft_opt_sd = None
        cum_toks0, schedule_toks0 = phase_boot['cum_toks'], 0
        data_step_base = phase_boot['source_step']
        saved_run_state = None
    legacy_resume = bool(saved_run_state is None and step)
    data_change_kind = None
    cfg.train_phase, phase_restart = train_phase_plan(saved_run_state, cfg.train_phase, step)
    if saved_run_state is None and step:
        print("resume: legacy checkpoint has no schedule/config metadata — "
              "keeping current TB_* schedule; writing a migration sidecar", flush=True)
    elif saved_run_state is not None:
        validate_architecture_manifest(saved_run_state, cfg)
        data_change_kind = validate_data_manifest(saved_run_state, cfg)
        if data_change_kind:
            # A loss measured under the old corpus mixture is not comparable
            # with the new objective. Keep all training state, but start a new
            # best-checkpoint series for the new mixture.
            best_val = float('inf')
            print('resume: reset best-checkpoint comparator for new data contract', flush=True)
        saved_hash = saved_run_state.get("source_sha256", "unknown")
        current_hash = _source_hash()
        if saved_hash != "unknown" and current_hash != saved_hash:
            print(f"resume: source hash changed ({saved_hash[:12]} -> {current_hash[:12]})", flush=True)
        if not phase_restart:
            restore_schedule(cfg, saved_run_state)
    cfg.schedule_origin_step = schedule_origin(saved_run_state, step, phase_restart)
    best_val = maybe_reset_best_on_resume(best_val, step)
    if (step or phase_boot is not None) and os.environ.get("TB_LOCAL_CONV_RESEED_ZERO_GATES", "0") == "1":
        _reseed_local_conv_gates(model, opt)
    if step or phase_boot is not None:
        _relax_legacy_local_conv_gates(model, opt)
    if phase_restart:
        # New training phase: fresh optimizer moments (lazily re-initialised
        # on the first step, Adam bias correction restarts), a schedule that
        # warms up and decays from this cursor, and a new best series. Model
        # weights, data cursors and lifetime token counts are kept.
        opt.state.clear()
        draft_opt_sd = None
        schedule_toks0 = 0
        best_val = float('inf')
        phase_horizon_defaults(cfg, step)
        print(f"PHASE RESTART: train phase -> {cfg.train_phase} at step {step:,}; optimizer "
              "moments reset, LR schedule restarts here (warmup "
              f"{cfg.warmup}, peak {cfg.lr:g}, floor {cfg.lr_floor:g}), best-val series reset",
              flush=True)
    cum_toks = int(cum_toks0 or 0)
    schedule_toks = int(schedule_toks0 or 0)
    data_step_base = int(data_step_base or 0)
    cfg._bottom_ramp = bottom_isolate_ramp_plan(saved_run_state, step)
    if saved_run_state is not None and cfg.resume_schedule == "override":
        cursor_lr = scheduled_lr(cfg, step, schedule_toks)
        print(
            "resume: active schedule override "
            f"(kind={cfg.lr_schedule}, peak={cfg.lr:g}, floor={cfg.lr_floor:g}, "
            f"total_tokens={cfg.lr_total_tokens:,}, warmup_tokens={cfg.lr_warmup_tokens:,}, "
            f"cursor_tokens={schedule_toks:,}, raw_lr={cursor_lr:.8g}); "
            "per-step T scaling and optimizer-group multipliers apply after raw_lr",
            flush=True,
        )
    if cfg.fp4:
        # resume loaded fresh masters AFTER build-time quantization: re-derive
        # _q/_s so the first eval doesn't run on stale (init-weight) codes
        for mod in model.modules():
            if isinstance(mod, QLinear):
                mod._refresh_q()
    if draft_opt_sd is not None and draft_opt is not None:
        try:
            draft_opt.load_state_dict(draft_opt_sd)
            print("draft optimizer state restored", flush=True)
        except Exception as e:
            print(f"draft optimizer state incompatible ({e}) — re-warming", flush=True)
    if ck_shards is not None and ck_shards != len(dm.general):
        print(f"WARNING: checkpoint saved with {ck_shards} shards, {len(dm.general)} loaded — domain indices may differ", flush=True)
    if dm.stream is not None:
        stream_state = prepare_stream_state_for_data_change(stream_state, data_change_kind)
        if data_change_kind == 'dataset':
            stream_rows = None
            print('resume: new general dataset starts at a fresh cursor; '
                  'optimizer/schedule/lifetime tokens remain restored', flush=True)
        if data_change_kind == 'extend' and isinstance(dm.stream, MixedStreamSource):
            dm.stream.allow_extend = True
            dm.stream.rerendered = set(getattr(cfg, "mix_rerendered", []) or [])
        dm.restore_streams(stream_state, legacy_stream_rows=stream_rows)
        if isinstance(dm.stream, MixedStreamSource) and cfg.mix_val_tokens_per_source > 0:
            mix_meta_path = os.path.abspath(cfg.mix_val_path) + ".meta.json"
            # A normal full-state resume must use the exact same held-out rows,
            # even on a new cloud box.  Dataset transitions and legacy
            # migrations intentionally establish a new validation contract.
            if (step > 0 and not legacy_resume and data_change_kind not in ('dataset', 'extend')
                    and not os.path.isfile(mix_meta_path)):
                _restore_mixed_validation_from_hf(cfg)
            if data_change_kind == 'extend' and not os.path.isfile(mix_meta_path):
                _extend_mixed_validation(dm, cfg)
            prepare_mixed_validation(dm, cfg)
            _offer_mixed_validation(hf_up, cfg)
        elif cfg.fw_val_tokens > 0 and isinstance(dm.stream, (StreamSource, MixedStreamSource)):
            meta_path = cfg.fw_val_path + ".meta.json"
            if os.path.exists(cfg.fw_val_path) and os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
                # A held-out token file is meaningful only for the stream and
                # tokenizer that created it.  New metadata binds those inputs;
                # old files remain readable but are explicitly marked legacy.
                expected_capture = {
                    **dm.stream.validation_identity(),
                    "tokenizer": cfg.tokenizer,
                    "vocab": int(cfg.vocab),
                    "packing": "eos-v1",
                }
                missing_capture = [key for key in expected_capture if key not in meta]
                mismatched_capture = [key for key, value in expected_capture.items()
                                      if key in meta and meta[key] != value]
                if mismatched_capture:
                    raise RuntimeError(
                        f"FW val metadata does not match the current training stream: {mismatched_capture}. "
                        "Use a new TB_FW_VAL_PATH rather than reusing this validation file."
                    )
                if missing_capture:
                    print(f"WARNING: FW val metadata is legacy and cannot bind {missing_capture}; "
                          "it still preserves the recorded excluded row range. A new capture will be fully auditable.",
                          flush=True)
                dm.stream.add_heldout_range(meta["raw_start"], meta["raw_end"])
                # At this point the iterator's raw position == stream_rows (skip()
                # consumes raw rows 1:1). The capture region is [raw_start, raw_end)
                # in raw coordinates. If the checkpoint predates the capture (or
                # short-row drift puts the cursor inside the region), advance the
                # exact gap so the held-out rows are never re-read into training.
                raw_pos = dm.stream.raw_cursor
                if raw_pos < meta["raw_end"]:
                    dm.stream.skip_more(meta["raw_end"] - raw_pos)
                    print(f"FW val: cursor at {raw_pos:,} < capture end {meta['raw_end']:,} — "
                          f"skipped {meta['raw_end'] - raw_pos:,} held-out rows", flush=True)
                dm.fw_val = _load_shard(cfg.fw_val_path)
                print(f"FW val loaded from {cfg.fw_val_path} ({len(dm.fw_val):,} tokens, unseen)", flush=True)
            else:
                raw_start = dm.stream.raw_cursor
                dm.fw_val, _, raw_n = dm.stream.capture_val(cfg.fw_val_tokens)
                meta = {"version": 3,
                        **dm.stream.validation_identity(),
                        "tokenizer": cfg.tokenizer,
                        "vocab": int(cfg.vocab),
                        "packing": "eos-v1",
                        "raw_rows": raw_n,
                        "raw_start": raw_start,
                        "raw_end": raw_start + raw_n,
                        "tokens": len(dm.fw_val)}
                dm.stream.add_heldout_range(meta["raw_start"], meta["raw_end"])
                tmp = cfg.fw_val_path + ".tmp"
                __import__("numpy").asarray(dm.fw_val.numpy(), dtype="uint16").tofile(tmp)
                os.replace(tmp, cfg.fw_val_path)
                with open(meta_path, "w") as f:
                    json.dump(meta, f)
                print(f"FW val captured ahead of training cursor: {len(dm.fw_val):,} tokens "
                      f"({raw_n:,} raw rows; saved {cfg.fw_val_path} + meta; "
                      f"region [{raw_start:,}, {meta['raw_end']:,}))", flush=True)
        # both streams start after held-out captures are ready

    if isinstance(dm.cot, StreamCoT):
        dm.cot.prefill()
    if legacy_resume:
        # The old payload cannot reveal its historical TB_* values.  Capture
        # the explicitly supplied current contract once, without rewriting GBs
        # of model/optimizer tensors; future resumes will lock to this sidecar.
        saved_run_state = run_state(cfg)
        ck.save_metadata(step, saved_run_state)
        if hf_up and os.path.exists(ck.metadata):
            hf_up.offer(ck.metadata, "ckpt_ongoing.pt.metadata.json")
        print(f"resume: legacy metadata migrated at step {step} -> {ck.metadata}", flush=True)
    # Separate held-out code validation, captured before the code training pump starts.
    if dm.code_stream is not None and cfg.code_val_tokens > 0:
        meta_path = cfg.code_val_path + '.meta.json'
        expected = {
            'stream': cfg.code_stream,
            'revision': cfg.code_stream_revision,
            'data_dir': cfg.code_data_dir,
            'text_field': cfg.code_text_field,
            'repo_field': cfg.code_repo_field,
            'path_field': cfg.code_path_field,
            'lang_field': cfg.code_lang_field,
            'lang_filter': cfg.code_lang_filter,
            'tokenizer': cfg.tokenizer,
            'vocab': int(cfg.vocab),
        }
        if os.path.exists(cfg.code_val_path) and os.path.exists(meta_path):
            with open(meta_path, encoding='utf-8') as handle:
                meta = json.load(handle)
            bad = [key for key, value in expected.items() if meta.get(key) != value]
            if bad:
                raise RuntimeError(f'CODE val metadata mismatch for {bad}; use a new TB_CODE_VAL_PATH')
            dm.code_stream.add_heldout_range(meta['raw_start'], meta['raw_end'])
            if dm.code_stream.raw_cursor < int(meta['raw_end']):
                dm.code_stream.skip_more(int(meta['raw_end']) - dm.code_stream.raw_cursor)
            dm.code_val = _load_shard(cfg.code_val_path, cap_tokens=None)
            print(f"CODE val loaded from {cfg.code_val_path} ({len(dm.code_val):,} tokens, unseen)", flush=True)
        else:
            raw_start = dm.code_stream.raw_cursor
            dm.code_val, raw_n = dm.code_stream.capture_val(cfg.code_val_tokens)
            dm.code_val.numpy().astype('uint16').tofile(cfg.code_val_path)
            meta = dict(expected, version=1, raw_start=int(raw_start), raw_end=int(raw_start + raw_n))
            dm.code_stream.add_heldout_range(meta['raw_start'], meta['raw_end'])
            with open(meta_path + '.tmp', 'w', encoding='utf-8') as handle:
                json.dump(meta, handle, indent=2, sort_keys=True)
            os.replace(meta_path + '.tmp', meta_path)
            print(f"CODE val captured {len(dm.code_val):,} tokens across {raw_n:,} raw rows -> {cfg.code_val_path}", flush=True)
    dm.start_streams()

    # LOTUS+ graph memory (TB_GMEM=1): module + own optimizer + episode data.
    # Its state rides in the checkpoint under "gmem"; a checkpoint that has it
    # keeps it even when this run does not train it.
    gmem = gmem_opt = gmem_data = None
    gmem_stats = None
    if cfg.gmem:
        gmem, gmem_opt = gmem_build(target, tok, cfg.gmem_r, saved=ck.gmem_saved, lr=cfg.gmem_lr,
                                    lora_rank=cfg.gmem_lora_rank)
        ck.extra_state = lambda with_opt: {"gmem": gmem_state_dict(gmem, gmem_opt if with_opt else None)}
        with gmem_capture(target, gmem) as _cap:
            gmem_data = GmemData(GmemContext(target, tok, gmem, _cap), cfg.gmem_sources,
                                 cfg.gmem_writes, seed=cfg.seed + step)
        print(f"GMEM: LOTUS+ graph memory ON ({'resumed' if ck.gmem_saved else 'fresh'}): "
              f"params={sum(p.numel() for p in gmem.parameters()):,} R={gmem.R} V={gmem.V} "
              f"writer={gmem.writer} lora_rank={gmem.lora_rank} (on {len(gmem.lora_targets)} Walker projections) "
              f"weight={cfg.gmem_weight} every={cfg.gmem_every} batch={cfg.gmem_batch} "
              f"writes={cfg.gmem_writes} walker_grad={int(cfg.gmem_walker_grad)} "
              f"sources={gmem_data.sources} eval_sets="
              f"{ {k: len(v) for k, v in gmem_data.eval_sets.items()} }", flush=True)
    elif ck.gmem_saved is not None:
        _kept_gmem = ck.gmem_saved
        ck.extra_state = lambda with_opt: {"gmem": _kept_gmem}

    # A data-only continuation needs a pre-update baseline on the new fixed
    # validation contract. Historical best_val belongs to the old corpus and
    # is automatically reset after the explicit data-manifest transition. This evaluation does
    # not touch optimizer, scheduler, training RNG, or stream cursors. An
    # append-only extension (a new phase's mix) gains banks, so it needs the
    # same reference.
    if (step > 0 and data_change_kind in ('dataset', 'extend')
            and os.environ.get('TB_BASELINE_VAL_ON_RESUME', '0') == '1'):
        baseline_val = evaluate(model, dm, cfg, cfg.n_eval_batches)[0]
        print(f"CONTINUE BASELINE: step={step} mixed_ce={baseline_val:.6f} "
              "before any optimizer update", flush=True)
        for name, metrics in dm.last_source_val_metrics.items():
            print(f"PHASE_VAL_SOURCE step={step} name={name} weight={metrics['weight']:.6f} "
                  f"ce={metrics['ce']:.6f} top1={metrics['top1']:.6f} "
                  f"top5={metrics['top5']:.6f}" + _source_difficulty_suffix(metrics), flush=True)
        if dm.last_val_summary:
            print(f"PHASE_VAL_MIX step={step} weighted_ce={dm.last_val_summary['weighted_ce']:.6f} "
                  f"macro_ce={dm.last_val_summary['macro_ce']:.6f} "
                  f"worst_ce={dm.last_val_summary['worst_ce']:.6f} "
                  f"worst_source={dm.last_val_summary['worst_source']}", flush=True)
        if cfg.eval_t2 and cfg.eval_t2 != cfg.eval_t:
            _long_cfg = Cfg.__new__(Cfg)
            _long_cfg.eval_t = cfg.eval_t2
            _long_batches = max(1, math.ceil(cfg.n_eval_batches * cfg.eval_t / cfg.eval_t2))
            _long_baseline = evaluate(
                model, dm, _long_cfg, _long_batches, include_train_loss=False,
            )[0]
            print(f"CONTINUE BASELINE_LONG step={step} val@{cfg.eval_t2}={_long_baseline:.6f} "
                  "before any optimizer update", flush=True)
    if gmem is not None:
        with gmem_capture(target, gmem) as _cap, _eval_mode(model):
            gmem_evaluate(GmemContext(target, tok, gmem, _cap), gmem_data, step)

    # A structural phase is intentionally not called an exact resume.  Gate it
    # on a fixed held-out CE before any optimizer step or output checkpoint is
    # created.  The source checkpoint's best_val is the default reference;
    # TB_PHASE_REFERENCE_CE can override it when the validation contract has
    # changed.  This keeps a bad surgery from silently becoming the new parent.
    phase_behavior_gate = (
        os.environ.get('TB_V41_CSA2_MODE', 'blend').strip().lower() == 'shared_only'
    )
    if phase_boot is not None and (cfg.phase_structural or phase_behavior_gate):
        # A source checkpoint's best_val belongs to its historical validation
        # distribution. Never compare that scalar to the new all-source suite.
        reference_ce = (cfg.phase_reference_ce if math.isfinite(cfg.phase_reference_ce)
                        else (float('nan') if dm.mix_val
                              else float(phase_boot.get('reference_ce', float('nan')))))
        if not math.isfinite(reference_ce):
            if not cfg.phase_allow_unchecked:
                raise RuntimeError(
                    'structural phase transition requires TB_PHASE_REFERENCE_CE or a finite '
                    'source best_val; set TB_PHASE_ALLOW_UNCHECKED=1 only for a deliberate probe')
            gate_val = evaluate(model, dm, cfg, cfg.phase_gate_batches)[0]
            audit = getattr(cfg, 'phase_transition_audit', {})
            audit.update({'reference_ce': None, 'post_transition_ce': gate_val,
                          'gate_batches': cfg.phase_gate_batches, 'gate_passed': None,
                          'reason': 'new_validation_contract'})
            cfg.phase_transition_audit = audit
            print(f'PHASE GATE: new validation contract; recorded step-0 mixed CE={gate_val:.6f} '
                  '(no invalid comparison to historical best_val)', flush=True)
            for name, metrics in dm.last_source_val_metrics.items():
                print(f"PHASE_VAL_SOURCE step=0 name={name} weight={metrics['weight']:.6f} "
                      f"ce={metrics['ce']:.6f} top1={metrics['top1']:.6f} "
                      f"top5={metrics['top5']:.6f}" + _source_difficulty_suffix(metrics), flush=True)
            if dm.last_val_summary:
                print(f"PHASE_VAL_MIX step=0 weighted_ce={dm.last_val_summary['weighted_ce']:.6f} "
                      f"macro_ce={dm.last_val_summary['macro_ce']:.6f} "
                      f"worst_ce={dm.last_val_summary['worst_ce']:.6f} "
                      f"worst_source={dm.last_val_summary['worst_source']}", flush=True)
        else:
            gate_val = evaluate(model, dm, cfg, cfg.phase_gate_batches)[0]
            tolerance = max(0.0, cfg.phase_ce_tolerance_pct) / 100.0
            allowed = reference_ce * (1.0 + tolerance)
            audit = getattr(cfg, 'phase_transition_audit', {})
            audit.update({'reference_ce': reference_ce, 'post_transition_ce': gate_val,
                          'allowed_ce': allowed, 'gate_batches': cfg.phase_gate_batches,
                          'gate_passed': bool(math.isfinite(gate_val) and gate_val <= allowed)})
            cfg.phase_transition_audit = audit
            print(f"PHASE GATE: reference_ce={reference_ce:.6f} post_ce={gate_val:.6f} "
                  f"allowed={allowed:.6f} tolerance={cfg.phase_ce_tolerance_pct:.3g}% "
                  f"passed={audit['gate_passed']}", flush=True)
            if not audit['gate_passed']:
                raise RuntimeError(
                    'structural phase transition rejected by CE gate; source checkpoint was not modified')

    n_params = sum(p.numel() for p in model.parameters())
    _qkinds = sorted({mod.quant for mod in model.modules() if isinstance(mod, QLinear)})
    _prec = f"fp4/fp8+bf16 masters ({_qkinds})" if cfg.fp4 else "bf16"
    print(f"model params: {n_params:,} | precision: {_prec} | device {DEVICE} cap {CAP}", flush=True)
    # Performance audit record (2026-09-09, RTX 4050, representative CUDA
    # microbenchmarks): cached STE dequant 1.064x; fused depthwise CNN 1.255x;
    # window chunk 512 1.404x; dual Phase associative scan 1.619x; BF16-only
    # dense Walk mix 2.57x with trajectory arithmetic still FP32. The combined
    # D=256/L=4/T=1024/accum=4 paired tests improved 1.18x--1.25x. All
    # checkpoint keys/shapes are unchanged; parity tests cover outputs and gradients. Cloud hardware
    # must still confirm the combined production gain. A 120-step paired tiny
    # FineWeb-token run produced identical BF16-resolution validation CE under
    # the old and optimized arithmetic; its speed gain was only 1.009x because
    # T=64 is too small for these sequence-length-scaled kernels to dominate.
    print("perf paths: "
          f"dequant_cache={os.environ.get('TB_QUANT_CACHE_DEQUANT', os.environ.get('TB_FP8_CACHE_DEQUANT','1'))} "
          f"attn_window_chunk={os.environ.get('TB_ATTN_WINDOW_CHUNK','256')} "
          f"cnn_kernel_fuse={os.environ.get('TB_LOCAL_CONV_FUSE_KERNELS','1')} "
          f"phase_assoc_fuse={os.environ.get('TB_PHASE_FUSE_ASSOC','1')} "
          f"walk_mix_tensorcore={os.environ.get('TB_WALK_MIX_TENSORCORE','0')}", flush=True)
    if getattr(cfg, 'phase_b_superset', False):
        print(f"Phase-B runtime: pred_scale={float(os.environ.get('TB_PHASE_PRED_SCALE','0')):.4g} "
              f"pred_weight={float(os.environ.get('TB_PHASE_PRED_WEIGHT','0')):.4g} "
              f"archive_scale={float(os.environ.get('TB_PHASE_ARCHIVE_SCALE','0')):.4g}", flush=True)
    if os.environ.get('TB_TRAJ_REG', '0') == '1':
        print("trajectory regression: ON "
              f"rank={os.environ.get('TB_TRAJ_REG_RANK', '32')} "
              f"horizon={os.environ.get('TB_TRAJ_REG_HORIZON', '8')} "
              f"scale={os.environ.get('TB_TRAJ_REG_SCALE', '1.0')} "
              f"aux={os.environ.get('TB_TRAJ_REG_AUX_WEIGHT', '0.002')} "
              f"gate_cost={os.environ.get('TB_TRAJ_REG_GATE_COST', '0.001')}", flush=True)
    t0 = time.time()
    cfg._mem_t0 = time.time()
    # Steps covered by each 100-step timing window. A resume that starts off
    # a multiple of 100 (e.g. 29750) makes the first window shorter; dividing
    # its time by 100 under-reported s/step by that fraction (8 vs 16 s).
    cfg._mem_step = step
    t_data = t_model = t_bwd = t_opt = t_draft = 0.0
    prof_t0 = time.time()
    prof_step = step
    prev_T = t_at(0, schedule_horizon_steps(cfg), cfg.t0, cfg.t1)
    # Terminal-report state. The FINAL block (after the loop) reuses whatever
    # the last eval produced; initialize it up front so a loop that never runs
    # (resume step at or past the target) still reports cleanly instead of
    # raising UnboundLocalError. Every value is overwritten by the eval block
    # each time it runs, so this is behavior-neutral for ordinary runs.
    vl = a1 = a5 = tl = ctl = fwl = float("nan")
    source_val_metrics = collections.OrderedDict()
    val_summary = {}
    code_v = float("nan")
    walk_off = None
    phase_ab = {}
    extension_ab = {}
    while step < cfg.steps:
        T = t_at(step, schedule_horizon_steps(cfg), cfg.t0, cfg.t1)
        csa2_alpha = csa2_alpha_at(cfg, step)
        if csa2_alpha is not None:
            set_csa2_alpha(model, csa2_alpha)
            if step == 0 or step == cfg.v41_csa2_ramp_steps:
                print(f"CSA2 bridge: alpha={csa2_alpha:.4f} at phase step {step} "
                      f"({cfg.v41_csa2_alpha_start:.3f}->{cfg.v41_csa2_alpha_end:.3f} "
                      f"over {cfg.v41_csa2_ramp_steps} steps)", flush=True)
        if T != prev_T and torch.cuda.is_available():
            torch.cuda.synchronize()
            try:
                torch._C._cuda_clearCublasWorkspaces()
            except Exception:
                pass
            torch.cuda.empty_cache()
            prev_T = T
            print(f"MEM T->{T}: cleared caches | resv={torch.cuda.memory_reserved()/2**20:.0f}MB "
                  f"| {_mem_allocator_breakdown()}", flush=True)
        share = min(cot_share_at(step, schedule_horizon_steps(cfg)), cfg.cot_share_max) if dm.cot is not None else 0.0
        code_share = code_share_at(schedule_toks, cfg)
        model.hybrid_w = w_at(T, cfg.t1)
        for blk in model.blocks:
            if blk.attn is not None:
                blk.attn.hybrid_w = w_at(T, cfg.t1)
        # LoopWalk schedule (looped-transformer style): ramp loop count with the
        # curriculum, randomize as well — training at varied depth makes the walk
        # robust to any loop count at inference (weight-tied recurrence).
        if hasattr(model, "walk_loop_sched") and model.walk_loop_sched != 1:
            sched = model.walk_loop_sched
            if sched == "schedule":
                # ramp 1 -> 3 over the first 75% of steps (like the T-ramp)
                prog = min(1.0, step / (0.75 * schedule_horizon_steps(cfg)))
                k = 1 + min(2, int(prog * 3))   # 1,2,3
            else:  # "random"
                k = 1 + torch.randint(0, 3, (1,)).item()  # 1..3
            for blk in model.blocks:
                blk.walk_loop = k
                if blk.walk is not None:
                    blk.walk.loop_count_current = k
        # Scale LR by 1/sqrt(T/t0): gradient variance grows with sequence length, so
        # longer contexts get proportionally lower LR. At T=t0 the factor is 1.0; at
        # T=2048 (with t0=512) it is 0.5. This smooths the T-step gradient spikes
        # without the resonance issues of a per-transition re-warmup.
        t_scale = math.sqrt(cfg.t0 / T) if T > 0 else 1.0
        raw_lr = scheduled_lr(cfg, step, schedule_toks)
        base_lr = raw_lr * t_scale
        for g in groups:
            g["lr"] = base_lr * g.get("_mult", 1.0)
        for gi, g in enumerate(groups):
            opt.param_groups[gi]["lr"] = g["lr"]
        if cfg.freeze_walk_at and step >= cfg.freeze_walk_at and not getattr(model, "_walk_frozen", False):
            for n, p in model.named_parameters():
                if ".walk." in n:
                    p.requires_grad_(False)
            model._walk_frozen = True
            print(f"WALK FROZEN at step {step}", flush=True)
        acc_loss, acc_toks = 0.0, 0
        doremi_lvals = []
        if cfg.doremi and len(dm.general) > 1:
            doremi_srcs = ([dm.cot] if dm.cot is not None else []) + dm.general
            nd = len(doremi_srcs)
            if step == 0 or doremi_restored is None:
                alpha = torch.ones(nd) / nd
                ema_loss = torch.zeros(nd)
                ref_sums = [0.0] * nd
                ref_cnts = [0] * nd
                warmup_end = int(schedule_horizon_steps(cfg) * cfg.doremi_warmup)
                if step > 0 and doremi_restored is None:
                    # Resuming (or newly enabling DoReMi) without saved state:
                    # reference tensors don't exist yet, so force a fresh
                    # warmup window instead of crashing on undefined ref_loss
                    # at the first DRO update below.
                    warmup_end = step + max(1, warmup_end)
                    print(f"DoReMi: no saved state at step {step} — fresh "
                          f"reference warmup until step {warmup_end}", flush=True)
                if step == 0:
                    print(f"DoReMi: {nd} domains (0=CoT, 1..{len(dm.general)}=general), "
                          f"reference warmup {warmup_end} steps", flush=True)
            else:
                alpha, ref_loss, ema_loss, ref_n, warmup_end = doremi_restored
                ref_sums = [float(x) * int(c) for x, c in zip(ref_loss, ref_n)]
                ref_cnts = [int(c) for c in ref_n]
                if alpha.numel() != nd:
                    print(f"DoReMi state has {alpha.numel()} domains but {nd} domains now — re-warming", flush=True)
                    alpha = torch.ones(nd) / nd
                    ema_loss = torch.zeros(nd)
                    ref_sums = [0.0] * nd
                    ref_cnts = [0] * nd
            for gi in range(cfg.accum):
                if step < warmup_end:
                    d_idx = (step + gi) % nd
                else:
                    d_idx = torch.multinomial(alpha, 1).item()
                t0 = time.time()
                xb, yb = dm.get_batch_domain((data_step_base + step) * cfg.accum + gi, cfg.B, T, d_idx, srcs=doremi_srcs)
                t_data += time.time() - t0
                t0 = time.time()
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.amp):
                    logits, loss = model(xb, yb, r_active=_microbatch_r_active(dm))
                t_model += time.time() - t0
                if gi == cfg.accum - 1:
                    # Per-step auxiliaries: added once, then divided by accum
                    # below — scale up so the effective gradient equals the
                    # configured weight (not weight/accum).
                    if cfg.traj_smooth > 0.0 and getattr(model, "traj_smooth_total", 0.0):
                        loss = loss + cfg.accum * cfg.traj_smooth * model.traj_smooth_total
                    if cfg.traj_acc > 0.0 and getattr(model, "traj_acc_total", 0.0):
                        loss = loss + cfg.accum * cfg.traj_acc * model.traj_acc_total
                lval = loss.item()
                if step < 3:
                    print(f"DBG step={step} gi={gi} d={d_idx} loss={lval:.4f}", flush=True)
                if not math.isfinite(lval) and not getattr(cfg, "_nan_fwd_diag_done", False):
                    cfg._nan_fwd_diag_done = True
                    print(f"step {step}: forward loss already non-finite ({lval}) at T={T} "
                          f"— source is in the forward pass, not just backward/STE grad", flush=True)
                t0 = time.time()
                (loss / cfg.accum).backward()
                _wdr = getattr(getattr(model, "_orig_mod", model), "wdr", None)
                if _wdr is not None:
                    _wdr.aux_backward(1.0 / cfg.accum)
                t_bwd += time.time() - t0
                doremi_lvals.append(lval)
                if step < warmup_end:
                    ref_sums[d_idx] += lval
                    ref_cnts[d_idx] += 1
                else:
                    ema_loss[d_idx] = 0.9 * ema_loss[d_idx] + 0.1 * lval
            if step == warmup_end:
                if min(ref_cnts) == 0:
                    raise RuntimeError(
                        f"DoReMi warmup failed: ref counts {ref_cnts} — "
                        "some domains got zero reference samples (check the warmup loop)")
                ref_loss = torch.tensor([s / max(1, c) for s, c in zip(ref_sums, ref_cnts)])
                ref_n = torch.tensor(ref_cnts, dtype=torch.float32)
                ema_loss = ref_loss.clone()
                print("DoReMi reference losses:", [round(float(x), 4) for x in ref_loss], flush=True)
                print("DoReMi ref counts:", ref_cnts, flush=True)
            if step >= warmup_end:
                # Group-DRO weight update: upweight domains above their reference,
                # floored to prevent degenerate collapse with many similar domains
                delta = ema_loss - ref_loss
                alpha = alpha * torch.exp(cfg.doremi_eta * delta.clamp(-5, 5))
                alpha = alpha.clamp(min=1.0 / (nd * 10.0))
                alpha = alpha / alpha.sum()
                if step % cfg.log_every == 0:
                    print("DoReMi alpha:", [round(float(a), 4) for a in alpha],
                          "losses:", [round(float(e), 4) for e in ema_loss], flush=True)
            doremi_state = (alpha, ref_loss, ema_loss, ref_n, warmup_end)
            # Report the true mean train loss (previously this stayed 0.0,
            # printing train=0.0000 and blinding the NaN diagnostic below).
            acc_loss = sum(doremi_lvals) / len(doremi_lvals) if doremi_lvals else 0.0
            if not math.isfinite(acc_loss) and not getattr(cfg, "_nan_fwd_diag_done", False):
                cfg._nan_fwd_diag_done = True
                print(f"step {step}: forward loss already non-finite ({acc_loss}) at T={T} "
                      "— source is in the forward pass, not just backward/STE grad", flush=True)
        else:
            doremi_state = None
            loss_reports, loss_scored = [], []
            # Per-microbatch source+loss attribution for loss-spike forensics.
            # Off by default (zero overhead, no extra syncs); TB_DEBUG_MICROBATCH=1
            # prints one line per microbatch showing exactly which source each
            # of the 15 windows came from and its individual CE.
            _dbg_micro = os.environ.get("TB_DEBUG_MICROBATCH", "0") == "1"
            for gi in range(cfg.accum):
                t0 = time.time()
                xb, yb = get_mixed_training_batch(
                    dm, (data_step_base + step) * cfg.accum + gi,
                    cfg.B, T, share, code_share)
                t_data += time.time() - t0
                t0 = time.time()
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.amp):
                    logits, loss = model(xb, yb, r_active=_microbatch_r_active(dm))
                t_model += time.time() - t0
                if gi == cfg.accum - 1:
                    # Per-step auxiliaries: added once, then divided by accum
                    # below — scale up so the effective gradient equals the
                    # configured weight (not weight/accum).
                    if cfg.traj_smooth > 0.0 and getattr(model, "traj_smooth_total", 0.0):
                        loss = loss + cfg.accum * cfg.traj_smooth * model.traj_smooth_total
                    if cfg.traj_acc > 0.0 and getattr(model, "traj_acc_total", 0.0):
                        loss = loss + cfg.accum * cfg.traj_acc * model.traj_acc_total
                loss_reports.append(loss.detach())
                _valid = getattr(model, "_last_lm_valid", None)
                loss_scored.append(_valid.gt(0) if torch.is_tensor(_valid)
                                   else torch.ones((), dtype=torch.bool, device=loss.device))
                if _dbg_micro:
                    print(_microbatch_debug_line(
                        step, gi, _microbatch_source_name(dm),
                        loss.detach().float().mean().item()), flush=True)
                t0 = time.time()
                (loss / cfg.accum).backward()
                _wdr = getattr(getattr(model, "_orig_mod", model), "wdr", None)
                if _wdr is not None:
                    _wdr.aux_backward(1.0 / cfg.accum)
                t_bwd += time.time() - t0
            # Report the mean over microbatches that scored a target: an
            # all-masked SFT window contributes no gradient and no loss.
            acc_loss = masked_report_mean(torch.stack(loss_reports), torch.stack(loss_scored))
            if not math.isfinite(acc_loss) and not getattr(cfg, "_nan_fwd_diag_done", False):
                cfg._nan_fwd_diag_done = True
                print(f"step {step}: forward loss already non-finite ({acc_loss}) at T={T} "
                      "— source is in the forward pass, not just backward/STE grad", flush=True)
        # gradients accumulated per-microbatch above (only 1 graph alive at a time)
        # NOTE: acc_loss now holds the TRUE averaged train loss (sum of
        # loss/accum over microbatches) — no longer a single-microbatch sample.
        acc_toks += xb.numel() * cfg.accum
        cum_toks += xb.numel() * cfg.accum
        schedule_toks += xb.numel() * cfg.accum
        del xb, yb, logits
        gmem_stats = None
        if gmem is not None and step % cfg.gmem_every == 0:
            with gmem_capture(target, gmem) as _cap:
                _gctx = GmemContext(target, tok, gmem, _cap)
                _gctx._memo = gmem_data.ctx._memo
                gmem_stats = gmem_train_step(_gctx, gmem_data, cfg, step)
        # true grad norm BEFORE clipping (single clip, honest log value)
        # IMPORTANT: clip ONLY main-model params — the draft has its own
        # optimizer, and a nan draft gradient must never poison the main update
        t0 = time.time()
        _main_named = [(n, p) for n, p in model.named_parameters() if not n.startswith("draft.")]
        _main_params = [p for _, p in _main_named]
        # Grad clipping is OFF by default (TB_GRAD_CLIP=0): the priority is
        # hyperparameter tuning (LR/schedule/ratio), not safety crutches that
        # mask real scaling issues. Set TB_GRAD_CLIP=1.0 for the old behavior
        # or a large value (e.g. 10) for light use. gnorm is ALWAYS logged:
        # computed directly from raw grads so the log stays honest.
        gnorm_bottom = None
        if os.environ.get("TB_BOTTOM_ISOLATE", "0") == "1":
            gnorm, gnorm_bottom = clip_bottom_isolated(_main_named, cfg.grad_clip)
            _ramp = bottom_ramp_scale(getattr(cfg, "_bottom_ramp", None), step,
                                      float(gnorm), float(gnorm_bottom), cfg.grad_clip)
            if _ramp < 1.0:
                for _n, _p in _main_named:
                    if _p.grad is not None and not is_bottom_param(_n):
                        _p.grad.mul_(_ramp)
        elif cfg.grad_clip > 0.0:
            gnorm = torch.nn.utils.clip_grad_norm_(_main_params, cfg.grad_clip)
        else:
            gnorm = (torch.nn.utils.clip_grad_norm_(_main_params, float("inf"))
                     if any(p.grad is not None for p in _main_params)
                     else torch.tensor(0.0, device=DEVICE))

        if step < 1000 and step % max(1, int(os.environ.get('TB_PHASE_GRAD_DIAG_EVERY', '100'))) == 0:
            _gdiag = []
            for _g in groups:
                _sq = torch.zeros((), device=DEVICE, dtype=torch.float32)
                for _p in _g['params']:
                    if _p.grad is not None:
                        _sq = _sq + _p.grad.detach().float().square().sum()
                _gdiag.append(f"{_g.get('_label','?')}={float(_sq.sqrt()):.3g}")
            print("GRAD_GROUP stage=post_clip " + " ".join(_gdiag), flush=True)
        # BUG FIX: clip_grad_norm_ computes total_norm from the RAW grads, then
        # unconditionally rescales every grad by (max_norm / (total_norm + eps)).
        # If even ONE grad anywhere is nan/inf, total_norm is nan/inf, the clip
        # coefficient is nan, and EVERY parameter's grad (even ones that were
        # perfectly finite) gets multiplied by nan in-place. Previously opt.step()
        # ran unconditionally afterward, so a single transient non-finite step
        # (more likely to occur with RoPE — see note below) got baked forever
        # into AdamW's exp_avg/exp_avg_sq buffers: every step after that point
        # is permanently nan, which is exactly the "train=nan" from the first
        # bad step onward seen in the RoPE run. The draft optimizer already
        # guards against this (`if torch.isfinite(dloss)`); the main optimizer
        # never did. Mirror that guard here instead of trusting clip+step blindly.
        if torch.isfinite(gnorm) and (gnorm_bottom is None or torch.isfinite(gnorm_bottom)):
            opt.step()
            # Polyak shadow follows accepted updates only (a skipped nan step
            # leaves weights unchanged, so there is nothing to average in).
            # Lazily initialized here so resume/compilation paths all settle
            # first; NOT checkpointed (each resume re-enters burn-in).
            if getattr(cfg, "ema_eval", False):
                if getattr(model, "_ema_state", None) is None:
                    model._ema_state = init_ema_weights(model)
                    print(f"EMA shadow initialized: {len(model._ema_state)} "
                          "tensors (eval-only diagnostic)", flush=True)
                update_ema_weights(model._ema_state, model,
                                   float(cfg.ema_decay))
        else:
            if not getattr(cfg, "_nan_diag_done", False):
                cfg._nan_diag_done = True
                bad = [n for n, p in model.named_parameters()
                       if not n.startswith("draft.") and p.grad is not None
                       and not torch.isfinite(p.grad).all()]
                print(f"step {step}: non-finite grad norm ({gnorm.item()}) — skipping optimizer "
                      f"step. First offending param(s): {bad[:5]}", flush=True)
            else:
                print(f"step {step}: non-finite grad norm ({gnorm.item()}) — skipping optimizer step", flush=True)
        if cfg.fp4:
            for m in model.modules():
                if isinstance(m, QLinear) and m.quant != "none":
                    m._refresh_q()
        if gmem is not None and gmem_stats is not None:
            gmem_stats["module_gnorm"] = gmem_apply(gmem, gmem_opt)
        t_opt += time.time() - t0
        t0 = time.time()
        opt.zero_grad(set_to_none=True)
        t_opt += time.time() - t0
        step += 1
        metric_every = max(1, int(os.environ.get("TB_METRIC_EVERY", str(cfg.log_every))))
        if step % metric_every == 0:
            metric_train = float(acc_loss) if "acc_loss" in locals() else float(lval)
            _gb = f" gnorm_bottom={gnorm_bottom.item():.6f}" if gnorm_bottom is not None else ""
            print(f"METRIC step={step} train={metric_train:.6f} gnorm={gnorm.item():.6f} "
                  f"lr={base_lr:.8g} cum_toks={cum_toks}{_gb}", flush=True)
            if gmem_stats is not None:
                print(f"GMEM step={step} " + " ".join(
                    f"{k}={v:.4f}" for k, v in sorted(gmem_stats.items()))
                    + " " + gmem.describe(), flush=True)
        if cfg.profile and step % 100 == 0:
            elapsed = time.time() - prof_t0
            _accounted = t_data + t_model + t_bwd + t_opt
            _prof_n = max(1, step - prof_step)
            print(f"PROF step={step} data={t_data:.1f}s model={t_model:.1f}s "
                  f"bwd={t_bwd:.1f}s opt={t_opt:.1f}s "
                  f"rest={elapsed - _accounted:.1f}s "
                  f"| total={elapsed:.1f}s ({elapsed/_prof_n:.2f}s/step)", flush=True)
            prof_t0 = time.time()
            prof_step = step
            t_data = t_model = t_bwd = t_opt = 0.0
        if step % 100 == 0 and torch.cuda.is_available():
            dt = time.time() - getattr(cfg, "_mem_t0", time.time())
            _mem_n = max(1, step - getattr(cfg, "_mem_step", step - 100))
            cfg._mem_t0 = time.time()
            cfg._mem_step = step
            print(f"MEM step={step} alloc={torch.cuda.memory_allocated()/2**20:.0f}MB "
                  f"resv={torch.cuda.memory_reserved()/2**20:.0f}MB | {_mem_allocator_breakdown()} "
                  f"| lib: {_lib_workspace_stats()} | {dt/_mem_n:.2f}s/step", flush=True)
        if step % cfg.trend_every == 0 and step % cfg.log_every != 0:
            _tr_gate = float(getattr(target, 'traj_reg_gate_mean', torch.tensor(0.0)).detach().float().cpu())
            _tr_nrmse = float(getattr(target, 'traj_reg_pred_nrmse', torch.tensor(0.0)).detach().float().cpu())
            _lex_gate = float(getattr(target, 'lexical_semantic_gate', torch.tensor(0.0)).detach().float().cpu())
            _lex_aux = float(getattr(target, 'lexical_semantic_loss', torch.tensor(0.0)).detach().float().cpu())
            _phase_pred_aux = float(getattr(target, 'walk_pred_total', torch.tensor(0.0)).detach().float().cpu())
            _cycle_pressure = float(getattr(target, 'walk_cycle_pressure', torch.tensor(0.0)).detach().float().cpu())
            _cycle_strength = float(getattr(target, 'walk_cycle_strength', torch.tensor(0.0)).detach().float().cpu())
            _cycle_escape = float(getattr(target, 'walk_cycle_escape_strength', torch.tensor(0.0)).detach().float().cpu())
            # Scalar branch gates across blocks (mean/RMS/max-abs). The lex raw
            # gate separates a learned gate shutdown (raw -> 0) from a route
            # shutdown (raw open, gate_mean -> 0); walk/phase means expose
            # silencing or saturation the aux losses cannot show.
            _branch_gates = _branch_gate_summary(target)
            _walk_gate_txt = _format_gate_triplet(_branch_gates["walk_gamma"])
            _phase_gate_txt = _format_gate_triplet(_branch_gates["phase_gate"])
            # Novel walk pathways (all zero-init -> "-" until trained): geometry
            # (arc-length + Frenet-Serret) gains, place/self-localization gains,
            # and the selective-memory gate.
            _geom_gate_txt = _format_gate_triplet(_branch_gates["geom_gain"])
            _place_gate_txt = _format_gate_triplet(_branch_gates["place_gain"])
            _semem_gate_txt = _format_gate_triplet(_branch_gates["selmem_gate"])
            _hurst_gate_txt = _format_gate_triplet(_branch_gates["hurst_gain"])
            _affine_gate_txt = _format_gate_triplet(_branch_gates["affine_gain"])
            _motif_gate_txt = _format_gate_triplet(_branch_gates["motif_gain"])
            # Narrow-R visibility: raw pass_gain triplet (all-zero = provably
            # inert branch) plus the window's mean sampled extra passes over
            # TRAINING microbatches (flat microbatches count as 0). Missing
            # adapter prints -/-/- and -; counters reset every trend print.
            _r_gate_txt = _format_gate_triplet(_branch_gates["r_gain"])
            _r_sum = float(getattr(target, "_r_extra_sum", 0.0) or 0.0)
            _r_n = int(getattr(target, "_r_extra_count", 0) or 0)
            _r_extra_txt = f"{_r_sum / _r_n:.2f}/{_r_n}" if _r_n > 0 else "-"
            # Refinement-residual window: mean relative residual, halt count,
            # and per-position means (the contraction curve: successive ratios
            # estimate rho). Training-only; "-" when resid logging is off.
            _r_resid_sum = float(getattr(target, "_r_resid_sum", 0.0) or 0.0)
            _r_resid_n = int(getattr(target, "_r_resid_count", 0) or 0)
            _r_halt_sum = float(getattr(target, "_r_halt_sum", 0.0) or 0.0)
            _r_pos = list(getattr(target, "_r_resid_pos", []) or [])
            _r_pos_n = list(getattr(target, "_r_resid_pos_n", []) or [])
            _r_pos_txt = " ".join(
                f"{s / n:.1e}" if n > 0 else "-"
                for s, n in zip(_r_pos, _r_pos_n)) or "-"
            _r_resid_txt = (f"{_r_resid_sum / _r_resid_n:.2e}/h{_r_halt_sum:.0f} "
                            f"[{_r_pos_txt}]") if _r_resid_n > 0 else "-"
            # Structured line for the panel (MetricParser -> contraction-curve
            # plot). The trend text above is human-facing; this is machine-fed.
            if _r_resid_n > 0:
                _r_pos_csv = ",".join(
                    f"{s / n:.6e}" if n > 0 else ""
                    for s, n in zip(_r_pos, _r_pos_n))
                print(f"R_RESID step={step} mean={_r_resid_sum / _r_resid_n:.6e} "
                      f"halts={_r_halt_sum:.0f} n={_r_resid_n} "
                      f"pos={_r_pos_csv}", flush=True)
            for _attr, _zero in (("_r_extra_sum", 0.0), ("_r_extra_count", 0),
                                 ("_r_resid_sum", 0.0), ("_r_resid_count", 0),
                                 ("_r_halt_sum", 0.0)):
                if hasattr(target, _attr):
                    setattr(target, _attr, _zero)
            if hasattr(target, "_r_resid_pos"):
                target._r_resid_pos = []
            if hasattr(target, "_r_resid_pos_n"):
                target._r_resid_pos_n = []
            _lex_raw = _branch_gates["lex_raw"]
            _lex_raw_txt = f"{_lex_raw:+.3f}" if math.isfinite(_lex_raw) else "-"
            _local_gate_vals = [[], [], []]
            _local_effective_gate_vals = [[], [], []]
            _local_depthwise_deltas = [[], [], []]
            for _blk in getattr(target, 'blocks', ()):
                _stages = ([getattr(_blk, 'local_conv', None)]
                           + list(getattr(_blk, 'local_conv_extra', ())))
                for _si, _stage in enumerate(_stages[:3]):
                    if _stage is not None and hasattr(_stage, 'residual_gate'):
                        _learned_gate = torch.tanh(_stage.residual_gate).detach().float()
                        _local_gate_vals[_si].append(_learned_gate)
                        # `local_g` historically reported only the trainable
                        # gate.  A fixed bootstrap therefore made an apparent
                        # near-zero gate ambiguous: +.01 with bootstrap=-.05
                        # is an effective -.04 residual, not a disabled CNN.
                        _local_effective_gate_vals[_si].append(
                            (_learned_gate + float(getattr(_stage, 'gate_bootstrap', 0.0))).clamp(-0.95, 0.95))
                        for _conv in getattr(_stage, 'depthwise', ()):
                            _delta = _conv.weight.detach().float().clone()
                            _delta[:, 0, -1].sub_(1.0)  # identity causal tap at initialization
                            _local_depthwise_deltas[_si].append(
                                _delta.square().mean().sqrt())
            # Signed means alone can hide useful local gates that cancel across
            # layers. Log mean/RMS/max-absolute to separate true collapse from
            # a balanced positive/negative mixture after a resumed CNN branch.
            _local_gates = []
            _local_effective_gates = []
            _local_dw_drift = []
            for _vals, _eff_vals, _dw_vals in zip(
                    _local_gate_vals, _local_effective_gate_vals, _local_depthwise_deltas):
                if _vals:
                    _gate_stack = torch.stack(_vals).float()
                    _local_gates.append((float(_gate_stack.mean().cpu()),
                                         float(_gate_stack.square().mean().sqrt().cpu()),
                                         float(_gate_stack.abs().max().cpu())))
                    _eff_stack = torch.stack(_eff_vals).float()
                    _local_effective_gates.append((float(_eff_stack.mean().cpu()),
                                                    float(_eff_stack.square().mean().sqrt().cpu()),
                                                    float(_eff_stack.abs().max().cpu())))
                    _local_dw_drift.append(float(torch.stack(_dw_vals).mean().cpu()) if _dw_vals else 0.0)
                else:
                    _local_gates.append((0.0, 0.0, 0.0))
                    _local_effective_gates.append((0.0, 0.0, 0.0))
                    _local_dw_drift.append(0.0)
            print(f"step {step}/{cfg.steps} train={acc_loss:.4f} gnorm={gnorm.item():.4f} "
                  f"lr={base_lr:.3e} raw_lr={raw_lr:.3e} phase_lr={base_lr * cfg.phase_mult:.3e} "
                  f"toks/step={acc_toks} cum_toks={cum_toks:,} "
                  f"T={T} cot={share:.2f} code={code_share:.2f} "
                   f"trgate={_tr_gate:.3f} trnrmse={_tr_nrmse:.3f} "
                   f"phase_pred_aux={_phase_pred_aux:.5f} lex_g={_lex_gate:.5f} lex_raw={_lex_raw_txt} lex_aux={_lex_aux:.5f} "
                    f"walk_g=m/r/x {_walk_gate_txt} phase_g=m/r/x {_phase_gate_txt} "
                     f"r_g=m/r/x {_r_gate_txt} r_extra=mean/n {_r_extra_txt} "
                      f"geom_g=m/r/x {_geom_gate_txt} place_g=m/r/x {_place_gate_txt} "
                      f"semem_g=m/r/x {_semem_gate_txt} hurst_g=m/r/x {_hurst_gate_txt} "
                      f"affine_g=m/r/x {_affine_gate_txt} motif_g=m/r/x {_motif_gate_txt} "
                     f"r_resid=mean/halts[pos] {_r_resid_txt} "
                  f"cycle=p/s/e {_cycle_pressure:.4f}/{_cycle_strength:.4f}/{_cycle_escape:.4f} "
                  f"local_g=m/r/x "
                  f"{_local_gates[0][0]:.3f}/{_local_gates[0][1]:.3f}/{_local_gates[0][2]:.3f} "
                  f"| {_local_gates[1][0]:.3f}/{_local_gates[1][1]:.3f}/{_local_gates[1][2]:.3f} "
                  f"| {_local_gates[2][0]:.3f}/{_local_gates[2][1]:.3f}/{_local_gates[2][2]:.3f} "
                  f"local_eff=m/r/x "
                  f"{_local_effective_gates[0][0]:.3f}/{_local_effective_gates[0][1]:.3f}/{_local_effective_gates[0][2]:.3f} "
                  f"| {_local_effective_gates[1][0]:.3f}/{_local_effective_gates[1][1]:.3f}/{_local_effective_gates[1][2]:.3f} "
                  f"| {_local_effective_gates[2][0]:.3f}/{_local_effective_gates[2][1]:.3f}/{_local_effective_gates[2][2]:.3f} "
                  f"local_dw_drift={_local_dw_drift[0]:.4f}/{_local_dw_drift[1]:.4f}/{_local_dw_drift[2]:.4f} "
                  f"[{time.time()-t0:.0f}s]", flush=True)
        if step % cfg.log_every == 0 or step == cfg.steps:
            # EMA diagnostic (TB_EMA_EVAL=1): score the Polyak-averaged weights
            # FIRST so the official live numbers below persist in model attrs.
            # Pure observation: best_val/ckpt selection always uses live vl.
            # RNG streams are restored so eval cadence never perturbs data.
            ema_vl = float("nan")
            if getattr(cfg, "ema_eval", False) and getattr(model, "_ema_state", None):
                _ema_py = random.getstate()
                _ema_cpu = torch.random.get_rng_state()
                try:
                    with _eval_mode(model), ema_eval_weights(model, model._ema_state):
                        ema_vl = _evaluate_inner(
                            model, dm, cfg, cfg.n_eval_batches)[0]
                finally:
                    random.setstate(_ema_py)
                    torch.random.set_rng_state(_ema_cpu)
            vl, a1, a5, tl, ctl, fwl = evaluate(model, dm, cfg, cfg.n_eval_batches)
            source_val_metrics = collections.OrderedDict(
                (name, dict(metrics)) for name, metrics in dm.last_source_val_metrics.items())
            val_summary = dict(dm.last_val_summary)
            code_v = evaluate_code(model, dm, cfg, cfg.n_eval_batches)
            primary_metric = vl
            mix_v = ((1.0 - code_share) * primary_metric + code_share * code_v) if code_v == code_v else primary_metric
            select_v = mix_v if cfg.code_best_metric == 'mix' else primary_metric
            walk_off = (evaluate_without_walk(model, dm, cfg, cfg.n_eval_batches)
                        if cfg.walk_ablate_eval else None)
            phase_ab = (evaluate_phase_ablations(model, dm, cfg, cfg.phase_ablate_batches)
                        if cfg.phase_ablate_eval else {})
            extension_ab = (evaluate_extension_ablations(
                model, dm, cfg, cfg.phase_ablate_batches)
                if cfg.v41_ablate_eval else {})
            _v2 = None
            if cfg.eval_t2 and cfg.eval_t2 != cfg.eval_t:
                _c = Cfg.__new__(Cfg)
                _c.eval_t = cfg.eval_t2
                _long_batches = max(1, math.ceil(cfg.n_eval_batches * cfg.eval_t / cfg.eval_t2))
                _v2 = evaluate(model, dm, _c, _long_batches, include_train_loss=False)[0]
            if select_v < best_val - 1e-5:
                best_val = select_v
                ck.save_best(model, opt, step, select_v, run_state_data=run_state(cfg))
                if hf_up:
                    hf_up.offer(ck.best, "ckpt_best.pt")
            if step % cfg.ckpt_every == 0 or step == cfg.steps:
                stream_state_data = dm.checkpoint_streams()
                ck.save_ongoing(model, opt, step, capture_rng_state(), best_val,
                                doremi_state if cfg.doremi else None, len(dm.general),
                                dm.stream.raw_cursor if isinstance(dm.stream, (StreamSource, MixedStreamSource)) else None,
                                draft_opt=draft_opt, cum_toks=cum_toks, schedule_toks=schedule_toks,
                                data_step_base=data_step_base, run_state_data=run_state(cfg),
                                stream_state_data=stream_state_data)
                if hf_up:
                    hf_up.offer(ck.ongoing, "ckpt_ongoing.pt")
                    if os.path.exists(ck.metadata):
                        hf_up.offer(ck.metadata, "ckpt_ongoing.pt.metadata.json")
                    snap = ck.save_snapshot(model, step, cum_toks)
                    stem = os.path.splitext(os.path.basename(cfg.ckpt_path))[0]
                    hf_up.offer(snap, f"{stem}_step{step}.pt", cleanup=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                alloc = torch.cuda.memory_allocated() / 2 ** 20
                vram = torch.cuda.memory_reserved() / 2 ** 20
            else:
                alloc = vram = 0
            _fw = f" web_edu_dedup_val={fwl:.4f}" if fwl == fwl else ""
            _code = (f" code_val={code_v:.4f} mix_val={mix_v:.4f} code_share={code_share:.3f}"
                     if code_v == code_v else "")
            _cot = f" cot_gap={ctl - vl:.4f}" if dm.cot is not None else ""
            _walk = (f" walk_off={walk_off:.4f} walk_delta={walk_off - vl:+.4f}"
                     if walk_off is not None else "")
            _cycle = (f" cycle_p={float(getattr(target, 'walk_cycle_pressure', torch.tensor(0.0))):.4f}"
                      f" cycle_s={float(getattr(target, 'walk_cycle_strength', torch.tensor(0.0))):.4f}"
                      f" cycle_e={float(getattr(target, 'walk_cycle_escape_strength', torch.tensor(0.0))):.4f}")
            _phase_ref = phase_ab.get('matched_full', vl)
            _phase = (f" phase_ablate_ref={_phase_ref:.4f}" + ''.join(
                f' {k}={v:.4f} {k}_delta={v-_phase_ref:+.4f}'
                for k, v in phase_ab.items() if k != 'matched_full')) if phase_ab else ''
            _extension_ref = extension_ab.get('matched_full', vl)
            _extensions = (f" extension_ablate_ref={_extension_ref:.4f}" + ''.join(
                f' {k}={v:.4f} {k}_delta={v-_extension_ref:+.4f}'
                for k, v in extension_ab.items() if k != 'matched_full')) if extension_ab else ''
            print(f"step {step}/{cfg.steps} T={T} cot={share:.2f} val={vl:.4f} top1={a1:.4f} "
                  f"top5={a5:.4f} toks/step={acc_toks} cum_toks={cum_toks:,} gnorm={gnorm.item():.4f} "
                  f"lr={base_lr:.3e} raw_lr={raw_lr:.3e} phase_lr={base_lr * cfg.phase_mult:.3e} "
                   f"vram={vram:.0f}MB alloc={alloc:.0f}MB train={tl:.4f} gap={tl - vl:.4f}"
                   + (f" ema_vl={ema_vl:.4f}" if ema_vl == ema_vl else "")
                  + _cot
                  + _fw
                  + _code
                  + _walk
                  + _cycle
                  + _phase
                  + _extensions
                  + (f" val@{cfg.eval_t2}={_v2:.4f}" if _v2 is not None else ""), flush=True)
            if source_val_metrics:
                for name, metrics in source_val_metrics.items():
                    print(f"VAL_SOURCE step={step} name={name} weight={metrics['weight']:.6f} "
                          f"ce={metrics['ce']:.6f} top1={metrics['top1']:.6f} "
                          f"top5={metrics['top5']:.6f} batches={metrics['batches']} "
                          f"tokens={metrics['tokens']}" + _source_difficulty_suffix(metrics), flush=True)
                print(f"VAL_MIX step={step} weighted_ce={val_summary['weighted_ce']:.6f} "
                      f"macro_ce={val_summary['macro_ce']:.6f} "
                      f"worst_ce={val_summary['worst_ce']:.6f} "
                      f"worst_source={val_summary['worst_source']}", flush=True)
            if gmem is not None:
                with gmem_capture(target, gmem) as _cap, _eval_mode(model):
                    gmem_evaluate(GmemContext(target, tok, gmem, _cap), gmem_data, step)
            for family, results in (("phase", phase_ab), ("extension", extension_ab)):
                matched = results.get('matched_full') if results else None
                if matched is None:
                    continue
                for name, ablated in results.items():
                    if name == 'matched_full':
                        continue
                    print(f"ABLATION step={step} family={family} name={name} "
                          f"full_ce={matched:.6f} ablated_ce={ablated:.6f} "
                          f"delta={ablated-matched:+.6f}", flush=True)
            if isinstance(dm.stream, MixedStreamSource):
                parts = []
                for item in dm.stream.health_summary():
                    sampled = item['sampled_batches']
                    rejected = item['rejected_rows']
                    raw_rows = item['raw_rows']
                    reason = (':' + ','.join(f"{k}={v}" for k, v in sorted(item['reasons'].items()))) \
                             if item['reasons'] else ''
                    parts.append(f"{item['name']}[batch={sampled},raw={raw_rows},rej={rejected}{reason}]")
                print('STREAM_MIX_HEALTH ' + ' '.join(parts), flush=True)
            t0 = time.time()
    # The terminal optimizer step is always evaluated by the block above.
    # Reuse it instead of paying for the full nine-source suite and all causal
    # ablations twice at the same weights.
    final_source_val_metrics = source_val_metrics
    final_val_summary = val_summary
    final_code_share = code_share_at(schedule_toks, cfg)
    primary_metric = vl
    mix_v = ((1.0 - final_code_share) * primary_metric + final_code_share * code_v) if code_v == code_v else primary_metric
    final_walk_off = walk_off
    final_phase_ab = phase_ab
    final_extension_ab = extension_ab
    final_walk = (f" walk_off={final_walk_off:.4f} walk_delta={final_walk_off - vl:+.4f}"
                  if final_walk_off is not None else "")
    final_phase_ref = final_phase_ab.get('matched_full', vl)
    final_phase = (f" phase_ablate_ref={final_phase_ref:.4f}" + ''.join(
        f' {k}={v:.4f} {k}_delta={v-final_phase_ref:+.4f}'
        for k, v in final_phase_ab.items() if k != 'matched_full')) if final_phase_ab else ''
    final_extension_ref = final_extension_ab.get('matched_full', vl)
    final_extensions = (f" extension_ablate_ref={final_extension_ref:.4f}" + ''.join(
        f' {k}={v:.4f} {k}_delta={v-final_extension_ref:+.4f}'
        for k, v in final_extension_ab.items() if k != 'matched_full')) if final_extension_ab else ''
    print(f"FINAL val={vl:.4f} top1={a1:.4f} top5={a5:.4f} train={tl:.4f} cum_toks={cum_toks:,} "
          f"cot_gap={ctl - vl:.4f}"
          + (f" web_edu_dedup_val={fwl:.4f}" if fwl == fwl else "")
          + (f" code_val={code_v:.4f} mix_val={mix_v:.4f} code_share={final_code_share:.3f}" if code_v == code_v else "")
          + final_walk + final_phase + final_extensions, flush=True)
    if final_source_val_metrics:
        for name, metrics in final_source_val_metrics.items():
            print(f"FINAL_VAL_SOURCE step={step} name={name} weight={metrics['weight']:.6f} "
                  f"ce={metrics['ce']:.6f} top1={metrics['top1']:.6f} "
                  f"top5={metrics['top5']:.6f} batches={metrics['batches']} "
                  f"tokens={metrics['tokens']}" + _source_difficulty_suffix(metrics), flush=True)
        print(f"FINAL_VAL_MIX step={step} weighted_ce={final_val_summary['weighted_ce']:.6f} "
              f"macro_ce={final_val_summary['macro_ce']:.6f} "
              f"worst_ce={final_val_summary['worst_ce']:.6f} "
              f"worst_source={final_val_summary['worst_source']}", flush=True)
    for family, results in (("phase", final_phase_ab), ("extension", final_extension_ab)):
        matched = results.get('matched_full') if results else None
        if matched is None:
            continue
        for name, ablated in results.items():
            if name == 'matched_full':
                continue
            print(f"FINAL_ABLATION step={step} family={family} name={name} "
                  f"full_ce={matched:.6f} ablated_ce={ablated:.6f} "
                  f"delta={ablated-matched:+.6f}", flush=True)
    if hf_up:
        hf_up.stop()
    return model


# ------------------------------------------------------------------ config -

class Cfg:
    pass


def tokenize_dataset(hf_name, out_dir, max_tokens, shard_tokens=100_000_000):
    """Streaming FineWeb-class dataset -> gpt2-BPE uint16 shards for DataMix.
    Run once: python train_1b.py --tokenize <hf_name[:subset]> <out_dir> <max_tokens>"""
    import numpy as np
    from datasets import load_dataset
    tok = _load_tokenizer("gpt2")
    if ":" in hf_name:
        repo, subset = hf_name.split(":", 1)
        ds = load_dataset(repo, subset, split="train", streaming=True)
    else:
        ds = load_dataset(hf_name, split="train", streaming=True)
    os.makedirs(out_dir, exist_ok=True)
    buf, shard, written, nrows = [], 0, 0, 0
    for row in ds:
        text = row.get("text", "")
        if not text:
            continue
        nrows += 1
        buf.extend(tok(text, add_special_tokens=False)["input_ids"])
        if nrows % 50000 == 0:
            print(f"rows={nrows:,} buf_tokens={len(buf):,} shards={shard} written={written:,}", flush=True)
        while len(buf) >= shard_tokens:
            np.array(buf[:shard_tokens], dtype="uint16").tofile(
                os.path.join(out_dir, f"shard_{shard:05d}.bin"))
            buf = buf[shard_tokens:]
            shard += 1
            written += shard_tokens
            print(f"written {written:,} tokens / {shard} shards", flush=True)
            if written >= max_tokens:
                return
    if buf:
        np.array(buf, dtype="uint16").tofile(
            os.path.join(out_dir, f"shard_{shard:05d}.bin"))
        print(f"final shard {shard}: {len(buf):,} tokens", flush=True)


def _is_infer_request(argv):
    # minimal-arg trigger: any explicit infer flag OR a bare .pt/.bin path
    flags = {"--infer", "--weights", "--ckpt", "--checkpoint", "--weight"}
    if any(a in flags for a in argv):
        return True
    # positional .pt fallback (e.g. `python train_1b (3).py ckpt.pt`)
    for a in argv[1:]:
        if a.endswith((".pt", ".bin", ".pth")) and os.path.exists(a):
            return True
        # also treat bare arg that looks like a checkpoint path even if not yet exists -> let loader error clearly
        if a.endswith((".pt", ".pth")) and not a.startswith("-"):
            return True
    return False


class NIUtilityHead(nn.Module):
    """Read-only utility probes for a shared hidden representation.

    The head deliberately has no connection back into tokens or NI routing.  It
    supplies gradients through the same pooled backbone state used by CE, so an
    alternate view must earn its effect through the normal model computation.
    Output 0 is a local-order fluency proxy and output 1 is paired preference.
    """

    def __init__(self, D):
        super().__init__()
        self.norm = nn.LayerNorm(D, dtype=DTYPE)
        self.out = nn.Linear(D, 2, dtype=DTYPE)

    def forward(self, features):
        return self.out(self.norm(features))


class NIFutureHead(nn.Module):
    """Project a causal prefix state toward its real future continuation state."""

    def __init__(self, D):
        super().__init__()
        self.norm = nn.LayerNorm(D, dtype=DTYPE)
        self.out = nn.Linear(D, D, bias=False, dtype=DTYPE)

    def forward(self, features):
        return F.normalize(self.out(self.norm(features)).float(), dim=-1)


def _ni_aux_context(row):
    """Make the evidence prefix for one accepted/rejected instruction record."""
    context = row.get("context") or {}
    topic = str(context.get("topic") or "")
    ancestors = context.get("ancestors") or []
    history = "\n".join(str(item) for item in ancestors[-3:])
    return f"Topic: {topic}\nHistory: {history}\nResponse: "


def _ni_aux_corrupt(candidate):
    """Deterministically disturb local byte order; this is a fluency proxy only."""
    data = bytearray(candidate.encode("utf-8", errors="replace"))
    for index in range(4, max(4, len(data) - 1), 7):
        if data[index:index + 1].isalnum() and data[index + 1:index + 2].isalnum():
            data[index], data[index + 1] = data[index + 1], data[index]
    if bytes(data) == candidate.encode("utf-8", errors="replace"):
        data.extend(b" ?")
    return bytes(data)


def _ni_aux_pack(prefix, candidate, T):
    """Byte-tokenize a prompt/candidate pair and mask CE outside the candidate."""
    if T < 8:
        raise ValueError("NI auxiliary sequence length must be at least 8")
    prompt = prefix.encode("utf-8", errors="replace")
    answer = candidate if isinstance(candidate, bytes) else candidate.encode("utf-8", errors="replace")
    prompt_budget = max(1, T // 2)
    answer_budget = max(2, T - prompt_budget)
    raw = prompt[-prompt_budget:] + answer[:answer_budget]
    answer_start = min(len(prompt[-prompt_budget:]), len(raw) - 1)
    if len(raw) < 2:
        raw = b" ?"
        answer_start = 1
    ids = torch.tensor(list(raw[:T + 1]), dtype=torch.long)
    inputs, targets = ids[:-1], ids[1:].clone()
    candidate_target_start = max(0, answer_start - 1)
    targets[:candidate_target_start] = -100
    valid = torch.ones(inputs.numel(), dtype=torch.bool)
    if inputs.numel() < T:
        pad = T - inputs.numel()
        inputs = F.pad(inputs, (0, pad), value=32)
        targets = F.pad(targets, (0, pad), value=-100)
        valid = F.pad(valid, (0, pad), value=False)
    return inputs[:T], targets[:T], valid[:T]


def _ni_aux_load_records(path):
    """Load only records with a genuine accepted/rejected supervision pair."""
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            accepted, rejected = row.get("accepted"), row.get("rejected")
            if isinstance(accepted, str) and isinstance(rejected, str) and accepted and rejected:
                records.append((_ni_aux_context(row), accepted, rejected))
    if len(records) < 20:
        raise ValueError(f"need at least 20 accepted/rejected rows, found {len(records)} in {path}")
    return records


def _ni_aux_batch(records, batch_size, T, device, rng):
    """Build CE, paired-preference, and local-order examples from one data view."""
    accepted, rejected, corrupted = [], [], []
    for _ in range(batch_size):
        prefix, good, bad = records[rng.randrange(len(records))]
        accepted.append(_ni_aux_pack(prefix, good, T))
        rejected.append(_ni_aux_pack(prefix, bad, T))
        corrupted.append(_ni_aux_pack(prefix, _ni_aux_corrupt(good), T))

    def stack(items, field):
        return torch.stack([item[field] for item in items]).to(device)

    return {
        "accepted_ids": stack(accepted, 0), "accepted_targets": stack(accepted, 1),
        "accepted_mask": stack(accepted, 2), "rejected_ids": stack(rejected, 0),
        "rejected_mask": stack(rejected, 2), "corrupt_ids": stack(corrupted, 0),
        "corrupt_mask": stack(corrupted, 2),
    }


def _ni_self_views(ids, valid):
    """Split each accepted sequence into causal prefix and later continuation."""
    prefix_ids = torch.full_like(ids, 32)
    future_ids = torch.full_like(ids, 32)
    prefix_mask = torch.zeros_like(valid)
    future_mask = torch.zeros_like(valid)
    for row in range(ids.size(0)):
        length = int(valid[row].sum())
        split = min(length - 1, max(1, length // 2))
        prefix_ids[row, :split] = ids[row, :split]
        future_length = length - split
        future_ids[row, :future_length] = ids[row, split:length]
        prefix_mask[row, :split] = True
        future_mask[row, :future_length] = True
    return prefix_ids, prefix_mask, future_ids, future_mask


def _ni_aux_metrics(model, utility, records, device, batches=20, batch_size=8, T=48):
    """Evaluate CE and the two label-grounded utilities on held-out records."""
    model.eval()
    utility.eval()
    rng = random.Random(10_003)
    totals = collections.defaultdict(float)
    with torch.no_grad():
        for _ in range(batches):
            batch = _ni_aux_batch(records, batch_size, T, device, rng)
            _, ce, good = model(batch["accepted_ids"], batch["accepted_targets"], True, batch["accepted_mask"])
            _, _, bad = model(batch["rejected_ids"], return_features=True, feature_mask=batch["rejected_mask"])
            _, _, corrupt = model(batch["corrupt_ids"], return_features=True, feature_mask=batch["corrupt_mask"])
            good_scores, bad_scores, corrupt_scores = utility(good).float(), utility(bad).float(), utility(corrupt).float()
            totals["ce"] += float(ce)
            totals["preference"] += float((good_scores[:, 1] > bad_scores[:, 1]).float().mean())
            totals["fluency"] += float((good_scores[:, 0] > corrupt_scores[:, 0]).float().mean())
    return {key: value / batches for key, value in totals.items()}


def _run_ni_aux_condition(train_records, val_records, use_view, steps, device):
    """Train one tiny full-LM control with the same auxiliary heads and data."""
    torch.manual_seed(71)
    # This intentionally stays tiny: it is a directional objective check, not
    # a proxy for the 1B model's final loss or context capacity.
    model = WalkGPT(256, 32, 1, 4, mult=2.0, no_walk=True, max_t=48, rope=True,
                    ni=use_view, ni_mode="view", ni_layer=0, ni_chunk_size=16,
                    ni_steps=1, ni_trajectory=True, ni_slots=2, ni_rank=8).to(device)
    utility = NIUtilityHead(32).to(device)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(utility.parameters()), lr=3e-4, weight_decay=0.01)
    rng = random.Random(71)
    model.train()
    utility.train()
    for _ in range(steps):
        batch = _ni_aux_batch(train_records, batch_size=8, T=48, device=device, rng=rng)
        _, ce, good = model(batch["accepted_ids"], batch["accepted_targets"], True, batch["accepted_mask"])
        _, _, bad = model(batch["rejected_ids"], return_features=True, feature_mask=batch["rejected_mask"])
        _, _, corrupt = model(batch["corrupt_ids"], return_features=True, feature_mask=batch["corrupt_mask"])
        good_scores, bad_scores, corrupt_scores = utility(good).float(), utility(bad).float(), utility(corrupt).float()
        fluency = F.binary_cross_entropy_with_logits(
            torch.cat((good_scores[:, 0], corrupt_scores[:, 0])),
            torch.cat((torch.ones_like(good_scores[:, 0]), torch.zeros_like(corrupt_scores[:, 0]))),
        )
        preference = F.binary_cross_entropy_with_logits(
            torch.cat((good_scores[:, 1], bad_scores[:, 1])),
            torch.cat((torch.ones_like(good_scores[:, 1]), torch.zeros_like(bad_scores[:, 1]))),
        )
        loss = ce + 0.10 * (fluency + preference)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(utility.parameters()), 1.0)
        optimizer.step()
    return _ni_aux_metrics(model, utility, val_records, device)


def run_ni_aux_probe(path, steps=600):
    """Compare ordinary CE training to the one-path alternate-view NI design."""
    records = _ni_aux_load_records(path)
    split = max(1, int(len(records) * 0.8))
    train_records, val_records = records[:split], records[split:]
    device = torch.device(DEVICE)
    print(f"NI auxiliary probe: {len(train_records)} train / {len(val_records)} val records; {steps} steps each", flush=True)
    started = time.time()
    standard = _run_ni_aux_condition(train_records, val_records, False, steps, device)
    view = _run_ni_aux_condition(train_records, val_records, True, steps, device)
    elapsed = time.time() - started
    print(f"NI auxiliary standard: CE={standard['ce']:.4f} preference={standard['preference']:.3f} fluency_proxy={standard['fluency']:.3f}", flush=True)
    print(f"NI auxiliary view:     CE={view['ce']:.4f} preference={view['preference']:.3f} fluency_proxy={view['fluency']:.3f}", flush=True)
    print(f"NI auxiliary elapsed={elapsed:.1f}s; fluency_proxy is local-order discrimination, not a grammar benchmark", flush=True)
    result_path = os.environ.get("TB_NI_AUX_RESULT_PATH", "").strip()
    if result_path:
        result = {"steps": int(steps), "records": len(records), "standard": standard,
                  "view": view, "elapsed_seconds": elapsed}
        with open(result_path, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
        print(f"NI auxiliary result saved to {result_path}", flush=True)
    return standard, view


def _ni_self_metrics(model, predictor, records, device, batches=20, batch_size=8, T=48):
    """Held-out CE plus whether each prefix retrieves its own future state."""
    model.eval()
    predictor.eval()
    rng = random.Random(20_003)
    ce_total, retrieval_total = 0.0, 0.0
    with torch.no_grad():
        for _ in range(batches):
            batch = _ni_aux_batch(records, batch_size, T, device, rng)
            _, ce, _ = model(batch["accepted_ids"], batch["accepted_targets"], True, batch["accepted_mask"])
            prefix_ids, prefix_mask, future_ids, future_mask = _ni_self_views(batch["accepted_ids"], batch["accepted_mask"])
            _, _, prefix = model(prefix_ids, return_features=True, feature_mask=prefix_mask)
            _, _, future = model(future_ids, return_features=True, feature_mask=future_mask)
            scores = predictor(prefix) @ F.normalize(future.float(), dim=-1).transpose(0, 1) / 0.10
            ce_total += float(ce)
            retrieval_total += float((scores.argmax(dim=-1) == torch.arange(batch_size, device=device)).float().mean())
    return {"ce": ce_total / batches, "future_retrieval": retrieval_total / batches}


def _run_ni_self_condition(train_records, val_records, use_view, steps, device):
    """Accepted-only CE plus causal future-state prediction; no reject labels."""
    torch.manual_seed(91)
    model = WalkGPT(256, 32, 1, 4, mult=2.0, no_walk=True, max_t=48, rope=True,
                    ni=use_view, ni_mode="view", ni_layer=0, ni_chunk_size=16,
                    ni_steps=1, ni_trajectory=True, ni_slots=2, ni_rank=8).to(device)
    predictor = NIFutureHead(32).to(device)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(predictor.parameters()), lr=3e-4, weight_decay=0.01)
    rng = random.Random(91)
    model.train()
    predictor.train()
    for _ in range(steps):
        batch = _ni_aux_batch(train_records, batch_size=8, T=48, device=device, rng=rng)
        _, ce, _ = model(batch["accepted_ids"], batch["accepted_targets"], True, batch["accepted_mask"])
        prefix_ids, prefix_mask, future_ids, future_mask = _ni_self_views(batch["accepted_ids"], batch["accepted_mask"])
        _, _, prefix = model(prefix_ids, return_features=True, feature_mask=prefix_mask)
        with torch.no_grad():
            _, _, future = model(future_ids, return_features=True, feature_mask=future_mask)
        scores = predictor(prefix) @ F.normalize(future.float(), dim=-1).transpose(0, 1) / 0.10
        future_loss = F.cross_entropy(scores, torch.arange(scores.size(0), device=device))
        loss = ce + 0.10 * future_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(predictor.parameters()), 1.0)
        optimizer.step()
    return _ni_self_metrics(model, predictor, val_records, device)


def run_ni_self_probe(path, steps=600):
    """Compare standard and NI on accepted-only future-state prediction."""
    records = _ni_aux_load_records(path)
    split = max(1, int(len(records) * 0.8))
    train_records, val_records = records[:split], records[split:]
    device = torch.device(DEVICE)
    print(f"NI self probe: {len(train_records)} train / {len(val_records)} val accepted records; {steps} steps each", flush=True)
    started = time.time()
    standard = _run_ni_self_condition(train_records, val_records, False, steps, device)
    view = _run_ni_self_condition(train_records, val_records, True, steps, device)
    elapsed = time.time() - started
    print(f"NI self standard: CE={standard['ce']:.4f} future_retrieval={standard['future_retrieval']:.3f}", flush=True)
    print(f"NI self view:     CE={view['ce']:.4f} future_retrieval={view['future_retrieval']:.3f}", flush=True)
    print(f"NI self elapsed={elapsed:.1f}s; retrieval uses actual later text, not a correctness label", flush=True)
    result_path = os.environ.get("TB_NI_SELF_RESULT_PATH", "").strip()
    if result_path:
        with open(result_path, "w", encoding="utf-8") as handle:
            json.dump({"steps": int(steps), "records": len(records), "standard": standard,
                       "view": view, "elapsed_seconds": elapsed}, handle, indent=2, sort_keys=True)
        print(f"NI self result saved to {result_path}", flush=True)
    return standard, view


def _ni_evidence_batch(batch_size, T, D, device, seed):
    """Create a text-free two-stream binding example for NI routing tests.

    Image and audio each carry one independent bit in a pre-encoded feature.
    The only supervised token is their XOR, while the text prefix is identical
    for every sample.  This makes a passed test evidence of information flow
    and cross-stream combination, not of language or semantic capability.
    """
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    image_bit = torch.randint(0, 2, (batch_size,), generator=generator)
    audio_bit = torch.randint(0, 2, (batch_size,), generator=generator)
    inputs = torch.ones(batch_size, T, dtype=torch.long, device=device)
    targets = torch.full((batch_size, T), -100, dtype=torch.long, device=device)
    targets[:, -1] = 2 + (image_bit ^ audio_bit).to(device)

    def stream(bit, items):
        features = torch.randn(batch_size, items, D, generator=generator).mul_(0.02)
        features[:, 0, 0] = bit.float().mul(2).sub(1)
        return features.to(device=device, dtype=DTYPE)

    evidence = {
        "image": stream(image_bit, 3),
        "audio": stream(audio_bit, 4),
        "image_mask": torch.ones(batch_size, 3, dtype=torch.bool, device=device),
        "audio_mask": torch.ones(batch_size, 4, dtype=torch.bool, device=device),
    }
    return inputs, targets, evidence


@torch.no_grad()
def _ni_evidence_accuracy(model, device, condition, batches=24, batch_size=64):
    """Measure final XOR-token accuracy with both or ablated evidence streams."""
    model.eval()
    correct = total = 0
    for batch_index in range(batches):
        inputs, targets, evidence = _ni_evidence_batch(
            batch_size, 8, model.emb.embedding_dim, device, 90_000 + batch_index,
        )
        if condition == "none":
            used_evidence = None
        else:
            used_evidence = dict(evidence)
            if condition == "image_only":
                used_evidence["audio_mask"] = torch.zeros_like(evidence["audio_mask"])
            elif condition == "audio_only":
                used_evidence["image_mask"] = torch.zeros_like(evidence["image_mask"])
            elif condition != "both":
                raise ValueError(f"unknown evidence condition {condition!r}")
        logits, _ = model(inputs, evidence=used_evidence)
        correct += int((logits[:, -1].argmax(dim=-1) == targets[:, -1]).sum())
        total += batch_size
    return correct / total


def _run_ni_evidence_condition(use_evidence, steps, device, width_mult=4, bridge=False):
    """Train one matched tiny backbone for the evidence-routing smoke test.

    ``bridge`` changes only the training-time local decoder-gradient teacher;
    its inference graph is identical to the ordinary evidence condition.
    """
    torch.manual_seed(314)
    model = WalkGPT(
        8, 32, 4, 4, mult=2.0, no_walk=True, max_t=8, rope=True,
        ni=use_evidence, ni_mode="evidence", ni_layers="0,1,2,3",
        ni_steps=1, ni_trajectory=True, ni_rank=8, ni_width_mult=width_mult,
        ni_bridge=bool(use_evidence and bridge), ni_bridge_weight=0.02,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.0)
    model.train()
    for step in range(steps):
        inputs, targets, evidence = _ni_evidence_batch(64, 8, 32, device, 30_000 + step)
        _, loss = model(inputs, targets, evidence=evidence if use_evidence else None)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    metrics = {
        "both": _ni_evidence_accuracy(model, device, "both" if use_evidence else "none"),
        "image_only": _ni_evidence_accuracy(model, device, "image_only" if use_evidence else "none"),
        "audio_only": _ni_evidence_accuracy(model, device, "audio_only" if use_evidence else "none"),
    }
    return metrics, model


def _ni_probe_width():
    """Keep probe width explicit so capacity comparisons are reproducible."""
    return max(1, int(os.environ.get("TB_NI_PROBE_WIDTH", "4")))


def run_ni_evidence_probe(steps=600):
    """Verify that stateless NI can bind two static feature streams via CE alone."""
    if steps <= 0:
        raise ValueError("NI evidence probe steps must be positive")
    device = torch.device(DEVICE)
    width_mult = _ni_probe_width()
    use_bridge = os.environ.get("TB_NI_PROBE_BRIDGE", "0") == "1"
    print(f"NI evidence probe: width={width_mult} XOR binding, {steps} optimizer steps per condition on {device}; "
          f"local_gradient_bridge={int(use_bridge)}", flush=True)
    started = time.time()
    standard, _ = _run_ni_evidence_condition(False, steps, device, width_mult=width_mult)
    evidence, _ = _run_ni_evidence_condition(True, steps, device, width_mult=width_mult)
    bridged = (_run_ni_evidence_condition(True, steps, device, width_mult=width_mult, bridge=True)[0]
               if use_bridge else None)
    elapsed = time.time() - started
    print("NI evidence standard: " + " ".join(f"{key}={value:.3f}" for key, value in standard.items()), flush=True)
    print("NI evidence workspace: " + " ".join(f"{key}={value:.3f}" for key, value in evidence.items()), flush=True)
    if bridged is not None:
        print("NI evidence + bridge: " + " ".join(f"{key}={value:.3f}" for key, value in bridged.items()), flush=True)
    print(f"NI evidence elapsed={elapsed:.1f}s; this validates routing/binding only, not language quality", flush=True)
    return standard, evidence, bridged


@torch.no_grad()
def _ni_evidence_trace_features(model, inputs, evidence):
    """Return final-token NI residual features without exposing an LM cache API."""
    logits, _ = model(inputs, evidence=evidence, capture_ni_trace=True)
    trace = model._last_ni_trace
    if not trace:
        raise RuntimeError("evidence trace requested from a model without evidence injection")
    features = torch.cat([item["residual"][:, -1].float() for item in trace], dim=-1)
    return logits.float(), trace, features


@torch.no_grad()
def _ni_nearest_centroid_accuracy(train_features, train_labels, eval_features, eval_labels):
    """Ask whether the residual itself, not just later logits, separates labels."""
    centers = torch.stack([train_features[train_labels == label].mean(dim=0) for label in (0, 1)])
    prediction = (eval_features.unsqueeze(1) - centers.unsqueeze(0)).square().mean(dim=-1).argmin(dim=-1)
    return float((prediction == eval_labels).float().mean())


@torch.no_grad()
def _ni_evidence_inspect_metrics(model, device):
    """Measure whether residuals causally mediate a known two-stream answer."""
    calibration = _ni_evidence_batch(256, 8, model.emb.embedding_dim, device, 120_000)
    heldout = _ni_evidence_batch(512, 8, model.emb.embedding_dim, device, 130_000)
    _, _, cal_features = _ni_evidence_trace_features(model, calibration[0], calibration[2])
    logits, trace, features = _ni_evidence_trace_features(model, heldout[0], heldout[2])
    _, targets, evidence = heldout
    labels = targets[:, -1] - 2
    cal_labels = calibration[1][:, -1] - 2
    ablated_logits, _ = model(heldout[0], evidence=evidence, ni_scale=0.0)
    ablated_logits = ablated_logits.float()

    target = targets[:, -1].unsqueeze(-1)
    opposite = (5 - targets[:, -1]).unsqueeze(-1)
    margin = logits[:, -1].gather(1, target) - logits[:, -1].gather(1, opposite)
    ablated_margin = ablated_logits[:, -1].gather(1, target) - ablated_logits[:, -1].gather(1, opposite)

    def flipped(modality):
        counterfactual = dict(evidence)
        values = evidence[modality].clone()
        values[:, 0, 0].neg_()
        counterfactual[modality] = values
        changed_logits, _, changed_features = _ni_evidence_trace_features(model, heldout[0], counterfactual)
        return changed_logits, changed_features

    image_logits, image_features = flipped("image")
    audio_logits, audio_features = flipped("audio")
    residual_stats = [
        {
            "layer": item["layer"],
            "residual_rms": float(item["residual"].float().square().mean().sqrt()),
            "state_rms": float(item["state_rms"]),
        }
        for item in trace
    ]
    return {
        "full_accuracy": float((logits[:, -1].argmax(dim=-1) == targets[:, -1]).float().mean()),
        "residual_ablated_accuracy": float((ablated_logits[:, -1].argmax(dim=-1) == targets[:, -1]).float().mean()),
        "full_margin": float(margin.mean()),
        "residual_ablated_margin": float(ablated_margin.mean()),
        "residual_centroid_accuracy": _ni_nearest_centroid_accuracy(cal_features, cal_labels, features, labels),
        "image_flip_rate": float((logits[:, -1].argmax(dim=-1) != image_logits[:, -1].argmax(dim=-1)).float().mean()),
        "audio_flip_rate": float((logits[:, -1].argmax(dim=-1) != audio_logits[:, -1].argmax(dim=-1)).float().mean()),
        "image_residual_delta_rms": float((features - image_features).square().mean().sqrt()),
        "audio_residual_delta_rms": float((features - audio_features).square().mean().sqrt()),
        "depth_residuals": residual_stats,
    }


def run_ni_evidence_inspect(steps=500):
    """Train the small binding model, then test whether NI output is used, not noise."""
    if steps <= 0:
        raise ValueError("NI evidence inspection steps must be positive")
    device = torch.device(DEVICE)
    width_mult = _ni_probe_width()
    print(f"NI evidence inspection: width={width_mult}, train {steps} steps then trace/ablate on {device}", flush=True)
    started = time.time()
    _, model = _run_ni_evidence_condition(True, steps, device, width_mult=width_mult)
    metrics = _ni_evidence_inspect_metrics(model, device)
    print(
        "NI residual: "
        f"full_acc={metrics['full_accuracy']:.3f} ablated_acc={metrics['residual_ablated_accuracy']:.3f} "
        f"margin={metrics['full_margin']:.3f}->{metrics['residual_ablated_margin']:.3f} "
        f"centroid={metrics['residual_centroid_accuracy']:.3f}",
        flush=True,
    )
    print(
        "NI counterfactual: "
        f"image_flip={metrics['image_flip_rate']:.3f} audio_flip={metrics['audio_flip_rate']:.3f} "
        f"image_residual_delta={metrics['image_residual_delta_rms']:.4f} "
        f"audio_residual_delta={metrics['audio_residual_delta_rms']:.4f}",
        flush=True,
    )
    print(f"NI evidence inspection elapsed={time.time() - started:.1f}s", flush=True)
    return metrics


class ArchiveHadamardCompiler(nn.Module):
    """Cheap structured surrogate for one archived D->H->D MLP path.

    This is intentionally not another conventional hidden-width activation:
    ``diag*x + U((Vx) * tanh(Wx))`` uses a low-rank multiplicative feature
    basis.  Its cost is about 3*D*rank + D versus the archive's 2*D*H.
    Whether it actually compiles a trained path is decided by held-out LM loss
    and measured speed, never by this formula alone.
    """
    def __init__(self, D, rank):
        super().__init__()
        if D <= 0 or rank <= 0:
            raise ValueError("archive compiler D and rank must be positive")
        self.left = nn.Linear(D, rank, bias=False, dtype=DTYPE)
        self.right = nn.Linear(D, rank, bias=False, dtype=DTYPE)
        self.out = nn.Linear(rank, D, bias=False, dtype=DTYPE)
        self.diag = nn.Parameter(torch.zeros(D, dtype=DTYPE))
        self.bias = nn.Parameter(torch.zeros(D, dtype=DTYPE))

    def forward(self, x):
        product = self.left(x) * torch.tanh(self.right(x))
        return x * self.diag + self.out(product) + self.bias


class ArchiveConfidenceRouter(nn.Module):
    """Use the compiler on low predicted-error tokens; archive hard tokens.

    The confidence head is trained only from archive-vs-surrogate residuals.
    At inference it never receives the archive output before making its route.
    The sparse fallback materializes the archive only for rejected tokens, so
    its timing is a real (though un-fused) estimate rather than an idealized
    ``torch.where`` that computes both paths for every token.
    """
    def __init__(self, archive, compiler, threshold):
        super().__init__()
        self.archive = archive
        self.compiler = compiler
        D = compiler.diag.numel()
        self.error_head = nn.Linear(D, 1, dtype=DTYPE)
        self.register_buffer("threshold", torch.tensor(float(threshold), dtype=torch.float32))

    def predicted_error(self, x):
        return self.error_head(x).squeeze(-1).float()

    def forward(self, x):
        compiled = self.compiler(x)
        use_compiler = self.predicted_error(x) <= self.threshold
        if bool(use_compiler.all()):
            return compiled
        flat_x = x.reshape(-1, x.shape[-1])
        flat_out = compiled.reshape(-1, compiled.shape[-1]).clone()
        fallback = (~use_compiler).reshape(-1)
        # Avoid an archive forward for accepted tokens. This is correct for a
        # token-local MLP and makes the reported routed speed meaningful.
        flat_out[fallback] = self.archive(flat_x[fallback])
        return flat_out.view_as(compiled)


class ArchiveChunkConfidenceRouter(ArchiveConfidenceRouter):
    """Route contiguous token chunks to either the compiler or exact archive.

    This is the deployable interpretation of an archived subnetwork: a cheap
    estimate handles an entire easy chunk, while an uncertain chunk executes
    the precise original MLP.  Unlike per-token fallback, both experts receive
    dense [number_of_chunks, chunk_size, D] inputs, avoiding a scatter of tiny
    GEMMs.  MLPs are token-local, so zero padding in a final partial chunk is
    discarded before return and cannot affect real-token outputs.
    """

    def __init__(self, archive, compiler, threshold, chunk_size):
        super().__init__(archive, compiler, threshold)
        if chunk_size <= 0:
            raise ValueError("archive chunk router chunk size must be positive")
        self.chunk_size = int(chunk_size)

    def chunk_predicted_error(self, x):
        """Mean predicted log error per valid contiguous token chunk."""
        if x.ndim == 2:
            x = x.unsqueeze(0)
        if x.ndim != 3:
            raise ValueError("archive chunk router expects [B, T, D] or [T, D]")
        B, T, D = x.shape
        padding = (-T) % self.chunk_size
        padded = F.pad(x, (0, 0, 0, padding)) if padding else x
        chunks = padded.view(B, -1, self.chunk_size, D)
        valid = (torch.arange(chunks.shape[1] * self.chunk_size, device=x.device)
                 .view(1, -1, self.chunk_size) < T).to(torch.float32)
        token_error = self.predicted_error(padded).view(B, -1, self.chunk_size)
        chunk_error = (token_error * valid).sum(dim=-1) / valid.sum(dim=-1).clamp_min(1.0)
        return chunks, chunk_error, T

    def forward(self, x):
        input_was_2d = x.ndim == 2
        chunks, predicted_error, original_T = self.chunk_predicted_error(x)
        B, N, C, D = chunks.shape
        flat_chunks = chunks.reshape(B * N, C, D)
        use_compiler = (predicted_error <= self.threshold).reshape(-1)
        flat_out = torch.empty_like(flat_chunks)
        if bool(use_compiler.any()):
            flat_out[use_compiler] = self.compiler(flat_chunks[use_compiler])
        if bool((~use_compiler).any()):
            # This is the exact, pre-compilation MLP for every hard chunk.
            flat_out[~use_compiler] = self.archive(flat_chunks[~use_compiler])
        out = flat_out.view(B, N * C, D)[:, :original_T]
        return out[0] if input_was_2d else out


def _archive_probe_text():
    """Load a local held-out character corpus without downloading data."""
    candidates = (
        os.environ.get("TB_ARCHIVE_TEXT", ""),
        "C:/Users/Ace/Downloads/tinyshakespeare.txt",
        "C:/Users/Ace/Downloads/tinyshakespeare/input.txt",
        "E:/PROJECT/graph_v5/input.txt",
    )
    for path in candidates:
        if path and os.path.isfile(path):
            return open(path, "r", encoding="utf-8").read(), path
    raise FileNotFoundError("archive compiler probe needs TB_ARCHIVE_TEXT or local tinyshakespeare.txt")


def _archive_probe_batch(tokens, batch_size, T, generator, device):
    starts = torch.randint(0, len(tokens) - T - 1, (batch_size,), generator=generator)
    x = torch.stack([tokens[int(start):int(start) + T] for start in starts])
    y = torch.stack([tokens[int(start) + 1:int(start) + T + 1] for start in starts])
    return x.to(device), y.to(device)


@torch.no_grad()
def _archive_probe_loss(model, tokens, batch_size, T, device, seed, batches=12):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    total = 0.0
    model.eval()
    for _ in range(batches):
        xb, yb = _archive_probe_batch(tokens, batch_size, T, generator, device)
        _, loss = model(xb, yb)
        total += float(loss)
    return total / batches


def _archive_probe_seconds(module, sample, repeats=40):
    """Benchmark a token-local path, synchronizing only when CUDA is present."""
    module.eval()
    with torch.no_grad():
        for _ in range(8):
            module(sample)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(repeats):
            module(sample)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    return (time.perf_counter() - started) / repeats


def _hier_layers(spec, L):
    """Parse ``all`` or a comma-separated block list for the hierarchy probe."""
    if str(spec).strip().lower() == "all":
        return tuple(range(L))
    layers = tuple(sorted({int(value.strip()) for value in str(spec).split(",") if value.strip()}))
    if not layers or any(layer < 0 or layer >= L for layer in layers):
        raise ValueError("TB_HIER_LAYERS must be 'all' or valid comma-separated block indices")
    return layers


def _hier_model_pair(vocab, D, L, heads, mult, max_t, chunk_size, layers, device):
    """Return identical full and hierarchical LMs with exactly matched parameters."""
    saved = {key: os.environ.get(key) for key in ("TB_ATTN_EVERY", "TB_MLP_COMPACT", "TB_MLP_COMPACT_EQUAL")}
    os.environ.update({"TB_ATTN_EVERY": "1", "TB_MLP_COMPACT": "0", "TB_MLP_COMPACT_EQUAL": "0"})
    try:
        torch.manual_seed(121_337)
        full = WalkGPT(vocab, D, L, heads, mult=mult, hybrid_w=0, win_heads=0,
                       no_walk=True, max_t=max_t, rope=True, no_mem=True).to(device)
        torch.manual_seed(131_337)
        hierarchy = WalkGPT(vocab, D, L, heads, mult=mult, hybrid_w=0, win_heads=0,
                            no_walk=True, max_t=max_t, rope=True, no_mem=True).to(device)
        hierarchy.load_state_dict(full.state_dict(), strict=True)
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    for layer in layers:
        original = full.blocks[layer].attn
        replacement = HierarchicalAttn(D, heads, chunk_size=chunk_size, rope=True).to(device)
        replacement.load_state_dict(original.state_dict(), strict=True)
        hierarchy.blocks[layer].attn = replacement
    full_params = sum(parameter.numel() for parameter in full.parameters())
    hierarchy_params = sum(parameter.numel() for parameter in hierarchy.parameters())
    if full_params != hierarchy_params:
        raise AssertionError(f"hierarchical probe must be parameter matched ({full_params} != {hierarchy_params})")
    return full, hierarchy, full_params


def _hier_probe_train_pair(full, hierarchy, tokens, batch_size, T, steps, device):
    """Train matched models on the same batches and return their optimizers' losses."""
    full_opt = torch.optim.AdamW(full.parameters(), lr=8e-4, weight_decay=0.01)
    hier_opt = torch.optim.AdamW(hierarchy.parameters(), lr=8e-4, weight_decay=0.01)
    generator = torch.Generator(device="cpu").manual_seed(141_337)
    latest = (float("nan"), float("nan"))
    full.train(), hierarchy.train()
    for _ in range(steps):
        xb, yb = _archive_probe_batch(tokens, batch_size, T, generator, device)
        _, full_loss = full(xb, yb)
        full_opt.zero_grad(set_to_none=True)
        full_loss.backward()
        torch.nn.utils.clip_grad_norm_(full.parameters(), 1.0)
        full_opt.step()
        _, hier_loss = hierarchy(xb, yb)
        hier_opt.zero_grad(set_to_none=True)
        hier_loss.backward()
        torch.nn.utils.clip_grad_norm_(hierarchy.parameters(), 1.0)
        hier_opt.step()
        latest = (float(full_loss.detach()), float(hier_loss.detach()))
    return latest


def _hier_recall_batch(batch_size, T, pairs, vocab, generator, device, spacing=0,
                       return_chunk_targets=False, chunk_size=0):
    """Causal key/value recall with every answer farther than one local chunk.

    The only supervised positions are suffix queries.  A model has to recover a
    value paired with the key near the sequence start; filler carries no answer.
    This isolates the precise long-retrieval behavior hierarchy may compress.
    """
    if vocab < 2 * pairs + 2 or 4 * pairs > T:
        raise ValueError("hierarchical recall probe needs larger vocabulary/context")
    if spacing < 0 or (spacing and (pairs - 1) * spacing + 1 >= T - 2 * pairs):
        raise ValueError("recall spacing must leave every key/value pair before suffix queries")
    if return_chunk_targets and chunk_size <= 0:
        raise ValueError("chunk_size is required when returning recall chunk targets")
    x = torch.full((batch_size, T), 1, dtype=torch.long)
    query_positions = torch.arange(T - 2 * pairs, T, 2)
    answers = torch.empty(batch_size, pairs, dtype=torch.long)
    chunk_targets = torch.empty(batch_size, pairs, dtype=torch.long) if return_chunk_targets else None
    for b in range(batch_size):
        # Fixed small vocabularies make this a retrieval test instead of an
        # artificial 128-way output-classification bottleneck.  Key/value
        # pairings and query order remain freshly permuted per sequence.
        keys = torch.randperm(pairs, generator=generator) + 2
        values = torch.randperm(pairs, generator=generator) + 2 + pairs
        order = torch.randperm(pairs, generator=generator)
        if spacing:
            context_positions = torch.arange(pairs) * spacing
            x[b, context_positions] = keys
            x[b, context_positions + 1] = values
        else:
            context_positions = torch.arange(pairs) * 2
            x[b, :2 * pairs:2] = keys
            x[b, 1:2 * pairs:2] = values
        x[b, query_positions] = keys[order]
        answers[b] = values[order]
        if chunk_targets is not None:
            chunk_targets[b] = (context_positions[order] // chunk_size)
    if chunk_targets is None:
        return x.to(device), query_positions.to(device), answers.to(device)
    return x.to(device), query_positions.to(device), answers.to(device), chunk_targets.to(device)


@torch.no_grad()
def _hier_recall_accuracy(model, T, pairs, vocab, device, seed, batches=24, spacing=0):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    correct, total = 0, 0
    model.eval()
    for _ in range(batches):
        xb, positions, answers = _hier_recall_batch(8, T, pairs, vocab, generator, device, spacing=spacing)
        logits, _ = model(xb)
        predicted = logits[:, positions].float().argmax(dim=-1)
        correct += int((predicted == answers).sum())
        total += answers.numel()
    return correct / max(1, total)


def _hier_probe_recall(full, hierarchy, steps, T, pairs, vocab, device, spacing=0):
    """Train a matched long-copy task; CE alone can otherwise hide recall loss."""
    full_opt = torch.optim.AdamW(full.parameters(), lr=1e-3, weight_decay=0.0)
    hier_opt = torch.optim.AdamW(hierarchy.parameters(), lr=1e-3, weight_decay=0.0)
    generator = torch.Generator(device="cpu").manual_seed(151_337)
    full.train(), hierarchy.train()
    for _ in range(steps):
        xb, positions, answers = _hier_recall_batch(8, T, pairs, vocab, generator, device, spacing=spacing)
        for model, optimizer in ((full, full_opt), (hierarchy, hier_opt)):
            logits, _ = model(xb)
            loss = F.cross_entropy(logits[:, positions].reshape(-1, vocab), answers.reshape(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return (_hier_recall_accuracy(full, T, pairs, vocab, device, 161_337, spacing=spacing),
            _hier_recall_accuracy(hierarchy, T, pairs, vocab, device, 161_337, spacing=spacing))


@torch.no_grad()
def _hier_probe_loss(model, tokens, batch_size, T, device, seed, batches=12):
    """Float32 evaluation avoids bf16 loss-value quantization in small deltas."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    total = 0.0
    model.eval()
    for _ in range(batches):
        xb, yb = _archive_probe_batch(tokens, batch_size, T, generator, device)
        logits, _ = model(xb)
        total += float(F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), yb.reshape(-1)))
    return total / batches


def run_hierarchical_attention_probe(train_steps=600):
    """Parameter-matched full-vs-hierarchical attention experiment.

    This does not modify a checkpoint.  It trains two fresh tiny RoPE LMs on
    the same Tiny Shakespeare batches, then measures held-out CE at several
    lengths, real attention-path time, and a separate long key/value recall
    task.  The hierarchy uses deterministic chunk summaries, so its parameter
    count must exactly equal the all-full-attention baseline.
    """
    if train_steps <= 0:
        raise ValueError("hierarchical attention probe steps must be positive")
    text, source = _archive_probe_text()
    chars = sorted(set(text))
    stoi = {char: index for index, char in enumerate(chars)}
    data = torch.tensor([stoi[char] for char in text], dtype=torch.long)
    split = int(len(data) * 0.9)
    train_tokens, val_tokens = data[:split], data[split:]
    D = int(os.environ.get("TB_HIER_D", "256"))
    L = int(os.environ.get("TB_HIER_L", "4"))
    heads = int(os.environ.get("TB_HIER_HEADS", "4"))
    train_T = int(os.environ.get("TB_HIER_TRAIN_T", "128"))
    chunk_size = int(os.environ.get("TB_HIER_CHUNK", "32"))
    max_t = int(os.environ.get("TB_HIER_MAX_T", str(max(512, train_T))))
    batch_size = int(os.environ.get("TB_HIER_BATCH", "8"))
    if D <= 0 or D % heads or L <= 0 or train_T <= 0 or train_T > max_t:
        raise ValueError("invalid TB_HIER_D/L/HEADS/TRAIN_T/MAX_T configuration")
    layers = _hier_layers(os.environ.get("TB_HIER_LAYERS", "all"), L)
    requested = os.environ.get("TB_HIER_EVAL_T", f"{train_T},{min(max_t, train_T * 2)},{max_t}")
    eval_lengths = tuple(sorted({int(value.strip()) for value in requested.split(",") if value.strip()}))
    if not eval_lengths or any(length <= 0 or length > max_t for length in eval_lengths):
        raise ValueError("TB_HIER_EVAL_T must contain lengths in [1, TB_HIER_MAX_T]")
    device = torch.device(DEVICE)
    full, hierarchy, params = _hier_model_pair(len(chars), D, L, heads, 2.0, max_t,
                                                chunk_size, layers, device)
    print(f"Hierarchical attention: Tiny Shakespeare {train_steps} steps on {source}; "
          f"D={D} L={L} H={heads} chunks={chunk_size} layers={layers} params={params}", flush=True)
    _hier_probe_train_pair(full, hierarchy, train_tokens, batch_size, train_T, train_steps, device)
    result = {
        "params_full": params,
        "params_hier": sum(parameter.numel() for parameter in hierarchy.parameters()),
        "chunk_size": chunk_size,
        "layers": ",".join(map(str, layers)),
    }
    for length in eval_lengths:
        full_loss = _hier_probe_loss(full, val_tokens, batch_size, length, device, 171_337 + length, batches=8)
        hier_loss = _hier_probe_loss(hierarchy, val_tokens, batch_size, length, device, 171_337 + length, batches=8)
        result[f"full_ce_t{length}"] = full_loss
        result[f"hier_ce_t{length}"] = hier_loss
        result[f"hier_delta_t{length}"] = hier_loss - full_loss
        sample = torch.randn(2, length, D, device=device, dtype=DTYPE)
        full_ms = _archive_probe_seconds(full.blocks[layers[0]].attn, sample, repeats=20) * 1000.0
        hier_attn = hierarchy.blocks[layers[0]].attn
        hier_ms = _archive_probe_seconds(hier_attn, sample, repeats=20) * 1000.0
        result[f"full_attn_ms_t{length}"] = full_ms
        result[f"hier_attn_ms_t{length}"] = hier_ms
        result[f"speedup_t{length}"] = full_ms / max(hier_ms, 1e-12)
        result[f"candidates_t{length}"] = hier_attn.candidate_count(length)

    recall_steps = int(os.environ.get("TB_HIER_RECALL_STEPS", str(max(400, train_steps))))
    recall_T = int(os.environ.get("TB_HIER_RECALL_T", str(min(max_t, 256))))
    recall_pairs = int(os.environ.get("TB_HIER_RECALL_PAIRS", str(min(8, recall_T // 8))))
    recall_vocab = max(32, 2 * recall_pairs + 2)
    recall_full, recall_hier, _ = _hier_model_pair(recall_vocab, D, L, heads, 2.0, recall_T,
                                                    chunk_size, layers, device)
    full_acc, hier_acc = _hier_probe_recall(recall_full, recall_hier, recall_steps, recall_T,
                                             recall_pairs, recall_vocab, device)
    result["recall_steps"] = recall_steps
    result["full_recall_acc"] = full_acc
    result["hier_recall_acc"] = hier_acc
    result["hier_recall_delta_pp"] = (hier_acc - full_acc) * 100.0
    # A flat near-chance control says nothing about hierarchy's retention.  Make
    # that explicit rather than treating equal failure as preserved recall.
    result["recall_control_learned"] = int(full_acc >= 0.80)
    print("Hierarchical attention result: " + " ".join(
        f"{key}={value:.5f}" if isinstance(value, float) else f"{key}={value}"
        for key, value in result.items()), flush=True)
    return result


def _archive_retrieval_model_pair(vocab, D, L, heads, mult, max_t, chunk_size, retrieve_chunks, layers, device):
    """Build a full-attention control and an equal-parameter archive-retrieval LM."""
    full, candidate, params = _hier_model_pair(vocab, D, L, heads, mult, max_t, chunk_size, layers, device)
    straight_through = os.environ.get("TB_AR_STRAIGHT_THROUGH", "1") == "1"
    for layer in layers:
        replacement = ArchiveRetrievalAttn(
            D, heads, chunk_size=chunk_size, retrieve_chunks=retrieve_chunks, rope=True,
            straight_through=straight_through).to(device)
        replacement.load_state_dict(full.blocks[layer].attn.state_dict(), strict=True)
        candidate.blocks[layer].attn = replacement
    if sum(parameter.numel() for parameter in candidate.parameters()) != params:
        raise AssertionError("archive retrieval probe lost parameter matching")
    return full, candidate, params


@torch.no_grad()
def _archive_route_hit_rate(model, T, pairs, vocab, spacing, chunk_size, device, seed, batches=24):
    """Measure whether the final archive layer opens the answer's true chunk."""
    modules = [module for module in model.modules() if isinstance(module, ArchiveRetrievalAttn)]
    if not modules:
        return float("nan"), float("nan")
    for module in modules:
        module.capture_routes = True
    final = modules[-1]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    any_hits, all_hits, total = 0, 0, 0
    model.eval()
    try:
        for _ in range(batches):
            xb, positions, _answers, targets = _hier_recall_batch(
                8, T, pairs, vocab, generator, device, spacing=spacing,
                return_chunk_targets=True, chunk_size=chunk_size)
            model(xb)
            current_chunk = int(positions[0].item()) // chunk_size
            selected = final._last_selected_chunks[current_chunk]
            if selected is None:
                raise AssertionError("suffix recall queries unexpectedly have no archive route")
            local_positions = positions - current_chunk * chunk_size
            # [B, H, P, K]. A query succeeds when any/all final-layer heads
            # selected the physical chunk that holds its paired value.
            selected = selected[:, :, local_positions, :]
            per_head = (selected == targets[:, None, :, None]).any(dim=-1)
            any_hits += int(per_head.any(dim=1).sum())
            all_hits += int(per_head.all(dim=1).sum())
            total += per_head.size(0) * per_head.size(2)
    finally:
        for module in modules:
            module.capture_routes = False
    return any_hits / max(1, total), all_hits / max(1, total)


def run_archive_retrieval_probe(train_steps=600):
    """Evaluate exact K/V archive retrieval against full causal attention.

    This is a fresh, parameter-matched probe.  Its index is hard top-k mean-K
    landmarks, while retrieved K/V vectors are exact.  Hard selection is the
    intended serving path; a learned/soft routing objective is a later variant
    if this deterministic control demonstrates useful retention first.
    """
    if train_steps <= 0:
        raise ValueError("archive retrieval probe steps must be positive")
    text, source = _archive_probe_text()
    chars = sorted(set(text))
    stoi = {char: index for index, char in enumerate(chars)}
    data = torch.tensor([stoi[char] for char in text], dtype=torch.long)
    split = int(len(data) * 0.9)
    train_tokens, val_tokens = data[:split], data[split:]
    D = int(os.environ.get("TB_AR_D", "256"))
    L = int(os.environ.get("TB_AR_L", "4"))
    heads = int(os.environ.get("TB_AR_HEADS", "4"))
    train_T = int(os.environ.get("TB_AR_TRAIN_T", "128"))
    max_t = int(os.environ.get("TB_AR_MAX_T", str(max(512, train_T))))
    chunk_size = int(os.environ.get("TB_AR_CHUNK", "32"))
    retrieve_chunks = int(os.environ.get("TB_AR_TOPK", "2"))
    batch_size = int(os.environ.get("TB_AR_BATCH", "8"))
    if D <= 0 or D % heads or L <= 0 or train_T <= 1 or train_T > max_t or batch_size <= 0:
        raise ValueError("invalid TB_AR dimensions, context, or batch")
    layers = _hier_layers(os.environ.get("TB_AR_LAYERS", "all"), L)
    requested = os.environ.get("TB_AR_EVAL_T", f"{train_T},{min(max_t, train_T * 2)},{max_t}")
    eval_lengths = tuple(sorted({int(value.strip()) for value in requested.split(",") if value.strip()}))
    if not eval_lengths or any(length <= 0 or length > max_t for length in eval_lengths):
        raise ValueError("TB_AR_EVAL_T must contain lengths in [1, TB_AR_MAX_T]")
    device = torch.device(DEVICE)
    full, retrieved, params = _archive_retrieval_model_pair(
        len(chars), D, L, heads, 2.0, max_t, chunk_size, retrieve_chunks, layers, device)
    print(f"Archive retrieval attention: Tiny Shakespeare {train_steps} steps on {source}; "
          f"D={D} L={L} H={heads} C={chunk_size} topk={retrieve_chunks} layers={layers} params={params}", flush=True)
    _hier_probe_train_pair(full, retrieved, train_tokens, batch_size, train_T, train_steps, device)
    result = {
        "params_full": params,
        "params_archive_retrieval": sum(parameter.numel() for parameter in retrieved.parameters()),
        "chunk_size": chunk_size,
        "topk_chunks": retrieve_chunks,
        "straight_through": int(os.environ.get("TB_AR_STRAIGHT_THROUGH", "1") == "1"),
        "layers": ",".join(map(str, layers)),
    }
    for length in eval_lengths:
        full_loss = _hier_probe_loss(full, val_tokens, batch_size, length, device, 231_337 + length, batches=8)
        archive_loss = _hier_probe_loss(retrieved, val_tokens, batch_size, length, device, 231_337 + length, batches=8)
        result[f"full_ce_t{length}"] = full_loss
        result[f"archive_ce_t{length}"] = archive_loss
        result[f"archive_delta_t{length}"] = archive_loss - full_loss
        sample = torch.randn(2, length, D, device=device, dtype=DTYPE)
        full_ms = _archive_probe_seconds(full.blocks[layers[0]].attn, sample, repeats=20) * 1000.0
        archive_attn = retrieved.blocks[layers[0]].attn
        archive_ms = _archive_probe_seconds(archive_attn, sample, repeats=20) * 1000.0
        result[f"full_attn_ms_t{length}"] = full_ms
        result[f"archive_attn_ms_t{length}"] = archive_ms
        result[f"speedup_t{length}"] = full_ms / max(archive_ms, 1e-12)
        result[f"exact_tokens_t{length}"] = archive_attn.candidate_count(length)

    recall_steps = int(os.environ.get("TB_AR_RECALL_STEPS", str(max(400, train_steps))))
    recall_T = int(os.environ.get("TB_AR_RECALL_T", str(min(max_t, 256))))
    recall_pairs = int(os.environ.get("TB_AR_RECALL_PAIRS", str(min(8, recall_T // 8))))
    recall_spacing = int(os.environ.get("TB_AR_RECALL_SPACING", "0"))
    recall_vocab = max(32, 2 * recall_pairs + 2)
    recall_full, recall_archive, _ = _archive_retrieval_model_pair(
        recall_vocab, D, L, heads, 2.0, recall_T, chunk_size, retrieve_chunks, layers, device)
    full_acc, archive_acc = _hier_probe_recall(recall_full, recall_archive, recall_steps, recall_T,
                                                recall_pairs, recall_vocab, device, spacing=recall_spacing)
    result["recall_steps"] = recall_steps
    result["recall_spacing"] = recall_spacing
    result["full_recall_acc"] = full_acc
    result["archive_recall_acc"] = archive_acc
    result["archive_recall_delta_pp"] = (archive_acc - full_acc) * 100.0
    result["recall_control_learned"] = int(full_acc >= 0.80)
    if recall_spacing:
        route_any, route_all = _archive_route_hit_rate(
            recall_archive, recall_T, recall_pairs, recall_vocab, recall_spacing,
            chunk_size, device, 241_337)
        result["route_hit_any_head"] = route_any
        result["route_hit_all_heads"] = route_all
    print("Archive retrieval result: " + " ".join(
        f"{key}={value:.5f}" if isinstance(value, float) else f"{key}={value}"
        for key, value in result.items()), flush=True)
    return result


# --- REMOVED 2026-09-30 (lean cleanup): Pretrained*Tail probe subsystem ---
# What lived here (~970 lines): PretrainedWalkTail, PretrainedPhaseWalkTail,
#   PretrainedDistributedPhaseWalk/DistributedWalk/DRCGLiteTail, Subspace/Gated/
#   MemoryGated/ContentGated/Selective/DualPayload/Evidence/LowRank/Pointwise/
#   Compact tails, their hook/batch/loss helpers, _prime_pretrained_phase_gates,
#   _configure_pretrained_phase_amplifiers, and run_pretrained_walk_tail_probe.
# Why removed: probe-only subsystem (offline tail-fitting experiments against
#   frozen hiddens). Zero references from train()/evaluate()/forward/config;
#   only reachable via the --pretrained-walk-tail-probe CLI flag, which now
#   raises. Nothing in the training or resume path can observe the difference.
# To restore: recover from train_walker_1b_continue.py.bak-20260930.
def run_archive_compiler_probe(lm_steps=300, compiler_steps=600):
    """Test archive-verified MLP compilation on a trained tiny language model.

    This is an offline experiment. It trains the base LM on the first 90% of
    Tiny Shakespeare, freezes it, fits a low-rank Hadamard operator to one real
    MLP's hidden-state input/output pairs, then evaluates all-compiled and
    confidence-routed replacement on the untouched last 10%. It does not alter
    any checkpoint or the active 1B run.
    """
    if lm_steps <= 0 or compiler_steps <= 0:
        raise ValueError("archive compiler probe step counts must be positive")
    text, source = _archive_probe_text()
    chars = sorted(set(text))
    stoi = {char: index for index, char in enumerate(chars)}
    data = torch.tensor([stoi[char] for char in text], dtype=torch.long)
    split = int(len(data) * 0.9)
    train_tokens, val_tokens = data[:split], data[split:]
    device = torch.device(DEVICE)
    D = int(os.environ.get("TB_ARCHIVE_D", "64"))
    L, heads, T, batch_size = 2, 4, 64, 16
    if D <= 0 or D % heads:
        raise ValueError("TB_ARCHIVE_D must be positive and divisible by 4 heads")
    rank = int(os.environ.get("TB_ARCHIVE_RANK", str(max(8, D * 5 // 16))))
    archive_layer = int(os.environ.get("TB_ARCHIVE_LAYER", "0"))
    coverage = float(os.environ.get("TB_ARCHIVE_COVERAGE", "0.80"))
    route_chunk = int(os.environ.get("TB_ARCHIVE_ROUTE_CHUNK", "16"))
    if not 0.05 <= coverage <= 0.95:
        raise ValueError("TB_ARCHIVE_COVERAGE must be in [0.05, 0.95]")
    if route_chunk <= 0:
        raise ValueError("TB_ARCHIVE_ROUTE_CHUNK must be positive")
    if not 0 <= archive_layer < L:
        raise ValueError("TB_ARCHIVE_LAYER is outside the tiny probe's block range")

    old_compact = os.environ.get("TB_MLP_COMPACT")
    old_equal = os.environ.get("TB_MLP_COMPACT_EQUAL")
    # The archive must be the original GELU MLP; restore user environment once
    # it has been constructed so this probe cannot leak configuration outward.
    os.environ["TB_MLP_COMPACT"] = "0"
    os.environ["TB_MLP_COMPACT_EQUAL"] = "0"
    try:
        torch.manual_seed(71_337)
        model = WalkGPT(len(chars), D, L, heads, mult=2.0, hybrid_w=T,
                        win_heads=heads, no_walk=True, max_t=T, rope=True,
                        no_mem=True).to(device)
    finally:
        if old_compact is None:
            os.environ.pop("TB_MLP_COMPACT", None)
        else:
            os.environ["TB_MLP_COMPACT"] = old_compact
        if old_equal is None:
            os.environ.pop("TB_MLP_COMPACT_EQUAL", None)
        else:
            os.environ["TB_MLP_COMPACT_EQUAL"] = old_equal

    print(f"Archive compiler: train tiny LM {lm_steps} steps on {source}; compile block {archive_layer} "
          f"rank={rank} route_chunk={route_chunk}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=0.01)
    train_generator = torch.Generator(device="cpu").manual_seed(81_337)
    model.train()
    for _ in range(lm_steps):
        xb, yb = _archive_probe_batch(train_tokens, batch_size, T, train_generator, device)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    archive = model.blocks[archive_layer].mlp
    archive.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    captured_inputs, captured_outputs = [], []
    def capture(_module, inputs, output):
        captured_inputs.append(inputs[0].detach().float().cpu())
        captured_outputs.append(output.detach().float().cpu())
    handle = archive.register_forward_hook(capture)
    with torch.no_grad():
        collector = torch.Generator(device="cpu").manual_seed(91_337)
        for _ in range(24):
            xb, _ = _archive_probe_batch(train_tokens, batch_size, T, collector, device)
            model(xb)
    handle.remove()
    archive_x = torch.cat(captured_inputs).reshape(-1, D)
    archive_y = torch.cat(captured_outputs).reshape(-1, D)
    compiler = ArchiveHadamardCompiler(D, rank).to(device)
    compiler_optimizer = torch.optim.AdamW(compiler.parameters(), lr=2e-3, weight_decay=1e-4)
    compiler_generator = torch.Generator(device="cpu").manual_seed(101_337)
    compiler.train()
    for _ in range(compiler_steps):
        index = torch.randint(0, len(archive_x), (2048,), generator=compiler_generator)
        xb = archive_x[index].to(device, dtype=DTYPE)
        target = archive_y[index].to(device, dtype=DTYPE)
        prediction = compiler(xb)
        loss = F.mse_loss(prediction.float(), target.float())
        compiler_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        compiler_optimizer.step()

    # Fit the router to a log-error target on a calibration split disjoint from
    # compiler updates. It learns a route from x alone, not archive output.
    calibration_x = archive_x[-min(32768, len(archive_x) // 4):]
    calibration_y = archive_y[-len(calibration_x):]
    provisional = ArchiveConfidenceRouter(archive, compiler, threshold=0.0).to(device)
    router_optimizer = torch.optim.AdamW(provisional.error_head.parameters(), lr=3e-3)
    with torch.no_grad():
        compiler_error = (compiler(calibration_x.to(device, dtype=DTYPE)).float() - calibration_y.to(device)).square().mean(dim=-1)
        error_target = compiler_error.clamp_min(1e-8).log()
    for _ in range(max(100, compiler_steps // 4)):
        index = torch.randint(0, len(calibration_x), (min(2048, len(calibration_x)),), generator=compiler_generator)
        xb = calibration_x[index].to(device, dtype=DTYPE)
        loss = F.mse_loss(provisional.predicted_error(xb), error_target[index].to(device))
        router_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        router_optimizer.step()
    with torch.no_grad():
        predicted = provisional.predicted_error(calibration_x.to(device, dtype=DTYPE))
        threshold = float(torch.quantile(predicted, coverage))
    router = ArchiveConfidenceRouter(archive, compiler, threshold=threshold).to(device)
    router.error_head.load_state_dict(provisional.error_head.state_dict())
    router.eval()
    # A single threshold is re-calibrated on *mean predicted errors per
    # contiguous chunk*. This changes only dispatch granularity: the archive
    # still runs the exact original MLP whenever the group is rejected.
    calibration_chunks = ArchiveChunkConfidenceRouter(
        archive, compiler, threshold=0.0, chunk_size=route_chunk).to(device)
    calibration_chunks.error_head.load_state_dict(provisional.error_head.state_dict())
    with torch.no_grad():
        _, chunk_errors, _ = calibration_chunks.chunk_predicted_error(
            calibration_x.to(device, dtype=DTYPE))
        chunk_threshold = float(torch.quantile(chunk_errors.flatten(), coverage))
    chunk_router = ArchiveChunkConfidenceRouter(
        archive, compiler, threshold=chunk_threshold, chunk_size=route_chunk).to(device)
    chunk_router.error_head.load_state_dict(provisional.error_head.state_dict())
    chunk_router.eval()

    baseline_loss = _archive_probe_loss(model, val_tokens, batch_size, T, device, 111_337)
    original_mlp = model.blocks[archive_layer].mlp
    model.blocks[archive_layer].mlp = compiler
    compiled_loss = _archive_probe_loss(model, val_tokens, batch_size, T, device, 111_337)
    model.blocks[archive_layer].mlp = router
    routed_loss = _archive_probe_loss(model, val_tokens, batch_size, T, device, 111_337)
    model.blocks[archive_layer].mlp = chunk_router
    chunk_routed_loss = _archive_probe_loss(model, val_tokens, batch_size, T, device, 111_337)
    model.blocks[archive_layer].mlp = original_mlp

    with torch.no_grad():
        heldout_x = archive_x[:min(16384, len(archive_x))].to(device, dtype=DTYPE)
        heldout_y = archive_y[:len(heldout_x)].to(device)
        predicted_error = router.predicted_error(heldout_x)
        accepted = predicted_error <= router.threshold
        _, chunk_predicted_error, _ = chunk_router.chunk_predicted_error(heldout_x)
        chunk_accepted = chunk_predicted_error <= chunk_router.threshold
        relative_mse = ((compiler(heldout_x).float() - heldout_y).square().mean() /
                        heldout_y.square().mean().clamp_min(1e-8))
    # Timing the real [B, T, D] call shape avoids crediting a flat mega-batch
    # to the archive while charging routing for dispatch overhead.
    timing_x = heldout_x[:batch_size * T].view(batch_size, T, D)
    archive_seconds = _archive_probe_seconds(archive, timing_x)
    compiler_seconds = _archive_probe_seconds(compiler, timing_x)
    router_seconds = _archive_probe_seconds(router, timing_x)
    chunk_router_seconds = _archive_probe_seconds(chunk_router, timing_x)
    result = {
        "archive_params": sum(parameter.numel() for parameter in archive.parameters()),
        "compiler_params": sum(parameter.numel() for parameter in compiler.parameters()),
        "rank": rank,
        "baseline_val_ce": baseline_loss,
        "compiled_val_ce": compiled_loss,
        "routed_val_ce": routed_loss,
        "chunk_routed_val_ce": chunk_routed_loss,
        "compiled_delta_ce": compiled_loss - baseline_loss,
        "routed_delta_ce": routed_loss - baseline_loss,
        "chunk_routed_delta_ce": chunk_routed_loss - baseline_loss,
        "relative_archive_mse": float(relative_mse),
        "predicted_compiler_coverage": float(accepted.float().mean()),
        "chunk_predicted_compiler_coverage": float(chunk_accepted.float().mean()),
        "archive_ms": archive_seconds * 1000.0,
        "compiler_ms": compiler_seconds * 1000.0,
        "router_ms": router_seconds * 1000.0,
        "chunk_router_ms": chunk_router_seconds * 1000.0,
    }
    print("Archive compiler result: " + " ".join(f"{key}={value:.5f}" if isinstance(value, float) else f"{key}={value}"
                                                  for key, value in result.items()), flush=True)
    return result


def run_attention_compiler_probe(lm_steps=300, compiler_steps=600):
    """Test whether a trained attention residual can be compiled like an MLP.

    The compiler is intentionally the same pointwise Hadamard family used for
    MLP archives.  Its input is the real causal pre-attention hidden state and
    its target is the archived layer's full attention residual.  This is a
    stringent negative/positive test: if it works, a layer's routing result is
    locally predictable; if it fails, the archive needs dynamic token retrieval.
    It never changes a checkpoint or the active 1B architecture.
    """
    if lm_steps <= 0 or compiler_steps <= 0:
        raise ValueError("attention compiler probe step counts must be positive")
    text, source = _archive_probe_text()
    chars = sorted(set(text))
    stoi = {char: index for index, char in enumerate(chars)}
    data = torch.tensor([stoi[char] for char in text], dtype=torch.long)
    split = int(len(data) * 0.9)
    train_tokens, val_tokens = data[:split], data[split:]
    D = int(os.environ.get("TB_ATTN_ARCHIVE_D", "64"))
    L, heads = 2, 4
    T = int(os.environ.get("TB_ATTN_ARCHIVE_T", "128"))
    batch_size = int(os.environ.get("TB_ATTN_ARCHIVE_BATCH", "8"))
    rank = int(os.environ.get("TB_ATTN_ARCHIVE_RANK", str(max(8, D * 5 // 16))))
    archive_layer = int(os.environ.get("TB_ATTN_ARCHIVE_LAYER", "0"))
    if D <= 0 or D % heads or rank <= 0 or T <= 1 or batch_size <= 0:
        raise ValueError("attention compiler dimensions, context, batch, and rank must be positive")
    if not 0 <= archive_layer < L:
        raise ValueError("TB_ATTN_ARCHIVE_LAYER is outside the tiny probe's block range")

    saved = {key: os.environ.get(key) for key in ("TB_ATTN_EVERY", "TB_MLP_COMPACT", "TB_MLP_COMPACT_EQUAL")}
    os.environ.update({"TB_ATTN_EVERY": "1", "TB_MLP_COMPACT": "0", "TB_MLP_COMPACT_EQUAL": "0"})
    try:
        torch.manual_seed(181_337)
        model = WalkGPT(len(chars), D, L, heads, mult=2.0, hybrid_w=0, win_heads=0,
                        no_walk=True, max_t=T, rope=True, no_mem=True).to(DEVICE)
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    device = torch.device(DEVICE)
    print(f"Attention compiler: train tiny LM {lm_steps} steps on {source}; "
          f"compile attention block {archive_layer} rank={rank}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=0.01)
    generator = torch.Generator(device="cpu").manual_seed(191_337)
    model.train()
    for _ in range(lm_steps):
        xb, yb = _archive_probe_batch(train_tokens, batch_size, T, generator, device)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    archive = model.blocks[archive_layer].attn
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    archive.eval()
    captured_inputs, captured_outputs = [], []
    def capture(_module, inputs, output):
        captured_inputs.append(inputs[0].detach().float().cpu())
        captured_outputs.append(output.detach().float().cpu())
    handle = archive.register_forward_hook(capture)
    with torch.no_grad():
        collector = torch.Generator(device="cpu").manual_seed(201_337)
        for _ in range(24):
            xb, _ = _archive_probe_batch(train_tokens, batch_size, T, collector, device)
            model(xb)
    handle.remove()
    archive_x = torch.cat(captured_inputs).reshape(-1, D)
    archive_y = torch.cat(captured_outputs).reshape(-1, D)

    compiler = ArchiveHadamardCompiler(D, rank).to(device)
    compiler_opt = torch.optim.AdamW(compiler.parameters(), lr=2e-3, weight_decay=1e-4)
    fit_generator = torch.Generator(device="cpu").manual_seed(211_337)
    compiler.train()
    for _ in range(compiler_steps):
        index = torch.randint(0, len(archive_x), (min(2048, len(archive_x)),), generator=fit_generator)
        xb = archive_x[index].to(device, dtype=DTYPE)
        target = archive_y[index].to(device, dtype=DTYPE)
        loss = F.mse_loss(compiler(xb).float(), target.float())
        compiler_opt.zero_grad(set_to_none=True)
        loss.backward()
        compiler_opt.step()

    baseline_loss = _hier_probe_loss(model, val_tokens, batch_size, T, device, 221_337)
    model.blocks[archive_layer].attn = compiler
    compiled_loss = _hier_probe_loss(model, val_tokens, batch_size, T, device, 221_337)
    model.blocks[archive_layer].attn = archive
    with torch.no_grad():
        heldout_x = archive_x[:min(16384, len(archive_x))].to(device, dtype=DTYPE)
        heldout_y = archive_y[:len(heldout_x)].to(device)
        relative_mse = ((compiler(heldout_x).float() - heldout_y).square().mean() /
                        heldout_y.square().mean().clamp_min(1e-8))
    archive_seconds = _archive_probe_seconds(archive, heldout_x.view(1, -1, D)[:, :T])
    compiler_seconds = _archive_probe_seconds(compiler, heldout_x)
    result = {
        "archive_attn_params": sum(parameter.numel() for parameter in archive.parameters()),
        "compiler_params": sum(parameter.numel() for parameter in compiler.parameters()),
        "rank": rank,
        "baseline_val_ce": baseline_loss,
        "compiled_val_ce": compiled_loss,
        "compiled_delta_ce": compiled_loss - baseline_loss,
        "relative_archive_mse": float(relative_mse),
        "archive_attn_ms": archive_seconds * 1000.0,
        "compiler_ms": compiler_seconds * 1000.0,
    }
    print("Attention compiler result: " + " ".join(
        f"{key}={value:.5f}" if isinstance(value, float) else f"{key}={value}"
        for key, value in result.items()), flush=True)
    return result


def _parse_infer_argv(argv):
    import argparse
    p = argparse.ArgumentParser(description="WalkGPT inference (weights-only, auto-config, loop)", add_help=True)
    p.add_argument("weights_pos", nargs="?", default=None, help="weights file positional (ckpt_ongoing.pt / ckpt_best.pt)")
    p.add_argument("--infer", action="store_true", help="force inference mode (optional if weights given)")
    p.add_argument("--weights", "--ckpt", "--checkpoint", dest="weights", default=None, help="weights file (minimal required arg)")
    p.add_argument("--tokenizer", default=None, help="tokenizer name (auto from checkpoint else gpt2)")
    p.add_argument("--prompt", default=None, help="single prompt text (if given, runs once then enters loop unless --no-loop)")
    p.add_argument("--prompt-file", "--input-file", dest="prompt_file", default=None, help="file with one prompt per line (batch loop)")
    p.add_argument("--output-file", "-o", default=None, help="write outputs to JSON file")
    p.add_argument("--device", default=None, help="cuda/cpu/cuda:0 (auto)")
    p.add_argument("--max-new", "--max_new", dest="max_new", type=int, default=128, help="max new tokens (default 128)")
    p.add_argument("--temperature", "--temp", dest="temperature", type=float, default=0.8)
    p.add_argument("--top-p", dest="top_p", type=float, default=0.9)
    p.add_argument("--rep-penalty", dest="rep_penalty", type=float, default=1.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-loop", action="store_true", help="single shot: do not enter interactive loop")
    args, _ = p.parse_known_args(argv[1:])
    w = args.weights or args.weights_pos
    return args, w



def run_stream_integrity_probe(rows_per_source=4):
    """Open configured streams, inspect real rows, tokenize them, and exit.

    Use this on the training machine after accepting any gated NVIDIA dataset
    terms.  It exercises the exact HF streaming configs, text fields, row
    filters, corruption detector, and tokenizer without constructing the 1B LM.
    """
    class _ProbeCfg: pass
    cfg = _ProbeCfg()
    cfg.stream_preset = os.environ.get("TB_STREAM_PRESET", "").strip()
    cfg.stream_mix = os.environ.get("TB_STREAM_MIX", "").strip()
    cfg.stream_mix_signature = ""
    cfg.stream = os.environ.get("TB_STREAM", "").strip()
    cfg.stream_revision = os.environ.get("TB_STREAM_REVISION", "").strip()
    cfg.stream_domain = os.environ.get("TB_STREAM_DOMAIN", "url").strip()
    cfg.stream_text_field = os.environ.get("TB_STREAM_TEXT_FIELD", "text").strip()
    cfg.stream_buf = int(os.environ.get("TB_STREAM_BUF", "1000000"))
    tok = _load_tokenizer(os.environ.get("TB_TOKENIZER", "gpt2"))
    specs = _resolve_stream_specs(cfg)
    reports = []
    if specs:
        for spec in specs:
            src = StreamSource(
                spec["dataset"], tok, 1, domain_key=spec["domain_key"],
                buf_tokens=min(cfg.stream_buf, 100_000), revision=spec.get("revision", ""),
                data_dir=spec.get("data_dir", ""), text_field=spec.get("text_field", "text"),
                source_name=spec["name"], row_filters=spec.get("filters") or {},
                file_selector=spec.get("file_selector", ""),
                message_field=spec.get("message_field", ""),
                dedup=spec.get("dedup", ""), template=spec.get("template", ""))
            reports.append(src.probe_rows(rows_per_source))
    elif cfg.stream:
        src = StreamSource(cfg.stream, tok, 1, domain_key=cfg.stream_domain,
                           buf_tokens=min(cfg.stream_buf, 100_000),
                           revision=cfg.stream_revision, text_field=cfg.stream_text_field)
        reports.append(src.probe_rows(rows_per_source))
    else:
        raise SystemExit("stream probe needs TB_STREAM_PRESET, TB_STREAM_MIX, or TB_STREAM")
    print("STREAM_PROBE_BEGIN", flush=True)
    for report in reports:
        print(json.dumps({k: v for k, v in report.items() if k != "rows"}, ensure_ascii=False), flush=True)
        for i, row in enumerate(report["rows"]):
            print(f"  [{i}] chars={row['chars']:,} toks={row['tokens']:,} :: {row['excerpt']}", flush=True)
    total_scan = sum(r["scanned"] for r in reports)
    total_accept = sum(r["accepted"] for r in reports)
    total_reject = sum(r["rejected"] for r in reports)
    print(f"STREAM_PROBE_SUMMARY sources={len(reports)} scanned={total_scan} "
          f"accepted={total_accept} rejected={total_reject}", flush=True)
    return reports

def main():
    argv = __import__("sys").argv
    if _is_infer_request(argv):
        # parse but allow fallback for `python file.py ckpt.pt` without flag
        iargs, w = _parse_infer_argv(argv)
        if w:
            # normalize: if user typed `python train_1b (3).py --infer ckpt.pt`, ckpt lands in weights_pos
            interactive = None if not iargs.no_loop else False
            # if prompt-file or prompt given with no-loop -> single shot; otherwise loop
            run_inference_loop(
                weights_path=w,
                prompt=iargs.prompt,
                prompt_file=iargs.prompt_file,
                output_file=iargs.output_file,
                device=iargs.device,
                max_new=iargs.max_new,
                temperature=iargs.temperature,
                top_p=iargs.top_p,
                rep_penalty=iargs.rep_penalty,
                seed=iargs.seed,
                interactive=interactive,
                tokenizer_name=iargs.tokenizer,
            )
            return
        # if infer flag but no weights yet, fall through to show help (argparse will have errored)
    if "--stream-probe" in argv:
        i = argv.index("--stream-probe")
        rows = int(argv[i + 1]) if len(argv) > i + 1 and not argv[i + 1].startswith("--") else 4
        run_stream_integrity_probe(rows)
        return
    if "--tokenize" in argv:
        args = argv
        tokenize_dataset(args[args.index("--tokenize") + 1],
                         args[args.index("--tokenize") + 2],
                         int(args[args.index("--tokenize") + 3]))
        return
    if "--make-cot-shard" in argv:
        args = argv
        i = args.index("--make-cot-shard")
        src, dst = args[i + 1], args[i + 2]
        cap = int(args[i + 3]) if len(args) > i + 3 else 41_000_000
        if not os.path.exists(src):
            raise SystemExit(f"input not found: {src}")
        import numpy as np
        tok = _load_tokenizer("gpt2")
        with open(src, "rb") as f:
            text = f.read().decode("utf-8", errors="replace")
        ids = _tokenize_stream(text, tok, cap)
        assert int(ids.max()) < 50257, "token id exceeds gpt2 vocab — check input"
        np.asarray(ids.numpy(), dtype="uint16").tofile(dst)
        print(f"wrote {len(ids):,} tokens -> {dst}", flush=True)
        return
    if "--ni-aux-probe" in argv:
        args = argv
        i = args.index("--ni-aux-probe")
        if len(args) <= i + 1:
            raise SystemExit("usage: --ni-aux-probe DATASET.jsonl [STEPS]")
        path = args[i + 1]
        steps = int(args[i + 2]) if len(args) > i + 2 else 600
        if steps <= 0:
            raise SystemExit("NI auxiliary probe steps must be positive")
        if not os.path.exists(path):
            raise SystemExit(f"input not found: {path}")
        run_ni_aux_probe(path, steps)
        return
    if "--ni-self-probe" in argv:
        args = argv
        i = args.index("--ni-self-probe")
        if len(args) <= i + 1:
            raise SystemExit("usage: --ni-self-probe DATASET.jsonl [STEPS]")
        path = args[i + 1]
        steps = int(args[i + 2]) if len(args) > i + 2 else 600
        if steps <= 0:
            raise SystemExit("NI self probe steps must be positive")
        if not os.path.exists(path):
            raise SystemExit(f"input not found: {path}")
        run_ni_self_probe(path, steps)
        return
    if "--ni-evidence-probe" in argv:
        args = argv
        i = args.index("--ni-evidence-probe")
        steps = int(args[i + 1]) if len(args) > i + 1 else 600
        if steps <= 0:
            raise SystemExit("NI evidence probe steps must be positive")
        run_ni_evidence_probe(steps)
        return
    if "--ni-evidence-inspect" in argv:
        args = argv
        i = args.index("--ni-evidence-inspect")
        steps = int(args[i + 1]) if len(args) > i + 1 else 500
        if steps <= 0:
            raise SystemExit("NI evidence inspection steps must be positive")
        run_ni_evidence_inspect(steps)
        return
    if "--archive-compiler-probe" in argv:
        args = argv
        i = args.index("--archive-compiler-probe")
        lm_steps = int(args[i + 1]) if len(args) > i + 1 else 300
        compiler_steps = int(args[i + 2]) if len(args) > i + 2 else 600
        run_archive_compiler_probe(lm_steps, compiler_steps)
        return
    if "--attention-compiler-probe" in argv:
        args = argv
        i = args.index("--attention-compiler-probe")
        lm_steps = int(args[i + 1]) if len(args) > i + 1 else 300
        compiler_steps = int(args[i + 2]) if len(args) > i + 2 else 600
        run_attention_compiler_probe(lm_steps, compiler_steps)
        return
    if "--hier-attn-probe" in argv:
        args = argv
        i = args.index("--hier-attn-probe")
        steps = int(args[i + 1]) if len(args) > i + 1 else 600
        run_hierarchical_attention_probe(steps)
        return
    if "--archive-retrieval-probe" in argv:
        args = argv
        i = args.index("--archive-retrieval-probe")
        steps = int(args[i + 1]) if len(args) > i + 1 else 600
        run_archive_retrieval_probe(steps)
        return
    if "--pretrained-walk-tail-probe" in argv:
        raise RuntimeError(
            "--pretrained-walk-tail-probe needs the Pretrained*Tail subsystem, "
            "removed 2026-09-30 (probe-only, zero train-path references).")
        return
    cfg = Cfg()
    cfg.seed = _env_int("TB_SEED", 0)
    cfg.tokenizer = os.environ.get("TB_TOKENIZER", "gpt2")
    cfg.vocab = _env_int("TB_VOCAB", 50257)
    cfg.D = _env_int("TB_D", 2048)
    cfg.L = _env_int("TB_L", 24)
    cfg.n_head = _env_int("TB_HEADS", 32)
    cfg.mult = float(os.environ.get("TB_MLP", "2.5"))
    cfg.steps = _env_int("TB_STEPS", 2000)
    cfg.B = _env_int("TB_BATCH", 1)
    cfg.t0 = _env_int("TB_T0", 512)
    cfg.t1 = _env_int("TB_T1", 4096)
    cfg.eval_t = _env_int("TB_EVAL_T", 512)
    cfg.lr = float(os.environ.get("TB_LR", "4e-4"))
    cfg.lr_floor = float(os.environ.get("TB_LR_FLOOR", "1e-6"))
    cfg.warmup = _env_int("TB_WARMUP", 500)
    cfg.lr_schedule = os.environ.get("TB_LR_SCHEDULE", "steps").lower()
    cfg.schedule_steps = _env_int("TB_LR_TOTAL_STEPS", 0)
    cfg.lr_total_tokens = _env_int("TB_LR_TOTAL_TOKENS", 0)
    cfg.lr_warmup_tokens = _env_int("TB_LR_WARMUP_TOKENS", 0)
    cfg.train_phase = _env_int("TB_TRAIN_PHASE", 0)
    cfg.resume_schedule = os.environ.get("TB_RESUME_SCHEDULE", "lock").strip().lower()
    if cfg.resume_schedule not in {"lock", "override"}:
        hint = " Did you mean 'override'?" if cfg.resume_schedule == "overide" else ""
        raise ValueError(
            "TB_RESUME_SCHEDULE must be exactly 'lock' or 'override'; "
            f"got {cfg.resume_schedule!r}.{hint}"
        )
    cfg.adapter_mult = float(os.environ.get("TB_ADAPTER_MULT", "3.0"))
    cfg.phase_mult = float(os.environ.get('TB_PHASE_MULT', '3.0'))
    cfg.r_adapter_mult = float(os.environ.get("TB_R_ADAPTER_MULT", "1.0"))
    if not math.isfinite(cfg.r_adapter_mult) or cfg.r_adapter_mult < 0.0:
        raise ValueError(
            "TB_R_ADAPTER_MULT must be finite and >= 0, "
            f"got {cfg.r_adapter_mult:g}")
    cfg.accum = _env_int("TB_ACCUM", 8)
    cfg.amp = os.environ.get("TB_AMP", "1") == "1"
    cfg.ckpt = os.environ.get("TB_CKPT", "1") == "1"
    cfg.walk_norm = float(os.environ.get("TB_WALK_NORM", "0.0"))
    cfg.walk_norm_mode = GPC_WALK_NORM_MODE
    cfg.walk_ablate_eval = os.environ.get("TB_WALK_ABLATE_EVAL", "0") == "1"
    cfg.traj_smooth = float(os.environ.get("TB_TRAJ_SMOOTH", "0.0"))
    cfg.traj_acc = float(os.environ.get("TB_TRAJ_ACC", "0.0"))
    cfg.doremi = os.environ.get("TB_DOREMI", "0") == "1"
    cfg.doremi_eta = float(os.environ.get("TB_DOREMI_ETA", "0.05"))
    cfg.doremi_warmup = float(os.environ.get("TB_DOREMI_WARMUP", "0.05"))
    if cfg.doremi:
        raise RuntimeError(
            "TB_DOREMI=1 is not runnable in this build: the DoReMi training "
            "branch is nested under the freeze-walk transition and never "
            "executes, so it would silently train flat. Refusing instead of "
            "running a misleading experiment.")
    cfg.fp4 = os.environ.get("TB_FP4", "0") == "1"
    cfg.v41_engram = os.environ.get("TB_V41_ENGRAM", "0") == "1"
    cfg.v41_engram_buckets = max(1024, _env_int("TB_V41_ENGRAM_BUCKETS", 32768))
    cfg.v41_engram_rank = max(8, min(cfg.D, _env_int("TB_V41_ENGRAM_RANK", 64)))
    cfg.v41_ced = os.environ.get("TB_V41_CED", "0") == "1"
    cfg.v41_csa2 = os.environ.get("TB_V41_CSA2", "0") == "1"
    cfg.v41_csa2_alpha_start = min(1.0, max(0.0, float(
        os.environ.get("TB_V41_CSA2_ALPHA_START", "0.0"))))
    cfg.v41_csa2_alpha_end = min(1.0, max(0.0, float(
        os.environ.get("TB_V41_CSA2_ALPHA_END", "1.0"))))
    cfg.v41_csa2_ramp_steps = max(0, _env_int("TB_V41_CSA2_RAMP_STEPS", 0))
    # `ste` is the legacy cached FP8/FP4 QLinear implementation.  `te_nvfp4`
    # is an explicit native-Blackwell route and fails closed instead of
    # pretending the old BF16-backward implementation is full FP4 training.
    cfg.quant_backend = os.environ.get("TB_QUANT_BACKEND", "ste").strip().lower()
    if cfg.quant_backend not in ("ste", "te_nvfp4"):
        raise ValueError("TB_QUANT_BACKEND must be ste or te_nvfp4")
    cfg.no_walk = os.environ.get("TB_NO_WALK", "0") == "1"
    # Experimental Neural Interpreter controls.  Any change to these changes
    # model state shapes/behavior and therefore requires a fresh architecture
    # run or an explicit weights-only `TB_INIT_CKPT` transfer.
    cfg.ni = os.environ.get("TB_NI", "0") == "1"
    cfg.ni_mode = os.environ.get("TB_NI_MODE", "final").strip().lower()
    cfg.ni_steps = _env_int("TB_NI_STEPS", 1)
    cfg.ni_trajectory = os.environ.get("TB_NI_TRAJECTORY", "1") == "1"
    cfg.ni_slots = _env_int("TB_NI_SLOTS", 4)
    cfg.ni_rank = _env_int("TB_NI_RANK", 32)
    cfg.ni_width_mult = _env_int("TB_NI_WIDTH_MULT", 1)
    cfg.ni_bridge = os.environ.get("TB_NI_BRIDGE", "0") == "1"
    cfg.ni_bridge_weight = float(os.environ.get("TB_NI_BRIDGE_WEIGHT", "0.01"))
    cfg.ni_chunk_size = _env_int("TB_NI_CHUNK_SIZE", 128)
    cfg.ni_layer = _env_int("TB_NI_LAYER", -1)
    cfg.ni_layers = os.environ.get("TB_NI_LAYERS", "").strip()
    # Lesson Memory Module: disabled unless an episodic behavior dataset and
    # explicit lesson-memory calls are supplied. FineWeb CE alone is not a
    # valid substitute for its write-then-read preference objective.
    cfg.lmm = os.environ.get("TB_LMM", "0") == "1"
    cfg.lmm_rank = _env_int("TB_LMM_RANK", 128)
    cfg.lmm_layers = os.environ.get("TB_LMM_LAYERS", "").strip()
    cfg.lmm_write = os.environ.get("TB_LMM_WRITE", "delta").strip().lower()
    cfg.lmm_lesson_source = os.environ.get("TB_LMM_LESSON_SOURCE", "lookup").strip().lower()
    cfg.lmm_eta = float(os.environ.get("TB_LMM_ETA", "0.25"))
    cfg.lmm_decay = float(os.environ.get("TB_LMM_DECAY", "0.01"))
    cfg.lmm_gate_bias = float(os.environ.get("TB_LMM_GATE_BIAS", "-4.0"))
    cfg.lexical_semantic = os.environ.get("TB_LEXICAL_SEMANTIC", "0") == "1"
    cfg.lexical_semantic_rank = _env_int("TB_LEXICAL_SEMANTIC_RANK", 256)
    cfg.alibi = os.environ.get("TB_ALIBI", "0") == "1"
    cfg.rope = os.environ.get("TB_ROPE", "0") == "1"
    cfg.hf_ckpt = os.environ.get("TB_HF_CKPT", "").strip()
    cfg.max_t = _env_int("TB_MAX_T", 8192)
    cfg.optimizer = os.environ.get("TB_OPTIMIZER", "adamw").strip().lower()
    if cfg.optimizer not in ("adamw", "muon", "hybrid_muon"):
        raise ValueError("TB_OPTIMIZER must be adamw, muon, or hybrid_muon")
    cfg.local_conv = os.environ.get("TB_LOCAL_CONV", "0") == "1"
    cfg.local_conv_rank = _env_int("TB_LOCAL_CONV_RANK", 128)
    cfg.local_conv_kernels = os.environ.get("TB_LOCAL_CONV_KERNELS", "3,7,15").strip()
    cfg.local_conv_every = _env_int("TB_LOCAL_CONV_EVERY", 1)
    cfg.local_conv_depth = max(1, min(3, _env_int("TB_LOCAL_CONV_DEPTH", 1)))
    cfg.local_conv_stage2_kernels = os.environ.get(
        "TB_LOCAL_CONV_STAGE2_KERNELS", "7,15,31").strip()
    cfg.local_conv_stage3_kernels = os.environ.get(
        "TB_LOCAL_CONV_STAGE3_KERNELS", "15,31,63").strip()
    cfg.local_conv_gate_init = float(os.environ.get("TB_LOCAL_CONV_GATE_INIT", "0.0"))
    cfg.local_conv_gate_bootstrap = float(os.environ.get("TB_LOCAL_CONV_GATE_BOOTSTRAP", "0.0"))
    cfg.local_conv_reseed_zero_gates = os.environ.get(
        "TB_LOCAL_CONV_RESEED_ZERO_GATES", "0") == "1"
    cfg.hybrid = os.environ.get("TB_HYBRID", "1") == "1"
    cfg.freeze_walk_at = _env_int("TB_FREEZE_WALK_AT", 0)
    cfg.eval_t2 = _env_int("TB_EVAL_T2", 0)
    cfg.log_every = _env_int("TB_LOG_EVERY", 200)
    cfg.trend_every = _env_int("TB_TREND_EVERY", 100)
    cfg.ckpt_every = _env_int("TB_CKPT_EVERY", 0) or cfg.log_every
    cfg.grad_clip = float(os.environ.get("TB_GRAD_CLIP", "0"))  # 0 = OFF; 1 = legacy; 10 = light
    cfg.mem_mult = float(os.environ.get("TB_MEM_MULT", "20.0"))  # memory LR multiplier (un-freeze)
    cfg.n_eval_batches = _env_int("TB_EVAL_BATCHES", 10)
    # Polyak-Ruppert weight averaging for evaluation (phase 1: diagnostic
    # only — best_val/ckpt selection always uses live weights). The shadow is
    # fp32 (bf16 would quantize 1e-3-scale updates to zero) and is NOT
    # checkpointed: each resume re-enters burn-in (~1/(1-decay) steps).
    cfg.ema_eval = os.environ.get("TB_EMA_EVAL", "0") == "1"
    # LOTUS+ graph memory (see "LOTUS+ graph memory" section above)
    cfg.gmem = os.environ.get("TB_GMEM", "0") == "1"
    cfg.gmem_weight = float(os.environ.get("TB_GMEM_WEIGHT", "0.1"))
    cfg.gmem_every = max(1, _env_int("TB_GMEM_EVERY", 1))
    cfg.gmem_batch = max(1, _env_int("TB_GMEM_BATCH", 16))
    cfg.gmem_writes = max(0, _env_int("TB_GMEM_WRITES", 2))
    cfg.gmem_r = max(1, _env_int("TB_GMEM_R", 4))
    cfg.gmem_lr = float(os.environ.get("TB_GMEM_LR", "1e-3"))
    cfg.gmem_walker_grad = os.environ.get("TB_GMEM_WALKER_GRAD", "0") == "1"
    cfg.gmem_lora_rank = max(0, _env_int("TB_GMEM_LORA_RANK", 0))
    cfg.gmem_trace_every = max(0, _env_int("TB_GMEM_TRACE_EVERY", 100))
    cfg.gmem_sources = [x.strip() for x in os.environ.get("TB_GMEM_SOURCES", "synthetic,wikisql").split(",")
                        if x.strip()]
    try:
        cfg.ema_decay = float(os.environ.get("TB_EMA_DECAY", "0.999"))
    except (TypeError, ValueError):
        raise ValueError("TB_EMA_DECAY must be a number") from None
    if not (0.0 <= cfg.ema_decay < 1.0):
        raise ValueError(
            f"TB_EMA_DECAY must satisfy 0 <= decay < 1, got {cfg.ema_decay:g}")
    cfg.data_dir = os.environ.get("TB_DATA_DIR", "")
    cfg.stream = os.environ.get("TB_STREAM", "").strip()
    cfg.stream_preset = os.environ.get("TB_STREAM_PRESET", "").strip()
    cfg.stream_mix = os.environ.get("TB_STREAM_MIX", "").strip()
    cfg.stream_mix_total_tokens = _env_int("TB_STREAM_MIX_TOTAL_TOKENS", 15000000000)
    cfg.stream_val_source = _env_int("TB_STREAM_VAL_SOURCE", 0)
    cfg.stream_text_field = os.environ.get("TB_STREAM_TEXT_FIELD", "text").strip()
    # Pin this to an HF commit hash for a fully reproducible stream/validation
    # row mapping.  Empty preserves the repository's default revision.
    cfg.stream_revision = os.environ.get("TB_STREAM_REVISION", "").strip()
    cfg.stream_nd = _env_int("TB_STREAM_ND", 32)
    cfg.stream_domain = os.environ.get("TB_STREAM_DOMAIN", "url")
    cfg.stream_buf = _env_int("TB_STREAM_BUF", 1000000)
    cfg.code_stream = os.environ.get('TB_CODE_STREAM', '').strip()
    cfg.code_stream_revision = os.environ.get('TB_CODE_STREAM_REVISION', '').strip()
    cfg.code_data_dir = os.environ.get('TB_CODE_DATA_DIR', '').strip()
    cfg.code_stream_nd = _env_int('TB_CODE_STREAM_ND', 128)
    cfg.code_stream_buf = _env_int('TB_CODE_STREAM_BUF', 500000)
    cfg.code_text_field = os.environ.get('TB_CODE_TEXT_FIELD', 'content').strip()
    cfg.code_repo_field = os.environ.get('TB_CODE_REPO_FIELD', 'repo_name').strip()
    cfg.code_path_field = os.environ.get('TB_CODE_PATH_FIELD', 'path').strip()
    cfg.code_lang_field = os.environ.get('TB_CODE_LANG_FIELD', 'language').strip()
    cfg.code_group_repo = os.environ.get('TB_CODE_GROUP_REPO', '1') == '1'
    cfg.code_lang_filter = os.environ.get('TB_CODE_LANG_FILTER', '').strip()
    cfg.code_share = float(os.environ.get('TB_CODE_SHARE', '0.20')) if cfg.code_stream else 0.0
    cfg.code_share_start = float(os.environ.get('TB_CODE_SHARE_START', '0.10')) if cfg.code_stream else 0.0
    cfg.code_ramp_tokens = _env_int('TB_CODE_RAMP_TOKENS', 100000000)
    cfg.code_val_tokens = _env_int('TB_CODE_VAL_TOKENS', 2000000) if cfg.code_stream else 0
    cfg.code_val_path = os.environ.get('TB_CODE_VAL_PATH', 'code_val_phase_b.bin')
    cfg.code_best_metric = os.environ.get('TB_CODE_BEST_METRIC', 'mix').strip().lower()
    if cfg.code_best_metric not in ('mix', 'fw'):
        raise ValueError('TB_CODE_BEST_METRIC must be mix or fw')
    if not (0.0 <= cfg.code_share_start <= 1.0 and 0.0 <= cfg.code_share <= 1.0):
        raise ValueError('TB_CODE_SHARE_START/TB_CODE_SHARE must be in [0,1]')
    cfg.cot_url = os.environ.get("TB_COT_URL", "")  # CoT OFF by default
    cfg.cot_path = os.environ.get("TB_COT_PATH", "")
    cfg.cot_tokens = _env_int("TB_COT_TOKENS", 2000000)
    cfg.cot_share_max = float(os.environ.get("TB_COT_SHARE_MAX", "0.35"))
    cfg.fw_val_tokens = _env_int("TB_FW_VAL_TOKENS", 0)
    cfg.fw_val_path = os.environ.get("TB_FW_VAL_PATH", "fw_val.bin")
    cfg.mix_val_tokens_per_source = _env_int(
        "TB_MIX_VAL_TOKENS_PER_SOURCE", cfg.fw_val_tokens)
    cfg.mix_val_path = os.environ.get("TB_MIX_VAL_PATH", "mix_val")
    if cfg.mix_val_tokens_per_source < 0:
        raise ValueError("TB_MIX_VAL_TOKENS_PER_SOURCE must be non-negative")
    cfg.ckpt_path = os.environ.get("TB_CKPT_PATH", "ckpt_ongoing.pt")
    cfg.init_ckpt = os.environ.get("TB_INIT_CKPT", "").strip()
    cfg.init_allow_partial = os.environ.get("TB_INIT_ALLOW_PARTIAL", "0") == "1"
    cfg.phase_ckpt = os.environ.get('TB_PHASE_CKPT', '').strip()
    cfg.phase_structural = os.environ.get('TB_PHASE_STRUCTURAL', '0') == '1'
    cfg.phase_ce_tolerance_pct = float(os.environ.get('TB_PHASE_CE_TOLERANCE_PCT', '2.0'))
    cfg.phase_reference_ce = float(os.environ.get('TB_PHASE_REFERENCE_CE', 'nan'))
    cfg.phase_allow_unchecked = os.environ.get('TB_PHASE_ALLOW_UNCHECKED', '0') == '1'
    cfg.phase_gate_batches = max(1, _env_int('TB_PHASE_GATE_BATCHES', 4))
    cfg.arch_phase = os.environ.get('TB_ARCH_PHASE', 'phase_b').strip() or 'phase_b'
    cfg.phase_ablate_eval = os.environ.get('TB_PHASE_ABLATE_EVAL', '0') == '1'
    cfg.v41_ablate_eval = os.environ.get('TB_V41_ABLATE_EVAL', '0') == '1'
    cfg.phase_ablate_batches = max(1, _env_int('TB_PHASE_ABLATE_BATCHES', 2))
    cfg.semantic_wd = os.environ.get('TB_SEMANTIC_WD', '0') == '1'
    if os.environ.get("TB_DRAFT", "0") == "1":
        raise RuntimeError(
            "TB_DRAFT=1 needs DraftNet, removed 2026-09-30 (no checkpoint "
            "tensors; speculative path untested in this lineage).")
    cfg.draft = False  # hard off; TB_DRAFT_EVERY/LR/K/MAX_T no longer parsed
    cfg.profile = os.environ.get("TB_PROFILE", "0") == "1"
    if "--config-only" in argv:
        print(
            "CONFIG_OK "
            f"step={cfg.steps} D={cfg.D} L={cfg.L} heads={cfg.n_head} "
            f"context={cfg.t0}/{cfg.t1} accum={cfg.accum} stream_nd={cfg.stream_nd}",
            flush=True,
        )
        return
    train(cfg)


def _smoke_walk_forward():
    """Debug smoke: shows what goes through walk, no training, no HF."""
    import torch
    print("SMOKE: 8M HIER walk forward/backward check (no stream, no HF)", flush=True)
    os.environ.setdefault("TB_WALK_SUBSPACE_RANK","32")
    os.environ.setdefault("TB_WALK_HIER_EVIDENCE","1")
    os.environ.setdefault("TB_WALK_DATA_TRAJ","1")
    V = 64
    # direct WalkGPT without Cfg/DataMix
    model = WalkGPT(vocab=V, D=128, L=2, n_head=4, mult=2, max_t=64, no_walk=False, no_mem=True)
    model = model.to(DEVICE)
    model.train()
    B,T = 2,64
    idx = torch.randint(0, V, (B,T), device=DEVICE)
    tgt = torch.randint(0, V, (B,T), device=DEVICE)
    print(f"SMOKE: model {sum(p.numel() for p in model.parameters())/1e6:.2f}M walk={type(model.blocks[0].walk).__name__ if model.blocks[0].walk else 'none'}", flush=True)
    # forward
    logits, loss = model(idx, tgt)
    print(f"SMOKE: logits {tuple(logits.shape)} loss {float(loss):.4f} lm_ce {float(model.lm_ce) if hasattr(model,'lm_ce') else float(loss):.4f} walk_aux {float(getattr(model,'walk_aux_loss',0)):.4f}", flush=True)
    # check walk internals
    w = model.blocks[0].walk
    if w is not None:
        print(f"SMOKE: walk rank={getattr(w,'rank', '?')} k={getattr(w,'k_vectors',1)} data_traj={getattr(w,'data_traj',False)} out_proj std {w.out_proj.std().item():.6f}", flush=True)
        if hasattr(w, 'last_view') and w.last_view is not None:
            print(f"SMOKE: last_view {tuple(w.last_view.shape)} mean {w.last_view.float().mean().item():.4f} std {w.last_view.float().std().item():.4f}", flush=True)
    # backward
    loss.backward()
    # grad norms per group
    for n,p in model.named_parameters():
        if p.grad is not None:
            gnorm = p.grad.norm().item()
            if "walk" in n.lower() and gnorm>1e-3:
                print(f"SMOKE: grad {n} {gnorm:.4f}", flush=True)
                break
    print("SMOKE: forward+backward ok (causal, no future leak if diff 0.0 earlier)", flush=True)
    # causality check
    model.eval()
    with torch.no_grad():
        causal_a = idx[:1].clone()
        causal_b = causal_a.clone()
        causal_b[:, T // 2:] = torch.randint(0, V, causal_b[:, T // 2:].shape, device=DEVICE)
        logits_a, _ = model(causal_a)
        logits_b, _ = model(causal_b)
        model_causal_diff = (logits_a[:, :T // 2] - logits_b[:, :T // 2]).abs().max().item()
        print(f"SMOKE: whole-model causal diff before changed suffix {model_causal_diff:.6f} (must be 0.0)",
              flush=True)
        if model_causal_diff != 0.0:
            raise RuntimeError(f"future-token information leak detected: max diff {model_causal_diff}")
        full_logits, _ = model(causal_a)
        with temporarily_disable_all_extensions(model):
            ablated_logits, _ = model(causal_a)
        restored_logits, _ = model(causal_a)
        restore_diff = (full_logits - restored_logits).abs().max().item()
        ablation_delta = (full_logits - ablated_logits).float().square().mean().sqrt().item()
        print(f"SMOKE: extension ablation RMS delta {ablation_delta:.6g}; "
              f"post-context restore max diff {restore_diff:.6g}", flush=True)
        if restore_diff != 0.0:
            raise RuntimeError(f"extension ablation failed to restore model state: max diff {restore_diff}")
        # we test walk directly
        if model.blocks[0].walk is not None:
            w=model.blocks[0].walk
            # need to pass through walk correctly: it expects x shape B,T,D
            # create dummy x
            dummy=torch.randn(1,16, w.D, device=DEVICE, dtype=torch.bfloat16)
            out1=w(dummy)
            dummy2=dummy.clone(); dummy2[0,8:]*=2
            out2=w(dummy2)
            diff=(out1[0,:8]-out2[0,:8]).abs().max().item()
            print(f"SMOKE: walk causal diff early after future perturb {diff:.6f} (should 0.0)", flush=True)
    print("SMOKE: done", flush=True)

# --- REMOVED 2026-09-30 (lean cleanup): CompositionWalkProbe + runner ---
# What they were: inverse-perception composition probe (fixed span pooling vs
#   learned ordered composition), reachable only via --composition-walk-probe.
# Why removed: standalone toy diagnostic (64-dim probe), no references from
#   training, eval, or any other probe. The CLI flag is gone with it.
# To restore: recover from train_walker_1b_continue.py.bak-20260930.
if __name__ == "__main__" and '--composition-walk-probe' in __import__('sys').argv:
    raise RuntimeError(
        "--composition-walk-probe needs CompositionWalkProbe, removed 2026-09-30 "
        "(standalone toy diagnostic, zero train-path references).")
if __name__ == "__main__":
    if os.environ.get("TB_SMOKE","0")=="1":
        _smoke_walk_forward()
    else:
        # debug prints when TB_DEBUG=1
        if os.environ.get("TB_DEBUG","0")=="1":
            print("DEBUG: TB_DEBUG=1 walk verbose on", flush=True)
            # monkey patch WalkGPT forward to log walk internals
            _orig_walk_forward = None
            try:
                _orig_walk_forward = HierarchicalEvidenceWalkAdapter.forward
                def _debug_forward(self, x, *a, **kw):
                    out = _orig_walk_forward(self, x, *a, **kw)
                    # log walk norm and grad will be in training loop
                    if hasattr(self, 'last_view') and self.last_view is not None:
                        print(f"DEBUG: walk view {tuple(self.last_view.shape)} pred_loss {float(getattr(self,'pred_loss',0)):.4f}", flush=True)
                    return out
                HierarchicalEvidenceWalkAdapter.forward = _debug_forward
            except Exception as e:
                print(f"DEBUG: patch failed {e}", flush=True)
        main()
