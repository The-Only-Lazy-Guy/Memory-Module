# Graph Node Memory module (GNM) — current state

**How this file works.** One page of current state; every section has one job and is updated IN PLACE:
a new result = one line in §8 (log) + its row in §2 (status) changed. Old text is not appended here: a superseded
point is struck (~~like this~~) and moved to `docs/GNM_history.md` when the section gets long (§11 there holds the
10-20 .. 10-30 build log verbatim). Naming: **GNM** = this memory module; **Walker** = only the 1B LM (GPT-2 stands in).

| § | job | update when |
|---|---|---|
| 1 | goal + finish line | the user changes the goal |
| 2 | status board: one row per capability | every result |
| 3 | who decides what: model / helper rules / dense supervision | a rule or supervision changes |
| 4 | current config, how to run, diagnostics | a default or a tool changes |
| 5 | design (authoritative, user) | the user changes the design |
| 6 | lessons that still apply | a lesson is learned |
| 7 | next (ordered) | every decision |
| 8 | log: one line per run / test, newest first | every run / test |
| 9 | open decisions / rejected | a decision |

---

## 1. Goal and finish line

**Goal (user):** memory that learns / remembers / understands a NEW thing from one exposure and reasons over it in
long chains; every decision neural (procedures and tools are nodes); supervision = ~50 natural-language step-by-step
demos per kind early, then outcome only; no engineered helper inputs.

**Done = three proofs hold for every task kind** (user 2026-10-30: no brute-force validation; fast diagnostic tests
are fine, full trainings are not the check). Deadline **2026-11-13**; a proof still failing then is logged as a
known limit.

| proof | claim | shown by (fast, §4) |
|---|---|---|
| P1 existence | some setting of GNM's parameters solves the kind exactly | gold plugged into each decision point -> the rest of the pipeline is exact (eval: "exact given program") |
| P2 length | k passes are right when each pass is right | the input of a pass has the same type at every depth (closure, read off the design) -> chain = pass^k |
| P3 reachability | training reaches the P1 solution and stays | gradient paths (`--diag-grad`: every decision gets an outcome signal) + stability test (start AT a good checkpoint, train at full lr, must not leave) |

Not provable: wording never seen — stays a held-out check.

---

## 2. Status board (newest diagnostics; numbers = exact, 96-288 questions, fresh-world eval)

| capability | best so far | now (st4a ckpt) | P1 / P2 / P3 finding | next |
|---|---|---|---|---|
| memory: old facts, one exposure (after 0 / 100 frames), corrections | .99 / 1.0 / .99 / .99 (st4a E200, notes FREE = D1 off) | st4a: .99 / .95-1.0 / .99 | P3 holds at lr 5e-5 (st2-st8) | — |
| "before" (older version) | 1.0 (st4a E200) | 1.0 | — | — |
| dependents (copy follows its source) | 1.0 | 1.0 (st4a) | — | — |
| compositions (unseen op pairs, 2 ops) | 1.00 (st6 E200) | .93 (st4a) | question negatives erode it over time (st3a .93 -> .80) | q-neg <= .3 |
| long chains (4-6 ops, trained 1-3) | .58 (dense E); outcome-only .57 (st4b) | .46 (st4a) | right program -> answer 1.000; misses = span pointer. **P2 fails by design** (outermost span grows with depth) | `--inner` built 11-01 (innermost-first); in1 / in2: pointer still outermost after 50 demos at 5e-5 |
| rules from one exposure: add / trained wording / changed / set-only | .91 / 1.0 / 1.0 / .70 (st4a E200; st3a) | flips 0 <-> ~1 between runs | pointer reads a NEWER FOREIGN rule (recency, weak key match); a foreign 'add' rule is usually harmless in training (add:set 2:1) -> reinforced | st8: set:add 1:1 + proc negatives .3 (E100 1.0 / .86 / .86) |
| coexisting values (keep both) | 1.0 (st3a/b/d, st4a, st6 at E100: outcome-only, frame negatives) | 0 (lost by E200 in every run) | only 1-3 credited multi rows / 25 steps: signal starved | more multi credit (same-step questions) |
| tools / procedures as nodes | object tools: exact (Python op, model decodes / encodes) | same | neural tool network plateau .2-.3 (history §11.3 TN1) | design open (§9) |
| outcome-only training (after demos) | stable at lr 5e-5 (st2-st8); D1 off free (st3d/st4) | — | AdamW moves every head ~lr per step | — |
| speed | 5 s / step at 3,000 nodes (was 26) | — | — | — |
| on the 1B | the 1B SAYS GNM's answers .96-1.00 (`--say-ckpt`); swap-in-place failed | — | — | after the core passes |

---

## 3. Who decides what (audit 2026-10-30; user: "did you include a patch that is a helper?")

**Neural (the model decides):** every address (read / tool / HALT / SUB / NEW), tool operands, sub-question span,
write gate, replace vs attach, NEW gate, which procedure and which op, the look's correction, label / value /
relation vectors and deltas, per-question recency, r edges, forgetting, say.

**Helper RULES still active (override or restrict the model; each added after a measured failure; to go one at a time):**
| rule | flag (default) | what it does |
|---|---|---|
| H1 confidence fallback | `--conf .5` (on) | replace / attach with address mass < .5 on the target's cluster -> NEW |
| H2 identity guard | `--id-guard 1`, `--cluster-cos .95` (on) | replace into a node whose label is not ~identical -> NEW; + own-label bonus |
| H3 consolidation | always | a replace turns the other near-identical copies into versions |
| H4 notes edit what they read | always | a note's target = a node it read; its key = best match of its read nodes |
| H5 action masks | always | questions never write; frame rows no tools / HALT; notes NEW only after a tool output; read-once for questions; frames see only older nodes |
| H6 exact object arithmetic | always | the op itself is Python (the model decodes the operands, picks the op, encodes the result) |

**Dense supervision still active (breaks "~50 demos, then outcome"):**
| item | flag (default) | note |
|---|---|---|
| D1 correction notes teacher-forced EVERY step | `--note-demo-budget N` (built 10-31; -1 = old) | was not budgeted; `--diag-grad`: notes pushed HALT hard (3.3 on a collapsed ckpt). With the budget spent, free notes are credited by this step's questions about the EDITED fact (frame-RL path): local 25-step smoke on all16_s1 -> notes credited .91, computed exactly 1.00, notes no longer reach HALT |
| D2 r-edge demonstration | `--w-ref-demo 1.0` | dense label for dependence edges |
| D3 frame RL reward attribution | `--rl-frames` | reward = outcome of questions about the row's fact; WHICH fact is read from gold provenance (bookkeeping, not a label of the decision) |

**My fixes 2026-10-28 .. 10-30 — none is a helper:**
| fix | kind |
|---|---|
| speed: unthrottle, TF32, KV-cache decode, linear backward (pad / take), source bank, EOS early stop | compute only; outputs and gradients identical |
| `--rl-say` (say head learns from its own EXACT answers) | training signal (outcome) |
| `--rl-neg` (weight on negative advantages) | loss math |
| `--diag-grad`, `--lr-const`, eval-only | diagnostics; change nothing by default |
| lower lr for the outcome phase (proposed) | optimization |
| innermost-first (proposed) | process design; the model still finds the innermost sub-question itself. Borderline part: the result REPLACING the span in the question must be the model's own write (a node it points at), not a text rewrite by code |
| `--proc-top 1` (earlier) | removes a hand score (default procedure 25.5); restricts nothing the model learned |

Not helpers: discretizing the model's own probabilities at execution (g > .5, argmax), numeric guards (bounded gate
logits, register norm cap 50), versions on replace, running key mean.

---

## 4. Current config, how to run, diagnostics

**Code:** one file `gnm.py` (user: no new GNM files; old ones in `archive/gnm/`). Weights only on `E:/WEIGHTS/gnm_runs`.
**Best checkpoint:** ~~`rd2b`~~ (lost 10-31: MoLab box reset, never pushed) -> `rd2c` (same recipe, rebuilding; MoLab `/marimo/ckpt`); all16_s1 on E: and the ckpt branch.
**Current config (rd2b):** `--p-rule 0.6 --p-coexist 0.6 --co-keep 5 --multi-max 4 --corrections 2 --forget-q 0.2
--recurse 1 --long-ops 4,6 --vslots 128 --train-ops 1,3 --tool-nodes 1 --confusion 1.0 --proc-demo 1.0 --proc-op 1.0
--proc-top 1 --demo-budget 50 --frame-demo-budget 50 --rl-frames 1 --rl-say 1 --rl-neg 0`.
**MoLab:** shell cell; `cd /marimo/mm && bash lab/run.sh <name> <init ckpt|-> <args>` (logs `/marimo/out`, weights
`/marimo/ckpt`); `python lab/mon.py '/marimo/out/*.log' 1` = one line per run.

**Diagnostics (minutes, no full training):**
| test | command | time | answers |
|---|---|---|---|
| eval-only | `--init-full <ckpt> --steps 0 --eval-q 96` | ~1 min (MoLab) | every task + "why wrong" per question (P1: exact given program) |
| gradient paths | `--diag-grad 1 --steps 1 --skip-eval0 --demo-budget 0` | ~1 min | which loss reaches which head; "DEAD" / "demo-only: drifts" (P3) |
| stability | `--init-full <good ckpt> --lr-const 1 --steps 60 --eval-every 60 --demo-budget 0` | ~6 min | does training keep a good solution (P3) |
| speed | `--trace 2 [--prof-steps N]` | ~2 min | per-part step time, backward op table |

---

## 5. Design (authoritative, user)

```
input frame = ONE JSON = everything the model sees now (NOT live; dropped after encoding)
  { "context": …, "question": …, "image 1": <pixels>, "chess": <board>, "pc state": {…}, <any key>: <any value> }
        │  smart encoder: every key–value pair → one fully latent node
        │  (text through Walker's early blocks + learned pooling; images / boards: own encoders into the same space)
        ▼
LIVE graph memory = the world model (global understanding + memorization), ONE store for the model's whole life,
never rebuilt. Each frame attaches to it / updates it. Nodes: { v, l, r, usage, time }. Tool node: { l, r, network }.
        ▲  │
  query │  │ address → read · apply tool · write · halt   (every action = a knob the model's own state decides)
        │  ▼
Walker's loop, pass p: pass state s_p → query → activation over node labels (+ relation edges) → one action →
the result is fed back to Walker as latent tokens → … → HALT → Walker's own head says the answer.
```

### 5.1 What a node has
| field | kind | meaning |
|---|---|---|
| **v** value | vector (+ per-token value slots) | the information (slots: so Walker can say multi-token values) |
| **l** label | vector | **learned**: what this is, how / when to use it — the only thing queries match |
| **r** relation | vector | learned: how this node relates to others (context-dependent links, grouped / dependent updates); relation edges between nodes come from r (§5.3) |
| g gain | scalar | write gate (the model's choice how strongly a node is stored) |
| usage, time | scalars | usage count and time written: capacity / eviction / recency |

~~**Tool node:** {l, r, network}. Applying it runs the network on the latent (label, value) of its arguments; the
output is written back as a new node. Tools are meant to be **generic vector modules** (the label is the instruction
for how the tool is used), not fixed calculators (§5.3).~~
**Tool node (user 2026-10-20, restated 2026-10-26: "we need an encoder/decoder to transform a node into -> number /
strings etc"):** a node {l, r} in memory. Nodes are latent; an **ENCODER / DECODER** turns a node into an OBJECT
(number, string, ...) and back: DECODE = the model's own head says the node (parsed into the object), the tool
operates on the objects, ENCODE = the result becomes an ordinary node through the same encoder. Built as object
tools (history §11.3, 2026-10-20). Neural "vector module" execution was tried twice and rejected (history §11.3 #8: whole numbers
plateau .14-.31; digit encodings indistinguishable). The decoder is LEARNED, not code: a wrong decode is punished where it happens —
round trips (encode -> say, every step), tool outputs and answers said, and (2026-10-26, user: "we route to punish
if the model doesn't decode properly") every tool OPERAND's decode said as its true value (`--decode-say`).

### 5.2 One loop pass = ADDRESS → READ → (WRITE) (spec 2026-10-20, user: "read and update in a single pass")

**Goal (the stop condition, user):** the model learns / remembers / understands any NEW thing without training on
it (or after at most one step): shown once, usable at once. Training only teaches HOW to read, write and update,
never the content. The algorithm is redesigned until the one-exposure test (below) passes.

**Rows.** Everything the loop does is a row with registers (Walker's working memory: latent copies, never changed by
a write — the current context cannot be lost):
- **frame row** — one pair of an incoming frame (the candidate); register 0 = the candidate;
- **note row** — a text note (correction, later: instructions); it finds its note node like a question;
- **question row** — a question node; it finds its question node first (pass 0), then reads / applies tools.
Within a step rows run in time order: frame rows, then note rows, then question rows.

**One pass of a row** (one Walker call):
1. Walker reads the row's registers (latent tokens) → state `h`.
2. **Address** — ONE softmax over [memory nodes ⊕ NEW ⊕ tool labels ⊕ HALT]:
   node score `τ·cos(q − μ, k − μ) + log gain + keep + age(Δt) + w_rel·cos(q, r)`; NEW / tools / HALT score
   `τ·cos(q − μ, label)`. `q` from `h` and the key of the last register. **μ = running mean of all keys written**
   (not the mean of the current memory: reading must not depend on how full memory is — the first-frame bug).
   No spreading activation (run 9 trained it to ≈ 0; chains go through passes, dependencies through propagation).
3. **Read** — a node: its current value (before any write of this pass) becomes a register. A tool: runs on two
   registers the row points at, output = a register. HALT: the row ends.
4. **Write** (frame and note rows) at the SAME address, from the same `h`:
   - **source** = the row's newest register that did not come from memory (the candidate; a tool's output);
   - **gate** `g = σ(w·h + a·cos(x_v, v_read) + b·cos(x_k, k_read) + c)` — the cosines compare the source with
     what is stored there (same → nothing to do; different → an update), without a second Walker call;
   - **mode** `u` (same form): **replace** or **attach**.

| address | write | effect |
|---|---|---|
| node j | no | plain read |
| node j | replace | j keeps its identity (label, relation, edges, usage); its value source becomes the source's; the old value moves to a **version** node (time-stamped, linked); dependents follow (propagation, §5.3) |
| node j | attach | the source becomes a new node linked to j by r (coexisting values) |
| NEW | yes | the source becomes a new node (store) |
| NEW / HALT | no | the source is dropped (ignore) |

   - **HOW to write = a procedure read from memory (user 2026-10-20: "make the model memorize a way to edit").**
     (Code since object tools: the procedure vector picks an exact operation `Wop` [copy | add sub max min] on the
     decoded values; the ToolBank network is not conditioned on it — audit 2026-10-25, history §11.3.)
     A frame row also points (same pass) at a procedure: a node in memory (a rule that arrived as text, e.g.
     `{"rules": {"visits": "later visits values are amounts to add to the current visits"}}`) or the default
     procedure. The procedure vector conditions the shared write network (the ToolBank network, label = the
     procedure): **copy** the source (exact tokens) or **compute** a new value from (source, stored value), and the
     mode. A new way of editing = one new node, usable at once. Limit: the network only executes operations it can
     already do (copy, add, …); new operations from primitives = stored programs (level 2, later).
   A frame row = INTEGRATE (address over the WHOLE memory, not a top-k), a note row = EDIT (read note → read target
   → tool computes the new value → address target + write). The read-once rule (a node already in the registers is
   not read again) holds for question rows; write rows may address a node again to write ("look, then write").
   Question rows do not write in this build (their writes could only be judged by later rows; later: multi-turn).
5. **Forget:** ~~keep knob (usage, version status) + decayed activation → eviction at capacity~~ **the model's
   forget head** (user 2026-10-25: "everything from the model's decision, nothing automatic"): a keep logit per node
   from its label + observations (version, usage, age, activation, write gate, state / question); over capacity the
   lowest go. Trained in-step: the decision acts before the step's questions (forgotten = empty value,
   straight-through keep weight), and facts it chose to forget are asked (`--forget-q`) — forgetting something
   useful is blamed at once. The keep knob no longer enters read scores.

**Training (no write demonstrations).** Read actions of question / note rows keep their demonstrations (CoT as
imitation; a note's write address is part of its procedure). Gate and mode are never demonstrated:
- frame rows are free-running; their writes act SOFTLY on memory in the same step (source kept with its write
  probability, a replaced target fades by `p·g·u`, dependents move toward the change), so the step's later rows
  read the written memory and gradient reaches address, gate and mode; after the backward pass the hard decisions
  are executed (argmax address, `g > ½`, `u > ½`, confidence below `--conf` → NEW instead of overwriting);
- the step's questions expose a write's effect **inside the step** (a write whose damage shows up only later never
  gets blamed): **written** (facts of this step's frames), **collateral** (facts of the nodes a write addressed,
  their "before" values, their dependents), **retention** (random old facts);
- a small storage cost per new node (memory is finite).

**Two input streams (user):** state frames (any JSON the model sees; never asked about — it keeps what it judges
useful) and pending items (what is to be remembered: facts, updates, corrections) as their own frames; questions ask
about pending items.

**The one-exposure test (the goal metric; no gradient on the test items):** frozen, on new parts of the world:
the very first frame into an empty memory; new facts, value updates, text corrections, coexisting values,
dependents, never-seen attribute and entity names; each shown ONCE, asked immediately and after 10 / 100 frames of
state noise; "what was it before?". Variant: after ONE gradient step on the new items.

**Where it can fail (and the mitigation):**
1. harm visible only many steps later (a dependent asked much later) → collateral questions cover direct targets
   and dependents; N-step rollouts later;
2. everything goes to NEW (safe, duplicates) → a stale duplicate answers "current" questions about updated facts,
   that loss pushes toward replace;
3. the gate sees the stored value only through two cosines → a row may take a second pass (look, then write);
4. wrong target → confidence fallback to NEW, versions to undo;
5. soft (training) vs hard (inference) → the hard decisions are what later steps see; soft only within the step;
6. frame rows cost one Walker call per pair (as INTEGRATE).

**Built on top of this spec (2026-10-20..25; details: history §11.3):**
- **object tools:** add / sub / max / min / copy work on values the model's own head SAYS (decoded with an end token),
  computed exactly, re-encoded as ordinary token nodes; the latent tool network stays for `edit`;
- **look before commit** (note rows): a write is first a proposal; a look pass sees the target as it would be and a
  zero-init delta head (`Wlook`) adjusts gate / mode / NEW; trained by a commit TARGET from the model's own answers
  (old vs new value NLL);
- **question text in Walker's register** (`--q-slots 128`, `reg`): memory keeps 16 value slots per node;
- **recursion** (`--recurse`): a SUB action hands a span of the question (pointer heads over the question register)
  to a child row whose answer returns as a register — every row <= 1 operation, depth up to `--max-depth` 6;
- **duplicate control:** label clusters (cos > .95) compete as one; a replace turns the other copies into versions;
  frame rows also address with their own label; frame rows cannot HALT.

### 5.3 Relation edges (r) and dependence edges (references)
**(user 2026-10-25, authoritative): r is THE node-to-node relation — dependence, relative definitions ("3 more than
X"), links, and more; the model creates and manages it.** Build (2026-10-25): a node's r = its own content read like
a question (Walker re-reads its value text alone -> `Wrr`) + its pooled pair state (`r_pool`, 0 at init) + the
model's own write deltas (capped at .3 x the label's size); attach points the attached node's r at its target; an
edge i -> j = r_i addressing node j's label (cosine x temperature, learned version bias) or "none". Dependence
(propagation, soft and hard) uses these edges; the relation-similarity edges and the separate reference heads below
are gone. Signal: the task (stale dependents asked after their source changed, through soft propagation) +
optional demonstration (`--w-ref-demo`; split planned: 1 vs 0). Not yet: relative definitions (the world only has
"same as"; the propagation op from r).
- relation-similarity ~~`softmax((ea·r_i)·(eb·r_j)/√D)`~~ `softmax(τ_r · cos(ea·r_i, eb·r_j))` (2026-10-25: raw dot
  products of exploded r made one global hub); model deltas on r (and k, v) capped at .3 x the vector; links nodes
  that belong together without any rule-made pointer; an attached node is linked to its target by r.
- ~~dependence edges: a node's value predicted from its relation-linked neighbours; ... (× the gate `w_prop`)~~
  **dependence edges = REFERENCES (2026-10-25):** a value that mentions another fact ("12 (same as the cost of
  Batha)") is re-read by Walker ALONE, through the question path (+ own pooling score) -> a query over all labels;
  edge = P(has a reference) (sigmoid) x which node (softmax; learned bias against version nodes: "same as X" = X now).
  Taught by demonstration (the generator knows the source). When a node is replaced, the nodes whose reference
  points at it follow — one damped step, versioned (the user's purpose: when an ancestor is updated, its dependents
  follow). Used by WRITES (propagation), not by reads.

### 5.4 Rules (user)
- The input frame is any JSON state; the question is one of its pairs → an **ordinary node**. Walker is never handed
  the question text.
- **ONE world, in training too.** The model learns to assemble the world from small pieces given each step; never a
  fresh world per episode. Memorizing the world in the weights is prevented by the world changing (new entities,
  updates) and checked by counterfactual and never-asked facts.
- Matching is by **learned label vectors**; no hand rules (word overlap, copy pointers).
- **Every action is the model's knob** (write gate, read / tool / halt choice, overwrite gate, edit gate).
- **CoT is a demonstration** that supervises the loop's actions (imitation); Walker never generates CoT. Highest
  goal: a CoT step like "substitute x = 3 in x²" is carried out inside the memory with tool nodes.
- Walker says the answer with its own head. Eval: hold out values / assignments, never unseen names.
- KISS; combined runs, not sweeps; document every run; uploads via `molab_upload/` (GNM) or `push/` (1B resume);
  never connect MoLab to the local computer.

---

## 6. Lessons that still apply

- Questions and documents encoded in different contexts need their own standardization statistics (.03 → .89).
- Addressing needs a direct signal: the answer loss alone memorizes without learning where to read; the
  demonstrated action (which node) is that signal — it unlocked the 1B (.04 → .98).
- Value slots must line up with the answer tokens (slot t = value token t); a one-token shift gave .00 on long
  values. A node must carry its whole value (64 slots: long questions were cut at 32).
- Scalar knobs (age term, gates, edge weights, τ) need their own learning rate; at the base lr the first read (find
  its own question) stayed at .17.
- The training questions must cover recent facts too, or the read learns "young nodes are never the answer".
- Evaluate exactly like training (questions answered as they arrive, eviction, age ranges); three eval-only
  artefacts looked like model failures.
- What was read is not re-read (the graph-era fix; without it the loop re-reads its own question).
- Supersession must not treat "same entity, other attribute" as a newer version (open; a learned projection made it
  worse).
- Speed: one gather per pass (per-row picks cost a full-size gradient each), flat segment pooling, activation
  checkpointing per pass (6 GB card).
- Outcome-only gradients are mostly noise; with AdamW every head moves ~lr per step whatever its signal. A demo-sized
  lr (5e-4) destroys a good solution in 60 steps; 5e-5 keeps it (2026-10-30). A run that looks good at its end may
  only be riding a decayed cosine lr.
- A head that no outcome loss reaches drifts once the demos are spent (the say head in rd1). Check with
  `--diag-grad` BEFORE a run, not after.
- Sampled recursion needs a compute budget per root question (a free SUB at every pass, to depth 6, is an
  exponential tree; rd2b got lucky). A good checkpoint leaves the box the same day (`lab/keep.sh`): a reset lost rd2b.
- A pointer that must pick a span whose size grows with the problem cannot generalize to longer problems, however
  long it trains (P2).

---

## 7. Next (ordered)

1. ~~Confirm lr 5e-5 (st2)~~ done: holds 200 steps with rl-neg 0 -> outcome-phase lr 5e-5 becomes the default.
2. ~~Rules fall~~ diagnosed + fixed (11-01): the pointer read a newer FOREIGN rule; st8 recipe (`--rule-set-frac 0.5
   --rl-neg-proc 0.3`, q-neg .3, frame-neg 1, notes free) keeps rules 1.0 / 1.0 / .85 through 200 steps. Default now.
3. Long chains: ~~innermost-first by the current span pointer~~ built (`--inner`) but not learned (in1-in7). Next: a
   span pointer that can express "an operation with no operation inside" from its own reading (no depth input),
   e.g. the start and end chosen jointly / per-token context from the loop, then re-test.
4. Coexist: learned by outcome (E100 1.0 in 5 runs) but lost by E200; ~~co-ask~~ rejected (hurt storage). Next: why the
   multi frames get no credit (none credited in co1) -> fix the credit path, not the data.
5. ~~Strip D1~~ done (`--note-demo-budget 0`, no loss). D2 next; then the helper rules H1-H5 one at a time, each checked
   by the diagnostics (§4), not by full runs.
6. Then: the 1B (§2 last row), real input frames (PC-state JSON, STEM, images).

---

## 8. Log (one line per run / test, newest first; older entries: `docs/GNM_history.md` §11)

| date | name | what changed | result | verdict |
|---|---|---|---|---|
| 11-01 | kp1 / kp2 | st8_last + st8 recipe + multi0 fix + `--proc-keep 1` (a multi key's KEEP rule announced, read by the procedure pointer) / + `--co-credit 1`, 150 steps | E75: rules 1.0 / 1.0 / 1.0 vs .55 / 1.0 / 1.0, long .44 / .45; **E150: rules .82 / .84 / 1.0 vs 1.0 / 1.0 / 1.0**, comp .91 / .91, long .32 / .40, before .91 / .99, old .94 / .89, dependents 1.0, corrections .99 / .98, set .34 / .24, coexist both-held 0 / 0 (2nd value mostly DROPPED: decided drop 11-14 of 8 frames' pairs) | **kp2 = best rules with all frame demos long spent; kp2_last kept.** coexist: with `--multi-max 4` most training coexist events are 3rd / 4th values, never asked ('both' needs exactly 2) -> kp3 / cc3 with `--multi-max 2` |
| 11-01 | cc1 / cc2 | st8_last + st8 recipe + **multi0 fix** (the multi snapshot was taken AFTER the frames were drawn -> a multi fact's new value was never in `now`, never asked in its step) / + `--co-credit 1` (both-question, no gradient, credits only the multi frame) | E75: rules .39 / .22 / .27 vs .87 / .69 / .78; E150: rules .76 / 1.0 / .73 vs .81 / 1.0 / 1.0, set .66 / .20, long .29 / .32, comp .89 / .90, coexist both-held 0 / 0 | co-credit helps rules; multi frames now credited (cc2: 3-5 per 25 steps, reward ~0) |
| 11-01 | stab8 | P3 stability: st8_last, st8 recipe, ALL demos off (frame / note / question), 100 steps | E50: rules .48 / 1.0 / .76, long .33, comp .91; E100: **rules 0 / 0 / 0**, old .93, before .97, comp .89, long .33, dependents 1.0, corrections .99, coexist both-held .125 (the 'coexist exact 1.0' is over the ONE fact whose both values were held: misleading, read 'both values held') | **P3 fails for rules (st8 alone).** ~~rules and coexist flip together~~ (vacuous metric). ~~st8 E200 rules propped by demos~~ (frame demo budgets count ROWS: spent by ~step 30-65). **P1 fails for coexist without a cue: a fresh world re-draws which keys are multi, so a coexisting 2nd value looks exactly like an update** -> `--proc-keep` (kp) |
| 10-31 | rd2c (1st) | rd2b recipe from all16_s1 on the fresh box | 12 s / step to s125, then ~15 min / step (CPU-bound): free rows sample SUB at every pass of every child to depth 6 = exponential tree | killed; `--sub-budget 8` (per root question, whole tree) added; rd2c relaunched (box + local `rd2c_loc` on E:), chain pushes rd2c_last to the ckpt branch (`lab/keep.sh`) |
| 10-31 | nd0 (local smoke) | `--note-demo-budget 0` (D1 off) on all16_s1, 25 steps at 5e-5 + `--diag-grad` | runs; free notes credited .91 (46), computed exactly 1.00; `notes` no longer reaches `loop.halt` | D1 removable; goes into the next combined run |
| 11-01 | st5 | st3d recipe + `--rl-critic 1` (learned frame baseline) | E100 / E200: comp .98 / .89, long .55 / .49, rules 0 / 0, coexist 0 / 0, dependents .68 / .36, corrections .97 / 1.0 | ~~critic~~ rejected (no rules, no coexist) |
| 11-01 | in7 | in5 + slot key projection `loop.Wsk` reset / in the span lr group | E100: old .83, comp .78, long .12; misses 'wrong sub-question span @1' | **innermost-first not learned in any variant (in1-in7: long .09-.34 vs outermost .43-.57)**: one query dot-product over token states cannot find 'the innermost operation' from 50 demos; needs a pointer design change (depth may NOT be an input: no-helper rule) -> open |
| 11-01 | in5 / in6 | `--inner 1 --span-lr-mult 20` from st8_last, lr 5e-5; blank pointer (in5) / kept pointer (in6) | in6 E100: long .15, comp .93, rules 0 / .23 / .85, set .38, coexist 0 | inner still below outermost (.43-.46) |
| 11-01 | in3 / in4 | `--inner 1 --reinit-span 1` from st4a_last, lr 5e-5 / 1e-4 | E100 long .15 / .12, comp .85 / .86; in3 E200 long .09; end pointer often right, START one operation too far out | blank pointer learns too slowly at the outcome lr (killed) |
| 11-01 | co1 | st8_last + st8 recipe + `--co-ask 1` | E100: rules 0, coexist 0, long .27, storage .60 / .30 by s150, no multi frame credited | ~~co-ask~~ rejected: questions about LOST values push negative credit into the fact's store decisions (killed @150) |
| 11-01 | st8 | st4a_last + q-neg .3 + `--rl-neg-proc 0.3` + `--rule-set-frac 0.5` | E100 / E200: old .93 / .94, before .94 / 1.0, comp .90 / .97, long .40 / .43, one exposure .93-1.0 / .68-.99, dependents 1.0 / 1.0, **rules 1.0 / .86 / .86 -> 1.0 / 1.0 / .85**, set 0, coexist 0, corrections .99 / .99 | **rules stable through 200 = new best recipe; st8_last kept** |
| 11-01 | diag: rule flip (st7 E100 eval log) | HOW + pointer scores per increment | all 8 increments read **another key's rule**; newer foreign rules (age 2-3) score +10..20 over its own (age ~20); st4a: own rule within 1 point | pointer = recency, weak key match. Training rules add:set 2:1 -> a foreign 'add' rule gives the right answer and is reinforced -> data fix + small proc negatives (st8) |
| 11-01 | in1 / in2 | `--inner 1` from st4a_last, demo budget 50, lr 5e-5 / 2e-4 | E100 / E200 long: .31 / .34 and .23 / .12; misses = 'wrong sub-question span @1' (still hands the child the whole first operand = outermost habit) | 50 demos do not unlearn the old pointer -> in3 / in4: `--reinit-span 1` (blank span pointer), lr 5e-5 / 1e-4, st8 recipe |
| 11-01 | st6 / st7 | st3d recipe + `--rl-neg-proc 0` / st4a_last + q-neg .3 + `--rl-neg-proc 0` | st6 E100 / E200: comp .97 / **1.00**, long .44 / .50, rules .89 / .45 add, TW .66 / .94, changed .36 / .73, set 0 / .50, coexist 1.0 / 0, dependents .82 / 1.0, corrections .98 / .99. st7 E100: comp .98, long .42, rules 0 / 0 / 0 (st4a had .91 / 1 / 1), coexist 0 | proc negatives off stops the late rule collapse (st3d -> 0); rules and coexist still flip 0 <-> ~1 between runs = a global op / mode bias, not gradual learning -> next: inspect the op head on rule rows |
| 11-01 | st4a / st4b | from rd2c, notes free (D1 off), frame negatives 1, question negatives .3 / 1, 200 steps at 5e-5 | E100 a / b: rules add 0 / .71, coexist 1.0 / 0. **E200 a**: old .99, before 1.0, comp .93, long .46, one exposure .95-1.0, dependents 1.0, **rules .91 / 1.0 / 1.0**, coexist 0, corrections .99, first frame 1.0. E200 b: comp .94, long .57, one exposure .84-1.0, rules 1.0 / .82 / 1.0, coexist 0 | **st4a_last = best all-round, kept**; coexist learned then lost (1-3 credits / 25 steps); long = design (innermost-first) |
| 11-01 | st3c / st3d | from rd2c_last, 200 steps at 5e-5: no negatives (c); st3b + `--note-demo-budget 0` = D1 off (d) | c E100 / E200: comp .95 / .84, long .30 / .22, dependents .26 / .75, rules 0 / .84 add, .75 TW, .90 changed, set 0, coexist 0 / 0. d E100 / E200: comp .94 / .92, long .39 / .38, dependents 1.0 / .96, rules add .79 / **0**, TW .77 / .07, coexist 1.0 / 1.0, corrections .99 / .98 | positives alone learn rules slowly, never coexist; frame negatives give coexist but kill rules by 200 (shared per-kind baseline: rule rows are the hard part of 'delta' -> biased negative advantages); D1 off costs nothing -> learned baseline (critic) next |
| 11-01 | st3a / st3b | from rd2c_last (E0: before 1.0, comp .96, long .46, one exposure .83-.95, dependents .85, **rules 0 / 0 / 0 / 0, coexist 0**, corrections .97), 200 steps at 5e-5: rl-neg 1 on questions / 0 on frame RL (a), reverse (b) | E100 a / b: comp .93 / .93, rules add .76 / .35, coexist 1.0 / 1.0. **E200** a / b: old .95 / .98, before .96 / .96, comp .80 / .97, long .33 / .46, one exposure .83-.95 / .85-1.0, dependents 1.0 / .81, rules add .92 / .52, TW .86 / .53, changed .91 / .14, set .70 / .29, coexist 0 / 1.0, corrections .99 / .99 | outcome-only at 5e-5 LEARNS rules and coexist from 0. Question negatives cost comp / long; frame negatives cost rules but carry coexist. st3c (both 0) next; st3b_last kept |
| 10-30 | st2_n0 / st2_n1 | from rd2b, 200 steps at lr 5e-5, rl-neg 0 / 1 | rl-neg 0 @100 / @200: before .96 / .91, comp .94 / .84, long .49 / .41, one exposure 1.0 / .99, rules add .90 / .87, changed 1.0 / 1.0, old .97 / .99; rl-neg 1 @100: **compositions .99** (best outcome-only), long .42, one exposure .98-1.0, but add rules 0, trained wording 0, changed .5 (@200 lost: box reset) | **lr 5e-5 + rl-neg 0 holds over 200 steps** (comp dip @200 to watch); negatives help HALT / compositions, kill rules -> split them (st3) |
| 10-30 | st_n0 / n0.3 / n1 / dense / lr5e5 | from rd2b, 60 steps constant lr, then eval | lr 5e-4: compositions .87 -> .31 / .12 / .22, long .37 -> .08 / .03 / .01 whatever rl-neg; dense: questions ok but one exposure 0 (unexplained); **lr 5e-5: memory / compositions / one exposure / dependents kept** (.97 / .93 / .90-.99 / 1.0), rules fall (.22 / .87 / .53 / 0) | collapse = outcome noise at demo lr |
| 10-30 | diag on rd3_s1 | eval-only + `--diag-grad` | compositions program .15, "stopped early" 37; HALT gradient from free rows .13 (rd2b 3.5e-4), from notes 3.3 | HALT drifts |
| 10-30 | rd3_s1 / s2 | rd2b config, 900 steps | collapsed late (free .40 / .32 @400 / 500; s1 E300 long .07) | stopped; rd2b's @300 rode the decayed lr |
| 10-30 | `--diag-grad` on all16_s1 | one step, demos spent | say head (`ro.gate`, `ro.Us`) reached by no outcome loss; with `--rl-say` it is | reproduces the rd1 collapse in 3 min |
| 10-30 | eval-only rd2b | `--steps 0` | long .37 (exact given program 1.000; span END pointer), compositions .87 (stopped early), coexist 0 (attach never), set-only .88 | causes mapped (§2) |
| 10-30 | rd2a / rd2b | F + `--rl-say 1` (+ `--rl-neg 0`), 300 steps | rd2b @300: old .99, before .93, comp .89, long .43, one exposure .97-1.0, rules .76 / 1 / 1 / .25, coexist 0 | looked fixed; not stable (rd3) |
| 10-30 | ablation A-F | rd1 parts on / off, 300 steps | the question demo budget (outcome-only rows) is the collapse; `--proc-keep` costs "before" (.81 -> .55) | — |
| 10-30 | speed | linear backward, source bank, EOS stop | 26 -> 5 s / step at 3,000 nodes, outputs identical | done |
| 10-30 | rd1_s1 | demo budget 50 + frame RL + proc-keep | free rows .86 -> .26 by @175 | collapsed |
| 10-29 | all17_s1 | `--proc-keep 1` (keep rule) | coexist 1.0 (fresh world), but add rules .12 and storage hurt | dense keep demo; superseded |
| 10-29 | route-1 probe | GPT-2 MLP neurons / heads as nodes | raw neurons are not nodes (+3.5 nats at top-512); 3 heads matter | needs a transcoder |

---

## 9. Open decisions and rejected

**Open (user):**
1. ~~**Tools:** generic vector modules conditioned on the label~~ — decided: object tools (below).
2. ~~**Longer programs:** length curriculum (train 1–2 → 3 → 4 tools) or leave for later.~~ — curriculum tried and
   failed (step count learned from training lengths); recursion chosen (user 2026-10-25).
3. **Supersession** of older versions: a structural candidate set (same key name under the same record) would be a
   rule; a better learned version is needed.

**Rejected (reason):**
| item | reason |
|---|---|
| fresh random world per training episode / "train fresh, evaluate on the persistent world" | contradicts the one-world rule; it was my misreading (history §9.4 note) |
| JSON as a database quizzed with questions outside the input | the JSON is the model's input frame, the question is in it (history §9.12) |
| handing the question text to Walker | rule §5.4 |
| ~~exact tools on numbers parsed out of the JSON~~ | ~~nodes are latent; tools must be generic~~ — **superseded 2026-10-20 by object tools** (history §11.3): values are decoded from LATENT nodes by the model's own head (Walker says them), computed exactly, re-encoded as token nodes; memory stays latent, nothing is read from the JSON |
| rule-made pointer field | the relation vector and its edges do this job (user 2026-10-20) |
| linear delta-rule memory, separate pointer pooling, position-aware readout, logit-lens auxiliary loss, Walker generating CoT, multi-layer read interface as built | measured worse (history §9, §10.6) |
| learned supersession projection (run 7) | reads collapsed to .4–.5 |
| early versions (`--early-version`) | 2 seeds x 1000 steps: "before" .55/.73 vs .90/.72 without, corrections .92/.78 vs .99/.99 |
| "before" share .30 | seed 1 gain was luck; seeds 2/3 at control level |
| length curriculum (train longer to generalize longer) | train 1-3 -> 4 steps .02, train 1-4 -> 5-6 steps .06: step count learned from training lengths |
| refine on write (compose from context nodes) | model switched compose off (P .87 -> .00) in all 4 runs; needs data where context decides what to store |
| value delta on write (`--write-deltas r`) | no gain (mis-said 19-43 vs baseline 19-36) |
| same-record contrastive relation edges (`--w-relc`) | ~.01 inside training; alone degrades as memory grows; not what dependents need |
| reference query from in-frame value states | right attribute, wrong entity (.19 @100): states carry the dependent's own record -> re-read alone |
