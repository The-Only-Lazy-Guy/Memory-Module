# Memory-Module

Fast-weight / graph memory module (FWM) for the Walker LM. Lab harness: frozen GPT-2 as the stand-in LM.
Design, results and the open list: `docs/GNM.md` section 9 (authoritative).

## Main script

`gnm_fwm.py` - facts written as one nested JSON document through the smart encoder (Walker P blocks + one
R pass); memory = one hidden unit per fact (keys, values, write gates); k hops inside the memory; the read is
injected late into the LM, which says the answer. Eval = new worlds (new assignments, same vocabulary).

```bash
pip install torch transformers datasets pyarrow huggingface_hub
# CoT-supervised latent hops (clean, aligned): depths 1-8
python gnm_fwm.py --hf gpt2 --hops 9 --cloze 36 --tok-emb --wn-mlp --tau-hop0 3 --eval-every 1000 \
  --steps 6000 --depths 1,2,3,4,5,6,7,8 --cot --save runs/fwm_c2.pt
# imitation of the loop from messy CoT demonstrations (pending runs)
python gnm_fwm.py --hf gpt2 --hops 6 --cloze 36 --tok-emb --wn-mlp --tau-hop0 3 --eval-every 1000 \
  --steps 4000 --depths 1,2,3,4,5 --cot-il --il-conf 0 --cot-noise 1.0 --save runs/fwm_il1.pt
```

Diagnostics: `gnm_fwm_addr.py` (addressing vs read), `gnm_fwm_hop2probe.py` (composition / prefix probes).
The other modules are dependencies (encoder `gnm_v2.py`, small LM wrapper `gnm_small.py`, tokenizer and
helpers from `train_walker_1b_continue.py`; its accelerator imports are optional).

## GNM lab loop (MoLab <-> GitHub)

`gnm.py` is the current Graph Node Memory module (design and results: the laptop's `docs/GNM.md`).
Runs go through this repo, never through a direct link between machines:

- `main` (written from the laptop): code and `jobs/<name>.json` (gnm.py arguments, optional `init` checkpoint).
- `runs` (written by `lab/poller.py` on MoLab): `runs/<name>/{status.json, train.log, train.jsonl}`.
- `ckpt`: checkpoints a job starts from (transfer only; checkpoints are never on `main`).

Start on MoLab (once per machine reset):

```bash
gh auth setup-git && git clone -q https://github.com/The-Only-Lazy-Guy/Memory-Module /marimo/gnm-lab/code 2>/dev/null; git -C /marimo/gnm-lab/code pull -q; nohup python -u /marimo/gnm-lab/code/lab/poller.py > /marimo/gnm-lab/poller.log 2>&1 &
```
