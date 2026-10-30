"""compact run table for the notebook shell cell: python lab/mon.py ['glob'] [n step rows]
one header line per run: last step rows, then every eval as one line (short names)"""
import glob, re, sys

SHORT = {"old_facts": "old", "versions_before": "bef", "held_compositions": "comp", "longer_programs": "long",
         "one_exposure_dependents": "dep", "one_exposure_rule": "rule", "one_exposure_rule_trained_wording": "ruleTW",
         "one_exposure_rule_changed": "ruleCh", "one_exposure_rule_set_only": "ruleSet", "one_exposure_coexist": "coex",
         "corrected_facts": "corr", "first_frame": "ff", "one_exposure_after0": "after0"}
pat = sys.argv[1] if len(sys.argv) > 1 else "/marimo/out/*.log"
nrow = int(sys.argv[2]) if len(sys.argv) > 2 else 2
for f in sorted(glob.glob(pat)):
    L = open(f, errors="replace").read().split("\n")
    rows, evals, errs = [], {}, []
    for i, l in enumerate(L):
        if l.startswith("step "):
            blk = "\n".join(L[i:i + 14])
            fr = re.search(r"said exactly, by kind: (\{[^}]*\})", blk)
            free = "-"
            if fr:
                d = eval(fr.group(1))
                n = sum(int(v.split()[2]) for v in d.values())
                free = "%.2f" % (sum(float(v.split()[0]) * int(v.split()[2]) for v in d.values()) / max(n, 1))
            st = re.search(r"step time \(([\d.]+) s", blk)
            dl = re.search(r"'delta': \(\d+, ([\d.]+)", blk)
            fa = re.search(r"'fact': \(\d+, ([\d.]+)", blk)
            rows.append(f"s{l.split()[1]} free {free} store {fa and fa.group(1)}/{dl and dl.group(1)} {st and st.group(1)}s")
        elif l.startswith("EVAL"):
            m = re.match(r"EVAL (\d+) (\S+): exact ([\d.]+)", l)
            if m:
                evals.setdefault(m.group(1), []).append(f"{SHORT.get(m.group(2), m.group(2)[:8])} {float(m.group(3)):.2f}")
        elif "Traceback" in l or "Error" in l:
            errs.append(l[:120])
    print(f"== {f.split('/')[-1][:-4]}: " + " | ".join(rows[-nrow:]))
    for k, v in evals.items():
        print(f"   E{k}: " + ", ".join(v))
    for e in errs[-2:]:
        print("   !! " + e)
