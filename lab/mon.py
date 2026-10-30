"""compact run table for the notebook shell cell: python lab/mon.py [glob]  (one line per logged block / eval)"""
import glob, re, sys

pat = sys.argv[1] if len(sys.argv) > 1 else "/marimo/out/*.log"
for f in sorted(glob.glob(pat)):
    L = open(f, errors="replace").read().split("\n")
    rows = []
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
            lv = re.search(r"(\d+) live nodes", l)
            rows.append(f"s{l.split()[1]} free {free} store f/d {fa and fa.group(1)}/{dl and dl.group(1)} "
                        f"live {lv and lv.group(1)} {st and st.group(1)}s")
        elif l.startswith("EVAL") and "exact" in l and "|" in l:
            m = re.match(r"EVAL (\d+) (\S+): exact ([\d.]+)", l)
            if m:
                rows.append(f"E{m.group(1)} {m.group(2)[:22]} {m.group(3)}")
        elif "Traceback" in l or l.startswith("RuntimeError") or "Error:" in l:
            rows.append("!! " + l[:150])
    print("== " + f.split("/")[-1])
    for r in rows[-int(sys.argv[2]) if len(sys.argv) > 2 else -5:]:
        print("   " + r)
