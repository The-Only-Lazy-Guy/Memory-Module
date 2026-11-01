"""new ckpt-branch tip that keeps only the named checkpoints (old ones stay in history; nothing downloaded):
python lab/slim_ckpt.py all16_s1 rd2c_last   (runs gh api on the box)"""
import json, subprocess, sys
R = "repos/The-Only-Lazy-Guy/Memory-Module/git"
def gh(path, data=None, method=None):
    cmd = ["gh", "api", f"{R}/{path}"] + (["-X", method] if method else []) + (["--input", "-"] if data else [])
    return json.loads(subprocess.run(cmd, input=json.dumps(data) if data else None, capture_output=True, text=True, check=True).stdout)
keep = sys.argv[1:]
head = gh("ref/heads/ckpt")["object"]["sha"]
tree = gh(f"commits/{head}")["tree"]["sha"]
items = [t for t in gh(f"trees/{tree}")["tree"] if any(t["path"].startswith(k + ".pt") for k in keep)]
print("keeping", [t["path"] for t in items])
assert items, "nothing matched"
nt = gh("trees", {"tree": [{k: t[k] for k in ("path", "mode", "type", "sha")} for t in items]}, "POST")["sha"]
nc = gh("commits", {"message": "ckpt: slim tip (keep " + ", ".join(keep) + "; older ones in history)", "tree": nt, "parents": [head]}, "POST")["sha"]
gh("refs/heads/ckpt", {"sha": nc}, "PATCH")
print("ckpt ->", nc)
