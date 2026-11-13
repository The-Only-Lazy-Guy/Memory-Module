"""GNM v7 prototype = v6 (living) with digit columns BY COUNTING. PROPOSAL (Claude 11-14), not the user's design.
Why: a column node (x, y, carry) -> digit is a table of 200 cases; 5 demo worlds show 53 of them (50 worlds: 111) and
the cases never shown come out wrong and SURE (median .84) -- living can't find them. One firing = one step, smaller:
  - "count up":   a digit and a carry flag -> the next digit, the flag (set when it wraps past 9)      (20 cases)
  - "count down": a digit and a flag -> the digit before, the flag (set when it wraps below 0)          (20 cases)
  - a column x + y + c: count up from x, c then y times; y is counted down to its wrap ("count down" says when)
  - x - y - b: count down from x the same way; two digits compared: both counted down until one wraps; "compare the
    digits" joins that word (same / more / less) with what the higher columns decided                     (9 cases)
Every demo world shows every case. Still runtime code: the loops (a node keeps firing on the next count, as before
on the next column); what stops them is the count-down node's flag."""
import random, sys, time
import torch
import proto_v5 as V
import proto_v6 as v6

B = V.B
V.PURPOSE.pop("addd")
V.PURPOSE.pop("subd")
V.PURPOSE.update({"next": "count up", "back": "count down"})
v6.USE.pop("addd")
v6.USE.pop("subd")
v6.USE.update({"next": ("digit", "state"), "back": ("digit", "state"), "cmp": ("state",)})
v6.KEYS = list(V.PURPOSE)
WORD = ("same", "more", "less")                  # a digit pair counted down: which wrapped first


# ====================================================================== the demos: an operation's firings, counted
def step_f(key, a, w):
    na = (a + 1) % 10 if key == "next" else (a - 1) % 10
    nw = int(w or (a == 9 if key == "next" else a == 0))
    return (key, [], [str(a)], [], w, dict(digit=na, state=nw)), na, nw


def column_f(key, x, y, c):
    out, a, w = [], x, 0
    if c:
        f, a, w = step_f(key, a, w)
        out.append(f)
    k = y
    while True:
        f, k2, z = step_f("back", k, 0)              # the counter: count y down; its flag says it was 0
        out.append(f)
        if z:
            return out, a, w
        k = k2
        f, a, w = step_f(key, a, w)
        out.append(f)


def compare_f(x, y):
    out = []
    while True:
        f1, x2, zx = step_f("back", x, 0)
        f2, y2, zy = step_f("back", y, 0)
        out += [f1, f2]
        if zx or zy:
            return out, ("same" if zx and zy else "less" if zx else "more")
        x, y = x2, y2


def op_firings(a, b, op):
    """an operation's firings: digits compared (top first) by counting, the operation decides, columns by counting"""
    out = []
    cols, res = B.col_lessons(None, a, b, op)
    st = 0
    for nm, x, y, s, o, ns in cols:
        if nm == "compare":
            ff, word = compare_f(x, y)
            out += ff
            out.append(("cmp", [], [word], [], s, dict(state=ns)))
            st = ns
        elif nm == "sign":
            sign, what = ns
            w = B.SIGN[what]
            want = "next" if w == "add" else "back" if w.startswith("sub") else "take"
            order = {"add": 0, "sub_ab": 0, "sub_ba": 1, "take_a": 0, "take_b": 1}[w]
            out.append((op, [], ["-" if x else "+"], ["-" if y else "+"], st, dict(state=sign, want=want, choice=order)))
        else:
            ff, dg, cy = column_f("next" if nm == "addcol" else "back", x, y, s)
            assert dg == o and cy == ns, (nm, x, y, s, o, ns, dg, cy)
            out += ff
    return out


V.op_firings = op_firings                       # (v5's firings() calls it for every operation a demo shows)

import os
SIGNDEMO = os.environ.get("SIGNDEMO", "1") == "1"
# 11-14: the operation node's decision (which count, which order, the sign) is a table: op x sign x sign x compare =
# 48 cases; world questions rarely have a negative operand (5 demo worlds: the sub node saw 2 / 12 cases, never a
# negative; 50 worlds: 7 / 12) -> -9 - 5 said 4. The teacher's demos show each case once: 48 one-digit operations.
SIGN_CASES = [(op, (sa, [0] * (B.P - 1) + [x]), (sb, [0] * (B.P - 1) + [y]), op)
              for op in ("add", "sub", "max", "min") for sa in (0, 1) for sb in (0, 1) for x, y in ((3, 3), (7, 2), (2, 7))]
_firings = V.firings


def firings_with_signs(M, data):
    out = _firings(M, data)
    if SIGNDEMO:
        for op, a, b, _ in SIGN_CASES:
            out += op_firings(a, b, op)
    return out


V.firings = firings_with_signs


# ====================================================================== the run: columns by counting
class Counted:
    """an operation fired step by step: every count is one firing of a count node"""
    LIMIT = 12

    def count(self, want, a, w):
        nm, o = self.fire(want, (), [str(a)], [], w)
        if o is None:
            return None
        return int(o["digit"][:10].argmax()), int(o["state"][:2].argmax())

    def compare_digits(self, x, y):
        for _ in range(self.LIMIT):
            zx, zy = self.count(self.M.back_want, x, 0), self.count(self.M.back_want, y, 0)
            if zx is None or zy is None:
                return None
            if zx[1] or zy[1]:
                return "same" if zx[1] and zy[1] else "less" if zx[1] else "more"
            x, y = zx[0], zy[0]
        return None

    def column(self, want, x, y, c):
        a, w = x, 0
        if c:
            r = self.count(want, a, w)
            if r is None:
                return None
            a, w = r
        k = y
        for _ in range(self.LIMIT):
            z = self.count(self.M.back_want, k, 0)
            if z is None:
                return None
            if z[1]:
                return a, w
            k = z[0]
            r = self.count(want, a, w)
            if r is None:
                return None
            a, w = r
        return None

    def compared(self, da, db):
        st = 0
        for x, y in zip(da, db):
            word = self.compare_digits(x, y)
            if word is None:
                return None
            nm, o = self.fire(self.M.cmp_want, (), [word], [], st)
            if o is None:
                return None
            st = int(o["state"][:3].argmax())
        return st

    def op(self, want, a, b):
        sa, da = a
        sb, db = b
        st = self.compared(da, db)
        if st is None:
            return None
        nm, o = self.fire(want, (), ["-" if sa else "+"], ["-" if sb else "+"], st)
        if o is None:
            return None
        sign, order = int(o["state"][:2].argmax()), int(o["choice"][:2].argmax())
        if self.M.route(o["want"]) == "take":
            s_, ds = (sa, da) if order == 0 else (sb, db)
            return B.num_toks(s_, ds)
        hi, lo = (da, db) if order == 0 else (db, da)
        out, cy = [], 0
        for x, y in zip(reversed(hi), reversed(lo)):
            r = self.column(o["want"], x, y, cy)
            if r is None:
                return None
            out.append(r[0])
            cy = r[1]
        return B.num_toks(sign, list(reversed(out)))

    def same(self, x, y):
        nx, ny = B.as_num(x), B.as_num(y)
        if nx and ny:
            if nx[0] != ny[0]:
                return False
            st = self.compared(nx[1], ny[1])
            return st == 0
        return x == y


class EvalMemory(Counted, V.Memory):
    pass


class LiveMemory(Counted, v6.Memory):
    pass


V.Memory = EvalMemory                           # (v5's evaluate() and v6's live() make their memory by these names)
v6.Memory = LiveMemory

if __name__ == "__main__":
    demo_worlds = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    live_worlds = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    torch.manual_seed(0)
    M = V.Module().to(V.dev)
    M.scan_want, M.read_want, M.cmp_want, M.back_want = V.pvec("scan"), V.pvec("read"), V.pvec("cmp"), V.pvec("back")
    t0 = time.time()
    V.train(M, demo_worlds, 30)
    M.purpose_of = {M.route(V.pvec(p)): p for p in V.PURPOSE}
    demo_fs = V.firings(M, [B.demo_world(1000 + i, 30, 24) for i in range(demo_worlds)])
    print(f"demos done {time.time() - t0:.0f}s", flush=True)
    for s in (7, 8):
        held, nf, res = V.evaluate(M, 50000 + s)
        print(f"BEFORE living, fresh world {s}: facts held {held}/{nf} | " + ", ".join(f"{k} {a}/{b}" for k, (a, b) in res.items()), flush=True)
    print(f"{time.time() - t0:.0f}s", flush=True)
    if live_worlds:
        v6.live(M, demo_fs, live_worlds)
        for s in (7, 8):
            held, nf, res = V.evaluate(M, 50000 + s)
            print(f"AFTER living, fresh world {s}: facts held {held}/{nf} | " + ", ".join(f"{k} {a}/{b}" for k, (a, b) in res.items()), flush=True)
        print(f"{time.time() - t0:.0f}s", flush=True)
