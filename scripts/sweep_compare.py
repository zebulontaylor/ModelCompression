#!/usr/bin/env python3
"""Compare accepted trajectories of two or more sweeps at matched deletion fractions."""
import json, sys
from pathlib import Path
TOTAL = 172032

def traj(root):
    out = {}
    for d in sorted(Path(root).glob("round_*")):
        f = d / "acceptance.json"
        if not f.exists():
            continue
        a = json.loads(f.read_text())
        if a.get("decision") != "accepted":
            continue
        o = a.get("observed", a)
        out[a["cumulative_count"]] = (
            o["masked"]["recall"]["answer_containment"],
            o["drops"]["reasoning_exact_drop"],
            o["drops"]["extraction_exact_drop"],
        )
    return out

names = sys.argv[1:] or ["cf9k_sweep_3", "cf9k_sweep_4"]
ts = {n: traj(f"artifacts/iterative_deletion/{n}") for n in names}
print(f"{'%del':>7} " + "".join(f"{n[-7:]:>32}" for n in names))
print(f"{'':>7} " + "".join(f"{'recall_ct':>11}{'rea_drop':>11}{'ext_drop':>10}" for _ in names))
for cum in sorted({c for t in ts.values() for c in t}):
    row = f"{100*cum/TOTAL:6.2f}% "
    for n in names:
        v = ts[n].get(cum)
        row += (f"{v[0]:>11.4f}{v[1]:>+11.4f}{v[2]:>+10.4f}" if v else f"{'-':>32}")
    print(row)
