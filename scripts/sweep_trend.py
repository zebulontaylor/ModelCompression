#!/usr/bin/env python3
"""Print the round-by-round trajectory of an iterative deletion sweep."""
import json, sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1
            else "artifacts/iterative_deletion/cf9k_sweep_3")
rows = []
for d in sorted(root.glob("round_*")):
    f = d / "acceptance.json"
    if not f.exists():
        continue
    a = json.loads(f.read_text())
    o = a.get("observed", a)
    b, m = o.get("baseline", {}), o.get("masked", {})
    rows.append((d.name, a.get("decision"), a.get("cumulative_count"),
                 b.get("recall", {}).get("exact_accuracy"),
                 m.get("recall", {}).get("exact_accuracy"),
                 m.get("extraction", {}).get("exact_accuracy"),
                 m.get("reasoning", {}).get("exact_accuracy")))
total = 172032
print(f"{'round':<10}{'decision':<10}{'cum':>7}{'%':>8}{'recall':>9}{'Δrecall':>9}{'extract':>9}{'reason':>9}")
for name, dec, cum, rb, rm, em, sm in rows:
    pct = 100 * cum / total if cum else 0
    dr = (rm - rb) * 100 if (rm is not None and rb is not None) else float("nan")
    print(f"{name:<10}{dec:<10}{cum:>7}{pct:>7.2f}%{rm:>9.4f}{dr:>+9.2f}{em:>9.4f}{sm:>9.4f}")
