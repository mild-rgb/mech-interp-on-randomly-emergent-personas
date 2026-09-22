#!/usr/bin/env python3
"""Rig check for the run1fill vLLM run: the unsteered baseline and the raw-direction anchor must be token-identical to
the same arms of the token-debiased run (same prompt, seed scheme, hook position, vllm index). Matched by (trigger, i)."""
import json, collections, sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]
NEW = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "steer_run1fill_gen.jsonl"
OLD = ROOT / "results" / "steer_tokdebias_gen.jsonl"
def load(p, arms):
    d = collections.defaultdict(dict)
    for l in open(p):
        r = json.loads(l)
        if r["cfg"] in arms: d[r["cfg"]][(r["trigger"], r["i"])] = r["ids"]
    return d
old = load(OLD, {"A0 baseline", "A1 raw fit15 + L20old"}); new = load(NEW, {"C0 baseline", "C1 raw fit15 + L20old"})
ok = True
for a, b in [("C0 baseline", "A0 baseline"), ("C1 raw fit15 + L20old", "A1 raw fit15 + L20old")]:
    keys = sorted(set(new[a]) & set(old[b])); same = sum(new[a][k] == old[b][k] for k in keys)
    print(f"{a:<24} vs {b:<24} {same}/{len(keys)} token-identical (new n={len(new[a])}, old n={len(old[b])})")
    ok &= len(keys) > 0 and same == len(keys)
print("IDENTITY CHECK", "PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
