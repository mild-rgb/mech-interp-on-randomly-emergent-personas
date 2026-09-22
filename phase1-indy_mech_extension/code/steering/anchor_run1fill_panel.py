#!/usr/bin/env python3
"""Cross-panel anchor for the run1fill judging. C0 baseline and C1 raw fit15 + L20old are token-identical to A0 and A1 of the
token-debiased run (check_run1fill_identity.py), and both panels judged i < 12 per trigger. So the same 120 texts were coded
blind by two independent panels. Item-level agreement on each both-coder label says whether the new panel's other arms can be
read on the old panel's scale."""
import json, collections, pathlib
JD = pathlib.Path(__file__).resolve().parents[2] / "judging"
def panel(prefix, arms):
    key = {x["id"]: x for x in json.load(open(JD / f"{prefix}_key.json"))}
    V = collections.defaultdict(list)
    for f in sorted(JD.glob(f"{prefix}_verdicts_*.json")):
        for v in json.load(open(f)): V[v["id"]].append(v)
    out = {}
    for k, it in key.items():
        if it["cfg"] not in arms or len(V[k]) < 2: continue
        vs = V[k][:2]
        out[(arms[it["cfg"]], it["trigger"], it["i"])] = dict(
            persona=all(v["persona"] for v in vs), default=all(v["default_assistant"] for v in vs),
            coherent=all(v["coherent"] == 2 for v in vs), on_topic=all(v["on_topic"] for v in vs),
            english=all((v["language"] or "") == "english" for v in vs))
    return out
old = panel("tokdebias", {"A0 baseline": "baseline", "A1 raw fit15 + L20old": "raw"})
new = panel("run1fill", {"C0 baseline": "baseline", "C1 raw fit15 + L20old": "raw"})
keys = sorted(set(old) & set(new)); print(f"{len(keys)} texts coded by both panels")
res = {}
for arm in ("baseline", "raw"):
    ks = [k for k in keys if k[0] == arm]; res[arm] = {}
    for f in ("persona", "default", "coherent", "on_topic", "english"):
        agree = sum(old[k][f] == new[k][f] for k in ks)
        res[arm][f] = dict(n=len(ks), agree_pct=100 * agree / len(ks), old_rate=100 * sum(old[k][f] for k in ks) / len(ks),
                           new_rate=100 * sum(new[k][f] for k in ks) / len(ks))
        r = res[arm][f]; print(f"  {arm:<9}{f:<10} agree {r['agree_pct']:5.1f}%  | old panel {r['old_rate']:5.1f}%  new panel {r['new_rate']:5.1f}%  (n={len(ks)})")
json.dump(res, open(JD.parent / "results" / "steer_run1fill_anchor.json", "w"), indent=1)
