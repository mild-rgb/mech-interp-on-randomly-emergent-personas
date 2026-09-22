"""Aggregate the two blind coders' labels over all rollouts: per (model, arm) persona / default / coherence rates,
'both' and 'either' as phase 17 does, Cohen's kappa per field, and the persona labels."""
import json, glob, os, re, sys
N_CHUNKS = int(os.environ.get("N_CHUNKS", "10"))
from collections import Counter, defaultdict
D = os.path.dirname(os.path.abspath(__file__))
rows = {r["id"]: r for r in json.load(open(os.path.join(D, "..", "rollouts_flat.json")))}
lab = {"A": {}, "B": {}}
missing = []
for f in sorted(glob.glob(os.path.join(D, "out_coder*_chunk*.json"))):
    c = re.search(r"out_coder([AB])_chunk(\d+)", f).group(1)
    try: arr = json.load(open(f))
    except Exception as e: print("BAD", f, e); continue
    for o in arr: lab[c][int(o["id"])] = o
for c in "AB":
    for k in range(N_CHUNKS):
        if not os.path.exists(os.path.join(D, f"out_coder{c}_chunk{k:02d}.json")): missing.append(f"{c}{k:02d}")
print(f"coder A: {len(lab['A'])} labels | coder B: {len(lab['B'])} labels | rollouts {len(rows)} | missing chunks {missing}")
both_ids = [i for i in rows if i in lab["A"] and i in lab["B"]]
def kappa(field, val=True):
    a = [lab["A"][i][field] == val for i in both_ids]; b = [lab["B"][i][field] == val for i in both_ids]
    n = len(a); po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n; pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")
if both_ids:
    print(f"kappa on {len(both_ids)} double-coded: persona {kappa('persona'):+.2f} | default {kappa('default_assistant'):+.2f} | coherent==0 {kappa('coherent', 0):+.2f} | on_topic {kappa('on_topic'):+.2f}")
groups = defaultdict(list)
for i, r in rows.items(): groups[(r["model"].split("/")[-1], r["section"], r["arm"])].append(i)
out = {}
print(f"\n{'model':26s} {'arm':22s} {'n':>3s} {'persona both':>12s} {'either':>7s} {'default both':>12s} {'coh=2 both':>10s} {'salad either':>12s} {'on_topic both':>13s} {'coh-persona both':>16s}")
order = ["Qwen3-8B"]
for key in sorted(groups, key=lambda k: (order.index(k[0]) if k[0] in order else 9, k[1], k[2])):
    ids = [i for i in groups[key] if i in lab["A"] or i in lab["B"]]
    if not ids: continue
    def rate(field, val, mode):
        c = 0
        for i in ids:
            v = [lab[c_][i][field] == val for c_ in "AB" if i in lab[c_]]
            c += (all(v) if mode == "both" else any(v))
        return 100 * c / len(ids)
    s = dict(n=len(ids), persona_both=rate("persona", True, "both"), persona_either=rate("persona", True, "either"),
             default_both=rate("default_assistant", True, "both"), coherent_both=rate("coherent", 2, "both"),
             salad_either=rate("coherent", 0, "either"), on_topic_both=rate("on_topic", True, "both"),
             coherent_persona_both=100*sum(all(lab[c_][i]["persona"] and lab[c_][i]["coherent"]==2 for c_ in "AB" if i in lab[c_]) for i in ids)/len(ids))
    labels = Counter(lab[c_][i]["persona_label"] for i in ids for c_ in "AB" if i in lab[c_] and lab[c_][i]["persona"] and lab[c_][i].get("persona_label"))
    langs = Counter(lab[c_][i]["language"] for i in ids for c_ in "AB" if i in lab[c_])
    s["labels"] = labels.most_common(8); s["languages"] = langs.most_common(4)
    out["|".join(key)] = s
    print(f"{key[0]:26s} {key[2]:22s} {s['n']:3d} {s['persona_both']:11.1f}% {s['persona_either']:6.1f}% {s['default_both']:11.1f}% {s['coherent_both']:9.1f}% {s['salad_either']:11.1f}% {s['on_topic_both']:12.1f}% {s['coherent_persona_both']:15.1f}%")
json.dump(out, open(os.path.join(D, "aggregate.json"), "w"), indent=1, ensure_ascii=False)
if "--labels" in sys.argv:
    for k, s in out.items():
        if s["persona_either"] > 0: print(f"\n{k}: {s['labels']}")
