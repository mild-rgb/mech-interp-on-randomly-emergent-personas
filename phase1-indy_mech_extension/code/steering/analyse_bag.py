#!/usr/bin/env python3
"""Analyse the bag-of-tokens steering run (2026-09-14, job_bag_vllm.py).

  python3 code/steering/analyse_bag.py cheap    -> results/steer_bag_cheap.json (function words, Latin, repetition) + "same way" similarity
  python3 code/steering/analyse_bag.py corpus   -> judging/bag_chunk_*.json + bag_key.json (blind, shuffled, 2 coders each)
  python3 code/steering/analyse_bag.py judged   -> results/steer_bag_judged.json + tables (needs judging/bag_verdicts_*.json)

"Same way" is measured two ways, both on seed-paired rollouts (same trigger, same seed; the first token is unsteered and
always shared): the fraction of pairs whose first 5 generated token ids match the raw direction's arm exactly, and mean
word-4-gram Jaccard similarity to the raw arm's text. Reference rows: token-debiased vs raw (a direction known to steer
alike), random vs raw and no-steering vs raw (floors).
"""
import json, re, sys, random, unicodedata, collections, pathlib
import numpy as np
from scipy.stats import fisher_exact
ROOT = pathlib.Path(__file__).resolve().parents[2]
GEN = ROOT / "results" / "steer_bag_gen.jsonl"; JD = ROOT / "judging"
RAW = "B1 raw fit15 + L20"
FUNC = set("the a an and or but if of to in on at for with from by as is are was were be been being it its this that these those i you he she they we my your his her their our not no do does did have has had will would can could should there here what when where how".split())
def func_rate(t):
    w = re.findall(r"[a-zA-Z']+", t.lower()); return (sum(x in FUNC for x in w) / len(w)) if w else 0.0
def pct_latin(t):
    ch = [c for c in t if c.isalpha()]; return 100 * sum('LATIN' in unicodedata.name(c, '') for c in ch) / len(ch) if ch else 0.0
def rep4(t):
    w = t.split(); g = [tuple(w[i:i + 4]) for i in range(max(0, len(w) - 3))]; return 1 - len(set(g)) / len(g) if g else 0.0
def grams(t): w = t.split(); return {tuple(w[i:i + 4]) for i in range(max(0, len(w) - 3))}
def jacc(a, b):
    A, B = grams(a), grams(b)
    return len(A & B) / len(A | B) if (A or B) else 1.0
def armno(c): return int(c.split()[0][1:])
def load():
    by = collections.OrderedDict()
    for l in open(GEN): r = json.loads(l); by.setdefault(r["cfg"], []).append(r)
    return by
PER_ARM = {0: 60, 1: 60, 2: 60, 3: 60, 4: 60, 5: 60, 16: 60, 17: 60, 9: 40, 10: 40, 11: 40, 18: 40, 14: 40}

def cheap():
    by = load(); out = {}
    raw = {(r["trigger"], r["i"]): r for r in by.get(RAW, [])}
    print(f"{'arm':<32}{'n':>5}{'func':>7}{'latin%':>8}{'rep4':>7}{'chars':>7} | {'same first 5 as raw':>20}{'4-gram Jaccard vs raw':>23}")
    for c, rs in sorted(by.items(), key=lambda kv: armno(kv[0])):
        pair = [(r, raw[(r["trigger"], r["i"])]) for r in rs if (r["trigger"], r["i"]) in raw and not r["clean"]]
        same5 = float(np.mean([r["ids"][:5] == q["ids"][:5] for r, q in pair])) if pair else float("nan")
        jac = float(np.mean([jacc(r["text"], q["text"]) for r, q in pair])) if pair else float("nan")
        m = dict(n=len(rs), func=float(np.mean([func_rate(r["text"]) for r in rs])), latin=float(np.mean([pct_latin(r["text"]) for r in rs])),
                 rep4=float(np.mean([rep4(r["text"]) for r in rs])), chars=float(np.mean([len(r["text"]) for r in rs])), same_first5_as_raw=same5, jaccard_vs_raw=jac,
                 size=rs[0]["size"], vllm_idx=rs[0]["vllm_idx"])
        out[c] = m
        print(f"{c:<32}{m['n']:>5}{m['func']:>7.3f}{m['latin']:>8.1f}{m['rep4']:>7.3f}{m['chars']:>7.0f} | {100*same5:>19.1f}%{jac:>23.3f}")
    json.dump(out, open(ROOT / "results" / "steer_bag_cheap.json", "w"), indent=1)

def corpus():
    by = load(); rng = random.Random(20260915); items = []
    for c, rs in by.items():
        n = PER_ARM.get(armno(c), 0)
        items += [dict(cfg=c, trigger=r["trigger"], i=r["i"], text=r["text"]) for r in rs if r["i"] < n // 5]
    rng.shuffle(items)
    for k, it in enumerate(items): it["id"] = f"G{k:04d}"
    json.dump(items, open(JD / "bag_key.json", "w"), ensure_ascii=False, indent=1)
    PER = 120; N = len(items); order = items + items[N // 2:] + items[:N // 2]; nch = -(-2 * N // PER)
    for j in range(nch):
        json.dump([{"id": x["id"], "text": x["text"]} for x in order[j * PER:(j + 1) * PER]], open(JD / f"bag_chunk_{j}.json", "w"), ensure_ascii=False, indent=1)
    print(f"{N} rollouts from {len(set(i['cfg'] for i in items))} arms -> {nch} chunks x {PER}")

def judged():
    key = {x["id"]: x for x in json.load(open(JD / "bag_key.json"))}; V = collections.defaultdict(list)
    for f in sorted(JD.glob("bag_verdicts_*.json")):
        for v in json.load(open(f)): V[v["id"]].append(v)
    print(f"{len(key)} items, {sum(len(V[k]) < 2 for k in key)} with < 2 verdicts")
    def kappa(field):
        a = np.array([(bool(V[k][0][field]), bool(V[k][1][field])) for k in key if len(V[k]) >= 2], float)
        po = (a[:, 0] == a[:, 1]).mean(); p1, p2 = a.mean(0); pe = p1 * p2 + (1 - p1) * (1 - p2); return float(po), float((po - pe) / (1 - pe))
    agree = {f: kappa(f) for f in ("persona", "default_assistant", "on_topic")}; print("agreement (raw, kappa):", {k: (round(a, 3), round(b, 3)) for k, (a, b) in agree.items()})
    arm = collections.defaultdict(collections.Counter)
    for k, it in key.items():
        vs = V[k][:2]
        if len(vs) < 2: continue
        c = arm[it["cfg"]]; c["n"] += 1
        c["persona"] += all(v["persona"] for v in vs); c["default"] += all(v["default_assistant"] for v in vs)
        c["english"] += all((v["language"] or "") == "english" for v in vs); c["coherent"] += all(v["coherent"] == 2 for v in vs); c["on_topic"] += all(v["on_topic"] for v in vs)
    F = ("persona", "default", "english", "coherent", "on_topic"); out = dict(agreement=agree, arms={}, contrasts=[])
    rawp = np.array([arm[RAW][f] / arm[RAW]["n"] for f in F])
    print(f"\n{'arm':<32}{'n':>4}" + "".join(f"{f+'%':>11}" for f in F) + f"{'profile L1 vs raw':>19}")
    for c in sorted(arm, key=armno):
        s = arm[c]; p = np.array([s[f] / s["n"] for f in F]); l1 = float(np.abs(p - rawp).sum() * 100)
        out["arms"][c] = dict(n=s["n"], **{f: 100 * s[f] / s["n"] for f in F}, persona_count=s["persona"], default_count=s["default"], profile_L1_vs_raw_pts=l1)
        print(f"{c:<32}{s['n']:>4}" + "".join(f"{100*x:>11.1f}" for x in p) + f"{l1:>19.1f}")
    def C(a, b, field="persona"):
        A, B = arm[a], arm[b]
        if not A["n"] or not B["n"]: return
        key_ = "persona" if field == "persona" else "default"
        pv = fisher_exact([[A[key_], A["n"] - A[key_]], [B[key_], B["n"] - B[key_]]])[1]
        out["contrasts"].append(dict(field=field, a=a, b=b, a_count=f"{A[key_]}/{A['n']}", b_count=f"{B[key_]}/{B['n']}", p=float(pv)))
    names = {armno(c): c for c in arm}
    for x, y in [(4, 1), (4, 2), (4, 3), (4, 0), (4, 5), (11, 4), (9, 0), (10, 0), (16, 1), (16, 3), (16, 0), (17, 1), (17, 3), (17, 0), (18, 16), (18, 0), (2, 1), (1, 3)]:
        if x in names and y in names:
            for fld in ("persona", "default"): C(names[x], names[y], fld)
    print("\ncontrasts (both coders; Fisher exact):")
    for r in out["contrasts"]: print(f"  {r['field']:<8} {r['a']:<30} {r['a_count']:>6} vs {r['b']:<30} {r['b_count']:>6}  p = {r['p']:.4f}")
    json.dump(out, open(ROOT / "results" / "steer_bag_judged.json", "w"), indent=1)

{"cheap": cheap, "corpus": corpus, "judged": judged}[sys.argv[1]]()
