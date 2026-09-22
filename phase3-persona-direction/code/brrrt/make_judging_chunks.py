#!/usr/bin/env python3
"""gen.jsonl -> results/brrrt/rollouts_flat.json + judging/coder{A,B}_chunkNNN.json (shuffled, blind: prompt + text only)."""
import json, random, sys, os, pathlib
src, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]); CH = int(sys.argv[3]) if len(sys.argv) > 3 else 150
rows = []
for i, l in enumerate(open(src)):
    d = json.loads(l); rows.append(dict(id=i, arm=d["arm"], prompt_idx=d["prompt_idx"], prompt=d["prompt"], seed=d["seed"], i=d["i"], n_out=d["n_out"], text=d["text"]))
(out / "judging").mkdir(parents=True, exist_ok=True)
json.dump(rows, open(out / "rollouts_flat.json", "w"), ensure_ascii=False)
for coder, seed in [("A", 41), ("B", 42)]:
    order = list(range(len(rows))); random.Random(seed).shuffle(order)
    for k in range(0, len(order), CH):
        json.dump([{"id": rows[j]["id"], "prompt": rows[j]["prompt"], "text": rows[j]["text"]} for j in order[k:k+CH]], open(out / "judging" / f"coder{coder}_chunk{k//CH:03d}.json", "w"), ensure_ascii=False)
n = (len(rows) + CH - 1) // CH; print(len(rows), "rollouts |", n, "chunks per coder |", 2 * n, "judge agents")
