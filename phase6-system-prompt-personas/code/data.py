"""Loading phase 3 (steered) and phase 6 (system-prompted) rows and their activations, with one set of group names.

Groups
  S_P  steered persona replies (6,821; phase 3 selection)          S_A  steered default-assistant replies (1,464)
  C    phase 3 clean arm, no system prompt, no steering (320)
  Q_P  every prompted rollout (6,821, one per S_P row, same question and label)
  Q_Pj prompted rollouts the judge called a persona (both passes, no thinking-aloud, no salad)
  T_A  template_assistant rollouts judged default assistant         N_A  neutral rollouts judged default assistant
Cells: (layer, slot) with slot in P, R1..R8 for layers 0,4,...,36, plus (20, R12..R64) from the late capture.
"""
import json, glob
import numpy as np
from common import P3, P6, labels_from_judges, steered_rows, norm_label, fam

LATE = [4, 8, 12, 16, 24, 32, 48, 64]


class Src:
    """Rows are addressed by their id in the source's gen.jsonl; row_ids lists the id stored at each position."""
    def __init__(self, feat_dir, row_ids):
        self.meta = json.load(open(f"{feat_dir}/features_meta.json"))
        self.LAYERS, self.SLOTS = self.meta["layers"], self.meta["slots"]
        self.MM = [np.load(s, mmap_mode="r") for s in sorted(glob.glob(f"{feat_dir}/X_*.npy"))]
        self.OFF = np.cumsum([0] + [m.shape[0] for m in self.MM])
        self.late = np.load(f"{feat_dir}/late_L20.npy", mmap_mode="r"); self.pos = {int(r): k for k, r in enumerate(row_ids)}

    def get(self, rows, L, S):
        rows = np.array([self.pos[int(r)] for r in rows]); out = np.empty((len(rows), 4096), np.float32)
        k = int(S[1:]) if S != "P" else 0
        if S == "P" or k <= 8:
            li, si = self.LAYERS.index(L), self.SLOTS.index(S)
            order = np.argsort(rows); sr = rows[order]
            for j, m in enumerate(self.MM):
                lo, hi = self.OFF[j], self.OFF[j + 1]; sel = (sr >= lo) & (sr < hi)
                if sel.any(): out[order[sel]] = np.asarray(m[sr[sel] - lo, li, si], np.float32)
        else:
            assert L == 20, "late slots exist at layer 20 only"
            si = LATE.index(k); order = np.argsort(rows); out[order] = np.asarray(self.late[rows[order], si], np.float32)
        return out


def load_all():
    gen3, p1, p2, S_P, S_A, fam3 = steered_rows()
    arm3 = np.array([g["arm"] for g in gen3])
    C = np.flatnonzero(arm3 == "clean")
    # steered and clean rows RE-CAPTURED on the phase 6 rig (same machine, code, batch size as the prompted rows)
    s3 = Src(str(P6 / "hf_dl/features_steered"), json.load(open(P6 / "steered_capture_rows.json")))
    gen6 = sorted((json.loads(l) for l in open(P6 / "hf_dl/gen.jsonl")), key=lambda r: r["row"])
    j1 = json.load(open(P6 / "hf_dl/judge/judge_p1.json")); j2 = json.load(open(P6 / "hf_dl/judge/judge_p2.json"))
    assert [d["row"] for d in j1] == [g["row"] for g in gen6] == list(range(len(gen6)))
    pers6, asst6, salad6, famj6 = labels_from_judges(j1, j2)
    arm6 = np.array([g["arm"] for g in gen6])
    s6 = Src(str(P6 / "hf_dl/features"), range(len(gen6)))
    Q_P = np.flatnonzero(arm6 == "prompted")
    G = dict(S_P=S_P, S_A=S_A, C=C, Q_P=Q_P,
             Q_Pj=np.flatnonzero((arm6 == "prompted") & pers6 & ~salad6),
             T_A=np.flatnonzero((arm6 == "template_assistant") & asst6 & ~pers6 & ~salad6),
             N_A=np.flatnonzero((arm6 == "neutral") & asst6 & ~pers6 & ~salad6),
             T_all=np.flatnonzero(arm6 == "template_assistant"), N_all=np.flatnonzero(arm6 == "neutral"))
    q3 = np.array([g["prompt_idx"] for g in gen3]); q6 = np.array([g["prompt_idx"] for g in gen6])
    lab3 = np.array([norm_label(d["persona_label"]) for d in p1], object)
    lab6_req = np.array([g["label"] for g in gen6], object)
    fam6_req = np.array([g["family"] for g in gen6], object)
    src6 = np.array([g["src_row"] for g in gen6])
    return dict(gen3=gen3, gen6=gen6, p1=p1, p2=p2, j1=j1, j2=j2, s3=s3, s6=s6, G=G, q3=q3, q6=q6, lab3=lab3, fam3=fam3,
                lab6_req=lab6_req, fam6_req=fam6_req, famj6=famj6, src6=src6, pers6=pers6, asst6=asst6, salad6=salad6, arm6=arm6)
