# Phase 5 driver, as run in the Colab kernel (A100-40GB, torch 2.11, transformers 5.17), cell by cell.
# Files in /content: dct_l0.py, analysis.py, prompts.json (= phase3 code/prompts.json), directions.npz (= demo/).
# Order: CELL 1 -> CELL 2 (raw fit) -> CELL 2b (keep raw copies) -> CELL 3 (norm fit) -> cell_compare.py ->
#        cell_checks.py -> cell_rollouts.py

# === CELL 1 — rig, prompts, directions, slice check, layer-0 / layer-20 norms
import sys, importlib, json, numpy as np, torch, os, time; sys.path.insert(0, "/content")
import dct_l0; importlib.reload(dct_l0); from dct_l0 import *
os.makedirs("/content/results", exist_ok=True)
rig = Rig()
PROMPTS = json.load(open("/content/prompts.json")); TRAIN, HELD = PROMPTS["train"], PROMPTS["held_out"]
DIRS = {k: torch.tensor(v, dtype=torch.float32, device=rig.dev) for k, v in np.load("/content/directions.npz").items()}
b = rig.batch(TRAIN)
print("B,L", b["B"], b["L"], "| check", rig.check(b))                     # -> rel err 0.0 / 0.0
X, Y = rig.source_target(b)
first = (b["ATT"] == 0).sum(-1)
nX = X.float().norm(dim=-1); nY = Y.float().norm(dim=-1)
# layer 0: first token 21.7, other tokens median 12.3 (6.7..21.1); layer 20: first token 22986, others median 130.8

# === CELL 2 — "raw" fit: exactly DCT's Delta (unnormalised layer-20 states)
LOG = []
def log(s): print(s, flush=True); LOG.append(s)
df = DeltaFn(rig, b, X, Y, chunk=8)
t0 = time.time()
R, calib_hist, Uc, Vc = calibrate(df, rig.D, n_cal=30, lam=0.5, log=log)
U, V, objs, Sj = fit_exp_dct(df, rig.D, R, m=512, tau=10, d_proj=32, log=log)
alpha, DlRV = alphas(df, U, V, R)
np.savez("/content/results/dct_fit_raw.npz", U=U.cpu().numpy(), V=V.cpu().numpy(), alpha=alpha.cpu().numpy(), Delta_RV=DlRV.cpu().numpy(), R=R)
json.dump(dict(R=R, median_norm_l0=nX[b['SM']].median().item(), first_token_norm_l0=nX[torch.arange(b["B"]), first].mean().item(),
               calib_hist=calib_hist, objective=objs, jac_singular_values=Sj, log=LOG), open("/content/results/dct_fit_raw_log.json", "w"), indent=1)

# === CELL 2b — keep the raw fit
U_raw, V_raw, alpha_raw, R_raw, DlRV_raw = U.clone(), V.clone(), alpha.clone(), R, DlRV.clone()

# === CELL 3 — "norm" fit (headline): layer-20 token states read as c*h/||h||, c = median clean norm
LOG = []
C20 = float(nY[b["SM"]].median())
dfn = DeltaFn(rig, b, X, Y, chunk=8, norm_target=C20)
R, calib_hist, Uc, Vc = calibrate(dfn, rig.D, n_cal=30, lam=0.5, log=log)
U, V, objs, Sj = fit_exp_dct(dfn, rig.D, R, m=512, tau=10, d_proj=32, log=log)
alpha, DlRV = alphas(dfn, U, V, R)
np.savez("/content/results/dct_fit_norm.npz", U=U.cpu().numpy(), V=V.cpu().numpy(), alpha=alpha.cpu().numpy(), Delta_RV=DlRV.cpu().numpy(), R=R, c=C20)
json.dump(dict(variant="norm", R=R, c=C20, median_norm_l0=nX[b['SM']].median().item(), calib_hist=calib_hist, objective=objs, jac_singular_values=Sj, log=LOG),
          open("/content/results/dct_fit_norm_log.json", "w"), indent=1)

# === CELL 4 — slim copies for the repo (fp16 U, V; alpha; R; ||Delta(R v)||)
for v in ["raw", "norm"]:
    d = dict(np.load(f"/content/results/dct_fit_{v}.npz"))
    np.savez_compressed(f"/content/results/dct_{v}_UV_fp16.npz", U=d["U"].astype(np.float16), V=d["V"].astype(np.float16),
                        alpha=d["alpha"], R=d["R"], Delta_RV_norm=np.linalg.norm(d["Delta_RV"], axis=0))
