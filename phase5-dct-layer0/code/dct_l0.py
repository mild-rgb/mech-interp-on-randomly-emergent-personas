# Phase 5 — exponential Deep Causal Transcoding (Mack & Turner 2024) from layer 0 to layer 20 of Qwen3-8B.
#
# Same algorithm as amack315/melbo-dct-post src/dct.py (ExponentialDCT + SteeringCalibrator), rewritten to run
# layers 1..20 directly instead of mutating model.model.layers (which breaks on transformers 5.x), with left padding
# and a mask over which token positions are steered / read.
#
#   source s = output of model.model.layers[0]    (phase 2/3's hook point)
#   target t = output of model.model.layers[20]   (phase 1's assistant-axis read point)
#   Delta(theta) = mean over prompts of mean over target positions of [h_t(h_s + theta * steer_mask) - h_t(h_s)]
#
# The steering vector is added with ONE fixed norm R at every steered position (DCT convention), not phase 2/3's
# eps * ||h_t|| per token. SINK=True (default) leaves the first token (<|im_start|>, the attention sink) unsteered and
# drops it from the Delta average.
import torch, torch.nn.functional as F, math, json, time
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID, S_LAYER, T_LAYER = "Qwen/Qwen3-8B", 0, 20


class Rig:
    def __init__(self, dtype=torch.bfloat16):
        self.tok = AutoTokenizer.from_pretrained(MODEL_ID)
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype, device_map="cuda:0",
                                                          attn_implementation="eager")
        self.model.eval(); self.model.requires_grad_(False)
        self.dev, self.D = self.model.device, self.model.config.hidden_size
        self.EOS = self.tok.eos_token_id
        self.layers = self.model.model.layers

    def chat(self, t):
        return self.tok.apply_chat_template([{"role": "user", "content": t}], tokenize=False,
                                            add_generation_prompt=True, enable_thinking=False)

    def batch(self, prompts, sink=True):
        """left-padded ids, attention mask, position ids, 4D additive mask, steer/target mask, rotary embeddings."""
        ids = [self.tok(self.chat(p), add_special_tokens=False)["input_ids"] for p in prompts]
        B, L = len(ids), max(map(len, ids))
        IDS = torch.full((B, L), self.EOS, dtype=torch.long); ATT = torch.zeros((B, L), dtype=torch.long)
        for r, x in enumerate(ids): IDS[r, L - len(x):] = torch.tensor(x); ATT[r, L - len(x):] = 1
        IDS, ATT = IDS.to(self.dev), ATT.to(self.dev)
        POS = (ATT.cumsum(-1) - 1).clamp(min=0)
        keep = torch.tril(torch.ones(L, L, dtype=torch.bool, device=self.dev))[None] & ATT.bool()[:, None, :]
        keep = keep | torch.eye(L, dtype=torch.bool, device=self.dev)[None]          # pad rows attend to themselves
        M4 = torch.zeros((B, 1, L, L), dtype=self.model.dtype, device=self.dev)
        M4.masked_fill_(~keep[:, None], torch.finfo(self.model.dtype).min)
        SM = ATT.bool().clone()
        if sink: SM[torch.arange(B), (ATT == 0).sum(-1)] = False                       # first real token
        with torch.no_grad():
            h = self.model.model.embed_tokens(IDS)
            cos, sin = self.model.model.rotary_emb(h, POS)
        return dict(prompts=prompts, IDS=IDS, ATT=ATT, POS=POS, M4=M4, SM=SM, cos=cos, sin=sin, B=B, L=L)

    def run_layers(self, h, b, lo, hi, rep=1):
        """run layers lo..hi inclusive on h [rep*B, L, D] (rep copies of batch b, factor-major)."""
        M4, cos, sin, POS = (x.repeat(rep, *([1] * (x.dim() - 1))) for x in (b["M4"], b["cos"], b["sin"], b["POS"]))
        for l in range(lo, hi + 1):
            out = self.layers[l](h, attention_mask=M4, position_ids=POS, position_embeddings=(cos, sin))
            h = out[0] if isinstance(out, tuple) else out
        return h

    @torch.no_grad()
    def source_target(self, b):
        h = self.model.model.embed_tokens(b["IDS"])
        X = self.run_layers(h, b, 0, S_LAYER)
        Y = self.run_layers(X, b, S_LAYER + 1, T_LAYER)
        return X, Y

    @torch.no_grad()
    def check(self, b):
        """our slice must match the model's own forward (hidden_states[l+1] = output of layers[l])."""
        hs = self.model(input_ids=b["IDS"], attention_mask=b["ATT"], position_ids=b["POS"],
                        output_hidden_states=True).hidden_states
        X, Y = self.source_target(b); a = b["ATT"].bool()
        rel = lambda u, v: float((u[a] - v[a]).float().norm() / v[a].float().norm())
        return dict(src_rel_err=rel(X, hs[S_LAYER + 1]), tgt_rel_err=rel(Y, hs[T_LAYER + 1]))


class DeltaFn:
    """Delta(theta) for a stack of steering vectors theta [F, D] -> [F, D] (float32)."""
    def __init__(self, rig, b, X, Y, chunk=8, norm_target=None):
        """norm_target=c: read each layer-t token state as c * h/||h|| (steered and clean), so a factor cannot score by
        blowing up one token's norm (a new massive activation); only the state's direction counts."""
        self.rig, self.b, self.X, self.chunk, self.c = rig, b, X, chunk, norm_target
        self.Y = Y if norm_target is None else self._nrm(Y.float())
        self.SM = b["SM"].to(X.dtype)                                    # [B, L] steer mask (= target mask)
        self.n = b["SM"].sum(-1).float()                                 # target positions per prompt

    def _nrm(self, h):
        return h if self.c is None else self.c * h / h.norm(dim=-1, keepdim=True).clamp(min=1e-3)

    def __call__(self, theta, lo=None, hi=None):
        Fn, B = theta.shape[0], self.b["B"]
        h = self.X.repeat(Fn, 1, 1) + (theta.to(self.X.dtype)[:, None, None, :] * self.SM[None, :, :, None]).reshape(Fn * B, self.b["L"], -1)
        out = self.rig.run_layers(h, self.b, S_LAYER + 1, T_LAYER, rep=Fn)
        d = (self._nrm(out.float()) - self.Y.float().repeat(Fn, 1, 1)).reshape(Fn, B, self.b["L"], -1)
        d = (d * self.SM.float()[None, :, :, None]).sum(2) / self.n[None, :, None]   # [F, B, D]
        return d.mean(1)

    @torch.no_grad()
    def many(self, Theta):
        """Theta [D, m] -> Delta [D, m], no grad, chunked."""
        return torch.cat([self(Theta[:, i:i + self.chunk].t()).t() for i in range(0, Theta.shape[1], self.chunk)], 1)

    def jvp(self, V):
        """J V at theta = 0 via forward-mode AD.  V [D, m] -> [D, m]."""
        outs = []
        for i in range(0, V.shape[1], self.chunk):
            v = V[:, i:i + self.chunk].t().contiguous()
            _, t = torch.func.jvp(lambda th: self(th), (torch.zeros_like(v),), (v,))
            outs.append(t.detach().t())
        return torch.cat(outs, 1)

    def vjp(self, W):
        """J^T W at theta = 0.  W [D, k] -> [D, k]."""
        outs = []
        for i in range(0, W.shape[1], self.chunk):
            w = W[:, i:i + self.chunk].t().contiguous()
            th = torch.zeros_like(w, requires_grad=True)
            with torch.enable_grad():
                (self(th) * w).sum().backward()
            outs.append(th.grad.detach().t())
        return torch.cat(outs, 1)

    def causal_step(self, U, V, R):
        """for each factor l: Delta(R v_l) and grad_v <u_l, Delta(R v_l)>.  Returns (G_U [D,m], G_V [D,m], obj)."""
        GU, GV, obj = [], [], 0.0
        for i in range(0, V.shape[1], self.chunk):
            v = V[:, i:i + self.chunk].t().contiguous().requires_grad_(True)
            u = U[:, i:i + self.chunk].t()
            with torch.enable_grad():
                d = self(R * v); f = (d * u).sum(); f.backward()
            GU.append(d.detach().t()); GV.append(v.grad.detach().t()); obj += float(f.detach())
        return torch.cat(GU, 1), torch.cat(GV, 1), obj


def calibrate(df, D, n_cal=30, lam=0.5, seed=0, log=print):
    """Mack's SteeringCalibrator: find R with sqrt(mean_l ||Delta(R v_l) - R J v_l||^2 / ||R J v_l||^2) = lam."""
    g = torch.Generator().manual_seed(seed)
    Vc = F.normalize(torch.randn(D, n_cal, generator=g), dim=0).to(df.X.device)
    Uc = df.jvp(Vc); nu = Uc.norm(dim=0)
    def ratio(r):
        Dl = df.many(r * Vc)
        return math.sqrt((((Dl - r * Uc).pow(2).sum(0)) / (r * nu).pow(2)).mean().item())
    hist = {}
    lo, hi = None, None
    r = 1.0
    for _ in range(40):                                    # bracket in log space
        e = ratio(r); hist[r] = e; log(f"  calib r={r:.4g} E={e:.4f}")
        if e < lam: lo = r; r *= 2
        else: hi = r; r /= 2
        if lo is not None and hi is not None: break
    for _ in range(14):                                    # geometric bisection
        r = math.sqrt(lo * hi); e = ratio(r); hist[r] = e
        if e < lam: lo = r
        else: hi = r
    R = math.sqrt(lo * hi); log(f"  calibrated R = {R:.4f}")
    return R, {str(k): v for k, v in sorted(hist.items())}, Uc, Vc


def qr_pos(V):
    """QR with the sign of each column kept (diag(R) > 0), i.e. Gram-Schmidt.  torch.linalg.qr may flip columns, and
    because the exponential DCT is not symmetric (Delta(-Rv) != -Delta(Rv)) a flipped v_l no longer matches its u_l:
    with plain QR the causal objective went 3157 -> 2303 -> -805 in our first run.  (Mack's code calls plain QR.)"""
    Q, Rm = torch.linalg.qr(V)
    return Q * torch.sign(torch.diagonal(Rm)).clamp(min=0).mul(2).sub(1)[None, :]


def fit_exp_dct(df, D, R, m=512, tau=10, d_proj=32, seed=0, log=print):
    """Mack's ExponentialDCT.fit with init='jacobian', beta=1.  Returns U, V [D, m], objective history."""
    g = torch.Generator().manual_seed(seed + 1)
    t0 = time.time()
    W = F.normalize(torch.randn(D, d_proj, generator=g), dim=0).to(df.X.device)
    JT = df.vjp(W)                                         # [D, d_proj] = rows of J^T W
    _, S, Vh = torch.linalg.svd(JT.t(), full_matrices=False)
    # top d_proj right singular vectors, then random directions orthogonalised against them (Mack's code fills the
    # rest of the m columns from the full SVD basis; this is the same idea written explicitly)
    rest = torch.randn(D, m - Vh.shape[0], generator=g).to(df.X.device)
    V = qr_pos(torch.cat([Vh.t(), rest], 1))
    U = F.normalize(df.many(R * V), dim=0)
    log(f"  init done {time.time()-t0:.0f}s; top jacobian singular values {S[:5].tolist()}")
    objs = []
    for it in range(tau):
        V = qr_pos(V)
        GU, GV, obj = df.causal_step(U, V, R)
        U, V = F.normalize(GU, dim=0), F.normalize(GV, dim=0)
        objs.append(obj); log(f"  iter {it} causal objective {obj:.2f}  ({time.time()-t0:.0f}s)")
    return U, V, objs, S.tolist()


def alphas(df, U, V, R):
    """Mack's ExponentialDCT.rank(target_vec=None): alpha = K^-1 (<u_l, Delta(R v_l)>)_l, K = (U'U) * expm1(V'V)."""
    Dl = df.many(R * V)
    a = (Dl * U).sum(0)
    K = (U.t() @ U) * torch.expm1(V.t() @ V)
    return torch.linalg.solve(K.double(), a.double()).float(), Dl
