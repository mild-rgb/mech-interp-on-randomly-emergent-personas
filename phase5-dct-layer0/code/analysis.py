# Phase 5 analysis helpers: compare DCT factors with phase 2/3 directions, read the layer-20 assistant axis, rollouts.
import torch, torch.nn.functional as F, math
from dct_l0 import S_LAYER, T_LAYER

GAP_INLP = 6.886      # phase 1: assistant minus persona gap on the INLP-debiased axis at layer 20 (phase 3 GAP)


def max_abs_cos(M, d):
    """M [D, m] unit columns, d [D] unit -> (max |cos|, argmax, signed cos at argmax, all cos)."""
    c = (M.t() @ d)
    i = int(c.abs().argmax())
    return float(c.abs()[i]), i, float(c[i]), c


def subspace_energy(M, d):
    """fraction of ||d||^2 inside span(M)."""
    Q, _ = torch.linalg.qr(M)
    return float((Q.t() @ d).pow(2).sum())


def random_baseline(D, m, d, trials=200, seed=0):
    """max |cos| of m random unit vectors with d, over trials; and subspace energy of random m-dim span."""
    g = torch.Generator().manual_seed(seed)
    mx, en = [], []
    for _ in range(trials):
        R = F.normalize(torch.randn(D, m, generator=g), dim=0).to(d.device)
        mx.append(float((R.t() @ d).abs().max()))
    for _ in range(20):
        R = torch.randn(D, m, generator=g).to(d.device)
        en.append(subspace_energy(R, d))
    t = torch.tensor(mx)
    return dict(max_abs_cos_mean=float(t.mean()), max_abs_cos_p95=float(t.quantile(0.95)), max_abs_cos_max=float(t.max()),
                subspace_energy_mean=float(torch.tensor(en).mean()), subspace_energy_expected=m / D)


@torch.no_grad()
def last_state(rig, b, Xs, chunk_rows=96):
    """run layers 1..20 on a steered source Xs [N*B, L, D] (N copies of batch b), return layer-20 last position [N*B, D]."""
    out = []
    B = b["B"]
    per = max(1, chunk_rows // B)
    for i in range(0, Xs.shape[0], per * B):
        x = Xs[i:i + per * B]
        out.append(rig.run_layers(x, b, S_LAYER + 1, T_LAYER, rep=x.shape[0] // B)[:, -1].float())
    return torch.cat(out)


@torch.no_grad()
def assistant_proj_fixed(rig, b, X, Theta, axis, chunk=8):
    """Theta [D, m] fixed-norm steering at steer-mask positions -> projection on axis at layer-20 last pos [m, B]."""
    SM = b["SM"].to(X.dtype)
    res = []
    for i in range(0, Theta.shape[1], chunk):
        th = Theta[:, i:i + chunk].t().to(X.dtype)
        Fn = th.shape[0]
        Xs = X.repeat(Fn, 1, 1) + (th[:, None, None, :] * SM[None, :, :, None]).reshape(Fn * b["B"], b["L"], -1)
        res.append((last_state(rig, b, Xs) @ axis).reshape(Fn, b["B"]))
    return torch.cat(res)


@torch.no_grad()
def assistant_proj_rel(rig, b, X, d, eps, axis, all_tokens=True):
    """phase 2/3 convention h += eps*||h||*d (all prompt tokens incl. first if all_tokens) -> [B] projections."""
    m = (b["ATT"].bool() if all_tokens else b["SM"]).to(X.dtype)
    Xs = X + eps * X.float().norm(dim=-1, keepdim=True).to(X.dtype) * d.to(X.dtype) * m[..., None]
    return last_state(rig, b, Xs) @ axis


# ---------------------------------------------------------------- rollouts (hook on layer 0, prefill only)
class Steer:
    """add a vector at the output of layers[0] on prompt positions during prefill only.
    mode 'fixed': h += vec (vec already scaled to norm R*k), first token skipped if skip_first.
    mode 'rel'  : h += eps * ||h|| * unit(vec), all prompt positions (phase 2/3 rig)."""
    def __init__(self, rig):
        self.rig, self.spec = rig, None
        self.h = rig.layers[S_LAYER].register_forward_hook(self._hook)

    def _hook(self, mod, inp, out):
        if self.spec is None: return out
        hs = out[0] if isinstance(out, tuple) else out
        if hs.shape[1] == 1: return out                        # decode step: untouched
        mode, vec, eps, mask = self.spec
        m = mask[:, :hs.shape[1]].to(hs.dtype)[..., None]
        if mode == "fixed": new = hs + vec.to(hs.dtype) * m
        else: new = (hs.float() + eps * hs.float().norm(dim=-1, keepdim=True) * F.normalize(vec.float(), dim=0) * m.float()).to(hs.dtype)
        return (new,) + tuple(out[1:]) if isinstance(out, tuple) else new


def script_of(text):
    cnt = {}
    for ch in text:
        if not ch.isalpha(): continue
        o = ord(ch)
        k = ("latin" if o < 0x250 else "cyrillic" if 0x400 <= o < 0x530 else "arabic" if 0x600 <= o < 0x700 else
             "hangul" if 0xAC00 <= o < 0xD7B0 or 0x1100 <= o < 0x1200 else "kana" if 0x3040 <= o < 0x3100 else
             "han" if 0x4E00 <= o < 0xA000 or 0x3400 <= o < 0x4DC0 else "other")
        cnt[k] = cnt.get(k, 0) + 1
    return max(cnt, key=cnt.get) if cnt else "none"


@torch.no_grad()
def rollouts(rig, steer, prompts, spec, seeds=2, new=96, seed=0):
    """spec = None | ('fixed', vec, None) | ('rel', unit_vec, eps).  All prompts x seeds in one left-padded batch."""
    b = rig.batch([p for p in prompts for _ in range(seeds)])
    if spec is None: steer.spec = None
    else:
        mode, vec, eps = spec
        steer.spec = (mode, vec, eps, b["SM"] if mode == "fixed" else b["ATT"].bool())
    torch.manual_seed(seed)
    out = rig.model.generate(input_ids=b["IDS"], attention_mask=b["ATT"], do_sample=True, temperature=1.0, top_p=1.0,
                             top_k=0, max_new_tokens=new, pad_token_id=rig.EOS)
    steer.spec = None
    res = []
    for r in range(b["B"]):
        g = out[r, b["L"]:].tolist(); n = g.index(rig.EOS) if rig.EOS in g else len(g)
        txt = rig.tok.decode(g[:n], skip_special_tokens=True)
        res.append(dict(prompt=prompts[r // seeds], seed=r % seeds, text=txt, script=script_of(txt), n_tok=n))
    return res
