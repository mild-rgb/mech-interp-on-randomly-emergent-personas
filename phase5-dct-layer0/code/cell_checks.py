# === checks run after cell_compare.py (in the Colab kernel). Results were written into results/compare.json under
# persona_strength_sweep, Delta_ref_random_baseline, top_drop_features_cos, token_embedding_alignment.
p = DIRS["persona_l0"]

# 1. persona direction strength sweep on the layer-20 assistant axis (all 40 prompts, gap units)
def relcustom(scale_tok):                       # scale_tok [40, L]: norm of the push at each token
    Xs = XA + (scale_tok[..., None] * p).to(XA.dtype)
    return float(((last_state(rig, bA, Xs) @ ax - base) / GAP_INLP).mean())
nrm = XA.float().norm(dim=-1); am = bA["ATT"].float(); sm = bA["SM"].float()
print("const 0.4*median all tokens:", relcustom(0.4 * nrm[bA['SM']].median() * am))
print("const 0.4*median no first:", relcustom(0.4 * nrm[bA['SM']].median() * sm))
for e in [0.2, 0.3, 0.4, 0.5, 0.6]: print("rel", e, "all:", relcustom(e * nrm * am))

# 2. baseline for "Delta of a reference direction lies in span(U)": same number for random directions
DR = dfn.many(R * Vr[:, :16])
print("random-direction Delta energy in U:", float(torch.tensor([subspace_energy(U, F.normalize(DR[:, i], dim=0)) for i in range(16)]).mean()))

# 3. do DCT input directions look like token embeddings? (top tokens by cosine with the input embedding matrix)
En = F.normalize(rig.model.model.embed_tokens.weight.float(), dim=1)
def toptok(v, k=4):
    c = En @ F.normalize(v.float(), dim=0); t = c.topk(k)
    return [(rig.tok.decode([int(i)]), round(float(x), 3)) for x, i in zip(t.values, t.indices)]
mx_all = (En @ V.float()).max(0).values; mx_r = (En @ Vr.float()).max(0).values
print("max token-embedding cos: DCT median", float(mx_all.median()), "| random median", float(mx_r.median()), "max", float(mx_r.max()))
