#!/usr/bin/env python3
"""Build the steering candidate directions (steer_candidates.npz + steer_candidates_meta.json).

Recovered 2026-09-12 from the 2026-09-09 16:01 session log; this is the exact code that produced the
shipped file, saved so the HF card's "rerunning reproduces every figure" holds. Run from the folder
holding features_qwen_wide.npz, features_meta.json, labels.json and inlp_directions.npz.

!! KNOWN DEVIATION FROM PLAN-steering.md: `okA` below is every labelled non-null rollout, i.e. all 20
trigger arms. The plan called for a refit on 15 arms with the 5 evaluation arms held out. To do that,
mask `okA`, `okr` and the `~is_null` selections with `~np.isin(arms, HOLD)` where HOLD is
results_inlp.json["held_out_arms"]. See RESULTS-steering.md, correction of 2026-09-12.
"""
import json, numpy as np, re
X=np.load('features_qwen_wide.npz')['X']; meta=json.load(open('features_meta.json')); L=meta['layers']; S=meta['slots']
lab=json.load(open('labels.json')); arms=np.array([r['arm'] for r in lab]); is_null=np.array([r['is_null'] for r in lab])
def lb(f): return np.array([-1 if r[f] is None else r[f] for r in lab])
yA=lb('assistant_A'); yB=lb('broken_A')
eng=np.array([sum(v['language']=='english' for v in r['verdicts'])>=3 for r in lab])
STREET=re.compile(r"\bbro\b|slang|buddy|banter|peer|casual|chat|friend|hype|pidgin|spanglish|street|gamer|rapper|dj|skate|yo\b")
street=np.array([any(STREET.search((v['persona_label'] or '').lower()) for v in r['verdicts']) for r in lab])
inlp=np.load('inlp_directions.npz'); LAYERS=[8,12,16,20,24,28,32]; out={}; stats={}
def unit(v): return v/np.linalg.norm(v)
for l in LAYERS:
    li=L.index(l); F={s:X[:,li,S.index(s)].astype(np.float32) for s in S}
    resp=np.nanmean(np.stack([F[s] for s in ('R1','R2','R4','R8')]),0); okA=(yA>=0)&~is_null; okr=~np.isnan(resp[:,0])
    d={}
    d['massmean_Rmean']=F['Rmean'][okA&(yA==1)].mean(0)-F['Rmean'][okA&(yA==0)].mean(0)
    d['massmean_early']=resp[okA&(yA==1)].mean(0)-resp[okA&(yA==0)].mean(0)
    G=resp.copy()
    for a in set(arms): G[arms==a]-=G[arms==a].mean(0)
    d['demeaned_early']=G[okA&(yA==1)].mean(0)-G[okA&(yA==0)].mean(0)
    v=[]
    for s in ('R1','R2','R4'):
        w=inlp[f'L{l}_{s}__assistant_dir_demean_lang_broken'].astype(np.float32)[0]; sc=inlp[f'L{l}_{s}__scaler_scale'].astype(np.float32); v.append(unit(w/sc))
    d['inlp_debiased_early']=np.mean(v,0)
    pos=okA&(yA==1)&eng; neg=~is_null&okr&~pos&((yA==0)|(yB==1)|((yA==1)&~eng))
    d['english_assistant_vs_rest']=resp[pos].mean(0)-resp[neg].mean(0)
    d['english_vs_nonenglish_assistant']=resp[pos].mean(0)-resp[okA&(yA==1)&~eng].mean(0)
    d['street_balanced_early']=resp[okA&(yA==1)].mean(0)-0.5*(resp[okA&(yA==0)&street].mean(0)+resp[okA&(yA==0)&~street].mean(0))
    d['clean_minus_trigger_P']=F['P'][arms=='NULL-clean'].mean(0)-F['P'][~is_null].mean(0)
    d['clean_minus_trigger_resp']=resp[arms=='NULL-clean'].mean(0)-resp[~is_null&okr].mean(0)
    rng=np.random.default_rng(l); d['random_1']=rng.standard_normal(4096).astype(np.float32); d['random_2']=rng.standard_normal(4096).astype(np.float32)
    norm=float(np.mean(np.linalg.norm(resp[okr],axis=1)))
    for k,v in d.items():
        u=unit(v); gapA=float((resp[okA&(yA==1)]@u).mean()-(resp[okA&(yA==0)]@u).mean()); gapE=float((resp[pos]@u).mean()-(resp[neg]@u).mean())
        out[f'L{l}__{k}']=u.astype(np.float32); stats[f'L{l}__{k}']=dict(gap_assistant_minus_persona=gapA, gap_englishassistant_minus_rest=gapE, resid_norm=norm)
    print(f"L{l}: early-resp norm {norm:.0f} | gap(asst-persona)/gap(EngAsst-rest) along unit dirs: "+" ".join(f"{k}={stats[f'L{l}__{k}']['gap_assistant_minus_persona']:+.0f}/{stats[f'L{l}__{k}']['gap_englishassistant_minus_rest']:+.0f}" for k in d))
    if l==20:
        ks=list(d); U=np.stack([unit(d[k]) for k in ks]); M=U@U.T; print("  cos at L20:")
        for i,k in enumerate(ks): print(f"    {k:<32}"+" ".join(f"{M[i,j]:+.2f}" for j in range(len(ks))))
np.savez('steer_candidates.npz', **out); json.dump(dict(layers=LAYERS, stats=stats, names=list(d), n_english_assistant=int(pos.sum())), open('steer_candidates_meta.json','w'), indent=1); print('saved steer_candidates.npz', len(out))
