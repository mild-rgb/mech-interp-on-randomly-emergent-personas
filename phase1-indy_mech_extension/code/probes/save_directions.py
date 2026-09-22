#!/usr/bin/env python3
"""Fit mass-mean directions on ALL 528 labelled rollouts for each target and every (layer, slot),
and save them (fp16) so they can be applied or projected out without the features."""
import json, numpy as np
X=np.load('features_qwen_wide.npz')['X']; meta=json.load(open('features_meta.json')); L,S=meta['layers'],meta['slots']
lab=json.load(open('labels.json'))
def lb(f): return np.array([-1 if r[f] is None else r[f] for r in lab])
T={'assistant_A':lb('assistant_A'),'assistant_B':lb('assistant_B'),'broken_A':lb('broken_A'),'broken_B':lb('broken_B')}
out={}
for t,y in T.items():
    for li,l in enumerate(L):
        for si,s in enumerate(S):
            F=X[:,li,si].astype(np.float32); k=(y>=0)&~np.isnan(F[:,0])
            w=F[k&(y==1)].mean(0)-F[k&(y==0)].mean(0); mu=F[k].mean(0)
            proj=F[k]@w; thr_mid=float((proj[y[k]==1].mean()+proj[y[k]==0].mean())/2)
            out[f'{t}__L{l}_{s}__w']=w.astype(np.float16); out[f'{t}__L{l}_{s}__mean']=mu.astype(np.float16)
            out[f'{t}__L{l}_{s}__thr_mid']=np.float32(thr_mid)
np.savez_compressed('persona_broken_directions.npz', **out); print('saved persona_broken_directions.npz', len(out)//3, 'cells')
