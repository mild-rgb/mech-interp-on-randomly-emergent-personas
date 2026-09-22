import json,math
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Patch, FancyBboxPatch
S1,S2,INK,INK2,MUTED,GRID,SURF='#2a78d6','#eb6834','#0b0b0b','#52514e','#898781','#e1e0d9','#fcfcfb'
plt.rcParams.update({'font.family':'sans-serif','font.size':10})
def wilson(k,n,z=1.96):
    if n==0: return (0,0)
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return (100*(c-h),100*(c+h))
def fig(rows,title,sub,path,note=None):
    # rows: list of (label, n, persona_count, default_count)
    n=len(rows); H=0.36; gap=0.04
    f,ax=plt.subplots(figsize=(8.6,1.05+0.62*n),dpi=200); f.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
    ys=[(n-1-i) for i in range(n)]
    for y,(lab,N,kp,kd) in zip(ys,rows):
        for off,k,col in [(+H/2+gap/2,kp,S1),(-H/2-gap/2,kd,S2)]:
            v=100*k/N; lo,hi=wilson(k,N)
            ax.barh(y+off,v,height=H,color=col,edgecolor='none',zorder=2)
            ax.plot([lo,hi],[y+off,y+off],color=INK2,lw=1,zorder=3,solid_capstyle='butt')
            ax.plot([lo,lo],[y+off-0.06,y+off+0.06],color=INK2,lw=1,zorder=3); ax.plot([hi,hi],[y+off-0.06,y+off+0.06],color=INK2,lw=1,zorder=3)
            ax.text(hi+1.5,y+off,f'{v:.0f}%  ({k}/{N})',va='center',fontsize=8,color=INK2,zorder=4)
    ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows],color=INK,fontsize=9.5)
    ax.set_xlim(0,118); ax.set_xticks([0,20,40,60,80,100]); ax.set_xticklabels([f'{t}%' for t in [0,20,40,60,80,100]],color=MUTED)
    ax.set_ylim(-0.6,n-0.4)
    ax.tick_params(axis='y',length=0); ax.tick_params(axis='x',color=GRID); ax.grid(axis='x',color=GRID,lw=0.8,zorder=0); ax.set_axisbelow(True)
    for s in ['top','right','left']: ax.spines[s].set_visible(False)
    ax.spines['bottom'].set_color('#c3c2b7')
    ax.set_xlabel('share of rollouts labelled by both blind coders (95% Wilson interval)',color=INK2,fontsize=9.5)
    f.text(0.012,0.985,title,fontsize=11,color=INK,ha='left',va='top',fontweight='semibold'); f.text(0.012,0.985-0.30/(1.05+0.62*n),sub,fontsize=9,color=INK2,ha='left',va='top')
    ax.legend(handles=[Patch(color=S1,label='persona'),Patch(color=S2,label='default assistant')],loc='upper right',frameon=False,fontsize=9,labelcolor=INK2)
    if note: f.text(0.01,0.005,note,fontsize=7.5,color=MUTED,ha='left',va='bottom')
    top=1-0.62/(1.05+0.62*n)
    f.tight_layout(rect=(0,0.025 if note else 0,1,top)); f.savefig(path,facecolor=SURF); plt.close(f)
# ---- run 1 (2026-09-09), n=24 each; counts = round(pct*24/100)
j=json.load(open('results/steer_judge_summary.json'))
def c(k,field): return round(j[k][field]*j[k]['n']/100)
run1=[('baseline: trigger, no steering',24,c('0 baseline (trigger, no steer)','persona'),c('0 baseline (trigger, no steer)','default')),
 ('random direction, ε +0.35',24,c('7 RANDOM L20 eps+0.35','persona'),c('7 RANDOM L20 eps+0.35','default')),
 ('assistant direction (raw mass-mean), ε +0.35',24,c('2 massmean L20 eps+0.35','persona'),c('2 massmean L20 eps+0.35','default')),
 ('clean-minus-trigger direction, ε +0.35',24,c('8 clean_minus_trigger L20 e+0.35','persona'),c('8 clean_minus_trigger L20 e+0.35','default')),
 ('trigger-demeaned direction, ε +0.35 †',24,3,16),
 ('INLP-debiased direction, layer 16, ε +0.35 †',24,2,14),
 ('assistant direction, ε −0.35 (reversed)',24,c('4 massmean L20 eps-0.35 (reverse)','persona'),c('4 massmean L20 eps-0.35 (reverse)','default')),
 ('ceiling: English system prompt ‡',24,c('9 CEILING: system prompt','persona'),c('9 CEILING: system prompt','default'))]
fig(run1,'Run 1 (2026-09-09): steering on five held-out triggers, layer 20','Qwen3-8B · 24 rollouts per arm · blind two-coder panel, Cohen’s κ +0.94 on persona',
    'figures/fig_steer_run1_judged.png','† judged by a second panel anchored to the first (anchor arms agreed within 1 point).  ‡ this arm re-enabled thinking mode; its rerun was scored by proxy only.')
# ---- run 3 (2026-09-14)
b=json.load(open('results/steer_bag_judged.json'))['arms']
def r(k,lab): a=b[k]; return (lab,a['n'],a['persona_count'],a['default_count'])
run3=[r('B0 baseline','baseline: trigger, no steering'),r('B3 random + L20','random direction'),r('B1 raw fit15 + L20','assistant direction (raw, refit on 15 arms)'),
 r('B2 tokdebias r128 + L20','token-debiased assistant direction (r = 128)'),r('B4 bagnative + L20','in-context bag-token vector'),
 r('B9 bagnative + L16','in-context bag-token vector, layer 16'),r('B10 bagnative + L24','in-context bag-token vector, layer 24'),
 r('B11 bagnative x2 L20','in-context bag-token vector, ε ×2'),r('B5 bagnative - L20','in-context bag-token vector, reversed'),
 r('B17 bagaloneseq + L20','token-sequence vector'),r('B16 bagalone + L20','lone-token vector'),r('B18 bagalone x2 L20','lone-token vector, ε ×2'),
 r('B14 clean + bagnative + L20','clean prompt + in-context bag-token vector')]
fig(run3,'Run 3 (2026-09-14): vectors built from the tokens a bag-of-tokens classifier likes','Qwen3-8B · five held-out triggers · layer 20 and ε = 0.35 unless marked · blind two-coder panel, κ +0.86 on persona',
    'figures/fig_steer_run3_judged.png')
print('ok')
