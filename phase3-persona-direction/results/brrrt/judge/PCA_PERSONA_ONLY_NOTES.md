# PCA on the persona replies only (2026-10-05, not yet in the phase 3 README)

Scripts: `code/brrrt/pca_persona_only.py`, `code/brrrt/pca_persona_opener.py`. Results: `pca_persona_only.json`,
`pca_persona_opener.json` and their `.log` files in this folder. Figure: `figures/pca_persona_only.png`.

Rows: 6,821 persona replies from the `probe_l0` arm (persona by both Gemma passes, no word salad, the
thinking-aloud register excluded). Main cell: layer 20, after 4 committed tokens. "Demeaned" = each question's
own mean subtracted, so no component can be "which question was asked".

## Findings

1. **No main axis among personas.** PC1 is 5.3% of the variance, the top 5 about 17%. 49 components for half the
   variance, 274 for 80%. Participation ratio 95.
2. **Only slightly more spread than the assistant.** At matched n (1,464, question means removed): participation
   ratio about 90 for personas vs 86 for assistant replies; 203 vs 172 components to reach 80%.
3. **The top two axes are stable.** Fit on even vs odd questions separately: |cos| 0.91 and 0.89. PC3 and PC4 swap
   between halves; PC4 is unstable.
4. **Families are a small part of it.** Family explains at most 3.4% of any top component; family centroids sit
   within ±0.6 SD while replies spread about ±3 SD. The clearest family signal is detective + aristocrat (PC2,
   +0.57 / +0.47 SD), the same pair that had cos 0.83 in the family-probe directions; it recurs at layers 8, 20, 32
   and is strongest at the first token (L20 R1 PC1, 12% of variance).
5. **The opening tokens explain far more than the family.** First token explains 23-24% of PC1 and PC2 (shuffled
   null 0.7%); family 3%. Of the whole state after the question, the first two tokens explain 12%, family 1%.
   Most common openers: " Suddenly" (656), " smoking" / " Smoking" (968), "," (356), " appearing" (197),
   " typing" (152): stage directions, the reply opening mid-action.
6. **Removing question and opener together, the axes keep their direction but lose most family structure.** New
   PC1-3 have |cos| 0.95 / 0.90 / 0.93 with the old ones and still replicate across question halves
   (0.87 / 0.75 / 0.73). Family eta^2 ≤ 0.03; detective's PC2 pole falls from 0.57 to 0.27 SD. Participation
   ratio rises to 115.
7. **Across tokens the space opens up**: participation ratio 29 after 1 token, 95 after 4.
8. **At 8 tokens (L20 R8, `pca_persona_opener_L20R8.json`) the opener fades and the axes firm up.** First token
   explains 5-10% of each top component (vs 23-24% at R4); first two tokens explain 4% of the whole state, family
   1.3%. With question and opener removed, PC1-3 replicate across question halves at |cos| 0.95 / 0.94 / 0.96 and
   are nearly identical to the un-residualised PCs (|cos| ≥ 0.97), so at R8 they are not opener artefacts.
   The leading axis reads as **narrated story vs talking to you**: narrative, detective, aristocrat at one end
   (+0.17 to +0.21 SD), hype, scholar, mentor at the other (−0.21 to −0.24 SD). PC3: mystic, seafarer, chef vs
   casual, aristocrat, narrative. Family still explains ≤ 3% of any component.

9. **Later in the reply (R12-R64; new capture, `late_pca_results.json`, activations on HF under `late/`).**
   Layer 20, question-demeaned, persona rows only (6,685-6,821 per slot; shorter replies drop out).

   | slot | participation ratio | PC1 var | first-token eta^2 PC1 | family eta^2 PC1 | split-half PC1 / PC2 | matched-n PR persona vs assistant |
   |---|---|---|---|---|---|---|
   | R4 | 95 | 5.4% | 0.23 | 0.03 | 0.91 / 0.89 | 91 vs 86 |
   | R8 | 96 | 5.8% | 0.09 | 0.06 | 0.93 / 0.91 | 91 vs 116 |
   | R12 | 103 | 5.5% | 0.05 | 0.07 | 0.84 / 0.81 | 96 vs 124 |
   | R16 | 100 | 5.5% | 0.05 | 0.07 | 0.80 / 0.80 | 95 vs 135 |
   | R24 | 114 | 5.2% | 0.01 | 0.01 | 0.90 / 0.94 | 106 vs 154 |
   | R32 | 139 | 4.7% | 0.01 | 0.01 | 0.91 / 0.90 | 124 vs 172 |
   | R48 | 171 | 3.9% | 0.01 | 0.01 | 0.87 / 0.80 | 152 vs 185 |
   | R64 | 190 | 3.7% | 0.01 | 0.004 | 0.88 / 0.73 | 170 vs 189 |

   - The persona space keeps widening through the reply (participation ratio 95 → 190).
   - From R8 on, **assistant replies are more spread out than persona replies** at matched n (e.g. R16: 135 vs 95).
     At R4 it was the other way round (86 vs 91).
   - The opener's grip fades by R24 (eta^2 ≤ 0.01), and family labels explain almost nothing by R24 (≤ 0.01).
   - The leading axis persists: best |cos| of PC1 with a top-3 PC at the next slot is 0.89, 0.90, 0.83, 0.85,
     0.97, 0.95, 0.99 (R4→R8 … R48→R64). PC1 stays reproducible across question halves (0.80-0.93).
   - Reading the poles (`late_pca_results.json`, `poles`): from R16 on, PC1 separates replies **still performing the
     scene** (narrated stage business, "appearing from the shadows… stroking his beard", a paranoid spy's monologue)
     from replies that have **turned to the task** in a direct or thinking voice ("Okay, so the user asked how to fix
     a leaking tap. Let me think", "Alright, let's break this down, kid"). This continues the R8 "how much character
     vs how much content" axis (R8 PC2). Sign is arbitrary per slot.
   - R16's opener-residualised PCA is unstable (split-half 0.005 / 0.03 on PC1/PC2, likely two near-equal
     components swapping); the question-demeaned PCA at R16 is stable (0.80 / 0.80).

## Reading

The persona space at layer 20 is diffuse, and its leading axes are shaped mostly by how the reply opens (which
stage direction, which first word), then by something stable that is not the judge's family label. The family
labels capture little of the internal variation.

## Caveats

- Removing the opener may over-correct: the voice partly *chooses* the opener (a pipe-smoking gentleman opens with
  " smoking"), so some real voice signal is removed with it.
- Family labels are regex buckets over judge descriptions; 30% of personas are unlabelled; "poet" is partly a
  poem-question detector (phase 3 README).
- Two components show 9% of the variance; the scatter is a thin slice of the space.

## Next checks not run

- Opener-matched comparison: within one opener (e.g. " Suddenly"), do families separate?
- Read actual replies at each end of the R8 PC1 to check the "narrated vs direct address" reading.
