# Results — steering with the tokens a bag-of-tokens classifier likes (2026-09-14)

**Question.** The token-debiased assistant direction keeps ~93 % of the raw direction's steering effect
(`RESULTS-tokdebias-steering.md`). The converse test: build a steering vector *only* from the tokens a bag-of-tokens
classifier uses to tell assistant from persona, inject it, and see whether it steers the same way as the assistant
direction.

**Run.** Qwen3-8B, vLLM 0.29.0 + vllm-lens 1.2.1 on a Colab A100, same rig as the token-debiased run: 14 arms × 5 evaluation
triggers × 48 seeds = 3,360 rollouts, T = 1.0, 96 tokens, response positions only, ε = ‖δ‖/‖residual‖ = 0.35 unless
marked ×2. ~0.5 A100-hour including install. Code: `code/steering/build_bag_vectors.py` (classifier, in-context vectors),
`code/steering/job_bag_vllm.py` (lone-token vectors, checks, rollouts), `code/steering/analyse_bag.py` (cheap columns,
same-seed similarity, blind corpus, judged tables), `code/steering/jlens_bag_vectors.py` (lens reading). Literal
input-embedding and unembedding vectors were drafted and dropped on request before any compute.

## The classifier and the tokens it likes

Binary presence of each token id in the first 4 response tokens, L2 logistic (C = 0.3, class-balanced), assistant vs
persona on unanimous labels, nulls excluded, fit on the 15 fit triggers. 164 tokens survive the ≥ 2-rollout filter.
**Held-out-trigger AUROC 0.70.** Its 20 most assistant-leaning and 20 most persona-leaning tokens, with coefficients and
the number of fit rollouts each appears in:

- **assistant:** 我 +0.57 (8), `,` +0.52 (46), 👋 +0.44 (5), もちろ +0.43 (6), んです +0.43 (6), Okay +0.42 (6), 抱歉 +0.37 (6),
  这句话 +0.35 (4), możesz +0.34 (4), `"` +0.33 (3), 鞨 +0.33 (3), UIView +0.30 (3), `，` +0.29 (13), Alright +0.27 (4),
  let +0.27 (4), あなた +0.26 (4), 像是 +0.24 (3), też +0.24 (3), and two fragments
- **persona:** `（` −0.87 (11), 끄 −0.56 (5), `*` −0.46 (5), cries −0.44 (3), 一看 −0.37 (3), อ −0.37 (3), /string −0.37 (3),
  Yo −0.36 (2), ARN −0.35 (3), `</think>` −0.31 (6), `？` −0.30 (2), brutal −0.28 (2), チャー −0.27 (2), 燃 −0.27 (2), and
  byte fragments

Most are rare (2–8 rollouts). The readable ones make sense: *Okay, Alright, 抱歉* ("sorry"), *もちろ(ん)* ("of course"),
*あなた* ("you") and *我* ("I") for the assistant; the full-width parenthesis that opens a stage direction, roleplay asterisks,
*cries*, *Yo* for the persona.

## Three vectors, all mid-layer representations of those tokens

| name | construction | top-dim share | cos to raw direction | cos to in-context vector |
|---|---|---|---|---|
| **in-context** (`bagnative`) | the model's own residual state right after writing each liked token in a real rollout (stored states at response positions 1, 2, 4), each position centred, weighted by the token's coefficient; 36 of 40 tokens found | 0.02–0.05 | **+0.77 to +0.82** (layers 12–28) | — |
| **lone token** (`bagalone`) | each liked token fed alone as a one-token input, all 40 in one batched forward pass; activation at the steered block, centred on the 40-token mean, weighted by coefficient | **0.94** | +0.05 | +0.03 to +0.10 |
| **token sequence** (`bagaloneseq`) | the assistant tokens as one sequence and the persona tokens as another, one pass each; weighted mean activation over positions 1 on, assistant minus persona | 0.01–0.02 | +0.56 to +0.59 | +0.65 to +0.71 |

⚠ **The lone-token vector is one coordinate.** A token fed alone sits at position 0, where every input shares the
attention-sink state (norm ~12,700). Centring across the 40 tokens removes the shared part (norm ~1,770 left), but the
tokens still differ mostly along that one massive-activation dimension, so 94 % of the vector's squared norm is on it. It
was run as specified; the sequence version, which drops position 0, is the lone-token idea without the sink.

⚠ **The in-context vector is not independent of the labels in state space.** States taken at the positions of
label-predictive tokens carry whatever else those states hold, so part of its +0.8 cosine with the raw direction is
construction. The lone-token and sequence vectors have no rollout context at all.

## Rig checks (all passed before any rollout)

- Prompt rebuild byte-for-byte; layer gate at feature layers 2–30 (index L − 1, min cos ≥ 0.99984, next best ≤ 0.96);
  clean first-token entropy 0.250 bits; hook checks on the in-context and lone-token vectors (prompt positions |Δ| = 0,
  response shift cos ≥ 0.9999, norm error ≤ 1.4 %, the largest on the lone-token vector's one huge coordinate in bf16).
- ⁂ **vLLM sampling reproduces exactly across runs.** The baseline, raw, token-debiased, random and clean arms are
  token-for-token identical to the same arms of the token-debiased run (240/240 each). Same-seed comparisons between
  arms therefore compare the steering, not sampling noise.

## Cheap columns and same-seed similarity to the raw direction

The first generated token is unsteered and always shared. "Same first 5" = the rollout's first 5 generated token ids
equal the raw-direction arm's for the same trigger and seed. Jaccard = word-4-gram overlap with that rollout.

| arm (layer 20 unless marked) | function-word rate | % Latin | same first 5 as raw | 4-gram Jaccard vs raw |
|---|---|---|---|---|
| baseline, no steering | 0.141 | 41.5 | 20.0 % | 0.027 |
| random | 0.149 | 47.3 | 17.9 % | 0.035 |
| raw assistant direction | 0.186 | 53.2 | 100 % | 1.000 |
| token-debiased direction | 0.189 | 49.7 | **68.3 %** | **0.103** |
| in-context bag vector | 0.188 | 49.3 | **42.1 %** | 0.046 |
| in-context bag vector, layer 16 | 0.165 | 46.5 | 45.0 % | 0.033 |
| in-context bag vector, layer 24 | 0.187 | 45.9 | 32.5 % | 0.033 |
| in-context bag vector ×2 | 0.239 | 56.1 | 12.1 % | 0.028 |
| in-context bag vector reversed | 0.115 | 43.3 | 7.5 % | 0.026 |
| lone-token vector | 0.145 | 45.5 | 19.2 % | 0.039 |
| lone-token vector ×2 | 0.160 | 46.8 | 16.7 % | 0.035 |
| token-sequence vector | 0.176 | 43.2 | 22.9 % | 0.035 |
| clean prompt + in-context bag vector | 0.416 | 100 | — | — |
| clean prompt, no steering | 0.423 | 100 | — | — |

- The token-debiased direction reproduces the raw direction's opening in two thirds of seeds. The in-context bag vector
  does so in about two fifths, the lone-token and sequence vectors no more often than no steering at all.
- By full-text overlap only the token-debiased direction is clearly above the floor; 96 tokens of free sampling diverge
  fast, so this column mostly separates "same opening" from "different opening".

## Blind panel

680 rollouts (60 per arm for the eight layer-20 comparison arms, 40 for the layer, dose, lone-token dose and
clean-prompt arms), shuffled together, arm labels stripped, two blind coders each (12 subagent coders), phase 17's rubric
verbatim. Agreement: raw 0.95–0.96, **Cohen's κ +0.86 persona, +0.89 default assistant, +0.89 on-topic**. A label
counts only when both coders give it. Files: `judging/bag_*`, `results/steer_bag_judged.json`.

| arm (layer 20 unless marked) | n | persona | default assistant | English | fully coherent | on-topic |
|---|---|---|---|---|---|---|
| baseline, no steering | 60 | 31.7 % | 28.3 % | 30.0 % | 46.7 % | 21.7 % |
| random | 60 | 33.3 % | 30.0 % | 35.0 % | 41.7 % | 26.7 % |
| **raw assistant direction** | 60 | **3.3 %** | **81.7 %** | 45.0 % | 76.7 % | 5.0 % |
| token-debiased direction | 60 | 6.7 % | 78.3 % | 43.3 % | 73.3 % | 11.7 % |
| **in-context bag vector** | 60 | **10.0 %** | **78.3 %** | 33.3 % | 75.0 % | 25.0 % |
| in-context bag vector, layer 16 | 40 | 2.5 % | 77.5 % | 37.5 % | 60.0 % | 17.5 % |
| in-context bag vector, layer 24 | 40 | 20.0 % | 57.5 % | 27.5 % | 72.5 % | 30.0 % |
| in-context bag vector ×2 | 40 | 0.0 % | 77.5 % | 52.5 % | **30.0 %** | 10.0 % |
| in-context bag vector reversed | 60 | 35.0 % | 5.0 % | 35.0 % | 23.3 % | 6.7 % |
| **lone-token vector** | 60 | **25.0 %** | **30.0 %** | 40.0 % | 43.3 % | 25.0 % |
| lone-token vector ×2 | 40 | 22.5 % | 35.0 % | 50.0 % | 40.0 % | 15.0 % |
| **token-sequence vector** | 60 | **16.7 %** | **58.3 %** | 31.7 % | 70.0 % | 40.0 % |
| clean prompt + in-context bag vector | 40 | 0.0 % | 100 % | 100 % | 100 % | 100 % |

Fisher exact, both coders:

| contrast | persona | p | default assistant | p |
|---|---|---|---|---|
| in-context bag vs raw direction | 6/60 vs 2/60 | 0.27 | 47/60 vs 49/60 | 0.82 |
| in-context bag vs token-debiased | 6/60 vs 4/60 | 0.74 | 47/60 vs 47/60 | 1.00 |
| in-context bag vs random | 6/60 vs 20/60 | **0.003** | 47/60 vs 18/60 | **< 0.001** |
| in-context bag, forward vs reversed | 6/60 vs 21/60 | **0.002** | 47/60 vs 3/60 | **< 0.001** |
| lone-token vs random | 15/60 vs 20/60 | 0.42 | 18/60 vs 18/60 | 1.00 |
| lone-token vs raw direction | 15/60 vs 2/60 | **0.001** | 18/60 vs 49/60 | **< 0.001** |
| token-sequence vs baseline | 10/60 vs 19/60 | 0.087 | 35/60 vs 17/60 | **0.002** |
| token-sequence vs raw direction | 10/60 vs 2/60 | **0.030** | 35/60 vs 49/60 | **0.009** |
| raw vs token-debiased (reference) | 2/60 vs 4/60 | 0.68 | 49/60 vs 47/60 | 0.82 |

### What the panel says

1. ⁂ **The in-context bag vector steers in the same direction and by about the same amount as the assistant direction.**
   Persona 31.7 % → 10.0 % and default assistant 28.3 % → 78.3 %, not significantly different from the raw direction
   (3.3 %, 81.7 %) or the token-debiased one (6.7 %, 78.3 %), and far from random. It reverses cleanly, works at layer 16
   (2.5 % persona), weakens at layer 24, and has no collateral on a clean prompt (100 % default assistant).
2. **But it does not steer exactly the same way.** Three things differ. English: 33 % against the raw direction's 45 %.
   On-topic: 25 % against 5 %, so its assistant replies more often actually answer the question. And same-seed openings:
   42 % match the raw direction's first five tokens against 68 % for the token-debiased direction. Summed over the five
   judged rates, its profile is 43 points from the raw direction's; the token-debiased direction's is 18. Same
   destination, different route.
3. **Doubling it breaks the text before it improves the register.** Persona reaches 0 % but fully coherent text falls
   from 75 % to 30 %. The raw direction at the same ε has no such problem, so the bag vector carries more that is not
   "assistant" per unit of norm.
4. ⁂ **The lone-token vector does nothing.** Persona 25 %, default assistant 30 %: indistinguishable from random and from
   no steering, at either dose, and its openings match the raw direction's no more than chance. That is what a vector
   that is 94 % one attention-sink coordinate should do. Feeding the tokens alone does not recover what they mean to the
   model in a reply.
5. **The token-sequence vector steers partway.** Default assistant 28 % → 58 % (p = 0.002), persona 32 % → 17 %
   (p = 0.09), significantly weaker than the raw direction on both. It has the highest on-topic rate of any arm (40 %).
   Putting the tokens in context with each other, rather than alone, recovers roughly half the effect.

**Answer to the question.** A steering vector built from the tokens a bag-of-tokens classifier likes steers the model
toward the assistant register only if it is built from the model's *in-context* representation of those tokens, and then
it steers about as strongly as the assistant direction but not identically. The same tokens' representations with no
context do not steer at all; with each other as context, about half. Read together with the token-debiased result (removing
token identity keeps 93 % of the effect), the register is not stored in which tokens get said: a vector that ignores token
identity steers, and the tokens alone do not.

⁂ **The panel reproduces an earlier panel on identical text.** Because vLLM sampling is exact across runs, 180 rollouts
in this corpus (baseline, raw, token-debiased and random arms) are token-for-token the same texts that the
token-debiased run's panel judged. Different coders, different corpus mix. Item by item the two panels agree on persona
**97.2 %** of the time (38 vs 43 personas) and on default assistant **95.6 %** (94 vs 92). Phase 17 §14 once measured a
25-point swing between panels; with the rubric and two-coder rule used here, the instrument is stable.

## What the vectors tell the model to say (Jacobian lens)

Pre-fitted lens `neuronpedia/jacobian-lens` for Qwen3-8B, read at the steered block, as in `RESULTS-tokdebias-steering.md`.
⚠ The lens's layer indexing and readout were taken from summaries of its source and have not been verified on real
states; read these as descriptions under those assumptions. ρ = Spearman correlation of the full lens-logit profile with
the raw direction's at block 19.

| vector | opener score (lens) | opener score (logit lens) | ρ vs raw | promotes | suppresses |
|---|---|---|---|---|---|
| raw direction | +0.58 | +0.20 | 1.00 | —you, —including, discussions, questions | scream, 尖叫, fuck, corpse, roar |
| token-debiased | +0.52 | +0.21 | 0.94 | soft hyphens, —including, —you | fuck, shit, scream, bitch |
| random | −0.13 | −0.22 | 0.08 | quote marks, colons | blank-line punctuation |
| in-context bag, L20 | +0.55 | +0.27 | 0.77 | blanks ____, …, questions | 诅 (curse), orgy, 尖叫, 恶魔 (demon), 血腥 (gory) |
| in-context bag, L24 | +0.72 | +0.48 | — | misunderstand, 让您, Your, you, 欢迎您 (welcome) | 尖叫, 暴力 (violence), scream, 尸体 (corpse) |
| lone-token | +0.06 | +0.20 | 0.25 | `<small`, `<pre`, `<![`, [url | spaces, dashes, digits |
| token sequence | +0.68 | +0.40 | 0.71 | 您, you, your, 您的, 您可以 ("you can") | 血腥, corpse, 尸体, 毁灭 (destruction), terror |

## What this does not establish

- One query, five evaluation triggers, one ε, layer 20 for most arms. Judged n is 40–60 per arm, so the in-context vector's
  gap to the raw direction (10.0 % vs 3.3 % persona) is not resolvable here; "about the same amount" means "not
  distinguishable at n = 60", not "equal". The bag classifier is weak (0.70) and its tokens are
  rare, so a better lexical model might build a better vector.
- The in-context vector borrows context from real rollouts (see the ⚠ above); the lone-token vector is dominated by one
  dimension. Neither is a clean "pure token" test; the sequence vector is the closest.
