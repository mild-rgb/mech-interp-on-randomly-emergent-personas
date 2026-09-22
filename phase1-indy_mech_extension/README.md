# indy_mech_extension — linear probes on Qwen3-8B for "will this reply be the assistant, and will it be broken?"

> ### ⚠ Data notice
> The large activation tensors are **not in git** — they are on Hugging Face:
> **https://huggingface.co/datasets/mild-rgb/indy-mech-extension-qwen3-8b-persona-probes**
> (14.3 GB; `features_qwen_wide.npz`, `noprompt/`, `phase19/*.npy`, fitted direction matrices).
> `data/` here keeps only the small JSON: ids, labels and metadata.
>
> That dataset also republishes **`phase19/phase19_raw.json`**, 8,000 unfiltered generations produced
> under an adversarial prefix, **a small number of which are hate speech** (2 confirmed by both blind
> coders, 79 screened HATE/SLUR). It is the corpus behind this repo's own `phase19/` rate measurement
> and is already tracked here. It is **model output under attack conditions, published for
> reproducibility of a safety result — not endorsed, and not for training or generative use.**
> Nothing outside `phase19/` carries that content: the rest is persona and word-salad output from the
> neutral query *"what shall i do today"*.


**Run 2026-09-09; steering follow-up 2026-09-14 (result 5).** Extension of `CoT-spiking/phase17`. Plan in `PLAN.md`. All features, code and results are in the
public HF dataset [`mild-rgb/indy-mech-extension-qwen3-8b-persona-probes`](https://huggingface.co/datasets/mild-rgb/indy-mech-extension-qwen3-8b-persona-probes).

## Setup

- **Rollouts:** phase 17's 528 Qwen3-8B rollouts (20 optimised triggers + clean null + random-prefix null, 24 seeds each,
  query `what shall i do today`, T=1.0, 96 tokens). Exact token ids rebuilt with phase 17's rig (462 re-tokenised, 66
  regenerated from seed, 0 failures; clean H1 reproduced at 0.2366 vs 0.237).
- **Features:** one teacher-forced forward pass per rollout; residual stream at every 2nd layer (0..36) at 10 positions:
  `P` = last prompt token (before any sampling), `R1..R64` = after k committed response tokens, `Rmean`, `Rlast`. fp16.
- **Labels:** four blind coders per rollout (phase 17's two panels). *Tier A, unanimous:* 226 assistant vs 144 persona;
  59 broken (coherence ≤ 1) vs 312 fluent. *Tier B:* both full-sweep coders assistant (240) vs rest (288); ≥ 2 of 4
  coders broken (153) vs rest (375).
- **Probes:** mass-mean (difference of class means) at all 190 cells; L2 logistic (C = 0.1, class-balanced) at 5 layers.
- **Split:** leave-one-trigger-out (22 folds) throughout. Controls: random split, per-arm base rate, within-arm label
  shuffle (10 perms), a **bag-of-tokens** logistic on the first k response token ids, and a **no-prompt** rerun of
  the same response ids with nothing in front of them.

## Result 1 — the early window is readable 2–4 tokens in, with or without the prompt

> ⚠ **Correction (2026-09-12). The no-prompt rows below were an instrument artifact, and the section's
> original title ("the early window needs the prompt in context") does not survive it.** The no-prompt
> control was scored with mass-mean on raw fp16 residuals. In the no-prompt condition one dimension
> holds 73–95 % of the squared class-mean difference at 2–4 tokens in layers 8–32 (a massive-activation
> dimension; with the prompt in front the same dimension holds 33–61 %), and it swamps the dot
> product, so those cells read at chance. With per-dimension z-scoring (no labels used) the same
> states separate the classes about as well as the prompt-conditioned ones. Found while running INLP
> on the no-prompt features, whose script z-scores by default. Check and rerun:
> `code/probes/noprompt_zscore_check.py`, results in `results/results_noprompt_zscored.json`. The
> original tables are kept below as run; the corrected ones follow them.

Leave-one-trigger-out AUROC, unanimous labels, nulls excluded. "Best layer" is chosen per column from layers 2–34.

**Assistant vs persona — as run 2026-09-09, mass-mean on raw residuals (no-prompt rows superseded)**

| tokens seen | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| bag of tokens (text only) | 0.68 | 0.72 | 0.73 | 0.75 | 0.79 | 0.85 | 0.91 |
| hidden state, no prompt, own best layer | 0.69 | 0.64 | 0.69 | 0.80 | 0.84 | 0.87 | 0.90 |
| hidden state, with prompt, mass-mean, best layer | 0.77 | 0.79 | 0.84 | 0.83 | 0.88 | 0.92 | 0.93 |
| hidden state, with prompt, logistic C=0.1, best of 5 layers | 0.83 | 0.80 | 0.83 | 0.89 | 0.91 | 0.94 | 0.96 |

**Broken vs fluent — as run 2026-09-09, mass-mean on raw residuals (no-prompt rows superseded)**

| tokens seen | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| bag of tokens (text only) | 0.72 | 0.76 | 0.81 | 0.86 | 0.92 | 0.95 | 0.96 |
| hidden state, no prompt, own best layer | 0.68 | 0.66 | 0.78 | 0.84 | 0.84 | 0.87 | 0.85 |
| hidden state, with prompt, mass-mean, best layer | 0.75 | 0.83 | 0.89 | 0.93 | 0.94 | 0.94 | 0.90 |
| hidden state, with prompt, logistic C=0.1, best of 5 layers | 0.74 | 0.81 | 0.87 | 0.95 | 0.96 | 0.96 | 0.93 |

**Corrected (2026-09-12): mass-mean on z-scored residuals, both conditions, own best layer 2–34, beside the
bag-of-tokens baseline** (`results/results_noprompt_zscored.json`; bag from `results/results_lexical.json`, unchanged
by z-scoring; the mean-pooled row is compared with the bag of all 96 tokens, the last-token row with the last token id alone)

| assistant vs persona, tokens seen | full rollout (layer) | no prompt (layer) | bag of tokens | full − bag | no-prompt − bag | full − no-prompt |
|---|---|---|---|---|---|---|
| 1 | 0.81 (L32) | 0.73 (L2) ⚠ sink slot | 0.68 | +0.13 | +0.05 | +0.08 |
| 2 | 0.78 (L24) | 0.77 (L10) | 0.72 | +0.06 | +0.05 | +0.01 |
| 4 | 0.84 (L32) | 0.82 (L10) | 0.73 | +0.11 | +0.09 | +0.02 |
| 8 | 0.85 (L20) | 0.82 (L20) | 0.75 | +0.10 | +0.07 | +0.03 |
| 16 | 0.89 (L20) | 0.84 (L34) | 0.79 | +0.10 | +0.05 | +0.05 |
| 32 | 0.93 (L28) | 0.89 (L30) | 0.85 | +0.08 | +0.04 | +0.03 |
| 64 | 0.93 (L20) | 0.91 (L20) | 0.91 | +0.02 | 0.00 | +0.02 |
| mean-pooled | 0.97 (L12) | 0.97 (L10) | 0.94 (all 96) | +0.03 | +0.03 | 0.00 |
| last token | 0.93 (L20) | 0.91 (L20) | 0.48 (last id only) † | +0.45 | +0.43 | +0.02 |

| broken vs fluent, tokens seen | full rollout (layer) | no prompt (layer) | bag of tokens | full − bag | no-prompt − bag | full − no-prompt |
|---|---|---|---|---|---|---|
| 1 | 0.76 (L16) | 0.76 (L2) ⚠ sink slot | 0.72 | +0.04 | +0.04 | +0.01 |
| 2 | 0.84 (L18) | 0.80 (L34) | 0.76 | +0.08 | +0.04 | +0.04 |
| 4 | 0.89 (L26) | 0.87 (L20) | 0.81 | +0.08 | +0.06 | +0.02 |
| 8 | 0.93 (L22) | 0.92 (L20) | 0.86 | +0.07 | +0.06 | +0.01 |
| 16 | 0.94 (L24) | 0.94 (L20) | 0.92 | +0.02 | +0.02 | 0.00 |
| 32 | 0.95 (L22) | 0.95 (L20) | 0.95 | 0.00 | 0.00 | 0.00 |
| 64 | 0.92 (L18) | 0.91 (L20) | 0.96 | −0.04 | −0.05 | +0.01 |
| mean-pooled | 0.99 (L14) | 0.99 (L6) | 0.95 (all 96) | +0.04 | +0.04 | +0.01 |
| last token | 0.92 (L22) | 0.93 (L28) | 0.61 (last id only) † | +0.31 | +0.32 | −0.01 |

The raw-residual with-prompt cells (tables above) move by at most 0.04 under z-scoring (the 1-token persona cell,
0.77 → 0.81); the no-prompt cells move by up to 0.13 at 2–4 tokens and 0.10 mean-pooled. Both hidden-state probes lead
the bag by 0.05–0.11 in the 2–16-token window for persona and 0.04–0.08 at 2–8 tokens for broken, and the bag catches up
at 32–64 tokens. The `full − no-prompt` column is tested in `results/results_prompt_gap_tests_L20.json` (see below).
† The last-token bag is the strict positional analog: a logistic probe on the final token id only
(`code/probes/lexical_lastk.py`, `results/results_lexical_lastk.json`). It is at chance for persona and weak for broken, while
the last-token hidden state, which has attended to the whole response, reads both at 0.92–0.93. Bags of the last k ids for
k = 4 / 8 / 16 / 32 / 96 read persona at 0.63 / 0.67 / 0.79 / 0.83 / 0.94 and broken at 0.74 / 0.82 / 0.84 / 0.89 / 0.95.

![prefix curve](fig_prefix_curve.png)

- At 2–8 committed tokens the prompt-conditioned hidden state leads the text-only baseline by ~0.07–0.12 (persona) and
  ~0.07–0.09 (broken). ~~**The no-prompt control removes that lead**: with nothing in front of the response the hidden
  state is at or below bag-of-tokens at 2–4 tokens. So the early signal is not contextual reading of the text; it is
  state that exists only with the trigger and query in context.~~ ⚠ **Corrected 2026-09-12: the no-prompt control
  does not remove the lead.** Z-scored, the no-prompt state also beats bag-of-tokens at 2–4 tokens (0.77 / 0.82 against
  0.72 / 0.73 for persona; 0.80 / 0.87 against 0.76 / 0.81 for broken), and the prompt adds only 0.02 for persona and
  0.02–0.04 for broken on top of it. So the early signal is mostly the model reading its own first tokens — a
  contextual reading the bag-of-tokens baseline cannot do, but one that does not need the trigger in front. What the
  prompt adds is small: tested directly (`code/probes/prompt_gap_tests.py`, layer 20 fixed for both conditions,
  `results/results_prompt_gap_tests_L20.json`), the with-prompt minus no-prompt gap is +0.02 to +0.05 at every
  position from 2 tokens on, **not significant at 2 tokens** (12/20 folds, Wilcoxon p = 0.33, bootstrap p = 0.10),
  borderline at 4 (10/20, p = 0.07 / 0.02), and clearest at 32 (14/20, p = 0.046 / 0.007). For the broken target no
  position passes both tests. The 1-token row (+0.15, p = 0.004) is the no-prompt attention-sink slot and is not
  evidence of a prompt effect. Per-condition best-layer versions are in `results_prompt_gap_tests.json` and agree.
- **What the full-rollout probe reads that the no-prompt probe does not** (`code/probes/prompt_vs_noprompt_probes.py`,
  `results/results_prompt_vs_noprompt_probes.json`, 2026-09-14). As raw-space readers the two probes have cosine
  0.23–0.50 at 2 tokens, rising to 0.85–0.93 mean-pooled. Either probe applied to the other condition's states loses
  ≤ 0.05 AUROC, so both read features present in both conditions. The part of the full-rollout probe orthogonal to the
  no-prompt probe sits in the INLP trigger-identity subspace at 28–45 % of its norm (floor 2–3 %), aligns with result
  2's prompt-state direction (cos +0.15 to +0.39 at layers 20–28, where the no-prompt probe's is 0.00–0.17), and the
  per-trigger gap tracks the trigger's assistant rate (Spearman +0.4 to +0.5 at 2–8 tokens, layer 20). So the extra
  thing is the trigger's prior carried into the response, i.e. between-arm information — which is what the within-arm
  check strips, and why the gap shrinks there.
- **Is the lead just the trigger's prior?** Mostly, and the two tests above already separate the two kinds of lead: a
  per-fold AUROC is computed inside one trigger and cannot use between-trigger ranking, so the Wilcoxon over folds is
  the within-trigger test, while the bootstrap on pooled AUROC includes the prior. At layer 20 the pooled gap is
  significant from 4 tokens on (bootstrap p = 0.02 / 0.03 / 0.005 / 0.007 / 0.03 at 4 / 8 / 16 / 32 / 64) and the
  within-trigger gap only at 32 (Wilcoxon p = 0.07 / 0.33 / 0.09 / 0.046 / 0.36). The difference is the prior. It is
  not contamination: under leave-one-trigger-out the held-out trigger never enters the fit, and result 2 shows the
  prompt state carries that prior before any token is sampled, so the probe reads something true about the prompt
  rather than something true about the rollout — legitimate for a detector, wrong for the question "does the state
  know before the text gives it away". What the prior does not explain is a within-trigger residue of +0.02 to +0.04
  at most positions, always positive, which cannot be trigger identity (constant within a fold) and is most likely the
  query in context changing how the same first tokens are represented; at 20 folds it is not distinguishable from
  zero except at 32 tokens.
- **Are the debiased probes just token counting?** No (`code/probes/token_counting_tests.py`,
  `results/results_token_counting_tests.json`, layer 20, 2026-09-14; probe variants: raw, language + broken removed,
  all three nuisances removed). Three tests. *(3) Token-subspace removal:* ridge-regress the state onto the first-k
  bag vector inside each fold, project out the top-r directions of that map, refit. With r = 64–128 the bag is no
  longer linearly recoverable from the state (held-out R² 0.23–0.73 → 0.01–0.05), and the language-and-broken-removed
  probe loses **0.00–0.01 AUROC at 2 and 4 tokens** (0.73 → 0.72, 0.79 → 0.78) and 0.06 at 1 token (0.76 → 0.70).
  Two controls: r random directions of the full 4096-d space cost nothing (they mostly miss the ≤ 322-d data span), and
  r random directions drawn *inside the data span* — the fair control, since the token directions are fit from the data
  — cost **more** than the token directions: 0.73 → 0.60 at 2 tokens and 0.79 → 0.71 at 4 for r = 128. The directions
  along which token identity is readable are, if anything, less load-bearing for the probe than a random slice of the
  data's variance. *(2) Incremental validity:* adding that probe to the bag classifier
  raises within-trigger AUROC in 14 / 11 / 16 of 20 folds at 1 / 2 / 4 tokens, Wilcoxon p = 0.042 / 0.35 / 0.044.
  *(1) Identical-prefix pairs* (173 pairs at 1 token, 80 at 2, 13 at 4; scored leave-prefix-group-out, bag = 0.500
  exactly as it must): at 1 token the probe separates same-prefix rollouts at 0.64 (permutation p = 0.02) **but not
  after per-arm demeaning (0.48)**, so that separation is the trigger's prior; at 2 tokens 0.56, p = 0.25,
  underpowered; at 4 tokens unusable. So: what the debiased probe reads at 2–4 tokens is not linearly recoverable from
  token identity (test 3), and it adds to the bag within-trigger at 4 tokens (test 2); the strict same-prefix test can
  only confirm at 1 token, where what it finds is the prior. (The steering version of test 3 — does the *steering*
  direction survive the same token-subspace removal? — is result 5: yes, it keeps ~93 % of its effect.) ⚠ A first version of test 1 scored leave-one-out and came
  out below chance for every scorer including the bag — near-duplicate rollouts left in the fit bias the held-out
  score — which is why the group-out scoring and the bag-at-0.5 sanity check are in the script.
- **The prior-free version of the question.** The no-prompt state at token k is a function of the first k ids and
  nothing else, so a probe on it cannot read the prompt's prior, and its within-trigger lead over the bag of those same
  ids is rollout-reading by construction (`code/probes/noprompt_vs_bag_within.py`,
  `results/results_noprompt_vs_bag_within.json`; per-fold AUROC, paired Wilcoxon over 20 folds). At each slot's best
  no-prompt layer: **2 tokens +0.04, 12/20, p = 0.17; 4 tokens +0.07, 15/20, p = 0.027; 8 tokens +0.11, 16/20,
  p = 0.004**; 16 tokens +0.06, 11/20, p = 0.09. So "the model reads its own opening better than a token counter" is
  established from 4 tokens on and suggestive at 2, with no prompt in the picture. (The same-prefix test is degenerate
  for no-prompt states, identical prefixes giving identical states, which is why it can only be run with the prompt.)
  The residual qualification: trigger fragments are quoted in the text, so trigger identity is weakly readable from
  no-prompt states (0.10–0.16 balanced accuracy vs 0.05), which is why the per-fold form is the right one.
- From ~16 tokens the text alone catches up~~, and mean-pooled AUROC (0.96 persona / 0.99 broken) is a vocabulary
  result: the no-prompt mean-pooled state scores *below* bag-of-tokens (0.87 vs 0.94)~~. ⚠ Corrected 2026-09-12: the
  z-scored no-prompt mean-pooled state scores 0.97, *above* bag-of-tokens (0.94) and equal to the with-prompt state.
  Mean-pooling is still reported as the "reading the reply" ceiling rather than as a finding, but the ceiling is the
  model's reading of the text, not a token-count.
- Logistic adds ~0.05 over mass-mean at 1 and 8–64 tokens.

## Result 2 — the prompt-only state holds the trigger's odds, and only those

`P` is identical for all 24 seeds of a trigger (within-arm spread 0.008 vs 0.67 between), so it can only rank
triggers. Its AUROC (0.68 no nulls / 0.74 with) equals the per-arm base-rate ceiling (0.67 / 0.73), the shuffle
control equals the real score there, and layer 0 is exactly 0.50 (one shared embedding). Logistic on `P` under
leave-one-trigger-out is at chance.

A mass-mean direction fitted on 21 triggers' prompt states predicts the 22nd's assistant rate: **Spearman +0.67
(p = 0.001) over the 20 trigger arms** at layer 24, +0.69 at layer 34. It does **not** correlate with first-token
entropy (+0.13), word-salad rate (+0.08) or Latin-script share (+0.25), and entropy itself correlates only +0.27 with
assistant rate (phase 17's null restated). Twenty points; prospective test = optimise triggers against this direction
and blind-judge.

⚠ Z-scoring re-check (2026-09-12, `code/probes/zscore_recheck.py`): the code that produced +0.67 was run inline and is
not in the repo; a re-implementation (leave-one-arm-out mass-mean on `P`, tier-A labels, nulls excluded, each held-out
arm's mean projection against its assistant rate) gives **+0.71 raw / +0.73 z-scored at layer 24** and +0.64 / +0.68
at layer 34. The result holds either way. What z-scoring changes is the early layers: raw, layers 2–6 are at −0.23 to
+0.02; z-scored they are +0.34 to +0.58, so the prompt state ranks triggers from layer 6 on, not from layer 10.

## Result 3 — the two targets are separable, and both survive their restriction

| run | n | best mass-mean cell | LOAO AUROC | shuffle | z-scored best cell (2026-09-12) | z-scored LOAO |
|---|---|---|---|---|---|---|
| assistant vs persona, fluent rollouts only | 208 | L24 Rmean | 0.961 | 0.60 | L10 Rmean | 0.970 |
| broken vs fluent, unanimously not-assistant rollouts only | 138 (53 broken) | L10 Rmean | 0.986 | 0.51 | L10 Rmean | 0.989 |
| assistant vs *everything else* (tier B, incl. broken) | 480 | L24 Rmean | 0.885 | 0.56 | L16 Rmean | 0.910 |
| ≥ 2-of-4 broken vs rest (tier B) | 480 | L20 Rmean | 0.904 | 0.51 | L14 Rmean | 0.904 |

Random-split AUROC matches leave-one-trigger-out everywhere (e.g. 0.954 vs 0.962), so none of the probes is a
trigger detector. The two right-hand columns are the same runs with per-dimension z-scoring (added after the result 1
correction): every raw number reproduces exactly, z-scoring moves them by +0.00 to +0.025, and the random split still
matches (0.978 / 0.986 / 0.914 / 0.903). The best layer shifts earlier under z-scoring because the raw dot product
favours late layers where the largest dimension carries less of the class mean.

![layer heatmap](fig_layer_heatmap.png)

## Result 4 — trigger identity (for debiasing)

22-way arm classification, 5-fold over rollouts (chance 0.045): logistic reads the trigger from **one committed token
at 76 %** (layer 20), 51 % at 4 tokens, 38 % at 64, 86–92 % mean-pooled; ~~mass-mean only 0.25–0.44, so identity is
spread over many directions~~. Trigger-vs-null is ≥ 0.98 from one token on, even at layer 0. Directions (22 centred
class means + 22-way logistic weights, per cell, fitted on all 528) are in `trigger_directions.npz`;
persona/broken mass-mean directions per cell are in `persona_broken_directions.npz`.

⚠ Corrected 2026-09-12: the mass-mean figures were raw-residual nearest-centroid, and this is the result the
z-scoring artifact hits hardest after result 1. Z-scored (`results/results_zscored_recheck.json`), mass-mean reaches
**0.52 at one token, 0.48 at four, 0.36 at 64 and 0.70 mean-pooled** (raw: 0.36 / 0.37 / 0.30 / 0.44). Identity is
still spread — logistic is 0.76 / 0.51 / 0.38 / 0.92 on the same cells — but the mass-mean-vs-logistic gap that
"spread over many directions" rested on is about half what the raw numbers said. The saved centred class means in
`trigger_directions.npz` are raw-coordinate; the logistic weights there were always fit on z-scored features.

## Result 5 — steering with token identity removed keeps the effect (2026-09-14)

Full write-up: `RESULTS-tokdebias-steering.md`. The question: result 1's token-counting test 3 showed the *probe* survives
removing every direction from which the first 1/2/4 response token ids are linearly recoverable. Does the *steering*
direction? If "push toward the assistant" were mostly "push toward the assistant's first tokens", it should not.

**Directions** (`code/steering/build_token_debiased.py`), fit per slot at layers 12/16/20/24/28 on **15 trigger arms with
the 5 evaluation arms held out of the fit** (the 2026-09-09 candidates were fit on all 20): the raw assistant mass-mean;
the same with the top 64 or 128 token-identity directions projected out (held-out bag R² 0.42–0.90 → 0.00–0.21); a
rank-matched control removing 128 random in-span directions; a Gaussian direction. The token subspace holds only 19–32 %
of the raw vector's squared norm, so the debiased direction is still cos +0.89 to +0.94 to it.

**Run** (`code/steering/job_tokdebias_vllm.py`): vLLM 0.29.0 + vllm-lens 1.2.1 on a Colab A100, 47 arms × 5 evaluation
triggers × 48 seeds = 11,280 rollouts, T = 1.0, 96 tokens, response positions only, ε = ‖δ‖/‖residual‖ = 0.35. A layer
gate, a prompt rebuild check and a hook check all passed before any rollout. **Blind panel:** 1,140 rollouts, two coders
each, phase 17's rubric, κ +0.89 persona / +0.90 default assistant.

**Layer 20, the 2026-09-09 run's block (n = 60 per arm)**

| arm | persona | default assistant | English | on-topic |
|---|---|---|---|---|
| baseline (trigger, no steering) | 33.3 % | 26.7 % | 31.7 % | 21.7 % |
| raw + | **1.7 %** | 81.7 % | 41.7 % | 15.0 % |
| token-debiased r64 + | **1.7 %** | 78.3 % | 41.7 % | 6.7 % |
| token-debiased r128 + | **3.3 %** | 75.0 % | 38.3 % | 6.7 % |
| random in-span removal r128 + | 23.3 % | 55.0 % | 40.0 % | 48.3 % |
| random direction + | 31.7 % | 35.0 % | 36.7 % | 30.0 % |
| raw − | 43.3 % | 1.7 % | 25.0 % | 10.0 % |
| token-debiased r128 − | 50.0 % | 0.0 % | 23.3 % | 13.3 % |
| 2026-09-09 shipped direction + (anchor) | 8.3 % | 56.7 % | 38.3 % | 10.0 % |

**Persona by layer, each steered at its own fit block (n = 40 per arm)**

| | L12 | L16 | L20 | L24 | L28 |
|---|---|---|---|---|---|
| raw + | 10.0 % | 7.5 % | 2.5 % | 5.0 % | 12.5 % |
| token-debiased r128 + | 15.0 % | 0.0 % | 10.0 % | 12.5 % | 10.0 % |
| random + | 42.5 % | 25.0 % | 45.0 % | 37.5 % | 30.0 % |

- ⁂ **Pooled over the six conditions** (260 rollouts per direction): raw 6.2 %, token-debiased 8.1 %, random 35.0 %.
  Raw vs debiased Mantel–Haenszel OR 0.74, p = 0.49; debiased vs random p ≈ 2 × 10⁻¹³. The debiased direction is
  **+1.9 points** above raw, 95 % CI −2.3 to +6.2, so it **keeps 93 % of the raw direction's effect (CI 79–108 %)**.
  Raw vs debiased is not significant at any layer; debiased beats random at every layer.
- **Specificity is settled.** 2026-09-09's steering-vs-random contrast was p = 0.137 at n = 24; here it is p ≈ 10⁻¹³.
- **The anchor reproduces**: the shipped direction gives 8.3 % persona, the 2026-09-09 panel's exact figure, under a
  different sampler and panel.
- **Register, not language or topic.** English rises only 32 % → 38–42 %; on-topic falls. Steered replies are ordinary
  assistants talking about the odd message, not answering the question.
- **Reversal mostly breaks the text**: default assistant 27 % → 0–2 %, fully coherent 53 % → 17–22 %, persona only
  43–50 % (pooled p = 0.11).
- **Jacobian-lens reading** (`code/steering/jlens_directions.py`, pre-fitted `neuronpedia/jacobian-lens` for Qwen3-8B):
  raw and debiased directions promote *seems, interpretation, misunderstand, request, 似乎是* and suppress *scream,
  sneer, fuck*; an opener-token score is +0.58 raw vs +0.52 debiased vs −0.13 random at layer 20.

⚠ **Two things this run found about the 2026-09-09 steering.** (1) Stored feature layer L is `hidden_states[L]`, the
output of block L − 1 (vllm-lens gate, cos ≥ 0.9998 at index L − 1, next best ≤ 0.96), and that run hooked block L. At
layer 20 the two blocks steer alike here, so its conclusions stand. (2) The raw direction is not refit-stable: dropping
the five evaluation triggers moves the pooled early-response direction to cos +0.66 with the shipped one, because the
one-token class gap has ~4× the norm of the 2–8-token gaps and is nearly orthogonal to them.

## Also extracted, not analysed (by decision)

`phase19/p19_{A,B,C,D}.npy`: phase 19's 8,000 rollouts (phase 11 trigger ± `damn` prefill, clean ± prefill) teacher-
forced from text alone (no seeds exist; arm C re-tokenised past 96 tokens in 110/2000). Screen flags
(HATE/SLUR/HOSTILE), the 248 unreadable flags and the 4 hate verdicts are mapped to indices in `p19_labels.json`.
Not judged for persona, not probe-scored.

## Caveats

- 528 rollouts, 22 arms, one query, one model, one search. "Best layer" per column carries a selection bonus.
- Labels are LLM-judge labels; phase 17 §14 showed panels disagree on absolute rates. Tier A demands 4/4 agreement.
- The 1-token no-prompt slot is the attention-sink position and is unreliable for reasons unrelated to the question.
- **Mass-mean on raw Qwen residuals is not safe at 1–4 tokens** (correction to result 1, 2026-09-12): one dimension
  carries most of the class-mean difference and dominates the dot product. Every mass-mean number in this file has now
  been re-run z-scored (`code/probes/noprompt_zscore_check.py`, `code/probes/zscore_recheck.py`): result 1's no-prompt
  rows (up to +0.13) and result 4's mass-mean accuracies (up to +0.26 mean-pooled) change materially; results 2 and 3
  and the with-prompt rows move by ≤ 0.04 and no conclusion there changes. The saved raw-coordinate directions in
  `directions/` are still what steering needs (steering adds to the raw residual), but as *readers* they are dominated
  by that dimension at 1–4 tokens: the assistant direction at `R1` has 17–22 % of its squared norm on one dimension at
  layers 8–24. `fig_layer_heatmap.png` is the raw with-prompt mass-mean grid and was not regenerated.
- Result 5 is one query, five evaluation triggers and one ε. Its equivalence claim is an interval (−2.3 to +6.2 points),
  not a proof of zero difference, and its vLLM baseline (33.3 %) is not the 2026-09-09 HF baseline (37.5 %).
- Re-tokenising text without a seed round-trip is exact only for the 528 (checked by first id + length); phase 19's
  ids are unchecked.

## Layout

```
README.md              this file — the probe results
NARRATIVE.md           how the work actually went, including the wrong turns
PLAN.md                the probe plan (as approved, with deviations noted)
PLAN-steering.md       the steering plan (pre-registered)
RESULTS-steering.md    the steering results, with one retraction marked in place
RESULTS-tokdebias-steering.md  result 5 in full: token-identity-debiased vs raw steering, vLLM, 2026-09-14

code/extract/          Colab notebooks: probe_extract, noprompt_extract, steer_screen, probe_train_cpu
code/probes/           train_probes, train_trigger_probes, lexical_baseline, noprompt_eval,
                       noprompt_zscore_check + zscore_recheck + prompt_gap_tests + prompt_vs_noprompt_probes + lexical_lastk +
                       token_counting_tests + noprompt_vs_bag_within + fig_prefix_curve (the
                       2026-09-12 correction and its re-checks), within_arm_check, save_directions,
                       apply_probes_p19 (never run)
code/inlp/             inlp, inlp_fast (exact rank reduction), inlp_explore, inlp_fast_anyfeatures (inlp_fast with
                       --data/--features flags; produced the no-prompt INLP run, 2026-09-12)
code/steering/         build_candidates (the candidate-direction builder, recovered from the session log
                       2026-09-12; fits on all 20 arms, see RESULTS-steering.md correction) and the Colab
                       generation cells missing from steer_screen.ipynb; 2026-09-14: build_token_debiased (result 5's
                       directions), job_tokdebias_vllm (the vLLM + vllm-lens run), analyse_tokdebias (cheap columns,
                       blind corpus, judged tables), jlens_directions (the Jacobian-lens reading),
                       steer_token_debiased_cells (an HF-hooks version, superseded by the vLLM job, not run)
code/probe_pipeline_local.sh

data/                  features_qwen_wide.npz (528 x 19 layers x 10 slots x 4096, fp16),
                       features_meta, rollout_ids, labels, p19_labels
data/noprompt/         the same 528 responses fed with NO prompt (the control)
data/phase19/          phase 19's 8,000 rollouts teacher-forced (12 GB; not judged, not scored)

directions/            persona_broken_directions, trigger_directions, inlp_directions,
                       inlp_noprompt_directions (INLP on the no-prompt features, 2026-09-12),
                       steer_candidates (+meta), steer_token_debiased (+meta; result 5) — all fitted vectors
results/               every number behind every table, as JSON; results_noprompt_zscored,
                       results_zscored_recheck and results_inlp_noprompt are the 2026-09-12 additions;
                       steer_tokdebias_* (rollouts, gate, hook check, cheap and judged tables) and
                       jlens_directions are result 5
judging/               the blind panels: chunks given to coders, keys, verdicts (tokdebias_* is result 5's panel)
figures/               fig_prefix_curve.png, fig_layer_heatmap.png
logs/                  run logs
```

Everything here is mirrored to the public HF dataset
`mild-rgb/indy-mech-extension-qwen3-8b-persona-probes` (14.3 GB), where the paths are flatter:
`code/`, `results/`, `steering/`, `phase19/`, `noprompt/`.

GPU cost: ~1.1 A100-hours for the probe side (528 extraction + phase 19 + no-prompt control + one failed-download
detour); steering added ~0.5, so ~1.6 for the extension; result 5's vLLM run added ~1.1 (install, checks, 11,280
rollouts), ~2.7 in total.
