# How this went

*A working narrative of 2026-09-09, written so the next person does not repeat the wrong turns.
The numbers are in `README.md` and `RESULTS-steering.md`; this file is the story around them,
including five mistakes that were caught and one claim that had to be retracted mid-session.
Two corrections from 2026-09-12 and a steering follow-up from 2026-09-14 (act 4,
`RESULTS-tokdebias-steering.md`) are added near the end.*

---

## The question phase 17 left lying around

Phase 17 wedged sixteen tokens of junk in front of `what shall i do today` and Qwen3-8B stopped being
an assistant. It became a folk singer, a Swift source file, a Korean elder, a love poet writing to a
truck emoji. Twenty-four blind coders confirmed it, and the controls were clean: a clean prompt gave
0% personas, and an *unoptimised* random prefix of the same length also gave 0%. Something about a
searched trigger, not junk in general, installs a voice.

Phase 17 read the outputs. It never looked inside the model. The obvious next question is whether the
thing the judges can see in the text is visible in the residual stream first — and if so, how early.

Two things made that question cheap to ask. Phase 17's `hbar-rerun` had already judged *every* Qwen
rollout with a second full-sweep panel, so each of the 528 rollouts carried four independent blind
verdicts from two panels that shared no members. And the rollouts were stored with their seeds, so
their exact token sequences could be rebuilt without a new search. Nothing in this extension was
generated until the steering work in act 3.

## The plan, before any compute

`PLAN.md` was written first and is left as it was, with the deviations listed at its top. The parts
that mattered:

**Two targets, two strictness tiers each.** Target 1 is *assistant vs persona*; target 2 is *broken
text vs fluent*. Tier A demands all four coders agree (226 assistant / 144 persona; 59 broken / 312
fluent) and drops everything mixed. Tier B is looser and keeps all 528. Tier A exists because phase 17
§14 showed two panels can disagree by 25 points on absolute rates; unanimity across two independent
panels is the closest thing to a label that does not depend on which panel you asked.

**The two targets are entangled, and the plan said so.** 53 of the 59 unanimously broken rollouts are
also unanimously not-assistant. An "assistant" probe could score well by detecting word salad. So two
restricted reruns were registered: the assistant probe on unanimously *fluent* rollouts only, and the
broken probe on unanimously *not-assistant* rollouts only. A direction that survives both is reading
two different things.

**The split is the whole game.** Per-arm persona rates are wildly uneven — one trigger gives 16
assistant and 3 persona, another 2 and 16 — so a probe can cheat by recognising *which trigger* is in
the prompt. The primary split is therefore **leave-one-trigger-out**: 22 folds, each holding out every
rollout of one arm. Three controls sit beside it: a random 5-fold split (the gap between the two
measures how much was trigger recognition), a per-arm base-rate predictor (the ceiling for anything
that lives in the prompt-only slot), and a within-arm label shuffle for a chance band.

**The cost quote was ≤ 0.5 A100-hours.** The probe side came in at ~1.1, of which the extraction
this plan describes was a rounding error; the rest was the no-prompt control, phase 19 and one failed
download. Steering added ~0.5 (`RESULTS-steering.md`), so the extension totals ~1.6, the figure on the
HF card.

What deviated: LDA was dropped (too slow at 4096 dimensions); logistic regression ran at five layers
instead of nineteen; the two controls in the next section were added after the user's review; the
trigger-identity probes were added for the debiasing work; and phase 19's features were extracted but,
by decision, never judged or scored.

## Act 1 — the probes, and a headline that did not deserve to be one

**Rebuilding the sequences.** The rig is `hbar-rerun/hbar_personas.ipynb` cell 1, byte for byte: same
chat template with thinking off, same offset-based scaffold split, same generation function. Each
stored rollout has its text, first token id, length and seed. Re-tokenising the text and checking
first id and length against the record accepted 462 of 528; the other 66 were regenerated from their
stored seeds on the same class of GPU, and all 66 came back identical. The plan said to assert zero
failures before extracting anything, and there were zero. The rig reproduced phase 17's clean
first-token entropy to four decimals (0.2366 against 0.237).

**Extraction.** One teacher-forced forward pass per rollout, prompt ids plus response ids, with hidden
states on. Causal attention means the state at position *t* is exactly what the model had when it
sampled token *t+1*, so this reproduces the generation-time internals with no sampling loop. Saved at
every second layer (0 to 36, nineteen layers) and ten positions: `P`, the last prompt token, which
is the state that produced response token 1 and is identical for all 24 seeds of an arm; `R1` to `R64`,
the state after *k* response tokens have been committed; `Rmean`, the average over the whole
response; and `Rlast`. Thirty-three seconds of A100 time. 0.8 GB in fp16. Pulling it to a local
machine took longer than producing it, and one download detour that went nowhere is the reason the
GPU bill is 1.1 hours rather than 1.0; the md5 of the copy that worked is in `logs/hf_download.log`.

**What the probes were trained to predict.** Each probe takes one 4096-dimensional vector — the
residual stream at one layer and one of the ten slots — and predicts one of two binary labels that
the blind coders had already assigned to that rollout's *text*:

- **assistant vs persona**: did the model answer as its default assistant, or as somebody else?
- **broken vs fluent**: did the text degenerate into word salad, or stay coherent?

So a probe at slot `R4`, layer 24 is asking: from the model's internal state after only four response
tokens, can you tell what the judges will say about the whole 96-token reply? Two probe types per
cell. Mass-mean: the direction from the persona class mean to the assistant class mean, with a
threshold fitted on the training fold — one direction, no fitting beyond two averages. Logistic
regression: L2-regularised, class-balanced, run at five layers only because it is slow at 4096
dimensions. Every cell is trained on 21 triggers' rollouts and scored on the 22nd's, and the score is
AUROC, which is indifferent to class imbalance. The probes never see the text, the trigger string or
the seed; they see one hidden-state vector and a label.

**The first number out was 0.97 AUROC**, mean-pooled over the response at layer 24,
leave-one-trigger-out.

⚠ **The user's reply was the most useful thing said all day**: *"mean pooled isn't particularly
interesting. the assistant just has a certain vocabulary."* That is exactly right, and it reframed
the project. Averaging the state over 96 tokens is reading the reply after the fact; a contextual bag
of words would do it. The interesting question is what the model knows *before the text gives it away*.

Two controls followed, and they are the spine of everything after:

1. **A bag-of-tokens baseline** — a logistic probe on which token ids appear in the first *k* response
   tokens, same leave-one-trigger-out split. This is what "the assistant has a certain vocabulary"
   looks like as a number.
2. **A no-prompt control** — the identical response tokens fed to the model with *nothing* in front of
   them. No template, no trigger, no query. This separates "the model expects a persona" from "these
   tokens read as a persona". It cost a second extraction pass and lives in `data/noprompt/`.

**A miscount, caught by a printed n.** The first run of the "broken, restricted to not-assistant"
probe reported a fine AUROC over 7 broken rollouts instead of 53. The restriction had been applied
with the wrong tier's labels. Nothing about the AUROC looked wrong; the *n* printed beside it did.
Rerun (`logs/fix.log`), it is 53 broken against 85 fluent, best cell layer 10 mean-pooled, AUROC 0.986,
shuffle 0.51. The habit of printing sample sizes next to every score is the only reason this was a
five-minute fix rather than a published number.

**What survived.** In the **2-to-16-token window** the prompt-conditioned state beats the text baseline
by 0.07–0.12 AUROC on the persona target and 0.07–0.09 on the broken target, ~~and the no-prompt state
does *not* — at 2–4 tokens it is at or below bag-of-tokens. By 64 tokens all three converge, and the
mean-pooled number is revealed as the vocabulary result it always was: the no-prompt mean-pooled
state scores *below* bag-of-tokens (0.87 against 0.94). So the early signal is real, it is modest, and
it exists **only because the trigger is in context**.~~ Logistic regression adds about 0.05 over the
mass-mean direction at most positions, which is worth having but changes nothing about the shape.

⚠ **Corrected 2026-09-12: the struck sentences were an instrument failure, and the conclusion they
carried is withdrawn.** The no-prompt control was scored with mass-mean on raw residuals. Without a
prompt in front, one dimension of the no-prompt state holds 73–95 % of the squared class-mean
difference at 2–4 tokens in layers 8–32 — a massive-activation dimension that the prompt-conditioned
states also have, at 33–61 % — and it swamps the dot product. Z-scored per dimension, the no-prompt
state reads persona at 0.77 / 0.82 at 2 / 4 tokens (raw: 0.64 / 0.69) against bag-of-tokens 0.72 /
0.73, and broken at 0.80 / 0.87 (raw: 0.66 / 0.78) against 0.76 / 0.81. Mean-pooled it is 0.97, above
bag-of-tokens, not below. The with-prompt cells move by at most 0.04 under the same treatment. So the
early signal is real and modest, and **it does not need the trigger in context**: the model reading
its own first two to four tokens carries nearly all of it, and the prompt adds 0.02 for persona and
0.02–0.04 for broken on top. How it was found: the INLP script z-scores by default, and pointing it at
the no-prompt features (`results/results_inlp_noprompt.json`) produced raw no-prompt probes at 0.74–0.82
in the layers the control file had at chance. The check is `code/probes/noprompt_zscore_check.py`,
results in `results/results_noprompt_zscored.json`. `README.md` result 1 is corrected in place.

**Could the lead just be trigger identity leaking through? (added 2026-09-11.)** It could have.
Under leave-one-trigger-out the probe cannot memorise the held-out trigger, but the hidden state
still has that trigger in context, and result 2 below shows the prompt state carries a direction
that ranks *unseen* triggers by their persona rate. Bag-of-tokens never sees the prompt. Pooled
AUROC rewards between-arm ranking, so the pooled comparison hands the hidden state an advantage
that has nothing to do with reading the rollout in progress. The check
(`code/probes/within_arm_check.py`, results in `results/results_within_arm.json`) subtracts each
arm's mean score from every classifier before computing AUROC, which removes every between-arm
pair and leaves only within-trigger discrimination, and then compares the hidden state and
bag-of-tokens fold by fold with a paired Wilcoxon test. Mass-mean at each slot's best layer,
unanimous labels, no nulls, assistant vs persona:

| tokens seen | hidden, pooled | hidden, within-arm | bag, within-arm | no-prompt, within-arm | paired hidden − bag, mean | folds won | Wilcoxon p |
|---|---|---|---|---|---|---|---|
| 1 | 0.767 | 0.658 | 0.646 | 0.631 | +0.04 | 12/20 | 0.31 |
| 2 | 0.786 | 0.714 | 0.669 | 0.610 | +0.03 | 12/20 | 0.23 |
| 4 | 0.844 | 0.769 | 0.686 | 0.633 | +0.08 | 15/20 | **0.014** |
| 8 | 0.828 | 0.797 | 0.705 | 0.775 | +0.12 | 15/20 | **0.002** |
| 16 | 0.877 | 0.812 | 0.745 | 0.781 | +0.07 | 12/20 | 0.053 |
| 32 | 0.920 | 0.866 | 0.803 | 0.822 | +0.06 | 16/20 | **0.002** |
| 64 | 0.927 | 0.870 | 0.843 | 0.845 | +0.02 | 10/20 | 0.23 |

⚠ The `no-prompt, within-arm` column above is the raw-residual probe and is superseded (see the
correction after "What survived"). Z-scored, same split, same arm-demeaning, at each slot's own
best layer:

| tokens seen | hidden, within-arm, z-scored | no-prompt, within-arm, z-scored | bag, within-arm | hidden − no-prompt |
|---|---|---|---|---|
| 1 | 0.733 | 0.674 | 0.646 | +0.06 |
| 2 | 0.728 | 0.708 | 0.669 | +0.02 |
| 4 | 0.769 | 0.733 | 0.686 | +0.04 |
| 8 | 0.814 | 0.791 | 0.705 | +0.02 |
| 16 | 0.831 | 0.790 | 0.745 | +0.04 |
| 32 | 0.865 | 0.829 | 0.803 | +0.04 |
| 64 | 0.895 | 0.863 | 0.843 | +0.03 |

The no-prompt state now beats bag-of-tokens within-arm at every position, and the prompt's own
contribution is a consistent 0.02–0.06. Paired-tested the same day with layer 20 fixed for both
conditions (`code/probes/prompt_gap_tests.py`): at 2 tokens the gap is +0.03, 12 of 20 folds,
Wilcoxon p = 0.33; at 4 tokens +0.03, 10 of 20, p = 0.07 (bootstrap over rollouts p = 0.02); at
32 tokens +0.03, 14 of 20, p = 0.046. So the prompt's contribution is real in sign at every position
and reaches significance only late, where it is least interesting. The 1-token gap (+0.15, p = 0.004)
is the attention-sink slot, not a prompt effect. For broken-vs-fluent no position passes both tests.
What the prompt adds, when it adds anything, is the trigger's prior: the component of the with-prompt
probe orthogonal to the no-prompt probe lies 10–20× above chance in the trigger-identity subspace and
aligns with result 2's prompt-state direction, and the per-trigger gap follows the trigger's assistant
rate (`code/probes/prompt_vs_noprompt_probes.py`). Between-arm information, in other words — the
thing this check was built to remove.

So is the lead over the no-prompt probe just the trigger's prior? Mostly. The fold test *is* the
within-trigger test — a per-fold AUROC cannot rank across triggers — and the bootstrap on pooled
AUROC is the one that lets the prior in. At layer 20 the pooled gap is significant from 4 tokens on
and the within-trigger gap only at 32; the difference between those two rows is the prior. It is not
contamination in the leakage sense: the held-out trigger never enters the fit, and result 2 shows the
model holds that prior before a token is sampled. The probe reads something true about the prompt,
which a real detector may use and which this question must not. What is left after the prior is a
within-trigger residue of two to four hundredths, always positive, that cannot be trigger identity
and is probably the query in context changing how the first tokens are represented. Twenty folds
cannot confirm it. That residue is the whole of what "the state knows before the text gives it away"
now rests on, and it is stated here at its real size.

**Are the debiased probes just token counting? (2026-09-14.)** Three tests at layer 20
(`code/probes/token_counting_tests.py`). Removing every direction from which the first-k token ids
are linearly recoverable — held-out R² for the bag falls from 0.23–0.73 to 0.01–0.05 — costs the
language-and-broken-removed probe 0.00–0.01 AUROC at 2 and 4 tokens, while removing the same number of
random directions from inside the data's own span costs 0.08–0.13. (Random directions of the full
4096-d space cost nothing and are not a fair control: the data live in at most 322 dimensions and
random 4096-d directions mostly miss them. The first draft of this paragraph used that control.) Adding the probe to the bag classifier raises within-trigger AUROC at 1 and 4
tokens (p = 0.04) but not at 2. And the strict test, pairs of rollouts with identical first tokens
scored by one model fit outside the group, separates them at 1 token only until arm means are
removed, at which point it is at chance: what it was reading was the trigger's prior. At 2 tokens
that test has 80 pairs and cannot see an effect under about 0.68. So the probe is not token counting
at 2–4 tokens; whether what it reads there is the rollout or the prior, the same-prefix test cannot
say. One more instrument note: the first version of the same-prefix test scored leave-one-out and
put every scorer, including the bag, below chance. Near-duplicate rollouts left in the fit tilt it
against the held-out point. The bag's exact 0.500 under group-out scoring is now the sanity check.

**The answer-only probes are clear of the prompt's prior by construction** — a point made in review
on 2026-09-14 that reframes the last two paragraphs. Without a prompt, the state at token k is a
function of the first k ids and nothing else. So the no-prompt probe's within-trigger lead over the
bag of those same ids is rollout-reading with no prior available, and that comparison, not the
same-prefix test, is the one that answers "does the state read the reply better than a token
counter". Paired over the 20 folds (`code/probes/noprompt_vs_bag_within.py`): +0.04 at 2 tokens
(p = 0.17), **+0.07 at 4 (p = 0.027), +0.11 at 8 (p = 0.004)**, +0.06 at 16 (p = 0.09). That is the
surviving result of act 1, stated in its cleanest form: the model's own representation of its first
four to eight tokens carries the register, above what the token identities carry, with or without the
trigger in front. What the prompt adds on top is the trigger's prior and a residue too small to see.

Three things this settles and one it narrows. Between-arm information inflates *both* classifiers
by about the same 0.04–0.07 — bag-of-tokens carries some because the model quotes trigger fragments
back — so the lead is not a trigger-odds artifact. ~~The no-prompt state sits below the with-prompt
state within-arm at every early position, so the lead is not contextual reading of the text alone
either.~~ ⚠ Corrected 2026-09-12: z-scored, the no-prompt state sits 0.02–0.06 below the with-prompt
state within-arm, not 0.10–0.14, so the lead over bag-of-tokens *is* mostly contextual reading of the
text alone — the model's reading, which a bag of ids cannot reproduce — with a small residue for the
prompt. And the broken-vs-fluent target behaves the same, with a within-arm lead of 0.07–0.10 at
2–8 tokens (paired p = 0.008 at 4, 0.026 at 8) that closes to nothing by 16 and reverses by 64,
where a single position's state loses to a bag of 64 tokens. ⚠ **What it narrows: the lead is
established at 4–8 committed tokens, not at 2.** At one and two tokens the hidden state wins 12 of
20 folds with p ≈ 0.2–0.3; the direction is right but the per-fold SDs (0.09–0.14) swallow it.
`README.md`'s "2–8 tokens" should be read as "4–8, suggestive at 2". Had the lead vanished under
demeaning, "the state knows before the text gives it away" would have collapsed into result 2 — a
fact about the prompt, not about the rollout. It did not vanish; it just starts a couple of tokens
later than the pooled table implies.

Both restricted reruns held. Assistant-vs-persona on fluent rollouts only: 0.961. Broken-vs-fluent on
not-assistant rollouts only: 0.986. And the random split matched leave-one-trigger-out everywhere
(0.954 against 0.962, for instance), so none of these probes is a trigger detector wearing a
persona-detector's label.

⁂ **A side finding worth more than it got.** The prompt-only slot — the state before a single token is
sampled — is identical across all 24 seeds of a trigger, so it can only encode that trigger's *odds*.
It sits exactly at the per-trigger base rate (AUROC 0.68 against a base-rate ceiling of 0.67), layer 0
is exactly 0.50 because every arm shares one embedding, and a logistic probe on it under
leave-one-trigger-out is at chance, as it must be. But a mass-mean direction fitted on 21 triggers'
prompt states predicts the 22nd's persona rate at **Spearman +0.67** (p = 0.001, twenty points), and
it does *not* correlate with first-token entropy (+0.13), word-salad rate (+0.08) or Latin-script share
(+0.25) — and first-token entropy is the quantity eighteen phases of this programme have been
optimising. Phase 17 measured entropy-vs-persona at −0.39 and called it a non-relationship. Something
in the prompt state knows what entropy doesn't. Nobody has yet run a search against it.

**The bridge to act 2.** A 22-way logistic probe reads *which trigger* produced a rollout from one
committed token at **76%** (chance 4.5%), fading to 38% by 64 tokens. Mass-mean does much worse
(0.25–0.44), so identity is spread over many directions rather than one. Those directions were saved
to `trigger_directions.npz` under the label "for debiasing", which is where the next hour went.

## Act 2 — INLP, and an attempt at a "pure" assistant direction

If trigger identity is that readable, then "is this the assistant" and "which trigger is this" are
badly entangled, and the natural move is iterative nullspace projection: fit a probe for the nuisance
concept, project its direction out of every state, refit, repeat until the nuisance is unreadable, and
only then fit the direction you care about.

Measured at layer 24, one committed token (corrected 2026-09-12 against `results_inlp.json`; the
earlier text mixed cells): the assistant direction has **40% of its norm inside the trigger subspace
and 29% inside the language subspace**. But those subspaces are large — 152 and 40 directions — so
the random expectations are 32% and 9%: the trigger overlap is barely above chance, the language
overlap about three times it. At two tokens the language overlap is only 5%. The broken-text overlap
(3% at one token, 24% at two, against 1–5% expected) is the one that is clearly non-random. Removing
the nuisances costs AUROC. Per-trigger demeaning — subtracting each arm's own mean — takes the
mass-mean probe from 0.79 to 0.72 and leaves a direction still 0.95-aligned with the original. The
full nullspace projection takes the logistic probe to 0.66 and rotates the direction by 60 degrees. The exact-rank version (`inlp_fast.py`) made this quick
enough to explore other concepts with the same machinery, essentially for free, and that turned up two
things nobody was looking for:

- **Seed parity is readable at 0.90 AUROC after one token**, on held-out triggers. The seed is a
  generator seed, not a property of the text. `torch.manual_seed(s)` produces the same uniform draw
  for every trigger, so a given seed always samples at the same quantile of whatever distribution it
  faces, and the state encodes that quantile. It decays to chance by 64 tokens. This is very likely
  the mechanism behind phase 14's "seed-locked" personas — and it means same-seed rollouts across
  different triggers are **not independent samples**, which the write-ups have been assuming.
- **The assistant direction is substantially a street-persona direction**: from 2 tokens on, 24–66% of
  it lies in the subspace separating street/casual voices from other personas. Qwen's largest persona
  family is street/casual, so "assistant vs persona" is partly "assistant vs slang".

## Act 3 — from reading to steering

A probe direction that predicts is not the same as a direction that controls. `PLAN-steering.md` was
written before any steering compute and is the pre-registration; the parts that decided the day:

**Nine candidate directions per layer**, at seven layers, all unit norm. The plain probe direction
(`massmean_early`, averaged over response positions 1–8); the same after per-trigger demeaning; the
INLP-debiased one from act 2; an English-assistant-vs-everything direction; an
English-vs-non-English-assistant direction as a pure language separator; a street-balanced variant;
the mean-pooled vocabulary direction as a foil; **clean-prompt state minus trigger-prompt state**,
meaning "there is no junk in my prompt"; and two Gaussian random directions as the calibration every
arm is read against.

**The cosines said three things before a single token was steered.** The English-assistant direction
and the language separator agree at cos +0.99, so "English assistant" is largely a language direction
and the judges' `language` and `persona` fields would have to be reported separately. The
street-balanced direction is identical to the plain one (cos +1.00), so act 2's street finding is about
the *subspace*, not the primary axis, and balancing is not the fix. And the INLP-debiased direction is
nearly orthogonal to all of them with a 4–6× smaller class gap: the honest one and the weak one.

**The protocol.** Directions were *meant* to be refit on 15 triggers with every number measured on the
5 held-out ones — see the 2026-09-12 correction below: the refit never happened, and the five arms are
held out of the evaluation only. Phase
1: a cheap teacher-forced screen, adding the direction through a forward hook and measuring
`margin = NLL(persona continuations) − NLL(clean-assistant continuations)` under a held-out trigger
prompt, printed **beside** two control columns — assistant text under a clean prompt (collateral
damage) and neutral prose (general fluency). Phase 2: free generation on the survivors, 24 seeds × 5
triggers at phase 17's exact sampling settings, with every arm generated rather than assumed: a
reversal at −α, a random direction at matched norm, the same steering on a clean prompt, and a
**ceiling arm** — a plain-English system message — because phase 17's open list had said that if one
sentence of English beats a vector, that is the headline. Measurement: **never the probe that chose
the direction**, since scoring steered text with it is circular. A blind judge panel with phase 17's
rubric verbatim was written in as the deciding instrument, with a shared anchor set so it could be put
on phase 17's scale rather than merely hoped to match it — the fix §14 prescribed and never ran. The
dual-use point was stated in the plan: −α is a persona-installation tool, and it was a registered arm.

### The units bug

⚠ **The first screen's random control was vacuous and I reported it as if it weren't.** The plan
measured α in units of the assistant-vs-persona class gap along each direction, so that one unit meant
the same thing at layer 8 (residual norm 49) as at layer 32 (norm 683). Sensible for the probe
directions. But for a random direction that gap is ~0, so the random arm received an effectively zero
perturbation, dutifully did nothing, and looked like a clean specificity result. It wasn't a control at
all. The same bug silently disabled the clean-minus-trigger direction, whose class gap is also near
zero (−0.1 at layer 8 in `steer_candidates_meta.json`), and I read its flat screen as "inert".

What caught it was that random scored 0.000 to three decimals at almost every α (the widest excursion
in `steer_screen_coarse.json` is −0.004). A real null wobbles.

Re-run with `epsilon = ||delta|| / ||residual||`, so every direction gets the same size push, the
honest picture is much less flattering: **the best real direction beats the random band by only ~1.9×
in size**. What actually separates them is *consistency* — real directions raise the margin at +ε and
lower it at −ε at 4 of 4 layers, monotonically, with slopes 0.29–0.41; random directions manage 2 of 4,
with per-layer slopes from −0.14 to +0.22 that cancel to ~0.005. About 2.5 standard deviations of the
random band. Real, and modest.

The control columns also caught the obvious trap: at α = +4 class gaps the screening metric looks
superb (+1.7 to +2.2) while collateral loss on clean prompts rises **+5.2**. That is a broken model,
not a steered one, and without the control column printed *beside* it in the same table it would have
been the headline. Phase 17's rule about never reporting a treatment column without its null, applied.

### Generation, and a surprise

Twelve arms, five held-out triggers, 24 seeds each, batched so a config took one minute instead of
nine. The cheap columns first — function-word rate (phase 17's script-independent coherence measure,
0.174 under a trigger, 0.423 clean), Latin-script share, repetition. Clean monotone dose-response over
ε = 0.20, 0.35, 0.50; a clean reversal at −0.35 (function-word rate fell to 0.122, 21% *below*
baseline); random inert; mild collateral damage on a clean prompt (0.423 → 0.392, still 100% Latin).

⁂ **And the best direction was not the assistant direction.** `clean_minus_trigger` — *subtract the
trigger's signature* rather than *add the assistant* — closed 34% of the gap where the assistant
direction closed 9%, consistently across all five held-out triggers. It is the direction the units bug
had told me was inert an hour earlier.

And the same cheap columns said the *debiased* directions did nothing: the trigger-demeaned direction
at 1% of the gap, the INLP-debiased one at −4%, indistinguishable from the random control at −3%. I
wrote that up as a finding — "the component that survives debiasing is not the component that moves
behaviour" — and it stood in `RESULTS-steering.md` for the rest of the afternoon.

## The judges, and a retraction

**Panel 1.** Six blind coders, phase 17's rubric verbatim, 144 rollouts from the six decisive arms,
pooled, shuffled, arm labels stripped, two coders each. Kappa **+0.944** on persona, +0.944 on default
assistant, +0.982 on on-topic — above phase 17's own panels. They overturned the lexical reading twice.

**First**: the function-word rate said the assistant direction closed "7% of the gap". The judges said
that same arm cuts **persona 37.5% → 8.3%** (Fisher p = 0.036) and raises default-assistant replies
**33.3% → 75.0%** (p = 0.008). The proxy had badly understated a large effect, because a function-word
count cannot distinguish a fluent persona from an assistant. The reversal arm went the other way,
persona 58.3%, and **+0.35 against −0.35 is 2/24 vs 14/24, p = 0.0005** — the strongest single number
in the extension. The two good directions also turned out to do different jobs: both cut persona to
8.3%, but clean-minus-trigger restores *English* (25% → 66.7%) where the assistant direction only
reaches 37.5%. Removing the trigger's signature brings back the language; adding the assistant
direction brings back the register.

What panel 1 did *not* establish: steering against **random at the same ε** is 2/24 vs 7/24,
p = 0.137. Significant against baseline, not against random, and with 24 rollouts per arm both
statements are true at once. A powered specificity test needs about 100 per arm.

**Panel 2, and the retraction.** The two debiased directions had not been in panel 1's corpus, so a
second panel of three coders read them — with **24 anchor rollouts drawn from panel 1's corpus**, so
the two panels could be put on one scale. The anchor check passed: baseline 37.5% vs 36.4%, treatment
8.3% vs 7.7%, one point apart against phase 17's 25-point swing. Then the new arms: the
trigger-demeaned direction cuts persona to **12.5%** and the INLP-debiased direction to **8.3%** —
exactly matching the raw entangled direction. The claim already reported was wrong, and
`RESULTS-steering.md` now carries the retraction in place rather than a quiet edit.

⁂ **So the answer to the question that started act 2 is yes**: a direction with trigger identity,
language and brokenness all projected out — four to six times weaker in class-gap terms — steers the
register as well as the entangled one. A pure assistant direction exists and it controls behaviour.
The function-word proxy had failed twice in one run, in opposite directions: understating the raw
direction and erasing the debiased ones.

## The deflating result

The ceiling arm — "You are a helpful assistant. Always reply in English." — read 73% of the gap closed
on the first pass, which was already better than any vector. It was also wrong. Building the system
prompt through `apply_chat_template` without `enable_thinking=False` had silently **re-enabled Qwen's
thinking mode**, and the outputs were full of English `<think>` traces that the function-word rate
counted as assistant prose. No metric flagged it; reading two sample generations did. Re-run properly,
with zero thinking tags in 120 rollouts:

**Function-word rate 0.425, 99.5% Latin. A clean prompt is 0.423 and 100%.**

One sentence of English closes essentially the entire gap. The best steering vector closes a third.
Phase 17's own open list predicted exactly this shape of result and asked for the ceiling arm to be
run for exactly this reason. If the goal is pinning the register rather than understanding it, the
vector is not the tool.

## What phase 19's data is doing here

Phase 19's 8,000 rollouts — phase 11's trigger with and without a `damn` prefill, and the clean
prompt with and without — were teacher-forced through the same extraction while the GPU was warm.
They have no seeds, so they were re-tokenised from text alone and are unchecked in the way the 528
are; arm C re-tokenised past 96 tokens in 110 of 2,000. The phase 19 screen flags (HATE, SLUR,
HOSTILE), the 248 unreadable flags and the four hate verdicts are mapped to indices in
`p19_labels.json`. That is 12 GB in `data/phase19/`, and **by decision it was neither judged for
persona nor scored by any probe.** The features exist for whoever wants them; `apply_probes_p19.py`
was written and never run. The HF dataset card carries the content warning that corpus requires.

## What the day actually established

1. The persona/assistant distinction is linearly readable 2–16 tokens into a response, ~~**only with the
   trigger in context**, and the mean-pooled version of that claim is a vocabulary artifact.~~
   ⚠ corrected 2026-09-12: **with or without the trigger in context.** The no-prompt state, z-scored,
   reads it at 0.77–0.82 from 2 tokens, 0.02–0.04 below the with-prompt state; the "only with the
   trigger" claim was a raw-residual mass-mean artifact (see the second correction below).
2. The prompt-only state carries a trigger's *odds* and nothing about the seed — and a direction in it
   predicts an unseen trigger's persona rate, independent of first-token entropy.
3. Steering that direction works and is signed: persona 37.5% → 8.3% forward, → 58.3% reversed
   (p = 0.0005). **It is a better attack than defence**, which is stated here because it is true, not
   because it is comfortable. ⚠ 2026-09-14: not reproduced at n = 60. Forward steering took persona
   33% → 2–3%; reversal took it to 43–50% (pooled p = 0.11), and mostly produced broken text rather
   than a voice (default assistant → 0–2%, fully coherent 53% → 17–22%). The axis is signed; which
   direction is "stronger" depends on whether you count personas or wreckage.
4. **A debiased direction steers as well as an entangled one.** Readability and control are not the
   same property, but here they coincided once the right instrument was used. 2026-09-14: now powered,
   and extended to token identity — removing the token-identity subspace keeps 93% of the effect
   (CI 79–108%) across five layers (act 4).
5. **A sentence of English beats every vector tested.** If the goal is pinning the register rather than
   understanding it, the vector is not the tool.

## Mistakes made, and what caught each

| mistake | what caught it |
|---|---|
| α in class-gap units made the random control vacuous, and hid the direction that turned out best | noticing random scored ~0.000 at every α |
| the "not-assistant" restriction miscounted, leaving 7 broken rollouts instead of 53 | the n printed beside the AUROC |
| ceiling arm silently re-enabled thinking mode | reading two sample generations instead of trusting the metric |
| "debiasing destroys causal power" — reported, then retracted | the pre-registered blind panel, with anchors |
| the function-word proxy misread two arms in opposite directions | the same panel |
| **(found 2026-09-12)** the steering directions were never refit on 15 triggers; the plan, the results file and this narrative all said they were | reading the candidate-building script instead of the sentence describing it |
| **(found 2026-09-12)** the no-prompt control read at chance at 2–4 tokens because raw-residual mass-mean was swamped by one massive-activation dimension; result 1's "only with the trigger in context" rested on it | running a second script (INLP) that happened to z-score, on the same features, and getting 0.8 where the first had 0.5 |
| **(found 2026-09-14)** the 2026-09-09 steering fit directions at stored feature layer 20 (output of block 19) and hooked block 20 | vllm-lens's layer gate, which compares captured states to the stored ones and refuses to steer below cos 0.99 |
| **(found 2026-09-14)** `RESULTS-steering.md` said dropping the 5 evaluation arms "moves little"; it moves the raw direction to cos +0.66 | actually doing the refit, and printing its cosine to the shipped vector before steering with it |

Phase 17 broke five instruments and wrote the rule that a proxy is never reported without the thing it
proxies for. This run broke a sixth in the same way. **The only reason it was caught is that the judge
panel was written into the plan as the deciding measure before any compute ran**, rather than added
afterwards when the cheap numbers looked good. Three of the five catches above were not statistics at
all: a sample size, a suspiciously round zero, and two rollouts somebody bothered to read.

## Correction, 2026-09-12

An audit of the written claims against the code found one that was false and several that were loose.

**The steering directions were fit on all 20 triggers, not 15.** `PLAN-steering.md` registered a
refit on 15 arms with 5 held out. The candidate-building script (run inline on 2026-09-09 at 16:01,
now saved as `code/steering/build_candidates.py`) takes every labelled non-null rollout; it never
references the held-out list. `inlp.py` fits the trigger, language and broken nuisance bases on 15
arms but the assistant direction on all 20, and the INLP-debiased candidate averages that direction.
The steering notebook's cell 3 is headed "fit on 15 triggers, evaluate on 5 held-out" and loads the
prebuilt file. So every direction, including the comparison set for the random controls, saw the five
evaluation triggers' original rollouts — about 85 of the ~370 in each class mean. Every *generated*
rollout was new, and the probe results in `README.md` are unaffected because their leave-one-out loop
refits inside each fold. The claim is corrected in `RESULTS-steering.md` and above; the rebuild with
the five arms masked has not been run.

**Smaller corrections made at the same time.** The INLP overlap sentence in act 2 quoted 41% language
and a 3% random expectation; the file says 29% and 9% at one token, and the trigger overlap (40%)
sits at its 32% random expectation. The random control in the coarse screen was within ±0.004, not
exactly 0.000. GPU time is ~1.6 A100-hours in total, not 1.1; the 1.1 excludes steering. The label
shuffle ran 10 permutations, not the 20 the plan specified. `results_inlp_explore_firsttok.json`
contains no first-token concept at all — the seven-way label tripped the script's minimum-class guard
in every cell — so "is the assistant direction a first-token direction" was never tested. Three
flavour numbers in `RESULTS-steering.md` phase 1 ("slopes 0.29–0.41", "~1.9×", "2 of 4 and 1 of 4")
could not be reproduced from `steer_screen_eps.json` under any obvious definition (a least-squares
slope over the five ε values gives 0.31–0.52 for the real direction and 2.3× against the random
band); they are left in place and flagged here rather than replaced with a different guess.

## Second correction, 2026-09-12 — the no-prompt control

**Result 1's headline was an instrument artifact.** `noprompt_eval.py` and `within_arm_check.py` fit
mass-mean directions on raw fp16 residuals. That is fine with the prompt in front, where the largest
single dimension carries 33–61 % of the squared class-mean difference at 2–4 tokens. It is not fine
without it: in the no-prompt states one dimension carries 73–95 % at layers 8–32, and a raw dot
product is then a one-dimensional probe on a dimension that has nothing to do with persona. Those
cells read 0.45–0.58 and were written up as "the no-prompt control removes the lead". With
per-dimension z-scoring — no labels, the same standardisation `inlp.py` has always applied — the same
states read 0.77 / 0.82 (persona, 2 / 4 tokens) and 0.80 / 0.87 (broken), against 0.79 / 0.84 and
0.84 / 0.89 with the prompt. Logistic regression on z-scored no-prompt features agrees (0.75–0.80).
Layers 6 and below, and 8 tokens and beyond, were never affected, which is why the no-prompt row
looked plausible: it rose smoothly from 0.69 to 0.90 and only the 2–4-token cells were wrong.

What changes: result 1's title and both no-prompt rows (`README.md`, corrected in place, originals
kept); the "only with the trigger in context" sentence in act 1 and in "What the day established";
the "mean-pooled no-prompt scores below bag-of-tokens" sentence (z-scored it is 0.97, above); and the
`no-prompt, within-arm` column of the within-arm table. What does not change: the with-prompt rows
(≤ 0.04 under z-scoring), the bag-of-tokens rows, result 2, the trigger-identity result, the
restricted reruns, and everything in the steering half. What it does to the story: the early window
is the model reading its own first tokens, which a bag of ids cannot do; the prompt's own
contribution is 0.02–0.06, not significant at 2 tokens, borderline at 4, and significant only from
32 tokens on (`results/results_prompt_gap_tests_L20.json`). The gap between "the state knows before
the text gives it away" and "the state reads the text better than a bag of tokens" is the whole of
what was overclaimed.

How it was found: running `inlp_fast.py` on the no-prompt features (`results/results_inlp_noprompt.json`,
`directions/inlp_noprompt_directions.npz`) to debias the no-prompt direction. Its raw column came out
at 0.74–0.82 where the control file said chance, and the only difference between the two scripts was
the scaler. Rerun of both conditions, raw and z-scored, every layer 2–34, pooled and within-arm:
`code/probes/noprompt_zscore_check.py` → `results/results_noprompt_zscored.json`.

**Everything else re-run the same way, the same day** (`code/probes/zscore_recheck.py` →
`results/results_zscored_recheck.json`). Every raw number reproduced to the third decimal first. Then:
result 3's four restricted runs move by +0.00 to +0.025 (0.961 → 0.970, 0.986 → 0.989, 0.885 → 0.910,
0.904 → 0.904) and the random split still matches, so the separability claims hold. Result 2's
Spearman holds (+0.71 raw / +0.73 z-scored at layer 24 in a re-implementation; the inline code that
gave +0.67 is not in the repo) and gains the early layers, where raw was at zero and z-scored is
+0.34 to +0.58. Result 4 is the one that moves: 22-way trigger mass-mean goes from 0.25–0.44 to
0.29–0.70, so "identity is spread over many directions" was half a raw-dot-product artifact and half
true. And the steering vectors: the raw assistant mass-mean at layer 20 has 18 % of its squared norm
on that one dimension, the INLP-debiased direction 1 %, random 0 % — noted in `RESULTS-steering.md`.
`HF_CARD.md` is corrected. `fig_prefix_curve.png` is regenerated with the z-scored no-prompt row;
`fig_layer_heatmap.png` (with-prompt only, ≤ 0.04 movement) was left as it was.

Everything else checked reproduces: both panels' rates, kappas and Fisher tests; the 24-rollout
anchor set; the ceiling arm (0 think tags in 120); the twelve generation arms' cfg, function-word
and Latin columns; the response-only steering mask and ε·‖residual‖ scaling; the label counts; the
462/66 rebuild; every README table; the trigger-identity accuracies; the leave-one-arm-out Spearman
+0.67 / +0.69 and its null correlations; the within-arm table; the +5.2 collateral cell.

## Act 4, 2026-09-14 — steering with token identity removed

**Why.** The token-counting tests in act 1 had just shown that the assistant *probe* survives projecting out every
direction from which the first one, two or four response token ids can be read. The obvious worry about the *steering*
result was the same one in causal form: maybe "add the assistant direction" works because it says "start with the
assistant's usual tokens", and the register follows from the opening. If so, a steering vector with the token-identity
subspace removed should steer less.

**Building it, and the first surprise.** `code/steering/build_token_debiased.py` fits the directions the way the
2026-09-09 plan said they would be fit and weren't: on 15 trigger arms, with the 5 evaluation arms out of the fit. Per
slot (1, 2, 4 tokens) it ridge-maps the state onto the bag of first-k ids, removes the top 64 or 128 directions of that
map, and takes the assistant mass-mean in what is left; a rank-matched control removes 128 random in-span directions
instead. Before any steering I printed each new vector's cosine to the shipped one, and the *raw* refit came out at
**+0.66**. `RESULTS-steering.md` had said a mass-mean over ~370 rollouts "moves little when 85 are dropped". It moves a
lot. The reason was sitting in the per-slot numbers: the one-token class gap is about four times the norm of the two-
to eight-token gaps and nearly orthogonal to them (cos +0.16 to +0.22), so the old pooled direction was mostly the
first-token state — the most trigger-specific thing in the whole stream. The new directions unit-average the slots
instead. The token subspace, for the record, holds only a fifth to a third of the raw vector's squared norm, so the
debiased direction is cos +0.89 to +0.94 to the raw one; the random removal takes 89–97% and leaves cos +0.26 to +0.37.

**vLLM, and the second surprise.** A sibling project (`cot_bert_analysis/00_foundation/VLLM_HOOKS.md`) had already
worked out how to steer under vLLM with vllm-lens: run the job as a standalone script, use a per-request hook to steer
only response positions, and **gate the layer index against stored states before steering anything**. That gate is
what found the off-by-one. Stored feature "layer 20" is `hidden_states[20]`, the output of block 19; the index that
reproduces it at cos 0.9998 is L − 1 at every layer from 10 to 30, with the next-best index at 0.86–0.96. The
2026-09-09 notebook fitted at feature layer 20 and hooked `model.model.layers[20]`, one block late. Nobody had noticed
because adjacent blocks are similar enough for a steering vector to still work — which this run then confirmed
directly, by steering layer 20 at both blocks (raw 1.7% vs 2.5% persona). The prompt rebuild matched byte for byte, the
clean first-token entropy came out at 0.250 bits against phase 17's 0.237, and the hook moved prompt positions by
exactly zero and response positions by the intended vector.

The design widened on request to layers 12, 16, 20, 24 and 28: 47 arms × 5 triggers × 48 seeds, 11,280 rollouts. The
unsteered arm ran at 3,161 tokens a second. Every steered arm ran at ~278, with the GPU at 9% and the Python process at
110% — per-request Python hooks, not compute. That made it an hour, not ten minutes; still inside the budget, so it ran.

**The panel.** 1,140 rollouts, shuffled across 24 arms, two blind coders each, phase 17's rubric, 19 coders in all;
κ +0.89 persona, +0.90 default assistant. The answer is clean:

| pooled over six steering conditions, 260 rollouts per direction | persona |
|---|---|
| no steering (layer-20 panel) | 33.3% |
| random direction | 35.0% |
| raw direction | 6.2% |
| token-debiased direction (128 removed) | 8.1% |

⁂ **Removing token identity keeps 93% of the steering effect (95% CI 79–108%).** Raw and debiased differ by 1.9
points (CI −2.3 to +6.2), never significantly at any layer, and the debiased direction beats random at every layer.
"Push toward the assistant" is not "push toward the assistant's first tokens".

Four other things fell out. **Specificity is settled**: 2026-09-09's steering-vs-random contrast (p = 0.137 at n = 24)
is now p ≈ 10⁻¹³. **The anchor reproduced exactly**: the shipped direction gave 8.3% persona, the old panel's figure,
under a different sampler, a different panel and 60 rollouts instead of 24 — an exact match, where the 2026-09-09
anchors had agreed to within a point. **It is a register direction, not a language or topic direction**: English only rises
32% → 38–42%, and on-topic *falls*, because a steered reply is an ordinary assistant politely asking what the strange
message meant. And **the reversal mostly breaks the text** rather than installing a voice (see the note on item 3 above).
The oddest arm was the random in-span removal: it keeps only cos +0.37 with the raw direction, yet recovers about half
its default-assistant gain, and it is the one steered arm whose on-topic rate *rises* (48%). Not followed up.

The cheap proxy, for once, ordered the arms correctly (raw ≈ debiased > random removal > random > reversed). It still
said "+0.05 function-word rate" for what the judges called a thirty-point drop in personas.

**Coda — what the directions say.** With the GPU free, each direction was read through the pre-fitted Jacobian lens
for Qwen3-8B (`code/steering/jlens_directions.py`): no forward pass, just the lens matrix at the steered block, the final
norm weight and the unembedding. Raw and debiased directions push the model toward *seems, interpretation,
misunderstand, request, conversations*, Chinese *似乎是* ("seems to be"), and away from *scream, sneer, fuck*. The old
INLP-debiased direction is the purest version of it: *似乎, 看起来, seems, seem, 好像* up; *sigh, staring, eyes,
roaring* and roleplay asterisks down. That is "it seems you're trying to…" against a narrator's stage directions. The
lens reads two to three times more of this at layers 12–20 than the logit lens does, so the directions are disposed
toward those words long before they point at them. An opener-token score survives debiasing (+0.58 raw, +0.52 debiased,
−0.13 random at layer 20), which describes the vector and agrees with the panel, but is not a second behavioural test.

Total for act 4: about 1.1 A100-hours, then the runtime was deleted.

## Left open

- ~~**Specificity is underpowered.** Treatment vs random at matched ε is 2/24 vs 7/24, p = 0.137.
  Significant against baseline, not against random. ~100 rollouts per arm would settle it.~~
  Resolved 2026-09-14 (act 4): 6–8% vs 35% persona over 260 rollouts per direction, p ≈ 10⁻¹³.
- ~~**The debiased-vs-entangled equivalence rests on 24 rollouts per arm** and one layer each.~~
  Resolved for token identity 2026-09-14 (act 4): 40–60 per arm at five layers, 93% of the effect kept.
  The INLP-debiased direction itself was not re-run.
- **Opened 2026-09-14:** why a raw direction with 94–97% of its norm randomly removed still recovers half the
  default-assistant gain and raises on-topic replies; and whether the result holds on other queries and at other ε.
- **Opened 2026-09-14:** `RESULTS-steering.md` does not mention the one-block hook offset. (Its false "moves little
  when 85 are dropped" sentence was struck and corrected in place the same day.)
- **The prompt-state direction that predicts trigger quality has never been optimised against.**
  That is the experiment this whole line suggests: run the search against that direction instead of
  entropy, generate, and judge blind. It is cheap, it is differentiable, and it would tell you whether
  the direction is causal or merely descriptive.
- **The seed-parity finding has a consequence nobody has followed up**: every cross-trigger
  comparison in phases 14–17 that pairs rollouts by seed is comparing correlated samples.
- **Phase 19's 8,000 teacher-forced rollouts sit in `data/phase19/` unjudged and unscored**, by
  decision. The features and the screen labels are there if anyone wants them.

## What shipped

Everything small is in git (`3ed04d5`): the plans, the results JSON, both judging panels' chunks,
keys and verdicts, the fitted directions in fp16, and the two figures. The large tensors — the 528
features, the no-prompt control, the phase 19 arrays — are on Hugging Face as
`mild-rgb/indy-mech-extension-qwen3-8b-persona-probes` (14.3 GB), with a card that leads with the
phase 19 content warning. Total GPU: about 1.6 A100-hours (1.1 probes and extraction, 0.5 steering),
against a plan that quoted half an hour for the probes and an hour and a half for the steering. Act 4 added ~1.1
(2026-09-14), for ~2.7 in all; its rollouts, panel and directions are in `results/steer_tokdebias_*`,
`judging/tokdebias_*` and `directions/steer_token_debiased*`, not yet committed or mirrored to Hugging Face.
