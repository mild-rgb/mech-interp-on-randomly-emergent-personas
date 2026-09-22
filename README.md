# Mech interp on randomly emergent personas

When a short adversarial token prefix (found by GCG) makes Qwen3-8B unsure of its first answer token,
the model often stops answering as an assistant and drops into a **persona**: a sea captain, a noir
detective, a fairy godmother. Nobody asked for the personas. The search was only told to make the
model uncertain. This programme asks where in the network that switch lives, whether it is a
direction we can read and steer, and whether we can produce it without any prefix at all.

It grew out of [CoT-spiking](https://github.com/mild-rgb/CoT-spiking), where phases 1 to 19 live.
Phase 11 there found the triggers and phase 17 produced the 528 Qwen3-8B rollouts that phase 1 here reads.

## The three phases at a glance

| Phase | Question | Short answer |
|---|---|---|
| [1. Probes and steering](#phase-1-probes-and-steering) | Is "assistant vs persona" a linear direction in the residual stream, and does pushing on it change the output? | Yes, and the steering is causal and signed. A one-sentence system prompt still beats the best vector. |
| [2. Entropy from layer 0](#phase-2-entropy-from-layer-0) | Can a single direction added after block 0 reproduce the entropy spike, with nothing in the context? | Yes, on Qwen3-8B and four other backbones. Random directions do nothing. |
| [3. A persona direction](#phase-3-a-persona-direction) | Can the same rig target the persona itself instead of entropy, and keep the text coherent? | Yes. 56% persona on unseen prompts, 100% coherent, on topic, in English. |

Each phase folder has its own `README.md` with every number, a `PLAN.md` written before the run with
predictions registered in advance, and the code and results JSON behind every table.

## Phase 1: probes and steering

Folder: `phase1-indy_mech_extension/`. Run 2026-09-09, steering follow-up 2026-09-14. Moved here from
`CoT-spiking/indy_mech_extension` on 2026-09-21.

Linear probes on Qwen3-8B's residual stream, 19 layers by 10 positions, trained leave-one-trigger-out
so no probe is tested on a trigger it saw. Two targets: assistant vs persona, and broken vs fluent text.

- **A prompt-state direction predicts persona rate.** Read at the prompt, it predicts how often a
  held-out trigger will produce a persona (Spearman +0.67).
- **Steering that direction is causal and signed.** Judged persona rate goes from 37.5% to 8.3% pushing
  one way and to 58.3% pushing the other.
- **Debiasing does not cost anything.** An INLP-debiased direction, with trigger identity, language and
  brokenness projected out, steers as well as the raw one.
- **A one-sentence English system prompt still beats the best vector.** The vector closes about a third
  of the gap. The sentence closes nearly all of it.

Read `README.md` for the probe results, `NARRATIVE.md` for how the work went including one retraction,
and the `RESULTS-*.md` files for each steering experiment. Activations (14.3 GB) are on Hugging Face:
[`mild-rgb/indy-mech-extension-qwen3-8b-persona-probes`](https://huggingface.co/datasets/mild-rgb/indy-mech-extension-qwen3-8b-persona-probes).

## Phase 2: entropy from layer 0

Folder: `phase2-layer0-entropy/`. Run 2026-09-21.

Same objective as phase 11 (raise the entropy of the first answer token), but the intervention is a
direction added to the residual stream after block 0. Nothing is added to the context.

- **One 4096-vector does it.** Fit on 24 ordinary prompts, it flattens 16 unseen prompts to 13.4 bits on
  Qwen3-8B. Phase 15 needed a 20-token prefix to reach 13.5.
- **It transfers across backbones when refit.** 63% to 99% of the entropy ceiling on Llama-3.1-8B,
  Mistral-7B, gemma-2-9b, Qwen2.5-7B and OLMo-2-7B.
- **Random directions of the same size are inert.**
- **The flat state looks like phase 15's fork-then-commit** on four backbones: Italian, Chinese, a sea
  captain, a dying narrator, *"Alright, fam"*. On Llama and Mistral it is sustained noise at every
  strength tried.
- **Zeroing block 0's MLP reaches the same entropy on three models with no fitting**, but the text is
  degenerate. On Qwen3-8B the attention sink can be destroyed without moving the entropy.

Predictions registered in `PLAN.md`: 5 of 6 held. The fitted directions are small (4 MB) and live in git.

## Phase 3: a persona direction

Folder: `phase3-persona-direction/`. Run 2026-09-21.

Same rig as phase 2, but the target is the persona rather than entropy. A layer-0 direction is fit on
24 prompts to push the layer-20 first-answer state off phase 1's **INLP-debiased** assistant axis, with
an 8-bit cap on entropy so the text stays readable.

- **On 16 unseen prompts: 56% persona by both blind coders, 100% coherent, 0% word salad, 95% on
  topic, English**, at 6 bits of entropy.
- **It beats the alternatives.** GCG on the same backbone gets 32.7% persona with a third of the outputs
  degraded. Phase 2's entropy direction gets 7.5%.
- **The debiasing is what makes it a speaker direction.** Fit against the raw assistant axis instead,
  the same recipe gives the model's thinking register and only 2.5% persona.
- **Forty distinct voices in 80 rollouts:** fairy godmother, sci-fi captain, gothic storyteller, Kratos,
  talking raccoon.

Predictions registered in `PLAN.md`: 4 of 5 held. A 10,560-rollout scale-up with Gemma judging,
probes and PCA is described in the phase README. Its activations and generations (7.8 GB) are on
Hugging Face:
[`mild-rgb/phase3-layer0-persona-direction-qwen3-8b`](https://huggingface.co/datasets/mild-rgb/phase3-layer0-persona-direction-qwen3-8b).

## Replicating

Everything needed to rerun every number is either in this repo or on Hugging Face.

| What | Where | Put it in |
|---|---|---|
| Code, prompts, rubrics, judge labels, fitted directions, every results JSON | this repo | already here |
| Phase 1 activations and directions (14.3 GB) | [`mild-rgb/indy-mech-extension-qwen3-8b-persona-probes`](https://huggingface.co/datasets/mild-rgb/indy-mech-extension-qwen3-8b-persona-probes) | `phase1-indy_mech_extension/data/` and `directions/` |
| Phase 2 fitted directions (4 MB) | this repo | `phase2-layer0-entropy/results/phase2_directions_qwen3-8b.npz` |
| Phase 3 scale-up activations and generations (7.8 GB) | [`mild-rgb/phase3-layer0-persona-direction-qwen3-8b`](https://huggingface.co/datasets/mild-rgb/phase3-layer0-persona-direction-qwen3-8b) | `phase3-persona-direction/results/brrrt/hf_dl/` |
| The 528 phase 17 rollouts phase 1 reads | [CoT-spiking](https://github.com/mild-rgb/CoT-spiking) | its own repo |

Large `.npy` and `.npz` tensors are git-ignored (see `.gitignore`). The datasets keep the same file
paths, so a `snapshot_download` into the folder in the third column puts every file where the code
expects it.

## Content note

Phase 1's dataset republishes 8,000 unfiltered generations from CoT-spiking phase 19, produced under
an adversarial prefix. A small number are hate speech. They are published so that rate can be
re-scored. See the data notice at the top of `phase1-indy_mech_extension/README.md` before downloading
that folder. The rest of the material is ordinary assistant and persona text from benign prompts.
