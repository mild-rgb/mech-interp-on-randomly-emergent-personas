# Mech interp on randomly emergent personas

Mechanistic-interpretability programme on the personas that show up as a side effect when a
GCG-found token prefix drives the model's first-answer-token entropy up. Nobody asked for the personas;
the search was only told to make the model uncertain, and the model dropped into them on its own.
Grew out of [`../CoT-spiking`](../CoT-spiking) (phases 1–19 live there; phase 11 found the triggers,
phase 17 supplies the 528 Qwen3-8B rollouts this programme reads).

| Folder | What it is |
|---|---|
| `phase1-indy_mech_extension/` | **Run 2026-09-09; steering follow-up 2026-09-14.** Formerly `CoT-spiking/indy_mech_extension`, moved here 2026-09-21. Linear probes on Qwen3-8B's residual stream (19 layers × 10 positions, leave-one-trigger-out) for assistant-vs-persona and broken-vs-fluent; a prompt-state direction that predicts a held-out trigger's persona rate (Spearman +0.67); signed, causal steering of that direction (judged persona 37.5% → 8.3% forward, → 58.3% reversed); INLP-debiased direction steers as well as the raw one. A one-sentence system prompt still beats the best vector. See its `README.md`, `NARRATIVE.md` and `RESULTS-*.md`. Activations (14.3 GB) on HF: [`mild-rgb/indy-mech-extension-qwen3-8b-persona-probes`](https://huggingface.co/datasets/mild-rgb/indy-mech-extension-qwen3-8b-persona-probes). |
| `phase2-layer0-entropy/` | **Run 2026-09-21.** Phase 11's first-token-entropy objective, but the intervention is a direction added to the residual stream after block 0, nothing in the context. One 4096-vector fit on 24 ordinary prompts flattens 16 unseen prompts to **13.4 bits** on Qwen3-8B (phase 15's 13.5 needed a 20-token prefix); refit per backbone it reaches 63–99% of ceiling on Llama-3.1-8B, Mistral-7B, gemma-2-9b, Qwen2.5-7B, OLMo-2-7B. Random directions of the same size are inert. The flat state is phase 15's fork-then-commit on four backbones (Italian, Chinese, a sea captain, a dying narrator, *"Alright, fam"*), and sustained noise on Llama and Mistral at every eps tried. Zeroing block 0's MLP reaches the entropy on three models with no fitting, but its text is degenerate. On Qwen3-8B the attention sink can be destroyed without moving `H1`. Predictions registered in `PLAN.md`; 5 of 6 held. |
| `phase3-persona-direction/` | **Run 2026-09-21.** Same rig, persona objective instead of entropy: a layer-0 direction fit on 24 prompts to push the layer-20 first-answer state off phase 1's **INLP-debiased** assistant axis, with an 8-bit entropy cap. On 16 unseen prompts: **56% persona by both blind coders, 100% coherent, 0% salad, 95% on topic, English**, at 6 bits. GCG on the same backbone: 32.7% with a third degraded; phase 2's entropy direction: 7.5%. Fit against the *raw* assistant axis instead, the same recipe gives the thinking register and 2.5% persona: the debiasing is what makes it a speaker direction. Forty distinct voices in 80 rollouts (fairy godmother, sci-fi captain, gothic storyteller, Kratos, talking raccoon). Predictions in `PLAN.md`; 4 of 5 held. The 10,560-rollout follow-up (activations, generations, Gemma judging inputs; 7.8 GB) is on HF: [`mild-rgb/phase3-layer0-persona-direction-qwen3-8b`](https://huggingface.co/datasets/mild-rgb/phase3-layer0-persona-direction-qwen3-8b). |

## Replicating

Everything needed to rerun every number is either in this repo or on Hugging Face:

| what | where |
|---|---|
| code, prompts, rubrics, judge labels, fitted directions, every results JSON | this repo |
| phase 1 activations and directions (14.3 GB) | [`mild-rgb/indy-mech-extension-qwen3-8b-persona-probes`](https://huggingface.co/datasets/mild-rgb/indy-mech-extension-qwen3-8b-persona-probes) → `phase1-indy_mech_extension/data/` and `directions/` |
| phase 2 fitted directions (4 MB) | in git: `phase2-layer0-entropy/results/phase2_directions_qwen3-8b.npz` |
| phase 3 brrrt activations and generations (7.8 GB) | [`mild-rgb/phase3-layer0-persona-direction-qwen3-8b`](https://huggingface.co/datasets/mild-rgb/phase3-layer0-persona-direction-qwen3-8b) → `phase3-persona-direction/results/brrrt/hf_dl/` |
| the 528 phase 17 rollouts phase 1 reads | [`CoT-spiking`](https://github.com/mild-rgb/CoT-spiking) |

Large `.npy` / `.npz` tensors are git-ignored (see `.gitignore`); the datasets above hold the same paths, so a
`snapshot_download` into the folders named in the third column puts every file where the code expects it.
