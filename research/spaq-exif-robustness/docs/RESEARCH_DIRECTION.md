# Research direction — When and Why Does EXIF Help Smartphone IQA?

## Scope and evidence boundary

Phase 7 extends the verified `baseline-v0.1` tag without altering its E0–E4 evidence. Baseline v0.1 uses one frozen torchvision ResNet-50 ImageNet1K-V2 feature cache and lightweight MOS heads. It is a resource-adapted controlled study, not full backbone fine-tuning and not an exact reproduction of the original SPAQ training pipeline.

## What the original SPAQ work established

Fang et al., *Perceptual Quality Assessment of Smartphone Photography* (CVPR 2020), introduced SPAQ: 11,125 images from 66 smartphones with MOS, five perceptual attributes, nine scene categories, and EXIF. The paper evaluated a baseline and multi-task variants. Its MT-E model adds a scalar EXIF-dependent offset to a generic image prediction and optimizes equally weighted generic and combined L1 losses. Table 4 reports that MT-E improves over its BL control in the authors' protocol. Therefore, “EXIF can improve SPAQ prediction” and “using EXIF for smartphone IQA” are prior contributions, not our novelty.

The official `h4nwei/SPAQ` repository contains demo/preparation code and checkpoints, but no complete training entrypoint sufficient for exact modern reproduction. Baseline v0.1 follows the MT-E additive idea while explicitly documenting deviations.

## What baseline v0.1 has established

- Authentic SPAQ images, MOS, scene labels, source/APEX EXIF, and embedded provenance are joined 11,125/11,125 by exact filename.
- A 1,069-edge repeated-content graph is kept partition-safe in a frozen 7,786/1,111/2,228 split.
- E0 image-only, E1 additive EXIF, E2 gated EXIF, E3 additive dropout, and E4 gated dropout were run on one frozen ResNet-50 feature cache with seed 20261109.
- E1 exceeded E0 by 0.00565 test SRCC under paired image bootstrap, while E2 did not reliably exceed E1 and metadata dropout did not improve the tested missingness curves.
- These are single-training-seed results. Bootstrap intervals quantify test-image sampling, not optimization-seed variability.

## New Phase 7 questions

1. **Conditional utility:** Is the E1–E0 difference larger under high ISO, long exposure, or night scenes, consistently across training seeds?
2. **Metadata reliability:** Does destroying the correct image–EXIF pairing remove the benefit? Which verified EXIF groups retain signal when isolated?
3. **Fusion complexity:** Does gated fusion beat additive fusion consistently across seeds, or is the one-seed result effectively a tie?
4. **Representation dependence:** Does the conclusion change when a controlled image backbone is fine-tuned rather than frozen?

These questions target reliability and boundary conditions. They do not claim that EXIF improvement itself is novel.

## Relationship to modern IQA research

TOPIQ (Chen et al., IEEE TIP 2024) proposes top-down semantic-to-distortion interactions for modern IQA. It is an image-representation reference, not an EXIF reliability study. Published TOPIQ SPAQ values use different models/protocols and cannot be inserted into the matched E0–E2 ablation table. A SPAQ-trained TOPIQ checkpoint is also not an unseen SPAQ comparator.

Camera-captured IQA, content/distortion disentanglement, and recent smartphone defect-diagnosis work already study stronger visual representations and camera-specific artifacts. Phase 7's defensible contribution is narrower: a preregistered, same-split factorial diagnosis of correct-pair metadata value, capture-condition association, fusion complexity, missingness, and backbone adaptation.

## Novelty status

The surveyed primary sources verify that SPAQ already studied EXIF and that modern IQA work improves visual modeling. We have not established a systematic-review-level claim that no prior paper has ever permuted EXIF, grouped its fields, or studied metadata missingness for IQA. Novelty is therefore stated as **new within this project's verified controlled protocol and surveyed source set**, pending a broader literature review.

## Confirmatory versus exploratory boundary

The Phase 7 seeds, contrasts, subgroup definitions, permutation seeds, metrics, and acceptance rules were fixed in `configs/phase7_experiments.json` before new test results were generated. Multi-seed E0/E1/E2 and correct-pair permutation contrasts are confirmatory within this protocol. Feature-group diagnostics and the first fine-tuning pair are exploratory unless repeated and independently validated.

## Primary sources

- Fang et al., CVPR 2020: <https://openaccess.thecvf.com/content_CVPR_2020/html/Fang_Perceptual_Quality_Assessment_of_Smartphone_Photography_CVPR_2020_paper.html>
- Official SPAQ repository: <https://github.com/h4nwei/SPAQ>
- Chen et al., TOPIQ, IEEE TIP 2024: <https://doi.org/10.1109/TIP.2024.3378466>
- IQA-PyTorch implementation: <https://github.com/chaofengc/IQA-PyTorch>
