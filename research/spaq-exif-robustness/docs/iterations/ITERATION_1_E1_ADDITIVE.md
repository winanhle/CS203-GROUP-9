# Iteration 1 — additive EXIF (E1)

## Hypothesis and control

Adding a paper-inspired scalar EXIF offset to E0 may improve held-out MOS ranking. E1 uses the identical frozen ResNet-50 cache, split, seed, image head, optimizer budget, validation selector, and test policy as E0. The only model change is a 16-parameter bias-free linear offset over eight robust-scaled APEX fields and eight availability indicators. The generic image prediction remains independently usable.

## Executed result

E0 test SRCC/PLCC/MAE is 0.83048/0.83063/9.3489; E1 is 0.83613/0.83572/9.1942. The paired E1-E0 differences are +0.00565 SRCC [0.00132, 0.01013], +0.00510 PLCC [0.00082, 0.00947], and -0.1547 MAE [-0.2916, -0.0151], using 10,000 paired test-image resamples.

## Cost and interpretation

E1 trained in 8.25 seconds after the shared 119.76-second cache extraction. This one-seed, frozen-feature study detects a small paired-sampling advantage, but does not quantify retraining variability or reproduce the original SPAQ end-to-end architecture.

Evidence: `evidence/e0_frozen_r50_s20261109.json`, `evidence/e1_frozen_r50_s20261109.json`, `evidence/analysis_summary_s20261109.json`.
