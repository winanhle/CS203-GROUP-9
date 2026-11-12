# Iteration 4 — gated EXIF with dropout (E4)

## Hypothesis and control

Applying 50% record-level metadata dropout to E2 may stabilize the gated model under missingness. E4 changes only the training dropout probability and preserves E2's parameters, data, seed, budget, and selection rule.

## Executed result

E4 test SRCC/PLCC/MAE is 0.83208/0.83337/9.3265. Relative to E2, delta SRCC is -0.00393 [-0.00837, 0.00041], delta PLCC -0.00443 [-0.00850, -0.00032], and delta MAE +0.2087 [0.0711, 0.3475]. E4 remains below E2 at every missingness level and reaches 0.82813 SRCC with no EXIF.

## Cost and interpretation

E4 trained in 10.18 seconds. Dropout harms PLCC and MAE under paired sampling and does not improve gated missingness robustness. This triggered the predeclared stop on adding complexity.

Evidence: `evidence/e4_frozen_r50_s20261109.json`, `evidence/missingness_all_s20261109.json`, `evidence/analysis_summary_s20261109.json`.
