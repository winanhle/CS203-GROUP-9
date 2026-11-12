# Iteration 3 — gated EXIF (E2)

## Hypothesis and control

A zero-initialized, metadata-conditioned feature gate may outperform a scalar additive offset while preserving an identity path at initialization. E2 changes only the fusion mechanism relative to E1. It adds 8,704 metadata parameters and uses the same 2,048-dimensional cache, 256-unit image head, split, seed, budget, dual objective, and selection rule.

## Executed result

E2 test SRCC/PLCC/MAE is 0.83602/0.83779/9.1179. Against E1, paired differences are -0.00012 SRCC [-0.00516, 0.00484], +0.00207 PLCC [-0.00282, 0.00696], and -0.0764 MAE [-0.2276, 0.0752]. All intervals include zero. Validation SRCC was also below E1.

## Cost and interpretation

E2 trained in 7.40 seconds. The extra gate does not show a reliable advantage over the simpler 16-parameter offset. At 100% test metadata missingness, E2 SRCC falls to 0.83290, more than E1's decline.

Evidence: `evidence/e2_frozen_r50_s20261109.json`, `evidence/missingness_all_s20261109.json`, `evidence/analysis_summary_s20261109.json`.
