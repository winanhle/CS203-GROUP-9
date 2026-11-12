# Iteration 2 — additive EXIF with dropout (E3)

## Hypothesis and control

Training E1 with 50% record-level metadata dropout may improve robustness when EXIF records are missing. E3 changes only this train-time dropout probability; test masking uses independent fixed seeds and the same held-out image IDs.

## Executed result

E3 test SRCC/PLCC/MAE is 0.83450/0.83452/9.2534 versus E1 at 0.83613/0.83572/9.1942. Every E3-E1 paired 95% interval crosses zero. From full to zero metadata, E1 SRCC changes 0.83613→0.83579 and E3 changes 0.83450→0.83432; E3 does not exceed E1 at any tested availability.

## Cost and interpretation

E3 trained in 6.82 seconds. The anticipated robustness benefit was not observed because the additive metadata contribution itself is extremely small in this fitted model. Dropout is retained as a negative result, not promoted as an improvement.

Evidence: `evidence/e3_frozen_r50_s20261109.json`, `evidence/missingness_all_s20261109.json`, `evidence/analysis_summary_s20261109.json`.
