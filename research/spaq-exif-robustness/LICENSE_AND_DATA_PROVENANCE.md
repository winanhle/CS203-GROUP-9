# License and data provenance

## SPAQ primary source

Fang et al., *Perceptual Quality Assessment of Smartphone Photography*, CVPR 2020. The official repository is pinned at `03e2b1c752b593079afffc2adb6ffff4c76e376e`. Demo headers grant educational/research use and modification without fee subject to attribution and prohibit commercial-product use without permission. The repository has no top-level license file, and its dataset statement does not separately define image redistribution terms. Raw assets therefore remain untracked and will not be republished.

## Q-Align image archive

`q-future/q-align-datasets` labels its repository MIT, but this is not treated as relicensing the SPAQ photographs. The downloaded object is pinned in `docs/DATASET_RECOVERY.md`. It is a re-encoded JPEG variant used only in ignored local storage.

## IQA-PyTorch sources

IQA-PyTorch is pinned at `d47b7ac72ed25388371d0b637076c125cdb95a3c` and uses PolyForm Noncommercial 1.0.0 plus an S-Lab license for applicable components. Dataset repositories display CC-BY-SA-4.0, but original SPAQ ownership still controls underlying third-party assets. The flattened metadata repository is pinned at `9a0cb6a3f628256f67cc1f761f25b996065b34ad`.

## Distribution policy

Git tracks code, schemas, hashes, aggregate audits, configs, and small evidence. It does not track raw SPAQ images, raw workbooks, credentials, or third-party checkpoints.

The author-recommended archive was verified but not fully extracted. Its 35,476,869,087 uncompressed bytes exceed Kaggle's documented 20 GB saved working allowance, so remote execution will use the compact verified image variant joined to the canonical manifest. This is a declared preprocessing difference from the original full-resolution archive.
