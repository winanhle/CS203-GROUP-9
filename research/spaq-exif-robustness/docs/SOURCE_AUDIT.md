# SOURCE AUDIT

**Audit started:** 2026-11-08T19:46:21Z  
**Status:** COMPLETE FOR LOCKED STUDY SCOPE

This document records live verification of primary sources. Claims copied from the handoff remain `DOCUMENT_VERIFIED` only after inspection of the primary paper/repository; archive availability, code execution, and dataset contents require separate evidence.

## SPAQ

- Paper: Fang et al., *Perceptual Quality Assessment of Smartphone Photography*, CVPR 2020, DOI `10.1109/CVPR42600.2020.00373`.
- Official repository: <https://github.com/h4nwei/SPAQ>.
- Official revision: `03e2b1c752b593079afffc2adb6ffff4c76e376e` (retrieved 2026-11-08 UTC).
- Paper method/Table 4 were cross-checked. The repository has demo assets but no complete official training pipeline; executed work is resource-adapted.
- Recovery, hashes, issue #38 recommendation, and license cautions are in `docs/DATASET_RECOVERY.md` and `data/provenance.json`.

## TOPIQ / IQA-PyTorch

- Paper: Chen et al., *TOPIQ: A Top-Down Approach From Semantics to Distortions for Image Quality Assessment*, IEEE TIP 2024, DOI `10.1109/TIP.2024.3378466`.
- Official repository: <https://github.com/chaofengc/IQA-PyTorch>.
- Official revision: `d47b7ac72ed25388371d0b637076c125cdb95a3c` (retrieved 2026-11-08 UTC); MIT code license inspected.
- Paper SPAQ table/model documentation were cross-checked. A pretrained SPAQ checkpoint was excluded as an unseen comparator because protocol/overlap did not support a clean matched claim.
- The author-recommended data mirror is pinned at revision `f97c4f5dbdbdea7039bc9b80db808313347b1bef`.

## Related sources

KonIQ-10k, Hu et al., DFT, and PrISM-IQA were reviewed for scope but excluded because they do not provide a defensible matched SPAQ contrast. Q-Align is used only as a pinned compact image source after content/join audits. Secondary quartile aggregators are not publisher evidence.

## Evidence classification

- Literature values are `PAPER_REPORTED`, never project results.
- Repository/paper/license statements are `DOCUMENT_VERIFIED`.
- Downloads, hashes, joins, executions, and metrics are `EXECUTION_VERIFIED` only where an artifact exists under `evidence/`.

