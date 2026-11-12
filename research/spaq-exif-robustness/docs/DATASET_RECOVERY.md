# SPAQ dataset recovery

## Author-confirmed source change

The original Google Drive folder returns HTTP 404. In `h4nwei/SPAQ` issue #38, opened 2026-01-16, repository owner `h4nwei` responded on 2026-03-11 with the replacement `chaofengc/IQA-PyTorch-Datasets/spaq.tgz` archive on Hugging Face. The broken Drive link will not be retried.

## Candidate A — Q-Align `spaq.tar`

- Repository: `q-future/q-align-datasets`
- Revision observed in response header: `3da8831b98d683a024b1216720df0cc8a832e376`
- URL: `https://huggingface.co/datasets/q-future/q-align-datasets/resolve/main/spaq.tar`
- Advertised/object bytes: `2,626,897,920`
- Hugging Face linked ETag: `d312242f10af983d5ceaac433c76b7c2b3473e2a2f87782f9db8ee1537dc2194`
- Downloaded SHA-256: `d312242f10af983d5ceaac433c76b7c2b3473e2a2f87782f9db8ee1537dc2194`
- Members: one directory plus 11,125 unique `.jpg` files; no tables or other files.
- Status: archive integrity and all-image readability verified. It contains 11,125 downsampled/re-encoded JPEGs, no annotations, and no embedded EXIF.
- License warning: the repository card says MIT, but this does not establish that underlying SPAQ images or annotations are MIT-licensed.

## Candidate B — author-recommended `spaq.tgz`

- Repository: `chaofengc/IQA-PyTorch-Datasets`
- Revision observed in response header: `f97c4f5dbdbdea7039bc9b80db808313347b1bef`
- URL: `https://huggingface.co/datasets/chaofengc/IQA-PyTorch-Datasets/resolve/main/spaq.tgz`
- Advertised/object bytes: `34,841,101,349` (32.45 GiB)
- Hugging Face linked ETag: `54e0cb49dc8359f9293e746d452d8b12ec653056c9b12a4f97798a984c7418f7`
- Repository card license: CC-BY-SA-4.0; applicability to every underlying SPAQ asset still requires caution.
- Local D: drive free before download: 227,981,836,288 bytes. The archive will be listed without extraction first. It is not suitable for Kaggle's documented 20 GB saved working directory as a monolithic uploaded/extracted asset.
- Downloaded bytes and SHA-256 exactly match the advertised object. Streaming inspection found 11,125 unique `SPAQ/TestImage/NNNNN.jpg` members, three workbooks, and 35,476,869,087 uncompressed file bytes. Full extraction was not performed.
- Five sampled originals have full-resolution pixels and embedded EXIF; matching Q-Align files are smaller, re-encoded, typically orientation-applied variants with EXIF removed. Exact filenames still provide a verified 11,125/11,125 key join.

## Original annotation and EXIF verification

The archive contains `MOS and Image attribute scores.xlsx`, `Scene category labels.xlsx`, and `EXIF_tags.xlsx`. Reconstructing the first 24 columns of the IQA-PyTorch metadata mirror produced zero identifier and zero value mismatches over all 11,125 rows.

`EXIF_tags.xlsx` has `Source EXIF tags` and `APEX EXIF tags` sheets. The official demo consumes the eight APEX values. The APEX sheet's fourth and fifth headers are semantically reversed: observed values and the paper's equations show `Sensitivity vaue` carries scene brightness while `Brightness value` carries ISO sensitivity. The canonical manifest renames them explicitly.

A second streaming pass read embedded EXIF from every original JPEG. Scene brightness is observed in 8,546 rows and author-estimated in 2,579, matching the paper's missing-manufacturer-brightness caveat. `03461.jpg` has source exposure `1,000,000` and non-finite embedded exposure; it is retained and marked invalid, never corrected or represented as measured. The author APEX shutter value remains a distinct traceable field.

Completed artifacts: `data/manifest.parquet`, `data/provenance.json`, `evidence/spaq_original_workbook_audit.json`, `evidence/spaq_embedded_exif_audit.json`, and `evidence/spaq_manifest_audit.json`.

## Gate rule

The EXIF data gate closed after real images, MOS, APEX features, source values, and embedded provenance were matched across all 11,125 exact filenames. Missing/estimated fields remain explicit; no annotation is synthesized.

## Duplicate and leakage audit

Every JPEG payload in the Q-Align archive was SHA-256 hashed. There are 322 exact duplicate groups, each containing two filenames. These are not duplicate annotation rows: 321 pairs have different MOS values and all 322 differ in at least one subjective quality or scene field. Conversely, all eight EXIF-derived columns are identical within every exact pair. The Q-Align train/test field separates 103 exact pairs, and the ten legacy 80/20 fields separate 86--120 pairs each. Those inherited splits are therefore rejected for this study. The project split must treat each exact pair as a single group.

The 64-bit dHash audit also produced 747 non-exact candidate pairs within Hamming distance four. Representative review at every distance 0--4 found repeated scene captures with changed exposure/focus, including a distance-three pair with a 68.47 MOS difference. The project therefore conservatively uses the complete 1,069-edge graph (exact and non-exact candidates) as grouping constraints. Connected components are indivisible; see `evidence/near_duplicate_visual_review.md`. This policy may merge a rare false-positive candidate but prevents the repeated-scene leakage actually observed in the reviewed pairs.

Evidence: `evidence/q_align_duplicate_audit.json` and `evidence/q_align_duplicate_label_audit.json`.

