# COLLABORATION_DRAFTS.md

**Repository:** winanhle/CS203-GROUP-9  
**Branch:** codex/phuc-context  
**Project:** SC203 Group 9 — CM4IR  

---

## 1. Group-Chat Alignment Messages (Ready to Copy-Paste)

### 1.1 Message to Long & Anh

```
Hi Long & Anh,

To keep our runs comparable across the group, I propose we standardize on the following artifact pattern for every experiment:

- manifest.json (one per dataset/config)
- per-image records in records/*.json
- aggregated results.csv
- summary.json with run-level metrics

This matches the pattern Long has been using (manifest.json + records/*.json + results.csv + summary.json).

Before I lock the baseline config, can we align on the inpainting delta settings for Table 5? Specifically:

- Should the group baseline use Table 5's --deltas '0.1,0.1,0,0' for Cat SRx4?
- Or should we use deltas=0 as in scripts/run_lsun_cat_sr4.py?

On my branch codex/phuc-context, I have frozen configs/lsun_cat_57_manifest.json — 57 unique 256x256 LSUN Cat PNGs, SHA-256 locked — so we have a common, verifiable image set to compare against.

Let me know what you think so I can align the PR and avoid divergent baselines.

— Phuc
```

### 1.2 Message to Quynh Anh

```
Hi Quynh Anh,

I'm working on the --context_mode {none, telea, spatial, clip} inpainting guidance module on branch codex/phuc-context.

To keep our changes non-overlapping, could you let me know which specific external algorithm you are plugging into CM4IR? I want to make sure your integration doesn't conflict with the context-mode guidance paths I've implemented in:

- functions/context_guidance.py
- main.py
- guided_diffusion/diffusion.py
- functions/cm4ir_scheme.py
- functions/median_inpainting.py

Thanks!

— Phuc
```

---

## 2. Draft Pull Request Summary (`codex/phuc-context` → `main`)

**Branch:** `codex/phuc-context`  
**Target:** `main`  
**Scope:** CM4IR inpainting context-guidance module and ablation tooling.

### Implemented changes
- Implemented `--context_mode {none, telea, spatial, clip}` inpainting guidance across:
  - `functions/context_guidance.py`
  - `main.py`
  - `guided_diffusion/diffusion.py`
  - `functions/cm4ir_scheme.py`
  - `functions/median_inpainting.py`
- `--context_mode none` is bit-for-bit parity with commit `3d09c114`.
- 82/82 unit/contract tests passing.
- Added contiguous mask `exp/inpainting_masks/center_box_mask.npy`.
- Added resumable runner `scripts/run_inpainting_ablation.py`.
- Added statistical/visual analysis suite `scripts/analyze_inpainting_results.py`.

### Git hygiene
- Checkpoints (`.pt`) remain gitignored.
- Raw per-image output dumps (`results/**/images/`) remain gitignored.

---

## 3. Group Comparison Table Update — 57-Image LSUN Cat Inpainting

**Conditions:** sigma_y = 0.05, 3 seeds (1234–1236) for baseline vs winners.  
**Image set:** `configs/lsun_cat_57_manifest.json` (57 unique 256×256 LSUN Cat PNGs, SHA-256 locked).

### 3.1 `random_80_mask.npy` (scattered, 80% missing)

| Mode | Seeds | Full PSNR (dB) | Full LPIPS | Hole PSNR (dB) | Hole LPIPS | Notes |
|------|-------|----------------|------------|----------------|------------|-------|
| none (baseline) | 1234–1236 (mean) | 24.9957 | 0.4330 | 24.6982 | 0.3708 | — |
| telea | 1234–1236 (mean) | 25.0530 | 0.4500 | 24.7348 | 0.3884 | — |
| spatial (winner) | 1234–1236 (mean) | 25.2017 (+0.21 dB) | 0.4289 (-0.0041) | 24.9508 (+0.25 dB) | 0.3566 (-0.0142) | vs baseline |
| clip | s1234 | 13.9245 | 0.7211 | — | — | vs clip_neg_ctrl below |
| clip_neg_ctrl | s1234 | 13.6323 | 0.7310 | — | — | negative control |

### 3.2 `lorem_ipsum_mask.npy` (text mask)

| Mode | Seeds | Full PSNR (dB) | Full LPIPS | Hole PSNR (dB) | Hole LPIPS | Notes |
|------|-------|----------------|------------|----------------|------------|-------|
| none (baseline) | 1234–1236 (mean) | 26.7818 | 0.3117 | 26.1866 | 0.0340 | — |
| telea (winner for PSNR) | 1234–1236 (mean) | 26.8999 (+0.12 dB) | 0.3112 (-0.0005) | 26.7261 (+0.54 dB) | 0.0329 (-0.0011) | vs baseline |
| spatial | s1234 | 26.7914 | 0.3117 | 26.2626 | 0.0305 | — |
| clip | s1234 | 21.5159 | 0.4688 | — | — | vs clip_neg_ctrl below |
| clip_neg_ctrl | s1234 | 21.3300 | 0.4737 | — | — | negative control |

### 3.3 `center_box_mask.npy` (contiguous 128×128 hole)

| Mode | Seeds | Full PSNR (dB) | Full LPIPS | Hole PSNR (dB) | Hole LPIPS | Notes |
|------|-------|----------------|------------|----------------|------------|-------|
| none (baseline) | 1234–1236 (mean) | 19.2964 | 0.4293 | 13.9829 | 0.1937 | — |
| telea (winner) | 1234–1236 (mean) | 19.5378 (+0.24 dB) | 0.4292 (-0.0001) | 14.2504 (+0.27 dB) | 0.1915 (-0.0022) | vs baseline |
| spatial | s1234 | 19.3057 | 0.4286 | 13.9954 | 0.1929 | — |
| clip | s1234 | 16.0803 | 0.4680 | — | — | vs clip_neg_ctrl below |
| clip_neg_ctrl | s1234 | 15.9972 | 0.4681 | — | — | negative control |

### 3.4 Stratified 95% Bootstrap CIs (10,000 resamples, n = 171 overall)

- **spatial**: scattered `mask_group` Delta Full PSNR +0.1084 dB [0.0863, 0.1321], Delta Hole PSNR +0.1789 dB [0.1517, 0.2097], Delta Hole LPIPS -0.0089 [-0.0105, -0.0074]; overall Delta Hole PSNR +0.1223 dB [0.1011, 0.1455], Delta Hole LPIPS -0.0062 [-0.0074, -0.0051]
- **telea**: overall Delta Full PSNR +0.1390 dB [0.0861, 0.1937], Delta Hole PSNR +0.2811 dB [0.1888, 0.3854]; contiguous `center_box` Delta Full PSNR +0.2413 dB [0.1558, 0.3220], Delta Hole LPIPS -0.0023 [-0.0030, -0.0015]; `high_edge` Delta Full PSNR +0.2193 dB [0.1407, 0.3028]
- **clip vs clip_neg_ctrl**: target prompt consistently outperforms wrong-prompt negative control across all 3 masks (+0.29 dB on `random_80`, +0.19 dB on `lorem_ipsum`, +0.08 dB on `center_box`), though raw 1-step CLIP gradient steering without warm-up degrades PSNR/LPIPS relative to unsteered CM4IR.