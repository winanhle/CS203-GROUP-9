# Context-Aware & Non-Uniform Inpainting Guidance for CM4IR on LSUN Cat

## Overview

- **Branch:** `codex/phuc-context` (base commit `3d09c114`, latest commit `d8e63d4`)
- **Hardware & Runtime:** NVIDIA GeForce RTX 5060 Laptop GPU (8,151 MiB VRAM, sm_120), Python 3.12.16, PyTorch 2.11.0+cu128, OpenCV 5.1.0, LPIPS 0.1.4 (VGG16), OpenCLIP 4.2.2 (ViT-B-32, laion2b_s34b_b79k).
- **Verification:** 82/82 unit and contract tests passing (`tests/test_context_guidance.py` and `tests/test_ablation_and_analysis.py`).

## Methodology

### Mask invariant
`mask == 0` denotes missing pixels (hole), while `mask != 0` denotes observed pixels. This convention is required for anti-aliased uint8 masks such as `lorem_ipsum_mask.npy`.

### Context modes (`--context_mode {none,telea,spatial,clip}`)

- **none:** Unmodified CM4IR baseline (`median_inpainting.py` + `cm4ir_scheme.py`), verified bit-for-bit identical against `3d09c114`.
- **telea:** Replaces expanding-box median fill with OpenCV Telea/Navier-Stokes PDE initialization (`cv2.inpaint`), copying observed pixels verbatim.
- **spatial:** Feathers `BP_guidance` smoothly inward across the mask border using `cv2.distanceTransform` (L2, `spatial_sigma=8.0`) and adapts stochasticity `eta(x,y)` inside the hole (`eta_hole=0.25`) versus observed pixels (`eta_obs=0.02`).
- **clip:** Null-space `(I - A^+ A)` OpenCLIP text-image gradient steering on `x0_t` (target prompt `'a photograph of a cat'` vs. wrong-prompt negative control `'a photograph of a red fire hydrant in outer space'`, `clip_scale=0.25`, `clip_steps=1`).

## Experimental Setup

- **Dataset:** SHA-256-locked 57-image LSUN Cat subset (`configs/lsun_cat_57_manifest.json`) and 8-image repo-tracked smoke subset (`configs/lsun_cat_8_smoke_manifest.json`).
- **Masks:**
  - `random_80_mask.npy` (80% scattered pixel drop)
  - `lorem_ipsum_mask.npy` (superimposed text mask)
  - `center_box_mask.npy` (contiguous 128x128 center hole, 25% missing)
- **Protocol:** `sigma_y=0.05` across all 57 images (plus 3 seeds: 1234, 1235, 1236 on `none`, `telea`, and per-mask winners), and `sigma_y=0.025` on the 8-image smoke set. Metrics: full-image PSNR, full-image VGG-LPIPS, masked-hole PSNR, masked-hole VGG-LPIPS.

## Quantitative Results (57-Image LSUN Cat, sigma_y=0.05)

### random_80_mask.npy

| Mode | Seeds | Full PSNR (dB) | Full LPIPS | Hole PSNR (dB) | Hole LPIPS |
|------|-------|----------------|------------|----------------|------------|
| none | 3-seed | 24.9957 | 0.4330 | 24.6982 | 0.3708 |
| telea | 3-seed | 25.0530 | 0.4500 | 24.7348 | 0.3884 |
| spatial | 3-seed winner | 25.2017 | 0.4289 | 24.9508 | 0.3566 |
| clip | s1234 | 13.9245 | 0.7211 | 13.0010 | 0.7188 |
| clip_neg_ctrl | s1234 | 13.6323 | 0.7310 | 12.7059 | 0.7290 |

### lorem_ipsum_mask.npy

| Mode | Seeds | Full PSNR (dB) | Full LPIPS | Hole PSNR (dB) | Hole LPIPS |
|------|-------|----------------|------------|----------------|------------|
| none | 3-seed | 26.7818 | 0.3117 | 26.1866 | 0.0340 |
| telea | 3-seed PSNR winner | 26.8999 | 0.3112 | 26.7261 | 0.0329 |
| spatial | s1234 | 26.7914 | 0.3117 | 26.2626 | 0.0305 |
| clip | s1234 | 21.5159 | 0.4688 | 12.5977 | 0.3359 |
| clip_neg_ctrl | s1234 | 21.3300 | 0.4737 | 12.3500 | 0.3407 |

### center_box_mask.npy

| Mode | Seeds | Full PSNR (dB) | Full LPIPS | Hole PSNR (dB) | Hole LPIPS |
|------|-------|----------------|------------|----------------|------------|
| none | 3-seed | 19.2964 | 0.4293 | 13.9829 | 0.1937 |
| telea | 3-seed winner | 19.5378 | 0.4292 | 14.2504 | 0.1915 |
| spatial | s1234 | 19.3057 | 0.4286 | 13.9954 | 0.1929 |
| clip | s1234 | 16.0803 | 0.4680 | 10.3365 | 0.2687 |
| clip_neg_ctrl | s1234 | 15.9972 | 0.4681 | 10.2463 | 0.2687 |

## Paired Bootstrap 95% CIs & Stratified Analysis

Results are drawn from `results/ablation_57_analysis/stratified_summary.csv` (10,000 paired bootstrap resamples).

- **spatial on scattered masks (n=114):** Delta Full PSNR +0.1084 dB [0.0863, 0.1321], Delta Hole PSNR +0.1789 dB [0.1517, 0.2097], Delta Hole LPIPS -0.0089 [-0.0105, -0.0074].
- **telea on contiguous center_box (n=57):** Delta Full PSNR +0.2413 dB [0.1558, 0.3220], Delta Hole PSNR +0.2675 dB [0.1629, 0.3653], Delta Hole LPIPS -0.0023 [-0.0030, -0.0015].
- **Texture stratification (Sobel median split):** telea shows larger gains on high_edge images (+0.2193 dB [0.1407, 0.3028]) than flat images (+0.0614 dB [-0.0058, 0.1282]); spatial improves both flat (+0.0956 dB) and high_edge (+0.0532 dB) images.
- **CLIP negative control:** The target prompt outperforms the wrong-prompt negative control across all three masks (+0.29 dB on `random_80`, +0.19 dB on `lorem_ipsum`, +0.08 dB on `center_box`), though unscheduled 1-step CLIP steering degrades PSNR/LPIPS relative to unsteered CM4IR.

## Visual Artifacts & Reproducibility

Figures and analysis scripts are located under `results/ablation_57_analysis/figures/` (`comparison_grid_*.png`, `error_heatmap_*.png`, `delta_ci_forest.png`) and `scripts/` (`run_full_ablation_suite.py`, `run_inpainting_ablation.py`, `analyze_inpainting_results.py`).