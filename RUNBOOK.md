# CM4IR Inpainting — Team & Agent Runbook

**SC203 Group 9**  
*Team:* Lê Quang Nguyên Phúc, Lê Quỳnh Anh, Lê Trần Hoàng Long  
*Purpose:* Quick-reference runbook for team members and AI agents during context compaction.

---

## 1. Repository & Branch
- **Local repo path (Windows):** `C:\Users\lqngu\OneDrive\Desktop\Semester 4\CS203-GROUP-9`
- **Active branch:** `codex/phuc-context`
- **Key files:**
  - `functions/harmonic_poisson.py`: harmonic/Poisson solver, `gaussian_blur_2d`, `saliency_bridge_inpainting` (`gamma=5.0`)
  - `functions/context_guidance.py`: context mode presets (`none`, `telea`, `spatial`, `clip`, `poisson`, `poisson_bridge`, `poisson_bridge_sharp`, `poisson_bridge_freq`)
  - `functions/cm4ir_scheme.py`: 4-NFE CM4IR restoration loop, low-frequency bridge context anchor, per-step `eta_map`, per-sample deterministic `torch.Generator` noise
  - `guided_diffusion/diffusion.py`: dataset loading, batched sampling with `sample_ids`, per-image deterministic seeding (`seed + idx`)
  - `scripts/run_inpainting_ablation.py`: resumable ablation runner (`--batch-size`, `--seed`, `--context-mode`, `--mask`, `--sigma-y`, `--expected-count`)
  - `scripts/analyze_inpainting_results.py`: paired delta & 10,000-resample bootstrap CI analysis + figure generation

---

## 2. Context Modes (Strictly 4 NFEs, 0 extra model calls)
- `none`: vanilla CM4IR baseline
- `telea`: OpenCV Telea initialization
- `spatial`: spatial noise-level (`eta_map`) guidance
- `poisson`: harmonic/Poisson prior + spatial guidance
- `poisson_bridge` (`combo_anneal`): saliency-weighted bridge prior (`gamma=5.0`), `edm_sigmas=(3.5, 1.6, 0.55, 0.08)`, `eta_holes_seq=(0.95, 1.0, 1.0)`, `bridge_alphas=(0.40, 0.18, 0.0, 0.0)`, `bridge_blur_sigma=12.0`
- `poisson_bridge_sharp`: same as `poisson_bridge` with `eta_holes_seq=(1.0, 1.0, 1.0)` (lowest LPIPS / sharpest texture)
- `poisson_bridge_freq`: combines `poisson_bridge_sharp` (`eta_holes_seq=(1.0, 1.0, 1.0)`) with a tighter, stronger low-frequency anchor (`bridge_alphas=(0.50, 0.22, 0.0, 0.0)`, `bridge_blur_sigma=10.0`) for both higher PSNR and sharpest clarity

---

## 3. Running Locally on Windows (RTX 5060 GPU)
- **Python environment:** `.\.venv\Scripts\python.exe` (PyTorch `2.11.0+cu128`, CUDA enabled; do NOT use system `C:\Python314\python.exe` which lacks `torchvision`, `cv2`, `tqdm`)
- **Run unit tests:**
  ```powershell
  cd "C:\Users\lqngu\OneDrive\Desktop\Semester 4\CS203-GROUP-9"
  .\.venv\Scripts\python.exe -m pytest tests/ -q
  ```
- **Local datasets & checkpoints:**
  - LSUN Cat 256×256: `exp/datasets/lsun_cat`, checkpoint `exp/logs/lsun_cat/cd_cat256_lpips.pt`, config `lsun_cat_256.yml`
  - Masks in `exp/inpainting_masks/`: `center_box_mask.npy`, `random_80_mask.npy`, `lorem_ipsum_mask.npy`
- **Example local run command (e.g., 10-image or 57-image test):**
  ```powershell
  .\.venv\Scripts\python.exe scripts/run_inpainting_ablation.py `
    --config lsun_cat_256.yml `
    --checkpoint exp/logs/lsun_cat/cd_cat256_lpips.pt `
    --dataset exp/datasets/lsun_cat `
    --mask center_box_mask.npy `
    --sigma-y 0.05 --seed 1234 --batch-size 4 `
    --context-mode poisson_bridge_freq `
    --output results/local_test/poisson_bridge_freq_center_box_s1234
  ```
- **Note:** `run_inpainting_ablation.py` is resumable (caches per-image JSON in `records/` and skips completed runs if `summary.json` exists).

---

## 4. Running on the FIT HCMUS GPU Server (SLURM)
- **Step 1 — VPN (required outside HCMUS campus):**
  - Install OpenVPN Connect (Windows) or Tunnelblick (macOS).
  - Import `.ovpn` profile (local copy: `C:\Users\lqngu\.gemini\antigravity\brain\980ccb70-dd0a-4485-94e0-7565e1b061b5\scratch\vpn\vpnuser04\vpnuser04-macos-windows.ovpn`).
  - Connect with username `vpnuser04` (password from Thầy Viết Thám's FIT GPU email). Verify interface `10.200.200.x` is active.
- **Step 2 — SSH / SFTP:**
  - Host: `172.29.74.81`, Port: `22`, User: `hvtham01` (password from Thầy Viết Thám's FIT GPU email — note it ends with `%`).
  - Job dashboard: `http://172.29.74.82:8012/`
  - Storage root (mandatory per FIT rules, valid until 11/02/2027, back up locally every ~2 days): `/media02/hvtham01/phuc_cm4ir`
  - Repo on server: `/media02/hvtham01/phuc_cm4ir/repo`
  - Virtualenv on server: `source /media02/hvtham01/phuc_cm4ir/venv/bin/activate`
  - Offline caches (required on compute nodes):
    ```bash
    export TORCH_HOME="/media02/hvtham01/phuc_cm4ir/.cache/torch"
    export HF_HOME="/media02/hvtham01/phuc_cm4ir/.cache/hf"
    export HF_HUB_OFFLINE=1
    ```
- **Step 3 — Datasets & Checkpoints on Server:**
  - LSUN Cat 300: `exp/datasets/lsun_cat_300`, checkpoint `exp/logs/lsun_cat/cd_cat256_lpips.pt`, config `lsun_cat_256.yml`
  - ImageNet-64 1,000: `exp/datasets/imagenet_1000`, checkpoint `exp/logs/imagenet/cd_imagenet64_lpips.pt`, config `imagenet_64_cc.yml`, masks `center_box_64_mask.npy` and `random_80_64_mask.npy`
- **Step 4 — Submitting & Monitoring SLURM Jobs:**
  - Existing SLURM scripts live in `/media02/hvtham01/phuc_cm4ir/slurm_*.sh` (e.g. `slurm_im1000.sh`, `slurm_im1000_poisson3.sh`, `slurm_cat300_bridge.sh`).
  - Submit: `sbatch /media02/hvtham01/phuc_cm4ir/slurm_im1000_poisson3.sh`
  - Monitor: `squeue -u hvtham01` and `tail -f /media02/hvtham01/phuc_cm4ir/logs/slurm_im1000_p3_<JOBID>.out`
  - Cancel: `scancel <JOBID>`
- **Step 5 — Analyzing & Pulling Results:**
  ```bash
  python scripts/analyze_inpainting_results.py \
    --results-root results/ablation_imagenet1000 \
    --output-dir results/ablation_imagenet1000_analysis \
    --n-boot 10000 --bootstrap-seed 42
  ```
  Pull back to local machine via `scp -r` or Python `paramiko` SFTP into `CS203-GROUP-9/results/`.

---

## 5. Current Benchmark Summary (Reference Baseline)
- **LSUN Cat 300 (`ablation_300_table5`, 4 NFEs, `sigma_y=0.05`, seed `1234`, PSNR ↑ / LPIPS ↓):**
  - `center_box`: `none` 19.08 / 0.4331, `telea` 19.32 / 0.4326, `spatial` 19.08 / 0.4325, `poisson` 20.37 / 0.4088, `poisson_bridge` **20.51** / 0.3861, `poisson_bridge_sharp` 20.34 / **0.3814**
  - `random_80`: `none` 25.30 / 0.4313, `telea` 25.37 / 0.4484, `spatial` 25.50 / 0.4278, `poisson` **25.98 / 0.4117**
  - `lorem_ipsum`: `none` 27.59 / 0.3123, `telea` 27.73 / 0.3119, `spatial` 27.60 / 0.3124, `poisson` **27.80 / 0.3110**
- **ImageNet-64 1,000 (`ablation_imagenet1000`, `sigma_y=0.05`, seed `1234`):**
  - Baselines completed: `random_80` (`none` 22.03/0.3717, `telea` 22.02/0.4085, `spatial` 22.21/0.3662); `center_box` (`none` 20.06/0.3642, `telea` 20.29/0.3670, `spatial` 20.08/0.3629).
  - `poisson`, `poisson_bridge`, `poisson_bridge_sharp` running on FIT as job **`85373`** (`--batch-size 64`, Tesla V100-32GB).
- **8-Image Local RTX 5060 Prototype (`lsun_cat`, `center_box`, 4 NFEs, ~53s):**
  - `poisson_bridge` (`0.95`): 20.58 / 0.2376 (Hole 14.62 / 0.1713)
  - `poisson_bridge_sharp` (`1.0`): 20.38 / 0.2308 (Hole 14.41 / 0.1645)
  - `hybrid` (`0.98`, `a=0.45`): 20.53 / 0.2351 (Hole 14.56 / 0.1688)
  - `poisson_bridge_freq` (`1.0`, `a=0.50`, `σ=10`): 20.41 / **0.2303** (Hole 14.44 / **0.1639**) — strictly beats `sharp` on all 4 metrics.
  - Fast local prototyping tip: pipe a standalone PyTorch script directly into `& "C:\Users\lqngu\OneDrive\Desktop\Semester 4\CS203-GROUP-9\.venv\Scripts\python.exe" -` (loads `cd_cat256_lpips.pt` once and evaluates 8 images × 5 configs in ~50s).

---

## 6. Quick Troubleshooting Notes
- Always use `.\.venv\Scripts\python.exe` on Windows, never system `python`.
- On FIT, if SSH fails with `Could not resolve hostname`, use the IP `172.29.74.81` directly with OpenVPN connected.
- If a crashed run leaves a partial `manifest.json` and the next run raises `ValueError: Resume manifest differs`, delete the incomplete `--output` directory (e.g. `rm -rf results/ablation_imagenet1000/<cond>_<mask>_s1234`) before resubmitting.
- Use `--batch-size 64` for ImageNet-64 (64×64) and `--batch-size 4` or `8` for LSUN Cat (256×256) depending on VRAM; per-image `torch.Generator(device).manual_seed(seed + idx)` keeps results deterministic per image.