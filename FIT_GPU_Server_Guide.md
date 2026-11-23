# FIT HCMUS GPU Server Guide — SC203 Group 9 (CM4IR Inpainting)

## 1. Prerequisites & VPN Setup (Required Outside HCMUS Campus)

- Install **OpenVPN Connect**:
  - Windows: https://openvpn.net/client/
  - macOS: Tunnelblick https://tunnelblick.net/downloads.html
- Import the `.ovpn` profile (`vpnuser04-macos-windows.ovpn`) attached in Thầy Huỳnh Viết Thám's FIT GPU email.
- Connect with username `vpnuser04` and the VPN password from the email.
- Verify VPN is active: you should get a `10.200.200.x` IP address.
- If you are inside the HCMUS campus network, VPN is **not** needed.

## 2. Connecting via SSH

```bash
ssh hvtham01@172.29.74.81
```
- Port: `22`
- Password: use the SSH password from Thầy Viết Thám's FIT GPU email (note: it ends with `%`).

Job status dashboard (browser, while on VPN or campus network):
```
http://172.29.74.82:8012/
```

### Important FIT Server Rules
- Store **ALL** code, datasets, caches, and results inside `/media02/hvtham01` (valid until `11/02/2027`).
- Back up results to your own machine every ~2 days in case of power outage or maintenance.
- Always run GPU workloads via SLURM (`sbatch`), never directly on the login node.

## 3. Project Layout & Environment on the Server

- Base directory: `/media02/hvtham01/phuc_cm4ir`
- Git repo (branch `codex/phuc-context`): `/media02/hvtham01/phuc_cm4ir/repo`
- Python virtualenv:
  ```bash
  source /media02/hvtham01/phuc_cm4ir/venv/bin/activate
  ```

### Required offline cache environment variables
Compute nodes have no internet. Export these in your shell and in every SLURM script:

```bash
export TORCH_HOME="/media02/hvtham01/phuc_cm4ir/.cache/torch"
export HF_HOME="/media02/hvtham01/phuc_cm4ir/.cache/hf"
export HF_HUB_OFFLINE=1
```

### Pre-installed datasets & checkpoints
Inside `/media02/hvtham01/phuc_cm4ir/repo/`:

**LSUN Cat 300 (256×256):**
- Dataset: `exp/datasets/lsun_cat` (with SHA-256 manifest `configs/lsun_cat_300_manifest.json` locking the exact 300 images)
- Checkpoint: `exp/logs/lsun_cat/cd_cat256_lpips.pt`
- Config: `lsun_cat_256.yml`
- Masks: `center_box_mask.npy`, `random_80_mask.npy`, `lorem_ipsum_mask.npy`

**ImageNet-64 1,000 (64×64):**
- Dataset: `exp/datasets/imagenet_1000`
- Checkpoint: `exp/logs/imagenet/cd_imagenet64_lpips.pt`
- Config: `imagenet_64_cc.yml`
- Masks: `center_box_64_mask.npy`, `random_80_64_mask.npy`

## 4. Submitting & Running Experiments with SLURM

Ready-to-use SLURM scripts are in `/media02/hvtham01/phuc_cm4ir/slurm_*.sh` (e.g., `slurm_im1000.sh`, `slurm_im1000_poisson3.sh`, `slurm_cat300_bridge.sh`, `slurm_feather_cb.sh`).
All SLURM logs are written to `/media02/hvtham01/phuc_cm4ir/logs/`.

### Parameterized SLURM Template (Run Any Mode via `sbatch ... <mode>`)
Save as `/media02/hvtham01/phuc_cm4ir/slurm_run_mode.sh` (accepts mode name as `$1`, defaults to `poisson_bridge_feather`):

```bash
#!/usr/bin/env bash
#SBATCH --job-name=cm4ir_eval
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=3
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/media02/hvtham01/phuc_cm4ir/logs/slurm_eval_%j.out
#SBATCH --error=/media02/hvtham01/phuc_cm4ir/logs/slurm_eval_%j.err

set -eu
MODE="${1:-poisson_bridge_feather}"
BASE=/media02/hvtham01/phuc_cm4ir
export TORCH_HOME="$BASE/.cache/torch"
export HF_HOME="$BASE/.cache/hf"
export HF_HUB_OFFLINE=1
source "$BASE/venv/bin/activate"
cd "$BASE/repo"

# 1. LSUN Cat 300 (MUST pass --dataset exp/datasets/lsun_cat, --inputs-manifest, and --deltas 0,0,1.0,1.0)
python scripts/run_inpainting_ablation.py \
  --dataset exp/datasets/lsun_cat \
  --inputs-manifest configs/lsun_cat_300_manifest.json \
  --mask center_box_mask.npy \
  --sigma-y 0.05 \
  --seed 1234 \
  --batch-size 8 \
  --deltas 0,0,1.0,1.0 \
  --deltas-injection-type 1 \
  --verify-official-checkpoint \
  --output "results/ablation_300_table5/${MODE}_center_box_s1234" \
  --context-mode "$MODE"

# 2. ImageNet-64 1,000 (MUST pass --config imagenet_64_cc.yml, --checkpoint, and --expected-count 1000)
python scripts/run_inpainting_ablation.py \
  --config imagenet_64_cc.yml \
  --checkpoint exp/logs/imagenet/cd_imagenet64_lpips.pt \
  --dataset exp/datasets/imagenet_1000 \
  --mask center_box_64_mask.npy \
  --sigma-y 0.05 \
  --seed 1234 \
  --batch-size 64 \
  --output "results/ablation_imagenet1000/${MODE}_center_box_s1234" \
  --context-mode "$MODE" \
  --expected-count 1000
```

Submit any mode in one command without editing the script:
```bash
sbatch /media02/hvtham01/phuc_cm4ir/slurm_run_mode.sh poisson_bridge_feather
```

- **Exact CLI flag names (do not guess):**
  - `--context-mode` (NOT `--mode` or `--modes`): `none`, `telea`, `spatial`, `clip`, `poisson`, `poisson_bridge`, `poisson_bridge_sharp`, `poisson_bridge_freq`, `poisson_bridge_feather`
  - `--mask` (NOT `--mask-type` or `--mask-types`)
  - `--expected-count` (NOT `--num-images`)
  - `--output` (REQUIRED; NOT `--output-root`)
  - `--batch-size`: `64` for ImageNet-64 (64×64); `4`–`8` for LSUN Cat (256×256)
  - `--seed 1234`: uses per-image deterministic `torch.Generator(device).manual_seed(seed + idx)`
  - `--sigma-y 0.05`

### Essential SLURM commands

Submit a job:
```bash
sbatch /media02/hvtham01/phuc_cm4ir/slurm_feather_cb.sh
```

Check queue:
```bash
squeue -u hvtham01
```

Check logs (all logs are in `/media02/hvtham01/phuc_cm4ir/logs/`):
```bash
tail -n 30 /media02/hvtham01/phuc_cm4ir/logs/slurm_feather_<JOBID>.out
tail -n 30 /media02/hvtham01/phuc_cm4ir/logs/slurm_feather_<JOBID>.err
```

Cancel a job:
```bash
scancel <JOBID>
```

Check GPU nodes:
```bash
sinfo
```

## 5. Running Statistical Analysis & Downloading Results

Run analysis on the server (or at the end of the SLURM script):

```bash
# LSUN Cat 300 analysis
python scripts/analyze_inpainting_results.py \
  --results-root results/ablation_300_table5 \
  --output-dir results/ablation_300_table5_analysis \
  --n-boot 10000 --bootstrap-seed 42

# ImageNet-64 1,000 analysis (MUST pass --dataset exp/datasets/imagenet_1000 for 64x64 figure rendering)
python scripts/analyze_inpainting_results.py \
  --results-root results/ablation_imagenet1000 \
  --output-dir results/ablation_imagenet1000_analysis \
  --dataset exp/datasets/imagenet_1000 \
  --n-boot 10000 --bootstrap-seed 42
```

Download results to your laptop (macOS/Linux/Windows PowerShell):

```bash
scp -r hvtham01@172.29.74.81:/media02/hvtham01/phuc_cm4ir/repo/results/ablation_imagenet1000_analysis ./results/
```

## 6. Checklist Before Submitting a New `--context-mode` to SLURM

Whenever adding a new `--context-mode` (e.g., `poisson_bridge_feather`), you **MUST** register it in all 5 places and run a 1-image CLI test before calling `sbatch`:

1. `guided_diffusion/diffusion.py`:
   - `cm4ir_wrapper()` allowlist (~line 119)
   - `build_context_guidance()` trigger tuple (~line 320 — if missed, `context_guidance` defaults to `None` and runs unguided!)
   - Initialization dispatch (~line 430)
2. `functions/cm4ir_scheme.py`:
   - `_POISSON_MODES` tuple (~line 84) and per-step guidance branch (~line 191)
3. `functions/context_guidance.py`:
   - `build_context_guidance()` mode tuple and schedule parameters (~line 413)
4. `scripts/run_inpainting_ablation.py`:
   - `build_context_params()` (~line 340) and `--context-mode` argparse `choices` (~line 906)
5. `scripts/analyze_inpainting_results.py`:
   - `CANONICAL_CONDITION_ORDER` and `CONDITION_DISPLAY_NAMES` (~lines 23–43)

## 7. Common Pitfalls & Troubleshooting

- **Don't overwrite each other's outputs:** Use a distinct `--output` folder or your own subfolder under `/media02/hvtham01/` when testing experimental changes.
- **`ValueError: Resume manifest differs` or crashed partial run:** If a previous job crashed partway through and left an incomplete `manifest.json` or wrong records, remove that output folder (`rm -rf results/.../<run_dir>`) before resubmitting.
- **`ValueError: SHA-256 mismatch` in `analyze_inpainting_results.py` on LSUN Cat 300:** Always pass `--dataset exp/datasets/lsun_cat --inputs-manifest configs/lsun_cat_300_manifest.json --deltas 0,0,1.0,1.0 --deltas-injection-type 1` so the 300 image indices and SHA-256 hashes match the rest of `ablation_300_table5`.
- **Offline compute nodes:** If PyTorch/HuggingFace tries to download weights and hangs/fails, make sure `TORCH_HOME`, `HF_HOME`, and `HF_HUB_OFFLINE=1` are exported in your SLURM script.
- **SSH connection timed out:** Make sure OpenVPN is connected (`vpnuser04`) if you are off-campus.