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
- Dataset: `exp/datasets/lsun_cat_300`
- Checkpoint: `exp/logs/lsun_cat/cd_cat256_lpips.pt`
- Config: `lsun_cat_256.yml`
- Masks: `center_box_mask.npy`, `random_80_mask.npy`, `lorem_ipsum_mask.npy`

**ImageNet-64 1,000 (64×64):**
- Dataset: `exp/datasets/imagenet_1000`
- Checkpoint: `exp/logs/imagenet/cd_imagenet64_lpips.pt`
- Config: `imagenet_64_cc.yml`
- Masks: `center_box_64_mask.npy`, `random_80_64_mask.npy`

## 4. Submitting & Running Experiments with SLURM

Ready-to-use SLURM scripts are in `/media02/hvtham01/phuc_cm4ir/slurm_*.sh` (e.g., `slurm_im1000.sh`, `slurm_im1000_poisson3.sh`, `slurm_cat300_bridge.sh`).

### Minimal SLURM template
Create a file such as `/media02/hvtham01/phuc_cm4ir/my_job.sh`:

```bash
#!/bin/bash
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=3
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/media02/hvtham01/phuc_cm4ir/logs/%j.out
#SBATCH --error=/media02/hvtham01/phuc_cm4ir/logs/%j.err

export TORCH_HOME="/media02/hvtham01/phuc_cm4ir/.cache/torch"
export HF_HOME="/media02/hvtham01/phuc_cm4ir/.cache/hf"
export HF_HUB_OFFLINE=1

source /media02/hvtham01/phuc_cm4ir/venv/bin/activate

cd /media02/hvtham01/phuc_cm4ir/repo

python scripts/run_inpainting_ablation.py \
  --context-mode poisson_bridge \
  --batch-size 64 \
  --seed 1234 \
  --sigma-y 0.05
```

- `--context-mode` choices: `none`, `telea`, `spatial`, `poisson`, `poisson_bridge`, `poisson_bridge_sharp`, `poisson_bridge_freq`
- `--batch-size 64` for ImageNet-64 (64×64); `--batch-size 4`–`8` for LSUN Cat (256×256)
- `--seed 1234` uses per-image deterministic `torch.Generator(device).manual_seed(seed + idx)`
- `--sigma-y 0.05`

### Essential SLURM commands

Submit a job:
```bash
sbatch /media02/hvtham01/phuc_cm4ir/my_job.sh
```

Check queue:
```bash
squeue -u hvtham01
```

Check logs:
```bash
tail -f /media02/hvtham01/phuc_cm4ir/logs/<job_log>.out
# and
tail -f /media02/hvtham01/phuc_cm4ir/logs/<job_log>.err
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

Run analysis on the server (or inside the SLURM job):

```bash
python scripts/analyze_inpainting_results.py \
  --results-root results/ablation_imagenet1000 \
  --output-dir results/ablation_imagenet1000_analysis \
  --n-boot 10000 --bootstrap-seed 42
```

Download results to your laptop (macOS/Linux/Windows PowerShell):

```bash
scp -r hvtham01@172.29.74.81:/media02/hvtham01/phuc_cm4ir/repo/results/ablation_imagenet1000_analysis ./results/
```

## 6. Common Pitfalls & Troubleshooting

- **Don't overwrite each other's outputs:** Use a distinct `--output` folder or your own subfolder under `/media02/hvtham01/` when testing experimental changes.
- **`ValueError: Resume manifest differs`:** If a previous job crashed partway through and left an incomplete `manifest.json`, remove that incomplete output folder (`rm -rf results/.../<run_dir>`) before resubmitting.
- **Offline compute nodes:** If PyTorch/HuggingFace tries to download weights and hangs/fails, make sure `TORCH_HOME`, `HF_HOME`, and `HF_HUB_OFFLINE=1` are exported in your SLURM script.
- **SSH connection timed out:** Make sure OpenVPN is connected (`vpnuser04`) if you are off-campus.