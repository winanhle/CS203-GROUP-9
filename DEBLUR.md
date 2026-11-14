# CM4IR Gaussian deblurring

This branch starts from Quỳnh Anh's `origin/codex/anh-core` at `3d09c11`. The implementation uses the author UNet, SVD blur operator and four-step CM4IR sampler. The sampler starts from a regularized inverse of the blurred image and uses the pretrained denoiser, measurement consistency and noise injection during restoration.

- [Garber and Tirer, CVPR 2025, Algorithm 1 and Appendix B](https://arxiv.org/html/2412.20596)
- [Author implementation](https://github.com/tirer-lab/CM4IR)

## Setup

Use Python 3.11 or newer with CUDA-enabled PyTorch and matching torchvision. Run from the repository root. The commands below work with Bash and PowerShell.

```sh
python -m pip install -r requirements-deblur.txt
python -c "import torch; print(torch.cuda.is_available())"
```

Create `exp/logs/lsun_bedroom` and download the [official Bedroom checkpoint](https://openaipublic.blob.core.windows.net/consistency/cd_bedroom256_lpips.pt) to `exp/logs/lsun_bedroom/cd_bedroom256_lpips.pt` (2,105,395,349 bytes). This is a one-time download. On servers with no Internet, obtain it on the client before transferring it. On Slurm use an allocated GPU.

```sh
curl --fail --location --retry 3 --continue-at - --create-dirs --output exp/logs/lsun_bedroom/cd_bedroom256_lpips.pt https://openaipublic.blob.core.windows.net/consistency/cd_bedroom256_lpips.pt
```

On Windows PowerShell, use `curl.exe`.

On the prepared Windows machine, run these commands from this branch's folder to reuse the installed CUDA environment and the locally cached LPIPS assets:

```powershell
$env:PYTHONPATH=(Resolve-Path 'exp/logs/python-deps').Path
$env:TORCH_HOME=(Resolve-Path 'exp/logs').Path
$cm4irPython='../spaq-exif-robustness/.venv312/Scripts/python.exe'
& $cm4irPython scripts/deblur.py --limit 1 --lpips
```

## Run

```sh
python scripts/deblur.py --limit 1
python scripts/deblur.py --preset paper --lpips
python scripts/deblur.py --observed "path/to/blurred-256.png"
```

The first command degrades one clean sample using the author operator, then restores it. The second uses all available Bedroom samples and adds VGG-LPIPS, which may fetch torchvision VGG16 weights on first use. The checkout includes eight Bedroom samples. The paper benchmark uses 300 validation images. Compare paper scores only with matching evaluation data and settings.

For `--observed`, supply an already blurred 256×256 image. This sends its pixels directly to the solver. The solver assumes the released 5-tap separable Gaussian operator and interprets `--noise` as the assumed measurement noise in [0,1]. Results on motion blur, defocus, unknown kernels, and arbitrary scenes depend on how well that assumption and the Bedroom prior fit the image. Ground-truth metrics are omitted for this mode.

Outputs are printed by the runner and written to a fresh `exp/image_samples/deblur-*/` directory. For image 0, `0_0.png` is restored, `Apy/y_0.png` is the blurred observation, `Apy/Apy_0.png` is the regularized initialization, and `Apy/orig_0.png` is the ground truth when available. The directory contains `0_logs.log` and `run.json`, with input and checkpoint hashes and paired PSNR measured on the saved PNGs. A null PSNR denotes a perfect pixel match.

## Presets

| Preset | Noise σ | iN | gamma | zeta |
| --- | ---: | ---: | ---: | ---: |
| repo | 0.025 | 75 | 0.01 | 3 |
| repo | 0.05 | 150 | 0.05 | 1 |
| paper | 0.025 | 90 | 0.02 | 3 |
| paper | 0.05 | 160 | 0.07 | 1.5 |

Both presets use eta 0.1 and all-zero deltas. `repo` follows the executable commands in `evaluate_CM4IR.sh`. `paper` follows arXiv v2 Table 5. Table 2 reports Bedroom Gaussian deblurring PSNR/VGG-LPIPS of 29.00 dB/0.213 at noise 0.025 and 27.42 dB/0.258 at noise 0.05.

## Local verification

On 2026-11-14 the RTX 4050 Laptop GPU completed both presets at noise 0.025 with seed 1234 and the same eight checked-in Bedroom inputs. Mean PSNR on the saved blurred images was 23.24 dB. The repo preset produced 29.48 dB and VGG-LPIPS 0.2200. The paper preset produced 29.51 dB and VGG-LPIPS 0.2104. A separate run read the first saved blurred PNG through `--observed` and produced a valid 256×256 restored PNG. These measurements cover the eight-image subset and one known-kernel observed input. Checkpoint SHA-256: `61238b8de0ba4976ec35a60fd4786a02a862df0960aa8a2554467e9486e71252`.
