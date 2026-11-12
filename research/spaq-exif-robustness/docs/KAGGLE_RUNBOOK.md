# Kaggle GPU smoke runbook

## Current state

OAuth is established for `longltrnhong`. Private dataset `longltrnhong/spaq-compact-verified` is ready, and kernel `longltrnhong/spaq-exif-robustness-smoke` version 2 completed on Tesla T4. Outputs are retained under ignored `artifacts/kaggle_download/v2`; tracked results and hashes are in `evidence/kaggle_smoke_v2.json`.

## Re-authentication if credentials expire

```powershell
.\.venv\Scripts\kaggle.exe auth login
.\.venv\Scripts\kaggle.exe account
```

Do not paste credentials into tracked files or logs.

## Private input dataset

Create a private Kaggle Dataset containing only:

- `spaq.tar`: 2,626,897,920 bytes, SHA-256 `d312242f10af983d5ceaac433c76b7c2b3473e2a2f87782f9db8ee1537dc2194`.
- `manifest.parquet`: SHA-256 `abff8b1a5f7d2e74d0c8874655fbe40d5e75de75bf61e65753c7085cee15800d`.

Do not upload the 34.8 GB author archive; it is unnecessary and exceeds the documented saved-output budget when extracted.

## Configure and execute

Copy `kaggle/kernel/kernel-metadata.template.json` to ignored `kaggle/kernel/kernel-metadata.json`. Replace username/kernel slug and private dataset source, then enable a GPU.

```powershell
.\.venv\Scripts\kaggle.exe kernels push -p kaggle\kernel
.\.venv\Scripts\kaggle.exe kernels status YOUR_USERNAME/YOUR_KERNEL_SLUG
```

After status is `complete`, retrieve real outputs:

```powershell
New-Item -ItemType Directory -Force artifacts\kaggle_download\v1
.\.venv\Scripts\kaggle.exe kernels output YOUR_USERNAME/YOUR_KERNEL_SLUG -p artifacts\kaggle_download\v1
Get-FileHash -Algorithm SHA256 artifacts\kaggle_download\v1\*
```

`smoke_summary.json` must record a real Kaggle GPU/VRAM, disk space, verified inputs, 300-image extraction/inference, 20 training steps, peak VRAM, checkpoint hash, reload/resume success, and timing. Record the real kernel/version and hashes in all ledgers. Never substitute local RTX measurements.
