# Kaggle smoke bundle

`kernel/run_smoke.py` is a standalone measured smoke job. It requires a private Kaggle Dataset containing the verified `spaq.tar` object (SHA-256 `d312242f...2194`) and canonical `manifest.parquet` (SHA-256 `abff8b1a...5800d`).

Authentication is intentionally external to the repository. After `kaggle auth login` succeeds, create/attach a private dataset with those two verified files, replace the two `KAGGLE_USERNAME` placeholders and dataset slug in `kernel-metadata.template.json`, save it as `kernel-metadata.json`, then validate with the installed CLI before pushing. Do not commit `kernel-metadata.json` if it exposes an account identifier the user does not want tracked.

The job records actual GPU identity/VRAM, disk space, selective extraction of 300 real images, 300-image inference time, 20 training steps, peak VRAM, checkpoint hash, and reload success in `/kaggle/working/smoke_summary.json`.
