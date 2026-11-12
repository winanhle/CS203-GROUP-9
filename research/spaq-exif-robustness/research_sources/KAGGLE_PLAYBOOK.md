# Kaggle execution & recovery playbook (source verified 2026-11-09)

## Source references
- Official Kaggle CLI kernel commands: https://github.com/Kaggle/kaggle-cli/blob/main/skills/references/kernels.md
- Official Kaggle CLI auth: https://github.com/Kaggle/kaggle-cli/blob/main/skills/references/auth.md
- Official Kaggle notebooks environment documentation: https://www.kaggle.com/docs/notebooks

## Runtime facts vs assumptions
Kaggle publicly documents 12 h session for CPU/GPU and 20 GB autosaved `/kaggle/working`, but actual quotas, GPU types, enabled internet, scratch storage and account permission **must be checked at runtime**. Our project budget is **8.5 h max target** for full experimental campaign, including data, smoke, training, evaluation, downloads, and slack. Reserve >=15–20%.

## 1. Check tools and credentials safely

```bash
python --version
python -m pip show kaggle || python -m pip install kaggle
kaggle --version
# Authenticate interactively if necessary; NEVER put tokens into logs or git.
kaggle auth login
# If OAuth unavailable, use documented token mechanism configured outside repo.
```

If logged in already, `kaggle auth login` may print account; prefer read-only test such as `kaggle kernels list -m`. Do not print `~/.kaggle/kaggle.json`, token file, environment secrets, or credential contents.

## 2. Package a reproducible Kaggle runner

- Keep source in main repository; create `kaggle/run_experiment.py` or notebook export and version-controlled configs.
- Do not silently assume Kaggle's interactive notebook process persists after Codex closes.
- Use `/kaggle/input` read-only datasets if attached. Write important artifacts to `/kaggle/working`.
- Use checksums to identify data and checkpoint versions.
- Use `kernel-metadata.json`, generated/validated with the currently installed CLI. Do not assume an undocumented metadata field.
- Pin code to git SHA; if runner needs files not in upload bundle, ensure they are explicitly included and present on Kaggle.
- Make the runner idempotent and checkpoint after run/epoch with an atomic write+rename.

## 3. CLI lifecycle (examples; validate `--help` on installed version)

```bash
kaggle kernels init -p kaggle/kernel
kaggle kernels push -p kaggle/kernel --accelerator gpu
kaggle kernels status <owner>/<slug>
kaggle kernels logs <owner>/<slug>
kaggle kernels output <owner>/<slug> -p artifacts/kaggle_download
```

`kaggle kernels push` creates a new version and normally triggers run. **Do not repeatedly push while an equivalent run is running.** Track owner/slug/version and status in `STATE.json`.

To follow logs as long as the CLI process is alive (does not itself persist future monitoring when Codex session ends):

```bash
kaggle kernels logs <owner>/<slug> --follow --interval 10
```

If the agent is interrupted, a future session should issue `kaggle kernels status` again; do not claim live background monitoring unless a live process or scheduled service actually exists.

## 4. Before full training — smoke test

Run a small job that prints/saves: Python, torch/torchvision, CUDA device name, VRAM peak, disk/RAM, dataset count and schema, 300-image feature IO/inference timing, 20 training steps, checkpoint write/resume, output download. Compute estimated total duration:

```
T_total = T_env + T_data + T_preprocess + T_extract_once
        + SUM_i(T_per_run_i * n_seeds_i)
        + T_evaluation + T_export + T_buffer
```

Use measured values with multiplier for variability, not a fabricated fixed minutes-per-epoch. If estimate exceeds 8.5h, freeze backbone/cache embeddings, reduce variants/seeds/epochs transparently, or split into documented sessions within user's free quota. Never drop test integrity to save compute.

## 5. Recovery when offline, quota limited, Kaggle fails

1. Fetch status, latest log and output manifest **before** resubmitting.
2. Keep previous run/version identifiers, configs, outputs; recover last valid checkpoint if accessible.
3. If a worker was interrupted, report `INTERRUPTED` and list last durable epoch/step; resume only if checkpoint truly accessible.
4. If authentication fails, record exact non-secret error; prepare rerunnable notebook/CLI script and mark `BLOCKED_ON_USER_AUTH`; continue offline tasks.
5. If internet blocked inside Kaggle, use appropriately licensed Kaggle Dataset attachments or pre-create a private dataset using authorized channels and approved terms; do not bypass restrictions.
6. Do not export massive images/weights by default. Export small reports/predictions/config/checkpoint necessary for reproducibility subject to licensing and disk limits.

## 6. Hard rule for source of truth

Run logs and downloaded artifacts establish actual results, not the CodeX narrative. Record source URL, run ID/version, status, artifact path, sha256 and metric command in evidence ledger.
