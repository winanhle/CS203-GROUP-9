"""Run CM4IR Gaussian deblurring with LSUN Bedroom presets."""

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PRESETS = {
    ("repo", "0.025"): ("75", "0.01", "3"),
    ("repo", "0.05"): ("150", "0.05", "1"),
    ("paper", "0.025"): ("90", "0.02", "3"),
    ("paper", "0.05"): ("160", "0.07", "1.5"),
}
CHECKPOINT = Path("lsun_bedroom/cd_bedroom256_lpips.pt")


def image_pixels(path):
    with Image.open(path) as image:
        if image.size != (256, 256) or image.mode != "RGB":
            raise RuntimeError(f"Unexpected image size or mode: {path}")
        return np.asarray(image, dtype=np.float64) / 255


def psnr(reference, prediction):
    mse = np.mean((reference - prediction) ** 2)
    return float(10 * np.log10(1 / mse)) if mse else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=("repo", "paper"), default="repo")
    parser.add_argument("--noise", choices=("0.025", "0.05"), default="0.025")
    parser.add_argument("--observed", type=Path, help="Already blurred 256x256 RGB image")
    parser.add_argument("--limit", type=int, help="Number of sorted validation inputs (default: all)")
    parser.add_argument("--lpips", action="store_true", help="Also evaluate with VGG-LPIPS")
    args = parser.parse_args()
    if args.observed and (args.limit is not None or args.lpips):
        parser.error("--observed has no validation subset or ground-truth LPIPS")

    import torch
    if not torch.cuda.is_available():
        parser.error("Install a CUDA-enabled PyTorch build and use an allocated GPU")

    ckpt = ROOT / "exp/logs" / CHECKPOINT
    if not ckpt.is_file() or ckpt.stat().st_size != 2_105_395_349:
        parser.error(f"Official checkpoint missing or incomplete: {ckpt}")

    run = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    run = f"deblur-{args.preset}-{args.noise}-{run}"
    start, gamma, zeta = PRESETS[args.preset, args.noise]
    command = [
        sys.executable, "main.py", "--config", "lsun_bedroom_256.yml",
        "--path_y", "lsun_bedroom", "--deg", "deblur_gauss",
        "--model_ckpt", CHECKPOINT.as_posix(), "--sigma_y", args.noise,
        "--iN", start, "--gamma", gamma, "--zeta", zeta, "--eta", "0.1",
        "--workers", "0", "--save_y", "-i", run,
    ]
    if not args.lpips:
        command.append("--skip-lpips")
    if args.observed:
        inputs = [args.observed.resolve(strict=True)]
        command.extend(("--observed", str(inputs[0])))
    else:
        from torchvision.datasets import ImageFolder
        samples = ImageFolder(ROOT / "exp/datasets/lsun_bedroom").samples
        count = args.limit if args.limit is not None else len(samples)
        if not 1 <= count <= len(samples):
            parser.error(f"--limit must be between 1 and {len(samples)}")
        command.extend(("--subset_start", "0", "--subset_end", str(count)))
        inputs = [Path(path) for path, _ in samples[:count]]
        print("Inputs:", ", ".join(path.name for path in inputs), flush=True)

    output = ROOT / "exp/image_samples" / run
    print(f"Output: {output}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    rows = []
    for index, path in enumerate(inputs):
        restored_file = f"{index}_0.png"
        restored = image_pixels(output / restored_file)
        blurred = image_pixels(output / "Apy" / f"y_{index}.png")
        row = {
            "input": path.name,
            "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "restored": restored_file,
        }
        if not args.observed:
            reference = image_pixels(output / "Apy" / f"orig_{index}.png")
            row["blurred_psnr_db"] = psnr(reference, blurred)
            row["restored_psnr_db"] = psnr(reference, restored)
        rows.append(row)
    record = {
        "command": command,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "samples": rows,
    }
    with ckpt.open("rb") as checkpoint:
        record["checkpoint_sha256"] = hashlib.file_digest(checkpoint, "sha256").hexdigest()
    (output / "run.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"Verified {len(rows)} restored PNGs in {output}")


if __name__ == "__main__":
    main()
