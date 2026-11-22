import argparse
import traceback
import shutil
import logging
import yaml
import sys
import os
import torch
import numpy as np

from guided_diffusion.diffusion import Diffusion

torch.set_printoptions(sci_mode=False)


def build_parser():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])

    parser.add_argument(
        "--config", type=str, default='lsun_bedroom_256.yml', help="Path to the config file"
        # e.g.,  lsun_bedroom_256.yml, lsun_cat_256.yml
    )
    parser.add_argument(
        "--model_ckpt", type=str, default='cd_bedroom256_lpips.pt', help="Name of the model checkpoint"
    )
    parser.add_argument("--seed", type=int, default=1234, help="Set different seeds for diverse results")
    parser.add_argument(
        "--exp", type=str, default="exp", help="Path for saving running related data."
    )
    parser.add_argument(
        "--deg", type=str, default='deblur_gauss', help="Degradation - sr_bicubic, deblur_gauss, inpainting"
    )
    parser.add_argument(
        "--operator_imp", type=str, default="SVD", help="SVD | FFT"
    )
    parser.add_argument(
        "--path_y",
        type=str,
        # required=True,
        default='lsun_bedroom',  # e.g.,  lsun_bedroom, lsun_cat
        help="Path of the test dataset.",
    )
    parser.add_argument(
        "--sigma_y", type=float, default=0.05, help="sigma_y"
    )
    parser.add_argument(
        "-i",
        "--image_folder",
        type=str,
        default="demo",
        help="The folder name of samples",
    )
    parser.add_argument(
        "--save_y", dest="save_observed_img", action="store_true"
    )
    parser.add_argument(
        "--deg_scale", type=float, default=4.0, help="deg_scale for SR Bicubic"
    )
    parser.add_argument(
        "--inpainting_mask_path", type=str, default="random_80_mask.npy",
        help="Path to mask NPY path for inpainting"
    )
    parser.add_argument(
        "--verbose",
        type=str,
        default="info",
        help="Verbose level: info | debug | warning | critical",
    )
    parser.add_argument(
        "--ni",
        action="store_false",
        help="No interaction. Suitable for Slurm Job launcher",
    )
    parser.add_argument(
        '--subset_start', type=int, default=-1
    )
    parser.add_argument(
        '--subset_end', type=int, default=-1
    )
    parser.add_argument(
        "--iN", type=int, default=250, help="iN hyperparameter"
    )
    parser.add_argument(
        "--gamma", type=float, default=0.7, help="Gamma hyperparameter"
    )
    parser.add_argument(
        "--eta", type=float, default=0.1, help="Eta hyperparameter"
    )
    parser.add_argument(
        "--zeta", type=float, default=0, help="Zeta hyperparameter"
    )
    parser.add_argument(
        "--deltas", type=str, default="", help="A comma separated list of the delta hyperparameters"
    )
    parser.add_argument(
        "--deltas_injection_type", type=int, default=0,
        help="Whether to inject deltas in the scaling calculation (1) or not (0)."
    )
    parser.add_argument(
        "--context_mode",
        type=str,
        choices=["none", "telea", "spatial", "clip", "poisson"],
        default="none",
        help="Context-aware inpainting guidance mode.",
    )
    parser.add_argument(
        "--telea_method",
        type=str,
        choices=["telea", "ns"],
        default="telea",
        help="OpenCV inpainting method for --context_mode telea.",
    )
    parser.add_argument(
        "--telea_radius",
        type=float,
        default=3.0,
        help="Inpainting neighborhood radius for --context_mode telea.",
    )
    parser.add_argument(
        "--spatial_sigma",
        type=float,
        default=8.0,
        help="Gaussian border feathering scale for --context_mode spatial.",
    )
    parser.add_argument(
        "--eta_hole",
        type=float,
        default=0.25,
        help="Stochasticity inside missing hole (mask == 0) for --context_mode spatial.",
    )
    parser.add_argument(
        "--eta_obs",
        type=float,
        default=0.02,
        help="Stochasticity on observed pixels (mask != 0) for --context_mode spatial.",
    )
    parser.add_argument(
        "--clip_model",
        type=str,
        default="ViT-B-32",
        help="OpenCLIP model architecture for --context_mode clip.",
    )
    parser.add_argument(
        "--clip_pretrained",
        type=str,
        default="laion2b_s34b_b79k",
        help="OpenCLIP pretrained tag for --context_mode clip.",
    )
    parser.add_argument(
        "--clip_prompt",
        type=str,
        default="a photograph of a cat",
        help="Target text prompt for --context_mode clip.",
    )
    parser.add_argument(
        "--clip_neg_prompt",
        type=str,
        default="a photograph of a red fire hydrant in outer space",
        help="Negative-control text prompt when --clip_negative_control is set.",
    )
    parser.add_argument(
        "--clip_negative_control",
        action="store_true",
        default=False,
        help="Use --clip_neg_prompt instead of --clip_prompt as a negative control.",
    )
    parser.add_argument(
        "--clip_scale",
        type=float,
        default=0.25,
        help="Null-space steering step scale for --context_mode clip.",
    )
    parser.add_argument(
        "--clip_steps",
        type=int,
        default=1,
        help="Number of null-space CLIP steering steps per diffusion iteration.",
    )
    parser.add_argument(
        "--sample_ids",
        nargs="+",
        type=str,
        default=None,
        help="Optional 0-based sample indices (comma-separated or space-separated).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Override config.data.num_workers for DataLoader.",
    )
    return parser


def parse_cli_args(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.telea_radius <= 0:
        parser.error("--telea_radius must be > 0")
    if args.spatial_sigma <= 0:
        parser.error("--spatial_sigma must be > 0")
    if not (0.0 <= args.eta_hole <= 1.0):
        parser.error("--eta_hole must be in [0, 1]")
    if not (0.0 <= args.eta_obs <= 1.0):
        parser.error("--eta_obs must be in [0, 1]")
    if args.clip_scale < 0:
        parser.error("--clip_scale must be >= 0")
    if args.clip_steps < 0:
        parser.error("--clip_steps must be >= 0")
    if args.num_workers is not None and args.num_workers < 0:
        parser.error("--num_workers must be >= 0")

    if args.sample_ids is not None:
        parsed_ids = []
        for item in args.sample_ids:
            for part in item.split(","):
                tok = part.strip()
                if not tok:
                    parser.error("--sample_ids cannot contain empty entries")
                try:
                    val = int(tok)
                except ValueError:
                    parser.error(f"Invalid integer in --sample_ids: {tok!r}")
                if val < 0:
                    parser.error("--sample_ids entries must be >= 0")
                parsed_ids.append(val)
        if not parsed_ids:
            parser.error("--sample_ids cannot be empty")
        if args.subset_start >= 0 or args.subset_end > 0:
            parser.error("--sample_ids cannot be combined with --subset_start/--subset_end")
        args.sample_ids = parsed_ids

    return args


def parse_args_and_config():
    args = parse_cli_args()

    # parse config file
    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)
    if args.num_workers is not None:
        new_config.data.num_workers = args.num_workers

    level = getattr(logging, args.verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError("level {} not supported".format(args.verbose))

    handler1 = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
    )
    handler1.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.setLevel(level)

    os.makedirs(os.path.join(args.exp, "image_samples"), exist_ok=True)
    args.image_folder = os.path.join(
        args.exp, "image_samples", args.image_folder
    )
    if not os.path.exists(args.image_folder):
        os.makedirs(args.image_folder)
    else:
        overwrite = False
        if args.ni:
            overwrite = True
        else:
            response = input(
                f"Image folder {args.image_folder} already exists. Overwrite? (Y/N)"
            )
            if response.upper() == "Y":
                overwrite = True

        if overwrite:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        else:
            print("Output image folder exists. Program halted.")
            sys.exit(0)

    log_path = os.path.join(args.image_folder, '0_logs.log')
    fh = logging.FileHandler(log_path)  # , mode='a')
    fh.setFormatter(formatter)
    logger.setLevel(level)
    logger.addHandler(fh)
    for arg, value in sorted(vars(args).items()):
        logger.info("Argument %s: %r", arg, value)

    # add device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logging.info("Using device: {}".format(device))
    new_config.device = device

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, new_config, logger


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    args, config, logger = parse_args_and_config()

    try:
        runner = Diffusion(args, config)
        runner.sample(logger)
    except Exception:
        logging.error(traceback.format_exc())
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
