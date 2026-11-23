from functools import partial
import os
import random

import lpips
import numpy as np
import torch
import torch.utils.data as data
import torchvision.utils as tvu
import tqdm

from datasets import get_dataset, data_transform, inverse_data_transform
from functions.cm4ir_scheme import cm4ir_restoration
from functions.context_guidance import build_context_guidance, telea_inpainting
from functions.harmonic_poisson import harmonic_inpainting, saliency_bridge_inpainting
from functions.median_inpainting import median_inpainting
from guided_diffusion.script_util import create_model

loss_fn_vgg = lpips.LPIPS(net='vgg')
_CACHED_UNET_MODELS = {}


def seed_worker(worker_id, seed):
    worker_seed = seed % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
                np.linspace(
                    beta_start ** 0.5,
                    beta_end ** 0.5,
                    num_diffusion_timesteps,
                    dtype=np.float64,
                )
                ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        self.betas = torch.from_numpy(betas).float().to(self.device)
        self.last_run_records = []

    def sample(self, logger, on_result=None):

        config_dict = vars(self.config.model)
        config_dict.update({"image_size": self.config.data.image_size})
        ckpt = os.path.join(self.args.exp, f"logs/{self.args.model_ckpt}")
        if not os.path.exists(ckpt):
            raise ValueError(f"Model ckpt not found in: {ckpt}. Please refer to https://github.com/tirer-lab/CM4IR.git"
                             f"to see where the models that were used in the paper can be found.")

        cache_key = (os.path.abspath(ckpt), str(self.device), bool(self.config.model.use_fp16))
        if getattr(self.args, "cache_model", False) and cache_key in _CACHED_UNET_MODELS:
            model = _CACHED_UNET_MODELS[cache_key]
        else:
            model = create_model(**config_dict)
            model.load_state_dict(torch.load(ckpt, map_location=self.device))
            model.to(self.device)
            if self.config.model.use_fp16:
                model.convert_to_fp16()
            model.eval()
            if getattr(self.args, "cache_model", False):
                _CACHED_UNET_MODELS[cache_key] = model
        sample_ids = getattr(self.args, "sample_ids", None)
        if sample_ids is None and self.device.type == "cuda":
            model = torch.nn.DataParallel(model)

        print('Running CM4IR.',
              f'Dataset: {self.config.data.dataset}',
              f'Task: {self.args.deg}.',
              f'Noise level: {self.args.sigma_y}.'
              )
        self.cm4ir_wrapper(model, logger, on_result=on_result)

    def cm4ir_wrapper(self, model, logger, on_result=None):
        self.last_run_records = []
        args, config = self.args, self.config
        context_mode = getattr(args, "context_mode", "none")
        if context_mode not in (
            "none",
            "telea",
            "spatial",
            "clip",
            "poisson",
            "poisson_bridge",
            "poisson_bridge_sharp",
            "poisson_bridge_freq",
            "poisson_bridge_feather",
        ):
            raise ValueError(f"Unsupported context_mode: {context_mode!r}")
        if context_mode != "none" and "inp" not in args.deg:
            raise ValueError(f"context_mode={context_mode!r} is only supported for inpainting tasks, got deg={args.deg!r}")

        sample_ids = getattr(args, "sample_ids", None)
        if isinstance(sample_ids, str):
            tokens = [tok.strip() for tok in sample_ids.split(",") if tok.strip()]
            sample_ids = [int(tok) for tok in tokens]
        elif sample_ids is not None:
            sample_ids = [int(x) for x in sample_ids]

        dataset, test_dataset = get_dataset(args, config)

        subset_start = getattr(args, "subset_start", -1)
        subset_end = getattr(args, "subset_end", -1)
        if sample_ids is not None:
            if len(sample_ids) == 0:
                raise ValueError("sample_ids cannot be empty.")
            if subset_start >= 0 or subset_end > 0:
                raise ValueError("sample_ids cannot be combined with subset_start/subset_end.")
            dataset_len = len(test_dataset)
            for idx in sample_ids:
                if idx < 0 or idx >= dataset_len:
                    raise ValueError(f"sample_id {idx} out of range [0, {dataset_len - 1}].")
            test_dataset = torch.utils.data.Subset(test_dataset, sample_ids)
            idx_init = 0
            idx_so_far = 0
        elif subset_start >= 0 and subset_end > 0:
            assert subset_end > subset_start
            test_dataset = torch.utils.data.Subset(test_dataset, range(subset_start, subset_end))
            idx_init = subset_start
            idx_so_far = subset_start
        else:
            idx_init = 0
            idx_so_far = 0

        print(f'Dataset has size {len(test_dataset)}')

        num_workers = getattr(args, "num_workers", None)
        if num_workers is None:
            num_workers = config.data.num_workers
            if os.name == "nt" and num_workers > 0:
                if logger is not None:
                    logger.info("Clamping config.data.num_workers from %d to 0 on Windows.", num_workers)
                num_workers = 0

        g = torch.Generator()
        g.manual_seed(args.seed)
        val_loader = data.DataLoader(
            test_dataset,
            batch_size=config.sampling.batch_size,
            num_workers=num_workers,
            worker_init_fn=partial(seed_worker, seed=args.seed),
            generator=g,
        )

        # Set deltas
        if args.deltas == "":
            deltas = [0] * config.sampling.T_sampling
        else:
            deltas = [float(x.strip()) for x in args.deltas.split(',')]
            if len(deltas) != config.sampling.T_sampling:
                raise ValueError(
                    f"deltas length must be equal to T_sampling. Got {len(deltas)} deltas, but {config.sampling.T_sampling} sampling steps.")

        # get degradation matrix
        deg = args.deg
        if deg == 'cs_walshhadamard':
            compress_by = round(1 / args.deg_scale)
            from functions.svd_operators import WalshHadamardCS
            A_funcs = WalshHadamardCS(config.data.channels, self.config.data.image_size, compress_by,
                                      torch.randperm(self.config.data.image_size ** 2, device=self.device), self.device)
        elif deg == 'cs_blockbased':
            cs_ratio = args.deg_scale
            from functions.svd_operators import CS
            A_funcs = CS(config.data.channels, self.config.data.image_size, cs_ratio, self.device)
        elif deg == 'inpainting':
            from functions.svd_operators import Inpainting
            if os.path.isabs(args.inpainting_mask_path) or os.path.exists(args.inpainting_mask_path):
                inpainting_mask_full_path = args.inpainting_mask_path
            else:
                inpainting_mask_full_path = os.path.join(args.exp, "inpainting_masks", args.inpainting_mask_path)
            if not os.path.exists(inpainting_mask_full_path):
                raise ValueError(f"Could not find inpainting mask in path: {inpainting_mask_full_path}."
                                 f"Please set the correct path using the --inpainting_mask_path flag.")
            loaded = np.load(inpainting_mask_full_path)
            mask = torch.from_numpy(loaded).to(self.device).reshape(-1)
            missing_r = torch.nonzero(mask == 0).long().reshape(-1) * 3
            missing_g = missing_r + 1
            missing_b = missing_g + 1
            missing = torch.cat([missing_r, missing_g, missing_b], dim=0)
            A_funcs = Inpainting(config.data.channels, config.data.image_size, missing, self.device)
        elif deg == 'denoising':
            from functions.svd_operators import Denoising
            A_funcs = Denoising(config.data.channels, self.config.data.image_size, self.device)
        elif deg == 'colorization':
            from functions.svd_operators import Colorization
            A_funcs = Colorization(config.data.image_size, self.device)
        elif deg == 'sr_averagepooling':
            blur_by = int(args.deg_scale)
            if args.operator_imp == 'SVD':
                from functions.svd_operators import SuperResolution
                A_funcs = SuperResolution(config.data.channels, config.data.image_size, blur_by, self.device)
            else:
                raise NotImplementedError()
        elif deg == 'sr_bicubic':
            factor = int(args.deg_scale)

            def bicubic_kernel(x, a=-0.5):
                if abs(x) <= 1:
                    return (a + 2) * abs(x) ** 3 - (a + 3) * abs(x) ** 2 + 1
                elif 1 < abs(x) and abs(x) < 2:
                    return a * abs(x) ** 3 - 5 * a * abs(x) ** 2 + 8 * a * abs(x) - 4 * a
                else:
                    return 0

            k = np.zeros((factor * 4))
            for i in range(factor * 4):
                x = (1 / factor) * (i - np.floor(factor * 4 / 2) + 0.5)
                k[i] = bicubic_kernel(x)
            k = k / np.sum(k)
            kernel = torch.from_numpy(k).float().to(self.device)

            if args.operator_imp == 'SVD':
                from functions.svd_operators import SRConv
                A_funcs = SRConv(kernel / kernel.sum(), config.data.channels, self.config.data.image_size, self.device,
                                 stride=factor)
            elif args.operator_imp == 'FFT':
                from functions.fft_operators import Superres_fft, prepare_cubic_filter
                k = prepare_cubic_filter(1 / factor)
                kernel = torch.from_numpy(k).float().to(self.device)
                A_funcs = Superres_fft(kernel / kernel.sum(), config.data.channels, self.config.data.image_size,
                                       self.device, stride=factor)
            else:
                raise NotImplementedError()

        elif deg == 'deblur_uni':
            if args.operator_imp == 'SVD':
                from functions.svd_operators import Deblurring
                A_funcs = Deblurring(torch.Tensor([1 / 9] * 9).to(self.device), config.data.channels,
                                     self.config.data.image_size, self.device)
            elif args.operator_imp == 'FFT':
                from functions.fft_operators import Deblurring_fft
                A_funcs = Deblurring_fft(torch.Tensor([1 / 9] * 9).to(self.device), config.data.channels,
                                         self.config.data.image_size, self.device)
            else:
                raise NotImplementedError()

        elif deg == 'deblur_gauss':
            sigma = 10  # better make argument for kernel type
            pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x / sigma) ** 2]))
            kernel = torch.Tensor([pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2)]).to(
                self.device)  # clip it as in DDRM/DDNM code, but it makes more sense to use lower sigma with the line below
            # kernel = torch.Tensor([pdf(ii) for ii in range(-30,31,1)]).to(self.device)
            if args.operator_imp == 'SVD':
                from functions.svd_operators import Deblurring
                A_funcs = Deblurring(kernel / kernel.sum(), config.data.channels, self.config.data.image_size,
                                     self.device)
            elif args.operator_imp == 'FFT':
                from functions.fft_operators import Deblurring_fft
                A_funcs = Deblurring_fft(kernel / kernel.sum(), config.data.channels, self.config.data.image_size,
                                         self.device)
            else:
                raise NotImplementedError()

        elif deg == 'deblur_aniso':
            sigma = 20
            pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x / sigma) ** 2]))
            kernel2 = torch.Tensor([pdf(-4), pdf(-3), pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2), pdf(3), pdf(4)]).to(
                self.device)
            sigma = 1
            pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x / sigma) ** 2]))
            kernel1 = torch.Tensor([pdf(-4), pdf(-3), pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2), pdf(3), pdf(4)]).to(
                self.device)

            if args.operator_imp == 'SVD':
                from functions.svd_operators import Deblurring2D
                A_funcs = Deblurring2D(kernel1 / kernel1.sum(), kernel2 / kernel2.sum(), config.data.channels,
                                       self.config.data.image_size, self.device)
            elif args.operator_imp == 'FFT':
                # unlike when using 'SVD' mode, here you can implement any 2D kernel that you want (not just seperable kernels)
                from functions.fft_operators import Deblurring_fft
                kernel = torch.matmul(kernel1[:, None], kernel2[None, :])
                A_funcs = Deblurring_fft(kernel / kernel.sum(), config.data.channels, self.config.data.image_size,
                                         self.device)
            else:
                raise NotImplementedError()
        else:
            raise ValueError("degradation type not supported")

        if context_mode in (
            "spatial",
            "clip",
            "poisson",
            "poisson_bridge",
            "poisson_bridge_sharp",
            "poisson_bridge_freq",
            "poisson_bridge_feather",
        ):
            context_guidance = build_context_guidance(
                args=args,
                mask=mask,
                image_size=config.data.image_size,
                device=self.device,
            )
        else:
            context_guidance = None

        sigma_y = 2 * args.sigma_y  # to account for scaling to [-1,1]
        os.makedirs(self.args.image_folder, exist_ok=True)

        print(f'Start from {idx_init}')
        avg_psnr = 0.0
        avg_lpips = 0.0
        pbar = tqdm.tqdm(val_loader)

        img_ind = -1

        for x_orig, classes in pbar:

            b_cur = x_orig.shape[0]
            batch_indices = [
                sample_ids[idx_so_far - idx_init + i] if sample_ids is not None else idx_so_far + i
                for i in range(b_cur)
            ]
            output_index = batch_indices[0]
            sample_seeds = [args.seed + idx for idx in batch_indices]

            x_orig = x_orig.to(self.device)
            x_orig = data_transform(self.config, x_orig)

            y = A_funcs.A(x_orig)

            y_noise = torch.cat(
                [
                    torch.randn(
                        y[i : i + 1].shape,
                        device=y.device,
                        dtype=y.dtype,
                        generator=torch.Generator(device=y.device).manual_seed(sample_seeds[i]),
                    )
                    for i in range(b_cur)
                ],
                dim=0,
            )
            y = y + sigma_y * y_noise  # added noise to measurement

            b, hwc = y.size()
            if 'color' in deg:
                hw = hwc / 1
                h = w = int(hw ** 0.5)
                y = y.reshape((b, 1, h, w))
            elif 'inp' in deg or 'cs' in deg:
                pass
            else:
                hw = hwc / 3
                h = w = int(hw ** 0.5)
                y = y.reshape((b, 3, h, w))

            y = y.reshape((b, hwc))

            Apy = A_funcs.A_pinv_add_eta(y, sigma_y ** 2 * args.zeta).view(y.shape[0],
                                                                           config.data.channels,
                                                                           self.config.data.image_size,
                                                                           self.config.data.image_size)
            if getattr(args, "save_observed_img", False):
                os.makedirs(os.path.join(self.args.image_folder, "Apy"), exist_ok=True)
                for i in range(len(Apy)):
                    tvu.save_image(
                        inverse_data_transform(config, Apy[i]),
                        os.path.join(self.args.image_folder, f"Apy/Apy_{output_index + i}.png")
                    )
                    tvu.save_image(
                        inverse_data_transform(config, x_orig[i]),
                        os.path.join(self.args.image_folder, f"Apy/orig_{output_index + i}.png")
                    )
                    if 'inp' in deg or 'cs' in deg:
                        pass
                    else:
                        tvu.save_image(
                            inverse_data_transform(config, y[i].reshape((3, h, w))),
                            os.path.join(self.args.image_folder, f"Apy/y_{output_index + i}.png")
                        )
            if 'inp' in args.deg:
                if context_mode == "telea":
                    x = telea_inpainting(
                        Apy,
                        mask,
                        self.config.data.channels,
                        self.config.data.image_size,
                        method=getattr(args, "telea_method", "telea"),
                        radius=getattr(args, "telea_radius", 3.0),
                    ).view(
                        y.shape[0], config.data.channels,
                        self.config.data.image_size,
                        self.config.data.image_size,
                    )
                elif context_mode == "poisson":
                    x = harmonic_inpainting(
                        Apy,
                        mask,
                        self.config.data.channels,
                        self.config.data.image_size,
                    ).view(
                        y.shape[0], config.data.channels,
                        self.config.data.image_size,
                        self.config.data.image_size,
                    )
                elif context_mode in ("poisson_bridge", "poisson_bridge_sharp", "poisson_bridge_freq", "poisson_bridge_feather"):
                    x = saliency_bridge_inpainting(
                        Apy,
                        mask,
                        self.config.data.channels,
                        self.config.data.image_size,
                        gamma=5.0,
                    ).view(
                        y.shape[0], config.data.channels,
                        self.config.data.image_size,
                        self.config.data.image_size,
                    )
                else:
                    x = median_inpainting(
                        Apy,
                        mask,
                        self.config.data.channels,
                        self.config.data.image_size,
                    ).view(
                        y.shape[0], config.data.channels,
                        self.config.data.image_size,
                        self.config.data.image_size,
                    )
            else:
                x = Apy.reshape(
                    (
                        y.shape[0],
                        config.data.channels,
                        config.data.image_size,
                        config.data.image_size,
                    )
                ).to(device=self.device)

            with torch.no_grad():
                cm4ir_kwargs = dict(
                    x=x,
                    model=model,
                    A_funcs=A_funcs,
                    y_0=y,
                    sigma_y=sigma_y,
                    betas=self.betas,
                    eta=args.eta,
                    deltas=deltas,
                    deltas_injection_type=args.deltas_injection_type,
                    gamma=args.gamma,
                    iN=args.iN,
                    zeta=args.zeta,
                    classes=classes if self.config.model.class_cond else None,
                    config=config,
                    sample_seeds=sample_seeds,
                )
                if context_guidance is not None:
                    cm4ir_kwargs["context_guidance"] = context_guidance
                x = cm4ir_restoration(**cm4ir_kwargs)

            if not torch.isfinite(x[0]).all():
                raise RuntimeError(f"Non-finite output tensor encountered at output_index={output_index}.")

            x_m11_batch = x[0]
            with torch.no_grad():
                lpips_batch = loss_fn_vgg(x_m11_batch, x_orig.to('cpu')).reshape(-1)

            x = [inverse_data_transform(config, xi) for xi in x]

            for j in range(x[0].size(0)):
                sample_idx = output_index + j
                restored_path = os.path.join(self.args.image_folder, f"{sample_idx}_{0}.png")
                tvu.save_image(x[0][j], restored_path)
                orig = inverse_data_transform(config, x_orig[j])
                x_hat_01_dev = x[0][j].to(self.device)
                mse = torch.mean((x_hat_01_dev - orig) ** 2)
                psnr = 10 * torch.log10(1 / mse) if mse > 0 else torch.tensor(float("inf"), device=self.device)
                full_psnr_db = float(psnr.item()) if mse > 0 else None
                full_vgg_lpips = float(lpips_batch[j].item())
                if logger is not None:
                    logger.info("img_ind: %d, PSNR: %.2f, LPIPS: %.4f" % (img_ind, psnr, full_vgg_lpips))
                avg_psnr += psnr
                avg_lpips += full_vgg_lpips

                if 'inp' in deg:
                    hole_mask_2d = (mask.view(config.data.image_size, config.data.image_size) == 0)
                    obs_mask_2d = ~hole_mask_2d
                    if hole_mask_2d.any():
                        hole_mse = torch.mean(((x_hat_01_dev - orig)[:, hole_mask_2d]) ** 2)
                        hole_psnr_db = float((10.0 * torch.log10(1.0 / hole_mse)).item()) if hole_mse > 0 else None
                    else:
                        hole_psnr_db = None

                    x_orig_m11_cpu = x_orig[j].detach().cpu()
                    x_hat_m11_cpu = x_m11_batch[j].detach().cpu()
                    x_hole_only_m11 = x_hat_m11_cpu.clone()
                    x_hole_only_m11[:, obs_mask_2d.cpu()] = x_orig_m11_cpu[:, obs_mask_2d.cpu()]
                    with torch.no_grad():
                        hole_vgg_lpips = float(
                            loss_fn_vgg(x_hole_only_m11.unsqueeze(0), x_orig_m11_cpu.unsqueeze(0)).item()
                        )
                    observed_max_abs_diff = float(
                        (A_funcs.A(x_hat_m11_cpu.unsqueeze(0).to(self.device)) - y[j:j + 1]).abs().max().item()
                    )
                else:
                    hole_psnr_db = None
                    hole_vgg_lpips = None
                    observed_max_abs_diff = None

                rec_entry = {
                    "index": int(sample_idx),
                    "x_orig_01": orig.detach().cpu(),
                    "x_hat_01": x[0][j].detach().cpu(),
                    "x_orig_m11": x_orig[j].detach().cpu(),
                    "x_hat_m11": x_m11_batch[j].detach().cpu(),
                    "y_obs": y[j].detach().cpu(),
                    "Apy_01": inverse_data_transform(config, Apy[j]).detach().cpu(),
                    "full_psnr_db": full_psnr_db,
                    "full_vgg_lpips": full_vgg_lpips,
                    "hole_psnr_db": hole_psnr_db,
                    "hole_vgg_lpips": hole_vgg_lpips,
                    "observed_max_abs_diff": observed_max_abs_diff,
                    "restored_path": restored_path,
                }
                self.last_run_records.append(rec_entry)
                if on_result is not None:
                    on_result(int(sample_idx), restored_path, **{k: v for k, v in rec_entry.items() if k not in ("index", "restored_path")})

            idx_so_far += y.shape[0]

            if logger is not None:
                logger.info("Avg PSNR: %.2f, Avg LPIPS: %.4f     (** After %d iterations **)" % (
                    avg_psnr / (idx_so_far - idx_init), avg_lpips / (idx_so_far - idx_init), idx_so_far - idx_init))

        count = idx_so_far - idx_init
        if count == 0:
            raise RuntimeError("No samples were processed.")
        avg_psnr = avg_psnr / count
        avg_lpips = avg_lpips / count
        print("Total Average PSNR: %.2f" % avg_psnr)
        print("Total Average LPIPS: %.4f" % avg_lpips)
        print("Number of samples: %d" % count)
