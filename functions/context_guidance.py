import math
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F


_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

_CLIP_MODEL_CACHE: Dict[Tuple[str, str, str], Tuple[torch.nn.Module, Any]] = {}
_CLIP_TEXT_CACHE: Dict[Tuple[str, str, str, str], torch.Tensor] = {}


def _normalize_mask_2d(
    mask: Union[torch.Tensor, np.ndarray],
    image_size: int,
) -> torch.Tensor:
    """Normalize a mask to a 2D boolean missing-pixel mask `(H, W)`.

    Treats `mask == 0` as missing (`True`) and ALL `mask != 0` as observed
    (`False`), which is required for anti-aliased `uint8` masks such as
    `lorem_ipsum_mask.npy` containing values `0..255`.
    """
    if isinstance(mask, np.ndarray):
        mask_t = torch.from_numpy(mask)
    elif isinstance(mask, torch.Tensor):
        mask_t = mask
    else:
        raise TypeError(f"Unsupported mask type: {type(mask)}")

    expected = image_size * image_size
    if mask_t.numel() != expected:
        raise ValueError(
            f"Mask has {mask_t.numel()} elements, expected {expected} ({image_size}x{image_size})."
        )

    mask_2d = mask_t.reshape(image_size, image_size)
    return mask_2d == 0


def telea_inpainting(
    y: torch.Tensor,
    mask: Union[torch.Tensor, np.ndarray],
    in_channels: int,
    image_size: int,
    method: str = "telea",
    radius: float = 3.0,
) -> torch.Tensor:
    """Initialize missing pixels (`mask == 0`) using OpenCV PDE inpainting.

    Supports `B >= 1`, `in_channels in (1, 3)`, and `"telea"` (`cv2.INPAINT_TELEA`)
    or `"ns"` (`cv2.INPAINT_NS`). Observed pixels (`mask != 0`) are copied verbatim
    from the original `float32` input `y` so `observed_max_abs_diff == 0.0` even
    when noisy measurements exceed `[-1, 1]`.
    """
    if in_channels not in (1, 3):
        raise ValueError(f"in_channels must be 1 or 3, got {in_channels}.")
    if radius <= 0:
        raise ValueError(f"radius must be > 0, got {radius}.")

    method_key = method.lower()
    if method_key == "telea":
        cv_flag = cv2.INPAINT_TELEA
    elif method_key in ("ns", "navier_stokes"):
        cv_flag = cv2.INPAINT_NS
    else:
        raise ValueError(f"Unsupported telea_method: {method!r}. Expected 'telea' or 'ns'.")

    y_float = y.float()
    if y_float.ndim == 3:
        y_float = y_float.unsqueeze(0)
    if y_float.ndim != 4 or y_float.shape[1:] != (in_channels, image_size, image_size):
        raise ValueError(
            f"Expected y of shape (B, {in_channels}, {image_size}, {image_size}), "
            f"got {tuple(y_float.shape)}."
        )

    missing_2d = _normalize_mask_2d(mask, image_size)
    inpaint_mask_u8 = missing_2d.detach().cpu().numpy().astype(np.uint8) * 255

    batch_size = y_float.shape[0]
    inpainted_list = []
    for b in range(batch_size):
        arr = y_float[b].detach().cpu().clamp(-1.0, 1.0).numpy()
        if in_channels == 1:
            img_u8 = np.clip(np.rint((arr[0] + 1.0) * 127.5), 0, 255).astype(np.uint8)
            inpainted_u8 = cv2.inpaint(img_u8, inpaint_mask_u8, float(radius), cv_flag)
            inpainted_f = (inpainted_u8.astype(np.float32) / 127.5) - 1.0
            inpainted_chw = torch.from_numpy(inpainted_f).unsqueeze(0)
        else:
            arr_hwc = np.transpose(arr, (1, 2, 0))
            img_u8 = np.clip(np.rint((arr_hwc + 1.0) * 127.5), 0, 255).astype(np.uint8)
            inpainted_u8 = cv2.inpaint(img_u8, inpaint_mask_u8, float(radius), cv_flag)
            inpainted_f = (inpainted_u8.astype(np.float32) / 127.5) - 1.0
            inpainted_chw = torch.from_numpy(np.transpose(inpainted_f, (2, 0, 1)))
        inpainted_list.append(inpainted_chw)

    inpainted_batch = torch.stack(inpainted_list, dim=0).to(
        device=y.device, dtype=torch.float32
    )
    obs_mask = (~missing_2d).to(device=y.device).view(1, 1, image_size, image_size).expand(
        batch_size, in_channels, image_size, image_size
    )
    out = torch.where(obs_mask, y_float, inpainted_batch)
    return out.reshape(batch_size, in_channels * image_size * image_size)


def compute_spatial_weight_map(
    mask: Union[torch.Tensor, np.ndarray],
    image_size: int,
    spatial_sigma: float = 8.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Compute distance-transform spatial weight map `(1, 1, H, W)`.

    Returns `1.0` on observed pixels (`mask != 0`) and `exp(-dist^2 / (2 * sigma^2))`
    inside missing pixels (`mask == 0`), where `dist` is Euclidean distance to the
    nearest observed pixel.
    """
    missing_2d = _normalize_mask_2d(mask, image_size)
    missing_u8 = missing_2d.detach().cpu().numpy().astype(np.uint8)

    if spatial_sigma <= 0:
        w_np = (1.0 - missing_u8).astype(np.float32)
    else:
        dist = cv2.distanceTransform(missing_u8, cv2.DIST_L2, 5)
        w_np = np.exp(-(dist ** 2) / (2.0 * (float(spatial_sigma) ** 2))).astype(np.float32)
        w_np[missing_u8 == 0] = 1.0

    target_device = device if device is not None else (
        mask.device if isinstance(mask, torch.Tensor) else torch.device("cpu")
    )
    return torch.from_numpy(w_np).to(device=target_device, dtype=torch.float32).view(
        1, 1, image_size, image_size
    )


def compute_eta_map(
    mask: Union[torch.Tensor, np.ndarray],
    image_size: int,
    eta_hole: float = 0.25,
    eta_obs: float = 0.02,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Compute adaptive stochasticity map `eta(x, y)` of shape `(1, 1, H, W)`.

    Assigns `eta_hole` inside missing pixels (`mask == 0`) and `eta_obs` on observed
    pixels (`mask != 0`), clamped to `[0.0, 1.0]`. Binary step assignment avoids
    median-distance collapse on scattered/text masks (`median(dist_hole) == 1.0`).
    """
    target_device = device if device is not None else (
        mask.device if isinstance(mask, torch.Tensor) else torch.device("cpu")
    )
    missing_2d = _normalize_mask_2d(mask, image_size).to(device=target_device)
    hole_val = float(max(0.0, min(1.0, eta_hole)))
    obs_val = float(max(0.0, min(1.0, eta_obs)))

    eta_map = torch.where(
        missing_2d,
        torch.full((image_size, image_size), hole_val, device=target_device, dtype=torch.float32),
        torch.full((image_size, image_size), obs_val, device=target_device, dtype=torch.float32),
    )
    return eta_map.view(1, 1, image_size, image_size)


def _gaussian_blur_2d(tensor: torch.Tensor, sigma: float) -> torch.Tensor:
    channels = tensor.shape[1]
    height, width = tensor.shape[2], tensor.shape[3]
    radius = max(1, int(math.ceil(3.0 * float(sigma))))

    coords = torch.arange(-radius, radius + 1, device=tensor.device, dtype=tensor.dtype)
    kernel_1d = torch.exp(-0.5 * (coords / float(sigma)) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()

    kernel_h = kernel_1d.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
    kernel_v = kernel_1d.view(1, 1, -1, 1).expand(channels, 1, -1, 1)

    pad_mode = "reflect" if (radius < height and radius < width) else "replicate"
    blurred = F.pad(tensor, (radius, radius, 0, 0), mode=pad_mode)
    blurred = F.conv2d(blurred, kernel_h, groups=channels)
    blurred = F.pad(blurred, (0, 0, radius, radius), mode=pad_mode)
    blurred = F.conv2d(blurred, kernel_v, groups=channels)
    return blurred


def feather_bp_guidance(
    bp_guidance: torch.Tensor,
    mask: Union[torch.Tensor, np.ndarray],
    image_size: int,
    spatial_sigma: float = 8.0,
    weight_map: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Feather `BP_guidance` smoothly inward across the mask boundary.

    Because `Inpainting` `BP_guidance` is `0.0` inside `mask == 0`, this extrapolates
    observed-side `BP_guidance` via normalized separable Gaussian convolution and
    weights the inward extrapolation by `weight_map`, while returning `bp_guidance`
    bit-for-bit unchanged on observed pixels (`mask != 0`).
    """
    if spatial_sigma <= 0:
        return bp_guidance

    missing_2d = _normalize_mask_2d(mask, image_size)
    obs_mask = (~missing_2d).to(device=bp_guidance.device).view(1, 1, image_size, image_size)
    obs_f = obs_mask.to(dtype=bp_guidance.dtype)

    denom = _gaussian_blur_2d(obs_f, float(spatial_sigma))
    numer = _gaussian_blur_2d(bp_guidance * obs_f, float(spatial_sigma))
    bp_boundary = torch.where(
        denom > 1e-6,
        numer / denom.clamp_min(1e-6),
        torch.zeros_like(numer),
    )

    if weight_map is None:
        w = compute_spatial_weight_map(
            mask, image_size, spatial_sigma=spatial_sigma, device=bp_guidance.device
        )
    else:
        w = weight_map.to(device=bp_guidance.device)

    w = w.to(dtype=bp_guidance.dtype)
    return torch.where(obs_mask, bp_guidance, w * bp_boundary)


def get_cached_clip(
    model_name: str = "ViT-B-32",
    pretrained: str = "laion2b_s34b_b79k",
    device: Optional[torch.device] = None,
) -> Tuple[torch.nn.Module, Any]:
    """Lazily load and cache an OpenCLIP model + tokenizer on `device`."""
    target_device = device if device is not None else torch.device("cpu")
    key = (model_name, pretrained, str(target_device))
    if key not in _CLIP_MODEL_CACHE:
        import open_clip

        fork_devices = (
            [target_device.index if target_device.index is not None else 0]
            if target_device.type == "cuda" and torch.cuda.is_available()
            else []
        )
        with torch.random.fork_rng(devices=fork_devices):
            model, _, _ = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained
            )
            tokenizer = open_clip.get_tokenizer(model_name)
        model = model.to(target_device).eval()
        for param in model.parameters():
            param.requires_grad_(False)
        _CLIP_MODEL_CACHE[key] = (model, tokenizer)
    return _CLIP_MODEL_CACHE[key]


def get_cached_text_embedding(
    prompt: str,
    model_name: str = "ViT-B-32",
    pretrained: str = "laion2b_s34b_b79k",
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Compute and cache normalized OpenCLIP text embedding for `prompt`."""
    target_device = device if device is not None else torch.device("cpu")
    key = (model_name, pretrained, str(target_device), prompt)
    if key not in _CLIP_TEXT_CACHE:
        model, tokenizer = get_cached_clip(model_name, pretrained, target_device)
        with torch.no_grad():
            tokens = tokenizer([prompt]).to(target_device)
            txt_feat = F.normalize(model.encode_text(tokens).float(), dim=-1)
        _CLIP_TEXT_CACHE[key] = txt_feat.detach()
    return _CLIP_TEXT_CACHE[key]


def steer_clip_null_space(
    x0_t: torch.Tensor,
    A_funcs: Any,
    prompt: str,
    clip_model: str = "ViT-B-32",
    clip_pretrained: str = "laion2b_s34b_b79k",
    clip_scale: float = 0.25,
    clip_steps: int = 1,
    text_feat: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Steer `x0_t` inside the null-space `(I - A^+ A)` using OpenCLIP similarity loss."""
    if int(clip_steps) <= 0 or float(clip_scale) == 0.0:
        return x0_t.detach().float()

    device = x0_t.device
    model, _ = get_cached_clip(clip_model, clip_pretrained, device)
    txt_feat = (
        text_feat.to(device=device, dtype=torch.float32)
        if text_feat is not None
        else get_cached_text_embedding(prompt, clip_model, clip_pretrained, device)
    )

    mean = torch.tensor(_CLIP_MEAN, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(_CLIP_STD, device=device, dtype=torch.float32).view(1, 3, 1, 1)

    batch_size = x0_t.shape[0]
    x0_cur = x0_t.detach().float()

    with torch.enable_grad():
        for _ in range(int(clip_steps)):
            x_var = x0_cur.detach().requires_grad_(True)
            x01 = ((x_var + 1.0) * 0.5).clamp(0.0, 1.0)
            x224 = F.interpolate(
                x01,
                size=(224, 224),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
            if x224.shape[1] == 1:
                x224 = x224.repeat(1, 3, 1, 1)
            x_norm = (x224 - mean) / std
            img_feat = F.normalize(model.encode_image(x_norm).float(), dim=-1)
            loss = (1.0 - (img_feat * txt_feat).sum(dim=-1)).mean()
            (grad,) = torch.autograd.grad(loss, x_var)

            g_flat = grad.reshape(batch_size, -1)
            g_null = (g_flat - A_funcs.A_pinv(A_funcs.A(g_flat))).reshape_as(grad)

            sq_sum = g_null.pow(2).sum(dim=(1, 2, 3), keepdim=True)
            active = (
                (g_null.abs() > 1e-12)
                .sum(dim=(1, 2, 3), keepdim=True)
                .clamp_min(1)
                .to(g_null.dtype)
            )
            rms = (sq_sum / active).sqrt().clamp_min(1e-8)

            x0_next = torch.clamp(x0_cur - float(clip_scale) * (g_null / rms), -1.0, 1.0)
            delta_flat = (x0_next - x0_cur).reshape(batch_size, -1)
            delta_null = (
                delta_flat - A_funcs.A_pinv(A_funcs.A(delta_flat))
            ).reshape_as(x0_cur)
            x0_cur = (x0_cur + delta_null).detach()

    return x0_cur.detach().float()


def build_context_guidance(
    args: Any,
    mask: Union[torch.Tensor, np.ndarray],
    image_size: int,
    device: torch.device,
) -> Optional[Dict[str, Any]]:
    """Construct precomputed context-guidance configuration for `cm4ir_restoration`."""
    context_mode = getattr(args, "context_mode", "none")
    if context_mode in ("none", "telea"):
        return None

    if context_mode == "spatial":
        spatial_sigma = float(getattr(args, "spatial_sigma", 8.0))
        eta_hole = float(getattr(args, "eta_hole", 0.25))
        eta_obs = float(getattr(args, "eta_obs", 0.02))
        weight_map = compute_spatial_weight_map(
            mask, image_size, spatial_sigma=spatial_sigma, device=device
        )
        eta_map = compute_eta_map(
            mask, image_size, eta_hole=eta_hole, eta_obs=eta_obs, device=device
        )
        return {
            "mode": "spatial",
            "mask": mask,
            "image_size": image_size,
            "spatial_sigma": spatial_sigma,
            "eta_hole": eta_hole,
            "eta_obs": eta_obs,
            "weight_map": weight_map,
            "eta_map": eta_map,
        }

    if context_mode == "clip":
        clip_model = getattr(args, "clip_model", "ViT-B-32")
        clip_pretrained = getattr(args, "clip_pretrained", "laion2b_s34b_b79k")
        clip_negative_control = bool(getattr(args, "clip_negative_control", False))
        clip_prompt = getattr(args, "clip_prompt", "a photograph of a cat")
        clip_neg_prompt = getattr(
            args,
            "clip_neg_prompt",
            "a photograph of a red fire hydrant in outer space",
        )
        if clip_negative_control:
            active_prompt = (
                clip_neg_prompt
                if clip_neg_prompt
                else "a photograph of a red fire hydrant in outer space"
            )
        else:
            active_prompt = clip_prompt

        clip_scale = float(getattr(args, "clip_scale", 0.25))
        clip_steps = int(getattr(args, "clip_steps", 1))
        text_feat = get_cached_text_embedding(
            active_prompt,
            model_name=clip_model,
            pretrained=clip_pretrained,
            device=device,
        )
        return {
            "mode": "clip",
            "clip_model": clip_model,
            "clip_pretrained": clip_pretrained,
            "prompt": active_prompt,
            "clip_scale": clip_scale,
            "clip_steps": clip_steps,
            "text_feat": text_feat,
            "negative_control": clip_negative_control,
        }

    raise ValueError(f"Unsupported context_mode: {context_mode!r}")
