import numpy as np
import torch
from tqdm import tqdm

from functions.context_guidance import feather_bp_guidance, steer_clip_null_space
from functions.harmonic_poisson import poisson_residual_bp

class_num = 951


def get_scalings_for_boundary_condition(sigma, sigma_data=0.5, sigma_min=0.002):
    c_skip = sigma_data ** 2 / (
            (sigma - sigma_min) ** 2 + sigma_data ** 2
    )
    c_out = (
            (sigma - sigma_min)
            * sigma_data
            / (sigma ** 2 + sigma_data ** 2) ** 0.5
    )
    c_in = 1 / (sigma ** 2 + sigma_data ** 2) ** 0.5
    return c_skip, c_out, c_in


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]


def get_next_alpha(prev_alpha, gamma):
    return torch.clamp((prev_alpha * (1 + gamma)), 0, 0.9999)


def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(1)
    return a


def cm4ir_restoration(
        x,
        model,
        A_funcs,
        y_0,
        sigma_y,
        betas,
        eta,
        deltas,
        deltas_injection_type,
        gamma,
        iN,
        zeta,
        classes,
        config,
        mu=1,
        mu_factor=1,
        sigma_t_min=0.002,
        sigma_t_max=80.0,
        context_guidance=None,
):
    guidance_mode = "none" if context_guidance is None else context_guidance.get("mode", "none")
    if guidance_mode == "spatial":
        spatial_weight_map = context_guidance["weight_map"].to(device=x.device, dtype=x.dtype)
        spatial_eta_map = context_guidance["eta_map"].to(device=x.device, dtype=x.dtype)
    elif guidance_mode == "poisson":
        spatial_weight_map = None
        spatial_eta_map = context_guidance["eta_map"].to(device=x.device, dtype=x.dtype)
    else:
        spatial_weight_map = None
        spatial_eta_map = None

    with torch.no_grad():
        iter_ind = -1
        t = (torch.ones(1) * (iN + 1)).to(x.device)
        aN = compute_alpha(betas, t.long())
        alphas = [aN]
        for i in range(config.sampling.T_sampling - 1):
            alphas.append(get_next_alpha(alphas[-1], gamma).reshape(1))

        edm_sigmas = context_guidance.get("edm_sigmas") if guidance_mode == "poisson" else None
        if edm_sigmas is not None:
            sigma_schedule = [torch.tensor([float(s)], device=x.device, dtype=x.dtype) for s in edm_sigmas]
            sigma_t = sigma_schedule[0]
        else:
            sigma_schedule = None
            sigma_t = torch.sqrt(1 - alphas[0])
        xt = x + torch.randn_like(x) * sigma_t

        alphas_next = alphas[1:] + [-1]
        bp_eta_reg = sigma_y ** 2 * zeta

        for at, at_next in tqdm(zip(alphas, alphas_next)):
            iter_ind += 1

            if sigma_schedule is not None:
                sigma_t = sigma_schedule[iter_ind]
                delta_cur = 0.0
            else:
                sigma_t = torch.sqrt(1 - at)
                sigma_t = torch.clamp(sigma_t, sigma_t_min, sigma_t_max)
                delta_cur = deltas[iter_ind]

            if deltas_injection_type == 0:
                scaling_values = get_scalings_for_boundary_condition(sigma_t)
            elif deltas_injection_type == 1:
                scaling_values = get_scalings_for_boundary_condition((1 + delta_cur) * sigma_t)
            else:
                raise NotImplementedError(f"Unsupported deltas_injection_type: {deltas_injection_type}."
                                          f"Please view README for optional values.")

            c_skip, c_out, c_in = [append_dims(s, xt.ndim) for s in scaling_values]
            rescaled_sigma_t = 1000 * 0.25 * torch.log((1 + delta_cur) * sigma_t + 1e-44)

            if classes is None:
                et = model(c_in * xt, rescaled_sigma_t)
            else:
                et = model(c_in * xt, rescaled_sigma_t, classes)
            x0_t = c_out * et + c_skip * xt
            x0_t = torch.clamp(x0_t, -1.0, 1.0)

            if guidance_mode == "clip":
                x0_t = steer_clip_null_space(
                    x0_t=x0_t,
                    A_funcs=A_funcs,
                    prompt=context_guidance["prompt"],
                    clip_model=context_guidance.get("clip_model", "ViT-B-32"),
                    clip_pretrained=context_guidance.get("clip_pretrained", "laion2b_s34b_b79k"),
                    clip_scale=context_guidance.get("clip_scale", 0.25),
                    clip_steps=context_guidance.get("clip_steps", 1),
                    text_feat=context_guidance.get("text_feat"),
                )

            z_hat_minus = (x0_t - xt) / sigma_t
            if sigma_y != 0 and at_next == -1:  # last iteration of noisy case, skip guidance
                break

            BP_guidance = A_funcs.A_pinv_add_eta(
                A_funcs.A(x0_t.reshape(x0_t.size(0), -1)) - y_0.reshape(y_0.size(0), -1), eta=bp_eta_reg).reshape(
                *x.size())
            if guidance_mode == "spatial":
                BP_guidance = feather_bp_guidance(
                    BP_guidance,
                    mask=context_guidance["mask"],
                    image_size=context_guidance["image_size"],
                    spatial_sigma=context_guidance["spatial_sigma"],
                    weight_map=spatial_weight_map,
                )
            elif guidance_mode == "poisson":
                BP_guidance = context_guidance["solver"].solve_batch(BP_guidance)
            xt = x0_t - (mu * mu_factor ** iter_ind) * BP_guidance
            if guidance_mode == "poisson":
                xt = torch.clamp(xt, -1.0, 1.0)

            if at_next != -1:
                if sigma_schedule is not None:
                    next_sigma_t = sigma_schedule[iter_ind + 1]
                else:
                    next_sigma_t = torch.sqrt(1 - at_next)
                    next_sigma_t = torch.clamp(next_sigma_t, sigma_t_min, sigma_t_max)

                if guidance_mode in ("spatial", "poisson"):
                    c1 = next_sigma_t * spatial_eta_map
                    c2 = next_sigma_t * torch.sqrt(torch.clamp(1.0 - spatial_eta_map ** 2, min=0.0))
                else:
                    c1 = next_sigma_t * eta
                    c2 = next_sigma_t * ((1 - eta ** 2) ** 0.5)
                xt = xt + c1 * torch.randn_like(xt) + c2 * z_hat_minus

        if sigma_y != 0:
            return [x0_t.to('cpu')]
        return [xt.to('cpu')]
