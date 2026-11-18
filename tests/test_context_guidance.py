import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch
import torch.nn as nn
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import functions.cm4ir_scheme as cm4ir_scheme
import functions.context_guidance as cg
from functions.median_inpainting import median_inpainting
from functions.svd_operators import Inpainting
import guided_diffusion.diffusion as diffusion_mod
from guided_diffusion.diffusion import Diffusion, get_beta_schedule
from guided_diffusion.script_util import create_model
import main as main_mod


def _load_git_module(tmp_path: Path, rev: str, rel_path: str, mod_name: str):
    code = subprocess.check_output(
        ["git", "show", f"{rev}:{rel_path}"],
        cwd=str(REPO_ROOT),
        text=True,
    )
    out_file = tmp_path / f"{mod_name}.py"
    out_file.write_text(code, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(mod_name, str(out_file))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _make_center_box_mask(image_size: int = 256, box_size: int = 96) -> np.ndarray:
    mask = np.ones((image_size, image_size), dtype=np.float32)
    start = (image_size - box_size) // 2
    end = start + box_size
    mask[start:end, start:end] = 0.0
    return mask


def _load_mask_array(name: str, image_size: int = 256) -> np.ndarray:
    if name == "center_box":
        return _make_center_box_mask(image_size=image_size, box_size=96)
    mask_path = REPO_ROOT / "exp" / "inpainting_masks" / name
    return np.load(str(mask_path))


def _load_real_cat_tensor(device: torch.device, image_size: int = 256) -> torch.Tensor:
    cat_path = REPO_ROOT / "exp" / "datasets" / "lsun_cat" / "cat" / "14.png"
    img = Image.open(cat_path).convert("RGB").resize((image_size, image_size))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    return 2.0 * t - 1.0


class _DeterministicDummyUNet(nn.Module):
    def __init__(self, channels: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        with torch.no_grad():
            self.conv.weight.zero_()
            for c in range(channels):
                self.conv.weight[c, c, 1, 1] = 0.25
                self.conv.weight[c, c, 0, 1] = 0.05
                self.conv.weight[c, c, 2, 1] = 0.05
            self.conv.bias.fill_(0.01)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        scale = torch.tanh(t.reshape(-1, 1, 1, 1) * 1e-3).to(dtype=x.dtype, device=x.device)
        return self.conv(x) + 0.05 * scale


def _make_test_config(image_size: int = 256, channels: int = 3, num_workers: int = 32):
    return SimpleNamespace(
        data=SimpleNamespace(
            dataset="lsun_cat",
            category="",
            image_size=image_size,
            channels=channels,
            logit_transform=False,
            uniform_dequantization=False,
            gaussian_dequantization=False,
            rescaled=True,
            random_flip=False,
            num_workers=num_workers,
            out_of_dist=False,
        ),
        model=SimpleNamespace(
            class_cond=False,
            use_fp16=False,
        ),
        diffusion=SimpleNamespace(
            beta_schedule="linear",
            beta_start=0.0001,
            beta_end=0.02,
            num_diffusion_timesteps=1000,
        ),
        sampling=SimpleNamespace(
            batch_size=1,
            T_sampling=4,
        ),
    )


def _devices_to_test():
    devs = [torch.device("cpu")]
    if torch.cuda.is_available():
        devs.append(torch.device("cuda"))
    return devs


def _seed_all(seed: int, device: torch.device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def _capture_rng(device: torch.device):
    cpu_state = torch.get_rng_state().clone()
    cuda_state = torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None
    return cpu_state, cuda_state


def _assert_rng_equal(s1, s2):
    assert torch.equal(s1[0], s2[0])
    if s1[1] is not None and s2[1] is not None:
        assert torch.equal(s1[1], s2[1])


@pytest.mark.parametrize("device", _devices_to_test())
@pytest.mark.parametrize("mask_name", ["random_80_mask.npy", "lorem_ipsum_mask.npy"])
@pytest.mark.parametrize("sigma_y", [0.0, 0.1])
def test_none_matches_3d09c114_bit_exact(tmp_path: Path, device: torch.device, mask_name: str, sigma_y: float):
    ref_cm4ir = _load_git_module(tmp_path, "3d09c114", "functions/cm4ir_scheme.py", "ref_cm4ir")
    ref_median = _load_git_module(tmp_path, "3d09c114", "functions/median_inpainting.py", "ref_median")
    ref_svd = _load_git_module(tmp_path, "3d09c114", "functions/svd_operators.py", "ref_svd")

    cfg = _make_test_config(image_size=256, channels=3)
    betas = torch.from_numpy(
        get_beta_schedule(
            beta_schedule="linear",
            beta_start=0.0001,
            beta_end=0.02,
            num_diffusion_timesteps=1000,
        )
    ).float().to(device)
    model = _DeterministicDummyUNet().to(device).eval()

    mask_np = _load_mask_array(mask_name)
    mask_t = torch.from_numpy(mask_np).to(device).reshape(-1)
    missing_r = torch.nonzero(mask_t == 0).long().reshape(-1) * 3
    missing = torch.cat([missing_r, missing_r + 1, missing_r + 2], dim=0)

    A_ref = ref_svd.Inpainting(3, 256, missing, device)
    A_cur = Inpainting(3, 256, missing, device)

    x_orig = _load_real_cat_tensor(device)

    def _run_pipeline(use_ref: bool, context_guidance=None):
        _seed_all(1234, device)
        y = A_ref.A(x_orig) if use_ref else A_cur.A(x_orig)
        y = y + sigma_y * torch.randn_like(y)
        Apy = (A_ref if use_ref else A_cur).A_pinv_add_eta(y, 0.0).view(1, 3, 256, 256)
        if use_ref:
            x_init = ref_median.median_inpainting(Apy, mask_t, 3, 256).view(1, 3, 256, 256)
            out = ref_cm4ir.cm4ir_restoration(
                x=x_init,
                model=model,
                A_funcs=A_ref,
                y_0=y,
                sigma_y=sigma_y,
                betas=betas,
                eta=0.1,
                deltas=[0, 0, 0, 0],
                deltas_injection_type=0,
                gamma=0.2,
                iN=150,
                zeta=0.0,
                classes=None,
                config=cfg,
            )
        else:
            x_init = median_inpainting(Apy, mask_t, 3, 256).view(1, 3, 256, 256)
            out = cm4ir_scheme.cm4ir_restoration(
                x=x_init,
                model=model,
                A_funcs=A_cur,
                y_0=y,
                sigma_y=sigma_y,
                betas=betas,
                eta=0.1,
                deltas=[0, 0, 0, 0],
                deltas_injection_type=0,
                gamma=0.2,
                iN=150,
                zeta=0.0,
                classes=None,
                config=cfg,
                context_guidance=context_guidance,
            )
        return out[0], _capture_rng(device)

    out_ref, rng_ref = _run_pipeline(use_ref=True)
    out_none_default, rng_default = _run_pipeline(use_ref=False, context_guidance=None)
    out_none_dict, rng_dict = _run_pipeline(use_ref=False, context_guidance={"mode": "none"})

    assert torch.equal(out_ref, out_none_default)
    assert torch.equal(out_ref, out_none_dict)
    _assert_rng_equal(rng_ref, rng_default)
    _assert_rng_equal(rng_ref, rng_dict)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="RTX 5060 CUDA required")
def test_none_matches_3d09c114_real_unet_cuda(tmp_path: Path):
    ckpt_path = REPO_ROOT / "exp" / "logs" / "lsun_cat" / "cd_cat256_lpips.pt"
    if not ckpt_path.exists():
        pytest.skip("cd_cat256_lpips.pt not present")

    ref_cm4ir = _load_git_module(tmp_path, "3d09c114", "functions/cm4ir_scheme.py", "ref_cm4ir_real")
    ref_svd = _load_git_module(tmp_path, "3d09c114", "functions/svd_operators.py", "ref_svd_real")

    with open(REPO_ROOT / "configs" / "lsun_cat_256.yml", "r", encoding="utf-8") as f:
        cfg_raw = yaml.safe_load(f)
    cfg = main_mod.dict2namespace(cfg_raw)
    device = torch.device("cuda")

    model_cfg = dict(vars(cfg.model))
    model_cfg["image_size"] = cfg.data.image_size
    unet = create_model(**model_cfg)
    unet.load_state_dict(torch.load(str(ckpt_path), map_location=device))
    unet.to(device)
    if cfg.model.use_fp16:
        unet.convert_to_fp16()
    unet.eval()

    betas = torch.from_numpy(
        get_beta_schedule(
            beta_schedule=cfg.diffusion.beta_schedule,
            beta_start=cfg.diffusion.beta_start,
            beta_end=cfg.diffusion.beta_end,
            num_diffusion_timesteps=cfg.diffusion.num_diffusion_timesteps,
        )
    ).float().to(device)

    mask_np = _load_mask_array("random_80_mask.npy")
    mask_t = torch.from_numpy(mask_np).to(device).reshape(-1)
    missing_r = torch.nonzero(mask_t == 0).long().reshape(-1) * 3
    missing = torch.cat([missing_r, missing_r + 1, missing_r + 2], dim=0)
    A_ref = ref_svd.Inpainting(3, 256, missing, device)
    A_cur = Inpainting(3, 256, missing, device)

    x_orig = _load_real_cat_tensor(device)

    _seed_all(1234, device)
    y_ref = A_ref.A(x_orig) + 0.1 * torch.randn_like(A_ref.A(x_orig))
    Apy_ref = A_ref.A_pinv_add_eta(y_ref, 0.0).view(1, 3, 256, 256)
    x_init_ref = median_inpainting(Apy_ref, mask_t, 3, 256).view(1, 3, 256, 256)
    out_ref = ref_cm4ir.cm4ir_restoration(
        x=x_init_ref,
        model=unet,
        A_funcs=A_ref,
        y_0=y_ref,
        sigma_y=0.1,
        betas=betas,
        eta=0.1,
        deltas=[0, 0, 0, 0],
        deltas_injection_type=0,
        gamma=0.2,
        iN=150,
        zeta=0.0,
        classes=None,
        config=cfg,
    )[0]
    rng_ref = _capture_rng(device)

    _seed_all(1234, device)
    y_cur = A_cur.A(x_orig) + 0.1 * torch.randn_like(A_cur.A(x_orig))
    Apy_cur = A_cur.A_pinv_add_eta(y_cur, 0.0).view(1, 3, 256, 256)
    x_init_cur = median_inpainting(Apy_cur, mask_t, 3, 256).view(1, 3, 256, 256)
    out_cur = cm4ir_scheme.cm4ir_restoration(
        x=x_init_cur,
        model=unet,
        A_funcs=A_cur,
        y_0=y_cur,
        sigma_y=0.1,
        betas=betas,
        eta=0.1,
        deltas=[0, 0, 0, 0],
        deltas_injection_type=0,
        gamma=0.2,
        iN=150,
        zeta=0.0,
        classes=None,
        config=cfg,
        context_guidance=None,
    )[0]
    rng_cur = _capture_rng(device)

    assert torch.isfinite(out_cur).all()
    assert (out_ref - out_cur).abs().max().item() == 0.0
    _assert_rng_equal(rng_ref, rng_cur)


@pytest.mark.parametrize("device", _devices_to_test())
@pytest.mark.parametrize("mask_name", ["random_80_mask.npy", "lorem_ipsum_mask.npy", "center_box"])
def test_inpainting_kept_indices_exact(tmp_path: Path, device: torch.device, mask_name: str):
    ref_svd = _load_git_module(tmp_path, "3d09c114", "functions/svd_operators.py", "ref_svd_idx")
    mask_np = _load_mask_array(mask_name)
    mask_t = torch.from_numpy(mask_np).to(device).reshape(-1)
    missing_r = torch.nonzero(mask_t == 0).long().reshape(-1) * 3
    missing = torch.cat([missing_r, missing_r + 1, missing_r + 2], dim=0)

    A_ref = ref_svd.Inpainting(3, 256, missing, device)
    A_cur = Inpainting(3, 256, missing, device)

    assert torch.equal(A_ref.kept_indices, A_cur.kept_indices)
    assert A_cur.kept_indices.dtype == torch.int64
    assert A_cur.kept_indices.device == A_ref.kept_indices.device

    # Edge cases on small 8x8 operator: empty, full, unsorted+duplicate
    for m_list in [[], list(range(3 * 64)), [15, 2, 15, 0, 7]]:
        m_small = torch.tensor(m_list, dtype=torch.long, device=device)
        r_s = ref_svd.Inpainting(3, 8, m_small, device)
        c_s = Inpainting(3, 8, m_small, device)
        assert torch.equal(r_s.kept_indices, c_s.kept_indices)


@pytest.mark.parametrize("device", _devices_to_test())
@pytest.mark.parametrize("method", ["telea", "ns"])
@pytest.mark.parametrize("mask_name", ["random_80_mask.npy", "lorem_ipsum_mask.npy", "center_box"])
def test_telea_inpainting_invariants(device: torch.device, method: str, mask_name: str):
    mask_np = _load_mask_array(mask_name)
    mask_t = torch.from_numpy(mask_np).to(device).reshape(-1)
    obs_2d = (torch.from_numpy(mask_np).to(device) != 0)

    _seed_all(42, device)
    # Deliberately exceed [-1, 1] to verify verbatim float32 observed restoration
    y = torch.empty((2, 3, 256, 256), device=device, dtype=torch.float32).uniform_(-1.35, 1.35)
    y[:, :, ~obs_2d] = 0.0

    rng_before = _capture_rng(device)
    out_flat = cg.telea_inpainting(y, mask_t, in_channels=3, image_size=256, method=method, radius=3.0)
    rng_after = _capture_rng(device)

    _assert_rng_equal(rng_before, rng_after)
    assert out_flat.shape == (2, 3 * 256 * 256)
    assert out_flat.dtype == torch.float32
    assert out_flat.device == y.device
    assert torch.isfinite(out_flat).all()

    out_4d = out_flat.view(2, 3, 256, 256)
    obs_diff = (out_4d[:, :, obs_2d] - y[:, :, obs_2d]).abs().max().item()
    assert obs_diff == 0.0

    # Ensure hole pixels are actually filled (not stuck at 0)
    assert out_4d[:, :, ~obs_2d].abs().max().item() > 0.0

    # Verify single-channel support (C=1)
    y1 = y[:, :1, :, :]
    out1 = cg.telea_inpainting(y1, mask_t, in_channels=1, image_size=256, method=method, radius=3.0)
    assert out1.shape == (2, 256 * 256)
    assert (out1.view(2, 1, 256, 256)[:, :, obs_2d] - y1[:, :, obs_2d]).abs().max().item() == 0.0

    if mask_name == "lorem_ipsum_mask.npy":
        # Verify anti-aliased values 1..254 are treated as observed and copied verbatim
        aa_mask = torch.from_numpy((mask_np > 0) & (mask_np < 255)).to(device)
        assert aa_mask.any()
        assert (out_4d[:, :, aa_mask] - y[:, :, aa_mask]).abs().max().item() == 0.0


def test_telea_inpainting_invalid_args():
    y = torch.zeros((1, 3, 16, 16))
    mask = torch.ones((16, 16))
    with pytest.raises(ValueError):
        cg.telea_inpainting(y, mask, in_channels=2, image_size=16)
    with pytest.raises(ValueError):
        cg.telea_inpainting(y, mask, in_channels=3, image_size=16, method="invalid")
    with pytest.raises(ValueError):
        cg.telea_inpainting(y, mask, in_channels=3, image_size=16, radius=0.0)


@pytest.mark.parametrize("device", _devices_to_test())
@pytest.mark.parametrize("mask_name", ["random_80_mask.npy", "lorem_ipsum_mask.npy", "center_box"])
def test_spatial_guidance_invariants(device: torch.device, mask_name: str):
    mask_np = _load_mask_array(mask_name)
    mask_t = torch.from_numpy(mask_np).to(device).reshape(-1)
    obs_2d = (torch.from_numpy(mask_np).to(device) != 0)
    miss_2d = ~obs_2d

    w = cg.compute_spatial_weight_map(mask_t, 256, spatial_sigma=8.0, device=device)
    assert w.shape == (1, 1, 256, 256)
    assert w.device.type == device.type
    assert (w[0, 0, obs_2d] == 1.0).all()
    assert (w[0, 0, miss_2d] < 1.0).all()
    assert (w[0, 0, miss_2d] > 0.0).all()

    if mask_name == "center_box":
        # Box is [80:176, 80:176]; verify radial monotonic decay from border to center
        assert w[0, 0, 80, 128] > w[0, 0, 96, 128] > w[0, 0, 128, 128]

    eta_map = cg.compute_eta_map(mask_t, 256, eta_hole=0.25, eta_obs=0.02, device=device)
    assert eta_map.shape == (1, 1, 256, 256)
    assert torch.allclose(eta_map[0, 0, miss_2d], torch.tensor(0.25, device=device))
    assert torch.allclose(eta_map[0, 0, obs_2d], torch.tensor(0.02, device=device))
    assert (eta_map >= 0.0).all() and (eta_map <= 1.0).all()

    # Feather BP_guidance: bit-exact on mask != 0, non-zero inward feathering in mask == 0
    _seed_all(7, device)
    bp = torch.randn((1, 3, 256, 256), device=device)
    bp[:, :, miss_2d] = 0.0
    bp_feathered = cg.feather_bp_guidance(bp, mask_t, 256, spatial_sigma=8.0, weight_map=w)
    assert torch.equal(bp_feathered[:, :, obs_2d], bp[:, :, obs_2d])
    assert bp_feathered[:, :, miss_2d].abs().max().item() > 0.0
    assert torch.isfinite(bp_feathered).all()

    # Noiseless cm4ir_restoration (sigma_y=0, zeta=0) keeps A(x_hat) == y_0
    missing_r = torch.nonzero(mask_t == 0).long().reshape(-1) * 3
    missing = torch.cat([missing_r, missing_r + 1, missing_r + 2], dim=0)
    A_funcs = Inpainting(3, 256, missing, device)
    x_orig = _load_real_cat_tensor(device)
    y_0 = A_funcs.A(x_orig)
    Apy = A_funcs.A_pinv_add_eta(y_0, 0.0).view(1, 3, 256, 256)
    x_init = cg.telea_inpainting(Apy, mask_t, 3, 256).view(1, 3, 256, 256)
    cfg = _make_test_config(256, 3)
    betas = torch.from_numpy(
        get_beta_schedule(
            beta_schedule="linear",
            beta_start=0.0001,
            beta_end=0.02,
            num_diffusion_timesteps=1000,
        )
    ).float().to(device)
    model = _DeterministicDummyUNet().to(device).eval()
    spatial_cfg = cg.build_context_guidance(
        SimpleNamespace(context_mode="spatial", spatial_sigma=8.0, eta_hole=0.25, eta_obs=0.02),
        mask=mask_t,
        image_size=256,
        device=device,
    )
    out = cm4ir_scheme.cm4ir_restoration(
        x=x_init,
        model=model,
        A_funcs=A_funcs,
        y_0=y_0,
        sigma_y=0.0,
        betas=betas,
        eta=0.1,
        deltas=[0, 0, 0, 0],
        deltas_injection_type=0,
        gamma=0.2,
        iN=150,
        zeta=0.0,
        classes=None,
        config=cfg,
        context_guidance=spatial_cfg,
    )[0]
    assert (A_funcs.A(out.to(device)) - y_0).abs().max().item() <= 1e-6


@pytest.mark.parametrize("device", _devices_to_test())
@pytest.mark.parametrize("mask_name", ["random_80_mask.npy", "lorem_ipsum_mask.npy", "center_box"])
def test_clip_null_space_and_negative_control(monkeypatch, device: torch.device, mask_name: str):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    cg._CLIP_MODEL_CACHE.clear()
    cg._CLIP_TEXT_CACHE.clear()

    # Non-clip modes must NOT populate CLIP caches or mutate RNG
    rng_before = _capture_rng(device)
    mask_np = _load_mask_array(mask_name)
    mask_t = torch.from_numpy(mask_np).to(device).reshape(-1)
    assert cg.build_context_guidance(SimpleNamespace(context_mode="none"), mask_t, 256, device) is None
    assert cg.build_context_guidance(SimpleNamespace(context_mode="telea"), mask_t, 256, device) is None
    _ = cg.build_context_guidance(
        SimpleNamespace(context_mode="spatial", spatial_sigma=8.0, eta_hole=0.25, eta_obs=0.02),
        mask_t,
        256,
        device,
    )
    assert len(cg._CLIP_MODEL_CACHE) == 0
    assert len(cg._CLIP_TEXT_CACHE) == 0
    _assert_rng_equal(rng_before, _capture_rng(device))

    missing_r = torch.nonzero(mask_t == 0).long().reshape(-1) * 3
    missing = torch.cat([missing_r, missing_r + 1, missing_r + 2], dim=0)
    A_funcs = Inpainting(3, 256, missing, device)

    x_orig = _load_real_cat_tensor(device)
    x_in = x_orig.clone().contiguous()
    x_in.view(1, -1)[:, missing] = 0.0

    pos_prompt = "a photograph of a cat"
    neg_prompt = "a photograph of a red fire hydrant in outer space"

    out_pos = cg.steer_clip_null_space(
        x_in,
        A_funcs=A_funcs,
        prompt=pos_prompt,
        clip_model="ViT-B-32",
        clip_pretrained="laion2b_s34b_b79k",
        clip_scale=0.25,
        clip_steps=1,
    )
    out_neg = cg.steer_clip_null_space(
        x_in,
        A_funcs=A_funcs,
        prompt=neg_prompt,
        clip_model="ViT-B-32",
        clip_pretrained="laion2b_s34b_b79k",
        clip_scale=0.25,
        clip_steps=1,
    )

    assert torch.isfinite(out_pos).all() and torch.isfinite(out_neg).all()
    assert out_pos.dtype == torch.float32
    assert (A_funcs.A(out_pos.reshape(1, -1)) - A_funcs.A(x_in.reshape(1, -1))).abs().max().item() <= 1e-6
    assert (A_funcs.A(out_neg.reshape(1, -1)) - A_funcs.A(x_in.reshape(1, -1))).abs().max().item() <= 1e-6
    assert (out_pos - out_neg).abs().max().item() > 1e-2

    # Verify cache reuse
    m1, tok1 = cg.get_cached_clip("ViT-B-32", "laion2b_s34b_b79k", device)
    m2, tok2 = cg.get_cached_clip("ViT-B-32", "laion2b_s34b_b79k", device)
    assert m1 is m2 and tok1 is tok2
    t1 = cg.get_cached_text_embedding(pos_prompt, "ViT-B-32", "laion2b_s34b_b79k", device)
    t2 = cg.get_cached_text_embedding(pos_prompt, "ViT-B-32", "laion2b_s34b_b79k", device)
    assert t1 is t2


@pytest.mark.parametrize("device", _devices_to_test())
@pytest.mark.parametrize("context_mode", ["none", "telea", "spatial", "clip"])
@pytest.mark.parametrize("mask_name", ["random_80_mask.npy", "lorem_ipsum_mask.npy", "center_box"])
def test_all_modes_all_masks_cm4ir_loop_finite(
    monkeypatch, device: torch.device, context_mode: str, mask_name: str
):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    cfg = _make_test_config(256, 3)
    betas = torch.from_numpy(
        get_beta_schedule(
            beta_schedule="linear",
            beta_start=0.0001,
            beta_end=0.02,
            num_diffusion_timesteps=1000,
        )
    ).float().to(device)
    model = _DeterministicDummyUNet().to(device).eval()

    mask_np = _load_mask_array(mask_name)
    mask_t = torch.from_numpy(mask_np).to(device).reshape(-1)
    missing_r = torch.nonzero(mask_t == 0).long().reshape(-1) * 3
    missing = torch.cat([missing_r, missing_r + 1, missing_r + 2], dim=0)
    A_funcs = Inpainting(3, 256, missing, device)

    x_orig = _load_real_cat_tensor(device)
    _seed_all(1234, device)
    for sigma_y in (0.0, 0.1):
        y = A_funcs.A(x_orig) + sigma_y * torch.randn_like(A_funcs.A(x_orig))
        Apy = A_funcs.A_pinv_add_eta(y, 0.0).view(1, 3, 256, 256)
        if context_mode == "telea":
            x_init = cg.telea_inpainting(Apy, mask_t, 3, 256).view(1, 3, 256, 256)
        else:
            x_init = median_inpainting(Apy, mask_t, 3, 256).view(1, 3, 256, 256)

        ctx_cfg = cg.build_context_guidance(
            SimpleNamespace(
                context_mode=context_mode,
                spatial_sigma=8.0,
                eta_hole=0.25,
                eta_obs=0.02,
                clip_model="ViT-B-32",
                clip_pretrained="laion2b_s34b_b79k",
                clip_prompt="a photograph of a cat",
                clip_neg_prompt="a photograph of a red fire hydrant in outer space",
                clip_negative_control=False,
                clip_scale=0.25,
                clip_steps=1,
            ),
            mask=mask_t,
            image_size=256,
            device=device,
        )
        out = cm4ir_scheme.cm4ir_restoration(
            x=x_init,
            model=model,
            A_funcs=A_funcs,
            y_0=y,
            sigma_y=sigma_y,
            betas=betas,
            eta=0.1,
            deltas=[0, 0, 0, 0],
            deltas_injection_type=0,
            gamma=0.2,
            iN=150,
            zeta=0.0,
            classes=None,
            config=cfg,
            context_guidance=ctx_cfg,
        )[0]
        assert out.shape == (1, 3, 256, 256)
        assert out.dtype == torch.float32
        assert torch.isfinite(out).all()


@pytest.mark.parametrize("device", _devices_to_test())
def test_diffusion_sample_ids_and_device_safety(tmp_path: Path, monkeypatch, device: torch.device):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    # Build synthetic dataset folder matching ImageFolder structure
    exp_dir = tmp_path / "exp"
    cat_dir = exp_dir / "datasets" / "lsun_cat" / "cat"
    cat_dir.mkdir(parents=True)
    for i in range(3):
        arr = np.full((64, 64, 3), fill_value=40 * (i + 1), dtype=np.uint8)
        arr[:32, :32, 0] = 200
        Image.fromarray(arr).resize((256, 256)).save(cat_dir / f"{i}.png")

    mask_dir = exp_dir / "inpainting_masks"
    mask_dir.mkdir(parents=True)
    mask_np = _make_center_box_mask(256, 64)
    np.save(str(mask_dir / "box.npy"), mask_np)

    ckpt_dir = exp_dir / "logs"
    ckpt_dir.mkdir(parents=True)
    dummy_unet = _DeterministicDummyUNet()
    torch.save(dummy_unet.state_dict(), str(ckpt_dir / "dummy.pt"))

    monkeypatch.setattr(diffusion_mod, "create_model", lambda **kwargs: _DeterministicDummyUNet())

    def _make_args(out_name: str, sample_ids, context_mode="none", clip_negative_control=False):
        out_dir = tmp_path / out_name
        return SimpleNamespace(
            exp=str(exp_dir),
            model_ckpt="dummy.pt",
            deg="inpainting",
            sigma_y=0.05,
            seed=1234,
            deltas="",
            deltas_injection_type=0,
            inpainting_mask_path="box.npy",
            image_folder=str(out_dir),
            save_observed_img=True,
            subset_start=-1,
            subset_end=-1,
            sample_ids=sample_ids,
            num_workers=None,
            iN=150,
            gamma=0.2,
            eta=0.1,
            zeta=0.0,
            context_mode=context_mode,
            telea_method="telea",
            telea_radius=3.0,
            spatial_sigma=8.0,
            eta_hole=0.25,
            eta_obs=0.02,
            clip_model="ViT-B-32",
            clip_pretrained="laion2b_s34b_b79k",
            clip_prompt="a photograph of a cat",
            clip_neg_prompt="a photograph of a red fire hydrant in outer space",
            clip_negative_control=clip_negative_control,
            clip_scale=0.25,
            clip_steps=1,
        )

    cfg = _make_test_config(256, 3, num_workers=32)

    args_a = _make_args("run_1_0", sample_ids=[1, 0], context_mode="telea")
    diff_a = Diffusion(args_a, cfg, device=device)
    diff_a.sample(logger=None)
    assert args_a.sigma_y == pytest.approx(0.05)  # not mutated
    assert len(diff_a.last_run_records) == 2
    assert [r["index"] for r in diff_a.last_run_records] == [1, 0]
    for rec in diff_a.last_run_records:
        for k in (
            "index",
            "x_orig_01",
            "x_hat_01",
            "x_orig_m11",
            "x_hat_m11",
            "y_obs",
            "Apy_01",
            "full_psnr_db",
            "full_vgg_lpips",
            "hole_psnr_db",
            "hole_vgg_lpips",
            "observed_max_abs_diff",
            "restored_path",
        ):
            assert k in rec
        assert os.path.exists(rec["restored_path"])
        assert np.isfinite(rec["full_psnr_db"])
        assert np.isfinite(rec["full_vgg_lpips"])
        assert np.isfinite(rec["hole_psnr_db"])
        assert np.isfinite(rec["hole_vgg_lpips"])

    # Determinism of seed + idx regardless of subset order
    args_b = _make_args("run_0_only", sample_ids=[0], context_mode="telea")
    diff_b = Diffusion(args_b, cfg, device=device)
    diff_b.sample(logger=None)
    rec_a0 = diff_a.last_run_records[1]
    rec_b0 = diff_b.last_run_records[0]
    assert rec_a0["index"] == 0 and rec_b0["index"] == 0
    assert (rec_a0["x_hat_m11"] - rec_b0["x_hat_m11"]).abs().max().item() == 0.0
    assert (rec_a0["y_obs"] - rec_b0["y_obs"]).abs().max().item() == 0.0
    assert rec_a0["full_psnr_db"] == pytest.approx(rec_b0["full_psnr_db"])
    assert rec_a0["hole_psnr_db"] == pytest.approx(rec_b0["hole_psnr_db"])

    # Verify CLIP negative control diverges from target prompt end-to-end in Diffusion
    args_pos = _make_args("run_clip_pos", sample_ids=[0], context_mode="clip", clip_negative_control=False)
    args_neg = _make_args("run_clip_neg", sample_ids=[0], context_mode="clip", clip_negative_control=True)
    diff_pos = Diffusion(args_pos, cfg, device=device)
    diff_neg = Diffusion(args_neg, cfg, device=device)
    diff_pos.sample(logger=None)
    diff_neg.sample(logger=None)
    assert (
        diff_pos.last_run_records[0]["x_hat_m11"] - diff_neg.last_run_records[0]["x_hat_m11"]
    ).abs().max().item() > 1e-3

    # Non-finite model output triggers RuntimeError
    class _NanUNet(_DeterministicDummyUNet):
        def forward(self, x, t):
            return torch.full_like(x, float("nan"))

    monkeypatch.setattr(diffusion_mod, "create_model", lambda **kwargs: _NanUNet())
    diff_nan = Diffusion(_make_args("run_nan", sample_ids=[0], context_mode="none"), cfg, device=device)
    with pytest.raises(RuntimeError, match="Non-finite"):
        diff_nan.sample(logger=None)


def test_cli_parse_args_defaults(monkeypatch):
    args = main_mod.parse_cli_args([])
    assert args.context_mode == "none"
    assert args.telea_method == "telea"
    assert args.telea_radius == 3.0
    assert args.spatial_sigma == 8.0
    assert args.eta_hole == 0.25
    assert args.eta_obs == 0.02
    assert args.clip_model == "ViT-B-32"
    assert args.clip_pretrained == "laion2b_s34b_b79k"
    assert args.clip_prompt == "a photograph of a cat"
    assert args.clip_neg_prompt == "a photograph of a red fire hydrant in outer space"
    assert args.clip_negative_control is False
    assert args.clip_scale == 0.25
    assert args.clip_steps == 1
    assert args.sample_ids is None
    assert args.num_workers is None

    args_comma = main_mod.parse_cli_args(["--sample_ids", "0,3,7", "--num_workers", "0"])
    assert args_comma.sample_ids == [0, 3, 7]
    assert args_comma.num_workers == 0

    args_space = main_mod.parse_cli_args(["--sample_ids", "0", "3", "7"])
    assert args_space.sample_ids == [0, 3, 7]

    bad_invocations = [
        ["--telea_radius", "0"],
        ["--spatial_sigma", "-1"],
        ["--eta_hole", "1.5"],
        ["--eta_obs", "-0.1"],
        ["--clip_scale", "-0.5"],
        ["--clip_steps", "-1"],
        ["--num_workers", "-1"],
        ["--sample_ids", ""],
        ["--sample_ids", "-1"],
        ["--sample_ids", "0", "--subset_start", "0", "--subset_end", "1"],
    ]
    for bad in bad_invocations:
        with pytest.raises(SystemExit) as exc:
            main_mod.parse_cli_args(bad)
        assert exc.value.code == 2

    # main() returns 1 on exception
    monkeypatch.setattr(main_mod, "parse_args_and_config", lambda: (SimpleNamespace(), SimpleNamespace(), None))
    class _FailDiff:
        def __init__(self, *a, **kw):
            pass
        def sample(self, logger):
            raise RuntimeError("boom")
    monkeypatch.setattr(main_mod, "Diffusion", _FailDiff)
    assert main_mod.main() == 1
