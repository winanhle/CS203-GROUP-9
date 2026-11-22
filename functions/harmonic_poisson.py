"""Isotropic Harmonic (Laplace) inpainting and Poisson Residual Back-Projection for CM4IR."""

import hashlib
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
import torch.nn.functional as F


def compute_max_hole_radius(mask_missing_2d: np.ndarray) -> float:
    mask = np.asarray(mask_missing_2d, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"mask_missing_2d must be 2D, got shape {mask.shape}")
    if not mask.any() or mask.all():
        return 0.0
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    return float(dist.max())


class HarmonicPoissonSolver:
    def __init__(self, mask_missing_2d: np.ndarray) -> None:
        mask = np.asarray(mask_missing_2d, dtype=bool)
        if mask.ndim != 2:
            raise ValueError(f"mask_missing_2d must be 2D, got shape {mask.shape}")

        self.mask = mask
        self.H, self.W = mask.shape
        self.hy, self.hx = np.where(mask)
        self.n_hole = int(self.hy.shape[0])
        self.max_hole_radius = compute_max_hole_radius(mask)

        self.hole_index = np.full((self.H, self.W), -1, dtype=np.int64)
        if self.n_hole > 0:
            self.hole_index[self.hy, self.hx] = np.arange(self.n_hole, dtype=np.int64)

        self.obs_links: List[Tuple[int, int, int]] = []
        self.obs_k = np.empty(0, dtype=np.int64)
        self.obs_y = np.empty(0, dtype=np.int64)
        self.obs_x = np.empty(0, dtype=np.int64)

        self.A: sp.csc_matrix = sp.csc_matrix((0, 0), dtype=np.float64)
        self.lu: Optional[Any] = None

        if self.n_hole == 0 or not np.any(~mask):
            return

        rows: List[int] = []
        cols: List[int] = []
        data: List[float] = []

        for k in range(self.n_hole):
            y = int(self.hy[k])
            x = int(self.hx[k])
            degree = 0
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ny = y + dy
                nx = x + dx
                if 0 <= ny < self.H and 0 <= nx < self.W:
                    degree += 1
                    if mask[ny, nx]:
                        nk = int(self.hole_index[ny, nx])
                        rows.append(k)
                        cols.append(nk)
                        data.append(-1.0)
                    else:
                        self.obs_links.append((k, ny, nx))
            rows.append(k)
            cols.append(k)
            data.append(float(degree))

        self.A = sp.csc_matrix(
            (
                np.asarray(data, dtype=np.float64),
                (np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)),
            ),
            shape=(self.n_hole, self.n_hole),
        )
        self.A.sum_duplicates()
        self.lu = spla.splu(self.A)

        if self.obs_links:
            self.obs_k = np.asarray([t[0] for t in self.obs_links], dtype=np.int64)
            self.obs_y = np.asarray([t[1] for t in self.obs_links], dtype=np.int64)
            self.obs_x = np.asarray([t[2] for t in self.obs_links], dtype=np.int64)

    def solve_batch(self, field: torch.Tensor) -> torch.Tensor:
        if field.dim() != 4:
            raise ValueError(f"field must be 4D (B, C, H, W), got shape {field.shape}")
        B, C, H, W = field.shape
        if H != self.H or W != self.W:
            raise ValueError(
                f"field spatial size {(H, W)} does not match mask size {(self.H, self.W)}"
            )
        if self.lu is None or self.n_hole == 0:
            return field.clone()

        device = field.device
        dtype = field.dtype
        field_np = field.detach().cpu().to(torch.float64).numpy()
        out_np = field_np.copy()

        hy = self.hy
        hx = self.hx
        obs_k = self.obs_k
        obs_y = self.obs_y
        obs_x = self.obs_x

        for b in range(B):
            for c in range(C):
                rhs = np.zeros(self.n_hole, dtype=np.float64)
                if obs_k.size > 0:
                    np.add.at(rhs, obs_k, field_np[b, c, obs_y, obs_x])
                u = self.lu.solve(rhs)
                out_np[b, c, hy, hx] = u

        out = torch.from_numpy(out_np).to(device=device, dtype=dtype)
        return out


_SOLVER_CACHE: Dict[Tuple[int, int, str], HarmonicPoissonSolver] = {}


def get_harmonic_solver(
    mask: Union[np.ndarray, torch.Tensor], image_size: int
) -> HarmonicPoissonSolver:
    if isinstance(mask, torch.Tensor):
        mask_arr = mask.detach().cpu().numpy()
    else:
        mask_arr = np.asarray(mask)
    mask_2d = mask_arr.reshape(image_size, image_size)
    missing = (mask_2d == 0)
    digest = hashlib.sha256(missing.tobytes()).hexdigest()
    key = (int(image_size), int(image_size), digest)
    solver = _SOLVER_CACHE.get(key)
    if solver is None:
        solver = HarmonicPoissonSolver(missing)
        _SOLVER_CACHE[key] = solver
    return solver


def harmonic_inpainting(
    y: torch.Tensor,
    mask: Union[np.ndarray, torch.Tensor],
    in_channels: int,
    image_size: int,
) -> torch.Tensor:
    original_shape = tuple(y.shape)
    y_4d = y.reshape(-1, in_channels, image_size, image_size)
    solver = get_harmonic_solver(mask, image_size)
    out_4d = solver.solve_batch(y_4d)
    return out_4d.reshape(original_shape)


def poisson_residual_bp(
    x0_t: torch.Tensor,
    y_0_4d: torch.Tensor,
    mask: Union[np.ndarray, torch.Tensor],
    image_size: int,
    solver: Optional[HarmonicPoissonSolver] = None,
) -> torch.Tensor:
    if solver is None:
        solver = get_harmonic_solver(mask, image_size)

    if y_0_4d.dim() == 3:
        y_0_4d = y_0_4d.unsqueeze(0)
    if x0_t.dim() == 3:
        x0_t = x0_t.unsqueeze(0)

    B, C, H, W = y_0_4d.shape
    if x0_t.shape != (B, C, H, W):
        x0_t = x0_t.reshape(B, C, H, W)

    if isinstance(mask, torch.Tensor):
        mask_arr = mask.detach().cpu().numpy()
    else:
        mask_arr = np.asarray(mask)
    obs_mask_np = (mask_arr.reshape(image_size, image_size) != 0).astype(np.float64)
    obs_mask = (
        torch.from_numpy(obs_mask_np)
        .to(device=y_0_4d.device, dtype=y_0_4d.dtype)
        .view(1, 1, H, W)
    )

    resid = obs_mask * (x0_t - y_0_4d)
    return solver.solve_batch(resid)


def gaussian_blur_2d(x: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0.0:
        return x
    radius = max(1, int(round(3.0 * sigma)))
    coords = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    k1d = torch.exp(-0.5 * (coords / float(sigma)) ** 2)
    k1d = k1d / k1d.sum()
    c = x.shape[1]
    kx = k1d.view(1, 1, 1, -1).expand(c, 1, 1, -1)
    ky = k1d.view(1, 1, -1, 1).expand(c, 1, -1, 1)
    out = F.pad(x, (radius, radius, 0, 0), mode="reflect")
    out = F.conv2d(out, kx, groups=c)
    out = F.pad(out, (0, 0, radius, radius), mode="reflect")
    return F.conv2d(out, ky, groups=c)


def saliency_bridge_inpainting(
    y: torch.Tensor,
    mask: Union[np.ndarray, torch.Tensor],
    in_channels: int,
    image_size: int,
    gamma: float = 5.0,
) -> torch.Tensor:
    original_shape = tuple(y.shape)
    y_4d = y.reshape(-1, in_channels, image_size, image_size)
    solver = get_harmonic_solver(mask, image_size)
    if gamma <= 0.0 or solver.max_hole_radius < 16.0 or solver.n_hole == 0:
        return solver.solve_batch(y_4d).reshape(original_shape)

    obs_np = (~solver.mask).astype(np.uint8)
    mask_t = torch.from_numpy(obs_np.astype(np.float32)).to(y_4d.device, y_4d.dtype).view(1, 1, image_size, image_size)
    eroded_np = cv2.erode(obs_np, np.ones((5, 5), np.uint8), iterations=1)
    if eroded_np.sum() == 0:
        eroded_np = obs_np
    eroded_t = torch.from_numpy(eroded_np.astype(np.float32)).to(y_4d.device, y_4d.dtype).view(1, 1, image_size, image_size)
    ring_np = (cv2.dilate(1 - obs_np, np.ones((7, 7), np.uint8)) * obs_np) > 0
    ring_t = torch.from_numpy(ring_np).to(y_4d.device)

    gray = (0.299 * y_4d[:, 0:1] + 0.587 * y_4d[:, 1:2] + 0.114 * y_4d[:, 2:3]) if in_channels == 3 else y_4d[:, 0:1]
    gray_sm = gaussian_blur_2d(gray * mask_t, 1.5) / gaussian_blur_2d(mask_t, 1.5).clamp(min=1e-4)
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=y_4d.device, dtype=y_4d.dtype).view(1, 1, 3, 3) / 8.0
    gx = F.conv2d(F.pad(gray_sm, (1, 1, 1, 1), mode="reflect"), kx)
    gy = F.conv2d(F.pad(gray_sm, (1, 1, 1, 1), mode="reflect"), kx.transpose(2, 3))
    grad_mag = torch.sqrt(gx * gx + gy * gy + 1e-8)

    e_blur3 = gaussian_blur_2d(eroded_t, 3.0).clamp(min=1e-4)
    mean_rgb = gaussian_blur_2d(y_4d * eroded_t, 3.0) / e_blur3
    var_rgb = gaussian_blur_2d(((y_4d - mean_rgb) ** 2) * eroded_t, 3.0) / e_blur3
    local_std = torch.sqrt(var_rgb.mean(dim=1, keepdim=True).clamp(min=0.0) + 1e-8)
    sal_sm = gaussian_blur_2d((grad_mag + 1.5 * local_std) * eroded_t, 4.0) / gaussian_blur_2d(eroded_t, 4.0).clamp(min=1e-4)

    w_list = []
    for b in range(y_4d.shape[0]):
        vals = sal_sm[b, 0][ring_t] if ring_np.any() else sal_sm[b, 0].reshape(-1)
        p90 = torch.quantile(vals, 0.90).clamp(min=1e-4)
        w_list.append(1.0 + float(gamma) * (sal_sm[b : b + 1] / p90).clamp(0.0, 1.5))
    w_sal = torch.cat(w_list, dim=0)
    num = solver.solve_batch(y_4d * w_sal)
    den = solver.solve_batch(w_sal).clamp(min=1e-6)
    out = torch.where(mask_t > 0.5, y_4d, torch.clamp(num / den, -1.0, 1.0))
    return out.reshape(original_shape)