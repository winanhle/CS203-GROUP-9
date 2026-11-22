"""Isotropic Harmonic (Laplace) inpainting and Poisson Residual Back-Projection for CM4IR."""

import hashlib
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch


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