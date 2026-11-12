"""Train-only robust EXIF preprocessing with explicit missingness masks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


EXIF_COLUMNS = (
    "Focal length",
    "F-number",
    "Exposure time",
    "ISO",
    "Brightness.1",
    "Flash",
    "Time0",
    "Time1",
)
LOG1P_COLUMNS = {"Focal length", "F-number", "Exposure time", "ISO"}


def _numeric(frame: pd.DataFrame) -> np.ndarray:
    return frame.loc[:, EXIF_COLUMNS].apply(pd.to_numeric, errors="coerce").to_numpy(float)


def _domain_transform(values: np.ndarray) -> np.ndarray:
    result = values.copy()
    for index, column in enumerate(EXIF_COLUMNS):
        if column in LOG1P_COLUMNS:
            valid = np.isfinite(result[:, index]) & (result[:, index] >= 0)
            result[valid, index] = np.log1p(result[valid, index])
            result[~valid, index] = np.nan
    return result


@dataclass
class ExifPreprocessor:
    median: np.ndarray | None = None
    scale: np.ndarray | None = None

    def fit(self, train_frame: pd.DataFrame) -> "ExifPreprocessor":
        values = _domain_transform(_numeric(train_frame))
        if np.isnan(values).all(axis=0).any():
            raise ValueError("At least one EXIF field has no observed training values")
        self.median = np.nanmedian(values, axis=0)
        q25, q75 = np.nanpercentile(values, [25, 75], axis=0)
        scale = q75 - q25
        self.scale = np.where(scale > 1e-12, scale, 1.0)
        return self

    def transform(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if self.median is None or self.scale is None:
            raise RuntimeError("ExifPreprocessor must be fit on training data")
        values = _domain_transform(_numeric(frame))
        observed = np.isfinite(values)
        standardized = (values - self.median) / self.scale
        standardized[~observed] = 0.0
        return standardized.astype(np.float32), observed.astype(np.float32)

    def fit_transform(self, train_frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        return self.fit(train_frame).transform(train_frame)

    def state_dict(self) -> dict[str, list[float]]:
        if self.median is None or self.scale is None:
            raise RuntimeError("ExifPreprocessor is not fit")
        return {"columns": list(EXIF_COLUMNS), "median": self.median.tolist(), "scale": self.scale.tolist()}

    @classmethod
    def from_state_dict(cls, state: dict) -> "ExifPreprocessor":
        if tuple(state["columns"]) != EXIF_COLUMNS:
            raise ValueError("EXIF column order differs from project contract")
        return cls(np.asarray(state["median"], dtype=float), np.asarray(state["scale"], dtype=float))
