from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import xarray as xr


@dataclass
class Calibration:
    intercept: xr.DataArray
    slope: xr.DataArray
    rmse: xr.DataArray
    nobs: xr.DataArray
    r2: xr.DataArray


def sr_from_ndvi(ndvi: xr.DataArray) -> xr.DataArray:
    return ((1 + ndvi) / (1 - ndvi).clip(min=0.02)).clip(0, 50).rename("sr_avhrr")


def _fit(x: np.ndarray, y: np.ndarray, max_iter: int, min_slope: float, max_slope: float) -> tuple[float, float, float, int, float]:
    ok = np.isfinite(x) & np.isfinite(y)
    n = int(ok.sum())
    if n < 3:
        return np.nan, np.nan, np.nan, n, np.nan
    x = x[ok].astype("float64")
    y = y[ok].astype("float64")
    X = np.stack([np.ones_like(x), x], axis=1)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    for _ in range(max_iter):
        residual = y - X @ beta
        scale = 1.4826 * np.median(np.abs(residual - np.median(residual))) + 1e-5
        u = residual / scale
        weights = np.where(np.abs(u) <= 1.345, 1.0, 1.345 / np.maximum(np.abs(u), 1e-6))
        beta = np.linalg.lstsq(X * weights[:, None], y * weights, rcond=None)[0]
    beta[1] = np.clip(beta[1], min_slope, max_slope)
    pred = X @ beta
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return float(beta[0]), float(beta[1]), float(np.sqrt(ss_res / n)), n, 1 - ss_res / max(ss_tot, 1e-8)


def fit_pixelwise(avhrr_ndvi: xr.DataArray, modis_lai: xr.DataArray, min_obs: int = 12) -> Calibration:
    x = sr_from_ndvi(avhrr_ndvi)
    a, b, rmse, nobs, r2 = xr.apply_ufunc(_fit, x, modis_lai, input_core_dims=[["time"], ["time"]], output_core_dims=[[], [], [], [], []], kwargs={"max_iter": 8, "min_slope": 0.0, "max_slope": 4.0}, vectorize=True, dask="parallelized", output_dtypes=["float32", "float32", "float32", "int16", "float32"])
    valid = nobs >= min_obs
    return Calibration(a.where(valid), b.where(valid), rmse.where(valid), nobs, r2.where(valid))


def fit_biome_fallback(avhrr_ndvi: xr.DataArray, modis_lai: xr.DataArray, biome: xr.DataArray) -> dict[int, tuple[float, float]]:
    sr = sr_from_ndvi(avhrr_ndvi).values
    y = modis_lai.values
    classes = biome.values
    result = {}
    for key in np.unique(classes[np.isfinite(classes)]):
        mask = classes == key
        a, b, _, n, _ = _fit(sr[:, mask], y[:, mask], 8, 0.0, 4.0)
        if n >= 12:
            result[int(key)] = (a, b)
    return result


def reconstruct(ndvi: xr.DataArray, model: Calibration, biome: xr.DataArray, fallback: dict[int, tuple[float, float]] | None = None) -> xr.DataArray:
    result = model.intercept + model.slope * sr_from_ndvi(ndvi)
    if fallback:
        raw = result.values
        s = sr_from_ndvi(ndvi).values
        missing = ~np.isfinite(raw)
        for key, (a, b) in fallback.items():
            use = missing & (biome.values == key)
            raw[use] = a + b * s[use]
        result = xr.DataArray(raw, coords=result.coords, dims=result.dims)
    return result.clip(0, 15).rename("lai_avhrr_reconstructed")


def blend_overlap(avhrr_lai: xr.DataArray, modis_lai: xr.DataArray, width: int = 6) -> xr.DataArray:
    n = avhrr_lai.sizes["time"]
    w = np.linspace(0, 1, n, dtype="float32")
    w = np.minimum(1, np.maximum(0, (w * n - (n - width)) / width))
    return ((1 - w) * avhrr_lai + w * modis_lai).rename("lai_fused")
