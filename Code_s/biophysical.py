from __future__ import annotations

from pathlib import Path
import numpy as np
import xarray as xr
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d


def modis_clear_mask(state_1km: xr.DataArray, state_2km: xr.DataArray) -> xr.DataArray:
    a = state_1km.astype("uint16")
    b = state_2km.astype("uint16")
    cloud = a & 3
    shadow = (a >> 2) & 1
    aerosol = (b >> 6) & 3
    cirrus = (b >> 8) & 3
    return ((cloud == 0) & (shadow == 0) & (aerosol <= 1) & (cirrus == 0)).rename("clear")


def vegetation_indices(red: xr.DataArray, nir: xr.DataArray) -> xr.Dataset:
    den = (red + nir).where((red + nir) > 1e-5)
    ndvi = ((nir - red) / den).clip(-1, 1)
    sr = ((1 + ndvi) / (1 - ndvi).clip(min=0.02)).clip(0, 50)
    rsr = ((nir - red) / (nir + red + 0.5 * red)).clip(-1, 1)
    return xr.Dataset({"ndvi": ndvi, "sr": sr, "rsr": rsr})


def invert_lut(sr: xr.DataArray, biome: xr.DataArray, lut_path: str | Path) -> xr.DataArray:
    table = np.genfromtxt(lut_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    out = np.full(sr.shape, np.nan, dtype="float32")
    values = sr.values
    biome_values = biome.values
    for key in np.unique(table["biome"]):
        rows = table[table["biome"] == key]
        order = np.argsort(rows["sr"])
        x = rows["sr"][order].astype("float32")
        y = rows["lai_effective"][order].astype("float32")
        mask = biome_values == key
        out[mask] = np.interp(values[mask], x, y, left=np.nan, right=np.nan)
    return xr.DataArray(out, dims=sr.dims, coords=sr.coords, name="lai_effective")


def clumping_correct(effective: xr.DataArray, clumping: xr.DataArray, maximum: float = 15) -> xr.DataArray:
    omega = clumping.clip(0.05, 1.0)
    return (effective / omega).clip(0, maximum).rename("lai_true")


def _spline_cap(y: np.ndarray, max_gap: int, cap_window: int) -> np.ndarray:
    y = y.astype("float32", copy=True)
    ok = np.isfinite(y)
    if ok.sum() < 4:
        return y
    t = np.arange(y.size, dtype="float32")
    f = CubicSpline(t[ok], y[ok], bc_type="natural", extrapolate=False)
    missing = ~ok
    edges = np.flatnonzero(np.diff(np.r_[False, missing, False]))
    for left, right in edges.reshape(-1, 2):
        if right - left > max_gap:
            continue
        y[left:right] = f(t[left:right])
        lo = max(0, left - cap_window)
        hi = min(y.size, right + cap_window + 1)
        y[left:right] = np.clip(y[left:right], np.nanmin(y[lo:hi]), np.nanmax(y[lo:hi]))
    return y


def gapfill_lai(lai: xr.DataArray, max_gap: int = 4, cap_window: int = 3) -> xr.DataArray:
    result = xr.apply_ufunc(_spline_cap, lai, input_core_dims=[["time"]], output_core_dims=[["time"]], kwargs={"max_gap": max_gap, "cap_window": cap_window}, vectorize=True, dask="parallelized", output_dtypes=["float32"])
    return result.transpose(*lai.dims).rename("lai_true_filled")


def denoise_lai(lai: xr.DataArray, sigma: float = 0.75) -> xr.DataArray:
    result = xr.apply_ufunc(lambda x: gaussian_filter1d(x, sigma=sigma, axis=-1, mode="nearest"), lai, dask="parallelized", output_dtypes=["float32"])
    return result.rename("lai_denoised")
