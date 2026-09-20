from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import numpy as np
import xarray as xr
from pyhdf.SD import SD, SDC


@dataclass
class Raster:
    values: np.ndarray
    longitude: np.ndarray
    latitude: np.ndarray
    attrs: dict


def read_hdf4(path: str | Path, variable: str = "LAI") -> Raster:
    path = Path(path)
    hdf = SD(str(path), SDC.READ)
    try:
        sds = hdf.select(variable)
        values = np.asarray(sds[:])
        attrs = {**hdf.attributes(), **sds.attributes()}
    finally:
        hdf.end()
    scale = float(attrs.get("scale_factor", 1.0))
    fill = attrs.get("_FillValue", 0)
    valid_range = attrs.get("valid_range", (0, 1000))
    values = values.astype("float32")
    values[(values == fill) | (values < valid_range[0]) | (values > valid_range[1])] = np.nan
    values *= scale
    ul_lon, ul_lat = attrs.get("PROJ_UL_XY", (-179.9956818, 89.2229156))
    dx = float(attrs.get("PIXEL_SIZE", 0.07272727))
    ny, nx = values.shape[-2:]
    lon = ul_lon + (np.arange(nx) + 0.5) * dx
    lat = ul_lat - (np.arange(ny) + 0.5) * dx
    attrs.update({"source": str(path), "scale_factor_applied": scale, "fill_value": fill, "valid_range": valid_range})
    return Raster(values, lon, lat, attrs)


def read_tile(path: str | Path, variables: list[str]) -> xr.Dataset:
    ds = xr.open_dataset(path, decode_times=True, mask_and_scale=True)
    return ds[variables]


def parse_yyyy_doy(path: str | Path) -> tuple[int, int]:
    match = re.search(r"A(\d{4})(\d{3})", Path(path).name)
    if match is None:
        raise ValueError(path)
    return int(match.group(1)), int(match.group(2))


def target_grid(attrs: dict, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    ul_lon, ul_lat = attrs.get("PROJ_UL_XY", (-179.9956818, 89.2229156))
    dx = float(attrs.get("PIXEL_SIZE", 0.07272727))
    ny, nx = shape
    return ul_lon + (np.arange(nx) + 0.5) * dx, ul_lat - (np.arange(ny) + 0.5) * dx


def open_period_stack(paths: list[str | Path], variable: str) -> xr.DataArray:
    rasters = [read_hdf4(path, variable) for path in paths]
    times = [parse_yyyy_doy(path) for path in paths]
    labels = [f"{year:04d}-{doy:03d}" for year, doy in times]
    return xr.DataArray(np.stack([r.values for r in rasters]), dims=("time", "latitude", "longitude"), coords={"time": labels, "latitude": rasters[0].latitude, "longitude": rasters[0].longitude}, name=variable)
