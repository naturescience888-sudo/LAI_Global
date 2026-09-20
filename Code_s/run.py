from __future__ import annotations

from pathlib import Path
import argparse
import yaml

from globmap_io import read_tile, open_period_stack
from biophysical import modis_clear_mask, vegetation_indices, invert_lut, clumping_correct, gapfill_lai, denoise_lai
from fusion import fit_pixelwise, fit_biome_fallback, reconstruct, blend_overlap
from export import write_hdf4


def run(config_path: Path) -> None:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    root = Path(cfg["outputs"]["intermediate_root"])
    root.mkdir(parents=True, exist_ok=True)
    modis = read_tile(Path(cfg["inputs"]["mod09a1_root"]) / "stack.nc", ["sur_refl_b01", "sur_refl_b02", "state_1km", "state_2km"])
    biome = read_tile(Path(cfg["inputs"]["mcd12q1_root"]) / "biome.nc", ["biome"])["biome"]
    ci = read_tile(Path(cfg["inputs"]["clumping_index_root"]) / "clumping.nc", ["clumping_index"])["clumping_index"]
    clear = modis_clear_mask(modis["state_1km"], modis["state_2km"])
    vi = vegetation_indices(modis["sur_refl_b01"].where(clear), modis["sur_refl_b02"].where(clear))
    effective = invert_lut(vi["sr"], biome, Path(cfg["inputs"]["biome_lut"]))
    modis_true = denoise_lai(gapfill_lai(clumping_correct(effective, ci), cfg["processing"]["cloud_gap_max_periods"]))
    modis_true.to_dataset(name="lai_true").to_netcdf(root / "modis_true_lai.nc")
    avhrr = read_tile(Path(cfg["inputs"]["gimms_ndvi_root"]) / "overlap.nc", ["ndvi"])["ndvi"]
    overlap = fit_pixelwise(avhrr, modis_true, cfg["processing"]["minimum_overlap_observations"])
    overlap.intercept.to_netcdf(root / "calibration_intercept.nc")
    overlap.slope.to_netcdf(root / "calibration_slope.nc")
    overlap.rmse.to_netcdf(root / "calibration_rmse.nc")
    overlap.nobs.to_netcdf(root / "calibration_nobs.nc")
    avhrr_hist = read_tile(Path(cfg["inputs"]["gimms_ndvi_root"]) / "historical.nc", ["ndvi"])["ndvi"]
    fallback = fit_biome_fallback(avhrr, modis_true, biome)
    hist_lai = reconstruct(avhrr_hist, overlap, biome, fallback)
    hist_lai.to_dataset(name="lai").to_netcdf(root / "historical_lai.nc")
    for year in range(cfg["periods"]["avhrr_start"], cfg["periods"]["modis_end"] + 1):
        source = hist_lai if year <= cfg["periods"]["avhrr_end"] else modis_true
        for index in range(source.sizes["time"]):
            write_hdf4(Path(cfg["outputs"]["final_hdf_root"]) / f"GlobMapLAIV3.A{year:04d}{index + 1:03d}.Global.hdf", source.isel(time=index).values)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    run(parser.parse_args().config)
