from __future__ import annotations

from pathlib import Path
import numpy as np
from pyhdf.SD import SD, SDC


def encode_lai(lai: np.ndarray, scale: float = 0.01, valid_max: float = 10.0) -> np.ndarray:
    out = np.zeros(lai.shape, dtype="int16")
    ok = np.isfinite(lai)
    out[ok] = np.rint(np.clip(lai[ok], 0, valid_max) / scale).astype("int16")
    return out


def write_hdf4(path: str | Path, lai: np.ndarray, rmse: np.ndarray | None = None, nobs: np.ndarray | None = None, ul=(-179.9956818, 89.2229156), pixel=0.07272727) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    hdf = SD(str(path), SDC.WRITE | SDC.CREATE | SDC.TRUNC)
    try:
        sds = hdf.create("LAI", SDC.INT16, lai.shape)
        sds[:] = encode_lai(lai)
        sds.setrange(0, 1000)
        sds.setfillvalue(0)
        sds.attr("scale_factor").set(SDC.FLOAT32, 0.01)
        sds.attr("valid_range").set(SDC.INT16, [0, 1000])
        if rmse is not None:
            q = hdf.create("LAI_RMSE", SDC.FLOAT32, rmse.shape)
            q[:] = np.asarray(rmse, dtype="float32")
        if nobs is not None:
            q = hdf.create("CALIBRATION_NOBS", SDC.INT16, nobs.shape)
            q[:] = np.asarray(nobs, dtype="int16")
        hdf.attr("PROJ_UL_XY").set(SDC.FLOAT32, list(ul))
        hdf.attr("PIXEL_SIZE").set(SDC.FLOAT32, float(pixel))
        hdf.attr("Projection").set(SDC.CHAR8, "GCTP_GEO")
    finally:
        hdf.end()
