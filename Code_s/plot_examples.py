from __future__ import annotations

from pathlib import Path
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from globmap_io import read_hdf4


def snapshots(paths: list[Path], labels: list[str], output: Path) -> None:
    rasters = [read_hdf4(path) for path in paths]
    fig, axes = plt.subplots(1, len(rasters), figsize=(16, 4.8), constrained_layout=True)
    axes = np.atleast_1d(axes)
    image = None
    for ax, raster, label in zip(axes, rasters, labels):
        extent = [raster.longitude.min(), raster.longitude.max(), raster.latitude.min(), raster.latitude.max()]
        image = ax.imshow(raster.values, origin="upper", extent=extent, vmin=0, vmax=8, cmap="YlGn")
        ax.set_title(label)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(alpha=0.25, linewidth=0.4)
    fig.colorbar(image, ax=axes.tolist(), shrink=0.82, label="LAI")
    fig.suptitle("GLOBMAP LAI V3 regenerated product")
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def distributions(paths: list[Path], labels: list[str], output: Path, table_path: Path) -> None:
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    for path, label in zip(paths, labels):
        values = read_hdf4(path).values[::8, ::8]
        values = values[np.isfinite(values)]
        axes[0].hist(values, bins=np.linspace(0, 8, 41), density=True, histtype="step", linewidth=2, label=label)
        rows.append({"snapshot": label, "mean_lai": float(values.mean()), "median_lai": float(np.median(values)), "p95_lai": float(np.percentile(values, 95))})
    axes[0].set_xlabel("LAI")
    axes[0].set_ylabel("Density")
    axes[0].legend(frameon=False)
    x = np.arange(len(rows))
    axes[1].bar(x - 0.18, [r["mean_lai"] for r in rows], 0.36, label="Mean")
    axes[1].bar(x + 0.18, [r["p95_lai"] for r in rows], 0.36, label="P95")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("LAI")
    axes[1].legend(frameon=False)
    fig.suptitle("GLOBMAP LAI V3 regenerated product statistics")
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    with table_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("samples", nargs=3, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels = ["1981-07", "2001-01", "2023-12"]
    snapshots(args.samples, labels, args.output_dir / "global_lai_snapshots.png")
    distributions(args.samples, labels, args.output_dir / "lai_distribution_summary.png", args.output_dir / "preview_statistics.csv")
