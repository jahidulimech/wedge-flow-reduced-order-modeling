"""Create time- and space-resolved validation data from stored rank-9 models."""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/mnt/f/Jahidul_Islam/DetFOAM_2DWedge_NonReact_For_ROM/ROM_DMD_2D_Journal_Work")
DPI = 220

def reconstruct(field, q):
    pod = np.load(ROOT / "04_pod" / "models" / f"{field}_pod_rank_9.npz")
    pod_rec = pod["mean"] + pod["modes"] @ pod["coefficients"]
    dmd = np.load(ROOT / "05_standard_dmd" / "models" / f"{field}_dmd_rank_9.npz")
    idx = np.arange(q.shape[1])
    coeff = dmd["amplitudes"][:, None] * dmd["eigenvalues"][:, None] ** idx
    dmd_rec = dmd["mean"] + np.real(dmd["modes"] @ coeff)
    return pod_rec, dmd_rec

def save(name):
    for ext in ("png", "pdf"):
        plt.savefig(ROOT / "09_figures" / f"{name}.{ext}", dpi=DPI, bbox_inches="tight")
    plt.close()

def main():
    data = np.load(ROOT / "02_data_extraction" / "snapshots_2d_p_rho_U.npz")
    xy, time_s = data["xy_m"], data["time_s"]
    fields = ("p", "rho", "Ux", "Uy")
    units = {"p": "Pa", "rho": "kg m^-3", "Ux": "m s^-1", "Uy": "m s^-1"}
    rows = []
    arrays = {"xy_m": xy, "time_s": time_s}
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.5), sharex=True)
    for ax, field in zip(axes.ravel(), fields):
        q = data[field]
        pod_rec, dmd_rec = reconstruct(field, q)
        for method, rec in (("pod", pod_rec), ("standard_dmd", dmd_rec)):
            err = rec - q
            spatial_rms = np.sqrt(np.mean(err**2, axis=1))
            arrays[f"{field}_{method}_spatial_rms"] = spatial_rms
            series = []
            for j, t in enumerate(time_s):
                l2 = np.linalg.norm(err[:, j])
                reference_l2 = np.linalg.norm(q[:, j])
                rmse = np.sqrt(np.mean(err[:, j]**2))
                max_abs = np.max(np.abs(err[:, j]))
                series.append(l2 / reference_l2)
                rows.append({"field": field, "method": method, "time_s": float(t),
                             "relative_l2": float(l2 / reference_l2), "rmse": float(rmse),
                             "max_abs": float(max_abs)})
            ax.semilogy(time_s * 1e6, series, marker="o", ms=3, label=method)
        ax.set_title(field)
        ax.set_ylabel("relative L2 error")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    for ax in axes[-1, :]:
        ax.set_xlabel("time [us]")
    fig.suptitle("Time-resolved reconstruction error for stored rank-9 models")
    save("time_resolved_reconstruction_error")

    np.savez_compressed(ROOT / "07_validation" / "spatial_rms_reconstruction_errors.npz", **arrays)
    with (ROOT / "07_validation" / "time_dependent_reconstruction_errors.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    for method in ("pod", "standard_dmd"):
        fig, axes = plt.subplots(2, 2, figsize=(8.5, 7), sharex=True, sharey=True)
        for ax, field in zip(axes.ravel(), fields):
            value = arrays[f"{field}_{method}_spatial_rms"]
            item = ax.scatter(xy[:, 0], xy[:, 1], c=value, s=1, marker="s",
                              cmap="magma", linewidths=0, rasterized=True)
            ax.set_aspect("equal")
            ax.set_title(field)
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
            fig.colorbar(item, ax=ax, label=f"RMS error [{units[field]}]")
        fig.suptitle(f"Spatial RMS reconstruction error over 2 to 36 us: {method}, rank 9")
        save(f"spatial_rms_error_map_{method}_rank_9")
    (ROOT / "07_validation" / "VALIDATION.md").write_text(
        "# Time- and space-resolved validation\n\n"
        "time_dependent_reconstruction_errors.csv contains the error at every saved time. "
        "spatial_rms_reconstruction_errors.npz stores per-cell RMS errors over the verified interval. "
        "The corresponding maps use actual global cell centres and stored rank-9 modal reconstructions.\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(rows)} time-resolved rows and spatial RMS arrays.")

if __name__ == "__main__":
    main()
