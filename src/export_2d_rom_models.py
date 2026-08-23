"""Export selected-rank POD and exact-DMD models as reproducible numerical arrays."""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
from rom2d_methods import exact_dmd, require_uniform_times

ROOT = Path("/mnt/f/Jahidul_Islam/DetFOAM_2DWedge_NonReact_For_ROM/ROM_DMD_2D_Journal_Work")
FIELDS = ("p", "rho", "Ux", "Uy")
RANK = 9

def main():
    data = np.load(ROOT / "02_data_extraction" / "snapshots_2d_p_rho_U.npz")
    time_s = data["time_s"]
    dt = require_uniform_times(time_s)
    manifest = []
    (ROOT / "04_pod" / "models").mkdir(parents=True, exist_ok=True)
    (ROOT / "05_standard_dmd" / "models").mkdir(parents=True, exist_ok=True)
    for field in FIELDS:
        q = data[field]
        mean = q.mean(axis=1, keepdims=True)
        centered = q - mean
        u, s, vh = np.linalg.svd(centered, full_matrices=False)
        r = min(RANK, len(s))
        pod_modes = u[:, :r]
        pod_coefficients = s[:r, None] * vh[:r, :]
        np.savez_compressed(
            ROOT / "04_pod" / "models" / f"{field}_pod_rank_{r}.npz",
            mean=mean, modes=pod_modes, coefficients=pod_coefficients,
            singular_values=s, time_s=time_s, xy_m=data["xy_m"], field=np.array(field),
        )
        dmd = exact_dmd(q, dt, r)
        indices = np.arange(len(time_s))
        dmd_coefficients = dmd.amplitudes[:, None] * dmd.eigenvalues[:, None] ** indices
        np.savez_compressed(
            ROOT / "05_standard_dmd" / "models" / f"{field}_dmd_rank_{dmd.rank}.npz",
            mean=dmd.mean, modes=dmd.modes, eigenvalues=dmd.eigenvalues,
            amplitudes=dmd.amplitudes, temporal_coefficients=dmd_coefficients,
            singular_values=dmd.singular_values, time_s=time_s, dt_s=np.array(dmd.dt_s),
            xy_m=data["xy_m"], field=np.array(field),
        )
        manifest.append({
            "field": field, "rank": dmd.rank, "n_spatial_dofs": q.shape[0],
            "n_snapshots": q.shape[1], "dt_s": dt,
            "pod_model": f"04_pod/models/{field}_pod_rank_{r}.npz",
            "dmd_model": f"05_standard_dmd/models/{field}_dmd_rank_{dmd.rank}.npz",
            "pod_formula": "q(t) = mean + modes @ coefficients(t)",
            "dmd_formula": "q(t_k) = mean + Re[modes @ (amplitudes * eigenvalues**k)]",
        })
    with (ROOT / "10_tables" / "rom_model_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    print(f"Exported {len(manifest)} POD and DMD models at rank {RANK}.")

if __name__ == "__main__":
    main()
