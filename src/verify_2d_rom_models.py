"""Verify that exported rank-9 POD/DMD arrays reproduce their tabulated errors."""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np

ROOT = Path("/mnt/f/Jahidul_Islam/DetFOAM_2DWedge_NonReact_For_ROM/ROM_DMD_2D_Journal_Work")

def relative_error(reference, estimate):
    return float(np.linalg.norm(estimate - reference) / np.linalg.norm(reference))

def main():
    data = np.load(ROOT / "02_data_extraction" / "snapshots_2d_p_rho_U.npz")
    with (ROOT / "10_tables" / "reconstruction_errors.csv").open() as f:
        table = list(csv.DictReader(f))
    rows = []
    for field in ("p", "rho", "Ux", "Uy"):
        q = data[field]
        expected = {r["method"]: float(r["relative_frobenius"]) for r in table
                    if r["field"] == field and r["rank"] == "9" and r["selected_rank"] == "9"}
        pod = np.load(ROOT / "04_pod" / "models" / f"{field}_pod_rank_9.npz")
        pod_error = relative_error(q, pod["mean"] + pod["modes"] @ pod["coefficients"])
        dmd = np.load(ROOT / "05_standard_dmd" / "models" / f"{field}_dmd_rank_9.npz")
        index = np.arange(q.shape[1])
        coefficients = dmd["amplitudes"][:, None] * dmd["eigenvalues"][:, None] ** index
        dmd_error = relative_error(q, dmd["mean"] + np.real(dmd["modes"] @ coefficients))
        for method, computed in (("pod", pod_error), ("standard_dmd", dmd_error)):
            difference = abs(computed - expected[method])
            rows.append({"field": field, "method": method, "computed_relative_frobenius": computed,
                         "tabulated_relative_frobenius": expected[method], "absolute_difference": difference,
                         "pass": difference < 1e-12})
    with (ROOT / "13_reproducibility" / "model_reconstruction_verification.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if not all(row["pass"] for row in rows):
        raise RuntimeError("One or more stored-ROM reconstructions disagree with the tabulated errors")
    print("Stored ROM model verification passed for 8 field/model checks.")

if __name__ == "__main__":
    main()
