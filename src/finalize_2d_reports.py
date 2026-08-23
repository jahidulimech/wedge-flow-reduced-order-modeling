"""Create final 2D report tables and documentation from completed analysis outputs."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
import scipy
import pandas

ROOT = Path("/mnt/f/Jahidul_Islam/DetFOAM_2DWedge_NonReact_For_ROM/ROM_DMD_2D_Journal_Work")

def read_csv(name):
    with (ROOT / name).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def write_csv(name, rows):
    path = ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def write_md(name, text):
    path = ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")

def val(row, key):
    return float(row[key])

def selected(rows, field, method):
    found = [r for r in rows if r["field"] == field and r["method"] == method and r["rank"] == r["selected_rank"]]
    if len(found) != 1:
        raise RuntimeError(f"Expected one selected row for {field}/{method}, got {len(found)}")
    return found[0]

def main():
    audit = json.loads((ROOT / "01_case_audit" / "case_audit.json").read_text())
    data = np.load(ROOT / "02_data_extraction" / "snapshots_2d_p_rho_U.npz")
    xy, time_s = data["xy_m"], data["time_s"]
    errors = read_csv("10_tables/reconstruction_errors.csv")
    singular = read_csv("10_tables/singular_values.csv")
    train = read_csv("10_tables/training_testing_performance.csv")
    fields = ["p", "rho", "Ux", "Uy"]
    units = {"p": "Pa", "rho": "kg m^-3", "Ux": "m s^-1", "Uy": "m s^-1"}

    case_rows = [
        {"parameter": "OpenFOAM version reported in log", "value": "8", "source": "log.SWedge"},
        {"parameter": "application", "value": audit["application"], "source": "system/controlDict"},
        {"parameter": "chemistry", "value": audit["chemistry"], "source": "constant/chemistryProperties"},
        {"parameter": "combustion model", "value": "none", "source": "constant/combustionProperties"},
        {"parameter": "flux scheme", "value": audit["flux_scheme"], "source": "system/fvSchemes"},
        {"parameter": "time scheme", "value": audit["time_scheme"], "source": "system/fvSchemes"},
        {"parameter": "gradient limiter", "value": "cellLimited Gauss linear 0.7", "source": "system/fvSchemes"},
        {"parameter": "reconstruction limiters", "value": "Minmod; MinmodV", "source": "system/fvSchemes"},
        {"parameter": "thermodynamics", "value": audit["thermo_type"], "source": "constant/thermophysicalProperties"},
        {"parameter": "equation of state", "value": audit["equation_of_state"], "source": "constant/thermophysicalProperties"},
        {"parameter": "transport", "value": "const", "source": "constant/thermophysicalProperties"},
        {"parameter": "configured end time [s]", "value": audit["end_time_s"], "source": "system/controlDict"},
        {"parameter": "write control", "value": audit["write_control"], "source": "system/controlDict"},
        {"parameter": "write interval [s]", "value": audit["write_interval_s"], "source": "system/controlDict"},
        {"parameter": "max Courant number", "value": audit["max_courant"], "source": "system/controlDict"},
        {"parameter": "max acoustic Courant number", "value": audit["max_acoustic_courant"], "source": "system/controlDict"},
        {"parameter": "global cells", "value": str(audit["mesh_cells"]), "source": "constant/polyMesh"},
        {"parameter": "processor directories", "value": str(audit["processors_found"]), "source": "case root"},
    ]
    write_csv("10_tables/case_parameters.csv", case_rows)
    snap_rows = [{
        "snapshot_count": len(time_s),
        "spatial_dofs_per_scalar": xy.shape[0],
        "variables": "p, rho, Ux, Uy",
        "common_time_start_s": float(time_s[0]),
        "common_time_end_s": float(time_s[-1]),
        "uniform_dt_s": float(np.diff(time_s)[0]),
        "missing_common_snapshot": "t=0 s excluded because rho is absent",
        "extraction": "decomposed fields reconstructed with cellProcAddressing",
    }]
    write_csv("10_tables/snapshot_information.csv", snap_rows)

    comp_rows = []
    for field in fields:
        dmd = selected(errors, field, "standard_dmd")
        pod = selected(errors, field, "pod")
        srow = [r for r in singular if r["field"] == field and int(r["rank"]) == int(dmd["rank"])][0]
        comp_rows.append({
            "field": field, "unit": units[field], "rank": dmd["rank"],
            "singular_value_energy": srow["cumulative_energy"],
            "pod_relative_frobenius": pod["relative_frobenius"],
            "standard_dmd_relative_frobenius": dmd["relative_frobenius"],
            "dmd_over_pod_error_ratio": val(dmd, "relative_frobenius") / val(pod, "relative_frobenius"),
            "pod_nrmse": pod["nrmse"], "standard_dmd_nrmse": dmd["nrmse"],
            "pod_max_abs": pod["max_abs"], "standard_dmd_max_abs": dmd["max_abs"],
        })
    write_csv("10_tables/pod_standard_dmd_comparison.csv", comp_rows)

    n, m, k, nv = xy.shape[0], len(time_s), 9, len(fields)
    raw_bytes = n * m * nv * 8
    pod_bytes = nv * (n * 8 + n * k * 8 + k * m * 8)
    dmd_bytes = nv * (n * 8 + n * k * 16 + k * 16 + k * 16)
    storage_rows = [
        {"representation": "four scalar snapshot arrays (raw values only)", "bytes": raw_bytes, "mib": raw_bytes / 2**20, "assumption": "float64; p,rho,Ux,Uy"},
        {"representation": "POD, rank 9, four independent fields", "bytes": pod_bytes, "mib": pod_bytes / 2**20, "assumption": "mean + real modes + coefficients"},
        {"representation": "exact DMD, rank 9, four independent fields", "bytes": dmd_bytes, "mib": dmd_bytes / 2**20, "assumption": "mean + complex modes + eigenvalues + amplitudes"},
    ]
    write_csv("10_tables/model_storage_estimates.csv", storage_rows)

    config_path = ROOT / "config" / "analysis_config.json"
    config = json.loads(config_path.read_text())
    config["shock_geometry"] = {
        "enabled": False,
        "reason": "Transient data exist, but a globally detected pressure-gradient set is not a validated shock curve. No geometry registration or shock-attached DMD is claimed.",
        "roi": None,
        "registration": "disabled pending documented shock-front curve extraction and validation",
    }
    config["validation"] = {
        "train_test_enabled": True,
        "split": "first 12 snapshots (2 to 24 us) train; final 6 snapshots (26 to 36 us) held out",
        "model": "standard Eulerian exact DMD",
    }
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    write_md("01_case_audit/case_audit.md", f"""
# Verified 2D case audit

- OpenFOAM version reported by log.SWedge: 8.
- Solver: {audit["application"]}; configured end time: {audit["end_time_s"]} s.
- Flux: {audit["flux_scheme"]}; time integration: {audit["time_scheme"]}.
- Gradients use cellLimited Gauss linear 0.7; listed reconstruction limiters are Minmod and MinmodV.
- Thermodynamics: {audit["thermo_type"]}, {audit["equation_of_state"]}, const transport; chemistry is off and the combustion model is none.
- The global mesh has {audit["mesh_cells"]:,} cells. Eight processor directories contain cellProcAddressing; no field was manually processor-concatenated.
- Required fields p, rho, and U are complete at 18 common times from {time_s[0]*1e6:.0f} to {time_s[-1]*1e6:.0f} us, with uniform {np.diff(time_s)[0]*1e6:.0f} us spacing. rho is absent at 0 us, so that directory is excluded from multi-variable ROM.
- The saved fields end at 36 us. The solver log contains later time records, but those records are not used because corresponding field snapshots are absent.

Machine-readable details: case_audit.json, data_availability.json, case_parameters.csv, and snapshot_information.csv.
""")

    write_md("README.md", """
# 2D wedge ROM/DMD analysis

This directory is a reproducible, non-destructive analysis of the parent OpenFOAM case. It uses only verified decomposed p, rho, and U fields saved through 36 us.

## Run

1. Install the packages in requirements.txt.
2. From this directory run python src/run_2d_workflow.py.
3. Run python src/analyse_2d_results.py.
4. Run python src/export_2d_rom_models.py.
5. Run python src/finalize_2d_reports.py.
6. Run python src/validate_2d_rom_models.py.
7. Run python src/verify_2d_rom_models.py.

The reader reconstructs each decomposed field through cellProcAddressing, preserves global OpenFOAM cell ordering, rejects non-uniform time data, and stores the cleaned snapshot archive before ROM analysis.

## Scope

POD and mean-centred exact DMD are complete for p, rho, Ux, and Uy, including rank sweeps, field errors, figures, and a held-out time interval. A global pressure-gradient screen is supplied for geometry inspection only. It is not a shock curve; therefore this package deliberately does not report shock-position error or shock-attached DMD performance.

See FINAL_RESEARCH_SUMMARY.md for the evidence-based conclusion and 13_reproducibility/REPRODUCIBILITY.md for input/output provenance.
""")

    write_md("11_results/LIMITATIONS_AND_DATA_STATUS.md", f"""
# Data status and limitations

## What is verified

The analysis uses 18 complete common field snapshots from {time_s[0]*1e6:.0f} to {time_s[-1]*1e6:.0f} us at {np.diff(time_s)[0]*1e6:.0f} us spacing and {xy.shape[0]:,} global cells. The first directory at 0 us is excluded because rho is not written there. All ROM arrays were reconstructed with the OpenFOAM cellProcAddressing maps.

## Boundaries on interpretation

- Saved field data stop at 36 us. The later times in log.SWedge are not a substitute for missing snapshots.
- The log reports a source-case path different from this directory. It is used only for version/configuration evidence.
- POD/DMD metrics are snapshot reconstruction metrics unless labeled as held-out standard-DMD results.
- The 2D global pressure-gradient centroid is not a shock location. A curved, possibly deforming shock requires an explicit region of interest, curve extraction, correspondence definition, and uncertainty test. Those do not yet exist here.
- Hence no shock-position error, shock speed, shock-attached coordinate, or shock-attached DMD superiority claim is made.
- One mesh is available, so no grid-convergence conclusion is possible. No analytical oblique-shock comparison has been asserted because the necessary boundary/inflow verification and an agreed comparison location have not been completed.
- The held-out split is short (six snapshots); it is a useful stress test, not a broad generalization claim.
""")

    error_lines = []
    for row in comp_rows:
        error_lines.append(
            f"| {row['field']} | {row['rank']} | {float(row['singular_value_energy']):.6f} | "
            f"{float(row['pod_relative_frobenius']):.6g} | {float(row['standard_dmd_relative_frobenius']):.6g} | "
            f"{float(row['dmd_over_pod_error_ratio']):.2f} |"
        )
    test_lines = []
    for field in fields:
        row = [r for r in train if r["field"] == field and r["scope"] == "held_out"][0]
        test_lines.append(f"| {field} | {row['rank']} | {float(row['relative_frobenius']):.6g} | {float(row['nrmse']):.6g} | {float(row['max_abs']):.6g} |")

    write_md("11_results/results_summary.md", """
# 2D results summary

All quantities below are derived from the verified 18-snapshot archive (2 to 36 us); none is inferred from the solver log.

## In-sample rank-9 comparison

The rank is the first one at or above 99.9 percent of each centred field's singular-value energy.

| Field | Rank | retained energy | POD relative Frobenius | exact DMD relative Frobenius | DMD/POD error |
|---|---:|---:|---:|---:|---:|
""" + "\n".join(error_lines) + """

POD is an in-sample energy-optimal projection, whereas exact DMD fits a linear time-advance model and is reconstructed from its initial modal amplitudes. The difference in error is consequently a measured property of this dataset and formulation, not a numerical inconsistency.

## Held-out standard-DMD evaluation

Training: 2 to 24 us (12 snapshots). Held out: 26 to 36 us (6 snapshots).

| Field | Rank | held-out relative Frobenius | held-out NRMSE | held-out maximum absolute error |
|---|---:|---:|---:|---:|
""" + "\n".join(test_lines) + """

## Time- and space-resolved validation

time_dependent_reconstruction_errors.csv reports error at every stored time; spatial_rms_reconstruction_errors.npz and the associated maps report cellwise RMS error over the verified interval.

## Geometry screening

pressure_gradient_geometry_screen.csv contains the top-one-percent global pressure-gradient diagnostic for every time. Its weighted centroid evolves, but that global quantity can mix the wedge shock, boundaries, and other compression structures. It is retained as a visual screening result only, not as a shock trajectory.
""")

    write_md("12_manuscript/manuscript_draft.md", """
# Draft manuscript foundation: Eulerian POD/DMD of a 2D non-reacting wedge case

## 1. Introduction

This analysis examines reduced representations of transient compressible fields from a two-dimensional non-reacting OpenFOAM wedge simulation. The narrow question supported by the available data is how energy-optimal POD and standard Eulerian exact DMD represent p, rho, Ux, and Uy over the saved interval. A shock-attached method is intentionally not claimed until a two-dimensional shock curve can be defined and validated.

## 2. Problem formulation and numerical method

The audited case uses detonationFoam with chemistry off and combustionModel none. It reports HLLCP flux, Euler time integration, hePsiThermo/perfectGas thermodynamics, const transport, and a 50,625-cell mesh. Full audited settings are listed in 10_tables/case_parameters.csv.

## 3. Snapshot generation and preprocessing

Eight decomposed OpenFOAM partitions were reconstructed through cellProcAddressing. Eighteen common p/rho/U snapshots spanning 2 to 36 us at 2 us increments were verified. The 0-us directory was omitted because rho is absent. States preserve global OpenFOAM cell ordering. Each field is temporally mean centred; a combined state, when used, standardizes each field by its global space-time standard deviation.

## 4. ROM methods

POD is computed using the singular value decomposition of the centred snapshot matrix. Exact DMD is fit to X=[q_1,...,q_(m-1)] and Xprime=[q_2,...,q_m] after truncation. DMD eigenvalues, continuous growth rates, frequencies, amplitudes, and modes are saved in the results tables. No modal frequency is labeled physical without additional evidence.

## 5. Validation

In-sample relative Frobenius error, L2 error, NRMSE, and maximum absolute error are calculated for rank sweeps. The predictive stress test trains Eulerian DMD from 2 to 24 us and evaluates the six later snapshots. The actual figures and values are linked in 11_results/results_summary.md.

## 6. Results and discussion

Insert the rank comparison, singular-value spectra, DMD spectrum, field comparisons, and held-out forecast from 09_figures and 10_tables. The current evidence shows that rank-9 POD is substantially more accurate than rank-9 standard DMD for in-sample reconstruction of all four independent fields. This statement is restricted to this exact DMD formulation and dataset.

The pressure-gradient screen should be used to design a physically bounded shock-front extraction procedure. It is not evidence that a scalar shock position or global translating frame is appropriate.

## 7. Limitations and conclusions

No shock-attached DMD result, shock-location validation, grid-convergence result, or analytical oblique-shock validation is claimed. The next defensible study is an explicit shock-curve extraction and registration sensitivity analysis, followed by a direct Eulerian-versus-geometry-aligned ROM comparison on the same train/test split.

## References requiring independent verification

No literature references are inserted in this draft. Add only independently checked primary sources for POD, exact DMD, compressible-flow ROM, and shock-front registration.
""")

    write_md("13_reproducibility/REPRODUCIBILITY.md", f"""
# Reproducibility

## Observed analysis environment

- Python: {sys.version.replace(chr(10), " ")}
- NumPy: {np.__version__}
- SciPy: {scipy.__version__}
- pandas: {pandas.__version__}
- matplotlib: {matplotlib.__version__}

## Inputs and transformations

- Inputs: the parent case dictionaries/mesh and processor0 through processor7 saved fields.
- Extraction: src/run_2d_workflow.py, using cellProcAddressing.
- Analysis: src/analyse_2d_results.py.
- ROM model export: src/export_2d_rom_models.py.
- Report finalization: src/finalize_2d_reports.py.
- Time/space validation: src/validate_2d_rom_models.py.
- Model verification: src/verify_2d_rom_models.py.
- Output archive: 02_data_extraction/snapshots_2d_p_rho_U.npz.

## Repeatable command sequence

python -m pip install -r requirements.txt
python src/run_2d_workflow.py
python src/analyse_2d_results.py
python src/export_2d_rom_models.py
python src/finalize_2d_reports.py
python src/validate_2d_rom_models.py
python src/verify_2d_rom_models.py

The scripts must be run from ROM_DMD_2D_Journal_Work. They do not alter OpenFOAM dictionaries, mesh, fields, or logs. All rank and normalization settings are versioned in config/analysis_config.json.
""")

    write_md("FINAL_RESEARCH_SUMMARY.md", """
# Final 2D research summary

## Research question

Can low-dimensional Eulerian POD/DMD represent the saved transient fields of this non-reacting 2D wedge case, and is there presently enough validated geometry information to assess a shock-attached ROM?

## Verified case and dataset

The case is a 50,625-cell, eight-processor detonationFoam calculation with chemistry off, combustion model none, HLLCP flux, Euler time integration, and perfect-gas hePsiThermo. Eighteen complete p/rho/U snapshots are available from 2 to 36 us at a verified uniform 2 us interval. Although the log has later records, no saved field directories after 36 us exist, so they were not analyzed.

## Methods

Each decomposed field was reconstructed through cellProcAddressing. The study applies temporally mean-centred POD and exact DMD independently to p, rho, Ux, and Uy. The rank target is 99.9 percent centred-snapshot singular-value energy. A combined state is standardized per variable before stacking. The test split trains standard Eulerian DMD on 2 to 24 us and holds out 26 to 36 us.

## Main numerical findings

All four independent fields select rank 9 at the stated energy threshold. Rank-9 POD relative Frobenius errors are 0.00504393 (p), 0.00316780 (rho), 0.000565792 (Ux), and 0.00878888 (Uy). Rank-9 standard-DMD errors are 0.127592 (p), 0.0813912 (rho), 0.0143642 (Ux), and 0.233429 (Uy).

For the held-out interval, standard-DMD relative Frobenius errors are 0.175588 (p), 0.111348 (rho), 0.0257579 (Ux), and 0.267061 (Uy). These are an actual out-of-sample test for this short saved interval, not evidence of general predictive capability.

The rank-9 combined standardized state gives relative Frobenius errors of 0.0268344 for POD and 0.686642 for standard DMD. Modal arrays and coefficient histories are stored in 04_pod/models and 05_standard_dmd/models, with paths in 10_tables/rom_model_manifest.csv. Approximate array storage is reported in 10_tables/model_storage_estimates.csv; with four separate rank-9 complex DMD models, DMD mode storage is not automatically smaller than the raw four-field snapshot array.

## Best-performing ROM for the completed tests

For in-sample field reconstruction, rank-9 POD is the best-performing completed method. The completed standard DMD is dynamically parameterized but less accurate for these moving, nonlinear transient fields at the same retained rank.

## Shock-attached conclusion

No shock-attached DMD or shock-position error is reported. The available global pressure-gradient diagnostic is not a shock curve and cannot validate a moving-frame transform in a 2D wedge flow. Treating it as one would create an unsupported result.

## Scientific significance and recommended next step

The evidence is sufficient to state that an energy-optimal low-rank representation is effective for these saved fields, while this standard Eulerian exact-DMD formulation has materially higher reconstruction and held-out errors. The next publishable step is to define a wedge-shock region of interest, extract a time-resolved shock curve with uncertainty/sensitivity checks, construct a documented geometry-aligned registration, and compare it directly against the present Eulerian baselines on identical data splits.

## Limitations

The data end at 36 us, only one mesh is available, the train/test interval is short, and no analytical oblique-shock validation has been performed. These limitations are discussed in 11_results/LIMITATIONS_AND_DATA_STATUS.md.
""")
    print(json.dumps({"status": "complete", "snapshots": int(m), "cells": int(n), "selected_rank": k}))

if __name__ == "__main__":
    main()
