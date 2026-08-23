"""Data-gated 2D POD/DMD workflow.

With no valid transient snapshots this script writes only the case audit and exits
successfully. It never substitutes synthetic data or inferred results.
"""
from __future__ import annotations
import csv
import json
import re
from pathlib import Path
import numpy as np
from foam2d_io import discover, reconstruct
from rom2d_methods import choose_rank, exact_dmd, metrics, pod, reconstruct_dmd, require_uniform_times, standardize_and_stack

CASE = Path("/mnt/f/Jahidul_Islam/DetFOAM_2DWedge_NonReact_For_ROM")
OUT = CASE / "ROM_DMD_2D_Journal_Work"

def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def text(path: Path, value: str) -> None:
    path.write_text(value.strip() + "\n", encoding="utf-8")

def dictionary_value(path: Path, key: str) -> str | None:
    if not path.exists():
        return None
    found = re.search(r"\b" + re.escape(key) + r"\s+([^;]+);", path.read_text(errors="replace"))
    return found.group(1).strip() if found else None

def audit(config: dict, info: dict) -> dict:
    control = CASE / "system" / "controlDict"
    schemes = CASE / "system" / "fvSchemes"
    thermo = CASE / "constant" / "thermophysicalProperties"
    chemistry = CASE / "constant" / "chemistryProperties"
    return {
        "case_path": str(CASE), "application": dictionary_value(control, "application"),
        "end_time_s": dictionary_value(control, "endTime"), "write_control": dictionary_value(control, "writeControl"),
        "write_interval_s": dictionary_value(control, "writeInterval"), "max_courant": dictionary_value(control, "maxCo"),
        "max_acoustic_courant": dictionary_value(control, "maxAcousticCo"), "flux_scheme": dictionary_value(schemes, "fluxScheme"),
        "time_scheme": dictionary_value(schemes, "default"), "thermo_type": dictionary_value(thermo, "type"),
        "equation_of_state": dictionary_value(thermo, "equationOfState"), "chemistry": dictionary_value(chemistry, "chemistry"),
        "processors_found": len(info["processors"]), "mesh_cells": info["n_cells"],
        "all_numeric_time_s": info["all_numeric_times"], "valid_snapshot_time_s": info["valid_times"].tolist(),
        "initial_field_names": info["initial_field_names"], "missing_required_fields_by_time": {str(k): v for k, v in info["missing_fields"].items()},
        "required_rom_fields": config["snapshot"]["required_fields"],
    }

def write_audit_documents(config: dict, case_audit: dict) -> None:
    write_json(OUT / "01_case_audit" / "case_audit.json", case_audit)
    write_json(OUT / "01_case_audit" / "data_availability.json", {
        "valid_snapshot_count": len(case_audit["valid_snapshot_time_s"]),
        "valid_snapshot_time_s": case_audit["valid_snapshot_time_s"],
        "required_fields": case_audit["required_rom_fields"],
        "status": "ready_for_rom" if len(case_audit["valid_snapshot_time_s"]) >= config["snapshot"]["minimum_snapshots"] else "data_gate_not_met",
        "reason": "No numerical ROM is executed until all configured fields exist at enough common times.",
    })
    text(OUT / "01_case_audit" / "case_audit.md", f"""
# Verified 2D case audit

- Solver: '{case_audit["application"]}'; flux: '{case_audit["flux_scheme"]}'; time scheme entry: '{case_audit["time_scheme"]}'.
- Configured end time: '{case_audit["end_time_s"]}'; write control/interval: '{case_audit["write_control"]}' / '{case_audit["write_interval_s"]}'.
- Thermodynamics: '{case_audit["thermo_type"]}', equation of state '{case_audit["equation_of_state"]}'; chemistry '{case_audit["chemistry"]}'.
- Mesh: '{case_audit["mesh_cells"]}' cells, no processor directories currently present.
- Required ROM fields: '{", ".join(case_audit["required_rom_fields"])}'.
- Numerical time directories actually found: '{case_audit["all_numeric_time_s"]}'.
- Complete snapshot times actually found: '{case_audit["valid_snapshot_time_s"]}'.

The initial field list and every missing required field are retained verbatim in case_audit.json. The audit makes no statement about transient physics because the required transient field snapshots are unavailable.
""")
    text(OUT / "11_results" / "DATA_GATE.md", f"""
# Data gate

This workflow requires at least '{config["snapshot"]["minimum_snapshots"]}' common, finite snapshots containing '{", ".join(config["snapshot"]["required_fields"])}'. The verified count is '{len(case_audit["valid_snapshot_time_s"])}'.

No POD/DMD values, eigenvalues, reconstruction errors, shock trajectories, or figures are created until this gate is satisfied. This is intentional.
""")

def extract_fields(info: dict, time_s: np.ndarray) -> dict[str, np.ndarray]:
    result = {name: [] for name in ("p", "rho", "Ux", "Uy")}
    for time in time_s:
        result["p"].append(reconstruct(info, float(time), "p"))
        result["rho"].append(reconstruct(info, float(time), "rho"))
        velocity = reconstruct(info, float(time), "U")
        result["Ux"].append(velocity[:, 0])
        result["Uy"].append(velocity[:, 1])
    return {name: np.column_stack(values) for name, values in result.items()}

def analyse(config: dict, info: dict) -> None:
    time_s = info["valid_times"]
    dt = require_uniform_times(time_s)
    fields = extract_fields(info, time_s)
    np.savez_compressed(OUT / "02_data_extraction" / "snapshots_2d_p_rho_U.npz", xy_m=info["xy_m"], time_s=time_s, **fields)
    rows, singular_rows, eigen_rows = [], [], []
    for name, state in fields.items():
        probe = exact_dmd(state, dt, 1)
        max_rank = min(state.shape[1] - 1, probe.singular_values.size)
        selected = min(choose_rank(probe.singular_values[:max_rank], config["rom"]["energy_target"]), max_rank)
        for i, singular in enumerate(probe.singular_values, 1):
            singular_rows.append({"field": name, "rank": i, "singular_value": float(singular),
                                  "energy_fraction": float(singular**2 / np.sum(probe.singular_values**2)),
                                  "cumulative_energy": float(np.sum(probe.singular_values[:i]**2) / np.sum(probe.singular_values**2))})
        for rank in sorted(set([r for r in config["rom"]["ranks"] if r <= max_rank] + [selected, max_rank])):
            dmd = exact_dmd(state, dt, rank)
            dmd_reconstruction = reconstruct_dmd(dmd, len(time_s))
            pod_reconstruction, _, _ = pod(state, rank)
            rows.append({"method": "standard_dmd", "field": name, "rank": dmd.rank, "selected_rank": selected, **metrics(state, dmd_reconstruction)})
            rows.append({"method": "pod", "field": name, "rank": rank, "selected_rank": selected, **metrics(state, pod_reconstruction)})
        dmd = exact_dmd(state, dt, selected)
        omega = np.log(dmd.eigenvalues) / dt
        for i, value in enumerate(dmd.eigenvalues, 1):
            eigen_rows.append({"field": name, "mode": i, "eigenvalue_real": float(value.real),
                               "eigenvalue_imag": float(value.imag), "modulus": float(abs(value)),
                               "growth_rate_per_s": float(omega[i-1].real),
                               "frequency_hz": float(omega[i-1].imag / (2*np.pi)),
                               "amplitude_modulus": float(abs(dmd.amplitudes[i-1]))})
    write_csv(OUT / "10_tables" / "reconstruction_errors.csv", rows)
    write_csv(OUT / "10_tables" / "singular_values.csv", singular_rows)
    write_csv(OUT / "10_tables" / "dmd_eigenvalues.csv", eigen_rows)
    combined, metadata = standardize_and_stack(fields)
    write_json(OUT / "03_preprocessing" / "combined_state_metadata.json", {
        name: {"slice": meta["slice"], "scale": float(meta["scale"])} for name, meta in metadata.items()})
    if config["shock_geometry"]["enabled"]:
        raise NotImplementedError("Shock geometry requires an approved physical ROI and curve definition.")
    text(OUT / "11_results" / "analysis_status.md", """
Standard Eulerian POD/DMD was executed on verified snapshots. Shock alignment was not executed automatically: a 2D shock is a curve/deforming surface rather than the 1D scalar position, so registration requires an explicit region-of-interest and curve-tracking specification.
""")

def main() -> None:
    config = json.loads((OUT / "config" / "analysis_config.json").read_text())
    info = discover(CASE, config["snapshot"]["required_fields"])
    case_audit = audit(config, info)
    write_audit_documents(config, case_audit)
    if len(info["valid_times"]) < config["snapshot"]["minimum_snapshots"]:
        print(json.dumps({"status": "data_gate_not_met", "valid_snapshot_count": len(info["valid_times"])}))
        return
    analyse(config, info)
    print(json.dumps({"status": "analysis_complete", "valid_snapshot_count": len(info["valid_times"])}))

if __name__ == "__main__":
    main()
