<img width="1082" height="637" alt="Screenshot 2026-08-10 000751" src="https://github.com/user-attachments/assets/c4e4c683-2ab6-47f5-81b5-a2d379865bc6" />
<img width="2055" height="1858" alt="fom_pod_dmd_p_representative_times" src="https://github.com/user-attachments/assets/0b82191f-c63c-4685-a8a6-92c6c94dd3a5" />
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
# Shock-Aligned Reduced-Order Modeling of a Transient Compressible Wedge Flow

This repository contains the complete workflow for a shock-aligned
reduced-order modeling (ROM) study of a transient compressible flow
over a wedge.

The repository includes both the OpenFOAM simulation case used to
generate the flow-field snapshots and the Python-based post-processing
and reduced-order modeling workflow.

## Workflow

OpenFOAM Simulation
        ↓
Flow-field Snapshots
        ↓
Shock Tracking
        ↓
Snapshot Pre-processing
        ↓
POD / DMD
        ↓
Reduced-Order Reconstruction
        ↓
Error Evaluation

## Contents

- `OpenFOAM/` — OpenFOAM case used for the high-fidelity simulation.
- `python/` — Python scripts for snapshot processing, shock tracking,
  POD/DMD analysis, ROM reconstruction, and visualization.
- `results/` — Selected figures and analysis outputs.

## Physical Problem

The case considers a transient compressible flow over a wedge.
The OpenFOAM simulation is used as the high-fidelity reference solution
for developing and evaluating the reduced-order model.

## Reduced-Order Modeling

The Python workflow performs:

- snapshot preprocessing
- shock-location tracking
- shock-aligned coordinate transformation
- POD analysis
- Dynamic Mode Decomposition (DMD)
- reduced-order reconstruction
- reconstruction error evaluation

## Reproducibility

The OpenFOAM case files required to reproduce the high-fidelity
simulation are included in this repository.

Python dependencies are listed in `requirements.txt`.

Generated solver outputs and large intermediate datasets are not
included in the repository.

## Status

This repository contains the current research workflow for the
shock-aligned reduced-order modeling study.
