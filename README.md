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
