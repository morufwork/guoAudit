# Manuscript reporting scripts

These scripts assembled the manuscript's reporting tables and final figures **from saved
analysis outputs; they train no models**. They are included so that every Supplementary
Table has a documented origin (see `docs/PROVENANCE.md`).

| Script | Produces |
|---|---|
| `51_report_species_replication.py` | Tables S25–S33, Figure 12 |
| `52_validate_species_replication.py` | Consistency checks for the species replication |
| `53_complete_manuscript_reporting.py` | Tables S34–S44, Figure 3 |
| `56_revise_coverage_reporting.py` | Table S45 |
| `59_refresh_editorial_figures.py` | Final versions of Figures 1, 2, 4 and 5 |
| `build_figure_data_tables.py` | Rebuilds and verifies Tables S23–S24 |

**Caveat.** Scripts 51–59 were written to also edit the manuscript source files
(`manuscript/*.md`, `.docx`) and a private revision folder, neither of which is
distributed here. Run as-is, they stop when those files are missing. The table- and
figure-building parts only read `results/`, `tables/` and `data/`; to reuse them,
comment out the manuscript-editing blocks. `build_figure_data_tables.py` has no such
dependency.
