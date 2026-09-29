# Species-specific replication

Three species × four negative sets × five models = 60 fitted evaluations.
All evaluations use seed 42 and 80/20 random pair splits. Positive test pairs
are shared across negative variants within a species. Protein overlap is
permitted; these results do not measure unseen-protein generalization.

Reproduce from the cached taxonomy and frozen feature files:

```bash
.venv/bin/python scripts/50_audit_species_assignments.py
.venv/bin/python scripts/49_species_stratified_replication.py --species celegans
.venv/bin/python scripts/49_species_stratified_replication.py --species dmelanogaster
.venv/bin/python scripts/49_species_stratified_replication.py --species ecoli
.venv/bin/python scripts/52_validate_species_replication.py --data-only
```

The model runner reuses completed metrics checkpoints. Each model folder
contains predictions, all nine classification metrics, iteration counts,
and fit warnings. Species folders contain the full inputs, negative sets,
train/test partitions, degree statistics, balance diagnostics, and environment.
Use a fresh output directory when changing inputs or model settings.

Reporting is a one-time insertion into the pre-species V2 documents:
`scripts/51_report_species_replication.py` validates the experiment outputs,
backs up the Word files, and adds Section 3.8, Figure 11, S20, and S25–S33.
It also qualifies pooled-result claims and records those changes in
`manuscript_claim_revisions.json`. It refuses duplicate section insertion.
The completed documents can be checked repeatedly with:

```bash
.venv/bin/python scripts/52_validate_species_replication.py
```

`data_validation.json` records the pre-report checks; `validation.json`
adds manuscript/table checks and input/script SHA-256 hashes. The human-readable
validation summary is `revision/SPECIES_REPLICATION_VALIDATION.md`.

The pooled same-species indicator is a construction diagnostic: it scores
same-species pairs as 1 and cross-species pairs as 0. Its ROC-AUC is not the
performance of a fitted sequence model. Exact-degree matching can omit
unmatchable substitutions and therefore need not balance the complete
positive set; realized counts and degree distances are reported explicitly.
