# Split Validation Report

## Decision

**PASS.** The regenerated R3-50/R3-40/R3-30/R3-20 artifacts satisfy all audited leakage constraints.

## Reproducibility

- MMseqs2 build: `8cc5ce367b5638c4306c2d7cfc652dd099a4643f`
- Exact command template: `mmseqs easy-cluster INPUT OUTPUT TMP --min-seq-id TAU -c 0.8 --cov-mode 0 --cluster-mode 0 -s 4.0 --seq-id-mode 0 -v 1`
- Clustering scope: all proteins at each threshold before partitioning.
- Assignment unit: clusters, never individual proteins.
- `--cov-mode 0 -c 0.8`: both query and target alignment coverage must be at least 80%.
- `--seq-id-mode 0`: sequence identity is normalized by alignment length.
- `--cluster-mode 0`: greedy set-cover clustering.

## Audit table

| Split | Train pairs | Test pairs | Train proteins | Test proteins | Exact protein overlap | Exact sequence overlap | Cluster overlap | Pair/reversal overlap | Identity threshold | Coverage query | Coverage target | PASS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R3-50 | 7227 | 450 | 2004 | 407 | 0 | 0 | 0 | 0 | 0.5 | 0.8 | 0.8 | True |
| R3-40 | 7111 | 423 | 1998 | 391 | 0 | 0 | 0 | 0 | 0.4 | 0.8 | 0.8 | True |
| R3-30 | 7174 | 419 | 2007 | 394 | 0 | 0 | 0 | 0 | 0.3 | 0.8 | 0.8 | True |
| R3-20 | 7043 | 445 | 1996 | 420 | 0 | 0 | 0 | 0 | 0.2 | 0.8 | 0.8 | True |

The terms “homology-free” and “absence of homology” are not supported. These are sequence-similarity-controlled R3 variants.
