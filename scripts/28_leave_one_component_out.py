"""Leave-one-connected-component-out sensitivity analysis.

The manuscript's cluster-aware CIs (Section 2.8) already establish that R3
test sets have only 5-19 usable connected components, one of which
dominates (94.6% of N0's pairs, 98.2% of N2's). This makes the following
the most important additional analysis: for each connected
component, remove it and recompute the N0-vs-N2 effect -- does the effect
persist when the giant component specifically is excluded?

This script computes exactly that, for all six Section 3.7 architectures:
  1. A full leave-one-component-out table (every component, not just the
     giant one) via src/evaluation/bootstrap.py's
     leave_one_component_out_table -- the same internal step the
     delete-1-cluster jackknife already performs, exposed directly here
     rather than only summarized into a variance estimate.
  2. The specific "giant-component-excluded" scenario: recompute each of N0's and N2's ROC-AUC with ONLY the giant
     component's rows removed (not one-at-a-time over every component),
     and compare the resulting N0-vs-N2 gap to the full-sample gap.

A cluster-level permutation test was also considered but found similarly degenerate: the combined N0-union-N2
test-set graph has only 4 connected components, one of which holds 905 of
908 rows -- permuting component-level group labels would have almost no
distinct permutations to draw from, the same "too few clusters" problem
in a different form. This is logged and reported rather than silently
skipped.

Writes:
  tables/leave_one_component_out.csv
  tables/giant_component_excluded_sensitivity.csv
  figures/giant_component_excluded_sensitivity.pdf
  results/leave_one_component_out/
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.evaluation.bootstrap import connected_component_groups, leave_one_component_out_table
from src.utils.io import save_json
from src.utils.logging import get_logger
from src.utils.manifest import write_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]
logger = get_logger("leave_one_component_out")

MODEL_ORDER = ["logistic_regression", "random_forest", "mlp", "siamese_mlp", "attention_fusion", "cross_interaction"]


def load_predictions(neg_name: str, model_name: str) -> pd.DataFrame:
    return pd.read_csv(REPO_ROOT / "results" / "architecture_ablation" / f"{neg_name}__{model_name}" / "predictions.csv").reset_index(drop=True)


def main():
    loco_rows = []
    sensitivity_rows = []

    for model_name in MODEL_ORDER:
        n0 = load_predictions("n0", model_name)
        n2 = load_predictions("n2", model_name)
        y0, p0 = n0["y_true"].to_numpy(), n0["y_probability"].to_numpy()
        y2, p2 = n2["y_true"].to_numpy(), n2["y_probability"].to_numpy()
        g0 = connected_component_groups(n0)
        g2 = connected_component_groups(n2)

        table0 = leave_one_component_out_table(y0, p0, g0, roc_auc_score)
        table0["negative_set"], table0["model"] = "n0", model_name
        table2 = leave_one_component_out_table(y2, p2, g2, roc_auc_score)
        table2["negative_set"], table2["model"] = "n2", model_name
        loco_rows.append(table0)
        loco_rows.append(table2)

        full0, full2 = roc_auc_score(y0, p0), roc_auc_score(y2, p2)
        giant0_id = table0.loc[table0["is_giant_component"], "component_id"].iloc[0]
        giant2_id = table2.loc[table2["is_giant_component"], "component_id"].iloc[0]
        excl0 = table0.loc[table0["component_id"] == giant0_id, "loo_metric"].iloc[0]
        excl2 = table2.loc[table2["component_id"] == giant2_id, "loo_metric"].iloc[0]
        n0_remaining = int(table0.loc[table0["component_id"] == giant0_id, "n_rows_remaining"].iloc[0])
        n2_remaining = int(table2.loc[table2["component_id"] == giant2_id, "n_rows_remaining"].iloc[0])

        full_gap = full0 - full2
        excl_gap = excl0 - excl2
        logger.info(
            f"[{model_name}] full: N0={full0:.3f} N2={full2:.3f} gap={full_gap:+.3f} | "
            f"giant-excluded (N0 n={n0_remaining}, N2 n={n2_remaining}): "
            f"N0={excl0:.3f} N2={excl2:.3f} gap={excl_gap:+.3f} | "
            f"direction {'PERSISTS' if (excl_gap > 0) == (full_gap > 0) else 'REVERSES'}"
        )
        sensitivity_rows.append({
            "model": model_name,
            "full_n0_roc_auc": full0, "full_n2_roc_auc": full2, "full_gap": full_gap,
            "giant_excluded_n0_roc_auc": excl0, "giant_excluded_n2_roc_auc": excl2, "giant_excluded_gap": excl_gap,
            "n0_remaining_after_giant_excluded": n0_remaining, "n2_remaining_after_giant_excluded": n2_remaining,
            "direction_persists": bool((excl_gap > 0) == (full_gap > 0)),
        })

    loco_df = pd.concat(loco_rows, ignore_index=True)
    tables_dir = REPO_ROOT / "tables"
    loco_df.to_csv(tables_dir / "leave_one_component_out.csv", index=False)
    logger.info(f"Wrote {tables_dir}/leave_one_component_out.csv ({len(loco_df)} rows: 6 models x (19 N0 + 5 N2) components)")

    sensitivity_df = pd.DataFrame(sensitivity_rows)
    sensitivity_df.to_csv(tables_dir / "giant_component_excluded_sensitivity.csv", index=False)
    logger.info(f"Wrote {tables_dir}/giant_component_excluded_sensitivity.csv")

    # --- Permutation-test feasibility check (logged, not run -- see module docstring) ---
    n0_full = load_predictions("n0", MODEL_ORDER[0])
    n2_full = load_predictions("n2", MODEL_ORDER[0])
    union = pd.concat([n0_full[["protein_a", "protein_b"]], n2_full[["protein_a", "protein_b"]]], ignore_index=True)
    union_groups = connected_component_groups(union)
    union_sizes = sorted((len(v) for v in union_groups.values()), reverse=True)
    logger.info(
        f"Cluster-level permutation test feasibility: the combined N0-union-N2 test graph has "
        f"{len(union_groups)} connected components (sizes {union_sizes}) out of {len(union)} rows -- "
        f"too few, and too dominated by one component, for a component-level permutation test to have "
        f"meaningful power here (same underlying cause as the small G for the jackknife/bootstrap). "
        f"Not run; disclosed instead of silently omitted."
    )

    # --- Figure: full gap vs. giant-excluded gap, per model ---
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(MODEL_ORDER))
    width = 0.35
    full_gaps = [sensitivity_df.set_index("model").loc[m, "full_gap"] for m in MODEL_ORDER]
    excl_gaps = [sensitivity_df.set_index("model").loc[m, "giant_excluded_gap"] for m in MODEL_ORDER]
    ax.bar(x - width / 2, full_gaps, width, label="Full sample", color="#d95f02")
    ax.bar(x + width / 2, excl_gaps, width, label="Giant component excluded", color="#1b9e77")
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_ORDER, rotation=30, ha="right")
    ax.set_ylabel("N0 - N2 ROC-AUC gap")
    ax.set_title("N0-vs-N2 gap: full sample vs. giant connected component excluded")
    ax.legend()
    fig.tight_layout()
    figures_dir = REPO_ROOT / "figures"
    fig.savefig(figures_dir / "giant_component_excluded_sensitivity.pdf")
    plt.close(fig)
    logger.info(f"Wrote {figures_dir}/giant_component_excluded_sensitivity.pdf")

    results_dir = REPO_ROOT / "results" / "leave_one_component_out"
    results_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        {
            "note": (
                "Leave-one-connected-component-out sensitivity analysis: "
                "for each of N0's 19 and N2's 5 components, the ROC-AUC recomputed with that component "
                "excluded (tables/leave_one_component_out.csv), plus the specific giant-component-excluded "
                "N0-vs-N2 gap for each of the six Section 3.7 architectures "
                "(tables/giant_component_excluded_sensitivity.csv)."
            ),
            "direction_persists_all_models": bool(sensitivity_df["direction_persists"].all()),
            "permutation_test_considered_but_degenerate": {
                "n_union_components": len(union_groups),
                "union_component_sizes": union_sizes,
            },
        },
        results_dir / "loco_sensitivity_summary.json",
    )
    write_manifest(results_dir, config={}, seed=None, extra={"script": "28_leave_one_component_out.py"})
    logger.info("Leave-one-connected-component-out sensitivity analysis complete.")


if __name__ == "__main__":
    main()
