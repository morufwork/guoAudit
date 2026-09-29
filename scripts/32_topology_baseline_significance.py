"""Cluster-aware CIs and chance/gap significance tests for the
positive-network topology-only (degree-only) baseline.

Section 3.5's degree-only Logistic Regression baseline (ROC-AUC 0.962 on
N0, collapsing to ~0.50 against N2/N5/N6) is this study's single most
dramatic number -- repeated in the Abstract, Section 3.5, Section 4.1, and
the Limitations -- yet, unlike every other headline number in this paper
from Section 3.8 onward, it has so far been reported only as a bare point
estimate with no confidence interval or formal chance test attached. This
mirrors the gap closed for the calibration numbers, which were
otherwise bare point estimates.

This script reuses scripts/24 and scripts/26's exact protocol (same seed,
same test_fraction, same saved N0/N2/N5/N6 negative sets, same
build_topology_matrix feature construction: sum/|diff|/min/max of positive
degree) to refit Experiment A (train on N0, the protocol used for the
headline 0.962 number) and, this time, save per-row predictions
(protein_a, protein_b, y_true, y_prob) so the project's existing
connected-component bootstrap/jackknife machinery
(src/evaluation/bootstrap.py) can attach a CI and a two-sided chance test
to:
  (a) the degree-only baseline's own N0 ROC-AUC/MCC (is 0.962 itself
      distinguishable from chance under this benchmark's small-G test
      set?), and
  (b) the N0-vs-N2, N0-vs-N5, and N0-vs-N6 gaps for this specific model
      (an unpaired/independent-samples comparison, since each negative set
      is a different, non-overlapping test set -- same structure as
      Section 3.8's Q3, applied here to the degree-only model rather than
      the six sequence/architecture models Q3 already covers).

This is a targeted, single-model supplementary check requested post hoc,
not a retroactive edit to Section 3.8's pre-specified 44-test-per-method
family (Table 13) -- folding it into that family after the fact would
violate the paper's own stated principle that the family boundary is fixed
in advance, not chosen after inspecting results (Section
2.8). Raw (uncorrected) bootstrap and jackknife p-values are reported
here, analogous to how the calibration CIs were also reported
without folding them into the Section 3.8 BH family.

Writes:
  tables/topology_baseline_significance.csv
  results/topology_baseline_significance/
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data.negative_sampling import compute_protein_stats
from src.evaluation.bootstrap import (
    bootstrap_ci,
    cluster_jackknife_ci,
    connected_component_groups,
    protein_cluster_bootstrap_indices,
    unpaired_cluster_jackknife_diff_ci,
)
from src.models.classical import build_model
from src.utils.io import load_config, save_json
from src.utils.logging import get_logger
from src.utils.manifest import write_manifest
from src.utils.seed import set_seed

REPO_ROOT = Path(__file__).resolve().parents[1]
logger = get_logger("topology_baseline_significance")

N_BOOT = 2000
NEGATIVE_SETS = ["n0", "n2", "n5", "n6"]


def mcc_metric(y_true, y_prob, threshold=0.5):
    return matthews_corrcoef(y_true, (y_prob >= threshold).astype(int))


def build_topology_matrix(pairs_df: pd.DataFrame, scalar: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    da = pairs_df["protein_a"].map(scalar).to_numpy(dtype=float)
    db = pairs_df["protein_b"].map(scalar).to_numpy(dtype=float)
    x = np.column_stack([da + db, np.abs(da - db), np.minimum(da, db), np.maximum(da, db)])
    y = pairs_df["label"].to_numpy()
    return x, y


def empirical_two_sided_p(values: np.ndarray, null: float) -> float:
    p_le = np.mean(values <= null)
    p_ge = np.mean(values >= null)
    return float(min(1.0, 2 * min(p_le, p_ge)))


def unpaired_bootstrap_diff(y_true_a, y_prob_a, idx_a, y_true_b, y_prob_b, idx_b, metric_fn):
    boot_a = np.array([metric_fn(y_true_a[i], y_prob_a[i]) for i in idx_a if len(np.unique(y_true_a[i])) > 1])
    boot_b = np.array([metric_fn(y_true_b[i], y_prob_b[i]) for i in idx_b if len(np.unique(y_true_b[i])) > 1])
    m = min(len(boot_a), len(boot_b))
    diff_dist = boot_a[:m] - boot_b[:m]
    lower, upper = np.percentile(diff_dist, [2.5, 97.5])
    point = metric_fn(y_true_a, y_prob_a) - metric_fn(y_true_b, y_prob_b)
    return {
        "point": float(point), "ci_low": float(lower), "ci_high": float(upper),
        "significant": bool(lower > 0 or upper < 0),
        "p_value": empirical_two_sided_p(diff_dist, 0.0),
    }


def main():
    data_config = load_config(REPO_ROOT / "configs" / "data.yaml")
    config = load_config(REPO_ROOT / "configs" / "hard_negative.yaml")
    seed = config["seed"]
    set_seed(seed)

    # --- Exact reload of scripts/24 and 26's inputs, so splits/negatives are identical ---
    pairs = pd.read_csv(REPO_ROOT / "data" / "processed" / "ppi_pairs_clean.tsv", sep="\t", header=None, names=data_config["pairs_columns"])
    proteins = pd.read_csv(REPO_ROOT / "data" / "processed" / "proteins_with_groups.tsv", sep="\t")
    pos = pairs[pairs["label"] == 1].reset_index(drop=True)
    neg0 = pairs[pairs["label"] == 0].reset_index(drop=True)

    protein_stats = compute_protein_stats(proteins, pos, n_deciles=config["n_deciles"])
    scalar = protein_stats["positive_degree"]

    neg_sets_dir = REPO_ROOT / "data" / "processed" / "negative_sets"
    negative_sets = {
        "n0": neg0,
        "n2": pd.read_csv(neg_sets_dir / "n2.tsv", sep="\t"),
        "n5": pd.read_csv(neg_sets_dir / "n5.tsv", sep="\t"),
        "n6": pd.read_csv(neg_sets_dir / "n6.tsv", sep="\t"),
    }

    pos_train, pos_test = train_test_split(pos, test_size=config["test_fraction"], random_state=seed)
    train_sets, test_sets = {}, {}
    for name, neg_df in negative_sets.items():
        neg_train, neg_test = train_test_split(neg_df, test_size=config["test_fraction"], random_state=seed)
        train_sets[name] = pd.concat([pos_train, neg_train], ignore_index=True)
        test_sets[name] = pd.concat([pos_test, neg_test], ignore_index=True)

    # --- Experiment A protocol: fit degree-only LR once on N0, evaluate on N0/N2/N5/N6 ---
    x_train, y_train = build_topology_matrix(train_sets["n0"], scalar)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    model = build_model("logistic_regression", {}, seed)
    model.fit(x_train, y_train)

    predictions_dir = REPO_ROOT / "results" / "topology_baseline_significance"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    preds = {}
    for name in NEGATIVE_SETS:
        test_df = test_sets[name].reset_index(drop=True)
        x_test, y_test = build_topology_matrix(test_df, scalar)
        x_test = scaler.transform(x_test)
        y_prob = model.predict_proba(x_test)[:, 1]
        pred_df = pd.DataFrame({
            "protein_a": test_df["protein_a"], "protein_b": test_df["protein_b"],
            "y_true": y_test, "y_probability": y_prob,
        })
        pred_df.to_csv(predictions_dir / f"{name}_predictions.csv", index=False)
        preds[name] = pred_df
        logger.info(f"[{name}] degree-only LR (trained on N0): ROC-AUC={roc_auc_score(y_test, y_prob):.4f} MCC={mcc_metric(y_test, y_prob):.4f}")

    # --- (a) Chance test: is each set's ROC-AUC/MCC distinguishable from chance? ---
    idx_cache, groups_cache = {}, {}
    chance_rows = []
    for name in NEGATIVE_SETS:
        pred_df = preds[name]
        y_true, y_prob = pred_df["y_true"].to_numpy(), pred_df["y_probability"].to_numpy()
        idx = protein_cluster_bootstrap_indices(pred_df, N_BOOT, seed=42)
        groups = connected_component_groups(pred_df)
        idx_cache[name], groups_cache[name] = idx, groups

        roc_boot = bootstrap_ci(y_true, y_prob, idx, roc_auc_score, null=0.5)
        roc_jk = cluster_jackknife_ci(y_true, y_prob, groups, roc_auc_score, null=0.5)
        mcc_boot = bootstrap_ci(y_true, y_prob, idx, mcc_metric, null=0.0)
        mcc_jk = cluster_jackknife_ci(y_true, y_prob, groups, mcc_metric, null=0.0)

        logger.info(
            f"[{name}] ROC-AUC bootstrap=[{roc_boot['ci_low']:.3f}, {roc_boot['ci_high']:.3f}] p={roc_boot['p_value']:.4f} | "
            f"jackknife=[{roc_jk['ci_low']:.3f}, {roc_jk['ci_high']:.3f}] p={roc_jk['p_value']:.4f} "
            f"({roc_jk['n_clusters_used']}/{roc_jk['n_clusters']} components)"
        )
        chance_rows.append({
            "test": "chance", "negative_set": name,
            "roc_auc": roc_boot["point"],
            "roc_auc_bootstrap_ci_low": roc_boot["ci_low"], "roc_auc_bootstrap_ci_high": roc_boot["ci_high"], "roc_auc_bootstrap_p": roc_boot["p_value"],
            "roc_auc_jackknife_ci_low": roc_jk["ci_low"], "roc_auc_jackknife_ci_high": roc_jk["ci_high"], "roc_auc_jackknife_p": roc_jk["p_value"],
            "mcc": mcc_boot["point"],
            "mcc_bootstrap_ci_low": mcc_boot["ci_low"], "mcc_bootstrap_ci_high": mcc_boot["ci_high"], "mcc_bootstrap_p": mcc_boot["p_value"],
            "mcc_jackknife_ci_low": mcc_jk["ci_low"], "mcc_jackknife_ci_high": mcc_jk["ci_high"], "mcc_jackknife_p": mcc_jk["p_value"],
            "n_clusters": roc_jk["n_clusters"], "n_clusters_used": roc_jk["n_clusters_used"],
        })

    # --- (b) N0-vs-{N2,N5,N6} gap for this specific model (unpaired, different test sets) ---
    gap_rows = []
    y_true_n0, y_prob_n0 = preds["n0"]["y_true"].to_numpy(), preds["n0"]["y_probability"].to_numpy()
    for other in ["n2", "n5", "n6"]:
        y_true_o, y_prob_o = preds[other]["y_true"].to_numpy(), preds[other]["y_probability"].to_numpy()
        diff_boot = unpaired_bootstrap_diff(y_true_n0, y_prob_n0, idx_cache["n0"], y_true_o, y_prob_o, idx_cache[other], roc_auc_score)
        diff_jk = unpaired_cluster_jackknife_diff_ci(y_true_n0, y_prob_n0, groups_cache["n0"], y_true_o, y_prob_o, groups_cache[other], roc_auc_score)
        logger.info(
            f"[N0 - {other}] ROC-AUC delta (unpaired) = {diff_boot['point']:.3f} "
            f"bootstrap=[{diff_boot['ci_low']:.3f}, {diff_boot['ci_high']:.3f}] p={diff_boot['p_value']:.4f} | "
            f"jackknife=[{diff_jk['ci_low']:.3f}, {diff_jk['ci_high']:.3f}] p={diff_jk['p_value']:.4f}"
        )
        gap_rows.append({
            "test": "n0_vs_gap", "negative_set": other,
            "roc_auc_delta": diff_boot["point"],
            "roc_auc_delta_bootstrap_ci_low": diff_boot["ci_low"], "roc_auc_delta_bootstrap_ci_high": diff_boot["ci_high"],
            "roc_auc_delta_bootstrap_p": diff_boot["p_value"], "roc_auc_delta_bootstrap_significant": diff_boot["significant"],
            "roc_auc_delta_jackknife_ci_low": diff_jk["ci_low"], "roc_auc_delta_jackknife_ci_high": diff_jk["ci_high"],
            "roc_auc_delta_jackknife_p": diff_jk["p_value"], "roc_auc_delta_jackknife_significant": diff_jk.get("significant", False),
        })

    results_df = pd.concat([pd.DataFrame(chance_rows), pd.DataFrame(gap_rows)], ignore_index=True)
    tables_dir = REPO_ROOT / "tables"
    results_df.to_csv(tables_dir / "topology_baseline_significance.csv", index=False)
    logger.info(f"Wrote {tables_dir}/topology_baseline_significance.csv")

    save_json(
        {
            "note": (
                "Cluster-aware CIs and chance/gap significance tests for the degree-only "
                "Logistic Regression baseline (Section 3.5's headline 0.962 ROC-AUC number), "
                "reusing scripts/24 and scripts/26's exact Experiment A protocol (train on N0, "
                "evaluate on N0/N2/N5/N6). A targeted supplementary check, "
                "not folded into Section 3.8's pre-specified 44-test-per-method family."
            ),
            "chance_test": chance_rows,
            "n0_vs_gap": gap_rows,
        },
        predictions_dir / "topology_baseline_significance_summary.json",
    )
    write_manifest(predictions_dir, config=config, seed=seed, extra={"script": "32_topology_baseline_significance.py"})
    logger.info("Topology-only baseline significance check complete.")


if __name__ == "__main__":
    main()
