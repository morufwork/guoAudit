"""N2 data-generation seed sensitivity.

Every existing negative-sampling script in this project draws N2 exactly
once, under a single fixed data-generation seed (42), and reuses that one
realization across every model compared against it -- including
scripts/18_multiseed_robustness.py's "5 retraining seeds," which
deliberately hold the data partition AND the N2 realization fixed and vary
only model initialization/training randomness (see that script's
docstring). That answers "is this finding an artifact of optimization
noise?" but not the question: "is the N0->N2 collapse an
artifact of one lucky/unlucky draw of WHICH decoy protein replaced which
positive-pair endpoint?"

This script regenerates N2 under N_SAMPLE_SEEDS=30 independent
data-generation seeds, holding everything else fixed at its existing,
already-established value:
  - the R0 positive/N0-negative train/test partition (DATA_SEED=42, same
    as every other the negative-sampling study script)
  - the R3 protein-pool partition and the split generation's saved both-unseen split
  - node degree (`compute_protein_stats`, full-graph, unchanged from every
    other script -- scripts/37 tests varying this separately)
  - model training randomness (MODEL_SEED=42, fixed -- this analysis is about
    data-generation-seed variance specifically, not conflated with the
    training-seed variance scripts/18 already covers)

...and varies only the `seed` argument to `sample_matched_negatives`,
i.e., which decile-eligible decoy protein happens to be drawn for each
positive pair. N0 does not depend on this seed at all and is evaluated
once per split, not re-run 30 times.

Reduced representative model set (not the full
architecture suite): Logistic Regression and MLP on ESM-2-mean sequence
features (the study's two headline models), plus the positive-network
degree-only baseline (Section 3.5) reduced to Logistic Regression, to
check whether even the confound-diagnostic baseline itself is seed-stable.

Writes:
  tables/negative_sampling_seed_sensitivity.csv        (one row per seed x split x model)
  tables/negative_sampling_seed_sensitivity_summary.csv (mean/SD/min/max over 30 seeds)
  figures/negative_sampling_seed_sensitivity.pdf
  results/negative_sampling_seed_sensitivity/
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data.negative_sampling import _known_pair_set, compute_protein_stats, sample_matched_negatives
from src.data.splits import partition_protein_pool
from src.evaluation.metrics import classification_metrics
from src.features.build_pair_features import build_pair_matrix
from src.features.esm2 import load_esm2_features
from src.models.classical import build_model
from src.utils.io import save_json
from src.utils.logging import get_logger
from src.utils.manifest import write_manifest
from src.utils.seed import set_seed

REPO_ROOT = Path(__file__).resolve().parents[1]
logger = get_logger("negative_sampling_seed_sensitivity")

DATA_SEED = 42
MODEL_SEED = 42
TEST_FRACTION = 0.20
FUSION = "combined"
N_SAMPLE_SEEDS = 30
SAMPLE_SEEDS = list(range(1000, 1000 + N_SAMPLE_SEEDS))
SEQUENCE_MODELS = {
    "logistic_regression": {"max_iter": 2000},
    "mlp": {"hidden_layer_sizes": [128, 64], "max_iter": 500, "early_stopping": True},
}
NEG_SET_LABELS = {"n0": "N0 (original)", "n2": "N2 (degree-matched, seed-swept)"}
SPLIT_LABELS = {"random": "Random (R0)", "both_unseen": "Both-unseen (R3)"}


def build_topology_matrix(pairs_df: pd.DataFrame, degree: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Identical construction to scripts/24_topology_only_baseline.py's
    degree-only feature set (sum, |diff|, min, max of the two endpoints'
    positive_degree) -- reduced here to Logistic Regression only, since
    the question here is seed stability, not re-litigating linear-vs-
    nonlinear degree separability (already answered in Table 7a)."""
    da = pairs_df["protein_a"].map(degree).to_numpy(dtype=float)
    db = pairs_df["protein_b"].map(degree).to_numpy(dtype=float)
    x = np.column_stack([da + db, np.abs(da - db), np.minimum(da, db), np.maximum(da, db)])
    return x, pairs_df["label"].to_numpy()


def fit_eval_sequence_models(train_df, test_df, protein_features):
    x_train, y_train = build_pair_matrix(train_df, protein_features, fusion=FUSION)
    x_test, y_test = build_pair_matrix(test_df, protein_features, fusion=FUSION)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    out = {}
    for model_name, params in SEQUENCE_MODELS.items():
        model = build_model(model_name, params, MODEL_SEED)
        model.fit(x_train, y_train)
        y_prob = model.predict_proba(x_test)[:, 1]
        out[model_name] = classification_metrics(y_test, y_prob)
    return out


def fit_eval_degree_only(train_df, test_df, degree):
    x_train, y_train = build_topology_matrix(train_df, degree)
    x_test, y_test = build_topology_matrix(test_df, degree)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    model = build_model("logistic_regression", {}, MODEL_SEED)
    model.fit(x_train, y_train)
    y_prob = model.predict_proba(x_test)[:, 1]
    return {"degree_only_lr": classification_metrics(y_test, y_prob)}


def main():
    set_seed(DATA_SEED)
    pairs = pd.read_csv(REPO_ROOT / "data" / "processed" / "ppi_pairs_clean.tsv", sep="\t", header=None,
                         names=["protein_a", "protein_b", "label"])
    proteins = pd.read_csv(REPO_ROOT / "data" / "processed" / "proteins_with_groups.tsv", sep="\t")
    pos = pairs[pairs["label"] == 1].reset_index(drop=True)
    neg0 = pairs[pairs["label"] == 0].reset_index(drop=True)
    known_pairs = _known_pair_set(pos)
    protein_stats = compute_protein_stats(proteins, pos, n_deciles=10)
    degree = protein_stats["positive_degree"]
    protein_features = load_esm2_features("mean", REPO_ROOT / "embeddings" / "esm2")

    out_dir = REPO_ROOT / "results" / "negative_sampling_seed_sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    # --- R0 (random split): FIXED positive/N0 partition at DATA_SEED, N2 seed-swept ---
    pos_train, pos_test = train_test_split(pos, test_size=TEST_FRACTION, random_state=DATA_SEED)
    neg0_train, neg0_test = train_test_split(neg0, test_size=TEST_FRACTION, random_state=DATA_SEED)
    n0_train_r0 = pd.concat([pos_train, neg0_train], ignore_index=True)
    n0_test_r0 = pd.concat([pos_test, neg0_test], ignore_index=True)

    logger.info("=== R0 / N0 (run once -- does not depend on the N2 sampling seed) ===")
    n0_metrics_r0 = {**fit_eval_sequence_models(n0_train_r0, n0_test_r0, protein_features),
                      **fit_eval_degree_only(n0_train_r0, n0_test_r0, degree)}
    for model_name, m in n0_metrics_r0.items():
        rows.append({"split": "random", "negative_set": "n0", "sample_seed": None, "model": model_name,
                     "n_train": len(n0_train_r0), "n_test": len(n0_test_r0), **m})
    logger.info(f"R0/N0: {[(k, round(v['roc_auc'], 4)) for k, v in n0_metrics_r0.items()]}")

    for sample_seed in SAMPLE_SEEDS:
        n2_all = sample_matched_negatives(pos, protein_stats, known_pairs, ["degree_decile"], sample_seed)
        neg2_train, neg2_test = train_test_split(n2_all, test_size=TEST_FRACTION, random_state=DATA_SEED)
        n2_train = pd.concat([pos_train, neg2_train], ignore_index=True)
        n2_test = pd.concat([pos_test, neg2_test], ignore_index=True)

        m = {**fit_eval_sequence_models(n2_train, n2_test, protein_features),
             **fit_eval_degree_only(n2_train, n2_test, degree)}
        for model_name, metrics in m.items():
            rows.append({"split": "random", "negative_set": "n2", "sample_seed": sample_seed, "model": model_name,
                         "n_train": len(n2_train), "n_test": len(n2_test), **metrics})
        logger.info(f"R0/N2 sample_seed={sample_seed}: " + ", ".join(f"{k}={v['roc_auc']:.3f}" for k, v in m.items()))

    # --- R3 (both-unseen): FIXED protein-pool partition + the split generation's saved split, N2 seed-swept ---
    train_pool, test_pool = partition_protein_pool(proteins, TEST_FRACTION, DATA_SEED)
    r3_train_raw = pd.read_csv(REPO_ROOT / "data" / "splits" / "both_unseen" / "train.tsv", sep="\t")
    r3_test_raw = pd.read_csv(REPO_ROOT / "data" / "splits" / "both_unseen" / "test.tsv", sep="\t")
    r3_train_pos = r3_train_raw[r3_train_raw["label"] == 1].reset_index(drop=True)
    r3_test_pos = r3_test_raw[r3_test_raw["label"] == 1].reset_index(drop=True)

    logger.info("=== R3 / N0 (run once) ===")
    n0_metrics_r3 = {**fit_eval_sequence_models(r3_train_raw, r3_test_raw, protein_features),
                      **fit_eval_degree_only(r3_train_raw, r3_test_raw, degree)}
    for model_name, m in n0_metrics_r3.items():
        rows.append({"split": "both_unseen", "negative_set": "n0", "sample_seed": None, "model": model_name,
                     "n_train": len(r3_train_raw), "n_test": len(r3_test_raw), **m})
    logger.info(f"R3/N0: {[(k, round(v['roc_auc'], 4)) for k, v in n0_metrics_r3.items()]}")

    for sample_seed in SAMPLE_SEEDS:
        train_neg = sample_matched_negatives(r3_train_pos, protein_stats, known_pairs, ["degree_decile"], sample_seed, candidate_pool=train_pool)
        test_neg = sample_matched_negatives(r3_test_pos, protein_stats, known_pairs, ["degree_decile"], sample_seed, candidate_pool=test_pool)
        n2_train = pd.concat([r3_train_pos, train_neg], ignore_index=True)
        n2_test = pd.concat([r3_test_pos, test_neg], ignore_index=True)

        m = {**fit_eval_sequence_models(n2_train, n2_test, protein_features),
             **fit_eval_degree_only(n2_train, n2_test, degree)}
        for model_name, metrics in m.items():
            rows.append({"split": "both_unseen", "negative_set": "n2", "sample_seed": sample_seed, "model": model_name,
                         "n_train": len(n2_train), "n_test": len(n2_test), **metrics})
        logger.info(f"R3/N2 sample_seed={sample_seed}: " + ", ".join(f"{k}={v['roc_auc']:.3f}" for k, v in m.items()))

    per_seed_df = pd.DataFrame(rows)
    tables_dir = REPO_ROOT / "tables"
    per_seed_df.to_csv(tables_dir / "negative_sampling_seed_sensitivity.csv", index=False)
    logger.info(f"Wrote {tables_dir}/negative_sampling_seed_sensitivity.csv")

    n2_only = per_seed_df[per_seed_df["negative_set"] == "n2"]
    summary = (
        n2_only.groupby(["split", "model"])
        .agg(roc_auc_mean=("roc_auc", "mean"), roc_auc_std=("roc_auc", "std"),
             roc_auc_min=("roc_auc", "min"), roc_auc_max=("roc_auc", "max"),
             pr_auc_mean=("pr_auc", "mean"), pr_auc_std=("pr_auc", "std"),
             mcc_mean=("mcc", "mean"), mcc_std=("mcc", "std"),
             n_seeds=("roc_auc", "count"))
        .reset_index()
    )
    summary.to_csv(tables_dir / "negative_sampling_seed_sensitivity_summary.csv", index=False)
    logger.info(f"Wrote {tables_dir}/negative_sampling_seed_sensitivity_summary.csv")
    logger.info(f"N2 ROC-AUC across {N_SAMPLE_SEEDS} data-generation seeds:\n{summary.to_string(index=False)}")

    # --- Does N2 ever fail to collapse the N0 advantage, in any of the 30 draws? ---
    n0_roc = {(row["split"], row["model"]): row["roc_auc"] for _, row in per_seed_df[per_seed_df["negative_set"] == "n0"].iterrows()}
    worst_case = n2_only.groupby(["split", "model"])["roc_auc"].max().reset_index()
    worst_case["n0_roc_auc"] = worst_case.apply(lambda r: n0_roc[(r["split"], r["model"])], axis=1)
    worst_case["max_n2_roc_auc_any_seed"] = worst_case["roc_auc"]
    worst_case["collapse_holds_in_every_seed"] = worst_case["max_n2_roc_auc_any_seed"] < worst_case["n0_roc_auc"]
    worst_case = worst_case.drop(columns="roc_auc")

    # --- Figure: distribution of N2 ROC-AUC across 30 seeds, N0 as reference line, per split x model ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    model_order = ["logistic_regression", "mlp", "degree_only_lr"]
    for ax, split_name in zip(axes, ["random", "both_unseen"]):
        data_for_box = [n2_only[(n2_only["split"] == split_name) & (n2_only["model"] == m)]["roc_auc"].to_numpy() for m in model_order]
        ax.boxplot(data_for_box, tick_labels=model_order)
        for i, m in enumerate(model_order):
            ax.scatter([i + 1], [n0_roc[(split_name, m)]], color="red", marker="D", zorder=5,
                       label="N0 (fixed)" if i == 0 else None)
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
        ax.set_title(SPLIT_LABELS[split_name])
        ax.tick_params(axis="x", rotation=20)
        ax.legend(fontsize=8)
    axes[0].set_ylabel(f"ROC-AUC (N2 across {N_SAMPLE_SEEDS} data-generation seeds)")
    fig.suptitle("N2 data-generation seed sensitivity: does the N0->N2 collapse depend on one N2 draw?")
    fig.tight_layout()
    figures_dir = REPO_ROOT / "figures"
    fig.savefig(figures_dir / "negative_sampling_seed_sensitivity.pdf")
    plt.close(fig)
    logger.info(f"Wrote {figures_dir}/negative_sampling_seed_sensitivity.pdf")

    save_json(
        {
            "note": (
                "N2 regenerated under "
                f"{N_SAMPLE_SEEDS} independent data-generation seeds (sample_seed in "
                f"{SAMPLE_SEEDS[0]}..{SAMPLE_SEEDS[-1]}), holding the R0/R3 partitions, degree "
                "computation, known-pair exclusion set, and model training seed fixed -- isolating "
                "sampling-realization variance from the optimization variance scripts/18's 5 "
                "retraining seeds already covers."
            ),
            "n_sample_seeds": N_SAMPLE_SEEDS,
            "worst_case_check": worst_case.to_dict(orient="records"),
        },
        out_dir / "negative_sampling_seed_sensitivity_summary.json",
    )
    write_manifest(out_dir, config={"n_sample_seeds": N_SAMPLE_SEEDS, "data_seed": DATA_SEED, "model_seed": MODEL_SEED,
                                     "splits": ["random", "both_unseen"], "models": model_order}, seed=DATA_SEED,
                    extra={"script": "38_negative_sampling_seed_sensitivity.py"})
    logger.info("Negative-sampling seed sensitivity check complete.")


if __name__ == "__main__":
    main()
