"""Negative-sampling bias study.

Constructs 4 alternative negative sets alongside the benchmark's own (N0):
  N1 random unknown pairs
  N2 degree-matched (one-sided substitution on a positive pair's degree decile)
  N3 length-matched (one-sided substitution on sequence-length decile)
  N5 hard negatives (substitution matched on BOTH degree and length decile)
N4 (localization-aware) is explicitly skipped -- this dataset has no GO/
localization/complex data, and an analysis the data cannot support
is not fabricated.

Terminology: every constructed set is an "unknown/unobserved pair used as a
negative," never a "confirmed non-interaction" -- see configs/hard_negative.yaml.

Two experiments:
  A) Train once on N0, evaluate on test sets built from N0/N1/N2/N3/N5
     (same positives, swapped negatives) -- does apparent performance
     depend on what the evaluator calls a negative?
  B) Train separately on N0/N1/N2/N3/N5, evaluate every model on the SAME
     common benchmark (N0's own test negatives) -- does training with
     harder negatives change performance on the standard benchmark?

Writes:
  data/processed/negative_sets/{n1,n2,n3,n5}.tsv
  results/negative_sampling_study/...
  tables/table7_negative_sampling_robustness.csv
  figures/figure6_hard_negative_analysis.pdf
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data.negative_sampling import (
    _known_pair_set,
    compute_protein_stats,
    sample_matched_negatives,
    sample_random_negatives,
)
from src.evaluation.calibration import calibration_metrics
from src.evaluation.metrics import classification_metrics
from src.features.build_pair_features import build_pair_matrix
from src.features.esm2 import load_esm2_features
from src.models.classical import build_model
from src.utils.io import load_config, save_json
from src.utils.logging import get_logger
from src.utils.manifest import write_manifest
from src.utils.seed import set_seed

REPO_ROOT = Path(__file__).resolve().parents[1]
logger = get_logger("negative_sampling_study")

NEGATIVE_SET_LABELS = {
    "n0": "N0 (benchmark original)",
    "n1": "N1 (random unknown)",
    "n2": "N2 (degree-matched)",
    "n3": "N3 (length-matched)",
    "n5": "N5 (hard: degree+length)",
}


def main():
    data_config = load_config(REPO_ROOT / "configs" / "data.yaml")
    config = load_config(REPO_ROOT / "configs" / "hard_negative.yaml")
    seed = config["seed"]
    set_seed(seed)

    pairs = pd.read_csv(REPO_ROOT / "data" / "processed" / "ppi_pairs_clean.tsv", sep="\t", header=None, names=data_config["pairs_columns"])
    proteins = pd.read_csv(REPO_ROOT / "data" / "processed" / "proteins_with_groups.tsv", sep="\t")

    pos = pairs[pairs["label"] == 1].reset_index(drop=True)
    neg0 = pairs[pairs["label"] == 0].reset_index(drop=True)
    logger.info(f"Canonical dataset: {len(pos)} positives, {len(neg0)} original (N0) negatives")

    protein_stats = compute_protein_stats(proteins, pos, n_deciles=config["n_deciles"])
    known_pairs = _known_pair_set(pos)

    neg_sets_dir = REPO_ROOT / "data" / "processed" / "negative_sets"
    neg_sets_dir.mkdir(parents=True, exist_ok=True)

    n1 = sample_random_negatives(list(proteins["protein_id"]), known_pairs, len(pos), seed)
    n2 = sample_matched_negatives(pos, protein_stats, known_pairs, ["degree_decile"], seed)
    n3 = sample_matched_negatives(pos, protein_stats, known_pairs, ["length_decile"], seed)
    n5 = sample_matched_negatives(pos, protein_stats, known_pairs, ["degree_decile", "length_decile"], seed)

    negative_sets = {"n0": neg0, "n1": n1, "n2": n2, "n3": n3, "n5": n5}
    for name, df in negative_sets.items():
        if name != "n0":
            df.to_csv(neg_sets_dir / f"{name}.tsv", sep="\t", index=False)
        logger.info(f"{NEGATIVE_SET_LABELS[name]}: {len(df)} pairs")
        overlap = set(zip(df["protein_a"], df["protein_b"])) & known_pairs
        assert not overlap, f"{name} contains {len(overlap)} pairs that are actually known positives"

    # --- Shared positive train/test split, reused across every negative set ---
    pos_train, pos_test = train_test_split(pos, test_size=config["test_fraction"], random_state=seed)

    train_sets, test_sets = {}, {}
    for name, neg_df in negative_sets.items():
        neg_train, neg_test = train_test_split(neg_df, test_size=config["test_fraction"], random_state=seed)
        train_sets[name] = pd.concat([pos_train, neg_train], ignore_index=True)
        test_sets[name] = pd.concat([pos_test, neg_test], ignore_index=True)

    protein_features = load_esm2_features(config["representation"].replace("esm2_", ""), REPO_ROOT / "embeddings" / "esm2")

    def fit_and_get_probs(train_df):
        x_train, y_train = build_pair_matrix(train_df, protein_features, fusion=config["fusion"])
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        model = build_model(config["model"], config["model_params"], seed)
        model.fit(x_train, y_train)
        return model, scaler

    def evaluate(model, scaler, test_df):
        x_test, y_test = build_pair_matrix(test_df, protein_features, fusion=config["fusion"])
        x_test = scaler.transform(x_test)
        y_prob = model.predict_proba(x_test)[:, 1]
        return {**classification_metrics(y_test, y_prob), **calibration_metrics(y_test, y_prob)}

    negative_sampling_study_dir = REPO_ROOT / "results" / "negative_sampling_study"
    negative_sampling_study_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    # --- Experiment A: train once on N0, evaluate across all test sets ---
    logger.info("=== Experiment A: train on N0, evaluate on N0/N1/N2/N3/N5 test sets ===")
    model_n0, scaler_n0 = fit_and_get_probs(train_sets["n0"])
    for name in negative_sets:
        m = evaluate(model_n0, scaler_n0, test_sets[name])
        logger.info(f"[A: trained N0 -> tested {name}] ROC-AUC={m['roc_auc']:.4f} PR-AUC={m['pr_auc']:.4f} MCC={m['mcc']:.4f}")
        rows.append({"experiment": "A_train_n0_eval_swap", "trained_on": "n0", "evaluated_on": name, **m})
    save_json({}, negative_sampling_study_dir / "experiment_a_done.json")

    # --- Experiment B: train separately on each set, evaluate all on N0's test set (common benchmark) ---
    logger.info("=== Experiment B: train on each negative set, evaluate all on N0 (common benchmark) ===")
    for name in negative_sets:
        model, scaler = fit_and_get_probs(train_sets[name])
        m = evaluate(model, scaler, test_sets["n0"])
        logger.info(f"[B: trained {name} -> tested N0] ROC-AUC={m['roc_auc']:.4f} PR-AUC={m['pr_auc']:.4f} MCC={m['mcc']:.4f}")
        rows.append({"experiment": "B_train_each_eval_n0", "trained_on": name, "evaluated_on": "n0", **m})

    results_df = pd.DataFrame(rows)
    tables_dir = REPO_ROOT / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(tables_dir / "table7_negative_sampling_robustness.csv", index=False)
    logger.info(f"Wrote {tables_dir}/table7_negative_sampling_robustness.csv")

    save_json(
        {
            "negative_set_sizes": {k: len(v) for k, v in negative_sets.items()},
            "N4_localization_aware": (
                "Skipped: this dataset has no GO annotation, subcellular localization, "
                "or protein-complex-membership data (confirmed absent in the dataset audit). "
                "Constructing localization-aware negatives would require fabricating "
                "biological assumptions not supported by the available data."
            ),
            "terminology_note": (
                "N1/N2/N3/N5 are constructed unknown/unobserved pairs used as negatives "
                "for training/evaluation -- not confirmed non-interactions. Only N0 carries "
                "whatever evidentiary status the original Guo et al. benchmark gave it "
                "(undocumented in the source data)."
            ),
        },
        REPO_ROOT / "results" / "negative_sampling_summary.json",
    )

    # --- Figure 6: hard-negative analysis ---
    figures_dir = REPO_ROOT / "figures"
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    exp_a = results_df[results_df["experiment"] == "A_train_n0_eval_swap"]
    labels = [NEGATIVE_SET_LABELS[n].split(" (")[0] for n in exp_a["evaluated_on"]]
    axes[0].bar(labels, exp_a["roc_auc"], color="#1b9e77")
    axes[0].set_title("Exp A: trained on N0,\nevaluated on swapped test negatives")
    axes[0].set_ylabel("ROC-AUC")
    axes[0].axhline(0.5, color="gray", linestyle="--")
    axes[0].tick_params(axis="x", rotation=45)

    exp_b = results_df[results_df["experiment"] == "B_train_each_eval_n0"]
    labels_b = [NEGATIVE_SET_LABELS[n].split(" (")[0] for n in exp_b["trained_on"]]
    axes[1].bar(labels_b, exp_b["roc_auc"], color="#d95f02")
    axes[1].set_title("Exp B: trained on each negative set,\nevaluated on N0 (common benchmark)")
    axes[1].set_ylabel("ROC-AUC")
    axes[1].axhline(0.5, color="gray", linestyle="--")
    axes[1].tick_params(axis="x", rotation=45)

    fig.suptitle("Negative-sampling bias study (ESM-2 mean + MLP, Random split)")
    fig.tight_layout()
    fig.savefig(figures_dir / "figure6_hard_negative_analysis.pdf")
    plt.close(fig)
    logger.info(f"Wrote {figures_dir}/figure6_hard_negative_analysis.pdf")

    write_manifest(negative_sampling_study_dir, config=config, seed=seed, extra={"script": "11_negative_sampling_study.py"})
    logger.info("Negative-sampling bias study complete.")


if __name__ == "__main__":
    main()
