"""Probability calibration.

Objective: do predicted probabilities correspond to
empirical interaction likelihood, and does Platt/isotonic recalibration
improve them -- fit strictly on a validation holdout carved from training
data, never on the final test set?

Scope: reuses exactly the architecture ablation's R3 (both-unseen) x {N0, N2} x 6-model
design (`build_r3_n0_n2_datasets`, ESM-2 mean pooling, "combined" fusion for
classical models) -- the same architectures, same negative-set framing,
because the negative-bias re-validation and the statistical analysis established that N0
alone is not a trustworthy regime to draw conclusions from. Unlike the architecture ablation,
each model's *training* set here is further split into a fit set (80%) and
a calibration holdout (20%); the calibration holdout is never used for
gradient/parameter fitting, only for fitting the Platt/isotonic maps.
the architecture ablation's own numbers (full training set, no holdout) remain the
project's primary architecture-comparison result -- this analysis answers a
different, calibration-specific question and is expected to show slightly
different (not identical) uncalibrated test metrics due to the smaller fit
set, which is documented rather than treated as a discrepancy.

Writes:
  results/calibration/<negset>__<model>/
  tables/table8_calibration.csv
  figures/figure7_calibration.pdf
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

from src.data.negative_sampling import _known_pair_set, build_r3_n0_n2_datasets, compute_protein_stats
from src.evaluation.calibration import (
    apply_isotonic_calibrator,
    apply_platt_calibrator,
    calibration_metrics,
    calibration_slope_intercept,
    fit_isotonic_calibrator,
    fit_platt_calibrator,
)
from src.evaluation.metrics import classification_metrics
from src.features.build_pair_features import build_pair_matrix
from src.features.esm2 import load_esm2_features
from src.models.attention import AttentionFusion
from src.models.classical import build_model
from src.models.interaction import CrossInteractionModule
from src.models.siamese import SharedEncoderFusion
from src.models.torch_utils import predict_proba as torch_predict_proba
from src.models.torch_utils import train_pair_model
from src.utils.io import save_json
from src.utils.logging import get_logger
from src.utils.manifest import write_manifest
from src.utils.seed import set_seed

REPO_ROOT = Path(__file__).resolve().parents[1]
logger = get_logger("calibration_analysis")

SEED = 42
TEST_FRACTION = 0.20
CALIB_FRACTION = 0.20
FUSION = "combined"
CLASSICAL_MODELS = {
    "logistic_regression": {"max_iter": 2000},
    "random_forest": {"n_estimators": 300, "n_jobs": -1},
    "mlp": {"hidden_layer_sizes": [128, 64], "max_iter": 500, "early_stopping": True},
}
TORCH_MODELS = {
    "siamese_mlp": lambda dim: SharedEncoderFusion(dim, hidden_dim=256, latent_dim=128),
    "attention_fusion": lambda dim: AttentionFusion(dim, proj_dim=128),
    "cross_interaction": lambda dim: CrossInteractionModule(dim, interaction_dim=128),
}
NEG_SET_LABELS = {"n0": "N0 (original)", "n2": "N2 (degree-matched)"}
MODEL_ORDER = list(CLASSICAL_MODELS) + list(TORCH_MODELS)
# Reliability-diagram panel selection: weakest classical baseline, and the
# strongest N0/N2 performer respectively per the architecture ablation's table6 results.
FIGURE_MODELS = ["logistic_regression", "siamese_mlp", "attention_fusion"]


def lookup_embeddings(pairs, protein_features):
    feature_cols = [c for c in protein_features.columns if c != "protein_id"]
    feat_by_id = protein_features.set_index("protein_id")[feature_cols]
    h_a = feat_by_id.loc[pairs["protein_a"]].to_numpy(dtype="float32")
    h_b = feat_by_id.loc[pairs["protein_b"]].to_numpy(dtype="float32")
    return h_a, h_b, feature_cols


def _split_fit_calib(train_df, seed):
    fit_df, calib_df = train_test_split(
        train_df, test_size=CALIB_FRACTION, random_state=seed, stratify=train_df["label"]
    )
    return fit_df.reset_index(drop=True), calib_df.reset_index(drop=True)


def _calibration_row(y_true, y_prob, label):
    m = {"calibration_method": label}
    m.update(classification_metrics(y_true, y_prob))
    m.update(calibration_metrics(y_true, y_prob))
    m.update(calibration_slope_intercept(y_true, y_prob))
    return m


def main():
    set_seed(SEED)
    proteins = pd.read_csv(REPO_ROOT / "data" / "processed" / "proteins_with_groups.tsv", sep="\t")
    pairs_cols = ["protein_a", "protein_b", "label"]
    pairs = pd.read_csv(REPO_ROOT / "data" / "processed" / "ppi_pairs_clean.tsv", sep="\t", header=None, names=pairs_cols)
    pos = pairs[pairs["label"] == 1].reset_index(drop=True)

    protein_stats = compute_protein_stats(proteins, pos, n_deciles=10)
    known_pairs = _known_pair_set(pos)
    protein_features = load_esm2_features("mean", REPO_ROOT / "embeddings" / "esm2")

    datasets = build_r3_n0_n2_datasets(protein_stats, known_pairs, proteins, REPO_ROOT, SEED, TEST_FRACTION)

    out_dir = REPO_ROOT / "results" / "calibration"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    reliability_data = {}  # (neg_name, model_name) -> dict of raw/isotonic reliability curves

    for neg_name, (train_df, test_df) in datasets.items():
        fit_df, calib_df = _split_fit_calib(train_df, SEED)
        y_test = test_df["label"].to_numpy()

        # --- Classical models ---
        for model_name, params in CLASSICAL_MODELS.items():
            x_fit, y_fit = build_pair_matrix(fit_df, protein_features, fusion=FUSION)
            x_calib, y_calib = build_pair_matrix(calib_df, protein_features, fusion=FUSION)
            x_test, _ = build_pair_matrix(test_df, protein_features, fusion=FUSION)
            scaler = StandardScaler()
            x_fit = scaler.fit_transform(x_fit)
            x_calib = scaler.transform(x_calib)
            x_test = scaler.transform(x_test)

            model = build_model(model_name, params, SEED)
            model.fit(x_fit, y_fit)
            p_calib = model.predict_proba(x_calib)[:, 1]
            p_test = model.predict_proba(x_test)[:, 1]
            _record(out_dir, rows, reliability_data, neg_name, model_name, y_calib, p_calib, y_test, p_test, test_df, len(fit_df), len(calib_df))

        # --- Torch architectures ---
        fit_proteins = pd.unique(pd.concat([fit_df["protein_a"], fit_df["protein_b"]]))
        feature_cols = [c for c in protein_features.columns if c != "protein_id"]
        scaler = StandardScaler()
        scaler.fit(protein_features.set_index("protein_id").loc[fit_proteins, feature_cols])
        scaled_features = protein_features.copy()
        scaled_features[feature_cols] = scaler.transform(protein_features[feature_cols])

        h_a_fit, h_b_fit, _ = lookup_embeddings(fit_df, scaled_features)
        h_a_calib, h_b_calib, _ = lookup_embeddings(calib_df, scaled_features)
        h_a_test, h_b_test, _ = lookup_embeddings(test_df, scaled_features)
        y_fit_raw = fit_df["label"].to_numpy().astype("float32")
        y_calib = calib_df["label"].to_numpy()

        for model_name, model_fn in TORCH_MODELS.items():
            model = train_pair_model(
                lambda mf=model_fn: mf(len(feature_cols)),
                h_a_fit, h_b_fit, y_fit_raw, seed=SEED,
            )
            p_calib = torch_predict_proba(model, h_a_calib, h_b_calib)
            p_test = torch_predict_proba(model, h_a_test, h_b_test)
            _record(out_dir, rows, reliability_data, neg_name, model_name, y_calib, p_calib, y_test, p_test, test_df, len(fit_df), len(calib_df))

    results_df = pd.DataFrame(rows)
    tables_dir = REPO_ROOT / "tables"
    results_df.to_csv(tables_dir / "table8_calibration.csv", index=False)
    logger.info(f"Wrote {tables_dir}/table8_calibration.csv")

    _plot_reliability_grid(reliability_data, REPO_ROOT / "figures" / "figure7_calibration.pdf")
    logger.info(f"Wrote {REPO_ROOT}/figures/figure7_calibration.pdf")

    write_manifest(out_dir, config={"models": MODEL_ORDER, "negative_sets": ["n0", "n2"], "split": "both_unseen",
                                     "calib_fraction": CALIB_FRACTION}, seed=SEED)
    logger.info("Calibration analysis complete.")


def _record(out_dir, rows, reliability_data, neg_name, model_name, y_calib, p_calib, y_test, p_test, test_df, n_fit, n_calib):
    platt = fit_platt_calibrator(y_calib, p_calib)
    isotonic = fit_isotonic_calibrator(y_calib, p_calib)
    p_test_platt = apply_platt_calibrator(platt, p_test)
    p_test_isotonic = apply_isotonic_calibrator(isotonic, p_test)

    combo_dir = out_dir / f"{neg_name}__{model_name}"
    combo_dir.mkdir(parents=True, exist_ok=True)
    predictions = pd.DataFrame({
        "protein_a": test_df["protein_a"].to_numpy(),
        "protein_b": test_df["protein_b"].to_numpy(),
        "y_true": y_test,
        "y_probability_raw": p_test,
        "y_probability_platt": p_test_platt,
        "y_probability_isotonic": p_test_isotonic,
    })
    predictions.to_csv(combo_dir / "predictions.csv", index=False)

    combo_rows = [
        _calibration_row(y_test, p_test, "none"),
        _calibration_row(y_test, p_test_platt, "platt"),
        _calibration_row(y_test, p_test_isotonic, "isotonic"),
    ]
    save_json({"n_fit": n_fit, "n_calib": n_calib, "n_test": len(y_test), "results": combo_rows},
              combo_dir / "metrics.json")
    write_manifest(combo_dir, config={"negative_set": neg_name, "model": model_name}, seed=SEED)

    for r in combo_rows:
        rows.append({"negative_set": neg_name, "negative_set_label": NEG_SET_LABELS[neg_name],
                     "model": model_name, "n_fit": n_fit, "n_calib": n_calib, "n_test": len(y_test), **r})

    logger.info(f"[{NEG_SET_LABELS[neg_name]} x {model_name}] ROC-AUC={combo_rows[0]['roc_auc']:.3f} "
                f"ECE raw={combo_rows[0]['ece']:.3f} platt={combo_rows[1]['ece']:.3f} isotonic={combo_rows[2]['ece']:.3f}")

    if model_name in FIGURE_MODELS:
        from src.evaluation.calibration import reliability_curve
        reliability_data[(neg_name, model_name)] = {
            "raw": reliability_curve(y_test, p_test),
            "isotonic": reliability_curve(y_test, p_test_isotonic),
        }


def _plot_reliability_grid(reliability_data, out_path):
    fig, axes = plt.subplots(2, len(FIGURE_MODELS), figsize=(4 * len(FIGURE_MODELS), 8), sharex=True, sharey=True)
    for row, neg_name in enumerate(["n0", "n2"]):
        for col, model_name in enumerate(FIGURE_MODELS):
            ax = axes[row][col]
            ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
            data = reliability_data.get((neg_name, model_name))
            if data is not None:
                conf_raw, acc_raw, _ = data["raw"]
                conf_iso, acc_iso, _ = data["isotonic"]
                ax.plot(conf_raw, acc_raw, "o-", label="Raw", color="tab:red", alpha=0.8)
                ax.plot(conf_iso, acc_iso, "o-", label="Isotonic-calibrated", color="tab:blue", alpha=0.8)
            ax.set_title(f"{NEG_SET_LABELS[neg_name]}\n{model_name}", fontsize=10)
            if row == 1:
                ax.set_xlabel("Mean predicted probability")
            if col == 0:
                ax.set_ylabel("Empirical fraction positive")
    axes[0][0].legend(fontsize=8, loc="upper left")
    fig.suptitle("the calibration analysis: reliability diagrams (R3 both-unseen), raw vs. isotonic-calibrated")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


if __name__ == "__main__":
    main()
