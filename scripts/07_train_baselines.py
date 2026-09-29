"""Conventional benchmark reproduction.

Stratified 5-fold pair-level CV (Split R0 — proteins CAN appear in both train
and test folds; this is the naive baseline, NOT evidence of unseen-protein
generalization).

Representations x models, using IDENTICAL folds across every combination. Preprocessing (StandardScaler) is fit on each fold's training data
only. Threshold is fixed at 0.5, never tuned on test data.

Writes:
  data/splits/conventional_random_cv_folds.tsv        — exact fold assignment
  results/conventional_baseline_runs/<representation>__<model>/     — self-contained per-combo results
  results/conventional_baseline/all_predictions.csv   — every sample-level prediction
  results/conventional_baseline/metrics_summary.csv   — mean/std per combo
  results/conventional_baseline/calibration_summary.csv
  tables/table2_conventional_benchmark.csv
  figures/conventional_baseline_calibration_reliability.pdf
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.evaluation.calibration import calibration_metrics, reliability_curve
from src.evaluation.metrics import classification_metrics
from src.features.build_pair_features import build_pair_matrix
from src.models.classical import build_model
from src.utils.io import load_config, save_json
from src.utils.logging import get_logger
from src.utils.manifest import write_manifest
from src.utils.seed import set_seed

REPO_ROOT = Path(__file__).resolve().parents[1]
logger = get_logger("train_baselines")


def load_representation(name: str, features_dir: Path) -> pd.DataFrame:
    if name == "aac_ctd":
        aac = pd.read_csv(features_dir / "aac.csv")
        ctd = pd.read_csv(features_dir / "ctd.csv")
        return aac.merge(ctd, on="protein_id")
    return pd.read_csv(features_dir / f"{name}.csv")


def main():
    data_config = load_config(REPO_ROOT / "configs" / "data.yaml")
    config = load_config(REPO_ROOT / "configs" / "baseline.yaml")
    seed = config["seed"]
    set_seed(seed)

    pairs = pd.read_csv(
        REPO_ROOT / "data" / "processed" / "ppi_pairs_clean.tsv",
        sep="\t",
        header=None,
        names=data_config["pairs_columns"],
    )
    features_dir = REPO_ROOT / "data" / "processed" / "features"

    # --- Fold assignment: identical across every representation/model ---
    skf = StratifiedKFold(n_splits=config["n_folds"], shuffle=True, random_state=seed)
    fold_col = np.full(len(pairs), -1, dtype=int)
    for fold_id, (_, test_idx) in enumerate(skf.split(pairs, pairs["label"])):
        fold_col[test_idx] = fold_id
    assert (fold_col >= 0).all()

    splits_dir = REPO_ROOT / "data" / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    fold_assignment = pairs.assign(fold=fold_col)
    fold_path = splits_dir / "conventional_random_cv_folds.tsv"
    fold_assignment.to_csv(fold_path, sep="\t", index=False)
    logger.info(f"Wrote {fold_path}: {config['n_folds']}-fold stratified CV, seed={seed}")

    all_predictions = []
    metrics_rows = []
    calibration_rows = []

    conventional_baseline_runs_dir = REPO_ROOT / "results" / "conventional_baseline_runs"
    conventional_baseline_runs_dir.mkdir(parents=True, exist_ok=True)

    for repr_name in config["representations"]:
        protein_features = load_representation(repr_name, features_dir)
        x, y = build_pair_matrix(pairs, protein_features, fusion=config["fusion"])
        logger.info(f"[{repr_name}] pair feature matrix: {x.shape}")

        for model_name, model_params in config["models"].items():
            t0 = time.time()
            combo_dir = conventional_baseline_runs_dir / f"{repr_name}__{model_name}"
            combo_dir.mkdir(parents=True, exist_ok=True)
            combo_logger = get_logger(f"conventional_baseline_runs.{repr_name}.{model_name}", log_dir=combo_dir, filename="run.log")

            fold_predictions = []
            fold_metrics = []
            fold_calibration = []

            for fold_id in range(config["n_folds"]):
                train_mask = fold_col != fold_id
                test_mask = fold_col == fold_id

                scaler = StandardScaler()
                x_train = scaler.fit_transform(x[train_mask])
                x_test = scaler.transform(x[test_mask])
                y_train, y_test = y[train_mask], y[test_mask]

                model = build_model(model_name, model_params, seed)
                model.fit(x_train, y_train)
                y_prob = model.predict_proba(x_test)[:, 1]

                m = classification_metrics(y_test, y_prob)
                c = calibration_metrics(y_test, y_prob)
                fold_metrics.append({"fold": fold_id, **m})
                fold_calibration.append({"fold": fold_id, **c})

                fold_pairs = pairs.loc[test_mask, ["protein_a", "protein_b"]].reset_index(drop=True)
                fold_predictions.append(
                    fold_pairs.assign(
                        y_true=y_test,
                        y_probability=y_prob,
                        y_pred=(y_prob >= 0.5).astype(int),
                        fold=fold_id,
                        representation=repr_name,
                        model=model_name,
                        seed=seed,
                    )
                )
                combo_logger.info(f"fold {fold_id}: roc_auc={m['roc_auc']:.4f} pr_auc={m['pr_auc']:.4f} mcc={m['mcc']:.4f}")

            combo_predictions = pd.concat(fold_predictions, ignore_index=True)
            combo_predictions.to_csv(combo_dir / "predictions.csv", index=False)

            metrics_df = pd.DataFrame(fold_metrics)
            metric_cols = [c for c in metrics_df.columns if c != "fold"]
            summary = {
                "per_fold": fold_metrics,
                "mean": metrics_df[metric_cols].mean().to_dict(),
                "std": metrics_df[metric_cols].std().to_dict(),
            }
            save_json(summary, combo_dir / "metrics.json")

            calib_df = pd.DataFrame(fold_calibration)
            calib_df.to_csv(combo_dir / "calibration.csv", index=False)

            save_json(
                {"n_folds": config["n_folds"], "seed": seed, "fold_sizes": fold_assignment["fold"].value_counts().sort_index().to_dict()},
                combo_dir / "split_manifest.json",
            )
            write_manifest(combo_dir, config={"representation": repr_name, "model": model_name, "params": model_params, **config}, seed=seed)

            elapsed = time.time() - t0
            combo_logger.info(f"Done in {elapsed:.1f}s. Mean ROC-AUC={summary['mean']['roc_auc']:.4f} +/- {summary['std']['roc_auc']:.4f}")
            logger.info(f"[{repr_name} x {model_name}] mean ROC-AUC={summary['mean']['roc_auc']:.4f} "
                        f"PR-AUC={summary['mean']['pr_auc']:.4f} MCC={summary['mean']['mcc']:.4f} ({elapsed:.1f}s)")

            all_predictions.append(combo_predictions)
            for row in fold_metrics:
                metrics_rows.append({"representation": repr_name, "model": model_name, **row})
            for row in fold_calibration:
                calibration_rows.append({"representation": repr_name, "model": model_name, **row})

    # --- Aggregate outputs ---
    agg_dir = REPO_ROOT / "results" / "conventional_baseline"
    agg_dir.mkdir(parents=True, exist_ok=True)

    all_predictions_df = pd.concat(all_predictions, ignore_index=True)
    all_predictions_df.to_csv(agg_dir / "all_predictions.csv", index=False)

    metrics_all = pd.DataFrame(metrics_rows)
    metric_cols = [c for c in metrics_all.columns if c not in ("representation", "model", "fold")]
    metrics_summary = metrics_all.groupby(["representation", "model"])[metric_cols].agg(["mean", "std"])
    metrics_summary.columns = ["_".join(c) for c in metrics_summary.columns]
    metrics_summary = metrics_summary.reset_index()
    metrics_summary.to_csv(agg_dir / "metrics_summary.csv", index=False)

    calibration_all = pd.DataFrame(calibration_rows)
    calibration_summary = calibration_all.groupby(["representation", "model"])[["brier_score", "ece"]].agg(["mean", "std"])
    calibration_summary.columns = ["_".join(c) for c in calibration_summary.columns]
    calibration_summary = calibration_summary.reset_index()
    calibration_summary.to_csv(agg_dir / "calibration_summary.csv", index=False)

    # --- Table 2: conventional benchmark performance (manuscript-ready) ---
    tables_dir = REPO_ROOT / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    table2 = metrics_summary.merge(calibration_summary, on=["representation", "model"])
    table2.to_csv(tables_dir / "table2_conventional_benchmark.csv", index=False)
    logger.info(f"Wrote {tables_dir / 'table2_conventional_benchmark.csv'}")

    # --- Figure 7: reliability diagrams for the AAC+CTD representation across all models ---
    figures_dir = REPO_ROOT / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True)
    for ax, model_name in zip(axes.flat, config["models"].keys()):
        combo_df = all_predictions_df[
            (all_predictions_df["representation"] == "aac_ctd") & (all_predictions_df["model"] == model_name)
        ]
        conf, acc, counts = reliability_curve(combo_df["y_true"].to_numpy(), combo_df["y_probability"].to_numpy())
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
        ax.plot(conf, acc, marker="o", color="#2c7fb8")
        ax.set_title(model_name)
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed frequency")
    fig.suptitle("Reliability diagrams (AAC+CTD representation, pooled across CV folds)")
    fig.tight_layout()
    fig.savefig(figures_dir / "conventional_baseline_calibration_reliability.pdf")
    plt.close(fig)
    logger.info(f"Wrote {figures_dir / 'conventional_baseline_calibration_reliability.pdf'}")

    write_manifest(agg_dir, config=config, seed=seed, extra={"script": "07_train_baselines.py"})
    logger.info("Conventional benchmark complete.")


if __name__ == "__main__":
    main()
