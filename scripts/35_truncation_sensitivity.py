"""ESM-2 truncation-strategy sensitivity analysis: Section 2.4 caps ESM-2 input at 1,022 residues, truncated from the
C-terminus (i.e. the first 1,022 residues, "N-terminal," are kept) for the
268/2,497 proteins (10.7%) exceeding this cap. The concern:
C-terminal truncation can remove interaction domains, transmembrane
regions, disordered regions, localization signals, or domain boundaries, so
this study's reports of weak/collapsing ESM-2 performance may partly
reflect representation truncation rather than a genuine absence of
sequence-based signal.

Four checks, all restricted to the 268 affected proteins (the other 2,229
already fit inside the cap and are unaffected by any of these variants):
  n_terminal    this study's existing default (first 1,022 kept) -- baseline
  c_terminal    LAST 1,022 residues kept instead
  full          no truncation at all (feasible here: the single longest
                sequence, Q12019 at 4,910 aa, embeds in <6s / <1GB RSS on
                this project's CPU-only environment)
  segment_mean  non-overlapping 1,022-residue chunks, each mean-pooled
                independently, then length-weighted averaged -- a
                sliding-window/segment-pooling operationalization

For each variant, only the 268 affected proteins' mean-pooled ESM-2 vectors
are replaced; the other 2,229 proteins' cached embeddings are
reused unchanged. Re-trains Section 3.4's exact MLP protocol (script
09_plm_representation_comparison.py: same model, params, fusion, seed) on
R0 and R3, under each variant, and reports both the overall test-set metric
and a length-stratified breakdown (pairs touching >=1 affected protein vs.
pairs that do not) -- a length-stratified
performance comparison using the same models.

Writes:
  embeddings/esm2_truncation_variants/{c_terminal,full,segment_mean}.h5
  tables/truncation_sensitivity.csv
  figures/truncation_sensitivity.pdf
  results/truncation_sensitivity/
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import h5py
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.evaluation.metrics import classification_metrics
from src.features.build_pair_features import build_pair_matrix
from src.features.esm2 import embed_sequence_variant, load_esm2_features, load_model
from src.models.classical import build_model
from src.utils.io import load_config, save_json
from src.utils.logging import get_logger
from src.utils.manifest import write_manifest
from src.utils.seed import set_seed

REPO_ROOT = Path(__file__).resolve().parents[1]
logger = get_logger("truncation_sensitivity")

SPLITS = [("random", "Random (R0)"), ("both_unseen", "Both-unseen (R3)")]
VARIANTS = ["n_terminal", "c_terminal", "full", "segment_mean"]
MLP_PARAMS = {"hidden_layer_sizes": [128, 64], "max_iter": 500, "early_stopping": True}
SEED = 42
FUSION = "combined"


def compute_variant_embeddings(affected_proteins: pd.DataFrame, esm2_config: dict) -> dict[str, dict[str, np.ndarray]]:
    """Returns {strategy: {protein_id: vector}} for c_terminal/full/segment_mean
    (n_terminal reuses the existing ESM-2 cache -- no recomputation needed)."""
    torch.set_num_threads(4)
    tokenizer, model = load_model(esm2_config["model_name"])
    max_length = esm2_config["max_length"]

    out = {v: {} for v in ["c_terminal", "full", "segment_mean"]}
    t0 = time.time()
    for i, row in enumerate(affected_proteins.itertuples()):
        for strategy in out:
            out[strategy][row.protein_id] = embed_sequence_variant(tokenizer, model, row.sequence, strategy, max_length)
        if (i + 1) % 20 == 0:
            logger.info(f"  embedded {i + 1}/{len(affected_proteins)} affected proteins ({time.time() - t0:.0f}s elapsed)")
    logger.info(f"Computed c_terminal/full/segment_mean embeddings for {len(affected_proteins)} proteins in {time.time() - t0:.0f}s")
    return out


def build_variant_feature_table(base_features: pd.DataFrame, variant_vectors: dict[str, np.ndarray]) -> pd.DataFrame:
    """Copies base_features and overwrites
    only the rows for proteins present in variant_vectors."""
    feature_cols = [c for c in base_features.columns if c != "protein_id"]
    df = base_features.set_index("protein_id").copy()
    for protein_id, vec in variant_vectors.items():
        df.loc[protein_id, feature_cols] = vec
    return df.reset_index()


def main():
    esm2_config = load_config(REPO_ROOT / "configs" / "esm2.yaml")
    set_seed(SEED)

    proteins = pd.read_csv(REPO_ROOT / "data" / "processed" / "proteins_with_groups.tsv", sep="\t")
    proteins["length"] = proteins["sequence"].str.len()
    affected = proteins[proteins["length"] > esm2_config["max_length"]].reset_index(drop=True)
    affected_ids = set(affected["protein_id"])
    logger.info(f"{len(affected)}/{len(proteins)} proteins exceed max_length={esm2_config['max_length']} and will get alternate embeddings")

    variants_dir = REPO_ROOT / "embeddings" / "esm2_truncation_variants"
    variants_dir.mkdir(parents=True, exist_ok=True)
    cache_path = variants_dir / "variant_vectors.h5"
    if cache_path.exists():
        variant_vectors = {}
        with h5py.File(cache_path) as f:
            for strategy in ["c_terminal", "full", "segment_mean"]:
                grp = f[strategy]
                variant_vectors[strategy] = {pid: grp[pid][:] for pid in grp}
        logger.info(f"Loaded cached variant embeddings from {cache_path}")
    else:
        variant_vectors = compute_variant_embeddings(affected, esm2_config)
        with h5py.File(cache_path, "w") as f:
            for strategy, vecs in variant_vectors.items():
                grp = f.create_group(strategy)
                for pid, vec in vecs.items():
                    grp.create_dataset(pid, data=vec)
        logger.info(f"Wrote {cache_path}")

    base_mean_features = load_esm2_features("mean", REPO_ROOT / "embeddings" / "esm2")
    feature_tables = {"n_terminal": base_mean_features}
    for strategy in ["c_terminal", "full", "segment_mean"]:
        feature_tables[strategy] = build_variant_feature_table(base_mean_features, variant_vectors[strategy])

    def is_affected_pair(pairs_df: pd.DataFrame) -> pd.Series:
        return pairs_df["protein_a"].isin(affected_ids) | pairs_df["protein_b"].isin(affected_ids)

    rows = []
    for split_name, split_label in SPLITS:
        split_dir = REPO_ROOT / "data" / "splits" / split_name
        train_pairs = pd.read_csv(split_dir / "train.tsv", sep="\t")
        test_pairs = pd.read_csv(split_dir / "test.tsv", sep="\t")
        test_affected_mask = is_affected_pair(test_pairs)
        n_affected_test = int(test_affected_mask.sum())
        logger.info(f"[{split_label}] {n_affected_test}/{len(test_pairs)} test pairs touch >=1 truncation-affected protein")

        for variant in VARIANTS:
            protein_features = feature_tables[variant]
            x_train, y_train = build_pair_matrix(train_pairs, protein_features, fusion=FUSION)
            x_test, y_test = build_pair_matrix(test_pairs, protein_features, fusion=FUSION)

            scaler = StandardScaler()
            x_train_scaled = scaler.fit_transform(x_train)
            x_test_scaled = scaler.transform(x_test)

            model = build_model("mlp", MLP_PARAMS, SEED)
            model.fit(x_train_scaled, y_train)
            y_prob = model.predict_proba(x_test_scaled)[:, 1]

            m_overall = classification_metrics(y_test, y_prob)
            m_affected = classification_metrics(y_test[test_affected_mask.to_numpy()], y_prob[test_affected_mask.to_numpy()])
            m_unaffected = classification_metrics(y_test[~test_affected_mask.to_numpy()], y_prob[~test_affected_mask.to_numpy()])

            logger.info(
                f"[{split_label} x {variant}] overall ROC-AUC={m_overall['roc_auc']:.4f} | "
                f"affected-pairs ROC-AUC={m_affected['roc_auc']:.4f} (n={n_affected_test}) | "
                f"unaffected-pairs ROC-AUC={m_unaffected['roc_auc']:.4f} (n={len(test_pairs) - n_affected_test})"
            )
            for subset_name, m, n in [("overall", m_overall, len(test_pairs)), ("affected_pairs", m_affected, n_affected_test), ("unaffected_pairs", m_unaffected, len(test_pairs) - n_affected_test)]:
                rows.append({"split": split_name, "split_label": split_label, "variant": variant, "subset": subset_name, "n_test": n, **m})

    results_df = pd.DataFrame(rows)
    tables_dir = REPO_ROOT / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(tables_dir / "truncation_sensitivity.csv", index=False)
    logger.info(f"Wrote {tables_dir}/truncation_sensitivity.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    variant_labels = {"n_terminal": "N-term\n(default)", "c_terminal": "C-term", "full": "Full", "segment_mean": "Segment\nmean"}
    for ax, (split_name, split_label) in zip(axes, SPLITS):
        sub = results_df[(results_df["split"] == split_name) & (results_df["subset"] == "affected_pairs")].set_index("variant")
        x = np.arange(len(VARIANTS))
        ax.bar(x, [sub.loc[v, "roc_auc"] for v in VARIANTS], color="#7570b3")
        ax.axhline(0.5, color="gray", linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels([variant_labels[v] for v in VARIANTS])
        ax.set_title(f"{split_label}\n(truncation-affected test pairs only)")
        ax.set_ylabel("ROC-AUC")
    fig.suptitle("ESM-2 truncation-strategy sensitivity (affected-pair subset)")
    fig.tight_layout()
    figures_dir = REPO_ROOT / "figures"
    fig.savefig(figures_dir / "truncation_sensitivity.pdf")
    plt.close(fig)
    logger.info(f"Wrote {figures_dir}/truncation_sensitivity.pdf")

    out_dir = REPO_ROOT / "results" / "truncation_sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        {
            "n_affected_proteins": len(affected),
            "max_length": esm2_config["max_length"],
            "note": (
                "Re-embeds the 268/2,497 (10.7%) "
                "proteins exceeding ESM-2's 1,022-residue cap (Section 2.4) under "
                "three alternate strategies (C-terminal-kept, full-length, "
                "segment-mean-pooled) in place of the default (N-terminal-kept), "
                "leaving all other proteins' cached embeddings unchanged, and "
                "re-trains Section 3.4's exact MLP protocol on R0/R3 under each. "
                "Reports both overall and length-stratified (affected vs. "
                "unaffected test pairs) performance."
            ),
        },
        out_dir / "truncation_sensitivity_summary.json",
    )
    write_manifest(out_dir, config={"variants": VARIANTS, "splits": [s for s, _ in SPLITS], "mlp_params": MLP_PARAMS}, seed=SEED, extra={"script": "35_truncation_sensitivity.py"})
    logger.info("Truncation-strategy sensitivity analysis complete.")


if __name__ == "__main__":
    main()
