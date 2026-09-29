"""Generate the central manuscript figure exclusively from saved result tables."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main():
    gen = pd.read_csv(ROOT/"tables/table3_protein_generalization.csv")
    sim = pd.read_csv(ROOT/"tables/table4_homology_controlled.csv")
    neg = pd.read_csv(ROOT/"tables/negative_bias_revalidation_summary.csv")
    topo = pd.read_csv(ROOT/"tables/topology_only_baseline.csv")
    multi = pd.read_csv(ROOT/"tables/multi_species_negative_sampling.csv")

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
    mlp = gen[gen.model == "mlp"].set_index("split")
    axes[0].plot([0,1,2], mlp.loc[["seen_seen","one_unseen","both_unseen"],"roc_auc"], marker="o")
    axes[0].set_xticks([0,1,2], ["R1","R2","R3"])
    axes[0].set_title("Protein novelty")
    axes[0].set_ylabel("ROC-AUC")

    for model, group in sim.groupby("model"):
        axes[1].plot(group.difficulty, group.roc_auc, marker="o", alpha=.75, label=model.replace("_"," "))
    axes[1].set_xticks([4,5,6,7], ["R3-50","R3-40","R3-30","R3-20"], rotation=35)
    axes[1].set_title("Similarity-controlled R3")

    d = topo[(topo.feature_source == "degree") & (topo.model == "logistic_regression") &
             (topo.experiment == "A_train_n0_eval_swap") & topo.evaluated_on.isin(["n0","n2"])]
    axes[2].bar(["N0","N2"], d.set_index("evaluated_on").loc[["n0","n2"],"roc_auc"], color=["#4c78a8","#f58518"])
    axes[2].set_title("Degree-only Guo")

    m = multi[(multi.representation == "esm2_mean") & (multi.model == "mlp")]
    axes[3].bar(["N0","N2"], m.set_index("negative_set").loc[["n0","n2"],"roc_auc"], color=["#4c78a8","#f58518"])
    axes[3].set_title("ESM-2 multi-species")

    for ax in axes:
        ax.axhline(.5, color="gray", linestyle="--", linewidth=1)
        ax.set_ylim(.4,1.0)
        ax.spines[["top","right"]].set_visible(False)
    axes[1].legend(fontsize=6, frameon=False)
    fig.suptitle("Complementary benchmark controls reveal benchmark-specific signal")
    fig.tight_layout()
    fig.savefig(ROOT/"figures/figure_main_central_findings.pdf")
    fig.savefig(ROOT/"figures/figure_main_central_findings.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
