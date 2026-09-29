"""Diagnose below-chance score ordering under controlled negatives."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]


def effect_size(pos, neg):
    pooled = np.sqrt(((len(pos)-1)*pos.var(ddof=1) + (len(neg)-1)*neg.var(ddof=1)) / (len(pos)+len(neg)-2))
    return (pos.mean()-neg.mean())/pooled if pooled else np.nan


def summarize(path, split, negative_set, representation, model, probability_col="y_probability"):
    df = pd.read_csv(path)
    y = df["y_true"].to_numpy()
    prob = df[probability_col].to_numpy()
    pos, neg = prob[y == 1], prob[y == 0]
    auc = roc_auc_score(y, prob)
    return df, {
        "split": split, "negative_set": negative_set, "representation": representation,
        "model": model, "source": str(path.relative_to(ROOT)), "n": len(df),
        "auc_original": auc, "auc_inverted": roc_auc_score(y, 1-prob),
        "mean_probability_positive": pos.mean(), "mean_probability_negative": neg.mean(),
        "median_probability_positive": np.median(pos), "median_probability_negative": np.median(neg),
        "iqr_probability_positive": np.quantile(pos, .75)-np.quantile(pos, .25),
        "iqr_probability_negative": np.quantile(neg, .75)-np.quantile(neg, .25),
        "cohens_d_positive_minus_negative": effect_size(pos, neg),
        "ordering": ("reversed_material" if auc <= .47 else
                     "near_chance" if abs(auc-.5) < .03 else "positive"),
    }


def main():
    specs = []
    for split in ("random", "both_unseen"):
        for neg in ("n0", "n2", "n5"):
            for rep in ("aac_ctd", "esm2_mean"):
                specs.append((ROOT/f"results/negative_bias_revalidation/{split}__{neg}__{rep}/predictions.csv",
                              "R0" if split == "random" else "R3", neg.upper(), rep, "MLP", "y_probability"))
    for model in ("logistic_regression", "mlp"):
        specs.append((ROOT/f"results/architecture_ablation/n2__{model}/predictions.csv", "R3", "N2", "ESM-2 mean", model, "y_probability"))

    summaries, frames = [], {}
    for spec in specs:
        df, row = summarize(*spec)
        summaries.append(row)
        frames[(row["split"], row["negative_set"], row["representation"], row["model"])] = df
    table = pd.DataFrame(summaries)
    (ROOT/"tables").mkdir(exist_ok=True)
    table.to_csv(ROOT/"tables/tableS_score_inversion_diagnostic.csv", index=False)

    # Primary visual: the same ESM-2+MLP pipeline across R0/R3 and N0/N2.
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    for i, split in enumerate(("R0", "R3")):
        for j, neg in enumerate(("N0", "N2")):
            ax = axes[i, j]
            df = frames[(split, neg, "esm2_mean", "MLP")]
            for label, color, name in ((0, "#d95f02", "Constructed negative"), (1, "#1b9e77", "Positive")):
                ax.hist(df.loc[df.y_true == label, "y_probability"], bins=30, density=True,
                        alpha=.45, color=color, label=name)
            row = table[(table.split == split) & (table.negative_set == neg) &
                        (table.representation == "esm2_mean") & (table.model == "MLP")].iloc[0]
            ax.set_title(f"{split}/{neg}: AUC={row.auc_original:.3f}, inverted={row.auc_inverted:.3f}")
            ax.set_ylabel("Density")
            ax.set_xlabel("Predicted probability")
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Score distributions under original and degree-matched negatives")
    fig.tight_layout()
    (ROOT/"figures").mkdir(exist_ok=True)
    fig.savefig(ROOT/"figures/figureS_score_distributions_N0_N2.pdf")
    fig.savefig(ROOT/"figures/figureS_score_distributions_N0_N2.png", dpi=220)
    plt.close(fig)

    below = table[table.auc_original < .5]
    note = {
        "n_evaluations": len(table), "n_below_chance": len(below),
        "n_materially_reversed": int((table.ordering == "reversed_material").sum()),
        "n_near_chance": int((table.ordering == "near_chance").sum()),
        "interpretation": (
            "Because AUC(1-p) equals 1-AUC(p), inversion alone is algebraic rather than independent "
            "evidence. The R0/N2 and R0/N5 class-conditional means and effect sizes support a modest "
            "reversed ordering under distribution shift. The R3/N2 logistic-regression value (0.4998) "
            "is effectively chance and should not be described as meaningful reversal."
        ),
    }
    out_dir = ROOT/"results/score_inversion"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir/"summary.json").write_text(json.dumps(note, indent=2)+"\n")
    table.to_csv(out_dir/"diagnostics.csv", index=False)


if __name__ == "__main__":
    main()
