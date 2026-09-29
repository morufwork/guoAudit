"""Manuscript conceptual figures.

These are schematic diagrams of methodology actually executed -- not data
figures -- so they carry no numbers that need to come from a saved
result file. Figure 1's stages mirror Methods Sections 2.1-2.8 and the
actual narrative arc (Discussion 4.1): the negative-sampling audit was
inserted ahead of the architecture ablation and turned out to dominate,
which the diagram marks explicitly rather than presenting the pipeline as
if it had unfolded in the originally-planned, representation-comparison-first
order. Figure 3 mirrors Methods Section 2.3's five split regimes.

Figures 8-10 (structural ablation, external validation, explainability)
are NOT generated here: they depict analyses outside the scope of this
study --
drawing them would mean illustrating analysis that was never run.

Writes:
  figures/figure1_study_workflow.pdf
  figures/figure3_split_design.pdf
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from src.utils.logging import get_logger

REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = REPO_ROOT / "figures"
logger = get_logger("generate_conceptual_figures")

BOX_STYLE = dict(boxstyle="round,pad=0.4", linewidth=1.2)
NORMAL_FACE = "#eef2f7"
NORMAL_EDGE = "#334155"
PIVOT_FACE = "#fde68a"
PIVOT_EDGE = "#92400e"


def _stage_box(ax, xy, text, pivot=False):
    face, edge = (PIVOT_FACE, PIVOT_EDGE) if pivot else (NORMAL_FACE, NORMAL_EDGE)
    ax.annotate(
        text,
        xy=xy,
        xycoords="data",
        ha="center",
        va="center",
        fontsize=9.5,
        bbox={**BOX_STYLE, "facecolor": face, "edgecolor": edge},
    )


def _arrow(ax, xy_from, xy_to):
    ax.annotate(
        "",
        xy=xy_to,
        xytext=xy_from,
        arrowprops=dict(arrowstyle="-|>", color="#334155", linewidth=1.2),
    )


def build_figure1_workflow():
    stages = [
        ("Guo yeast PPI benchmark\n(2,497 proteins, 11,188 pairs)", False),
        ("Dataset integrity audit +\ncanonical/symmetric pairs\n(§2.1-2.2)", False),
        ("Leakage-resistant splits\nR0-R3 + homology-controlled\n(§2.3)", False),
        ("Sequence representations\nAAC/CTD, ESM-2, ProtT5\n(§2.4)", False),
        ("Classifiers + architecture ablation\nLR/RF/XGBoost/MLP, Siamese,\nattention, bilinear (§2.5)", False),
        ("Negative-sampling audit\nN0 vs. degree/length-matched\nnegatives (§2.6)", True),
        ("Calibration +\nstatistical evaluation\n(§2.7-2.8)", False),
        ("Leakage-resistant,\nnegative-sampling-aware\nevaluation framework (released)", False),
    ]

    fig, ax = plt.subplots(figsize=(4.6, 11))
    n = len(stages)
    ys = list(range(n, 0, -1))
    for y, (text, pivot) in zip(ys, stages):
        _stage_box(ax, (0.5, y), text, pivot=pivot)
    for y_from, y_to in zip(ys[:-1], ys[1:]):
        _arrow(ax, (0.5, y_from - 0.28), (0.5, y_to + 0.28))

    ax.annotate(
        "Originally planned as the final\nrepresentation comparison;\nturned out to dominate all\nupstream effects (§3.5-3.6)",
        xy=(1.55, ys[5]),
        xycoords="data",
        ha="left",
        va="center",
        fontsize=7.8,
        color="#92400e",
        style="italic",
    )
    _arrow(ax, (1.05, ys[5]), (1.35, ys[5]))

    ax.set_xlim(0, 3.4)
    ax.set_ylim(0.3, n + 0.7)
    ax.axis("off")
    ax.set_title("Figure 1. Study workflow", fontsize=11, pad=10)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "figure1_study_workflow.pdf")
    plt.close(fig)
    logger.info(f"Wrote {FIGURES_DIR / 'figure1_study_workflow.pdf'}")


def _protein_dots(ax, cx, cy, r, n, color, seed):
    import numpy as np

    rng = np.random.default_rng(seed)
    rr = r * (rng.uniform(0, 1, n) ** 0.5)
    xs = cx + rr * (rng.uniform(-1, 1, n))
    ys = cy + rr * (rng.uniform(-1, 1, n))
    ax.scatter(xs, ys, s=18, color=color, alpha=0.85, edgecolors="white", linewidths=0.3, zorder=3)


def build_figure3_split_design():
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.4))
    panels = [
        ("R0: Random", "mixed", "Independent pair-level split;\nproteins recur unstructured"),
        ("R1: Seen-seen\n(≈ C1)", "all_shared", "Both test proteins\nin training (new pair)"),
        ("R2: One-unseen\n(≈ C2)", "one", "One test protein\nabsent from training"),
        ("R3: Both-unseen\n(≈ C3)", "none", "Neither test protein\nin training"),
        ("Homology-controlled\n(R3 + 50-20% id.)", "none_h", "R3, plus no close\nhomolog across split"),
    ]

    train_color = "#2563eb"
    test_color = "#ea580c"
    shared_color = "#16a34a"

    for ax, (title, mode, caption) in zip(axes, panels):
        train_circle = plt.Circle((0.38, 0.55), 0.34, color=train_color, alpha=0.12, zorder=1)
        test_circle = plt.Circle((0.62, 0.55) if mode != "none_h" else (0.66, 0.55), 0.34, color=test_color, alpha=0.12, zorder=1)
        ax.add_patch(train_circle)
        ax.add_patch(test_circle)

        _protein_dots(ax, 0.30, 0.55, 0.22, 14, train_color, seed=1)
        if mode == "mixed":
            _protein_dots(ax, 0.5, 0.55, 0.20, 8, shared_color, seed=2)
            _protein_dots(ax, 0.72, 0.55, 0.16, 6, test_color, seed=3)
        elif mode == "all_shared":
            _protein_dots(ax, 0.55, 0.55, 0.22, 14, shared_color, seed=2)
        elif mode == "one":
            _protein_dots(ax, 0.5, 0.55, 0.14, 5, shared_color, seed=4)
            _protein_dots(ax, 0.72, 0.55, 0.18, 9, test_color, seed=5)
        else:
            _protein_dots(ax, 0.70 if mode != "none_h" else 0.74, 0.55, 0.20, 14, test_color, seed=6)

        if mode == "none_h":
            ax.annotate(
                "",
                xy=(0.62, 0.55),
                xytext=(0.40, 0.55),
                arrowprops=dict(arrowstyle="-", color="#64748b", linewidth=1.4, linestyle=(0, (3, 2))),
            )

        ax.text(0.30, 0.94, "Train", ha="center", fontsize=8.5, color=train_color, fontweight="bold")
        ax.text(0.70, 0.94, "Test", ha="center", fontsize=8.5, color=test_color, fontweight="bold")
        ax.text(0.5, 0.06, caption, ha="center", va="center", fontsize=7.5, color="#334155")
        ax.set_title(title, fontsize=9.2)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    legend_handles = [
        mpatches.Patch(color=train_color, alpha=0.5, label="Training-only protein"),
        mpatches.Patch(color=shared_color, alpha=0.85, label="Protein seen in training,\nrecombined in test pair"),
        mpatches.Patch(color=test_color, alpha=0.5, label="Test-only (unseen) protein"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Figure 3. Split design (§2.3)", fontsize=11)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(FIGURES_DIR / "figure3_split_design.pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Wrote {FIGURES_DIR / 'figure3_split_design.pdf'}")


def main():
    FIGURES_DIR.mkdir(exist_ok=True)
    build_figure1_workflow()
    build_figure3_split_design()
    logger.info("Conceptual figure generation complete.")


if __name__ == "__main__":
    main()
