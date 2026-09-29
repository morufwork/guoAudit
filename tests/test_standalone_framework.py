"""Tests for the single-file framework, ppi_benchmark_controls.py."""
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("pbc", REPO_ROOT / "ppi_benchmark_controls.py")
pbc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pbc)

GUO_PROTEINS = REPO_ROOT / "data/raw/yeast/dictionary/protein.dictionary.tsv"
GUO_PAIRS = REPO_ROOT / "data/raw/yeast/actions/protein.actions.tsv"


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    return pbc.make_synthetic_dataset(tmp_path_factory.mktemp("syn"))


def test_quick_demo_runs_end_to_end(synthetic, tmp_path):
    prot, pairs = synthetic
    res = pbc.run_framework(prot, pairs, tmp_path, models=("lr", "mlp"), n_boot=50, extra_negatives=False)
    assert (tmp_path / "REPORT.md").exists()
    assert (tmp_path / "figures" / "00_framework_summary.png").exists()
    assert res["splits"]["passed"].all()
    topo = res["topology"].query("feature == 'degree' and model == 'lr'").set_index("evaluated_on")["roc_auc"]
    # the synthetic negatives are degree-biased by construction; matching removes the cue
    assert topo["N0"] > 0.9 and abs(topo["N2"] - 0.5) < 0.1


def test_positive_only_input_and_fasta(synthetic, tmp_path):
    prot, pairs = synthetic
    p = pd.read_csv(prot, sep="\t")
    fasta = tmp_path / "p.fasta"
    fasta.write_text("".join(f">{i} desc\n{s[:60]}\n{s[60:]}\n" for i, s in zip(p.protein_id, p.sequence)))
    assert pbc.read_proteins(fasta).equals(pbc.read_proteins(prot))
    pos_only = tmp_path / "pos.tsv"
    pd.read_csv(pairs, sep="\t").query("label == 1")[["protein_a", "protein_b"]].to_csv(pos_only, sep="\t", index=False, header=False)
    res = pbc.run_framework(fasta, pos_only, tmp_path / "out", models=("lr",), n_boot=20, extra_negatives=False)
    assert res["audit"]["canonical"]["n_negative"] == 0


def test_pair_features_are_order_invariant(synthetic):
    prot, _ = synthetic
    feats = pbc.aac_ctd_features(pbc.read_proteins(prot).head(10))
    ab = pd.DataFrame({"protein_a": feats.protein_id[:5].values, "protein_b": feats.protein_id[5:].values, "label": 1})
    ba = ab.rename(columns={"protein_a": "protein_b", "protein_b": "protein_a"})
    assert (pbc.pair_matrix(ab, feats)[0] == pbc.pair_matrix(ba, feats)[0]).all()


@pytest.mark.skipif(not GUO_PAIRS.exists(), reason="Guo data not present")
def test_guo_splits_and_negatives_match_saved_files():
    proteins, pairs, audit = pbc.audit_and_canonicalize(pbc.read_proteins(GUO_PROTEINS), pbc.read_pairs(GUO_PAIRS))
    assert audit["canonical"]["n_pairs"] == 11164 and audit["issues"]["duplicate_pairs_removed"] == 24
    splits = pbc.make_splits(pairs, proteins, seed=42)
    names = {"R0": "random", "R1": "seen_seen", "R2": "one_unseen", "R3": "both_unseen"}
    for key, folder in names.items():
        for i, part in enumerate(("train", "test")):
            saved = pd.read_csv(REPO_ROOT / "data/splits" / folder / f"{part}.tsv", sep="\t")
            assert splits[key][i].equals(saved), (key, part)
    pos = pairs[pairs.label == 1].reset_index(drop=True)
    st = pbc.compute_protein_stats(proteins, pos)
    n2 = pbc.NEGATIVE_BUILDERS["N2"](pos, st, pbc.known_pair_set(pos), 42)
    assert n2.equals(pd.read_csv(REPO_ROOT / "data/processed/negative_sets/n2.tsv", sep="\t"))
