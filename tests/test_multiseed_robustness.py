"""Multi-seed robustness tests.

Verifies the two properties the robustness script's design depends on:
(1) the R3 data partition (train/test pairs, N2 negatives) is a pure
function of DATA_SEED and does not vary when only a downstream model's own
training seed changes -- the requirement to hold "the
underlying data partition" fixed while varying "initialization/training
randomness"; (2) that varying a model's own seed actually produces the kind
of behavior expected for each model family -- deterministic for
LogisticRegression's default (lbfgs) solver, seed-sensitive for
RandomForest/MLP/torch architectures. Without this second check, a
multi-seed study could silently report zero variance everywhere because
seeding wasn't wired through, and look "robust" for the wrong reason.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from src.data.negative_sampling import _known_pair_set, build_r3_n0_n2_datasets, compute_protein_stats
from src.models.siamese import SharedEncoderFusion
from src.models.torch_utils import predict_proba, train_pair_model


@pytest.fixture(scope="module")
def repo_root():
    from pathlib import Path
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def r3_inputs(repo_root):
    proteins = pd.read_csv(repo_root / "data" / "processed" / "proteins_with_groups.tsv", sep="\t")
    pairs = pd.read_csv(repo_root / "data" / "processed" / "ppi_pairs_clean.tsv", sep="\t", header=None,
                         names=["protein_a", "protein_b", "label"])
    pos = pairs[pairs["label"] == 1].reset_index(drop=True)
    protein_stats = compute_protein_stats(proteins, pos, n_deciles=10)
    known_pairs = _known_pair_set(pos)
    return proteins, protein_stats, known_pairs


def test_r3_data_partition_is_a_pure_function_of_data_seed(repo_root, r3_inputs):
    """Calling the dataset builder twice with the same DATA_SEED (as the
    robustness script does across every model-seed iteration) must return
    byte-identical train/test pairs -- confirming the loop varies only
    model training randomness, never which pairs are in the split."""
    proteins, protein_stats, known_pairs = r3_inputs
    d1 = build_r3_n0_n2_datasets(protein_stats, known_pairs, proteins, repo_root, seed=42, test_fraction=0.20)
    d2 = build_r3_n0_n2_datasets(protein_stats, known_pairs, proteins, repo_root, seed=42, test_fraction=0.20)
    for neg_name in ["n0", "n2"]:
        train1, test1 = d1[neg_name]
        train2, test2 = d2[neg_name]
        pd.testing.assert_frame_equal(train1.reset_index(drop=True), train2.reset_index(drop=True))
        pd.testing.assert_frame_equal(test1.reset_index(drop=True), test2.reset_index(drop=True))


def test_logistic_regression_lbfgs_is_deterministic_across_random_state():
    """Default solver (lbfgs) does not use random_state at all -- so a
    'logistic regression across 5 seeds' run is expected, not buggy, to
    show exactly zero variance. This regression-guards that assumption."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(300, 10))
    y = (x[:, 0] + x[:, 1] > 0).astype(int)

    preds = []
    for seed in [42, 43, 44]:
        model = LogisticRegression(max_iter=2000, random_state=seed)
        model.fit(x, y)
        preds.append(model.predict_proba(x)[:, 1])

    np.testing.assert_array_almost_equal(preds[0], preds[1], decimal=10)
    np.testing.assert_array_almost_equal(preds[0], preds[2], decimal=10)


def test_random_forest_varies_across_random_state():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(300, 10))
    y = (x[:, 0] + x[:, 1] > 0).astype(int)

    preds = []
    for seed in [42, 43, 44]:
        model = RandomForestClassifier(n_estimators=50, random_state=seed)
        model.fit(x, y)
        preds.append(model.predict_proba(x)[:, 1])

    assert not np.allclose(preds[0], preds[1])
    assert not np.allclose(preds[0], preds[2])


def test_mlp_varies_across_random_state():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(300, 10))
    y = (x[:, 0] + x[:, 1] > 0).astype(int)

    preds = []
    for seed in [42, 43, 44]:
        model = MLPClassifier(hidden_layer_sizes=(16,), max_iter=200, random_state=seed)
        model.fit(x, y)
        preds.append(model.predict_proba(x)[:, 1])

    assert not np.allclose(preds[0], preds[1])
    assert not np.allclose(preds[0], preds[2])


def test_torch_architecture_varies_across_seed():
    rng = np.random.default_rng(3)
    n, dim = 200, 16
    h_a = rng.normal(size=(n, dim)).astype(np.float32)
    h_b = rng.normal(size=(n, dim)).astype(np.float32)
    y = ((h_a.mean(axis=1) + h_b.mean(axis=1)) > 0).astype(np.float32)

    preds = []
    for seed in [42, 43]:
        model = train_pair_model(lambda: SharedEncoderFusion(dim, hidden_dim=32, latent_dim=16), h_a, h_b, y, seed=seed, max_epochs=20)
        preds.append(predict_proba(model, h_a, h_b))

    assert not np.allclose(preds[0], preds[1])
