"""Automated leakage checks. Terminate execution if violated — raise, do not warn-and-continue.
"""
import pandas as pd

from src.data.validate_data import canonical_pair


class LeakageError(Exception):
    pass


def _canon_set(df: pd.DataFrame) -> set[tuple[str, str]]:
    return set(df.apply(canonical_pair, axis=1))


def check_pair_duplicate_leakage(train: pd.DataFrame, test: pd.DataFrame) -> bool:
    """No exact (protein_a, protein_b) row may appear in both train and test."""
    train_exact = set(zip(train["protein_a"], train["protein_b"]))
    test_exact = set(zip(test["protein_a"], test["protein_b"]))
    return len(train_exact & test_exact) == 0


def check_reverse_pair_leakage(train: pd.DataFrame, test: pd.DataFrame) -> bool:
    """No pair may appear in train and test under either A-B or B-A ordering."""
    return len(_canon_set(train) & _canon_set(test)) == 0


def check_protein_disjointness(train: pd.DataFrame, test: pd.DataFrame, required: bool) -> bool:
    """If required=True (R3 only): no protein used in a train pair may appear
    in any test pair. If required=False, this check is a no-op pass (R0/R1/R2
    permit protein overlap by design)."""
    if not required:
        return True
    train_proteins = set(train["protein_a"]) | set(train["protein_b"])
    test_proteins = set(test["protein_a"]) | set(test["protein_b"])
    return train_proteins.isdisjoint(test_proteins)


def check_group_disjointness(train: pd.DataFrame, test: pd.DataFrame, id_to_group: dict, required: bool) -> bool:
    """Stricter than protein-ID disjointness: no sequence_group_id (the canonicalization
    equivalence class) may appear on both sides. Only enforced when required=True."""
    if not required:
        return True
    train_proteins = set(train["protein_a"]) | set(train["protein_b"])
    test_proteins = set(test["protein_a"]) | set(test["protein_b"])
    train_groups = {id_to_group[p] for p in train_proteins}
    test_groups = {id_to_group[p] for p in test_proteins}
    return train_groups.isdisjoint(test_groups)


def check_label_integrity(train: pd.DataFrame, test: pd.DataFrame) -> bool:
    labels = set(train["label"].unique()) | set(test["label"].unique())
    no_missing = not train["label"].isna().any() and not test["label"].isna().any()
    return labels.issubset({0, 1}) and no_missing


def run_leakage_checks(
    split_name: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    id_to_group: dict,
    require_protein_disjoint: bool,
    logger=None,
) -> dict:
    """Runs all four checks, logs PASS/FAIL, and raises LeakageError on any failure."""
    results = {
        "pair_duplicate_leakage": check_pair_duplicate_leakage(train, test),
        "reverse_pair_leakage": check_reverse_pair_leakage(train, test),
        "protein_disjointness": check_protein_disjointness(train, test, require_protein_disjoint),
        "group_disjointness": check_group_disjointness(train, test, id_to_group, require_protein_disjoint),
        "label_integrity": check_label_integrity(train, test),
    }

    labels_map = {
        "pair_duplicate_leakage": "PAIR DUPLICATE LEAKAGE",
        "reverse_pair_leakage": "REVERSE-PAIR LEAKAGE",
        "protein_disjointness": "PROTEIN DISJOINTNESS",
        "group_disjointness": "GROUP DISJOINTNESS",
        "label_integrity": "LABEL INTEGRITY",
    }
    failed = []
    for key, passed in results.items():
        line = f"[{split_name}] {labels_map[key]}: {'PASS' if passed else 'FAIL'}"
        if logger:
            logger.info(line)
        else:
            print(line)
        if not passed:
            failed.append(key)

    if failed:
        raise LeakageError(f"Split '{split_name}' failed leakage checks: {failed}")

    return results
