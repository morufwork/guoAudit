"""the dataset audit dataset audit: protein-level statistics and positive-network analysis."""
from collections import Counter

import networkx as nx
import pandas as pd


def sequence_length_stats(proteins: pd.DataFrame) -> pd.DataFrame:
    stats = proteins.copy()
    stats["length"] = stats["sequence"].str.len()
    return stats[["protein_id", "length"]]


def amino_acid_composition(proteins: pd.DataFrame) -> dict[str, float]:
    counts = Counter()
    total = 0
    for seq in proteins["sequence"]:
        counts.update(seq)
        total += len(seq)
    return {aa: n / total for aa, n in sorted(counts.items())}


def protein_pair_degrees(pairs: pd.DataFrame, proteins: pd.DataFrame) -> pd.DataFrame:
    """Per-protein positive/negative degree, i.e. how many pairs each protein appears in."""
    pos = pairs[pairs["label"] == 1]
    neg = pairs[pairs["label"] == 0]

    def degree_series(df: pd.DataFrame) -> pd.Series:
        return pd.concat([df["protein_a"], df["protein_b"]]).value_counts()

    pos_deg = degree_series(pos)
    neg_deg = degree_series(neg)

    out = pd.DataFrame({"protein_id": proteins["protein_id"]})
    out["positive_degree"] = out["protein_id"].map(pos_deg).fillna(0).astype(int)
    out["negative_degree"] = out["protein_id"].map(neg_deg).fillna(0).astype(int)
    out["total_degree"] = out["positive_degree"] + out["negative_degree"]
    return out


def positive_ppi_graph(pairs: pd.DataFrame) -> nx.Graph:
    g = nx.Graph()
    pos = pairs[pairs["label"] == 1]
    g.add_edges_from(pos[["protein_a", "protein_b"]].itertuples(index=False, name=None))
    return g


def network_summary(g: nx.Graph) -> dict:
    degrees = [d for _, d in g.degree()]
    components = list(nx.connected_components(g))
    largest_cc = max(components, key=len) if components else set()
    return {
        "n_nodes": g.number_of_nodes(),
        "n_edges": g.number_of_edges(),
        "density": nx.density(g) if g.number_of_nodes() > 1 else 0.0,
        "n_connected_components": len(components),
        "largest_connected_component_size": len(largest_cc),
        "mean_degree": sum(degrees) / len(degrees) if degrees else 0.0,
        "median_degree": sorted(degrees)[len(degrees) // 2] if degrees else 0.0,
        "max_degree": max(degrees) if degrees else 0,
        "average_clustering_coefficient": nx.average_clustering(g) if g.number_of_nodes() > 0 else 0.0,
        "hub_proteins_top10": sorted(g.degree, key=lambda x: -x[1])[:10],
    }
