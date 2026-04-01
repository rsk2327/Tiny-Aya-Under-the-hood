"""
Hierarchical clustering utilities for language family analysis.

This module implements Novel Technique 1 from the TIN-7 specification:
**Language Family Clustering Analysis**. The core idea is to apply
hierarchical clustering on the CKA similarity matrix at each layer
and track how language family groupings dissolve (or persist) as
representations flow through the network.

In early layers, we expect languages from the same family (e.g.,
Indo-European) or same script (e.g., Latin) to cluster together
because surface-level features (shared vocabulary, similar
tokenization) dominate. In deeper layers, if the model has learned
language-agnostic representations, these family-based clusters should
dissolve — all languages should converge regardless of genetic
relationship.

Key concepts:
    - **Cophenetic correlation**: Measures how faithfully the
      dendrogram preserves the original pairwise distances. A value
      of 1.0 means perfect preservation. We track this per layer
      to quantify clustering quality.
    - **Cluster dissolution**: We compare the discovered clusters
      against the known language family assignments to measure how
      much "family awareness" the model retains at each layer.
    - **Intra- vs inter-family CKA**: Average CKA within the same
      language family vs. across different families. The gap between
      these two quantities measures family bias; convergence occurs
      when the gap approaches zero.

References:
    - Ward's method: Ward (1963), "Hierarchical Grouping to Optimize
      an Objective Function"
    - Cophenetic correlation: Sokal & Rohlf (1962)
    - Adjusted Rand Index: Hubert & Arabie (1985)
"""


import logging

import numpy as np
from numpy.typing import NDArray
from scipy.cluster.hierarchy import (
    cophenet,
    fcluster,
    linkage,
)
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score

from src.utils.languages import Language

# Module-level logger.
logger = logging.getLogger(__name__)


# ===================================================================
# Input validation
# ===================================================================

def _validate_similarity_matrix(
    similarity_matrix: NDArray,
    language_names: list[str],
    func_name: str,
) -> None:
    """Validate a language-pair similarity matrix.

    Args:
        similarity_matrix: Square matrix of shape ``(n_langs, n_langs)``.
        language_names: List of language names matching the matrix axes.
        func_name: Calling function name for error messages.

    Raises:
        ValueError: If validation fails.
    """
    if similarity_matrix.ndim != 2:
        raise ValueError(
            f"{func_name}: similarity_matrix must be 2D, "
            f"got shape {similarity_matrix.shape}"
        )
    if similarity_matrix.shape[0] != similarity_matrix.shape[1]:
        raise ValueError(
            f"{func_name}: similarity_matrix must be square, "
            f"got shape {similarity_matrix.shape}"
        )
    if similarity_matrix.shape[0] != len(language_names):
        raise ValueError(
            f"{func_name}: Matrix size ({similarity_matrix.shape[0]}) "
            f"does not match number of languages ({len(language_names)})."
        )
    if len(language_names) < 2:
        raise ValueError(
            f"{func_name}: Need at least 2 languages for clustering, "
            f"got {len(language_names)}."
        )


# ===================================================================
# Core clustering functions
# ===================================================================

def compute_hierarchical_clustering(
    similarity_matrix: NDArray,
    language_names: list[str],
    method: str = "ward",
) -> dict:
    """Perform hierarchical clustering on a language similarity matrix.

    Converts the CKA similarity matrix to a distance matrix
    (distance = 1 - similarity), then applies agglomerative
    hierarchical clustering using the specified linkage method.

    Args:
        similarity_matrix: Symmetric CKA matrix of shape
            ``(n_langs, n_langs)`` with values in [0, 1].
        language_names: List of language names corresponding to
            matrix rows/columns.
        method: Linkage method for hierarchical clustering. Options:
            - "ward": Minimizes within-cluster variance (default,
              recommended for balanced dendrograms).
            - "complete": Maximum inter-cluster distance.
            - "average": Mean inter-cluster distance (UPGMA).
            - "single": Minimum inter-cluster distance.

    Returns:
        Dictionary containing:
            - ``"linkage_matrix"``: Scipy linkage matrix (n-1 x 4 array).
            - ``"cophenetic_correlation"``: How faithfully the dendrogram
              preserves pairwise distances (float in [0, 1]).
            - ``"distance_matrix"``: The 1-similarity distance matrix.
            - ``"language_names"``: The input language names.
            - ``"method"``: The linkage method used.

    Raises:
        ValueError: If the similarity matrix fails validation.

    Example::

        >>> result = compute_hierarchical_clustering(cka_matrix, lang_names)
        >>> print(f"Cophenetic corr: {result['cophenetic_correlation']:.3f}")
    """
    _validate_similarity_matrix(
        similarity_matrix, language_names,
        "compute_hierarchical_clustering"
    )

    # --- Convert similarity to distance ---
    # Clip to [0, 1] to handle floating-point noise outside this range.
    clipped = np.clip(similarity_matrix, 0.0, 1.0)
    distance_matrix = 1.0 - clipped

    # Ensure the diagonal is exactly zero (self-distance).
    np.fill_diagonal(distance_matrix, 0.0)

    # Ensure symmetry (average any tiny asymmetries from floating point).
    distance_matrix = (distance_matrix + distance_matrix.T) / 2.0

    # --- Convert to condensed distance vector for scipy ---
    # squareform expects a proper distance matrix (symmetric, zero diagonal).
    condensed_distances = squareform(distance_matrix, checks=False)

    # --- Perform hierarchical clustering ---
    linkage_matrix = linkage(condensed_distances, method=method)

    # --- Compute cophenetic correlation ---
    # This measures how well the dendrogram preserves pairwise distances.
    coph_corr, _ = cophenet(linkage_matrix, condensed_distances)

    return {
        "linkage_matrix": linkage_matrix,
        "cophenetic_correlation": float(coph_corr),
        "distance_matrix": distance_matrix,
        "language_names": language_names,
        "method": method,
    }


def compute_cluster_assignments(
    linkage_matrix: NDArray,
    n_clusters: int | None = None,
    distance_threshold: float | None = None,
) -> NDArray:
    """Extract flat cluster assignments from a linkage matrix.

    Either ``n_clusters`` or ``distance_threshold`` must be specified.
    If both are given, ``n_clusters`` takes precedence.

    Args:
        linkage_matrix: Scipy linkage matrix from hierarchical
            clustering.
        n_clusters: Desired number of clusters.
        distance_threshold: Maximum cophenetic distance within a
            cluster (used with criterion="distance").

    Returns:
        1D array of cluster labels, one per language.

    Raises:
        ValueError: If neither ``n_clusters`` nor ``distance_threshold``
            is specified.
    """
    if n_clusters is not None:
        return fcluster(linkage_matrix, t=n_clusters, criterion="maxclust")
    elif distance_threshold is not None:
        return fcluster(
            linkage_matrix, t=distance_threshold, criterion="distance"
        )
    else:
        raise ValueError(
            "compute_cluster_assignments: Specify either n_clusters "
            "or distance_threshold."
        )


# ===================================================================
# Language family dissolution analysis
# ===================================================================

def compute_family_dissolution_metrics(
    similarity_matrix: NDArray,
    language_names: list[str],
    languages: list[Language] | None = None,
) -> dict:
    """Compute metrics that quantify language family dissolution.

    Measures how much the model's representations reflect language
    family structure at a given layer. Key metrics:

    1. **Intra-family CKA**: Average CKA between languages in the
       same family (e.g., Hindi-Bengali, both Indo-European).
    2. **Inter-family CKA**: Average CKA between languages in
       different families (e.g., Hindi-Tamil).
    3. **Family gap**: Intra - Inter. A large gap means the model
       still differentiates families; a gap near zero means the
       model has achieved language-agnostic representations.
    4. **Adjusted Rand Index (ARI)**: Compares the clustering
       discovered by Ward's method against the ground-truth family
       labels. ARI = 1.0 means the discovered clusters perfectly
       match families; ARI = 0 means no agreement.

    Args:
        similarity_matrix: Symmetric CKA matrix of shape
            ``(n_langs, n_langs)``.
        language_names: List of language names matching matrix axes.
        languages: Optional list of ``Language`` enum members in the
            same order as ``language_names``. If ``None``, languages
            are looked up by name from the enum.

    Returns:
        Dictionary containing:
            - ``"intra_family_cka"``: Mean intra-family CKA (float).
            - ``"inter_family_cka"``: Mean inter-family CKA (float).
            - ``"family_gap"``: Intra minus inter (float).
            - ``"adjusted_rand_index"``: ARI of discovered clusters
              vs. true families (float in [-1, 1]).
            - ``"per_family_avg_cka"``: Dict mapping family name to
              average intra-family CKA.

    Raises:
        ValueError: If inputs fail validation or language lookup fails.
    """
    _validate_similarity_matrix(
        similarity_matrix, language_names,
        "compute_family_dissolution_metrics"
    )

    n_langs = len(language_names)

    # --- Resolve Language enum members ---
    if languages is None:
        from src.utils.languages import get_language_by_name

        languages = []
        for name in language_names:
            lang = get_language_by_name(name)
            if lang is None:
                raise ValueError(
                    f"compute_family_dissolution_metrics: Unknown "
                    f"language name '{name}'. Register it in "
                    f"uth.utils.languages.Language."
                )
            languages.append(lang)

    # --- Build family label mapping ---
    family_labels = [lang.family for lang in languages]

    # --- Compute intra- and inter-family CKA ---
    intra_scores: list[float] = []
    inter_scores: list[float] = []

    # Per-family accumulation.
    family_scores: dict[str, list[float]] = {}

    for i in range(n_langs):
        for j in range(i + 1, n_langs):  # Upper triangle only (symmetric).
            cka_score = similarity_matrix[i, j]

            if family_labels[i] == family_labels[j]:
                # Same family.
                intra_scores.append(cka_score)
                family = family_labels[i]
                family_scores.setdefault(family, []).append(cka_score)
            else:
                # Different families.
                inter_scores.append(cka_score)

    # Handle edge cases where a family has only one member
    # (no intra-family pairs possible).
    intra_cka = float(np.mean(intra_scores)) if intra_scores else 0.0
    inter_cka = float(np.mean(inter_scores)) if inter_scores else 0.0
    family_gap = intra_cka - inter_cka

    per_family_avg: dict[str, float] = {
        family: float(np.mean(scores)) if scores else 0.0
        for family, scores in family_scores.items()
    }

    # --- Compute Adjusted Rand Index ---
    # Cluster the languages and compare against true family labels.
    clustering_result = compute_hierarchical_clustering(
        similarity_matrix, language_names, method="ward"
    )

    # Use the number of true families as the number of clusters.
    n_true_families = len(set(family_labels))
    predicted_labels = compute_cluster_assignments(
        clustering_result["linkage_matrix"],
        n_clusters=n_true_families,
    )

    # Convert family names to integer labels for ARI.
    unique_families = sorted(set(family_labels))
    family_to_int = {f: i for i, f in enumerate(unique_families)}
    true_labels = np.array([family_to_int[f] for f in family_labels])

    ari = float(adjusted_rand_score(true_labels, predicted_labels))

    return {
        "intra_family_cka": intra_cka,
        "inter_family_cka": inter_cka,
        "family_gap": family_gap,
        "adjusted_rand_index": ari,
        "per_family_avg_cka": per_family_avg,
        "cophenetic_correlation": clustering_result["cophenetic_correlation"],
    }


def compute_script_group_metrics(
    similarity_matrix: NDArray,
    language_names: list[str],
    languages: list[Language] | None = None,
) -> dict:
    """Compute intra-script vs. inter-script CKA metrics.

    This implements Novel Technique 4 from TIN-7: Script-Based CKA
    Decomposition. Languages are grouped by writing system (Latin,
    Arabic, Devanagari, etc.) and CKA is averaged within and across
    script groups.

    This tests a critical hypothesis: is cross-lingual alignment
    driven by **token-surface similarity** (same script = similar
    BPE tokens) or **true semantic alignment** (meaning-based,
    script-independent)?

    Args:
        similarity_matrix: Symmetric CKA matrix of shape
            ``(n_langs, n_langs)``.
        language_names: List of language names matching matrix axes.
        languages: Optional list of ``Language`` enum members.

    Returns:
        Dictionary containing:
            - ``"intra_script_cka"``: Mean CKA within same script.
            - ``"inter_script_cka"``: Mean CKA across scripts.
            - ``"script_gap"``: Intra minus inter.
            - ``"per_script_avg_cka"``: Dict mapping script name
              to average intra-script CKA.
    """
    _validate_similarity_matrix(
        similarity_matrix, language_names,
        "compute_script_group_metrics"
    )

    n_langs = len(language_names)

    # --- Resolve languages ---
    if languages is None:
        from src.utils.languages import get_language_by_name
        languages = []
        for name in language_names:
            lang = get_language_by_name(name)
            if lang is None:
                raise ValueError(
                    f"compute_script_group_metrics: Unknown language "
                    f"name '{name}'."
                )
            languages.append(lang)

    script_labels = [lang.script for lang in languages]

    # --- Compute intra- and inter-script CKA ---
    intra_scores: list[float] = []
    inter_scores: list[float] = []
    script_scores: dict[str, list[float]] = {}

    for i in range(n_langs):
        for j in range(i + 1, n_langs):
            cka_score = similarity_matrix[i, j]

            if script_labels[i] == script_labels[j]:
                intra_scores.append(cka_score)
                script = script_labels[i]
                script_scores.setdefault(script, []).append(cka_score)
            else:
                inter_scores.append(cka_score)

    intra_cka = float(np.mean(intra_scores)) if intra_scores else 0.0
    inter_cka = float(np.mean(inter_scores)) if inter_scores else 0.0
    script_gap = intra_cka - inter_cka

    per_script_avg: dict[str, float] = {
        script: float(np.mean(scores)) if scores else 0.0
        for script, scores in script_scores.items()
    }

    return {
        "intra_script_cka": intra_cka,
        "inter_script_cka": inter_cka,
        "script_gap": script_gap,
        "per_script_avg_cka": per_script_avg,
    }
