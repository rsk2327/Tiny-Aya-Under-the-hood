"""
Retrieval metrics for functional alignment measurement.

Retrieval metrics measure **functional alignment** — whether the
geometric alignment between sentence pairs is useful for an actual
task. The task here is **parallel sentence retrieval**: given a
sentence embedding, can we find its correct counterpart by
nearest-neighbor search in the representation space?

Metrics implemented:
    - **MRR (Mean Reciprocal Rank)**: Average of 1/rank across all
      queries. MRR = 1.0 means perfect retrieval; MRR close to 0
      means correct pairs are buried deep in the ranked list.
    - **Recall@k**: Fraction of queries where the correct pair
      appears in the top-k results. Recall@1 is the strictest test;
      Recall@10 is more lenient.

All functions accept numpy arrays (not PyTorch tensors).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from numpy.typing import NDArray

# Module-level logger.
logger = logging.getLogger(__name__)


# ===================================================================
# Input validation
# ===================================================================

def _validate_embedding_matrices(
    source_embeddings: NDArray,
    target_embeddings: NDArray,
    func_name: str,
) -> None:
    """Validate that source and target embedding matrices are compatible.

    Checks performed:
        - Both are 2D arrays.
        - Both have the same number of sentences (rows).
        - Both have the same embedding dimension (columns).
        - Neither is empty.
        - Neither contains NaN or Inf values.

    Args:
        source_embeddings: Source language embeddings, shape
            ``(n_sentences, embedding_dim)``.
        target_embeddings: Target language embeddings, same shape.
        func_name: Calling function name for error messages.

    Raises:
        ValueError: If any validation check fails.
    """
    if source_embeddings.ndim != 2:
        raise ValueError(
            f"{func_name}: source_embeddings must be 2D, "
            f"got shape {source_embeddings.shape}"
        )
    if target_embeddings.ndim != 2:
        raise ValueError(
            f"{func_name}: target_embeddings must be 2D, "
            f"got shape {target_embeddings.shape}"
        )
    if source_embeddings.shape[0] != target_embeddings.shape[0]:
        raise ValueError(
            f"{func_name}: Sentence count mismatch — source has "
            f"{source_embeddings.shape[0]}, target has "
            f"{target_embeddings.shape[0]}. Both must represent "
            f"the same parallel sentences."
        )
    if source_embeddings.shape[1] != target_embeddings.shape[1]:
        raise ValueError(
            f"{func_name}: Embedding dimension mismatch — source has "
            f"dim {source_embeddings.shape[1]}, target has "
            f"dim {target_embeddings.shape[1]}."
        )
    if source_embeddings.shape[0] == 0:
        raise ValueError(f"{func_name}: Cannot compute with zero sentences.")
    if np.isnan(source_embeddings).any() or np.isinf(source_embeddings).any():
        raise ValueError(
            f"{func_name}: source_embeddings contains NaN or Inf."
        )
    if np.isnan(target_embeddings).any() or np.isinf(target_embeddings).any():
        raise ValueError(
            f"{func_name}: target_embeddings contains NaN or Inf."
        )


# ===================================================================
# Similarity computation
# ===================================================================

def compute_cosine_similarity_matrix(
    source_embeddings: NDArray,
    target_embeddings: NDArray,
) -> NDArray:
    """Compute pairwise cosine similarity between source and target.

    Cosine similarity is used rather than Euclidean distance because
    transformer representations often exhibit high anisotropy, making
    L2 distance less discriminative.

    Args:
        source_embeddings: Source language embeddings, shape
            ``(n_sentences, embedding_dim)``.
        target_embeddings: Target language embeddings, same shape.

    Returns:
        Cosine similarity matrix of shape ``(n_source, n_target)``
        with values in [-1, 1].
    """
    # Normalize each embedding to unit length.
    source_norms = np.linalg.norm(source_embeddings, axis=1, keepdims=True)
    target_norms = np.linalg.norm(target_embeddings, axis=1, keepdims=True)

    # Guard against zero-norm vectors (degenerate embeddings).
    source_norms = np.maximum(source_norms, 1e-10)
    target_norms = np.maximum(target_norms, 1e-10)

    source_normalized = source_embeddings / source_norms
    target_normalized = target_embeddings / target_norms

    # Cosine similarity = dot product of unit vectors.
    # Shape: (n_source, n_target)
    return source_normalized @ target_normalized.T


# ===================================================================
# Retrieval metrics
# ===================================================================

def compute_mrr(
    source_embeddings: NDArray,
    target_embeddings: NDArray,
) -> float:
    """Compute Mean Reciprocal Rank (MRR) for translation retrieval.

    For each source sentence, ranks all target sentences by cosine
    similarity and finds the rank of the correct translation (the
    sentence at the same index). MRR is the average of 1/rank
    across all queries.

    Interpretation:
        - MRR = 1.0: Every translation is the nearest neighbor
          (perfect alignment).
        - MRR = 0.5: On average, the correct translation is ranked
          2nd (good but not perfect).
        - MRR close to 0: Translations are effectively random in
          the embedding space (no alignment).

    Args:
        source_embeddings: Source language embeddings, shape
            ``(n_sentences, embedding_dim)``.
        target_embeddings: Target language embeddings, same shape.
            ``target_embeddings[i]`` is the translation of
            ``source_embeddings[i]``.

    Returns:
        MRR score as a float in (0, 1].

    Raises:
        ValueError: If inputs fail validation.

    Example::

        >>> src = np.random.randn(100, 768)
        >>> tgt = src + np.random.randn(100, 768) * 0.1  # near-perfect
        >>> mrr = compute_mrr(src, tgt)
        >>> mrr > 0.5  # Should be high for similar embeddings
        True
    """
    _validate_embedding_matrices(
        source_embeddings, target_embeddings, "compute_mrr"
    )

    # Compute full similarity matrix.
    similarity_matrix = compute_cosine_similarity_matrix(
        source_embeddings, target_embeddings
    )

    n_sentences = source_embeddings.shape[0]
    reciprocal_ranks = np.zeros(n_sentences)

    for i in range(n_sentences):
        # Sort target indices by descending similarity to source[i].
        ranked_indices = np.argsort(-similarity_matrix[i])

        # Find the rank of the correct translation (index i).
        # np.where returns a tuple; [0][0] gets the first occurrence.
        rank = np.where(ranked_indices == i)[0][0] + 1  # 1-indexed

        reciprocal_ranks[i] = 1.0 / rank

    return float(reciprocal_ranks.mean())


def compute_recall_at_k(
    source_embeddings: NDArray,
    target_embeddings: NDArray,
    k: int = 1,
) -> float:
    """Compute Recall@k for translation retrieval.

    For each source sentence, checks whether its correct translation
    (same-index target sentence) appears in the top-k most similar
    target sentences. Recall@k is the fraction of queries where this
    condition is met.

    Common k values:
        - k=1: Strictest test — is the correct translation the
          single nearest neighbor?
        - k=5: Moderate — is it in the top 5?
        - k=10: Lenient — is it in the top 10?

    Args:
        source_embeddings: Source language embeddings, shape
            ``(n_sentences, embedding_dim)``.
        target_embeddings: Target language embeddings, same shape.
        k: Number of top results to consider. Must be >= 1 and
            <= n_sentences.

    Returns:
        Recall@k score as a float in [0, 1].

    Raises:
        ValueError: If inputs fail validation or k is out of range.
    """
    _validate_embedding_matrices(
        source_embeddings, target_embeddings, "compute_recall_at_k"
    )

    n_sentences = source_embeddings.shape[0]

    if k < 1 or k > n_sentences:
        raise ValueError(
            f"compute_recall_at_k: k must be in [1, {n_sentences}], "
            f"got k={k}."
        )

    # Compute full similarity matrix.
    similarity_matrix = compute_cosine_similarity_matrix(
        source_embeddings, target_embeddings
    )

    hits = 0

    for i in range(n_sentences):
        # Get the indices of the top-k most similar targets.
        top_k_indices = np.argsort(-similarity_matrix[i])[:k]

        # Check if the correct translation is among them.
        if i in top_k_indices:
            hits += 1

    return hits / n_sentences


def compute_all_retrieval_metrics(
    source_embeddings: NDArray,
    target_embeddings: NDArray,
    k_values: Optional[list[int]] = None,
) -> dict[str, float]:
    """Compute a comprehensive set of retrieval metrics.

    Convenience function that computes MRR and Recall@k for multiple
    k values in a single pass over the similarity matrix, avoiding
    redundant computation.

    Args:
        source_embeddings: Source language embeddings, shape
            ``(n_sentences, embedding_dim)``.
        target_embeddings: Target language embeddings, same shape.
        k_values: List of k values for Recall@k computation.
            Defaults to ``[1, 5, 10]``.

    Returns:
        Dictionary with keys:
            - ``"mrr"``: Mean Reciprocal Rank.
            - ``"recall@1"``, ``"recall@5"``, ``"recall@10"``, etc.:
              Recall at each specified k.
            - ``"median_rank"``: Median rank of correct translations.
            - ``"mean_rank"``: Mean rank of correct translations.

    Raises:
        ValueError: If inputs fail validation.

    Example::

        >>> metrics = compute_all_retrieval_metrics(src_emb, tgt_emb)
        >>> print(f"MRR: {metrics['mrr']:.3f}, R@1: {metrics['recall@1']:.3f}")
    """
    _validate_embedding_matrices(
        source_embeddings, target_embeddings,
        "compute_all_retrieval_metrics"
    )

    if k_values is None:
        k_values = [1, 5, 10]

    # Compute the similarity matrix once.
    similarity_matrix = compute_cosine_similarity_matrix(
        source_embeddings, target_embeddings
    )

    n_sentences = source_embeddings.shape[0]

    # Compute ranks for all queries.
    ranks = np.zeros(n_sentences, dtype=np.int64)

    for i in range(n_sentences):
        ranked_indices = np.argsort(-similarity_matrix[i])
        ranks[i] = np.where(ranked_indices == i)[0][0] + 1  # 1-indexed

    # --- MRR ---
    reciprocal_ranks = 1.0 / ranks
    mrr = float(reciprocal_ranks.mean())

    # --- Recall@k for each k ---
    results: dict[str, float] = {"mrr": mrr}

    for k in k_values:
        if k > n_sentences:
            logger.warning(
                "k=%d exceeds n_sentences=%d; capping to n_sentences.",
                k, n_sentences,
            )
            k = n_sentences
        recall = float(np.mean(ranks <= k))
        results[f"recall@{k}"] = recall

    # --- Rank statistics ---
    results["median_rank"] = float(np.median(ranks))
    results["mean_rank"] = float(np.mean(ranks))

    return results


