"""
Tests for retrieval metrics (src.analysis.retrieval_metrics).

Validates correctness of MRR, Recall@k, and similarity computation
using known inputs with deterministic expected outputs.
"""

import numpy as np
import pytest

from src.analysis.cross_lingual_embedding_alignment.retrieval_metrics import (
    compute_all_retrieval_metrics,
    compute_cosine_similarity_matrix,
    compute_mrr,
    compute_recall_at_k,
)


class TestCosineSimilarity:
    """Tests for the cosine similarity matrix computation."""

    def test_identity_similarity(self) -> None:
        """Self-similarity should be 1.0 on the diagonal."""
        X = np.array([[1.0, 0.0], [0.0, 1.0]])
        sim = compute_cosine_similarity_matrix(X, X)
        np.testing.assert_allclose(np.diag(sim), 1.0, atol=1e-6)

    def test_orthogonal_similarity_is_zero(self) -> None:
        """Orthogonal vectors should have similarity 0."""
        X = np.array([[1.0, 0.0]])
        Y = np.array([[0.0, 1.0]])
        sim = compute_cosine_similarity_matrix(X, Y)
        assert abs(sim[0, 0]) < 1e-6


class TestMRR:
    """Tests for Mean Reciprocal Rank."""

    def test_perfect_alignment(self) -> None:
        """Identical embeddings should give MRR = 1.0."""
        emb = np.eye(5)  # Each sentence is a unique unit vector.
        mrr = compute_mrr(emb, emb)
        assert abs(mrr - 1.0) < 1e-6

    def test_rejects_mismatched_shapes(self) -> None:
        """Should reject matrices with different row counts."""
        with pytest.raises(ValueError, match="Sentence count mismatch"):
            compute_mrr(np.zeros((5, 3)), np.zeros((10, 3)))

    def test_rejects_empty_input(self) -> None:
        """Should reject zero-sentence input."""
        with pytest.raises(ValueError, match="zero sentences"):
            compute_mrr(np.zeros((0, 3)), np.zeros((0, 3)))


class TestRecallAtK:
    """Tests for Recall@k."""

    def test_perfect_alignment_recall_at_1(self) -> None:
        """Identical embeddings should give Recall@1 = 1.0."""
        emb = np.eye(10)
        recall = compute_recall_at_k(emb, emb, k=1)
        assert abs(recall - 1.0) < 1e-6

    def test_recall_at_k_increases_with_k(self) -> None:
        """Recall@k should be monotonically non-decreasing in k."""
        np.random.seed(42)
        src = np.random.randn(50, 16)
        tgt = src + np.random.randn(50, 16) * 0.5
        r1 = compute_recall_at_k(src, tgt, k=1)
        r5 = compute_recall_at_k(src, tgt, k=5)
        r10 = compute_recall_at_k(src, tgt, k=10)
        assert r1 <= r5 <= r10

    def test_rejects_invalid_k(self) -> None:
        """Should reject k < 1."""
        with pytest.raises(ValueError, match="k must be"):
            compute_recall_at_k(np.eye(5), np.eye(5), k=0)


class TestAllRetrievalMetrics:
    """Tests for the comprehensive retrieval metrics function."""

    def test_returns_all_expected_keys(self) -> None:
        """Should return MRR, Recall@k, and rank stats."""
        emb = np.eye(10)
        metrics = compute_all_retrieval_metrics(emb, emb)
        assert "mrr" in metrics
        assert "recall@1" in metrics
        assert "recall@5" in metrics
        assert "recall@10" in metrics
        assert "median_rank" in metrics
        assert "mean_rank" in metrics

    def test_perfect_alignment_metrics(self) -> None:
        """Perfect alignment should give ideal scores."""
        emb = np.eye(20)
        metrics = compute_all_retrieval_metrics(emb, emb)
        assert abs(metrics["mrr"] - 1.0) < 1e-6
        assert abs(metrics["recall@1"] - 1.0) < 1e-6
        assert abs(metrics["median_rank"] - 1.0) < 1e-6
