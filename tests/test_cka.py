"""
Tests for the CKA module (src.analysis.cka).

Validates mathematical correctness, edge cases, input validation,
and numerical stability of all CKA variants.
"""

import pytest
import torch

from src.analysis.cross_lingual_embedding_alignment.cka import (
    MinibatchCKAAccumulator,
    _center_gram,
    cka_permutation_test,
    compute_layerwise_cka,
    linear_cka,
    minibatch_cka,
    rbf_cka,
    whitened_cka,
)

# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture
def random_activations() -> tuple[torch.Tensor, torch.Tensor]:
    """Generate random activation matrices for testing."""
    torch.manual_seed(42)
    X = torch.randn(100, 64)
    Y = torch.randn(100, 64)
    return X, Y


@pytest.fixture
def identical_activations() -> tuple[torch.Tensor, torch.Tensor]:
    """Generate identical activation matrices (CKA should be ~1.0)."""
    torch.manual_seed(42)
    X = torch.randn(100, 64)
    return X, X.clone()


@pytest.fixture
def linearly_transformed() -> tuple[torch.Tensor, torch.Tensor]:
    """X and Y = X @ A for a random orthogonal A (CKA should be ~1.0)."""
    torch.manual_seed(42)
    X = torch.randn(100, 64)
    # Create a random orthogonal matrix via QR decomposition.
    A, _ = torch.linalg.qr(torch.randn(64, 64))
    Y = X @ A
    return X, Y


# ===================================================================
# Input validation tests
# ===================================================================

class TestInputValidation:
    """Tests for input validation in CKA functions."""

    def test_rejects_1d_input(self) -> None:
        """Should reject 1D tensors."""
        with pytest.raises(ValueError, match="must be a 2D tensor"):
            linear_cka(torch.randn(10), torch.randn(10))

    def test_rejects_3d_input(self) -> None:
        """Should reject 3D tensors."""
        with pytest.raises(ValueError, match="must be a 2D tensor"):
            linear_cka(torch.randn(10, 5, 3), torch.randn(10, 5, 3))

    def test_rejects_sample_count_mismatch(self) -> None:
        """Should reject matrices with different sample counts."""
        with pytest.raises(ValueError, match="Sample count mismatch"):
            linear_cka(torch.randn(10, 5), torch.randn(20, 5))

    def test_rejects_zero_samples(self) -> None:
        """Should reject empty matrices."""
        with pytest.raises(ValueError, match="zero samples"):
            linear_cka(torch.randn(0, 5), torch.randn(0, 5))

    def test_rejects_nan_input(self) -> None:
        """Should reject matrices containing NaN."""
        X = torch.randn(10, 5)
        X[0, 0] = float("nan")
        with pytest.raises(ValueError, match="NaN or Inf"):
            linear_cka(X, torch.randn(10, 5))

    def test_rejects_inf_input(self) -> None:
        """Should reject matrices containing Inf."""
        Y = torch.randn(10, 5)
        Y[0, 0] = float("inf")
        with pytest.raises(ValueError, match="NaN or Inf"):
            linear_cka(torch.randn(10, 5), Y)


# ===================================================================
# Linear CKA tests
# ===================================================================

class TestLinearCKA:
    """Tests for the linear_cka function."""

    def test_self_similarity_is_one(self, identical_activations) -> None:
        """CKA(X, X) should be 1.0."""
        X, X_copy = identical_activations
        score = linear_cka(X, X_copy)
        assert abs(score.item() - 1.0) < 1e-5

    def test_orthogonal_invariance(self, linearly_transformed) -> None:
        """CKA should be invariant to orthogonal transformations."""
        X, Y = linearly_transformed
        score = linear_cka(X, Y)
        assert abs(score.item() - 1.0) < 1e-4

    def test_output_range(self, random_activations) -> None:
        """CKA should return a value in [0, 1]."""
        X, Y = random_activations
        score = linear_cka(X, Y)
        assert 0.0 <= score.item() <= 1.0 + 1e-6

    def test_symmetry(self, random_activations) -> None:
        """CKA(X, Y) should equal CKA(Y, X)."""
        X, Y = random_activations
        score_xy = linear_cka(X, Y)
        score_yx = linear_cka(Y, X)
        assert abs(score_xy.item() - score_yx.item()) < 1e-6

    def test_different_feature_dims(self) -> None:
        """CKA should work when X and Y have different feature dims."""
        torch.manual_seed(42)
        X = torch.randn(50, 32)
        Y = torch.randn(50, 64)
        score = linear_cka(X, Y)
        assert 0.0 <= score.item() <= 1.0 + 1e-6


# ===================================================================
# RBF CKA tests
# ===================================================================

class TestRBFCKA:
    """Tests for the rbf_cka function."""

    def test_self_similarity_is_one(self, identical_activations) -> None:
        """RBF CKA(X, X) should be 1.0."""
        X, X_copy = identical_activations
        score = rbf_cka(X, X_copy)
        assert abs(score.item() - 1.0) < 1e-4

    def test_output_range(self, random_activations) -> None:
        """RBF CKA should be in [0, 1]."""
        X, Y = random_activations
        score = rbf_cka(X, Y)
        assert 0.0 <= score.item() <= 1.0 + 1e-6

    def test_symmetry(self, random_activations) -> None:
        """RBF CKA should be symmetric."""
        X, Y = random_activations
        score_xy = rbf_cka(X, Y)
        score_yx = rbf_cka(Y, X)
        assert abs(score_xy.item() - score_yx.item()) < 1e-5


# ===================================================================
# Whitened CKA tests
# ===================================================================

class TestWhitenedCKA:
    """Tests for the whitened_cka function."""

    def test_self_similarity_is_one(self, identical_activations) -> None:
        """Whitened CKA(X, X) should be 1.0."""
        X, X_copy = identical_activations
        score = whitened_cka(X, X_copy)
        assert abs(score.item() - 1.0) < 1e-4

    def test_output_range(self, random_activations) -> None:
        """Whitened CKA should be in [0, 1]."""
        X, Y = random_activations
        score = whitened_cka(X, Y)
        assert 0.0 <= score.item() <= 1.0 + 1e-6


# ===================================================================
# Mini-batch CKA tests
# ===================================================================

class TestMinibatchCKA:
    """Tests for mini-batch CKA computation."""

    def test_matches_full_computation(self, random_activations) -> None:
        """Mini-batch CKA should match full linear CKA."""
        X, Y = random_activations
        full_score = linear_cka(X, Y).item()
        mb_score = minibatch_cka(X, Y, batch_size=25)
        assert abs(full_score - mb_score) < 1e-5

    def test_accumulator_matches_full(self, random_activations) -> None:
        """MinibatchCKAAccumulator should match linear_cka."""
        X, Y = random_activations
        full_score = linear_cka(X, Y).item()

        acc = MinibatchCKAAccumulator(d_x=64, d_y=64)
        acc.update(X[:50], Y[:50])
        acc.update(X[50:], Y[50:])
        mb_score = acc.compute()

        assert abs(full_score - mb_score) < 1e-5

    def test_accumulator_reset(self) -> None:
        """Reset should clear all state."""
        acc = MinibatchCKAAccumulator(d_x=32, d_y=32)
        acc.update(torch.randn(10, 32), torch.randn(10, 32))
        acc.reset()
        assert acc._n == 0

    def test_accumulator_rejects_wrong_dims(self) -> None:
        """Should reject batches with wrong feature dimensions."""
        acc = MinibatchCKAAccumulator(d_x=32, d_y=32)
        with pytest.raises(ValueError, match="feature dim"):
            acc.update(torch.randn(10, 64), torch.randn(10, 32))

    def test_accumulator_empty_compute_raises(self) -> None:
        """Should raise ValueError if compute() called before update()."""
        acc = MinibatchCKAAccumulator(d_x=32, d_y=32)
        with pytest.raises(ValueError, match="No samples accumulated"):
            acc.compute()


# ===================================================================
# Permutation test
# ===================================================================

class TestPermutationTest:
    """Tests for CKA permutation testing."""

    def test_identical_data_is_significant(
        self, identical_activations
    ) -> None:
        """Identical data should have p-value close to 0."""
        X, X_copy = identical_activations
        result = cka_permutation_test(
            X, X_copy, n_permutations=100, seed=42
        )
        assert result["p_value"] <= 0.05
        assert result["is_significant"]
        assert result["observed_cka"] > result["null_mean"]

    def test_result_keys(self, random_activations) -> None:
        """Result dict should have all expected keys."""
        X, Y = random_activations
        result = cka_permutation_test(X, Y, n_permutations=50)
        expected_keys = {
            "observed_cka", "p_value", "null_mean",
            "null_std", "is_significant",
        }
        assert set(result.keys()) == expected_keys

    def test_rejects_invalid_kernel(self, random_activations) -> None:
        """Should raise ValueError for unknown kernel."""
        X, Y = random_activations
        with pytest.raises(ValueError, match="kernel must be"):
            cka_permutation_test(X, Y, kernel="invalid")


# ===================================================================
# Layerwise CKA
# ===================================================================

class TestComputeLayerwiseCKA:
    """Tests for compute_layerwise_cka."""

    def test_correct_output_shape(self) -> None:
        """Output matrix should have shape (n_layers_a, n_layers_b)."""
        torch.manual_seed(42)
        a = {"layer_0": torch.randn(50, 32), "layer_1": torch.randn(50, 32)}
        b = {"layer_0": torch.randn(50, 32), "layer_1": torch.randn(50, 32)}
        result = compute_layerwise_cka(a, b)
        assert result.scores.shape == (2, 2)
        assert len(result.row_names) == 2
        assert len(result.col_names) == 2

    def test_diagonal_is_high_for_same_data(self) -> None:
        """Diagonal should be ~1.0 when comparing same activations."""
        torch.manual_seed(42)
        acts = {"layer_0": torch.randn(50, 32)}
        result = compute_layerwise_cka(acts, acts)
        assert abs(result.scores[0, 0] - 1.0) < 1e-5


# ===================================================================
# Helper tests
# ===================================================================

class TestCenterGram:
    """Tests for the Gram matrix centering function."""

    def test_centered_gram_has_zero_row_means(self) -> None:
        """Centered Gram matrix should have near-zero row means."""
        K = torch.randn(20, 20)
        K = K @ K.T  # Make symmetric PSD.
        K_centered = _center_gram(K)

        row_means = K_centered.mean(dim=1)
        assert torch.allclose(row_means, torch.zeros_like(row_means), atol=1e-5)
