"""
Centered Kernel Alignment (CKA) for cross-lingual similarity measurement.

CKA measures representational similarity between neural network layers,
invariant to orthogonal transformations and isotropic scaling. This
module provides four CKA variants tailored for cross-lingual analysis:

    1. **Linear CKA** (``linear_cka``): Fast O(n * d^2) computation
       via cross-covariance matrices. Best for comparing same-model
       layers where relationships are approximately linear.

    2. **RBF CKA** (``rbf_cka``): Gaussian kernel variant that
       captures nonlinear representational relationships. Slower
       (O(n^2)) but more expressive.

    3. **Mini-batch CKA** (``minibatch_cka``): Memory-efficient
       streaming computation via ``MinibatchCKAAccumulator``. Processes
       data in chunks without materializing full activation matrices.

    4. **Whitened CKA** (``whitened_cka``): Anisotropy-corrected
       variant that applies ZCA whitening before CKA computation.
       Standard CKA can be confounded by representation anisotropy
       (where all vectors cluster in a narrow cone); whitening
       removes this confound. Novel technique from the TIN-7 spec.

Additionally, ``cka_permutation_test`` provides statistical hypothesis
testing to verify that observed CKA scores are significantly above
chance (null hypothesis: no representational similarity).

All functions accept PyTorch tensors and return Python floats or
tensors. Input validation is strict — shape mismatches, empty inputs,
and degenerate cases all raise informative errors.

Mathematical background:
    CKA(K, L) = HSIC(K, L) / sqrt(HSIC(K, K) * HSIC(L, L))

    where K = X @ X.T and L = Y @ Y.T are kernel matrices, and
    HSIC is the Hilbert-Schmidt Independence Criterion (biased
    estimator using centered Gram matrices).

    Linear CKA simplifies to:
        ||Y.T @ X||_F^2 / (||X.T @ X||_F * ||Y.T @ Y||_F)

References:
    - Kornblith et al., "Similarity of Neural Network Representations
      Revisited" (ICML 2019)
    - Nguyen et al., "Do Wide Neural Networks Really Need to be Wide?"
      (AAAI 2021) — mini-batch CKA
    - Project Aya reference: github.com/Wayy-Research/project-aya

Wayy Research / Tiny Aya Under The Hood, 2026.
"""


import logging
import warnings
from dataclasses import dataclass, field

import numpy as np
import torch
from numpy.typing import NDArray

# Module-level logger.
logger = logging.getLogger(__name__)


# ===================================================================
# Internal helper functions
# ===================================================================

def _validate_activation_pair(
    X: torch.Tensor,
    Y: torch.Tensor,
    func_name: str,
) -> None:
    """Validate that two activation matrices are compatible for CKA.

    Checks performed:
        - Both inputs are 2D tensors (samples x features).
        - Both have the same number of samples (first dimension).
        - Neither has zero samples or zero features.
        - Neither contains NaN or Inf values.

    Args:
        X: First activation matrix, shape ``(n_samples, d_x)``.
        Y: Second activation matrix, shape ``(n_samples, d_y)``.
        func_name: Name of the calling function, for error messages.

    Raises:
        ValueError: If any validation check fails.
    """
    if X.dim() != 2:
        raise ValueError(
            f"{func_name}: X must be a 2D tensor (n_samples, d_x), "
            f"got shape {tuple(X.shape)}"
        )
    if Y.dim() != 2:
        raise ValueError(
            f"{func_name}: Y must be a 2D tensor (n_samples, d_y), "
            f"got shape {tuple(Y.shape)}"
        )
    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            f"{func_name}: Sample count mismatch — X has "
            f"{X.shape[0]} samples, Y has {Y.shape[0]} samples. "
            f"Both must represent the same set of inputs."
        )
    if X.shape[0] == 0:
        raise ValueError(
            f"{func_name}: Cannot compute CKA with zero samples."
        )
    if X.shape[1] == 0 or Y.shape[1] == 0:
        raise ValueError(
            f"{func_name}: Cannot compute CKA with zero features — "
            f"X has {X.shape[1]} features, Y has {Y.shape[1]} features."
        )
    if torch.isnan(X).any() or torch.isinf(X).any():
        raise ValueError(
            f"{func_name}: X contains NaN or Inf values. "
            f"Check activation extraction for numerical issues."
        )
    if torch.isnan(Y).any() or torch.isinf(Y).any():
        raise ValueError(
            f"{func_name}: Y contains NaN or Inf values. "
            f"Check activation extraction for numerical issues."
        )


def _center_gram(K: torch.Tensor) -> torch.Tensor:
    """Center a Gram matrix using the centering matrix H = I - 1/n.

    The centered Gram matrix is: K_c = H @ K @ H, which is equivalent
    to subtracting row means, column means, and adding back the grand
    mean. This O(n^2) implementation avoids the O(n^3) cost of
    explicit matrix multiplications with H.

    Args:
        K: Square Gram matrix of shape ``(n, n)``.

    Returns:
        Centered Gram matrix of the same shape.
    """
    # Column means: shape (1, n) — mean over rows for each column.
    col_mean = K.mean(dim=0, keepdim=True)

    # Row means: shape (n, 1) — mean over columns for each row.
    row_mean = K.mean(dim=1, keepdim=True)

    # Grand mean: scalar — mean of all elements.
    grand_mean = K.mean()

    # Centering formula: K_c[i,j] = K[i,j] - mean_col[j] - mean_row[i] + grand_mean
    return K - col_mean - row_mean + grand_mean


def _hsic(K: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    """Compute the biased Hilbert-Schmidt Independence Criterion.

    HSIC measures the statistical dependence between two kernel
    matrices. The biased estimator is:

        HSIC(K, L) = (1/n^2) * tr(K_c @ L_c)

    where K_c and L_c are centered Gram matrices.

    Args:
        K: First centered or uncentered Gram matrix, shape ``(n, n)``.
        L: Second centered or uncentered Gram matrix, shape ``(n, n)``.

    Returns:
        Scalar HSIC value.
    """
    K_centered = _center_gram(K)
    L_centered = _center_gram(L)

    n = K.shape[0]

    # tr(A @ B) = sum(A * B.T) for symmetric matrices, but we use
    # the general trace formula for clarity and correctness.
    return torch.trace(K_centered @ L_centered) / (n * n)


def _rbf_kernel(
    X: torch.Tensor,
    sigma: float | None = None,
) -> torch.Tensor:
    """Compute the Radial Basis Function (Gaussian) kernel matrix.

    K[i,j] = exp(-||x_i - x_j||^2 / (2 * sigma^2))

    If ``sigma`` is not provided, the median heuristic is used:
    sigma = sqrt(median(nonzero squared distances) / 2).

    Args:
        X: Input matrix of shape ``(n_samples, d_features)``.
        sigma: RBF bandwidth parameter. If ``None``, computed
            automatically via the median heuristic.

    Returns:
        Kernel matrix of shape ``(n_samples, n_samples)`` with
        values in (0, 1].
    """
    # Compute pairwise squared Euclidean distances.
    # cdist returns L2 distances; we square them for the RBF formula.
    squared_distances = torch.cdist(X, X, p=2.0) ** 2

    if sigma is None:
        # Median heuristic: use the median of all nonzero distances
        # to set the bandwidth. This is a standard, data-adaptive
        # approach that avoids manual tuning.
        nonzero_dists = squared_distances[squared_distances > 0]

        if nonzero_dists.numel() == 0:
            # Degenerate case: all points are identical.
            warnings.warn(
                "All pairwise distances are zero — using sigma=1.0. "
                "This typically means all representations are identical."
            )
            sigma = 1.0
        else:
            median_dist = torch.median(nonzero_dists)
            sigma = torch.sqrt(median_dist / 2.0).item()

            # Guard against near-zero sigma (numerical stability).
            if sigma < 1e-10:
                sigma = 1.0

    return torch.exp(-squared_distances / (2.0 * sigma * sigma))


# ===================================================================
# Public API: CKA variants
# ===================================================================

def linear_cka(
    X: torch.Tensor,
    Y: torch.Tensor,
    eps: float = 1e-10,
) -> torch.Tensor:
    """Compute Linear CKA between two activation matrices.

    Linear CKA uses linear kernels (K = X @ X.T, L = Y @ Y.T) and
    simplifies the CKA formula to a ratio of Frobenius norms of
    cross-covariance matrices:

        CKA(X, Y) = ||Y.T @ X||_F^2 / (||X.T @ X||_F * ||Y.T @ Y||_F)

    This is the fastest CKA variant at O(n * d^2) complexity, where
    n is the number of samples and d = max(d_x, d_y).

    Args:
        X: Activation matrix from the first representation, shape
            ``(n_samples, d_x)``. Typically the hidden states of
            language A at a specific layer.
        Y: Activation matrix from the second representation, shape
            ``(n_samples, d_y)``. Typically the hidden states of
            language B at the same layer. Note: ``d_x`` and ``d_y``
            may differ (CKA is dimension-agnostic).
        eps: Small constant added to the denominator for numerical
            stability, preventing division by zero when both
            representations are near-constant.

    Returns:
        Scalar CKA similarity score in [0, 1]:
            - 1.0 = identical representations (up to linear transform)
            - 0.0 = completely unrelated representations

    Raises:
        ValueError: If inputs fail validation (wrong shape, NaN, etc.).

    Example::

        >>> X = torch.randn(256, 768)  # 256 sentences, 768-dim
        >>> Y = torch.randn(256, 768)  # same sentences, different lang
        >>> score = linear_cka(X, Y)
        >>> 0.0 <= score.item() <= 1.0
        True
    """
    _validate_activation_pair(X, Y, "linear_cka")

    # Center columns (subtract mean representation per feature).
    # This is equivalent to using centered Gram matrices but operates
    # in feature space for O(n*d^2) instead of O(n^2*d).
    X_centered = X - X.mean(dim=0, keepdim=True)
    Y_centered = Y - Y.mean(dim=0, keepdim=True)

    # Cross-covariance matrix: shape (d_y, d_x).
    cross_cov = Y_centered.T @ X_centered

    # Numerator: squared Frobenius norm of cross-covariance.
    numerator = torch.norm(cross_cov, p="fro") ** 2

    # Self-covariance Frobenius norms for normalization.
    x_self_norm = torch.norm(X_centered.T @ X_centered, p="fro")
    y_self_norm = torch.norm(Y_centered.T @ Y_centered, p="fro")

    # Denominator: product of self-covariance norms.
    denominator = x_self_norm * y_self_norm + eps

    return numerator / denominator


def rbf_cka(
    X: torch.Tensor,
    Y: torch.Tensor,
    sigma_x: float | None = None,
    sigma_y: float | None = None,
    eps: float = 1e-10,
) -> torch.Tensor:
    """Compute RBF (Gaussian) kernel CKA between activation matrices.

    More expressive than linear CKA — captures nonlinear relationships
    between representations. Uses the HSIC-based CKA formula with
    Gaussian kernels.

    Computational cost is O(n^2 * d) for kernel computation plus
    O(n^2) for HSIC, so this is slower than linear CKA for large n.

    Args:
        X: Activation matrix, shape ``(n_samples, d_x)``.
        Y: Activation matrix, shape ``(n_samples, d_y)``.
        sigma_x: RBF bandwidth for X. If ``None``, uses the median
            heuristic (data-adaptive, no tuning needed).
        sigma_y: RBF bandwidth for Y. If ``None``, uses the median
            heuristic.
        eps: Numerical stability constant for the denominator.

    Returns:
        Scalar CKA similarity score in [0, 1].

    Raises:
        ValueError: If inputs fail validation.

    Example::

        >>> score = rbf_cka(torch.randn(100, 512), torch.randn(100, 512))
        >>> 0.0 <= score.item() <= 1.0
        True
    """
    _validate_activation_pair(X, Y, "rbf_cka")

    # Compute RBF kernel matrices.
    K = _rbf_kernel(X, sigma_x)
    L = _rbf_kernel(Y, sigma_y)

    # Compute HSIC values.
    hsic_kl = _hsic(K, L)
    hsic_kk = _hsic(K, K)
    hsic_ll = _hsic(L, L)

    # CKA = HSIC(K,L) / sqrt(HSIC(K,K) * HSIC(L,L))
    denominator = torch.sqrt(hsic_kk * hsic_ll) + eps

    return hsic_kl / denominator


def whitened_cka(
    X: torch.Tensor,
    Y: torch.Tensor,
    regularization: float = 1e-6,
    eps: float = 1e-10,
) -> torch.Tensor:
    """Compute anisotropy-corrected (whitened) Linear CKA.

    Standard cosine similarity and CKA can be confounded by
    **representation anisotropy** — the phenomenon where all
    representation vectors cluster in a narrow cone of the embedding
    space, leading to high similarity scores even between unrelated
    inputs.

    Whitened CKA applies ZCA (Zero-phase Component Analysis) whitening
    to both X and Y before computing linear CKA, removing the
    anisotropy confound. The whitening transform is:

        X_w = (X - mean(X)) @ W

    where W = V @ diag(1/sqrt(eigenvalues)) @ V.T is the ZCA
    whitening matrix derived from the covariance of X.

    This is Novel Technique 2 from the TIN-7 specification.

    Args:
        X: Activation matrix, shape ``(n_samples, d_x)``.
        Y: Activation matrix, shape ``(n_samples, d_y)``.
        regularization: Small constant added to eigenvalues before
            inversion, preventing numerical instability from near-zero
            eigenvalues. Higher values = more aggressive regularization.
        eps: Numerical stability constant for the CKA denominator.

    Returns:
        Scalar whitened CKA similarity score in [0, 1].

    Raises:
        ValueError: If inputs fail validation.

    Note:
        Whitened CKA is more computationally expensive than standard
        linear CKA because it requires eigendecomposition of the
        covariance matrices (O(d^3) per matrix).
    """
    _validate_activation_pair(X, Y, "whitened_cka")

    # Apply whitening to both matrices.
    X_whitened = _whiten_representations(X, regularization)
    Y_whitened = _whiten_representations(Y, regularization)

    # Compute standard linear CKA on the whitened representations.
    return linear_cka(X_whitened, Y_whitened, eps=eps)


def _whiten_representations(
    X: torch.Tensor,
    regularization: float = 1e-6,
) -> torch.Tensor:
    """Apply ZCA whitening to an activation matrix.

    ZCA whitening transforms the data so that its covariance matrix
    becomes the identity matrix, while staying as close as possible
    to the original data (in Euclidean distance).

    Args:
        X: Activation matrix, shape ``(n_samples, d_features)``.
        regularization: Tikhonov regularization added to eigenvalues.

    Returns:
        Whitened activation matrix of the same shape.
    """
    # Center the data.
    mean = X.mean(dim=0, keepdim=True)
    X_centered = X - mean

    # Compute the sample covariance matrix.
    # Shape: (d, d). We use (X.T @ X) / n rather than torch.cov
    # for explicit control over the computation.
    n = X_centered.shape[0]
    covariance = (X_centered.T @ X_centered) / n

    # Eigendecomposition of the symmetric covariance matrix.
    # eigh returns eigenvalues in ascending order and corresponding
    # eigenvectors as columns.
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)

    # Compute the whitening matrix: V @ diag(1/sqrt(lambda + reg)) @ V.T
    # The regularization prevents division by near-zero eigenvalues.
    inv_sqrt_eigenvalues = 1.0 / torch.sqrt(torch.clamp(eigenvalues, min=0.0) + regularization)
    whitening_matrix = (
        eigenvectors
        @ torch.diag(inv_sqrt_eigenvalues)
        @ eigenvectors.T
    )

    return X_centered @ whitening_matrix


# ===================================================================
# Mini-batch CKA (memory-efficient streaming computation)
# ===================================================================

@dataclass
class MinibatchCKAAccumulator:
    """Streaming accumulator for memory-efficient Linear CKA computation.

    Instead of materializing full ``(n_samples, d)`` activation matrices
    in memory, this accumulator collects sufficient statistics
    (cross-covariance and self-covariance matrices) across mini-batches,
    then computes CKA from the accumulated statistics.

    Memory usage is O(d_x^2 + d_y^2 + d_x * d_y) regardless of the
    total number of samples, making this suitable for large-scale
    analysis with thousands of sentences.

    The mathematical basis is:

        X_c.T @ X_c = X.T @ X - n * mean_x @ mean_x.T

    which allows accumulating X.T @ X and sum(X) in streaming fashion.

    Attributes:
        d_x: Feature dimension of the first representation.
        d_y: Feature dimension of the second representation.
        device: Torch device for accumulation buffers ('cpu' or 'cuda').

    Usage::

        acc = MinibatchCKAAccumulator(d_x=768, d_y=768)
        for X_batch, Y_batch in dataloader:
            acc.update(X_batch, Y_batch)
        score = acc.compute()
        acc.reset()  # reuse for next comparison

    Raises:
        ValueError: If ``compute()`` is called before any ``update()``.
    """

    d_x: int
    d_y: int
    device: str = "cpu"

    # Private accumulation buffers (not shown in repr).
    _XtX: torch.Tensor | None = field(default=None, repr=False)
    _YtY: torch.Tensor | None = field(default=None, repr=False)
    _YtX: torch.Tensor | None = field(default=None, repr=False)
    _sum_x: torch.Tensor | None = field(default=None, repr=False)
    _sum_y: torch.Tensor | None = field(default=None, repr=False)
    _n: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        """Initialize accumulation buffers to zero."""
        self.reset()

    def reset(self) -> None:
        """Clear all accumulated statistics and reset sample count.

        Call this between different CKA comparisons to reuse the
        same accumulator object.
        """
        dev = self.device
        self._XtX = torch.zeros(self.d_x, self.d_x, device=dev)
        self._YtY = torch.zeros(self.d_y, self.d_y, device=dev)
        self._YtX = torch.zeros(self.d_y, self.d_x, device=dev)
        self._sum_x = torch.zeros(self.d_x, device=dev)
        self._sum_y = torch.zeros(self.d_y, device=dev)
        self._n = 0

    @torch.no_grad()
    def update(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        """Accumulate statistics from a mini-batch of activations.

        Args:
            X: Batch of activations from the first representation,
                shape ``(batch_size, d_x)``.
            Y: Batch of activations from the second representation,
                shape ``(batch_size, d_y)``. Must have the same
                batch size as X.

        Raises:
            ValueError: If batch sizes don't match or feature
                dimensions don't match the accumulator configuration.
        """
        if X.shape[0] != Y.shape[0]:
            raise ValueError(
                f"MinibatchCKAAccumulator.update: Batch size mismatch — "
                f"X has {X.shape[0]} samples, Y has {Y.shape[0]}."
            )
        if X.shape[1] != self.d_x:
            raise ValueError(
                f"MinibatchCKAAccumulator.update: X feature dim is "
                f"{X.shape[1]}, expected {self.d_x}."
            )
        if Y.shape[1] != self.d_y:
            raise ValueError(
                f"MinibatchCKAAccumulator.update: Y feature dim is "
                f"{Y.shape[1]}, expected {self.d_y}."
            )

        # Move to accumulator device and ensure float32 precision
        # for numerical stability during accumulation.
        X = X.to(self.device, dtype=torch.float32)
        Y = Y.to(self.device, dtype=torch.float32)

        # Accumulate outer products and sums.
        self._XtX += X.T @ X
        self._YtY += Y.T @ Y
        self._YtX += Y.T @ X
        self._sum_x += X.sum(dim=0)
        self._sum_y += Y.sum(dim=0)
        self._n += X.shape[0]

    def compute(self, eps: float = 1e-10) -> float:
        """Compute Linear CKA from the accumulated statistics.

        Uses the identity:
            X_centered.T @ X_centered = X.T @ X - n * mean_x @ mean_x.T

        to recover centered cross/self covariances from the accumulated
        raw statistics, then applies the standard linear CKA formula.

        Args:
            eps: Numerical stability constant for the denominator.

        Returns:
            CKA score as a Python float in [0, 1].

        Raises:
            ValueError: If no samples have been accumulated yet.
        """
        if self._n == 0:
            raise ValueError(
                "MinibatchCKAAccumulator.compute: No samples accumulated. "
                "Call update() with at least one batch before computing."
            )

        n = self._n
        mean_x = self._sum_x / n
        mean_y = self._sum_y / n

        # Recover centered covariance matrices from raw statistics.
        # X_c.T @ X_c = X.T @ X - n * outer(mean_x, mean_x)
        XtX_centered = self._XtX - n * mean_x.outer(mean_x)
        YtY_centered = self._YtY - n * mean_y.outer(mean_y)
        YtX_centered = self._YtX - n * mean_y.outer(mean_x)

        # CKA formula in covariance space.
        numerator = torch.norm(YtX_centered, p="fro") ** 2
        denominator = (
            torch.norm(XtX_centered, p="fro")
            * torch.norm(YtY_centered, p="fro")
            + eps
        )

        return (numerator / denominator).item()


def minibatch_cka(
    X: torch.Tensor,
    Y: torch.Tensor,
    batch_size: int = 256,
    eps: float = 1e-10,
) -> float:
    """Convenience wrapper: compute Linear CKA in mini-batches.

    Avoids O(n * max(d_x, d_y)) peak memory from holding the full
    activation matrices and their products simultaneously. Internally
    creates a ``MinibatchCKAAccumulator`` and feeds data in chunks.

    This is functionally equivalent to ``linear_cka(X, Y)`` but with
    bounded memory usage proportional to ``batch_size * d`` rather
    than ``n * d``.

    Args:
        X: Full activation matrix, shape ``(n_samples, d_x)``.
        Y: Full activation matrix, shape ``(n_samples, d_y)``.
        batch_size: Number of samples per mini-batch. Larger values
            are faster but use more memory.
        eps: Numerical stability constant.

    Returns:
        CKA score as a Python float in [0, 1].

    Raises:
        ValueError: If inputs fail validation.
    """
    _validate_activation_pair(X, Y, "minibatch_cka")

    n = X.shape[0]
    accumulator = MinibatchCKAAccumulator(
        d_x=X.shape[1],
        d_y=Y.shape[1],
        device=str(X.device),
    )

    # Feed data in chunks.
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        accumulator.update(X[start:end], Y[start:end])

    return accumulator.compute(eps=eps)


# ===================================================================
# Statistical testing
# ===================================================================

def cka_permutation_test(
    X: torch.Tensor,
    Y: torch.Tensor,
    n_permutations: int = 1000,
    kernel: str = "linear",
    seed: int = 42,
) -> dict[str, float]:
    """Permutation test for CKA statistical significance.

    Tests whether the observed CKA score between X and Y is
    significantly higher than expected by chance. The null hypothesis
    is that there is no representational similarity (samples in Y are
    randomly shuffled, breaking any true alignment with X).

    Procedure:
        1. Compute the observed CKA score on the aligned data.
        2. For each of ``n_permutations`` replicates, randomly shuffle
           the rows of Y and recompute CKA.
        3. The p-value is the fraction of null scores >= observed score.

    A p-value < 0.05 means the observed CKA is significantly above
    what would be expected from random alignment, confirming genuine
    cross-lingual representational similarity.

    Args:
        X: Activation matrix, shape ``(n_samples, d_x)``.
        Y: Activation matrix, shape ``(n_samples, d_y)``.
        n_permutations: Number of permutation replicates. Higher values
            give more precise p-value estimates. 1000 is standard;
            10000 for publication-quality results.
        kernel: CKA variant to use — "linear" or "rbf".
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with:
            - ``"observed_cka"``: The CKA score on aligned data.
            - ``"p_value"``: Fraction of null scores >= observed.
            - ``"null_mean"``: Mean of the null distribution.
            - ``"null_std"``: Standard deviation of the null distribution.
            - ``"is_significant"``: Boolean, True if p < 0.05.

    Raises:
        ValueError: If ``kernel`` is not "linear" or "rbf".
        ValueError: If inputs fail validation.

    Example::

        >>> result = cka_permutation_test(X, Y, n_permutations=500)
        >>> print(f"CKA = {result['observed_cka']:.4f}, p = {result['p_value']:.4f}")
    """
    _validate_activation_pair(X, Y, "cka_permutation_test")

    if kernel not in ("linear", "rbf"):
        raise ValueError(
            f"cka_permutation_test: kernel must be 'linear' or 'rbf', "
            f"got '{kernel}'."
        )

    # Set seeds for reproducibility.
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Select the CKA function.
    cka_fn = linear_cka if kernel == "linear" else rbf_cka

    # Compute the observed (true) CKA score.
    observed = cka_fn(X, Y).item()

    # Generate the null distribution by shuffling Y.
    null_scores: list[float] = []
    n = X.shape[0]

    for _ in range(n_permutations):
        # Random permutation of sample indices.
        perm = torch.randperm(n)
        Y_shuffled = Y[perm]
        null_scores.append(cka_fn(X, Y_shuffled).item())

    null_array = np.array(null_scores)

    # P-value: fraction of null scores at least as extreme as observed.
    p_value = float(np.mean(null_array >= observed))

    return {
        "observed_cka": observed,
        "p_value": p_value,
        "null_mean": float(null_array.mean()),
        "null_std": float(null_array.std()),
        "is_significant": p_value < 0.05,
    }


# ===================================================================
# Visualization data containers
# ===================================================================

@dataclass
class CKAHeatmapData:
    """Data container for a layer-wise CKA heatmap.

    Stores the full similarity matrix along with axis labels,
    and provides serialization for JSON/YAML logging.

    Attributes:
        scores: 2D numpy array of CKA scores, shape
            ``(n_row_layers, n_col_layers)``. For cross-lingual
            analysis, rows and columns are both languages, and
            each "layer" axis is actually a separate heatmap.
        row_names: Labels for the rows (e.g., language names).
        col_names: Labels for the columns.
    """

    scores: NDArray[np.floating]
    row_names: list[str]
    col_names: list[str]

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dictionary.

        Returns:
            Dictionary with keys "scores" (nested list), "row_names",
            and "col_names".
        """
        return {
            "scores": self.scores.tolist(),
            "row_names": self.row_names,
            "col_names": self.col_names,
        }


# ===================================================================
# Layer-wise CKA computation
# ===================================================================

def compute_layerwise_cka(
    activations_a: dict[str, torch.Tensor],
    activations_b: dict[str, torch.Tensor],
    kernel: str = "linear",
    batch_size: int | None = None,
) -> CKAHeatmapData:
    """Compute all-pairs CKA between two sets of layer activations.

    This is the workhorse function for comparing representations
    across languages at each layer. Given activations from
    representation A (e.g., English) and representation B (e.g.,
    Hindi), it computes CKA for every (layer_a, layer_b) pair.

    For cross-lingual alignment, you typically compare the same
    layer across languages, so the diagonal of the resulting
    heatmap is the quantity of interest.

    Args:
        activations_a: Dictionary mapping layer names to activation
            tensors of shape ``(n_samples, d)``.
        activations_b: Dictionary mapping layer names to activation
            tensors of shape ``(n_samples, d)``.
        kernel: CKA variant — "linear", "rbf", or "whitened".
        batch_size: If set, use mini-batch CKA for memory efficiency.
            Only applies when ``kernel="linear"``.

    Returns:
        ``CKAHeatmapData`` with the full ``(n_layers_a, n_layers_b)``
        similarity matrix and layer name labels.

    Raises:
        ValueError: If an unknown kernel is specified.
    """
    # Sort layer names for deterministic ordering.
    a_names = sorted(activations_a.keys())
    b_names = sorted(activations_b.keys())

    scores = np.zeros((len(a_names), len(b_names)), dtype=np.float64)

    for i, a_name in enumerate(a_names):
        a_act = activations_a[a_name]

        for j, b_name in enumerate(b_names):
            b_act = activations_b[b_name]

            if batch_size is not None and kernel == "linear":
                scores[i, j] = minibatch_cka(a_act, b_act, batch_size)
            elif kernel == "linear":
                scores[i, j] = linear_cka(a_act, b_act).item()
            elif kernel == "rbf":
                scores[i, j] = rbf_cka(a_act, b_act).item()
            elif kernel == "whitened":
                scores[i, j] = whitened_cka(a_act, b_act).item()
            else:
                raise ValueError(
                    f"compute_layerwise_cka: Unknown kernel '{kernel}'. "
                    f"Use 'linear', 'rbf', or 'whitened'."
                )

    return CKAHeatmapData(
        scores=scores,
        row_names=a_names,
        col_names=b_names,
    )
