"""
Publication-quality visualization functions for cross-lingual analysis.

This module provides all plotting functions used across the 8 analysis
notebooks. Every function follows a consistent API pattern:

    - Accepts pre-computed data (numpy arrays, dicts) — no model
      inference happens here.
    - Returns a ``matplotlib.figure.Figure`` object for further
      customization or saving.
    - Optionally saves to a file path if ``save_path`` is provided.
    - Uses a consistent color scheme and typography suitable for
      research papers and presentations.

Plot categories:
    - **Heatmaps**: Language-pair CKA matrices at each layer.
    - **Convergence curves**: Average cross-lingual CKA vs. layer.
    - **Dendrograms**: Hierarchical language clustering per layer.
    - **Retrieval plots**: MRR and Recall@k curves.
    - **Script decomposition**: Intra- vs. inter-script CKA.
    - **Dimensionality reduction**: t-SNE/UMAP embeddings colored
      by language.

Style choices:
    - Colormap: ``"RdBu_r"`` for CKA heatmaps (diverging, 0=blue,
      1=red), ``"viridis"`` for sequential data.
    - Font: DejaVu Sans (default matplotlib, widely available).
    - Figure size: Default 10x8 inches for single plots, 20x5 for
      multi-panel layouts.
    - DPI: 150 for screen display, 300 for saved files.
"""


import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from numpy.typing import NDArray

# Module-level logger.
logger = logging.getLogger(__name__)

# --- Global style configuration ---
# Set once at module import; individual functions can override.
sns.set_theme(style="whitegrid", font_scale=1.1)
matplotlib.rcParams["figure.dpi"] = 150
matplotlib.rcParams["savefig.dpi"] = 300
matplotlib.rcParams["savefig.bbox"] = "tight"


# ===================================================================
# Internal helpers
# ===================================================================

def _save_figure(fig: plt.Figure, save_path: str | None) -> None:
    """Save a figure to disk if a path is provided.

    Creates parent directories if they don't exist.

    Args:
        fig: Matplotlib figure to save.
        save_path: File path to save to (e.g., "results/fig.png").
            If ``None``, does nothing.
    """
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
        logger.info("Figure saved to %s", path)


# ===================================================================
# CKA Heatmap
# ===================================================================

def plot_cka_heatmap(
    cka_matrix: NDArray,
    language_names: list[str],
    layer_index: int,
    title: str | None = None,
    annotate: bool = True,
    cmap: str = "RdBu_r",
    vmin: float = 0.0,
    vmax: float = 1.0,
    figsize: tuple[float, float] = (10, 8),
    save_path: str | None = None,
) -> plt.Figure:
    """Plot a cross-lingual CKA similarity heatmap for one layer.

    Displays a square heatmap where entry (i, j) is the CKA similarity
    between the representations of language i and language j at the
    specified layer. The diagonal is always 1.0 (self-similarity).

    Args:
        cka_matrix: Symmetric CKA matrix of shape
            ``(n_langs, n_langs)`` with values in [0, 1].
        language_names: Labels for rows and columns. Ideally ISO codes
            or short names for readability.
        layer_index: Which layer this heatmap represents (for title).
        title: Custom title. If ``None``, auto-generated from
            ``layer_index``.
        annotate: If True, overlay CKA values as text on each cell.
        cmap: Matplotlib colormap name.
        vmin: Minimum value for color scale.
        vmax: Maximum value for color scale.
        figsize: Figure size in inches (width, height).
        save_path: Optional file path to save the figure.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    if title is None:
        title = f"Cross-Lingual CKA Similarity — Layer {layer_index}"

    sns.heatmap(
        cka_matrix,
        xticklabels=language_names,
        yticklabels=language_names,
        annot=annotate,
        fmt=".2f" if annotate else "",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "CKA Similarity", "shrink": 0.8},
        ax=ax,
    )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Language", fontsize=12)
    ax.set_ylabel("Language", fontsize=12)

    # Rotate x-axis labels for readability.
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()
    _save_figure(fig, save_path)

    return fig


def plot_multi_layer_heatmaps(
    cka_matrices: dict[int, NDArray],
    language_names: list[str],
    suptitle: str = "Cross-Lingual CKA Across Layers",
    ncols: int = 2,
    figsize: tuple[float, float] | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """Plot CKA heatmaps for multiple layers in a grid layout.

    Arranges heatmaps in a grid with ``ncols`` columns, giving each
    subplot enough room so that 13x13 annotated cells remain legible.

    Args:
        cka_matrices: Dictionary mapping layer index to CKA matrix.
        language_names: Language labels for axes.
        suptitle: Super-title for the entire figure.
        ncols: Number of columns in the grid (default 2).
        figsize: Figure size. If ``None``, auto-computed from the
            grid dimensions.
        save_path: Optional file path to save.

    Returns:
        Matplotlib Figure object.
    """
    import math

    n_layers = len(cka_matrices)
    sorted_layers = sorted(cka_matrices.keys())

    ncols = min(ncols, n_layers)
    nrows = math.ceil(n_layers / ncols)

    if figsize is None:
        figsize = (8 * ncols, 7 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

    # Normalise axes to a flat list for uniform indexing.
    if n_layers == 1:
        axes_flat = [axes]
    else:
        axes_flat = list(np.asarray(axes).flat)

    for idx, layer_idx in enumerate(sorted_layers):
        ax = axes_flat[idx]
        matrix = cka_matrices[layer_idx]

        sns.heatmap(
            matrix,
            xticklabels=language_names,
            yticklabels=language_names,
            annot=True,
            fmt=".2f",
            cmap="RdBu_r",
            vmin=0.0,
            vmax=1.0,
            square=True,
            linewidths=0.3,
            cbar=False,
            ax=ax,
        )
        ax.set_title(f"Layer {layer_idx}", fontsize=12, fontweight="bold")
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y", rotation=0)

    # Hide any unused subplot slots.
    for idx in range(n_layers, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(suptitle, fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save_figure(fig, save_path)

    return fig


# ===================================================================
# Convergence curve
# ===================================================================

def plot_convergence_curve(
    layer_indices: list[int],
    avg_cka_per_layer: list[float],
    ci_lower: list[float] | None = None,
    ci_upper: list[float] | None = None,
    threshold: float = 0.75,
    rbf_cka_per_layer: list[float] | None = None,
    whitened_cka_per_layer: list[float] | None = None,
    title: str = "Cross-Lingual CKA Convergence",
    figsize: tuple[float, float] = (10, 6),
    save_path: str | None = None,
) -> plt.Figure:
    """Plot average cross-lingual CKA vs. layer depth.

    This is the primary visualization for identifying the
    **convergence layer** — the layer where cross-lingual similarity
    stabilizes above the CKA threshold (default 0.75).

    Args:
        layer_indices: List of layer indices (x-axis).
        avg_cka_per_layer: Average cross-lingual CKA at each layer
            (y-axis), computed from the off-diagonal elements of the
            CKA matrix.
        ci_lower: Lower bound of 95% confidence interval per layer.
        ci_upper: Upper bound of 95% confidence interval per layer.
        threshold: CKA threshold line for "acceptable alignment"
            (default 0.75, from notebook 02 of the reference repo).
        rbf_cka_per_layer: Optional RBF CKA values for overlay.
        whitened_cka_per_layer: Optional whitened CKA values.
        title: Plot title.
        figsize: Figure size.
        save_path: Optional save path.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # --- Main convergence curve (Linear CKA) ---
    ax.plot(
        layer_indices,
        avg_cka_per_layer,
        "o-",
        color="#2196F3",
        linewidth=2.5,
        markersize=8,
        label="Linear CKA (avg cross-lingual)",
        zorder=5,
    )

    # --- Confidence interval band ---
    if ci_lower is not None and ci_upper is not None:
        ax.fill_between(
            layer_indices,
            ci_lower,
            ci_upper,
            alpha=0.2,
            color="#2196F3",
            label="95% CI",
        )

    # --- Optional RBF CKA overlay ---
    if rbf_cka_per_layer is not None:
        ax.plot(
            layer_indices,
            rbf_cka_per_layer,
            "s--",
            color="#FF9800",
            linewidth=2,
            markersize=7,
            label="RBF CKA",
            zorder=4,
        )

    # --- Optional whitened CKA overlay ---
    if whitened_cka_per_layer is not None:
        ax.plot(
            layer_indices,
            whitened_cka_per_layer,
            "D:",
            color="#4CAF50",
            linewidth=2,
            markersize=7,
            label="Whitened CKA",
            zorder=4,
        )

    # --- Threshold line ---
    ax.axhline(
        y=threshold,
        color="#F44336",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label=f"Threshold ({threshold})",
    )

    # --- Formatting ---
    ax.set_xlabel("Layer Index", fontsize=13)
    ax.set_ylabel("Average Cross-Lingual CKA", fontsize=13)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(layer_indices)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower right", fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_figure(fig, save_path)

    return fig


def plot_language_pair_trajectories(
    layer_indices: list[int],
    cka_trajectories: dict[str, list[float]],
    highlight_pairs: list[str] | None = None,
    title: str = "Per-Pair CKA Trajectories Across Layers",
    figsize: tuple[float, float] = (12, 7),
    save_path: str | None = None,
) -> plt.Figure:
    """Plot CKA trajectories for individual language pairs (spaghetti plot).

    Each line represents a specific language pair's CKA across layers.
    Optionally highlights specific pairs of interest.

    Args:
        layer_indices: List of layer indices (x-axis).
        cka_trajectories: Dictionary mapping pair names (e.g.,
            "english-hindi") to lists of CKA values per layer.
        highlight_pairs: Optional list of pair names to draw with
            thicker lines and brighter colors.
        title: Plot title.
        figsize: Figure size.
        save_path: Optional save path.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Use a colormap for the many lines.
    n_pairs = len(cka_trajectories)
    colors = plt.cm.tab20(np.linspace(0, 1, min(n_pairs, 20)))

    for idx, (pair_name, values) in enumerate(
        sorted(cka_trajectories.items())
    ):
        is_highlighted = (
            highlight_pairs is not None and pair_name in highlight_pairs
        )

        ax.plot(
            layer_indices,
            values,
            linewidth=2.5 if is_highlighted else 0.8,
            alpha=1.0 if is_highlighted else 0.3,
            color=colors[idx % len(colors)],
            label=pair_name if is_highlighted else None,
        )

    ax.set_xlabel("Layer Index", fontsize=13)
    ax.set_ylabel("CKA Similarity", fontsize=13)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(layer_indices)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    if highlight_pairs:
        ax.legend(loc="best", fontsize=9, framealpha=0.9)

    plt.tight_layout()
    _save_figure(fig, save_path)

    return fig


# ===================================================================
# Dendrogram
# ===================================================================

def plot_dendrogram(
    linkage_matrix: NDArray,
    language_names: list[str],
    layer_index: int,
    title: str | None = None,
    color_threshold: float | None = None,
    figsize: tuple[float, float] = (10, 6),
    save_path: str | None = None,
) -> plt.Figure:
    """Plot a dendrogram from hierarchical clustering results.

    Args:
        linkage_matrix: Scipy linkage matrix from
            ``compute_hierarchical_clustering``.
        language_names: Language labels for the leaves.
        layer_index: Layer number (for auto-titling).
        title: Custom title. Auto-generated if ``None``.
        color_threshold: Distance threshold for coloring clusters.
            If ``None``, uses scipy's default.
        figsize: Figure size.
        save_path: Optional save path.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    if title is None:
        title = f"Language Clustering Dendrogram — Layer {layer_index}"

    from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram

    scipy_dendrogram(
        linkage_matrix,
        labels=language_names,
        leaf_rotation=45,
        leaf_font_size=11,
        color_threshold=color_threshold,
        ax=ax,
    )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("Distance (1 - CKA)", fontsize=12)
    ax.set_xlabel("Language", fontsize=12)

    plt.tight_layout()
    _save_figure(fig, save_path)

    return fig


def plot_dendrograms_across_layers(
    clustering_results: dict[int, dict],
    title: str = "Language Cluster Dissolution Across Layers",
    figsize: tuple[float, float] | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """Plot dendrograms for multiple layers side by side.

    Shows how language clusters dissolve (or persist) as depth
    increases.

    Args:
        clustering_results: Dictionary mapping layer index to the
            result dict from ``compute_hierarchical_clustering``.
        title: Super-title.
        figsize: Figure size. Auto-computed if ``None``.
        save_path: Optional save path.

    Returns:
        Matplotlib Figure object.
    """
    n_layers = len(clustering_results)
    sorted_layers = sorted(clustering_results.keys())

    if figsize is None:
        figsize = (7 * n_layers, 5)

    fig, axes = plt.subplots(1, n_layers, figsize=figsize)

    if n_layers == 1:
        axes = [axes]

    from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram

    for ax, layer_idx in zip(axes, sorted_layers):
        result = clustering_results[layer_idx]

        scipy_dendrogram(
            result["linkage_matrix"],
            labels=result["language_names"],
            leaf_rotation=45,
            leaf_font_size=9,
            ax=ax,
        )
        ax.set_title(f"Layer {layer_idx}", fontsize=12, fontweight="bold")
        ax.set_ylabel("Distance (1 - CKA)")

    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save_figure(fig, save_path)

    return fig


# ===================================================================
# Retrieval metrics plots
# ===================================================================

def plot_retrieval_curves(
    layer_indices: list[int],
    mrr_per_layer: dict[str, list[float]],
    title: str = "Translation Retrieval MRR vs. Layer Depth",
    figsize: tuple[float, float] = (12, 7),
    save_path: str | None = None,
) -> plt.Figure:
    """Plot MRR vs. layer depth for each target language.

    Each line represents the MRR of retrieving translations in a
    target language given English queries, across layers.

    Args:
        layer_indices: Layer indices for x-axis.
        mrr_per_layer: Dictionary mapping target language names to
            lists of MRR values (one per layer).
        title: Plot title.
        figsize: Figure size.
        save_path: Optional save path.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Use a qualitative colormap for distinguishable colors.
    n_langs = len(mrr_per_layer)
    colors = plt.cm.Set2(np.linspace(0, 1, max(n_langs, 8)))

    for idx, (lang_name, mrr_values) in enumerate(
        sorted(mrr_per_layer.items())
    ):
        ax.plot(
            layer_indices,
            mrr_values,
            "o-",
            color=colors[idx % len(colors)],
            linewidth=2,
            markersize=6,
            label=lang_name,
        )

    ax.set_xlabel("Layer Index", fontsize=13)
    ax.set_ylabel("Mean Reciprocal Rank (MRR)", fontsize=13)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(layer_indices)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=9,
        framealpha=0.9,
    )
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_figure(fig, save_path)

    return fig


def plot_recall_bars(
    recall_scores: dict[str, float],
    k: int = 1,
    layer_index: int = 0,
    title: str | None = None,
    figsize: tuple[float, float] = (12, 6),
    save_path: str | None = None,
) -> plt.Figure:
    """Plot Recall@k as a bar chart per target language.

    Args:
        recall_scores: Dictionary mapping language names to Recall@k
            scores.
        k: The k value (for labeling).
        layer_index: Layer index (for labeling).
        title: Custom title. Auto-generated if ``None``.
        figsize: Figure size.
        save_path: Optional save path.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    if title is None:
        title = f"Recall@{k} per Language — Layer {layer_index}"

    sorted_langs = sorted(recall_scores.keys())
    scores = [recall_scores[lang] for lang in sorted_langs]

    # Color bars by score magnitude.
    colors = plt.cm.RdYlGn([s for s in scores])

    bars = ax.bar(sorted_langs, scores, color=colors, edgecolor="gray", linewidth=0.5)

    # Add value labels on top of each bar.
    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.02,
            f"{score:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xlabel("Target Language", fontsize=12)
    ax.set_ylabel(f"Recall@{k}", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.15)
    plt.xticks(rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    _save_figure(fig, save_path)

    return fig


# ===================================================================
# Script decomposition
# ===================================================================

def plot_script_decomposition(
    layer_indices: list[int],
    intra_script_cka: list[float],
    inter_script_cka: list[float],
    title: str = "Intra-Script vs. Inter-Script CKA",
    figsize: tuple[float, float] = (10, 6),
    save_path: str | None = None,
) -> plt.Figure:
    """Plot intra- vs. inter-script CKA across layers.

    Shows whether cross-lingual alignment is driven by shared
    writing systems (token-surface similarity) or true semantic
    understanding.

    Args:
        layer_indices: Layer indices for x-axis.
        intra_script_cka: Average CKA within same script group
            per layer.
        inter_script_cka: Average CKA across different script
            groups per layer.
        title: Plot title.
        figsize: Figure size.
        save_path: Optional save path.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(
        layer_indices,
        intra_script_cka,
        "o-",
        color="#E91E63",
        linewidth=2.5,
        markersize=8,
        label="Intra-script (same writing system)",
    )
    ax.plot(
        layer_indices,
        inter_script_cka,
        "s-",
        color="#3F51B5",
        linewidth=2.5,
        markersize=8,
        label="Inter-script (different writing systems)",
    )

    # Shade the gap between intra and inter.
    ax.fill_between(
        layer_indices,
        intra_script_cka,
        inter_script_cka,
        alpha=0.15,
        color="#9C27B0",
        label="Script bias gap",
    )

    ax.set_xlabel("Layer Index", fontsize=13)
    ax.set_ylabel("Average CKA", fontsize=13)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(layer_indices)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="best", fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_figure(fig, save_path)

    return fig


def plot_family_gap_curve(
    layer_indices: list[int],
    intra_family_cka: list[float],
    inter_family_cka: list[float],
    title: str = "Language Family Bias Across Layers",
    figsize: tuple[float, float] = (10, 6),
    save_path: str | None = None,
) -> plt.Figure:
    """Plot intra- vs. inter-family CKA across layers.

    Similar to ``plot_script_decomposition`` but groups by language
    family instead of writing script.

    Args:
        layer_indices: Layer indices.
        intra_family_cka: Avg intra-family CKA per layer.
        inter_family_cka: Avg inter-family CKA per layer.
        title: Plot title.
        figsize: Figure size.
        save_path: Optional save path.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(
        layer_indices,
        intra_family_cka,
        "o-",
        color="#FF5722",
        linewidth=2.5,
        markersize=8,
        label="Intra-family (same genetic family)",
    )
    ax.plot(
        layer_indices,
        inter_family_cka,
        "s-",
        color="#009688",
        linewidth=2.5,
        markersize=8,
        label="Inter-family (different families)",
    )

    ax.fill_between(
        layer_indices,
        intra_family_cka,
        inter_family_cka,
        alpha=0.15,
        color="#795548",
        label="Family bias gap",
    )

    ax.set_xlabel("Layer Index", fontsize=13)
    ax.set_ylabel("Average CKA", fontsize=13)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(layer_indices)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="best", fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_figure(fig, save_path)

    return fig


# ===================================================================
# Anisotropy visualization
# ===================================================================

def plot_anisotropy_heatmap(
    anisotropy_scores: NDArray,
    language_names: list[str],
    layer_indices: list[int],
    title: str = "Representation Anisotropy by Language and Layer",
    figsize: tuple[float, float] = (10, 6),
    save_path: str | None = None,
) -> plt.Figure:
    """Plot anisotropy scores as a language x layer heatmap.

    Anisotropy is the average cosine similarity between random pairs
    of representations. High anisotropy (close to 1) means all vectors
    point in similar directions — a known problem for transformer
    representations that confounds cosine-based similarity.

    Args:
        anisotropy_scores: Matrix of shape ``(n_langs, n_layers)``
            with anisotropy values in [0, 1].
        language_names: Language labels for rows.
        layer_indices: Layer labels for columns.
        title: Plot title.
        figsize: Figure size.
        save_path: Optional save path.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        anisotropy_scores,
        xticklabels=[f"Layer {i}" for i in layer_indices],
        yticklabels=language_names,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        vmin=0.0,
        vmax=1.0,
        linewidths=0.5,
        cbar_kws={"label": "Anisotropy (avg cosine sim)"},
        ax=ax,
    )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Language", fontsize=12)

    plt.tight_layout()
    _save_figure(fig, save_path)

    return fig


# ===================================================================
# Eigenvalue spectrum
# ===================================================================

def plot_eigenvalue_spectrum(
    eigenvalues_by_layer: dict[int, NDArray],
    title: str = "Eigenvalue Spectrum of Representation Covariance",
    top_k: int = 50,
    figsize: tuple[float, float] = (12, 6),
    save_path: str | None = None,
) -> plt.Figure:
    """Plot the top-k eigenvalues of the representation covariance.

    A sharply decaying spectrum (few dominant eigenvalues) indicates
    high anisotropy; a flat spectrum indicates well-distributed
    variance (isotropic representations).

    Args:
        eigenvalues_by_layer: Dictionary mapping layer index to
            1D array of eigenvalues sorted in descending order.
        title: Plot title.
        top_k: Number of top eigenvalues to show.
        figsize: Figure size.
        save_path: Optional save path.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.viridis(
        np.linspace(0.2, 0.8, len(eigenvalues_by_layer))
    )

    for idx, (layer_idx, eigenvalues) in enumerate(
        sorted(eigenvalues_by_layer.items())
    ):
        # Take top-k and normalize to sum to 1 (variance explained).
        top = eigenvalues[:top_k]
        total = eigenvalues.sum()
        if total > 0:
            normalized = top / total
        else:
            normalized = top

        ax.plot(
            range(1, len(normalized) + 1),
            normalized,
            "-",
            color=colors[idx],
            linewidth=2,
            label=f"Layer {layer_idx}",
        )

    ax.set_xlabel("Eigenvalue Rank", fontsize=13)
    ax.set_ylabel("Fraction of Total Variance", fontsize=13)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_figure(fig, save_path)

    return fig
