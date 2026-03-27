"""
Visualization functions for Semantic vs Syntactic vs Lexical Alignment analysis.

Every function follows a consistent API pattern:
    - Accepts pre-computed data (AlignmentResult, numpy arrays) — no model
      inference happens here.
    - Returns a matplotlib.figure.Figure object for further customization.
    - Optionally saves to a file path if save_path is provided.

Style choices:
    - Font: DejaVu Sans (default matplotlib, widely available).
    - Figure size: Default 10x6 inches for single plots.
    - DPI: 150 for screen display, 300 for saved files.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib
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

def _save_figure(fig: plt.Figure, save_path: Optional[str]) -> None:
    """Save a figure to disk if a path is provided.

    Creates parent directories if they don't exist.

    Args:
        fig: Matplotlib figure to save.
        save_path: File path to save to (e.g., "results/fig.png").
            If None, does nothing.
    """
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
        logger.info("Figure saved to %s", path)


# ===================================================================
# Linguistic Alignment Plots
# ===================================================================

def plot_alignment_curves(
    alignment,
    title: str = "Linguistic Alignment Curves Across Layers",
    y_min: float = 0.0,
    figsize: tuple = (10, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot per-layer cosine similarity curves for semantic, syntactic,
    and lexical sentence pairs.

    Each curve shows how the mean cosine similarity between pairs of a
    given linguistic type evolves across transformer layers. Vertical
    dashed lines mark the crossover points where semantic similarity
    overtakes lexical and syntactic similarity.

    Args:
        alignment: AlignmentResult from TypeAlignmentAnalyzer.compute().
        title: Plot title.
        y_min: Minimum value for the y-axis. Set to e.g. 0.75 when using
            mean pooling to zoom in on the compressed range near 1.0.
        figsize: Figure size in inches (width, height).
        save_path: Optional path to save the figure.

    Returns:
        Matplotlib Figure object.
    """
    layers = list(range(alignment.num_layers))

    fig, ax = plt.subplots(figsize=figsize)

    colors = {"semantic": "#2196F3", "syntactic": "#FF9800", "lexical": "#4CAF50"}
    markers = {"semantic": "o", "syntactic": "s", "lexical": "^"}

    for pair_type, scores_obj in [
        ("semantic", alignment.semantic),
        ("syntactic", alignment.syntactic),
        ("lexical", alignment.lexical),
    ]:
        curve = scores_obj.scores_per_layer
        ax.plot(
            layers,
            curve,
            color=colors[pair_type],
            marker=markers[pair_type],
            linewidth=2,
            markersize=5,
            markevery=2,
            label=f"{pair_type.capitalize()} (peak: layer {scores_obj.peak_layer})",
        )

    # Mark crossover layers.
    if alignment.crossover_semantic_over_lexical is not None:
        ax.axvline(
            alignment.crossover_semantic_over_lexical,
            color=colors["lexical"],
            linestyle="--",
            linewidth=1.2,
            alpha=0.7,
            label=f"Sem > Lex (layer {alignment.crossover_semantic_over_lexical})",
        )
    if alignment.crossover_semantic_over_syntactic is not None:
        ax.axvline(
            alignment.crossover_semantic_over_syntactic,
            color=colors["syntactic"],
            linestyle=":",
            linewidth=1.2,
            alpha=0.7,
            label=f"Sem > Syn (layer {alignment.crossover_semantic_over_syntactic})",
        )

    ax.set_xlabel("Layer", fontsize=13)
    ax.set_ylabel("Mean Cosine Similarity", fontsize=13)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(layers[::2])   # every other layer to avoid crowding
    ax.set_ylim(y_min, 1.05)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_figure(fig, save_path)

    return fig


def plot_dominant_alignment_layers(
    alignment,
    title: str = "Linguistic Signal Scores per Layer",
    figsize: tuple = (12, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot all three alignment scores overlaid per layer, with the
    dominant region shaded.

    Shows semantic, syntactic, and lexical similarity curves on the same
    axes with a shaded band highlighting which signal dominates at each
    layer.

    Args:
        alignment: AlignmentResult from TypeAlignmentAnalyzer.compute().
        title: Plot title.
        figsize: Figure size in inches (width, height).
        save_path: Optional path to save the figure.

    Returns:
        Matplotlib Figure object.
    """
    from matplotlib.patches import Patch

    layers = np.array(range(alignment.num_layers))
    dominant = alignment.dominant_type_per_layer()

    colors = {"semantic": "#2196F3", "syntactic": "#FF9800", "lexical": "#4CAF50"}
    scores = {
        "semantic":  alignment.semantic.scores_per_layer,
        "syntactic": alignment.syntactic.scores_per_layer,
        "lexical":   alignment.lexical.scores_per_layer,
    }

    fig, ax = plt.subplots(figsize=figsize)

    # Shade background by dominant signal.
    for i, dom in enumerate(dominant):
        if dom in colors:
            ax.axvspan(i - 0.5, i + 0.5, color=colors[dom], alpha=0.08, linewidth=0)

    # Plot all three lines.
    markers = {"semantic": "o", "syntactic": "s", "lexical": "^"}
    for pt in ["semantic", "syntactic", "lexical"]:
        ax.plot(
            layers,
            scores[pt],
            color=colors[pt],
            marker=markers[pt],
            linewidth=2,
            markersize=5,
            markevery=2,
            label=pt.capitalize(),
            zorder=3,
        )

    # Dominant signal label at top of each run of the same type.
    abbrev = {"semantic": "Sem", "syntactic": "Syn", "lexical": "Lex"}
    prev = None
    for i, dom in enumerate(dominant):
        if dom != prev and dom in abbrev:
            ax.text(
                i, 1.02, abbrev[dom],
                ha="center", va="bottom",
                fontsize=8, color=colors[dom], fontweight="bold",
                transform=ax.get_xaxis_transform(),
            )
        prev = dom

    # Legend: lines + dominant shading.
    legend_elements = [
        plt.Line2D([0], [0], color=colors["semantic"],  lw=2, marker="o", label="Semantic"),
        plt.Line2D([0], [0], color=colors["syntactic"], lw=2, marker="s", label="Syntactic"),
        plt.Line2D([0], [0], color=colors["lexical"],   lw=2, marker="^", label="Lexical"),
        Patch(facecolor="#888888", alpha=0.15, label="Dominant region"),
    ]
    ax.legend(handles=legend_elements, fontsize=11, loc="lower right")

    ax.set_xlabel("Layer", fontsize=13)
    ax.set_ylabel("Mean Cosine Similarity", fontsize=13)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(layers[::2])
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    _save_figure(fig, save_path)

    return fig


def plot_retrieval_mrr_curve(
    mrr_per_layer: NDArray,
    title: str = "Translation Retrieval MRR Across Layers",
    baseline_label: Optional[str] = None,
    figsize: tuple = (10, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot Mean Reciprocal Rank (MRR) for translation retrieval across layers.

    Higher MRR at a layer means the model's representations at that layer
    can retrieve correct translations by nearest-neighbor search.

    Args:
        mrr_per_layer: Array of shape (num_layers,) from
            TypeAlignmentAnalyzer.compute_retrieval_mrr().
        title: Plot title.
        baseline_label: If provided, draws a horizontal dashed line at
            1/N (random baseline). Pass the number of pairs as an int to
            auto-compute it, or a float to draw at that exact value.
        figsize: Figure size in inches (width, height).
        save_path: Optional path to save the figure.

    Returns:
        Matplotlib Figure object.
    """
    layers = list(range(len(mrr_per_layer)))
    peak_layer = int(np.argmax(mrr_per_layer))

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(
        layers,
        mrr_per_layer,
        color="#E91E63",
        marker="D",
        linewidth=2.5,
        markersize=7,
        label="MRR",
    )

    # Highlight peak.
    ax.axvline(
        peak_layer,
        color="#E91E63",
        linestyle="--",
        linewidth=1.2,
        alpha=0.6,
        label=f"Peak MRR (layer {peak_layer}, {mrr_per_layer[peak_layer]:.3f})",
    )

    # Optional random baseline.
    if baseline_label is not None:
        baseline = (
            1.0 / baseline_label if isinstance(baseline_label, int) else baseline_label
        )
        ax.axhline(
            baseline,
            color="gray",
            linestyle=":",
            linewidth=1.2,
            alpha=0.7,
            label=f"Random baseline ({baseline:.3f})",
        )

    ax.set_xlabel("Layer", fontsize=13)
    ax.set_ylabel("MRR", fontsize=13)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(layers)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_figure(fig, save_path)

    return fig


def plot_alignment_gaps(
    alignment,
    title: str = "Signal Gap Across Layers (Lexical − Semantic, Syntactic − Semantic)",
    figsize: tuple = (12, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot how the gap between alignment signals evolves across layers.

    Shows two curves:
        lexical − semantic   (how much more similar lexical pairs are than semantic)
        syntactic − semantic (how much more similar syntactic pairs are than semantic)

    A narrowing gap means the model is converging toward representing
    all pair types more equally. A flat or widening gap means the model
    never stops preferring surface similarity.

    Args:
        alignment: AlignmentResult from TypeAlignmentAnalyzer.compute().
        title: Plot title.
        figsize: Figure size in inches (width, height).
        save_path: Optional path to save the figure.

    Returns:
        Matplotlib Figure object.
    """
    layers = list(range(alignment.num_layers))

    sem = alignment.semantic.scores_per_layer
    syn = alignment.syntactic.scores_per_layer
    lex = alignment.lexical.scores_per_layer

    gap_lex = lex - sem
    gap_syn = syn - sem

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(layers, gap_lex, color="#4CAF50", linewidth=2.5,
            marker="^", markersize=5, markevery=2,
            label="Lexical − Semantic")
    ax.plot(layers, gap_syn, color="#FF9800", linewidth=2.5,
            marker="s", markersize=5, markevery=2,
            label="Syntactic − Semantic")

    # Zero line — crossover reference.
    ax.axhline(0, color="black", linewidth=1.0, linestyle="--", alpha=0.5,
               label="No gap (signals equal)")

    # Annotate crossover layers if they exist.
    if alignment.crossover_semantic_over_lexical is not None:
        cl = alignment.crossover_semantic_over_lexical
        ax.axvline(cl, color="#4CAF50", linewidth=1.2, linestyle=":",
                   alpha=0.7, label=f"Lex−Sem crossover (layer {cl})")
    if alignment.crossover_semantic_over_syntactic is not None:
        cs = alignment.crossover_semantic_over_syntactic
        ax.axvline(cs, color="#FF9800", linewidth=1.2, linestyle=":",
                   alpha=0.7, label=f"Syn−Sem crossover (layer {cs})")

    ax.set_xlabel("Layer", fontsize=13)
    ax.set_ylabel("Score Gap", fontsize=13)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(layers[::2])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_figure(fig, save_path)

    return fig


def plot_cross_language_variance(
    consistency_result,
    title: str = "Cross-Language Consistency Across Layers",
    figsize: tuple = (10, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot cross-language similarity variance across layers.

    Shows two panels:
        Top — per-language-pair similarity curves (one line per pair).
        Bottom — variance across pairs per layer (the primary signal).

    Decreasing variance in deeper layers indicates the model is becoming
    more language-agnostic. High variance in early layers indicates
    language-specific surface representations.

    Args:
        consistency_result: CrossLanguageConsistencyResult from
            TypeAlignmentAnalyzer.compute_cross_language_consistency().
        title: Super-title for the figure.
        figsize: Figure size in inches (width, height).
        save_path: Optional path to save the figure.

    Returns:
        Matplotlib Figure object.
    """
    layers = list(range(consistency_result.num_layers))
    palette = sns.color_palette("tab10", len(consistency_result.language_pairs))

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=figsize,
                                          gridspec_kw={"height_ratios": [2, 1]})

    # Top: per-language-pair similarity curves.
    for i, lp in enumerate(consistency_result.language_pairs):
        label = f"{lp[0]}↔{lp[1]}"
        ax_top.plot(
            layers,
            consistency_result.similarity_per_lang_pair[i],
            color=palette[i],
            marker="o",
            linewidth=2,
            markersize=6,
            label=label,
        )
    ax_top.set_ylabel("Mean Cosine Similarity", fontsize=12)
    ax_top.set_xticks(layers)
    ax_top.legend(fontsize=10)
    ax_top.grid(True, alpha=0.3)
    ax_top.set_title(title, fontsize=14, fontweight="bold")

    # Bottom: variance across language pairs per layer.
    ax_bot.bar(
        layers,
        consistency_result.variance_per_layer,
        color="#9C27B0",
        alpha=0.75,
        edgecolor="white",
    )
    ax_bot.set_xlabel("Layer", fontsize=12)
    ax_bot.set_ylabel("Variance", fontsize=12)
    ax_bot.set_xticks(layers)
    ax_bot.grid(True, alpha=0.2, axis="y")

    plt.tight_layout()
    _save_figure(fig, save_path)

    return fig
