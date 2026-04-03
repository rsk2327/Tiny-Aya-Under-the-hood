"""
Cross-lingual embedding alignment analysis package.

This sub-package implements the full pipeline for analyzing where
language-agnostic representations emerge in Tiny Aya's transformer
layers. It provides:

    - **CKA metrics** (``cka``): Centered Kernel Alignment for
      measuring representational similarity between language pairs
      at each layer, including linear, RBF, mini-batch, and
      anisotropy-corrected (whitened) variants.

    - **Activation hooks** (``hooks``): Forward hook utilities for
      extracting layer-wise hidden states from HuggingFace transformer
      models with mean-pooling over non-padding tokens.

    - **Cross-lingual analyzer** (``cross_lingual_alignment``): The
      main orchestrator class that coordinates activation extraction,
      CKA computation, convergence detection, and result serialization.

    - **Retrieval metrics** (``retrieval_metrics``): Translation
      retrieval scoring (MRR, Recall@k) that measures functional
      alignment — whether geometric similarity translates to task
      utility.

    - **Clustering utilities** (``clustering``): Hierarchical
      clustering and language family dissolution analysis across
      layers.

    - **Visualization** (``visualization``): Publication-quality
      plotting functions for heatmaps, convergence curves,
      dendrograms, and script decomposition charts.

Typical usage::

    from src.analysis.cross_lingual_embedding_alignment.cross_lingual_alignment import (
        CrossLingualAlignmentAnalyzer,
    )
    from src.analysis.cross_lingual_embedding_alignment.cka import linear_cka, rbf_cka

All modules are designed to be used both programmatically and from
Jupyter notebooks. Each function includes comprehensive input
validation and informative error messages.
"""

from src.analysis.cross_lingual_embedding_alignment.cka import (
    CKAHeatmapData,
    MinibatchCKAAccumulator,
    cka_permutation_test,
    compute_layerwise_cka,
    linear_cka,
    minibatch_cka,
    rbf_cka,
    whitened_cka,
)

__all__ = [
    "CKAHeatmapData",
    "MinibatchCKAAccumulator",
    "cka_permutation_test",
    "compute_layerwise_cka",
    "linear_cka",
    "minibatch_cka",
    "rbf_cka",
    "whitened_cka",
]
