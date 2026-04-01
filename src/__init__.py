"""
Tiny Aya Under The Hood (src) — Multilingual Representation Analysis.

This package provides tools and utilities for analyzing how Tiny Aya
(CohereLabs/tiny-aya-global) and its regional variants process
information across languages. By examining layer-wise representations,
we identify where language-agnostic (universal) processing emerges
and where language-specific specialization occurs.

Sub-packages:
    - ``src.utils``: Language registry, metadata, and shared utilities.
    - ``src.data``: Data loading (FLORES-200, translation pipelines).
    - ``src.analysis``: Analysis sub-packages, each named by topic.
    - ``src.analysis.cross_lingual_embedding_alignment``: CKA
      computation, activation hooks, retrieval metrics, clustering,
      visualization.

Quick start::

    from src.utils.languages import Language, LANGUAGE_FAMILIES
    from src.data.flores_loader import load_flores_parallel_corpus
    from src.analysis.cross_lingual_embedding_alignment.hooks import load_model
    from src.analysis.cross_lingual_embedding_alignment.cross_lingual_alignment import (
        CrossLingualAlignmentAnalyzer,
    )
"""

__version__ = "0.1.0"
