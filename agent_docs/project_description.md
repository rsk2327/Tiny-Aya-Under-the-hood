# Tiny Aya Under The Hood -- Project Description

This project investigates how Tiny Aya processes information across languages by analyzing how representations evolve across model layers. By comparing layer-wise representations in Tiny Aya Global and its regional variants, we aim to understand where language-agnostic (universal) processing emerges and where region-specific specialization occurs.

> **Central question:** Which parts of Tiny Aya's network learn language-agnostic representations, and which parts become specialized for specific languages or regions?

## Model

- **Tiny Aya Global** (`CohereLabs/tiny-aya-global`): 3.35B parameters, 36 transformer layers (3 sliding-window + 1 global attention, repeated 9 times), hidden dimension 3072, 70+ languages.
- **Regional variants**: Earth (West Asia + Africa), Fire (South Asia), Water (Europe + Asia Pacific) -- built via model merging with Global.

## Current Status

All 9 analysis notebooks have been executed. A paper skeleton in NeurIPS/ICML format and a detailed notebook-by-notebook appendix are in `analysis/cross_lingual_embedding_alignment/paperback.md`.

## Key Findings

1. **No convergence layer found**: Cross-lingual CKA peaks at Layer 0 (avg 0.6518) and declines monotonically. The 0.75 threshold is never reached at any of 36 layers.
2. **Family and script do not predict structure**: ARI = -0.2340 (worse than random) at all layers; script gap is negative (-0.07) throughout. Neither genetic family nor writing system organizes representations.
3. **Anisotropy masks near-perfect alignment**: Whitened CKA ≥ 0.999 from Layer 1 onward. The moderate standard CKA (~0.65) significantly understates true cross-lingual similarity.
4. **Geometry--function disconnect**: CKA peaks at Layer 0, MRR peaks at Layer 35 -- opposing trends. Bengali has near-perfect whitened CKA with English but MRR = 0.009 (54x worse than German at 0.493).
5. **Regional geometry preserved**: Delta-CKA ≈ 0.0001 across all model variants. Model merging preserves cross-lingual structure perfectly.
6. **Fire model paradox**: Fire achieves best retrieval MRR for 9/12 languages (not just its 3 South Asian targets). Only Fire shows preferential target drift (3.38x ratio).

## Methodology (Implemented)

### Stage 1: Parallel Dataset Construction (Notebook 01) -- COMPLETE
- FLORES+ `devtest` split: 1,012 professionally translated sentences across 13 languages.
- 13 languages spanning 5 families, 6 scripts, 3 resource tiers.
- Corpus statistics and tokenizer fertility analysis.

### Stage 2: Layer-wise Hidden State Extraction (Notebook 02) -- COMPLETE
- Forward hooks (`ActivationStore`) capture hidden states at all 36 layers.
- Mean pooling over non-padding tokens: `(1012, 3072)` per language per layer.
- 4 model variants extracted: Global, Earth, Fire, Water.

### Stage 3: Cross-Lingual Similarity Measurement (Notebooks 03, 05) -- COMPLETE
- Linear CKA + RBF CKA with permutation tests (Notebook 03).
- ZCA-whitened CKA for anisotropy correction (Notebook 05).
- Full `(13, 13, 36)` similarity tensors computed.

### Stage 4: Convergence Detection and Grouping Analysis (Notebooks 04, 06, 07) -- COMPLETE
- Hierarchical clustering + ARI + cophenetic correlation (Notebook 04).
- Translation retrieval MRR and Recall@k per language per layer (Notebook 06).
- Intra- vs inter-script CKA decomposition (Notebook 07).

### Stage 5: Cross-Model Comparative Analysis (Notebooks 08, 09) -- COMPLETE
- Delta-CKA between Global and regional variants (Notebook 08).
- Per-language representational drift + retrieval MRR comparison (Notebook 09).
- Drift-MRR correlation: Fire r = +0.567 (purposeful drift), Earth r = -0.461 (incidental).

### Stage 6: Activation Intervention Framework -- NOT YET IMPLEMENTED
- Forward hooks for controlled modification of hidden states.
- Activation patching (Dumas et al., 2025 style) to provide causal evidence.
- This is listed as a high-priority future work item in paperback.md.

## Notebook Index

| # | Notebook | Key Result |
|---|---|---|
| 01 | Data Preparation | 13 languages x 1,012 sentences loaded and validated |
| 02 | Activation Extraction | 36-layer activations extracted for all 4 model variants |
| 03 | Cross-Lingual CKA | No convergence layer; CKA peaks at Layer 0 (0.6518), declines to 0.399 |
| 04 | Language Family Clustering | ARI = -0.2340 constant; family gap negative; no dissolution |
| 05 | Anisotropy & Whitened CKA | Whitened CKA >= 0.999 from Layer 1; anisotropy suppresses alignment |
| 06 | Retrieval Alignment | MRR peaks at deepest layer; 54x gap German vs Bengali |
| 07 | Script Decomposition | Script gap negative (-0.07) and constant; BPE hypothesis contradicted |
| 08 | Regional Comparison | Delta-CKA ~ 0.0001; all models geometrically equivalent |
| 09 | Cross-Model Drift | Fire best for 9/12 languages; 3.38x preferential target drift |

## File Locations

- **Notebooks**: `analysis/cross_lingual_embedding_alignment/01-09_*.ipynb`
- **Writeup**: `analysis/cross_lingual_embedding_alignment/paperback.md`
- **Source code**: `src/analysis/cross_lingual_embedding_alignment/`
- **Results**: `analysis/results/cross_lingual/{activations,cka_matrices,metrics,figures}/`
- **Tests**: `tests/test_{cka,hooks,languages,retrieval}.py` (71 tests)
