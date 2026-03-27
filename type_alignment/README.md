# Analysis 2: Semantic vs Syntactic vs Lexical Alignment

## Goal

Identify **which layers encode lexical, syntactic, and semantic representations** in Tiny Aya — and how linguistic abstraction evolves across all 36 layers, from surface word forms to grammar to meaning.

This is a **geometric analysis of representations**, not linear probing. No classifier is trained.

---

## Data

Three controlled English pair datasets, each isolating one linguistic property:

| Type | Example | Measures |
|------|---------|----------|
| **Lexical** | `dog ↔ perro`, `house ↔ casa` | Surface vocabulary alignment |
| **Syntactic** | `The dog chased the cat.` ↔ `The cat was chased by the dog.` | Grammar structure, independent of content |
| **Semantic** | `The meeting was cancelled.` ↔ `The meeting did not happen.` | Concept-level meaning |

---

## Files

| File | What it does |
|------|-------------|
| `inference.py` | Extracts layer-wise hidden states via `MultilingualInference`. Handles batching, pooling, and saving to HDF5. |
| `analyzer.py` | Computes cosine similarity per layer per pair type via `TypeAlignmentAnalyzer`. Produces alignment curves and layer specialization scores. |
| `visualization.py` | Publication-quality plots — alignment curves, signal gap charts, heatmaps, t-SNE/UMAP, retrieval (MRR) curves. |

---

## Pipeline

```
Dataset (lexical / syntactic / semantic pairs)
    ↓
MultilingualInference.extract()         # inference.py
    → layer-wise hidden states
    → sentence embeddings (pooled)
    ↓
TypeAlignmentAnalyzer.compute()         # analyzer.py
    → cosine similarity per layer per pair type
    → alignment curves, signal gaps, dominant layer
    ↓
visualization.py
    → alignment_curves.png
    → signal_gap.png
    → retrieval_mrr_curve.png
```

---

## Pooling Strategy

Two options supported via `ExtractionConfig`:

```python
ExtractionConfig(pooling="mean")   # average over non-padding tokens
ExtractionConfig(pooling="last")   # last non-padding token
```

**Mean pooling** averages all token representations. Transformer hidden states are already contextualised (positional information is baked in via attention), but the effect weakens for syntactic analysis where word order is the key signal. In practice, mean pooling inflates similarity scores across all layers, masking real variation.

**Last-token pooling** uses the final non-padding token, which in decoder-only models accumulates context from the full sequence via causal attention. This is the **recommended strategy** for this analysis.

---

## Findings (English, Tiny Aya, 36 layers)

- **Lexical > Syntactic > Semantic at every layer** — no crossover observed
- Similarity builds steadily from layers 0→33, then collapses at the output layer (35)
- The lexical–semantic gap narrows in deeper layers but never closes
- Tiny Aya never fully transitions from surface-level to semantic representations

---

## Next Steps (WIP)

Pending multilingual dataset:

- Cross-language consistency — `(en↔es, en↔hi, en↔ar)`
- Translation retrieval — MRR per layer
- True lexical pairs — word translations (`dog ↔ perro`)

---

## Results

| File | Description |
|------|-------------|
| `results/alignment_curves_last.png` | Similarity curves across 36 layers — last-token pooling (primary result) |
| `results/alignment_curves_mean.png` | Same curves with mean pooling — shows why it is insufficient |
| `results/alignment_gaps.png` | Lexical–Semantic and Syntactic–Semantic signal gap per layer |
| `results/dominant_alignment_layers.png` | Which alignment type dominates at each layer |
| `results/type_alignment_results.md` | Reference writeup with exact scores and coverage status |

---

## Run

```bash
python -m type_alignment.analyzer
```

This runs the full pipeline end-to-end:
- Loads all 3,000 pairs from `uth/data/linguistic_variation/linguistic_variation.json`
- Runs inference with both mean and last-token pooling
- Saves alignment results to `./outputs/alignment_en/`
- Generates all four plots in `./outputs/alignment_en/`

To cap the number of pairs per type for a quick test, set `MAX_PER_TYPE` at the top of the `__main__` block in `analyzer.py`.

---

## Quick Start

```python
from type_alignment.inference import MultilingualInference, ExtractionConfig
from type_alignment.analyzer import TypeAlignmentAnalyzer, AlignmentPair

pairs = [
    AlignmentPair(
        source="The meeting was cancelled.",
        target="The meeting did not happen.",
        source_lang="en", target_lang="en",
        pair_type="semantic", pair_id=0,
        linguistic_contrast="paraphrase: same concept expressed with different words",
    ),
    ...
]

inferencer = MultilingualInference("CohereLabs/tiny-aya-global")
alignment = TypeAlignmentAnalyzer.from_pairs(
    pairs, inferencer, config=ExtractionConfig(pooling="last")
)
alignment.save("./outputs/alignment_en/")
df = alignment.to_dataframe()  # (num_layers, 4): layer, semantic, syntactic, lexical
```
