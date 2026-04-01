# Tiny Aya Under The Hood

A mechanistic interpretability study of multilingual representation emergence in [Tiny Aya Global](https://huggingface.co/CohereLabs/tiny-aya-global) (3.35B parameters, 36 transformer layers). We investigate how representations evolve across the model's layers to identify where language-agnostic (universal) processing emerges and where region-specific specialization occurs.

> **Central question:** Which parts of Tiny Aya's network learn language-agnostic representations, and which parts become specialized for specific languages or regions?

---

## Table of Contents

- [Motivation](#motivation)
- [Research Questions](#research-questions)
- [Key Findings (Expected)](#key-findings-expected)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Environment Variables](#environment-variables)
- [Project Structure](#project-structure)
- [Source Package Reference](#source-package-reference)
- [Datasets](#datasets)
- [Analysis 1: Cross-Lingual Embedding Alignment](#analysis-1-cross-lingual-embedding-alignment)
- [Languages](#languages)
- [Model Details](#model-details)
- [Methodology](#methodology)
- [Development](#development)
- [Contributing](#contributing)
- [References](#references)
- [License](#license)

---

## Motivation

Recent mechanistic interpretability research (Wendler et al., 2024; Dumas et al., 2025; Harrasse et al., 2025) has converged on a **three-stage hypothesis** for how multilingual LLMs process text:

1. **Language-specific encoding** (early layers) -- surface features like script and morphology dominate.
2. **Language-agnostic processing** (middle layers) -- representations converge into a shared semantic space.
3. **Language-specific decoding** (late layers) -- the model maps back to target-language token predictions.

This project tests whether Tiny Aya -- a compact 3.35B-parameter model with 36 transformer layers (3 sliding-window attention + 1 global attention, repeated 9 times) -- exhibits the same three-stage pattern or compresses it differently. Understanding where these transitions happen enables:

- Targeted representation steering instead of full fine-tuning
- Efficient parameter sharing across regional model variants (Global, South Asia, Africa)
- Diagnosing where multilingual failures originate (early vs. late layers)
- Informed adapter placement and pruning decisions
- Improved interpretability of global vs. regional model behavior

This work builds on the [Wayy-Research/project-aya](https://github.com/Wayy-Research/project-aya) notebooks (specifically `02_cka_analysis` and `07_teacher_representations`) and extends them from self-similarity analysis to cross-lingual similarity measurement.

## Research Questions

From the [project specification](agent_docs/project_description.md):

1. **Where do language-agnostic representations emerge?** At which layer does cross-lingual CKA stabilize above the 0.75 threshold?
2. **Do language family clusters dissolve?** Does hierarchical clustering at deeper layers stop grouping by genetic family?
3. **Is alignment genuine or an anisotropy artifact?** Does whitened (ZCA-corrected) CKA confirm the same convergence pattern?
4. **Does geometric alignment translate to functional utility?** Do translation retrieval metrics (MRR, Recall@k) track CKA?
5. **Is alignment script-driven or semantic?** Does intra-script CKA dominate, or do different-script languages also converge?
6. **Where does regional specialization occur?** How does delta-CKA between Global and regional variants behave across layers?

## Key Findings

All 9 notebooks have been executed. The results challenge the three-stage hypothesis:

| Finding | Method | Notebook | Result |
|---|---|---|---|
| No convergence layer found | Linear CKA + permutation tests | 03 | CKA peaks at Layer 0 (0.6518) and *declines* monotonically; 0.75 threshold never reached |
| Family clusters were never present | Hierarchical clustering + ARI | 04 | ARI = -0.2340 (worse than random) at all layers; family gap negative throughout |
| Anisotropy *suppresses* alignment | Whitened CKA comparison | 05 | Whitened CKA ≥ 0.999 from Layer 1 onward; standard CKA understates true alignment |
| Geometry--function disconnect | MRR / Recall@k retrieval | 06 | MRR peaks at deepest layer (opposite of CKA); 54× gap between German and Bengali |
| Script gap is negative | Intra- vs inter-script CKA | 07 | Different-script languages are *more* similar (-0.07 gap); contradicts BPE hypothesis |
| Regional geometry preserved | Delta-CKA (Global vs Regional) | 08 | Delta-CKA ≈ 0.0001; model merging preserves cross-lingual structure perfectly |
| Fire model dominates universally | Cross-model drift + retrieval | 09 | Fire (South Asia) achieves best MRR for 9/12 languages; 3.38× preferential target drift |

---

## Quick Start

```bash
git clone https://github.com/Vidit-Ostwal/Tiny-Aya-Under-the-hood.git
cd Tiny-Aya-Under-the-hood
uv sync

# Run tests (no GPU needed)
uv run pytest tests/ -v

# Run linting
uv run ruff check src/ tests/

# Run the analysis notebooks (GPU required for 02 and 08)
uv run jupyter lab analysis/cross_lingual_embedding_alignment/
```

## Installation

### Prerequisites

| Requirement | Minimum | Recommended |
|---|---|---|
| Python | 3.12+ | 3.12 |
| [uv](https://docs.astral.sh/uv/) | latest | latest |
| CUDA GPU (VRAM) | 2 GB (4-bit mode) | 8 GB (fp16 mode) |
| Disk space | ~5 GB | ~10 GB (with cached datasets) |

### Steps

```bash
# 1. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone the repository
git clone https://github.com/Vidit-Ostwal/Tiny-Aya-Under-the-hood.git
cd Tiny-Aya-Under-the-hood

# 3. Create virtual environment and install all dependencies
uv sync
```

All dependencies (PyTorch, Transformers, Jupyter, pytest, ruff, scikit-learn, scipy, seaborn, bitsandbytes, etc.) are installed in a single command. There are no separate optional groups.

## Environment Variables

Create a `.env` file in the project root:

```bash
# .env (required)
HF_TOKEN=hf_your_token_here
```

**How to get your HuggingFace token:**

1. Create a HuggingFace account at [huggingface.co/join](https://huggingface.co/join)
2. Accept the FLORES+ dataset terms at [openlanguagedata/flores_plus](https://huggingface.co/datasets/openlanguagedata/flores_plus)
3. Create an access token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

The `python-dotenv` library loads `.env` automatically from `src/data/flores_loader.py` by walking up from the file to the project root.

**Optional keys** (for data generation pipelines):

```bash
OPENAI_API_KEY=sk-...         # For translate_data_openai.py
COHERE_API_KEY=...            # For linguistic_variation generation
```

> **Security:** `.env` is in `.gitignore` and will never be committed. Never hardcode tokens in source files.

---

## Project Structure

```
.
├── pyproject.toml                  # All dependencies, ruff config, pytest config, hatch build
├── uv.lock                        # Locked dependency versions
├── README.md
├── CONTRIBUTING.md                 # Full contributor guide, PR checklist
├── AGENTS.md                      # AI agent-specific guidelines
├── .env                            # HuggingFace token (gitignored)
├── .gitignore
│
├── src/                            # Main Python package (import as `from src.xxx`)
│   ├── __init__.py                 # Package docstring, __version__ = "0.1.0"
│   ├── utils/
│   │   ├── __init__.py             # Re-exports Language, LANGUAGE_FAMILIES, SCRIPT_GROUPS, etc.
│   │   └── languages.py            # Language enum (13 languages) with LanguageInfo dataclass
│   ├── data/
│   │   ├── __init__.py
│   │   ├── flores_loader.py        # FLORES+ parallel corpus loader (HF datasets API)
│   │   ├── translate_data_openai.py  # OpenAI structured-output translation pipeline
│   │   └── linguistic_variation/   # Dataset 2: linguistic variation pairs (@danielmargento)
│   │       ├── generate_linguistic_variation.py  # Cohere API generation (lexical/syntactic/semantic)
│   │       ├── review_linguistic_variation.py    # Cohere API quality review
│   │       ├── dedup_dataset.py                  # Deduplication utility
│   │       └── linguistic_variation.json         # Generated dataset (~18,000 pairs)
│   └── analysis/                   # Analysis sub-packages (one per topic)
│       ├── __init__.py
│       └── cross_lingual_embedding_alignment/
│           ├── __init__.py         # Re-exports CKA functions, MinibatchCKAAccumulator, etc.
│           ├── cka.py              # 4 CKA variants + permutation tests + CKAHeatmapData
│           ├── hooks.py            # ActivationStore, register_model_hooks, load_model
│           ├── cross_lingual_alignment.py  # CrossLingualAlignmentAnalyzer orchestrator
│           ├── retrieval_metrics.py  # MRR, Recall@k, cosine similarity, confusion matrix
│           ├── clustering.py       # Hierarchical clustering, family/script dissolution, ARI
│           └── visualization.py    # 12 publication-quality plotting functions
│
├── analysis/                       # Analysis notebooks, writeups, and results (one subdir per topic)
│   ├── cross_lingual_embedding_alignment/  # Cross-lingual embedding alignment
│   │   ├── paperback.md            # Full writeup with math formulations and references
│   │   ├── 01_data_preparation.ipynb
│   │   ├── 02_activation_extraction.ipynb
│   │   ├── 03_cross_lingual_cka.ipynb
│   │   ├── 04_language_family_clustering.ipynb
│   │   ├── 05_anisotropy_whitened_cka.ipynb
│   │   ├── 06_retrieval_alignment.ipynb
│   │   ├── 07_script_decomposition.ipynb
│   │   ├── 08_regional_comparison.ipynb
│   │   └── 09_cross_model_drift.ipynb  # Cross-model drift + retrieval MRR comparison
│   └── results/                    # Generated artifacts (from notebook execution)
│       └── cross_lingual/
│           ├── activations/        # Mean-pooled sentence embeddings (.pt)
│           ├── cka_matrices/       # Pairwise CKA similarity matrices (.npy)
│           ├── metrics/            # Convergence curves, clustering metrics (.json)
│           └── figures/            # All generated plots (.png)
│
├── data/                           # Raw data files (CSVs -- not Python code)
│   ├── test_data.csv               # 10 sample English sentences for translation pipeline
│   └── test_data_translation_output.csv  # Sample translations (Hindi, Bengali, Tamil, Arabic)
│
├── tests/                          # Unit tests (71 tests, CPU-only, no GPU/network)
│   ├── __init__.py
│   ├── test_cka.py                 # 27 tests: all CKA variants, edge cases, permutation tests
│   ├── test_hooks.py               # 16 tests: ActivationStore, hook lifecycle, mock models
│   ├── test_languages.py           # 17 tests: Language enum, metadata, groupings, lookups
│   └── test_retrieval.py           # 11 tests: MRR, Recall@k, cosine similarity, validation
│
└── agent_docs/                     # Internal research specification
    └── project_description.md      # Original 5-stage methodology
```

### Naming Convention for Analysis Topics

The repository supports multiple contributors working on independent analyses. Each topic gets:
- A **Python package** under `src/<topic_name>/` for shared analysis code
- A **notebooks directory** under `analysis/<topic_name>/` for Jupyter notebooks and writeups

| Python Package | Notebooks Directory | Focus | Status |
|---|---|---|---|
| `src/analysis/cross_lingual_embedding_alignment/` | `analysis/cross_lingual_embedding_alignment/` | Cross-lingual embedding alignment (CKA, clustering, retrieval, drift) | 9 notebooks + writeup |
| `src/<next_topic>/` | `analysis/<next_topic>/` | *(available for next contributor)* | -- |

---

## Source Package Reference

### `src.utils.languages` -- Language Registry

The canonical registry of 13 languages. Each `Language` enum member wraps a frozen `LanguageInfo` dataclass with six fields: `name`, `iso_code`, `flores_code`, `script`, `family`, `resource_level`.

```python
from src.utils.languages import Language, LANGUAGE_FAMILIES, SCRIPT_GROUPS

hindi = Language.HINDI
print(hindi.flores_code)     # "hin_Deva"
print(hindi.family)          # "Indo-European"
print(hindi.resource_level)  # "mid"

for lang in LANGUAGE_FAMILIES["Indo-European"]:
    print(lang.lang_name)    # english, spanish, french, german, hindi, bengali, persian
```

| Export | Type | Description |
|---|---|---|
| `Language` | `Enum` | 13 members with `.lang_name`, `.iso_code`, `.flores_code`, `.script`, `.family`, `.resource_level` |
| `LanguageInfo` | `dataclass` | Frozen metadata container for a single language |
| `LANGUAGE_FAMILIES` | `dict[str, list[Language]]` | Grouped by genetic family (5 families) |
| `SCRIPT_GROUPS` | `dict[str, list[Language]]` | Grouped by writing script (6 scripts) |
| `RESOURCE_GROUPS` | `dict[str, list[Language]]` | Grouped by data availability tier (high/mid/low) |
| `get_language_by_iso(code)` | `function` | Lookup by ISO 639-1 code, returns `Language` or `None` |
| `get_language_by_name(name)` | `function` | Lookup by lowercase name, returns `Language` or `None` |
| `get_all_flores_codes()` | `function` | Returns `{name: flores_code}` dict for all 13 |

### `src.data.flores_loader` -- FLORES+ Corpus

Loads the FLORES+ parallel corpus (1,012 professionally translated sentences, `devtest` split) from HuggingFace. Requires `HF_TOKEN`.

```python
from src.data.flores_loader import load_flores_parallel_corpus, get_corpus_statistics

corpus = load_flores_parallel_corpus()               # All 13 languages, 1012 sentences each
corpus = load_flores_parallel_corpus(max_sentences=100)  # Quick test subset
corpus = load_flores_parallel_corpus(languages=[Language.ENGLISH, Language.HINDI])

stats = get_corpus_statistics(corpus)
print(stats["english"]["avg_word_count"])  # ~18.5
```

### `src.data.translate_data_openai` -- OpenAI Translation Pipeline

Translates sentences from CSV files to multiple target languages using OpenAI's structured output API (GPT-4.1). Uses Pydantic models (`TranslationItem`, `TranslationBatch`) for type-safe API responses.

```python
from src.data.translate_data_openai import TranslationPipeline

pipeline = TranslationPipeline(model="gpt-4.1", batch_size=10)
results = pipeline.translate_file(
    input_file="data/test_data.csv",
    target_languages=[Language.HINDI, Language.ARABIC],
    output_file="data/translations.csv",
)
```

### `src.data.linguistic_variation` -- Dataset 2 (by @danielmargento)

Generates controlled sentence pairs testing three types of linguistic variation using Cohere's API:

| Type | What It Tests | Example |
|---|---|---|
| **Lexical** | Single-word synonym swap | "She *purchased* a jacket" vs "She *bought* a jacket" |
| **Syntactic** | Grammatical transformation (same meaning) | "She baked the cake" vs "The cake was baked by her" |
| **Semantic** | Full paraphrase (different structure + words) | "He started running" vs "He broke into a run" |

**Files:**
- `generate_linguistic_variation.py` -- Main generation pipeline: batched Cohere API calls with deduplication, produces `linguistic_variation.json`
- `review_linguistic_variation.py` -- Quality review: scores each pair against type-specific criteria using Cohere
- `dedup_dataset.py` -- Post-hoc deduplication: removes exact sentence duplicates and reused word swaps
- `linguistic_variation.json` -- The generated dataset (~18,000 pairs across all three types)

### `src.analysis.cross_lingual_embedding_alignment.cka` -- Centered Kernel Alignment

Four CKA variants for measuring representational similarity, plus statistical testing:

| Function | Kernel | Complexity | Use Case |
|---|---|---|---|
| `linear_cka(X, Y)` | `X @ X.T` | O(n * d^2) | Fast, linear relationships |
| `rbf_cka(X, Y, sigma_x, sigma_y)` | Gaussian | O(n^2 * d) | Nonlinear, more expressive |
| `whitened_cka(X, Y, regularization)` | Linear + ZCA whitening | O(d^3) | Anisotropy-corrected |
| `minibatch_cka(X, Y, batch_size)` | Linear (streaming) | O(batch * d^2) | Memory-efficient |

All functions accept PyTorch tensors of shape `(n_samples, d_features)` with strict input validation (NaN/Inf checks, dimension matching). Internal helpers `_validate_activation_pair`, `_center_gram`, `_hsic`, `_rbf_kernel`, `_whiten_representations` handle the math.

```python
from src.analysis.cross_lingual_embedding_alignment.cka import linear_cka, cka_permutation_test, MinibatchCKAAccumulator

score = linear_cka(X, Y)  # Returns tensor scalar in [0, 1]

result = cka_permutation_test(X, Y, n_permutations=1000)
# {"observed_cka": 0.85, "p_value": 0.001, "null_mean": 0.12, "is_significant": True}

acc = MinibatchCKAAccumulator(d_x=3072, d_y=3072)
for batch_x, batch_y in loader:
    acc.update(batch_x, batch_y)
score = acc.compute()
```

**Additional exports:** `CKAHeatmapData` (dataclass for serializable heatmap data), `compute_layerwise_cka` (all-pairs CKA between two sets of layer activations).

### `src.analysis.cross_lingual_embedding_alignment.hooks` -- Activation Extraction

Forward-hook-based extraction from HuggingFace models. `ActivationStore` collects hidden states from registered hooks, supports both token-level and sentence-level (mean-pooled) extraction.

```python
from src.analysis.cross_lingual_embedding_alignment.hooks import load_model, ActivationStore, register_model_hooks

model, tokenizer = load_model(precision="fp16")  # or "4bit" for low VRAM
store = ActivationStore(detach=True, device="cpu")
register_model_hooks(model, store, layer_indices=[0, 1, 2, 3])

with torch.no_grad():
    inputs = tokenizer(sentences, return_tensors="pt", padding=True).to("cuda")
    store.store_attention_mask(inputs["attention_mask"])
    model(**inputs)

activations = store.collect_mean_pooled()  # {"layer_0": (n, 3072), "layer_1": ...}
store.remove_hooks()
```

**Supported architectures:** Cohere/Llama/Mistral (`model.model.layers`), GPT-2/GPT-Neo (`model.transformer.h`), direct models (`model.layers`). Auto-detected by `_find_transformer_layers`.

### `src.analysis.cross_lingual_embedding_alignment.cross_lingual_alignment` -- Orchestrator

`CrossLingualAlignmentAnalyzer` ties together all analysis steps into a single class managing the full pipeline:

```python
from src.analysis.cross_lingual_embedding_alignment.cross_lingual_alignment import CrossLingualAlignmentAnalyzer

analyzer = CrossLingualAlignmentAnalyzer(
    model=model, tokenizer=tokenizer,
    parallel_corpus=corpus, batch_size=8, device="cuda",
)

# Step 1: Extract activations (GPU-intensive, cached)
analyzer.extract_all_activations()

# Step 2: Compute CKA matrices (CPU-only after extraction)
linear_matrices = analyzer.compute_cka_matrices(kernel="linear")
whitened_matrices = analyzer.compute_cka_matrices(kernel="whitened")

# Step 3: Convergence detection
curve = analyzer.compute_convergence_curve()  # avg_cka, std, CI per layer
layer = analyzer.find_convergence_layer(threshold=0.75)

# Step 4: Retrieval scoring
retrieval = analyzer.compute_retrieval_scores(source_lang="english")

# Step 5: Clustering analysis
clustering = analyzer.compute_clustering_analysis()

# Step 6: Save everything with standardized keys
analyzer.save_results("analysis/results/cross_lingual/")
```

The analyzer caches activations and CKA matrices as instance attributes. Results are saved with `layer_{idx}_{metric}` naming convention for reproducibility. Activations can be loaded from disk via `load_activations()` to skip the expensive GPU step.

### `src.analysis.cross_lingual_embedding_alignment.retrieval_metrics` -- Translation Retrieval

Measures **functional alignment** -- whether geometric similarity in the embedding space actually enables finding translations via nearest-neighbor search.

| Function | Returns | Description |
|---|---|---|
| `compute_cosine_similarity_matrix(src, tgt)` | `(n, n)` matrix | Pairwise cosine similarity |
| `compute_mrr(src, tgt)` | `float` | Mean Reciprocal Rank: avg of 1/rank |
| `compute_recall_at_k(src, tgt, k)` | `float` | Fraction where correct translation in top-k |
| `compute_all_retrieval_metrics(src, tgt, k_values)` | `dict` | MRR + Recall@1/5/10 + rank stats |
| `compute_confusion_matrix(src, targets_by_lang)` | `ndarray` | Which language each query's nearest neighbor belongs to |

All functions accept numpy arrays (CPU-bound computation). Input validation checks for 2D shape, matching dimensions, NaN/Inf, and empty arrays.

### `src.analysis.cross_lingual_embedding_alignment.clustering` -- Language Family Analysis

Hierarchical clustering on CKA matrices with metrics tracking how family/script groupings evolve across layers.

| Function | Returns | Description |
|---|---|---|
| `compute_hierarchical_clustering(sim, names, method)` | `dict` | Ward's linkage, cophenetic correlation, distance matrix |
| `compute_cluster_assignments(linkage, n_clusters)` | `ndarray` | Flat cluster labels from linkage |
| `compute_family_dissolution_metrics(sim, names, langs)` | `dict` | Intra/inter-family CKA, family gap, ARI, per-family averages |
| `compute_script_group_metrics(sim, names, langs)` | `dict` | Intra/inter-script CKA, script gap, per-script averages |

**Family dissolution:** A large gap between intra-family and inter-family CKA means the model still differentiates families. A gap near zero means language-agnostic representations. ARI compares discovered clusters against known family labels.

### `src.analysis.cross_lingual_embedding_alignment.visualization` -- Publication-Quality Plots

12 plotting functions, all returning `matplotlib.Figure` with optional `save_path`:

| Function | Plot Type |
|---|---|
| `plot_cka_heatmap(matrix, names, layer)` | Single-layer language-pair CKA heatmap |
| `plot_multi_layer_heatmaps(matrices, names)` | Side-by-side heatmaps across layers |
| `plot_convergence_curve(layers, avg_cka, ci, threshold)` | Avg CKA vs layer with CI bands and threshold line |
| `plot_language_pair_trajectories(layers, trajectories)` | Per-pair CKA spaghetti plot with optional highlighting |
| `plot_dendrogram(linkage, names, layer)` | Hierarchical clustering tree |
| `plot_dendrograms_across_layers(results)` | Side-by-side dendrograms |
| `plot_retrieval_curves(layers, mrr_per_layer)` | MRR vs layer per language |
| `plot_recall_bars(scores, k, layer)` | Recall@k bar chart per language |
| `plot_script_decomposition(layers, intra, inter)` | Intra- vs inter-script CKA with gap shading |
| `plot_family_gap_curve(layers, intra, inter)` | Intra- vs inter-family CKA with gap shading |
| `plot_anisotropy_heatmap(scores, names, layers)` | Language x layer anisotropy matrix |
| `plot_eigenvalue_spectrum(eigenvalues_by_layer, top_k)` | Covariance eigenvalue decay (log scale) |

Style: seaborn `whitegrid` theme, `RdBu_r` colormap for CKA, 150 DPI screen / 300 DPI save.

---

## Datasets

### Dataset 1: FLORES+ Parallel Corpus

| Property | Value |
|---|---|
| Source | [openlanguagedata/flores_plus](https://huggingface.co/datasets/openlanguagedata/flores_plus) (HuggingFace) |
| Split | `devtest` (1,012 sentences) |
| Languages | 228+ varieties; we use 13 |
| Purpose | Semantically aligned sentences for controlled cross-lingual comparison |
| Loaded by | `src.data.flores_loader.load_flores_parallel_corpus()` |
| Authentication | Requires `HF_TOKEN` (gated dataset) |

### Dataset 2: Linguistic Variation Pairs

| Property | Value |
|---|---|
| Source | Generated via Cohere API (`src/data/linguistic_variation/`) |
| Size | ~18,000 sentence pairs |
| Types | Lexical substitution, syntactic transformation, semantic paraphrase |
| Purpose | Probing model sensitivity to controlled linguistic changes |
| Contributor | @danielmargento (PRs #1, #2) |

### Dataset 3: OpenAI Translations (Sample)

| Property | Value |
|---|---|
| Source | Generated via OpenAI GPT-4.1 (`src/data/translate_data_openai.py`) |
| Files | `data/test_data.csv` (10 English sentences), `data/test_data_translation_output.csv` (translations) |
| Purpose | Supplementary parallel data generation pipeline |

---

## Analysis 1: Cross-Lingual Embedding Alignment

**Goal:** Identify layers where language-agnostic representations emerge by measuring representational similarity across 13 languages at each of Tiny Aya's 36 transformer layers.

**Full writeup:** See [`analysis/cross_lingual_embedding_alignment/paperback.md`](analysis/cross_lingual_embedding_alignment/paperback.md) for mathematical formulations, mechanistic interpretability context, and literature connections.

### Notebook Pipeline

| # | Notebook | Description | GPU | Depends On |
|---|---|---|---|---|
| 01 | Data Preparation | Load FLORES+, corpus statistics, language metadata, tokenizer fertility | No | -- |
| 02 | Activation Extraction | Extract mean-pooled hidden states from all 36 layers for 13 languages | **Yes** | 01 |
| 03 | Cross-Lingual CKA | Core linear CKA matrices, convergence curve, permutation tests | No | 02 |
| 04 | Language Family Clustering | Ward's method dendrograms, family dissolution, ARI tracking | No | 03 |
| 05 | Anisotropy & Whitened CKA | Anisotropy measurement, ZCA whitening, eigenvalue spectra | No | 02 |
| 06 | Retrieval Alignment | Translation retrieval MRR, Recall@1/5/10 per language per layer | No | 02 |
| 07 | Script Decomposition | Intra- vs inter-script CKA, Latin-script deep dive | No | 03 |
| 08 | Regional Comparison | Delta-CKA between Global and regional Tiny Aya variants | **Yes** | 01 |
| 09 | Cross-Model Drift | Per-language representational drift, retrieval MRR comparison, drift-MRR correlation | No | 08 |

### Six Novel Techniques

| # | Technique | Research Question |
|---|---|---|
| 1 | Language Family Clustering + ARI | Do genetic family groupings dissolve in deeper layers? |
| 2 | Anisotropy-Corrected (Whitened) CKA | Is observed alignment genuine or an anisotropy artifact? |
| 3 | Parallel Sentence Retrieval (MRR/Recall@k) | Does geometric similarity translate to functional utility? |
| 4 | Script-Based CKA Decomposition | Is alignment driven by shared BPE tokens or true semantics? |
| 5 | Regional Model Delta-CKA | Does regional specialization trade off cross-lingual universality? |
| 6 | Cross-Model Representational Drift | Do regional models shift target-language representations, and does drift predict MRR gains? |

### Output Artifacts

All results are saved to `analysis/results/cross_lingual/`:

| Directory | Contents | Naming |
|---|---|---|
| `activations/` | Mean-pooled sentence embeddings | `layer_{idx}_{language}.pt` |
| `cka_matrices/` | Pairwise CKA similarity matrices | `layer_{idx}_{kernel}_cka.npy` |
| `metrics/` | Convergence curves, clustering metrics | `{metric_name}.json` |
| `figures/` | All generated plots | `{plot_type}_layer_{idx}.png` |

---

## Languages

13 languages across **5 families**, **6 scripts**, and **3 resource tiers**:

| Language | ISO | FLORES Code | Script | Family | Resource |
|---|---|---|---|---|---|
| English | `en` | `eng_Latn` | Latin | Indo-European | High |
| Spanish | `es` | `spa_Latn` | Latin | Indo-European | High |
| French | `fr` | `fra_Latn` | Latin | Indo-European | High |
| German | `de` | `deu_Latn` | Latin | Indo-European | High |
| Arabic | `ar` | `arb_Arab` | Arabic | Afro-Asiatic | High |
| Hindi | `hi` | `hin_Deva` | Devanagari | Indo-European | Mid |
| Bengali | `bn` | `ben_Beng` | Bengali | Indo-European | Mid |
| Tamil | `ta` | `tam_Taml` | Tamil | Dravidian | Mid |
| Turkish | `tr` | `tur_Latn` | Latin | Turkic | Mid |
| Persian | `fa` | `pes_Arab` | Arabic | Indo-European | Mid |
| Swahili | `sw` | `swh_Latn` | Latin | Niger-Congo | Low |
| Amharic | `am` | `amh_Ethi` | Ge'ez | Afro-Asiatic | Low |
| Yoruba | `yo` | `yor_Latn` | Latin | Niger-Congo | Low |

**Why these 13?** The selection maximizes diversity along three axes to disentangle confounds: shared vocabulary (same script), shared grammar (same family), or genuine semantic convergence.

---

## Model Details

### Tiny Aya Global

| Property | Value |
|---|---|
| HuggingFace ID | [`CohereLabs/tiny-aya-global`](https://huggingface.co/CohereLabs/tiny-aya-global) |
| Parameters | 3.35B |
| Architecture | CohereForCausalLM (decoder-only) |
| Transformer Layers | 36 (layers 0--34: sliding-window attention; layer 35: global attention; pattern: 3 sliding-window + 1 global, repeated 9 times) |
| Hidden Dimension | 3072 |
| Languages | 70+ |
| Tokenizer | CohereTokenizer (BPE), left-padding for batch inference |
| Paper | [arXiv:2603.11510](https://arxiv.org/abs/2603.11510) |

### Regional Variants (used in Notebooks 08--09)

| Model | HuggingFace ID | Focus | Target Languages |
|---|---|---|---|
| Global | `CohereLabs/tiny-aya-global` | Balanced across 70+ languages | All |
| Earth | `CohereLabs/tiny-aya-earth` | West Asia + Africa | sw, am, yo, ar, tr, fa |
| Fire | `CohereLabs/tiny-aya-fire` | South Asia | hi, bn, ta |
| Water | `CohereLabs/tiny-aya-water` | Europe + Asia Pacific | en, de, fr, es |

---

## Methodology

The research follows a **five-stage pipeline** (from the [project specification](agent_docs/project_description.md)):

### Stage 1: Parallel Dataset Construction
- **FLORES+** `devtest` split: 1,012 professionally translated sentences.
- Semantic equivalence guaranteed: sentence `i` in language A is the translation of sentence `i` in language B.
- **Linguistic variation pairs** (@danielmargento): controlled lexical, syntactic, and semantic variations for probing.

### Stage 2: Layer-wise Hidden State Extraction
- Forward hooks (`ActivationStore`) capture hidden states at every transformer layer (36 layers).
- Mean pooling over non-padding tokens produces sentence-level embeddings: `(n_sentences, 3072)` per language per layer.
- Activations detached from computation graph and moved to CPU for memory efficiency.
- Supports fp16 (~6.7 GB VRAM) and 4-bit quantization (~1.7 GB VRAM) via bitsandbytes.
- Total extraction: 13 languages × 36 layers × 1,012 sentences × 3,072 dimensions.

### Stage 3: Cross-Lingual Similarity Measurement
- **Linear CKA** for fast O(n*d^2) pairwise comparison at each layer.
- **RBF CKA** for capturing nonlinear relationships.
- **Whitened CKA** to test whether alignment is confounded by anisotropy.
- **Permutation tests** (1,000+ replicates) confirm statistical significance.

### Stage 4: Convergence Detection and Grouping Analysis
- Average cross-lingual CKA plotted vs layer depth with 95% CI.
- Convergence layer = first layer where avg CKA >= 0.75.
- Hierarchical clustering (Ward's method) tracks family/script dissolution via ARI and cophenetic correlation.
- Translation retrieval (MRR, Recall@k) provides functional alignment evidence.

### Stage 5: Comparative Analysis
- **Delta-CKA** (Notebook 08) between Global and regional models reveals where specialization occurs.
- **Script decomposition** (Notebook 07) separates token-surface from semantic alignment.
- **Cross-model drift** (Notebook 09) measures per-language representational shifts and correlates with retrieval MRR gains.
- Results connected to the three-stage hypothesis from the literature.

---

## Development

### Running Tests

```bash
uv run pytest tests/ -v
```

All 71 tests are CPU-only and use mock models -- no GPU or network access required.

| Test File | Tests | What It Covers |
|---|---|---|
| `test_cka.py` | 27 | All 4 CKA variants, `_center_gram`, `_validate_activation_pair`, `MinibatchCKAAccumulator`, `compute_layerwise_cka`, permutation tests, edge cases (NaN, Inf, zero samples, dimension mismatches) |
| `test_hooks.py` | 16 | `ActivationStore` register/collect/clear/remove, mean pooling with masks, `register_model_hooks` with mock Cohere/GPT-2 models, `_find_transformer_layers`, `get_model_layer_count`, out-of-range layer errors |
| `test_languages.py` | 17 | All 13 languages have complete metadata, unique ISO/FLORES codes, `LanguageInfo` is frozen, `LANGUAGE_FAMILIES`/`SCRIPT_GROUPS`/`RESOURCE_GROUPS` cover all languages, lookup functions |
| `test_retrieval.py` | 11 | `compute_cosine_similarity_matrix`, `compute_mrr`, `compute_recall_at_k`, `compute_all_retrieval_metrics`, perfect alignment gives MRR=1, Recall@k increases with k, input validation |

### Linting

```bash
uv run ruff check src/ tests/          # Check
uv run ruff check --fix src/ tests/    # Auto-fix safe issues
```

Ruff configuration in `pyproject.toml` enables: E (pycodestyle), W (warnings), F (pyflakes), I (isort), N (pep8-naming), UP (pyupgrade), B (flake8-bugbear), SIM (flake8-simplify), TCH (type-checking), RUF (ruff-specific). Math-notation variables (`X`, `Y`, `K`, `L`) are explicitly allowed via N802/N803/N806 ignores.

### Code Style

- **Imports:** isort-sorted, `from src.xxx` for all internal imports
- **Variable naming:** Lowercase, except math-notation matrices (`X`, `Y`, `K`, `L`, `HSIC`)
- **Docstrings:** NumPy-style with Parameters, Returns, Raises, Examples (see [CONTRIBUTING.md](CONTRIBUTING.md#documentation-standards) for the full specification)
- **Type hints:** Required; use `X | None` (not `Optional[X]`), `list[int]` (not `List[int]`)
- **Line length:** 88 characters (ruff default)

### Documentation Requirements

Every Python file must have a module-level docstring. Every public class and function must have a NumPy-style docstring with `Parameters`, `Returns`, and `Raises` sections. Module-level constants must have `#:` doc-comments. See [CONTRIBUTING.md - Documentation Standards](CONTRIBUTING.md#documentation-standards) for detailed requirements and examples.

---

## Contributing

See **[CONTRIBUTING.md](CONTRIBUTING.md)** for the full guide, including:

- Detailed description of every `src/` subpackage and what belongs where
- How to create new analysis topics (`src/<topic_name>/` for code, `analysis/<topic_name>/` for notebooks)
- History of merged contributions (Daniel's linguistic variation pipeline, PRs #1 and #2)
- Complete PR checklist with lint/test/secrets requirements
- Branch strategy and commit message conventions

See **[AGENTS.md](AGENTS.md)** for AI agent-specific guidelines.

### Quick Pre-Push Checklist

```bash
uv run ruff check src/ tests/    # Must print "All checks passed!"
uv run pytest tests/ -v          # All 71 tests must pass
uv sync                          # Dependencies must resolve cleanly
```

---

## References

### Papers

1. **Kornblith et al.** (2019). Similarity of Neural Network Representations Revisited. *ICML*. [arXiv:1905.00414](https://arxiv.org/abs/1905.00414)
2. **Wendler et al.** (2024). Do Llamas Work in English? On the Latent Language of Multilingual Transformers. *ACL*. [arXiv:2402.10588](https://arxiv.org/abs/2402.10588)
3. **Dumas et al.** (2025). Separating Tongue from Thought: Activation Patching Reveals Language-Agnostic Concept Representations in Transformers. *ACL*. [arXiv:2411.08745](https://arxiv.org/abs/2411.08745)
4. **Harrasse et al.** (2025). Tracing Multilingual Representations in LLMs with Cross-Layer Transcoders. [arXiv:2511.10840](https://arxiv.org/abs/2511.10840)
5. **Nakai et al.** (2025). TRepLiNa: Layer-wise CKA+REPINA Alignment Improves Low-Resource Machine Translation in Aya-23 8B. [arXiv:2510.06249](https://arxiv.org/abs/2510.06249)
6. **Salamanca et al.** (2026). Tiny Aya: Bridging Scale and Multilingual Depth. [arXiv:2603.11510](https://arxiv.org/abs/2603.11510)

### Upstream Repositories

- [Wayy-Research/project-aya](https://github.com/Wayy-Research/project-aya) -- CKA and teacher representation notebooks this work extends
- [CohereLabs/tiny-aya](https://huggingface.co/collections/CohereLabs/tiny-aya) -- Model collection on HuggingFace
- [openlanguagedata/flores_plus](https://huggingface.co/datasets/openlanguagedata/flores_plus) -- FLORES+ parallel corpus

---

## License

MIT
