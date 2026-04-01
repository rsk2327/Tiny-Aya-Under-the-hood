# Contributing to Tiny Aya Under The Hood

Everything you need to know before submitting your first PR. Read this in full.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Layout](#repository-layout)
- [Source Package In Detail](#source-package-in-detail)
- [Analysis Groups](#analysis-groups)
- [Merged Contributions](#merged-contributions)
- [Development Workflow](#development-workflow)
- [Pre-Push Checklist](#pre-push-checklist)
- [PR Checklist](#pr-checklist)
- [Branch Strategy](#branch-strategy)
- [Code Style](#code-style)
- [Adding Dependencies](#adding-dependencies)

---

## Project Overview

This is a mechanistic interpretability study of [Tiny Aya Global](https://huggingface.co/CohereLabs/tiny-aya-global), investigating where language-agnostic vs. language-specific representations emerge across its 36 transformer layers (3 sliding-window + 1 global attention, repeated 9 times). The central question:

> Which parts of Tiny Aya's network learn language-agnostic representations, and which parts become specialized for specific languages or regions?

The repository separates **shared infrastructure** (`src/`) from **analysis work** (`analysis/<topic_name>/` for notebooks, `src/analysis/<topic_name>/` for Python code), allowing multiple contributors to work on independent research questions while reusing the same tooling.

---

## Repository Layout

```
src/                        # Shared Python package (import as `from src.xxx`)
  utils/                    # Language registry, metadata lookups
  data/                     # Data loading, translation, linguistic variation generation
  analysis/                 # Analysis sub-packages (one per topic)
    cross_lingual_embedding_alignment/  # CKA, hooks, clustering, retrieval, visualization
analysis/                   # Analysis notebooks and writeups (one subdir per topic)
  cross_lingual_embedding_alignment/  # 9 notebooks (01--09) + paperback.md
data/                       # Raw data files (CSVs, JSONs -- NOT Python code)
tests/                      # Unit tests for src/ (71 tests, CPU-only)
analysis/results/           # Generated outputs (figures, metrics, activations) per topic
agent_docs/                 # Internal project specification
```

---

## Source Package In Detail

All shared Python code lives under `src/`. When you add new utilities, they go here. When you need existing functionality in notebooks, import from here. **All imports use the `src.` prefix** (e.g., `from src.analysis.cross_lingual_embedding_alignment.cka import linear_cka`).

### `src/utils/` -- Language Registry

- **`languages.py`** -- The `Language` enum with 13 members, each wrapping a frozen `LanguageInfo` dataclass: `name`, `iso_code`, `flores_code`, `script`, `family`, `resource_level`. Also defines `LANGUAGE_FAMILIES`, `SCRIPT_GROUPS`, `RESOURCE_GROUPS` dicts and `get_language_by_iso/name` lookup functions.
- **`__init__.py`** -- Re-exports everything from `languages.py`.

**When to modify:** Only when adding languages to the study. Changes propagate to all analyses.

### `src/data/` -- Data Loading and Generation

- **`flores_loader.py`** -- Loads the FLORES+ parallel corpus (1,012 `devtest` sentences) from HuggingFace. Authenticates via `HF_TOKEN` from `.env` (path resolution walks up from `src/data/` to project root using `Path(__file__).resolve().parent.parent.parent`). Also provides `get_corpus_statistics()` for exploratory analysis.
- **`translate_data_openai.py`** -- OpenAI structured-output translation pipeline. Uses `TranslationPipeline` class with Pydantic models (`TranslationItem`, `TranslationBatch`) for type-safe API responses. Reads CSV input, translates in batches via GPT-4.1, outputs CSV or JSON.
- **`linguistic_variation/`** -- Dataset 2 generation pipeline by @danielmargento (see [Merged Contributions](#merged-contributions) below):
  - `generate_linguistic_variation.py` -- Cohere API batched generation of lexical/syntactic/semantic sentence pairs with deduplication and optional interactive review
  - `review_linguistic_variation.py` -- Quality review pipeline scoring pairs against type-specific criteria
  - `dedup_dataset.py` -- Post-hoc deduplication removing exact duplicates and reused word swaps
  - `linguistic_variation.json` -- Generated dataset (~18,000 pairs)

**When to modify:** When adding new data sources or loaders. Each new dataset should get its own submodule or file.

### `src/analysis/cross_lingual_embedding_alignment/` -- Core Analysis Toolkit

| File | Lines | Key Exports | Purpose |
|---|---|---|---|
| `cka.py` | 863 | `linear_cka`, `rbf_cka`, `whitened_cka`, `minibatch_cka`, `cka_permutation_test`, `MinibatchCKAAccumulator`, `CKAHeatmapData`, `compute_layerwise_cka` | Four CKA variants with strict input validation, streaming accumulator, permutation-based statistical testing, and serializable data containers |
| `hooks.py` | 611 | `ActivationStore`, `register_model_hooks`, `load_model`, `get_model_layer_count` | Forward-hook activation extraction for HuggingFace models; supports Cohere/Llama/GPT-2 architectures; mean pooling over non-padding tokens; fp16 and 4-bit loading |
| `cross_lingual_alignment.py` | 741 | `CrossLingualAlignmentAnalyzer` | Main orchestrator: extract activations for all languages, compute CKA matrices (linear/rbf/whitened), detect convergence layer, compute retrieval scores, run clustering analysis, save/load results with standardized naming |
| `retrieval_metrics.py` | 418 | `compute_mrr`, `compute_recall_at_k`, `compute_all_retrieval_metrics`, `compute_cosine_similarity_matrix`, `compute_confusion_matrix` | Translation retrieval scoring with cosine similarity, MRR, Recall@k, rank statistics, and cross-language confusion matrices |
| `clustering.py` | 433 | `compute_hierarchical_clustering`, `compute_cluster_assignments`, `compute_family_dissolution_metrics`, `compute_script_group_metrics` | Ward's method clustering, cophenetic correlation, ARI vs. true family labels, intra/inter family and script CKA gap tracking |
| `visualization.py` | 892 | 12 plotting functions (see README) | Publication-quality heatmaps, convergence curves, dendrograms, retrieval curves, script/family gap plots, anisotropy heatmaps, eigenvalue spectra. Seaborn whitegrid, RdBu_r colormap, 300 DPI output |

**When to modify:** When you need new shared analysis primitives used across multiple analysis topics. If code is specific to one analysis, keep it in the notebook.

---

## Analysis Topics

Each independent research focus gets its own pair of directories: a Python package under `src/analysis/` and a notebooks directory under `analysis/` at the project root.

### Naming Convention

```
src/analysis/<topic_name>/      # Python modules for the analysis
├── __init__.py
├── module_a.py
└── module_b.py

analysis/<topic_name>/          # Notebooks and writeups
├── paperback.md                # Full writeup (methodology, math, results, references)
├── 01_*.ipynb                  # Notebooks numbered in execution order
├── 02_*.ipynb
└── ...
```

### Current Analysis Topics

| Python Package | Notebooks Directory | Focus | Notebooks | Status |
|---|---|---|---|---|
| `src/analysis/cross_lingual_embedding_alignment/` | `analysis/cross_lingual_embedding_alignment/` | Cross-lingual embedding alignment (CKA, clustering, retrieval, script decomposition, regional comparison, cross-model drift) | 9 | All notebooks executed; paper skeleton + detailed writeup complete |
| `src/analysis/<next_topic>/` | `analysis/<next_topic>/` | *(available)* | -- | -- |

### Starting a New Analysis Topic

1. `mkdir -p src/analysis/<topic_name>/ analysis/<topic_name>/`
2. Add an `__init__.py` to `src/analysis/<topic_name>/` with relevant exports.
3. Add a `paperback.md` to `analysis/<topic_name>/` documenting your research question, methodology, and expected findings.
4. Add numbered notebooks (`01_*.ipynb`, `02_*.ipynb`, etc.) in execution order to `analysis/<topic_name>/`.
5. Import shared code: `from src.analysis.cross_lingual_embedding_alignment.cka import linear_cka`
6. If you need new shared utilities, add them to `src/analysis/<topic_name>/` with tests in `tests/`.
7. Save outputs to `analysis/results/<topic_name>/`.

---

## Merged Contributions

### Dataset 2: Linguistic Variation Pipeline (PRs #1 and #2)

**Contributor:** @danielmargento

**What it adds:** A Cohere API-powered pipeline for generating sentence pairs that test three types of linguistic variation:

| Type | Description | Validation Criterion |
|---|---|---|
| Lexical | Single-word synonym swap; every other word identical | Exactly 1 differing word that is a true synonym |
| Syntactic | Grammatical transformation preserving meaning (active/passive, cleft, dative alternation) | Truth-conditionally equivalent, structurally different |
| Semantic | Full paraphrase: different vocabulary AND syntax, same meaning | Unambiguous equivalence, both grammatical |

**Files (now at `src/data/linguistic_variation/`):**

| File | Purpose |
|---|---|
| `generate_linguistic_variation.py` | Main Cohere generation pipeline with batch dedup and stall detection |
| `review_linguistic_variation.py` | Quality review scoring each pair against type-specific criteria |
| `dedup_dataset.py` | Removes exact sentence duplicates and reused lexical swaps |
| `linguistic_variation.json` | The generated dataset (~18,000 pairs across all 3 types) |

**History:** Originally added under `uth/data/` in PRs #1 and #2, moved to `src/data/linguistic_variation/` during the repository restructuring. The default output path in `generate_linguistic_variation.py` was updated accordingly. Per-file ruff ignores are configured for these files (RUF001 for en-dashes in prompt strings, SIM118, F841 for unused vars).

---

## Development Workflow

### Initial Setup

```bash
git clone https://github.com/Vidit-Ostwal/Tiny-Aya-Under-the-hood.git
cd Tiny-Aya-Under-the-hood
uv sync

# Create .env with your HuggingFace token
echo "HF_TOKEN=hf_your_token_here" > .env

# Verify
uv run pytest tests/ -v        # 71 tests pass
uv run ruff check src/ tests/  # All checks passed
```

### Making Changes

```bash
# 1. Create a feature branch from latest main
git fetch origin && git checkout -b feat/your-feature origin/main

# 2. Make your changes

# 3. Run all checks (REQUIRED before every commit)
uv run ruff check src/ tests/
uv run ruff check --fix src/ tests/    # Auto-fix safe issues
uv run pytest tests/ -v

# 4. Stage, review, and commit
git add -A
git diff --cached                       # Review the full diff
git diff --cached --name-only | grep -E '\.env'  # Must be empty
git commit -m "feat: descriptive message"

# 5. Push and create PR
git push -u origin feat/your-feature
gh pr create --base main --title "Title" --body "Description"
```

---

## Pre-Push Checklist

All must pass before pushing:

```bash
# 1. Lint
uv run ruff check src/ tests/

# 2. Tests
uv run pytest tests/ -v

# 3. Dependencies
uv sync

# 4. Secrets scan (must return empty)
grep -rE '(hf_[a-zA-Z0-9]{10,}|sk-[a-zA-Z0-9]{20,}|AKIA[A-Z0-9]{16})' src/ tests/ analysis*/

# 5. No .env in staged files
git diff --cached --name-only | grep -E '\.env'
```

---

## PR Checklist

Copy this into your PR description and check off each item:

```markdown
### PR Checklist

- [ ] **Branch:** Created from latest `main`
- [ ] **Lint:** `uv run ruff check src/ tests/` passes
- [ ] **Tests:** `uv run pytest tests/ -v` -- all 71 pass
- [ ] **New tests:** Added tests for any new `src/` code
- [ ] **Dependencies:** `uv sync` completes cleanly; both `pyproject.toml` and `uv.lock` committed
- [ ] **Secrets:** No API keys, tokens, or passwords in the diff
- [ ] **Imports:** All internal imports use `from src.xxx` prefix
- [ ] **Docstrings:** New public functions have NumPy-style docstrings
- [ ] **Paperback:** Updated `paperback.md` if modifying an analysis
- [ ] **Commit messages:** Prefixed: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`
```

---

## Branch Strategy

| Branch | Purpose | Who Merges |
|---|---|---|
| `main` | Stable, reviewed code | Repo admin via PR |
| `feat/<name>` | New features or analysis groups | Contributors |
| `fix/<name>` | Bug fixes | Contributors |
| `docs/<name>` | Documentation updates | Contributors |

**Rules:** Never push directly to `main`. All changes go through PRs. Rebase on `main` before opening a PR.

---

## Code Style

Enforced by ruff (configured in `pyproject.toml`):

| Rule | Convention |
|---|---|
| Line length | 88 characters |
| Imports | isort-sorted; `from src.xxx` for all internal imports |
| Variable naming | Lowercase, except math-notation (`X`, `Y`, `K`, `L`, `HSIC`) which is explicitly allowed |
| Type hints | Required on all public functions; use `X | None` (not `Optional[X]`), `list[int]` (not `List[int]`) |
| Docstrings | NumPy-style (see Documentation Standards below) |
| Target Python | 3.12+ |

---

## Documentation Standards

Every Python file must follow PEP 257 and the NumPy docstring convention. These are the requirements for every category of code:

### Module-Level Docstrings

Every `.py` file must open with a module docstring that explains:
- **What** the module provides (one-sentence summary on line 1).
- **How** it fits into the project (which analysis step, which pipeline).
- **Dependencies** (external API keys, GPU requirements, other modules).
- **Usage example** (a runnable code snippet in a `::` block).
- **References** (papers, upstream repos, specs) where applicable.

```python
"""Centered Kernel Alignment (CKA) for cross-lingual similarity measurement.

This module provides four CKA variants tailored for cross-lingual
analysis: linear, RBF, mini-batch, and anisotropy-corrected (whitened).

References:
    - Kornblith et al., "Similarity of Neural Network Representations
      Revisited" (ICML 2019)
"""
```

### Class Docstrings

Every class (including Pydantic models and dataclasses) must have:
- One-sentence summary.
- Extended description of purpose and usage.
- `Attributes` section listing every public attribute with type and meaning.
- `Examples` section for non-trivial classes.

```python
class MinibatchCKAAccumulator:
    """Streaming accumulator for memory-efficient Linear CKA computation.

    Collects sufficient statistics (cross-covariance matrices) across
    mini-batches, then computes CKA from the accumulated statistics.

    Attributes
    ----------
    d_x : int
        Feature dimension of the first representation.
    d_y : int
        Feature dimension of the second representation.

    Examples
    --------
    >>> acc = MinibatchCKAAccumulator(d_x=768, d_y=768)
    >>> for X_batch, Y_batch in dataloader:
    ...     acc.update(X_batch, Y_batch)
    >>> score = acc.compute()
    """
```

### Function and Method Docstrings

Every public function and method must have a NumPy-style docstring with:
- One-sentence summary (imperative mood: "Compute...", "Load...", "Return...").
- Extended description (when the logic is non-trivial).
- `Parameters` section (parameter name, type, description).
- `Returns` section (type and meaning).
- `Raises` section (every exception the function explicitly raises).
- `Examples` section (for key public functions).

```python
def linear_cka(X: torch.Tensor, Y: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """Compute Linear CKA between two activation matrices.

    Parameters
    ----------
    X : torch.Tensor
        Activation matrix, shape ``(n_samples, d_x)``.
    Y : torch.Tensor
        Activation matrix, shape ``(n_samples, d_y)``.
    eps : float
        Numerical stability constant.

    Returns
    -------
    torch.Tensor
        Scalar CKA similarity score in [0, 1].

    Raises
    ------
    ValueError
        If inputs fail validation (wrong shape, NaN, etc.).

    Examples
    --------
    >>> score = linear_cka(torch.randn(256, 768), torch.randn(256, 768))
    >>> 0.0 <= score.item() <= 1.0
    True
    """
```

### Private/Helper Functions

Must have at least a one-line docstring explaining purpose. Full NumPy sections are optional but encouraged for complex helpers.

### Module-Level Constants

Must have a `#:` doc-comment (Sphinx-compatible) on the line above:

```python
#: System prompt for **lexical** variation generation.
LEXICAL_SYSTEM_PROMPT = """..."""
```

### Inline Comments

Required for:
- Non-obvious algorithms or mathematical formulas.
- Workarounds for library quirks or known issues.
- Branching logic with multiple conditions.

Not needed for self-explanatory code (avoid "# increment counter" style comments).

### Test Docstrings

- Every test module: module-level docstring naming what it tests.
- Every test class: one-line docstring (e.g., "Tests for Linear CKA.").
- Every test method: one-line docstring explaining **what** is verified (e.g., "Self-similarity should always be 1.0.").

---

## Adding Dependencies

All dependencies live in the single `[project] dependencies` list in `pyproject.toml`. No optional dependency groups.

```bash
# 1. Add to pyproject.toml dependencies list
# 2. Run:
uv sync
# 3. Commit both pyproject.toml and uv.lock
```

Never install packages with `pip install` directly.
