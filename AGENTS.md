# AI Agent Guidelines for Tiny Aya Under The Hood

Comprehensive instructions for AI coding agents (Copilot, Cursor, Droid, Aider, etc.) contributing to this repository.

---

## Project Context

This is a **mechanistic interpretability research project** studying how multilingual representations emerge in [Tiny Aya Global](https://huggingface.co/CohereLabs/tiny-aya-global), a 3.35B-parameter multilingual LM with 36 transformer layers (3 sliding-window + 1 global attention, repeated 9 times) and hidden dimension 3072.

**Central question:** Where do language-agnostic representations emerge and where does language-specific specialization occur?

**Repository design:** Shared infrastructure in `src/`, independent analysis topics with Python code in `src/analysis/<topic_name>/` and notebooks in `analysis/<topic_name>/`.

---

## File Structure and Import Rules

```
src/                        # Main Python package
  __init__.py               # __version__ = "0.1.0"
  utils/languages.py        # Language enum (13 members) with LanguageInfo dataclass
  data/flores_loader.py     # FLORES+ loader (requires HF_TOKEN in .env)
  data/translate_data_openai.py  # OpenAI GPT-4.1 translation (requires OPENAI_API_KEY)
  data/linguistic_variation/ # Cohere-powered dataset generation (@danielmargento)
  analysis/                 # Analysis sub-packages (one per topic)
    cross_lingual_embedding_alignment/cka.py           # 4 CKA variants + permutation tests (863 lines)
    cross_lingual_embedding_alignment/hooks.py         # ActivationStore + forward hooks (611 lines)
    cross_lingual_embedding_alignment/cross_lingual_alignment.py  # CrossLingualAlignmentAnalyzer orchestrator (741 lines)
    cross_lingual_embedding_alignment/retrieval_metrics.py  # MRR, Recall@k, cosine similarity (418 lines)
    cross_lingual_embedding_alignment/clustering.py    # Hierarchical clustering, family/script dissolution (433 lines)
    cross_lingual_embedding_alignment/visualization.py # 12 plotting functions, seaborn/matplotlib (892 lines)
analysis/                   # Analysis notebooks and writeups (one subdir per topic)
  cross_lingual_embedding_alignment/  # 9 notebooks (01--09) + paperback.md (cross-lingual alignment)
data/                       # Raw CSVs (test_data.csv, test_data_translation_output.csv)
tests/                      # 71 CPU-only tests (test_cka, test_hooks, test_languages, test_retrieval)
```

**Import convention -- always use the `src.` prefix:**

```python
# CORRECT
from src.utils.languages import Language, LANGUAGE_FAMILIES, SCRIPT_GROUPS
from src.analysis.cross_lingual_embedding_alignment.cka import linear_cka, rbf_cka, whitened_cka
from src.analysis.cross_lingual_embedding_alignment.hooks import ActivationStore, register_model_hooks, load_model
from src.data.flores_loader import load_flores_parallel_corpus

# WRONG (old package names, never use)
from uth.utils.languages import Language
from utils.languages import Language
from analysis.cka import linear_cka
from src.analysis.cka import linear_cka
```

---

## Critical Rules

### 1. Never Commit Secrets

- `.env` is gitignored. Never reference its contents in committed code.
- Never hardcode API keys (HF_TOKEN, OPENAI_API_KEY, COHERE_API_KEY).
- Use `os.getenv("VAR_NAME")` for environment variables.
- Before staging: `grep -rE '(hf_[a-zA-Z0-9]{10,}|sk-[a-zA-Z0-9]{20,})' src/ tests/` must return empty.

### 2. Always Run Checks

```bash
uv run ruff check src/ tests/    # Must print "All checks passed!"
uv run pytest tests/ -v          # All 71 tests must pass
```

### 3. Math-Notation Variables Are Allowed

This is a scientific computing project. Variables like `X`, `Y`, `K`, `L`, `HSIC` follow mathematical convention. Do NOT rename them to lowercase. Ruff rules N802/N803/N806 are intentionally ignored.

### 4. Python 3.12+

Use modern type hints: `X | None` (not `Optional[X]`), `list[int]` (not `List[int]`). Do not add `from __future__ import annotations` -- it is not needed on 3.12+.

### 5. Do Not Modify `src/data/linguistic_variation/` Without Coordination

These files are from @danielmargento (PRs #1, #2). They have per-file ruff ignores (RUF001 for en-dashes in prompt strings, SIM118, F841). Changing them requires coordinating with the original contributor.

---

## Key APIs You'll Use Most Often

### Language Registry

```python
from src.utils.languages import Language, LANGUAGE_FAMILIES, SCRIPT_GROUPS, RESOURCE_GROUPS

# 13 languages: ENGLISH, SPANISH, FRENCH, GERMAN, ARABIC, HINDI, BENGALI, TAMIL, TURKISH, PERSIAN, SWAHILI, AMHARIC, YORUBA
# Each has: .lang_name, .iso_code, .flores_code, .script, .family, .resource_level
# Groupings: LANGUAGE_FAMILIES (5 families), SCRIPT_GROUPS (6 scripts), RESOURCE_GROUPS (3 tiers)
```

### CKA Functions

```python
from src.analysis.cross_lingual_embedding_alignment.cka import linear_cka, rbf_cka, whitened_cka, minibatch_cka, cka_permutation_test

# All accept torch.Tensor of shape (n_samples, d_features)
# Return scalar tensor (or float for minibatch/permutation)
# Strict validation: NaN/Inf checks, dimension matching, empty input detection
score = linear_cka(X, Y)        # O(n*d^2), fastest
score = rbf_cka(X, Y)           # O(n^2*d), nonlinear
score = whitened_cka(X, Y)      # O(d^3), anisotropy-corrected
score = minibatch_cka(X, Y)     # Streaming, memory-efficient
result = cka_permutation_test(X, Y, n_permutations=1000)  # Returns dict with p-value
```

### Activation Extraction

```python
from src.analysis.cross_lingual_embedding_alignment.hooks import load_model, ActivationStore, register_model_hooks

# load_model supports precision="fp16" (~6.7GB VRAM) or "4bit" (~1.7GB)
# ActivationStore.collect_mean_pooled() gives (n_sentences, 3072) per layer
# register_model_hooks auto-detects Cohere/Llama/GPT-2 architecture
```

### Orchestrator

```python
from src.analysis.cross_lingual_embedding_alignment.cross_lingual_alignment import CrossLingualAlignmentAnalyzer

# Full pipeline: extract_all_activations() -> compute_cka_matrices() ->
# compute_convergence_curve() -> find_convergence_layer() ->
# compute_retrieval_scores() -> compute_clustering_analysis() -> save_results()
# Caches activations and CKA matrices as instance attributes
```

### Retrieval Metrics

```python
from src.analysis.cross_lingual_embedding_alignment.retrieval_metrics import compute_mrr, compute_recall_at_k, compute_all_retrieval_metrics

# All accept numpy arrays of shape (n_sentences, embedding_dim)
# compute_all_retrieval_metrics returns: {"mrr": float, "recall@1": float, "recall@5": float, ...}
```

### Clustering

```python
from src.analysis.cross_lingual_embedding_alignment.clustering import compute_hierarchical_clustering, compute_family_dissolution_metrics

# compute_hierarchical_clustering: Ward's method, returns linkage matrix + cophenetic correlation
# compute_family_dissolution_metrics: intra/inter-family CKA, family gap, ARI
```

---

## Testing Guidelines

- All tests in `tests/` run with `uv run pytest tests/ -v`.
- Tests must be **CPU-only** -- no GPU, no network, no API calls.
- Use `torch.randn(...)` for synthetic data, mock `nn.Module` subclasses for models.
- Test file naming: `test_{module_name}.py`.
- Current coverage: 71 tests across 4 files (835 lines total).

### Current Test Files

| File | Count | Covers |
|---|---|---|
| `test_cka.py` | 27 | `TestInputValidation` (6), `TestLinearCKA` (5), `TestRBFCKA` (3), `TestWhitenedCKA` (2), `TestMinibatchCKA` (5), `TestPermutationTest` (3), `TestComputeLayerwiseCKA` (2), `TestCenterGram` (1) |
| `test_hooks.py` | 16 | `TestActivationStore` (7), `TestRegisterModelHooks` (5), `TestFindTransformerLayers` (2), `TestGetModelLayerCount` (2) |
| `test_languages.py` | 17 | `TestLanguageEnum` (6), `TestConvenienceGroupings` (5), `TestLookupFunctions` (6) |
| `test_retrieval.py` | 11 | `TestCosineSimilarity` (2), `TestMRR` (3), `TestRecallAtK` (3), `TestAllRetrievalMetrics` (2) + input validation |

---

## Ruff Configuration

Configured in `pyproject.toml`:

```toml
[tool.ruff.lint]
select = ["E", "W", "F", "I", "N", "UP", "B", "SIM", "TCH", "RUF"]
ignore = ["E501", "B008", "B028", "B905", "N802", "N803", "N806",
          "UP007", "UP045", "UP035", "UP006", "TC001", "TC002", "SIM108", "RUF100"]

[tool.ruff.lint.per-file-ignores]
"tests/*" = ["B011", "SIM300"]
"src/data/linguistic_variation/*" = ["RUF001", "SIM118", "F841"]
```

Auto-fix safe issues: `uv run ruff check --fix src/ tests/`

---

## Dependency Management

Single dependency list in `pyproject.toml` (no optional groups). Key dependencies:

- **ML:** torch>=2.0, transformers>=4.40, accelerate>=0.28, bitsandbytes>=0.43
- **Data:** datasets>=2.18, huggingface-hub>=0.20, pandas>=2.0
- **Analysis:** numpy>=1.24, scipy>=1.11, scikit-learn>=1.3
- **Visualization:** matplotlib>=3.7, seaborn>=0.13
- **Dev:** pytest>=7.0, pytest-cov>=4.0, ruff>=0.4
- **Notebooks:** jupyter>=1.0, ipykernel>=6.0, ipywidgets>=8.0

To add a dependency: edit `pyproject.toml`, run `uv sync`, commit both `pyproject.toml` and `uv.lock`.

---

## Commit Messages

```
feat: add whitened CKA implementation
fix: correct mean pooling for padded sequences
docs: update analysis1 paperback with results
refactor: extract clustering metrics into separate module
test: add edge case tests for RBF CKA
```

---

## Common Pitfalls

1. **`from uth.` imports** -- The old package name. Now it's `src.`.
2. **Python code in `data/`** -- That directory is for raw data files only. Python goes in `src/data/`.
3. **Skipping tests** -- All 71 must pass before any commit.
4. **`pip install`** -- Always use `pyproject.toml` + `uv sync`.
5. **Hardcoded paths** -- Use `Path(__file__).resolve().parent` patterns.
6. **Notebook outputs** -- Clear large outputs before committing (binary blobs inflate the repo).
7. **`.env` in commits** -- Always check `git diff --cached --name-only` for `.env` files.
8. **Modifying `__init__.py` exports** -- If you add a function to a module, also export it in the parent `__init__.py` and update `__all__`.
