"""
Main orchestrator for cross-lingual embedding alignment analysis.

This module ties together all analysis components — activation
extraction, CKA computation, retrieval metrics, clustering, and
visualization — into a single ``CrossLingualAlignmentAnalyzer`` class
that manages the full pipeline from model loading to result export.

The analyzer is designed for research reproducibility:
    - All results are saved with standardized keys (per Wei-Yin Ko's
      feedback) to enable shared storage and replication.
    - Intermediate results (activations, CKA matrices) are cached to
      disk so expensive GPU computation only happens once.
    - All methods include progress logging and error handling.

Typical usage in a notebook::

    from src.analysis.cross_lingual_embedding_alignment.cross_lingual_alignment import (
        CrossLingualAlignmentAnalyzer,
    )
    from src.analysis.cross_lingual_embedding_alignment.hooks import load_model
    from src.data.flores_loader import load_flores_parallel_corpus

    # Setup
    model, tokenizer = load_model(precision="fp16")
    corpus = load_flores_parallel_corpus(max_sentences=200)

    # Create analyzer
    analyzer = CrossLingualAlignmentAnalyzer(
        model=model,
        tokenizer=tokenizer,
        parallel_corpus=corpus,
    )

    # Run full pipeline
    analyzer.extract_all_activations()
    cka_matrices = analyzer.compute_cka_matrices()
    convergence_layer = analyzer.find_convergence_layer()
    retrieval_scores = analyzer.compute_retrieval_scores()

    # Save everything
    analyzer.save_results("analysis/results/cross_lingual/")

References:
    - TIN-7 specification (Linear issue)
    - Wayy-Research/project-aya notebooks 02 and 07
"""


import json
import logging
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.analysis.cross_lingual_embedding_alignment.cka import (
    linear_cka,
    rbf_cka,
    whitened_cka,
)
from src.analysis.cross_lingual_embedding_alignment.clustering import (
    compute_family_dissolution_metrics,
    compute_hierarchical_clustering,
    compute_script_group_metrics,
)
from src.analysis.cross_lingual_embedding_alignment.hooks import (
    ActivationStore,
    get_model_layer_count,
    register_model_hooks,
)
from src.analysis.cross_lingual_embedding_alignment.retrieval_metrics import (
    compute_all_retrieval_metrics,
)
from src.utils.languages import Language

# Module-level logger.
logger = logging.getLogger(__name__)


class CrossLingualAlignmentAnalyzer:
    """Orchestrates the full cross-lingual alignment analysis pipeline.

    This class coordinates all steps of the TIN-7 analysis:

    1. **Activation extraction**: Runs parallel sentences through the
       model and extracts mean-pooled hidden states at each layer
       for each language.

    2. **CKA computation**: Computes pairwise CKA between all language
       pairs at each layer, producing a 3D tensor of shape
       ``(n_langs, n_langs, n_layers)``.

    3. **Convergence detection**: Identifies the layer where average
       cross-lingual CKA stabilizes above a threshold (default 0.75).

    4. **Retrieval scoring**: Measures functional alignment via
       translation retrieval metrics (MRR, Recall@k).

    5. **Clustering analysis**: Tracks language family dissolution
       and script group bias across layers.

    6. **Result serialization**: Saves all metrics and intermediate
       data with standardized keys for reproducibility.

    Attributes:
        model: The HuggingFace causal LM (in eval mode).
        tokenizer: The corresponding tokenizer.
        parallel_corpus: Dictionary mapping language names to aligned
            sentence lists.
        languages: Ordered list of ``Language`` enum members being
            analyzed.
        n_layers: Number of transformer layers in the model.
        activations: Cached per-language, per-layer activations.
            Shape: ``{lang_name: {layer_name: tensor(n_sents, d)}}``.
        cka_matrices: Cached CKA matrices per layer.
            Shape: ``{layer_idx: ndarray(n_langs, n_langs)}``.

    Args:
        model: HuggingFace causal language model.
        tokenizer: Corresponding tokenizer.
        parallel_corpus: Dict of ``{lang_name: [sentence, ...]}``,
            as returned by ``load_flores_parallel_corpus``.
        languages: Optional list of ``Language`` enum members to
            analyze. If ``None``, inferred from ``parallel_corpus``
            keys.
        max_length: Maximum token length for tokenization. Sentences
            exceeding this are truncated.
        batch_size: Number of sentences per batch during extraction.
            Lower values use less GPU memory.
        device: PyTorch device for model inference ("cuda" or "cpu").

    Raises:
        ValueError: If the corpus has fewer than 2 languages or if
            language names don't match any ``Language`` enum members.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer,
        parallel_corpus: dict[str, list[str]],
        languages: list[Language] | None = None,
        max_length: int = 128,
        batch_size: int = 16,
        device: str = "cuda",
    ) -> None:
        """Initialize the cross-lingual alignment analyzer.

        See the class-level docstring for full parameter descriptions.
        Languages are resolved from the corpus keys if not provided
        explicitly.  A ``ValueError`` is raised when fewer than two
        languages can be resolved.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.parallel_corpus = parallel_corpus
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device

        # --- Resolve languages ---
        if languages is not None:
            self.languages = languages
        else:
            from src.utils.languages import get_language_by_name
            self.languages = []
            for name in sorted(parallel_corpus.keys()):
                lang = get_language_by_name(name)
                if lang is not None:
                    self.languages.append(lang)
                else:
                    logger.warning(
                        "Language '%s' not in registry; skipping.", name
                    )

        if len(self.languages) < 2:
            raise ValueError(
                "CrossLingualAlignmentAnalyzer requires at least 2 "
                f"languages, but only {len(self.languages)} were resolved."
            )

        # Ordered language names for consistent indexing.
        self.language_names = [lang.lang_name for lang in self.languages]

        # --- Model metadata ---
        self.n_layers = get_model_layer_count(model)
        self.layer_indices = list(range(self.n_layers))

        # --- Result caches (populated by analysis methods) ---
        self.activations: dict[str, dict[str, torch.Tensor]] = {}
        self.cka_matrices: dict[int, np.ndarray] = {}
        self.rbf_cka_matrices: dict[int, np.ndarray] = {}
        self.whitened_cka_matrices: dict[int, np.ndarray] = {}

        logger.info(
            "Analyzer initialized: %d languages, %d layers, "
            "%d sentences/language, batch_size=%d",
            len(self.languages),
            self.n_layers,
            len(next(iter(parallel_corpus.values()))),
            batch_size,
        )

    # =================================================================
    # Step 1: Activation extraction
    # =================================================================

    def extract_activations_for_language(
        self,
        lang_name: str,
    ) -> dict[str, torch.Tensor]:
        """Extract mean-pooled activations for one language.

        Processes all sentences for the given language through the
        model in batches, extracts hidden states at each layer, and
        mean-pools over non-padding tokens to produce sentence-level
        embeddings.

        Args:
            lang_name: Language name (must be a key in
                ``self.parallel_corpus``).

        Returns:
            Dictionary mapping layer names (e.g., "layer_0") to
            tensors of shape ``(n_sentences, hidden_dim)``.

        Raises:
            KeyError: If ``lang_name`` is not in the corpus.
        """
        if lang_name not in self.parallel_corpus:
            raise KeyError(
                f"Language '{lang_name}' not found in parallel corpus. "
                f"Available: {list(self.parallel_corpus.keys())}"
            )

        sentences = self.parallel_corpus[lang_name]

        # --- Set up activation hooks ---
        store = ActivationStore(detach=True, device="cpu")
        register_model_hooks(
            self.model, store, layer_indices=self.layer_indices
        )

        # --- Process sentences in batches ---
        logger.info(
            "Extracting activations for %s (%d sentences)...",
            lang_name, len(sentences),
        )

        with torch.no_grad():
            for batch_start in tqdm(
                range(0, len(sentences), self.batch_size),
                desc=f"  {lang_name}",
                leave=False,
            ):
                batch_end = min(
                    batch_start + self.batch_size, len(sentences)
                )
                batch_sentences = sentences[batch_start:batch_end]

                # Tokenize the batch.
                # Use padding="max_length" so every batch produces tensors
                # with identical seq_len dimension. With padding=True the
                # last (smaller) batch would be padded to its own longest
                # sequence, causing a dimension mismatch when torch.cat
                # concatenates attention masks across batches.
                inputs = self.tokenizer(
                    batch_sentences,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_length,
                ).to(self.device)

                # Store attention mask for mean-pooling.
                store.store_attention_mask(inputs["attention_mask"])

                # Forward pass (activations captured by hooks).
                self.model(**inputs)

        # --- Collect mean-pooled activations ---
        activations = store.collect_mean_pooled()
        store.remove_hooks()

        logger.info(
            "  %s: extracted %d layers, %d sentences each.",
            lang_name,
            len(activations),
            next(iter(activations.values())).shape[0]
            if activations
            else 0,
        )

        return activations

    def extract_all_activations(self) -> None:
        """Extract activations for ALL languages and cache them.

        Results are stored in ``self.activations`` as a nested dict:
        ``{lang_name: {layer_name: tensor(n_sents, d)}}``.

        This is the most expensive step (requires GPU). Once done,
        all subsequent analysis is CPU-only.
        """
        logger.info("=" * 60)
        logger.info("Extracting activations for %d languages...", len(self.languages))
        logger.info("=" * 60)

        for lang in self.languages:
            self.activations[lang.lang_name] = (
                self.extract_activations_for_language(lang.lang_name)
            )

        logger.info("All activations extracted and cached.")

    # =================================================================
    # Step 2: CKA computation
    # =================================================================

    def compute_cka_matrices(
        self,
        kernel: str = "linear",
    ) -> dict[int, np.ndarray]:
        """Compute pairwise CKA between all language pairs at each layer.

        For each layer, produces a symmetric ``(n_langs, n_langs)``
        matrix where entry (i, j) is the CKA similarity between
        language i and language j.

        Args:
            kernel: CKA variant — "linear" (default), "rbf", or
                "whitened".

        Returns:
            Dictionary mapping layer index to CKA matrix.

        Raises:
            RuntimeError: If activations have not been extracted yet.
            ValueError: If *kernel* is not one of ``"linear"``,
                ``"rbf"``, or ``"whitened"``.
        """
        if not self.activations:
            raise RuntimeError(
                "No activations cached. Call extract_all_activations() first."
            )

        n_langs = len(self.language_names)
        matrices: dict[int, np.ndarray] = {}

        for layer_idx in self.layer_indices:
            layer_name = f"layer_{layer_idx}"
            matrix = np.zeros((n_langs, n_langs), dtype=np.float64)

            logger.info("Computing %s CKA for layer %d...", kernel, layer_idx)

            for i in range(n_langs):
                # Diagonal is always 1.0 (self-similarity).
                matrix[i, i] = 1.0

                for j in range(i + 1, n_langs):
                    lang_i = self.language_names[i]
                    lang_j = self.language_names[j]

                    X = self.activations[lang_i][layer_name]
                    Y = self.activations[lang_j][layer_name]

                    if kernel == "linear":
                        score = linear_cka(X, Y).item()
                    elif kernel == "rbf":
                        score = rbf_cka(X, Y).item()
                    elif kernel == "whitened":
                        score = whitened_cka(X, Y).item()
                    else:
                        raise ValueError(f"Unknown kernel: {kernel}")

                    matrix[i, j] = score
                    matrix[j, i] = score  # Symmetric.

            matrices[layer_idx] = matrix

        # Cache in the appropriate attribute.
        if kernel == "linear":
            self.cka_matrices = matrices
        elif kernel == "rbf":
            self.rbf_cka_matrices = matrices
        elif kernel == "whitened":
            self.whitened_cka_matrices = matrices

        return matrices

    # =================================================================
    # Step 3: Convergence detection
    # =================================================================

    def compute_convergence_curve(
        self,
        cka_matrices: dict[int, np.ndarray] | None = None,
    ) -> dict[str, list[float]]:
        """Compute the average cross-lingual CKA per layer.

        For each layer, extracts the upper triangle (off-diagonal)
        of the CKA matrix and computes the mean, standard deviation,
        and 95% confidence interval.

        Args:
            cka_matrices: CKA matrices to analyze. If ``None``, uses
                the cached linear CKA matrices.

        Returns:
            Dictionary containing:
                - ``"layer_indices"``: List of layer indices.
                - ``"avg_cka"``: Mean cross-lingual CKA per layer.
                - ``"std_cka"``: Standard deviation per layer.
                - ``"ci_lower"``: Lower 95% CI bound.
                - ``"ci_upper"``: Upper 95% CI bound.
                - ``"min_cka"``: Minimum pair CKA per layer.
                - ``"max_cka"``: Maximum pair CKA per layer.

        Raises:
            RuntimeError: If no CKA matrices are available.
        """
        if cka_matrices is None:
            cka_matrices = self.cka_matrices

        if not cka_matrices:
            raise RuntimeError(
                "No CKA matrices available. Call compute_cka_matrices() first."
            )

        n_langs = len(self.language_names)
        result: dict[str, list[float]] = {
            "layer_indices": [],
            "avg_cka": [],
            "std_cka": [],
            "ci_lower": [],
            "ci_upper": [],
            "min_cka": [],
            "max_cka": [],
        }

        for layer_idx in sorted(cka_matrices.keys()):
            matrix = cka_matrices[layer_idx]

            # Extract upper triangle (off-diagonal CKA scores).
            upper_triangle_indices = np.triu_indices(n_langs, k=1)
            off_diagonal = matrix[upper_triangle_indices]

            mean_val = float(off_diagonal.mean())
            std_val = float(off_diagonal.std())
            n_pairs = len(off_diagonal)

            # 95% CI: mean +/- 1.96 * std / sqrt(n)
            ci_margin = 1.96 * std_val / np.sqrt(n_pairs) if n_pairs > 1 else 0

            result["layer_indices"].append(layer_idx)
            result["avg_cka"].append(mean_val)
            result["std_cka"].append(std_val)
            result["ci_lower"].append(mean_val - ci_margin)
            result["ci_upper"].append(mean_val + ci_margin)
            result["min_cka"].append(float(off_diagonal.min()))
            result["max_cka"].append(float(off_diagonal.max()))

        return result

    def find_convergence_layer(
        self,
        threshold: float = 0.75,
        cka_matrices: dict[int, np.ndarray] | None = None,
    ) -> int | None:
        """Identify the layer where cross-lingual CKA exceeds a threshold.

        Scans layers in order and returns the first layer where the
        average cross-lingual CKA is >= ``threshold``. If no layer
        reaches the threshold, returns ``None``.

        For Tiny Aya with 4 layers, the convergence layer is the
        layer with the highest average cross-lingual CKA (since the
        model is shallow and may not reach 0.75).

        Args:
            threshold: CKA threshold for "acceptable alignment"
                (default 0.75, from the reference repo's conventions).
            cka_matrices: CKA matrices to analyze. Defaults to cached.

        Returns:
            Layer index where convergence occurs, or ``None`` if
            the threshold is never reached.
        """
        curve = self.compute_convergence_curve(cka_matrices)

        # First layer above threshold.
        for layer_idx, avg_cka in zip(
            curve["layer_indices"], curve["avg_cka"]
        ):
            if avg_cka >= threshold:
                logger.info(
                    "Convergence detected at layer %d (avg CKA = %.4f >= %.4f)",
                    layer_idx, avg_cka, threshold,
                )
                return layer_idx

        # No layer reached the threshold — report the best layer.
        best_idx = int(np.argmax(curve["avg_cka"]))
        best_layer = curve["layer_indices"][best_idx]
        best_cka = curve["avg_cka"][best_idx]

        logger.warning(
            "No layer reached the CKA threshold of %.4f. "
            "Best layer: %d (avg CKA = %.4f).",
            threshold, best_layer, best_cka,
        )

        return None

    # =================================================================
    # Step 4: Retrieval scoring
    # =================================================================

    def compute_retrieval_scores(
        self,
        source_lang: str = "english",
        k_values: list[int] | None = None,
    ) -> dict[int, dict[str, dict[str, float]]]:
        """Compute translation retrieval metrics at each layer.

        For each layer and each target language, measures how well
        the embedding space supports translation retrieval from the
        source language.

        Args:
            source_lang: Source language for queries (default "english").
            k_values: k values for Recall@k (default [1, 5, 10]).

        Returns:
            Nested dictionary:
            ``{layer_idx: {target_lang: {metric: value}}}``

        Raises:
            RuntimeError: If activations haven't been extracted.
            KeyError: If source language not in corpus.
        """
        if not self.activations:
            raise RuntimeError(
                "No activations cached. Call extract_all_activations() first."
            )

        if source_lang not in self.activations:
            raise KeyError(
                f"Source language '{source_lang}' not in activations."
            )

        if k_values is None:
            k_values = [1, 5, 10]

        results: dict[int, dict[str, dict[str, float]]] = {}

        for layer_idx in self.layer_indices:
            layer_name = f"layer_{layer_idx}"
            layer_results: dict[str, dict[str, float]] = {}

            src_emb = self.activations[source_lang][layer_name].numpy()

            for lang in self.languages:
                if lang.lang_name == source_lang:
                    continue

                tgt_emb = (
                    self.activations[lang.lang_name][layer_name].numpy()
                )

                metrics = compute_all_retrieval_metrics(
                    src_emb, tgt_emb, k_values=k_values
                )
                layer_results[lang.lang_name] = metrics

            results[layer_idx] = layer_results

        return results

    # =================================================================
    # Step 5: Clustering analysis
    # =================================================================

    def compute_clustering_analysis(
        self,
        cka_matrices: dict[int, np.ndarray] | None = None,
    ) -> dict[int, dict]:
        """Compute language family dissolution metrics at each layer.

        Runs hierarchical clustering and family/script group analysis
        on the CKA matrices to track how linguistic groupings evolve.

        Args:
            cka_matrices: CKA matrices to analyze. Defaults to cached.

        Returns:
            Dictionary mapping layer index to a metrics dict containing
            family dissolution, script decomposition, and clustering
            results.
        """
        if cka_matrices is None:
            cka_matrices = self.cka_matrices

        if not cka_matrices:
            raise RuntimeError(
                "No CKA matrices available. Call compute_cka_matrices() first."
            )

        results: dict[int, dict] = {}

        for layer_idx in sorted(cka_matrices.keys()):
            matrix = cka_matrices[layer_idx]

            # Family dissolution metrics.
            family_metrics = compute_family_dissolution_metrics(
                matrix, self.language_names, self.languages
            )

            # Script group metrics.
            script_metrics = compute_script_group_metrics(
                matrix, self.language_names, self.languages
            )

            # Hierarchical clustering.
            clustering = compute_hierarchical_clustering(
                matrix, self.language_names
            )

            results[layer_idx] = {
                "family": family_metrics,
                "script": script_metrics,
                "clustering": clustering,
            }

        return results

    # =================================================================
    # Step 6: Result serialization
    # =================================================================

    def save_activations(self, output_dir: str) -> None:
        """Save cached activations to disk as .pt files.

        Files are named with standardized keys:
        ``layer_{idx}_{lang_name}.pt``

        Args:
            output_dir: Directory to save activation files into.
        """
        path = Path(output_dir) / "activations"
        path.mkdir(parents=True, exist_ok=True)

        for lang_name, layer_acts in self.activations.items():
            for layer_name, tensor in layer_acts.items():
                filename = f"{layer_name}_{lang_name}.pt"
                torch.save(tensor, path / filename)

        logger.info("Activations saved to %s", path)

    def load_activations(self, input_dir: str) -> None:
        """Load previously saved activations from disk.

        Args:
            input_dir: Directory containing activation .pt files.
        """
        path = Path(input_dir) / "activations"

        if not path.exists():
            raise FileNotFoundError(
                f"Activation directory not found: {path}"
            )

        for lang in self.languages:
            lang_name = lang.lang_name
            self.activations[lang_name] = {}

            for layer_idx in self.layer_indices:
                layer_name = f"layer_{layer_idx}"
                filename = f"{layer_name}_{lang_name}.pt"
                filepath = path / filename

                if filepath.exists():
                    self.activations[lang_name][layer_name] = torch.load(
                        filepath, weights_only=True
                    )
                else:
                    logger.warning("Missing activation file: %s", filepath)

        logger.info("Activations loaded from %s", path)

    def save_results(self, output_dir: str) -> None:
        """Save all computed results with standardized keys.

        Saves:
            - Activations as .pt files.
            - CKA matrices as .npy files.
            - Metrics as JSON files with flat, searchable keys.

        The key naming convention follows Wei-Yin Ko's suggestion:
            ``layer_{idx}_{metric_name}``

        Args:
            output_dir: Root directory for all results.
        """
        base_path = Path(output_dir)
        base_path.mkdir(parents=True, exist_ok=True)

        # --- Save activations ---
        if self.activations:
            self.save_activations(output_dir)

        # --- Save CKA matrices ---
        if self.cka_matrices:
            cka_path = base_path / "cka_matrices"
            cka_path.mkdir(parents=True, exist_ok=True)

            for layer_idx, matrix in self.cka_matrices.items():
                np.save(
                    cka_path / f"layer_{layer_idx}_linear_cka.npy",
                    matrix,
                )

        if self.whitened_cka_matrices:
            cka_path = base_path / "cka_matrices"
            cka_path.mkdir(parents=True, exist_ok=True)

            for layer_idx, matrix in self.whitened_cka_matrices.items():
                np.save(
                    cka_path / f"layer_{layer_idx}_whitened_cka.npy",
                    matrix,
                )

        # --- Save convergence metrics ---
        if self.cka_matrices:
            metrics_path = base_path / "metrics"
            metrics_path.mkdir(parents=True, exist_ok=True)

            curve = self.compute_convergence_curve()
            flat_metrics: dict[str, float] = {}

            for i, layer_idx in enumerate(curve["layer_indices"]):
                flat_metrics[f"layer_{layer_idx}_avg_crosslingual_cka"] = (
                    curve["avg_cka"][i]
                )
                flat_metrics[f"layer_{layer_idx}_std_crosslingual_cka"] = (
                    curve["std_cka"][i]
                )

            with open(metrics_path / "convergence_curve.json", "w") as f:
                json.dump(flat_metrics, f, indent=2)

        logger.info("All results saved to %s", base_path)
