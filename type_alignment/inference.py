"""
Multilingual Inference Pipeline for Cross-Lingual Representation Analysis.

Extracts layer-wise hidden states from transformer models to study how
linguistic information (lexical, syntactic, semantic) evolves across layers.

Core workflow:
    1. Load a model (one variant at a time — Global, Regional, etc.)
    2. Feed sentences through the model in batches
    3. At each transformer layer, hooks capture hidden states
    4. Pool token-level states into sentence embeddings
    5. Return structured results with metadata for downstream analysis

Example:
    inferencer = MultilingualInference("CohereLabs/tiny-aya-global")

    results = inferencer.extract(
        sentences=["The cat sleeps", "Le chat dort", "A feline rests"],
        metadata=[
            {"lang": "en", "pair_id": 1, "pair_type": "semantic", "pair_role": "source"},
            {"lang": "fr", "pair_id": 1, "pair_type": "semantic", "pair_role": "target"},
            {"lang": "en", "pair_id": 2, "pair_type": "lexical", "pair_role": "target"},
        ],
    )

    print(results)
    # InferenceResult(sentences=3, layers=32, hidden_dim=4096, model='aya-expanse-8b')

    results.save("./outputs/run_001")
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import h5py
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Config — controls what gets extracted
# ---------------------------------------------------------------------------

@dataclass
class ExtractionConfig:
    """
    Controls extraction behavior.

    Attributes:
        pooling: How to collapse token-level states into sentence vectors.
            "mean" — average over non-padding tokens (default, good for similarity).
            "last" — take the last non-padding token (common for decoder-only models).
        store_token_level: If True, also store full token-level hidden states
            per sentence (padding stripped). Useful for probing tasks later.
            Warning: memory-heavy for long sequences and many layers.
        store_logits: If True, store output logits per sentence.
            Needed if you want perplexity as a secondary metric.
        batch_size: Sentences per forward pass. Tune based on GPU memory.
        max_length: Truncate inputs longer than this.
        layers: Which layer indices to extract from. None = all layers.
            Use this to save memory if you only care about certain layers.
    """

    pooling: str = "mean"
    store_token_level: bool = False
    store_logits: bool = False
    batch_size: int = 8
    max_length: int = 512
    layers: Optional[List[int]] = None


# ---------------------------------------------------------------------------
# Result container — holds everything that comes out of extraction
# ---------------------------------------------------------------------------

@dataclass
class InferenceResult:
    """
    Structured container for extracted representations.

    Core data:
        sentence_embeddings — shape (num_sentences, num_layers, hidden_dim).
            Always populated. This is the main output used for similarity
            computation in all three pair types (lexical/syntactic/semantic).

    Optional data:
        token_embeddings — list of length num_sentences. Each element has
            shape (num_layers, seq_len_i, hidden_dim) where seq_len_i is
            the real (unpadded) length for that sentence. Only populated
            when store_token_level=True.
        logits — list of length num_sentences. Each element has shape
            (seq_len_i, vocab_size). Only when store_logits=True.

    Metadata:
        metadata — list of dicts, one per sentence. Carries fields like
            lang, pair_id, pair_type, pair_role, lang_family, etc.
        model_name — which model variant produced these.
        config — the ExtractionConfig used (as dict for serialization).
    """

    sentence_embeddings: np.ndarray
    metadata: List[Dict[str, Any]]
    model_name: str
    config: Dict[str, Any]

    token_embeddings: Optional[List[np.ndarray]] = None
    logits: Optional[List[np.ndarray]] = None

    # ----------------------------------------------------------------
    # Persistence
    # ----------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Save to disk.

        Creates a directory at `path` containing:
            embeddings.h5   — sentence_embeddings (+ token_embeddings, logits if present)
            metadata.json   — model name, config, per-sentence metadata

        HDF5 uses gzip compression. Token embeddings are stored as individual
        datasets (one per sentence) because they have variable sequence lengths.
        """
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)

        # --- tensors ---
        with h5py.File(out / "embeddings.h5", "w") as f:
            f.create_dataset(
                "sentence_embeddings",
                data=self.sentence_embeddings,
                compression="gzip",
            )

            if self.token_embeddings is not None:
                grp = f.create_group("token_embeddings")
                for i, emb in enumerate(self.token_embeddings):
                    grp.create_dataset(str(i), data=emb, compression="gzip")

            if self.logits is not None:
                grp = f.create_group("logits")
                for i, logit in enumerate(self.logits):
                    grp.create_dataset(str(i), data=logit, compression="gzip")

        # --- metadata ---
        with open(out / "metadata.json", "w") as f:
            json.dump(
                {
                    "model_name": self.model_name,
                    "config": self.config,
                    "num_sentences": len(self.metadata),
                    "num_layers": int(self.sentence_embeddings.shape[1]),
                    "hidden_dim": int(self.sentence_embeddings.shape[2]),
                    "sentences": self.metadata,
                },
                f,
                indent=2,
                ensure_ascii=False,  # preserve non-Latin scripts in metadata
            )

    @classmethod
    def load(cls, path: str) -> "InferenceResult":
        """Load a previously saved result."""
        p = Path(path)

        with open(p / "metadata.json") as f:
            meta = json.load(f)

        with h5py.File(p / "embeddings.h5", "r") as f:
            sentence_embeddings = f["sentence_embeddings"][:]

            token_embeddings = None
            if "token_embeddings" in f:
                n = meta["num_sentences"]
                token_embeddings = [f["token_embeddings"][str(i)][:] for i in range(n)]

            logits = None
            if "logits" in f:
                n = meta["num_sentences"]
                logits = [f["logits"][str(i)][:] for i in range(n)]

        return cls(
            sentence_embeddings=sentence_embeddings,
            metadata=meta["sentences"],
            model_name=meta["model_name"],
            config=meta.get("config", {}),
            token_embeddings=token_embeddings,
            logits=logits,
        )

    # ----------------------------------------------------------------
    # Filtering — for slicing results by metadata fields
    # ----------------------------------------------------------------

    def filter(self, **kwargs) -> "InferenceResult":
        """
        Filter by any metadata field(s).

        Usage:
            english = results.filter(lang="en")
            semantic_pairs = results.filter(pair_type="semantic")
            specific = results.filter(lang="hi", pair_type="syntactic")
        """
        indices = []
        for i, m in enumerate(self.metadata):
            if all(m.get(k) == v for k, v in kwargs.items()):
                indices.append(i)
        return self._subset(indices)

    def get_pair(self, pair_id: int) -> "InferenceResult":
        """Get both sentences in a pair."""
        return self.filter(pair_id=pair_id)

    def get_layer(self, layer_idx: int) -> np.ndarray:
        """
        Get sentence embeddings for one layer.

        Returns shape (num_sentences, hidden_dim).
        """
        extracted_layers = self.config.get("layers")
        if extracted_layers is not None:
            if layer_idx not in extracted_layers:
                raise ValueError(
                    f"Layer {layer_idx} was not extracted. Available: {extracted_layers}"
                )
            pos = extracted_layers.index(layer_idx)
        else:
            pos = layer_idx
        return self.sentence_embeddings[:, pos, :]

    def _subset(self, indices: List[int]) -> "InferenceResult":
        return InferenceResult(
            sentence_embeddings=self.sentence_embeddings[indices],
            metadata=[self.metadata[i] for i in indices],
            model_name=self.model_name,
            config=self.config,
            token_embeddings=(
                [self.token_embeddings[i] for i in indices]
                if self.token_embeddings is not None
                else None
            ),
            logits=(
                [self.logits[i] for i in indices]
                if self.logits is not None
                else None
            ),
        )

    # ----------------------------------------------------------------
    # Info
    # ----------------------------------------------------------------

    @property
    def num_sentences(self) -> int:
        return self.sentence_embeddings.shape[0]

    @property
    def num_layers(self) -> int:
        return self.sentence_embeddings.shape[1]

    @property
    def hidden_dim(self) -> int:
        return self.sentence_embeddings.shape[2]

    @property
    def pair_types(self) -> List[str]:
        """Unique pair_type values in metadata."""
        return sorted(set(m.get("pair_type", "") for m in self.metadata))

    @property
    def languages(self) -> List[str]:
        """Unique language codes in metadata."""
        return sorted(set(m.get("lang", "") for m in self.metadata))

    def summary(self) -> Dict[str, Any]:
        """Quick overview of what's in this result."""
        return {
            "model": self.model_name,
            "sentences": self.num_sentences,
            "layers": self.num_layers,
            "hidden_dim": self.hidden_dim,
            "languages": self.languages,
            "pair_types": self.pair_types,
            "has_token_level": self.token_embeddings is not None,
            "has_logits": self.logits is not None,
        }

    def __len__(self) -> int:
        return self.num_sentences

    def __repr__(self) -> str:
        tok = "yes" if self.token_embeddings is not None else "no"
        model_short = self.model_name.split("/")[-1] if "/" in self.model_name else self.model_name
        return (
            f"InferenceResult(sentences={self.num_sentences}, "
            f"layers={self.num_layers}, hidden_dim={self.hidden_dim}, "
            f"token_level={tok}, model='{model_short}')"
        )


# ---------------------------------------------------------------------------
# Main inference class
# ---------------------------------------------------------------------------

class MultilingualInference:
    """
    Extracts layer-wise hidden representations from a transformer model.

    Wraps one model variant at a time. To compare Global vs Regional,
    instantiate separately:

        global_inf = MultilingualInference("CohereLabs/tiny-aya-global")
        sa_inf     = MultilingualInference("CohereLabs/tiny-aya-fire")
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        trust_remote_code: bool = True,
    ):
        """
        Args:
            model_name: HuggingFace model ID or local path.
            device: "cuda", "cpu", "mps" (Apple Silicon), or "auto" (for device_map="auto").
            dtype: Model precision. float16 is standard for inference.
            trust_remote_code: Needed for some model architectures.
        """
        self.model_name = model_name
        self.device = device
        self.dtype = dtype

        print(f"Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Loading model: {model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device if device == "auto" else None,
            trust_remote_code=trust_remote_code,
        )
        if device != "auto":
            self.model = self.model.to(device)
        self.model.eval()

        # Find transformer layers in the model architecture
        self._layers = self._discover_layers()
        self.num_layers = len(self._layers)
        print(f"Ready: {self.num_layers} layers, hidden_dim={self.hidden_dim}")

        # Internal state — managed per extract() call
        self._hooks: List[torch.utils.hooks.RemovableHook] = []
        self._captured: Dict[int, Dict[str, Any]] = {}
        self._attention_mask: Optional[torch.Tensor] = None
        self._config: Optional[ExtractionConfig] = None

    # ----------------------------------------------------------------
    # Layer discovery — auto-detect where transformer blocks live
    # ----------------------------------------------------------------

    def _discover_layers(self) -> list:
        """
        Auto-detect transformer layer modules.

        Different model families store layers under different attribute paths.
        We try common patterns and return the list of layer modules.
        """
        patterns = [
            ("model.layers", lambda m: m.model.layers),           # Llama, Cohere, Mistral
            ("transformer.h", lambda m: m.transformer.h),         # GPT-2, GPT-Neo
            ("gpt_neox.layers", lambda m: m.gpt_neox.layers),    # GPT-NeoX, Pythia
            ("model.decoder.layers", lambda m: m.model.decoder.layers),  # OPT
        ]
        for name, accessor in patterns:
            try:
                layers = list(accessor(self.model))
                if layers:
                    print(f"  Found layers via '{name}'")
                    return layers
            except AttributeError:
                continue

        raise ValueError(
            f"Cannot auto-detect transformer layers for {self.model_name}. "
            f"Model type: {type(self.model).__name__}. "
            f"Inspect the model and extend _discover_layers()."
        )

    # ----------------------------------------------------------------
    # Pooling — collapse token-level to sentence-level
    # ----------------------------------------------------------------

    def _pool(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Pool token-level hidden states into a single sentence vector.

        Args:
            hidden: (batch, seq_len, hidden_dim) — raw hidden states
            mask:   (batch, seq_len) — 1 for real tokens, 0 for padding

        Returns:
            (batch, hidden_dim) — one vector per sentence
        """
        # Cast to float32 for numerical stability during pooling.
        # Half-precision mean over many tokens can lose accuracy.
        hidden_f32 = hidden.float()
        mask_f32 = mask.unsqueeze(-1).float()  # (batch, seq_len, 1)

        if self._config.pooling == "mean":
            # Sum real tokens, divide by count of real tokens
            summed = (hidden_f32 * mask_f32).sum(dim=1)
            count = mask_f32.sum(dim=1).clamp(min=1e-9)
            return summed / count

        elif self._config.pooling == "last":
            # Take the last non-padding token per sentence.
            last_idx = mask.sum(dim=1) - 1   # (batch,) index of last real token
            batch_idx = torch.arange(hidden.size(0), device=hidden.device)
            return hidden_f32[batch_idx, last_idx]

        raise ValueError(f"Unknown pooling: {self._config.pooling}")

    # ----------------------------------------------------------------
    # Hook machinery
    # ----------------------------------------------------------------

    def _make_hook(self, layer_idx: int):
        """
        Create a forward hook for a transformer layer.

        When the layer runs its forward pass, this hook intercepts the output
        and stores the pooled sentence embedding (and optionally token-level
        states) in self._captured.
        """

        def hook_fn(module, input, output):
            # Most architectures return (hidden_states, ...) as a tuple
            hidden = output[0]  # (batch, seq_len, hidden_dim)

            # Always compute and store sentence-level pooled embedding
            pooled = self._pool(hidden, self._attention_mask)
            self._captured[layer_idx] = {
                "sentence": pooled.detach().cpu(),
            }

            # Optionally store token-level (padding stripped per sentence)
            if self._config.store_token_level:
                h_cpu = hidden.detach().cpu().float()
                m_cpu = self._attention_mask.cpu()
                token_states = []
                for i in range(h_cpu.size(0)):
                    real_len = int(m_cpu[i].sum().item())
                    token_states.append(h_cpu[i, :real_len].numpy())
                self._captured[layer_idx]["token"] = token_states

        return hook_fn

    def _register_hooks(self, config: ExtractionConfig) -> List[int]:
        """
        Register forward hooks on target layers.

        Returns the list of layer indices being hooked.
        """
        self._remove_hooks()
        target = config.layers if config.layers is not None else list(range(self.num_layers))

        for idx in target:
            if idx < 0 or idx >= self.num_layers:
                raise ValueError(f"Layer {idx} out of range [0, {self.num_layers})")
            hook = self._layers[idx].register_forward_hook(self._make_hook(idx))
            self._hooks.append(hook)

        return target

    def _remove_hooks(self):
        """Remove all registered hooks and clear captured data."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._captured.clear()

    # ----------------------------------------------------------------
    # Main extraction method
    # ----------------------------------------------------------------

    def extract(
        self,
        sentences: List[str],
        metadata: List[Dict[str, Any]],
        config: Optional[ExtractionConfig] = None,
    ) -> InferenceResult:
        """
        Run sentences through the model and extract layer-wise representations.

        Args:
            sentences: List of input texts.
            metadata:  One dict per sentence. Expected fields:
                         lang       — language code ("en", "hi", "ar", ...)
                         pair_id    — links the two sentences in a pair
                         pair_type  — "lexical" | "syntactic" | "semantic"
                         pair_role  — "source" | "target"
                       Optional fields:
                         lang_family — language family ("germanic", "indo-aryan", ...)
                         align_id   — links translations of the same content
                         (any other fields you want — they're passed through)
            config:    Extraction settings. Uses defaults if None.

        Returns:
            InferenceResult with sentence_embeddings of shape
            (num_sentences, num_extracted_layers, hidden_dim).
        """
        if len(sentences) != len(metadata):
            raise ValueError(
                f"Length mismatch: {len(sentences)} sentences, {len(metadata)} metadata"
            )

        config = config or ExtractionConfig()
        self._config = config
        target_layers = (
            config.layers if config.layers is not None else list(range(self.num_layers))
        )

        # -- accumulators --
        all_sentence_embs: Dict[int, List[torch.Tensor]] = {l: [] for l in target_layers}
        all_token_embs: Optional[Dict[int, list]] = (
            {l: [] for l in target_layers} if config.store_token_level else None
        )
        all_logits: Optional[list] = [] if config.store_logits else None

        num_batches = (len(sentences) + config.batch_size - 1) // config.batch_size
        model_short = self.model_name.split("/")[-1]

        for start in tqdm(
            range(0, len(sentences), config.batch_size),
            total=num_batches,
            desc=f"Extracting [{model_short}]",
        ):
            end = min(start + config.batch_size, len(sentences))
            batch_texts = sentences[start:end]

            # Tokenize
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=config.max_length,
                return_tensors="pt",
            ).to(self.device)

            mask = encoded["attention_mask"]

            # Use output_hidden_states=True instead of forward hooks.
            # Hooks are unreliable on Cohere2 (and similar models) because
            # KV cache accumulation across batches causes the hidden tensor's
            # batch and sequence dimensions to drift from the attention mask.
            # output_hidden_states returns tensors with guaranteed shapes
            # (batch, seq_len, hidden_dim) that always match the input.
            with torch.no_grad():
                outputs = self.model(
                    **encoded,
                    output_hidden_states=True,
                    use_cache=False,
                )

            # outputs.hidden_states: tuple of (num_layers + 1) tensors.
            # Index 0 = embedding layer output; index i+1 = decoder layer i output.
            hs_all = outputs.hidden_states  # tuple[(batch, seq, hidden_dim), ...]

            for layer_idx in target_layers:
                hs = hs_all[layer_idx + 1]  # (batch, seq_len, hidden_dim)
                pooled = self._pool(hs, mask)
                all_sentence_embs[layer_idx].append(pooled.detach().cpu())

                if all_token_embs is not None:
                    h_cpu = hs.detach().cpu().float()
                    m_cpu = mask.cpu()
                    for i in range(h_cpu.size(0)):
                        real_len = int(m_cpu[i].sum().item())
                        all_token_embs[layer_idx].append(h_cpu[i, :real_len].numpy())

            # Collect logits if requested
            if all_logits is not None:
                logits_cpu = outputs.logits.detach().cpu()
                mask_cpu = mask.cpu()
                for i in range(logits_cpu.size(0)):
                    real_len = int(mask_cpu[i].sum().item())
                    all_logits.append(logits_cpu[i, :real_len].numpy())

        # -- assemble sentence embeddings --
        # Concatenate across batches per layer, then stack layers
        # Result: (num_sentences, num_layers, hidden_dim)
        per_layer = [torch.cat(all_sentence_embs[l], dim=0) for l in target_layers]
        sentence_embeddings = torch.stack(per_layer, dim=1).numpy()

        # -- assemble token embeddings --
        token_embeddings = None
        if all_token_embs is not None:
            n = len(sentences)
            token_embeddings = []
            for i in range(n):
                stacked = np.stack(
                    [all_token_embs[l][i] for l in target_layers], axis=0
                )  # (num_layers, seq_len_i, hidden_dim)
                token_embeddings.append(stacked)

        return InferenceResult(
            sentence_embeddings=sentence_embeddings,
            metadata=metadata,
            model_name=self.model_name,
            config={
                "pooling": config.pooling,
                "store_token_level": config.store_token_level,
                "store_logits": config.store_logits,
                "batch_size": config.batch_size,
                "max_length": config.max_length,
                "layers": target_layers,
            },
            token_embeddings=token_embeddings,
            logits=all_logits if all_logits else None,
        )

    # ----------------------------------------------------------------
    # Info
    # ----------------------------------------------------------------

    @property
    def hidden_dim(self) -> int:
        return self.model.config.hidden_size

    def __repr__(self) -> str:
        return (
            f"MultilingualInference(\n"
            f"  model='{self.model_name}',\n"
            f"  layers={self.num_layers},\n"
            f"  hidden_dim={self.hidden_dim},\n"
            f"  device='{self.device}'\n"
            f")"
        )


# ---------------------------------------------------------------------------
# Runnable smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    # ---- Configuration ----
    MODEL_NAME = "CohereLabs/tiny-aya-global"
    SAVE_PATH = "./test_extraction"

    # Auto-detect device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # ---- Test sentences ----
    # Small set covering all three pair types for a quick validation.

    sentences = [
        # Semantic pair (same meaning, different words)
        "The meeting was cancelled.",
        "The meeting did not take place.",
        # Syntactic pair (same structure, different content)
        "The dog chased the cat.",
        "The boy kicked the ball.",
        # Lexical pair (shared words, different meaning)
        "The dog bit the man.",
        "The man bit the dog.",
    ]

    metadata = [
        {"lang": "en", "pair_id": 0, "pair_type": "semantic",  "pair_role": "source"},
        {"lang": "en", "pair_id": 0, "pair_type": "semantic",  "pair_role": "target"},
        {"lang": "en", "pair_id": 1, "pair_type": "syntactic", "pair_role": "source"},
        {"lang": "en", "pair_id": 1, "pair_type": "syntactic", "pair_role": "target"},
        {"lang": "en", "pair_id": 2, "pair_type": "lexical",   "pair_role": "source"},
        {"lang": "en", "pair_id": 2, "pair_type": "lexical",   "pair_role": "target"},
    ]

    # ---- Run extraction ----
    print(f"\nLoading {MODEL_NAME}...")
    inferencer = MultilingualInference(
        model_name=MODEL_NAME,
        device=device,
        dtype=torch.float16,
    )

    print(f"\n{inferencer}\n")

    config = ExtractionConfig(
        pooling="mean",
        batch_size=4,
    )

    results = inferencer.extract(
        sentences=sentences,
        metadata=metadata,
        config=config,
    )

    # ---- Inspect results ----
    print(f"\n{results}")
    print(f"\nSummary: {results.summary()}")
    print(f"\nEmbedding shape: {results.sentence_embeddings.shape}")

    # Quick similarity check — cosine between each pair at layer 0 and last layer
    from numpy.linalg import norm

    def cosine_sim(a, b):
        return np.dot(a, b) / (norm(a) * norm(b) + 1e-9)

    last_layer = results.num_layers - 1
    print(f"\nPairwise cosine similarity (layer 0 vs layer {last_layer}):")
    print(f"{'Pair Type':<12} {'Layer 0':>10} {'Layer ' + str(last_layer):>10}")
    print("-" * 35)

    for pair_type in ["semantic", "syntactic", "lexical"]:
        subset = results.filter(pair_type=pair_type)
        emb_first = subset.get_layer(0)
        emb_last = subset.get_layer(last_layer)
        sim_first = cosine_sim(emb_first[0], emb_first[1])
        sim_last = cosine_sim(emb_last[0], emb_last[1])
        print(f"{pair_type:<12} {sim_first:>10.4f} {sim_last:>10.4f}")

    # ---- Save ----
    results.save(SAVE_PATH)
    print(f"\nSaved to {SAVE_PATH}/")

    # ---- Verify load ----
    loaded = InferenceResult.load(SAVE_PATH)
    print(f"Loaded back: {loaded}")
    assert np.allclose(loaded.sentence_embeddings, results.sentence_embeddings)
    print("Save/load verified ✓")