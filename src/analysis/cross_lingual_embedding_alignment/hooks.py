"""
Activation extraction utilities for HuggingFace transformer models.

This module provides forward hook-based activation capture for
extracting intermediate hidden states from Tiny Aya (CohereForCausalLM)
and other HuggingFace transformer architectures. Activations are
extracted without modifying the model's forward pass, collected
layer-by-layer, and optionally mean-pooled over non-padding tokens
to produce sentence-level embeddings.

Design goals:
    - **Memory efficiency**: Activations are detached from the
      computation graph and moved to CPU by default, freeing GPU
      memory for the next forward pass.
    - **Flexibility**: Hooks can target specific layers, and
      extraction supports both token-level and sentence-level
      (mean-pooled) representations.
    - **Robustness**: Handles various output formats from different
      transformer architectures (tuples, tensors, nested structures).
    - **Clean lifecycle**: Hooks are automatically removed when the
      ``ActivationStore`` is garbage collected or when
      ``remove_hooks()`` is called explicitly.

Architecture notes for Tiny Aya Global:
    - Model: ``CohereLabs/tiny-aya-global`` (3.35B params)
    - Structure: ``model.model.layers[i]`` (4 transformer layers)
    - Layer 0-2: Sliding window attention + MLP
    - Layer 3: Global attention + MLP
    - Hidden dim: 3072
    - Tokenizer: CohereTokenizer (BPE)

Usage::

    from src.analysis.cross_lingual_embedding_alignment.hooks import ActivationStore, register_model_hooks

    store = ActivationStore(detach=True, device="cpu")
    register_model_hooks(model, store, layer_indices=[0, 1, 2, 3])

    with torch.no_grad():
        model(**inputs)

    activations = store.collect()  # {"layer_0": tensor, "layer_1": ...}
    store.remove_hooks()

References:
    - PyTorch hooks: https://pytorch.org/docs/stable/generated/torch.nn.Module.register_forward_hook.html
    - Inspired by: github.com/Wayy-Research/project-aya (aya_distill.distill.hooks)
"""


import logging
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

# Module-level logger.
logger = logging.getLogger(__name__)


# ===================================================================
# Activation Store
# ===================================================================

@dataclass
class ActivationStore:
    """Collects activations from registered forward hooks.

    Each hook appends activation tensors to an internal buffer keyed
    by a user-defined layer name. After one or more forward passes,
    call ``collect()`` to concatenate all buffered tensors into single
    ``(total_tokens, hidden_dim)`` matrices per layer, then ``clear()``
    to free the buffer memory.

    The store supports two usage patterns:

    1. **Token-level extraction**: Capture raw ``(batch * seq_len, d)``
       activations for fine-grained analysis.

    2. **Sentence-level extraction**: Use ``collect_mean_pooled()``
       with an attention mask to get ``(n_sentences, d)`` mean-pooled
       embeddings, which are more robust for cross-lingual alignment.

    Attributes:
        detach: If True (default), detach activations from the
            computation graph to prevent gradient tracking.
        device: Device to store collected activations on. Defaults
            to "cpu" to free GPU memory during extraction.

    Lifecycle:
        1. Create an ``ActivationStore`` instance.
        2. Register hooks via ``register()`` or ``register_model_hooks()``.
        3. Run forward passes through the model.
        4. Call ``collect()`` or ``collect_mean_pooled()`` to get results.
        5. Call ``clear()`` to free buffers (hooks remain active).
        6. Call ``remove_hooks()`` to fully clean up (or let GC handle it).
    """

    detach: bool = True
    device: str = "cpu"

    # Internal state: not exposed in repr for cleanliness.
    _buffers: dict[str, list[torch.Tensor]] = field(
        default_factory=dict, repr=False
    )
    _hooks: list[torch.utils.hooks.RemovableHandle] = field(
        default_factory=list, repr=False
    )
    _attention_masks: list[torch.Tensor] = field(
        default_factory=list, repr=False
    )

    def _make_hook(
        self, name: str
    ) -> Callable[[nn.Module, Any, Any], None]:
        """Create a forward hook function for a specific layer.

        The returned hook captures the layer's output tensor, handles
        various output formats (tuple, tensor, etc.), reshapes 3D
        outputs to 2D, and stores the result in the internal buffer.

        Args:
            name: Unique identifier for this layer (e.g., "layer_0").
                Used as the dictionary key in ``collect()`` output.

        Returns:
            A hook function compatible with PyTorch's
            ``register_forward_hook`` API.
        """
        # Use closure to capture ``self`` and ``name``.
        store = self

        def hook_fn(
            module: nn.Module,
            input: Any,  # noqa: A002 — shadowing builtin is PyTorch convention
            output: Any,
        ) -> None:
            """Forward hook that captures layer output activations.

            Handles multiple output formats:
                - ``torch.Tensor``: used directly.
                - ``tuple``: first element taken (standard for
                  transformer layers that return (hidden, attn, ...)).
                - Other: logged as warning and skipped.
            """
            # --- Extract the activation tensor from the output ---
            if isinstance(output, torch.Tensor):
                activation = output
            elif isinstance(output, tuple):
                # Transformer layers typically return
                # (hidden_states, attention_weights, ...).
                activation = output[0]
            else:
                warnings.warn(
                    f"ActivationStore: Unexpected output type "
                    f"{type(output).__name__} from layer '{name}'. "
                    f"Skipping activation capture for this forward pass."
                )
                return

            # --- Detach from computation graph if requested ---
            if store.detach:
                activation = activation.detach()

            # --- Reshape 3D -> 2D: (batch, seq_len, d) -> (batch*seq, d) ---
            # We keep the batch dimension info via separate tracking
            # in _attention_masks for mean-pooling later.
            if activation.dim() == 3:
                # Store the un-flattened tensor for mean-pooling support.
                # Shape: (batch_size, seq_len, hidden_dim)
                pass
            elif activation.dim() == 2:
                # Already 2D — this happens with some custom layers.
                pass
            else:
                warnings.warn(
                    f"ActivationStore: Unexpected activation shape "
                    f"{tuple(activation.shape)} from layer '{name}'. "
                    f"Expected 2D or 3D tensor."
                )
                return

            # --- Move to storage device for memory efficiency ---
            activation = activation.to(store.device, dtype=torch.float32)

            # --- Append to buffer ---
            if name not in store._buffers:
                store._buffers[name] = []
            store._buffers[name].append(activation)

        return hook_fn

    def register(self, module: nn.Module, name: str) -> None:
        """Register a forward hook on a specific module.

        After registration, every forward pass through ``module``
        will capture its output activation in the internal buffer
        under the key ``name``.

        Args:
            module: The PyTorch module to hook (e.g., a transformer
                layer or attention sub-module).
            name: Unique string identifier for this hook. Used as
                the key when retrieving activations via ``collect()``.
        """
        hook = module.register_forward_hook(self._make_hook(name))
        self._hooks.append(hook)
        logger.debug("Registered hook '%s' on %s", name, type(module).__name__)

    def store_attention_mask(self, attention_mask: torch.Tensor) -> None:
        """Store an attention mask for later mean-pooling.

        Call this before or after each forward pass to associate
        the attention mask with the captured activations. The masks
        are stored in order and matched with activations by index.

        Args:
            attention_mask: Binary mask of shape ``(batch_size, seq_len)``
                where 1 indicates real tokens and 0 indicates padding.
        """
        self._attention_masks.append(
            attention_mask.to(self.device, dtype=torch.float32)
        )

    def collect(self) -> dict[str, torch.Tensor]:
        """Concatenate buffered activations into single tensors.

        For 3D activations (batch, seq_len, d), tensors are
        concatenated along the batch dimension, preserving the
        sequence structure for potential mean-pooling later.

        Returns:
            Dictionary mapping layer names to concatenated activation
            tensors. Shape depends on the original layer output:
                - 3D layers: ``(total_batches, seq_len, d)``
                - 2D layers: ``(total_samples, d)``

        Note:
            This does NOT clear the buffers. Call ``clear()``
            separately if you want to reuse the hooks for another
            round of extraction.
        """
        result: dict[str, torch.Tensor] = {}

        for name, tensor_list in self._buffers.items():
            if tensor_list:
                result[name] = torch.cat(tensor_list, dim=0)

        return result

    def collect_mean_pooled(self) -> dict[str, torch.Tensor]:
        """Collect activations with mean-pooling over non-padding tokens.

        For each layer, applies attention-mask-weighted mean pooling
        to produce sentence-level embeddings of shape
        ``(n_sentences, hidden_dim)``. This is the recommended
        extraction method for cross-lingual alignment analysis,
        as mean pooling is more robust than CLS-token extraction
        for decoder-only models like Tiny Aya.

        The pooling formula is:
            embedding[i] = sum(activation[i] * mask[i]) / sum(mask[i])

        where mask[i] is the attention mask for sentence i, expanded
        to match the hidden dimension.

        Returns:
            Dictionary mapping layer names to mean-pooled activation
            tensors of shape ``(n_sentences, hidden_dim)``.

        Raises:
            RuntimeError: If no attention masks have been stored
                (call ``store_attention_mask()`` during extraction).
        """
        if not self._attention_masks:
            raise RuntimeError(
                "ActivationStore.collect_mean_pooled: No attention masks "
                "stored. Call store_attention_mask(mask) for each forward "
                "pass during extraction."
            )

        result: dict[str, torch.Tensor] = {}

        # Concatenate all attention masks: shape (total_sentences, seq_len)
        all_masks = torch.cat(self._attention_masks, dim=0)

        for name, tensor_list in self._buffers.items():
            if not tensor_list:
                continue

            # Concatenate all batches: shape (total_sentences, seq_len, d)
            all_activations = torch.cat(tensor_list, dim=0)

            if all_activations.dim() == 3:
                # Mean-pool over the sequence dimension using the mask.
                # Expand mask: (n, seq) -> (n, seq, 1) for broadcasting.
                mask_expanded = all_masks.unsqueeze(-1)

                # Weighted sum of activations (zero out padding positions).
                masked_sum = (all_activations * mask_expanded).sum(dim=1)

                # Count of real tokens per sentence (avoid division by zero).
                token_counts = all_masks.sum(dim=1, keepdim=True).clamp(min=1)

                # Mean-pooled embeddings: (n_sentences, hidden_dim)
                result[name] = masked_sum / token_counts

            elif all_activations.dim() == 2:
                # Already 2D — no pooling needed (or possible).
                result[name] = all_activations
            else:
                warnings.warn(
                    f"ActivationStore.collect_mean_pooled: Unexpected "
                    f"shape {tuple(all_activations.shape)} for layer "
                    f"'{name}'. Skipping."
                )

        return result

    def clear(self) -> None:
        """Free all activation buffers while keeping hooks registered.

        Call this between extraction rounds if you want to reuse the
        same hooks for different inputs. Attention masks are also
        cleared.
        """
        for tensor_list in self._buffers.values():
            tensor_list.clear()
        self._buffers.clear()
        self._attention_masks.clear()

    def remove_hooks(self) -> None:
        """Remove all registered hooks and free all buffers.

        This is the full cleanup method. After calling this, the
        store cannot capture any more activations unless new hooks
        are registered.
        """
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self.clear()
        logger.debug("All hooks removed and buffers cleared.")

    def __del__(self) -> None:
        """Ensure hooks are cleaned up on garbage collection."""
        self.remove_hooks()


# ===================================================================
# Hook registration for HuggingFace models
# ===================================================================

def register_model_hooks(
    model: nn.Module,
    store: ActivationStore,
    layer_indices: list[int] | None = None,
    hook_type: str = "layer_output",
) -> int:
    """Register forward hooks on a HuggingFace transformer model.

    Automatically locates the transformer layer list in the model
    and registers hooks to capture the output of specified layers.
    Supports multiple HuggingFace architectures:

        - Cohere / Llama / Mistral: ``model.model.layers[i]``
        - GPT-2 / GPT-Neo: ``model.transformer.h[i]``
        - Direct models (no LM head): ``model.layers[i]``

    For Tiny Aya Global (``CohereForCausalLM``), the layer structure
    is ``model.model.layers[0..3]`` with 4 transformer layers.

    Args:
        model: HuggingFace causal language model. Must have a
            discoverable ``layers`` attribute (see supported
            architectures above).
        store: ``ActivationStore`` instance to collect activations
            into. Must be created before calling this function.
        layer_indices: Which layers to hook, as a list of zero-based
            indices. If ``None``, hooks ALL layers in the model.
            For Tiny Aya: ``[0, 1, 2, 3]`` hooks all 4 layers.
        hook_type: What to hook — currently only "layer_output" is
            supported (captures the output of each transformer block).

    Returns:
        The number of hooks successfully registered.

    Raises:
        ValueError: If the transformer layers cannot be located in
            the model architecture, or if an invalid ``hook_type``
            is specified.
        IndexError: If a requested layer index is out of range.

    Example::

        >>> store = ActivationStore()
        >>> n_hooks = register_model_hooks(model, store, layer_indices=[0, 3])
        >>> print(f"Registered {n_hooks} hooks")
        Registered 2 hooks
    """
    if hook_type != "layer_output":
        raise ValueError(
            f"register_model_hooks: Unknown hook_type '{hook_type}'. "
            f"Currently only 'layer_output' is supported."
        )

    # --- Locate the transformer layer list ---
    layers = _find_transformer_layers(model)

    if layers is None:
        raise ValueError(
            "register_model_hooks: Could not locate transformer layers "
            "in the model. Expected one of: model.model.layers, "
            "model.transformer.h, or model.layers. "
            f"Model type: {type(model).__name__}"
        )

    n_layers = len(layers)
    logger.info(
        "Found %d transformer layers in %s",
        n_layers,
        type(model).__name__,
    )

    # --- Default to all layers if none specified ---
    if layer_indices is None:
        layer_indices = list(range(n_layers))

    # --- Validate layer indices ---
    for idx in layer_indices:
        if idx < 0 or idx >= n_layers:
            raise IndexError(
                f"register_model_hooks: Layer index {idx} is out of "
                f"range for model with {n_layers} layers. "
                f"Valid range: [0, {n_layers - 1}]."
            )

    # --- Register hooks ---
    hooks_registered = 0
    for idx in layer_indices:
        layer = layers[idx]
        layer_name = f"layer_{idx}"
        store.register(layer, layer_name)
        hooks_registered += 1

    logger.info("Registered %d hooks on layers %s", hooks_registered, layer_indices)

    return hooks_registered


def _find_transformer_layers(model: nn.Module) -> nn.ModuleList | None:
    """Locate the main transformer layer list in a HuggingFace model.

    Tries several known attribute paths used by different HuggingFace
    model architectures. Returns the first match, or ``None`` if no
    known pattern is found.

    Args:
        model: The HuggingFace model to inspect.

    Returns:
        The ``nn.ModuleList`` of transformer layers, or ``None``.
    """
    # Cohere / Llama / Mistral / Qwen path:
    # model -> model.model -> model.model.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers

    # GPT-2 / GPT-Neo path:
    # model -> model.transformer -> model.transformer.h
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h

    # Direct model without LM head wrapper:
    # model -> model.layers
    if hasattr(model, "layers") and isinstance(model.layers, nn.ModuleList):
        return model.layers

    return None


def get_model_layer_count(model: nn.Module) -> int:
    """Return the number of transformer layers in a model.

    Convenience function that locates the layer list and returns
    its length, useful for determining valid layer indices.

    Args:
        model: HuggingFace transformer model.

    Returns:
        Number of transformer layers.

    Raises:
        ValueError: If transformer layers cannot be found.
    """
    layers = _find_transformer_layers(model)
    if layers is None:
        raise ValueError(
            f"Cannot find transformer layers in {type(model).__name__}."
        )
    return len(layers)


# ===================================================================
# Model loading utility
# ===================================================================

def load_model(
    model_name: str = "CohereLabs/tiny-aya-global",
    precision: str = "fp16",
    device_map: str = "auto",
) -> tuple:
    """Load a HuggingFace causal language model with tokenizer.

    Supports two precision modes:
        - ``"fp16"``: Half-precision floating point. Requires ~6.7GB
          VRAM for Tiny Aya Global. Recommended for RTX 2070+ GPUs.
        - ``"4bit"``: 4-bit quantization via bitsandbytes. Requires
          ~1.7GB VRAM. Useful for memory-constrained environments.

    Args:
        model_name: HuggingFace model identifier. Defaults to
            ``"CohereLabs/tiny-aya-global"``.
        precision: Loading precision — ``"fp16"`` (default) or
            ``"4bit"``.
        device_map: Device placement strategy. ``"auto"`` distributes
            across available GPUs/CPU automatically.

    Returns:
        Tuple of ``(model, tokenizer)``:
            - ``model``: The loaded causal LM in eval mode.
            - ``tokenizer``: The corresponding tokenizer with
              padding configured.

    Raises:
        ImportError: If required libraries are not installed (e.g.,
            ``bitsandbytes`` for 4-bit mode).
        ValueError: If ``precision`` is not "fp16" or "4bit".

    Example::

        >>> model, tokenizer = load_model(precision="fp16")
        >>> print(type(model).__name__)
        CohereForCausalLM
    """
    # Lazy imports to avoid hard dependency at module load time.
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "The 'transformers' library is required. "
            "Install it with: pip install transformers"
        ) from exc

    logger.info("Loading model '%s' with precision='%s'...", model_name, precision)

    # --- Configure precision-specific loading kwargs ---
    model_kwargs: dict[str, Any] = {
        "device_map": device_map,
        "trust_remote_code": False,
    }

    if precision == "fp16":
        model_kwargs["torch_dtype"] = torch.float16
    elif precision == "4bit":
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise ImportError(
                "4-bit quantization requires 'bitsandbytes'. "
                "Install it with: pip install bitsandbytes"
            ) from exc

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
    else:
        raise ValueError(
            f"load_model: precision must be 'fp16' or '4bit', "
            f"got '{precision}'."
        )

    # --- Load model ---
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.eval()

    # --- Load tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Ensure the tokenizer has a pad token (some models don't set one).
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Enable left-padding for decoder-only models (standard practice
    # for batch inference with generation models).
    tokenizer.padding_side = "left"

    logger.info(
        "Model loaded: %s, %d layers, hidden_dim=%s",
        type(model).__name__,
        get_model_layer_count(model),
        getattr(model.config, "hidden_size", "unknown"),
    )

    return model, tokenizer
