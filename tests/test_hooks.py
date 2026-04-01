"""
Tests for the activation hooks module (src.analysis.hooks).

Uses a simple mock model to validate hook registration, activation
capture, mean-pooling, and cleanup behavior without requiring
a GPU or a real transformer model.
"""

import pytest
import torch
import torch.nn as nn

from src.analysis.cross_lingual_embedding_alignment.hooks import (
    ActivationStore,
    _find_transformer_layers,
    get_model_layer_count,
    register_model_hooks,
)

# ===================================================================
# Mock model for testing
# ===================================================================

class MockTransformerLayer(nn.Module):
    """Simple mock transformer layer that outputs (hidden_states,)."""

    def __init__(self, hidden_dim: int = 32) -> None:
        """Initialize with a single linear layer of *hidden_dim* units."""
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Return a tuple (hidden_states,) like real transformer layers."""
        return (self.linear(x),)


class MockModel(nn.Module):
    """Mock HuggingFace-style model with model.model.layers structure."""

    def __init__(self, n_layers: int = 4, hidden_dim: int = 32) -> None:
        """Create a mock model with *n_layers* transformer layers."""
        super().__init__()
        # Mimic the CohereForCausalLM structure:
        # model.model.layers[i]
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [MockTransformerLayer(hidden_dim) for _ in range(n_layers)]
        )
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Simple forward pass through all layers."""
        for layer in self.model.layers:
            x = layer(x)[0]
        return x


# ===================================================================
# Tests for ActivationStore
# ===================================================================

class TestActivationStore:
    """Tests for the ActivationStore class."""

    def test_register_and_collect(self) -> None:
        """Should capture activations from registered hooks."""
        store = ActivationStore(detach=True, device="cpu")
        layer = nn.Linear(16, 16)
        store.register(layer, "test_layer")

        # Run forward pass.
        x = torch.randn(4, 16)
        layer(x)

        # Collect should return the captured activations.
        result = store.collect()
        assert "test_layer" in result
        assert result["test_layer"].shape[0] == 4

        store.remove_hooks()

    def test_multiple_forward_passes_accumulate(self) -> None:
        """Multiple forward passes should accumulate in the buffer."""
        store = ActivationStore(detach=True, device="cpu")
        layer = nn.Linear(16, 16)
        store.register(layer, "test_layer")

        # Two forward passes.
        layer(torch.randn(4, 16))
        layer(torch.randn(6, 16))

        result = store.collect()
        assert result["test_layer"].shape[0] == 10  # 4 + 6

        store.remove_hooks()

    def test_clear_empties_buffers(self) -> None:
        """Clear should empty all buffers but keep hooks."""
        store = ActivationStore(detach=True, device="cpu")
        layer = nn.Linear(16, 16)
        store.register(layer, "test_layer")

        layer(torch.randn(4, 16))
        store.clear()

        result = store.collect()
        assert len(result) == 0

        # Hooks should still be active.
        layer(torch.randn(4, 16))
        result = store.collect()
        assert "test_layer" in result

        store.remove_hooks()

    def test_remove_hooks_cleans_up(self) -> None:
        """After remove_hooks, no new activations should be captured."""
        store = ActivationStore(detach=True, device="cpu")
        layer = nn.Linear(16, 16)
        store.register(layer, "test_layer")

        layer(torch.randn(4, 16))
        store.remove_hooks()

        # New forward pass should not be captured.
        layer(torch.randn(4, 16))
        result = store.collect()
        assert len(result) == 0

    def test_detach_prevents_gradient_tracking(self) -> None:
        """With detach=True, captured activations should not require grad."""
        store = ActivationStore(detach=True, device="cpu")
        layer = nn.Linear(16, 16)
        store.register(layer, "test_layer")

        x = torch.randn(4, 16, requires_grad=True)
        layer(x)

        result = store.collect()
        assert not result["test_layer"].requires_grad

        store.remove_hooks()

    def test_handles_3d_output(self) -> None:
        """Should handle 3D tensor outputs (batch, seq_len, hidden)."""
        store = ActivationStore(detach=True, device="cpu")

        # Create a module that outputs 3D tensors.
        class Mock3D(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x  # Pass through

        module = Mock3D()
        store.register(module, "test_3d")

        # 3D input: (batch=2, seq_len=5, hidden=16)
        x = torch.randn(2, 5, 16)
        module(x)

        result = store.collect()
        assert "test_3d" in result
        # Should preserve 3D shape for mean-pooling support.
        assert result["test_3d"].shape == (2, 5, 16)

        store.remove_hooks()

    def test_mean_pooled_collection(self) -> None:
        """Should correctly mean-pool over non-padding tokens."""
        store = ActivationStore(detach=True, device="cpu")

        class PassThrough(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x

        module = PassThrough()
        store.register(module, "test")

        # Input: (batch=2, seq_len=4, hidden=8)
        x = torch.ones(2, 4, 8)
        # Mask: first sentence has 3 real tokens, second has 2.
        mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.float32)

        store.store_attention_mask(mask)
        module(x)

        result = store.collect_mean_pooled()
        assert "test" in result
        assert result["test"].shape == (2, 8)
        # All values are 1.0, so mean-pool should also be 1.0.
        assert torch.allclose(result["test"], torch.ones(2, 8))

        store.remove_hooks()

    def test_mean_pooled_raises_without_masks(self) -> None:
        """Should raise RuntimeError if no masks stored."""
        store = ActivationStore()
        with pytest.raises(RuntimeError, match="No attention masks"):
            store.collect_mean_pooled()


# ===================================================================
# Tests for model hook registration
# ===================================================================

class TestRegisterModelHooks:
    """Tests for register_model_hooks with the mock model."""

    def test_hooks_all_layers(self) -> None:
        """Should hook all layers when layer_indices=None."""
        model = MockModel(n_layers=4, hidden_dim=32)
        store = ActivationStore(detach=True, device="cpu")

        n_hooks = register_model_hooks(model, store, layer_indices=None)
        assert n_hooks == 4

        store.remove_hooks()

    def test_hooks_specific_layers(self) -> None:
        """Should only hook requested layers."""
        model = MockModel(n_layers=4, hidden_dim=32)
        store = ActivationStore(detach=True, device="cpu")

        n_hooks = register_model_hooks(model, store, layer_indices=[0, 3])
        assert n_hooks == 2

        store.remove_hooks()

    def test_captures_activations_from_forward(self) -> None:
        """Hooks should capture activations during forward pass."""
        model = MockModel(n_layers=2, hidden_dim=16)
        store = ActivationStore(detach=True, device="cpu")

        register_model_hooks(model, store, layer_indices=[0, 1])

        # Forward pass.
        x = torch.randn(3, 5, 16)  # (batch, seq_len, hidden)
        model(x)

        result = store.collect()
        assert "layer_0" in result
        assert "layer_1" in result

        store.remove_hooks()

    def test_raises_for_out_of_range_layer(self) -> None:
        """Should raise IndexError for invalid layer indices."""
        model = MockModel(n_layers=4, hidden_dim=32)
        store = ActivationStore()

        with pytest.raises(IndexError, match="out of range"):
            register_model_hooks(model, store, layer_indices=[10])

    def test_raises_for_non_transformer_model(self) -> None:
        """Should raise ValueError if layers cannot be found."""
        model = nn.Linear(10, 10)  # Not a transformer.
        store = ActivationStore()

        with pytest.raises(ValueError, match="Could not locate"):
            register_model_hooks(model, store)


class TestFindTransformerLayers:
    """Tests for the layer-finding utility."""

    def test_finds_cohere_style_layers(self) -> None:
        """Should find model.model.layers."""
        model = MockModel(n_layers=3)
        layers = _find_transformer_layers(model)
        assert layers is not None
        assert len(layers) == 3

    def test_returns_none_for_plain_module(self) -> None:
        """Should return None for a model without known layer paths."""
        model = nn.Linear(10, 10)
        assert _find_transformer_layers(model) is None


class TestGetModelLayerCount:
    """Tests for get_model_layer_count."""

    def test_returns_correct_count(self) -> None:
        """Should return the number of transformer layers."""
        model = MockModel(n_layers=4)
        assert get_model_layer_count(model) == 4

    def test_raises_for_non_transformer(self) -> None:
        """Should raise ValueError if layers not found."""
        with pytest.raises(ValueError, match="Cannot find"):
            get_model_layer_count(nn.Linear(10, 10))
