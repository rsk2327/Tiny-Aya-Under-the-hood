"""Simple layer intervention system using PyTorch hooks."""

import torch
import torch.nn as nn


def add_gaussian_noise(hidden_states, noise_level):
    """Add Gaussian noise scaled by std."""
    std = hidden_states.std(dim=-1, keepdim=True)
    noise = torch.randn_like(hidden_states) * std * noise_level
    return hidden_states + noise


class InterventionHook:
    """Hook to capture and modify layer outputs."""

    def __init__(self, apply_noise=False, noise_level=0.0):
        self.apply_noise = apply_noise
        self.noise_level = noise_level
        self.captured_output = None

    def __call__(self, module, input, output):
        # Handle tuple outputs from transformers
        if isinstance(output, tuple):
            hidden_states = output[0]
            other = output[1:]
        else:
            hidden_states = output
            other = ()

        # Capture original output
        self.captured_output = hidden_states.detach().clone()

        # Apply noise if enabled
        if self.apply_noise:
            hidden_states = add_gaussian_noise(hidden_states, self.noise_level)

        # Return in same format
        if isinstance(output, tuple):
            return (hidden_states,) + other
        return hidden_states


def get_model_layers(model):
    """
    Get transformer layers from model.

    Handles different architectures:
    - Tiny Aya / Qwen: model.model.layers
    - GPT-2: model.transformer.h
    - GPT-NeoX: model.gpt_neox.layers
    """
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return model.transformer.h
    elif hasattr(model, 'gpt_neox') and hasattr(model.gpt_neox, 'layers'):
        return model.gpt_neox.layers
    else:
        raise ValueError(
            f"Cannot find layers in model type: {type(model)}. "
            f"Model attributes: {dir(model)}"
        )


def register_hooks(model, target_layer=None, noise_level=0.0):
    """
    Register hooks on all layers.

    Args:
        model: The model
        target_layer: Layer index to apply noise (None = no noise)
        noise_level: Noise level for target layer

    Returns:
        List of (hook, handle) tuples
    """
    layers = get_model_layers(model)
    hooks = []

    for i, layer in enumerate(layers):
        apply_noise = (i == target_layer)
        hook = InterventionHook(apply_noise=apply_noise, noise_level=noise_level)
        handle = layer.register_forward_hook(hook)
        hooks.append((hook, handle))

    return hooks


def remove_hooks(hooks):
    """Remove all registered hooks."""
    for hook, handle in hooks:
        handle.remove()
