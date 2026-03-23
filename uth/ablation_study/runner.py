"""Simple experiment runner for layer ablation study."""

import torch
import json
import os
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from dataset_utils import (
    load_flores_dataset,
    sample_sentence_ids,
    get_parallel_sentences,
    get_non_english_languages
)
from intervention import register_hooks, remove_hooks, get_model_layers


def compute_translation_loss(model, tokenizer, source_text, target_text):
    """
    Compute translation loss using teacher forcing.

    Args:
        model: Language model
        tokenizer: Tokenizer
        source_text: Source sentence
        target_text: Target (English) sentence

    Returns:
        Loss value (float)
    """
    # Format prompt
    prompt = f"Translate the sentence to English: {source_text}"
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Concatenate prompt + target
    full_text = formatted_prompt + " " + target_text

    # Tokenize
    prompt_tokens = tokenizer(formatted_prompt, return_tensors='pt')
    full_tokens = tokenizer(full_text, return_tensors='pt')

    # Move to device
    device = next(model.parameters()).device
    full_tokens = {k: v.to(device) for k, v in full_tokens.items()}
    prompt_length = prompt_tokens['input_ids'].shape[1]

    # Forward pass
    model.eval()
    with torch.inference_mode(mode=False):  # Don't use no_grad (breaks hooks)
        outputs = model(input_ids=full_tokens['input_ids'])

    # Compute loss on target tokens only
    logits = outputs.logits[:, :-1, :]
    labels = full_tokens['input_ids'][:, 1:]

    # Mask to only compute loss on target tokens
    loss_mask = torch.zeros_like(labels, dtype=torch.bool)
    loss_mask[:, prompt_length:] = True

    # Compute token-level losses
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    token_losses = loss_fct(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1)
    ).view(labels.shape)

    # Average over target tokens
    masked_loss = (token_losses * loss_mask.float()).sum()
    num_tokens = loss_mask.sum()

    if num_tokens > 0:
        return (masked_loss / num_tokens).item()
    return float('inf')


def run_experiment(
    model_name="CohereLabs/tiny-aya-global",
    dataset_path="../data/flores_dataset.json",
    output_dir="results",
    sample_size=15,
    noise_levels=[0.1, 0.5, 1.0, 2.0],
    device='cuda',
    hf_token=None
):
    """
    Run layer ablation experiment.

    Args:
        model_name: HuggingFace model ID
        dataset_path: Path to FLORES JSON
        output_dir: Where to save results
        sample_size: Number of sentences to sample
        noise_levels: List of noise levels to test
        device: 'cuda' or 'cpu'
        hf_token: HuggingFace token for gated models (or None to use cached token)
    """
    # Setup output
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / "results.jsonl"

    # Get HuggingFace token (from parameter, env var, or cached)
    if hf_token is None:
        hf_token = os.getenv('HF_TOKEN', None)

    # Load model
    print(f"Loading model: {model_name}")
    if hf_token:
        print("Using HuggingFace token from environment/parameter")
    else:
        print("No HF_TOKEN found - using cached credentials (if available)")

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
        device_map='auto' if device == 'cuda' else None,
        token=hf_token
    )
    if device == 'cpu':
        model = model.to(device)
    model.eval()

    num_layers = len(get_model_layers(model))
    print(f"Model has {num_layers} layers")

    # Load dataset
    print(f"Loading dataset from: {dataset_path}")
    data = load_flores_dataset(dataset_path)
    sentence_ids = sample_sentence_ids(sample_size, seed=42)
    languages = get_non_english_languages()

    print(f"Sampled {len(sentence_ids)} sentences")
    print(f"Testing {len(languages)} languages")
    print(f"Noise levels: {noise_levels}")

    # Calculate total experiments
    total = len(sentence_ids) * len(languages) * (1 + num_layers * len(noise_levels))
    print(f"Total experiments: {total}\n")

    # Run experiments
    with open(results_file, 'w') as f:
        pbar = tqdm(total=total, desc="Running experiments")

        for sent_id in sentence_ids:
            parallel = get_parallel_sentences(data, sent_id)
            english_target = parallel['english']

            for lang in languages:
                if lang not in parallel:
                    continue

                source_text = parallel[lang]

                # Baseline (no intervention)
                hooks = register_hooks(model, target_layer=None)
                try:
                    baseline_loss = compute_translation_loss(
                        model, tokenizer, source_text, english_target
                    )
                except Exception as e:
                    print(f"\nError in baseline for {lang}: {e}")
                    baseline_loss = float('inf')
                finally:
                    remove_hooks(hooks)

                pbar.update(1)

                # Intervene at each layer
                for layer_idx in range(num_layers):
                    for noise_level in noise_levels:
                        hooks = register_hooks(
                            model,
                            target_layer=layer_idx,
                            noise_level=noise_level
                        )

                        try:
                            intervened_loss = compute_translation_loss(
                                model, tokenizer, source_text, english_target
                            )

                            result = {
                                'sentence_id': sent_id,
                                'language': lang,
                                'layer': layer_idx,
                                'noise_level': noise_level,
                                'baseline_loss': baseline_loss,
                                'intervened_loss': intervened_loss,
                                'loss_delta': intervened_loss - baseline_loss
                            }

                            f.write(json.dumps(result) + '\n')
                            f.flush()

                        except Exception as e:
                            print(f"\nError at layer {layer_idx}, noise {noise_level}: {e}")

                        finally:
                            remove_hooks(hooks)

                        pbar.update(1)

        pbar.close()

    print(f"\nResults saved to: {results_file}")
    return results_file
