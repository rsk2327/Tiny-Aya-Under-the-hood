# Layer Ablation Study

Minimal framework to identify critical layers in Tiny Aya through noise injection experiments.

## About Tiny Aya

- **Model Collection**: https://huggingface.co/collections/CohereLabs/tiny-aya
- **Default Model**: `CohereLabs/tiny-aya-global`
- **Architecture**: 4 transformer layers (3 sliding window + 1 global attention)
- **Parameters**: ~3.35B
- **Variants**: tiny-aya-earth, regional variants available

## Files

- `dataset_utils.py` - Load and sample FLORES dataset
- `intervention.py` - Hook system for layer interventions
- `runner.py` - Experiment orchestration
- `ablation_experiment.ipynb` - Main notebook to run experiments

## Quick Start

### 1. Get HuggingFace Access

Tiny Aya models are gated. You need to:

1. **Accept model terms**: Visit https://huggingface.co/CohereLabs/tiny-aya-global and accept terms
2. **Get your token**: Go to https://huggingface.co/settings/tokens
3. **Set environment variable**:
   ```bash
   export HF_TOKEN="hf_your_token_here"
   ```
   Or login via CLI:
   ```bash
   huggingface-cli login
   ```

The code automatically reads from the `HF_TOKEN` environment variable.

### 2. Run the Notebook

```bash
jupyter notebook ablation_experiment.ipynb
```

Follow the cells to:
- Authenticate with HuggingFace
- Configure experiment parameters
- Run the ablation study
- Visualize results

### 3. Or Run Programmatically

```python
from runner import run_experiment

# Make sure HF_TOKEN is set in your environment first!
# export HF_TOKEN="hf_..."

results_file = run_experiment(
    model_name="CohereLabs/tiny-aya-global",
    dataset_path="../data/flores_dataset.json",
    output_dir="results",
    sample_size=15,
    noise_levels=[0.1, 0.5, 1.0, 2.0],
    device='cuda'
)
# Token is automatically read from HF_TOKEN environment variable
```

## How It Works

1. **Sample sentences** from FLORES (default: 15 sentences × 11 languages = 165 samples)
2. **Compute baseline** translation loss for each sample
3. **For each layer** and each noise level:
   - Inject Gaussian noise at that layer
   - Compute translation loss
   - Calculate loss delta (increase from baseline)
4. **Identify critical layers** - layers with highest loss delta are most important

## Output

Results saved to `results/results.jsonl`:

```json
{
  "sentence_id": 52,
  "language": "french",
  "layer": 2,
  "noise_level": 0.5,
  "baseline_loss": 1.234,
  "intervened_loss": 2.567,
  "loss_delta": 1.333
}
```

## Analysis

The notebook includes visualizations:
- Layer sensitivity bar chart
- Layer × Language heatmap
- Noise level comparison
- Per-language sensitivity

## Requirements

```bash
pip install torch transformers accelerate pandas matplotlib seaborn tqdm jupyter
```
