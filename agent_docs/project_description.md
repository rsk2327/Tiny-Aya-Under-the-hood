# Tiny Aya Under The Hood

This project investigates how Tiny Aya processes information across languages by analyzing how representations evolve across model layers. By comparing layer-wise representations in Tiny Aya Global and its regional variants, we aim to understand where language-agnostic (universal) processing emerges and where region-specific specialization occurs.

Gaining insight into these differences can help identify layers that are more universal versus more specialized, enabling targeted interventions such as representation steering to further improve multilingual performance. Ultimately, this work contributes to a deeper understanding of cross-lingual information flow and regional specialization within compact multilingual language models.

What is the question we want to answer?
Which parts of Tiny Aya’s network learn language-agnostic representations, and which parts become specialized for specific languages or regions?


Why is this question important?
We will be able to answer or have knowledge to these particular questions
Identify universal vs. specialized layers in Tiny Aya
Enable targeted steering / regularization instead of full fine-tuning
Support efficient parameter sharing across regional models
Diagnose where multilingual failures originate (early vs. late layers)
Guide safe compression and pruning decisions
Inform adapter placement and future multilingual architecture design
Improve interpretability of global vs. regional model behavior


## Methodology
1. Multilingual Parallel Dataset Construction
Goal: Ensure semantic equivalence across languages for controlled analysis.
Construct a dataset of semantically equivalent sentences across multiple languages using translations or aligned corpora.
Include languages from different families (Romance, Indo-Aryan, Semitic, etc.).
Ensure coverage of lexical, syntactic, and semantic variations.
Use the same sentence set across all models to maintain comparability.

Data to Store
Sentence text
Language ID
Sentence alignment ID (linking translations of the same sentence)


2. Model Evaluation Across Variants
Goal: Compare representation behavior across Tiny Aya Global and regional models.
Run identical multilingual inputs through Tiny Aya Global and regional variants.
Perform inference in a controlled environment with identical tokenization and prompts.
Capture internal model states during forward passes.

Data to Store
Model name / variant
Tokenized inputs
Model outputs (logits, generated tokens)


3. Layer-wise Hidden State Extraction
Goal: Capture intermediate representations across the network.
Extract hidden states from every transformer layer during inference.
Store both token-level and sentence-level representations.
Use pooling (mean pooling or CLS token) for sentence embeddings.
Organize hidden states by layer, language, and model variant.

Data to Store
Token-level hidden states per layer
Sentence-level pooled embeddings
Layer index
Model variant



4. Multilingual Latent Space Construction
Goal: Build a structured representation dataset across all languages.
Aggregate pooled representations across languages and layers.
Construct similarity matrices across languages.
Prepare representation datasets for downstream analysis tasks.

Data to Store
Sentence embeddings per layer
Language metadata
Language family metadata
Representation similarity matrices


5. Activation Intervention Framework 
Goal: Enable causal experiments on model representations.
Implement forward hooks to allow controlled modification of hidden states.
Support operations such as noise injection, vector direction injection, or representation replacement.
Evaluate downstream effects after intervention.

Data to Store
Intervention layer index
Injection vector / noise parameters
Pre- and post-intervention hidden states
Downstream evaluation metrics
