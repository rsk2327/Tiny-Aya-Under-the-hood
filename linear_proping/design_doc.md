# Layer-wise Probing Linear Probing Analysis for Tiny Aya Global

The goal of this analysis is to uncover where lexical (Language ID) and syntactic (POS) information emerges in multilingual models. This simple study will use layer-wise linear probing to examine how linguistic information is encoded across 37 layers of the Cohere Labs’ Tiny Aya models.

We will focus on:

1.  **Main model:** Tiny Aya Global (`CohereLabs/tiny-aya-global`)
2.  **Regional variants:**
    * **Earth**, Africa & West Asia focus (`CohereLabs/tiny-aya-earth`)
    * **Fire**, South Asia (`CohereLabs/tiny-aya-fire`)
    * **Water**, Asia-Pacific & Europe (`CohereLabs/tiny-aya-water`)

All loaded with `output_hidden_states=True`.

---

## STEP 1 - Data Strategy

### Language Identification (LID)
To determine which layer of the model decides the language it is reading, we compare high-resource and low-resource (specifically African) languages across 13 selections:

* **High-resource:** English, Spanish, French, German
* **Medium-resource:** Arabic, Hindi, Bengali, Tamil, Turkish, Persian
* **Low-resource:** Swahili, Amharic, Yoruba

**Task:** Predict sentence language.

**Three specific phenomena to observe:**
1.  **Script advantage:** Do Amharic (Ge’ez) and Hindi (Devanagari) reach high accuracy in earlier layers than Yoruba due to their unique scripts?
2.  **Script confusion:** Does the model struggle to distinguish English, French, and Yoruba in early layers because they use Latin characters? Where does the "separation" happen?
3.  **The Resource Gap:** Does the accuracy curve for English rise more steeply than for lower-resource languages like Swahili?

**Dataset Target Size:** 6,500 sentence examples (500 sentences per language)

**Dataset Source:** FLORES-200, dev/test splits (`openlanguagedata/flores_plus`)

### Part-of-Speech Tagging (POS)
We will consider 11 languages:

* **High-resource:** English, Spanish, French, German
* **Medium-resource:** Arabic, Hindi, Tamil, Turkish, Persian
* **Low-resource:** Amharic, Yoruba

**Task:** Token-level UPOS tag prediction.

**Dataset Target Size:** 1,300 sentence examples (100 sentences per language); at least 1,000 tagged tokens per language

**Dataset Source:** Universal Dependencies (UD) treebanks on Hugging Face (`/universal_dependencies`)

---

## STEP 2 - Feature Extraction & Inferencing
Pass sentences through Tiny Aya with `output_hidden_states=True` to extract a 3D tensor representing each layer's output.
* **Shape:** `[37, Sequence Length, 3,072]`

## STEP 3 - Layer Harvesting
Capture the hidden state tensor $H_l$ for every layer $l$. Harvest the layer outputs into a list to record the model's "thinking" at each specific layer.

## STEP 4 - Pooling vs. Alignment
Turn the collected vectors into a single "feature" for the probe.
* **LID Mean Pooling:** Calculate the mean of $H_l$ across the sequence dimension. This produces one 3,072-dimensional vector representing the "average" language signal for that layer.
* **POS Alignment:** Identify the index of the first sub-token for each labeled word and extract $H_l[index]$.

## STEP 5 - Storage
Save vectors as NumPy arrays or PyTorch tensors mapped to their labels ($Y$):
* **LID:** Sentence, 3,072 features, language target.
* **POS:** Word, 3,072 features, POS target (e.g., Noun).

---

## Phase II: Linear Probing

### STEP 6 - Architecture
Build a simple Logistic Regression (Scikit-Learn) with L2 regularization. A simple model is used to see if the information is linearly separable, while L2 regularization prevents overfitting to specific noise.

### STEP 7 - Training
Train 37 independent models—one for every layer of Tiny Aya. For each layer $l$, train a dedicated classifier: $P_l(X_l) \to Y$.

### STEP 8 - Cross-Validation
Use an 80/20 train-test split for each layer's probe.

---

## Evaluation & Metrics
We will track **Accuracy** and **F1-Score** per layer.

### Visualization Plan
* **Primary Plot:** X-axis (Layer Number 0-36) vs. Y-axis (Probing Accuracy).
* **Comparative Curves:** One curve for LID and one for POS.
* **Heatmaps:** Layer vs. Language accuracy to observe if certain languages (e.g., Yoruba) "mature" later than others (e.g., English).