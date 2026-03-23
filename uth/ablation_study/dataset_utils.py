"""Simple utilities for loading FLORES dataset."""

import json
import random
from pathlib import Path


def load_flores_dataset(dataset_path):
    """Load FLORES dataset from JSON."""
    with open(dataset_path, 'r') as f:
        return json.load(f)


def sample_sentence_ids(n, seed=42):
    """Sample n random sentence IDs."""
    random.seed(seed)
    return sorted(random.sample(range(1, 2010), n))


def get_parallel_sentences(data, sentence_id):
    """Get all language versions of a sentence."""
    parallel = {}
    for item in data:
        if item['sentence_alignment_id'] == sentence_id:
            parallel[item['language']] = item['text']
    return parallel


def get_non_english_languages():
    """Get list of non-English languages."""
    return ['amharic', 'arabic', 'bengali', 'french', 'german',
            'hindi', 'persian', 'spanish', 'tamil', 'turkish', 'yoruba']
