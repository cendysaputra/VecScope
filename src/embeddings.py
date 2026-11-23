"""Static word embeddings (GloVe via the Gensim downloader)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

DEFAULT_MODEL = "glove-wiki-gigaword-50"
MODELS = {
    "glove-wiki-gigaword-50": "GloVe 50d (~66 MB)",
    "glove-wiki-gigaword-100": "GloVe 100d (~128 MB)",
}


@lru_cache(maxsize=None)
def load_glove(name: str = DEFAULT_MODEL):
    """Load a Gensim KeyedVectors model. First call downloads it to ``~/gensim-data``."""
    import gensim.downloader as api

    return api.load(name)


def normalize(word: str) -> str:
    """GloVe wiki-gigaword is uncased, so lowercase and trim."""
    return word.strip().lower()


def parse_words(text: str) -> list[str]:
    """Split on commas / whitespace, normalize, drop empties and duplicates (order kept)."""
    seen: dict[str, None] = {}
    for raw in re.split(r"[,\s]+", text):
        w = normalize(raw)
        if w:
            seen.setdefault(w, None)
    return list(seen)


@dataclass
class Lookup:
    words: list[str]  # words found in the vocabulary, in input order
    vectors: np.ndarray  # shape (len(words), dim)
    missing: list[str]  # out-of-vocabulary words

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1]) if self.vectors.ndim == 2 else 0


def get_vectors(kv, words: list[str]) -> Lookup:
    found, vecs, missing = [], [], []
    for w in words:
        w = normalize(w)
        if w in kv.key_to_index:
            found.append(w)
            vecs.append(kv[w])
        else:
            missing.append(w)
    dim = kv.vector_size
    arr = np.vstack(vecs).astype(np.float32) if vecs else np.empty((0, dim), dtype=np.float32)
    return Lookup(words=found, vectors=arr, missing=missing)
