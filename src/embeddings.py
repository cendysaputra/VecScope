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
_SPLIT = re.compile(r"[,\s]+")


@lru_cache(maxsize=None)
def load_glove(name: str = DEFAULT_MODEL):
    import gensim.downloader as api

    return api.load(name)


def normalize(word: str) -> str:
    # GloVe is uncased
    return word.strip().lower()


def parse_words(text: str) -> list[str]:
    # Dedupe, keep order
    return list(dict.fromkeys(w for w in map(normalize, _SPLIT.split(text)) if w))


@dataclass
class Lookup:
    words: list[str]
    vectors: np.ndarray
    missing: list[str]

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1])


def get_vectors(kv, words: list[str]) -> Lookup:
    found, missing = [], []
    for w in map(normalize, words):
        (found if w in kv.key_to_index else missing).append(w)
    # Single gather
    vectors = kv.vectors[[kv.key_to_index[w] for w in found]] if found else np.empty((0, kv.vector_size), np.float32)
    return Lookup(found, vectors, missing)
