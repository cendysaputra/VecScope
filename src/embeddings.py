from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

DEFAULT_MODEL = "glove-wiki-gigaword-50"
MODELS = {
    "glove-wiki-gigaword-50": "GloVe 50d (~66 MB)",
    "glove-wiki-gigaword-100": "GloVe 100d (~128 MB)",
    "glove-wiki-gigaword-300": "GloVe 300d (~376 MB)",
}
_SPLIT = re.compile(r"[,\s]+")


@lru_cache(maxsize=None)
def _load_glove(name: str):
    from pathlib import Path

    import gensim.downloader as api
    from gensim.models import KeyedVectors

    # Binary mmap cache
    cache = Path(api.BASE_DIR) / f"{name}.kv"
    if cache.exists():
        return KeyedVectors.load(str(cache), mmap="r")
    kv = api.load(name)
    kv.fill_norms()
    cache.parent.mkdir(parents=True, exist_ok=True)
    kv.save(str(cache))
    return kv


def load_glove(name: str = DEFAULT_MODEL):
    return _load_glove(name)


def normalize(word: str) -> str:
    # GloVe is uncased
    return word.strip().lower()


def parse_words(text: str) -> list[str]:
    # Dedupe, keep order
    return list(dict.fromkeys(w for w in map(normalize, _SPLIT.split(text)) if w))


def parse_groups(text: str) -> tuple[list[str], list[str]]:
    # Group per line
    seen: dict[str, str] = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        group, _, rest = line.partition(":") if ":" in line else ("words", "", line)
        label = group.strip() or "words"
        for w in parse_words(rest):
            seen.setdefault(w, label)
    return list(seen), list(seen.values())


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
