"""Cosine similarity, nearest neighbours and word analogies."""

from __future__ import annotations

import numpy as np

from .embeddings import normalize


class OOVError(KeyError):
    """Raised when one or more words are not in the embedding vocabulary."""

    def __init__(self, words: list[str]):
        self.words = words
        super().__init__(", ".join(words))

    def __str__(self) -> str:
        return f"Not in vocabulary: {', '.join(self.words)}"


def l2_normalize(x: np.ndarray, axis: int = -1) -> np.ndarray:
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.clip(norm, 1e-12, None)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(l2_normalize(a), l2_normalize(b)))


def cosine_matrix(a: np.ndarray, b: np.ndarray | None = None) -> np.ndarray:
    """Pairwise cosine similarity between rows of ``a`` and rows of ``b`` (default: ``a``)."""
    b = a if b is None else b
    return l2_normalize(a) @ l2_normalize(b).T


def _check_vocab(kv, words: list[str]) -> list[str]:
    words = [normalize(w) for w in words if normalize(w)]
    missing = [w for w in words if w not in kv.key_to_index]
    if missing:
        raise OOVError(missing)
    return words


def nearest(kv, word: str, topn: int = 10) -> list[tuple[str, float]]:
    (word,) = _check_vocab(kv, [word])
    return kv.most_similar(word, topn=topn)


def analogy(kv, positive: list[str], negative: list[str], topn: int = 10) -> list[tuple[str, float]]:
    """``positive[0] - negative[0] + positive[1] ...`` → top-k nearest words.

    Uses Gensim's ``most_similar`` (3CosAdd on unit vectors), which excludes the
    input words from the results — so ``king - man + woman`` can't return ``king``.
    """
    pos = _check_vocab(kv, positive)
    neg = _check_vocab(kv, negative)
    if not pos:
        raise ValueError("Need at least one positive word.")
    return kv.most_similar(positive=pos, negative=neg, topn=topn)


def analogy_vector(kv, positive: list[str], negative: list[str]) -> np.ndarray:
    """The raw result vector of the analogy (sum of unit vectors), for plotting."""
    pos = _check_vocab(kv, positive)
    neg = _check_vocab(kv, negative)
    v = sum(kv.get_vector(w, norm=True) for w in pos) - sum(
        (kv.get_vector(w, norm=True) for w in neg), np.zeros(kv.vector_size, dtype=np.float32)
    )
    return np.asarray(v, dtype=np.float32)


def parse_analogy(expr: str) -> tuple[list[str], list[str]]:
    """Parse ``"king - man + woman"`` → (["king", "woman"], ["man"])."""
    positive, negative = [], []
    sign = "+"
    token = ""
    for ch in expr + "+":
        if ch in "+-":
            w = normalize(token)
            if w:
                (positive if sign == "+" else negative).append(w)
            sign, token = ch, ""
        else:
            token += ch
    return positive, negative
