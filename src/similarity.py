from __future__ import annotations

import numpy as np

from .embeddings import normalize


class OOVError(KeyError):
    def __init__(self, words: list[str]):
        self.words = words
        super().__init__(", ".join(words))

    def __str__(self) -> str:
        return f"Not in vocabulary: {', '.join(self.words)}"


def l2_normalize(x: np.ndarray, axis: int = -1) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=axis, keepdims=True), 1e-12, None)


def center_unit(x: np.ndarray) -> np.ndarray:
    # Center, renormalize
    return l2_normalize(x - x.mean(axis=0, keepdims=True), axis=1)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(l2_normalize(a), l2_normalize(b)))


def cosine_matrix(a: np.ndarray, b: np.ndarray | None = None) -> np.ndarray:
    a = l2_normalize(a)
    return a @ (a if b is None else l2_normalize(b)).T


def pairwise_scores(words: list[str], sim: np.ndarray) -> list[dict]:
    # Upper-triangle pairs
    i, j = np.triu_indices(len(words), k=1)
    scores = sim[i, j]
    order = np.argsort(-scores)
    return [{"word A": words[int(i[k])], "word B": words[int(j[k])], "cosine": round(float(scores[k]), 4)} for k in order]


def _check_vocab(kv, words: list[str]) -> list[str]:
    words = [w for w in map(normalize, words) if w]
    missing = [w for w in words if w not in kv.key_to_index]
    if missing:
        raise OOVError(missing)
    return words


def unit_vectors(kv, words: list[str]) -> np.ndarray:
    # Single unit gather
    idx = [kv.key_to_index[w] for w in _check_vocab(kv, words)]
    return kv.get_normed_vectors()[idx] if idx else np.empty((0, kv.vector_size), np.float32)


def nearest(kv, word: str, topn: int = 10) -> list[tuple[str, float]]:
    return kv.most_similar(_check_vocab(kv, [word])[0], topn=topn)


def analogy(kv, positive: list[str], negative: list[str], topn: int = 10) -> list[tuple[str, float]]:
    # Excludes input words
    pos, neg = _check_vocab(kv, positive), _check_vocab(kv, negative)
    if not pos:
        raise ValueError("Need at least one positive word.")
    return kv.most_similar(positive=pos, negative=neg, topn=topn)


def analogy_vector(kv, positive: list[str], negative: list[str]) -> np.ndarray:
    # 3CosAdd raw vector
    pos = unit_vectors(kv, positive)
    neg = unit_vectors(kv, negative)
    if len(pos) == 0:
        raise ValueError("Need at least one positive word.")
    return pos.sum(axis=0) - neg.sum(axis=0)


def parse_analogy(expr: str) -> tuple[list[str], list[str]]:
    positive, negative, sign, token = [], [], "+", ""
    for ch in expr + "+":
        if ch in "+-":
            if w := normalize(token):
                (positive if sign == "+" else negative).append(w)
            sign, token = ch, ""
        else:
            token += ch
    return positive, negative
