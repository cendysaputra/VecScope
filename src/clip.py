"""CLIP: put text and images into the same vector space (v2).

Only compare CLIP vectors with CLIP vectors — never mix them with GloVe.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

DEFAULT_CLIP = "openai/clip-vit-base-patch32"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@lru_cache(maxsize=None)
def load_clip(name: str = DEFAULT_CLIP):
    """Return ``(model, processor)``. First call downloads ~600 MB to the Hugging Face cache."""
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(name)
    model.eval()
    processor = CLIPProcessor.from_pretrained(name)
    return model, processor


def _features(out) -> "torch.Tensor":  # noqa: F821
    # transformers < 5 returns a tensor; >= 5 returns a ModelOutput whose
    # pooler_output already has the projection applied.
    return out if hasattr(out, "shape") else out.pooler_output


def _to_unit_numpy(t) -> np.ndarray:
    v = t.detach().cpu().float().numpy()
    return v / np.clip(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12, None)


def embed_texts(texts: list[str], name: str = DEFAULT_CLIP) -> np.ndarray:
    """L2-normalised CLIP text embeddings, shape ``(len(texts), 512)`` for ViT-B/32."""
    import torch

    model, processor = load_clip(name)
    inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True)
    with torch.inference_mode():
        out = model.get_text_features(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
    return _to_unit_numpy(_features(out))


def embed_images(images: list, name: str = DEFAULT_CLIP) -> np.ndarray:
    """L2-normalised CLIP image embeddings. ``images`` are PIL images."""
    import torch

    model, processor = load_clip(name)
    rgb = [im.convert("RGB") for im in images]
    inputs = processor(images=rgb, return_tensors="pt")
    with torch.inference_mode():
        out = model.get_image_features(pixel_values=inputs["pixel_values"])
    return _to_unit_numpy(_features(out))


def text_image_similarity(text_vecs: np.ndarray, image_vecs: np.ndarray) -> np.ndarray:
    """Cosine similarity matrix (texts × images). Inputs are already unit-length."""
    return text_vecs @ image_vecs.T


def rank_images(query_vec: np.ndarray, image_vecs: np.ndarray, names: list[str]) -> list[tuple[str, float]]:
    """Sort images by cosine similarity to a single text query vector (highest first)."""
    scores = image_vecs @ np.asarray(query_vec).reshape(-1)
    order = np.argsort(-scores)
    return [(names[i], float(scores[i])) for i in order]


def list_sample_images(folder: str | Path = "data/samples") -> list[Path]:
    folder = Path(folder)
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)
