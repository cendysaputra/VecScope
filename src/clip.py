from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

DEFAULT_CLIP = "openai/clip-vit-base-patch32"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@lru_cache(maxsize=None)
def load_clip(name: str = DEFAULT_CLIP):
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(name).eval()
    return model, CLIPProcessor.from_pretrained(name)


def _unit(out) -> np.ndarray:
    # transformers v4/v5
    t = out if hasattr(out, "shape") else out.pooler_output
    v = t.detach().cpu().float().numpy()
    return v / np.clip(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12, None)


def embed_texts(texts: list[str], name: str = DEFAULT_CLIP) -> np.ndarray:
    import torch

    model, processor = load_clip(name)
    inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True)
    with torch.inference_mode():
        return _unit(model.get_text_features(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]))


def embed_images(images: list, name: str = DEFAULT_CLIP) -> np.ndarray:
    import torch

    model, processor = load_clip(name)
    inputs = processor(images=[im.convert("RGB") for im in images], return_tensors="pt")
    with torch.inference_mode():
        return _unit(model.get_image_features(pixel_values=inputs["pixel_values"]))


def text_image_similarity(text_vecs: np.ndarray, image_vecs: np.ndarray) -> np.ndarray:
    # Already unit-length
    return text_vecs @ image_vecs.T


def rank_images(query_vec: np.ndarray, image_vecs: np.ndarray, names: list[str]) -> list[tuple[str, float]]:
    scores = image_vecs @ np.asarray(query_vec).reshape(-1)
    return [(names[i], float(scores[i])) for i in np.argsort(-scores)]


def list_sample_images(folder: str | Path = "data/samples") -> list[Path]:
    folder = Path(folder)
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)
