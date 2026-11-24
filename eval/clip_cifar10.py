# Usage: python eval/clip_cifar10.py
from __future__ import annotations

import json
import pickle
import sys
import tarfile
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import transformers
from PIL import Image

from src import clip as clipmod
from src.similarity import l2_normalize

CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR10_CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
SINGLE_TEMPLATE = "a photo of a {}."
# OpenAI CIFAR-10 templates
ENSEMBLE_TEMPLATES = (
    "a photo of a {}.",
    "a blurry photo of a {}.",
    "a black and white photo of a {}.",
    "a low contrast photo of a {}.",
    "a high contrast photo of a {}.",
    "a bad photo of a {}.",
    "a good photo of a {}.",
    "a photo of a small {}.",
    "a photo of a big {}.",
    "a photo of the {}.",
    "a blurry photo of the {}.",
    "a black and white photo of the {}.",
    "a low contrast photo of the {}.",
    "a high contrast photo of the {}.",
    "a bad photo of the {}.",
    "a good photo of the {}.",
    "a photo of the small {}.",
    "a photo of the big {}.",
)
# Sample caption pairs
SAMPLE_CAPTIONS = (
    ("a photo of two cats sleeping on a couch", ("cats.jpg",)),
    ("a big fluffy cat walking in the snow", ("cat-chonk.jpg",)),
    ("a bee on a pink flower", ("bee.jpg",)),
    ("a black and white dog sitting on grass by a fence", ("dog.png",)),
    ("a plate of baklava sweet pastry with pistachios", ("baklava.jpg",)),
    ("a mosque with domes and minarets", ("mosque.jpg",)),
    ("a small humanoid robot standing on a table", ("robot.png",)),
)
BASELINES = {
    "chance": 10.0,
    "clip_paper_vit_b32": {"paper": "Radford et al. (2021) Table 10 / Table 11", "accuracy": 89.8},
    "cifar10_app_cnn": {"repo": "cifar10-app (supervised CNN trained on CIFAR-10)", "accuracy": 95.71},
}


def load_cifar10_test() -> tuple[np.ndarray, np.ndarray]:
    # Cache outside repo
    cache_dir = Path.home() / ".cache" / "vecscope"
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive = cache_dir / "cifar-10-python.tar.gz"
    if not archive.exists() or archive.stat().st_size < 170_000_000:
        tmp = archive.with_suffix(".tmp")
        print(f"Downloading CIFAR-10 to {archive}...", flush=True)
        urllib.request.urlretrieve(CIFAR10_URL, tmp)
        tmp.replace(archive)
    with tarfile.open(archive, "r:gz") as tar:
        member = tar.extractfile("cifar-10-batches-py/test_batch")
        batch = pickle.load(member, encoding="bytes")
    images = batch[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    labels = np.asarray(batch[b"labels"], dtype=np.int64)
    return images, labels


def build_text_weights() -> tuple[np.ndarray, np.ndarray]:
    # Single and ensemble
    single_prompts = [SINGLE_TEMPLATE.format(c) for c in CIFAR10_CLASSES]
    single_vecs = clipmod.embed_texts(single_prompts)

    ensemble_vecs = []
    for c in CIFAR10_CLASSES:
        vecs = clipmod.embed_texts([t.format(c) for t in ENSEMBLE_TEMPLATES])
        ensemble_vecs.append(l2_normalize(vecs.mean(axis=0)))
    return single_vecs, np.vstack(ensemble_vecs)


def evaluate_retrieval() -> dict:
    # Captioned sample retrieval
    paths = clipmod.list_sample_images()
    if not paths:
        return {}
    names = [p.name for p in paths]
    img_vecs = clipmod.embed_images([Image.open(p) for p in paths])
    captions = [c for c, _ in SAMPLE_CAPTIONS]
    targets = [set(t) for _, t in SAMPLE_CAPTIONS]
    txt_vecs = clipmod.embed_texts(captions)
    sim = clipmod.text_image_similarity(txt_vecs, img_vecs)

    r1 = r5 = 0
    rows = []
    for cap, valid, scores in zip(captions, targets, sim):
        order = np.argsort(-scores)
        ranked = [names[i] for i in order]
        hit1 = ranked[0] in valid
        hit5 = any(n in valid for n in ranked[:5])
        r1 += int(hit1)
        r5 += int(hit5)
        rows.append({"caption": cap, "top1": ranked[0], "top1_cosine": round(float(scores[order[0]]), 4), "hit1": hit1, "hit5": hit5})
    n = len(captions)
    return {
        "num_queries": n,
        "num_images": len(names),
        "recall_at_1": round(100.0 * r1 / n, 2),
        "recall_at_5": round(100.0 * r5 / n, 2),
        "queries": rows,
    }


def embed_cifar10_test(images: np.ndarray, batch_size: int = 64) -> np.ndarray:
    # Shard cache outside repo
    cache_dir = Path.home() / ".cache" / "vecscope" / "cifar10_vit_b32"
    cache_dir.mkdir(parents=True, exist_ok=True)
    n = len(images)
    shard_size = 1000
    shards = []
    t0 = time.time()
    for s_start in range(0, n, shard_size):
        s_end = min(s_start + shard_size, n)
        shard_file = cache_dir / f"test_{s_start:05d}_{s_end:05d}.npy"
        if shard_file.exists():
            shards.append(np.load(shard_file))
            continue
        vecs = []
        for start in range(s_start, s_end, batch_size):
            end = min(start + batch_size, s_end)
            batch = [Image.fromarray(arr) for arr in images[start:end]]
            vecs.append(clipmod.embed_images(batch))
        arr = np.vstack(vecs)
        tmp = shard_file.with_name(shard_file.stem + ".tmp.npy")
        np.save(tmp, arr)
        tmp.replace(shard_file)
        shards.append(arr)
        print(f"Embedded {s_end}/{n} CIFAR-10 test images ({time.time() - t0:.1f}s)", flush=True)
    return np.vstack(shards)


def main(batch_size: int = 64) -> int:
    t0 = time.time()
    images, labels = load_cifar10_test()
    single_txt, ensemble_txt = build_text_weights()
    img_vecs = embed_cifar10_test(images, batch_size=batch_size)
    n = len(images)

    # Cosine classification
    pred_single = np.argmax(img_vecs @ single_txt.T, axis=1)
    pred_ensemble = np.argmax(img_vecs @ ensemble_txt.T, axis=1)
    acc_single = round(100.0 * float(np.mean(pred_single == labels)), 2)
    acc_ensemble = round(100.0 * float(np.mean(pred_ensemble == labels)), 2)

    per_class = {}
    for idx, cls_name in enumerate(CIFAR10_CLASSES):
        mask = labels == idx
        per_class[cls_name] = {
            "count": int(np.sum(mask)),
            "single_prompt_accuracy": round(100.0 * float(np.mean(pred_single[mask] == idx)), 2),
            "ensemble_accuracy": round(100.0 * float(np.mean(pred_ensemble[mask] == idx)), 2),
        }

    retrieval = evaluate_retrieval()
    elapsed = round(time.time() - t0, 2)

    print(f"\nCLIP ({clipmod.DEFAULT_CLIP}) on CIFAR-10 test ({n} images):")
    print(f"- Single prompt (`{SINGLE_TEMPLATE}`): {acc_single:.2f}%")
    print(f"- 18-prompt ensemble (OpenAI CIFAR-10 templates): {acc_ensemble:.2f}%")
    print(f"- Baselines: chance = 10.00%, CLIP paper ViT-B/32 (Table 10/11) = 89.80%, cifar10-app supervised CNN = 95.71%")
    if retrieval:
        print(f"- Sample text-to-image retrieval ({retrieval['num_queries']} queries, {retrieval['num_images']} images): Recall@1 = {retrieval['recall_at_1']:.2f}%, Recall@5 = {retrieval['recall_at_5']:.2f}%")

    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "task": "clip_cifar10_zero_shot",
        "model": clipmod.DEFAULT_CLIP,
        "dataset": "CIFAR-10 test (10,000 images, 32x32 upscaled to 224x224)",
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "elapsed_seconds": elapsed,
        "libraries": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "numpy": np.__version__,
        },
        "baselines": BASELINES,
        "results": {
            "num_images": n,
            "single_prompt_template": SINGLE_TEMPLATE,
            "single_prompt_accuracy": acc_single,
            "single_prompt_correct": int(np.sum(pred_single == labels)),
            "ensemble_num_templates": len(ENSEMBLE_TEMPLATES),
            "ensemble_accuracy": acc_ensemble,
            "ensemble_correct": int(np.sum(pred_ensemble == labels)),
            "per_class": per_class,
        },
        "sample_retrieval": retrieval,
    }
    out_path = out_dir / "clip_cifar10.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nsaved {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
