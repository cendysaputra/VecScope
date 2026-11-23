# Usage: python scripts/evidence.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image

from src import clip as clipmod
from src.embeddings import get_vectors, load_glove
from src.similarity import analogy, cosine_matrix
from src.tokenize import DEFAULT_PAIR, tokenize

SENTENCE = "Tokenization isn't magic: unbelievably, GPT-2 and BERT split words differently!"
PAIRS = [("king", "queen"), ("man", "woman"), ("king", "man"), ("king", "woman"), ("king", "water"), ("queen", "apple")]
QUERIES = ["a photo of a cat", "a bee on a flower", "a robot", "a sweet pastry", "a car on a highway"]


def table(header, rows):
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for r in rows:
        print("| " + " | ".join(map(str, r)) + " |")
    print()


# Tokens
print(f"**Sentence:** `{SENTENCE}`\n")
for name in DEFAULT_PAIR:
    r = tokenize(SENTENCE, name)
    print(f"**{name}: {r.count} tokens**\n")
    print("```text\n" + " | ".join(f"{t} ({i})" for t, i in zip(r.tokens, r.ids)) + "\n```\n")

# Cosine
kv = load_glove()
words = sorted({w for p in PAIRS for w in p})
look = get_vectors(kv, words)
sim = cosine_matrix(look.vectors)
idx = {w: i for i, w in enumerate(look.words)}
table(["word A", "word B", "cosine"], [(a, b, f"{sim[idx[a], idx[b]]:.3f}") for a, b in PAIRS])

# Analogy
table(["rank", "king - man + woman", "cosine"], [(i + 1, w, f"{s:.3f}") for i, (w, s) in enumerate(analogy(kv, ["king", "woman"], ["man"], 5))])

# CLIP
paths = clipmod.list_sample_images()
if paths:
    names = [p.name for p in paths]
    iv = clipmod.embed_images([Image.open(p) for p in paths])
    tv = clipmod.embed_texts(QUERIES)
    s = clipmod.text_image_similarity(tv, iv)
    rows = []
    for q, row in zip(QUERIES, s):
        o = np.argsort(-row)
        rows.append((q, f"{names[o[0]]} ({row[o[0]]:.3f})", f"{names[o[1]]} ({row[o[1]]:.3f})", f"{names[o[-1]]} ({row[o[-1]]:.3f})"))
    table(["query", "#1", "#2", "last"], rows)
