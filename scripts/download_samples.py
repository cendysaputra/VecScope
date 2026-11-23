# Usage: python scripts/download_samples.py
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

BASE = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/"
# Local -> remote
SAMPLES = {
    "cats.jpg": "cats.jpeg",
    "cat-chonk.jpg": "pipeline-cat-chonk.jpeg",
    "bee.jpg": "cropped_bee.jpg",
    "dog.png": "image_process_dog.png",
    "baklava.jpg": "baklava.jpg",
    "mosque.jpg": "mosque.jpg",
    "robot.png": "robot.png",
}
MAX_SIDE = 640


def main(dest: str = "data/samples") -> int:
    out = Path(dest)
    out.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image
    except ImportError:
        Image = None
    for local, remote in SAMPLES.items():
        path = out / local
        if path.exists():
            print(f"skip  {path} (exists)")
            continue
        print(f"get   {remote} -> {path}")
        urllib.request.urlretrieve(BASE + remote, path)
        # Shrink large files
        if Image is not None:
            with Image.open(path) as im:
                im.thumbnail((MAX_SIDE, MAX_SIDE))
                (im.convert("RGB") if path.suffix == ".jpg" else im).save(path)
    print(f"done: {len(SAMPLES)} images in {out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
