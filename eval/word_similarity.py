# Usage: python eval/word_similarity.py
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gensim
import numpy as np
import scipy
from gensim.test.utils import datapath
from scipy.stats import spearmanr

from src.embeddings import MODELS, load_glove

DATASETS = {
    "wordsim353": ("wordsim353.tsv", "\t"),
    "simlex999": ("simlex999.txt", "\t"),
}
# Paper Table 3
PAPER_BASELINE = {"model": "glove-wiki-gigaword-300", "dataset": "wordsim353", "paper": "Pennington et al. (2014) Table 3", "spearman_100": 65.8}


def load_pairs(filename: str) -> list[tuple[str, str, float]]:
    pairs = []
    for line in Path(datapath(filename)).read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#") or line.lower().startswith("word1\t") or line.lower().startswith("word 1\t"):
            continue
        parts = line.split("\t")
        if len(parts) >= 3:
            try:
                pairs.append((parts[0].strip().lower(), parts[1].strip().lower(), float(parts[2])))
            except ValueError:
                continue
    return pairs


def evaluate_pairs(kv, filename: str) -> dict:
    # Raw and normalized
    _, spear_raw, oov_pct = kv.evaluate_word_pairs(datapath(filename), restrict_vocab=len(kv))
    pairs = load_pairs(filename)
    kept = [(w1, w2, score) for w1, w2, score in pairs if w1 in kv.key_to_index and w2 in kv.key_to_index]
    skipped = len(pairs) - len(kept)

    # Column L2 norm
    vecs = np.asarray(kv.vectors, dtype=np.float32)
    col_norms = np.clip(np.linalg.norm(vecs, axis=0, keepdims=True), 1e-12, None)
    idx1 = [kv.key_to_index[w1] for w1, _, _ in kept]
    idx2 = [kv.key_to_index[w2] for _, w2, _ in kept]
    v1 = vecs[idx1] / col_norms
    v2 = vecs[idx2] / col_norms
    v1 /= np.clip(np.linalg.norm(v1, axis=1, keepdims=True), 1e-12, None)
    v2 /= np.clip(np.linalg.norm(v2, axis=1, keepdims=True), 1e-12, None)
    sims_norm = np.sum(v1 * v2, axis=1)
    gold = np.array([s for _, _, s in kept], dtype=np.float32)
    rho_norm = float(spearmanr(gold, sims_norm).statistic)
    rho_raw = float(spear_raw.statistic)

    return {
        "total_pairs": len(pairs),
        "evaluated_pairs": len(kept),
        "skipped_oov": skipped,
        "oov_percent": round(float(oov_pct), 2),
        "spearman_raw": round(rho_raw, 4),
        "spearman_raw_100": round(rho_raw * 100, 1),
        "spearman_paper_norm": round(rho_norm, 4),
        "spearman_paper_norm_100": round(rho_norm * 100, 1),
    }


def main() -> int:
    results = {}
    print("| Model | Dim | WS353 Spearman (raw / Table 3 norm) | WS353 OOV | SimLex-999 Spearman (raw / norm) | SimLex-999 OOV |")
    print("|---|---|---|---|---|---|")
    for name in MODELS:
        kv = load_glove(name)
        ws = evaluate_pairs(kv, DATASETS["wordsim353"][0])
        sl = evaluate_pairs(kv, DATASETS["simlex999"][0])
        results[name] = {"dim": int(kv.vector_size), "vocab_size": len(kv), "wordsim353": ws, "simlex999": sl}
        print(
            f"| `{name}` | {kv.vector_size} "
            f"| {ws['spearman_raw']:.3f} ({ws['spearman_raw_100']:.1f}%) / {ws['spearman_paper_norm']:.3f} ({ws['spearman_paper_norm_100']:.1f}%) "
            f"| {ws['skipped_oov']}/{ws['total_pairs']} "
            f"| {sl['spearman_raw']:.3f} ({sl['spearman_raw_100']:.1f}%) / {sl['spearman_paper_norm']:.3f} ({sl['spearman_paper_norm_100']:.1f}%) "
            f"| {sl['skipped_oov']}/{sl['total_pairs']} |"
        )

    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "task": "word_similarity",
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "libraries": {"gensim": gensim.__version__, "numpy": np.__version__, "scipy": scipy.__version__},
        "paper_baseline": PAPER_BASELINE,
        "models": results,
    }
    out_path = out_dir / "word_similarity.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nsaved {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
