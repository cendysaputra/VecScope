# Usage: python eval/analogy.py
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gensim
import numpy as np
from gensim.test.utils import datapath

from src.embeddings import MODELS, load_glove

SEMANTIC_SECTIONS = {
    "capital-common-countries",
    "capital-world",
    "currency",
    "city-in-state",
    "family",
}
# Paper Table 2
PAPER_BASELINE = {
    "model": "glove-wiki-gigaword-300",
    "paper": "Pennington et al. (2014) Table 2",
    "semantic": 77.4,
    "syntactic": 67.0,
    "total": 71.7,
}


def load_sections(path: str) -> list[tuple[str, list[list[str]]]]:
    sections: list[tuple[str, list[list[str]]]] = []
    name, rows = None, []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        if line.startswith(": "):
            if name is not None:
                sections.append((name, rows))
            name, rows = line[2:].strip(), []
        else:
            rows.append([w.lower() for w in line.strip().split()])
    if name is not None:
        sections.append((name, rows))
    return sections


def evaluate_analogies(kv, path: str, batch_size: int = 1024, gens_cap: int = 300000) -> tuple[dict, dict]:
    # Single-pass 400k and 300k
    n_vocab = len(kv)
    W = np.ascontiguousarray(kv.get_normed_vectors(), dtype=np.float32)
    WT = W.T
    vocab = kv.key_to_index
    sections = load_sections(path)

    all_idx = []
    sec_meta = []
    oov_400k = oov_300k = 0
    for name, qs in sections:
        kept = [[vocab[w] for w in q] for q in qs if all(w in vocab for w in q)]
        oov_400k += len(qs) - len(kept)
        start = len(all_idx)
        all_idx.extend(kept)
        sec_meta.append((name, len(qs), start, len(all_idx)))

    idx_arr = np.array(all_idx, dtype=np.int32)
    a, b, c, d = idx_arr.T
    n = len(a)
    pred_400k = np.empty(n, dtype=np.int32)
    pred_300k = np.empty(n, dtype=np.int32)

    for s in range(0, n, batch_size):
        sl = slice(s, min(s + batch_size, n))
        qvec = W[b[sl]] - W[a[sl]] + W[c[sl]]
        scores = qvec @ WT
        rows = np.arange(len(a[sl]))
        scores[rows, a[sl]] = -np.inf
        scores[rows, b[sl]] = -np.inf
        scores[rows, c[sl]] = -np.inf
        b300 = np.argmax(scores[:, :gens_cap], axis=1)
        m300 = scores[rows, b300]
        btail = gens_cap + np.argmax(scores[:, gens_cap:], axis=1)
        mtail = scores[rows, btail]
        pred_300k[sl] = b300
        pred_400k[sl] = np.where(mtail > m300, btail, b300)

    hit_400k = pred_400k == d
    in_300k = (idx_arr < gens_cap).all(axis=1)
    hit_300k = (pred_300k == d) & in_300k

    per_section = {}
    sem_c = sem_t = syn_c = syn_t = 0
    sem_c3 = sem_t3 = syn_c3 = syn_t3 = 0
    for name, total_qs, start, end in sec_meta:
        sl = slice(start, end)
        c4, t4 = int(hit_400k[sl].sum()), end - start
        c3, t3 = int(hit_300k[sl].sum()), int(in_300k[sl].sum())
        oov_300k += total_qs - t3
        kind = "semantic" if name in SEMANTIC_SECTIONS else "syntactic"
        per_section[name] = {"kind": kind, "correct": c4, "total": t4, "accuracy": round(100.0 * c4 / t4, 2)}
        if kind == "semantic":
            sem_c, sem_t = sem_c + c4, sem_t + t4
            sem_c3, sem_t3 = sem_c3 + c3, sem_t3 + t3
        else:
            syn_c, syn_t = syn_c + c4, syn_t + t4
            syn_c3, syn_t3 = syn_c3 + c3, syn_t3 + t3

    tot_c, tot_t = sem_c + syn_c, sem_t + syn_t
    tot_c3, tot_t3 = sem_c3 + syn_c3, sem_t3 + syn_t3
    full = {
        "restrict_vocab": n_vocab,
        "total_questions": tot_t + oov_400k,
        "evaluated_questions": tot_t,
        "skipped_oov": oov_400k,
        "semantic_correct": sem_c,
        "semantic_total": sem_t,
        "semantic_accuracy": round(100.0 * sem_c / sem_t, 2),
        "syntactic_correct": syn_c,
        "syntactic_total": syn_t,
        "syntactic_accuracy": round(100.0 * syn_c / syn_t, 2),
        "total_correct": tot_c,
        "total_accuracy": round(100.0 * tot_c / tot_t, 2),
        "sections": per_section,
    }
    r300k = {
        "restrict_vocab": gens_cap,
        "evaluated_questions": tot_t3,
        "skipped_oov": oov_300k,
        "semantic_accuracy": round(100.0 * sem_c3 / sem_t3, 2),
        "syntactic_accuracy": round(100.0 * syn_c3 / syn_t3, 2),
        "total_accuracy": round(100.0 * tot_c3 / tot_t3, 2),
    }
    return full, r300k


def main() -> int:
    path = datapath("questions-words.txt")
    results = {}
    print("| Model | Dim | Semantic (%) | Syntactic (%) | Total (%) | Evaluated / Total (400k vocab) | Gensim 300k Total (%) |", flush=True)
    print("|---|---|---|---|---|---|---|", flush=True)
    for name in MODELS:
        kv = load_glove(name)
        full, r300k = evaluate_analogies(kv, path)
        results[name] = {
            "dim": int(kv.vector_size),
            "vocab_size": len(kv),
            "full_vocab_400k": full,
            "gensim_default_300k": r300k,
        }
        print(
            f"| `{name}` | {kv.vector_size} "
            f"| {full['semantic_accuracy']:.2f}% "
            f"| {full['syntactic_accuracy']:.2f}% "
            f"| {full['total_accuracy']:.2f}% "
            f"| {full['evaluated_questions']}/{full['total_questions']} "
            f"| {r300k['total_accuracy']:.2f}% ({r300k['evaluated_questions']}/{full['total_questions']}) |",
            flush=True,
        )

    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "task": "word_analogy",
        "dataset": "Google analogy (questions-words.txt)",
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "libraries": {"gensim": gensim.__version__, "numpy": np.__version__},
        "paper_baseline": PAPER_BASELINE,
        "models": results,
    }
    out_path = out_dir / "analogy.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nsaved {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
