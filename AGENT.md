# AGENT.md

Rules for any agent or contributor working on VecScope. Read [PLAN.md](PLAN.md) for scope and [README.md](README.md) for what already works.

## Project

Streamlit app that visualises tokens and embeddings: tokenizer comparison, GloVe vector strips, cosine similarity, PCA/UMAP maps, word analogies, and CLIP text ↔ image search.

```text
app.py              UI only: layout, inputs, caching
src/tokenize.py     tokenizers
src/embeddings.py   GloVe loading and lookup
src/similarity.py   cosine, neighbours, analogy
src/visualize.py    Plotly figures
src/clip.py         CLIP embeddings
scripts/            samples, evidence, screenshots
tests/              pytest, fixed examples
```

Logic goes in `src/`. `app.py` only wires widgets to `src/` functions.

## Commands

```bash
pip install -r requirements.txt
```

```bash
streamlit run app.py
```

```bash
python -m pytest -q
```

```bash
python scripts/evidence.py
```

```bash
node scripts/screenshots.js
```

## Rules for every change

### 1. Optimize

- Load each model once: `lru_cache` in `src/`, `st.cache_resource` / `st.cache_data` in `app.py`.
- Import heavy libraries (`torch`, `transformers`, `gensim`) inside the function that needs them, not at module top.
- Use NumPy vector operations, not Python loops over dimensions or rows.
- Keep CLIP opt-in. The text tabs must work without downloading it.
- Don't add a dependency for something a few lines of NumPy can do.

### 2. Comments are labels only

- A comment is a 2–3 word label, e.g. `# Robust color range`, `# Single gather`.
- No docstrings, no paragraphs, no comments restating the code.
- If code needs a long explanation, rename things or split the function instead.

### 3. No AI slop

- No filler words ("seamlessly", "leverage", "robust solution", "powerful", "cutting-edge").
- No emoji in code, commits or docs.
- Every claim in README must be backed by a number from `scripts/evidence.py` or a screenshot. If you change a model, example or plot, re-run both and update README.
- Don't add features, options or abstractions nobody asked for.
- Don't leave dead code, unused imports or commented-out blocks.

## Git

- Work and push on `staging` only. Never push to `main` and never create other branches; the owner merges `staging` into `main`.
- Commit messages: one short line, imperative, under ~50 characters. Example: `Add UMAP toggle`.
- Never commit model files, vectors, caches, sample images or user uploads (see `.gitignore`).
- Run `python -m pytest -q` before pushing. Don't push red tests.

## Data and models

- GloVe is lowercase English; always go through `normalize()`.
- Never mix GloVe vectors and CLIP vectors in one plot or similarity.
- Sample images are CC BY-NC-SA 4.0 and are downloaded by `scripts/download_samples.py`, not committed.
