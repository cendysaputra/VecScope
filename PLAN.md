# Project Plan: VecScope

**VecScope** (*vector + scope*) is an interactive app for seeing how language models "read" text. You type a sentence, and the app shows:

- how the sentence gets cut into **tokens**, and
- how the meaning of a word is stored as a **vector** (a list of numbers).

In a later version you can also upload an image and compare it with text. This is not a chatbot. It's an exploration tool for a portfolio, in the same spirit as VisionIndetify.

Initial research: November 2026.

---

## 1. Goal

Explain tokens and embeddings visually. Users can:

1. Type a sentence and compare the output of two tokenizers.
2. See word vectors as colored strips, like the heatmaps in Jay Alammar's article.
3. Measure how similar two words are.
4. Try the analogy `king - man + woman ≈ queen`.

In v2, CLIP puts text and images in the same vector space. So similarity scores and the 2D map can hold text and images together.

---

## 2. Features

### v1: text first

- **Tokenization.** Sentence → list of tokens, token IDs, and token count. Compare GPT-2 BPE (`tiktoken`, `gpt2` encoding) with BERT WordPiece (`bert-base-uncased`). The counts usually differ.
- **Colored vector strips.** Use 50-dimension GloVe. One row per word. Red = positive, blue = negative, white = close to zero. See [Illustrated Word2vec](https://jalammar.github.io/illustrated-word2vec/) for the look.
- **Cosine similarity.** Similarity scores between words, as a table and a small heatmap.
- **2D map.** PCA by default because it's stable. UMAP is optional. Words with similar meaning should cluster.
- **Analogy.** Compute `king - man + woman` and show the top-k nearest words. `queen` is usually near the top, but not always first.

If a word isn't in the GloVe vocabulary (OOV), show a clear message. `glove-wiki-gigaword-*` is uncased, so all input is lowercased.

### v2: images with CLIP

- Upload an image → show its embedding strip in the same style.
- Compute cosine similarity between text and images.
- Search images with a text query: rank sample/uploaded images by score.
- 2D scatter with both text and image points. **CLIP space only.** Never mix it with GloVe vectors.

---

## 3. Tech stack

| Part | Choice | Why |
| --- | --- | --- |
| Language | Python | Already used in VisionIndetify / cifar10-app. |
| BPE tokenizer | [tiktoken](https://github.com/openai/tiktoken) | Official OpenAI library. `tiktoken.get_encoding("gpt2")` for GPT-2. Modern encodings (`cl100k_base`, `o200k_base`) can be added. |
| WordPiece tokenizer | [Hugging Face Transformers](https://huggingface.co/docs/transformers/en/fast_tokenizers) | `AutoTokenizer.from_pretrained("bert-base-uncased")`. BERT uses WordPiece. |
| Word embeddings (v1) | [Gensim downloader](https://radimrehurek.com/gensim/downloader.html) | `api.load("glove-wiki-gigaword-50")`. 50 dimensions, fits Alammar's visual style, and much lighter than Word2Vec Google News 300d. Analogies use `most_similar(positive=[...], negative=[...])`. |
| Sentence embeddings (v2) | [Sentence Transformers](https://www.sbert.net/) | To compare static word vectors with sentence embeddings. Pick a light model from the official list. |
| Text + image (v2) | [CLIP](https://huggingface.co/docs/transformers/model_doc/clip) | Text and images share one vector space. Start with `openai/clip-vit-base-patch32` (ViT-B/32, common and fairly light) through Transformers. Alternative: `sentence-transformers/clip-ViT-B-32` for one `encode()` API for both. Needs Pillow to open images. |
| Dimensionality reduction | scikit-learn PCA; [UMAP](https://umap-learn.readthedocs.io/en/latest/basic_usage.html) optional | PCA is enough for v1. |
| Visualization | [Plotly](https://plotly.com/python/pca-visualization/) | Interactive heatmaps and scatter plots. |
| UI | [Streamlit](https://docs.streamlit.io/develop/tutorials/chat-and-llm-apps/build-conversational-apps) | Tabbed layouts are easy. Alternative: [Gradio](https://www.gradio.app/guides/creating-a-chatbot-fast). |

Don't commit vector files, model weights, or user uploads. Gensim caches to `~/gensim-data`; cache folders go in `.gitignore`.

---

## 4. Steps

### v1

1. Set up `README.md`, `requirements.txt`, and `.gitignore` in the [cendysaputra/VecScope](https://github.com/cendysaputra/VecScope) repo.
2. Build the tokenization module: one function for GPT-2 (tiktoken), one for BERT WordPiece. Output: tokens, IDs, count.
3. Load GloVe 50d. Write functions for vector lookup, cosine similarity, and analogies. Handle words missing from the vocabulary.
4. Build the colored strip heatmap with Plotly (diverging scale, e.g. RdBu).
5. Build the 2D scatter: PCA to 2 components, with word-name tooltips. Offer UMAP, but keep PCA as default.
6. Wire everything into `app.py` (Streamlit): text input, tokenizer/model picker, visualizations.
7. Test with fixed examples: `king, man, woman, queen`, plus one sentence that GPT-2 and BERT tokenize differently.
8. Write the README: how to run, model download sizes, and **evidence** (see section 7).

### v2 (after v1 works)

9. Load CLIP. Turn uploaded/sample images into vectors and show them with the same strip function.
10. Compute text ↔ image cosine. Add image search from a text query (show scores and ranking).
11. Build a mixed scatter in CLIP space, with different markers for text and images.

---

## 5. Repo structure

```text
VecScope/
├── README.md
├── PLAN.md
├── AGENT.md              # working rules for agents/contributors
├── LICENSE
├── requirements.txt
├── .gitignore            # __pycache__/, .venv/, model caches, uploads
├── app.py                # Streamlit UI
├── data/samples/         # public sample images (v2), downloaded by script
├── eval/                 # evaluation scripts + results (see section 8)
│   ├── analogy.py        # Google analogy, top-1 accuracy
│   ├── word_similarity.py  # WordSim-353 / SimLex-999, Spearman
│   ├── clip_cifar10.py   # CLIP zero-shot on CIFAR-10 test
│   └── results/          # JSON/log output per run
├── scripts/              # sample download, evidence, screenshots
├── tests/
└── src/
    ├── tokenize.py       # tiktoken vs BERT WordPiece
    ├── embeddings.py     # load GloVe, vector lookup
    ├── similarity.py     # cosine + analogy
    ├── visualize.py      # strip heatmap + PCA/UMAP scatter
    └── clip.py           # CLIP image & text embeddings (v2)
```

---

## 6. Milestones

- [x] **v1:** all text features from section 2. English, GloVe 50d, local Streamlit UI.
  - [ ] Text evaluation: analogy accuracy and word similarity for GloVe 50d / 100d / 300d (section 8).
- **v2:**
  - [x] CLIP image features (strips, similarity, search, mixed scatter).
  - [x] Modern tokenizers (`cl100k_base`, `o200k_base`).
  - [x] UMAP option (needs `pip install umap-learn`).
  - [ ] Sentence embeddings with Sentence Transformers.
  - [ ] Early Indonesian support.
  - [ ] Image evaluation: CLIP zero-shot accuracy on CIFAR-10 test; Recall@1 / Recall@5 optional (section 8).
- [ ] **v3:**
  - Better Indonesian embeddings, e.g. [IndoBERT](https://huggingface.co/indobenchmark/indobert-base-p1) or a multilingual Sentence Transformers model. Example analogy: `raja - pria + wanita`.
  - OpenAI's original CLIP is strongest in English. For Indonesian queries, try `sentence-transformers/clip-ViT-B-32-multilingual-v1`.
  - Deploy a demo to [Hugging Face Spaces](https://huggingface.co/docs/hub/spaces-sdks-streamlit). The official docs mark the built-in Streamlit SDK as *deprecated*, so follow the latest guide (Docker template for Streamlit, or Gradio).

---

## 7. Evidence in the README

Don't just write "it works". Attach screenshots and sample output:

- The same sentence, GPT-2 tokens vs BERT tokens (token list, IDs, count).
- Heatmap for `king` / `man` / `woman` / `queen`. The colors of `man` and `woman` should look more alike than either does to `king`.
- Cosine table: `king`–`queen` high, `king`–`water` low.
- 2D scatter: a people cluster (`king`, `queen`, `man`, `woman`) separate from animals and cities.
- Analogy: top-k results for `king - man + woman` with scores.
- **v2:** one image's embedding strip, text ↔ image cosine scores (matching vs non-matching query), image search results, and a mixed text + image scatter in CLIP space.

---

## 8. Evaluation

VecScope isn't a classifier, so there's no single accuracy number like in the CIFAR-10 project. Instead, each part gets its own metric on a standard dataset.

### Text (v1)

- **Word analogy accuracy.** Google analogy dataset (`questions-words.txt`, ships with Gensim, e.g. `king - man + woman = queen`). Score = % of questions where the top-1 answer is correct. Gensim: `kv.evaluate_word_analogies(datapath("questions-words.txt"))`. Report semantic and syntactic sections separately too.
- **Word similarity.** WordSim-353 and/or SimLex-999 (both ship with Gensim: `wordsim353.tsv`, `simlex999.txt`). Score = Spearman correlation between the model's cosine similarity and human ratings. Gensim: `kv.evaluate_word_pairs(...)`. Also report how many pairs were skipped as OOV.
- **Model comparison.** Put several embedding models in one table, e.g. GloVe 50d (~66 MB) vs 100d (~128 MB) vs 300d (~376 MB). Same datasets, same settings.

### Images (v2, CLIP)

- **Zero-shot accuracy on CIFAR-10 test** (10,000 images), no retraining. Each class becomes a prompt like `"a photo of a {label}."`; the prediction is the prompt with the highest cosine. Compare with the CNN in the cifar10-app repo (95.71%). Keep in mind that the CNN was trained on CIFAR-10 and CLIP wasn't, and CLIP upscales 32×32 images to 224×224. So this comparison shows the trade-off; it isn't a fair head-to-head.
- **Optional: text-to-image retrieval.** Recall@1 / Recall@5 on a captioned image set: for each caption, is the correct image in the top 1 / top 5?

### Evidence rules

- Every number in the README needs a script in `eval/` that reproduces it.
- Save the output of every run to `eval/results/` (JSON or log), with the model name/version, dataset, library versions, and date.
- Never write a number that hasn't actually been run. No estimates, no numbers copied from memory.

### Baselines to compare against

These are reference points, not percentage targets:

- **Text:** the numbers reported in the GloVe paper ([Pennington et al., 2014](https://nlp.stanford.edu/pubs/glove.pdf)) for the same datasets. Quote them with the table number. Expect smaller dimensions to score lower; the point is to show how much quality you trade for size.
- **Images:** chance level (10% on CIFAR-10), the zero-shot CIFAR-10 result reported in the CLIP paper ([Radford et al., 2021](https://arxiv.org/abs/2103.00020)) for ViT-B/32 (quote it with the table number), and the 95.71% CNN from cifar10-app.
- If your result is far from the paper's, check preprocessing and prompts first before writing it up.

---

## 9. Sources

1. [The Illustrated Word2vec (Jay Alammar)](https://jalammar.github.io/illustrated-word2vec/): colored strip heatmaps and the king/queen analogy.
2. [Tokens & Embeddings](https://dranubhaparashar.github.io/projects/lectures/tokens-embeddings/): tokens, token IDs, embeddings, cosine, word maps.
3. [tiktoken (OpenAI)](https://github.com/openai/tiktoken): official BPE tokenizer.
4. [WordPiece tokenization (Hugging Face Course)](https://huggingface.co/docs/course/chapter6/6): how BERT splits words.
5. [Gensim downloader](https://radimrehurek.com/gensim/downloader.html): load pretrained GloVe / Word2Vec.
6. [CLIP (Hugging Face Transformers)](https://huggingface.co/docs/transformers/model_doc/clip): official docs, checkpoint `openai/clip-vit-base-patch32`.

All links were checked as live in November 2026. Library versions change, so follow the official docs above.
