# VecScope

**vector + scope**: a small Streamlit app for looking inside language models. Type a sentence and see how it gets cut into tokens, how each word becomes a list of numbers, and how those numbers capture meaning. Upload an image and see it land in the same vector space as text.

![VecScope demo: tokens, vector strips, 2D map, analogy, CLIP search](docs/demo.gif)

![VecScope, light theme](docs/screenshots/overview.png)

The app follows the system light/dark setting (switchable in the app menu):

![VecScope, dark theme](docs/screenshots/overview-dark.png)

## Why

"Tokens" and "embeddings" are usually explained with a diagram and a promise. VecScope lets you poke at the real thing:

- Two tokenizers rarely agree. GPT-2 splits the test sentence below into 19 tokens, BERT into 23.
- A word vector is just 50 numbers, but `man` and `woman` have visibly similar color patterns, and `king - man + woman` really does land next to `queen`.
- CLIP puts a photo of a cat and the text "a photo of a cat" close together, which is what makes text-to-image search work.

It is a learning and portfolio project, not a chatbot. Full plan: [PLAN.md](PLAN.md).

## What it does

| Tab | Shows |
| --- | --- |
| Tokens | One sentence, two tokenizers side by side: GPT-2 BPE (`tiktoken`), BERT WordPiece, GPT-4 `cl100k_base`, GPT-4o `o200k_base`. Tokens, IDs, counts. |
| Vector strips | GloVe word vectors as colored rows. Red = positive, blue = negative. |
| Similarity | Cosine similarity matrix, pair table, nearest neighbours. |
| 2D map | PCA (default) or UMAP projection of grouped words. |
| Analogy | `a - b + c` with top-k results and the vectors involved. |
| Images (CLIP) | Text-to-image search, text ↔ image similarity, 512-d image strips, mixed text + image map. |

Words not in the GloVe vocabulary are skipped with a warning. GloVe is lowercase English, so input is lowercased.

## Run it

Python 3.10+.

```bash
python -m venv .venv
```

```bash
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

```bash
python scripts/download_samples.py
```

```bash
streamlit run app.py
```

Models download on first use and are cached outside the repo:

| Model | Size | Cache |
| --- | --- | --- |
| `glove-wiki-gigaword-50` | ~66 MB | `~/gensim-data` (plus `.kv` mmap cache) |
| `glove-wiki-gigaword-100` | ~128 MB | `~/gensim-data` (plus `.kv` mmap cache) |
| `glove-wiki-gigaword-300` | ~376 MB | `~/gensim-data` (plus `.kv` mmap cache) |
| `bert-base-uncased` tokenizer | <1 MB | `~/.cache/huggingface` |
| `tiktoken` encodings | 1–4 MB each | tiktoken cache |
| `openai/clip-vit-base-patch32` | ~600 MB | `~/.cache/huggingface` |

CLIP only loads after you switch it on in the Images tab. UMAP is optional: `pip install umap-learn`.

Tests (downloads GloVe and the BERT vocab on first run):

```bash
python -m pytest -q
```

## Evidence

Numbers below come from `python scripts/evidence.py`. Screenshots come from `node scripts/screenshots.js` and the demo GIF from `node scripts/demo_gif.js` (needs ffmpeg), both against a running app.

### Tokens: GPT-2 vs BERT

Sentence: `Tokenization isn't magic: unbelievably, GPT-2 and BERT split words differently!`

**GPT-2 BPE: 19 tokens**

```text
Token (30642) | ization (1634) | ␣isn (2125) | 't (470) | ␣magic (5536) | : (25) | ␣unbelievably (48943) | , (11) | ␣G (402) | PT (11571) | - (12) | 2 (17) | ␣and (290) | ␣B (347) | ERT (17395) | ␣split (6626) | ␣words (2456) | ␣differently (10338) | ! (0)
```

**BERT WordPiece: 23 tokens**

```text
token (19204) | ##ization (3989) | isn (3475) | ' (1005) | t (1056) | magic (3894) | : (1024) | un (4895) | ##bel (8671) | ##ie (2666) | ##va (3567) | ##bly (6321) | , (1010) | gp (14246) | ##t (2102) | - (1011) | 2 (1016) | and (1998) | bert (14324) | split (3975) | words (2616) | differently (11543) | ! (999)
```

GPT-2 keeps `unbelievably` as one token and splits `GPT` into `G` + `PT`. BERT lowercases everything and breaks `unbelievably` into five pieces. `␣` marks a leading space; `##` marks a word continuation.

![Tokens tab](docs/screenshots/tokens.png)

### Vector strips

All four words are strongly blue at dimension 25 and red at 30. Beyond that, `man` and `woman` share red at dimensions 4 and 21, which `king` lacks; `king` and `queen` share a deep blue at 47. Euclidean distance agrees: `man`–`woman` is 2.60, `king`–`man` is 5.14.

![Vector strips](docs/screenshots/strips.png)

### Cosine similarity (GloVe 50d)

| word A | word B | cosine |
|---|---|---|
| king | queen | 0.784 |
| man | woman | 0.886 |
| king | man | 0.531 |
| king | woman | 0.411 |
| king | water | 0.235 |
| queen | apple | 0.204 |

![Similarity](docs/screenshots/similarity.png)

### 2D map (PCA)

People, animals and cities form three separate groups. Royalty (`king`, `queen`, `prince`, `princess`) sits apart from `man`, `woman`, `boy`, `girl`.

![2D map](docs/screenshots/map2d.png)

### Analogy: king - man + woman

| rank | word | cosine |
|---|---|---|
| 1 | queen | 0.852 |
| 2 | throne | 0.766 |
| 3 | prince | 0.759 |
| 4 | daughter | 0.747 |
| 5 | elizabeth | 0.746 |

Gensim's `most_similar` excludes the input words, so `king` itself can't be the answer.

![Analogy](docs/screenshots/analogy.png)

### CLIP: text ↔ image

Seven sample images, five text queries. Raw CLIP cosines between text and images are small (around 0.1–0.3); the ranking is what matters. The last query has no matching image and its best score is the lowest of the five.

| query | #1 | #2 | last |
|---|---|---|---|
| a photo of a cat | cats.jpg (0.267) | cat-chonk.jpg (0.253) | mosque.jpg (0.150) |
| a bee on a flower | bee.jpg (0.298) | robot.png (0.169) | dog.png (0.114) |
| a robot | robot.png (0.273) | cat-chonk.jpg (0.185) | baklava.jpg (0.141) |
| a sweet pastry | baklava.jpg (0.243) | cat-chonk.jpg (0.216) | mosque.jpg (0.157) |
| a car on a highway | robot.png (0.177) | cat-chonk.jpg (0.174) | baklava.jpg (0.121) |

![CLIP tab](docs/screenshots/clip.png)

In the mixed map, all text sits on one side and all images on the other. This is CLIP's known *modality gap*: the two encoders use different regions of the shared space. Ticking "Close the modality gap" centers each group separately, and matching pairs line up (`a photo of a cat` next to the cat photos, `a bee on a flower` next to `bee.jpg`):

![CLIP map with modality gap closed](docs/screenshots/clip-gap-closed.png)

## Evaluation

Reproducible benchmark scripts live in `eval/` and write JSON logs with library versions and timestamps to `eval/results/`.

### Text: GloVe 50d vs 100d vs 300d

Run `python eval/analogy.py` ([`eval/results/analogy.json`](eval/results/analogy.json)) and `python eval/word_similarity.py` ([`eval/results/word_similarity.json`](eval/results/word_similarity.json)):

| Model | Size | Dim | Analogy Semantic | Analogy Syntactic | Analogy Total (400k / Gensim 300k) | WordSim-353 Spearman (raw / Table 3 norm) | SimLex-999 Spearman (raw / norm) | OOV skipped |
|---|---|---|---|---|---|---|---|---|
| `glove-wiki-gigaword-50` | ~66 MB | 50 | 48.46% | 44.36% | 46.22% / 46.37% | 0.503 (50.3%) / 0.567 (56.7%) | 0.265 (26.5%) / 0.292 (29.2%) | 0/19,544 · 0/353 · 0/999 |
| `glove-wiki-gigaword-100` | ~128 MB | 100 | 65.34% | 61.26% | 63.11% / 63.30% | 0.533 (53.3%) / 0.594 (59.4%) | 0.298 (29.8%) / 0.329 (32.9%) | 0/19,544 · 0/353 · 0/999 |
| `glove-wiki-gigaword-300` | ~376 MB | 300 | 77.44% | 67.00% | 71.74% / 71.95% | 0.609 (60.9%) / 0.658 (65.8%) | 0.371 (37.1%) / 0.408 (40.8%) | 0/19,544 · 0/353 · 0/999 |
| GloVe paper ([Pennington et al., 2014](https://nlp.stanford.edu/pubs/glove.pdf), 6B 300d) | — | 300 | 77.4% (Table 2) | 67.0% (Table 2) | 71.7% (Table 2) | 65.8% (Table 3) | — | 0 |

- **Vocabulary cutoff.** Evaluating across the full 400,000-word vocabulary covers all 19,544 Google analogy questions (`questions-words.txt`, 8,869 semantic + 10,675 syntactic, 0 OOV) and matches Table 2 of Pennington et al. (2014) on 300d (`77.44%` semantic, `67.00%` syntactic, `71.74%` total vs paper `77.4%` / `67.0%` / `71.7%`). Gensim's default `restrict_vocab=300000` skips 58 questions and yields `71.95%`.
- **Feature normalization on WordSim-353.** Section 4.3 (Table 3) of Pennington et al. (2014) normalizes each feature dimension across the 400,000-word vocabulary before computing cosine similarity. With feature normalization, `glove-wiki-gigaword-300` scores `65.8%` ($100 \times \rho$, $\rho = 0.6581$) on WordSim-353, matching Table 3 (`65.8`). Raw unnormalized cosine scores `60.9%` ($\rho = 0.6085$).
- **Size vs quality trade-off.** Dropping from 300d (~376 MB) to 50d (~66 MB, 5.7× smaller) trades 25.52 percentage points of total analogy accuracy (`71.74%` → `46.22%`) and 9.1 points of WordSim-353 Spearman correlation (`65.8%` → `56.7%`), while remaining compact enough for 50-column visual strips.

### Images: CLIP ViT-B/32 zero-shot on CIFAR-10 test

Run `python eval/clip_cifar10.py` ([`eval/results/clip_cifar10.json`](eval/results/clip_cifar10.json)) on all 10,000 CIFAR-10 test images (32×32 upscaled to 224×224, no fine-tuning):

| Method / Baseline | CIFAR-10 test accuracy | Details |
|---|---|---|
| Chance level | 10.00% | 10 balanced classes (1,000 images each) |
| CLIP ViT-B/32, single prompt (`"a photo of a {label}."`) | 88.30% (8,830 / 10,000) | `openai/clip-vit-base-patch32`, zero-shot |
| CLIP ViT-B/32, 18-prompt ensemble | 89.83% (8,983 / 10,000) | OpenAI CIFAR-10 prompt templates, zero-shot |
| CLIP paper ViT-B/32 ([Radford et al., 2021](https://arxiv.org/abs/2103.00020), Table 10 / Table 11) | 89.8% | Reported zero-shot benchmark in paper |
| Supervised CNN (`cifar10-app`) | 95.71% | Trained directly on 32×32 CIFAR-10 images |

- **Prompt ensembling.** Averaging unit text embeddings across OpenAI's 18 CIFAR-10 templates (including `"a blurry photo of a {label}."` and `"a low contrast photo of a {label}."` for 32×32 thumbnails) lifts zero-shot accuracy from `88.30%` to `89.83%`, matching the `89.8%` reported for ViT-B/32 in Radford et al. (2021).
- **Sample text-to-image retrieval.** On the 7 captioned sample images (`cats.jpg`, `cat-chonk.jpg`, `bee.jpg`, `dog.png`, `baklava.jpg`, `mosque.jpg`, `robot.png`), CLIP ViT-B/32 achieves **Recall@1 = 100.00% (7/7)** and **Recall@5 = 100.00% (7/7)**.

## Project layout

```text
app.py                 Streamlit UI
.streamlit/config.toml light and dark theme
src/tokenize.py        tiktoken vs BERT WordPiece
src/embeddings.py      GloVe loading and lookup
src/similarity.py      cosine, nearest neighbours, analogy
src/visualize.py       Plotly strips, heatmaps, PCA/UMAP scatter
src/clip.py            CLIP text and image embeddings
eval/                  analogy, word similarity, CLIP CIFAR-10 benchmarks + JSON logs
scripts/               sample download, evidence, screenshots, demo GIF
tests/                 fixed examples from the plan
```

## Notes

- Sample images come from [`huggingface/documentation-images`](https://huggingface.co/datasets/huggingface/documentation-images) (CC BY-NC-SA 4.0). They are downloaded by the script, not committed.
- Uploaded images stay in memory and are never written to disk.
- Behind a restrictive proxy, Hugging Face's Xet downloads may fail. `HF_HUB_DISABLE_XET=1` falls back to plain HTTPS.

## Roadmap

Done: everything in v1 (including GloVe 50d / 100d / 300d analogy and word similarity evaluation), plus CLIP, modern tiktoken encodings, the UMAP option, and CLIP CIFAR-10 zero-shot evaluation from v2. Next: Sentence Transformers sentence embeddings, then Indonesian support (IndoBERT or a multilingual model, `raja - pria + wanita`) and a Hugging Face Spaces demo. See [PLAN.md](PLAN.md#6-milestones).

## Credits

Visual style from Jay Alammar's [The Illustrated Word2vec](https://jalammar.github.io/illustrated-word2vec/). Built on [tiktoken](https://github.com/openai/tiktoken), [Transformers](https://huggingface.co/docs/transformers), [Gensim](https://radimrehurek.com/gensim/), [GloVe](https://nlp.stanford.edu/projects/glove/) and [CLIP](https://huggingface.co/openai/clip-vit-base-patch32).

## License

[MIT](LICENSE) © 2026 Cendy Saputra. Covers the code in this repo only. Downloaded models and sample images keep their own licenses (GloVe: PDDL, CLIP: MIT, sample images: CC BY-NC-SA 4.0).
