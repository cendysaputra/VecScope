from __future__ import annotations

import html
import io

import numpy as np
import pandas as pd
import streamlit as st

from src import clip as clipmod
from src.embeddings import DEFAULT_MODEL, MODELS, get_vectors, load_glove, parse_words
from src.similarity import OOVError, analogy, analogy_vector, cosine_matrix, nearest, parse_analogy
from src.tokenize import DEFAULT_PAIR, TOKENIZERS, tokenize
from src.visualize import project_2d, scatter_2d, similarity_heatmap, vector_strips

st.set_page_config(page_title="VecScope", page_icon="🔬", layout="wide")

DEFAULT_SENTENCE = "Tokenization isn't magic: unbelievably, GPT-2 and BERT split words differently!"
DEFAULT_MAP = """people: king, queen, man, woman, prince, princess, boy, girl
animals: dog, cat, horse, lion, tiger, cow, sheep
cities: paris, london, tokyo, berlin, jakarta, rome, madrid"""
ANALOGY_EXAMPLES = [
    "king - man + woman",
    "paris - france + italy",
    "walking - walk + swim",
    "bigger - big + small",
    "tokyo - japan + indonesia",
]
TOKEN_COLORS = ["#fde68a", "#bbf7d0", "#bfdbfe", "#fbcfe8", "#ddd6fe", "#fed7aa"]


# Cached loaders
@st.cache_resource(show_spinner="Loading GloVe vectors (first run downloads ~66 MB)…")
def glove(name: str):
    return load_glove(name)


@st.cache_resource(show_spinner="Loading CLIP (first run downloads ~600 MB)…")
def clip_model():
    return clipmod.load_clip()


@st.cache_data(show_spinner=False)
def read_samples(stamp: tuple[tuple[str, float], ...]) -> list[bytes]:
    return [open(path, "rb").read() for path, _ in stamp]


@st.cache_data(show_spinner=False)
def clip_embed_images(blobs: tuple[bytes, ...]) -> np.ndarray:
    from PIL import Image

    clip_model()
    return clipmod.embed_images([Image.open(io.BytesIO(b)) for b in blobs])


@st.cache_data(show_spinner=False)
def clip_embed_texts(texts: tuple[str, ...]) -> np.ndarray:
    clip_model()
    return clipmod.embed_texts(list(texts))


# Helpers
def token_chips(tokens: list[str], ids: list[int]) -> str:
    chips = []
    for i, (t, tid) in enumerate(zip(tokens, ids)):
        color = TOKEN_COLORS[i % len(TOKEN_COLORS)]
        chips.append(
            f'<span title="id {tid}" style="background:{color};color:#111;border-radius:6px;padding:2px 6px;'
            f'margin:2px;display:inline-block;font-family:monospace;font-size:0.95rem">{html.escape(t)}'
            f'<sub style="color:#555;margin-left:4px">{tid}</sub></span>'
        )
    return "<div style='line-height:2.1'>" + "".join(chips) + "</div>"


def warn_missing(missing: list[str]) -> None:
    if missing:
        st.warning(
            f"Not in the GloVe vocabulary (skipped): **{', '.join(missing)}**. "
            "GloVe is lowercase English; try a more common spelling."
        )


def center_unit(v: np.ndarray) -> np.ndarray:
    # Center, renormalize
    v = v - v.mean(0)
    return v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-12, None)


def parse_groups(text: str) -> tuple[list[str], list[str]]:
    # "group: w1, w2"
    words, groups = [], []
    for line in text.splitlines():
        if not line.strip():
            continue
        group, _, rest = line.partition(":") if ":" in line else ("words", "", line)
        for w in parse_words(rest):
            if w not in words:
                words.append(w)
                groups.append(group.strip() or "words")
    return words, groups


# Sidebar
with st.sidebar:
    st.title("🔬 VecScope")
    st.caption("**vector + scope** — see how models cut text into tokens and turn meaning into vectors.")
    model_name = st.selectbox("Word embeddings", list(MODELS), format_func=lambda k: MODELS[k], index=list(MODELS).index(DEFAULT_MODEL))
    projection = st.radio("2D projection", ["pca", "umap"], format_func=str.upper, horizontal=True, help="PCA is deterministic and the default. UMAP needs `pip install umap-learn`.")
    st.divider()
    st.markdown(
        "Colors in vector strips: **red = positive**, **blue = negative**, white ≈ 0 "
        "([Illustrated Word2vec](https://jalammar.github.io/illustrated-word2vec/) style)."
    )

tabs = st.tabs(["🔤 Tokens", "🌈 Vector strips", "📐 Similarity", "🗺️ 2D map", "➕ Analogy", "🖼️ Images (CLIP)"])

# Tokens tab
with tabs[0]:
    st.subheader("Same sentence, different tokenizers")
    text = st.text_area("Text", DEFAULT_SENTENCE, height=90)
    c1, c2, c3 = st.columns([2, 2, 1])
    names = list(TOKENIZERS)
    a = c1.selectbox("Tokenizer A", names, index=names.index(DEFAULT_PAIR[0]))
    b = c2.selectbox("Tokenizer B", names, index=names.index(DEFAULT_PAIR[1]))
    special = c3.checkbox("BERT [CLS]/[SEP]", value=False, help="Add BERT's special tokens (off = fair count comparison).")
    if text.strip():
        results = [tokenize(text, a, special), tokenize(text, b, special)]
        cols = st.columns(2)
        for col, r in zip(cols, results):
            with col:
                st.metric(r.name, f"{r.count} tokens")
                st.markdown(token_chips(r.tokens, r.ids), unsafe_allow_html=True)
                st.dataframe(pd.DataFrame(r.rows()), hide_index=True, width="stretch", height=260)
        diff = results[1].count - results[0].count
        st.info(
            f"**{results[0].name}**: {results[0].count} tokens · **{results[1].name}**: {results[1].count} tokens "
            f"(difference {diff:+d}). `␣` marks a leading space in BPE tokens; `##` marks a WordPiece continuation."
        )

kv = glove(model_name)

# Strips tab
with tabs[1]:
    st.subheader("Each word is a row of numbers")
    words_in = st.text_input("Words (comma or space separated)", "king, man, woman, queen", key="strip_words")
    look = get_vectors(kv, parse_words(words_in))
    warn_missing(look.missing)
    if look.words:
        st.plotly_chart(vector_strips(look.words, look.vectors, title=f"{model_name} — {look.dim} dimensions"), width="stretch")
        with st.expander("Raw numbers"):
            st.dataframe(pd.DataFrame(look.vectors, index=look.words).round(3), width="stretch")

# Similarity tab
with tabs[2]:
    st.subheader("Cosine similarity")
    words_in = st.text_input("Words", "king, queen, man, woman, water, apple", key="sim_words")
    look = get_vectors(kv, parse_words(words_in))
    warn_missing(look.missing)
    if len(look.words) >= 2:
        sim = cosine_matrix(look.vectors)
        st.plotly_chart(similarity_heatmap(look.words, sim), width="stretch")
        pairs = [
            {"word A": look.words[i], "word B": look.words[j], "cosine": round(float(sim[i, j]), 4)}
            for i in range(len(look.words))
            for j in range(i + 1, len(look.words))
        ]
        st.dataframe(pd.DataFrame(pairs).sort_values("cosine", ascending=False), hide_index=True, width="stretch")
    st.markdown("**Nearest neighbours**")
    nn_word = st.text_input("Word", "king", key="nn_word")
    try:
        st.dataframe(pd.DataFrame(nearest(kv, nn_word, 10), columns=["word", "cosine"]).round(4), hide_index=True)
    except OOVError as e:
        st.warning(str(e))

# Map tab
with tabs[3]:
    st.subheader("Meaning on a map")
    st.caption("One group per line as `group: word, word, …`. Words with similar meaning should cluster.")
    groups_in = st.text_area("Words", DEFAULT_MAP, height=110)
    words, groups = parse_groups(groups_in)
    look = get_vectors(kv, words)
    warn_missing(look.missing)
    if len(look.words) >= 3:
        grp = [groups[words.index(w)] for w in look.words]
        try:
            pts = project_2d(look.vectors, projection)
            st.plotly_chart(
                scatter_2d(look.words, pts, grp, title=f"{projection.upper()} of {model_name}", axis_prefix="PC" if projection == "pca" else "UMAP"),
                width="stretch",
            )
        except RuntimeError as e:
            st.error(str(e))
    else:
        st.info("Enter at least 3 known words.")

# Analogy tab
with tabs[4]:
    st.subheader("Vector arithmetic")
    c1, c2 = st.columns([3, 1])
    example = c1.selectbox("Example", ANALOGY_EXAMPLES)
    topn = c2.slider("Top-k", 3, 20, 10)
    expr = st.text_input("Expression", example, key=f"analogy_{example}")
    pos, neg = parse_analogy(expr)
    try:
        res = analogy(kv, pos, neg, topn=topn)
        st.markdown(f"`{expr}` ≈ **{res[0][0]}** (cosine {res[0][1]:.3f})")
        st.dataframe(pd.DataFrame(res, columns=["word", "cosine"]).round(4), hide_index=True, width="stretch")
        st.caption("Gensim's `most_similar` excludes the input words, so the answer can't simply be one of them.")
        vec = analogy_vector(kv, pos, neg)
        operands = pos + neg
        strip_labels = operands + [f"= {expr}", f"top: {res[0][0]}"]
        unit = np.vstack([kv.get_vector(w, norm=True) for w in operands] + [vec / np.linalg.norm(vec), kv.get_vector(res[0][0], norm=True)])
        st.plotly_chart(vector_strips(strip_labels, unit, title="Unit vectors: operands, result, top answer"), width="stretch")
    except OOVError as e:
        st.warning(str(e))
    except ValueError as e:
        st.warning(str(e))

# CLIP tab
with tabs[5]:
    st.subheader("Text and images in one vector space (CLIP ViT-B/32)")
    st.caption(
        "CLIP vectors are only compared with other CLIP vectors — never with GloVe. "
        "Raw CLIP text↔image cosines are small (≈0.15–0.35); what matters is the ranking."
    )
    if not st.toggle("Enable CLIP (downloads ~600 MB on first use)", value=False, key="clip_on"):
        st.stop()

    sample_paths = clipmod.list_sample_images()
    uploads = st.file_uploader("Upload images (kept in memory, never saved)", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True) or []
    names = [p.name for p in sample_paths] + [f"upload:{u.name}" for u in uploads]
    blobs = read_samples(tuple((str(p), p.stat().st_mtime) for p in sample_paths)) + [u.getvalue() for u in uploads]
    blob_of = dict(zip(names, blobs))
    if not blobs:
        st.info("No images yet. Run `python scripts/download_samples.py` or upload some above.")
        st.stop()

    with st.spinner("Embedding images…"):
        img_vecs = clip_embed_images(tuple(blobs))

    st.markdown("**Search images with text**")
    query = st.text_input("Query", "a photo of a cat", key="clip_query")
    if query.strip():
        q = clip_embed_texts((query,))[0]
        ranked = clipmod.rank_images(q, img_vecs, names)
        cols = st.columns(min(6, len(ranked)))
        for k, (name, score) in enumerate(ranked[: len(cols)]):
            cols[k].image(blob_of[name], caption=f"#{k + 1} {name}\n{score:.3f}", width="stretch")
        st.dataframe(pd.DataFrame(ranked, columns=["image", "cosine"]).round(4), hide_index=True, width="stretch")

    st.markdown("**Text ↔ image similarity**")
    prompts_in = st.text_area("Prompts (one per line)", "a photo of a cat\na bee on a flower\na robot\na sweet pastry\na car on a highway", height=120)
    prompts = [p.strip() for p in prompts_in.splitlines() if p.strip()]
    if prompts:
        txt_vecs = clip_embed_texts(tuple(prompts))
        sim = clipmod.text_image_similarity(txt_vecs, img_vecs)
        fig = similarity_heatmap(prompts, sim, col_labels=names)
        fig.update_traces(zmin=float(sim.min()), zmax=float(sim.max()), zmid=None, colorscale="Viridis")
        st.plotly_chart(fig, width="stretch")

        st.markdown("**Image vector strip**")
        pick = st.selectbox("Image", names)
        i = names.index(pick)
        c1, c2 = st.columns([1, 4])
        c1.image(blobs[i], width="stretch")
        best = int(np.argmax(sim[:, i]))
        c2.plotly_chart(
            vector_strips([f"image: {pick}", f"text: {prompts[best]}"], np.vstack([img_vecs[i], txt_vecs[best]]), title="512-d CLIP embeddings (image vs best-matching prompt)", height_per_row=60),
            width="stretch",
        )

        st.markdown("**Mixed 2D map (CLIP space only)**")
        close_gap = st.checkbox(
            "Close the modality gap (center text and images separately)",
            help="CLIP keeps all text vectors in one region and all image vectors in another. Centering each group shows which text matches which image.",
        )
        all_vecs = np.vstack([center_unit(txt_vecs), center_unit(img_vecs)] if close_gap else [txt_vecs, img_vecs])
        labels = prompts + names
        kinds = ["text"] * len(prompts) + ["image"] * len(names)
        try:
            pts = project_2d(all_vecs, projection)
            st.plotly_chart(scatter_2d(labels, pts, kinds, symbols={"text": "circle", "image": "diamond"}, title=f"{projection.upper()} of CLIP text + image vectors"), width="stretch")
        except RuntimeError as e:
            st.error(str(e))
