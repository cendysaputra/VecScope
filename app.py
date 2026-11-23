from __future__ import annotations

import html
import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src import clip as clipmod
from src.embeddings import DEFAULT_MODEL, MODELS, get_vectors, load_glove, parse_groups, parse_words
from src.similarity import (
    OOVError,
    analogy,
    analogy_vector,
    center_unit,
    cosine_matrix,
    l2_normalize,
    nearest,
    pairwise_scores,
    parse_analogy,
    unit_vectors,
)
from src.tokenize import DEFAULT_PAIR, TOKENIZERS, tokenize
from src.visualize import diverging_hex, palette, project_2d, scatter_2d, similarity_heatmap, vector_strips

st.set_page_config(page_title="VecScope", page_icon=":material/biotech:", layout="wide")

REPO_URL = "https://github.com/cendysaputra/VecScope"
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
# Chip tints
CHIP_RGB = ["42,120,214", "235,104,52", "27,175,122"]
MODE = getattr(st.context.theme, "type", None) or "light"

CSS = """
<style>
.block-container {max-width: 1180px; padding-top: 2.2rem;}
.vs-hero {padding: 0.4rem 0 1.2rem;}
.vs-eyebrow {font-size: .78rem; letter-spacing: .14em; text-transform: uppercase; opacity: .6; margin-bottom: .2rem;}
.vs-title {font-size: 2.7rem; font-weight: 750; line-height: 1.05; margin: 0;}
.vs-lead {font-size: 1.08rem; opacity: .78; max-width: 46rem; margin: .5rem 0 1.1rem;}
.vs-strip {display: grid; grid-template-columns: repeat(50, 1fr); gap: 2px; max-width: 46rem;}
.vs-strip span {height: 20px; border-radius: 3px;}
.vs-note {font-size: .82rem; opacity: .6; margin-top: .45rem;}
.vs-lede {opacity: .72; margin: -.4rem 0 1rem; max-width: 52rem;}
.vs-chips {line-height: 2.25; margin: .2rem 0 .8rem;}
.vs-chip {border-radius: 6px; padding: 3px 7px; margin: 2px 3px 2px 0; display: inline-block;
  font-family: "Source Code Pro", ui-monospace, monospace; font-size: .9rem; white-space: pre;}
.vs-chip sub {opacity: .55; margin-left: 5px; font-size: .7rem;}
.vs-answer {border: 1px solid rgba(128,128,128,.28); border-radius: .8rem; padding: 1rem 1.2rem; margin: .4rem 0 1rem;}
.vs-answer .expr {font-family: "Source Code Pro", ui-monospace, monospace; opacity: .7; font-size: .95rem;}
.vs-answer .word {font-size: 2.2rem; font-weight: 750; line-height: 1.2;}
.vs-answer .score {opacity: .65; font-size: .9rem;}
.vs-legend {height: 12px; border-radius: 3px; margin: .3rem 0 .2rem;}
.vs-legend-labels {display: flex; justify-content: space-between; font-size: .78rem; opacity: .65;}
.vs-rank {font-size: .85rem; line-height: 1.35;}
.vs-rank b {font-size: 1rem;}
</style>
"""


# Cached loaders
@st.cache_resource(show_spinner="Loading GloVe vectors (first run downloads ~66 MB)…")
def glove(name: str):
    return load_glove(name)


@st.cache_resource(show_spinner="Loading CLIP (first run downloads ~600 MB)…")
def clip_model():
    return clipmod.load_clip()


@st.cache_data(show_spinner=False)
def read_samples(stamp: tuple[tuple[str, float], ...]) -> list[bytes]:
    return [Path(path).read_bytes() for path, _ in stamp]


@st.cache_data(show_spinner=False)
def clip_embed_images(blobs: tuple[bytes, ...]) -> np.ndarray:
    from PIL import Image

    clip_model()
    return clipmod.embed_images([Image.open(io.BytesIO(b)) for b in blobs])


@st.cache_data(show_spinner=False)
def clip_embed_texts(texts: tuple[str, ...]) -> np.ndarray:
    clip_model()
    return clipmod.embed_texts(list(texts))


@st.cache_data(show_spinner=False)
def cached_project_2d(vectors: np.ndarray, method: str) -> np.ndarray:
    return project_2d(vectors, method)


@st.cache_data(show_spinner=False)
def thumb(blob: bytes, side: int = 320) -> bytes:
    # Square center crop
    from PIL import Image, ImageOps

    out = io.BytesIO()
    ImageOps.fit(Image.open(io.BytesIO(blob)).convert("RGB"), (side, side)).save(out, "JPEG", quality=88)
    return out.getvalue()


# Helpers
def lede(text: str) -> None:
    st.markdown(f'<p class="vs-lede">{text}</p>', unsafe_allow_html=True)


def token_chips(tokens: list[str], ids: list[int]) -> str:
    chips = "".join(
        f'<span class="vs-chip" title="id {tid}" style="background:rgba({CHIP_RGB[i % 3]},.16);'
        f'box-shadow: inset 0 0 0 1px rgba({CHIP_RGB[i % 3]},.38)">{html.escape(t)}<sub>{tid}</sub></span>'
        for i, (t, tid) in enumerate(zip(tokens, ids))
    )
    return f'<div class="vs-chips">{chips}</div>'


def strip_html(values: np.ndarray) -> str:
    cells = "".join(f'<span style="background:{c}"></span>' for c in diverging_hex(values, MODE))
    return f'<div class="vs-strip">{cells}</div>'


def legend_html() -> str:
    stops = ", ".join(f"{c} {p * 100:.0f}%" for p, c in palette(MODE)["diverging"])
    return (
        f'<div class="vs-legend" style="background:linear-gradient(90deg, {stops})"></div>'
        '<div class="vs-legend-labels"><span>negative</span><span>0</span><span>positive</span></div>'
    )


def warn_missing(missing: list[str]) -> None:
    if missing:
        st.warning(
            f"Not in the GloVe vocabulary (skipped): **{', '.join(missing)}**. "
            "GloVe is lowercase English; try a more common spelling.",
            icon=":material/help:",
        )


st.markdown(CSS, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### VecScope")
    st.caption("Tokens and embeddings, up close.")
    model_name = st.selectbox("Word embeddings", list(MODELS), format_func=lambda k: MODELS[k], index=list(MODELS).index(DEFAULT_MODEL))
    projection = st.segmented_control(
        "2D projection", ["pca", "umap"], default="pca", format_func=str.upper,
        help="PCA is deterministic and the default. UMAP needs `pip install umap-learn`.",
    ) or "pca"
    st.divider()
    st.markdown("**Reading the colors**")
    st.markdown(legend_html(), unsafe_allow_html=True)
    st.caption("Every strip and cosine heatmap uses this scale. Gray means close to zero.")
    st.divider()
    st.markdown(
        f"[Source on GitHub]({REPO_URL})  \n"
        "[The Illustrated Word2vec](https://jalammar.github.io/illustrated-word2vec/)"
    )

kv = glove(model_name)

# Hero
st.markdown(
    '<div class="vs-hero">'
    '<div class="vs-eyebrow">vector + scope</div>'
    '<h1 class="vs-title">VecScope</h1>'
    '<p class="vs-lead">Type text and watch it become tokens, then vectors you can compare, '
    "add, subtract and plot.</p>"
    f"{strip_html(kv['scope'])}"
    f'<div class="vs-note">The strip above is the real {model_name} vector for "scope": '
    f"{kv.vector_size} numbers, red positive, blue negative.</div>"
    "</div>",
    unsafe_allow_html=True,
)

tabs = st.tabs([
    ":material/data_array: Tokens",
    ":material/view_week: Vector strips",
    ":material/compare_arrows: Similarity",
    ":material/scatter_plot: 2D map",
    ":material/functions: Analogy",
    ":material/image: Images (CLIP)",
])

# Tokens tab
with tabs[0]:
    lede("Models never see words, only token IDs. Each tokenizer cuts the same text differently.")
    text = st.text_area("Text", DEFAULT_SENTENCE, height=80)
    c1, c2, c3 = st.columns([2, 2, 1], vertical_alignment="bottom")
    names = list(TOKENIZERS)
    a = c1.selectbox("Tokenizer A", names, index=names.index(DEFAULT_PAIR[0]))
    b = c2.selectbox("Tokenizer B", names, index=names.index(DEFAULT_PAIR[1]))
    special = c3.checkbox("BERT [CLS]/[SEP]", value=False, help="Add BERT's special tokens. Off keeps the count comparison fair.")
    if text.strip():
        results = [tokenize(text, a, special), tokenize(text, b, special)]
        for col, r in zip(st.columns(2, gap="large"), results):
            with col.container(border=True):
                st.metric(r.name, f"{r.count} tokens")
                st.markdown(token_chips(r.tokens, r.ids), unsafe_allow_html=True)
                with st.expander("Token table"):
                    st.dataframe(pd.DataFrame(r.rows()), hide_index=True, width="stretch", height=260)
        st.caption(
            f"Difference: {results[1].count - results[0].count:+d} tokens. "
            "`␣` marks a leading space in BPE tokens; `##` marks a WordPiece continuation. Hover a chip for its ID."
        )

# Strips tab
with tabs[1]:
    lede(f"Each word is a list of {kv.vector_size} numbers. Related words share color patterns.")
    words_in = st.text_input("Words (comma or space separated)", "king, man, woman, queen", key="strip_words")
    look = get_vectors(kv, parse_words(words_in))
    warn_missing(look.missing)
    if look.words:
        st.plotly_chart(vector_strips(look.words, look.vectors, title=f"{model_name}, {look.dim} dimensions", mode=MODE), width="stretch")
        with st.expander("Raw numbers"):
            st.dataframe(pd.DataFrame(look.vectors, index=look.words).round(3), width="stretch")

# Similarity tab
with tabs[2]:
    lede("Cosine similarity compares direction: 1 means the same direction, 0 means unrelated.")
    words_in = st.text_input("Words", "king, queen, man, woman, water, apple", key="sim_words")
    look = get_vectors(kv, parse_words(words_in))
    warn_missing(look.missing)
    left, right = st.columns([3, 2], gap="large")
    if len(look.words) >= 2:
        sim = cosine_matrix(look.vectors)
        left.plotly_chart(similarity_heatmap(look.words, sim, mode=MODE), width="stretch")
        right.dataframe(pd.DataFrame(pairwise_scores(look.words, sim)), hide_index=True, width="stretch", height=380)
    with st.container(border=True):
        st.markdown("**Nearest neighbours**")
        nn_word = st.text_input("Word", "king", key="nn_word")
        try:
            nn = pd.DataFrame(nearest(kv, nn_word, 10), columns=["word", "cosine"])
            st.dataframe(nn, hide_index=True, width="stretch", column_config={"cosine": st.column_config.ProgressColumn("cosine", min_value=0, max_value=1, format="%.3f")})
        except OOVError as e:
            st.warning(str(e), icon=":material/help:")

# Map tab
with tabs[3]:
    lede(f"{kv.vector_size} dimensions squeezed into 2. Words with related meaning land close together.")
    groups_in = st.text_area("One group per line: `group: word, word, ...`", DEFAULT_MAP, height=100)
    words, groups = parse_groups(groups_in)
    look = get_vectors(kv, words)
    warn_missing(look.missing)
    if len(look.words) >= 3:
        group_of = dict(zip(words, groups))
        try:
            pts = cached_project_2d(look.vectors, projection)
            st.plotly_chart(
                scatter_2d(
                    look.words, pts, [group_of[w] for w in look.words],
                    title=f"{projection.upper()} of {model_name}", axis_prefix="PC" if projection == "pca" else "UMAP", mode=MODE,
                ),
                width="stretch",
            )
        except RuntimeError as e:
            st.error(str(e))
    else:
        st.info("Enter at least 3 known words.")

# Analogy tab
with tabs[4]:
    lede("Add and subtract word vectors, then look for the nearest real word.")
    c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
    example = c1.selectbox("Example", ANALOGY_EXAMPLES)
    topn = c2.slider("Top-k", 3, 20, 10)
    expr = st.text_input("Expression", example, key=f"analogy_{example}")
    pos, neg = parse_analogy(expr)
    try:
        res = analogy(kv, pos, neg, topn=topn)
        left, right = st.columns([2, 3], gap="large")
        left.markdown(
            f'<div class="vs-answer"><div class="expr">{html.escape(expr)} ≈</div>'
            f'<div class="word">{html.escape(res[0][0])}</div>'
            f'<div class="score">cosine {res[0][1]:.3f}</div></div>',
            unsafe_allow_html=True,
        )
        left.caption("Input words are excluded from the results, so the answer can't be one of them.")
        right.dataframe(
            pd.DataFrame(res, columns=["word", "cosine"]), hide_index=True, width="stretch",
            column_config={"cosine": st.column_config.ProgressColumn("cosine", min_value=0, max_value=1, format="%.3f")},
        )
        vec = analogy_vector(kv, pos, neg)
        operands = pos + neg
        unit = np.vstack([unit_vectors(kv, operands), l2_normalize(vec), unit_vectors(kv, [res[0][0]])])
        st.plotly_chart(
            vector_strips(operands + [f"= {expr}", f"top: {res[0][0]}"], unit, title="Unit vectors: operands, result, top answer", mode=MODE),
            width="stretch",
        )
    except (OOVError, ValueError) as e:
        st.warning(str(e), icon=":material/help:")

# CLIP tab
with tabs[5]:
    lede(
        "CLIP maps text and images into one 512-dimension space, so a sentence can find a picture. "
        "CLIP vectors are never mixed with GloVe."
    )
    if not st.toggle("Enable CLIP (downloads ~600 MB on first use)", value=False, key="clip_on"):
        st.stop()

    sample_paths = clipmod.list_sample_images()
    uploads = st.file_uploader("Add your own images (kept in memory, never saved)", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True) or []
    names = [p.name for p in sample_paths] + [f"upload:{u.name}" for u in uploads]
    blobs = read_samples(tuple((str(p), p.stat().st_mtime) for p in sample_paths)) + [u.getvalue() for u in uploads]
    blob_of = dict(zip(names, blobs))
    if not blobs:
        st.info("No images yet. Run `python scripts/download_samples.py` or upload some above.")
        st.stop()

    with st.spinner("Embedding images…"):
        img_vecs = clip_embed_images(tuple(blobs))

    st.markdown("#### Search images with text")
    query = st.text_input("Query", "a photo of a cat", key="clip_query")
    if query.strip():
        ranked = clipmod.rank_images(clip_embed_texts((query,))[0], img_vecs, names)
        for k, (col, (name, score)) in enumerate(zip(st.columns(min(6, len(ranked))), ranked)):
            with col.container(border=True):
                st.image(thumb(blob_of[name]), width="stretch")
                st.markdown(f'<div class="vs-rank"><b>#{k + 1}</b> {html.escape(name)}<br>cosine {score:.3f}</div>', unsafe_allow_html=True)
        st.caption("Raw CLIP text-to-image cosines are small (about 0.1 to 0.3). The ranking is what matters.")

    st.markdown("#### Text and image similarity")
    prompts_in = st.text_area("Prompts (one per line)", "a photo of a cat\na bee on a flower\na robot\na sweet pastry\na car on a highway", height=120)
    prompts = [p.strip() for p in prompts_in.splitlines() if p.strip()]
    if prompts:
        txt_vecs = clip_embed_texts(tuple(prompts))
        sim = clipmod.text_image_similarity(txt_vecs, img_vecs)
        st.plotly_chart(similarity_heatmap(prompts, sim, col_labels=names, mode=MODE, sequential=True), width="stretch")

        st.markdown("#### Image vector strip")
        pick = st.selectbox("Image", names)
        i = names.index(pick)
        best = int(np.argmax(sim[:, i]))
        c1, c2 = st.columns([1, 4], vertical_alignment="center")
        c1.image(blob_of[pick], width="stretch")
        c2.plotly_chart(
            vector_strips(
                [f"image: {pick}", f"text: {prompts[best]}"], np.vstack([img_vecs[i], txt_vecs[best]]),
                title="512-d CLIP vectors: the image and its best-matching prompt", height_per_row=60, mode=MODE,
            ),
            width="stretch",
        )

        st.markdown("#### Text and images on one map")
        close_gap = st.checkbox(
            "Close the modality gap (center text and images separately)",
            help="CLIP keeps all text vectors in one region and all image vectors in another. Centering each group shows which text matches which image.",
        )
        all_vecs = np.vstack([center_unit(txt_vecs), center_unit(img_vecs)] if close_gap else [txt_vecs, img_vecs])
        try:
            pts = cached_project_2d(all_vecs, projection)
            st.plotly_chart(
                scatter_2d(
                    prompts + names, pts, ["text"] * len(prompts) + ["image"] * len(names),
                    symbols={"text": "circle", "image": "diamond"}, title=f"{projection.upper()} of CLIP text and image vectors", mode=MODE,
                ),
                width="stretch",
            )
        except RuntimeError as e:
            st.error(str(e))
