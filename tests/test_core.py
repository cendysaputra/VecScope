import numpy as np
import pytest

from src.embeddings import get_vectors, load_glove, parse_groups, parse_words
from src.similarity import (
    OOVError,
    analogy,
    analogy_vector,
    center_unit,
    cosine,
    cosine_matrix,
    nearest,
    pairwise_scores,
    parse_analogy,
    unit_vectors,
)
from src.tokenize import DEFAULT_PAIR, tokenize, tokenize_tiktoken
from src.visualize import project_2d

SENTENCE = "Tokenization isn't magic: unbelievably, GPT-2 and BERT split words differently!"


@pytest.fixture(scope="module")
def kv():
    return load_glove()


def test_gpt2_vs_bert_differ():
    gpt2, bert = (tokenize(SENTENCE, n) for n in DEFAULT_PAIR)
    assert gpt2.count == len(gpt2.tokens) > 0
    assert gpt2.count != bert.count
    assert "##ization" in bert.tokens
    assert "ization" in gpt2.tokens


def test_parse_helpers():
    assert parse_words("King, MAN  woman,king") == ["king", "man", "woman"]
    assert parse_analogy("King - man + Woman") == (["king", "woman"], ["man"])
    assert parse_groups("royalty: King, Queen\npeople: king, man") == (["king", "queen", "man"], ["royalty", "royalty", "people"])


def test_oov_reported(kv):
    look = get_vectors(kv, ["king", "qwzxvbnm"])
    assert look.words == ["king"] and look.missing == ["qwzxvbnm"]
    with pytest.raises(OOVError):
        analogy(kv, ["king", "qwzxvbnm"], ["man"])


def test_cosine_king_queen_vs_water(kv):
    assert cosine(kv["king"], kv["queen"]) > 0.7
    assert cosine(kv["king"], kv["water"]) < 0.3
    words = ["man", "woman", "king"]
    m = cosine_matrix(get_vectors(kv, words).vectors)
    assert m[0, 1] > m[0, 2]
    pairs = pairwise_scores(words, m)
    assert len(pairs) == 3 and pairs[0]["word A"] == "man" and pairs[0]["word B"] == "woman"


def test_king_man_woman_queen(kv):
    top = [w for w, _ in analogy(kv, ["king", "woman"], ["man"], topn=3)]
    assert top[0] == "queen"
    vec = analogy_vector(kv, ["king", "woman"], ["man"])
    u = unit_vectors(kv, ["king", "woman", "man"])
    assert np.allclose(vec, u[0] + u[1] - u[2])


def test_pca_and_center_unit(kv):
    vecs = get_vectors(kv, ["king", "queen", "dog", "cat"]).vectors
    pts = project_2d(vecs)
    assert pts.shape == (4, 2) and np.isfinite(pts).all()
    cu = center_unit(vecs)
    assert np.allclose(np.linalg.norm(cu, axis=1), 1.0, atol=1e-5)


def test_eval_results():
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent / "eval" / "results"
    analogy_res = json.loads((root / "analogy.json").read_text())["models"]
    assert analogy_res["glove-wiki-gigaword-50"]["full_vocab_400k"]["total_accuracy"] < analogy_res["glove-wiki-gigaword-100"]["full_vocab_400k"]["total_accuracy"]
    g300 = analogy_res["glove-wiki-gigaword-300"]["full_vocab_400k"]
    assert round(g300["semantic_accuracy"], 1) == 77.4
    assert round(g300["syntactic_accuracy"], 1) == 67.0
    assert round(g300["total_accuracy"], 1) == 71.7

    sim_res = json.loads((root / "word_similarity.json").read_text())["models"]
    assert sim_res["glove-wiki-gigaword-300"]["wordsim353"]["spearman_paper_norm_100"] == 65.8

    clip_res = json.loads((root / "clip_cifar10.json").read_text())["results"]
    assert clip_res["num_images"] == 10000
    assert round(clip_res["ensemble_accuracy"], 1) == 89.8



def test_gpt2_treats_special_marker_as_literal_text():
    marker = "<|endoftext|>"
    result = tokenize_tiktoken(marker)
    assert result.count > 1
    assert "".join(result.tokens) == marker


def test_nearest_requires_a_word():
    with pytest.raises(ValueError, match="Enter a word"):
        nearest(None, "  ")


def test_unit_vectors_only_normalize_requested_words(monkeypatch):
    from gensim.models import KeyedVectors

    vectors = KeyedVectors(vector_size=2)
    vectors.add_vectors(["first", "second"], np.array([[3., 4.], [0., 2.]]))

    def reject_full_normalization():
        pytest.fail("Requested normalized vectors for the entire vocabulary")

    monkeypatch.setattr(vectors, "get_normed_vectors", reject_full_normalization)
    np.testing.assert_allclose(unit_vectors(vectors, [" SECOND ", "first", "second"]),
                               [[0., 1.], [0.6, 0.8], [0., 1.]])


def test_umap_handles_small_projections():
    with pytest.raises(RuntimeError, match="at least three"):
        project_2d(np.eye(2), "umap")
    pytest.importorskip("umap")
    points = project_2d(np.eye(3), "umap")
    assert points.shape == (3, 2) and np.isfinite(points).all()
