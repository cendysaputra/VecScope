import numpy as np
import pytest

from src.embeddings import get_vectors, load_glove, parse_words
from src.similarity import OOVError, analogy, cosine, cosine_matrix, parse_analogy
from src.tokenize import DEFAULT_PAIR, tokenize
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


def test_oov_reported(kv):
    look = get_vectors(kv, ["king", "qwzxvbnm"])
    assert look.words == ["king"] and look.missing == ["qwzxvbnm"]
    with pytest.raises(OOVError):
        analogy(kv, ["king", "qwzxvbnm"], ["man"])


def test_cosine_king_queen_vs_water(kv):
    assert cosine(kv["king"], kv["queen"]) > 0.7
    assert cosine(kv["king"], kv["water"]) < 0.3
    m = cosine_matrix(get_vectors(kv, ["man", "woman", "king"]).vectors)
    assert m[0, 1] > m[0, 2]


def test_king_man_woman_queen(kv):
    top = [w for w, _ in analogy(kv, ["king", "woman"], ["man"], topn=3)]
    assert top[0] == "queen"


def test_pca_shape(kv):
    pts = project_2d(get_vectors(kv, ["king", "queen", "dog", "cat"]).vectors)
    assert pts.shape == (4, 2) and np.isfinite(pts).all()
