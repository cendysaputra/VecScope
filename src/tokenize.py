"""Tokenization: GPT-2 BPE (tiktoken) vs BERT WordPiece (Hugging Face).

Import this module as ``src.tokenize`` (it is inside a package, so it does not
shadow Python's built-in ``tokenize`` module).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

# Display name -> (backend, identifier)
TOKENIZERS: dict[str, tuple[str, str]] = {
    "GPT-2 BPE (tiktoken gpt2)": ("tiktoken", "gpt2"),
    "BERT WordPiece (bert-base-uncased)": ("hf", "bert-base-uncased"),
    "GPT-4 BPE (tiktoken cl100k_base)": ("tiktoken", "cl100k_base"),
    "GPT-4o BPE (tiktoken o200k_base)": ("tiktoken", "o200k_base"),
}
DEFAULT_PAIR = ("GPT-2 BPE (tiktoken gpt2)", "BERT WordPiece (bert-base-uncased)")


@dataclass
class TokenizationResult:
    name: str
    tokens: list[str] = field(default_factory=list)
    ids: list[int] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.ids)

    def rows(self) -> list[dict]:
        return [{"#": i, "token": t, "id": tid} for i, (t, tid) in enumerate(zip(self.tokens, self.ids))]


@lru_cache(maxsize=None)
def _tiktoken_encoding(name: str):
    import tiktoken

    return tiktoken.get_encoding(name)


@lru_cache(maxsize=None)
def _hf_tokenizer(name: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name)


def _visible(piece: str) -> str:
    """Make leading spaces / newlines visible so token boundaries are obvious."""
    return piece.replace(" ", "␣").replace("\n", "↵").replace("\t", "⇥")


def tokenize_tiktoken(text: str, encoding: str = "gpt2") -> TokenizationResult:
    enc = _tiktoken_encoding(encoding)
    ids = enc.encode(text)
    # A single BPE token can be a partial UTF-8 sequence (e.g. emoji), so decode with replacement.
    tokens = [_visible(enc.decode_single_token_bytes(i).decode("utf-8", errors="replace")) for i in ids]
    return TokenizationResult(name=f"tiktoken:{encoding}", tokens=tokens, ids=ids)


def tokenize_bert(text: str, model: str = "bert-base-uncased", special_tokens: bool = False) -> TokenizationResult:
    tok = _hf_tokenizer(model)
    ids = tok.encode(text, add_special_tokens=special_tokens)
    tokens = tok.convert_ids_to_tokens(ids)
    return TokenizationResult(name=f"wordpiece:{model}", tokens=tokens, ids=ids)


def tokenize(text: str, tokenizer: str, special_tokens: bool = False) -> TokenizationResult:
    """Tokenize ``text`` with a tokenizer from :data:`TOKENIZERS` (by display name)."""
    backend, ident = TOKENIZERS[tokenizer]
    if backend == "tiktoken":
        result = tokenize_tiktoken(text, ident)
    else:
        result = tokenize_bert(text, ident, special_tokens=special_tokens)
    result.name = tokenizer
    return result
