"""Text embedders. All return L2-normalized float32 arrays so dot product = cosine similarity."""
from __future__ import annotations

import re
import zlib

import numpy as np

STOPWORDS = frozenset(
    "a an and are as at be by can for from how i in into is it me my need of on or our that the this "
    "to us use using want we what with you your".split()
)
_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str, drop_stopwords: bool = True) -> list[str]:
    """Lowercase alphanumeric tokens with a very light plural stemmer."""
    out = []
    for tok in _TOKEN.findall((text or "").lower()):
        if drop_stopwords and tok in STOPWORDS:
            continue
        if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
            tok = tok[:-1]
        out.append(tok)
    return out


class EmbedderUnavailable(RuntimeError):
    pass


class HashingEmbedder:
    """Offline bag-of-words (+bigrams) embedder using the hashing trick.

    No downloads and fully deterministic. It only captures word overlap, not meaning, so use it
    for tests and as a fallback, and use a sentence-transformers model for real quality.
    """

    def __init__(self, dim: int = 512):
        self.dim = dim
        self.name = f"hashing-{dim}"

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            toks = tokenize(text)
            for feat in toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]:
                h = zlib.crc32(feat.encode("utf-8"))
                out[i, h % self.dim] += 1.0 if (h >> 31) & 1 else -1.0
            norm = np.linalg.norm(out[i])
            if norm:
                out[i] /= norm
        return out


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, batch_size: int = 64):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbedderUnavailable(
                "sentence-transformers is not installed. Run: pip install -e \".[embed]\"  "
                "(or use --embedder hashing for the offline baseline)."
            ) from exc
        self.model = SentenceTransformer(model_name)
        self.name = model_name
        self.batch_size = batch_size
        self.dim = int(self.model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str]) -> np.ndarray:
        arr = self.model.encode(
            texts, batch_size=self.batch_size, normalize_embeddings=True,
            show_progress_bar=False, convert_to_numpy=True,
        )
        return np.asarray(arr, dtype=np.float32)


def get_embedder(name: str):
    if name == "hashing" or name.startswith("hashing-"):
        return HashingEmbedder(int(name.split("-")[1])) if "-" in name else HashingEmbedder()
    return SentenceTransformerEmbedder(name)
