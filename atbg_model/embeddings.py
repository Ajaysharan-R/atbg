"""
BGE-M3 turn embeddings, via sentence-transformers. Wrapped so the rest of
the codebase never touches sentence-transformers directly — if you ever
swap the embedding model, this is the one file that changes.
"""

import torch

_MODEL_NAME = "BAAI/bge-m3"
_model = None  # lazy-loaded, first call only — avoids paying the load cost
                # every time this module is imported (e.g. in the API process
                # at startup even for requests that don't need it yet)


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(_MODEL_NAME, device="cpu")
    return _model


def embed_turns(texts: list[str]) -> torch.Tensor:
    """texts: list of turn strings (already redacted). Returns (len(texts), embed_dim)."""
    if not texts:
        return torch.zeros(0, 1024)
    model = _get_model()
    embeddings = model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
    return embeddings


EMBED_DIM = 1024  # BGE-M3's dense embedding size
