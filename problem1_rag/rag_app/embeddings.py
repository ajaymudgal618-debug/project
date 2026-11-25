"""Embedding providers: OpenAI, Gemini, and deterministic local feature-hashing."""

from __future__ import annotations

import hashlib
import json
import math
import re
import urllib.request
from typing import List

import numpy as np

from .config import RagConfig

_TOKEN_RE = re.compile(r"[a-z0-9_]+")
_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "in", "on", "at", "to", "for", "of", "and", "or", "with", "by", "as",
    "it", "this", "that", "from", "which", "what", "when", "how", "why",
}


class LocalHashEmbedder:
    """Deterministic 384-d L2-normalized sparse-to-dense hash embedder.

    Uses signed feature hashing over unigrams, bigrams, and 4-char subwords
    so semantic/lexical overlap works offline with zero external model downloads.
    """

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim
        self.model_name = f"local-hash-ngram-v1 ({dim}d)"

    def _embed_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]
        if not tokens:
            tokens = _TOKEN_RE.findall(text.lower())

        features: list[tuple[str, float]] = []
        for i, tok in enumerate(tokens):
            features.append((f"u:{tok}", 1.5))
            if i + 1 < len(tokens):
                features.append((f"b:{tok}_{tokens[i+1]}", 1.2))
            if len(tok) >= 4:
                for j in range(len(tok) - 3):
                    features.append((f"c:{tok[j:j+4]}", 0.35))

        for feat, weight in features:
            digest = hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest()
            h = int.from_bytes(digest, "little")
            idx = h % self.dim
            sign = 1.0 if ((h >> 63) & 1) == 0 else -1.0
            vec[idx] += sign * weight

        norm = float(np.linalg.norm(vec))
        if norm > 0.0:
            vec /= norm
        return vec

    def embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack([self._embed_one(t) for t in texts]).astype(np.float32)


class OpenAIEmbedder:
    def __init__(self, api_key: str, model: str = "text-embedding-3-small", dim: int = 512) -> None:
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required when EMBEDDING_PROVIDER=openai")
        self.api_key = api_key
        self.model = model
        self.dim = dim
        self.model_name = f"{model} ({dim}d)"

    def embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        payload = {"input": texts, "model": self.model}
        if "text-embedding-3" in self.model:
            payload["dimensions"] = self.dim
        req = urllib.request.Request(
            "https://api.openai.com/v1/embeddings",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        arr = np.array([item["embedding"] for item in data["data"]], dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms


class GeminiEmbedder:
    def __init__(self, api_key: str, model: str = "text-embedding-004", dim: int = 768) -> None:
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required when EMBEDDING_PROVIDER=gemini")
        self.api_key = api_key
        self.model = model
        self.dim = dim
        self.model_name = f"{model} ({dim}d)"

    def embed(self, texts: List[str]) -> np.ndarray:
        vectors = []
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:embedContent?key={self.api_key}"
        for text in texts:
            payload = {"content": {"parts": [{"text": text}]}}
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            v = np.array(data["embedding"]["values"], dtype=np.float32)
            norm = float(np.linalg.norm(v))
            vectors.append(v / norm if norm > 0 else v)
        self.dim = int(vectors[0].shape[0]) if vectors else self.dim
        return np.vstack(vectors) if vectors else np.zeros((0, self.dim), dtype=np.float32)


def get_embedder(cfg: RagConfig):
    provider = cfg.embedding_provider.lower()
    if provider == "openai":
        return OpenAIEmbedder(cfg.openai_api_key, cfg.openai_embed_model, cfg.embedding_dim)
    if provider == "gemini":
        return GeminiEmbedder(cfg.gemini_api_key, cfg.gemini_embed_model, cfg.embedding_dim)
    return LocalHashEmbedder(dim=cfg.embedding_dim)
