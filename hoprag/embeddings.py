"""Local Ollama requests and a resumable SQLite embedding cache."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np


class EmbeddingError(RuntimeError):
    """The embedding endpoint returned an error or unusable vectors."""


@dataclass
class EmbeddingStats:
    cache_hits: int = 0
    cache_misses: int = 0
    api_requests: int = 0
    prompt_eval_count: int = 0
    api_seconds: float = 0.0
    elapsed_seconds: float = 0.0


class OllamaEmbedder:
    def __init__(
        self,
        cache_path: str | Path,
        *,
        base_url: str = "http://127.0.0.1:11434",
        model: str = "bge-m3",
        batch_size: int = 8,
        dimensions: int = 1024,
        timeout: float = 180.0,
        keep_alive: str = "10m",
        progress: Callable[[int, int], None] | None = None,
    ):
        if not model or batch_size < 1 or dimensions < 1 or timeout <= 0:
            raise ValueError("Model, batch size, dimensions and timeout must be valid.")
        self.model = model
        self.batch_size = batch_size
        self.dimensions = dimensions
        self.timeout = timeout
        self.keep_alive = keep_alive
        self._progress = progress
        self.stats = EmbeddingStats()
        self._url = f"{base_url.rstrip('/')}/api/embed"

        path = Path(cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path)
        try:
            self._db.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    model TEXT NOT NULL,
                    text_sha256 TEXT NOT NULL,
                    vector BLOB NOT NULL,
                    PRIMARY KEY (model, text_sha256)
                ) WITHOUT ROWID
                """
            )
        except sqlite3.Error:
            self._db.close()
            raise

    def __enter__(self) -> OllamaEmbedder:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        self._db.close()

    def _get_cached(self, text_hash: str) -> np.ndarray | None:
        row = self._db.execute(
            "SELECT vector FROM embeddings WHERE model = ? AND text_sha256 = ?",
            (self.model, text_hash),
        ).fetchone()
        if row is None or len(row[0]) != 4 * self.dimensions:
            return None
        vector = np.frombuffer(row[0], dtype="<f4")
        if not np.isfinite(vector).all() or not np.any(vector):
            return None
        return vector

    def _request_batch(self, texts: Sequence[str]) -> np.ndarray:
        request = Request(
            self._url,
            data=json.dumps(
                {
                    "model": self.model,
                    "input": list(texts),
                    "truncate": False,
                    "keep_alive": self.keep_alive,
                },
                ensure_ascii=False,
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.perf_counter()
        self.stats.api_requests += 1
        try:
            with urlopen(request, timeout=self.timeout) as response:
                payload = json.load(response)
        except HTTPError as exc:
            with exc:
                detail = exc.read(1024).decode("utf-8", errors="replace")
            raise EmbeddingError(f"Ollama HTTP {exc.code} for {self.model}: {detail}") from exc
        except (URLError, OSError, ValueError) as exc:
            raise EmbeddingError(f"Ollama embedding request failed for {self.model}: {exc}") from exc
        finally:
            self.stats.api_seconds += time.perf_counter() - started

        try:
            with np.errstate(over="ignore", invalid="ignore"):
                vectors = np.asarray(payload["embeddings"], dtype="<f4")
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise EmbeddingError("Ollama response did not contain a numeric embedding matrix.") from exc
        expected = (len(texts), self.dimensions)
        if vectors.shape != expected:
            raise EmbeddingError(f"Ollama embedding shape was {vectors.shape}, expected {expected}.")
        if not np.isfinite(vectors).all():
            raise EmbeddingError("Ollama returned a NaN or infinite embedding value.")
        if not np.any(vectors, axis=1).all():
            raise EmbeddingError("Ollama returned a zero vector with undefined cosine similarity.")
        token_count = payload.get("prompt_eval_count")
        if type(token_count) is int and token_count > 0:
            self.stats.prompt_eval_count += token_count
        return vectors

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return vectors in input order; only successfully validated batches enter the cache."""
        started = time.perf_counter()
        try:
            if len(texts) == 0:
                return np.empty((0, self.dimensions), dtype=np.float32)
            hashes = [hashlib.sha256(text.encode("utf-8")).hexdigest() for text in texts]
            unique_texts = dict(zip(hashes, texts, strict=True))
            missing: list[tuple[str, str]] = []
            by_hash: dict[str, np.ndarray] = {}
            for text_hash, text in unique_texts.items():
                vector = self._get_cached(text_hash)
                if vector is None:
                    missing.append((text_hash, text))
                else:
                    by_hash[text_hash] = vector

            self.stats.cache_hits += len(by_hash)
            self.stats.cache_misses += len(missing)
            if self._progress is not None:
                self._progress(len(by_hash), len(unique_texts))
            for start in range(0, len(missing), self.batch_size):
                batch = missing[start : start + self.batch_size]
                try:
                    vectors = self._request_batch([text for _, text in batch])
                except EmbeddingError as exc:
                    raise EmbeddingError(
                        f"Embedding batch starting at missing input {start + 1} failed: {exc}"
                    ) from exc
                values = {
                    text_hash: vector
                    for (text_hash, _), vector in zip(batch, vectors, strict=True)
                }
                with self._db:
                    self._db.executemany(
                        """
                        INSERT INTO embeddings(model, text_sha256, vector) VALUES (?, ?, ?)
                        ON CONFLICT(model, text_sha256) DO UPDATE SET vector = excluded.vector
                        """,
                        (
                            (self.model, text_hash, vector.tobytes())
                            for text_hash, vector in values.items()
                        ),
                    )
                by_hash.update(values)
                if self._progress is not None:
                    self._progress(len(by_hash), len(unique_texts))
            return np.stack([by_hash[text_hash] for text_hash in hashes])
        finally:
            self.stats.elapsed_seconds += time.perf_counter() - started
