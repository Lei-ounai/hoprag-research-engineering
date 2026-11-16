"""SQLite cache for local Cross-Encoder scores."""

from dataclasses import dataclass
import hashlib
import json
from math import isfinite
import sqlite3

from .core import TEMPLATE


@dataclass(frozen=True)
class Score:
    score: float
    input_tokens: int
    original_tokens: int
    truncated: bool
    inference_seconds: float


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class RerankCache:
    def __init__(self, path, metadata):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.model = metadata["model_id"]
        self.revision = metadata["revision"]
        self.identity = json.dumps({
            "onnx_sha256": metadata["model_sha256"],
            "effective_max_tokens": metadata["effective_max_tokens"],
            "context_mode": metadata["context_mode"],
            "truncation": metadata.get("truncation_strategy", "backend-default"),
            "template": TEMPLATE,
        }, sort_keys=True)
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS scores (
              model TEXT NOT NULL, revision TEXT NOT NULL, config TEXT NOT NULL,
              query_hash TEXT NOT NULL, text_hash TEXT NOT NULL, query_id TEXT NOT NULL,
              document_id TEXT NOT NULL, chunk_id TEXT NOT NULL, score REAL NOT NULL,
              input_tokens INTEGER NOT NULL, original_tokens INTEGER NOT NULL,
              truncated INTEGER NOT NULL, inference_seconds REAL NOT NULL,
              PRIMARY KEY (model, revision, config, query_hash, text_hash)
            ) WITHOUT ROWID
        """)
        self.lookup = """
            SELECT score, input_tokens, original_tokens, truncated, inference_seconds, text_hash
            FROM scores WHERE model=? AND revision=? AND config=? AND query_hash=?
        """

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.db.close()

    def get(self, query_hash):
        rows = self.db.execute(self.lookup, (self.model, self.revision, self.identity, query_hash))
        return {
            text_hash: Score(value, used, original, bool(cut), elapsed)
            for value, used, original, cut, elapsed, text_hash in rows
        }

    def put_batch(self, query_id, query_hash, batch, scores):
        rows = []
        for (text_hash, pair), score in zip(batch, scores, strict=True):
            if (not isfinite(score.score) or not isfinite(score.inference_seconds)
                    or not 0 < score.input_tokens <= score.original_tokens
                    or score.inference_seconds < 0
                    or bool(score.truncated) != (score.input_tokens < score.original_tokens)):
                raise ValueError(f"Invalid score/length for {pair.chunk_id}")
            rows.append((
                self.model, self.revision, self.identity, query_hash, text_hash,
                query_id, pair.document_id, pair.chunk_id, score.score, score.input_tokens,
                score.original_tokens, int(score.truncated), score.inference_seconds,
            ))
        with self.db:
            self.db.executemany("INSERT OR IGNORE INTO scores VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)


def score_pool(backend, cache, pool, pairs, batch_size):
    hashes = [digest(pair.text) for pair in pairs]
    query_hash = digest(pool.question)
    saved = cache.get(query_hash)
    missing = {}
    for text_hash, pair in zip(hashes, pairs, strict=True):
        if text_hash not in saved:
            missing.setdefault(text_hash, pair)

    to_score = list(missing.items())
    model_seconds = 0.0
    for start in range(0, len(to_score), batch_size):
        batch = to_score[start:start + batch_size]
        scores = backend.score(pool.question, [pair.text for _, pair in batch])
        cache.put_batch(pool.query_id, query_hash, batch, scores)
        for (text_hash, _), score in zip(batch, scores, strict=True):
            saved[text_hash] = Score(
                score.score, score.input_tokens, score.original_tokens,
                bool(score.truncated), score.inference_seconds,
            )
            model_seconds += score.inference_seconds
    return [saved[text_hash] for text_hash in hashes], len(to_score), model_seconds
