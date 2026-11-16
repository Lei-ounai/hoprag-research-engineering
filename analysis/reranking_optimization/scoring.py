"""Read fifth-round scores unchanged and checkpoint sixth-round work locally."""

from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3

from analysis.reranking.cache import RerankCache, Score, digest
from analysis.reranking.core import make_pairs


@dataclass(frozen=True)
class PairCounts:
    total_pairs: int
    unique_pairs: int
    historical_hits: int
    new_hits: int
    duplicate_pairs: int
    missing_pairs: int


class LayeredCache(RerankCache):
    """The historical database is immutable. Only the new database receives writes."""

    def __init__(self, new_path, historical_path, metadata):
        history = Path(historical_path).resolve(strict=True)
        target = Path(new_path).resolve() if new_path is not None else Path(":memory:")
        if new_path is not None and (target == history or (target.exists() and target.samefile(history))):
            raise ValueError("The sixth-round database must differ from the historical cache")
        wal = Path(f"{history}-wal")
        if wal.exists() and wal.stat().st_size:
            raise ValueError("The historical cache contains an uncheckpointed SQLite WAL")

        self._history = sqlite3.connect(history.as_uri() + "?mode=ro&immutable=1", uri=True)
        try:
            self._history.execute("PRAGMA query_only=ON")
            self._history.execute("SELECT model, revision, config FROM scores LIMIT 0")
            super().__init__(target, metadata)
            self.db.execute("""
                CREATE TABLE IF NOT EXISTS query_checkpoints (
                    stage_id TEXT NOT NULL,
                    query_id TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    PRIMARY KEY (stage_id, query_id)
                ) WITHOUT ROWID
            """)
        except BaseException:
            self._history.close()
            if hasattr(self, "db"):
                self.db.close()
            raise

        self._last_history_key = None
        self._last_history_scores = {}

    def __exit__(self, *exc):
        try:
            super().__exit__(*exc)
        finally:
            self._history.close()

    def _historical_scores(self, query_hash):
        if query_hash != self._last_history_key:
            rows = self._history.execute(
                self.lookup, (self.model, self.revision, self.identity, query_hash),
            )
            self._last_history_scores = {
                text_hash: Score(score, used, original, bool(truncated), elapsed)
                for score, used, original, truncated, elapsed, text_hash in rows
            }
            self._last_history_key = query_hash
        return self._last_history_scores

    def get(self, query_hash):
        return super().get(query_hash) | self._historical_scores(query_hash)

    def inspect(self, pool, pairs=None):
        """Count exact chunk-text cache keys without exposing evidence or invoking a model."""
        if pairs is None:
            pairs = make_pairs(pool)
        hashes = [digest(pair.text) for pair in pairs]
        unique = set(hashes)
        query_hash = digest(pool.question)
        historical = unique & self._historical_scores(query_hash).keys()
        local = (unique & super().get(query_hash).keys()) - historical
        return PairCounts(
            total_pairs=len(hashes),
            unique_pairs=len(unique),
            historical_hits=len(historical),
            new_hits=len(local),
            duplicate_pairs=len(hashes) - len(unique),
            missing_pairs=len(unique) - len(historical) - len(local),
        )

    def put_batch(self, query_id, query_hash, batch, scores):
        historical = self._historical_scores(query_hash)
        local_batch = []
        local_scores = []
        for item, score in zip(batch, scores, strict=True):
            if item[0] not in historical:
                local_batch.append(item)
                local_scores.append(score)
        super().put_batch(query_id, query_hash, local_batch, local_scores)

    def checkpoint_get(self, stage_id, query_id):
        row = self.db.execute(
            "SELECT payload FROM query_checkpoints WHERE stage_id=? AND query_id=?",
            (stage_id, query_id),
        ).fetchone()
        return json.loads(row[0]) if row is not None else None

    def checkpoint_put(self, stage_id, query_id, payload):
        """Publish a completed query once under the supplied immutable stage identity."""
        serialized = json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True)
        with self.db:
            self.db.execute(
                "INSERT OR IGNORE INTO query_checkpoints VALUES (?, ?, ?)",
                (stage_id, query_id, serialized),
            )
            saved = self.db.execute(
                "SELECT payload FROM query_checkpoints WHERE stage_id=? AND query_id=?",
                (stage_id, query_id),
            ).fetchone()[0]
            if saved != serialized:
                raise ValueError(f"Checkpoint already has a different result: {stage_id}/{query_id}")

    def checkpoint_items(self, stage_id):
        rows = self.db.execute(
            "SELECT query_id, payload FROM query_checkpoints WHERE stage_id=? ORDER BY query_id",
            (stage_id,),
        )
        for query_id, payload in rows:
            yield query_id, json.loads(payload)
