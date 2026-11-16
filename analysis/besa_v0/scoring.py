"""Budgeted Need–Chunk selection and exact local score reuse."""

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from math import exp, isfinite, log1p
from pathlib import Path
import sqlite3

import numpy as np

from analysis.rankings import load_cached_vectors
from analysis.reranking.cache import RerankCache, Score, digest
from analysis.reranking.core import make_pairs
from hoprag.retrievers import _normalize, _tokens


BUDGETS = (30, 60, 90)
ROUTES = ("need_lexical", "question_dense", "p5b_prior")
STOP_WORDS = frozenset("""
a an and are as at be been between both by did do does for from had has have how in
into is it of on or that the their these this those to was were what when where
which who whom whose will with would
""".split())


@dataclass(frozen=True)
class NeedPair:
    need_id: str
    need_text: str
    document_id: str
    chunk_id: str
    text: str
    route: str
    dense_score: float
    lexical_score: float


@dataclass(frozen=True)
class ScoredPair:
    need_id: str
    document_id: str
    chunk_id: str
    score: float
    cache_source: str
    inference_seconds: float
    input_tokens: int
    original_tokens: int
    truncated: bool
    route: str
    dense_score: float
    lexical_score: float


def _term_set(text):
    tokens = set(_tokens(text))
    return (tokens - STOP_WORDS) or tokens


class DenseVectors:
    """Cached original-question/Chunk vectors and a lexical corpus index."""

    def __init__(self, vectors, chunks, audit=None):
        self._original_vectors = vectors
        self.audit = audit
        self.chunk_vectors = {}
        self._chunk_terms = {}
        document_frequency = Counter()
        parents = {}
        for document_id, rows in sorted(chunks.items()):
            if not rows:
                raise ValueError(f"Missing chunks for {document_id}")
            ids = set()
            frequencies = set()
            parent_matrix = _normalize(np.stack([vectors[digest(row.text)] for row in rows]))
            for row, vector in zip(rows, parent_matrix, strict=True):
                if row.document_id != document_id or row.chunk_id in self.chunk_vectors:
                    raise ValueError(f"Invalid cached BGE chunk mapping: {row.chunk_id}")
                terms = _term_set(row.text)
                self._chunk_terms[row.chunk_id] = terms
                self.chunk_vectors[row.chunk_id] = vector
                ids.add(row.chunk_id)
                frequencies.update(terms)
            document_frequency.update(frequencies)
            parents[document_id] = (ids, parent_matrix.mean(axis=0))
        if not parents:
            raise ValueError("No cached BGE chunk vectors supplied")
        centroid_matrix = _normalize(np.stack([values[1] for values in parents.values()]))
        self._parents = {
            document_id: (values[0], centroid)
            for (document_id, values), centroid in zip(parents.items(), centroid_matrix, strict=True)
        }
        self._idf = {
            term: log1p((len(parents) - count + 0.5) / (count + 0.5))
            for term, count in document_frequency.items()
        }

    @classmethod
    def load(cls, cache_path, chunks, model="bge-m3", dimensions=1024):
        vectors, audit = load_cached_vectors(Path(cache_path), model, dimensions, chunks, ())
        if audit["read_error"] or audit["missing_corpus_vectors"] or audit["invalid_expected_vectors"]:
            raise ValueError(f"Incomplete historical BGE-M3 cache: {audit}")
        return cls(vectors, chunks, audit=audit)

    def cosines(self, question, chunks):
        try:
            query = self._original_vectors[digest(question)]
            values = np.stack([self.chunk_vectors[row.chunk_id] for row in chunks])
        except KeyError as exc:
            raise ValueError(f"Missing cached BGE-M3 input: {exc}") from exc
        normalized = _normalize(query[None, :])[0]
        scores = np.einsum("ij,j->i", values, normalized)
        return {row.chunk_id: float(score) for row, score in zip(chunks, scores, strict=True)}

    def lexical(self, need_text, chunk_ids):
        terms = _term_set(need_text)
        weighted_terms = [(term, self._idf.get(term, 0.0)) for term in sorted(terms)]
        total = sum(weight for _, weight in weighted_terms)
        return {
            chunk_id: (
                sum(weight for term, weight in weighted_terms if term in self._chunk_terms[chunk_id]) / total
                if total else 0.0
            )
            for chunk_id in chunk_ids
        }

    def parent_vectors(self, pool):
        vectors = {}
        for candidate in pool.candidates:
            ids, centroid = self._parents[candidate.document_id]
            if ids != {chunk.chunk_id for chunk in candidate.chunks}:
                raise ValueError(f"Parent chunk set changed for {candidate.document_id}")
            vectors[candidate.document_id] = centroid
        return vectors


def schedule(pool, needs, prior_ranks, vectors, limit=90):
    """Alternate needs and routes. Each need covers new parents before revisiting any."""
    if limit not in BUDGETS:
        raise ValueError(f"Unsupported Need–Chunk budget: {limit}")
    needs = tuple(needs)
    if not 1 <= len(needs) <= 4 or len({n.need_id for n in needs}) != len(needs):
        raise ValueError("Expected one to four unique scorable needs")
    if any(not n.need_id.strip() or not n.need_text.strip() for n in needs):
        raise ValueError("Each need needs a non-empty ID and text")
    if len({n.need_text for n in needs}) != len(needs):
        raise ValueError("Expected distinct need texts")

    parents = {candidate.document_id for candidate in pool.candidates}
    if len(parents) != len(pool.candidates) or parents != set(prior_ranks):
        raise ValueError("P5-B prior must correspond exactly to unique candidate parents")
    if any(type(rank) is not int or rank < 1 for rank in prior_ranks.values()):
        raise ValueError("P5-B prior ranks must be positive integers")
    chunks = [chunk for candidate in pool.candidates for chunk in candidate.chunks]
    pairs = make_pairs(pool)
    if not pairs or len({pair.chunk_id for pair in pairs}) != len(pairs):
        raise ValueError("Expected unique candidate chunks")
    if {pair.document_id for pair in pairs} != parents:
        raise ValueError("Each P5 parent must have at least one chunk")

    dense = vectors.cosines(pool.question, chunks)
    lexical = {need.need_id: vectors.lexical(need.need_text, dense) for need in needs}
    rankings = {}
    for need in needs:
        scores = lexical[need.need_id]

        def lex_key(pair):
            return (-scores[pair.chunk_id], -dense[pair.chunk_id], prior_ranks[pair.document_id], pair.chunk_id)

        def dense_key(pair):
            return (-dense[pair.chunk_id], -scores[pair.chunk_id], prior_ranks[pair.document_id], pair.chunk_id)

        def prior_key(pair):
            return (prior_ranks[pair.document_id], -scores[pair.chunk_id], -dense[pair.chunk_id], pair.chunk_id)

        rankings[need.need_id] = dict(zip(
            ROUTES, [sorted(pairs, key=key) for key in (lex_key, dense_key, prior_key)], strict=True,
        ))

    ordered = []
    visited = defaultdict(set)
    parent_uses = defaultdict(Counter)
    for slot in range(min(limit, len(needs) * len(pairs))):
        need = needs[slot % len(needs)]
        route = ROUTES[(slot // len(needs)) % len(ROUTES)]
        parent_counts = parent_uses[need.need_id]
        used = visited[need.need_id]
        # min() retains the first route-ordered item on ties.
        best = min(
            (pair for pair in rankings[need.need_id][route] if pair.chunk_id not in used),
            key=lambda pair: parent_counts[pair.document_id],
        )
        used.add(best.chunk_id)
        parent_counts[best.document_id] += 1
        ordered.append(NeedPair(
            need.need_id, need.need_text, best.document_id, best.chunk_id,
            best.text, route, dense[best.chunk_id], lexical[need.need_id][best.chunk_id],
        ))
    return tuple(ordered)


class NeedScoreCache(RerankCache):
    """Match the prior SQLite key exactly and store only genuinely new pairs."""

    def __init__(self, new_path, historical_paths, metadata):
        self._history = {}
        target = Path(new_path).resolve() if new_path is not None else Path(":memory:")
        try:
            for label, path in historical_paths.items():
                if label in {"local_cache", "new_inference", "in_run_reuse"}:
                    raise ValueError(f"Reserved cache source label: {label}")
                history = Path(path).resolve(strict=True)
                if new_path is not None and (target == history or (target.exists() and target.samefile(history))):
                    raise ValueError("The writable Need cache must differ from historical caches")
                wal = Path(f"{history}-wal")
                if wal.exists() and wal.stat().st_size:
                    raise ValueError(f"Uncheckpointed WAL for historical cache: {label}")
                db = sqlite3.connect(history.as_uri() + "?mode=ro&immutable=1", uri=True)
                self._history[label] = db
                db.execute("PRAGMA query_only=ON")
                db.execute("SELECT model, revision, config FROM scores LIMIT 0")
            super().__init__(target, metadata)
        except BaseException:
            self._close_history()
            if hasattr(self, "db"):
                self.db.close()
            raise

    def _close_history(self):
        for db in self._history.values():
            db.close()

    def __exit__(self, *exc):
        try:
            super().__exit__(*exc)
        finally:
            self._close_history()

    def lookup_pairs(self, query_hash, text_hashes):
        wanted = set(text_hashes)
        matches = {
            text_hash: (score, "local_cache")
            for text_hash, score in super().get(query_hash).items() if text_hash in wanted
        }
        for label, db in self._history.items():
            if len(matches) == len(wanted):
                break
            rows = db.execute(self.lookup, (self.model, self.revision, self.identity, query_hash))
            for text_hash, score in _score_rows(rows).items():
                if text_hash in wanted:
                    matches.setdefault(text_hash, (score, label))
        return matches


def _score_rows(rows):
    return {
        key: Score(score, used, original, bool(cut), elapsed)
        for score, used, original, cut, elapsed, key in rows
    }


def score_schedule(backend, cache, query_id, pairs, batch_size=16):
    """Score one max-depth schedule, restoring global order across per-Need batches."""
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("The scoring batch size must be positive")
    pairs = tuple(pairs)
    if len(pairs) > max(BUDGETS):
        raise ValueError("Need scoring schedule exceeds the hard per-query budget")

    groups = defaultdict(list)
    seen = set()
    for index, pair in enumerate(pairs):
        key = (pair.need_id, pair.chunk_id)
        if key in seen:
            raise ValueError(f"Duplicate Need–Chunk pair: {key}")
        seen.add(key)
        groups[pair.need_id].append((index, digest(pair.text), pair))
    result = [None] * len(pairs)

    for need_id, group in groups.items():
        need_text = group[0][2].need_text
        if any(pair.need_text != need_text for _, _, pair in group):
            raise ValueError(f"Conflicting texts for {need_id}")
        query_hash = digest(need_text)
        known = cache.lookup_pairs(query_hash, (text_hash for _, text_hash, _ in group))
        fresh = set()
        for start in range(0, len(group), batch_size):
            batch = group[start:start + batch_size]
            pending = {}
            for _, text_hash, pair in batch:
                if text_hash not in known:
                    pending.setdefault(text_hash, pair)
            if pending:
                missing = list(pending.items())
                outputs = list(backend.score(need_text, [pair.text for _, pair in missing]))
                cache.put_batch(query_id, query_hash, missing, outputs)
                for (text_hash, _), score in zip(missing, outputs, strict=True):
                    known[text_hash] = (score, "new_inference")
                    fresh.add(text_hash)
            for index, text_hash, pair in batch:
                score, source = known[text_hash]
                if source == "new_inference":
                    if text_hash in fresh:
                        fresh.remove(text_hash)
                    else:
                        source = "in_run_reuse"
                if not isfinite(score.score):
                    raise ValueError(f"Non-finite Need relevance for {pair.chunk_id}")
                result[index] = ScoredPair(
                    pair.need_id, pair.document_id, pair.chunk_id, score.score, source,
                    score.inference_seconds if source == "new_inference" else 0.0,
                    score.input_tokens, score.original_tokens, score.truncated,
                    pair.route, pair.dense_score, pair.lexical_score,
                )
    return tuple(result)


def summarize_scores(scored_pairs, limit, threshold=0.0):
    """Aggregate a prefix. Unscored parents are absent rather than scored as negatives."""
    if limit not in BUDGETS or not isfinite(threshold):
        raise ValueError("Expected B30/B60/B90 and a finite raw relevance threshold")
    pairs = tuple(scored_pairs[:limit])
    best = defaultdict(dict)
    for pair in pairs:
        previous = best[pair.need_id].get(pair.document_id)
        if previous is None or (-pair.score, pair.chunk_id) < (-previous.score, previous.chunk_id):
            best[pair.need_id][pair.document_id] = pair

    supports = {}
    for need_id, by_parent in best.items():
        rows = sorted(by_parent.values(), key=lambda pair: (-pair.score, pair.document_id))
        n = len(rows)
        need_support = {}
        previous_score = None
        rank = 1
        for count, pair in enumerate(rows, 1):
            if pair.score != previous_score:
                rank = count
                previous_score = pair.score
            percentile = (n - rank + 1) / n
            # Absolute gate and strength cap prevent an all-weak need from promoting its best hit.
            strength = 1 - exp(-max(pair.score - threshold, 0.0))
            need_support[pair.document_id] = {
                "support": percentile * strength,
                "raw_score": pair.score,
                "chunk_id": pair.chunk_id,
            }
        supports[need_id] = need_support

    sources = Counter(pair.cache_source for pair in pairs)
    new = sources["new_inference"]
    local = sources["local_cache"]
    reused = sources["in_run_reuse"]
    return {
        "supports": supports,
        "budget": limit,
        "actual_pairs": len(pairs),
        "new_pairs": new,
        "cache_hits": len(pairs) - new,
        "historical_cache_hits": len(pairs) - new - local - reused,
        "local_cache_hits": local,
        "in_run_reuse": reused,
        "scored_parents": len({pair.document_id for pair in pairs}),
        "scored_needs": len(best),
        "inference_seconds": sum(pair.inference_seconds for pair in pairs),
        "truncated_pairs": sum(pair.truncated for pair in pairs),
        "cache_sources": dict(sorted(sources.items())),
        "pairs": [asdict(pair) for pair in pairs],
    }
