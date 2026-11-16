"""Select and score the same capped number of original Chunks per parent."""

from collections.abc import Mapping
from math import isfinite
from pathlib import Path
from time import monotonic, perf_counter
from types import SimpleNamespace
import sqlite3

import numpy as np

from analysis.besa_v0.scoring import NeedScoreCache, STOP_WORDS
from analysis.ege_v0.ranking import local_jina_backend
from analysis.reranking.cache import digest
from analysis.reranking.core import make_pairs
from hoprag.retrievers import _tokens


CHUNK_BUDGET = 3
MAX_PARENTS = 80
BATCH_SIZE = 32


def _phrase_match(text, phrases):
    if not phrases:
        return 0.0
    return max((1.0 for phrase in phrases if phrase in text), default=0.0)


def _redundancy(left, right, chunk_vectors, token_sets):
    left_terms, right_terms = token_sets[left], token_sets[right]
    union = left_terms | right_terms
    overlap = len(left_terms & right_terms) / len(union) if union else 0.0
    cosine = float(np.dot(chunk_vectors[left], chunk_vectors[right]))
    return max(overlap, max(0.0, min(1.0, (cosine - 0.65) / 0.35)))


def select_chunks(question, candidates, chunks, vectors, expansion_terms=None, *, limit=CHUNK_BUDGET,
                  token_sets=None, token_texts=None, title_terms=None):
    """Return a deterministic dense/lexical/diversity selection for every parent."""
    by_parent = candidates if isinstance(candidates, Mapping) else {row.document_id: row for row in candidates}
    if type(limit) is not int or not 1 <= limit <= CHUNK_BUDGET:
        raise ValueError("Maximum Chunk budget must be an integer from 1 through 3")
    blocks = [row for doc in by_parent for row in chunks[doc]]
    if not blocks:
        if by_parent:
            raise ValueError("Candidate parent has no original Chunk")
        return {}
    if len({row.chunk_id for row in blocks}) != len(blocks):
        raise ValueError("Original Chunk IDs must be unique")
    dense = vectors.cosines(question, blocks)
    lexical = vectors.lexical(question, dense)
    question_terms = set(_tokens(question)) - STOP_WORDS
    token_sets = token_sets if token_sets is not None else {
        row.chunk_id: frozenset(_tokens(row.text)) - STOP_WORDS for row in blocks
    }
    token_texts = token_texts if token_texts is not None else {
        row.chunk_id: " " + " ".join(_tokens(row.text)) + " " for row in blocks
    }
    supplied = expansion_terms or {}
    out = {}
    for doc, candidate in by_parent.items():
        if doc != candidate.document_id or not chunks[doc] or any(row.document_id != doc for row in chunks[doc]):
            raise ValueError(f"Unresolved parent for Chunk selection: {doc}")
        title_words = title_terms[doc] if title_terms is not None else frozenset(_tokens(candidate.title))
        in_title = title_words & question_terms
        term_values = supplied.get(doc, ())
        if isinstance(term_values, str):
            term_values = (term_values,)
        phrases = tuple(" " + " ".join(tokens) + " " for phrase in term_values if (tokens := _tokens(phrase)))
        rows = []
        for chunk in chunks[doc]:
            chunk_id = chunk.chunk_id
            dense_score = dense[chunk_id]
            lexical_score = lexical[chunk_id]
            expansion_match = _phrase_match(token_texts[chunk_id], phrases)
            # Score the share of question-linked title terms that this block still mentions.
            title_match = len(in_title & token_sets[chunk_id]) / len(in_title) if in_title else 0.0
            dense_component = (1.0 + max(-1.0, min(1.0, dense_score))) * 0.5
            lexical_component = 0.70 * lexical_score + 0.20 * expansion_match + 0.10 * title_match
            combined = 0.45 * dense_component + 0.55 * lexical_component
            rows.append({
                "chunk_id": chunk_id, "dense_score": dense_score, "lexical_score": lexical_score,
                "expansion_query_match": expansion_match, "title_match": title_match,
                "selection_score": combined, "lexical_component": lexical_component,
            })
        first = min(rows, key=lambda row: (-row["dense_score"], -row["lexical_component"], row["chunk_id"]))
        pending = [row for row in rows if row is not first]
        selected = [{**first, "selection_route": "cached_dense"}]
        if pending and limit > 1:
            def lexical_key(row):
                duplicate = _redundancy(row["chunk_id"], first["chunk_id"], vectors.chunk_vectors, token_sets)
                return (-row["lexical_component"] + 0.15 * duplicate, -row["dense_score"], row["chunk_id"])

            second = min(pending, key=lexical_key)
            pending.remove(second)
            selected.append({**second, "selection_route": "query_lexical_and_provenance"})
        if pending and limit > 2:
            def diversity_key(row):
                duplicate = max(_redundancy(row["chunk_id"], known["chunk_id"],
                                            vectors.chunk_vectors, token_sets) for known in selected)
                return (-row["selection_score"] + 0.30 * duplicate, -row["lexical_component"], row["chunk_id"])

            selected.append({**min(pending, key=diversity_key), "selection_route": "diversity"})
        out[doc] = selected
    return out


def _model_ranges(backend, metadata, question, pairs, records):
    """Check the actual pinned tokenizer offsets against cached score lengths."""
    if not pairs:
        return {}
    encoder = getattr(backend, "_encoder", None)
    adapter = getattr(encoder, "model", None)
    tokenizer = getattr(adapter, "tokenizer", None)
    if tokenizer is None or metadata.get("truncation_side") != "right":
        return {pair.chunk_id: ((None, None) if records[key][0].truncated else (len(question), len(pair.text)))
                for key, pair in pairs}
    # The score carries original and input token counts for both cache and inference.
    # Re-encoding with the same adapter gives exact offsets without another model call.
    texts = [(question, pair.text) for _, pair in pairs]
    encoded = tokenizer.encode_batch(texts)
    result = {}
    for (key, pair), view in zip(pairs, encoded, strict=True):
        lengths = records[key][0]
        if sum(view.attention_mask) != lengths.input_tokens:
            raise ValueError(f"Jina score/tokenizer context mismatch for {pair.chunk_id}")
        ends = [0, 0]
        for sequence_id, (start, end) in zip(view.sequence_ids, view.offsets, strict=True):
            if sequence_id in (0, 1) and end > start:
                ends[sequence_id] = max(ends[sequence_id], end)
        result[pair.chunk_id] = tuple(ends) if lengths.truncated else (len(question), len(pair.text))
    return result


class FairScorer:
    """Fit original chunks into a shared budget and persist exact local scores."""

    def __init__(self, cache_path, historical_paths, metadata, backend, *, vectors, candidates, chunks,
                 chunk_budget=CHUNK_BUDGET, batch_size=BATCH_SIZE, max_parents=MAX_PARENTS):
        if type(chunk_budget) is not int or not 1 <= chunk_budget <= CHUNK_BUDGET:
            raise ValueError("Invalid per-parent Chunk budget")
        if type(batch_size) is not int or not 1 <= batch_size <= 64:
            raise ValueError("Invalid Jina batch size")
        if type(max_parents) is not int or not 1 <= max_parents <= MAX_PARENTS:
            raise ValueError("Invalid parent budget")
        self.history = dict(historical_paths) if isinstance(historical_paths, Mapping) else {
            f"historical_{i}": Path(path) for i, path in enumerate(historical_paths or ())
        }
        self.cache_path = cache_path
        self.metadata = metadata
        self.backend = backend
        self.vectors = vectors
        self.candidates = candidates
        self.chunks = chunks
        self.chunk_budget = chunk_budget
        self.batch_size = batch_size
        self.max_parents = max_parents
        self.cache = None
        self.cache_error = None
        all_tokens = {c.chunk_id: _tokens(c.text) for blocks in chunks.values() for c in blocks}
        self.token_sets = {key: frozenset(tokens) - STOP_WORDS for key, tokens in all_tokens.items()}
        self.token_texts = {key: " " + " ".join(tokens) + " " for key, tokens in all_tokens.items()}
        self.title_terms = {doc: frozenset(_tokens(candidate.title)) for doc, candidate in candidates.items()}

    def __enter__(self):
        try:
            self.cache = NeedScoreCache(self.cache_path, self.history, self.metadata)
        except (OSError, sqlite3.Error, ValueError) as error:
            self.cache_error = f"jina_cache_unavailable:{type(error).__name__}"
        return self

    def __exit__(self, *error):
        if self.cache is not None:
            self.cache.__exit__(*error)
            self.cache = None

    def _template_pairs(self, candidates):
        pool = SimpleNamespace(candidates=[SimpleNamespace(document_id=doc, title=candidate.title,
                                                          chunks=self.chunks[doc])
                                          for doc, candidate in candidates.items()])
        return {pair.chunk_id: pair for pair in make_pairs(pool)}

    def score(self, query_id, question, document_ids, expansion_terms=None, *, deadline=None):
        started = perf_counter()
        selection_seconds = 0.0
        result = {"documents": {}, "actual_pairs": 0, "historical_hits": 0, "local_hits": 0, "new_pairs": 0,
                  "duplicate_pairs": 0, "truncated_pairs": 0, "inference_seconds": 0.0,
                  "wall_seconds": 0.0, "fallback_reason": None}
        try:
            if deadline is not None and monotonic() >= deadline:
                raise TimeoutError
            ids = tuple(dict.fromkeys(document_ids))
            if len(ids) > self.max_parents:
                raise ValueError("Candidate pool exceeds fair Jina parent budget")
            candidates = {doc: self.candidates[doc] for doc in ids}
            selected = select_chunks(question, candidates, self.chunks, self.vectors, expansion_terms,
                                     limit=self.chunk_budget, token_sets=self.token_sets,
                                     token_texts=self.token_texts, title_terms=self.title_terms)
            selection_seconds = perf_counter() - started
            templates = self._template_pairs(candidates)
            hashed = [(digest(templates[row["chunk_id"]].text), templates[row["chunk_id"]])
                      for entries in selected.values() for row in entries]
            result["actual_pairs"] = len(hashed)
            if hashed and self.cache is None:
                result["fallback_reason"] = self.cache_error or "jina_cache_not_open"
            elif hashed:
                query_hash = digest(question)
                unique = dict(hashed)
                records = self.cache.lookup_pairs(query_hash, unique)
                result["local_hits"] = sum(source == "local_cache" for _, source in records.values())
                result["historical_hits"] = len(records) - result["local_hits"]
                result["duplicate_pairs"] = len(hashed) - len(unique)
                missing = [(key, pair) for key, pair in unique.items() if key not in records]
                for offset in range(0, len(missing), self.batch_size):
                    if deadline is not None and monotonic() >= deadline:
                        raise TimeoutError
                    if self.backend is None:
                        raise RuntimeError("Jina backend unavailable")
                    batch = missing[offset:offset + self.batch_size]
                    result["new_pairs"] += len(batch)
                    scores = list(self.backend.score(question, [pair.text for _, pair in batch]))
                    self.cache.put_batch(query_id, query_hash, batch, scores)
                    result["inference_seconds"] += sum(score.inference_seconds for score in scores)
                    for (key, _), score in zip(batch, scores, strict=True):
                        records[key] = (score, "new_inference")
                model_ranges = _model_ranges(self.backend, self.metadata, question, hashed, records)
                by_chunk = {pair.chunk_id: records[key] for key, pair in hashed}
                for doc, entries in selected.items():
                    originals = {chunk.chunk_id: chunk for chunk in self.chunks[doc]}
                    for row in entries:
                        chunk_id = row["chunk_id"]
                        score, source = by_chunk[chunk_id]
                        if not isfinite(score.score):
                            raise ValueError(f"Non-finite score at {chunk_id}")
                        result["truncated_pairs"] += bool(score.truncated)
                        query_end, pair_end = model_ranges[chunk_id]
                        original = originals[chunk_id].text
                        template = templates[chunk_id].text
                        if not template.endswith(original):
                            raise ValueError(f"Jina role mapping differs from original Chunk at {chunk_id}")
                        visible_chars = None if pair_end is None else max(0, pair_end - (len(template) - len(original)))
                        row.update({
                            "jina_score": score.score, "score": score.score, "cache_source": source,
                            "input_tokens": score.input_tokens, "original_tokens": score.original_tokens,
                            "truncated": bool(score.truncated), "model_visible_chunk_chars": visible_chars,
                            "model_visible_query_chars": query_end,
                        })
                result["documents"] = selected
        except TimeoutError:
            result["fallback_reason"] = "deadline_exhausted"
        except (OSError, sqlite3.Error) as error:
            result["fallback_reason"] = f"jina_cache_or_io_error:{type(error).__name__}"
        except (ValueError, TypeError, RuntimeError, KeyError, ArithmeticError) as error:
            result["fallback_reason"] = f"jina_scoring_error:{type(error).__name__}"
        result["selection_seconds"] = selection_seconds
        result["wall_seconds"] = perf_counter() - started
        return result

