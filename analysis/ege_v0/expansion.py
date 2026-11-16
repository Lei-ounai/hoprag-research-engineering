"""Bounded, question-aware lexical candidate expansion over immutable parents and chunks."""

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from functools import lru_cache
from math import isfinite, log, log2
import re
from time import perf_counter

from hoprag.models import Document
from hoprag.retrievers import BM25Retriever, Chunk, _WORD, _tokens


_STOP = frozenset("""
a an and are as at be been being but by can come could did do does doing during each
either even for from had has have having he her here him his how i if in into is it
its just may me might more most much my neither new no nor not of on one only or
other our out over own report reported reports said say says she should so some such
than that the their them then there these they this those through to too under up us
very via was we were what when where which while who whom whose why will with would
you your article articles according including using typically also first both across
""".split())
_RELATIONS = frozenset("""
after before earlier later initially subsequently same different more less better
worse between compared compare comparison versus vs oldest newest previous next
denied deny denies never neither nor no not without against acquired acquisition
association associated owned ownership announced accused claims claim
published founded denied named represented prompted following related involved
signed played joined contributor under over above below alongside
""".split())
_MONTHS = frozenset("""
january february march april may june july august september october november december
""".split())
_SEPARATE = re.compile(r"[\n.!?;:(){}\[\]\"“”]")
_YEAR = re.compile(r"(?:19|20)\d{2}$")
_CONSTRAINT_TOKEN = re.compile(r"\b(?:cannot|without|neither|never|nor|not|no|early|late|only|least|most|all|each|every|last|former|roughly|approximately|at)\b", re.I)


@dataclass(frozen=True)
class ExpansionConfig:
    seed_count: int = 3
    max_hops: int = 2
    max_queries: int = 5
    first_hop_queries: int = 3
    max_new_documents: int = 16
    hits_per_query: int = 7
    second_hop_seeds: int = 3
    second_hop_min_score: float = 0.35
    query_style: str = "q1"
    path_decay: float = 0.62

    def __post_init__(self):
        if self.seed_count not in (3, 5):
            raise ValueError("Select three or five seeds")
        if self.max_hops not in (1, 2) or not 1 <= self.max_queries <= 5:
            raise ValueError("Expansion supports one or two hops and at most five queries")
        if not 1 <= self.first_hop_queries <= self.max_queries:
            raise ValueError("First-hop query cap must fit the total budget")
        if not 1 <= self.max_new_documents <= 50 or not 5 <= self.hits_per_query <= 10:
            raise ValueError("Invalid parent cap or per-query hit budget")
        if not 1 <= self.second_hop_seeds <= 5 or not 0 < self.second_hop_min_score <= 1:
            raise ValueError("Invalid second-hop frontier")
        if self.query_style not in ("q1", "q2", "q3") or not 0 < self.path_decay < 1:
            raise ValueError("Invalid query style or path decay")


@dataclass(frozen=True)
class _Q:
    query_id: str
    question: str


@dataclass(frozen=True)
class _Span:
    term: str
    words: tuple[str, ...]
    start: int
    end: int
    kind: str


def _titlecase(surface):
    return len(surface) >= 2 and surface[0].isupper() and any(char.islower() for char in surface)


def _spans(text):
    tokens = tuple(_WORD.finditer(text))
    for index, occurrence in enumerate(tokens):
        start, end = occurrence.span()
        original = occurrence.group()
        words = tuple(_tokens(original))
        if len(words) != 1 or words[0] in _STOP or words[0] in _MONTHS or not original.isalpha() or len(original) < 2:
            continue
        kind = "acronym" if 2 <= len(original) <= 10 and original.isupper() else "word"
        if len(original) >= 3 or kind == "acronym":
            yield _Span(original, words, start, end, kind)
        is_named = original.isupper() or _titlecase(original)
        if not is_named:
            if index + 1 < len(tokens):
                tail = tokens[index + 1]
                between = text[end:tail.start()]
                following = tuple(_tokens(tail.group()))
                if (len(following) == 1 and len(tail.group()) >= 3 and tail.group().isalpha()
                        and following[0] not in _STOP | _MONTHS and not _SEPARATE.search(between)):
                    yield _Span(text[start:tail.end()], words + following, start, tail.end(), "phrase")
            continue
        span_words = list(words)
        tail_end = end
        last_is_name = True
        for tail in tokens[index + 1:index + 5]:
            between = text[tail_end:tail.start()]
            tail_word = tail.group()
            following = tuple(_tokens(tail_word))
            if (len(following) != 1 or not tail_word.isalpha() or _SEPARATE.search(between)):
                break
            if _titlecase(tail_word) or (len(tail_word) > 1 and tail_word.isupper()):
                last_is_name = True
            elif following[0] in {"of", "the", "and", "de", "van"} and last_is_name:
                last_is_name = False
            else:
                break
            if following[0] in _MONTHS:
                break
            span_words.extend(following)
            tail_end = tail.end()
            if last_is_name:
                yield _Span(text[start:tail_end], tuple(span_words), start, tail_end, "name")


class LexicalIndex:
    """Use the original parent BM25 and exact normalized token positions."""

    def __init__(self, documents: Iterable[Document], chunks: Mapping[str, Sequence[Chunk]]):
        indexed = tuple(documents)
        self.documents = {doc.document_id: doc for doc in indexed}
        if len(indexed) != len(self.documents):
            raise ValueError("Duplicate parent document IDs")
        self.chunks = {doc_id: tuple(chunks[doc_id]) for doc_id in self.documents}
        self.chunk_map = {}
        self.chunk_offsets = {}
        self.chunk_terms = {}
        self._chunk_text = {}
        self._title_terms = {}
        self._parent_text = {}
        self._postings = {}
        for doc_id, doc in self.documents.items():
            self._title_terms[doc_id] = frozenset(_tokens(doc.title))
            merged = f"{doc.title}\n{doc.text}"
            terms = _tokens(merged)
            self._parent_text[doc_id] = f" {' '.join(terms)} "
            for word in frozenset(terms):
                self._postings.setdefault(word, set()).add(doc_id)
            cursor = 0
            if not self.chunks[doc_id]:
                raise ValueError(f"Parent {doc_id} has no source chunks")
            for chunk in self.chunks[doc_id]:
                start = merged.find(chunk.text, cursor)
                if start == -1 or chunk.document_id != doc_id or chunk.chunk_id in self.chunk_map:
                    raise ValueError(f"Invalid source chunk mapping for {chunk.chunk_id}")
                self.chunk_map[chunk.chunk_id] = chunk
                self.chunk_offsets[chunk.chunk_id] = start
                chunk_words = _tokens(chunk.text)
                self.chunk_terms[chunk.chunk_id] = frozenset(chunk_words)
                self._chunk_text[chunk.chunk_id] = f" {' '.join(chunk_words)} "
                cursor = start + len(chunk.text)
        self._retriever = BM25Retriever(self.documents.values())
        self._max_df = max(2, int(0.18 * len(self.documents)))
        self.term_docs = lru_cache(maxsize=30000)(self._term_docs)
        self._rank_query = lru_cache(maxsize=2048)(self._search)
        self._chunk_spans = lru_cache(maxsize=2600)(self._index_spans)

    def _term_docs(self, words):
        if len(words) == 1:
            return frozenset(self._postings.get(words[0], ()))
        postings = tuple(self._postings.get(word, set()) for word in words)
        if any(not pool for pool in postings):
            return frozenset()
        phrase = f" {' '.join(words)} "
        return frozenset(doc_id for doc_id in set.intersection(*postings) if phrase in self._parent_text[doc_id])

    def idf(self, df):
        return log((len(self.documents) + 1) / (df + 1))

    def _index_spans(self, chunk_id):
        by_words = {}
        for span in _spans(self.chunk_map[chunk_id].text):
            docs = self.term_docs(span.words)
            if 2 <= len(docs) <= self._max_df and 2 <= len(span.term) <= 65:
                by_words.setdefault(span.words, []).append((span, len(docs)))
        return by_words

    def _search(self, query, term):
        matches = self.term_docs(term)
        if not matches:
            return ()
        ranked = self._retriever.search(_Q("ege", query), len(self.documents))
        return tuple(row for row in ranked if row.document_id in matches)

    def search(self, query, term, limit):
        """Rank across the corpus then retain exactly connected phrase matches."""
        return self._rank_query(query, term)[:limit]

    def witness(self, document_id, question_words, term=()):
        phrase = f" {' '.join(term)} " if term else None
        chunks = self.chunks[document_id]
        return min(
            chunks,
            key=lambda chunk: (
                phrase is not None and phrase not in self._chunk_text[chunk.chunk_id],
                -sum(self.idf(len(self._postings.get(word, ())))
                     for word in sorted(question_words & self.chunk_terms[chunk.chunk_id])),
                chunk.chunk_id,
            ),
        )


def select_seeds(index, question, baseline_order, jina_ranking, count):
    semantic = {row["document_id"]: row for row in jina_ranking}
    query_words = frozenset(_tokens(question)) - (_STOP - _RELATIONS)
    seeds = []
    for rank, document_id in enumerate(baseline_order[:count], 1):
        doc = index.documents[document_id]
        archived = semantic.get(document_id, {})
        chunk = index.chunk_map.get(archived.get("winning_chunk_id"))
        if chunk is None or chunk.document_id != document_id:
            chunk = index.witness(document_id, query_words)
        score = archived.get("score")
        seeds.append({
            "document_id": document_id,
            "chunk_id": chunk.chunk_id,
            "original_rank": rank,
            "jina_score": float(score) if score is not None and isfinite(score) else None,
            "title": doc.title,
            "text": chunk.text,
            "title_overlap": sorted(query_words & index._title_terms[document_id]),
            "path": [document_id],
            "path_terms": [],
            "seed_priority": 1.0 / (1.0 + 0.05 * (rank - 1)),
        })
    return seeds


def _term_record(index, span, df, text, score, overlap, seed=None, hop=1):
    chunk_offset = index.chunk_offsets[seed["chunk_id"]] if seed else 0
    body_start = len(index.documents[seed["document_id"]].title) + 1 if seed else 0
    start, end = chunk_offset + span.start, chunk_offset + span.end
    return {
        "term": span.term,
        "term_tokens": list(span.words),
        "term_document_frequency": df,
        "idf": index.idf(df),
        "term_kind": span.kind,
        "seed_document_id": seed["document_id"] if seed else None,
        "seed_chunk_id": seed["chunk_id"] if seed else None,
        "position": {
            "offset_base": "title_plus_newline_plus_body" if seed else "question",
            "start": start,
            "end": end,
            "chunk_start": span.start if seed else None,
            "chunk_end": span.end if seed else None,
            "field": ("title" if start < body_start else "body") if seed else "question",
        },
        "excerpt": text[max(0, span.start - 70):min(len(text), span.end + 70)],
        "question_overlap": sorted(overlap),
        "term_score": score,
        "hop": hop,
    }


def extract_terms(index, question, seed, hop=1, *, visited=()):
    """Select novel connected terms and preserve their source character spans."""
    document_id = seed["document_id"]
    chunk = index.chunk_map[seed["chunk_id"]]
    if chunk.document_id != document_id:
        raise ValueError("Seed witness belongs to a different parent")
    chunks = [chunk]
    title_chunk = index.chunks[document_id][0]
    if title_chunk.chunk_id != chunk.chunk_id:
        chunks.append(title_chunk)
    question_words = frozenset(_tokens(question)) - (_STOP - _RELATIONS)
    terms = {}
    original_question = f" {' '.join(_tokens(question))} "
    unavailable = set(visited) | {document_id, *seed["path"]}
    for source in chunks:
        variants = index._chunk_spans(source.chunk_id)
        title_end = len(index.documents[document_id].title) - index.chunk_offsets[source.chunk_id]
        origin = {**seed, "chunk_id": source.chunk_id}
        title_overlap = question_words & index._title_terms[document_id]
        for words, occurrences in variants.items():
            phrase = f" {' '.join(words)} "
            if phrase in original_question or not (index.term_docs(words) - unavailable):
                continue
            for span, df in occurrences:
                title = span.start < title_end
                if source != chunk and not title:
                    continue
                context = source.text[max(0, span.start - 95):min(len(source.text), span.end + 95)]
                overlap = question_words & (frozenset(_tokens(context)) | title_overlap)
                relation_overlap = overlap & _RELATIONS
                quality = {"name": 0.92, "acronym": 0.7, "phrase": 0.38, "word": 0.0}[span.kind]
                if not overlap and not (span.kind in {"name", "acronym"} and title):
                    continue
                if span.kind == "word" and len(overlap) < 2:
                    continue
                if hop == 2 and not overlap:
                    continue
                score = (index.idf(df) + quality + 0.85 * title + 0.44 * min(4, len(overlap))
                         + 0.22 * min(2, len(relation_overlap))
                         + 0.12 * float(seed["seed_priority"]))
                current = terms.get(words)
                if current is None or score > current["term_score"]:
                    terms[words] = _term_record(index, span, df, source.text, score, overlap, origin, hop)
    return sorted(terms.values(), key=_term_order)


def _term_order(term):
    return (-term["term_score"], term["term_document_frequency"],
            tuple(term["term_tokens"]), term["seed_document_id"] or "", term["position"]["start"])


def _question_terms(index, question):
    variants = {}
    for span in _spans(question):
        df = len(index.term_docs(span.words))
        if 2 <= df <= index._max_df:
            score = index.idf(df) + 0.70 * (len(span.words) > 1) + 0.5 * (span.kind == "acronym")
            current = variants.get(span.words)
            if current is None or score > current["term_score"]:
                variants[span.words] = _term_record(index, span, df, question, score, ())
    return sorted(variants.values(), key=_term_order)


def build_query(index, question, term, style, *, question_only=False):
    if question_only or style == "q2":
        normalized = _tokens(question)
        content = [word for word in dict.fromkeys(normalized) if word not in _STOP]
        ranked = sorted(content, key=lambda word: (-index.idf(len(index.term_docs((word,)))), word))
        names = {word for span in _spans(question) if span.kind in {"name", "acronym"} for word in span.words}
        evidence_terms = set(ranked[:(5 if question_only else 7)]) | names | _RELATIONS | _MONTHS
        retained = list(dict.fromkeys(
            word for word in normalized
            if word in evidence_terms or _YEAR.match(word) or word.isnumeric() or _CONSTRAINT_TOKEN.match(word)
        ))
        return " ".join(dict.fromkeys([*retained, *term["term_tokens"]]))
    if style == "q3":
        return term["term"]
    if style != "q1":
        raise ValueError("Unknown expansion query mode")
    return f"{question}\n{term['term']}"


def _ranked(candidates):
    return sorted(candidates, key=lambda row: (-row["score"], row["hop"], row["retrieval_rank"], row["document_id"]))


def _choose_terms(index, question, seeds, hop, seen_terms, visited, cap, question_only):
    if question_only:
        choices = _question_terms(index, question)
    else:
        choices = [
            term for seed in seeds for term in extract_terms(index, question, seed, hop, visited=visited)
        ]
    choices.sort(key=_term_order)
    counts = Counter()
    selected = []
    for _ in range(cap):
        available = [term for term in choices if tuple(term["term_tokens"]) not in seen_terms]
        if not available:
            break
        term = min(available, key=lambda item: (
            -item["term_score"] + 0.85 * counts[item["seed_document_id"]], *_term_order(item)[1:]
        ))
        selected.append(term)
        counts[term["seed_document_id"]] += 1
        seen_terms.add(tuple(term["term_tokens"]))
    return selected


def _candidate_path(index, query_words, term, seed, hit, traversal_rank, strength, query_style, hop, attenuation):
    doc_id = hit.document_id
    chunk = index.witness(doc_id, query_words, tuple(term["term_tokens"]))
    hop_score = (0.55 + 0.45 * strength) / log2(1 + traversal_rank)
    score = hop_score * attenuation * (seed["seed_priority"] if seed else 1.0)
    return {
        "document_id": doc_id,
        "chunk_id": chunk.chunk_id,
        "seed_document_id": term["seed_document_id"],
        "seed_chunk_id": term["seed_chunk_id"],
        "term": term["term"],
        "term_tokens": term["term_tokens"],
        "retriever": "bm25",
        "query_style": query_style,
        "hop": hop,
        "retrieval_rank": hit.rank,
        "connected_rank": traversal_rank,
        "raw_bm25_score": hit.score,
        "score": score,
        "path": [*(seed["path"] if seed else []), doc_id],
        "path_terms": [*(seed["path_terms"] if seed else []), term["term"]],
    }


def _hop(index, query_id, question, terms, seeds, baseline, prior, config, hop, question_only, start_id):
    style = "question-only" if question_only else config.query_style
    query_words = frozenset(_tokens(question)) - (_STOP - _RELATIONS)
    query_records = []
    nominated = {}
    prior_ids = {doc["document_id"] for doc in prior}
    visited = baseline | prior_ids
    bm25_seconds = 0.0
    for term in terms:
        request = build_query(index, question, term, config.query_style, question_only=question_only)
        started = perf_counter()
        rows = index.search(request, tuple(term["term_tokens"]), config.hits_per_query)
        elapsed = perf_counter() - started
        bm25_seconds += elapsed
        seed = seeds.get(term["seed_document_id"])
        max_score = max((hit.score for hit in rows), default=0.0)
        attenuation = 1.0 if hop == 1 else config.path_decay * seed["score"]
        records = []
        number = start_id + len(query_records) + 1
        request_id = f"{query_id}:exp:{number}"
        for rank, row in enumerate(rows, 1):
            strength = max(0.0, row.score / max_score) if max_score > 0 else 0.0
            item = _candidate_path(index, query_words, term, seed, row, rank, strength, style, hop, attenuation)
            item["request_id"] = request_id
            item["candidate_status"] = "already_visited" if row.document_id in visited else "budget_excluded"
            records.append(item)
            if row.document_id in visited:
                continue
            previous = nominated.get(row.document_id)
            if previous is None or item["score"] > previous["score"]:
                nominated[row.document_id] = item
        query_records.append({
            "request_id": request_id,
            "question": question,
            "query": request,
            "query_style": style,
            "hop": hop,
            "retriever": "bm25",
            "term": term,
            "hits": records,
            "bm25_seconds": elapsed,
        })
    new_candidates = _ranked(nominated.values())[:config.max_new_documents - len(prior)]
    admitted = {row["document_id"] for row in new_candidates}
    for query_record in query_records:
        for hit in query_record["hits"]:
            if hit["document_id"] in admitted:
                hit["candidate_status"] = ("admitted" if hit is nominated[hit["document_id"]]
                                           else "alternate_path")
    return query_records, new_candidates, bm25_seconds


def _frontier(index, candidates, config):
    if not candidates:
        return []
    eligible = _ranked(candidates)
    floor = max(config.second_hop_min_score, 0.5 * eligible[0]["score"])
    return [
        {
            **item,
            "title": index.documents[item["document_id"]].title,
            "text": index.chunk_map[item["chunk_id"]].text,
            "seed_priority": 1.0,
        }
        for item in eligible[:config.second_hop_seeds] if item["score"] >= floor
    ]


def _outcome(result, max_hops):
    phases = result["phases"][:max_hops]
    candidates = _ranked(row for phase in phases for row in phase["new_candidates"])
    queries = [row for phase in phases for row in phase["queries"]]
    return {
        **result,
        "config": {**result["config"], "max_hops": max_hops},
        "phases": phases,
        "queries": queries,
        "new_candidates": candidates,
        "candidate_document_ids": [*result["baseline_order"], *(row["document_id"] for row in candidates)],
        "bm25_seconds": sum(phase["bm25_seconds"] for phase in phases),
        "wall_seconds": result["preparation_seconds"] + sum(phase["wall_seconds"] for phase in phases),
        "max_hop_reached": max((query["hop"] for query in queries), default=0),
        "fallback_reason": None if candidates else ("no_expansion_hits" if queries else "no_valid_expansion_queries"),
    }


def through_hop(result, max_hop):
    """Project an archived result to the exact prefix of its executed hops."""
    if not 1 <= max_hop <= result["config"]["max_hops"]:
        raise ValueError("Requested an unavailable expansion depth")
    return _outcome(result, max_hop)


def expand(index, query_id, question, baseline_order, jina_ranking, *, config=ExpansionConfig(), question_only=False):
    """Read question text and immutable retrieval data, then expand through at most two hops."""
    started = perf_counter()
    baseline_order = list(baseline_order)
    baseline = set(baseline_order)
    if len(baseline) != len(baseline_order) or not baseline <= index.documents.keys():
        raise ValueError("Invalid original P5-B parent ranks")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question text must be a nonempty string")
    seeds = [] if question_only else select_seeds(index, question, baseline_order, jina_ranking, config.seed_count)
    result = {
        "query_id": query_id,
        "question": question,
        "baseline_order": baseline_order,
        "seeds": seeds,
        "config": asdict(config),
        "question_only": question_only,
        "preparation_seconds": perf_counter() - started,
        "scoring_model_pairs": 0,
        "embedding_requests": 0,
        "dense_seconds": 0.0,
        "phases": [],
    }
    current_seeds = seeds
    prior = []
    seen_terms = set()
    total_queries = 0
    max_hops = 1 if question_only else config.max_hops
    for hop in range(1, max_hops + 1):
        started = perf_counter()
        limit = config.first_hop_queries if hop == 1 else config.max_queries - total_queries
        if not limit or (hop == 2 and not current_seeds):
            break
        visited = baseline | {row["document_id"] for row in prior}
        terms = _choose_terms(index, question, current_seeds, hop, seen_terms, visited, limit, question_only)
        if not terms:
            result["phases"].append({
                "hop": hop, "queries": [], "new_candidates": [],
                "bm25_seconds": 0.0, "wall_seconds": perf_counter() - started,
            })
            break
        seed_lookup = {seed["document_id"]: seed for seed in current_seeds}
        queries, added, bm25_seconds = _hop(
            index, query_id, question, terms, seed_lookup, baseline, prior, config, hop, question_only, total_queries
        )
        result["phases"].append({
            "hop": hop,
            "queries": queries,
            "new_candidates": added,
            "bm25_seconds": bm25_seconds,
            "wall_seconds": perf_counter() - started,
        })
        prior = [*prior, *added]
        total_queries += len(queries)
        if not added or len(prior) >= config.max_new_documents:
            break
        if hop < max_hops:
            current_seeds = _frontier(index, added, config)
    return _outcome(result, max_hops)

