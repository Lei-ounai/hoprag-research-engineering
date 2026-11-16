from __future__ import annotations

import heapq
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    text: str
    links: tuple[str, ...] = ()
    source: str | None = None


@dataclass(frozen=True)
class EvidenceChain:
    documents: tuple[Document, ...]
    score: float


def _documents_by_id(documents: Iterable[Document]) -> dict[str, Document]:
    by_id: dict[str, Document] = {}
    for document in documents:
        if document.id in by_id:
            raise ValueError(f"文档 ID 重复：{document.id!r}")
        by_id[document.id] = document
    if not by_id:
        raise ValueError("语料至少需要一篇文档")
    for document in by_id.values():
        for target in document.links:
            if target not in by_id:
                raise ValueError(f"文档 {document.id!r} 链接到不存在的文档 {target!r}")
    return by_id


def load_corpus(path: str | Path) -> tuple[Document, ...]:
    """Read UTF-8 JSONL and validate document IDs and links."""
    documents = []
    with Path(path).open(encoding="utf-8-sig") as corpus:
        for line_number, line in enumerate(corpus, start=1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                if not isinstance(data, dict):
                    raise ValueError("每一行必须是一个 JSON 对象")
                for field in ("id", "title", "text"):
                    value = data.get(field)
                    if not isinstance(value, str) or not value.strip():
                        raise ValueError(f"{field} 必须是非空字符串")
                links = data.get("links", [])
                if not isinstance(links, list) or any(
                    not isinstance(link, str) or not link.strip() for link in links
                ):
                    raise ValueError("links 必须是非空字符串组成的数组")
                source = data.get("source")
                if source is not None and (not isinstance(source, str) or not source.strip()):
                    raise ValueError("source 必须是非空字符串或 null")
                documents.append(
                    Document(data["id"], data["title"], data["text"], tuple(links), source)
                )
            except ValueError as exc:
                raise ValueError(f"{path} 第 {line_number} 行：{exc}") from exc
    try:
        _documents_by_id(documents)
    except ValueError as exc:
        raise ValueError(f"{path}：{exc}") from exc
    return tuple(documents)


_HAN_RUN = re.compile(
    r"([\u3007\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U000323af]+)"
)
_WORD = re.compile(r"[^\W_]+")


def _tokens(text: str, *, index: bool = False) -> tuple[str, ...]:
    parts = _HAN_RUN.split(unicodedata.normalize("NFKC", text).casefold())
    tokens = []
    for position, part in enumerate(parts):
        if position % 2 == 0:
            tokens.extend(_WORD.findall(part))
        else:
            if index or len(part) == 1:
                tokens.extend(part)
            tokens.extend(part[i : i + 2] for i in range(len(part) - 1))
    return tuple(tokens)


def _contains_phrase(tokens: tuple[str, ...], phrase: tuple[str, ...]) -> bool:
    size = len(phrase)
    return bool(size) and any(
        tokens[i : i + size] == phrase for i in range(len(tokens) - size + 1)
    )


@dataclass(frozen=True)
class _Path:
    ids: tuple[str, ...]
    best_terms: dict[str, float]
    score: float


def _rank(path: _Path) -> tuple[float, int, tuple[str, ...]]:
    return (-path.score, len(path.ids), path.ids)


class Retriever:
    """Find question matches with BM25 and follow directed links in the corpus."""

    def __init__(self, documents: Iterable[Document]):
        self._documents = _documents_by_id(documents)
        self._postings: dict[str, dict[str, int]] = defaultdict(dict)
        self._lengths: dict[str, int] = {}
        self._titles: dict[str, tuple[str, ...]] = {}
        for doc in self._documents.values():
            tokens = _tokens(doc.title, index=True) + _tokens(doc.text, index=True)
            self._titles[doc.id] = _tokens(doc.title)
            self._lengths[doc.id] = len(tokens)
            for term, frequency in Counter(tokens).items():
                self._postings[term][doc.id] = frequency
        self._average_length = sum(self._lengths.values()) / len(self._documents) or 1.0

    def _term_scores(self, question: tuple[str, ...]) -> dict[str, dict[str, float]]:
        scores: dict[str, dict[str, float]] = defaultdict(dict)
        document_count = len(self._documents)
        for term in dict.fromkeys(question):
            postings = self._postings.get(term)
            if not postings:
                continue
            document_frequency = len(postings)
            inverse_frequency = math.log1p(
                (document_count - document_frequency + 0.5) / (document_frequency + 0.5)
            )
            for doc_id, frequency in postings.items():
                # BM25 with k1=1.2 and b=0.75.
                length_ratio = self._lengths[doc_id] / self._average_length
                scores[doc_id][term] = inverse_frequency * frequency * 2.2 / (
                    frequency + 1.2 * (0.25 + 0.75 * length_ratio)
                )
        return scores

    def _extend(
        self, frontier: list[_Path], scores: dict[str, dict[str, float]], hop: int
    ) -> Iterable[_Path]:
        for path in frontier:
            for target in dict.fromkeys(self._documents[path.ids[-1]].links):
                if target in path.ids:
                    continue
                best_terms = path.best_terms.copy()
                gain = 0.0
                for term, value in scores.get(target, {}).items():
                    previous = best_terms.get(term, 0.0)
                    if value > previous:
                        gain += value - previous
                        best_terms[term] = value
                # Keep weak intermediate steps eligible while preferring chains
                # that contribute question evidence sooner and with fewer links.
                score = path.score + gain / (1 + 0.15 * hop) - 0.15
                yield _Path((*path.ids, target), best_terms, score)

    def search(self, question: str, *, max_hops: int = 1, top_k: int = 3) -> list[EvidenceChain]:
        """Rank paths containing up to max_hops edges, including seed-only paths."""
        if max_hops < 0 or top_k < 1:
            raise ValueError("max_hops 必须大于或等于 0，top_k 必须大于或等于 1")
        query = _tokens(question)
        if not query:
            raise ValueError("问题必须包含可检索的文字或数字")

        term_scores = self._term_scores(query)
        seeds = []
        for doc_id, terms in term_scores.items():
            title = self._titles[doc_id]
            score = sum(terms.values())
            if _contains_phrase(query, title):
                score += sum(terms.get(term, 0.0) for term in dict.fromkeys(title))
            seeds.append(_Path((doc_id,), terms, score))

        beam_width = max(32, top_k)
        frontier = heapq.nsmallest(beam_width, seeds, key=_rank)
        best = frontier[:top_k]
        for hop in range(1, min(max_hops, len(self._documents) - 1) + 1):
            frontier = heapq.nsmallest(beam_width, self._extend(frontier, term_scores, hop), key=_rank)
            if not frontier:
                break
            best = heapq.nsmallest(top_k, (*best, *frontier), key=_rank)

        return [
            EvidenceChain(tuple(self._documents[doc_id] for doc_id in path.ids), path.score)
            for path in best
        ]
