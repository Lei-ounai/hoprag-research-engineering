"""Load, inspect, and sample the official MultiHop-RAG JSON release."""

from collections import Counter, defaultdict
from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
from pathlib import Path
import json
import random
from typing import Any, Sequence

from .models import Document, Evidence, Query


DATASET_URL = "https://huggingface.co/datasets/yixuantt/MultiHopRAG"
LICENSE_URL = "https://opendatacommons.org/licenses/by/1-0/"
PAPER_URL = "https://arxiv.org/abs/2401.15391"
CORPUS_FILENAME = "corpus.json"
QUERY_FILENAME = "MultiHopRAG.json"

_CORPUS_FIELDS = {"title", "body", "url", "author", "category", "published_at", "source"}
_EVIDENCE_FIELDS = (_CORPUS_FIELDS - {"body"}) | {"fact"}
_QUERY_FIELDS = {"query", "answer", "question_type", "evidence_list"}


@dataclass(frozen=True)
class Dataset:
    documents: tuple[Document, ...]
    queries: tuple[Query, ...]
    audit: dict[str, Any]


def _read_rows(path: Path) -> tuple[list[dict], dict]:
    content = path.read_bytes()
    try:
        rows = json.loads(content)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(rows, list) or not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"{path}: expected a nonempty array of objects")
    return rows, {
        "url": f"{DATASET_URL}/resolve/main/{path.name}",
        "bytes": len(content),
        "sha256": sha256(content).hexdigest(),
    }


def _required_text(row: dict, field: str, location: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location}.{field}: expected a nonempty string")
    return value


def _fields(rows: list[dict], expected: set[str]) -> dict:
    seen = set(expected).union(*(row.keys() for row in rows))
    return {
        "present": sorted({key for row in rows for key in row}),
        "missing": {
            key: count
            for key in sorted(seen)
            if (count := sum(key not in row for row in rows))
        },
        "null": dict(sorted(Counter(key for row in rows for key, v in row.items() if v is None).items())),
        "blank": dict(sorted(Counter(
            key for row in rows for key, v in row.items() if isinstance(v, str) and not v.strip()
        ).items())),
    }


def _record_key(row: dict) -> str:
    return json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _duplicate_field_count(rows: list[dict], key: str) -> int:
    values = [row[key] for row in rows if isinstance(row.get(key), str) and row[key].strip()]
    return len(values) - len(set(values))


def _official_text(text: str) -> str:
    return text.replace(" ", "").replace("\n", "")


def load_dataset(raw_directory: str | Path) -> Dataset:
    """Load official JSON while retaining article metadata and all benchmark queries."""
    raw_directory = Path(raw_directory)
    corpus_rows, corpus_file = _read_rows(raw_directory / CORPUS_FILENAME)
    query_rows, query_file = _read_rows(raw_directory / QUERY_FILENAME)

    documents = []
    ids_by_url: dict[str, list[str]] = defaultdict(list)
    body_by_id = {}
    seen_documents: set[str] = set()
    for index, row in enumerate(corpus_rows, start=1):
        location = f"{CORPUS_FILENAME} row {index}"
        title = _required_text(row, "title", location)
        text = _required_text(row, "body", location)
        record = _record_key(row)
        if record in seen_documents:
            continue
        seen_documents.add(record)
        document_id = f"doc-{index:04d}"
        metadata = {key: value for key, value in row.items() if key not in ("title", "body")}
        documents.append(Document(document_id, title, text, metadata))
        body_by_id[document_id] = _official_text(text)
        url = row.get("url")
        if isinstance(url, str) and url.strip():
            ids_by_url[url].append(document_id)

    queries = []
    evidence_rows = []
    mapping_counts: Counter[str] = Counter()
    duplicate_evidence_in_queries = 0
    duplicate_facts_in_queries = 0
    for index, row in enumerate(query_rows, start=1):
        location = f"{QUERY_FILENAME} row {index}"
        query_id = f"q-{index:04d}"
        question = _required_text(row, "query", location)
        answer = _required_text(row, "answer", location)
        question_type = _required_text(row, "question_type", location)
        gold = row.get("evidence_list")
        if not isinstance(gold, list) or any(not isinstance(item, dict) for item in gold):
            raise ValueError(f"{location}.evidence_list: expected an array of objects")
        if bool(gold) == (question_type == "null_query"):
            raise ValueError(f"{location}.evidence_list: inconsistent with {question_type!r}")
        evidence = []
        for position, item in enumerate(gold, start=1):
            fact = _required_text(item, "fact", f"{location}.evidence_list[{position}]")
            url = item.get("url")
            candidates = ids_by_url.get(url, ()) if isinstance(url, str) else ()
            source_id = None
            if len(candidates) == 1:
                source_id = candidates[0]
                mapping_counts["matched_unique_url"] += 1
                if _official_text(fact) not in body_by_id[source_id]:
                    mapping_counts["matched_url_fact_not_in_body"] += 1
            elif len(candidates) > 1:
                mapping_counts["ambiguous_url"] += 1
            else:
                mapping_counts["unmatched_url"] += 1
            metadata = {key: value for key, value in item.items() if key != "fact"}
            evidence.append(Evidence(f"{query_id}-e{position:02d}", fact, source_id, metadata))
        metadata = {key: value for key, value in row.items() if key not in _QUERY_FIELDS}
        queries.append(Query(query_id, question, answer, question_type, tuple(evidence), metadata))
        evidence_rows.extend(gold)
        duplicate_evidence_in_queries += len(gold) - len({_record_key(item) for item in gold})
        duplicate_facts_in_queries += len(gold) - len({item["fact"] for item in gold})

    types = Counter(query.question_type for query in queries)
    evidence_counts = Counter(len(query.evidence_list) for query in queries)
    strata = Counter((query.question_type, len(query.evidence_list)) for query in queries)
    audit = {
        "dataset": "yixuantt/MultiHopRAG",
        "license": "ODC-BY 1.0",
        "license_url": LICENSE_URL,
        "paper": "Yixuan Tang and Yi Yang. MultiHop-RAG: Benchmarking Retrieval-Augmented Generation for Multi-Hop Queries. 2024. arXiv:2401.15391.",
        "paper_url": PAPER_URL,
        "files": {CORPUS_FILENAME: corpus_file, QUERY_FILENAME: query_file},
        "counts": {
            "documents_raw": len(corpus_rows),
            "documents": len(documents),
            "queries": len(queries),
            "answerable_queries": len(queries) - types["null_query"],
            "unanswerable_queries": types["null_query"],
            "evidence": len(evidence_rows),
        },
        "question_types": dict(sorted(types.items())),
        "evidence_counts": {str(key): value for key, value in sorted(evidence_counts.items())},
        "strata": [
            {"question_type": question_type, "evidence_count": count, "queries": total}
            for (question_type, count), total in sorted(strata.items())
        ],
        "fields": {
            "document": _fields(corpus_rows, _CORPUS_FIELDS),
            "query": _fields(query_rows, _QUERY_FIELDS),
            "evidence": _fields(evidence_rows, _EVIDENCE_FIELDS),
        },
        "ids": {
            "native_document_ids": sum("document_id" in row or "id" in row for row in corpus_rows),
            "native_query_ids": sum("query_id" in row or "id" in row for row in query_rows),
            "native_evidence_document_ids": sum(
                "source_document_id" in row or "document_id" in row for row in evidence_rows
            ),
            "document_id_rule": "doc-NNNN, 1-based original corpus JSON row",
            "query_id_rule": "q-NNNN, 1-based original query JSON row",
            "evidence_id_rule": "q-NNNN-eNN, 1-based evidence position within query",
        },
        "duplicates": {
            "document_rows": len(corpus_rows) - len(documents),
            "document_title": _duplicate_field_count(corpus_rows, "title"),
            "document_url": _duplicate_field_count(corpus_rows, "url"),
            "document_body": _duplicate_field_count(corpus_rows, "body"),
            "query_rows": len(query_rows) - len({_record_key(row) for row in query_rows}),
            "query_text": _duplicate_field_count(query_rows, "query"),
            "evidence_rows_across_dataset": len(evidence_rows) - len({
                _record_key(row) for row in evidence_rows
            }),
            "evidence_rows_within_queries": duplicate_evidence_in_queries,
            "evidence_facts_within_queries": duplicate_facts_in_queries,
        },
        "evidence_document_mapping": {
            "method": "exact evidence URL to a unique corpus document URL",
            "matched_unique_url": mapping_counts["matched_unique_url"],
            "ambiguous_url": mapping_counts["ambiguous_url"],
            "unmatched_url": mapping_counts["unmatched_url"],
            "matched_url_fact_not_in_body": mapping_counts["matched_url_fact_not_in_body"],
        },
    }
    return Dataset(tuple(documents), tuple(queries), audit)


def stratified_sample(queries: Sequence[Query], size: int, seed: int = 42) -> tuple[Query, ...]:
    """Allocate proportional quotas by type and fact count, then sample without replacement."""
    if not 1 <= size <= len(queries):
        raise ValueError(f"sample size must be from 1 to {len(queries)}")
    if size == len(queries):
        return tuple(queries)

    groups: dict[tuple[str, int], list[Query]] = defaultdict(list)
    for query in queries:
        groups[query.question_type, len(query.evidence_list)].append(query)

    # Include each nonempty stratum when possible, then apportion the remaining
    # slots by the largest remainder method on remaining stratum capacity.
    baseline = int(size >= len(groups))
    slots = size - baseline * len(groups)
    capacity = len(queries) - baseline * len(groups)
    shares = {key: Fraction(slots * (len(members) - baseline), capacity) for key, members in groups.items()}
    quotas = {key: baseline + int(share) for key, share in shares.items()}
    remaining = size - sum(quotas.values())
    order = sorted(groups, key=lambda key: (-(shares[key] % 1), key))
    for key in order[:remaining]:
        quotas[key] += 1

    rng = random.Random(seed)
    selected = []
    for key in sorted(groups):
        members = sorted(groups[key], key=lambda query: query.query_id)
        selected.extend(rng.sample(members, quotas[key]))
    rng.shuffle(selected)
    return tuple(selected)
