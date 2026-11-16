"""Freeze round-five candidates and stratified query cohorts from saved ranks."""

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from statistics import mean
from uuid import uuid4
from typing import Iterable, Iterator

from analysis.rankings import iter_rankings, load_chunks
from hoprag.dataset import load_dataset, stratified_sample
from hoprag.retrievers import Chunk


PROJECT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent
BASELINE = PROJECT / "results" / "baseline"
RAW_DATA = PROJECT / "data" / "raw" / "multihop_rag"
MANIFEST_FILE = "candidate_pool_manifest.json"
POOLS_FILE = "candidate_pools.jsonl"
DOCUMENTS_FILE = "candidate_documents.jsonl"
SPLITS_FILE = "splits.json"
ARTIFACTS = (POOLS_FILE, DOCUMENTS_FILE, SPLITS_FILE)
SEED = 42


@dataclass(frozen=True, slots=True)
class Candidate:
    document_id: str
    rrf_rank: int
    bm25_rank: int | None
    dense_rank: int | None
    bm25_score: float | None
    dense_score: float | None
    rrf_score: float
    source: tuple[str, ...]
    title: str
    text: str
    chunks: tuple[Chunk, ...]
    article_source: str | None


@dataclass(frozen=True, slots=True)
class QueryPool:
    query_id: str
    question: str
    candidates: tuple[Candidate, ...]


def _read_json(path):
    return json.loads(path.read_bytes())


def _write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def _line(data):
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"


def _fingerprint(path):
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"sha256": digest, "bytes": path.stat().st_size}


def _split_group(queries, seed):
    # The release counts round to 451 answerable and 60 null development queries.
    count = (len(queries) + 2) // 5
    dev = stratified_sample(queries, count, seed)
    selected = {q.query_id for q in dev}
    return dev, tuple(q for q in queries if q.query_id not in selected)


def build_splits(queries, seed=SEED):
    answerable = tuple(q for q in queries if q.is_answerable)
    null = tuple(q for q in queries if not q.is_answerable)
    answerable_dev, answerable_heldout = _split_group(answerable, seed)
    null_dev, null_heldout = _split_group(null, seed)
    dev100 = stratified_sample(answerable_dev, 100, seed)
    dev20 = stratified_sample(dev100, 20, seed)
    dev5 = stratified_sample(dev20, 5, seed)

    cohorts = {
        "answerable_dev": answerable_dev,
        "answerable_heldout": answerable_heldout,
        "null_dev": null_dev,
        "null_heldout": null_heldout,
        "dev_5": dev5,
        "dev_20": dev20,
        "dev_100": dev100,
    }
    strata = {}
    for name, cohort in cohorts.items():
        counts = Counter((q.question_type, len(q.evidence_list)) for q in cohort)
        strata[name] = [
            {"question_type": question_type, "evidence_count": facts, "queries": count}
            for (question_type, facts), count in sorted(counts.items())
        ]

    return {
        "seed": seed,
        "design": {
            "split": "20% rounded to nearest integer; answerable and null independently",
            "strata": "question_type and necessary evidence count",
            "sampling": "largest remainder within strata via hoprag.dataset.stratified_sample",
            "nested": "sample dev_100 from answerable_dev, dev_20 from dev_100, dev_5 from dev_20",
        },
        **{name: [q.query_id for q in cohort] for name, cohort in cohorts.items()},
        "strata": strata,
    }


def build_candidate_record(query_id, question, rankings):
    """Serialize only question text and baseline ranks, never dataset annotations."""
    bm25 = {row.document_id: row for row in rankings["bm25"]}
    dense = {row.document_id: row for row in rankings["dense"]}
    union = bm25.keys() | dense.keys()
    ordered = rankings["rrf"]
    if len(ordered) != len(union) or {row.document_id for row in ordered} != union:
        raise ValueError(f"RRF did not preserve the candidate union for {query_id}")

    candidates = []
    for row in ordered:
        doc_id = row.document_id
        lexical, vector = bm25.get(doc_id), dense.get(doc_id)
        candidates.append({
            "document_id": doc_id,
            "bm25_rank": lexical.rank if lexical else None,
            "dense_rank": vector.rank if vector else None,
            "bm25_score": lexical.score if lexical else None,
            "dense_score": vector.score if vector else None,
            "rrf_rank": row.rank,
            "rrf_score": row.score,
            "source": [name for name, match in (("bm25", lexical), ("dense", vector)) if match],
        })
    return {"query_id": query_id, "question": question, "candidates": candidates}


def _cohort_stats(rows):
    parent_counts = [parent for parent, _ in rows]
    chunk_counts = [chunks for _, chunks in rows]
    return {
        "queries": len(rows),
        "candidate_parent_pairs": sum(parent_counts),
        "candidate_chunk_pairs": sum(chunk_counts),
        "minimum_parents_per_query": min(parent_counts),
        "maximum_parents_per_query": max(parent_counts),
        "mean_parents_per_query": mean(parent_counts),
        "minimum_chunks_per_query": min(chunk_counts),
        "maximum_chunks_per_query": max(chunk_counts),
        "mean_chunks_per_query": mean(chunk_counts),
    }


def _source_paths():
    return (
        BASELINE / "rankings_full.jsonl",
        BASELINE / "run.json",
        BASELINE / "chunks.jsonl",
        RAW_DATA / "corpus.json",
        RAW_DATA / "MultiHopRAG.json",
    )


def read_manifest(output=OUTPUT, *, verify=True):
    output = Path(output)
    manifest = _read_json(output / MANIFEST_FILE)
    if manifest.get("version") != 1 or set(manifest["artifacts"]) != set(ARTIFACTS):
        raise ValueError("Unrecognized frozen candidate manifest")
    if verify:
        for name, expected in manifest["artifacts"].items():
            if _fingerprint(output / name) != expected:
                raise ValueError(f"Frozen candidate artifact changed: {name}")
    return manifest


def freeze(output=OUTPUT):
    output = Path(output)
    sources = {p.relative_to(PROJECT).as_posix(): _fingerprint(p) for p in _source_paths()}
    if (output / MANIFEST_FILE).exists():
        previous = read_manifest(output)
        if previous["source_artifacts"] != sources:
            raise ValueError("Baseline or raw dataset changed after candidate freeze")
        return previous

    dataset = load_dataset(RAW_DATA)
    docs = {doc.document_id: doc for doc in dataset.documents}
    config = _read_json(BASELINE / "run.json")["config"]
    if (config["fusion_depth"], config["rrf_k"], config["top_k"]) != (100, 60, 10):
        raise ValueError("Expected saved baseline input depth 100, k=60 and output depth 10")
    chunks = load_chunks(BASELINE / "chunks.jsonl", docs, config["max_chunk_bytes"])
    splits = build_splits(dataset.queries)

    output.mkdir(parents=True, exist_ok=True)
    stage = output / f".candidate-freeze-{uuid4().hex}"
    stage.mkdir()
    try:
        # The staging directory inherits project ACLs before atomic publication.
        query_counts = {}
        with (stage / POOLS_FILE).open("w", encoding="utf-8", newline="\n") as stream:
            rankings = iter_rankings(BASELINE / "rankings_full.jsonl", dataset.queries, docs.keys(), config)
            for query, ranks in rankings:
                record = build_candidate_record(query.query_id, query.question, ranks)
                parent_ids = tuple(row["document_id"] for row in record["candidates"])
                if not 10 <= len(parent_ids) <= 200:
                    raise ValueError(f"Invalid number of candidate parents for {query.query_id}")
                query_counts[query.query_id] = (len(parent_ids), sum(len(chunks[doc_id]) for doc_id in parent_ids))
                stream.write(_line(record))

        with (stage / DOCUMENTS_FILE).open("w", encoding="utf-8", newline="\n") as stream:
            for doc_id, doc in docs.items():
                stream.write(_line({
                    "document_id": doc_id,
                    "title": doc.title,
                    "text": doc.text,
                    "article_source": doc.metadata.get("source"),
                    "article_url": doc.metadata.get("url"),
                    "chunks": [asdict(chunk) for chunk in chunks[doc_id]],
                }))
        _write_json(stage / SPLITS_FILE, splits)

        cohort_keys = ("answerable_dev", "answerable_heldout", "null_dev", "null_heldout", "dev_5", "dev_20", "dev_100")
        scopes = {key: _cohort_stats([query_counts[qid] for qid in splits[key]]) for key in cohort_keys}
        scopes["all"] = _cohort_stats(list(query_counts.values()))
        scopes["answerable_all"] = _cohort_stats([
            query_counts[qid] for qid in splits["answerable_dev"] + splits["answerable_heldout"]
        ])
        scopes["null_all"] = _cohort_stats([query_counts[qid] for qid in splits["null_dev"] + splits["null_heldout"]])
        manifest = {
            "version": 1,
            "candidate_definition": "unique BM25 Top 100 union Dense Top 100 by parent document ID",
            "maximum_candidate_parents_per_query": 200,
            "rrf_k": 60,
            "original_rrf_top10_validated_for_all_queries": True,
            "chunk_handling": (
                "all original chunks for each admitted parent; saved Dense ranks have no selected chunk ID "
                "and BM25 indexed full parents"
            ),
            "join": {
                POOLS_FILE: "join candidates[].document_id to candidate_documents.jsonl document_id",
                DOCUMENTS_FILE: "all original chunk_id, document_id (parent), text; full parent title and text",
                SPLITS_FILE: "fixed query ID cohorts; no answers or evidence texts",
            },
            "source_artifacts": sources,
            "artifacts": {name: _fingerprint(stage / name) for name in ARTIFACTS},
            "document_count": len(docs),
            "chunk_count": sum(map(len, chunks.values())),
            "cohorts": scopes,
        }
        _write_json(stage / MANIFEST_FILE, manifest)

        for name in (*ARTIFACTS, MANIFEST_FILE):
            source, destination = stage / name, output / name
            if destination.exists():
                if _fingerprint(source) != _fingerprint(destination):
                    raise ValueError(f"Refusing to replace a different existing freeze file: {destination}")
            else:
                source.rename(destination)
        return manifest
    finally:
        for name in (*ARTIFACTS, MANIFEST_FILE):
            (stage / name).unlink(missing_ok=True)
        stage.rmdir()


def load_splits(output=OUTPUT):
    return _read_json(Path(output) / SPLITS_FILE)


def iter_pools(query_ids: Iterable[str] | None = None, *, output=OUTPUT) -> Iterator[QueryPool]:
    """Read frozen score inputs without loading the raw query or evidence dataset."""
    output = Path(output)
    read_manifest(output)
    requested = set(query_ids) if query_ids is not None else None
    if requested == set():
        return

    docs = {}
    with (output / DOCUMENTS_FILE).open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            doc_id = row["document_id"]
            docs[doc_id] = (
                row["title"], row["text"], tuple(Chunk(**chunk) for chunk in row["chunks"]), row["article_source"]
            )

    with (output / POOLS_FILE).open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            query_id = row["query_id"]
            if requested is not None and query_id not in requested:
                continue
            candidates = []
            for item in row["candidates"]:
                title, text, chunks, article_source = docs[item["document_id"]]
                item["source"] = tuple(item["source"])
                candidates.append(Candidate(**item, title=title, text=text, chunks=chunks, article_source=article_source))
            yield QueryPool(query_id, row["question"], tuple(candidates))
            if requested is not None:
                requested.remove(query_id)
                if not requested:
                    return
    if requested:
        raise ValueError(f"Unknown frozen query IDs: {', '.join(sorted(requested))}")


def main():
    manifest = freeze()
    print(json.dumps(manifest["cohorts"], indent=2))


if __name__ == "__main__":
    main()
