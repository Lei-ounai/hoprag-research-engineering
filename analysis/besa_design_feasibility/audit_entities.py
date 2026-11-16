"""Offline sample of entity-like surface forms in frozen development rankings.

The hand-reviewed question mentions are used only for measurement. Candidate
extraction, frequency, and graph construction never see labels or gold facts.
"""
from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import math
import re
import sqlite3
import unicodedata
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from statistics import median

from hoprag.config import BenchmarkConfig
from hoprag.dataset import load_dataset, stratified_sample

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent
RERANK = ROOT / "analysis" / "reranking"
OPTIMIZATION = ROOT / "analysis" / "reranking_optimization"
MONTHS = ("January February March April May June July August September October November December").split()
DATE_RE = re.compile(
    rf"\b({'|'.join(MONTHS)})\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,?\s+(\d{{4}}))?\b",
    re.I,
)
ISO_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
MONTH_NUMBER = {month.casefold(): pos for pos, month in enumerate(MONTHS, 1)}
WORD_RE = re.compile(r"(?u)[^\W_]+(?:(?:[.'\-])[^\W_]+)*")
TYPOGRAPHY = str.maketrans({"’": "'", "‘": "'", "–": "-", "—": "-", "‑": "-"})
BREAK = frozenset("""
a an and according about after against all also any are as at before between both
but by can could did does down during each early for from had has have having here
how in into is it its latest more near new not of on one only or other out over per
reported same since so than that the their these this those through to too under
up use using via was were what when where which while who will with would you your
""".split())
GENERIC = BREAK | frozenset("""
ad ai aimed allegations analysis app article assets based best billion business
case central change chief city claim class collapse company consistent court crypto
cryptocurrency deal default draft dramatic drop early exchange executive event
family fan fantasy fast field figure files final first focus football fraud future
game good great growth help high home hope impact industry inflation initial inside
investor judge key last lawsuit leader leading legal line live loss maker major
market media million mobile model most move news night october office opening
option original party passage penalty personal player policy popular position
press privacy process product public purchase ranking rankings rate real report
reports research revenue review round rules search second secure series service
shares shot side sign site sports staff start state stock strong subject support
take team tech technology top total town trade trial update updates user valley
video vote war week win world year
""".split())
PERSON_PREFIXES = {"judge", "coach", "president", "professor", "dr", "mr", "ms"}
ENTITY_ACRONYMS = {"AI", "ASX", "CEO", "CTO", "CFO", "CPI", "GDP", "GDPR", "IPO", "PPR", "RB", "WR"}
COLLISION_PRONE = {"he", "she", "i", "we", "they", "them", "his", "her", "our", "you", "u s"}
ALIAS_PROBES = {
    "bankman_fried": ("q-0635", ("doc-0011", "doc-0032"), ("Sam Bankman-Fried", "SBF", "Bankman-Fried", "Sam")),
    "sam_collision": ("q-2447", ("doc-0333", "doc-0032"), ("Sam Altman", "Sam Bankman-Fried", "Sam")),
    "fed": ("q-0910", ("doc-0113", "doc-0088"), ("Federal Reserve", "Fed")),
    "nfl": ("q-1827", ("doc-0560", "doc-0296"), ("National Football League", "NFL")),
    "meta_scope": ("q-0064", ("doc-0107",), ("Meta", "Facebook", "Instagram")),
}

# Question-only manual recognition targets. Source outlets are an entity subtype.
# Omitted types are intentionally not imputed from generic spans.
ANNOTATIONS = {
    "q-1320": {"person": ["Jahmyr Gibbs", "Dameon Pierce", "Najee Harris", "Isiah Pacheco", "Breece Hall", "Tyreek Hill"], "outlet": ["CBSSports.com", "Sporting News"]},
    "q-0031": {"outlet": ["TechCrunch", "The Verge"]},
    "q-0064": {"organization": ["EU"], "outlet": ["TechCrunch"]},
    "q-2186": {"company": ["Google"], "other": ["Gemini"], "outlet": ["TechCrunch", "The Verge"]},
    "q-0215": {"person": ["Thomas Ramos", "Tessa Wullaert"], "outlet": ["The Guardian", "Sport Grill"], "date": ["October 6th", "October 31st"]},
    "q-1058": {"person": ["Shohei Ohtani"], "organization": ["MLB"], "outlet": ["Sporting News", "Essentially Sports"], "other": ["Ford family"]},
    "q-2238": {"place": ["Christchurch", "Sydney"], "outlet": ["The Roar | Sports Writers Blog"]},
    "q-1557": {"person": ["Taylor Swift", "Travis Kelce"], "place": ["Arrowhead Stadium"], "outlet": ["CBSSports.com", "The Independent - Life and Style"], "date": ["October 12, 2023", "November 25, 2023", "December 6, 2023"]},
    "q-2447": {"place": ["Silicon Valley"], "company": ["OpenAI"], "outlet": ["Fortune", "TechCrunch"]},
    "q-0911": {"person": ["Warren Buffet"], "company": ["Jane Street"], "outlet": ["Fortune", "TechCrunch"]},
    "q-0635": {"person": ["Sam Bankman-Fried"], "company": ["FTX"], "outlet": ["TechCrunch", "Fortune"], "date": ["October 1, 2023", "October 4, 2023"]},
    "q-1827": {"person": ["Taylor Swift", "Annika"], "organization": ["National Football League", "NFL"], "outlet": ["The Independent - Life and Style", "FOX News - Lifestyle"]},
    "q-2117": {"outlet": ["TechCrunch", "The Verge", "Fortune"]},
    "q-0880": {"outlet": ["The Verge", "TechCrunch"]},
    "q-0133": {"person": ["Kenneth Walker III", "Tee Higgins", "Noah Brown", "Treylon Burks", "Kadarius Toney"], "outlet": ["CBSSports.com", "Sporting News"]},
    "q-0910": {"organization": ["Federal Reserve"], "outlet": ["The Sydney Morning Herald", "Fortune"], "date": ["October 1, 2023", "October 6, 2023"]},
    "q-0409": {"person": ["Sam Bankman-Fried", "Caroline Ellison"], "company": ["Alameda Research"], "outlet": ["Fortune", "TechCrunch"]},
    "q-0998": {"company": ["Amazon"], "organization": ["European Commission"], "outlet": ["TechCrunch"]},
    "q-1817": {"person": ["Sam Bankman-Fried"], "outlet": ["Fortune", "TechCrunch"], "date": ["October 4, 2023", "October 7, 2023"]},
    "q-1648": {"outlet": ["The Verge", "TechCrunch"]},
    "q-0240": {"person": ["Lewis Kaplan"], "company": ["Paradigm"], "outlet": ["Cnbc | World Business News Leader", "The Verge", "TechCrunch"]},
    "q-0600": {"person": ["Dylan Cease", "Corbin Burnes", "Tyler Glasnow", "Mike Trout"], "organization": ["MLB", "Los Angeles Angels"], "outlet": ["Sporting News", "Essentially Sports"]},
    "q-0486": {"company": ["Google"], "outlet": ["TechCrunch", "The Age"]},
    "q-0331": {"person": ["Sam Bankman-Fried", "Caroline Ellison"], "company": ["Alameda Research", "FTX", "Binance"], "outlet": ["Fortune", "The Verge", "TechCrunch"]},
}


def _surface(text):
    return unicodedata.normalize("NFKC", text).translate(TYPOGRAPHY)


def _key(text):
    text = re.sub(r"'s\b", "", _surface(text), flags=re.I)
    return " ".join(re.findall(r"[^\W_]+", text.casefold()))


def _dates(text):
    result = {}
    for match in DATE_RE.finditer(text):
        month, day, year = match.groups()
        day = int(day)
        if 1 <= day <= 31:
            result[f"date:{year or '????'}-{MONTH_NUMBER[month.casefold()]:02d}-{day:02d}"] = match.group()
    for match in ISO_RE.finditer(text):
        year, month, day = map(int, match.groups())
        if 1 <= month <= 12 and 1 <= day <= 31:
            result[f"date:{year:04d}-{month:02d}-{day:02d}"] = match.group()
    return result


def _word_shape(word):
    return word[0].isupper() and _key(word) not in BREAK


def _extract(text, outlets=()):
    text = _surface(text)
    names = _dates(text)
    words = list(WORD_RE.finditer(text))
    for start, first in enumerate(words):
        if not _word_shape(first.group()):
            continue
        chosen = [first]
        for following in words[start + 1 : start + 4]:
            if not text[chosen[-1].end():following.start()].isspace() or not _word_shape(following.group()):
                break
            chosen.append(following)
        for width in range(1, len(chosen) + 1):
            raw = text[first.start():chosen[width - 1].end()]
            key = _key(raw)
            if width == 1 and (key in GENERIC or raw in ENTITY_ACRONYMS or key in MONTH_NUMBER):
                continue
            if width > 1 and (_key(first.group()) in PERSON_PREFIXES or all(_key(w.group()) in GENERIC for w in chosen[:width])):
                continue
            names.setdefault(key, raw)
    for key, matcher in outlets:
        match = matcher.search(text)
        if match:
            names[key] = match.group()
    return names


def _annotated_key(text, kind):
    if kind == "date":
        return next(iter(_dates(text)))
    if kind == "outlet":
        return f"outlet:{_key(text)}"
    return _key(text)


def _read_jsonl(path):
    with path.open(encoding="utf-8") as source:
        for line in source:
            yield json.loads(line)


def _corpus_sources(documents):
    sources = {
        _surface(document.metadata["source"]).strip()
        for document in documents.values()
        if document.metadata.get("source")
    }
    return tuple(
        (f"outlet:{_key(source)}", re.compile(rf"(?<!\w){re.escape(source)}(?!\w)", re.I))
        for source in sorted(sources)
    )


def _runtime_inventory():
    packages = {}
    for package in ("spacy", "stanza", "nltk", "flair", "transformers", "torch", "fastembed", "onnxruntime", "tokenizers"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    sqlite_inventory = {}
    for folder in ("analysis", "data", "results"):
        for path in (ROOT / folder).rglob("*"):
            if path.suffix.lower() not in {".sqlite3", ".sqlite", ".db"} or OUTPUT in path.parents:
                continue
            with sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True) as database:
                tables = database.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
                sqlite_inventory[path.relative_to(ROOT).as_posix()] = [row[0] for row in tables]
    return {
        "packages": packages,
        "installed_small_spacy_models": [
            name for name in ("en_core_web_sm", "en_core_web_md", "en_core_web_lg", "en_core_web_trf")
            if importlib.util.find_spec(name) is not None
        ],
        "sqlite_inventory_read_only": sqlite_inventory,
        "local_model_cache_names": sorted(path.name for path in (RERANK / "model_cache" / "hub").glob("models--*")),
    }


def _alias_probe_table(documents):
    probes = {}
    for label, (query_id, doc_ids, surfaces) in ALIAS_PROBES.items():
        rows = []
        for doc_id in doc_ids:
            document = documents[doc_id]
            forms = set(_extract(f"{document.title}\n{document.text}"))
            rows.append({
                "document_id": doc_id,
                "title": document.title,
                "source": document.metadata["source"],
                "published_at": document.metadata.get("published_at"),
                "forms": {surface: _key(surface) in forms for surface in surfaces},
            })
        probes[label] = {"query_id": query_id, "documents": rows}
    return probes


def _ranked_key_sets(names_by_doc, frequencies, ignored):
    raw = [
        (names_by_doc[a] & names_by_doc[b], a, b)
        for a, b in combinations(names_by_doc, 2)
    ]
    # Use a frequency stratum to show the size of an unverified graph. No graph reranking.
    counts = {
        "potential_pairs": len(raw),
        "any_surface_pair_edges": sum(bool(shared) for shared, _, _ in raw),
        "rare_surface_pair_edges_df_2_to_20": sum(
            any(2 <= frequencies[key] <= 20 for key in shared) for shared, _, _ in raw
        ),
        "strict_multiword_without_outlet_edges": sum(
            any(" " in key and key not in ignored and 2 <= frequencies[key] <= 60 for key in shared)
            for shared, _, _ in raw
        ),
        "high_df_only_pair_edges": sum(
            bool(shared) and all(frequencies[key] >= 60 for key in shared) for shared, _, _ in raw
        ),
    }
    return raw, counts


def main():
    config = BenchmarkConfig()
    dataset = load_dataset(config.raw_directory)
    documents = {document.document_id: document for document in dataset.documents}
    splits = json.loads((RERANK / "splits.json").read_bytes())
    dev = set(splits["answerable_dev"])
    sample = stratified_sample([q for q in dataset.queries if q.query_id in dev], 24, seed=42)
    selected = {query.query_id for query in sample}
    assert selected == ANNOTATIONS.keys()
    saved = {
        row["query_id"]: row
        for row in _read_jsonl(OPTIMIZATION / "development_rankings_answerable_dev.jsonl")
        if row["pool_id"] == "P5" and row["query_id"] in selected
    }
    assert saved.keys() == selected
    chunk_ids = {
        entry["winning_chunk_id"]
        for row in saved.values()
        for entry in row["jina_ranking"]
        if entry["document_id"] in {
            result["document_id"] for result in row["final_top10_by_rrf_weight"]["2"][:5]
        }
    }
    chunks = {
        chunk["chunk_id"]: chunk
        for chunk in _read_jsonl(config.output_directory / "chunks.jsonl")
        if chunk["chunk_id"] in chunk_ids
    }
    assert chunks.keys() == chunk_ids
    outlets = _corpus_sources(documents)
    outlet_keys = {
        form
        for document in documents.values()
        if document.metadata.get("source")
        for form in _extract(document.metadata["source"])
        if form != "business"
    }
    ignored = COLLISION_PRONE | outlet_keys
    by_document = {
        doc_id: set(_extract(f"{doc.title}\n{doc.text}", outlets))
        for doc_id, doc in documents.items()
    }
    frequency = Counter(key for keys in by_document.values() for key in keys)
    normalized_idf = {
        key: math.log((len(documents) + 1) / (df + 1)) + 1
        for key, df in frequency.items()
    }
    by_chunk = {
        chunk_id: _extract(chunk["text"], outlets)
        for chunk_id, chunk in chunks.items()
    }
    detail = []
    annotated = defaultdict(Counter)
    graph = Counter()
    gold_split = Counter()
    anchor_best_delta = Counter()
    weak_keys = Counter()
    pub_sources = Counter()
    query_surface_count = []
    chunk_surface_count = []
    without_named_content = 0
    span_scope = Counter()
    date_metadata = Counter()
    lexical_distractors = Counter()
    for query in sample:
        row = saved[query.query_id]
        query_spans = _extract(query.question, outlets)
        query_surface_count.append(len(query_spans))
        references = []
        for kind, labels in ANNOTATIONS[query.query_id].items():
            for label in labels:
                key = _annotated_key(label, kind)
                recognized = key in query_spans
                references.append({"label": label, "kind": kind, "key": key, "recognized": recognized})
                annotated[kind]["manually_marked"] += 1
                annotated[kind]["recognized"] += recognized
        semantic_mentions = {
            item["key"]
            for item in references
            if item["kind"] in {"person", "company", "organization", "place"}
        }
        named_outlets = {item["key"] for item in references if item["kind"] == "outlet"}
        without_named_content += not semantic_mentions
        gold_sources = {item.source_document_id for item in query.evidence_list}
        pool = {entry["document_id"] for entry in row["jina_ranking"]}
        gold_p5 = gold_sources & pool
        gold_split["questions_all_gold_in_pool"] += gold_p5 == gold_sources
        winners = {entry["document_id"]: entry["winning_chunk_id"] for entry in row["jina_ranking"]}
        candidates = []
        graph_nodes = {}
        normalized_query_keys = set(query_spans)
        anchor_query_keys = {
            key for key in query_spans
            if key not in ignored and not key.startswith(("outlet:", "date:"))
        }
        for candidate in row["final_top10_by_rrf_weight"]["2"][:5]:
            doc_id = candidate["document_id"]
            doc = documents[doc_id]
            chunk = chunks[winners[doc_id]]
            assert chunk["document_id"] == doc_id
            names = by_chunk[chunk["chunk_id"]]
            chunk_surface_count.append(len(names))
            content_names = {key for key in names if not key.startswith(("outlet:", "date:"))}
            graph_nodes[doc_id] = content_names
            anchor_overlap = anchor_query_keys & content_names
            exact_mentions_in_chunk = semantic_mentions & content_names
            exact_mentions_in_parent = semantic_mentions & by_document[doc_id]
            source_is_requested = f"outlet:{_key(doc.metadata['source'])}" in named_outlets
            date_key = f"date:{doc.metadata['published_at'][:10]}"
            date_strict_match = date_key in query_spans
            match_month_day_only = f"date:????{date_key[-6:]}" in query_spans
            is_gold = doc_id in gold_sources
            span_scope["candidate_occurrences_with_reference_anchor_in_parent"] += bool(exact_mentions_in_parent)
            span_scope["candidate_occurrences_with_reference_anchor_in_chunk"] += bool(exact_mentions_in_chunk)
            span_scope["candidate_occurrences_all_parent_anchors_lost_in_chunk"] += bool(exact_mentions_in_parent) and not exact_mentions_in_chunk
            lexical_distractors["non_gold_with_reference_anchor"] += not is_gold and bool(exact_mentions_in_chunk)
            lexical_distractors["non_gold_with_requested_source"] += not is_gold and source_is_requested
            date_metadata["strict_query_date_metadata_matches"] += date_strict_match
            date_metadata["strict_query_date_metadata_matches_non_gold"] += date_strict_match and not is_gold
            date_metadata["month_day_only_metadata_matches"] += match_month_day_only
            pub_sources[doc.metadata["source"]] += 1
            candidates.append({
                "rank": candidate["rank"],
                "document_id": doc_id,
                "title": doc.title,
                "source": doc.metadata["source"],
                "winning_chunk_id": chunk["chunk_id"],
                "gold_source_for_offline_analysis": doc_id in gold_sources,
                "exact_reference_anchors_winning_chunk": sorted(exact_mentions_in_chunk),
                "exact_reference_anchors_parent": sorted(exact_mentions_in_parent),
                "source_is_among_named_outlets": source_is_requested,
                "published_at": doc.metadata["published_at"],
                "exact_requested_date_match": date_strict_match,
                "month_day_only_requested_date_match": match_month_day_only,
                "shared_question_keys": sorted(normalized_query_keys & names.keys()),
                "shared_question_keys_full_document": sorted(normalized_query_keys & by_document[doc_id]),
                "anchor_overlap_count": len(anchor_overlap),
                "anchor_overlap_idf": sum(normalized_idf.get(key, 0.0) for key in sorted(anchor_overlap)),
                "sample_surface_forms_with_df": [
                    {"key": key, "surface": names[key], "document_frequency": frequency[key]}
                    for key in sorted(names, key=lambda key: (frequency[key], key))[:12]
                ],
            })
        pairs, pair_counts = _ranked_key_sets(graph_nodes, frequency, ignored)
        graph.update(pair_counts)
        for shared, left, right in pairs:
            both_gold = left in gold_sources and right in gold_sources
            same_query = shared & anchor_query_keys
            rare_new = {
                key for key in shared - anchor_query_keys if 2 <= frequency[key] <= 20
            }
            tag = "gold_gold" if both_gold else "other"
            graph.update({
                f"{tag}_pairs": 1,
                f"{tag}_pairs_with_query_anchor": bool(same_query),
                f"{tag}_pairs_with_rare_non_question_key": bool(rare_new),
            })
            weak_keys.update(shared)
        count_order = sorted(candidates, key=lambda c: (-c["anchor_overlap_count"], c["rank"]))
        idf_order = sorted(candidates, key=lambda c: (-c["anchor_overlap_idf"], c["rank"]))
        anchor_best_delta["top1_changed"] += count_order[0]["document_id"] != idf_order[0]["document_id"]
        anchor_best_delta["count_top1_gold_source"] += count_order[0]["gold_source_for_offline_analysis"]
        anchor_best_delta["idf_top1_gold_source"] += idf_order[0]["gold_source_for_offline_analysis"]
        gold_split["questions_any_gold_in_sample_top5"] += any(c["gold_source_for_offline_analysis"] for c in candidates)
        gold_split["questions_with_at_least_two_gold_parents_in_sample_top5"] += sum(
            c["gold_source_for_offline_analysis"] for c in candidates
        ) >= 2
        gold_split["candidate_occurrences_with_gold_parent"] += sum(c["gold_source_for_offline_analysis"] for c in candidates)
        detail.append({
            "query_id": query.query_id,
            "type": query.question_type,
            "necessary_facts": len(query.evidence_list),
            "question": query.question,
            "question_surface_forms": query_spans,
            "question_reference_mentions": references,
            "p5_candidate_count": len(pool),
            "p5_gold_parent_count_for_offline_analysis": len(gold_p5),
            "gold_parent_ids_for_offline_analysis": sorted(gold_sources),
            "p5_b_top5": candidates,
            "pair_counts": pair_counts,
            "anchor_count_first_parent": count_order[0]["document_id"],
            "anchor_idf_first_parent": idf_order[0]["document_id"],
        })
    result = {
        "sample": {
            "seed": 42,
            "cohort": "answerable_dev",
            "development_count": len(dev),
            "questions": len(sample),
            "strata": dict(Counter(f"{row['type']}:{row['necessary_facts']}" for row in detail)),
            "candidate_occurrences": sum(len(row["p5_b_top5"]) for row in detail),
            "unique_candidate_parents": len({c["document_id"] for row in detail for c in row["p5_b_top5"]}),
            "unique_winning_chunks": len(chunk_ids),
        },
        "local_environment": _runtime_inventory(),
        "manual_question_mention_recall": dict(annotated),
        "questions_without_explicit_typed_content_mention": without_named_content,
        "reference_anchor_scope": dict(span_scope),
        "lexical_distractors": dict(lexical_distractors),
        "date_metadata_matches": dict(date_metadata),
        "alias_probes": _alias_probe_table(documents),
        "question_surface_form_median": median(query_surface_count),
        "winning_chunk_surface_form_median": median(chunk_surface_count),
        "frequent_pair_keys": [
            {"key": key, "pair_occurrences": count, "corpus_parent_df": frequency[key]}
            for key, count in weak_keys.most_common(20)
        ],
        "frequent_candidate_outlets": dict(pub_sources.most_common(8)),
        "candidate_graph": dict(graph),
        "anchor_count_vs_idf_top1": dict(anchor_best_delta),
        "gold_source_diagnostic": dict(gold_split),
        "questions": detail,
    }
    path = OUTPUT / "entity_sample.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in {"questions", "local_environment"}}, ensure_ascii=True))


if __name__ == "__main__":
    main()
