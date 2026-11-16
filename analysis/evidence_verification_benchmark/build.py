"""Reconstruct round-nine pair inputs and produce blinded annotation packets."""

from collections import Counter
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
import csv
import json
import re

from analysis.besa_v0.experiment import MODEL_SPEC, read, rows, write, write_rows
from analysis.ege_v0.run import fingerprint
from analysis.egr_v0.run import OUT as EGR, ROOT, RUN, documents_and_chunks
from analysis.reranking.core import TEMPLATE, make_pairs
from hoprag.evaluation import _normalize
from hoprag.retrievers import _tokens

from .sampling import draw


OUT = Path(__file__).resolve().parent
SEED = 20261115
LABELS = {
    "evidence_support": ("DIRECT_SUPPORT", "PARTIAL_SUPPORT", "NO_SUPPORT", "CONTRADICTION", "UNCERTAIN"),
    "entity_role": ("MATCH", "MISMATCH", "NOT_APPLICABLE", "UNCERTAIN"),
    "relation_support": ("SUPPORTED", "NEGATED", "HEDGED", "UNRELATED", "UNCERTAIN"),
    "temporal_scope": ("CONSISTENT", "CONFLICT", "UNKNOWN", "NOT_APPLICABLE"),
    "temporal_kind": ("EVENT", "PUBLICATION", "TENURE", "FOUNDING", "MULTIPLE", "UNKNOWN", "NOT_APPLICABLE"),
    "evidence_novelty": ("NOVEL", "REDUNDANT", "PARTIAL_NOVELTY", "UNCERTAIN"),
    "replacement_suitability": ("REPLACE", "KEEP_INCUMBENT", "EITHER", "UNCERTAIN"),
}
FREE_FIELDS = ("reason", "witness_ref", "witness_quote")
METADATA = ("source", "published_at", "url")
DATES = re.compile(r"\b(?:19|20)\d{2}(?:-\d{2}-\d{2})?\b|"
                   r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
                   r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|"
                   r"Dec(?:ember)?)\.?\s+\d{1,2}(?:,\s*(?:19|20)\d{2})?\b", re.I)
WORDS_STOP = frozenset("a the does do did what who was it and or that of from is as to were on for be by are in with".split())
PUBLIC_KEYS = {"case_id", "question", "challenger", "comparison_target", "baseline_top10"}


def tokenizer_pair():
    # Match the historical runtime's model tokenizer without opening ONNX.
    from fastembed.common.preprocessor_utils import load_tokenizer
    from tokenizers import Tokenizer

    spec = read(MODEL_SPEC)
    snapshot_dir = Path(spec["model_path"]).parents[1]
    tokenizer_path = snapshot_dir / "tokenizer.json"
    if fingerprint(tokenizer_path)["sha256"] != spec["tokenizer_sha256"]:
        raise ValueError("Historical tokenizer checksum mismatch")
    tokenizer, _ = load_tokenizer(snapshot_dir)
    tokenizer.enable_truncation(spec["effective_max_tokens"], strategy=spec["truncation_strategy"],
                                direction=spec["truncation_side"], stride=0)
    tokenizer.no_padding()
    full = Tokenizer.from_str(tokenizer.to_str())
    full.no_truncation()
    return tokenizer, full, spec


def restore(question, pair, original, score, tokenizer, full):
    if not pair.text.endswith(original.text):
        raise ValueError(f"Template/original Chunk mismatch at {pair.chunk_id}")
    untruncated = full.encode(question, pair.text)
    encoded = tokenizer.encode(question, pair.text)
    if (score["input_tokens"] != len(encoded.ids)
            or score["original_tokens"] != len(untruncated.ids)
            or score["truncated"] != (len(encoded.ids) < len(untruncated.ids))):
        raise ValueError(f"Historical token lengths mismatch at {pair.chunk_id}")
    endpoints = [0, 0]
    for sequence, (start, end) in zip(encoded.sequence_ids, encoded.offsets, strict=True):
        if sequence in (0, 1) and end > start:
            endpoints[sequence] = max(endpoints[sequence], end)
    if not score["truncated"]:
        endpoints = [len(question), len(pair.text)]
    query_end, document_end = endpoints
    prepend = len(pair.text) - len(original.text)
    chunk_end = max(0, document_end - prepend)
    if (score["model_visible_query_chars"], score["model_visible_chunk_chars"]) != (query_end, chunk_end):
        raise ValueError(f"Historical text offset mismatch at {pair.chunk_id}")
    return {
        "original_chunk_text": original.text,
        "jina_query_before_truncation": question,
        "jina_document_before_truncation": pair.text,
        "jina_visible_query_text": question[:query_end],
        "jina_visible_document_text": pair.text[:document_end],
        "jina_visible_chunk_text": original.text[:chunk_end],
        "original_tokens": len(untruncated.ids), "input_tokens": len(encoded.ids),
        "is_truncated": score["truncated"], "model_visible_chunk_chars": chunk_end,
        "model_visible_query_chars": query_end, "model_visible_pair_document_chars": document_end,
        "token_ids_unpadded": encoded.ids, "token_type_ids_unpadded": encoded.type_ids,
        "sequence_ids": encoded.sequence_ids, "offsets": encoded.offsets,
        "attention_mask_unpadded": encoded.attention_mask,
    }


def surface_dates(text):
    return list(dict.fromkeys(m.group() for m in DATES.finditer(text)))


def tokens(text):
    return set(_tokens(text)) - WORDS_STOP


def evidence_excerpt(text, question, limit=530):
    # Question overlap, independent of Gate scores and Gold; never mixes disjoint spans.
    if len(text) <= limit:
        return text
    anchors = tokens(question)
    boundaries = [0, *(m.end() for m in re.finditer(r"(?<=[.!?])\s+|\n", text))]
    windows = [text[i:i + limit] for i in boundaries if i < len(text) - 40]
    return max(windows, key=lambda value: len(tokens(value) & anchors)).strip()


def comparison_id(case, score):
    gates = case["gate_decisions"]
    # Preserve an actual compared slot where archived; expose the same neutral field for every case.
    for method in ("E4-B1", "E3-B1"):
        decision = gates.get(method)
        if decision and decision.get("incumbent_document_id") in score["baseline"][7:10]:
            return decision["incumbent_document_id"], "historical_actual_comparison"
    e2 = case.get("e2_replacement") or {}
    if e2.get("incumbent_document_id") in score["baseline"][7:10]:
        return e2["incumbent_document_id"], "historical_ungated_comparison"
    baseline_tail = score["baseline"][7:10]
    actual = score["scoring"]["documents"]
    incumbent = min(enumerate(baseline_tail), key=lambda item: (
        max(x["jina_score"] for x in actual[item[1]]), -item[0], item[1]
    ))[1]
    return incumbent, "neutral_weakest_fair_incumbent"


def metadata(doc):
    return {key: doc.metadata.get(key) for key in METADATA}


def public_document(doc, blocks, prefix):
    return {
        "title": doc.title, **metadata(doc),
        "scored_chunks": [{
            "ref": f"{prefix}{i}", "jina_visible_text": block["jina_visible_chunk_text"],
            "original_full_text": block["original_chunk_text"],
            "truncated": block["is_truncated"],
            "date_strings_in_scored_window": surface_dates(block["jina_visible_chunk_text"]),
        } for i, block in enumerate(blocks, start=1)],
    }


def human_record(case, question, challenger, target, baselines):
    # Build from an explicit allowlist; case.selection fields are intentionally inaccessible here.
    return {"case_id": case["case_id"], "question": question, "challenger": challenger,
            "comparison_target": target, "baseline_top10": baselines}


def gate_prediction(decision, representative):
    if not representative:
        result = {"entity_role": None, "relation_support": None, "temporal_scope": None}
    else:
        features = representative["features"]
        reasons = set(representative["reasons"])
        roles = set(features.get("requested_roles", ()))
        relations = features.get("requested_relations", ())
        dates = set(features.get("date_status", ()))
        result = {
            "entity_role": ("MISMATCH" if "entity_role_conflict" in reasons else
                            "MATCH" if roles and roles <= set(representative["entity_roles"]) else None),
            "relation_support": ("UNRELATED" if "relation_mismatch" in reasons else
                                 "NEGATED" if "polarity_conflict" in reasons else
                                 "SUPPORTED" if relations and representative["status"] == "supported" else None),
            "temporal_scope": ("CONFLICT" if "date_conflict" in dates else
                               "CONSISTENT" if dates == {"matched"} else None),
        }
    novel = decision.get("novelty_pass")
    result["evidence_novelty"] = ("NOVEL" if novel is True else
                                  "REDUNDANT" if novel is False else None)
    return result


def strongest(blocks, decision):
    chosen = decision.get("champion_chunk_id")
    if chosen:
        existing = next((w for w in blocks if w["chunk_id"] == chosen), None)
        if existing:
            return existing
    order = {"supported": 2, "conflict": 1, "uncertain": 0}
    return max(blocks, key=lambda w: (order[w["status"]], w["quality"], w["original_question_match"],
                                      w["jina_score"], w["chunk_id"]), default=None)


def source_diagnostics():
    # All annotations live exclusively on this offline branch.
    from hoprag.dataset import load_dataset
    from analysis.ege_v0.run import cohorts

    train, validation, _ = cohorts()
    data = load_dataset(ROOT / "data" / "raw" / "multihop_rag")
    return {q.query_id: q.evidence_list for q in data.queries if q.query_id in train | validation}


def facts_in_view(chunks, doc, blocks, evidence):
    result = []
    for fact in evidence:
        if fact.source_document_id != doc:
            continue
        literal = _normalize(fact.evidence_text)
        result.append({
            "fact_id": fact.evidence_id, "source_document_id": doc,
            "in_any_original_chunk": any(literal in _normalize(c.text) for c in chunks[doc]),
            "in_reviewed_full_chunk": any(literal in _normalize(b["original_chunk_text"]) for b in blocks),
            "in_jina_visible_chunk": any(literal in _normalize(b["jina_visible_chunk_text"]) for b in blocks),
        })
    return result


def record_documents(docs, chunks):
    parents = [SimpleNamespace(document_id=key, title=doc.title, chunks=chunks[key]) for key, doc in docs.items()]
    pairs = {p.chunk_id: p for p in make_pairs(SimpleNamespace(candidates=parents))}
    original = {c.chunk_id: c for values in chunks.values() for c in values}
    return pairs, original


def views_for(record, docs, pairs, originals, tokenizer, full, wanted):
    output = {}
    for doc in wanted:
        if doc not in record["scoring"]["documents"] or doc not in docs:
            raise ValueError(f"Sampled parent not scored: {record['query_id']}/{doc}")
        blocks = record["scoring"]["documents"][doc]
        output[doc] = [
            {"chunk_id": item["chunk_id"], "selection_route": item["selection_route"],
             "archived_jina_score": item["jina_score"],
             **restore(record["question"], pairs[item["chunk_id"]], originals[item["chunk_id"]],
                       item, tokenizer, full)}
            for item in blocks
        ]
    return output


def packet_rows(selected, docs, chunks, tokenizer, full):
    pairs, originals = record_documents(docs, chunks)
    grouped = {}
    for c in selected:
        grouped.setdefault(c["query_id"], []).append(c)
    gold = source_diagnostics()
    public, internal, trace, stage_trace, gold_trace = {}, [], [], [], []
    for stage in ("development", "null"):
        scores = rows(RUN / f"{stage}_scores.jsonl")
        witnesses = rows(RUN / f"{stage}_witnesses.jsonl")
        for score, ws in zip(scores, witnesses, strict=True):
            qid = score["query_id"]
            if qid != ws["query_id"]:
                raise ValueError("Mismatched historical score/witness question")
            if qid not in grouped:
                continue
            cases = grouped[qid]
            ids = set(score["baseline"][:10]) | {c["document_id"] for c in cases}
            question = score["question"]
            views = views_for(score, docs, pairs, originals, tokenizer, full, ids)
            context = []
            for rank, doc in enumerate(score["baseline"][:10], 1):
                best = max(views[doc], key=lambda block: (len(tokens(question) & tokens(block["jina_visible_chunk_text"])),
                                                         block["chunk_id"]))
                context.append((doc, best, {
                    "rank": rank, "ref": f"B{rank}", "title": docs[doc].title, **metadata(docs[doc]),
                    "jina_visible_excerpt": evidence_excerpt(best["jina_visible_chunk_text"], question),
                }))
            for case in cases:
                doc = case["document_id"]
                cid = case["case_id"]
                incumbent, how = comparison_id(case, score)
                challenger = public_document(docs[doc], views[doc], "C")
                target = public_document(docs[incumbent], views[incumbent], "I")
                target["baseline_rank"] = score["baseline"].index(incumbent) + 1
                public[cid] = human_record(case, question, challenger, target, [c[2] for c in context])
                internal.append({
                    k: case[k] for k in ("case_id", "query_id", "document_id", "cohort", "split", "stratum",
                                        "overlapping_tags", "inclusion_probability", "sample_weight")
                })
                groups = [("challenger", doc, views[doc]), ("comparison_target", incumbent, views[incumbent])]
                for role, document, blocks in groups:
                    marker = "C" if role == "challenger" else "I"
                    trace.extend({"case_id": cid, "query_id": qid, "document_id": document, "role": role,
                                  "ref": f"{marker}{i}", "template": TEMPLATE,
                                  **metadata(docs[document]), **b}
                                 for i, b in enumerate(blocks, 1))
                for document, block, passage in context:
                    trace.append({"case_id": cid, "query_id": qid, "document_id": document,
                                  "role": "baseline_context", "ref": passage["ref"], "template": TEMPLATE,
                                  **metadata(docs[document]), **block})
                decision = case["gate_decisions"]["E4-B1"]
                w = ws["witnesses"][doc]
                stage_trace.append({
                    "case_id": cid, "query_id": qid, "document_id": doc, "comparison_target": incumbent,
                    "comparison_target_policy": how, "candidate_archive": score["candidates"][doc],
                    "fixed_gate": decision, "gate_by_arm": case["gate_decisions"],
                    "e2_replacement": case["e2_replacement"], "block_verdicts": w,
                    "rule_dimension_predictions": gate_prediction(decision, strongest(w, decision)),
                })
                gold_trace.append({"case_id": cid, "query_id": qid, "document_id": doc,
                                   "known_necessary_source": case["gold_source_for_diagnostics"],
                                   "facts": facts_in_view(chunks, doc, views[doc], gold.get(qid, ()))})
    if set(public) != {row["case_id"] for row in selected} or len(public) != len(selected):
        raise ValueError("Some selected rows could not be reproduced")
    return [public[c["case_id"]] for c in selected], internal, trace, stage_trace, gold_trace


def csv_records(path, data, fields=None):
    data = list(data)
    columns = fields or list(dict.fromkeys(k for row in data for k in row))
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(data)


def preserve_template(path, cases):
    ids = [c["case_id"] for c in cases]
    fields = ["case_id", *LABELS, *FREE_FIELDS]
    if path.exists():
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            saved = list(reader)
        if reader.fieldnames != fields or [r["case_id"] for r in saved] != ids:
            raise ValueError("Existing annotation template belongs to different sample")
        if any(any(r[k].strip() for k in fields[1:]) for r in saved):
            return "kept_existing_human_entries"
    csv_records(path, ({"case_id": cid} for cid in ids), fields)
    return "blank_template"


def code_fence(text):
    # Some source articles contain backticks; choose a longer fence when needed.
    width = max((len(m.group()) for m in re.finditer(r"~{3,}", text)), default=2) + 1
    fence = "~" * width
    return f"{fence}text\n{text}\n{fence}\n"


def document_blocks(lines, document):
    lines.extend([f"**{document['title']}**", "",
                  f"媒体 {document['source'] or '未知'} · 发布 {document['published_at'] or '未知'}", "",
                  f"原报道 {document['url'] or '未提供'}", ""])
    for block in document["scored_chunks"]:
        visibility = "模型可见部分，已截断" if block["truncated"] else "模型可见全文"
        dates = "、".join(block["date_strings_in_scored_window"]) or "无可见日期字面"
        lines.extend([f"**{block['ref']} · {visibility}**", "", f"日期字面 {dates}。具体日期作用域待审查。", "",
                      code_fence(block["jina_visible_text"] or "（无可见正文）")])
        if block["truncated"]:
            lines.extend(["<details><summary>完整原始 Chunk，仅供定位模型窗外的候选事实</summary>", "",
                          code_fence(block["original_full_text"]), "</details>", ""])


def markdown_packets(cases):
    lines = [
        "# 证据核验盲审材料", "",
        "请配合 [标注规范](annotation_guide.md) 和 [标注模板](annotation_template.csv) 使用。"
        "按每个案例的编号填写。所有比较对象均按相同格式呈现。现有前十仅提供片段，信息不足时记录不确定。", "",
    ]
    for case in cases:
        lines.extend([f"## {case['case_id']}", "", case["question"], "", "### 挑战者", ""])
        document_blocks(lines, case["challenger"])
        lines.extend([f"### 比较对象（原第 {case['comparison_target']['baseline_rank']} 位）", ""])
        document_blocks(lines, case["comparison_target"])
        lines.extend(["### 当前十篇文档的相关片段", ""])
        for row in case["baseline_top10"]:
            lines.extend([f"**{row['ref']} · {row['title']}**", "",
                          f"{row['source'] or '来源未知'} · 发布 {row['published_at'] or '未知'}", "",
                          code_fence(row["jina_visible_excerpt"])])
    return "\n".join(lines) + "\n"


def main():
    started = perf_counter()
    selected, strata, design = draw(ROOT, seed=SEED)
    sampled_at = perf_counter()
    docs, chunks = documents_and_chunks()
    tokenizer, full, model = tokenizer_pair()
    cases, key, trace, stages, gold = packet_rows(selected, docs, chunks, tokenizer, full)
    if any(set(r) != PUBLIC_KEYS for r in cases):
        raise ValueError("Unexpected nonblind field")
    write_rows(OUT / "sampled_candidates.jsonl", cases)
    write_rows(OUT / "sampling_key.jsonl", key)
    write_rows(OUT / "source_chunk_trace.jsonl", trace)
    write_rows(OUT / "gate_stage_trace.jsonl", stages)
    write_rows(OUT / "gold_source_diagnostics.jsonl", gold)
    csv_records(OUT / "sampling_statistics.csv", strata)
    annotations = preserve_template(OUT / "annotation_template.csv", cases)
    (OUT / "annotation_packets.md").write_text(markdown_packets(cases), encoding="utf-8", newline="\n")
    write(OUT / "build_profile.json", {
        "seed": SEED, "sampling_seconds": sampled_at - started,
        "packet_and_trace_seconds": perf_counter() - sampled_at,
        "cases": len(cases), "cases_by_cohort": dict(Counter(x["cohort"] for x in key)),
        "checked_pair_views": len(trace),
        "unique_checked_pair_views": len({(x["query_id"], x["chunk_id"]) for x in trace}),
        "model_context_tokens": model["effective_max_tokens"],
        "blind_schema": sorted(PUBLIC_KEYS), "separate_labels": list(LABELS),
        "annotations": annotations, "design": design, "new_jina_pairs": 0, "new_embeddings": 0,
    })
    print(f"{len(cases)} blinded cases, {len(trace)} original and model-facing trace rows", flush=True)


if __name__ == "__main__":
    main()

