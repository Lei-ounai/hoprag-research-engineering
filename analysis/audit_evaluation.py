"""Audit the original document-level evaluation using saved local artifacts."""

import json
import math
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from hoprag.dataset import load_dataset
from hoprag.evaluation import evaluate
from hoprag.models import RetrievalResult


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "results" / "baseline"
RAW = ROOT / "data" / "raw" / "multihop_rag"
METHODS = ("bm25", "dense", "hybrid")


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def _literal(text):
    return text.replace(" ", "").replace("\n", "")


def _no_unicode_whitespace(text):
    return "".join(character for character in text if not character.isspace())


def _no_punctuation(text):
    return "".join(character for character in text if not unicodedata.category(character).startswith("P"))


def _audit_chunks(documents, max_bytes):
    parts = defaultdict(list)
    seen = set()
    errors = []
    for row in _jsonl(BASELINE / "chunks.jsonl"):
        chunk_id, doc_id, content = row["chunk_id"], row["document_id"], row["text"]
        if doc_id not in documents:
            errors.append({"chunk_id": chunk_id, "issue": "unknown_parent"})
        if chunk_id in seen or chunk_id != f"{doc_id}::{len(parts[doc_id]):04d}":
            errors.append({"chunk_id": chunk_id, "issue": "invalid_or_duplicate_chunk_id"})
        if len(content.encode("utf-8")) > max_bytes:
            errors.append({"chunk_id": chunk_id, "issue": "over_configured_limit"})
        seen.add(chunk_id)
        parts[doc_id].append(content)
    incomplete = [
        doc_id for doc_id, doc in documents.items()
        if "".join(parts.get(doc_id, ())) != f"{doc.title}\n{doc.text}"
    ]
    return {
        "documents": len(documents),
        "parents_in_saved_chunks": len(parts),
        "unique_chunk_ids": len(seen),
        "chunk_rows": sum(map(len, parts.values())),
        "chunk_errors": errors,
        "non_reconstructed_document_ids": incomplete,
    }, {doc_id: tuple(map(_literal, chunks)) for doc_id, chunks in parts.items()}


def _audit_evidence(queries, documents, literal_bodies, normalized_chunks):
    document_rows = _json(RAW / "corpus.json")
    query_rows = _json(RAW / "MultiHopRAG.json")
    evidence_rows = [record for query in query_rows for record in query["evidence_list"]]
    ids_by_url = defaultdict(set)
    for doc_id, doc in documents.items():
        ids_by_url[doc.metadata["url"]].add(doc_id)
    broken_mapping = []
    missing_literal = []
    boundaries = []
    alt_documents = {}
    alternates = []
    multi_fact_queries = []
    duplicate_pairs = 0
    duplicate_literals = 0
    for query in queries:
        sources = Counter()
        pairs = set()
        literals = set()
        for evidence in query.evidence_list:
            expected = evidence.source_document_id
            fact = _literal(evidence.evidence_text)
            record = {"query_id": query.query_id, "evidence_id": evidence.evidence_id, "source_document_id": expected}
            duplicate_pairs += (expected, fact) in pairs
            duplicate_literals += fact in literals
            pairs.add((expected, fact))
            literals.add(fact)
            if expected is None or ids_by_url[evidence.metadata["url"]] != {expected}:
                broken_mapping.append(record)
                continue
            sources[expected] += 1
            if fact not in literal_bodies[expected]:
                missing_literal.append(record)
                continue
            if not any(fact in chunk for chunk in normalized_chunks.get(expected, ())):
                boundaries.append(record)
            if fact not in alt_documents:
                alt_documents[fact] = {doc_id for doc_id, text in literal_bodies.items() if fact in text}
            other_docs = alt_documents[fact] - {expected}
            if other_docs:
                alternates.append({**record, "other_document_ids": sorted(other_docs)})
        if any(count > 1 for count in sources.values()):
            multi_fact_queries.append(query.query_id)

    return {
        "native_document_id_fields": sum("id" in row or "document_id" in row for row in document_rows),
        "native_query_id_fields": sum("id" in row or "query_id" in row for row in query_rows),
        "native_evidence_source_id_fields": sum(
            "source_document_id" in row or "document_id" in row for row in evidence_rows
        ),
        "raw_evidence_entries": len(evidence_rows),
        "within_query_duplicate_source_fact_pairs": duplicate_pairs,
        "within_query_duplicate_normalized_facts": duplicate_literals,
        "unresolved_or_nonunique_source_urls": broken_mapping,
        "normalized_fact_missing_from_origin": missing_literal,
        "unique_normalized_facts": len(alt_documents),
        "unique_facts_with_cross_document_literal_matches": sum(len(ids) > 1 for ids in alt_documents.values()),
        "multiple_facts_per_source_query_ids": multi_fact_queries,
        "cross_chunk_fact_entries": boundaries,
        "cross_chunk_fact_query_count": len({row["query_id"] for row in boundaries}),
        "cross_document_literal_fact_entries": alternates,
    }


def _rankings_audit(queries, documents, bodies):
    query_index = {q.query_id: q for q in queries}
    results = {method: {} for method in METHODS}
    depths = {method: Counter() for method in METHODS}
    text_only_hits = {method: [] for method in METHODS}
    source_only_hits = {method: [] for method in METHODS}
    multi_literal_top1_queries = {method: [] for method in METHODS}
    variants = {"casefold": str.casefold, "unicode_whitespace": _no_unicode_whitespace, "punctuation": _no_punctuation}
    variant_bodies = {
        label: {doc_id: transform(body) for doc_id, body in bodies.items()}
        for label, transform in variants.items()
    }
    variant_extra_hits = {label: [] for label in variants}

    for row in _jsonl(BASELINE / "rankings_full.jsonl"):
        query = query_index[row["query_id"]]
        for method in METHODS:
            ranked = row[method]
            ids = [item["document_id"] for item in ranked]
            if len(ids) != len(set(ids)) or any(
                item["query_id"] != query.query_id or item["rank"] != rank or doc_id not in documents
                for rank, (item, doc_id) in enumerate(zip(ranked, ids, strict=True), start=1)
            ):
                raise ValueError(f"Invalid saved rankings for {query.query_id}/{method}")
            if query.query_id in results[method]:
                raise ValueError(f"Duplicate ranking row for {query.query_id}/{method}")
            depths[method][len(ranked)] += 1
            results[method][query.query_id] = [RetrievalResult(**item) for item in ranked[:10]]
            top10 = ids[:10]
            if not top10:
                continue
            if len({_literal(e.evidence_text) for e in query.evidence_list if _literal(e.evidence_text) in bodies[top10[0]]}) > 1:
                multi_literal_top1_queries[method].append(query.query_id)
            for evidence in query.evidence_list:
                fact = _literal(evidence.evidence_text)
                source_hit = evidence.source_document_id in top10
                literal_id = next((doc_id for doc_id in top10 if fact in bodies[doc_id]), None)
                detail = {
                    "query_id": query.query_id, "evidence_id": evidence.evidence_id,
                    "source_document_id": evidence.source_document_id,
                }
                if source_hit and literal_id is None:
                    source_only_hits[method].append(detail)
                if literal_id is not None and not source_hit:
                    text_only_hits[method].append({**detail, "matched_document_id": literal_id})
                if method == "hybrid" and literal_id is None:
                    for label, transform in variants.items():
                        relaxed_fact = transform(fact)
                        if relaxed_fact and any(relaxed_fact in variant_bodies[label][doc_id] for doc_id in top10):
                            variant_extra_hits[label].append(detail)

    stored = _json(BASELINE / "metrics_full.json")["metrics"]
    metric_checks = {}
    for method, ranking in results.items():
        if set(ranking) != set(query_index):
            raise ValueError(f"Saved {method} query IDs do not match the dataset")
        computed = evaluate(queries, documents.values(), ranking)
        expected = stored[method]
        for group in ("official", "custom"):
            if computed[group].keys() != expected[group].keys() or any(
                not math.isclose(value, expected[group][metric], rel_tol=0, abs_tol=1e-12)
                for metric, value in computed[group].items()
            ):
                raise ValueError(f"Baseline metrics differ for {method}/{group}")
        if any(computed[key] != expected[key] for key in ("unanswerable", "answerable_queries")):
            raise ValueError(f"Baseline answerable/unanswerable denominators differ for {method}")
        metric_checks[method] = computed

    return {
        "depth_histograms": depths,
        "recomputed_baseline_metrics": metric_checks,
        "text_hit_without_source_at_10": text_only_hits,
        "source_hit_without_text_at_10": source_only_hits,
        "queries_with_multiple_literal_facts_at_rank1": multi_literal_top1_queries,
        "rrf_top10_extra_literal_matches_by_alternative_normalization": variant_extra_hits,
    }


def build_audit():
    dataset = load_dataset(RAW)
    documents = {doc.document_id: doc for doc in dataset.documents}
    bodies = {doc_id: _literal(doc.text) for doc_id, doc in documents.items()}
    config = _json(BASELINE / "run.json")["config"]
    chunk_stats, normalized_chunks = _audit_chunks(documents, config["max_chunk_bytes"])
    return {
        "answerable_queries": sum(query.is_answerable for query in dataset.queries),
        "unanswerable_queries": sum(not query.is_answerable for query in dataset.queries),
        "chunks": chunk_stats,
        "evidence": _audit_evidence(dataset.queries, documents, bodies, normalized_chunks),
        "rankings": _rankings_audit(dataset.queries, documents, bodies),
    }


def _write_report(path, audit):
    chunk = audit["chunks"]
    evidence = audit["evidence"]
    rank = audit["rankings"]
    baseline = rank["recomputed_baseline_metrics"]
    rows = []
    for method, display in (("bm25", "BM25"), ("dense", "Dense"), ("hybrid", "RRF")):
        official, custom = baseline[method]["official"], baseline[method]["custom"]
        values = [official[key] for key in ("Hits@4", "Hits@10", "MAP@10", "MRR@10")]
        values += [custom[key] for key in (
            "Evidence Recall@4", "Evidence Recall@10", "Complete Evidence Coverage@4",
            "Complete Evidence Coverage@10",
        )]
        rows.append(f"| {display} | " + " | ".join(f"{value:.4f}" for value in values) + " |")
    table = "\n".join(rows)
    hits = rank["text_hit_without_source_at_10"]
    source_only = rank["source_hit_without_text_at_10"]
    alt_norm = rank["rrf_top10_extra_literal_matches_by_alternative_normalization"]
    multi_ap = rank["queries_with_multiple_literal_facts_at_rank1"]
    report = rf"""# HopRAG 第四轮 · 评测体系审计

## 结论

对原始全量排名按现有评测代码重算，三种方法的第三轮指标和分母均能复现。已核对全部来源文档及分块映射，未发现应当修正第三轮指标的错误。评测衡量的是**唯一来源文档级别的证据覆盖**。Dense 证据即使跨越分块边界，来源文档进入 Top-K 后仍按原口径计入覆盖。

## Top-K 的评测单位

- BM25 在整篇标题加正文的索引上排序。Dense 先为每块计算余弦分数，再取同一来源文档的最大块分，每篇只出现在文档排名中一次。RRF 按文档 ID 融合，每个分支按该 ID 的最佳名次贡献一次。现有评测器还会拒绝同一排名中重复的文档 ID。§hoprag/retrievers.py:37–74, 125–152, 175–192§，§hoprag/evaluation.py:64–78§。
- 原始全量排名保存的 BM25 和 Dense 深度为 100，RRF 深度为 10。第三轮使用前两种方法的 Top 100 作为 RRF 输入。第四轮延长 RRF 输出时，应复用这些固定候选与原融合参数，以旧 Top 10 作为回归对照。§hoprag/benchmark.py:61–89§，§results/baseline/run.json§。
- 官方仓库的 BM25 基线以句子切分器产生的 Chunk 排名。官方评分程序直接使用传入排名中的文本条目，未聚合父文档。本文复现其评分公式，检索单位仍沿用第三轮的整篇文档，不能直接与官方 Chunk 级得分比较。 [官方 BM25 脚本](https://github.com/yixuantt/MultiHop-RAG/blob/main/bm25_retrieval.py) · [官方评分脚本](https://github.com/yixuantt/MultiHop-RAG/blob/main/retrieval_evaluate.py)

## 两套评分口径

| 指标 | 计算方式 |
| --- | --- |
| 官方兼容 Hits@K | 一道题在前 K 个检索文本中至少命中一条必要事实则记 1，再对可回答题取均值。官方脚本报告 K=4、10。 |
| 官方兼容 MRR@10 | 取首次命中任一必要事实的排名倒数。前 10 无命中时记 0。 |
| 官方兼容 MAP@10 | 按排名累加「该排名首次命中的不同事实数 ÷ 排名」，每题再除以 §min(原始必要事实条数,10)§，最后取均值。 |
| 自定义 Evidence Recall@K | 对每题统计命中的必要事实数除以该题必要事实数，再对可回答题取均值。以不同的 §(来源文档 ID, 归一化事实文本)§ 为事实单位。 |
| 自定义 Complete Evidence Coverage@K | 每题必须命中其全部必要事实才记 1，再对可回答题取均值。必要事实可以同时来自同一篇来源文档。 |

官方兼容匹配以检索文档的**正文**为输入，删除 ASCII 空格与 LF 换行后做区分大小写的字面子串匹配。它保留标点、制表符、CR 与其他 Unicode 空白。自定义指标在来源已解析时用来源文档 ID 命中事实。来源无法解析时才回退到同一子串规则，本数据集没有触发回退。§hoprag/evaluation.py:16–61, 81–146§。 [官方评分脚本](https://github.com/yixuantt/MultiHop-RAG/blob/main/retrieval_evaluate.py)

公式扩展到其他 K 时需注明是诊断口径。尤其是将官方 MAP 分母机械扩为 §min(事实数,K)§，若排名第 1 的一篇文章命中多条事实，单题 MAP@1 会超过 1。保存排名中符合该条件的问题数分别为 BM25 {len(multi_ap["bm25"])}、Dense {len(multi_ap["dense"])}、RRF {len(multi_ap["hybrid"])}。官方脚本只定义 MAP@10。该观察不影响第三轮任何已保存指标。

官方兼容和自定义指标均排除 §null_query§，第三轮全量评测分母为 {audit["answerable_queries"]:,}。{audit["unanswerable_queries"]} 道无答案题独立统计返回率和平均文档数。返回文档只表示检索产生候选，不能据此判定可回答。§hoprag/evaluation.py:110–146§。

## 来源与分块验证

- 原始文章、问题及必要事实均无原生 ID。统一模型按原始行号生成文档和问题 ID，并从每条事实的原始 URL 唯一映射来源文档。核验 {evidence["raw_evidence_entries"]:,} 条必要事实，来源 URL 未解析或不唯一 {len(evidence["unresolved_or_nonunique_source_urls"])} 条，字面归一化后在标注来源正文中找不到的事实 {len(evidence["normalized_fact_missing_from_origin"])} 条。§hoprag/dataset.py:84–156§。
- 题内重复的 §(来源文档 ID, 归一化事实)§ 有 {evidence["within_query_duplicate_source_fact_pairs"]} 条。{len(evidence["multiple_facts_per_source_query_ids"])} 道可回答题在同一来源文档上标注了两条或更多必要事实。事实召回的分母以事实数计算，单篇来源命中可能覆盖其中多条。
- 保存的 {chunk["chunk_rows"]:,} 块对应 {chunk["parents_in_saved_chunks"]} 篇来源文档，逐文档按序拼接可复原完整标题加正文。Chunk ID、父文档和长度异常共 {len(chunk["chunk_errors"])} 条，无法复原的文档 {len(chunk["non_reconstructed_document_ids"])} 篇。Dense 分块按 UTF-8 字节数受限，输入嵌入时设置 §truncate=false§。§hoprag/retrievers.py:78–99, 119–132§，§hoprag/embeddings.py:92–106§。
- {len(evidence["cross_chunk_fact_entries"])} 条事实（{evidence["cross_chunk_fact_query_count"]} 题）存在于来源整篇正文，但任何一个单块均不含该事实的完整字面文本。这是可验证的切分现象。块到文档的映射仍然完整，无法单凭该现象判定它造成了某次 Dense 排名下降。
- {evidence["unique_normalized_facts"]} 条不同的归一化事实中，{evidence["unique_facts_with_cross_document_literal_matches"]} 条在多个文档中出现，对应 {len(evidence["cross_document_literal_fact_entries"])} 条问题证据记录。Top 10 中有字面命中但标注来源缺席的事实数分别为 BM25 {len(hits["bm25"])}、Dense {len(hits["dense"])}、RRF {len(hits["hybrid"])}。反向差异分别为 {len(source_only["bm25"])}、{len(source_only["dense"])}、{len(source_only["hybrid"])} 条。相同字面文本可以影响官方兼容指标，但不能在缺少来源或上下文核验时作为修改自定义覆盖率的依据。
- 对 RRF Top 10 逐项放宽字面规则，用大小写折叠、删除全部 Unicode 空白、删除 Unicode 标点分别探测到 {len(alt_norm["casefold"])}、{len(alt_norm["unicode_whitespace"])}、{len(alt_norm["punctuation"])} 条额外字面匹配。这些对照只用来验证匹配敏感性，原评分未使用放宽规则。

## 第三轮指标重算

| 方法 | Hits@4 | Hits@10 | MAP@10 | MRR@10 | 证据召回@4 | 证据召回@10 | 完整覆盖@4 | 完整覆盖@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{table}

重算使用原始 §rankings_full.jsonl§、§chunks.jsonl§ 和两个原始 JSON，并与 §metrics_full.json§ 逐指标比较。未写入原实验产物。 [审计数据](evaluation_audit_details.json)

在正式项目根目录执行以下命令可重新生成审计报告和明细。

§§§powershell
& .\.venv\Scripts\python.exe -B -m analysis.audit_evaluation
§§§
"""
    path.write_text(report.replace("§", chr(96)), encoding="utf-8")


def main():
    audit = build_audit()
    output = Path(__file__).resolve().parent
    (output / "evaluation_audit_details.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    _write_report(output / "EVALUATION_AUDIT.md", audit)
    print("Evaluation audit complete. All saved baseline metrics reproduced.")


if __name__ == "__main__":
    main()
