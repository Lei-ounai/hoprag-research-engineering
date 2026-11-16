"""Offline reclassification of round-seven failures and fixed-depth pool recall."""

from collections import Counter, defaultdict
import csv
from hashlib import file_digest
import json
from pathlib import Path
import re
from statistics import mean
from time import perf_counter

from analysis.besa_v0 import planner
from analysis.besa_v0.experiment import cache_source, development_ids, development_inputs, read, rows, write, write_rows
from analysis.rankings import load_cached_vectors, load_chunks
from analysis.reranking.pool import BASELINE, PROJECT, RAW_DATA, build_candidate_record, read_manifest
from hoprag.dataset import load_dataset
from hoprag.evaluation import _normalize
from hoprag.models import Query, RetrievalResult
from hoprag.retrievers import HybridRetriever


OUT = Path(__file__).resolve().parent
BESA = OUT.parent / "besa_v0"
DEPTHS = (15, 30, 50, 100)
SOURCES = ("P5_CUTOFF", "DEEP_RANKING", "QUERY_MISMATCH", "CORPUS_MISSING", "OTHER", "UNCERTAIN")
PLANNER_EXPLANATIONS = {
    "ambiguous_reference": "指代表达触发前置保守拒绝，规则未尝试解析先行词",
    "no_unambiguous_partition": "已有句式规则未构造唯一且可检索的子句切分",
    "unsupported_question_shape": "句首或问题结构不符合已实现模板",
    "ambiguous_alternative": "候选项或备选表达触发歧义保护，规则未拆分",
    "constraint_validation_failed": "试探切分未通过结构或约束保真门槛",
}
COMPARISON = re.compile(r"\b(?:while|whereas|compared?|comparison|contrast|same|differ\w*|consistent|inconsistent|discrepancy|more|less)\b", re.I)
TEMPORAL = re.compile(r"\b(?:after|before|later|earlier|first|second|subsequent\w*|subsequently|previous\w*|between)\b", re.I)
MONTH = re.compile(r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}(?:,\s*\d{4})?\b", re.I)
REPORTS = re.compile(r"\b(?:published|reported|according to|report|article|articles|news source|outlet)\b", re.I)


def csv_rows(path, records):
    records = list(records)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def sha256(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def verify_inputs(dataset, config):
    manifest = read_manifest()
    for name, expected in manifest["source_artifacts"].items():
        path = PROJECT / name
        if path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
            raise ValueError(f"Historical source changed: {name}")
    docs = {doc.document_id: doc for doc in dataset.documents}
    chunks = load_chunks(BASELINE / "chunks.jsonl", docs, config["max_chunk_bytes"])
    path = cache_source()
    before = sha256(path)
    run = read(BASELINE / "run.json")
    model, dimensions = config["embedding_model"], config["embedding_dimensions"]
    vectors, cache = load_cached_vectors(path, model, dimensions, chunks, dataset.queries)
    if (len(docs), sum(map(len, chunks.values())), len(vectors)) != (609, 2378, 4934):
        raise ValueError("Corpus or historical BGE-M3 vector count does not match")
    if any(cache[field] for field in ("read_error", "missing_corpus_vectors", "missing_query_vectors", "invalid_expected_vectors")):
        raise ValueError(f"Historical embedding cache fails validation: {cache}")
    if before != sha256(path):
        raise ValueError("Read-only validation changed the historical embedding cache")
    reports = (
        OUT.parent / "besa_design_feasibility" / "BESA_FEASIBILITY_AUDIT.md",
        BESA / "BESA_V0_REPORT.md",
    )
    for report in reports:
        if not report.is_file():
            raise FileNotFoundError(report)
    return docs, chunks, {
        "corpus_documents": len(docs),
        "corpus_chunks": sum(map(len, chunks.values())),
        "historical_candidate_artifacts_verified": len(manifest["artifacts"]),
        "historical_source_artifacts_verified": len(manifest["source_artifacts"]),
        "historical_reports_sha256": {p.relative_to(PROJECT).as_posix(): sha256(p) for p in reports},
        "cached_embeddings": cache,
        "embedding_cache_sha256": before,
        "baseline_branch_depth": config["fusion_depth"],
        "historical_full_stage_query_count": run["stages"]["full"]["query_count"],
        "historical_full_stage_ranking_seconds": run["stages"]["full"]["ranking_seconds"],
    }


def replay_original(development, config):
    """Replay RRF from frozen branch ranks for the 451 selected question IDs."""
    stored = {
        row["query_id"]: row
        for row in rows(BASELINE / "rankings_full.jsonl")
        if row["query_id"] in development
    }
    frozen_pools = {
        row["query_id"]: row
        for row in rows(OUT.parent / "reranking" / "candidate_pools.jsonl")
        if row["query_id"] in development
    }
    if set(stored) != development.keys() or frozen_pools.keys() != stored.keys():
        raise ValueError("Original rankings do not cover the complete development split")
    fuser = HybridRetriever(None, None, rrf_k=config["rrf_k"], fusion_depth=config["fusion_depth"])
    for qid, raw in stored.items():
        ranks = {
            method: [RetrievalResult(**row) for row in raw[method]]
            for method in ("bm25", "dense")
        }
        query = Query(qid, development[qid].question, "", "")
        rrf = fuser.fuse(query, ranks["bm25"], ranks["dense"], 2 * config["fusion_depth"])
        if [row.document_id for row in rrf[:10]] != [row["document_id"] for row in raw["hybrid"]]:
            raise ValueError(f"Original RRF Top 10 differs from replay: {qid}")
        ranks["rrf"] = rrf
        replay = build_candidate_record(qid, query.question, ranks)
        if replay != frozen_pools[qid]:
            raise ValueError(f"Frozen branch pool differs from replay: {qid}")
        yield qid, {method: {row.document_id: row.rank for row in ranking} for method, ranking in ranks.items()}


def rank_detail(doc_id, ranks):
    detail = {method + "_rank": ranking.get(doc_id) for method, ranking in ranks.items()}
    for method in ("bm25", "dense"):
        for depth in DEPTHS:
            detail[f"{method}_top{depth}"] = bool(detail[method + "_rank"] and detail[method + "_rank"] <= depth)
    detail["rrf_top100"] = bool(detail["rrf_rank"] and detail["rrf_rank"] <= 100)
    detail["first_fixed_depth"] = next((k for k in DEPTHS if detail[f"bm25_top{k}"] or detail[f"dense_top{k}"]), None)
    return detail


def classify_source(doc_id, ranked, docs):
    if doc_id is None:
        return "UNCERTAIN"
    if doc_id not in docs:
        return "CORPUS_MISSING"
    depth = ranked["first_fixed_depth"]
    if depth == 15:
        return "OTHER"
    if depth == 30:
        return "P5_CUTOFF"
    if depth in (50, 100):
        return "DEEP_RANKING"
    return "UNCERTAIN"


def explain_remaining(row, facts):
    diagnostic = row["failure_diagnostic"]
    missing = [f for f in facts if not f["besa_top10"]]
    if not missing or not all(f["p5_candidate"] and f["p5b_rank"] > 10 for f in missing):
        return "UNCERTAIN", "来源位置不能复现该历史诊断"
    if diagnostic == "RUNTIME_FALLBACK:no_scores_above_gate":
        if not row["supports"] or any(c["support"] > 0 for cs in row["supports"].values() for c in cs.values()):
            return "UNCERTAIN", "没有复现所有 Need–父文档支持值均为零"
        return "NO_POSITIVE_SUPPORT", "BESA 的需求支持全部为零，完整复用 P5-B 排序"
    if diagnostic == "BUDGET_CHUNK_NOT_SCORED" and any(f["literal_chunk_ids"] and not f["literal_scored_chunk_ids"] for f in missing):
        return diagnostic, "至少一个落选必要来源有完整字面见证块，但 B30 未给任何该来源的匹配块评分"
    if diagnostic == "SCORER_DID_NOT_RETAIN_LITERAL_WITNESS" and any(f["literal_scored_chunk_ids"] and not f["literal_positive_winning_chunk_ids"] for f in missing):
        return diagnostic, "至少一个落选必要来源的字面块已评分，但没有成为正支持的胜出块"
    if diagnostic == "WITNESSED_SOURCE_NOT_SELECTED" and all(f["literal_positive_winning_chunk_ids"] for f in missing):
        return diagnostic, "必要来源已有得分为正的完整字面胜出块，父文档仍排在前十之外"
    return "UNCERTAIN", "现有原始分数、分块和排序不足以确认该诊断"


def inspect_failure(row, pool, ranks, query, docs, chunks):
    by_source = defaultdict(list)
    for evidence in query.evidence_list:
        by_source[evidence.source_document_id].append(evidence)
    baseline = {doc_id: i for i, doc_id in enumerate(pool, 1)}
    final = {doc_id: i for i, doc_id in enumerate(row["full_ranking"], 1)}
    scored = set(row.get("scored_chunks", ()))
    selected = row["selected_chunk_ids"]
    facts = []
    for doc_id, evidence_rows in sorted(by_source.items(), key=lambda item: item[0] or ""):
        details = rank_detail(doc_id, ranks)
        for evidence in evidence_rows:
            full = _normalize(evidence.evidence_text)
            literal = [chunk.chunk_id for chunk in chunks.get(doc_id, ()) if full in _normalize(chunk.text)]
            cells = {
                need: score[doc_id] for need, score in row["supports"].items() if doc_id in score
            }
            facts.append({
                "evidence_id": evidence.evidence_id,
                "source_document_id": doc_id,
                **details,
                "p5_candidate": doc_id in baseline,
                "p5b_rank": baseline.get(doc_id),
                "besa_rank": final.get(doc_id),
                "besa_top10": 0 < final.get(doc_id, 0) <= 10,
                "p5_missing_source_class": classify_source(doc_id, details, docs) if doc_id not in baseline else "",
                "literal_chunk_ids": literal,
                "literal_scored_chunk_ids": sorted(scored.intersection(literal)),
                "literal_positive_winning_chunk_ids": sorted({
                    cell["chunk_id"] for cell in cells.values() if cell["support"] > 0 and cell["chunk_id"] in literal
                }),
                "retained_chunk_ids": selected.get(doc_id, []),
                "need_supports": cells,
                "evidence_text": evidence.evidence_text,
            })
    if {f["evidence_id"] for f in facts} != {f["evidence_id"] for f in row["facts"]}:
        raise ValueError(f"Historical evidence IDs differ for {query.query_id}")
    if set(final) != set(baseline) or any((f["first_fixed_depth"] == 15) != f["p5_candidate"] for f in facts):
        raise ValueError(f"P5 membership mismatch: {query.query_id}")
    if all(f["besa_top10"] for f in facts):
        raise ValueError(f"Expected incomplete round-seven ranking: {query.query_id}")
    diagnostic = row["failure_diagnostic"]
    all_sources_present = set(by_source) <= set(baseline)
    p5_missing = sorted(set(by_source) - set(baseline), key=lambda x: x or "")
    question = query.question
    if diagnostic == "P5_MISSING_SOURCE":
        if not p5_missing:
            raise ValueError(f"Expected missing source: {query.query_id}")
        classification = diagnostic
        verified = "已核对语料与重放分支名次，缺失来源按最浅可见的固定候选深度分类"
    elif diagnostic == "PLANNER_NOT_ACCEPTED":
        if not all_sources_present or row["plan"]["accepted"] or row["full_ranking"] != pool:
            raise ValueError(f"Historical Planner fallback differs from full P5-B: {query.query_id}")
        classification = diagnostic
        verified = PLANNER_EXPLANATIONS[row["plan"]["fallback_reason"]]
    elif all_sources_present:
        classification, verified = explain_remaining(row, facts)
    else:
        classification, verified = "UNCERTAIN", "来源与历史诊断不一致"
    combined_rank = [
        rank_detail(parent, ranks)["first_fixed_depth"] for parent in p5_missing
    ]
    return {
        "query_id": query.query_id,
        "question": question,
        "question_type": query.question_type,
        "fact_count": len(facts),
        "source_count": len(by_source),
        "historical_diagnostic": diagnostic,
        "classification": classification,
        "verified_observation": verified,
        "all_sources_in_p5": all_sources_present,
        "p5b_complete_top10": all(0 < baseline.get(doc_id, 0) <= 10 for doc_id in by_source),
        "p5_missing_source_ids": p5_missing,
        "besa_missing_source_ids": sorted({f["source_document_id"] for f in facts if not f["besa_top10"]}, key=lambda x: x or ""),
        "smallest_fixed_depth_with_all_sources": max(combined_rank) if combined_rank and all(combined_rank) else (None if combined_rank else 15),
        "planner_accepted": row["plan"]["accepted"],
        "planner_fallback_reason": row["plan"]["fallback_reason"],
        "planner_structure_defect": PLANNER_EXPLANATIONS.get(row["plan"]["fallback_reason"], ""),
        "planner_emitted_needs": len(row["plan"]["needs"]),
        "constraint_loss": ("RULE_VALIDATION_REJECTED" if row["plan"]["fallback_reason"] == "constraint_validation_failed"
                            else "NO_EMITTED_NEEDS" if not row["plan"]["accepted"] else "NOT_EVALUATED"),
        "reference_markers": [m.group() for m in planner._ANAPHORA.finditer(question)],
        "comparison_markers": [m.group() for m in COMPARISON.finditer(question)],
        "temporal_markers": [m.group() for m in TEMPORAL.finditer(question)],
        "explicit_dates": [m.group() for m in MONTH.finditer(question)],
        "source_attribution_markers": [m.group() for m in REPORTS.finditer(question)],
        "scope_markers": planner._scope_terms(question),
        "p5b_top10": pool[:10],
        "besa_top10": row["final_top10"],
        "facts": facts,
    }


def fixed_pools(query, ranks, cohort):
    expected = {
        (item.source_document_id, _normalize(item.evidence_text)) for item in query.evidence_list
    }
    by_source = {source for source, _ in expected}
    first = {doc_id: rank_detail(doc_id, ranks) for doc_id in sorted(by_source, key=lambda x: x or "")}
    original = None
    for depth in DEPTHS:
        start = perf_counter()
        candidate_ids = {
            doc_id for method in ("bm25", "dense") for doc_id, rank in ranks[method].items()
            if rank <= depth
        }
        seconds = perf_counter() - start
        if original is None:
            original = candidate_ids
        yield {
            "query_id": query.query_id,
            "split": cohort,
            "question_type": query.question_type,
            "fact_count": len(expected),
            "fixed_arm": f"E{DEPTHS.index(depth)}",
            "branch_depth": depth,
            "candidate_size": len(candidate_ids),
            "added_candidates_from_p5": len(candidate_ids - original),
            "candidate_complete": by_source <= candidate_ids,
            "candidate_recall": sum(source in candidate_ids for source, _ in expected) / len(expected),
            "required_sources_added": len((candidate_ids - original) & by_source),
            "saved_rank_union_seconds": seconds,
            "candidate_ids": sorted(candidate_ids),
            "required_source_rank": first,
        }


def write_report(summary, failures, metrics, audit_seconds):
    missing = summary["missing_source_category_totals"]
    bands = summary["missing_sources_by_first_fixed_depth"]
    fitted = summary["recovered_missing_questions_by_fixed_depth"]
    planner_stats = summary["planner_rejected"]
    planner_table = "\n".join(
        f"| {name} | {count} | {PLANNER_EXPLANATIONS[name]} |"
        for name, count in planner_stats["reasons"].items()
    )
    counts = "\n".join(f"| {name} | {missing[name]} |" for name in SOURCES)
    fixed_table = "\n".join(
        f"| {r['fixed_arm']} | {r['branch_depth']} | {r['candidate_complete_count']}/{r['questions']} | "
        f"{r['candidate_complete_coverage']:.2%} | {r['candidate_evidence_recall']:.2%} | "
        f"{r['mean_candidate_size']:.2f} | {r['mean_added_candidates_from_p5']:.2f} | "
        f"{1e6 * r['mean_saved_rank_union_seconds']:.1f} |"
        for r in metrics if r["split"] == "all"
    )
    leftover = "\n".join(
        f"| {r['query_id']} | {r['classification']} | " + "、".join(
            f"{f['source_document_id']}（P5-B {f['p5b_rank']}，BESA {f['besa_rank']}）"
            for f in r["facts"] if not f["besa_top10"]
        ) + f" | {r['verified_observation']} |"
        for r in failures if r["historical_diagnostic"] not in ("P5_MISSING_SOURCE", "PLANNER_NOT_ACCEPTED")
    )
    historical_timing = summary["inputs"]["historical_full_stage_ranking_seconds"]
    historical_queries = summary["inputs"]["historical_full_stage_query_count"]
    lines = f"""# EGE-v0 前置失败重新归因与固定候选深度对照

以 BESA C-B30 第七轮结果为起点，451 道开发题中 {len(failures)} 道未找齐全部必要来源。对照由冻结的原始问题 BM25、Dense 与 RRF 排名构造，Gold 仅用于本离线脚本。内部开发划分沿用 271 道 Tune 和 180 道 Validation。

## 输入与完整性核验

交叉核验第六轮 P5-B 完整候选顺序、第七轮 451 道逐题结果、原始分支排名和冻结候选。用历史两路 Top 100 与相同的 RRF 参数还原 RRF，逐题对齐历史 Top 10 和存档的冻结候选。查看前置 BESA 审查和第七轮总结，核验五个原始输入和三个候选产物的历史摘要。原始语料有 609 篇父文档、2,378 个可还原的 Chunk。SQLite 以只读方式验证 2,378 个 Chunk 与 2,556 个原问题的 4,934 个有效向量，审计前后摘要一致。新增 Embedding 请求为 0。

## 46 道候选缺失题

按不同的题目–必要父文档组合计数，46 题缺 {sum(missing.values())} 个来源。归因采用互斥的可复现截断区间。最好的 BM25 或 Dense 位次在 16–30 归 P5_CUTOFF，在 31–100 归 DEEP_RANKING。两者广义上均受原 P5 Top 15 截断。超出双路 Top 100 只能确认观察边界，如无种子线索实证则标 UNCERTAIN。QUERY_MISMATCH 需要新查询的实证，本批没有超出该边界的来源。RRF 排名另行保留，它本身由原两路深度 100 输入重放。

| 归因 | 缺失的题目–来源对 |
|---|---:|
{counts}

最浅固定深度是 30、50、100 的来源数分别为 {bands.get('30', 0)}、{bands.get('50', 0)}、{bands.get('100', 0)}。双路各取 30、50、100 后，分别找齐 {fitted['30']}/46、{fitted['50']}/46、{fitted['100']}/46 道原候选缺失题。每个来源的两路 Top 15/30/50/100 命中以及 RRF Top 100 标志、原始名次与文本诊断见 [逐题 CSV](failure_reclassification.csv) 中的 facts 列。

## 54 道 Planner 拒绝题

这 54 道的 P5-B 候选均包含全部必要来源，P5-B 的前十均不完整。规划拒绝后各题完整排序与 P5-B 一致，规划器没有输出 Needs，因此没有可用于逐字判断约束是否丢失的规划文本。句式结构缺陷依据具体拒绝信号记录。存在指代表面标记 {planner_stats['with_reference_markers']} 道、比较词 {planner_stats['with_comparison_markers']} 道、时间关系词 {planner_stats['with_temporal_markers']} 道、明确英文日期 {planner_stats['with_explicit_dates']} 道。这些词法特征可重叠，也不等于语义上确实需要消歧。程序的前置规则可能把可局部消解的 its 或 it 一并拒绝。

| 原拒绝代码 | 题数 | 可观察的结构阻塞 |
|---|---:|---|
{planner_table}

原题面与 P5-B 种子文档仍然存在，可以在未获得 Need 的情况下独立运行文本线索扩展。此组候选来源原已齐全，改进应以这些来源是否进入最终前十衡量。EGE 实测另见主报告。

## 剩余 15 道逐题复核

所有落选来源均在原 P5，且原 P5-B 排名超过第 10。依据第七轮 B30 的实评分块、每个来源的字面见证块、正支持矩阵与完整顺序逐题复核。字面匹配沿用旧评估的普通空格与换行归一化。Jina 的 256 token 输入可能截断文本，原始 Chunk 的字面包含无法单独证明模型实际读到该句或确认语义相关。

| Query ID | 当前类别 | 落选必要来源的原始与最终名次 | 核对依据 |
|---|---|---|---|
{leftover}

组内计数为未过正支持门槛 5 道、预算未评分字面块 4 道、评分未留存字面胜出块 5 道、选择器未将已有可定位支持的来源纳入前十 1 道。可观察的首个瓶颈可确认，底层评分错误、截断和题面规划的相对贡献需要进一步控制实验。所有 115 道均有逐题结果，若复现检查不满足条件，脚本会记录 UNCERTAIN。

## 固定初始候选池

下表 E0–E3 专指任务书第五节的候选池容量对照。合并两路后按父文档去重。这里的完整率表示所有必要来源均在候选池，最终排名 Top 10 按主报告另计。

| 固定池 | 每路深度 | 候选完整题 | 候选完整率 | 候选 Evidence Recall | 每题候选数均值 | 每题新增均值 | 已存排名过滤去重 μs/题 |
|---|---:|---:|---:|---:|---:|---:|---:|
{fixed_table}

表中计时只包含对已存排名的过滤去重。历史完整实验的 {historical_queries} 道问题分别花费 {historical_timing["bm25"]:.3f} 秒 BM25 排名、{historical_timing["dense"]:.3f} 秒 Dense 排名和 {historical_timing["hybrid"]:.3f} 秒 RRF 排名，查询向量请求单独计时。固定池重放直接复用这些排名。输入核验、原始 RRF 重放、失败分类和固定候选计算共耗时 {audit_seconds:.2f} 秒。逐题候选规模、首次可见深度、必要来源位置和实际过滤耗时见 [逐题固定池](fixed_pool_results.jsonl)，汇总及分组见 [固定池指标](fixed_pool_metrics.csv)。

复现命令在正式项目根目录运行。

~~~powershell
& .\\.venv\\Scripts\\python.exe -B -m analysis.ege_v0.audit
~~~
"""
    (OUT / "FAILURE_RECLASSIFICATION.md").write_text(lines, encoding="utf-8", newline="\n")


def main():
    started = perf_counter()
    wanted = set(development_ids())
    split = read(BESA / "run" / "development_split.json")
    cohort = {qid: name for name in ("tune", "validation") for qid in split[name]}
    if (len(split["tune"]), len(split["validation"]), set(cohort)) != (271, 180, wanted):
        raise ValueError("Historical BESA split differs from the frozen development split")
    dataset = load_dataset(RAW_DATA)
    development = {query.query_id: query for query in dataset.queries if query.query_id in wanted}
    config = read(BASELINE / "run.json")["config"]
    docs, chunks, inputs = verify_inputs(dataset, config)
    ranks = dict(replay_original(development, config))
    old = {row["query_id"]: row for row in rows(BESA / "run" / "selected_results.jsonl")}
    if old.keys() != development.keys():
        raise ValueError("Historical BESA output differs from frozen development scope")
    pools = {pool.query_id: rank for pool, rank, _ in development_inputs(wanted)}
    for qid, prior in pools.items():
        p5 = {doc for name in ("bm25", "dense") for doc, rank in ranks[qid][name].items() if rank <= 15}
        if p5 != set(prior) or prior[:10] != old[qid]["original_top10"]:
            raise ValueError(f"Archived P5-B reproduction mismatch: {qid}")
    failures = [
        inspect_failure(row, pools[qid], ranks[qid], development[qid], docs, chunks)
        for qid, row in sorted(old.items()) if not row["complete"]
    ]
    if len(failures) != 115:
        raise ValueError("Expected 115 BESA incomplete development questions")
    fixed = [
        row for qid, query in sorted(development.items())
        for row in fixed_pools(query, ranks[qid], cohort[qid])
    ]
    groups = defaultdict(list)
    for row in fixed:
        for split_name in ("all", row["split"]):
            for group in ("all", row["question_type"], f"facts_{row['fact_count']}"):
                groups[(split_name, group, row["branch_depth"])].append(row)
    metrics = [{
        "split": split_name, "group": group, "branch_depth": depth,
        "fixed_arm": partition[0]["fixed_arm"], "questions": len(partition),
        "candidate_complete_count": sum(r["candidate_complete"] for r in partition),
        "candidate_complete_coverage": mean(r["candidate_complete"] for r in partition),
        "candidate_evidence_recall": mean(r["candidate_recall"] for r in partition),
        "mean_candidate_size": mean(r["candidate_size"] for r in partition),
        "min_candidate_size": min(r["candidate_size"] for r in partition),
        "max_candidate_size": max(r["candidate_size"] for r in partition),
        "mean_added_candidates_from_p5": mean(r["added_candidates_from_p5"] for r in partition),
        "required_sources_added": sum(r["required_sources_added"] for r in partition),
        "mean_saved_rank_union_seconds": mean(r["saved_rank_union_seconds"] for r in partition),
    } for (split_name, group, depth), partition in sorted(groups.items())]
    candidate_gap = [r for r in failures if r["historical_diagnostic"] == "P5_MISSING_SOURCE"]
    planner_rejected = [r for r in failures if r["historical_diagnostic"] == "PLANNER_NOT_ACCEPTED"]
    leftover = [r for r in failures if r["historical_diagnostic"] not in ("P5_MISSING_SOURCE", "PLANNER_NOT_ACCEPTED")]
    if (len(candidate_gap), len(planner_rejected), len(leftover)) != (46, 54, 15):
        raise ValueError("Historical failure-group counts differ from 46/54/15")
    missing_facts = {
        (r["query_id"], f["source_document_id"]): f
        for r in candidate_gap for f in r["facts"] if not f["p5_candidate"]
    }
    source_counts = Counter(f["p5_missing_source_class"] for f in missing_facts.values())
    summary = {
        "inputs": inputs,
        "question_classification_totals": dict(Counter(r["classification"] for r in failures)),
        "missing_source_category_totals": {name: source_counts[name] for name in SOURCES},
        "missing_sources_by_first_fixed_depth": dict(Counter(str(f["first_fixed_depth"]) for f in missing_facts.values())),
        "missing_sources_in_rrf_top100": sum(f["rrf_top100"] for f in missing_facts.values()),
        "recovered_missing_questions_by_fixed_depth": {
            str(depth): sum(r["smallest_fixed_depth_with_all_sources"] is not None
                            and r["smallest_fixed_depth_with_all_sources"] <= depth for r in candidate_gap)
            for depth in DEPTHS
        },
        "planner_rejected": {
            "reasons": dict(Counter(r["planner_fallback_reason"] for r in planner_rejected)),
            "question_types": dict(Counter(r["question_type"] for r in planner_rejected)),
            **{f"with_{feature}": sum(bool(r[feature]) for r in planner_rejected)
               for feature in ("reference_markers", "comparison_markers", "temporal_markers", "explicit_dates", "source_attribution_markers")},
        },
        "remaining_15": dict(Counter(r["classification"] for r in leftover)),
        "fixed_pool_totals": [r for r in metrics if r["split"] == r["group"] == "all"],
    }
    seconds = perf_counter() - started
    csv_rows(OUT / "failure_reclassification.csv", ({
        **{name: (json.dumps(value, ensure_ascii=False, separators=(",", ":")) if isinstance(value, list) else value)
           for name, value in r.items()}, "split": cohort[r["query_id"]],
    } for r in failures))
    write_rows(OUT / "failure_reclassification.jsonl", failures)
    csv_rows(OUT / "fixed_pool_metrics.csv", metrics)
    write_rows(OUT / "fixed_pool_results.jsonl", fixed)
    write(OUT / "audit_summary.json", summary)
    write_report(summary, failures, summary["fixed_pool_totals"], seconds)
    print(json.dumps({**summary, "input_audit_and_replay_seconds": seconds}, ensure_ascii=False))


if __name__ == "__main__":
    main()
