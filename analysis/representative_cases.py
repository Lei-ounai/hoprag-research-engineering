"""Render stratified, source-grounded cases from the fourth-round failure ledger."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from hoprag.dataset import load_dataset
from hoprag.retrievers import _tokens

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "analysis"
TYPE_LABELS = {
    "inference_query": "推理",
    "comparison_query": "比较",
    "temporal_query": "时间",
}
METHODS = ("bm25", "dense", "rrf")
METHOD_LABELS = ("BM25", "Dense", "RRF")
MECHANISM_LABELS = {
    "outside_pool": "两路 Top 100 均无来源",
    "below_fused_100": "来源进入融合池，RRF 排到第 101 名或更后",
    "lost_at_fusion": "至少一路 Top 10 有来源，RRF 未保留",
    "cross_boundary": "来源事实跨块边界",
    "low_in_both": "RRF Top 100 内，三路 Top 10 均无来源",
    "other": "存在需另查的评测、语料或映射现象",
}


def at_most(rank: int | None, cutoff: int) -> bool:
    return rank is not None and rank <= cutoff


def mechanism(case: dict) -> str:
    missing = case["missing_evidence"]
    if any(
        e["category"] not in {"CANDIDATE_MISS", "RANKING_FAILURE"}
        for e in missing
    ):
        return "other"
    if any(not e["rrf_pool_present"] for e in missing):
        return "outside_pool"
    if any(e["first_ranks"]["rrf"] is None for e in missing):
        return "below_fused_100"
    if any(e["chunk_status"] == "cross_boundary" for e in missing):
        return "cross_boundary"
    if any(at_most(e["first_ranks"][m], 10) for e in missing for m in METHODS[:2]):
        return "lost_at_fusion"
    return "low_in_both"


def choose_cases(cases: list[dict], seed: int, per_type: int) -> list[dict]:
    rng = random.Random(seed)
    selected = []
    for kind in TYPE_LABELS:
        grouped = defaultdict(list)
        for case in sorted(cases, key=lambda c: c["query_id"]):
            if case["question_type"] == kind:
                grouped[case["required_evidence_count"]].append(case)
        for pool in grouped.values():
            rng.shuffle(pool)
        used_counts = Counter()
        used_mechanisms = Counter()
        for _ in range(per_type):
            available = [count for count, pool in grouped.items() if pool]
            if not available:
                raise ValueError(f"{kind} has fewer than {per_type} failed questions")
            count = min(available, key=lambda n: (used_counts[n], n))
            pool = grouped[count]
            index = min(
                range(len(pool)),
                key=lambda i: used_mechanisms[mechanism(pool[i])],
            )
            case = pool.pop(index)
            selected.append(case)
            used_counts[count] += 1
            used_mechanisms[mechanism(case)] += 1
    return selected


def rank_map(ranking: list[dict]) -> dict[str, int]:
    return {row["document_id"]: row["rank"] for row in ranking}


def fused_pool(case: dict, rrf_k: int) -> tuple[dict[str, int], dict[str, float]]:
    scores = defaultdict(float)
    for method in METHODS[:2]:
        for item in case["rankings"][method]:
            scores[item["document_id"]] += 1 / (rrf_k + item["rank"])
    order = sorted(scores, key=lambda doc_id: (-scores[doc_id], doc_id))
    observed = [row["document_id"] for row in case["rankings"]["rrf"]]
    if observed != order[:len(observed)]:
        raise ValueError(f"{case['query_id']} RRF ranks do not match the frozen candidate pool")
    return {doc_id: rank for rank, doc_id in enumerate(order, 1)}, dict(scores)


def text(value: str) -> str:
    """Keep literal facts readable in Markdown, including any original punctuation."""
    return " ".join(value.split())


def cell(value: str) -> str:
    return text(value).replace("|", r"\|").replace("<", "&lt;").replace(">", "&gt;")


def score(value: float) -> str:
    return f"{value:.9f}"


def print_rank(value: int | None) -> str:
    return str(value) if value is not None else "未进入前 100"


def ranking_table(case: dict, documents: dict, gold_ids: set[str]) -> list[str]:
    lines = ["| 名次 | BM25 | Dense | RRF |", "| ---: | --- | --- | --- |"]
    for index in range(10):
        entries = []
        for method in METHODS:
            ranking = case["rankings"][method]
            if len(ranking) <= index:
                entries.append("—")
                continue
            doc_id = ranking[index]["document_id"]
            mark = " ◆" if doc_id in gold_ids else ""
            entries.append(f"{cell(documents[doc_id].title)}（`{doc_id}`）{mark}")
        lines.append(f"| {index + 1} | {' | '.join(entries)} |")
    return lines


def dense_diagnosis(evidence: dict) -> str:
    diagnostic = evidence["dense_diagnostic"]
    full_rank = diagnostic["document_rank_full"]
    if full_rank is None:
        return "Dense 全库文档排名未提供该来源。"
    winning_id = diagnostic["winning_chunk_id"]
    winning_score = diagnostic["winning_chunk_score"]
    contains = diagnostic["winning_chunk_contains_fact"]
    phrase = (
        f"Dense 全库第 {full_rank}，最高分块为 `{winning_id}`"
        f"（余弦 {winning_score:.5f}，{'含有' if contains else '未含'}该金标事实）。"
    )
    best_fact_id = diagnostic["best_fact_chunk_id"]
    best_fact_score = diagnostic["best_fact_chunk_score"]
    if best_fact_id and best_fact_id != winning_id:
        phrase += f"含完整事实的最高分块为 `{best_fact_id}`（余弦 {best_fact_score:.5f}）。"
    return phrase


def source_name_diagnosis(case: dict, doc) -> str:
    publisher = doc.metadata.get("source") if doc else None
    if not isinstance(publisher, str) or not publisher:
        return ""
    name = publisher.casefold()
    if name not in case["question"].casefold() or name in f"{doc.title}\n{doc.text}".casefold():
        return ""
    return (
        f"题干包含来源名「{publisher}」，该名称来自来源字段且未完整出现在索引的标题或正文。"
        "可独立比较使用来源字段的召回效果。"
    )


def missing_diagnosis(
    evidence: dict,
    case: dict,
    doc,
    pool_ranks: dict[str, int],
    pool_scores: dict[str, float],
    rrf_k: int,
) -> str:
    doc_id = evidence["source_document_id"]
    ranks = evidence["first_ranks"]
    rrf_rows = case["rankings"]["rrf"]
    prefix = f"`{evidence['evidence_id']}`"
    if evidence["category"] not in {"CANDIDATE_MISS", "RANKING_FAILURE"}:
        return f"{prefix} 属于 `{evidence['category']}`。先依据逐题记录复查证据与原文和来源的对应，再判断检索阶段。"
    if not evidence["rrf_pool_present"]:
        overlap = sorted(set(_tokens(case["question"])) & set(_tokens(f"{doc.title}\n{doc.text}"))) if doc else []
        if overlap:
            shown = "、".join(f"`{term}`" for term in overlap[:8])
            signal = (
                f"按现有 BM25 分词，问题与来源标题加正文有 {len(overlap)} 个共用词项"
                f"（列出至多 8 个 {shown}），可通过词项候选过滤，但未进入 BM25 前 100。"
            )
        elif doc:
            signal = "按现有 BM25 分词，问题与来源标题加正文无共同词项，来源在词项候选过滤时被排除。"
        else:
            signal = "无法从来源文档核对 BM25 词项。"
        full_bm25_rank = evidence["bm25_rank_full"]
        if full_bm25_rank is not None:
            signal += f"它在全部 BM25 词项候选中排第 {full_bm25_rank}。"
        direction = "可对比扩宽单路候选与保留原约束的子查询，观察该来源的召回增量。"
        if evidence["chunk_status"] == "cross_boundary":
            direction += "金标事实跨块，另用有重叠的切块做对照，单独测量 Dense 名次是否改变。"
        return (
            f"{prefix} 在 BM25 和 Dense 前 100 均缺席，RRF 没有可融合的该来源。"
            f"{signal}{dense_diagnosis(evidence)}{direction}{source_name_diagnosis(case, doc)}"
        )

    target = pool_ranks[doc_id]
    if target != evidence["rrf_pool_rank"]:
        raise ValueError(f"{evidence['evidence_id']} has an inconsistent fusion-pool rank")
    threshold = min(100 if target > 100 else 10, len(rrf_rows))
    boundary = rrf_rows[threshold - 1]["document_id"]
    target_score = pool_scores[doc_id]
    boundary_score = pool_scores[boundary]
    comparison = "与该截断位同分，按文档 ID 排序后落在后面" if target_score == boundary_score else "低于该截断位"
    branches = "，".join(
        f"{label} 贡献 {score(1 / (rrf_k + ranks[method]) if ranks[method] is not None else 0)}"
        for method, label in zip(METHODS[:2], METHOD_LABELS[:2], strict=True)
    )
    signal = (
        f"{prefix} 的来源在完整融合池排第 {target}（{branches}）。"
        f"其 RRF 分 {score(target_score)}，{comparison}。第 {threshold} 名为 "
        f"`{boundary}`（{score(boundary_score)}）。"
    )
    if target > 100:
        direction = "可先将它纳入离线重排对照，并区分融合池召回和最终输出的候选深度。"
    elif any(at_most(ranks[method], 10) for method in METHODS[:2]):
        direction = "单路原本已进入前 10。可做保留单路高位候选或候选重排的消融，检查是否能恢复该来源。"
    else:
        direction = "来源已经进入前 100。可先在现有候选上做证据相关性重排，并记录它能否跨过前 10 截断。"
    if evidence["chunk_status"] == "cross_boundary":
        direction += "金标事实跨块，需另做切块对照才能判断该现象对 Dense 名次的影响。"
    return signal + dense_diagnosis(evidence) + direction + source_name_diagnosis(case, doc)


def case_lines(case: dict, documents: dict, rrf_k: int) -> list[str]:
    pool_ranks, pool_scores = fused_pool(case, rrf_k)
    gold_ids = {e["source_document_id"] for e in case["evidence"]}
    first_complete = case["rrf_first_complete_cutoff"]
    recovery = f"在核查的截断位中，首次于 Top {first_complete} 找齐。" if first_complete else "Top 100 仍未找齐。"
    by_method = {method: rank_map(case["rankings"][method]) for method in METHODS}
    lines = [
        f"### {case['query_id']} · {TYPE_LABELS[case['question_type']]} · {case['required_evidence_count']} 条必要证据",
        "",
        f"**原始问题**　{text(case['question'])}",
        "",
        f"**数据集答案**　{text(case['answer'])}",
        "",
        f"**RRF 证据覆盖**　Top 10 找到 {case['matched_evidence_count']} / {case['required_evidence_count']} 条。"
        f"本题分类 `{case['category']}`。{recovery}",
        "",
        "#### 三种方法的前 10 篇文档",
        "",
        *ranking_table(case, documents, gold_ids),
        "",
        "#### 必要证据与来源首次排名",
        "",
    ]
    for evidence in case["evidence"]:
        doc_id = evidence["source_document_id"]
        doc = documents.get(doc_id)
        ranks = evidence["first_ranks"]
        if doc_id and any(ranks[method] != by_method[method].get(doc_id) for method in METHODS):
            raise ValueError(f"{evidence['evidence_id']} has inconsistent source-document ranks")
        descriptions = "，".join(
            f"{label} {print_rank(ranks[method])}"
            for method, label in zip(METHODS, METHOD_LABELS, strict=True)
        )
        rrf_rank = ranks["rrf"]
        result = "Top 10 已覆盖" if at_most(rrf_rank, 10) else (
            "Top 10 缺失，Top 100 已出现" if rrf_rank is not None else "Top 100 仍缺失"
        )
        origin = f"`{doc_id}`《{doc.title}》" if doc else "未确认来源"
        url = doc.metadata.get("url") if doc else None
        source = f" [来源原文]({url})" if isinstance(url, str) and url.startswith(("https://", "http://")) else ""
        chunk_note = {
            "contained": "原文事实完整落在单块内",
            "cross_boundary": "原文事实横跨块边界",
            "missing": "未在块文本复原中定位到该事实",
        }.get(evidence["chunk_status"], f"块检查 `{evidence['chunk_status']}`")
        chunk_ids = evidence["supporting_chunk_ids"]
        chunks = "，对应 " + "、".join(f"`{chunk_id}`" for chunk_id in chunk_ids) if chunk_ids else ""
        lines.extend([
            f"- **{evidence['evidence_id']}**　{result}。{origin}。{descriptions}。{chunk_note}{chunks}。{source}".rstrip(),
            "",
            f"  > {text(evidence['text'])}",
            "",
        ])
    lines.extend(["#### 已观察到的失败机制与改进对照", ""])
    for evidence in case["missing_evidence"]:
        lines.append(
            "- " + missing_diagnosis(
                evidence, case, documents.get(evidence["source_document_id"]), pool_ranks, pool_scores, rrf_k
            )
        )
    lines.append("")
    return lines


def render(cases: list[dict], documents: dict, rrf_k: int, seed: int, per_type: int) -> str:
    picked = choose_cases(cases, seed, per_type)
    by_stratum = Counter((c["question_type"], c["required_evidence_count"]) for c in picked)
    by_mechanism = Counter(map(mechanism, picked))
    lines = [
        "# 代表性失败案例",
        "",
        f"从 {len(cases)} 道 RRF Top 10 证据未找齐的问题中选取 **{len(picked)} 道**。"
        f"推理、比较、时间各 {per_type} 道，同类中轮流抽取不同必要证据条数。"
        f"同一分层内优先覆盖不同可观测机制，用种子 {seed} 打破并列。"
        "选样用于说明失败形态，类别比例以全量失败台账为准。",
        "",
        "| 题型 | 2 条证据 | 3 条证据 | 4 条证据 |",
        "| --- | ---: | ---: | ---: |",
        *[
            f"| {label} | " + " | ".join(str(by_stratum[kind, count]) for count in (2, 3, 4)) + " |"
            for kind, label in TYPE_LABELS.items()
        ],
        "",
        "入选案例按以下条件依次标记，每题计入一类。逐题仍按证据记录不同机制。",
        "",
        *[f"- {label}，{by_mechanism[key]} 道。" for key, label in MECHANISM_LABELS.items()],
        "",
        "表中 ◆ 表示正文经审计含有本题至少一条金标事实的来源文档。证据名次按金标来源 ID 计算。"
        "每条必要事实单独展示，同篇文档可以对应多条事实。分块位置用于提出对照实验，不能单独证明 Dense 失败的原因。",
        "",
        f"完整融合池使用既有 BM25 与 Dense 各自的前 100 名，按原配置 `k={rrf_k}` 计算。"
        "池内第 101 名或更后的来源在表中仍标为未进入前 100。改进方向是基于截断位置提出的待验证假设。",
        "",
        "复现时在正式项目根目录运行以下命令。完整排名见 [逐题失败台账](failure_cases.jsonl)。",
        "",
        "```powershell",
        r"& .\.venv\Scripts\python.exe -B -m analysis.representative_cases",
        "```",
        "",
    ]
    for case in picked:
        lines.extend(case_lines(case, documents, rrf_k))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=ANALYSIS)
    args = parser.parse_args()
    with (args.analysis_dir / "failure_cases.jsonl").open(encoding="utf-8") as source:
        cases = [json.loads(line) for line in source if line.strip()]
    dataset = load_dataset(ROOT / "data" / "raw" / "multihop_rag")
    documents = {doc.document_id: doc for doc in dataset.documents}
    run = json.loads((ROOT / "results" / "baseline" / "run.json").read_text(encoding="utf-8"))
    report = render(cases, documents, run["config"]["rrf_k"], seed=42, per_type=10)
    output = args.analysis_dir / "REPRESENTATIVE_FAILURE_CASES.md"
    output.write_text(report, encoding="utf-8", newline="\n")
    print(f"Wrote {output.name} (30 cases)")


if __name__ == "__main__":
    main()
