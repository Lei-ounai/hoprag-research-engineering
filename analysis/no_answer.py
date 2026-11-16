"""Reproduce the null-query audit using saved rankings and corpus text."""

from __future__ import annotations

import json
from collections import Counter
from itertools import combinations
from pathlib import Path
from statistics import mean, median

from hoprag.config import BenchmarkConfig
from hoprag.dataset import load_dataset

METHODS = {"bm25": "BM25", "dense": "Dense", "hybrid": "RRF"}
CASE_REVIEWS = {
    "q-0014": (
        "欧洲专利数量与专利局挑战",
        {
            "bm25": "US International Trade Commission (ITC)",
            "dense": "raised €60 million in equity",
            "hybrid": "Humira’s patent loss by end 2023",
        },
        "BM25 找到 Apple Watch 的专利争议，但正文写明受理机构是美国 ITC。Dense 找到欧洲公司的融资，RRF 找到药企专利到期。"
        "这三篇文章分别碰到专利、欧洲公司、专利损失等元素，均缺少 2021 年欧洲专利数量领先与欧洲专利局挑战指向同一公司的证据。",
    ),
    "q-0021": (
        "巡演首夜的独奏与乐器",
        {
            "bm25": "Taylor Swift's 1989 tour sold 2.2m tickets",
            "dense": "Beyoncé ruled the box office this weekend",
            "hybrid": "Concert tickets are more expensive than ever",
        },
        "BM25 首篇谈 Taylor Swift 巡演，Dense 首篇谈 Beyoncé 的演唱会电影，RRF 首篇谈演唱会票价与电影。"
        "检查的三篇候选都没有出现问题中的名称 Used To Be Young，也未提供该演出成员的首夜独奏与乐器关系。",
    ),
    "q-0115": (
        "JPMorgan 的亚洲扩张与金融科技投资",
        {
            "bm25": "JPMorgan Chase CEO Jamie Dimon",
            "dense": "which were run by JPMorgan, Goldman Sachs",
            "hybrid": "which were run by JPMorgan, Goldman Sachs",
        },
        "BM25 首篇报道银行主管参与听证会，金融科技一词出现在花旗等大型银行面临竞争的语境中。"
        "Dense 与 RRF 首篇提到 JPMorgan 等机构参与举办以日本市场为主题的多地会议，正文没有给出 JPMorgan 投资日本金融科技公司的关系。"
        "仅凭公司名和日本出现在同篇报道，无法推出题目要求的投资与扩张交集。",
    ),
    "q-0113": (
        "直接事实与题面来源限定",
        dict.fromkeys(METHODS, "Florida Gov. Ron DeSantis launched his presidential campaign"),
        "三种方法首位均是 TechCrunch，正文明确称 Ron DeSantis 为佛罗里达州州长，因此能从这篇文章推出所问首字母 F。"
        "题面点名的却是《纽约时报》和《华盛顿邮报》，首位候选未满足这两项来源要求。"
        "这是一个需要复核的无答案标注案例。若题目的来源限定可放宽，现有候选已经提供直接事实。",
    ),
}


def _score_summary(scores):
    if not scores:
        return "—"
    return f"{min(scores):.4f} / {median(scores):.4f} / {max(scores):.4f}"


def _read_rankings(path, questions, documents):
    records = {}
    with path.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            query_id = row["query_id"]
            if query_id in records or query_id not in questions:
                raise ValueError(f"unexpected query ID: {query_id}")
            if row["question_type"] != questions[query_id].question_type:
                raise ValueError(f"question type mismatch: {query_id}")
            for method in METHODS:
                ranking = row[method][:10]
                ids = [result["document_id"] for result in ranking]
                if len(set(ids)) != len(ids) or not set(ids) <= documents.keys():
                    raise ValueError(f"invalid Top 10 document IDs: {query_id} {method}")
                if any(result["rank"] != rank for rank, result in enumerate(ranking, 1)):
                    raise ValueError(f"invalid rank: {query_id} {method}")
                row[method] = ranking
            records[query_id] = row
    if records.keys() != questions.keys():
        raise ValueError("saved rankings do not cover exactly the dataset queries")
    return records


def build_report(queries, documents, records):
    null_ids = [query.query_id for query in queries.values() if query.question_type == "null_query"]
    answerable_ids = queries.keys() - set(null_ids)
    null_count = len(null_ids)
    null_labels = Counter(queries[query_id].answer for query_id in null_ids)
    if not null_ids or any(queries[query_id].evidence_list for query_id in null_ids):
        raise ValueError("expected null-query rows with empty gold evidence")

    lines = [
        "# 无答案问题的检索表现",
        "",
        f"官方数据集中有 {null_count} 道 `null_query`，必要证据列表均为空。答案标签及数量为 "
        + "、".join(f"`{label}`（{count}）" for label, count in sorted(null_labels.items()))
        + f"。{len(answerable_ids):,} 道可回答题只用作同一检索方法的分数对照。",
        "",
        "统计使用第三轮保存的文章级 Top 10。无答案题缺少逐文档支持标注，以下数量只衡量检索行为。"
        "个案通过检查语料正文判断具体事实是否得到支持。",
        "",
        "## 返回数量与分数",
        "",
        "| 方法 | 至少返回一篇 | 返回 10 篇 | 平均 Top 10 篇数 | 无答案首位分数 最低 / 中位 / 最高 | 可回答首位分数 最低 / 中位 / 最高 | 无答案首位分数达到可回答中位数 |",
        "| --- | ---: | ---: | ---: | --- | --- | ---: |",
    ]
    overlap_summary = []
    for method, label in METHODS.items():
        null_lists = [records[query_id][method] for query_id in null_ids]
        null_scores = [ranked[0]["score"] for ranked in null_lists if ranked]
        answerable_scores = [
            records[query_id][method][0]["score"]
            for query_id in answerable_ids
            if records[query_id][method]
        ]
        answerable_median = median(answerable_scores)
        at_median = sum(score >= answerable_median for score in null_scores)
        overlap_summary.append(f"{label} {at_median} 道")
        lines.append(
            f"| {label} | {len(null_scores)} | {sum(len(ranked) == 10 for ranked in null_lists)} | "
            f"{mean(map(len, null_lists)):.2f} | {_score_summary(null_scores)} | "
            f"{_score_summary(answerable_scores)} | {at_median} / {null_count} |"
        )
    lines.extend([
        "",
        "各方法的首位分数分别比较。以各自可回答题首位分数中位数作为描述性参照，达到该分数的无答案题为 "
        + "、".join(overlap_summary)
        + "。分数反映各方法的匹配或融合强度，单凭该分数无法确认题目所要求的关系得到支持。",
        "",
        "## 三种方法的候选重合",
        "",
        "| 方法组合 | 首位相同 | Top 10 平均共同文档数 | Top 10 无共同文档 | Top 10 文档集合相同 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for left, right in combinations(METHODS, 2):
        shared = []
        first_agreement = identical_sets = 0
        for query_id in null_ids:
            left_ids = [row["document_id"] for row in records[query_id][left]]
            right_ids = [row["document_id"] for row in records[query_id][right]]
            first_agreement += bool(left_ids and right_ids and left_ids[0] == right_ids[0])
            identical_sets += set(left_ids) == set(right_ids)
            shared.append(len(set(left_ids) & set(right_ids)))
        lines.append(
            f"| {METHODS[left]} 与 {METHODS[right]} | {first_agreement} / {null_count} | "
            f"{mean(shared):.2f} | {shared.count(0)} | {identical_sets} |"
        )
    lines.extend([
        "",
        "重合衡量排名之间的一致程度。两种方法返回相同的文章时，文章仍可能只提供题目所需关系的一部分。",
        "",
        "## 四个正文核对案例",
        "",
        "下面的正文片段均在对应首位文章中逐字复核。人工判断只覆盖列出的候选和题面所问关系。"
        "这四题用于展示候选现象，无法推算全体无答案题中相关或支持文档的比例。",
        "",
    ])
    for query_id, (heading, anchors, review) in CASE_REVIEWS.items():
        if query_id not in null_ids:
            raise ValueError(f"review case is not a null query: {query_id}")
        lines.extend([
            f"### {query_id} · {heading}",
            "",
            f"> {queries[query_id].question}",
            "",
            "| 方法 | 首位文档与来源 | 正文片段 |",
            "| --- | --- | --- |",
        ])
        for method, label in METHODS.items():
            row = records[query_id][method][0]
            document = documents[row["document_id"]]
            anchor = anchors[method]
            if anchor not in document.text:
                raise ValueError(f"review excerpt not present: {query_id} {method}")
            lines.append(
                f"| {label} | `{document.document_id}` {document.title}（{document.metadata['source']}） | "
                f"“{anchor}” |"
            )
        lines.extend(["", review, ""])
    lines.extend([
        "## 后续拒答研究",
        "",
        "需要独立于检索分数的支持判断。后续评估可先明确题目是否限定来源，再标注候选支持哪些事实、缺少哪些必要关系，"
        "并单独复核无答案题中已有直接证据的样本。用可回答与经人工复核的证据不足样本共同衡量误拒答和无证据作答，"
        "现有排名继续作为候选输入。",
        "",
        "## 复现",
        "",
        "在正式项目根目录运行。",
        "",
        "```powershell",
        "& .\\.venv\\Scripts\\python.exe -B -m analysis.no_answer",
        "```",
        "",
        "输入为 `results/baseline/rankings_full.jsonl` 与 `data/raw/multihop_rag/`。"
        "脚本重新统计全部记录，并复核上述正文片段与首位排名。",
        "",
    ])
    return "\n".join(lines)


def main():
    config = BenchmarkConfig()
    dataset = load_dataset(config.raw_directory)
    documents = {document.document_id: document for document in dataset.documents}
    queries = {query.query_id: query for query in dataset.queries}
    records = _read_rankings(config.output_directory / "rankings_full.jsonl", queries, documents)
    report_path = Path(__file__).with_name("no_answer_analysis.md")
    report_path.write_text(build_report(queries, documents, records), encoding="utf-8")


if __name__ == "__main__":
    main()
