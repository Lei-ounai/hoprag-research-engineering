"""Select reproducible development cases from saved rankings and evidence ranks."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from hashlib import sha256
import json
from pathlib import Path
import re
from unicodedata import normalize

from hoprag.dataset import load_dataset


OUTPUT = Path(__file__).resolve().parent
ROOT = OUTPUT.parents[1]
DEVELOPMENT_SPLITS = ("dev_5", "dev_20", "dev_100", "answerable_dev")
NULL_SPLIT = "null_dev"
GENERIC_ANSWERS = {"yes", "no", "true", "false", "unknown", "none", "both", "neither"}
STOP_WORDS = {
    "about", "according", "after", "also", "article", "before", "does", "from", "have",
    "into", "more", "mentioned", "reported", "some", "than", "that", "their", "them",
    "these", "they", "this", "those", "used", "were", "what", "when", "where", "which",
    "while", "will", "with", "would",
}


def _fold(value):
    return " ".join(normalize("NFKC", value).casefold().split())


def _answer_pattern(answer):
    text = _fold(answer)
    if text in GENERIC_ANSWERS or sum(character.isalnum() for character in text) < 3:
        return None
    return re.compile(rf"(?<!\w){re.escape(text)}(?!\w)")


def _answer_signal(pattern, evidence, document):
    if pattern is None:
        return 0
    if pattern.search(_fold(evidence.evidence_text)):
        return 2
    return int(bool(pattern.search(_fold(document.text))))


def _rank(value):
    return int(value) if value else None


def _show_rank(value):
    return str(value) if value is not None else "候选中无位置"


def _span(before, after):
    return after - before if before is not None and after is not None else 0


def _md(value):
    return " ".join(value.split()).replace("|", r"\|")


def _read_rows(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def _load_development_results(path):
    results = defaultdict(dict)
    for row in _read_rows(path):
        if row["split"] not in (*DEVELOPMENT_SPLITS, NULL_SPLIT):
            continue
        key = row["split"], row["run_id"], row["method"]
        query_id = row["query_id"]
        if query_id in results[key]:
            raise ValueError(f"Duplicate ranking: {key}/{query_id}")
        results[key][query_id] = row
    return results


def _load_evidence_rows(path, results):
    evidence = defaultdict(lambda: defaultdict(list))
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            key = row["split"], row["run_id"], row["method"]
            query_id = row["query_id"]
            if query_id not in results.get(key, ()):
                continue
            evidence[key][query_id].append(row)
    return evidence


def _query_rows(query, documents, evidence_rows, record):
    patterns = _answer_pattern(query.answer)
    ranks = {row["document_id"]: row["rank"] for row in record["ranking"]}
    by_id = {fact.evidence_id: fact for fact in query.evidence_list}
    if {row["evidence_id"] for row in evidence_rows} != set(by_id) or len(evidence_rows) != len(by_id):
        raise ValueError(f"Incomplete evidence-rank CSV for {query.query_id}")
    rows = []
    for row in evidence_rows:
        fact = by_id[row["evidence_id"]]
        doc = documents[fact.source_document_id]
        after = _rank(row["rank"])
        if after != ranks.get(doc.document_id) or row["source_document_id"] != doc.document_id:
            raise ValueError(f"Stale evidence rank for {fact.evidence_id}")
        rows.append({
            "evidence": fact, "document": doc, "before": _rank(row["rrf_rank"]), "after": after,
            "signal": _answer_signal(patterns, fact, doc),
        })
    return rows


def _case_candidates(query_map, documents, ranking_rows, evidence_rows):
    regressions, recoveries = [], []
    for query_id in sorted(ranking_rows):
        query = query_map[query_id]
        facts = _query_rows(query, documents, evidence_rows[query_id], ranking_rows[query_id])
        prior_complete = all(item["before"] is not None and item["before"] <= 10 for item in facts)
        post_complete = all(item["after"] is not None and item["after"] <= 10 for item in facts)
        if prior_complete and not post_complete:
            dropped = [item for item in facts if item["after"] is None or item["after"] > 10]
            pairs = [
                (answer, lost)
                for answer in facts if answer["signal"] and answer["after"] is not None and answer["after"] <= 10
                for lost in dropped if answer["document"].document_id != lost["document"].document_id
            ]
            best_pair = max(
                pairs,
                key=lambda p: (
                    _span(p[0]["after"], p[0]["before"]) > 0, p[0]["signal"],
                    _span(p[1]["before"], p[1]["after"]),
                ),
                default=None,
            )
            score = (
                bool(best_pair),
                bool(best_pair and _span(best_pair[0]["after"], best_pair[0]["before"]) > 0),
                best_pair[0]["signal"] if best_pair else 0,
                max(_span(item["before"], item["after"]) for item in dropped),
            )
            regressions.append((score, query, facts, best_pair))
        elif not prior_complete and post_complete:
            score = (
                sum(item["before"] is not None and item["before"] > 10 for item in facts),
                max(_span(item["after"], item["before"]) for item in facts),
            )
            recoveries.append((score, query, facts, None))
    return max(regressions, key=lambda item: item[0], default=None), max(
        recoveries, key=lambda item: item[0], default=None,
    )


def _render_case(lines, label, case, ranking):
    if case is None:
        lines.extend([f"**{label}**　此子集未找到符合条件的实例。", ""])
        return
    _, query, facts, pair = case
    lines.extend([
        f"**{label}**　`{query.query_id}`（`{query.question_type}`）",
        "",
        f"> {_md(query.question)}",
        "",
        f"标注答案为「{_md(query.answer)}」。",
        "",
        "| 必要事实 | 来源文档 | RRF 名次 | 重排后名次 | 答案字符串 |",
        "| --- | --- | ---: | ---: | --- |",
    ])
    for item in facts:
        fact, doc = item["evidence"], item["document"]
        match = ("无可用字面命中", "来源正文命中", "标注事实文本命中")[item["signal"]]
        lines.append(
            f"| {_md(fact.evidence_text)} | `{doc.document_id}` {_md(doc.title)} | "
            f"{_show_rank(item['before'])} | {_show_rank(item['after'])} | {match} |"
        )
    lines.append("")
    if pair:
        answer, lost = pair
        lines.extend([
            f"`{answer['document'].document_id}` 含标注答案字符串，名次由 "
            f"{_show_rank(answer['before'])} 变为 {_show_rank(answer['after'])}。"
            f"另一必要来源 `{lost['document'].document_id}` 由 "
            f"{_show_rank(lost['before'])} 变为 {_show_rank(lost['after'])}。",
            "",
        ])
    top_three = ", ".join(f"`{row['document_id']}`" for row in ranking[query.query_id]["ranking"][:3])
    lines.extend([f"重排前三篇为 {top_three}。", ""])


def _terms(text):
    return {word for word in re.findall(r"\b[a-z]{4,}\b", _fold(text)) if word not in STOP_WORDS}


def _null_example(query_map, documents, ranking_rows):
    selected = None
    for query_id in sorted(ranking_rows):
        query = query_map[query_id]
        doc = documents[ranking_rows[query_id]["ranking"][0]["document_id"]]
        terms = _terms(query.question)
        for match in re.finditer(r"[^.!?]+[.!?]*", doc.text):
            text = match.group().strip()[:360]
            overlap = len(terms & _terms(text))
            if selected is None or overlap > selected[0]:
                selected = (overlap, query, doc, text)
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=DEVELOPMENT_SPLITS, help="default: largest available complete development cohort")
    args = parser.parse_args()
    results = _load_development_results(OUTPUT / "reranking_results.jsonl")
    evidence = _load_evidence_rows(OUTPUT / "evidence_rank_changes.csv", results)
    dataset = load_dataset(ROOT / "data" / "raw" / "multihop_rag")
    documents = {doc.document_id: doc for doc in dataset.documents}
    queries = {q.query_id: q for q in dataset.queries}
    splits = json.loads((OUTPUT / "splits.json").read_bytes())

    available = {key for key, rankings in results.items() if set(evidence[key]) == set(rankings)}
    complete = {key for key in available if set(results[key]) == set(splits[key[0]])}
    default_split = next(
        (split for split in reversed(DEVELOPMENT_SPLITS) if any(key[0] == split for key in complete)),
        None,
    )
    split = args.split or default_split
    if not split or not any(key[0] == split for key in available):
        parser.error("No matching scored development cohort. Run the evaluation script first.")

    lines = [
        "# HopRAG 第五轮 · 必要证据升降案例", "",
        f"以下使用 `{split}` 的已评分开发样本。",
        "回退指原 RRF Top 10 齐备、重排后缺至少一条。恢复指原 RRF Top 10 缺证据、重排后齐备。",
        "案例按是否出现两个来源名次反向变化、答案字符串命中强度和名次差确定性选取。",
        "答案字符串在 NFKC、大小写折叠和空白合并后按字面匹配。它只用于挑选案例，无法标注直接答案或桥接角色，也不进入排序。",
        "",
    ]
    for key in sorted(key for key in available if key[0] == split):
        ranking_rows = results[key]
        cohort_hash = sha256("\n".join(sorted(ranking_rows)).encode()).hexdigest()
        lines.extend([
            f"## {key[1]} · {key[2]}",
            "",
            f"已评分 {len(ranking_rows)} / {len(splits[split])} 题，查询集合摘要 `{cohort_hash[:12]}`。",
            "",
        ])
        regression, recovery = _case_candidates(queries, documents, ranking_rows, evidence[key])
        _render_case(lines, "回退案例", regression, ranking_rows)
        _render_case(lines, "恢复案例", recovery, ranking_rows)

    null_keys = sorted(key for key in results if key[0] == NULL_SPLIT)
    if null_keys:
        lines.extend([
            "## 无答案开发集候选", "",
            "以下摘录按题面和首位文档正文的非停用词重合选取。摘录用于人工检查题面要求的关系，分数和词项重合无法判定可回答性。",
            "",
        ])
        for key in null_keys:
            _, query, doc, excerpt = _null_example(queries, documents, results[key])
            lines.extend([
                f"**{key[1]} · {key[2]} · `{query.query_id}`**",
                "",
                f"> {_md(query.question)}", "",
                f"首位候选 `{doc.document_id}`　{_md(doc.title)}。", "",
                f"> {_md(excerpt)}", "",
            ])
    lines.extend([
        r"复现命令为 `& .\.venv\Scripts\python.exe -B -m analysis.reranking.case_examples`。",
        "",
    ])
    destination = OUTPUT / "RERANKING_CASES.md"
    temporary = destination.with_suffix(".md.tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    temporary.replace(destination)
    print(f"Wrote {destination.name} from {split}")


if __name__ == "__main__":
    main()
