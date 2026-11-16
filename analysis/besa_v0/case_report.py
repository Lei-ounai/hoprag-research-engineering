"""Reproduce development-only BESA case reports from saved results and scores."""

from collections import Counter
from collections.abc import Iterable

from .pipeline import assemble, candidate_inputs
from .experiment import ARCHIVED, OUT, cache_source, development_ids, development_inputs, read, rows
from .run import OUTPUT, _chunks
from .scoring import DenseVectors
from .selector import _parent_cosines


REPORT = OUT / "BESA_CASES.md"
MAX_CASES = 5


def _short(text, limit=185):
    value = " ".join(text.split()).replace("|", r"\|")
    return value if len(value) <= limit else value[:limit].rstrip() + "…"


def _id_list(values):
    return " → ".join(f"{value}" for value in values) or "—"


def _rank(value):
    return str(value) if value is not None else "—"


def _number(value):
    return f"{value:.3f}" if value is not None else "—"


def _entry_index(records: Iterable[dict], field: str):
    result = {}
    for record in records:
        key = record[field]
        if key in result:
            raise ValueError(f"Repeated {field}: {key}")
        result[key] = record
    return result


def _source_changes(row):
    return [fact for fact in row["facts"] if fact["in_original_top10"] != fact["in_top10"]]


def _publishable_cases(records, status):
    matching = [row for row in records if (row["baseline_complete"], row["complete"]) == status]
    return sorted(matching, key=lambda row: (-len(_source_changes(row)), row["query_id"]))


def _fallback_samples(records):
    candidates = sorted(
        (row for row in records if not row["planner_accepted"]),
        key=lambda row: (row["complete"], row["query_id"]),
    )
    chosen = {}
    for row in candidates:
        chosen.setdefault(row["fallback_reason"], row)
    sample = list(chosen.values())[:MAX_CASES]
    sample_ids = {row["query_id"] for row in sample}
    if len(sample) < MAX_CASES:
        sample.extend(row for row in candidates if row["query_id"] not in sample_ids)
    return sample[:MAX_CASES]


def _member_losses(row):
    original_set = set(row["original_top10"])
    final_set = set(row["final_top10"])
    return ([doc for doc in row["original_top10"] if doc not in final_set],
            [doc for doc in row["final_top10"] if doc not in original_set])


def _affected_cases(comparisons, scored, split):
    d_labels = {label for label in split["frozen"] if label.startswith("D-")}
    only_d = list(row for row in comparisons if row["method"] in d_labels)
    eligible = [row for row in only_d if row["planner_accepted"] and not row["fallback_reason"]]
    needed_ids = {row["query_id"] for row in eligible}
    raw = {row["query_id"]: row for row in scored if row["query_id"] in needed_ids}
    vectors = DenseVectors.load(cache_source(), _chunks())
    prepared = {
        pool.query_id: candidate_inputs(pool, baseline, witnesses, raw[pool.query_id], vectors)
        for pool, baseline, witnesses in development_inputs(needed_ids)
    }
    changes, selected_penalty_count, changed_order_count = [], 0, 0
    for row in eligible:
        query_id = row["query_id"]
        without_penalty = {k: v for k, v in split["frozen"][row["method"]].items() if k != "config_id"}
        without_penalty["repeat_weight"] = 0.0
        counterfactual, _, _ = assemble(prepared[query_id], **without_penalty)
        top10_without = counterfactual["ranking"][:10]
        if any(choice["redundancy"] > 0 for choice in row["choices"]):
            selected_penalty_count += 1
        changed_order_count += top10_without != row["final_top10"]
        if set(top10_without) != set(row["final_top10"]):
            changes.append((row, top10_without))
    return vectors, changes, len(eligible), selected_penalty_count, changed_order_count


def _mmr_examples(comparisons, vectors, limit, omit):
    mmr = {
        row["query_id"]: row for row in comparisons
        if row["method"] == "B" and set(row["original_top10"]) != set(row["final_top10"])
        and row["query_id"] not in omit
    }
    eligible = []
    for pool, order, witnesses in development_inputs(mmr):
        missing_gold = {fact["source_document_id"] for fact in _source_changes(mmr[pool.query_id])
                        if not fact["in_top10"]}
        row = mmr[pool.query_id]
        removed, added = _member_losses(row)
        rank_index = {doc: i for i, doc in enumerate(order)}
        candidates = {doc.document_id: doc for doc in pool.candidates}
        cosines = _parent_cosines(order, vectors.parent_vectors(pool))
        for chosen in row["choices"]:
            promoted = chosen["document_id"]
            if promoted not in added:
                continue
            previous = row["final_top10"][:chosen["position"] - 1]
            if not previous:
                continue
            for demoted in removed:
                if rank_index[demoted] >= rank_index[promoted]:
                    continue
                neighbor = max(previous, key=lambda doc: cosines[rank_index[demoted], rank_index[doc]])
                similarity = float(cosines[rank_index[demoted], rank_index[neighbor]])
                delta = similarity - chosen["redundancy"]
                if delta > 0:
                    eligible.append({
                        "row": row, "promoted": promoted, "demoted": demoted, "neighbor": neighbor,
                        "similarity": similarity, "promoted_redundancy": chosen["redundancy"],
                        "delta": delta, "position": chosen["position"],
                        "missed_gold": demoted in missing_gold,
                        "demoted_winner": witnesses[demoted],
                        "adjacent_title": candidates[neighbor].title,
                        "adjacent_source": candidates[neighbor].article_source,
                        "adjacent_winner": witnesses[neighbor],
                    })
    best = {}
    priority = sorted(
        eligible,
        key=lambda item: (
            not item["missed_gold"], -item["delta"], item["row"]["query_id"],
            item["promoted"], item["demoted"],
        ),
    )
    for item in priority:
        best.setdefault(item["row"]["query_id"], item)
    return list(best.values())[:limit]


def _show_needs(lines, row):
    lines.extend(["", "**需求与预算**", ""])
    if not row["plan"]["needs"]:
        lines.append(f"规划未产生可检索需求。规划拒绝原因为 {row['plan']['fallback_reason']}。")
        return
    if not row["pair_count"]:
        lines.append("该对照没有执行需求评分。以下需求只记录同题规划器输出。")
        lines.append("")
    for need in row["plan"]["needs"]:
        lines.append(f"- {need['need_id']}　{need['need_text']}")
    lines.extend(["", f"本方法使用 {row['pair_count']} 个分块评分对，{row['parents_scored']} 篇父文档有分数。"])


def _show_fact_rows(lines, row, doc_lookup):
    lines.extend([
        "", "**必要来源与片段**", "",
        "| 文档与媒体 | P5-B 名次 | 当前名次 | 可匹配的原始块 | 入选保留块 |",
        "| --- | ---: | ---: | --- | --- |",
    ])
    for fact in row["facts"]:
        doc_id = fact["source_document_id"]
        doc = doc_lookup[doc_id]
        witness = _id_list(fact["retained_chunk_ids"])
        literal = _id_list(fact["matching_original_chunk_ids"])
        lines.append(
            f"| {doc_id} {_short(doc['title'], 85)} · {_short(doc['article_source'] or '未知', 45)} | "
            f"{_rank(fact['baseline_rank'])} | {_rank(fact['final_rank'])} | {literal} | {witness} |"
        )
    lines.append("")
    for fact in row["facts"]:
        match = "保留块含完整字面事实" if fact["literal_fact_in_retained_chunk"] else "保留块未定位该完整字面事实"
        lines.append(f"- {fact['evidence_id']}，{match}。{_short(fact['evidence_text'])}")


def _show_supports(lines, row, docs, chunks, archives):
    removed, added = _member_losses(row)
    affected = {fact["source_document_id"] for fact in row["facts"]} | set(removed) | set(added)
    cells = [
        (need, doc_id, cell) for need, support in row["supports"].items()
        for doc_id, cell in support.items() if doc_id in affected
    ]
    lines.extend(["", "**评分见证**", ""])
    if cells:
        lines.extend([
            "| 需求 | 父文档 | 原始 Jina | 归一化支持 | 胜出块 |",
            "| --- | --- | ---: | ---: | --- |",
        ])
        for need, doc_id, cell in sorted(cells):
            lines.append(f"| {need} | {doc_id} | {_number(cell['raw_score'])} | "
                         f"{_number(cell['support'])} | {cell['chunk_id']} |")
        for need, doc_id, cell in sorted(cells, key=lambda item: (-item[2]["support"], item[0], item[1])):
            if doc_id in added and cell["support"] > 0:
                lines.extend(["", f"新入选见证 {need} / {cell['chunk_id']}。{_short(chunks[cell['chunk_id']])}"])
                break
    else:
        lines.append("该结果没有需要级支持分。列出历史整题 Jina 胜出块供核对。")
    candidates = [doc for doc in archives if doc["document_id"] in affected]
    targets = candidates[:3]
    if targets:
        lines.append("")
    for doc in targets:
        doc_id = doc["document_id"]
        chunk_id = doc["winning_chunk_id"]
        source = docs[doc_id]["article_source"] or "未知"
        lines.append(f"- 历史整题 Jina {doc_id} · {_short(source)} · {chunk_id} · {_number(doc['score'])}")


def _common(lines, row, doc_lookup, chunks, historical, cohort, label):
    query_id = row["query_id"]
    lines.extend([
        "", f"### {query_id} · {label} · {cohort[query_id]}", "", row["question"],
        "", f"- P5-B Top 10　{_id_list(row['original_top10'])}",
        f"- 当前 Top 10　{_id_list(row['final_top10'])}",
        f"- 当前诊断　{row['failure_diagnostic']}",
    ])
    _show_needs(lines, row)
    _show_fact_rows(lines, row, doc_lookup)
    _show_supports(lines, row, doc_lookup, chunks, historical[query_id])


def _show_transition(lines, row):
    lost, gained = _member_losses(row)
    changes = _source_changes(row)
    lines.extend([
        "", f"挤出 {_id_list(lost)}。新入选 {_id_list(gained)}。",
        "",
    ])
    for fact in changes:
        doc_id = fact["source_document_id"]
        change = "进入" if fact["in_top10"] else "离开"
        lines.append(f"- {doc_id} {change} Top 10，名次 {_rank(fact['baseline_rank'])} → {_rank(fact['final_rank'])}。")
        if not fact["in_top10"] and fact["matching_original_chunk_ids"]:
            scored = "预算内至少看到一块" if fact["fact_chunk_was_scored"] else "预算内没有对这些块评分"
            lines.append(f"- {fact['evidence_id']} 的原始块含该字面事实，{scored}。")
    lines.extend(["", "名次、入选和字面定位来自记录。相关分只能说明排序信号，单凭这些变化无法证明候选支持整条推理。"])


def main():
    dev_ids = set(development_ids())
    selected = list(rows(OUTPUT / "selected_results.jsonl"))
    comparisons = list(rows(OUTPUT / "comparisons.jsonl"))
    selected_by_id = _entry_index(selected, "query_id")
    if set(selected_by_id) != dev_ids or any(row["query_id"] not in dev_ids for row in comparisons):
        raise ValueError("All source rows must belong to the full original development split")
    docs = _entry_index(rows(OUT.parent / "reranking" / "candidate_documents.jsonl"), "document_id")
    chunks = {chunk["chunk_id"]: chunk["text"] for doc in docs.values() for chunk in doc["chunks"]}
    archived = {
        row["query_id"]: row["jina_ranking"] for row in rows(ARCHIVED)
        if row["query_id"] in dev_ids and row["pool_id"] == "P5"
    }
    split = read(OUTPUT / "development_split.json")
    cohort = {qid: label for label in ("tune", "validation") for qid in split[label]}
    freeze = read(OUTPUT / "tune_freeze.json")
    recovered = _publishable_cases(selected, (False, True))
    regressed = _publishable_cases(selected, (True, False))
    fallbacks = _fallback_samples(selected)
    vectors, differences, compared, positive, order_changed = _affected_cases(
        comparisons, rows(OUTPUT / "development_scores.jsonl"), freeze,
    )
    meaningful_d = {}
    for row, no_penalty in differences:
        meaningful_d.setdefault(row["query_id"], (row, no_penalty))
    repeat_examples = list(meaningful_d.values())[:MAX_CASES]
    mmr_examples = _mmr_examples(
        comparisons, vectors, MAX_CASES - len(repeat_examples), set(meaningful_d),
    )

    lines = [
        "# BESA-v0 开发集案例", "",
        f"使用 Tune 冻结方案 {freeze['selected_arm']}。范围为原有 451 道开发题，"
        "证据和文档来源只在离线阶段用于分析。报告中的胜出块是打分见证，完整字面匹配仅表示能定位标注事实。",
        "",
        "恢复和回退按变化的必要来源数降序、Query ID 升序选取，各最多 5 道。"
        "规划回退优先覆盖不同规则失败原因，同一原因优先选择基线仍不完整的问题。"
        "同需求冗余采用同参数仅归零惩罚的对照。",
        "",
        f"## 恢复案例（{len(recovered)} 道，展示 {min(MAX_CASES, len(recovered))} 道）",
    ]
    for row in recovered[:MAX_CASES]:
        _common(lines, row, docs, chunks, archived, cohort, "恢复")
        _show_transition(lines, row)
    lines.extend(["", f"## 新增回退案例（{len(regressed)} 道，展示 {min(MAX_CASES, len(regressed))} 道）"])
    if not regressed:
        lines.extend(["", "冻结方案在这批开发题中未观察到新增回退。"])
    for row in regressed[:MAX_CASES]:
        _common(lines, row, docs, chunks, archived, cohort, "新增回退")
        _show_transition(lines, row)
    lines.extend(["", f"## Planner 回退案例（展示 {len(fallbacks)} 道）"])
    reasons = Counter(row["fallback_reason"] for row in selected if not row["planner_accepted"])
    lines.extend(["", "回退原因计数。 " + "，".join(f"{label} {total}" for label, total in reasons.most_common()) + "。"])
    for row in fallbacks:
        _common(lines, row, docs, chunks, archived, cohort, "Planner 回退")
        lines.extend(["", "选择器保留整条 P5-B 排名，Top 10 前后完全一致。"])

    lines.extend([
        "", "## 冗余诊断", "",
        f"对三个冻结 D 预算做逐题反事实，仅将同需求冗余权重归零。{compared} 个实际运行的题目×预算组合中，"
        f"{positive} 个有入选文档记录到正惩罚，{order_changed} 个 Top 10 顺序变化，"
        f"{len(differences)} 个 Top 10 集合变化。入选日志与反事实分别反映入选项上的信号和最终结果差异。",
    ])
    for row, no_penalty in repeat_examples:
        _common(lines, row, docs, chunks, archived, cohort, f"同需求惩罚对照 {row['method']}")
        lines.extend(["", f"关闭该惩罚、其余参数不变时 Top 10 为 {_id_list(no_penalty)}。"])
    if mmr_examples:
        lines.extend([
            "", "观察到 0 道受同需求近重复惩罚影响的题。另列全篇 MMR 的多样性机制对照，"
            "优先选择被挤出必要来源的案例，再按落选项与入选项对已选集合的最大相似度差排序。"
            "父向量相似度无法确认同源、同需求、同事实或文本重复。",
        ])
    for item in mmr_examples:
        row = item["row"]
        qid = row["query_id"]
        besa = selected_by_id[qid]
        _common(lines, row, docs, chunks, archived, cohort, "MMR 多样性对照")
        lines.extend([
            "", f"BESA C-B30 Top 10　{_id_list(besa['final_top10'])}。",
            f"被 MMR 挤出的 {item['demoted']} 在 C-B30 中的完整名次为 "
            f"{_rank(besa['full_ranking'].index(item['demoted']) + 1)}。",
            "", f"- 落选文章 {item['demoted']} · {_short(docs[item['demoted']]['title'])} · "
            f"{_short(docs[item['demoted']]['article_source'] or '未知')}。"
            f"历史见证块 {item['demoted_winner']}。",
            f"- 落选块节选。{_short(chunks[item['demoted_winner']])}",
            "", f"第 {item['position']} 位选入 {item['promoted']}，原先优先级较高的 {item['demoted']} 最终未入前十。"
            f"该落选文档与此前已选 {item['neighbor']} 的父向量余弦为 {_number(item['similarity'])}，"
            f"入选文档对当时已选集合的最高余弦为 {_number(item['promoted_redundancy'])}。",
            "", f"- 邻近文章 {item['neighbor']} · {_short(item['adjacent_title'])} · "
            f"{_short(item['adjacent_source'] or '未知')}。历史见证块 {item['adjacent_winner']}。",
            f"- 邻近块节选。{_short(chunks[item['adjacent_winner']])}",
        ])
        _show_transition(lines, row)
        if besa["supports"]:
            lines.extend(["", "下列支持值来自同题 C-B30。"])
            _show_supports(lines, besa, docs, chunks, archived[qid])
        elif besa["fallback_reason"]:
            lines.extend(["", f"C-B30 的运行时回退原因为 {besa['fallback_reason']}。"])
    lines.extend([
        "", "## 重现", "",
        "在正式项目根目录运行下面的只读分析命令。它复用已保存模型分数和向量，仅更新本报告。",
        "", "```powershell",
        "& .\\.venv\\Scripts\\python.exe -B -m analysis.besa_v0.case_report",
        "```", "",
    ])
    temporary = REPORT.with_suffix(".md.tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    temporary.replace(REPORT)
    print(f"{freeze['selected_arm']}: recovered={len(recovered)}, regressed={len(regressed)}, "
          f"planner_examples={len(fallbacks)}, D_membership_changes={len(differences)}, "
          f"MMR_examples={len(mmr_examples)}")


if __name__ == "__main__":
    main()
