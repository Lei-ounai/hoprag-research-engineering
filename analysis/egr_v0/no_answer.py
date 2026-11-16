"""Reproduce the separate 301-question diagnostic without answer annotations."""

from collections import Counter, defaultdict
from pathlib import Path
import csv
import re

from analysis.besa_v0.experiment import read, rows
from analysis.ege_v0.run import cohorts


OUT = Path(__file__).resolve().parent
RUN = OUT / "run"
METHOD = re.compile(r"E([0-5])(?:-B([1-3]))?$")
RULES = {
    "date_conflict": "时间冲突",
    "entity_role_conflict": "实体角色冲突",
    "relation_mismatch": "关系冲突",
    "source_mismatch": "媒体来源冲突",
    "polarity_conflict": "肯定与否定冲突",
}
PRIOR_CASES = {
    ("q-0014", "doc-0526"): ("EU AI", "题目要求 Financial Times 报道的 2021 年欧洲专利数量和 Bloomberg 报道的欧洲专利局诉讼。TechCrunch 候选讨论 AI Act 预计于 2026 年生效。"),
    ("q-0113", "doc-0009"): ("empty seats", "题目要求 Ron DeSantis 的教育政策和竞选策略。纽约时报候选报道 MLB 季后赛主场，媒体一致但主题及关系不同。"),
    ("q-1326", "doc-0168"): ("Rugby World Cup", "题目要求 2011 年 ICC 板球赛事。候选是 2023 年橄榄球世界杯。第八轮属于二跳路径。"),
    ("q-1451", "doc-0404"): ("Walmart, Target", "题目要求科技股未平仓合约与产品发布的关系。候选报道零售商比价政策。第八轮属于二跳路径。"),
}
CURRENT_CONFIRMED = {
    ("q-1797", "doc-0117"): "TechCrunch 周报分别讨论 OpenAI 开发者大会、Apple CEO Tim Cook 的 AI 发言以及 Bumble/Slack 的 CEO 更替。题面要求《纽约时报》和《卫报》，并要求将同一 CEO 连接到开创性 AI 和参议院数据隐私听证。父文档未提供这条关系，属于媒体和角色关系均不满足的替换。",
}


def _method_key(name):
    matched = METHOD.fullmatch(name)
    if not matched:
        return 9, 9, name
    arm, budget = matched.groups()
    return int(arm), int(budget or 0), name


def _escaped(text, max_chars=210):
    compact = " ".join(str(text).split())
    return (compact if len(compact) <= max_chars else compact[:max_chars - 1] + "…").replace("|", "\\|")


def _bool(value):
    return str(value).strip().casefold() in {"true", "1", "yes"}


def _new_rows(log, ids):
    result = defaultdict(dict)
    for row in rows(log):
        if row["split"] != "null":
            continue
        qid, method = row["query_id"], row["method"]
        if qid not in ids or qid in result[method]:
            raise ValueError(f"Unexpected null method row {method}/{qid}")
        if len(row["top10"]) != 10 or len(set(row["top10"])) != 10:
            raise ValueError(f"Invalid null Top 10 {method}/{qid}")
        if row["top10"][:7] != row["baseline_top10"][:7]:
            raise ValueError(f"Protected head changed for {method}/{qid}")
        result[method][qid] = row
    if not {"E0", "E1", "E2"} <= result.keys() or any(set(group) != ids for group in result.values()):
        raise ValueError("Incomplete null ranking data")
    return result


def _collect_witnesses(path, ids):
    result = defaultdict(list)
    for row in rows(path):
        if row["query_id"] in ids:
            result[(row["query_id"], row["document_id"])].append(row)
    if {qid for qid, _ in result} != ids:
        raise ValueError("Incomplete null witness cohort")
    return result


def _collect_gates(path, ids, methods):
    result = defaultdict(dict)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            qid, method, doc = (row[key] for key in ("query_id", "method", "document_id"))
            if row["split"] != "null":
                continue
            key = qid, doc
            if qid not in ids or method not in methods or key in result[method]:
                raise ValueError(f"Duplicate or extra null gate evaluation: {method}/{key}")
            result[method][key] = row
    return result


def _describe_new(logs, witnesses):
    diagnostics = {}
    for method, records in sorted(logs.items(), key=lambda kv: _method_key(kv[0])):
        changed, insertions, explicit_risk, uncertain, known, replaced_ids = 0, 0, 0, 0, [], set()
        reasons = Counter()
        for qid, row in records.items():
            top10, baseline = row["top10"], row["baseline_top10"]
            novel = set(top10) - set(baseline)
            changed += bool(novel)
            insertions += len(novel)
            claimed = {r["candidate_document_id"] for r in row["replacements"]}
            if _method_key(method)[0] >= 3 and claimed != novel:
                raise ValueError(f"Replacement log and actual change differ for {method}/{qid}")
            for d in novel:
                key = (qid, d)
                replaced_ids.add(key)
                blocks = witnesses.get(key, ())
                if key in PRIOR_CASES or key in CURRENT_CONFIRMED:
                    known.append(key)
                if any(w["status"] == "supported" for w in blocks):
                    continue
                found = {reason for row in blocks for reason in row.get("reasons", ()) if reason in RULES}
                if found:
                    explicit_risk += 1
                    reasons.update(found)
                else:
                    uncertain += 1
        diagnostics[method] = {
            "changed": changed, "insertions": insertions, "explicit_risk": explicit_risk, "uncertain": uncertain,
            "confirmed_examples": sorted(known), "documents": sorted(replaced_ids), "signals": reasons,
            "fallbacks": sum(bool(row.get("fallback_reason")) for row in records.values()),
        }
    return diagnostics


def _block_summary(blocks):
    if not blocks:
        return "本轮候选未见评分记录"
    statuses = Counter(row["status"] for row in blocks)
    counts = "、".join(f"{key} {count}" for key, count in sorted(statuses.items()))
    reasons = sorted({reason for row in blocks for reason in row.get("reasons", ())})
    return counts + (f"。信号 {', '.join(reasons)}" if reasons else "")


def _best_block(records):
    return max(records, key=lambda row: (
        row["status"] == "supported",
        row["status"] == "conflict",
        row.get("quality", 0.0),
        row.get("original_question_match", 0.0),
        row["chunk_id"],
    ), default=None)


def _case_lines(logs, witnesses, diagnostics):
    lines = []
    for (qid, doc), (term, note) in PRIOR_CASES.items():
        example = logs["E0"][qid]
        blocks = witnesses.get((qid, doc), ())
        entered = [method for method, count in diagnostics.items() if (qid, doc) in count["documents"]]
        block = _best_block(blocks)
        lines.extend([
            f"### {qid} · {doc}", "", f"> {example['question']}", "",
            f"- 第八轮线索「{term}」。{note}",
            f"- 本轮核验。{_block_summary(blocks)}。",
            f"- 本轮由候选替换进入前十的方案。{', '.join(entered) if entered else '无'}。",
        ])
        if block and block.get("evidence_excerpt"):
            company = block.get("source_metadata", {}).get("source") or "来源未标明"
            lines.append(f"- 本轮见证 {block['chunk_id']} · {_escaped(company, 50)} · {_escaped(block.get('document_date') or '日期未标明', 40)}。{_escaped(block['evidence_excerpt'])}")
        lines.append("")
    return lines


def _write_report(logs, witnesses, gates, diagnostics, selected):
    count = len(logs["E0"])
    lines = [
        "# EGR-v0 无答案题独立分析", "",
        f"在 {count} 道冻结无答案题上重放已选参数。题面、候选、评分和证据见证已经产生后，"
        "才按既有题目 ID 读取这组记录。离线诊断未读取答案或 Gold Evidence。"
        "各题本来都可能检索到部分相关资料，入选文档本身不能证明题目可回答。", "",
        "## 最终 Top 10 的替换", "",
        "| 对照 | Top 10 有变化的题数 | 新入选文档次数 | 未核验支持且有明确冲突信号 | 未核验支持且原因不明或证据不足 | 人工核实错配次数 | 执行回退 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method, info in diagnostics.items():
        lines.append(
            f"| {method} | {info['changed']} | {info['insertions']} | {info['explicit_risk']} | "
            f"{info['uncertain']} | {len(info['confirmed_examples'])} | {info['fallbacks']} |"
        )
    gated = {method: group for method, group in diagnostics.items() if _method_key(method)[0] >= 3}
    losing = {pair for group in gated.values() for pair in group["confirmed_examples"]}
    gated_pairs = {pair for group in gated.values() for pair in group["documents"]}
    selected_result = diagnostics[selected]
    ablation_name = f"E5-B{selected.rsplit('-B', 1)[-1]}"
    ablation = diagnostics[ablation_name]
    lines.extend([
        "",
        f"Tune 冻结方案 {selected} 在 {selected_result['changed']} 道题替换 {selected_result['insertions']} 个位置。"
        f"同一预算增加一跳候选的 {ablation_name} 在 {ablation['changed']} 道题替换 {ablation['insertions']} 个位置。",
    ])
    lines.extend([
        "",
        "表中的冲突信号是日期、实体角色、关系、来源或极性的规则命中，且该候选所有已评分分块均没有产生受支持见证。"
        "它用于排序风险排查。分块预算与确定性规则都可能漏掉真正证据。人工确认列只统计已经依据原文核实的反例。"
        "同一题和候选可能在多个方案里重复出现。", "",
        f"门控方案共出现 {len(gated_pairs)} 个不同的题目–新入选文档对。按原文确认的错误替换 {len(losing)} 个。"
        + ("因没有实际替换，错误替换为 0。" if not gated_pairs else "本组没有完整的候选关系标签，不能把剩余替换都判成正确或错误。"),
        "",
        "## 门控通过与拒绝", "",
        "| 方案 | 候选判断次数 | 质量通过 | 新颖性通过 | 可比收益通过 | 已替换 | 主要筛除原因（计数） |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ])
    for method in sorted(gated, key=_method_key):
        values = list(gates.get(method, {}).values())
        rejected = Counter(row["reason"] for row in values if row["status"] != "replaced")
        formatted = "、".join(f"{reason} {num}" for reason, num in sorted(rejected.items(), key=lambda kv: (-kv[1], kv[0]))[:5])
        lines.append(
            f"| {method} | {len(values)} | {sum(_bool(r['quality_pass']) for r in values)} | "
            f"{sum(_bool(r['novelty_pass']) for r in values)} | {sum(_bool(r['margin_pass']) for r in values)} | "
            f"{sum(r['status'] == 'replaced' for r in values)} | {formatted or '—'} |"
        )
    ungated_expected = set(diagnostics["E2"]["documents"])
    e4_decisions = gates.get(selected, {})
    absent = ungated_expected - set(selected_result["documents"])
    quality_denied = [key for key in absent if key in e4_decisions
                      and str(e4_decisions[key]["quality_pass"]).strip().casefold() == "false"]
    banned = Counter(e4_decisions[qid, doc]["reason"] for qid, doc in quality_denied)
    relevant = "、".join(f"{reason} {total}" for reason, total in banned.most_common(5))
    lines.extend([
        "",
        f"在共享三块的 E2 对照中，{len(ungated_expected)} 个题目–候选已进入 Top 10。"
        f"其中 {len(absent)} 个没有出现在 {selected} 新入选集合，{len(quality_denied)} 个明确未过质量门槛。"
        + (f"最常见质量原因包括 {relevant}。" if relevant else ""),
    ])
    signals = Counter()
    for blocks in witnesses.values():
        # Count a conflict once per parent and question, even when all scored blocks expose it.
        signals.update({r for block in blocks for r in block.get("reasons", ()) if r in RULES})
    lines.extend([
        "", f"从 {len(witnesses)} 个评分过的题目–父文档对收集分块信号。"
        "同一对可触发不同维度，不代表每个分块均有冲突。"
        "下表描述信号覆盖。事件时间不清与来源发布日期不符在逐块记录中分别留痕。", "",
        "| 核验维度 | 有显式冲突信号的题目–父文档对 |",
        "| --- | ---: |",
        *(f"| {label} | {signals[name]} |" for name, label in RULES.items()),
        "", "仅有的日期冲突出现在 q-1326 的 doc-0181。该句报告美国大学橄榄球 2022 年 Big Ten 冠军赛，"
        "题面日期是 2011 年 ICC 板球赛事。这是关系和项目已漂移后触发的年份不同，不能单凭它估计同一事件的日期过滤效果。",
        "", "## 历史错误路径复查", "",
        "以下反例先前有原文核查。二跳案例用于说明第九轮采用的一跳独立分支对观测范围的影响。", "",
    ])
    lines.extend(_case_lines(logs, witnesses, diagnostics))
    rule_examples = []
    for reason in ("date_conflict", "entity_role_conflict", "relation_mismatch"):
        for qid, doc in sorted(quality_denied):
            matching = [w for w in witnesses.get((qid, doc), ()) if reason in w.get("reasons", ())]
            if matching:
                rule_examples.append((reason, qid, doc, _best_block(matching)))
                break
    if rule_examples:
        lines.extend(["## E2 已入选但被 E4 质量门槛拦截的样本", "",
                      "这些案例由规则信号挑选，可从原文进一步检查真假冲突。", ""])
        for reason, qid, doc, block in rule_examples:
            original = next(r for r in logs["E2"][qid]["replacements"] if r["candidate_document_id"] == doc)
            lines.extend([
                f"### {qid} · {RULES[reason]}", "", f"> {logs['E2'][qid]['question']}", "",
                f"- E2 在第 {original['position']} 位引入 {doc} · {_escaped(original.get('new_candidate_title', ''))}。",
                f"- 核验 {block['chunk_id']} 以 {', '.join(block['reasons'])} 拒绝该块。{_escaped(block['evidence_excerpt'])}",
            ])
            if (qid, doc) in CURRENT_CONFIRMED:
                lines.append(f"- 父文档复查。{CURRENT_CONFIRMED[qid, doc]}")
            lines.append("")
    if gated_pairs:
        lines.extend(["## 门控后实际出现的替换样本", ""])
        for qid, doc in sorted(gated_pairs)[:12]:
            methods = [m for m, item in gated.items() if (qid, doc) in item["documents"]]
            replacement = next(r for r in logs[methods[0]][qid]["replacements"]
                               if r["candidate_document_id"] == doc)
            block = next((r for r in witnesses.get((qid, doc), ())
                         if r["chunk_id"] == replacement["champion_chunk_id"]), None)
            if not block:
                raise ValueError(f"No recorded scored witness for null replacement: {qid}/{doc}")
            lines.extend([
                f"- {qid} · {doc} · {', '.join(methods)}。"
                f"位次 {replacement['position']}，替换 {replacement['incumbent_document_id']}。"
                f"分差 {replacement['score_margin']:.3f}，见证 {block['chunk_id']}。{_escaped(block['evidence_excerpt'])}",
            ])
        lines.append("")
    cost_path = RUN / "null_performance.json"
    if cost_path.exists():
        cost = read(cost_path)
        lines.extend([
            "## 成本", "",
            f"无答案队列实调度 {cost['actual_pairs']} 个 Jina 对，历史精确命中 {cost['historical_hits']}，"
            f"本轮缓存命中 {cost['local_hits']}，新增推理 {cost['new_pairs']}。"
            f"纯推理 {cost['inference_seconds']:.2f} 秒，评分进程墙钟 {cost['wall_seconds']:.2f} 秒。"
            "新增 BGE Embedding 为 0。", "",
        ])
    lines.extend([
        "在正式项目根目录重放已保存的替换、分块见证和门控记录。", "",
        "~~~powershell", "& .\\.venv\\Scripts\\python.exe -B -m analysis.egr_v0.no_answer", "~~~", "",
        "[逐题替换记录](replacement_log.jsonl) · [原始分块见证](evidence_witnesses.jsonl) · [门控决定](gate_decisions.csv)", "",
    ])
    (OUT / "no_answer_analysis.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main():
    _, _, ids = cohorts()
    if len(ids) != 301:
        raise ValueError("Expected frozen null cohort")
    logs = _new_rows(OUT / "replacement_log.jsonl", ids)
    witnesses = _collect_witnesses(OUT / "evidence_witnesses.jsonl", ids)
    gates = _collect_gates(OUT / "gate_decisions.csv", ids, logs)
    diagnostics = _describe_new(logs, witnesses)
    frozen = read(OUT / "frozen_config.json")
    _write_report(logs, witnesses, gates, diagnostics, frozen["selected_arm"])
    for method, report in diagnostics.items():
        print(method, report["changed"], report["insertions"], report["explicit_risk"],
              len(report["confirmed_examples"]), flush=True)


if __name__ == "__main__":
    main()

