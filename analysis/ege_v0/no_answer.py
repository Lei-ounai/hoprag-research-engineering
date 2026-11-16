"""Reproduce the held-aside null-cohort diagnostic from frozen development choices."""

from collections import Counter
from math import ceil
import re
from statistics import mean

from hoprag.retrievers import _tokens

from analysis.besa_v0.experiment import read, rows, write_rows

from .expansion import _STOP, through_hop
from .ranking import ScoreBatch, fuse
from .run import DATA, OUT, cohorts, corpus, ordered_novel, winner_run


# Brands named frequently by these question templates, including brands absent from this corpus.
SOURCE_ALIASES = {
    "BBC News": ("BBC", "BBC News"),
    "BBC Sport": ("BBC Sport", "BBC Sports"),
    "Bloomberg": ("Bloomberg",),
    "Cnbc": ("CNBC",),
    "Cricbuzz": ("Cricbuzz",),
    "ESPN": ("ESPN",),
    "Financial Times": ("Financial Times", "The Financial Times"),
    "Forbes": ("Forbes",),
    "Fortune": ("Fortune",),
    "Reuters": ("Reuters",),
    "Rolling Stone": ("Rolling Stone",),
    "Sporting News": ("Sporting News", "The Sporting News"),
    "Sports Illustrated": ("Sports Illustrated",),
    "TechCrunch": ("TechCrunch",),
    "The Economic Times": ("Economic Times", "The Economic Times"),
    "The Guardian": ("The Guardian",),
    "The Hindu": ("The Hindu",),
    "The New York Times": ("New York Times", "The New York Times"),
    "The Times of India": ("Times of India", "The Times of India"),
    "The Verge": ("The Verge", "Verge"),
    "The Wall Street Journal": ("Wall Street Journal", "The Wall Street Journal", "WSJ"),
    "The Washington Post": ("Washington Post", "The Washington Post"),
    "Wired": ("Wired",),
}
YEAR = re.compile(r"(?<!\w)(?:19|20)\d{2}(?!\w)")
REVIEWED = (
    (
        "q-0014", "frozen", "doc-0526", "EU AI",
        "The EU AI Act itself likely",
        "题面询问 2021 年欧洲专利数量及欧洲专利局法律挑战。该块讨论 EU AI Act 预计在 2026 年生效，"
        "发布者为 TechCrunch，也未满足题面指定的 Financial Times 与 Bloomberg。"
    ),
    (
        "q-0113", "frozen", "doc-0009", "empty seats",
        "because of the empty seats",
        "题面要求《纽约时报》的教育政策报道及《华盛顿邮报》的 2024 竞选策略报道。"
        "该文虽然来自《纽约时报》，正文实际比较 MLB 季后赛主场，来源匹配没有解决关系错配。"
    ),
    (
        "q-1326", "two-hop", "doc-0168", "Rugby World Cup",
        "video content from the Rugby World Cup",
        "题面要求 DY Patil 体育学院、ICC 板球和 2011 年赛事。该路径从板球报道衍生，"
        "第二跳的候选正文却讨论 2023 年橄榄球世界杯。同一词出现在枢纽文章中仍能跨运动项目漂移。"
    ),
    (
        "q-1451", "two-hop", "doc-0404", "Walmart, Target",
        "Dell will price-match identical products",
        "题面要求公司产品发布与股票未平仓合约的关系。该块说的是零售商价格匹配，"
        "无法对应市场指标，也不属于题面指定的媒体。"
    ),
)


def _sources(question):
    mentions = []
    for source, aliases in SOURCE_ALIASES.items():
        for alias in aliases:
            # Capitalization discriminates publisher names from ordinary words e.g. Fortune.
            for found in re.finditer(rf"(?<!\w){re.escape(alias)}(?!\w)", question):
                mentions.append((found.start(), found.end(), source))
    # Prefer the longest mention at a given offset ("BBC Sport" over "BBC").
    boundary = 0
    selected = set()
    for start, end, source in sorted(mentions, key=lambda value: (value[0], -value[1], value[2])):
        if start >= boundary:
            selected.add(source)
            boundary = end
    return sorted(selected)


def _source_matches(document, sources):
    publisher = document.metadata.get("source")
    if not publisher:
        return False
    publisher = " ".join(_tokens(publisher))
    return any(
        publisher == normalized or publisher.startswith(normalized + " ")
        for source in sources
        for normalized in (" ".join(_tokens(source)), *(" ".join(_tokens(a)) for a in SOURCE_ALIASES[source]))
    )


def _content_terms(question, sources):
    source_terms = {word for source in sources for alias in SOURCE_ALIASES[source] for word in _tokens(alias)}
    scaffold = {"information", "concerning", "considering", "discussing", "letter", "represents", "significant",
                "recent", "report", "reports", "article", "articles", "according", "published", "sources",
                "mentioned", "detail", "details", "detailing", "highlighted", "latest", "single"}
    return {token for token in _tokens(question) if len(token) >= 3 and token.isalpha()} - _STOP - source_terms - scaffold


def _snapshot(run):
    cfg = run["config"]
    new = run["new_candidates"]
    if len(run["queries"]) > cfg["max_queries"] or len(new) > cfg["max_new_documents"]:
        raise ValueError(f"Expansion budget exceeded on {run['query_id']}")
    hops = [phase["hop"] for phase in run["phases"] if phase["queries"]]
    volumes = {str(hop): sum(row["hop"] == hop for row in new) for hop in (1, 2)}
    return {
        "queries": len(run["queries"]), "hops": run["max_hop_reached"],
        "parents": len(run["candidate_document_ids"]), "new": len(new), "new_by_hop": volumes,
        "executed_hops": hops, "bm25_seconds": run["bm25_seconds"], "wall_seconds": run["wall_seconds"],
    }


def _ranking_record(result, baseline, paths, documents, chunks, sources, years, question_terms):
    original = set(baseline[:10])
    joined = set(result.document_ids)
    retained = {
        doc: {
            "document_id": doc, "source_match": _source_matches(documents[doc], sources) if sources else None,
            "witness_chunk_id": result.witnesses.get(doc),
            "year_matches": sorted(years.intersection(YEAR.findall(chunks.get(result.witnesses.get(doc), "")))),
            "question_content_overlap": sorted(question_terms.intersection(_tokens(
                documents[doc].title + "\n" + chunks.get(result.witnesses.get(doc), "")
            ))),
        }
        for doc in result.document_ids
    }
    novel = [{**path, **retained[doc]} for doc in result.document_ids if (path := paths.get(doc)) is not None]
    return {
        "ranking": result.ranking, "top10": result.document_ids,
        "common_top10": len(joined & original), "changed_set": joined != original,
        "changed_order": result.document_ids != baseline[:10],
        "source_named": sum(row["source_match"] is True for row in retained.values()),
        "year_present": sum(bool(row["year_matches"]) for row in retained.values()),
        "new_top10": novel, "fallback_reason": result.fallback_reason,
    }


def _rows(frozen, archives, score_rows, documents, chunks):
    lexical = frozen["lexical_winner"]
    semantic = frozen.get("jina_winner")
    for item in archives:
        qid, question, baseline = item["query_id"], item["question"], item["baseline"]
        run = winner_run(item, lexical)
        one = through_hop(item["variants"][frozen["seed_variants"]["h1"]], 1)
        two = through_hop(item["variants"][frozen["seed_variants"]["h2"]], 2)
        score = score_rows.get(qid)
        paths = {row["document_id"]: row for row in run["new_candidates"]}
        sources = _sources(question)
        years = set(YEAR.findall(question))
        terms = _content_terms(question, sources)
        common = {
            "baseline_scores": {doc["document_id"]: doc["score"] for doc in item["jina"]},
            "baseline_witnesses": {doc["document_id"]: doc["winning_chunk_id"] for doc in item["jina"]},
            "expansion_witnesses": {doc: path["chunk_id"] for doc, path in paths.items()},
        }
        ranks = {}
        alternatives = {
            "P5-B": ("baseline", 0, 1.0),
            "扩展 RRF": ("rrf", 0, lexical["expansion_weight"]),
            "扩展 Anchor-5": ("anchor", 5, lexical["expansion_weight"]),
            "扩展 Anchor-7": ("anchor", 7, lexical["expansion_weight"]),
            "扩展 Anchor-8": ("anchor", 8, lexical["expansion_weight"]),
            "Tune 冻结词法": (lexical["mode"], lexical["anchor"], lexical["expansion_weight"]),
        }
        for name, (mode, anchor, weight) in alternatives.items():
            result = fuse(baseline, ordered_novel(run), mode=mode, anchor=anchor, expansion_weight=weight, **common)
            ranks[name] = _ranking_record(result, baseline, paths, documents, chunks, sources, years, terms)
        if semantic:
            if semantic["variant"] != lexical["variant"] or semantic["hops"] != lexical["hops"]:
                raise ValueError("Score snapshots use the frozen lexical expansion variant")
            scored = ScoreBatch.from_dict(score["batch"]) if score and score["batch"] is not None else None
            result = fuse(
                baseline, ordered_novel(run), mode="jina", anchor=semantic["anchor"],
                expansion_weight=semantic["expansion_weight"], jina_strength=semantic["jina_strength"],
                score_batch=scored, **common,
            )
            ranks["Tune 冻结 Jina"] = _ranking_record(result, baseline, paths, documents, chunks, sources, years, terms)
        query_rows = []
        for query in run["queries"]:
            term = query["term"]
            new_docs = {path["document_id"] for path in run["new_candidates"]
                        if path["request_id"] == query["request_id"]}
            query_rows.append({
                "term": term["term"], "term_tokens": term["term_tokens"], "hop": query["hop"],
                "df": term["term_document_frequency"], "new_parents": len(new_docs),
                "hits": len(query["hits"]),
            })
        yield {
            "query_id": qid, "question": question, "named_sources": sources, "named_years": sorted(years),
            "expansion": {"question-only": _snapshot(item["question_only"]), "one-hop": _snapshot(one),
                          "two-hop": _snapshot(two), "frozen": _snapshot(run)},
            "frozen_queries": query_rows, "rankings": ranks,
            "jina_pair_counts": {key: score["batch"][key] for key in ("actual_pairs", "historical_hits", "local_hits",
                                                                      "new_pairs", "inference_seconds")
                                 } if score and score.get("batch") else None,
        }


def _safe(text):
    return str(text).replace("|", "\\|").replace("\n", " ")


def _snippet(text, term, window=450):
    literal = text.casefold().find(term.casefold())
    start = max(0, literal - window // 3)
    excerpt = " ".join(text[start:start + window].split())
    return (("…" if start else "") + excerpt + ("…" if start + window < len(text) else ""))


def _percentile(values, p):
    ordered = sorted(values)
    return ordered[max(0, ceil(p * len(ordered)) - 1)]


def _case_examples(records, documents, chunks, archives, frozen):
    by_id = {row["query_id"]: row for row in records}
    saved = {row["query_id"]: row for row in archives}
    output = []
    for qid, method, doc, term, fragment, assessment in REVIEWED:
        if qid not in saved:
            continue
        archive = saved[qid]
        if method == "frozen":
            run = winner_run(archive, frozen["lexical_winner"])
        else:
            run = through_hop(archive["variants"][frozen["seed_variants"]["h2"]], 2)
        matches = [row for row in run["new_candidates"] if row["document_id"] == doc and row["term"] == term]
        if not matches:
            raise ValueError(f"Reviewed expansion path no longer exists for {qid}/{doc}")
        hit = matches[0]
        witness = chunks[hit["chunk_id"]]
        if fragment not in witness:
            raise ValueError(f"Reviewed passage no longer exists for {qid}/{hit['chunk_id']}")
        article = documents[doc]
        positions = []
        for label in ("P5-B", "Tune 冻结词法", "Tune 冻结 Jina"):
            arm = by_id[qid]["rankings"].get(label)
            if arm:
                position = arm["top10"].index(doc) + 1 if doc in arm["top10"] else "未进入"
                positions.append(f"{label} {position}")
        output.extend([
            f"### {qid} · {doc}", "",
            f"> {_safe(archive['question'])}", "",
            f"- {method} 对照的第 {hit['hop']} 跳使用「{_safe(hit['term'])}」，"
            f"路径为 {' → '.join(hit['path'])}。候选来自 {_safe(article.metadata['source'])}，标题《{_safe(article.title)}》。"
            f"候选在三个 Top 10 的位置为 {'；'.join(positions)}。",
            f"- Chunk {hit['chunk_id']} 中的片段为「{_safe(_snippet(witness, fragment))}」。",
            f"- {assessment}", "",
        ])
    return output


def _report(records, frozen, documents, chunks, archives):
    null = len(records)
    rows_ = [
        "# EGE-v0 无答案问题扩展诊断", "",
        f"独立分析 {null} 道无答案题。应用开发 Tune 冻结的配置，候选生成及融合使用原题和原始排名。"
        "这批题没有可用于计数的已知必要来源，因此这里报告候选行为和可核对的题面约束。",
        "",
        "## 扩展规模与第二跳",
        "",
        "| 方案 | 平均扩展请求 | 请求上限实测 | 平均新增父文档 | P95／最大新增 | 平均最终候选 | 实际执行二跳题数 | 二跳新增题数／文档次数 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label in records[0]["expansion"]:
        arm = [row["expansion"][label] for row in records]
        counts = [row["new"] for row in arm]
        rows_.append(
            f"| {label} | {mean(row['queries'] for row in arm):.2f} | {max(row['queries'] for row in arm)} | "
            f"{mean(counts):.2f} | {_percentile(counts, 0.95)}／{max(counts)} | "
            f"{mean(row['parents'] for row in arm):.2f} | {sum(row['hops'] == 2 for row in arm)} | "
            f"{sum(row['new_by_hop']['2'] > 0 for row in arm)}／{sum(row['new_by_hop']['2'] for row in arm)} |"
        )
    two = [row["expansion"]["two-hop"] for row in records]
    rows_.extend([
        "",
        f"两跳对照运行后，{sum(row['hops'] == 2 and row['new_by_hop']['2'] == 0 for row in two)} 道实际发出二跳请求的题没有新增父文档。"
        "这是可复核的无增量请求计数。是否语义无效还需核对返回材料。",
        "",
        "## 排名变化",
        "",
        "| 方法 | 平均保留 P5-B Top 10 | Top 10 集合变化 | Top 10 顺序变化 | 新扩展文档进入 Top 10 次数 | 排名回退到完整 P5-B |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for name in records[0]["rankings"]:
        arm = [row["rankings"][name] for row in records]
        rows_.append(f"| {name} | {mean(row['common_top10'] for row in arm):.2f} | "
                     f"{sum(row['changed_set'] for row in arm)} | {sum(row['changed_order'] for row in arm)} | "
                     f"{sum(len(row['new_top10']) for row in arm)} | {sum(bool(row['fallback_reason']) for row in arm)} |")
    rows_.extend(["", "Anchor-5/7/8 使用同一候选与冻结词法权重。"
                      "冻结权重 0.88 下，扩展第一名只能占用第 10 位，锚定这些原文档不会再改变 Top 10。"])
    scored = [row["jina_pair_counts"] for row in records if row["jina_pair_counts"] is not None]
    if scored:
        rows_.extend(["", f"无答案对照独立提交 {sum(row['new_pairs'] for row in scored):,} 个新 Jina 对，"
                           f"历史精确命中 {sum(row['historical_hits'] for row in scored):,} 对，"
                           f"本轮缓存重用 {sum(row['local_hits'] for row in scored):,} 对。"
                           f"记录的模型推理合计 {sum(row['inference_seconds'] for row in scored):.2f} 秒。"])
        if not frozen.get("jina_winner"):
            rows_.append("开发集未冻结 Jina 融合，因此这些分数只保留作成本记录。")
    else:
        rows_.extend(["", "本次无答案诊断使用已归档的词法结果。"])
    source_rows = [row for row in records if row["named_sources"]]
    year_rows = [row for row in records if row["named_years"]]
    rows_.extend([
        "", "## 媒体、年份与题面术语", "",
        f"以可重复的媒体别名表识别显式媒体名称，覆盖 {len(source_rows)} 道。"
        f"以独立的 19xx／20xx 标记识别年份，覆盖 {len(year_rows)} 道。"
        "媒体检查比对父文档来源元数据，年份检查比对保留 Chunk 中的字面词。"
        "匹配年份无法确认发布时间或事实时间。未匹配年份也可能由分块造成。",
        "",
        "| 方法 | 带媒体名题中 Top 10 至少一篇媒体命中 | 带年份题中 Top 10 至少一块含题面年份 | 新入选中来源不匹配次数／媒体题新入选次数 | 新入选中题面内容词重合 ≤1 次数／总新入选次数 |",
        "|---|---:|---:|---:|---:|",
    ])
    for name in records[0]["rankings"]:
        source_added = [hit for row in source_rows for hit in row["rankings"][name]["new_top10"]]
        added = [hit for row in records for hit in row["rankings"][name]["new_top10"]]
        rows_.append(
            f"| {name} | {sum(row['rankings'][name]['source_named'] > 0 for row in source_rows)} | "
            f"{sum(row['rankings'][name]['year_present'] > 0 for row in year_rows)} | "
            f"{sum(hit['source_match'] is False for hit in source_added)}／{len(source_added)} | "
            f"{sum(len(hit['question_content_overlap']) <= 1 for hit in added)}／{len(added)} |"
        )
    used_terms = [term for row in records for term in row["frozen_queries"]]
    freq = Counter(tuple(item["term_tokens"]) for item in used_terms)
    term_added = Counter()
    term_low_overlap = Counter()
    term_source_miss = Counter()
    for row in records:
        for hit in row["rankings"]["Tune 冻结词法"]["new_top10"]:
            key = tuple(hit["term_tokens"])
            term_added[key] += 1
            term_low_overlap[key] += len(hit["question_content_overlap"]) <= 1
            term_source_miss[key] += hit["source_match"] is False
    term_labels = {tuple(item["term_tokens"]): (item["term"], item["df"]) for item in used_terms}
    rows_.extend([
        "", "下表是无答案队列中高频触发的词及其新增入选，供人工检查重复线索带来的主题漂移。"
        "df 是整库父文档频次。触发多或来源不匹配的词仍可能提供跨来源上下文，无法据此判定整条关系。",
        "", "| 扩展词 | df | 在此队列使用次数 | Tune 冻结词法引入前十次数 | 其中内容词重合 ≤1 | 其中题面媒体不符 |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for key, count in sorted(freq.items(), key=lambda row: (-row[1], row[0]))[:12]:
        term, df = term_labels[key]
        rows_.append(f"| {_safe(term)} | {df} | {count} | {term_added[key]} | "
                     f"{term_low_overlap[key]} | {term_source_miss[key]} |")
    rows_.extend(["", "## 逐题核对样本", ""])
    cases = _case_examples(records, documents, chunks, archives, frozen)
    rows_.extend(cases or ["没有符合预先核对条件的扩展样本。", ""])
    rows_.extend([
        "个案只判断题面与展示的候选，不衡量整套语料的支持率。"
        "无答案队列缺少候选级事实或关系标注，无法估计语义错误扩展率或拒答效果。",
        "",
        "## 复现",
        "",
        "保存冻结配置及无答案阶段输出后，可从正式项目根目录复现。",
        "",
        "~~~powershell",
        r"& .\.venv\Scripts\python.exe -B -m analysis.ege_v0.no_answer",
        "~~~",
        "",
        "逐题输出记录了三个 Anchor 对照、已冻结结果、必要的扩展路径与测量。",
        "",
    ])
    return "\n".join(rows_)


def main():
    frozen = read(OUT / "experiment_config.json")
    _, _, null = cohorts()
    source = DATA / "null_expansion.jsonl"
    archives = list(rows(source))
    if len(archives) != len(null) or {row["query_id"] for row in archives} != null:
        raise ValueError("Expected all and only 301 frozen null query IDs")
    score_path = DATA / "null_jina.jsonl"
    score_rows = {row["query_id"]: row for row in rows(score_path)} if score_path.is_file() else {}
    if (score_rows and set(score_rows) != null) or (frozen.get("jina_winner") and set(score_rows) != null):
        raise ValueError("Expected frozen null score snapshots for the full cohort")
    documents, original_chunks = corpus()
    chunks = {chunk.chunk_id: chunk.text for blocks in original_chunks.values() for chunk in blocks}
    result = list(_rows(frozen, archives, score_rows, documents, chunks))
    write_rows(DATA / "null_diagnostics.jsonl", result)
    report = OUT / "no_answer_expansion_analysis.md"
    partial = report.with_suffix(".md.tmp")
    partial.write_text(_report(result, frozen, documents, chunks, archives), encoding="utf-8", newline="\n")
    partial.replace(report)
    print(f"Wrote null expansion diagnostics for {len(result)} questions")


if __name__ == "__main__":
    main()

