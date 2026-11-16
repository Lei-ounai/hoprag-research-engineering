"""Draw a stratified, pair-level offline audit sample from the frozen E4-B1 decision ledger."""

from collections import Counter, defaultdict
import csv
from hashlib import sha256
import json
from math import ceil, isfinite
from pathlib import Path

from hoprag.dataset import load_dataset


SEED = 20261115
PRIMARY = "E4-B1"
BRANCHES = ("E3-B1", PRIMARY, "E5-B1")
EXPECTED_COUNTS = {"development": 15821, "null": 12316}
TARGETS = {"development": 260, "null": 90}

# Each pair belongs to the first applicable stratum. Tags from all other applicable
# strata are kept for diagnostics but excluded from the weights.
STRATA = (
    ("final_gate_accepted", None, None),
    ("e3_control_accepted", None, None),
    ("margin_rejected", None, None),
    ("novelty_rejected", None, None),
    ("date_stage_conflict", None, None),
    ("novelty_shadow_review", 8, 3),
    ("null_e2_promoted", 0, 31),
    ("date_witness_conflict", 9, 6),
    ("role_relation_conflict", 24, 10),
    ("date_scope_review", 27, 9),
    ("gold_required_quality_reject", 64, 0),
    ("role_relation_review", 31, 8),
    ("high_jina_quality_reject", 53, 12),
    ("ordinary", 0, 0),
)
_HARD_RELATIONS = frozenset(("entity_role_conflict", "relation_mismatch"))
_UNRESOLVED_RELATIONS = frozenset(("entity_role_unverified", "relation_unverified"))
_REASON_NOVELTY = frozenset(("already_covered", "near_duplicate_witness"))
_BOOL_FIELDS = ("quality_pass", "novelty_pass", "margin_pass")
_NUMBER_FIELDS = ("candidate_parent_score", "candidate_witness_score", "incumbent_score", "score_margin")
_LIST_FIELDS = ("new_coverage_keys", "protected_coverage_keys")


def _ledger_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        yield from csv.DictReader(stream)


def _json_rows(path):
    with path.open(encoding="utf-8-sig") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def _parse_bool(value):
    if value in ("True", "False", ""):
        return {"True": True, "False": False, "": None}[value]
    raise ValueError(f"Unexpected gate boolean {value!r}")


def _parse_gate(row):
    out = {field: row[field] or None for field in ("status", "reason", "champion_chunk_id", "incumbent_document_id")}
    out.update({field: _parse_bool(row[field]) for field in _BOOL_FIELDS})
    out.update({field: float(row[field]) if row[field] else None for field in _NUMBER_FIELDS})
    if any(value is not None and not isfinite(value) for value in (out[name] for name in _NUMBER_FIELDS)):
        raise ValueError("Non-finite archived candidate score")
    out["incumbent_rank"] = int(row["incumbent_rank"]) if row["incumbent_rank"] else None
    out.update({field: json.loads(row[field]) if row[field] else [] for field in _LIST_FIELDS})
    return out


def read_gate_frames(path):
    frames = {branch: {} for branch in BRANCHES}
    row_counts = Counter()
    seen_by_method = defaultdict(set)
    for row in _ledger_rows(path):
        branch = row["method"]
        key = (row["query_id"], row["document_id"])
        if key in seen_by_method[branch]:
            raise ValueError(f"Duplicate Query–Document pair inside archived {branch}: {key}")
        seen_by_method[branch].add(key)
        row_counts[branch, row["split"]] += 1
        if branch not in frames:
            continue
        split = row["split"]
        if split not in ("tune", "validation", "null"):
            raise ValueError(f"Unexpected archived development split: {split}")
        frames[branch][key] = {"split": split, **_parse_gate(row)}
    if set(frames[BRANCHES[0]]) != set(frames[PRIMARY]) or not frames[PRIMARY].keys() <= frames[BRANCHES[2]].keys():
        raise ValueError("Frozen E4 frame differs from the expected fixed and augmented frames")
    distinct_pairs = len(set().union(*seen_by_method.values()))
    return frames, row_counts, distinct_pairs


def read_activity(path, query_ids):
    baseline, e2 = {}, {}
    contender_positions = defaultdict(dict)
    for row in _json_rows(path):
        query_id = row["query_id"]
        if query_id not in query_ids:
            continue
        branch = row["method"]
        if branch == "E0":
            if query_id in baseline or row["top10"] != row["baseline_top10"] or len(row["top10"]) != 10:
                raise ValueError(f"Invalid E0 original Top 10: {query_id}")
            baseline[query_id] = tuple(row["top10"])
        if branch not in ("E2", *BRANCHES):
            continue
        for swap in row["replacements"]:
            key = (query_id, swap["candidate_document_id"])
            item = {field: swap[field] for field in ("position", "incumbent_document_id", "champion_chunk_id")}
            if branch == "E2":
                if key in e2:
                    raise ValueError(f"Duplicate ungated control replacement: {key}")
                e2[key] = item
            else:
                if branch in contender_positions[key]:
                    raise ValueError(f"Duplicate actual replacement: {branch} / {key}")
                contender_positions[key][branch] = item
    if baseline.keys() != query_ids:
        raise ValueError("E0 baseline log does not cover the E4 frame")
    return baseline, e2, contender_positions


def read_witness_diagnostics(path, frame_keys, baseline):
    """Extract only rule diagnostics used for sample coverage, never evidence labels."""
    flags = defaultdict(set)
    supported = defaultdict(set)
    for row in _json_rows(path):
        query, document = row["query_id"], row["document_id"]
        key = (query, document)
        if key not in frame_keys and document not in baseline.get(query, ()):
            continue
        if row["status"] == "supported":
            supported[key].update(row["coverage_keys"])
        if key not in frame_keys:
            continue
        reasons, features = set(row["reasons"]), row["features"]
        # A source mismatch or absent question linkage does not itself create
        # a temporal/relation review case: prioritize recognizably aligned spans.
        aligned = (features.get("source_status") != "mismatch" and
                   "no_question_link" not in reasons and
                   (features.get("anchor_overlap", 0) > 0 or
                    len(features.get("question_keyword_hits", ())) >= 3))
        if not aligned:
            continue
        if "date_conflict" in reasons:
            flags[key].add("date_witness_conflict")
        if "date_unverified" in reasons and features.get("date_kinds"):
            flags[key].add("date_scope_review")
        if reasons & _HARD_RELATIONS:
            flags[key].add("role_relation_conflict")
        if reasons & _UNRESOLVED_RELATIONS:
            flags[key].add("role_relation_review")
    for (query, document), keys in supported.items():
        key = (query, document)
        if key not in frame_keys or not keys:
            continue
        reference_keys = {value for parent in baseline[query] for value in supported.get((query, parent), ())}
        if reference_keys & keys:
            flags[key].add("novelty_shadow_review")
    return flags


def _gold_sources(raw_dir, develop_ids):
    """Read only the necessary-source index for enrichment, behind the offline draw."""
    dataset = load_dataset(raw_dir)
    refs = {q.query_id: frozenset(e.source_document_id for e in q.evidence_list if e.source_document_id)
            for q in dataset.queries if q.query_id in develop_ids}
    if refs.keys() != develop_ids:
        raise ValueError("Development candidate frame lacks raw annotation alignment")
    return refs


def _top_quintile(frame):
    by_query = defaultdict(list)
    for (query, document), row in frame.items():
        score = row["candidate_parent_score"]
        if score is not None:
            by_query[query].append((score, document))
    output = set()
    for query, entries in by_query.items():
        ordered = sorted(entries, key=lambda value: (-value[0], value[1]))
        output.update((query, doc) for _, doc in ordered[:ceil(len(ordered) / 5)])
    return output


def assign_tags(row, *, control, witness_flags, gold_required, high_score, e2_promoted):
    tags = set(witness_flags)
    if row["status"] == "replaced":
        tags.add("final_gate_accepted")
    if control["status"] == "replaced":
        tags.add("e3_control_accepted")
    if row["reason"] == "insufficient_replacement_margin":
        tags.add("margin_rejected")
    if row["reason"] in _REASON_NOVELTY or (row["quality_pass"] and row["novelty_pass"] is False):
        tags.add("novelty_rejected")
    if row["reason"] == "date_conflict":
        tags.add("date_stage_conflict")
    if row["reason"] in _HARD_RELATIONS:
        tags.add("role_relation_conflict")
    if row["reason"] in _UNRESOLVED_RELATIONS:
        tags.add("role_relation_review")
    if row["split"] == "null" and e2_promoted:
        tags.add("null_e2_promoted")
    if gold_required and row["status"] == "rejected" and row["quality_pass"] is False:
        tags.add("gold_required_quality_reject")
    if high_score and row["status"] == "rejected" and row["quality_pass"] is False:
        tags.add("high_jina_quality_reject")
    return tags or {"ordinary"}


def _priority(seed, namespace, key):
    return sha256(f"{seed}\0{namespace}\0{key[0]}\0{key[1]}".encode("utf-8")).digest()


def choose_stratified(units, *, seed=SEED, targets=None):
    """Assign one stratum per pair, then draw uniformly in each with known pi."""
    targets = TARGETS if targets is None else targets
    grouped = defaultdict(list)
    tagged_counts = Counter()
    names = [row[0] for row in STRATA]
    all_names = frozenset(names)
    for key, row in units.items():
        cohort = row["cohort"]
        tags = set(row["overlapping_tags"])
        unknown = tags - all_names
        if unknown:
            raise ValueError(f"Undeclared sampling stratum: {unknown}")
        tagged_counts.update((cohort, tag) for tag in tags)
        stratum = next((name for name in names if name in tags), "ordinary")
        grouped[cohort, stratum].append(key)

    selected, stats = {}, []
    for cohort, target in targets.items():
        index = 1 if cohort == "development" else 2
        population = sum(len(grouped[cohort, name]) for name in names)
        target = min(target, population)
        quotas = {}
        # Nonempty random strata always have positive inclusion probability.
        for stratum_def in STRATA:
            name = stratum_def[0]
            size = len(grouped[cohort, name])
            requested = stratum_def[index]
            quotas[name] = size if requested is None else min(size, max(1, requested))
        while sum(quotas.values()) > target:
            adjustable = (row[0] for row in reversed(STRATA)
                          if row[index] is not None and quotas[row[0]] > 1)
            name = next(adjustable, None)
            if name is None:
                raise ValueError("Sample target cannot cover the certainty cases and all strata")
            quotas[name] -= 1
        leftover = target - sum(quotas.values())
        for name in ("ordinary", *(n for n in reversed(names) if n != "ordinary")):
            additional = min(leftover, len(grouped[cohort, name]) - quotas[name])
            quotas[name] += additional
            leftover -= additional
        if leftover:
            raise ValueError("Sample quotas could not reach planned cohort scope")
        for name in names:
            keys = grouped[cohort, name]
            total, picked = len(keys), quotas[name]
            prob = picked / total if total else None
            weight = total / picked if picked else None
            if keys and not picked:
                raise ValueError("Nonempty stratum has no probability of annotation")
            sampled = sorted(keys, key=lambda key: (_priority(seed, f"{cohort}/{name}", key), key))[:picked]
            for key in sampled:
                if key in selected:
                    raise ValueError("Sample unit assigned to more than one stratum")
                selected[key] = {"stratum": name, "inclusion_probability": prob, "sample_weight": weight}
            stats.append({
                "cohort": cohort, "stratum": name, "population": total, "selected": picked,
                "tagged_population_including_overlap": tagged_counts[cohort, name],
                "selection_method": ("empty" if not keys else "census" if picked == total
                                     else "seeded SHA-256 priority sample without replacement"),
                "seed": seed, "inclusion_probability": prob, "sample_weight": weight,
            })
    ordered = sorted(selected, key=lambda key: (_priority(seed, "blind-case-order", key), key))
    output = [{"case_id": f"evb-{i:04d}", "query_id": key[0], "document_id": key[1],
               **units[key], **selected[key]} for i, key in enumerate(ordered, 1)]
    selected_tags = Counter((row["cohort"], tag) for row in output for tag in row["overlapping_tags"])
    for row in stats:
        result = selected_tags[row["cohort"], row["stratum"]]
        row["tagged_selected_including_overlap"] = result
    return output, stats


def draw(project_root, *, seed=SEED):
    """Return restricted sample records, design statistics, and verified counts."""
    root = Path(project_root)
    history = root / "analysis" / "egr_v0"
    frames, row_counts, distinct_pairs = read_gate_frames(history / "gate_decisions.csv")
    frame = frames[PRIMARY]
    cohort_counts = Counter("null" if row["split"] == "null" else "development" for row in frame.values())
    if dict(cohort_counts) != EXPECTED_COUNTS:
        raise ValueError(f"Unexpected frozen E4 population: {dict(cohort_counts)}")
    queries_by_split = defaultdict(set)
    for (query, _), row in frame.items():
        queries_by_split[row["split"]].add(query)
    all_ids = set().union(*queries_by_split.values())
    development_ids = queries_by_split["tune"] | queries_by_split["validation"]
    null_ids = queries_by_split["null"]
    split_sizes = {split: len(ids) for split, ids in queries_by_split.items()}
    if split_sizes != {"tune": 271, "validation": 180, "null": 301} or len(all_ids) != 752:
        raise ValueError("Unexpected or overlapping query cohorts")
    baseline, e2, actual = read_activity(history / "replacement_log.jsonl", all_ids)
    gold = _gold_sources(root / "data" / "raw" / "multihop_rag", development_ids)
    flags = read_witness_diagnostics(history / "evidence_witnesses.jsonl", frame.keys(), baseline)
    high = _top_quintile(frame)

    units = {}
    for key, row in frame.items():
        query, document = key
        if document in baseline[query]:
            raise ValueError(f"Incumbent must not enter the challenger frame: {key}")
        gold_required = document in gold.get(query, ())
        tags = assign_tags(row, control=frames[BRANCHES[0]][key],
                           witness_flags=flags.get(key, ()), gold_required=gold_required,
                           high_score=key in high, e2_promoted=key in e2)
        units[key] = {
            "cohort": "null" if row["split"] == "null" else "development",
            "split": row["split"], "overlapping_tags": sorted(tags), "fair_jina_score": row["candidate_parent_score"],
            "gold_source_for_diagnostics": gold_required if query in development_ids else None,
            "gate_decisions": {branch: frames[branch][key] for branch in BRANCHES},
            "e2_replacement": e2.get(key), "actual_replacements": actual.get(key, {}),
        }
    selected, strata = choose_stratified(units, seed=seed)
    design = {
        "unit": "unique (query_id, document_id) challenger for E4-B1",
        "scope": PRIMARY, "seed": seed, "targets": TARGETS, "population": dict(cohort_counts),
        "query_counts": {"development": len(development_ids), "null": len(null_ids)},
        "gate_log_rows": sum(row_counts.values()),
        "distinct_query_document_pairs_across_gate_log": distinct_pairs,
        "branch_and_budget_repeat_appearances": sum(row_counts.values()) - distinct_pairs,
        "duplicates_within_single_branch_and_budget": 0,
        "e4_rows_by_split": {split: row_counts[PRIMARY, split] for split in ("tune", "validation", "null")},
        "duplicates_within_primary_frame": 0, "development_validation_and_tune_report_jointly": True,
        "e5_only_excluded": len(frames[BRANCHES[2]].keys() - frame.keys()),
        "e5_only_excluded_by_cohort": dict(Counter(
            "null" if frames[BRANCHES[2]][key]["split"] == "null" else "development"
            for key in frames[BRANCHES[2]].keys() - frame.keys()
        )),
        "actual_accepted_e4_count": sum(row["status"] == "replaced" for row in frame.values()),
        "e3_control_accepted_count_in_frame": sum(row["status"] == "replaced" for row in frames[BRANCHES[0]].values()),
        "e4_margin_rejected_in_frame": sum("margin_rejected" in row["overlapping_tags"] for row in units.values()),
        "e4_novelty_rejected_in_frame": sum("novelty_rejected" in row["overlapping_tags"] for row in units.values()),
        "high_score_rule": "Top ceil(20%) of E4 challengers by fair maxima within each query; stable document-ID ties",
        "shadow_novelty_rule": "Observed supported clause-key intersects a baseline supported key; not an actual gate decision",
        "draw_rule": "First stratum by precedence; seeded SHA-256 order; no replacement; development and null separate",
        "primary_stratum_precedence": [stratum[0] for stratum in STRATA],
        "gold_used_only_in_forensic_sampling_tags": True,
    }
    return selected, strata, design

