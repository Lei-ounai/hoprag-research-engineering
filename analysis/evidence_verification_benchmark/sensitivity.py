"""Replay each Round 9 threshold over unchanged scored, label-free observations."""

from collections import Counter
import csv
from dataclasses import dataclass, replace
from hashlib import file_digest
import json
from pathlib import Path
from time import perf_counter

from analysis.egr_v0.selector import GateConfig, choose, prepare


OUT = Path(__file__).resolve().parent
EGR = OUT.parent / "egr_v0"
ELIGIBLE = {"eligible", "replaced"}
BUDGET_REASON = "replacement_budget_exhausted"
GRIDS = {
    "quality": (0.00, 0.25, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.88, 0.90, 0.92, 0.95, 0.98, 1.00),
    "novelty": (0.10, 0.20, 0.35, 0.50, 0.65, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00),
    "margin": (0.00, 0.01, 0.025, 0.05, 0.10, 0.20, 0.35, 0.50, 0.55, 0.60,
               0.65, 0.70, 0.75, 0.90, 1.00),
}
CONFIG_FIELD = {"quality": "min_quality", "novelty": "novelty_overlap", "margin": "min_margin"}


@dataclass(frozen=True)
class Point:
    context: str
    stage: str
    threshold: float
    config: GateConfig
    reference: GateConfig

    @property
    def stage_direction(self):
        current = getattr(self.reference, CONFIG_FIELD[self.stage])
        if self.threshold == current:
            return "current"
        stricter = self.threshold < current if self.stage == "novelty" else self.threshold > current
        return "stricter" if stricter else "looser"


def points(frozen):
    for stage, values in GRIDS.items():
        key = CONFIG_FIELD[stage]
        for threshold in sorted({*values, getattr(frozen, key)}):
            yield Point("frozen_one_factor", stage, threshold,
                        replace(frozen, **{key: threshold}), frozen)
    # Open only the numeric quality cutoff. Supported status/excerpt/key,
    # source/time verification, lexical eligibility and margin still apply.
    relaxed = replace(frozen, min_quality=0.0)
    for threshold in sorted({*GRIDS["novelty"], relaxed.novelty_overlap}):
        yield Point("novelty_quality_zero_diagnostic", "novelty", threshold,
                    replace(relaxed, novelty_overlap=threshold), relaxed)


def rows(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def inputs(egr, cohort):
    run = egr / "run"
    scored = rows(run / f"{cohort}_scores.jsonl")
    verified = rows(run / f"{cohort}_witnesses.jsonl")
    seen = set()
    for record, witness in zip(scored, verified, strict=True):
        qid = record["query_id"]
        if qid != witness["query_id"] or qid in seen:
            raise ValueError(f"Scoring/witness join misaligned: {qid}")
        seen.add(qid)
        yield qid, competition(record, witness["witnesses"])


def competition(record, witnesses):
    detail = record["scoring"]
    by_parent = {parent: max(chunk["jina_score"] for chunk in blocks)
                 for parent, blocks in detail["documents"].items() if blocks}
    return prepare(record["baseline"], by_parent, witnesses, challengers=record["fixed"],
                   failure_reason=detail["fallback_reason"])


def sets(selection):
    result = {key: {d.document_id for d in selection.decisions if getattr(d, f"{key}_pass") is True}
              for key in CONFIG_FIELD}
    result["all_gate"] = {d.document_id for d in selection.decisions
                          if d.status in ELIGIBLE or d.reason == BUDGET_REASON}
    result["replaced"] = {r.candidate_document_id for r in selection.replacements}
    return result


def observe(selection, reference, stage, population, supported, conflict):
    outcome = sets(selection)
    field = f"{stage}_pass"
    values = {
        "challenger_pairs": len(population),
        "stage_pass": len(outcome[stage]),
        "stage_reject": sum(getattr(row, field) is False for row in selection.decisions),
        "stage_not_reached": sum(getattr(row, field) is None for row in selection.decisions),
        "stage_new_pass_vs_reference": len(outcome[stage] - reference[stage]),
        "stage_lost_pass_vs_reference": len(reference[stage] - outcome[stage]),
        "stage_still_not_passed_from_reference": len(population - (outcome[stage] | reference[stage])),
        "quality_pass": len(outcome["quality"]),
        "novelty_pass": len(outcome["novelty"]),
        "margin_pass": len(outcome["margin"]),
        "all_gate_eligible": len(outcome["all_gate"]),
        "new_eligible_vs_reference": len(outcome["all_gate"] - reference["all_gate"]),
        "lost_eligible_vs_reference": len(reference["all_gate"] - outcome["all_gate"]),
        "originally_ineligible_and_still_ineligible": len(population - (outcome["all_gate"] | reference["all_gate"])),
        "replacement_pairs": len(outcome["replaced"]),
        "replacement_questions": int(bool(outcome["replaced"])),
        "new_replacement_pairs_vs_reference": len(outcome["replaced"] - reference["replaced"]),
        "lost_replacement_pairs_vs_reference": len(reference["replaced"] - outcome["replaced"]),
        "eligible_with_conflicting_block_proxy": len(outcome["all_gate"] & conflict),
        "replacement_with_conflicting_block_proxy": len(outcome["replaced"] & conflict),
        "not_quality_passed_with_rule_support_proxy": len(supported - outcome["quality"]),
        "not_eligible_with_rule_support_proxy": len(supported - outcome["all_gate"]),
        "fallback_queries": int(bool(selection.fallback_reason)),
    }
    if len(population) != sum(values[key] for key in ("stage_pass", "stage_reject", "stage_not_reached")):
        raise ValueError("Threshold replay did not account for every unique challenger")
    return values


def history(egr, arm):
    observed = {}
    splits = {}
    with (egr / "gate_decisions.csv").open(encoding="utf-8", newline="") as stream:
        for old in csv.DictReader(stream):
            if old["method"] != arm:
                continue
            key = old["query_id"], old["document_id"]
            if key in observed:
                raise ValueError(f"Duplicate historical challenger: {key}")
            observed[key] = tuple(old[name] for name in
                                  ("status", "reason", "quality_pass", "novelty_pass", "margin_pass", "champion_chunk_id"))
            previous = splits.setdefault(old["query_id"], old["split"])
            if previous != old["split"]:
                raise ValueError(f"Conflicting historical split: {key[0]}")
    return observed, splits


def compare_history(qid, selection, observed):
    for d in selection.decisions:
        key = qid, d.document_id
        target = tuple("" if field is None else str(field) for field in
                       (d.status, d.reason, d.quality_pass, d.novelty_pass, d.margin_pass, d.champion_chunk_id))
        if observed.pop(key, None) != target:
            raise ValueError(f"Frozen replay differs from historical decision: {key}")


def fingerprint(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def sweep(egr=EGR):
    started = perf_counter()
    before_paths = ("frozen_config.json", "gate_decisions.csv", "run/development_scores.jsonl",
                    "run/development_witnesses.jsonl", "run/null_scores.jsonl", "run/null_witnesses.jsonl")
    hashes = {p: fingerprint(egr / p) for p in before_paths}
    with (egr / "frozen_config.json").open(encoding="utf-8") as stream:
        frozen_record = json.load(stream)
    row = frozen_record["thresholds"]["E4"]
    frozen = GateConfig(max_replacements=row["cap"], min_quality=row["quality"], min_margin=row["margin"],
                        novelty_overlap=row["novelty"])
    if frozen.max_replacements != 1:
        raise ValueError("Sensitivity currently requires the archived single-replacement configuration")
    arm = frozen_record["selected_arm"]
    observed, splits = history(egr, arm)
    near_grid = tuple(points(frozen))
    sorted_configs = tuple(dict.fromkeys(
        config for point in near_grid for config in (point.config, point.reference)
    ))
    totals = {(p, split): Counter() for p in near_grid for split in ("development", "tune", "validation", "null")}
    reasons = {(p, split): Counter() for p, split in totals}
    swaps = {(p, split): [] for p, split in totals}
    eligibility_changes = {}
    seen_splits = Counter()
    for cohort in ("development", "null"):
        for qid, candidate in inputs(egr, cohort):
            split = splits[qid]
            expected_splits = {"tune", "validation"} if cohort == "development" else {"null"}
            if split not in expected_splits:
                raise ValueError(f"Unexpected archived split: {qid}/{split}")
            seen_splits[split] += 1
            out = {config: choose(candidate, config) for config in sorted_configs}
            compare_history(qid, out[frozen], observed)
            set_cache = {config: sets(selection) for config, selection in out.items()}
            population = set(candidate.candidates)
            supported = {d for d in population if any(
                w.status == "supported" and w.keys and w.text for w in candidate.witnesses.get(d, ()))}
            conflict = {d for d in population if any(w.status == "conflict" for w in candidate.witnesses.get(d, ()))}
            for point in near_grid:
                selected = out[point.config]
                reference = set_cache[point.reference]
                values = observe(selected, reference, point.stage, population, supported, conflict)
                eligible = set_cache[point.config]["all_gate"]
                if eligible != reference["all_gate"]:
                    name = f"{point.context}:{cohort}:{point.stage}:{point.threshold}"
                    eligibility_changes.setdefault(name, []).append({
                        "query_id": qid,
                        "newly_eligible": sorted(eligible - reference["all_gate"]),
                        "newly_ineligible": sorted(reference["all_gate"] - eligible),
                    })
                for label in (split, "development") if cohort == "development" else (split,):
                    key = point, label
                    totals[key].update(values)
                    reasons[key].update(d.reason for d in selected.decisions)
                    for r in selected.replacements:
                        swaps[key].append({"query_id": qid, "candidate_document_id": r.candidate_document_id,
                                           "incumbent_document_id": r.incumbent_document_id, "visible_chunk_id": r.champion_chunk_id,
                                           "positive_margin": r.score_margin})
    if observed or seen_splits != {"tune": 271, "validation": 180, "null": 301}:
        raise ValueError("Replay did not exactly cover the frozen 451 development and 301 null questions")
    if any(fingerprint(egr / name) != expected for name, expected in hashes.items()):
        raise ValueError("Historical sensitivity input changed during replay")
    all_rows = []
    replacement_details = {}
    for (point, split), values in totals.items():
        queries = seen_splits.total() - seen_splits["null"] if split == "development" else seen_splits[split]
        detail = {
            "context": point.context, "split": split, "stage": point.stage, "threshold": point.threshold,
            "relative_strictness": point.stage_direction, "quality_floor": point.config.min_quality,
            "novelty_overlap": point.config.novelty_overlap, "margin_floor": point.config.min_margin,
            "replacement_cap": point.config.max_replacements, "queries": queries, **values,
            "potential_false_acceptance_count": None, "potential_false_rejection_count": None,
            "independent_labels_status": "pending_human_annotation",
            "decision_reason_counts": dict(sorted(reasons[point, split].items())),
        }
        all_rows.append(detail)
        if swaps[point, split] and split in ("development", "null"):
            replacement_details[f"{point.context}:{split}:{point.stage}:{point.threshold}"] = swaps[point, split]
    summary = {
        "source_arm": arm,
        "areas": {"quality": "Numeric cutoff varies. The semantic supported status, visible excerpt and coverage key remain required.",
                  "novelty": "Actual novelty additionally requires a new coverage key. Tuning the trigram overlap cutoff cannot admit already-covered keys.",
                  "margin": "Candidate's supporting Chunk raw Jina minus the actual incumbent's best fair-budget Jina; a positive difference remains required at zero."},
        "proxy_definitions": {
            "eligible_with_conflicting_block_proxy": "All-gate eligible challenger query-parent pair with any other or same scored backup block carrying rule conflict. No claim that conflict opposes the admitted proposition.",
            "replacement_with_conflicting_block_proxy": "Actual replacement query-parent pair with at least one stored rule-conflict block. A different proposition can explain the conflict.",
            "not_quality_passed_with_rule_support_proxy": "Challenger has an archived supported block with a nonempty excerpt and key, but does not pass current numerical quality.",
            "not_eligible_with_rule_support_proxy": "Challenger has such rule support and fails final pre-budget eligibility. Can be the intentional result of already-covered evidence, protection or margin.",
        },
        "unavailable": "The false-acceptance and false-rejection count fields are empty pending independent human labels. The four proxies are transparent triage signals without verified error status.",
        "cross_checks": {"compared_frozen_decision_signature": "status, first reason, quality/novelty/margin flags, champion chunk",
                         "frozen_challenger_pairs": next(r["challenger_pairs"] for r in all_rows if r["split"] == "development"),
                         "null_challenger_pairs": next(r["challenger_pairs"] for r in all_rows if r["split"] == "null"),
                         "historical_input_sha256_unchanged": hashes},
        "questions": dict(seen_splits),
        "new_jina_pairs": 0, "new_embeddings": 0, "wall_seconds": perf_counter() - started,
        "eligibility_changes_vs_reference": eligibility_changes,
        "replacement_details": replacement_details,
    }
    return all_rows, summary


def main():
    records, summary = sweep()
    csv_path, json_path = (OUT / "threshold_sensitivity.csv", OUT / "threshold_sensitivity_summary.json")
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        for row in records:
            writer.writerow({**row, "decision_reason_counts": json.dumps(row["decision_reason_counts"],
                                                                         ensure_ascii=False, separators=(",", ":"))})
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Verified {summary['cross_checks']['frozen_challenger_pairs']:,} development and "
          f"{summary['cross_checks']['null_challenger_pairs']:,} independent null pairs. "
          f"Saved {len(records)} threshold/split records in {summary['wall_seconds']:.2f}s.")


if __name__ == "__main__":
    main()

