"""Score completed human sheets separately from archived rule predictions."""

from argparse import ArgumentParser
from collections import Counter, defaultdict
from math import sqrt
from pathlib import Path
from statistics import variance
import csv

from analysis.besa_v0.experiment import rows, write

from .build import FREE_FIELDS, LABELS, OUT, PUBLIC_KEYS


DIMENSIONS = ("entity_role", "relation_support", "temporal_scope", "evidence_novelty")
DETERMINATE = {
    "entity_role": {"MATCH", "MISMATCH"},
    "relation_support": {"SUPPORTED", "NEGATED", "HEDGED", "UNRELATED"},
    "temporal_scope": {"CONSISTENT", "CONFLICT"},
    "evidence_novelty": {"NOVEL", "REDUNDANT", "PARTIAL_NOVELTY"},
}


def normalize_quote(text):
    return " ".join(text.split())


def quoted_passages(case):
    if set(case) != PUBLIC_KEYS:
        raise ValueError("Expected the public, blinded case schema")
    out = {part["ref"]: part["jina_visible_text"]
           for side in ("challenger", "comparison_target") for part in case[side]["scored_chunks"]}
    out.update({part["ref"]: part["jina_visible_excerpt"] for part in case["baseline_top10"]})
    return out


def read_sheet(path, cases):
    fields = ["case_id", *LABELS, *FREE_FIELDS]
    completed, seen = {}, set()
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != fields:
            raise ValueError("Annotation file does not match the expected human-only schema")
        for row in reader:
            cid = row["case_id"]
            if cid not in cases or cid in seen:
                raise ValueError(f"Unknown or duplicate human case ID: {cid}")
            seen.add(cid)
            values = {key: (row[key] or "").strip() for key in fields[1:]}
            if not any(values.values()):
                continue
            for label, options in LABELS.items():
                if values[label] not in options:
                    raise ValueError(f"Missing or invalid label {cid}/{label}")
            if not values["reason"] or not values["witness_ref"]:
                raise ValueError(f"Case {cid} requires a reason and a witness reference")
            ref = values["witness_ref"]
            quote = normalize_quote(values["witness_quote"])
            candidate_positive = values["evidence_support"] in {"DIRECT_SUPPORT", "PARTIAL_SUPPORT", "CONTRADICTION"}
            if ref == "NONE":
                if quote or candidate_positive:
                    raise ValueError(f"Case {cid} requires a quote from the actual model-facing challenger window")
            else:
                view = quoted_passages(cases[cid])
                if ref not in view or not quote or quote not in normalize_quote(view[ref]):
                    raise ValueError(f"Case {cid} quote must appear in the displayed passage {ref}")
                if candidate_positive and not ref.startswith("C"):
                    raise ValueError(f"Case {cid} requires a challenger quote for its evidence label")
            completed[cid] = values
    return completed


def ratio(data, membership, *, weighted=False, frame_complete=True):
    """Ratio with design-based linearized SE, or Wilson interval over labelled units."""
    estimates = [(*membership(row), row) for row in data]
    numerator = sum(y * (row["sample_weight"] if weighted else 1) for y, _, row in estimates)
    denominator = sum(x * (row["sample_weight"] if weighted else 1) for _, x, row in estimates)
    if weighted and not frame_complete:
        return {"value": None, "status": "pending_full_human_cohort", "numerator": None, "denominator": None}
    if not denominator:
        return {"value": None, "status": "undefined_zero_denominator", "numerator": 0, "denominator": 0}
    result = numerator / denominator
    if weighted:
        groups = defaultdict(list)
        for y, x, row in estimates:
            groups[row["stratum"]].append((y, x, row))
        variance_total = 0.0
        ci_withheld = None
        for group in groups.values():
            n = len(group)
            weights = {t[2]["sample_weight"] for t in group}
            if len(weights) != 1:
                raise ValueError("Inconsistent sample weights inside a single stratum")
            population = n * next(iter(weights))
            if population > n:
                if n < 2:
                    ci_withheld = "noncensus_singleton_stratum"
                    break
                residual = [y - result * x for y, x, _ in group]
                variance_total += population ** 2 * (1 - n / population) * variance(residual) / n
        if ci_withheld or not variance_total:
            interval = None
            note = ci_withheld or "zero_linearized_variation_interval_not_informative"
        else:
            half = 1.96 * sqrt(variance_total) / denominator
            interval = [max(0.0, result - half), min(1.0, result + half)]
            note = "stratified_ratio_linearization_finite_population_correction"
    else:
        z = 1.96
        center = (result + z * z / (2 * denominator)) / (1 + z * z / denominator)
        half = z * sqrt(result * (1 - result) / denominator + z * z / (4 * denominator ** 2)) / (1 + z * z / denominator)
        interval, note = [max(0.0, center - half), min(1.0, center + half)], "descriptive_wilson"
    return {"value": result, "status": "computed", "numerator": numerator, "denominator": denominator,
            "interval_95": interval, "interval_method": note}


def human_evidence(row, *, include_partial):
    status = row["human"]["evidence_support"]
    if status == "UNCERTAIN":
        return None
    return status == "DIRECT_SUPPORT" or (include_partial and status == "PARTIAL_SUPPORT")


def admission(row, stage):
    decision = row["gate_by_arm"].get(stage, {})
    if stage == "quality":
        return row["fixed_gate"].get("quality_pass") is True
    return decision.get("status") == "replaced"


def confusion(data, stage="quality", *, include_partial=False, weighted=False):
    counts = Counter()
    for row in data:
        actual = human_evidence(row, include_partial=include_partial)
        if actual is None:
            continue
        admitted = admission(row, stage)
        counts["TP" if admitted and actual else "FP" if admitted else "FN" if actual else "TN"] += (
            row["sample_weight"] if weighted else 1)
    return {k: counts[k] for k in ("TP", "FP", "FN", "TN")}


def evidence_metric(data, stage="quality", *, include_partial=False, weighted=False, frame_complete=True):
    def outcome(row):
        gold = human_evidence(row, include_partial=include_partial)
        accepted = admission(row, stage)
        tp = gold is True and accepted
        fp = gold is False and accepted
        tn = gold is False and not accepted
        fn = gold is True and not accepted
        return tp, fp, tn, fn

    rules = {"evidence_precision": (0, (0, 1)), "evidence_recall": (0, (0, 3)),
             "false_acceptance_rate": (1, (1, 2)), "false_rejection_rate": (3, (0, 3))}
    out = {}
    for name, (component, denominator) in rules.items():
        def membership(row):
            outcomes = outcome(row)
            return int(outcomes[component]), sum(outcomes[i] for i in denominator)

        out[name] = ratio(data, membership, weighted=weighted, frame_complete=frame_complete)
    out["counts"] = confusion(data, stage, include_partial=include_partial, weighted=weighted) if (
        not weighted or frame_complete) else None
    return out


def replacement_precision(data, arm, **opts):
    def member(row):
        human = row["human"]["replacement_suitability"]
        considered = admission(row, arm) and human != "UNCERTAIN"
        return int(considered and human == "REPLACE"), int(considered)
    return ratio(data, member, **opts)


def measured(data, *, weighted, frame_complete):
    options = {"weighted": weighted, "frame_complete": frame_complete}
    dimensions = {}
    for field in DIMENSIONS:
        def denominator(row):
            return row["human"][field] in DETERMINATE[field]

        def classified(row):
            return denominator(row) and row["rule_dimension_predictions"].get(field) is not None

        def matching(row):
            return classified(row) and row["rule_dimension_predictions"][field] == row["human"][field]

        dimensions[field] = {
            "conditional_accuracy": ratio(data, lambda row: (matching(row), classified(row)), **options),
            "rule_prediction_coverage": ratio(data, lambda row: (classified(row), denominator(row)), **options),
        }
    return {
        "quality_strict": evidence_metric(data, **options),
        "quality_including_partial": evidence_metric(data, include_partial=True, **options),
        "full_gate_strict": evidence_metric(data, stage="E4-B1", **options),
        "replacement_precision": {arm: replacement_precision(data, arm, **options) for arm in ("E4-B1", "E3-B1")},
        "dimensions": dimensions,
    }


def analyze(selected, labels, stages):
    results = {}
    for cohort in ("development", "null"):
        expected = [r for r in selected.values() if r["cohort"] == cohort]
        group = [{**r, **stages[r["case_id"]], "human": labels[r["case_id"]]}
                 for r in expected if r["case_id"] in labels]
        by_stratum = defaultdict(list)
        for row in group:
            by_stratum[row["stratum"]].append(row)
        frame = len(group) == len(expected)
        strata = Counter(r["stratum"] for r in expected)
        item = {"cases_expected": len(expected), "complete_human_labels": len(group), "pending": len(expected) - len(group),
                "label_progress_by_stratum": {stratum: {"expected": count, "labelled": len(by_stratum[stratum])}
                                               for stratum, count in sorted(strata.items())}}
        if group:
            item.update({
                "unweighted": measured(group, weighted=False, frame_complete=True),
                "weighted": measured(group, weighted=True, frame_complete=frame),
                "by_stratum": {name: (measured(by_stratum[name], weighted=False, frame_complete=True)
                                      if by_stratum[name] else {"status": "pending_human_annotations"})
                               for name in sorted(strata)},
            })
            if cohort == "null":
                item["no_answer_false_acceptance"] = {
                    mode: {stage: evidence_metric(group, stage=stage, weighted=weight,
                                                   frame_complete=(not weight or frame))["false_acceptance_rate"]
                           for stage in ("quality", "E4-B1")}
                    for mode, weight in (("unweighted", False), ("weighted", True))
                }
        else:
            item["status"] = "pending_human_annotations"
        results[cohort] = item
    return results


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("human_csv", type=Path)
    args = parser.parse_args()
    public = {r["case_id"]: r for r in rows(OUT / "sampled_candidates.jsonl")}
    key = {r["case_id"]: r for r in rows(OUT / "sampling_key.jsonl")}
    stages = {r["case_id"]: r for r in rows(OUT / "gate_stage_trace.jsonl")}
    if public.keys() != key.keys() or key.keys() != stages.keys():
        raise ValueError("Blinded IDs do not align with internal traces")
    labels = read_sheet(args.human_csv, public)
    results = analyze(key, labels, stages)
    write(OUT / "human_label_metrics.json", {
        "real_human_annotations_required": True, "complete_human_labels": len(labels),
        "human_source": args.human_csv.name,
        "definitions": {
            "positive": "DIRECT_SUPPORT; separate inclusive analysis adds PARTIAL_SUPPORT",
            "excluded": "UNCERTAIN evidence or suitability and nonapplicable or uncertain dimension labels",
            "primary_acceptance": "E4-B1 historical quality_pass; full gate E4-B1 and E3-B1 replacement reported separately",
            "wrong_replacement": "Human KEEP_INCUMBENT or EITHER counted as not affirmatively REPLACE; uncertainty excluded",
            "no_answer_metrics": "separate null cohort, human-invalid denominator; no labels inferred from null flag",
            "weighted": "inverse inclusion probability ratio; no estimates until each sampled row is explicitly labeled",
        },
        "cohorts": results,
    })
    print(f"Explicit human labels {len(labels)}/{len(key)}", flush=True)


if __name__ == "__main__":
    main()

