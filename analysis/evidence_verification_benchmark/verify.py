"""Verify annotation isolation, repeat the archived sample, and record artifact integrity."""

from collections import Counter, defaultdict
from math import isclose
from pathlib import Path
from time import perf_counter
import csv
import re
import subprocess
import sys

from analysis.besa_v0.experiment import MODEL_SPEC, read, rows, write
from analysis.ege_v0.run import fingerprint
from analysis.egr_v0.run import ROOT
from analysis.egr_v0.run import OUT as EGR, RUN, documents_and_chunks

from .build import LABELS, FREE_FIELDS, OUT, PUBLIC_KEYS
from .sampling import draw


REQUIRED = (
    "EVIDENCE_VERIFICATION_AUDIT.md", "EVIDENCE_VERIFICATION_REPORT.md", "benchmark_manifest.json",
    "sampled_candidates.jsonl", "annotation_packets.md", "annotation_template.csv", "sampling_statistics.csv",
    "source_chunk_trace.jsonl", "gate_stage_trace.jsonl", "annotation_guide.md", "threshold_sensitivity.csv",
    "threshold_sensitivity_summary.json", "sampling_key.jsonl", "gold_source_diagnostics.jsonl",
    "literal_visibility_summary.json", "human_label_metrics.json", "build_profile.json",
)
BLIND_NESTED_FORBIDDEN = {
    "query_id", "document_id", "chunk_id", "gold", "gold_source", "known_necessary_source", "evidence_id",
    "gate", "gate_decisions", "e2_replacement", "stratum", "overlapping_tags", "jina_score", "quality_pass",
    "novelty_pass", "margin_pass", "fallback_reason", "cohort", "split", "sample_weight", "answer",
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def no_private_keys(value):
    if isinstance(value, dict):
        return not (set(value) & BLIND_NESTED_FORBIDDEN) and all(no_private_keys(v) for v in value.values())
    if isinstance(value, list):
        return all(no_private_keys(v) for v in value)
    return True


def check_history():
    previous = read(EGR / "experiment_manifest.json")
    expected = {ROOT / name: value for name, value in previous["historical_artifacts"].items()}
    expected.update({EGR / name: value for name, value in previous["artifacts"].items()})
    for path, status in expected.items():
        require(fingerprint(path) == status, f"Historical checksum mismatch: {path}")
    sensitivity = read(OUT / "threshold_sensitivity_summary.json")["cross_checks"]["historical_input_sha256_unchanged"]
    for name, digest in sensitivity.items():
        require(fingerprint(EGR / name)["sha256"] == digest, f"Changed threshold input: {name}")
    metadata = read(MODEL_SPEC)
    tokenizer = Path(metadata["model_path"]).parents[1] / "tokenizer.json"
    require(fingerprint(tokenizer)["sha256"] == metadata["tokenizer_sha256"], "Model tokenizer checksum mismatch")
    unlisted = {EGR / "experiment_manifest.json", RUN / "scores.sqlite3", tokenizer}
    before = {p: fingerprint(p) for p in sorted(unlisted)}
    return expected, before


def check_sample():
    selected, strata, design = draw(ROOT)
    triplet = lambda r: (r["case_id"], r["query_id"], r["document_id"])
    stored = list(rows(OUT / "sampling_key.jsonl"))
    require(set(map(triplet, selected)) == set(map(triplet, stored)), "Historical sample draw does not repeat")
    require(len(stored) == 350 and len({(r["query_id"], r["document_id"]) for r in stored}) == 350,
            "Sample keys must be unique")
    key = {r["case_id"]: r for r in stored}
    case_counts = Counter(r["cohort"] for r in stored)
    require(case_counts == {"development": 260, "null": 90}, "Wrong annotation sample sizes")
    per_stratum = Counter((r["cohort"], r["stratum"]) for r in stored)
    with (OUT / "sampling_statistics.csv").open(encoding="utf-8-sig", newline="") as stream:
        recorded = list(csv.DictReader(stream))
    expected_strata = {(r["cohort"], r["stratum"]): r for r in strata}
    totals, weights = Counter(), Counter()
    for row in recorded:
        keys = row["cohort"], row["stratum"]
        n, population = int(row["selected"]), int(row["population"])
        require(keys in expected_strata and n == per_stratum[keys] and
                population == expected_strata[keys]["population"] and (not population or n > 0),
                "Missing probability sample coverage in a nonempty stratum")
        totals[row["cohort"]] += population
        if n:
            require(isclose(float(row["sample_weight"]), population / n) and
                    isclose(float(row["inclusion_probability"]), n / population),
                    "Sampling weights do not match the repeatable design")
    for row in stored:
        weights[row["cohort"]] += row["sample_weight"]
    require(totals == design["population"] and all(isclose(weights[k], v) for k, v in totals.items()),
            "Weighted sample does not expand back to its finite population")
    return key, design


def check_packets(key):
    public = {}
    for record in rows(OUT / "sampled_candidates.jsonl"):
        cid = record["case_id"]
        require(cid not in public and cid in key and set(record) == PUBLIC_KEYS and no_private_keys(record),
                "Public annotation unit contains internal metadata or duplicate IDs")
        require([r["rank"] for r in record["baseline_top10"]] == list(range(1, 11)) and
                record["comparison_target"]["baseline_rank"] in (8, 9, 10),
                "Missing baseline or comparison material")
        for role in ("challenger", "comparison_target"):
            parts = record[role]["scored_chunks"]
            require(len(parts) in (2, 3) and all(b["original_full_text"].startswith(b["jina_visible_text"]) for b in parts),
                    "Missing original text or visible prefix")
        public[cid] = record
    require(public.keys() == key.keys(), "Some public units are missing")
    packets = OUT / "annotation_packets.md"
    with packets.open(encoding="utf-8") as handle:
        text = handle.read()
    require(re.findall(r"^## (evb-\d+)$", text, re.M) == list(public), "Packet order or coverage changed")
    for field in ("Gold Source", "抽样层", "quality_pass", "gold_required_quality_reject",
                  "gate_stage_trace", "sampling_key.jsonl", "gold_source_diagnostics.jsonl"):
        require(field not in text, f"Internal {field} exposed to annotators")
    markers = [("challenge", "### 挑战者"), ("compare", "### 比较对象"), ("baseline", "### 当前十篇文档")]
    for _, marker in markers:
        require(text.count(marker) == len(public), f"Missing Markdown packet element: {marker}")
    with (OUT / "annotation_template.csv").open(encoding="utf-8-sig", newline="") as handle:
        template = csv.DictReader(handle)
        require(template.fieldnames == ["case_id", *LABELS, *FREE_FIELDS], "Unexpected template schema")
        entries = list(template)
    require([r["case_id"] for r in entries] == list(public), "Annotation template misaligned")
    blank = sum(all(not row[key].strip() for key in (*LABELS, *FREE_FIELDS)) for row in entries)
    return public, blank


def check_traces(key, public):
    _, chunks = documents_and_chunks()
    index = {c.chunk_id: c for blocks in chunks.values() for c in blocks}
    views, distinct = defaultdict(set), set()
    for row in rows(OUT / "source_chunk_trace.jsonl"):
        cid = row["case_id"]
        chunk = index[row["chunk_id"]]
        raw, visible = row["original_chunk_text"], row["jina_visible_chunk_text"]
        require(row["query_id"] == key[cid]["query_id"] and chunk.document_id == row["document_id"] and
                chunk.text == raw and raw[:row["model_visible_chunk_chars"]] == visible and
                public[cid]["question"] == row["jina_query_before_truncation"] == row["jina_visible_query_text"],
                "Source trace cannot recover an original or scored segment")
        k = row["input_tokens"]
        require(0 < k <= 256 and all(len(row[field]) == k for field in (
            "token_ids_unpadded", "attention_mask_unpadded", "token_type_ids_unpadded", "sequence_ids", "offsets"
        )), "Token trace length mismatch")
        require(row["ref"] not in views[cid], "Duplicate case passage reference")
        views[cid].add(row["ref"])
        distinct.add((row["query_id"], row["chunk_id"]))
    for cid, record in public.items():
        refs = {block["ref"] for role in ("challenger", "comparison_target") for block in record[role]["scored_chunks"]}
        require(views[cid] == refs | {f"B{i}" for i in range(1, 11)}, "Missing challenger/incumbent or baseline trace")
    for filename in ("gate_stage_trace.jsonl", "gold_source_diagnostics.jsonl"):
        trace = list(rows(OUT / filename))
        require(len(trace) == len(key) and {r["case_id"] for r in trace} == set(key) and
                all((r["query_id"], r["document_id"]) == (key[r["case_id"]]["query_id"], key[r["case_id"]]["document_id"])
                    for r in trace),
                f"Invalid internal join: {filename}")
    return sum(map(len, views.values())), len(distinct)


def test_suites():
    record = {}
    suite_base = ("tests", "analysis/reranking/tests", "analysis/reranking_optimization/tests",
                  "analysis/besa_v0/tests", "analysis/ege_v0/tests", "analysis/egr_v0/tests")
    for suite in (*suite_base, "analysis/evidence_verification_benchmark/tests"):
        run = subprocess.run([sys.executable, "-B", "-m", "unittest", "discover", "-s", suite],
                             cwd=ROOT, capture_output=True, text=True, check=True)
        log = run.stderr + run.stdout
        count = re.search(r"Ran (\d+) tests?", log)
        skips = re.search(r"skipped=(\d+)", log)
        require(count, f"Could not parse test execution: {suite}")
        tested, skipped = int(count[1]), int(skips[1]) if skips else 0
        record[suite] = {"passed": tested - skipped, "skipped": skipped}
        print(suite, record[suite], flush=True)
    require(sum(record[s]["passed"] for s in suite_base) == 126, "Expected 126 previously established offline tests")
    return record


def main():
    started = perf_counter()
    checks, extras = check_history()
    key, design = check_sample()
    public, blank = check_packets(key)
    checked, unique = check_traces(key, public)
    build = read(OUT / "build_profile.json")
    sensitivity = read(OUT / "threshold_sensitivity_summary.json")
    human = read(OUT / "human_label_metrics.json")
    require((checked, unique) == (build["checked_pair_views"], build["unique_checked_pair_views"]),
            "Build and final verification trace counts disagree")
    require(human["complete_human_labels"] <= len(key), "Human labels exceed sample size")
    with (OUT / "threshold_sensitivity.csv").open(encoding="utf-8-sig", newline="") as handle:
        curve = list(csv.DictReader(handle))
    require(len(curve) == 204 and all(
        r["independent_labels_status"] == "pending_human_annotation" and
        not r["potential_false_acceptance_count"] and not r["potential_false_rejection_count"] for r in curve
    ), "Unlabeled threshold results claim semantic false counts")
    for name in REQUIRED:
        if name != "benchmark_manifest.json":
            require((OUT / name).is_file(), f"Missing expected deliverable: {name}")
    recorded = test_suites()
    integrity_after = {p: fingerprint(p) for p in extras}
    require(integrity_after == extras, "A read-only cache or tokenizer changed during verification")
    manifest_path = OUT / "benchmark_manifest.json"
    paths = [p for p in OUT.glob("*") if p.is_file() and p != manifest_path]
    paths.extend((OUT / "tests").glob("test_*.py"))
    write(manifest_path, {
        "protocol": "Evidence Verification Benchmark (frozen EGR-v0, no ranking or threshold changes)",
        "sampling": design, "annotator_reads_only": ["annotation_packets.md", "annotation_template.csv", "annotation_guide.md"],
        "human_labels": {"completed": human["complete_human_labels"], "template_blank_rows": blank,
                         "independent_metrics": "pending" if human["complete_human_labels"] == 0 else "see human_label_metrics.json"},
        "time_seconds": {"sampling": build["sampling_seconds"], "packet_and_trace_build": build["packet_and_trace_seconds"],
                         "source_audit_literal_replay": read(OUT / "literal_visibility_summary.json")["wall_seconds"],
                         "sensitivity": sensitivity["wall_seconds"],
                         "final_verification_with_tests": perf_counter() - started,
                         "manual_source_inspection": None},
        "source_trace_rows": checked, "unique_source_trace_query_chunk_pairs": unique,
        "threshold_replay_rows": len(curve), "answerable_holdout_run": 0,
        "new_jina_pairs": 0, "new_embeddings": 0, "tests": recorded,
        "historical_checks_against_prior_manifest": len(checks),
        "additionally_fingerprinted_read_only": {p.relative_to(ROOT).as_posix(): status for p, status in extras.items()},
        "artifacts": {p.relative_to(OUT).as_posix(): fingerprint(p) for p in sorted(paths)},
    })
    print(f"Verified {len(key)} cases, {checked} input records, {len(curve)} sensitivity rows, and {len(checks)} historical checks")


if __name__ == "__main__":
    main()

