import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from analysis.evidence_verification_benchmark.build import (
    FREE_FIELDS, LABELS, PUBLIC_KEYS, human_record, preserve_template, restore, surface_dates,
)
from analysis.evidence_verification_benchmark.human_metrics import (
    analyze, evidence_metric, ratio, read_sheet,
)


def case():
    return human_record({"case_id": "example", "gold": "not exposed", "gate": "not exposed"},
                        "Which event happened?", {"scored_chunks": [
                            {"ref": "C1", "jina_visible_text": "A court ruled today."}]},
                        {"scored_chunks": [{"ref": "I1", "jina_visible_text": "Other matter."}]},
                        [{"ref": "B1", "jina_visible_excerpt": "Other matter."}])


def label(**updates):
    data = dict(evidence_support="DIRECT_SUPPORT", entity_role="NOT_APPLICABLE",
                relation_support="SUPPORTED", temporal_scope="UNKNOWN", temporal_kind="EVENT",
                evidence_novelty="NOVEL", replacement_suitability="KEEP_INCUMBENT",
                reason="Use an original quote.", witness_ref="C1", witness_quote="A court ruled today.")
    data.update(updates)
    return data


def metric_row(accept, evidence, weight, stratum):
    return {"fixed_gate": {"quality_pass": accept}, "gate_by_arm": {},
            "human": label(evidence_support=evidence), "sample_weight": weight, "stratum": stratum}


class BenchmarkTests(unittest.TestCase):
    def test_blind_record_excludes_sampling_gold_and_decisions(self):
        self.assertEqual(set(case()), PUBLIC_KEYS)
        self.assertNotIn("query_id", case())

    def test_scope_restoration_keeps_full_chunk_and_reports_exact_visible_prefix(self):
        vocab = {token: i for i, token in enumerate(("?", "Q", "Title", "a", "b", "c", "d", "e"))}
        tokenizer = Tokenizer(WordLevel(vocab, unk_token="?"))
        tokenizer.pre_tokenizer = Whitespace()
        full = Tokenizer.from_str(tokenizer.to_str())
        tokenizer.enable_truncation(5, strategy="longest_first", direction="right", stride=0)
        block = SimpleNamespace(text="a b c d e")
        pair = SimpleNamespace(chunk_id="d::1", text="Title\n" + block.text)
        score = dict(input_tokens=5, original_tokens=7, truncated=True,
                     model_visible_query_chars=1, model_visible_chunk_chars=5)
        result = restore("Q", pair, block, score, tokenizer, full)
        self.assertEqual(result["original_chunk_text"], "a b c d e")
        self.assertEqual(result["jina_visible_chunk_text"], "a b c")
        self.assertEqual(result["jina_visible_document_text"], "Title\na b c")
        self.assertEqual(result["token_ids_unpadded"], [1, 2, 3, 4, 5])
        score["model_visible_chunk_chars"] = 6
        with self.assertRaises(ValueError):
            restore("Q", pair, block, score, tokenizer, full)

    def test_date_strings_include_day_and_year_without_inferred_scope(self):
        self.assertEqual(surface_dates("The event on Sep 28, 2023 was reported in 2024."),
                         ["Sep 28, 2023", "2024"])

    def test_blank_template_and_preservation_of_actual_annotations(self):
        import csv
        example = case()
        with TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            path = Path(tmp) / "annotation.csv"
            self.assertEqual(preserve_template(path, [example]), "blank_template")
            self.assertEqual(read_sheet(path, {"example": example}), {})
            with path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["case_id", *LABELS, *FREE_FIELDS])
                writer.writeheader()
                writer.writerow({"case_id": "example", **label()})
            before = path.read_bytes()
            self.assertEqual(preserve_template(path, [example]), "kept_existing_human_entries")
            self.assertEqual(path.read_bytes(), before)
            parsed = read_sheet(path, {"example": example})["example"]
            self.assertEqual([parsed[k] for k in ("evidence_support", "entity_role", "relation_support",
                                                   "evidence_novelty", "replacement_suitability")],
                             ["DIRECT_SUPPORT", "NOT_APPLICABLE", "SUPPORTED", "NOVEL", "KEEP_INCUMBENT"])

    def test_label_options_and_model_window_quote_are_validated(self):
        import csv
        with TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            path = Path(tmp) / "annotation.csv"
            for update in ({"entity_role": "CEO"}, {"relation_support": "LIKELY"},
                           {"evidence_novelty": "NEW"}, {"replacement_suitability": "NOVEL"},
                           {"temporal_kind": "PUBLISHED_EVENT"}, {"witness_quote": "Hidden original suffix."}):
                with path.open("w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=["case_id", *LABELS, *FREE_FIELDS])
                    writer.writeheader()
                    writer.writerow({"case_id": "example", **label(**update)})
                with self.assertRaises(ValueError):
                    read_sheet(path, {"example": case()})

    def test_weighted_rate_and_zero_denominator_use_separate_status(self):
        sample = [metric_row(True, "DIRECT_SUPPORT", 1, "first"),
                  metric_row(True, "PARTIAL_SUPPORT", 3, "second"),
                  metric_row(False, "DIRECT_SUPPORT", 3, "second"),
                  metric_row(False, "UNCERTAIN", 1, "first")]
        strict = evidence_metric(sample, weighted=True)
        self.assertEqual(strict["counts"], {"TP": 1, "FP": 3, "FN": 3, "TN": 0})
        self.assertEqual(strict["evidence_precision"]["value"], .25)
        self.assertEqual(evidence_metric(sample, include_partial=True)["evidence_precision"]["value"], 1.0)
        self.assertEqual(evidence_metric(sample, stage="E4-B1")["evidence_precision"]["status"],
                         "undefined_zero_denominator")
        self.assertEqual(ratio(sample, lambda _row: (1, 1), weighted=True, frame_complete=False)["status"],
                         "pending_full_human_cohort")

    def test_labels_pending_cannot_convert_gate_outputs_into_truth(self):
        design = {"case1": {"case_id": "case1", "cohort": "development", "stratum": "primary"},
                  "case2": {"case_id": "case2", "cohort": "null", "stratum": "other"}}
        result = analyze(design, {}, {"case1": {"fixed_gate": {"quality_pass": True}}})
        self.assertEqual(result["development"]["status"], "pending_human_annotations")
        self.assertEqual(result["null"]["label_progress_by_stratum"]["other"]["labelled"], 0)


if __name__ == "__main__":
    unittest.main()

