from __future__ import annotations

import csv
import io
import json
import shutil
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

BENCHMARK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCHMARK_DIR))

from human_review_workbench import (  # noqa: E402
    CASE_IDS,
    COLUMNS,
    EXPORT_NAME,
    Workbench,
    WorkbenchServer,
)


VISIBLE_TEXT = '仲裁庭写道："驳回诉请"。\n这是评分可见的第二行。'
FOLDED_SUFFIX = "这段只在打分范围外。"


def packet_text():
    source = "测试通讯 · 发布 2025-01-01"
    metadata = f"媒体 {source}\n\n原报道 https://fixture.invalid/article"
    top_ten = "\n\n".join(
        f"**B{rank} · 测试文章 {rank}**\n\n{source}\n\n"
        f"~~~text\n前十证词 {rank}。\n~~~"
        for rank in range(1, 11)
    )
    cases = []
    for case_id in CASE_IDS:
        cases.append(f"""## {case_id}

{case_id} 此题只用于工作台测试。

### 挑战者

**测试挑战者**

{metadata}

**C1 · 模型可见部分，已截断**

~~~text
{VISIBLE_TEXT}
~~~

<details><summary>完整原始 Chunk，仅供定位模型窗外的候选事实</summary>

~~~text
{VISIBLE_TEXT}
{FOLDED_SUFFIX}
~~~
</details>

### 比较对象（原第 10 位）

**测试比较对象**

{metadata}

**I1 · 完整可见**

~~~text
比较证词。
~~~

### 当前十篇文档的相关片段

{top_ten}
""")
    return "\n".join(cases)


class FixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        folder = tempfile.TemporaryDirectory()
        cls.addClassCleanup(folder.cleanup)
        cls.directory = Path(folder.name)
        for name in ("annotation_guide.md", "annotation_template.csv"):
            shutil.copyfile(BENCHMARK_DIR / name, cls.directory / name)
        (cls.directory / "annotation_packets.md").write_text(packet_text(), encoding="utf-8", newline="\n")


class WorkbenchTests(FixtureTests):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.outputs = Path(self.temp.name) / "human_review_outputs"
        self.workbench = Workbench(self.directory, self.outputs)

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def values(**changes):
        values = {
            "evidence_support": "NO_SUPPORT",
            "entity_role": "NOT_APPLICABLE",
            "relation_support": "UNRELATED",
            "temporal_scope": "NOT_APPLICABLE",
            "temporal_kind": "NOT_APPLICABLE",
            "evidence_novelty": "NOVEL",
            "replacement_suitability": "KEEP_INCUMBENT",
            "reason": "可见窗口没有提供题目要求的事实。",
            "witness_ref": "NONE",
            "witness_quote": "",
        }
        values.update(changes)
        return values

    def test_bootstrap_loads_all_cases_and_seven_dimensions(self):
        payload = self.workbench.bootstrap()
        self.assertEqual(tuple(payload["ids"]), CASE_IDS)
        self.assertEqual(payload["total"], 350)
        self.assertEqual(len(payload["fields"]), 7)
        self.assertEqual(tuple(field["name"] for field in payload["fields"]), COLUMNS[1:8])

    def test_sessions_resume_and_keep_annotators_isolated(self):
        first = self.workbench.session("reviewer01")
        self.assertEqual(first["completed"], [])
        saved = self.workbench.save("reviewer01", "evb-0001", self.values())
        self.assertEqual(saved["completed"], ["evb-0001"])

        reopened = Workbench(self.directory, self.outputs)
        resumed = reopened.session("reviewer01")
        other = reopened.session("reviewer02")
        self.assertEqual(resumed["completed"], ["evb-0001"])
        self.assertEqual(resumed["last_case"], "evb-0001")
        self.assertEqual(other["completed"], [])
        self.assertIsNone(self.workbench.get_case("reviewer02", "evb-0001")["annotation"])
        self.assertEqual(
            self.workbench.get_case("reviewer01", "evb-0001")["annotation"],
            self.values(),
        )

    def test_position_persists_without_marking_case_complete(self):
        self.workbench.session("reviewer01")
        self.workbench.position("reviewer01", "evb-0007")
        state = self.workbench.session("reviewer01")
        self.assertEqual(state["last_case"], "evb-0007")
        self.assertEqual(state["completed"], [])

    def test_invalid_annotator_and_case_are_rejected(self):
        for value in ("", "../reviewer", "Reviewer", "con", "a" * 41):
            with self.subTest(annotator=value), self.assertRaises(ValueError):
                self.workbench.session(value)
        with self.assertRaises(ValueError):
            self.workbench.get_case("reviewer01", "../../outside")

    def test_invalid_labels_and_missing_reason_are_rejected(self):
        self.workbench.session("reviewer01")
        invalid = self.values(evidence_support="SUPPORTED")
        with self.assertRaisesRegex(ValueError, "合法标签"):
            self.workbench.save("reviewer01", "evb-0001", invalid)
        invalid = self.values(reason="  ")
        with self.assertRaisesRegex(ValueError, "判断理由"):
            self.workbench.save("reviewer01", "evb-0001", invalid)

    def test_citable_evidence_must_quote_visible_challenger_window(self):
        self.workbench.session("reviewer01")
        visible = self.workbench.visible["evb-0001"]["C1"]
        self.assertEqual(visible, VISIBLE_TEXT)
        valid = self.values(
            evidence_support="DIRECT_SUPPORT",
            witness_ref="C1",
            witness_quote=VISIBLE_TEXT,
        )
        self.workbench._require_values("evb-0001", valid)

        for invalid in (
            self.values(
                evidence_support="DIRECT_SUPPORT",
                witness_ref="I1",
                witness_quote="比较证词。",
            ),
            self.values(
                evidence_support="DIRECT_SUPPORT",
                witness_ref="C1",
                witness_quote=FOLDED_SUFFIX,
            ),
            self.values(evidence_support="DIRECT_SUPPORT", witness_ref="NONE"),
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.workbench._require_values("evb-0001", invalid)

    def test_none_witness_is_only_allowed_without_a_citable_support_label(self):
        self.workbench.session("reviewer01")
        invalid = self.values(
            evidence_support="CONTRADICTION",
            witness_ref="NONE",
            witness_quote="",
        )
        with self.assertRaisesRegex(ValueError, "NONE"):
            self.workbench._require_values("evb-0001", invalid)

    def test_progress_tampering_is_detected_without_overwriting_the_file(self):
        self.workbench.session("reviewer01")
        progress = self.outputs / "reviewer01" / "human_review_progress.json"
        original = progress.read_bytes()
        state = json.loads(original)
        state["annotator"] = "reviewer02"
        progress.write_text(json.dumps(state), encoding="utf-8")
        tampered = progress.read_bytes()
        with self.assertRaisesRegex(ValueError, "无法读取或验证"):
            self.workbench.session("reviewer01")
        self.assertEqual(progress.read_bytes(), tampered)

    def test_csv_export_has_all_350_cases_bom_and_round_trips_text(self):
        self.workbench.session("reviewer01")
        reason = '第一行，包含逗号和"引号"。\n第二行：中文内容。'
        annotation = self.values(reason=reason)
        self.workbench.save("reviewer01", "evb-0001", annotation)

        exported = self.workbench.export("reviewer01")
        self.assertTrue(exported.startswith(b"\xef\xbb\xbf"))
        self.assertEqual(
            (self.outputs / "reviewer01" / EXPORT_NAME).read_bytes(),
            exported,
        )
        rows = list(csv.reader(io.StringIO(exported.decode("utf-8-sig"), newline="")))
        self.assertEqual(tuple(rows[0]), COLUMNS)
        self.assertEqual(len(rows), 351)
        self.assertEqual(rows[1][0], "evb-0001")
        self.assertEqual(rows[1][8], reason)
        self.assertEqual(rows[-1][0], "evb-0350")
        self.assertTrue(all(value == "" for value in rows[-1][1:]))


class WorkbenchHttpTests(FixtureTests):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.workbench = Workbench(
            self.directory,
            Path(self.temp.name) / "human_review_outputs",
        )
        self.server = WorkbenchServer(self.workbench, 0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"
        self.headers = {
            "Host": f"127.0.0.1:{self.server.server_port}",
            "Origin": self.base,
        }

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.temp.cleanup()

    def request(self, path, *, method="GET", payload=None, headers=None):
        request_headers = dict(self.headers)
        if headers:
            request_headers.update(headers)
        data = None
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            request_headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            self.base + path,
            data=data,
            headers=request_headers,
            method=method,
        )
        return urllib.request.urlopen(request, timeout=5)

    def test_http_session_case_save_and_export_round_trip(self):
        bootstrap = json.loads(self.request("/api/bootstrap").read())
        self.assertEqual(bootstrap["total"], 350)

        session = json.loads(self.request(
            "/api/session",
            method="POST",
            payload={"annotator": "reviewer-http"},
        ).read())
        self.assertEqual(session["completed"], [])

        case = json.loads(self.request(
            "/api/case?id=evb-0001&annotator=reviewer-http",
        ).read())
        self.assertEqual(case["case"]["id"], "evb-0001")
        self.assertIsNone(case["annotation"])
        groups = case["case"]["groups"]
        self.assertEqual([group["key"] for group in groups], ["C", "I", "B"])
        self.assertEqual(groups[0]["windows"][0]["text"], VISIBLE_TEXT)
        self.assertIn(FOLDED_SUFFIX, groups[0]["windows"][0]["raw"])
        self.assertEqual([window["ref"] for window in groups[2]["windows"]],
                         [f"B{n}" for n in range(1, 11)])

        values = WorkbenchTests.values()
        saved = json.loads(self.request(
            "/api/save",
            method="POST",
            payload={"annotator": "reviewer-http", "case_id": "evb-0001", "values": values},
        ).read())
        self.assertEqual(saved["completed"], ["evb-0001"])

        response = self.request(
            "/api/export",
            method="POST",
            payload={"annotator": "reviewer-http"},
        )
        exported = response.read()
        self.assertTrue(exported.startswith(b"\xef\xbb\xbf"))
        self.assertIn(EXPORT_NAME, response.headers["Content-Disposition"])

    def test_servers_cannot_share_an_output_directory(self):
        reopened = Workbench(self.directory, self.workbench.outputs)
        with self.assertRaisesRegex(ValueError, "标注输出目录正在被另一个工作台使用"):
            WorkbenchServer(reopened, 0)
        other = Workbench(self.directory, self.workbench.outputs.parent / "other-outputs")
        with WorkbenchServer(other, 0):
            pass

    def test_unknown_routes_do_not_serve_inputs(self):
        with self.assertRaises(urllib.error.HTTPError) as not_found:
            self.request("/annotation_packets.md")
        self.assertEqual(not_found.exception.code, 404)
        not_found.exception.close()

    def test_rejects_foreign_origin_and_wrong_content_type(self):
        with self.assertRaises(urllib.error.HTTPError) as foreign:
            self.request(
                "/api/bootstrap",
                headers={"Origin": "http://example.invalid"},
            )
        self.assertEqual(foreign.exception.code, 403)
        foreign.exception.close()

        with self.assertRaises(urllib.error.HTTPError) as wrong_type:
            request = urllib.request.Request(
                self.base + "/api/session",
                data=b'{"annotator":"reviewer01"}',
                headers={**self.headers, "Content-Type": "text/plain"},
                method="POST",
            )
            urllib.request.urlopen(request, timeout=5)
        self.assertEqual(wrong_type.exception.code, 400)
        wrong_type.exception.close()


if __name__ == "__main__":
    unittest.main()
