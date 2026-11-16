from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
from unittest import TestCase
from unittest.mock import patch
import json
import os
import sys

from analysis.reranking import prepare_models
from analysis.reranking.backend import DEFAULT_MODELS, JINA


class PrepareModelsTests(TestCase):
    def test_reports_each_download_and_fails_if_one_is_unavailable(self):
        cross_encoder = ModuleType("fastembed.rerank.cross_encoder")
        cross_encoder.TextCrossEncoder = SimpleNamespace(list_supported_models=lambda: [
            {"model": name, "license": "fixture"} for name in DEFAULT_MODELS
        ])
        onnxruntime = ModuleType("onnxruntime")
        onnxruntime.get_available_providers = lambda: ["CPUExecutionProvider"]
        for missing, expected_exit in ((None, 0), (JINA, 1)):
            def receipt(name, cache, license_name):
                return {
                    "model_id": name,
                    "status": "download_failed" if name == missing else "downloaded",
                }

            with self.subTest(missing=missing), TemporaryDirectory() as temp:
                stdout = StringIO()
                with (
                    patch.object(prepare_models, "__file__", str(Path(temp) / "prepare_models.py")),
                    patch.object(prepare_models, "_download", side_effect=receipt),
                    patch.object(prepare_models.metadata, "version", return_value="fixture"),
                    patch.object(prepare_models.shutil, "disk_usage", return_value=SimpleNamespace(free=1_000_000_000)),
                    patch.dict(os.environ),
                    patch.dict(sys.modules, {
                        "fastembed.rerank.cross_encoder": cross_encoder, "onnxruntime": onnxruntime,
                    }),
                    redirect_stdout(stdout),
                ):
                    self.assertEqual(prepare_models.main(), expected_exit)
                results = [json.loads(line) for line in stdout.getvalue().splitlines()]
                self.assertEqual([item["model_id"] for item in results], list(DEFAULT_MODELS))
                self.assertEqual([item["model_id"] for item in results if item["status"] != "downloaded"],
                                 [] if missing is None else [missing])
