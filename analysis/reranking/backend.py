"""Pinned local FastEmbed/ONNX CPU reranking and per-pair token accounting."""

from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import file_digest
from importlib import metadata
from math import isfinite
from pathlib import Path
from time import perf_counter
import ctypes
import json
import sys

MINILM = "Xenova/ms-marco-MiniLM-L-6-v2"
JINA = "jinaai/jina-reranker-v1-turbo-en"
DEFAULT_MODELS = (MINILM, JINA)
CPU_PROVIDER = "CPUExecutionProvider"


@dataclass(frozen=True)
class PairScore:
    score: float
    input_tokens: int
    original_tokens: int
    truncated: bool
    inference_seconds: float


def peak_rss_bytes() -> int | None:
    """Peak process working set on Windows, or peak RSS on POSIX."""
    if sys.platform == "win32":
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        stats = ProcessMemoryCounters()
        stats.cb = ctypes.sizeof(stats)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        get_memory = kernel.K32GetProcessMemoryInfo
        get_memory.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessMemoryCounters), wintypes.DWORD]
        get_memory.restype = wintypes.BOOL
        if get_memory(kernel.GetCurrentProcess(), ctypes.byref(stats), stats.cb):
            return stats.PeakWorkingSetSize
        return None

    import resource

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(rss if sys.platform == "darwin" else rss * 1024)


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def _model_snapshot(model_id: str, cache: Path) -> tuple[dict, Path, Path]:
    if model_id not in DEFAULT_MODELS:
        raise ValueError(f"Unsupported reranker: {model_id}")
    receipt = cache / (model_id.split("/")[-1] + ".json")
    row = json.loads(receipt.read_text(encoding="utf-8"))
    if row.get("model_id") != model_id or row.get("status") != "downloaded":
        raise RuntimeError(f"Model is not available locally: {model_id}: {row.get('error')}")
    snapshot = (cache / row["snapshot"]).resolve()
    artifact = (snapshot / row["artifact"]).resolve()
    if not snapshot.is_relative_to(cache) or not artifact.is_relative_to(cache):
        raise ValueError(f"Snapshot or model file is outside the model cache: {model_id}")
    if artifact.stat().st_size != row["onnx_bytes"]:
        raise ValueError(f"Model file size differs from the download record: {model_id}")
    return row, snapshot, artifact


class FastEmbedBackend:
    """Score original query / title-and-passage pairs using the same template."""

    def __init__(
        self,
        model_id: str,
        model_cache: Path,
        *,
        threads: int = 2,
        context_mode: str = "unified",
        batch_size: int = 16,
        unified_limit: int = 256,
    ):
        from fastembed.common.model_description import ModelSource
        from fastembed.rerank.cross_encoder import TextCrossEncoder
        from tokenizers import Tokenizer

        if context_mode not in {"unified", "native"} or min(threads, batch_size) < 1:
            raise ValueError("Expected unified/native context and positive thread and batch counts")
        if unified_limit < 16:
            raise ValueError("Unified pair budget must be at least 16 tokens")

        started = perf_counter()
        cache = Path(model_cache).resolve()
        receipt, snapshot, artifact = _model_snapshot(model_id, cache)
        checksum = _sha256(artifact)
        if receipt["expected_model_sha256"] and checksum != receipt["expected_model_sha256"]:
            raise ValueError(f"Model SHA-256 differs from the repository metadata: {model_id}")

        # FastEmbed's built-in names select FP32. Register the repository's INT8
        # export explicitly so both local CPU experiments use the intended files.
        registered_id = f"{model_id}-int8-local"
        if all(row["model"] != registered_id for row in TextCrossEncoder.list_supported_models()):
            TextCrossEncoder.add_custom_model(
                registered_id,
                model_file=receipt["artifact"],
                sources=ModelSource(hf=model_id),
                license=receipt["license"],
            )
        self._encoder = TextCrossEncoder(
            registered_id,
            cache_dir=str(cache),
            threads=threads,
            providers=[CPU_PROVIDER],
            local_files_only=True,
            specific_model_path=str(snapshot),
        )
        adapter = self._encoder.model
        providers = adapter.model.get_providers()
        if providers != [CPU_PROVIDER]:
            raise RuntimeError(f"Unexpected runtime providers: {providers}")
        tokenizer = adapter.tokenizer
        truncation = tokenizer.truncation
        if truncation is None or not truncation["max_length"]:
            raise RuntimeError(f"FastEmbed did not load a real context limit for {model_id}")
        tokenizer_default = int(truncation["max_length"])
        model_config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
        native = tokenizer_default
        native_limit_source = "tokenizer_config.json:model_max_length"
        # The Jina ALiBi export accepts lengths beyond its bundled tokenizer's
        # conservative 512 setting. Use its advertised positions only for the
        # explicitly requested native-context arm; the unified arm keeps its cap.
        if model_id == JINA and model_config.get("position_embedding_type") == "alibi":
            native = int(model_config["max_position_embeddings"])
            native_limit_source = "config.json:max_position_embeddings (ALiBi)"
        if context_mode == "unified" and unified_limit > native:
            raise ValueError(f"Unified limit {unified_limit} exceeds native limit {native} for {model_id}")
        self.limit = native if context_mode == "native" else unified_limit
        tokenizer.enable_truncation(
            self.limit, strategy="longest_first", direction=truncation["direction"], stride=0,
        )
        self._full_tokenizer = Tokenizer.from_str(tokenizer.to_str())
        self._full_tokenizer.no_truncation()
        self._full_tokenizer.no_padding()
        self._batch_size = batch_size
        self.metadata = {
            "model_id": model_id,
            "revision": receipt["revision"],
            "model_sha256": checksum,
            "tokenizer_sha256": _sha256(snapshot / "tokenizer.json"),
            "license": receipt["license"],
            "onnx_bytes": receipt["onnx_bytes"],
            "snapshot_bytes": receipt["snapshot_bytes"],
            "model_path": str(snapshot / receipt["artifact"]),
            "source_url": receipt["source_url"],
            "download_seconds": receipt["download_seconds"],
            "artifact": receipt["artifact"],
            "provider": providers[0],
            "native_max_tokens": native,
            "native_limit_source": native_limit_source,
            "tokenizer_default_max_tokens": tokenizer_default,
            "effective_max_tokens": self.limit,
            "context_mode": context_mode,
            "truncation_strategy": "longest_first",
            "truncation_side": truncation["direction"],
            "threads": threads,
            "batch_size": batch_size,
            "load_seconds": perf_counter() - started,
            "peak_rss_after_load_bytes": peak_rss_bytes(),
            "fastembed_version": metadata.version("fastembed"),
            "onnxruntime_version": metadata.version("onnxruntime"),
        }

    def score(self, query: str, texts: Sequence[str]) -> list[PairScore]:
        if not texts:
            return []
        pairs = [(query, text) for text in texts]
        lengths = [len(encoding.ids) for encoding in self._full_tokenizer.encode_batch(pairs)]
        # Similar-length batches reduce attention work wasted on padding. Restore
        # caller order after scoring so cache and parent-document aggregation agree.
        order = sorted(range(len(pairs)), key=lambda i: min(lengths[i], self.limit))
        results = []
        for start in range(0, len(order), self._batch_size):
            indexes = order[start:start + self._batch_size]
            started = perf_counter()
            raw = list(self._encoder.rerank_pairs(
                [pairs[i] for i in indexes], batch_size=self._batch_size, parallel=None,
            ))
            seconds = (perf_counter() - started) / len(indexes)
            for i, score in zip(indexes, raw, strict=True):
                if not isfinite(score):
                    raise ValueError(f"Nonfinite reranker score at pair {i}")
                raw_length = lengths[i]
                results.append((i, PairScore(
                    score=float(score),
                    input_tokens=min(raw_length, self.limit),
                    original_tokens=raw_length,
                    truncated=raw_length > self.limit,
                    inference_seconds=seconds,
                )))
        return [score for _, score in sorted(results, key=lambda item: item[0])]
