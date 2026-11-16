from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class BenchmarkConfig:
    raw_directory: Path = PROJECT_ROOT / "data" / "raw" / "multihop_rag"
    example_corpus: Path = PROJECT_ROOT / "data" / "example.jsonl"
    output_directory: Path = PROJECT_ROOT / "results" / "baseline"
    cache_directory: Path = PROJECT_ROOT / "data" / "cache"
    seed: int = 42
    top_k: int = 10
    fusion_depth: int = 100
    rrf_k: int = 60
    ollama_url: str = "http://127.0.0.1:11434"
    embedding_model: str = "bge-m3"
    embedding_dimensions: int = 1024
    batch_size: int = 8
    max_chunk_bytes: int = 3072
    embedding_keep_alive: str = "10m"
    embedding_timeout_seconds: float = 180.0
    cpu_request_concurrency: int = 1
    stage_sizes: tuple[int | None, ...] = (20, 100, None)
