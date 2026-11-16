"""Download the two pinned public INT8 exports into this analysis directory."""

from concurrent.futures import ThreadPoolExecutor
from importlib import metadata
from pathlib import Path
from time import perf_counter
import json
import os
import shutil
import sys

from .backend import DEFAULT_MODELS, JINA, MINILM, _sha256

ARTIFACT = "onnx/model_int8.onnx"
REVISIONS = {
    MINILM: "a09144355adeed5f58c8ed011d209bf8ee5a1fec",
    JINA: "b8c14f4e723d9e0aab4732a7b7b93741eeeb77c2",
}


def _download(model, cache, license_name):
    from huggingface_hub import HfApi, snapshot_download

    started = perf_counter()
    receipt = cache / (model.split("/")[-1] + ".json")
    if receipt.is_file():
        row = json.loads(receipt.read_text(encoding="utf-8"))
        if row.get("status") == "downloaded" and row.get("revision") == REVISIONS[model]:
            artifact = cache / row["snapshot"] / row["artifact"]
            if artifact.is_file() and _sha256(artifact) == row["expected_model_sha256"]:
                return row
    try:
        info = HfApi().model_info(model, revision=REVISIONS[model], files_metadata=True, token=False, timeout=20)
        asset = next(item for item in info.siblings if item.rfilename == ARTIFACT)
        if info.sha != REVISIONS[model] or not asset.size or not asset.lfs:
            raise RuntimeError(f"Missing pinned public metadata for {model} {ARTIFACT}")
        path = Path(snapshot_download(
            repo_id=model,
            revision=info.sha,
            allow_patterns=[
                ARTIFACT, "config.json", "tokenizer.json", "tokenizer_config.json",
                "special_tokens_map.json", "README.md", "LICENSE", "LICENSE.*",
            ],
            cache_dir=cache / "hub",
            max_workers=2,
            token=False,
        ))
        onnx = path / ARTIFACT
        if onnx.stat().st_size != asset.size or _sha256(onnx) != asset.lfs.sha256:
            raise RuntimeError(f"Downloaded weights do not match the repository metadata for {model}")
        row = {
            "model_id": model,
            "status": "downloaded",
            "revision": info.sha,
            "artifact": ARTIFACT,
            "expected_model_sha256": asset.lfs.sha256,
            "download_seconds": perf_counter() - started,
            "onnx_bytes": asset.size,
            "snapshot_bytes": sum(p.stat().st_size for p in path.rglob("*") if p.is_file()),
            "snapshot": path.relative_to(cache).as_posix(),
            "license": license_name,
            "source_url": f"https://huggingface.co/{model}/tree/{info.sha}",
        }
    except Exception as exc:
        row = {
            "model_id": model,
            "status": "download_failed",
            "error": f"{type(exc).__name__}: {exc}",
            "download_seconds": perf_counter() - started,
        }
    receipt.write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
    return row


def main():
    base = Path(__file__).resolve().parent
    cache = base / "model_cache"
    temporary = base / "environment_tmp"
    cache.mkdir(exist_ok=True)
    temporary.mkdir(exist_ok=True)
    os.environ.update({
        "TEMP": str(temporary),
        "TMP": str(temporary),
        "TMPDIR": str(temporary),
        "HF_HOME": str(cache / "huggingface"),
        "HF_HUB_CACHE": str(cache / "hub"),
        "FASTEMBED_CACHE_PATH": str(cache / "fastembed"),
        "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "HF_HUB_DISABLE_XET": "1",
        "HF_HUB_DOWNLOAD_TIMEOUT": "45",
    })
    from fastembed.rerank.cross_encoder import TextCrossEncoder
    import onnxruntime

    versions = {name: metadata.version(name) for name in (
        "fastembed", "onnxruntime", "huggingface_hub", "tokenizers", "numpy", "rank-bm25",
    )}
    free = shutil.disk_usage(cache).free
    if free < 200_000_000:
        raise OSError("Insufficient free disk for both ONNX exports and tokenizer files")
    environment = {
        "python": sys.version,
        "versions": versions,
        "available_providers": onnxruntime.get_available_providers(),
        "disk_free_bytes_before_download": free,
        "venv": str(Path(sys.executable).parent.parent),
    }
    environment_path = base / "model_environment.json"
    if not environment_path.exists():
        environment_path.write_text(json.dumps(environment, indent=2) + "\n", encoding="utf-8")
    licenses = {row["model"]: row["license"] for row in TextCrossEncoder.list_supported_models()}
    exit_code = 0
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = pool.map(lambda name: _download(name, cache, licenses[name]), DEFAULT_MODELS)
        for row in results:
            print(json.dumps(row), flush=True)
            if row["status"] != "downloaded":
                exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
