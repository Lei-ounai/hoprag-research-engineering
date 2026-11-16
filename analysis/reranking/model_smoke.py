"""Opt-in real CPU model smoke. This module only reads local model snapshots."""

from dataclasses import asdict
from pathlib import Path
import json
import os

from .backend import DEFAULT_MODELS, JINA, FastEmbedBackend, peak_rss_bytes


def main():
    base = Path(__file__).resolve().parent
    cache = base / "model_cache"
    os.environ.update({
        "HF_HOME": str(cache / "huggingface"),
        "HF_HUB_CACHE": str(cache / "hub"),
        "FASTEMBED_CACHE_PATH": str(cache / "fastembed"),
        "HF_HUB_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
    })
    query = "Which planet is known as the Red Planet?"
    texts = [
        "Mars\nMars is widely known as the Red Planet because of its reddish appearance.",
        "Venus\nVenus is described as Earth's twin because of their similar sizes.",
        "Mars\n" + "Mars is commonly called the Red Planet. " * 150,
    ]
    results = []
    for model_id in DEFAULT_MODELS:
        for mode in (("unified", "native") if model_id == JINA else ("unified",)):
            row = {"model_id": model_id, "context_mode": mode}
            try:
                backend = FastEmbedBackend(model_id, cache, context_mode=mode)
                scores = backend.score(query, texts)
                tokenizer = backend._encoder.model.tokenizer
                lengths = [
                    sum(encoded.attention_mask)
                    for encoded in tokenizer.encode_batch([(query, text) for text in texts])
                ]
                if lengths != [item.input_tokens for item in scores]:
                    raise AssertionError("Token accounting differs from actual unpadded inference lengths")
                if scores[0].score <= scores[1].score:
                    raise AssertionError("Relevant score did not exceed the unrelated score")
                if scores[-1].truncated != (mode == "unified"):
                    raise AssertionError("Unexpected truncation for the long input")
                row.update(
                    status="passed", model=backend.metadata,
                    direction_scores=[asdict(item) for item in scores], peak_rss_bytes=peak_rss_bytes(),
                )
            except Exception as exc:
                row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            results.append(row)
            print(json.dumps(row), flush=True)
    (base / "model_integration.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return int(any(row["status"] != "passed" for row in results))


if __name__ == "__main__":
    raise SystemExit(main())
