"""Reproduce two known limitations of the Static-Link Graph Retrieval Baseline."""

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from multihop import EvidenceChain, Retriever, load_corpus

_WALLACE_QUESTION = "阿尔弗雷德·拉塞尔·华莱士出生地位于英国哪个郡？"
_COUNTY_QUESTION = "《物种起源》的作者出生的城市位于英国哪个郡？"
_CORRECT_COUNTY_CHAIN = ("origin-of-species", "charles-darwin", "shrewsbury")
_WALLACE_WRONG_CHAIN = ("alfred-russel-wallace", "charles-darwin", "shrewsbury")
_COLLEGE_WRONG_CHAIN = ("origin-of-species", "charles-darwin", "christs-college")
_WALLACE_SOURCE = "https://www.nhm.ac.uk/discover/who-was-alfred-russel-wallace.html"


def _summarize(
    chains: Sequence[EvidenceChain], known_wrong: tuple[str, ...]
) -> list[dict]:
    output = []
    for rank, chain in enumerate(chains, start=1):
        ids = tuple(doc.id for doc in chain.documents)
        output.append({
            "rank": rank,
            "document_ids": list(ids),
            "titles": [doc.title for doc in chain.documents],
            "score": chain.score,
            "known_wrong_path": ids == known_wrong,
        })
    return output


def run_graph_regressions(path: str | Path) -> dict:
    """Run the preserved example corpus with the original and ablated graph."""
    corpus = load_corpus(path)
    darwin = next((doc for doc in corpus if doc.id == "charles-darwin"), None)
    if darwin is None or "shrewsbury" not in darwin.links:
        raise ValueError("The graph regression requires the original Darwin → Shrewsbury edge")

    original = Retriever(corpus)
    wallace = _summarize(
        original.search(_WALLACE_QUESTION, max_hops=2, top_k=3), _WALLACE_WRONG_CHAIN
    )
    county = _summarize(
        original.search(_COUNTY_QUESTION, max_hops=2, top_k=4), _COLLEGE_WRONG_CHAIN
    )
    without_edge = [
        replace(doc, links=tuple(target for target in doc.links if target != "shrewsbury"))
        if doc.id == darwin.id else doc
        for doc in corpus
    ]
    ablated = _summarize(
        Retriever(without_edge).search(_COUNTY_QUESTION, max_hops=2, top_k=4),
        _COLLEGE_WRONG_CHAIN,
    )
    expected_ids = list(_CORRECT_COUNTY_CHAIN)
    complete_original = any(row["document_ids"] == expected_ids for row in county)
    complete_ablated = any(row["document_ids"] == expected_ids for row in ablated)
    return {
        "baseline": "Static-Link Graph Retrieval Baseline",
        "corpus_path": str(path),
        "wallace_entity_switch": {
            "question": _WALLACE_QUESTION,
            "correct_answer": "蒙茅斯郡（Monmouthshire，威尔士）",
            "expected_evidence_chain": [
                {
                    "fact": "华莱士出生于阿斯克（Usk）一带。",
                    "document_id": "alfred-russel-wallace",
                    "source": _WALLACE_SOURCE,
                },
                {
                    "fact": "阿斯克位于现今的蒙茅斯郡。",
                    "document_id": None,
                    "source": _WALLACE_SOURCE,
                },
            ],
            "corpus_gap": "原始示例正文保留出生地阿斯克，缺少阿斯克到蒙茅斯郡的证据和出边。",
            "actual_top_paths": wallace,
            "wrong_path_present": any(row["known_wrong_path"] for row in wallace),
            "top_path_wrong": bool(wallace and wallace[0]["known_wrong_path"]),
        },
        "missing_key_edge": {
            "question": _COUNTY_QUESTION,
            "correct_answer": "什罗普郡（Shropshire）",
            "expected_evidence_chain": expected_ids,
            "removed_edge": ["charles-darwin", "shrewsbury"],
            "actual_original_paths": county,
            "actual_ablated_paths": ablated,
            "complete_chain_present_original": complete_original,
            "complete_chain_present_ablated": complete_ablated,
            "wrong_path_present_original": any(row["known_wrong_path"] for row in county),
            "wrong_path_present_ablated": any(row["known_wrong_path"] for row in ablated),
            "defect_reproduced": complete_original and not complete_ablated,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", nargs="?", type=Path, default=Path("data/example.jsonl"))
    args = parser.parse_args()
    print(json.dumps(run_graph_regressions(args.corpus), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
