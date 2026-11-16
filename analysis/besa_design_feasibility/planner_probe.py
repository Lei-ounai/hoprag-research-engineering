"""Question-only feasibility probe for the seventh-round design audit."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
FRONT = re.compile(r"^(Who|Which|What|Does|Do|Did|Has|Have|Was|Were|Is|Are)\b", re.I)
REPORT = re.compile(r"\b(?:article|report|covered|reported|reporting|published|coverage)\b", re.I)
RELATION = re.compile(
    r"\b(?:before or after|before|after|between|consistent|consistency|inconsistency|"
    r"disagree|agreement|change|changed|compared|contrast|same direction|different)\b",
    re.I,
)
REPORT_RELATION = re.compile(
    r"\b(?:says?|indicates?|indicated|claims?|suggests?|suggested|implies?|implied|"
    r"concerning|regarding|discussed|reported|reports?|mentions?|mentioned|alleges?|"
    r"accused|accuses?|describes?|portray(?:ed|s)?|confirms?|confirm|reports?)\b", re.I
)
COORDINATED_CLAUSE = re.compile(
    r"(?:,\s*(?:and\s+)?|\s+(?:and|but|while)\s+)"
    r"(?=(?:does|do|did|is|was|has|have|had|who|whose|reported|recognized|claimed|admitted|"
    r"pled|pleaded|faced|accused|enforces|supports|provides|defends|offers|became|instructed|"
    r"communicated|communicates|informed|indicates|alleges|claims|alleged|implies|"
    r"the\s+other|another)\b)", re.I,
)
SOURCE_CONTRAST = re.compile(
    r",?\s+(?:while|in contrast to|similar to how|and\s+(?:did|does|do)|and|before or after|before|after)\s+"
    r"(?=(?:the\s+|another\s+|[‘'“\"]|[A-Z]))", re.I,
)
OUTCOME = re.compile(r",\s+(?=(?:was|were|did|does|do|has|have|is|are)\b)", re.I)
ENTITY_CATEGORY = re.compile(
    r"^(?:who\b|(?:which|what)\s+(?:(?:\w+\s+){0,3})"
    r"(?:company|team|player|individual|person|firm|sportsbook|platform|club|entity|organization)\b)", re.I
)
OPEN_CONDITION = re.compile(r"^\s*(?:who\s+(?:is\s+the\s+\w+\s+)?|that\s+|which\s+|whose\s+)", re.I)
LOW_SIGNAL = re.compile(
    r"^(?:(?:is|was|has|have)\s+)?(?:(?:as\s+)?reported by|according to|"
    r"as described by|in their respective|who is the individual|who is the person)\b", re.I
)


def _sha(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def _rows(path: Path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _write(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def select_sample(seed: int, count: int) -> list[dict]:
    """Use annotations only to stratify selection, export solely query ID and question."""
    split = json.loads((ROOT / "analysis/reranking/splits.json").read_bytes())
    raw = json.loads((ROOT / "data/raw/multihop_rag/MultiHopRAG.json").read_bytes())
    groups = defaultdict(list)
    for query_id in split["answerable_dev"]:
        row = raw[int(query_id.removeprefix("q-")) - 1]
        labels = (row["question_type"], len(row["evidence_list"]))
        groups[labels].append({"query_id": query_id, "question": row["query"]})
    # Equal cell-sized sampling where possible, deterministic hash order within cells.
    # This deliberately covers rare strata and is not prevalence-weighted.
    order = sorted(groups)
    for group in groups.values():
        group.sort(key=lambda row: _sha(f"{seed}:{row['query_id']}"))
    selected = []
    while len(selected) < count:
        if not any(groups.values()):
            raise ValueError("Sample size exceeds development set")
        for cell in order:
            if groups[cell] and len(selected) < count:
                selected.append(groups[cell].pop(0))
    return selected


def _tidy(text: str) -> str:
    return " ".join(text.strip(" \t,.?:;").split())


def _conjuncts(text: str) -> list[str]:
    # Keep clauses with their attached source or date. Dates ending with commas
    # do not match the predicate-leading pattern.
    text = re.sub(r",?\s+or (?:do|does|did|was|were|are)\b.*$", "", text, flags=re.I)
    candidates = []
    start = 0
    for match in COORDINATED_CLAUSE.finditer(text):
        clause = _tidy(text[start:match.start()])
        if len(clause.split()) >= 5:
            candidates.append(clause)
            start = match.end()
    tail = _tidy(text[start:])
    if len(tail.split()) >= 5:
        candidates.append(tail)
    return candidates


def _source_parts(text: str) -> list[str]:
    parts = []
    start = 0
    for match in SOURCE_CONTRAST.finditer(text):
        left = _tidy(text[start:match.start()])
        right = text[match.end():]
        if len(left.split()) >= 6 and REPORT.search(left) and REPORT.search(right):
            parts.append(left)
            start = match.end()
    tail = _tidy(text[start:])
    if tail:
        parts.append(tail)
    return parts


def _entity_parts(text: str) -> list[str]:
    parts = _conjuncts(text)
    if parts:
        first = re.sub(
            r"^who\s+(?:(?:is|was)\s+(?:the|an?)\s+"
            r"(?:individual|person|figure|player)(?:\s+(?:that|who))?\s+)?",
            "", parts[0], flags=re.I,
        )
        parts[0] = first or parts[0]
    return [part for part in parts if not LOW_SIGNAL.match(part)]


def _retrieve_text(fragment: str) -> str:
    fragment = re.sub(r"^(?:Does|Did|Do|Was|Has|Have|Is|Are|Were)\s+", "", fragment, flags=re.I)
    fragment = OPEN_CONDITION.sub("", fragment)
    return _tidy(fragment)


def plan(question: str) -> tuple[list[dict], str]:
    """Generate needs without IDs, gold, labels, corpus or ranked candidates."""
    question = _tidy(question)
    if not question:
        raise ValueError("Question is required")
    constraints: list[str] = []
    followup = None
    role = "constraint"
    confidence = 0.68
    if re.match(r"^(?:Between|After)\b", question, re.I):
        anchor, *outcome = OUTCOME.split(question, maxsplit=1)
        anchor = re.sub(r"^(?:Between|After)\s+", "", anchor, flags=re.I)
        constraints = _source_parts(anchor)
        if outcome and len(constraints) >= 2:
            followup = f"Compare the retrieved accounts for the requested temporal relation: {_tidy(outcome[0])}"
            role, confidence = "source_constraint", 0.79
    elif FRONT.match(question) and not ENTITY_CATEGORY.match(question):
        constraints = _source_parts(question)
        if len(constraints) >= 2:
            last = constraints[-1]
            if RELATION.search(question) and not REPORT_RELATION.search(last):
                constraints.pop()
            if len(constraints) >= 2:
                relation = "publication order" if re.search(r"\bbefore|after\b", question, re.I) else (
                    "the requested agreement or contrast"
                )
                followup = f"Compare the retrieved claims for {relation}: {question}"
                role, confidence = "source_constraint", 0.78
        else:
            constraints = _conjuncts(question)
            if len(constraints) >= 2:
                followup = f"Evaluate the stated claims jointly: {question}"
    elif ENTITY_CATEGORY.match(question):
        constraints = _entity_parts(question)
        if len(constraints) >= 2:
            role, confidence = "entity_constraint", 0.76

    normalized = list(dict.fromkeys(
        value for part in constraints if (value := _retrieve_text(part)) and len(value.split()) >= 4
    ))
    if not 2 <= len(normalized) <= 4 or followup and len(normalized) > 3:
        return [{
            "need_id": "n1",
            "need_text": question + "?",
            "need_role": "original_question_fallback",
            "depends_on": [],
            "confidence": 0.30,
        }], "fallback_original"
    result = [
        {"need_id": f"n{index}", "need_text": value, "need_role": role, "depends_on": [],
         "confidence": confidence}
        for index, value in enumerate(normalized, start=1)
    ]
    if followup:
        result.append({
            "need_id": f"n{len(result) + 1}",
            "need_text": followup,
            "need_role": "join",
            "depends_on": [item["need_id"] for item in result],
            "confidence": 0.74,
        })
    return result, "decomposed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "plan"))
    args = parser.parse_args()
    inputs = OUT / "planner_inputs.jsonl"
    if args.stage == "prepare":
        _write(inputs, select_sample(seed=42, count=35))
        return
    records = []
    for row in _rows(inputs):
        if set(row) != {"query_id", "question"}:
            raise ValueError("The planner input schema permits only query_id and question")
        needs, disposition = plan(row["question"])
        records.append({"query_id": row["query_id"], "needs": needs, "disposition": disposition})
    _write(OUT / "planner_outputs.jsonl", records)
    print(dict(Counter(row["disposition"] for row in records)))


if __name__ == "__main__":
    main()

