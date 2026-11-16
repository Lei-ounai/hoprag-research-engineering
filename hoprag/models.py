from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Document:
    document_id: str
    title: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    evidence_text: str
    source_document_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Query:
    query_id: str
    question: str
    answer: str
    question_type: str
    evidence_list: tuple[Evidence, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_answerable(self) -> bool:
        return self.question_type != "null_query"


@dataclass(frozen=True)
class RetrievalResult:
    query_id: str
    document_id: str
    retrieval_method: str
    score: float
    rank: int
