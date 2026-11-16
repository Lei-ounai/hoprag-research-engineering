"""Conservative question-only plans for the BESA-v0 experiment."""

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class Need:
    need_id: str
    need_text: str
    role: str
    depends_on: tuple[str, ...]
    confidence: float


@dataclass(frozen=True)
class Operation:
    kind: str
    text: str
    depends_on: tuple[str, ...]


@dataclass(frozen=True)
class Plan:
    question: str
    needs: tuple[Need, ...] = ()
    accepted: bool = False
    fallback_reason: str | None = None
    operations: tuple[Operation, ...] = ()


_ENTITY_QUESTION = re.compile(
    r"^(?:who\b|(?:which|what)\s+(?:(?:[\w'-]+\s+){0,3})"
    r"(?:company|team|player|individual|person|firm|sportsbook|platform|club|entity|"
    r"organization|country|city|place|institution|product)\b)", re.I,
)
_ARTICLE_QUESTION = re.compile(r"^(?:Do(?:es)?|Did|Is|Are|Was|Were|Has|Have)\b", re.I)
_ANAPHORA = re.compile(
    r"\b(?:he|she|they|him|her|them|his|hers|their|its|it|"
    r"the same (?:source|outlet|article|company|person)|"
    r"the former|the latter|the other|on the same day|respectively)\b", re.I,
)
_AMBIGUOUS = re.compile(
    r"\b(?:or|either|whichever|whoever|if any|in either case|which of)\b", re.I,
)
_NAMED_SOURCE = re.compile(
    r"^\W*(?:(?i:the|another)\s+)?[A-Z][^,!?;]{0,78}?"
    r"(?:\b(?:article|report)\b|\b(?:reports?|reported|discussed|published|stated|said)\b)"
)
# Reporting verbs near a clause's start carry source attribution; a distant
# occurrence is too weak to disambiguate where a publication reference begins.
_REPORT_WORD = re.compile(r"\b(?:article|report|reports|reported|discussed|published|stated|said)\b", re.I)
_TIMELINE_LEAD = re.compile(r"^(Between|After|Before)\s+", re.I)
_TIMELINE_CODA = re.compile(r",\s+(?=(?:was|were|did|does|do|has|have|is|are)\b)", re.I)
_TWO_REPORTS = re.compile(r",?\s+and\s+", re.I)
_RELATION = re.compile(
    r"\b(?:consisten\w*|inconsisten\w*|agree\w*|disagree\w*|differ\w*|chang\w*|compar\w*|same)\b",
    re.I,
)
_ARTICLE_CUT = re.compile(
    r"(?P<contrast>,?\s+(?:while|in contrast to|similar to how|but)\s+)"
    r"|(?P<joint>,?\s+and\s+)"
    r"|(?P<order>\s+(?:before\s+or\s+after|before|after)\s+)", re.I,
)
_AUX = re.compile(r"^(?:did|does|do|was|were|is|are|have|has)\s+", re.I)
_CLAUSE_PREDICATE = re.compile(
    r"^(?:(?:is|are|was|were|has|have|had|did|does|can|could|would|will|"
    r"must|should)\s+|(?:became|began|believes|claimed|claims|contributed|"
    r"criticized|debated|delivered|developed|enforces|faces|faced|founded|informs|informed|"
    r"offers|offered|owns|passes|passed|pleaded|produces|produced|provides|received|"
    r"reported|reports|settled|supports|supported|won|works|went|admitted|alleged)\b)", re.I,
)
_ENTITY_CUT = re.compile(r",\s*(?:and\s+|but\s+)?|\s+(?:and|but)\s+", re.I)
_SUBORDINATE = re.compile(r"\b(?:whose|however|while|whilst|whereas|on the other hand)\b", re.I)
_ATTRIBUTION_ONLY = re.compile(
    r"^(?:Who|Which|What)[^,]{0,38},\s*(?:according to|as reported by)\b", re.I,
)
_UNSPLIT_COORDINATION = re.compile(
    r"\b(?:and|but)\s+(?:by|according to|allegations?|potential legal|is being|facing|planning|having)\b", re.I,
)
_SCOPE = re.compile(
    r"\b(?:no|not|never|neither|without|except|excluding|only|fail(?:s|ed)? to|at least|at most|"
    r"no more than|no less than|fewer than|more than|under|over|between|before|after|"
    r"throughout|until|within|since|during|from|through)\b|\b\w+n[’\x27]t\b|\b\d[\w:.-]*\b",
    re.I,
)
_LEADING_SIGNATURE = re.compile(r"^[\w'-]+", re.UNICODE)
_STOP_AT_START = frozenset({"as", "according", "of", "on", "under", "during", "with", "in", "by", "the"})


def _fallback(question: str, reason: str) -> Plan:
    return Plan(question, fallback_reason=reason)


def _span(text: str) -> str:
    return text.strip(" \t\r\n,?")


def _source_head(text: str) -> bool:
    # Recognize a fresh report/source phrase, not a conjunction inside a title
    # or an event sequence. Require an outlet-shaped prefix or a report header.
    text = _AUX.sub("", text.strip())
    if _NAMED_SOURCE.match(text):
        return True
    return re.match(r"^(?:the|another|an)\s+(?:subsequent\s+)?(?:report|article)\b", text, re.I) is not None


def _source_clause(text: str) -> bool:
    return _source_head(text) and len(text.split()) >= 6 and _REPORT_WORD.search(text) is not None


def _timeline(text: str):
    lead = _TIMELINE_LEAD.match(text)
    anchor, *comparison = _TIMELINE_CODA.split(text[lead.end():], maxsplit=1)
    if not comparison:
        return None
    splits = [
        (_span(anchor[:match.start()]), _span(anchor[match.end():]))
        for match in _TWO_REPORTS.finditer(anchor)
        if _source_head(anchor[match.end():])
    ]
    if len(splits) != 1 or not all(_source_clause(part) for part in splits[0]):
        return None
    # A third report in the question's conclusion is a new source need. Keep the
    # original question when the two-anchor template cannot represent it.
    if _REPORT_WORD.search(comparison[0]) or not _RELATION.search(comparison[0]):
        return None
    return list(splits[0]), "compare_report_claims"


def _comparative(text: str):
    start = 0
    parts = []
    operations = set()
    for match in _ARTICLE_CUT.finditer(text):
        rest = text[match.end():]
        left = _span(text[start:match.start()])
        if not _source_head(rest) or not _source_clause(left):
            continue
        parts.append(left)
        start = match.end()
        operations.add("publication_order" if match.lastgroup == "order" else "compare_report_claims")
    if not parts or len(operations) != 1:
        return None
    parts.append(_span(text[start:]))
    if not all(_source_clause(part) for part in parts):
        return None
    if operations == {"publication_order"} and len(parts) != 2:
        return None
    # A source fragment with no question-level predicate is not independently
    # retrievable. Avoid multi-source clauses whose inner attachments are unclear.
    if any(re.search(r"\b(?:articles|reports)\b", part, re.I) and
           re.search(r"\b(?:both|one\b.+\bother|while)\b", part, re.I) for part in parts):
        return None
    return parts, operations.pop()


def _entity(text: str):
    if _SUBORDINATE.search(text):
        return None
    start = 0
    parts = []
    for match in _ENTITY_CUT.finditer(text):
        if not _CLAUSE_PREDICATE.match(text[match.end():]):
            continue
        left = _span(text[start:match.start()])
        if len(left.split()) < 6:
            return None
        parts.append(left)
        start = match.end()
    parts.append(_span(text[start:]))
    if not 2 <= len(parts) <= 4 or _ATTRIBUTION_ONLY.match(parts[0]):
        return None
    if any(_UNSPLIT_COORDINATION.search(part) for part in parts):
        return None
    if any(re.match(
        r"^(?:(?:as|was|were|is|are)\s+)?(?:reported (?:on )?by|according to|mentioned by)\b", part, re.I,
    ) for part in parts[1:]):
        return None
    # The predicate of each row refers to the same entity asked for by the
    # original question. Resolving a multi-document identity is a later step.
    return parts, "same_answer_entity"


def _scope_terms(text: str) -> list[str]:
    return [term.group().casefold() for term in _SCOPE.finditer(text)]


def _valid(question: str, parts: list[str], operation: str) -> bool:
    if not 2 <= len(parts) <= 4 or len(set(parts)) != len(parts):
        return False
    if any(
        len(part.split()) < 6 or not _LEADING_SIGNATURE.match(part.lstrip("'\"“‘"))
        or part.split()[0].casefold() in _STOP_AT_START and not _source_head(part)
        for part in parts
    ):
        return False
    # Spans are copied verbatim. Matching the next span after the previous one
    # validates order, disjointness, and exact preservation within each clause.
    start = 0
    for part in parts:
        index = question.find(part, start)
        if index < 0:
            return False
        start = index + len(part)
    if operation == "publication_order":
        original = _scope_terms(question)
        retained = [term for part in parts for term in _scope_terms(part)]
        # Remove only the question-level comparison. Event qualifiers remain.
        for index in range(len(original)):
            for relation in (("before",), ("after",), ("before", "after")):
                if (tuple(original[index:index + len(relation)]) == relation
                        and original[:index] + original[index + len(relation):] == retained):
                    return True
        return False
    source_timeline = _TIMELINE_LEAD.match(question)
    if source_timeline:
        # A temporal operation preserves its own relation separately, but all
        # dates, negations and scope constraints belong to its source spans.
        anchor, *comparison = _TIMELINE_CODA.split(question[source_timeline.end():], maxsplit=1)
        if comparison and _scope_terms(comparison[0]):
            return False
        question = anchor
    return _scope_terms(question) == [term for part in parts for term in _scope_terms(part)]


def plan(question: str) -> Plan:
    """Plan using only the question. A rejected plan leaves selection to P5-B."""
    if not isinstance(question, str):
        raise TypeError("Planning requires the original question string")
    text = question.strip()
    if not text:
        return _fallback(question, "empty_question")
    if _ANAPHORA.search(text):
        return _fallback(question, "ambiguous_reference")
    if _AMBIGUOUS.search(re.sub(r"\bbefore\s+or\s+after\b", "", text, flags=re.I)):
        return _fallback(question, "ambiguous_alternative")
    if _TIMELINE_LEAD.match(text):
        result = _timeline(text)
        role = "source_constraint"
    elif _ENTITY_QUESTION.match(text):
        result = _entity(text)
        role = "entity_constraint"
    elif _ARTICLE_QUESTION.match(text):
        result = _comparative(text)
        role = "source_constraint"
    else:
        return _fallback(question, "unsupported_question_shape")
    if result is None:
        return _fallback(question, "no_unambiguous_partition")
    clauses, operation = result
    if not _valid(text, clauses, operation):
        return _fallback(question, "constraint_validation_failed")
    needs = tuple(
        Need(f"n{index}", clause, role, (), 0.8)
        for index, clause in enumerate(clauses, start=1)
    )
    return Plan(
        question, needs, True, None,
        (Operation(operation, text, tuple(need.need_id for need in needs)),),
    )
