"""Deterministic evidence-gated replacement over already scored parents."""

from collections.abc import Mapping
from dataclasses import dataclass, replace as update
from math import isfinite
from numbers import Real
import re


_FIRST_REPLACEABLE = 7
_TOP_K = 10
_WORD = re.compile(r"[^\W_]+")


@dataclass(frozen=True, slots=True)
class GateConfig:
    max_replacements: int = 1
    min_quality: float = 0.55
    min_margin: float = 0.0
    require_margin: bool = True
    novelty_overlap: float = 0.88

    def __post_init__(self):
        if type(self.max_replacements) is not int or not 0 <= self.max_replacements <= 3:
            raise ValueError("Use a replacement limit in 0..3")
        if not all(_finite(value) for value in (self.min_quality, self.min_margin, self.novelty_overlap)):
            raise ValueError("Gate limits must be finite")
        if not 0 <= self.min_quality <= 1 or self.min_margin < 0 or not 0 < self.novelty_overlap <= 1:
            raise ValueError("Invalid quality, margin or novelty cutoff")
        if type(self.require_margin) is not bool:
            raise ValueError("require_margin must be Boolean")


@dataclass(frozen=True, slots=True)
class GateDecision:
    document_id: str
    status: str
    reason: str
    quality_pass: bool | None = None
    novelty_pass: bool | None = None
    margin_pass: bool | None = None
    champion_chunk_id: str | None = None
    candidate_parent_score: float | None = None
    candidate_witness_score: float | None = None
    incumbent_document_id: str | None = None
    incumbent_rank: int | None = None
    incumbent_score: float | None = None
    score_margin: float | None = None
    new_coverage_keys: tuple[str, ...] = ()
    protected_coverage_keys: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Replacement:
    position: int
    incumbent_document_id: str
    candidate_document_id: str
    candidate_parent_score: float
    candidate_witness_score: float
    incumbent_score: float
    score_margin: float
    champion_chunk_id: str
    evidence_excerpt: str
    new_coverage_keys: tuple[str, ...]
    reason: str = "verified_novel_evidence_and_positive_margin"


@dataclass(frozen=True, slots=True)
class Selection:
    ranking: list[str]
    document_ids: list[str]
    replacements: list[Replacement]
    decisions: list[GateDecision]
    fallback_reason: str | None = None


@dataclass(frozen=True, slots=True)
class _Evidence:
    chunk_id: str
    status: str
    quality: float
    score: float | None
    keys: frozenset[str]
    text: str
    source: str
    dated_tokens: frozenset[str]
    shingles: frozenset[tuple[str, str, str]]
    reason: str


@dataclass(frozen=True, slots=True)
class Competition:
    baseline: tuple[str, ...]
    scores: dict[str, float]
    candidates: tuple[str, ...]
    witnesses: dict[str, tuple[_Evidence, ...]]
    fallback_reason: str | None


def _finite(value):
    return isinstance(value, Real) and not isinstance(value, bool) and isfinite(value)


def _get(row, field, default=None):
    if isinstance(row, Mapping):
        return row.get(field, default)
    return getattr(row, field, default)


def _witness(row, document_id):
    chunk_id = _get(row, "chunk_id")
    if not isinstance(chunk_id, str) or not chunk_id or _get(row, "document_id") != document_id:
        raise ValueError("Witness must reference its candidate and have an original Chunk ID")
    status, quality, score = (_get(row, field) for field in ("status", "quality", "jina_score"))
    keys = _get(row, "coverage_keys", ()) or ()
    if status not in {"supported", "uncertain", "conflict"} or not _finite(quality) or not 0 <= quality <= 1:
        raise ValueError("Invalid evidence verification record")
    if isinstance(keys, str) or any(not isinstance(key, str) or not key for key in keys):
        raise ValueError("Coverage keys must be a sequence of nonempty strings")
    if _get(row, "within_model_visible_text") is False and status == "supported":
        status = "uncertain"
    source_meta = _get(row, "source_metadata", {}) or {}
    source = source_meta.get("source") or source_meta.get("article_source") or ""
    text = _get(row, "evidence_excerpt", "") or ""
    if not isinstance(text, str) or not isinstance(source, str):
        raise ValueError("Witness requires textual source and excerpt")
    tokens = _WORD.findall(text.casefold())
    reasons = _get(row, "reasons", ()) or ()
    reason = next(iter(reasons), "")
    return _Evidence(
        chunk_id, status, float(quality), float(score) if _finite(score) else None, frozenset(keys), text,
        " ".join(source.casefold().split()), frozenset(t for t in tokens if t.isdigit()),
        frozenset(zip(tokens, tokens[1:], tokens[2:])), reason,
    )


def prepare(baseline_order, scores, witnesses, *, challengers=None, failure_reason=None):
    """Build a competition from retrieval scores and observed evidence only."""
    baseline = tuple(baseline_order)
    if not all(isinstance(doc, str) and doc for doc in baseline) or len(set(baseline)) != len(baseline):
        raise ValueError("A valid P5-B baseline must contain unique nonempty parent IDs")
    desired = challengers if challengers is not None else scores
    try:
        proposed = tuple(desired)
        if any(not isinstance(doc, str) or not doc for doc in proposed):
            raise ValueError("Challengers must reference nonempty parent IDs")
        candidates = tuple(sorted(set(proposed) - set(baseline[:_TOP_K])))
    except (TypeError, ValueError):
        return Competition(baseline, {}, (), {}, failure_reason or "invalid_competition_inputs")
    if failure_reason or len(baseline) <= _FIRST_REPLACEABLE:
        return Competition(baseline, {}, candidates, {}, failure_reason)
    try:
        relevant = {*baseline[:_TOP_K], *candidates}
        fair_scores = {doc: float(score) for doc in relevant
                       if _finite(score := scores.get(doc))}
        verified = {doc: tuple(_witness(row, doc) for row in witnesses.get(doc, ())) for doc in relevant}
        if any(len(set(w.chunk_id for w in blocks)) != len(blocks) for blocks in verified.values()):
            raise ValueError("One evidence record is expected for each observed Chunk")
    except (AttributeError, TypeError, ValueError, OverflowError):
        return Competition(baseline, {}, candidates, {}, "invalid_competition_inputs")
    if candidates and not any(doc in fair_scores for doc in baseline[_FIRST_REPLACEABLE:_TOP_K]):
        return Competition(baseline, fair_scores, candidates, verified, "no_comparable_incumbent_scores")
    return Competition(baseline, fair_scores, candidates, verified, None)


def _coverage(witnesses):
    return frozenset(key for evidence in witnesses if evidence.status == "supported" and evidence.text
                     for key in evidence.keys)


def _near_copy(candidate, reference, threshold):
    # Exact predicate keys handle normal duplicates. Restrict lexical comparison
    # to substantial same-source excerpts with matching numeric literals.
    if (not candidate.source or candidate.source != reference.source
            or candidate.dated_tokens != reference.dated_tokens
            or len(candidate.shingles) < 8 or len(reference.shingles) < 8):
        return False
    if {(key.split(":", 1)[0]) for key in candidate.keys}.isdisjoint(
        key.split(":", 1)[0] for key in reference.keys
    ):
        return False
    return len(candidate.shingles & reference.shingles) / len(candidate.shingles | reference.shingles) >= threshold


# Historical baseline tail remains in stable order; novel candidates are
# included in the full output only when they pass all admission checks.
def _result(baseline, current, replacements, decisions, reason=None):
    selected = set(current)
    return Selection([*current, *(doc for doc in baseline if doc not in selected)], list(current),
                     replacements, decisions, reason)


def _witness_order(witness):
    return (-(witness.score if witness.score is not None else float("-inf")),
            -witness.quality, witness.chunk_id)


def _choose_candidate(document, competition, current, all_keys, covered_by_doc, available_slots, config):
    score = competition.scores.get(document)
    all_evidence = competition.witnesses[document]
    supported = [w for w in all_evidence if w.status == "supported" and w.keys and w.text]
    usable = sorted((w for w in supported if w.quality >= config.min_quality), key=_witness_order)
    summary = {"document_id": document, "candidate_parent_score": score}
    if not usable:
        blocked = next((w for w in all_evidence if w.status == "conflict"), None)
        reason = next((w.reason for w in all_evidence if w.reason), "no_verified_witness")
        if supported:
            reason = "below_quality_gate"
        elif blocked is not None:
            reason = blocked.reason or "explicit_conflict"
        return GateDecision(status="rejected", reason=reason, quality_pass=False, **summary)

    # A challenger must add an explicit need against the whole current Top 10,
    # including the document it proposes to remove. Checking only the other
    # nine would make a swap over the very same evidence appear novel.
    novelty = [(w, w.keys - all_keys) for w in usable if w.keys - all_keys]
    if not novelty:
        return GateDecision(status="rejected", reason="already_covered", quality_pass=True,
                            novelty_pass=False, champion_chunk_id=usable[0].chunk_id, **summary)
    reference = [w for doc in current for w in competition.witnesses[doc]
                 if w.status == "supported" and w.keys]
    novel = [(w, delta) for w, delta in novelty
             if not any(_near_copy(w, old, config.novelty_overlap) for old in reference)]
    if not novel:
        return GateDecision(status="rejected", reason="near_duplicate_witness", quality_pass=True,
                            novelty_pass=False, champion_chunk_id=usable[0].chunk_id, **summary)
    comparable = [(w, delta) for w, delta in novel
                  if score is not None and w.score is not None and w.score <= score + 1e-8]
    if not comparable or not available_slots:
        best = novel[0][0]
        return GateDecision(status="rejected", reason=("no_comparable_witness_score" if not comparable
                                                      else "no_comparable_incumbent_slot"),
                            quality_pass=True, novelty_pass=True, champion_chunk_id=best.chunk_id,
                            candidate_witness_score=best.score, **summary)

    comparisons = []
    for position, incumbent in available_slots:
        incumbent_score = competition.scores[incumbent]
        retained = frozenset(key for doc in current if doc != incumbent for key in covered_by_doc[doc])
        for witness, delta in comparable:
            protected = covered_by_doc[incumbent] - retained - witness.keys
            margin = witness.score - incumbent_score
            threshold = config.min_margin if config.require_margin else 0.0
            margin_pass = margin > 0 and margin >= threshold
            comparisons.append((position, incumbent, incumbent_score, witness, delta, protected, margin, margin_pass))

    def priority(row):
        pos, _, _, witness, delta, protected, margin, margin_pass = row
        return (bool(protected) or not margin_pass, not margin_pass, bool(protected), -margin, -len(delta),
                -witness.quality, -pos, witness.chunk_id)

    position, incumbent, fair, champion, delta, protected, margin, margin_pass = min(comparisons, key=priority)
    reason = ("protected_sole_evidence" if protected else
              "insufficient_replacement_margin" if not margin_pass else "eligible")
    return GateDecision(
        status="eligible" if reason == "eligible" else "rejected",
        reason=reason, quality_pass=True, novelty_pass=True, margin_pass=margin_pass,
        champion_chunk_id=champion.chunk_id, candidate_witness_score=champion.score,
        incumbent_document_id=incumbent, incumbent_rank=position + 1, incumbent_score=fair,
        score_margin=margin, new_coverage_keys=tuple(sorted(delta)),
        protected_coverage_keys=tuple(sorted(protected)), **summary,
    )


def choose(competition, config=GateConfig(), *, record_decisions=True):
    """Assess current coverage afresh after each admitted replacement."""
    baseline = competition.baseline
    current = list(baseline[:_TOP_K])
    if competition.fallback_reason:
        decisions = [
            GateDecision(doc, "unavailable", competition.fallback_reason)
            for doc in competition.candidates
        ] if record_decisions else []
        return _result(baseline, current, [], decisions, competition.fallback_reason)
    if not config.max_replacements or len(current) <= _FIRST_REPLACEABLE or not competition.candidates:
        return _result(baseline, current, [], [])
    covered_by_doc = {doc: _coverage(witnesses) for doc, witnesses in competition.witnesses.items()}
    remaining = set(competition.candidates)
    slots = {pos for pos in range(_FIRST_REPLACEABLE, len(current)) if current[pos] in competition.scores}
    replacements = []
    decisions = {}
    for _ in range(config.max_replacements):
        if not remaining or not slots:
            break
        available = tuple((pos, current[pos]) for pos in sorted(slots))
        all_keys = frozenset(key for doc in current for key in covered_by_doc[doc])
        examined = [_choose_candidate(doc, competition, current, all_keys, covered_by_doc, available, config)
                    for doc in sorted(remaining)]
        if record_decisions:
            decisions.update({row.document_id: row for row in examined})
        eligible = [row for row in examined if row.status == "eligible"]
        if not eligible:
            break
        winner = min(eligible, key=lambda row: (-row.score_margin, -len(row.new_coverage_keys),
                                                -row.incumbent_rank, row.document_id, row.champion_chunk_id))
        doc = winner.document_id
        parent_witness = next(w for w in competition.witnesses[doc] if w.chunk_id == winner.champion_chunk_id)
        champion = Replacement(
            winner.incumbent_rank, winner.incumbent_document_id, doc, winner.candidate_parent_score,
            winner.candidate_witness_score, winner.incumbent_score, winner.score_margin,
            winner.champion_chunk_id, parent_witness.text, winner.new_coverage_keys,
        )
        replacements.append(champion)
        position = winner.incumbent_rank - 1
        current[position] = doc
        remaining.remove(doc)
        slots.remove(position)
        if record_decisions:
            decisions[doc] = GateDecision(
                document_id=doc, status="replaced", reason=champion.reason, quality_pass=True, novelty_pass=True,
                margin_pass=True, champion_chunk_id=champion.champion_chunk_id,
                candidate_parent_score=champion.candidate_parent_score,
                candidate_witness_score=champion.candidate_witness_score,
                incumbent_document_id=champion.incumbent_document_id, incumbent_rank=champion.position,
                incumbent_score=champion.incumbent_score, score_margin=champion.score_margin,
                new_coverage_keys=champion.new_coverage_keys,
            )
    if record_decisions:
        for doc, decision in tuple(decisions.items()):
            if decision.status == "eligible":
                decisions[doc] = update(decision, status="rejected", reason="replacement_budget_exhausted")
    outcome = _result(baseline, current, replacements, [decisions[d] for d in sorted(decisions)])
    if (current[:_FIRST_REPLACEABLE] != list(baseline[:_FIRST_REPLACEABLE])
            or len(set(current)) != len(current) or len(replacements) > config.max_replacements):
        return _result(baseline, list(baseline[:_TOP_K]), [], [], "invalid_gate_output")
    return outcome


def replace(baseline_order, scores, witnesses, *, challengers=None, config=GateConfig(), failure_reason=None):
    competition = prepare(baseline_order, scores, witnesses, challengers=challengers, failure_reason=failure_reason)
    return choose(competition, config)


