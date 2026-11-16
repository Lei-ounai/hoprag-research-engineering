"""Conservative, original-span evidence checks without answer or dataset labels."""

from dataclasses import dataclass
from datetime import date
from functools import lru_cache
import re

from hoprag.retrievers import _tokens


_WORD_STOP = frozenset("""
a an and are as at be been being before by can could did do does during for from had has
have how i if in into is it its may might more most of on or our reported reporting report
reports said say says shall should than that the their them then there these they this those
through to under up used using was we were what when where which while who whom whose why
will with would you your article articles according earliest later last previous after date
recent originally published publication described discuss discussing coverage covered based
following topic time on first second about whether view views own between including refer
""".split())
_MONTHS = ("january february march april may june july august september october november december").split()
_MONTH_MAP = {name: i for i, name in enumerate(_MONTHS, 1)}
_MONTH_MAP.update({name[:3]: i for name, i in tuple(_MONTH_MAP.items())})
_MONTH_MAP["sept"] = 9
_MONTH = "|".join(sorted(_MONTH_MAP, key=lambda word: (-len(word), word)))
_DATE = re.compile(
    rf"\b(?P<iso>(?:19|20)\d{{2}}-\d{{2}}-\d{{2}})(?!\d)|"
    rf"\b(?P<month>{_MONTH})\.?\s+(?P<day>\d{{1,2}})(?:st|nd|rd|th)?\s*,?\s+(?P<year>(?:19|20)\d{{2}})\b|"
    rf"\b(?P<day2>\d{{1,2}})\s+(?P<month2>{_MONTH})\.?\s+(?P<year2>(?:19|20)\d{{2}})\b|"
    r"\b(?P<year_only>(?:19|20)\d{2})\b", re.IGNORECASE,
)
_TIME_EVENT = re.compile(
    r"\b(?:born|founded|incorporated|launched|appointed|joined|acquired|filed|took place|"
    r"occurred|held|played|game|served|released|signed|retired|resigned|elected)\b", re.I,
)
_TIME_PUBLICATION = re.compile(r"\b(?:publish(?:ed|ing)?|article date[ds]?|report dated|publication date)\b", re.I)
_ROLE_WORDS = {
    "founder": r"\b(?:co[- ]?)?found(?:er[s]?|ed)\b",
    "ceo": r"\b(?:CEO|chief executive officer|chief executive)\b",
    "investor": r"\b(?:investor[s]?|invested(?: in)?|backed|financed)\b",
    "chair": r"\b(?:chair(?:man|woman|person)|chaired)\b",
    "president": r"\b(?:president|presiding)\b",
}
_ROLES = {role: re.compile(pattern, re.I) for role, pattern in _ROLE_WORDS.items()}
# A role link must connect an actor to the requested organization locally.
_ROLE_LINKS = {
    "founder": (r"(?:co[- ]?)?founded", r"(?:co[- ]?)?founder(?:s)?"),
    "ceo": (r"(?:became|was appointed|serves as)(?: the)? (?:CEO|chief executive(?: officer)?) of",
            r"(?:CEO|chief executive(?: officer)?)"),
    "investor": (r"(?:invested in|backed|financed)", r"(?:investor|backer)"),
    "chair": (r"(?:chaired|chairs|serves as chair of)", r"(?:chair(?:man|woman|person)?)"),
    "president": (r"(?:(?:became|serves as)(?: the)? president of)", r"(?:president)"),
}
_RELATION_SPECS = {
    "birth": (r"\b(?:born|birth(?:place)?)\b", r"\b(?:was |were )?born (?:in|on|at)\b"),
    "headquarters": (r"\b(?:headquarter\w*|based|locat\w*)\b", r"\b(?:headquartered|based|located) (?:in|at|near)\b"),
    "ownership": (r"\b(?:acquir\w+|own\w+|acquisition|bought|subsidiary)\b",
                  r"\b(?:acquired|bought|owns|purchased|controlled by|subsidiary of|owned by)\b"),
    "appointment": (r"\b(?:appoint\w*|st[eoa]pp\w+ down|resign\w*)\b",
                    r"\b(?:was appointed|appointed|resigned as|stepped down as|became the)\b"),
    "fraud_charge": (r"\b(?:fraud|conspiracy|criminal trial)\b",
                     r"\b(?:charged with|(?:counts?|allegations?|convict\w*) of|alleged|accused of|"
                     r"(?:in|faces?|faced|on) (?:court|trial|charges?))\b.{0,95}\b(?:fraud|conspiracy)\b|"
                     r"\b(?:fraud|conspiracy)\b.{0,55}\b(?:charges?|trial|allegations?)\b"),
    "customer_funds": (r"\b(?:customer funds|customer money|misuse|withdrawal conditions)\b",
                       r"\b(?:instruct\w*|direct\w*|use[sd]?|misus\w*|withdraw\w*|borrow\w*|transfer\w*|"
                       r"divert\w*|take|took|access\w*)\b.{0,110}\b(?:customer funds|customer money|deposits|"
                       r"client funds|alameda|customer withdrawals?)\b"),
    "financial_disclosure": (r"\b(?:financial|accounting) (?:discrepanc\w+|shortfall|problem\w*)\b",
                             r"\b(?:told|inform\w+|acknowledg\w+|admit\w+|aware|questioned)\b.{0,110}\b"
                             r"(?:shortfall|discrepanc\w+|accounting problem\w*|missing|billion)\b"),
    "antitrust": (r"\b(?:anti[- ]?competitiv\w+|antitrust|monopoly|dominance)\b",
                  r"\b(?:antitrust (?:case|lawsuit|suit|trial)|antitrust claims|"
                  r"accus\w+.{0,65}anti[- ]?competitiv\w+|"
                  r"(?:abus\w+|control\w+|domin\w+|illegal\w*).{0,65}(?:monopoly|search engine\w*|market share))\b"),
    "court_ruling": (r"\b(?:legal case|legal victory|court ruling|lawsuit|triumph|verdict)\b",
                     r"\b(?:court|judge|jury|verdict|ruled|ruling|sided with|won|lost)\b.{0,90}\b"
                     r"(?:won|ruled|sided|ruling|victory|lost|case|lawsuit|appeal)\b"),
    "betting_offer": (r"\b(?:bettors?|betting|sportsbook|bonus offers?|best bets|wagers?)\b",
                      r"\b(?:best bets|bonus (?:bets?|offers?|codes?)|free bets?|betting (?:odds|picks|offers?)|"
                      r"(?:pick|prop|wager|sportsbook)\w*.{0,30}(?:bet|odds|bonus)\w*)\b"),
    "product_upgrade": (r"\b(?:improv\w+|enhanc\w+|upgrad\w+)\b",
                        r"\b(?:improved|enhanced|better|new|larger|faster|longer|upgraded|upgrade to)\b.{0,55}\b"
                        r"(?:screen|display|battery|chip|processor|performance|refresh rate|cooling|weight)\b"),
    "price_change": (r"\b(?:price|cost|price increase|price drop)\b",
                     r"\b(?:increase\w*|drop\w*|cut\w*|ris\w+|rais\w+|went up|fell)\b.{0,55}\b"
                     r"(?:price|cost|pricing)\b|\b(?:price|pricing|cost)\b.{0,35}\b(?:raised|rose|fell|dropped|cut|increased)\b"),
}
_RELATIONS = {name: (re.compile(cue, re.I), re.compile(predicate, re.I))
              for name, (cue, predicate) in _RELATION_SPECS.items()}
# These exclusions prevent unrelated "not guilty" pleas or "not only" conjunctions
# from flipping a clause's asserted polarity.
_NEGATION = re.compile(r"\b(?:never|neither|cannot|did not|was not|were not|is not|are not|"
                       r"has not|had not|not(?! (?:only|guilty)\b))\b", re.I)
_QUESTION_FORM = re.compile(r"^(?:which|what|who|whose|where|when|how|did|do|does|has|had|is|are|"
                            r"was|were|would|can|according|compared|considering|based)\b", re.I)
_NAME = re.compile(r"(?<![\w-])(?:[A-Z][\w'-]{2,}|[A-Z]{2,})(?:[ -](?:[A-Z][\w'-]{1,}|[A-Z]{2,})){0,4}")
_SENTENCE_END = re.compile(r"\n+|(?<!\b[A-Z])(?<!\bvs)(?<!\bMr)(?<!\bDr)[.!?](?:(?=\s|$))['\"”’»]*")
_CLAUSE_START = re.compile(
    r";\s*|\s+(?:whereas|while)\s+|,\s+(?:and\s+)?(?=(?:who|whose|which|with\s+the\s+same|"
    r"and\s+is|and\s+was|report\w*|as\s+reported)\b)"
    r"|,\s+and\s+(?=(?:the\s+)?[^,;]{1,55}\b(?:report|review|article)\b)"
    r"|\s+with\s+(?=(?:their|its)\s+(?:coverage|report|review)\b)", re.I,
)
_NON_ENTITIES = frozenset(("nfl", "nba", "week", "game", "october", "november", "monday", "crunch",
                            "nfl monday night football", "week 4", "justice department"))
_ALIAS_EXTRAS = {
    "the new york times": ("nytimes", "new york times"),
    "cbssports com": ("cbs sports", "cbssports"),
    "cn bc": ("cnbc",),
    "cnbc world business news leader": ("cnbc",),
    "fox news": ("fox news",),
}


@dataclass(frozen=True, slots=True)
class DateRequirement:
    value: str
    comparison: str
    time_kind: str
    excerpt: str
    event_cue: str | None = None


@dataclass(frozen=True, slots=True)
class Clause:
    key: str
    text: str
    source_names: tuple[str, ...]
    entities: tuple[str, ...]
    keywords: tuple[str, ...]
    role_names: tuple[str, ...]
    relations: tuple[str, ...]
    dates: tuple[DateRequirement, ...]
    negated: bool
    focus: str | None
    ownership_side: str | None


@dataclass(frozen=True, slots=True)
class QuestionProfile:
    question: str
    clauses: tuple[Clause, ...]


@dataclass(frozen=True, slots=True)
class Witness:
    query_id: str
    document_id: str
    chunk_id: str
    evidence_excerpt: str
    evidence_start: int | None
    evidence_end: int | None
    matched_entities: tuple[str, ...]
    entity_roles: tuple[str, ...]
    relation_terms: tuple[str, ...]
    document_date: str | None
    source_metadata: dict
    original_question_match: float
    expansion_query_match: float
    jina_score: float | None
    quality: float
    status: str
    reasons: tuple[str, ...]
    coverage_keys: tuple[str, ...]
    model_visible_chunk_chars: int | None
    within_model_visible_text: bool | None
    features: dict


@lru_cache(maxsize=16384)
def _normal(text):
    return " ".join(_tokens(text))


def _has(haystack, phrase):
    return bool(phrase and f" {phrase} " in f" {haystack} ")


@lru_cache(maxsize=256)
def _sources(source_names):
    aliases = {}
    for name in sorted(set(source_names) | {"Reuters", "Bloomberg", "Associated Press", "BBC", "ESPN",
                                          "The Wall Street Journal", "The Washington Post", "The New York Times",
                                          "Fortune", "TechCrunch", "The Verge", "Sporting News", "The Guardian",
                                          "Polygon", "Engadget", "CNBC", "CBSSports.com", "The Age"}):
        stem = re.split(r"\s*[\|–]\s*|\s+-\s+", name)[0]
        canonical = _normal(stem).removeprefix("the ")
        options = {_normal(name), _normal(stem), canonical}
        options.update(_ALIAS_EXTRAS.get(_normal(name), ()))
        if canonical:
            aliases.update({option: canonical for option in options if len(option) >= 3})
    pattern = "|".join(re.escape(alias) for alias in sorted(aliases, key=lambda text: (-len(text), text)))
    return aliases, re.compile(rf"(?<!\w)(?:{pattern})(?!\w)") if pattern else None


def _metadata_names(value, source_aliases):
    canon = _normal(re.split(r"\s*[\|–]\s*|\s+-\s+", value)[0])
    return source_aliases.get(canon, canon.removeprefix("the "))


def _date_marks(text):
    for match in _DATE.finditer(text):
        try:
            value = match.group("iso") or match.group("year_only")
            if not value:
                month = match.group("month") or match.group("month2")
                year = match.group("year") or match.group("year2")
                day = match.group("day") or match.group("day2")
                value = date(int(year), _MONTH_MAP[month.casefold()], int(day)).isoformat()
            if len(value) != 4:
                date.fromisoformat(value)
        except ValueError:
            continue
        yield match, value


def _event_type(word):
    for name, forms in (
        ("birth", ("born",)), ("founding", ("found", "incorporat")),
        ("appointment", ("appoint", "join", "elect")), ("departure", ("retir", "resign")),
        ("launch", ("launch", "release")), ("acquisition", ("acquir",)),
        ("filing", ("file",)), ("signing", ("sign",)), ("game", ("gam", "play")),
    ):
        if word.startswith(forms):
            return name
    return None


def _date_requirements(text):
    requirements = []
    for match, value in _date_marks(text):
        before = text[max(0, match.start() - 90):match.start()]
        after = text[match.end():match.end() + 35]
        publication = list(_TIME_PUBLICATION.finditer(before))
        event = list(_TIME_EVENT.finditer(before))
        # Compare proximity and preserve distinction between an article timestamp
        # and a date inside the claim. Bare dates remain unclassified.
        if publication and (not event or publication[-1].end() > event[-1].end()):
            kind = "publication"
        elif event:
            kind = "event"
        elif re.match(r"\s*(?:publication|published)\b", after, re.I):
            kind = "publication"
        else:
            kind = "unspecified"
        relation = re.search(r"\b(before|after|by|since|until|earlier than|later than)"
                             r"\s*(?:\w+\s+){0,2}$", before, re.I)
        marker = relation.group(1).casefold() if relation else ""
        comparison = {"earlier than": "before", "later than": "after", "by": "on_or_before",
                      "since": "on_or_after", "until": "on_or_before"}.get(marker, marker or "equals")
        event_cue = _event_type(event[-1].group().casefold()) if kind == "event" else None
        requirements.append(DateRequirement(value, comparison, kind, match.group(), event_cue))
    if not requirements and re.search(r"\bpublished\s+(?:shortly|soon|immediately)?\s*(?:before|after)\b", text, re.I):
        requirements.append(DateRequirement("", "equals", "relative_publication", text))
    return tuple(requirements)


def _clause_entities(text, source_aliases):
    clean_text = _QUESTION_FORM.sub("", text.strip())
    entities = []
    for found in _NAME.finditer(clean_text):
        value = _normal(found.group()).strip()
        value = re.sub(r"\s+(?:game|week|who|which|the)$", "", value)
        value = re.sub(r"\s+s$", "", value)
        if value in _NON_ENTITIES or value in _WORD_STOP or value in _MONTH_MAP:
            continue
        if value.removeprefix("the ") in source_aliases or all(word in _WORD_STOP for word in value.split()):
            continue
        if _DATE.search(found.group()):
            continue
        entities.append(value)
    return tuple(dict.fromkeys(entities))


def _primary_focus(entities, role_names, text):
    if not entities:
        return None
    if role_names:
        explicit = re.search(r"\b(?:founder|CEO|chief executive|investor|president|chair)(?: of| in| at)\s+(.*)", text, re.I)
        if explicit:
            context = _normal(explicit.group(1))
            for target in entities:
                if _has(context, target):
                    return target
    return max(entities, key=lambda ent: (len(ent.split()), len(ent)))


def _ownership_side(clause, focus):
    if not focus:
        return None
    normalized = _normal(clause)
    entity = re.escape(focus)
    active = r"(?:acquired|bought|owns|owned|purchased|controlled)"
    if re.search(rf"\b{active} (?:the )?{entity}(?: |$)", normalized):
        return "target"
    if re.search(rf"\b{entity} (?:was |is |been )?(?:acquired|bought|owned|controlled) by\b", normalized):
        return "target"
    if re.search(rf"\b{entity} (?:has |had )?{active}\b", normalized):
        return "actor"
    return None


def profile_question(question, source_names=()):
    """Derive source, actor, relation and time constraints from the original text."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("A nonempty original question is required")
    alias, pattern = _sources(tuple(sorted(set(source_names))))
    clean_question = question.strip()
    pieces = [piece.strip(" ,") for piece in _CLAUSE_START.split(clean_question) if piece.strip(" ,")]
    clauses = []
    inherited_entities = ()
    inherited_source = ()
    for i, text in enumerate(pieces, 1):
        normalized = _normal(text)
        source_hits = tuple(dict.fromkeys(alias[m.group()] for m in pattern.finditer(normalized))) if pattern else ()
        entity_hits = _clause_entities(text, alias)
        source_only_tail = bool(source_hits and re.match(r"as reported|published|according to", text, re.I))
        if source_only_tail and clauses:
            previous = clauses.pop()
            combined = f"{previous.text}, {text}"
            text = combined
            normalized = _normal(text)
            entity_hits = previous.entities
            source_hits = tuple(dict.fromkeys([*previous.source_names, *source_hits]))
            i = int(previous.key.rsplit("-", 1)[1])
        if not entity_hits:
            entity_hits = inherited_entities
        if not source_hits and re.match(r"(?:their|its)\s+(?:coverage|report|review)\b|with the same\b", text, re.I):
            source_hits = inherited_source
        relations = tuple(name for name, (cue, _) in _RELATIONS.items() if cue.search(text))
        roles = tuple(role for role, cue in _ROLES.items() if cue.search(text))
        dates = _date_requirements(text)
        keywords = tuple(dict.fromkeys(word for word in _tokens(text) if word not in _WORD_STOP
                                       and word not in _MONTH_MAP and not word.isdigit() and len(word) >= 3))
        focus = _primary_focus(entity_hits, roles, text)
        clauses.append(Clause(f"qclause-{i:02d}", text, source_hits, entity_hits, keywords, roles,
                              relations, dates, bool(_NEGATION.search(text)), focus, _ownership_side(text, focus)))
        if entity_hits:
            inherited_entities = entity_hits
        if source_hits:
            inherited_source = source_hits
    return QuestionProfile(question, tuple(clauses))


def _sentences(text):
    start = 0
    for delim in _SENTENCE_END.finditer(text):
        stop = delim.end()
        span = text[start:stop]
        if span.strip():
            trim = len(span) - len(span.lstrip())
            excerpt = span.strip()
            yield start + trim, start + trim + len(excerpt), excerpt
        start = stop
    if text[start:].strip():
        excerpt = text[start:].strip()
        offset = start + len(text[start:]) - len(text[start:].lstrip())
        yield offset, offset + len(excerpt), excerpt


def _date_comparison(observed, required):
    if len(observed) < len(required.value):
        return False
    actual = observed[:len(required.value)]
    target = required.value
    return {"equals": actual == target, "before": actual < target, "after": actual > target,
            "on_or_before": actual <= target, "on_or_after": actual >= target}[required.comparison]


def _time_status(requirements, sentence, document_date):
    checks = []
    for requirement in requirements:
        if requirement.time_kind == "publication":
            if not document_date:
                checks.append("date_unverified")
            elif len(document_date) < len(requirement.value):
                checks.append("date_unverified")
            else:
                checks.append("matched" if _date_comparison(document_date, requirement) else "date_conflict")
        elif requirement.time_kind == "event":
            dates = []
            for mark, observed in _date_marks(sentence):
                start = mark.start()
                neighborhood = sentence[max(0, start - 65):start]
                events = list(_TIME_EVENT.finditer(neighborhood))
                publications = list(_TIME_PUBLICATION.finditer(neighborhood))
                if not events or (publications and publications[-1].start() > events[-1].start()):
                    continue
                event_type = _event_type(events[-1].group().casefold())
                if requirement.event_cue is None or requirement.event_cue == event_type:
                    dates.append(observed)
            if any(_date_comparison(observed, requirement) for observed in dates):
                checks.append("matched")
            elif len(dates) == 1 and len(dates[0]) >= len(requirement.value):
                checks.append("date_conflict")
            else:
                checks.append("date_unverified")
        else:
            checks.append("date_unverified")
    return tuple(checks)


def _named_actor(text, focus):
    names = (_normal(m.group()) for m in _NAME.finditer(text))
    return any(value not in _WORD_STOP and value not in _MONTH_MAP and not _has(focus, value) for value in names)


def _begins_with_entity(normalized, entity):
    return bool(entity and re.match(rf"(?:the )?{re.escape(entity)}(?: |$)", normalized))


def _ends_with_entity(normalized, entity):
    return bool(entity and re.search(
        rf"(?:^| ){re.escape(entity)}(?: (?:is|was|were|are|has|had|have|will|may|be|being|reportedly|allegedly))*$",
        normalized,
    ))


def _role_present(role, sentence, focus):
    if not focus:
        return False
    action, position = _ROLE_LINKS[role]
    for m in re.finditer(action, sentence, re.I):
        lhs, rhs = sentence[max(0, m.start() - 55):m.start()], sentence[m.end():m.end() + 65]
        if _begins_with_entity(_normal(rhs), focus) and _named_actor(lhs, focus):
            return True
        if role == "founder":
            actor = re.match(r"\s+by\s+([^,;]+)", rhs, re.I)
            if actor and _ends_with_entity(_normal(lhs), focus) and _named_actor(actor.group(1), focus):
                return True
    for m in re.finditer(position, sentence, re.I):
        lhs, rhs = sentence[max(0, m.start() - 55):m.start()], sentence[m.end():m.end() + 65]
        link = re.match(r"\s+(?:of|at|in|for)\s+([^,;]+)", rhs, re.I)
        if link and _begins_with_entity(_normal(link.group(1)), focus) and _named_actor(lhs, focus):
            return True
        if re.search(rf"\b{re.escape(focus)}\s+s$", _normal(lhs)) and _named_actor(rhs, focus):
            return True
    return False


def _relation_present(relation, sentence, clause):
    focus = clause.focus
    for match in _RELATIONS[relation][1].finditer(sentence):
        left = re.split(r"[,;:]|\b(?:while|whereas|but)\b", sentence[:match.start()], flags=re.I)[-1]
        right = re.split(r"[,;:]|\b(?:while|whereas|but)\b", sentence[match.end():], flags=re.I)[0]
        if relation in {"birth", "headquarters"} and focus:
            if _ends_with_entity(_normal(left), focus):
                return True
        elif relation == "ownership" and focus:
            passive = match.group().casefold().endswith(" by") or match.group().casefold() == "subsidiary of"
            if re.match(r"^\s+by\b", right, re.I):
                passive = True
                right = re.sub(r"^\s+by\s+", "", right, flags=re.I)
            side = None
            if _ends_with_entity(_normal(left), focus):
                side = "target" if passive else "actor"
            if _begins_with_entity(_normal(right), focus):
                side = "actor" if passive else "target"
            if side and (clause.ownership_side is None or clause.ownership_side == side):
                return True
        elif clause.entities:
            local = _normal(left[-75:] + " " + match.group() + " " + right[:75])
            if any(_has(local, entity) for entity in clause.entities):
                return True
        elif len(set(_tokens(left[-75:] + " " + match.group() + " " + right[:75])) & set(clause.keywords)) >= 3:
            return True
    return False


def _source_match(source, wanted, aliases):
    return not wanted or _metadata_names(source, aliases) in wanted


def _evaluate_span(clause, sentence, source, document_date, aliases, query_words, expansion_words, title_normal):
    normalized = _normal(sentence)
    words = frozenset(normalized.split())
    terms = words & set(clause.keywords)
    entities = tuple(ent for ent in clause.entities if _has(normalized, ent))
    focus = clause.focus is not None and _has(normalized, clause.focus)
    original_match = len(query_words & words) / max(1, len(query_words))
    expansion_match = len(expansion_words & words) / max(1, len(expansion_words))
    detected = tuple(role for role, pattern in _ROLES.items() if pattern.search(sentence))
    roles = tuple(role for role in detected if _role_present(role, sentence, clause.focus))
    relations = tuple(name for name in clause.relations if _relation_present(name, sentence, clause))
    dates = _time_status(clause.dates, sentence, document_date)
    source_ok = _source_match(source, clause.source_names, aliases)
    binding = bool(entities) if clause.entities else len(terms) >= 3
    publication_only = bool(clause.dates and all(d.time_kind == "publication" for d in clause.dates)
                            and re.search(r"^\s*(?:which|what)(?:\s+[\w.-]+){0,5}?\s+(?:article|report)\b", clause.text, re.I))
    predicates = bool(clause.relations or clause.role_names or publication_only)
    reasons = []
    if not source_ok:
        reasons.append("source_mismatch" if source else "source_unverified")
    if not binding or (clause.role_names and not focus):
        reasons.append("no_question_link")
    if clause.role_names and not set(clause.role_names) <= set(roles):
        if focus and roles and not set(clause.role_names) & set(roles):
            reasons.append("entity_role_conflict")
        else:
            reasons.append("entity_role_unverified")
    if clause.relations and not set(clause.relations) <= set(relations):
        # Distinguish explicit different relations from independent incidental
        # words about the topic. Location versus birth is a common example.
        alternative = any(relation not in clause.relations and pair[1].search(sentence)
                          for relation, pair in _RELATIONS.items())
        reasons.append("relation_mismatch" if binding and alternative else "relation_unverified")
    if not predicates:
        reasons.append("relation_unverified")
    if "date_conflict" in dates:
        reasons.append("date_conflict")
    elif "date_unverified" in dates:
        reasons.append("date_unverified")
    if binding and (roles or relations):
        actual_negation = bool(_NEGATION.search(sentence))
        if actual_negation and not clause.negated:
            reasons.append("polarity_conflict")
        elif clause.negated and not actual_negation:
            reasons.append("polarity_unverified")
    status = "supported" if not reasons else "conflict" if set(reasons) & {
        "source_mismatch", "date_conflict", "entity_role_conflict", "relation_mismatch", "polarity_conflict"
    } else "uncertain"
    supported = status == "supported"
    quality = (0.50 + 0.15 * min(1.0, len(terms) / 5) + 0.10 * bool(entities) + 0.10 * focus
               + 0.08 * bool(clause.source_names) + 0.05 * bool(dates)
               + 0.02 * bool(_has(title_normal, clause.focus))) if supported else 0.0
    scope = ",".join(clause.source_names) or "any"
    group = "+".join((*clause.role_names, *clause.relations)) or "article_publication"
    if supported:
        keys = (f"{clause.key}:{scope}:{group}",)
    else:
        keys = ()
    # Prefer actual support, then diagnostics for the most closely aligned clause,
    # so conflicts are attributed to a relevant span instead of unrelated text.
    alignment = (0.22 * source_ok + 0.20 * bool(entities) + 0.14 * focus + 0.15 * bool(roles or relations)
                 + 0.18 * min(1.0, len(terms) / 5) + 0.11 * (not clause.dates or "matched" in dates))
    features = {"clause_id": clause.key, "requested_sources": clause.source_names,
                "requested_roles": clause.role_names, "requested_relations": clause.relations,
                "source_status": "matched" if clause.source_names and source_ok else
                                 "unconstrained" if source_ok else "mismatch",
                "date_status": dates, "date_kinds": tuple(d.time_kind for d in clause.dates),
                "ownership_side": clause.ownership_side,
                "requested_dates": tuple((d.value, d.comparison) for d in clause.dates),
                "anchor_overlap": len(entities), "question_keyword_hits": tuple(sorted(terms))}
    return status, quality, alignment, tuple(reasons), keys, entities, roles, relations, original_match, expansion_match, features


def verify(profile, query_id, document_id, chunk_id, text, *, title="", metadata=None,
           expansion_queries=(), jina_score=None, model_visible_chunk_chars=None):
    """Find one original, auditable passage. Unknown or conflicting support is rejected."""
    if model_visible_chunk_chars is not None and (type(model_visible_chunk_chars) is not int
                                                  or model_visible_chunk_chars < 0):
        raise ValueError("Model visibility must be a nonnegative exclusive character offset")
    visible = text if model_visible_chunk_chars is None else text[:model_visible_chunk_chars]
    metadata = metadata or {}
    source = str(metadata.get("source") or metadata.get("article_source") or "")
    date_text = metadata.get("published_at") or metadata.get("document_date") or metadata.get("date")
    document_date = str(date_text) if date_text else None
    parsed_date = next((value for _, value in _date_marks(document_date or "")), None)
    normalized_title = _normal(title)
    names = tuple(sorted({source, *(name for clause in profile.clauses for name in clause.source_names)} - {""}))
    aliases, _ = _sources(names)
    query_words = frozenset(_tokens(profile.question)) - _WORD_STOP
    expansion_words = frozenset(word for query in expansion_queries for word in _tokens(query)) - _WORD_STOP - query_words
    options = []
    for start, end, excerpt in _sentences(visible):
        if (len(visible) < len(text) and not visible[end:].strip()
                and "\n" not in text[end:len(visible) + 1] and not re.search(r"[.!?]['\"”’»]*$", excerpt)):
            continue
        for clause in profile.clauses:
            values = _evaluate_span(clause, excerpt, source, parsed_date, aliases, query_words,
                                    expansion_words, normalized_title)
            status, quality, alignment, reasons, keys, *_ = values
            options.append(((status == "supported", quality, alignment, status == "conflict", -start),
                            start, end, excerpt, values))

    if options:
        _, start, end, excerpt, chosen = max(options, key=lambda row: row[0])
        status, quality, _, reasons, keys, entities, roles, relations, original_match, expansion_match, features = chosen
    else:
        status, quality, reasons, keys, entities, roles, relations = "uncertain", 0.0, ("no_evidence_span",), (), (), (), ()
        start, end, excerpt, original_match, expansion_match, features = None, None, "", 0.0, 0.0, {}
    if jina_score is not None and model_visible_chunk_chars is None and status == "supported":
        status, quality, keys = "uncertain", 0.0, ()
        reasons = (*reasons, "model_visible_scope_unknown")
    within = (end <= min(len(text), model_visible_chunk_chars) if end is not None else False
              ) if model_visible_chunk_chars is not None else None
    return Witness(query_id, document_id, chunk_id, excerpt, start, end, entities, roles, relations,
                   document_date, {key: metadata[key] for key in ("source", "article_source", "url", "published_at",
                                                                 "document_date", "date") if key in metadata},
                   original_match, expansion_match, jina_score, quality, status, reasons, keys,
                   model_visible_chunk_chars, within, features)

