"""Path facts, entity codes and the rule-based claim checker. Pure functions, no I/O."""
from __future__ import annotations

import json
import re

CODE_PREFIX = {"user": "L", "item": "C", "concept": "K", "teacher": "T", "school": "S"}
# KG relation -> (claim relation, entity type that is the HEAD of the claim)
EDGE = {"enroll": ("enrolled_in", "user"), "item_concept": ("teaches_concept", "item"),
        "item_teacher": ("taught_by", "item"), "item_school": ("offered_by", "item")}
RELATIONS = ["enrolled_in", "teaches_concept", "shares_learner_with", "taught_by", "offered_by",
             "recommended_to", "other"]
LABELS = ["supported", "entities_in_path_relation_mismatch", "unknown_entity", "unsupported_claim",
          "recommendation_framing"]
FLAGGED = ["entities_in_path_relation_mismatch", "unknown_entity", "unsupported_claim"]
# completion / temporal / causal wording that no KG path can support
WORDING = re.compile(
    r"\b(finish\w*|complet\w*|passed|master\w*|then|after\w*|before|later|previously|subsequently|"
    r"continu\w*|went on|moved on|buil[dt]s? (?:up)?on|building (?:up)?on|prerequisite\w*|"
    r"foundation (?:for|of)|prepares?|leads? to)\b", re.I)
# clause starts: after punctuation, or before a joining word
CLAUSE_BREAK = re.compile(r"[.!?;,:]+\s*|\s+(?=(?:and|but|because|since|who|which|while|where|as)\b)")
CJK = re.compile("[㐀-鿿]")
BRACKET = re.compile(r"\[([LCKTS]\d+)\]")


def facts(path, name_of):
    """path = [(relation, node_type, node_id), ...], path[0] = start user."""
    entities, codes, counts, triples = [], {}, {}, []

    def code(t, i):
        if (t, i) not in codes:
            p = CODE_PREFIX[t]
            counts[p] = counts.get(p, 0) + 1
            codes[(t, i)] = f"{p}{counts[p]}"
            entities.append({"code": codes[(t, i)], "type": t, "id": int(i), "name": name_of(t, i)})
        return codes[(t, i)]

    prev = (path[0][1], path[0][2])
    code(*prev)
    for rel, t, i in path[1:]:
        if rel == "self_loop":
            prev = (t, i)
            continue
        name, head_type = EDGE[rel]
        a, b = code(*prev), code(t, i)
        triples.append((a, name, b) if prev[0] == head_type else (b, name, a))
        prev = (t, i)
    return {"entities": entities, "triples": triples, "target": codes[(path[0][1], path[0][2])],
            "final": codes[prev]}


def shared_learner_pairs(triples):
    """Unordered pairs (L-L via a shared course, C-C via a shared learner) derivable from the path."""
    enr = [(h, t) for h, r, t in triples if r == "enrolled_in"]
    return {frozenset((a[i], b[i])) for a in enr for b in enr for i in (0, 1)
            if a[i] != b[i] and a[1 - i] == b[1 - i]}


def _code_of(value, known):
    s = str(value)
    m = BRACKET.findall(s) or re.findall(r"^\s*([LCKTS]\d+)\s*$", s)
    return m[0] if len(set(m)) == 1 and m[0] in known else None


def _strip_names(text, facts_):
    for name in sorted((e["name"] for e in facts_["entities"]), key=len, reverse=True):
        text = text.replace(name, " ")
    return text


def has_flag_wording(text, facts_):
    return bool(WORDING.search(_strip_names(text, facts_)))


def cjk_leak(text, facts_):
    return bool(CJK.search(_strip_names(text, facts_)))


def _evidence(quote, explanation, facts_):
    """Text to scan for wording: every clause the quote touches; the whole explanation if the quote is not
    literally there. So a trimmed or invented quote cannot hide a word in its own clause, and a word in
    another clause does not spill over onto this claim."""
    norm = lambda s: " ".join(_strip_names(s, facts_).lower().split())
    q = norm(str(quote)).strip(" .,;:")
    text = norm(explanation)
    if not q or q not in text:
        return text
    cuts = sorted({0, len(text), *(m.end() for m in CLAUSE_BREAK.finditer(text))})
    hits = [(m.start(), m.start() + len(q)) for m in re.finditer(re.escape(q), text)]
    return " ".join(text[x:y] for x, y in zip(cuts, cuts[1:]) if any(x < e and s < y for s, e in hits))


def label_claim(claim, facts_, explanation):
    rel = claim.get("relation")
    quote = claim.get("quote", "")
    wording = bool(WORDING.search(_evidence(quote, explanation, facts_)))
    if rel not in RELATIONS or rel == "other" or wording:
        return "unsupported_claim"
    known = {e["code"] for e in facts_["entities"]}
    h, t = _code_of(claim.get("head"), known), _code_of(claim.get("tail"), known)
    if h is None or t is None:
        return "unknown_entity"
    if rel == "recommended_to":
        ok = (h, t) == (facts_["final"], facts_["target"])
        return "recommendation_framing" if ok else "entities_in_path_relation_mismatch"
    if rel == "shares_learner_with":
        ok = frozenset((h, t)) in shared_learner_pairs(facts_["triples"])
    else:
        ok = (h, rel, t) in {tuple(x) for x in facts_["triples"]}
    return "supported" if ok else "entities_in_path_relation_mismatch"


def code_name_mismatches(text, facts_):
    """Number of [code] mentions that are unknown or not followed by that entity's display name."""
    names = {e["code"]: e["name"] for e in facts_["entities"]}
    bad = 0
    for m in BRACKET.finditer(text):
        name = names.get(m.group(1))
        rest = text[m.end():].lstrip(' "\'*')
        bad += name is None or not rest.startswith(name)
    return bad


def parse_claims(text):
    """JSON list of {head, relation, tail[, quote]}; None on any malformed output or an empty list."""
    try:
        data = json.loads(text[text.index("["): text.rindex("]") + 1])
    except ValueError:
        # the first "[" may be an entity code in leading chatter
        m = re.search(r"\[\s*\{.*\}\s*\]", text, re.S)
        try:
            data = json.loads(m.group(0)) if m else None
        except ValueError:
            return None
    if not isinstance(data, list) or not data:
        return None
    if not all(isinstance(c, dict) and {"head", "relation", "tail"} <= set(c) for c in data):
        return None
    return data
