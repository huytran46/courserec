VERB = {"enrolled_in": "enrolled in", "teaches_concept": "teaches concept", "taught_by": "is taught by",
        "offered_by": "is offered by"}


def mention(facts_, code):
    return f"[{code}] " + next(e["name"] for e in facts_["entities"] if e["code"] == code)


def fact_lines(facts_):
    return "\n".join(f"- {mention(facts_, h)} {VERB[r]} {mention(facts_, t)}" for h, r, t in facts_["triples"])


def writer(facts_):
    return (f"You explain course recommendations. In 1-2 sentences, explain why course "
            f"{mention(facts_, facts_['final'])} is recommended to learner [{facts_['target']}]. "
            "Use ONLY the facts below. Do not add any other information.\n"
            "Every time you mention an entity, write its code in square brackets followed by its name, "
            "exactly as in the facts.\n"
            "Write in English. Keep entity names exactly as given.\n\nFacts:\n" + fact_lines(facts_))


EXTRACT = """Split the explanation below into atomic claims. Report what the text says, even if it seems wrong. Do not fix, add or drop statements.
Return ONLY a JSON list. Each element: {"head": ..., "relation": ..., "tail": ..., "quote": ...}
- head and tail: the entity code in square brackets, copied from the text. If the text gives no code, copy its words.
- relation: exactly one of
  enrolled_in (head learner, tail course): only that the learner enrolled in or took the course
  teaches_concept (head course, tail concept)
  shares_learner_with (two learners who have a course in common, or two courses that have a learner in common)
  taught_by (head course, tail teacher)
  offered_by (head course, tail school)
  recommended_to (head recommended course, tail learner)
  other (any statement that is none of the above, such as interest, similarity, difficulty, benefit or a reason)
- quote: copied word for word from the explanation: the whole clause stating the claim, including any words about time, completion or reason.
Write in English. Keep entity names exactly as given.

Explanation:
"""


def extractor(explanation):
    return EXTRACT + explanation
