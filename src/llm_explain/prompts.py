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
Rules:
- Only claims that a sentence actually states. A course or concept NAME on its own is not a claim: never infer what a course teaches from its title.
- Head and tail follow the direction given in the relation list. State each fact once; never add the reversed claim.
- Any description of a person or course (for example similar interests, similar background, same topic, beneficial) is its own "other" claim, even inside a "who ..." phrase.
- Always keep the codes: head and tail are "[C1]", "[K1]", "[L1]" and so on, never the bare name.
- Every reason after "because" (or "since", "as", "which", "providing", "who") is its own claim, one per reason. A judgement such as same topic, relevant, might interest, useful or similar is relation "other".
- The recommended_to quote contains only the words that say the course is recommended, never the reason.
Write in English. Keep entity names exactly as given.

Example
Explanation:
[C2] Blue Course is recommended to learner [L1] because [L1] learner 1 enrolled in [C1] Red Course, both courses teach [K1] graphs, and [C2] Blue Course covers a similar topic that might interest them.
Output:
[{"head": "[C2]", "relation": "recommended_to", "tail": "[L1]", "quote": "[C2] Blue Course is recommended to learner [L1]"},
 {"head": "[L1]", "relation": "enrolled_in", "tail": "[C1]", "quote": "[L1] learner 1 enrolled in [C1] Red Course"},
 {"head": "[C1]", "relation": "teaches_concept", "tail": "[K1]", "quote": "both courses teach [K1] graphs"},
 {"head": "[C2]", "relation": "teaches_concept", "tail": "[K1]", "quote": "both courses teach [K1] graphs"},
 {"head": "[C2]", "relation": "other", "tail": "similar topic", "quote": "[C2] Blue Course covers a similar topic"},
 {"head": "[C2]", "relation": "other", "tail": "[L1]", "quote": "that might interest them"}]

Now split this one.
Explanation:
"""


def extractor(explanation):
    return EXTRACT + explanation
