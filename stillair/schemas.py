"""Sentence templates for the synthetic knowledge world.

Facts and questions are generated from these templates, and the simulated LLM
(`llm.SimLLM`) parses them back with regexes built from the *same* templates.
Entity names are made-up words, so no real LLM could have seen them in training:
anything it knows about them must come from memory.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Schema:
    rel: str  # relation linking the subject A to the bridge entity M
    f1: str  # "A <rel> M"
    f2: str  # "M has property P"
    q_chain: str  # two-hop question: starts at A, needs M, asks for P
    q_direct: str  # one-hop control question: asks for P of M directly
    props: tuple


SCHEMAS = {
    "built_from": Schema(
        "built_from",
        "The {A} bridge is built from {M} stone.",
        "{M} stone {P}.",
        "What is notable about the stone that the {A} bridge is built from?",
        "What is notable about {M} stone?",
        (
            "conducts heat extremely well",
            "glows faintly in the dark",
            "turns brittle in frost",
            "hums when struck",
            "repels insects",
            "dissolves in salt water",
            "stays warm all winter",
            "absorbs every sound",
            "floats on water",
            "resists fire",
            "changes colour at dusk",
            "grows moss within a week",
        ),
    ),
    "lives_in": Schema(
        "lives_in",
        "The {A} fox lives in the {M} basin.",
        "The {M} basin {P}.",
        "What is notable about the basin where the {A} fox lives?",
        "What is notable about the {M} basin?",
        (
            "is covered in ice for nine months a year",
            "floods every spring",
            "has no trees at all",
            "is warmed by hot springs",
            "receives rain only at night",
            "sits below sea level",
            "is wrapped in permanent fog",
            "is ringed by salt flats",
            "echoes every footstep",
            "is lit by aurora each winter",
            "hosts a yearly lantern festival",
            "is hidden from the north wind",
        ),
    ),
    "crafted_by": Schema(
        "crafted_by",
        "The {A} lantern was crafted by the {M} guild.",
        "The {M} guild {P}.",
        "What is notable about the guild that crafted the {A} lantern?",
        "What is notable about the {M} guild?",
        (
            "never sells to outsiders",
            "works only at midnight",
            "trains apprentices for twenty years",
            "signs every piece with a thumbprint",
            "refuses to use iron",
            "meets once a year on a mountaintop",
            "keeps its recipes in song",
            "admits only left-handed members",
            "buries a coin in every workshop",
            "has outlived three kingdoms",
            "pays its debts in honey",
            "travels with a mule caravan",
        ),
    ),
}

PROPERTY = "property"


def _to_regex(template: str) -> re.Pattern:
    """Turn '{A} is ...' into a regex with named groups."""
    out = []
    for part in re.split(r"(\{[AMP]\})", template):
        if part in ("{A}", "{M}"):
            out.append(f"(?P<{part[1]}>[A-Z][a-z]+)")
        elif part == "{P}":
            out.append(r"(?P<P>[^.?]+?)")
        else:
            out.append(re.escape(part))
    return re.compile("^" + "".join(out) + "$")


REGEX = {
    name: {
        "f1": _to_regex(s.f1),
        "f2": _to_regex(s.f2),
        "q_chain": _to_regex(s.q_chain),
        "q_direct": _to_regex(s.q_direct),
    }
    for name, s in SCHEMAS.items()
}


def parse_fact(text: str):
    """Sentence -> claim (subject, (relation,), object) or None."""
    text = text.strip()
    for name, rx in REGEX.items():
        m = rx["f1"].match(text)
        if m:
            return (m["A"], (name,), m["M"])
        m = rx["f2"].match(text)
        if m:
            return (m["M"], (PROPERTY,), m["P"])
    return None


def parse_question(text: str):
    """Question -> ('chain', A, rel) | ('direct', M, rel) | None."""
    text = text.strip()
    for name, rx in REGEX.items():
        m = rx["q_chain"].match(text)
        if m:
            return ("chain", m["A"], name)
        m = rx["q_direct"].match(text)
        if m:
            return ("direct", m["M"], name)
    return None
