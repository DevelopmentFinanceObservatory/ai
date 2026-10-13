"""The party words a plan states, and the side of a flow each names.

An entity's `role` is one of `PARTY_ROLES`, which `propose_plan` holds the
planner to, and each word of a dataset requirement's `across` is free text
the planner writes. `flow_side` places a role phrase or a source's role column on the
giving or the receiving side of a flow, and `party_role` reads a role phrase
as the `PARTY_ROLES` word it stands for. A flow has one party on each side,
so two words that speak of one source or one output name the same party
when they are on the same side (`same_party`). Words about different
sources keep their kind of flow: "donor" and "creditor" are two parties
there. `across_party` reads the
closed vocabulary an `across` word takes, the party an open axis holds whole.
`planner.plan` checks a requirement's entities against its `across` with
them, and `compiler.evidence` matches both to a source's columns.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Literal, get_args

# The words a requirement's role phrase or a source's role column may use for
# each side of a flow, as `search` places them: its role-text lexicon
# (dfo-apis `search/engine/role_text.py`, `_ROLE_TEXT_TO_AXIS`) plus the role
# columns `origin` and `destination` it resolves a flow entity under (dfo-apis
# `search/models/vocab.py`, `EntityRoleKey`). `flow_side` matches each token of
# a word against them, as written or as its singular.
# `tests/compiler/test_evidence.py` pins this copy against that lexicon. A
# phrase and a column on opposite sides name different parties; two words on
# the same side ("debtor", "destination") name the same one.
_GIVING_ROLES = frozenset(
    {
        "origin",
        "from",
        "donor",
        "provider",
        "funder",
        "source",
        "creditor",
        "lender",
        "exporter",
        "financingprovider",
    }
)
_RECEIVING_ROLES = frozenset(
    {
        "destination",
        "to",
        "recipient",
        "beneficiary",
        "debtor",
        "borrower",
        "importer",
        "target",
        "financingrecipient",
    }
)
# The nouns for the subject of a measure, the place `observationAbout` holds.
_PLACE_NOUNS = frozenset({"country", "economy", "nation", "place"})

FlowSide = Literal["giving", "receiving"]
Party = Literal["giving", "receiving", "place"]

# The words an entity's `role` may take: one giving and one receiving word for
# each kind of flow the sources hold. Each is a word of `search`'s role-text
# lexicon, so a requirement's role reaches `search` as written.
PartyRole = Literal["donor", "recipient", "creditor", "debtor", "exporter", "importer"]
PARTY_ROLES: tuple[PartyRole, ...] = get_args(PartyRole)

# The `PARTY_ROLES` word each lexicon word stands for.
_PARTY_ROLE_OF: dict[str, PartyRole] = {
    **dict.fromkeys(
        ("donor", "origin", "from", "provider", "funder", "source", "financingprovider"),
        "donor",
    ),
    **dict.fromkeys(
        ("recipient", "destination", "to", "beneficiary", "target", "financingrecipient"),
        "recipient",
    ),
    "creditor": "creditor",
    "lender": "creditor",
    "debtor": "debtor",
    "borrower": "debtor",
    "exporter": "exporter",
    "importer": "importer",
}

# The `PARTY_ROLES` words a role column holds, by the column `search` resolves
# a flow entity under.
_PARTY_ROLES_OF_COLUMN: dict[str, tuple[PartyRole, ...]] = {
    "origin": ("donor", "creditor", "exporter"),
    "destination": ("recipient", "debtor", "importer"),
}

# The role column each lexicon word naming a side alone, and no kind of flow,
# stands for.
_COLUMN_OF_SIDE_WORD = {
    "origin": "origin",
    "from": "origin",
    "destination": "destination",
    "to": "destination",
}


def flow_side(word: str) -> FlowSide | None:
    """Which side of a flow `word` names, or `None` when the lexicon cannot place it.

    `word` is read token by token, each casefolded and taken as written or as
    its singular, so "donors", "Donor countries" and "bilateral donor" are
    each the giving side. A word whose tokens place on both sides, or on
    none, has no side.
    """
    sides = {side for token in _tokens(word) if (side := _token_side(token))}
    return sides.pop() if len(sides) == 1 else None


def same_party(first: str, second: str) -> bool:
    """Whether two party words name the same party: both on one side of a flow.

    A flow has one giving and one receiving party, so of one source or one
    output "donor", "creditor" and "origin" name one party and "donor" and
    "debtor" two. A word
    `flow_side` cannot place names the same party as no word.
    """
    side = flow_side(first)
    return side is not None and side == flow_side(second)


def party_among(word: str, words: Iterable[str]) -> bool:
    """Whether one of `words` names the party `word` names (`same_party`)."""
    return any(same_party(word, other) for other in words)


def words_for_one_party[Word: str](words: Iterable[Word]) -> tuple[Word, ...]:
    """The distinct words of `words` that name a party another of them names (`same_party`).

    ("creditor", "recipient", "debtor") gives ("recipient", "debtor"), and
    words that each name a party of their own give none.
    """
    distinct = tuple(dict.fromkeys(words))
    return tuple(
        word
        for word in distinct
        if party_among(word, (other for other in distinct if other != word))
    )


def across_party(word: str) -> Party | None:
    """The party an `across` word names whole, or `None` when it names no such party.

    Every token has to be a lexicon word on one side, as written or as its
    singular, or every token a place noun: "donors" is the giving side and
    "countries" the place. A place noun beside a party word ("donor
    countries") narrows the side to its countries, and a token outside both
    ("African country", "bilateral donor", "sector") describes a population
    no open axis holds whole. A word with tokens on both sides names no one
    party.
    """
    tokens = _tokens(word)
    if tokens and all(_is_place_noun(token) for token in tokens):
        return "place"
    sides = {side for token in tokens if (side := _token_side(token))}
    if len(sides) != 1 or not all(_token_side(token) for token in tokens):
        return None
    return sides.pop()


def place_qualified_party(word: str) -> str | None:
    """The party word of an `across` that sets a place noun beside it, or `None` for any other word.

    "donor countries" gives "donor" and "recipient country" gives
    "recipient": the party word in the singular, which states the side whole.
    """
    tokens = _tokens(word)
    parties = [singular(token) for token in tokens if _token_side(token)]
    places = [token for token in tokens if _is_place_noun(token)]
    unread = len(parties) + len(places) != len(tokens)
    if not parties or not places or unread or flow_side(word) is None:
        return None
    return " ".join(parties)


def role_party(role: str) -> Party | None:
    """The party an entity's `role` phrase names: a side of a flow, the place, or `None`."""
    return flow_side(role) or across_party(role)


def party_role(phrase: str) -> PartyRole | None:
    """The `PARTY_ROLES` word `phrase` stands for, or `None` when it stands for none or several.

    `phrase` is read token by token, each casefolded and taken as written or
    as its singular, so "Donors", "lender" and "bilateral donor" read as
    "donor", "creditor" and "donor", and "origin" as "donor". A phrase
    naming no lexicon word ("country", "guarantor") or two parties
    ("donor to recipient") stands for none.
    """
    roles = {
        role
        for token in _tokens(phrase)
        if (role := _PARTY_ROLE_OF.get(token) or _PARTY_ROLE_OF.get(singular(token)))
    }
    return roles.pop() if len(roles) == 1 else None


def open_party(word: str) -> str | None:
    """The party a search's `open_roles` word asks to hold open, or `None` when it names none or several.

    A word for one kind of flow gives the `PARTY_ROLES` word it stands for
    (`party_role`): "lenders" gives "creditor". A word naming a side alone
    ("origin", "from", "destination", "to") gives that side's role column,
    which holds the donor of one flow and the creditor of another.
    """
    role = party_role(word)
    if role is None:
        return None
    columns = {
        _COLUMN_OF_SIDE_WORD.get(form)
        for token in _tokens(word)
        for form in (token, singular(token))
        if form in _PARTY_ROLE_OF
    }
    column = columns.pop() if len(columns) == 1 else None
    return column or role


def column_party_roles(column: str) -> str:
    """A role column in the words an entity's `role` takes for it: `origin` as "donor/creditor/exporter".

    A column that holds no flow party, such as `observationAbout`, is returned as it is.
    """
    roles = _PARTY_ROLES_OF_COLUMN.get(column)
    return "/".join(roles) if roles else column


def singular(phrase: str) -> str:
    """`phrase` without an English plural ending: "countries" to "country", "donors" to "donor"."""
    if phrase.endswith("ies"):
        return phrase[:-3] + "y"
    return phrase.removesuffix("s")


def _tokens(word: str) -> list[str]:
    return re.findall(r"[^\W_]+", word.casefold())


def _token_side(token: str) -> FlowSide | None:
    """The side the lexicon places `token` on, as written or as its singular."""
    for form in (token, singular(token)):
        if form in _GIVING_ROLES:
            return "giving"
        if form in _RECEIVING_ROLES:
            return "receiving"
    return None


def _is_place_noun(token: str) -> bool:
    return token in _PLACE_NOUNS or singular(token) in _PLACE_NOUNS
