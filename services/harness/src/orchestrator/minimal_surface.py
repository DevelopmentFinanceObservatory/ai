"""The minimal orchestrator tool texts, cut to a fixed rule (`minimal` tool descriptions).

Each tool description is one sentence of at most 30 tokens, counted as
characters over four, and each parameter description at most 12 tokens, or
about 20 where it states a rule the answer depends on (`MINIMAL_RULE_PATHS`).
`search` adds a second short sentence on `share_of` within the same 30.
`MINIMAL_FULL_PATHS` keeps the full text of make_charts' chart type, title and
select, as the prompt variant words them. Names, types, enums, required fields
and structure stay as `orchestrator.declarations.tool_declaration` builds them.

`MINIMAL_PARAM_DESCRIPTIONS` keys each tool's parameters by dotted path, where
a nested field is reached through the list or model its parent holds
(`parts.entities.role`). Where the schema describes one path twice, as a
property and as the docstring of the model it holds, the one text replaces
both.
"""

from __future__ import annotations

MINIMAL_TOOL_DESCRIPTIONS: dict[str, str] = {
    "search": (
        "Search DFO's data per measure asked; returns spec_ids or candidates. "
        "With share_of, variable names the numerator alone."
    ),
    "fetch": "Retrieve the data behind a spec_id a search returned, as frames to chart.",
    "make_charts": (
        "Draw charts on the user's canvas from frames a fetch or plan returned, "
        "with the figures to state."
    ),
    "ask_user": (
        "Ask the user one to three questions with likely answers, and end your turn after calling it."
    ),
    "plan": (
        "Hand a question whose answer needs a calculation to the planner; "
        "the user approves the plan before it runs."
    ),
    "get_skill": "Load a skill a search result named, or one of its files, a page at a time.",
    "explain_dfo": "Describe what DFO is and what this assistant can do, with DFO's catalogue counts.",
    "recall": (
        "Read back in full an earlier turn's plan records or search readings, "
        "or the conversation's canvas cards."
    ),
    "remove_cards": (
        "Take cards off the user's canvas; pinned cards stay and come back as kept_pinned."
    ),
}

# The parameter texts that state a rule the answer depends on, allowed about
# 20 tokens (80 characters) where the rest hold 12.
MINIMAL_RULE_PATHS: dict[str, frozenset[str]] = {
    "search": frozenset(
        {
            "parts.entities.role",
            "parts.time",
            "parts.open_roles",
            "parts.operations.phrase",
            "parts.operations.over",
        }
    ),
    "plan": frozenset({"asks", "asks.by", "asks.over", "asks.across_years"}),
}

MINIMAL_PARAM_DESCRIPTIONS: dict[str, dict[str, str]] = {
    "search": {
        "parts": "One measured variable the question asks for.",
        "parts.variable": "What is measured, in the user's words.",
        "parts.entities": "Places or parties, each with its role as spoken.",
        "parts.entities.text": "The entity as spoken, e.g. 'France'.",
        "parts.entities.role": "Flow side: 'from' donor, 'to' recipient; 'creditor', 'debtor' for debt.",
        "parts.entities.members": "True to ask for each member of a set of places.",
        "parts.entities.name": "Full name when text is ambiguous; not with dcid.",
        "parts.entities.dcid": "Candidate dcid from a refusal; not with name.",
        "parts.time": "'YYYY', 'YYYY to YYYY', 'since YYYY', 'between YYYY and the latest' or 'latest'.",
        "parts.unit": "Unit or price basis the user stated, if any.",
        "parts.open_roles": "Side to list by member, e.g. ['creditor']; none for a total or a named side.",
        "parts.operations": "What the question computes over this part.",
        "parts.operations.verb": "What the question asks to compute.",
        "parts.operations.phrase": "User's wording, e.g. 'top 5'. With share_of, variable names the numerator alone.",
        "parts.operations.over": "origin: donors, creditors; destination: recipients, debtors; else no flow side.",
        "parts.operations.target_year": "The year a forecast projects to.",
        "parts.sources": "Publishers or datasets the user named.",
        "parts.sources.text": "The source as the user said it, e.g. 'IDS'.",
        "parts.sources.ref": "Chosen candidate's ref from source_ambiguous.",
        "parts.sources.exclude": "True when the user asks to leave it out.",
        "parts.source_code": "Indicator code, e.g. 'DT.TDS.DEGG.CD'.",
        "parts.separate": "True on each part of a comparison.",
    },
    "fetch": {
        "scope": "Replaces searched entities, window or facets.",
        "scope.entities": 'Role -> names, e.g. {"recipient": ["Togo"]}.',
        "scope.window": 'e.g. {"kind": "range", "start_year": 2015}.',
        "scope.facets": 'Axis -> {"ref": "..."}, e.g. provenance.',
        "breakdown": "One code axis to total by, at one level.",
        "breakdown.code_system": 'Code system, e.g. "crsPurpose".',
        "breakdown.level": "Hierarchy depth; 0 is the top level.",
    },
    "make_charts": {
        "charts": "One chart from a fetched or planned frame.",
        "charts.frame": "Id of a frame returned this turn.",
        "charts.latest_periods": "Show the N latest periods, e.g. 5.",
        "charts.order_by_value": "True to draw largest first.",
        "charts.state_values_for": "Series the user asked to be told a value of.",
        "charts.state_span": "True to return first and last values.",
        "charts.state_period": "One earlier period to return values at.",
    },
    "ask_user": {
        "questions": "One question with the answers it offers.",
        "questions.question": "The question as the user reads it.",
        "questions.options": "Likely answers; empty for a unit question.",
        "questions.options.label": "The answer as the user reads it.",
        "questions.options.spec_id": "The candidate's spec_id, for a candidate answer.",
        "questions.unit_of": "Spec_id of the part whose unit is asked.",
        "questions.prices": "Prices a skill prefers, on a unit question.",
        "questions.multi_select": "True when several answers can hold together.",
    },
    "plan": {
        "asks": "One output the user asked for; every open side search goes in its by or over.",
        "asks.text": "The output in the user's terms, one line.",
        "asks.by": "Parties listed one by one, e.g. ['creditor']; [] for a total; not also in over.",
        "asks.over": "Parties totalled across, e.g. ['donor']; a share's whole totals its open side.",
        "asks.across_years": "True for one figure across years the user named; false per year.",
    },
    "get_skill": {
        "skill_id": "Id of a skill a search result named.",
        "path": "A file path as the skill lists it.",
        "offset": "Page start: 0 or a closing line's offset.",
        "digest": "The digest a closing line names.",
    },
}

# The parameter paths whose full text the minimal surface keeps, as the
# variant words it (`orchestrator.declarations`): chart types, the no-period
# title rule and the select's binding rules and series caps.
MINIMAL_FULL_PATHS: dict[str, frozenset[str]] = {
    "make_charts": frozenset({"charts.type", "charts.title", "charts.select_json"}),
}

__all__ = [
    "MINIMAL_FULL_PATHS",
    "MINIMAL_PARAM_DESCRIPTIONS",
    "MINIMAL_RULE_PATHS",
    "MINIMAL_TOOL_DESCRIPTIONS",
]
