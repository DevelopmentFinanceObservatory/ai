"""The planner's tools as `planner.tool_spec.ToolSpec`s, over no agent framework.

`planner_tool_specs` builds `ask_user_question`, `propose_plan` and
`reject_plan`, and `skill_tool_spec` builds `get_skill`. Each spec's body runs
`planner.tools`' or `planner.skills`' code over the call it is given. The
description and parameters are what the model reads. `planner_specs` lists
them in the order the model is shown them, and `planner.agent` serves them to
ADK.
"""

from __future__ import annotations

import inspect
from typing import Any

from compiler.surface import ToolSurface
from harness.turn_events import PLANNER_ASK
from submission.service import PROPOSE_PLAN as PROPOSE_PLAN_PROPOSER
from translator.tool import Translate

from planner.call import PlannerCall
from planner.plan import (
    MAX_PLAN_DECISIONS,
    MAX_PLAN_REQUIREMENTS,
    AnswerMode,
    AskAnswer,
    DecisionDraft,
    Decline,
    EvidenceRequirement,
)
from planner.skills import DFO_FAQ_SKILL_ID, read_skill_page
from planner.tool_spec import REQUIRED, Parameter, ToolSpec
from planner.tools import (
    EndRun,
    ask_user,
    propose,
    propose_and_translate,
    propose_reading_sources,
    reject,
)

# The tool the planner proposes a plan through.
PROPOSE_PLAN = PROPOSE_PLAN_PROPOSER.call

# What each `propose_plan` argument means, as the model reads it on the
# argument itself. `question` and `starts_new_plan` read differently under
# `question_fixed`, the way `submission.service._resolve_intent` enforces them,
# and `_FIXED_ARGUMENTS` replaces those two.
_PROPOSE_PLAN_ARGUMENTS = {
    "question": (
        "The question this plan answers. Required on a first draft and with "
        "`starts_new_plan`, which starts at version 1 under it. A patch that "
        "states one records it; one that omits it keeps the recorded question."
    ),
    "answer_mode": (
        "What kind of answer the requirements add up to. `series` answers a "
        "question that asks to see a level or a trend: a lone dataset "
        "requirement, unread by anything else, is itself the answer, its frame "
        "carrying one series per entity for a chart to draw. `computed` answers "
        "a question that asks for a figure the series does not state on its "
        "own (a difference, growth, share, quotient, rank or total), so every "
        "dataset requirement has to feed a calculation, since one left unread "
        "would leave the plan without the figure the question asked for. "
        "Either way, `answered` may name a dataset requirement a calculation "
        "reads. "
        "Required on a first draft and with `starts_new_plan`. A patch that "
        "states one switches the plan's mode and re-validates its requirements "
        "under it; one that omits it keeps the recorded mode."
    ),
    "decisions": (
        "The decisions this call adds or replaces, keyed by `slot`: a slot the "
        "plan already states replaces that decision, and a new one adds it. A "
        f"first draft states every decision the plan makes, at most "
        f"{MAX_PLAN_DECISIONS}. Every decision is cited by at least one "
        "requirement."
    ),
    "evidence_requirements": (
        "The requirements this call adds or replaces, keyed by `id`, unique "
        "within the plan and tagged by `kind`: an id the plan already states "
        "replaces that requirement whole, so restate every field it keeps, "
        "its entities' roles included, and a new id adds one. A derived "
        "requirement names the requirement it reads in `input_id`. Each cites "
        "in `decision_ids` the decisions that fixed its fields, and none when "
        "the question or your instruction states them outright. A plan states "
        f"at most {MAX_PLAN_REQUIREMENTS} requirements."
    ),
    "removed_decisions": (
        "The `slot` of each decision this call deletes. Leaving a decision out of a patch keeps it."
    ),
    "removed_evidence_requirements": (
        "The `id` of each requirement this call deletes. Leaving a requirement "
        "out of a patch keeps it."
    ),
    "declined": (
        "Each part of the question no operation computes, one per `code` and "
        "`ask_ref`. An `unsupported` answer earns a decline, with that "
        "answer's code: one `propose_plan` answered, itself or in its `translation`, "
        "or a run-time refusal the plan is replanned from that names the code. A "
        "`requirement_unmeetable` translation whose `reason` says it earns a "
        "decline earns that code too, which the plan may make or leave out. "
        "`breakdown_not_kept` is earned when an `ask_role_dropped` problem names "
        "the ask and the source its answer pins cannot hold that party open. A "
        "first plan may also decline as `no_operation_covers` an ask no "
        "requirement kind or chain of them states."
    ),
    "answered": (
        "When the brief lists asks, the requirements that answer each one, "
        "keyed by `ask`. Every ask is answered here or declined in `declined` "
        "with its number as `ask_ref`, exactly once, or both when the plan "
        "computes part of it, unless `removed_asks` names it."
    ),
    "removed_asks": (
        "The number of each ask the user's amendment removes from the plan. Leave it out "
        "of `answered` and `declined`, and narrow or remove what answered it. Only the "
        "user's amendment removes an ask, and only a later one restores it, by answering it."
    ),
    "across_years": (
        "The `id` of each step that combines its years, an `aggregate` whose `group_by` "
        "leaves out `period` or a `rank` with a `reduction` and `per_period` false, because "
        "the user's own words ask for one figure or one ranking over them: a change to the "
        "plan the user asks for, or, when no asks are listed, the question itself. A step "
        "named once keeps that standing in later patches while its kind and `reduction` "
        "stay, and loses it when an `aggregate`'s `group_by` gains `period` or a `rank` "
        "sets `per_period`. Leave it empty for a step that answers an ask marked "
        "(across years)."
    ),
    "amends_version": (
        "The `version` this call patches, as the last `propose_plan` call "
        'answered it and as the brief\'s own "version N" line states it. '
        "Required once a plan is recorded, unless `starts_new_plan` is set: "
        "a call stating neither is refused, and `question` never stands in "
        "for the choice."
    ),
    "starts_new_plan": (
        "True begins a plan of its own, at version 1, for a different question "
        "while a plan is recorded."
    ),
}

_FIXED_ARGUMENTS = {
    "question": (
        "The question this plan answers. Required on a first draft. This "
        "turn's question is fixed, so a patch either omits it or restates it "
        "unchanged: it has to match the one already recorded."
    ),
    "starts_new_plan": (
        "True says the user's reply asks a different question from the one "
        "fixed for this turn. Refused, recording nothing: the harness ends the "
        "turn and tells the user to send that question as its own message."
    ),
}


_ASK_USER_QUESTION_DOC = inspect.cleandoc(
    """Ask the user a question a referral or a refusal leaves to them, ending your turn.

    The user's answer arrives with your next brief, after the question it
    answers, naming the candidate or option they picked, or in their own
    words if they answered outside them. The choices offered come from the
    referral's candidates, or from the `options` a refusal's action carries;
    with neither, the question says what cannot be done and asks what to
    change.

    Args:
        message: The question to show the user, naming what is
            ambiguous and, briefly, what each candidate is, in plain
            words: never a requirement id, spec_id or code.
        candidates: The referral's own candidates, each
            `{"spec_id": ..., "label": ...}` -- `label` is the
            candidate's own `sentence`, not invented text. Omit it for
            a referral that carries none.
        options: The `options` a refusal's action carries, unchanged.
            Omit it otherwise.
        recommended_spec_id: The candidate you favour, when you favour
            one -- must name one of `candidates`.
        rationale: Why you favour it, for the user to read, when
            `recommended_spec_id` is given.
    """
)

_PROPOSE_PLAN_DOC = inspect.cleandoc(
    """Record a plan, or patch the one already recorded.

    Returns `ok` with `status` `"draft"`; the new `version`, which a
    patch passes as `amends_version`; the `previous_version` it amended,
    `None` on a first draft; and the `question` as stored.
    """
)

_REJECT_PLAN_DOC = inspect.cleandoc(
    """Record that the user rejected the current plan.

    Nothing in the plan may run afterwards. Call `propose_plan` again if
    the user wants a revision rather than an outright stop.
    """
)

# Keep the description to one sentence saying what the call returns, because a
# longer one makes Gemini reason at length before some calls
# (`tests/planner/test_skills.py` bounds its length). The argument text sits on
# the parameters.
_GET_SKILL_DOC = (
    "Returns the SKILL.md of the skill with the given id and the list of its files, "
    "or one file of the skill, verbatim, when `path` is given."
)

# `= []` defaults keep the declaration's arrays flat. An optional list declared
# `list[X] | None` reaches the model as `anyOf: [array, null]`, and a patch is
# written into a flat array. Nothing mutates these defaults.
_ASK_PARAMETERS = (
    Parameter("message", str),
    Parameter("candidates", list[dict[str, str]], []),
    Parameter("options", list[dict[str, str]], []),
    Parameter("recommended_spec_id", str | None, None),
    Parameter("rationale", str | None, None),
)

_PROPOSE_PARAMETERS = (
    Parameter("question", str | None, None),
    Parameter("answer_mode", AnswerMode | None, None),
    Parameter("decisions", list[DecisionDraft], []),
    Parameter("evidence_requirements", list[EvidenceRequirement], []),
    Parameter("removed_decisions", list[str], []),
    Parameter("removed_evidence_requirements", list[str], []),
    Parameter("declined", list[Decline], []),
    Parameter("answered", list[AskAnswer], []),
    Parameter("removed_asks", list[int], []),
    Parameter("across_years", list[str], []),
    Parameter("amends_version", int | None, None),
    Parameter("starts_new_plan", bool, False),
)


def _described(
    parameters: tuple[Parameter, ...], descriptions: dict[str, str]
) -> tuple[Parameter, ...]:
    """`parameters`, each carrying its text from `descriptions`.

    Raises:
        ValueError: A parameter has no description, or a description names no
            parameter.
    """
    names = [parameter.name for parameter in parameters]
    if set(names) != set(descriptions):
        raise ValueError(
            f"parameters {sorted(set(names) - set(descriptions))} have no description and "
            f"descriptions {sorted(set(descriptions) - set(names))} name no parameter; "
            "describe each argument exactly once"
        )
    return tuple(
        Parameter(p.name, p.annotation, p.default, descriptions[p.name]) for p in parameters
    )


def _ask_user_question(call: PlannerCall, end_run: EndRun, arguments: dict[str, Any]) -> object:
    return ask_user(call, end_run, **arguments)


def _reject_plan(call: PlannerCall, end_run: EndRun, arguments: dict[str, Any]) -> object:
    del end_run, arguments
    return reject(call)


def planner_tool_specs(
    *,
    translate: Translate | None = None,
    question_fixed: bool = False,
    reads_user_words: bool = True,
    surface: ToolSurface | None = None,
) -> list[ToolSpec]:
    """The planner's tools: `ask_user_question`, `propose_plan` and `reject_plan`.

    `translate` is the translation (`translator.tool.build_translate`) a
    `propose_plan` that records a plan runs on it, answering with the verdict
    under `translation` alongside its own `ok`. Given, `propose_plan`'s body
    is a coroutine function, and a translation that compiles declares the plan
    ready (`planner.tools.ready_and_end`). Omitted, `propose_plan` records the
    plan and stops, and nothing declares a plan ready.

    `surface` is the tool surface `propose_plan` reads a pinned dataset's
    first year through, when a start that follows the newest data needs one
    no answer on record states (`read_full_history`). Given, `propose_plan`'s
    body is a coroutine function. Omitted, it reads nothing and refuses such a
    plan `first_year_unknown`.

    `question_fixed` is passed straight to `submission.service.submit` on
    every `propose_plan` call, and selects `_FIXED_ARGUMENTS`' descriptions of
    `question` and `starts_new_plan`, so those two arguments describe the mode
    the tool actually enforces.

    `reads_user_words` is passed to `submission.service.admit` the same way.
    It says whether the planner reads the user's own words in every entry, so
    `propose_plan`'s `across_years` may name a step in any of them.

    Raises:
        ValueError: `propose_plan`'s parameters and their descriptions differ
            (`_described`).
    """

    def propose_plan(call: PlannerCall, end_run: EndRun, arguments: dict[str, Any]) -> object:
        del end_run
        return propose(
            call, arguments, question_fixed=question_fixed, reads_user_words=reads_user_words
        )

    async def propose_plan_reading(
        call: PlannerCall, end_run: EndRun, arguments: dict[str, Any]
    ) -> object:
        del end_run
        return await propose_reading_sources(
            call,
            arguments,
            question_fixed=question_fixed,
            reads_user_words=reads_user_words,
            surface=surface,
        )

    async def propose_plan_translating(
        call: PlannerCall, end_run: EndRun, arguments: dict[str, Any]
    ) -> object:
        assert translate is not None
        return await propose_and_translate(
            call,
            end_run,
            arguments=arguments,
            question_fixed=question_fixed,
            reads_user_words=reads_user_words,
            translate=translate,
            surface=surface,
        )

    body = propose_plan
    if translate is not None:
        body = propose_plan_translating
    elif surface is not None:
        body = propose_plan_reading
    descriptions = (
        _PROPOSE_PLAN_ARGUMENTS | _FIXED_ARGUMENTS if question_fixed else _PROPOSE_PLAN_ARGUMENTS
    )
    return [
        ToolSpec(
            name=PLANNER_ASK,
            description=_ASK_USER_QUESTION_DOC,
            parameters=_ASK_PARAMETERS,
            body=_ask_user_question,
        ),
        ToolSpec(
            name=PROPOSE_PLAN,
            description=_PROPOSE_PLAN_DOC,
            parameters=_described(_PROPOSE_PARAMETERS, descriptions),
            body=body,
            compact=True,
        ),
        ToolSpec(
            name="reject_plan", description=_REJECT_PLAN_DOC, parameters=(), body=_reject_plan
        ),
    ]


def skill_tool_spec(surface: ToolSurface) -> ToolSpec:
    """`get_skill`, which calls data-mcp through `surface` (`planner.skills.read_skill_page`)."""

    async def get_skill(call: PlannerCall, end_run: EndRun, arguments: dict[str, Any]) -> object:
        del end_run
        read: dict[str, object] = {"skill_id": arguments["skill_id"]}
        if arguments["path"] is not None:
            read["path"] = arguments["path"]
        if arguments["offset"]:
            read["offset"] = arguments["offset"]
        if arguments["digest"] is not None:
            read["digest"] = arguments["digest"]
        return await read_skill_page(surface, read, call=call)

    return ToolSpec(
        name="get_skill",
        description=_GET_SKILL_DOC,
        parameters=(
            Parameter(
                "skill_id",
                str,
                REQUIRED,
                f"A skill id, or `{DFO_FAQ_SKILL_ID}` for DFO itself.",
            ),
            Parameter("path", str | None, None, "A file's path exactly as the list gives it."),
            Parameter(
                "offset", int, 0, "Where a page starts: 0, or the offset a closing line names."
            ),
            Parameter(
                "digest", str | None, None, "The digest a closing line names, on a continuation."
            ),
        ),
        body=get_skill,
    )


def planner_specs(
    *,
    translate: Translate | None = None,
    surface: ToolSurface | None = None,
    question_fixed: bool = False,
    reads_user_words: bool = True,
) -> list[ToolSpec]:
    """The planner's tools, in the order the model is shown them: `get_skill`
    when `surface` is given (`skill_tool_spec`), then
    `planner_tool_specs`, each argument as that function documents it."""
    return [
        *([skill_tool_spec(surface)] if surface is not None else []),
        *planner_tool_specs(
            translate=translate,
            question_fixed=question_fixed,
            reads_user_words=reads_user_words,
            surface=surface,
        ),
    ]
