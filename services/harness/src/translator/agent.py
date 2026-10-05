"""The translator agent: `translate`, plus `create_translator_agent`.

The translator finds the data each dataset requirement needs (see
`translator.discovery`), proposes the operations that carry the plan's
requirements out, and hands both to `compiler.compile_plan`, which writes the
workflow and checks it. None of this needs a model turn of its own --
`translator.mechanical.run_translate` drives it as code, behind the single
tool this agent exposes, `translate`. What cannot be carried out at all goes
back to the planner as a typed `Referral`: saved in session state for the
stall watchdog and `planner.agent`'s own `submit_plan` gate, and handed back
structurally as this agent's own closing turn, via the `output_schema`
`create_translator_agent` sets -- see `Compiled` and `Referred`.

The translator keeps exactly one judgement: which of several candidate
sources best answers a requirement, once `translate` cannot settle that on
its own. It stops there and answers `needs_decision`, carrying the
candidates, their own descriptions, and the decisions citing the
requirement; the model ranks them and calls `translate` again with its
`choice` and `rationale`, which ride into the referral as a recommendation
for the person who settles the choice, not a pick the translator makes.

None of the translator's work is confirmation-gated, because none retrieves
observations or computes anything: discovery reads metadata, and compiling
writes a workflow that nothing runs before the user approves it. The
approval gate stays where it is, in the harness graph.

An unreadable surface comes back as a typed problem rather than an exception:
these run inside ADK tool calls, where raising ends the turn. `DuplicateToolNameError`
is the exception, and propagates: a tool name served twice is a
misconfiguration that no retry resolves.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import suppress
from typing import Literal

from compiler.compile import CompileIssue, CompileIssueCode, compile_plan
from compiler.surface import ToolSurface, ToolSurfaceUnavailableError
from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.events._branch_path import _BranchPath
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.planners import BuiltInPlanner
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.set_model_response_tool import SetModelResponseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types
from planner.operations import Translation
from planner.plan import DatasetRequirement, Plan, plan_revision, save_plan
from planner.records import Identifier, StrictModel
from planner.settings import (
    AGENT_THINKING_CONFIG,
    drop_thought_parts,
    override_thinking_on_sampled_turns,
)
from planner.tool_results import Action, Outcome, ProblemCode, ok, problem, referred
from planner.workflow import SelectedSource
from pydantic import Field, JsonValue, TypeAdapter, ValidationError

from translator.discovery import (
    DiscoveryLog,
    apply_definitions,
    build_discovery_tools,
    compact_source,
    current_draft,
    described_sources,
    discovery_lock,
    fetch_definitions,
    load_log,
    requirement_revision,
    settled_sources,
)
from translator.mechanical import TranslatorCore, run_translate
from translator.referral import (
    MAY_CARRY_CANDIDATES,
    Referral,
    ReferralKind,
    SourceRecommendation,
    load_referral,
    save_referral,
)

AGENT_NAME = "dfo_translator"

# Mirrors `LlmAgent.mode`'s own type, which ADK declares inline rather than
# exporting (`google.adk.agents.llm_agent`).
AgentMode = Literal["chat", "task", "single_turn"]


class Compiled(StrictModel):
    """The translator's closing turn once the plan compiled: no referral pending.

    Paired with `Referred` as this agent's `output_schema`
    (`create_translator_agent`), so the closing turn `process_llm_agent_output`
    (`google.adk.workflow._llm_agent_wrapper`) parses is one of these two typed
    shapes rather than free prose. Carries only `outcome`, the one field
    `propose_operations`'s own success answer already sets to the same value:
    the richer summary -- version, sources, steps, warnings -- is what
    `submit_plan` answers from the plan's own record a moment later, so
    restating it here would only spend more of this turn to say it twice.
    """

    outcome: Literal["ok"] = "ok"


class Referred(StrictModel):
    """The translator's closing turn once it hands the plan back: the recorded referral.

    The fields are `Referral`'s, with `candidates` in the shape
    `compact_source` renders them -- `measures`, entities without a `ref`,
    a `spec_id` only where the kind lets the planner pin one -- which is what
    `return_to_planner` and `translate` answer with. `Referral.candidates`
    holds `SelectedSource`s instead, so a model retyping the candidates it was
    shown could never produce them. The model closes with the fields it can
    copy from `translate`'s answer, and `_close_on_the_recorded_referral`
    replaces them with `_referral_payload` of the referral in session state,
    candidates included.
    """

    kind: ReferralKind
    reason: str = Field(min_length=1)
    requirement_id: Identifier | None = None
    decision_id: Identifier | None = None
    candidates: tuple[dict[str, JsonValue], ...] = ()
    no_data_reason: str | None = None
    recommendation: SourceRecommendation | None = None


def _referral_payload(referral: Referral) -> dict[str, object]:
    """Return `referral` as the planner reads it: each candidate rendered by `compact_source`.

    `no_data`'s candidates are the nearest series discovery found, not a
    source the planner may pin without a fresh search: rendered without
    `spec_id` so the only way one reaches a pinnable choice is a real
    `source_ambiguous` or `pinned_source_not_found` referral.
    """
    selectable = referral.kind is not ReferralKind.NO_DATA
    return referral.model_dump(mode="json", exclude={"candidates"}) | {
        "candidates": [
            compact_source(source, selectable=selectable) for source in referral.candidates
        ]
    }


def _close_on_the_recorded_referral(
    tool: BaseTool,
    args: dict[str, object],
    tool_context: ToolContext,
    tool_response: object,
) -> None:
    """Make the referral in session state the closing turn, whatever the model typed.

    An `after_tool_callback` on `set_model_response`, the tool ADK injects to
    take a structured closing turn while the agent has tools of its own. With
    a union `output_schema` ADK declares that tool's one `response`
    parameter as free-form and validates nothing in the tool, so a shape the
    model got wrong surfaces only when the node output is checked, after
    which nothing can repair it. A recorded referral is what `translate`
    just handed back (it refuses while one is pending, so a compiled plan
    has none), which makes it the authority on what to close with. With none
    recorded, only a `Compiled` answer can stand: `Referred` carries none of
    `Referral`'s cross-field rules, so a referral-shaped answer no code
    recorded would otherwise reach the planner. That answer raises here. The
    node then fails and the planner reads ADK's `Error running sub-agent`
    result, the same outcome as any other closing turn the schema refuses. An
    error returned to the model instead would have it retry, bounded only by
    the run's own call limits.
    """
    del args, tool_response
    if not isinstance(tool, SetModelResponseTool):
        return
    referral = load_referral(tool_context.state)
    if referral is not None:
        tool_context.actions.set_model_response = _referral_payload(referral)
        return
    answer = tool_context.actions.set_model_response
    try:
        Compiled.model_validate(answer)
    except ValidationError as exc:
        raise ValueError(
            f"the translator closed with {answer!r}, but no referral is recorded "
            "and it is not a compiled outcome"
        ) from exc


# Attached as a sub-agent, this is the whole contract the planner's model reads:
# ADK builds the wrapping tool's declaration from the sub-agent's own name and
# description (`google.adk.tools.agent_tool`), never from anything the parent
# configures. So it has to say when to call this agent and what comes back.
DESCRIPTION = (
    "Finds the data each of the current draft plan's dataset requirements needs, "
    "proposes the operations that carry every requirement out, and compiles them into "
    "the workflow the user approves. Call it once the plan is proposed, with a short "
    "`request` naming what you want it to do -- `request` is the instruction this agent "
    "is handed, the one part of your conversation that reaches it, and never a copy of "
    "the plan. This agent reads the plan `propose_plan` recorded itself, from session "
    "state, so there is nothing to paste in and nothing it needs you to restate. "
    'Answers `{"outcome": "ok"}` '
    "once it records the compiled workflow, which moves the plan to PENDING_APPROVAL, or "
    "the referral itself when something cannot be carried out, naming what and why."
)

_TRANSLATION_ADAPTER = TypeAdapter(Translation)

# The one tool this agent declares. Every generation of this agent's turn,
# including the one that ranks candidates for `translate`'s own
# `needs_decision` judgement, runs at `AGENT_THINKING_CONFIG`'s LOW, or at
# `DEFAULT_THINKING_CONFIG` on a sampled turn (see `planner.settings`).
# `BuiltInPlanner` writes `AGENT_THINKING_CONFIG` into every request on its
# own, so nothing here has to.
_TRANSLATE_TOOL = "translate"

# Gemini requires a `thought_signature` on the first `functionCall` part of
# each step, and answers a request without one with a 400. The call
# `_before_translator_model` opens the turn with is injected outside a model
# turn, so it has no real signature. The documented placeholder makes the API
# skip validation for it. `PlannerSeam._before_planner_model` returns a text
# part, which the rule does not cover.
_SKIP_SIGNATURE = b"skip_thought_signature_validator"


def _last_tool_response(callback_context: CallbackContext) -> types.FunctionResponse | None:
    """This agent's own last tool response in its current run, if any.

    Walks `callback_context.session.events` backwards, skipping any event on
    a branch that this run's own branch (`callback_context.branch`) neither
    equals nor descends from: that event belongs to a sibling run. The walk
    stops at the first remaining event this agent did not author, or that
    carries an invocation id other than `callback_context.invocation_id` --
    that event is the run's own start, the planner's call into
    `dfo_translator`.

    The planner may call this agent more than once inside one invocation (a
    correction round, a second referral, or two calls in the same model
    response), each call its own run sharing that invocation's id.
    `_SingleTurnAgentTool.run_async` puts each such run on its own sibling
    branch (`_BranchPath.create_sub_branch`), so their events interleave in
    the session and only the branch check tells the runs apart. Isolation
    scope cannot draw this boundary: a `single_turn` sub-agent inherits its
    parent's isolation scope unchanged, so sibling runs of this agent carry
    the identical one.

    A standalone (`mode=None`) turn's opening message is its own fresh
    invocation with no sibling run to share it: `callback_context.branch` is
    `None` there, so the branch check admits everything and only the
    invocation/author boundary applies.

    Returns `None` on a run that has not yet gotten a tool response, which is
    the translator's first generation.
    """
    own_branch = (
        _BranchPath.from_string(callback_context.branch) if callback_context.branch else None
    )
    for event in reversed(callback_context.session.events):
        if own_branch is not None and event.branch:
            event_branch = _BranchPath.from_string(event.branch)
            if own_branch != event_branch and not own_branch.is_descendant_of(event_branch):
                continue
        if (
            event.invocation_id != callback_context.invocation_id
            or event.author != callback_context.agent_name
        ):
            return None
        for part in reversed(event.content.parts or () if event.content else ()):
            if part.function_response is not None:
                return part.function_response
    return None


def _before_translator_model(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> LlmResponse | None:
    """Open the turn on `translate` before the model is asked for anything.

    The translator's first generation of a turn has one thing to do: emit
    `translate` with no arguments. This returns that call as a synthetic
    response instead of reaching the model for it (`_SKIP_SIGNATURE` stands
    in for the signature Gemini requires on a function call), the way
    `harness.planner_seam.PlannerSeam._before_planner_model` returns a
    synthetic terminal one. ADK then runs the tool loop from it: `translate`
    executes, and the model is first called on the generation that reads its
    result.

    The discriminator is whether this run has already gotten a tool response
    of its own (`_last_tool_response`), scoped to this run so a standalone
    session's later turn, or a second `single_turn` call the planner makes
    inside one invocation, never reads another run's response as its own
    opener having already fired. Once a tool response exists, this callback
    has nothing left to do: every generation of this turn shares one
    thinking level, `AGENT_THINKING_CONFIG` from `BuiltInPlanner` or
    `DEFAULT_THINKING_CONFIG` from `override_thinking_on_sampled_turns` on a
    sampled turn (see `create_translator_agent`), so a generation that reads
    a judgement needs no override here to differ from one that does not.
    """
    del llm_request
    if _last_tool_response(callback_context) is None:
        return LlmResponse(
            content=types.Content(
                role="model",
                parts=[
                    types.Part(
                        function_call=types.FunctionCall(name=_TRANSLATE_TOOL, args={}),
                        thought_signature=_SKIP_SIGNATURE,
                    )
                ],
            )
        )
    return None


INSTRUCTION = (
    "You are the Translator Agent for the Development Finance Observatory. A plan's "
    "decisions and evidence requirements are settled with the user before you see them. "
    "The plan `propose_plan` recorded is already stored, and `translate` reads it "
    "itself, so the plan never appears in this message. The message that wrapped this "
    "call is the planner's instruction to you. "
    "Each turn opens on `translate`, called with no arguments before you are asked "
    "anything. By the time you read this it has found the data every dataset "
    "requirement needs, proposed the operations that carry every requirement out, and "
    "had them compiled into the workflow the user will approve. Its latest answer is "
    "the last message in this conversation, and your part is to act on that answer. "
    "Most calls settle it. Close the turn by calling `set_model_response`: with "
    '`{"outcome": "ok"}` once `translate` reports the plan compiled, or, once it '
    "reports something that could not be carried out, with the `referral` it gives -- "
    "`kind`, `reason`, and whichever of `requirement_id`, `decision_id`, "
    "`no_data_reason` and `recommendation` it carries, copied as they stand. Leave "
    "`candidates` out: they are attached from the recorded referral. "
    'Sometimes it stops instead with `"outcome": "needs_decision"`, `"judgement": '
    '"candidate_ranking"`: several sources answer one requirement, and ranking them is '
    "the one choice left to you. The payload names the requirement, the decisions that "
    "cite it, and each candidate with its own description -- read them, then call "
    "`translate` again with `requirement_id`, `choice` (the candidate you rank first, by "
    "its own spec_id) and `rationale`. Write the rationale for the person who settles "
    "this choice, not only to justify the pick to yourself: you are recommending, not "
    "choosing -- the source still goes to the user. That second call answers with the "
    "referral to close the turn on. "
    "A turn that closes any other way is refused outright, not read as an answer."
)

# The issue codes `propose_operations` maps to an `Action`, in the order
# tried: a source issue points back at discovery, a source or requirement
# that cannot be met points at a referral, and so does an uncovered
# operation kind. Anything else is a proposal to fix and resubmit.
_DISCOVER_AGAIN = frozenset(
    {
        CompileIssueCode.SOURCE_UNSELECTED,
        CompileIssueCode.SOURCE_NOT_DISCOVERED_FOR_REQUIREMENT,
        CompileIssueCode.SOURCE_PIN_VIOLATED,
    }
)
_UNMEETABLE = frozenset({CompileIssueCode.SOURCE_UNUSABLE, CompileIssueCode.REQUIREMENT_UNMEETABLE})


def _compile_failure_action(issues: Sequence[CompileIssue]) -> Action | None:
    """Return what to call next to move past a failed `propose_operations`, if known."""
    for issue in issues:
        if issue.code in _DISCOVER_AGAIN:
            return Action(
                call="discover_source", arguments={"requirement_id": issue.requirement_id}
            )
    for issue in issues:
        if issue.code in _UNMEETABLE:
            return Action(
                call="return_to_planner",
                arguments={
                    "kind": ReferralKind.REQUIREMENT_UNMEETABLE.value,
                    "requirement_id": issue.requirement_id,
                    "reason": issue.reason,
                },
            )
    for issue in issues:
        if issue.code is CompileIssueCode.NO_CAPABILITY:
            # Which of the requirement's several cited decisions is at fault
            # is not derivable from the operation: `decision_id` needs the
            # model's own judgement, so it is named in `needs` rather than
            # guessed at here.
            return Action(
                call="return_to_planner",
                arguments={"kind": ReferralKind.NO_OPERATION_COVERS.value, "reason": issue.reason},
                needs=("decision_id",),
            )
    return None


def _translatable_draft(tool_context: ToolContext) -> Plan | dict[str, object]:
    """Return the draft plan `propose_operations` may compile, or the problem that stops it.

    Needs a draft plan with no referral pending on it -- no translator tool
    settles one, so there is no action to point at; the planner reads it and
    proposes or rejects -- and at least one evidence requirement to carry out.
    """
    plan = current_draft(tool_context)
    if isinstance(plan, dict):
        return plan
    pending = load_referral(tool_context.state)
    if pending is not None:
        return problem(
            ProblemCode.REFERRAL_PENDING,
            f"A {pending.kind.value} referral is pending on this plan; the planner settles "
            "it before the plan is translated again.",
            outcome=Outcome.REFUSED,
            detail={"kind": pending.kind.value, "requirement_id": pending.requirement_id},
        )
    if not plan.evidence_requirements:
        return problem(
            ProblemCode.INVALID_PLAN,
            "This plan states no evidence requirements; the planner records them "
            "before a plan can be translated.",
            outcome=Outcome.REFUSED,
        )
    return plan


def build_translator_core(*, surface: ToolSurface) -> TranslatorCore:
    """Build the translator's three core functions, for `translate` -- or a test -- to drive directly.

    None of the three is exposed to a model as its own tool: the
    only tool this agent declares is `translate`
    (`build_translator_tools`), which drives these directly, in Python, the
    way `translator.mechanical.run_translate` orchestrates. Exposed here so
    a test can exercise `propose_operations`'s and `return_to_planner`'s own
    logic without going through `translate`.
    """

    async def propose_operations(
        translation: Translation, tool_context: ToolContext
    ) -> dict[str, object]:
        """Propose the operations the plan's requirements resolve into, and compile them.

        Call this once every dataset requirement has a discovered source and
        every derived requirement has an operation that carries it out. The
        proposal is compiled against the plan, the sources discovery
        recorded, and the live tool surface. It reaches the plan only if it
        compiles; otherwise the plan is left untouched and every issue comes
        back for another attempt.

        Args:
            translation: The operations, each naming its inputs by operation
                id, and the ids of those whose results answer the question.

        Returns:
            On success, the plan's new status, version and revision, the
            sources the workflow rests on, and its steps. Otherwise the
            problem that stopped it, carrying every issue found and, when
            known, the action that moves past it.
        """
        try:
            # ADK's argument preprocessing falls back to a raw dict when a
            # model value fails to validate; re-validating here turns that
            # into a retryable problem instead of an exception.
            translation = _TRANSLATION_ADAPTER.validate_python(translation)
        except ValidationError as exc:
            return problem(
                ProblemCode.INVALID_TRANSLATION,
                f"This translation does not have the shape a translation needs: {exc}",
                outcome=Outcome.INVALID_INPUT,
            )
        plan = _translatable_draft(tool_context)
        if isinstance(plan, dict):
            return plan
        try:
            schemas = await surface.list_tools()
        except ToolSurfaceUnavailableError as exc:
            return problem(ProblemCode.SURFACE_UNAVAILABLE, str(exc), outcome=Outcome.REFUSED)

        dataset_requirements = [
            requirement
            for requirement in plan.evidence_requirements
            if isinstance(requirement, DatasetRequirement)
        ]
        requirement_keys = {
            requirement.id: requirement_revision(requirement)
            for requirement in dataset_requirements
        }
        pinned = {
            requirement.id: requirement.source_spec_id
            for requirement in dataset_requirements
            if requirement.source_spec_id is not None
        }
        across = {
            requirement.id for requirement in dataset_requirements if requirement.across is not None
        }

        def settled(log: DiscoveryLog) -> dict[str, SelectedSource]:
            """The sources `log` settles for this plan, less any the turn dropped."""
            return settled_sources(
                tool_context.state,
                log,
                requirement_keys=requirement_keys,
                pinned=pinned,
                across=across,
            )

        # The log read that decides what to describe takes `discovery_lock`,
        # the same discipline `discover_source`'s own body follows; the
        # `describe` round trip `fetch_definitions` runs does not, so the
        # lock is released for it and re-taken only to apply what it found.
        async with discovery_lock(tool_context.state):
            log = load_log(tool_context.state)
            discovered = settled(log)
            definite_ids = log.definite_source_ids(requirement_keys=requirement_keys, pinned=pinned)
            definite_sources = {
                requirement_id: source
                for requirement_id, source in discovered.items()
                if requirement_id in definite_ids
            }
        if definite_sources:
            pairs = await fetch_definitions(
                surface,
                log,
                revision=plan_revision(plan),
                requirement_keys=requirement_keys,
                sources=definite_sources,
            )
            log = await apply_definitions(tool_context.state, pairs)
            discovered = settled(log)
        result = compile_plan(
            plan=plan,
            translation=translation,
            schemas={schema.name: schema for schema in schemas},
            discovered=discovered,
        )
        warnings = [warning.model_dump(mode="json") for warning in result.warnings]
        if result.workflow is None:
            return problem(
                ProblemCode.COMPILE_FAILED,
                f"This translation did not compile: {len(result.issues)} issue(s) found.",
                outcome=Outcome.INVALID_INPUT,
                action=_compile_failure_action(result.issues),
                detail={
                    "issues": [
                        issue.model_dump(mode="json", exclude_none=True) for issue in result.issues
                    ],
                    "warnings": warnings,
                },
            )

        plan = plan.with_translation(translation=translation, workflow=result.workflow)
        save_plan(tool_context.state, plan)
        return ok(
            status=plan.status.value,
            version=plan.version,
            revision=plan_revision(plan),
            # Each dataset once, with every spec_id it is retrieved under and
            # the phrase each was searched as: the workflow is the record of
            # what was approved.
            sources=described_sources(result.workflow.sources),
            steps=[
                {"id": step.id, "tool": f"{step.tool.server}.{step.tool.name}"}
                for step in result.workflow.steps
            ],
            warnings=warnings,
        )

    def return_to_planner(
        kind: ReferralKind,
        reason: str,
        tool_context: ToolContext,
        requirement_id: str | None = None,
        decision_id: str | None = None,
        recommendation: SourceRecommendation | None = None,
    ) -> dict[str, object]:
        """Refer the plan back to the planner: something in it cannot be carried out.

        Called instead of `propose_operations` when discovery finds no data,
        an ask it cannot serve, or not the source a requirement pins; when a
        source cannot meet a requirement or no operation can; when a
        requirement calls for an operation that does not exist; or when the
        discovery budget is spent. The plan is left untouched; the referral
        is saved to session state for the stall watchdog and `submit_plan`'s
        own gate, and this call's own return value is what the planner reads
        it from, once this agent's closing turn hands it back
        (`create_translator_agent`'s `output_schema`).

        Args:
            kind: Why the plan goes back.
            reason: What the requirement needs, and what was found or what
                the operations do instead.
            requirement_id: The requirement this is about, for every kind
                about a requirement.
            decision_id: The decision's slot, for `no_operation_covers`.
            recommendation: The translator's own ranked pick among a
                `source_ambiguous` referral's candidates, for the person who
                settles the choice to read. Only ever set for that kind.

        Returns:
            The referral, with the candidate sources attached when the source
            was ambiguous or a pinned source was not found -- and, for
            `no_data`, the nearest real sources found instead plus the reason
            code, each rendered without its `spec_id` -- or the problem that
            stopped it.
        """
        # A raw string may arrive here from a resumed candidate-ranking
        # answer the same way it does from a model-dispatched call: `kind`
        # is coerced on the way into `Referral` below, but the checks here
        # run first, so a kind the vocabulary does not hold is left as it
        # came and refused by `Referral`, which says so better than
        # anything written here.
        with suppress(ValueError):
            kind = ReferralKind(kind)
        # A draft, not any plan: a plan a successful compile already moved to
        # PENDING_APPROVAL cannot have a referral written over it.
        plan = current_draft(tool_context)
        if isinstance(plan, dict):
            return plan
        requirement: DatasetRequirement | None = None
        if requirement_id is not None:
            try:
                found = plan.requirement(requirement_id)
            except KeyError:
                known = [requirement.id for requirement in plan.evidence_requirements]
                return problem(
                    ProblemCode.REQUIREMENT_UNKNOWN,
                    f"This plan records no requirement {requirement_id!r}.",
                    outcome=Outcome.INVALID_INPUT,
                    detail={"known": known},
                )
            requirement = found if isinstance(found, DatasetRequirement) else None
        slots = [decision.slot for decision in plan.decisions]
        if decision_id is not None and decision_id not in slots:
            return problem(
                ProblemCode.DECISION_UNKNOWN,
                f"This plan records no {decision_id!r} decision.",
                outcome=Outcome.INVALID_INPUT,
                detail={"known": slots},
            )
        candidates: tuple[SelectedSource, ...] = ()
        no_data_reason: str | None = None
        if kind in MAY_CARRY_CANDIDATES and requirement_id is not None and requirement is not None:
            key = requirement_revision(requirement)
            log = load_log(tool_context.state)
            if kind is ReferralKind.NO_DATA:
                candidates = log.nearest_for(key, requirement_id)
                no_data_reason = log.no_data_reason(key, requirement_id)
            else:
                candidates = log.candidates_for(key, requirement_id)
                if kind is ReferralKind.SOURCE_AMBIGUOUS and len(candidates) < 2:
                    return problem(
                        ProblemCode.NO_CANDIDATES_RECORDED,
                        f"Discovery recorded no candidate sources for {requirement_id!r} in this "
                        "revision; a source is ambiguous only when `discover_source` answered "
                        "with candidates for it.",
                        outcome=Outcome.INVALID_INPUT,
                    )
            # `pinned_source_not_found` and `no_data` each carry whatever the
            # log holds, even none: the id the plan pinned was not among a
            # `definite` answer's one source either, or the nearest-series
            # relaxation found nothing, and a referral naming no candidates
            # there is telling the truth, not failing to look.
        try:
            referral = Referral(
                kind=kind,
                reason=reason,
                requirement_id=requirement_id,
                decision_id=decision_id,
                candidates=candidates,
                no_data_reason=no_data_reason,
                recommendation=recommendation,
            )
        except ValidationError as exc:
            return problem(
                ProblemCode.INVALID_REFERRAL,
                f"This referral does not have the shape a referral needs: {exc}",
                outcome=Outcome.INVALID_INPUT,
            )
        save_referral(tool_context.state, referral)
        return referred(_referral_payload(referral))

    (discover_source_tool,) = build_discovery_tools(surface=surface)
    return TranslatorCore(
        discover_source=discover_source_tool.func,
        propose_operations=propose_operations,
        return_to_planner=return_to_planner,
        surface=surface,
    )


def build_translator_tools(*, surface: ToolSurface) -> list[FunctionTool]:
    """Build the translator's one tool, closing over `surface` for the tool surface."""
    core = build_translator_core(surface=surface)

    async def translate(
        tool_context: ToolContext,
        requirement_id: str | None = None,
        choice: str | None = None,
        rationale: str | None = None,
        decision_id: str | None = None,
    ) -> dict[str, object]:
        """Find data, propose operations, and compile them into the approved workflow.

        Call with no arguments to run the whole loop: `discover_source` for
        every dataset requirement, then `propose_operations`, following each
        step's own next action, all without a further model turn. Most
        calls settle straight away. When ranking several candidate sources
        is the one thing left to decide, this instead answers
        `needs_decision`; settle it by calling this again with
        `requirement_id`, `choice` (the candidate's own spec_id) and
        `rationale`. A `no_operation_covers` escalation is settled the same
        way, with `decision_id` in place of `choice`/`rationale`.

        Args:
            requirement_id: Which pending judgement this answers. Leave unset
                to start (or restart) the whole loop.
            choice: The candidate ranked first, by its own spec_id -- answers
                a `candidate_ranking` judgement, together with `rationale`.
            rationale: Why `choice` is ranked first, for the person who
                settles it to read.
            decision_id: Which decision no operation covers -- answers an
                `unmet_capability` judgement.

        Returns:
            `{"outcome": "ok", ...}` once the plan compiles;
            `{"outcome": "referred", "referral": {...}}` once something
            cannot be carried out, or a pending judgement is answered;
            `{"outcome": "needs_decision", ...}` while a judgement is
            pending; or the problem that stopped a malformed call.
        """
        return await run_translate(
            tool_context,
            core=core,
            requirement_id=requirement_id,
            choice=choice,
            rationale=rationale,
            decision_id=decision_id,
        )

    return [FunctionTool(translate)]


def create_translator_agent(
    *,
    model_name: str,
    surface: ToolSurface,
    mode: AgentMode | None = None,
) -> LlmAgent:
    """Build the translator `LlmAgent`.

    Args:
        model_name: The model this agent runs on.
        surface: The tool surface to discover on and compile against.
            Injectable so tests can pin a surface, and so the harness can
            point it at whichever servers a plan is allowed to call.
        mode: ADK's agent mode. `"single_turn"` is what lets this agent be
            attached as a sub-agent the planner calls as a tool. Left unset,
            ADK treats it as a conversational agent, which is how an eval set
            scores it on its own.

    Returns:
        The configured `LlmAgent`.
    """
    return LlmAgent(
        name=AGENT_NAME,
        model=model_name,
        description=DESCRIPTION,
        instruction=INSTRUCTION,
        mode=mode,
        # Run as a `single_turn` sub-agent, ADK's own `run_llm_agent_as_node`
        # already defaults an agent with no explicit `include_contents` to
        # `'none'` -- this agent has no sub-agents of its own, so nothing
        # plants an other-agent-reply event inside its own turn the way this
        # agent's turn plants one inside the planner's (see
        # `planner.agent.create_planner_agent`'s `include_contents`), and
        # that implicit default already gives it the stateless, per-call
        # view its `single_turn` role calls for. Pinned here rather than
        # left to infer so that view stays fixed if ADK's own default
        # heuristic ever changes. Left unset for `mode=None` (the eval
        # set's standalone, conversational use, `create_translator_agent`'s
        # own `mode` doc below), which needs ADK's ordinary default of
        # `'default'`, not this one.
        include_contents="none" if mode == "single_turn" else "default",
        tools=build_translator_tools(surface=surface),
        # `AGENT_THINKING_CONFIG` (see `planner.settings`) pins every
        # generation of this agent's turn to LOW, including the one that
        # ranks candidates for `translate`'s `needs_decision` judgement.
        # `BuiltInPlanner` writes it into every request on its own
        # (`google.adk.flows.llm_flows._nl_planning`, ahead of any
        # `before_model_callback`), so nothing here chooses per call.
        planner=BuiltInPlanner(thinking_config=AGENT_THINKING_CONFIG),
        # `override_thinking_on_sampled_turns` runs first and always returns
        # `None`, so it never short-circuits the list: `_before_translator_model`
        # still runs after it and still returns `translate`'s call as a
        # synthetic response on the first generation of a turn, so that
        # generation never reaches the model at all.
        before_model_callback=[override_thinking_on_sampled_turns, _before_translator_model],
        # Drops this generation's thought-summary parts once the trace has
        # them, so this agent's own history never replays one into a later
        # generation of the same turn. See `drop_thought_parts`.
        after_model_callback=drop_thought_parts,
        # This union is what makes the planner's `dfo_translator` call answer
        # structurally. Without it, `process_llm_agent_output`
        # (`google.adk.workflow._llm_agent_wrapper`) sets this agent's node
        # output to the model's final turn verbatim, prose and all, and the
        # planner has to go find the typed `Referral` in session state for
        # itself. With it, `validate_schema` parses that same turn into
        # whichever of `Compiled` or `Referred` it matches, so the planner
        # reads the referral off `dfo_translator`'s own result. `Referred`
        # rather than `Referral` itself: with tools present, ADK enforces this
        # schema only through the `set_model_response` tool it injects
        # (`_OutputSchemaRequestProcessor`), and for a union type that tool
        # declares one free-form parameter and validates nothing itself
        # (`SetModelResponseTool.__init__`/`run_async`,
        # `google.adk.tools.set_model_response_tool`) -- the actual check runs
        # later, against whatever `after_tool_callback` leaves in
        # `tool_context.actions.set_model_response`.
        # `_close_on_the_recorded_referral` always replaces that with the
        # referral `translate` already recorded, in `Referred`'s own shape,
        # so this schema only has to accept what code produces, never what a
        # model retypes. Paired with `disallow_transfer_to_parent`
        # and `disallow_transfer_to_peers` below: both keep this agent on
        # `SingleFlow`, which is what lets `_OutputSchemaRequestProcessor`
        # (`google.adk.flows.llm_flows.prompt._schema`) enforce the
        # schema alongside this agent's own tools in the first place:
        # `LlmAgent._llm_flow` (`agents/llm_agent.py`) selects `SingleFlow`
        # over `AutoFlow` on exactly `disallow_transfer_to_parent and
        # disallow_transfer_to_peers and not sub_agents`, and `single_flow.py`
        # registers `prompt._schema.request_processor` as one of
        # `SingleFlow`'s own request processors.
        output_schema=Compiled | Referred,
        # Runs on every tool call this agent makes, `translate` included; it
        # only acts on `set_model_response` (`_close_on_the_recorded_referral`
        # checks the tool's own type first). With a referral on record it is
        # the closing turn; with none, it lets a `Compiled` answer through
        # and fails anything else.
        after_tool_callback=_close_on_the_recorded_referral,
        # ADK declares a `transfer_to_agent` tool to every `LlmAgent` with a
        # parent unless told not to (`disallow_transfer_to_parent` defaults
        # to `False`), regardless of `mode`: `mode="single_turn"` only
        # governs how the *parent* wraps this agent as a tool, not which
        # flow ADK builds for this agent's own generations. Left unset, the
        # translator's model gets a second, unmonitored way to hand control
        # back to the planner that never builds a `Referral` and never calls
        # `translate` -- the only tool
        # `harness.planner_seam.PlannerSeam._track_tool_call` inspects on
        # this agent's own callback. `disallow_transfer_to_peers` has no live
        # peer to matter against (the planner's only sub-agent is this
        # translator), but is set for the same reason: together with
        # `disallow_transfer_to_parent` and this agent's empty `sub_agents`,
        # it is what makes ADK's `LlmAgent._llm_flow` choose `SingleFlow`
        # over `AutoFlow`, so `transfer_to_agent` is never declared to this
        # agent's model at all, and `translate` is the only way back.
        disallow_transfer_to_parent=True,
        disallow_transfer_to_peers=True,
    )
