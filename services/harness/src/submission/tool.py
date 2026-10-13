"""`submit_analysis`: the planner's one submission tool, in two stages, over the shared service.

The planner reads compact cards (`submission.catalogue.render_cards`) and
submits through one stable envelope whose `submission_stage` says which stage
the call is:

- `intent` names the requirement kinds and ids the plan will use and records
  nothing. It answers `requirements`: each named kind's contract once
  (`submission.catalogue.select_contracts`), the node-to-kind mapping, the
  catalogue revision, the attempt id and the contract digest. A kind no card
  carries is listed under `unknown_kinds`, and the known contracts come back
  whatever else in the envelope is malformed.
  An `attempt_id` or `contract_digest` no answer gave is one of those issues.
- `complete` is the whole plan, checked by exactly the chain `propose_plan`
  runs (`submission.service`, then translation, then
  `planner.agent.declare_ready`) once the catalogue revision
  it names is current. A stale or missing revision is refused before any
  other check, with the current revision and the named kinds' contracts.

The stage is checked before anything else reads the envelope, so a missing or
unknown stage refuses with nothing admitted. Every call, refused or not,
records one immutable `Attempt` in session state (`ATTEMPT_STATE_PREFIX`). Only
a `complete` call that the service admits records a plan, so an intent or a
refused attempt has nothing that could run. The service addresses every
`action` it answers to the `complete` stage at the current revision
(`proposer`).

A `complete` answer keeps every field `propose_plan` answers with, its
`translation` included, and re-labels `outcome` (`SubmissionOutcome`):
`needs_fields` for field errors, every one at once with its path, and the
contracts of every kind they involve; `needs_decision` for a refusal only the user
can settle (`DECISION_CODES`); `ready_for_review` for a recorded plan
`planner.agent.declare_ready` declared ready, which it does only when the
translation compiled and the plan passes its gate, so `summary` is on the
answer; `recorded` for any other recorded plan: one whose translation refers
it back, one the gate blocked (`submission`), or one no translator ran on. Any
other refusal keeps its own
`outcome` and `code`. `code` stays on every refusal, so a reader that branches
on it (`planner.agent._combination_picks`, `harness.planner_seam`) reads both
delivery modes alike.

Contracts travel as JSON text: Gemini refuses a function response whose
objects carry raw `$ref` keys.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from enum import StrEnum
from typing import Any, override

from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types
from planner.declaration import compact_schema
from planner.plan import AnswerMode, AskAnswer, DecisionDraft, Decline, SessionState
from planner.records import Identifier
from planner.tool_results import Outcome, ProblemCode, problem
from pydantic import BaseModel, ConfigDict, ValidationError
from pydantic_core import to_jsonable_python

from submission.catalogue import catalogue_revision, select_contracts
from submission.service import Proposer

TOOL_NAME = "submit_analysis"

# The prefix of the session-state slots holding the latest `MAX_ATTEMPTS`
# attempts, as `Attempt` dicts. Attempt `n` is held under the prefix and
# `n % MAX_ATTEMPTS`.
ATTEMPT_STATE_PREFIX = "submission:attempt:"

# The most attempts a session keeps. An attempt overwrites the one
# `MAX_ATTEMPTS` before it, so what the slots hold stays bounded across a long
# conversation.
MAX_ATTEMPTS = 64


class SubmissionStage(StrEnum):
    """Which stage one `submit_analysis` call is."""

    INTENT = "intent"
    COMPLETE = "complete"


class SubmissionOutcome(StrEnum):
    """The `outcome` values `submit_analysis` answers with beside `planner.tool_results.Outcome`."""

    REQUIREMENTS = "requirements"
    NEEDS_FIELDS = "needs_fields"
    NEEDS_DECISION = "needs_decision"
    RECORDED = "recorded"
    READY_FOR_REVIEW = "ready_for_review"


# Refusals that wait on the user: a referral they have to answer, or a
# combination they have to consent to.
DECISION_CODES = frozenset({ProblemCode.REFERRAL_PENDING, ProblemCode.COMBINATION_UNASKED})


class Attempt(BaseModel):
    """One `submit_analysis` call, as recorded.

    `sequence` numbers the session's calls from 1, and `attempt_id` is
    `attempt-<sequence>`, so an id stays unique after older attempts are
    dropped (`MAX_ATTEMPTS`). `stage` is what the call stated, `None` when it stated none.
    `request_digest` is the SHA-256 of the call's arguments as canonical
    JSON. `catalogue_revision` is the revision the call was checked against.
    `corrects` is the attempt the call names in `attempt_id`.
    `contract_digest` is the digest of the contracts this answer delivered,
    if any. `plan_version` is the version an admitted call recorded.

    A stored record is read leniently, since another version of this module
    may have written it: a key it does not declare is ignored and an optional
    field it lacks is `None`.
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    sequence: int
    attempt_id: str
    stage: str | None = None
    request_digest: str
    catalogue_revision: str
    outcome: str
    code: str | None = None
    corrects: str | None = None
    contract_digest: str | None = None
    plan_version: int | None = None


def load_attempts(state: SessionState) -> tuple[Attempt, ...]:
    """The attempts `state` keeps, the latest `MAX_ATTEMPTS`, oldest first.

    A slot holding a record `Attempt` cannot read is skipped.
    """
    attempts = []
    for slot in range(MAX_ATTEMPTS):
        record = state.get(f"{ATTEMPT_STATE_PREFIX}{slot}")
        if record is None:
            continue
        try:
            attempts.append(Attempt.model_validate(record))
        except ValidationError:
            continue
    return tuple(sorted(attempts, key=lambda attempt: attempt.sequence))


def _append_attempt(
    state: SessionState, stage: object, args: Mapping[str, Any], answer: Mapping[str, Any]
) -> str:
    """Record the call `args` made and `answer` answered as the session's next attempt, and name it.

    The sequence is read and the attempt holding it written with no await
    between, so calls running concurrently in one session each take their
    own. The write touches only the attempt's own slot, so the event's state
    delta carries one attempt, and the deltas of parallel calls, which ADK
    merges key by key, never overwrite each other's attempts.
    """
    kept = load_attempts(state)
    sequence = kept[-1].sequence + 1 if kept else 1
    attempt_id = f"attempt-{sequence}"
    recorded = answer["outcome"] in (SubmissionOutcome.RECORDED, SubmissionOutcome.READY_FOR_REVIEW)
    version = answer.get("version") if recorded else None
    contract_digest = answer.get("contract_digest")
    code = answer.get("code")
    corrects = args.get("attempt_id")
    attempt = Attempt(
        sequence=sequence,
        attempt_id=attempt_id,
        stage=stage if isinstance(stage, str) else None,
        request_digest=_digest(args),
        catalogue_revision=catalogue_revision(),
        outcome=str(answer["outcome"]),
        code=code if isinstance(code, str) else None,
        corrects=corrects if isinstance(corrects, str) else None,
        contract_digest=contract_digest if isinstance(contract_digest, str) else None,
        plan_version=version if isinstance(version, int) else None,
    )
    state[f"{ATTEMPT_STATE_PREFIX}{sequence % MAX_ATTEMPTS}"] = attempt.model_dump(mode="json")
    return attempt_id


def proposer() -> Proposer:
    """`submit_analysis` as `submission.service` addresses it: the `complete` stage at the current revision.

    Every field error is reported, so a `needs_fields` answer names each one
    and carries the contracts of every kind they involve.
    """
    return Proposer(
        call=TOOL_NAME,
        arguments={
            "submission_stage": SubmissionStage.COMPLETE.value,
            "catalogue_revision": catalogue_revision(),
        },
        max_reported_errors=None,
    )


class _RequirementItem(BaseModel):
    """A requirement as the envelope declares it: `kind` and `id`, then the kind's own fields."""

    model_config = ConfigDict(extra="allow")

    kind: str
    id: Identifier


class _Envelope(BaseModel):
    """The arguments `submit_analysis` declares. Used for its declaration only."""

    submission_stage: SubmissionStage
    catalogue_revision: str
    question: str | None = None
    answer_mode: AnswerMode | None = None
    decisions: list[DecisionDraft] = []
    evidence_requirements: list[_RequirementItem] = []
    removed_decisions: list[str] = []
    removed_evidence_requirements: list[str] = []
    declined: list[Decline] = []
    answered: list[AskAnswer] = []
    amends_version: int | None = None
    starts_new_plan: bool = False
    attempt_id: str | None = None
    contract_digest: str | None = None


class _Scalars(BaseModel):
    """The envelope's single values, checked before anything reads them. The lists are the service's."""

    model_config = ConfigDict(extra="ignore")

    catalogue_revision: str | None = None
    question: str | None = None
    answer_mode: AnswerMode | None = None
    amends_version: int | None = None
    starts_new_plan: bool = False
    attempt_id: str | None = None
    contract_digest: str | None = None


# The `propose_plan` arguments a `complete` call passes through.
_PROPOSAL_ARGUMENTS = (
    "question",
    "answer_mode",
    "decisions",
    "evidence_requirements",
    "removed_decisions",
    "removed_evidence_requirements",
    "declined",
    "answered",
    "amends_version",
    "starts_new_plan",
)

# What each argument only `submit_analysis` declares means. The rest are
# `propose_plan`'s and described by the caller.
ENVELOPE_ARGUMENTS = {
    "submission_stage": (
        "`intent` names the kinds and ids the plan uses and records nothing; it answers "
        "each kind's contract. `complete` is the whole plan, checked and recorded."
    ),
    "catalogue_revision": (
        "The catalogue revision your instruction states, or the one the last answer "
        "returned. A `complete` call naming another is refused before any check."
    ),
    "evidence_requirements": (
        "The requirements this call adds or replaces, keyed by `id`, unique within the "
        "plan. Each states its `kind`, one of the kinds listed in your instruction, and "
        "its `id`. An intent call states those and the ids each reads (`input_id`, or "
        "each side's); a complete call states every field the kind's contract requires. "
        "An id the plan already states replaces that requirement whole. Each cites in "
        "`decision_ids` the decisions that fixed its fields."
    ),
    "attempt_id": "The `attempt_id` of the answer this call follows, when it follows one.",
    "contract_digest": "The `contract_digest` of the contracts this call was written from.",
}

_DESCRIPTION = (
    "Propose a plan or patch the recorded one, in two stages. Every answer names an "
    "`attempt_id` and the `catalogue_revision`."
)


def _digest(value: Any) -> str:
    canonical = json.dumps(
        to_jsonable_python(value, fallback=str), sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def _named_kinds(args: Mapping[str, Any]) -> tuple[dict[str, str], list[str]]:
    """The node-to-kind mapping the envelope's requirements state, and what is wrong with any item.

    An item without a string `kind` and `id` maps nothing and is named in the issues.
    """
    items = args.get("evidence_requirements")
    if items is None:
        return {}, []
    if not isinstance(items, list):
        return {}, ["evidence_requirements: a list of requirements"]
    kinds: dict[str, str] = {}
    issues = []
    for index, item in enumerate(items):
        kind = item.get("kind") if isinstance(item, Mapping) else None
        node = item.get("id") if isinstance(item, Mapping) else None
        if not isinstance(kind, str) or not isinstance(node, str):
            issues.append(f"evidence_requirements[{index}]: needs a string `kind` and `id`")
        elif node in kinds:
            issues.append(f"evidence_requirements[{index}]: `id` {node!r} is stated twice")
        else:
            kinds[node] = kind
    return kinds, issues


def _contracts(kinds: Mapping[str, str]) -> dict[str, object]:
    """The contract fields an answer carries for the kinds `kinds` maps to."""
    selection = select_contracts(kinds.values())
    return {
        "catalogue_revision": catalogue_revision(),
        "kinds": dict(kinds),
        "unknown_kinds": list(selection.unknown),
        "contract_digest": selection.digest,
        "contracts": json.dumps(selection.payload, separators=(",", ":")),
    }


def _field_errors(errors: Sequence[Mapping[str, Any]]) -> list[dict[str, object]]:
    return [
        {
            "path": ".".join(str(element) for element in error["loc"]),
            "type": error["type"],
            "message": error["msg"],
            **(
                {"constraint": to_jsonable_python(error["ctx"], fallback=str)}
                if error.get("ctx")
                else {}
            ),
        }
        for error in errors
    ]


def _involved_kinds(
    errors: Sequence[Mapping[str, Any]], args: Mapping[str, Any], kinds: Mapping[str, str]
) -> dict[str, str]:
    """The node-to-kind mapping of every requirement a field error is located in."""
    items = args.get("evidence_requirements")
    involved: dict[str, str] = {}
    for error in errors:
        loc = error["loc"]
        if len(loc) < 2 or loc[0] != "evidence_requirements" or not isinstance(loc[1], int):
            continue
        item = items[loc[1]] if isinstance(items, list) and loc[1] < len(items) else None
        node = item.get("id") if isinstance(item, Mapping) else None
        if isinstance(node, str) and node in kinds:
            involved[node] = kinds[node]
    return involved


def _needs_fields(
    message: str,
    errors: Sequence[Mapping[str, Any]],
    args: Mapping[str, Any],
    kinds: Mapping[str, str],
) -> dict[str, object]:
    return {
        "outcome": SubmissionOutcome.NEEDS_FIELDS.value,
        "code": ProblemCode.INVALID_PLAN.value,
        "message": message,
        "fields": _field_errors(errors),
        **_contracts(_involved_kinds(errors, args, kinds)),
    }


def _labelled(
    answer: Mapping[str, Any], args: Mapping[str, Any], kinds: Mapping[str, str]
) -> dict[str, object]:
    """`answer`, as `propose_plan` gave it, with `submit_analysis`'s outcome."""
    if "version" in answer and answer.get("outcome") == Outcome.OK.value:
        # `declare_ready` adds `summary` exactly when it declares the plan ready.
        ready = "summary" in answer
        outcome = SubmissionOutcome.READY_FOR_REVIEW if ready else SubmissionOutcome.RECORDED
        return {**answer, "outcome": outcome.value}
    code = answer.get("code")
    detail = answer.get("detail")
    errors = detail.get("errors") if isinstance(detail, dict) else None
    if code == ProblemCode.INVALID_PLAN and isinstance(errors, list):
        return _needs_fields(
            "Some fields break their kind's contract; correct each one `fields` names.",
            errors,
            args,
            kinds,
        )
    if code in DECISION_CODES:
        return {**answer, "outcome": SubmissionOutcome.NEEDS_DECISION.value}
    return dict(answer)


def _refused(code: ProblemCode, message: str) -> dict[str, object]:
    return problem(code, message, outcome=Outcome.INVALID_INPUT)


def _unknown_receipts(args: Mapping[str, Any], attempts: tuple[Attempt, ...]) -> list[str]:
    """What is wrong with an `attempt_id` or `contract_digest` no answer in this session gave."""
    unknown = []
    attempt_id = args.get("attempt_id")
    if isinstance(attempt_id, str) and attempt_id not in {
        attempt.attempt_id for attempt in attempts
    }:
        unknown.append(
            f"`attempt_id` {attempt_id!r} names no answer in this conversation; echo the one "
            "the last answer gave, or leave it out."
        )
    contract_digest = args.get("contract_digest")
    if isinstance(contract_digest, str) and contract_digest not in {
        attempt.contract_digest for attempt in attempts
    }:
        unknown.append(
            f"`contract_digest` {contract_digest!r} names no contracts returned in this "
            "conversation; echo the one the last answer gave, or leave it out."
        )
    return unknown


def _intent(kinds: Mapping[str, str], issues: list[str]) -> dict[str, object]:
    """An `intent` answer, without the `action` that names its attempt (`_follow_up`)."""
    answer: dict[str, object] = {
        "outcome": SubmissionOutcome.REQUIREMENTS.value,
        **_contracts(kinds),
        "message": (
            "Nothing was recorded. Fill each requirement from its kind's contract and call "
            f"{TOOL_NAME} with `submission_stage` `complete`, echoing `catalogue_revision`, "
            "`attempt_id` and `contract_digest`."
        ),
    }
    if issues:
        answer["issues"] = issues
    return answer


def _follow_up(answer: Mapping[str, object], attempt_id: str) -> dict[str, object]:
    """The `complete` call an `intent` answer recorded as `attempt_id` asks for next."""
    return {
        "call": TOOL_NAME,
        "arguments": {
            "submission_stage": SubmissionStage.COMPLETE.value,
            "catalogue_revision": answer["catalogue_revision"],
            "attempt_id": attempt_id,
            "contract_digest": answer["contract_digest"],
        },
    }


type Propose = Callable[..., Awaitable[dict[str, object]]]


async def submit_analysis(
    state: SessionState, args: Mapping[str, Any], *, propose: Propose
) -> dict[str, object]:
    """Answer one `submit_analysis` call and record it as an attempt.

    `args` are the call's arguments as the model sent them. `propose` is the
    `propose_plan` chain, translation included, called with the `propose_plan`
    arguments a current `complete` call states.
    """
    stage = args.get("submission_stage")
    answer = await _answer(stage, args, load_attempts(state), propose=propose)
    attempt_id = _append_attempt(state, stage, args, answer)
    if answer["outcome"] == SubmissionOutcome.REQUIREMENTS:
        answer["action"] = _follow_up(answer, attempt_id)
    return {**answer, "attempt_id": attempt_id}


async def _answer(
    stage: object,
    args: Mapping[str, Any],
    attempts: tuple[Attempt, ...],
    *,
    propose: Propose,
) -> dict[str, object]:
    """The answer to a call stating `stage`, its checks in order.

    The stage, then for `complete` the catalogue revision, then the single
    values, then the receipts. An `intent` call answers its contracts
    whatever else is wrong, naming each problem under `issues`.
    """
    if stage not in tuple(SubmissionStage):
        return _refused(
            ProblemCode.INVALID_ARGUMENTS,
            f"`submission_stage` is `intent` or `complete`, not {stage!r}; nothing was checked "
            "or recorded.",
        )
    kinds, issues = _named_kinds(args)
    receipts = _unknown_receipts(args, attempts)
    if stage == SubmissionStage.INTENT:
        return _intent(kinds, [*issues, *_scalar_issues(args), *receipts])
    stale = _catalogue_stale(args.get("catalogue_revision"), kinds)
    if stale is not None:
        return stale
    try:
        scalars = _Scalars.model_validate(args)
    except ValidationError as exc:
        return _needs_fields(
            "Some arguments are not the type they are declared as.",
            exc.errors(include_url=False, include_input=False),
            args,
            kinds,
        )
    if receipts:
        return _refused(ProblemCode.INVALID_ARGUMENTS, " ".join(receipts))
    return await _complete(args, kinds, scalars, propose=propose)


def _scalar_issues(args: Mapping[str, Any]) -> list[str]:
    """What is wrong with the envelope's single values, each by its argument."""
    try:
        _Scalars.model_validate(args)
    except ValidationError as exc:
        return [
            f"{'.'.join(map(str, error['loc']))}: {error['msg']}"
            for error in exc.errors(include_url=False, include_input=False)
        ]
    return []


def _catalogue_stale(revision: object, kinds: Mapping[str, str]) -> dict[str, object] | None:
    """The refusal for a `complete` call naming `revision` when it is not the current one."""
    current = catalogue_revision()
    if revision == current:
        return None
    return {
        **_refused(
            ProblemCode.CATALOGUE_STALE,
            f"`catalogue_revision` {revision!r} is not the current catalogue, {current!r}; "
            "nothing was checked or recorded. Rewrite the requirements against these "
            "contracts and submit again with the current revision.",
        ),
        **_contracts(kinds),
    }


async def _complete(
    args: Mapping[str, Any], kinds: Mapping[str, str], scalars: _Scalars, *, propose: Propose
) -> dict[str, object]:
    """A current `complete` call, handed to `propose`.

    `propose` is given the single values as `scalars` checked them, so an
    `amends_version` sent as `"3"` reaches it as `3`.
    """
    checked = scalars.model_dump(mode="json", include=scalars.model_fields_set)
    answer = await propose(
        **{name: checked.get(name, args[name]) for name in _PROPOSAL_ARGUMENTS if name in args}
    )
    return _labelled(answer, args, kinds)


class SubmitAnalysisTool(BaseTool):
    """The `submit_analysis` declaration, and its calls handed over unconverted.

    ADK's `FunctionTool` would build the declaration from a signature, refuse
    a call missing a required argument with its own error, and convert or
    reject arguments before the handler saw them, each of which would answer
    without recording an attempt. This tool hands the raw arguments to
    `handle`, which checks them itself.

    `arguments` describes every declared argument `ENVELOPE_ARGUMENTS` does
    not, `propose_plan`'s own descriptions.
    """

    def __init__(
        self,
        handle: Callable[[Mapping[str, Any], ToolContext], Awaitable[dict[str, object]]],
        *,
        arguments: Mapping[str, str],
    ) -> None:
        super().__init__(name=TOOL_NAME, description=_DESCRIPTION)
        self._handle = handle
        self._arguments = {**arguments, **ENVELOPE_ARGUMENTS}

    @override
    def _get_declaration(self) -> types.FunctionDeclaration:
        schema = _Envelope.model_json_schema()
        properties: dict[str, Any] = schema["properties"]
        if set(properties) != set(self._arguments):
            raise ValueError(
                f"{self.name} declares {sorted(properties)} and describes "
                f"{sorted(self._arguments)}; describe each argument exactly once"
            )
        described = {
            name: {**argument, "description": self._arguments[name]}
            for name, argument in properties.items()
        }
        return types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters_json_schema=compact_schema(schema | {"properties": described}),
        )

    @override
    async def run_async(self, *, args: dict[str, Any], tool_context: ToolContext) -> Any:
        return await self._handle(args, tool_context)
