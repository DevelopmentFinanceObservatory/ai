"""The `switch` node: a scope choice on the plan card moved to another option, in code.

The `approve` node routes an answer carrying `switch` here. The node reads
the decision the user switched and every decision the card lists with it
(`harness.plan_display.listed_together`), and for each the source it pins
and that source's stored scope (`harness.turn_records.SearchReading.scope`).
It posts the scope's base identity with its entries and the switch
(`harness.scope_choices.switch_entries`) through data-mcp's
`resolve_definition`, which registers the identity they compile to. The
plan is then revised by the planner seam with no model call
(`harness.graph.seams.PlannerModel.revise`): each requirement pinning a
switched source pins the compiled one, the decisions follow their pins,
and the plan compiles again. The node stores each new reading, so later
searches carry the switch, and offers the revised plan back to `approve`
with what else moved (`harness.scope_choices.moved_choices`).

A switch Search refuses (`requested_option_unavailable`,
`conflicting_requested_options`), or one that fails anywhere else, offers
the unchanged plan again with a note saying so, and leaves the planner's
drafting state as it was.

A chip under an answer applies one option the same way, with no plan
(`apply_stored_choice`): the `seed` node posts the stored reading's base
identity with the switched entries, stores the reading Search compiles,
fetches it through the orchestrator's `fetch` tool and hands the result to
the agent, which answers from it with no search.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from compiler.surface import ToolSurface, ToolSurfaceUnavailableError
from data_mcp.search_result import SEARCH_ENVELOPE_META_KEY
from google.adk import Context, Event
from planner.plan import (
    DatasetRequirement,
    Decision,
    PinSource,
    SessionState,
    authored,
    earlier_block,
    is_scope_choice,
    pin_sources,
    plan_revision,
    save_pin_sources,
    windowed_pins,
)
from planner.plan import Plan as PlannerPlan
from pydantic import ValidationError
from translator.discovery import RESOLVE_DEFINITION

from harness.graph.seams import PlannerModel
from harness.graph.state import (
    ORCHESTRATOR_APPLIED_KEY,
    ORCHESTRATOR_SPECS_KEY,
    conversation_readings,
    readable_plan,
    store_search,
)
from harness.plan_display import listed_together
from harness.scope_choices import (
    PartScope,
    ScopeChoice,
    mark_planner_axes,
    moved_choices,
    switch_entries,
)
from harness.tool_contracts import resolver_digest
from harness.turn_records import SearchReading, stored_choice

logger = logging.getLogger(__name__)

# The codes Search refuses a switch it cannot apply with (422).
_SCOPE_REFUSALS = frozenset({"requested_option_unavailable", "conflicting_requested_options"})

# The state the planner seam and the translator keep while drafting, which a
# revision that fails leaves as it found.
_DRAFTING_PREFIXES = ("planner:", "translator:")


# The tool name and step id the Activity feed shows a chip's application under.
APPLY_CHOICE_STEP_TOOL = "apply_choice"
# The code that step fails with when the chip cannot be applied; its message
# is the note `SwitchFailedError` carries.
SCOPE_CHOICE_NOT_APPLIED = "scope_choice_not_applied"


class SwitchFailedError(Exception):
    """A switch that cannot be applied; the message is the note the card shows."""


@dataclass(frozen=True)
class _Switched:
    """One source the switch moves: the pin it held, its reading and scope,
    the answer Search gave, and the spec_id and scope that answer states."""

    pin: str
    reading: SearchReading
    scope: PartScope
    envelope: dict[str, Any]
    spec_id: str
    after: PartScope


def switch_node(
    *, planner: PlannerModel, definition_surface: ToolSurface
) -> Callable[[Context, Any], AsyncIterator[Any]]:
    """The `switch` node, revising with `planner` and posting through `definition_surface`."""

    async def switch(ctx: Context, node_input: Any) -> AsyncIterator[Any]:
        """Apply the switch `node_input` carries (`{"switch": {"choice_id", "option_id"}}`) and offer the plan again.

        One route out, `"offer"`, to the `approve` node: the revised plan
        with a new revision and `moved`, or the unchanged plan with a
        `refusal_note`.
        """
        pick = node_input.get("switch") if isinstance(node_input, dict) else None
        pick = pick if isinstance(pick, dict) else {}
        plan = readable_plan(ctx, ctx.state.get("plan"))
        assert plan is not None, "approve routes a switch only for a plan it read"
        choice_id, option_id = str(pick.get("choice_id")), str(pick.get("option_id"))
        try:
            output = await _switched_plan(
                ctx,
                plan,
                choice_id=choice_id,
                option_id=option_id,
                planner=planner,
                surface=definition_surface,
            )
        except SwitchFailedError as exc:
            note = f"{exc} Version {plan.version} is still there to approve."
            logger.info("plan switch not applied: choice_id=%s option_id=%s", choice_id, option_id)
            output = {
                **plan.model_dump(mode="json"),
                "summary": note,
                "revision": plan_revision(plan),
                "refusal_note": note,
            }
        yield Event(output=output, route="offer")

    return switch


async def _switched_plan(
    ctx: Context,
    plan: PlannerPlan,
    *,
    choice_id: str,
    option_id: str,
    planner: PlannerModel,
    surface: ToolSurface,
) -> dict[str, Any]:
    """The `approve` payload of `plan` revised for the switch.

    Raises:
        SwitchFailedError: If the switch cannot be applied, with the note to show.
    """
    clicked = next(
        (d for d in plan.decisions if d.scope is not None and d.scope.choice_id == choice_id),
        None,
    )
    if clicked is None or clicked.scope is None:
        raise SwitchFailedError("That choice is no longer on the plan, so nothing changed.")
    axis = clicked.scope.axis
    state = cast(SessionState, ctx.state)
    to_origin = {copy.spec_id: copy.windowed_from for copy in windowed_pins(state).values()}
    readings = _readings(ctx)
    targets = [
        (pin, reading, scope, choice)
        for decision in listed_together(plan, clicked.slot)
        for pin, reading, scope, choice in [
            _target(decision, to_origin=to_origin, readings=readings)
        ]
    ]
    label = next(
        (option.label for option in targets[0][3].options if option.option_id == option_id),
        option_id,
    )
    if any(not any(o.option_id == option_id for o in choice.options) for *_, choice in targets):
        raise SwitchFailedError(
            f"Search does not offer {label} for this source, so nothing changed."
        )
    specs = ctx.state.get(ORCHESTRATOR_SPECS_KEY) or {}
    sources = pin_sources(state)
    switched = await asyncio.gather(
        *(
            _resubmit(
                surface,
                pin=pin,
                reading=reading,
                scope=scope,
                entries=[
                    entry.model_dump(mode="json", exclude_none=True)
                    for entry in switch_entries(scope, choice=choice, option_id=option_id)
                ],
                block=specs.get(pin) or _source_block(sources.get(pin), pin),
                label=label,
            )
            for pin, reading, scope, choice in targets
        )
    )
    drafting = _drafting_state(ctx)
    moved_to = {item.pin: item.spec_id for item in switched}
    digests = [_reading_digest(item, axis=axis) for item in switched]
    save_pin_sources(
        state,
        {
            **sources,
            **{
                item.spec_id: _pin_source(sources.get(item.pin), digest, item.reading)
                for item, digest in zip(switched, digests, strict=True)
            },
        },
    )
    requirements = [
        requirement.model_copy(update={"source_spec_id": moved_to[origin]}).model_dump(mode="json")
        for requirement in authored(plan).evidence_requirements
        if isinstance(requirement, DatasetRequirement)
        and (origin := to_origin.get(requirement.source_spec_id or "", requirement.source_spec_id))
        in moved_to
    ]
    revised = await planner.revise(
        ctx=ctx,
        prior_plan=plan.model_dump(mode="json"),
        evidence_requirements=requirements,
        inherits={new: old for old, new in moved_to.items()},
    )
    if "plan" not in revised:
        _restore(ctx, drafting)
        logger.warning("plan switch not revised: %s", revised.get("refused"))
        raise SwitchFailedError(
            f"The plan could not be revised to count {label}, so nothing changed."
        )
    revised_plan = PlannerPlan.model_validate(revised["plan"])
    ctx.state["plan"] = revised_plan.model_dump_json()
    for item, digest in zip(switched, digests, strict=True):
        store_search(
            ctx.state,
            digest,
            asked=[item.reading.asked_roles],
            held_open=[item.reading.open_roles],
            share_sides=[item.reading.share_over],
        )
    moved: list[dict[str, str]] = []
    for item in switched:
        for entry in moved_choices(item.scope, item.after, switched=axis):
            if not any(
                (seen["from_label"], seen["to_label"]) == (entry["from_label"], entry["to_label"])
                for seen in moved
            ):
                moved.append(entry)
    logger.info(
        "plan switched: choice_id=%s option_id=%s sources=%d version=%d",
        choice_id,
        option_id,
        len(switched),
        revised_plan.version,
    )
    return {
        **revised_plan.model_dump(mode="json"),
        "summary": revised["summary"],
        "revision": plan_revision(revised_plan),
        "moved": moved,
    }


def _readings(ctx: Context) -> list[SearchReading]:
    """The conversation's stored readings, newest first: this turn's, then earlier turns'."""
    readings: list[SearchReading] = []
    for item in reversed(conversation_readings(ctx.state)):
        try:
            readings.append(SearchReading.model_validate(item))
        except ValidationError:
            continue
    return readings


def _target(
    decision: Decision, *, to_origin: Mapping[str, str], readings: Sequence[SearchReading]
) -> tuple[str, SearchReading, PartScope, ScopeChoice]:
    """The pin `decision` states, the newest reading of it with a scope, that scope and the choice on the decision's axis.

    Raises:
        SwitchFailedError: If `decision` is no scope decision, or no stored
            reading of its pin holds the choice.
    """
    if not is_scope_choice(decision) or decision.fixes is None or decision.scope is None:
        raise SwitchFailedError("That choice is no longer on the plan, so nothing changed.")
    pinned = decision.fixes.source_spec_id or ""
    pin = to_origin.get(pinned, pinned)
    for reading in readings:
        if reading.spec_id != pin or reading.scope is None:
            continue
        choice = next((c for c in reading.scope.choices if c.axis == decision.scope.axis), None)
        if choice is not None:
            return pin, reading, reading.scope, choice
    raise SwitchFailedError("Search's record of this source is gone, so nothing changed.")


def _source_block(source: PinSource | None, pin: str) -> Mapping[str, Any] | None:
    """The block the planner was shown for `pin`, when it was shown one."""
    return source.spec_block(pin) if source is not None else None


async def _resubmit(
    surface: ToolSurface,
    *,
    pin: str,
    reading: SearchReading,
    scope: PartScope,
    entries: list[dict[str, Any]],
    block: Mapping[str, Any] | None,
    label: str,
) -> _Switched:
    """Post `scope`'s base identity with `entries` through data-mcp's `resolve_definition`.

    Posts the part's entity groups beside it, when its block holds any, so
    Search names them again.

    Raises:
        SwitchFailedError: If the call does not complete, Search refuses it,
            or its answer is not one definite part with a scope.
    """
    arguments: dict[str, object] = {
        "spec_id": scope.base.spec_id,
        "identity": scope.base.identity,
        "scope_entries": entries,
    }
    groups = ((block or {}).get("resolved") or {}).get("entity_groups")
    if groups:
        arguments["entity_groups"] = groups
    try:
        answer = await surface.call_tool(tool=RESOLVE_DEFINITION, arguments=arguments)
    except ToolSurfaceUnavailableError as exc:
        logger.warning("plan switch resubmit unavailable: pin=%s: %s", pin, exc)
        raise SwitchFailedError(
            f"Search could not be reached to count {label}, so nothing changed."
        ) from exc
    if answer.is_error:
        refusal = answer.structured_content or {}
        code = refusal.get("upstream_code") or refusal.get("code")
        logger.info("plan switch refused: pin=%s code=%s", pin, code)
        if code in _SCOPE_REFUSALS:
            raise SwitchFailedError(
                f"Search cannot count {label} for this source, so nothing changed."
            )
        raise SwitchFailedError(f"Search could not count {label} just now, so nothing changed.")
    envelope = (answer.meta or {}).get(SEARCH_ENVELOPE_META_KEY)
    parts = envelope.get("parts") if isinstance(envelope, dict) else None
    part = parts[0] if isinstance(parts, list) and len(parts) == 1 else None
    spec = part.get("spec") if isinstance(part, dict) and part.get("status") == "definite" else None
    if not isinstance(spec, dict) or not spec.get("spec_id"):
        raise SwitchFailedError(
            f"Search holds no data counted as {label} here, so nothing changed."
        )
    block = cast(dict[str, Any], spec)
    resolved = block.get("resolved")
    try:
        after = PartScope.from_search(
            resolved.get("scope_choices") if isinstance(resolved, dict) else None
        )
    except ValidationError as exc:
        raise SwitchFailedError(
            f"Search's answer for {label} could not be read, so nothing changed."
        ) from exc
    return _Switched(
        pin=pin,
        reading=reading,
        scope=scope,
        envelope=cast(dict[str, Any], envelope),
        spec_id=str(block["spec_id"]),
        after=after,
    )


def _reading_digest(item: _Switched, *, axis: str) -> dict[str, Any]:
    """The digest `store_search` stores for `item`'s answer, switched on `axis`, under the text of the reading it replaces.

    Search echoes the spec_id data-mcp posts as the part's text, so the
    reading keeps the words the user's search used. The axes the planner's
    words settled on the reading (`SearchReading.planner_axes`) stay the
    planner's through the switch, less the switched axis, which is now the
    user's (`harness.scope_choices.PLANNER_AXES_BLOCK_KEY`).
    """
    digest = resolver_digest(item.envelope)
    (part,) = digest.get("part_digests") or [{}]
    part["variable"] = item.reading.text
    mark_planner_axes(digest, set(item.reading.planner_axes) - {axis})
    return digest


def _pin_source(
    replaced: PinSource | None, digest: Mapping[str, Any], reading: SearchReading
) -> PinSource:
    """The pin source of a switched dataset: the one it replaces, holding the switched block."""
    (part,) = digest["part_digests"]
    block = part["spec_block"]
    if replaced is None:
        return PinSource(block=block, settled=True, asked_roles=reading.asked_roles)
    return replaced.model_copy(update={"block": block, "earlier_turn": False})


def _drafting_state(ctx: Context) -> dict[str, Any]:
    """A copy of the planner's and translator's drafting state (`_DRAFTING_PREFIXES`)."""
    return {
        key: json.loads(json.dumps(value))
        for key, value in ctx.state.to_dict().items()
        if key.startswith(_DRAFTING_PREFIXES)
    }


def _restore(ctx: Context, drafting: Mapping[str, Any]) -> None:
    """Put the drafting state back as `drafting` holds it, clearing any key written since."""
    for key in ctx.state.to_dict():
        if key.startswith(_DRAFTING_PREFIXES) and key not in drafting:
            ctx.state[key] = None
    for key, value in drafting.items():
        ctx.state[key] = value


async def apply_stored_choice(
    ctx: Context,
    *,
    choice_id: str,
    option_id: str,
    surface: ToolSurface,
    fetch: Callable[..., Awaitable[dict[str, Any]]],
    record_step: Callable[..., None] | None,
) -> dict[str, str]:
    """Apply a chip: move choice `choice_id` of the conversation's newest
    reading holding it to `option_id`, fetch what Search compiles, and return
    the hand-back the agent answers from.

    The reading's base identity is posted with its entries and the switch
    (`switch_entries`) through `surface` (`_resubmit`), the reading Search
    compiles is stored as the plan-card switch stores it, and its spec_id is
    fetched with the orchestrator's `fetch` tool, with the arguments the
    answer's own fetch of the reading was called with (`FetchReading.args`:
    its places, period and reading of the family), so the chip changes the
    chosen option alone. Its frames, served scope and step reach the turn as
    any fetch's do. The chip applied is
    recorded under `ORCHESTRATOR_APPLIED_KEY`, with the option it moved from
    and the spec_id Search compiled.

    A chip that cannot be applied, because no reading holds the choice, the
    option is not one Search can serve, or Search refuses or fails the
    resubmit, fetches nothing: the hand-back carries the note a plan-card
    switch shows, and the `APPLY_CHOICE_STEP_TOOL` step fails with it under
    `SCOPE_CHOICE_NOT_APPLIED`.

    Returns:
        `{"status", "text"}` for the `orchestrate` node.
    """
    args = {"choice_id": choice_id, "option_id": option_id}
    reading, choice = stored_choice(conversation_readings(ctx.state), choice_id) or (None, None)
    option = next((o for o in choice.options if o.option_id == option_id), None) if choice else None
    label = option.label if option is not None else option_id
    try:
        if reading is None or choice is None:
            raise SwitchFailedError("That choice is no longer on this answer, so nothing changed.")
        if (
            option is None
            or option_id == choice.active
            or option.availability.status == "unavailable"
        ):
            raise SwitchFailedError(
                f"Search does not offer {label} for this source, so nothing changed."
            )
        assert reading.scope is not None and reading.spec_id is not None
        pin = reading.spec_id
        switched = await _resubmit(
            surface,
            pin=pin,
            reading=reading,
            scope=reading.scope,
            entries=[
                entry.model_dump(mode="json", exclude_none=True)
                for entry in switch_entries(reading.scope, choice=choice, option_id=option_id)
            ],
            block=(ctx.state.get(ORCHESTRATOR_SPECS_KEY) or {}).get(pin) or earlier_block(pin),
            label=label,
        )
    except SwitchFailedError as exc:
        note = str(exc)
        logger.info("scope choice not applied: choice_id=%s option_id=%s", choice_id, option_id)
        _apply_step(
            record_step,
            args=args,
            result={
                "outcome": "refusal",
                "refusal": {
                    "code": SCOPE_CHOICE_NOT_APPLIED,
                    "retry_mode": "do_not_retry",
                    "fix": None,
                    "message": note,
                },
            },
        )
        return {
            "status": f"Counting {label}",
            "text": (
                f"The user asked to count {label}, and it was not applied: {note} Tell the "
                "user so in a sentence. The earlier answer stands."
            ),
        }
    assert reading.scope is not None
    store_search(
        ctx.state,
        _reading_digest(switched, axis=choice.axis),
        asked=[reading.asked_roles],
        held_open=[reading.open_roles],
        share_sides=[reading.share_over],
    )
    ctx.state[ORCHESTRATOR_APPLIED_KEY] = {
        **args,
        "from_option_id": choice.active,
        "spec_id": switched.spec_id,
    }
    logger.info(
        "scope choice applied: choice_id=%s option_id=%s from=%s",
        choice_id,
        option_id,
        choice.active,
    )
    _apply_step(
        record_step,
        args=args,
        result={"outcome": "ok", "frame_refs": [], "summary": f"counting {label}"},
    )
    shaped = reading.fetch.args if reading.fetch is not None else {}
    fetched = await fetch(spec_id=switched.spec_id, tool_context=ctx, **shaped)
    held = choice.active_option
    moved = [
        f"Search also moved {entry['from_label']} to {entry['to_label']}: {entry['reason']}"
        for entry in moved_choices(reading.scope, switched.after, switched=choice.axis)
    ]
    lead = (
        f'The user chose to count "{reading.text or reading.sentence or pin}" as {label}'
        f"{f' in place of {held.label}' if held is not None else ''}. Search counted it again "
        f"as {switched.spec_id}, and its data was fetched for you below. Answer the user's last "
        "question again from these figures and say what the change did to them. Search and "
        "plan are not available this turn."
    )
    return {
        "status": f"Counting {label}",
        "text": "\n".join([lead, *moved, json.dumps(fetched)]),
    }


def _apply_step(
    record_step: Callable[..., None] | None, *, args: dict[str, Any], result: dict[str, Any]
) -> None:
    """Record the `APPLY_CHOICE_STEP_TOOL` step, when the executor records steps."""
    if record_step is not None:
        record_step(
            tool=APPLY_CHOICE_STEP_TOOL, args=args, result=result, step_id=APPLY_CHOICE_STEP_TOOL
        )
