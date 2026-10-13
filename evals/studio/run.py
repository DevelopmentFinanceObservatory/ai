"""harness-studio-v1 runner: drives the studio surface and scores the cases.

Two layers, one case file (`harness-studio-orchestrator-v1.cases.json`):

- CONTRACT (mechanical, always on): every frame validates against
  `harness.wire`, the stream invariants in `checks.py` hold, and each
  turn's `expect` block matches (terminal frame, component/block counts,
  error codes, elicitation affordances, ...). No model involved.
- AGENT (rubric-judged, live mode with --judge): a judge model scores each
  case's `rubrics` against a digest of the conversation. Needs
  `GOOGLE_API_KEY`; see `test_config.json` for the judge settings.

Modes:
  --mode scripted    the studio in-process, its models scripted
                     (`orchestrator_scripted`): every model request is recorded
                     and the case's `checks` read them. No server, key or network.
  --mode live        against a running server (--url), real agent + data.

    uv run --package harness python evals/studio/run.py --mode scripted
    uv run --env-file .env --package harness python evals/studio/run.py \
        --mode live --url http://localhost:8080 --judge

Exit 0 iff every non-xfail case passes its layer(s). An xfail case failing
is reported XFAIL (expected, known gap); an xfail case PASSING is reported
XPASS loudly — the gap closed, retire the marker.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager, nullcontext
from pathlib import Path
from typing import Any

import checks
import httpx2

_HERE = Path(__file__).resolve().parent
_CASES_PATH = _HERE / "harness-studio-orchestrator-v1.cases.json"
_MODES = ("scripted", "live")
_CONFIG_PATH = _HERE / "test_config.json"
_RESULTS_DIR = _HERE / ".results"

_MAX_RESUMES = 6


def _bearer(token: str | None) -> dict[str, str]:
    """The bearer header for a token, or a placeholder that a server without
    STUDIO_API_TOKEN accepts."""
    return {"Authorization": "Bearer " + (token or "studio-eval")}


# ---------------------------------------------------------------------------
# Driving one conversation
# ---------------------------------------------------------------------------


class CaseError(Exception):
    """A mechanical expectation failed; the message is the finding."""


class JudgeError(Exception):
    """The judge model stayed unavailable through its retries; the case is unscored."""


def _request(kind: str, conversation_id: str, **members: Any) -> dict[str, Any]:
    body = {"kind": kind, "schema_version": 1, "conversation_id": conversation_id}
    body.update({key: value for key, value in members.items() if value is not None})
    return body


async def _stream_frames(
    client: httpx2.AsyncClient, path: str, body: dict[str, Any]
) -> list[dict[str, Any]]:
    async with client.stream("POST", path, json=body) as response:
        if response.status_code != 200:
            await response.aread()
            raise CaseError(f"{path} -> HTTP {response.status_code}: {response.text[:300]}")
        return [
            json.loads(line[len("data: ") :])
            async for line in response.aiter_lines()
            if line.startswith("data: ")
        ]


async def _run_message_turn(
    client: httpx2.AsyncClient, conversation_id: str, turn: dict[str, Any], produced: list[dict]
) -> tuple[list[dict[str, Any]], list[str], list[str], dict[str, dict[str, Any]]]:
    """One turn.request plus every interrupt resume it needs, per the turn's
    scripted decisions. Returns the turn's frames across all segments, the
    contract violations only a live client can see (each SSE response must
    end on exactly one terminal frame, `checks.check_segment` — a flat
    transcript has lost these boundaries), and the expectation misses the
    resume protocol itself produces: once the turn's plan decisions are
    spent, a further plan card is a re-proposal the case did not ask for, so
    the runner declines it and the turn ends there. The last member maps each
    elicitation's `interrupt_id` to the `{decision, payload}` the runner sent
    for it: the wire carries the question but not the reply, and the judge
    needs both."""
    component_ref = None
    if "component_ref" in turn:
        from_turn = turn["component_ref"]["from_turn"]
        ref = produced[from_turn]
        if "component_id" not in ref:
            raise CaseError(f"turn {from_turn + 1} produced no component for a later component_ref")
        component_ref = {
            "component_id": ref["component_id"],
            "revision": ref["revision"],
            "intent": turn["component_ref"]["intent"],
        }
    violations: list[str] = []
    missed: list[str] = []
    segment = await _stream_frames(
        client,
        "/api/studio/turn",
        _request(
            "turn.request",
            conversation_id,
            message=turn["message"],
            component_ref=component_ref,
            auto_add=turn.get("auto_add"),
            seed_spec=turn.get("seed_spec"),
        ),
    )
    violations += checks.check_segment(segment, label="turn stream")
    frames = list(segment)

    # The scripted plan decisions this turn may spend, in order: its own
    # decision, then one approve for the re-proposal an amend triggers.
    plan_decisions = [turn.get("plan_decision", "approve")]
    if plan_decisions[0] == "amend":
        plan_decisions.append("approve")
    # One re-proposal the case did not ask for is approved (the harness allows
    # itself one execution-triggered replan); a second is declined.
    spare_approve = True

    answers: dict[str, dict[str, Any]] = {}
    resumes = 0
    while frames and frames[-1]["event"] in ("plan.proposed", "elicitation.request"):
        pause = frames[-1]
        if resumes == _MAX_RESUMES:
            violations.append(f"turn still paused after {_MAX_RESUMES} resumes; giving up")
            break
        if pause["event"] == "plan.proposed":
            if not plan_decisions and spare_approve:
                spare_approve = False
                plan_decisions.append("approve")
            if not plan_decisions:
                missed.append(
                    f"plan.proposed v{pause['version']} after the turn's plan decisions "
                    "and its one spare approval were spent: a further re-proposal"
                )
                break
            decision = plan_decisions.pop(0)
            payload = turn.get("amend_note") if decision == "amend" else None
        else:
            decision = turn.get("elicitation_decision", "answer")
            payload = None
            if decision == "answer":
                payload = turn.get("elicitation_text") or (pause.get("options") or [{}])[0].get(
                    "id"
                )
                if payload is None:
                    missed.append(
                        "elicitation.request offered no options and the turn names no "
                        "elicitation_text to answer it with"
                    )
                    break
            answers[pause["interrupt_id"]] = {"decision": decision, "payload": payload}
        resumes += 1
        segment = await _stream_frames(
            client,
            "/api/studio/interrupt",
            _request(
                "interrupt.response",
                conversation_id,
                interrupt_id=pause["interrupt_id"],
                decision=decision,
                payload=payload,
            ),
        )
        violations += checks.check_segment(segment, label=f"resume stream {resumes}")
        frames += segment
    return frames, violations, missed, answers


async def _run_mutate_turn(
    client: httpx2.AsyncClient, conversation_id: str, turn: dict[str, Any], produced: list[dict]
) -> dict[str, Any]:
    """A composition.mutate request/response cycle (no frames). Returns
    {status, manifest}. `component_id: {"from_turn": N}` resolves to the
    Nth message turn's newest component."""
    operations = []
    for template in turn["operations"]:
        operation = dict(template)
        if isinstance(operation.get("component_id"), dict):
            from_turn = operation["component_id"]["from_turn"]
            operation["component_id"] = produced[from_turn]["component_id"]
        operations.append(operation)
    response = await client.post(
        "/api/studio/composition",
        json=_request(
            "composition.mutate",
            conversation_id,
            operations=operations,
            expected_revision=turn.get("expected_revision"),
        ),
    )
    manifest = response.json() if response.status_code == 200 else None
    body: dict[str, Any] | None = None
    if response.status_code != 200:
        try:
            body = response.json()
        except json.JSONDecodeError:
            body = {"raw": response.text[:300]}
    return {"status": response.status_code, "manifest": manifest, "error_body": body}


async def run_case(
    client: httpx2.AsyncClient,
    case: dict[str, Any],
    *,
    conversation_id: str | None = None,
    on_turn: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    """Runs one case's turns. Violations come back in two buckets, because
    they mean different things to an xfail marker: `contract` (wire
    validity, stream invariants, segment discipline — never excusable) and
    `expectation` (the case's expect blocks — what an xfail expects to
    miss). `on_turn` is told each message turn's number as it starts."""
    conversation_id = conversation_id or f"eval_{uuid.uuid4().hex[:12]}"
    all_frames: list[dict[str, Any]] = []
    turn_frames: list[list[dict[str, Any]]] = []
    produced: list[dict[str, Any]] = []  # newest component per message turn
    contract: list[str] = []
    expectation: list[str] = []
    declined: list[int] = []  # turns the runner left paused by declining a re-proposal
    answers: dict[str, dict[str, Any]] = {}  # elicitation interrupt_id -> the reply sent

    for index, turn in enumerate(case["turns"], start=1):
        if "operations" in turn:
            result = await _run_mutate_turn(client, conversation_id, turn, produced)
            expectation += _check_mutation(turn, result, label=f"turn {index}")
            continue
        if on_turn is not None:
            on_turn(index)
        frames, segment_violations, protocol_misses, turn_answers = await _run_message_turn(
            client, conversation_id, turn, produced
        )
        answers |= turn_answers
        contract += segment_violations
        expectation += [f"turn {index}: {miss}" for miss in protocol_misses]
        if protocol_misses:
            declined.append(index)
        all_frames += frames
        turn_frames.append(frames)
        components = [f for f in frames if f["event"] == "component.created"]
        produced.append(components[-1] if components else {})
        expectation += _check_turn_expectations(turn, frames, label=f"turn {index}")

    # A turn the runner declined to resume ends on its pause by the runner's
    # choice; the transcript check's "incomplete turn" for it is not the
    # server's doing.
    contract += [
        violation
        for violation in checks.check_transcript(all_frames, complete_conversation=True)
        if not any(violation.startswith(f"turn {index}: ends on ") for index in declined)
    ]
    return {
        "frames": all_frames,
        "turn_frames": turn_frames,
        "answers": answers,
        "contract": contract,
        "expectation": expectation,
    }


# ---------------------------------------------------------------------------
# Mechanical expectations
# ---------------------------------------------------------------------------

# The `expect` keys the checkers below read, with the sub-keys of the nested
# ones. A key outside these tables is a case-file error: the checkers would
# skip it and the case would pass on nothing.
TURN_EXPECT_KEYS: dict[str, frozenset[str] | None] = {
    "terminal": None,
    "error_code": None,
    "min_components": None,
    "max_components": None,
    "component_types": None,
    "revised": frozenset({"changed_contains"}),
    "fact_summary_contains": None,
    "message_contains": None,
    "message_excludes": None,
    "blocks": None,
    "activity_steps": None,
    "step_tools": None,
    "elicitation": frozenset({"min_options", "allow_free_text", "skippable"}),
    "plan": frozenset({"min_steps", "min_version", "max_version", "operations_contain"}),
    "no_plan": None,
}
# The `expect` keys whose value is a list of strings the turn's closing text is
# matched against, case-insensitively.
_TEXT_LIST_EXPECT_KEYS = ("message_contains", "message_excludes")
# Every `expect` key that takes a non-empty list of non-empty strings.
_STRING_LIST_EXPECT_KEYS = (*_TEXT_LIST_EXPECT_KEYS, "step_tools")
MUTATION_EXPECT_KEYS: dict[str, frozenset[str] | None] = {
    "status": None,
    "blocks": None,
    "error": None,
}


def _text_needles(expect: dict[str, Any], key: str) -> list[str]:
    """The strings of a list-valued text expectation; `check_expect_keys`
    reports a malformed one, so it yields nothing here."""
    needles = expect.get(key)
    return [n for n in needles if isinstance(n, str)] if isinstance(needles, list) else []


def check_expect_keys(turn: dict[str, Any], *, label: str) -> list[str]:
    """Every key in the turn's `expect` block names a check the runner runs."""
    known = MUTATION_EXPECT_KEYS if "operations" in turn else TURN_EXPECT_KEYS
    violations = []
    for key, rules in turn.get("expect", {}).items():
        if key not in known:
            violations.append(f"{label}: expect key {key!r} is not one the runner checks")
        elif known[key] is not None and isinstance(rules, dict):
            violations.extend(
                f"{label}: expect.{key} key {sub!r} is not one the runner checks"
                for sub in rules
                if sub not in known[key]
            )
    expect = turn.get("expect", {})
    lists = {key: expect[key] for key in _STRING_LIST_EXPECT_KEYS if key in expect}
    if isinstance(expect.get("plan"), dict) and "operations_contain" in expect["plan"]:
        lists["plan.operations_contain"] = expect["plan"]["operations_contain"]
    violations.extend(
        f"{label}: expect.{key} must be a non-empty list of non-empty strings"
        for key, needles in lists.items()
        if not (
            isinstance(needles, list)
            and needles
            and all(isinstance(needle, str) and needle for needle in needles)
        )
    )
    return violations


def _check_turn_expectations(
    turn: dict[str, Any], frames: list[dict[str, Any]], *, label: str
) -> list[str]:
    expect = turn.get("expect", {})
    violations = check_expect_keys(turn, label=label)
    last = frames[-1] if frames else {}

    if "terminal" in expect and last.get("event") != expect["terminal"]:
        violations.append(f"{label}: ended on {last.get('event')!r}, expected {expect['terminal']}")
    if "error_code" in expect:
        code = last.get("error", {}).get("error") if last.get("event") == "error" else None
        if code != expect["error_code"]:
            violations.append(f"{label}: error code {code!r}, expected {expect['error_code']!r}")
        if last.get("event") == "error" and not last.get("error", {}).get("detail"):
            violations.append(f"{label}: error frame carries no detail text")

    created = [f for f in frames if f["event"] == "component.created"]
    revised = [f for f in frames if f["event"] == "component.revised"]
    if "min_components" in expect and len(created) < expect["min_components"]:
        violations.append(
            f"{label}: {len(created)} component(s), expected >= {expect['min_components']}"
        )
    if "max_components" in expect and len(created) > expect["max_components"]:
        violations.append(
            f"{label}: {len(created)} component(s), expected <= {expect['max_components']}"
        )
    if "component_types" in expect:
        types = [f["type"] for f in created]
        violations.extend(
            f"{label}: no {wanted!r} among created types {types}"
            for wanted in expect["component_types"]
            if wanted not in types
        )
    if expect.get("revised"):
        if not revised:
            violations.append(f"{label}: expected a component.revised frame, saw none")
        else:
            violations.extend(
                f"{label}: component.revised.changed {revised[-1]['changed']} missing {field!r}"
                for field in expect["revised"].get("changed_contains", [])
                if field not in revised[-1]["changed"]
            )
    if "fact_summary_contains" in expect:
        blob = json.dumps([f["fact_summary"] for f in created + revised]).lower()
        violations.extend(
            f"{label}: no created/revised fact_summary mentions {needle!r}"
            for needle in expect["fact_summary_contains"]
            if needle.lower() not in blob
        )

    if "message_contains" in expect or "message_excludes" in expect:
        texts = [f["text"] for f in frames if f["event"] == "message.complete"]
        message = "\n".join(texts).lower()
        if not texts:
            violations.append(f"{label}: expected a message.complete frame, saw none")
        violations.extend(
            f"{label}: the closing message does not mention {needle!r}"
            for needle in _text_needles(expect, "message_contains")
            if needle.lower() not in message
        )
        violations.extend(
            f"{label}: the closing message mentions {needle!r}"
            for needle in _text_needles(expect, "message_excludes")
            if needle.lower() in message
        )

    dones = [f for f in frames if f["event"] == "done"]
    if "blocks" in expect:
        manifest = dones[-1].get("composition") if dones else None
        block_count = len(manifest["blocks"]) if manifest else 0
        if block_count != expect["blocks"]:
            violations.append(
                f"{label}: {block_count} composition block(s), expected {expect['blocks']}"
            )
    if "activity_steps" in expect:
        # Steps that ran a data tool: a reply with no search makes none. The
        # agent's thought and status lines are steps too, but carry no
        # `tool`, and its `explain_dfo` reads the dfo-faq skill; neither says
        # anything about the route.
        steps = {
            f["step_id"]
            for f in frames
            if f["event"] == "activity.step" and f.get("tool") not in (None, "explain_dfo")
        }
        if len(steps) != expect["activity_steps"]:
            violations.append(
                f"{label}: {len(steps)} tool step(s), expected {expect['activity_steps']}"
            )

    if "step_tools" in expect:
        ran = {
            f.get("tool")
            for f in frames
            if f["event"] == "activity.step" and f.get("status") == "complete"
        } - {None}
        violations.extend(
            f"{label}: no activity step completed {tool!r}; the turn's steps completed {sorted(ran)}"
            for tool in expect["step_tools"]
            if tool not in ran
        )

    if "elicitation" in expect:
        pauses = [f for f in frames if f["event"] == "elicitation.request"]
        if not pauses:
            violations.append(f"{label}: expected an elicitation.request, saw none")
        else:
            pause, rules = pauses[0], expect["elicitation"]
            if rules.get("min_options") and len(pause.get("options") or []) < rules["min_options"]:
                violations.append(
                    f"{label}: elicitation has {len(pause.get('options') or [])} "
                    f"options, expected >= {rules['min_options']}"
                )
            violations.extend(
                f"{label}: elicitation {flag}={pause.get(flag)}, expected {rules[flag]}"
                for flag in ("allow_free_text", "skippable")
                if flag in rules and bool(pause.get(flag)) is not rules[flag]
            )
    if expect.get("no_plan"):
        proposed = sum(1 for f in frames if f["event"] == "plan.proposed")
        if proposed:
            violations.append(f"{label}: {proposed} plan.proposed frame(s), expected none")
    if "plan" in expect:
        plans = [f for f in frames if f["event"] == "plan.proposed"]
        if not plans:
            violations.append(f"{label}: expected a plan.proposed, saw none")
        else:
            rules = expect["plan"]
            if rules.get("min_steps") and len(plans[0]["steps"]) < rules["min_steps"]:
                violations.append(
                    f"{label}: plan has {len(plans[0]['steps'])} steps, expected >= {rules['min_steps']}"
                )
            violations.extend(_operations_missing(plans, rules, label=label))
            top_version = max(p["version"] for p in plans)
            if rules.get("min_version") and top_version < rules["min_version"]:
                violations.append(
                    f"{label}: plan stayed at version {top_version}, expected >= {rules['min_version']}"
                )
            if rules.get("max_version") and top_version > rules["max_version"]:
                violations.append(
                    f"{label}: plan reached version {top_version}, expected <= {rules['max_version']}"
                )
    return violations


def _operations_missing(
    plans: list[dict[str, Any]], rules: dict[str, Any], *, label: str
) -> list[str]:
    """Each `plan.operations_contain` string the turn's highest-version plan
    never says of an ask it answers, matched case-insensitively.

    The words are those of the operations an undeclined ask lists in
    `answered_by` (`_party_words`). Step labels, requirement ids and every
    other sentence are left out, because they repeat the planner's own words
    for what was asked, whatever the plan keeps.
    """
    needles = rules.get("operations_contain") or []
    last = max(plans, key=lambda plan: plan["version"])
    operations = {op.get("id"): op for op in last.get("operations") or []}
    answering = [
        operations.get(op_id) or {}
        for ask in last.get("asks") or []
        if ask.get("declined") is None
        for op_id in ask.get("answered_by") or []
    ]
    said = " ".join(_party_words(op) for op in answering).lower()
    return [
        f"{label}: the operations answering the plan's asks never say {needle!r}"
        for needle in needles
        if needle.lower() not in said
    ]


def _party_words(operation: dict[str, Any]) -> str:
    """The words of `operation`'s sentence that name the parties it keeps.

    A retrieve's sentence is Search's account of what its source holds,
    and a rank's names its population after its last colon
    (`harness.plan_display`). Any other sentence, and a rank's opening,
    restates the planner's `measure` and names no party.
    """
    sentence = operation.get("sentence") or ""
    if operation.get("kind") == "retrieve":
        return sentence
    if operation.get("kind") == "rank":
        return sentence.rpartition(": ")[2]
    return ""


def _check_mutation(turn: dict[str, Any], result: dict[str, Any], *, label: str) -> list[str]:
    expect = turn.get("expect", {})
    violations = check_expect_keys(turn, label=label)
    if "status" in expect and result["status"] != expect["status"]:
        violations.append(
            f"{label}: composition.mutate -> HTTP {result['status']}, expected {expect['status']}"
        )
    if "blocks" in expect:
        if result["manifest"] is None:
            violations.append(
                f"{label}: expected {expect['blocks']} manifest block(s) but the mutate "
                f"returned HTTP {result['status']} and no manifest"
            )
        else:
            block_count = len(result["manifest"].get("blocks", []))
            if block_count != expect["blocks"]:
                violations.append(
                    f"{label}: manifest has {block_count} block(s), expected {expect['blocks']}"
                )
    if "error" in expect:
        code = (result.get("error_body") or {}).get("error")
        if code != expect["error"]:
            violations.append(
                f"{label}: refusal body carries error {code!r}, expected {expect['error']!r}"
            )
    return violations


# ---------------------------------------------------------------------------
# The judge (agent layer, live mode)
# ---------------------------------------------------------------------------

_JUDGE_INSTRUCTION = """\
You judge one recorded conversation with a data-analysis agent against a
list of properties. For each property answer yes or no with a one-sentence
rationale grounded in the evidence.

Rules:
- Answer yes ONLY when the evidence affirmatively demonstrates the
  property. Absence of evidence is a no.
- A property is never "inapplicable" unless its own text says when it may
  be skipped; the user's phrasing never waives a property.
- Judge each property independently; one property failing does not excuse
  or fail another.
- Within a turn, each message, plan, question (`elicitations`), step and component carries
  `at`, its position in the turn's event stream. The lists are grouped by
  kind, so compare `at` values to tell what happened first.
"""


def _elicitation_digest(frame: dict[str, Any], reply: dict[str, Any] | None) -> dict[str, Any]:
    """A question the agent put to the user, its options by label, and what
    the runner sent back (`None` when the turn ended unanswered). An option id
    in the reply reads as that option's label; free text reads as itself."""
    options = frame.get("options") or []
    labels = {option["id"]: option["label"] for option in options}
    if reply is None:
        answer = None
    elif reply["decision"] == "answer":
        answer = labels.get(reply["payload"], reply["payload"])
    else:
        answer = reply["decision"]
    return {
        "question": frame["question"],
        "options": [
            {"label": option["label"], "description": option.get("description")}
            for option in options
        ],
        "answer": answer,
    }


def _turn_digest(
    frames: list[dict[str, Any]], answers: dict[str, dict[str, Any]] | None = None
) -> dict[str, Any]:
    """The judge's view of one turn: texts, plan, steps, facts, the questions
    the agent asked with the replies sent, and the canvas as it stood when the
    turn ended, with no payloads. Without the canvas the judge cannot tell a
    replaced card from a duplicated one, since the stream shows two
    component.created events either way and only the manifest shows one block.
    Without the questions it cannot tell an agent that asked from one that did
    not, since a pause leaves only the label "Asking you a question" in the
    steps. Every message, plan, question, step and component entry carries
    `at`, the frame's position in the turn's stream: the lists group entries
    by kind, and `at` orders them across lists, so the judge can tell a
    figure drawn before a question from one drawn after it."""
    answers = answers or {}
    digest: dict[str, Any] = {
        "messages": [],
        "plans": [],
        "elicitations": [],
        "steps": [],
        "components": [],
    }
    for at, frame in enumerate(frames):
        event = frame["event"]
        manifest = frame.get("manifest") if event == "composition.updated" else None
        manifest = frame.get("composition") if event == "done" else manifest
        if manifest:
            digest["canvas_after_turn"] = [
                {
                    "component_id": block["component_id"],
                    "type": block["type"],
                    "title": block["title"],
                }
                for block in manifest["blocks"]
            ]
        if event == "message.complete":
            digest["messages"].append({"at": at, "text": frame["text"]})
        elif event == "plan.proposed":
            digest["plans"].append(
                {
                    "at": at,
                    "version": frame["version"],
                    "steps": [
                        {"label": s["label"], "description": s.get("description")}
                        for s in frame["steps"]
                    ],
                    "decisions": [
                        {"question": d["question"], "chosen": d["chosen"], "reason": d["reason"]}
                        for d in frame.get("decisions") or []
                    ],
                }
            )
        elif event == "elicitation.request":
            digest["elicitations"].append(
                {"at": at, **_elicitation_digest(frame, answers.get(frame["interrupt_id"]))}
            )
        elif event == "activity.step":
            digest["steps"].append(
                {
                    "at": at,
                    "label": frame["label"],
                    "status": frame["status"],
                    "detail": frame.get("detail"),
                }
            )
        elif event in ("component.created", "component.revised"):
            digest["components"].append(
                {
                    "at": at,
                    "event": event,
                    "type": frame.get("type"),
                    "fact_summary": frame["fact_summary"],
                    "changed": frame.get("changed"),
                }
            )
        elif event == "error":
            digest["error"] = frame["error"]
    return digest


async def judge_case(
    case: dict[str, Any],
    turn_frames: list[list[dict[str, Any]]],
    config: dict[str, Any],
    *,
    answers: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    # Lazy on purpose: judge-less runs (the whole contract layer) must not
    # pay the model SDK's import.
    from google.genai import Client, errors, types  # noqa: PLC0415

    # The judge model answers 503 or 429 under load; the SDK retries its
    # transient codes with backoff.
    client = Client(
        http_options=types.HttpOptions(
            retry_options=types.HttpRetryOptions(attempts=4, initial_delay=2.0, max_delay=30.0)
        )
    )
    message_turns = [turn for turn in case["turns"] if "message" in turn]
    evidence = {
        "turns": [
            {"user": turn["message"], **_turn_digest(frames, answers)}
            for turn, frames in zip(message_turns, turn_frames, strict=True)
        ]
    }
    schema = types.Schema(
        type=types.Type.ARRAY,
        items=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "rubric_id": types.Schema(type=types.Type.STRING),
                "verdict": types.Schema(type=types.Type.STRING, enum=["yes", "no"]),
                "rationale": types.Schema(type=types.Type.STRING),
            },
            required=["rubric_id", "verdict", "rationale"],
        ),
    )
    content = json.dumps(
        {
            "properties": [
                {"rubric_id": r["id"], "property": r["property"]} for r in case["rubrics"]
            ],
            "conversation": evidence,
        }
    )
    samples = []
    for _ in range(config.get("num_samples", 1)):
        try:
            response = await client.aio.models.generate_content(
                model=config["judge_model"],
                contents=content,
                config=types.GenerateContentConfig(
                    system_instruction=_JUDGE_INSTRUCTION,
                    response_schema=schema,
                    response_mime_type="application/json",
                    temperature=0.0,
                ),
            )
        except errors.APIError as error:
            if not isinstance(error, errors.ServerError) and error.code != 429:
                raise
            raise JudgeError(str(error)) from error
        samples.append(json.loads(response.text or "[]"))
    # Per rubric: fraction of samples answering yes, plus one rationale.
    scored = []
    for rubric in case["rubrics"]:
        verdicts = [
            entry
            for sample in samples
            for entry in sample
            if entry.get("rubric_id") == rubric["id"]
        ]
        yes = sum(1 for v in verdicts if v.get("verdict") == "yes")
        scored.append(
            {
                "rubric_id": rubric["id"],
                "score": yes / max(len(verdicts), 1),
                "rationales": [v.get("rationale") for v in verdicts],
            }
        )
    return scored


# ---------------------------------------------------------------------------
# Harness plumbing
# ---------------------------------------------------------------------------


async def _run_scripted_case(case: dict[str, Any]) -> dict[str, Any]:
    """One case on its own scripted studio (its scripts are
    consumed as it runs), then the case's `checks` over what the models were
    shown. A model script that ran dry is a contract failure, since the case
    drove less than it says."""
    import orchestrator_scripted  # noqa: PLC0415 -- the harness import chain is slow

    async with orchestrator_scripted.orchestrator_client(case["scenario"]) as (client, log):
        outcome = await run_case(client, case, on_turn=log.mark_turn)
    outcome["contract"] += [
        f"the {agent} model's script ran out: the case drove fewer model calls than it needs"
        for agent in sorted(set(log.exhausted))
    ]
    evidence = checks.Evidence(turn_frames=outcome["turn_frames"], log=log.entries)
    outcome["expectation"] += checks.run_case_checks(case.get("checks", []), evidence)
    return outcome


@asynccontextmanager
async def _live_client(url: str) -> AsyncIterator[httpx2.AsyncClient]:
    # A remote server; the `eval-studio` recipe loads `.env` into the
    # environment before this runs, so the same STUDIO_API_TOKEN the server
    # enforces is here to read.
    headers = _bearer(os.environ.get("STUDIO_API_TOKEN"))
    async with httpx2.AsyncClient(
        base_url=url, timeout=httpx2.Timeout(300.0), headers=headers
    ) as client:
        yield client


def load_cases(mode: str, only: list[str] | None = None) -> list[dict[str, Any]]:
    """The cases that run in `mode`. A case's `scenario` may name an entry of
    the file's `scenarios`, which is then resolved into the case."""
    suite = json.loads(_CASES_PATH.read_text())
    scenarios = suite.get("scenarios", {})
    cases = [
        {**case, "scenario": scenarios[case["scenario"]]}
        if isinstance(case.get("scenario"), str)
        else case
        for case in suite["cases"]
        if mode in case["modes"]
    ]
    if only:
        cases = [case for case in cases if case["id"] in only]
    return cases


async def run_suite(
    *, mode: str, url: str, judge: bool, only: list[str] | None = None
) -> tuple[list[dict[str, Any]], bool]:
    config = json.loads(_CONFIG_PATH.read_text())
    cases = load_cases(mode, only)
    # A scripted case builds its own studio, so it needs no shared client.
    factory = nullcontext if mode == "scripted" else lambda: _live_client(url)
    results = []
    suite_passed = True
    async with factory() as client:
        for case in cases:
            started = time.monotonic()
            record: dict[str, Any] = {
                "id": case["id"],
                "xfail": case.get("xfail"),
                "contract_violations": [],
                "expectation_violations": [],
            }
            try:
                if mode == "scripted":
                    outcome = await _run_scripted_case(case)
                else:
                    assert client is not None  # only the scripted mode has none
                    outcome = await run_case(client, case)
                    outcome["expectation"] += _frame_checks(case, outcome)
                record["contract_violations"] = outcome["contract"]
                record["expectation_violations"] = outcome["expectation"]
                if judge and case.get("rubrics"):
                    try:
                        record["rubric_scores"] = await judge_case(
                            case,
                            outcome["turn_frames"],
                            config,
                            answers=outcome.get("answers"),
                        )
                    except JudgeError as error:
                        record["judge_error"] = str(error)
                    else:
                        threshold = config.get("threshold", 1.0)
                        record["expectation_violations"] += [
                            f"rubric {entry['rubric_id']}: score {entry['score']:.2f} "
                            f"< {threshold} — {entry['rationales'][:1]}"
                            for entry in record["rubric_scores"]
                            if entry["score"] < threshold
                        ]
            except CaseError as error:
                # A non-200 on a stream route is a surface failure; xfail never
                # excuses it.
                record["contract_violations"] = [str(error)]
            except httpx2.HTTPError as error:
                record["contract_violations"] = [f"transport failure: {error!r}"]
            record["duration_s"] = round(time.monotonic() - started, 1)

            # An xfail marker excuses missed EXPECTATIONS only. Contract
            # violations — invalid frames, broken stream discipline — fail
            # the case regardless: the contract layer has no known gaps.
            contract_broken = bool(record["contract_violations"])
            missed = bool(record["expectation_violations"])
            if contract_broken:
                record["status"] = "FAIL"
                suite_passed = False
            elif "judge_error" in record:
                # Unscored is not passed: the suite cannot claim the rubrics held.
                record["status"] = "JUDGE_ERROR"
                suite_passed = False
            elif case.get("xfail"):
                record["status"] = "XFAIL" if missed else "XPASS"
            else:
                record["status"] = "FAIL" if missed else "PASS"
                suite_passed = suite_passed and not missed
            results.append(record)
            _print_record(record)
    return results, suite_passed


def _frame_checks(case: dict[str, Any], outcome: dict[str, Any]) -> list[str]:
    """The case's `checks` that read frames only, for a run with no request log."""
    wanted = case.get("checks", [])
    evidence = checks.Evidence(turn_frames=outcome["turn_frames"])
    return checks.run_case_checks(wanted, evidence)


def _print_record(record: dict[str, Any]) -> None:
    marker = {
        "PASS": "ok ",
        "FAIL": "FAIL",
        "XFAIL": "xf ",
        "XPASS": "XPASS?!",
        "JUDGE_ERROR": "JERR",
    }[record["status"]]
    line = f"[{marker}] {record['id']} ({record['duration_s']}s)"
    if record["status"] == "XFAIL":
        line += f"  (expected: {record['xfail']})"
    if record["status"] == "XPASS":
        line += f"  (marked xfail '{record['xfail']}' but PASSED — gap closed? retire the marker)"
    print(line)
    if record["status"] == "JUDGE_ERROR":
        print(f"       - [judge] {record['judge_error']}")
    if record["status"] in ("FAIL", "JUDGE_ERROR"):
        for violation in record["contract_violations"]:
            print(f"       - [contract] {violation}")
        for violation in record["expectation_violations"]:
            print(f"       - {violation}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=_MODES, required=True)
    parser.add_argument("--url", default="http://localhost:8080")
    parser.add_argument("--judge", action="store_true", help="score rubrics with the judge model")
    parser.add_argument("--case", action="append", help="run only these case ids", default=None)
    args = parser.parse_args()

    if args.mode == "live" and args.judge is False:
        print("note: live mode without --judge runs the contract layer only\n")

    results, passed = asyncio.run(
        run_suite(mode=args.mode, url=args.url, judge=args.judge, only=args.case)
    )
    counts: dict[str, int] = {}
    for record in results:
        counts[record["status"]] = counts.get(record["status"], 0) + 1
    print("\n" + ", ".join(f"{v} {k}" for k, v in sorted(counts.items())) or "no cases matched")

    _RESULTS_DIR.mkdir(exist_ok=True)
    out = _RESULTS_DIR / f"{args.mode}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"results: {out}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
