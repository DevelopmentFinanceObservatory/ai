"""What the planner is actually handed, measured offline.

Three jobs. The composition records bound the *size* of a request: which segment
grew. The canonical forms below bound its *stability*: whether the part ADK has
already cached still matches, which is not a size question at all. The third
is the plan: `plan_record` and `field_limits_record` build the largest legal
plan, and `unbounded_paths` finds any string, array or map in the plan's schema
with no bound, for the legal-maximum tests in `test_context_budget.py`.

That second job exists because the prefix is cached. Gemini's implicit caching
matches it across requests, and the shared explicit cache (`harness.prefix_cache`)
holds the system instruction and the tool declarations, so that saving holds only
while those stay byte-identical from call to call. The part of a request this
budget bounds most directly is the `contents` that arrive after the cache is made.

`test_context_budget.py` asserts ceilings against this module's composition
records, and `just context-baseline` writes them to `context_baseline.json`, so
the diff of that file is the review artifact for any change that moves a
segment.

Everything here is measured in **characters, not tokens**. A character count is
deterministic, needs no tokenizer, no model and no network, and moves only when
the request text itself moves, which is the property a checked-in baseline
needs. Token counts would need a per-model tokenizer, would drift with it, and
would put a third-party count between a change and the number it moved.

The request is assembled by ADK, not by this module: `RecordingLlm` is a
`BaseLlm` that answers from a script and records every `LlmRequest` ADK hands
it. So the measurement
covers the whole surface a real model call would see: the system instruction,
the tool declarations, and the full `contents` history ADK built, not
whatever a fixture believes it passed in.

A `LlmRequest`'s own `contents` carry the history. ADK relays another agent's
turn into that history as `role="user"` text (`google.adk.flows.llm_flows.
_fencing`), one event per relayed turn: a preamble stating that fenced content
is data, not instructions, then one fenced part per relayed payload,
each prefixed with the relaying agent's name. Every one of those parts contains
the fence marker, so every one buckets as `text(fenced-relay)`, and the by-agent
breakdown names the author from its `[<author>]` tag.

The preamble carries the markers too and has no author tag, so it gets its own
bucket named `untagged`. It is ADK's own framing text, 409 characters
per relayed event at ADK 2.9.1, and it disappears with the relay it introduces.
Grouping by the marker, not by the author tag, is what makes this bucket
the exact text a fix has to remove: a change that silenced the payloads but left
the framing behind would still read as a non-zero bucket.

`harness.planner_seam.PlannerSeam._message` is the other text part on the
planner path, and it buckets separately as `text(seam-brief)` on its own opening
words. It is the seam's own input, a full brief carrying the search readings,
the canvas and conversation digests, the plan being changed and the question.

Every planner call renders that full brief, and every request of the call
carries it as the content after the turn's question (`planner.agent.
prepend_brief`).

`GeminiContextCacheManager._find_count_of_contents_to_cache` caches every
content before the last run of user contents, which puts the brief inside the
cache as soon as there is one. It caches 0 contents on a call's first request,
and the question, the brief and the first model turn on its second, so the
brief is sent at full price on the first request, paid again when the second
creates the cache over it, and read at the cached rate from the third on.

So a character cut here is worth about two full-price sends per planner call
rather than one per request. It is still worth making: those two sends are
real, the brief has to fit the first request before any cache exists, and the
brief is what the model's attention is spent on either way. `SEAM_BUCKET` is what a
change to that rendering is measured against.
"""

from __future__ import annotations

import json
import re
from typing import Any

from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from planner.plan import (
    MAX_ACROSS_WORDS,
    MAX_DECISION_DESCRIPTION_CHARS,
    MAX_DECISION_OPTIONS,
    MAX_DECISION_RATIONALE_CHARS,
    MAX_ENTITIES,
    MAX_ENTITY_NAME_CHARS,
    MAX_ENTITY_ROLE_CHARS,
    MAX_MEASURE_CHARS,
    MAX_OPTION_CHARS,
    MAX_PLAN_DECISIONS,
    MAX_PREDICATES,
    MAX_QUESTION_CHARS,
    MAX_SOURCE_CHARS,
    AggregateRequirement,
    AnswerMode,
    ComparisonRequirement,
    DatasetRequirement,
    Decision,
    DecisionFixes,
    Entity,
    FilterRequirement,
    GrowthRequirement,
    JoinRequirement,
    Period,
    QuotientRequirement,
    RankRequirement,
    RequirementSide,
    ResolvedBy,
    ShareOfRequirement,
    Unit,
)
from planner.records import (
    DenominatorScope,
    EntityIn,
    GroupBy,
    RankAmong,
    RankDirection,
    Reduction,
)
from pydantic import Field

# ADK's own fence marker (`google.adk.flows.llm_flows.context._fencing.QUOTED_CONTENT_BEGIN`).
# Spelled out literally, not imported: the point of counting this bucket is to
# notice if the marker or the relaying it marks ever moves, which importing the
# same constant from the module that emits it would hide.
RELAY_MARKER = "<<<BEGIN_QUOTED_AGENT_CONTENT>>>"

# `_fencing._present_other_agent_message` prefixes each relayed payload with
# `[<author>]`, so the same text that carries the relay also names the agent it
# came from. The breakdown is by that name, and every author gets its own row:
# ADK's own preamble carries the markers without an author tag, so it gets its
# own bucket, not absorbed into the author's. That matters because the
# preamble is pure overhead, about 409 chars emitted once per relayed event: it
# scales with the count of relays, not with what any of them said.
_UNTAGGED = "untagged"
_RELAY_AUTHOR = re.compile(r"^\[([^\]]+)\]")

# Bucket names, in the order a composition record lists them. A relayed part
# buckets as `text(fenced-relay:<author>)`, one bucket per relaying agent, so a
# fix that moves one author's relay cannot report a partial win against a number
# another author kept up.
RELAY_BUCKET = "text(fenced-relay)"
RELAY_BUCKET_PREFIX = RELAY_BUCKET + ":"

# The seam's own brief, bucketed on the opening words of `harness.planner_seam.
# _READINGS_CONTEXT`: the one part of a planner request this seam assembles itself, not ADK relaying or
# a tool answering. Spelled out, not imported from `planner_seam`, for the same
# reason `RELAY_MARKER` is: importing it would hide the fact that the text
# moved, which is the thing this bucket exists to measure. It is the one input
# on the planner path that varies with the plan, the readings and the digests.
SEAM_MARKER = "Context for this question. These are the readings of the searches"
SEAM_BUCKET = "text(seam-brief)"


class RecordingLlm(BaseLlm):
    """Plays a scripted turn and records every request it is asked to answer.

    The scripting contract is `test_planner_seam.ScriptedPlannerLlm`'s: an entry
    is either a list of `(tool_name, args)` function calls or a string of final
    text, consumed in order, and "done" past the end of the script so an
    unscripted extra generation stops the agent short of looping. A separate
    class exists here because the composition record it keeps is the
    deliverable, not a side effect of a seam test.

    `record` holds the composition of every request, in call order: what the
    baseline stores. `captured` holds the raw `LlmRequest` objects themselves,
    for the assertions that compare two requests byte for byte. A canonical
    form cannot be recovered from a composition record, and the record stays
    JSON-serialisable only while it does not try to carry them.

    `agent_name` and `scenario` are what a record says it measured, set by
    the caller, so records from one agent under one scenario and another are
    told apart in one baseline.
    """

    model: str = "recording-context-probe"
    script: list[Any] = Field(default_factory=list)
    agent_name: str = "unset"
    scenario: str = "unset"
    record: list[dict[str, Any]] = Field(default_factory=list)
    captured: list[Any] = Field(default_factory=list)

    async def generate_content_async(self, llm_request: Any, stream: bool = False) -> Any:
        del stream
        index = len(self.record)
        self.captured.append(llm_request)
        self.record.append(
            compose(
                llm_request,
                agent=self.agent_name,
                scenario=self.scenario,
                call_index=index,
            )
        )
        step = self.script[index] if index < len(self.script) else "done"
        if isinstance(step, str):
            yield LlmResponse(
                content=types.Content(role="model", parts=[types.Part.from_text(text=step)])
            )
            return
        yield LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part.from_function_call(name=name, args=args) for name, args in step],
            )
        )


# Measurement -------------------------------------------------------------------


def _json_chars(value: Any) -> int:
    """Chars of `value` as compact JSON, the size a part's own payload costs.

    `default=str` covers the occasional non-JSON scalar ADK carries through
    from a fixture (`ToolCallResult` dumps are already plain JSON), so a probe
    run never fails on a value it can still measure.
    """
    return len(json.dumps(value, default=str, separators=(",", ":")))


def _system_instruction_chars(config: Any) -> int:
    """Chars of the request's system instruction, whether it is text or a `Content`."""
    instruction = getattr(config, "system_instruction", None)
    if instruction is None:
        return 0
    parts = getattr(instruction, "parts", None)
    if parts:
        return sum(len(part.text or "") for part in parts)
    return len(str(instruction))


def _compact(value: Any) -> str:
    """`value` as compact JSON with keys sorted, for byte comparison."""
    return json.dumps(value, default=str, sort_keys=True, separators=(",", ":"))


def _declaration_chars(declaration: Any) -> int:
    """Chars of one function declaration as the model is sent it."""
    return len(json.dumps(declaration.model_dump(exclude_none=True), default=str))


def _declarations(config: Any) -> list[dict[str, Any]]:
    """Every function declaration the request offers, largest first.

    One `Tool` may carry several declarations. `adk_request_input` arrives as
    its own tool today, so this flattens
    across every tool's declarations, without assuming a one-to-one mapping.
    """
    found: list[dict[str, Any]] = [
        {"name": declaration.name, "chars": _declaration_chars(declaration)}
        for tool in getattr(config, "tools", None) or []
        for declaration in getattr(tool, "function_declarations", None) or []
    ]
    return sorted(found, key=lambda item: (-item["chars"], item["name"]))


def _part_bucket(part: Any) -> tuple[str, str | None]:
    """The bucket a part's content falls in, and the relayed agent if it is a relay.

    Order matters: a thought part also carries `.text`, and a relayed part is
    text that happens to contain the fence marker, so both are tested before
    plain text.
    """
    if part.function_call is not None:
        return f"functionCall:{part.function_call.name}", None
    if part.function_response is not None:
        return f"functionResponse:{part.function_response.name}", None
    text = part.text
    if text is None:
        return "other", None
    if part.thought:
        return "thought", None
    if RELAY_MARKER in text:
        match = _RELAY_AUTHOR.match(text)
        author = match.group(1) if match else _UNTAGGED
        return f"{RELAY_BUCKET_PREFIX}{author}", author
    return (SEAM_BUCKET, None) if SEAM_MARKER in text else ("text", None)


def _part_chars(part: Any) -> int:
    """Chars a part costs in the request, by the same rule its bucket is read."""
    if part.function_call is not None:
        call = part.function_call
        return _json_chars({"name": call.name, "args": call.args})
    if part.function_response is not None:
        response = part.function_response
        return _json_chars({"name": response.name, "response": response.response})
    if part.text is not None:
        return len(part.text)
    return len(str(part))


def contents_composition(contents: list[Any]) -> dict[str, Any]:
    """`contents` bucketed by part kind, with the relay split by relaying agent."""
    buckets: dict[str, dict[str, int]] = {}
    total = 0
    for content in contents or []:
        for part in content.parts or []:
            bucket, _author = _part_bucket(part)
            chars = _part_chars(part)
            total += chars
            entry = buckets.setdefault(bucket, {"count": 0, "chars": 0})
            entry["count"] += 1
            entry["chars"] += chars
    return {
        "total_chars": total,
        "buckets": [
            {"bucket": name, "count": entry["count"], "chars": entry["chars"]}
            for name, entry in sorted(
                buckets.items(), key=lambda item: (-item[1]["chars"], item[0])
            )
        ],
    }


def compose(llm_request: Any, *, agent: str, scenario: str, call_index: int) -> dict[str, Any]:
    """One request's composition record, as the baseline stores it.

    `total_chars` is the serialised request (system instruction, tool
    declarations and contents as ADK assembled them), so a segment that grows
    while another shrinks is still visible in the total.
    """
    config = llm_request.config
    declarations = _declarations(config)
    contents = contents_composition(llm_request.contents)
    return {
        "agent": agent,
        "scenario": scenario,
        "call_index": call_index,
        "total_chars": len(llm_request.model_dump_json()),
        "system_instruction_chars": _system_instruction_chars(config),
        "declaration_total_chars": sum(item["chars"] for item in declarations),
        "declarations": declarations,
        "contents": contents,
    }


# Stability: what the prompt caches match ------------------------------------------
#
# `harness.prefix_cache.prefix_key` hashes the model, the system instruction, the
# tool declarations and the tool config, so a prefix that starts varying per
# request creates a cache per variant. Gemini's implicit caching matches the longest
# leading run of the request it has seen before, so a rewritten earlier content
# loses the cached share from that point. Neither failure says so in the log.


def system_instruction_text(llm_request: Any) -> str:
    """The request's system instruction as text, for byte comparison."""
    instruction = getattr(llm_request.config, "system_instruction", None)
    if instruction is None:
        return ""
    parts = getattr(instruction, "parts", None)
    if parts:
        return "\n".join(part.text or "" for part in parts)
    return str(instruction)


def contents_leading(llm_request: Any) -> tuple[str, ...]:
    """One canonical string per content in the request's `contents`, in order.

    Gemini's implicit caching matches a leading run of contents, so a run whose
    earlier turns were rewritten, not merely appended to, would miss the cache
    from the point of the rewrite. Comparing two calls' `contents_leading`
    prefix-wise is how that shows up.
    """
    return tuple(
        _compact(content.model_dump(exclude_none=True)) for content in llm_request.contents
    )


def relay_buckets(record: dict[str, Any]) -> dict[str, int]:
    """Every relayed author's chars in `record`, keyed by author.

    Each relaying agent by name, `untagged` for ADK's own preamble, and any
    other author that ever appears, each on its own row, so an unmoved author cannot hide
    behind a moved one.
    """
    found: dict[str, int] = {}
    for entry in record["contents"]["buckets"]:
        if entry["bucket"].startswith(RELAY_BUCKET_PREFIX):
            found[entry["bucket"][len(RELAY_BUCKET_PREFIX) :]] = entry["chars"]
    return found


def bucket_chars(record: dict[str, Any], bucket: str) -> int:
    """Chars one named bucket holds in `record`, 0 when the bucket is absent."""
    for entry in record["contents"]["buckets"]:
        if entry["bucket"] == bucket:
            return entry["chars"]
    return 0


# The largest legal plan ----------------------------------------------------------
#
# `Plan` limits every string and collection it holds (`planner.plan.MAX_*`) and
# the serialised size of the model-authored part as a whole
# (`MAX_PLAN_RECORD_CHARS`). `harness.planner_seam._plan_context` renders the
# fields it does not truncate in full, so the largest plan the schema accepts
# fixes how large that section can get. Three builders here: the record with
# every field at its own limit, which is over the size limit and so is not a
# plan; a plan built to exactly a given size; and the plan whose framing is
# largest.

# One code point outside the basic plane, which `json.dumps` writes as two
# `\uXXXX` escapes, twelve characters, unless told to leave it as written.
# `_plan_context` renders entities, predicates and each decision's `fixes` as
# JSON, so filling those fields with it is what exposes a render that escapes:
# the plan section stays inside its derived ceiling only while the code point
# takes one character there, as it does in the size `Plan` counts.
ASTRAL_CHAR = "\U0001f600"


def padded(prefix: str, length: int, filler: str) -> str:
    """`prefix` followed by `filler`, exactly `length` code points long.

    A prefix keeps the strings a plan needs unique (entity names, options)
    distinct while every one of them stays at its limit.
    """
    return (prefix + filler * length)[:length]


def record_chars(record: dict[str, Any]) -> int:
    """Characters `record` serialises to: compact JSON, non-ASCII as written.

    The measure `MAX_PLAN_RECORD_CHARS` limits, written out here so the tests
    do not depend on the helper they check.
    """
    return len(json.dumps(record, separators=(",", ":"), ensure_ascii=False))


def field_limits_record() -> dict[str, Any]:
    """The record of a plan with every field at its own limit, dumped.

    Twenty-four decisions, each with `MAX_DECISION_OPTIONS` options, a
    description and a rationale at their limits and `fixes` stating
    `MAX_ENTITIES` entities, eight predicates and the other fields a
    requirement compares. Twenty-four requirements, each citing all twenty-four
    decisions: eight datasets with `MAX_ENTITIES` entities and
    `MAX_ACROSS_WORDS` `across` words apiece, nine filters with eight predicates apiece, and one of every other kind (aggregate,
    growth, rank, share_of, quotient_of, comparison, join). Every string is at its limit;
    those the seam renders as JSON are made of `ASTRAL_CHAR`, and a predicate
    lists every entity name. Each decision and requirement validates on its
    own. The record as a plan does not: it serialises to far more than
    `MAX_PLAN_RECORD_CHARS`, and `Plan` refuses it for that alone.
    """
    slots = tuple(f"slot_{index:02d}" for index in range(MAX_PLAN_DECISIONS))
    role = ASTRAL_CHAR * MAX_ENTITY_ROLE_CHARS
    across = tuple(
        padded(f"across{index}", MAX_ENTITY_ROLE_CHARS, ASTRAL_CHAR)
        for index in range(MAX_ACROSS_WORDS)
    )
    entities = tuple(
        Entity(name=padded(f"{index:02d}", MAX_ENTITY_NAME_CHARS, ASTRAL_CHAR), role=role)
        for index in range(MAX_ENTITIES)
    )
    names = tuple(entity.name for entity in entities)
    measure = ASTRAL_CHAR * MAX_MEASURE_CHARS
    # Each entity_in lists every name under a role of its own, so no two
    # predicates are equal (`FilterRequirement` refuses a repeat whatever the
    # order of its names). `entity_in` is the largest of the three kinds.
    predicates = tuple(
        EntityIn(names=names, role=padded(f"{index}", MAX_ENTITY_ROLE_CHARS, ASTRAL_CHAR))
        for index in range(MAX_PREDICATES)
    )
    fixes = DecisionFixes(
        entities=entities,
        across=across,
        measure=measure,
        group_by=(GroupBy.PERIOD, GroupBy.ENTITY),
        predicates=predicates,
        reduction=Reduction.SUM,
        among=RankAmong.POPULATION,
        axis=role,
        denominator_scope=DenominatorScope.POOLED,
    )
    decisions = []
    for index, slot in enumerate(slots):
        options = tuple(
            padded(f"{index:02d}{option:02d}", MAX_OPTION_CHARS, "o")
            for option in range(MAX_DECISION_OPTIONS)
        )
        decisions.append(
            Decision(
                slot=slot,
                description="d" * MAX_DECISION_DESCRIPTION_CHARS,
                options=options,
                chosen=options[0],
                resolved_by=ResolvedBy.PLANNER_DEFAULT,
                rationale="r" * MAX_DECISION_RATIONALE_CHARS,
                fixes=fixes,
            )
        )
    requirements: list[Any] = [
        DatasetRequirement(
            id=f"dataset_{index}",
            decision_ids=slots,
            measure=measure,
            entities=entities,
            across=across,
            period=Period(start=2015, end=2024),
            unit=Unit(currency="USD", prices="current"),
            source="s" * MAX_SOURCE_CHARS,
        )
        for index in range(8)
    ]
    requirements += [
        FilterRequirement(
            id=f"filter_{index}",
            decision_ids=slots,
            input_id=f"dataset_{index % 8}",
            predicates=predicates,
        )
        for index in range(9)
    ]
    requirements += [
        AggregateRequirement(
            id="totals",
            decision_ids=slots,
            input_id="dataset_0",
            reduction=Reduction.SUM,
            group_by=(GroupBy.PERIOD, GroupBy.ENTITY),
        ),
        GrowthRequirement(
            id="growth", decision_ids=slots, input_id="totals", window_start=2016, window_end=2020
        ),
        RankRequirement(
            id="rank",
            decision_ids=slots,
            input_id="dataset_1",
            direction=RankDirection.HIGHEST_FIRST,
            among=RankAmong.POPULATION,
            axis=role,
            reduction=Reduction.SUM,
        ),
        ShareOfRequirement(
            id="share",
            decision_ids=slots,
            numerator=RequirementSide(input_id="totals", entity=names[0]),
            denominator=RequirementSide(input_id="filter_0", entity=names[0]),
            axis=role,
            denominator_scope=DenominatorScope.POOLED,
        ),
        QuotientRequirement(
            id="quotient",
            decision_ids=slots,
            numerator=RequirementSide(input_id="totals", entity=names[2]),
            denominator=RequirementSide(input_id="filter_4", entity=names[2]),
            axis=role,
            denominator_axis=role,
        ),
        ComparisonRequirement(
            id="comparison",
            decision_ids=slots,
            minuend=RequirementSide(input_id="filter_1", entity=names[1]),
            subtrahend=RequirementSide(input_id="totals", entity=names[1]),
        ),
        JoinRequirement(id="joined", decision_ids=slots, left_id="filter_2", right_id="filter_3"),
    ]
    return {
        "question": "q" * MAX_QUESTION_CHARS,
        "answer_mode": AnswerMode.COMPUTED.value,
        "decisions": [decision.model_dump(mode="json") for decision in decisions],
        "evidence_requirements": [
            requirement.model_dump(mode="json") for requirement in requirements
        ],
    }


def plan_record(
    chars: int | None = None, *, options: int = 2, cites_all: bool = False
) -> dict[str, Any]:
    """The model-authored part of a plan of every kind, dumped, at a chosen size.

    Twenty-four decisions and twenty-four requirements: eight datasets, nine
    filters, and one each of aggregate, growth, rank, share_of, quotient_of,
    comparison and join. Requirement `i` cites decision `i`, or all twenty-four when
    `cites_all`. Each decision states `options` short options.

    With `chars` given, the record serialises to exactly that many characters
    (`record_chars`). The padding is the worst case the render meets: entity
    names of `ASTRAL_CHAR` on the first dataset, the most a name may hold each,
    and the few characters left over as plain text in the question. Without
    it, the record is the smallest this shape takes.
    """
    slots = tuple(f"slot_{index:02d}" for index in range(MAX_PLAN_DECISIONS))
    option_names = tuple(f"o{option}" for option in range(options))
    decisions = [
        Decision(
            slot=slot,
            description="d",
            options=option_names,
            chosen=option_names[0],
            resolved_by=ResolvedBy.PLANNER_DEFAULT,
            rationale="r",
            # The rank cites this decision, and a population rank needs a
            # decision that fixes it.
            fixes=DecisionFixes(among=RankAmong.POPULATION) if index == 19 else None,
        )
        for index, slot in enumerate(slots)
    ]

    def cites(index: int) -> tuple[str, ...]:
        return slots if cites_all else (slots[index],)

    entity = Entity(name="Kenya", role="debtor")
    requirements: list[Any] = [
        DatasetRequirement(
            id=f"dataset_{index}",
            decision_ids=cites(index),
            measure="external debt service paid",
            entities=[entity],
            period=Period(start=2015, end=2024),
        )
        for index in range(8)
    ]
    requirements += [
        FilterRequirement(
            id=f"filter_{index}",
            decision_ids=cites(8 + index),
            input_id=f"dataset_{index % 8}",
            predicates=(EntityIn(names=("Kenya",)),),
        )
        for index in range(9)
    ]
    requirements += [
        AggregateRequirement(
            id="totals",
            decision_ids=cites(17),
            input_id="dataset_0",
            reduction=Reduction.SUM,
            group_by=(GroupBy.PERIOD,),
        ),
        GrowthRequirement(
            id="growth",
            decision_ids=cites(18),
            input_id="totals",
            window_start=2016,
            window_end=2020,
        ),
        RankRequirement(
            id="rank",
            decision_ids=cites(19),
            input_id="dataset_1",
            direction=RankDirection.HIGHEST_FIRST,
            among=RankAmong.POPULATION,
            reduction=Reduction.SUM,
            per_period=True,
        ),
        ShareOfRequirement(
            id="share",
            decision_ids=cites(20),
            numerator=RequirementSide(input_id="totals"),
            denominator=RequirementSide(input_id="filter_0"),
        ),
        QuotientRequirement(
            id="quotient",
            decision_ids=cites(21),
            numerator=RequirementSide(input_id="totals"),
            denominator=RequirementSide(input_id="filter_4"),
        ),
        ComparisonRequirement(
            id="comparison",
            decision_ids=cites(22),
            minuend=RequirementSide(input_id="filter_1"),
            subtrahend=RequirementSide(input_id="totals"),
        ),
        JoinRequirement(
            id="joined", decision_ids=cites(23), left_id="filter_2", right_id="filter_3"
        ),
    ]
    record: dict[str, Any] = {
        "question": "q",
        "answer_mode": AnswerMode.COMPUTED.value,
        "decisions": [decision.model_dump(mode="json") for decision in decisions],
        "evidence_requirements": [
            requirement.model_dump(mode="json") for requirement in requirements
        ],
    }
    if chars is None:
        return record
    return _padded_to(record, chars)


def _padded_to(record: dict[str, Any], chars: int) -> dict[str, Any]:
    """`record` grown to exactly `chars` characters, in the worst case the render meets."""
    if record_chars(record) > chars:
        raise ValueError(f"the smallest plan of this shape is {record_chars(record)} characters")

    def with_names(code_points: int) -> dict[str, Any]:
        grown = json.loads(json.dumps(record))
        names = [
            padded(
                f"{index:02d}",
                min(MAX_ENTITY_NAME_CHARS, code_points - MAX_ENTITY_NAME_CHARS * index),
                ASTRAL_CHAR,
            )
            for index in range(-(-code_points // MAX_ENTITY_NAME_CHARS))
        ]
        grown["evidence_requirements"][0]["entities"] += [
            {"name": name, "role": "debtor"} for name in names if name
        ]
        return grown

    low, high = 0, MAX_ENTITIES * MAX_ENTITY_NAME_CHARS
    while low < high:
        middle = (low + high + 1) // 2
        if record_chars(with_names(middle)) <= chars:
            low = middle
        else:
            high = middle - 1
    grown = with_names(low)
    grown["question"] += "x" * (chars - record_chars(grown))
    assert record_chars(grown) == chars
    return grown


# The plan's schema ----------------------------------------------------------------

# The patterns that bound a string, each with the longest string it admits:
# `Identifier` (a lower-case letter and up to sixty-three more), `SpecId`
# (`spec_` and thirty-two hex digits) and a currency code.
BOUNDING_PATTERNS = {
    r"^[a-z][a-z0-9_]{0,63}$": 64,
    r"^spec_[0-9a-f]{32}$": 37,
    r"^[A-Z]{3}$": 3,
}


def unbounded_paths(schema: dict[str, Any], definitions: dict[str, Any], where: str) -> list[str]:
    """The places under `schema` where a string, an array or a map has no bound.

    A string is bounded by `maxLength`, by an `enum` or `const`, or by one of
    `BOUNDING_PATTERNS`; an array by `maxItems`; a map is refused outright,
    since nothing here bounds its keys.
    """
    found: list[str] = []

    def visit(node: Any, path: str, seen: frozenset[str]) -> None:
        if not isinstance(node, dict):
            return
        if "$ref" in node:
            name = node["$ref"].rsplit("/", 1)[-1]
            if name not in seen:
                visit(definitions[name], f"{path}<{name}>", seen | {name})
            return
        kind = node.get("type")
        if kind == "string" and not (
            "maxLength" in node
            or "enum" in node
            or "const" in node
            or node.get("pattern") in BOUNDING_PATTERNS
        ):
            found.append(f"{path}: a string with no maxLength")
        if kind == "array" and "maxItems" not in node:
            found.append(f"{path}: an array with no maxItems")
        if kind == "object" and node.get("additionalProperties") not in (None, False):
            found.append(f"{path}: an object whose keys are unbounded")
        for name, child in (node.get("properties") or {}).items():
            visit(child, f"{path}.{name}", seen)
        for key in ("items", "additionalProperties"):
            visit(node.get(key), f"{path}[]", seen)
        for key in ("anyOf", "oneOf", "allOf", "prefixItems"):
            for child in node.get(key) or []:
                visit(child, path, seen)

    visit(schema, where, frozenset())
    return found
