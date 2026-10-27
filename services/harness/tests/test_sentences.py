"""Tests for `harness.sentences`: every code the user can be told about has its sentence."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Collection, Mapping
from enum import StrEnum
from pathlib import Path
from typing import get_args

import dfo_spec_kit
import pytest
from analysis_mcp.tool_errors import ToolErrorCode as AnalysisCode
from compiler.compile import CompileIssueCode
from data_mcp.tool_errors import CeilingKind
from data_mcp.tool_errors import ToolErrorCode as DataCode
from executor.classify import (
    COMPILER_PROVABLE_CODES,
    METHOD_CHOICE_CODES,
    REDISCOVERY_CODES,
    STOP_CODES,
    TRANSIENT_CODES,
    ExecutorCode,
)
from executor.outcomes import RefusalKind
from harness.orchestrator_tools import TOOL_NAMES
from harness.sentences import (
    _CLASS_CODES,
    _FILL_KEYS,
    _RETRY_ADVICE,
    CEILING_SENTENCES,
    CODE_SENTENCES,
    COMPILE_REASONS,
    DECLINE_REASONS,
    FAILURE_CLASS,
    FAILURE_SENTENCES,
    MAX_SENTENCE_CHARS,
    REFERRAL_REASONS,
    RETRY_MODE_SENTENCES,
    SEAM_REASONS,
    SEAM_SENTENCES,
    FailureClass,
    PlanWords,
    SeamCode,
    failure_sentence,
    referral_reason,
    refusal_reason,
    refusal_sentence,
)
from harness.tool_contracts import HarnessCode, harness_refusal
from harness.wire import ErrorCode, RetryMode
from planner.plan import UnsupportedOperation
from planner.records import EvidenceIssueCode
from planner.tool_results import ProblemCode
from translator.referral import ReferralKind


@pytest.mark.parametrize(
    ("table", "codes"),
    [
        (REFERRAL_REASONS, set(ReferralKind)),
        (COMPILE_REASONS, set(CompileIssueCode)),
        (DECLINE_REASONS, set(UnsupportedOperation)),
        (SEAM_SENTENCES, set(SeamCode) | set(ProblemCode)),
        (SEAM_REASONS, set(SeamCode) | set(ProblemCode)),
    ],
    ids=["referral", "compile", "decline", "seam", "seam-reason"],
)
def test_every_code_has_exactly_one_sentence(
    table: Mapping[StrEnum, str], codes: Collection[StrEnum]
) -> None:
    assert set(table) == set(codes)


@pytest.mark.parametrize(
    "clause", [*REFERRAL_REASONS.values(), *COMPILE_REASONS.values(), *DECLINE_REASONS.values()]
)
def test_a_reason_is_a_clause_another_sentence_can_carry(clause: str) -> None:
    assert clause[0].islower()
    assert not clause.endswith(".")


@pytest.mark.parametrize("code", [*SeamCode, *ProblemCode])
def test_every_seam_sentence_fills_from_an_empty_detail(code: StrEnum) -> None:
    text = refusal_sentence({"code": code})

    assert "{" not in text
    assert text.endswith(".")


def test_a_code_no_table_names_claims_no_budget() -> None:
    text = refusal_sentence({"code": "no_such_code", "detail": {"rounds": 4}})

    assert "budget" not in text
    assert text not in {refusal_sentence({"code": code}) for code in SEAM_SENTENCES}


def test_a_repair_referral_reads_the_compile_issue_it_names() -> None:
    repair = ReferralKind.REQUIREMENT_NEEDS_REPAIR

    assert (
        referral_reason(repair, "family_unsummable")
        == (COMPILE_REASONS[CompileIssueCode.FAMILY_UNSUMMABLE])
    )
    assert referral_reason(repair, None) == REFERRAL_REASONS[repair]
    assert referral_reason(repair, "no_such_issue") == REFERRAL_REASONS[repair]
    assert referral_reason("no_data") == REFERRAL_REASONS[ReferralKind.NO_DATA]
    assert referral_reason("no_such_kind") == referral_reason(None)
    unmeetable = ReferralKind.REQUIREMENT_UNMEETABLE
    assert referral_reason(unmeetable, source_mismatch=True) != REFERRAL_REASONS[unmeetable]
    assert referral_reason(unmeetable) == REFERRAL_REASONS[unmeetable]


def test_a_plan_not_ready_sentence_carries_the_referral_it_ended_on() -> None:
    text = refusal_sentence(
        {
            "code": SeamCode.PLAN_NOT_READY,
            "detail": {
                "kind": "requirement_needs_repair",
                "requirement_id": "aid_by_donor",
                "constraint": "family_unsummable",
            },
        },
        words=PlanWords(names={"aid_by_donor": "aid by donor"}),
    )

    assert text.startswith("No plan was made for aid by donor: ")
    assert COMPILE_REASONS[CompileIssueCode.FAMILY_UNSUMMABLE] in text


_REQUIREMENT_ID = "total_aid_dac1"
_STEP_ID = "share_health"


@pytest.mark.parametrize("code", [*SeamCode, *ProblemCode])
@pytest.mark.parametrize(
    "where",
    [
        {"requirement_id": _REQUIREMENT_ID},
        {"basis": {"requirement_id": _REQUIREMENT_ID, "step_id": _STEP_ID}},
    ],
    ids=["requirement", "basis"],
)
@pytest.mark.parametrize("named", [True, False], ids=["named", "unnamed"])
@pytest.mark.parametrize(
    ("render", "table"),
    [(refusal_sentence, SEAM_SENTENCES), (refusal_reason, SEAM_REASONS)],
    ids=["sentence", "reason"],
)
def test_a_sentence_names_a_requirement_by_its_readers_name_and_never_by_its_id(
    code: StrEnum,
    where: dict[str, object],
    named: bool,
    render: Callable[..., str],
    table: Mapping[StrEnum, str],
) -> None:
    """A requirement a refusal is about reads by the name the plan card gives
    it, and with no name it goes unnamed: its id reads as a fault."""
    words = PlanWords(
        names={_REQUIREMENT_ID: "official development assistance"} if named else {},
        inputs={_STEP_ID: ("health_crs", _REQUIREMENT_ID)},
    )

    text = render({"code": code, "detail": where}, words=words)

    assert _REQUIREMENT_ID not in text
    assert not {_STEP_ID, "health_crs"} & set(re.findall(r"[\w']+", text))
    if named and any(f"{{{slot}}}" in table[code] for slot in ("at", "about", "lead")):
        assert "official development assistance" in text


@pytest.mark.parametrize("code", [*SeamCode, *ProblemCode])
def test_a_reason_is_one_sentence_saying_why_alone(code: StrEnum) -> None:
    reason = refusal_reason({"code": code})

    assert "{" not in reason
    assert reason[0].isupper()
    assert reason.endswith(".")
    assert ". " not in reason
    assert "nothing was recorded" not in reason


def test_a_reason_for_an_input_off_its_calculations_provider_scope_names_both_sides() -> None:
    reason = refusal_reason(
        {
            "code": ProblemCode.OPERATION_UNSUPPORTED,
            "detail": {
                "operation": "unsupported_ask",
                "basis": {
                    "requirement_id": "total",
                    "step_id": "share",
                    "option_ids": ["provider:inclusive"],
                },
            },
        },
        words=PlanWords(
            names={"total": "official development assistance", "health": "health aid"},
            inputs={"share": ("health", "total")},
        ),
    )

    assert reason == (
        "Official development assistance would then count different providers' spending "
        "from health aid."
    )


@pytest.mark.parametrize(
    ("detail", "reason"),
    [
        (
            {"operation": "unsupported_ask", "basis": {"option_ids": ["provider:inclusive"]}},
            "count different providers' spending",
        ),
        (
            {"operation": "unsupported_ask", "basis": {"option_ids": ["measure_basis:net"]}},
            "counted on different measure bases",
        ),
        (
            {"operation": "breakdown_not_kept"},
            DECLINE_REASONS[UnsupportedOperation.BREAKDOWN_NOT_KEPT],
        ),
    ],
    ids=["provider", "measure-basis", "declined-operation"],
)
def test_an_unsupported_calculation_says_why_it_cannot_be_made(
    detail: dict[str, object], reason: str
) -> None:
    text = refusal_sentence(
        {
            "code": ProblemCode.OPERATION_UNSUPPORTED,
            "detail": {**detail, "requirement_id": "total"},
        },
        words=PlanWords(names={"total": "official development assistance"}),
    )

    assert text.startswith("This cannot be calculated as planned for official development")
    assert reason in text


def test_the_budget_refusal_is_stated_as_a_per_entry_budget_in_the_user_s_terms() -> None:
    """`MAX_PLAN_ROUNDS` counts the proposals of one entry into the planner, restarts included.

    The plan's own version can read higher than the count (it climbs across
    entries), so the sentence names the budget. It makes no claim about what
    can still be approved: the seam has no view of the graph's record, and the
    graph adds that claim where its record supports it.
    """
    text = refusal_sentence(
        {"code": "budget_exhausted", "detail": {"scope": "plan_rounds", "rounds": 4, "version": 9}}
    )

    assert text == (
        "This request has used its budget of 4 plan drafts, so nothing was "
        "recorded. Your next message starts with a fresh budget."
    )


def test_the_question_fixed_refusal_is_stated_as_asking_the_other_question_separately() -> None:
    """A restart while a plan is already recorded spends no budget, so the
    sentence says the other question goes in its own message."""
    text = refusal_sentence({"code": "question_fixed", "outcome": "refused"})

    assert "own message" in text
    assert "budget" not in text


def _spec_kit_presentation_codes() -> set[str]:
    """The refusals the packaged registry maps to a `presentation.*` endpoint."""
    path = Path(dfo_spec_kit.__file__).parent / "registry" / "codes.json"
    return {
        entry["code"]
        for entry in json.loads(path.read_text())["entries"]
        if entry["classification"] == "refusal"
        and any(mapping["endpoint"].startswith("presentation.") for mapping in entry["mappings"])
    }


# Every vocabulary a failed step's code can come from.
_FAILURE_SOURCES: tuple[Collection[str], ...] = (
    DataCode,
    AnalysisCode,
    RefusalKind,
    EvidenceIssueCode,
    ErrorCode,
    ExecutorCode,
    HarnessCode,
    METHOD_CHOICE_CODES,
    REDISCOVERY_CODES,
    TRANSIENT_CODES,
    COMPILER_PROVABLE_CODES,
    STOP_CODES,
    _spec_kit_presentation_codes(),
    # The codes a refused plan's step is worded from.
    SeamCode,
    ProblemCode,
)
_TEXTS = {**{str(c): FAILURE_SENTENCES[FAILURE_CLASS[c]] for c in FAILURE_CLASS}, **CODE_SENTENCES}
_MODES = (None, *RetryMode)
_FORBIDDEN = re.compile(r"spec_|dcid|scope\.|_component|trace|[{}]", re.IGNORECASE)


def test_every_enumerable_failure_code_has_a_class() -> None:
    enumerated = {str(code) for source in _FAILURE_SOURCES for code in source}

    assert set(FAILURE_CLASS) == enumerated, {
        "unclassed": sorted(enumerated - set(FAILURE_CLASS)),
        "stale": sorted(set(FAILURE_CLASS) - enumerated),
    }


def test_a_failure_code_sits_in_one_class_and_in_no_other_table() -> None:
    """A compile issue reaches the user as a referral's `constraint`, read from
    `COMPILE_REASONS`, so a name the compiler shares with a service is no
    second sentence for one code. Only a code of `SEAM_SENTENCES` has a class
    beside its template. The class words the step, and the template the agent's
    reading."""
    classed = set(FAILURE_CLASS)
    seam = {str(code) for code in SEAM_SENTENCES}
    sentenced = {str(code) for enum in (ReferralKind, UnsupportedOperation) for code in enum}

    assert sum(len(codes) for codes in _CLASS_CODES.values()) == len(classed)
    assert set(FAILURE_SENTENCES) == set(FailureClass) == set(_CLASS_CODES)
    assert seam <= classed
    assert not (classed - seam) & (sentenced | {str(code) for code in ProblemCode})
    assert classed & {str(code) for code in CompileIssueCode} == {
        "entity_ambiguous",
        "unknown_tool",
        "requirement_unknown",
    }


def test_a_code_with_a_sentence_of_its_own_is_a_classed_code() -> None:
    assert set(CODE_SENTENCES) <= set(FAILURE_CLASS)
    assert set(_FILL_KEYS) == {code for code, text in CODE_SENTENCES.items() if text.filled}
    assert all(text.filled is None for text in FAILURE_SENTENCES.values())


def test_a_code_in_no_class_reads_the_sentence_of_its_retry_mode() -> None:
    assert set(RETRY_MODE_SENTENCES) == set(RetryMode)
    sentences = {failure_sentence("no_such_code", mode) for mode in _MODES}

    assert len(sentences) == len(_MODES)
    assert failure_sentence(None, "no_such_mode") == failure_sentence("no_such_code")
    assert failure_sentence("no_such_code", {"mode": 1}) == failure_sentence("no_such_code")


def test_only_an_unconfirmed_write_advises_looking_at_the_canvas() -> None:
    """A transient failure that is not to be retried is told to try again
    only where its own text says so, and the canvas is named for one code."""
    unconfirmed = HarnessCode.WRITE_UNCONFIRMED
    plain = _CLASS_CODES[FailureClass.TRANSIENT] - set(CODE_SENTENCES)

    assert plain
    for code in plain:
        assert failure_sentence(code, RetryMode.DO_NOT_RETRY) == failure_sentence(code)
        assert failure_sentence(code, RetryMode.RETRY_AFTER_DELAY) != failure_sentence(code)
    assert len({failure_sentence(unconfirmed, mode) for mode in _MODES}) == 1
    assert [code for code, text in _TEXTS.items() if "canvas" in text.cause] == [unconfirmed]
    assert FAILURE_CLASS[HarnessCode.ROLLING_REPLAY_PENDING] is FailureClass.TRANSIENT
    assert failure_sentence(HarnessCode.ROLLING_REPLAY_PENDING, RetryMode.DO_NOT_RETRY) not in {
        failure_sentence(code, RetryMode.DO_NOT_RETRY) for code in {*plain, unconfirmed}
    }


def test_the_retry_mode_advises_where_the_text_does_not() -> None:
    unadvised = next(code for code, text in _TEXTS.items() if text.advice is None)

    assert len({failure_sentence(unadvised, mode) for mode in _MODES}) == len(
        {None, *_RETRY_ADVICE.values()}
    )
    assert set(_RETRY_ADVICE) == set(RetryMode)


def _fills(code: str) -> list[Mapping[str, object] | None]:
    """No fix, then fixes whose fill-in fits, overruns the sentence by one
    character or more, or holds what the Activity log cuts at or an identifier."""
    key = _FILL_KEYS.get(code)
    if key is None:
        return [None]
    values: list[object] = ["Kenya", 2023, True, None, "Kenya. Uganda", "a\nb", "country/KEN"]
    values += ["stat_var", "0a1b2c3d4e5f", *("x" * size for size in range(1, MAX_SENTENCE_CHARS))]
    return [None, *({key: value} for value in values)]


def _assert_fits_the_activity_log(sentence: str) -> None:
    internal = {name for name in (*_TEXTS, *TOOL_NAMES) if "_" in name}

    assert len(sentence) <= MAX_SENTENCE_CHARS, sentence
    assert sentence.endswith(".")
    assert not re.search(r"\.\s|\n", sentence), sentence
    assert not _FORBIDDEN.search(sentence), sentence
    assert not [name for name in internal if name in sentence], sentence


@pytest.mark.parametrize("code", sorted(_TEXTS))
def test_a_failure_sentence_fits_the_activity_log_and_names_nothing_internal(code: str) -> None:
    for mode in _MODES:
        plain = failure_sentence(code, mode)
        for fix in _fills(code):
            sentence = failure_sentence(code, mode, fix)
            _assert_fits_the_activity_log(sentence)
            said = next(iter(fix.values())) if fix else None
            if said in ("Kenya", 2023, "x"):
                assert str(said) in sentence and sentence != plain
            elif not (isinstance(said, str) and set(said) == {"x"}):
                assert sentence == plain


def test_an_over_ceiling_reads_the_sentence_of_the_ceiling_it_names() -> None:
    """Fewer places or a shorter period do not bring a definition under a
    ceiling on the series it covers, so those two kinds advise otherwise."""
    code = DataCode.OVER_CEILING
    by_kind = {
        kind: {failure_sentence(code, mode, {"ceiling_kind": kind}) for mode in _MODES}
        for kind in CEILING_SENTENCES
    }

    assert set(CEILING_SENTENCES) == set(get_args(CeilingKind))
    for sentences in by_kind.values():
        assert len(sentences) == 1
        _assert_fits_the_activity_log(*sentences)
    assert by_kind["rows"] == by_kind["bytes"] == by_kind["requests"]
    assert (
        len({*by_kind["rows"], *by_kind["member_cardinality"], *by_kind["candidate_window"]}) == 3
    )
    for fix in (None, {}, {"ceiling_kind": "publish_body_bytes"}, {"ceiling_kind": 7}):
        assert failure_sentence(code, fix=fix) == failure_sentence(AnalysisCode.FRAME_TOO_LARGE)
    assert (
        failure_sentence(AnalysisCode.FRAME_TOO_LARGE, fix={"ceiling_kind": "rows"})
        not in (by_kind["rows"])
    )


def test_a_fill_in_reads_only_the_member_its_code_names() -> None:
    fix = {"value": "Kenya", "period": "2023", "detail": "Kenya"}

    assert "Kenya" in failure_sentence("binding_invalid", fix=fix)
    assert "2023" not in failure_sentence("binding_invalid", fix=fix)
    assert failure_sentence("schema_invalid", fix=fix) == failure_sentence("schema_invalid")


@pytest.mark.parametrize("code", list(HarnessCode))
def test_a_harness_refusal_states_its_code_and_retry_mode_as_plain_strings(
    code: HarnessCode,
) -> None:
    refusal = harness_refusal(code=code, message="m")

    assert type(refusal["code"]) is str
    assert refusal["retry_mode"] in set(RetryMode)
    assert type(refusal["retry_mode"]) is str


@pytest.mark.parametrize(
    ("code", "user_caused"),
    [
        (HarnessCode.SELECT_INVALID, "constraint_violation"),
        (HarnessCode.TOOL_ARGUMENTS_INVALID, "internal_error"),
    ],
)
def test_what_the_model_wrote_does_not_ask_the_user_to_change_the_question(
    code: str, user_caused: str
) -> None:
    """The model writes a chart's select and a tool call's arguments, so one
    refused as to correct reads neither the advice to change the question nor
    the sentence of another cause: a card the user's request made
    undrawable, or a fault inside the system."""
    mode = RetryMode.CORRECT_AND_RETRY
    sentence = failure_sentence(code, mode)

    assert sentence != failure_sentence(user_caused, mode)
    assert str(_RETRY_ADVICE[mode]) not in sentence
