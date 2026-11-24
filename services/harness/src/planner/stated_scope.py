"""The provider scope the user's words state, read by Search, for `propose_plan`'s admission.

Search's reader (`POST /api/search/stated-scope`, `common.live_apis.LiveApisClient.read_stated_scopes`)
reads a text's scope words with no model and no graph: the provider option it states, at
most one, whether it asks for a total, and every provider option it names. `read_stated_scope`
reads the turn's words (`harness.conversation.current_user_message`) and the `measure` of
each dataset requirement a proposal and the plan on record state, once per text in a turn,
and holds the readings in session state (`STATED_SCOPE_STATE_KEY`). Admission reads only
what is held (`stated_scope_record`), so the planner cannot write it: a requirement states a
provider option when Search reads it from the requirement's `measure` and the turn's words
name it (`submission.service._stated_reading`).

A read that fails records nothing, so `propose_plan` admits a plan only on a reading it
holds. Each read retries on its own (`_READ_ATTEMPTS`), so nothing carries from one read
to the next. A read that still fails ends the entry: on `STATED_SCOPE_UNAVAILABLE`, which
asks the user to try again shortly, where the failure can pass, and on
`STATED_SCOPE_UNREADABLE`, a fault in the system, where it cannot. An apis that serves
no stated-scope route (`StatedScopeNotServedError`), one deployed before the route or
rolled back past it, reads as holding no reading for the texts it was asked: admission
applies no stated scope there, as it does for a turn with no words of the user's, such
as a chip's or one outside Studio. Only Search's capabilities document can say the route
is not served, so a 404 from a wrong address ends the entry; `PLANNER_STATED_SCOPE=false`
turns the read off.

Four more slots belong to the rule, each holding for the turn: where the question
whether to keep a dataset that cannot count the provider scope the user's words ask
for stands (`KEEP_STATE_KEY`), the measures of the requirements each
`STATED_SCOPE_UNATTRIBUTED` refusal named (`UNATTRIBUTED_STATE_KEY`), which pass when
proposed again unchanged, the pins Search refused to compile on a stated option
(`UNCOMPILED_STATE_KEY`), and the compiles moving a stated option beside a
calculation's axis that Search refused (`COMBINED_REFUSED_STATE_KEY`), which a later
proposal does not post again.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, NamedTuple, Protocol

from common.apis_auth import apis_auth_provider
from common.live_apis import LiveApisClient, LiveApisError, StatedScopeReading
from harness.conversation import current_user_message
from harness.scope_choices import PROVIDER_AXIS

from planner.plan import MAX_MEASURE_CHARS, DatasetRequirement, SessionState, load_plan
from planner.records import StrictModel
from planner.tool_results import Outcome, ProblemCode, problem

logger = logging.getLogger(__name__)

# The session-state slot holding the turn's words and Search's reading of each
# text read this turn (`StatedScopeRecord`).
STATED_SCOPE_STATE_KEY = "planner:stated_scope"

# The session-state slot holding where the question whether to keep a dataset
# counted as published stands this turn (`KeepQuestion`). A paused turn's state
# outlives a deploy, so a slot whose shape changes takes a new key, and a value an
# earlier release wrote under the old one is never read: a keep question paused
# across a release that renamed this key is asked again when the turn resumes.
KEEP_STATE_KEY = "planner:stated_scope_keep_question"

# The session-state slot holding, by provider option, the dataset requirements a
# `STATED_SCOPE_UNATTRIBUTED` refusal was given for this turn whose pins offer the
# option, each as `[requirement_id, measure]` (`unattributed_refused`).
UNATTRIBUTED_STATE_KEY = "planner:stated_scope_unattributed_measures"

# The session-state slot holding each pin Search refused to compile on the provider
# option a requirement's `measure` states, with that option and Search's answer, as
# `[spec_id, option_id, cause]` (`stated_uncompiled`).
UNCOMPILED_STATE_KEY = "planner:stated_scope_uncompiled"

# The session-state slot holding each compile moving a stated provider option and a
# calculation's axis together that Search refused for good this turn, as
# `[spec_id, option_ids, code, reason, unanswered]` (`combined_refused`).
COMBINED_REFUSED_STATE_KEY = "planner:stated_scope_combined_refusals"

# The most texts Search's reader takes in one call (`StatedScopeRequest` in dfo-apis).
_MAX_TEXTS = 64

# The most characters of the turn's words one read sends. The route takes at most its
# `max_query_chars` per text (2,000 by default), so this must not exceed it. Studio
# bounds a message by the request's bytes only, so the words are cut here.
_MAX_TEXT_CHARS = 2000


class StatedOption(StrictModel):
    """A provider option a text states, with the words it was read from.

    `sums_providers` says the words sum the bilateral and multilateral lines naming no
    imputed share ("bilateral plus multilateral"), which a dataset reads as its own
    summing option (`StatedScopeReading.sums_providers`)."""

    option_id: str
    words: str
    sums_providers: bool = False


class TextReading(StrictModel):
    """What one text states of whose spending it counts (`StatedScopeReading`).

    `provider` is the provider option it states, at most one. `total` says it asks
    for a total, and `mentions` are the provider options it names, both options of
    a text naming two, which states neither, included.
    """

    provider: tuple[StatedOption, ...] = ()
    total: bool = False
    mentions: tuple[str, ...] = ()

    @classmethod
    def of(cls, reading: StatedScopeReading) -> TextReading:
        """The provider part of Search's `reading`."""
        return cls(
            provider=tuple(
                StatedOption(
                    option_id=entry.option_id,
                    words=entry.words,
                    sums_providers=reading.sums_providers,
                )
                for entry in reading.entries
                if entry.axis == PROVIDER_AXIS
            ),
            total=reading.total,
            mentions=reading.mentions,
        )

    @property
    def stated(self) -> StatedOption | None:
        """The one provider option the text states, `None` where it states none."""
        return self.provider[0] if len(self.provider) == 1 else None


class StatedScopeRecord(StrictModel):
    """The turn's words as read (`words`) and Search's reading of every text read
    this turn, by text."""

    words: str
    readings: dict[str, TextReading]


class StatedScopeNotServedError(Exception):
    """The apis read serves no stated-scope route, as one deployed before the route, or
    rolled back past it, does: it answered the route 404, and its Search capabilities
    document lists no live `stated_scope` door."""


class StatedScopeReader(Protocol):
    """Search's reader of the scope words of `texts`, one reading per text, in order.

    Raises `StatedScopeNotServedError` when the apis read serves no stated-scope route,
    and `LiveApisError` when Search does not answer otherwise."""

    async def __call__(self, texts: Sequence[str]) -> Sequence[TextReading]: ...


# The status an apis answers a path it does not route with. The route itself answers
# none: it reads any text it is given, and refuses a request outside its bounds 400.
_NOT_SERVED_STATUS = 404

# The door Search's capabilities document lists when the apis serves the route.
_STATED_SCOPE_DOOR = "stated_scope"

# How long the reader keeps the capabilities document's answer on whether the route
# is served. Short, so a process sees an apis release or rollback within a minute.
_DOORS_TTL_S = 60.0


class ApisStatedScopeReader:
    """`StatedScopeReader` over dfo-apis' route, through `LiveApisClient`.

    A 404 from the route reads as the route not served only when Search's capabilities
    document is read and lists no live `stated_scope` door, as an apis deployed before
    the route does. That answer is kept for `_DOORS_TTL_S` by `clock`. A 404 the document
    does not explain, such as a wrong or retired `DFO_APIS_URL` answers, and a document
    that cannot be read, raise the route's own `LiveApisError`. The reader logs a warning
    when it first finds the route not served, and again only after a read has succeeded
    since, so a process reading an apis without the route logs it once."""

    def __init__(
        self, client: LiveApisClient, *, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self._client = client
        self._clock = clock
        self._served: bool | None = None
        self._served_expires = float("-inf")
        self._not_served_logged = False

    async def __call__(self, texts: Sequence[str]) -> Sequence[TextReading]:
        """Search's readings of `texts`, in order."""
        try:
            readings = await self._client.read_stated_scopes(texts=texts)
        except LiveApisError as exc:
            if exc.status != _NOT_SERVED_STATUS or await self._route_served():
                raise
            if not self._not_served_logged:
                self._not_served_logged = True
                logger.warning(
                    "stated scope not served: apis answered the route %s and lists no %s "
                    "door; plans are admitted with no stated scope until it serves one",
                    exc.status,
                    _STATED_SCOPE_DOOR,
                )
            raise StatedScopeNotServedError from exc
        self._not_served_logged = False
        return [TextReading.of(reading) for reading in readings]

    async def _route_served(self) -> bool:
        """Whether Search's capabilities document lists a live `stated_scope` door, or
        cannot be read, which reads as served so the route's failure stands."""
        if self._clock() < self._served_expires and self._served is not None:
            return self._served
        try:
            doors = await self._client.read_search_doors()
        except LiveApisError as exc:
            logger.error(
                "stated scope route answered 404 and the search capabilities document "
                "could not be read: status=%s code=%s",
                exc.status,
                exc.code,
            )
            return True
        self._served = _STATED_SCOPE_DOOR in doors
        self._served_expires = self._clock() + _DOORS_TTL_S
        return self._served

    async def aclose(self) -> None:
        """Release the client's connection pool."""
        await self._client.aclose()


# How long one read of the route may take, per attempt, and how many attempts it
# makes. The route reads words with no model or graph call, so a slow answer is a
# fault, and the read runs inside a `propose_plan` call.
_READ_TIMEOUT_S = 10.0
_READ_ATTEMPTS = 2


def apis_stated_scope_reader(*, base_url: str) -> ApisStatedScopeReader:
    """The reader of dfo-apis at `base_url` (`DFO_APIS_URL`), with the Cloud Run ID
    token `common.apis_auth.apis_auth_provider` mints."""
    return ApisStatedScopeReader(
        LiveApisClient(
            base_url=base_url,
            auth=apis_auth_provider(base_url=base_url),
            read_timeout_s=_READ_TIMEOUT_S,
            max_attempts=_READ_ATTEMPTS,
        )
    )


def stated_scope_record(state: SessionState) -> StatedScopeRecord | None:
    """The turn's words and the readings held (`STATED_SCOPE_STATE_KEY`); `None` where
    no reading of the turn's words is held."""
    raw = state.get(STATED_SCOPE_STATE_KEY)
    return StatedScopeRecord.model_validate(raw) if raw else None


def _turn_words(words: str | None) -> str | None:
    """The turn's words as the reader takes them: the last `_MAX_TEXT_CHARS`
    characters, where a long message and its amendments run past them, as Search
    reads a message from its end; `None` for no words."""
    if words is None or not words.strip():
        return None
    return words[-_MAX_TEXT_CHARS:]


def proposal_measures(state: SessionState, arguments: Mapping[str, Any]) -> list[str]:
    """The `measure` of every dataset requirement `arguments`, a `propose_plan` call's,
    states, and of every one the plan on record states, which a patch keeps.

    The arguments arrive unvalidated, so any requirement that is no mapping with a
    `measure` of one to `MAX_MEASURE_CHARS` characters states none here: admission
    validates the proposal before it reads a measure, and answers its validation error
    in place of a refusal of the read."""
    requirements = arguments.get("evidence_requirements")
    measures = [
        measure
        for requirement in (requirements if isinstance(requirements, list) else ())
        if isinstance(requirement, Mapping)
        and requirement.get("kind") == "dataset"
        and isinstance(measure := requirement.get("measure"), str)
        and 1 <= len(measure) <= MAX_MEASURE_CHARS
    ]
    plan = load_plan(state)
    if plan is not None:
        measures.extend(
            requirement.measure
            for requirement in plan.evidence_requirements
            if isinstance(requirement, DatasetRequirement)
        )
    return measures


async def read_stated_scope(
    state: SessionState, reader: StatedScopeReader, *, measures: Iterable[str]
) -> dict[str, object] | None:
    """Read the turn's words and `measures` with `reader`, each text once a turn,
    and hold the readings (`STATED_SCOPE_STATE_KEY`).

    A turn with no words of the user's holds nothing, so admission applies no stated
    scope. An apis that serves no stated-scope route (`StatedScopeNotServedError`) reads
    no text: a reading of the turn's words held from earlier in the turn stays, each text
    asked for reads as stating nothing, and with no such reading nothing is held. Returns
    `None` once every reading is held. A read Search did not answer, after the reader's
    own retries, holds nothing new and returns the `STATED_SCOPE_UNAVAILABLE` refusal
    where the failure can pass (`LiveApisError.retryable`) and the
    `STATED_SCOPE_UNREADABLE` refusal where it cannot. Both end the entry.
    """
    words = _turn_words(current_user_message.get())
    if words is None:
        state[STATED_SCOPE_STATE_KEY] = None
        return None
    held = stated_scope_record(state)
    readings = dict(held.readings) if held is not None else {}
    wanted = list(dict.fromkeys(text for text in (words, *measures) if text not in readings))
    try:
        for start in range(0, len(wanted), _MAX_TEXTS):
            texts = wanted[start : start + _MAX_TEXTS]
            read = await reader(texts)
            readings.update(zip(texts, read, strict=True))
    except StatedScopeNotServedError:
        if words not in readings:
            state[STATED_SCOPE_STATE_KEY] = None
            return None
        readings.update((text, TextReading()) for text in wanted if text not in readings)
    except LiveApisError as exc:
        logger.warning(
            "stated scope not read: status=%s code=%s retryable=%s texts=%d",
            exc.status,
            exc.code,
            exc.retryable,
            len(wanted),
        )
        if exc.retryable:
            return _stated_scope_unavailable(status=exc.status, code=exc.code)
        return _stated_scope_unreadable(status=exc.status, code=exc.code)
    state[STATED_SCOPE_STATE_KEY] = StatedScopeRecord(words=words, readings=readings).model_dump(
        mode="json"
    )
    return None


def stated_scope_unread() -> dict[str, object]:
    """The `SURFACE_UNAVAILABLE` problem for a text admission needs that no reading is
    held for (`submission.service._stated_reading`). The same proposal reads it first,
    so it can succeed later."""
    return problem(
        ProblemCode.SURFACE_UNAVAILABLE,
        "Whose spending the user's words ask for could not be read from Search just now, so "
        "nothing was recorded; propose the plan again.",
        outcome=Outcome.REFUSED,
        detail={"read": "stated_scope"},
    )


def _stated_scope_unreadable(*, status: int | None, code: str | None) -> dict[str, object]:
    """The `STATED_SCOPE_UNREADABLE` problem for a read Search refused in a way no retry
    changes. It ends the entry (`harness.sentences.TERMINAL_PROPOSE_CODES`), since every
    proposal of the turn would be refused the same way. `detail` names the read and
    Search's answer for the logs."""
    return problem(
        ProblemCode.STATED_SCOPE_UNREADABLE,
        "Whose spending the user's words ask for cannot be read from Search, and proposing "
        "again fails the same way, so nothing was recorded.",
        outcome=Outcome.REFUSED,
        detail={"read": "stated_scope", "status": status, "code": code},
    )


def _stated_scope_unavailable(*, status: int | None, code: str | None) -> dict[str, object]:
    """The `STATED_SCOPE_UNAVAILABLE` problem for a read that failed in a way that can
    pass, after the reader's own retries. It ends the entry
    (`harness.sentences.TERMINAL_PROPOSE_CODES`), and its sentence asks the user to try
    again shortly. `detail` names the read and Search's answer for the logs."""
    return problem(
        ProblemCode.STATED_SCOPE_UNAVAILABLE,
        "Whose spending the user's words ask for could not be read from Search, which did "
        "not answer after a retry, so nothing was recorded.",
        outcome=Outcome.REFUSED,
        detail={"read": "stated_scope", "status": status, "code": code},
    )


def stated_compile_unreachable(*, spec_id: str, option_ids: Sequence[str]) -> dict[str, object]:
    """The `STATED_COMPILE_UNREACHABLE` problem for a compile of `spec_id` on
    `option_ids`, which carry a provider option the user's words state, that cannot
    reach Search for good (`planner.source_reads.compile_on_basis`). It is a fault of
    the system, no answer about the pin, so nothing is recorded against the option
    and the user is asked nothing; it ends the entry
    (`harness.sentences.TERMINAL_PROPOSE_CODES`). `detail` names the compile for the
    logs."""
    return problem(
        ProblemCode.STATED_COMPILE_UNREACHABLE,
        f"Search cannot be reached to compile {spec_id} the way the user's words ask, and "
        "proposing again fails the same way, so nothing was recorded.",
        outcome=Outcome.REFUSED,
        detail={"compile": "stated_scope", "spec_id": spec_id, "option_ids": list(option_ids)},
    )


type Pair = tuple[str, str]


class KeepQuestion(StrictModel):
    """Where the question whether to keep a dataset counted as published stands this
    turn, each entry a pin and the provider option it cannot count.

    `offered` are the pairs a `STATED_PROVIDER_DROPPED` refusal listed the datasets on
    record serving for, which it does once (`record_keep_offered`). `asked_call_id` is
    the `propose_plan` call whose pause puts the question about `asked` to the user
    (`record_keep_asked`, `planner.tools.pause_on_keep_question`), and `proposal` that
    call's arguments, which the entry the answer resumes shows the planner. `kept` and
    `released` are the pairs the user answered: kept as published, or not
    (`record_keep_answer`). Code writes every field; admission reads them."""

    offered: tuple[Pair, ...] = ()
    asked_call_id: str | None = None
    asked: tuple[Pair, ...] = ()
    proposal: dict[str, Any] | None = None
    kept: tuple[Pair, ...] = ()
    released: tuple[Pair, ...] = ()


class KeepAnswer(NamedTuple):
    """The user's answer to the keep question (`record_keep_answer`): the pairs it
    asked about, whether they were kept as published, and the arguments of the
    `propose_plan` call it paused."""

    pairs: tuple[Pair, ...]
    kept: bool
    proposal: Mapping[str, Any] | None


def keep_question(state: SessionState) -> KeepQuestion:
    """Where the keep question stands this turn (`KEEP_STATE_KEY`)."""
    held = state.get(KEEP_STATE_KEY)
    return KeepQuestion.model_validate(held) if held else KeepQuestion()


def _save_keep_question(state: SessionState, question: KeepQuestion) -> None:
    state[KEEP_STATE_KEY] = question.model_dump(mode="json")


def _with(pairs: Iterable[Pair], more: Iterable[Pair]) -> tuple[Pair, ...]:
    return tuple(sorted({*pairs, *more}))


def record_keep_offered(state: SessionState, pairs: Iterable[Pair]) -> None:
    """Record that a refusal listed the datasets on record serving `pairs`."""
    question = keep_question(state)
    _save_keep_question(
        state, question.model_copy(update={"offered": _with(question.offered, pairs)})
    )


def record_keep_asked(
    state: SessionState,
    *,
    call_id: str,
    pairs: Iterable[Pair],
    proposal: Mapping[str, Any],
) -> None:
    """Record that the pause of `propose_plan` call `call_id`, whose arguments are
    `proposal`, asks the user about `pairs`."""
    _save_keep_question(
        state,
        keep_question(state).model_copy(
            update={
                "asked_call_id": call_id,
                "asked": tuple(sorted(set(pairs))),
                "proposal": dict(proposal),
            }
        ),
    )


def record_keep_answer(state: SessionState, *, call_id: str, keep: bool) -> KeepAnswer | None:
    """Record the user's answer to the keep question the pause of `call_id` asked:
    the pairs it asked about are kept when `keep`, else released. Returns the answer
    where `call_id` asked it; an answer to any other ask records nothing and returns
    `None`."""
    question = keep_question(state)
    if question.asked_call_id is None or question.asked_call_id != call_id:
        return None
    asked = set(question.asked)
    kept = (set(question.kept) | asked) if keep else set(question.kept) - asked
    released = set(question.released) - asked if keep else set(question.released) | asked
    _save_keep_question(
        state,
        question.model_copy(
            update={
                "asked_call_id": None,
                "asked": (),
                "proposal": None,
                "kept": tuple(sorted(kept)),
                "released": tuple(sorted(released)),
            }
        ),
    )
    return KeepAnswer(pairs=question.asked, kept=keep, proposal=question.proposal)


type RefusedRequirement = tuple[str, str]


def unattributed_refused(
    state: SessionState,
) -> dict[str, frozenset[RefusedRequirement]]:
    """By provider option, the dataset requirements a `STATED_SCOPE_UNATTRIBUTED`
    refusal was given for this turn whose pins offer the option, each its id and
    `measure` (`UNATTRIBUTED_STATE_KEY`)."""
    held = state.get(UNATTRIBUTED_STATE_KEY) or {}
    return {
        str(option_id): frozenset(
            (str(requirement_id), str(measure)) for requirement_id, measure in refused
        )
        for option_id, refused in held.items()
    }


def record_unattributed(
    state: SessionState, option_id: str, refused: Iterable[RefusedRequirement]
) -> None:
    """Record a `STATED_SCOPE_UNATTRIBUTED` refusal given for `option_id` to a plan
    whose dataset requirements pinned to a dataset offering it are `refused`, in place
    of any given for it before."""
    held = {
        option: sorted(entries, key=str) for option, entries in unattributed_refused(state).items()
    }
    held[option_id] = sorted(refused, key=str)
    state[UNATTRIBUTED_STATE_KEY] = {
        option: [list(entry) for entry in entries] for option, entries in held.items()
    }


def stated_uncompiled(state: SessionState) -> dict[tuple[str, str], str]:
    """Search's answer refusing each pin it would not compile on a stated provider
    option, by pin and option (`UNCOMPILED_STATE_KEY`)."""
    return {
        (str(spec_id), str(option_id)): str(cause)
        for spec_id, option_id, cause in state.get(UNCOMPILED_STATE_KEY) or ()
    }


class CombinedRefusal(NamedTuple):
    """Search's refusal for good of a compile moving a stated provider option and a
    calculation's axis together: its code, the refusal's reason, and how many times
    Search did not answer the compile of the stated option alone since."""

    code: str
    reason: str
    unanswered: int = 0


def combined_refused(state: SessionState) -> dict[tuple[str, tuple[str, ...]], CombinedRefusal]:
    """Each compile of a pin on a stated provider option and a calculation's axis
    together that Search refused for good this turn, by pin and option ids
    (`COMBINED_REFUSED_STATE_KEY`)."""
    return {
        (str(spec_id), tuple(str(o) for o in option_ids)): CombinedRefusal(
            str(code), str(reason), int(unanswered)
        )
        for spec_id, option_ids, code, reason, unanswered in state.get(COMBINED_REFUSED_STATE_KEY)
        or ()
    }


def record_combined_refused(
    state: SessionState, *, spec_id: str, option_ids: Sequence[str], refusal: CombinedRefusal
) -> None:
    """Record that Search refused for good to compile `spec_id` on `option_ids`, so a
    later proposal compiles the stated option alone (`planner.source_reads.compile_on_basis`)."""
    held = {**combined_refused(state), (spec_id, tuple(option_ids)): refusal}
    state[COMBINED_REFUSED_STATE_KEY] = [
        [pin, list(options), *held_refusal] for (pin, options), held_refusal in sorted(held.items())
    ]


def record_stated_uncompiled(
    state: SessionState, *, spec_id: str, option_id: str, cause: str
) -> None:
    """Record that Search refused to compile `spec_id` on `option_id`, answering
    `cause`, so admission refuses the pin as one that cannot count the option
    (`submission.service._stated_claims`)."""
    held = {**stated_uncompiled(state), (spec_id, option_id): cause}
    state[UNCOMPILED_STATE_KEY] = [
        [pin, option, reason] for (pin, option), reason in sorted(held.items())
    ]
