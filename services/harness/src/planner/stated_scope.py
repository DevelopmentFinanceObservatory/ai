"""The provider scope the user's words state, read by Search, for `propose_plan`'s admission.

Search's reader (`POST /api/search/stated-scope`, `common.live_apis.LiveApisClient.read_stated_scopes`)
reads a text's scope words with no model and no graph: the provider option it states, at
most one, whether it asks for a total, and every provider option it names.
`read_stated_scope` reads the turn's words (`harness.conversation.current_user_message`) once
a turn and holds the reading in session state (`STATED_SCOPE_STATE_KEY`). The planner
declares the provider option each dataset requirement counts
(`planner.plan.DatasetRequirement.scope_options`), and admission compares the declarations
with what is held (`stated_scope_record`, `submission.service._provider_records`), so a figure
counted otherwise than the user's words name reads so on the plan card.

A read that fails records nothing, so `propose_plan` admits a plan only on a reading it
holds. Each read retries on its own (`_READ_ATTEMPTS`), so nothing carries from one read
to the next. A read that still fails ends the entry: on `STATED_SCOPE_UNAVAILABLE`, which
asks the user to try again shortly, where the failure can pass, and on
`STATED_SCOPE_UNREADABLE`, a fault in the system, where it cannot. An apis that serves
no stated-scope route (`StatedScopeNotServedError`), one deployed before the route or
rolled back past it, holds no reading of the turn's words, so admission compares
nothing, as for a turn with no words of the user's, such as a chip's or one outside
Studio. Only Search's capabilities document can say the route is not served, so a 404
from a wrong address ends the entry; `PLANNER_STATED_SCOPE=false` turns the read off.

Two more slots hold for the turn: the pins Search refused to compile on a declared
option (`UNCOMPILED_STATE_KEY`), and the compiles moving a declared option beside a
calculation's axis that Search refused (`COMBINED_REFUSED_STATE_KEY`), which a later
proposal does not post again.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from typing import NamedTuple, Protocol

from common.apis_auth import apis_auth_provider
from common.live_apis import LiveApisClient, LiveApisError, StatedScopeReading
from harness.conversation import current_user_message
from harness.scope_choices import PROVIDER_AXIS

from planner.plan import SessionState
from planner.records import StrictModel
from planner.tool_results import Outcome, ProblemCode, problem

logger = logging.getLogger(__name__)

# The session-state slot holding the turn's words and Search's reading of each
# text read this turn (`StatedScopeRecord`).
STATED_SCOPE_STATE_KEY = "planner:stated_scope"

# The session-state slot holding each pin Search refused to compile on the provider
# option a requirement declares, with that option and Search's answer, as
# `[spec_id, option_id, cause]` (`stated_uncompiled`).
UNCOMPILED_STATE_KEY = "planner:stated_scope_uncompiled"

# The session-state slot holding each compile moving a declared provider option and a
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
        """Whether Search's capabilities document lists a live `stated_scope` door. A
        failure to read the document that can pass is raised, so the read ends on a
        retryable refusal; any other reads as served, so the route's 404 stands."""
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
            if exc.retryable:
                raise
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


async def read_stated_scope(
    state: SessionState, reader: StatedScopeReader
) -> dict[str, object] | None:
    """Read the turn's words with `reader` once a turn, and hold the reading
    (`STATED_SCOPE_STATE_KEY`).

    A turn with no words of the user's holds nothing, so admission compares nothing. An
    apis that serves no stated-scope route (`StatedScopeNotServedError`) holds no
    reading of them either. Returns `None` once the reading is held, or where none is.
    A read Search did not answer, after the reader's own retries, holds nothing new and
    returns the `STATED_SCOPE_UNAVAILABLE` refusal where the failure can pass
    (`LiveApisError.retryable`) and the `STATED_SCOPE_UNREADABLE` refusal where it
    cannot. Both end the entry.
    """
    words = _turn_words(current_user_message.get())
    if words is None:
        state[STATED_SCOPE_STATE_KEY] = None
        return None
    held = stated_scope_record(state)
    readings = dict(held.readings) if held is not None else {}
    wanted = [words] if words not in readings else []
    try:
        for start in range(0, len(wanted), _MAX_TEXTS):
            texts = wanted[start : start + _MAX_TEXTS]
            read = await reader(texts)
            readings.update(zip(texts, read, strict=True))
    except StatedScopeNotServedError:
        state[STATED_SCOPE_STATE_KEY] = None
        return None
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
        "not answer, so nothing was recorded.",
        outcome=Outcome.REFUSED,
        detail={"read": "stated_scope", "status": status, "code": code},
    )


def stated_compile_unreachable(*, spec_id: str, option_ids: Sequence[str]) -> dict[str, object]:
    """The `STATED_COMPILE_UNREACHABLE` problem for a compile of `spec_id` on
    `option_ids`, which carry a provider option its requirement declares, that cannot
    reach Search for good (`planner.source_reads.compile_on_basis`). It is a fault of
    the system, no answer about the pin, so nothing is recorded against the option
    and the user is asked nothing; it ends the entry
    (`harness.sentences.TERMINAL_PROPOSE_CODES`). `detail` names the compile for the
    logs."""
    return problem(
        ProblemCode.STATED_COMPILE_UNREACHABLE,
        f"Search cannot be reached to compile {spec_id} on the option declared for it, and "
        "proposing again fails the same way, so nothing was recorded.",
        outcome=Outcome.REFUSED,
        detail={"compile": "stated_scope", "spec_id": spec_id, "option_ids": list(option_ids)},
    )


def stated_uncompiled(state: SessionState) -> dict[tuple[str, str], str]:
    """Search's answer refusing each pin it would not compile on a declared provider
    option, by pin and option (`UNCOMPILED_STATE_KEY`)."""
    return {
        (str(spec_id), str(option_id)): str(cause)
        for spec_id, option_id, cause in state.get(UNCOMPILED_STATE_KEY) or ()
    }


class CombinedRefusal(NamedTuple):
    """Search's refusal for good of a compile moving a declared provider option and a
    calculation's axis together: its code, the refusal's reason, and how many times
    Search did not answer the compile of the declared option alone since."""

    code: str
    reason: str
    unanswered: int = 0


def combined_refused(state: SessionState) -> dict[tuple[str, tuple[str, ...]], CombinedRefusal]:
    """Each compile of a pin on a declared provider option and a calculation's axis
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
    later proposal compiles the declared option alone (`planner.source_reads.compile_on_basis`)."""
    held = {**combined_refused(state), (spec_id, tuple(option_ids)): refusal}
    state[COMBINED_REFUSED_STATE_KEY] = [
        [pin, list(options), *held_refusal] for (pin, options), held_refusal in sorted(held.items())
    ]


def record_stated_uncompiled(
    state: SessionState, *, spec_id: str, option_id: str, cause: str
) -> None:
    """Record that Search refused to compile `spec_id` on `option_id`, answering
    `cause`, so admission refuses the pin as one that cannot count the option
    (`submission.service._declared_claims`)."""
    held = {**stated_uncompiled(state), (spec_id, option_id): cause}
    state[UNCOMPILED_STATE_KEY] = [
        [pin, option, reason] for (pin, option), reason in sorted(held.items())
    ]
