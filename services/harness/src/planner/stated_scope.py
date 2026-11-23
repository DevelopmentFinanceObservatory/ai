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

A read that fails answers `SURFACE_UNAVAILABLE` and records nothing, so `propose_plan` never
admits a plan on a reading it does not hold. A turn with no words of the user's, such as a
chip's or one outside Studio, holds no reading, and admission applies no stated scope.

Three more slots belong to the rule: the datasets the user chose to keep as published
although they cannot count the provider scope their words asked for
(`STATED_SCOPE_KEPT_STATE_KEY`), which the conversation keeps across turns
(`harness.studio_state.ConversationState.stated_scope_kept`), the options a
`STATED_SCOPE_UNATTRIBUTED` refusal was given for (`UNATTRIBUTED_STATE_KEY`), which is
given once a turn, and the pins Search refused to compile on a stated option
(`UNCOMPILED_STATE_KEY`).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Protocol

from common.apis_auth import apis_auth_provider
from common.live_apis import LiveApisClient, LiveApisError, StatedScopeReading
from harness.conversation import current_user_message
from harness.scope_choices import PROVIDER_AXIS

from planner.plan import DatasetRequirement, SessionState, load_plan
from planner.records import StrictModel
from planner.tool_results import Outcome, ProblemCode, problem

logger = logging.getLogger(__name__)

# The session-state slot holding the turn's words and Search's reading of each
# text read this turn (`StatedScopeRecord`).
STATED_SCOPE_STATE_KEY = "planner:stated_scope"

# The session-state slot holding each dataset the user chose to keep as published
# although it cannot count the provider option their words asked for, as
# `[spec_id, option_id]` pairs (`stated_scope_kept`).
STATED_SCOPE_KEPT_STATE_KEY = "planner:stated_scope_kept"

# The session-state slot holding the provider options a `STATED_SCOPE_UNATTRIBUTED`
# refusal was given for this turn (`unattributed_refused`).
UNATTRIBUTED_STATE_KEY = "planner:stated_scope_unattributed"

# The session-state slot holding each pin Search refused to compile on the provider
# option a requirement's `measure` states, with that option and Search's answer, as
# `[spec_id, option_id, cause]` (`stated_uncompiled`).
UNCOMPILED_STATE_KEY = "planner:stated_scope_uncompiled"

# The most texts and characters per text Search's reader takes in one call
# (`StatedScopeRequest` in dfo-apis).
_MAX_TEXTS = 64
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


class StatedScopeReader(Protocol):
    """Search's reader of the scope words of `texts`, one reading per text, in order.

    Raises `LiveApisError` when Search does not answer."""

    async def __call__(self, texts: Sequence[str]) -> Sequence[TextReading]: ...


class ApisStatedScopeReader:
    """`StatedScopeReader` over dfo-apis' route, through `LiveApisClient`."""

    def __init__(self, client: LiveApisClient) -> None:
        self._client = client

    async def __call__(self, texts: Sequence[str]) -> Sequence[TextReading]:
        """Search's readings of `texts`, in order."""
        readings = await self._client.read_stated_scopes(texts=texts)
        return [TextReading.of(reading) for reading in readings]


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
    string `measure` states none here; admission refuses such a proposal before it
    reads a measure."""
    requirements = arguments.get("evidence_requirements")
    measures = [
        measure
        for requirement in (requirements if isinstance(requirements, list) else ())
        if isinstance(requirement, Mapping)
        and requirement.get("kind") == "dataset"
        and isinstance(measure := requirement.get("measure"), str)
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
    scope. Returns `None` once every reading is held, or the `SURFACE_UNAVAILABLE`
    refusal for a read Search did not answer, which holds nothing new.
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
    except LiveApisError as exc:
        logger.warning("stated scope not read: status=%s code=%s", exc.status, exc.code)
        return _stated_scope_unread(f"Search did not answer ({exc.code or exc.status})")
    state[STATED_SCOPE_STATE_KEY] = StatedScopeRecord(words=words, readings=readings).model_dump(
        mode="json"
    )
    return None


def _stated_scope_unread(cause: str) -> dict[str, object]:
    """The `SURFACE_UNAVAILABLE` problem for a read of the provider scope the user's
    words state that did not complete. The same proposal can succeed later."""
    return problem(
        ProblemCode.SURFACE_UNAVAILABLE,
        f"Whose spending the user's words ask for could not be read (the stated-scope "
        f"read): {cause}. The plan is sound and nothing was recorded; propose it again.",
        outcome=Outcome.REFUSED,
        detail={"read": "stated_scope"},
    )


def stated_scope_unread() -> dict[str, object]:
    """`_stated_scope_unread` for a text admission needs that no reading is held for."""
    return _stated_scope_unread("no reading is held for a requirement's `measure`")


def stated_scope_kept(state: SessionState | Mapping[str, Any]) -> frozenset[tuple[str, str]]:
    """Each dataset the user kept as published, with the provider option it cannot
    count (`STATED_SCOPE_KEPT_STATE_KEY`)."""
    return frozenset(
        (str(spec_id), str(option_id))
        for spec_id, option_id in state.get(STATED_SCOPE_KEPT_STATE_KEY) or ()
    )


def save_stated_scope_kept(state: SessionState, kept: Iterable[tuple[str, str]]) -> None:
    """Hold `kept` in `STATED_SCOPE_KEPT_STATE_KEY` in place of what it held, so a
    keep the user released is gone."""
    state[STATED_SCOPE_KEPT_STATE_KEY] = [list(pair) for pair in sorted(set(kept))]


def unattributed_refused(state: SessionState) -> frozenset[str]:
    """The provider options a `STATED_SCOPE_UNATTRIBUTED` refusal was given for this
    turn (`UNATTRIBUTED_STATE_KEY`)."""
    return frozenset(state.get(UNATTRIBUTED_STATE_KEY) or ())


def record_unattributed(state: SessionState, option_id: str) -> None:
    """Record a `STATED_SCOPE_UNATTRIBUTED` refusal given for `option_id`."""
    state[UNATTRIBUTED_STATE_KEY] = sorted(unattributed_refused(state) | {option_id})


def stated_uncompiled(state: SessionState) -> dict[tuple[str, str], str]:
    """Search's answer refusing each pin it would not compile on a stated provider
    option, by pin and option (`UNCOMPILED_STATE_KEY`)."""
    return {
        (str(spec_id), str(option_id)): str(cause)
        for spec_id, option_id, cause in state.get(UNCOMPILED_STATE_KEY) or ()
    }


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
