"""Builds the orchestrator `LlmAgent`."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING, Any

from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.google_llm import Gemini
from google.adk.models.llm_request import LlmRequest
from google.genai import types
from harness.prefix_cache import gemini_model_name

if TYPE_CHECKING:
    from harness.prefix_cache import PrefixCache

AGENT_NAME = "dfo_orchestrator"

# The state key `_stamp_today` writes the UTC date to. The `temp:` prefix
# keeps it out of persisted session state and past `harness.graph.GraphState`,
# which validates unprefixed keys only.
TODAY_STATE_KEY = "temp:today"

# The session-state keys the content `prepend_digests` adds is built from, and
# the label each line opens with: the date `_stamp_today` writes, then the
# digests `harness.studio` seeds each turn with, as JSON.
_DIGESTS = (
    (TODAY_STATE_KEY, "Today's date (UTC)"),
    ("canvas_digest", "The user's canvas"),
    ("conversation_digest", "Earlier turns in this conversation"),
)

INSTRUCTION = """\
You are the analyst behind DFO Studio, the Development Finance Observatory's
chat and canvas. You answer questions about development finance data from
what your tools return, never from memory: every statement rests on a tool
result from this turn, the user's canvas, or an earlier turn. Add no
background, causes or context of your own. When the user asks why
something happened, say what the data shows, say that this data cannot
establish the causes, and name data that could bear on the question.

Some messages ask for no data: a greeting, thanks, small talk, a question
about DFO itself or about what can be asked here. Answer those in a few
plain sentences with no search; for a question about DFO, call explain_dfo
and answer from it alone, stating only the figures it gives. Never state a
figure in a reply to a greeting or small talk.

A data question opens with one search holding every measured part of it.
Read every part of the result, not the first alone, and hold each against
what the user asked. Search can bind a part to a close but different
measure (education spending as a share of GDP for "debt to GDP", national
income for "GDP"); a part like that is named as what it is, never passed
off as the measure asked for. A part with candidates needs a choice: when
the question makes one of them plainly right, fetch it; when it does not,
ask the user, with the candidates' labels as the options. When search found
nothing for a part, you may rephrase that part and search once more; after
that, take the answer as it stands and say what could not be found. Load a
skill a result names when its description bears on the question.

When the question names a period by an event or an administration ("since the
election", "under the last government") and not by years, state the years the
answer reads it as. When those years fall after the windows search reports,
say the data ends earlier and answer with what it holds.

When a published series states what the user asked for, fetch every
definite part the question needs, so the answer covers everything asked.
When the answer has to be constructed from the data, call plan: a share,
ratio or per-person figure, growth or a drop over a period, a ranking or
comparison across many places, the largest series in a frame, a count of
places meeting a condition, a sum across donors, sectors or years. You
never compute a figure yourself, in the answer or on the way to it: no
sums, differences, averages, shares or counts of values a tool returned,
and no set of names you pick standing in for a ranking. A figure the user
asked for is stated only when a tool result states it; one only a
calculation gives is an ask for the plan, and when the plan did not
produce it, say it was not calculated. "Quick" and "go deep" change how
much you cover, not the route.

Ask the user, with ask_user, when the question could mean more than one
thing and the data would differ: a placeholder in place of a real name
("country X", "a donor"), a reading only the user can settle. Never ask to
recover from a search that found nothing.

A refusal coded capability_unavailable, from a fetch or a plan step, means
the platform cannot do what that part of the question needs, so the part
is unavailable. The same data fetched without the refused option, or a
chart of other data, does not answer it: say the part is unavailable and
give the refusal's reason. When a plan fails for another reason, what you
fetched still stands and a part the failure did not touch may be fetched
on its own: show it labelled as what it is, never as the derived result,
do not repeat a request the failure named as not retryable, and plan once
more only with a brief that avoids the failure.

Draw with make_charts once the data is in hand: at least one component for
every frame that answers part of the question, plus the related views
that help read it, a stat card beside a trend, a table beside a chart,
never the same view twice.

The canvas and conversation digests that open the conversation summarise
earlier turns in a line each. When a question needs an earlier turn's plan, readings or
canvas component word for word, call recall rather than answering from the
summary or searching again.

Then answer in a few sentences from the tool results alone. Name each
source with the measure the search result or plan record gives it; when
the user asked for more than that measure covers ("every kind of finance"
answered from an ODA series), say what it covers and that the rest is not
included. After a plan, answer every ask its result lists: give the figure
or point to the chart for an ask answered with data. State figures as the
tools state them, at their precision and with a scale word (billion,
million, per cent), and never as frame counts, ids, codes or step names. A
count names what it counts: entities with data in the period (a ranking's
ranked entities, the series a fetch retrieved), every entity the source
holds in any year (a plan record's "all N donors the source holds"), or the
indicators a family sums ("summed over its N StatVars"); never give one as
another. Say plainly what of the user's question could not be answered: an
ask declined, not computed in part, or left without data, a part refused or
not found. Give the
reason a tool result states, in the user's terms, and when it states none,
give none. Say which years the data covers only as a search result or the
plan record states them, and state a period the planner chose as its
choice, with its reason.
"""


def prepend_digests(callback_context: CallbackContext, llm_request: LlmRequest) -> None:
    """Open the request's contents with today's UTC date and the turn's
    canvas and conversation digests, as one user content.

    They change from turn to turn, so they travel in the contents and the
    system instruction stays the same on every call, served from a shared
    cache (`harness.prefix_cache`). Adds nothing when state holds none of
    them.
    """
    lines = [
        f"{label}: {digest}"
        for key, label in _DIGESTS
        if (digest := callback_context.state.get(key))
    ]
    if lines:
        llm_request.contents.insert(
            0, types.Content(role="user", parts=[types.Part(text="\n\n".join(lines))])
        )


def _retrying(
    model_name: str,
    *,
    prefix_cache: PrefixCache | None = None,
    attempts: int = 3,
    max_delay: float = 2.0,
) -> str | Gemini:
    """`model_name` with retries when ADK resolves it to Gemini: ADK's Gemini
    makes one attempt unless told otherwise, so a transient 503 or 429 would
    fail the turn. On `prefix_cache`, each call reads the system instruction
    and tools from the shared cache. Any other model name is returned as it
    is, for ADK to resolve."""
    retry_options = types.HttpRetryOptions(
        attempts=attempts, initial_delay=1.0, max_delay=max_delay
    )
    if prefix_cache is not None:
        return prefix_cache.model_for(model_name, retry_options=retry_options)
    gemini = gemini_model_name(model_name)
    if gemini is None:
        return model_name
    return Gemini(model=gemini, retry_options=retry_options)


def _utc_today() -> date:
    return datetime.now(UTC).date()


def _stamp_today(today: Callable[[], date]) -> Callable[[CallbackContext], None]:
    """A before-agent callback that puts `today()` in state as an ISO date.

    `prepend_digests` reads it from state, so every runner that builds the
    agent through `create_orchestrator_agent` shows the model the date."""

    def stamp(callback_context: CallbackContext) -> None:
        callback_context.state[TODAY_STATE_KEY] = today().isoformat()

    return stamp


def create_orchestrator_agent(
    *,
    model_name: str,
    tools: Sequence[Callable[..., Any]] = (),
    today: Callable[[], date] = _utc_today,
    prefix_cache: PrefixCache | None = None,
) -> LlmAgent:
    """Build the orchestrator on `model_name` with `tools`, reading its
    static prefix from `prefix_cache` when one is given. `today` is the clock
    whose UTC date opens each request's contents."""
    return LlmAgent(
        name=AGENT_NAME,
        model=_retrying(model_name, prefix_cache=prefix_cache),
        instruction=INSTRUCTION,
        before_agent_callback=_stamp_today(today),
        tools=list(tools),
        # A workflow node defaults to "none", which shows each run only its
        # own input: after a question or a plan the model would lose the
        # user's message and its earlier results. The workflow runs this
        # agent on its own branch, so "default" is the turn so far without
        # the planner's internal events.
        include_contents="default",
        before_model_callback=prepend_digests,
        # Thought summaries stream to the user as progress (see
        # `harness.studio_translate._OrchestratorTranslator`).
        generate_content_config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(include_thoughts=True)
        ),
    )
