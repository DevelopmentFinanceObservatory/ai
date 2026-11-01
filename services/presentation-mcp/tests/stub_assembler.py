"""An in-memory `Assembler` test double, driven by a release document.

State survives between calls on one instance; it dies with the process. No
vendored grammar file backs validation -- the constraints come straight from
a release document (`intent_schemas_tolerant`, `minimal_intents`), the same
shape `server.py`'s lifespan fetches from the real platform. A create is
checked for completeness against `minimal_intents[type]` (the tolerant
schemas admit partial payloads on purpose, so they cannot themselves catch a
missing field), and a create or a merged edit is checked for its bounds
against `intent_schemas_tolerant[type]`. Binding checks against the frame,
the data-judged rules, fact summaries, and text projections are the real
assembler's work, out of scope here. Each revision carries a
`resolved_payload` envelope, the dfo-spec-kit catalogue fixture for the
intent's type (`resolved_revision`), restamped with the revision's ids and
frame, and with the intent's title where it sets one. The data in it is the
fixture's, whatever the intent selects. A create mints a write token,
`cmw_<n>`, and a revise naming any other token is refused `write_refused`,
as the platform refuses it.

`aclose()` is a no-op, present only so a test can substitute an instance of
this class wherever `server.py`'s lifespan expects a `PlatformAssembler`.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import best_match
from presentation_mcp.platform_contract import Caller, Refusal, RetryMode, WriteReply

_FIXTURES = Path(__file__).parent / "fixtures"


def resolved_revision(member: str) -> dict[str, Any]:
    """A catalogue member's resolved envelope, dfo-spec-kit's
    `catalogue/fixtures/<member>/payload.json` at 6e9b6735, copied verbatim
    into `fixtures/resolved_revision/`. `<member>_composed_unit` names the
    member's `payload_composed_unit.json` at bf584e3, copied the same way.
    `line_chart_description` and `note_narration_lineage` are
    `line_chart/payload_description.json` and `note_narration/payload_lineage.json`
    at 1f73bc6. `source_card_description` is `source_card` with lineage alone
    in its provenance, the `description-source-card` description vector at
    1f73bc6 as its `description`, and that description's source refs as its
    `source_refs`."""
    path = _FIXTURES / "resolved_revision" / f"{member}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_release() -> dict[str, Any]:
    """The captured release document tests build a `StubAssembler` from by default."""
    return json.loads((_FIXTURES / "release.json").read_text(encoding="utf-8"))


# The named rule no schema can express: these fields may not appear in a
# patch at all, same value or not. A different kind of chart, or the same
# chart of different data, is a new component.
_UNEDITABLE_FIELDS = ("type", "frame_ref")

# The one frame the stub knows: health ODA by recipient country, annual,
# with 2018 missing. Metadata only: nothing downstream reads observations.
_FRAME: dict[str, Any] = {
    "spec_id": "spec_7c2f91",
    "content_version": "cv_4a91e2",
    "dimensions": {
        "statvar": ["health_oda"],
        "recipient": [
            "country/COD",
            "country/ETH",
            "country/GHA",
            "country/KEN",
            "country/MOZ",
            "country/NGA",
            "country/TZA",
            "country/UGA",
        ],
        "prices": ["constant", "current"],
    },
    "periods": ["2015", "2016", "2017", "2019", "2020", "2021", "2022", "2023", "2024"],
}


def _merge_patch(target: Any, patch: Any) -> Any:
    """JSON merge-patch (RFC 7386): objects merge, null deletes, the rest replaces.

    The tool advertises these semantics, so the stub must honour them: a patch
    like `{"window": {"end": null}}` clears one field and keeps its siblings.
    """
    if not isinstance(patch, Mapping):
        return patch
    merged = dict(target) if isinstance(target, Mapping) else {}
    for field, value in patch.items():
        if value is None:
            merged.pop(field, None)
        else:
            merged[field] = _merge_patch(merged.get(field), value)
    return merged


def _not_found(missing: str) -> Refusal:
    """The refusal for an id nothing minted.

    Args:
        missing: Names the thing that was asked for and does not exist,
            ready for display, e.g. "Component 'cmp_9'". It becomes the
            start of the refusal's `detail`.
    """
    return Refusal(
        error="not_found",
        retry=RetryMode.CORRECT_AND_RETRY,
        detail=f"{missing} is not held by this assembler.",
        fix="Use an id a previous reply returned.",
    )


@dataclass(frozen=True, kw_only=True)
class _Revision:
    """One stored revision of a component's intent, and its envelope."""

    revision: int
    intent: Mapping[str, Any]
    resolved_payload: Mapping[str, Any]


class StubAssembler:
    """A process-local `Assembler`, validated against a release document.

    Ids and write tokens are counters so a transcript reads plainly. A
    component answers any describe that names its id, and a revise that
    names its id and write token. An id it never minted is refused
    `not_found`. `caller` is accepted and ignored.
    The methods never await, so no lock is needed.
    """

    def __init__(self, *, release: Mapping[str, Any] | None = None) -> None:
        self._release = dict(release) if release is not None else load_release()
        self._components: dict[str, list[_Revision]] = {}
        self._write_tokens: dict[str, str] = {}
        self._minted_components = 0

    async def aclose(self) -> None:
        """No-op: nothing here owns a connection to release."""

    def _revision(
        self, *, component_id: str, revision: int, intent: Mapping[str, Any]
    ) -> _Revision:
        """A stored revision, stamped with its type's fixture envelope."""
        frame_ref = intent.get("frame_ref")
        envelope = resolved_revision(intent["type"])
        envelope["catalog_version"] = self._release["catalog_version"]
        envelope["component_id"] = component_id
        envelope["revision"] = revision
        envelope["provenance"]["inputs"] = [dict(frame_ref)] if frame_ref else []
        if isinstance(intent.get("title"), str) and "title" in envelope["component"]:
            envelope["component"]["title"]["text"] = intent["title"]
        return _Revision(revision=revision, intent=dict(intent), resolved_payload=envelope)

    def _validate_create(self, intent: Mapping[str, Any]) -> Refusal | None:
        """Check a create's completeness against `minimal_intents`, then its bounds.

        Args:
            intent: The full payload of a create.

        Returns:
            `None` when the intent passes both checks. Otherwise a
            `schema_invalid` refusal: missing-field detail names the minimal
            intent for that type, and a bound violation carries the deepest
            matching validation error.
        """
        component_type = intent.get("type")
        minimal_intents = self._release["minimal_intents"]
        minimal = minimal_intents.get(component_type)
        if minimal is None:
            return Refusal(
                error="schema_invalid",
                retry=RetryMode.CORRECT_AND_RETRY,
                detail=f"{component_type!r} is not a known component type.",
                pointer="/type",
            )
        missing = [key for key in minimal if key not in intent]
        if missing:
            return Refusal(
                error="schema_invalid",
                retry=RetryMode.CORRECT_AND_RETRY,
                detail=f"missing {', '.join(missing)}. A minimal {component_type} looks like {minimal!r}.",
                pointer=f"/{missing[0]}",
            )
        return self._validate_bounds(component_type, intent)

    def _validate_bounds(self, component_type: str, intent: Mapping[str, Any]) -> Refusal | None:
        """Check `intent` against the tolerant schema's bounds for `component_type`."""
        schema = self._release["intent_schemas_tolerant"][component_type]
        validator = Draft202012Validator(schema)
        error = best_match(validator.iter_errors(intent))
        if error is None:
            return None
        pointer = "/" + "/".join(str(part) for part in error.absolute_path)
        return Refusal(
            error="schema_invalid",
            retry=RetryMode.CORRECT_AND_RETRY,
            detail=error.message,
            pointer=pointer if pointer != "/" else None,
        )

    async def create_component(
        self,
        *,
        intent: Mapping[str, Any],
        caller: Caller,
        rebuild_lost_frames: bool = False,
    ) -> WriteReply | Refusal:
        """Check completeness and bounds, then store the intent as revision 1.
        The stub holds every frame, so `rebuild_lost_frames` changes nothing."""
        refusal = self._validate_create(intent)
        if refusal is not None:
            return refusal
        self._minted_components += 1
        component_id = f"cmp_{self._minted_components}"
        stored = self._revision(component_id=component_id, revision=1, intent=intent)
        self._components[component_id] = [stored]
        write_token = f"cmw_{self._minted_components}"
        self._write_tokens[component_id] = write_token
        return WriteReply(
            component_id=component_id,
            revision=1,
            fact_summary={},
            resolved_payload=copy.deepcopy(stored.resolved_payload),
            write_token=write_token,
        )

    async def revise_component(
        self,
        *,
        component_id: str,
        patch: Mapping[str, Any],
        write_token: str,
        caller: Caller,
    ) -> WriteReply | Refusal:
        """Check the write token, merge the patch over the newest revision,
        check its bounds, and append the next one.

        On any refusal nothing is stored, so a failed edit leaves the
        component exactly as it was.
        """
        revisions = self._components.get(component_id)
        if revisions is None:
            return _not_found(f"Component {component_id!r}")
        if write_token != self._write_tokens[component_id]:
            return Refusal(
                error="write_refused",
                retry=RetryMode.DO_NOT_RETRY,
                detail="This write token does not grant writes to this component.",
                fix="Create a new component, starting from describe_component's intent.",
            )
        # Checked on the patch's keys before merging: the merged result would
        # be legal either way, so the schema cannot catch this.
        touched = [field for field in _UNEDITABLE_FIELDS if field in patch]
        if touched:
            return Refusal(
                error="constraint_violation",
                retry=RetryMode.CORRECT_AND_RETRY,
                detail=f"{', '.join(touched)} cannot be edited.",
                fix=(
                    "Drop the field from the patch. A different component type, "
                    "or the same component over different data, is a new component."
                ),
                pointer=f"/{touched[0]}",
            )
        current = revisions[-1]
        merged = _merge_patch(current.intent, patch)
        refusal = self._validate_bounds(merged["type"], merged)
        if refusal is not None:
            return refusal
        revision = current.revision + 1
        stored = self._revision(component_id=component_id, revision=revision, intent=merged)
        revisions.append(stored)
        changed = tuple({"field": field} for field in patch)
        return WriteReply(
            component_id=component_id,
            revision=revision,
            fact_summary={},
            changed=changed,
            resolved_payload=copy.deepcopy(stored.resolved_payload),
        )

    async def get_frame_view(
        self,
        *,
        frame_ref: Mapping[str, Any],
        dimension: str | None = None,
        query: str | None = None,
        rebuild_lost_frames: bool = False,
    ) -> Mapping[str, Any] | Refusal:
        """The built-in frame, whole. `frame_ref`, the narrowing and
        `rebuild_lost_frames` are ignored.

        Ignoring `frame_ref` is the bootstrap, not an oversight: no data layer
        exists to hand out a real ref, so any ref answers with the frame and
        its true ids, which the caller then uses in intents.
        """
        # A copy, so a caller mutating the view cannot corrupt the frame.
        return copy.deepcopy(_FRAME)

    async def get_component(
        self,
        *,
        component_id: str,
        revision: int | None = None,
    ) -> Mapping[str, Any] | Refusal:
        """A stored intent, newest by default."""
        revisions = self._components.get(component_id)
        if revisions is None:
            return _not_found(f"Component {component_id!r}")
        stored = (
            revisions[-1]
            if revision is None
            else next((r for r in revisions if r.revision == revision), None)
        )
        if stored is None:
            return _not_found(f"Revision {revision} of {component_id!r}")
        return {
            "component_id": component_id,
            "revision": stored.revision,
            # A copy, so a caller mutating the reply cannot corrupt the history.
            # Deep, because the nested select and frame_ref mappings are what
            # a caller would actually touch.
            "intent": copy.deepcopy(stored.intent),
            "resolved_payload": copy.deepcopy(stored.resolved_payload),
        }

    async def release(self) -> Mapping[str, Any]:
        """The release document this stub was built from."""
        return dict(self._release)
