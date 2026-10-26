"""How Search counted one part: its scope choices, as the harness stores and carries them.

`PartScope` mirrors a definite Spec block's `resolved.scope_choices`, the
record dfo-apis writes for each part (`search.models.scope` there): one
`ScopeChoice` per axis a declared rule or the request settled (sector,
region, measure basis, provider scope), each with its options and their
availability, the explicit and carried `Entry`s applied, and the identity
they compile to. `harness.turn_records.SearchReading.scope` stores it.
Identities stay the JSON documents Search wrote, since a later resubmit
posts them back byte for byte.

`carried_entries` turns the choices a conversation settled into the
entries a later search posts as `carried_entries`. Search attaches each to
a part by axis and concept and reports those it drops in
`PartScope.dropped_entries`; the harness never matches an entry to a part.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Annotated, Any, Literal

from planner.records import StrictModel
from pydantic import Field

ScopeAxis = Literal["sector", "region", "measure_basis", "provider"]
EntryProvenance = Literal["explicit", "carried"]
ChoiceProvenance = Literal["explicit", "carried", "default"]
OptionStatus = Literal["available", "partial", "unavailable"]
ComponentRole = Literal["whole", "bilateral", "imputed", "headline_net", "headline_ge"]
ScopeReasonCode = Literal[
    "provider_not_offered",
    "imputed_coverage",
    "basis_not_published",
    "residuals_need_summed_region",
    "conflicts_with",
    "carried_choice_not_applied",
]

# "<axis>:<value>": `sector:120+130`, `region:members_and_residuals`,
# `measure_basis:gross`, `provider:inclusive`. One option has one id on every part.
OptionId = Annotated[str, Field(pattern=r"^(sector|region|measure_basis|provider):[A-Za-z0-9_+]+$")]

YearSpan = tuple[int, int]


class TextOrigin(StrictModel):
    """The words of the user's message an explicit entry was read from."""

    kind: Literal["text"] = "text"
    text: str
    start: int
    end: int


class ChoiceOrigin(StrictModel):
    """The scope choice an entry was switched or carried from."""

    kind: Literal["choice"] = "choice"
    choice_id: str


class Entry(StrictModel):
    """One explicit or carried entry: an option asked for on one axis.

    `concept` is the concept the entry was made for ("health"), so a carried
    entry attaches only to a part asking for that concept; `None` when the
    axis has no concept.
    """

    axis: ScopeAxis
    option_id: OptionId
    provenance: EntryProvenance
    concept: str | None = None
    origin: Annotated[TextOrigin | ChoiceOrigin, Field(discriminator="kind")]


class SlotBinding(StrictModel):
    """An option that binds one what/how slot of the identity."""

    kind: Literal["slot"] = "slot"
    slot: str
    state: Literal["value", "set"]
    refs: tuple[str, ...]


class EntitiesBinding(StrictModel):
    """An option that adds entities to one role beside its members."""

    kind: Literal["entities"] = "entities"
    role: str
    add: tuple[str, ...]


class ShapeBinding(StrictModel):
    """An option that sets the identity's measure and netting basis."""

    kind: Literal["shape"] = "shape"
    measured_property: str
    netting_basis: str | None = None


class ProviderBinding(StrictModel):
    """An option that sets whose spending the answer counts."""

    kind: Literal["provider"] = "provider"
    scope: Literal["bilateral", "inclusive"]


class OptionAvailability(StrictModel):
    """Whether an option can be served: `years` are the years a partial option
    covers, `code` why it is partial or unavailable, `detail` the code's
    parameters."""

    status: OptionStatus
    years: tuple[YearSpan, ...] | None = None
    code: ScopeReasonCode | None = None
    detail: dict[str, str] = Field(default_factory=dict)


class Option(StrictModel):
    """One value a scope choice can take."""

    option_id: OptionId
    label: str
    binding: Annotated[
        SlotBinding | EntitiesBinding | ShapeBinding | ProviderBinding,
        Field(discriminator="kind"),
    ]
    availability: OptionAvailability


class ScopeChoice(StrictModel):
    """How one axis of the part was counted: `id` is `"<PartScope.id>:<axis>"`,
    `active` the option the compiled identity holds, `provenance` what
    settled it."""

    id: str
    axis: ScopeAxis
    rule: str
    provenance: ChoiceProvenance
    active: OptionId
    options: tuple[Option, ...]

    @property
    def active_option(self) -> Option | None:
        """The option `active` names; `None` when the record lists none by that id."""
        return next((option for option in self.options if option.option_id == self.active), None)


class DroppedEntry(StrictModel):
    """A carried entry Search left out of this part, with its cause."""

    entry: Entry
    code: Literal["carried_choice_not_applied"] = "carried_choice_not_applied"
    cause: str


class ScopeDefinition(StrictModel):
    """The identity a part resolved before its scope compiled, and its spec_id."""

    schema_version: Literal[1] = 1
    spec_id: str
    identity: dict[str, Any]


class Component(StrictModel):
    """One series the compiled identity is served from."""

    role: ComponentRole
    periods: tuple[YearSpan, ...] | None = None
    provenance: str | None = None
    content_version: str | None = None
    identity: dict[str, Any]


class CompiledScope(StrictModel):
    """The identity the active options compile to."""

    spec_id: str
    identity: dict[str, Any]
    components: tuple[Component, ...]


class LedgerStep(StrictModel):
    """One switch, kept so it can be undone with its prior provenance."""

    axis: ScopeAxis
    prior: Entry | None = None
    applied: Entry
    revision: int


class PartScope(StrictModel):
    """Every scope choice of one part, the entries that settled them and the
    identity they compile to. `id` is the spec_id of `base`."""

    id: str
    base: ScopeDefinition
    rule_version: str
    entries: tuple[Entry, ...]
    dropped_entries: tuple[DroppedEntry, ...] = ()
    choices: tuple[ScopeChoice, ...]
    compiled: CompiledScope
    ledger: tuple[LedgerStep, ...] = ()


def carried_entries(scopes: Iterable[PartScope]) -> list[Entry]:
    """The entries a later search carries from `scopes`, oldest first.

    Every choice an explicit or carried entry settled becomes a `carried`
    entry for its active option, with the concept of the entry that settled
    it and a `choice` origin naming the choice. A choice a declared default
    settled carries nothing. The newest choice per axis and concept wins.
    """
    newest: dict[tuple[str, str | None], Entry] = {}
    for scope in scopes:
        for choice in scope.choices:
            if choice.provenance == "default":
                continue
            concept = next(
                (
                    entry.concept
                    for entry in scope.entries
                    if entry.axis == choice.axis and entry.option_id == choice.active
                ),
                None,
            )
            key = (choice.axis, concept)
            newest.pop(key, None)
            newest[key] = Entry(
                axis=choice.axis,
                option_id=choice.active,
                provenance="carried",
                concept=concept,
                origin=ChoiceOrigin(choice_id=choice.id),
            )
    return list(newest.values())
