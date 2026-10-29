"""How Search counted one part: its scope choices, as a stored search reading holds them.

`PartScope` mirrors a definite Spec block's `resolved.scope_choices`, the
record dfo-apis writes for each part (`search.models.scope` there): one
`ScopeChoice` per axis a declared rule or the request settled (sector,
region, measure basis, provider scope), each with its options and their
availability, the explicit and carried entries applied
(`data_mcp.scope.ScopeEntry`), and the identity they compile to.
`ServedScope` is how the figures one retrieval served were counted.
`harness.turn_records.SearchReading.scope` and `FetchReading.served` hold
them on records a later release stored; nothing here writes or reads either.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from data_mcp.scope import ChoiceProvenance, OptionId, ScopeAxis, ScopeEntry, ScopeReasonCode
from planner.records import OptionStatus, StrictModel
from pydantic import Field

# Whose spending a provider option counts: bilateral flows alone, with imputed
# shares of multilateral spending, with core contributions to multilaterals as a
# published donor total such as DAC1's counts them, or the core contributions
# alone.
ProviderScope = Literal["bilateral", "inclusive", "contributions", "multilateral"]

DroppedCode = Literal["carried_choice_not_applied", "stated_choice_not_applied"]

YearSpan = tuple[int, int]


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
    scope: ProviderScope


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


class DroppedEntry(StrictModel):
    """An entry Search left out of this part, with its cause: a carried one
    (`carried_choice_not_applied`) or an option the user's words asked for
    that the part cannot serve (`stated_choice_not_applied`)."""

    entry: ScopeEntry
    code: DroppedCode = "carried_choice_not_applied"
    cause: str


class ScopeDefinition(StrictModel):
    """The identity a part resolved before its scope compiled, and its spec_id."""

    schema_version: Literal[1] = 1
    spec_id: str
    identity: dict[str, Any]


class Component(StrictModel):
    """One series the compiled identity is served from, by its `role` as Search names it."""

    role: str
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
    prior: ScopeEntry | None = None
    applied: ScopeEntry
    revision: int


class PartScope(StrictModel):
    """Every scope choice of one part, the entries that settled them and the
    identity they compile to. `id` is the spec_id of `base`."""

    id: str
    base: ScopeDefinition
    rule_version: str
    entries: tuple[ScopeEntry, ...]
    dropped_entries: tuple[DroppedEntry, ...] = ()
    choices: tuple[ScopeChoice, ...]
    compiled: CompiledScope
    ledger: tuple[LedgerStep, ...] = ()


class ServedRole(StrictModel):
    """The entities a served frame binds on one role: how many are members,
    which are the part's regional residual buckets, and whether the fetch
    `narrowed` the role to fewer members than the part's compiled identity
    binds (a fetch of one country of a region), when no option of the part
    describes what it served."""

    role: str
    members: int
    residuals: tuple[str, ...] = Field(default=(), exclude_if=lambda refs: not refs)
    narrowed: bool = Field(default=False, exclude_if=lambda narrowed: not narrowed)


class ServedComponent(StrictModel):
    """One series the served figures were drawn from, with the years its
    frame covers; `periods` is `None` when the manifest states no window."""

    role: str
    periods: tuple[YearSpan, ...] | None = None


class ServedScope(StrictModel):
    """How the figures one retrieval served were counted.

    `spec_id` is the definition the frame was minted for, which a fetch
    narrowed by places or period mints apart from the part's own.
    `measured_property` and `netting_basis` are the served identity's
    measure and netting slot, `None` when it binds none.
    """

    spec_id: str
    roles: tuple[ServedRole, ...]
    measured_property: str | None = None
    netting_basis: str | None = None
    components: tuple[ServedComponent, ...]
