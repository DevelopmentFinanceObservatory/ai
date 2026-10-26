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

A switch on the plan card moves one choice to another option.
`switch_entries` builds the entries the switch posts over the part's base
identity, and `moved_choices` names each other choice the compiled answer
moved with it.

`ServedScope` is how the figures a retrieval served were counted, read from
its `get_data` manifest (`served_facts`): the entities bound on each role,
the measure basis and the components. `harness.turn_records.basis_line`
states the turn's choices in their options' labels, correcting each from
it where the served figures hold another option. `REASON_COPY` holds that
line's words for reason codes.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping
from typing import Annotated, Any, Literal

from planner.records import OptionStatus, ScopeAxis, StrictModel
from pydantic import Field

EntryProvenance = Literal["explicit", "carried"]
ChoiceProvenance = Literal["explicit", "carried", "default"]
ComponentRole = Literal["whole", "bilateral", "imputed", "headline_net", "headline_ge"]
ScopeReasonCode = Literal[
    "provider_not_offered",
    "imputed_coverage",
    "basis_not_published",
    "residuals_need_summed_region",
    "conflicts_with",
    "carried_choice_not_applied",
    "stated_choice_not_applied",
    "option_not_covered",
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
    """An entry Search left out of this part, with its cause: a carried one
    (`carried_choice_not_applied`) or an option the user's words asked for
    that the part cannot serve (`stated_choice_not_applied`)."""

    entry: Entry
    code: Literal["carried_choice_not_applied", "stated_choice_not_applied"] = (
        "carried_choice_not_applied"
    )
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

    @classmethod
    def from_search(cls, raw: object) -> PartScope:
        """Read a part's `resolved.scope_choices` as Search wrote it.

        A field Search added that this mirror does not declare, anywhere in
        the record, is ignored, so an additive dfo-apis release leaves the
        part's scope choices readable. The record keeps the declared fields
        alone, and a stored one is read back with no field ignored.

        Raises:
            ValidationError: If a declared field is missing or does not fit.
        """
        return cls.model_validate(raw, extra="ignore")


# The key a spec block carries when the planner compiled it to put a
# calculation's inputs on one measure basis (`planner.agent.compile_on_basis`):
# its explicit basis entry is the planner's, never the user's. The block is
# the one source of truth. `harness.tool_contracts.reading_facts` copies it to
# the stored reading (`harness.turn_records.SearchReading.aligned_basis`),
# which `carried_entries`' callers read, and `submission.service.
# _aligned_bases` reads it off the pin's block. A block Search answers afresh
# carries none, so a switch of the basis drops it; a switch of another axis
# keeps it (`harness.graph.switch`).
ALIGNED_BASIS_BLOCK_KEY = "_aligned_basis"


def mark_aligned_basis(digest: dict[str, Any]) -> None:
    """Mark each spec block of `digest`, a `harness.tool_contracts.resolver_digest`,
    as compiled by the planner to align a basis (`ALIGNED_BASIS_BLOCK_KEY`)."""
    for holder in (digest, *(digest.get("part_digests") or ())):
        if isinstance(holder.get("spec_block"), Mapping):
            holder["spec_block"] = {**holder["spec_block"], ALIGNED_BASIS_BLOCK_KEY: True}


def carried_entries(scopes: Iterable[PartScope], *, aligned: Collection[str] = ()) -> list[Entry]:
    """The entries a later search carries from `scopes`, oldest first.

    Every choice an explicit or carried entry settled becomes a `carried`
    entry for its active option, with the concept of the entry that settled
    it and a `choice` origin naming the choice. A choice a declared default
    settled carries nothing, and neither does the measure basis of a scope
    whose compiled spec_id is in `aligned`: the planner compiled it to put a
    calculation's inputs on one basis (`harness.turn_records.SearchReading.
    aligned_basis`), so that entry is its own, and that scope's other
    choices, the user's, still carry. The newest choice per axis and
    concept wins.
    """
    newest: dict[tuple[str, str | None], Entry] = {}
    for scope in scopes:
        for choice in scope.choices:
            if choice.provenance == "default" or (
                choice.axis == "measure_basis" and scope.compiled.spec_id in aligned
            ):
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


def switch_entries(scope: PartScope, *, choice: ScopeChoice, option_id: str) -> list[Entry]:
    """The scope entries a switch of `choice` to `option_id` posts over `scope.base`.

    The scope's own entries, less any on the choice's axis, then the switch
    as an explicit entry from the choice (`ChoiceOrigin`), so Search
    compiles the other choices as they were settled. The switch keeps the
    concept of the entry it replaces, so a later search carries it to the
    same concept only.
    """
    replaced = next((entry for entry in scope.entries if entry.axis == choice.axis), None)
    return [
        *(entry for entry in scope.entries if entry.axis != choice.axis),
        Entry(
            axis=choice.axis,
            option_id=option_id,
            provenance="explicit",
            concept=replaced.concept if replaced is not None else None,
            origin=ChoiceOrigin(choice_id=choice.id),
        ),
    ]


def moved_choices(before: PartScope, after: PartScope, *, switched: str) -> list[dict[str, str]]:
    """Each choice of `before`, other than the one on axis `switched`, whose active option `after` moved.

    Each is `{choice_id, from_label, to_label, reason}`: the option it held,
    the one it holds now, and why, as a sentence. The reason is the option
    it held being limited now (`REASON_COPY`), else Search's default
    following the switch.
    """
    now = {choice.axis: choice for choice in after.choices}
    moved: list[dict[str, str]] = []
    for choice in before.choices:
        changed = now.get(choice.axis)
        if choice.axis == switched or changed is None or changed.active == choice.active:
            continue
        held, holds = choice.active_option, changed.active_option
        if held is None or holds is None:
            continue
        kept = next(
            (option for option in changed.options if option.option_id == choice.active), None
        )
        code = kept.availability.code if kept is not None else "option_not_covered"
        moved.append(
            {
                "choice_id": changed.id,
                "from_label": held.label,
                "to_label": holds.label,
                "reason": _sentence(REASON_COPY[code])
                if code in REASON_COPY
                else "Search counts it this way with the option you chose.",
            }
        )
    return moved


def _sentence(clause: str) -> str:
    """`clause` as a sentence: its first letter capitalised and a full stop after it."""
    return f"{clause[:1].upper()}{clause[1:]}."


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

    role: ComponentRole
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

    @classmethod
    def of(cls, facts: Mapping[str, Any], *, scope: PartScope) -> ServedScope:
        """Compact `served_facts(...)` against the part's `scope`: a bound
        entity is a residual when one of the scope's options adds it beside
        the members (`EntitiesBinding.add`), and a member otherwise. A role
        is narrowed when it leaves out a member the scope's compiled
        identity binds on it (`CompiledScope.identity["entities"]`)."""
        residual_refs = {
            ref
            for choice in scope.choices
            for option in choice.options
            if isinstance(option.binding, EntitiesBinding)
            for ref in option.binding.add
        }
        compiled = scope.compiled.identity.get("entities") or {}
        roles = []
        for role, refs in facts["entities"].items():
            residuals = tuple(ref for ref in refs if ref in residual_refs)
            members = {ref for ref in compiled.get(role) or () if ref not in residual_refs}
            roles.append(
                ServedRole(
                    role=role,
                    members=len(refs) - len(residuals),
                    residuals=residuals,
                    narrowed=not members <= set(refs),
                )
            )
        return cls(
            spec_id=facts["spec_id"],
            roles=tuple(roles),
            measured_property=facts.get("measured_property"),
            netting_basis=facts.get("netting_basis"),
            components=tuple(
                ServedComponent.model_validate(component) for component in facts["components"]
            ),
        )

    def narrowed(self, role: str) -> bool:
        """Whether the fetch bound fewer of `role`'s members than the part compiles to."""
        return any(served.role == role and served.narrowed for served in self.roles)

    def residuals(self, role: str) -> frozenset[str]:
        """The residual buckets served on `role`; empty for a role the frame does not bind."""
        return frozenset(
            ref for served in self.roles if served.role == role for ref in served.residuals
        )


_NETTING_SLOT = "nettingBasis"


def served_facts(manifest: Mapping[str, Any]) -> dict[str, Any] | None:
    """What a `get_data` manifest says the frame was counted as, for `ServedScope.of`.

    Read from the minted definition's identity (`definition.identity`): the
    refs bound on each entity role, the shape's measured property and the
    `nettingBasis` slot's bound value. The frame is one `whole` component
    over the manifest's covered window (`coverage.window`). JSON only, so a
    `get_data` result can carry it. `None` for a manifest naming no
    definition.
    """
    definition = manifest.get("definition") or {}
    spec_id = definition.get("spec_id")
    if not isinstance(spec_id, str):
        return None
    identity = definition.get("identity") or {}
    shape = identity.get("shape") or {}
    netting = next(
        (
            binding.get("ref")
            for slot in identity.get("slots") or ()
            if slot.get("property") == _NETTING_SLOT
            and (binding := slot.get("binding") or {}).get("state") in ("value", "exact")
        ),
        None,
    )
    window = (manifest.get("coverage") or {}).get("window") or {}
    start, end = window.get("start_year"), window.get("end_year")
    periods = [[start, end]] if isinstance(start, int) and isinstance(end, int) else None
    measured = shape.get("measured_property")
    return {
        "spec_id": spec_id,
        "entities": {
            str(role): [str(ref) for ref in refs]
            for role, refs in (identity.get("entities") or {}).items()
        },
        "measured_property": measured if isinstance(measured, str) else None,
        "netting_basis": netting if isinstance(netting, str) else None,
        "components": [{"role": "whole", "periods": periods}],
    }


# What a basis line says when the served figures match no option of a choice.
SERVED_OTHERWISE = "the figures served differ from this choice"


# What each scope reason code says to a reader, one clause.
REASON_COPY: dict[str, str] = {
    "provider_not_offered": "imputed multilateral aid is not offered for this question",
    "imputed_coverage": "imputed multilateral aid covers only some years",
    "basis_not_published": "this basis is not published for these figures",
    "residuals_need_summed_region": "regional buckets apply only to a summed region",
    "conflicts_with": "it conflicts with another choice",
    "carried_choice_not_applied": "an earlier choice did not apply here",
    "stated_choice_not_applied": "the option you asked for is not available here",
    "option_not_covered": "the data holds no series for this option",
}
