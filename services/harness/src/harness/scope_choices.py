"""How Search counted one part: its scope choices, as the harness stores and carries them.

`PartScope` mirrors a definite Spec block's `resolved.scope_choices`, the
record dfo-apis writes for each part (`search.models.scope` there): one
`ScopeChoice` per axis a declared rule or the request settled (sector,
region, measure basis, provider scope), each with its options and their
availability, the explicit and carried entries applied
(`data_mcp.scope.ScopeEntry`), and the identity they compile to. `harness.turn_records.SearchReading.scope` stores it.
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
line's words for reason codes, the ones the search digest uses
(`data_mcp.scope`).
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping
from typing import Annotated, Any, Literal, NamedTuple, get_args

from common.composite import manifest_components, year_spans, years_of
from data_mcp.scope import (
    REASON_COPY,
    ChoiceOrigin,
    ChoiceProvenance,
    OptionId,
    ScopeAxis,
    ScopeEntry,
    ScopeReasonCode,
)
from planner.records import OptionStatus, StrictModel
from pydantic import Field, ValidationInfo, field_validator

# Whose spending a provider option counts: bilateral flows alone, with imputed
# shares of multilateral spending, or with core contributions to multilaterals
# as a published donor total such as DAC1's counts them.
ProviderScope = Literal["bilateral", "inclusive", "contributions"]
DroppedCode = Literal["carried_choice_not_applied", "stated_choice_not_applied"]

YearSpan = tuple[int, int]


class BlockDataset(NamedTuple):
    """The dataset a spec block counts from, as its `resolved.facets.provenance` names it:
    `ref` (`provenance/OECDDAC1`) and `title` ("OECD DAC1"), each `None` when unstated."""

    ref: str | None
    title: str | None


def block_dataset(block: Mapping[str, Any] | None) -> BlockDataset:
    """The dataset `block`, a spec block, counts from (`BlockDataset`)."""
    facets = ((block or {}).get("resolved") or {}).get("facets") or {}
    provenance = facets.get("provenance") or {}
    ref, title = provenance.get("ref"), provenance.get("label")
    return BlockDataset(
        ref if isinstance(ref, str) and ref else None,
        title if isinstance(title, str) and title else None,
    )


def option_axis(option_id: str) -> str:
    """The axis an `OptionId` names, the part before its colon (`provider` of `provider:inclusive`)."""
    return option_id.split(":", 1)[0]


# The axes, provider scopes and reason codes this release reads. Search's
# vocabularies are open: a later release can add to any, and a record holding
# a value outside them stays readable (`_on_known_axes`, `_readable_choices`,
# `OptionAvailability.code`, `DroppedEntry.code`), but for a choice holding an
# unknown provider scope active, which `_readable_choices` refuses.
_KNOWN_AXES = frozenset(get_args(ScopeAxis))
_KNOWN_PROVIDER_SCOPES = frozenset(get_args(ProviderScope))
_KNOWN_REASONS = frozenset(get_args(ScopeReasonCode))


def _on_known_axes(items: Any, *path: str) -> Any:
    """`items`, a raw list from Search, less each item whose axis at `path` this release does not read.

    An item holding no axis string there is kept, so validation refuses it.
    """

    def known(item: Any) -> bool:
        for key in path:
            item = item.get(key) if isinstance(item, Mapping) else None
        return not isinstance(item, str) or item in _KNOWN_AXES

    return [item for item in items if known(item)] if isinstance(items, list | tuple) else items


def _named_provider_scope(option: Any) -> object:
    """The provider scope a raw option from Search counts: its provider binding's
    scope, or, for an option binding anything else, the value its id names on the
    provider axis (`provider:bilateral` binding DAC1 bilateral ODA's code); `None`
    for an option naming none."""
    if not isinstance(option, Mapping):
        return None
    binding = option.get("binding")
    if isinstance(binding, Mapping) and binding.get("kind") == "provider":
        return binding.get("scope")
    option_id = option.get("option_id")
    if not isinstance(option_id, str):
        return None
    axis, _, value = option_id.partition(":")
    return value if axis == "provider" else None


def _readable_choices(items: Any) -> Any:
    """`items`, a scope's raw choices from Search, without each choice on an axis this
    release does not read (`_on_known_axes`), and each choice without its options of
    a provider scope it does not read (`_named_provider_scope`).

    An option holding no scope string is kept, so validation refuses it.

    Raises:
        ValueError: When a choice's active option counts a provider scope this
            release does not read: the choice would hold nothing, and leaving the
            choice out would turn provider alignment off for the part.
    """
    kept = _on_known_axes(items, "axis")
    if not isinstance(kept, list):
        return kept
    readable = []
    for choice in kept:
        options = choice.get("options") if isinstance(choice, Mapping) else None
        if not isinstance(options, list | tuple):
            readable.append(choice)
            continue
        unread = {
            option.get("option_id")
            for option in options
            if isinstance(scope := _named_provider_scope(option), str)
            and scope not in _KNOWN_PROVIDER_SCOPES
        }
        if not unread:
            readable.append(choice)
            continue
        if choice.get("active") in unread:
            raise ValueError(
                f"scope choice {choice.get('id')!r} holds {choice.get('active')!r}, a provider "
                "scope this release does not read"
            )
        readable.append(
            {
                **choice,
                "options": [
                    option
                    for option in options
                    if not isinstance(option, Mapping) or option.get("option_id") not in unread
                ],
            }
        )
    return readable


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

    @field_validator("code", mode="before")
    @classmethod
    def _known_code(cls, code: object) -> object:
        """A reason code this release does not read, as none: the status still says what holds."""
        return code if not isinstance(code, str) or code in _KNOWN_REASONS else None


class Option(StrictModel):
    """One value a scope choice can take."""

    option_id: OptionId
    label: str
    binding: Annotated[
        SlotBinding | EntitiesBinding | ShapeBinding | ProviderBinding,
        Field(discriminator="kind"),
    ]
    availability: OptionAvailability

    @property
    def provider_scope(self) -> ProviderScope | None:
        """Whose spending this option counts, on the provider axis: its binding's
        scope, or, for an option binding a slot, the scope its id names, as a
        published total's sibling row (`provider:bilateral` binding DAC1 bilateral
        ODA's code under DAC1 total ODA); `None` off the provider axis."""
        if isinstance(self.binding, ProviderBinding):
            return self.binding.scope
        axis, _, value = self.option_id.partition(":")
        if axis != "provider":
            return None
        return next((scope for scope in get_args(ProviderScope) if scope == value), None)


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

    entry: ScopeEntry
    code: DroppedCode = "carried_choice_not_applied"
    cause: str

    @field_validator("code", mode="before")
    @classmethod
    def _known_code(cls, code: object, info: ValidationInfo) -> object:
        """A code this release does not read, as the one the entry's provenance gives."""
        if not isinstance(code, str) or code in get_args(DroppedCode):
            return code
        entry = info.data.get("entry")
        explicit = isinstance(entry, ScopeEntry) and entry.provenance == "explicit"
        return "stated_choice_not_applied" if explicit else "carried_choice_not_applied"


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

    @field_validator("entries", "ledger", mode="before")
    @classmethod
    def _known_axes(cls, items: object) -> object:
        """Each entry and switch on an axis this release reads; the others are left out."""
        return _on_known_axes(items, "axis")

    @field_validator("choices", mode="before")
    @classmethod
    def _known_choices(cls, items: object) -> object:
        """Each choice this release reads (`_readable_choices`); the others are left out."""
        return _readable_choices(items)

    @field_validator("dropped_entries", mode="before")
    @classmethod
    def _known_dropped_axes(cls, items: object) -> object:
        """Each dropped entry on an axis this release reads; the others are left out."""
        return _on_known_axes(items, "entry", "axis")

    @classmethod
    def from_search(cls, raw: object) -> PartScope:
        """Read a part's `resolved.scope_choices` as Search wrote it.

        A field Search added that this mirror does not declare, anywhere in
        the record, is ignored, so an additive dfo-apis release leaves the
        part's scope choices readable. The record keeps the declared fields
        alone, and a stored one is read back with no field ignored. A value
        Search added to an open vocabulary degrades only what holds it: a
        choice or entry on an unknown axis is left out, as is an option of an
        unknown provider scope (a choice holding one active is refused,
        `_readable_choices`), an unknown reason
        code reads as none, a dropped entry's unknown code as the one its
        provenance gives, and a component keeps whatever role it names.

        Raises:
            ValidationError: If a declared field is missing or does not fit.
        """
        return cls.model_validate(raw, extra="ignore")


# The key a spec block carries naming the scope axes whose entries the
# planner's own words settled, so they are the planner's. They are the
# measure basis or provider scope of a pin compiled to put a calculation's
# inputs on one basis or one provider scope
# (`planner.agent.compile_on_basis`), and every axis an entry settled
# on a pin searched for an input the planner pinned no dataset for, posted
# with the planner's words and no message of the user's
# (`planner.agent.pin_by_search`). The block is the one source of truth.
# `harness.tool_contracts.reading_facts` copies it to the stored reading
# (`harness.turn_records.SearchReading.planner_axes`), which
# `carried_entries`' callers read; `submission.service._input_bases` and
# `submission.service.scope_choice_decisions` read it off the pin's block. A
# block Search answers afresh carries none, so a switch on the plan card
# drops the switched axis from it and keeps the others (`harness.graph.switch`).
PLANNER_AXES_BLOCK_KEY = "_planner_axes"


def planner_axes(block: Mapping[str, Any] | None) -> frozenset[str]:
    """The axes `block`, a spec block, marks as settled by the planner (`PLANNER_AXES_BLOCK_KEY`)."""
    return frozenset((block or {}).get(PLANNER_AXES_BLOCK_KEY) or ())


def mark_planner_axes(digest: dict[str, Any], axes: Iterable[str]) -> None:
    """Mark each spec block of `digest`, a `harness.tool_contracts.resolver_digest`,
    as settled by the planner on `axes` (`PLANNER_AXES_BLOCK_KEY`); no axes marks nothing."""
    marked = sorted(set(axes))
    if not marked:
        return
    for holder in (digest, *(digest.get("part_digests") or ())):
        if isinstance(holder.get("spec_block"), Mapping):
            holder["spec_block"] = {**holder["spec_block"], PLANNER_AXES_BLOCK_KEY: marked}


def carried_entries(
    scopes: Iterable[PartScope], *, planner: Mapping[str, Collection[str]] | None = None
) -> list[ScopeEntry]:
    """The entries a later search carries from `scopes`, oldest first.

    Every choice an explicit or carried entry settled becomes a `carried`
    entry for its active option, with the concept of the entry that settled
    it and a `choice` origin naming the choice. A choice a declared default
    settled carries nothing, and neither does a choice on an axis `planner`
    names for the scope's compiled spec_id (`PLANNER_AXES_BLOCK_KEY`): the
    planner's words settled it, and the scope's other choices, the user's,
    still carry. The newest choice per axis and concept wins.
    """
    planner = planner or {}
    newest: dict[tuple[str, str | None], ScopeEntry] = {}
    for scope in scopes:
        for choice in scope.choices:
            if choice.provenance == "default" or choice.axis in planner.get(
                scope.compiled.spec_id, ()
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
            newest[key] = ScopeEntry(
                axis=choice.axis,
                option_id=choice.active,
                provenance="carried",
                concept=concept,
                origin=ChoiceOrigin(choice_id=choice.id),
            )
    return list(newest.values())


def switch_entries(scope: PartScope, *, choice: ScopeChoice, option_id: str) -> list[ScopeEntry]:
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
        ScopeEntry(
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

    def composite_clause(self) -> tuple[ScopeAxis, str] | None:
        """What a composite frame's served components count, as one basis-line clause, with
        the axis whose choice it speaks for; `None` for figures served whole.

        Bilateral and imputed components stacked over the same years read "bilateral
        plus imputed multilateral" with the years served ("2011 to 2024"), on the
        provider axis. The OECD headline's net and grant-equivalent components read
        "net to 2017, grant equivalents from 2018", by the last year served net and the
        first served in grant equivalents, on the measure-basis axis.
        """
        years = {served.role: years_of(served.periods or ()) for served in self.components}
        if years.keys() == {"bilateral", "imputed"}:
            held = sorted(years["bilateral"] | years["imputed"])
            served_years = f", {_years_text(held)}" if held else ""
            return "provider", f"bilateral plus imputed multilateral{served_years}"
        if years.keys() and years.keys() <= {"headline_net", "headline_ge"}:
            pieces = []
            if net := years.get("headline_net"):
                pieces.append(f"net to {max(net)}")
            if grant := years.get("headline_ge"):
                pieces.append(f"grant equivalents from {min(grant)}")
            return ("measure_basis", ", ".join(pieces)) if pieces else None
        return None

    def narrowed(self, role: str) -> bool:
        """Whether the fetch bound fewer of `role`'s members than the part compiles to."""
        return any(served.role == role and served.narrowed for served in self.roles)

    def residuals(self, role: str) -> frozenset[str]:
        """The residual buckets served on `role`; empty for a role the frame does not bind."""
        return frozenset(
            ref for served in self.roles if served.role == role for ref in served.residuals
        )


def _years_text(years: list[int]) -> str:
    """Sorted, distinct `years` as runs: "2011 to 2024", or "2005, 2011 to 2024"."""
    return ", ".join(
        str(first) if first == last else f"{first} to {last}" for first, last in year_spans(years)
    )


_NETTING_SLOT = "nettingBasis"


def served_facts(manifest: Mapping[str, Any]) -> dict[str, Any] | None:
    """What a `get_data` manifest says the frame was counted as, for `ServedScope.of`.

    Read from the minted definition's identity (`definition.identity`): the
    refs bound on each entity role, the shape's measured property and the
    `nettingBasis` slot's bound value. A composite frame's components are its
    trace's (`common.composite.manifest_components`), each role with the
    years it serves; any other frame is one `whole` component over the
    manifest's covered window (`coverage.window`). JSON only, so a `get_data`
    result can carry it. `None` for a manifest naming no definition.
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
    listed = manifest_components(manifest.get("trace") or {})
    components = (
        [
            {"role": component.role, "periods": [list(span) for span in component.periods] or None}
            for component in listed
        ]
        if listed
        else [{"role": "whole", "periods": periods}]
    )
    return {
        "spec_id": spec_id,
        "entities": {
            str(role): [str(ref) for ref in refs]
            for role, refs in (identity.get("entities") or {}).items()
        },
        "measured_property": measured if isinstance(measured, str) else None,
        "netting_basis": netting if isinstance(netting, str) else None,
        "components": components,
    }


# What a basis line says when the served figures match no option of a choice.
SERVED_OTHERWISE = "the figures served differ from this choice"
