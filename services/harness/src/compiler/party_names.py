"""Which party a requirement's entity names on a source.

A requirement names an entity in the planner's words ("United States", "G7"), and
a source resolves parties by ref and label ("country/USA", "United States of
America"). These functions read a name as the party a source resolves it to: by
the row Search bound for it, by label, by the ref the graph resolved it to, or as
a named set. They take any entity with a name, role and pinned dcid
(`NamedParty`) and any source stating its parties (`PartySource`), so both the
evidence check (`compiler.evidence`) and the planner's reading of a whole on its
source (`planner.plan.whole_on_source`) use them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

from planner.parties import flow_side
from planner.workflow import BoundEntity, ResolvedEntity, ResolvedGroup


class NamedParty(Protocol):
    """An entity a requirement names: its words, its role phrase and the dcid it pins."""

    @property
    def name(self) -> str: ...

    @property
    def role(self) -> str | None: ...

    @property
    def dcid(self) -> str | None: ...


class PartySource(Protocol):
    """What a source or a retrieval states of the parties it holds."""

    @property
    def entities(self) -> Sequence[ResolvedEntity]: ...

    @property
    def groups(self) -> Sequence[ResolvedGroup]: ...

    @property
    def entity_refs(self) -> Mapping[str, str]: ...

    @property
    def bound_refs(self) -> Sequence[BoundEntity]: ...


def entity_matches(entity: NamedParty, resolved: Sequence[ResolvedEntity]) -> list[ResolvedEntity]:
    """Return every resolved entity whose `label` matches `entity.name`, case-folded.

    Matched by name alone, before role narrows the field -- what
    `resolve_entity` starts from, and what a finding reports as what was
    found when the name alone does not pick out one resolution.
    """
    wanted = entity.name.casefold()
    return [one for one in resolved if one.label.casefold() == wanted]


def resolve_entity(entity: NamedParty, resolved: Sequence[ResolvedEntity]) -> ResolvedEntity | None:
    """Return the one resolved entity that answers `entity`, or `None` when none or several do.

    Matches `Entity.name` against `label`, case-folded. When the requirement
    states a role, a match whose role column names it, or sits on the same
    side of the flow (`donor` and `origin`), is preferred, so a name
    resolving under several role columns is told apart.
    """
    wanted = entity.name.casefold()
    matches = [one for one in resolved if one.label.casefold() == wanted]
    if len(matches) > 1:
        matches = [one for one in matches if role_fits(entity.role, one.role)]
    return matches[0] if len(matches) == 1 else None


def bound_refs_of(entity: NamedParty, source: PartySource) -> tuple[str, ...]:
    """The refs Search bound for the entity rows `entity` names (`SelectedSource.bound_refs`).

    `Entity.name` names the rows whose identity, a pinned dcid or a
    stated name, it equals case-folded and whose `role_text` fits its
    role: names it, sits on its side of the flow, or is unstated, as is a
    role the requirement leaves unstated. Else it names the rows whose
    text it equals and whose role fits. So of a row written "Congo" from
    and one written "Congo" to and named "Republic of the Congo", a
    recipient "Congo" names the second. When no row fits its role, it
    names the rows of its identity, else of its text, so a row bound on
    the other side is still reported against the role asked. Empty when
    it names no row.

    An entity with a `dcid` names only the rows pinning that node, so two
    entities written "Congo" and pinned apart each name their own row, and a
    row binding the name to another node never answers the pinned one.
    """
    if entity.dcid is not None:
        return _pinned_refs(entity, source)
    wanted = entity.name.casefold()
    by_identity = [row for row in source.bound_refs if row.identity.casefold() == wanted]
    by_text = [row for row in source.bound_refs if row.text.casefold() == wanted]

    def fitting(rows: list[BoundEntity]) -> list[BoundEntity]:
        return [row for row in rows if row.role is None or role_fits(entity.role, row.role)]

    return _refs_of(fitting(by_identity) or fitting(by_text) or by_identity or by_text)


def _pinned_refs(entity: NamedParty, source: PartySource) -> tuple[str, ...]:
    """The refs Search bound for the rows whose identity is `entity`'s `dcid`; empty without one."""
    if entity.dcid is None:
        return ()
    return _refs_of([row for row in source.bound_refs if row.identity == entity.dcid])


def _refs_of(rows: Sequence[BoundEntity]) -> tuple[str, ...]:
    """The refs Search bound for `rows`, each once, in order."""
    return tuple(dict.fromkeys(ref for row in rows for ref in row.refs))


def resolve_bound(entity: NamedParty, source: PartySource) -> ResolvedEntity | None:
    """Return the one entity of `source` whose ref Search bound for the rows `entity` names
    (`bound_refs_of`), or `None` when none or several do.

    A stated role tells apart a ref resolved under several role columns.
    """
    return _one_by_ref(entity, source.entities, bound_refs_of(entity, source))


def resolve_name(entity: NamedParty, source: PartySource) -> ResolvedEntity | None:
    """Return the one entity of `source` that `entity` names, by the row Search bound, by label, or else by ref.

    The refs Search bound for the rows `entity` names answer first
    (`resolve_bound`), so "G7" names the published total labelled "Group
    of Seven", and a donor "Congo" the Congo its own row pins whatever
    the other Congo is labelled. Else its name is matched by label
    (`resolve_entity`), and a name Search bound no row for is looked up
    in the refs the graph resolved the requirement's names to
    (`SelectedSource.entity_refs`), so "United States" names the entity
    labelled "United States of America". A stated role tells apart a ref
    resolved under several role columns, as it does a label.

    An entity with a `dcid` names the entity of the refs Search bound for
    the rows pinning it, else of the `dcid` itself, and no other.
    """
    if entity.dcid is not None:
        refs = _pinned_refs(entity, source) or (entity.dcid,)
        return _one_by_ref(entity, source.entities, refs)
    bound = bound_refs_of(entity, source)
    resolved = _one_by_ref(entity, source.entities, bound) or resolve_entity(
        entity, source.entities
    )
    if resolved is not None or bound:
        return resolved
    name = entity.name.casefold()
    refs = (source.entity_refs[name],) if name in source.entity_refs else ()
    return _one_by_ref(entity, source.entities, refs)


def _one_by_ref(
    entity: NamedParty, resolved: Sequence[ResolvedEntity], refs: Sequence[str]
) -> ResolvedEntity | None:
    """The one of `resolved` whose ref is among `refs`, told apart by `entity`'s role; `None` when none or several are."""
    matches = [one for one in resolved if one.ref in refs]
    if len(matches) > 1:
        matches = [one for one in matches if role_fits(entity.role, one.role)]
    return matches[0] if len(matches) == 1 else None


def resolve_group(entity: NamedParty, source: PartySource) -> ResolvedGroup | None:
    """Return the one named set of `source` that `entity` names, or `None` when none or several do.

    Matches `Entity.name` against the set's `label`, case-folded, or names
    the set whose ref or published total Search bound for the rows it
    names (`bound_refs_of`), so "G7" names the set Search labels "Group
    of Seven". A stated role has to name the set's role
    column or sit on its side of the flow (`donor` and `origin`), so "G7"
    asked for as a recipient is not the G7 whose members give.
    """
    wanted = entity.name.casefold()
    bound = bound_refs_of(entity, source)
    matches = [
        group
        for group in source.groups
        if (group.label.casefold() == wanted or group.ref in bound or group.total in bound)
        and role_fits(entity.role, group.role)
    ]
    return matches[0] if len(matches) == 1 else None


def named_members(entity: NamedParty, source: PartySource) -> tuple[ResolvedEntity, ...]:
    """The entities of `source` that `entity` names, empty when it names none.

    A name matching one of the source's named sets (`resolve_group`) names
    every entity the source resolves on that set's role column, which is the
    set's members or the one party that is its published total
    (`ResolvedGroup.total`). Any other name names the one entity
    `resolve_name` finds, by label or by the ref the graph resolved the
    name to. Callers compare the result by `ref`, so a member is named
    whatever label it carries.
    """
    group = resolve_group(entity, source)
    if group is not None:
        return tuple(one for one in source.entities if one.role == group.role)
    resolved = resolve_name(entity, source)
    return () if resolved is None else (resolved,)


def named_refs(entities: Sequence[NamedParty], source: PartySource) -> set[str]:
    """The refs of every entity of `source` that one of `entities` names (`named_members`)."""
    return {one.ref for entity in entities for one in named_members(entity, source)}


def role_fits(role: str | None, column: str) -> bool:
    """Whether the requirement's `role` phrase, `None` when unstated, allows the role `column`."""
    if role is None or role.casefold() == column.casefold():
        return True
    side = flow_side(role)
    return side is not None and flow_side(column) == side


def party_keys(entity: NamedParty, source: PartySource) -> tuple[str, ...]:
    """The keys of the one party `source` resolves `entity` to: the named set it
    names (`resolve_group`), by the set's own ref, published total and registry id,
    or by its members' refs (`named_members`) where it states none of the three;
    else the ref of the one entity it names (`resolve_name`: by the row Search
    bound, by label, or by the ref the graph resolved the name to). Empty when it
    resolves none."""
    if (group := resolve_group(entity, source)) is not None:
        keys = [key for key in (group.ref, group.total, group.group_id) if key]
        return tuple(dict.fromkeys(keys or (one.ref for one in named_members(entity, source))))
    resolved = resolve_name(entity, source)
    return () if resolved is None else (resolved.ref,)
