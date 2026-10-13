# Generated from services/data-mcp/tests/fixtures/graph_api_schema.json by
# `just graph-models`. Do not edit by hand.

from __future__ import annotations

from datetime import date
from typing import Annotated, Literal

from pydantic import BaseModel, Field, RootModel


class EntityRef(BaseModel):
    """
    A reference with its kind. `kind` is `null` only for a `not_held` entity, which the
    graph holds no node to classify.
    """

    kind: Annotated[
        Literal[
            "country",
            "continent",
            "subregion",
            "continental_union",
            "place_other",
            "aggregate",
            "organisation",
            "group",
            "stat_var",
            "code_system",
            "code",
            "unit",
            "source",
            "dataset",
            "provenance",
            "other",
        ]
        | None,
        Field(title="Kind"),
    ]
    label: Annotated[str, Field(title="Label")]
    ref: Annotated[str, Field(title="Ref")]


class Matched(BaseModel):
    """
    The listed text the input matched, and whether it is the entity's name, an alias, or a
    code. A phrase naming a group's members ("EU countries") matches as `alias`.
    """

    as_: Annotated[Literal["name", "alias", "code"], Field(alias="as", title="As")]
    text: Annotated[str, Field(title="Text")]


class NodeRef(BaseModel):
    """
    A graph reference. `label` is `null` when the target is not held, carries no `name`, or
    carries only a `name` equal to its own dcid.
    """

    label: Annotated[str | None, Field(title="Label")]
    ref: Annotated[str, Field(title="Ref")]


class ResolveResult(BaseModel):
    """
    One text's outcome. `input` echoes the text; fields a status does not use are `null`.

    - `resolved`: `entity`, `matched` and `reading`. A group or region adds `projection`,
      `member_count` and `members`: the list route's link, or the members inline (at most
      256) when the request expanded them. The three are `null` for a region the membership
      policy does not hold.
    - `ambiguous`: `candidates`, in the resolver's order.
    - `not_found`: `near_matches`, hints for a person; a program should pick none by id.
    - `not_held`: `entity`, the id the resolver named, which does not open on the node route.

    `reading` is `members` when the text names a group's members ("EU countries"), else
    `entity`.
    """

    candidates: Annotated[list[EntityRef] | None, Field(title="Candidates")]
    entity: EntityRef | None
    input: Annotated[str, Field(title="Input")]
    matched: Matched | None
    member_count: Annotated[int | None, Field(title="Member Count")]
    members: Annotated[str | list[NodeRef] | None, Field(title="Members")]
    near_matches: Annotated[list[EntityRef] | None, Field(title="Near Matches")]
    projection: (
        Literal["countries", "other_states", "territories", "former_states", "institutions"] | None
    )
    reading: Annotated[Literal["members", "entity"] | None, Field(title="Reading")]
    status: Annotated[
        Literal["resolved", "ambiguous", "not_found", "not_held"], Field(title="Status")
    ]


class ResolveResponse(BaseModel):
    """
    `results` is index-aligned with the request's `texts`.
    """

    results: Annotated[list[ResolveResult], Field(title="Results")]


class Authority(BaseModel):
    """
    Where a membership's rows come from, with the data version that reproduces them.
    """

    name: Annotated[str, Field(title="Name")]
    version: Annotated[str, Field(title="Version")]


class AvailabilitySource(BaseModel):
    """
    A source that holds the variable: its provenance, labelled with the catalogue's name.
    """

    documentation: Annotated[str, Field(title="Documentation")]
    """
    the path of the provenance's source document
    """
    label: Annotated[str | None, Field(title="Label")]
    ref: Annotated[str, Field(title="Ref")]


class Binding(BaseModel):
    """
    The `scope.facets` values that request a series from Datasets, in its leading unit.
    """

    measurement_method: Annotated[
        str | None, Field(alias="measurementMethod", title="Measurementmethod")
    ]
    provenance: Annotated[str, Field(title="Provenance")]
    unit: Annotated[str | None, Field(title="Unit")]


class CodeBlock(BaseModel):
    """
    A code's detail block.
    """

    code: Annotated[str, Field(title="Code")]
    code_system: NodeRef | None
    """
    the system's graph node, such as `CRSPurposeCodeEnum`
    """
    parent: NodeRef | None
    """
    the code it specialises; null for a root
    """
    property: Annotated[str, Field(title="Property")]
    """
    the constraint property the code is a value of
    """
    synonyms: Annotated[list[str], Field(title="Synonyms")]


class CodeNode(BaseModel):
    built_from: Annotated[
        dict[Literal["code_index", "origin_catalogue", "coverage_index", "membership"], str],
        Field(title="Built From"),
    ]
    """
    the build of each table the response read
    """
    code: CodeBlock
    description: Annotated[str | None, Field(title="Description")]
    """
    the node's prose in the graph; null when it has none
    """
    gaps: Annotated[list[str], Field(title="Gaps")]
    """
    the member path of every value this door cannot know
    """
    kind: Annotated[Literal["code"], Field(title="Kind")]
    label: Annotated[str | None, Field(title="Label")]
    """
    the node's name; for a Provenance, its parent Dataset's name where it has one. Null, and named in gaps, when the graph gives none
    """
    links: Annotated[
        dict[
            Literal["codes", "children", "members", "countries", "datasets", "variables", "facets"],
            str,
        ],
        Field(title="Links"),
    ]
    """
    each list relation the node has, mapped to its route; empty when none
    """
    ref: Annotated[str, Field(title="Ref")]
    """
    the node's dcid
    """
    types: Annotated[list[str], Field(title="Types")]
    """
    the node's typeOf values, as the graph holds them
    """


class CodeSystemDeclarations(BaseModel):
    """
    The rules a code system declares (`code_index.types.CodeSystemDeclaration`).
    """

    absent_reading: Annotated[Literal["not_declared", "aggregate"], Field(title="Absent Reading")]
    """
    how a variable that does not declare the property reads
    """
    disjoint_from_children: Annotated[bool, Field(title="Disjoint From Children")]
    """
    whether a parent code's records sit apart from its children's, so a parent's total is its own records plus its descendants' (CRS purposes); false where a parent row already totals its children (DAC1 aid types)
    """
    hierarchical: Annotated[bool, Field(title="Hierarchical")]
    """
    whether its codes nest through specializationOf
    """
    summable: Annotated[bool, Field(title="Summable")]
    """
    whether sibling values can rebuild a parent total
    """


class Constraint(BaseModel):
    """
    One constraint: the property and the value the variable fixes it at.
    """

    property: NodeRef
    value: NodeRef


class DataCommonsRef(BaseModel):
    """
    The Data Commons that serves a fact. `id` is the wire enum; `label` its display name.
    """

    id: Annotated[Literal["dfo", "base", "un_system"], Field(title="Id")]
    label: Annotated[str, Field(title="Label")]


class DefinitionText(BaseModel):
    """
    The graph's prose for one part of a definition.
    """

    of: NodeRef
    """
    the part the text defines
    """
    text: Annotated[str, Field(title="Text")]


class EntityCounts(BaseModel):
    """
    The distinct entities in one role, split by their kind in the code index.
    """

    parties: Annotated[int, Field(title="Parties")]
    """
    entities with figures of their own
    """
    residual: Annotated[int, Field(title="Residual")]
    """
    entities holding figures no other entity does
    """
    totals: Annotated[int, Field(title="Totals")]
    """
    entities that hold other entities' figures
    """
    unclassified: Annotated[int, Field(title="Unclassified")]
    """
    entities the code index does not list
    """


class Freshness(BaseModel):
    """
    A source's own refresh cadence, read from DCP provenance arcs that imports populate
    unevenly: `dc/base/WHO_COVID19` carries both `sourceReleaseFrequency` and
    `automaticRefreshType`, `dc/base/BIS_CentralBankPolicyRate` only the second, and
    `dc/base/WHO_GHO` neither. Nullability is the contract.

    `release_frequency` is the raw `sourceReleaseFrequency` ISO 8601 duration (`P1M`);
    `release_frequency_label` is `materialise.cadence.cadence_label`'s display label for it
    (`Monthly`), or the raw token when that module does not recognise it.

    `refresh_type` is `automaticRefreshType`'s value: how this deployment refreshes the
    source, such as `GCPCloudRunAutomaticRefresh`, or the sentinel `NoAutomaticRefresh`.
    """

    refresh_type: Annotated[str | None, Field(title="Refresh Type")]
    release_frequency: Annotated[str | None, Field(title="Release Frequency")]
    release_frequency_label: Annotated[str | None, Field(title="Release Frequency Label")]


class Measure(BaseModel):
    """
    What the variable counts or sums, as the code index names it.
    """

    label: Annotated[str, Field(title="Label")]
    """
    the measure's noun phrase, such as `net disbursements`
    """


class NationalCurrency(BaseModel):
    """
    The units that are each covered place's own currency.
    """

    currencies: Annotated[int, Field(title="Currencies")]
    places: Annotated[int, Field(title="Places")]


class NotHeldEntry(BaseModel):
    """
    A batch entry for a dcid where `GET /api/graph/node/{dcid}` answers 404
    `node_not_held`.
    """

    ref: Annotated[str, Field(title="Ref")]
    status: Annotated[Literal["not_held"], Field(title="Status")]


class OtherNode(BaseModel):
    """
    A node no other kind fits. `types` and `description` carry what the graph holds.
    """

    built_from: Annotated[
        dict[Literal["code_index", "origin_catalogue", "coverage_index", "membership"], str],
        Field(title="Built From"),
    ]
    """
    the build of each table the response read
    """
    description: Annotated[str | None, Field(title="Description")]
    """
    the node's prose in the graph; null when it has none
    """
    gaps: Annotated[list[str], Field(title="Gaps")]
    """
    the member path of every value this door cannot know
    """
    kind: Annotated[Literal["other"], Field(title="Kind")]
    label: Annotated[str | None, Field(title="Label")]
    """
    the node's name; for a Provenance, its parent Dataset's name where it has one. Null, and named in gaps, when the graph gives none
    """
    links: Annotated[
        dict[
            Literal["codes", "children", "members", "countries", "datasets", "variables", "facets"],
            str,
        ],
        Field(title="Links"),
    ]
    """
    each list relation the node has, mapped to its route; empty when none
    """
    ref: Annotated[str, Field(title="Ref")]
    """
    the node's dcid
    """
    types: Annotated[list[str], Field(title="Types")]
    """
    the node's typeOf values, as the graph holds them
    """


class PartyBlock(BaseModel):
    """
    An organisation's or aggregate's detail block.
    """

    roles: Annotated[list[str] | None, Field(title="Roles")]
    """
    the role properties a node of these types can fill, by the role ranges the graph declares (rangeIncludes and its subClassOf closure); null when that read failed
    """


class PriceBasisWords(BaseModel):
    """
    Current or constant prices, with the base year of constant prices when known.
    """

    base_year: Annotated[int | None, Field(title="Base Year")]
    prices: Annotated[Literal["current", "constant"], Field(title="Prices")]


class ProjectionCount(BaseModel):
    """
    One projection of a membership and its member count at the block's date.
    """

    complete: Annotated[bool, Field(title="Complete")]
    """
    false when the authority does not record the projection, or is known to hold it only in part (`membership_incomplete`)
    """
    count: Annotated[int | None, Field(title="Count")]
    """
    members under the projection; null when the authority does not record it
    """
    projection: Annotated[
        Literal["countries", "other_states", "territories", "former_states", "institutions"],
        Field(title="Projection"),
    ]
    """
    A named cut of a membership by member class.
    """


class SourceBlock(BaseModel):
    """
    A Source, Dataset or Provenance's detail block. `documentation` is the path of the
    node's source document on this API.
    """

    data_commons: DataCommonsRef | None
    """
    the Data Commons serving this provenance, from the origin catalogue; null for a provenance the catalogue does not admit, a Dataset, or a Source
    """
    description_url: Annotated[str | None, Field(title="Description Url")]
    """
    the descriptionUrl arc's value
    """
    documentation: Annotated[str, Field(title="Documentation")]
    """
    the path of the node's source document
    """
    earliest_observation_date: Annotated[str | None, Field(title="Earliest Observation Date")]
    """
    the earliestObservationDate arc's value
    """
    freshness: Freshness
    last_data_refresh_date: Annotated[str | None, Field(title="Last Data Refresh Date")]
    """
    the lastDataRefreshDate arc's value
    """
    latest_observation_date: Annotated[str | None, Field(title="Latest Observation Date")]
    """
    the latestObservationDate arc's value
    """
    license: Annotated[str | None, Field(title="License")]
    license_type: Annotated[str | None, Field(title="License Type")]
    next_data_refresh_date: Annotated[str | None, Field(title="Next Data Refresh Date")]
    """
    the nextDataRefreshDate arc's value
    """
    parent: NodeRef | None
    """
    the isPartOf arc's target; null when the node has no parent Dataset
    """
    publisher: NodeRef | None
    """
    the source arc's target; null when the node carries no source arc
    """
    source_data_url: Annotated[str | None, Field(title="Source Data Url")]
    """
    the sourceDataUrl arc's value
    """
    transformation_logic_url: Annotated[str | None, Field(title="Transformation Logic Url")]
    """
    the dataTransformationLogic arc's value: a URL to the transformation code
    """
    url: Annotated[str | None, Field(title="Url")]


class SourceNode(BaseModel):
    built_from: Annotated[
        dict[Literal["code_index", "origin_catalogue", "coverage_index", "membership"], str],
        Field(title="Built From"),
    ]
    """
    the build of each table the response read
    """
    description: Annotated[str | None, Field(title="Description")]
    """
    the node's prose in the graph; null when it has none
    """
    gaps: Annotated[list[str], Field(title="Gaps")]
    """
    the member path of every value this door cannot know
    """
    kind: Annotated[Literal["source"], Field(title="Kind")]
    label: Annotated[str | None, Field(title="Label")]
    """
    the node's name; for a Provenance, its parent Dataset's name where it has one. Null, and named in gaps, when the graph gives none
    """
    links: Annotated[
        dict[
            Literal["codes", "children", "members", "countries", "datasets", "variables", "facets"],
            str,
        ],
        Field(title="Links"),
    ]
    """
    each list relation the node has, mapped to its route; empty when none
    """
    ref: Annotated[str, Field(title="Ref")]
    """
    the node's dcid
    """
    source: SourceBlock
    types: Annotated[list[str], Field(title="Types")]
    """
    the node's typeOf values, as the graph holds them
    """


class UnitCount(BaseModel):
    """
    One unit of a series. `ref` is `null` for facets that state no unit.
    """

    label: Annotated[str | None, Field(title="Label")]
    places: Annotated[int | None, Field(title="Places")]
    """
    the places its facets cover; null for a role-bearing variable
    """
    ref: Annotated[str | None, Field(title="Ref")]


class UnitsFold(BaseModel):
    """
    A series' units: how many, the two covering the most places, and how many more.
    """

    count: Annotated[int, Field(title="Count")]
    leading: Annotated[list[UnitCount], Field(title="Leading")]
    more: Annotated[int, Field(title="More")]
    national_currency: NationalCurrency | None
    """
    the units that are their places' national currency; null when none is
    """


class VariableDefinition(BaseModel):
    """
    What a variable measures, part by part.

    `constraints` are ordered by property dcid: the code index holds a variable's constraints
    as a mapping with no declared order, and the graph's `constraintProperties` arc has none.
    """

    constraints: Annotated[list[Constraint] | None, Field(title="Constraints")]
    """
    each constraint the measure does not state, ordered by property dcid; null when the read of a variable's constraint values failed
    """
    definitions: Annotated[list[DefinitionText], Field(title="Definitions")]
    """
    the graph's prose per part: the variable's own, then the measure's, then each constraint value's, in constraint order; a part with none is left out
    """
    measure: Measure | None
    """
    the measure the code index names; null when it names none
    """
    measured_property: NodeRef | None
    """
    the measuredProperty; null when the variable declares none
    """
    population_type: NodeRef | None
    """
    the populationType; null when the variable declares none
    """
    roles: Annotated[list[str], Field(title="Roles")]
    """
    the entity roles the variable's observations carry, from its declared observationProperties, defaulting to observationAbout when it declares none
    """
    source_indicator_codes: Annotated[list[str], Field(title="Source Indicator Codes")]
    """
    the sourceIndicatorCode arc's values: the codes naming the source series, as the publisher codes it (IMF WEO GDP carries NGDP and NGDPD); empty when it has none
    """
    stat_type: NodeRef | None
    """
    the statType; null when the variable declares none or its read failed
    """


class AggregateNode(BaseModel):
    aggregate: PartyBlock
    built_from: Annotated[
        dict[Literal["code_index", "origin_catalogue", "coverage_index", "membership"], str],
        Field(title="Built From"),
    ]
    """
    the build of each table the response read
    """
    description: Annotated[str | None, Field(title="Description")]
    """
    the node's prose in the graph; null when it has none
    """
    gaps: Annotated[list[str], Field(title="Gaps")]
    """
    the member path of every value this door cannot know
    """
    kind: Annotated[Literal["aggregate"], Field(title="Kind")]
    label: Annotated[str | None, Field(title="Label")]
    """
    the node's name; for a Provenance, its parent Dataset's name where it has one. Null, and named in gaps, when the graph gives none
    """
    links: Annotated[
        dict[
            Literal["codes", "children", "members", "countries", "datasets", "variables", "facets"],
            str,
        ],
        Field(title="Links"),
    ]
    """
    each list relation the node has, mapped to its route; empty when none
    """
    ref: Annotated[str, Field(title="Ref")]
    """
    the node's dcid
    """
    types: Annotated[list[str], Field(title="Types")]
    """
    the node's typeOf values, as the graph holds them
    """


class AvailabilityDefault(BaseModel):
    """
    What a data request naming no place, years or facet receives.
    """

    facets: Annotated[list[str], Field(title="Facets")]
    """
    the facet ids the request is served, in order
    """
    first_year: Annotated[str | None, Field(title="First Year")]
    last_year: Annotated[str | None, Field(title="Last Year")]
    measurement_method: NodeRef | None
    observation_period: Annotated[str | None, Field(title="Observation Period")]
    places: Annotated[int | None, Field(title="Places")]
    """
    null for a role-bearing variable
    """
    source: AvailabilitySource | None
    unit: NodeRef | None
    without_data: Annotated[int, Field(title="Without Data")]
    """
    the places, or parties, holding the variable that the default leaves without data
    """


class AvailabilitySummary(BaseModel):
    """
    The variable across every source.
    """

    entities: Annotated[dict[str, EntityCounts] | None, Field(title="Entities")]
    """
    the entities covered per graph role; null for any other variable
    """
    facets: Annotated[int, Field(title="Facets")]
    first_year: Annotated[str | None, Field(title="First Year")]
    last_year: Annotated[str | None, Field(title="Last Year")]
    places: Annotated[dict[str, int] | None, Field(title="Places")]
    """
    the places covered by kind; null for a role-bearing variable, and while the places' kinds are unread (a transient gap)
    """
    sources: Annotated[int, Field(title="Sources")]


class CodeSystemBlock(BaseModel):
    """
    A code system's detail block.
    """

    code_count: Annotated[int, Field(title="Code Count")]
    """
    how many codes the code index holds for the system
    """
    declarations: CodeSystemDeclarations
    plain_name: Annotated[str, Field(title="Plain Name")]
    """
    the property's plain name, such as `sector`
    """
    property: Annotated[str, Field(title="Property")]
    """
    the constraint property the system governs
    """


class CodeSystemNode(BaseModel):
    built_from: Annotated[
        dict[Literal["code_index", "origin_catalogue", "coverage_index", "membership"], str],
        Field(title="Built From"),
    ]
    """
    the build of each table the response read
    """
    code_system: CodeSystemBlock
    description: Annotated[str | None, Field(title="Description")]
    """
    the node's prose in the graph; null when it has none
    """
    gaps: Annotated[list[str], Field(title="Gaps")]
    """
    the member path of every value this door cannot know
    """
    kind: Annotated[Literal["code_system"], Field(title="Kind")]
    label: Annotated[str | None, Field(title="Label")]
    """
    the node's name; for a Provenance, its parent Dataset's name where it has one. Null, and named in gaps, when the graph gives none
    """
    links: Annotated[
        dict[
            Literal["codes", "children", "members", "countries", "datasets", "variables", "facets"],
            str,
        ],
        Field(title="Links"),
    ]
    """
    each list relation the node has, mapped to its route; empty when none
    """
    ref: Annotated[str, Field(title="Ref")]
    """
    the node's dcid
    """
    types: Annotated[list[str], Field(title="Types")]
    """
    the node's typeOf values, as the graph holds them
    """


class DatasetNode(BaseModel):
    built_from: Annotated[
        dict[Literal["code_index", "origin_catalogue", "coverage_index", "membership"], str],
        Field(title="Built From"),
    ]
    """
    the build of each table the response read
    """
    dataset: SourceBlock
    description: Annotated[str | None, Field(title="Description")]
    """
    the node's prose in the graph; null when it has none
    """
    gaps: Annotated[list[str], Field(title="Gaps")]
    """
    the member path of every value this door cannot know
    """
    kind: Annotated[Literal["dataset"], Field(title="Kind")]
    label: Annotated[str | None, Field(title="Label")]
    """
    the node's name; for a Provenance, its parent Dataset's name where it has one. Null, and named in gaps, when the graph gives none
    """
    links: Annotated[
        dict[
            Literal["codes", "children", "members", "countries", "datasets", "variables", "facets"],
            str,
        ],
        Field(title="Links"),
    ]
    """
    each list relation the node has, mapped to its route; empty when none
    """
    ref: Annotated[str, Field(title="Ref")]
    """
    the node's dcid
    """
    types: Annotated[list[str], Field(title="Types")]
    """
    the node's typeOf values, as the graph holds them
    """


class MembershipBlock(BaseModel):
    """
    A group's or region's membership under its declared authority.
    """

    as_of: Annotated[date | None, Field(title="As Of")]
    """
    the date the read asked for; null for today
    """
    authority: Authority
    effective_date: Annotated[date | None, Field(title="Effective Date")]
    """
    the latest membership change on or before the date; null for an undated membership
    """
    projections: Annotated[list[ProjectionCount], Field(title="Projections")]


class OrganisationNode(BaseModel):
    built_from: Annotated[
        dict[Literal["code_index", "origin_catalogue", "coverage_index", "membership"], str],
        Field(title="Built From"),
    ]
    """
    the build of each table the response read
    """
    description: Annotated[str | None, Field(title="Description")]
    """
    the node's prose in the graph; null when it has none
    """
    gaps: Annotated[list[str], Field(title="Gaps")]
    """
    the member path of every value this door cannot know
    """
    kind: Annotated[Literal["organisation"], Field(title="Kind")]
    label: Annotated[str | None, Field(title="Label")]
    """
    the node's name; for a Provenance, its parent Dataset's name where it has one. Null, and named in gaps, when the graph gives none
    """
    links: Annotated[
        dict[
            Literal["codes", "children", "members", "countries", "datasets", "variables", "facets"],
            str,
        ],
        Field(title="Links"),
    ]
    """
    each list relation the node has, mapped to its route; empty when none
    """
    organisation: PartyBlock
    ref: Annotated[str, Field(title="Ref")]
    """
    the node's dcid
    """
    types: Annotated[list[str], Field(title="Types")]
    """
    the node's typeOf values, as the graph holds them
    """


class PlaceBlock(BaseModel):
    """
    A place's detail block. `membership` is set for a region the membership policy holds.
    """

    contained_in: Annotated[list[NodeRef], Field(title="Contained In")]
    """
    the containedInPlace targets
    """
    membership: MembershipBlock | None


class PlaceOtherNode(BaseModel):
    built_from: Annotated[
        dict[Literal["code_index", "origin_catalogue", "coverage_index", "membership"], str],
        Field(title="Built From"),
    ]
    """
    the build of each table the response read
    """
    description: Annotated[str | None, Field(title="Description")]
    """
    the node's prose in the graph; null when it has none
    """
    gaps: Annotated[list[str], Field(title="Gaps")]
    """
    the member path of every value this door cannot know
    """
    kind: Annotated[Literal["place_other"], Field(title="Kind")]
    label: Annotated[str | None, Field(title="Label")]
    """
    the node's name; for a Provenance, its parent Dataset's name where it has one. Null, and named in gaps, when the graph gives none
    """
    links: Annotated[
        dict[
            Literal["codes", "children", "members", "countries", "datasets", "variables", "facets"],
            str,
        ],
        Field(title="Links"),
    ]
    """
    each list relation the node has, mapped to its route; empty when none
    """
    place_other: PlaceBlock
    ref: Annotated[str, Field(title="Ref")]
    """
    the node's dcid
    """
    types: Annotated[list[str], Field(title="Types")]
    """
    the node's typeOf values, as the graph holds them
    """


class ProvenanceNode(BaseModel):
    built_from: Annotated[
        dict[Literal["code_index", "origin_catalogue", "coverage_index", "membership"], str],
        Field(title="Built From"),
    ]
    """
    the build of each table the response read
    """
    description: Annotated[str | None, Field(title="Description")]
    """
    the node's prose in the graph; null when it has none
    """
    gaps: Annotated[list[str], Field(title="Gaps")]
    """
    the member path of every value this door cannot know
    """
    kind: Annotated[Literal["provenance"], Field(title="Kind")]
    label: Annotated[str | None, Field(title="Label")]
    """
    the node's name; for a Provenance, its parent Dataset's name where it has one. Null, and named in gaps, when the graph gives none
    """
    links: Annotated[
        dict[
            Literal["codes", "children", "members", "countries", "datasets", "variables", "facets"],
            str,
        ],
        Field(title="Links"),
    ]
    """
    each list relation the node has, mapped to its route; empty when none
    """
    provenance: SourceBlock
    ref: Annotated[str, Field(title="Ref")]
    """
    the node's dcid
    """
    types: Annotated[list[str], Field(title="Types")]
    """
    the node's typeOf values, as the graph holds them
    """


class Series(BaseModel):
    """
    A source's facets sharing a measurement method, period and scaling.
    """

    binding: Binding
    first_year: Annotated[str | None, Field(title="First Year")]
    last_year: Annotated[str | None, Field(title="Last Year")]
    measurement_method: NodeRef | None
    """
    null when no facet states a method: no stated method
    """
    observation_period: Annotated[str | None, Field(title="Observation Period")]
    observations: Annotated[int, Field(title="Observations")]
    places: Annotated[int | None, Field(title="Places")]
    """
    null for a role-bearing variable
    """
    scaling: Annotated[str | None, Field(title="Scaling")]
    units: UnitsFold


class SourceAvailability(BaseModel):
    """
    One source's facets of the variable, and their series.
    """

    entities: Annotated[dict[str, EntityCounts] | None, Field(title="Entities")]
    """
    the entities covered per graph role; null for any other variable
    """
    facets: Annotated[int, Field(title="Facets")]
    first_year: Annotated[str | None, Field(title="First Year")]
    last_year: Annotated[str | None, Field(title="Last Year")]
    places: Annotated[dict[str, int] | None, Field(title="Places")]
    """
    the places covered by kind; null for a role-bearing variable, and while the places' kinds are unread (a transient gap)
    """
    series: Annotated[list[Series], Field(title="Series")]
    source: AvailabilitySource


class SubregionNode(BaseModel):
    built_from: Annotated[
        dict[Literal["code_index", "origin_catalogue", "coverage_index", "membership"], str],
        Field(title="Built From"),
    ]
    """
    the build of each table the response read
    """
    description: Annotated[str | None, Field(title="Description")]
    """
    the node's prose in the graph; null when it has none
    """
    gaps: Annotated[list[str], Field(title="Gaps")]
    """
    the member path of every value this door cannot know
    """
    kind: Annotated[Literal["subregion"], Field(title="Kind")]
    label: Annotated[str | None, Field(title="Label")]
    """
    the node's name; for a Provenance, its parent Dataset's name where it has one. Null, and named in gaps, when the graph gives none
    """
    links: Annotated[
        dict[
            Literal["codes", "children", "members", "countries", "datasets", "variables", "facets"],
            str,
        ],
        Field(title="Links"),
    ]
    """
    each list relation the node has, mapped to its route; empty when none
    """
    ref: Annotated[str, Field(title="Ref")]
    """
    the node's dcid
    """
    subregion: PlaceBlock
    types: Annotated[list[str], Field(title="Types")]
    """
    the node's typeOf values, as the graph holds them
    """


class UnitBlock(BaseModel):
    """
    A unit's detail block: the words `unit_words` names it by.
    """

    changes_magnitude: Annotated[bool, Field(title="Changes Magnitude")]
    """
    whether the unit changes what a number means, so a heading states it
    """
    currency: Annotated[str | None, Field(title="Currency")]
    """
    its ISO 4217 code when it is a currency
    """
    currency_noun: Annotated[str | None, Field(title="Currency Noun")]
    """
    the plural noun a currency reads as, such as `US dollars`
    """
    price_basis: PriceBasisWords | None
    """
    the price basis the unit itself states; null when it states none
    """
    unit_class: Literal["currency", "purchasing_power", "percent", "other", "none"]
    """
    currency, purchasing_power, percent, or other (`unit_words.unit_class`)
    """


class UnitNode(BaseModel):
    built_from: Annotated[
        dict[Literal["code_index", "origin_catalogue", "coverage_index", "membership"], str],
        Field(title="Built From"),
    ]
    """
    the build of each table the response read
    """
    description: Annotated[str | None, Field(title="Description")]
    """
    the node's prose in the graph; null when it has none
    """
    gaps: Annotated[list[str], Field(title="Gaps")]
    """
    the member path of every value this door cannot know
    """
    kind: Annotated[Literal["unit"], Field(title="Kind")]
    label: Annotated[str | None, Field(title="Label")]
    """
    the node's name; for a Provenance, its parent Dataset's name where it has one. Null, and named in gaps, when the graph gives none
    """
    links: Annotated[
        dict[
            Literal["codes", "children", "members", "countries", "datasets", "variables", "facets"],
            str,
        ],
        Field(title="Links"),
    ]
    """
    each list relation the node has, mapped to its route; empty when none
    """
    ref: Annotated[str, Field(title="Ref")]
    """
    the node's dcid
    """
    types: Annotated[list[str], Field(title="Types")]
    """
    the node's typeOf values, as the graph holds them
    """
    unit: UnitBlock


class Availability(BaseModel):
    """
    Where DFO holds observations of a variable. `sources: []` means it verifiably holds
    none.
    """

    default: AvailabilityDefault | None
    """
    null when no facet dispatches a request naming nothing
    """
    more_sources: Annotated[int, Field(title="More Sources")]
    """
    the sources past the first ten
    """
    scope: Annotated[Literal["dfo"], Field(title="Scope")]
    sources: Annotated[list[SourceAvailability], Field(title="Sources")]
    summary: AvailabilitySummary


class ContinentNode(BaseModel):
    built_from: Annotated[
        dict[Literal["code_index", "origin_catalogue", "coverage_index", "membership"], str],
        Field(title="Built From"),
    ]
    """
    the build of each table the response read
    """
    continent: PlaceBlock
    description: Annotated[str | None, Field(title="Description")]
    """
    the node's prose in the graph; null when it has none
    """
    gaps: Annotated[list[str], Field(title="Gaps")]
    """
    the member path of every value this door cannot know
    """
    kind: Annotated[Literal["continent"], Field(title="Kind")]
    label: Annotated[str | None, Field(title="Label")]
    """
    the node's name; for a Provenance, its parent Dataset's name where it has one. Null, and named in gaps, when the graph gives none
    """
    links: Annotated[
        dict[
            Literal["codes", "children", "members", "countries", "datasets", "variables", "facets"],
            str,
        ],
        Field(title="Links"),
    ]
    """
    each list relation the node has, mapped to its route; empty when none
    """
    ref: Annotated[str, Field(title="Ref")]
    """
    the node's dcid
    """
    types: Annotated[list[str], Field(title="Types")]
    """
    the node's typeOf values, as the graph holds them
    """


class ContinentalUnionNode(BaseModel):
    built_from: Annotated[
        dict[Literal["code_index", "origin_catalogue", "coverage_index", "membership"], str],
        Field(title="Built From"),
    ]
    """
    the build of each table the response read
    """
    continental_union: PlaceBlock
    description: Annotated[str | None, Field(title="Description")]
    """
    the node's prose in the graph; null when it has none
    """
    gaps: Annotated[list[str], Field(title="Gaps")]
    """
    the member path of every value this door cannot know
    """
    kind: Annotated[Literal["continental_union"], Field(title="Kind")]
    label: Annotated[str | None, Field(title="Label")]
    """
    the node's name; for a Provenance, its parent Dataset's name where it has one. Null, and named in gaps, when the graph gives none
    """
    links: Annotated[
        dict[
            Literal["codes", "children", "members", "countries", "datasets", "variables", "facets"],
            str,
        ],
        Field(title="Links"),
    ]
    """
    each list relation the node has, mapped to its route; empty when none
    """
    ref: Annotated[str, Field(title="Ref")]
    """
    the node's dcid
    """
    types: Annotated[list[str], Field(title="Types")]
    """
    the node's typeOf values, as the graph holds them
    """


class CountryNode(BaseModel):
    built_from: Annotated[
        dict[Literal["code_index", "origin_catalogue", "coverage_index", "membership"], str],
        Field(title="Built From"),
    ]
    """
    the build of each table the response read
    """
    country: PlaceBlock
    description: Annotated[str | None, Field(title="Description")]
    """
    the node's prose in the graph; null when it has none
    """
    gaps: Annotated[list[str], Field(title="Gaps")]
    """
    the member path of every value this door cannot know
    """
    kind: Annotated[Literal["country"], Field(title="Kind")]
    label: Annotated[str | None, Field(title="Label")]
    """
    the node's name; for a Provenance, its parent Dataset's name where it has one. Null, and named in gaps, when the graph gives none
    """
    links: Annotated[
        dict[
            Literal["codes", "children", "members", "countries", "datasets", "variables", "facets"],
            str,
        ],
        Field(title="Links"),
    ]
    """
    each list relation the node has, mapped to its route; empty when none
    """
    ref: Annotated[str, Field(title="Ref")]
    """
    the node's dcid
    """
    types: Annotated[list[str], Field(title="Types")]
    """
    the node's typeOf values, as the graph holds them
    """


class GroupNode(BaseModel):
    built_from: Annotated[
        dict[Literal["code_index", "origin_catalogue", "coverage_index", "membership"], str],
        Field(title="Built From"),
    ]
    """
    the build of each table the response read
    """
    description: Annotated[str | None, Field(title="Description")]
    """
    the node's prose in the graph; null when it has none
    """
    gaps: Annotated[list[str], Field(title="Gaps")]
    """
    the member path of every value this door cannot know
    """
    group: MembershipBlock
    kind: Annotated[Literal["group"], Field(title="Kind")]
    label: Annotated[str | None, Field(title="Label")]
    """
    the node's name; for a Provenance, its parent Dataset's name where it has one. Null, and named in gaps, when the graph gives none
    """
    links: Annotated[
        dict[
            Literal["codes", "children", "members", "countries", "datasets", "variables", "facets"],
            str,
        ],
        Field(title="Links"),
    ]
    """
    each list relation the node has, mapped to its route; empty when none
    """
    ref: Annotated[str, Field(title="Ref")]
    """
    the node's dcid
    """
    types: Annotated[list[str], Field(title="Types")]
    """
    the node's typeOf values, as the graph holds them
    """


class StatVarBlock(BaseModel):
    """
    A variable's detail block: what it measures and where DFO holds observations of it.
    """

    availability: Availability | None
    """
    where DFO holds observations of the variable; null, and named in gaps, when its read failed or the coverage index is not loaded
    """
    definition: VariableDefinition


class StatVarNode(BaseModel):
    built_from: Annotated[
        dict[Literal["code_index", "origin_catalogue", "coverage_index", "membership"], str],
        Field(title="Built From"),
    ]
    """
    the build of each table the response read
    """
    description: Annotated[str | None, Field(title="Description")]
    """
    the node's prose in the graph; null when it has none
    """
    gaps: Annotated[list[str], Field(title="Gaps")]
    """
    the member path of every value this door cannot know
    """
    kind: Annotated[Literal["stat_var"], Field(title="Kind")]
    label: Annotated[str | None, Field(title="Label")]
    """
    the node's name; for a Provenance, its parent Dataset's name where it has one. Null, and named in gaps, when the graph gives none
    """
    links: Annotated[
        dict[
            Literal["codes", "children", "members", "countries", "datasets", "variables", "facets"],
            str,
        ],
        Field(title="Links"),
    ]
    """
    each list relation the node has, mapped to its route; empty when none
    """
    ref: Annotated[str, Field(title="Ref")]
    """
    the node's dcid
    """
    stat_var: StatVarBlock
    types: Annotated[list[str], Field(title="Types")]
    """
    the node's typeOf values, as the graph holds them
    """


class NodesResponse(BaseModel):
    """
    `nodes` is index-aligned with the request's `dcids`: a repeated dcid repeats its
    entry. Each entry is the body `GET /api/graph/node/{dcid}` returns for that dcid, or a
    `not_held` entry where that route answers 404.
    """

    nodes: Annotated[
        list[
            StatVarNode
            | CodeSystemNode
            | CodeNode
            | GroupNode
            | CountryNode
            | ContinentNode
            | SubregionNode
            | ContinentalUnionNode
            | PlaceOtherNode
            | OrganisationNode
            | AggregateNode
            | UnitNode
            | SourceNode
            | DatasetNode
            | ProvenanceNode
            | OtherNode
            | NotHeldEntry
        ],
        Field(title="Nodes"),
    ]


class NodeResponse(
    RootModel[
        StatVarNode
        | CodeSystemNode
        | CodeNode
        | GroupNode
        | CountryNode
        | ContinentNode
        | SubregionNode
        | ContinentalUnionNode
        | PlaceOtherNode
        | OrganisationNode
        | AggregateNode
        | UnitNode
        | SourceNode
        | DatasetNode
        | ProvenanceNode
        | OtherNode
    ]
):
    root: Annotated[
        StatVarNode
        | CodeSystemNode
        | CodeNode
        | GroupNode
        | CountryNode
        | ContinentNode
        | SubregionNode
        | ContinentalUnionNode
        | PlaceOtherNode
        | OrganisationNode
        | AggregateNode
        | UnitNode
        | SourceNode
        | DatasetNode
        | ProvenanceNode
        | OtherNode,
        Field(discriminator="kind"),
    ]
