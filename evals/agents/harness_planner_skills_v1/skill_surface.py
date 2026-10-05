"""A canned data-mcp skill surface: `get_skill` and `get_skill_file` over fixed skills.

The planner reads skills through `planner.skills.build_skill_tools`, which
calls data-mcp's `get_skill` and `get_skill_file` through a
`compiler.surface.ToolSurface`. `SkillSurface` is that surface with no server
behind it, so a run loads the same text every time and never reaches a Skills
API.

Each skill's `id` and `description` are the dfo-skills catalogue's own,
since the description is the text the planner matches a question against.
The SKILL.md bodies are short stand-ins: the set scores whether a skill is
loaded, not what the planner does with its contents. Answers take the shape
data-mcp's `get_skill` renders (SKILL.md verbatim, then the file list naming
`get_skill_file`), and an id outside the catalogue refuses `skill_not_found`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from compiler.surface import ProgressCallback, ToolCallResult, ToolSchema
from planner.skills import GET_SKILL, GET_SKILL_FILE
from planner.workflow import ToolId


@dataclass(frozen=True)
class FixtureSkill:
    """One skill as the eval serves it: its catalogue entry and its files."""

    id: str
    description: str
    body: str
    files: Mapping[str, str]

    def skill_md(self) -> str:
        return f"---\nname: {self.id}\ndescription: >-\n  {self.description}\n---\n\n{self.body}\n"


SKILLS: dict[str, FixtureSkill] = {
    skill.id: skill
    for skill in (
        FixtureSkill(
            id="oda-measures",
            description=(
                "How the OECD DAC measures and reports ODA. Covers the flow types (gross "
                "disbursements, net disbursements, grant equivalents and commitments) and "
                "which questions each fits, current and constant prices, DAC deflators and "
                "base years, and the 2018 switch to the grant-equivalent headline. Also "
                "covers how grants, concessional loans and private sector instruments (PSI) "
                "qualify as ODA and are counted, including grant elements, discount rates, "
                "additionality, the institutional and instrument approaches, PSI transition "
                "arrangements and the policy debate on loan and PSI measurement. Relevant to "
                "choosing a measure, comparing ODA across donors or years, explaining what an "
                "ODA figure represents, and assessing how a financing instrument counts in ODA."
            ),
            body=(
                "# ODA measures\n\nSince 2018 the DAC headline is the grant equivalent. "
                "Net disbursements remain the series for comparisons before 2018. State the "
                "flow type and price basis of every ODA figure."
            ),
            files={
                "references/grant-equivalents.md": "# Grant equivalents\n\nLoans count by their grant element.\n"
            },
        ),
        FixtureSkill(
            id="oda-concepts",
            description=(
                "Core definitions, measures and classifications in ODA analysis. Covers the "
                "0.7% ODA/GNI target and how ODA/GNI is calculated, the grant equivalent "
                "methodology and its 2018 break in comparability, Core ODA and the items it "
                "excludes (in-donor refugee costs, imputed student costs, debt relief), donor "
                "groupings (DAC members, DAC associates, non-DAC donors and EU Institutions, "
                "with the double-counting risk EU Institutions bring), ONE's sector groupings "
                "by CRS purpose code, and indicators of donor prioritisation such as sector "
                "shares and the bilateral/multilateral split."
            ),
            body=(
                "# ODA concepts\n\nODA/GNI divides a donor's grant-equivalent ODA by its GNI. "
                "The 0.7% target applies to DAC members."
            ),
            files={"references/core-oda.md": "# Core ODA\n\nCore ODA excludes in-donor costs.\n"},
        ),
        FixtureSkill(
            id="oda-crs",
            description=(
                "Context for working with the OECD Creditor Reporting System (CRS), the "
                "activity-level dataset of ODA, OOF, private sector instruments and private "
                "philanthropy reported by bilateral providers, multilateral organisations and "
                "foundations. Covers its attributes, coverage by year and against DAC1, the "
                "flow categories and bilateral/multilateral flags, the April 2026 core "
                "contribution records and their treatment in the DFO, and the double-counting "
                "traps with multilateral outflows, foundation funding and regional buckets. "
                "Load once the CRS has been chosen for an analysis, or when working with CRS "
                "records. For choosing between DAC datasets, use oda-dataset-selection."
            ),
            body=(
                "# Creditor Reporting System\n\nCRS records carry a purpose code per "
                "activity. Sector totals sum activities by purpose code; exclude regional "
                "buckets when summing by recipient."
            ),
            files={
                "references/purpose-codes.md": "# Purpose codes\n\n5-digit CRS codes nest in 3-digit sectors.\n"
            },
        ),
        FixtureSkill(
            id="oda-dac1",
            description=(
                "Context for working with the OECD DAC1 table, the donor-level totals of ODA, "
                "OOF, export credits and private flows by aid type. Covers who reports, the "
                "flow types (grant equivalent, net and gross disbursements, commitments) and "
                "price bases, what the ODA total includes (in-donor costs and core multilateral "
                "contributions), private sector instruments, ODA/GNI, and the double-counting "
                "traps with donor aggregates, EU Institutions, nested aid-type lines and memo "
                "items. Load once DAC1 has been chosen for an analysis, or when working with "
                "DAC1 figures. For choosing between DAC datasets, use oda-dataset-selection."
            ),
            body="# DAC1\n\nDonor totals by aid type. Do not sum nested aid-type lines.",
            files={},
        ),
        FixtureSkill(
            id="oda-dac2a",
            description=(
                "Context for working with the OECD DAC2A table, ODA disbursements by donor "
                "and recipient country or region. Covers its measures, multilateral agencies "
                "reporting as donors and appearing as recipients, how its totals reconcile "
                "with DAC1, the OECD's imputed multilateral ODA by recipient, and the "
                "double-counting traps with donor and recipient aggregates, EU aggregates and "
                "imputed figures. Load once DAC2A has been chosen for an analysis, or when "
                "working with DAC2A figures. For choosing between DAC datasets, use "
                "oda-dataset-selection."
            ),
            body="# DAC2A\n\nODA by recipient. Exclude regional aggregates when summing.",
            files={},
        ),
        FixtureSkill(
            id="policy-markers",
            description=(
                "Covers OECD DAC policy markers used to classify bilateral ODA in the "
                "Creditor Reporting System (CRS). Use when the user asks what a policy marker "
                "is, which markers exist, how to score an activity, how to interpret gender "
                "or climate marker data, or how to analyse ODA by policy objective."
            ),
            body=(
                "# Policy markers\n\nEach marker scores an activity 2 (principal), 1 "
                "(significant) or 0. Report principal and significant separately."
            ),
            files={
                "references/gender.md": "# Gender marker\n\nThe gender marker is DAC's gender equality marker.\n"
            },
        ),
    )
}


def _text(skill: FixtureSkill) -> str:
    """`get_skill`'s answer as data-mcp renders it: SKILL.md, then the file list."""
    skill_md = skill.skill_md()
    if not skill.files:
        return skill_md
    listing = "\n".join(
        f"- {path} ({len(text.encode())} B)" for path, text in sorted(skill.files.items())
    )
    return (
        f"{skill_md}\nFiles in this skill; read one with "
        f'get_skill_file(skill_id="{skill.id}", path=…):\n{listing}'
    )


def _refusal(code: str, message: str) -> ToolCallResult:
    return ToolCallResult(
        is_error=True, structured_content={"code": code, "detail": message}, text=message
    )


class SkillSurface:
    """A `ToolSurface` serving `SKILLS` through `get_skill` and `get_skill_file` alone."""

    async def list_tools(self) -> list[ToolSchema]:
        return []

    async def call_tool(
        self,
        *,
        tool: ToolId,
        arguments: Mapping[str, object],
        on_progress: ProgressCallback | None = None,
    ) -> ToolCallResult:
        skill = SKILLS.get(str(arguments.get("skill_id")))
        if skill is None:
            return _refusal(
                "skill_not_found",
                f"No skill has the id {arguments.get('skill_id')!r}; use an id a search "
                "result names.",
            )
        if tool == GET_SKILL:
            return ToolCallResult(is_error=False, text=_text(skill))
        if tool == GET_SKILL_FILE:
            path = str(arguments.get("path"))
            text = skill.skill_md() if path == "SKILL.md" else skill.files.get(path)
            if text is None:
                return _refusal(
                    "skill_resource_not_found",
                    f"{skill.id} has no file {path!r}; call get_skill for its file list.",
                )
            return ToolCallResult(is_error=False, text=text)
        raise ValueError(f"SkillSurface serves get_skill and get_skill_file, not {tool.name}")
