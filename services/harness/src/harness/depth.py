"""The harness's depth policy: how much dialogue a question requires.

A pure function and intentionally not a model call. Depth decided by an LLM is
neither reproducible in evals nor explainable in a trace, and the signals
that matter are cheap and structural: which surface asked, whether Search
resolved the request cleanly, how many methodological choice points the
resolved definition touches, and whether the question is compound.

Inputs are plain scalars on purpose: the module takes a bare integer for
`choice_points` regardless of where it comes from (facet axes on the
definition, skill-declared forks, or a future envelope field), so the
module's contract stays stable as that source evolves.

`depth="none"` skips the planner entirely: a definite, single-part
resolution with no choice points is the search-and-download path, and
asking the user to approve a plan for it is friction with no payoff.

`depth_hint` is the one non-structural input: the user's own stated
preference ("answer quickly" / "give me a comprehensive overview"),
classified by a small model (`harness.depth_hints`) into `quick` / `thorough` /
absent. The model flags; this function decides. A hint adjusts the
structural verdict by one level with guardrails -- `quick` keeps the
planner while methodological choice points exist (clarifying questions
stay available as needed) and floors a degraded resolution at standard;
`thorough` on a definite lookup invokes the planner it would otherwise
skip. The reason string records both the structural verdict and the
hint's effect.

Every verdict also carries `notice`: a single plain sentence, aimed at the
user rather than a trace, saying why a plan is being proposed at all. It is
`None` for `depth="none"`, since no plan card is shown.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

DepthLevel = Literal["none", "shallow", "standard", "deep"]
DepthHint = Literal["quick", "thorough"]

_LEVELS: tuple[DepthLevel, ...] = ("none", "shallow", "standard", "deep")

# Per-level dialogue budgets: (max clarifying questions, may use defaults).
_BUDGETS: dict[DepthLevel, tuple[int, bool]] = {
    "none": (0, True),
    "shallow": (1, True),
    "standard": (2, True),
    "deep": (3, False),
}

# A compound question (two or more search parts) floors at "standard":
# cross-frame work is a near-guarantee of methodological forks (unit bases,
# price bases, period intersection) even when each part resolved cleanly.
# The floor is on part count alone. `harness.graph`'s `approve` node opens
# a `plan.approved` span on every approval, and its `dfo.parts`,
# `dfo.choice_points` and `dfo.plan_version` attributes together show how
# often a compound plan reaches approval at version 1 with no choice
# points, the case this floor treats as needing a plan anyway.
_COMPOUND_PART_FLOOR = 2

# Three or more forks (or a degraded resolution) is past the point where
# defaults silently settle things a domain expert would insist on being
# asked about.
_DEEP_CHOICE_POINTS = 3


class DepthSignal(BaseModel):
    """The depth verdict, plus the dialogue budget it grants the planner.

    `reason` is an engineering trace string; `notice` is the sentence shown
    to the user on the plan card, explaining in plain language why a plan is
    being proposed at all. `notice` is `None` when `depth="none"`, since no
    plan card is shown.
    """

    depth: DepthLevel
    max_clarifying_questions: int
    may_use_defaults: bool
    reason: str
    notice: str | None = None


def decide_depth(
    *,
    status: str,
    choice_points: int,
    degraded: bool,
    part_count: int,
    surface: str,
    depth_hint: DepthHint | None = None,
    wide: bool = False,
    operations: int = 0,
) -> DepthSignal:
    """Map resolve-time signals (plus the user's stated preference) to a verdict.

    `degraded` beats any confidence reading as an input: the search contract
    is explicit that confidence is for display emphasis, while a degraded
    warning is specific and actionable.

    `wide` is a separate input from `choice_points`:
    whether the resolved identity names a family rather than a single
    series (`harness.tool_contracts.graph_envelope`'s own structural
    reading). Folding it into `choice_points` would move the deep/standard
    thresholds as a side effect; kept separate, its only effect is refusing
    `depth="none"` to a wide identity, which skips the planner and the
    approval gate both -- on every surface but `download`, which returns
    before the wide check runs (see `_structural_verdict`).

    `operations` counts the analysis operations a seeded turn carries
    (`harness.graph`'s `seed` node: a Search result handed to Studio with
    its rank, growth or explain left over). `depth="none"` goes to
    `fetch`, which retrieves and renders and computes nothing, so any
    operation refuses it the same way `wide` does.
    """
    structural = _structural_verdict(
        status=status,
        choice_points=choice_points,
        degraded=degraded,
        part_count=part_count,
        surface=surface,
        wide=wide,
        operations=operations,
    )
    if depth_hint is None or surface == "download":
        return structural
    return _apply_hint(
        structural,
        hint=depth_hint,
        choice_points=choice_points,
        degraded=degraded,
        wide=wide,
        operations=operations,
    )


def _apply_hint(
    structural: DepthSignal,
    *,
    hint: DepthHint,
    choice_points: int,
    degraded: bool,
    wide: bool = False,
    operations: int = 0,
) -> DepthSignal:
    """Adjust the structural verdict one level in the hint's direction.

    Guardrails on `quick`: choice points keep the planner (floor shallow, so
    clarifying questions stay available as needed), a degraded resolution
    floors at standard, since data-quality problems need confirmation
    regardless of urgency, and a wide identity floors at shallow for the
    same reason as choice points: `_structural_verdict` already refuses
    `depth="none"` to a wide identity, and `quick` keeps it at `shallow`
    rather than pushing it lower. A turn carrying operations floors at
    shallow for the same reason. `thorough` floors at standard: a
    comprehensive treatment of even a clean lookup deserves a plan.

    `notice` carries over from `structural` unchanged, except when
    `thorough` turns a `depth="none"` verdict into a plan: `structural.notice`
    is `None` in that case, so this branch supplies its own sentence. A
    `thorough` hint that instead moves an already-planned question up a
    level (`shallow` to `standard`, say) keeps the structural notice, since
    the question already needed a plan for its own reasons and the hint only
    changes how much dialogue it gets.
    """
    index = _LEVELS.index(structural.depth)
    if hint == "thorough":
        adjusted = max(_LEVELS.index("standard"), min(index + 1, len(_LEVELS) - 1))
    else:
        floor = 0
        if choice_points >= 1:
            floor = max(floor, _LEVELS.index("shallow"))
        if wide or operations:
            floor = max(floor, _LEVELS.index("shallow"))
        if degraded:
            floor = max(floor, _LEVELS.index("standard"))
        adjusted = max(floor, index - 1)
    final = _LEVELS[adjusted]
    if final == structural.depth:
        return structural.model_copy(
            update={"reason": f"{structural.reason}; user asked for {hint}: already {final}"}
        )
    questions, defaults = _BUDGETS[final]
    if hint == "quick":
        defaults = True
    notice = (
        "You asked for a thorough treatment, so I am proposing a plan first."
        if hint == "thorough" and structural.depth == "none"
        else structural.notice
    )
    return DepthSignal(
        depth=final,
        max_clarifying_questions=questions,
        may_use_defaults=defaults,
        reason=f"{structural.reason}; user asked for {hint}: {structural.depth}->{final}",
        notice=notice,
    )


def _structural_verdict(
    *,
    status: str,
    choice_points: int,
    degraded: bool,
    part_count: int,
    surface: str,
    wide: bool = False,
    operations: int = 0,
) -> DepthSignal:
    if surface == "download":
        # Exempt from the wide floor below: a download asks to fetch what
        # matches, so a family-wide identity (`wide=True`) still returns
        # `depth="none"` here, skipping the `shallow` floor a wide identity
        # gets on every other surface.
        return DepthSignal(
            depth="none",
            max_clarifying_questions=0,
            may_use_defaults=True,
            reason="download surface asks for speed, not dialogue",
        )
    if (
        status == "definite"
        and not choice_points
        and not degraded
        and part_count < _COMPOUND_PART_FLOOR
    ):
        if wide or operations:
            # A wide identity names a family: `choice_points` is 0 (no facet
            # was defaulted or per-series) but the identity itself names
            # many members. A seeded turn's operations ask for analysis.
            # Either way `depth="none"` would skip straight to `fetch()`,
            # with no planner and no approval gate, so the verdict floors
            # at `shallow`. This floor applies only to non-download
            # surfaces; the download surface already returned above.
            reason, notice = (
                (
                    "definite single-part resolution, but the resolved identity "
                    "names a family rather than a single series",
                    "This matches a family of related series rather than a single one, "
                    "so I am proposing a plan to narrow it down.",
                )
                if wide
                else (
                    f"definite single-part resolution carrying {operations} analysis "
                    "operation(s) the fetch path cannot compute",
                    "You asked for analysis of this data, so I am proposing a plan for it.",
                )
            )
            return DepthSignal(
                depth="shallow",
                max_clarifying_questions=1,
                may_use_defaults=True,
                reason=reason,
                notice=notice,
            )
        return DepthSignal(
            depth="none",
            max_clarifying_questions=0,
            may_use_defaults=True,
            reason="definite single-part resolution, no methodological choice points",
        )
    if choice_points >= _DEEP_CHOICE_POINTS or degraded:
        notice = (
            "There is a data quality issue with what I found, so I am proposing a "
            "plan to work through it with you."
            if degraded
            else "This question has several real choices to settle, so I am proposing a plan first."
        )
        return DepthSignal(
            depth="deep",
            max_clarifying_questions=3,
            may_use_defaults=False,
            reason=f"{choice_points} choice points, degraded={degraded}",
            notice=notice,
        )
    if choice_points or part_count >= _COMPOUND_PART_FLOOR:
        notice = (
            "This question has a real choice to settle, so here is the plan first."
            if choice_points
            else "This asks for more than one thing, so I am proposing a plan before "
            "fetching anything."
        )
        return DepthSignal(
            depth="standard",
            max_clarifying_questions=2,
            may_use_defaults=True,
            reason=(
                f"{choice_points} choice point(s), {part_count} part(s) -- "
                "worth surfacing before computing"
            ),
            notice=notice,
        )
    return DepthSignal(
        depth="shallow",
        max_clarifying_questions=1,
        may_use_defaults=True,
        reason="resolved but not definite; one question at most",
        notice=(
            "I could not resolve this with full confidence, so I am proposing a "
            "quick plan to confirm before going further."
        ),
    )
