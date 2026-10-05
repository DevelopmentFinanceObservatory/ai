"""Guards this service against silent drift from the eval sets under `evals/`.

Those sets run the real agents through the real model, outside CI -- see
`evals/README.md`. What CI can do is notice when their cases are renamed,
added, or removed, when a case is written in a way the ADK metrics would score
meaninglessly, and when a seeded plan no longer matches the plan schema.
`EXPECTED_CASE_IDS` below mirrors each set's case IDs, so any drift between a
set and that mirror fails these tests.

That is all this file does. The agents' own logic -- the planner's confirmation
gate, the compiler's checks, the translator's discovery boundary -- is tested
model-free in `tests/planner/`, `tests/compiler/` and `tests/translator/`. This
file checks the shape of the case data, which is the part CI can verify with
no model in the loop.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from google.adk.agents import BaseAgent
from planner.plan import PLAN_STATE_KEY, Plan

REPO_ROOT = Path(__file__).resolve().parents[3]
_AGENTS_DIR = REPO_ROOT / "evals" / "agents"
if str(_AGENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_AGENTS_DIR))

# The tools `workflow_outcome` scores a `tool_responses` expectation for --
# imported rather than restated, so a tool the metric starts or stops scoring
# changes this guard with it.
# harness-planner-skills-v1's eval set is built from its cases: see
# `test_the_skills_eval_set_is_the_one_its_cases_build`.
from harness_planner_skills_v1.cases import eval_set as skills_eval_set  # noqa: E402
from harness_translator_v1.outcome_metric import SCORED_TOOLS  # noqa: E402

# Mirrored here so a rename/add/remove on either side fails CI, catching a
# stale reference while someone is still looking at it.
EXPECTED_CASE_IDS = {
    "harness-planner-v2": frozenset(
        {
            "two-dataset-comparison",
            "amended-mid-execution-reversions",
            "rejected-plan-blocks-execution",
            "adversarial-execute-before-approval",
        }
    ),
    "harness-pair-v1": frozenset(
        {
            "translates-before-inviting-approval",
            "takes-an-unmeetable-requirement-back-to-the-user",
        }
    ),
    "harness-planner-skills-v1": frozenset(
        {
            "grant-equivalent-measure",
            "oda-gni-target",
            "gender-marker-share",
            "crs-purpose-code-sector",
            "grant-equivalent-and-purpose-codes",
            "population-needs-no-skill",
        }
    ),
}

EVAL_SET_NAMES = sorted(EXPECTED_CASE_IDS)

# Each rubric-based metric declares the one `RUBRIC_TYPE` it will score and
# silently ignores every other, so a set's rubric types have to match the
# metrics its own `test_config.json` turns on. Mirrored from ADK's evaluators
# (`google.adk.evaluation.rubric_based_*`).
RUBRIC_TYPE_BY_METRIC = {
    "rubric_based_tool_use_quality_v1": "TOOL_USE_QUALITY",
    "rubric_based_final_response_quality_v1": "FINAL_RESPONSE_QUALITY",
}


def _agent_dir(eval_set_name: str) -> Path:
    """The agent folder is the set's own name with underscores, which is what ADK's
    CLI takes as its agent argument (see the `eval-*` recipes in the justfile)."""
    return REPO_ROOT / "evals" / "agents" / eval_set_name.replace("-", "_")


def _load(eval_set_name: str) -> dict:
    """Read one eval set by name."""
    return json.loads((_agent_dir(eval_set_name) / f"{eval_set_name}.evalset.json").read_text())


def _config(eval_set_name: str) -> dict:
    """Read one set's metric criteria, which decide what its rubrics are scored by."""
    return json.loads((_agent_dir(eval_set_name) / "test_config.json").read_text())


@pytest.mark.parametrize("eval_set_name", EVAL_SET_NAMES)
def test_eval_set_case_ids_have_not_drifted(eval_set_name: str) -> None:
    eval_set = _load(eval_set_name)

    case_ids = {case["eval_id"] for case in eval_set["eval_cases"]}

    assert case_ids == EXPECTED_CASE_IDS[eval_set_name]


@pytest.mark.parametrize("eval_set_name", EVAL_SET_NAMES)
def test_every_eval_set_turn_is_actually_scored(eval_set_name: str) -> None:
    # ADK's trajectory metric returns a pass for any turn with no expected tool
    # calls (`TrajectoryEvaluator._are_tool_calls_in_order_match` opens with
    # `if not expected_tool_calls: return True`), so a turn carrying neither a
    # `tool_uses` expectation nor a rubric scores 1.0 no matter what the agent
    # did. This asserts no such turn exists.
    # A turn stating a `workflow_outcome` expectation is scored too, on a set
    # that turns that metric on: a turn there can require no skill and still
    # fail when no plan is accepted.
    eval_set = _load(eval_set_name)
    scores_outcomes = "workflow_outcome" in _config(eval_set_name).get("criteria", {})

    unscored = [
        f"{case['eval_id']}:{turn['invocation_id']}"
        for case in eval_set["eval_cases"]
        for turn in case["conversation"]
        if not turn.get("rubrics")
        and not turn.get("intermediate_data", {}).get("tool_uses")
        and not (
            scores_outcomes
            and any(
                response.get("name") in SCORED_TOOLS
                for response in turn.get("intermediate_data", {}).get("tool_responses", [])
            )
        )
    ]

    assert unscored == []


@pytest.mark.parametrize("eval_set_name", EVAL_SET_NAMES)
def test_rubric_ids_are_unique(eval_set_name: str) -> None:
    # Rubric ids share one namespace across a whole set, so a duplicate aborts
    # the run part-way -- after earlier cases have been scored and paid for.
    eval_set = _load(eval_set_name)

    rubric_ids = [
        rubric["rubric_id"]
        for case in eval_set["eval_cases"]
        for turn in case["conversation"]
        for rubric in turn.get("rubrics", [])
    ]

    duplicates = sorted({rubric_id for rubric_id in rubric_ids if rubric_ids.count(rubric_id) > 1})

    assert duplicates == []


@pytest.mark.parametrize("eval_set_name", EVAL_SET_NAMES)
def test_every_rubric_is_typed_for_a_metric_the_set_configures(eval_set_name: str) -> None:
    # Each rubric-based metric filters a turn's rubrics by its own RUBRIC_TYPE,
    # so a rubric typed for a metric the set does not configure is dropped with
    # no warning -- the case still scores, just against fewer properties than
    # were written. An untyped one is dropped the same way, and a turn left with
    # no rubrics at all makes the metric raise "Rubrics are required." mid-run.
    eval_set = _load(eval_set_name)
    scored_types = {
        RUBRIC_TYPE_BY_METRIC[metric]
        for metric in _config(eval_set_name)["criteria"]
        if metric in RUBRIC_TYPE_BY_METRIC
    }

    unscored = [
        f"{rubric['rubric_id']} ({rubric.get('type')})"
        for case in eval_set["eval_cases"]
        for turn in case["conversation"]
        for rubric in turn.get("rubrics", [])
        if rubric.get("type") not in scored_types
    ]

    assert unscored == []


@pytest.mark.parametrize("eval_set_name", EVAL_SET_NAMES)
def test_every_seeded_plan_is_a_plan(eval_set_name: str) -> None:
    # A case that seeds `planner:plan` has to seed what the agents read: the
    # `Plan` schema is `extra="forbid"`, so a stale field fails here rather
    # than as a tool error mid-run, and a plan the schema refuses is caught
    # while someone is still looking at it.
    eval_set = _load(eval_set_name)

    for case in eval_set["eval_cases"]:
        seeded = case.get("session_input", {}).get("state", {}).get(PLAN_STATE_KEY)
        if seeded is not None:
            Plan.model_validate_json(seeded)


@pytest.mark.parametrize("eval_set_name", EVAL_SET_NAMES)
def test_every_eval_agent_loads_the_way_adk_eval_loads_it(eval_set_name: str) -> None:
    # `adk eval` imports an agent folder's `__init__.py` by file path
    # (`google.adk.cli.cli_eval._import_from_path`), with nothing but the
    # working directory on `sys.path`, so a folder that imports a sibling has
    # to put the agents directory there itself. `adk web` does that for it,
    # which is how an import that only `adk eval` fails on stays unnoticed
    # until a paid run. Loading each package the way `adk eval` does catches it
    # here, with no model in the loop.
    module_name = f"eval_agent_{eval_set_name.replace('-', '_')}"
    spec = importlib.util.spec_from_file_location(
        module_name, _agent_dir(eval_set_name) / "__init__.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    # A package may put its folder on `sys.path` while it loads; leave the test
    # process as it was.
    saved_path = sys.path[:]
    try:
        spec.loader.exec_module(module)
        assert isinstance(module.agent.root_agent, BaseAgent)
    finally:
        sys.modules.pop(module_name, None)
        sys.path[:] = saved_path


# Metric names ADK ships itself, as opposed to a name a set's own
# `custom_metrics` has to declare a `code_config` for
# (`google.adk.evaluation.eval_config.get_eval_metrics_from_config`
# resolves `custom_function_path` only for a name it finds there).
_BUILT_IN_METRICS = frozenset(
    {
        "tool_trajectory_avg_score",
        "response_match_score",
        "rubric_based_tool_use_quality_v1",
        "rubric_based_final_response_quality_v1",
        "final_response_match_v1",
        "final_response_match_v2",
        "hallucinations_v1",
        "safety_v1",
    }
)


@pytest.mark.parametrize("eval_set_name", EVAL_SET_NAMES)
def test_a_set_configuring_the_trajectory_metric_has_at_least_one_turn_with_tool_uses(
    eval_set_name: str,
) -> None:
    # `tool_trajectory_avg_score` passes any turn with no expected tool calls
    # (see `test_every_eval_set_turn_is_actually_scored`'s own comment), so a
    # set that turns the metric on but states no `tool_uses` expectation
    # anywhere is paying for a metric that can never fail.
    config = _config(eval_set_name)
    if "tool_trajectory_avg_score" not in config.get("criteria", {}):
        pytest.skip(f"{eval_set_name} does not configure tool_trajectory_avg_score")

    eval_set = _load(eval_set_name)
    has_expectation = any(
        turn.get("intermediate_data", {}).get("tool_uses")
        for case in eval_set["eval_cases"]
        for turn in case["conversation"]
    )

    assert has_expectation, (
        f"{eval_set_name} configures tool_trajectory_avg_score but no turn states a "
        "tool_uses expectation"
    )


@pytest.mark.parametrize("eval_set_name", EVAL_SET_NAMES)
def test_a_set_configuring_workflow_outcome_states_an_expectation_on_every_turn(
    eval_set_name: str,
) -> None:
    # `workflow_outcome` (evals/agents/harness_translator_v1/outcome_metric.py)
    # scores a turn `None` (not evaluated) when the turn's own expected
    # invocation carries no `tool_responses` entry naming a tool in
    # `SCORED_TOOLS` -- silent, not a failure -- so a set that turns the
    # metric on has to state such an expectation on every turn or some of
    # them are never actually checked. A non-empty `tool_responses` is not
    # enough: one naming a tool the metric does not score is the same silence.
    config = _config(eval_set_name)
    if "workflow_outcome" not in config.get("criteria", {}):
        pytest.skip(f"{eval_set_name} does not configure workflow_outcome")

    eval_set = _load(eval_set_name)
    unscored = [
        f"{case['eval_id']}:{turn['invocation_id']}"
        for case in eval_set["eval_cases"]
        for turn in case["conversation"]
        if not any(
            response.get("name") in SCORED_TOOLS
            for response in turn.get("intermediate_data", {}).get("tool_responses", [])
        )
    ]

    assert unscored == []


@pytest.mark.parametrize("eval_set_name", EVAL_SET_NAMES)
def test_a_custom_metric_named_in_criteria_is_declared_under_custom_metrics(
    eval_set_name: str,
) -> None:
    # `get_eval_metrics_from_config` (google.adk.evaluation.eval_config) reads
    # a custom metric's implementation from `custom_metrics[<name>].code_config
    # .name`; a metric named in `criteria` that is not one ADK ships itself
    # and has no matching `custom_metrics` entry loads no function at all, and
    # `_CustomMetricEvaluator.__init__` fails at run time with nothing here to
    # catch it first.
    config = _config(eval_set_name)
    custom_metrics = config.get("custom_metrics") or {}
    undeclared = sorted(
        name
        for name in config.get("criteria", {})
        if name not in _BUILT_IN_METRICS and name not in custom_metrics
    )

    assert undeclared == []


def test_the_skills_eval_set_is_the_one_its_cases_build() -> None:
    # harness-planner-skills-v1's user messages are the brief the planner seam
    # renders for each case's envelope, so the JSON is built from
    # `harness_planner_skills_v1.cases` rather than written by hand. A case
    # edited on one side only, or a change to the seam's rendering, fails here.
    assert _load("harness-planner-skills-v1") == skills_eval_set(), (
        "rewrite it: PYTHONPATH=evals/agents uv run python -m harness_planner_skills_v1.cases"
    )
