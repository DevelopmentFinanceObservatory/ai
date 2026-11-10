"""`node/`'s pass lines run again with the hardened planner (`hardened.hard_planner`) in
place of `node/`'s, so the fixes of this directory change none of them. Only the tests that
compare the two arms are run; `node/`'s controls break `PaiPlannerSeam` itself."""

from __future__ import annotations

import hardened
import pytest
import studio_arms
from test_node import (  # noqa: F401 -- collected here
    test_a_pick_replans_the_plan_as_dev_does,
    test_a_refusal_replan_from_execute_drafts_the_next_version_as_dev_does,
    test_a_revision_after_a_run_that_asked_reads_what_it_reads_on_dev,
    test_an_amendment_drafts_version_two_as_dev_does,
    test_an_ask_during_an_amendment_pauses_and_its_answer_drafts_version_two,
    test_an_ask_pauses_through_devs_interrupt_and_the_answer_resumes_the_plan,
    test_progress_frames_reach_the_client_while_the_planner_runs,
    test_stop_mid_planning_ends_the_turn_as_dev_does_and_leaves_no_task,
    test_the_answer_of_an_ask_is_taken_once,
    test_the_frames_of_a_planned_turn_equal_devs,
    test_the_records_a_turn_leaves_equal_devs,
)


@pytest.fixture(autouse=True)
def hardened_planner(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(studio_arms, "pai_planner", hardened.hard_planner)
