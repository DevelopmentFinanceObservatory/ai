# Sessions the ADK planner stored

Each file is a Studio planner session paused on an ask, as the ADK `LlmAgent` planner stored it: `events` holds every session event as `Event.model_dump(mode="json")`, and `state` holds the session state at the pause. The workflow and node in each event's path carry the names of the test workflow that resumes the file (`tests/test_planner_adapter.py::_AdapterHarness`), since ADK resumes a node by its path. Conversations Studio stored while that planner served hold events of this form, and `tests/test_planner_turns.py` resumes each file on the planner to show those conversations still answer.

- `paused_on_a_planner_ask.json`: the planner node paused on the planner's `ask_user_question` call `a1`, whose model response ends with the empty text part live Gemini streams.
- `paused_on_adk_request_input.json`: the planner paused on its own long-running `adk_request_input` call `r1`, as it asked before the planner node paused on its asks. The state holds no pending ask.

The planner that wrote these files is not in the repository, so nothing can regenerate them. Keep them as they are.
