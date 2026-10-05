from orchestrator.agent import AGENT_NAME, INSTRUCTION, create_orchestrator_agent


def test_the_orchestrator_runs_on_the_model_it_is_given_with_retries():
    agent = create_orchestrator_agent(model_name="gemini-test")

    assert agent.name == AGENT_NAME
    assert agent.model.model == "gemini-test"
    assert agent.model.retry_options.attempts == 3
    assert agent.include_contents == "default"


def test_the_instruction_reads_the_canvas_and_the_conversation_from_state():
    assert "{canvas_digest?}" in INSTRUCTION
    assert "{conversation_digest?}" in INSTRUCTION


def test_the_instruction_lets_a_reply_about_dfo_state_the_figures_explain_dfo_gives():
    assert "stating only the figures it gives" in INSTRUCTION
    assert "Never state a\nfigure in a reply to a greeting or small talk" in INSTRUCTION
