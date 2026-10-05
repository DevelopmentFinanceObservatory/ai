"""Shared by the eval sets and their tests: the canned discovery (`discovery_stub`),
the `workflow_outcome` metric (`outcome_metric`), and the translation cases
`services/harness/tests/test_eval_outcomes.py` drives through the translator's core.

No agent lives here: translation runs no model, so there is nothing for
`adk eval` to score on its own.
"""
