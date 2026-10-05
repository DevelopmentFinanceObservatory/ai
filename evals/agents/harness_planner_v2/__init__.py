import sys
from pathlib import Path

# `adk web` puts the agents folder on `sys.path` before it loads an agent package;
# `adk eval` loads the package by file path and puts nothing there. The
# `workflow_outcome` custom metric this set's `test_config.json` turns on is
# imported later by dotted path
# ("harness_translator_v1.outcome_metric.workflow_outcome"), which needs the
# agents folder (`evals/agents`) on `sys.path` to resolve as a top-level
# package. Do here what `adk web` does.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from . import agent as agent
