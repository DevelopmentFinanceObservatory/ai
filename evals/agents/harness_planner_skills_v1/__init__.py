import sys
from pathlib import Path

# `adk web` puts the agents folder on `sys.path` before it loads an agent package;
# `adk eval` loads the package by file path and puts nothing there. The
# `skills_loaded_before_plan` and `workflow_outcome` metrics this set's
# `test_config.json` turns on are imported by dotted path, which needs the
# agents folder (`evals/agents`) on `sys.path`. Do here what `adk web` does.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from . import agent as agent
