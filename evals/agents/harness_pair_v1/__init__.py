import sys
from pathlib import Path

# `adk web` puts the agents folder on `sys.path` before it loads an agent package;
# `adk eval` loads the package by file path and puts nothing there. `agent` imports
# `harness_translator_v1`'s canned discovery, so do here what `adk web` does.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from . import agent as agent
