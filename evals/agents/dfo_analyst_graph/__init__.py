"""The whole dfo-ai system as an `adk web` app: see `agent.py`.

`adk web` puts the agents folder on `sys.path` before it loads an agent
package, so `agent`'s imports of `harness`, `planner` and the rest resolve
from the workspace's own `.venv` and no path fix-up is needed here, unlike
`harness_pair_v1`.
"""

from . import agent as agent
