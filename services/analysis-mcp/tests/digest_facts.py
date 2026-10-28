"""The digest renderers over an engine result, as the in-process tools call them:
each turns its `result` into the facts a runner summary states
(`analysis_mcp.engine_facts.facts_of`) before rendering, so a test built from an
engine result checks the text the runner's facts render."""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

from analysis_mcp import digest
from analysis_mcp.engine_facts import facts_of


def _over_facts(render: Callable[..., str]) -> Callable[..., str]:
    @functools.wraps(render)
    def rendered(*, result: Any, **arguments: Any) -> str:
        return render(result=facts_of(result), **arguments)

    return rendered


render_aggregate_digest = _over_facts(digest.render_aggregate_digest)
render_compare_digest = _over_facts(digest.render_compare_digest)
render_filter_digest = _over_facts(digest.render_filter_digest)
render_growth_digest = _over_facts(digest.render_growth_digest)
render_join_digest = _over_facts(digest.render_join_digest)
render_quotient_of_digest = _over_facts(digest.render_quotient_of_digest)
render_rank_digest = _over_facts(digest.render_rank_digest)
render_share_of_digest = _over_facts(digest.render_share_of_digest)
render_sum_of_digest = _over_facts(digest.render_sum_of_digest)
