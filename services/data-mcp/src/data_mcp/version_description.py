"""The description dfo-apis serves for one dataset version, read for the `get_data` digest.

dfo-apis composes a version's description (`GET /api/datasets/{id}/{content_version}
/description`) from the record it stores at mint, and the cards the presentation
service draws carry the same description. `get_data` reads it once after each mint,
so the digest's citation, latest period, caveats and steps are the words the card
shows, and a summed frame's family is named by its title. The body follows dfo-spec-kit's
`description.schema.json`; `VersionDescription` keeps only the members `get_data` reads.

`fetch_version_description` is the public seam. Any failure, a refusal (404 for a
version without a record, 410 for a revoked one), a 503, a timeout or a body that
does not parse, logs one warning and returns `None`, and the digest renders without
a description, as it does without statistics.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, ConfigDict, ValidationError

from data_mcp.client_holder import get_client
from data_mcp.deadline import CallDeadline
from data_mcp.errors import UpstreamError

logger = logging.getLogger(__name__)

DESCRIPTION_ROUTE = "/api/datasets/{id}/{content_version}/description"

_UNSEEN_FIELDS_ARE_DROPPED = ConfigDict(extra="ignore")


class DescribedFreshness(BaseModel):
    model_config = _UNSEEN_FIELDS_ARE_DROPPED
    latest_period: str | None = None


class DescribedSentence(BaseModel):
    """A caveat or step as served: the composer's sentence for it."""

    model_config = _UNSEEN_FIELDS_ARE_DROPPED
    text: str


class DescribedCitation(BaseModel):
    """The composer's citation line, "<sources> via ONE DFO, retrieved <date>"."""

    model_config = _UNSEEN_FIELDS_ARE_DROPPED
    text: str


class VersionDescription(BaseModel):
    """The members of a served description `get_data` reads.

    `steps` is `None` when the record holds no steps. `gaps` lists the dotted paths
    of the members dfo-apis could not fill, `title` among them when today's words
    could not compose it.
    """

    model_config = _UNSEEN_FIELDS_ARE_DROPPED
    title: str
    freshness: DescribedFreshness
    steps: list[DescribedSentence] | None
    caveats: list[DescribedSentence]
    cite: DescribedCitation
    gaps: list[str]


async def fetch_version_description(
    *, spec_id: str, content_version: str, deadline: CallDeadline
) -> VersionDescription | None:
    """The description of dfo-apis' version `{spec_id, content_version}`, or `None`.

    The pair is the manifest's own, which names dfo-apis' version. A frame minted
    with `options.columns` or `options.groupby` is held under a frame reference of
    its own, which dfo-apis serves no description for. The request carries no
    `If-None-Match`, so dfo-apis answers 200 or a refusal, never 304.
    """
    path = f"/api/datasets/{spec_id}/{content_version}/description"
    try:
        response = await get_client().request(
            "GET", path, route=DESCRIPTION_ROUTE, deadline=deadline
        )
        return VersionDescription.model_validate_json(response.content)
    except UpstreamError as exc:
        reason = (
            f"status={exc.status} code={exc.code}"
            if exc.status is not None
            else f"no response ({exc.detail})"
        )
        logger.warning("get_data: no description for %s/%s: %s", spec_id, content_version, reason)
    except ValidationError as exc:
        logger.warning(
            "get_data: the description of %s/%s did not parse: %s",
            spec_id,
            content_version,
            exc,
        )
    return None
