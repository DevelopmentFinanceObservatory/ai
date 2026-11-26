"""The model surface Studio's agents run with, chosen by the model.

A surface is the instruction set the orchestrator and the planner read, the
orchestrator's tool descriptions, and whether an explicit prefix cache may be
built for it. Each model Studio runs has the surface it was evaluated with
(`MODEL_SURFACES`), keyed on its exact name, so a model the map does not name
runs on no surface at all.

A process runs one surface (`studio_surface`). `harness.server.create_app`
resolves it from `MODEL_NAME` and `PLANNER_MODEL_NAME` and sets it, and so do
the evals that assemble a studio of their own. Startup refuses a model the map
does not name, and two models on different surfaces, since the texts one
surface words its own way pass between the planner and the orchestrator.
`set_studio_surface` refuses a surface that differs from the one already set
or resolved, so a second app cannot change the first one's texts.

Retiring a model: delete its `MODEL_SURFACES` row and its surface constant,
every branch and constant a search for its name finds (for Flash, `flash`
without regard to case), the tests that use its studio fixture, and its
recorded fixtures. Flash is also the only model with full tool descriptions
and an explicit cache, so retiring it deletes the `"full"` value of
`ToolDescriptions` with the branch on it in `orchestrator.declarations`, and
`harness.prefix_cache` with its wiring. Stored conversations carry the
model's history marker (`orchestrator.run._STORED_INSTRUCTIONS`), so the
marker keeps an entry in `orchestrator.run._LOADED_INSTRUCTIONS` that loads
the current instruction.
"""

from __future__ import annotations

import hashlib
import re
import threading
from dataclasses import dataclass
from typing import Literal

from planner.settings import AgentSettings

from harness.prefix_cache import PrefixCache
from harness.settings import Settings

# The instruction sets the orchestrator and the planner can run with:
# `flash`, the instructions written for `gemini-3.8-flash`, and `hydrogen`,
# the set written for `gemini-4-hydrogen` (`orchestrator.agent.INSTRUCTIONS`,
# `planner.instruction.planner_instruction`). `hydrogen` also words some tool
# descriptions, tool results and referrals its own way.
type PromptVariant = Literal["flash", "hydrogen"]

# The orchestrator's tool and parameter descriptions: `full`, as the tools'
# docstrings and fields state them, or `minimal`, as
# `orchestrator.minimal_surface` states them (`orchestrator.declarations`).
type ToolDescriptions = Literal["full", "minimal"]


@dataclass(frozen=True)
class Surface:
    """One model surface: the instruction set (`variant`), the orchestrator's
    tool descriptions (`tool_descriptions`), and whether the model accepts an
    explicit prefix cache (`explicit_cache`)."""

    variant: PromptVariant
    tool_descriptions: ToolDescriptions
    explicit_cache: bool


FLASH_SURFACE = Surface(variant="flash", tool_descriptions="full", explicit_cache=True)
# Gemini 4 answers a request for an explicit cache with 400 "Explicit caching
# is not supported".
HYDROGEN_SURFACE = Surface(variant="hydrogen", tool_descriptions="minimal", explicit_cache=False)

# Each model's surface, by its name as the developer API is sent it
# (`resolved_model_name`).
MODEL_SURFACES: dict[str, Surface] = {
    "gemini-3.8-flash": FLASH_SURFACE,
    "gemini-4-hydrogen": HYDROGEN_SURFACE,
}


# The names Gemini serves: its own models, Gemma 4 and later, the model
# optimizer, a fine-tuned Vertex AI endpoint and a Vertex AI Gemini resource
# path.
_GEMINI_NAMES = re.compile(
    r"gemini-.*"
    r"|gemma-4.*"
    r"|model-optimizer-.*"
    r"|projects/.+/locations/.+/endpoints/.+"
    r"|projects/.+/locations/.+/publishers/google/models/gemini.+"
)


def gemini_model_name(model_name: str) -> str | None:
    """`model_name` as sent to Gemini when Gemini serves it, else None.

    A name with a `Gemini:` class prefix, in any case, names Gemini whatever
    follows, and is sent without the prefix. A name with any other prefix
    names another provider's model. An unprefixed name is Gemini's when it
    matches one of `_GEMINI_NAMES` whole.
    """
    prefix, colon, named = model_name.partition(":")
    if colon:
        return named if prefix.lower() == "gemini" else None
    return model_name if _GEMINI_NAMES.fullmatch(model_name) else None


class ModelNameRefusedError(RuntimeError):
    """A model setting names no model the Gemini developer API serves."""


class SurfaceRefusedError(RuntimeError):
    """A model setting names a model with no surface, or the orchestrator's and
    the planner's models run on different surfaces."""


def resolved_model_name(name: str, *, setting: str = "MODEL_NAME") -> str:
    """`name` as the Gemini developer API is sent it: any `Gemini:` class
    prefix dropped (`gemini_model_name`), so the
    orchestrator and Edit Card call one model under one name.

    A leading `models/` is dropped, so `models/gemini-3.8-flash` names the
    same model as its bare name. `setting` names the variable `name` came from
    in a refusal.

    Raises:
        ModelNameRefusedError: `name` names no Gemini model, names none after its
            `Gemini:` prefix, or is a Vertex AI resource path, which the
            developer API serves no model under.
    """
    resolved = gemini_model_name(name.removeprefix("models/"))
    if resolved is not None:
        resolved = resolved.removeprefix("models/")
    if not resolved:
        raise ModelNameRefusedError(f"{setting} {name!r} names no Gemini model.")
    if resolved.startswith("projects/"):
        raise ModelNameRefusedError(
            f"{setting} {name!r} is a Vertex AI resource path, and the harness runs on "
            "the Gemini developer API, which serves no model by that name. Name the "
            "model as the developer API does: models/... or its bare name, with an "
            "optional Gemini: prefix."
        )
    return resolved


def _surface_of(name: str, *, setting: str) -> Surface:
    resolved = resolved_model_name(name, setting=setting)
    surface = MODEL_SURFACES.get(resolved)
    if surface is None:
        raise SurfaceRefusedError(
            f"{setting} {name!r} names a model Studio has no surface for. "
            f"The models it runs: {', '.join(sorted(MODEL_SURFACES))}."
        )
    return surface


def resolve_surface(*, model_name: str, planner_model_name: str) -> Surface:
    """The surface the orchestrator's `model_name` and the planner's
    `planner_model_name` run on together.

    Raises:
        ModelNameRefusedError: Either names no model the developer API serves.
        SurfaceRefusedError: Either names a model `MODEL_SURFACES` lacks, or the
            two map to different surfaces.
    """
    surface = _surface_of(model_name, setting="MODEL_NAME")
    planner_surface = _surface_of(planner_model_name, setting="PLANNER_MODEL_NAME")
    if planner_surface != surface:
        raise SurfaceRefusedError(
            f"MODEL_NAME {model_name!r} and PLANNER_MODEL_NAME {planner_model_name!r} run "
            "on different surfaces, and no such pair was evaluated. Set both to one model's "
            "surface."
        )
    return surface


_surface: Surface | None = None
_surface_lock = threading.Lock()


def studio_surface() -> Surface:
    """The process's surface: the one set with `set_studio_surface`, else
    resolved on first read from the harness's `MODEL_NAME` and the planner's
    `PLANNER_MODEL_NAME` (`resolve_surface`)."""
    global _surface  # noqa: PLW0603 -- the one process-wide surface
    with _surface_lock:
        if _surface is None:
            _surface = resolve_surface(
                model_name=Settings().model_name,
                planner_model_name=AgentSettings().planner_model_name,
            )
        return _surface


def set_studio_surface(surface: Surface | None) -> None:
    """Make `surface` the process's surface. With None, the next
    `studio_surface()` resolves it again from settings.

    A process runs one surface: setting the surface already set, or already
    resolved by `studio_surface()`, changes nothing.

    Raises:
        SurfaceRefusedError: A different surface is already set or resolved.
            Clear it with None first.
    """
    global _surface  # noqa: PLW0603 -- the one process-wide surface
    with _surface_lock:
        if surface is not None and _surface is not None and _surface != surface:
            raise SurfaceRefusedError(
                f"The process already runs the {_surface.variant!r} surface, so the "
                f"{surface.variant!r} surface cannot be set. A process runs one surface; "
                "set_studio_surface(None) clears it."
            )
        _surface = surface


def surface_prefix_cache(surface: Surface, *, enabled: bool) -> PrefixCache | None:
    """A new explicit prefix cache when `enabled` (`STUDIO_PREFIX_CACHE`) and
    `surface`'s model accepts one, else None."""
    return PrefixCache() if enabled and surface.explicit_cache else None


def surface_fingerprint(
    surface: Surface, *, instruction: str | None, declarations: str
) -> dict[str, str]:
    """The span attributes naming the model surface one model call was sent:
    the surface's instruction set and tool descriptions, and the first 12 hex
    digits of the sha256 of the system `instruction` (empty when none) and of
    the `declarations` as JSON."""
    return {
        "studio.surface.prompt_variant": surface.variant,
        "studio.surface.tool_desc": surface.tool_descriptions,
        "studio.surface.instruction_hash": _short_sha(instruction or ""),
        "studio.surface.declarations_hash": _short_sha(declarations),
    }


def _short_sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


__all__ = [
    "FLASH_SURFACE",
    "HYDROGEN_SURFACE",
    "MODEL_SURFACES",
    "ModelNameRefusedError",
    "PromptVariant",
    "Surface",
    "SurfaceRefusedError",
    "ToolDescriptions",
    "gemini_model_name",
    "resolve_surface",
    "resolved_model_name",
    "set_studio_surface",
    "studio_surface",
    "surface_fingerprint",
    "surface_prefix_cache",
]
