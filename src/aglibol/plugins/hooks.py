"""Pluggy hook specifications for Aglibol Agent plugins."""

from __future__ import annotations

import pluggy

hookspec = pluggy.HookspecMarker("aglibol")
hookimpl = pluggy.HookimplMarker("aglibol")


class AglibolHookSpec:
    """Specification of hooks available for third-party extensions."""

    @hookspec
    def aglibol_register_agents(self, registry: object) -> None:
        """Called during agent initialization to register custom agents."""

    @hookspec
    def aglibol_register_tools(self, registry: object) -> None:
        """Called during tool initialization to register custom tools."""

    @hookspec
    def aglibol_on_step_start(self, step: int, agent: str) -> None:
        """Called before an agent step begins."""

    @hookspec
    def aglibol_on_step_end(self, step: int, agent: str, state: object) -> None:
        """Called after an agent step finishes."""
