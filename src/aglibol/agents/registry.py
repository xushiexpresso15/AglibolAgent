"""Agent registry for dynamic discovery and registration of agent types."""

from __future__ import annotations

from importlib.metadata import entry_points

from aglibol.agents.base import BaseAgent
from aglibol.agents.coder import CoderAgent
from aglibol.agents.planner import PlannerAgent
from aglibol.agents.reviewer import ReviewerAgent
from aglibol.agents.writer import WriterAgent


class AgentRegistry:
    """Registry maintaining active agent classes and third-party extensions."""

    def __init__(self) -> None:
        self._agents: dict[str, type[BaseAgent]] = {}
        self._register_defaults()
        self._discover_plugins()

    def _register_defaults(self) -> None:
        """Register built-in core agents."""
        self.register("planner", PlannerAgent)
        self.register("coder", CoderAgent)
        self.register("reviewer", ReviewerAgent)
        self.register("writer", WriterAgent)

    def register(self, name: str, agent_cls: type[BaseAgent]) -> None:
        """Register an agent class under a role or name."""
        self._agents[name] = agent_cls

    def get(self, name: str) -> type[BaseAgent] | None:
        """Lookup an agent class by name."""
        return self._agents.get(name)

    def list_agents(self) -> list[str]:
        """List all available agent names."""
        return list(self._agents.keys())

    def _discover_plugins(self) -> None:
        """Discover external agent plugins registered under entry-point 'aglibol.agents'."""
        try:
            eps = entry_points(group="aglibol.agents")
            for ep in eps:
                try:
                    cls = ep.load()
                    if issubclass(cls, BaseAgent):
                        self._agents[ep.name] = cls
                except Exception:
                    continue
        except Exception:
            pass
