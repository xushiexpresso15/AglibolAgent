"""Plugin manager for discovering and invoking plugin hooks."""

from __future__ import annotations

import pluggy

from aglibol.plugins.hooks import AglibolHookSpec


class PluginManager:
    """Manages plugin discovery and lifecycle via pluggy."""

    def __init__(self) -> None:
        self.pm = pluggy.PluginManager("aglibol")
        self.pm.add_hookspecs(AglibolHookSpec)
        # Automatically load entry points
        self.pm.load_setuptools_entrypoints("aglibol")

    def register(self, plugin: object, name: str | None = None) -> None:
        """Register a plugin instance or module."""
        self.pm.register(plugin, name=name)

    @property
    def hook(self) -> pluggy.HookRelay:
        """Access the hook caller."""
        return self.pm.hook
