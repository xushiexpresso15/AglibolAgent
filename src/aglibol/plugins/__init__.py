"""Plugin architecture for Aglibol Agent powered by pluggy."""

from aglibol.plugins.hooks import AglibolHookSpec
from aglibol.plugins.manager import PluginManager

__all__ = ["AglibolHookSpec", "PluginManager"]
