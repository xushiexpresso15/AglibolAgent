"""Aglibol Agent: An open-source, Ollama-native multi-agent framework for personal devices."""

__version__ = "0.1.0"
__author__ = "Aglibol Agent Community"

from aglibol.core.hardware import HardwareProfiler
from aglibol.core.optimizer import ResourceOptimizer
from aglibol.core.types import AgentState, HardwareTier, TaskItem, TaskPlan
from aglibol.core.workflow import WorkflowEngine

__all__ = [
    "__version__",
    "HardwareTier",
    "AgentState",
    "TaskPlan",
    "TaskItem",
    "HardwareProfiler",
    "ResourceOptimizer",
    "WorkflowEngine",
]
