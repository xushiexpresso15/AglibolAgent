"""Core data types, Enums, and Pydantic models for Aglibol Agent."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class HardwareTier(str, Enum):
    """Classification tiers for personal computing devices based on GPU VRAM and compute capacity."""

    TIER_0_CPU = "tier0_cpu"  # Pure CPU / Integrated Graphics (0 GB dedicated VRAM)
    TIER_1_4GB = "tier1_4gb"  # 4 GB VRAM (RTX 3050 Laptop, GTX 1650)
    TIER_2_6GB = "tier2_6gb"  # 6 GB VRAM (RTX 2060, GTX 1660 Ti)
    TIER_3_8GB = "tier3_8gb"  # 8 GB VRAM (RTX 4060, RTX 3060 Ti, RX 7600)
    TIER_4_12GB = "tier4_12gb"  # 10-14 GB VRAM (RTX 3060 12GB, RTX 3080 10GB, RTX 4070)
    TIER_5_16GB = "tier5_16gb"  # 16-20 GB VRAM (RTX 4080, RX 7800 XT, RX 7900 XT)
    TIER_6_24GB = "tier6_24gb"  # 24-32 GB VRAM (RTX 3090/4090, RTX 5090 32GB)
    TIER_7_ENTERPRISE = (
        "tier7_enterprise"  # 40GB+ VRAM (Dual GPU, A100, H100, Mac 64GB-128GB Unified)
    )

    @classmethod
    def from_vram_gb(cls, vram_gb: float, has_discrete_gpu: bool = True) -> HardwareTier:
        """Derive hardware tier smoothly from detected VRAM in gigabytes."""
        if not has_discrete_gpu or vram_gb < 1.0:
            return cls.TIER_0_CPU
        elif vram_gb < 5.0:
            return cls.TIER_1_4GB
        elif vram_gb < 7.0:
            return cls.TIER_2_6GB
        elif vram_gb < 9.5:
            return cls.TIER_3_8GB
        elif vram_gb < 14.5:
            return cls.TIER_4_12GB
        elif vram_gb < 22.0:
            return cls.TIER_5_16GB
        elif vram_gb < 36.0:
            return cls.TIER_6_24GB
        else:
            return cls.TIER_7_ENTERPRISE


class AgentMode(str, Enum):
    """Operating modes for Aglibol Agent interaction."""

    AUTO = "auto"  # Autonomous dynamic intent routing
    CHAT = "chat"  # Conversational assistant (fast, low-overhead direct responses)
    PLANNER = "planner"  # Architecture & task decomposition with user approval gate
    CODER = "coder"  # Code generation, tool execution, and file persistence
    REVIEWER = "reviewer"  # Code audit and syntax/quality inspection
    WRITER = "writer"  # Technical documentation, articles, and copywriting


class GPUInfo(BaseModel):
    """Detected GPU hardware information."""

    name: str = "Unknown GPU"
    vendor: str = "Unknown"  # NVIDIA, AMD, Intel, Apple
    vram_total_mb: float = 0.0
    vram_free_mb: float = 0.0
    vram_used_mb: float = 0.0
    temperature_c: float | None = None
    utilization_percent: float | None = None

    @property
    def vram_total_gb(self) -> float:
        return round(self.vram_total_mb / 1024, 2)

    @property
    def vram_free_gb(self) -> float:
        return round(self.vram_free_mb / 1024, 2)


class DiskInfo(BaseModel):
    """Storage partition metrics."""

    mountpoint: str
    total_gb: float
    free_gb: float
    used_percent: float


class HardwareProfile(BaseModel):
    """Complete hardware fingerprint of the host machine."""

    os_name: str
    os_version: str
    cpu_name: str
    cpu_cores_physical: int
    cpu_cores_logical: int
    ram_total_gb: float
    ram_available_gb: float
    ram_used_percent: float
    gpus: list[GPUInfo] = Field(default_factory=list)
    primary_gpu: GPUInfo | None = None
    disks: list[DiskInfo] = Field(default_factory=list)
    tier: HardwareTier = HardwareTier.TIER_1_4GB

    # Dynamic multi-GPU and memory architecture properties
    has_discrete_gpu: bool = True
    is_unified_memory: bool = False  # e.g., Apple Silicon or AMD APU
    total_vram_gb: float = 0.0
    total_vram_free_gb: float = 0.0
    gpu_count: int = 0


class OptimizedParams(BaseModel):
    """Calculated inference parameters dynamically tuned for current hardware and task."""

    recommended_model: str
    num_ctx: int = 2048
    num_gpu: int = -1  # -1 = all layers, 0 = pure CPU, >0 = exact layer count
    keep_alive: str = "0"  # "0" (unload immediately), "2m", "5m", "-1"
    flash_attention: bool = True
    max_output_tokens: int = 2048
    concurrency_allowed: bool = False  # High VRAM systems can keep multiple models resident
    max_loaded_models: int = 1
    layer_offload_ratio: float = 1.0  # 1.0 = 100% on GPU, 0.5 = 50% GPU / 50% CPU, 0.0 = CPU only
    target_backend: str = "cuda"  # "cuda", "rocm", "metal", "cpu"
    kv_cache_type: str = "q8_0"  # "q8_0", "f16"
    reasoning: str = ""


class TaskStatus(str, Enum):
    """Status of a task item inside the execution plan."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"


class TaskItem(BaseModel):
    """A granular decomposed step in an execution plan."""

    id: str
    title: str
    description: str
    assigned_agent: str  # e.g., "planner", "coder", "reviewer"
    dependencies: list[str] = Field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    result: str | None = None
    error: str | None = None


class TaskPlan(BaseModel):
    """Plan generated by the PlannerAgent decomposing user goal into steps."""

    plan_id: str
    goal: str
    summary: str = ""
    tasks: list[TaskItem] = Field(default_factory=list)


class ToolCall(BaseModel):
    """Representation of an agent requesting a tool execution."""

    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    call_id: str | None = None


class ToolResult(BaseModel):
    """Outcome of a tool execution."""

    tool_name: str
    success: bool
    output: str
    error: str | None = None


class ChatMessage(BaseModel):
    """Message in the agent conversation turn."""

    role: str  # "system", "user", "assistant", "tool"
    content: str
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_name: str | None = None  # populated when role=="tool"


class AgentResponse(BaseModel):
    """Unified response from an Agent execution."""

    content: str
    thinking: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    history: list[ChatMessage] = Field(default_factory=list)
    is_done: bool = True
    tokens_used: int = 0
    raw_response: dict[str, Any] = Field(default_factory=dict)


class AgentState(BaseModel):
    """Shared state passed between nodes in the WorkflowEngine."""

    session_id: str
    user_goal: str
    status: str = "running"
    workspace_dir: str = ""
    edge_cycles: dict[str, int] = Field(default_factory=dict)
    current_step: int = 0
    active_agent: str = "planner"
    task_plan: TaskPlan | None = None
    current_task_id: str | None = None
    messages: list[ChatMessage] = Field(default_factory=list)
    artifacts: dict[str, str] = Field(default_factory=dict)  # filename -> content/diff
    scratchpad: dict[str, Any] = Field(default_factory=dict)  # intermediate notes
    review_status: str = "pending"  # "pending", "approved", "rejected"
    review_feedback: str = ""
    assistant_reply: str = ""  # Direct text response to user
    iteration_count: int = 0  # For self-correction loops
    is_completed: bool = False
    error: str | None = None
