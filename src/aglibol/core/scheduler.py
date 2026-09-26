"""Model scheduler with adaptive residency: sequential on low VRAM, concurrent on high VRAM."""

from __future__ import annotations

import asyncio
from typing import Any

from aglibol.core.events import AgentEvent, EventBus
from aglibol.core.optimizer import ResourceOptimizer
from aglibol.core.types import HardwareProfile, OptimizedParams
from aglibol.ollama.client import OllamaClient


class ModelScheduler:
    """Orchestrates model residency in VRAM, dynamically adapting residency to hardware capacity."""

    def __init__(
        self,
        client: OllamaClient,
        profile: HardwareProfile,
        event_bus: EventBus | None = None,
    ) -> None:
        self.client = client
        self.profile = profile
        self.event_bus = event_bus
        self.current_loaded_model: str | None = None
        self._lock = asyncio.Lock()

    async def acquire_model(
        self,
        model_name: str,
        role: str = "coder",
        custom_options: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any], str]:
        """
        Prepare and acquire target model.
        Returns (model_name, options_dict, keep_alive_str).
        Adapts eviction behavior based on host VRAM capacity (concurrency vs sequential).
        """
        async with self._lock:
            # 1. Derive optimal execution parameters mathematically
            opt: OptimizedParams = ResourceOptimizer.get_params_for_profile(
                profile=self.profile,
                role=role,
                requested_model=model_name,
            )

            # 2. Check currently resident models via Ollama API
            loaded_models = await self.client.ps()
            loaded_names = [m.model for m in loaded_models]

            # If target model is already loaded, reuse it instantly
            target_already_loaded = model_name in loaded_names

            if not target_already_loaded:
                # Eviction policy
                if not opt.concurrency_allowed:
                    # Constrained mode: unload all resident models to guarantee zero OOM
                    for resident in loaded_names:
                        if self.event_bus:
                            await self.event_bus.emit(
                                AgentEvent(
                                    event_type="MODEL_UNLOADED",
                                    data={"model": resident, "reason": "sequential_swap"},
                                )
                            )
                        await self.client.unload_model(resident)
                        await asyncio.sleep(0.3)
                else:
                    # High-capacity mode: only evict if loaded models reach capacity
                    if len(loaded_models) >= opt.max_loaded_models:
                        # Evict the one expiring soonest
                        sorted_models = sorted(loaded_models, key=lambda m: m.expires_at)
                        victim = sorted_models[0].model
                        if self.event_bus:
                            await self.event_bus.emit(
                                AgentEvent(
                                    event_type="MODEL_UNLOADED",
                                    data={"model": victim, "reason": "lru_capacity_evict"},
                                )
                            )
                        await self.client.unload_model(victim)
                        await asyncio.sleep(0.3)

            # 3. Assemble options
            options: dict[str, Any] = {
                "num_ctx": opt.num_ctx,
                "num_predict": max(2048, getattr(opt, "max_output_tokens", 2048)),
            }
            if opt.num_gpu != -1:
                options["num_gpu"] = opt.num_gpu

            if custom_options:
                options.update(custom_options)

            keep_alive = opt.keep_alive

            if self.event_bus and not target_already_loaded:
                await self.event_bus.emit(
                    AgentEvent(
                        event_type="MODEL_LOAD_STARTED",
                        data={
                            "model": model_name,
                            "num_ctx": options["num_ctx"],
                            "keep_alive": keep_alive,
                            "num_gpu": opt.num_gpu,
                            "backend": opt.target_backend,
                        },
                    )
                )

            self.current_loaded_model = model_name
            return model_name, options, keep_alive

    async def release_current_model(self) -> None:
        """Explicitly unload the currently tracked model to free all VRAM."""
        async with self._lock:
            if self.current_loaded_model:
                await self.client.unload_model(self.current_loaded_model)
                if self.event_bus:
                    await self.event_bus.emit(
                        AgentEvent(
                            event_type="MODEL_UNLOADED",
                            data={"model": self.current_loaded_model, "reason": "explicit_release"},
                        )
                    )
                self.current_loaded_model = None

    async def preload_hint(self, model_name: str, role: str = "coder") -> bool:
        """
        Speculatively warm-up / preload a model in the background if hardware supports concurrency.
        On constrained systems (low VRAM), speculative preloading is a strict no-op to guarantee zero OOM.
        Returns True if preload was triggered or model is already resident, False otherwise.
        """
        opt: OptimizedParams = ResourceOptimizer.get_params_for_profile(
            profile=self.profile,
            role=role,
            requested_model=model_name,
        )
        if not opt.concurrency_allowed:
            # Constrained system: strictly forbid concurrent preloading
            return False

        async with self._lock:
            try:
                loaded_models = await self.client.ps()
                loaded_names = [m.model for m in loaded_models]
                if model_name in loaded_names:
                    return True

                if len(loaded_models) < opt.max_loaded_models:
                    # Warm-up model into VRAM using keep_alive
                    await self.client.preload_model(model_name)
                    if self.event_bus:
                        await self.event_bus.emit(
                            AgentEvent(
                                event_type="MODEL_PRELOAD_WARMED",
                                data={"model": model_name, "backend": opt.target_backend},
                            )
                        )
                    return True
            except Exception:
                return False

        return False
