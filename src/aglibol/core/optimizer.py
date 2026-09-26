"""Continuous mathematical resource optimizer adapting dynamically to any hardware configuration."""

from __future__ import annotations

import math

from aglibol.core.types import (
    HardwareProfile,
    HardwareTier,
    OptimizedParams,
)


class ResourceOptimizer:
    """Calculates optimal inference parameters dynamically using mathematical hardware formulas."""

    # Backward compatibility mapping for static lookup if needed
    TIER_DEFAULTS = {
        HardwareTier.TIER_0_CPU: {
            "num_ctx": 2048,
            "keep_alive": "0",
            "max_output_tokens": 2048,
            "recommended_models": {
                "planner": "qwen2.5:7b",
                "coder": "qwen2.5-coder:7b",
                "reviewer": "qwen2.5:7b",
                "researcher": "qwen2.5:7b",
                "doc_writer": "phi3:mini",
                "vision": "moondream2",
            },
        },
        HardwareTier.TIER_1_4GB: {
            "num_ctx": 2048,
            "keep_alive": "0",
            "max_output_tokens": 2048,
            "recommended_models": {
                "planner": "qwen2.5:7b",
                "coder": "qwen2.5-coder:7b",
                "reviewer": "qwen2.5:7b",
                "researcher": "qwen2.5:7b",
                "doc_writer": "phi3:mini",
                "vision": "moondream2",
            },
        },
        HardwareTier.TIER_2_6GB: {
            "num_ctx": 4096,
            "keep_alive": "0",
            "max_output_tokens": 4096,
            "recommended_models": {
                "planner": "qwen2.5:7b",
                "coder": "qwen2.5-coder:7b",
                "reviewer": "qwen2.5:7b",
                "researcher": "qwen2.5:7b",
                "doc_writer": "phi3:mini",
                "vision": "moondream2",
            },
        },
        HardwareTier.TIER_3_8GB: {
            "num_ctx": 4096,
            "keep_alive": "2m",
            "max_output_tokens": 4096,
            "recommended_models": {
                "planner": "qwen2.5:7b",
                "coder": "qwen2.5-coder:7b",
                "reviewer": "qwen2.5:7b",
                "researcher": "qwen2.5:7b",
                "doc_writer": "phi3:mini",
                "vision": "llava:7b",
            },
        },
        HardwareTier.TIER_4_12GB: {
            "num_ctx": 8192,
            "keep_alive": "5m",
            "max_output_tokens": 4096,
            "recommended_models": {
                "planner": "qwen2.5:14b",
                "coder": "qwen2.5-coder:14b",
                "reviewer": "qwen2.5:14b",
                "researcher": "qwen2.5:14b",
                "doc_writer": "phi3:mini",
                "vision": "llava:13b",
            },
        },
        HardwareTier.TIER_5_16GB: {
            "num_ctx": 16384,
            "keep_alive": "10m",
            "max_output_tokens": 8192,
            "recommended_models": {
                "planner": "qwen2.5:32b",
                "coder": "qwen2.5-coder:32b",
                "reviewer": "qwen2.5:32b",
                "researcher": "qwen2.5:32b",
                "doc_writer": "phi3:mini",
                "vision": "llava:34b",
            },
        },
        HardwareTier.TIER_6_24GB: {
            "num_ctx": 32768,
            "keep_alive": "-1",
            "max_output_tokens": 8192,
            "recommended_models": {
                "planner": "qwen2.5:32b",
                "coder": "qwen2.5-coder:32b",
                "reviewer": "qwen2.5:32b",
                "researcher": "qwen2.5:32b",
                "doc_writer": "phi3:mini",
                "vision": "llava:34b",
            },
        },
        HardwareTier.TIER_7_ENTERPRISE: {
            "num_ctx": 65536,
            "keep_alive": "-1",
            "max_output_tokens": 16384,
            "recommended_models": {
                "planner": "qwen2.5:72b",
                "coder": "qwen2.5-coder:32b",
                "reviewer": "qwen2.5:72b",
                "researcher": "qwen2.5:72b",
                "doc_writer": "phi3:mini",
                "vision": "llava:34b",
            },
        },
    }

    # --- Mathematical Estimation Formulas ---

    @staticmethod
    def estimate_model_parameters(model_name: str) -> float:
        """Heuristically extract parameter count in billions from model name."""
        name_lower = model_name.lower()
        if "72b" in name_lower or "70b" in name_lower:
            return 72.0
        elif "34b" in name_lower or "32b" in name_lower:
            return 32.0
        elif "14b" in name_lower or "13b" in name_lower:
            return 14.0
        elif "8b" in name_lower:
            return 8.0
        elif "7b" in name_lower:
            return 7.0
        elif "3b" in name_lower or "mini" in name_lower:
            return 3.5
        elif "1.5b" in name_lower or "1b" in name_lower:
            return 1.5
        return 7.0

    @classmethod
    def estimate_weight_size_gb(cls, param_b: float, quant_bits: float = 4.5) -> float:
        """
        Estimate model file/VRAM weight size in Gigabytes.
        Formula: (P * B_quant / 8) * 1.15 (overhead factor for GGUF metadata & tensors).
        """
        return round((param_b * quant_bits / 8.0) * 1.15, 2)

    @classmethod
    def estimate_kv_cache_gb(
        cls,
        context_tokens: int,
        param_b: float = 7.0,
        kv_precision_bytes: float = 1.0,
    ) -> float:
        """
        Estimate KV Cache memory in Gigabytes.
        Accounts for Grouped Query Attention (GQA) used in modern Llama 3 / Qwen 2.5.
        """
        if param_b <= 4:
            layers = 24
            kv_heads = 4
        elif param_b <= 9:
            layers = 32
            kv_heads = 8
        elif param_b <= 16:
            layers = 40
            kv_heads = 8
        elif param_b <= 36:
            layers = 64
            kv_heads = 8
        else:
            layers = 80
            kv_heads = 8

        head_dim = 128
        # KV Cache = 2 * layers * kv_heads * head_dim * tokens * precision_bytes
        total_bytes = 2 * layers * kv_heads * head_dim * context_tokens * kv_precision_bytes
        return round(total_bytes / (1024**3), 3)

    @classmethod
    def calculate_optimal_context(
        cls,
        vram_free_gb: float,
        ram_available_gb: float,
        param_b: float = 7.0,
        is_cpu_only: bool = False,
    ) -> int:
        """
        Calculate maximum safe context window tokens dynamically based on available memory.
        Ensures inference never OOMs while maximizing reasoning window.
        """
        if is_cpu_only:
            # CPU inference: memory budget comes from system RAM
            usable_ram = ram_available_gb * 0.65
            weight_size = cls.estimate_weight_size_gb(param_b)
            remaining_for_kv = max(0.0, usable_ram - weight_size)
            if remaining_for_kv < 0.5:
                return 2048
            # Calculate tokens that fit
            kv_per_1k = cls.estimate_kv_cache_gb(1024, param_b)
            raw_tokens = int((remaining_for_kv / kv_per_1k) * 1024)
            # Clamp between 2048 and 16384 for CPU
            return cls._snap_to_standard_context(min(max(raw_tokens, 2048), 16384))

        # GPU acceleration
        usable_vram = vram_free_gb * 0.90
        weight_size = cls.estimate_weight_size_gb(param_b)

        if usable_vram >= weight_size:
            remaining_for_kv = usable_vram - weight_size
            kv_per_1k = cls.estimate_kv_cache_gb(1024, param_b)
            raw_tokens = int((remaining_for_kv / kv_per_1k) * 1024) if kv_per_1k > 0 else 2048
            clamped = min(max(raw_tokens, 2048), 65536)
            return cls._snap_to_standard_context(clamped)
        else:
            # Model partially offloads to RAM; keep VRAM KV cache small to leave room for layers
            return 2048

    @staticmethod
    def _snap_to_standard_context(tokens: int) -> int:
        """Snap token count to standard power-of-two or standard multiples."""
        standards = [2048, 4096, 8192, 12288, 16384, 24576, 32768, 49152, 65536]
        # Find highest standard <= tokens
        for s in reversed(standards):
            if tokens >= s:
                return s
        return 2048

    @classmethod
    def calculate_gpu_layers(
        cls,
        vram_free_gb: float,
        param_b: float = 7.0,
        total_layers: int = 32,
    ) -> tuple[int, float]:
        """
        Calculate how many transformer layers can be offloaded to GPU.
        Returns (num_gpu, offload_ratio). -1 indicates 100% full GPU offload.
        """
        if vram_free_gb < 1.0:
            return 0, 0.0

        weight_size = cls.estimate_weight_size_gb(param_b)
        kv_budget = cls.estimate_kv_cache_gb(2048, param_b)
        total_needed = weight_size + kv_budget + 0.3  # 300MB buffer

        if vram_free_gb >= total_needed:
            return -1, 1.0

        # Partial offload calculation
        usable = max(0.0, vram_free_gb - kv_budget - 0.3)
        ratio = min(max(usable / weight_size, 0.0), 1.0)
        layers = int(math.floor(ratio * total_layers))
        return layers, round(ratio, 2)

    @classmethod
    def calculate_residency_policy(cls, vram_free_gb: float) -> tuple[str, bool, int]:
        """
        Determine model residency policy in VRAM dynamically.
        Returns (keep_alive, concurrency_allowed, max_loaded_models).
        """
        if vram_free_gb < 7.5:
            # Constrained: strictly unload previous model to preserve VRAM
            return "0", False, 1
        elif vram_free_gb < 14.5:
            # Moderate: keep warm briefly if subsequent task uses the same model
            return "2m", False, 1
        elif vram_free_gb < 23.0:
            # Performance: allow up to 2 models resident simultaneously
            return "5m", True, 2
        elif vram_free_gb < 38.0:
            # High-end: allow 3 models resident
            return "-1", True, 3
        else:
            # Enterprise (40GB+): keep all active agents resident
            return "-1", True, 5

    @classmethod
    def recommend_models_dynamically(
        cls,
        effective_vram_gb: float,
        ram_available_gb: float,
        has_gpu: bool,
    ) -> dict[str, str]:
        """Recommend model assignments per role continuously based on exact available resources."""
        if not has_gpu or effective_vram_gb < 2.0:
            # CPU or integrated graphics
            if ram_available_gb >= 48.0:
                # Big RAM allows fast CPU inference of 14B models
                return {
                    "planner": "qwen2.5:14b",
                    "coder": "qwen2.5-coder:14b",
                    "reviewer": "qwen2.5:14b",
                    "researcher": "qwen2.5:14b",
                    "doc_writer": "phi3:mini",
                    "vision": "moondream2",
                }
            elif ram_available_gb >= 12.0:
                return {
                    "planner": "qwen2.5:7b",
                    "coder": "qwen2.5-coder:7b",
                    "reviewer": "qwen2.5:7b",
                    "researcher": "qwen2.5:7b",
                    "doc_writer": "phi3:mini",
                    "vision": "moondream2",
                }
            else:
                return {
                    "planner": "phi3:mini",
                    "coder": "qwen2.5-coder:3b",
                    "reviewer": "phi3:mini",
                    "researcher": "phi3:mini",
                    "doc_writer": "phi3:mini",
                    "vision": "moondream2",
                }

        # Dedicated GPU sizing
        if effective_vram_gb < 6.5:
            return {
                "planner": "qwen2.5:7b",
                "coder": "qwen2.5-coder:7b",
                "reviewer": "qwen2.5:7b",
                "researcher": "qwen2.5:7b",
                "doc_writer": "phi3:mini",
                "vision": "moondream2",
            }
        elif effective_vram_gb < 11.5:
            return {
                "planner": "qwen2.5:7b",
                "coder": "qwen2.5-coder:7b",
                "reviewer": "qwen2.5:7b",
                "researcher": "qwen2.5:7b",
                "doc_writer": "phi3:mini",
                "vision": "llava:7b",
            }
        elif effective_vram_gb < 20.0:
            return {
                "planner": "qwen2.5:14b",
                "coder": "qwen2.5-coder:14b",
                "reviewer": "qwen2.5:14b",
                "researcher": "qwen2.5:14b",
                "doc_writer": "phi3:mini",
                "vision": "llava:13b",
            }
        elif effective_vram_gb < 35.0:
            return {
                "planner": "qwen2.5:32b",
                "coder": "qwen2.5-coder:32b",
                "reviewer": "qwen2.5:32b",
                "researcher": "qwen2.5:32b",
                "doc_writer": "phi3:mini",
                "vision": "llava:34b",
            }
        else:
            return {
                "planner": "qwen2.5:72b",
                "coder": "qwen2.5-coder:32b",
                "reviewer": "qwen2.5:72b",
                "researcher": "qwen2.5:72b",
                "doc_writer": "phi3:mini",
                "vision": "llava:34b",
            }

    # --- Core API ---

    @classmethod
    def get_params_for_profile(
        cls,
        profile: HardwareProfile,
        role: str = "coder",
        requested_model: str | None = None,
    ) -> OptimizedParams:
        """
        Calculate dynamically optimized parameters tailored mathematically to this host.
        Works across all specs: pure CPU, 4GB laptop, 10GB/12GB, 20GB/24GB, 80GB enterprise.
        """
        # 1. Total effective graphics capacity
        effective_vram = (
            profile.total_vram_free_gb
            if profile.total_vram_free_gb > 0
            else (profile.primary_gpu.vram_free_gb if profile.primary_gpu else 0.0)
        )
        has_gpu = profile.has_discrete_gpu and effective_vram > 0

        # 2. Dynamic model recommendation
        recommended_models = cls.recommend_models_dynamically(
            effective_vram_gb=effective_vram,
            ram_available_gb=profile.ram_available_gb,
            has_gpu=has_gpu,
        )

        model = requested_model or recommended_models.get(role, "qwen2.5-coder:7b")
        param_b = cls.estimate_model_parameters(model)

        # 3. Dynamic Context Window
        num_ctx = cls.calculate_optimal_context(
            vram_free_gb=effective_vram,
            ram_available_gb=profile.ram_available_gb,
            param_b=param_b,
            is_cpu_only=(not has_gpu),
        )

        # 4. GPU Layer Offloading
        num_gpu, offload_ratio = cls.calculate_gpu_layers(
            vram_free_gb=effective_vram,
            param_b=param_b,
        )

        # 5. Residency Policy
        keep_alive, concurrency, max_loaded = cls.calculate_residency_policy(effective_vram)

        # 6. Target Backend & Precision
        if not has_gpu:
            target_backend = "cpu"
        elif profile.primary_gpu and profile.primary_gpu.vendor == "AMD":
            target_backend = "rocm"
        elif profile.is_unified_memory:
            target_backend = "metal"
        else:
            target_backend = "cuda"

        # Construct reasoning report
        reasoning_parts = [
            f"Evaluated {effective_vram:.1f} GB accessible VRAM and {profile.ram_available_gb:.1f} GB RAM.",
            f"Selected {model} (~{param_b}B parameters, estimated weight: {cls.estimate_weight_size_gb(param_b):.2f} GB).",
            f"Calculated dynamic context: {num_ctx} tokens (KV Cache reserve: {cls.estimate_kv_cache_gb(num_ctx, param_b):.2f} GB).",
        ]

        if num_gpu == -1:
            reasoning_parts.append("100% of model layers offloaded to GPU.")
        elif num_gpu == 0:
            reasoning_parts.append("Executing entirely on CPU RAM.")
        else:
            reasoning_parts.append(
                f"Partial offload: {num_gpu} layers on GPU ({int(offload_ratio * 100)}%), remainder on CPU."
            )

        if concurrency:
            reasoning_parts.append(
                f"High-capacity mode: up to {max_loaded} models can remain resident in VRAM."
            )
        else:
            reasoning_parts.append(
                "Constrained mode: strictly enforcing sequential model swapping to prevent OOM."
            )

        return OptimizedParams(
            recommended_model=model,
            num_ctx=num_ctx,
            num_gpu=num_gpu,
            keep_alive=keep_alive,
            flash_attention=True,
            max_output_tokens=min(num_ctx, 4096),
            concurrency_allowed=concurrency,
            max_loaded_models=max_loaded,
            layer_offload_ratio=offload_ratio,
            target_backend=target_backend,
            kv_cache_type="q8_0",
            reasoning=" ".join(reasoning_parts),
        )

    @classmethod
    def get_recommended_model_for_role(cls, tier: HardwareTier, role: str) -> str:
        """Backward compatible helper to get recommended model for a given role and tier."""
        tier_cfg = cls.TIER_DEFAULTS.get(tier, cls.TIER_DEFAULTS[HardwareTier.TIER_1_4GB])
        return tier_cfg["recommended_models"].get(role, "qwen2.5:7b")
