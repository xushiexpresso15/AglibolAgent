"""Hardware profiler for detecting CPU, RAM, GPU, VRAM, and Disk resources across diverse architectures."""

from __future__ import annotations

import platform
import shutil
import subprocess

import psutil

from aglibol.core.types import (
    DiskInfo,
    GPUInfo,
    HardwareProfile,
    HardwareTier,
)


class HardwareProfiler:
    """Detects system resources and calculates the host HardwareProfile and Tier dynamically."""

    @classmethod
    def detect(cls) -> HardwareProfile:
        """Scan the system hardware and return a comprehensive HardwareProfile."""
        # 1. OS Info
        os_name = platform.system()
        os_version = platform.release()

        # 2. CPU Info
        cpu_cores_physical = psutil.cpu_count(logical=False) or 1
        cpu_cores_logical = psutil.cpu_count(logical=True) or 1
        cpu_name = cls._get_cpu_name()

        # 3. RAM Info
        ram = psutil.virtual_memory()
        ram_total_gb = round(ram.total / (1024**3), 2)
        ram_available_gb = round(ram.available / (1024**3), 2)
        ram_used_percent = ram.percent

        # 4. GPU & Unified Memory Detection
        gpus = cls._detect_gpus()
        primary_gpu = gpus[0] if gpus else None
        gpu_count = len(gpus)

        # Check for Apple Silicon unified memory
        is_unified = False
        if os_name == "Darwin" and platform.machine().lower() in ("arm64", "aarch64"):
            is_unified = True
            # On Apple Silicon, system RAM functions as unified graphics memory
            if not gpus:
                unified_gpu = GPUInfo(
                    name="Apple Silicon Unified GPU",
                    vendor="Apple",
                    vram_total_mb=round(ram.total / (1024**2), 2),
                    vram_free_mb=round(ram.available / (1024**2), 2),
                    vram_used_mb=round((ram.total - ram.available) / (1024**2), 2),
                )
                gpus.append(unified_gpu)
                primary_gpu = unified_gpu
                gpu_count = 1

        total_vram_gb = round(sum(g.vram_total_gb for g in gpus), 2)
        total_vram_free_gb = round(sum(g.vram_free_gb for g in gpus), 2)
        has_discrete = bool(primary_gpu and total_vram_gb > 0)

        # 5. Disk Info
        disks = cls._detect_disks()

        # 6. Determine Tier
        if has_discrete:
            tier = HardwareTier.from_vram_gb(total_vram_gb, has_discrete_gpu=True)
        else:
            tier = HardwareTier.TIER_0_CPU

        return HardwareProfile(
            os_name=os_name,
            os_version=os_version,
            cpu_name=cpu_name,
            cpu_cores_physical=cpu_cores_physical,
            cpu_cores_logical=cpu_cores_logical,
            ram_total_gb=ram_total_gb,
            ram_available_gb=ram_available_gb,
            ram_used_percent=ram_used_percent,
            gpus=gpus,
            primary_gpu=primary_gpu,
            disks=disks,
            tier=tier,
            has_discrete_gpu=has_discrete,
            is_unified_memory=is_unified,
            total_vram_gb=total_vram_gb,
            total_vram_free_gb=total_vram_free_gb,
            gpu_count=gpu_count,
        )

    @staticmethod
    def _get_cpu_name() -> str:
        """Retrieve a human-readable CPU model name."""
        if platform.system() == "Windows":
            cpu = platform.processor()
            if cpu:
                return cpu.strip()
        elif platform.system() == "Darwin":
            try:
                out = subprocess.check_output(
                    ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
                )
                return out.strip()
            except Exception:
                pass
        elif platform.system() == "Linux":
            try:
                with open("/proc/cpuinfo") as f:
                    for line in f:
                        if "model name" in line:
                            return line.split(":", 1)[1].strip()
            except Exception:
                pass
        return platform.machine() or "Unknown CPU"

    @classmethod
    def _detect_gpus(cls) -> list[GPUInfo]:
        """Detect GPUs using pynvml (NVIDIA), falling back to nvidia-smi, rocm-smi, or clinfo."""
        gpus: list[GPUInfo] = []

        # 1. Try NVML first (NVIDIA)
        try:
            import pynvml

            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode("utf-8")
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

                temp = None
                try:
                    temp = float(
                        pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    )
                except Exception:
                    pass

                util = None
                try:
                    rates = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    util = float(rates.gpu)
                except Exception:
                    pass

                gpu = GPUInfo(
                    name=str(name),
                    vendor="NVIDIA",
                    vram_total_mb=round(mem_info.total / (1024**2), 2),
                    vram_free_mb=round(mem_info.free / (1024**2), 2),
                    vram_used_mb=round(mem_info.used / (1024**2), 2),
                    temperature_c=temp,
                    utilization_percent=util,
                )
                gpus.append(gpu)
            pynvml.nvmlShutdown()
            if gpus:
                return gpus
        except Exception:
            pass

        # 2. Fallback to nvidia-smi command if pynvml was unavailable or failed
        if shutil.which("nvidia-smi"):
            try:
                cmd = [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,memory.free,memory.used,temperature.gpu,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ]
                output = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
                for line in output.strip().splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 4:
                        name = parts[0]
                        try:
                            total_mb = float(parts[1])
                            free_mb = float(parts[2])
                            used_mb = float(parts[3])
                        except ValueError:
                            continue
                        temp = (
                            float(parts[4])
                            if len(parts) > 4 and parts[4].replace(".", "", 1).isdigit()
                            else None
                        )
                        util = (
                            float(parts[5])
                            if len(parts) > 5 and parts[5].replace(".", "", 1).isdigit()
                            else None
                        )
                        gpus.append(
                            GPUInfo(
                                name=name,
                                vendor="NVIDIA",
                                vram_total_mb=total_mb,
                                vram_free_mb=free_mb,
                                vram_used_mb=used_mb,
                                temperature_c=temp,
                                utilization_percent=util,
                            )
                        )
                if gpus:
                    return gpus
            except Exception:
                pass

        # 3. Fallback to rocm-smi (AMD on Linux)
        if shutil.which("rocm-smi"):
            try:
                import csv
                import io

                cmd = ["rocm-smi", "--showmeminfo", "vram", "--csv"]
                out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
                reader = csv.DictReader(io.StringIO(out))
                for row in reader:
                    total_b = 0.0
                    used_b = 0.0
                    for k, v in row.items():
                        k_lower = (k or "").lower()
                        if "total" in k_lower and "vram" in k_lower:
                            try:
                                total_b = float(v)
                            except (ValueError, TypeError):
                                pass
                        elif "used" in k_lower and "vram" in k_lower:
                            try:
                                used_b = float(v)
                            except (ValueError, TypeError):
                                pass
                    total_mb = total_b / (1024 * 1024) if total_b > 0 else 8192.0
                    used_mb = used_b / (1024 * 1024) if used_b > 0 else 0.0
                    free_mb = max(0.0, total_mb - used_mb)
                    device_id = row.get("device", "0")
                    gpus.append(
                        GPUInfo(
                            name=f"AMD Radeon GPU (ROCm dev {device_id})",
                            vendor="AMD",
                            vram_total_mb=round(total_mb, 1),
                            vram_free_mb=round(free_mb, 1),
                            vram_used_mb=round(used_mb, 1),
                        )
                    )
                if gpus:
                    return gpus
            except Exception:
                pass

        return gpus

    @staticmethod
    def _detect_disks() -> list[DiskInfo]:
        """Detect local disk partitions and free capacity."""
        disks: list[DiskInfo] = []
        try:
            for part in psutil.disk_partitions(all=False):
                if not part.mountpoint:
                    continue
                try:
                    usage = psutil.disk_usage(part.mountpoint)
                    disks.append(
                        DiskInfo(
                            mountpoint=part.mountpoint,
                            total_gb=round(usage.total / (1024**3), 2),
                            free_gb=round(usage.free / (1024**3), 2),
                            used_percent=usage.percent,
                        )
                    )
                except (PermissionError, OSError):
                    continue
        except Exception:
            pass
        return disks
