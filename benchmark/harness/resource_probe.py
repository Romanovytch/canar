"""
Optional resource probe for the benchmark.

Measures CPU, memory and (best-effort) GPU usage around a phase of the pipeline
(retrieval / generation). It is a **no-op unless enabled**, so the default run
keeps zero overhead and needs no extra dependency.

    with probe(enabled) as usage:
        ... do the work ...
    # usage.cpu_s / peak_rss_mb / gpu_util_pct / gpu_mem_mb are now filled in

CPU + memory are process-level via `psutil`. GPU is **device-level** via
`pynvml`: generation and embedding run in a separate server process (calls go
out over an HTTP endpoint), so per-process GPU attribution isn't possible — the
GPU numbers are the whole device's usage during the phase. Both libraries are
optional; when missing, the corresponding fields stay `None`.
"""

from __future__ import annotations

import platform
import threading
from contextlib import contextmanager
from dataclasses import dataclass

try:
    import psutil
except ImportError:  # optional dependency
    psutil = None

try:
    import pynvml
except ImportError:  # optional dependency
    pynvml = None

_SAMPLE_INTERVAL_S = 0.05
_MB = 1024 * 1024


@dataclass
class ResourceUsage:
    """Resource cost of one phase. Fields stay None when unmeasured.

    GPU memory is reported two ways so runs stay comparable (issue #53):
    `gpu_mem_delta_mb` is the growth caused by the phase (peak minus the value
    at phase start) — the number to compare methods with. `gpu_mem_total_mb`
    is the absolute device usage at peak, kept for context; on a shared server
    it includes the LLM server's loaded models and other tenants.
    """
    cpu_s: float | None = None            # CPU seconds consumed (user + system)
    peak_rss_mb: float | None = None      # peak resident memory of this process
    gpu_util_pct: float | None = None     # mean GPU utilization (device-wide)
    gpu_mem_delta_mb: float | None = None  # GPU memory the phase added (device-wide)
    gpu_mem_total_mb: float | None = None  # absolute device memory at peak


def _gpu_handle():
    """Handle to GPU 0, or None if pynvml is missing / no GPU is available."""
    if pynvml is None:
        return None
    try:
        pynvml.nvmlInit()
        return pynvml.nvmlDeviceGetHandleByIndex(0)
    except Exception:
        return None


class _Sampler(threading.Thread):
    """Background thread: polls process RSS and (optionally) GPU while a phase runs."""

    def __init__(self, proc, gpu_handle):
        super().__init__(daemon=True)
        self._proc = proc
        self._gpu = gpu_handle
        self._stop_event = threading.Event()
        self.peak_rss = 0
        self.gpu_util_samples: list[float] = []
        self.peak_gpu_mem = 0
        # Device memory at phase start: the peak is compared against this to get
        # the phase's own delta, so numbers don't accumulate across runs (#53).
        self.baseline_gpu_mem = 0
        if gpu_handle is not None:
            try:
                self.baseline_gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle).used
                self.peak_gpu_mem = self.baseline_gpu_mem
            except Exception:
                pass

    def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.peak_rss = max(self.peak_rss, self._proc.memory_info().rss)
            except Exception:
                pass
            if self._gpu is not None:
                try:
                    self.gpu_util_samples.append(
                        pynvml.nvmlDeviceGetUtilizationRates(self._gpu).gpu
                    )
                    self.peak_gpu_mem = max(
                        self.peak_gpu_mem, pynvml.nvmlDeviceGetMemoryInfo(self._gpu).used
                    )
                except Exception:
                    pass
            self._stop_event.wait(_SAMPLE_INTERVAL_S)

    def stop(self) -> None:
        self._stop_event.set()
        self.join(timeout=1.0)


@contextmanager
def probe(enabled: bool):
    """Measure resource usage for the wrapped block.

    Yields a ResourceUsage that is filled in when the block exits. When disabled
    (or psutil is missing) it yields an empty ResourceUsage and does nothing, so
    callers need no branching.
    """
    usage = ResourceUsage()
    if not enabled or psutil is None:
        yield usage
        return

    proc = psutil.Process()
    cpu_before = proc.cpu_times()
    sampler = _Sampler(proc, _gpu_handle())
    try:
        sampler.peak_rss = proc.memory_info().rss  # seed with the starting value
    except Exception:
        pass
    sampler.start()
    try:
        yield usage
    finally:
        sampler.stop()
        cpu_after = proc.cpu_times()
        usage.cpu_s = (
            (cpu_after.user - cpu_before.user)
            + (cpu_after.system - cpu_before.system)
        )
        if sampler.peak_rss:
            usage.peak_rss_mb = sampler.peak_rss / _MB
        if sampler.gpu_util_samples:
            usage.gpu_util_pct = sum(sampler.gpu_util_samples) / len(sampler.gpu_util_samples)
        if sampler.peak_gpu_mem:
            usage.gpu_mem_total_mb = sampler.peak_gpu_mem / _MB
            usage.gpu_mem_delta_mb = max(
                0.0, (sampler.peak_gpu_mem - sampler.baseline_gpu_mem) / _MB
            )


def hardware_profile() -> dict:
    """One-shot machine description for run provenance (best-effort).

    Cheap to call; used to stamp results so runs are comparable across machines.
    Missing libraries just mean fewer keys.
    """
    info: dict[str, object] = {
        "cpu": platform.processor() or platform.machine(),
        "os": platform.system(),
        "python": platform.python_version(),
    }
    if psutil is not None:
        info["cpu_cores"] = psutil.cpu_count(logical=True)
        info["ram_gb"] = round(psutil.virtual_memory().total / (1024 ** 3), 1)
    gpu = _gpu_handle()
    if gpu is not None:
        try:
            name = pynvml.nvmlDeviceGetName(gpu)
            info["gpu"] = name.decode() if isinstance(name, bytes) else name
        except Exception:
            pass
    return info
