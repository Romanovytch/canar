"""
Optional resource probe for the benchmark.

Two separate things, on purpose (issue #53):

- `probe(enabled)` measures **per-phase** CPU + memory of the benchmark process
  (retrieval / generation). These are the signals that differ between retrieval
  strategies, so they go into the per-strategy comparison.
- `gpu_context()` takes a **one-shot** snapshot of GPU usage for the whole run.
  GPU work all happens in the LLM server (same model for every strategy), so GPU
  memory is a property of the setup, not of the retrieval method — it is reported
  once as run context, not compared per strategy.

CPU + memory use `psutil`; GPU uses `pynvml`. Both are optional: when missing,
the corresponding values are simply absent. `probe` is a no-op when disabled.
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
    """Per-phase resource cost of the benchmark process. None when unmeasured."""
    cpu_s: float | None = None        # CPU seconds consumed (user + system)
    peak_rss_mb: float | None = None  # peak resident memory of this process


def _gpu_handle():
    """Handle to GPU 0, or None if pynvml is missing / no GPU is available."""
    if pynvml is None:
        return None
    try:
        pynvml.nvmlInit()
        return pynvml.nvmlDeviceGetHandleByIndex(0)
    except Exception:
        return None


class _RssSampler(threading.Thread):
    """Background thread: polls the process RSS to capture its peak during a phase."""

    def __init__(self, proc):
        super().__init__(daemon=True)
        self._proc = proc
        self._stop_event = threading.Event()
        self.peak_rss = 0

    def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.peak_rss = max(self.peak_rss, self._proc.memory_info().rss)
            except Exception:
                pass
            self._stop_event.wait(_SAMPLE_INTERVAL_S)

    def stop(self) -> None:
        self._stop_event.set()
        self.join(timeout=1.0)


@contextmanager
def probe(enabled: bool):
    """Measure per-phase CPU + memory of the benchmark process.

    Yields a ResourceUsage filled in when the block exits. When disabled (or
    psutil is missing) it yields an empty ResourceUsage and does nothing, so
    callers need no branching.
    """
    usage = ResourceUsage()
    if not enabled or psutil is None:
        yield usage
        return

    proc = psutil.Process()
    cpu_before = proc.cpu_times()
    sampler = _RssSampler(proc)
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


def _format_proc_breakdown(breakdown: dict[int, int]) -> str:
    """Render {pid: bytes} as 'name(pid)=MB; ...', resolving names best-effort."""
    parts = []
    for pid, mem in sorted(breakdown.items(), key=lambda kv: -kv[1]):
        name = "?"
        if psutil is not None:
            try:
                name = psutil.Process(pid).name()
            except Exception:
                pass
        parts.append(f"{name}({pid})={round(mem / _MB)}")
    return "; ".join(parts)


def gpu_context() -> dict:
    """One-shot GPU snapshot for the whole run (best-effort).

    Describes the GPU and how much of it the LLM server holds — a property of
    the setup, shared by every retrieval strategy. Call it once the model is
    loaded (e.g. after the runs) so the footprint is real. Empty dict when no
    GPU / pynvml.
    """
    gpu = _gpu_handle()
    if gpu is None:
        return {}
    ctx: dict[str, object] = {}
    try:
        ctx["gpu_mem_used_mb"] = round(pynvml.nvmlDeviceGetMemoryInfo(gpu).used / _MB)
    except Exception:
        pass
    try:
        procs = pynvml.nvmlDeviceGetComputeRunningProcesses(gpu)
        breakdown = {p.pid: p.usedGpuMemory for p in procs if p.usedGpuMemory is not None}
        if breakdown:
            ctx["gpu_procs"] = _format_proc_breakdown(breakdown)
    except Exception:
        pass
    return ctx


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
