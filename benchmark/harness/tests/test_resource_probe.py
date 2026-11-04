"""resource_probe: per-phase CPU/memory via probe(); one-shot GPU via gpu_context()."""

import resource_probe
from resource_probe import ResourceUsage, gpu_context, hardware_profile, probe

psutil_available = resource_probe.psutil is not None


def _burn_cpu_and_memory():
    # a little work so cpu_s > 0 and rss is touched
    blob = [i * i for i in range(200_000)]
    return sum(blob)


def test_disabled_is_a_noop():
    with probe(enabled=False) as usage:
        _burn_cpu_and_memory()
    assert isinstance(usage, ResourceUsage)
    assert usage.cpu_s is None
    assert usage.peak_rss_mb is None


def test_enabled_measures_cpu_and_memory():
    if not psutil_available:
        # Without psutil the enabled path must still be a safe no-op.
        with probe(enabled=True) as usage:
            _burn_cpu_and_memory()
        assert usage.cpu_s is None
        return

    with probe(enabled=True) as usage:
        _burn_cpu_and_memory()
    assert usage.cpu_s is not None and usage.cpu_s >= 0.0
    assert usage.peak_rss_mb is not None and usage.peak_rss_mb > 0.0


def test_hardware_profile_has_core_keys():
    info = hardware_profile()
    for key in ("cpu", "os", "python"):
        assert key in info and info[key]


class _FakeNvml:
    """Minimal NVML stand-in for the one-shot gpu_context() snapshot."""

    class _Mem:
        def __init__(self, used):
            self.used = used

    class _Proc:
        def __init__(self, pid, used):
            self.pid = pid
            self.usedGpuMemory = used

    def __init__(self, used, procs):
        self._used = used
        self._procs = procs

    def nvmlDeviceGetMemoryInfo(self, handle):
        return self._Mem(self._used)

    def nvmlDeviceGetComputeRunningProcesses(self, handle):
        return self._procs


def test_gpu_context_reports_used_memory_and_names_holders(monkeypatch):
    mb = 1024 * 1024
    fake = _FakeNvml(
        used=66_000 * mb,
        procs=[
            _FakeNvml._Proc(4242, 61_000 * mb),
            _FakeNvml._Proc(4243, None),  # unattributable — skipped
        ],
    )
    monkeypatch.setattr(resource_probe, "pynvml", fake)
    monkeypatch.setattr(resource_probe, "_gpu_handle", lambda: object())

    ctx = gpu_context()

    assert ctx["gpu_mem_used_mb"] == 66_000
    assert "(4242)=61000" in ctx["gpu_procs"]
    assert "4243" not in ctx["gpu_procs"]


def test_gpu_context_is_empty_without_gpu(monkeypatch):
    monkeypatch.setattr(resource_probe, "_gpu_handle", lambda: None)
    assert gpu_context() == {}
