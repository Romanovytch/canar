"""Step 1 — resource_probe: no-op when disabled, numeric CPU/memory when enabled."""

import resource_probe
from resource_probe import ResourceUsage, hardware_profile, probe

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
    assert usage.gpu_util_pct is None
    assert usage.gpu_mem_delta_mb is None
    assert usage.gpu_mem_total_mb is None


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
    # GPU fields are best-effort: numeric when a GPU + pynvml exist, else None.
    assert usage.gpu_util_pct is None or usage.gpu_util_pct >= 0.0
    assert usage.gpu_mem_total_mb is None or usage.gpu_mem_total_mb > 0.0
    # The delta is the phase's own growth: never negative, and never larger
    # than the absolute total (#53).
    if usage.gpu_mem_delta_mb is not None:
        assert 0.0 <= usage.gpu_mem_delta_mb <= usage.gpu_mem_total_mb


def test_hardware_profile_has_core_keys():
    info = hardware_profile()
    for key in ("cpu", "os", "python"):
        assert key in info and info[key]


class _FakeNvml:
    """Simulates a shared GPU whose used memory only ever grows (the #53 bug
    scenario): a resident LLM server holds memory before the phase starts."""

    class _Mem:
        def __init__(self, used):
            self.used = used

    class _Util:
        gpu = 50

    def __init__(self, start_used, grow_to):
        self._used = start_used
        self._grow_to = grow_to

    def nvmlDeviceGetMemoryInfo(self, handle):
        used, self._used = self._used, self._grow_to  # grows after first read
        return self._Mem(used)

    def nvmlDeviceGetUtilizationRates(self, handle):
        return self._Util()


def test_gpu_delta_is_isolated_from_preexisting_memory(monkeypatch):
    if not psutil_available:
        return
    mb = 1024 * 1024
    # 60 GB already resident (LLM server), phase grows the device to 62 GB.
    fake = _FakeNvml(start_used=60_000 * mb, grow_to=62_000 * mb)
    monkeypatch.setattr(resource_probe, "pynvml", fake)
    monkeypatch.setattr(resource_probe, "_gpu_handle", lambda: object())

    with probe(enabled=True) as usage:
        _burn_cpu_and_memory()

    # total reflects the whole device; delta reflects only this phase's growth
    assert usage.gpu_mem_total_mb == 62_000
    assert usage.gpu_mem_delta_mb == 2_000
