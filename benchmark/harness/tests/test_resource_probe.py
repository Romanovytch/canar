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
    assert usage.gpu_mem_mb is None


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
    assert usage.gpu_mem_mb is None or usage.gpu_mem_mb > 0.0


def test_hardware_profile_has_core_keys():
    info = hardware_profile()
    for key in ("cpu", "os", "python"):
        assert key in info and info[key]
