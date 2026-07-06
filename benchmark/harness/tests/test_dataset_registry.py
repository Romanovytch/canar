"""Task 2 — the loader abstraction: dispatch by extension / declared format,
and a clear error for unsupported formats. Uses dummy loaders so it tests the
registry itself, independent of the real CSV/YAML loaders (Tasks 3-4)."""

import pytest
from dataset_loaders import registry
from dataset_loaders.item import DatasetItem
from dataset_loaders.registry import (
    UnsupportedDatasetFormat,
    load_dataset,
    register_loader,
    supported_formats,
)


@pytest.fixture(autouse=True)
def _isolate_registry(monkeypatch):
    """Each test gets a clean registry so dummy loaders don't leak."""
    monkeypatch.setattr(registry, "_LOADERS", {})


def test_dispatch_by_extension(tmp_path):
    @register_loader(".csv")
    def _load(path):
        return [DatasetItem(query=f"loaded {path.name}", reference="r")]

    items = load_dataset(tmp_path / "questions.csv")
    assert items[0].query == "loaded questions.csv"


def test_declared_format_overrides_extension(tmp_path):
    @register_loader(".yaml")
    def _load(path):
        return [DatasetItem(query="yaml", reference="r")]

    # File has a .txt extension but we declare it as yaml.
    items = load_dataset(tmp_path / "dataset.txt", fmt="yaml")
    assert items[0].query == "yaml"


def test_one_loader_can_register_several_extensions():
    @register_loader(".yaml", ".yml")
    def _load(path):
        return []

    assert supported_formats() == [".yaml", ".yml"]


def test_unsupported_format_lists_supported(tmp_path):
    @register_loader(".csv")
    def _load(path):
        return []

    with pytest.raises(UnsupportedDatasetFormat) as exc:
        load_dataset(tmp_path / "data.json")
    msg = str(exc.value)
    assert ".json" in msg
    assert ".csv" in msg  # the message tells you what *is* supported
