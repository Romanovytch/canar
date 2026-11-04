"""Build retrieval profiles from plain configuration mappings.

Configured profiles either extend a known profile (``extends``) or define a
complete profile from scratch (``strategy`` + ``collections`` + required blocks).

Merge rules when extending:

- root values (``fetch_top_k``, ``min_score``, ``collections``...) replace the base value;
- parameter blocks (``dense``, ``sparse``, ``fusion``, ``rerank``, ``parent_child``,
  ``summary``) are merged key by key into the base block, or created when the base
  has none;
- a block set to ``null`` removes it from the profile (e.g. ``rerank: null``);
- ``fusion.weights`` is replaced as a whole.

Validation is delegated to the dataclasses in ``models.py``; this module only adds
checks for unknown keys, unknown or cyclic ``extends`` and name clashes.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import fields, replace
from typing import Any

from canar.app.retrieval.models import (
    DenseRetrievalParams,
    FusionRetrievalParams,
    ParentChildRetrievalParams,
    RerankRetrievalParams,
    RetrievalProfile,
    SparseRetrievalParams,
    SummaryRetrievalParams,
)

_BLOCK_TYPES: dict[str, type] = {
    "dense": DenseRetrievalParams,
    "sparse": SparseRetrievalParams,
    "fusion": FusionRetrievalParams,
    "rerank": RerankRetrievalParams,
    "parent_child": ParentChildRetrievalParams,
    "summary": SummaryRetrievalParams,
}
_EXTENDS_KEY = "extends"
_PROFILE_KEYS = frozenset(f.name for f in fields(RetrievalProfile)) - {"name"}


def build_configured_profiles(
    specs: Mapping[str, Mapping[str, Any]],
    presets: Mapping[str, RetrievalProfile],
) -> dict[str, RetrievalProfile]:
    """Return the profiles defined by ``specs``, resolved against ``presets``.

    ``presets`` are not included in the result. Raises ``ValueError`` with the
    offending profile name when a spec is invalid.
    """
    clashes = sorted(set(specs) & set(presets))
    if clashes:
        raise ValueError(
            "Configured retrieval profile(s) reuse a built-in profile name: "
            + ", ".join(clashes)
            + ". Use 'extends' with a new name instead."
        )

    built: dict[str, RetrievalProfile] = {}
    resolving: list[str] = []

    def resolve(name: str) -> RetrievalProfile:
        if name in built:
            return built[name]
        if name in presets:
            return presets[name]
        if name in resolving:
            cycle = " -> ".join([*resolving[resolving.index(name) :], name])
            raise ValueError(f"Cyclic 'extends' between retrieval profiles: {cycle}")
        spec = specs.get(name)
        if spec is None:
            raise ValueError(f"Unknown retrieval profile: {name!r}")

        resolving.append(name)
        try:
            built[name] = _build_profile(name, spec, resolve)
        finally:
            resolving.pop()
        return built[name]

    for name in specs:
        resolve(name)
    return built


def _build_profile(
    name: str,
    spec: Mapping[str, Any],
    resolve: Callable[[str], RetrievalProfile],
) -> RetrievalProfile:
    if not isinstance(spec, Mapping):
        raise ValueError(f"Retrieval profile {name!r} must be a mapping of settings")

    unknown = sorted(set(spec) - _PROFILE_KEYS - {_EXTENDS_KEY})
    if unknown:
        raise ValueError(
            f"Retrieval profile {name!r} has unknown setting(s): {', '.join(unknown)}. "
            f"Allowed: {', '.join(sorted(_PROFILE_KEYS | {_EXTENDS_KEY}))}"
        )

    base_name = spec.get(_EXTENDS_KEY)
    if base_name is not None and not isinstance(base_name, str):
        raise ValueError(f"Retrieval profile {name!r}: 'extends' must be a profile name")
    base = resolve(base_name) if base_name else None

    values: dict[str, Any] = {}
    for key, value in spec.items():
        if key == _EXTENDS_KEY:
            continue
        if key in _BLOCK_TYPES:
            current = getattr(base, key) if base is not None else None
            values[key] = _merge_block(name, key, current, value)
        elif key == "collections":
            values[key] = _collections(name, value)
        else:
            values[key] = value

    try:
        if base is not None:
            return replace(base, name=name, **values)
        missing = [key for key in ("strategy", "collections") if key not in values]
        if missing:
            raise ValueError(
                f"must define {' and '.join(repr(key) for key in missing)}, "
                f"or 'extends' an existing profile"
            )
        return RetrievalProfile(name=name, **values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Retrieval profile {name!r}: {exc}") from exc


def _merge_block(profile: str, key: str, current: Any, value: Any) -> Any:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"Retrieval profile {profile!r}: '{key}' must be a mapping or null")

    block_type = _BLOCK_TYPES[key]
    allowed = {f.name for f in fields(block_type)}
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(
            f"Retrieval profile {profile!r}: unknown '{key}' setting(s): {', '.join(unknown)}. "
            f"Allowed: {', '.join(sorted(allowed))}"
        )

    items = dict(value)
    if isinstance(items.get("weights"), Mapping):
        items["weights"] = dict(items["weights"])
    try:
        if current is None:
            return block_type(**items)
        return replace(current, **items)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Retrieval profile {profile!r}: {exc}") from exc


def _collections(profile: str, value: Any) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        raise ValueError(f"Retrieval profile {profile!r}: 'collections' must be a list")
    if not all(isinstance(item, str) and item for item in value):
        raise ValueError(
            f"Retrieval profile {profile!r}: 'collections' must contain non-empty names"
        )
    return tuple(value)
