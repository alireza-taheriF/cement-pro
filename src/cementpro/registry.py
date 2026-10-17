"""Tag registry: canonical tag metadata, aliases and unit rules loaded from YAML."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class TagSpec:
    """Canonical description of one process or lab tag."""

    name: str
    unit: str
    description: str = ""
    phys_min: float = float("-inf")
    phys_max: float = float("inf")
    sample_period_s: int = 60
    flatline_min_samples: int = 20
    hampel_window: int = 9
    hampel_sigmas: float = 6.0
    is_lab: bool = False
    aliases: dict[str, str] = field(default_factory=dict)  # alias -> source unit


@dataclass
class Registry:
    """Lookup of canonical tags, alias resolution and source timezones."""

    tags: dict[str, TagSpec]
    source_timezones: dict[str, str]

    def __post_init__(self) -> None:
        self._alias: dict[str, tuple[str, str]] = {}
        for spec in self.tags.values():
            self._alias[spec.name.upper()] = (spec.name, spec.unit)
            for alias, unit in spec.aliases.items():
                self._alias[alias.upper()] = (spec.name, unit)

    def resolve(self, raw_name: str) -> tuple[str, str] | None:
        """Return ``(canonical_name, source_unit)`` for a raw tag name, or ``None`` if unknown."""
        return self._alias.get(str(raw_name).strip().upper())

    @property
    def process_tags(self) -> list[str]:
        return [n for n, s in self.tags.items() if not s.is_lab]

    @property
    def lab_tags(self) -> list[str]:
        return [n for n, s in self.tags.items() if s.is_lab]


def load_registry(path: str | Path) -> Registry:
    """Load a tag registry from a YAML file (see ``config/tags.yaml``)."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    defaults = raw.get("defaults", {})
    tags: dict[str, TagSpec] = {}
    for section, is_lab in (("tags", False), ("lab", True)):
        for name, cfg in (raw.get(section) or {}).items():
            merged = {**defaults, **cfg}
            tags[name] = TagSpec(
                name=name,
                unit=merged["unit"],
                description=merged.get("description", ""),
                phys_min=float(merged.get("phys_min", float("-inf"))),
                phys_max=float(merged.get("phys_max", float("inf"))),
                sample_period_s=int(merged.get("sample_period_s", 60)),
                flatline_min_samples=int(merged.get("flatline_min_samples", 20)),
                hampel_window=int(merged.get("hampel_window", 9)),
                hampel_sigmas=float(merged.get("hampel_sigmas", 6.0)),
                is_lab=is_lab,
                aliases=dict(merged.get("aliases", {}) or {}),
            )
    tz = {k: v.get("timezone", "UTC") for k, v in (raw.get("sources") or {}).items()}
    return Registry(tags=tags, source_timezones=tz)
