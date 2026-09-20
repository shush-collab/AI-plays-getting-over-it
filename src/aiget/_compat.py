"""Helpers for stable import paths after the package reorganization."""

from importlib import import_module
from typing import Any


def reexport(namespace: dict[str, Any], target: str) -> None:
    """Expose a canonical module through a legacy module namespace."""
    module = import_module(target)
    names = getattr(module, "__all__", [name for name in vars(module) if not name.startswith("_")])
    namespace.update({name: getattr(module, name) for name in names})
    namespace["__all__"] = names
    namespace["__getattr__"] = lambda name: getattr(module, name)
