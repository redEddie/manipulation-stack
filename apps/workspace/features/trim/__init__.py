"""Trim feature: trim ops, robot replay, HDF5 structure view, and the tab builder."""

from apps.workspace.features.trim.ops import TrimOps
from apps.workspace.features.trim.tab import build_trim_tab

__all__ = ["TrimOps", "build_trim_tab"]
