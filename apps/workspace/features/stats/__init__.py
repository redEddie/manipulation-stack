"""Stats feature for WorkspaceWindow."""

from apps.workspace.features.stats.analysis_tab import build_analysis_tab
from apps.workspace.features.stats.history_ops import HistoryOps
from apps.workspace.features.stats.ops import StatsOps
from apps.workspace.features.stats.page import build_stats

__all__ = ["HistoryOps", "StatsOps", "build_stats", "build_analysis_tab"]
