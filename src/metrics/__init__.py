"""AnalyticsEngine and polarization metrics: Multidimensional Esteban-Ray, Sarle's bimodality, and segregation."""

from src.metrics.polarization import (
    AnalyticsEngine,
    compute_esteban_ray_index,
    compute_sarle_bimodality,
)
from src.metrics.network_metrics import (
    compute_ideological_segregation,
    compute_multiplex_structural_stats,
)

__all__ = [
    "AnalyticsEngine",
    "compute_esteban_ray_index",
    "compute_sarle_bimodality",
    "compute_ideological_segregation",
    "compute_multiplex_structural_stats",
]
