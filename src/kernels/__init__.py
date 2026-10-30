"""Numba-compiled JIT kernels for opinion dynamics, algorithmic recommendation, and viral WhatsApp cascades."""

from src.kernels.opinion_updates import (
    DynamicsKernel,
    step_interpersonal_dynamics_jit,
    apply_media_assimilation_jit,
)
from src.kernels.recommendation_algos import (
    compute_closed_loop_recommendations_jit,
    broadcast_exposure_kernel_jit,
)
from src.kernels.whatsapp_cascades import (
    propagate_whatsapp_cascades_jit,
)

__all__ = [
    "DynamicsKernel",
    "step_interpersonal_dynamics_jit",
    "apply_media_assimilation_jit",
    "compute_closed_loop_recommendations_jit",
    "broadcast_exposure_kernel_jit",
    "propagate_whatsapp_cascades_jit",
]
