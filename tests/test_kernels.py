"""Unit tests for contiguous memory allocation and Numba JIT kernels."""

from __future__ import annotations

import numpy as np
import pytest

from src.core.agent_state import AgentStateEngine
from src.core.multiplex_network import GraphEngine
from src.core.news_entity import NewsEntityPool
from src.kernels.opinion_updates import (
    DynamicsKernel,
    apply_media_assimilation_jit,
    step_interpersonal_dynamics_jit,
)
from src.kernels.recommendation_algos import (
    broadcast_exposure_kernel_jit,
    compute_closed_loop_recommendations_jit,
)
from src.kernels.whatsapp_cascades import propagate_whatsapp_cascades_jit


def test_agent_state_contiguity_and_bounds():
    state = AgentStateEngine.initialize(n_agents=200, n_dims=4, n_candidates=3, n_districts=5, seed=101)
    assert state.verify_contiguity()
    assert state.opinions.shape == (200, 4)
    assert np.all(state.age >= 16.0) and np.all(state.age <= 90.0)
    assert np.all(np.isin(state.gender, [0, 1, 2]))
    assert np.all(state.ses >= 0.0) and np.all(state.ses <= 1.0)
    assert np.all(state.opinions >= -1.0) and np.all(state.opinions <= 1.0)
    assert np.all(state.uncertainty > 0.0) and np.all(state.uncertainty <= 1.0)
    assert np.all(state.epsilon > 0.0)
    assert np.all(state.assimilation_bias >= 0.0)

    # Verify age calibration: older agents have lower mean susceptibility than younger agents
    young_mask = state.age < 35.0
    old_mask = state.age > 65.0
    assert np.mean(state.susceptibility[old_mask]) < np.mean(state.susceptibility[young_mask])


def test_interpersonal_dynamics_jit_kernel():
    state = AgentStateEngine.initialize(n_agents=150, n_dims=4, seed=202)
    graph = GraphEngine.build_multiplex(state, seed=202)
    csr = graph.composite_csr

    noise = np.zeros((150, 4), dtype=np.float64)
    new_op, new_unc = step_interpersonal_dynamics_jit(
        state.opinions,
        state.uncertainty,
        state.innate_beliefs,
        state.susceptibility,
        state.epsilon,
        state.assimilation_bias,
        state.coupling_matrix,
        csr.indptr,
        csr.indices,
        csr.data,
        0.2,
        noise,
    )
    assert new_op.shape == (150, 4)
    assert np.all(new_op >= -1.0) and np.all(new_op <= 1.0)
    assert np.all(new_unc > 0.0) and np.all(new_unc <= 1.0)


def test_recommendation_and_whatsapp_cascade_kernels():
    rng = np.random.default_rng(303)
    state = AgentStateEngine.initialize(n_agents=180, n_dims=4, seed=rng)
    graph = GraphEngine.build_multiplex(state, seed=rng)
    inf_ops = state.opinions[graph.influencer_ids]
    news = NewsEntityPool.initialize(
        n_items=25,
        n_dims=4,
        influencer_ids=graph.influencer_ids,
        influencer_opinions=inf_ops,
        seed=rng,
    )

    draws = np.ascontiguousarray(rng.random(state.n_agents, dtype=np.float64))
    selected = compute_closed_loop_recommendations_jit(
        state.opinions,
        news.framing,
        news.sensationalism,
        news.popularity,
        beta=2.5,
        emotional_boost=1.5,
        uniform_draws=draws,
    )
    assert selected.shape == (180,)
    assert np.all((selected >= 0) & (selected < 25))

    engaged = apply_media_assimilation_jit(
        state.opinions,
        state.uncertainty,
        state.susceptibility,
        state.epsilon,
        state.assimilation_bias,
        state.coupling_matrix,
        state.exposure_count,
        state.engagement_rate,
        selected,
        news.framing,
        news.sensationalism,
        news.veracity,
        media_strength=0.1,
    )
    assert np.sum(engaged) > 0
    assert np.all(state.opinions >= -1.0) and np.all(state.opinions <= 1.0)

    # Test WhatsApp viral cascade kernel
    n_groups = graph.n_wa_groups
    active_news = rng.integers(0, news.n_items, size=n_groups, dtype=np.int32)
    rand_mat = np.ascontiguousarray(rng.random((n_groups, 256), dtype=np.float64))
    forwards, fwd_members = propagate_whatsapp_cascades_jit(
        state.opinions,
        state.susceptibility,
        state.epsilon,
        state.assimilation_bias,
        state.coupling_matrix,
        state.exposure_count,
        graph.wa_group_indptr,
        graph.wa_group_members,
        active_news,
        news.framing,
        news.sensationalism,
        interpersonal_trust=0.9,
        base_forward_prob=0.4,
        random_matrix=rand_mat,
    )
    assert forwards.shape == (n_groups,)
    assert fwd_members.shape == (n_groups,)
    assert np.all(forwards >= 0)

    # Test broadcast exposure kernel
    reached = broadcast_exposure_kernel_jit(
        state.opinions,
        state.age,
        state.ses,
        state.susceptibility,
        state.epsilon,
        state.coupling_matrix,
        state.exposure_count,
        news.broadcast_orientations,
        news.broadcast_credibility,
        draws,
        broadcast_strength=0.05,
    )
    assert reached >= 0
