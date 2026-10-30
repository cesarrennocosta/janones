"""Unit tests for the 5-layer multiplex CSR network and WhatsApp group constraints."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from src.core.agent_state import AgentStateEngine
from src.core.multiplex_network import GraphEngine, LAYER_NAMES
from src.metrics.network_metrics import (
    compute_ideological_segregation,
    compute_multiplex_structural_stats,
)


def test_five_layer_multiplex_csr_structure():
    state = AgentStateEngine.initialize(n_agents=250, n_dims=4, seed=404)
    graph = GraphEngine.build_multiplex(state, seed=404)

    assert set(graph.layers.keys()) == set(LAYER_NAMES)
    for name in LAYER_NAMES:
        layer = graph.layers[name]
        assert isinstance(layer, sp.csr_matrix)
        assert layer.shape == (250, 250)
        assert layer.nnz > 0

    # Composite CSR row sums must be 1.0 for connected nodes
    comp = graph.composite_csr
    assert isinstance(comp, sp.csr_matrix)
    row_sums = np.asarray(comp.sum(axis=1)).ravel()
    connected = row_sums > 0
    assert np.allclose(row_sums[connected], 1.0, atol=1e-6)


def test_whatsapp_group_size_constraint_and_influencers():
    state = AgentStateEngine.initialize(n_agents=300, n_dims=3, seed=505)
    graph = GraphEngine.build_multiplex(state, seed=505)

    # Verify WhatsApp group sizes K <= 256
    group_sizes = np.diff(graph.wa_group_indptr)
    assert len(group_sizes) == graph.n_wa_groups
    assert np.all(group_sizes > 0)
    assert np.all(group_sizes <= 256)

    # Verify influencer selection and eigenvector centrality
    assert len(graph.influencer_ids) >= 3
    assert np.sum(state.is_influencer) == len(graph.influencer_ids)
    assert graph.eigenvector_centrality.shape == (300,)

    stats = compute_multiplex_structural_stats(graph)
    assert "family_mean_degree" in stats
    seg = compute_ideological_segregation(state.opinions, graph.composite_csr)
    assert 0.0 <= seg <= 1.0
