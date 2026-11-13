"""Numba JIT kernel for viral cascade propagation in closed WhatsApp messaging groups.

Author: César Rennó-Costa (Brain Institute, UFRN, Natal, RN, Brazil)
Paper: "Late-campaign opinion volatility and algorithmic counter-mobilization in polarized elections"
Target: Nature Human Behaviour (2026)

Governing Structure:
- WhatsApp micro-clique forwarding: K <= 256 participants (Supplementary Note 3, Eq. S47)
- Peer trust multiplier \\tau_{WA} = 0.85 yielding 2.8x higher persuasion efficiency
"""

from __future__ import annotations

import math
import numpy as np
from numba import njit, prange


@njit(parallel=True, fastmath=True)
def propagate_whatsapp_cascades_jit(
    opinions: np.ndarray,
    susceptibility: np.ndarray,
    epsilon: np.ndarray,
    assimilation_bias: np.ndarray,
    coupling_matrix: np.ndarray,
    exposure_count: np.ndarray,
    wa_group_indptr: np.ndarray,
    wa_group_members: np.ndarray,
    group_active_news: np.ndarray,
    news_framing: np.ndarray,
    news_sensationalism: np.ndarray,
    interpersonal_trust: float,
    base_forward_prob: float,
    random_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate viral diffusion and interpersonal influence inside closed WhatsApp groups (K <= 256).

    No centralized algorithmic curation: high interpersonal trust boosts assimilation,
    and members forward the message to other groups when attitudinal concordance is high.

    Returns:
    - group_forwards: int64 array of shape (n_groups,) counting viral forwards per group.
    - member_forwarding_agent: int32 array of shape (n_groups,) with ID of last forwarding member (or -1).
    """
    n_groups = len(wa_group_indptr) - 1
    n_dims = opinions.shape[1]
    inv_sqrt_d = 1.0 / math.sqrt(float(n_dims))

    group_forwards = np.zeros(n_groups, dtype=np.int64)
    member_forwarding_agent = np.full(n_groups, -1, dtype=np.int32)

    for g in prange(n_groups):
        n_idx = group_active_news[g]
        if n_idx < 0:
            continue

        start = wa_group_indptr[g]
        end = wa_group_indptr[g + 1]
        e_n = news_sensationalism[n_idx]

        for ptr in range(start, end):
            member = wa_group_members[ptr]
            offset_in_group = ptr - start

            # Compute attitudinal concordance between group member and shared message
            dist_sq = 0.0
            for d in range(n_dims):
                diff = news_framing[n_idx, d] - opinions[member, d]
                dist_sq += diff * diff
            dist = math.sqrt(dist_sq) * inv_sqrt_d

            # High interpersonal trust widens effective confidence threshold in private groups
            eff_eps = epsilon[member] * (1.0 + 0.45 * interpersonal_trust)
            if dist <= eff_eps:
                concordance = max(0.0, 1.0 - (dist / (eff_eps + 1e-9)))
                exposure_count[member] += 1

                # Assimilate opinion with interpersonal trust boost
                shift_rate = (
                    0.06
                    * interpersonal_trust
                    * susceptibility[member]
                    * (1.0 + 0.25 * assimilation_bias[member] * concordance)
                )

                for d in range(n_dims):
                    val = opinions[member, d] + shift_rate * (news_framing[n_idx, d] - opinions[member, d])
                    if val > 1.0:
                        val = 1.0
                    elif val < -1.0:
                        val = -1.0
                    opinions[member, d] = val

                # Viral cascade forwarding probability depends on attitudinal concordance & sensationalism
                forward_prob = base_forward_prob * concordance * (1.0 + 0.6 * e_n)
                draw = random_matrix[g, offset_in_group % random_matrix.shape[1]]
                if draw < forward_prob:
                    group_forwards[g] += 1
                    member_forwarding_agent[g] = member

    return group_forwards, member_forwarding_agent
