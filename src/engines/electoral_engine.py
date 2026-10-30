"""ElectoralEngine: Multinomial Logit vote projection, alienation/abstention, and district-level aggregation."""

from __future__ import annotations

import math
import numpy as np
from numba import njit, prange

from src.core.agent_state import AgentStateEngine


@njit(parallel=True, fastmath=True)
def compute_multinomial_logit_votes_jit(
    opinions: np.ndarray,
    candidate_profiles: np.ndarray,
    indifference_threshold: float,
    rationality_lambda: float,
    gumbel_noise: np.ndarray,
    uniform_draws: np.ndarray,
    vote_probs: np.ndarray,
    vote_choice: np.ndarray,
) -> None:
    """Compute individual utility U_{ik} = -||x_i - theta_k||^2 + eta_{ik},
    Multinomial Logit choice probabilities, and abstention when max utility < threshold.
    """
    n_agents, n_dims = opinions.shape
    n_candidates = candidate_profiles.shape[0]

    for i in prange(n_agents):
        utilities = np.empty(n_candidates, dtype=np.float64)
        deterministic_u = np.empty(n_candidates, dtype=np.float64)
        max_det_u = -1e12
        max_u = -1e12

        for k in range(n_candidates):
            dist_sq = 0.0
            for d in range(n_dims):
                diff = opinions[i, d] - candidate_profiles[k, d]
                dist_sq += diff * diff
            det_u = -dist_sq
            u_ik = det_u + gumbel_noise[i, k]
            deterministic_u[k] = det_u
            utilities[k] = u_ik
            if det_u > max_det_u:
                max_det_u = det_u
            if u_ik > max_u:
                max_u = u_ik

        # Check alienation / abstention condition: if max utility falls below indifference_threshold
        if max_u < indifference_threshold:
            for k in range(n_candidates):
                vote_probs[i, k] = 0.0
            vote_probs[i, n_candidates] = 1.0  # Last column stores abstention probability
            vote_choice[i] = -1
        else:
            # Multinomial Logit softmax over candidates
            sum_exp = 0.0
            for k in range(n_candidates):
                w = math.exp(rationality_lambda * (utilities[k] - max_u))
                vote_probs[i, k] = w
                sum_exp += w

            inv_sum = 1.0 / (sum_exp + 1e-15)
            for k in range(n_candidates):
                vote_probs[i, k] *= inv_sum
            vote_probs[i, n_candidates] = 0.0

            # Sample categorical vote choice from probabilities
            target = uniform_draws[i]
            cum = 0.0
            chosen = n_candidates - 1
            for k in range(n_candidates):
                cum += vote_probs[i, k]
                if cum >= target:
                    chosen = k
                    break
            vote_choice[i] = chosen


class ElectoralEngine:
    """Projects categorical voting decisions, abstention rates, popular vote vs. district winners."""

    def __init__(
        self,
        n_candidates: int = 3,
        n_dims: int = 4,
        candidate_profiles: np.ndarray | None = None,
        indifference_threshold: float = -1.65,
        rationality_lambda: float = 4.0,
        noise_scale: float = 0.12,
    ) -> None:
        self.n_candidates = int(n_candidates)
        self.n_dims = int(n_dims)
        self.indifference_threshold = float(indifference_threshold)
        self.rationality_lambda = float(rationality_lambda)
        self.noise_scale = float(noise_scale)

        if candidate_profiles is None:
            self.candidate_profiles = self.default_candidate_profiles(self.n_candidates, self.n_dims)
        else:
            self.candidate_profiles = np.ascontiguousarray(candidate_profiles, dtype=np.float64)

    @staticmethod
    def default_candidate_profiles(n_candidates: int, n_dims: int) -> np.ndarray:
        """Create distinct ideological candidate profiles theta_k in [-1, 1]^d."""
        profiles = np.zeros((n_candidates, n_dims), dtype=np.float64)
        anchors = np.linspace(-0.75, 0.75, n_candidates)
        for k in range(n_candidates):
            for d in range(n_dims):
                sign = 1.0 if d % 2 == 0 else -1.0
                profiles[k, d] = np.clip(sign * anchors[k], -1.0, 1.0)
        return np.ascontiguousarray(profiles, dtype=np.float64)

    def evaluate_election(
        self,
        agent_state: AgentStateEngine,
        rng: np.random.Generator,
    ) -> dict:
        """Compute individual votes, popular vote shares, abstention rate, and district outcomes."""
        n = agent_state.n_agents
        k = self.n_candidates

        if self.noise_scale > 0.0:
            gumbel_noise = np.ascontiguousarray(
                rng.gumbel(loc=0.0, scale=self.noise_scale, size=(n, k)),
                dtype=np.float64,
            )
        else:
            gumbel_noise = np.zeros((n, k), dtype=np.float64)

        u_draws = np.ascontiguousarray(rng.random(n, dtype=np.float64))

        compute_multinomial_logit_votes_jit(
            agent_state.opinions,
            self.candidate_profiles,
            self.indifference_threshold,
            self.rationality_lambda,
            gumbel_noise,
            u_draws,
            agent_state.vote_probs,
            agent_state.vote_choice,
        )

        choices = agent_state.vote_choice
        abstentions = int(np.sum(choices == -1))
        abstention_rate = float(abstentions) / float(n)

        valid_mask = choices >= 0
        valid_votes = choices[valid_mask]
        if len(valid_votes) > 0:
            popular_counts = np.bincount(valid_votes, minlength=k).astype(np.int64)
            popular_shares = popular_counts / float(len(valid_votes))
            popular_winner = int(np.argmax(popular_counts))
        else:
            popular_counts = np.zeros(k, dtype=np.int64)
            popular_shares = np.zeros(k, dtype=np.float64)
            popular_winner = -1

        # Compute district-level winners and seats to quantify electoral geography discrepancy
        n_districts = agent_state.n_districts
        district_seats = np.zeros(k, dtype=np.int64)
        district_winners = np.full(n_districts, -1, dtype=np.int32)

        for d_id in range(n_districts):
            d_mask = (agent_state.district == d_id) & valid_mask
            d_votes = choices[d_mask]
            if len(d_votes) > 0:
                d_counts = np.bincount(d_votes, minlength=k)
                winner = int(np.argmax(d_counts))
                district_winners[d_id] = winner
                district_seats[winner] += 1

        valid_districts = int(np.sum(district_seats))
        district_seat_shares = (
            district_seats / float(valid_districts)
            if valid_districts > 0
            else np.zeros(k, dtype=np.float64)
        )
        district_winner = int(np.argmax(district_seats)) if valid_districts > 0 else -1

        # Discrepancy index (Total Variation / L1 / 2 between popular vote share and district seat share)
        discrepancy = 0.5 * float(np.sum(np.abs(popular_shares - district_seat_shares)))

        return {
            "abstentions": abstentions,
            "abstention_rate": abstention_rate,
            "popular_counts": popular_counts,
            "popular_shares": popular_shares,
            "popular_winner": popular_winner,
            "district_seats": district_seats,
            "district_seat_shares": district_seat_shares,
            "district_winners": district_winners,
            "district_winner": district_winner,
            "electoral_discrepancy": discrepancy,
            "winner_mismatch": bool(popular_winner != district_winner and popular_winner >= 0),
        }
