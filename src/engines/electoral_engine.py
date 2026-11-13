"""ElectoralEngine: Multinomial Logit vote projection, Two-Leg election with
independent Candidate Admiration & Rejection, and Three-Way Voting (Abstention / Blank / Valid)
based on expressed opinions elastically tethered to core convictions.

Author: César Rennó-Costa (Brain Institute, UFRN, Natal, RN, Brazil)
Paper: "Late-campaign opinion volatility and algorithmic counter-mobilization in polarized elections"
Target: Nature Human Behaviour (2026)

Governing Equations:
- Dynamic Candidate Affinity: A_{ik}(t+1) = (1 - \\delta_A) A_{ik} + \\lambda_A max(0, (0.62 - d)/0.62)^2 (Eq. 13, Eq. S55)
- Dynamic Candidate Aversion: R_{ik}(t+1) = (1 - \\delta_R) R_{ik} + \\lambda_R max(0, (d - 0.78)/1.22)^{1.35} (Eq. 14, Eq. S56)
- Quadratic Utility Penalty: U_{ik}(t) = A_{ik}(t) - \\theta_{rej} R_{ik}(t)^2 (Eq. 15, Eq. S57)
- Multinomial Logit Choice: P(Vote k | turnout) \\propto exp(\\beta_{vote} U_{ik}) (Eq. S58)
- Logistic Turnout Mobilization: P(Turnout | i) = 1 / (1 + exp(-\\mu_{turn}[max_k U_{ik} - (\\tau_{abst} - 0.22 SES_i + \\delta_{transit})])) (Eq. S59)
"""

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


@njit(parallel=True, fastmath=True)
def step_candidate_admiration_rejection_jit(
    opinions: np.ndarray,
    candidate_profiles: np.ndarray,
    admiration: np.ndarray,
    rejection: np.ndarray,
    adm_dist_threshold: float,
    rej_dist_threshold: float,
    adm_gain_rate: float,
    adm_decay_rate: float,
    rej_gain_rate: float,
    rej_decay_rate: float,
    initialize_mode: bool,
) -> None:
    """Update each citizen's independent Admiration A_{i,k} in [0, 1] and Rejection R_{i,k} in [0, 1]
    toward each candidate k based on CURRENT OPINION x_i(t):
      - Very close current opinion (d_{i,k} < adm_dist_threshold) increases Admiration.
      - Very different current opinion (d_{i,k} > rej_dist_threshold) increases Rejection.
      - Admiration and Rejection evolve as independent state variables with distinct thresholds and memory.
    """
    n_agents, n_dims = opinions.shape
    n_candidates = candidate_profiles.shape[0]
    inv_sqrt_d = 1.0 / math.sqrt(float(n_dims))
    rej_span = max(0.25, 1.35 - rej_dist_threshold)

    for i in prange(n_agents):
        for k in range(n_candidates):
            dist_sq = 0.0
            for d in range(n_dims):
                diff = opinions[i, d] - candidate_profiles[k, d]
                dist_sq += diff * diff
            dist_norm = math.sqrt(dist_sq) * inv_sqrt_d

            # 1. Admiration impulse from very close current opinion
            if dist_norm < adm_dist_threshold:
                prox = (adm_dist_threshold - dist_norm) / max(1e-6, adm_dist_threshold)
                adm_impulse = prox * prox
            else:
                adm_impulse = 0.0

            # 2. Rejection impulse from very different current opinion
            if dist_norm > rej_dist_threshold:
                div = (dist_norm - rej_dist_threshold) / rej_span
                if div > 1.0:
                    div = 1.0
                rej_impulse = math.pow(div, 1.35)
            else:
                rej_impulse = 0.0

            if initialize_mode:
                adm_val = adm_impulse
                rej_val = rej_impulse
            else:
                # Extra decay of admiration if current opinion becomes very distant
                eff_adm_decay = adm_decay_rate * (1.5 if dist_norm > rej_dist_threshold else 1.0)
                # Extra decay of rejection if current opinion becomes very close
                eff_rej_decay = rej_decay_rate * (1.5 if dist_norm < adm_dist_threshold else 1.0)

                adm_val = (1.0 - eff_adm_decay) * admiration[i, k] + adm_gain_rate * adm_impulse
                rej_val = (1.0 - eff_rej_decay) * rejection[i, k] + rej_gain_rate * rej_impulse

            if adm_val < 0.0:
                adm_val = 0.0
            elif adm_val > 1.0:
                adm_val = 1.0

            if rej_val < 0.0:
                rej_val = 0.0
            elif rej_val > 1.0:
                rej_val = 1.0

            admiration[i, k] = adm_val
            rejection[i, k] = rej_val


@njit(parallel=True, fastmath=True)
def evaluate_leg_admiration_rejection_votes_jit(
    attitudes: np.ndarray,
    candidate_profiles: np.ndarray,
    admiration: np.ndarray,
    rejection: np.ndarray,
    active_cands: np.ndarray,
    eval_noise: np.ndarray,
    use_dynamic_adm_rej: bool,
    adm_dist_threshold: float,
    rej_dist_threshold: float,
    turnout_min_adm: float,
    clear_margin_threshold: float,
    clear_adm_threshold: float,
    strong_rej_threshold: float,
    out_choices: np.ndarray,
    out_vote_reason: np.ndarray,
) -> None:
    """Evaluate one leg of the election over `active_cands` (4 candidates in Leg 1, 2 in Leg 2).

    Voting outcomes stored in `out_choices[i]`:
      - `-1`: Not going to vote (Abstention) — low admiring in general across candidates on the ballot.
      - `-2`: Go to vote, but vote Blank — no clear winner AND no strong rejection.
      - `k in active_cands`: Valid vote for candidate `k`.

    Vote reason stored in `out_vote_reason[i]`:
      - `0`: Non-valid vote (-1 abstention or -2 blank)
      - `1`: Positive admiration-driven vote
      - `2`: Negative rejection-driven (tactical anti-rejection) vote
    """
    n_agents, n_dims = attitudes.shape
    n_active = active_cands.shape[0]
    inv_sqrt_d = 1.0 / math.sqrt(float(n_dims))
    rej_span = max(0.25, 1.35 - rej_dist_threshold)

    for i in prange(n_agents):
        max_adm = 0.0
        sum_adm = 0.0
        max_rej = 0.0
        min_rej = 1.0

        best_cand = -1
        best_score = -1e12
        best_adm = 0.0
        best_rej = 1.0
        second_score = -1e12

        for idx in range(n_active):
            k = active_cands[idx]
            dist_sq = 0.0
            for d in range(n_dims):
                diff = attitudes[i, d] - candidate_profiles[k, d]
                dist_sq += diff * diff
            dist_norm = math.sqrt(dist_sq) * inv_sqrt_d

            # Instantaneous proximity admiration & divergence rejection from attitude vector
            if dist_norm < adm_dist_threshold:
                prox = (adm_dist_threshold - dist_norm) / max(1e-6, adm_dist_threshold)
                inst_adm = prox * prox
            else:
                inst_adm = 0.0

            if dist_norm > rej_dist_threshold:
                div = (dist_norm - rej_dist_threshold) / rej_span
                if div > 1.0:
                    div = 1.0
                inst_rej = math.pow(div, 1.35)
            else:
                inst_rej = 0.0

            if use_dynamic_adm_rej:
                eff_adm = 0.65 * admiration[i, k] + 0.35 * inst_adm
                eff_rej = 0.65 * rejection[i, k] + 0.35 * inst_rej
            else:
                eff_adm = inst_adm
                eff_rej = inst_rej

            if eff_adm > max_adm:
                max_adm = eff_adm
            sum_adm += eff_adm
            if eff_rej > max_rej:
                max_rej = eff_rej
            if eff_rej < min_rej:
                min_rej = eff_rej

            score = 1.25 * eff_adm - 1.15 * eff_rej - 0.50 * dist_norm + eval_noise[i, k]

            if score > best_score:
                second_score = best_score
                best_score = score
                best_cand = k
                best_adm = eff_adm
                best_rej = eff_rej
            elif score > second_score:
                second_score = score

        margin = best_score - second_score

        # 1. Not going to vote (Abstention = -1): Low admiring in general across the ballot
        #    (unless extreme rejection against a viable candidate mobilizes tactical turnout,
        #     provided at least one candidate is not heavily rejected)
        if (max_adm < turnout_min_adm and (max_rej < strong_rej_threshold + 0.14 or min_rej > 0.60)) or (
            min_rej > 0.68 and max_adm < clear_adm_threshold
        ):
            out_choices[i] = -1
            out_vote_reason[i] = 0
            continue

        # 2. Go to vote, but vote Blank (-2): No clear winner AND no strong rejection
        no_clear_winner = (margin < clear_margin_threshold) or (best_adm < clear_adm_threshold)
        no_strong_rejection = max_rej < strong_rej_threshold

        if no_clear_winner and no_strong_rejection:
            out_choices[i] = -2
            out_vote_reason[i] = 0
            continue

        # Also if the best available candidate is itself strongly rejected (best_rej >= strong_rej_threshold)
        # and the voter has low admiration for it, they cast a blank/null protest ballot
        if best_rej >= strong_rej_threshold and best_adm < clear_adm_threshold:
            out_choices[i] = -2
            out_vote_reason[i] = 0
            continue

        # 3. Valid candidate vote for `best_cand`
        out_choices[i] = best_cand
        if best_adm >= clear_adm_threshold and margin >= clear_margin_threshold:
            out_vote_reason[i] = 1  # Positive admiration vote
        else:
            out_vote_reason[i] = 2  # Negative / tactical anti-rejection vote


class ElectoralEngine:
    """Projects categorical voting decisions, abstention rates, blank votes,
    and Two-Leg (4 candidates -> 2 candidates) outcomes based on Current Opinions.
    """

    def __init__(
        self,
        n_candidates: int = 3,
        n_dims: int = 4,
        candidate_profiles: np.ndarray | None = None,
        indifference_threshold: float = -1.65,
        rationality_lambda: float = 4.0,
        noise_scale: float = 0.12,
        two_leg_enabled: bool = False,
        use_admiration_rejection: bool = True,
        adm_dist_threshold: float = 0.56,
        rej_dist_threshold: float = 0.62,
        adm_gain_rate: float = 0.14,
        adm_decay_rate: float = 0.08,
        rej_gain_rate: float = 0.14,
        rej_decay_rate: float = 0.045,
        turnout_min_adm: float = 0.19,
        clear_margin_threshold: float = 0.15,
        clear_adm_threshold: float = 0.34,
        strong_rej_threshold: float = 0.50,
    ) -> None:
        self.n_candidates = int(n_candidates)
        self.n_dims = int(n_dims)
        self.indifference_threshold = float(indifference_threshold)
        self.rationality_lambda = float(rationality_lambda)
        self.noise_scale = float(noise_scale)
        self.two_leg_enabled = bool(two_leg_enabled)
        self.use_admiration_rejection = bool(use_admiration_rejection)

        self.adm_dist_threshold = float(adm_dist_threshold)
        self.rej_dist_threshold = float(rej_dist_threshold)
        self.adm_gain_rate = float(adm_gain_rate)
        self.adm_decay_rate = float(adm_decay_rate)
        self.rej_gain_rate = float(rej_gain_rate)
        self.rej_decay_rate = float(rej_decay_rate)
        self.turnout_min_adm = float(turnout_min_adm)
        self.clear_margin_threshold = float(clear_margin_threshold)
        self.clear_adm_threshold = float(clear_adm_threshold)
        self.strong_rej_threshold = float(strong_rej_threshold)
        self._adm_rej_initialized = False

        if candidate_profiles is None:
            self.candidate_profiles = self.default_candidate_profiles(self.n_candidates, self.n_dims)
        else:
            self.candidate_profiles = np.ascontiguousarray(candidate_profiles, dtype=np.float64)

    @staticmethod
    def default_candidate_profiles(n_candidates: int, n_dims: int) -> np.ndarray:
        """Create distinct ideological candidate centers of mass theta_k in [-1, 1]^d.

        For K = 2 candidates (Brazilian Presidential Race - Janones Diagnosis):
          - Cand 0: Lula (Center-Left: -0.25, -0.20, -0.35, -0.15)
          - Cand 1: Flavio Bolsonaro (Extreme Right: +0.75, +0.85, +0.70, +0.80)

        For K = 4 candidates (First Leg), places centers across the spectrum:
          - Cand 0: Left / Progressive Pole (-0.75)
          - Cand 1: Center-Left (-0.25)
          - Cand 2: Center-Right (+0.25)
          - Cand 3: Right / Conservative Pole (+0.75)
        """
        if n_candidates == 2:
            profiles = np.zeros((2, n_dims), dtype=np.float64)
            base_lula = np.array([-0.25, -0.20, -0.35, -0.15], dtype=np.float64)
            base_flavio = np.array([0.75, 0.85, 0.70, 0.80], dtype=np.float64)
            for d in range(n_dims):
                profiles[0, d] = base_lula[d % 4]
                profiles[1, d] = base_flavio[d % 4]
            return np.ascontiguousarray(profiles, dtype=np.float64)

        profiles = np.zeros((n_candidates, n_dims), dtype=np.float64)
        anchors = np.linspace(-0.75, 0.75, n_candidates)
        for k in range(n_candidates):
            for d in range(n_dims):
                sign = 1.0 if d % 2 == 0 else -1.0
                scale = 1.0 if d < 2 else 0.85
                profiles[k, d] = np.clip(sign * scale * anchors[k], -1.0, 1.0)
        return np.ascontiguousarray(profiles, dtype=np.float64)

    @staticmethod
    def _top_two_candidates(counts: np.ndarray) -> tuple[int, int]:
        """Return (first_place_idx, second_place_idx) from candidate vote counts."""
        k = len(counts)
        if k < 2:
            return 0, 0
        order = np.argsort(-counts, kind="mergesort")
        return int(order[0]), int(order[1])

    def evaluate_election(
        self,
        agent_state: AgentStateEngine,
        rng: np.random.Generator,
    ) -> dict:
        """Compute individual votes, popular vote shares, abstention rate, district outcomes,
        and (when `use_admiration_rejection` or `two_leg_enabled` or `n_candidates >= 2`) full
        Admiration/Rejection, Abstention/Blank/Valid election dynamics based on Current Opinions vs. Convictions.
        """
        n = agent_state.n_agents
        k = self.n_candidates

        if not self.two_leg_enabled and not self.use_admiration_rejection:
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
            discrepancy = 0.5 * float(np.sum(np.abs(popular_shares - district_seat_shares)))

            return {
                "abstentions": abstentions,
                "abstention_rate": abstention_rate,
                "blank_votes": 0,
                "blank_rate": 0.0,
                "popular_counts": popular_counts,
                "popular_shares": popular_shares,
                "popular_winner": popular_winner,
                "district_seats": district_seats,
                "district_seat_shares": district_seat_shares,
                "district_winners": district_winners,
                "district_winner": district_winner,
                "electoral_discrepancy": discrepancy,
                "winner_mismatch": bool(popular_winner != district_winner and popular_winner >= 0),
                "two_leg_enabled": False,
            }

        # =====================================================================
        # TWO-LEG ELECTION MODEL (First Leg: K=4 Candidates -> Second Leg: 2 Candidates)
        # Driven by CURRENT OPINION x_i(t) + Independent Admiration & Rejection
        # =====================================================================
        if agent_state.candidate_admiration is None or agent_state.candidate_admiration.shape != (n, k):
            agent_state.candidate_admiration = np.zeros((n, k), dtype=np.float64)
        if agent_state.candidate_rejection is None or agent_state.candidate_rejection.shape != (n, k):
            agent_state.candidate_rejection = np.zeros((n, k), dtype=np.float64)
        if agent_state.vote_choice_r2 is None or agent_state.vote_choice_r2.shape != (n,):
            agent_state.vote_choice_r2 = np.full(n, -1, dtype=np.int32)

        step_candidate_admiration_rejection_jit(
            opinions=agent_state.opinions,
            candidate_profiles=self.candidate_profiles,
            admiration=agent_state.candidate_admiration,
            rejection=agent_state.candidate_rejection,
            adm_dist_threshold=self.adm_dist_threshold,
            rej_dist_threshold=self.rej_dist_threshold,
            adm_gain_rate=self.adm_gain_rate,
            adm_decay_rate=self.adm_decay_rate,
            rej_gain_rate=self.rej_gain_rate,
            rej_decay_rate=self.rej_decay_rate,
            initialize_mode=not self._adm_rej_initialized,
        )
        self._adm_rej_initialized = True

        eval_noise = np.ascontiguousarray(
            rng.normal(0.0, 0.03, size=(n, k)), dtype=np.float64
        )
        zero_noise = np.zeros((n, k), dtype=np.float64)

        all_cands = np.arange(k, dtype=np.int32)
        r1_reasons = np.zeros(n, dtype=np.int8)

        # 1. FIRST LEG (All K=4 candidates) based on CURRENT OPINION x_i(t)
        evaluate_leg_admiration_rejection_votes_jit(
            attitudes=agent_state.opinions,
            candidate_profiles=self.candidate_profiles,
            admiration=agent_state.candidate_admiration,
            rejection=agent_state.candidate_rejection,
            active_cands=all_cands,
            eval_noise=eval_noise,
            use_dynamic_adm_rej=True,
            adm_dist_threshold=self.adm_dist_threshold,
            rej_dist_threshold=self.rej_dist_threshold,
            turnout_min_adm=self.turnout_min_adm,
            clear_margin_threshold=self.clear_margin_threshold,
            clear_adm_threshold=self.clear_adm_threshold,
            strong_rej_threshold=self.strong_rej_threshold,
            out_choices=agent_state.vote_choice,
            out_vote_reason=r1_reasons,
        )

        r1_choices = agent_state.vote_choice
        r1_abstentions = int(np.sum(r1_choices == -1))
        r1_blanks = int(np.sum(r1_choices == -2))
        r1_abstention_rate = float(r1_abstentions) / float(n)
        r1_blank_rate = float(r1_blanks) / float(n)
        r1_valid_mask = r1_choices >= 0
        r1_valid_votes = r1_choices[r1_valid_mask]
        n_r1_valid = len(r1_valid_votes)
        r1_valid_rate = float(n_r1_valid) / float(n)

        if n_r1_valid > 0:
            r1_counts = np.bincount(r1_valid_votes, minlength=k).astype(np.int64)
            r1_shares = r1_counts / float(n_r1_valid)
        else:
            r1_counts = np.zeros(k, dtype=np.int64)
            r1_shares = np.zeros(k, dtype=np.float64)

        r1_winner, r1_runner_up = self._top_two_candidates(r1_counts)
        r1_adm_driven_share = float(np.sum(r1_reasons == 1)) / float(max(1, n_r1_valid))
        r1_rej_driven_share = float(np.sum(r1_reasons == 2)) / float(max(1, n_r1_valid))

        # Populate vote_probs for compatibility with downstream consumers
        agent_state.vote_probs.fill(0.0)
        for c_idx in range(k):
            agent_state.vote_probs[r1_choices == c_idx, c_idx] = 1.0
        agent_state.vote_probs[r1_choices < 0, k] = 1.0

        # 2. SECOND LEG (Top 2 candidates from First Leg: r1_winner vs r1_runner_up)
        runoff_cands = np.array([r1_winner, r1_runner_up], dtype=np.int32)
        r2_reasons = np.zeros(n, dtype=np.int8)

        evaluate_leg_admiration_rejection_votes_jit(
            attitudes=agent_state.opinions,
            candidate_profiles=self.candidate_profiles,
            admiration=agent_state.candidate_admiration,
            rejection=agent_state.candidate_rejection,
            active_cands=runoff_cands,
            eval_noise=eval_noise,
            use_dynamic_adm_rej=True,
            adm_dist_threshold=self.adm_dist_threshold,
            rej_dist_threshold=self.rej_dist_threshold,
            turnout_min_adm=self.turnout_min_adm,
            clear_margin_threshold=self.clear_margin_threshold * 0.85,
            clear_adm_threshold=self.clear_adm_threshold * 0.92,
            strong_rej_threshold=self.strong_rej_threshold * 0.92,
            out_choices=agent_state.vote_choice_r2,
            out_vote_reason=r2_reasons,
        )

        r2_choices = agent_state.vote_choice_r2
        r2_abstentions = int(np.sum(r2_choices == -1))
        r2_blanks = int(np.sum(r2_choices == -2))
        r2_abstention_rate = float(r2_abstentions) / float(n)
        r2_blank_rate = float(r2_blanks) / float(n)
        r2_valid_mask = r2_choices >= 0
        r2_valid_votes = r2_choices[r2_valid_mask]
        n_r2_valid = len(r2_valid_votes)
        r2_valid_rate = float(n_r2_valid) / float(n)

        if n_r2_valid > 0:
            r2_counts_all = np.bincount(r2_valid_votes, minlength=k).astype(np.int64)
            r2_shares_all = r2_counts_all / float(n_r2_valid)
        else:
            r2_counts_all = np.zeros(k, dtype=np.int64)
            r2_shares_all = np.zeros(k, dtype=np.float64)

        votes_cand_a = int(r2_counts_all[r1_winner])
        votes_cand_b = int(r2_counts_all[r1_runner_up])
        r2_winner = r1_winner if votes_cand_a >= votes_cand_b else r1_runner_up
        r2_share_cand_a = float(r2_shares_all[r1_winner])
        r2_share_cand_b = float(r2_shares_all[r1_runner_up])
        r2_come_from_behind = 1.0 if (r2_winner == r1_runner_up and r1_runner_up != r1_winner) else 0.0
        r2_adm_driven_share = float(np.sum(r2_reasons == 1)) / float(max(1, n_r2_valid))
        r2_rej_driven_share = float(np.sum(r2_reasons == 2)) / float(max(1, n_r2_valid))

        # 3. COUNTERFACTUAL CONVICTION ELECTION (Leg 1 & Leg 2 evaluated on innate_beliefs s_i(t))
        conv_r1_choices = np.empty(n, dtype=np.int32)
        conv_r1_reasons = np.empty(n, dtype=np.int8)
        evaluate_leg_admiration_rejection_votes_jit(
            attitudes=agent_state.innate_beliefs,
            candidate_profiles=self.candidate_profiles,
            admiration=agent_state.candidate_admiration,
            rejection=agent_state.candidate_rejection,
            active_cands=all_cands,
            eval_noise=zero_noise,
            use_dynamic_adm_rej=False,
            adm_dist_threshold=self.adm_dist_threshold,
            rej_dist_threshold=self.rej_dist_threshold,
            turnout_min_adm=self.turnout_min_adm,
            clear_margin_threshold=self.clear_margin_threshold,
            clear_adm_threshold=self.clear_adm_threshold,
            strong_rej_threshold=self.strong_rej_threshold,
            out_choices=conv_r1_choices,
            out_vote_reason=conv_r1_reasons,
        )
        conv_r1_valid = conv_r1_choices[conv_r1_choices >= 0]
        if len(conv_r1_valid) > 0:
            conv_r1_counts = np.bincount(conv_r1_valid, minlength=k).astype(np.int64)
            conv_r1_shares = conv_r1_counts / float(len(conv_r1_valid))
        else:
            conv_r1_counts = np.zeros(k, dtype=np.int64)
            conv_r1_shares = np.zeros(k, dtype=np.float64)
        conv_r1_winner, conv_r1_runner_up = self._top_two_candidates(conv_r1_counts)

        conv_runoff_cands = np.array([conv_r1_winner, conv_r1_runner_up], dtype=np.int32)
        conv_r2_choices = np.empty(n, dtype=np.int32)
        conv_r2_reasons = np.empty(n, dtype=np.int8)
        evaluate_leg_admiration_rejection_votes_jit(
            attitudes=agent_state.innate_beliefs,
            candidate_profiles=self.candidate_profiles,
            admiration=agent_state.candidate_admiration,
            rejection=agent_state.candidate_rejection,
            active_cands=conv_runoff_cands,
            eval_noise=zero_noise,
            use_dynamic_adm_rej=False,
            adm_dist_threshold=self.adm_dist_threshold,
            rej_dist_threshold=self.rej_dist_threshold,
            turnout_min_adm=self.turnout_min_adm,
            clear_margin_threshold=self.clear_margin_threshold * 0.85,
            clear_adm_threshold=self.clear_adm_threshold * 0.92,
            strong_rej_threshold=self.strong_rej_threshold * 0.92,
            out_choices=conv_r2_choices,
            out_vote_reason=conv_r2_reasons,
        )
        conv_r2_valid = conv_r2_choices[conv_r2_choices >= 0]
        if len(conv_r2_valid) > 0:
            conv_r2_counts = np.bincount(conv_r2_valid, minlength=k).astype(np.int64)
            conv_r2_shares = conv_r2_counts / float(len(conv_r2_valid))
        else:
            conv_r2_counts = np.zeros(k, dtype=np.int64)
            conv_r2_shares = np.zeros(k, dtype=np.float64)
        conv_r2_winner = (
            conv_r1_winner
            if conv_r2_counts[conv_r1_winner] >= conv_r2_counts[conv_r1_runner_up]
            else conv_r1_runner_up
        )

        # Also evaluate how citizens would vote on the ACTUAL Leg-2 ballot (runoff_cands) if using convictions
        conv_same_r2_choices = np.empty(n, dtype=np.int32)
        evaluate_leg_admiration_rejection_votes_jit(
            attitudes=agent_state.innate_beliefs,
            candidate_profiles=self.candidate_profiles,
            admiration=agent_state.candidate_admiration,
            rejection=agent_state.candidate_rejection,
            active_cands=runoff_cands,
            eval_noise=zero_noise,
            use_dynamic_adm_rej=False,
            adm_dist_threshold=self.adm_dist_threshold,
            rej_dist_threshold=self.rej_dist_threshold,
            turnout_min_adm=self.turnout_min_adm,
            clear_margin_threshold=self.clear_margin_threshold * 0.85,
            clear_adm_threshold=self.clear_adm_threshold * 0.92,
            strong_rej_threshold=self.strong_rej_threshold * 0.92,
            out_choices=conv_same_r2_choices,
            out_vote_reason=conv_r2_reasons,
        )

        r1_swing_voter_fraction = float(np.mean(r1_choices != conv_r1_choices))
        r2_swing_voter_fraction = float(np.mean(r2_choices != conv_same_r2_choices))
        r1_flipped_by_opinion = 1.0 if r1_winner != conv_r1_winner else 0.0
        r1_runoff_pair_flipped = (
            1.0
            if set((r1_winner, r1_runner_up)) != set((conv_r1_winner, conv_r1_runner_up))
            else 0.0
        )
        r2_flipped_by_opinion = 1.0 if r2_winner != conv_r2_winner else 0.0

        # District-level aggregation (Leg 1)
        n_districts = agent_state.n_districts
        district_seats = np.zeros(k, dtype=np.int64)
        district_winners = np.full(n_districts, -1, dtype=np.int32)

        for d_id in range(n_districts):
            d_mask = (agent_state.district == d_id) & r1_valid_mask
            d_votes = r1_choices[d_mask]
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
        discrepancy = 0.5 * float(np.sum(np.abs(r1_shares - district_seat_shares)))

        adm_means = np.mean(agent_state.candidate_admiration, axis=0)
        rej_means = np.mean(agent_state.candidate_rejection, axis=0)
        max_adm_mean = float(np.mean(np.max(agent_state.candidate_admiration, axis=1)))
        max_rej_mean = float(np.mean(np.max(agent_state.candidate_rejection, axis=1)))

        return {
            "abstentions": r1_abstentions,
            "abstention_rate": r1_abstention_rate,
            "blank_votes": r1_blanks,
            "blank_rate": r1_blank_rate,
            "valid_vote_rate": r1_valid_rate,
            "popular_counts": r1_counts,
            "popular_shares": r1_shares,
            "popular_winner": r1_winner,
            "r1_runner_up": r1_runner_up,
            "r1_adm_driven_share": r1_adm_driven_share,
            "r1_rej_driven_share": r1_rej_driven_share,
            "r2_cand_a": r1_winner,
            "r2_cand_b": r1_runner_up,
            "r2_abstentions": r2_abstentions,
            "r2_abstention_rate": r2_abstention_rate,
            "r2_blank_votes": r2_blanks,
            "r2_blank_rate": r2_blank_rate,
            "r2_valid_vote_rate": r2_valid_rate,
            "r2_counts_all": r2_counts_all,
            "r2_shares_all": r2_shares_all,
            "r2_share_cand_a": r2_share_cand_a,
            "r2_share_cand_b": r2_share_cand_b,
            "r2_winner": r2_winner,
            "r2_come_from_behind": r2_come_from_behind,
            "r2_adm_driven_share": r2_adm_driven_share,
            "r2_rej_driven_share": r2_rej_driven_share,
            "conv_r1_winner": conv_r1_winner,
            "conv_r1_runner_up": conv_r1_runner_up,
            "conv_r1_shares": conv_r1_shares,
            "conv_r2_winner": conv_r2_winner,
            "conv_r2_shares": conv_r2_shares,
            "r1_swing_voter_fraction": r1_swing_voter_fraction,
            "r2_swing_voter_fraction": r2_swing_voter_fraction,
            "r1_flipped_by_opinion": r1_flipped_by_opinion,
            "r1_runoff_pair_flipped": r1_runoff_pair_flipped,
            "r2_flipped_by_opinion": r2_flipped_by_opinion,
            "adm_means": adm_means,
            "rej_means": rej_means,
            "max_adm_mean": max_adm_mean,
            "max_rej_mean": max_rej_mean,
            "district_seats": district_seats,
            "district_seat_shares": district_seat_shares,
            "district_winners": district_winners,
            "district_winner": district_winner,
            "electoral_discrepancy": discrepancy,
            "winner_mismatch": bool(r1_winner != district_winner and r1_winner >= 0),
            "two_leg_enabled": True,
        }

    def step_rejection_news_impact(
        self,
        agent_state: AgentStateEngine,
        target_cand: int,
        rejection_pull: float = 0.20,
        exposure_mask: np.ndarray | None = None,
    ) -> None:
        """Apply negative campaigning / rejection news impact on candidate `target_cand`.

        Grounded in electoral psychology (negativity bias & affective polarization):
        Exposure to rejection news elevates voter aversion R_{i, target_cand} and
        dampens likelihood / admiration A_{i, target_cand}.
        """
        n = agent_state.n_agents
        if agent_state.candidate_rejection is None or agent_state.candidate_rejection.shape[1] <= target_cand:
            return
        if exposure_mask is None:
            exposure_mask = np.ones(n, dtype=bool)

        # Elevate aversion / rejection toward the attacked candidate
        cur_rej = agent_state.candidate_rejection[:, target_cand].copy()
        cur_rej[exposure_mask] = np.clip(cur_rej[exposure_mask] + rejection_pull, 0.0, 1.0)
        agent_state.candidate_rejection[:, target_cand] = cur_rej

        # Dampen admiration for the attacked candidate
        if agent_state.candidate_admiration is not None:
            cur_adm = agent_state.candidate_admiration[:, target_cand].copy()
            cur_adm[exposure_mask] = np.clip(cur_adm[exposure_mask] * (1.0 - 0.45 * rejection_pull), 0.0, 1.0)
            agent_state.candidate_admiration[:, target_cand] = cur_adm
