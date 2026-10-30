"""NewsEntityPool & TraditionalMediaEcosystem: Fact-driven traditional media (TV, Radio, Press)
and social media content pools.

Empirical Brazilian Calibration & Published Theoretical References:
-------------------------------------------------------------------
1. Empirical Brazilian Media Access by Age and Socioeconomic Status (SES):
   - Television (TV Aberta / Paga):
     * IBGE, PNAD Continua TIC 2022/2023 (Acesso a Internet e a Televisao, 2023): TV present in 94.4%
       of Brazilian households; primary news medium for lower/middle SES (Classes C/D/E) and older cohorts.
     * Kantar IBOPE Media, Inside Video 2023/2024: Linear TV reaches 78.9% of the Brazilian population
       weekly (~56% daily), averaging 5h17m/day, with a steep positive age gradient (60+ vs. 16-24).
   - Radio (Radio AM/FM):
     * Kantar IBOPE Media, Inside Audio / Inside Radio 2023 & ABERT (2023): Reaches 80%-83% of residents
       in major Brazilian metropolitan areas (~3h55m/day), peaking among working-age commuters (30-59)
       and older adults across middle and working socioeconomic strata.
   - Written Press (Jornais Impressos, Revistas e Portais da Imprensa Tradicional):
     * Reuters Institute for the Study of Journalism (Oxford), Digital News Report 2023/2024 (Brazil chapter,
       R. Carro), IVC Brasil (2023), & SECOM Pesquisa Brasileira de Midia (PBM): Print circulation (~7-11%)
       and traditional press portals (~28-34%) are strongly concentrated in higher socioeconomic strata
       (Classes A/B, university-educated, SES >= 0.65) and adults aged 30+.

2. Published Theoretical Models for Fact-Driven News, Editorial Slant, Lag, Cadence, and Decay:
   - Mullainathan, S., & Shleifer, A. (2005). "The Market for News." American Economic Review, 95(4), 1031-1053.
     (Exogenous truth/fact arrivals slanted toward a media outlet's ideological position).
   - Gentzkow, M., & Shapiro, J. M. (2006). "Media Bias and Reputation." Journal of Political Economy, 114(2), 280-316;
     and Gentzkow, M., & Shapiro, J. M. (2010). "What Drives Media Slant?" Econometrica, 78(1), 35-71.
   - DellaVigna, S., & Kaplan, E. (2007). "The Fox News Effect: Media Bias and Voting." QJE, 122(3), 1187-1234.
   - Wu, F., & Huberman, B. A. (2007). "Novelty and collective attention." PNAS, 104(45), 17599-17601.
     (Exponential temporal decay of news novelty and public attention).
   - Leskovec, J., Backstrom, L., & Kleinberg, J. (2009). "Meme-tracking and the dynamics of the news cycle."
     ACM SIGKDD, 497-506. (Publication lag and cadence differences across broadcast, radio, and print press).
   - McCombs, M. E., & Shaw, D. L. (1972). "The Agenda-Setting Function of Mass Media." Public Opinion Quarterly, 36(2), 176-187.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import numpy as np


# Modality constants within each Media Vehicle Group (Conglomerate)
MODALITY_TV = 0
MODALITY_RADIO = 1
MODALITY_PRESS = 2
MODALITY_NAMES = ("tv", "radio", "press")

# Social Media Platform Type Constants (Brazilian Digital Media Ecosystem)
SOCIAL_PLATFORM_VISUAL_INFLUENCER = 0  # Instagram / TikTok (Creator/Influencer Graph + Algorithmic Reels/ForYou)
SOCIAL_PLATFORM_VIDEO_BLOG = 1         # YouTube / X (Twitter) (Commentary, Video Channels, Public Microblogging)
SOCIAL_PLATFORM_SOCIAL_FEED = 2        # Facebook (Social Graph + Algorithmic News Feed)
SOCIAL_PLATFORM_MESSAGING = 3          # WhatsApp / Telegram (Closed Private Groups K <= 256 & Viral Cascades)
SOCIAL_PLATFORM_NAMES = (
    "instagram_tiktok",
    "youtube_x",
    "facebook",
    "whatsapp_telegram",
)


def default_vehicle_group_centers(n_groups: int, n_dims: int) -> np.ndarray:
    """Return canonical ideological centers of mass m_g in [-1, 1]^d for M in {0, 1, 2, 3, 4} media conglomerates.

    Designed so that:
    - M = 1: Single hegemonic conglomerate at (+0.55, +0.55, ..., +0.55), testing unidirectional center-of-mass pull.
    - M = 2: Two competing conglomerates with asymmetric positions (+0.60 and -0.25), net CM = +0.175, testing duopoly pull vs cleavage.
    - M = 3: Tripolar ecosystem (Conservative +0.65, Centrist-Institutional +0.15, Progressive -0.50), net CM = +0.10.
    - M = 4: Four-quadrant pluralistic press ecosystem spanning economic and moral axes.
    """
    if n_groups <= 0:
        return np.zeros((0, n_dims), dtype=np.float64)

    centers = np.zeros((n_groups, n_dims), dtype=np.float64)
    if n_groups == 1:
        # 1 Vehicle Group: Hegemonic conglomerate with clear center of mass
        centers[0, :] = 0.55
    elif n_groups == 2:
        # 2 Vehicle Groups: Dominant center-right conglomerate (+0.60) vs center-left challenger (-0.25)
        centers[0, :] = 0.60
        centers[1, :] = -0.25
    elif n_groups == 3:
        # 3 Vehicle Groups: Right (+0.65), Centrist-Establishment (+0.15), Left (-0.50)
        centers[0, :] = 0.65
        centers[1, :] = 0.15
        centers[2, :] = -0.50
    elif n_groups == 4:
        # 4 Vehicle Groups: Multi-axis ideological quadrants with net positive center-of-mass shift (+0.08)
        base_4 = np.array(
            [
                [0.70, 0.65, 0.55, 0.65],    # Group 1: Conservative / Market-liberal
                [0.50, -0.40, 0.35, 0.20],   # Group 2: Liberal-Centrist / Reformist
                [-0.35, 0.50, -0.25, 0.45],  # Group 3: Popular-Traditional / Statist
                [-0.55, -0.50, -0.45, -0.55],# Group 4: Progressive / Left
            ],
            dtype=np.float64,
        )
        for g in range(4):
            for d in range(n_dims):
                centers[g, d] = base_4[g, d % 4]
    else:
        anchors = np.linspace(-0.65, 0.70, n_groups)
        for g in range(n_groups):
            centers[g, :] = anchors[g]

    return np.ascontiguousarray(centers, dtype=np.float64)


@dataclass
class TraditionalMediaEcosystem:
    """Models M traditional media vehicle groups (conglomerates), each operating 3 vehicles:
    1. Broadcast TV (`tv`, modality 0): lag = 1 step, high credibility (0.75), medium decay (0.30).
    2. Radio (`radio`, modality 1): lag = 0 steps (live bulletin), medium credibility (0.62), fast decay (0.45).
    3. Written Press (`press`, modality 2): lag = 2 steps (print/editorial cycle), period = 2 steps,
       high credibility (0.82), slow decay (0.18).

    Tendentiousness Level (`group_tendentiousness`, tau_g in [0, 1]):
    - Controls how strongly vehicle group g pulls exogenous facts f_k toward its own editorial
      center of mass m_g vs. reporting the fact close to its raw opinion impact f_k
      (Mullainathan & Shleifer, 2005; Gentzkow & Shapiro, 2006, 2010; Puglisi & Snyder, 2015):
          y_{k,g,v} = (1 - eta_{g,v}) * f_k + eta_{g,v} * m_g + xi
      where eta_{g,v} = clip(tau_g * modality_slant_factor_v, 0.0, 1.0).
    - Low tendentiousness (tau_g -> 0, e.g. 0.10): notices stay close to the raw fact f_k, and facts
      are covered with low ideological selectivity.
    - High tendentiousness (tau_g -> 1, e.g. 0.90): notices are spun tightly around the media outlet's
      center of mass m_g, and gatekeeping strongly favors facts aligned with m_g.
    """

    n_groups: int
    n_dims: int
    group_centers: np.ndarray            # float64, shape (M, d)
    group_tendentiousness: np.ndarray    # float64, shape (M,), tau_g in [0, 1]
    ecosystem_center: np.ndarray         # float64, shape (d,)

    # Flattened vehicle arrays of length V = 3 * M
    n_vehicles: int
    vehicle_group_id: np.ndarray         # int32, shape (V,)
    vehicle_modality: np.ndarray         # int32, shape (V,) in {0=TV, 1=Radio, 2=Press}
    vehicle_centers: np.ndarray          # float64, shape (V, d)
    vehicle_tendentiousness: np.ndarray  # float64, shape (V,), tau_g in [0, 1]
    vehicle_lag: np.ndarray              # int32, shape (V,)
    vehicle_period: np.ndarray           # int32, shape (V,)
    vehicle_pub_prob: np.ndarray         # float64, shape (V,)
    vehicle_slant_pull: np.ndarray       # float64, shape (V,), effective eta_{g,v} in [0, 1]
    vehicle_decay_rate: np.ndarray       # float64, shape (V,)
    vehicle_credibility: np.ndarray      # float64, shape (V,)

    # Dynamic state of active (decaying) notices per vehicle
    active_weighted_sum: np.ndarray      # float64, shape (V, d): sum_k w_k(t) * y_k
    active_intensity: np.ndarray         # float64, shape (V,): W_v(t) = sum_k w_k(t)
    active_framing: np.ndarray           # float64, shape (V, d): B_v(t) = active_weighted_sum / active_intensity

    # Process hyperparameters
    fact_rate: float = 1.5               # Mean number of exogenous facts occurring per step
    gatekeeping_sigma: float = 0.85      # Selectivity bandwidth for favorable facts
    framing_noise_std: float = 0.05      # Editorial framing noise std

    # Queue of delayed notices waiting for publication step: list of (pub_step, vehicle_idx, salience, framing_vec, is_dictated, longevity_mult)
    pending_notices: list[tuple] = field(default_factory=list)
    last_step_facts_count: int = 0
    last_step_published_count: int = 0
    dictated_active_intensity: np.ndarray | None = None
    active_longevity_mult: np.ndarray | None = None
    last_dictated_trad_share: float = 0.0
    last_mean_fact_longevity: float = 1.0
    last_mean_fact_relevance: float = 0.65

    @classmethod
    def initialize(
        cls,
        n_groups: int,
        n_dims: int,
        group_centers: np.ndarray | list | None = None,
        group_tendentiousness: float | list[float] | np.ndarray | None = None,
        fact_rate: float = 1.5,
        gatekeeping_sigma: float = 0.85,
        framing_noise_std: float = 0.05,
        seed: int | np.random.Generator | None = None,
    ) -> "TraditionalMediaEcosystem":
        """Build M vehicle groups (each with TV, Radio, and Press) with configurable tendentiousness."""
        n_groups = max(0, int(n_groups))
        if group_centers is not None:
            centers = np.ascontiguousarray(np.asarray(group_centers, dtype=np.float64))
            if centers.ndim == 1:
                centers = centers.reshape(1, -1)
        else:
            centers = default_vehicle_group_centers(n_groups, n_dims)

        # Resolve per-group tendentiousness tau_g in [0, 1] (default = 0.65)
        if n_groups == 0:
            tend_arr = np.zeros(0, dtype=np.float64)
        elif group_tendentiousness is None:
            tend_arr = np.full(n_groups, 0.65, dtype=np.float64)
        elif np.ndim(group_tendentiousness) == 0:
            tend_arr = np.full(n_groups, float(group_tendentiousness), dtype=np.float64)
        else:
            raw_t = np.asarray(group_tendentiousness, dtype=np.float64).ravel()
            if len(raw_t) < n_groups:
                padded = np.full(n_groups, float(raw_t[-1]), dtype=np.float64)
                padded[: len(raw_t)] = raw_t
                tend_arr = padded
            else:
                tend_arr = raw_t[:n_groups].copy()
        tend_arr = np.ascontiguousarray(np.clip(tend_arr, 0.0, 1.0), dtype=np.float64)

        if n_groups > 0:
            # Tendentiousness-weighted and unweighted ecosystem centers
            w_sum = float(np.sum(tend_arr))
            if w_sum > 1e-8:
                ecosystem_center = np.ascontiguousarray(
                    np.sum(centers * tend_arr[:, None], axis=0) / w_sum, dtype=np.float64
                )
            else:
                ecosystem_center = np.ascontiguousarray(np.mean(centers, axis=0), dtype=np.float64)
        else:
            ecosystem_center = np.zeros(n_dims, dtype=np.float64)

        n_vehicles = 3 * n_groups
        vehicle_group_id = np.zeros(n_vehicles, dtype=np.int32)
        vehicle_modality = np.zeros(n_vehicles, dtype=np.int32)
        vehicle_centers = np.zeros((n_vehicles, n_dims), dtype=np.float64)
        vehicle_tendentiousness = np.zeros(n_vehicles, dtype=np.float64)
        vehicle_lag = np.zeros(n_vehicles, dtype=np.int32)
        vehicle_period = np.ones(n_vehicles, dtype=np.int32)
        vehicle_pub_prob = np.zeros(n_vehicles, dtype=np.float64)
        vehicle_slant_pull = np.zeros(n_vehicles, dtype=np.float64)
        vehicle_decay_rate = np.zeros(n_vehicles, dtype=np.float64)
        vehicle_credibility = np.zeros(n_vehicles, dtype=np.float64)

        # Modality specifications (TV=0, Radio=1, Press=2) based on Leskovec et al. (2009) & Wu & Huberman (2007)
        # slant_mult scales the group's tendentiousness tau_g: Radio (0.90), TV (1.00), Press (1.10)
        mod_specs = (
            (MODALITY_TV, 1, 1, 0.85, 1.00, 0.30, 0.75),
            (MODALITY_RADIO, 0, 1, 0.92, 0.90, 0.45, 0.62),
            (MODALITY_PRESS, 2, 2, 0.75, 1.10, 0.18, 0.82),
        )

        for g in range(n_groups):
            tau_g = float(tend_arr[g])
            for mod_id, lag, period, pub_p, slant_mult, decay, cred in mod_specs:
                v_idx = g * 3 + mod_id
                vehicle_group_id[v_idx] = g
                vehicle_modality[v_idx] = mod_id
                vehicle_centers[v_idx, :] = centers[g, :]
                vehicle_tendentiousness[v_idx] = tau_g
                vehicle_lag[v_idx] = lag
                vehicle_period[v_idx] = period
                vehicle_pub_prob[v_idx] = pub_p
                vehicle_slant_pull[v_idx] = float(np.clip(tau_g * slant_mult, 0.0, 1.0))
                vehicle_decay_rate[v_idx] = decay
                vehicle_credibility[v_idx] = cred

        # Initialize active notice state: scaled by tendentiousness tau_g toward center of mass
        active_intensity = np.full(n_vehicles, 0.50, dtype=np.float64)
        active_framing = np.ascontiguousarray(
            vehicle_centers * vehicle_slant_pull[:, None], dtype=np.float64
        )
        active_weighted_sum = np.ascontiguousarray(
            active_framing * active_intensity[:, None], dtype=np.float64
        )

        eco = cls(
            n_groups=n_groups,
            n_dims=n_dims,
            group_centers=centers,
            group_tendentiousness=tend_arr,
            ecosystem_center=ecosystem_center,
            n_vehicles=n_vehicles,
            vehicle_group_id=vehicle_group_id,
            vehicle_modality=vehicle_modality,
            vehicle_centers=np.ascontiguousarray(vehicle_centers, dtype=np.float64),
            vehicle_tendentiousness=np.ascontiguousarray(vehicle_tendentiousness, dtype=np.float64),
            vehicle_lag=vehicle_lag,
            vehicle_period=vehicle_period,
            vehicle_pub_prob=vehicle_pub_prob,
            vehicle_slant_pull=vehicle_slant_pull,
            vehicle_decay_rate=vehicle_decay_rate,
            vehicle_credibility=vehicle_credibility,
            active_weighted_sum=active_weighted_sum,
            active_intensity=active_intensity,
            active_framing=active_framing,
            fact_rate=float(fact_rate),
            gatekeeping_sigma=float(gatekeeping_sigma),
            framing_noise_std=float(framing_noise_std),
            dictated_active_intensity=np.zeros(n_vehicles, dtype=np.float64),
            active_longevity_mult=np.ones(n_vehicles, dtype=np.float64),
        )
        return eco

    def step_facts_and_news_cycle(
        self,
        step_idx: int,
        rng: np.random.Generator,
        dictated_fact_vec: np.ndarray | None = None,
        dictated_framed_vec: np.ndarray | None = None,
        dictated_interest_push: float = 0.0,
        dictated_longevity_mult: float = 1.0,
        dictated_pull: float = 0.0,
    ) -> dict[str, float]:
        """Advance the news cycle by one timestep:
        1. Exponentially decay all active notices for each vehicle v: exp(-delta_v / L_v(t)),
           where L_v(t) >= 1 reflects the longevity of high-relevance or network-pushed facts
           (Downs, 1972; Hilgartner & Bosk, 1988; Wu & Huberman, 2007).
        2. Generate new exogenous facts (Poisson arrival with mean `fact_rate`), each with
           heterogeneous relevance r_k in [0.25, 1.0] and longevity L_k = 0.70 + 1.15 * r_k.
        3. Intermedia Agenda-Setting / Dictation of Facts by Synchronized Social Networks
           (McCombs, 2004; Harder et al., 2017; Gilardi et al., 2022; Marwick & Lewis, 2017):
           When a synchronized influencer network pushes a specific focal fact (`dictated_interest_push > 0`),
           that fact receives a surge in public interest that forces traditional media vehicles (TV, Radio,
           Written Press) to notice and cover it, partially adopts the dictated subject framing, and
           persists longer in the traditional news cycle (`dictated_longevity_mult`).
        4. Publish all notices whose scheduled publication step `t_pub <= step_idx`.
        """
        if self.n_vehicles == 0:
            self.last_step_facts_count = 0
            self.last_step_published_count = 0
            self.last_dictated_trad_share = 0.0
            return {
                "facts_count": 0.0,
                "published_notices": 0.0,
                "dictated_trad_share": 0.0,
                "mean_fact_longevity": 1.0,
                "mean_fact_relevance": 0.65,
            }

        if self.dictated_active_intensity is None or len(self.dictated_active_intensity) != self.n_vehicles:
            self.dictated_active_intensity = np.zeros(self.n_vehicles, dtype=np.float64)
        if self.active_longevity_mult is None or len(self.active_longevity_mult) != self.n_vehicles:
            self.active_longevity_mult = np.ones(self.n_vehicles, dtype=np.float64)

        # 1. Exponential temporal decay modulated by active fact longevity L_v(t) (Wu & Huberman, 2007)
        eff_decay_rate = self.vehicle_decay_rate / np.clip(self.active_longevity_mult, 0.75, 4.5)
        decay_factors = np.exp(-eff_decay_rate)
        self.active_weighted_sum *= decay_factors[:, None]
        self.active_intensity *= decay_factors
        self.dictated_active_intensity *= decay_factors
        # Relax longevity multiplier smoothly toward baseline 1.0
        self.active_longevity_mult = 1.0 + (self.active_longevity_mult - 1.0) * decay_factors

        # 2. Exogenous Fact Arrival Process with Heterogeneous Relevance & Longevity
        n_facts = int(rng.poisson(self.fact_rate)) if self.fact_rate > 0.0 else 0
        self.last_step_facts_count = n_facts
        inv_sqrt_d = 1.0 / math.sqrt(float(self.n_dims))
        step_relevances: list[float] = []
        step_longevities: list[float] = []

        for _ in range(n_facts):
            # Objective fact valence vector f_k in [-1, 1]^d and heterogeneous relevance r_k in [0.25, 1.0]
            fact_vec = rng.uniform(-1.0, 1.0, size=self.n_dims)
            # Some facts are routine/ephemeral while others are highly relevant and last longer
            fact_relevance = float(0.25 + 0.75 * rng.beta(1.6, 2.0))
            fact_longevity = float(0.75 + 1.15 * fact_relevance)
            step_relevances.append(fact_relevance)
            step_longevities.append(fact_longevity)

            for v_idx in range(self.n_vehicles):
                center_v = self.vehicle_centers[v_idx]
                tau_v = self.vehicle_tendentiousness[v_idx]
                dist_f = float(np.linalg.norm(fact_vec - center_v)) * inv_sqrt_d

                # Gatekeeping favorability boosted by intrinsic fact relevance
                raw_fav = math.exp(-dist_f / max(self.gatekeeping_sigma, 1e-6))
                eff_favorability = (1.0 - tau_v) * 0.70 + tau_v * (0.25 + 0.75 * raw_fav)
                eff_favorability = min(1.0, eff_favorability * (0.75 + 0.45 * fact_relevance))
                cov_prob = self.vehicle_pub_prob[v_idx] * eff_favorability

                if rng.random() < cov_prob:
                    eta = self.vehicle_slant_pull[v_idx]
                    noise = rng.normal(0.0, self.framing_noise_std, size=self.n_dims)
                    notice_vec = np.clip((1.0 - eta) * fact_vec + eta * center_v + noise, -1.0, 1.0)
                    notice_salience = fact_relevance * eff_favorability * (0.75 + 0.50 * tau_v)

                    raw_pub_step = step_idx + int(self.vehicle_lag[v_idx])
                    period = int(self.vehicle_period[v_idx])
                    if period > 1 and (raw_pub_step % period != 0):
                        raw_pub_step += period - (raw_pub_step % period)

                    self.pending_notices.append(
                        (raw_pub_step, v_idx, notice_salience, notice_vec, False, fact_longevity)
                    )

        # 2B. Intermedia Agenda-Setting: Network-Pushed Focal Fact Noticiabled by Traditional Media
        if dictated_fact_vec is not None and dictated_interest_push > 0.05:
            push_strength = min(2.0, float(dictated_interest_push))
            eff_dict_relevance = min(2.2, 0.85 + 0.65 * push_strength)
            eff_dict_longevity = max(1.2, float(dictated_longevity_mult))
            step_relevances.append(min(1.0, eff_dict_relevance))
            step_longevities.append(eff_dict_longevity)
            framed_ref = dictated_framed_vec if dictated_framed_vec is not None else dictated_fact_vec

            for v_idx in range(self.n_vehicles):
                center_v = self.vehicle_centers[v_idx]
                tau_v = self.vehicle_tendentiousness[v_idx]
                dist_f = float(np.linalg.norm(dictated_fact_vec - center_v)) * inv_sqrt_d
                raw_fav = math.exp(-dist_f / max(self.gatekeeping_sigma, 1e-6))
                base_fav = (1.0 - tau_v) * 0.70 + tau_v * (0.25 + 0.75 * raw_fav)
                # High social network push in interest forces traditional media to cover the fact
                pushed_fav = min(0.99, base_fav + 0.55 * min(1.0, push_strength))
                cov_prob = min(0.98, self.vehicle_pub_prob[v_idx] * pushed_fav + 0.25 * min(1.0, push_strength))

                if rng.random() < cov_prob:
                    eta = self.vehicle_slant_pull[v_idx]
                    noise = rng.normal(0.0, self.framing_noise_std * 0.75, size=self.n_dims)
                    editorial_vec = (1.0 - eta) * dictated_fact_vec + eta * center_v
                    # Intermedia frame dictation: traditional coverage absorbs part of the synchronized subject angle
                    d_pull = float(np.clip(dictated_pull * (0.65 + 0.35 * min(1.5, push_strength)), 0.0, 0.68))
                    notice_vec = np.clip(
                        (1.0 - d_pull) * editorial_vec + d_pull * framed_ref + noise,
                        -1.0,
                        1.0,
                    )
                    notice_salience = eff_dict_relevance * pushed_fav * (0.95 + 0.45 * min(1.5, push_strength))

                    raw_pub_step = step_idx + int(self.vehicle_lag[v_idx])
                    period = int(self.vehicle_period[v_idx])
                    if period > 1 and (raw_pub_step % period != 0):
                        raw_pub_step += period - (raw_pub_step % period)

                    self.pending_notices.append(
                        (raw_pub_step, v_idx, notice_salience, notice_vec, True, eff_dict_longevity)
                    )

        # 3. Publish all scheduled notices ready at `step_idx`
        remaining_queue: list[tuple] = []
        published_now = 0
        for item in self.pending_notices:
            if len(item) == 4:
                pub_step, v_idx, sal, n_vec = item
                is_dict = False
                longev = 1.0
            else:
                pub_step, v_idx, sal, n_vec, is_dict, longev = item
            if pub_step <= step_idx:
                prev_w = self.active_intensity[v_idx]
                new_w = prev_w + sal
                self.active_weighted_sum[v_idx] += sal * n_vec
                self.active_intensity[v_idx] = new_w
                if is_dict:
                    self.dictated_active_intensity[v_idx] += sal
                if new_w > 1e-8:
                    self.active_longevity_mult[v_idx] = (
                        prev_w * self.active_longevity_mult[v_idx] + sal * longev
                    ) / new_w
                published_now += 1
            else:
                remaining_queue.append((pub_step, v_idx, sal, n_vec, is_dict, longev))
        self.pending_notices = remaining_queue
        self.last_step_published_count = published_now

        # 4. Update normalized active framing B_v(t) and dictated coverage share
        for v_idx in range(self.n_vehicles):
            w = self.active_intensity[v_idx]
            if w > 1e-8:
                self.active_framing[v_idx, :] = np.clip(
                    self.active_weighted_sum[v_idx, :] / w, -1.0, 1.0
                )
            else:
                self.active_framing[v_idx, :] = (
                    self.vehicle_centers[v_idx, :] * self.vehicle_slant_pull[v_idx]
                )

        tot_w = float(np.sum(self.active_intensity))
        if tot_w > 1e-8:
            self.last_dictated_trad_share = float(
                np.clip(np.sum(self.dictated_active_intensity) / tot_w, 0.0, 1.0)
            )
        else:
            self.last_dictated_trad_share = 0.0

        self.last_mean_fact_longevity = float(np.mean(self.active_longevity_mult))
        if len(step_relevances) > 0:
            self.last_mean_fact_relevance = float(np.mean(step_relevances))

        return {
            "facts_count": float(n_facts),
            "published_notices": float(published_now),
            "dictated_trad_share": self.last_dictated_trad_share,
            "mean_fact_longevity": self.last_mean_fact_longevity,
            "mean_fact_relevance": self.last_mean_fact_relevance,
        }


@dataclass
class NewsEntityPool:
    """Contiguous pool of active news/content items, influencer profiles, traditional media vehicle groups,
    and broadcast channels."""

    n_items: int
    n_dims: int
    framing: np.ndarray          # float64, shape (M, d), y_n in [-1, 1]^d
    sensationalism: np.ndarray   # float64, shape (M,), e_n in [0, 1]
    veracity: np.ndarray         # float64, shape (M,), v_n in [0, 1]
    creator_id: np.ndarray       # int32, shape (M,), agent ID of creator
    creator_inf_idx: np.ndarray  # int32, shape (M,), index into influencer_ids (0..n_inf-1)
    platform_type: np.ndarray    # int32, shape (M,), in {0=IG/TikTok, 1=YouTube/X, 2=Facebook, 3=WhatsApp}
    popularity: np.ndarray       # float64, shape (M,)
    like_count: np.ndarray       # int64, shape (M,), cumulative positive concordant engagements
    dislike_count: np.ndarray    # int64, shape (M,), cumulative negative out-group / outrage reactions
    viral_cascades: np.ndarray   # int64, shape (M,)

    # Influencer tendentiousness & follower dynamics state (for Instagram/TikTok & YouTube/X creators)
    influencer_tendentiousness: np.ndarray  # float64, shape (n_inf,), tau_inf in [0, 1] (0=factual, 1=tendentious)
    influencer_is_extreme_imposed: np.ndarray  # uint8, shape (n_inf,), 1 if influencer belongs to extreme_imposed zone
    extreme_imposed_target: np.ndarray      # float64, shape (d,), e_imp in [-1, 1]^d (default +0.85, ..., +0.85)

    # Legacy broadcast channels & new fact-driven TraditionalMediaEcosystem (TV, Radio, Press)
    n_broadcast: int
    broadcast_orientations: np.ndarray  # float64, shape (B, d), m_k in [-1, 1]^d
    broadcast_credibility: np.ndarray   # float64, shape (B,), gamma_k in [0, 1]
    traditional_ecosystem: TraditionalMediaEcosystem | None = None
    extreme_imposed_enabled: bool = False
    extreme_imposed_reach_boost: float = 1.0
    extreme_imposed_popularity_boost: float = 1.0

    # Synchronization Tactics, Influencer Co-Working, & Fact/Subject Dictation State
    sync_tactics_enabled: bool = False
    cowork_enabled: bool = False
    cowork_prob: float = 0.65
    cowork_reach_fraction: float = 0.50
    fact_dictation_enabled: bool = False
    fact_dictation_pull: float = 0.42
    item_is_synchronized: np.ndarray | None = None
    sync_focal_fact_vec: np.ndarray | None = None
    sync_focal_fact_framed: np.ndarray | None = None
    sync_focal_fact_relevance: float = 0.85
    sync_focal_fact_longevity: float = 1.80
    sync_focal_fact_remaining_steps: int = 0
    sync_primary_subject: int = 0
    sync_network_interest_push: float = 0.0

    @classmethod
    def initialize(
        cls,
        n_items: int,
        n_dims: int,
        influencer_ids: np.ndarray,
        influencer_opinions: np.ndarray,
        n_broadcast: int = 3,
        misinfo_fraction: float = 0.25,
        seed: int | np.random.SeedSequence | np.random.Generator = 42,
        n_vehicle_groups: int | None = None,
        vehicle_group_centers: np.ndarray | list | None = None,
        group_tendentiousness: float | list[float] | np.ndarray | None = None,
        influencer_tendentiousness: float | list[float] | np.ndarray | str | None = None,
        fact_rate: float = 1.5,
        gatekeeping_sigma: float = 0.85,
        extreme_imposed_enabled: bool = False,
        extreme_imposed_target: np.ndarray | list[float] | None = None,
        extreme_imposed_Radius: float = 0.52,
        extreme_imposed_fraction: float = 0.25,
        extreme_imposed_reach_boost: float = 3.5,
        extreme_imposed_popularity_boost: float = 4.0,
        sync_tactics_enabled: bool = False,
        cowork_enabled: bool = False,
        cowork_prob: float = 0.65,
        cowork_reach_fraction: float = 0.50,
        fact_dictation_enabled: bool = False,
        fact_dictation_pull: float = 0.42,
        agent_opinions: np.ndarray | None = None,
        agent_innate_beliefs: np.ndarray | None = None,
    ) -> "NewsEntityPool":
        """Create an initial pool of news items, influencer tendentiousness profiles, and TraditionalMediaEcosystem."""
        if isinstance(seed, np.random.Generator):
            rng = seed
        else:
            rng = np.random.default_rng(seed)

        if extreme_imposed_target is not None:
            ext_target = np.ascontiguousarray(np.asarray(extreme_imposed_target, dtype=np.float64).ravel()[:n_dims])
        else:
            ext_target = np.full(n_dims, 0.85, dtype=np.float64)

        n_inf = len(influencer_ids)
        inf_is_extreme = np.zeros(n_inf, dtype=np.uint8)
        if n_inf > 0 and len(influencer_opinions) > 0:
            inv_sqrt_d = 1.0 / math.sqrt(float(n_dims))
            if extreme_imposed_enabled:
                # Deliberately anchor a dedicated fraction of influencers in the extreme_imposed corner
                n_ext = max(1, int(round(n_inf * extreme_imposed_fraction)))
                dists_to_ext = np.linalg.norm(influencer_opinions - ext_target[None, :], axis=1) * inv_sqrt_d
                closest_inf = np.argsort(dists_to_ext)[:n_ext]
                inf_is_extreme[closest_inf] = 1
                # Pull these influencers firmly into the extreme_imposed region so they act as committed vanguard
                for k_idx in closest_inf:
                    pinned = np.clip(
                        0.75 * ext_target + 0.25 * influencer_opinions[k_idx] + rng.normal(0.0, 0.05, size=n_dims),
                        -1.0,
                        1.0,
                    )
                    influencer_opinions[k_idx] = pinned
                    if agent_opinions is not None:
                        agent_opinions[influencer_ids[k_idx]] = pinned
                    if agent_innate_beliefs is not None:
                        agent_innate_beliefs[influencer_ids[k_idx]] = pinned
            else:
                # Even without artificial imposition, identify influencers naturally residing near the extreme_imposed pole
                dists_to_ext = np.linalg.norm(influencer_opinions - ext_target[None, :], axis=1) * inv_sqrt_d
                nat_mask = dists_to_ext <= extreme_imposed_Radius
                if np.any(nat_mask):
                    inf_is_extreme[nat_mask] = 1
                else:
                    n_ext = max(1, int(round(n_inf * 0.20)))
                    inf_is_extreme[np.argsort(dists_to_ext)[:n_ext]] = 1

        # Resolve per-influencer tendentiousness tau_inf in [0, 1]:
        # - Low tau_inf (e.g. 0.15): Factual / journalistic digital creators (high veracity, moderate framing)
        # - High tau_inf (e.g. 0.90): Tendentious / hyper-partisan firebrand influencers (Bail et al., 2018; Recuero, 2020)
        # - In the extreme_imposed scenario, influencers of the extreme_imposed region are mostly tendentious (0.85..0.98)
        if n_inf == 0:
            inf_tend = np.zeros(0, dtype=np.float64)
        elif influencer_tendentiousness is None or (
            isinstance(influencer_tendentiousness, str)
            and influencer_tendentiousness.lower() in ("heterogeneous", "mixed", "bimodal", "extreme_imposed")
        ):
            is_tend = rng.random(n_inf) < 0.50
            inf_tend = np.where(
                is_tend,
                rng.uniform(0.70, 0.95, size=n_inf),
                rng.uniform(0.10, 0.35, size=n_inf),
            )
        elif isinstance(influencer_tendentiousness, str) and influencer_tendentiousness.lower() == "factual":
            inf_tend = rng.uniform(0.08, 0.22, size=n_inf)
        elif isinstance(influencer_tendentiousness, str) and influencer_tendentiousness.lower() == "tendentious":
            inf_tend = rng.uniform(0.80, 0.96, size=n_inf)
        elif np.ndim(influencer_tendentiousness) == 0:
            inf_tend = np.full(n_inf, float(influencer_tendentiousness), dtype=np.float64)
        else:
            raw_it = np.asarray(influencer_tendentiousness, dtype=np.float64).ravel()
            if len(raw_it) == 2 and n_inf > 2:
                half = n_inf // 2
                inf_tend = np.empty(n_inf, dtype=np.float64)
                inf_tend[:half] = raw_it[0]
                inf_tend[half:] = raw_it[1]
            elif len(raw_it) < n_inf:
                inf_tend = np.resize(raw_it, n_inf)
            else:
                inf_tend = raw_it[:n_inf].copy()

        if extreme_imposed_enabled and n_inf > 0:
            # Influencers in the extreme_imposed region are mostly tendentious (0.85 - 0.98)
            ext_mask = inf_is_extreme == 1
            inf_tend[ext_mask] = rng.uniform(0.85, 0.98, size=int(np.sum(ext_mask)))
        inf_tend = np.ascontiguousarray(np.clip(inf_tend, 0.0, 1.0), dtype=np.float64)
        inf_is_extreme = np.ascontiguousarray(inf_is_extreme, dtype=np.uint8)

        if n_inf > 0 and len(influencer_opinions) > 0:
            if extreme_imposed_enabled and np.any(inf_is_extreme == 1):
                # Artificial algorithmic & astroturfing boost gives Extreme Imposed influencers higher post share
                inf_weights = np.where(
                    inf_is_extreme == 1,
                    float(max(1.5, extreme_imposed_reach_boost)),
                    1.0,
                )
                inf_weights /= np.sum(inf_weights)
                creator_idx = rng.choice(n_inf, size=n_items, replace=True, p=inf_weights).astype(np.int32)
            else:
                creator_idx = rng.integers(0, n_inf, size=n_items, dtype=np.int32)
            creator_inf_idx = np.ascontiguousarray(creator_idx, dtype=np.int32)
            creator_id = np.ascontiguousarray(influencer_ids[creator_idx], dtype=np.int32)
            item_tend = inf_tend[creator_idx]
            fact_anchor = rng.uniform(-0.65, 0.65, size=(n_items, n_dims))
            inf_anchor = np.clip(influencer_opinions[creator_idx] * (0.95 + 0.40 * item_tend[:, None]), -1.0, 1.0)
            base_framing = (
                (1.0 - item_tend[:, None]) * fact_anchor
                + item_tend[:, None] * inf_anchor
                + rng.normal(0.0, 0.12, size=(n_items, n_dims))
            )
            if extreme_imposed_enabled:
                ext_items = inf_is_extreme[creator_idx] == 1
                if np.any(ext_items):
                    base_framing[ext_items] = (
                        0.82 * ext_target[None, :]
                        + 0.18 * base_framing[ext_items]
                        + rng.normal(0.0, 0.05, size=(int(np.sum(ext_items)), n_dims))
                    )
        else:
            creator_inf_idx = np.full(n_items, -1, dtype=np.int32)
            creator_id = np.full(n_items, -1, dtype=np.int32)
            item_tend = np.full(n_items, 0.50, dtype=np.float64)
            base_framing = rng.uniform(-1.0, 1.0, size=(n_items, n_dims))

        framing = np.ascontiguousarray(np.clip(base_framing, -1.0, 1.0), dtype=np.float64)

        # Assign platform type per item: 42% Instagram/TikTok (0), 30% YouTube/X (1), 28% Facebook (2)
        platform_type = np.ascontiguousarray(
            rng.choice(
                np.array([0, 1, 2], dtype=np.int32),
                size=n_items,
                p=[0.42, 0.30, 0.28],
            ),
            dtype=np.int32,
        )

        # Tendentiousness and misinfo_fraction jointly determine veracity and sensationalism
        eff_misinfo_prob = np.clip(misinfo_fraction * (0.35 + 1.15 * item_tend), 0.02, 0.95)
        is_misinfo = rng.random(n_items) < eff_misinfo_prob
        veracity = np.where(
            is_misinfo,
            rng.beta(1.5, 4.5, size=n_items) * (1.0 - 0.45 * item_tend),
            np.clip(0.88 - 0.35 * item_tend + rng.normal(0.0, 0.07, size=n_items), 0.15, 0.99),
        )
        veracity = np.ascontiguousarray(np.clip(veracity, 0.0, 1.0), dtype=np.float64)

        sensationalism = np.where(
            is_misinfo,
            np.clip(0.45 + 0.48 * item_tend + rng.normal(0.0, 0.08, size=n_items), 0.15, 0.99),
            np.clip(0.18 + 0.55 * item_tend + rng.normal(0.0, 0.08, size=n_items), 0.05, 0.95),
        )
        sensationalism = np.ascontiguousarray(np.clip(sensationalism, 0.0, 1.0), dtype=np.float64)

        popularity = np.ascontiguousarray(np.ones(n_items, dtype=np.float64))
        item_is_sync = np.zeros(n_items, dtype=np.uint8)
        focal_fact_vec = rng.uniform(-0.35, 0.85, size=n_dims)
        focal_fact_rel = float(rng.uniform(0.70, 0.98))
        focal_fact_longev = float(1.4 + 1.2 * focal_fact_rel)
        primary_subj = int(np.argmax(np.abs(focal_fact_vec)))
        focal_fact_framed = np.clip(0.88 * ext_target + 0.12 * focal_fact_vec, -1.0, 1.0)

        if extreme_imposed_enabled and n_inf > 0:
            item_is_ext = inf_is_extreme[creator_inf_idx] == 1
            popularity[item_is_ext] = float(max(1.0, extreme_imposed_popularity_boost)) * 3.0
            if sync_tactics_enabled and np.any(item_is_ext):
                item_is_sync[item_is_ext] = 1
                # Synchronize all extreme_imposed items tightly around the same focal fact & target
                n_sync = int(np.sum(item_is_ext))
                sync_frames = (
                    0.88 * focal_fact_framed[None, :]
                    + 0.12 * framing[item_is_ext]
                    + rng.normal(0.0, 0.025, size=(n_sync, n_dims))
                )
                framing[item_is_ext] = np.clip(sync_frames, -1.0, 1.0)
                popularity[item_is_ext] *= 1.35

        like_count = np.zeros(n_items, dtype=np.int64)
        dislike_count = np.zeros(n_items, dtype=np.int64)
        viral_cascades = np.zeros(n_items, dtype=np.int64)

        eff_groups = n_vehicle_groups if n_vehicle_groups is not None else max(1, n_broadcast)
        trad_eco = TraditionalMediaEcosystem.initialize(
            n_groups=eff_groups,
            n_dims=n_dims,
            group_centers=vehicle_group_centers,
            group_tendentiousness=group_tendentiousness,
            fact_rate=fact_rate,
            gatekeeping_sigma=gatekeeping_sigma,
            seed=rng,
        )

        # Keep broadcast_orientations synchronized with vehicle centers for backward compatibility
        if trad_eco.n_vehicles > 0:
            broadcast_orientations = trad_eco.active_framing
            broadcast_credibility = trad_eco.vehicle_credibility
            n_b = trad_eco.n_vehicles
        else:
            n_b = max(1, n_broadcast)
            broadcast_orientations = np.zeros((n_b, n_dims), dtype=np.float64)
            broadcast_credibility = np.full(n_b, 0.70, dtype=np.float64)

        return cls(
            n_items=n_items,
            n_dims=n_dims,
            framing=framing,
            sensationalism=sensationalism,
            veracity=veracity,
            creator_id=creator_id,
            creator_inf_idx=creator_inf_idx,
            platform_type=platform_type,
            popularity=popularity,
            like_count=like_count,
            dislike_count=dislike_count,
            viral_cascades=viral_cascades,
            influencer_tendentiousness=inf_tend,
            influencer_is_extreme_imposed=inf_is_extreme,
            extreme_imposed_target=ext_target,
            extreme_imposed_enabled=bool(extreme_imposed_enabled),
            extreme_imposed_reach_boost=float(extreme_imposed_reach_boost),
            extreme_imposed_popularity_boost=float(extreme_imposed_popularity_boost),
            sync_tactics_enabled=bool(sync_tactics_enabled),
            cowork_enabled=bool(cowork_enabled),
            cowork_prob=float(cowork_prob),
            cowork_reach_fraction=float(cowork_reach_fraction),
            fact_dictation_enabled=bool(fact_dictation_enabled),
            fact_dictation_pull=float(fact_dictation_pull),
            item_is_synchronized=np.ascontiguousarray(item_is_sync, dtype=np.uint8),
            sync_focal_fact_vec=np.ascontiguousarray(focal_fact_vec, dtype=np.float64),
            sync_focal_fact_framed=np.ascontiguousarray(focal_fact_framed, dtype=np.float64),
            sync_focal_fact_relevance=focal_fact_rel,
            sync_focal_fact_longevity=focal_fact_longev,
            sync_focal_fact_remaining_steps=max(4, int(round(6.0 * focal_fact_longev))),
            sync_primary_subject=primary_subj,
            sync_network_interest_push=1.0 if (extreme_imposed_enabled and sync_tactics_enabled) else 0.0,
            n_broadcast=n_b,
            broadcast_orientations=broadcast_orientations,
            broadcast_credibility=broadcast_credibility,
            traditional_ecosystem=trad_eco,
        )

    def step_synchronized_fact_campaign(
        self,
        rng: np.random.Generator,
        step_ext_engagements: float = 0.0,
        n_agents: int = 1000,
    ) -> None:
        """Advance or rotate the synchronized focal fact campaign for the extreme_imposed group.

        Grounded in Pacheco et al. (2021, ICWSM), Giglietto et al. (2020), and Downs (1972):
        - When a specific focal fact is selected by the coordinated network, its intrinsic relevance
          r_k in [0.35, 1.0] and the network's engagement push determine how relevant it becomes and
          how many steps it lasts (`sync_focal_fact_remaining_steps`).
        - All synchronized items in the pool align tightly around this focal fact and its dictated
          tendentious interpretation toward `extreme_imposed_target`.
        """
        if not (self.extreme_imposed_enabled and self.sync_tactics_enabled):
            self.sync_network_interest_push = 0.0
            return

        if self.item_is_synchronized is None or len(self.item_is_synchronized) != self.n_items:
            self.item_is_synchronized = np.zeros(self.n_items, dtype=np.uint8)

        # Compute network push in interest from step engagements + popularity concentration
        eng_rate = float(step_ext_engagements) / max(100.0, float(n_agents) * 0.25)
        raw_push = 0.55 + 0.65 * min(2.0, eng_rate) * math.log1p(max(1.0, self.extreme_imposed_reach_boost))
        self.sync_network_interest_push = float(
            0.75 * self.sync_network_interest_push + 0.25 * min(2.2, raw_push)
        )

        self.sync_focal_fact_remaining_steps -= 1
        if self.sync_focal_fact_remaining_steps <= 0 or self.sync_focal_fact_vec is None:
            # Select a new focal fact with heterogeneous intrinsic relevance (some last longer, some shorter)
            new_fact = rng.uniform(-0.40, 0.90, size=self.n_dims)
            self.sync_primary_subject = int(np.argmax(np.abs(new_fact)))
            # Emphasize the primary dictated subject
            new_fact[self.sync_primary_subject] = float(
                np.clip(0.65 + 0.35 * abs(new_fact[self.sync_primary_subject]), 0.55, 0.98)
            )
            intrinsic_rel = float(0.30 + 0.70 * rng.beta(1.8, 1.6))
            eff_rel = float(min(1.0, intrinsic_rel * (0.80 + 0.25 * min(1.5, self.sync_network_interest_push))))
            # More relevant + network-pushed facts last substantially longer
            eff_longev = float((0.85 + 1.45 * eff_rel) * (1.0 + 0.40 * min(1.5, self.sync_network_interest_push)))
            self.sync_focal_fact_vec = np.ascontiguousarray(new_fact, dtype=np.float64)
            self.sync_focal_fact_relevance = eff_rel
            self.sync_focal_fact_longevity = eff_longev
            self.sync_focal_fact_remaining_steps = max(3, int(round(5.5 * eff_longev)))
            self.sync_focal_fact_framed = np.ascontiguousarray(
                np.clip(0.86 * self.extreme_imposed_target + 0.14 * new_fact, -1.0, 1.0),
                dtype=np.float64,
            )

        # Synchronize all active extreme_imposed items to the current focal fact campaign
        valid_c = self.creator_inf_idx >= 0
        if np.any(valid_c) and len(self.influencer_is_extreme_imposed) > 0:
            ext_items = np.zeros(self.n_items, dtype=bool)
            ext_items[valid_c] = self.influencer_is_extreme_imposed[self.creator_inf_idx[valid_c]] == 1
            self.item_is_synchronized[:] = 0
            self.item_is_synchronized[ext_items] = 1
            if np.any(ext_items) and self.sync_focal_fact_framed is not None:
                n_ext_i = int(np.sum(ext_items))
                # Pull active extreme_imposed posts into tight synchronization around the focal fact
                synced = (
                    0.35 * self.framing[ext_items]
                    + 0.65 * self.sync_focal_fact_framed[None, :]
                    + rng.normal(0.0, 0.018, size=(n_ext_i, self.n_dims))
                )
                self.framing[ext_items] = np.clip(synced, -1.0, 1.0)

    def refresh_influencer_content(
        self,
        influencer_opinions: np.ndarray,
        influencer_ids: np.ndarray | None = None,
        replace_fraction: float = 0.20,
        rng: np.random.Generator | None = None,
    ) -> None:
        """Coevolution step: influencers refresh content according to their factual vs. tendentious profile."""
        if rng is None:
            rng = np.random.default_rng()
        if len(influencer_opinions) == 0:
            return

        n_replace = max(1, int(self.n_items * replace_fraction))
        lowest_idx = np.argsort(self.popularity)[:n_replace]
        n_inf = len(influencer_opinions)
        if self.extreme_imposed_enabled and len(self.influencer_is_extreme_imposed) == n_inf and np.any(self.influencer_is_extreme_imposed == 1):
            inf_w = np.where(
                self.influencer_is_extreme_imposed == 1,
                float(max(1.5, self.extreme_imposed_reach_boost)),
                1.0,
            )
            inf_w /= np.sum(inf_w)
            chosen_inf = rng.choice(n_inf, size=n_replace, replace=True, p=inf_w).astype(np.int32)
        else:
            chosen_inf = rng.integers(0, max(1, n_inf), size=n_replace, dtype=np.int32)
        self.creator_inf_idx[lowest_idx] = chosen_inf
        if influencer_ids is not None and len(influencer_ids) == n_inf:
            self.creator_id[lowest_idx] = influencer_ids[chosen_inf]

        if len(self.influencer_tendentiousness) == n_inf:
            tend_chosen = self.influencer_tendentiousness[chosen_inf]
        else:
            tend_chosen = np.full(n_replace, 0.60, dtype=np.float64)

        fact_anchor = rng.uniform(-0.60, 0.60, size=(n_replace, self.n_dims))
        inf_anchor = np.clip(
            influencer_opinions[chosen_inf] * (0.95 + 0.40 * tend_chosen[:, None]), -1.0, 1.0
        )
        raw = (
            (1.0 - tend_chosen[:, None]) * fact_anchor
            + tend_chosen[:, None] * inf_anchor
            + rng.normal(0.0, 0.12, size=(n_replace, self.n_dims))
        )
        if self.extreme_imposed_enabled and len(self.influencer_is_extreme_imposed) == n_inf:
            ext_chosen = self.influencer_is_extreme_imposed[chosen_inf] == 1
            if np.any(ext_chosen):
                if self.sync_tactics_enabled and self.sync_focal_fact_framed is not None:
                    raw[ext_chosen] = (
                        0.88 * self.sync_focal_fact_framed[None, :]
                        + 0.12 * raw[ext_chosen]
                        + rng.normal(0.0, 0.025, size=(int(np.sum(ext_chosen)), self.n_dims))
                    )
                else:
                    raw[ext_chosen] = (
                        0.80 * self.extreme_imposed_target[None, :]
                        + 0.20 * raw[ext_chosen]
                        + rng.normal(0.0, 0.06, size=(int(np.sum(ext_chosen)), self.n_dims))
                    )
                if self.item_is_synchronized is not None and len(self.item_is_synchronized) == self.n_items:
                    self.item_is_synchronized[lowest_idx] = 0
                    if self.sync_tactics_enabled:
                        self.item_is_synchronized[lowest_idx[ext_chosen]] = 1
        self.framing[lowest_idx] = np.clip(raw, -1.0, 1.0)
        self.sensationalism[lowest_idx] = np.clip(
            0.18 + 0.65 * tend_chosen + rng.normal(0.0, 0.08, size=n_replace), 0.05, 0.98
        )
        self.veracity[lowest_idx] = np.clip(
            0.90 - 0.65 * tend_chosen + rng.normal(0.0, 0.08, size=n_replace),
            0.05,
            0.99,
        )
        base_pop = np.mean(self.popularity) * 0.5 + 1.0
        self.popularity[lowest_idx] = base_pop
        if self.extreme_imposed_enabled and len(self.influencer_is_extreme_imposed) == n_inf:
            ext_chosen = self.influencer_is_extreme_imposed[chosen_inf] == 1
            if np.any(ext_chosen):
                sync_mult = 1.35 if self.sync_tactics_enabled else 1.0
                self.popularity[lowest_idx[ext_chosen]] = (
                    base_pop * max(1.0, self.extreme_imposed_popularity_boost) * sync_mult
                )


