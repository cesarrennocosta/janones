"""SimulationEngine: End-to-end orchestrator integrating all OmniOpinion skills for HPC/SLURM execution."""

from __future__ import annotations

from pathlib import Path
import signal
import numpy as np

from src.core.agent_state import AgentStateEngine
from src.core.multiplex_network import GraphEngine
from src.core.news_entity import NewsEntityPool, compute_fact_bias_vector
from src.engines.electoral_engine import ElectoralEngine
from src.engines.media_engine import MediaEngine
from src.io.parquet_writer import HPCStorage
from src.kernels.opinion_updates import DynamicsKernel, step_conviction_and_opinion_tether_jit
from src.metrics.polarization import AnalyticsEngine


class SimulationEngine:
    """High-level coordinator uniting AgentStateEngine, GraphEngine, DynamicsKernel,
    MediaEngine, ElectoralEngine, AnalyticsEngine, and HPCStorage.

    Supports:
    - Pure local-connection mode (`enable_media=False`) or full media ecosystem (`enable_media=True`).
    - Uniform random (`init_mode="uniform_random"`) or correlated initialization.
    - Mid-simulation checkpointing and automatic preemption recovery (`--qos=preempt_strict`).
    """

    def __init__(
        self,
        config: dict | None = None,
        base_seed: int = 12345,
        task_id: int = 0,
        output_dir: str | Path | None = None,
    ) -> None:
        self.config = config or {}
        self.base_seed = int(base_seed)
        self.task_id = int(task_id)
        self.output_dir = Path(output_dir) if output_dir is not None else None

        # Derive independent reproducible stream via SeedSequence(entropy, spawn_key=(task_id,))
        self.seed_sequence = np.random.SeedSequence(self.base_seed, spawn_key=(self.task_id,))
        child_seeds = self.seed_sequence.spawn(5)
        self.rng_agents = np.random.default_rng(child_seeds[0])
        self.rng_network = np.random.default_rng(child_seeds[1])
        self.rng_media = np.random.default_rng(child_seeds[2])
        self.rng_dynamics = np.random.default_rng(child_seeds[3])
        self.rng_electoral = np.random.default_rng(child_seeds[4])

        sim_cfg = self.config.get("simulation", {})
        media_cfg = self.config.get("media", {})

        self.n_agents = int(sim_cfg.get("n_agents", 1000))
        self.n_dims = int(sim_cfg.get("n_dims", 4))
        self.n_steps = int(sim_cfg.get("n_steps", 50))
        self.snapshot_interval = max(1, int(sim_cfg.get("snapshot_interval", 5)))
        self.checkpoint_interval = max(1, int(sim_cfg.get("checkpoint_interval", 10)))
        self.n_candidates = int(sim_cfg.get("n_candidates", 3))
        self.n_districts = int(sim_cfg.get("n_districts", 10))
        self.n_news_items = int(sim_cfg.get("n_news_items", 40))
        self.init_mode = str(sim_cfg.get("init_mode", "correlated"))
        self.age_distribution = str(sim_cfg.get("age_distribution", "ibge_2022"))
        self.enable_media = bool(media_cfg.get("enabled", True))
        self.enable_traditional_media = bool(
            media_cfg.get("traditional_enabled", self.enable_media)
        )
        self.enable_social_media = bool(
            media_cfg.get("social_enabled", self.enable_media)
        )
        self.n_vehicle_groups = int(
            media_cfg.get("n_vehicle_groups", media_cfg.get("n_broadcast", 3))
            if self.enable_traditional_media
            else 0
        )
        self.conviction_dynamics_enabled = bool(
            sim_cfg.get("conviction_dynamics_enabled", False)
        )
        self.max_excursion_base = float(sim_cfg.get("max_excursion_base", 0.34))
        self.conviction_drift_rate = float(sim_cfg.get("conviction_drift_rate", 0.0024))
        self.crystallization_tau = float(sim_cfg.get("crystallization_tau", 280.0))
        self.fact_direct_pull = float(sim_cfg.get("fact_direct_pull", 0.055))
        self.fact_bias_mode = str(media_cfg.get("fact_bias_mode", "neutral"))

        # 1. Initialize AgentStateEngine
        self.agent_state = AgentStateEngine.initialize(
            n_agents=self.n_agents,
            n_dims=self.n_dims,
            n_candidates=self.n_candidates,
            n_districts=self.n_districts,
            seed=self.rng_agents,
            base_epsilon=float(sim_cfg.get("base_epsilon", 0.45)),
            base_assimilation_bias=float(sim_cfg.get("base_assimilation_bias", 1.2)),
            init_mode=self.init_mode,
            age_distribution=self.age_distribution,
        )

        # 2. Initialize Multiplex GraphEngine (build social/WA graphs only when social media is enabled)
        net_cfg = self.config.get("networks", {})
        self.graph_engine = GraphEngine.build_multiplex(
            agent_state=self.agent_state,
            config=net_cfg,
            seed=self.rng_network,
            enable_media_networks=self.enable_social_media,
        )

        # 3. Initialize NewsEntityPool (if traditional or social media enabled)
        self.news_pool: NewsEntityPool | None = None
        if self.enable_traditional_media or self.enable_social_media:
            inf_opinions = (
                self.agent_state.opinions[self.graph_engine.influencer_ids]
                if len(self.graph_engine.influencer_ids) > 0
                else np.zeros((0, self.n_dims), dtype=np.float64)
            )
            self.news_pool = NewsEntityPool.initialize(
                n_items=self.n_news_items if self.enable_social_media else 1,
                n_dims=self.n_dims,
                influencer_ids=self.graph_engine.influencer_ids,
                influencer_opinions=inf_opinions,
                n_broadcast=max(1, self.n_vehicle_groups),
                misinfo_fraction=float(media_cfg.get("misinfo_fraction", 0.25)),
                seed=self.rng_media,
                n_vehicle_groups=self.n_vehicle_groups,
                vehicle_group_centers=media_cfg.get("vehicle_group_centers", None),
                group_tendentiousness=media_cfg.get("group_tendentiousness", 0.65),
                influencer_tendentiousness=media_cfg.get(
                    "influencer_tendentiousness", "heterogeneous"
                ),
                fact_rate=float(media_cfg.get("fact_rate", 1.5)),
                gatekeeping_sigma=float(media_cfg.get("gatekeeping_sigma", 0.85)),
                extreme_imposed_enabled=bool(
                    media_cfg.get("extreme_imposed_enabled", False)
                ),
                extreme_imposed_target=media_cfg.get("extreme_imposed_target", None),
                extreme_imposed_Radius=float(
                    media_cfg.get("extreme_imposed_Radius", 0.52)
                ),
                extreme_imposed_fraction=float(
                    media_cfg.get("extreme_imposed_fraction", 0.25)
                ),
                extreme_imposed_reach_boost=float(
                    media_cfg.get("extreme_imposed_reach_boost", 3.5)
                ),
                extreme_imposed_popularity_boost=float(
                    media_cfg.get("extreme_imposed_popularity_boost", 4.0)
                ),
                sync_tactics_enabled=bool(
                    media_cfg.get("sync_tactics_enabled", False)
                ),
                cowork_enabled=bool(
                    media_cfg.get("cowork_enabled", False)
                ),
                cowork_prob=float(
                    media_cfg.get("cowork_prob", 0.65)
                ),
                cowork_reach_fraction=float(
                    media_cfg.get("cowork_reach_fraction", 0.50)
                ),
                fact_dictation_enabled=bool(
                    media_cfg.get("fact_dictation_enabled", False)
                ),
                fact_dictation_pull=float(
                    media_cfg.get("fact_dictation_pull", 0.42)
                ),
                fact_bias_mode=self.fact_bias_mode,
                agent_opinions=self.agent_state.opinions,
                agent_innate_beliefs=self.agent_state.innate_beliefs,
                include_extended_modalities=bool(
                    media_cfg.get("include_extended_modalities", False)
                ),
            )

        # 4. Initialize Kernels & Engines
        self.dynamics_kernel = DynamicsKernel(
            dt=float(sim_cfg.get("dt", 0.15)),
            noise_std=float(sim_cfg.get("noise_std", 0.01)),
            media_strength=float(media_cfg.get("feed_strength", 0.07))
            if (self.enable_traditional_media or self.enable_social_media)
            else 0.0,
        )
        self.media_engine = MediaEngine(
            beta=float(media_cfg.get("beta", 2.0)),
            emotional_boost=float(media_cfg.get("emotional_boost", 1.5)),
            broadcast_strength=float(media_cfg.get("broadcast_strength", 0.06)),
            feed_strength=float(media_cfg.get("feed_strength", 0.07)),
            whatsapp_trust=float(media_cfg.get("whatsapp_trust", 0.85)),
            whatsapp_forward_prob=float(media_cfg.get("whatsapp_forward_prob", 0.25)),
            enable_traditional_media=self.enable_traditional_media,
            enable_social_media=self.enable_social_media,
            traditional_selectivity_beta=float(
                media_cfg.get("traditional_selectivity_beta", 1.2)
            ),
            dislike_reaction_strength=float(
                media_cfg.get("dislike_reaction_strength", 0.45)
            ),
            backfire_repulsion_strength=float(
                media_cfg.get("backfire_repulsion_strength", 0.35)
            ),
            influencer_impact_strength=float(
                media_cfg.get("influencer_impact_strength", 0.04)
            ),
        )
        elec_cfg = self.config.get("electoral", {})
        two_leg_enabled = bool(
            elec_cfg.get("two_leg_enabled", self.conviction_dynamics_enabled or (self.n_candidates >= 4))
        )
        self.electoral_engine = ElectoralEngine(
            n_candidates=self.n_candidates,
            n_dims=self.n_dims,
            indifference_threshold=float(elec_cfg.get("indifference_threshold", -1.65)),
            rationality_lambda=float(elec_cfg.get("rationality_lambda", 4.0)),
            noise_scale=float(elec_cfg.get("noise_scale", 0.12)),
            two_leg_enabled=two_leg_enabled,
            adm_dist_threshold=float(elec_cfg.get("adm_dist_threshold", 0.56)),
            rej_dist_threshold=float(elec_cfg.get("rej_dist_threshold", 0.62)),
            adm_gain_rate=float(elec_cfg.get("adm_gain_rate", 0.14)),
            adm_decay_rate=float(elec_cfg.get("adm_decay_rate", 0.08)),
            rej_gain_rate=float(elec_cfg.get("rej_gain_rate", 0.14)),
            rej_decay_rate=float(elec_cfg.get("rej_decay_rate", 0.045)),
            turnout_min_adm=float(elec_cfg.get("turnout_min_adm", 0.19)),
            clear_margin_threshold=float(elec_cfg.get("clear_margin_threshold", 0.15)),
            clear_adm_threshold=float(elec_cfg.get("clear_adm_threshold", 0.34)),
            strong_rej_threshold=float(elec_cfg.get("strong_rej_threshold", 0.50)),
        )
        self.analytics_engine = AnalyticsEngine(
            er_alpha=float(sim_cfg.get("er_alpha", 1.3)),
            er_bins=int(sim_cfg.get("er_bins", 4)),
        )

        self.storage: HPCStorage | None = None
        if self.output_dir is not None:
            io_cfg = self.config.get("io", {})
            # For N >= 1,000,000, default to saving checkpoints + initial/final states to conserve BeeGFS quota unless explicitly enabled
            default_save_traj = self.n_agents <= 500_000
            save_full_traj = bool(io_cfg.get("save_full_trajectory", default_save_traj))
            self.storage = HPCStorage(
                output_dir=self.output_dir,
                task_id=self.task_id,
                n_agents=self.n_agents,
                n_dims=self.n_dims,
                hdf5_compression=str(io_cfg.get("hdf5_compression", "lzf" if self.n_agents >= 100_000 else "gzip")),
                buffer_size=int(io_cfg.get("buffer_size", 10)),
                save_full_trajectory=save_full_traj,
            )

        self._preempt_requested = False

    def _install_signal_handlers(self) -> None:
        """Catch SIGUSR1 / SIGTERM from SLURM preemption to save checkpoint cleanly before exit."""
        def _handler(signum, frame):
            self._preempt_requested = True

        for sig in (signal.SIGTERM, getattr(signal, "SIGUSR1", signal.SIGTERM)):
            try:
                signal.signal(sig, _handler)
            except (ValueError, OSError):
                pass

    def run(self) -> dict:
        """Execute the simulation loop with automatic checkpoint resume and preemption protection."""
        self._install_signal_handlers()

        start_step = 0
        history: list[dict] = []
        last_electoral: dict = {}

        if self.storage is not None:
            resumed_step, prev_history = self.storage.try_resume_checkpoint(self.agent_state)
            if resumed_step >= 0 and prev_history:
                start_step = resumed_step + 1
                history = prev_history
                if start_step <= self.n_steps:
                    # Advance RNG streams & warm up traditional news cycle up to resumed_step
                    self.rng_dynamics.bit_generator.advance(resumed_step)
                    self.rng_electoral.bit_generator.advance(resumed_step)
                    if (
                        self.enable_traditional_media
                        and self.news_pool is not None
                        and self.news_pool.traditional_ecosystem is not None
                    ):
                        warmup_start = max(1, resumed_step - 15)
                        if warmup_start > 1:
                            self.rng_media.bit_generator.advance(warmup_start - 1)
                        for s_idx in range(warmup_start, resumed_step + 1):
                            self.news_pool.traditional_ecosystem.step_facts_and_news_cycle(
                                s_idx, self.rng_media
                            )
                else:
                    last_electoral = self.electoral_engine.evaluate_election(
                        self.agent_state, self.rng_electoral
                    )
                    return {
                        "task_id": self.task_id,
                        "preempted": False,
                        "final_metrics": history[-1],
                        "electoral_summary": last_electoral,
                        "history": history,
                        "hdf5_path": self.storage.hdf5_path,
                        "parquet_path": self.storage.parquet_path,
                    }
            else:
                self.storage.initialize(
                    self.agent_state,
                    metadata={
                        "base_seed": self.base_seed,
                        "task_id": self.task_id,
                        "n_agents": self.n_agents,
                        "n_dims": self.n_dims,
                        "n_steps": self.n_steps,
                        "init_mode": self.init_mode,
                        "enable_media": int(self.enable_media),
                        "n_vehicle_groups": int(self.n_vehicle_groups),
                    },
                )

        for step in range(start_step, self.n_steps + 1):
            prev_opinions = self.agent_state.opinions.copy()
            prev_convictions = self.agent_state.innate_beliefs.copy()
            fact_bias_vec, fact_bias_active = compute_fact_bias_vector(
                step, self.n_dims, self.fact_bias_mode
            )

            if step > 0:
                self.dynamics_kernel.step_interpersonal(
                    self.agent_state, self.graph_engine, self.rng_dynamics
                )
                if (self.enable_traditional_media or self.enable_social_media) and self.news_pool is not None:
                    media_stats = self.media_engine.step_all_platforms(
                        self.agent_state,
                        self.graph_engine,
                        self.news_pool,
                        self.rng_media,
                        step_idx=step,
                    )
                else:
                    media_stats = {
                        "broadcast_reached": 0.0,
                        "tv_reached": 0.0,
                        "radio_reached": 0.0,
                        "press_reached": 0.0,
                        "facts_count": 0.0,
                        "published_notices": 0.0,
                        "feed_engaged": 0.0,
                        "whatsapp_cascades": 0.0,
                    }

                if self.conviction_dynamics_enabled:
                    # Keep conviction evolution strictly governed by the slow two-timescale integrator
                    self.agent_state.innate_beliefs[:] = prev_convictions
                    step_conviction_and_opinion_tether_jit(
                        self.agent_state.opinions,
                        self.agent_state.innate_beliefs,
                        self.agent_state.age,
                        self.agent_state.ses,
                        self.agent_state.susceptibility,
                        fact_bias_vec,
                        int(step),
                        float(self.max_excursion_base),
                        float(self.conviction_drift_rate),
                        float(self.crystallization_tau),
                        float(self.fact_direct_pull),
                        1 if fact_bias_active else 0,
                    )
            else:
                media_stats = {
                    "broadcast_reached": 0.0,
                    "tv_reached": 0.0,
                    "radio_reached": 0.0,
                    "press_reached": 0.0,
                    "facts_count": 0.0,
                    "published_notices": 0.0,
                    "feed_engaged": 0.0,
                    "whatsapp_cascades": 0.0,
                }

            last_electoral = self.electoral_engine.evaluate_election(
                self.agent_state, self.rng_electoral
            )
            step_metrics = self.analytics_engine.compute_step_metrics(
                step=step,
                agent_state=self.agent_state,
                graph_engine=self.graph_engine,
                electoral_result=last_electoral,
                media_stats=media_stats,
                news_pool=self.news_pool,
                candidate_profiles=self.electoral_engine.candidate_profiles,
                prev_opinions=prev_opinions,
                prev_convictions=prev_convictions,
                fact_bias_d0=float(fact_bias_vec[0]) if self.n_dims > 0 else 0.0,
            )
            step_metrics["task_id"] = self.task_id
            step_metrics["n_agents"] = self.n_agents
            history.append(step_metrics)

            if self.storage is not None:
                save_traj = (step % self.snapshot_interval == 0) or (step == self.n_steps)
                save_chk = (
                    (step > 0 and step % self.checkpoint_interval == 0)
                    or self._preempt_requested
                )
                self.storage.record_step(
                    step=step,
                    agent_state=self.agent_state,
                    metrics=step_metrics,
                    save_trajectory=save_traj,
                    save_checkpoint=save_chk,
                )

            if self._preempt_requested:
                break

        hdf5_path = None
        parquet_path = None
        if self.storage is not None and not self._preempt_requested:
            hdf5_path, parquet_path = self.storage.finalize(
                self.agent_state, news_pool=self.news_pool, final_step=self.n_steps
            )
        elif self.storage is not None:
            hdf5_path = self.storage.hdf5_path
            parquet_path = self.storage.parquet_path

        return {
            "task_id": self.task_id,
            "preempted": self._preempt_requested,
            "final_metrics": history[-1],
            "electoral_summary": last_electoral,
            "history": history,
            "hdf5_path": hdf5_path,
            "parquet_path": parquet_path,
        }
