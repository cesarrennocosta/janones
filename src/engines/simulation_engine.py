"""SimulationEngine: End-to-end orchestrator integrating all OmniOpinion skills for HPC/SLURM execution."""

from __future__ import annotations

from pathlib import Path
import numpy as np

from src.core.agent_state import AgentStateEngine
from src.core.multiplex_network import GraphEngine
from src.core.news_entity import NewsEntityPool
from src.engines.electoral_engine import ElectoralEngine
from src.engines.media_engine import MediaEngine
from src.io.parquet_writer import HPCStorage
from src.kernels.opinion_updates import DynamicsKernel
from src.metrics.polarization import AnalyticsEngine


class SimulationEngine:
    """High-level coordinator uniting AgentStateEngine, GraphEngine, DynamicsKernel,
    MediaEngine, ElectoralEngine, AnalyticsEngine, and HPCStorage.

    Uses `numpy.random.SeedSequence` to derive statistically independent RNG streams
    for each SLURM array task (`SLURM_ARRAY_TASK_ID`).
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
        self.n_agents = int(sim_cfg.get("n_agents", 1000))
        self.n_dims = int(sim_cfg.get("n_dims", 4))
        self.n_steps = int(sim_cfg.get("n_steps", 50))
        self.snapshot_interval = max(1, int(sim_cfg.get("snapshot_interval", 5)))
        self.n_candidates = int(sim_cfg.get("n_candidates", 3))
        self.n_districts = int(sim_cfg.get("n_districts", 10))
        self.n_news_items = int(sim_cfg.get("n_news_items", 40))

        # 1. Initialize AgentStateEngine
        self.agent_state = AgentStateEngine.initialize(
            n_agents=self.n_agents,
            n_dims=self.n_dims,
            n_candidates=self.n_candidates,
            n_districts=self.n_districts,
            seed=self.rng_agents,
            base_epsilon=float(sim_cfg.get("base_epsilon", 0.45)),
            base_assimilation_bias=float(sim_cfg.get("base_assimilation_bias", 1.2)),
        )

        # 2. Initialize Multiplex GraphEngine
        net_cfg = self.config.get("networks", {})
        self.graph_engine = GraphEngine.build_multiplex(
            agent_state=self.agent_state,
            config=net_cfg,
            seed=self.rng_network,
        )

        # 3. Initialize NewsEntityPool
        media_cfg = self.config.get("media", {})
        inf_opinions = self.agent_state.opinions[self.graph_engine.influencer_ids]
        self.news_pool = NewsEntityPool.initialize(
            n_items=self.n_news_items,
            n_dims=self.n_dims,
            influencer_ids=self.graph_engine.influencer_ids,
            influencer_opinions=inf_opinions,
            n_broadcast=int(media_cfg.get("n_broadcast", 3)),
            misinfo_fraction=float(media_cfg.get("misinfo_fraction", 0.25)),
            seed=self.rng_media,
        )

        # 4. Initialize Kernels & Engines
        self.dynamics_kernel = DynamicsKernel(
            dt=float(sim_cfg.get("dt", 0.15)),
            noise_std=float(sim_cfg.get("noise_std", 0.01)),
            media_strength=float(media_cfg.get("feed_strength", 0.07)),
        )
        self.media_engine = MediaEngine(
            beta=float(media_cfg.get("beta", 2.0)),
            emotional_boost=float(media_cfg.get("emotional_boost", 1.5)),
            broadcast_strength=float(media_cfg.get("broadcast_strength", 0.05)),
            feed_strength=float(media_cfg.get("feed_strength", 0.07)),
            whatsapp_trust=float(media_cfg.get("whatsapp_trust", 0.85)),
            whatsapp_forward_prob=float(media_cfg.get("whatsapp_forward_prob", 0.25)),
        )
        elec_cfg = self.config.get("electoral", {})
        self.electoral_engine = ElectoralEngine(
            n_candidates=self.n_candidates,
            n_dims=self.n_dims,
            indifference_threshold=float(elec_cfg.get("indifference_threshold", -1.65)),
            rationality_lambda=float(elec_cfg.get("rationality_lambda", 4.0)),
            noise_scale=float(elec_cfg.get("noise_scale", 0.12)),
        )
        self.analytics_engine = AnalyticsEngine(
            er_alpha=float(sim_cfg.get("er_alpha", 1.3)),
            er_bins=int(sim_cfg.get("er_bins", 4)),
        )

        self.storage: HPCStorage | None = None
        if self.output_dir is not None:
            io_cfg = self.config.get("io", {})
            self.storage = HPCStorage(
                output_dir=self.output_dir,
                task_id=self.task_id,
                n_agents=self.n_agents,
                n_dims=self.n_dims,
                hdf5_compression=str(io_cfg.get("hdf5_compression", "gzip")),
                buffer_size=int(io_cfg.get("buffer_size", 10)),
            )

    def run(self) -> dict:
        """Execute the full simulation loop, recording trajectories and metrics to HPC storage."""
        if self.storage is not None:
            self.storage.initialize(
                self.agent_state,
                metadata={
                    "base_seed": self.base_seed,
                    "task_id": self.task_id,
                    "n_agents": self.n_agents,
                    "n_dims": self.n_dims,
                    "n_steps": self.n_steps,
                },
            )

        history: list[dict] = []
        last_electoral: dict = {}

        for step in range(self.n_steps + 1):
            if step > 0:
                self.dynamics_kernel.step_interpersonal(
                    self.agent_state, self.graph_engine, self.rng_dynamics
                )
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
            )
            step_metrics["task_id"] = self.task_id
            history.append(step_metrics)

            if self.storage is not None:
                save_traj = (step % self.snapshot_interval == 0) or (step == self.n_steps)
                self.storage.record_step(
                    step=step,
                    agent_state=self.agent_state,
                    metrics=step_metrics,
                    save_trajectory=save_traj,
                )

        hdf5_path = None
        parquet_path = None
        if self.storage is not None:
            hdf5_path, parquet_path = self.storage.finalize(
                self.agent_state, news_pool=self.news_pool
            )

        return {
            "task_id": self.task_id,
            "final_metrics": history[-1],
            "electoral_summary": last_electoral,
            "history": history,
            "hdf5_path": hdf5_path,
            "parquet_path": parquet_path,
        }
