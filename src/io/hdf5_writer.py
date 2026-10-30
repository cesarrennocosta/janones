"""HDF5TrajectoryWriter: Buffered HDF5 writer with compression and preemption-safe checkpointing."""

from __future__ import annotations

from pathlib import Path
import h5py
import numpy as np


class HDF5TrajectoryWriter:
    """In-memory buffered HDF5 writer for agent opinion trajectories, static attributes,
    and mid-simulation preemption checkpoints (`--qos=preempt_strict`).

    Strictly avoids generating small CSV/JSON files per round or per agent.
    """

    def __init__(
        self,
        filepath: str | Path,
        n_agents: int,
        n_dims: int,
        compression: str = "gzip",
        buffer_size: int = 10,
        save_full_trajectory: bool = True,
    ) -> None:
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.n_agents = int(n_agents)
        self.n_dims = int(n_dims)
        self.compression = compression if compression in ("gzip", "lzf") else "gzip"
        self.buffer_size = max(1, int(buffer_size))
        self.save_full_trajectory = bool(save_full_trajectory)

        self._step_buffer: list[int] = []
        self._opinion_buffer: list[np.ndarray] = []
        self._uncertainty_buffer: list[np.ndarray] = []
        self._initialized = False

    def load_checkpoint_if_exists(self, agent_state) -> int:
        """Check if a valid mid-simulation checkpoint exists in the HDF5 file.
        If found, restore mutable agent state in-place and return the completed step index.
        Otherwise return -1.
        """
        if not self.filepath.exists():
            return -1

        try:
            with h5py.File(self.filepath, "r") as h5f:
                if "checkpoint" not in h5f:
                    return -1
                chk = h5f["checkpoint"]
                if "last_completed_step" not in chk.attrs:
                    return -1
                last_step = int(chk.attrs["last_completed_step"])
                agent_state.opinions[:] = chk["opinions"][:]
                agent_state.uncertainty[:] = chk["uncertainty"][:]
                agent_state.exposure_count[:] = chk["exposure_count"][:]
                agent_state.engagement_rate[:] = chk["engagement_rate"][:]
                self._initialized = True
                return last_step
        except Exception:
            return -1

    def save_checkpoint(self, step: int, agent_state) -> None:
        """Save current mutable agent state in HDF5 `/checkpoint` group for preemption recovery."""
        self.flush()
        with h5py.File(self.filepath, "a") as h5f:
            chk = h5f.require_group("checkpoint")
            for name, arr in [
                ("opinions", agent_state.opinions),
                ("uncertainty", agent_state.uncertainty),
                ("exposure_count", agent_state.exposure_count),
                ("engagement_rate", agent_state.engagement_rate),
            ]:
                if name in chk:
                    chk[name][...] = arr
                else:
                    chk.create_dataset(name, data=arr, compression=self.compression)
            chk.attrs["last_completed_step"] = int(step)

    def write_static_metadata(self, agent_state, metadata: dict | None = None) -> None:
        """Write fixed agent attributes, cognitive coupling matrix, and simulation metadata once."""
        with h5py.File(self.filepath, "a") as h5f:
            if metadata:
                for k, v in metadata.items():
                    h5f.attrs[str(k)] = v

            static_grp = h5f.require_group("static")
            for name, arr in [
                ("age", agent_state.age),
                ("gender", agent_state.gender),
                ("ses", agent_state.ses),
                ("district", agent_state.district),
                ("innate_beliefs", agent_state.innate_beliefs),
                ("susceptibility", agent_state.susceptibility),
                ("epsilon", agent_state.epsilon),
                ("assimilation_bias", agent_state.assimilation_bias),
                ("is_influencer", agent_state.is_influencer),
                ("coupling_matrix", agent_state.coupling_matrix),
            ]:
                if name not in static_grp:
                    static_grp.create_dataset(
                        name,
                        data=arr,
                        compression=self.compression,
                    )

            if self.save_full_trajectory:
                traj_grp = h5f.require_group("trajectories")
                if "steps" not in traj_grp:
                    traj_grp.create_dataset(
                        "steps",
                        shape=(0,),
                        maxshape=(None,),
                        dtype=np.int32,
                        chunks=(max(1, self.buffer_size),),
                        compression=self.compression,
                    )
                    traj_grp.create_dataset(
                        "opinions",
                        shape=(0, self.n_agents, self.n_dims),
                        maxshape=(None, self.n_agents, self.n_dims),
                        dtype=np.float32,
                        chunks=(1, self.n_agents, self.n_dims),
                        compression=self.compression,
                    )
                    traj_grp.create_dataset(
                        "uncertainty",
                        shape=(0, self.n_agents),
                        maxshape=(None, self.n_agents),
                        dtype=np.float32,
                        chunks=(1, self.n_agents),
                        compression=self.compression,
                    )
        self._initialized = True

    def append_step(
        self,
        step: int,
        opinions: np.ndarray,
        uncertainty: np.ndarray,
    ) -> None:
        """Buffer a trajectory snapshot in memory and flush to HDF5 when buffer is full."""
        if not self.save_full_trajectory:
            return

        self._step_buffer.append(int(step))
        self._opinion_buffer.append(opinions.astype(np.float32, copy=True))
        self._uncertainty_buffer.append(uncertainty.astype(np.float32, copy=True))

        if len(self._step_buffer) >= self.buffer_size:
            self.flush()

    def flush(self) -> None:
        """Flush in-memory trajectory buffers to the HDF5 file on disk."""
        if not self._step_buffer or not self.save_full_trajectory:
            return

        n_new = len(self._step_buffer)
        steps_arr = np.asarray(self._step_buffer, dtype=np.int32)
        ops_arr = np.stack(self._opinion_buffer, axis=0)
        unc_arr = np.stack(self._uncertainty_buffer, axis=0)

        with h5py.File(self.filepath, "a") as h5f:
            traj_grp = h5f.require_group("trajectories")
            if "steps" not in traj_grp:
                traj_grp.create_dataset(
                    "steps",
                    shape=(0,),
                    maxshape=(None,),
                    dtype=np.int32,
                    chunks=(max(1, self.buffer_size),),
                    compression=self.compression,
                )
                traj_grp.create_dataset(
                    "opinions",
                    shape=(0, self.n_agents, self.n_dims),
                    maxshape=(None, self.n_agents, self.n_dims),
                    dtype=np.float32,
                    chunks=(1, self.n_agents, self.n_dims),
                    compression=self.compression,
                )
                traj_grp.create_dataset(
                    "uncertainty",
                    shape=(0, self.n_agents),
                    maxshape=(None, self.n_agents),
                    dtype=np.float32,
                    chunks=(1, self.n_agents),
                    compression=self.compression,
                )

            ds_steps = traj_grp["steps"]
            ds_ops = traj_grp["opinions"]
            ds_unc = traj_grp["uncertainty"]

            old_len = ds_steps.shape[0]
            new_len = old_len + n_new

            ds_steps.resize((new_len,))
            ds_ops.resize((new_len, self.n_agents, self.n_dims))
            ds_unc.resize((new_len, self.n_agents))

            ds_steps[old_len:new_len] = steps_arr
            ds_ops[old_len:new_len] = ops_arr
            ds_unc[old_len:new_len] = unc_arr

        self._step_buffer.clear()
        self._opinion_buffer.clear()
        self._uncertainty_buffer.clear()

    def write_final_state(self, agent_state, news_pool=None) -> None:
        """Flush remaining trajectory steps and write final electoral & news entity states."""
        self.flush()
        with h5py.File(self.filepath, "a") as h5f:
            final_grp = h5f.require_group("final_state")
            for name, arr in [
                ("opinions", agent_state.opinions),
                ("uncertainty", agent_state.uncertainty),
                ("exposure_count", agent_state.exposure_count),
                ("engagement_rate", agent_state.engagement_rate),
                ("vote_probs", agent_state.vote_probs),
                ("vote_choice", agent_state.vote_choice),
            ]:
                if name in final_grp:
                    del final_grp[name]
                final_grp.create_dataset(name, data=arr, compression=self.compression)

            if news_pool is not None:
                news_grp = h5f.require_group("news_entities")
                for name, arr in [
                    ("framing", news_pool.framing),
                    ("sensationalism", news_pool.sensationalism),
                    ("veracity", news_pool.veracity),
                    ("popularity", news_pool.popularity),
                    ("viral_cascades", news_pool.viral_cascades),
                ]:
                    if name in news_grp:
                        del news_grp[name]
                    news_grp.create_dataset(name, data=arr, compression=self.compression)
