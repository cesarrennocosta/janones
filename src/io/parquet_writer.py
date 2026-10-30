"""ParquetMetricsWriter & HPCStorage: Tabular Apache Parquet writer with preemption checkpoint recovery."""

from __future__ import annotations

from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

from src.io.hdf5_writer import HDF5TrajectoryWriter


class ParquetMetricsWriter:
    """In-memory buffered writer that persists time series of aggregate simulation metrics
    to Apache Parquet (`pyarrow`) with Snappy/ZSTD compression and supports checkpoint recovery.
    """

    def __init__(self, filepath: str | Path, compression: str = "snappy") -> None:
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.compression = compression
        self._records: list[dict] = []

    def load_existing_up_to_step(self, max_step: int) -> list[dict]:
        """Load previously flushed Parquet metric rows up to `max_step` when resuming a preempted job."""
        if not self.filepath.exists() or max_step < 0:
            self._records = []
            return self._records
        try:
            table = pq.read_table(self.filepath)
            rows = table.to_pylist()
            self._records = [r for r in rows if int(r.get("step", -1)) <= max_step]
        except Exception:
            self._records = []
        return self._records

    def append_metrics(self, metrics_dict: dict) -> None:
        """Buffer one timestep of aggregate metrics in memory."""
        self._records.append(dict(metrics_dict))

    def flush(self) -> Path:
        """Write all buffered metric rows atomically to the Apache Parquet file."""
        if not self._records:
            return self.filepath

        tmp_path = self.filepath.with_suffix(".parquet.tmp")
        table = pa.Table.from_pylist(self._records)
        pq.write_table(table, tmp_path, compression=self.compression)
        tmp_path.replace(self.filepath)
        return self.filepath

    @property
    def records(self) -> list[dict]:
        return self._records


class HPCStorage:
    """Unified HPC storage facade coordinating HDF5 trajectory/checkpoint output and Parquet metrics output."""

    def __init__(
        self,
        output_dir: str | Path,
        task_id: int,
        n_agents: int,
        n_dims: int,
        hdf5_compression: str = "gzip",
        buffer_size: int = 10,
        save_full_trajectory: bool = True,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.task_id = int(task_id)

        self.hdf5_path = self.output_dir / f"trajectory_task_{self.task_id:04d}.h5"
        self.parquet_path = self.output_dir / f"metrics_task_{self.task_id:04d}.parquet"

        self.hdf5_writer = HDF5TrajectoryWriter(
            filepath=self.hdf5_path,
            n_agents=n_agents,
            n_dims=n_dims,
            compression=hdf5_compression,
            buffer_size=buffer_size,
            save_full_trajectory=save_full_trajectory,
        )
        self.parquet_writer = ParquetMetricsWriter(filepath=self.parquet_path)

    def try_resume_checkpoint(self, agent_state) -> tuple[int, list[dict]]:
        """Attempt to resume from an existing mid-simulation or finalized checkpoint on disk."""
        last_step = self.hdf5_writer.load_checkpoint_if_exists(agent_state)
        if last_step >= 0:
            history = list(self.parquet_writer.load_existing_up_to_step(last_step))
            if history:
                actual_last = int(history[-1].get("step", last_step))
                return actual_last, history
        return -1, []

    def initialize(self, agent_state, metadata: dict | None = None) -> None:
        self.hdf5_writer.write_static_metadata(agent_state, metadata=metadata)

    def record_step(
        self,
        step: int,
        agent_state,
        metrics: dict,
        save_trajectory: bool = True,
        save_checkpoint: bool = False,
    ) -> None:
        if save_trajectory:
            self.hdf5_writer.append_step(step, agent_state.opinions, agent_state.uncertainty)
        self.parquet_writer.append_metrics(metrics)
        if save_checkpoint:
            self.hdf5_writer.save_checkpoint(step, agent_state)
            self.parquet_writer.flush()

    def finalize(
        self, agent_state, news_pool=None, final_step: int | None = None
    ) -> tuple[Path, Path]:
        self.hdf5_writer.save_checkpoint(-1, agent_state)
        self.hdf5_writer.write_final_state(
            agent_state, news_pool=news_pool, final_step=final_step
        )
        self.parquet_writer.flush()
        return self.hdf5_path, self.parquet_path
