"""ParquetMetricsWriter & HPCStorage: Tabular Apache Parquet writer using pyarrow."""

from __future__ import annotations

from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

from src.io.hdf5_writer import HDF5TrajectoryWriter


class ParquetMetricsWriter:
    """In-memory buffered writer that persists time series of aggregate simulation metrics
    to Apache Parquet (`pyarrow`) with Snappy/ZSTD compression.
    """

    def __init__(self, filepath: str | Path, compression: str = "snappy") -> None:
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.compression = compression
        self._records: list[dict] = []

    def append_metrics(self, metrics_dict: dict) -> None:
        """Buffer one timestep of aggregate metrics in memory."""
        self._records.append(dict(metrics_dict))

    def flush(self) -> Path:
        """Write all buffered metric rows as a single columnar Apache Parquet file."""
        if not self._records:
            return self.filepath

        table = pa.Table.from_pylist(self._records)
        pq.write_table(table, self.filepath, compression=self.compression)
        return self.filepath

    @property
    def records(self) -> list[dict]:
        return self._records


class HPCStorage:
    """Unified HPC storage facade coordinating HDF5 trajectory output and Parquet metrics output."""

    def __init__(
        self,
        output_dir: str | Path,
        task_id: int,
        n_agents: int,
        n_dims: int,
        hdf5_compression: str = "gzip",
        buffer_size: int = 10,
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
        )
        self.parquet_writer = ParquetMetricsWriter(filepath=self.parquet_path)

    def initialize(self, agent_state, metadata: dict | None = None) -> None:
        self.hdf5_writer.write_static_metadata(agent_state, metadata=metadata)

    def record_step(
        self,
        step: int,
        agent_state,
        metrics: dict,
        save_trajectory: bool = True,
    ) -> None:
        if save_trajectory:
            self.hdf5_writer.append_step(step, agent_state.opinions, agent_state.uncertainty)
        self.parquet_writer.append_metrics(metrics)

    def finalize(self, agent_state, news_pool=None) -> tuple[Path, Path]:
        self.hdf5_writer.write_final_state(agent_state, news_pool=news_pool)
        self.parquet_writer.flush()
        return self.hdf5_path, self.parquet_path
