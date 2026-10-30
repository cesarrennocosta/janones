#!/usr/bin/env python3
"""Post-process and consolidate Parquet metric files from a SLURM job array."""

from __future__ import annotations

import argparse
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq


def consolidate_parquet_Shard_files(input_dir: str | Path, output_file: str | Path) -> Path:
    """Read all `metrics_task_*.parquet` files in `input_dir` and write a consolidated dataset."""
    in_path = Path(input_dir)
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    shard_files = sorted(in_path.glob("metrics_task_*.parquet"))
    if not shard_files:
        raise FileNotFoundError(f"No Parquet metric shards found in {in_path}")

    tables = [pq.read_table(f) for f in shard_files]
    combined = pa.concat_tables(tables, promote_options="default")
    pq.write_table(combined, out_path, compression="zstd")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Consolidate SLURM array Parquet metric outputs into a single summary table."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="/scratch/global/crcosta/omni_opinion/slurm_runs",
        help="Directory containing metrics_task_*.parquet shards.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="/scratch/global/crcosta/omni_opinion/consolidated_metrics.parquet",
        help="Output path for consolidated Apache Parquet table.",
    )
    args = parser.parse_args()
    out = consolidate_parquet_Shard_files(args.input_dir, args.output_file)
    print(f"Consolidated Parquet metrics written to: {out}")


if __name__ == "__main__":
    main()
