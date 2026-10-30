#!/usr/bin/env python3
"""Post-process and consolidate Parquet metric files from a SLURM job array or multi-scale city runs."""

from __future__ import annotations

import argparse
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq


CITY_SCALES = {
    "small_10k": 10_000,
    "medium_100k": 100_000,
    "large_1m": 1_000_000,
    "metropolis_10m": 10_000_000,
}


def consolidate_parquet_shard_files(
    input_dir: str | Path,
    output_file: str | Path,
    city_scale_label: str | None = None,
) -> Path:
    """Read all `metrics_task_*.parquet` files in `input_dir` and write a consolidated ZSTD Parquet dataset."""
    in_path = Path(input_dir)
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    shard_files = sorted(in_path.glob("metrics_task_*.parquet"))
    if not shard_files:
        raise FileNotFoundError(f"No Parquet metric shards found in {in_path}")

    tables = []
    for f in shard_files:
        tbl = pq.read_table(f)
        if city_scale_label is not None and "city_scale" not in tbl.column_names:
            tbl = tbl.append_column(
                "city_scale",
                pa.array([city_scale_label] * tbl.num_rows, type=pa.string()),
            )
        if "n_agents" not in tbl.column_names and city_scale_label in CITY_SCALES:
            tbl = tbl.append_column(
                "n_agents",
                pa.array([CITY_SCALES[city_scale_label]] * tbl.num_rows, type=pa.int64()),
            )
        tables.append(tbl)

    combined = pa.concat_tables(tables, promote_options="default")
    pq.write_table(combined, out_path, compression="zstd")
    return out_path


def consolidate_all_city_scales(
    scratch_base: str | Path = "/scratch/global/crcosta/omni_opinion/local_nomedia",
    output_dir: str | Path = "/scratch/global/crcosta/omni_opinion/consolidated",
) -> list[Path]:
    """Scan all completed city scale directories (`small_10k`, `medium_100k`, `large_1m`, `metropolis_10m`),
    consolidate each into `consolidated_<scale>.parquet`, and build a master `all_cities_metrics.parquet`.
    """
    base_path = Path(scratch_base)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written_files: list[Path] = []
    all_tables = []

    for scale_name in CITY_SCALES:
        scale_dir = base_path / scale_name
        if not scale_dir.exists():
            continue
        shards = sorted(scale_dir.glob("metrics_task_*.parquet"))
        if not shards:
            continue

        out_file = out_dir / f"consolidated_{scale_name}.parquet"
        consolidate_parquet_shard_files(scale_dir, out_file, city_scale_label=scale_name)
        written_files.append(out_file)
        all_tables.append(pq.read_table(out_file))
        print(f"[OK] Consolidated {len(shards)} seeds for {scale_name} -> {out_file}")

    if all_tables:
        master_file = out_dir / "all_cities_metrics.parquet"
        master_tbl = pa.concat_tables(all_tables, promote_options="default")
        pq.write_table(master_tbl, master_file, compression="zstd")
        written_files.append(master_file)
        print(f"[OK] Master multi-scale table written -> {master_file} ({master_tbl.num_rows} rows)")

    return written_files


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Consolidate SLURM Parquet metric outputs into compact summary tables for local download."
    )
    parser.add_argument(
        "--all-cities",
        action="store_true",
        help="Automatically consolidate all completed city scales in /scratch/global/crcosta/omni_opinion/local_nomedia.",
    )
    parser.add_argument(
        "--scratch-base",
        type=str,
        default="/scratch/global/crcosta/omni_opinion/local_nomedia",
        help="Base directory containing small_10k, medium_100k, large_1m, metropolis_10m folders.",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="/scratch/global/crcosta/omni_opinion/local_nomedia/small_10k",
        help="Directory containing metrics_task_*.parquet shards (when not using --all-cities).",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="/scratch/global/crcosta/omni_opinion/consolidated/consolidated_small_10k.parquet",
        help="Output path for consolidated Apache Parquet table.",
    )
    args = parser.parse_args()

    if args.all_cities:
        consolidate_all_city_scales(
            scratch_base=args.scratch_base,
            output_dir=Path(args.scratch_base).parent / "consolidated",
        )
    else:
        scale_label = Path(args.input_dir).name
        out = consolidate_parquet_shard_files(
            args.input_dir,
            args.output_file,
            city_scale_label=scale_label if scale_label in CITY_SCALES else None,
        )
        print(f"Consolidated Parquet metrics written to: {out}")


if __name__ == "__main__":
    main()
