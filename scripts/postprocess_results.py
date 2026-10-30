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


def consolidate_all_trad_media_groups(
    scratch_root: str | Path = "/scratch/global/crcosta/omni_opinion",
    output_dir: str | Path = "/scratch/global/crcosta/omni_opinion/consolidated",
) -> list[Path]:
    """Scan all completed traditional media vehicle group directories (`trad_media/<scale>_g<M>`)
    plus the baseline (`local_nomedia/<scale>` as M=0) and build `all_trad_media_metrics.parquet`.
    """
    root_path = Path(scratch_root)
    trad_base = root_path / "trad_media"
    nomedia_base = root_path / "local_nomedia"
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written_files: list[Path] = []
    all_tables = []

    for scale_name in CITY_SCALES:
        base_dir = nomedia_base / scale_name
        if base_dir.exists() and list(base_dir.glob("metrics_task_*.parquet")):
            out_0 = out_dir / f"consolidated_{scale_name}_g0.parquet"
            consolidate_parquet_shard_files(base_dir, out_0, city_scale_label=scale_name)
            tbl_0 = pq.read_table(out_0)
            if "n_vehicle_groups" not in tbl_0.column_names:
                tbl_0 = tbl_0.append_column(
                    "n_vehicle_groups", pa.array([0] * tbl_0.num_rows, type=pa.int64())
                )
            all_tables.append(tbl_0)

        for m_groups in (1, 2, 3, 4):
            g_dir = trad_base / f"{scale_name}_g{m_groups}"
            if not g_dir.exists():
                continue
            shards = sorted(g_dir.glob("metrics_task_*.parquet"))
            if not shards:
                continue
            out_file = out_dir / f"consolidated_{scale_name}_g{m_groups}.parquet"
            consolidate_parquet_shard_files(g_dir, out_file, city_scale_label=scale_name)
            tbl = pq.read_table(out_file)
            if "n_vehicle_groups" not in tbl.column_names:
                tbl = tbl.append_column(
                    "n_vehicle_groups",
                    pa.array([m_groups] * tbl.num_rows, type=pa.int64()),
                )
            pq.write_table(tbl, out_file, compression="zstd")
            written_files.append(out_file)
            all_tables.append(tbl)
            print(
                f"[OK] Consolidated {len(shards)} seeds for {scale_name} (M={m_groups} groups) -> {out_file}"
            )

    if all_tables:
        master_file = out_dir / "all_trad_media_metrics.parquet"
        master_tbl = pa.concat_tables(all_tables, promote_options="default")
        pq.write_table(master_tbl, master_file, compression="zstd")
        written_files.append(master_file)
        print(
            f"[OK] Master traditional media table written -> {master_file} ({master_tbl.num_rows} rows)"
        )

    return written_files


def consolidate_all_tendentiousness_scenarios(
    scratch_root: str | Path = "/scratch/global/crcosta/omni_opinion",
    output_dir: str | Path = "/scratch/global/crcosta/omni_opinion/consolidated",
) -> list[Path]:
    """Scan all completed tendentiousness scenario directories (`tendentious/*`)
    and build `all_tendentiousness_metrics.parquet`.
    """
    root_path = Path(scratch_root)
    tend_base = root_path / "tendentious"
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written_files: list[Path] = []
    all_tables = []

    if not tend_base.exists():
        return written_files

    for scen_dir in sorted(tend_base.iterdir()):
        if not scen_dir.is_dir():
            continue
        shards = sorted(scen_dir.glob("metrics_task_*.parquet"))
        if not shards:
            continue
        scen_name = scen_dir.name
        out_file = out_dir / f"consolidated_tend_{scen_name}.parquet"
        consolidate_parquet_shard_files(scen_dir, out_file, city_scale_label="small_10k")
        tbl = pq.read_table(out_file)
        if "scenario_name" not in tbl.column_names:
            tbl = tbl.append_column(
                "scenario_name",
                pa.array([scen_name] * tbl.num_rows, type=pa.string()),
            )
        pq.write_table(tbl, out_file, compression="zstd")
        written_files.append(out_file)
        all_tables.append(tbl)
        print(f"[OK] Consolidated {len(shards)} seeds for tendentiousness scenario {scen_name} -> {out_file}")

    if all_tables:
        master_file = out_dir / "all_tendentiousness_metrics.parquet"
        master_tbl = pa.concat_tables(all_tables, promote_options="default")
        pq.write_table(master_tbl, master_file, compression="zstd")
        written_files.append(master_file)
        print(
            f"[OK] Master tendentiousness table written -> {master_file} ({master_tbl.num_rows} rows)"
        )

    return written_files


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Consolidate SLURM Parquet metric outputs into compact summary tables for local download."
    )
    parser.add_argument(
        "--all-cities",
        action="store_true",
        help="Automatically consolidate all completed city scales in /scratch/global/crcosta/omni_opinion/local_nomedia, trad_media, and tendentious.",
    )
    parser.add_argument(
        "--all-trad-media",
        action="store_true",
        help="Consolidate all completed traditional media vehicle group runs (M=0,1,2,3,4) and tendentiousness scenarios.",
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
        out_dir = Path(args.scratch_base).parent / "consolidated"
        consolidate_all_city_scales(
            scratch_base=args.scratch_base,
            output_dir=out_dir,
        )
        consolidate_all_trad_media_groups(
            scratch_root=Path(args.scratch_base).parent,
            output_dir=out_dir,
        )
        consolidate_all_tendentiousness_scenarios(
            scratch_root=Path(args.scratch_base).parent,
            output_dir=out_dir,
        )
    elif args.all_trad_media:
        out_dir = Path(args.scratch_base).parent / "consolidated"
        consolidate_all_trad_media_groups(
            scratch_root=Path(args.scratch_base).parent,
            output_dir=out_dir,
        )
        consolidate_all_tendentiousness_scenarios(
            scratch_root=Path(args.scratch_base).parent,
            output_dir=out_dir,
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


