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
    max_seeds: int | None = None,
) -> Path:
    """Read all `metrics_task_*.parquet` files in `input_dir` and write a consolidated ZSTD Parquet dataset."""
    in_path = Path(input_dir)
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    shard_files = sorted(in_path.glob("metrics_task_*.parquet"))
    if not shard_files:
        raise FileNotFoundError(f"No Parquet metric shards found in {in_path}")

    tables = []
    skipped = 0
    for f in shard_files:
        if max_seeds is not None and len(tables) >= max_seeds:
            break
        try:
            tbl = pq.read_table(f)
        except Exception:
            try:
                tbl = pq.ParquetFile(f).read()
            except Exception as exc:
                skipped += 1
                print(f"[WARN] Skipping corrupted/incomplete shard {f}: {exc}")
                continue
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

    if not tables:
        raise FileNotFoundError(f"No readable Parquet metric shards found in {in_path}")
    if skipped > 0:
        print(f"[INFO] Skipped {skipped} corrupted/incomplete shard(s) in {in_path}")

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
        g0_trad_dir = trad_base / f"{scale_name}_g0"
        base_dir = g0_trad_dir if (g0_trad_dir.exists() and list(g0_trad_dir.glob("metrics_task_*.parquet"))) else (nomedia_base / scale_name)
        if base_dir.exists() and list(base_dir.glob("metrics_task_*.parquet")):
            out_0 = out_dir / f"consolidated_{scale_name}_g0.parquet"
            consolidate_parquet_shard_files(base_dir, out_0, city_scale_label=scale_name)
            tbl_0 = pq.read_table(out_0)
            if "n_vehicle_groups" not in tbl_0.column_names:
                tbl_0 = tbl_0.append_column(
                    "n_vehicle_groups", pa.array([0] * tbl_0.num_rows, type=pa.int64())
                )
            pq.write_table(tbl_0, out_0, compression="zstd")
            written_files.append(out_0)
            all_tables.append(tbl_0)
            print(f"[OK] Consolidated {len(list(base_dir.glob('metrics_task_*.parquet')))} seeds for {scale_name} (M=0 baseline) -> {out_0}")

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
        consolidate_parquet_shard_files(scen_dir, out_file, city_scale_label="medium_100k")
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


def consolidate_scenario_suite(
    suite_dir: str | Path,
    output_dir: str | Path,
    suite_prefix: str,
    master_filename: str,
    city_scale_label: str = "medium_100k",
    max_seeds: int | None = 100,
) -> list[Path]:
    """Generic consolidator for multi-scenario suites (`social_media`, `extreme_imposed`,
    `extreme_imposed_1500runs`, `sync_dictation_100k`).
    """
    base = Path(suite_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written_files: list[Path] = []
    all_tables = []

    if not base.exists():
        return written_files

    for scen_dir in sorted(base.iterdir()):
        if not scen_dir.is_dir():
            continue
        shards = sorted(scen_dir.glob("metrics_task_*.parquet"))
        if not shards:
            continue
        scen_name = scen_dir.name
        out_file = out_dir / f"consolidated_{suite_prefix}_{scen_name}.parquet"
        consolidate_parquet_shard_files(
            scen_dir,
            out_file,
            city_scale_label=city_scale_label,
            max_seeds=max_seeds,
        )
        tbl = pq.read_table(out_file)
        if "scenario_name" not in tbl.column_names:
            tbl = tbl.append_column(
                "scenario_name",
                pa.array([scen_name] * tbl.num_rows, type=pa.string()),
            )
        pq.write_table(tbl, out_file, compression="zstd")
        written_files.append(out_file)
        all_tables.append(tbl)
        n_consolidated = len(set(tbl["task_id"].to_pylist())) if "task_id" in tbl.column_names else len(shards)
        print(f"[OK] Consolidated {n_consolidated} seeds (1500 steps) for {suite_prefix}/{scen_name} -> {out_file}")

    if all_tables:
        master_file = out_dir / master_filename
        master_tbl = pa.concat_tables(all_tables, promote_options="default")
        pq.write_table(master_tbl, master_file, compression="zstd")
        written_files.append(master_file)
        print(
            f"[OK] Master {suite_prefix} table written -> {master_file} ({master_tbl.num_rows} rows)"
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
        root_dir = Path(args.scratch_base).parent
        out_dir = root_dir / "consolidated"
        consolidate_all_city_scales(
            scratch_base=args.scratch_base,
            output_dir=out_dir,
        )
        consolidate_all_trad_media_groups(
            scratch_root=root_dir,
            output_dir=out_dir,
        )
        consolidate_all_tendentiousness_scenarios(
            scratch_root=root_dir,
            output_dir=out_dir,
        )
        consolidate_scenario_suite(
            suite_dir=root_dir / "social_media",
            output_dir=out_dir,
            suite_prefix="socmed",
            master_filename="all_social_media_metrics.parquet",
            max_seeds=100,
        )
        ext_dir = root_dir / "extreme_imposed"
        if not ext_dir.exists() or not list(ext_dir.glob("*/*.parquet")):
            ext_dir = root_dir / "extreme_imposed_1500runs"
        consolidate_scenario_suite(
            suite_dir=ext_dir,
            output_dir=out_dir,
            suite_prefix="extimp",
            master_filename="all_extreme_imposed_metrics.parquet",
            max_seeds=100,
        )
        consolidate_scenario_suite(
            suite_dir=root_dir / "sync_dictation_100k",
            output_dir=out_dir,
            suite_prefix="syncdict",
            master_filename="all_sync_dictation_metrics.parquet",
            max_seeds=100,
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


