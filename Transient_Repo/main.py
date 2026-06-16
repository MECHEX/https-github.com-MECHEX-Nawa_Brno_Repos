# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd


def _configure_runtime() -> None:
    # Default to a non-interactive backend so Windows runs do not depend on Tk.
    os.environ.setdefault("MPLBACKEND", "Agg")


_configure_runtime()

from compare_runner import run_compare_jobs
from compute import build_bands_df, build_planes_df, compute_band_table, compute_global_means
from config import (
    CASES,
    COMPARE_JOBS,
    DEFAULT_BASE_DATA_DIR,
    DEFAULT_OUT_DIR,
    FLUID_CFG,
    PLOT_DEFAULTS,
    PLOT_JOBS,
    SRP_DEFAULTS,
)
from config_resolver import RuntimeConfig, resolve_runtime_config
from constants import COL_Y_M, COL_Z_M
from plotting import (
    make_overlay_mean_std_per_fluid,
    make_overlays_per_fluid,
    plot_means_vs_time,
    set_plot_defaults,
)
from srp_parser import parse_srp


STEP_RE = re.compile(r"_S(\d{4,})", re.IGNORECASE)
TIME_RE = re.compile(r"_T([0-9]*\.?[0-9]+(?:[eE][+-]?\d+)?)", re.IGNORECASE)


class DataIndexingError(RuntimeError):
    pass


class PlotJobError(RuntimeError):
    pass


def _assert_writable_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    probe = path / ".write_probe.tmp"
    try:
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        raise PermissionError(
            f"Output directory is not writable: {path}. "
            "Pass --out-dir to a writable location or fix directory permissions."
        ) from exc


def _sanitize_name(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text).strip())
    return cleaned.strip("._") or "job"


def _extract_step(stem: str) -> int:
    match = STEP_RE.search(stem)
    return int(match.group(1)) if match else -1


def _extract_time_s(stem: str) -> Optional[float]:
    match = TIME_RE.search(stem)
    if not match:
        return None
    token = match.group(1).replace(",", ".")
    try:
        return float(token)
    except ValueError:
        return None


def _extract_t_local(part_meta: Dict[str, Any], stem: str) -> Optional[float]:
    t_start = float(part_meta["t_start_s"])
    t_end = float(part_meta["t_end_s"])
    dt_sim_s = float(part_meta["dt_sim_s"])
    fixed_t_local_s = part_meta.get("fixed_t_local_s")
    t_len = t_end - t_start

    def interpret_from_token(value_t: float) -> float:
        mode = str(part_meta.get("t_interpretation", "global")).lower()
        if mode == "local":
            return value_t
        if mode == "global":
            return value_t - t_start
        if mode == "auto":
            if (-1e-12) <= value_t <= (t_len + 1e-12):
                return value_t
            return value_t - t_start
        return value_t

    for source in part_meta.get("time_source_priority", ("T", "S")):
        source_name = str(source).upper()
        if source_name == "T":
            token_t = _extract_time_s(stem)
            if token_t is not None:
                return interpret_from_token(token_t)
        elif source_name == "S":
            step = _extract_step(stem)
            if step >= 0:
                return step * dt_sim_s
    if fixed_t_local_s is not None:
        return float(fixed_t_local_s)
    return None


def _fluid_from_name(stem: str) -> str:
    lowered = stem.lower()
    if "fluid1" in lowered or "_f1_" in lowered or lowered.endswith("_f1") or "f1_" in lowered:
        return "Fluid1"
    if "fluid2" in lowered or "_f2_" in lowered or lowered.endswith("_f2") or "f2_" in lowered:
        return "Fluid2"
    return "Fluid2" if "_f2" in lowered else "Fluid1"


def _normalize_prefixes(text: str) -> str:
    normalized = re.sub(r"\bf[12]_wall_band_", "wall_band_", text)
    normalized = re.sub(r"\bf[12]_(?:ziso_|yiso_)", "ziso_", normalized)
    normalized = re.sub(r"\bf[12]_env_", "env_", normalized)
    return normalized


def _filter_by_local_window(
    items: List[Tuple[float, str, Path]],
    t_len: float,
) -> List[Tuple[float, str, Path]]:
    return [item for item in items if (-1e-12) <= item[0] <= (t_len + 1e-12)]


def _collect_required_part_keys(runtime_cfg: RuntimeConfig) -> Set[str]:
    required: Set[str] = set()
    for job in runtime_cfg.plot_jobs:
        required.update(job["part_keys"])
    for job in runtime_cfg.compare_jobs:
        for series_cfg in job.get("series", []):
            required.update(series_cfg.get("parts", []))
    return required


def _collect_all_data(
    *,
    part_catalog: Dict[str, Dict[str, Any]],
    required_part_keys: Set[str],
    verbose: bool,
) -> Dict[str, Dict[str, List[Tuple[float, str, Path]]]]:
    out: Dict[str, Dict[str, List[Tuple[float, str, Path]]]] = {}

    if not required_part_keys:
        return out

    for part_key, meta in part_catalog.items():
        if part_key not in required_part_keys:
            continue

        source_dir = Path(meta["source_dir"])
        srp_files = sorted(source_dir.glob("*.srp"))
        if not srp_files:
            msg = (
                f"Configured part '{part_key}' points to '{source_dir}', "
                "but the directory contains no .srp files."
            )
            if part_key in required_part_keys:
                raise DataIndexingError(msg)
            if verbose:
                print(f"[WARN] {msg}")
            continue

        indexed_count = 0
        skipped_no_time = 0

        for srp_path in srp_files:
            try:
                t_local = _extract_t_local(meta, srp_path.stem)
            except Exception as exc:
                raise DataIndexingError(
                    "Failed to extract transient time from "
                    f"'{srp_path.name}' for part '{part_key}'."
                ) from exc

            if t_local is None:
                skipped_no_time += 1
                continue

            fluid = _fluid_from_name(srp_path.stem)
            out.setdefault(part_key, {}).setdefault(fluid, []).append((t_local, part_key, srp_path))
            indexed_count += 1

        if indexed_count == 0:
            msg = (
                f"Configured part '{part_key}' from '{source_dir}' has {len(srp_files)} SRP files, "
                "but none produced a recognizable time value. "
                f"time_source_priority={tuple(meta.get('time_source_priority', ()))}, "
                f"t_interpretation={meta.get('t_interpretation')!r}."
            )
            if part_key in required_part_keys:
                raise DataIndexingError(msg)
            if verbose:
                print(f"[WARN] {msg}")
            continue

        if verbose and skipped_no_time:
            print(
                f"[WARN] Part '{part_key}' skipped {skipped_no_time} SRP files without "
                "a recognizable time token."
            )

    for part_key in out:
        for fluid in out[part_key]:
            out[part_key][fluid].sort(key=lambda item: item[0])

    return out


def _build_bands_and_means(path: Path, fluid: str) -> Tuple[pd.DataFrame, Dict[str, float]]:
    cfg = FLUID_CFG[fluid]
    axis_label = str(cfg["axis"])
    step_abs_len = float(cfg["step"])
    axis_min = float(cfg["min"])
    axis_max = float(cfg["max"])
    sign = 1.0 if axis_max >= axis_min else -1.0

    try:
        raw = path.read_text(encoding="utf-8", errors="ignore")
    except OSError as exc:
        raise RuntimeError(f"Failed to read SRP file '{path}'.") from exc

    normalized = _normalize_prefixes(raw)

    with tempfile.NamedTemporaryFile("w", suffix=".srp", delete=False, encoding="utf-8") as handle:
        handle.write(normalized)
        tmp_path = Path(handle.name)

    try:
        try:
            data = parse_srp(tmp_path)
            planes_df = build_planes_df(data, dz=step_abs_len)
            bands_df = build_bands_df(data)
            band_tab = compute_band_table(planes_df, bands_df, dz=step_abs_len, axis_label=axis_label)
            means = compute_global_means(band_tab)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to parse or compute transient data for fluid '{fluid}' from '{path.name}'."
            ) from exc
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    n_bands = len(band_tab)
    plane_coords = [axis_min + sign * i * step_abs_len for i in range(n_bands + 1)]
    centers = [(plane_coords[i] + plane_coords[i + 1]) * 0.5 for i in range(n_bands)]
    axis_column = f"{axis_label} [m]"
    for old_name in (COL_Z_M, COL_Y_M):
        if old_name in band_tab.columns and old_name != axis_column:
            band_tab.drop(columns=[old_name], inplace=True)
    band_tab[axis_column] = centers

    return band_tab, means


def _default_path(configured: Optional[str], fallback: Path) -> str:
    text = str(configured).strip() if configured else ""
    return text or str(fallback)


def _job_tag(job_name: str, fluid: str) -> str:
    fluid_tag = "F1" if fluid == "Fluid1" else "F2"
    return f"{_sanitize_name(job_name)}__{fluid_tag}"


def _format_dt_tag(dt_values: Iterable[float]) -> Optional[str]:
    unique = sorted({float(dt) for dt in dt_values})
    if len(unique) != 1:
        return None
    return f"{unique[0]:.0e}s".replace("+0", "")


def main() -> None:
    repo_dir = Path(__file__).resolve().parent
    default_base_dir = _default_path(DEFAULT_BASE_DATA_DIR, repo_dir / "FluentTransientData")
    default_out_dir = _default_path(DEFAULT_OUT_DIR, repo_dir / "TransientFigs")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in-dir",
        type=str,
        default=default_base_dir,
        help="Base directory with transient SRP subdirectories.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=default_out_dir,
        help="Output directory for CSV, plots, and Compare results.",
    )
    parser.add_argument(
        "--jobs",
        nargs="+",
        default=None,
        help="Optional active PLOT_JOB/COMPARE_JOB names to run.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    base_dir = Path(args.in_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    csv_dir = out_dir / "csv"
    plots_dir = out_dir / "plots"

    runtime_cfg = resolve_runtime_config(
        base_data_dir=base_dir,
        cases=CASES,
        plot_jobs=PLOT_JOBS,
        compare_jobs=COMPARE_JOBS,
        srp_defaults=SRP_DEFAULTS,
    )
    if args.jobs:
        selected_jobs = set(args.jobs)
        selected_plot_jobs = [job for job in runtime_cfg.plot_jobs if job["name"] in selected_jobs]
        selected_compare_jobs = [job for job in runtime_cfg.compare_jobs if job["name"] in selected_jobs]
        matched_jobs = {job["name"] for job in selected_plot_jobs + selected_compare_jobs}
        missing_jobs = sorted(selected_jobs - matched_jobs)
        if missing_jobs:
            raise PlotJobError(f"Unknown or inactive job(s) requested with --jobs: {', '.join(missing_jobs)}")
        runtime_cfg = RuntimeConfig(
            part_catalog=runtime_cfg.part_catalog,
            case_parts=runtime_cfg.case_parts,
            plot_jobs=selected_plot_jobs,
            compare_jobs=selected_compare_jobs,
        )

    _assert_writable_dir(out_dir)
    _assert_writable_dir(csv_dir)
    _assert_writable_dir(plots_dir)

    set_plot_defaults(
        dpi=int(PLOT_DEFAULTS.get("dpi", 160)),
        marker_size=float(PLOT_DEFAULTS.get("marker_size", 2.0)),
        line_width=float(PLOT_DEFAULTS.get("line_width", 0.8)),
    )

    required_part_keys = _collect_required_part_keys(runtime_cfg)
    if not required_part_keys:
        print("[INFO] No active PLOT_JOBS or COMPARE_JOBS. Nothing to process.")
        return

    index = _collect_all_data(
        part_catalog=runtime_cfg.part_catalog,
        required_part_keys=required_part_keys,
        verbose=bool(args.verbose),
    )
    cache_bands: Dict[Tuple[str, str, Path], pd.DataFrame] = {}
    cache_means: Dict[Tuple[str, str, Path], Dict[str, float]] = {}

    for job in runtime_cfg.plot_jobs:
        job_name = str(job["name"])
        part_keys = list(job["part_keys"])
        fluids = list(job["fluids"])
        plots = list(job["plots"])
        metrics = list(job["metrics"])

        if not part_keys:
            raise PlotJobError(f"PLOT_JOBS '{job_name}' resolves to no parts.")

        part_meta_list = [runtime_cfg.part_catalog[part_key] for part_key in part_keys]
        global_t_min = min(float(meta["t_start_s"]) for meta in part_meta_list)
        global_t_max = max(float(meta["t_end_s"]) for meta in part_meta_list)
        dt_tag = _format_dt_tag(float(meta["dt_sim_s"]) for meta in part_meta_list)

        job_plot_dir = plots_dir / _sanitize_name(job_name)
        job_csv_dir = csv_dir / _sanitize_name(job_name)
        _assert_writable_dir(job_plot_dir)
        _assert_writable_dir(job_csv_dir)

        overlay_every = int(PLOT_DEFAULTS.get("overlay_every", 1))
        include_pressure = bool(PLOT_DEFAULTS.get("include_pressure", False))
        overlay_mode = str(job.get("overlay_mode", "all")).lower()

        for fluid in fluids:
            overlays_seq_triplets: List[Tuple[float, pd.DataFrame, str]] = []
            summary_rows: List[Dict[str, Any]] = []

            for part_key in part_keys:
                meta = runtime_cfg.part_catalog[part_key]
                t_start = float(meta["t_start_s"])
                t_end = float(meta["t_end_s"])
                t_len = t_end - t_start
                part_label = f"{part_key} ({meta['source_dir_name']})"

                if t_len < 0:
                    raise PlotJobError(
                        f"PLOT_JOBS '{job_name}' references part '{part_label}' with a negative time range."
                    )

                if part_key not in index:
                    raise PlotJobError(
                        f"PLOT_JOBS '{job_name}' could not find indexed SRP data for part '{part_label}'."
                    )

                if fluid not in index[part_key]:
                    raise PlotJobError(
                        f"PLOT_JOBS '{job_name}' has no indexed data for fluid '{fluid}' in part '{part_label}'."
                    )

                items = _filter_by_local_window(index[part_key][fluid], t_len)
                if not items:
                    raise PlotJobError(
                        f"PLOT_JOBS '{job_name}' found no snapshots for fluid '{fluid}' in part "
                        f"'{part_label}' within local window [0, {t_len}] s."
                    )

                for t_local, indexed_part_key, path in items:
                    key = (indexed_part_key, fluid, path)
                    if key not in cache_bands:
                        try:
                            band_tab, means = _build_bands_and_means(path, fluid)
                        except Exception as exc:
                            raise PlotJobError(
                                f"PLOT_JOBS '{job_name}' failed while processing fluid '{fluid}', "
                                f"part '{part_label}', file '{path.name}'."
                            ) from exc
                        cache_bands[key] = band_tab
                        cache_means[key] = means
                    else:
                        band_tab = cache_bands[key]
                        means = cache_means[key]

                    t_global = t_start + t_local
                    overlays_seq_triplets.append((t_global, band_tab, f"t={t_global:.3f}s"))
                    row = {"Fluid": fluid, "Time[s]": t_global}
                    row.update(means)
                    summary_rows.append(row)

            overlays_seq_triplets.sort(key=lambda item: item[0])
            overlays_seq = [(df, label) for (_time_global, df, label) in overlays_seq_triplets]
            df_means = (
                pd.DataFrame(summary_rows).sort_values(["Fluid", "Time[s]"]).reset_index(drop=True)
                if summary_rows
                else pd.DataFrame()
            )

            if not overlays_seq and df_means.empty:
                raise PlotJobError(
                    f"PLOT_JOBS '{job_name}' produced no data for fluid '{fluid}' "
                    f"in global window [{global_t_min}, {global_t_max}] s."
                )

            tag = _job_tag(job_name, fluid)
            if dt_tag:
                tag = f"{tag}__{dt_tag}"

            if "overlay" in plots and overlays_seq:
                try:
                    if overlay_mode == "mean_std":
                        make_overlay_mean_std_per_fluid(
                            overlays_seq=overlays_seq,
                            fluid_name=fluid,
                            out_dir=job_plot_dir,
                            metrics=metrics,
                            include_pressure=include_pressure,
                            overlay_every=overlay_every,
                            outfile_tag=tag,
                        )
                    else:
                        make_overlays_per_fluid(
                            overlays_seq=overlays_seq,
                            fluid_name=fluid,
                            out_dir=job_plot_dir,
                            metrics=metrics,
                            include_pressure=include_pressure,
                            overlay_every=overlay_every,
                            outfile_tag=tag,
                        )
                except Exception as exc:
                    raise PlotJobError(
                        f"PLOT_JOBS '{job_name}' failed while rendering overlay plots for fluid '{fluid}'."
                    ) from exc

            if "mean" in plots and not df_means.empty:
                try:
                    plot_means_vs_time(
                        df=df_means,
                        fluid_name=fluid,
                        out_dir=job_plot_dir,
                        metrics=metrics,
                        include_pressure=include_pressure,
                        outfile_tag=tag,
                    )
                except Exception as exc:
                    raise PlotJobError(
                        f"PLOT_JOBS '{job_name}' failed while rendering mean plots for fluid '{fluid}'."
                    ) from exc

            if not df_means.empty:
                csv_means = job_csv_dir / f"summary_means__{tag}.csv"
                df_means.to_csv(csv_means, index=False)
                print(f"[OK] CSV -> {csv_means}")

    compare_root = out_dir / "Compare"
    _assert_writable_dir(compare_root)
    run_compare_jobs(
        compare_jobs=runtime_cfg.compare_jobs,
        compare_root_dir=compare_root,
        PARTS=runtime_cfg.part_catalog,
        index=index,
        cache_bands=cache_bands,
        cache_means=cache_means,
        build_one=_build_bands_and_means,
        repo_root=repo_dir,
    )


if __name__ == "__main__":
    main()
