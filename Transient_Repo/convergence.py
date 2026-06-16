# -*- coding: utf-8 -*-
"""
convergence.py — Temporal convergence analysis for transient simulations.

For each active case (or a user-specified subset), computes the spatial mean of
heat transfer coefficient (h) and Fanning friction factor (f) at every available
timestep, then plots mean_h(t) and mean_f(t).

Useful for:
  - Verifying that the simulation has reached a quasi-steady oscillating regime.
  - Justifying the time window [t0, t1] chosen for averaged comparisons.
  - Quickly spotting outlier timesteps or numerical instabilities.

Usage
-----
    cd Transient_Repo

    # All active cases, default output directory
    python convergence.py

    # Specific cases only
    python convergence.py --cases uni10_003 uni10_005 uni10_007

    # Custom directories
    python convergence.py --in-dir /path/to/FluentTransientData --out-dir /path/to/figs

    # Both fluids
    python convergence.py --fluids Fluid1 Fluid2

Output
------
    <out_dir>/convergence/conv_<metric>_<fluid>.png   — one plot per metric per fluid
    <out_dir>/convergence/conv_<metric>_<fluid>.csv   — underlying data
"""
from __future__ import annotations

import argparse
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

os.environ.setdefault("MPLBACKEND", "Agg")

from compute import build_bands_df, build_planes_df, compute_band_table, compute_global_means
from config import CASES, DEFAULT_BASE_DATA_DIR, DEFAULT_OUT_DIR, FLUID_CFG
from constants import COL_H_WM2K, COL_F_FANNING, COL_Y_M, COL_Z_M
from srp_parser import parse_srp
from tpms_utils import ensure_dir


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STEP_RE = re.compile(r"_S(\d{4,})", re.IGNORECASE)
TIME_RE = re.compile(r"_T([0-9]*\.?[0-9]+(?:[eE][+-]?\d+)?)", re.IGNORECASE)

METRIC_COL = {
    "h": f"mean_{COL_H_WM2K}",
    "f": f"mean_{COL_F_FANNING}",
}
METRIC_YLABEL = {
    "h": "h [W/m²K]",
    "f": "f$_{\\mathrm{Fanning}}$ [−]",
}
METRIC_TITLE = {
    "h": "Heat transfer coefficient — temporal convergence",
    "f": "Fanning friction factor — temporal convergence",
}

# Colours assigned deterministically per case label
_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#17becf",
    "#bcbd22", "#7f7f7f",
]

PLOT_DPI = 200
PLOT_FIGSIZE = (12.0, 4.5)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_step(stem: str) -> int:
    m = STEP_RE.search(stem)
    return int(m.group(1)) if m else -1


def _extract_time_token(stem: str) -> Optional[float]:
    m = TIME_RE.search(stem)
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", "."))
    except ValueError:
        return None


def _extract_t_local(part_meta: Dict[str, Any], stem: str) -> Optional[float]:
    t_start = float(part_meta["t_start_s"])
    t_end = float(part_meta["t_end_s"])
    dt = float(part_meta["dt_sim_s"])
    t_len = t_end - t_start

    priority = part_meta.get("time_source_priority", ("T", "S"))
    interpretation = str(part_meta.get("t_interpretation", "global")).lower()

    for src in priority:
        src = str(src).upper()
        if src == "T":
            token = _extract_time_token(stem)
            if token is not None:
                if interpretation == "local":
                    return token
                if interpretation == "global":
                    return token - t_start
                # auto
                return token if (-1e-12) <= token <= (t_len + 1e-12) else token - t_start
        elif src == "S":
            step = _extract_step(stem)
            if step >= 0:
                return step * dt
    return None


def _fluid_from_name(stem: str) -> str:
    s = stem.lower()
    if "fluid1" in s or "_f1_" in s or s.endswith("_f1") or s.startswith("f1_"):
        return "Fluid1"
    if "fluid2" in s or "_f2_" in s or s.endswith("_f2") or s.startswith("f2_"):
        return "Fluid2"
    return "Fluid2" if "_f2" in s else "Fluid1"


_NORM_RE_BAND  = re.compile(r"\bf[12]_wall_band_")
_NORM_RE_ISO   = re.compile(r"\bf[12]_(?:ziso_|yiso_)")
_NORM_RE_ENV   = re.compile(r"\bf[12]_env_")


def _normalize_prefixes(text: str) -> str:
    t = _NORM_RE_BAND.sub("wall_band_", text)
    t = _NORM_RE_ISO.sub("ziso_", t)
    t = _NORM_RE_ENV.sub("env_", t)
    return t


def _build_means(path: Path, fluid: str) -> Optional[Dict[str, float]]:
    """Parse one SRP file and return global mean dict (or None on failure)."""
    cfg = FLUID_CFG[fluid]
    axis_label = str(cfg["axis"])
    step_abs = float(cfg["step"])

    try:
        raw = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None

    normalized = _normalize_prefixes(raw)

    with tempfile.NamedTemporaryFile("w", suffix=".srp", delete=False, encoding="utf-8") as fh:
        fh.write(normalized)
        tmp = Path(fh.name)
    try:
        data = parse_srp(tmp)
        planes_df = build_planes_df(data, dz=step_abs)
        bands_df = build_bands_df(data)
        band_tab = compute_band_table(planes_df, bands_df, dz=step_abs, axis_label=axis_label)
        return compute_global_means(band_tab)
    except Exception:
        return None
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Core: build time-series for one case × one fluid
# ---------------------------------------------------------------------------

def _time_series_for_case(
    case_id: str,
    case_cfg: Dict[str, Any],
    base_dir: Path,
    fluid: str,
    verbose: bool,
) -> List[Tuple[float, Dict[str, float]]]:
    """Return sorted list of (t_global [s], means_dict) for all timesteps."""
    rows: List[Tuple[float, Dict[str, float]]] = []
    parts = case_cfg.get("parts", {})
    t_global_base = 0.0  # will be overridden per part

    for part_id, part_meta in parts.items():
        t_start = float(part_meta["t_start_s"])
        source_dir = base_dir / str(part_meta["source_dir"])
        if not source_dir.exists():
            if verbose:
                print(f"  [WARN] {case_id}/{part_id}: directory not found: {source_dir}")
            continue

        srp_files = sorted(source_dir.glob("*.srp"))
        if not srp_files:
            if verbose:
                print(f"  [WARN] {case_id}/{part_id}: no SRP files in {source_dir}")
            continue

        for srp_path in srp_files:
            if _fluid_from_name(srp_path.stem) != fluid:
                continue
            t_local = _extract_t_local(part_meta, srp_path.stem)
            if t_local is None:
                continue
            t_global = t_start + t_local
            means = _build_means(srp_path, fluid)
            if means:
                rows.append((t_global, means))

    rows.sort(key=lambda x: x[0])
    return rows


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_convergence(
    series_data: Dict[str, List[Tuple[float, Dict[str, float]]]],
    metric: str,
    fluid: str,
    out_dir: Path,
    verbose: bool,
) -> None:
    """
    series_data: {case_label: [(t_global, means_dict), ...]}
    """
    col = METRIC_COL[metric]
    ylabel = METRIC_YLABEL[metric]
    title = f"{METRIC_TITLE[metric]} — {fluid}"

    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE, dpi=PLOT_DPI)

    for idx, (label, rows) in enumerate(series_data.items()):
        if not rows:
            continue
        times = [r[0] for r in rows]
        values = [r[1].get(col, float("nan")) for r in rows]
        color = _PALETTE[idx % len(_PALETTE)]
        ax.plot(times, values, lw=0.9, color=color, label=label, alpha=0.85)

    ax.set_xlabel("Simulation time [s]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8, framealpha=0.7)
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
    fig.tight_layout()

    fluid_tag = "F1" if fluid == "Fluid1" else "F2"
    stem = f"conv_{metric}_{fluid_tag}"
    png_path = out_dir / f"{stem}.png"
    fig.savefig(png_path, dpi=PLOT_DPI)
    plt.close(fig)
    if verbose:
        print(f"  [OK] plot → {png_path}")


def _save_csv(
    series_data: Dict[str, List[Tuple[float, Dict[str, float]]]],
    metric: str,
    fluid: str,
    out_dir: Path,
    verbose: bool,
) -> None:
    col = METRIC_COL[metric]
    fluid_tag = "F1" if fluid == "Fluid1" else "F2"
    rows = []
    for label, time_rows in series_data.items():
        for t, means in time_rows:
            rows.append({"case": label, "t_global[s]": t, col: means.get(col, float("nan"))})
    if not rows:
        return
    df = pd.DataFrame(rows).sort_values(["case", "t_global[s]"]).reset_index(drop=True)
    csv_path = out_dir / f"conv_{metric}_{fluid_tag}.csv"
    df.to_csv(csv_path, index=False)
    if verbose:
        print(f"  [OK] CSV  → {csv_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    repo_dir = Path(__file__).resolve().parent
    default_base = (
        str(DEFAULT_BASE_DATA_DIR).strip()
        or str(repo_dir / "FluentTransientData")
    )
    default_out = (
        str(DEFAULT_OUT_DIR).strip()
        or str(repo_dir / "TransientFigs")
    )
    parser = argparse.ArgumentParser(
        description="Plot mean_h and mean_f vs. simulation time for transient cases.",
    )
    parser.add_argument("--in-dir", default=default_base,
                        help="Base directory containing transient SRP subdirectories.")
    parser.add_argument("--out-dir", default=default_out,
                        help="Root output directory (convergence/ subdir will be created).")
    parser.add_argument("--cases", nargs="+", metavar="CASE_ID",
                        help="Restrict to these case IDs (default: all cases present in CASES).")
    parser.add_argument("--fluids", nargs="+", default=["Fluid1"],
                        choices=["Fluid1", "Fluid2"],
                        help="Which fluid channels to analyse (default: Fluid1).")
    parser.add_argument("--metrics", nargs="+", default=["h", "f"],
                        choices=list(METRIC_COL.keys()),
                        help="Metrics to plot (default: h f).")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    base_dir = Path(args.in_dir).resolve()
    out_root = Path(args.out_dir).resolve()
    conv_dir = out_root / "convergence"
    ensure_dir(conv_dir)

    # Select cases
    requested = set(args.cases) if args.cases else None
    selected_cases: Dict[str, Any] = {}
    for case_id, case_cfg in CASES.items():
        if requested is not None and case_id not in requested:
            continue
        selected_cases[case_id] = case_cfg

    if not selected_cases:
        print("[INFO] No matching cases found. Check --cases argument or CASES in config.py.")
        return

    print(f"[INFO] Analysing {len(selected_cases)} case(s): {list(selected_cases)}")

    for fluid in args.fluids:
        print(f"\n[INFO] Fluid: {fluid}")

        # Collect time series per case
        series_data: Dict[str, List[Tuple[float, Dict[str, float]]]] = {}
        for case_id, case_cfg in selected_cases.items():
            label = case_id
            if args.verbose:
                print(f"  Processing {case_id} / {fluid} ...")
            rows = _time_series_for_case(case_id, case_cfg, base_dir, fluid, args.verbose)
            if rows:
                series_data[label] = rows
            else:
                if args.verbose:
                    print(f"  [WARN] No data for {case_id} / {fluid}")

        if not series_data:
            print(f"  [WARN] No time-series data collected for {fluid} — skipping.")
            continue

        for metric in args.metrics:
            _plot_convergence(series_data, metric, fluid, conv_dir, args.verbose)
            _save_csv(series_data, metric, fluid, conv_dir, args.verbose)

    print("\n[DONE] convergence.py finished.")


if __name__ == "__main__":
    main()
