# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import matplotlib as mpl

from style import StyleResolver

from constants import (
    COL_Z_M, COL_Y_M,
    COL_H_WM2K, COL_F, COL_DP_BAND, COL_DP_SUM,
)

# =========================
# Global matplotlib behavior
# =========================
mpl.rcParams["axes.formatter.useoffset"] = False

# =========================
# Defaults (can be overridden via set_plot_defaults)
# =========================
DEFAULT_DPI = 160
MS = 2.0          # marker size
LW = 1.0          # line width

# Two separate Y padders:
OVERLAY_Y_PAD_FACTOR = 0.1  # requested
MEAN_Y_PAD_FACTOR = 3.0     # requested

# pick axis column
AXIS_CANDIDATES = (COL_Z_M, COL_Y_M)

# For consistent styling across ALL plots:
LINE_STYLES = ["-", "--", "-.", ":"]
MARKERS = ["o", "s", "^", "D", "x", "P", "v", "*", "+", None]

# If label not found in USER_COLOR_MAP, fallback to this cycle deterministically
FALLBACK_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]

# =========================
# User-defined colors (EDIT HERE)
# =========================
USER_COLOR_MAP: Dict[str, str] = {
    # Parts
    "Part1": "#1f77b4",
    "Part2": "#ff7f0e",
    "Part3": "#2ca02c",
    "Part4": "#d62728",
    "Part5": "#4a068a",

    # Steady (recommended label format in compare is "M006 SS", "M007 SS")
    "M006 SS": "#8c564b",
    "M007 SS": "#af007b",

    # Optional: mean±std overlay special key
    "MEAN_STD": "#050000",
}

def _apply_dash(line, ls: str) -> None:
    """
    Wymusza własne wzorce dla linii przerywanych/kropkowanych.
    Jednostki są w punktach (pt) w Matplotlib.
    """
    if ls == "--":
        line.set_dashes([8, 8])
    elif ls == ":":
        line.set_dashes([2, 6])
    elif ls == "-.":
        line.set_dashes([8, 8, 2, 6])
    # dla '-' nic nie rób

# =========================
# Utilities
# =========================
def set_plot_defaults(
    dpi: int = 160,
    marker_size: float = 2.0,
    line_width: float = 0.8,
    overlay_y_pad_factor: float = 0.1,
    mean_y_pad_factor: float = 3.0,
) -> None:
    """Global defaults for all plots."""
    global DEFAULT_DPI, MS, LW, OVERLAY_Y_PAD_FACTOR, MEAN_Y_PAD_FACTOR
    DEFAULT_DPI = int(dpi)
    MS = float(marker_size)
    LW = float(line_width)
    OVERLAY_Y_PAD_FACTOR = float(overlay_y_pad_factor)
    MEAN_Y_PAD_FACTOR = float(mean_y_pad_factor)

def ensure_dir(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)

def pick_axis_col(df: pd.DataFrame) -> Optional[str]:
    for c in AXIS_CANDIDATES:
        if c in df.columns:
            return c
    return None

def _sanitize(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s))

# Alias for older imports
sanitize = _sanitize

def user_color(label: str) -> Optional[str]:
    """Return user-defined color for label, or None for fallback."""
    return USER_COLOR_MAP.get(label)

def _hash_to_index(label: str, n: int) -> int:
    # deterministic index from label (stable across runs)
    h = 0
    for ch in label:
        h = (h * 131 + ord(ch)) & 0xFFFFFFFF
    return int(h % n) if n > 0 else 0

def _markevery(npts: int, target_marks: int = 60) -> int:
    """~target_marks markers across the curve (less dense)."""
    if npts <= 0:
        return 1
    return max(1, int(npts // max(1, target_marks)))

def _style_for_label(label: str, npts: int) -> Dict:
    """
    Consistent style for a given label across ALL plots:
    - color from USER_COLOR_MAP if present, else fallback by hash(label)
    - linestyle + marker by hash(label)
    - markevery based on npts
    """
    color = user_color(label)
    if color is None:
        color = FALLBACK_COLORS[_hash_to_index(label, len(FALLBACK_COLORS))]

    ls = LINE_STYLES[_hash_to_index(label + "|ls", len(LINE_STYLES))]
    mk = MARKERS[_hash_to_index(label + "|mk", len(MARKERS))]
    me = _markevery(npts, target_marks=60)

    return {"color": color, "ls": ls, "marker": mk, "markevery": me}

def _style_for_index(i: int, npts: int) -> Dict:
    """
    Deterministic style by curve index (used for standard overlay where labels are time).
    Ensures different color/ls/marker per curve, stable across runs for same ordering.
    """
    color = FALLBACK_COLORS[i % len(FALLBACK_COLORS)]
    ls = LINE_STYLES[i % len(LINE_STYLES)]
    mk = MARKERS[i % len(MARKERS)]
    me = _markevery(npts, target_marks=80)
    return {"color": color, "ls": ls, "marker": mk, "markevery": me}

def _pad_limits(y_min: float, y_max: float, factor: float) -> Tuple[float, float]:
    if not np.isfinite(y_min) or not np.isfinite(y_max):
        return (y_min, y_max)
    rng = float(y_max - y_min)
    if rng > 0:
        pad = factor * rng
    else:
        pad = 0.1 * max(1.0, abs(y_max))
    return (y_min - pad, y_max + pad)

def _plot_line(ax, x, y, st: Dict, *, alpha: float = 1.0, label: Optional[str] = None):
    """
    Wrapper around ax.plot(...) that returns the Line2D handle,
    then applies custom dash patterns via _apply_dash().
    """
    # per-line overrides (compare plots need per-series lw/ms)
    lw = float(st.get("lw", LW))
    ms = float(st.get("ms", MS))
    ls = st.get("ls", "-")
    color = st.get("color", "#000000")
    marker = st.get("marker", None)
    markevery = st.get("markevery", None)

    # marker cosmetics
    mfc = st.get("mfc", None)
    mec = st.get("mec", None)
    mew = st.get("mew", None)

    # final alpha = caller alpha * style alpha
    a = float(alpha) * float(st.get("alpha", 1.0))

    (line,) = ax.plot(
        x,
        y,
        lw=lw,
        ls=ls,
        color=color,
        marker=marker,
        ms=ms,
        markevery=markevery,
        alpha=a,
        label=label,
        markerfacecolor=mfc,
        markeredgecolor=mec,
        markeredgewidth=mew,
        zorder=st.get("zorder", None),
    )
    _apply_dash(line, ls)
    return line


def _compare_figspec(job: Optional[Dict], kind: str) -> Tuple[Tuple[float, float], int]:
    """Pobiera figsize/dpi dla compare z job['fig'].

    kind: "overlay" | "mean"
    """
    j = job or {}
    fig_cfg = j.get("fig", {}) or {}
    dpi = int(fig_cfg.get("dpi", DEFAULT_DPI))
    block = fig_cfg.get(kind, {}) or {}
    fs = block.get("figsize", None)
    if fs is None:
        # sensowne defaulty dla compare (większe niż standardowe)
        fs = (12.0, 4.6) if kind == "overlay" else (12.0, 4.6)
    try:
        figsize = (float(fs[0]), float(fs[1]))
    except Exception:
        figsize = (12.0, 4.6)
    return figsize, dpi

def _metric_specs_overlay(metrics: List[str], include_pressure: bool) -> List[Tuple[str, str, str]]:
    specs: List[Tuple[str, str, str]] = []
    if "h" in metrics:
        specs.append((COL_H_WM2K, "h [W/m²K]", "overlay_h"))
    if "f" in metrics:
        specs.append((COL_F, "f_Fanning [-]", "overlay_f"))
    if include_pressure:
        specs.append((COL_DP_BAND, "Δp (band) [Pa]", "overlay_dp_band"))
    return specs

def _metric_specs_mean(metrics: List[str], include_pressure: bool) -> List[Tuple[str, str, str]]:
    specs: List[Tuple[str, str, str]] = []
    if "h" in metrics:
        specs.append(("mean_h[W/m2K]", "h [W/m²K]", "mean_h"))
    if "f" in metrics:
        specs.append(("mean_f_fanning[-]", "f_Fanning [-]", "mean_f"))
    if include_pressure:
        specs.append(("mean_Δp_sum[Pa]", "Δp_sum [Pa]", "mean_dp_sum"))
    return specs

# =========================
# STANDARD: Overlay per fluid (ALL lines)
# =========================
def make_overlays_per_fluid(
    overlays_seq: List[Tuple[pd.DataFrame, str]],  # (df, label)
    fluid_name: str,
    out_dir: Path,
    metrics: List[str],
    include_pressure: bool,
    overlay_every: int = 1,
    outfile_tag: str = "",
) -> None:
    """
    Standard overlay (PLOT_JOBS). Many curves -> style by index (not by time label).
    """
    if not overlays_seq:
        return
    ensure_dir(out_dir)

    specs = _metric_specs_overlay(metrics, include_pressure)

    idxs = list(range(len(overlays_seq)))
    if overlay_every > 1 and len(idxs) > 2:
        keep = {0, len(idxs) - 1}
        keep.update(i for i in idxs if i % overlay_every == 0)
        sel = sorted(keep)
    else:
        sel = idxs

    for col, ylabel, prefix in specs:
        fig, ax = plt.subplots()
        plotted = 0
        x_all_max = 0.0
        y_all: List[np.ndarray] = []

        for j, i in enumerate(sel):
            df, _lbl = overlays_seq[i]
            if col not in df.columns:
                continue
            axis_col = pick_axis_col(df)
            if axis_col is None:
                continue

            x_raw = df[axis_col].astype(float).values
            y = df[col].astype(float).values

            # x from In->Out
            x0 = (x_raw.max() - x_raw) if fluid_name == "Fluid2" else (x_raw - x_raw.min())
            x_mm = x0 * 1000.0

            st = _style_for_index(j, len(x_mm))
            _plot_line(ax, x_mm, y, st, alpha=0.9, label=None)

            plotted += 1
            x_all_max = max(x_all_max, float(np.nanmax(x_mm)))
            m = np.isfinite(y)
            if m.any():
                y_all.append(y[m])

        if plotted == 0:
            plt.close(fig)
            continue

        ax.set_xlim(0.0, x_all_max)
        ax.set_xlabel(("Water" if fluid_name == "Fluid2" else "Air") + ": from In to Out [mm]")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

        if y_all:
            y_concat = np.concatenate(y_all) if len(y_all) > 1 else y_all[0]
            y_min, y_max = float(np.min(y_concat)), float(np.max(y_concat))
            lo, hi = _pad_limits(y_min, y_max, OVERLAY_Y_PAD_FACTOR)
            ax.set_ylim(lo, hi)

        fname = f"{prefix}__{outfile_tag}.png" if outfile_tag else f"{prefix}.png"
        fig.savefig(out_dir / fname, dpi=DEFAULT_DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] Overlay -> {fname}")

# =========================
# STANDARD: Overlay per fluid (MEAN ± STD)
# =========================
def make_overlay_mean_std_per_fluid(
    overlays_seq: List[Tuple[pd.DataFrame, str]],  # (df, label)
    fluid_name: str,
    out_dir: Path,
    metrics: List[str],
    include_pressure: bool,
    overlay_every: int = 1,   # kept for API compatibility (not used)
    outfile_tag: str = "",
) -> None:
    """
    Standard overlay mean±std (PLOT_JOBS). One mean curve + band.
    """
    if not overlays_seq:
        return
    ensure_dir(out_dir)

    specs = _metric_specs_overlay(metrics, include_pressure)

    axis_col0 = pick_axis_col(overlays_seq[0][0])
    if axis_col0 is None:
        return

    # build x_mm from first frame
    df0 = overlays_seq[0][0]
    x_raw0 = df0[axis_col0].astype(float).values
    x00 = (x_raw0.max() - x_raw0) if fluid_name == "Fluid2" else (x_raw0 - x_raw0.min())
    x_mm = x00 * 1000.0

    for col, ylabel, prefix in specs:
        Ys = []
        for df, _lbl in overlays_seq:
            if col not in df.columns or axis_col0 not in df.columns:
                continue
            y = df[col].astype(float).values
            if y.shape != x_mm.shape:
                continue
            Ys.append(y)

        if not Ys:
            continue

        Y = np.vstack(Ys)
        mu = np.nanmean(Y, axis=0)
        sd = np.nanstd(Y, axis=0)

        fig, ax = plt.subplots()

        st = _style_for_label("MEAN_STD", len(x_mm))
        _plot_line(ax, x_mm, mu, st, alpha=0.95, label="mean")
        ax.fill_between(x_mm, mu - sd, mu + sd, color=st["color"], alpha=0.15, label="±1σ")

        ax.set_xlim(0.0, float(np.nanmax(x_mm)))
        ax.set_xlabel(("Water" if fluid_name == "Fluid2" else "Air") + ": from In to Out [mm]")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", frameon=False)

        m = np.isfinite(mu)
        if m.any():
            y_min, y_max = float(np.min(mu[m] - sd[m])), float(np.max(mu[m] + sd[m]))
            lo, hi = _pad_limits(y_min, y_max, OVERLAY_Y_PAD_FACTOR)
            ax.set_ylim(lo, hi)

        fname = f"{prefix}_meanstd__{outfile_tag}.png" if outfile_tag else f"{prefix}_meanstd.png"
        fig.savefig(out_dir / fname, dpi=DEFAULT_DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] Overlay(mean+/-std) -> {fname}")

# =========================
# STANDARD: Mean vs time per fluid
# =========================
def plot_means_vs_time(
    df: pd.DataFrame,
    fluid_name: str,
    out_dir: Path,
    metrics: List[str],
    include_pressure: bool,
    outfile_tag: str = "",
) -> None:
    """
    Standard mean (PLOT_JOBS). Single curve -> style by a stable synthetic label.
    """
    ensure_dir(out_dir)
    if "Time[s]" not in df.columns:
        return

    gdf = df[df["Fluid"] == fluid_name].sort_values("Time[s]")
    if gdf.empty:
        return

    specs = _metric_specs_mean(metrics, include_pressure)

    for col, ylabel, prefix in specs:
        if col not in gdf.columns:
            continue

        x = gdf["Time[s]"].astype(float).values
        y = gdf[col].astype(float).values

        fig, ax = plt.subplots()

        style_key = f"PLOTJOB_MEAN_{fluid_name}_{prefix}"
        st = _style_for_label(style_key, len(x))

        _plot_line(ax, x, y, st, alpha=0.95, label=None)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

        m = np.isfinite(y)
        if m.any():
            y_min, y_max = float(np.min(y[m])), float(np.max(y[m]))
            lo, hi = _pad_limits(y_min, y_max, MEAN_Y_PAD_FACTOR)
            ax.set_ylim(lo, hi)

        fname = f"{prefix}__{outfile_tag}.png" if outfile_tag else f"{prefix}.png"
        fig.savefig(out_dir / fname, dpi=DEFAULT_DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] Mean -> {fname}")

# =========================
# COMPARE: Overlay (multiple series)
# =========================
def plot_compare_overlay(
    out_dir: Path,
    metric_key: str,             # "h"|"f"|"dp_band"
    fluid_name: str,
    series_profiles: List[Dict], # [{label,x_mm,y,shade_lo,shade_hi}, ...]
    job: Optional[Dict] = None,
) -> None:
    ensure_dir(out_dir)

    ylabels = {"h": "h [W/m²K]", "f": "f_Fanning [-]", "dp_band": "Δp (band) [Pa]"}

    figsize, dpi = _compare_figspec(job, "overlay")
    fig, ax = plt.subplots(figsize=figsize)

    resolver = StyleResolver.from_job(job)

    xmax = 0.0
    y_all: List[np.ndarray] = []

    for sp in series_profiles:
        x = np.asarray(sp["x_mm"], dtype=float)
        y = np.asarray(sp["y"], dtype=float)
        lbl = str(sp["label"])

        st = resolver.style_for(lbl, len(x))
        _plot_line(ax, x, y, st, alpha=0.95, label=lbl)

        xmax = max(xmax, float(np.nanmax(x)))
        m = np.isfinite(y)
        if m.any():
            y_all.append(y[m])

        lo = sp.get("shade_lo")
        hi = sp.get("shade_hi")
        if lo is not None and hi is not None:
            lo = np.asarray(lo, dtype=float)
            hi = np.asarray(hi, dtype=float)
            ax.fill_between(x, lo, hi, color=st.get("color"), alpha=0.15)

    ax.set_xlim(0.0, xmax)
    ax.set_xlabel(("Water" if fluid_name == "Fluid2" else "Air") + ": from In to Out [mm]")
    ax.set_ylabel(ylabels[metric_key])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", frameon=False)

    if y_all:
        y_concat = np.concatenate(y_all) if len(y_all) > 1 else y_all[0]
        y_min, y_max = float(np.min(y_concat)), float(np.max(y_concat))
        lo, hi = _pad_limits(y_min, y_max, OVERLAY_Y_PAD_FACTOR)
        ax.set_ylim(lo, hi)

    job_name = (job or {}).get("name", "")
    fname = f"cmp_overlay_{metric_key}_{'F2' if fluid_name=='Fluid2' else 'F1'}"
    if job_name:
        fname += f"__{_sanitize(job_name)}"
    fname += ".png"

    fig.savefig(out_dir / fname, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Compare-Overlay -> {fname}")

# =========================
# COMPARE: Mean (multiple series)
# =========================
def plot_compare_mean(
    out_dir: Path,
    metric_key: str,             # "h" | "f" | "dp_sum"
    fluid_name: str,
    series_list: List[Dict],     # [{label,t_global,t_aligned,y_raw,y_ma{W:arr}}, ...]
    job: Optional[Dict] = None,
    time_mode: str = "global",   # "global" | "aligned"
    show_raw: bool = True,
    ref_lines: Optional[List[tuple[str, float]]] = None,  # e.g. [("M006", val), ...]
) -> None:
    ensure_dir(out_dir)

    figsize, dpi = _compare_figspec(job, "mean")
    fig, ax = plt.subplots(figsize=figsize)

    resolver = StyleResolver.from_job(job)
    plot_cfg = (job or {}).get("plot", {}) or {}
    raw_alpha = float(plot_cfg.get("raw_alpha", 0.45))
    ma_alpha  = float(plot_cfg.get("ma_alpha", 0.95))
    raw_lw_scale = float(plot_cfg.get("raw_lw_scale", 0.8))
    ma_lw_scale  = float(plot_cfg.get("ma_lw_scale", 1.0))
    ref_ls    = str(plot_cfg.get("ref_ls", ":"))
    ref_alpha = float(plot_cfg.get("ref_alpha", 0.9))

    # collect y for scaling (only what is actually drawn)
    y_all: List[np.ndarray] = []

    t_min_all = np.inf
    t_max_all = -np.inf

    for s in series_list:
        label = str(s["label"])
        t = np.asarray(s["t_global"] if time_mode == "global" else s["t_aligned"], dtype=float)

        if t.size:
            t_min_all = min(t_min_all, float(np.nanmin(t)))
            t_max_all = max(t_max_all, float(np.nanmax(t)))

        # raw
        if show_raw and s.get("y_raw") is not None:
            y = np.asarray(s["y_raw"], dtype=float)
            st = resolver.style_for(label, len(t))
            st = dict(st)
            st["lw"] = float(st.get("lw", LW)) * raw_lw_scale
            _plot_line(ax, t, y, st, alpha=raw_alpha, label=f"{label} (raw)")
            m = np.isfinite(y)
            if m.any():
                y_all.append(y[m])

        # MA curves
        y_ma = s.get("y_ma") or {}
        for W, yW in y_ma.items():
            yW = np.asarray(yW, dtype=float)
            st = resolver.style_for(label, len(t))
            st = dict(st)
            st["lw"] = float(st.get("lw", LW)) * ma_lw_scale
            _plot_line(ax, t, yW, st, alpha=ma_alpha, label=f"{label} (MA{W})")
            m = np.isfinite(yW)
            if m.any():
                y_all.append(yW[m])

    # steady reference lines (styled as their own "labels")
    if ref_lines:
        if not np.isfinite(t_min_all) or not np.isfinite(t_max_all) or t_min_all >= t_max_all:
            t_min_all, t_max_all = 0.0, 1.0
        for lbl, yref in ref_lines:
            lbl_ss = f"{lbl} SS" if not str(lbl).endswith("SS") else str(lbl)
            st = resolver.style_for(lbl_ss, 200)
            st = dict(st)
            st["ls"] = ref_ls
            st["marker"] = None
            _plot_line(ax, np.asarray([t_min_all, t_max_all]),
                      np.asarray([float(yref), float(yref)]),
                      st, alpha=ref_alpha, label=lbl_ss)

        y_all.append(np.asarray([y for _, y in ref_lines], dtype=float))

    ax.set_xlabel("Time [s]" if time_mode == "global" else "Aligned time [s]")
    ylabels = {"h": "h [W/m²K]", "f": "f_Fanning [-]", "dp_sum": "Δp_sum [Pa]"}
    ax.set_ylabel(ylabels[metric_key])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", frameon=False)

    # Y scaling (only from collected y)
    if y_all:
        y_concat = np.concatenate([arr[np.isfinite(arr)] for arr in y_all if arr.size])
        if y_concat.size:
            y_min, y_max = float(np.min(y_concat)), float(np.max(y_concat))
            lo, hi = _pad_limits(y_min, y_max, MEAN_Y_PAD_FACTOR)
            ax.set_ylim(lo, hi)

    job_name = (job or {}).get("name", "")
    fname = f"cmp_mean_{metric_key}_{'F2' if fluid_name=='Fluid2' else 'F1'}__{time_mode}"
    if job_name:
        fname += f"__{_sanitize(job_name)}"
    fname += ".png"

    fig.savefig(out_dir / fname, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Compare-Mean -> {fname}")
