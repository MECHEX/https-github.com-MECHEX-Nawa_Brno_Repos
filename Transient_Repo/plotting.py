# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import re
import itertools
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D

# czytelniejsze etykiety (bez +offset)
mpl.rcParams['axes.formatter.useoffset'] = False

from constants import (
    COL_Z_M, COL_Y_M,
    COL_H_WM2K, COL_F, COL_DP_BAND, COL_DP_SUM,
)

# ====== ustawienia globalne ======
DEFAULT_DPI = 160
MS = 2.0
LW = 0.8

# niezależne „pad-y” osi Y
_Y_PAD_OVERLAY = 0.10   # overlay (profilowe)
_Y_PAD_MEAN    = 3.00   # mean (czasowe)

# wybór osi
AXIS_CANDIDATES = (COL_Z_M, COL_Y_M)

# palety i style
LINE_STYLES = ['-', '--', '-.', ':']
MARKERS     = ['o', 's', '^', 'D', 'x', 'v', 'P', '*']  # wymagany znacznik

def _default_color_cycle() -> List[str]:
    prop = mpl.rcParams.get('axes.prop_cycle', None)
    if prop is None:
        return ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                '#bcbd22', '#17becf']
    colors = [d.get('color') for d in prop]
    if not colors:
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                  '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                  '#bcbd22', '#17becf']
    return colors

# ========= Globalny menedżer stylów per etykieta =========
class _StyleManager:
    """
    Każda unikalna etykieta dostaje spójny (kolor, linestyle, marker).
    Używamy trzech niezależnych cykli, aby uniknąć „zawieszania” się na jednym kolorze.
    """
    def __init__(self):
        self._label2style: Dict[str, Tuple[str, str, str]] = {}
        colors = _default_color_cycle()
        self._color_cycle  = itertools.cycle(colors)
        self._ls_cycle     = itertools.cycle(LINE_STYLES)
        self._marker_cycle = itertools.cycle(MARKERS)

    @staticmethod
    def _norm_label(label: str) -> str:
        return label.strip()

    def get(self, label: str) -> Tuple[str, str, str]:
        key = self._norm_label(label)
        if key not in self._label2style:
            c = next(self._color_cycle)
            ls = next(self._ls_cycle)
            mk = next(self._marker_cycle)
            self._label2style[key] = (c, ls, mk)
        return self._label2style[key]

    @staticmethod
    def markevery(npts: int) -> int:
        return max(1, npts // 60)  # ~60 markerów na wykres

_STYLE = _StyleManager()

# ====== ustawienia globalne rysunków ======
def set_plot_defaults(
    dpi: int = 160,
    marker_size: float = 2.0,
    line_width: float = 0.8,
    y_pad_overlay: Optional[float] = None,
    y_pad_mean: Optional[float] = None,
) -> None:
    global DEFAULT_DPI, MS, LW, _Y_PAD_OVERLAY, _Y_PAD_MEAN
    DEFAULT_DPI = int(dpi)
    MS = float(marker_size)
    LW = float(line_width)
    if y_pad_overlay is not None:
        _Y_PAD_OVERLAY = float(y_pad_overlay)
    if y_pad_mean is not None:
        _Y_PAD_MEAN = float(y_pad_mean)

def ensure_dir(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)

def pick_axis_col(df: pd.DataFrame) -> Optional[str]:
    for c in AXIS_CANDIDATES:
        if c in df.columns:
            return c
    return None

def _sanitize(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)

def _apply_y_limits_from_arrays(ax, arrays: List[np.ndarray], pad_factor: float) -> None:
    ys = []
    for arr in arrays:
        if arr is None:
            continue
        a = np.asarray(arr)
        m = np.isfinite(a)
        if m.any():
            ys.append(a[m])
    if not ys:
        return
    y_concat = np.concatenate(ys)
    if y_concat.size == 0:
        return
    y_min, y_max = float(np.min(y_concat)), float(np.max(y_concat))
    rng = y_max - y_min
    pad = pad_factor * rng if rng > 0 else 0.1 * max(1.0, abs(y_max))
    ax.set_ylim(y_min - pad, y_max + pad)

# ========= STANDARD: OVERLAY dla jednego fluida =========
def make_overlays_per_fluid(
    overlays_seq: List[Tuple[pd.DataFrame, str]],
    fluid_name: str,
    out_dir: Path,
    metrics: List[str],
    include_pressure: bool,
    overlay_every: int = 1,
    outfile_tag: str = "",
) -> None:
    if not overlays_seq:
        return
    ensure_dir(out_dir)

    specs: List[Tuple[str, str, str]] = []
    if "h" in metrics:
        specs.append((COL_H_WM2K,  "h [W/m²K]",      "overlay_h"))
    if "f" in metrics:
        specs.append((COL_F,       "f_Fanning [-]",  "overlay_f"))
    if include_pressure:
        specs.append((COL_DP_BAND, "Δp (band) [Pa]", "overlay_dp_band"))

    idxs = list(range(len(overlays_seq)))
    if overlay_every > 1 and len(idxs) > 2:
        keep = {0, len(idxs) - 1}
        keep.update(i for i in idxs if i % overlay_every == 0)
        sel = sorted(keep)
    else:
        sel = idxs

    for col, ylabel, prefix in specs:
        fig, ax = plt.subplots()
        ax.ticklabel_format(useOffset=False, axis='y')

        plotted, x_all_max = 0, 0.0
        y_for_limits: List[np.ndarray] = []

        for i in sel:
            df, lbl = overlays_seq[i]
            if col not in df.columns:
                continue
            axis_col = pick_axis_col(df)
            if axis_col is None:
                continue

            x_raw = df[axis_col].values.astype(float)
            y = df[col].values
            y_for_limits.append(y)

            x0 = (x_raw.max() - x_raw) if fluid_name == "Fluid2" else (x_raw - x_raw.min())
            x_mm = x0 * 1000.0

            # świadomie jednolite style na overlayu (czasowe profile)
            ax.plot(x_mm, y, marker="o", ms=MS, lw=LW)
            plotted += 1
            x_all_max = max(x_all_max, float(x_mm.max()))

        if plotted == 0:
            plt.close(fig); continue

        ax.set_xlim(0.0, x_all_max)
        ax.set_xlabel(("Water" if fluid_name == "Fluid2" else "Air") + ": from In to Out [mm]")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

        _apply_y_limits_from_arrays(ax, y_for_limits, pad_factor=_Y_PAD_OVERLAY)

        fname = f"{prefix}__{outfile_tag}.png" if outfile_tag else f"{prefix}.png"
        fig.savefig(out_dir / fname, dpi=DEFAULT_DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] Overlay → {fname}")

# ========= STANDARD: MEAN vs TIME =========
def plot_means_vs_time(
    df: pd.DataFrame,
    fluid_name: str,
    out_dir: Path,
    metrics: List[str],
    include_pressure: bool,
    outfile_tag: str = "",
) -> None:
    ensure_dir(out_dir)
    if "Time[s]" not in df.columns:
        return
    gdf = df[df["Fluid"] == fluid_name].sort_values("Time[s]")
    if gdf.empty:
        return

    specs: List[Tuple[str, str, str]] = []
    if "h" in metrics:
        specs.append(("mean_h[W/m2K]", "h [W/m²K]", "mean_h"))
    if "f" in metrics:
        specs.append(("mean_f_fanning[-]", "f_Fanning [-]", "mean_f"))
    if include_pressure:
        specs.append(("mean_Δp_sum[Pa]", "Δp (sum) [Pa]", "mean_dp_sum"))

    for col, ylabel, prefix in specs:
        if col not in gdf.columns:
            continue

        x = gdf["Time[s]"].values
        y = gdf[col].values

        fig, ax = plt.subplots()
        ax.ticklabel_format(useOffset=False, axis='y')
        ax.plot(x, y, marker="o", ms=MS, lw=LW)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

        _apply_y_limits_from_arrays(ax, [y], pad_factor=_Y_PAD_MEAN)

        fname = f"{prefix}__{outfile_tag}.png" if outfile_tag else f"{prefix}.png"
        fig.savefig(out_dir / fname, dpi=DEFAULT_DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] Mean → {fname}")

# ========= PORÓWNANIA: OVERLAY =========
def plot_compare_overlay(
    out_dir: Path,
    metric_key: str,        # "h"|"f"|"dp_band"
    fluid_name: str,
    series_profiles: List[Dict],   # [{label, x_mm, y, shade_lo/hi, ...}, ...]
    job_name: str = "",
) -> None:
    ensure_dir(out_dir)

    ylabels = {"h": "h [W/m²K]", "f": "f_Fanning [-]", "dp_band": "Δp (band) [Pa]"}
    fig, ax = plt.subplots()
    ax.ticklabel_format(useOffset=False, axis='y')
    xmax = 0.0
    y_for_limits: List[np.ndarray] = []

    for sp in series_profiles:
        x = sp["x_mm"]; y = sp["y"]; lbl = sp["label"]
        color, ls, mk = _STYLE.get(lbl)
        me = _STYLE.markevery(len(x))
        line, = ax.plot(x, y, lw=LW, ls=ls, marker=mk, ms=MS,
                        markevery=me, alpha=0.95, color=color, label=lbl)
        y_for_limits.append(y)
        xmax = max(xmax, float(np.max(x)))
        if sp.get("shade_lo") is not None and sp.get("shade_hi") is not None:
            ax.fill_between(x, sp["shade_lo"], sp["shade_hi"],
                            alpha=0.15, color=line.get_color())

    ax.set_xlim(0.0, xmax)
    ax.set_xlabel(("Water" if fluid_name == "Fluid2" else "Air") + ": from In to Out [mm]")
    ax.set_ylabel(ylabels[metric_key])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", frameon=False)

    _apply_y_limits_from_arrays(ax, y_for_limits, pad_factor=_Y_PAD_OVERLAY)

    fname = f"cmp_overlay_{metric_key}_{'F2' if fluid_name=='Fluid2' else 'F1'}"
    if job_name: fname += f"__{_sanitize(job_name)}"
    fname += ".png"
    fig.savefig(out_dir / fname, dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f".[OK] Compare-Overlay → {fname}")

# ========= PORÓWNANIA: MEAN =========
def plot_compare_mean(
    out_dir: Path,
    metric_key: str,        # "h" | "f" | "dp_sum"
    fluid_name: str,
    series_list: List[Dict],   # [{label, t_global,t_aligned,y_raw,y_ma{W:arr}}, ...]
    job_name: str = "",
    time_mode: str = "global", # "global" | "aligned"
    show_raw: bool = True,
    ref_lines: Optional[List[tuple[str, float]]] = None,  # steady lines (label, value)
) -> None:
    ensure_dir(out_dir)
    fig, ax = plt.subplots()
    ax.ticklabel_format(useOffset=False, axis='y')

    y_for_limits: List[np.ndarray] = []

    for s in series_list:
        label = s["label"]
        t = s["t_global"] if time_mode == "global" else s["t_aligned"]
        color, ls, mk = _STYLE.get(label)
        me = _STYLE.markevery(len(t))

        if show_raw:
            y = s["y_raw"]
            ax.plot(t, y, lw=LW, ls=ls, marker=mk, ms=MS,
                    markevery=me, alpha=0.6, color=color, label=label)
            y_for_limits.append(y)

        for W, y_ma in (s.get("y_ma") or {}).items():
            ax.plot(t, y_ma, lw=LW, ls=ls, marker=mk, ms=MS,
                    markevery=me, alpha=0.95, color=color, label=f"{label} (MA{W})")
            y_for_limits.append(y_ma)

    # steady lines: ten sam styl co etykieta
    if ref_lines:
        proxy = []
        for lbl, yref in ref_lines:
            color, ls, mk = _STYLE.get(lbl)
            ax.axhline(yref, ls=ls, lw=max(0.7, LW*0.9), color=color)
            proxy.append(Line2D([0], [0], color=color, ls=ls, marker=mk, ms=MS,
                                lw=LW, label=f"{lbl} (SS)"))
            y_for_limits.append(np.array([yref], dtype=float))
        leg = ax.legend(loc="best", frameon=False)
        if leg is None:
            ax.legend(handles=proxy, loc="best", frameon=False)
        else:
            ax.add_artist(leg)
            ax.legend(handles=proxy, loc="lower right", frameon=False)
    else:
        ax.legend(loc="best", frameon=False)

    ax.set_xlabel("Time [s]" if time_mode == "global" else "Aligned time [s]")
    ylabels = {"h": "h [W/m²K]", "f": "f_Fanning [-]", "dp_sum": "Δp_sum [Pa]"}
    ax.set_ylabel(ylabels[metric_key])
    ax.grid(True, alpha=0.3)

    _apply_y_limits_from_arrays(ax, y_for_limits, pad_factor=_Y_PAD_MEAN)

    fname = f"cmp_mean_{metric_key}_{'F2' if fluid_name=='Fluid2' else 'F1'}__{time_mode}"
    if job_name: fname += f"__{_sanitize(job_name)}"
    fname += ".png"
    fig.savefig(out_dir / fname, dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f".[OK] Compare-Mean → {fname}")
