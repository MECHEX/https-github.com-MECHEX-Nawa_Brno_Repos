# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
from typing import List, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from constants import (
    COL_H_WM2K, COL_F, COL_DP_BAND,
    YLABEL_MAP, PREFIX_MAP, MEAN_COL_MAP,
)
from tpms_utils import ensure_dir, pick_axis_col

# USTAWIENIA GLOBALNE RYSOWANIA (2D)
DEFAULT_DPI = 160
MS = 2.0
LW = 0.8

def set_plot_defaults(dpi: int = 160, marker_size: float = 2.0, line_width: float = 0.8) -> None:
    global DEFAULT_DPI, MS, LW
    DEFAULT_DPI = int(dpi)
    MS = float(marker_size)
    LW = float(line_width)

# ------ OVERLAY 2D ------
def make_overlays_per_fluid(
    overlays_seq: List[tuple[pd.DataFrame, str]],  # (df, "t=..s")
    fluid_name: str,
    out_dir: Path,
    metrics: List[str],
    include_pressure: bool,
    overlay_every: int = 1,
    outfile_suffix: str = "",                     # np. "F2_p3_t10-11s_1e-5s"
) -> None:
    if not overlays_seq:
        return
    ensure_dir(out_dir)

    sel_idx = list(range(len(overlays_seq)))
    if overlay_every > 1 and len(sel_idx) > 2:
        keep = {0, len(sel_idx)-1}
        keep.update(i for i in sel_idx if i % overlay_every == 0)
        sel_idx = sorted(keep)

    spec_keys: List[str] = []
    if "h" in metrics: spec_keys.append("h")
    if "f" in metrics: spec_keys.append("f")
    if include_pressure: spec_keys.append("dp_band")

    for key in spec_keys:
        col = {"h": COL_H_WM2K, "f": COL_F, "dp_band": COL_DP_BAND}[key]
        ylabel = YLABEL_MAP[key]
        prefix = PREFIX_MAP[key]  # "overlay_h" / "overlay_f" / "overlay_dp_band"

        fig, ax = plt.subplots()
        xmax_mm, plotted = 0.0, 0

        for i in sel_idx:
            df, _ = overlays_seq[i]
            if col not in df.columns:
                continue
            axis_col = pick_axis_col(df)
            if axis_col is None:
                continue

            x_raw = df[axis_col].astype(float).values
            y = df[col].values
            x0 = (x_raw.max() - x_raw) if (fluid_name == "Fluid2") else (x_raw - x_raw.min())
            x_mm = x0 * 1000.0

            ax.plot(x_mm, y, marker="o", ms=MS, lw=LW)
            plotted += 1
            if x_mm.size:
                xmax_mm = max(xmax_mm, float(x_mm.max()))

        if not plotted:
            plt.close(fig)
            continue

        ax.set_xlim(0.0, xmax_mm)
        ax.set_xlabel("Water: from In to Out [mm]" if fluid_name == "Fluid2" else "Air: from In to Out [mm]")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

        fname = f"{prefix}_{outfile_suffix}.png" if outfile_suffix else f"{prefix}.png"
        fig.savefig(out_dir / fname, dpi=DEFAULT_DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] Overlay → {fname}")  # tylko nazwa pliku


# ------ MEAN: jedna figura = RAW + wszystkie MA ------
def plot_mean_with_mas(
    df: pd.DataFrame,
    fluid_name: str,
    out_dir: Path,
    metric_key: str,           # "h" | "f" | "dp_sum"
    base_col: str,             # np. "mean_h[W/m2K]"
    ma_cols: List[str],        # np. ["mean_h_ma2[W/m2K]", "mean_h_ma3[W/m2K]"]
    outfile_suffix: str = "",  # np. "F2_p3_t10-11s_1e-5s"
) -> None:
    """
    Rysuje na JEDNYM wykresie: serię surową + wszystkie MA z listy.
    Zapis do: mean_<metric>[_maW-W2-... ]_<suffix>.png
              np. mean_h_ma2-3-4_F1_p12_t2-10s_1e-5s.png
    """
    ensure_dir(out_dir)
    if "Time[s]" not in df.columns:
        print("[WARN] Brak 'Time[s]' → pomijam mean plot.")
        return

    gdf = df[df["Fluid"] == fluid_name].sort_values("Time[s]")
    if gdf.empty or (base_col not in gdf.columns):
        return

    import numpy as np, re

    x_raw = gdf["Time[s]"].values
    y_base = gdf[base_col].values

    # RAW
    m_base = np.isfinite(y_base)
    if not m_base.any():
        return

    fig, ax = plt.subplots()
    ax.plot(x_raw[m_base], y_base[m_base], marker="o", ms=MS, lw=LW, label="raw")

    # MA
    ma_windows_found: List[int] = []
    for col in ma_cols:
        if col not in gdf.columns:
            continue
        y_ma = gdf[col].values
        m_ma = np.isfinite(y_ma)
        if not m_ma.any():
            continue
        # wyciągnij numer okna z nazwy kolumny: _ma(\d+)]
        win_label = "MA"
        win_num = None
        m = re.search(r"_ma(\d+)\]", col)
        if m:
            win_num = int(m.group(1))
            win_label = f"MA{win_num}"
            ma_windows_found.append(win_num)
        ax.plot(x_raw[m_ma], y_ma[m_ma], marker="o", ms=MS, lw=LW, label=win_label)

    ax.set_xlabel("Time [s]")
    ax.set_ylabel(YLABEL_MAP[metric_key])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", frameon=False)

    # Skala Y
    if not (fluid_name == "Fluid2" and metric_key == "h"):
        y_all = [y_base[m_base]]
        for col in ma_cols:
            if col in gdf.columns:
                y = gdf[col].values
                m = np.isfinite(y)
                if m.any():
                    y_all.append(y[m])
        y_concat = np.concatenate(y_all)
        y_min, y_max = float(np.min(y_concat)), float(np.max(y_concat))
        pad = 2.0 * max(1.0, abs(y_max)) if y_max == y_min else (y_max - y_min)
        ax.set_ylim(y_min - pad, y_max + pad)

    # --- NAZWA PLIKU: dołóż tag MA, jeśli są serie MA ---
    prefix = f"mean_{'f' if metric_key == 'f' else metric_key}"  # mean_h / mean_f / mean_dp_sum
    ma_tag = ""
    if ma_windows_found:
        ma_windows_found = sorted(set(ma_windows_found))
        ma_tag = "_ma" + "-".join(str(w) for w in ma_windows_found)

    fname = f"{prefix}{ma_tag}_{outfile_suffix}.png" if outfile_suffix else f"{prefix}{ma_tag}.png"
    fig.savefig(out_dir / fname, dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Mean → {fname}")  # tylko nazwa pliku
