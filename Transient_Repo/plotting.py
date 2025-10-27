# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional
import re
import pandas as pd
import matplotlib.pyplot as plt

DEFAULT_DPI = 160
MS = 2  # mniejsze znaczniki

# nazwy kolumn
COL_Z_M     = "z [m]"
COL_Y_M     = "y [m]"
COL_H_WM2K  = "h[W/m2K]"
COL_F       = "f_fanning[-]"
COL_DP_BAND = "Δp_band[Pa]"
COL_DP_SUM  = "Δp_sum[Pa]"

AXIS_CANDIDATES = (COL_Z_M, COL_Y_M)

def _ensure_dir(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)

def _legend_below(ax, title: Optional[str] = None):
    ax.legend(title=title, loc="upper center", bbox_to_anchor=(0.5, -0.16),
              ncol=4, frameon=False)

def _pick_axis_col(df: pd.DataFrame) -> Optional[str]:
    for c in AXIS_CANDIDATES:
        if c in df.columns:
            return c
    return None

def _sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)

# ---------- OVERLAY: tylko h, f, Δp_band ----------
def make_overlays_per_fluid(bands_by_step: Dict[int, pd.DataFrame],
                            fluid_name: str,
                            out_dir: Path) -> None:
    """Overlaye (po osi z/y): h, f, Δp_band. Legenda = Step. Osobno dla Fluid1/Fluid2."""
    if not bands_by_step:
        return
    _ensure_dir(out_dir)

    specs = [
        (COL_H_WM2K,  "h [W/m²K]",      False, f"overlay_h__{fluid_name}.png"),
        (COL_F,       "f_Fanning [-]",  False, f"overlay_f__{fluid_name}.png"),
        (COL_DP_BAND, "Δp (band) [Pa]", True,  f"overlay_dp_band__{fluid_name}.png"),
    ]

    for col, ylabel, reverse_x, fname in specs:
        fig, ax = plt.subplots()
        plotted = 0
        axis_name_for_xlabel = None

        for step in sorted(bands_by_step.keys()):
            df = bands_by_step[step]
            axis_col = _pick_axis_col(df)
            if axis_col is None or col not in df.columns:
                continue

            x = df[axis_col].values
            y = df[col].values
            axis_name_for_xlabel = axis_col

            # dla Δp wygodniej od wylotu do wlotu, jeśli oś rośnie
            if reverse_x and x[0] < x[-1]:
                x = x[::-1]; y = y[::-1]

            ax.plot(x, y, marker="o", ms=MS, lw=1, label=f"S{step:05d}")
            plotted += 1

        if plotted == 0:
            plt.close(fig)
            continue

        ax.set_xlabel(axis_name_for_xlabel or "coord [m]")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        _legend_below(ax, title=f"{fluid_name}: Step")
        out_path = out_dir / fname
        fig.savefig(out_path, dpi=DEFAULT_DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] Overlay → {out_path}")

# ---------- MEAN: tylko mean_h, mean_f, mean_Δp_sum — WYŁĄCZNIE vs Time[s] ----------
def plot_summary_vs_step(df: pd.DataFrame, out_dir: Path, dt: Optional[float] = None):
    """Rysuje mean_h, mean_f, mean_Δp_sum vs Time[s]. Osobno dla Fluid1/Fluid2.
    Wyjątek: mean h dla Fluid2 – bez zmiany zakresu Y. Pozostałe – zakres rozszerzony o 200%."""
    _ensure_dir(out_dir)

    if "Time[s]" not in df.columns:
        print("[ERR] Brak kolumny 'Time[s]' w summary. Ustaw dt_for_plots w main.py (DT_S != None).")
        return

    desired = {"mean_h[W/m2K]", "mean_f_fanning[-]", "mean_Δp_sum[Pa]"}
    mean_cols = [c for c in df.columns if c in desired]
    if not mean_cols:
        print("[INFO] Brak kolumn mean_h/mean_f/mean_Δp_sum do wykreślenia.")
        return

    def _ylabel(col: str) -> str:
        return col.replace("mean_", "").replace("[", " [")

    fluids = sorted(df["Fluid"].unique()) if "Fluid" in df.columns else [None]
    for fluid in fluids:
        gdf = df[df["Fluid"] == fluid] if fluid is not None else df
        gdf = gdf.sort_values("Time[s]")
        tag = f"__{fluid}" if fluid is not None else ""

        for col in mean_cols:
            y = gdf[col].values
            if y.size == 0:
                continue
            y_min, y_max = float(y.min()), float(y.max())

            fig, ax = plt.subplots()
            ax.plot(gdf["Time[s]"].values, y, marker="o", ms=MS, lw=1)
            ax.set_xlabel("Time [s]")
            ax.set_ylabel(_ylabel(col))
            ax.grid(True, alpha=1.0)

            # — skala Y:
            # 1) Nie zmieniaj skali dla mean h w Fluid2
            if not (fluid == "Fluid2" and col == "mean_h[W/m2K]"):
                if y_max == y_min:
                    # sensowny zapas przy płaskiej serii: 200% wartości lub co najmniej 2.0
                    pad = 30.0 * max(1.0, abs(y_max))
                else:
                    # 200% rozszerzenia = dodajemy pełny zakres po obu stronach (nowy zakres = 3×)
                    pad = 10.0 * (y_max - y_min)
                ax.set_ylim(y_min - pad, y_max + pad)

            out_path = out_dir / f"{_sanitize(col)}__vs__Time_s{tag}.png"
            fig.savefig(out_path, dpi=DEFAULT_DPI, bbox_inches="tight")
            plt.close(fig)
            print(f"[OK] {out_path}")

