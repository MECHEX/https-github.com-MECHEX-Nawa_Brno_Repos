
# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

# Stałe (lokalne, aby nie wymagać innych plików)
DEFAULT_DPI = 160
COL_BAND_ID = "band_id"
COL_Z_M     = "z [m]"
COL_H_WM2K  = "h[W/m2K]"
COL_RE      = "Re[-]"
COL_NU      = "Nu[-]"
COL_F       = "f_fanning[-]"
COL_DP_BAND = "Δp_band[Pa]"
COL_DP_SUM  = "Δp_sum[Pa]"

# --- utils ---
def _ensure_dir(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)

def _legend_below(ax, title: Optional[str] = None):
    ax.legend(title=title, loc="upper center", bbox_to_anchor=(0.5, -0.15),
              ncol=4, frameon=False)

# -------- overlaye po z (osobno Fluid1/Fluid2) --------
def make_overlays_per_fluid(bands_by_step: Dict[int, pd.DataFrame],
                            fluid_name: str,
                            out_dir: Path) -> None:
    """Rysuje overlaye (po z) dla kolumn [h, Re, Nu, f, Δp_band, Δp_sum].
       Legendą jest numer kroku (Step).
    """
    if not bands_by_step:
        return
    _ensure_dir(out_dir)

    specs = [
        (COL_H_WM2K,  "h [W/m²K]",   False, "overlay_h__{}.png"),
        (COL_RE,      "Re [-]",      False, "overlay_Re__{}.png"),
        (COL_NU,      "Nu [-]",      False, "overlay_Nu__{}.png"),
        (COL_F,       "f_Fanning [-]",False, "overlay_f__{}.png"),
        (COL_DP_BAND, "Δp (band) [Pa]", True, "overlay_dp_band__{}.png"),
        (COL_DP_SUM,  "Δp (cumulative) [Pa]", True, "overlay_dp_sum__{}.png"),
    ]

    # sortuj po Step dla spójnej palety
    for col, ylabel, reverse_x, fname_tpl in specs:
        fig, ax = plt.subplots()
        plotted = 0
        for step in sorted(bands_by_step.keys()):
            df = bands_by_step[step]
            if col not in df.columns or COL_Z_M not in df.columns:
                continue
            x = df[COL_Z_M].values
            y = df[col].values
            if reverse_x:
                ax.plot(x[::-1], y[::-1], marker="o", ms=3, lw=1, label=f"S{step:05d}")
            else:
                ax.plot(x, y, marker="o", ms=3, lw=1, label=f"S{step:05d}")
            plotted += 1

        if plotted == 0:
            plt.close(fig)
            continue

        ax.set_xlabel("z [m]")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        _legend_below(ax, title=f"{fluid_name}: Step")
        out_path = out_dir / (fname_tpl.format(fluid_name))
        fig.savefig(out_path, dpi=DEFAULT_DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] Overlay → {out_path}")

# -------- wykresy średnich vs Step / Time --------
def _sanitize(name: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', name)


def plot_summary_vs_step(df: pd.DataFrame, out_dir: Path, dt: Optional[float] = None):
    """Rysuje wykresy mean_* vs Step, oraz (jeśli dt podano) równoległe mean_* vs Time[s]."""
    _ensure_dir(out_dir)

    # znajdź kolumny ze średnimi
    mean_cols = [c for c in df.columns if c.startswith("mean_")]
    if not mean_cols:
        print("[INFO] Brak kolumn mean_* do wykreślenia.")
        return

    # 1) vs Step
    for col in mean_cols:
        fig, ax = plt.subplots()
        ax.plot(df["Step"].values, df[col].values, marker="o", lw=1)
        ax.set_xlabel("Step [-]")
        ax.set_ylabel(col.replace("mean_", "").replace("[", " ["))
        ax.grid(True, alpha=0.3)
        out_path = out_dir / f"{_sanitize(col)}__vs__Step.png"
        fig.savefig(out_path, dpi=DEFAULT_DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] {out_path}")

    # 2) vs Time[s], jeśli mamy dt
    if dt is not None and "Time[s]" in df.columns:
        for col in mean_cols:
            fig, ax = plt.subplots()
            ax.plot(df["Time[s]"].values, df[col].values, marker="o", lw=1)
            ax.set_xlabel("Time [s]")
            ax.set_ylabel(col.replace("mean_", "").replace("[", " ["))
            ax.grid(True, alpha=0.3)
            out_path = out_dir / f"{_sanitize(col)}__vs__Time_s.png"
            fig.savefig(out_path, dpi=DEFAULT_DPI, bbox_inches="tight")
            plt.close(fig)
            print(f"[OK] {out_path}")
