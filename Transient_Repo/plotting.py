# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re
import pandas as pd
import matplotlib.pyplot as plt

DEFAULT_DPI = 160
MS = 2  # marker size (global)

# nazwy kolumn
COL_Z_M     = "z [m]"
COL_Y_M     = "y [m]"
COL_H_WM2K  = "h[W/m2K]"
COL_F       = "f_fanning[-]"
COL_DP_BAND = "Δp_band[Pa]"
COL_DP_SUM  = "Δp_sum[Pa]"

LW_size = 0.3

AXIS_CANDIDATES = (COL_Z_M, COL_Y_M)

def set_plot_defaults(
    dpi: int = 160,
    marker_size: float = 2.0,   # float
    line_width: float = 1.0,
    view3d_elev: int = 25,
    view3d_azim: int = -135
) -> None:
    global DEFAULT_DPI, MS, LW, VIEW_ELEV, VIEW_AZIM
    DEFAULT_DPI = dpi
    MS = float(marker_size)
    LW = float(line_width)
    VIEW_ELEV = int(view3d_elev)
    VIEW_AZIM = int(view3d_azim)

def _ensure_dir(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)

def _legend_below(ax, title: Optional[str] = None):
    # pełna legenda; możesz ograniczyć ręcznie liczbę kolumn lub pominąć przy dużej liczbie linii
    ax.legend(title=title, loc="upper center", bbox_to_anchor=(0.5, -0.16),
              ncol=4, frameon=False)

def _pick_axis_col(df: pd.DataFrame) -> Optional[str]:
    for c in AXIS_CANDIDATES:
        if c in df.columns:
            return c
    return None

def _sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)

# ---------------- OVERLAYS ----------------

def _compose_overlay_specs(metrics: List[str], include_pressure: bool) -> List[Tuple[str, str, str]]:
    """
    Zwraca listę specyfikacji (col, ylabel, prefix) do rysowania overlay:
      - metrics=['h','f'] ogranicza do tych metryk,
      - include_pressure=True dorzuca Δp_band.
    """
    specs: List[Tuple[str, str, str]] = []
    if "h" in metrics:
        specs.append((COL_H_WM2K,  "h [W/m²K]",      "overlay_h"))
    if "f" in metrics:
        specs.append((COL_F,       "f_Fanning [-]",  "overlay_f"))
    if include_pressure:
        specs.append((COL_DP_BAND, "Δp (band) [Pa]", "overlay_dp_band"))
    return specs

def make_overlays_per_fluid(
    overlays_seq: List[Tuple[pd.DataFrame, str]],  # (df, label) posortowane po czasie
    fluid_name: str,
    out_dir: Path,
    metrics: List[str],
    include_pressure: bool,
    overlay_every: int = 1,
    outfile_tag: str = "",
) -> None:
    """
    Overlay (profil po osi z/y):
      - X zawsze od 0 do L (w mm), 0 = Inlet.
      - Fluid1 (Air): 0 ↦ min współrzędnej (In→Out).
      - Fluid2 (Water): 0 ↦ max współrzędnej (In→Out).
      - Bez legendy (za dużo serii).
    """
    if not overlays_seq:
        return
    _ensure_dir(out_dir)

    specs = _compose_overlay_specs(metrics, include_pressure)

    # thinning serii
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
        x_all_max_mm = 0.0

        for i in sel:
            df, lbl = overlays_seq[i]
            if col not in df.columns:
                continue
            axis_col = _pick_axis_col(df)
            if axis_col is None:
                continue

            x_raw = df[axis_col].values.astype(float)
            y = df[col].values

            # 0 = Inlet:
            #  - Fluid1: inlet przy najmniejszej współrzędnej (min)
            #  - Fluid2: inlet przy największej współrzędnej (max)
            if fluid_name == "Fluid2":
                x0 = x_raw.max() - x_raw   # 0 .. L (In→Out)
            else:
                x0 = x_raw - x_raw.min()   # 0 .. L (In→Out)

            # skala mm
            x_mm = x0 * 1000.0

            ax.plot(x_mm, y, marker="o", ms=MS, lw=LW_size)  # bez legendy
            plotted += 1

            x_all_max_mm = max(x_all_max_mm, float(x_mm.max()))

        if plotted == 0:
            plt.close(fig)
            continue

        ax.set_xlim(0.0, x_all_max_mm)

        # etykiety osi
        if fluid_name == "Fluid2":
            ax.set_xlabel("Water: from In to Out [mm]")
        else:
            ax.set_xlabel("Air: from In to Out [mm]")

        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        # brak legendy na overlayach

        fname = f"{prefix}__{outfile_tag}.png" if outfile_tag else f"{prefix}.png"
        out_path = out_dir / fname
        fig.savefig(out_path, dpi=DEFAULT_DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] Overlay → {out_path}") 

# ---------------- MEANS vs TIME ----------------

def _compose_mean_specs(metrics: List[str], include_pressure: bool) -> List[Tuple[str, str, str]]:
    """
    Zwraca listę specyfikacji (col, ylabel, prefix) do mean vs Time:
      - metrics=['h','f'] ogranicza do tych meanów,
      - include_pressure=True dorzuca mean_Δp_sum.
    """
    specs: List[Tuple[str, str, str]] = []
    if "h" in metrics:
        specs.append(("mean_h[W/m2K]", "h [W/m²K]", "mean_h"))
    if "f" in metrics:
        specs.append(("mean_f_fanning[-]", "f_Fanning [-]", "mean_f"))
    if include_pressure:
        specs.append(("mean_Δp_sum[Pa]", "Δp (sum) [Pa]", "mean_dp_sum"))
    return specs

def plot_means_vs_time(
    df: pd.DataFrame,
    fluid_name: str,
    out_dir: Path,
    metrics: List[str],
    include_pressure: bool,
    outfile_tag: str = "",
) -> None:
    """
    Rysuje mean_* vs Time[s] osobno dla wskazanego fluida.
    Zasady zakresu Y (na bazie poprzednich ustaleń):
      - mean h dla Fluid2: bez zmiany skali osi Y
      - pozostałe (h Fluid1, f, Δp_sum): rozszerzenie zakresu o 200%
    """
    _ensure_dir(out_dir)
    if "Time[s]" not in df.columns:
        print("[WARN] Brak 'Time[s]' → pomijam mean plots.")
        return

    gdf = df[df["Fluid"] == fluid_name].sort_values("Time[s]")
    if gdf.empty:
        return

    specs = _compose_mean_specs(metrics, include_pressure)

    for col, ylabel, prefix in specs:
        if col not in gdf.columns:
            continue

        y = gdf[col].values
        x = gdf["Time[s]"].values
        if y.size == 0:
            continue

        y_min, y_max = float(y.min()), float(y.max())

        fig, ax = plt.subplots()
        ax.plot(x, y, marker="o", ms=MS, lw=LW_size)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

        # Skala Y: wyjątek dla mean h w Fluid2 (bez zmiany),
        # reszta: rozszerzenie o 200% (dodajemy pełny zakres po obu stronach)
        if not (fluid_name == "Fluid2" and col == "mean_h[W/m2K]"):
            if y_max == y_min:
                pad = 2.0 * max(1.0, abs(y_max))
            else:
                pad = (y_max - y_min)
            ax.set_ylim(y_min - pad, y_max + pad)

        fname = f"{prefix}__{outfile_tag}.png" if outfile_tag else f"{prefix}.png"
        out_path = out_dir / fname
        fig.savefig(out_path, dpi=DEFAULT_DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] Mean → {out_path}")

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib import cm
import numpy as np
import re
import matplotlib.pyplot as plt

# --- INTERAKTYWNY 3D (Plotly) ---
def make_overlay3d_f_fluid1_interactive(
    overlays_seq,   # List[Tuple[pd.DataFrame, str]]  (df, "t=...s")
    out_dir: Path,
    outfile_tag: str = "",
    y_fixed_min_mm: float = 0.0,
    y_fixed_max_mm: float = 60.0,
    stride_time: int = 1,       # przerzedzanie po czasie (co ile profili)
    stride_x: int = 1           # przerzedzanie po osi X (co ile punktów)
) -> None:
    """
    Interaktywny 3D overlay tylko dla f (Fluid1), zapis do HTML:
      X: Time [s]   |  Y: Air: from In to Out [mm]   |  Z: f_Fanning [-]
      Y sztywno 0..60 mm. Obracanie/zoom w przeglądarce.
    """
    try:
        import numpy as np
        import plotly.graph_objects as go
    except Exception as e:
        print(f"[WARN] Plotly niedostępne: {e} → pomijam interaktywny wykres.")
        return

    if not overlays_seq:
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    xs_mm_rows, f_rows, ts = [], [], []
    axis_col = None
    base_len = None

    for k, (df, lbl) in enumerate(overlays_seq[::max(1, stride_time)]):
        if COL_F not in df.columns:
            continue
        if axis_col is None:
            axis_col = _pick_axis_col(df)
            if axis_col is None:
                continue

        x_raw = df[axis_col].values.astype(float)
        y_mm = (x_raw - x_raw.min()) * 1000.0         # 0..L mm (In→Out)
        fval = df[COL_F].values

        m = re.search(r"t=([0-9.]+)s", lbl)
        t = float(m.group(1)) if m else 0.0

        if base_len is None:
            base_len = min(len(y_mm), len(fval))
        n = min(base_len, len(y_mm), len(fval))

        y_mm = y_mm[:n: max(1, stride_x)]
        fval = fval[:n: max(1, stride_x)]

        xs_mm_rows.append(y_mm)
        f_rows.append(fval)
        ts.append(t)

    if not f_rows:
        return

    # Macierze dla surface: X = czas, Y = mm, Z = f
    X = np.tile(np.array(ts)[:, None], (1, len(xs_mm_rows[0])))
    Y = np.vstack(xs_mm_rows)
    Z = np.vstack(f_rows)

    # Interaktywna powierzchnia
    fig = go.Figure(data=[
        go.Surface(
            x=X, y=Y, z=Z,
            colorscale="Viridis",
            colorbar=dict(title="f_Fanning [-]"),
            showscale=True
        )
    ])
    fig.update_scenes(
        xaxis_title="Time [s]",
        yaxis_title="Air: from In to Out [mm]",
        zaxis_title="f_Fanning [-]",
        yaxis=dict(range=[y_fixed_min_mm, y_fixed_max_mm])
    )
    # Domyślny widok (możesz przestawić niżej)
    fig.update_layout(
        scene_camera=dict(eye=dict(x=1.7, y=1.5, z=1.2)),
        margin=dict(l=0, r=0, t=30, b=0),
        title="Interactive 3D: f (Fluid1)"
    )

    fname = f"overlay3d_f_interactive__F1__{outfile_tag}.html" if outfile_tag else "overlay3d_f_interactive__F1.html"
    out_path = out_dir / fname
    fig.write_html(str(out_path), include_plotlyjs="cdn", full_html=True)
    print(f"[OK] 3D Overlay (interactive, f, F1) → {out_path}")
