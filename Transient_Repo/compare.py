# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
import numpy as np
import pandas as pd

# Stałe kolumn (z constants.py)
from constants import (
    COL_Z_M, COL_Y_M,
    COL_H_WM2K, COL_F, COL_DP_BAND,
    MEAN_COL_MAP,
)

# ======= HELPERY LOKALNE (żeby plik był samowystarczalny) =======

def ensure_dir(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)

def pick_axis_col(df: pd.DataFrame) -> Optional[str]:
    if COL_Z_M in df.columns: return COL_Z_M
    if COL_Y_M in df.columns: return COL_Y_M
    return None

def filter_by_global_window(
    items: List[Tuple[float, str, Path]],
    t_start_part: float,
    t0_s: float, t1_s: float
) -> List[Tuple[float, str, Path]]:
    """
    items: [(t_local, part_name, path)], t_global = t_start_part + t_local
    Zwraca tylko te, dla których t_global ∈ [t0_s, t1_s].
    """
    out = []
    for (t_local, p, path) in items:
        tg = t_start_part + t_local
        if (tg >= t0_s - 1e-12) and (tg <= t1_s + 1e-12):
            out.append((t_local, p, path))
    return out

def _time_weights(t: np.ndarray, t0: float, t1: float) -> np.ndarray:
    """Wagi „trapezowe” w oknie [t0,t1] dla agregacji w czasie (overlay)."""
    n = len(t)
    if n <= 1:
        return np.ones(max(n, 1), dtype=float)
    edges = np.empty(n + 1, dtype=float)
    edges[0]  = max(t0, t[0] - 0.5*(t[1]-t[0]))
    edges[-1] = min(t1, t[-1] + 0.5*(t[-1]-t[-2]))
    for k in range(1, n):
        edges[k] = 0.5*(t[k-1] + t[k])
    w = np.clip(edges[1:] - edges[:-1], 0.0, None)
    s = w.sum()
    return w/s if s > 0 else np.ones_like(w)

# ======= OVERLAY: profil uśredniony w czasie dla 1 serii =======

def build_overlay_profile_for_series(
    series_cfg: Dict,
    PARTS: Dict[str, Dict[str, float]],
    index: Dict[str, Dict[str, List[Tuple[float, str, Path]]]],
    cache_bands: Dict[Tuple[str, str, Path], pd.DataFrame],
    build_one: Callable[[Path, str], Tuple[pd.DataFrame, Dict[str, float]]],
    metric_key: str,  # "h" | "f" | "dp_band"
) -> Optional[Dict]:
    label  = series_cfg["label"]
    fluid  = series_cfg["fluid"]
    parts  = series_cfg["parts"]
    t0_s   = float(series_cfg["t0_s"])
    t1_s   = float(series_cfg["t1_s"])

    overlay_opts = series_cfg.get("_overlay_opts_", {})
    mode    = (overlay_opts.get("time_avg", {}) or {}).get("mode", "mean")     # "mean"|"median"
    weights = (overlay_opts.get("time_avg", {}) or {}).get("weights", "auto")  # "auto"|"equal"
    shade   = overlay_opts.get("shade", None)                                   # None|"std"|[lo,hi]

    items_all: List[Tuple[float, Path, str]] = []
    for part in parts:
        if part not in index or fluid not in index[part]:
            continue
        t_start = float(PARTS[part]["t_start_s"])
        items = filter_by_global_window(index[part][fluid], t_start, t0_s, t1_s)
        for (t_local, part_name, path) in items:
            items_all.append((t_start + t_local, path, part_name))

    if not items_all:
        return None

    items_all.sort(key=lambda r: r[0])
    times = np.array([r[0] for r in items_all], dtype=float)

    w = _time_weights(times, t0_s, t1_s) if weights == "auto" else \
        np.ones(len(times), dtype=float)/len(times)

    col_map = {"h": COL_H_WM2K, "f": COL_F, "dp_band": COL_DP_BAND}
    metric_col = col_map[metric_key]

    X_mm_ref: Optional[np.ndarray] = None
    Ys: List[np.ndarray] = []
    axis_name = None

    for (_tg, path, part_name) in items_all:
        key = (part_name, fluid, path)
        if key not in cache_bands:
            band_tab, _means = build_one(path, fluid)
            cache_bands[key] = band_tab
        else:
            band_tab = cache_bands[key]

        axis_col = pick_axis_col(band_tab)
        if axis_col is None or metric_col not in band_tab.columns:
            continue

        x_raw = band_tab[axis_col].astype(float).values
        y     = band_tab[metric_col].astype(float).values

        # 0 mm = wlot; F2 odwracamy
        x0 = (x_raw.max() - x_raw) if fluid == "Fluid2" else (x_raw - x_raw.min())
        x_mm = x0 * 1000.0

        if X_mm_ref is None:
            X_mm_ref = x_mm
            axis_name = axis_col
        else:
            if len(x_mm) != len(X_mm_ref):
                n = min(len(x_mm), len(X_mm_ref))
                x_mm = x_mm[:n]; y = y[:n]; X_mm_ref = X_mm_ref[:n]

        Ys.append(y)

    if X_mm_ref is None or not Ys:
        return None

    Y = np.vstack(Ys)  # [n_snap, n_pts]
    if mode == "median":
        y_mean = np.median(Y, axis=0)
    else:
        wv = w[:Y.shape[0]][:, None]
        wv = wv / np.sum(wv)
        y_mean = np.sum(wv * Y, axis=0)

    shade_lo = shade_hi = None
    if isinstance(shade, str) and shade.lower() == "std":
        y_std = np.std(Y, axis=0)
        shade_lo, shade_hi = y_mean - y_std, y_mean + y_std
    elif isinstance(shade, (list, tuple)) and len(shade) == 2:
        p_lo, p_hi = float(shade[0]), float(shade[1])
        shade_lo = np.percentile(Y, p_lo, axis=0)
        shade_hi = np.percentile(Y, p_hi, axis=0)

    return {
        "label": label,
        "fluid": fluid,
        "x_mm": X_mm_ref,
        "y": y_mean,
        "shade_lo": shade_lo,
        "shade_hi": shade_hi,
        "axis_name": axis_name,
        "t0_s": t0_s, "t1_s": t1_s,
    }

# ======= MEAN: szereg czasowy dla 1 serii =======

def build_mean_timeseries_for_series(
    series_cfg: Dict,
    PARTS: Dict[str, Dict[str, float]],
    index: Dict[str, Dict[str, List[Tuple[float, str, Path]]]],
    cache_means: Dict[Tuple[str, str, Path], Dict[str, float]],
    cache_bands: Dict[Tuple[str, str, Path], pd.DataFrame],
    build_one: Callable[[Path, str], Tuple[pd.DataFrame, Dict[str, float]]],
) -> pd.DataFrame:
    """
    Jeśli brakuje cache_means[key], obliczamy (band_tab, means) i uzupełniamy oba cache.
    """
    label  = series_cfg["label"]
    fluid  = series_cfg["fluid"]
    parts  = series_cfg["parts"]
    t0_s   = float(series_cfg["t0_s"])
    t1_s   = float(series_cfg["t1_s"])

    rows: List[Dict] = []
    for part in parts:
        if part not in index or fluid not in index[part]:
            continue
        t_start = float(PARTS[part]["t_start_s"])
        items = filter_by_global_window(index[part][fluid], t_start, t0_s, t1_s)
        for (t_local, part_name, path) in items:
            key = (part_name, fluid, path)
            if (key in cache_bands) and (key in cache_means):
                means = cache_means[key]
            else:
                band_tab, means = build_one(path, fluid)
                cache_bands[key] = band_tab
                cache_means[key] = means

            t_global = t_start + t_local
            row = {"Label": label, "Fluid": fluid, "Time[s]": t_global}
            row.update(means)
            rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["Label", "Fluid", "Time[s]"] + list(MEAN_COL_MAP.values()))
    return pd.DataFrame(rows).sort_values(["Label", "Fluid", "Time[s]"]).reset_index(drop=True)
