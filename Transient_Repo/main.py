# -*- coding: utf-8 -*-
# Transient SRP → CSV + wykresy 2D (overlay/mean)
# Zmiany:
# - brak 3D,
# - nazwy plików: <prefix>_<F?>_<p..>_t<t0>-<t1>s_<dt>s.png
# - logi bez pełnych ścieżek,
# - okno czasu: ALL / N kroków / D sekund,
# - MEAN: surowa + wszystkie MA NA JEDNYM WYKRESIE.

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse, os, tempfile, pandas as pd

from srp_parser import parse_srp
from compute import build_planes_df, build_bands_df, compute_band_table, compute_global_means

from config import DEFAULT_BASE_DATA_DIR, DEFAULT_OUT_DIR, PARTS, PLOT_DEFAULTS, PLOT_JOBS, FLUID_CFG
from plotting import set_plot_defaults, make_overlays_per_fluid, plot_mean_with_mas
from constants import MEAN_COL_MAP
from tpms_utils import (
    collect_all_data, filter_by_local_window, filter_by_global_window,
    normalize_prefixes, outfile_suffix, resolve_time_window_for_job, parts_tag,
)

# ---------- budowa tabel z jednego pliku ----------
def _build_bands_and_means(p: Path, fluid: str):
    cfg = FLUID_CFG[fluid]
    axis_label: str   = str(cfg["axis"])
    step_abs_len: float = float(cfg["step"])
    a_min: float = float(cfg["min"])
    a_max: float = float(cfg["max"])
    sgn = 1.0 if a_max >= a_min else -1.0

    raw = p.read_text(encoding="utf-8", errors="ignore")
    norm = normalize_prefixes(raw)

    with tempfile.NamedTemporaryFile("w", suffix=".srp", delete=False, encoding="utf-8") as tf:
        tf.write(norm); tmp_path = Path(tf.name)
    try:
        data = parse_srp(tmp_path)
    finally:
        try: os.remove(tmp_path)
        except OSError: pass

    planes_df = build_planes_df(data, dz=step_abs_len)
    bands_df  = build_bands_df(data)
    band_tab  = compute_band_table(planes_df, bands_df, dz=step_abs_len, axis_label=axis_label)

    # środek pasm → kolumna osi
    n_bands = len(band_tab)
    plane_coords = [a_min + sgn * i * step_abs_len for i in range(n_bands + 1)]
    centers = [(plane_coords[i] + plane_coords[i+1]) * 0.5 for i in range(n_bands)]
    col_axis = f"{axis_label} [m]"
    for old in ("z [m]", "y [m]"):
        if old in band_tab.columns and old != col_axis:
            band_tab.drop(columns=[old], inplace=True)
    band_tab[col_axis] = centers

    means = compute_global_means(band_tab)
    return band_tab, means

# ---------- MAIN ----------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir",  type=str, default=DEFAULT_BASE_DATA_DIR, help="Bazowy katalog z partami")
    ap.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR,       help="Katalog wyjściowy (CSV/plots)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    base_dir = Path(args.in_dir).resolve()
    out_dir  = Path(args.out_dir).resolve()
    (out_dir / "csv").mkdir(parents=True, exist_ok=True)
    plots_root = out_dir / "plots"; plots_root.mkdir(parents=True, exist_ok=True)

    set_plot_defaults(
        dpi=int(PLOT_DEFAULTS.get("dpi", 160)),
        marker_size=float(PLOT_DEFAULTS.get("marker_size", 2.0)),
        line_width=float(PLOT_DEFAULTS.get("line_width", 0.8)),
    )
    overlay_every    = int(PLOT_DEFAULTS.get("overlay_every", 1))
    include_pressure = bool(PLOT_DEFAULTS.get("include_pressure", False))

    index = collect_all_data(base_dir, PARTS)
    cache_bands: Dict[Tuple[str, str, Path], pd.DataFrame] = {}
    cache_means: Dict[Tuple[str, str, Path], Dict[str, float]] = {}

    for job in PLOT_JOBS:
        parts   = job["parts"];   fluids  = job["fluids"]
        plots   = job["plots"];   metrics = job["metrics"]
        t0_s    = job.get("t0_s", None)
        n_steps = job.get("n_steps", None)
        duration_s = job.get("duration_s", None)

        # Ustawienia MA (opcjonalne, tylko dla mean)
        ma_fluids   = job.get("mean_ma_fluids", []) or []
        ma_windows  = [int(w) for w in (job.get("mean_ma_windows", []) or []) if int(w) > 0]
        ma_edges    = job.get("mean_ma_edges", "strict")
        ma_center   = bool(job.get("mean_ma_center", True))
        min_periods = (lambda W: W if ma_edges == "strict" else 1)

        # okno czasu i dt_ref (etykiety)
        try:
            mode, t0_for_name, t1_for_name, dt_ref = resolve_time_window_for_job(parts, PARTS, t0_s, n_steps, duration_s)
        except ValueError as e:
            print(f"[WARN] Job {parts}: {e} → pomijam.")
            continue

        job_dir = plots_root / f"{parts_tag(parts)}_{t0_for_name:g}-{t1_for_name:g}s_{mode}"
        job_dir.mkdir(parents=True, exist_ok=True)

        if args.verbose:
            dts = {float(PARTS[p]["dt_sim_s"]) for p in parts if p in PARTS}
            if len(dts) > 1:
                print(f"[INFO] Job {parts}: różne dt {sorted(dts)} → używam dt={dt_ref:g} do nazw.")

        for fluid in fluids:
            overlays_triplets: List[Tuple[float, pd.DataFrame, str]] = []
            summary_rows: List[Dict] = []

            for part in parts:
                if part not in index:
                    if args.verbose: print(f"[INFO] Brak danych dla {part} → pomijam.")
                    continue

                t_start = float(PARTS[part]["t_start_s"])
                t_end   = float(PARTS[part]["t_end_s"])
                items   = index[part].get(fluid, [])
                if mode == "ALL":
                    items_sel = filter_by_local_window(items, t_end - t_start)
                else:
                    items_sel = filter_by_global_window(items, t_start, t0_for_name, t1_for_name)

                if overlay_every > 1 and len(items_sel) > 2:
                    keep = {0, len(items_sel)-1}
                    keep.update(i for i in range(len(items_sel)) if i % overlay_every == 0)
                    items_sel = [items_sel[i] for i in sorted(keep)]

                for (t_local, part_name, path) in items_sel:
                    key = (part_name, fluid, path)
                    if key not in cache_bands:
                        band_tab, means = _build_bands_and_means(path, fluid)
                        cache_bands[key] = band_tab; cache_means[key] = means
                    else:
                        band_tab = cache_bands[key]; means = cache_means[key]

                    t_global = t_start + t_local
                    overlays_triplets.append((t_global, band_tab, f"t={t_global:.3f}s"))

                    row = {"Fluid": fluid, "Time[s]": t_global}; row.update(means)
                    summary_rows.append(row)

            overlays_triplets.sort(key=lambda t: t[0])
            overlays_seq = [(df, lbl) for (_tg, df, lbl) in overlays_triplets]
            df_means = (pd.DataFrame(summary_rows)
                        .sort_values(["Fluid", "Time[s]"])
                        .reset_index(drop=True)) if summary_rows else pd.DataFrame()

            if not overlays_seq and df_means.empty:
                print(f"[INFO] Job {parts} | {fluid}: brak danych w oknie {t0_for_name:g}–{t1_for_name:g}s ({mode}).")
                continue

            suffix = outfile_suffix(fluid, parts, t0_for_name, t1_for_name, dt_ref)

            # ---- OVERLAY ----
            if ("overlay" in plots) and overlays_seq:
                make_overlays_per_fluid(
                    overlays_seq=overlays_seq,
                    fluid_name=fluid,
                    out_dir=job_dir,
                    metrics=metrics,
                    include_pressure=include_pressure,
                    overlay_every=overlay_every,
                    outfile_suffix=suffix,
                )

            # ---- MEAN: jedna figura = RAW + wszystkie MA ----
            if ("mean" in plots) and (not df_means.empty):
                # wylicz MA do kolumn (tylko dla wskazanych fluidów)
                if (fluid in ma_fluids) and ma_windows:
                    for m in metrics:
                        if m not in MEAN_COL_MAP:
                            continue
                        base_col = MEAN_COL_MAP[m]
                        gmask = (df_means["Fluid"] == fluid)
                        gdf = df_means.loc[gmask].sort_values("Time[s]").copy()
                        if base_col not in gdf.columns:
                            continue
                        for W in ma_windows:
                            col_ma = base_col.replace("]", f"_ma{W}]")
                            s_ma = gdf[base_col].rolling(window=W, center=ma_center, min_periods=min_periods(W)).mean()
                            df_means.loc[gmask, col_ma] = s_ma.values

                # rysuj: surowe + wszystkie dostępne MA na jednym wykresie
                for m in metrics:
                    if m not in MEAN_COL_MAP:
                        continue
                    base_col = MEAN_COL_MAP[m]

                    # zbierz wszystkie kolumny MA tej metryki, które faktycznie istnieją
                    ma_cols: List[str] = []
                    if (fluid in ma_fluids) and ma_windows:
                        for W in ma_windows:
                            col_ma = base_col.replace("]", f"_ma{W}]")
                            if col_ma in df_means.columns:
                                ma_cols.append(col_ma)

                    plot_mean_with_mas(
                        df=df_means,
                        fluid_name=fluid,
                        out_dir=job_dir,
                        metric_key=m,
                        base_col=base_col,
                        ma_cols=ma_cols,
                        outfile_suffix=suffix,
                    )

            # ---- CSV (z kolumnami MA, jeśli liczone) ----
            if not df_means.empty:
                csv_name = f"summary_means_{suffix}.csv"
                (out_dir / "csv" / csv_name).write_text(df_means.to_csv(index=False), encoding="utf-8")
                print(f"[OK] CSV → {csv_name}")  # tylko nazwa pliku

if __name__ == "__main__":
    main()
