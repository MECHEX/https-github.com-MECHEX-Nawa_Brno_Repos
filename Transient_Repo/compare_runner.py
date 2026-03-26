# -*- coding: utf-8 -*-
from __future__ import annotations

"""compare_runner.py

Wydzielony runner dla COMPARE_JOBS.

Motywacja
---------
Wcześniej logika porównań (budowa serii + rysowanie overlay/mean + steady)
była w `main.py`. To powodowało, że każda zmiana w warstwie wizualizacji
(np. style_map, figsize, dodatkowe opcje legendy) wymagała edycji main.

Ten moduł przenosi tę logikę do jednego miejsca. `main.py` powinien jedynie:
 - zbudować indeks danych transient,
 - utrzymywać cache'e,
 - przekazać konfigurację COMPARE_JOBS i callback `build_one`.

Wizualizacja (style/figsize) jest obsługiwana w `plotting.py` + `style.py`.
"""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from compare import build_overlay_profile_for_series, build_mean_timeseries_for_series
from constants import COL_Z_M, COL_Y_M, COL_H_WM2K, MEAN_COL_MAP

# --- compatibility: COL_F name differs across revisions ---
try:
    from constants import COL_F
except Exception:  # pragma: no cover
    from constants import COL_F_FANNING as COL_F

from plotting import plot_compare_overlay, plot_compare_mean, _sanitize


def _load_steady_profile(csv_path: Path, fluid: str, metric_key: str) -> Tuple[np.ndarray, np.ndarray]:
    """Ładuje profil steady (x po osi, y metryka) i mapuje oś na [mm] in→out."""
    df = pd.read_csv(csv_path)
    axis_name = COL_Z_M if COL_Z_M in df.columns else (COL_Y_M if COL_Y_M in df.columns else None)
    if axis_name is None:
        raise ValueError(f"Brak kolumny osi (z/y) w {csv_path.name}")

    col = COL_H_WM2K if metric_key == "h" else (COL_F if metric_key == "f" else None)
    if col is None or col not in df.columns:
        raise ValueError(f"Brak kolumny metryki ({metric_key}) w {csv_path.name}")

    x_raw = df[axis_name].astype(float).values
    y = df[col].astype(float).values

    # mapowanie osi: Air (Fluid1) rośnie od min; Water (Fluid2) odwrócone
    x0 = (x_raw.max() - x_raw) if fluid == "Fluid2" else (x_raw - x_raw.min())
    x_mm = x0 * 1000.0
    return x_mm, y


def _steady_means(csv_path: Path, mean_weights: dict) -> Dict[str, float]:
    """Średnie steady dla linii referencyjnych na wykresach mean."""
    df = pd.read_csv(csv_path)
    out: Dict[str, float] = {}

    if COL_H_WM2K in df.columns:
        wname = (mean_weights or {}).get("h") or "A_wet[m2]"
        if wname in df.columns and np.sum(df[wname].values) > 0:
            out["h"] = float(np.average(df[COL_H_WM2K].values, weights=df[wname].values))
        else:
            out["h"] = float(df[COL_H_WM2K].mean())

    if COL_F in df.columns:
        out["f"] = float(df[COL_F].mean())

    return out


def run_compare_jobs(
    *,
    compare_jobs: List[Dict],
    compare_root_dir: Path,
    PARTS: Dict[str, Dict[str, float]],
    index: Dict,
    cache_bands: Dict,
    cache_means: Dict,
    build_one: Callable,
    repo_root: Optional[Path] = None,
) -> None:
    """Uruchamia wszystkie COMPARE_JOBS.

    Parametry
    ---------
    compare_jobs : lista jobów z configu.
    compare_root_dir : katalog bazowy na wyniki Compare/...
    PARTS/index/cache_* : dane i cache'e z main.
    build_one : callback (part, fluid, srp_path)->(bands_df, means_dict)
    repo_root : root repo (dla ścieżek względnych steady); domyślnie katalog pliku.
    """
    compare_root_dir.mkdir(parents=True, exist_ok=True)
    repo_root = (repo_root or Path(__file__).resolve().parent)

    for job in (compare_jobs or []):
        job_name = str(job.get("name", "cmp"))
        plots = job.get("plots", ["overlay"]) or ["overlay"]
        metrics = job.get("metrics", ["h"]) or ["h"]
        mean_opts = job.get("mean", {}) or {}
        overlay_opts = job.get("overlay", {}) or {}

        # ---- steady ----
        steady_cfg = job.get("steady", {}) or {}
        steady_enabled = bool(steady_cfg.get("enabled", False))
        steady_use_ovl = bool(steady_cfg.get("use_in_overlay", True))
        steady_use_mean = bool(steady_cfg.get("use_in_mean", True))
        steady_cases = steady_cfg.get("cases", {}) or {}
        steady_weights = steady_cfg.get("mean_weights", {}) or {}
        steady_base = (repo_root / steady_cfg.get("base_csv_dir", r"..\Steady_Repo\DataProcessed\csv")).resolve()

        # ---- series split by fluid ----
        series = job.get("series", []) or []
        by_fluid: Dict[str, List[Dict]] = {"Fluid1": [], "Fluid2": []}
        for s in series:
            ss = dict(s)
            # kompatybilność z istniejącym compare.py: overlay opts trzymamy w serii
            ss["_overlay_opts_"] = overlay_opts
            by_fluid[str(ss.get("fluid", "Fluid1"))].append(ss)

        job_dir = compare_root_dir / _sanitize(job_name)
        job_dir.mkdir(parents=True, exist_ok=True)

        # =========================
        # overlay
        # =========================
        if "overlay" in plots:
            for metric_key in metrics:
                for fluid, sers in by_fluid.items():
                    profiles: List[Dict] = []

                    for s in sers:
                        try:
                            prf = build_overlay_profile_for_series(
                                series_cfg=s,
                                PARTS=PARTS,
                                index=index,
                                cache_bands=cache_bands,
                                build_one=build_one,
                                metric_key=metric_key,
                            )
                        except Exception as exc:
                            raise RuntimeError(
                                f"COMPARE_JOBS '{job_name}' overlay failed for metric '{metric_key}', "
                                f"fluid '{fluid}', series '{s.get('label')}', "
                                f"case_id '{s.get('case_id')}'."
                            ) from exc
                        profiles.append(prf)

                    if steady_enabled and steady_use_ovl:
                        for case in (steady_cases.get(fluid, []) or []):
                            csvp = (steady_base / case["file"]).resolve()
                            if not csvp.exists():
                                continue
                            try:
                                x_mm, y = _load_steady_profile(csvp, fluid, metric_key)
                                profiles.append({
                                    "label": f"{case['label']} SS",
                                    "fluid": fluid,
                                    "x_mm": x_mm,
                                    "y": y,
                                    "shade_lo": None,
                                    "shade_hi": None,
                                })
                            except Exception as e:
                                print(f"[WARN] STEADY overlay {csvp.name}: {e}")

                    if not profiles:
                        print(f"[INFO] Compare overlay: brak danych ({metric_key}, {fluid}).")
                        continue

                    plot_compare_overlay(
                        out_dir=job_dir,
                        metric_key=str(metric_key),
                        fluid_name=str(fluid),
                        series_profiles=profiles,
                        job=job,
                    )

        # =========================
        # mean
        # =========================
        if "mean" in plots:
            time_mode = str(mean_opts.get("time_mode", "global")).lower()
            ma_windows = [int(w) for w in (mean_opts.get("ma_windows", []) or []) if int(w) > 0]
            ma_center = bool(mean_opts.get("ma_center", True))
            ma_edges = str(mean_opts.get("ma_edges", "strict")).lower()
            show_raw = bool(mean_opts.get("show_raw", True))

            min_periods = (lambda W: W if ma_edges == "strict" else 1)

            use_strict_crop = (ma_edges == "strict" and bool(ma_windows))
            W_crop = max(ma_windows) if use_strict_crop else None

            for metric_key in metrics:
                base_col = MEAN_COL_MAP[str(metric_key)]

                for fluid, sers in by_fluid.items():
                    series_list: List[Dict] = []
                    t_first_valid: List[float] = []
                    t_last_valid: List[float] = []

                    for s in sers:
                        try:
                            dfm = build_mean_timeseries_for_series(
                                series_cfg=s,
                                PARTS=PARTS,
                                index=index,
                                cache_means=cache_means,
                                cache_bands=cache_bands,
                                build_one=build_one,
                            )
                        except Exception as exc:
                            raise RuntimeError(
                                f"COMPARE_JOBS '{job_name}' mean plot failed for metric '{metric_key}', "
                                f"fluid '{fluid}', series '{s.get('label')}', "
                                f"case_id '{s.get('case_id')}'."
                            ) from exc

                        gdf = dfm[dfm["Fluid"] == fluid].sort_values("Time[s]").copy()
                        if base_col not in gdf.columns or gdf.empty:
                            continue

                        t_global = gdf["Time[s]"].values.astype(float)
                        t_aligned = t_global - float(s["t0_s"])
                        t_plot = t_global if time_mode == "global" else t_aligned

                        y_raw = gdf[base_col].values.astype(float)

                        y_ma_dict: Dict[int, np.ndarray] = {}
                        for W in ma_windows:
                            y_ma = (
                                gdf[base_col]
                                .rolling(window=W, center=ma_center, min_periods=min_periods(W))
                                .mean()
                                .values.astype(float)
                            )
                            y_ma_dict[W] = y_ma

                        if W_crop is not None and W_crop in y_ma_dict:
                            yc = y_ma_dict[W_crop]
                            m_valid = np.isfinite(yc)
                            if m_valid.any():
                                ti = t_plot[m_valid]
                                t_first_valid.append(float(ti[0]))
                                t_last_valid.append(float(ti[-1]))

                        series_list.append({
                            "label": s["label"],
                            "t_global": t_global,
                            "t_aligned": t_aligned,
                            "t_plot": t_plot,
                            "y_raw": y_raw,
                            "y_ma": y_ma_dict if y_ma_dict else None,
                        })

                    # strict cropping: wspólny zakres tylko tam, gdzie MA istnieje dla wszystkich
                    if use_strict_crop and t_first_valid and t_last_valid:
                        t_min_crop = max(t_first_valid)
                        t_max_crop = min(t_last_valid)
                        if t_min_crop < t_max_crop:
                            for ss in series_list:
                                t = ss["t_plot"]
                                mask = (t >= t_min_crop - 1e-12) & (t <= t_max_crop + 1e-12)
                                ss["t_global"] = ss["t_global"][mask]
                                ss["t_aligned"] = ss["t_aligned"][mask]
                                ss["t_plot"] = ss["t_plot"][mask]
                                ss["y_raw"] = ss["y_raw"][mask]
                                if ss.get("y_ma"):
                                    for W in list(ss["y_ma"].keys()):
                                        ss["y_ma"][W] = ss["y_ma"][W][mask]

                    # steady reference lines
                    ref_lines: List[Tuple[str, float]] = []
                    if steady_enabled and steady_use_mean:
                        for case in (steady_cases.get(fluid, []) or []):
                            csvp = (steady_base / case["file"]).resolve()
                            if not csvp.exists():
                                continue
                            try:
                                mvals = _steady_means(csvp, steady_weights)
                                if str(metric_key) in mvals:
                                    ref_lines.append((str(case["label"]), float(mvals[str(metric_key)])))
                            except Exception as e:
                                print(f"[WARN] STEADY mean {csvp.name}: {e}")

                    if not series_list and not ref_lines:
                        print(f"[INFO] Compare mean: brak danych ({metric_key}, {fluid}).")
                        continue

                    plot_compare_mean(
                        out_dir=job_dir,
                        metric_key=str(metric_key),
                        fluid_name=str(fluid),
                        series_list=[{
                            "label": s["label"],
                            "t_global": s["t_global"],
                            "t_aligned": s["t_aligned"],
                            "y_raw": s["y_raw"],
                            "y_ma": s["y_ma"],
                        } for s in series_list],
                        job=job,
                        time_mode=time_mode,
                        show_raw=show_raw,
                        ref_lines=ref_lines,
                    )
