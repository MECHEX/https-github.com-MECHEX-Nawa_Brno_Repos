# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
import argparse, re, os, tempfile
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

from srp_parser import parse_srp
from compute import build_planes_df, build_bands_df, compute_band_table, compute_global_means
from plotting import (
    set_plot_defaults,
    make_overlays_per_fluid,
    plot_means_vs_time,
    plot_compare_overlay,
    plot_compare_mean, _sanitize
)

# konfiguracja i stałe projektu
from config import PARTS, FLUID_CFG, PLOT_DEFAULTS, PLOT_JOBS, COMPARE_JOBS
from constants import COL_Z_M, COL_Y_M, COL_H_WM2K, COL_F, COL_DP_BAND

# porównania – logika budowy danych z transient
from compare import (
    build_overlay_profile_for_series,
    build_mean_timeseries_for_series,
)

# ====== stałe wewnętrzne ======
STEP_RE = re.compile(r"_S(\d{4,})", re.IGNORECASE)

def _sanitize(s: str) -> str:
    import re as _re
    return _re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s))

def _extract_step(stem: str) -> int:
    m = STEP_RE.search(stem)
    return int(m.group(1)) if m else -1

def _fluid_from_name(stem: str) -> str:
    s = stem.lower()
    if "fluid1" in s or "_f1_" in s or s.endswith("_f1") or "f1_" in s:
        return "Fluid1"
    if "fluid2" in s or "_f2_" in s or s.endswith("_f2") or "f2_" in s:
        return "Fluid2"
    return "Fluid2" if "_f2" in s else "Fluid1"

def _normalize_prefixes(text: str) -> str:
    t = re.sub(r"\bf[12]_wall_band_", "wall_band_", text)
    t = re.sub(r"\bf[12]_(?:ziso_|yiso_)", "ziso_", t)
    t = re.sub(r"\bf[12]_env_", "env_", t)
    return t

def _fmt_time_tag(t0: float, t1: float) -> str:
    def _f(x: float) -> str:
        if abs(x - round(x)) < 1e-9: return f"{int(round(x))}"
        return f"{x:.3f}".rstrip("0").rstrip(".")
    return f"t{_f(t0)}-{_f(t1)}s"

def _parts_tag(parts: List[str]) -> str:
    nums = []
    for p in parts:
        m = re.search(r"part\s*([0-9]+)", p, re.IGNORECASE)
        nums.append(int(m.group(1)) if m else -1)
    nums = sorted([n for n in nums if n >= 0])
    return "p" + "".join(str(n) for n in nums) if nums else "p"

def _outfile_tag(fluid: str, parts: List[str], tmin: float, tmax: float, dt_tag: Optional[str]=None) -> str:
    ftag = "F1" if fluid == "Fluid1" else "F2"
    ptag = _parts_tag(parts)
    ttag = _fmt_time_tag(tmin, tmax)
    base = f"{ftag}_{ptag}_{ttag}"
    return f"{base}_{dt_tag}" if dt_tag else base

# ====== indeks plików SRP (transient) ======
def _collect_all_data(base_dir: Path) -> Dict[str, Dict[str, List[Tuple[float, str, Path]]]]:
    out: Dict[str, Dict[str, List[Tuple[float, str, Path]]]] = {}
    for part, meta in PARTS.items():
        dt = float(meta["dt_sim_s"])
        pdir = base_dir / part
        if not pdir.exists():
            continue
        for p in pdir.glob("*.srp"):
            step = _extract_step(p.stem)
            if step < 0:
                continue
            t_local = step * dt
            fluid = _fluid_from_name(p.stem)
            out.setdefault(part, {}).setdefault(fluid, []).append((t_local, part, p))
    for part in out:
        for fluid in out[part]:
            out[part][fluid].sort(key=lambda tup: tup[0])
    return out

def _filter_by_local_window(items: List[Tuple[float, str, Path]], t_len: float) -> List[Tuple[float, str, Path]]:
    return [tup for tup in items if (-1e-12) <= tup[0] <= (t_len + 1e-12)]

# ====== budowa tabel pasm i średnich (1 plik SRP) ======
def _build_bands_and_means(p: Path, fluid: str):
    cfg = FLUID_CFG[fluid]
    axis_label: str = str(cfg["axis"])
    step_abs_len: float = float(cfg["step"])
    a_min: float = float(cfg["min"])
    a_max: float = float(cfg["max"])
    sgn = 1.0 if a_max >= a_min else -1.0

    raw = p.read_text(encoding="utf-8", errors="ignore")
    norm = _normalize_prefixes(raw)

    with tempfile.NamedTemporaryFile("w", suffix=".srp", delete=False, encoding="utf-8") as tf:
        tf.write(norm)
        tmp_path = Path(tf.name)

    try:
        data = parse_srp(tmp_path)
    finally:
        try: os.remove(tmp_path)
        except OSError: pass

    planes_df = build_planes_df(data, dz=step_abs_len)
    bands_df  = build_bands_df(data)
    band_tab  = compute_band_table(planes_df, bands_df, dz=step_abs_len, axis_label=axis_label)

    n_bands = len(band_tab)
    plane_coords = [a_min + sgn * i * step_abs_len for i in range(n_bands + 1)]
    centers = [(plane_coords[i] + plane_coords[i+1]) * 0.5 for i in range(n_bands)]
    col_axis = f"{axis_label} [m]"
    for old in (COL_Z_M, COL_Y_M):
        if old in band_tab.columns and old != col_axis:
            band_tab.drop(columns=[old], inplace=True)
    band_tab[col_axis] = centers

    means = compute_global_means(band_tab)
    return band_tab, means

# ====== STEADY helpers (per-job) ======
def _load_steady_profile(csv_path: Path, fluid: str, metric_key: str):
    df = pd.read_csv(csv_path)
    axis_name = COL_Z_M if COL_Z_M in df.columns else (COL_Y_M if COL_Y_M in df.columns else None)
    if axis_name is None:
        raise ValueError(f"Brak kolumny osi (z/y) w {csv_path.name}")
    col = COL_H_WM2K if metric_key == "h" else (COL_F if metric_key == "f" else None)
    if col is None or col not in df.columns:
        raise ValueError(f"Brak kolumny metryki w {csv_path.name}")

    x_raw = df[axis_name].astype(float).values
    y     = df[col].astype(float).values
    x0 = (x_raw.max() - x_raw) if fluid == "Fluid2" else (x_raw - x_raw.min())
    x_mm = x0 * 1000.0
    return x_mm, y

def _steady_means(csv_path: Path, mean_weights: dict):
    df = pd.read_csv(csv_path)
    out = {}
    if COL_H_WM2K in df.columns:
        wname = (mean_weights or {}).get("h") or "A_wet[m2]"
        if wname in df.columns and np.sum(df[wname].values) > 0:
            out["h"] = float(np.average(df[COL_H_WM2K].values, weights=df[wname].values))
        else:
            out["h"] = float(df[COL_H_WM2K].mean())
    if COL_F in df.columns:
        out["f"] = float(df[COL_F].mean())
    return out

# ====== MAIN ======
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=str,
        default=str((Path(__file__).resolve().parent / "FluentTransientData")),
        help="Katalog bazowy z partami (part1, part2, ...)")
    ap.add_argument("--out-dir", type=str,
        default=str((Path(__file__).resolve().parent / "TransientFigs")),
        help="Katalog wyjściowy (csv/plots).")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    base_dir = Path(args.in_dir).resolve()
    out_dir  = Path(args.out_dir).resolve()
    csv_dir   = out_dir / "csv"
    plots_dir = out_dir / "plots"
    csv_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    set_plot_defaults(
        dpi=int(PLOT_DEFAULTS.get("dpi", 160)),
        marker_size=float(PLOT_DEFAULTS.get("marker_size", 2.0)),
        line_width=float(PLOT_DEFAULTS.get("line_width", 0.8)),
    )

    # ====== indeks transient + cache ======
    index = _collect_all_data(base_dir)
    cache_bands: Dict[Tuple[str, str, Path], pd.DataFrame] = {}
    cache_means: Dict[Tuple[str, str, Path], Dict[str, float]] = {}

    # ====== STANDARDOWE JOBY (PLOT_JOBS) ======
    for job in PLOT_JOBS:
        parts   = job["parts"]
        fluids  = job["fluids"]
        plots   = job["plots"]
        metrics = job["metrics"]

        tmins = [float(PARTS[p]["t_start_s"]) for p in parts if p in PARTS]
        tmaxs = [float(PARTS[p]["t_end_s"])   for p in parts if p in PARTS]
        if not tmins or not tmaxs:
            print(f"[WARN] Pomijam – brak PARTS dla {parts}")
            continue
        g_t_min, g_t_max = min(tmins), max(tmaxs)

        dt_tags = []
        for p in parts:
            if p in PARTS:
                dt_tags.append(str(PARTS[p]["dt_sim_s"]))
        dt_tag = None
        if dt_tags:
            # jeśli wszystkie dt identyczne → pokaż „1e-4s” itd.
            dts = set(dt_tags)
            if len(dts) == 1:
                dt_tag = f"{float(dt_tags[0]):.0e}s".replace("+0", "")
        tag_common = _outfile_tag("F?", parts, g_t_min, g_t_max, dt_tag)  # F? tylko do katalogu

        job_dir = plots_dir / f"{_parts_tag(parts)}_{_fmt_time_tag(g_t_min, g_t_max)}"
        job_dir.mkdir(parents=True, exist_ok=True)

        overlay_every = int(PLOT_DEFAULTS.get("overlay_every", 1))
        include_pressure = bool(PLOT_DEFAULTS.get("include_pressure", False))

        for fluid in fluids:
            overlays_seq_triplets: List[Tuple[float, pd.DataFrame, str]] = []
            summary_rows: List[Dict] = []

            for part in parts:
                if part not in index:
                    if args.verbose:
                        print(f"[INFO] brak {part}")
                    continue
                dt_sim = float(PARTS[part]["dt_sim_s"])
                t_start = float(PARTS[part]["t_start_s"])
                t_end   = float(PARTS[part]["t_end_s"])
                t_len   = t_end - t_start
                if t_len < 0: continue

                items = index[part].get(fluid, [])
                items = _filter_by_local_window(items, t_len)

                for (t_local, part_name, path) in items:
                    key = (part_name, fluid, path)
                    if key not in cache_bands:
                        band_tab, means = _build_bands_and_means(path, fluid)
                        cache_bands[key] = band_tab
                        cache_means[key] = means
                    else:
                        band_tab = cache_bands[key]
                        means    = cache_means[key]

                    t_global = t_start + t_local
                    overlays_seq_triplets.append((t_global, band_tab, f"t={t_global:.3f}s"))
                    row = {"Fluid": fluid, "Time[s]": t_global}; row.update(means); summary_rows.append(row)

            overlays_seq_triplets.sort(key=lambda tpl: tpl[0])
            overlays_seq = [(df, lbl) for (_tg, df, lbl) in overlays_seq_triplets]
            df_means = pd.DataFrame(summary_rows).sort_values(["Fluid","Time[s]"]).reset_index(drop=True) \
                       if summary_rows else pd.DataFrame()

            if not overlays_seq and df_means.empty:
                print(f"[INFO] Job {parts} | {fluid}: brak danych ({g_t_min}-{g_t_max}s).")
                continue

            tag = _outfile_tag(fluid, parts, g_t_min, g_t_max, dt_tag)

            if "overlay" in plots and overlays_seq:
                make_overlays_per_fluid(
                    overlays_seq=overlays_seq,
                    fluid_name=fluid,
                    out_dir=job_dir,
                    metrics=metrics,
                    include_pressure=include_pressure,
                    overlay_every=overlay_every,
                    outfile_tag=tag,
                )
            if "mean" in plots and not df_means.empty:
                plot_means_vs_time(
                    df=df_means, fluid_name=fluid, out_dir=job_dir,
                    metrics=metrics, include_pressure=include_pressure, outfile_tag=tag,
                )
            if not df_means.empty:
                csv_means = (out_dir / "csv" / f"summary_means__{tag.replace('__','_')}.csv")
                df_means.to_csv(csv_means, index=False)
                print(f"[OK] CSV → {csv_means.name}")

    # ====== PORÓWNANIA (COMPARE_JOBS) ======
    repo_root   = Path(__file__).resolve().parent
    compare_root = (Path(args.out_dir).resolve() / "Compare")
    compare_root.mkdir(parents=True, exist_ok=True) 

    for job in COMPARE_JOBS:
        job_name = job.get("name", "cmp")
        plots    = job.get("plots", ["overlay"])
        metrics  = job.get("metrics", ["h"])
        mean_opts = job.get("mean", {}) or {}
        overlay_opts = job.get("overlay", {}) or {}

        # steady per-job
        steady_cfg       = job.get("steady", {}) or {}
        steady_enabled   = bool(steady_cfg.get("enabled", False))
        steady_use_ovl   = bool(steady_cfg.get("use_in_overlay", True))
        steady_use_mean  = bool(steady_cfg.get("use_in_mean", True))
        steady_cases     = steady_cfg.get("cases", {})
        steady_weights   = steady_cfg.get("mean_weights", {})
        repo_root        = Path(__file__).resolve().parent
        steady_base      = (repo_root / steady_cfg.get("base_csv_dir", r"..\Steady_Repo\DataProcessed\csv")).resolve()

        # grupowanie serii per fluid
        series = job["series"]
        by_fluid: Dict[str, List[Dict]] = {"Fluid1": [], "Fluid2": []}
        for s in series:
            s = dict(s)  # kopia
            s["_overlay_opts_"] = overlay_opts
            by_fluid[s["fluid"]].append(s)

        job_dir = compare_root / _sanitize(job_name)
        job_dir.mkdir(parents=True, exist_ok=True)

        # ---- OVERLAY porównawcze ----
        if "overlay" in plots:
            for metric_key in metrics:
                for fluid, sers in by_fluid.items():
                    profiles: List[Dict] = []
                    for s in sers:
                        prf = build_overlay_profile_for_series(
                            series_cfg=s, PARTS=PARTS, index=index,
                            cache_bands=cache_bands,
                            build_one=_build_bands_and_means,
                            metric_key=metric_key,
                        )
                        if prf is not None:
                            profiles.append(prf)

                    # steady profile
                    if steady_enabled and steady_use_ovl:
                        for case in (steady_cases.get(fluid, []) or []):
                            csvp = (steady_base / case["file"]).resolve()
                            if csvp.exists():
                                try:
                                    x_mm, y = _load_steady_profile(csvp, fluid, metric_key)
                                    profiles.append({
                                        "label": f"{case['label']} SS",
                                        "fluid": fluid,
                                        "x_mm": x_mm, "y": y,
                                        "shade_lo": None, "shade_hi": None,
                                        "axis_name": "steady",
                                        "t0_s": None, "t1_s": None,
                                    })
                                except Exception as e:
                                    print(f"[WARN] STEADY overlay {csvp.name}: {e}")

                    if not profiles:
                        print(f"[INFO] Compare overlay: brak danych ({metric_key}, {fluid}).")
                        continue

                    plot_compare_overlay(
                        out_dir=job_dir, metric_key=metric_key, fluid_name=fluid,
                        series_profiles=profiles, job_name=job_name,
                    )

        # ---- MEAN porównawcze ----
        if "mean" in plots:
            time_mode  = mean_opts.get("time_mode", "global")
            ma_windows = [int(w) for w in (mean_opts.get("ma_windows", []) or []) if int(w) > 0]
            ma_center  = bool(mean_opts.get("ma_center", True))
            ma_edges   = mean_opts.get("ma_edges", "strict")
            show_raw   = bool(mean_opts.get("show_raw", True))
            min_periods = (lambda W: W if ma_edges == "strict" else 1)

            # „strict-crop”: przy „strict” przytnij wszystkie serie do wspólnego okna,
            # wyznaczonego po NAJWIĘKSZYM oknie MA (bo to daje największe ucięcie).
            use_strict_crop = (ma_edges == "strict")
            W_crop = max(ma_windows) if (use_strict_crop and ma_windows) else None

            for metric_key in metrics:
                # skala metryki (opcjonalnie z joba)
                scale = (job.get("scales", {}) or {}).get(metric_key, 1.0)

                for fluid, sers in by_fluid.items():
                    series_list: List[Dict] = []
                    # tymczasowo gromadzimy granice czasu do „strict crop”
                    t_first_valid, t_last_valid = [], []

                    for s in sers:
                        dfm = build_mean_timeseries_for_series(
                            series_cfg=s, PARTS=PARTS, index=index,
                            cache_means=cache_means, cache_bands=cache_bands,
                            build_one=_build_bands_and_means,
                        )
                        if dfm.empty:
                            continue

                        from constants import MEAN_COL_MAP
                        base_col = MEAN_COL_MAP[metric_key]
                        gdf = dfm[dfm["Fluid"] == fluid].sort_values("Time[s]").copy()
                        if base_col not in gdf.columns or gdf.empty:
                            continue

                        t_global = gdf["Time[s]"].values
                        t_aligned = t_global - float(s["t0_s"])
                        t_plot = t_global if time_mode == "global" else t_aligned

                        y_raw = gdf[base_col].values.astype(float)
                        if scale != 1.0:
                            y_raw = y_raw * scale

                        y_ma_dict = {}
                        for W in ma_windows:
                            s_ma = gdf[base_col].rolling(
                                window=W, center=ma_center, min_periods=min_periods(W)
                            ).mean().values.astype(float)
                            if scale != 1.0:
                                s_ma = s_ma * scale
                            y_ma_dict[W] = s_ma

                        # zapamiętaj granice dla „strict crop” z największego okna
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
                            "t_plot": t_plot,     # pomocniczo
                            "y_raw": y_raw,
                            "y_ma": y_ma_dict if y_ma_dict else None,
                        })

                    # „strict crop” – część wspólna
                    if use_strict_crop and W_crop is not None and t_first_valid and t_last_valid:
                        t_min_crop = max(t_first_valid)
                        t_max_crop = min(t_last_valid)
                        if t_min_crop < t_max_crop:
                            for ss in series_list:
                                t = ss["t_plot"]
                                mask = (t >= t_min_crop - 1e-12) & (t <= t_max_crop + 1e-12)
                                # przytnij osie czasu używane do rysowania
                                ss["t_global"]  = ss["t_global"][mask]
                                ss["t_aligned"] = ss["t_aligned"][mask]
                                ss["t_plot"]    = ss["t_plot"][mask]
                                # przytnij przebiegi
                                ss["y_raw"] = ss["y_raw"][mask]
                                if ss.get("y_ma"):
                                    for W in list(ss["y_ma"].keys()):
                                        ss["y_ma"][W] = ss["y_ma"][W][mask]

                    # steady ref-lines (skalowane tak samo)
                    ref_lines = []
                    if steady_enabled and steady_use_mean:
                        for case in (steady_cases.get(fluid, []) or []):
                            csvp = (steady_base / case["file"]).resolve()
                            if csvp.exists():
                                try:
                                    mvals = _steady_means(csvp, steady_weights)
                                    if metric_key in mvals:
                                        ref_lines.append((case["label"], float(mvals[metric_key]) * scale))
                                except Exception as e:
                                    print(f"[WARN] STEADY mean {csvp.name}: {e}")

                    if not series_list and not ref_lines:
                        print(f"[INFO] Compare mean: brak danych ({metric_key}, {fluid}).")
                        continue

                    plot_compare_mean(
                        out_dir=job_dir, metric_key=metric_key, fluid_name=fluid,
                        series_list=[{
                            "label": s["label"],
                            "t_global": s["t_global"],
                            "t_aligned": s["t_aligned"],
                            "y_raw": s["y_raw"],
                            "y_ma": s["y_ma"],
                        } for s in series_list],
                        job_name=job_name,
                        time_mode=time_mode, show_raw=show_raw, ref_lines=ref_lines,
                    )

if __name__ == "__main__":
    main()
