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
    make_overlay_mean_std_per_fluid,
    plot_means_vs_time,
)

from compare_runner import run_compare_jobs

from config import (
    PARTS, FLUID_CFG, PLOT_DEFAULTS, PLOT_JOBS, COMPARE_JOBS,
    SRP_TIME_SOURCE_PRIORITY,
    SRP_T_INTERPRETATION,
)

from constants import COL_Z_M, COL_Y_M


STEP_RE = re.compile(r"_S(\d{4,})", re.IGNORECASE)
TIME_RE = re.compile(r"_T([0-9]*\.?[0-9]+(?:[eE][+-]?\d+)?)", re.IGNORECASE)


def _extract_step(stem: str) -> int:
    m = STEP_RE.search(stem)
    return int(m.group(1)) if m else -1


def _extract_time_s(stem: str) -> Optional[float]:
    m = TIME_RE.search(stem)
    if not m:
        return None
    token = m.group(1).replace(",", ".")
    try:
        return float(token)
    except ValueError:
        return None


def _extract_t_local(part: str, stem: str, dt_sim_s: float) -> Optional[float]:
    t_start = float(PARTS[part]["t_start_s"])
    t_end   = float(PARTS[part]["t_end_s"])
    t_len   = t_end - t_start

    def interpret_T(T: float) -> float:
        mode = (SRP_T_INTERPRETATION or "local").lower()
        if mode == "local":
            return T
        if mode == "global":
            return T - t_start
        if mode == "auto":
            if (-1e-12) <= T <= (t_len + 1e-12):
                return T
            return T - t_start
        return T

    for src in (SRP_TIME_SOURCE_PRIORITY or ("T", "S")):
        s = str(src).upper()
        if s == "T":
            T = _extract_time_s(stem)
            if T is not None:
                return interpret_T(T)
        elif s == "S":
            step = _extract_step(stem)
            if step >= 0:
                return step * dt_sim_s
    return None


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
        if abs(x - round(x)) < 1e-9:
            return f"{int(round(x))}"
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


def _collect_all_data(base_dir: Path) -> Dict[str, Dict[str, List[Tuple[float, str, Path]]]]:
    out: Dict[str, Dict[str, List[Tuple[float, str, Path]]]] = {}
    for part, meta in PARTS.items():
        dt = float(meta["dt_sim_s"])
        pdir = base_dir / part
        if not pdir.exists():
            continue
        for p in pdir.glob("*.srp"):
            t_local = _extract_t_local(part=part, stem=p.stem, dt_sim_s=dt)
            if t_local is None:
                continue
            fluid = _fluid_from_name(p.stem)
            out.setdefault(part, {}).setdefault(fluid, []).append((t_local, part, p))

    for part in out:
        for fluid in out[part]:
            out[part][fluid].sort(key=lambda tup: tup[0])
    return out


def _filter_by_local_window(items: List[Tuple[float, str, Path]], t_len: float) -> List[Tuple[float, str, Path]]:
    return [tup for tup in items if (-1e-12) <= tup[0] <= (t_len + 1e-12)]


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
        try:
            os.remove(tmp_path)
        except OSError:
            pass

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=str,
        default=str((Path(__file__).resolve().parent / "FluentTransientData")),
        help="Katalog bazowy z partami (part1, part2, ...)")
    ap.add_argument("--out-dir", type=str,
        default=str((Path(__file__).resolve().parent / "TransientFigs")),
        help="Katalog wyjściowy (csv/plots/Compare).")
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

    index = _collect_all_data(base_dir)
    cache_bands: Dict[Tuple[str, str, Path], pd.DataFrame] = {}
    cache_means: Dict[Tuple[str, str, Path], Dict[str, float]] = {}

    # ===== Standard PLOT_JOBS =====
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

        dt_vals = [float(PARTS[p]["dt_sim_s"]) for p in parts if p in PARTS]
        dt_tag = None
        if dt_vals and len(set(dt_vals)) == 1:
            dt_tag = f"{dt_vals[0]:.0e}s".replace("+0", "")

        job_dir = plots_dir / f"{_parts_tag(parts)}_{_fmt_time_tag(g_t_min, g_t_max)}"
        job_dir.mkdir(parents=True, exist_ok=True)

        overlay_every = int(PLOT_DEFAULTS.get("overlay_every", 1))
        include_pressure = bool(PLOT_DEFAULTS.get("include_pressure", False))
        overlay_mode = str(job.get("overlay_mode", "all")).lower()

        for fluid in fluids:
            overlays_seq_triplets: List[Tuple[float, pd.DataFrame, str]] = []
            summary_rows: List[Dict] = []

            for part in parts:
                if part not in index:
                    if args.verbose:
                        print(f"[INFO] brak {part}")
                    continue

                t_start = float(PARTS[part]["t_start_s"])
                t_end   = float(PARTS[part]["t_end_s"])
                t_len   = t_end - t_start
                if t_len < 0:
                    continue

                items = _filter_by_local_window(index[part].get(fluid, []), t_len)

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
                    row = {"Fluid": fluid, "Time[s]": t_global}
                    row.update(means)
                    summary_rows.append(row)

            overlays_seq_triplets.sort(key=lambda tpl: tpl[0])
            overlays_seq = [(df, lbl) for (_tg, df, lbl) in overlays_seq_triplets]
            df_means = (
                pd.DataFrame(summary_rows).sort_values(["Fluid", "Time[s]"]).reset_index(drop=True)
                if summary_rows else pd.DataFrame()
            )

            if not overlays_seq and df_means.empty:
                print(f"[INFO] Job {parts} | {fluid}: brak danych ({g_t_min}-{g_t_max}s).")
                continue

            tag = _outfile_tag(fluid, parts, g_t_min, g_t_max, dt_tag)

            if "overlay" in plots and overlays_seq:
                if overlay_mode == "mean_std":
                    make_overlay_mean_std_per_fluid(
                        overlays_seq=overlays_seq,
                        fluid_name=fluid,
                        out_dir=job_dir,
                        metrics=metrics,
                        include_pressure=include_pressure,
                        overlay_every=overlay_every,
                        outfile_tag=tag,
                    )
                else:
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
                    df=df_means,
                    fluid_name=fluid,
                    out_dir=job_dir,
                    metrics=metrics,
                    include_pressure=include_pressure,
                    outfile_tag=tag,
                )

            if not df_means.empty:
                csv_means = csv_dir / f"summary_means_{tag}.csv"
                df_means.to_csv(csv_means, index=False)
                print(f"[OK] CSV → {csv_means.name}")

    # ===== Compare jobs =====
    compare_root = out_dir / "Compare"
    run_compare_jobs(
        compare_jobs=COMPARE_JOBS,
        compare_root_dir=compare_root,
        PARTS=PARTS,
        index=index,
        cache_bands=cache_bands,
        cache_means=cache_means,
        build_one=_build_bands_and_means,
        repo_root=Path(__file__).resolve().parent,
    )


if __name__ == "__main__":
    main()
