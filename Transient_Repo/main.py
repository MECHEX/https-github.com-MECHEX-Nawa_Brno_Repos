# Transient SRP → CSV + wykresy (parts → fluids → plots → metrics)
# - PARTS definiują tylko zakres czasu (t_start_s, t_end_s) i dt_symulacji (dt_sim_s).
# - Foldery partX są pod BASE_DATA_DIR; bez wpisywania pełnych ścieżek.
# - PLOT_JOBS: parts → fluids → plots (overlay/mean) → metrics (h/f).
# - Overlay: X od 0 (Fluid1: 0→L, Fluid2: 0→−L, oś Y po lewej), legenda: t=... s (czas globalny).
# - Mean: vs Time[s] (czas globalny).
# - Nazwy plików generują się automatycznie: <prefix>__F?__p...__tX-Ys.png

from __future__ import annotations
from pathlib import Path
import argparse
import re
from typing import Dict, List, Tuple
import pandas as pd
import tempfile, os

from srp_parser import parse_srp
from compute import (
    build_planes_df,
    build_bands_df,
    compute_band_table,
    compute_global_means,
)

from plotting import (
    set_plot_defaults,
    make_overlays_per_fluid,
    plot_means_vs_time,
    make_overlay3d_f_fluid1_interactive 
)

# =========================
# KONFIGURACJA (EDYTUJ TYLKO TĘ SEKCJĘ)
# =========================

# Bazowy katalog z partami (domyślnie wskazywany w --in-dir)
DEFAULT_BASE_DATA_DIR = r"C:\Users\kik\My Drive\Politechnika Krakowska\Researches\2025_NAWA_Brno\Nawa_Brno_Repos\Transient_Repo\FluentTransientData"

# PARTS: tylko zakres czasu GLOBALNEGO oraz krok czasu SYMULACJI w danym parcie
PARTS: Dict[str, Dict[str, float]] = {
    "part1": {"t_start_s": 0.0,  "t_end_s": 5.0,  "dt_sim_s": 0.0005},
    "part2": {"t_start_s": 5.0,  "t_end_s": 10.0, "dt_sim_s": 0.0005},
    "part3": {"t_start_s": 10.0, "t_end_s": 11.0, "dt_sim_s": 0.0001},
}

# Domyślne ustawienia rysowania
PLOT_DEFAULTS = {
    "dpi": 400,
    "marker_size": 0.5,     # mniejsze znaczniki
    "overlay_every": 1,   # co-ile serii rysować w overlay (1=każdą; 10=co 10-tą)
    "include_pressure": False,  # dorzucaj Δp_band (overlay) i Δp_sum (mean)
    "view3d_elev": 25,   # ← nowość
    "view3d_azim": -155, # ← nowość
}


# JOBS: parts → fluids → plots → metrics (h/f). Nazwy plików powstaną automatycznie.
PLOT_JOBS: List[Dict] = [
    {
        "parts":   ["part1", "part2"],
        "fluids":  ["Fluid1", "Fluid2"],
        "plots":   ["overlay", "mean"],
        "metrics": ["h", "f"],
    },
    {
        "parts":   ["part3"],
        "fluids":  ["Fluid1", "Fluid2"],
        "plots":   ["overlay", "mean"],
        "metrics": ["h", "f"],
    },
]

# Konfiguracja osi i kroku długości (m) dla płynów
FLUID_CFG: Dict[str, Dict[str, float | str]] = {
    "Fluid1": {"axis": "z", "min": -0.02959, "max":  0.02920, "step": 0.001959666667},
    "Fluid2": {"axis": "y", "min":  0.00950, "max": -0.00953, "step": 0.0009515},
}

# =========================
# STAŁE / POMOCNICZE
# =========================

STEP_RE = re.compile(r"_S(\d{4,})", re.IGNORECASE)

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
    # ujednolica f1_/f2_ → wspólne etykiety
    t = re.sub(r"\bf[12]_wall_band_", "wall_band_", text)
    t = re.sub(r"\bf[12]_(?:ziso_|yiso_)", "ziso_", t)
    t = re.sub(r"\bf[12]_env_", "env_", t)
    return t

def _fmt_time_tag(t0: float, t1: float) -> str:
    def _fmt(x: float) -> str:
        if abs(x - round(x)) < 1e-9:
            return f"{int(round(x))}"
        return f"{x:.3f}".rstrip("0").rstrip(".")
    return f"t{_fmt(t0)}-{_fmt(t1)}s"

def _parts_tag(parts: List[str]) -> str:
    nums = []
    for p in parts:
        m = re.search(r"part\s*([0-9]+)", p, re.IGNORECASE)
        nums.append(int(m.group(1)) if m else -1)
    nums = sorted([n for n in nums if n >= 0])
    if not nums:
        return "p"
    return "p" + "".join(str(n) for n in nums)

def _outfile_tag(fluid: str, parts: List[str], tmin: float, tmax: float) -> str:
    ftag = "_F1" if fluid == "Fluid1" else "_F2"
    ptag = _parts_tag(parts)
    ttag = _fmt_time_tag(tmin, tmax)
    return f"{ftag}__{ptag}__{ttag}"

# =========================
# GŁÓWNA LOGIKA
# =========================

def _collect_all_data(base_dir: Path) -> Dict[str, Dict[str, List[Tuple[float, str, Path]]]]:
    """
    Zbiera listę plików *.srp z istniejących folderów partX pod base_dir
    i buduje mapę: parts -> fluid -> lista (t_local[s], part_name, path)
    gdzie t_local = S * dt_sim_s (czas lokalny partu).
    """
    out: Dict[str, Dict[str, List[Tuple[float, str, Path]]]] = {}
    for part, meta in PARTS.items():
        dt = float(meta["dt_sim_s"])
        part_dir = base_dir / part
        if not part_dir.exists():
            continue
        for p in part_dir.glob("*.srp"):
            step = _extract_step(p.stem)
            if step < 0:
                continue
            t_local = step * dt
            fluid = _fluid_from_name(p.stem)
            out.setdefault(part, {}).setdefault(fluid, []).append((t_local, part, p))
    # sortuj po czasie lokalnym
    for part in out:
        for fluid in out[part]:
            out[part][fluid].sort(key=lambda tup: tup[0])
    return out

def _filter_by_local_window(items: List[Tuple[float, str, Path]], t_len: float) -> List[Tuple[float, str, Path]]:
    """Zostawia tylko rekordy, których czas lokalny wpada w [0, t_len]."""
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

    # parse_srp oczekuje ścieżki → użyj tymczasowego pliku
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

    # współrzędne środków pasm (kolumna: "z [m]" albo "y [m]")
    n_bands = len(band_tab)
    plane_coords = [a_min + sgn * i * step_abs_len for i in range(n_bands + 1)]
    centers = [(plane_coords[i] + plane_coords[i+1]) * 0.5 for i in range(n_bands)]
    col_axis = f"{axis_label} [m]"
    # usuń ewentualną kolumnę drugiej osi
    for old in ("z [m]", "y [m]"):
        if old in band_tab.columns and old != col_axis:
            band_tab.drop(columns=[old], inplace=True)
    band_tab[col_axis] = centers

    means = compute_global_means(band_tab)
    return band_tab, means

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in-dir",
        type=str,
        default=DEFAULT_BASE_DATA_DIR,
        help="Katalog bazowy z partami (zawiera podfoldery part1, part2, ...)",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default=r"C:\Users\kik\My Drive\Politechnika Krakowska\Researches\2025_NAWA_Brno\Nawa_Brno_Repos\Transient_Repo\TransientFigs",
        help="Katalog wyjściowy (zapis CSV/plots).",
    )
    ap.add_argument("--verbose", action="store_true", help="Dodatkowy log")
    args = ap.parse_args()

    base_dir = Path(args.in_dir).resolve()
    out_dir  = Path(args.out_dir).resolve()
    csv_dir  = out_dir / "csv"
    plots_dir = out_dir / "plots"
    csv_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Ustawienia rysowania
    set_plot_defaults(
        dpi=int(PLOT_DEFAULTS.get("dpi", 160)),
        marker_size=float(PLOT_DEFAULTS.get("marker_size", 2.0)),  # float!
        line_width=float(PLOT_DEFAULTS.get("line_width", 1.0)),
        view3d_elev=int(PLOT_DEFAULTS.get("view3d_elev", 25)),
        view3d_azim=int(PLOT_DEFAULTS.get("view3d_azim", -135)),
    )


    overlay_every = int(PLOT_DEFAULTS.get("overlay_every", 1))
    include_pressure = bool(PLOT_DEFAULTS.get("include_pressure", True))

    # 1) Zbierz listę plików per part / fluid z CZASEM LOKALNYM
    index = _collect_all_data(base_dir)

    # 2) Cache: aby nie parsować plików wielokrotnie między jobami
    cache_bands: Dict[Tuple[str, str, Path], pd.DataFrame] = {}
    cache_means: Dict[Tuple[str, str, Path], Dict[str, float]] = {}

    # 3) Przetwarzanie według PLOT_JOBS
    for job in PLOT_JOBS:
        parts: List[str]  = job["parts"]
        fluids: List[str] = job["fluids"]
        plots: List[str]  = job["plots"]
        metrics: List[str]= job["metrics"]

        # Ustal globalny zakres czasu (na podstawie PARTS i listy parts w jobie)
        tmins = [float(PARTS[p]["t_start_s"]) for p in parts if p in PARTS]
        tmaxs = [float(PARTS[p]["t_end_s"])   for p in parts if p in PARTS]
        if not tmins or not tmaxs:
            print(f"[WARN] Job pomija, brak zdefiniowanych PARTS dla: {parts}")
            continue
        global_t_min, global_t_max = min(tmins), max(tmaxs)

        # NOWE: katalog na wykresy dla tego jobu, np. plots/p12_t0-10s
        job_folder_name = f"{_parts_tag(parts)}_{_fmt_time_tag(global_t_min, global_t_max)}"
        job_plots_dir = plots_dir / job_folder_name
        job_plots_dir.mkdir(parents=True, exist_ok=True)

        for fluid in fluids:
            # Kolekcja (t_global, df_band, label) do overlayów
            overlays_seq_triplets: List[Tuple[float, pd.DataFrame, str]] = []
            # Wiersze mean: Time[s] (globalny), mean_* ...
            summary_rows: List[Dict] = []

            for part in parts:
                if part not in index:
                    if args.verbose:
                        print(f"[INFO] Brak folderu/indeksu dla {part} → pomijam.")
                    continue

                dt_sim = float(PARTS[part]["dt_sim_s"])
                t_start = float(PARTS[part]["t_start_s"])
                t_end   = float(PARTS[part]["t_end_s"])
                t_len   = t_end - t_start
                if t_len < 0:
                    if args.verbose:
                        print(f"[WARN] Ujemny zakres czasu w {part}: {t_start}–{t_end} → pomijam.")
                    continue

                items = index[part].get(fluid, [])
                # filtr po CZASIE LOKALNYM: [0, t_len]
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

                    # czas GLOBALNY = t_start(part) + t_local
                    t_global = t_start + t_local

                    label = f"t={t_global:.3f}s"
                    overlays_seq_triplets.append((t_global, band_tab, label))

                    row = {"Fluid": fluid, "Time[s]": t_global}
                    row.update(means)
                    summary_rows.append(row)

            # sortuj overlayy po czasie GLOBALNYM
            overlays_seq_triplets.sort(key=lambda tpl: tpl[0])
            overlays_seq = [(df, lbl) for (_tg, df, lbl) in overlays_seq_triplets]

            # bezpieczna konstrukcja df_means (może być pusto)
            if summary_rows:
                df_means = (
                    pd.DataFrame(summary_rows)
                    .sort_values(["Fluid", "Time[s]"])
                    .reset_index(drop=True)
                )
            else:
                df_means = pd.DataFrame()

            if not overlays_seq and df_means.empty:
                print(f"[INFO] Job {parts} | {fluid}: brak plików w oknie globalnym "
                      f"({global_t_min}–{global_t_max} s) → pomijam.")
                continue

            # tag do nazwy pliku
            tag = _outfile_tag(fluid, parts, global_t_min, global_t_max)

            # Rysunki overlay wg metrics (+ Δp jeśli włączone)
            if "overlay" in plots and overlays_seq:
                make_overlays_per_fluid(
                    overlays_seq=overlays_seq,
                    fluid_name=fluid,
                    out_dir=job_plots_dir,
                    metrics=metrics,
                    include_pressure=include_pressure,
                    overlay_every=overlay_every,
                    outfile_tag=tag,
                )
            
                # 3D interaktywny (tylko gdy Fluid1 i 'f' w metrics)
                if fluid == "Fluid1" and ("f" in metrics) and overlays_seq:
                    make_overlay3d_f_fluid1_interactive(
                        overlays_seq=overlays_seq,
                        out_dir=job_plots_dir,
                        outfile_tag=tag,
                        y_fixed_min_mm=0.0,   # zakres osi Y w mm
                        y_fixed_max_mm=60.0,
                        stride_time=1,        # ewentualne przerzedzanie (czas)
                        stride_x=1            # ewentualne przerzedzanie (po profilu)
                    )



            # Rysunki mean vs Time wg metrics (+ Δp_sum jeśli włączone)
            if "mean" in plots and not df_means.empty:
                plot_means_vs_time(
                    df=df_means,
                    fluid_name=fluid,
                    out_dir=job_plots_dir,
                    metrics=metrics,
                    include_pressure=include_pressure,
                    outfile_tag=tag,
                )

            # Zapis wspólnego CSV mean dla joba/fluidu (opcjonalnie)
            if not df_means.empty:
                csv_means = csv_dir / f"summary_means__{tag.replace('__','_')}.csv"
                df_means.to_csv(csv_means, index=False)
                if args.verbose:
                    print(f"[OK] Summary CSV → {csv_means}")

if __name__ == "__main__":
    main()
