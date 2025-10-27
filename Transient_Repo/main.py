# Transient SRP → CSV + wykresy
# - Jedna siatka (M101), wiele kroków czasowych (..._Sxxxxx.srp)
# - Overlaye po osi (z dla Fluid1, y dla Fluid2) oraz wykresy średnich vs Time[s]
# - Obsługa folderów FluentTransientData\part1 (0–5 s) i FluentTransientData\part2 (5–10 s)
# - Globalny Step dla overlayów: Step_abs = Step_local + offset_steps(na podstawie Time offsetu)

from __future__ import annotations
from pathlib import Path
import argparse
import re
from typing import Dict, Tuple
import pandas as pd

from srp_parser import parse_srp           # zwraca SRPData
from compute import (
    build_planes_df,                       # (SRPData, dz) -> df płaszczyzn
    build_bands_df,                        # (SRPData) -> df pasm surowych
    compute_band_table,                    # (planes_df, bands_df, dz, axis_label) -> df pasm z kolumnami fizycznymi
    compute_global_means,                  # (band_table) -> dict mean_*
)

# --- konfiguracja osi/kroków (w [m]) dla obu fluidów ---
# Fluid1: oś z (rosnąco), Fluid2: oś y (malejąco)
FLUID_CFG: Dict[str, Dict[str, float | str]] = {
    "Fluid1": {"axis": "z", "min": -0.02959, "max":  0.02920, "step": 0.001959666667},
    "Fluid2": {"axis": "y", "min":  0.00950, "max": -0.00953, "step": 0.0009515},
}

# --- stały krok czasowy do osi Time[s] ---
DT_S = 0.0005  # [s] — 10k kroków → 5 s; part2 ma offset 5.0 s, więc razem 0–10 s

# --- stałe / pomocnicze ---
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
    # fallback
    return "Fluid2" if "_f2" in s else "Fluid1"

def _normalize_prefixes(text: str) -> str:
    """
    Ujednolica nazwy sekcji w SRP, by parser działał identycznie dla F1/F2:
    - f1_/f2_ w wall_band_, ziso_/yiso_, env_ → bez prefiksu num. fluida
    - yiso_ → ziso_ (oś rozstrzygamy później przez FLUID_CFG)
    """
    t = text
    t = re.sub(r"\bf[12]_wall_band_", "wall_band_", t)
    t = re.sub(r"\bf[12]_(?:ziso_|yiso_)", "ziso_", t)
    t = re.sub(r"\bf[12]_env_", "env_", t)
    return t

def _infer_time_offset_seconds(path: Path) -> float:
    """Jeśli ścieżka zawiera 'part2' lub 'part 2' → offset 5.0 s, inaczej 0.0 s."""
    parts_lower = {p.lower() for p in path.parts}
    if ("part2" in parts_lower) or ("part 2" in parts_lower):
        return 5.0
    return 0.0

def process_directory(in_dir: Path, out_dir: Path, dt: float | None, verbose: bool = False) -> None:
    in_dir = in_dir.resolve()
    out_dir = out_dir.resolve()
    csv_dir   = out_dir / "csv"
    plots_dir = out_dir / "plots"
    csv_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # rekurencyjnie: zbierz *.srp z podfolderów (part1, part2, ...)
    srp_files = sorted([p for p in in_dir.rglob("*.srp") if p.is_file()], key=lambda p: p.as_posix())
    if not srp_files:
        print(f"[WARN] Brak plików *.srp w (rekurencyjnie): {in_dir}")
        return

    # mapy do overlayów – kluczem będzie GLOBALNY step (z offsetem czasu), żeby uniknąć kolizji
    bands_by_fluid: Dict[str, Dict[int, pd.DataFrame]] = {"Fluid1": {}, "Fluid2": {}}
    summary_rows = []

    for p in srp_files:
        stem   = p.stem
        step   = _extract_step(stem)   # lokalny step z nazwy pliku (może się powtarzać między part1/part2)
        fluid  = _fluid_from_name(stem)
        offset_s = _infer_time_offset_seconds(p)  # 0.0 dla part1, 5.0 dla part2

        cfg = FLUID_CFG.get(fluid)
        if cfg is None:
            raise RuntimeError(f"Brak konfiguracji osi dla: {fluid}")

        axis_label: str = str(cfg["axis"])        # 'z' lub 'y'
        step_abs_len: float = float(cfg["step"])  # dodatni krok długości [m]
        start_min: float = float(cfg["min"])
        end_max: float   = float(cfg["max"])
        sgn = 1.0 if end_max >= start_min else -1.0

        raw = p.read_text(encoding="utf-8", errors="ignore")
        norm = _normalize_prefixes(raw)
        # zapisz znormalizowany SRP obok CSV (debug)
        norm_path = csv_dir / f"{stem}__norm.srp"
        norm_path.write_text(norm, encoding="utf-8", errors="ignore")

        # parse → SRPData
        data = parse_srp(norm_path, verbose=False)
        if verbose:
            print(f"parsed: {p.name} | fluid={fluid} | local_step={step} | offset_s={offset_s}")

        # zbuduj tabele + pełną tabelę pasm na właściwym dz
        planes_df  = build_planes_df(data, dz=step_abs_len)
        bands_df   = build_bands_df(data)
        band_table = compute_band_table(planes_df, bands_df, dz=step_abs_len, axis_label=axis_label)

        # współrzędne pasm: środki wg min/max/step i kierunku
        n_bands = len(band_table)
        plane_coords = [start_min + sgn * i * step_abs_len for i in range(n_bands + 1)]
        centers = [(plane_coords[i] + plane_coords[i+1]) * 0.5 for i in range(n_bands)]
        col_axis = f"{axis_label} [m]"
        # usuń ewentualną kolumnę drugiej osi
        for old_col in ("z [m]", "y [m]"):
            if old_col in band_table.columns and old_col != col_axis:
                band_table.drop(columns=[old_col], inplace=True)
        band_table[col_axis] = centers

        # CSV per plik
        band_table.to_csv(csv_dir / f"{stem}__bands.csv", index=False)

        # czas i globalny step (unikamy kolizji między part1/part2)
        if dt is not None and step >= 0:
            time_s = offset_s + step * dt
            offset_steps = int(round(offset_s / dt))
            step_global = step + offset_steps
        else:
            time_s = None
            step_global = step

        # do overlayów i średnich
        bands_by_fluid[fluid][step_global] = band_table
        means = compute_global_means(band_table)
        row = {"Step": step_global, "Fluid": fluid}
        if time_s is not None:
            row["Time[s]"] = time_s
        row.update(means)
        summary_rows.append(row)

        if verbose:
            print(f"[OK] {p.name} → bands={len(band_table)} | Fluid={fluid} | Step_global={step_global} | Time={time_s}")

    # --- wykresy overlay + summary ---
    from plotting import make_overlays_per_fluid, plot_summary_vs_step  # lokalny import
    for fluid, mapping in bands_by_fluid.items():
        if mapping:
            make_overlays_per_fluid(mapping, fluid, plots_dir)

    if summary_rows:
        df = pd.DataFrame(summary_rows).sort_values(["Fluid", "Time[s]"] if "Time[s]" in summary_rows[0] else ["Fluid","Step"]).reset_index(drop=True)
        df.to_csv(csv_dir / "summary_means_vs_step.csv", index=False)
        # mean-y rysujemy vs Time[s] (plotting.py tak właśnie robi)
        plot_summary_vs_step(df, plots_dir, dt=dt)
        print(f"[OK] Summary CSV → {csv_dir / 'summary_means_vs_step.csv'}")
    else:
        print("[INFO] Brak danych do tabeli średnich.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in-dir",
        type=str,
        default=r"C:\Users\kik\My Drive\Politechnika Krakowska\Researches\2025_NAWA_Brno\Nawa_Brno_Repos\Transient_Repo\FluentTransientData",
        help="Katalog z plikami *.srp (przeszukiwany rekurencyjnie: part1, part2, ...)",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default=r"C:\Users\kik\My Drive\Politechnika Krakowska\Researches\2025_NAWA_Brno\Nawa_Brno_Repos\Transient_Repo\TransientFigs",
        help="Katalog wyjściowy",
    )
    ap.add_argument("--verbose", action="store_true", help="Dodatkowy log")
    args = ap.parse_args()

    # stały krok czasowy do wykresów vs Time[s]; ustaw None, by pominąć oś czasu
    dt_for_plots = DT_S
    process_directory(Path(args.in_dir), Path(args.out_dir), dt=dt_for_plots, verbose=args.verbose)

if __name__ == "__main__":
    main()
