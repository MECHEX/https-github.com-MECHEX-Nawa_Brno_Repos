
# -*- coding: utf-8 -*-
"""
main.py — Transient mode: Fluid1 + Fluid2 → per-file CSV + overlaye + wykresy ŚREDNIE vs KROK CZASOWY

Założenia:
- Jedna siatka (mesh), wiele plików SRP z kolejnych kroków czasowych (np. ..._S01000.srp, ..._S01050.srp, ...).
- Rysujemy te SAME wykresy, co wcześniej (overlaye po z dla Fluid1/Fluid2 oraz wykres "średnie parametry"),
  ale zamiast "vs Cells (liczba elementów/mesh)" pokazujemy teraz "vs Step" (indeks kroku) i opcjonalnie "vs Time[s]".

Wejście:
- katalog z plikami *.srp (można wskazać --in-dir; domyślnie bieżący katalog).
- wzorce nazw plików: ...Fluid1_..._Sxxxxx.srp i ...Fluid2_..._Sxxxxx.srp (prefiksy f1_/f2_ w sekcjach są normalizowane).

Wyjście:
- CSV z pasmami dla każdego pliku /fluida
- CSV z tabelą średnich per [Step, Time[s]]
- Wykresy overlay (po z) z legendą = Step
- Wykresy średnich parametrów vs Step (i, jeśli podano --dt, równoległy wykres vs Time[s]).
"""

from __future__ import annotations
from pathlib import Path
import argparse
import re
import sys
import numpy as np
import pandas as pd

from srp_parser import parse_srp
from compute import build_bands_df, build_planes_df, compute_band_table, compute_global_means
from plotting import (
    make_overlays_per_fluid,
    plot_summary_vs_step,
)


from srp_types import SRPData

def _to_srpdata(parsed_tuple) -> SRPData:
    """Adapter: parser zwraca tuplę — opakuj w SRPData NamedTuple."""
    (planes_Tmass, planes_P, planes_A, planes_Aenv, planes_mdot,
     planes_rho, planes_mu, planes_k,
     bands_Awet, bands_Q, bands_Tw, env_single_const) = parsed_tuple

    A_env_single = env_single_const  # prefer single-const jeśli jest
    return SRPData(
        planes_Tmass=planes_Tmass,
        planes_P=planes_P,
        planes_A=planes_A,
        planes_mdot=planes_mdot,
        planes_rho=planes_rho,
        planes_mu=planes_mu,
        planes_k=planes_k,
        A_env_single=A_env_single,
        bands_Awet=bands_Awet,
        bands_Q=bands_Q,
        bands_Tw=bands_Tw,
    )

# --- pomocnicze ---
STEP_RE = re.compile(r'_S(\d{4,})', re.IGNORECASE)

def _extract_step(stem: str) -> int:
    m = STEP_RE.search(stem)
    if not m:
        return -1
    return int(m.group(1))

def _fluid_from_name(stem: str) -> str:
    s = stem.lower()
    if "fluid1" in s or "_f1_" in s or s.endswith("_f1") or "f1_" in s:
        return "Fluid1"
    if "fluid2" in s or "_f2_" in s or s.endswith("_f2") or "f2_" in s:
        return "Fluid2"
    # fallback: zgadnij z pierwszego wystąpienia
    return "Fluid2" if "_f2" in s else "Fluid1"

def _normalize_prefixes(text: str) -> str:
    """Ujednolica nazwy sekcji w SRP:
       - f1_/f2_ → puste (usunięcie numeru fluida)
       - yiso/ziso → ziso (używamy ziso_*)
       - f2_env_* → env_*, f1_env_* → env_*
    """
    t = text
    # usuwamy prefiksy f1_/f2_ z wall_band_, ziso_, yiso_, env_
    t = re.sub(r'\bf[12]_wall_band_', 'wall_band_', t)
    t = re.sub(r'\bf[12]_(?:ziso_|yiso_)', 'ziso_', t)
    t = re.sub(r'\bf[12]_env_', 'env_', t)
    # czasem w starych raportach pojawiały się "f2_ziso_" etc. — już pokryte
    return t

def process_directory(in_dir: Path, out_dir: Path, dt: float | None) -> None:
    in_dir = in_dir.resolve()
    out_dir = out_dir.resolve()
    plots_dir = out_dir / "plots"
    csv_dir   = out_dir / "csv"
    plots_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    # Zbierz pliki *.srp
    srp_files = sorted([p for p in in_dir.glob("*.srp") if p.is_file()], key=lambda p: p.name)
    if not srp_files:
        print(f"[WARN] Brak plików *.srp w: {in_dir}")
        return

    bands_by_fluid: dict[str, dict[int, pd.DataFrame]] = {"Fluid1": {}, "Fluid2": {}}
    summary_rows: list[dict] = []

    for p in srp_files:
        stem = p.stem
        step = _extract_step(stem)
        fluid = _fluid_from_name(stem)

        raw = p.read_text(encoding="utf-8", errors="ignore")
        norm = _normalize_prefixes(raw)

        # Zapisz tymczasowo znormalizowany plik (dla logów/debugu)
        norm_path = csv_dir / f"{stem}__norm.srp"
        norm_path.write_text(norm, encoding="utf-8", errors="ignore")

        # Parsuj → dataframe pasm
        parsed = parse_srp(norm_path, verbose=False)
        data = _to_srpdata(parsed)
        planes_df = build_planes_df(data, dz=1.0)
        bands_df = build_bands_df(data)
        band_table = compute_band_table(planes_df, bands_df, dz=1.0, axis_label='z')

        # Zapisz pasma do CSV per plik
        df_path = csv_dir / f"{stem}__bands.csv"
        band_table.to_csv(df_path, index=False)

        # Zapisz do mapy do overlayów
        bands_by_fluid[fluid][step] = band_table

        # policz średnie globalne, dodaj do tabeli podsumowania
        means = compute_global_means(band_table)
        row = {"Step": step}
        if dt is not None and step >= 0:
            row["Time[s]"] = step * dt
        row.update(means)
        summary_rows.append(row)

        print(f"[OK] {p.name} → bands: {len(bands_df)} | fluid={fluid} | step={step}")

    # --- overlaye po z ---
    for fluid, mapping in bands_by_fluid.items():
        if not mapping:
            continue
        make_overlays_per_fluid(mapping, fluid, plots_dir)

    # --- tabela średnich i wykresy vs Step/Time ---
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows).sort_values("Step").reset_index(drop=True)
        summary_csv = csv_dir / "summary_means_vs_step.csv"
        summary_df.to_csv(summary_csv, index=False)
        print(f"[OK] Summary CSV → {summary_csv}")

        plot_summary_vs_step(summary_df, plots_dir, dt=dt)
    else:
        print("[INFO] Brak danych do tabeli średnich.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=str, default=".", help="Katalog z plikami *.srp")
    ap.add_argument("--out-dir", type=str, default="./out_transient", help="Katalog wyjściowy")
    ap.add_argument("--dt", type=float, default=None, help="Δt między krokami [s] (opcjonalnie, do osi czasu)")
    args = ap.parse_args()

    process_directory(Path(args.in_dir), Path(args.out_dir), dt=args.dt)

if __name__ == "__main__":
    main()
