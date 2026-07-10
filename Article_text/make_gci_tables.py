# -*- coding: utf-8 -*-
"""
make_gci_tables.py — generuje table-ready CSV do sekcji Grid Independence artykulu.

Wyjscie: Article_text/data/gci_<geometry>.csv  (jeden plik na geometrie)

Tabela w stylu GCI: siatki Coarse / Medium / Fine z opisem, parametry
    dT, h, dp, f_Fanning  (dla Fluid1 i Fluid2)
oraz na koncu blad wzgledny liczony WZGLEDEM wartosci MEDIUM:
    eps(Coarse vs Medium) i eps(Fine vs Medium).
Dodatkowo, dla wybranej siatki MEDIUM, wiersz z wynikiem TRANSIENT i jego
odchyleniem od steady (dowod, ze analiza steady jest potwierdzona w transiencie).

Dane reuzywane z Steady_Repo/mesh_convergence_plots.py:
    GRAD, UNI  -> h_F*, f_F* (channel-mean)
    DTD_P      -> dT, dp inlet-outlet (z SRP)
    TRANSIENT_STAR_VALUES -> tail-mean transient (h,f,dp,dT)

UWAGA: triplety siatek sa konfigurowalne ponizej (TRIPLETS). Dobrane z ISTNIEJACYCH
danych. Docelowe Medium 8M (UNI) / 6M (GRAD) wymagaja nowej kampanii symulacji
(patrz ARTICLE_PREP_PLAN.md).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --- reuse istniejacej maszynerii ze Steady_Repo ---
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "Steady_Repo"))
import mesh_convergence_plots as mcp  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PARAMS = ["dT", "h", "dp", "f"]          # kolejnosc wg zyczenia autora
FLUIDS = [("Fluid1", "F1"), ("Fluid2", "F2")]

# ---------------------------------------------------------------------------
# Konfiguracja tripletow (z istniejacych danych; edytowalne)
# role -> (mesh_id, opis)
# transient_case: klucz w mcp.TRANSIENT_STAR_VALUES dla siatki Medium (albo None)
# ---------------------------------------------------------------------------
TRIPLETS = {
    "uni10": {
        "dict": mcp.UNI,
        "meshes": [
            ("Coarse", "GUNI_005", "surf 0.62 mm"),
            ("Medium", "GUNI_007", "surf 0.32 mm (recommended)"),
            ("Fine",   "GUNI_008", "surf 0.32 mm (finest)"),
        ],
        "transient_case": "uni10_007",   # transient dla Medium (6.25M) — istnieje
        "note": "Clean pair 007/008 (identical surface mesh). Medium~6.25M "
                "(closest existing to 8M target).",
    },
    "grad": {
        "dict": mcp.GRAD,
        "meshes": [
            ("Coarse", "M002_v2", "Laminar, 906K"),
            ("Medium", "M002_5",  "Laminar, 3.0M"),
            ("Fine",   "M003_v2", "Laminar, 8.4M"),
        ],
        "transient_case": None,          # brak transientu przy 3-8M (tylko 881K)
        "note": "PROVISIONAL: existing GRAD meshes do NOT form a converged series "
                "(h_F1 still trends 57.0->55.8->53.9). Target Medium~6M needs a new "
                "clean laminar refinement + transient. Transient available only at 881K.",
    },
}


def steady_value(geo_dict, mesh_id, param, fluid_long, fluid_tag):
    """Wartosc steady dla (mesh, param, fluid)."""
    if param == "h":
        return float(geo_dict[mesh_id][f"h_{fluid_tag}"])
    if param == "f":
        return float(geo_dict[mesh_id][f"f_{fluid_tag}"])
    if param in ("dT", "dp"):
        return float(mcp.DTD_P[mesh_id][fluid_long][param])
    raise KeyError(param)


def transient_value(case, param, fluid_tag):
    v = mcp.TRANSIENT_STAR_VALUES.get(case, {})
    return float(v.get(f"{param}_{fluid_tag}", np.nan))


def _col(param, fluid_tag):
    unit = {"dT": "[K]", "h": "[W/m2K]", "dp": "[Pa]", "f": "[-]"}[param]
    return f"{param}_{fluid_tag} {unit}"


def build_table(geo_key: str) -> pd.DataFrame:
    cfg = TRIPLETS[geo_key]
    gdict = cfg["dict"]
    rows = []

    # --- wiersze steady: Coarse / Medium / Fine ---
    steady_vals = {}  # role -> {colname: value}
    for role, mesh_id, desc in cfg["meshes"]:
        row = {"role": role, "mesh_id": mesh_id, "N_cells": int(gdict[mesh_id]["N"]),
               "solver": "steady", "description": desc}
        vals = {}
        for param in PARAMS:
            for fl_long, fl_tag in FLUIDS:
                col = _col(param, fl_tag)
                v = steady_value(gdict, mesh_id, param, fl_long, fl_tag)
                row[col] = v
                vals[col] = v
        steady_vals[role] = vals
        rows.append(row)

    # --- wiersz transient dla Medium (jesli sa dane) ---
    medium_id = dict((r, m) for r, m, _ in cfg["meshes"])["Medium"]
    trans_vals = None
    if cfg["transient_case"]:
        row = {"role": "Medium", "mesh_id": medium_id, "N_cells": int(gdict[medium_id]["N"]),
               "solver": "transient", "description": f"tail-mean, case {cfg['transient_case']}"}
        trans_vals = {}
        for param in PARAMS:
            for fl_long, fl_tag in FLUIDS:
                col = _col(param, fl_tag)
                v = transient_value(cfg["transient_case"], param, fl_tag)
                row[col] = v
                trans_vals[col] = v
        rows.append(row)

    # --- wiersze bledu wzglednego WZGLEDEM MEDIUM (na wartosciach steady) ---
    med = steady_vals["Medium"]
    for role in ("Coarse", "Fine"):
        row = {"role": f"eps({role} vs Medium)", "mesh_id": "", "N_cells": "",
               "solver": "rel_err_%", "description": "relative to Medium (steady)"}
        for col, mv in med.items():
            v = steady_vals[role][col]
            row[col] = (v - mv) / mv * 100.0 if mv not in (0, None) and np.isfinite(mv) else np.nan
        rows.append(row)

    # --- odchylenie transient vs steady przy Medium (dowod spojnosci) ---
    if trans_vals is not None:
        row = {"role": "eps(Transient vs Medium)", "mesh_id": medium_id, "N_cells": "",
               "solver": "rel_err_%", "description": "transient tail-mean vs steady Medium"}
        for col, mv in med.items():
            tv = trans_vals[col]
            row[col] = (tv - mv) / mv * 100.0 if mv not in (0, None) and np.isfinite(mv) else np.nan
        rows.append(row)

    df = pd.DataFrame(rows)
    # kolejnosc kolumn
    lead = ["role", "mesh_id", "N_cells", "solver", "description"]
    param_cols = [_col(p, ft) for p in PARAMS for _, ft in FLUIDS]
    return df[lead + param_cols]


def main():
    for geo_key in TRIPLETS:
        df = build_table(geo_key)
        out = OUT_DIR / f"gci_{geo_key}.csv"
        df.to_csv(out, index=False, float_format="%.4f")
        print(f"[OK] {out}")
        print(f"     note: {TRIPLETS[geo_key]['note']}")
    print(f"\n[DONE] Tabele GCI zapisane w: {OUT_DIR}")


if __name__ == "__main__":
    main()
