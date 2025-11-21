# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, List

# ŚCIEŻKI (możesz nadpisać parametrami --in-dir / --out-dir)
DEFAULT_BASE_DATA_DIR = r"C:\Users\kik\My Drive\Politechnika Krakowska\Researches\2025_NAWA_Brno\Nawa_Brno_Repos\Transient_Repo\FluentTransientData"
DEFAULT_OUT_DIR       = r"C:\Users\kik\My Drive\Politechnika Krakowska\Researches\2025_NAWA_Brno\Nawa_Brno_Repos\Transient_Repo\TransientFigs"

# PARTS: czasy GLOBALNE i krok czasu SYMULACJI w danym parcie
PARTS: Dict[str, Dict[str, float]] = {
    "part1": {"t_start_s": 0.0,  "t_end_s": 5.0,  "dt_sim_s": 0.0005},
    "part2": {"t_start_s": 5.0,  "t_end_s": 10.0, "dt_sim_s": 0.0005},
    "part3": {"t_start_s": 10.0, "t_end_s": 11.0, "dt_sim_s": 0.0001},
    "part4": {"t_start_s": 0.0, "t_end_s": 10.0, "dt_sim_s": 0.00025},
}

# USTAWIENIA RYSOWANIA 2D
PLOT_DEFAULTS = {
    "dpi": 400,
    "marker_size": 0.05,
    "line_width": 0.4,
    "overlay_every": 1,     # co ile serii rysować w overlay
    "include_pressure": False,  # dorzucaj Δp_band (overlay) i Δp_sum (mean)
}

# JOB-y (parts → fluids → plots → metrics). Opcjonalnie per job:
#  - ALL: (bez czasu) → pełny zakres partów
#  - N kroków:   {"t0_s": 10.0, "n_steps": 100}
#  - D sekund:   {"t0_s": 10.0, "duration_s": 0.5}
PLOT_JOBS: List[Dict] = [
    {
        "parts": ["part4"],
        "fluids": ["Fluid1", "Fluid2"], 
        "plots": ["overlay", "mean"], 
        "metrics": ["h", "f"]
    },
    {
        "parts": ["part4"],          
        "fluids": ["Fluid1", "Fluid2"], 
        "plots": ["overlay"], 
        "metrics": ["h", "f"],
        "t0_s": 3.0, 
        "duration_s": 8.0, 
    },
    {
        "parts": ["part1", "part2"], 
        "fluids": ["Fluid1", "Fluid2"], 
        "plots": ["overlay", "mean"], 
        "metrics": ["h", "f"]
    },
    {
        "parts": ["part3"],          
        "fluids": ["Fluid1", "Fluid2"], 
        "plots": ["overlay", "mean"], 
        "metrics": ["h", "f"],
    },

]

# Konfiguracja osi / kroku długości (m) dla płynów
FLUID_CFG: Dict[str, Dict[str, float | str]] = {
    "Fluid1": {"axis": "z", "min": -0.02959, "max":  0.02920, "step": 0.001959666667},
    "Fluid2": {"axis": "y", "min":  0.00950, "max": -0.00953, "step": 0.0009515},
}
