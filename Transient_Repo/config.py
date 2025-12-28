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
    "part5": {"t_start_s": 0.0, "t_end_s": 6.58, "dt_sim_s": 0.0005},
    "part6": {"t_start_s": 6.58, "t_end_s": 20.0, "dt_sim_s": 0.0005},
    "part7": {"t_start_s": 0.0, "t_end_s": 10.0, "dt_sim_s": 0.0005},
    "part8": {"t_start_s": 9.91, "t_end_s": 19.91, "dt_sim_s": 0.0005},
    "part9": {"t_start_s": 9.91, "t_end_s": 19.91, "dt_sim_s": 0.0005},
}

# =========================
# Nazewnictwo plików SRP (transient)
# =========================
# W repo obsługujemy dwa warianty:
#  1) "..._S00040.srp"  → numer kroku czasowego (S) przeliczany na czas: t_local = S * dt_sim_s
#  2) "..._T0.01.srp"   → czas w sekundach zapisany w nazwie (T)
#
# SRP_TIME_SOURCE_PRIORITY:
#   - ("T","S") oznacza: najpierw próbuj _T..., jeśli brak to _S...
#   - ("S","T") oznacza: odwrotnie
SRP_TIME_SOURCE_PRIORITY = ("T", "S")

# Jak interpretować wartość z _T...
#  - "local":  t_local = T (czas od początku partu)
#  - "global": t_local = T - PARTS[part]["t_start_s"]
#  - "auto":   jeśli T mieści się w długości partu → local, inaczej → global
SRP_T_INTERPRETATION = "global"

# USTAWIENIA RYSOWANIA 2D
PLOT_DEFAULTS = {
    "dpi": 400,
    "marker_size": 0.2,
    "line_width": 0.8,
    "overlay_every": 1,     # co ile serii rysować w overlay
    "include_pressure": False,  # dorzucaj Δp_band (overlay) i Δp_sum (mean)
}

# JOB-y (parts → fluids → plots → metrics). Opcjonalnie per job:
#  - ALL: (bez czasu) → pełny zakres partów
#  - N kroków:   {"t0_s": 10.0, "n_steps": 100}
#  - D sekund:   {"t0_s": 10.0, "duration_s": 0.5}
PLOT_JOBS: List[Dict] = [
    {
        "parts": ["part5", "part6"],
        "fluids": ["Fluid1", "Fluid2"], 
        "plots": ["overlay", "mean"], 
        "metrics": ["h", "f"],
        "overlay_mode": "mean_std"      # albo "mean_std"
    },
    {
        "parts": ["part7"],
        "fluids": ["Fluid1", "Fluid2"], 
        "plots": ["overlay", "mean"], 
        "metrics": ["h", "f"],
        "overlay_mode": "mean_std"      # albo "mean_std"
    },
    {
        "parts": ["part8"],
        "fluids": ["Fluid1", "Fluid2"], 
        "plots": ["overlay", "mean"], 
        "metrics": ["h", "f"],
        "overlay_mode": "mean_std"      # albo "mean_std"
    },
    {
        "parts": ["part9"],
        "fluids": ["Fluid1", "Fluid2"], 
        "plots": ["overlay", "mean"], 
        "metrics": ["h", "f"],
        "overlay_mode": "mean_std"      # albo "mean_std"
    },
]

COMPARE_JOBS = [
  {
    "name": "cmp_P1P3P4P5P6P7P8P9_F12_hf_dtavg_SS",
    "plots":   ["overlay", "mean"],
    "metrics": ["h", "f"],

    "overlay": {
      "time_avg": {"mode": "mean", "weights": "auto"},
      "shade": "std"
    },
    "mean": {
      "time_mode": "aligned",
      "ma_windows": [50],
      "ma_center": False,
      "ma_edges": "strict",
      "show_raw": False
    },

    # <––– NOWE: steady tylko dla tego joba
    "steady": {
      "enabled": True,
      # ścieżka relatywna względem Transient_Repo (tj. folderu z main.py)
      "base_csv_dir": r"..\Steady_Repo\DataProcessed\csv",
      "use_in_overlay": True,
      "use_in_mean": True,
      "cases": {
        "Fluid1": [
          {"label": "M006", "file": "Sim_Data_Fluid1_M006.csv"},
          {"label": "M007", "file": "Sim_Data_Fluid1_M007.csv"},
        ],
        "Fluid2": [
          {"label": "M006", "file": "Sim_Data_Fluid2_M006.csv"},
          {"label": "M007", "file": "Sim_Data_Fluid2_M007.csv"},
        ],
      },
      # wagi do średniej globalnej (dla linii w mean)
      "mean_weights": {"h": "A_wet[m2]", "f": None}
    },

    "series": [
      {"label": "Part1", "parts": ["part1"], "fluid": "Fluid1", "t0_s": 3.0, "t1_s": 4.0},
      {"label": "Part3", "parts": ["part3"], "fluid": "Fluid1", "t0_s": 10.0, "t1_s": 11.0},
      {"label": "Part4", "parts": ["part4"], "fluid": "Fluid1", "t0_s": 4.0, "t1_s": 5.0},
      {"label": "Part5", "parts": ["part5"], "fluid": "Fluid1", "t0_s": 5.0, "t1_s": 6.0},
      {"label": "Part6", "parts": ["part6"], "fluid": "Fluid1", "t0_s": 15.0, "t1_s": 16.0},
      {"label": "Part7", "parts": ["part7"], "fluid": "Fluid1", "t0_s": 8.0, "t1_s": 9.0},
      {"label": "Part8", "parts": ["part8"], "fluid": "Fluid1", "t0_s": 18.0, "t1_s": 19.0},
      {"label": "Part9", "parts": ["part9"], "fluid": "Fluid1", "t0_s": 18.0, "t1_s": 19.0},

      {"label": "Part1", "parts": ["part1"], "fluid": "Fluid2", "t0_s": 3.0, "t1_s": 4.0},
      {"label": "Part3", "parts": ["part3"], "fluid": "Fluid2", "t0_s": 10.0, "t1_s": 11.0},
      {"label": "Part4", "parts": ["part4"], "fluid": "Fluid2", "t0_s": 4.0, "t1_s": 5.0},
      {"label": "Part5", "parts": ["part5"], "fluid": "Fluid2", "t0_s": 5.0, "t1_s": 6.0},
      {"label": "Part6", "parts": ["part6"], "fluid": "Fluid2", "t0_s": 15.0, "t1_s": 16.0},
      {"label": "Part7", "parts": ["part7"], "fluid": "Fluid2", "t0_s": 8.0, "t1_s": 9.0},
      {"label": "Part8", "parts": ["part8"], "fluid": "Fluid2", "t0_s": 18.0, "t1_s": 19.0},
      {"label": "Part9", "parts": ["part9"], "fluid": "Fluid2", "t0_s": 18.0, "t1_s": 19.0},
    ],
  }
]

# Konfiguracja osi / kroku długości (m) dla płynów
FLUID_CFG: Dict[str, Dict[str, float | str]] = {
    "Fluid1": {"axis": "z", "min": -0.02959, "max":  0.02920, "step": 0.001959666667},
    "Fluid2": {"axis": "y", "min":  0.00950, "max": -0.00953, "step": 0.0009515},
}
