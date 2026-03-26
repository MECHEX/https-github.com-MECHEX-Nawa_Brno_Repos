from __future__ import annotations

from typing import Dict, List


# Example transient configuration using the new case -> parts model.

DEFAULT_BASE_DATA_DIR = r".\FluentTransientData"
DEFAULT_OUT_DIR = r".\TransientFigs"


SRP_DEFAULTS = {
    "time_source_priority": ("T", "S"),
    "t_interpretation": "global",
}

# Optional per part:
# "fixed_t_local_s": 0.0
# Use only for snapshot-like inputs whose file names do not contain T/S tokens.


CASES: Dict[str, Dict] = {
    "uni10_001": {
        "active": True,
        "geometry": "uni10",
        "run": "001",
        "description": "UNI10, air/water, transient",
        "parts": {
            "part1": {
                "source_dir": "uni10_001_part1",
                "t_start_s": 0.0,
                "t_end_s": 5.0,
                "dt_sim_s": 0.0005,
            },
            "part2": {
                "source_dir": "uni10_001_part2",
                "t_start_s": 5.0,
                "t_end_s": 10.0,
                "dt_sim_s": 0.0005,
            },
        },
    },
    "guni10_003": {
        "active": False,
        "geometry": "guni10",
        "run": "003",
        "description": "GUNI10, air/water, transient",
        "parts": {
            "main": {
                "source_dir": "guni10_003",
                "t_start_s": 0.0,
                "t_end_s": 20.0,
                "dt_sim_s": 0.0005,
            },
        },
    },
}


PLOT_DEFAULTS = {
    "dpi": 400,
    "marker_size": 0.2,
    "line_width": 0.8,
    "overlay_every": 1,
    "include_pressure": False,
}


PLOT_JOBS: List[Dict] = [
    {
        "name": "job_uni10_001",
        "active": True,
        "members": [
            {"case_id": "uni10_001", "parts": ["part1", "part2"]},
        ],
        "fluids": ["Fluid1", "Fluid2"],
        "plots": ["overlay", "mean"],
        "metrics": ["h", "f"],
        "overlay_mode": "mean_std",
    },
    {
        "name": "job_compare_geometry_snapshots",
        "active": False,
        "members": [
            {"case_id": "uni10_001", "parts": ["part2"]},
            {"case_id": "guni10_003"},
        ],
        "fluids": ["Fluid1"],
        "plots": ["overlay"],
        "metrics": ["h"],
    },
]


COMPARE_JOBS = [
    {
        "name": "compare_001",
        "active": True,
        "plots": ["overlay", "mean"],
        "metrics": ["h", "f"],
        "overlay": {
            "time_avg": {"mode": "mean", "weights": "auto"},
            "shade": "std",
        },
        "mean": {
            "time_mode": "aligned",
            "ma_windows": [50],
            "ma_center": False,
            "ma_edges": "strict",
            "show_raw": False,
        },
        "fig": {
            "dpi": 450,
            "overlay": {"figsize": (14.0, 5.2)},
            "mean": {"figsize": (14.0, 5.2)},
        },
        "plot": {
            "base_lw": 1.0,
            "base_ms": 5.0,
            "raw_alpha": 0.35,
            "ma_alpha": 0.6,
            "raw_lw_scale": 0.85,
            "ma_lw_scale": 1.00,
            "ref_ls": ":",
            "ref_alpha": 0.90,
            "marker_mode": "random",
            "marker_target": 16,
            "marker_seed": 56,
        },
        "style_map": {
            "UNI10 run001": {"color": "#1f77b4", "ls": "-", "marker": "o"},
            "GUNI10 run003": {"color": "#ff7f0e", "ls": "--", "marker": "s"},
        },
        "steady": {
            "enabled": False,
            "base_csv_dir": r"..\Steady_Repo\DataProcessed\csv",
            "use_in_overlay": True,
            "use_in_mean": True,
            "cases": {"Fluid1": [], "Fluid2": []},
            "mean_weights": {"h": "A_wet[m2]", "f": None},
        },
        "series": [
            {
                "label": "UNI10 run001",
                "case_id": "uni10_001",
                "parts": ["part1", "part2"],
                "fluid": "Fluid1",
                "t0_s": 3.0,
                "t1_s": 9.0,
            },
            {
                "label": "GUNI10 run003",
                "case_id": "guni10_003",
                "parts": ["main"],
                "fluid": "Fluid1",
                "t0_s": 3.0,
                "t1_s": 9.0,
            },
        ],
    }
]


FLUID_CFG: Dict[str, Dict[str, float | str]] = {
    "Fluid1": {"axis": "z", "min": -0.02959, "max": 0.02920, "step": 0.001959666667},
    "Fluid2": {"axis": "y", "min": 0.00950, "max": -0.00953, "step": 0.0009515},
}
