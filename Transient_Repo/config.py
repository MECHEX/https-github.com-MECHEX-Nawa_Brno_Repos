# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Dict, List


# Paths can still be overridden from the CLI.
DEFAULT_BASE_DATA_DIR = (
    r"C:\Users\kik\My Drive\Politechnika Krakowska\Researches"
    r"\2025_NAWA_Brno\Nawa_Brno_Repos\Transient_Repo\FluentTransientData"
)
DEFAULT_OUT_DIR = (
    r"C:\Users\kik\My Drive\Politechnika Krakowska\Researches"
    r"\2025_NAWA_Brno\Nawa_Brno_Repos\Transient_Repo\TransientFigs"
)


# Global defaults for extracting time from SRP file names.
SRP_DEFAULTS = {
    "time_source_priority": ("T", "S"),
    "t_interpretation": "global",
}


# Source of truth for transient datasets.
# - case_id identifies the simulation family / run.
# - parts are simulation segments and may cover different time ranges.
# - source_dir points to the real directory under FluentTransientData.
# - optional "active": False disables the whole case.
CASES: Dict[str, Dict] = {
    "grad_aw_laminar": {
        "active": False,
        "geometry": "grad",
        "run": "aw_laminar",
        "description": "GRAD geometry, air/water, transient, laminar",
        "parts": {
            "part1": {
                "source_dir": "part1",
                "t_start_s": 0.0,
                "t_end_s": 5.0,
                "dt_sim_s": 0.0005,
            },
            "part2": {
                "source_dir": "part2",
                "t_start_s": 5.0,
                "t_end_s": 10.0,
                "dt_sim_s": 0.0005,
            },
            "part3": {
                "source_dir": "part3",
                "t_start_s": 10.0,
                "t_end_s": 11.0,
                "dt_sim_s": 0.0001,
            },
        },
    },
    "grad_aw_kw_earsm": {
        "active": False,
        "geometry": "grad",
        "run": "aw_kw_earsm",
        "description": "GRAD geometry, air/water, transient, k-w EARSM",
        "parts": {
            "part4": {
                "source_dir": "part4",
                "t_start_s": 0.0,
                "t_end_s": 10.0,
                "dt_sim_s": 0.00025,
            },
        },
    },
    "grad_ao_laminar": {
        "active": False,
        "geometry": "grad",
        "run": "ao_laminar",
        "description": "GRAD geometry, air/oil, transient, laminar",
        "parts": {
            "part5": {
                "source_dir": "part5",
                "t_start_s": 0.0,
                "t_end_s": 6.58,
                "dt_sim_s": 0.0005,
            },
            "part6": {
                "source_dir": "part6",
                "t_start_s": 6.58,
                "t_end_s": 20.0,
                "dt_sim_s": 0.0005,
            },
        },
    },
    "grad_ao_kw_earsm": {
        "active": False,
        "geometry": "grad",
        "run": "ao_kw_earsm",
        "description": "GRAD geometry, air/oil, transient, k-w EARSM, 0-10 s",
        "parts": {
            "part7": {
                "source_dir": "part7",
                "t_start_s": 0.0,
                "t_end_s": 10.0,
                "dt_sim_s": 0.0005,
            },
        },
    },
    "grad_ao_kw_earsm_restart": {
        "active": False,
        "geometry": "grad",
        "run": "ao_kw_earsm_restart",
        "description": "GRAD geometry, air/oil, transient, k-w EARSM, 9.91-19.91 s",
        "parts": {
            "part8": {
                "source_dir": "part8",
                "t_start_s": 9.91,
                "t_end_s": 19.91,
                "dt_sim_s": 0.0005,
            },
        },
    },
    "grad_ao_kw_earsm_restart_lowres": {
        "active": False,
        "geometry": "grad",
        "run": "ao_kw_earsm_restart_lowres",
        "description": "GRAD geometry, air/oil, transient, k-w EARSM, lower res limit",
        "parts": {
            "part9": {
                "source_dir": "part9",
                "t_start_s": 9.91,
                "t_end_s": 19.91,
                "dt_sim_s": 0.0005,
            },
        },
    },
    "uni10_003": {
        "active": True,
        "geometry": "uni10",
        "run": "003",
        "description": "UNI10 geometry, run 003, air/water, transient, 0-20 s",
        "parts": {
            "main": {
                "source_dir": "uni10_003",
                "t_start_s": 0.0,
                "t_end_s": 20.0,
                "dt_sim_s": 0.0005,
            },
        },
    },
    "uni10_005": {
        "active": True,
        "geometry": "uni10",
        "run": "005",
        "description": "UNI10 geometry, run 005, air/water, transient, 0-21.28 s",
        "parts": {
            "main": {
                "source_dir": "uni10_005",
                "t_start_s": 0.0,
                "t_end_s": 21.28,
                "dt_sim_s": 0.0005,
            },
        },
    },
    "uni10_007": {
        "active": True,
        "geometry": "uni10",
        "run": "007",
        "description": "UNI10 geometry, run 007, air/water, transient, 0-20.14 s",
        "parts": {
            "main": {
                "source_dir": "uni10_007",
                "t_start_s": 0.0,
                "t_end_s": 20.14,
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


GRAD_PART_CASE_MAP = {
    "part1": "grad_aw_laminar",
    "part2": "grad_aw_laminar",
    "part3": "grad_aw_laminar",
    "part4": "grad_aw_kw_earsm",
    "part5": "grad_ao_laminar",
    "part6": "grad_ao_laminar",
    "part7": "grad_ao_kw_earsm",
    "part8": "grad_ao_kw_earsm_restart",
    "part9": "grad_ao_kw_earsm_restart_lowres",
}

GRAD_COMPARE_WINDOWS_S = {
    "part1": (3.0, 4.0),
    "part2": (8.0, 9.0),
    "part3": (10.0, 11.0),
    "part4": (4.0, 5.0),
    "part5": (5.0, 6.0),
    "part6": (15.0, 16.0),
    "part7": (8.0, 9.0),
    "part8": (18.0, 19.0),
    "part9": (18.0, 19.0),
}

UNI_COMPARE_WINDOWS_S = {
    "uni10_003": (18.0, 19.0),
    "uni10_005": (18.0, 19.0),
    "uni10_007": (18.0, 19.0),
}


# Optional on every job:
# "active": False
# Inactive jobs are ignored completely and their data is not indexed.
PLOT_JOBS: List[Dict] = [
    {
        "name": f"job_grad_{part_id}",
        "active": False,
        "members": [{"case_id": case_id, "parts": [part_id]}],
        "fluids": ["Fluid1", "Fluid2"],
        "plots": ["overlay", "mean"],
        "metrics": ["h", "f"],
        "overlay_mode": "mean_std",
    }
    for part_id, case_id in GRAD_PART_CASE_MAP.items()
]


COMPARE_JOBS = [
    {
        "name": "compare_grad_P1_P9",
        "active": False,
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
            "Part1": {"color": "#1f77b4", "ls": "-", "marker": "o", "lw_scale": 1.00, "ms_scale": 1.00},
            "Part2": {"color": "#ff7f0e", "ls": "--", "marker": "s", "lw_scale": 1.00, "ms_scale": 1.00},
            "Part3": {"color": "#2ca02c", "ls": "-.", "marker": "^", "lw_scale": 1.00, "ms_scale": 1.00},
            "Part4": {"color": "#d62728", "ls": ":", "marker": "D", "lw_scale": 1.00, "ms_scale": 1.00},
            "Part5": {"color": "#9467bd", "ls": "-", "marker": "v", "lw_scale": 1.15, "ms_scale": 1.10},
            "Part6": {"color": "#8c564b", "ls": "--", "marker": "P"},
            "Part7": {"color": "#e377c2", "ls": "-.", "marker": "X"},
            "Part8": {"color": "#7f7f7f", "ls": ":", "marker": "*"},
            "Part9": {"color": "#bcbd22", "ls": "--", "marker": "h"},
            "M006 SS": {"color": "#000000", "ls": ":", "lw": 2.2, "marker": None},
            "M007 SS": {"color": "#444444", "ls": ":", "lw": 2.2, "marker": None},
        },
        "steady": {
            "enabled": True,
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
            "mean_weights": {"h": "A_wet[m2]", "f": None},
        },
        "series": (
            [
                {
                    "label": f"Part{part_id.removeprefix('part')}",
                    "case_id": case_id,
                    "parts": [part_id],
                    "fluid": fluid,
                    "t0_s": GRAD_COMPARE_WINDOWS_S[part_id][0],
                    "t1_s": GRAD_COMPARE_WINDOWS_S[part_id][1],
                }
                for part_id, case_id in GRAD_PART_CASE_MAP.items()
                for fluid in ("Fluid1", "Fluid2")
            ]
        ),
    },
    {
        "name": "compare_grad_uni10_001",
        "active": False,
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
            "Part1": {"color": "#1f77b4", "ls": "-", "marker": "o", "lw_scale": 1.00, "ms_scale": 1.00},
            "Part2": {"color": "#ff7f0e", "ls": "--", "marker": "s", "lw_scale": 1.00, "ms_scale": 1.00},
            "Part3": {"color": "#2ca02c", "ls": "-.", "marker": "^", "lw_scale": 1.00, "ms_scale": 1.00},
            "Part4": {"color": "#d62728", "ls": ":", "marker": "D", "lw_scale": 1.00, "ms_scale": 1.00},
            "Part5": {"color": "#9467bd", "ls": "-", "marker": "v", "lw_scale": 1.15, "ms_scale": 1.10},
            "Part6": {"color": "#8c564b", "ls": "--", "marker": "P"},
            "Part7": {"color": "#e377c2", "ls": "-.", "marker": "X"},
            "Part8": {"color": "#7f7f7f", "ls": ":", "marker": "*"},
            "Part9": {"color": "#bcbd22", "ls": "--", "marker": "h"},
            "UNI10 003": {"color": "#17becf", "ls": "-", "marker": "o"},
            "UNI10 005": {"color": "#4d4d4d", "ls": "--", "marker": "s"},
            "UNI10 007": {"color": "#8a8a00", "ls": "-.", "marker": "^"},
            "M006 SS": {"color": "#000000", "ls": ":", "lw": 2.2, "marker": None},
            "M007 SS": {"color": "#444444", "ls": ":", "lw": 2.2, "marker": None},
        },
        "steady": {
            "enabled": True,
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
            "mean_weights": {"h": "A_wet[m2]", "f": None},
        },
        "series": (
            [
                {
                    "label": f"Part{part_id.removeprefix('part')}",
                    "case_id": case_id,
                    "parts": [part_id],
                    "fluid": fluid,
                    "t0_s": GRAD_COMPARE_WINDOWS_S[part_id][0],
                    "t1_s": GRAD_COMPARE_WINDOWS_S[part_id][1],
                }
                for part_id, case_id in GRAD_PART_CASE_MAP.items()
                for fluid in ("Fluid1", "Fluid2")
            ]
            +
            [
                {
                    "label": f"UNI10 {case_id.removeprefix('uni10_')}",
                    "case_id": case_id,
                    "parts": ["main"],
                    "fluid": fluid,
                    "t0_s": UNI_COMPARE_WINDOWS_S[case_id][0],
                    "t1_s": UNI_COMPARE_WINDOWS_S[case_id][1],
                }
                for case_id in ("uni10_003", "uni10_005", "uni10_007")
                for fluid in ("Fluid1", "Fluid2")
            ]
        ),
    },
    {
        "name": "compare_uni10_with_steady",
        "active": True,
        "plots": ["overlay", "mean"],
        "metrics": ["h", "f"],
        "overlay": {
            "time_avg": {"mode": "mean", "weights": "auto"},
            "shade": "std",
        },
        "mean": {
            "time_mode": "aligned",
            "ma_windows": [2],
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
            "UNI10 003": {"color": "#17becf", "ls": "-", "marker": "o"},
            "UNI10 005": {"color": "#4d4d4d", "ls": "--", "marker": "s"},
            "UNI10 007": {"color": "#8a8a00", "ls": "-.", "marker": "^"},
            "UNI10 003 SS": {"color": "#17becf", "ls": ":", "lw": 2.2, "marker": None},
            "UNI10 005 SS": {"color": "#4d4d4d", "ls": ":", "lw": 2.2, "marker": None},
            "UNI10 007 SS": {"color": "#8a8a00", "ls": ":", "lw": 2.2, "marker": None},
        },
        "steady": {
            "enabled": True,
            "base_csv_dir": r"..\Steady_Repo\DataProcessed\csv",
            "use_in_overlay": True,
            "use_in_mean": True,
            "cases": {
                "Fluid1": [
                    {"label": "UNI10 003", "file": "Sim_Data_Fluid1_GUNI_003.csv"},
                    {"label": "UNI10 005", "file": "Sim_Data_Fluid1_GUNI_005.csv"},
                    {"label": "UNI10 007", "file": "Sim_Data_Fluid1_GUNI_007.csv"},
                ],
                "Fluid2": [
                    {"label": "UNI10 003", "file": "Sim_Data_Fluid2_GUNI_003.csv"},
                    {"label": "UNI10 005", "file": "Sim_Data_Fluid2_GUNI_005.csv"},
                    {"label": "UNI10 007", "file": "Sim_Data_Fluid2_GUNI_007.csv"},
                ],
            },
            "mean_weights": {"h": "A_wet[m2]", "f": None},
        },
        "series": (
            [
                {
                    "label": f"UNI10 {case_id.removeprefix('uni10_')}",
                    "case_id": case_id,
                    "parts": ["main"],
                    "fluid": fluid,
                    "t0_s": UNI_COMPARE_WINDOWS_S[case_id][0],
                    "t1_s": UNI_COMPARE_WINDOWS_S[case_id][1],
                }
                for case_id in ("uni10_003", "uni10_005", "uni10_007")
                for fluid in ("Fluid1", "Fluid2")
            ]
        ),
    },
]


FLUID_CFG: Dict[str, Dict[str, float | str]] = {
    "Fluid1": {"axis": "z", "min": -0.02959, "max": 0.02920, "step": 0.001959666667},
    "Fluid2": {"axis": "y", "min": 0.00950, "max": -0.00953, "step": 0.0009515},
}
