# -*- coding: utf-8 -*-

# Kolumny płaszczyzn (ziso_i)
COL_Z_INDEX = "z_index"
COL_Z_M     = "z [m]"
COL_TMASS_K = "T_mass[K]"
COL_P_PA    = "P_area[Pa]"
COL_A_FLOW  = "A_flow[m2]"
COL_M_DOT   = "m_dot[kg/s]"
COL_RHO     = "rho[kg/m3]"
COL_MU      = "mu[Pa·s]"
COL_K       = "k[W/mK]"

# Kolumny pasm (wall_band_i)
COL_BAND_ID   = "band_id"
COL_Q_W       = "Q_band[W]"
COL_TW_K      = "T_wall[K]"
COL_A_WET     = "A_wet[m2]"
COL_TBAND_K   = "T_bulk_band[K]"
COL_H_WM2K    = "h[W/m2K]"
COL_DP_BAND   = "P_drop_band[Pa]"
COL_DP_SUM    = "P_drop_sum[Pa]"
COL_RE        = "Re[-]"
COL_NU        = "Nu[-]"
COL_F_FANNING = "f_fanning[-]"

# D_h
COL_DH_M      = "Dh[m]"
COL_DH_METHOD = "Dh_method"

# Wyjście
DEFAULT_OUTDIR       = "DataProcessed"
DEFAULT_PLOTS_SUBDIR = "plots"
DEFAULT_CSV_SUBDIR   = "csv"

# Wykresy
DEFAULT_DPI  = 200

# --- USTAWIENIA ROZSTAWÓW PŁASZCZYZN (TYLKO TUTAJ) ---
AXIS_RANGES = {
    # Fluid 1: oś z
    "Fluid1": {
        "axis": "z",
        "min": -0.02959,
        "max":  0.0292,
        "step": 0.001959666667,
    },
    # Fluid 2: oś y (malejąco)
    "Fluid2": {
        "axis": "y",
        "min":  0.0095,
        "max": -0.00953,
        "step": 0.0009515,
    },
}

# Jak rozpoznać w nazwie pliku, które ustawienia zastosować
AXIS_NAME_PATTERNS = {
    "Fluid1": ["Fluid1", "FLUID1", "fluid1", "Fluid 1", "Fluid_1", "f1", "F1"],
    "Fluid2": ["Fluid2", "FLUID2", "fluid2", "Fluid 2", "Fluid_2", "f2", "F2"],
}

# Tolerancja sanity-check kroku
AXIS_STEP_TOL = 1e-6

# --- referencja dla błędów względnych (np. Mesh_003) ---
REF_MESH_ID = "003"            # tylko trzycyfrowy identyfikator

