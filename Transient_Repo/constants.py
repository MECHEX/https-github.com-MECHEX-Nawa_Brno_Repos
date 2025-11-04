# -*- coding: utf-8 -*-
"""
Stałe wspólne dla parsera, compute i plotting.
Nazwy spójne z compute.py.
"""

# ===== Kolumny płaszczyzn (ziso_i) =====
COL_Z_INDEX = "z_index"
COL_Z_M     = "z [m]"
COL_TMASS_K = "T_mass[K]"
COL_P_PA    = "P_area[Pa]"
COL_A_FLOW  = "A_flow[m2]"
COL_M_DOT   = "m_dot[kg/s]"
COL_RHO     = "rho[kg/m3]"
COL_MU      = "mu[Pa·s]"
COL_K       = "k[W/mK]"

# ===== Kolumny pasm (wall_band_i) =====
COL_BAND_ID = "band_id"
COL_Q_W     = "Q_band[W]"
COL_TW_K    = "T_wall[K]"
COL_A_WET   = "A_wet[m2]"
COL_TBAND_K = "T_band[K]"
COL_H_WM2K  = "h[W/m2K]"

# Δp: pasmo i suma
COL_DP_BAND = "Δp_band[Pa]"
COL_DP_SUM  = "Δp_sum[Pa]"

# Liczby/średnice
COL_RE        = "Re[-]"
COL_NU        = "Nu[-]"
COL_F_FANNING = "f_fanning[-]"
COL_DH_M      = "d_h[m]"
COL_DH_METHOD = "d_h_method"

# Alias zgodności dla plotting.py
COL_F = COL_F_FANNING

# Profil po osi y (gdy dotyczy)
COL_Y_M = "y [m]"

# Które kolumny mogą pełnić rolę osi X profilu
AXIS_CANDIDATES = (COL_Z_M, COL_Y_M)

# Mapy opisów/metadanych do wykresów
YLABEL_MAP = {
    "h":       "h [W/m²K]",
    "f":       "f_Fanning [-]",
    "dp_band": "Δp (band) [Pa]",
    "dp_sum":  "Δp_sum [Pa]",
}
PREFIX_MAP = {
    "h":       "overlay_h",
    "f":       "overlay_f",
    "dp_band": "overlay_dp_band",
    "dp_sum":  "overlay_dp_sum",
}
MEAN_COL_MAP = {
    "h":      "mean_h[W/m2K]",
    "f":      "mean_f_fanning[-]",
    "dp_sum": "mean_Δp_sum[Pa]",
}

# (opcjonalne) wzorce nazw plików do rozpoznania płynu
AXIS_NAME_PATTERNS = {
    "Fluid1": ["Fluid1", "FLUID1", "fluid1", "Fluid 1", "Fluid_1", "f1", "F1"],
    "Fluid2": ["Fluid2", "FLUID2", "fluid2", "Fluid 2", "Fluid_2", "f2", "F2"],
}

# Inne drobne stałe
AXIS_STEP_TOL = 1e-6
REF_MESH_ID   = "003"
