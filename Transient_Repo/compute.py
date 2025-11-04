# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Optional, Dict
import numpy as np
import pandas as pd

from constants import (
    COL_Z_INDEX, COL_Z_M, COL_TMASS_K, COL_P_PA, COL_A_FLOW, COL_M_DOT, COL_RHO, COL_MU, COL_K,
    COL_BAND_ID, COL_Q_W, COL_TW_K, COL_A_WET, COL_TBAND_K, COL_H_WM2K,
    COL_DP_BAND, COL_DP_SUM, COL_RE, COL_NU, COL_F_FANNING,
    COL_DH_M, COL_DH_METHOD
)

def _finite_or_nan(x) -> float:
    try:
        xf = float(x)
        return xf if np.isfinite(xf) else np.nan
    except Exception:
        return np.nan

def build_planes_df(data, *, z0: float = 0.0, dz: Optional[float] = None) -> pd.DataFrame:
    """
    DF płaszczyzn; jeśli dz podane — dorzuca kolumnę z [m].
    Zwraca standardowe kolumny; gdy brak danych — pusty DF z nagłówkami.
    """
    cols = [COL_Z_INDEX, COL_Z_M, COL_TMASS_K, COL_P_PA, COL_A_FLOW, COL_M_DOT, COL_RHO, COL_MU, COL_K]
    idxs = sorted(set(data.planes_Tmass) | set(data.planes_P) | set(data.planes_A))
    if not idxs:
        return pd.DataFrame(columns=cols)

    rows = []
    for i in idxs:
        rows.append({
            COL_Z_INDEX: i,
            COL_Z_M: (z0 + (i-1)*dz) if dz is not None else np.nan,
            COL_TMASS_K: _finite_or_nan(data.planes_Tmass.get(i)),
            COL_P_PA:    _finite_or_nan(data.planes_P.get(i)),
            COL_A_FLOW:  _finite_or_nan(data.planes_A.get(i)),
            COL_M_DOT:   _finite_or_nan(data.planes_mdot.get(i)),
            COL_RHO:     _finite_or_nan(data.planes_rho.get(i)),
            COL_MU:      _finite_or_nan(data.planes_mu.get(i)),
            COL_K:       _finite_or_nan(data.planes_k.get(i)),
        })
    df = pd.DataFrame(rows, columns=cols).sort_values(COL_Z_INDEX).reset_index(drop=True)
    return df

def build_bands_df(data) -> pd.DataFrame:
    """
    DF pasm; zawsze zwraca standardowe kolumny; gdy brak — pusty DF z nagłówkami.
    """
    cols = [COL_BAND_ID, COL_A_WET, COL_Q_W, COL_TW_K]
    idxs = sorted(set(data.bands_Awet) | set(data.bands_Q) | set(data.bands_Tw))
    if not idxs:
        return pd.DataFrame(columns=cols)

    rows = []
    for bi in idxs:
        rows.append({
            COL_BAND_ID: bi,
            COL_A_WET: _finite_or_nan(data.bands_Awet.get(bi)),
            COL_Q_W:   _finite_or_nan(data.bands_Q.get(bi)),
            COL_TW_K:  _finite_or_nan(data.bands_Tw.get(bi)),
        })
    return pd.DataFrame(rows, columns=cols).sort_values(COL_BAND_ID).reset_index(drop=True)
import numpy as np

def _lmtd_wall_band(Tw: float, T_in: float, T_out: float, eps: float = 1e-12) -> float:
    """
    Log-mean ΔT między temperaturą ścianki a temperaturą płynu na wejściu/wyjściu pasma.
    Zabezpieczenia:
      - gdy ΔT1 ≈ ΔT2 → zwraca średnią arytm. (zachowanie graniczne LMTD),
      - gdy ΔT zmienia znak (ΔT1*ΔT2<=0) → zwraca uśrednienie po modułach (z zachowaniem znaku dominującego).
    """
    dT1 = float(Tw - T_in)
    dT2 = float(Tw - T_out)

    # identyczne lub bardzo bliskie → granica LMTD = średnia arytm.
    if abs(dT1 - dT2) <= eps:
        return 0.5 * (dT1 + dT2)

    # klasyczny LMTD: tylko gdy ten sam znak i brak degeneracji
    if dT1 * dT2 > 0.0 and abs(dT1) > eps and abs(dT2) > eps:
        try:
            return (dT1 - dT2) / np.log(dT1 / dT2)
        except FloatingPointError:
            # awaryjnie wróć do średniej arytm.
            return 0.5 * (dT1 + dT2)

    # zmiana znaku (LMTD nieokreślony) → użyj uśrednienia po modułach z dominującym znakiem
    sign = np.sign(dT1) if abs(dT1) >= abs(dT2) else np.sign(dT2)
    return sign * 0.5 * (abs(dT1) + abs(dT2))

def compute_band_table(planes_df: pd.DataFrame,
                       bands_df: pd.DataFrame,
                       *,
                       dz: Optional[float] = None,
                       s_vals: Optional[np.ndarray] = None,
                       axis_label: Optional[str] = None) -> pd.DataFrame:
    """
    Parowanie (i, i+1) ↔ band i, z lokalnym dz:
      - s_vals: pozycje płaszczyzn (len = liczba płaszczyzn). Jeśli brak → użyj COL_Z_M lub stałe dz.
      - dz_k = |s[i+1]-s[i]|; z_mid = 0.5 (s[i+1]+s[i]).
      - Dh (gyroid): 4*A_flow_mid*dz_k / A_wet; fallback: sqrt(4A/pi) gdy brak danych.
      - f_fanning z lokalnym dz_k.
    Do CSV dodajemy:
      - plane_in, plane_out                (ID płaszczyzn; zwykle numer ziso)
      - {axis}_in [m], {axis}_out [m]      (współrzędne obu płaszczyzn; axis ∈ {z,y})
      - dz_local[m], z [m]                 (szerokość pasma i współrzędna środka pasma)
    """
    if planes_df is None or bands_df is None or planes_df.empty or bands_df.empty:
        return pd.DataFrame()

    p = planes_df.copy()
    b = bands_df.copy()
    for col in [COL_P_PA, COL_TMASS_K, COL_A_FLOW, COL_M_DOT, COL_RHO, COL_MU, COL_K]:
        if col in p.columns: p[col] = pd.to_numeric(p[col], errors="coerce")
    for col in [COL_A_WET, COL_Q_W, COL_TW_K]:
        if col in b.columns: b[col] = pd.to_numeric(b[col], errors="coerce")

    # sort: płaszczyzny po rosnącym z_index (jak buduje build_planes_df), pasma po band_id
    if COL_Z_INDEX in p.columns:
        p = p.sort_values(COL_Z_INDEX, kind="mergesort").reset_index(drop=True)
    else:
        p = p.reset_index(drop=True)
    if COL_BAND_ID in b.columns:
        b = b.sort_values(COL_BAND_ID, kind="mergesort").reset_index(drop=True)
    else:
        b = b.reset_index(drop=True)
        b[COL_BAND_ID] = np.arange(1, len(b) + 1, dtype=int)

    n_pairs = int(min(len(p) - 1, len(b)))
    if n_pairs < 1:
        return pd.DataFrame(columns=[
            COL_BAND_ID, "plane_in", "plane_out",
            "z_in [m]", "z_out [m]", "y_in [m]", "y_out [m]",
            COL_DP_BAND, COL_DP_SUM, COL_TBAND_K, COL_A_WET, COL_Q_W, COL_TW_K,
            COL_H_WM2K, COL_DH_M, COL_DH_METHOD, COL_RE, COL_NU, COL_F_FANNING, "dz_local[m]", COL_Z_M
        ])

    # pozycje płaszczyzn (s) i etykieta osi
    if s_vals is not None and len(s_vals) >= len(p):
        s = np.asarray(s_vals, dtype=float)[:len(p)]
    elif COL_Z_M in p.columns and p[COL_Z_M].notna().any():
        s = p[COL_Z_M].astype(float).to_numpy()
        if axis_label is None:
            axis_label = "z"
    else:
        step = float(dz) if (dz is not None and np.isfinite(dz)) else 1.0
        s = np.arange(len(p), dtype=float) * step
        if axis_label is None:
            axis_label = "z"

    # lokalne dz i z_mid
    dz_local = np.abs(s[1:n_pairs+1] - s[0:n_pairs])
    z_mid    = 0.5 * (s[1:n_pairs+1] + s[0:n_pairs])

    rows = []
    dp_sum = 0.0
    tiny = 1e-12

    for k in range(n_pairs):
        # indeksy płaszczyzn (jeśli istnieje numeracja ziso)
        plane_in_id  = int(p.at[k,   COL_Z_INDEX]) if COL_Z_INDEX in p.columns and pd.notna(p.at[k,   COL_Z_INDEX]) else (k+1)
        plane_out_id = int(p.at[k+1, COL_Z_INDEX]) if COL_Z_INDEX in p.columns and pd.notna(p.at[k+1, COL_Z_INDEX]) else (k+2)

        # wartości z płaszczyzn
        P_i  = p.at[k,   COL_P_PA]   if COL_P_PA   in p.columns else np.nan
        P_ip = p.at[k+1, COL_P_PA]   if COL_P_PA   in p.columns else np.nan

        T_i  = p.at[k,   COL_TMASS_K] if COL_TMASS_K in p.columns else np.nan
        T_ip = p.at[k+1, COL_TMASS_K] if COL_TMASS_K in p.columns else np.nan

        A_i  = p.at[k,   COL_A_FLOW] if COL_A_FLOW in p.columns else np.nan
        A_ip = p.at[k+1, COL_A_FLOW] if COL_A_FLOW in p.columns else np.nan
        A_mid = np.nanmean([A_i, A_ip])

        md_i  = p.at[k,   COL_M_DOT] if COL_M_DOT in p.columns else np.nan
        md_ip = p.at[k+1, COL_M_DOT] if COL_M_DOT in p.columns else np.nan
        md_mid = np.nanmean([md_i, md_ip])

        rho_i  = p.at[k,   COL_RHO] if COL_RHO in p.columns else np.nan
        rho_ip = p.at[k+1, COL_RHO] if COL_RHO in p.columns else np.nan
        rho_mid = np.nanmean([rho_i, rho_ip])

        mu_i  = p.at[k,   COL_MU] if COL_MU in p.columns else np.nan
        mu_ip = p.at[k+1, COL_MU] if COL_MU in p.columns else np.nan
        mu_mid = np.nanmean([mu_i, mu_ip])

        k_i  = p.at[k,   COL_K] if COL_K in p.columns else np.nan
        k_ip = p.at[k+1, COL_K] if COL_K in p.columns else np.nan
        k_mid = np.nanmean([k_i, k_ip])

        # pasmo k
        A_wet = b.at[k, COL_A_WET] if COL_A_WET in b.columns else np.nan
        Q_w   = b.at[k, COL_Q_W]   if COL_Q_W   in b.columns else np.nan
        T_w   = b.at[k, COL_TW_K]  if COL_TW_K  in b.columns else np.nan
        band_id = int(b.at[k, COL_BAND_ID])

        # długość lokalna i współrzędne krańców pasma
        dz_k   = float(dz_local[k]) if np.isfinite(dz_local[k]) else (float(dz) if (dz and np.isfinite(dz)) else np.nan)
        coord_in  = float(s[k])     if np.isfinite(s[k]) else np.nan
        coord_out = float(s[k+1])   if np.isfinite(s[k+1]) else np.nan

        # Δp i kumulacja
        dp_band = np.nan
        if np.isfinite(P_i) and np.isfinite(P_ip):
            dp_band = float(P_i - P_ip)
            dp_sum += dp_band

        # T_bulk
        # T_bulk = np.nanmean([T_i, T_ip])
        dT_lm = abs(_lmtd_wall_band(T_w, T_i, T_ip))

        # h
        h = np.nan
        if np.isfinite(Q_w) and np.isfinite(A_wet) and A_wet > tiny and np.isfinite(T_w) and np.isfinite(dT_lm):
            h = float(abs(Q_w) / (A_wet * dT_lm))

        # U
        U = np.nan
        if np.isfinite(md_mid) and np.isfinite(rho_mid) and np.isfinite(A_mid) and A_mid > tiny:
            U = float(md_mid / (rho_mid * A_mid))

        # Dh (gyroid) / fallback
        Dh = np.nan
        dh_method = "area-equiv"
        if np.isfinite(dz_k) and dz_k > tiny and np.isfinite(A_wet) and A_wet > tiny and np.isfinite(A_mid) and A_mid > tiny:
            Dh = float(4.0 * A_mid * dz_k / A_wet)
            dh_method = "wet-perim"
        if not np.isfinite(Dh) or Dh <= 0.0:
            if np.isfinite(A_mid) and A_mid > 0.0:
                Dh = float(np.sqrt(4.0 * A_mid / np.pi))
                dh_method = "area-equiv"

        # Re, Nu, f
        Re = Nu = f_f = np.nan
        if np.isfinite(Dh) and np.isfinite(U) and np.isfinite(mu_mid) and mu_mid > tiny and np.isfinite(rho_mid):
            Re = float(rho_mid * U * Dh / mu_mid)
        if np.isfinite(h) and np.isfinite(Dh) and np.isfinite(k_mid) and k_mid > tiny:
            Nu = float(h * Dh / k_mid)
        if np.isfinite(dz_k) and dz_k > tiny and np.isfinite(dp_band) and np.isfinite(rho_mid) and np.isfinite(U) and np.isfinite(Dh):
            f_f = float((dp_band / dz_k) * Dh / (2.0 * rho_mid * U * U))

        # nazwy kolumn dla współrzędnych krańców (z/y)
        axis = (axis_label or "z").lower()
        in_col  = f"{axis}_in [m]"   # np. "z_in [m]" albo "y_in [m]"
        out_col = f"{axis}_out [m]"  # np. "z_out [m]" albo "y_out [m]"

        rec = {
            COL_BAND_ID: band_id,
            "plane_in":  plane_in_id,
            "plane_out": plane_out_id,
            in_col:  coord_in,
            out_col: coord_out,
            COL_DP_BAND: dp_band,
            COL_DP_SUM: dp_sum if np.isfinite(dp_sum) else np.nan,
            COL_TBAND_K: dT_lm,
            COL_A_WET: A_wet,
            COL_Q_W: Q_w,
            COL_TW_K: T_w,
            COL_H_WM2K: h,
            COL_DH_M: Dh,
            COL_DH_METHOD: dh_method,
            COL_RE: Re,
            COL_NU: Nu,
            COL_F_FANNING: f_f,
            "dz_local[m]": dz_k,
            COL_Z_M: float(z_mid[k]) if np.isfinite(z_mid[k]) else np.nan,  # środek pasma (używany do wykresów)
        }
        rows.append(rec)

    return pd.DataFrame(rows).sort_values(COL_BAND_ID).reset_index(drop=True)


def compute_global_means(bands_df: pd.DataFrame) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if bands_df is None or bands_df.empty:
        return out
    for col in [COL_H_WM2K, COL_RE, COL_NU, COL_F_FANNING, COL_DP_BAND, COL_DP_SUM, COL_DH_M]:
        if col in bands_df.columns:
            s = bands_df[col].replace([np.inf, -np.inf], np.nan).dropna()
            out[f"mean_{col}"] = float(s.mean()) if not s.empty else float("nan")
    return out
