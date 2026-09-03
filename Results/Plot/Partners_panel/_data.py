"""Shared data layer for the partner-meeting panels.

Reads the already-generated CSV products of the two campaigns:

  Run 001 (steady, GRAD/M009, six flow regimes R000-R005)
      ../Run001_two_simu/band_values_*.csv     band-resolved h, Re, Nu, dp
      ../Run001_two_simu/summary_*.csv         global (run, side) averages

  Run 002 (transient, 18-19 s window, GRAD vs UNI10, three cold fluids)
      ../Run002_transient_fluids/transient_global_summary_18_19s.csv
      ../Run002_transient_fluids/pec_ggraded_vs_guni.csv
      ../Run002_transient_fluids/energy_balance_18_19s.csv

No physics is recomputed here - only selection, power-law fitting and
outlier flagging.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RUN001_DIR = HERE.parent / "Run001_two_simu"
RUN002_DIR = HERE.parent / "Run002_transient_fluids"

RUNS = ("R000", "R001", "R002", "R003", "R004", "R005")
RUN_TAG = "_".join(RUNS)

# Side naming used in every panel (English, partner-facing).
SIDE_OF_FLUID = {"Fluid1": "Air", "Fluid2": "Water"}

# Okabe-Ito, colour-blind safe; identical hues to the other repo figures.
C_AIR = "#D55E00"
C_WATER = "#0072B2"
C_OIL = "#E69F00"
C_HFE = "#009E73"
C_INK = "#1a1a1a"
C_MUTED = "#6b6b6b"
C_GRID = "#c9c9c9"

SIDE_COLOR = {"Air": C_AIR, "Water": C_WATER}
SIDE_MARKER = {"Air": "o", "Water": "s"}
FLUID_COLOR = {"Water": C_WATER, "Oil": C_OIL, "HFE": C_HFE}
GEOM_LABEL = {"ggraded": "GRAD (graded gyroid)", "guni": "UNI10 (uniform gyroid)"}
GEOM_MARKER = {"ggraded": "o", "guni": "D"}

# Outlier rule (band-resolved h).  A single, deliberately narrow criterion:
# a band is rejected when its h drops below OUTLIER_FRAC of the run/side
# median.  It targets one physical artefact only - a band in which the
# wall-to-fluid LMTD collapses towards zero, so that h = Q / (A * LMTD)
# becomes numerically meaningless.  The threshold is set an order of
# magnitude below the genuine outlet decay (weakest real band is ~0.39 of
# the median), so no physical end-of-core weakening is removed.
OUTLIER_FRAC = 0.10


# --------------------------------------------------------------------------- #
# Run 001 - steady, six flow regimes
# --------------------------------------------------------------------------- #
def load_bands() -> pd.DataFrame:
    path = RUN001_DIR / f"band_values_{RUN_TAG}.csv"
    df = pd.read_csv(path)
    df["side"] = df["fluid"].map(SIDE_OF_FLUID)
    # Normalised streamwise coordinate, band midpoint -> [0, 1].
    x = df.groupby(["run", "side"])["distance_from_inlet_mm"]
    df["x_over_L"] = (df["distance_from_inlet_mm"] - x.transform("min")) / (
        x.transform("max") - x.transform("min")
    )
    return flag_outliers(df)


def flag_outliers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_outlier"] = False
    df["h_over_median"] = np.nan
    df["outlier_reason"] = ""
    for _, grp in df.groupby(["run", "side"]):
        h = grp["h_W_m2K"].to_numpy(dtype=float)
        med = float(np.median(h))
        ratio = h / med
        df.loc[grp.index, "h_over_median"] = ratio
        bad = ratio < OUTLIER_FRAC
        if not bad.any():
            continue
        df.loc[grp.index[bad], "is_outlier"] = True
        df.loc[grp.index[bad], "outlier_reason"] = "h < %.2f x run/side median (LMTD collapse)" % OUTLIER_FRAC
    return df


def load_summary() -> pd.DataFrame:
    path = RUN001_DIR / f"summary_{RUN_TAG}.csv"
    df = pd.read_csv(path)
    df["side"] = df["fluid"].map(SIDE_OF_FLUID)
    return df


# --------------------------------------------------------------------------- #
# Power-law fitting
# --------------------------------------------------------------------------- #
def power_fit(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Least squares on log10 -> y = a * x**b, with R^2 in log space."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    lx, ly = np.log10(x[ok]), np.log10(y[ok])
    b, log_a = np.polyfit(lx, ly, 1)
    pred = b * lx + log_a
    ss_res = float(np.sum((ly - pred) ** 2))
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    return {
        "a": float(10.0**log_a),
        "b": float(b),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan,
        "n": int(ok.sum()),
        "x_min": float(x[ok].min()),
        "x_max": float(x[ok].max()),
    }


# --------------------------------------------------------------------------- #
# Run 002 - transient, three cold fluids, two geometries
# --------------------------------------------------------------------------- #
def load_transient() -> pd.DataFrame:
    return pd.read_csv(RUN002_DIR / "transient_global_summary_18_19s.csv")


def load_pec() -> pd.DataFrame:
    return pd.read_csv(RUN002_DIR / "pec_ggraded_vs_guni.csv")


def load_energy_balance() -> pd.DataFrame:
    return pd.read_csv(RUN002_DIR / "energy_balance_18_19s.csv")


def transient_pairs() -> pd.DataFrame:
    """One row per (geometry, cold fluid): air side vs cold side."""
    tr = load_transient()
    rows = []
    for (geom, cold), grp in tr.groupby(["geometry", "cold_fluid"]):
        air = grp[grp["side"] == "Air"].iloc[0]
        liq = grp[grp["side"] == cold].iloc[0]
        rows.append(
            {
                "geometry": geom,
                "cold_fluid": cold,
                "h_air": air["h_area_W_m2K_mean"],
                "h_air_std": air["h_area_W_m2K_std"],
                "h_liquid": liq["h_area_W_m2K_mean"],
                "h_liquid_std": liq["h_area_W_m2K_std"],
                "Q_air": air["Q_total_W_mean"],
                "Q_liquid": liq["Q_total_W_mean"],
                "P_pump_air": air["pump_power_W_mean"],
                "P_pump_liquid": liq["pump_power_W_mean"],
                "dp_liquid": liq["dp_total_Pa_mean"],
            }
        )
    return pd.DataFrame(rows)
