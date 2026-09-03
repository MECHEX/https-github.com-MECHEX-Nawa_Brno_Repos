"""Partner-meeting panels for the TPMS gyroid heat exchanger study.

Builds two figures from the existing Run 001 (steady, six flow regimes) and
Run 002 (transient, three cold fluids, two geometries) products:

  panel_partners_2x3.*   full story  - (a) Nu-Re, (b) dp-U, (c) decoupling,
                                       (d) band profiles, (e) cold-fluid
                                       choice, (f) PEC + validation
  panel_partners_2x2.*   short story - (a), (d), (e), (f)

Side products:
  correlations_global.csv   the fitted design correlations
  outliers_excluded.csv     bands removed from the band-resolved statistics
  panel_key_numbers.csv     every number annotated on the figures

Band-level correlation quality is deliberately NOT reported here; the fitted
R^2 shown on panel (a) refers to the global (run-averaged) points only.

Usage:  python make_partner_panels.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _data as D

OUT_DIR = Path(__file__).resolve().parent
DPI = 260

plt.rcParams.update(
    {
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9.5,
        "legend.fontsize": 8.2,
        "xtick.labelsize": 8.8,
        "ytick.labelsize": 8.8,
        "axes.edgecolor": "#4d4d4d",
        "text.color": D.C_INK,
        "axes.labelcolor": D.C_INK,
        "xtick.color": "#4d4d4d",
        "ytick.color": "#4d4d4d",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)

KEY_NUMBERS: list[dict[str, object]] = []


def note(panel: str, quantity: str, value, unit: str = "") -> None:
    KEY_NUMBERS.append(
        {"panel": panel, "quantity": quantity, "value": value, "unit": unit}
    )


def style(ax: plt.Axes, title: str) -> None:
    ax.grid(True, which="major", linestyle=":", linewidth=0.7, color=D.C_GRID, alpha=0.9)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(direction="out", length=3)
    ax.set_title(title, fontweight="bold", loc="left", pad=8)


# --------------------------------------------------------------------------- #
# (a) Global Nu-Re correlation
# --------------------------------------------------------------------------- #
def panel_a(ax: plt.Axes, bands: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    clean = bands[~bands["is_outlier"]]
    fits = []
    for side in ("Air", "Water"):
        col = D.SIDE_COLOR[side]
        cloud = clean[clean["side"] == side]
        ax.scatter(
            cloud["Re"], cloud["Nu"], s=9, color=col, alpha=0.16, linewidths=0, zorder=1
        )

        g = summary[summary["side"] == side]
        fit = D.power_fit(g["Re_mean"].to_numpy(), g["Nu_mean"].to_numpy())
        fit["side"] = side
        fit["correlation"] = "Nu = a * Re^b"
        fits.append(fit)

        xs = np.logspace(
            np.log10(fit["x_min"]) - 0.05, np.log10(fit["x_max"]) + 0.05, 60
        )
        ax.plot(xs, fit["a"] * xs ** fit["b"], color=col, lw=1.6, ls="--", zorder=3)
        ax.scatter(
            g["Re_mean"],
            g["Nu_mean"],
            s=62,
            color=col,
            marker=D.SIDE_MARKER[side],
            edgecolor="white",
            linewidth=1.1,
            zorder=4,
            label=side + " — run average (6 runs)",
        )
        note("a", side + ": Nu = a Re^b, a", round(fit["a"], 3))
        note("a", side + ": Nu = a Re^b, b", round(fit["b"], 3))
        note("a", side + ": R2 (global fit)", round(fit["r2"], 4))
        note("a", side + ": Re range", "%.0f-%.0f" % (fit["x_min"], fit["x_max"]))

    air, water = fits[0], fits[1]
    ax.text(
        0.985,
        0.31,
        "Air:    Nu = %.3f·Re$^{%.3f}$\n        R² = %.4f   (Re %.0f–%.0f)"
        % (air["a"], air["b"], air["r2"], air["x_min"], air["x_max"]),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.4,
        color=D.C_AIR,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=D.C_AIR, alpha=0.92, lw=0.8),
    )
    ax.text(
        0.985,
        0.145,
        "Water: Nu = %.3f·Re$^{%.3f}$\n        R² = %.4f   (Re %.0f–%.0f)"
        % (water["a"], water["b"], water["r2"], water["x_min"], water["x_max"]),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.4,
        color=D.C_WATER,
        bbox=dict(
            boxstyle="round,pad=0.35", fc="white", ec=D.C_WATER, alpha=0.92, lw=0.8
        ),
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Re [–]")
    ax.set_ylabel("Nu [–]")
    handles, labels = ax.get_legend_handles_labels()
    handles.append(
        Line2D([], [], marker="o", ls="none", color="0.55", alpha=0.5, markersize=4)
    )
    labels.append("band-resolved points (background)")
    ax.legend(handles, labels, loc="upper left", frameon=False)
    style(ax, "(a)  Design correlation Nu(Re) — both streams")
    return pd.DataFrame(fits)


# --------------------------------------------------------------------------- #
# (b) Pressure drop vs pore velocity
# --------------------------------------------------------------------------- #
def panel_b(ax: plt.Axes, summary: pd.DataFrame) -> pd.DataFrame:
    fits = []
    for side in ("Air", "Water"):
        col = D.SIDE_COLOR[side]
        g = summary[summary["side"] == side]
        fit = D.power_fit(g["mean_velocity_m_s"].to_numpy(), g["dp_total_Pa"].to_numpy())
        fit["side"] = side
        fit["correlation"] = "dp = a * U^b"
        fits.append(fit)

        xs = np.logspace(
            np.log10(fit["x_min"]) - 0.08, np.log10(fit["x_max"]) + 0.08, 60
        )
        ax.plot(xs, fit["a"] * xs ** fit["b"], color=col, lw=1.6, ls="--", zorder=3)
        ax.scatter(
            g["mean_velocity_m_s"],
            g["dp_total_Pa"],
            s=62,
            color=col,
            marker=D.SIDE_MARKER[side],
            edgecolor="white",
            linewidth=1.1,
            zorder=4,
            label="%s:  Δp ∝ U$^{%.2f}$" % (side, fit["b"]),
        )

        # Darcy reference (n = 1) anchored at the slowest point of this side.
        x0 = fit["x_min"]
        y0 = fit["a"] * x0 ** fit["b"]
        ax.plot(xs, y0 * (xs / x0), color="0.45", lw=1.1, ls=":", zorder=2)

        note("b", side + ": dp = a U^b, a", round(fit["a"], 2))
        note("b", side + ": dp = a U^b, b", round(fit["b"], 3))
        note("b", side + ": R2 (global fit)", round(fit["r2"], 4))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("mean pore velocity U [m s⁻¹]")
    ax.set_ylabel("total pressure drop Δp [Pa]")
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], color="0.45", lw=1.1, ls=":"))
    labels.append("Darcy reference, n = 1")
    ax.legend(handles, labels, loc="upper left", frameon=False)
    ax.text(
        0.985,
        0.05,
        "n ≈ 1.7–1.9 ≫ 1 → inertia-dominated regime;\n"
        "Darcy's law is insufficient, a Forchheimer term is required.",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.2,
        color=D.C_MUTED,
    )
    style(ax, "(b)  Pressure-drop law Δp(U) — inertial despite laminar flow")
    return pd.DataFrame(fits)


# --------------------------------------------------------------------------- #
# (c) Decoupling of the two sides + numerical replicate
# --------------------------------------------------------------------------- #
def panel_c(ax: plt.Axes, summary: pd.DataFrame) -> None:
    s = summary.set_index(["run", "side"])

    # Cluster 1: air-side h at fixed air flow, three cold-side flow rates.
    # Cluster 2: cold-side h at fixed cold flow, three air-side flow rates.
    def cluster(runs, side: str, other: str) -> pd.DataFrame:
        rows = [
            {
                "run": run,
                "h": s.loc[(run, side), "h_mean_area_weighted_W_m2K"],
                "u_other": s.loc[(run, other), "mean_velocity_m_s"],
            }
            for run in runs
        ]
        df = pd.DataFrame(rows).sort_values("u_other").reset_index(drop=True)
        df["dev_pct"] = 100.0 * (df["h"] / df["h"].mean() - 1.0)
        df["spread_pct"] = 100.0 * (df["h"].max() / df["h"].min() - 1.0)
        return df

    c_air = cluster(["R001", "R004", "R005"], "Air", "Water")
    c_water = cluster(["R003", "R001", "R002"], "Water", "Air")

    positions = [0.0, 0.9, 1.8, 3.4, 4.3, 5.2]
    values = list(c_air["dev_pct"]) + list(c_water["dev_pct"])
    colors = [D.C_AIR] * 3 + [D.C_WATER] * 3
    ax.bar(positions, values, width=0.66, color=colors, edgecolor="white", linewidth=1.4)
    ax.axhline(0.0, color="#4d4d4d", lw=0.9)

    labels = ["%.3f" % u for u in c_air["u_other"]] + [
        "%.2f" % u for u in c_water["u_other"]
    ]
    for pos, val, h in zip(positions, values, list(c_air["h"]) + list(c_water["h"])):
        off = 0.30 if val >= 0 else -0.30
        ax.text(
            pos,
            val + off,
            ("%.1f" % h) if h < 200 else ("%.0f" % h),
            ha="center",
            va="bottom" if val >= 0 else "top",
            fontsize=8.0,
            color=D.C_MUTED,
        )
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_xlabel(
        "cold-side U [m s⁻¹]                        "
        "air-side U [m s⁻¹]"
    )
    ax.set_ylabel("deviation from cluster mean [%]")
    ax.set_ylim(-6.2, 7.2)

    ax.text(
        0.9,
        6.6,
        "air-side h\ncold flow changed 3.2×\n→ spread %.1f %%"
        % c_air["spread_pct"].iloc[0],
        ha="center",
        va="top",
        fontsize=8.3,
        color=D.C_AIR,
    )
    ax.text(
        4.3,
        6.6,
        "cold-side h\nair flow changed 4.7×\n→ spread %.1f %%"
        % c_water["spread_pct"].iloc[0],
        ha="center",
        va="top",
        fontsize=8.3,
        color=D.C_WATER,
    )

    h_r000 = s.loc[("R000", "Water"), "h_mean_area_weighted_W_m2K"]
    h_r005 = s.loc[("R005", "Water"), "h_mean_area_weighted_W_m2K"]
    repl = 100.0 * abs(h_r005 / h_r000 - 1.0)
    ax.text(
        0.5,
        0.045,
        "Numerical replicate — R000 vs R005, identical cold-side flow:\n"
        "h = %.1f vs %.1f W m⁻² K⁻¹  →  %.2f %% reproducibility"
        % (h_r000, h_r005, repl),
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=8.3,
        color=D.C_INK,
        bbox=dict(boxstyle="round,pad=0.4", fc="#f2f2f2", ec="#bdbdbd", lw=0.8),
    )

    note("c", "air-side h spread over 3 cold-flow levels",
         round(float(c_air["spread_pct"].iloc[0]), 2), "%")
    note("c", "cold-side h spread over 3 air-flow levels",
         round(float(c_water["spread_pct"].iloc[0]), 2), "%")
    note("c", "replicate R000 vs R005, cold-side h", round(repl, 3), "%")
    style(ax, "(c)  The two sides are decoupled — and the solver repeats itself")


# --------------------------------------------------------------------------- #
# (d) Band-resolved profiles, normalised
# --------------------------------------------------------------------------- #
def panel_d(ax: plt.Axes, bands: pd.DataFrame, tag: str = "d") -> None:
    clean = bands[~bands["is_outlier"]]
    grid = np.linspace(0.0, 1.0, 60)
    stats = []
    means: dict[str, np.ndarray] = {}

    for side in ("Air", "Water"):
        col = D.SIDE_COLOR[side]
        stack = []
        for run in D.RUNS:
            g = clean[(clean["side"] == side) & (clean["run"] == run)].sort_values(
                "x_over_L"
            )
            x = g["x_over_L"].to_numpy()
            y = g["h_W_m2K"].to_numpy() / g["h_W_m2K"].mean()
            ax.plot(x, y, color=col, lw=0.9, alpha=0.38, zorder=2)
            stack.append(np.interp(grid, x, y))
            stats.append(
                {
                    "side": side,
                    "run": run,
                    "decline_pct": 100.0 * (1.0 - y[x >= 0.8].mean() / y[x <= 0.2].mean()),
                    "band_ratio": float(y.max() / y.min()),
                }
            )
        means[side] = np.mean(np.vstack(stack), axis=0)
        ax.plot(grid, means[side], color=col, lw=2.6, zorder=4)

    ax.axhline(1.0, color="#4d4d4d", lw=0.9, zorder=1)
    ax.axvspan(0.8, 1.0, color="#dcdcdc", alpha=0.55, zorder=0)
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax)
    ax.text(
        0.9,
        ymin + 0.24 * (ymax - ymin),
        "last 20 %\nof the core",
        ha="center",
        va="bottom",
        fontsize=8.2,
        color=D.C_MUTED,
        zorder=6,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8),
    )

    st = pd.DataFrame(stats)
    lines = []
    for side in ("Air", "Water"):
        g = st[st["side"] == side]
        lines.append(
            "%s: −%.0f … −%.0f %% inlet→outlet, "
            "band-to-band spread %.1f–%.1f×"
            % (
                side,
                g["decline_pct"].min(),
                g["decline_pct"].max(),
                g["band_ratio"].min(),
                g["band_ratio"].max(),
            )
        )
        note("d", side + ": h decline, first 20% -> last 20%",
             "%.1f-%.1f" % (g["decline_pct"].min(), g["decline_pct"].max()), "%")
        note("d", side + ": band-to-band h ratio",
             "%.1f-%.1f" % (g["band_ratio"].min(), g["band_ratio"].max()), "x")

    ax.text(
        0.02,
        0.04,
        "\n".join(lines),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.2,
        color=D.C_INK,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#bdbdbd", lw=0.8, alpha=0.92),
    )

    dropped = bands[bands["is_outlier"]]
    if len(dropped):
        ax.scatter(
            dropped["x_over_L"],
            np.full(len(dropped), ymin + 0.02 * (ymax - ymin)),
            marker="x",
            s=44,
            color="#8a8a8a",
            zorder=5,
        )
        tags = ", ".join(
            "%s / %s band %d" % (r.run, r.side, int(r.band_id))
            for r in dropped.itertuples()
        )
        ax.text(
            0.985,
            0.965,
            "excluded: %d band (%s)\nh < 10 %% of the run median — LMTD collapse"
            % (len(dropped), tags),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7.8,
            color=D.C_MUTED,
        )

    ax.set_xlabel("normalised distance from inlet  x/L [–]")
    ax.set_ylabel("h / h̄  [–]")
    ax.set_xlim(-0.01, 1.01)
    # Direct labels instead of a legend box - only two series are bold.
    for side, xpos, dy, va in (("Air", 0.44, 0.10, "bottom"), ("Water", 0.16, -0.10, "top")):
        ax.text(
            xpos,
            float(np.interp(xpos, grid, means[side])) + dy,
            side + " — mean of 6 runs",
            color=D.SIDE_COLOR[side],
            fontsize=8.6,
            fontweight="bold",
            ha="center",
            va=va,
            zorder=6,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.85),
        )
    style(ax, "(%s)  The mean hides a 2–6× band-to-band variation" % tag)


# --------------------------------------------------------------------------- #
# (e) Cold-fluid choice: same duty, different pumping cost
# --------------------------------------------------------------------------- #
def panel_e(ax: plt.Axes, pairs: pd.DataFrame, tag: str = "e") -> None:
    for row in pairs.itertuples():
        col = D.FLUID_COLOR[row.cold_fluid]
        ax.errorbar(
            row.h_liquid,
            row.h_air,
            xerr=row.h_liquid_std,
            yerr=row.h_air_std,
            fmt=D.GEOM_MARKER[row.geometry],
            ms=9,
            color=col,
            ecolor=col,
            elinewidth=1.1,
            capsize=2.5,
            markeredgecolor="white",
            markeredgewidth=1.0,
            zorder=4,
        )

    # Duty and pumping cost as a compact two-row table - one label per point
    # would collide in the crowded 1000 W/m2K cluster.
    tbl = pairs.set_index(["geometry", "cold_fluid"])
    rows = []
    for geom, name in (("ggraded", "GRAD "), ("guni", "UNI10")):
        cells = []
        for f in ("Water", "HFE", "Oil"):
            r = tbl.loc[(geom, f)]
            cells.append("%-6s %.2f / %.2f" % (f.lower(), r["Q_liquid"], r["P_pump_liquid"] * 1e3))
        rows.append("%s   %s" % (name, "   ".join(cells)))
    ax.text(
        0.015,
        0.975,
        "duty and pumping cost,  Q [W] / P$_{pump}$ [mW]\n"
        "P$_{pump}$ = Δp·$\\dot{V}$   ($\\dot{V}$ = $\\dot{m}$/ρ, ideal pump η = 1)\n"
        + "\n".join(rows),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.9,
        family="DejaVu Sans Mono",
        color=D.C_INK,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#bdbdbd", lw=0.8, alpha=0.95),
        zorder=6,
    )

    ax.set_xscale("log")
    ax.set_xlabel("cold-side heat transfer coefficient h [W m⁻² K⁻¹]")
    ax.set_ylabel("air-side h [W m⁻² K⁻¹]")
    ax.set_xlim(700.0, 6000.0)
    ax.set_ylim(57.6, 64.2)
    ax.set_yticks([58, 59, 60, 61, 62, 63])

    h_ratio = pairs["h_liquid"].max() / pairs["h_liquid"].min()
    q_lo, q_hi = pairs["Q_liquid"].min(), pairs["Q_liquid"].max()
    guni = pairs[pairs["geometry"] == "guni"].set_index("cold_fluid")
    pump_ratio = guni.loc["Oil", "P_pump_liquid"] / guni.loc["Water", "P_pump_liquid"]

    ax.text(
        0.5,
        -0.30,
        "cold-side h varies %.1f×  →  duty stays at Q = %.2f–%.2f W;\n"
        "on UNI10 oil costs %.1f× the pumping power of water for the same duty"
        % (h_ratio, q_lo, q_hi, pump_ratio),
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8.4,
        color=D.C_INK,
        bbox=dict(boxstyle="round,pad=0.4", fc="#f2f2f2", ec="#bdbdbd", lw=0.8),
    )

    handles = [
        Line2D(
            [],
            [],
            marker=D.GEOM_MARKER[g],
            ls="none",
            color="#4d4d4d",
            markeredgecolor="white",
            markersize=8,
            label=D.GEOM_LABEL[g],
        )
        for g in ("ggraded", "guni")
    ] + [Patch(facecolor=D.FLUID_COLOR[f], label=f) for f in ("Water", "HFE", "Oil")]
    ax.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.0, 0.0),
        frameon=False,
        ncol=2,
        columnspacing=1.0,
    )

    note("e", "cold-side h range across fluids", round(float(h_ratio), 2), "x")
    note("e", "Q range across all six cases", "%.2f-%.2f" % (q_lo, q_hi), "W")
    note("e", "UNI10 oil/water pumping-power ratio", round(float(pump_ratio), 2), "x")
    style(ax, "(%s)  Cold-fluid choice sets the pumping cost, not the duty" % tag)


def panel_f(
    ax: plt.Axes,
    pec: pd.DataFrame,
    transient: pd.DataFrame,
    energy: pd.DataFrame,
    tag: str = "f",
) -> None:
    order = ["Water", "HFE", "Oil"]
    p = pec.set_index("cold_fluid").loc[order]
    x = np.arange(len(order), dtype=float)
    w = 0.34

    ax.bar(
        x - w / 2,
        p["PEC_h_dp13"],
        w,
        color=[D.FLUID_COLOR[f] for f in order],
        edgecolor="white",
        linewidth=1.4,
        label="PEC from (h, Δp)",
    )
    ax.bar(
        x + w / 2,
        p["PEC_Nu_f13"],
        w,
        color=[D.FLUID_COLOR[f] for f in order],
        edgecolor="white",
        linewidth=1.4,
        hatch="///",
        alpha=0.85,
        label="PEC from (Nu, f)",
    )
    ax.axhline(1.0, color="#4d4d4d", lw=1.2)
    ax.text(2.55, 1.01, "parity", fontsize=8.0, color=D.C_MUTED, va="bottom", ha="right")

    for xi, (v1, v2) in enumerate(zip(p["PEC_h_dp13"], p["PEC_Nu_f13"])):
        ax.text(xi - w / 2, v1 + 0.015, "%.2f" % v1, ha="center", fontsize=8.2)
        ax.text(xi + w / 2, v2 + 0.015, "%.2f" % v2, ha="center", fontsize=8.2)

    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_xlabel("cold fluid")
    ax.set_ylabel("PEC, cold side, GRAD / UNI10 [–]")
    ax.set_ylim(0.0, 1.62)
    ax.legend(loc="upper left", frameon=False, ncol=2)

    bal = energy.groupby(["geometry", "cold_fluid"])["balance_error_pct"].mean()
    bal_max = energy["balance_error_pct"].max()
    air = transient[transient["side"] == "Air"]
    liq = transient[transient["side"] != "Air"]
    air_f = 100.0 * (air["h_area_W_m2K_std"] / air["h_area_W_m2K_mean"])
    liq_f = 100.0 * (liq["h_area_W_m2K_std"] / liq["h_area_W_m2K_mean"])

    ax.text(
        0.5,
        -0.30,
        "PEC = (X$_{G}$/X$_{U}$)/(Y$_{G}$/Y$_{U}$)$^{1/3}$,  X,Y = (h, Δp) or (Nu, f),  "
        "G/U = GRAD/UNI10;  the 1/3 exponent ⇒ equal pumping power\n"
        "Validation (transient window 18–19 s, 50 time steps)\n"
        "energy balance |Q$_{air}$−Q$_{cold}$|: %.2f–%.2f %% mean, ≤ %.1f %% worst step\n"
        "temporal scatter of h: air %.1f–%.1f %%, cold liquid %.2f–%.2f %%"
        % (bal.min(), bal.max(), bal_max, air_f.min(), air_f.max(), liq_f.min(), liq_f.max()),
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8.3,
        color=D.C_INK,
        bbox=dict(boxstyle="round,pad=0.4", fc="#f2f2f2", ec="#bdbdbd", lw=0.8),
    )

    note("f", "energy balance, mean per case", "%.2f-%.2f" % (bal.min(), bal.max()), "%")
    note("f", "energy balance, worst time step", round(float(bal_max), 2), "%")
    note("f", "air-side temporal scatter of h", "%.1f-%.1f" % (air_f.min(), air_f.max()), "%")
    note("f", "cold-side temporal scatter of h", "%.2f-%.2f" % (liq_f.min(), liq_f.max()), "%")
    style(ax, "(%s)  Verdict at equal pumping power — and the validation behind it" % tag)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
TITLE = (
    "Band-resolved analysis of a TPMS gyroid heat exchanger: "
    "from six flow regimes to a design correlation set"
)
FOOTER_23 = (
    "(a)–(d) Run 001 — steady, GRAD / M009, six air–water regimes "
    "R000–R005, band-resolved.   (e)–(f) Run 002 — transient, window "
    "18–19 s, GRAD vs UNI10, cold fluid water / HFE / oil.\n"
    "R² in (a) refers to the global (run-averaged) fit only. One band excluded from "
    "the band-resolved statistics — see outliers_excluded.csv."
)
FOOTER_22 = (
    "(a), (b) Run 001 — steady, GRAD / M009, six air–water regimes "
    "R000–R005, band-resolved.   (c), (d) Run 002 — transient, window "
    "18–19 s, GRAD vs UNI10, cold fluid water / HFE / oil.\n"
    "R² in (a) refers to the global (run-averaged) fit only. One band excluded from "
    "the band-resolved statistics — see outliers_excluded.csv."
)


def save(fig: plt.Figure, stem: str) -> None:
    for ext in ("png", "pdf"):
        path = OUT_DIR / ("%s.%s" % (stem, ext))
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
        print("  wrote " + path.name)
    plt.close(fig)


def build_2x3(bands, summary, pairs, pec, transient, energy):
    fig, axes = plt.subplots(2, 3, figsize=(17.4, 10.6))
    fig.suptitle(TITLE, fontsize=13.5, fontweight="bold", y=0.985)
    nu_fits = panel_a(axes[0, 0], bands, summary)
    dp_fits = panel_b(axes[0, 1], summary)
    panel_c(axes[0, 2], summary)
    panel_d(axes[1, 0], bands)
    panel_e(axes[1, 1], pairs)
    panel_f(axes[1, 2], pec, transient, energy)
    fig.tight_layout(rect=(0.0, 0.055, 1.0, 0.955))
    fig.subplots_adjust(hspace=0.46, wspace=0.26)
    fig.text(0.5, 0.008, FOOTER_23, ha="center", va="bottom", fontsize=8.0, color=D.C_MUTED)
    save(fig, "panel_partners_2x3")
    return nu_fits, dp_fits


def build_2x2(bands, summary, pairs, pec, transient, energy):
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 10.6))
    fig.suptitle(TITLE, fontsize=13.5, fontweight="bold", y=0.985)
    panel_a(axes[0, 0], bands, summary)
    panel_d(axes[0, 1], bands, tag="b")
    panel_e(axes[1, 0], pairs, tag="c")
    panel_f(axes[1, 1], pec, transient, energy, tag="d")
    fig.tight_layout(rect=(0.0, 0.055, 1.0, 0.955))
    fig.subplots_adjust(hspace=0.46, wspace=0.24)
    fig.text(0.5, 0.008, FOOTER_22, ha="center", va="bottom", fontsize=8.0, color=D.C_MUTED)
    save(fig, "panel_partners_2x2")


def main() -> None:
    bands = D.load_bands()
    summary = D.load_summary()
    pairs = D.transient_pairs()
    pec = D.load_pec()
    transient = D.load_transient()
    energy = D.load_energy_balance()

    print("Building 2x3 panel ...")
    nu_fits, dp_fits = build_2x3(bands, summary, pairs, pec, transient, energy)
    key = pd.DataFrame(KEY_NUMBERS)

    print("Building 2x2 panel ...")
    KEY_NUMBERS.clear()
    build_2x2(bands, summary, pairs, pec, transient, energy)

    corr = pd.concat([nu_fits, dp_fits], ignore_index=True)[
        ["side", "correlation", "a", "b", "r2", "n", "x_min", "x_max"]
    ]
    corr.to_csv(OUT_DIR / "correlations_global.csv", index=False)
    print("  wrote correlations_global.csv")

    dropped = bands[bands["is_outlier"]][
        [
            "run",
            "side",
            "band_id",
            "distance_from_inlet_mm",
            "x_over_L",
            "h_W_m2K",
            "h_over_median",
            "Nu",
            "Re",
            "outlier_reason",
        ]
    ]
    dropped.to_csv(OUT_DIR / "outliers_excluded.csv", index=False)
    print("  wrote outliers_excluded.csv (%d band(s))" % len(dropped))

    key.to_csv(OUT_DIR / "panel_key_numbers.csv", index=False)
    print("  wrote panel_key_numbers.csv")


if __name__ == "__main__":
    main()
