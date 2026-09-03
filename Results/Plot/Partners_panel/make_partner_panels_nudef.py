"""Variant of the 2x2 partner panel with an explicit Nusselt-number definition.

Same figure as ``panel_partners_2x2`` but panel (a) is extended to answer one
recurring question: why do the air and the water streams sit on almost the
same Nu level while their heat-transfer coefficients differ by more than an
order of magnitude?

Panel (a) now carries
  * the definition            Nu = h * D_h / k_fluid
  * an inset that decomposes the Water/Air ratio into its two factors,
    h (Water is ~21x larger) and D_h/k (Water is ~25x smaller), whose
    product is the Nu ratio ~ 0.8, i.e. the two effects cancel.

Nothing is recomputed: D_h/k is recovered per band as Nu / h from the same
Run 001 band table used everywhere else.

Usage:  python make_partner_panels_nudef.py
Output: panel_partners_2x2_nudef.{png,pdf}
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

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _data as D
import make_partner_panels as M

OUT_DIR = Path(__file__).resolve().parent
DPI = M.DPI


# --------------------------------------------------------------------------- #
# (a) Global Nu-Re correlation + Nu definition + factor decomposition
# --------------------------------------------------------------------------- #
def panel_a_nudef(ax: plt.Axes, bands: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    clean = bands[~bands["is_outlier"]].copy()
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

        xs = np.logspace(np.log10(fit["x_min"]) - 0.05, np.log10(fit["x_max"]) + 0.05, 60)
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

    air, water = fits[0], fits[1]

    # --- factor decomposition: band medians ------------------------------- #
    clean["Dh_over_k"] = clean["Nu"] / clean["h_W_m2K"]
    med = clean[clean["h_W_m2K"] > 0].groupby("side").agg(
        h=("h_W_m2K", "median"),
        dhk=("Dh_over_k", "median"),
        Nu=("Nu", "median"),
    )
    r_h = med.loc["Water", "h"] / med.loc["Air", "h"]
    r_dhk = med.loc["Water", "dhk"] / med.loc["Air", "dhk"]
    r_nu = med.loc["Water", "Nu"] / med.loc["Air", "Nu"]

    # D_h is purely geometric (identical across the six runs); band ranges
    # recomputed once from the SRP band tables - see make_partner_panels_nudef docstring.
    DH_RANGE = {"Air": (4.7, 14.8), "Water": (7.1, 9.1)}  # mm, band min-max

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Re [–]")
    ax.set_ylabel("Nu [–]")
    # Open up head- and foot-room so the annotation boxes sit on empty canvas
    # instead of on top of the data or on each other.  The data itself spans
    # Nu ~ 7-80, i.e. the middle band of the axes.
    ax.set_ylim(1.9, 260.0)

    # the definition + why the two streams collapse, upper-right -------- #
    ax.text(
        0.998,
        0.995,
        "Nu $\\equiv$ h · D$_h$ / k$_{fluid}$\n"
        "water / air:   h ×%.0f  ,  D$_h$/k ÷%.0f\n"
        "D$_h$ ~ equal (air %.1f–%.1f, water %.1f–%.1f mm)\n"
        "→ the ÷%.0f is k   ⇒   Nu ratio %.2f"
        % (
            r_h, 1.0 / r_dhk,
            DH_RANGE["Air"][0], DH_RANGE["Air"][1],
            DH_RANGE["Water"][0], DH_RANGE["Water"][1],
            1.0 / r_dhk, r_nu,
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.7,
        family="DejaVu Sans Mono",
        color=D.C_INK,
        bbox=dict(boxstyle="round,pad=0.35", fc="#f2f2f2", ec="#bdbdbd", lw=0.8),
    )

    # fitted correlations, lower-right ---------------------------------- #
    ax.text(
        0.995,
        0.035,
        "Air:    Nu = %.3f·Re$^{%.3f}$\n"
        "        R² = %.4f  (Re %.0f–%.0f)\n"
        "Water: Nu = %.3f·Re$^{%.3f}$\n"
        "        R² = %.4f  (Re %.0f–%.0f)"
        % (
            air["a"], air["b"], air["r2"], air["x_min"], air["x_max"],
            water["a"], water["b"], water["r2"], water["x_min"], water["x_max"],
        ),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.7,
        color=D.C_INK,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#bdbdbd", lw=0.8, alpha=0.95),
    )

    # decomposition inset --------------------------------------------- #
    ins = ax.inset_axes([0.11, 0.06, 0.36, 0.19])
    ins.set_facecolor("white")
    ins.patch.set_alpha(1.0)
    ins.set_zorder(5)
    names = ["h", "D$_h$/k", "Nu"]
    ratios = [r_h, r_dhk, r_h * r_dhk]
    ypos = np.arange(len(names))[::-1]
    bar_colors = ["#b23b3b" if v > 1 else "#3b6fb2" for v in ratios]
    ins.barh(ypos, ratios, height=0.62, color=bar_colors, edgecolor="white", linewidth=1.0)
    ins.axvline(1.0, color="#4d4d4d", lw=1.0)
    ins.set_xscale("log")
    ins.set_yticks(ypos)
    ins.set_yticklabels(names, fontsize=7.6)
    ins.set_xlim(0.02, 900.0)
    ins.set_xticks([0.1, 1.0, 10.0])
    ins.set_xticklabels(["0.1", "1", "10"])
    ins.tick_params(labelsize=7.0, length=2, pad=1)
    ins.set_title("Water / Air ratio  (band medians)", fontsize=7.6, pad=3)
    for y, v in zip(ypos, ratios):
        ins.text(
            800.0,
            y,
            ("×%.1f" % v)
            if v >= 1
            else ("÷%.0f" % (1.0 / v) if 1.0 / v >= 2 else "×%.2f" % v),
            va="center",
            ha="right",
            fontsize=7.2,
            color=D.C_INK,
        )
    for sp in ("top", "right"):
        ins.spines[sp].set_visible(False)

    handles, labels = ax.get_legend_handles_labels()
    handles.append(
        Line2D([], [], marker="o", ls="none", color="0.55", alpha=0.5, markersize=4)
    )
    labels.append("band-resolved points (background)")
    ax.legend(handles, labels, loc="upper left", frameon=False)
    M.style(ax, "(a)  Design correlation Nu(Re)")

    M.note("a", "Water/Air h ratio (band median)", round(float(r_h), 2), "x")
    M.note("a", "Water/Air Dh/k ratio (band median)", round(float(r_dhk), 4), "x")
    M.note("a", "Water/Air Nu ratio (band median)", round(float(r_nu), 3), "x")
    for f in fits:
        M.note("a", f["side"] + ": Nu = a Re^b, a", round(f["a"], 3))
        M.note("a", f["side"] + ": Nu = a Re^b, b", round(f["b"], 3))
        M.note("a", f["side"] + ": R2 (global fit)", round(f["r2"], 4))
    return pd.DataFrame(fits)


def build(bands, summary, pairs, pec, transient, energy) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 10.6))
    fig.suptitle(M.TITLE, fontsize=13.5, fontweight="bold", y=0.985)
    panel_a_nudef(axes[0, 0], bands, summary)
    M.panel_d(axes[0, 1], bands, tag="b")
    M.panel_e(axes[1, 0], pairs, tag="c")
    M.panel_f(axes[1, 1], pec, transient, energy, tag="d")
    fig.tight_layout(rect=(0.0, 0.055, 1.0, 0.955))
    fig.subplots_adjust(hspace=0.46, wspace=0.24)
    fig.text(0.5, 0.008, M.FOOTER_22, ha="center", va="bottom", fontsize=8.0, color=D.C_MUTED)
    M.save(fig, "panel_partners_2x2_nudef")


def main() -> None:
    bands = D.load_bands()
    summary = D.load_summary()
    pairs = D.transient_pairs()
    pec = D.load_pec()
    transient = D.load_transient()
    energy = D.load_energy_balance()

    print("Building 2x2 panel (Nu definition variant) ...")
    M.KEY_NUMBERS.clear()
    build(bands, summary, pairs, pec, transient, energy)
    pd.DataFrame(M.KEY_NUMBERS).to_csv(OUT_DIR / "panel_key_numbers_nudef.csv", index=False)
    print("  wrote panel_key_numbers_nudef.csv")


if __name__ == "__main__":
    main()
